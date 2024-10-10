import Mathlib

namespace polynomial_simplification_l3366_336665

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 3) - (r^3 + 3 * r^2 + 9 * r - 2) = r^3 - 2 * r^2 - 4 * r - 1 := by
  sorry

end polynomial_simplification_l3366_336665


namespace greatest_three_digit_divisible_by_3_6_5_l3366_336662

theorem greatest_three_digit_divisible_by_3_6_5 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 ∣ n ∧ 6 ∣ n ∧ 5 ∣ n → n ≤ 990 := by
  sorry

end greatest_three_digit_divisible_by_3_6_5_l3366_336662


namespace rectangle_perimeter_l3366_336667

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = (4/3) * original_area →
  2 * (new_area / width + width) = 60 :=
by sorry

end rectangle_perimeter_l3366_336667


namespace total_turtles_count_l3366_336699

/-- Represents the total number of turtles in the lake -/
def total_turtles : ℕ := sorry

/-- Represents the number of striped male adult common turtles -/
def striped_male_adult_common : ℕ := 70

/-- Percentage of common turtles in the lake -/
def common_percentage : ℚ := 1/2

/-- Percentage of female common turtles -/
def common_female_percentage : ℚ := 3/5

/-- Percentage of striped male common turtles among male common turtles -/
def striped_male_common_percentage : ℚ := 1/4

/-- Percentage of adult striped male common turtles among striped male common turtles -/
def adult_striped_male_common_percentage : ℚ := 4/5

theorem total_turtles_count : total_turtles = 1760 := by sorry

end total_turtles_count_l3366_336699


namespace reggie_shopping_spree_l3366_336628

def initial_amount : ℕ := 150
def num_books : ℕ := 5
def book_price : ℕ := 12
def game_price : ℕ := 45
def bottle_price : ℕ := 13
def snack_price : ℕ := 7

theorem reggie_shopping_spree :
  initial_amount - (num_books * book_price + game_price + bottle_price + snack_price) = 25 := by
  sorry

end reggie_shopping_spree_l3366_336628


namespace harry_seed_purchase_cost_l3366_336633

/-- The cost of a garden seed purchase --/
def garden_seed_cost (pumpkin_price tomato_price chili_price : ℚ) 
  (pumpkin_qty tomato_qty chili_qty : ℕ) : ℚ :=
  pumpkin_price * pumpkin_qty + tomato_price * tomato_qty + chili_price * chili_qty

/-- Theorem stating the total cost of Harry's seed purchase --/
theorem harry_seed_purchase_cost : 
  garden_seed_cost 2.5 1.5 0.9 3 4 5 = 18 := by
  sorry

end harry_seed_purchase_cost_l3366_336633


namespace trip_cost_proof_l3366_336679

/-- Calculates the total cost of a trip for two people with a discount -/
def total_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Proves that the total cost of the trip for two people is $266 -/
theorem trip_cost_proof (original_price discount : ℕ) 
  (h1 : original_price = 147) 
  (h2 : discount = 14) : 
  total_cost original_price discount = 266 := by
  sorry

#eval total_cost 147 14

end trip_cost_proof_l3366_336679


namespace sqrt_sum_fractions_l3366_336671

theorem sqrt_sum_fractions : 
  Real.sqrt ((1 : ℝ) / 25 + (1 : ℝ) / 36) = Real.sqrt 61 / 30 := by
  sorry

end sqrt_sum_fractions_l3366_336671


namespace arithmetic_sequence_properties_l3366_336636

/-- Arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ S 4 = 20 ∧
  ∀ n : ℕ, S n = n * (a 1) + (n * (n - 1)) / 2 * (a 2 - a 1)

/-- Theorem stating the common difference and S_6 for the given arithmetic sequence -/
theorem arithmetic_sequence_properties (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h : arithmetic_sequence a S) : (a 2 - a 1 = 3) ∧ (S 6 = 48) := by
  sorry

end arithmetic_sequence_properties_l3366_336636


namespace book_selection_theorem_l3366_336693

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem book_selection_theorem :
  let biology_ways := choose 10 3
  let chemistry_ways := choose 8 2
  let physics_ways := choose 5 1
  biology_ways * chemistry_ways * physics_ways = 16800 := by
  sorry

end book_selection_theorem_l3366_336693


namespace third_defendant_guilty_l3366_336648

-- Define the set of defendants
inductive Defendant : Type
  | A
  | B
  | C

-- Define the accusation function
def accuses : Defendant → Defendant → Prop := sorry

-- Define the truth-telling property
def tells_truth (d : Defendant) : Prop := sorry

-- Define the guilt property
def is_guilty (d : Defendant) : Prop := sorry

-- Define the condition that each defendant accuses one of the other two
axiom each_accuses_one : ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ ∨ accuses d₁ d₃) ∧ (accuses d₂ d₁ ∨ accuses d₂ d₃) ∧ (accuses d₃ d₁ ∨ accuses d₃ d₂)

-- Define the condition that the first defendant (A) is the only one telling the truth
axiom A_tells_truth : tells_truth Defendant.A ∧ ¬tells_truth Defendant.B ∧ ¬tells_truth Defendant.C

-- Define the condition that if accusations were changed, B would be the only one telling the truth
axiom if_changed_B_tells_truth : 
  ∀ d₁ d₂ d₃ : Defendant, d₁ ≠ d₂ → d₁ ≠ d₃ → d₂ ≠ d₃ → 
  (accuses d₁ d₂ → accuses d₁ d₃) → (accuses d₂ d₁ → accuses d₂ d₃) → (accuses d₃ d₁ → accuses d₃ d₂) →
  tells_truth Defendant.B ∧ ¬tells_truth Defendant.A ∧ ¬tells_truth Defendant.C

-- Theorem: Given the conditions, the third defendant (C) is guilty
theorem third_defendant_guilty : is_guilty Defendant.C := by
  sorry

end third_defendant_guilty_l3366_336648


namespace max_value_problem_l3366_336690

theorem max_value_problem (m n k : ℕ) (a b c : ℕ → ℕ) :
  (∀ i ∈ Finset.range m, a i % 3 = 1) →
  (∀ i ∈ Finset.range n, b i % 3 = 2) →
  (∀ i ∈ Finset.range k, c i % 3 = 0) →
  (∀ i j, i ≠ j → (a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j ∧ 
                   a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)) →
  (Finset.sum (Finset.range m) a + Finset.sum (Finset.range n) b + 
   Finset.sum (Finset.range k) c = 2007) →
  4 * m + 3 * n + 5 * k ≤ 256 := by
sorry

end max_value_problem_l3366_336690


namespace domain_fxPlus2_l3366_336631

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x-3)
def domain_f2xMinus3 : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem domain_fxPlus2 (h : ∀ x ∈ domain_f2xMinus3, f (2*x - 3) = f (2*x - 3)) :
  {x : ℝ | f (x + 2) = f (x + 2)} = {x : ℝ | -9 ≤ x ∧ x ≤ 1} :=
sorry

end domain_fxPlus2_l3366_336631


namespace max_product_constrained_sum_l3366_336600

theorem max_product_constrained_sum (a b : ℝ) : 
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 8 * b₀ = 80 ∧ a₀ * b₀ = 40 := by
  sorry

end max_product_constrained_sum_l3366_336600


namespace bens_initial_money_l3366_336680

theorem bens_initial_money (initial_amount : ℕ) : 
  (((initial_amount - 600) + 800) - 1200 = 1000) → 
  initial_amount = 2000 := by
  sorry

end bens_initial_money_l3366_336680


namespace sum_of_digits_consecutive_numbers_l3366_336655

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sum_of_digits_consecutive_numbers 
  (N : ℕ) 
  (h1 : sumOfDigits N + sumOfDigits (N + 1) = 200)
  (h2 : sumOfDigits (N + 2) + sumOfDigits (N + 3) = 105) :
  sumOfDigits (N + 1) + sumOfDigits (N + 2) = 202 := by sorry

end sum_of_digits_consecutive_numbers_l3366_336655


namespace sum_of_G_from_2_to_100_l3366_336656

-- Define G(n) as the number of solutions to sin x = sin (n^2 x) on [0, 2π]
def G (n : ℕ) : ℕ := 
  if n > 1 then 2 * n^2 + 1 else 0

-- Theorem statement
theorem sum_of_G_from_2_to_100 : 
  (Finset.range 99).sum (fun i => G (i + 2)) = 676797 := by
  sorry

end sum_of_G_from_2_to_100_l3366_336656


namespace chocolate_distribution_l3366_336647

theorem chocolate_distribution (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 6) (h2 : total_pieces = 3000) :
  total_pieces / num_boxes = 500 := by
  sorry

end chocolate_distribution_l3366_336647


namespace expression_value_l3366_336635

theorem expression_value (n m : ℤ) (h : m = 2 * n^2 + n + 1) :
  8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end expression_value_l3366_336635


namespace parallel_vectors_m_value_l3366_336663

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (m + 1, -3)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → m = -3 := by
sorry

end parallel_vectors_m_value_l3366_336663


namespace geometric_sequence_sum_l3366_336604

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 2/5
  let n : ℕ := 8
  geometric_sum a r n = 390369/1171875 := by
sorry

end geometric_sequence_sum_l3366_336604


namespace saltwater_volume_l3366_336629

/-- Proves that the initial volume of a saltwater solution is 200 gallons, given the conditions stated in the problem. -/
theorem saltwater_volume : ∃ (x : ℝ),
  -- Initial solution is 20% salt by volume
  let initial_salt := 0.2 * x
  -- Volume after evaporation (3/4 of initial volume)
  let volume_after_evap := 0.75 * x
  -- New volume after adding water and salt
  let new_volume := volume_after_evap + 10 + 20
  -- New amount of salt
  let new_salt := initial_salt + 20
  -- The resulting mixture is 33 1/3% salt by volume
  new_salt = (1/3) * new_volume ∧ x = 200 :=
sorry

end saltwater_volume_l3366_336629


namespace max_value_sqrt_sum_l3366_336630

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 8 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = 3 * Real.sqrt 8 :=
by sorry

end max_value_sqrt_sum_l3366_336630


namespace number_of_gharials_l3366_336682

/-- Represents the number of flies eaten per day by one frog -/
def flies_per_frog : ℕ := 30

/-- Represents the number of frogs eaten per day by one fish -/
def frogs_per_fish : ℕ := 8

/-- Represents the number of fish eaten per day by one gharial -/
def fish_per_gharial : ℕ := 15

/-- Represents the total number of flies eaten per day in the swamp -/
def total_flies_eaten : ℕ := 32400

/-- Proves that the number of gharials in the swamp is 9 -/
theorem number_of_gharials : ℕ := by
  sorry

end number_of_gharials_l3366_336682


namespace equilateral_triangle_extension_equality_l3366_336614

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the equilateral property
def is_equilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define points D and E
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions for D and E
def D_on_AC_extension (A C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = A + t • (C - A)

def E_on_BC_extension (B C E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ s : ℝ, s > 1 ∧ E = B + s • (C - B)

-- Define the equality of BD and DE
def BD_equals_DE (B D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist B D = dist D E

-- State the theorem
theorem equilateral_triangle_extension_equality
  (h1 : is_equilateral A B C)
  (h2 : D_on_AC_extension A C D)
  (h3 : E_on_BC_extension B C E)
  (h4 : BD_equals_DE B D E) :
  dist A D = dist C E :=
sorry

end equilateral_triangle_extension_equality_l3366_336614


namespace flu_outbreak_theorem_l3366_336670

/-- Represents the state of a dwarf --/
inductive DwarfState
| Sick
| Healthy
| Immune

/-- Represents the population of dwarves --/
structure DwarfPopulation where
  sick : Set Nat
  healthy : Set Nat
  immune : Set Nat

/-- Represents the flu outbreak --/
structure FluOutbreak where
  initialVaccinated : Bool
  population : Nat → DwarfPopulation

/-- The flu lasts indefinitely if some dwarves are initially vaccinated --/
def fluLastsIndefinitely (outbreak : FluOutbreak) : Prop :=
  outbreak.initialVaccinated ∧
  ∀ n : Nat, ∃ i : Nat, i ∈ (outbreak.population n).sick

/-- The flu eventually ends if no dwarves are initially immune --/
def fluEventuallyEnds (outbreak : FluOutbreak) : Prop :=
  ¬outbreak.initialVaccinated ∧
  ∃ n : Nat, ∀ i : Nat, i ∉ (outbreak.population n).sick

theorem flu_outbreak_theorem (outbreak : FluOutbreak) :
  (outbreak.initialVaccinated → fluLastsIndefinitely outbreak) ∧
  (¬outbreak.initialVaccinated → fluEventuallyEnds outbreak) := by
  sorry


end flu_outbreak_theorem_l3366_336670


namespace parabola_point_distance_l3366_336658

/-- Theorem: For a parabola y² = 4x with focus F(1, 0), and a point P(x₀, y₀) on the parabola 
    such that |PF| = 3/2 * x₀, the value of x₀ is 2. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = 4*x₀ →                             -- P(x₀, y₀) is on the parabola
  (x₀ - 1)^2 + y₀^2 = (3/2 * x₀)^2 →        -- |PF| = 3/2 * x₀
  x₀ = 2 := by
sorry

end parabola_point_distance_l3366_336658


namespace mn_squared_equals_half_sum_l3366_336697

/-- Represents a quadrilateral ABCD with a segment MN parallel to CD -/
structure QuadrilateralWithSegment where
  /-- Length of segment from A parallel to CD intersecting BC -/
  a : ℝ
  /-- Length of segment from B parallel to CD intersecting AD -/
  b : ℝ
  /-- Length of CD -/
  c : ℝ
  /-- Length of MN -/
  mn : ℝ
  /-- MN is parallel to CD -/
  mn_parallel_cd : True
  /-- M lies on BC and N lies on AD -/
  m_on_bc_n_on_ad : True
  /-- MN divides the quadrilateral ABCD into two equal areas -/
  mn_divides_equally : True

/-- Theorem stating the relationship between MN, a, b, and c -/
theorem mn_squared_equals_half_sum (q : QuadrilateralWithSegment) :
  q.mn ^ 2 = (q.a * q.b + q.c ^ 2) / 2 := by sorry

end mn_squared_equals_half_sum_l3366_336697


namespace student_response_change_difference_l3366_336619

/-- Represents the percentages of student responses --/
structure ResponsePercentages :=
  (yes : ℝ)
  (no : ℝ)
  (undecided : ℝ)

/-- The problem statement --/
theorem student_response_change_difference 
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 40)
  (h_initial_undecided : initial.undecided = 20)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 30)
  (h_final_undecided : final.undecided = 10) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 40 :=
sorry

end student_response_change_difference_l3366_336619


namespace unique_satisfying_function_l3366_336610

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + f (x + y) - f x * f y = 0

/-- The theorem stating that there is exactly one function satisfying the equation -/
theorem unique_satisfying_function : ∃! f : ℝ → ℝ, SatisfyingFunction f := by sorry

end unique_satisfying_function_l3366_336610


namespace min_value_sum_fractions_l3366_336645

theorem min_value_sum_fractions (a b c k : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  ((a + b + k) / c + (a + c + k) / b + (b + c + k) / a = 9 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_fractions_l3366_336645


namespace solve_equation_l3366_336666

theorem solve_equation : ∃ x : ℝ, 25 - 5 = 3 + x - 4 → x = 21 := by
  sorry

end solve_equation_l3366_336666


namespace sum_bound_l3366_336626

theorem sum_bound (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : |a - b| + |b - c| + |c - a| = 1) : 
  a + b + c ≥ 1/2 := by
  sorry

end sum_bound_l3366_336626


namespace subset_implies_complement_subset_l3366_336620

theorem subset_implies_complement_subset (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) 
  (h_intersection : P ∩ Q = P) : 
  ∀ x, x ∉ Q → x ∉ P := by
  sorry

end subset_implies_complement_subset_l3366_336620


namespace polynomial_sum_l3366_336643

/-- Given polynomial f -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Given polynomial g -/
def g (a b c : ℤ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + a

/-- The main theorem -/
theorem polynomial_sum (a b c : ℤ) : c ≠ 0 →
  f a b c 1 = 0 →
  (∀ x : ℝ, g a b c x = 0 ↔ ∃ y : ℝ, f a b c y = 0 ∧ x = y^2) →
  a^2013 + b^2013 + c^2013 = -1 := by
  sorry


end polynomial_sum_l3366_336643


namespace sequence_increasing_and_divergent_l3366_336684

open Real MeasureTheory Interval Set

noncomputable section

variables (a b : ℝ) (f g : ℝ → ℝ)

def I (n : ℕ) := ∫ x in a..b, (f x)^(n+1) / (g x)^n

theorem sequence_increasing_and_divergent
  (hab : a < b)
  (hf : ContinuousOn f (Icc a b))
  (hg : ContinuousOn g (Icc a b))
  (hfg_pos : ∀ x ∈ Icc a b, 0 < f x ∧ 0 < g x)
  (hfg_int : ∫ x in a..b, f x = ∫ x in a..b, g x)
  (hfg_neq : f ≠ g) :
  (∀ n : ℕ, I a b f g n < I a b f g (n + 1)) ∧
  (∀ M : ℝ, ∃ N : ℕ, ∀ n ≥ N, M < I a b f g n) :=
sorry

end sequence_increasing_and_divergent_l3366_336684


namespace minimum_red_chips_l3366_336609

theorem minimum_red_chips (w b r : ℕ) 
  (blue_white : b ≥ (1/3 : ℚ) * w)
  (blue_red : b ≤ (1/4 : ℚ) * r)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 76 :=
sorry

end minimum_red_chips_l3366_336609


namespace shoe_difference_l3366_336637

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end shoe_difference_l3366_336637


namespace walmart_ground_beef_sales_l3366_336698

theorem walmart_ground_beef_sales (thursday_sales : ℕ) (friday_sales : ℕ) (saturday_sales : ℕ) 
  (h1 : thursday_sales = 210)
  (h2 : friday_sales = 2 * thursday_sales)
  (h3 : (thursday_sales + friday_sales + saturday_sales) / 3 = 260) :
  saturday_sales = 150 := by
sorry

end walmart_ground_beef_sales_l3366_336698


namespace line_slope_intercept_product_l3366_336689

/-- Given a line y = mx + b, prove that mb < -1 --/
theorem line_slope_intercept_product (m b : ℝ) : m * b < -1 := by
  sorry

end line_slope_intercept_product_l3366_336689


namespace not_proportional_l3366_336673

/-- A function f is directly proportional to x if there exists a constant k such that f x = k * x for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function f is inversely proportional to x if there exists a constant k such that f x = k / x for all non-zero x -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by the equation x^2 + y = 1 -/
def f (x : ℝ) : ℝ := 1 - x^2

theorem not_proportional : ¬(DirectlyProportional f) ∧ ¬(InverselyProportional f) := by
  sorry

end not_proportional_l3366_336673


namespace quadratic_roots_theorem_l3366_336687

theorem quadratic_roots_theorem (a b c : ℝ) :
  (∃ x y : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ∧ y^2 - (a+b)*y + (a*b-c^2) = 0) ∧
  (∃! x : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ↔ a = b ∧ c = 0) := by
  sorry

end quadratic_roots_theorem_l3366_336687


namespace bullet_problem_l3366_336668

theorem bullet_problem (n : ℕ) (h1 : n > 4) :
  (5 * (n - 4) = n) → n = 5 := by
  sorry

end bullet_problem_l3366_336668


namespace local_extrema_of_f_l3366_336678

open Real

/-- The function f(x) = x^3 - 3x^2 - 9x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem local_extrema_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 2 ∧
  IsLocalMax f x ∧
  f x = 5 ∧
  (∀ y ∈ Set.Ioo (-2 : ℝ) 2, ¬IsLocalMin f y) := by
  sorry

#check local_extrema_of_f

end local_extrema_of_f_l3366_336678


namespace imaginary_part_of_z_l3366_336618

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  z.im = -3/2 := by
  sorry

end imaginary_part_of_z_l3366_336618


namespace straw_length_theorem_l3366_336653

/-- The total length of overlapping straws -/
def total_length (straw_length : ℕ) (overlap : ℕ) (num_straws : ℕ) : ℕ :=
  straw_length + (straw_length - overlap) * (num_straws - 1)

/-- Theorem: The total length of 30 straws is 576 cm -/
theorem straw_length_theorem :
  total_length 25 6 30 = 576 := by
  sorry

end straw_length_theorem_l3366_336653


namespace intersection_line_equation_l3366_336644

/-- The equation of the line passing through the intersection points of two circles -/
theorem intersection_line_equation (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) :
  c1_center = (-8, -6) →
  c2_center = (4, 5) →
  r1 = 10 →
  r2 = Real.sqrt 41 →
  ∃ (x y : ℝ), ((x - c1_center.1)^2 + (y - c1_center.2)^2 = r1^2) ∧
                ((x - c2_center.1)^2 + (y - c2_center.2)^2 = r2^2) ∧
                (x + y = -59/11) :=
by sorry


end intersection_line_equation_l3366_336644


namespace inequality_proof_l3366_336676

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end inequality_proof_l3366_336676


namespace no_1999_primes_in_ap_l3366_336602

theorem no_1999_primes_in_ap (a d : ℕ) (h : a > 0 ∧ d > 0) :
  (∀ k : ℕ, k < 1999 → a + k * d < 12345 ∧ Nat.Prime (a + k * d)) →
  False :=
sorry

end no_1999_primes_in_ap_l3366_336602


namespace no_three_distinct_real_roots_l3366_336659

theorem no_three_distinct_real_roots (c : ℝ) : 
  ¬ ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧ 
    (x₁^3 + 4*x₁^2 + 6*x₁ + c = 0) ∧ 
    (x₂^3 + 4*x₂^2 + 6*x₂ + c = 0) ∧ 
    (x₃^3 + 4*x₃^2 + 6*x₃ + c = 0) :=
by sorry

end no_three_distinct_real_roots_l3366_336659


namespace problems_left_to_grade_l3366_336615

theorem problems_left_to_grade (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : 
  problems_per_worksheet = 4 →
  total_worksheets = 16 →
  graded_worksheets = 8 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 32 := by
  sorry

end problems_left_to_grade_l3366_336615


namespace sufficient_condition_implies_m_geq_6_l3366_336652

-- Define the conditions p and q as functions
def p (x : ℝ) : Prop := (x - 1) / x ≤ 0
def q (x m : ℝ) : Prop := 4^x + 2^x - m ≤ 0

-- State the theorem
theorem sufficient_condition_implies_m_geq_6 :
  (∀ x m : ℝ, p x → q x m) → ∀ m : ℝ, m ≥ 6 := by
  sorry

end sufficient_condition_implies_m_geq_6_l3366_336652


namespace fifth_group_number_l3366_336640

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_elements : ℕ
  sample_size : ℕ
  first_drawn : ℕ
  h_positive : 0 < total_elements
  h_sample_size : 0 < sample_size
  h_first_drawn : first_drawn ≤ total_elements
  h_divisible : total_elements % sample_size = 0

/-- The number drawn in a specific group -/
def number_in_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_drawn + (group - 1) * (s.total_elements / s.sample_size)

/-- Theorem stating the number drawn in the fifth group -/
theorem fifth_group_number (s : SystematicSampling)
  (h1 : s.total_elements = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.first_drawn = 3) :
  number_in_group s 5 = 35 := by
  sorry

end fifth_group_number_l3366_336640


namespace distance_from_B_to_center_l3366_336623

-- Define the circle and points
def circle_radius : ℝ := 10
def vertical_distance : ℝ := 6
def horizontal_distance : ℝ := 4

-- Define the points A, B, and C
def point_B (a b : ℝ) : ℝ × ℝ := (a, b)
def point_A (a b : ℝ) : ℝ × ℝ := (a, b + vertical_distance)
def point_C (a b : ℝ) : ℝ × ℝ := (a + horizontal_distance, b)

-- Define the conditions
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = circle_radius^2
def right_angle (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2) = 0

-- Theorem statement
theorem distance_from_B_to_center (a b : ℝ) :
  on_circle a (b + vertical_distance) →
  on_circle (a + horizontal_distance) b →
  right_angle (point_A a b) (point_B a b) (point_C a b) →
  a^2 + b^2 = 74 :=
sorry

end distance_from_B_to_center_l3366_336623


namespace walkers_on_same_side_l3366_336692

/-- Represents a person walking around a regular pentagon -/
structure Walker where
  speed : ℝ
  startPosition : ℕ

/-- The time when two walkers start walking on the same side of a regular pentagon -/
def timeOnSameSide (perimeterLength : ℝ) (walker1 walker2 : Walker) : ℝ :=
  sorry

/-- Theorem stating the time when two specific walkers start on the same side of a regular pentagon -/
theorem walkers_on_same_side :
  let perimeterLength : ℝ := 2000
  let walker1 : Walker := { speed := 50, startPosition := 0 }
  let walker2 : Walker := { speed := 46, startPosition := 2 }
  timeOnSameSide perimeterLength walker1 walker2 = 104 := by
  sorry

end walkers_on_same_side_l3366_336692


namespace sum_digits_M_times_2013_l3366_336675

/-- A number composed of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem: The sum of digits of M × 2013 is 1200, where M is composed of 200 consecutive ones -/
theorem sum_digits_M_times_2013 :
  sum_of_digits (consecutive_ones 200 * 2013) = 1200 := by
  sorry

end sum_digits_M_times_2013_l3366_336675


namespace mean_equality_implies_x_equals_8_l3366_336691

theorem mean_equality_implies_x_equals_8 :
  let mean1 := (8 + 10 + 24) / 3
  let mean2 := (16 + x + 18) / 3
  mean1 = mean2 → x = 8 := by
  sorry

end mean_equality_implies_x_equals_8_l3366_336691


namespace camera_and_lens_cost_l3366_336641

theorem camera_and_lens_cost
  (old_camera_cost : ℝ)
  (new_camera_percentage : ℝ)
  (lens_original_price : ℝ)
  (lens_discount : ℝ)
  (h1 : old_camera_cost = 4000)
  (h2 : new_camera_percentage = 1.3)
  (h3 : lens_original_price = 400)
  (h4 : lens_discount = 200) :
  old_camera_cost * new_camera_percentage + (lens_original_price - lens_discount) = 5400 :=
by sorry

end camera_and_lens_cost_l3366_336641


namespace preceding_sum_40_times_l3366_336681

theorem preceding_sum_40_times (n : ℕ) : 
  (n ≠ 0) → ((n * (n - 1)) / 2 = 40 * n) → n = 81 := by
  sorry

end preceding_sum_40_times_l3366_336681


namespace stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l3366_336627

theorem stewart_farm_sheep_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun sheep_count horse_count sheep_ratio horse_ratio =>
    sheep_count * horse_ratio = horse_count * sheep_ratio ∧
    horse_count * 230 = 12880 →
    sheep_count = 40

-- The proof is omitted
theorem stewart_farm_sheep_count_proof : stewart_farm_sheep_count 40 56 5 7 := by
  sorry

end stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l3366_336627


namespace coin_problem_l3366_336634

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

theorem coin_problem (p d n q : ℕ) : 
  p + n + d + q = 12 →  -- Total number of coins
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 →  -- At least one of each type
  q = 2 * d →  -- Twice as many quarters as dimes
  p * penny + n * nickel + d * dime + q * quarter = 128 →  -- Total value in cents
  n = 3 := by sorry

end coin_problem_l3366_336634


namespace cube_colorings_correct_dodecahedron_colorings_correct_l3366_336669

/-- The number of rotational symmetries of a cube -/
def cubeSymmetries : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedronSymmetries : ℕ := 60

/-- The number of geometrically distinct colorings of a cube with 6 different colors -/
def cubeColorings : ℕ := 30

/-- The number of geometrically distinct colorings of a dodecahedron with 12 different colors -/
def dodecahedronColorings : ℕ := (Nat.factorial 11) / 5

theorem cube_colorings_correct :
  cubeColorings = (Nat.factorial 6) / cubeSymmetries :=
sorry

theorem dodecahedron_colorings_correct :
  dodecahedronColorings = (Nat.factorial 12) / dodecahedronSymmetries :=
sorry

end cube_colorings_correct_dodecahedron_colorings_correct_l3366_336669


namespace percentage_not_red_roses_is_92_percent_l3366_336688

/-- Represents the number of flowers of each type in the garden -/
structure GardenFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers in the garden -/
def totalFlowers (g : GardenFlowers) : ℕ :=
  g.roses + g.tulips + g.daisies + g.lilies + g.sunflowers

/-- Calculates the number of red roses in the garden -/
def redRoses (g : GardenFlowers) : ℕ :=
  g.roses / 2

/-- Calculates the percentage of flowers that are not red roses -/
def percentageNotRedRoses (g : GardenFlowers) : ℚ :=
  (totalFlowers g - redRoses g : ℚ) / (totalFlowers g : ℚ) * 100

/-- Theorem stating that 92% of flowers in the given garden are not red roses -/
theorem percentage_not_red_roses_is_92_percent (g : GardenFlowers) 
  (h1 : g.roses = 25)
  (h2 : g.tulips = 40)
  (h3 : g.daisies = 60)
  (h4 : g.lilies = 15)
  (h5 : g.sunflowers = 10) :
  percentageNotRedRoses g = 92 := by
  sorry

end percentage_not_red_roses_is_92_percent_l3366_336688


namespace hot_chocolate_servings_l3366_336686

/-- Represents the recipe requirements for 6 servings --/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Represents the available ingredients --/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Calculates the number of servings possible for a given ingredient --/
def servings_for_ingredient (required : ℚ) (available : ℚ) : ℚ :=
  (available / required) * 6

/-- Finds the minimum number of servings possible across all ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min
    (servings_for_ingredient recipe.chocolate available.chocolate)
    (min
      (servings_for_ingredient recipe.sugar available.sugar)
      (min
        (servings_for_ingredient recipe.milk available.milk)
        (servings_for_ingredient recipe.vanilla available.vanilla)))

theorem hot_chocolate_servings
  (recipe : Recipe)
  (available : Available)
  (h_recipe : recipe = { chocolate := 3, sugar := 1/2, milk := 6, vanilla := 3/2 })
  (h_available : available = { chocolate := 8, sugar := 3, milk := 15, vanilla := 5 }) :
  max_servings recipe available = 15 := by
  sorry

end hot_chocolate_servings_l3366_336686


namespace max_value_of_function_l3366_336625

theorem max_value_of_function (x : Real) (h : x ∈ Set.Ioo 0 Real.pi) :
  (2 * Real.sin (x / 2) * (1 - Real.sin (x / 2)) * (1 + Real.sin (x / 2))^2) ≤ (107 + 51 * Real.sqrt 17) / 256 := by
  sorry

end max_value_of_function_l3366_336625


namespace no_square_from_square_cut_l3366_336612

-- Define a square
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

-- Define a straight cut
def StraightCut (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem: It's impossible to create a square from a larger square by a single straight cut
theorem no_square_from_square_cut (s₁ s₂ : ℝ) (h₁ : 0 < s₁) (h₂ : 0 < s₂) (h₃ : s₂ < s₁) :
  ¬∃ (a b c : ℝ), (Square s₁ ∩ StraightCut a b c).Nonempty ∧ 
    (Square s₂).Subset (Square s₁ ∩ StraightCut a b c) :=
sorry

end no_square_from_square_cut_l3366_336612


namespace incenter_in_triangular_prism_l3366_336608

structure TriangularPrism where
  A : Point
  B : Point
  C : Point
  D : Point

def orthogonal_projection (p : Point) (plane : Set Point) : Point :=
  sorry

def distance_to_face (p : Point) (face : Set Point) : ℝ :=
  sorry

def is_incenter (p : Point) (triangle : Set Point) : Prop :=
  sorry

theorem incenter_in_triangular_prism (prism : TriangularPrism) 
  (O : Point) 
  (h1 : O = orthogonal_projection prism.A {prism.B, prism.C, prism.D}) 
  (h2 : distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.B, prism.D} ∧
        distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.C, prism.D}) : 
  is_incenter O {prism.B, prism.C, prism.D} :=
sorry

end incenter_in_triangular_prism_l3366_336608


namespace not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l3366_336660

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (intersects : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem not_parallel_if_intersect_and_contain 
  (a b : Line) (α : Plane) (P : Point) :
  intersects a α ∧ contains α b → ¬ parallel a b := by sorry

-- Theorem 2
theorem perpendicular_if_parallel_and_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b ∧ perpendicular b α → perpendicular a α := by sorry

end not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l3366_336660


namespace positive_integer_solutions_5x_plus_y_11_l3366_336601

theorem positive_integer_solutions_5x_plus_y_11 :
  {(x, y) : ℕ × ℕ | 5 * x + y = 11 ∧ x > 0 ∧ y > 0} = {(1, 6), (2, 1)} := by
  sorry

end positive_integer_solutions_5x_plus_y_11_l3366_336601


namespace sum_of_ratios_ge_six_l3366_336617

theorem sum_of_ratios_ge_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end sum_of_ratios_ge_six_l3366_336617


namespace range_of_a_when_p_or_q_false_l3366_336611

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a_when_p_or_q_false :
  {a : ℝ | ¬(p a ∨ q a)} = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 :=
sorry

end range_of_a_when_p_or_q_false_l3366_336611


namespace marks_trees_l3366_336664

theorem marks_trees (initial_trees : ℕ) (planted_trees : ℕ) : 
  initial_trees = 13 → planted_trees = 12 → initial_trees + planted_trees = 25 := by
  sorry

end marks_trees_l3366_336664


namespace lcm_problem_l3366_336624

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 16) (h2 : Nat.lcm b c = 21) :
  Nat.lcm a c ≥ 336 := by
  sorry

end lcm_problem_l3366_336624


namespace ice_cream_earnings_theorem_l3366_336642

def ice_cream_earnings (daily_increase : ℕ) : List ℕ :=
  [10, 10 + daily_increase, 10 + 2 * daily_increase, 10 + 3 * daily_increase, 10 + 4 * daily_increase]

theorem ice_cream_earnings_theorem (daily_increase : ℕ) :
  (List.sum (ice_cream_earnings daily_increase) = 90) → daily_increase = 4 := by
  sorry

end ice_cream_earnings_theorem_l3366_336642


namespace peters_contribution_l3366_336672

/-- Given four friends pooling money for a purchase, prove Peter's contribution --/
theorem peters_contribution (john quincy andrew peter : ℝ) : 
  john > 0 ∧ 
  peter = 2 * john ∧ 
  quincy = peter + 20 ∧ 
  andrew = 1.15 * quincy ∧ 
  john + peter + quincy + andrew = 1211 →
  peter = 370.80 := by
  sorry

end peters_contribution_l3366_336672


namespace range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l3366_336603

/-- Proposition p: x^2 - 4ax + 3a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem range_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

theorem range_a_for_not_p_sufficient_not_necessary_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, (¬p x a → (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
    ∃ x : ℝ, (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ p x a) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l3366_336603


namespace range_of_a_l3366_336639

open Real

theorem range_of_a (x₁ x₂ a : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_distinct : x₁ ≠ x₂)
  (h_equation : x₁ + a * (x₂ - 2 * ℯ * x₁) * (log x₂ - log x₁) = 0) :
  a < 0 ∨ a ≥ 1 / ℯ := by sorry

end range_of_a_l3366_336639


namespace paving_cost_l3366_336638

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end paving_cost_l3366_336638


namespace probability_even_sum_le_8_l3366_336694

def dice_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 12

theorem probability_even_sum_le_8 : 
  (favorable_outcomes : ℚ) / dice_outcomes = 1 / 3 := by sorry

end probability_even_sum_le_8_l3366_336694


namespace prime_sequence_ones_digit_l3366_336657

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
sorry

end prime_sequence_ones_digit_l3366_336657


namespace infinitely_many_superabundant_numbers_l3366_336674

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define superabundant numbers
def is_superabundant (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

-- Define the set of superabundant numbers
def superabundant_set : Set ℕ :=
  {m : ℕ | is_superabundant m}

-- Theorem statement
theorem infinitely_many_superabundant_numbers :
  Set.Infinite superabundant_set := by sorry

end infinitely_many_superabundant_numbers_l3366_336674


namespace library_visitors_average_l3366_336613

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 5 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 285 := by
  sorry

end library_visitors_average_l3366_336613


namespace triangle_properties_l3366_336646

/-- Triangle ABC with vertices A(5,1), B(1,3), and C(4,4) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from AB in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 4 = 0

/-- The circumcircle of triangle ABC -/
def circumcircle (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => (p.1 - 3)^2 + (p.2 - 2)^2 = 5

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = (5, 1)) 
  (h2 : t.B = (1, 3)) 
  (h3 : t.C = (4, 4)) : 
  (∀ p, altitude t p ↔ 2 * p.1 - p.2 - 4 = 0) ∧ 
  (∀ p, circumcircle t p ↔ (p.1 - 3)^2 + (p.2 - 2)^2 = 5) := by
  sorry

end triangle_properties_l3366_336646


namespace geometric_sequence_seventh_term_l3366_336632

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_product : a 2 * a 4 = 16) :
  a 7 = 64 := by
  sorry

end geometric_sequence_seventh_term_l3366_336632


namespace circle_radius_l3366_336616

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 16

-- Define the ellipse equation (not used in the proof, but included for completeness)
def ellipse_equation (x y : ℝ) : Prop :=
  (x - 2)^2 / 25 + (y - 1)^2 / 9 = 1

-- Theorem: The radius of the circle is 4
theorem circle_radius : ∃ (r : ℝ), r = 4 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - 1)^2 = r^2 :=
sorry

end circle_radius_l3366_336616


namespace tree_watering_boys_l3366_336677

theorem tree_watering_boys (total_trees : ℕ) (trees_per_boy : ℕ) (h1 : total_trees = 29) (h2 : trees_per_boy = 3) :
  ∃ (num_boys : ℕ), num_boys * trees_per_boy ≥ total_trees ∧ (num_boys - 1) * trees_per_boy < total_trees ∧ num_boys = 10 :=
sorry

end tree_watering_boys_l3366_336677


namespace min_value_x_l3366_336661

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : 2 * Real.log x ≥ Real.log 8 + Real.log x) (h3 : x ≤ 32) :
  x ≥ 8 ∧ ∀ y : ℝ, y > 0 → 2 * Real.log y ≥ Real.log 8 + Real.log y → y ≤ 32 → y ≥ x := by
  sorry

end min_value_x_l3366_336661


namespace symmetry_axis_sine_function_l3366_336649

theorem symmetry_axis_sine_function (φ : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (3 * x + φ)) →
  (|φ| < Real.pi / 2) →
  (∀ x : ℝ, Real.sin (3 * x + φ) = Real.sin (3 * (3 * Real.pi / 2 - x) + φ)) →
  φ = Real.pi / 4 :=
by sorry

end symmetry_axis_sine_function_l3366_336649


namespace age_difference_is_twelve_l3366_336696

/-- The ages of three people A, B, and C, where C is 12 years younger than A -/
structure Ages where
  A : ℕ
  B : ℕ
  C : ℕ
  h : C = A - 12

/-- The difference between the total age of A and B and the total age of B and C -/
def ageDifference (ages : Ages) : ℕ := ages.A + ages.B - (ages.B + ages.C)

/-- Theorem stating that the age difference is always 12 years -/
theorem age_difference_is_twelve (ages : Ages) : ageDifference ages = 12 := by
  sorry

end age_difference_is_twelve_l3366_336696


namespace summer_work_hours_adjustment_l3366_336654

theorem summer_work_hours_adjustment (
  original_hours_per_week : ℝ)
  (original_weeks : ℕ)
  (total_earnings : ℝ)
  (lost_weeks : ℕ)
  (h1 : original_hours_per_week = 20)
  (h2 : original_weeks = 12)
  (h3 : total_earnings = 3000)
  (h4 : lost_weeks = 2)
  (h5 : total_earnings = original_hours_per_week * original_weeks * (total_earnings / (original_hours_per_week * original_weeks)))
  : ∃ new_hours_per_week : ℝ,
    new_hours_per_week * (original_weeks - lost_weeks) * (total_earnings / (original_hours_per_week * original_weeks)) = total_earnings ∧
    new_hours_per_week = 24 :=
by sorry

end summer_work_hours_adjustment_l3366_336654


namespace prime_square_sum_l3366_336650

theorem prime_square_sum (p q n : ℕ) : 
  Prime p → Prime q → n^2 = p^2 + q^2 + p^2 * q^2 → 
  ((p = 2 ∧ q = 3 ∧ n = 7) ∨ (p = 3 ∧ q = 2 ∧ n = 7)) := by
  sorry

end prime_square_sum_l3366_336650


namespace certain_number_sum_l3366_336605

theorem certain_number_sum (x : ℤ) : x + (-27) = 30 → x = 57 := by
  sorry

end certain_number_sum_l3366_336605


namespace numbers_below_nine_and_twenty_four_are_composite_l3366_336606

def below_nine (k : ℕ) : ℕ := 4 * k^2 + 5 * k + 1

def below_twenty_four (k : ℕ) : ℕ := 4 * k^2 + 5 * k

theorem numbers_below_nine_and_twenty_four_are_composite :
  (∀ k : ℕ, k ≥ 1 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_nine k = a * b) ∧
  (∀ k : ℕ, k ≥ 2 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_twenty_four k = a * b) :=
sorry

end numbers_below_nine_and_twenty_four_are_composite_l3366_336606


namespace saree_price_calculation_l3366_336683

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.15) = 272) → P = 400 := by
  sorry

end saree_price_calculation_l3366_336683


namespace jimin_calculation_l3366_336651

theorem jimin_calculation (x : ℤ) : x + 20 = 60 → 34 - x = -6 := by
  sorry

end jimin_calculation_l3366_336651


namespace dual_colored_cubes_count_l3366_336622

/-- Represents a cube painted with two colors on opposite face pairs --/
structure PaintedCube where
  size : ℕ
  color1 : String
  color2 : String

/-- Represents a smaller cube after cutting the original cube --/
structure SmallCube where
  hasColor1 : Bool
  hasColor2 : Bool

/-- Cuts a painted cube into smaller cubes --/
def cutCube (c : PaintedCube) : List SmallCube :=
  sorry

/-- Counts the number of small cubes with both colors --/
def countDualColorCubes (cubes : List SmallCube) : ℕ :=
  sorry

/-- Theorem stating that a cube painted as described and cut into 64 pieces will have 16 dual-colored cubes --/
theorem dual_colored_cubes_count 
  (c : PaintedCube) 
  (h1 : c.size = 4) 
  (h2 : c.color1 ≠ c.color2) : 
  countDualColorCubes (cutCube c) = 16 :=
sorry

end dual_colored_cubes_count_l3366_336622


namespace shaded_area_fraction_l3366_336621

/-- 
Given a rectangle with length l and width w, and points P and Q as midpoints of two adjacent sides,
prove that the shaded area is 7/8 of the total area when the triangle formed by P, Q, and the 
vertex at the intersection of uncut sides is unshaded.
-/
theorem shaded_area_fraction (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  let total_area := l * w
  let unshaded_triangle_area := (l / 2) * (w / 2) / 2
  let shaded_area := total_area - unshaded_triangle_area
  (shaded_area / total_area) = 7 / 8 := by
  sorry

end shaded_area_fraction_l3366_336621


namespace equilateral_triangle_area_perimeter_ratio_l3366_336685

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l3366_336685


namespace x_value_proof_l3366_336607

theorem x_value_proof (x : ℝ) : 
  3.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 3.6 := by
sorry

end x_value_proof_l3366_336607


namespace max_candies_eaten_is_27_l3366_336695

/-- Represents a box of candies with a label -/
structure CandyBox where
  label : Nat
  candies : Nat

/-- Represents the state of all candy boxes -/
def GameState := List CandyBox

/-- Initializes the game state with three boxes -/
def initialState : GameState :=
  [{ label := 4, candies := 10 }, { label := 7, candies := 10 }, { label := 10, candies := 10 }]

/-- Performs one operation on the game state -/
def performOperation (state : GameState) (boxIndex : Nat) : Option GameState :=
  sorry

/-- Calculates the total number of candies eaten after a sequence of operations -/
def candiesEaten (operations : List Nat) : Nat :=
  sorry

/-- The maximum number of candies that can be eaten -/
def maxCandiesEaten : Nat :=
  sorry

/-- Theorem stating the maximum number of candies that can be eaten is 27 -/
theorem max_candies_eaten_is_27 : maxCandiesEaten = 27 := by
  sorry

end max_candies_eaten_is_27_l3366_336695
