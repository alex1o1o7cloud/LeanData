import Mathlib

namespace brother_bought_ten_books_l3089_308977

/-- The number of books Sarah's brother bought -/
def brothers_total_books (sarah_paperbacks sarah_hardbacks : ℕ) : ℕ :=
  (sarah_paperbacks / 3) + (sarah_hardbacks * 2)

/-- Theorem stating that Sarah's brother bought 10 books in total -/
theorem brother_bought_ten_books :
  brothers_total_books 6 4 = 10 := by
  sorry

end brother_bought_ten_books_l3089_308977


namespace hexagon_parallelogram_theorem_l3089_308929

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a hexagon as a collection of 6 points
structure Hexagon :=
  (A B C D E F : Point)

-- Define a property for convex hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define a quadrilateral as a collection of 4 points
structure Quadrilateral :=
  (P Q R S : Point)

-- Define a property for parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem hexagon_parallelogram_theorem (h : Hexagon) 
  (convex_h : is_convex h)
  (para_ABDE : is_parallelogram ⟨h.A, h.B, h.D, h.E⟩)
  (para_ACDF : is_parallelogram ⟨h.A, h.C, h.D, h.F⟩) :
  is_parallelogram ⟨h.B, h.C, h.E, h.F⟩ := by
  sorry

end hexagon_parallelogram_theorem_l3089_308929


namespace solution_set_inequality_l3089_308973

theorem solution_set_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end solution_set_inequality_l3089_308973


namespace first_investment_rate_l3089_308938

/-- Proves that given the conditions, the interest rate of the first investment is 10% --/
theorem first_investment_rate (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 650 →
  second_rate = 0.08 →
  income_difference = 83 →
  ∃ (first_rate : ℝ),
    first_rate * (total_investment - second_investment) - second_rate * second_investment = income_difference ∧
    first_rate = 0.10 :=
by sorry

end first_investment_rate_l3089_308938


namespace factorization_equality_l3089_308988

theorem factorization_equality (x : ℝ) : 
  (x + 1)^4 + (x + 3)^4 - 272 = 2 * (x^2 + 4*x + 19) * (x + 5) * (x - 1) := by
  sorry

end factorization_equality_l3089_308988


namespace divisibility_condition_l3089_308986

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def form_number (B : ℕ) : ℕ := 5000 + 200 + 10 * B + 6

theorem divisibility_condition (B : ℕ) (h : B ≤ 9) :
  is_divisible_by_3 (form_number B) ↔ (B = 2 ∨ B = 5 ∨ B = 8) := by
  sorry

end divisibility_condition_l3089_308986


namespace ramsey_3_3_l3089_308918

/-- A complete graph with 6 vertices where each edge is colored either blue or red. -/
def ColoredGraph := Fin 6 → Fin 6 → Bool

/-- The graph is complete and each edge has a color (blue or red). -/
def is_valid_coloring (g : ColoredGraph) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g i j = true ∨ g i j = false

/-- A triangle in the graph with all edges of the same color. -/
def monochromatic_triangle (g : ColoredGraph) : Prop :=
  ∃ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ((g i j = true ∧ g j k = true ∧ g i k = true) ∨
     (g i j = false ∧ g j k = false ∧ g i k = false))

/-- The Ramsey theorem for R(3,3) -/
theorem ramsey_3_3 (g : ColoredGraph) (h : is_valid_coloring g) : 
  monochromatic_triangle g := by
  sorry

end ramsey_3_3_l3089_308918


namespace calculation_proof_l3089_308906

theorem calculation_proof : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculation_proof_l3089_308906


namespace expression_expansion_l3089_308945

theorem expression_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 - 5 * x^3 + 2) = 3 / x^2 - 15 * x^3 / 7 + 6 / 7 := by
  sorry

end expression_expansion_l3089_308945


namespace reciprocal_sum_theorem_l3089_308901

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end reciprocal_sum_theorem_l3089_308901


namespace ellipse_hyperbola_intersection_l3089_308905

/-- Given an ellipse and a hyperbola sharing a common focus, prove that a² = 11 under specific conditions --/
theorem ellipse_hyperbola_intersection (a b : ℝ) : 
  a > b → b > 0 →
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →  -- Ellipse C1
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / 2 - (y t)^2 / 8 = 1) →  -- Hyperbola C2
  (a^2 - b^2 = 10) →  -- Common focus condition
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = a^2) ∧ (B.1^2 + B.2^2 = a^2) ∧  -- A and B on circle
    (∃ (k : ℝ), A.2 = k * A.1 ∧ B.2 = k * B.1) ∧  -- A and B on asymptote
    (∃ (C D : ℝ × ℝ), 
      C.1^2 / a^2 + C.2^2 / b^2 = 1 ∧
      D.1^2 / a^2 + D.2^2 / b^2 = 1 ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2*a/3)^2)) →  -- C1 divides AB into three equal parts
  a^2 = 11 := by
sorry


end ellipse_hyperbola_intersection_l3089_308905


namespace inequality_proof_l3089_308921

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^4 + y^2*z^2) / (x^(5/2)*(y+z)) + (y^4 + z^2*x^2) / (y^(5/2)*(z+x)) +
  (z^4 + x^2*y^2) / (z^(5/2)*(x+y)) ≥ 1 := by
  sorry

end inequality_proof_l3089_308921


namespace tan_half_angle_l3089_308937

theorem tan_half_angle (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.sin α + 2 * Real.cos α = 2) : Real.tan (α / 2) = 3 / 2 := by
  sorry

end tan_half_angle_l3089_308937


namespace negation_of_existence_negation_of_quadratic_inequality_l3089_308931

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬∃ x, P x) ↔ (∀ x, ¬P x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3089_308931


namespace scale_division_theorem_l3089_308994

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- The total length of the scale -/
def totalLength : Length := ⟨6, 8, by norm_num⟩

/-- Number of equal parts to divide the scale into -/
def numParts : ℕ := 2

/-- Represents the result of dividing the scale -/
def dividedLength : Length := ⟨3, 4, by norm_num⟩

theorem scale_division_theorem :
  (totalLength.toInches / numParts : ℕ) = dividedLength.toInches := by
  sorry

end scale_division_theorem_l3089_308994


namespace gcd_problems_l3089_308912

theorem gcd_problems : 
  (Nat.gcd 120 168 = 24) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end gcd_problems_l3089_308912


namespace simplify_expression_l3089_308972

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 15 - (6 - 3*y - 9*y^2) = 18*y^2 + 6*y + 9 := by
  sorry

end simplify_expression_l3089_308972


namespace milan_bill_cost_l3089_308927

/-- Calculates the total cost of a long distance phone bill -/
def long_distance_bill_cost (monthly_fee : ℚ) (cost_per_minute : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + cost_per_minute * minutes_used

/-- Proves that Milan's long distance bill cost is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  long_distance_bill_cost monthly_fee cost_per_minute minutes_used = 2336 / 100 := by
  sorry

end milan_bill_cost_l3089_308927


namespace complex_division_equals_i_l3089_308967

theorem complex_division_equals_i : (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end complex_division_equals_i_l3089_308967


namespace p_or_not_q_l3089_308957

def p : Prop := ∃ α : ℝ, Real.sin (Real.pi - α) = Real.cos α

def q : Prop := ∀ m : ℝ, m > 0 → 
  (∀ x y : ℝ, x^2/m^2 - y^2/m^2 = 1 → 
    Real.sqrt (1 + (m^2/m^2)) = Real.sqrt 2) ∧
  (∃ n : ℝ, n ≤ 0 ∧ 
    (∀ x y : ℝ, x^2/n^2 - y^2/n^2 = 1 → 
      Real.sqrt (1 + (n^2/n^2)) = Real.sqrt 2))

theorem p_or_not_q : (¬p) ∨ q := by sorry

end p_or_not_q_l3089_308957


namespace sam_average_letters_per_day_l3089_308956

/-- The average number of letters Sam wrote per day -/
theorem sam_average_letters_per_day :
  let tuesday_letters : ℕ := 7
  let wednesday_letters : ℕ := 3
  let total_days : ℕ := 2
  let total_letters : ℕ := tuesday_letters + wednesday_letters
  (total_letters : ℚ) / total_days = 5 := by
  sorry

end sam_average_letters_per_day_l3089_308956


namespace quadratic_no_real_roots_l3089_308940

theorem quadratic_no_real_roots :
  ∀ x : ℝ, 2 * (x - 1)^2 + 2 ≠ 0 := by
sorry

end quadratic_no_real_roots_l3089_308940


namespace a_in_A_l3089_308944

def A : Set ℝ := {x | x < 2 * Real.sqrt 3}

theorem a_in_A : 2 ∈ A := by
  sorry

end a_in_A_l3089_308944


namespace good_quality_sufficient_for_not_cheap_l3089_308917

-- Define the propositions
variable (good_quality : Prop)
variable (not_cheap : Prop)

-- Define the given equivalence
axiom you_get_what_you_pay_for : (good_quality → not_cheap) ↔ (¬not_cheap → ¬good_quality)

-- Theorem to prove
theorem good_quality_sufficient_for_not_cheap : good_quality → not_cheap := by
  sorry

end good_quality_sufficient_for_not_cheap_l3089_308917


namespace binomial_12_9_l3089_308928

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l3089_308928


namespace z_in_first_quadrant_l3089_308935

def z : ℂ := (2 + Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end z_in_first_quadrant_l3089_308935


namespace sum_of_xyz_l3089_308936

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y + 5) : 
  x + y + z = 7 * x + 5 := by
sorry

end sum_of_xyz_l3089_308936


namespace jana_walking_distance_l3089_308903

/-- Given a walking speed of 1 mile per 24 minutes, prove that the distance walked in 36 minutes is 1.5 miles. -/
theorem jana_walking_distance (speed : ℚ) (time : ℕ) (distance : ℚ) : 
  speed = 1 / 24 → time = 36 → distance = speed * time → distance = 3/2 := by
  sorry

end jana_walking_distance_l3089_308903


namespace stratified_sample_size_l3089_308909

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  workshops : List Workshop
  sampleSize : ℕ
  sampledFromC : ℕ

/-- Calculates the total production quantity across all workshops -/
def totalQuantity (s : StratifiedSample) : ℕ :=
  s.workshops.foldl (fun acc w => acc + w.quantity) 0

/-- Theorem stating the relationship between sample size and workshop quantities -/
theorem stratified_sample_size 
  (s : StratifiedSample)
  (hWorkshops : s.workshops = [⟨600⟩, ⟨400⟩, ⟨300⟩])
  (hSampledC : s.sampledFromC = 6) :
  s.sampleSize = 26 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l3089_308909


namespace f_f_zero_equals_three_pi_squared_minus_four_l3089_308910

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero_equals_three_pi_squared_minus_four :
  f (f 0) = 3 * Real.pi^2 - 4 := by sorry

end f_f_zero_equals_three_pi_squared_minus_four_l3089_308910


namespace full_house_prob_modified_deck_l3089_308983

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (ranks : Nat)
  (cards_per_rank : Nat)
  (hand_size : Nat)

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of drawing a full house -/
def full_house_probability (deck : ModifiedDeck) : Rat :=
  let total_cards := deck.ranks * deck.cards_per_rank
  let total_combinations := choose total_cards deck.hand_size
  let full_house_combinations := 
    deck.ranks * choose deck.cards_per_rank 3 * (deck.ranks - 1) * choose deck.cards_per_rank 2
  full_house_combinations / total_combinations

/-- Theorem: The probability of drawing a full house in the given modified deck is 40/1292 -/
theorem full_house_prob_modified_deck :
  full_house_probability ⟨5, 4, 5⟩ = 40 / 1292 := by
  sorry


end full_house_prob_modified_deck_l3089_308983


namespace planting_schemes_count_l3089_308949

/-- The number of seed types -/
def num_seed_types : ℕ := 5

/-- The number of plots -/
def num_plots : ℕ := 4

/-- The number of seed types to be selected -/
def num_selected : ℕ := 4

/-- The number of options for the first plot (pumpkins or pomegranates) -/
def first_plot_options : ℕ := 2

/-- Calculate the number of planting schemes -/
def num_planting_schemes : ℕ :=
  first_plot_options * (Nat.choose (num_seed_types - 1) (num_selected - 1)) * (Nat.factorial (num_plots - 1))

theorem planting_schemes_count : num_planting_schemes = 48 := by sorry

end planting_schemes_count_l3089_308949


namespace sequence_split_equal_sum_l3089_308926

theorem sequence_split_equal_sum (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (b : ℕ) (S : ℕ) (splits : List (List ℕ)),
    b > 1 ∧
    splits.length = b ∧
    (∀ l ∈ splits, l.sum = S) ∧
    splits.join = List.range p) ↔ p = 3 := by
  sorry

end sequence_split_equal_sum_l3089_308926


namespace expression_evaluation_l3089_308997

/-- Given real numbers x, y, and z, prove that the expression
    ((P+Q)/(P-Q) - (P-Q)/(P+Q)) equals (x^2 - y^2 - 2yz - z^2) / (xy + xz),
    where P = x + y + z and Q = x - y - z. -/
theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*y + x*z) := by
  sorry

end expression_evaluation_l3089_308997


namespace orange_juice_fraction_l3089_308900

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 500
  let pitcher2_capacity : ℚ := 600
  let pitcher1_juice_ratio : ℚ := 1/4
  let pitcher2_juice_ratio : ℚ := 1/3
  let total_juice := pitcher1_capacity * pitcher1_juice_ratio + pitcher2_capacity * pitcher2_juice_ratio
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 13/44 := by sorry

end orange_juice_fraction_l3089_308900


namespace expression_simplification_l3089_308970

theorem expression_simplification :
  (((3 + 4 + 5 + 6 + 7) / 3) + ((3 * 6 + 9)^2 / 9)) = 268 / 3 :=
by sorry

end expression_simplification_l3089_308970


namespace sum_of_numbers_l3089_308984

theorem sum_of_numbers : (6 / 5 : ℚ) + (1 / 10 : ℚ) + (156 / 100 : ℚ) = 286 / 100 := by
  sorry

end sum_of_numbers_l3089_308984


namespace quadratic_equation_roots_l3089_308946

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + a = 0 ∧ x = 2) → 
  (a = 2 ∧ ∃ y : ℝ, y^2 - 3*y + a = 0 ∧ y = 1) := by
  sorry

end quadratic_equation_roots_l3089_308946


namespace range_of_a_l3089_308975

/-- Proposition p: The real number x satisfies x^2 - 4ax + 3a^2 < 0 -/
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: The real number x satisfies x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

/-- The set of x satisfying proposition p -/
def A (a : ℝ) : Set ℝ := {x | p a x}

/-- The set of x satisfying proposition q -/
def B : Set ℝ := {x | q x}

theorem range_of_a (a : ℝ) :
  a > 0 ∧ 
  (∀ x, ¬(q x) → ¬(p a x)) ∧
  (∃ x, ¬(q x) ∧ p a x) →
  1 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l3089_308975


namespace unique_a_value_l3089_308947

def U : Set ℤ := {-5, -3, 1, 2, 3, 4, 5, 6}

def A : Set ℤ := {x | x^2 - 7*x + 12 = 0}

def B (a : ℤ) : Set ℤ := {a^2, 2*a - 1, 6}

theorem unique_a_value : 
  ∃! a : ℤ, A ∩ B a = {4} ∧ B a ⊆ U ∧ a = -2 :=
sorry

end unique_a_value_l3089_308947


namespace length_width_ratio_l3089_308996

/-- Represents a rectangular roof --/
structure RectangularRoof where
  length : ℝ
  width : ℝ

/-- Properties of the specific roof in the problem --/
def problem_roof : RectangularRoof → Prop
  | roof => roof.length * roof.width = 900 ∧ 
            roof.length - roof.width = 45

/-- The theorem stating the ratio of length to width --/
theorem length_width_ratio (roof : RectangularRoof) 
  (h : problem_roof roof) : 
  roof.length / roof.width = 4 := by
  sorry

#check length_width_ratio

end length_width_ratio_l3089_308996


namespace sin_C_value_area_when_b_is_6_l3089_308961

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 + t.c^2 - Real.sqrt 3 * t.a * t.c = t.b^2 ∧
  3 * t.a = 2 * t.b

-- Theorem for part (I)
theorem sin_C_value (t : Triangle) (h : satisfies_conditions t) :
  Real.sin t.C = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

-- Theorem for part (II)
theorem area_when_b_is_6 (t : Triangle) (h : satisfies_conditions t) (h_b : t.b = 6) :
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end sin_C_value_area_when_b_is_6_l3089_308961


namespace min_length_MN_l3089_308985

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Theorem: Minimum length of MN in a unit cube -/
theorem min_length_MN (cube : Cube) (M N L : Point3D) : 
  (cube.A.x = 0 ∧ cube.A.y = 0 ∧ cube.A.z = 0) →  -- A is at origin
  (cube.B.x = 1 ∧ cube.B.y = 0 ∧ cube.B.z = 0) →  -- B is at (1,0,0)
  (cube.C.x = 1 ∧ cube.C.y = 1 ∧ cube.C.z = 0) →  -- C is at (1,1,0)
  (cube.D.x = 0 ∧ cube.D.y = 1 ∧ cube.D.z = 0) →  -- D is at (0,1,0)
  (cube.A1.x = 0 ∧ cube.A1.y = 0 ∧ cube.A1.z = 1) →  -- A1 is at (0,0,1)
  (cube.C1.x = 1 ∧ cube.C1.y = 1 ∧ cube.C1.z = 1) →  -- C1 is at (1,1,1)
  (cube.D1.x = 0 ∧ cube.D1.y = 1 ∧ cube.D1.z = 1) →  -- D1 is at (0,1,1)
  (∃ t : ℝ, M.x = t * cube.A1.x ∧ M.y = t * cube.A1.y ∧ M.z = t * cube.A1.z) →  -- M is on ray AA1
  (∃ s : ℝ, N.x = cube.B.x + s * (cube.C.x - cube.B.x) ∧ 
            N.y = cube.B.y + s * (cube.C.y - cube.B.y) ∧ 
            N.z = cube.B.z + s * (cube.C.z - cube.B.z)) →  -- N is on ray BC
  (∃ u : ℝ, L.x = cube.C1.x + u * (cube.D1.x - cube.C1.x) ∧ 
            L.y = cube.C1.y + u * (cube.D1.y - cube.C1.y) ∧ 
            L.z = cube.C1.z + u * (cube.D1.z - cube.C1.z)) →  -- L is on edge C1D1
  (∃ v : ℝ, M.x + v * (N.x - M.x) = L.x ∧ 
            M.y + v * (N.y - M.y) = L.y ∧ 
            M.z + v * (N.z - M.z) = L.z) →  -- MN intersects C1D1 at L
  (∀ M' N' : Point3D, 
    (∃ t' : ℝ, M'.x = t' * cube.A1.x ∧ M'.y = t' * cube.A1.y ∧ M'.z = t' * cube.A1.z) →
    (∃ s' : ℝ, N'.x = cube.B.x + s' * (cube.C.x - cube.B.x) ∧ 
              N'.y = cube.B.y + s' * (cube.C.y - cube.B.y) ∧ 
              N'.z = cube.B.z + s' * (cube.C.z - cube.B.z)) →
    Real.sqrt ((M'.x - N'.x)^2 + (M'.y - N'.y)^2 + (M'.z - N'.z)^2) ≥ 3) →
  Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2) = 3 :=
by sorry

end min_length_MN_l3089_308985


namespace largest_x_floor_fraction_l3089_308916

open Real

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 11 / 12) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 11 / 12 → y ≤ 120 / 11) := by
  sorry

end largest_x_floor_fraction_l3089_308916


namespace students_studying_both_subjects_difference_l3089_308980

theorem students_studying_both_subjects_difference (total : ℕ) 
  (math_min math_max science_min science_max : ℕ) : 
  total = 2500 →
  math_min = 1875 →
  math_max = 2000 →
  science_min = 875 →
  science_max = 1125 →
  let max_both := math_min + science_min - total
  let min_both := total - math_max - science_max
  max_both - min_both = 625 := by sorry

end students_studying_both_subjects_difference_l3089_308980


namespace helen_baked_554_cookies_this_morning_l3089_308991

/-- Given the total number of chocolate chip cookies and the number baked yesterday,
    calculate the number of chocolate chip cookies baked this morning. -/
def cookies_baked_this_morning (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

/-- Theorem stating that Helen baked 554 chocolate chip cookies this morning. -/
theorem helen_baked_554_cookies_this_morning :
  cookies_baked_this_morning 1081 527 = 554 := by
  sorry

end helen_baked_554_cookies_this_morning_l3089_308991


namespace jacob_peter_age_difference_l3089_308989

/-- Given that Peter's age 10 years ago was one-third of Jacob's age at that time,
    and Peter is currently 16 years old, prove that Jacob's current age is 12 years
    more than Peter's current age. -/
theorem jacob_peter_age_difference :
  ∀ (peter_age_10_years_ago jacob_age_10_years_ago : ℕ),
  peter_age_10_years_ago = jacob_age_10_years_ago / 3 →
  peter_age_10_years_ago + 10 = 16 →
  jacob_age_10_years_ago + 10 - (peter_age_10_years_ago + 10) = 12 :=
by
  sorry

end jacob_peter_age_difference_l3089_308989


namespace line_distance_theorem_l3089_308943

/-- The line equation 4x - 3y + c = 0 -/
def line_equation (x y c : ℝ) : Prop := 4 * x - 3 * y + c = 0

/-- The distance function from (1,1) to (a,b) -/
def distance_squared (a b : ℝ) : ℝ := (a - 1)^2 + (b - 1)^2

/-- The theorem stating the relationship between the line and the minimum distance -/
theorem line_distance_theorem (a b c : ℝ) :
  line_equation a b c →
  (∀ x y, line_equation x y c → distance_squared a b ≤ distance_squared x y) →
  distance_squared a b = 4 →
  c = -11 ∨ c = 9 := by sorry

end line_distance_theorem_l3089_308943


namespace root_sum_ratio_l3089_308968

theorem root_sum_ratio (m₁ m₂ : ℝ) (a b : ℝ → ℝ) : 
  (∀ m, m * (a m)^2 - (3 * m - 2) * (a m) + 7 = 0) →
  (∀ m, m * (b m)^2 - (3 * m - 2) * (b m) + 7 = 0) →
  (a m₁ / b m₁ + b m₁ / a m₁ = 2) →
  (a m₂ / b m₂ + b m₂ / a m₂ = 2) →
  m₁ / m₂ + m₂ / m₁ = 194 / 9 := by
  sorry


end root_sum_ratio_l3089_308968


namespace passengers_taken_at_second_station_is_12_l3089_308941

/-- Represents the number of passengers on a train at different stages --/
structure TrainPassengers where
  initial : Nat
  after_first_drop : Nat
  after_first_pickup : Nat
  after_second_drop : Nat
  final : Nat

/-- Calculates the number of passengers taken at the second station --/
def passengers_taken_at_second_station (train : TrainPassengers) : Nat :=
  train.final - train.after_second_drop

/-- Theorem stating the number of passengers taken at the second station --/
theorem passengers_taken_at_second_station_is_12 :
  ∃ (train : TrainPassengers),
    train.initial = 270 ∧
    train.after_first_drop = train.initial - train.initial / 3 ∧
    train.after_first_pickup = train.after_first_drop + 280 ∧
    train.after_second_drop = train.after_first_pickup / 2 ∧
    train.final = 242 ∧
    passengers_taken_at_second_station train = 12 := by
  sorry

#check passengers_taken_at_second_station_is_12

end passengers_taken_at_second_station_is_12_l3089_308941


namespace equidistant_points_in_quadrants_I_II_l3089_308925

/-- A point (x, y) on the line 4x + 6y = 18 that is equidistant from both coordinate axes -/
def EquidistantPoint (x y : ℝ) : Prop :=
  4 * x + 6 * y = 18 ∧ |x| = |y|

/-- A point (x, y) is in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is in quadrant III -/
def InQuadrantIII (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- A point (x, y) is in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, EquidistantPoint x y →
  (InQuadrantI x y ∨ InQuadrantII x y) ∧
  ¬(InQuadrantIII x y ∨ InQuadrantIV x y) :=
by sorry

end equidistant_points_in_quadrants_I_II_l3089_308925


namespace red_peppers_weight_l3089_308982

/-- The weight of red peppers bought by Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ :=
  5.666666667 - 2.8333333333333335

/-- Theorem stating that the weight of red peppers is the difference between
    the total weight of peppers and the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666667 - 2.8333333333333335 := by
  sorry

end red_peppers_weight_l3089_308982


namespace arithmetic_geometric_sequence_l3089_308933

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- not a constant sequence
  (a 2) * (a 6) = (a 3) * (a 3) →  -- 2nd, 3rd, and 6th terms form a geometric sequence
  (a 3) / (a 2) = 3 :=  -- common ratio is 3
by sorry

end arithmetic_geometric_sequence_l3089_308933


namespace multiplication_puzzle_l3089_308953

/-- Given distinct digits a, b, and c different from 1,
    prove that abb × c = bcb1 implies a = 5, b = 3, and c = 7 -/
theorem multiplication_puzzle :
  ∀ a b c : ℕ,
    a ≠ b → b ≠ c → a ≠ c →
    a ≠ 1 → b ≠ 1 → c ≠ 1 →
    a < 10 → b < 10 → c < 10 →
    (100 * a + 10 * b + b) * c = 1000 * b + 100 * c + 10 * b + 1 →
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end multiplication_puzzle_l3089_308953


namespace max_sections_with_five_lines_l3089_308923

/-- The number of sections created by drawing n line segments through a rectangle -/
def num_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else num_sections (n - 1) + n

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_with_five_lines :
  num_sections 5 = 16 := by
  sorry

end max_sections_with_five_lines_l3089_308923


namespace triangle_area_l3089_308958

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem triangle_area (a b c A B C : ℝ) : 
  a = 2 * Real.sqrt 3 →
  f A = 2 →
  b + c = 6 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end triangle_area_l3089_308958


namespace salary_calculation_l3089_308908

def initial_salary : ℚ := 3000
def raise_percentage : ℚ := 10 / 100
def pay_cut_percentage : ℚ := 15 / 100
def bonus : ℚ := 500

def final_salary : ℚ := 
  (initial_salary * (1 + raise_percentage) * (1 - pay_cut_percentage)) + bonus

theorem salary_calculation : final_salary = 3305 := by sorry

end salary_calculation_l3089_308908


namespace geometric_series_common_ratio_l3089_308971

/-- The first term of the geometric series -/
def a₁ : ℚ := 4/3

/-- The second term of the geometric series -/
def a₂ : ℚ := 16/9

/-- The third term of the geometric series -/
def a₃ : ℚ := 64/27

/-- The common ratio of the geometric series -/
def r : ℚ := 4/3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end geometric_series_common_ratio_l3089_308971


namespace percentage_difference_l3089_308976

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 500 / 6 := by
  sorry

end percentage_difference_l3089_308976


namespace min_value_of_sum_l3089_308914

theorem min_value_of_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) :
  x + y ≥ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 8*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 18 := by
  sorry

end min_value_of_sum_l3089_308914


namespace sin_cos_sum_ratio_equals_tan_60_l3089_308990

theorem sin_cos_sum_ratio_equals_tan_60 :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by sorry

end sin_cos_sum_ratio_equals_tan_60_l3089_308990


namespace find_A_l3089_308969

theorem find_A : ∃ A : ℕ, A = 38 ∧ A / 7 = 5 ∧ A % 7 = 3 := by
  sorry

end find_A_l3089_308969


namespace infinite_sum_equality_l3089_308922

/-- For positive real numbers a and b where a > b, the sum of the infinite series
    1/(ba) + 1/(a(2a + b)) + 1/((2a + b)(3a + 2b)) + 1/((3a + 2b)(4a + 3b)) + ...
    is equal to 1/((a + b)b) -/
theorem infinite_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n : ℕ => 1 / ((n * a + (n - 1) * b) * ((n + 1) * a + n * b))
  tsum series = 1 / ((a + b) * b) := by sorry

end infinite_sum_equality_l3089_308922


namespace right_triangle_acute_angles_l3089_308962

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 5 / 4 →  -- Ratio of angles is 5:4
  (a = 50 ∧ b = 40) ∨ (a = 40 ∧ b = 50) := by
sorry

end right_triangle_acute_angles_l3089_308962


namespace total_mechanical_pencils_l3089_308934

/-- Given 4 sets of school supplies with 16 mechanical pencils each, 
    prove that the total number of mechanical pencils is 64. -/
theorem total_mechanical_pencils : 
  let num_sets : ℕ := 4
  let pencils_per_set : ℕ := 16
  num_sets * pencils_per_set = 64 := by
  sorry

end total_mechanical_pencils_l3089_308934


namespace sphere_only_circular_cross_sections_l3089_308992

-- Define the possible geometric shapes
inductive GeometricShape
  | Cylinder
  | Cone
  | Sphere
  | ConeWithCircularBase

-- Define a function to check if a shape has circular cross-sections for all plane intersections
def hasCircularCrossSections (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | _ => false

-- Theorem statement
theorem sphere_only_circular_cross_sections :
  ∀ (shape : GeometricShape),
    hasCircularCrossSections shape ↔ shape = GeometricShape.Sphere :=
by sorry

end sphere_only_circular_cross_sections_l3089_308992


namespace bundle_promotion_better_l3089_308907

-- Define the prices and discounts
def cellphone_price : ℝ := 800
def earbud_price : ℝ := 150
def case_price : ℝ := 40
def cellphone_discount : ℝ := 0.05
def earbud_discount : ℝ := 0.10
def bundle_discount : ℝ := 0.07
def loyalty_discount : ℝ := 0.03
def sales_tax : ℝ := 0.08

-- Define the total cost before promotions
def total_before_promotions : ℝ :=
  (2 * cellphone_price * (1 - cellphone_discount)) +
  (2 * earbud_price * (1 - earbud_discount)) +
  case_price

-- Define the cost after each promotion
def bundle_promotion_cost : ℝ :=
  total_before_promotions * (1 - bundle_discount)

def loyalty_promotion_cost : ℝ :=
  total_before_promotions * (1 - loyalty_discount)

-- Define the final costs including tax
def bundle_final_cost : ℝ :=
  bundle_promotion_cost * (1 + sales_tax)

def loyalty_final_cost : ℝ :=
  loyalty_promotion_cost * (1 + sales_tax)

-- Theorem statement
theorem bundle_promotion_better :
  bundle_final_cost < loyalty_final_cost :=
sorry

end bundle_promotion_better_l3089_308907


namespace birch_tree_arrangement_probability_l3089_308966

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := 9

/-- The number of birch trees -/
def birch_trees : ℕ := 3

/-- The total number of trees -/
def total_trees : ℕ := non_birch_trees + birch_trees

/-- The number of slots available for birch trees -/
def available_slots : ℕ := non_birch_trees + 1

/-- The probability of no two birch trees being adjacent when randomly arranged -/
theorem birch_tree_arrangement_probability :
  (Nat.choose available_slots birch_trees : ℚ) / (Nat.choose total_trees birch_trees : ℚ) = 6 / 11 := by
  sorry

#eval Nat.choose available_slots birch_trees + Nat.choose total_trees birch_trees

end birch_tree_arrangement_probability_l3089_308966


namespace parabola_latus_rectum_p_l3089_308963

/-- A parabola with equation y^2 = 2px and latus rectum line x = -2 has p = 4 -/
theorem parabola_latus_rectum_p (y x p : ℝ) : 
  (y^2 = 2*p*x) →  -- Parabola equation
  (x = -2)      →  -- Latus rectum line equation
  p = 4         :=  -- Conclusion: p equals 4
by sorry

end parabola_latus_rectum_p_l3089_308963


namespace fraction_equality_l3089_308964

theorem fraction_equality (m n s u : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 := by
  sorry

end fraction_equality_l3089_308964


namespace salary_distribution_l3089_308952

/-- Represents the salary distribution problem for three teams of workers --/
theorem salary_distribution
  (total_value : ℝ)
  (team1_people team1_days : ℕ)
  (team2_people team2_days : ℕ)
  (team3_days : ℕ)
  (team3_people_ratio : ℝ)
  (h1 : total_value = 325500)
  (h2 : team1_people = 15)
  (h3 : team1_days = 21)
  (h4 : team2_people = 14)
  (h5 : team2_days = 25)
  (h6 : team3_days = 20)
  (h7 : team3_people_ratio = 1.4) :
  ∃ (salary_per_day : ℝ),
    let team1_salary := salary_per_day * team1_people * team1_days
    let team2_salary := salary_per_day * team2_people * team2_days
    let team3_salary := salary_per_day * (team3_people_ratio * team1_people) * team3_days
    team1_salary + team2_salary + team3_salary = total_value ∧
    team1_salary = 94500 ∧
    team2_salary = 105000 ∧
    team3_salary = 126000 :=
by sorry

end salary_distribution_l3089_308952


namespace power_function_sum_l3089_308959

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (α β : ℝ), ∀ x, f x = α * x ^ β

-- State the theorem
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end power_function_sum_l3089_308959


namespace shaded_area_equals_triangle_area_l3089_308979

/-- Given a circle with diameter d and a perpendicular line from the center to the diameter,
    the area of the shaded region formed by a semicircle with radius equal to the radius
    of the original circle, minus the area of the right triangle formed by the diameter
    and radius of the original circle, is equal to the area of the right triangle. -/
theorem shaded_area_equals_triangle_area (d : ℝ) (h : d > 0) : 
  let r := d / 2
  let semicircle_area := π * r^2 / 2
  let triangle_area := d * r / 2
  semicircle_area - (semicircle_area - triangle_area) = triangle_area := by
  sorry

#check shaded_area_equals_triangle_area

end shaded_area_equals_triangle_area_l3089_308979


namespace largest_expression_l3089_308978

def expr_A : ℚ := 3 + 0 + 4 + 8
def expr_B : ℚ := 3 * 0 + 4 + 8
def expr_C : ℚ := 3 + 0 * 4 + 8
def expr_D : ℚ := 3 + 0 + 4 * 8
def expr_E : ℚ := 3 * 0 * 4 * 8
def expr_F : ℚ := (3 + 0 + 4) / 8

theorem largest_expression :
  expr_D = 35 ∧
  expr_D > expr_A ∧
  expr_D > expr_B ∧
  expr_D > expr_C ∧
  expr_D > expr_E ∧
  expr_D > expr_F :=
by sorry

end largest_expression_l3089_308978


namespace day2_sale_is_1043_l3089_308904

/-- Represents the sales data for a grocer over 5 days -/
structure SalesData where
  average : ℕ
  day1 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ

/-- Calculates the sale on the second day given the sales data -/
def calculateDay2Sale (data : SalesData) : ℕ :=
  5 * data.average - (data.day1 + data.day3 + data.day4 + data.day5)

/-- Proves that the sale on the second day is 1043 given the specified sales data -/
theorem day2_sale_is_1043 (data : SalesData) 
    (h1 : data.average = 625)
    (h2 : data.day1 = 435)
    (h3 : data.day3 = 855)
    (h4 : data.day4 = 230)
    (h5 : data.day5 = 562) :
    calculateDay2Sale data = 1043 := by
  sorry

end day2_sale_is_1043_l3089_308904


namespace soccer_team_starters_l3089_308965

theorem soccer_team_starters (total_players : ℕ) (first_half_subs : ℕ) (players_not_played : ℕ) :
  total_players = 24 →
  first_half_subs = 2 →
  players_not_played = 7 →
  ∃ (starters : ℕ), starters = 11 ∧ 
    starters + first_half_subs + 2 * first_half_subs + players_not_played = total_players :=
by sorry

end soccer_team_starters_l3089_308965


namespace jack_daily_reading_rate_l3089_308932

-- Define the number of books Jack reads in a year
def books_per_year : ℕ := 3285

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- State the theorem
theorem jack_daily_reading_rate :
  books_per_year / days_per_year = 9 := by
  sorry

end jack_daily_reading_rate_l3089_308932


namespace brothers_money_l3089_308993

theorem brothers_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by sorry

end brothers_money_l3089_308993


namespace factorial_15_value_l3089_308998

theorem factorial_15_value : Nat.factorial 15 = 1307674368000 := by
  sorry

#eval Nat.factorial 15

end factorial_15_value_l3089_308998


namespace nearest_integer_to_sum_l3089_308955

theorem nearest_integer_to_sum : ∃ (n : ℤ), n = 3 ∧ 
  ∀ (m : ℤ), abs (m - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) ≥ 
              abs (n - (2007 / 2999 + 8001 / 5998 + 2001 / 3999 : ℚ)) := by
  sorry

end nearest_integer_to_sum_l3089_308955


namespace doctor_assignment_theorem_l3089_308974

/-- Represents the number of doctors -/
def num_doctors : ℕ := 4

/-- Represents the number of companies -/
def num_companies : ℕ := 3

/-- Calculates the total number of valid assignment schemes -/
def total_assignments : ℕ := sorry

/-- Calculates the number of assignments when one doctor is fixed to a company -/
def fixed_doctor_assignments : ℕ := sorry

/-- Calculates the number of assignments when two doctors cannot be in the same company -/
def separated_doctors_assignments : ℕ := sorry

theorem doctor_assignment_theorem :
  (total_assignments = 36) ∧
  (fixed_doctor_assignments = 12) ∧
  (separated_doctors_assignments = 30) := by sorry

end doctor_assignment_theorem_l3089_308974


namespace division_problem_l3089_308911

theorem division_problem (x : ℝ) (h : 82.04 / x = 28) : x = 2.93 := by
  sorry

end division_problem_l3089_308911


namespace seashells_to_glass_ratio_l3089_308915

/-- Represents the number of treasures Simon collected -/
structure Treasures where
  sandDollars : ℕ
  glasspieces : ℕ
  seashells : ℕ

/-- The conditions of Simon's treasure collection -/
def simonsTreasures : Treasures where
  sandDollars := 10
  glasspieces := 3 * 10
  seashells := 3 * 10

/-- The total number of treasures Simon collected -/
def totalTreasures : ℕ := 190

/-- Theorem stating that the ratio of seashells to glass pieces is 1:1 -/
theorem seashells_to_glass_ratio (t : Treasures) 
  (h1 : t.sandDollars = simonsTreasures.sandDollars)
  (h2 : t.glasspieces = 3 * t.sandDollars)
  (h3 : t.seashells = t.glasspieces)
  (h4 : t.sandDollars + t.glasspieces + t.seashells = totalTreasures) :
  t.seashells = t.glasspieces := by
  sorry

#check seashells_to_glass_ratio

end seashells_to_glass_ratio_l3089_308915


namespace fraction_equality_l3089_308930

theorem fraction_equality (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 8 * y) / (x - 2 * y) = 29 := by
  sorry

end fraction_equality_l3089_308930


namespace nonagon_perimeter_l3089_308902

theorem nonagon_perimeter : 
  let side_lengths : List ℕ := [2, 2, 3, 3, 1, 3, 2, 2, 2]
  List.sum side_lengths = 20 := by sorry

end nonagon_perimeter_l3089_308902


namespace hyperbola_center_l3089_308950

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 8^2 - (5 * y + 10)^2 / 3^2 = 1

-- Theorem stating that the center of the hyperbola is at (2, -2)
theorem hyperbola_center :
  ∃ (h k : ℝ), h = 2 ∧ k = -2 ∧
  (∀ (x y : ℝ), hyperbola_equation x y ↔ hyperbola_equation (x - h) (y - k)) :=
sorry

end hyperbola_center_l3089_308950


namespace inner_rectangle_length_l3089_308919

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the floor layout with three regions -/
structure FloorLayout where
  inner : Region
  middle : Region
  outer : Region

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem inner_rectangle_length (layout : FloorLayout) : 
  layout.inner.width = 2 →
  layout.middle.length = layout.inner.length + 2 →
  layout.middle.width = layout.inner.width + 2 →
  layout.outer.length = layout.middle.length + 2 →
  layout.outer.width = layout.middle.width + 2 →
  isArithmeticProgression (area layout.inner) (area layout.middle) (area layout.outer) →
  layout.inner.length = 8 := by
  sorry

#check inner_rectangle_length

end inner_rectangle_length_l3089_308919


namespace circle_radius_from_longest_chord_l3089_308924

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ 2 * r = c) → c / 2 = 7 :=
by sorry

end circle_radius_from_longest_chord_l3089_308924


namespace equation_solution_l3089_308987

theorem equation_solution : ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end equation_solution_l3089_308987


namespace subtraction_problem_l3089_308954

theorem subtraction_problem : ∃ x : ℕ, x - 56 = 11 ∧ x = 67 := by sorry

end subtraction_problem_l3089_308954


namespace senior_class_size_l3089_308920

theorem senior_class_size (total : ℕ) 
  (h1 : total / 5 = total / 5)  -- A fifth of the senior class is in the marching band
  (h2 : (total / 5) / 2 = (total / 5) / 2)  -- Half of the marching band plays brass instruments
  (h3 : ((total / 5) / 2) / 5 = ((total / 5) / 2) / 5)  -- A fifth of the brass instrument players play saxophone
  (h4 : (((total / 5) / 2) / 5) / 3 = (((total / 5) / 2) / 5) / 3)  -- A third of the saxophone players play alto saxophone
  (h5 : (((total / 5) / 2) / 5) / 3 = 4)  -- 4 students play alto saxophone
  : total = 600 := by
  sorry

end senior_class_size_l3089_308920


namespace no_real_graph_l3089_308913

/-- The equation x^2 + y^2 + 2x + 4y + 6 = 0 does not represent any real graph in the xy-plane. -/
theorem no_real_graph : ¬∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := by
  sorry

end no_real_graph_l3089_308913


namespace sum_divisible_by_17_l3089_308948

theorem sum_divisible_by_17 : 
  ∃ k : ℤ, 82 + 83 + 84 + 85 + 86 + 87 + 88 + 89 = 17 * k := by
  sorry

end sum_divisible_by_17_l3089_308948


namespace distance_between_points_l3089_308999

theorem distance_between_points (max_speed : ℝ) (total_time : ℝ) (stream_speed_ab : ℝ) (stream_speed_ba : ℝ) (speed_percentage_ab : ℝ) (speed_percentage_ba : ℝ) (D : ℝ) :
  max_speed = 5 →
  total_time = 5 →
  stream_speed_ab = 1 →
  stream_speed_ba = 2 →
  speed_percentage_ab = 0.9 →
  speed_percentage_ba = 0.8 →
  D / (speed_percentage_ab * max_speed + stream_speed_ab) + D / (speed_percentage_ba * max_speed - stream_speed_ba) = total_time →
  26 * D = 110 := by
sorry

end distance_between_points_l3089_308999


namespace unique_birth_year_exists_l3089_308942

def sumOfDigits (year : Nat) : Nat :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

theorem unique_birth_year_exists : 
  ∃! year : Nat, 1900 ≤ year ∧ year < 2003 ∧ 2003 - year = sumOfDigits year := by
  sorry

end unique_birth_year_exists_l3089_308942


namespace average_after_removing_numbers_l3089_308995

theorem average_after_removing_numbers (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) :
  n = 50 →
  initial_avg = 38 →
  removed1 = 45 →
  removed2 = 55 →
  (n : ℚ) * initial_avg - (removed1 + removed2) = ((n - 2) : ℚ) * 37.5 :=
by sorry

end average_after_removing_numbers_l3089_308995


namespace percent_relation_l3089_308951

theorem percent_relation (x y z w v : ℝ) 
  (hx : x = 1.3 * y) 
  (hy : y = 0.6 * z) 
  (hw : w = 1.25 * x) 
  (hv : v = 0.85 * w) : 
  v = 0.82875 * z := by
sorry

end percent_relation_l3089_308951


namespace events_mutually_exclusive_but_not_opposite_l3089_308981

/-- A type representing the balls -/
inductive Ball : Type
| one
| two
| three
| four

/-- A type representing the boxes -/
inductive Box : Type
| one
| two
| three
| four

/-- A function representing the placement of balls into boxes -/
def Placement := Ball → Box

/-- The event "ball number 1 is placed into box number 1" -/
def event1 (p : Placement) : Prop := p Ball.one = Box.one

/-- The event "ball number 1 is placed into box number 2" -/
def event2 (p : Placement) : Prop := p Ball.one = Box.two

/-- The sample space of all possible placements -/
def Ω : Set Placement := {p | ∀ b : Box, ∃! ball : Ball, p ball = b}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ p ∈ Ω, ¬(event1 p ∧ event2 p)) ∧
  ¬(∀ p ∈ Ω, event1 p ↔ ¬event2 p) :=
sorry

end events_mutually_exclusive_but_not_opposite_l3089_308981


namespace min_distinct_terms_scalene_triangle_l3089_308939

/-- Represents a scalene triangle with side lengths and angles -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ
  angle_sum : α + β + γ = π
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ
  law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ

/-- The minimum number of distinct terms in the 6-tuple (a,b,c,α,β,γ) for a scalene triangle is 4 -/
theorem min_distinct_terms_scalene_triangle (t : ScaleneTriangle) :
  ∃ (s : Finset ℝ), s.card = 4 ∧ {t.a, t.b, t.c, t.α, t.β, t.γ} ⊆ s :=
sorry

end min_distinct_terms_scalene_triangle_l3089_308939


namespace train_passing_time_l3089_308960

/-- Prove that a train with given speed and platform crossing time will take 16 seconds to pass a stationary point -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 280 →
  platform_crossing_time = 30 →
  (train_speed_kmph * 1000 / 3600) * ((platform_length + train_speed_kmph * 1000 / 3600 * platform_crossing_time) / (train_speed_kmph * 1000 / 3600)) = 16 := by
  sorry

end train_passing_time_l3089_308960
