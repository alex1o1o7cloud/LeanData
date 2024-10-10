import Mathlib

namespace absolute_value_equation_product_l585_58523

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (2 * x₁) + 4 = 38) ∧ 
   (abs (2 * x₂) + 4 = 38) ∧ 
   x₁ * x₂ = -289) := by
sorry

end absolute_value_equation_product_l585_58523


namespace luxury_to_suv_ratio_l585_58511

/-- Represents the number of cars of each type -/
structure CarInventory where
  economy : ℕ
  luxury : ℕ
  suv : ℕ

/-- The ratio of economy cars to luxury cars is 3:2 -/
def economy_to_luxury_ratio (inventory : CarInventory) : Prop :=
  3 * inventory.luxury = 2 * inventory.economy

/-- The ratio of economy cars to SUVs is 4:1 -/
def economy_to_suv_ratio (inventory : CarInventory) : Prop :=
  4 * inventory.suv = inventory.economy

/-- The theorem stating the ratio of luxury cars to SUVs -/
theorem luxury_to_suv_ratio (inventory : CarInventory) 
  (h1 : economy_to_luxury_ratio inventory) 
  (h2 : economy_to_suv_ratio inventory) : 
  8 * inventory.suv = 3 * inventory.luxury := by
  sorry

#check luxury_to_suv_ratio

end luxury_to_suv_ratio_l585_58511


namespace periodic_decimal_is_rational_l585_58589

/-- A real number with a periodic decimal expansion can be expressed as a rational number. -/
theorem periodic_decimal_is_rational (x : ℝ) (d : ℕ) (k : ℕ) (a b : ℕ) 
  (h1 : x = (a : ℝ) / 10^k + (b : ℝ) / (10^k * (10^d - 1)))
  (h2 : b < 10^d) :
  ∃ (p q : ℤ), x = (p : ℝ) / (q : ℝ) ∧ q ≠ 0 :=
sorry

end periodic_decimal_is_rational_l585_58589


namespace max_cut_length_30x30_225parts_l585_58554

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a division of the board -/
structure Division :=
  (num_parts : ℕ)
  (equal_area : Bool)

/-- Calculates the maximum possible total length of cuts for a given board and division -/
def max_cut_length (b : Board) (d : Division) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_cut_length_30x30_225parts (b : Board) (d : Division) :
  b.size = 30 ∧ d.num_parts = 225 ∧ d.equal_area = true →
  max_cut_length b d = 1065 :=
sorry

end max_cut_length_30x30_225parts_l585_58554


namespace f_range_l585_58542

def closest_multiple (k : ℤ) (n : ℤ) : ℤ :=
  n * round (k / n)

def f (k : ℤ) : ℤ :=
  closest_multiple k 3 + closest_multiple (2*k) 5 + closest_multiple (3*k) 7 - 6*k

theorem f_range :
  (∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6) ∧
  (∀ m : ℤ, -6 ≤ m ∧ m ≤ 6 → ∃ k : ℤ, f k = m) :=
sorry

end f_range_l585_58542


namespace solve_for_b_l585_58526

-- Define the @ operation
def at_op (k : ℕ) (j : ℕ) : ℕ := (List.range j).foldl (λ acc i => acc * (k + i)) k

-- Define the problem parameters
def a : ℕ := 2020
def q : ℚ := 1/2

-- Theorem statement
theorem solve_for_b (b : ℕ) (h : (a : ℚ) / b = q) : b = 4040 := by
  sorry

end solve_for_b_l585_58526


namespace jersey_cost_l585_58510

theorem jersey_cost (initial_amount : ℕ) (num_jerseys : ℕ) (basketball_cost : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 50 ∧
  num_jerseys = 5 ∧
  basketball_cost = 18 ∧
  shorts_cost = 8 ∧
  remaining_amount = 14 →
  ∃ (jersey_cost : ℕ), jersey_cost = 2 ∧ initial_amount = num_jerseys * jersey_cost + basketball_cost + shorts_cost + remaining_amount :=
by sorry

end jersey_cost_l585_58510


namespace select_workers_count_l585_58593

/-- The number of ways to select two workers from a group of three for day and night shifts -/
def select_workers : ℕ :=
  let workers := 3
  let day_shift_choices := workers
  let night_shift_choices := workers - 1
  day_shift_choices * night_shift_choices

/-- Theorem: The number of ways to select two workers from a group of three for day and night shifts is 6 -/
theorem select_workers_count : select_workers = 6 := by
  sorry

end select_workers_count_l585_58593


namespace class_average_after_exclusion_l585_58558

/-- Proves that given a class of 10 students with an average mark of 80,
    if 5 students with an average mark of 70 are excluded,
    the average mark of the remaining students is 90. -/
theorem class_average_after_exclusion
  (total_students : ℕ)
  (total_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 10)
  (h2 : total_average = 80)
  (h3 : excluded_students = 5)
  (h4 : excluded_average = 70) :
  let remaining_students := total_students - excluded_students
  let total_marks := total_students * total_average
  let excluded_marks := excluded_students * excluded_average
  let remaining_marks := total_marks - excluded_marks
  remaining_marks / remaining_students = 90 := by
  sorry


end class_average_after_exclusion_l585_58558


namespace basketball_weight_prove_basketball_weight_l585_58500

theorem basketball_weight : ℝ → ℝ → ℝ → Prop :=
  fun basketball_weight tricycle_weight motorbike_weight =>
    (9 * basketball_weight = 6 * tricycle_weight) ∧
    (6 * tricycle_weight = 4 * motorbike_weight) ∧
    (2 * motorbike_weight = 144) →
    basketball_weight = 32

-- Proof
theorem prove_basketball_weight :
  ∃ (b t m : ℝ), basketball_weight b t m :=
by
  sorry

end basketball_weight_prove_basketball_weight_l585_58500


namespace sofia_survey_l585_58536

theorem sofia_survey (liked : ℕ) (disliked : ℕ) (h1 : liked = 235) (h2 : disliked = 165) :
  liked + disliked = 400 := by
  sorry

end sofia_survey_l585_58536


namespace harmonic_geometric_sequence_ratio_l585_58546

theorem harmonic_geometric_sequence_ratio (x y z : ℝ) :
  (1 / y - 1 / x) / (1 / x - 1 / z) = 1 →  -- harmonic sequence condition
  (5 * y * z) / (3 * x * y) = (7 * z * x) / (5 * y * z) →  -- geometric sequence condition
  y / z + z / y = 58 / 21 := by sorry

end harmonic_geometric_sequence_ratio_l585_58546


namespace triangle_construction_exists_l585_58518

-- Define the necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

def Line (p q : Point) : Set Point :=
  {r : Point | ∃ t : ℝ, r = Point.mk (p.x + t * (q.x - p.x)) (p.y + t * (q.y - p.y))}

def CircumscribedCircle (a b c : Point) : Set Point :=
  sorry -- Definition of circumscribed circle

def Diameter (circle : Set Point) (p q : Point) : Prop :=
  sorry -- Definition of diameter in a circle

def FirstPicturePlane : Set Point :=
  sorry -- Definition of the first picture plane

-- State the theorem
theorem triangle_construction_exists (a b d : Point) (α : ℝ) 
  (h1 : d ∈ Line a b) : 
  ∃ c : Point, 
    c ∈ FirstPicturePlane ∧ 
    d ∈ Line a b ∧ 
    Diameter (CircumscribedCircle a b c) c d := by
  sorry

end triangle_construction_exists_l585_58518


namespace bella_current_beads_l585_58590

/-- The number of friends Bella is making bracelets for -/
def num_friends : ℕ := 6

/-- The number of beads needed per bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of additional beads Bella needs -/
def additional_beads_needed : ℕ := 12

/-- The total number of beads Bella needs for all bracelets -/
def total_beads_needed : ℕ := num_friends * beads_per_bracelet

/-- Theorem: Bella currently has 36 beads -/
theorem bella_current_beads : 
  total_beads_needed - additional_beads_needed = 36 := by
  sorry

end bella_current_beads_l585_58590


namespace five_dice_not_same_probability_l585_58588

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 6-sided dice won't all show the same number -/
theorem five_dice_not_same_probability :
  (1 - (num_sides : ℚ) / (num_sides ^ num_dice)) = 1295 / 1296 := by
  sorry

end five_dice_not_same_probability_l585_58588


namespace mismatched_pens_probability_l585_58594

def num_pens : ℕ := 3

def total_arrangements : ℕ := 6

def mismatched_arrangements : ℕ := 3

theorem mismatched_pens_probability :
  (mismatched_arrangements : ℚ) / total_arrangements = 1 / 2 := by sorry

end mismatched_pens_probability_l585_58594


namespace cake_area_theorem_l585_58515

/-- Represents the dimensions of a piece of cake -/
structure PieceDimensions where
  length : ℝ
  width : ℝ

/-- Represents a cake -/
structure Cake where
  pieces : ℕ
  pieceDimensions : PieceDimensions

/-- Calculates the total area of a cake -/
def cakeArea (c : Cake) : ℝ :=
  c.pieces * (c.pieceDimensions.length * c.pieceDimensions.width)

theorem cake_area_theorem (c : Cake) 
  (h1 : c.pieces = 25)
  (h2 : c.pieceDimensions.length = 4)
  (h3 : c.pieceDimensions.width = 4) :
  cakeArea c = 400 := by
  sorry

end cake_area_theorem_l585_58515


namespace right_rectangular_prism_volume_l585_58560

theorem right_rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 24)
  (h_front : front_area = 18)
  (h_bottom : bottom_area = 12) :
  ∃ a b c : ℝ,
    a * b = side_area ∧
    b * c = front_area ∧
    c * a = bottom_area ∧
    a * b * c = 72 :=
by sorry

end right_rectangular_prism_volume_l585_58560


namespace monic_quartic_with_given_roots_l585_58528

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 17*x^2 + 18*x - 12

-- Theorem statement
theorem monic_quartic_with_given_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 17*x^2 + 18*x + (-12)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3+√5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2-√7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry

end monic_quartic_with_given_roots_l585_58528


namespace smallest_prime_factor_of_four_consecutive_integers_sum_l585_58571

theorem smallest_prime_factor_of_four_consecutive_integers_sum (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k ∧
  ∀ (p : ℕ), p < 2 → ¬(Prime p ∧ ∃ (m : ℤ), (n - 1) + n + (n + 1) + (n + 2) = p * m) :=
by sorry

end smallest_prime_factor_of_four_consecutive_integers_sum_l585_58571


namespace tennis_tournament_l585_58505

theorem tennis_tournament (n : ℕ) : n > 0 → (
  let total_players := 4 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * n * (3 * n)
  let men_wins := 3 * n * n
  women_wins + men_wins = total_matches ∧
  3 * men_wins = 2 * women_wins
) → n = 4 := by sorry

end tennis_tournament_l585_58505


namespace f_decreasing_interval_l585_58508

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x - 1)

-- State the theorem
theorem f_decreasing_interval :
  ∀ x y : ℝ, x > 0 → y > 0 → 
  (Real.log (x + y) = Real.log x + Real.log y) →
  (∀ a b : ℝ, a > 1 → b > 1 → a < b → f a > f b) :=
by sorry

end f_decreasing_interval_l585_58508


namespace division_problem_l585_58522

theorem division_problem (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end division_problem_l585_58522


namespace complex_number_quadrant_l585_58577

theorem complex_number_quadrant : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_quadrant_l585_58577


namespace f_properties_l585_58501

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (x + 2) * Real.exp (-x) - 2
  else (x - 2) * Real.exp x + 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x ≤ 0, f x = (x + 2) * Real.exp (-x) - 2) →
  (∀ x > 0, f x = (x - 2) * Real.exp x + 2) ∧
  (∀ m : ℝ, (∃ x ∈ Set.Icc 0 2, f x = m) ↔ m ∈ Set.Icc (2 - Real.exp 1) 2) :=
by sorry

end f_properties_l585_58501


namespace maximal_k_for_triangle_l585_58533

theorem maximal_k_for_triangle : ∃ (k : ℝ), k = 5 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → k * a * b * c > a^3 + b^3 + c^3 → 
    a + b > c ∧ b + c > a ∧ c + a > b) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ k' * a * b * c > a^3 + b^3 + c^3 ∧
      (a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b)) :=
sorry

end maximal_k_for_triangle_l585_58533


namespace tangent_line_at_one_two_l585_58568

/-- The equation of the tangent line to y = -x^3 + 3x^2 at (1, 2) is y = 3x - 1 -/
theorem tangent_line_at_one_two (x : ℝ) :
  let f (x : ℝ) := -x^3 + 3*x^2
  let tangent_line (x : ℝ) := 3*x - 1
  f 1 = 2 ∧ 
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) ≠ tangent_line x - tangent_line 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(f x - f 1) / (x - 1) - (tangent_line x - tangent_line 1) / (x - 1)| < ε) :=
by sorry

end tangent_line_at_one_two_l585_58568


namespace right_triangle_inscribed_in_equilateral_l585_58569

theorem right_triangle_inscribed_in_equilateral (XC BX CZ : ℝ) :
  XC = 4 →
  BX = 3 →
  CZ = 3 →
  let XZ := XC + CZ
  let XY := XZ
  let YZ := XZ
  let BC := Real.sqrt (BX^2 + XC^2 - 2 * BX * XC * Real.cos (π/3))
  let AB := Real.sqrt (BX^2 + BC^2)
  let AZ := Real.sqrt (CZ^2 + BC^2)
  AB^2 = BC^2 + AZ^2 →
  AZ = 3 := by sorry

end right_triangle_inscribed_in_equilateral_l585_58569


namespace total_eggs_count_l585_58564

/-- The number of Easter eggs found at the club house -/
def club_house_eggs : ℕ := 60

/-- The number of Easter eggs found at the park -/
def park_eggs : ℕ := 40

/-- The number of Easter eggs found at the town hall -/
def town_hall_eggs : ℕ := 30

/-- The number of Easter eggs found at the local library -/
def local_library_eggs : ℕ := 50

/-- The number of Easter eggs found at the community center -/
def community_center_eggs : ℕ := 35

/-- The total number of Easter eggs found that day -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs + local_library_eggs + community_center_eggs

theorem total_eggs_count : total_eggs = 215 := by
  sorry

end total_eggs_count_l585_58564


namespace revenue_is_78_l585_58503

/-- The revenue per t-shirt for a shop selling t-shirts during two games -/
def revenue_per_tshirt (total_tshirts : ℕ) (first_game_tshirts : ℕ) (second_game_revenue : ℕ) : ℚ :=
  second_game_revenue / (total_tshirts - first_game_tshirts)

/-- Theorem stating that the revenue per t-shirt is $78 given the specified conditions -/
theorem revenue_is_78 :
  revenue_per_tshirt 186 172 1092 = 78 := by
  sorry

#eval revenue_per_tshirt 186 172 1092

end revenue_is_78_l585_58503


namespace largest_unreachable_sum_eighty_eight_unreachable_l585_58521

theorem largest_unreachable_sum : ∀ n : ℕ, n > 88 →
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 8 * a + 11 * b = n :=
by sorry

theorem eighty_eight_unreachable : ¬∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 8 * a + 11 * b = 88 :=
by sorry

end largest_unreachable_sum_eighty_eight_unreachable_l585_58521


namespace original_equals_scientific_l585_58547

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 140000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.4
    exponent := 8
    coeff_range := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l585_58547


namespace function_identity_l585_58527

theorem function_identity (f g h : ℕ → ℕ) 
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end function_identity_l585_58527


namespace hippopotamus_crayons_l585_58548

theorem hippopotamus_crayons (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 62)
  (h2 : remaining_crayons = 10) :
  initial_crayons - remaining_crayons = 52 := by
  sorry

end hippopotamus_crayons_l585_58548


namespace marie_erasers_l585_58563

/-- The number of erasers Marie loses -/
def erasers_lost : ℕ := 42

/-- The number of erasers Marie ends up with -/
def erasers_left : ℕ := 53

/-- The initial number of erasers Marie had -/
def initial_erasers : ℕ := erasers_left + erasers_lost

theorem marie_erasers : initial_erasers = 95 := by sorry

end marie_erasers_l585_58563


namespace smallest_lcm_with_gcd_5_l585_58596

theorem smallest_lcm_with_gcd_5 (m n : ℕ) : 
  1000 ≤ m ∧ m < 10000 ∧ 
  1000 ≤ n ∧ n < 10000 ∧ 
  Nat.gcd m n = 5 →
  201000 ≤ Nat.lcm m n ∧ 
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 
               1000 ≤ b ∧ b < 10000 ∧ 
               Nat.gcd a b = 5 ∧ 
               Nat.lcm a b = 201000 :=
by sorry

end smallest_lcm_with_gcd_5_l585_58596


namespace characterization_of_solution_l585_58543

/-- A real-valued function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be of the form ax^2 + bx -/
theorem characterization_of_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end characterization_of_solution_l585_58543


namespace parallel_tangents_imply_a_value_l585_58538

/-- Given two curves C₁ and C₂, where C₁ is defined by y = ax³ - 6x² + 12x and C₂ is defined by y = e^x,
    if their tangent lines at x = 1 are parallel, then a = e/3. -/
theorem parallel_tangents_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (3 * a * x^2 - 12 * x + 12) = Real.exp x) → a = Real.exp 1 / 3 := by
  sorry

end parallel_tangents_imply_a_value_l585_58538


namespace interest_calculation_l585_58551

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Interest earned calculation --/
def interest_earned (total : ℝ) (principal : ℝ) : ℝ :=
  total - principal

theorem interest_calculation (P : ℝ) (h1 : P > 0) :
  let rate : ℝ := 0.08
  let time : ℕ := 2
  let total : ℝ := 19828.80
  compound_interest P rate time = total →
  interest_earned total P = 2828.80 := by
sorry


end interest_calculation_l585_58551


namespace parabola_abc_value_l585_58529

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_abc_value (a b c : ℝ) :
  -- Vertex condition
  (∀ x, parabola a b c x = a * (x - 4)^2 + 2) →
  -- Point (2, 0) lies on the parabola
  parabola a b c 2 = 0 →
  -- Conclusion: abc = 12
  a * b * c = 12 := by
  sorry

end parabola_abc_value_l585_58529


namespace scientific_notation_equivalence_l585_58513

theorem scientific_notation_equivalence : 
  56000000 = 5.6 * (10 ^ 7) := by sorry

end scientific_notation_equivalence_l585_58513


namespace f_2014_equals_zero_l585_58531

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The given property of f
def f_property (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 4) = f x + f 2

-- Theorem statement
theorem f_2014_equals_zero 
  (h_even : even_function f) 
  (h_prop : f_property f) : 
  f 2014 = 0 := by sorry

end f_2014_equals_zero_l585_58531


namespace jerseys_sold_equals_tshirts_sold_l585_58586

theorem jerseys_sold_equals_tshirts_sold (jersey_profit : ℕ) (tshirt_profit : ℕ) 
  (tshirts_sold : ℕ) (jersey_cost_difference : ℕ) :
  jersey_profit = 115 →
  tshirt_profit = 25 →
  tshirts_sold = 113 →
  jersey_cost_difference = 90 →
  jersey_profit = tshirt_profit + jersey_cost_difference →
  ∃ (jerseys_sold : ℕ), jerseys_sold = tshirts_sold :=
by sorry


end jerseys_sold_equals_tshirts_sold_l585_58586


namespace quadratic_always_positive_l585_58576

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 1 > 0 := by
  sorry

end quadratic_always_positive_l585_58576


namespace distance_CX_l585_58592

/-- Given five points A, B, C, D, X on a plane with specific distances between them,
    prove that the distance between C and X is 3. -/
theorem distance_CX (A B C D X : EuclideanSpace ℝ (Fin 2)) 
  (h1 : dist A C = 2)
  (h2 : dist A X = 5)
  (h3 : dist A D = 11)
  (h4 : dist C D = 9)
  (h5 : dist C B = 10)
  (h6 : dist D B = 1)
  (h7 : dist X B = 7) :
  dist C X = 3 := by
  sorry


end distance_CX_l585_58592


namespace multiple_implies_equal_l585_58534

theorem multiple_implies_equal (a b : ℕ+) (h : ∃ k : ℕ, (a^2 + a*b + 1 : ℕ) = k * (b^2 + a*b + 1)) : a = b := by
  sorry

end multiple_implies_equal_l585_58534


namespace prime_sum_2003_l585_58504

theorem prime_sum_2003 (a b : ℕ) (ha : Prime a) (hb : Prime b) (heq : a^2 + b = 2003) :
  a + b = 2001 := by
  sorry

end prime_sum_2003_l585_58504


namespace exists_all_intersecting_segment_l585_58573

/-- A segment on a line -/
structure Segment where
  left : ℝ
  right : ℝ
  h : left < right

/-- A configuration of segments on a line -/
structure SegmentConfiguration where
  n : ℕ
  segments : Finset Segment
  total_count : segments.card = 2 * n + 1
  intersection_condition : ∀ s ∈ segments, (segments.filter (λ t => s.left < t.right ∧ t.left < s.right)).card ≥ n

/-- There exists a segment that intersects all others -/
theorem exists_all_intersecting_segment (config : SegmentConfiguration) :
  ∃ s ∈ config.segments, ∀ t ∈ config.segments, t ≠ s → s.left < t.right ∧ t.left < s.right :=
sorry

end exists_all_intersecting_segment_l585_58573


namespace remainder_of_p_l585_58550

-- Define the polynomial p(x)
def p (x : ℝ) (r : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (x + 1) * (x - 2)^2 * r x + a * x + b

-- State the theorem
theorem remainder_of_p (r : ℝ → ℝ) (a b : ℝ) :
  (p 2 r a b = 6) →
  (p (-1) r a b = 0) →
  ∃ q : ℝ → ℝ, ∀ x, p x r a b = (x + 1) * (x - 2)^2 * q x + 2 * x + 2 :=
by sorry

end remainder_of_p_l585_58550


namespace hyperbola_asymptote_b_value_l585_58595

-- Define the hyperbola equation
def is_hyperbola (x y b : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define the asymptote equation
def is_asymptote (x y : ℝ) : Prop := y = 2*x

theorem hyperbola_asymptote_b_value (b : ℝ) :
  b > 0 →
  (∃ x y : ℝ, is_hyperbola x y b ∧ is_asymptote x y) →
  b = 2 :=
by sorry

end hyperbola_asymptote_b_value_l585_58595


namespace sum_complex_exp_argument_l585_58597

/-- The sum of five complex exponentials has an argument of 59π/120 -/
theorem sum_complex_exp_argument :
  let z₁ := Complex.exp (11 * Real.pi * Complex.I / 120)
  let z₂ := Complex.exp (31 * Real.pi * Complex.I / 120)
  let z₃ := Complex.exp (-13 * Real.pi * Complex.I / 120)
  let z₄ := Complex.exp (-53 * Real.pi * Complex.I / 120)
  let z₅ := Complex.exp (-73 * Real.pi * Complex.I / 120)
  let sum := z₁ + z₂ + z₃ + z₄ + z₅
  ∃ (r : ℝ), sum = r * Complex.exp (59 * Real.pi * Complex.I / 120) ∧ r > 0 :=
by sorry

end sum_complex_exp_argument_l585_58597


namespace polynomial_expansion_simplification_l585_58514

theorem polynomial_expansion_simplification (x : ℝ) : 
  (x^3 - 3*x^2 + (1/2)*x - 1) * (x^2 + 3*x + 3/2) = 
  x^5 - (15/2)*x^3 - 4*x^2 - (9/4)*x - 3/2 := by sorry

end polynomial_expansion_simplification_l585_58514


namespace find_m_l585_58517

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end find_m_l585_58517


namespace square_of_number_l585_58572

theorem square_of_number (x : ℝ) : 2 * x = x / 5 + 9 → x^2 = 25 := by
  sorry

end square_of_number_l585_58572


namespace complex_equation_solution_l585_58598

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) - 2 * i * z = -4 + 5 * i * z ∧ z = -i :=
by sorry

end complex_equation_solution_l585_58598


namespace point_on_line_l585_58583

/-- Given two points on a line and a third point with a known y-coordinate,
    prove that the x-coordinate of the third point is -6. -/
theorem point_on_line (x : ℝ) :
  let p1 : ℝ × ℝ := (0, 8)
  let p2 : ℝ × ℝ := (-4, 0)
  let p3 : ℝ × ℝ := (x, -4)
  (p3.2 - p1.2) / (p3.1 - p1.1) = (p2.2 - p1.2) / (p2.1 - p1.1) →
  x = -6 := by
sorry

end point_on_line_l585_58583


namespace polygon_square_equal_area_l585_58566

/-- Given a polygon with perimeter 800 cm and each side tangent to a circle of radius 100 cm,
    the side length of a square with equal area is 200 cm. -/
theorem polygon_square_equal_area (polygon_perimeter : ℝ) (circle_radius : ℝ) :
  polygon_perimeter = 800 ∧ circle_radius = 100 →
  ∃ (square_side : ℝ),
    square_side = 200 ∧
    square_side ^ 2 = (polygon_perimeter * circle_radius) / 2 := by
  sorry

end polygon_square_equal_area_l585_58566


namespace complex_exp_13pi_over_2_l585_58506

theorem complex_exp_13pi_over_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end complex_exp_13pi_over_2_l585_58506


namespace angle_sum_around_point_l585_58578

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 210 > 0 ∧ x + x + 210 = 360 → x = 75 := by
  sorry

end angle_sum_around_point_l585_58578


namespace ball_probability_theorem_l585_58540

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing k balls of a specific color from a bag -/
def prob_draw (bag : Bag) (color : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose color k : ℚ) / (Nat.choose (bag.white + bag.black) k)

/-- The probability of drawing all black balls from both bags -/
def prob_all_black (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.black 2)

/-- The probability of drawing exactly one white ball from both bags -/
def prob_one_white (bagA bagB : Bag) : ℚ :=
  (prob_draw bagA bagA.black 2) * (prob_draw bagB bagB.white 1) * (prob_draw bagB bagB.black 1) +
  (prob_draw bagA bagA.white 1) * (prob_draw bagA bagA.black 1) * (prob_draw bagB bagB.black 2)

theorem ball_probability_theorem (bagA bagB : Bag) 
  (hA : bagA = ⟨2, 4⟩) (hB : bagB = ⟨1, 4⟩) : 
  prob_all_black bagA bagB = 6/25 ∧ prob_one_white bagA bagB = 12/25 := by
  sorry

end ball_probability_theorem_l585_58540


namespace geometric_sequence_sum_l585_58541

/-- Given a geometric sequence {a_n} satisfying the condition
    a_4 · a_6 + 2a_5 · a_7 + a_6 · a_8 = 36, prove that a_5 + a_7 = ±6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_condition : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  (a 5 + a 7 = 6) ∨ (a 5 + a 7 = -6) := by
  sorry

end geometric_sequence_sum_l585_58541


namespace square_difference_120_pairs_l585_58539

theorem square_difference_120_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > n ∧ m^2 - n^2 = 120) ∧
    pairs.card = 4 := by
  sorry

end square_difference_120_pairs_l585_58539


namespace mode_is_25_l585_58575

def sales_volumes : List ℕ := [10, 14, 25, 13]

def is_mode (x : ℕ) (list : List ℕ) : Prop :=
  ∀ y ∈ list, (list.count x ≥ list.count y)

theorem mode_is_25 (s : ℕ) : is_mode 25 (sales_volumes ++ [s]) := by
  sorry

end mode_is_25_l585_58575


namespace triangle_area_proof_l585_58530

/-- The slope of the line -/
def m : ℚ := -1/2

/-- A point on the line -/
def p : ℝ × ℝ := (2, -3)

/-- The equation of the line in the form ax + by + c = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + 2*y + 4 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := -4

/-- The y-intercept of the line -/
def y_intercept : ℝ := -2

/-- The area of the triangle formed by the line and coordinate axes -/
def triangle_area : ℝ := 4

theorem triangle_area_proof :
  line_equation p.1 p.2 ∧
  (∀ x y : ℝ, line_equation x y → y - p.2 = m * (x - p.1)) →
  triangle_area = (1/2) * |x_intercept| * |y_intercept| :=
sorry

end triangle_area_proof_l585_58530


namespace unique_start_day_l585_58556

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- A function that determines if a given day is the first day of a 30-day month with equal Saturdays and Sundays -/
def is_valid_start_day (d : DayOfWeek) : Prop :=
  ∃ (sat_count sun_count : ℕ),
    sat_count = sun_count ∧
    sat_count + sun_count ≤ 30 ∧
    (match d with
      | DayOfWeek.Monday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Tuesday   => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Wednesday => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Thursday  => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Friday    => sat_count = 4 ∧ sun_count = 4
      | DayOfWeek.Saturday  => sat_count = 5 ∧ sun_count = 5
      | DayOfWeek.Sunday    => sat_count = 5 ∧ sun_count = 4)

/-- Theorem stating that there is exactly one day of the week that can be the first day of a 30-day month with equal Saturdays and Sundays -/
theorem unique_start_day :
  ∃! (d : DayOfWeek), is_valid_start_day d :=
sorry

end unique_start_day_l585_58556


namespace ellipse_area_irrational_l585_58544

-- Define the major and minor radii as rational numbers
variable (a b : ℚ)

-- Define π as an irrational constant
noncomputable def π : ℝ := Real.pi

-- Define the area of the ellipse
noncomputable def ellipseArea (a b : ℚ) : ℝ := π * (a * b)

-- Theorem statement
theorem ellipse_area_irrational (a b : ℚ) (h1 : a > 0) (h2 : b > 0) :
  Irrational (ellipseArea a b) := by
  sorry

end ellipse_area_irrational_l585_58544


namespace units_digit_of_composite_product_l585_58532

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_composite_product :
  (first_four_composites.prod % 10) = 8 := by
  sorry

end units_digit_of_composite_product_l585_58532


namespace curve_is_line_segment_l585_58553

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the domain
def domain (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 5

-- Theorem: The curve is a line segment
theorem curve_is_line_segment :
  ∃ (a b : ℝ), ∀ (t : ℝ), domain t →
    ∃ (k : ℝ), 0 ≤ k ∧ k ≤ 1 ∧
      x t = a + k * (b - a) ∧
      y t = (x t - 2) / 3 ∧
      -1 ≤ y t ∧ y t ≤ 24 :=
by sorry


end curve_is_line_segment_l585_58553


namespace equal_roots_condition_l585_58562

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (x * (x - 1) - (m^2 + 2*m + 1)) / ((x - 1) * (m^2 - 1) + 1) = x / m) →
  (∃! x : ℝ, x * (x - 1) - (m^2 + 2*m + 1) = 0) →
  m = -1 := by
sorry

end equal_roots_condition_l585_58562


namespace problem_statement_l585_58574

theorem problem_statement : (-1)^53 + 2^(4^4 + 3^3 - 5^2) = -1 + 2^258 := by
  sorry

end problem_statement_l585_58574


namespace expression_value_l585_58535

theorem expression_value : (4 - 2)^3 = 8 := by
  sorry

end expression_value_l585_58535


namespace correct_operations_l585_58591

theorem correct_operations (x : ℝ) : 
  (x / 9 - 20 = 8) → (x * 9 + 20 = 2288) := by
  sorry

end correct_operations_l585_58591


namespace sufficient_not_necessary_l585_58525

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x > 0 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x > 0) :=
by sorry

end sufficient_not_necessary_l585_58525


namespace solve_exponential_equation_l585_58552

theorem solve_exponential_equation :
  ∃! y : ℝ, (64 : ℝ)^(3*y) = (16 : ℝ)^(4*y - 5) :=
by
  -- The unique solution is y = -10
  use -10
  sorry

end solve_exponential_equation_l585_58552


namespace lines_parallel_l585_58559

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

theorem lines_parallel : 
  let line1 : Line := ⟨-1, 0⟩
  let line2 : Line := ⟨-1, 6⟩
  parallel line1 line2 := by
  sorry

end lines_parallel_l585_58559


namespace root_equation_value_l585_58545

theorem root_equation_value (a : ℝ) : 
  a^2 + 2*a - 2 = 0 → 3*a^2 + 6*a + 2023 = 2029 := by
  sorry

end root_equation_value_l585_58545


namespace money_division_l585_58585

/-- Represents the share of money for each person -/
structure Share :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The problem statement and proof -/
theorem money_division (s : Share) : 
  s.c = 64 ∧ 
  s.b = 0.65 * s.a ∧ 
  s.c = 0.40 * s.a → 
  s.a + s.b + s.c = 328 := by
sorry


end money_division_l585_58585


namespace prime_product_minus_sum_l585_58570

theorem prime_product_minus_sum : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ 
  p ≠ q ∧ 
  4 < p ∧ p < 18 ∧ 
  4 < q ∧ q < 18 ∧ 
  p * q - (p + q) = 119 := by
sorry

end prime_product_minus_sum_l585_58570


namespace chessboard_covering_impossible_l585_58565

/-- Represents a chessboard with given dimensions -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile with given dimensions -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Determines if a chessboard can be covered by a given number of tiles -/
def can_cover (board : Chessboard) (tile : Tile) (num_tiles : Nat) : Prop :=
  ∃ (arrangement : Nat), 
    arrangement > 0 ∧ 
    tile.length * tile.width * num_tiles = board.rows * board.cols

/-- The main theorem stating that a 10x10 chessboard cannot be covered by 25 4x1 tiles -/
theorem chessboard_covering_impossible :
  ¬(can_cover (Chessboard.mk 10 10) (Tile.mk 4 1) 25) :=
sorry

end chessboard_covering_impossible_l585_58565


namespace three_digit_append_divisibility_l585_58512

theorem three_digit_append_divisibility :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (594000 + n) % 651 = 0 :=
by
  -- The proof would go here
  sorry

end three_digit_append_divisibility_l585_58512


namespace smallest_positive_number_l585_58524

theorem smallest_positive_number : 
  let a := 8 - 2 * Real.sqrt 17
  let b := 2 * Real.sqrt 17 - 8
  let c := 25 - 7 * Real.sqrt 5
  let d := 40 - 9 * Real.sqrt 2
  let e := 9 * Real.sqrt 2 - 40
  (0 < b) ∧ 
  (a ≤ b ∨ a ≤ 0) ∧ 
  (b ≤ c ∨ c ≤ 0) ∧ 
  (b ≤ d ∨ d ≤ 0) ∧ 
  (b ≤ e ∨ e ≤ 0) :=
by sorry

end smallest_positive_number_l585_58524


namespace bakery_ratio_l585_58555

/-- Given the conditions of a bakery's storage room, prove the ratio of flour to baking soda --/
theorem bakery_ratio (sugar flour baking_soda : ℕ) : 
  sugar = 6000 ∧ 
  5 * flour = 2 * sugar ∧ 
  8 * (baking_soda + 60) = flour → 
  10 * baking_soda = flour := by sorry

end bakery_ratio_l585_58555


namespace pi_approx_thousandth_l585_58581

/-- The approximation of π to the thousandth place -/
def pi_approx : ℝ := 3.142

/-- The theorem stating that the approximation of π to the thousandth place is equal to 3.142 -/
theorem pi_approx_thousandth : |π - pi_approx| < 0.0005 := by
  sorry

end pi_approx_thousandth_l585_58581


namespace inequality_proof_l585_58557

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (1/a) + (1/b) + (9/c) + (25/d) ≥ 100/(a + b + c + d) := by
  sorry

end inequality_proof_l585_58557


namespace days_passed_before_realization_l585_58519

/-- Represents the contractor's job scenario -/
structure JobScenario where
  totalDays : ℕ
  initialWorkers : ℕ
  workCompletedFraction : ℚ
  workersFired : ℕ
  remainingDays : ℕ

/-- Calculates the number of days passed before the contractor realized a fraction of work was done -/
def daysPassedBeforeRealization (scenario : JobScenario) : ℕ :=
  sorry

/-- The theorem stating that for the given scenario, 20 days passed before realization -/
theorem days_passed_before_realization :
  let scenario : JobScenario := {
    totalDays := 100,
    initialWorkers := 10,
    workCompletedFraction := 1/4,
    workersFired := 2,
    remainingDays := 75
  }
  daysPassedBeforeRealization scenario = 20 := by
  sorry

end days_passed_before_realization_l585_58519


namespace audiobook_listening_time_l585_58549

theorem audiobook_listening_time 
  (num_books : ℕ) 
  (book_length : ℕ) 
  (daily_listening : ℕ) 
  (h1 : num_books = 6) 
  (h2 : book_length = 30) 
  (h3 : daily_listening = 2) : 
  (num_books * book_length) / daily_listening = 90 := by
sorry

end audiobook_listening_time_l585_58549


namespace remainder_problem_l585_58584

theorem remainder_problem (n : ℕ) 
  (h1 : n^3 % 7 = 3) 
  (h2 : n^4 % 7 = 2) : 
  n % 7 = 6 := by
  sorry

end remainder_problem_l585_58584


namespace ab_nonzero_sufficient_not_necessary_for_a_nonzero_l585_58561

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero (a b : ℝ) :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end ab_nonzero_sufficient_not_necessary_for_a_nonzero_l585_58561


namespace unique_prime_divisor_l585_58599

theorem unique_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k ∈ Finset.range n,
    ∃ p : ℕ, Nat.Prime p ∧ 
      (p ∣ (n.factorial + k + 1)) ∧
      (∀ j ∈ Finset.range n, j ≠ k → ¬(p ∣ (n.factorial + j + 1))) :=
by sorry

end unique_prime_divisor_l585_58599


namespace correct_fraction_proof_l585_58579

theorem correct_fraction_proof (x y : ℕ) (h : x > 0 ∧ y > 0) :
  (5 : ℚ) / 6 * 480 = x / y * 480 + 250 → x / y = (5 : ℚ) / 16 := by
  sorry

end correct_fraction_proof_l585_58579


namespace derivative_ln_inverse_sqrt_plus_one_squared_l585_58537

open Real

theorem derivative_ln_inverse_sqrt_plus_one_squared (x : ℝ) :
  deriv (λ x => Real.log (1 / Real.sqrt (1 + x^2))) x = -x / (1 + x^2) := by
  sorry

end derivative_ln_inverse_sqrt_plus_one_squared_l585_58537


namespace cf_length_l585_58516

/-- A rectangle ABCD with point F such that C is on DF and B is on DE -/
structure SpecialRectangle where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
  /-- AB = 8 -/
  ab_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  /-- BC = 6 -/
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36
  /-- C is on DF -/
  c_on_df : ∃ t : ℝ, C = (1 - t) • D + t • F
  /-- B is the quarter-point of DE -/
  b_quarter_point : B = (3/4) • D + (1/4) • E
  /-- DEF is a right triangle -/
  def_right_triangle : (D.1 - E.1) * (E.1 - F.1) + (D.2 - E.2) * (E.2 - F.2) = 0

/-- The length of CF is 12 -/
theorem cf_length (rect : SpecialRectangle) : 
  (rect.C.1 - rect.F.1)^2 + (rect.C.2 - rect.F.2)^2 = 144 := by
  sorry

end cf_length_l585_58516


namespace book_arrangement_count_l585_58509

/-- Represents the number of books of each type -/
structure BookCounts where
  chinese : Nat
  english : Nat
  math : Nat

/-- Represents the arrangement constraints -/
structure ArrangementConstraints where
  chinese_adjacent : Bool
  english_adjacent : Bool
  math_not_adjacent : Bool

/-- Calculates the number of valid book arrangements -/
def count_arrangements (counts : BookCounts) (constraints : ArrangementConstraints) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  let counts : BookCounts := ⟨2, 2, 3⟩
  let constraints : ArrangementConstraints := ⟨true, true, true⟩
  count_arrangements counts constraints = 48 := by
  sorry

end book_arrangement_count_l585_58509


namespace extremum_and_minimum_l585_58567

-- Define the function f(x) = x³ - 3ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem extremum_and_minimum (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| ∧ |h| < ε → f a (-1 + h) ≤ f a (-1) ∨ f a (-1 + h) ≥ f a (-1)) →
  a = 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) → f a x ≥ -3 :=
by sorry

end extremum_and_minimum_l585_58567


namespace office_network_connections_l585_58502

/-- A network of switches with connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- The theorem stating that a network of 40 switches, each connected to 4 others, has 80 connections. -/
theorem office_network_connections :
  let network : SwitchNetwork := { num_switches := 40, connections_per_switch := 4 }
  total_connections network = 80 := by
  sorry

end office_network_connections_l585_58502


namespace f_greater_than_log_over_x_minus_one_l585_58580

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1) + 1 / x

theorem f_greater_than_log_over_x_minus_one (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  f x > Real.log x / (x - 1) := by
  sorry

end f_greater_than_log_over_x_minus_one_l585_58580


namespace dog_bird_time_difference_l585_58587

def dogs : ℕ := 3
def dog_hours : ℕ := 7
def holes : ℕ := 9
def birds : ℕ := 5
def bird_minutes : ℕ := 40
def nests : ℕ := 2

def dog_dig_time : ℚ := (dog_hours * 60 : ℚ) * holes / dogs
def bird_build_time : ℚ := (bird_minutes : ℚ) * birds / nests

theorem dog_bird_time_difference :
  dog_dig_time - bird_build_time = 40 := by sorry

end dog_bird_time_difference_l585_58587


namespace certain_amount_calculation_l585_58507

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 170) (h2 : 0.65 * x = 0.2 * A) : A = 552.5 := by
  sorry

end certain_amount_calculation_l585_58507


namespace unique_angle_D_l585_58520

/-- Represents a convex pentagon with equal sides -/
structure EqualSidedPentagon where
  -- Angles in degrees
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  angleE : ℝ
  -- Conditions
  convex : angleA > 0 ∧ angleB > 0 ∧ angleC > 0 ∧ angleD > 0 ∧ angleE > 0
  sum_of_angles : angleA + angleB + angleC + angleD + angleE = 540
  angleA_is_120 : angleA = 120
  angleC_is_135 : angleC = 135

/-- The main theorem -/
theorem unique_angle_D (p : EqualSidedPentagon) : p.angleD = 90 := by
  sorry


end unique_angle_D_l585_58520


namespace quadratic_sequence_sum_l585_58582

theorem quadratic_sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 10)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ + 81*x₈ = 40)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 170) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 400 := by
  sorry

end quadratic_sequence_sum_l585_58582
