import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_expression_l447_44714

theorem evaluate_expression : 
  2100^3 - 2 * 2099 * 2100^2 - 2099^2 * 2100 + 2099^3 = 4404902 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l447_44714


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l447_44738

theorem least_addition_for_divisibility : ∃! x : ℕ, x < 37 ∧ (1052 + x) % 37 = 0 ∧ ∀ y : ℕ, y < x → (1052 + y) % 37 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l447_44738


namespace NUMINAMATH_CALUDE_puzzle_solution_l447_44790

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_four_digit_number (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def construct_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem puzzle_solution :
  ∀ t h e a b g m,
    distinct_digits t h e a ∧
    distinct_digits b e t a ∧
    distinct_digits g a m m ∧
    is_four_digit_number (construct_number t h e a) ∧
    is_four_digit_number (construct_number b e t a) ∧
    is_four_digit_number (construct_number g a m m) ∧
    construct_number t h e a + construct_number b e t a = construct_number g a m m →
    t = 4 ∧ h = 9 ∧ e = 4 ∧ a = 0 ∧ b = 5 ∧ g = 1 ∧ m = 8 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l447_44790


namespace NUMINAMATH_CALUDE_earrings_ratio_is_two_to_one_l447_44785

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_pairs : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs should last -/
def total_days : ℕ := 42

/-- The number of pairs of earrings Kim brings on the second day -/
def second_day_pairs : ℕ := 6

theorem earrings_ratio_is_two_to_one :
  let total_gumballs := gumballs_per_pair * (first_day_pairs + second_day_pairs + (second_day_pairs - 1))
  total_gumballs = gumballs_eaten_per_day * total_days ∧
  second_day_pairs / first_day_pairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_earrings_ratio_is_two_to_one_l447_44785


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_eight_digit_repeated_l447_44734

/-- Represents an 8-digit positive integer with repeated digits -/
def EightDigitRepeated : Type := 
  {n : ℕ // 10000000 ≤ n ∧ n < 100000000 ∧ ∃ a b c d : ℕ, n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + a * 1000 + b * 100 + c * 10 + d}

/-- Theorem stating that 10001 is a factor of any EightDigitRepeated number -/
theorem ten_thousand_one_divides_eight_digit_repeated (z : EightDigitRepeated) : 
  10001 ∣ z.val := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_eight_digit_repeated_l447_44734


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l447_44708

-- Define a geometric sequence of three terms
def is_geometric_sequence (a b c : ℝ) : Prop := b * b = a * c

-- Theorem statement
theorem geometric_sequence_middle_term :
  ∀ m : ℝ, is_geometric_sequence 1 m 4 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l447_44708


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l447_44731

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  /-- The number of vertices in the original cube -/
  originalVertices : Nat
  /-- The number of edges in the original cube -/
  originalEdges : Nat
  /-- The fraction of each edge removed by truncation -/
  truncationFraction : Rat
  /-- The number of edges affected by truncation at each vertex -/
  edgesAffectedPerVertex : Nat
  /-- The number of new edges created by truncation at each vertex -/
  newEdgesPerVertex : Nat

/-- The number of edges in a truncated cube -/
def edgesInTruncatedCube (c : TruncatedCube) : Nat :=
  c.originalEdges + c.originalVertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
    c.originalVertices = 8 ∧
    c.originalEdges = 12 ∧
    c.truncationFraction = 1/4 ∧
    c.edgesAffectedPerVertex = 2 ∧
    c.newEdgesPerVertex = 3 →
    edgesInTruncatedCube c = 36 :=
by sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l447_44731


namespace NUMINAMATH_CALUDE_canteen_banana_units_l447_44716

/-- Represents the number of bananas in a unit -/
def bananas_per_unit (daily_units : ℕ) (total_bananas : ℕ) (weeks : ℕ) : ℕ :=
  (total_bananas / (weeks * 7)) / daily_units

/-- Theorem stating that given the conditions, each unit consists of 12 bananas -/
theorem canteen_banana_units :
  bananas_per_unit 13 9828 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_canteen_banana_units_l447_44716


namespace NUMINAMATH_CALUDE_powerSum7Seq_36th_l447_44784

/-- Sequence of sums of distinct powers of 7 -/
def powerSum7Seq : ℕ → ℕ
  | 0 => 1
  | n + 1 => powerSum7Seq n + 7^(n.log2)

/-- The 36th number in the sequence is 16856 -/
theorem powerSum7Seq_36th : powerSum7Seq 35 = 16856 := by
  sorry

end NUMINAMATH_CALUDE_powerSum7Seq_36th_l447_44784


namespace NUMINAMATH_CALUDE_expression_independence_l447_44756

theorem expression_independence (x a b c : ℝ) 
  (hxa : x ≠ a) (hxb : x ≠ b) (hxc : x ≠ c) : 
  (x - a) * (x - b) * (x - c) * 
  ((a - b) / (x - c) + (b - c) / (x - a) + (c - a) / (x - b)) = 
  (b - a) * (a - c) * (c - b) := by
  sorry

end NUMINAMATH_CALUDE_expression_independence_l447_44756


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l447_44766

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 16 * y = x * y) :
  x + y ≥ 25 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 16 * y₀ = x₀ * y₀ ∧ x₀ + y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l447_44766


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l447_44726

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l447_44726


namespace NUMINAMATH_CALUDE_fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l447_44786

theorem fraction_multiplication (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_half_of_one_third_of_one_sixth_of_72 :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l447_44786


namespace NUMINAMATH_CALUDE_walking_speed_problem_l447_44719

/-- Given two people walking in the same direction for 10 hours, where one walks at 7.5 kmph
    and they end up 20 km apart, prove that the speed of the other person is 9.5 kmph. -/
theorem walking_speed_problem (v : ℝ) 
  (h1 : (v - 7.5) * 10 = 20) : v = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l447_44719


namespace NUMINAMATH_CALUDE_vacation_fund_adjustment_l447_44711

/-- Calculates the required hours per week to earn a target amount given initial conditions and unexpected events --/
theorem vacation_fund_adjustment (initial_weeks : ℕ) (initial_hours_per_week : ℝ) (sick_weeks : ℕ) (target_amount : ℝ) :
  let remaining_weeks := initial_weeks - sick_weeks
  let total_hours := initial_weeks * initial_hours_per_week
  let hourly_rate := target_amount / total_hours
  let required_hours_per_week := (target_amount / hourly_rate) / remaining_weeks
  required_hours_per_week = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_vacation_fund_adjustment_l447_44711


namespace NUMINAMATH_CALUDE_rebus_solution_l447_44730

theorem rebus_solution : ∃! (a b c d : ℕ),
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (1000 * a + 100 * b + 10 * c + a = 182 * (10 * c + d)) ∧
  (a = 2 ∧ b = 9 ∧ c = 1 ∧ d = 6) :=
by sorry

end NUMINAMATH_CALUDE_rebus_solution_l447_44730


namespace NUMINAMATH_CALUDE_farm_animals_count_l447_44797

theorem farm_animals_count : 
  ∀ (total_legs ducks dogs : ℕ),
  total_legs = 24 →
  ducks = 4 →
  total_legs = 2 * ducks + 4 * dogs →
  ducks + dogs = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_count_l447_44797


namespace NUMINAMATH_CALUDE_irrational_functional_equation_implies_constant_l447_44783

/-- A function satisfying f(ab) = f(a+b) for all irrational a and b -/
def IrrationalFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, Irrational a → Irrational b → f (a * b) = f (a + b)

/-- Theorem: If a function satisfies the irrational functional equation, then it is constant -/
theorem irrational_functional_equation_implies_constant
  (f : ℝ → ℝ) (h : IrrationalFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_irrational_functional_equation_implies_constant_l447_44783


namespace NUMINAMATH_CALUDE_simplify_cube_root_l447_44763

theorem simplify_cube_root (a b : ℝ) (h : a < 0) : 
  Real.sqrt (a^3 * b) = -a * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l447_44763


namespace NUMINAMATH_CALUDE_refrigerator_price_l447_44768

/-- The price paid for a refrigerator given specific conditions --/
theorem refrigerator_price (discount_rate : ℝ) (transport_cost : ℝ) (installation_cost : ℝ)
  (profit_rate : ℝ) (selling_price : ℝ) :
  discount_rate = 0.20 →
  transport_cost = 125 →
  installation_cost = 250 →
  profit_rate = 0.16 →
  selling_price = 18560 →
  ∃ (labelled_price : ℝ),
    selling_price = labelled_price * (1 + profit_rate) ∧
    labelled_price * (1 - discount_rate) + transport_cost + installation_cost = 13175 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_price_l447_44768


namespace NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l447_44718

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

theorem intersection_theorem (m : ℝ) :
  A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 := by sorry

theorem subset_theorem (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l447_44718


namespace NUMINAMATH_CALUDE_engineers_teachers_ratio_l447_44701

theorem engineers_teachers_ratio (e t : ℕ) (he : e > 0) (ht : t > 0) :
  (40 * e + 55 * t : ℚ) / (e + t) = 46 →
  e / t = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_engineers_teachers_ratio_l447_44701


namespace NUMINAMATH_CALUDE_lindas_savings_l447_44775

theorem lindas_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 230 → savings = 920 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l447_44775


namespace NUMINAMATH_CALUDE_geometric_arithmetic_relation_l447_44737

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  a 3 * a 11 = 4 * a 7 →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_relation_l447_44737


namespace NUMINAMATH_CALUDE_expression_evaluation_l447_44761

theorem expression_evaluation (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l447_44761


namespace NUMINAMATH_CALUDE_min_r_for_B_subset_C_l447_44781

open Set Real

-- Define the sets A, B, and C(r)
def A : Set ℝ := {t | 0 < t ∧ t < 2 * π}

def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = sin t ∧ p.2 = 2 * sin t * cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

-- State the theorem
theorem min_r_for_B_subset_C : 
  (∀ r, B ⊆ C r → r ≥ 5/4) ∧ B ⊆ C (5/4) := by sorry

end NUMINAMATH_CALUDE_min_r_for_B_subset_C_l447_44781


namespace NUMINAMATH_CALUDE_beavers_still_working_l447_44794

def initial_beavers : ℕ := 7
def swimming_beavers : ℕ := 2
def stick_collecting_beaver : ℕ := 1
def food_searching_beaver : ℕ := 1

theorem beavers_still_working : ℕ := by
  sorry

end NUMINAMATH_CALUDE_beavers_still_working_l447_44794


namespace NUMINAMATH_CALUDE_function_domain_condition_l447_44744

/-- Given a function f(x) = √(kx² - 4x + 3), prove that for f to have a domain of ℝ, 
    k must be in the range [4/3, +∞). -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (k * x^2 - 4 * x + 3)) ↔ k ≥ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_condition_l447_44744


namespace NUMINAMATH_CALUDE_rikki_poetry_pricing_l447_44704

-- Define the constants
def words_per_interval : ℕ := 25
def minutes_per_interval : ℕ := 5
def total_minutes : ℕ := 120
def expected_earnings : ℚ := 6

-- Define the function to calculate the price per word
def price_per_word : ℚ :=
  let intervals : ℕ := total_minutes / minutes_per_interval
  let total_words : ℕ := words_per_interval * intervals
  expected_earnings / total_words

-- Theorem statement
theorem rikki_poetry_pricing :
  price_per_word = 1/100 := by sorry

end NUMINAMATH_CALUDE_rikki_poetry_pricing_l447_44704


namespace NUMINAMATH_CALUDE_solution_value_l447_44712

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the function representing x^2 + ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem solution_value (a b : ℝ) : 
  (∀ x, x ∈ A_intersect_B ↔ f a b x < 0) → a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_solution_value_l447_44712


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l447_44736

def M : Set Int := {-1, 1, -2, 2}
def N : Set Int := {1, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l447_44736


namespace NUMINAMATH_CALUDE_min_distance_to_line_l447_44788

/-- The minimum value of (a+1)^2 + b^2 for a point (a, b) on the line y = √3x - √3 is 3 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l447_44788


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l447_44787

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l447_44787


namespace NUMINAMATH_CALUDE_frog_eyes_count_l447_44725

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := num_frogs * eyes_per_frog

theorem frog_eyes_count : total_frog_eyes = 12 := by
  sorry

end NUMINAMATH_CALUDE_frog_eyes_count_l447_44725


namespace NUMINAMATH_CALUDE_f_min_max_l447_44755

def f (x : ℝ) : ℝ := -2 * x + 1

theorem f_min_max :
  let a : ℝ := -2
  let b : ℝ := 2
  (∀ x ∈ Set.Icc a b, f x ≥ -3) ∧
  (∃ x ∈ Set.Icc a b, f x = -3) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 5) ∧
  (∃ x ∈ Set.Icc a b, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_f_min_max_l447_44755


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l447_44792

theorem tangent_perpendicular_to_line (a b : ℝ) : 
  b = a^3 →                             -- point (a, b) is on the curve y = x^3
  (3 * a^2) * (-1/3) = -1 →             -- tangent is perpendicular to x + 3y + 1 = 0
  a = 1 ∨ a = -1 :=                     -- conclusion: a = 1 or a = -1
by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l447_44792


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l447_44765

theorem workshop_salary_problem (total_workers : ℕ) (all_avg_salary : ℚ) 
  (num_technicians : ℕ) (tech_avg_salary : ℚ) :
  total_workers = 21 →
  all_avg_salary = 8000 →
  num_technicians = 7 →
  tech_avg_salary = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := all_avg_salary * total_workers
  let tech_total_salary := tech_avg_salary * num_technicians
  let remaining_total_salary := total_salary - tech_total_salary
  let remaining_avg_salary := remaining_total_salary / remaining_workers
  remaining_avg_salary = 6000 := by
sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l447_44765


namespace NUMINAMATH_CALUDE_logarithm_sum_l447_44749

theorem logarithm_sum (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = 2 * Real.sqrt 2 + 5^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_l447_44749


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l447_44747

theorem geometry_biology_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119) :
  (min geometry biology) - (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_difference_l447_44747


namespace NUMINAMATH_CALUDE_triangle_right_angled_l447_44702

theorem triangle_right_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : 2 * (a^8 + b^8 + c^8) = (a^4 + b^4 + c^4)^2) : 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l447_44702


namespace NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l447_44746

theorem modulus_of_5_minus_12i : Complex.abs (5 - 12*I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_5_minus_12i_l447_44746


namespace NUMINAMATH_CALUDE_correct_number_probability_l447_44760

def first_four_options : List ℕ := [2960, 2961, 2990, 2991]
def last_three_digits : List ℕ := [6, 7, 8]

def total_possible_numbers : ℕ := (List.length first_four_options) * (Nat.factorial (List.length last_three_digits))

theorem correct_number_probability :
  (1 : ℚ) / total_possible_numbers = 1 / 24 :=
sorry

end NUMINAMATH_CALUDE_correct_number_probability_l447_44760


namespace NUMINAMATH_CALUDE_cards_playing_with_l447_44741

/-- The number of cards in a standard deck --/
def standard_deck : Nat := 52

/-- The number of cards kept away --/
def cards_kept_away : Nat := 7

/-- Theorem: The number of cards they were playing with is 45 --/
theorem cards_playing_with : 
  standard_deck - cards_kept_away = 45 := by
  sorry

end NUMINAMATH_CALUDE_cards_playing_with_l447_44741


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l447_44743

theorem sin_2alpha_value (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l447_44743


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l447_44753

/-- Represents a quadratic equation in two variables -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Checks if a quadratic equation represents a circle -/
def isCircle (eq : QuadraticEquation) : Prop :=
  eq.a = eq.b ∧ eq.a ≠ 0 ∧ eq.c^2 + eq.d^2 - 4 * eq.a * eq.f > 0

/-- The specific equation x^2 + y^2 - 4x + 2y + m = 0 -/
def specificEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := 1, c := -4, d := 2, e := 0, f := m }

/-- Theorem stating that m = 0 is sufficient but not necessary for the equation to represent a circle -/
theorem m_zero_sufficient_not_necessary :
  (∀ m : ℝ, m = 0 → isCircle (specificEquation m)) ∧
  ¬(∀ m : ℝ, isCircle (specificEquation m) → m = 0) :=
sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l447_44753


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l447_44727

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l447_44727


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l447_44709

theorem min_value_expression (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 4) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 ≥ 12 - 8 * Real.sqrt 2 :=
by sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (4/c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l447_44709


namespace NUMINAMATH_CALUDE_multiples_of_three_l447_44732

theorem multiples_of_three (n : ℕ) : (∃ k, k = 33 ∧ k * 3 = n) ↔ n = 99 := by sorry

end NUMINAMATH_CALUDE_multiples_of_three_l447_44732


namespace NUMINAMATH_CALUDE_sequence_growth_l447_44764

theorem sequence_growth (a : ℕ → ℕ) (h1 : a 1 > a 0) 
  (h2 : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l447_44764


namespace NUMINAMATH_CALUDE_jan_drove_more_than_ian_l447_44750

/-- Prove that Jan drove 174 miles more than Ian given the conditions --/
theorem jan_drove_more_than_ian (ian_time : ℝ) (ian_speed : ℝ) : 
  let han_time := ian_time + 1.5
  let han_speed := ian_speed + 6
  let jan_time := ian_time + 3
  let jan_speed := ian_speed + 8
  let ian_distance := ian_speed * ian_time
  let han_distance := han_speed * han_time
  han_distance - ian_distance = 84 →
  jan_speed * jan_time - ian_speed * ian_time = 174 :=
by sorry

end NUMINAMATH_CALUDE_jan_drove_more_than_ian_l447_44750


namespace NUMINAMATH_CALUDE_cake_pieces_l447_44757

theorem cake_pieces (cake_length : ℕ) (cake_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  cake_length = 24 →
  cake_width = 20 →
  piece_length = 3 →
  piece_width = 2 →
  (cake_length * cake_width) / (piece_length * piece_width) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_pieces_l447_44757


namespace NUMINAMATH_CALUDE_polynomial_inequality_l447_44780

/-- A polynomial satisfying the given property -/
def GoodPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x, p (x + 1) - p x = x^100

/-- The main theorem to prove -/
theorem polynomial_inequality (p : ℝ → ℝ) (hp : GoodPolynomial p) :
  ∀ t, 0 ≤ t → t ≤ 1/2 → p (1 - t) ≥ p t := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l447_44780


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l447_44707

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (- p.2, - p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (3, -4)

/-- The expected center after reflection -/
def expected_reflected_center : ℝ × ℝ := (4, -3)

theorem reflection_of_circle_center :
  reflect_about_y_eq_neg_x original_center = expected_reflected_center :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l447_44707


namespace NUMINAMATH_CALUDE_university_weighted_average_age_l447_44778

/-- Calculates the weighted average age of a university given the number of arts and technical classes,
    their respective average ages, and assuming each class has the same number of students. -/
theorem university_weighted_average_age
  (num_arts_classes : ℕ)
  (num_tech_classes : ℕ)
  (avg_age_arts : ℝ)
  (avg_age_tech : ℝ)
  (h1 : num_arts_classes = 8)
  (h2 : num_tech_classes = 5)
  (h3 : avg_age_arts = 21)
  (h4 : avg_age_tech = 18) :
  (num_arts_classes * avg_age_arts + num_tech_classes * avg_age_tech) / (num_arts_classes + num_tech_classes) = 258 / 13 := by
sorry

end NUMINAMATH_CALUDE_university_weighted_average_age_l447_44778


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l447_44706

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), a = n^2) 
  (hb : ¬ ∃ (n : ℤ), b = n^2) :
  (∃ (x0 y0 z0 w0 : ℤ), x0^2 - a*y0^2 - b*z0^2 + a*b*w0^2 = 0 ∧ (x0, y0, z0, w0) ≠ (0, 0, 0, 0)) →
  (∃ (x1 y1 z1 : ℤ), x1^2 - a*y1^2 - b*z1^2 = 0 ∧ (x1, y1, z1) ≠ (0, 0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l447_44706


namespace NUMINAMATH_CALUDE_clerical_staff_reduction_l447_44772

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_fraction : ℚ) (final_clerical_fraction : ℚ) 
  (h1 : total_employees = 3600)
  (h2 : initial_clerical_fraction = 1/3)
  (h3 : final_clerical_fraction = 1/5) : 
  ∃ (f : ℚ), 
    (initial_clerical_fraction * total_employees) * (1 - f) = 
    final_clerical_fraction * (total_employees - initial_clerical_fraction * total_employees * f) ∧ 
    f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_reduction_l447_44772


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l447_44742

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x + 3 = a * (x - h)^2 + k) → a + h + k = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l447_44742


namespace NUMINAMATH_CALUDE_max_cross_section_area_l447_44728

/-- Represents a rectangular prism in 3D space -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of the cross-section when a plane intersects a rectangular prism -/
def crossSectionArea (prism : RectangularPrism) (plane : Plane) : ℝ :=
  sorry

/-- The maximum area of the cross-sectional cut theorem -/
theorem max_cross_section_area :
  ∀ (prism : RectangularPrism) (plane : Plane),
    prism.width = 8 →
    prism.length = 12 →
    plane.a = 3 →
    plane.b = 5 →
    plane.c = -2 →
    plane.d = 30 →
    crossSectionArea prism plane = (1 / 2) * Real.sqrt 56016 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l447_44728


namespace NUMINAMATH_CALUDE_equation_solution_l447_44754

theorem equation_solution : ∃! x : ℚ, 3 * x - 4 = -6 * x + 11 ∧ x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l447_44754


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l447_44773

/-- A sequence is geometric if the ratio between any two consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence {a_n} where a_4 + a_6 = 8, 
    prove that a_1a_7 + 2a_3a_7 + a_3a_9 = 64. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_sum : a 4 + a 6 = 8) : 
    a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l447_44773


namespace NUMINAMATH_CALUDE_inequality_proof_l447_44722

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l447_44722


namespace NUMINAMATH_CALUDE_tank_capacity_l447_44767

theorem tank_capacity (initial_fill : ℚ) (added_gallons : ℚ) (final_fill : ℚ) :
  initial_fill = 3 / 4 →
  added_gallons = 9 →
  final_fill = 9 / 10 →
  ∃ (capacity : ℚ), capacity = 60 ∧ 
    final_fill * capacity = initial_fill * capacity + added_gallons :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l447_44767


namespace NUMINAMATH_CALUDE_johns_dad_age_l447_44795

theorem johns_dad_age (j d : ℕ) : j + 28 = d → j + d = 76 → d = 52 := by sorry

end NUMINAMATH_CALUDE_johns_dad_age_l447_44795


namespace NUMINAMATH_CALUDE_jordans_money_exceeds_alexs_by_12_5_percent_l447_44789

/-- Proves that Jordan's money value exceeds Alex's by 12.5% given the specified conditions -/
theorem jordans_money_exceeds_alexs_by_12_5_percent 
  (exchange_rate : ℝ) 
  (alex_dollars : ℝ) 
  (jordan_pounds : ℝ) 
  (h1 : exchange_rate = 1.5)
  (h2 : alex_dollars = 600)
  (h3 : jordan_pounds = 450) :
  (jordan_pounds * exchange_rate - alex_dollars) / alex_dollars * 100 = 12.5 := by
  sorry

#check jordans_money_exceeds_alexs_by_12_5_percent

end NUMINAMATH_CALUDE_jordans_money_exceeds_alexs_by_12_5_percent_l447_44789


namespace NUMINAMATH_CALUDE_odd_expression_l447_44791

theorem odd_expression (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (2 * p^2 - q) := by
  sorry

end NUMINAMATH_CALUDE_odd_expression_l447_44791


namespace NUMINAMATH_CALUDE_nine_point_centers_property_l447_44762

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The intersection point of the diagonals -/
def diagonalIntersection (q : Quadrilateral) : Point :=
  sorry

/-- Checks if four points are collinear -/
def areCollinear (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if four points form a parallelogram -/
def formParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Computes the nine-point center of a triangle -/
def ninePointCenter (a b c : Point) : Point :=
  sorry

/-- The main theorem -/
theorem nine_point_centers_property (q : Quadrilateral) :
  let X := diagonalIntersection q
  let center1 := ninePointCenter X q.A q.B
  let center2 := ninePointCenter X q.B q.C
  let center3 := ninePointCenter X q.C q.D
  let center4 := ninePointCenter X q.D q.A
  areCollinear center1 center2 center3 center4 ∨ 
  formParallelogram center1 center2 center3 center4 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_centers_property_l447_44762


namespace NUMINAMATH_CALUDE_frisbee_sales_receipts_l447_44729

/-- Represents the total receipts from frisbee sales for a week -/
def total_receipts (x y : ℕ) : ℕ := 3 * x + 4 * y

/-- Theorem stating that the total receipts from frisbee sales for the week is $200 -/
theorem frisbee_sales_receipts :
  ∃ (x y : ℕ), x + y = 60 ∧ y ≥ 20 ∧ total_receipts x y = 200 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_sales_receipts_l447_44729


namespace NUMINAMATH_CALUDE_no_solutions_prime_equation_l447_44777

theorem no_solutions_prime_equation (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p^a - 1 ≠ 2^n * (p - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_prime_equation_l447_44777


namespace NUMINAMATH_CALUDE_multiplier_satisfies_equation_l447_44715

/-- The multiplier that satisfies the equation when the number is 5.0 -/
def multiplier : ℝ := 7

/-- The given number in the problem -/
def number : ℝ := 5.0

/-- Theorem stating that the multiplier satisfies the equation -/
theorem multiplier_satisfies_equation : 
  4 * number + multiplier * number = 55 := by sorry

end NUMINAMATH_CALUDE_multiplier_satisfies_equation_l447_44715


namespace NUMINAMATH_CALUDE_ellipse_properties_l447_44705

/-- Given an ellipse C with equation (x^2 / a^2) + (y^2 / b^2) = 1, where a > b > 0,
    eccentricity 1/2, and the area of the quadrilateral formed by its vertices is 4√3,
    we prove properties about its equation and intersecting lines. -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := 1 / 2  -- eccentricity
  let quad_area := 4 * Real.sqrt 3  -- area of quadrilateral formed by vertices
  ∀ x y : ℝ,
    (x^2 / a^2 + y^2 / b^2 = 1) →  -- equation of ellipse C
    (e = Real.sqrt (1 - b^2 / a^2)) →  -- definition of eccentricity
    (quad_area = 4 * a * b) →  -- area of quadrilateral
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      (x₁^2 / a^2 + y₁^2 / b^2 = 1) →  -- P(x₁, y₁) on ellipse
      (x₂^2 / a^2 + y₂^2 / b^2 = 1) →  -- Q(x₂, y₂) on ellipse
      (1/2 * |x₁ * y₂ - x₂ * y₁| = Real.sqrt 3) →  -- area of triangle OPQ is √3
      (x₁^2 / 4 + y₁^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₂^2 / 4 + y₂^2 / 3 = 1) ∧  -- equation of ellipse C
      (x₁^2 + x₂^2 = 4))  -- constant sum of squares
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l447_44705


namespace NUMINAMATH_CALUDE_pierre_birthday_l447_44782

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a person's age and birthday -/
structure Person where
  age : Nat
  birthday : Date

def nextYear (d : Date) : Date :=
  if d.month = 12 && d.day = 31 then { day := 1, month := 1 }
  else { day := d.day, month := d.month }

def yesterday (d : Date) : Date :=
  if d.day = 1 && d.month = 1 then { day := 31, month := 12 }
  else if d.day = 1 then { day := 31, month := d.month - 1 }
  else { day := d.day - 1, month := d.month }

def dayBeforeYesterday (d : Date) : Date := yesterday (yesterday d)

theorem pierre_birthday (today : Date) (pierre : Person) : 
  pierre.age = 11 → 
  (dayBeforeYesterday today).day = 31 → 
  (dayBeforeYesterday today).month = 12 →
  pierre.birthday = yesterday today →
  (nextYear today).day = 1 → 
  (nextYear today).month = 1 →
  today.day = 1 ∧ today.month = 1 := by
  sorry

#check pierre_birthday

end NUMINAMATH_CALUDE_pierre_birthday_l447_44782


namespace NUMINAMATH_CALUDE_max_ships_on_battleship_board_l447_44758

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a placement of ships on a board -/
def Placement := List (ℕ × ℕ)

/-- Checks if two ships are adjacent or overlapping -/
def are_adjacent_or_overlapping (p1 p2 : ℕ × ℕ) (s : Ship) : Prop := sorry

/-- Checks if a placement is valid (no adjacent or overlapping ships) -/
def is_valid_placement (b : Board) (s : Ship) (p : Placement) : Prop := sorry

/-- The maximum number of ships that can be placed on the board -/
def max_ships (b : Board) (s : Ship) : ℕ := sorry

/-- The main theorem stating the maximum number of 1x4 ships on a 10x10 board -/
theorem max_ships_on_battleship_board :
  let b : Board := ⟨10, 10⟩
  let s : Ship := ⟨4, 1⟩
  max_ships b s = 24 := by sorry

end NUMINAMATH_CALUDE_max_ships_on_battleship_board_l447_44758


namespace NUMINAMATH_CALUDE_B_power_99_l447_44739

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, 1; 0, -1, 0]

theorem B_power_99 : B^99 = !![0, 0, 0; 0, 0, -1; 0, 1, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_99_l447_44739


namespace NUMINAMATH_CALUDE_symmetry_conditions_l447_44723

/-- A function is symmetric about a point (a, b) if f(x) + f(2a - x) = 2b for all x in its domain -/
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

theorem symmetry_conditions (m a : ℝ) :
  let f := fun x : ℝ => (x^2 + m*x + m) / x
  let g := fun x : ℝ => if x > 0 then x^2 + a*x + 1 else -x^2 + a*x + 1
  (SymmetricAbout f 0 1) ∧
  (∀ x ≠ 0, SymmetricAbout g 0 1) ∧
  (∀ x t, x < 0 → t > 0 → g x < f t) →
  (m = 1) ∧
  (∀ x < 0, g x = -x^2 + a*x + 1) ∧
  (-2 * Real.sqrt 2 < a) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_conditions_l447_44723


namespace NUMINAMATH_CALUDE_michael_earnings_l447_44748

/-- Calculates the total money earned from selling birdhouses --/
def total_money_earned (large_price medium_price small_price : ℕ) 
                       (large_sold medium_sold small_sold : ℕ) : ℕ :=
  large_price * large_sold + medium_price * medium_sold + small_price * small_sold

/-- Theorem: Michael's earnings from selling birdhouses --/
theorem michael_earnings : 
  total_money_earned 22 16 7 2 2 3 = 97 := by sorry

end NUMINAMATH_CALUDE_michael_earnings_l447_44748


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l447_44793

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (books : BookCollection) : Prop :=
  books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 220 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ (books : BookCollection), is_valid_collection books → books.hardcover_nonfiction = 40 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l447_44793


namespace NUMINAMATH_CALUDE_pegboard_empty_holes_l447_44710

/-- Represents a square pegboard -/
structure Pegboard :=
  (size : ℕ)

/-- Calculates the total number of holes on the pegboard -/
def total_holes (p : Pegboard) : ℕ := (p.size + 1) ^ 2

/-- Calculates the number of holes with pegs (on diagonals) -/
def holes_with_pegs (p : Pegboard) : ℕ := 2 * (p.size + 1) - 1

/-- Calculates the number of empty holes on the pegboard -/
def empty_holes (p : Pegboard) : ℕ := total_holes p - holes_with_pegs p

theorem pegboard_empty_holes :
  ∃ (p : Pegboard), p.size = 10 ∧ empty_holes p = 100 :=
sorry

end NUMINAMATH_CALUDE_pegboard_empty_holes_l447_44710


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l447_44720

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 700)
  (h2 : blue_percent = 45/100)
  (h3 : red_percent = 23/100)
  (h4 : green_percent = 15/100) :
  ⌊(1 - (blue_percent + red_percent + green_percent)) * total_students⌋ = 119 := by
sorry

end NUMINAMATH_CALUDE_students_wearing_other_colors_l447_44720


namespace NUMINAMATH_CALUDE_grid_transform_iff_even_l447_44771

/-- Represents a grid operation that changes adjacent entries' signs -/
def GridOperation (n : ℕ) := Fin n → Fin n → Unit

/-- Represents the state of the grid -/
def GridState (n : ℕ) := Fin n → Fin n → Int

/-- Initial grid state with all entries 1 -/
def initialGrid (n : ℕ) : GridState n :=
  λ _ _ => 1

/-- Final grid state with all entries -1 -/
def finalGrid (n : ℕ) : GridState n :=
  λ _ _ => -1

/-- Predicate to check if a sequence of operations can transform the grid -/
def canTransform (n : ℕ) : Prop :=
  ∃ (seq : List (GridOperation n)), 
    ∃ (result : GridState n), 
      result = finalGrid n

/-- Main theorem: Grid can be transformed iff n is even -/
theorem grid_transform_iff_even (n : ℕ) (h : n ≥ 2) : 
  canTransform n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_grid_transform_iff_even_l447_44771


namespace NUMINAMATH_CALUDE_square_fold_distance_l447_44796

/-- Given a square ABCD with side length 4, folded along diagonal BD to form a dihedral angle of 60°,
    the distance between the midpoint of BC and point A is 2√2. -/
theorem square_fold_distance (A B C D : ℝ × ℝ) : 
  let side_length : ℝ := 4
  let dihedral_angle : ℝ := 60
  let is_square := (A.1 = 0 ∧ A.2 = 0) ∧ 
                   (B.1 = side_length ∧ B.2 = 0) ∧ 
                   (C.1 = side_length ∧ C.2 = side_length) ∧ 
                   (D.1 = 0 ∧ D.2 = side_length)
  let midpoint_BC := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let distance := Real.sqrt ((A.1 - midpoint_BC.1)^2 + (A.2 - midpoint_BC.2)^2)
  is_square → distance = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_fold_distance_l447_44796


namespace NUMINAMATH_CALUDE_original_kittens_correct_l447_44745

/-- The number of kittens Tim's cat originally had -/
def original_kittens : ℕ := 6

/-- The number of kittens Tim gave away -/
def kittens_given_away : ℕ := 3

/-- The number of kittens Tim received -/
def kittens_received : ℕ := 9

/-- The number of kittens Tim has now -/
def current_kittens : ℕ := 12

/-- Theorem stating that the original number of kittens is correct -/
theorem original_kittens_correct : 
  original_kittens + kittens_received - kittens_given_away = current_kittens := by
  sorry

end NUMINAMATH_CALUDE_original_kittens_correct_l447_44745


namespace NUMINAMATH_CALUDE_max_value_a_inequality_l447_44776

theorem max_value_a_inequality (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/2 → 
    (x₂ * Real.sin x₁ - x₁ * Real.sin x₂) / (x₁ - x₂) > a) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_inequality_l447_44776


namespace NUMINAMATH_CALUDE_eggs_in_box_l447_44774

/-- The number of eggs Harry takes from the box -/
def eggs_taken : ℕ := 5

/-- The number of eggs left in the box after Harry takes some -/
def eggs_left : ℕ := 42

/-- The initial number of eggs in the box -/
def initial_eggs : ℕ := eggs_taken + eggs_left

theorem eggs_in_box : initial_eggs = 47 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l447_44774


namespace NUMINAMATH_CALUDE_cube_sum_plus_triple_product_l447_44703

theorem cube_sum_plus_triple_product (x y : ℝ) (h : x + y = 1) :
  x^3 + y^3 + 3*x*y = 1 := by sorry

end NUMINAMATH_CALUDE_cube_sum_plus_triple_product_l447_44703


namespace NUMINAMATH_CALUDE_square_root_sum_equals_eight_l447_44717

theorem square_root_sum_equals_eight (x : ℝ) : 
  (Real.sqrt (49 - x^2) - Real.sqrt (25 - x^2) = 3) → 
  (Real.sqrt (49 - x^2) + Real.sqrt (25 - x^2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_eight_l447_44717


namespace NUMINAMATH_CALUDE_average_position_l447_44740

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position (average : ℚ := (fractions.sum) / 6) :
  average = 223/840 ∧ 1/4 < average ∧ average < 1/3 := by sorry

end NUMINAMATH_CALUDE_average_position_l447_44740


namespace NUMINAMATH_CALUDE_elaine_jerry_ratio_l447_44752

/-- Represents the time spent in the pool by each person --/
structure PoolTime where
  jerry : ℚ
  elaine : ℚ
  george : ℚ
  kramer : ℚ

/-- Conditions of the problem --/
def pool_conditions (t : PoolTime) : Prop :=
  t.jerry = 3 ∧
  t.george = t.elaine / 3 ∧
  t.kramer = 0 ∧
  t.jerry + t.elaine + t.george + t.kramer = 11

/-- The theorem to be proved --/
theorem elaine_jerry_ratio (t : PoolTime) :
  pool_conditions t → t.elaine / t.jerry = 2 := by
  sorry


end NUMINAMATH_CALUDE_elaine_jerry_ratio_l447_44752


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l447_44724

/-- Proves that arctan(tan 75° - 3 tan 30°) is approximately 124.1°. -/
theorem arctan_tan_difference (ε : ℝ) (h : ε > 0) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 180 ∧ |θ - 124.1| < ε ∧ θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) * 180 / π :=
sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l447_44724


namespace NUMINAMATH_CALUDE_bakery_rolls_distribution_l447_44733

theorem bakery_rolls_distribution (n k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  Nat.choose (n + k - 1) (k - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_distribution_l447_44733


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_target_l447_44721

/-- A function f satisfying the given condition for all non-zero real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The condition that f satisfies for all non-zero real x -/
axiom f_condition (x : ℝ) (hx : x ≠ 0) : 2 * f x + f (1 / x) = 5 * x + 4

/-- The value we're looking for -/
def target_value : ℝ := 2004

/-- The theorem to prove -/
theorem sum_of_roots_equals_target (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := -((3 * target_value - 4) / 10)
  let c : ℝ := 5 / 2
  x^2 + b*x + c = 0 → x + (-b/a) = (3 * target_value - 4) / 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_target_l447_44721


namespace NUMINAMATH_CALUDE_class_size_l447_44759

theorem class_size (average_weight : ℝ) (teacher_weight : ℝ) (new_average : ℝ) :
  average_weight = 35 →
  teacher_weight = 45 →
  new_average = 35.4 →
  ∃ n : ℕ, (n : ℝ) * average_weight + teacher_weight = new_average * ((n : ℝ) + 1) ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l447_44759


namespace NUMINAMATH_CALUDE_combined_prism_volume_l447_44735

/-- The volume of a structure consisting of a triangular prism on top of a rectangular prism -/
theorem combined_prism_volume (rect_length rect_width rect_height tri_base tri_height tri_length : ℝ) :
  rect_length = 6 →
  rect_width = 4 →
  rect_height = 2 →
  tri_base = 3 →
  tri_height = 3 →
  tri_length = 4 →
  (rect_length * rect_width * rect_height) + (1/2 * tri_base * tri_height * tri_length) = 66 := by
  sorry

end NUMINAMATH_CALUDE_combined_prism_volume_l447_44735


namespace NUMINAMATH_CALUDE_max_a_value_l447_44713

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, x * Real.log x - (1 + a) * x + 1 ≥ 0) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l447_44713


namespace NUMINAMATH_CALUDE_decimal_place_values_l447_44700

/-- Represents the place value in a decimal number system. -/
inductive PlaceValue
| Ones
| Tens
| Hundreds
| Thousands
| TenThousands
| HundredThousands
| Millions
| TenMillions
| HundredMillions

/-- Returns the position of a place value from right to left. -/
def position (pv : PlaceValue) : Nat :=
  match pv with
  | .Ones => 1
  | .Tens => 2
  | .Hundreds => 3
  | .Thousands => 4
  | .TenThousands => 5
  | .HundredThousands => 6
  | .Millions => 7
  | .TenMillions => 8
  | .HundredMillions => 9

theorem decimal_place_values :
  (position PlaceValue.Hundreds = 3) ∧
  (position PlaceValue.TenThousands = 5) ∧
  (position PlaceValue.Thousands = 4) := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_values_l447_44700


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l447_44751

theorem geometric_sequence_common_ratio 
  (a b : ℝ) 
  (h1 : 2 * a = 1 + b) 
  (h2 : (a + 2)^2 = 3 * (b + 5)) 
  (h3 : a + 2 ≠ 0) 
  (h4 : b + 5 ≠ 0) : 
  (a + 2) / 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l447_44751


namespace NUMINAMATH_CALUDE_all_children_receive_candy_l447_44799

/-- The function that determines which child receives a candy on each turn -/
def candy_distribution (n : ℕ) (x : ℕ) : ℕ :=
  (x * (x + 1) / 2) % n

/-- Proposition: All children receive candy iff the number of children is a power of 2 -/
theorem all_children_receive_candy (n : ℕ) :
  (∀ k : ℕ, k < n → ∃ x : ℕ, candy_distribution n x = k) ↔ ∃ a : ℕ, n = 2^a :=
sorry

end NUMINAMATH_CALUDE_all_children_receive_candy_l447_44799


namespace NUMINAMATH_CALUDE_crown_composition_l447_44779

theorem crown_composition (total_weight : ℝ) (gold copper tin iron : ℝ)
  (h1 : total_weight = 60)
  (h2 : gold + copper + tin + iron = total_weight)
  (h3 : gold + copper = 2/3 * total_weight)
  (h4 : gold + tin = 3/4 * total_weight)
  (h5 : gold + iron = 3/5 * total_weight) :
  gold = 30.5 ∧ copper = 9.5 ∧ tin = 14.5 ∧ iron = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_crown_composition_l447_44779


namespace NUMINAMATH_CALUDE_sons_age_l447_44769

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 6 * son_age →
  father_age + 6 + son_age + 6 = 68 →
  son_age = 8 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l447_44769


namespace NUMINAMATH_CALUDE_arithmetic_is_F_sequence_l447_44798

def is_F_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n ≥ 3, ∃ i j : ℕ, i ≠ j ∧ i < n ∧ j < n ∧ a n = a i + a j

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = 2 * n

theorem arithmetic_is_F_sequence :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → is_F_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_is_F_sequence_l447_44798


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l447_44770

-- Define what it means for an angle to be in the third quadrant
def in_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

-- Define what it means for an angle to be in the second quadrant
def in_second_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

-- Theorem statement
theorem half_angle_quadrant (α : ℝ) :
  in_third_quadrant α → in_second_quadrant (α/2) ∨ in_fourth_quadrant (α/2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l447_44770
