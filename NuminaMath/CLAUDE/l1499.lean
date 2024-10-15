import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_l1499_149984

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x > 0, (x + 1) * Real.exp x > 1) ↔ (∃ x₀ > 0, (x₀ + 1) * Real.exp x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1499_149984


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1499_149911

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - 4 * (1 + 2 + 3) / (5 + 10 + 15) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1499_149911


namespace NUMINAMATH_CALUDE_sheridan_fish_count_l1499_149941

/-- Calculates the remaining number of fish after giving some away -/
def remaining_fish (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 25.0 fish after giving away 22.0 from her initial 47.0 fish -/
theorem sheridan_fish_count : remaining_fish 47.0 22.0 = 25.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_fish_count_l1499_149941


namespace NUMINAMATH_CALUDE_smallest_k_for_same_remainder_l1499_149971

theorem smallest_k_for_same_remainder : ∃ (k : ℕ), k > 0 ∧
  (∀ (n : ℕ), n > 0 → n < k → ¬((201 + n) % 24 = (9 + n) % 24)) ∧
  ((201 + k) % 24 = (9 + k) % 24) ∧
  (201 % 24 = 9 % 24) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_same_remainder_l1499_149971


namespace NUMINAMATH_CALUDE_opera_house_earnings_correct_l1499_149912

/-- Calculates the earnings of an opera house for a single show. -/
def opera_house_earnings (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℕ) (percent_empty : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let occupied_seats := total_seats - (total_seats * percent_empty / 100)
  occupied_seats * ticket_price

/-- Theorem stating that the opera house earnings for the given conditions equal $12000. -/
theorem opera_house_earnings_correct : opera_house_earnings 150 10 10 20 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_opera_house_earnings_correct_l1499_149912


namespace NUMINAMATH_CALUDE_sqrt_four_plus_abs_sqrt_three_minus_two_l1499_149914

theorem sqrt_four_plus_abs_sqrt_three_minus_two :
  Real.sqrt 4 + |Real.sqrt 3 - 2| = 4 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_plus_abs_sqrt_three_minus_two_l1499_149914


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1499_149961

theorem smallest_solution_of_equation (x : ℝ) : 
  (x = (7 - Real.sqrt 33) / 2) ↔ 
  (x < (7 + Real.sqrt 33) / 2 ∧ 1 / (x - 1) + 1 / (x - 5) = 4 / (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1499_149961


namespace NUMINAMATH_CALUDE_total_toys_count_l1499_149991

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The total number of toys given -/
def total_toys : ℕ := toy_cars + dolls

theorem total_toys_count : total_toys = 403 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l1499_149991


namespace NUMINAMATH_CALUDE_midpoint_calculation_l1499_149988

/-- Given two points A and B in a 2D plane, proves that 3x - 5y = -18,
    where (x, y) is the midpoint of AB. -/
theorem midpoint_calculation (A B : ℝ × ℝ) (h1 : A = (-8, 15)) (h2 : B = (16, -3)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  3 * C.1 - 5 * C.2 = -18 := by
sorry

end NUMINAMATH_CALUDE_midpoint_calculation_l1499_149988


namespace NUMINAMATH_CALUDE_cubic_root_relation_l1499_149993

theorem cubic_root_relation (m n p x₃ : ℝ) : 
  (∃ (z : ℂ), z^3 + (m/3)*z^2 + (n/3)*z + (p/3) = 0 ∧ 
               (z = 4 + 3*Complex.I ∨ z = 4 - 3*Complex.I ∨ z = x₃)) →
  x₃ > 0 →
  p = -75 * x₃ := by
sorry

end NUMINAMATH_CALUDE_cubic_root_relation_l1499_149993


namespace NUMINAMATH_CALUDE_f_negative_l1499_149970

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ := x * (1 - x)

-- Theorem to prove
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = x * (1 + x) :=
by sorry

end NUMINAMATH_CALUDE_f_negative_l1499_149970


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1499_149917

def f (x : ℝ) := x^3 + x

theorem tangent_line_at_one :
  ∀ y : ℝ, (4 * 1 - y - 2 = 0) ↔ (∃ m : ℝ, m = (f 1 - f x) / (1 - x) ∧ y = m * (1 - x) + f 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1499_149917


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1499_149913

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1499_149913


namespace NUMINAMATH_CALUDE_world_cup_merchandise_problem_l1499_149946

def total_items : ℕ := 90
def ornament_cost : ℕ := 40
def pendant_cost : ℕ := 25
def total_cost : ℕ := 2850
def ornament_price : ℕ := 50
def pendant_price : ℕ := 30
def min_profit : ℕ := 725

theorem world_cup_merchandise_problem :
  ∃ (ornaments pendants : ℕ),
    ornaments + pendants = total_items ∧
    ornament_cost * ornaments + pendant_cost * pendants = total_cost ∧
    ornaments = 40 ∧
    pendants = 50 ∧
    (∀ m : ℕ,
      m ≤ total_items ∧
      (ornament_price - ornament_cost) * (total_items - m) + (pendant_price - pendant_cost) * m ≥ min_profit
      → m ≤ 35) :=
by sorry

end NUMINAMATH_CALUDE_world_cup_merchandise_problem_l1499_149946


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1499_149901

theorem sqrt_expression_equality (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1499_149901


namespace NUMINAMATH_CALUDE_simplify_and_sum_fraction_l1499_149933

theorem simplify_and_sum_fraction : ∃ (a b : ℕ), 
  (a : ℚ) / b = 63 / 126 ∧ 
  (∀ (c d : ℕ), (c : ℚ) / d = 63 / 126 → a ≤ c ∧ b ≤ d) ∧ 
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_sum_fraction_l1499_149933


namespace NUMINAMATH_CALUDE_tetrahedron_distance_altitude_inequality_l1499_149974

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between any pair of opposite edges -/
  d : ℝ
  /-- The length of the shortest altitude -/
  h : ℝ
  /-- Assumption that d and h are positive -/
  d_pos : d > 0
  h_pos : h > 0

/-- Theorem: For any tetrahedron, twice the minimum distance between opposite edges
    is greater than the length of the shortest altitude -/
theorem tetrahedron_distance_altitude_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_altitude_inequality_l1499_149974


namespace NUMINAMATH_CALUDE_div_problem_l1499_149999

theorem div_problem (a b c : ℚ) (h1 : a / b = 5) (h2 : b / c = 2/5) : c / a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_div_problem_l1499_149999


namespace NUMINAMATH_CALUDE_existence_of_ratio_triplet_l1499_149918

def TwoColorFunction := ℕ → Bool

theorem existence_of_ratio_triplet (color : TwoColorFunction) :
  ∃ A B C : ℕ, (color A = color B) ∧ (color B = color C) ∧ (A * B = C * C) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_ratio_triplet_l1499_149918


namespace NUMINAMATH_CALUDE_money_distribution_l1499_149904

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 450)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1499_149904


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1499_149965

def is_hyperbola (a b : ℝ) (x y : ℝ → ℝ) : Prop :=
  ∀ t, (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

def has_focus_at (x y : ℝ → ℝ) (fx fy : ℝ) : Prop :=
  ∃ t, x t = fx ∧ y t = fy

def has_asymptotes (x y : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ t, y t = m * x t ∨ y t = -m * x t

theorem hyperbola_equation (a b : ℝ) (x y : ℝ → ℝ) :
  is_hyperbola a b x y →
  has_focus_at x y 5 0 →
  has_asymptotes x y (3/4) →
  a^2 = 16 ∧ b^2 = 9 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1499_149965


namespace NUMINAMATH_CALUDE_focus_of_specific_ellipse_l1499_149935

/-- An ellipse with given major and minor axis endpoints -/
structure Ellipse where
  major_axis_start : ℝ × ℝ
  major_axis_end : ℝ × ℝ
  minor_axis_start : ℝ × ℝ
  minor_axis_end : ℝ × ℝ

/-- The focus of an ellipse with the greater x-coordinate -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate is at (3, -2) -/
theorem focus_of_specific_ellipse :
  let e : Ellipse := {
    major_axis_start := (0, -2),
    major_axis_end := (6, -2),
    minor_axis_start := (3, 1),
    minor_axis_end := (3, -5)
  }
  focus_with_greater_x e = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_ellipse_l1499_149935


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l1499_149959

/-- Given a monotonically decreasing exponential function f(x) = a^x on ℝ,
    prove that when f(x+1) ≥ 1, the range of x is (-∞, -1]. -/
theorem exponential_decreasing_range (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a^x) :
  (∀ x y, x < y → f x > f y) →
  {x : ℝ | f (x + 1) ≥ 1} = Set.Iic (-1) := by
sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l1499_149959


namespace NUMINAMATH_CALUDE_max_perfect_squares_l1499_149905

/-- The sequence (a_n) defined recursively -/
def a : ℕ → ℕ → ℕ
  | m, 0 => m
  | m, n + 1 => (a m n)^5 + 487

/-- Proposition: m = 9 is the unique positive integer that maximizes perfect squares in the sequence -/
theorem max_perfect_squares (m : ℕ) : m > 0 → (∀ k : ℕ, k > 0 → (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a k i = j^2) → 
  (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a m i = j^2)) → m = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l1499_149905


namespace NUMINAMATH_CALUDE_larger_number_proof_l1499_149916

theorem larger_number_proof (a b : ℝ) : 
  a + b = 104 → 
  a^2 - b^2 = 208 → 
  max a b = 53 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1499_149916


namespace NUMINAMATH_CALUDE_license_plate_theorem_l1499_149949

def alphabet_size : ℕ := 25  -- Excluding 'A'
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2
def total_digits : ℕ := 10

-- Define the function to calculate the number of license plate combinations
def license_plate_combinations : ℕ :=
  (alphabet_size.choose 2) *  -- Choose 2 letters from 25
  (letter_positions.choose 2) *  -- Choose 2 positions for one letter
  (total_digits) *  -- Choose first digit
  (total_digits - 1)  -- Choose second digit

-- Theorem statement
theorem license_plate_theorem :
  license_plate_combinations = 162000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l1499_149949


namespace NUMINAMATH_CALUDE_problem_solution_l1499_149931

theorem problem_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h1 : 1/p + 1/q = 2) (h2 : p*q = 1) : p = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1499_149931


namespace NUMINAMATH_CALUDE_triangle_side_length_l1499_149972

theorem triangle_side_length (A B C : ℝ) (AC AB BC : ℝ) (angle_A : ℝ) :
  AC = Real.sqrt 2 →
  AB = 2 →
  (Real.sqrt 3 * Real.sin angle_A + Real.cos angle_A) / (Real.sqrt 3 * Real.cos angle_A - Real.sin angle_A) = Real.tan (5 * Real.pi / 12) →
  BC = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1499_149972


namespace NUMINAMATH_CALUDE_b_share_is_3000_l1499_149902

/-- Proves that B's share is 3000 when money is distributed in the proportion 6:3:5:4 and C gets 1000 more than D -/
theorem b_share_is_3000 (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →  -- Sum of all shares equals the total
  6 * b = 3 * a →          -- A:B proportion is 6:3
  5 * b = 5 * a →          -- B:C proportion is 3:5
  4 * b = 3 * d →          -- B:D proportion is 3:4
  c = d + 1000 →           -- C gets 1000 more than D
  b = 3000 := by
sorry

end NUMINAMATH_CALUDE_b_share_is_3000_l1499_149902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1499_149900

theorem arithmetic_sequence_length :
  ∀ (a₁ d n : ℤ),
    a₁ = -48 →
    d = 8 →
    a₁ + (n - 1) * d = 80 →
    n = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1499_149900


namespace NUMINAMATH_CALUDE_walker_catch_up_equations_l1499_149950

theorem walker_catch_up_equations 
  (good_efficiency bad_efficiency initial_lead : ℕ) 
  (h_efficiency : good_efficiency > bad_efficiency) 
  (h_initial_lead : initial_lead > 0) : 
  ∃ (x y : ℚ), 
    x - y = initial_lead ∧ 
    x = (good_efficiency : ℚ) / bad_efficiency * y ∧ 
    x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_walker_catch_up_equations_l1499_149950


namespace NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1499_149976

theorem least_coins (b : ℕ) : b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] → b ≥ 10 := by
  sorry

theorem ten_coins : 10 ≡ 3 [ZMOD 7] ∧ 10 ≡ 2 [ZMOD 4] := by
  sorry

theorem coins_in_wallet : ∃ (b : ℕ), b ≡ 3 [ZMOD 7] ∧ b ≡ 2 [ZMOD 4] ∧ 
  ∀ (n : ℕ), n ≡ 3 [ZMOD 7] ∧ n ≡ 2 [ZMOD 4] → b ≤ n := by
  sorry

end NUMINAMATH_CALUDE_least_coins_ten_coins_coins_in_wallet_l1499_149976


namespace NUMINAMATH_CALUDE_rainbow_preschool_full_day_students_l1499_149921

theorem rainbow_preschool_full_day_students 
  (total_students : ℕ) 
  (half_day_percentage : ℚ) 
  (h1 : total_students = 80)
  (h2 : half_day_percentage = 1/4) : 
  (1 - half_day_percentage) * total_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_preschool_full_day_students_l1499_149921


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1499_149926

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def P : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (
    -- The line y = mx + b passes through P
    m * P.1 + b = P.2 ∧
    -- The slope m is equal to f'(1)
    m = (6 : ℝ) * P.1 - 1 ∧
    -- The resulting equation is 2x - y + 1 = 0
    m = 2 ∧ b = 1 ∧
    ∀ x y, y = m * x + b ↔ 2 * x - y + 1 = 0
  ) := by sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l1499_149926


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l1499_149998

/-- The probability of getting at least one head when tossing four fair coins -/
theorem prob_at_least_one_head_four_coins : ℝ :=
  let p_tail : ℝ := 1 / 2  -- probability of getting a tail on one coin toss
  let p_all_tails : ℝ := p_tail ^ 4  -- probability of getting all tails
  1 - p_all_tails

/-- Proof that the probability of getting at least one head when tossing four fair coins is 15/16 -/
theorem prob_at_least_one_head_four_coins_is_15_16 :
  prob_at_least_one_head_four_coins = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_four_coins_prob_at_least_one_head_four_coins_is_15_16_l1499_149998


namespace NUMINAMATH_CALUDE_max_reflections_before_target_angle_max_reflections_is_optimal_l1499_149989

/-- The angle between the two reflecting lines in degrees -/
def angle_between_lines : ℝ := 5

/-- The target angle of incidence in degrees -/
def target_angle : ℝ := 85

/-- The maximum number of reflections -/
def max_reflections : ℕ := 17

theorem max_reflections_before_target_angle :
  ∀ n : ℕ, n * angle_between_lines ≤ target_angle ↔ n ≤ max_reflections :=
by sorry

theorem max_reflections_is_optimal :
  (max_reflections + 1) * angle_between_lines > target_angle :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_before_target_angle_max_reflections_is_optimal_l1499_149989


namespace NUMINAMATH_CALUDE_function_exponent_proof_l1499_149920

theorem function_exponent_proof (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) → f 3 = Real.sqrt 3 → n = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_exponent_proof_l1499_149920


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1499_149973

theorem complex_magnitude_equation :
  ∃! (x : ℝ), x > 0 ∧ Complex.abs (x - 3 * Complex.I * Real.sqrt 5) * Complex.abs (8 - 5 * Complex.I) = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1499_149973


namespace NUMINAMATH_CALUDE_expression_value_l1499_149927

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - (z - 1)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1499_149927


namespace NUMINAMATH_CALUDE_charles_total_money_l1499_149930

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The number of pennies Charles found on his way to school -/
def pennies_found : ℕ := 6

/-- The number of nickels Charles found on his way to school -/
def nickels_found : ℕ := 8

/-- The number of dimes Charles found on his way to school -/
def dimes_found : ℕ := 6

/-- The number of quarters Charles found on his way to school -/
def quarters_found : ℕ := 5

/-- The number of nickels Charles had at home -/
def nickels_at_home : ℕ := 3

/-- The number of dimes Charles had at home -/
def dimes_at_home : ℕ := 12

/-- The number of quarters Charles had at home -/
def quarters_at_home : ℕ := 7

/-- The number of half-dollars Charles had at home -/
def half_dollars_at_home : ℕ := 2

/-- The total amount of money Charles has -/
def total_money : ℚ :=
  penny_value * pennies_found +
  nickel_value * (nickels_found + nickels_at_home) +
  dime_value * (dimes_found + dimes_at_home) +
  quarter_value * (quarters_found + quarters_at_home) +
  half_dollar_value * half_dollars_at_home

theorem charles_total_money :
  total_money = 6.41 := by sorry

end NUMINAMATH_CALUDE_charles_total_money_l1499_149930


namespace NUMINAMATH_CALUDE_sum_of_roots_l1499_149981

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1499_149981


namespace NUMINAMATH_CALUDE_function_composition_result_l1499_149908

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem function_composition_result : f (g (Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l1499_149908


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1499_149997

theorem complex_exponential_sum : 
  12 * Complex.exp (Complex.I * Real.pi / 7) + 12 * Complex.exp (Complex.I * 19 * Real.pi / 14) = 
  24 * Real.cos (5 * Real.pi / 28) * Complex.exp (Complex.I * 3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1499_149997


namespace NUMINAMATH_CALUDE_circle_point_range_l1499_149994

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 →
  (∃ a b : ℝ, C a b ∧
    dot_product (a + m, b) (a - m, b) = 0) →
  4 ≤ m ∧ m ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l1499_149994


namespace NUMINAMATH_CALUDE_x_congruence_l1499_149940

theorem x_congruence (x : ℤ) 
  (h1 : (2 + x) % 4 = 3 % 4)
  (h2 : (4 + x) % 16 = 8 % 16)
  (h3 : (6 + x) % 36 = 7 % 36) :
  x % 48 = 1 % 48 := by
sorry

end NUMINAMATH_CALUDE_x_congruence_l1499_149940


namespace NUMINAMATH_CALUDE_gas_cost_proof_l1499_149967

/-- The original total cost of gas for a group of friends -/
def original_cost : ℝ := 200

/-- The number of friends initially -/
def initial_friends : ℕ := 5

/-- The number of additional friends who joined -/
def additional_friends : ℕ := 3

/-- The decrease in cost per person for the original friends -/
def cost_decrease : ℝ := 15

theorem gas_cost_proof :
  let total_friends := initial_friends + additional_friends
  let initial_cost_per_person := original_cost / initial_friends
  let final_cost_per_person := original_cost / total_friends
  initial_cost_per_person - final_cost_per_person = cost_decrease :=
by sorry

end NUMINAMATH_CALUDE_gas_cost_proof_l1499_149967


namespace NUMINAMATH_CALUDE_tyler_erasers_count_l1499_149944

def tyler_problem (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
  (eraser_price : ℕ) (remaining_money : ℕ) : ℕ := 
  let money_after_scissors := initial_money - scissors_count * scissors_price
  let money_spent_on_erasers := money_after_scissors - remaining_money
  money_spent_on_erasers / eraser_price

theorem tyler_erasers_count : 
  tyler_problem 100 8 5 4 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tyler_erasers_count_l1499_149944


namespace NUMINAMATH_CALUDE_ellipse_sum_parameters_l1499_149938

/-- An ellipse with foci F₁ and F₂, and constant sum of distances 2a -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  a : ℝ

/-- The standard form equation of an ellipse -/
structure EllipseEquation where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, this function returns its standard form equation -/
def ellipse_to_equation (e : Ellipse) : EllipseEquation :=
  sorry

theorem ellipse_sum_parameters (e : Ellipse) (eq : EllipseEquation) :
  e.F₁ = (0, 0) →
  e.F₂ = (6, 0) →
  e.a = 5 →
  eq = ellipse_to_equation e →
  eq.h + eq.k + eq.a + eq.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_parameters_l1499_149938


namespace NUMINAMATH_CALUDE_ellipse_equation_l1499_149958

/-- Represents an ellipse with foci on coordinate axes and midpoint at origin -/
structure Ellipse where
  focal_distance : ℝ
  sum_distances : ℝ

/-- The equation of the ellipse when foci are on the x-axis -/
def ellipse_equation_x (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / ((e.sum_distances / 2)^2) + y^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- The equation of the ellipse when foci are on the y-axis -/
def ellipse_equation_y (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / ((e.sum_distances / 2)^2) + x^2 / ((e.sum_distances / 2)^2 - (e.focal_distance / 2)^2) = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation (e : Ellipse) (h1 : e.focal_distance = 8) (h2 : e.sum_distances = 12) :
  ∀ x y : ℝ, ellipse_equation_x e x y ∨ ellipse_equation_y e x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1499_149958


namespace NUMINAMATH_CALUDE_not_divisible_by_fifteen_l1499_149963

theorem not_divisible_by_fifteen (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_fifteen_l1499_149963


namespace NUMINAMATH_CALUDE_cubs_series_win_probability_l1499_149923

def probability_cubs_win_game : ℚ := 2/3

def probability_cubs_win_series : ℚ :=
  (1 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^0) +
  (3 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^1) +
  (6 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^2)

theorem cubs_series_win_probability :
  probability_cubs_win_series = 64/81 :=
by sorry

end NUMINAMATH_CALUDE_cubs_series_win_probability_l1499_149923


namespace NUMINAMATH_CALUDE_hyperbola_a_minus_h_l1499_149982

/-- The standard form equation of a hyperbola -/
def is_hyperbola (a b h k x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The equation of an asymptote -/
def is_asymptote (m c x y : ℝ) : Prop :=
  y = m * x + c

theorem hyperbola_a_minus_h (a b h k : ℝ) :
  a > 0 →
  b > 0 →
  is_asymptote 3 4 h k →
  is_asymptote (-3) 6 h k →
  is_hyperbola a b h k 1 9 →
  a - h = 2 * Real.sqrt 3 - 1/3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_a_minus_h_l1499_149982


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1499_149939

theorem max_x_minus_y (x y : Real) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π/2) (h4 : Real.tan x = 3 * Real.tan y) :
  ∃ (max_val : Real), max_val = π/6 ∧ x - y ≤ max_val ∧ ∃ (x' y' : Real), 0 < y' ∧ y' ≤ x' ∧ x' < π/2 ∧ Real.tan x' = 3 * Real.tan y' ∧ x' - y' = max_val :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1499_149939


namespace NUMINAMATH_CALUDE_first_cube_weight_l1499_149919

/-- Given two cubical blocks of the same metal, where the sides of the second cube
    are twice as long as the first cube, and the second cube weighs 24 pounds,
    prove that the weight of the first cubical block is 3 pounds. -/
theorem first_cube_weight (s : ℝ) (weight : ℝ → ℝ) :
  (∀ x, weight (8 * x) = 8 * weight x) →  -- Weight is proportional to volume
  weight (8 * s^3) = 24 →                 -- Second cube weighs 24 pounds
  weight (s^3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_first_cube_weight_l1499_149919


namespace NUMINAMATH_CALUDE_xander_miles_more_l1499_149948

/-- The problem statement and conditions --/
theorem xander_miles_more (t s : ℝ) 
  (h1 : t > 0) 
  (h2 : s > 0) 
  (h3 : s * t + 100 = (s + 10) * (t + 1.5)) : 
  (s + 15) * (t + 3) - s * t = 215 := by
  sorry

end NUMINAMATH_CALUDE_xander_miles_more_l1499_149948


namespace NUMINAMATH_CALUDE_no_factor_of_polynomial_l1499_149937

theorem no_factor_of_polynomial : ¬ ∃ (p : Polynomial ℝ), 
  (p = X^2 + 4*X + 4 ∨ 
   p = X^2 - 4*X + 4 ∨ 
   p = X^2 + 2*X + 4 ∨ 
   p = X^2 + 4) ∧ 
  (∃ (q : Polynomial ℝ), X^4 - 4*X^2 + 16 = p * q) := by
  sorry

end NUMINAMATH_CALUDE_no_factor_of_polynomial_l1499_149937


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1499_149979

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) ↔ a * x^2 + b * x + c ≥ 0) →
  (a > 0 ∧
   (∀ x : ℝ, bx + c > 0 ↔ x < -6) ∧
   a + b + c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1499_149979


namespace NUMINAMATH_CALUDE_edge_coloring_theorem_l1499_149977

/-- Given a complete graph K_n with n vertices, this theorem states that:
    1) If we color the edges with at least n colors, there will be a triangle with all edges in different colors.
    2) If we color the edges with at most n-3 colors, there will be a cycle of length 3 or 4 with all edges in the same color.
    3) For n = 2023, it's possible to color the edges using 2022 colors without violating the conditions,
       and it's also possible using 2020 colors without violating the conditions.
    4) The difference between the maximum and minimum number of colors that satisfy the conditions is 2. -/
theorem edge_coloring_theorem (n : ℕ) (h : n = 2023) :
  (∀ (coloring : Fin n → Fin n → Fin n), 
    (∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
  (∀ (coloring : Fin n → Fin n → Fin (n-3)), 
    (∃ (i j k l : Fin n), (i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i) ∧ 
      ((coloring i j = coloring j k ∧ coloring j k = coloring k i) ∨
       (coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i)))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2022), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (∃ (coloring : Fin n → Fin n → Fin 2020), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring i k) ∧
      ¬(coloring i j ≠ coloring j k ∧ coloring j k ≠ coloring i k ∧ coloring i j ≠ coloring i k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬(coloring i j = coloring j k ∧ coloring j k = coloring k l ∧ coloring k l = coloring l i))) ∧
  (2022 - 2020 = 2) :=
by sorry


end NUMINAMATH_CALUDE_edge_coloring_theorem_l1499_149977


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l1499_149996

/-- The total distance traveled by a bouncing ball -/
def total_distance (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + initial_height * rebound_ratio + initial_height * rebound_ratio

/-- Theorem: A ball dropped from 100 cm with 50% rebound travels 200 cm when it touches the floor the third time -/
theorem bouncing_ball_distance :
  total_distance 100 0.5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l1499_149996


namespace NUMINAMATH_CALUDE_shifted_function_is_linear_l1499_149929

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  λ x => -2 * x

/-- The result of shifting the original function 3 units to the right -/
def shiftedFunction : ℝ → ℝ :=
  horizontalShift originalFunction 3

theorem shifted_function_is_linear :
  ∃ (k b : ℝ), k ≠ 0 ∧ (∀ x, shiftedFunction x = k * x + b) ∧ k = -2 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_is_linear_l1499_149929


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l1499_149906

theorem logarithmic_equation_solution :
  ∃ x : ℝ, x > 0 ∧ (Real.log x / Real.log 8 + 3 * Real.log (x^2) / Real.log 2 - Real.log x / Real.log 4 = 14) ∧
  x = 2^(12/5) := by
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l1499_149906


namespace NUMINAMATH_CALUDE_planes_parallel_conditions_l1499_149986

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_conditions 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∃ a : Line, perpendicular a α ∧ perpendicular a β → plane_parallel α β) ∧
  (∃ a b : Line, skew a b ∧ contains α a ∧ contains β b ∧ 
    parallel a β ∧ parallel b α → plane_parallel α β) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_conditions_l1499_149986


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1499_149943

/-- Given a principal amount and an interest rate, calculates the total interest after 10 years
    when the principal is trebled after 5 years and the initial 10-year simple interest is 1200. -/
theorem total_interest_calculation (P R : ℝ) : 
  (P * R * 10) / 100 = 1200 → 
  (P * R * 5) / 100 + (3 * P * R * 5) / 100 = 3000 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l1499_149943


namespace NUMINAMATH_CALUDE_initial_profit_percentage_l1499_149903

/-- Proves that given an article with a cost price of Rs. 50, if reducing the cost price by 20% 
    and the selling price by Rs. 10.50 results in a 30% profit, then the initial profit percentage is 25%. -/
theorem initial_profit_percentage 
  (cost : ℝ) 
  (reduced_cost_percentage : ℝ) 
  (reduced_selling_price : ℝ) 
  (new_profit_percentage : ℝ) :
  cost = 50 →
  reduced_cost_percentage = 0.8 →
  reduced_selling_price = 10.5 →
  new_profit_percentage = 0.3 →
  (reduced_cost_percentage * cost * (1 + new_profit_percentage) - reduced_selling_price) / cost * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_l1499_149903


namespace NUMINAMATH_CALUDE_two_point_zero_six_recurring_l1499_149987

def recurring_decimal_02 : ℚ := 2 / 99

theorem two_point_zero_six_recurring (h : recurring_decimal_02 = 2 / 99) :
  2 + 3 * recurring_decimal_02 = 68 / 33 := by
  sorry

end NUMINAMATH_CALUDE_two_point_zero_six_recurring_l1499_149987


namespace NUMINAMATH_CALUDE_product_of_numbers_l1499_149952

theorem product_of_numbers (x y : ℝ) : 
  x + y = 24 → x^2 + y^2 = 404 → x * y = 86 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1499_149952


namespace NUMINAMATH_CALUDE_m_value_l1499_149968

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_m_value_l1499_149968


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1499_149909

theorem arithmetic_operations :
  ((-16) + (-29) = -45) ∧
  ((-10) - 7 = -17) ∧
  (5 * (-2) = -10) ∧
  ((-16) / (-2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1499_149909


namespace NUMINAMATH_CALUDE_horner_method_v3_l1499_149953

def horner_polynomial (x : ℚ) : ℚ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (x : ℚ) : ℚ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

theorem horner_method_v3 :
  horner_v3 (-4) = -49 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1499_149953


namespace NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l1499_149966

/-- Represents a set of data with its variance -/
structure DataSet where
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines stability comparison between two DataSets -/
def more_stable (a b : DataSet) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : DataSet) :
  a.variance < b.variance → more_stable a b :=
by
  sorry

/-- Example datasets A and B -/
def A : DataSet := ⟨0.03, by norm_num⟩
def B : DataSet := ⟨0.13, by norm_num⟩

/-- Theorem stating that A is more stable than B -/
theorem A_more_stable_than_B : more_stable A B :=
by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_A_more_stable_than_B_l1499_149966


namespace NUMINAMATH_CALUDE_anne_travel_distance_l1499_149955

/-- The distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: Anne's travel distance -/
theorem anne_travel_distance :
  let time : ℝ := 3
  let speed : ℝ := 2
  distance time speed = 6 := by sorry

end NUMINAMATH_CALUDE_anne_travel_distance_l1499_149955


namespace NUMINAMATH_CALUDE_smallest_sum_c_d_l1499_149983

theorem smallest_sum_c_d (c d : ℝ) (hc : c > 0) (hd : d > 0)
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (192/81)^(1/3) + (12 * (192/81)^(1/3))^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_c_d_l1499_149983


namespace NUMINAMATH_CALUDE_compute_expression_l1499_149975

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1499_149975


namespace NUMINAMATH_CALUDE_union_of_S_and_T_l1499_149992

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_of_S_and_T : S ∪ T = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_S_and_T_l1499_149992


namespace NUMINAMATH_CALUDE_problem1_l1499_149960

theorem problem1 : Real.sqrt 4 - (1/2)⁻¹ + (2 - 1/7)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l1499_149960


namespace NUMINAMATH_CALUDE_even_odd_property_l1499_149947

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem even_odd_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_odd : is_odd_function (fun x ↦ f (x - 1)))
  (h_f2 : f 2 = 3) :
  f 5 + f 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_even_odd_property_l1499_149947


namespace NUMINAMATH_CALUDE_concentric_circles_track_width_l1499_149934

def track_width (r1 r2 r3 : ℝ) : Prop :=
  2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi ∧
  2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi ∧
  r3 - r1 = 25

theorem concentric_circles_track_width :
  ∀ r1 r2 r3 : ℝ, track_width r1 r2 r3 :=
by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_track_width_l1499_149934


namespace NUMINAMATH_CALUDE_inverse_of_A_l1499_149951

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -3; -2, 1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![-1/2, -3/2; -1, -2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1499_149951


namespace NUMINAMATH_CALUDE_money_problem_l1499_149990

theorem money_problem (a b : ℚ) : 
  a = 80/7 ∧ b = 40/7 →
  7*a + b < 100 ∧ 4*a - b = 40 ∧ b = (1/2) * a := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1499_149990


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1499_149954

theorem no_solution_to_inequality_system :
  ¬ ∃ x : ℝ, (2 * x + 3 ≥ x + 11) ∧ ((2 * x + 5) / 3 - 1 < 2 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l1499_149954


namespace NUMINAMATH_CALUDE_john_text_messages_l1499_149907

theorem john_text_messages 
  (total_messages_per_day : ℕ) 
  (unintended_messages_per_week : ℕ) 
  (days_per_week : ℕ) 
  (h1 : total_messages_per_day = 55) 
  (h2 : unintended_messages_per_week = 245) 
  (h3 : days_per_week = 7) : 
  total_messages_per_day - (unintended_messages_per_week / days_per_week) = 20 := by
sorry

end NUMINAMATH_CALUDE_john_text_messages_l1499_149907


namespace NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l1499_149932

theorem no_equal_roots_for_quadratic :
  ¬ ∃ k : ℝ, ∃ x : ℝ, x^2 - (k + 1) * x + (k - 3) = 0 ∧
    ∀ y : ℝ, y^2 - (k + 1) * y + (k - 3) = 0 → y = x := by
  sorry

end NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l1499_149932


namespace NUMINAMATH_CALUDE_log3_of_9_cubed_l1499_149962

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_of_9_cubed : log3 (9^3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log3_of_9_cubed_l1499_149962


namespace NUMINAMATH_CALUDE_simplify_fraction_l1499_149910

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) (ha2 : a ≠ 2) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = (a - 3) / a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1499_149910


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l1499_149985

/-- Given a sphere with volume 36π cubic inches, its surface area is 36π square inches. -/
theorem sphere_surface_area_from_volume : 
  ∀ (r : ℝ), (4 / 3 : ℝ) * π * r^3 = 36 * π → 4 * π * r^2 = 36 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l1499_149985


namespace NUMINAMATH_CALUDE_distance_between_cities_l1499_149980

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- Yesterday's travel time from A to B in hours -/
def yesterday_time : ℝ := 6

/-- Today's travel time from B to A in hours -/
def today_time : ℝ := 4.5

/-- Time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- Average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  yesterday_time = 6 ∧
  today_time = 4.5 ∧
  (2 * distance) / (yesterday_time + today_time - 2 * time_saved) = average_speed :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1499_149980


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1499_149915

theorem complex_number_quadrant : 
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1499_149915


namespace NUMINAMATH_CALUDE_raft_capacity_l1499_149995

theorem raft_capacity (total_capacity : ℕ) (reduction_with_jackets : ℕ) (people_needing_jackets : ℕ) : 
  total_capacity = 21 → 
  reduction_with_jackets = 7 → 
  people_needing_jackets = 8 → 
  (total_capacity - (reduction_with_jackets * people_needing_jackets / (total_capacity - reduction_with_jackets))) = 17 := by
sorry

end NUMINAMATH_CALUDE_raft_capacity_l1499_149995


namespace NUMINAMATH_CALUDE_inequality_implies_zero_for_nonpositive_l1499_149936

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem: if f satisfies the inequality, then f(x) = 0 for all x ≤ 0 -/
theorem inequality_implies_zero_for_nonpositive
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_zero_for_nonpositive_l1499_149936


namespace NUMINAMATH_CALUDE_inequality_proof_l1499_149925

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x) : 
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 ≥ 4 / (x*y + y*z + z*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1499_149925


namespace NUMINAMATH_CALUDE_max_regions_circular_disk_l1499_149969

/-- 
Given a circular disk divided by 2n equally spaced radii (n > 0) and one chord,
the maximum number of non-overlapping regions is 3n + 1.
-/
theorem max_regions_circular_disk (n : ℕ) (h : n > 0) : 
  ∃ (num_regions : ℕ), num_regions = 3 * n + 1 ∧ 
  (∀ (m : ℕ), m ≤ num_regions) := by
sorry

end NUMINAMATH_CALUDE_max_regions_circular_disk_l1499_149969


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l1499_149924

theorem mean_of_three_numbers (p q r : ℝ) : 
  (p + q) / 2 = 13 →
  (q + r) / 2 = 16 →
  (r + p) / 2 = 7 →
  (p + q + r) / 3 = 12 := by
sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l1499_149924


namespace NUMINAMATH_CALUDE_sleep_variance_proof_l1499_149942

def sleep_data : List ℝ := [6, 6, 7, 6, 7, 8, 9]

theorem sleep_variance_proof :
  let n : ℕ := sleep_data.length
  let mean : ℝ := (sleep_data.sum) / n
  let variance : ℝ := (sleep_data.map (λ x => (x - mean)^2)).sum / n
  mean = 7 → variance = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_sleep_variance_proof_l1499_149942


namespace NUMINAMATH_CALUDE_supermarket_spending_l1499_149978

theorem supermarket_spending (total_spent : ℚ) 
  (h1 : total_spent = 150)
  (h2 : ∃ (fruits_veg meat bakery candy : ℚ),
    fruits_veg = 1/2 * total_spent ∧
    meat = 1/3 * total_spent ∧
    candy = 10 ∧
    fruits_veg + meat + bakery + candy = total_spent) :
  ∃ (bakery : ℚ), bakery = 1/10 * total_spent := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1499_149978


namespace NUMINAMATH_CALUDE_min_balls_for_three_same_color_l1499_149945

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : Nat
  black : Nat
  blue : Nat

/-- Calculates the minimum number of balls to draw to ensure at least three of the same color -/
def minBallsToEnsureThreeSameColor (bag : BagContents) : Nat :=
  7

/-- Theorem stating that for a bag with 5 white, 5 black, and 2 blue balls,
    the minimum number of balls to draw to ensure at least three of the same color is 7 -/
theorem min_balls_for_three_same_color :
  let bag : BagContents := { white := 5, black := 5, blue := 2 }
  minBallsToEnsureThreeSameColor bag = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_same_color_l1499_149945


namespace NUMINAMATH_CALUDE_piano_harmonies_count_l1499_149928

theorem piano_harmonies_count : 
  (Nat.choose 7 3) + (Nat.choose 7 4) + (Nat.choose 7 5) + (Nat.choose 7 6) + (Nat.choose 7 7) = 99 := by
  sorry

end NUMINAMATH_CALUDE_piano_harmonies_count_l1499_149928


namespace NUMINAMATH_CALUDE_square_circumcircle_integer_points_l1499_149956

/-- The circumcircle of a square with side length 1978 contains no integer points other than the vertices of the square. -/
theorem square_circumcircle_integer_points :
  ∀ x y : ℤ,
  (x - 989)^2 + (y - 989)^2 = 2 * 989^2 →
  (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 1978) ∨ (x = 1978 ∧ y = 0) ∨ (x = 1978 ∧ y = 1978) :=
by sorry


end NUMINAMATH_CALUDE_square_circumcircle_integer_points_l1499_149956


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1499_149922

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1499_149922


namespace NUMINAMATH_CALUDE_inequality_sum_l1499_149957

theorem inequality_sum (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : c ≠ 0) (h6 : d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_sum_l1499_149957


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l1499_149964

theorem quadratic_two_roots (m : ℝ) (h : m < (1 : ℝ) / 4) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + m = 0 ∧ x₂^2 - x₂ + m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l1499_149964
