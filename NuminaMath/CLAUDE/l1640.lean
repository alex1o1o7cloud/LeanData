import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_equation_simplification_l1640_164062

theorem absolute_value_equation_simplification
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, |5*x - 4| + a ≠ 0)
  (h2 : ∃ x y : ℝ, x ≠ y ∧ |4*x - 3| + b = 0 ∧ |4*y - 3| + b = 0)
  (h3 : ∃! x : ℝ, |3*x - 2| + c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_simplification_l1640_164062


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l1640_164064

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l1640_164064


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_fourth_powers_l1640_164025

theorem unique_prime_sum_of_fourth_powers (p a b c : ℕ) : 
  Prime p ∧ Prime a ∧ Prime b ∧ Prime c ∧ p = a^4 + b^4 + c^4 - 3 → p = 719 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_fourth_powers_l1640_164025


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l1640_164073

theorem cubic_roots_relation (p q r : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 4*x^2 + 5*x - 14 = (x - p) * (x - q) * (x - r)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 34 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l1640_164073


namespace NUMINAMATH_CALUDE_one_third_circle_equals_167_l1640_164029

/-- The number of clerts in a full circle on Mars. -/
def full_circle_clerts : ℕ := 500

/-- The number of clerts in one-third of a full circle on Mars. -/
def one_third_circle_clerts : ℕ := 167

/-- Theorem stating that one-third of a full circle in clerts is equal to 167. -/
theorem one_third_circle_equals_167 :
  (full_circle_clerts : ℚ) / 3 = one_third_circle_clerts := by sorry

end NUMINAMATH_CALUDE_one_third_circle_equals_167_l1640_164029


namespace NUMINAMATH_CALUDE_company_sales_royalties_l1640_164043

/-- A company's sales and royalties problem -/
theorem company_sales_royalties
  (initial_sales : ℝ)
  (initial_royalties : ℝ)
  (next_royalties : ℝ)
  (royalty_ratio_decrease : ℝ)
  (h1 : initial_sales = 10000000)
  (h2 : initial_royalties = 2000000)
  (h3 : next_royalties = 8000000)
  (h4 : royalty_ratio_decrease = 0.6)
  : ∃ (next_sales : ℝ), next_sales = 100000000 ∧
    next_royalties / next_sales = (initial_royalties / initial_sales) * (1 - royalty_ratio_decrease) :=
by sorry

end NUMINAMATH_CALUDE_company_sales_royalties_l1640_164043


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1640_164061

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1640_164061


namespace NUMINAMATH_CALUDE_v_2015_equals_2_l1640_164019

/-- Function g as defined in the problem -/
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

/-- Sequence v defined recursively -/
def v : ℕ → ℕ
| 0 => 3
| n + 1 => g (v n)

/-- Theorem stating that the 2015th term of the sequence is 2 -/
theorem v_2015_equals_2 : v 2015 = 2 := by
  sorry

end NUMINAMATH_CALUDE_v_2015_equals_2_l1640_164019


namespace NUMINAMATH_CALUDE_greatest_b_value_l1640_164041

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1640_164041


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l1640_164017

theorem largest_prime_divisor_to_test (n : ℕ) (h : 800 ≤ n ∧ n ≤ 850) :
  (∀ p : ℕ, p ≤ 29 → Prime p → ¬(p ∣ n)) → Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l1640_164017


namespace NUMINAMATH_CALUDE_factorial_division_l1640_164000

theorem factorial_division :
  (10 : ℕ).factorial / (5 : ℕ).factorial = 30240 :=
by
  -- Given: 10! = 3628800
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  
  -- Definition of 5!
  have h2 : (5 : ℕ).factorial = 120 := by sorry
  
  -- Proof that 10! / 5! = 30240
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1640_164000


namespace NUMINAMATH_CALUDE_base_eight_23456_equals_10030_l1640_164087

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_23456_equals_10030 :
  base_eight_to_ten [6, 5, 4, 3, 2] = 10030 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_23456_equals_10030_l1640_164087


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1640_164021

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 1) + 3 / (1 - x) = 1) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1640_164021


namespace NUMINAMATH_CALUDE_custom_op_nested_result_l1640_164036

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - 3*x*y

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem custom_op_nested_result (h : ℝ) : custom_op h (custom_op h h) = h^3 * (10 - 3*h) := by
  sorry

end NUMINAMATH_CALUDE_custom_op_nested_result_l1640_164036


namespace NUMINAMATH_CALUDE_parabola_b_value_l1640_164093

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- The y-intercept of a parabola -/
def yIntercept (p : Parabola) : ℝ := p.c

theorem parabola_b_value (p : Parabola) (h k : ℝ) :
  h > 0 ∧ k > 0 ∧
  vertex p = (h, k) ∧
  yIntercept p = -k ∧
  k = 2 * h →
  p.b = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_b_value_l1640_164093


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l1640_164070

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 8 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l1640_164070


namespace NUMINAMATH_CALUDE_select_team_count_l1640_164079

/-- The number of ways to select a team of 7 people from a group of 7 boys and 9 girls, 
    with at least 3 girls in the team -/
def selectTeam (numBoys numGirls teamSize minGirls : ℕ) : ℕ :=
  (Finset.range (teamSize - minGirls + 1)).sum fun i =>
    Nat.choose numGirls (minGirls + i) * Nat.choose numBoys (teamSize - minGirls - i)

/-- Theorem stating that the number of ways to select the team is 10620 -/
theorem select_team_count : selectTeam 7 9 7 3 = 10620 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l1640_164079


namespace NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_cubes_l1640_164010

theorem max_sum_given_sum_squares_and_cubes :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 7 → x^3 + y^3 = 10 →
  x + y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_cubes_l1640_164010


namespace NUMINAMATH_CALUDE_project_completion_time_l1640_164022

/-- Represents the number of days it takes to complete the project. -/
def total_days : ℕ := 21

/-- Represents the rate at which A completes the project per day. -/
def rate_A : ℚ := 1 / 20

/-- Represents the rate at which B completes the project per day. -/
def rate_B : ℚ := 1 / 30

/-- Represents the combined rate at which A and B complete the project per day when working together. -/
def combined_rate : ℚ := rate_A + rate_B

theorem project_completion_time (x : ℕ) :
  (↑(total_days - x) * combined_rate + ↑x * rate_B = 1) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l1640_164022


namespace NUMINAMATH_CALUDE_quadrilateral_rod_count_l1640_164072

theorem quadrilateral_rod_count : 
  let a : ℕ := 5
  let b : ℕ := 12
  let c : ℕ := 20
  let valid_length (d : ℕ) : Prop := 
    1 ≤ d ∧ d ≤ 40 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a
  (Finset.filter valid_length (Finset.range 41)).card = 30 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_rod_count_l1640_164072


namespace NUMINAMATH_CALUDE_dress_final_cost_l1640_164033

/-- The final cost of a dress after applying a discount --/
theorem dress_final_cost (original_price discount_percentage : ℚ) 
  (h1 : original_price = 50)
  (h2 : discount_percentage = 30) :
  original_price * (1 - discount_percentage / 100) = 35 := by
  sorry

#check dress_final_cost

end NUMINAMATH_CALUDE_dress_final_cost_l1640_164033


namespace NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_seven_l1640_164009

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem two_digit_perfect_square_divisible_by_seven :
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ n % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_two_digit_perfect_square_divisible_by_seven_l1640_164009


namespace NUMINAMATH_CALUDE_luke_stars_made_l1640_164006

-- Define the given conditions
def stars_per_jar : ℕ := 85
def bottles_to_fill : ℕ := 4
def additional_stars_needed : ℕ := 307

-- Define the theorem
theorem luke_stars_made : 
  (stars_per_jar * bottles_to_fill) - additional_stars_needed = 33 := by
  sorry

end NUMINAMATH_CALUDE_luke_stars_made_l1640_164006


namespace NUMINAMATH_CALUDE_triangle_inequality_iff_squared_sum_l1640_164027

theorem triangle_inequality_iff_squared_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_iff_squared_sum_l1640_164027


namespace NUMINAMATH_CALUDE_farmers_extra_days_l1640_164081

/-- A farmer's ploughing problem -/
theorem farmers_extra_days
  (total_area : ℕ)
  (planned_daily_area : ℕ)
  (actual_daily_area : ℕ)
  (area_left : ℕ)
  (h1 : total_area = 720)
  (h2 : planned_daily_area = 120)
  (h3 : actual_daily_area = 85)
  (h4 : area_left = 40) :
  ∃ extra_days : ℕ,
    actual_daily_area * (total_area / planned_daily_area + extra_days) = total_area - area_left ∧
    extra_days = 2 := by
  sorry

#check farmers_extra_days

end NUMINAMATH_CALUDE_farmers_extra_days_l1640_164081


namespace NUMINAMATH_CALUDE_topsoil_discounted_cost_l1640_164034

-- Define constants
def price_per_cubic_foot : ℝ := 7
def purchase_volume_yards : ℝ := 10
def discount_threshold : ℝ := 200
def discount_rate : ℝ := 0.05

-- Define conversion factor
def cubic_yards_to_cubic_feet : ℝ := 27

-- Theorem statement
theorem topsoil_discounted_cost :
  let volume_feet := purchase_volume_yards * cubic_yards_to_cubic_feet
  let base_cost := volume_feet * price_per_cubic_foot
  let discounted_cost := if volume_feet > discount_threshold
                         then base_cost * (1 - discount_rate)
                         else base_cost
  discounted_cost = 1795.50 := by
sorry

end NUMINAMATH_CALUDE_topsoil_discounted_cost_l1640_164034


namespace NUMINAMATH_CALUDE_one_percent_as_decimal_l1640_164089

theorem one_percent_as_decimal : (1 : ℚ) / 100 = (1 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_one_percent_as_decimal_l1640_164089


namespace NUMINAMATH_CALUDE_jerrys_roofing_problem_l1640_164058

/-- Proves that the length of the other side of a rectangular roof is 40 feet
    given the conditions of Jerry's roofing project. -/
theorem jerrys_roofing_problem (
  num_roofs : ℕ)
  (known_side_length : ℝ)
  (shingles_per_sqft : ℕ)
  (total_shingles : ℕ)
  (h1 : num_roofs = 3)
  (h2 : known_side_length = 20)
  (h3 : shingles_per_sqft = 8)
  (h4 : total_shingles = 38400)
  : ∃ (other_side_length : ℝ), other_side_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_roofing_problem_l1640_164058


namespace NUMINAMATH_CALUDE_broadcasting_methods_count_l1640_164037

/-- The number of different commercial advertisements -/
def num_commercial : ℕ := 3

/-- The number of different Olympic promotional advertisements -/
def num_olympic : ℕ := 2

/-- The total number of advertisements -/
def total_ads : ℕ := 5

/-- Function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem broadcasting_methods_count :
  num_broadcasting_methods = 36 :=
sorry

end NUMINAMATH_CALUDE_broadcasting_methods_count_l1640_164037


namespace NUMINAMATH_CALUDE_range_of_f_is_real_l1640_164095

noncomputable def f (x : ℝ) := x^3 - 3*x

theorem range_of_f_is_real : 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (f 2 = 2) ∧ 
  (deriv f 2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_is_real_l1640_164095


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l1640_164065

/-- Parabola C: y² = 4x -/
def C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Line with slope k passing through the focus -/
def line (k x : ℝ) : ℝ := k*(x - focus.1)

/-- Intersection points of the line and the parabola -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C p.1 p.2 ∧ p.2 = line k p.1}

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_intersection_slope (k : ℝ) :
  (∃ A B, A ∈ intersectionPoints k ∧ B ∈ intersectionPoints k ∧ A ≠ B ∧
    dot_product (vector_MP A) (vector_MP B) = 0) →
  k = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l1640_164065


namespace NUMINAMATH_CALUDE_room_length_proof_l1640_164086

theorem room_length_proof (L : ℝ) : 
  L > 0 → -- Ensure length is positive
  ((L + 4) * 16 - L * 12 = 136) → -- Area of veranda equation
  L = 18 := by
sorry

end NUMINAMATH_CALUDE_room_length_proof_l1640_164086


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l1640_164057

/-- A random variable following a binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type :=
  Unit

/-- The expectation of a binomial distribution -/
def expectation (X : binomial_distribution n p) : ℝ :=
  n * p

/-- The variance of a binomial distribution -/
def variance (X : binomial_distribution n p) : ℝ :=
  n * p * (1 - p)

theorem binomial_expectation_and_variance :
  ∀ (X : binomial_distribution 10 (3/5)),
    expectation X = 6 ∧ variance X = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l1640_164057


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l1640_164092

-- Define the lines and conditions
def l1 (x y m : ℝ) : Prop := x + y - 3*m = 0
def l2 (x y m : ℝ) : Prop := 2*x - y + 2*m - 1 = 0
def y_intercept_l1 : ℝ := 3

-- Define the theorem
theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ),
    (∀ m : ℝ, l1 x y m ∧ l2 x y m) ∧
    x = 2/3 ∧ y = 7/3 ∧
    ∃ (k : ℝ), 3*x + 6*y + k = 0 ∧
    ∀ (x' y' : ℝ), l2 x' y' 1 → (x' - 2/3) + 2*(y' - 7/3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l1640_164092


namespace NUMINAMATH_CALUDE_division_zero_implies_divisor_greater_l1640_164059

theorem division_zero_implies_divisor_greater (d : ℕ) :
  2016 / d = 0 → d > 2016 := by
sorry

end NUMINAMATH_CALUDE_division_zero_implies_divisor_greater_l1640_164059


namespace NUMINAMATH_CALUDE_remaining_sweets_theorem_l1640_164044

/-- The number of remaining sweets after Aaron's actions -/
def remaining_sweets (C S P R L : ℕ) : ℕ :=
  let eaten_C := (2 * C) / 5
  let eaten_S := S / 4
  let eaten_P := (3 * P) / 5
  let given_C := (C - P / 4) / 3
  let discarded_R := (3 * R) / 2
  let eaten_L := (eaten_S * 6) / 5
  (C - eaten_C - given_C) + (S - eaten_S) + (P - eaten_P) + (if R > discarded_R then R - discarded_R else 0) + (L - eaten_L)

theorem remaining_sweets_theorem :
  remaining_sweets 30 100 60 25 150 = 232 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sweets_theorem_l1640_164044


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1640_164013

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1640_164013


namespace NUMINAMATH_CALUDE_scientific_notation_103000000_l1640_164045

theorem scientific_notation_103000000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 103000000 = a * (10 : ℝ) ^ n ∧ a = 1.03 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_103000000_l1640_164045


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l1640_164068

/-- A positive geometric progression -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ q : ℝ), q > 0 ∧ ∀ n, a n = a₁ * q ^ (n - 1)

/-- An arithmetic progression -/
def arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ d : ℝ), ∀ n, b n = b₁ + (n - 1) * d

theorem geometric_arithmetic_inequality (a b : ℕ → ℝ) :
  geometric_progression a →
  arithmetic_progression b →
  (∀ n, a n > 0) →
  a 6 = b 7 →
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l1640_164068


namespace NUMINAMATH_CALUDE_blaine_win_probability_l1640_164020

/-- Probability of Amelia getting heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine getting heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Charlie getting heads -/
def p_charlie : ℚ := 1/5

/-- The probability that Blaine wins the game -/
def p_blaine_wins : ℚ := 25/36

theorem blaine_win_probability :
  let p_cycle : ℚ := (1 - p_amelia) * (1 - p_blaine) * (1 - p_charlie)
  p_blaine_wins = (1 - p_amelia) * p_blaine / (1 - p_cycle) := by
  sorry

end NUMINAMATH_CALUDE_blaine_win_probability_l1640_164020


namespace NUMINAMATH_CALUDE_picture_area_calculation_l1640_164097

/-- Given a sheet of paper with width, length, and margin, calculates the area of the picture. -/
def picture_area (paper_width paper_length margin : ℝ) : ℝ :=
  (paper_width - 2 * margin) * (paper_length - 2 * margin)

/-- Theorem stating that for a paper of 8.5 by 10 inches with a 1.5-inch margin, 
    the picture area is 38.5 square inches. -/
theorem picture_area_calculation :
  picture_area 8.5 10 1.5 = 38.5 := by
  sorry

#eval picture_area 8.5 10 1.5

end NUMINAMATH_CALUDE_picture_area_calculation_l1640_164097


namespace NUMINAMATH_CALUDE_correct_raisin_distribution_l1640_164049

/-- The number of raisins received by each person -/
structure RaisinDistribution where
  bryce : ℕ
  carter : ℕ
  alice : ℕ

/-- The conditions of the raisin distribution problem -/
def valid_distribution (d : RaisinDistribution) : Prop :=
  d.bryce = d.carter + 10 ∧
  d.carter = d.bryce / 2 ∧
  d.alice = 2 * d.carter

/-- The theorem stating the correct raisin distribution -/
theorem correct_raisin_distribution :
  ∃ (d : RaisinDistribution), valid_distribution d ∧ d.bryce = 20 ∧ d.carter = 10 ∧ d.alice = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_raisin_distribution_l1640_164049


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l1640_164083

theorem hyperbola_standard_form (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 2 * a = 8) (h4 : (a^2 + b^2) / a^2 = (5/4)^2) :
  a = 4 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l1640_164083


namespace NUMINAMATH_CALUDE_inequality_proof_l1640_164024

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31/8)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1640_164024


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l1640_164053

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (data : Fin n → ℝ) : ℝ := sorry

def transform (data : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * data i + 1

theorem variance_of_transformed_data 
  (h : variance x = 2) : 
  variance (transform x) = 18 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l1640_164053


namespace NUMINAMATH_CALUDE_f_2020_is_sin_l1640_164004

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2020_is_sin : f 2020 = Real.sin := by
  sorry

end NUMINAMATH_CALUDE_f_2020_is_sin_l1640_164004


namespace NUMINAMATH_CALUDE_conference_handshakes_l1640_164015

/-- The number of people at the conference -/
def n : ℕ := 27

/-- The number of people who don't shake hands with each other -/
def k : ℕ := 3

/-- The maximum number of handshakes possible under the given conditions -/
def max_handshakes : ℕ := n.choose 2 - k.choose 2

/-- Theorem stating the maximum number of handshakes at the conference -/
theorem conference_handshakes :
  max_handshakes = 348 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1640_164015


namespace NUMINAMATH_CALUDE_conjecture_counterexample_l1640_164076

theorem conjecture_counterexample : ∃ n : ℕ, 
  (n % 2 = 1 ∧ n > 5) ∧ 
  ¬∃ (p k : ℕ), Prime p ∧ n = p + 2 * k^2 :=
sorry

end NUMINAMATH_CALUDE_conjecture_counterexample_l1640_164076


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l1640_164091

/-- Given a ratio of blue to white paint and an amount of white paint, 
    calculate the amount of blue paint required. -/
def blue_paint_amount (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the blue paint amount is 12 quarts. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 4
  let white_ratio : ℚ := 5
  let white_amount : ℚ := 15
  blue_paint_amount blue_ratio white_ratio white_amount = 12 := by
sorry

#eval blue_paint_amount 4 5 15

end NUMINAMATH_CALUDE_blue_paint_calculation_l1640_164091


namespace NUMINAMATH_CALUDE_x_range_l1640_164096

theorem x_range (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) :
  x ∈ Set.Icc 1 (5/3) :=
sorry

end NUMINAMATH_CALUDE_x_range_l1640_164096


namespace NUMINAMATH_CALUDE_one_plus_i_fourth_power_l1640_164014

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The fourth power of (1 + i) equals -4 -/
theorem one_plus_i_fourth_power : (1 + i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_one_plus_i_fourth_power_l1640_164014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1640_164005

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₂ = 9 and S₄ = 22, S₈ = 60 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h₂ : seq.S 2 = 9)
    (h₄ : seq.S 4 = 22) :
    seq.S 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1640_164005


namespace NUMINAMATH_CALUDE_log_inequality_l1640_164040

theorem log_inequality (a : ℝ) (h : 0 < a ∧ a < 1/4) :
  ∀ x : ℝ, (0 < x ∧ x ≠ 1 ∧ x + a > 0 ∧ x + a ≠ 1) →
  (Real.log 2 / Real.log (x + a) < Real.log 4 / Real.log x ↔
    (0 < x ∧ x < 1/2 - a - Real.sqrt (1/4 - a)) ∨
    (1/2 - a + Real.sqrt (1/4 - a) < x ∧ x < 1 - a) ∨
    (1 < x)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l1640_164040


namespace NUMINAMATH_CALUDE_expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1640_164063

/-- The expression is not defined when x = 10 -/
theorem expression_undefined_at_ten : 
  ∀ x : ℝ, x = 10 → (x^3 - 30*x^2 + 300*x - 1000 = 0) := by
  sorry

/-- The denominator of the expression -/
def denominator (x : ℝ) : ℝ := x^3 - 30*x^2 + 300*x - 1000

/-- The expression is undefined when the denominator is zero -/
theorem expression_undefined_when_denominator_zero (x : ℝ) : 
  denominator x = 0 → ¬∃y : ℝ, y = (3*x^4 + 2*x + 6) / (x^3 - 30*x^2 + 300*x - 1000) := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1640_164063


namespace NUMINAMATH_CALUDE_checkers_inequality_l1640_164028

theorem checkers_inequality (n : ℕ) (A B : ℕ) : A ≤ 3 * B :=
  by
  -- Assume n is the number of black checkers (equal to the number of white checkers)
  -- A is the number of triples with white majority
  -- B is the number of triples with black majority
  sorry

end NUMINAMATH_CALUDE_checkers_inequality_l1640_164028


namespace NUMINAMATH_CALUDE_exam_results_l1640_164048

/-- Represents a student in the autonomous recruitment exam -/
structure Student where
  writtenProb : ℝ  -- Probability of passing the written exam
  oralProb : ℝ     -- Probability of passing the oral exam

/-- The autonomous recruitment exam setup -/
def ExamSetup : (Student × Student × Student) :=
  (⟨0.6, 0.5⟩, ⟨0.5, 0.6⟩, ⟨0.4, 0.75⟩)

/-- Calculates the probability of exactly one student passing the written exam -/
noncomputable def probExactlyOnePassWritten (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Calculates the expected number of pre-admitted students -/
noncomputable def expectedPreAdmitted (setup : Student × Student × Student) : ℝ :=
  sorry

/-- Main theorem stating the results of the calculations -/
theorem exam_results :
  let setup := ExamSetup
  probExactlyOnePassWritten setup = 0.38 ∧
  expectedPreAdmitted setup = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1640_164048


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1640_164078

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line
def Line (k m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + m}

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from P to Q
def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

-- Define the squared magnitude of a vector
def magnitude_squared (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem ellipse_line_intersection :
  ∃ (k m : ℝ), 
    let C := Ellipse 2 (Real.sqrt 3)
    let l := Line k m
    let P := (2, 1)
    let M := (1, 3/2)
    (∀ x y, (x, y) ∈ C → (x^2 / 4) + (y^2 / 3) = 1) ∧ 
    (1, 3/2) ∈ C ∧
    (2, 1) ∈ l ∧
    (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B ∧
      dot_product (vector P A) (vector P B) = magnitude_squared (vector P M)) ∧
    k = 1/2 ∧ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1640_164078


namespace NUMINAMATH_CALUDE_series_sum_l1640_164026

theorem series_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hc_def : c = a - b) :
  (∑' n : ℕ, 1 / (n * c * ((n + 1) * c))) = 1 / (b * c) :=
sorry

end NUMINAMATH_CALUDE_series_sum_l1640_164026


namespace NUMINAMATH_CALUDE_soda_cost_l1640_164039

/-- Given the total cost of an order and the cost of sandwiches, 
    calculate the cost of each soda. -/
theorem soda_cost (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = 10.46)
  (h2 : sandwich_cost = 3.49)
  (h3 : 2 * sandwich_cost + 4 * (total_cost - 2 * sandwich_cost) / 4 = total_cost) :
  (total_cost - 2 * sandwich_cost) / 4 = 0.87 := by sorry

end NUMINAMATH_CALUDE_soda_cost_l1640_164039


namespace NUMINAMATH_CALUDE_number_of_hens_l1640_164071

def number_of_goats : ℕ := 5
def total_cost : ℕ := 2500
def price_of_hen : ℕ := 50
def price_of_goat : ℕ := 400

theorem number_of_hens : 
  ∃ (h : ℕ), h * price_of_hen + number_of_goats * price_of_goat = total_cost ∧ h = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l1640_164071


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1640_164082

/-- Given a geometric sequence {a_n} with a_3 * a_7 = 8 and a_4 + a_6 = 6, prove that a_2 + a_8 = 9 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 3 * a 7 = 8) (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1640_164082


namespace NUMINAMATH_CALUDE_percentage_more_than_6_years_is_21_875_l1640_164077

/-- Represents the employee tenure distribution of a company -/
structure EmployeeTenure where
  less_than_3_years : ℕ
  between_3_and_6_years : ℕ
  more_than_6_years : ℕ

/-- Calculates the percentage of employees who have worked for more than 6 years -/
def percentage_more_than_6_years (e : EmployeeTenure) : ℚ :=
  (e.more_than_6_years : ℚ) / (e.less_than_3_years + e.between_3_and_6_years + e.more_than_6_years) * 100

/-- Proves that the percentage of employees who have worked for more than 6 years is 21.875% -/
theorem percentage_more_than_6_years_is_21_875 (e : EmployeeTenure) 
  (h : ∃ (x : ℕ), e.less_than_3_years = 10 * x ∧ 
                   e.between_3_and_6_years = 15 * x ∧ 
                   e.more_than_6_years = 7 * x) : 
  percentage_more_than_6_years e = 21875 / 1000 := by
  sorry

#eval (21875 : ℚ) / 1000  -- To verify that 21875/1000 = 21.875

end NUMINAMATH_CALUDE_percentage_more_than_6_years_is_21_875_l1640_164077


namespace NUMINAMATH_CALUDE_paint_theorem_l1640_164069

def paint_problem (initial_amount : ℚ) : Prop :=
  let first_day_remaining := initial_amount - (1/2 * initial_amount)
  let second_day_remaining := first_day_remaining - (1/4 * first_day_remaining)
  let third_day_remaining := second_day_remaining - (1/3 * second_day_remaining)
  third_day_remaining = 1/4 * initial_amount

theorem paint_theorem : paint_problem 1 := by
  sorry

end NUMINAMATH_CALUDE_paint_theorem_l1640_164069


namespace NUMINAMATH_CALUDE_ceva_theorem_l1640_164018

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (X Y Z : Point)

/-- Represents a line segment -/
structure LineSegment :=
  (A B : Point)

/-- Represents the intersection point of three lines -/
def intersectionPoint (l1 l2 l3 : LineSegment) : Point := sorry

/-- Calculates the ratio of distances from a point to two other points -/
def distanceRatio (P A B : Point) : ℝ := sorry

theorem ceva_theorem (T : Triangle) (X' Y' Z' : Point) (P : Point) :
  let XX' := LineSegment.mk T.X X'
  let YY' := LineSegment.mk T.Y Y'
  let ZZ' := LineSegment.mk T.Z Z'
  P = intersectionPoint XX' YY' ZZ' →
  (distanceRatio P T.X X' + distanceRatio P T.Y Y' + distanceRatio P T.Z Z' = 100) →
  (distanceRatio P T.X X' * distanceRatio P T.Y Y' * distanceRatio P T.Z Z' = 102) := by
  sorry

end NUMINAMATH_CALUDE_ceva_theorem_l1640_164018


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l1640_164074

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_value (x : ℝ) :
  is_pure_imaginary ((x^2 - 1 : ℝ) + (x^2 + 3*x + 2 : ℝ) * I) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l1640_164074


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l1640_164055

theorem milford_lake_algae_increase (original_algae current_algae : ℕ) 
  (h1 : original_algae = 809)
  (h2 : current_algae = 3263) : 
  current_algae - original_algae = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l1640_164055


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_sum_of_divisors_450_l1640_164085

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 450 is 3 -/
theorem distinct_prime_factors_of_sum_of_divisors_450 :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_sum_of_divisors_450_l1640_164085


namespace NUMINAMATH_CALUDE_polynomial_square_root_l1640_164056

theorem polynomial_square_root (a b c : ℝ) :
  (2 * a - 3 * b + 4 * c)^2 = 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_root_l1640_164056


namespace NUMINAMATH_CALUDE_equation_solution_l1640_164047

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 16))) = 55 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1640_164047


namespace NUMINAMATH_CALUDE_max_result_ahn_max_result_ahn_achievable_l1640_164088

theorem max_result_ahn (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 999) → 3 * (200 + n) ≤ 3597 := by
  sorry

theorem max_result_ahn_achievable : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (200 + n) = 3597 := by
  sorry

end NUMINAMATH_CALUDE_max_result_ahn_max_result_ahn_achievable_l1640_164088


namespace NUMINAMATH_CALUDE_opposite_side_length_l1640_164054

/-- Represents a right triangle with one acute angle of 30 degrees and hypotenuse of 10 units -/
structure RightTriangle30 where
  -- The hypotenuse length is 10 units
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = 10
  -- One acute angle is 30 degrees (π/6 radians)
  acute_angle : ℝ
  acute_angle_eq : acute_angle = π/6

/-- 
Theorem: In a right triangle with one acute angle of 30° and a hypotenuse of 10 units, 
the length of the side opposite to the 30° angle is 5 units.
-/
theorem opposite_side_length (t : RightTriangle30) : 
  Real.sin t.acute_angle * t.hypotenuse = 5 := by
  sorry


end NUMINAMATH_CALUDE_opposite_side_length_l1640_164054


namespace NUMINAMATH_CALUDE_correct_arrangements_l1640_164001

/-- The number of ways to assign students to tasks with restrictions -/
def assignment_arrangements (n m : ℕ) (r : ℕ) : ℕ :=
  Nat.descFactorial n m - 2 * Nat.descFactorial (n - 1) (m - 1)

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  assignment_arrangements 6 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangements_l1640_164001


namespace NUMINAMATH_CALUDE_married_men_fraction_l1640_164099

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : total_women > 0) :
  let single_women := (1 : ℚ) / 4 * total_women
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  married_men / total_people = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1640_164099


namespace NUMINAMATH_CALUDE_acid_dilution_l1640_164094

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution yields a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration := by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l1640_164094


namespace NUMINAMATH_CALUDE_softball_team_size_l1640_164084

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 5555555555555556 / 10000000000000000 →
  men + women = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l1640_164084


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_3_l1640_164052

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

-- Theorem statement
theorem horner_method_for_f_at_3 : f 3 = 283 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_3_l1640_164052


namespace NUMINAMATH_CALUDE_cost_per_meter_l1640_164011

def total_cost : ℝ := 416.25
def total_length : ℝ := 9.25

theorem cost_per_meter : total_cost / total_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_meter_l1640_164011


namespace NUMINAMATH_CALUDE_tully_kate_age_ratio_l1640_164035

/-- Represents a person's age --/
structure Person where
  name : String
  current_age : ℕ

/-- Calculates the ratio of two numbers --/
def ratio (a b : ℕ) : ℚ := a / b

theorem tully_kate_age_ratio :
  let tully : Person := { name := "Tully", current_age := 61 }
  let kate : Person := { name := "Kate", current_age := 29 }
  let tully_future_age := tully.current_age + 3
  let kate_future_age := kate.current_age + 3
  ratio tully_future_age kate_future_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_tully_kate_age_ratio_l1640_164035


namespace NUMINAMATH_CALUDE_seventh_graders_count_l1640_164098

/-- The number of fifth graders going on the trip -/
def fifth_graders : ℕ := 109

/-- The number of sixth graders going on the trip -/
def sixth_graders : ℕ := 115

/-- The number of teachers chaperoning -/
def teachers : ℕ := 4

/-- The number of parent chaperones per grade -/
def parents_per_grade : ℕ := 2

/-- The number of grades participating -/
def grades : ℕ := 3

/-- The number of buses for the trip -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- Theorem: The number of seventh graders going on the trip is 118 -/
theorem seventh_graders_count : 
  (buses * seats_per_bus) - 
  (fifth_graders + sixth_graders + (teachers + parents_per_grade * grades)) = 118 := by
  sorry

end NUMINAMATH_CALUDE_seventh_graders_count_l1640_164098


namespace NUMINAMATH_CALUDE_no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l1640_164060

theorem no_integer_triangle_with_integer_altitudes_and_perimeter_1995 :
  ¬ ∃ (a b c h_a h_b h_c : ℕ), 
    (a + b + c = 1995) ∧ 
    (h_a^2 * (4*a^2) = 2*a^2*b^2 + 2*a^2*c^2 + 2*c^2*b^2 - a^4 - b^4 - c^4) ∧
    (h_b^2 * (4*b^2) = 2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 - a^4 - b^4 - c^4) ∧
    (h_c^2 * (4*c^2) = 2*a^2*c^2 + 2*b^2*c^2 + 2*a^2*b^2 - a^4 - b^4 - c^4) :=
by sorry


end NUMINAMATH_CALUDE_no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l1640_164060


namespace NUMINAMATH_CALUDE_new_gross_profit_percentage_l1640_164007

theorem new_gross_profit_percentage
  (old_selling_price : ℝ)
  (old_gross_profit_percentage : ℝ)
  (new_selling_price : ℝ)
  (h1 : old_selling_price = 88)
  (h2 : old_gross_profit_percentage = 10)
  (h3 : new_selling_price = 92) :
  let cost := old_selling_price / (1 + old_gross_profit_percentage / 100)
  let new_gross_profit := new_selling_price - cost
  new_gross_profit / cost * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_new_gross_profit_percentage_l1640_164007


namespace NUMINAMATH_CALUDE_set_operations_l1640_164050

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x < 2}) ∧
  (Set.compl A = {x | x ≥ 2}) ∧
  (Set.compl A ∩ B = {x | 2 ≤ x ∧ x < 5}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1640_164050


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_one_two_l1640_164046

-- Define the universal set U
def U : Set ℤ := {x : ℤ | |x - 1| < 3}

-- Define set A
def A : Set ℤ := {1, 2, 3}

-- Define the complement of B in U
def C_U_B : Set ℤ := {-1, 3}

-- Theorem to prove
theorem A_intersect_B_equals_one_two : A ∩ (U \ C_U_B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_one_two_l1640_164046


namespace NUMINAMATH_CALUDE_evaluate_expression_l1640_164066

theorem evaluate_expression (x y z : ℚ) :
  x = 1/3 → y = 2/3 → z = -9 → x^2 * y^3 * z = -8/27 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1640_164066


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l1640_164080

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l1640_164080


namespace NUMINAMATH_CALUDE_water_level_rise_l1640_164075

/-- The rise in water level when submerging a cone in a rectangular vessel -/
theorem water_level_rise (r h l w : ℝ) (hr : r = 5) (hh : h = 15) (hl : l = 20) (hw : w = 15) :
  (1 / 3 * π * r^2 * h) / (l * w) = 5 / 12 * π := by sorry

end NUMINAMATH_CALUDE_water_level_rise_l1640_164075


namespace NUMINAMATH_CALUDE_article_cost_price_l1640_164042

theorem article_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (price_increase : ℝ) 
  (h1 : loss_percentage = 25)
  (h2 : gain_percentage = 15)
  (h3 : price_increase = 500) : 
  ∃ (cost_price : ℝ), 
    cost_price * (1 - loss_percentage / 100) + price_increase = cost_price * (1 + gain_percentage / 100) ∧ 
    cost_price = 1250 := by
  sorry

#check article_cost_price

end NUMINAMATH_CALUDE_article_cost_price_l1640_164042


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1640_164003

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Theorem for part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Theorem for part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1640_164003


namespace NUMINAMATH_CALUDE_train_meeting_probability_l1640_164031

-- Define the time intervals
def train_arrival_interval : ℝ := 60  -- 60 minutes between 1:00 and 2:00
def alex_arrival_interval : ℝ := 75   -- 75 minutes between 1:00 and 2:15
def train_wait_time : ℝ := 15         -- 15 minutes wait time

-- Define the probability calculation function
def calculate_probability (train_interval : ℝ) (alex_interval : ℝ) (wait_time : ℝ) : ℚ :=
  -- The actual calculation is not implemented, just the type signature
  0

-- Theorem statement
theorem train_meeting_probability :
  calculate_probability train_arrival_interval alex_arrival_interval train_wait_time = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_probability_l1640_164031


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_60_45045_l1640_164051

theorem gcd_lcm_sum_60_45045 : Nat.gcd 60 45045 + Nat.lcm 60 45045 = 180195 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_60_45045_l1640_164051


namespace NUMINAMATH_CALUDE_prime_abs_nsquared_minus_6n_minus_27_l1640_164032

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_abs_nsquared_minus_6n_minus_27 (n : ℤ) :
  is_prime (Int.natAbs (n^2 - 6*n - 27)) ↔ n = -4 ∨ n = -2 ∨ n = 8 ∨ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_abs_nsquared_minus_6n_minus_27_l1640_164032


namespace NUMINAMATH_CALUDE_tim_bodyguard_cost_l1640_164008

def bodyguards_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

theorem tim_bodyguard_cost :
  bodyguards_cost 2 20 8 7 = 2240 :=
by sorry

end NUMINAMATH_CALUDE_tim_bodyguard_cost_l1640_164008


namespace NUMINAMATH_CALUDE_debby_drinks_six_bottles_per_day_l1640_164023

-- Define the total number of bottles
def total_bottles : ℕ := 12

-- Define the number of days the bottles last
def days_last : ℕ := 2

-- Define the function to calculate bottles per day
def bottles_per_day (total : ℕ) (days : ℕ) : ℚ :=
  (total : ℚ) / (days : ℚ)

-- Theorem statement
theorem debby_drinks_six_bottles_per_day :
  bottles_per_day total_bottles days_last = 6 := by
  sorry

end NUMINAMATH_CALUDE_debby_drinks_six_bottles_per_day_l1640_164023


namespace NUMINAMATH_CALUDE_flea_jump_rational_angle_l1640_164002

/-- Represents a flea jumping between two intersecting lines -/
structure FleaJump where
  α : ℝ  -- Angle between the lines in radians
  jump_length : ℝ  -- Length of each jump
  returns_to_start : Prop  -- Flea eventually returns to the starting point

/-- Main theorem: If a flea jumps between two intersecting lines and returns to the start,
    the angle between the lines is a rational multiple of π -/
theorem flea_jump_rational_angle (fj : FleaJump) 
  (h1 : fj.jump_length = 1)
  (h2 : fj.returns_to_start)
  (h3 : fj.α > 0)
  (h4 : fj.α < π) :
  ∃ q : ℚ, fj.α = q * π :=
sorry

end NUMINAMATH_CALUDE_flea_jump_rational_angle_l1640_164002


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l1640_164012

-- Part 1
theorem simplify_and_evaluate_1 (x : ℝ) (h : x = 3) :
  3 * x^2 - (5 * x - (6 * x - 4) - 2 * x^2) = 44 := by sorry

-- Part 2
theorem simplify_and_evaluate_2 (m n : ℝ) (h1 : m = -1) (h2 : n = 2) :
  (8 * m * n - 3 * m^2) - 5 * m * n - 2 * (3 * m * n - 2 * m^2) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l1640_164012


namespace NUMINAMATH_CALUDE_invisible_dots_count_l1640_164038

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : Nat := 21

/-- The total number of dots on five standard six-sided dice -/
def total_dots (n : Nat) : Nat := n * standard_die_sum

/-- The sum of visible dots in the given configuration -/
def visible_dots : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 2 + 3 + 4 + 5 + 6 + 4 + 5 + 6

/-- The number of dice in the problem -/
def num_dice : Nat := 5

/-- The number of visible faces in the problem -/
def num_visible_faces : Nat := 14

theorem invisible_dots_count :
  total_dots num_dice - visible_dots = 49 :=
sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l1640_164038


namespace NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_digit_l1640_164067

def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_five_divisible_unit_digit : 
  ∀ d : ℕ, is_digit d → 
  (∀ n : ℕ, is_divisible_by_five n → units_digit n ≠ d) → 
  d ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_digit_l1640_164067


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1640_164030

/-- Given a point P and two lines in the plane, this theorem states that
    the second line passes through P and is perpendicular to the first line. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let P : ℝ × ℝ := (1, -2)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 2 * y - 5
  let line2 : ℝ → ℝ → ℝ := λ x y => 2 * x - 3 * y - 8
  (line2 P.1 P.2 = 0) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, line1 x y = 0 → line2 x y = k * line1 y (-x)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1640_164030


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1640_164016

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1640_164016


namespace NUMINAMATH_CALUDE_parabola_kite_theorem_l1640_164090

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coeff_sum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_kite_theorem (k : Kite) :
  k.p1 = Parabola.mk a (-4) ∧
  k.p2 = Parabola.mk (-b) 8 ∧
  kite_area k = 24 →
  coeff_sum k = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_kite_theorem_l1640_164090
