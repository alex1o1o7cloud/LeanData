import Mathlib

namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l2151_215196

theorem number_of_elements_in_set (initial_average : ℝ) (incorrect_number : ℝ) (correct_number : ℝ) (correct_average : ℝ) :
  initial_average = 16 ∧ 
  incorrect_number = 26 ∧ 
  correct_number = 46 ∧ 
  correct_average = 18 →
  ∃ n : ℕ, n = 10 ∧ 
    n * initial_average = (n - 1) * initial_average + incorrect_number ∧
    n * correct_average = (n - 1) * initial_average + correct_number :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l2151_215196


namespace NUMINAMATH_CALUDE_fruit_profit_equation_l2151_215128

/-- Represents the profit equation for a fruit selling scenario -/
theorem fruit_profit_equation 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ) 
  (profit : ℝ) :
  cost = 40 →
  initial_price = 50 →
  initial_volume = 500 →
  price_increase > 0 →
  volume_decrease = 10 * price_increase →
  profit = 8000 →
  ∃ x : ℝ, x > 50 ∧ (x - cost) * (initial_volume - volume_decrease) = profit :=
by sorry

end NUMINAMATH_CALUDE_fruit_profit_equation_l2151_215128


namespace NUMINAMATH_CALUDE_angle_complement_supplement_difference_l2151_215120

theorem angle_complement_supplement_difference : 
  ∀ α : ℝ, (90 - α) - (180 - α) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_difference_l2151_215120


namespace NUMINAMATH_CALUDE_book_cost_problem_l2151_215194

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 540)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19)
  (h_equal_sell : (1 - loss_percent / 100) * cost_loss = (1 + gain_percent / 100) * (total_cost - cost_loss)) :
  ∃ (cost_loss : ℝ), cost_loss = 315 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l2151_215194


namespace NUMINAMATH_CALUDE_some_number_value_l2151_215188

theorem some_number_value (x : ℝ) : 40 + 5 * 12 / (180 / x) = 41 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2151_215188


namespace NUMINAMATH_CALUDE_granola_bars_per_box_l2151_215137

theorem granola_bars_per_box 
  (num_kids : ℕ) 
  (bars_per_kid : ℕ) 
  (num_boxes : ℕ) 
  (h1 : num_kids = 30) 
  (h2 : bars_per_kid = 2) 
  (h3 : num_boxes = 5) :
  (num_kids * bars_per_kid) / num_boxes = 12 := by
sorry

end NUMINAMATH_CALUDE_granola_bars_per_box_l2151_215137


namespace NUMINAMATH_CALUDE_dividend_calculation_l2151_215185

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 167 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2151_215185


namespace NUMINAMATH_CALUDE_red_balls_count_l2151_215184

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) (red_balls : ℕ) : 
  total_balls = 1000 →
  prob_red = 1/5 →
  red_balls = (total_balls : ℚ) * prob_red →
  red_balls = 200 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2151_215184


namespace NUMINAMATH_CALUDE_three_digit_four_digit_count_l2151_215116

theorem three_digit_four_digit_count : 
  (Finset.filter (fun x : ℕ => 
    100 ≤ 3 * x ∧ 3 * x ≤ 999 ∧ 
    1000 ≤ 4 * x ∧ 4 * x ≤ 9999) (Finset.range 10000)).card = 84 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_four_digit_count_l2151_215116


namespace NUMINAMATH_CALUDE_bookstore_problem_l2151_215158

theorem bookstore_problem (total_notebooks : ℕ) (cost_A cost_B total_cost : ℚ) 
  (sell_A sell_B : ℚ) (discount_A : ℚ) (profit_threshold : ℚ) :
  total_notebooks = 350 →
  cost_A = 12 →
  cost_B = 15 →
  total_cost = 4800 →
  sell_A = 20 →
  sell_B = 25 →
  discount_A = 0.7 →
  profit_threshold = 2348 →
  ∃ (num_A num_B : ℕ) (m : ℕ),
    num_A + num_B = total_notebooks ∧
    num_A * cost_A + num_B * cost_B = total_cost ∧
    num_A = 150 ∧
    m * sell_A + m * sell_B + (num_A - m) * sell_A * discount_A + (num_B - m) * cost_B - total_cost ≥ profit_threshold ∧
    ∀ k : ℕ, k < m → k * sell_A + k * sell_B + (num_A - k) * sell_A * discount_A + (num_B - k) * cost_B - total_cost < profit_threshold :=
by sorry

end NUMINAMATH_CALUDE_bookstore_problem_l2151_215158


namespace NUMINAMATH_CALUDE_cube_root_problem_l2151_215161

theorem cube_root_problem (a m : ℝ) (h1 : a > 0) 
  (h2 : (m + 7)^2 = a) (h3 : (2*m - 1)^2 = a) : 
  (a - m)^(1/3 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_problem_l2151_215161


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2151_215102

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | Real.log x / Real.log 2 > Real.log x / Real.log 3}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2151_215102


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2151_215156

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (a b : ℤ), (n - 6 : ℚ) / 15 = a ∧ (n - 5 : ℚ) / 24 = b) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2151_215156


namespace NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l2151_215173

theorem gcd_nine_factorial_six_factorial_squared : Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 51840 := by
  sorry

end NUMINAMATH_CALUDE_gcd_nine_factorial_six_factorial_squared_l2151_215173


namespace NUMINAMATH_CALUDE_third_day_distance_is_15_l2151_215187

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ

/-- Calculates the distance hiked on the third day --/
def third_day_distance (hike : ThreeDayHike) : ℝ :=
  hike.total_distance - hike.first_day_distance - hike.second_day_distance

/-- Theorem: The distance hiked on the third day is 15 kilometers --/
theorem third_day_distance_is_15 (hike : ThreeDayHike)
    (h1 : hike.total_distance = 50)
    (h2 : hike.first_day_distance = 10)
    (h3 : hike.second_day_distance = hike.total_distance / 2) :
    third_day_distance hike = 15 := by
  sorry

end NUMINAMATH_CALUDE_third_day_distance_is_15_l2151_215187


namespace NUMINAMATH_CALUDE_coefficient_x3_is_negative_540_l2151_215112

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (3x^2 - 1/x)^6
def coefficient_x3 : ℤ :=
  -3^3 * binomial 6 3

-- Theorem statement
theorem coefficient_x3_is_negative_540 : coefficient_x3 = -540 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_negative_540_l2151_215112


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2151_215134

theorem cosine_sine_identity : Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2151_215134


namespace NUMINAMATH_CALUDE_steve_distance_theorem_l2151_215101

def steve_problem (distance : ℝ) : Prop :=
  let speed_to_work : ℝ := 17.5 / 2
  let speed_from_work : ℝ := 17.5
  let time_to_work : ℝ := distance / speed_to_work
  let time_from_work : ℝ := distance / speed_from_work
  (time_to_work + time_from_work = 6) ∧ (speed_from_work = 2 * speed_to_work)

theorem steve_distance_theorem : 
  ∃ (distance : ℝ), steve_problem distance ∧ distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_steve_distance_theorem_l2151_215101


namespace NUMINAMATH_CALUDE_car_overtake_distance_l2151_215104

/-- Represents the distance between two cars -/
def distance_between_cars (v1 v2 t : ℝ) : ℝ := (v2 - v1) * t

/-- Theorem stating the distance between two cars under given conditions -/
theorem car_overtake_distance :
  let red_speed : ℝ := 30
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 1
  distance_between_cars red_speed black_speed overtake_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_overtake_distance_l2151_215104


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2151_215149

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → perpendicular_line x y → x * 2 + y * 1 = -1) ∧
  (perpendicular_line point_P.1 point_P.2) ∧
  (∀ x y, given_line x y → ∀ a b, perpendicular_line a b → 
    (y - point_P.2) * (x - point_P.1) = -(b - point_P.2) * (a - point_P.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2151_215149


namespace NUMINAMATH_CALUDE_exists_n_power_half_eq_ten_l2151_215107

theorem exists_n_power_half_eq_ten :
  ∃ n : ℝ, n > 0 ∧ n ^ (n / 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_exists_n_power_half_eq_ten_l2151_215107


namespace NUMINAMATH_CALUDE_min_value_sum_l2151_215174

theorem min_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 6 * z ≥ 18 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 8 ∧ x₀ + 3 * y₀ + 6 * z₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2151_215174


namespace NUMINAMATH_CALUDE_r_earns_75_l2151_215136

/-- Represents the daily earnings of individuals p, q, r, and s -/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions of the problem -/
def earnings_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2400 / 8 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7 ∧
  e.s + e.r = 800 / 4 ∧
  e.p + e.s = 700 / 6

/-- Theorem stating that under the given conditions, r earns 75 per day -/
theorem r_earns_75 (e : DailyEarnings) : 
  earnings_conditions e → e.r = 75 := by
  sorry

#check r_earns_75

end NUMINAMATH_CALUDE_r_earns_75_l2151_215136


namespace NUMINAMATH_CALUDE_addition_closed_in_P_l2151_215166

-- Define the set P
def P : Set ℝ := {n | ∃ k : ℕ+, n = Real.log k}

-- State the theorem
theorem addition_closed_in_P (a b : ℝ) (ha : a ∈ P) (hb : b ∈ P) : 
  a + b ∈ P := by sorry

end NUMINAMATH_CALUDE_addition_closed_in_P_l2151_215166


namespace NUMINAMATH_CALUDE_circle_parameter_range_l2151_215163

theorem circle_parameter_range (a : ℝ) : 
  (∃ (h : ℝ) (k : ℝ) (r : ℝ), ∀ (x y : ℝ), 
    x^2 + y^2 + 2*x - 4*y + a + 1 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) → 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l2151_215163


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_2010_l2151_215139

theorem smallest_k_for_divisibility_by_2010 :
  ∃ (k : ℕ), k > 1 ∧
  (∀ (n : ℕ), n > 0 → (n^k - n) % 2010 = 0) ∧
  (∀ (m : ℕ), m > 1 ∧ m < k → ∃ (n : ℕ), n > 0 ∧ (n^m - n) % 2010 ≠ 0) ∧
  k = 133 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_2010_l2151_215139


namespace NUMINAMATH_CALUDE_ratio_closest_to_nine_l2151_215110

theorem ratio_closest_to_nine : 
  ∀ n : ℕ, |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - 9| ≤ 
           |((10^3000 + 10^3003) : ℝ) / (10^3001 + 10^3002) - n| :=
by sorry

end NUMINAMATH_CALUDE_ratio_closest_to_nine_l2151_215110


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2151_215171

/-- The sum of the digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : 
  (Nat.digits 2 300).sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2151_215171


namespace NUMINAMATH_CALUDE_prime_sum_product_l2151_215191

theorem prime_sum_product (x y z : ℕ) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  x ≤ y ∧ y ≤ z ∧
  x + y + z = 12 ∧
  x * y + y * z + x * z = 41 →
  x + 2 * y + 3 * z = 29 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2151_215191


namespace NUMINAMATH_CALUDE_second_solution_concentration_l2151_215124

/-- 
Given two solutions that are mixed to form a new solution,
this theorem proves that the concentration of the second solution
must be 10% under the specified conditions.
-/
theorem second_solution_concentration
  (volume_first : ℝ)
  (concentration_first : ℝ)
  (volume_second : ℝ)
  (concentration_final : ℝ)
  (h1 : volume_first = 4)
  (h2 : concentration_first = 0.04)
  (h3 : volume_second = 2)
  (h4 : concentration_final = 0.06)
  (h5 : volume_first * concentration_first + volume_second * (concentration_second / 100) = 
        (volume_first + volume_second) * concentration_final) :
  concentration_second = 10 := by
  sorry

#check second_solution_concentration

end NUMINAMATH_CALUDE_second_solution_concentration_l2151_215124


namespace NUMINAMATH_CALUDE_revenue_maximized_at_20_l2151_215100

-- Define the revenue function
def R (p : ℝ) : ℝ := p * (160 - 4 * p)

-- State the theorem
theorem revenue_maximized_at_20 :
  ∃ (p_max : ℝ), p_max ≤ 40 ∧ 
  ∀ (p : ℝ), p ≤ 40 → R p ≤ R p_max ∧
  p_max = 20 := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_20_l2151_215100


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l2151_215131

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l2151_215131


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_power_plus_a_l2151_215145

theorem infinite_primes_dividing_power_plus_a (a : ℕ) (ha : a > 0) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ 2^(2^n) + a} :=
by sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_power_plus_a_l2151_215145


namespace NUMINAMATH_CALUDE_parabola_properties_l2151_215147

/-- Definition of the parabola function -/
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -3)

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Theorem stating that the given vertex and axis of symmetry are correct for the parabola -/
theorem parabola_properties :
  (∀ x, f x ≥ f (vertex.1)) ∧
  (∀ x, f x = f (2 * axis_of_symmetry - x)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2151_215147


namespace NUMINAMATH_CALUDE_min_value_expression_l2151_215189

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 3 ∧
  (a^2 / b + b^2 / c + c^2 / a = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2151_215189


namespace NUMINAMATH_CALUDE_rogers_shelves_l2151_215118

/-- Given the conditions of Roger's book shelving problem, prove that he needs 4 shelves. -/
theorem rogers_shelves (total_books : ℕ) (librarian_books : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 14) 
  (h2 : librarian_books = 2) 
  (h3 : books_per_shelf = 3) : 
  ((total_books - librarian_books) / books_per_shelf : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rogers_shelves_l2151_215118


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l2151_215142

def is_reducible (n : ℕ) : Prop :=
  n > 17 ∧ Nat.gcd (n - 17) (7 * n + 4) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 20 → ¬ is_reducible m) ∧ is_reducible 20 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l2151_215142


namespace NUMINAMATH_CALUDE_max_an_over_n_is_half_l2151_215199

/-- The number of trailing zeroes in the base-n representation of n! -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum value of a_n/n is 1/2 -/
theorem max_an_over_n_is_half :
  (∀ n > 1, (a n : ℚ) / n ≤ 1/2) ∧ (∃ n > 1, (a n : ℚ) / n = 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_an_over_n_is_half_l2151_215199


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2151_215157

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  0 < n → 0 < r → r < 1 / 500 → 
  (↑m : ℝ) ^ (1/3 : ℝ) = n + r → 
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/500 ∧ (↑k : ℝ) ^ (1/3 : ℝ) = ↑(n-1) + s) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2151_215157


namespace NUMINAMATH_CALUDE_triangle_problem_l2151_215123

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2 * t.a * Real.cos t.A * Real.cos t.B - 2 * t.b * Real.sin t.A * Real.sin t.A)
  (h2 : t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 3 / 4)
  (h3 : t.a + t.b + t.c = 15) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2151_215123


namespace NUMINAMATH_CALUDE_canoe_kayak_rental_difference_l2151_215152

theorem canoe_kayak_rental_difference :
  ∀ (canoe_cost kayak_cost : ℚ) 
    (canoe_count kayak_count : ℕ) 
    (total_revenue : ℚ),
  canoe_cost = 12 →
  kayak_cost = 18 →
  canoe_count = (3 * kayak_count) / 2 →
  total_revenue = canoe_cost * canoe_count + kayak_cost * kayak_count →
  total_revenue = 504 →
  canoe_count - kayak_count = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_rental_difference_l2151_215152


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_l2151_215195

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_l2151_215195


namespace NUMINAMATH_CALUDE_bike_shop_profit_l2151_215177

/-- The cost of parts for fixing a single bike tire -/
def tire_part_cost : ℝ := 5

theorem bike_shop_profit (tire_repair_price : ℝ) (tire_repairs : ℕ) 
  (complex_repair_price : ℝ) (complex_repair_cost : ℝ) (complex_repairs : ℕ)
  (retail_profit : ℝ) (fixed_expenses : ℝ) (total_profit : ℝ) :
  tire_repair_price = 20 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  retail_profit = 2000 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  tire_part_cost = 5 := by
sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l2151_215177


namespace NUMINAMATH_CALUDE_sin_pi_half_plus_two_alpha_l2151_215119

theorem sin_pi_half_plus_two_alpha (y₀ : ℝ) (α : ℝ) : 
  (1/2)^2 + y₀^2 = 1 → 
  Real.cos α = 1/2 →
  Real.sin (π/2 + 2*α) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_half_plus_two_alpha_l2151_215119


namespace NUMINAMATH_CALUDE_eugene_shoes_count_l2151_215151

/-- The cost of a T-shirt before discount -/
def t_shirt_cost : ℚ := 20

/-- The cost of a pair of pants before discount -/
def pants_cost : ℚ := 80

/-- The cost of a pair of shoes before discount -/
def shoes_cost : ℚ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℚ := 1/10

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The total amount Eugene pays -/
def total_paid : ℚ := 558

/-- The function to calculate the discounted price -/
def discounted_price (price : ℚ) : ℚ := price * (1 - discount_rate)

/-- The theorem stating the number of pairs of shoes Eugene buys -/
theorem eugene_shoes_count :
  ∃ (n : ℕ), n * discounted_price shoes_cost = 
    total_paid - (num_tshirts * discounted_price t_shirt_cost + num_pants * discounted_price pants_cost) ∧
    n = 2 := by sorry

end NUMINAMATH_CALUDE_eugene_shoes_count_l2151_215151


namespace NUMINAMATH_CALUDE_walk_bike_time_difference_l2151_215155

def blocks : ℕ := 18
def walk_time_per_block : ℚ := 1
def bike_time_per_block : ℚ := 20 / 60

theorem walk_bike_time_difference :
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 12 := by
  sorry

end NUMINAMATH_CALUDE_walk_bike_time_difference_l2151_215155


namespace NUMINAMATH_CALUDE_trapezium_longer_side_length_l2151_215113

/-- Given a trapezium with the following properties:
    - One parallel side is 10 cm long
    - The distance between parallel sides is 15 cm
    - The area is 210 square centimeters
    This theorem proves that the length of the other parallel side is 18 cm. -/
theorem trapezium_longer_side_length (a b h : ℝ) : 
  a = 10 → h = 15 → (a + b) * h / 2 = 210 → b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_longer_side_length_l2151_215113


namespace NUMINAMATH_CALUDE_f_of_two_equals_zero_l2151_215126

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_two_equals_zero : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_zero_l2151_215126


namespace NUMINAMATH_CALUDE_ratio_of_tenth_terms_l2151_215159

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem ratio_of_tenth_terms 
  (a b : ArithmeticSequence)
  (h : ∀ n, sumFirstN a n / sumFirstN b n = (3 * n - 1) / (2 * n + 3)) :
  a.a 10 / b.a 10 = 57 / 41 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_tenth_terms_l2151_215159


namespace NUMINAMATH_CALUDE_smallest_coin_count_l2151_215182

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def count_proper_factors (n : ℕ) : ℕ := (count_factors n) - 2

theorem smallest_coin_count :
  ∀ m : ℕ, m > 0 →
    (count_factors m = 19 ∧ count_proper_factors m = 17) →
    m ≥ 786432 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coin_count_l2151_215182


namespace NUMINAMATH_CALUDE_largest_fraction_l2151_215132

theorem largest_fraction : 
  let a := (1 / 17 - 1 / 19) / 20
  let b := (1 / 15 - 1 / 21) / 60
  let c := (1 / 13 - 1 / 23) / 100
  let d := (1 / 11 - 1 / 25) / 140
  d > a ∧ d > b ∧ d > c := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l2151_215132


namespace NUMINAMATH_CALUDE_hexagonal_prism_diagonals_truncated_cube_diagonals_l2151_215180

/- Right hexagonal prism -/
theorem hexagonal_prism_diagonals (n : Nat) (v : Nat) (d : Nat) :
  n = 12 → v = 3 → d = n * v / 2 → d = 18 := by sorry

/- Truncated cube -/
theorem truncated_cube_diagonals (n : Nat) (v : Nat) (d : Nat) :
  n = 24 → v = 10 → d = n * v / 2 → d = 120 := by sorry

end NUMINAMATH_CALUDE_hexagonal_prism_diagonals_truncated_cube_diagonals_l2151_215180


namespace NUMINAMATH_CALUDE_xiao_ming_final_score_l2151_215150

/-- Calculate the final score given individual scores and weights -/
def final_score (content_score language_score demeanor_score : ℝ)
  (content_weight language_weight demeanor_weight : ℝ) : ℝ :=
  content_score * content_weight +
  language_score * language_weight +
  demeanor_score * demeanor_weight

/-- Theorem stating that Xiao Ming's final score is 86.2 -/
theorem xiao_ming_final_score :
  final_score 85 90 82 0.6 0.3 0.1 = 86.2 := by
  sorry

#eval final_score 85 90 82 0.6 0.3 0.1

end NUMINAMATH_CALUDE_xiao_ming_final_score_l2151_215150


namespace NUMINAMATH_CALUDE_opposite_face_of_ten_l2151_215144

/-- Represents a cube with six faces labeled with distinct integers -/
structure Cube where
  faces : Finset ℕ
  distinct : faces.card = 6
  range : ∀ n ∈ faces, 6 ≤ n ∧ n ≤ 11

/-- The sum of all numbers on the cube's faces -/
def Cube.total_sum (c : Cube) : ℕ := c.faces.sum id

/-- Represents a roll of the cube, showing four lateral faces -/
structure Roll (c : Cube) where
  lateral_sum : ℕ
  valid : lateral_sum = c.total_sum - (c.faces.sum id - lateral_sum)

theorem opposite_face_of_ten (c : Cube) 
  (roll1 : Roll c) (roll2 : Roll c)
  (h1 : roll1.lateral_sum = 36)
  (h2 : roll2.lateral_sum = 33)
  : ∃ n ∈ c.faces, n = 8 ∧ (c.faces.sum id - (10 + n) = roll1.lateral_sum ∨ 
                            c.faces.sum id - (10 + n) = roll2.lateral_sum) :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_ten_l2151_215144


namespace NUMINAMATH_CALUDE_intimate_interval_is_two_three_l2151_215153

def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

def intimate_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

theorem intimate_interval_is_two_three :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧
  intimate_functions f g a b ∧
  ∀ (c d : ℝ), c < 2 ∨ d > 3 → ¬intimate_functions f g c d :=
sorry

end NUMINAMATH_CALUDE_intimate_interval_is_two_three_l2151_215153


namespace NUMINAMATH_CALUDE_hot_dog_discount_calculation_l2151_215114

theorem hot_dog_discount_calculation (num_hot_dogs : ℕ) (price_per_hot_dog : ℕ) (discount_rate : ℚ) :
  num_hot_dogs = 6 →
  price_per_hot_dog = 50 →
  discount_rate = 1/10 →
  (num_hot_dogs * price_per_hot_dog) * (1 - discount_rate) = 270 :=
by sorry

end NUMINAMATH_CALUDE_hot_dog_discount_calculation_l2151_215114


namespace NUMINAMATH_CALUDE_smallest_possible_a_l2151_215167

theorem smallest_possible_a (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 1/3)^2 - 1/4) →  -- parabola with vertex (1/3, -1/4)
  (∃ (x y : ℝ), y = a * x^2 + b * x + c) →    -- equation of parabola
  (a > 0) →                                   -- a is positive
  (∃ (n : ℤ), 2 * a + b + 3 * c = n) →        -- 2a + b + 3c is an integer
  (∀ (a' : ℝ), a' ≥ 9/16 ∨ ¬(
    (∃ (x y : ℝ), y = a' * (x - 1/3)^2 - 1/4) ∧
    (∃ (x y : ℝ), y = a' * x^2 + b * x + c) ∧
    (a' > 0) ∧
    (∃ (n : ℤ), 2 * a' + b + 3 * c = n)
  )) :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l2151_215167


namespace NUMINAMATH_CALUDE_tangent_line_at_one_minimum_value_of_f_l2151_215133

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + f 1 → (x - y - 1 = 0) :=
sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value_of_f :
  ∃ x, f x = -1 / Real.exp 1 ∧ ∀ y, f y ≥ -1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_minimum_value_of_f_l2151_215133


namespace NUMINAMATH_CALUDE_complement_M_in_U_l2151_215175

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x^2 - 5*x + 6 = 0}

theorem complement_M_in_U : (U \ M) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l2151_215175


namespace NUMINAMATH_CALUDE_problem_solution_l2151_215198

theorem problem_solution (a : ℝ) : 
  (∀ b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  a * 15 * 11 = 1 →
  a = 6 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2151_215198


namespace NUMINAMATH_CALUDE_inequality_proof_l2151_215178

theorem inequality_proof (a b m n p : ℝ) 
  (h1 : a > b) (h2 : m > n) (h3 : p > 0) : 
  n - a * p < m - b * p := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2151_215178


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2151_215154

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def num_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 8 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : num_distributions 6 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2151_215154


namespace NUMINAMATH_CALUDE_salt_concentration_change_l2151_215138

/-- Proves that adding 1.25 kg of pure salt to 20 kg of 15% saltwater results in 20% saltwater -/
theorem salt_concentration_change (initial_water : ℝ) (initial_concentration : ℝ) 
  (added_salt : ℝ) (final_concentration : ℝ) 
  (h1 : initial_water = 20)
  (h2 : initial_concentration = 0.15)
  (h3 : added_salt = 1.25)
  (h4 : final_concentration = 0.2) :
  initial_water * initial_concentration + added_salt = 
  (initial_water + added_salt) * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_concentration_change_l2151_215138


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2151_215164

theorem quadratic_inequality_solution (x : ℝ) :
  (-5 * x^2 + 10 * x - 3 > 0) ↔ (x > 1 - Real.sqrt 10 / 5 ∧ x < 1 + Real.sqrt 10 / 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2151_215164


namespace NUMINAMATH_CALUDE_student_count_correct_l2151_215176

/-- Represents the changes in student numbers for a grade --/
structure GradeChanges where
  initial : Nat
  left : Nat
  joined : Nat
  transferredIn : Nat
  transferredOut : Nat

/-- Calculates the final number of students in a grade --/
def finalStudents (changes : GradeChanges) : Nat :=
  changes.initial - changes.left + changes.joined + changes.transferredIn - changes.transferredOut

/-- Theorem: The calculated final numbers of students in each grade and their total are correct --/
theorem student_count_correct (fourth : GradeChanges) (fifth : GradeChanges) (sixth : GradeChanges) 
    (h4 : fourth = ⟨4, 3, 42, 0, 10⟩)
    (h5 : fifth = ⟨10, 5, 25, 10, 5⟩)
    (h6 : sixth = ⟨15, 7, 30, 5, 0⟩) : 
    finalStudents fourth = 33 ∧ 
    finalStudents fifth = 35 ∧ 
    finalStudents sixth = 43 ∧
    finalStudents fourth + finalStudents fifth + finalStudents sixth = 111 := by
  sorry

end NUMINAMATH_CALUDE_student_count_correct_l2151_215176


namespace NUMINAMATH_CALUDE_box_surface_area_is_288_l2151_215129

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_side : ℕ) : ℕ :=
  let new_length := sheet_length - 2 * corner_side
  let new_width := sheet_width - 2 * corner_side
  new_length * new_width

/-- Theorem: The surface area of the interior of the open box is 288 square units. -/
theorem box_surface_area_is_288 :
  interior_surface_area 36 24 6 = 288 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_is_288_l2151_215129


namespace NUMINAMATH_CALUDE_cubic_equation_roots_difference_l2151_215172

theorem cubic_equation_roots_difference (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^3 + 3*p*x^2 + (4*p - 1)*x + p = 0 ∧ 
   y^3 + 3*p*y^2 + (4*p - 1)*y + p = 0 ∧ 
   y - x = 1) ↔ 
  (p = 0 ∨ p = 6/5 ∨ p = 10/9) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_difference_l2151_215172


namespace NUMINAMATH_CALUDE_s2_side_length_l2151_215181

/-- A composite rectangle structure -/
structure CompositeRectangle where
  width : ℕ
  height : ℕ
  s2_side : ℕ

/-- The composite rectangle satisfies the given conditions -/
def satisfies_conditions (cr : CompositeRectangle) : Prop :=
  cr.width = 3782 ∧ cr.height = 2260 ∧
  ∃ (r : ℕ), 2 * r + cr.s2_side = cr.height ∧ 2 * r + 3 * cr.s2_side = cr.width

/-- Theorem: The side length of S2 in the composite rectangle is 761 units -/
theorem s2_side_length :
  ∀ (cr : CompositeRectangle), satisfies_conditions cr → cr.s2_side = 761 :=
by
  sorry

end NUMINAMATH_CALUDE_s2_side_length_l2151_215181


namespace NUMINAMATH_CALUDE_vector_magnitude_l2151_215146

def a : ℝ × ℝ := (1, 1)
def b : ℝ → ℝ × ℝ := λ y ↦ (3, y)

theorem vector_magnitude (y : ℝ) : 
  (∃ k : ℝ, b y - a = k • a) → ‖b y - a‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2151_215146


namespace NUMINAMATH_CALUDE_min_decimal_digits_l2151_215148

def fraction : ℚ := 987654321 / (2^30 * 5^2)

theorem min_decimal_digits (f : ℚ) (h : f = fraction) : 
  (∃ (n : ℕ), n ≥ 30 ∧ ∃ (m : ℤ), f * 10^n = m) ∧ 
  (∀ (k : ℕ), k < 30 → ¬∃ (m : ℤ), f * 10^k = m) := by
  sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l2151_215148


namespace NUMINAMATH_CALUDE_patricia_money_l2151_215190

theorem patricia_money (jethro carmen patricia : ℕ) : 
  carmen = 2 * jethro - 7 →
  patricia = 3 * jethro →
  jethro + carmen + patricia = 113 →
  patricia = 60 := by
sorry

end NUMINAMATH_CALUDE_patricia_money_l2151_215190


namespace NUMINAMATH_CALUDE_gummy_vitamins_cost_l2151_215169

/-- Calculates the total cost of gummy vitamin bottles after discounts and coupons -/
def calculate_total_cost (regular_price : ℚ) (individual_discount : ℚ) (coupon_value : ℚ) (num_bottles : ℕ) (bulk_discount : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - individual_discount)
  let price_after_coupon := discounted_price - coupon_value
  let total_before_bulk := price_after_coupon * num_bottles
  let bulk_discount_amount := total_before_bulk * bulk_discount
  total_before_bulk - bulk_discount_amount

/-- Theorem stating that the total cost for 3 bottles of gummy vitamins is $29.78 -/
theorem gummy_vitamins_cost :
  calculate_total_cost 15 (17/100) 2 3 (5/100) = 2978/100 :=
by sorry

end NUMINAMATH_CALUDE_gummy_vitamins_cost_l2151_215169


namespace NUMINAMATH_CALUDE_a_n_property_smallest_n_for_perfect_square_sum_l2151_215108

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

def is_sum_or_diff_of_squares (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a * a + b * b ∨ x = a * a - b * b ∨ x = b * b - a * a

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def a_n (n : ℕ) : ℕ := 10^n - 2

def sum_of_squares_of_digits (x : ℕ) : ℕ :=
  (x.digits 10).map (λ d => d * d) |>.sum

theorem a_n_property (n : ℕ) (h : n > 2) :
  ¬(is_sum_or_diff_of_squares (a_n n)) ∧
  ∀ m : ℕ, m > a_n n → m ≤ largest_n_digit_number n → is_sum_or_diff_of_squares m :=
sorry

theorem smallest_n_for_perfect_square_sum :
  ∀ n : ℕ, n < 66 → ¬(is_perfect_square (sum_of_squares_of_digits (a_n n))) ∧
  is_perfect_square (sum_of_squares_of_digits (a_n 66)) :=
sorry

end NUMINAMATH_CALUDE_a_n_property_smallest_n_for_perfect_square_sum_l2151_215108


namespace NUMINAMATH_CALUDE_complex_product_imaginary_l2151_215121

theorem complex_product_imaginary (a : ℝ) : 
  (Complex.I * (1 + a * Complex.I) + (2 : ℂ) * (1 + a * Complex.I)).re = 0 → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_imaginary_l2151_215121


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2151_215105

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2151_215105


namespace NUMINAMATH_CALUDE_last_ball_is_red_l2151_215125

/-- Represents the color of a ball -/
inductive BallColor
  | Blue
  | Red
  | Green

/-- Represents the state of the bottle -/
structure BottleState where
  blue : Nat
  red : Nat
  green : Nat

/-- Represents a single ball removal operation -/
inductive RemovalOperation
  | BlueGreen
  | RedGreen
  | TwoRed
  | Other

/-- Defines the initial state of the bottle -/
def initialState : BottleState :=
  { blue := 1001, red := 1000, green := 1000 }

/-- Applies a single removal operation to the bottle state -/
def applyOperation (state : BottleState) (op : RemovalOperation) : BottleState :=
  match op with
  | RemovalOperation.BlueGreen => { blue := state.blue - 1, red := state.red + 1, green := state.green - 1 }
  | RemovalOperation.RedGreen => { blue := state.blue, red := state.red, green := state.green - 1 }
  | RemovalOperation.TwoRed => { blue := state.blue + 2, red := state.red - 2, green := state.green }
  | RemovalOperation.Other => { blue := state.blue, red := state.red, green := state.green - 1 }

/-- Determines if the game has ended (only one ball left) -/
def isGameOver (state : BottleState) : Bool :=
  state.blue + state.red + state.green = 1

/-- Theorem: The last remaining ball is red -/
theorem last_ball_is_red :
  ∃ (operations : List RemovalOperation),
    let finalState := operations.foldl applyOperation initialState
    isGameOver finalState ∧ finalState.red = 1 :=
  sorry


end NUMINAMATH_CALUDE_last_ball_is_red_l2151_215125


namespace NUMINAMATH_CALUDE_chromium_percentage_calculation_l2151_215130

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 12

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 8

/-- The mass of the first alloy in kg -/
def mass_first : ℝ := 15

/-- The mass of the second alloy in kg -/
def mass_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 9.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first / 100) * mass_first + 
  (chromium_percentage_second / 100) * mass_second = 
  (chromium_percentage_result / 100) * (mass_first + mass_second) :=
by sorry

#check chromium_percentage_calculation

end NUMINAMATH_CALUDE_chromium_percentage_calculation_l2151_215130


namespace NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l2151_215162

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem f_monotonicity_and_m_range :
  ∀ (a : ℝ),
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ a ≤ Real.sqrt 2 → f a x₁ < f a x₂) ∧
  (a > Real.sqrt 2 → 
    ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x₁ < x ∧ x < x₂ → f a x > f a x₁ ∧ f a x > f a x₃) ∧
    (∀ x : ℝ, 0 < x ∧ x < x₁ → f a x < f a x₁) ∧
    (∀ x : ℝ, x > x₃ → f a x > f a x₃)) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ 1 ∧
    (∀ m : ℝ, (∀ a : ℝ, -2 < a ∧ a ≤ 0 → 
      2*m*Real.exp a*(a+1) + f a x₀ > a^2 + 2*a + 4) ↔ 1 < m ∧ m ≤ Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_m_range_l2151_215162


namespace NUMINAMATH_CALUDE_rectangle_width_on_square_diagonal_l2151_215186

theorem rectangle_width_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let rectangle_length := diagonal
  let rectangle_width := s / Real.sqrt 2
  square_area = rectangle_length * rectangle_width :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_on_square_diagonal_l2151_215186


namespace NUMINAMATH_CALUDE_problem_statement_l2151_215165

theorem problem_statement (x y : ℝ) 
  (eq1 : x + x*y + y = 2 + 3*Real.sqrt 2) 
  (eq2 : x^2 + y^2 = 6) : 
  |x + y + 1| = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2151_215165


namespace NUMINAMATH_CALUDE_base8_palindrome_count_l2151_215127

/-- Represents a digit in base 8 -/
def Base8Digit := Fin 8

/-- Represents a six-digit palindrome in base 8 -/
structure Base8Palindrome where
  a : Base8Digit
  b : Base8Digit
  c : Base8Digit
  d : Base8Digit
  h : a.val ≠ 0

/-- The count of six-digit palindromes in base 8 -/
def count_base8_palindromes : Nat :=
  (Finset.range 7).card * (Finset.range 8).card * (Finset.range 8).card * (Finset.range 8).card

theorem base8_palindrome_count :
  count_base8_palindromes = 3584 :=
sorry

end NUMINAMATH_CALUDE_base8_palindrome_count_l2151_215127


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2151_215140

/-- The complex number z -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem: The point corresponding to z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2151_215140


namespace NUMINAMATH_CALUDE_green_tea_leaves_needed_l2151_215183

/-- The number of sprigs of mint added to each batch of mud -/
def sprigs_of_mint : ℕ := 3

/-- The number of green tea leaves added per sprig of mint -/
def leaves_per_sprig : ℕ := 2

/-- The factor by which the efficacy of ingredients is reduced in the new mud -/
def efficacy_reduction : ℚ := 1/2

/-- The number of green tea leaves needed for the new batch of mud to maintain the same efficacy -/
def new_leaves_needed : ℕ := 12

/-- Theorem stating that the number of green tea leaves needed for the new batch of mud
    to maintain the same efficacy is equal to 12 -/
theorem green_tea_leaves_needed :
  (sprigs_of_mint * leaves_per_sprig : ℚ) / efficacy_reduction = new_leaves_needed := by
  sorry

end NUMINAMATH_CALUDE_green_tea_leaves_needed_l2151_215183


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l2151_215115

/-- Given a circular arrangement of students, if the 7th and 27th positions
    are opposite each other, then the total number of students is 40. -/
theorem circular_arrangement_students (n : ℕ) : 
  (7 + n / 2 = 27 ∨ 27 + n / 2 = n + 7) → n = 40 :=
by sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l2151_215115


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l2151_215160

theorem factorization_of_cubic (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l2151_215160


namespace NUMINAMATH_CALUDE_Jose_age_is_14_l2151_215170

-- Define the ages as natural numbers
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 3
def Jose_age : ℕ := Zack_age - 4

-- Theorem statement
theorem Jose_age_is_14 : Jose_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_Jose_age_is_14_l2151_215170


namespace NUMINAMATH_CALUDE_mike_savings_rate_l2151_215197

theorem mike_savings_rate (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (weeks : ℕ) :
  carol_initial = 60 →
  carol_weekly = 9 →
  mike_initial = 90 →
  weeks = 5 →
  ∃ (mike_weekly : ℕ),
    carol_initial + carol_weekly * weeks = mike_initial + mike_weekly * weeks ∧
    mike_weekly = 3 :=
by sorry

end NUMINAMATH_CALUDE_mike_savings_rate_l2151_215197


namespace NUMINAMATH_CALUDE_symmetric_quadratic_inequality_l2151_215109

/-- A quadratic function with positive leading coefficient and symmetric about x = 2 -/
def SymmetricQuadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, f (2 + x) = f (2 - x))

theorem symmetric_quadratic_inequality
  (f : ℝ → ℝ) (h : SymmetricQuadratic f) (x : ℝ) :
  f (1 - 2 * x^2) < f (1 + 2 * x - x^2) → -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_inequality_l2151_215109


namespace NUMINAMATH_CALUDE_cube_sum_power_of_two_l2151_215135

theorem cube_sum_power_of_two (k : ℕ+) :
  (∃ (a b c : ℕ+), |((a:ℤ) - b)^3 + ((b:ℤ) - c)^3 + ((c:ℤ) - a)^3| = 3 * 2^(k:ℕ)) ↔
  (∃ (n : ℕ), k = 3 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_power_of_two_l2151_215135


namespace NUMINAMATH_CALUDE_even_odd_sum_zero_l2151_215106

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Main theorem: If f is even and g(x) = f(x-1) is odd, then f(2009) + f(2011) = 0 -/
theorem even_odd_sum_zero (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (h_even : IsEven f) (h_odd : IsOdd g) (h_g : ∀ x, g x = f (x - 1)) :
    f 2009 + f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_zero_l2151_215106


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2151_215193

theorem negative_fraction_comparison : -5/6 < -7/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2151_215193


namespace NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2151_215103

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 4 / x^2 ≥ 7 ∧
  (3 * x^(1/3) + 4 / x^2 = 7 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2151_215103


namespace NUMINAMATH_CALUDE_remainder_theorem_l2151_215117

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom rem_20 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 20) * (P x) + 120
axiom rem_100 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 100) * (P x) + 40

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * (R x) + (-x + 140) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2151_215117


namespace NUMINAMATH_CALUDE_rest_albums_count_l2151_215179

def total_pictures : ℕ := 25
def first_album_pictures : ℕ := 10
def pictures_per_remaining_album : ℕ := 3

theorem rest_albums_count : 
  (total_pictures - first_album_pictures) / pictures_per_remaining_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_rest_albums_count_l2151_215179


namespace NUMINAMATH_CALUDE_sand_amount_l2151_215111

/-- The total amount of sand in tons -/
def total_sand : ℕ := 180

/-- The originally scheduled daily transport rate in tons -/
def scheduled_rate : ℕ := 15

/-- The actual daily transport rate in tons -/
def actual_rate : ℕ := 20

/-- The number of days the task was completed ahead of schedule -/
def days_ahead : ℕ := 3

/-- Theorem stating that the total amount of sand is 180 tons -/
theorem sand_amount :
  ∃ (scheduled_days : ℕ),
    scheduled_days * scheduled_rate = total_sand ∧
    (scheduled_days - days_ahead) * actual_rate = total_sand :=
by sorry

end NUMINAMATH_CALUDE_sand_amount_l2151_215111


namespace NUMINAMATH_CALUDE_swimming_pool_kids_jose_swimming_pool_l2151_215122

theorem swimming_pool_kids (kids_charge : ℕ) (adults_charge : ℕ) 
  (adults_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  let kids_per_day := 
    (weekly_earnings / 7 - adults_per_day * adults_charge) / kids_charge
  kids_per_day

theorem jose_swimming_pool : swimming_pool_kids 3 6 10 588 = 8 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_kids_jose_swimming_pool_l2151_215122


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l2151_215192

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 1.67

/-- The weight of solution A in grams -/
def weight_A : ℝ := 600

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 1.8

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 1.74

theorem liquid_X_percentage :
  (percentage_X_in_A * weight_A + percentage_X_in_B * weight_B) / (weight_A + weight_B) = percentage_X_in_mixture := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l2151_215192


namespace NUMINAMATH_CALUDE_common_ratio_sum_l2151_215141

theorem common_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ 1) (h3 : r ≠ 1) (h4 : p ≠ r) 
  (h5 : k * p^2 - k * r^2 = 2 * (k * p - k * r)) : 
  p + r = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_sum_l2151_215141


namespace NUMINAMATH_CALUDE_binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l2151_215143

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number with n ones -/
def all_ones (n : ℕ) : List Bool :=
  List.replicate n true

theorem binary_representation_of_2_pow_n_minus_1 (n : ℕ) :
  binary_to_decimal (all_ones n) = 2^n - 1 := by
  sorry

/-- The main theorem proving that (1111111111)₂ in decimal form is 2^10 - 1 -/
theorem binary_to_decimal_ten_ones :
  binary_to_decimal (all_ones 10) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_2_pow_n_minus_1_binary_to_decimal_ten_ones_l2151_215143


namespace NUMINAMATH_CALUDE_mean_diesel_cost_l2151_215168

def diesel_rates : List ℝ := [1.2, 1.3, 1.8, 2.1]

theorem mean_diesel_cost (rates : List ℝ) (h : rates = diesel_rates) :
  (rates.sum / rates.length : ℝ) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_diesel_cost_l2151_215168
