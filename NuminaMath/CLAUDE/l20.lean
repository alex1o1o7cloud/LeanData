import Mathlib

namespace NUMINAMATH_CALUDE_product_sum_max_l20_2090

theorem product_sum_max (a b c d : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 5) (h4 : d = 6) :
  a * b + b * c + c * d + d * a = 63 := by sorry

end NUMINAMATH_CALUDE_product_sum_max_l20_2090


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l20_2035

/-- The eccentricity of a hyperbola with equation x^2 - y^2/m = 1 is 2 if and only if m = 3 -/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 - y^2/m = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = 1 + 1/m) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l20_2035


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l20_2038

theorem smallest_n_for_inequality : ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) ∧
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l20_2038


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l20_2097

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - (1/6)*x - 1/6 < 0} = Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l20_2097


namespace NUMINAMATH_CALUDE_screws_per_section_l20_2062

theorem screws_per_section
  (initial_screws : ℕ)
  (buy_factor : ℕ)
  (num_sections : ℕ)
  (h1 : initial_screws = 8)
  (h2 : buy_factor = 2)
  (h3 : num_sections = 4) :
  (initial_screws + initial_screws * buy_factor) / num_sections = 6 :=
by sorry

end NUMINAMATH_CALUDE_screws_per_section_l20_2062


namespace NUMINAMATH_CALUDE_pyramid_volume_integer_heights_l20_2088

theorem pyramid_volume_integer_heights (base_side : ℕ) (height : ℕ) :
  base_side = 640 →
  height = 1024 →
  (∃ (n : ℕ), n = 85 ∧
    (∀ h : ℕ, h < height →
      (25 * (height - h)^3) % 192 = 0 ↔ h ∈ Finset.range (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_integer_heights_l20_2088


namespace NUMINAMATH_CALUDE_read_book_series_l20_2043

/-- The number of weeks needed to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - (first_week + second_week)
  2 + (remaining_books + subsequent_weeks - 1) / subsequent_weeks

/-- Proof that it takes 7 weeks to read the book series -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_series_l20_2043


namespace NUMINAMATH_CALUDE_samuel_coaching_fee_l20_2013

/-- Calculate the total coaching fee for Samuel --/
theorem samuel_coaching_fee :
  let days_in_period : ℕ := 307 -- Days from Jan 1 to Nov 4 in a non-leap year
  let holidays : ℕ := 5
  let daily_fee : ℕ := 23
  let discount_period : ℕ := 30
  let discount_rate : ℚ := 1 / 10

  let coaching_days : ℕ := days_in_period - holidays
  let full_discount_periods : ℕ := coaching_days / discount_period
  let base_fee : ℕ := coaching_days * daily_fee
  let discount_per_period : ℚ := (discount_period * daily_fee : ℚ) * discount_rate
  let total_discount : ℚ := discount_per_period * full_discount_periods
  
  (base_fee : ℚ) - total_discount = 6256 := by
  sorry

end NUMINAMATH_CALUDE_samuel_coaching_fee_l20_2013


namespace NUMINAMATH_CALUDE_two_digit_integers_count_l20_2006

def available_digits : Finset Nat := {1, 2, 3, 8, 9}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def count_two_digit_integers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d ↦ d ≤ 9)).card * (digits.filter (λ d ↦ d ≤ 9)).card

theorem two_digit_integers_count :
  count_two_digit_integers available_digits = 25 := by sorry

end NUMINAMATH_CALUDE_two_digit_integers_count_l20_2006


namespace NUMINAMATH_CALUDE_max_log_sum_l20_2080

/-- Given that xyz + y + z = 12, the maximum value of log₄x + log₂y + log₂z is 3 -/
theorem max_log_sum (x y z : ℝ) (h : x * y * z + y + z = 12) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.log x / Real.log 4) + (Real.log y / Real.log 2) + (Real.log z / Real.log 2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l20_2080


namespace NUMINAMATH_CALUDE_M_characterization_l20_2051

def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

def valid_set (S : Set ℝ) : Prop :=
  S = {2, 3} ∨ S = {1, 6} ∨ S = ∅

def valid_m (m : ℝ) : Prop :=
  m = 7 ∨ m = 5 ∨ (m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)

theorem M_characterization (m : ℝ) :
  (M m ∩ {1, 2, 3, 6} = M m) →
  (valid_set (M m) ∧ valid_m m) :=
sorry

end NUMINAMATH_CALUDE_M_characterization_l20_2051


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_l20_2071

def repeating_707 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def repeating_909 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def product : ℕ := repeating_707 * repeating_909

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_tens_and_units_digits :
  tens_digit product + units_digit product = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_l20_2071


namespace NUMINAMATH_CALUDE_second_divisor_is_nine_l20_2082

theorem second_divisor_is_nine (least_number : Nat) (second_divisor : Nat) : 
  least_number = 282 →
  least_number % 31 = 3 →
  least_number % second_divisor = 3 →
  second_divisor ≠ 31 →
  second_divisor = 9 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_is_nine_l20_2082


namespace NUMINAMATH_CALUDE_exponent_division_l20_2086

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_exponent_division_l20_2086


namespace NUMINAMATH_CALUDE_triangle_larger_segment_is_82_5_l20_2072

/-- A triangle with sides a, b, c, where c is the longest side --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  h_c_longest : c ≥ a ∧ c ≥ b

/-- The angle opposite to the longest side of the triangle --/
def Triangle.angle_opposite_longest (t : Triangle) : ℝ := sorry

/-- The altitude to the longest side of the triangle --/
def Triangle.altitude_to_longest (t : Triangle) : ℝ := sorry

/-- The larger segment cut off by the altitude on the longest side --/
def Triangle.larger_segment (t : Triangle) : ℝ := sorry

theorem triangle_larger_segment_is_82_5 (t : Triangle) 
  (h_sides : t.a = 40 ∧ t.b = 90 ∧ t.c = 100) 
  (h_angle : t.angle_opposite_longest = Real.pi / 3) : 
  t.larger_segment = 82.5 := by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_is_82_5_l20_2072


namespace NUMINAMATH_CALUDE_rational_roots_of_quadratic_l20_2085

theorem rational_roots_of_quadratic 
  (p q n : ℚ) : 
  ∃ (x : ℚ), (p + q + n) * x^2 - 2*(p + q) * x + (p + q - n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_quadratic_l20_2085


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_l20_2024

theorem condition_p_necessary_not_sufficient :
  (∀ a : ℝ, (|a| ≤ 1 → a ≤ 1)) ∧
  (∃ a : ℝ, a ≤ 1 ∧ ¬(|a| ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_l20_2024


namespace NUMINAMATH_CALUDE_not_q_is_true_l20_2063

theorem not_q_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬¬q = False := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_true_l20_2063


namespace NUMINAMATH_CALUDE_giraffe_ratio_l20_2092

theorem giraffe_ratio (total_giraffes : ℕ) (difference : ℕ) : 
  total_giraffes = 300 →
  difference = 290 →
  total_giraffes = (total_giraffes - difference) + difference →
  (total_giraffes : ℚ) / (total_giraffes - difference) = 30 := by
  sorry

end NUMINAMATH_CALUDE_giraffe_ratio_l20_2092


namespace NUMINAMATH_CALUDE_collinear_vectors_l20_2052

/-- Given vectors a and b in ℝ², if ma + nb is collinear with a - 2b, then m/n = -1/2 -/
theorem collinear_vectors (a b : ℝ × ℝ) (m n : ℝ) 
  (h1 : a = (2, 3))
  (h2 : b = (-1, 2))
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (m • a + n • b) = k • (a - 2 • b)) :
  m / n = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l20_2052


namespace NUMINAMATH_CALUDE_james_work_hours_james_work_hours_proof_l20_2004

/-- Calculates the number of hours James needs to work to pay for food waste and janitorial costs -/
theorem james_work_hours (james_wage : ℝ) (meat_cost meat_wasted : ℝ) 
  (fruit_veg_cost fruit_veg_wasted : ℝ) (bread_cost bread_wasted : ℝ)
  (janitor_wage janitor_hours : ℝ) : ℝ :=
  let total_cost := meat_cost * meat_wasted + fruit_veg_cost * fruit_veg_wasted + 
                    bread_cost * bread_wasted + janitor_wage * 1.5 * janitor_hours
  total_cost / james_wage

/-- Proves that James needs to work 50 hours given the specific conditions -/
theorem james_work_hours_proof : 
  james_work_hours 8 5 20 4 15 1.5 60 10 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_work_hours_james_work_hours_proof_l20_2004


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l20_2055

theorem arithmetic_expression_equality : 2 - 3*(-4) - 7 + 2*(-5) - 9 + 6*(-2) = -24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l20_2055


namespace NUMINAMATH_CALUDE_open_box_volume_l20_2064

/-- Calculate the volume of an open box created from a rectangular sheet --/
theorem open_box_volume (sheet_length sheet_width cut_side : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_side = 8 →
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side = 5120 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l20_2064


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l20_2070

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l20_2070


namespace NUMINAMATH_CALUDE_y_increases_with_x_l20_2046

theorem y_increases_with_x (m : ℝ) (x y : ℝ → ℝ) :
  (∀ t, y t = (m^2 + 2) * x t) →
  StrictMono y :=
sorry

end NUMINAMATH_CALUDE_y_increases_with_x_l20_2046


namespace NUMINAMATH_CALUDE_count_congruent_integers_l20_2009

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 14 = 9) (Finset.range 500)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_integers_l20_2009


namespace NUMINAMATH_CALUDE_solution_range_l20_2018

theorem solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) ∧
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) ∧
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∀ x : ℝ, x^2 - b*x - 5 = 0 ↔ (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l20_2018


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l20_2042

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem units_digit_of_p_plus_five (p : ℕ) 
  (h1 : p > 0)
  (h2 : isEven p)
  (h3 : unitsDigit p > 0)
  (h4 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_five_l20_2042


namespace NUMINAMATH_CALUDE_sons_age_l20_2050

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  (man_age + 2) = 2 * (son_age + 2) →
  son_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sons_age_l20_2050


namespace NUMINAMATH_CALUDE_smallest_marble_count_l20_2003

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the number of ways to select marbles according to the given events -/
def event_probability (mc : MarbleCount) (event : Fin 4) : ℕ :=
  match event with
  | 0 => mc.blue.choose 4
  | 1 => (mc.red.choose 2) * (mc.white.choose 2)
  | 2 => (mc.red.choose 2) * (mc.white.choose 1) * (mc.blue.choose 1)
  | 3 => mc.red * mc.white * mc.blue * mc.green

/-- Checks if all events have equal probability -/
def events_equally_likely (mc : MarbleCount) : Prop :=
  ∀ i j : Fin 4, event_probability mc i = event_probability mc j

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    events_equally_likely mc ∧ 
    total_marbles mc = 13 ∧ 
    (∀ (mc' : MarbleCount), events_equally_likely mc' → total_marbles mc' ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l20_2003


namespace NUMINAMATH_CALUDE_difference_of_squares_73_47_l20_2022

theorem difference_of_squares_73_47 : 73^2 - 47^2 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_73_47_l20_2022


namespace NUMINAMATH_CALUDE_range_of_a_for_two_integer_solutions_l20_2044

/-- A system of inequalities has exactly two integer solutions -/
def has_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (↑x : ℝ)^2 - (↑x : ℝ) + a - a^2 < 0 ∧ (↑x : ℝ) + 2*a > 1 ∧
    (↑y : ℝ)^2 - (↑y : ℝ) + a - a^2 < 0 ∧ (↑y : ℝ) + 2*a > 1 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y →
      ¬((↑z : ℝ)^2 - (↑z : ℝ) + a - a^2 < 0 ∧ (↑z : ℝ) + 2*a > 1)

/-- The range of a that satisfies the conditions -/
theorem range_of_a_for_two_integer_solutions :
  ∀ a : ℝ, has_two_integer_solutions a ↔ 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_integer_solutions_l20_2044


namespace NUMINAMATH_CALUDE_power_of_two_equation_l20_2060

theorem power_of_two_equation (m : ℤ) :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = m * 2^2006 →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l20_2060


namespace NUMINAMATH_CALUDE_geometric_series_sum_l20_2020

theorem geometric_series_sum : ∑' i, (2/3:ℝ)^i = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l20_2020


namespace NUMINAMATH_CALUDE_diamond_property_false_l20_2007

def diamond (x y : ℝ) : ℝ := 2 * |x - y| + 1

theorem diamond_property_false :
  ¬ ∀ x y : ℝ, 3 * (diamond x y) = 3 * (diamond (2*x) (2*y)) :=
sorry

end NUMINAMATH_CALUDE_diamond_property_false_l20_2007


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l20_2016

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l20_2016


namespace NUMINAMATH_CALUDE_selfie_difference_l20_2029

theorem selfie_difference (a b c : ℕ) (h1 : a + b + c = 2430) (h2 : 10 * b = 17 * a) (h3 : 10 * c = 23 * a) : c - a = 637 := by
  sorry

end NUMINAMATH_CALUDE_selfie_difference_l20_2029


namespace NUMINAMATH_CALUDE_soccer_ball_hexagons_l20_2039

/-- Represents a soccer ball with black pentagons and white hexagons -/
structure SoccerBall where
  black_pentagons : ℕ
  white_hexagons : ℕ
  pentagon_sides : ℕ
  hexagon_sides : ℕ
  pentagon_hexagon_connections : ℕ
  hexagon_pentagon_connections : ℕ
  hexagon_hexagon_connections : ℕ

/-- Theorem stating the number of white hexagons on a soccer ball with specific conditions -/
theorem soccer_ball_hexagons (ball : SoccerBall) :
  ball.black_pentagons = 12 ∧
  ball.pentagon_sides = 5 ∧
  ball.hexagon_sides = 6 ∧
  ball.pentagon_hexagon_connections = 5 ∧
  ball.hexagon_pentagon_connections = 3 ∧
  ball.hexagon_hexagon_connections = 3 →
  ball.white_hexagons = 20 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_hexagons_l20_2039


namespace NUMINAMATH_CALUDE_three_number_sum_l20_2005

theorem three_number_sum : ∀ (a b c : ℝ),
  b = 150 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 550 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l20_2005


namespace NUMINAMATH_CALUDE_typist_margin_l20_2049

theorem typist_margin (sheet_width sheet_length side_margin : ℝ)
  (percentage_used : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : side_margin = 2)
  (h4 : percentage_used = 0.64) :
  let total_area := sheet_width * sheet_length
  let typing_width := sheet_width - 2 * side_margin
  let top_bottom_margin := (total_area * percentage_used / typing_width - sheet_length) / (-2)
  top_bottom_margin = 3 := by
  sorry

end NUMINAMATH_CALUDE_typist_margin_l20_2049


namespace NUMINAMATH_CALUDE_brick_factory_workers_l20_2015

/-- The maximum number of workers that can be hired at a brick factory -/
def max_workers : ℕ := 8

theorem brick_factory_workers :
  ∀ n : ℕ,
  n ≤ max_workers ↔
  (10 * n - n * n ≥ 13) ∧
  ∀ m : ℕ, m > n → (10 * m - m * m < 13) :=
by sorry

end NUMINAMATH_CALUDE_brick_factory_workers_l20_2015


namespace NUMINAMATH_CALUDE_largest_expression_l20_2076

theorem largest_expression (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  (a + b) ≥ max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l20_2076


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l20_2008

/-- Given a town's initial population and population after one year,
    calculate the population increase rate as a percentage. -/
def populationIncreaseRate (initialPopulation finalPopulation : ℕ) : ℚ :=
  (finalPopulation - initialPopulation : ℚ) / initialPopulation * 100

/-- Theorem stating that for a town with an initial population of 200
    and a population of 220 after 1 year, the population increase rate is 10%. -/
theorem population_increase_rate_example :
  populationIncreaseRate 200 220 = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l20_2008


namespace NUMINAMATH_CALUDE_extra_lambs_found_l20_2069

def lambs_problem (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (final_lambs : ℕ) : ℕ :=
  let lambs_after_babies := initial_lambs + lambs_with_babies * babies_per_lamb
  let lambs_after_trade := lambs_after_babies - traded_lambs
  final_lambs - lambs_after_trade

theorem extra_lambs_found :
  lambs_problem 6 2 2 3 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_extra_lambs_found_l20_2069


namespace NUMINAMATH_CALUDE_percent_of_number_l20_2081

theorem percent_of_number (x : ℝ) : 120 = 1.5 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l20_2081


namespace NUMINAMATH_CALUDE_two_Z_one_eq_one_l20_2089

/-- The Z operation on two real numbers -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: 2 Z 1 = 1 -/
theorem two_Z_one_eq_one : Z 2 1 = 1 := by sorry

end NUMINAMATH_CALUDE_two_Z_one_eq_one_l20_2089


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l20_2047

/-- Proves that for an infinite geometric series with first term 400 and sum 2500, the common ratio is 21/25 -/
theorem infinite_geometric_series_ratio : ∃ (r : ℝ), 
  let a : ℝ := 400
  let S : ℝ := 2500
  r > 0 ∧ r < 1 ∧ S = a / (1 - r) ∧ r = 21 / 25 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l20_2047


namespace NUMINAMATH_CALUDE_probability_two_yellow_one_red_l20_2059

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 5

/-- The number of orange marbles in the jar -/
def orange_marbles : ℕ := 4

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + yellow_marbles + orange_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 3

/-- Calculates the number of combinations of n items taken k at a time -/
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- The probability of choosing 2 yellow and 1 red marble from the jar -/
theorem probability_two_yellow_one_red : 
  (combination yellow_marbles 2 * combination red_marbles 1) / 
  (combination total_marbles chosen_marbles) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_yellow_one_red_l20_2059


namespace NUMINAMATH_CALUDE_jacks_change_jacks_change_is_five_l20_2021

/-- Given Jack's sandwich order and payment, calculate his change -/
theorem jacks_change (num_sandwiches : ℕ) (price_per_sandwich : ℕ) (payment : ℕ) : ℕ :=
  let total_cost := num_sandwiches * price_per_sandwich
  payment - total_cost

/-- Prove that Jack's change is $5 given the problem conditions -/
theorem jacks_change_is_five : 
  jacks_change 3 5 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jacks_change_jacks_change_is_five_l20_2021


namespace NUMINAMATH_CALUDE_total_sodas_sold_restaurant_soda_sales_l20_2045

/-- Theorem: Total sodas sold given diet soda count and ratio of regular to diet --/
theorem total_sodas_sold (diet_count : ℕ) (regular_ratio diet_ratio : ℕ) : ℕ :=
  let regular_count := (regular_ratio * diet_count) / diet_ratio
  diet_count + regular_count

/-- Proof of the specific problem --/
theorem restaurant_soda_sales : total_sodas_sold 28 9 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_sodas_sold_restaurant_soda_sales_l20_2045


namespace NUMINAMATH_CALUDE_average_age_of_new_men_l20_2078

theorem average_age_of_new_men (n : ℕ) (initial_average : ℝ) 
  (replaced_ages : List ℝ) (age_increase : ℝ) :
  n = 20 ∧ 
  replaced_ages = [21, 23, 25, 27] ∧ 
  age_increase = 2 →
  (n * (initial_average + age_increase) - n * initial_average + replaced_ages.sum) / replaced_ages.length = 34 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_new_men_l20_2078


namespace NUMINAMATH_CALUDE_circle_equation_standard_form_tangent_line_b_value_l20_2040

open Real

/-- A line ax + by = c is tangent to a circle (x - h)^2 + (y - k)^2 = r^2 if and only if
    the distance from the center (h, k) to the line is equal to the radius r. -/
def is_tangent_line_to_circle (a b c h k r : ℝ) : Prop :=
  (|a * h + b * k - c| / sqrt (a^2 + b^2)) = r

/-- The equation of the circle x^2 + y^2 - 2x - 2y + 1 = 0 in standard form is (x - 1)^2 + (y - 1)^2 = 1 -/
theorem circle_equation_standard_form :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 2*y + 1 = 0 ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

/-- The main theorem: If the line 3x + 4y = b is tangent to the circle x^2 + y^2 - 2x - 2y + 1 = 0,
    then b = 2 or b = 12 -/
theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, is_tangent_line_to_circle 3 4 b 1 1 1) → (b = 2 ∨ b = 12) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_standard_form_tangent_line_b_value_l20_2040


namespace NUMINAMATH_CALUDE_tom_books_count_l20_2000

/-- Given that Joan has 10 books and the total number of books is 48,
    prove that Tom has 38 books. -/
theorem tom_books_count (joan_books : ℕ) (total_books : ℕ) (tom_books : ℕ) 
    (h1 : joan_books = 10)
    (h2 : total_books = 48)
    (h3 : tom_books + joan_books = total_books) : 
  tom_books = 38 := by
sorry

end NUMINAMATH_CALUDE_tom_books_count_l20_2000


namespace NUMINAMATH_CALUDE_not_perfect_square_l20_2053

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l20_2053


namespace NUMINAMATH_CALUDE_max_c_value_l20_2033

theorem max_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) :
  c ≤ 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l20_2033


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l20_2087

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (7/5) ∧
  (∀ x : ℝ, g x = 0 → r ≤ x) ∧
  g r = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l20_2087


namespace NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l20_2066

/-- A parabola with vertex (h, k) has the general form y = a(x - h)² + k where a ≠ 0 -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- The analytical expression of a parabola -/
def Parabola.expression (p : Parabola) (x : ℝ) : ℝ := p.a * (x - p.h)^2 + p.k

theorem parabola_with_vertex_two_three :
  ∃ (p : Parabola), p.vertex = (2, 3) ∧ ∀ x, p.expression x = -(x - 2)^2 + 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l20_2066


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l20_2068

theorem consecutive_numbers_product (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  35 * 36 * 37 * 38 * 39 = 120 * (100000 * A + 10000 * B + 1000 * A + 100 * B + 10 * A + B) → 
  A = 5 ∧ B = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l20_2068


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_l20_2058

theorem tangent_slope_three_points (x y : ℝ) : 
  y = x^3 ∧ (3 * x^2 = 3) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_three_points_l20_2058


namespace NUMINAMATH_CALUDE_gcf_of_16_and_24_l20_2012

theorem gcf_of_16_and_24 : Nat.gcd 16 24 = 8 :=
by
  have h1 : Nat.lcm 16 24 = 48 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcf_of_16_and_24_l20_2012


namespace NUMINAMATH_CALUDE_worksheets_graded_l20_2032

/-- 
Given:
- There are 9 worksheets in total
- Each worksheet has 4 problems
- There are 16 problems left to grade

Prove that the number of worksheets already graded is 5.
-/
theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets * problems_per_worksheet - problems_left = 5 * problems_per_worksheet :=
by
  sorry

#check worksheets_graded

end NUMINAMATH_CALUDE_worksheets_graded_l20_2032


namespace NUMINAMATH_CALUDE_function_property_l20_2091

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the constant k
variable (k : ℝ)

-- State the theorem
theorem function_property (h1 : ∀ x : ℝ, f x + f (1 - x) = k)
                          (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
                          (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  k = 10 := by sorry

end NUMINAMATH_CALUDE_function_property_l20_2091


namespace NUMINAMATH_CALUDE_z_equals_2_minus_12i_z_is_pure_imaginary_l20_2048

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

/-- Theorem for the first condition -/
theorem z_equals_2_minus_12i (m : ℝ) : z m = Complex.mk 2 (-12) ↔ m = -1 := by sorry

/-- Theorem for the second condition -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_z_equals_2_minus_12i_z_is_pure_imaginary_l20_2048


namespace NUMINAMATH_CALUDE_set_intersection_subset_l20_2037

theorem set_intersection_subset (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (A.Nonempty ∧ B.Nonempty) → (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_subset_l20_2037


namespace NUMINAMATH_CALUDE_flowchart_output_l20_2001

def swap_operation (a b c : ℕ) : ℕ × ℕ × ℕ := 
  let (a', c') := (c, a)
  let (b', c'') := (c', b)
  (a', b', c'')

theorem flowchart_output (a b c : ℕ) (h1 : a = 21) (h2 : b = 32) (h3 : c = 75) :
  swap_operation a b c = (75, 21, 32) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_output_l20_2001


namespace NUMINAMATH_CALUDE_baron_munchausen_contradiction_l20_2079

theorem baron_munchausen_contradiction (d : ℝ) (T : ℝ) (h1 : d > 0) (h2 : T > 0) : 
  ¬(d / 2 = 5 * (d / (2 * 5)) ∧ d / 2 = 6 * (T / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_contradiction_l20_2079


namespace NUMINAMATH_CALUDE_junhyun_travel_distance_l20_2083

/-- The distance Junhyun traveled by bus in kilometers -/
def bus_distance : ℝ := 2.6

/-- The distance Junhyun traveled by subway in kilometers -/
def subway_distance : ℝ := 5.98

/-- The total distance Junhyun traveled using public transportation -/
def total_distance : ℝ := bus_distance + subway_distance

/-- Theorem stating that the total distance Junhyun traveled is 8.58 km -/
theorem junhyun_travel_distance : total_distance = 8.58 := by sorry

end NUMINAMATH_CALUDE_junhyun_travel_distance_l20_2083


namespace NUMINAMATH_CALUDE_barneys_inventory_l20_2027

/-- The number of items left in Barney's grocery store -/
def items_left (restocked : ℕ) (sold : ℕ) (in_storeroom : ℕ) : ℕ :=
  (restocked - sold) + in_storeroom

/-- Theorem stating the total number of items left in Barney's grocery store -/
theorem barneys_inventory : items_left 4458 1561 575 = 3472 := by
  sorry

end NUMINAMATH_CALUDE_barneys_inventory_l20_2027


namespace NUMINAMATH_CALUDE_exactly_three_numbers_l20_2073

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- Predicate for numbers satisfying the given conditions -/
def satisfies_conditions (n : TwoDigitNumber) : Prop :=
  (n.val - sum_of_digits n) % 10 = 2 ∧ n.val % 3 = 0

/-- The main theorem stating there are exactly 3 numbers satisfying the conditions -/
theorem exactly_three_numbers :
  ∃! (s : Finset TwoDigitNumber), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_numbers_l20_2073


namespace NUMINAMATH_CALUDE_total_bread_served_l20_2098

-- Define the quantities of bread served
def wheat_bread : ℚ := 1.25
def white_bread : ℚ := 3/4
def rye_bread : ℚ := 0.6
def multigrain_bread : ℚ := 7/10

-- Theorem to prove
theorem total_bread_served :
  wheat_bread + white_bread + rye_bread + multigrain_bread = 3 + 3/10 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_served_l20_2098


namespace NUMINAMATH_CALUDE_minimum_cans_for_target_gallons_l20_2074

/-- The number of ounces in one gallon -/
def ounces_per_gallon : ℕ := 128

/-- The number of ounces each can holds -/
def ounces_per_can : ℕ := 16

/-- The number of gallons we want to have at least -/
def target_gallons : ℚ := 3/2

theorem minimum_cans_for_target_gallons :
  let total_ounces := (target_gallons * ounces_per_gallon).ceil
  let num_cans := (total_ounces + ounces_per_can - 1) / ounces_per_can
  num_cans = 12 := by sorry

end NUMINAMATH_CALUDE_minimum_cans_for_target_gallons_l20_2074


namespace NUMINAMATH_CALUDE_min_value_zero_l20_2057

open Real

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  3 * x^2 - 4 * c * x * y + (2 * c^2 + 1) * y^2 - 6 * x - 3 * y + 5

/-- The theorem stating the condition for minimum value of f to be 0 -/
theorem min_value_zero (c : ℝ) :
  (∀ x y : ℝ, f c x y ≥ 0) ∧ (∃ x y : ℝ, f c x y = 0) ↔ c = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_l20_2057


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l20_2041

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt x = 2.507936507936508) →
  x = 0.49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l20_2041


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l20_2002

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l20_2002


namespace NUMINAMATH_CALUDE_sum_of_digits_7ab_l20_2065

/-- Integer consisting of 1234 sevens in base 10 -/
def a : ℕ := 7 * (10^1234 - 1) / 9

/-- Integer consisting of 1234 twos in base 10 -/
def b : ℕ := 2 * (10^1234 - 1) / 9

/-- Sum of digits in the base 10 representation of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_7ab : sum_of_digits (7 * a * b) = 11100 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_7ab_l20_2065


namespace NUMINAMATH_CALUDE_cage_cost_l20_2095

/-- The cost of the cage given the payment and change -/
theorem cage_cost (bill : ℝ) (change : ℝ) (h1 : bill = 20) (h2 : change = 0.26) :
  bill - change = 19.74 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_l20_2095


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l20_2031

theorem major_premise_incorrect : ¬(∀ (a : ℝ) (n : ℕ), n > 0 → (a^(1/n : ℝ))^n = a) := by
  sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l20_2031


namespace NUMINAMATH_CALUDE_gcd_15n_plus_4_9n_plus_2_max_2_l20_2025

theorem gcd_15n_plus_4_9n_plus_2_max_2 :
  (∃ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_15n_plus_4_9n_plus_2_max_2_l20_2025


namespace NUMINAMATH_CALUDE_intersection_equality_l20_2023

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- Define the intersection set
def intersection_set : Set ℝ := {x | 1/3 ≤ x ∧ x < 16}

-- State the theorem
theorem intersection_equality : M ∩ N = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l20_2023


namespace NUMINAMATH_CALUDE_two_pythagorean_triples_l20_2017

-- Define a Pythagorean triple
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- State the theorem
theorem two_pythagorean_triples :
  isPythagoreanTriple 3 4 5 ∧ isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_two_pythagorean_triples_l20_2017


namespace NUMINAMATH_CALUDE_power_of_ten_multiplication_l20_2093

theorem power_of_ten_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_multiplication_l20_2093


namespace NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l20_2099

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)
  (is_triangle : A + B + C = Real.pi)

-- State the theorem
theorem triangle_angle_A_is_30_degrees
  (abc : Triangle)
  (h1 : abc.a^2 - abc.b^2 = Real.sqrt 3 * abc.b * abc.c)
  (h2 : Real.sin abc.C = 2 * Real.sqrt 3 * Real.sin abc.B) :
  abc.A = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_is_30_degrees_l20_2099


namespace NUMINAMATH_CALUDE_lemon_juice_test_point_l20_2026

theorem lemon_juice_test_point (lower_bound upper_bound : ℝ) 
  (h_lower : lower_bound = 500)
  (h_upper : upper_bound = 1500)
  (golden_ratio : ℝ) 
  (h_golden : golden_ratio = 0.618) : 
  let x₁ := lower_bound + golden_ratio * (upper_bound - lower_bound)
  let x₂ := upper_bound + lower_bound - x₁
  x₂ = 882 := by
sorry

end NUMINAMATH_CALUDE_lemon_juice_test_point_l20_2026


namespace NUMINAMATH_CALUDE_total_shirts_bought_l20_2014

theorem total_shirts_bought (cost_15 : ℕ) (price_15 : ℕ) (price_20 : ℕ) (total_cost : ℕ) :
  cost_15 = 3 →
  price_15 = 15 →
  price_20 = 20 →
  total_cost = 85 →
  ∃ (cost_20 : ℕ), cost_15 * price_15 + cost_20 * price_20 = total_cost ∧ cost_15 + cost_20 = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_shirts_bought_l20_2014


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l20_2010

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l20_2010


namespace NUMINAMATH_CALUDE_value_of_b_l20_2061

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l20_2061


namespace NUMINAMATH_CALUDE_gcd_lcm_equalities_l20_2036

/-- Define * as the greatest common divisor operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Define ∘ as the least common multiple operation -/
def lcm_op (a b : ℕ) : ℕ := Nat.lcm a b

/-- The main theorem stating the equalities for gcd and lcm operations -/
theorem gcd_lcm_equalities (a b c : ℕ) :
  (gcd_op a (lcm_op b c) = lcm_op (gcd_op a b) (gcd_op a c)) ∧
  (lcm_op a (gcd_op b c) = gcd_op (lcm_op a b) (lcm_op a c)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_equalities_l20_2036


namespace NUMINAMATH_CALUDE_inequality_proof_l20_2067

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l20_2067


namespace NUMINAMATH_CALUDE_range_of_a_l20_2028

/-- The equation |x^2 - a| - x + 2 = 0 has two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ |x^2 - a| - x + 2 = 0 ∧ |y^2 - a| - y + 2 = 0

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : has_two_distinct_roots a) : a > 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l20_2028


namespace NUMINAMATH_CALUDE_probability_of_stopping_is_43_103_l20_2077

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightCycle where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the probability of stopping at a traffic light -/
def probabilityOfStopping (cycle : TrafficLightCycle) : ℚ :=
  let totalCycleTime := cycle.red + cycle.green + cycle.yellow
  let stoppingTime := cycle.red + cycle.yellow
  stoppingTime / totalCycleTime

/-- The specific traffic light cycle in the problem -/
def problemCycle : TrafficLightCycle :=
  { red := 40, green := 60, yellow := 3 }

theorem probability_of_stopping_is_43_103 :
  probabilityOfStopping problemCycle = 43 / 103 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_stopping_is_43_103_l20_2077


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l20_2011

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : perp l β) :
  perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l20_2011


namespace NUMINAMATH_CALUDE_trig_identity_l20_2030

theorem trig_identity : 
  Real.cos (π / 3) * Real.tan (π / 4) + 3 / 4 * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l20_2030


namespace NUMINAMATH_CALUDE_xiaomin_house_coordinates_l20_2096

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The school's position -/
def school : Point := { x := 0, y := 0 }

/-- Xiaomin's house position relative to the school -/
def house_relative : Point := { x := 200, y := -150 }

/-- Theorem stating that Xiaomin's house coordinates are (200, -150) -/
theorem xiaomin_house_coordinates :
  ∃ (p : Point), p.x = school.x + house_relative.x ∧ p.y = school.y + house_relative.y ∧ 
  p.x = 200 ∧ p.y = -150 := by
  sorry

end NUMINAMATH_CALUDE_xiaomin_house_coordinates_l20_2096


namespace NUMINAMATH_CALUDE_magnet_to_stuffed_animals_ratio_l20_2075

-- Define the cost of the magnet
def magnet_cost : ℚ := 3

-- Define the cost of a single stuffed animal
def stuffed_animal_cost : ℚ := 6

-- Define the combined cost of two stuffed animals
def two_stuffed_animals_cost : ℚ := 2 * stuffed_animal_cost

-- Theorem stating the ratio of magnet cost to combined stuffed animals cost
theorem magnet_to_stuffed_animals_ratio :
  magnet_cost / two_stuffed_animals_cost = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_magnet_to_stuffed_animals_ratio_l20_2075


namespace NUMINAMATH_CALUDE_arithmetic_operations_l20_2094

theorem arithmetic_operations (a b : ℝ) : 
  (a ≠ 0 → a / a = 1) ∧ 
  (b ≠ 0 → a / b = a * (1 / b)) ∧ 
  (a * 1 = a) ∧ 
  (0 / b = 0) :=
sorry

#check arithmetic_operations

end NUMINAMATH_CALUDE_arithmetic_operations_l20_2094


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l20_2019

/-- Calculates the number of peanut butter and jelly sandwiches eaten in a school year --/
def pbj_sandwiches_eaten (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) 
  (ham_cheese_interval : ℕ) (wed_missed : ℕ) (fri_missed : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let wed_after_holidays := total_wed - wed_holidays
  let fri_after_holidays := total_fri - fri_holidays
  let wed_after_missed := wed_after_holidays - wed_missed
  let fri_after_missed := fri_after_holidays - fri_missed
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let pbj_wed := wed_after_missed - ham_cheese_weeks
  let pbj_fri := fri_after_missed - (2 * ham_cheese_weeks)
  pbj_wed + pbj_fri

theorem jackson_pbj_sandwiches :
  pbj_sandwiches_eaten 36 2 3 4 1 2 = 37 := by
  sorry

#eval pbj_sandwiches_eaten 36 2 3 4 1 2

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l20_2019


namespace NUMINAMATH_CALUDE_cassandra_apple_pie_l20_2056

/-- The number of apples in each slice of pie -/
def apples_per_slice (total_apples : ℕ) (num_pies : ℕ) (slices_per_pie : ℕ) : ℚ :=
  (total_apples : ℚ) / (num_pies * slices_per_pie)

/-- Cassandra's apple pie problem -/
theorem cassandra_apple_pie :
  let total_apples : ℕ := 4 * 12  -- 4 dozen
  let num_pies : ℕ := 4
  let slices_per_pie : ℕ := 6
  apples_per_slice total_apples num_pies slices_per_pie = 2 := by
sorry

end NUMINAMATH_CALUDE_cassandra_apple_pie_l20_2056


namespace NUMINAMATH_CALUDE_f_has_one_min_no_max_l20_2084

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

-- State the theorem
theorem f_has_one_min_no_max :
  ∃! x : ℝ, IsLocalMin f x ∧ ∀ y : ℝ, ¬IsLocalMax f y :=
by sorry

end NUMINAMATH_CALUDE_f_has_one_min_no_max_l20_2084


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_330_l20_2054

theorem largest_common_divisor_408_330 : Nat.gcd 408 330 = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_330_l20_2054


namespace NUMINAMATH_CALUDE_expression_proof_l20_2034

/-- An expression that, when divided by (3x + 29), equals 2 -/
def E (x : ℝ) : ℝ := 6 * x + 58

theorem expression_proof (x : ℝ) : E x / (3 * x + 29) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_proof_l20_2034
