import Mathlib

namespace NUMINAMATH_CALUDE_fraction_inequality_l3062_306268

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a * d > b * c) 
  (h2 : a / b > c / d) 
  (hb : b > 0) 
  (hd : d > 0) : 
  a / b > (a + c) / (b + d) ∧ (a + c) / (b + d) > c / d := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3062_306268


namespace NUMINAMATH_CALUDE_chime_time_at_12_l3062_306277

/-- Represents a clock with hourly chimes -/
structure ChimeClock where
  /-- The time in seconds it takes to complete chimes at 4 o'clock -/
  time_at_4 : ℕ
  /-- Assertion that the clock chimes once every hour -/
  chimes_hourly : Prop

/-- Calculates the time it takes to complete chimes at a given hour -/
def chime_time (clock : ChimeClock) (hour : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 44 seconds to complete chimes at 12 o'clock -/
theorem chime_time_at_12 (clock : ChimeClock) 
  (h1 : clock.time_at_4 = 12) 
  (h2 : clock.chimes_hourly) : 
  chime_time clock 12 = 44 :=
sorry

end NUMINAMATH_CALUDE_chime_time_at_12_l3062_306277


namespace NUMINAMATH_CALUDE_square_sum_of_system_l3062_306232

theorem square_sum_of_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_system_l3062_306232


namespace NUMINAMATH_CALUDE_farm_ploughing_problem_l3062_306257

/-- Calculates the actual ploughing rate given the conditions of the farm problem -/
def actualPloughingRate (totalArea plannedRate extraDays unploughedArea : ℕ) : ℕ :=
  let plannedDays := totalArea / plannedRate
  let actualDays := plannedDays + extraDays
  let ploughedArea := totalArea - unploughedArea
  ploughedArea / actualDays

/-- Theorem stating the actual ploughing rate for the given farm problem -/
theorem farm_ploughing_problem :
  actualPloughingRate 3780 90 2 40 = 85 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_problem_l3062_306257


namespace NUMINAMATH_CALUDE_rain_probability_l3062_306292

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.2)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.44 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l3062_306292


namespace NUMINAMATH_CALUDE_female_students_count_l3062_306294

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 85)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l3062_306294


namespace NUMINAMATH_CALUDE_line_through_circle_center_l3062_306201

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- 
If the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0,
then a = 1
-/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l3062_306201


namespace NUMINAMATH_CALUDE_trajectory_characterization_l3062_306265

-- Define the fixed points
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  |P.1 - F₁.1| + |P.2 - F₁.2| - (|P.1 - F₂.1| + |P.2 - F₂.2|) = 2 * a

-- Define what it means to be on one branch of a hyperbola
def on_hyperbola_branch (P : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ satisfies_condition P a ∧ 
  (P.1 < -5 ∨ (P.1 > 5 ∧ P.2 ≠ 0))

-- Define what it means to be on a ray starting from (5, 0) in positive x direction
def on_positive_x_ray (P : ℝ × ℝ) : Prop :=
  P.2 = 0 ∧ P.1 ≥ 5

theorem trajectory_characterization :
  (∀ P : ℝ × ℝ, satisfies_condition P 3 → on_hyperbola_branch P) ∧
  (∀ P : ℝ × ℝ, satisfies_condition P 5 → on_positive_x_ray P) :=
sorry

end NUMINAMATH_CALUDE_trajectory_characterization_l3062_306265


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3062_306216

/-- Given a class of students, calculate the number of students who play both football and long tennis. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 35) 
  (h2 : football = 26) 
  (h3 : tennis = 20) 
  (h4 : neither = 6) : 
  football + tennis - (total - neither) = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l3062_306216


namespace NUMINAMATH_CALUDE_exists_zero_term_l3062_306210

def recursion (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 
    (a n ≥ b n → a (n + 1) = a n - b n ∧ b (n + 1) = 2 * b n) ∧
    (a n < b n → a (n + 1) = 2 * a n ∧ b (n + 1) = b n - a n)

theorem exists_zero_term (a b : ℕ → ℕ) :
  recursion a b →
  (∃ k : ℕ, a k = 0) ↔
  (∃ m : ℕ, m > 0 ∧ (a 1 + b 1) / Nat.gcd (a 1) (b 1) = 2^m) :=
sorry

end NUMINAMATH_CALUDE_exists_zero_term_l3062_306210


namespace NUMINAMATH_CALUDE_ratio_b_to_sum_ac_l3062_306289

theorem ratio_b_to_sum_ac (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (c_eq : c = 35) : 
  b * 5 = a + c := by sorry

end NUMINAMATH_CALUDE_ratio_b_to_sum_ac_l3062_306289


namespace NUMINAMATH_CALUDE_intersection_points_range_l3062_306293

/-- The range of m for which curves C₁ and C₂ have 4 distinct intersection points -/
theorem intersection_points_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ), 
    (∀ i j, (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (i - 1)^2 + j^2 = 1 ∧ j * (j - m*i - m) = 0) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (k, l) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → (i, j) ≠ (k, l))) ↔ 
  (m > -Real.sqrt 3 / 3 ∧ m < 0) ∨ (m > 0 ∧ m < Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_range_l3062_306293


namespace NUMINAMATH_CALUDE_total_cats_l3062_306279

/-- Represents the Clevercat Academy with cats that can perform various tricks. -/
structure ClevercatAcademy where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The theorem states that given the specific numbers of cats that can perform
    various combinations of tricks, the total number of cats in the academy is 99. -/
theorem total_cats (academy : ClevercatAcademy)
  (h_jump : academy.jump = 60)
  (h_fetch : academy.fetch = 35)
  (h_spin : academy.spin = 40)
  (h_jump_fetch : academy.jump_fetch = 20)
  (h_fetch_spin : academy.fetch_spin = 15)
  (h_jump_spin : academy.jump_spin = 22)
  (h_all_three : academy.all_three = 11)
  (h_none : academy.none = 10) :
  (academy.jump - academy.jump_fetch - academy.jump_spin + academy.all_three) +
  (academy.fetch - academy.jump_fetch - academy.fetch_spin + academy.all_three) +
  (academy.spin - academy.jump_spin - academy.fetch_spin + academy.all_three) +
  academy.jump_fetch + academy.fetch_spin + academy.jump_spin -
  2 * academy.all_three + academy.none = 99 :=
by sorry

end NUMINAMATH_CALUDE_total_cats_l3062_306279


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3062_306273

theorem simplify_sqrt_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3062_306273


namespace NUMINAMATH_CALUDE_xyz_stock_price_evolution_l3062_306298

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem xyz_stock_price_evolution :
  stock_price_evolution 120 1 0.3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_xyz_stock_price_evolution_l3062_306298


namespace NUMINAMATH_CALUDE_sets_relationship_l3062_306253

def M : Set ℝ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℝ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℝ := {x | ∃ p : ℤ, x = p/2 + 1/6}

theorem sets_relationship : N = P ∧ M ≠ N :=
sorry

end NUMINAMATH_CALUDE_sets_relationship_l3062_306253


namespace NUMINAMATH_CALUDE_existence_of_xy_l3062_306200

theorem existence_of_xy : ∃ x y : ℕ+, 
  (x.val < 30 ∧ y.val < 30) ∧ 
  (x.val + y.val + x.val * y.val = 119) ∧
  (x.val + y.val = 20) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l3062_306200


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3062_306204

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + X^2 + 1 : Polynomial ℝ) = q * (X^2 - 4*X + 7) + (12*X - 69) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3062_306204


namespace NUMINAMATH_CALUDE_max_value_of_f_range_of_t_inequality_for_positive_reals_l3062_306278

-- Define the function f(x) = |x+1| - |x-2|
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The maximum value of f(x) is 3
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 3 :=
sorry

-- Theorem 2: The range of t given the inequality
theorem range_of_t : ∀ t : ℝ, (∃ x : ℝ, f x ≥ |t - 1| + t) ↔ t ≤ 2 :=
sorry

-- Theorem 3: Inequality for positive real numbers
theorem inequality_for_positive_reals :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 2*a + b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_range_of_t_inequality_for_positive_reals_l3062_306278


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l3062_306235

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

theorem trailing_zeros_30_factorial :
  trailingZeros (factorial 30) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l3062_306235


namespace NUMINAMATH_CALUDE_modulus_of_special_complex_l3062_306258

/-- The modulus of a complex number Z = 3a - 4ai where a < 0 is equal to -5a -/
theorem modulus_of_special_complex (a : ℝ) (ha : a < 0) :
  Complex.abs (Complex.mk (3 * a) (-4 * a)) = -5 * a := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_special_complex_l3062_306258


namespace NUMINAMATH_CALUDE_complex_root_pair_l3062_306283

theorem complex_root_pair (z : ℂ) :
  (3 + 8*I : ℂ)^2 = -55 + 48*I →
  z^2 = -55 + 48*I →
  z = 3 + 8*I ∨ z = -3 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_root_pair_l3062_306283


namespace NUMINAMATH_CALUDE_milburg_population_l3062_306280

/-- The total population of Milburg is the sum of grown-ups and children. -/
theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l3062_306280


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l3062_306295

def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def uses_digits (n : ℕ) (digits : List ℕ) : Prop :=
  (n.digits 10).toFinset = digits.toFinset

theorem greatest_five_digit_multiple_of_6 :
  ∃ (n : ℕ),
    n ≥ 10000 ∧
    n < 100000 ∧
    is_multiple_of_6 n ∧
    uses_digits n [2, 5, 6, 8, 9] ∧
    ∀ (m : ℕ),
      m ≥ 10000 →
      m < 100000 →
      is_multiple_of_6 m →
      uses_digits m [2, 5, 6, 8, 9] →
      m ≤ n ∧
    n = 98652 :=
  sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l3062_306295


namespace NUMINAMATH_CALUDE_weight_problem_l3062_306202

theorem weight_problem (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 84) →
  ((a + b + c + d) / 4 = 80) →
  (e = d + 3) →
  ((b + c + d + e) / 4 = 79) →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l3062_306202


namespace NUMINAMATH_CALUDE_shop_discount_percentage_l3062_306272

/-- Calculate the percentage discount given the original price and discounted price -/
def calculate_discount_percentage (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- The shop's discount percentage is 30% -/
theorem shop_discount_percentage :
  let original_price : ℚ := 800
  let discounted_price : ℚ := 560
  calculate_discount_percentage original_price discounted_price = 30 := by
sorry

end NUMINAMATH_CALUDE_shop_discount_percentage_l3062_306272


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3062_306227

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l3062_306227


namespace NUMINAMATH_CALUDE_pizza_slices_left_l3062_306211

theorem pizza_slices_left (total_slices : ℕ) (eaten_fraction : ℚ) (h1 : total_slices = 16) (h2 : eaten_fraction = 3/4) : 
  total_slices * (1 - eaten_fraction) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l3062_306211


namespace NUMINAMATH_CALUDE_meeting_point_distance_l3062_306206

/-- 
Given two people walking towards each other from a distance of 50 km, 
with one person walking at 4 km/h and the other at 6 km/h, 
the distance traveled by the slower person when they meet is 20 km.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (h1 : total_distance = 50) 
  (h2 : speed_a = 4) 
  (h3 : speed_b = 6) : 
  (total_distance * speed_a) / (speed_a + speed_b) = 20 := by
sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l3062_306206


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3062_306244

/-- Given a boat that travels 20 km downstream in 2 hours and 20 km upstream in 5 hours,
    prove that its speed in still water is 7 km/h. -/
theorem boat_speed_in_still_water :
  ∀ (downstream_speed upstream_speed : ℝ),
  downstream_speed = 20 / 2 →
  upstream_speed = 20 / 5 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_speed ∧
    boat_speed - stream_speed = upstream_speed ∧
    boat_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3062_306244


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3062_306228

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (3^11 + 5^13) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q ∧
    p = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3062_306228


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3062_306239

-- Define a line passing through (1, 3) with equal intercepts
def line_equal_intercepts (a b c : ℝ) : Prop :=
  a * 1 + b * 3 + c = 0 ∧  -- Line passes through (1, 3)
  ∃ k : ℝ, k ≠ 0 ∧ a * k + c = 0 ∧ b * k + c = 0  -- Equal intercepts

-- Theorem statement
theorem line_through_point_equal_intercepts :
  ∀ a b c : ℝ, line_equal_intercepts a b c →
  (a = -3 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3062_306239


namespace NUMINAMATH_CALUDE_pasture_rent_is_140_l3062_306270

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  payment : ℕ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def totalRent (a b c : RentShare) : ℕ :=
  let totalOxenMonths := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let costPerOxenMonth := c.payment / (c.oxen * c.months)
  costPerOxenMonth * totalOxenMonths

/-- Theorem stating that the total rent of the pasture is 140 -/
theorem pasture_rent_is_140 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.payment = 36) :
  totalRent a b c = 140 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_is_140_l3062_306270


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_l3062_306251

/-- Represents a regression model with its R² value -/
structure RegressionModel where
  r_squared : ℝ
  h_nonneg : 0 ≤ r_squared
  h_le_one : r_squared ≤ 1

/-- Determines if a model has better fit than another based on R² values -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.r_squared > model2.r_squared

/-- Theorem: The model with the highest R² value has the best fitting effect -/
theorem best_fit_highest_r_squared (models : List RegressionModel) (best_model : RegressionModel) 
    (h_best_in_models : best_model ∈ models)
    (h_best_r_squared : ∀ model ∈ models, model.r_squared ≤ best_model.r_squared) :
    ∀ model ∈ models, better_fit best_model model ∨ best_model = model :=
  sorry

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_l3062_306251


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l3062_306255

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l3062_306255


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3062_306212

theorem dice_roll_probability (m : ℝ) : 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → (x^2 : ℝ) + y^2 ≤ m) ↔ 
  72 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3062_306212


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3062_306266

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3062_306266


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3062_306259

theorem min_value_quadratic_sum (x y : ℝ) (h : x + y = 1) :
  ∀ z w : ℝ, z + w = 1 → 2 * x^2 + 3 * y^2 ≤ 2 * z^2 + 3 * w^2 ∧
  ∃ a b : ℝ, a + b = 1 ∧ 2 * a^2 + 3 * b^2 = 6/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3062_306259


namespace NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_naturals_l3062_306243

theorem largest_divisor_of_four_consecutive_naturals :
  ∀ n : ℕ, ∃ k : ℕ, k * 120 = n * (n + 1) * (n + 2) * (n + 3) ∧
  ∀ m : ℕ, m > 120 → ¬(∀ n : ℕ, ∃ k : ℕ, k * m = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_naturals_l3062_306243


namespace NUMINAMATH_CALUDE_container_capacity_is_20_l3062_306231

-- Define the capacity of the container
def container_capacity : ℝ := 20

-- Define the initial fill percentage
def initial_fill_percentage : ℝ := 0.30

-- Define the final fill percentage
def final_fill_percentage : ℝ := 0.75

-- Define the amount of water added
def water_added : ℝ := 9

-- Theorem stating the container capacity is 20 liters
theorem container_capacity_is_20 :
  (final_fill_percentage * container_capacity - initial_fill_percentage * container_capacity = water_added) ∧
  (container_capacity = 20) :=
sorry

end NUMINAMATH_CALUDE_container_capacity_is_20_l3062_306231


namespace NUMINAMATH_CALUDE_dividend_calculation_l3062_306262

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 40)
  (h2 : quotient = 6)
  (h3 : remainder = 28) :
  quotient * divisor + remainder = 268 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3062_306262


namespace NUMINAMATH_CALUDE_fourth_person_height_l3062_306248

/-- Theorem: Height of the fourth person in a specific arrangement --/
theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_first_three : h₂ - h₁ = 2 ∧ h₃ - h₂ = 2)
  (diff_last_two : h₄ - h₃ = 6)
  (average_height : (h₁ + h₂ + h₃ + h₄) / 4 = 76) :
  h₄ = 82 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3062_306248


namespace NUMINAMATH_CALUDE_congruence_problem_l3062_306217

theorem congruence_problem (c d : ℤ) (h_c : c ≡ 25 [ZMOD 53]) (h_d : d ≡ 88 [ZMOD 53]) :
  ∃ m : ℤ, m = 149 ∧ 150 ≤ m ∧ m ≤ 200 ∧ c - d ≡ m [ZMOD 53] ∧
  ∀ k : ℤ, 150 ≤ k ∧ k ≤ 200 ∧ c - d ≡ k [ZMOD 53] → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l3062_306217


namespace NUMINAMATH_CALUDE_remaining_milk_l3062_306269

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 2 + 3/4

-- Theorem statement
theorem remaining_milk :
  initial_milk - given_away = 2 + 1/4 := by sorry

end NUMINAMATH_CALUDE_remaining_milk_l3062_306269


namespace NUMINAMATH_CALUDE_band_members_count_l3062_306252

theorem band_members_count :
  ∃! N : ℕ, 100 < N ∧ N ≤ 200 ∧
  (∃ k : ℕ, N + 2 = 8 * k) ∧
  (∃ m : ℕ, N + 3 = 9 * m) :=
by sorry

end NUMINAMATH_CALUDE_band_members_count_l3062_306252


namespace NUMINAMATH_CALUDE_value_of_y_l3062_306203

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3062_306203


namespace NUMINAMATH_CALUDE_breakfast_calories_proof_l3062_306223

/-- Calculates the breakfast calories given the daily calorie limit, remaining calories, dinner calories, and lunch calories. -/
def breakfast_calories (daily_limit : ℕ) (remaining : ℕ) (dinner : ℕ) (lunch : ℕ) : ℕ :=
  daily_limit - remaining - (dinner + lunch)

/-- Proves that given the specific calorie values, the breakfast calories are 560. -/
theorem breakfast_calories_proof :
  breakfast_calories 2500 525 635 780 = 560 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_calories_proof_l3062_306223


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3062_306229

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3062_306229


namespace NUMINAMATH_CALUDE_locus_of_equidistant_points_l3062_306205

-- Define the oblique coordinate system
structure ObliqueCoordSystem where
  angle : ℝ
  e₁ : ℝ × ℝ
  e₂ : ℝ × ℝ

-- Define a point in the oblique coordinate system
structure ObliquePoint where
  x : ℝ
  y : ℝ

-- Define the locus equation
def locusEquation (p : ObliquePoint) : Prop :=
  Real.sqrt 2 * p.x + p.y = 0

-- State the theorem
theorem locus_of_equidistant_points
  (sys : ObliqueCoordSystem)
  (F₁ F₂ M : ObliquePoint)
  (h_angle : sys.angle = Real.pi / 4)
  (h_F₁ : F₁ = ⟨-1, 0⟩)
  (h_F₂ : F₂ = ⟨1, 0⟩)
  (h_equidistant : ‖(M.x - F₁.x, M.y - F₁.y)‖ = ‖(M.x - F₂.x, M.y - F₂.y)‖) :
  locusEquation M :=
sorry

end NUMINAMATH_CALUDE_locus_of_equidistant_points_l3062_306205


namespace NUMINAMATH_CALUDE_H_surjective_l3062_306261

-- Define the function H
def H (x : ℝ) : ℝ := 2 * |2 * x + 3| - 3 * |x - 2|

-- Theorem statement
theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l3062_306261


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3062_306207

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3062_306207


namespace NUMINAMATH_CALUDE_max_contribution_l3062_306291

theorem max_contribution 
  (n : ℕ) 
  (total : ℚ) 
  (min_contribution : ℚ) 
  (h1 : n = 15)
  (h2 : total = 30)
  (h3 : min_contribution = 1)
  (h4 : ∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≥ min_contribution) :
  ∃ max_contribution : ℚ, 
    max_contribution ≤ total ∧ 
    (∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≤ max_contribution) ∧
    max_contribution = 16 :=
sorry

end NUMINAMATH_CALUDE_max_contribution_l3062_306291


namespace NUMINAMATH_CALUDE_tim_initial_balls_l3062_306287

theorem tim_initial_balls (robert_initial : ℕ) (robert_final : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  robert_final = 45 → 
  robert_final = robert_initial + tim_initial / 2 → 
  tim_initial = 40 := by
sorry

end NUMINAMATH_CALUDE_tim_initial_balls_l3062_306287


namespace NUMINAMATH_CALUDE_twenty_one_three_four_zero_is_base5_l3062_306237

def is_base5_digit (d : Nat) : Prop := d < 5

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem twenty_one_three_four_zero_is_base5 :
  is_base5_number 21340 :=
sorry

end NUMINAMATH_CALUDE_twenty_one_three_four_zero_is_base5_l3062_306237


namespace NUMINAMATH_CALUDE_prob_red_or_green_is_two_thirds_l3062_306254

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 3
def green_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + green_balls

-- Define the number of favorable outcomes (red or green balls)
def favorable_outcomes : ℕ := red_balls + green_balls

-- Define the probability of drawing a red or green ball
def prob_red_or_green : ℚ := favorable_outcomes / total_balls

-- Theorem statement
theorem prob_red_or_green_is_two_thirds : 
  prob_red_or_green = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_green_is_two_thirds_l3062_306254


namespace NUMINAMATH_CALUDE_combination_square_28_l3062_306284

theorem combination_square_28 (n : ℕ) : (n.choose 2 = 28) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_square_28_l3062_306284


namespace NUMINAMATH_CALUDE_prime_sqrt_sum_integer_l3062_306234

theorem prime_sqrt_sum_integer (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, ∃ m : ℕ, (Nat.sqrt (p + n) + Nat.sqrt n : ℕ) = m :=
sorry

end NUMINAMATH_CALUDE_prime_sqrt_sum_integer_l3062_306234


namespace NUMINAMATH_CALUDE_special_circle_equation_l3062_306221

/-- A circle symmetric about the y-axis, passing through (1,0), 
    and divided by the x-axis into arc lengths with ratio 1:2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  symmetric_about_y_axis : center.1 = 0
  passes_through_1_0 : (1 - center.1)^2 + (0 - center.2)^2 = radius^2
  arc_ratio : Real.cos (Real.pi / 3) = center.2 / radius

/-- The equation of the special circle -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem special_circle_equation (c : SpecialCircle) :
  ∃ a : ℝ, a = Real.sqrt 3 / 3 ∧
    (∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - a)^2 = 4/3 ∨ x^2 + (y + a)^2 = 4/3) :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l3062_306221


namespace NUMINAMATH_CALUDE_right_triangle_area_l3062_306236

/-- Given a right-angled triangle with height 5 cm and median to hypotenuse 6 cm, its area is 30 cm². -/
theorem right_triangle_area (h : ℝ) (m : ℝ) (area : ℝ) : 
  h = 5 → m = 6 → area = (1/2) * (2*m) * h → area = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3062_306236


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3062_306260

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  (x^3 % p = y^3 % p) → (y^3 % p = z^3 % p) →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_l3062_306260


namespace NUMINAMATH_CALUDE_computer_table_markup_l3062_306209

/-- The percentage markup on a product's cost price, given its selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Theorem stating that the percentage markup on a computer table with a selling price of 8215 
    and a cost price of 6625 is 24%. -/
theorem computer_table_markup :
  percentageMarkup 8215 6625 = 24 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l3062_306209


namespace NUMINAMATH_CALUDE_min_value_theorem_l3062_306230

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : m * 1 - n * (-1) - 1 = 0) : 
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l3062_306230


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3062_306250

/-- The coefficient of x³y²z in the expansion of (x+y+z)⁶ -/
def coefficient_x3y2z : ℕ :=
  Nat.choose 6 3 * Nat.choose 3 2

theorem expansion_coefficient :
  coefficient_x3y2z = 60 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3062_306250


namespace NUMINAMATH_CALUDE_floor_ceiling_difference_l3062_306276

theorem floor_ceiling_difference : 
  ⌊(14 : ℝ) / 5 * (31 : ℝ) / 4⌋ - ⌈(14 : ℝ) / 5 * ⌈(31 : ℝ) / 4⌉⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_difference_l3062_306276


namespace NUMINAMATH_CALUDE_meeting_speed_l3062_306263

theorem meeting_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 200)
  (h2 : time = 8)
  (h3 : speed_difference = 7) :
  ∃ (speed : ℝ), 
    speed > 0 ∧ 
    (speed + (speed + speed_difference)) * time = distance ∧
    speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_meeting_speed_l3062_306263


namespace NUMINAMATH_CALUDE_blown_away_leaves_calculation_mikeys_leaves_calculation_l3062_306218

/-- Given an initial number of leaves and the number of leaves remaining,
    calculate the number of leaves that blew away. -/
def leaves_blown_away (initial_leaves remaining_leaves : ℕ) : ℕ :=
  initial_leaves - remaining_leaves

/-- Theorem: The number of leaves that blew away is equal to the difference
    between the initial number of leaves and the remaining number of leaves. -/
theorem blown_away_leaves_calculation 
  (initial_leaves remaining_leaves : ℕ) 
  (h : initial_leaves ≥ remaining_leaves) :
  leaves_blown_away initial_leaves remaining_leaves = initial_leaves - remaining_leaves :=
by
  sorry

/-- In Mikey's specific case -/
theorem mikeys_leaves_calculation :
  leaves_blown_away 356 112 = 244 :=
by
  sorry

end NUMINAMATH_CALUDE_blown_away_leaves_calculation_mikeys_leaves_calculation_l3062_306218


namespace NUMINAMATH_CALUDE_bobby_paycheck_l3062_306215

/-- Calculates the final paycheck amount given the salary and deductions --/
def final_paycheck_amount (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (salary * federal_tax_rate + salary * state_tax_rate + 
    health_insurance + life_insurance + parking_fee)

/-- Theorem stating that Bobby's final paycheck amount is $184 --/
theorem bobby_paycheck :
  final_paycheck_amount 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end NUMINAMATH_CALUDE_bobby_paycheck_l3062_306215


namespace NUMINAMATH_CALUDE_angle_construction_error_bound_l3062_306226

/-- Represents a 4-digit trigonometric table -/
structure TrigTable :=
  (sin : ℚ → ℚ)
  (cos : ℚ → ℚ)
  (precision : ℕ := 4)

/-- Represents the construction of a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)
  (centralAngle : ℚ)

/-- The error bound for angle construction using a 4-digit trig table -/
def angleErrorBound (p : RegularPolygon) (t : TrigTable) : ℚ := sorry

theorem angle_construction_error_bound 
  (p : RegularPolygon) 
  (t : TrigTable) 
  (h1 : p.sides = 18) 
  (h2 : p.centralAngle = 20) 
  (h3 : t.precision = 4) :
  angleErrorBound p t < 21 / 3600 := by sorry

end NUMINAMATH_CALUDE_angle_construction_error_bound_l3062_306226


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3062_306247

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < 0) :
  Real.sqrt (y / (1 - (y - 2) / y)) = -y / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l3062_306247


namespace NUMINAMATH_CALUDE_initial_workers_count_l3062_306274

/-- Represents the construction project scenario -/
structure ConstructionProject where
  initial_duration : ℕ
  actual_duration : ℕ
  initial_workers : ℕ
  double_rate_workers : ℕ
  triple_rate_workers : ℕ
  double_rate_join_day : ℕ
  triple_rate_join_day : ℕ

/-- Theorem stating that the initial number of workers is 55 -/
theorem initial_workers_count (project : ConstructionProject) 
  (h1 : project.initial_duration = 24)
  (h2 : project.actual_duration = 19)
  (h3 : project.double_rate_workers = 8)
  (h4 : project.triple_rate_workers = 5)
  (h5 : project.double_rate_join_day = 11)
  (h6 : project.triple_rate_join_day = 17) :
  project.initial_workers = 55 := by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l3062_306274


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l3062_306256

theorem complex_power_one_minus_i_six :
  let i : ℂ := Complex.I
  (1 - i)^6 = 8*i := by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l3062_306256


namespace NUMINAMATH_CALUDE_area_NPQ_approx_l3062_306233

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  xy_length : dist X Y = 15
  xz_length : dist X Z = 20
  yz_length : dist Y Z = 13

/-- P is the circumcenter of triangle XYZ -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Q is the incenter of triangle XYZ -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- N is the center of a circle tangent to sides XZ, YZ, and the circumcircle of XYZ -/
def excircle_center (t : Triangle) : ℝ × ℝ := sorry

/-- The area of triangle NPQ -/
def area_NPQ (t : Triangle) : ℝ := sorry

/-- Theorem stating the area of triangle NPQ is approximately 49.21 -/
theorem area_NPQ_approx (t : Triangle) : 
  abs (area_NPQ t - 49.21) < 0.01 := by sorry

end NUMINAMATH_CALUDE_area_NPQ_approx_l3062_306233


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3062_306288

theorem quadratic_root_relation (p r : ℝ) (hr : r > 0) :
  (∃ x y : ℝ, x^2 + p*x + r = 0 ∧ y^2 + p*y + r = 0 ∧ y = 2*x) →
  p = Real.sqrt (9*r/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3062_306288


namespace NUMINAMATH_CALUDE_dolphin_altitude_l3062_306286

/-- Given a submarine at an altitude of -50 meters and a dolphin 10 meters above it,
    the altitude of the dolphin is -40 meters. -/
theorem dolphin_altitude (submarine_altitude dolphin_distance : ℝ) :
  submarine_altitude = -50 ∧ dolphin_distance = 10 →
  submarine_altitude + dolphin_distance = -40 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_altitude_l3062_306286


namespace NUMINAMATH_CALUDE_initial_articles_sold_l3062_306271

/-- The number of articles sold to gain 20% when the total selling price is $60 -/
def articles_sold_gain (n : ℕ) : Prop :=
  ∃ (cp : ℚ), 1.2 * cp * n = 60

/-- The number of articles that should be sold to incur a loss of 20% when the total selling price is $60 -/
def articles_sold_loss : ℚ := 29.99999625000047

/-- The proposition that the initial number of articles sold is correct -/
def correct_initial_articles (n : ℕ) : Prop :=
  articles_sold_gain n ∧
  ∃ (cp : ℚ), 0.8 * cp * articles_sold_loss = 60 ∧
              cp * articles_sold_loss = 75 ∧
              cp * n = 50

theorem initial_articles_sold :
  ∃ (n : ℕ), correct_initial_articles n ∧ n = 20 := by sorry

end NUMINAMATH_CALUDE_initial_articles_sold_l3062_306271


namespace NUMINAMATH_CALUDE_polynomial_characterization_l3062_306224

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by a, b, and c -/
def SatisfiesCondition (a b c : ℝ) : Prop :=
  a * b + b * c + c * a = 0

/-- The equation that P must satisfy for all a, b, c satisfying the condition -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The form of the polynomial we're trying to prove -/
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ x, P x = α * x^4 + β * x^2

theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesEquation P → IsQuarticQuadratic P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l3062_306224


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l3062_306219

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (5, -3)
  dot_product a b = 7 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l3062_306219


namespace NUMINAMATH_CALUDE_point_above_line_l3062_306275

/-- A point (x, y) is above a line Ax + By + C = 0 if Ax + By + C < 0 -/
def IsAboveLine (x y A B C : ℝ) : Prop := A * x + B * y + C < 0

/-- The theorem states that for the point (-3, -1) to be above the line 3x - 2y - a = 0,
    a must be greater than -7 -/
theorem point_above_line (a : ℝ) :
  IsAboveLine (-3) (-1) 3 (-2) (-a) ↔ a > -7 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l3062_306275


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3062_306214

theorem fraction_evaluation (x : ℝ) (h : x = 8) :
  (x^10 - 32*x^5 + 1024) / (x^5 - 32) = 32768 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3062_306214


namespace NUMINAMATH_CALUDE_billy_lemon_heads_l3062_306282

/-- The number of friends Billy gave Lemon Heads to -/
def num_friends : ℕ := 6

/-- The number of Lemon Heads each friend ate -/
def lemon_heads_per_friend : ℕ := 12

/-- The initial number of Lemon Heads Billy had -/
def initial_lemon_heads : ℕ := num_friends * lemon_heads_per_friend

theorem billy_lemon_heads :
  initial_lemon_heads = 72 :=
by sorry

end NUMINAMATH_CALUDE_billy_lemon_heads_l3062_306282


namespace NUMINAMATH_CALUDE_thursday_return_count_l3062_306240

/-- Calculates the number of books brought back on Thursday given the initial
    number of books, books taken out on Tuesday and Friday, and the final
    number of books in the library. -/
def books_brought_back (initial : ℕ) (taken_tuesday : ℕ) (taken_friday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + taken_friday - final

theorem thursday_return_count :
  books_brought_back 235 227 35 29 = 56 := by
  sorry

end NUMINAMATH_CALUDE_thursday_return_count_l3062_306240


namespace NUMINAMATH_CALUDE_max_boxes_A_l3062_306225

def price_A : ℝ := 24
def price_B : ℝ := 16
def total_boxes : ℕ := 200
def max_cost : ℝ := 3920

theorem max_boxes_A : 
  price_A + 2 * price_B = 56 →
  2 * price_A + price_B = 64 →
  (∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost →
    m ≤ 90) ∧
  (∃ m : ℕ, m = 90 ∧ 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost) :=
by sorry

end NUMINAMATH_CALUDE_max_boxes_A_l3062_306225


namespace NUMINAMATH_CALUDE_complex_cosine_geometric_representation_l3062_306249

/-- The set of points represented by z = i cos θ, where θ ∈ [0, 2π], 
    is equal to the line segment from (0, -1) to (0, 1) in the complex plane. -/
theorem complex_cosine_geometric_representation :
  {z : ℂ | ∃ θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) ∧ z = Complex.I * Complex.cos θ} =
  {z : ℂ | z.re = 0 ∧ z.im ∈ Set.Icc (-1) 1} :=
sorry

end NUMINAMATH_CALUDE_complex_cosine_geometric_representation_l3062_306249


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3062_306290

theorem units_digit_sum_of_powers : (2016^2017 + 2017^2016) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3062_306290


namespace NUMINAMATH_CALUDE_quarter_probability_l3062_306245

/-- The probability of choosing a quarter from a jar containing quarters, nickels, and pennies -/
theorem quarter_probability (quarter_value nickel_value penny_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value : ℚ)
  (h_quarter : quarter_value = 25/100)
  (h_nickel : nickel_value = 5/100)
  (h_penny : penny_value = 1/100)
  (h_total_quarter : total_quarter_value = 15/2)
  (h_total_nickel : total_nickel_value = 25/2)
  (h_total_penny : total_penny_value = 15) :
  (total_quarter_value / quarter_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value)) = 15/890 := by
  sorry


end NUMINAMATH_CALUDE_quarter_probability_l3062_306245


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l3062_306297

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l3062_306297


namespace NUMINAMATH_CALUDE_bennett_window_screens_l3062_306242

theorem bennett_window_screens (january february march : ℕ) : 
  february = 2 * january →
  february = march / 4 →
  january + february + march = 12100 →
  march = 8800 := by sorry

end NUMINAMATH_CALUDE_bennett_window_screens_l3062_306242


namespace NUMINAMATH_CALUDE_symmetric_points_a_value_l3062_306222

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- Given that point A(a,1) is symmetric to point B(-3,-1) with respect to the origin, prove that a = 3 -/
theorem symmetric_points_a_value :
  ∀ a : ℝ, symmetric_wrt_origin (a, 1) (-3, -1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_a_value_l3062_306222


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l3062_306238

/-- Given a trader who sells cloth, this theorem proves the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 95 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l3062_306238


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l3062_306220

theorem number_of_divisors_30030 : Nat.card {d : ℕ | d > 0 ∧ 30030 % d = 0} = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l3062_306220


namespace NUMINAMATH_CALUDE_min_value_expression_l3062_306299

theorem min_value_expression (x y : ℝ) : 
  (x + y - 1)^2 + (x * y)^2 ≥ 0 ∧ 
  ∃ a b : ℝ, (a + b - 1)^2 + (a * b)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3062_306299


namespace NUMINAMATH_CALUDE_y_coordinate_range_of_C_l3062_306213

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define point A
def A : ℝ × ℝ := (0, 2)

-- Define perpendicularity of line segments
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem y_coordinate_range_of_C 
  (B C : ℝ × ℝ) 
  (hB : parabola B.1 B.2)
  (hC : parabola C.1 C.2)
  (h_perp : perpendicular A B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_y_coordinate_range_of_C_l3062_306213


namespace NUMINAMATH_CALUDE_students_satisfy_equation_unique_solution_l3062_306241

/-- The number of students in class 5A -/
def students : ℕ := 36

/-- The equation that describes the problem conditions -/
def problem_equation (x : ℕ) : Prop :=
  (x - 23) * 23 = (x - 13) * 13

/-- Theorem stating that the number of students in class 5A satisfies the problem conditions -/
theorem students_satisfy_equation : problem_equation students := by
  sorry

/-- Theorem stating that 36 is the unique solution to the problem -/
theorem unique_solution : ∀ x : ℕ, problem_equation x → x = students := by
  sorry

end NUMINAMATH_CALUDE_students_satisfy_equation_unique_solution_l3062_306241


namespace NUMINAMATH_CALUDE_taco_truck_lunch_rush_earnings_l3062_306208

/-- Calculates the total earnings of a taco truck during lunch rush -/
def taco_truck_earnings (soft_taco_price : ℕ) (hard_taco_price : ℕ) 
  (family_hard_tacos : ℕ) (family_soft_tacos : ℕ) 
  (other_customers : ℕ) (tacos_per_customer : ℕ) : ℕ :=
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price) + 
  (other_customers * tacos_per_customer * soft_taco_price)

/-- The taco truck's earnings during lunch rush is $66 -/
theorem taco_truck_lunch_rush_earnings : 
  taco_truck_earnings 2 5 4 3 10 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_lunch_rush_earnings_l3062_306208


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3062_306296

theorem sin_equality_proof (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (945 * π / 180) → m = -135 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3062_306296


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l3062_306264

/-- Given a parabola with equation y = 2x^2 + 8x - 1, its focus coordinates are (-2, -8.875) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 8 * x - 1
  ∃ (h k : ℝ), h = -2 ∧ k = -8.875 ∧
    ∀ (x y : ℝ), y = f x → (x - h)^2 = 4 * (y - k) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l3062_306264


namespace NUMINAMATH_CALUDE_digital_root_theorem_l3062_306267

/-- Digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

/-- List of digital roots of first n natural numbers -/
def digitalRootList (n : ℕ) : List ℕ :=
  List.map digitalRoot (List.range n)

theorem digital_root_theorem :
  let l := digitalRootList 20092009
  (l.count 4 > l.count 5) ∧
  (l.count 9 = 2232445) ∧
  (digitalRoot (3^2009) = 9) ∧
  (digitalRoot (17^2009) = 8) := by
  sorry


end NUMINAMATH_CALUDE_digital_root_theorem_l3062_306267


namespace NUMINAMATH_CALUDE_median_salary_is_45000_l3062_306285

structure Position :=
  (title : String)
  (count : ℕ)
  (salary : ℕ)

def company_data : List Position := [
  ⟨"CEO", 1, 150000⟩,
  ⟨"Senior Manager", 4, 95000⟩,
  ⟨"Manager", 15, 70000⟩,
  ⟨"Assistant Manager", 20, 45000⟩,
  ⟨"Clerk", 40, 18000⟩
]

def total_employees : ℕ := (company_data.map Position.count).sum

def median_salary (data : List Position) : ℕ := 
  if total_employees % 2 = 0 
  then 45000  -- As both (total_employees / 2) and (total_employees / 2 + 1) fall under Assistant Manager
  else 45000  -- As (total_employees / 2 + 1) falls under Assistant Manager

theorem median_salary_is_45000 : 
  median_salary company_data = 45000 := by sorry

end NUMINAMATH_CALUDE_median_salary_is_45000_l3062_306285


namespace NUMINAMATH_CALUDE_sameColorPairsTheorem_l3062_306281

/-- The number of ways to choose a pair of socks of the same color from a drawer -/
def sameColorPairs (white green brown blue : ℕ) : ℕ :=
  Nat.choose white 2 + Nat.choose green 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: Given a drawer with 16 distinguishable socks (6 white, 4 green, 4 brown, and 2 blue),
    the number of ways to choose a pair of socks of the same color is 28. -/
theorem sameColorPairsTheorem :
  sameColorPairs 6 4 4 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_sameColorPairsTheorem_l3062_306281


namespace NUMINAMATH_CALUDE_cold_water_time_l3062_306246

/-- The combined total time Jerry and his friends spent in the cold water pool --/
def total_time (jerry_time elaine_time george_time kramer_time : ℝ) : ℝ :=
  jerry_time + elaine_time + george_time + kramer_time

/-- Theorem stating the total time spent in the cold water pool --/
theorem cold_water_time : ∃ (jerry_time elaine_time george_time kramer_time : ℝ),
  jerry_time = 3 ∧
  elaine_time = 2 * jerry_time ∧
  george_time = (1/3) * elaine_time ∧
  kramer_time = 0 ∧
  total_time jerry_time elaine_time george_time kramer_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_cold_water_time_l3062_306246
