import Mathlib

namespace NUMINAMATH_CALUDE_chloe_carrots_theorem_l2812_281278

/-- Calculates the total number of carrots Chloe has after throwing some out and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Chloe's total carrots is correct given the initial conditions. -/
theorem chloe_carrots_theorem (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by sorry

end NUMINAMATH_CALUDE_chloe_carrots_theorem_l2812_281278


namespace NUMINAMATH_CALUDE_circle_radius_with_tangents_l2812_281219

/-- Given a circle with parallel tangents and a third tangent, prove the radius. -/
theorem circle_radius_with_tangents 
  (AB CD DE : ℝ) 
  (h_AB : AB = 7)
  (h_CD : CD = 12)
  (h_DE : DE = 3) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_with_tangents_l2812_281219


namespace NUMINAMATH_CALUDE_function_equality_implies_zero_l2812_281245

/-- Given a function f(x, y) = kx + 1/y, prove that if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_equality_implies_zero (k : ℝ) (a b : ℝ) (h1 : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_zero_l2812_281245


namespace NUMINAMATH_CALUDE_fifteen_machines_six_minutes_l2812_281287

/-- The number of paperclips produced by a given number of machines in a given time -/
def paperclips_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let base_machines := 8
  let base_production := 560
  let production_per_machine := base_production / base_machines
  machines * production_per_machine * minutes

/-- Theorem stating that 15 machines will produce 6300 paperclips in 6 minutes -/
theorem fifteen_machines_six_minutes :
  paperclips_produced 15 6 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_machines_six_minutes_l2812_281287


namespace NUMINAMATH_CALUDE_triangle_properties_l2812_281236

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (a * Real.cos B - b * Real.cos A = c - b) →
  (Real.tan A + Real.tan B + Real.tan C - Real.sqrt 3 * Real.tan B * Real.tan C = 0) →
  ((1/2) * a * (b * Real.sin B + c * Real.sin C - a * Real.sin A) = (1/2) * a * b * Real.sin C) →
  -- Conclusions
  (A = π/3) ∧
  (a = 8 → (1/2) * a * b * Real.sin C = 11 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2812_281236


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2812_281254

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l2812_281254


namespace NUMINAMATH_CALUDE_louis_current_age_l2812_281241

/-- Carla's current age -/
def carla_age : ℕ := 30 - 6

/-- Louis's current age -/
def louis_age : ℕ := 55 - carla_age

theorem louis_current_age : louis_age = 31 := by
  sorry

end NUMINAMATH_CALUDE_louis_current_age_l2812_281241


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2812_281217

-- Define the conditions
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

def condition_q (m : ℝ) : Prop :=
  ∀ x, x^2 - 2*m*x + m^2 - 1 = 0 → -2 < x ∧ x < 4

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m, condition_q m → condition_p m) ∧
  (∃ m, condition_p m ∧ ¬condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2812_281217


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_main_result_l2812_281232

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^4) = (376 : ℚ) / 10000 := by sorry

theorem decimal_representation : (376 : ℚ) / 10000 = 0.0376 := by sorry

theorem main_result : (47 : ℚ) / (2 * 5^4) = 0.0376 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_main_result_l2812_281232


namespace NUMINAMATH_CALUDE_kamal_english_marks_l2812_281225

/-- Kamal's marks in English given his other marks and average --/
theorem kamal_english_marks (math physics chem bio : ℕ) (avg : ℚ) :
  math = 65 →
  physics = 82 →
  chem = 67 →
  bio = 85 →
  avg = 75 →
  (math + physics + chem + bio + english : ℚ) / 5 = avg →
  english = 76 := by
  sorry

end NUMINAMATH_CALUDE_kamal_english_marks_l2812_281225


namespace NUMINAMATH_CALUDE_candy_container_count_l2812_281239

theorem candy_container_count : ℕ := by
  -- Define the number of people
  let people : ℕ := 157

  -- Define the number of candies each person receives
  let candies_per_person : ℕ := 235

  -- Define the number of candies left after distribution
  let leftover_candies : ℕ := 98

  -- Define the total number of candies
  let total_candies : ℕ := people * candies_per_person + leftover_candies

  -- Prove that the total number of candies is 36,993
  have h : total_candies = 36993 := by sorry

  -- Return the result
  exact 36993

end NUMINAMATH_CALUDE_candy_container_count_l2812_281239


namespace NUMINAMATH_CALUDE_tom_drives_12_miles_l2812_281270

/-- A car race between Karen and Tom -/
structure CarRace where
  karen_speed : ℝ  -- Karen's speed in mph
  tom_speed : ℝ    -- Tom's speed in mph
  karen_delay : ℝ  -- Karen's delay in minutes
  win_margin : ℝ   -- Karen's winning margin in miles

/-- Calculate the distance Tom drives before Karen wins -/
def distance_tom_drives (race : CarRace) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 12 miles before Karen wins -/
theorem tom_drives_12_miles (race : CarRace) 
  (h1 : race.karen_speed = 60)
  (h2 : race.tom_speed = 45)
  (h3 : race.karen_delay = 4)
  (h4 : race.win_margin = 4) :
  distance_tom_drives race = 12 :=
sorry

end NUMINAMATH_CALUDE_tom_drives_12_miles_l2812_281270


namespace NUMINAMATH_CALUDE_no_seventh_power_sum_l2812_281263

def a : ℕ → ℤ
  | 0 => 8
  | 1 => 20
  | (n + 2) => (a (n + 1))^2 + 12 * (a (n + 1)) * (a n) + (a (n + 1)) + 11 * (a n)

def seventh_power_sum_mod_29 (x y z : ℤ) : ℤ :=
  ((x^7 % 29) + (y^7 % 29) + (z^7 % 29)) % 29

theorem no_seventh_power_sum (n : ℕ) :
  ∀ x y z : ℤ, (a n) % 29 ≠ seventh_power_sum_mod_29 x y z :=
by sorry

end NUMINAMATH_CALUDE_no_seventh_power_sum_l2812_281263


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_100_unique_solution_for_105_l2812_281271

/-- Represents the score calculation function for the math examination. -/
def score (c w : ℕ) : ℕ := 50 + 5 * c - 2 * w

/-- Theorem stating that 105 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  ∀ s : ℕ, s > 100 → s < 105 → 
  (∃ c w : ℕ, c + w ≤ 50 ∧ score c w = s) → 
  (∃ c₁ w₁ c₂ w₂ : ℕ, 
    c₁ + w₁ ≤ 50 ∧ c₂ + w₂ ≤ 50 ∧ 
    score c₁ w₁ = s ∧ score c₂ w₂ = s ∧ 
    (c₁ ≠ c₂ ∨ w₁ ≠ w₂)) :=
by sorry

/-- Theorem stating that 105 has a unique solution for c and w. -/
theorem unique_solution_for_105 : 
  ∃! c w : ℕ, c + w ≤ 50 ∧ score c w = 105 :=
by sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_100_unique_solution_for_105_l2812_281271


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2812_281267

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value 
  (l1 : ℝ → ℝ → Prop) 
  (l2 : ℝ → ℝ → Prop)
  (a : ℝ) 
  (h1 : ∀ x y, l1 x y ↔ x + 2*a*y - 1 = 0)
  (h2 : ∀ x y, l2 x y ↔ x - 4*y = 0)
  (h3 : perpendicular (2*a) (1/4)) :
  a = 1/8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2812_281267


namespace NUMINAMATH_CALUDE_conditional_inequality_l2812_281277

theorem conditional_inequality (a b c : ℝ) (h1 : c > 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

end NUMINAMATH_CALUDE_conditional_inequality_l2812_281277


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2812_281208

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - 3*y - 3

/-- Point A coordinates -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (3, 0)

/-- Point C coordinates -/
def C : ℝ × ℝ := (1, 4)

/-- Theorem: The circle_equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation A.1 A.2 = 0 ∧
  circle_equation B.1 B.2 = 0 ∧
  circle_equation C.1 C.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2812_281208


namespace NUMINAMATH_CALUDE_cold_brew_time_per_batch_l2812_281272

/-- Proves that the time to make one batch of cold brew coffee is 20 hours -/
theorem cold_brew_time_per_batch : 
  ∀ (batch_size : ℝ) (daily_consumption : ℝ) (total_time : ℝ) (total_days : ℕ),
    batch_size = 1.5 →  -- size of one batch in gallons
    daily_consumption = 48 →  -- 96 ounces every 2 days = 48 ounces per day
    total_time = 120 →  -- total hours spent making coffee
    total_days = 24 →  -- number of days
    (total_time / (total_days * daily_consumption / (batch_size * 128))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_cold_brew_time_per_batch_l2812_281272


namespace NUMINAMATH_CALUDE_prime_sum_and_product_l2812_281237

def smallest_one_digit_prime : ℕ := 2
def second_smallest_two_digit_prime : ℕ := 13
def smallest_three_digit_prime : ℕ := 101

theorem prime_sum_and_product :
  (smallest_one_digit_prime + second_smallest_two_digit_prime + smallest_three_digit_prime = 116) ∧
  (smallest_one_digit_prime * second_smallest_two_digit_prime * smallest_three_digit_prime = 2626) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_and_product_l2812_281237


namespace NUMINAMATH_CALUDE_gondor_laptop_repair_fee_l2812_281234

/-- The amount Gondor earns from repairing a phone -/
def phone_repair_fee : ℝ := 10

/-- The number of phones Gondor repaired on Monday -/
def monday_phones : ℕ := 3

/-- The number of phones Gondor repaired on Tuesday -/
def tuesday_phones : ℕ := 5

/-- The number of laptops Gondor repaired on Wednesday -/
def wednesday_laptops : ℕ := 2

/-- The number of laptops Gondor repaired on Thursday -/
def thursday_laptops : ℕ := 4

/-- The total amount Gondor earned -/
def total_earnings : ℝ := 200

/-- The amount Gondor earns from repairing a laptop -/
def laptop_repair_fee : ℝ := 20

theorem gondor_laptop_repair_fee :
  laptop_repair_fee = 20 ∧
  (monday_phones + tuesday_phones) * phone_repair_fee +
  (wednesday_laptops + thursday_laptops) * laptop_repair_fee = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_gondor_laptop_repair_fee_l2812_281234


namespace NUMINAMATH_CALUDE_largest_s_proof_l2812_281247

/-- The largest possible value of s for which there exist regular polygons P1 (r-gon) and P2 (s-gon)
    satisfying the given conditions -/
def largest_s : ℕ := 117

theorem largest_s_proof (r s : ℕ) : 
  r ≥ s → 
  s ≥ 3 → 
  (r - 2) * s * 60 = (s - 2) * r * 59 → 
  s ≤ largest_s := by
  sorry

#check largest_s_proof

end NUMINAMATH_CALUDE_largest_s_proof_l2812_281247


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2812_281250

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2812_281250


namespace NUMINAMATH_CALUDE_power_of_seven_mod_nine_l2812_281259

theorem power_of_seven_mod_nine : 7^15 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_nine_l2812_281259


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l2812_281221

/-- Given a class of students and information about their participation in two competitions,
    calculate the number of students who participated in both competitions. -/
theorem students_in_both_competitions
  (total : ℕ)
  (volleyball : ℕ)
  (track_field : ℕ)
  (none : ℕ)
  (h1 : total = 45)
  (h2 : volleyball = 12)
  (h3 : track_field = 20)
  (h4 : none = 19)
  : volleyball + track_field - (total - none) = 6 := by
  sorry

#check students_in_both_competitions

end NUMINAMATH_CALUDE_students_in_both_competitions_l2812_281221


namespace NUMINAMATH_CALUDE_ratio_equality_l2812_281281

theorem ratio_equality (a b : ℝ) (h1 : 5 * a = 3 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : b / a = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2812_281281


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2812_281251

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 10
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-2, 18), (8, 38)}

-- Theorem statement
theorem parabolas_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2812_281251


namespace NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l2812_281257

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_cosine_value_on_unit_circle_l2812_281257


namespace NUMINAMATH_CALUDE_train_distance_l2812_281288

/-- The distance traveled by a train in a given time, given its speed -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Convert hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ :=
  hours * 60

theorem train_distance (train_speed : ℚ) (travel_time : ℚ) :
  train_speed = 2 / 2 →
  travel_time = 3 →
  distance_traveled train_speed (hours_to_minutes travel_time) = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2812_281288


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2812_281238

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 22 ∧ 
  b = 34 ∧ 
  c = d → 
  c * d = 144 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2812_281238


namespace NUMINAMATH_CALUDE_f_max_at_neg_three_l2812_281290

/-- The quadratic function f(x) = -x^2 - 6x + 12 -/
def f (x : ℝ) : ℝ := -x^2 - 6*x + 12

/-- Theorem stating that f(x) attains its maximum value when x = -3 -/
theorem f_max_at_neg_three :
  ∃ (max : ℝ), f (-3) = max ∧ ∀ x, f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_three_l2812_281290


namespace NUMINAMATH_CALUDE_students_exceed_pets_l2812_281280

/-- Proves that in 6 classrooms, where each classroom has 22 students, 3 pet rabbits, 
    and 1 pet hamster, the number of students exceeds the number of pets by 108. -/
theorem students_exceed_pets : 
  let num_classrooms : ℕ := 6
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 3
  let hamsters_per_classroom : ℕ := 1
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)
  total_students - total_pets = 108 := by
  sorry

end NUMINAMATH_CALUDE_students_exceed_pets_l2812_281280


namespace NUMINAMATH_CALUDE_find_T_l2812_281262

theorem find_T : ∃ T : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * T = (1/2 : ℚ) * (1/6 : ℚ) * 72 ∧ T = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l2812_281262


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2812_281258

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2812_281258


namespace NUMINAMATH_CALUDE_g_of_3_l2812_281284

def g (x : ℝ) : ℝ := 5 * x^3 - 6 * x^2 - 3 * x + 5

theorem g_of_3 : g 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l2812_281284


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2812_281268

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 23^7 → 
  m % 10 = 9 → 
  n % 10 = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2812_281268


namespace NUMINAMATH_CALUDE_waterpark_total_cost_calculation_l2812_281226

def waterpark_total_cost (adult_price child_price teen_price : ℚ)
                         (num_adults num_children num_teens : ℕ)
                         (activity_discount coupon_discount : ℚ)
                         (soda_price : ℚ) (num_sodas : ℕ) : ℚ :=
  let base_cost := adult_price * num_adults + child_price * num_children + teen_price * num_teens
  let discounted_cost := base_cost * (1 - activity_discount) * (1 - coupon_discount)
  let soda_cost := soda_price * num_sodas
  discounted_cost + soda_cost

theorem waterpark_total_cost_calculation :
  waterpark_total_cost 30 15 20 4 2 4 (1/10) (1/20) 5 5 = 221.65 := by
  sorry

end NUMINAMATH_CALUDE_waterpark_total_cost_calculation_l2812_281226


namespace NUMINAMATH_CALUDE_three_zeros_implies_m_in_open_interval_l2812_281230

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating that if f has three zeros, then m is in the open interval (-4, 4) -/
theorem three_zeros_implies_m_in_open_interval (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) →
  m ∈ Set.Ioo (-4 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_m_in_open_interval_l2812_281230


namespace NUMINAMATH_CALUDE_mean_sales_is_five_point_five_l2812_281293

def monday_sales : ℕ := 8
def tuesday_sales : ℕ := 3
def wednesday_sales : ℕ := 10
def thursday_sales : ℕ := 4
def friday_sales : ℕ := 4
def saturday_sales : ℕ := 4

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales
def number_of_days : ℕ := 6

theorem mean_sales_is_five_point_five :
  (total_sales : ℚ) / (number_of_days : ℚ) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_sales_is_five_point_five_l2812_281293


namespace NUMINAMATH_CALUDE_all_divisible_by_nine_l2812_281207

/-- A five-digit number represented as a tuple of five natural numbers -/
def FiveDigitNumber := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- The sum of digits in a five-digit number -/
def digitSum (n : FiveDigitNumber) : ℕ :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2.1 + n.2.2.2.2

/-- Predicate for a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  1 ≤ n.1 ∧ n.1 ≤ 9 ∧ 0 ≤ n.2.1 ∧ n.2.1 ≤ 9 ∧
  0 ≤ n.2.2.1 ∧ n.2.2.1 ≤ 9 ∧ 0 ≤ n.2.2.2.1 ∧ n.2.2.2.1 ≤ 9 ∧
  0 ≤ n.2.2.2.2 ∧ n.2.2.2.2 ≤ 9

/-- The set of all valid five-digit numbers with digit sum 36 -/
def S : Set FiveDigitNumber :=
  {n | isValidFiveDigitNumber n ∧ digitSum n = 36}

/-- The numeric value of a five-digit number -/
def numericValue (n : FiveDigitNumber) : ℕ :=
  10000 * n.1 + 1000 * n.2.1 + 100 * n.2.2.1 + 10 * n.2.2.2.1 + n.2.2.2.2

theorem all_divisible_by_nine :
  ∀ n ∈ S, (numericValue n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_divisible_by_nine_l2812_281207


namespace NUMINAMATH_CALUDE_f_properties_l2812_281255

noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

theorem f_properties :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧
  (f (π / 2) = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2812_281255


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_60_l2812_281282

theorem smallest_divisible_by_18_and_60 : ∀ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 60 ∣ n → n ≥ 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_60_l2812_281282


namespace NUMINAMATH_CALUDE_twice_x_minus_three_greater_than_four_l2812_281298

theorem twice_x_minus_three_greater_than_four (x : ℝ) :
  (2 * x - 3 > 4) ↔ (∃ y, y = 2 * x - 3 ∧ y > 4) :=
sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_greater_than_four_l2812_281298


namespace NUMINAMATH_CALUDE_working_days_is_twenty_main_theorem_l2812_281210

/-- Represents the commute data for a period of working days -/
structure CommuteData where
  car_to_work : ℕ
  train_from_work : ℕ
  total_train_trips : ℕ

/-- Calculates the total number of working days based on commute data -/
def calculate_working_days (data : CommuteData) : ℕ :=
  data.car_to_work + data.total_train_trips

/-- Theorem stating that the number of working days is 20 given the specific commute data -/
theorem working_days_is_twenty (data : CommuteData) 
  (h1 : data.car_to_work = 12)
  (h2 : data.train_from_work = 11)
  (h3 : data.total_train_trips = 8)
  (h4 : data.car_to_work = data.train_from_work + 1) :
  calculate_working_days data = 20 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ (data : CommuteData), 
  data.car_to_work = 12 ∧ 
  data.train_from_work = 11 ∧ 
  data.total_train_trips = 8 ∧ 
  data.car_to_work = data.train_from_work + 1 ∧
  calculate_working_days data = 20 := by
  sorry

end NUMINAMATH_CALUDE_working_days_is_twenty_main_theorem_l2812_281210


namespace NUMINAMATH_CALUDE_seonmi_money_problem_l2812_281261

theorem seonmi_money_problem (initial_money : ℕ) : 
  (initial_money / 2 / 3 / 2 = 250) → initial_money = 1500 := by
sorry

end NUMINAMATH_CALUDE_seonmi_money_problem_l2812_281261


namespace NUMINAMATH_CALUDE_triangle_sin_c_l2812_281215

theorem triangle_sin_c (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = 1 →
  b = Real.sqrt 2 →
  A + C = 2 * B →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_c_l2812_281215


namespace NUMINAMATH_CALUDE_ellipse_properties_l2812_281264

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_major_axis : 2 * b = a
  h_rhombus_area : 4 * a * b = 8

/-- A line passing through a point on the ellipse -/
structure IntersectingLine (ε : Ellipse) where
  k : ℝ
  h_length : (4 * Real.sqrt 2) / 5 = 4 * Real.sqrt (1 + k^2) / (1 + 4 * k^2)

/-- A point on the perpendicular bisector of the chord -/
structure PerpendicularPoint (ε : Ellipse) (l : IntersectingLine ε) where
  y₀ : ℝ
  h_dot_product : 4 = (y₀^2 + ε.a^2) - (y₀^2 + (ε.a * (1 - k^2) / (1 + k^2))^2)

/-- The main theorem capturing the problem's assertions -/
theorem ellipse_properties (ε : Ellipse) (l : IntersectingLine ε) (p : PerpendicularPoint ε l) :
  ε.a = 2 ∧ ε.b = 1 ∧
  (l.k = 1 ∨ l.k = -1) ∧
  (p.y₀ = 2 * Real.sqrt 2 ∨ p.y₀ = -2 * Real.sqrt 2 ∨
   p.y₀ = 2 * Real.sqrt 14 / 5 ∨ p.y₀ = -2 * Real.sqrt 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2812_281264


namespace NUMINAMATH_CALUDE_expression_value_l2812_281243

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 7
  x^2 + y^2 + z^2 - 2*x*y = 74 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2812_281243


namespace NUMINAMATH_CALUDE_alternating_sum_fraction_equals_two_l2812_281233

theorem alternating_sum_fraction_equals_two :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_fraction_equals_two_l2812_281233


namespace NUMINAMATH_CALUDE_bench_press_theorem_l2812_281206

def bench_press_problem (initial_weight : ℝ) (injury_reduction : ℝ) (training_multiplier : ℝ) : Prop :=
  let after_injury := initial_weight * (1 - injury_reduction)
  let final_weight := after_injury * training_multiplier
  final_weight = 300

theorem bench_press_theorem :
  bench_press_problem 500 0.8 3 := by
  sorry

end NUMINAMATH_CALUDE_bench_press_theorem_l2812_281206


namespace NUMINAMATH_CALUDE_clock_angle_at_3_30_l2812_281260

/-- The angle of the hour hand at 3:30 -/
def hour_hand_angle : ℝ := 105

/-- The angle of the minute hand at 3:30 -/
def minute_hand_angle : ℝ := 180

/-- The total degrees in a circle -/
def total_degrees : ℝ := 360

/-- The larger angle between the hour and minute hands at 3:30 -/
def larger_angle : ℝ := total_degrees - (minute_hand_angle - hour_hand_angle)

theorem clock_angle_at_3_30 :
  larger_angle = 285 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_30_l2812_281260


namespace NUMINAMATH_CALUDE_S_min_value_l2812_281223

/-- The area function S(a) for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := a^2 / Real.sqrt (a^2 - 1)

/-- Theorem stating the minimum value of S(a) -/
theorem S_min_value (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ S (Real.sqrt 2) = min_val ∧ ∀ x > 1, S x ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_S_min_value_l2812_281223


namespace NUMINAMATH_CALUDE_chip_price_reduction_l2812_281222

/-- Represents the price reduction process for a chip -/
theorem chip_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 256) 
  (h2 : final_price = 196) 
  (x : ℝ) -- x represents the percentage of each price reduction
  (h3 : 0 ≤ x ∧ x < 1) -- ensure x is a valid percentage
  : initial_price * (1 - x)^2 = final_price ↔ 
    initial_price * (1 - x)^2 = 196 ∧ initial_price = 256 :=
by sorry

end NUMINAMATH_CALUDE_chip_price_reduction_l2812_281222


namespace NUMINAMATH_CALUDE_equation_solution_l2812_281216

theorem equation_solution : 
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 92 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2812_281216


namespace NUMINAMATH_CALUDE_remainder_problem_l2812_281202

theorem remainder_problem (G : ℕ) (h1 : G = 144) (h2 : 6215 % G = 23) : 7373 % G = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2812_281202


namespace NUMINAMATH_CALUDE_equation_solution_l2812_281231

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ, a * (x : ℤ) = 3) → (a = 1 ∨ a = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2812_281231


namespace NUMINAMATH_CALUDE_cubic_inequality_l2812_281256

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2812_281256


namespace NUMINAMATH_CALUDE_minimum_b_value_l2812_281266

theorem minimum_b_value (a b : ℕ) : 
  a = 23 →
  (a + b) % 10 = 5 →
  (a + b) % 7 = 4 →
  b ≥ 2 ∧ ∃ (b' : ℕ), b' ≥ 2 → b ≤ b' :=
by sorry

end NUMINAMATH_CALUDE_minimum_b_value_l2812_281266


namespace NUMINAMATH_CALUDE_katies_new_friends_games_l2812_281292

/-- The number of games Katie's new friends have -/
def new_friends_games (total_friends_games old_friends_games : ℕ) : ℕ :=
  total_friends_games - old_friends_games

/-- Theorem: Katie's new friends have 88 games -/
theorem katies_new_friends_games :
  new_friends_games 141 53 = 88 := by
  sorry

end NUMINAMATH_CALUDE_katies_new_friends_games_l2812_281292


namespace NUMINAMATH_CALUDE_dot_product_value_l2812_281296

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def f (a b : n) (x : ℝ) : ℝ := ‖a + x • b‖

theorem dot_product_value (a b : n) 
  (ha : ‖a‖ = Real.sqrt 2) 
  (hb : ‖b‖ = Real.sqrt 2)
  (hmin : ∀ x : ℝ, f a b x ≥ 1)
  (hf : ∃ x : ℝ, f a b x = 1) :
  inner a b = Real.sqrt 2 ∨ inner a b = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_dot_product_value_l2812_281296


namespace NUMINAMATH_CALUDE_circle_equation_polar_l2812_281248

/-- The equation of a circle in polar coordinates with center at (√2, π) passing through the pole -/
theorem circle_equation_polar (ρ θ : ℝ) : 
  (ρ = -2 * Real.sqrt 2 * Real.cos θ) ↔ 
  (∃ (x y : ℝ), 
    -- Convert polar to Cartesian coordinates
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
    -- Circle equation in Cartesian coordinates
    ((x + Real.sqrt 2)^2 + y^2 = 2) ∧
    -- Circle passes through the pole (origin in Cartesian)
    (∃ (θ₀ : ℝ), ρ * Real.cos θ₀ = 0 ∧ ρ * Real.sin θ₀ = 0)) := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_polar_l2812_281248


namespace NUMINAMATH_CALUDE_total_toads_l2812_281229

theorem total_toads (in_pond : ℕ) (outside_pond : ℕ) 
  (h1 : in_pond = 12) (h2 : outside_pond = 6) : 
  in_pond + outside_pond = 18 := by
sorry

end NUMINAMATH_CALUDE_total_toads_l2812_281229


namespace NUMINAMATH_CALUDE_exist_four_distinct_naturals_perfect_squares_l2812_281218

theorem exist_four_distinct_naturals_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by
  sorry

end NUMINAMATH_CALUDE_exist_four_distinct_naturals_perfect_squares_l2812_281218


namespace NUMINAMATH_CALUDE_always_two_distinct_roots_find_p_values_l2812_281276

-- Define the quadratic equation
def quadratic_equation (x p : ℝ) : ℝ := (x - 3) * (x - 2) - p^2

-- Part 1: Prove that the equation always has two distinct real roots
theorem always_two_distinct_roots (p : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ p = 0 ∧ quadratic_equation x₂ p = 0 := by
  sorry

-- Part 2: Find the values of p given the condition x₁ = 4x₂
theorem find_p_values :
  ∃ p : ℝ, ∃ x₁ x₂ : ℝ, 
    quadratic_equation x₁ p = 0 ∧ 
    quadratic_equation x₂ p = 0 ∧ 
    x₁ = 4 * x₂ ∧ 
    (p = Real.sqrt 2 ∨ p = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_always_two_distinct_roots_find_p_values_l2812_281276


namespace NUMINAMATH_CALUDE_josh_pencils_l2812_281279

theorem josh_pencils (pencils_given : ℕ) (pencils_left : ℕ) 
  (h1 : pencils_given = 31) 
  (h2 : pencils_left = 111) : 
  pencils_given + pencils_left = 142 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_l2812_281279


namespace NUMINAMATH_CALUDE_train_crossing_time_l2812_281213

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 120 →
  train_speed = 67 * (1000 / 3600) →
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
sorry


end NUMINAMATH_CALUDE_train_crossing_time_l2812_281213


namespace NUMINAMATH_CALUDE_marathon_average_time_l2812_281240

/-- Given Casey's time to complete a marathon and Zendaya's relative time compared to Casey,
    calculate the average time for both to complete the race. -/
theorem marathon_average_time (casey_time : ℝ) (zendaya_relative_time : ℝ) :
  casey_time = 6 →
  zendaya_relative_time = 1/3 →
  let zendaya_time := casey_time + zendaya_relative_time * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
  sorry


end NUMINAMATH_CALUDE_marathon_average_time_l2812_281240


namespace NUMINAMATH_CALUDE_parabola_focus_l2812_281295

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The parabola opens upwards -/
def opens_upwards (p : (ℝ → ℝ → Prop)) : Prop := sorry

/-- The focus lies on the y-axis -/
def focus_on_y_axis (f : Focus) : Prop := f.x = 0

/-- Theorem stating that the focus of the given parabola is at (0, 1) -/
theorem parabola_focus :
  ∃ (f : Focus),
    (∀ x y, parabola x y → opens_upwards parabola) ∧
    focus_on_y_axis f ∧
    f.x = 0 ∧ f.y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2812_281295


namespace NUMINAMATH_CALUDE_quadrilateral_triangle_product_l2812_281224

/-- Represents a convex quadrilateral with its four triangles formed by diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by diagonals -/
  triangle_areas : Fin 4 → ℕ

/-- Theorem stating that the product of the four triangle areas in a convex quadrilateral
    cannot be congruent to 2014 modulo 10000 -/
theorem quadrilateral_triangle_product (q : ConvexQuadrilateral) :
  (q.triangle_areas 0 * q.triangle_areas 1 * q.triangle_areas 2 * q.triangle_areas 3) % 10000 ≠ 2014 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_triangle_product_l2812_281224


namespace NUMINAMATH_CALUDE_farm_animals_l2812_281253

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8) 
  (h2 : total_legs = 24) : 
  ∃ (ducks dogs : ℕ), 
    ducks + dogs = total_animals ∧ 
    2 * ducks + 4 * dogs = total_legs ∧ 
    ducks = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2812_281253


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2812_281212

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 17.5)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2812_281212


namespace NUMINAMATH_CALUDE_worker_efficiency_l2812_281244

theorem worker_efficiency (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  p = 1 / 22 → p + q = 1 / 12 → p / q = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l2812_281244


namespace NUMINAMATH_CALUDE_scientists_from_usa_l2812_281235

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h1 : total = 70)
  (h2 : europe = total / 2)
  (h3 : canada = total / 5)
  (h4 : usa = total - (europe + canada)) :
  usa = 21 := by
  sorry

end NUMINAMATH_CALUDE_scientists_from_usa_l2812_281235


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2812_281209

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2812_281209


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2812_281200

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first 4 terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2812_281200


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2812_281297

-- Problem 1
theorem problem_one : (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_two : |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2812_281297


namespace NUMINAMATH_CALUDE_whisky_alcohol_percentage_l2812_281294

/-- The initial percentage of alcohol in a jar of whisky -/
def initial_alcohol_percentage : ℝ := 40

/-- The percentage of alcohol in the replacement whisky -/
def replacement_alcohol_percentage : ℝ := 19

/-- The percentage of alcohol after replacement -/
def final_alcohol_percentage : ℝ := 24

/-- The quantity of whisky replaced -/
def replaced_quantity : ℝ := 0.7619047619047619

/-- The total volume of whisky in the jar -/
def total_volume : ℝ := 1

theorem whisky_alcohol_percentage :
  initial_alcohol_percentage / 100 * (total_volume - replaced_quantity) +
  replacement_alcohol_percentage / 100 * replaced_quantity =
  final_alcohol_percentage / 100 * total_volume := by
  sorry

end NUMINAMATH_CALUDE_whisky_alcohol_percentage_l2812_281294


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l2812_281203

theorem restaurant_tip_percentage : 
  let james_meal : ℚ := 16
  let friend_meal : ℚ := 14
  let total_bill : ℚ := james_meal + friend_meal
  let james_paid : ℚ := 21
  let friend_paid : ℚ := total_bill / 2
  let tip : ℚ := james_paid - friend_paid
  tip / total_bill = 1/5 := by sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l2812_281203


namespace NUMINAMATH_CALUDE_class_size_l2812_281252

theorem class_size (n : ℕ) 
  (h1 : n < 50) 
  (h2 : n % 8 = 5) 
  (h3 : n % 6 = 4) : 
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_l2812_281252


namespace NUMINAMATH_CALUDE_polynomial_bound_l2812_281286

theorem polynomial_bound (n : ℕ) (p : ℝ → ℝ) :
  (∀ x, ∃ (c : ℝ) (k : ℕ), k ≤ 2*n ∧ p x = c * x^k) →
  (∀ k : ℤ, -n ≤ k ∧ k ≤ n → |p k| ≤ 1) →
  ∀ x : ℝ, -n ≤ x ∧ x ≤ n → |p x| ≤ 2^(2*n) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l2812_281286


namespace NUMINAMATH_CALUDE_ratio_of_13th_terms_l2812_281201

/-- Two arithmetic sequences with sums U_n and V_n for the first n terms -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (a b c d : ℚ), ∀ n : ℕ,
    U n = n * (2 * a + (n - 1) * b) / 2 ∧
    V n = n * (2 * c + (n - 1) * d) / 2

/-- The ratio condition for U_n and V_n -/
def ratio_condition (U V : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, U n * (3 * n + 17) = V n * (5 * n + 3)

/-- The 13th term of an arithmetic sequence -/
def term_13 (seq : ℕ → ℚ) : ℚ :=
  seq 13 - seq 12

/-- Main theorem -/
theorem ratio_of_13th_terms
  (U V : ℕ → ℚ)
  (h1 : arithmetic_sequences U V)
  (h2 : ratio_condition U V) :
  term_13 U / term_13 V = 52 / 89 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_13th_terms_l2812_281201


namespace NUMINAMATH_CALUDE_sum_lent_is_2000_l2812_281246

/-- Prove that the sum lent is 2000, given the conditions of the loan --/
theorem sum_lent_is_2000 
  (interest_rate : ℝ) 
  (loan_duration : ℝ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.03) 
  (h2 : loan_duration = 3) 
  (h3 : ∀ sum_lent : ℝ, sum_lent * interest_rate * loan_duration = sum_lent - interest_difference) 
  (h4 : interest_difference = 1820) : 
  ∃ sum_lent : ℝ, sum_lent = 2000 := by
  sorry


end NUMINAMATH_CALUDE_sum_lent_is_2000_l2812_281246


namespace NUMINAMATH_CALUDE_total_students_l2812_281274

theorem total_students (group_a group_b : ℕ) : 
  (group_a : ℚ) / group_b = 3 / 2 →
  (group_a : ℚ) * (1 / 10) - (group_b : ℚ) * (1 / 5) = 190 →
  group_b = 650 →
  group_a + group_b = 1625 := by
sorry


end NUMINAMATH_CALUDE_total_students_l2812_281274


namespace NUMINAMATH_CALUDE_arithmetic_statement_not_basic_unique_non_basic_statement_l2812_281249

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"input statement", "output statement", "assignment statement", "conditional statement", "loop statement"}

/-- The list of options given in the problem -/
def Options : List String :=
  ["assignment statement", "arithmetic statement", "conditional statement", "loop statement"]

/-- Theorem: The arithmetic statement is not a member of the set of basic algorithmic statements -/
theorem arithmetic_statement_not_basic : "arithmetic statement" ∉ BasicAlgorithmicStatements := by
  sorry

/-- Theorem: The arithmetic statement is the only option not in the set of basic algorithmic statements -/
theorem unique_non_basic_statement :
  ∀ s ∈ Options, s ∉ BasicAlgorithmicStatements → s = "arithmetic statement" := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_statement_not_basic_unique_non_basic_statement_l2812_281249


namespace NUMINAMATH_CALUDE_cat_mouse_position_after_323_moves_l2812_281273

-- Define the positions for the cat
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

-- Define the positions for the mouse
inductive MousePosition
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle
  | TopLeft

-- Function to calculate cat's position after n moves
def catPositionAfterMoves (n : ℕ) : CatPosition :=
  match n % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

-- Function to calculate mouse's position after n moves
def mousePositionAfterMoves (n : ℕ) : MousePosition :=
  match n % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem cat_mouse_position_after_323_moves :
  (catPositionAfterMoves 323 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 323 = MousePosition.RightMiddle) :=
by sorry

end NUMINAMATH_CALUDE_cat_mouse_position_after_323_moves_l2812_281273


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2812_281205

theorem polynomial_factorization (u : ℝ) : 
  u^4 - 81*u^2 + 144 = (u^2 - 72)*(u - 3)*(u + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2812_281205


namespace NUMINAMATH_CALUDE_unique_number_l2812_281265

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def middle_digits_39 (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + 390 + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

theorem unique_number :
  ∃! n : ℕ, is_four_digit n ∧ middle_digits_39 n ∧ n % 45 = 0 ∧ n ≤ 5000 ∧ n = 1395 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l2812_281265


namespace NUMINAMATH_CALUDE_race_distance_proof_l2812_281227

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (arianna_position : ℕ) : ℕ :=
  race_length - arianna_position

theorem race_distance_proof :
  let race_length : ℕ := 1000  -- 1 km in meters
  let arianna_position : ℕ := 184
  distance_between_runners race_length arianna_position = 816 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_proof_l2812_281227


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l2812_281228

-- Define the computer's cost
variable (cost : ℝ)

-- Define the two selling prices
def selling_price_1 : ℝ := 2240
def selling_price_2 : ℝ := 2400

-- Define the profit percentages
def profit_percentage_1 : ℝ := 0.4  -- 40%
def profit_percentage_2 : ℝ := 0.5  -- 50%

-- Theorem statement
theorem computer_profit_percentage :
  (selling_price_2 - cost = profit_percentage_2 * cost) →
  (selling_price_1 - cost = profit_percentage_1 * cost) :=
by sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l2812_281228


namespace NUMINAMATH_CALUDE_defendant_statement_implies_innocence_l2812_281285

-- Define the types of people on the island
inductive Person
| Knight
| Liar

-- Define the crime and accusation
def Crime : Type := Unit
def Accusation : Type := Unit

-- Define the statement made by the defendant
def DefendantStatement (criminal : Person) : Prop :=
  criminal = Person.Liar

-- Define the concept of telling the truth
def TellsTruth (p : Person) (statement : Prop) : Prop :=
  match p with
  | Person.Knight => statement
  | Person.Liar => ¬statement

-- Theorem: The defendant's statement implies innocence regardless of their type
theorem defendant_statement_implies_innocence 
  (defendant : Person) 
  (crime : Crime) 
  (accusation : Accusation) :
  TellsTruth defendant (DefendantStatement (Person.Liar)) → 
  defendant ≠ Person.Liar :=
sorry

end NUMINAMATH_CALUDE_defendant_statement_implies_innocence_l2812_281285


namespace NUMINAMATH_CALUDE_alice_painted_six_cuboids_l2812_281289

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end NUMINAMATH_CALUDE_alice_painted_six_cuboids_l2812_281289


namespace NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2812_281204

-- Define the ellipse C
structure Ellipse :=
  (center : ℝ × ℝ)
  (a b : ℝ)
  (eccentricity : ℝ)

-- Define the parabola
def Parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}

-- Define a point on the ellipse
structure PointOnEllipse (C : Ellipse) :=
  (point : ℝ × ℝ)
  (on_ellipse : (point.1 - C.center.1)^2 / C.a^2 + (point.2 - C.center.2)^2 / C.b^2 = 1)

-- Define a tangent line to the ellipse
structure TangentLine (C : Ellipse) :=
  (point : PointOnEllipse C)
  (slope : ℝ)

-- Theorem statement
theorem ellipse_tangent_intersection 
  (C : Ellipse)
  (h1 : C.center = (0, 0))
  (h2 : C.eccentricity = Real.sqrt 2 / 2)
  (h3 : ∃ (f : ℝ × ℝ), f ∈ Parabola ∧ f ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 / C.a^2 + (p.2 - C.center.2)^2 / C.b^2 = C.eccentricity^2})
  (A : PointOnEllipse C)
  (tAB tAC : TangentLine C)
  (h4 : tAB.point = A ∧ tAC.point = A)
  (h5 : tAB.slope * tAC.slope = 1/4) :
  ∃ (P : ℝ × ℝ), P = (0, 3) ∧ 
    ∀ (B C : ℝ × ℝ), 
      (B.2 - A.point.2 = tAB.slope * (B.1 - A.point.1)) → 
      (C.2 - A.point.2 = tAC.slope * (C.1 - A.point.1)) → 
      (P.2 - B.2) / (P.1 - B.1) = (C.2 - B.2) / (C.1 - B.1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_intersection_l2812_281204


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l2812_281299

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -1; 3, 7]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 5, -2]

theorem matrix_sum_theorem : A + B = !![(-2), 7; 8, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l2812_281299


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2812_281211

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49 : ℝ) 49) :
  ∃ (M : ℝ), M = 14 ∧ Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ M ∧
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-49 : ℝ) 49 ∧ Real.sqrt (49 + x₀) + Real.sqrt (49 - x₀) = M :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2812_281211


namespace NUMINAMATH_CALUDE_ticket_difference_l2812_281269

/-- Represents the number of tickets sold for each category -/
structure TicketSales where
  vip : ℕ
  premium : ℕ
  general : ℕ

/-- Checks if the given ticket sales satisfy the problem conditions -/
def satisfiesConditions (sales : TicketSales) : Prop :=
  sales.vip + sales.premium + sales.general = 420 ∧
  50 * sales.vip + 30 * sales.premium + 10 * sales.general = 12000

/-- Theorem stating the difference between general admission and VIP tickets -/
theorem ticket_difference (sales : TicketSales) 
  (h : satisfiesConditions sales) : 
  sales.general - sales.vip = 30 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l2812_281269


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2812_281291

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_a1 : a 1 = 1)
    (h_a2a4 : a 2 * a 4 = 16) :
    a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2812_281291


namespace NUMINAMATH_CALUDE_coronavirus_recoveries_l2812_281220

/-- Calculates the number of recoveries on the third day of a coronavirus outbreak --/
theorem coronavirus_recoveries 
  (initial_cases : ℕ) 
  (second_day_new_cases : ℕ) 
  (second_day_recoveries : ℕ) 
  (third_day_new_cases : ℕ) 
  (final_total_cases : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : second_day_new_cases = 500)
  (h3 : second_day_recoveries = 50)
  (h4 : third_day_new_cases = 1500)
  (h5 : final_total_cases = 3750) :
  initial_cases + second_day_new_cases - second_day_recoveries + third_day_new_cases - final_total_cases = 200 :=
by
  sorry

#check coronavirus_recoveries

end NUMINAMATH_CALUDE_coronavirus_recoveries_l2812_281220


namespace NUMINAMATH_CALUDE_hamburger_price_correct_l2812_281242

/-- The price of a hamburger that satisfies the given conditions -/
def hamburger_price : ℝ := 3.125

/-- The number of hamburgers already sold -/
def hamburgers_sold : ℕ := 12

/-- The number of additional hamburgers needed to be sold -/
def additional_hamburgers : ℕ := 4

/-- The total revenue target -/
def total_revenue : ℝ := 50

/-- Theorem stating that the hamburger price satisfies the given conditions -/
theorem hamburger_price_correct : 
  hamburger_price * (hamburgers_sold + additional_hamburgers) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_hamburger_price_correct_l2812_281242


namespace NUMINAMATH_CALUDE_greatest_b_value_l2812_281275

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2812_281275


namespace NUMINAMATH_CALUDE_square_difference_501_499_l2812_281214

theorem square_difference_501_499 : 501^2 - 499^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_501_499_l2812_281214


namespace NUMINAMATH_CALUDE_davids_english_marks_l2812_281283

/-- Represents the marks of a student in various subjects -/
structure Marks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem davids_english_marks (m : Marks) (h1 : m.mathematics = 60) 
    (h2 : m.physics = 78) (h3 : m.chemistry = 60) (h4 : m.biology = 65) 
    (h5 : average [m.mathematics, m.physics, m.chemistry, m.biology, m.english] = 66.6) :
    m.english = 70 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l2812_281283
