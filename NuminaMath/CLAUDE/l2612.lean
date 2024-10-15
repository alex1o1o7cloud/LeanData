import Mathlib

namespace NUMINAMATH_CALUDE_son_age_l2612_261258

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2612_261258


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2612_261272

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) :
  V = (1/3) * π * r^2 * h →
  V = 27 * π →
  h = 9 →
  2 * π * r = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2612_261272


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sqrt_9_l2612_261245

theorem cube_root_27_times_fourth_root_81_times_sqrt_9 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^2 = 9 → a * b * c = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sqrt_9_l2612_261245


namespace NUMINAMATH_CALUDE_total_raisins_l2612_261206

theorem total_raisins (yellow_raisins black_raisins : ℝ) 
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l2612_261206


namespace NUMINAMATH_CALUDE_coffee_mix_proof_l2612_261293

/-- The price of Colombian coffee beans in dollars per pound -/
def colombian_price : ℝ := 5.50

/-- The price of Peruvian coffee beans in dollars per pound -/
def peruvian_price : ℝ := 4.25

/-- The total weight of the mix in pounds -/
def total_weight : ℝ := 40

/-- The desired price of the mix in dollars per pound -/
def mix_price : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mix -/
def colombian_amount : ℝ := 11.2

theorem coffee_mix_proof :
  colombian_amount * colombian_price + (total_weight - colombian_amount) * peruvian_price = 
  mix_price * total_weight :=
sorry

end NUMINAMATH_CALUDE_coffee_mix_proof_l2612_261293


namespace NUMINAMATH_CALUDE_fraction_puzzle_l2612_261255

theorem fraction_puzzle : ∃ (x y : ℕ), 
  x + 35 = y ∧ 
  x ≠ 0 ∧ 
  y ≠ 0 ∧
  (x : ℚ) / y + (x.gcd y : ℚ) * x / ((y.gcd x) * y) = 16 / 13 ∧
  x = 56 ∧
  y = 91 := by
sorry

end NUMINAMATH_CALUDE_fraction_puzzle_l2612_261255


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2612_261251

/-- Proves that for a rectangle with breadth 10 meters and length 10 meters greater than its breadth,
    the ratio of its area to its breadth is 20:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (breadth length area : ℝ),
    breadth = 10 →
    length = breadth + 10 →
    area = length * breadth →
    area / breadth = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2612_261251


namespace NUMINAMATH_CALUDE_sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l2612_261268

/-- A function f is balanceable if there exist real numbers m and k (m ≠ 0) such that
    m * f x = f (x + k) + f (x - k) for all x in the domain of f. -/
def Balanceable (f : ℝ → ℝ) : Prop :=
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

/-- A balancing pair for a function f is a pair (m, k) that satisfies the balanceable condition. -/
def BalancingPair (f : ℝ → ℝ) (m k : ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

theorem sin_balanceable :
  ∃ n : ℤ, BalancingPair Real.sin 1 (2 * π * n + π / 3) ∨ BalancingPair Real.sin 1 (2 * π * n - π / 3) :=
sorry

theorem same_balancing_pair_for_square_and_exp :
  ∀ a : ℝ, a ≠ 0 →
  (BalancingPair (fun x ↦ x^2) 2 0 ∧ BalancingPair (fun x ↦ a + 2^x) 2 0) :=
sorry

theorem cos_squared_balancing_pair_range :
  ∃ m₁ m₂ : ℝ,
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₁ (π / 2) ∧
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₂ (π / 4) ∧
  ∀ x, 0 ≤ x ∧ x ≤ π / 4 → 1 ≤ m₁^2 + m₂^2 ∧ m₁^2 + m₂^2 ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l2612_261268


namespace NUMINAMATH_CALUDE_special_shape_is_regular_tetrahedron_l2612_261274

/-- A 3D shape with the property that the angle between diagonals of adjacent sides is 60 degrees -/
structure SpecialShape :=
  (is_3d : Bool)
  (diagonal_angle : ℝ)
  (angle_property : diagonal_angle = 60)

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron :=
  (is_3d : Bool)
  (num_faces : Nat)
  (face_type : String)
  (num_faces_property : num_faces = 4)
  (face_type_property : face_type = "equilateral triangle")

/-- Theorem stating that a SpecialShape is equivalent to a RegularTetrahedron -/
theorem special_shape_is_regular_tetrahedron (s : SpecialShape) : 
  ∃ (t : RegularTetrahedron), true :=
sorry

end NUMINAMATH_CALUDE_special_shape_is_regular_tetrahedron_l2612_261274


namespace NUMINAMATH_CALUDE_slower_speed_fraction_l2612_261247

/-- Given that a person arrives at a bus stop 9 minutes later than normal when walking
    at a certain fraction of their usual speed, and it takes 36 minutes to walk to the
    bus stop at their usual speed, prove that the fraction of the usual speed they
    were walking at is 4/5. -/
theorem slower_speed_fraction (usual_time : ℕ) (delay : ℕ) (usual_time_eq : usual_time = 36) (delay_eq : delay = 9) :
  (usual_time : ℚ) / (usual_time + delay : ℚ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_fraction_l2612_261247


namespace NUMINAMATH_CALUDE_crescent_lake_loop_length_l2612_261259

/-- Represents the distance walked on each day of the trip -/
structure DailyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (d : DailyDistances) : Prop :=
  d.day1 + d.day2 + d.day3 = 32 ∧
  (d.day2 + d.day3) / 2 = 12 ∧
  d.day3 + d.day4 + d.day5 = 45 ∧
  d.day1 + d.day4 = 30

/-- The theorem stating that if the conditions are satisfied, the total distance is 69 miles -/
theorem crescent_lake_loop_length 
  (d : DailyDistances) 
  (h : satisfies_conditions d) : 
  d.day1 + d.day2 + d.day3 + d.day4 + d.day5 = 69 := by
  sorry

end NUMINAMATH_CALUDE_crescent_lake_loop_length_l2612_261259


namespace NUMINAMATH_CALUDE_sum_eight_fib_not_fib_l2612_261253

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the sum of eight consecutive Fibonacci numbers
def sum_eight_fib (k : ℕ) : ℕ :=
  (fib (k + 1)) + (fib (k + 2)) + (fib (k + 3)) + (fib (k + 4)) +
  (fib (k + 5)) + (fib (k + 6)) + (fib (k + 7)) + (fib (k + 8))

-- Theorem statement
theorem sum_eight_fib_not_fib (k : ℕ) :
  (sum_eight_fib k > fib (k + 9)) ∧ (sum_eight_fib k < fib (k + 10)) :=
by sorry

end NUMINAMATH_CALUDE_sum_eight_fib_not_fib_l2612_261253


namespace NUMINAMATH_CALUDE_least_number_divisible_l2612_261203

def numbers : List ℕ := [52, 84, 114, 133, 221, 379]

def result : ℕ := 1097897218492

theorem least_number_divisible (n : ℕ) : n = result ↔ 
  (∀ m ∈ numbers, (n + 20) % m = 0) ∧ 
  (∀ k < n, ∃ m ∈ numbers, (k + 20) % m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_l2612_261203


namespace NUMINAMATH_CALUDE_estimation_theorem_l2612_261243

-- Define a function to estimate multiplication
def estimate_mult (a b : ℕ) : ℕ :=
  let a' := (a + 5) / 10 * 10  -- Round to nearest ten
  a' * b

-- Define a function to estimate division
def estimate_div (a b : ℕ) : ℕ :=
  let a' := (a + 50) / 100 * 100  -- Round to nearest hundred
  a' / b

-- State the theorem
theorem estimation_theorem :
  estimate_mult 47 20 = 1000 ∧ estimate_div 744 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_estimation_theorem_l2612_261243


namespace NUMINAMATH_CALUDE_unit_digit_14_power_100_l2612_261290

theorem unit_digit_14_power_100 : (14^100) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_14_power_100_l2612_261290


namespace NUMINAMATH_CALUDE_neil_cookies_fraction_l2612_261242

theorem neil_cookies_fraction (total : ℕ) (remaining : ℕ) (h1 : total = 20) (h2 : remaining = 12) :
  (total - remaining : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_neil_cookies_fraction_l2612_261242


namespace NUMINAMATH_CALUDE_proposition_variants_l2612_261221

theorem proposition_variants (a b : ℝ) :
  (((a - 2 > b - 2) → (a > b)) ∧
   ((a ≤ b) → (a - 2 ≤ b - 2)) ∧
   ((a - 2 ≤ b - 2) → (a ≤ b)) ∧
   ¬((a > b) → (a - 2 ≤ b - 2))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_variants_l2612_261221


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2612_261260

/-- The value of j that makes the line 4x + 7y + j = 0 tangent to the parabola y^2 = 32x -/
def tangent_j : ℝ := 98

/-- The line equation -/
def line (x y j : ℝ) : Prop := 4 * x + 7 * y + j = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- Theorem stating that tangent_j is the unique value making the line tangent to the parabola -/
theorem tangent_line_to_parabola :
  ∃! j : ℝ, ∀ x y : ℝ, line x y j ∧ parabola x y → 
    (∃! p : ℝ × ℝ, line p.1 p.2 j ∧ parabola p.1 p.2) ∧ j = tangent_j :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2612_261260


namespace NUMINAMATH_CALUDE_a_5_equals_one_l2612_261212

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_a_5_equals_one_l2612_261212


namespace NUMINAMATH_CALUDE_car_speed_second_hour_car_speed_second_hour_value_l2612_261287

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 100)
  (h2 : average_speed = 90) : 
  (speed_first_hour + (2 * average_speed - speed_first_hour)) / 2 = average_speed := by
  sorry

/-- The speed of the car in the second hour is 80 km/h. -/
theorem car_speed_second_hour_value : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 80 ∧ 
    (100 + speed_second_hour) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_car_speed_second_hour_value_l2612_261287


namespace NUMINAMATH_CALUDE_equation_solver_l2612_261282

theorem equation_solver (a b x y : ℝ) (h1 : x^2 / y + y^2 / x = a) (h2 : x / y + y / x = b) :
  (x = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) ∨
  (x = (a * (b + 2 - Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2)) ∧
   y = (a * (b + 2 + Real.sqrt (b^2 - 4))) / (2 * (b - 1) * (b + 2))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solver_l2612_261282


namespace NUMINAMATH_CALUDE_find_x_value_l2612_261211

theorem find_x_value (x y : ℝ) (h1 : (12 : ℝ)^3 * 6^2 / x = y) (h2 : y = 144) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l2612_261211


namespace NUMINAMATH_CALUDE_dale_toast_count_l2612_261266

/-- The cost of breakfast for Dale and Andrew -/
def breakfast_cost (toast_price egg_price : ℕ) (dale_toast : ℕ) : Prop :=
  toast_price * dale_toast + 2 * egg_price + toast_price + 2 * egg_price = 15

/-- Theorem stating that Dale had 2 slices of toast -/
theorem dale_toast_count : breakfast_cost 1 3 2 := by sorry

end NUMINAMATH_CALUDE_dale_toast_count_l2612_261266


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equal_area_l2612_261238

theorem right_triangle_perimeter_equal_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive integers
  a^2 + b^2 = c^2 →        -- Right-angled triangle (Pythagorean theorem)
  a + b + c = (a * b) / 2  -- Perimeter equals area
  → (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
    (a = 8 ∧ b = 6 ∧ c = 10) ∨ (a = 12 ∧ b = 5 ∧ c = 13) :=
by sorry

#check right_triangle_perimeter_equal_area

end NUMINAMATH_CALUDE_right_triangle_perimeter_equal_area_l2612_261238


namespace NUMINAMATH_CALUDE_distance_between_centers_is_sqrt_5_l2612_261271

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle with sides 6, 8, and 10
def rightTriangle : Triangle := { a := 6, b := 8, c := 10 }

-- Define the distance between centers of inscribed and circumscribed circles
def distanceBetweenCenters (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem distance_between_centers_is_sqrt_5 :
  distanceBetweenCenters rightTriangle = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_is_sqrt_5_l2612_261271


namespace NUMINAMATH_CALUDE_clock_notes_in_week_total_notes_in_week_l2612_261226

/-- Represents the ringing pattern for a single hour -/
structure HourPattern where
  quarter_past : Nat
  half_past : Nat
  quarter_to : Nat
  on_hour : Nat → Nat

/-- Represents the ringing pattern for a 12-hour period (day or night) -/
structure PeriodPattern where
  pattern : HourPattern
  on_hour_even : Nat → Nat
  on_hour_odd : Nat → Nat

def day_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 2
      half_past := 4
      quarter_to := 6
      on_hour := λ h => 8
    }
    on_hour_even := λ h => h
    on_hour_odd := λ h => h / 2
  }

def night_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 3
      half_past := 5
      quarter_to := 7
      on_hour := λ h => 9
    }
    on_hour_even := λ h => h / 2
    on_hour_odd := λ h => h
  }

def count_notes_for_period (pattern : PeriodPattern) : Nat :=
  12 * (pattern.pattern.quarter_past + pattern.pattern.half_past + pattern.pattern.quarter_to) +
  (pattern.pattern.on_hour 6 + pattern.on_hour_even 6 +
   pattern.pattern.on_hour 8 + pattern.on_hour_even 8 +
   pattern.pattern.on_hour 10 + pattern.on_hour_even 10 +
   pattern.pattern.on_hour 12 + pattern.on_hour_even 12 +
   pattern.pattern.on_hour 2 + pattern.on_hour_even 2 +
   pattern.pattern.on_hour 4 + pattern.on_hour_even 4 +
   pattern.pattern.on_hour 7 + pattern.on_hour_odd 7 +
   pattern.pattern.on_hour 9 + pattern.on_hour_odd 9 +
   pattern.pattern.on_hour 11 + pattern.on_hour_odd 11 +
   pattern.pattern.on_hour 1 + pattern.on_hour_odd 1 +
   pattern.pattern.on_hour 3 + pattern.on_hour_odd 3 +
   pattern.pattern.on_hour 5 + pattern.on_hour_odd 5)

theorem clock_notes_in_week :
  count_notes_for_period day_pattern + count_notes_for_period night_pattern = 471 ∧
  471 * 7 = 3297 := by sorry

theorem total_notes_in_week : (count_notes_for_period day_pattern + count_notes_for_period night_pattern) * 7 = 3297 := by sorry

end NUMINAMATH_CALUDE_clock_notes_in_week_total_notes_in_week_l2612_261226


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_expansion_main_proof_l2612_261261

theorem fraction_to_decimal : (7 : ℚ) / 200 = (35 : ℚ) / 1000 := by sorry

theorem decimal_expansion : (35 : ℚ) / 1000 = 0.035 := by sorry

theorem main_proof : (7 : ℚ) / 200 = 0.035 := by
  rw [fraction_to_decimal]
  exact decimal_expansion

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_expansion_main_proof_l2612_261261


namespace NUMINAMATH_CALUDE_cost_per_pound_mixed_feed_l2612_261297

/-- Calculates the cost per pound of mixed dog feed --/
theorem cost_per_pound_mixed_feed 
  (total_weight : ℝ) 
  (cheap_price : ℝ) 
  (expensive_price : ℝ) 
  (cheap_amount : ℝ) 
  (h1 : total_weight = 35) 
  (h2 : cheap_price = 0.18) 
  (h3 : expensive_price = 0.53) 
  (h4 : cheap_amount = 17) :
  (cheap_amount * cheap_price + (total_weight - cheap_amount) * expensive_price) / total_weight = 0.36 := by
sorry


end NUMINAMATH_CALUDE_cost_per_pound_mixed_feed_l2612_261297


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l2612_261295

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  sum_factorials 2010 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l2612_261295


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2612_261285

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l2612_261285


namespace NUMINAMATH_CALUDE_no_solution_l2612_261276

/-- The function f(t) = t^3 + t -/
def f (t : ℚ) : ℚ := t^3 + t

/-- Iterative application of f, n times -/
def f_iter (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

/-- There do not exist rational numbers x and y and positive integers m and n
    such that xy = 3 and f^m(x) = f^n(y) -/
theorem no_solution :
  ¬ ∃ (x y : ℚ) (m n : ℕ+), x * y = 3 ∧ f_iter m x = f_iter n y := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2612_261276


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2612_261224

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2612_261224


namespace NUMINAMATH_CALUDE_prob_different_colors_is_148_225_l2612_261214

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def total_chips : ℕ := blue_chips + red_chips + yellow_chips

def prob_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem prob_different_colors_is_148_225 :
  prob_different_colors = 148 / 225 :=
sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_148_225_l2612_261214


namespace NUMINAMATH_CALUDE_AMC10_paths_count_l2612_261296

/-- Represents the number of paths to spell "AMC10" given specific adjacency conditions -/
def number_of_AMC10_paths (
  adjacent_Ms : Nat
  ) (adjacent_Cs : Nat)
  (adjacent_1s : Nat)
  (adjacent_0s : Nat) : Nat :=
  adjacent_Ms * adjacent_Cs * adjacent_1s * adjacent_0s

/-- Theorem stating that the number of paths to spell "AMC10" is 48 -/
theorem AMC10_paths_count :
  number_of_AMC10_paths 4 3 2 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_AMC10_paths_count_l2612_261296


namespace NUMINAMATH_CALUDE_total_time_equals_sum_of_activities_l2612_261230

/-- The total time Joan initially had for her music practice -/
def total_time : ℕ := 120

/-- Time Joan spent on the piano -/
def piano_time : ℕ := 30

/-- Time Joan spent writing music -/
def writing_time : ℕ := 25

/-- Time Joan spent reading about piano history -/
def reading_time : ℕ := 38

/-- Time Joan has left for finger exerciser -/
def exerciser_time : ℕ := 27

/-- Theorem stating that the total time is equal to the sum of individual activity times -/
theorem total_time_equals_sum_of_activities : 
  total_time = piano_time + writing_time + reading_time + exerciser_time := by
  sorry

end NUMINAMATH_CALUDE_total_time_equals_sum_of_activities_l2612_261230


namespace NUMINAMATH_CALUDE_smaug_hoard_theorem_l2612_261288

/-- Calculates the total value of Smaug's hoard in copper coins -/
def smaug_hoard_value : ℕ :=
  let gold_coins : ℕ := 100
  let silver_coins : ℕ := 60
  let copper_coins : ℕ := 33
  let silver_to_copper : ℕ := 8
  let gold_to_silver : ℕ := 3
  
  let gold_value : ℕ := gold_coins * gold_to_silver * silver_to_copper
  let silver_value : ℕ := silver_coins * silver_to_copper
  let total_value : ℕ := gold_value + silver_value + copper_coins
  
  total_value

theorem smaug_hoard_theorem : smaug_hoard_value = 2913 := by
  sorry

end NUMINAMATH_CALUDE_smaug_hoard_theorem_l2612_261288


namespace NUMINAMATH_CALUDE_min_value_theorem_l2612_261254

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9*x + 1/x^6 ≥ 10 ∧ (9*x + 1/x^6 = 10 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2612_261254


namespace NUMINAMATH_CALUDE_square_less_than_triple_l2612_261202

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l2612_261202


namespace NUMINAMATH_CALUDE_total_pawns_left_l2612_261283

/-- The number of pawns each player starts with in a chess game -/
def initial_pawns : ℕ := 8

/-- The number of pawns Kennedy has lost -/
def kennedy_lost : ℕ := 4

/-- The number of pawns Riley has lost -/
def riley_lost : ℕ := 1

/-- Theorem: The total number of pawns left in the game is 11 -/
theorem total_pawns_left : 
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pawns_left_l2612_261283


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l2612_261213

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with four ones in a row -/
theorem zeros_not_adjacent_probability :
  (Nat.choose (total_elements - 1) num_zeros) / (Nat.choose total_elements num_zeros) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l2612_261213


namespace NUMINAMATH_CALUDE_rational_solution_system_l2612_261222

theorem rational_solution_system (x y z t w : ℚ) :
  t^2 - w^2 + z^2 = 2*x*y ∧
  t^2 - y^2 + w^2 = 2*x*z ∧
  t^2 - w^2 + x^2 = 2*y*z →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

#check rational_solution_system

end NUMINAMATH_CALUDE_rational_solution_system_l2612_261222


namespace NUMINAMATH_CALUDE_complement_union_problem_l2612_261223

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_union_problem (A B : Finset Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3})
  (h4 : (U \ B) ∩ A = {1,2})
  (h5 : (U \ A) ∩ B = {4,5}) :
  U \ (A ∪ B) = {6,7,8} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2612_261223


namespace NUMINAMATH_CALUDE_letter_writing_is_permutation_problem_l2612_261246

/-- A function that represents the number of letters written when n people write to each other once -/
def letters_written (n : ℕ) : ℕ := n * (n - 1)

/-- A function that represents whether a scenario is a permutation problem -/
def is_permutation_problem (scenario : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ scenario n ≠ scenario (n - 1)

theorem letter_writing_is_permutation_problem :
  is_permutation_problem letters_written :=
sorry


end NUMINAMATH_CALUDE_letter_writing_is_permutation_problem_l2612_261246


namespace NUMINAMATH_CALUDE_mia_bought_three_more_notebooks_l2612_261279

/-- Represents the price of a single notebook in cents -/
def notebook_price : ℕ := 50

/-- Represents the number of notebooks Colin bought -/
def colin_notebooks : ℕ := 5

/-- Represents the number of notebooks Mia bought -/
def mia_notebooks : ℕ := 8

/-- Represents Colin's total payment in cents -/
def colin_payment : ℕ := 250

/-- Represents Mia's total payment in cents -/
def mia_payment : ℕ := 400

theorem mia_bought_three_more_notebooks :
  mia_notebooks = colin_notebooks + 3 ∧
  notebook_price > 1 ∧
  notebook_price * colin_notebooks = colin_payment ∧
  notebook_price * mia_notebooks = mia_payment :=
by sorry

end NUMINAMATH_CALUDE_mia_bought_three_more_notebooks_l2612_261279


namespace NUMINAMATH_CALUDE_f_negative_a_value_l2612_261277

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem f_negative_a_value (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_value_l2612_261277


namespace NUMINAMATH_CALUDE_rainfall_ratio_rainfall_ratio_is_three_to_two_l2612_261289

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 25 →
  second_week = 15 →
  second_week / (total - second_week) = 3 / 2 := by
  sorry

/-- The ratio of rainfall in the second week to the first week is 3:2. -/
theorem rainfall_ratio_is_three_to_two :
  ∃ (total : ℝ) (second_week : ℝ),
    total = 25 ∧
    second_week = 15 ∧
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_rainfall_ratio_is_three_to_two_l2612_261289


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2612_261269

/-- Given a geometric sequence with common ratio 2, prove that S_4 / a_1 = 15 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  (a 0 * (1 - 2^4)) / (a 0 * (1 - 2)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2612_261269


namespace NUMINAMATH_CALUDE_arithmetic_operations_correctness_l2612_261235

theorem arithmetic_operations_correctness :
  ((-2 : ℤ) + 8 ≠ 10) ∧
  ((-1 : ℤ) - 3 = -4) ∧
  ((-2 : ℤ) * 2 ≠ 4) ∧
  ((-8 : ℚ) / (-1) ≠ -1/8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_correctness_l2612_261235


namespace NUMINAMATH_CALUDE_objects_per_hour_l2612_261217

/-- The number of objects one person can make in an hour -/
def n : ℕ := 12

/-- The time Ann works in hours -/
def ann_time : ℚ := 1

/-- The time Bob works in hours -/
def bob_time : ℚ := 2/3

/-- The time Cody works in hours -/
def cody_time : ℚ := 1/3

/-- The time Deb works in hours -/
def deb_time : ℚ := 1/3

/-- The total number of objects made -/
def total_objects : ℕ := 28

theorem objects_per_hour :
  n * (ann_time + bob_time + cody_time + deb_time) = total_objects := by
  sorry

end NUMINAMATH_CALUDE_objects_per_hour_l2612_261217


namespace NUMINAMATH_CALUDE_point_condition_y_intercept_condition_l2612_261256

/-- The equation of the line -/
def line_equation (x y t : ℝ) : Prop :=
  2 * x + (t - 2) * y + 3 - 2 * t = 0

/-- Theorem: If the line passes through (1, 1), then t = 5 -/
theorem point_condition (t : ℝ) : line_equation 1 1 t → t = 5 := by
  sorry

/-- Theorem: If the y-intercept of the line is -3, then t = 9/5 -/
theorem y_intercept_condition (t : ℝ) : line_equation 0 (-3) t → t = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_point_condition_y_intercept_condition_l2612_261256


namespace NUMINAMATH_CALUDE_system_solution_l2612_261280

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * y = |2 * x + 3| - |2 * x - 3|
def equation2 (x y : ℝ) : Prop := 4 * x = |y + 2| - |y - 2|

-- Define the solution set
def solutionSet (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y ↔ solutionSet x y :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2612_261280


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2612_261233

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 + a 13 = 20) 
  (h3 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2612_261233


namespace NUMINAMATH_CALUDE_parabola_sum_l2612_261262

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) : 
  p.y_coord 4 = 2 ∧  -- vertex (4,2)
  p.y_coord 1 = -4 ∧  -- point (1,-4)
  p.y_coord 7 = 0 ∧  -- point (7,0)
  (∀ x : ℝ, p.y_coord (8 - x) = p.y_coord x) →  -- vertical axis of symmetry at x = 4
  p.a + p.b + p.c = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l2612_261262


namespace NUMINAMATH_CALUDE_lost_revenue_calculation_l2612_261237

/-- Represents the movie theater scenario --/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater --/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity : ℚ) * theater.ticketPrice - (theater.ticketsSold : ℚ) * theater.ticketPrice

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation (theater : MovieTheater) 
  (h1 : theater.capacity = 50)
  (h2 : theater.ticketPrice = 8)
  (h3 : theater.ticketsSold = 24) : 
  lostRevenue theater = 208 := by
  sorry

#eval lostRevenue { capacity := 50, ticketPrice := 8, ticketsSold := 24 }

end NUMINAMATH_CALUDE_lost_revenue_calculation_l2612_261237


namespace NUMINAMATH_CALUDE_addition_point_value_l2612_261229

/-- The 0.618 method for finding the optimal addition amount --/
def addition_point (lower upper good : ℝ) : ℝ :=
  upper + lower - good

/-- Theorem: The addition point value using the 0.618 method --/
theorem addition_point_value (lower upper good : ℝ)
  (h_range : lower = 628 ∧ upper = 774)
  (h_good : good = lower + 0.618 * (upper - lower))
  (h_good_value : good = 718) :
  addition_point lower upper good = 684 := by
  sorry

#eval addition_point 628 774 718

end NUMINAMATH_CALUDE_addition_point_value_l2612_261229


namespace NUMINAMATH_CALUDE_budget_reduction_proof_l2612_261216

def magazine_cost : ℝ := 840.00
def online_cost_pounds : ℝ := 960.00
def exchange_rate : ℝ := 1.40
def magazine_cut_rate : ℝ := 0.30
def online_cut_rate : ℝ := 0.20

def total_reduction : ℝ :=
  (magazine_cost * magazine_cut_rate) +
  (online_cost_pounds * online_cut_rate * exchange_rate)

theorem budget_reduction_proof :
  total_reduction = 520.80 := by
sorry

end NUMINAMATH_CALUDE_budget_reduction_proof_l2612_261216


namespace NUMINAMATH_CALUDE_range_of_a_l2612_261241

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2612_261241


namespace NUMINAMATH_CALUDE_classroom_ratio_l2612_261298

theorem classroom_ratio :
  ∀ (boys girls : ℕ),
  boys + girls = 36 →
  boys = girls + 6 →
  (boys : ℚ) / girls = 7 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l2612_261298


namespace NUMINAMATH_CALUDE_train_length_calculation_l2612_261205

/-- Calculates the length of a train given its speed, the time it takes to pass a platform, and the length of the platform. -/
theorem train_length_calculation (train_speed : Real) (platform_pass_time : Real) (platform_length : Real) :
  train_speed = 60 →
  platform_pass_time = 23.998080153587715 →
  platform_length = 260 →
  let train_speed_mps := train_speed * 1000 / 3600
  let total_distance := train_speed_mps * platform_pass_time
  let train_length := total_distance - platform_length
  train_length = 139.968003071754 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2612_261205


namespace NUMINAMATH_CALUDE_meeting_size_l2612_261278

/-- Represents the number of people attending the meeting -/
def n : ℕ → ℕ := λ k => 12 * k

/-- Represents the number of handshakes each person makes -/
def handshakes : ℕ → ℕ := λ k => 3 * k + 6

/-- Represents the number of mutual handshakes between any two people -/
def mutual_handshakes : ℕ → ℚ := λ k => 
  ((3 * k + 6) * (3 * k + 5)) / (12 * k - 1)

theorem meeting_size : 
  ∃ k : ℕ, k > 0 ∧ 
    (∀ i j : Fin (n k), i ≠ j → 
      (mutual_handshakes k).num % (mutual_handshakes k).den = 0) ∧
    n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_meeting_size_l2612_261278


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_and_perimeter_l2612_261257

/-- Given a square with diagonal 2x and perimeter 16x, prove its area is 16x² -/
theorem square_area_from_diagonal_and_perimeter (x : ℝ) :
  let diagonal := 2 * x
  let perimeter := 16 * x
  let side := perimeter / 4
  let area := side ^ 2
  diagonal ^ 2 = 2 * side ^ 2 ∧ perimeter = 4 * side → area = 16 * x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_and_perimeter_l2612_261257


namespace NUMINAMATH_CALUDE_existence_of_equal_sums_l2612_261228

theorem existence_of_equal_sums (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (ha : ∀ i j : Fin m, i ≤ j → a i ≤ a j) 
  (hb : ∀ i j : Fin n, i ≤ j → b i ≤ b j)
  (ha_bound : ∀ i : Fin m, a i ≤ n)
  (hb_bound : ∀ i : Fin n, b i ≤ m) :
  ∃ (i : Fin m) (j : Fin n), a i + i.val + 1 = b j + j.val + 1 := by
sorry

end NUMINAMATH_CALUDE_existence_of_equal_sums_l2612_261228


namespace NUMINAMATH_CALUDE_license_plate_combinations_count_l2612_261275

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of choices for the last character (letters + digits) -/
def last_char_choices : ℕ := num_letters + num_digits

/-- A function to calculate the number of valid license plate combinations -/
def license_plate_combinations : ℕ :=
  num_letters * last_char_choices * 2

/-- Theorem stating that the number of valid license plate combinations is 1872 -/
theorem license_plate_combinations_count :
  license_plate_combinations = 1872 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_count_l2612_261275


namespace NUMINAMATH_CALUDE_number_of_boys_l2612_261236

/-- The number of boys in a school, given the number of girls and the difference between boys and girls. -/
theorem number_of_boys (girls : ℕ) (difference : ℕ) : girls = 1225 → difference = 1750 → girls + difference = 2975 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l2612_261236


namespace NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l2612_261252

/-- Given two perpendicular lines l₁ and l₂, prove that the minimum value of |ab| is 2 -/
theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, a^2 * x + y + 2 = 0 → b * x - (a^2 + 1) * y - 1 = 0 → 
   (a^2 * 1) * (b / (a^2 + 1)) = -1) →
  ∃ (min : ℝ), min = 2 ∧ ∀ a' b' : ℝ, 
    (∀ x y : ℝ, (a')^2 * x + y + 2 = 0 → b' * x - ((a')^2 + 1) * y - 1 = 0 → 
     ((a')^2 * 1) * (b' / ((a')^2 + 1)) = -1) →
    |a' * b'| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l2612_261252


namespace NUMINAMATH_CALUDE_mumblian_language_word_count_l2612_261225

/-- The number of letters in the Mumblian alphabet -/
def alphabet_size : ℕ := 5

/-- The maximum word length in the Mumblian language -/
def max_word_length : ℕ := 3

/-- The number of words of a given length in the Mumblian language -/
def words_of_length (n : ℕ) : ℕ := 
  if n > 0 ∧ n ≤ max_word_length then alphabet_size ^ n else 0

/-- The total number of words in the Mumblian language -/
def total_words : ℕ := 
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3)

theorem mumblian_language_word_count : total_words = 155 := by
  sorry

end NUMINAMATH_CALUDE_mumblian_language_word_count_l2612_261225


namespace NUMINAMATH_CALUDE_function_is_linear_l2612_261219

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_function_is_linear_l2612_261219


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l2612_261270

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the property of being invertible on [c, ∞)
def is_invertible_on_range (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → f x = f y → x = y

-- Theorem statement
theorem smallest_invertible_domain : 
  (∀ c < 3, ¬(is_invertible_on_range f c)) ∧ 
  (is_invertible_on_range f 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l2612_261270


namespace NUMINAMATH_CALUDE_valid_x_values_l2612_261286

-- Define the property for x
def is_valid_x (x : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    x ^ 2 = 2525000000 + a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f * 1 + 89 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

-- State the theorem
theorem valid_x_values : 
  ∀ x : ℕ, is_valid_x x ↔ (x = 502567 ∨ x = 502583) :=
sorry

end NUMINAMATH_CALUDE_valid_x_values_l2612_261286


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2612_261284

/-- Given a point in polar coordinates (-5, 5π/6), prove that it is equivalent to (5, 11π/6) in standard polar coordinate representation. -/
theorem polar_coordinate_equivalence :
  let given_point : ℝ × ℝ := (-5, 5 * Real.pi / 6)
  let standard_point : ℝ × ℝ := (5, 11 * Real.pi / 6)
  (∀ (r θ : ℝ), r > 0 → 0 ≤ θ → θ < 2 * Real.pi →
    (r * (Real.cos θ), r * (Real.sin θ)) =
    (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2))) →
  (standard_point.1 * (Real.cos standard_point.2), standard_point.1 * (Real.sin standard_point.2)) =
  (given_point.1 * (Real.cos given_point.2), given_point.1 * (Real.sin given_point.2)) :=
by sorry


end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l2612_261284


namespace NUMINAMATH_CALUDE_bookstore_shipment_l2612_261273

theorem bookstore_shipment (displayed_percentage : ℚ) (storeroom_books : ℕ) : 
  displayed_percentage = 30 / 100 →
  storeroom_books = 210 →
  ∃ total_books : ℕ, 
    (1 - displayed_percentage) * total_books = storeroom_books ∧
    total_books = 300 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_shipment_l2612_261273


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2612_261239

theorem complex_magnitude_problem (z : ℂ) : 
  z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2612_261239


namespace NUMINAMATH_CALUDE_book_club_unique_books_l2612_261218

theorem book_club_unique_books (tony dean breanna piper asher : ℕ)
  (tony_dean breanna_piper_asher dean_piper_tony asher_breanna_tony all_five : ℕ)
  (h_tony : tony = 23)
  (h_dean : dean = 20)
  (h_breanna : breanna = 30)
  (h_piper : piper = 26)
  (h_asher : asher = 25)
  (h_tony_dean : tony_dean = 5)
  (h_breanna_piper_asher : breanna_piper_asher = 6)
  (h_dean_piper_tony : dean_piper_tony = 4)
  (h_asher_breanna_tony : asher_breanna_tony = 3)
  (h_all_five : all_five = 2) :
  tony + dean + breanna + piper + asher -
  ((tony_dean - all_five) + (breanna_piper_asher - all_five) +
   (dean_piper_tony - all_five) + (asher_breanna_tony - all_five) + all_five) = 112 :=
by sorry

end NUMINAMATH_CALUDE_book_club_unique_books_l2612_261218


namespace NUMINAMATH_CALUDE_senate_arrangement_l2612_261248

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats num_republicans : ℕ) : ℕ :=
  (num_republicans - 1).factorial * (num_republicans.choose num_democrats) * num_democrats.factorial

/-- Theorem: The number of ways to arrange 4 Democrats and 6 Republicans around a circular table
    such that no two Democrats sit next to each other is 43,200. -/
theorem senate_arrangement :
  arrange_senators 4 6 = 43200 :=
sorry

end NUMINAMATH_CALUDE_senate_arrangement_l2612_261248


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2612_261281

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  contrapose!
  intro h
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2612_261281


namespace NUMINAMATH_CALUDE_mrs_sheridans_cats_l2612_261299

theorem mrs_sheridans_cats (initial_cats additional_cats : ℕ) :
  initial_cats = 17 →
  additional_cats = 14 →
  initial_cats + additional_cats = 31 :=
by sorry

end NUMINAMATH_CALUDE_mrs_sheridans_cats_l2612_261299


namespace NUMINAMATH_CALUDE_product_count_in_range_l2612_261210

theorem product_count_in_range (total_sample : ℕ) 
  (freq_96_100 : ℚ) (freq_98_104 : ℚ) (count_less_100 : ℕ) :
  freq_96_100 = 3/10 →
  freq_98_104 = 3/8 →
  count_less_100 = 36 →
  total_sample = count_less_100 / freq_96_100 →
  (freq_98_104 * total_sample : ℚ) = 60 :=
by sorry

end NUMINAMATH_CALUDE_product_count_in_range_l2612_261210


namespace NUMINAMATH_CALUDE_tom_reading_speed_l2612_261250

/-- Given that Tom reads 10 hours over 5 days, reads the same amount every day,
    and reads 700 pages in 7 days, prove that he can read 50 pages per hour. -/
theorem tom_reading_speed :
  ∀ (total_hours : ℕ) (days : ℕ) (total_pages : ℕ) (week_days : ℕ),
    total_hours = 10 →
    days = 5 →
    total_pages = 700 →
    week_days = 7 →
    (total_hours / days) * week_days ≠ 0 →
    total_pages / ((total_hours / days) * week_days) = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_speed_l2612_261250


namespace NUMINAMATH_CALUDE_gamma_value_l2612_261291

theorem gamma_value (γ δ : ℂ) : 
  (γ + δ).re > 0 →
  (Complex.I * (γ - 3 * δ)).re > 0 →
  δ = 4 + 3 * Complex.I →
  γ = 16 - 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_gamma_value_l2612_261291


namespace NUMINAMATH_CALUDE_mary_shirts_left_l2612_261231

/-- The number of shirts Mary has left after giving away some of her blue and brown shirts -/
def shirts_left (blue : ℕ) (brown : ℕ) : ℕ :=
  (blue - blue / 2) + (brown - brown / 3)

/-- Theorem stating that Mary has 37 shirts left -/
theorem mary_shirts_left : shirts_left 26 36 = 37 := by
  sorry

end NUMINAMATH_CALUDE_mary_shirts_left_l2612_261231


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l2612_261244

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 / a + 1 / b = 1 → 2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l2612_261244


namespace NUMINAMATH_CALUDE_tommy_initial_balloons_l2612_261200

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- The number of balloons Tommy had to start with -/
def initial_balloons : ℕ := total_balloons - balloons_from_mom

theorem tommy_initial_balloons : initial_balloons = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_initial_balloons_l2612_261200


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l2612_261220

theorem division_multiplication_problem : ((-128) / (-16)) * 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l2612_261220


namespace NUMINAMATH_CALUDE_product_of_positive_reals_l2612_261208

theorem product_of_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * (3 ^ (1/4)))
  (h2 : x * z = 42 * (3 ^ (1/4)))
  (h3 : y * z = 21 * (3 ^ (1/4))) :
  x * y * z = Real.sqrt 63504 := by
sorry

end NUMINAMATH_CALUDE_product_of_positive_reals_l2612_261208


namespace NUMINAMATH_CALUDE_triangle_ratio_l2612_261263

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

#check triangle_ratio

end NUMINAMATH_CALUDE_triangle_ratio_l2612_261263


namespace NUMINAMATH_CALUDE_no_solution_system_l2612_261265

/-- Proves that the system of equations 3x - 4y = 10 and 6x - 8y = 12 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 10) ∧ (6 * x - 8 * y = 12) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l2612_261265


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l2612_261267

theorem not_all_greater_than_one (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬(a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l2612_261267


namespace NUMINAMATH_CALUDE_toy_cost_price_l2612_261204

theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) :
  total_selling_price = 18900 →
  num_toys_sold = 18 →
  num_toys_gain = 3 →
  ∃ (cost_price : ℕ),
    cost_price * num_toys_sold + cost_price * num_toys_gain = total_selling_price ∧
    cost_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2612_261204


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2612_261264

theorem power_fraction_simplification :
  (2^2020 - 2^2018) / (2^2020 + 2^2018) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2612_261264


namespace NUMINAMATH_CALUDE_distance_between_locations_l2612_261207

/-- The distance between two locations A and B given two cars meeting conditions --/
theorem distance_between_locations (speed_B : ℝ) (h1 : speed_B > 0) : 
  let speed_A := 1.2 * speed_B
  let midpoint_to_meeting := 8
  let time := 2 * midpoint_to_meeting / (speed_A - speed_B)
  (speed_A + speed_B) * time = 176 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l2612_261207


namespace NUMINAMATH_CALUDE_ball_bounce_ratio_l2612_261294

theorem ball_bounce_ratio (h₀ : ℝ) (h₅ : ℝ) (r : ℝ) :
  h₀ = 96 →
  h₅ = 3 →
  h₅ = h₀ * r^5 →
  r = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_ratio_l2612_261294


namespace NUMINAMATH_CALUDE_another_two_digit_prime_digit_number_l2612_261209

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem another_two_digit_prime_digit_number : 
  ∃ n : ℕ, is_two_digit n ∧ 
           is_prime (n / 10) ∧ 
           is_prime (n % 10) ∧ 
           n ≠ 23 :=
sorry

end NUMINAMATH_CALUDE_another_two_digit_prime_digit_number_l2612_261209


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_44_l2612_261215

/-- The area of a rectangle containing two smaller squares and one larger square -/
theorem rectangle_area (small_square_area : ℝ) (h1 : small_square_area = 4) : ℝ :=
  let small_side := Real.sqrt small_square_area
  let large_side := 3 * small_side
  2 * small_square_area + large_side ^ 2

/-- Proof that the area of the rectangle is 44 square inches -/
theorem rectangle_area_is_44 : rectangle_area 4 rfl = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_44_l2612_261215


namespace NUMINAMATH_CALUDE_max_profit_l2612_261232

def factory_price_A : ℕ := 10
def factory_price_B : ℕ := 18
def selling_price_A : ℕ := 12
def selling_price_B : ℕ := 22
def total_vehicles : ℕ := 130

def profit_function (x : ℕ) : ℤ :=
  -2 * x + 520

def is_valid_purchase (x : ℕ) : Prop :=
  x ≤ total_vehicles ∧ (total_vehicles - x) ≤ 2 * x

theorem max_profit :
  ∃ (x : ℕ), is_valid_purchase x ∧
    ∀ (y : ℕ), is_valid_purchase y → profit_function x ≥ profit_function y ∧
    profit_function x = 432 :=
  sorry

end NUMINAMATH_CALUDE_max_profit_l2612_261232


namespace NUMINAMATH_CALUDE_chord_angle_cosine_l2612_261249

theorem chord_angle_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  2 * r * Real.sin (α / 2) = 2 ∧
  2 * r * Real.sin (β / 2) = 3 ∧
  2 * r * Real.sin ((α + β) / 2) = 4 ∧
  α + β < π →
  Real.cos α = 17 / 32 := by
sorry

end NUMINAMATH_CALUDE_chord_angle_cosine_l2612_261249


namespace NUMINAMATH_CALUDE_max_value_of_fraction_sum_l2612_261234

theorem max_value_of_fraction_sum (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 1 ∧
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 ∧
    (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_sum_l2612_261234


namespace NUMINAMATH_CALUDE_existence_of_special_set_l2612_261292

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l2612_261292


namespace NUMINAMATH_CALUDE_exam_marks_problem_l2612_261201

/-- Examination marks problem -/
theorem exam_marks_problem (full_marks : ℕ) (a_marks b_marks c_marks d_marks : ℕ) :
  full_marks = 500 →
  a_marks = (9 : ℕ) * b_marks / 10 →
  c_marks = (4 : ℕ) * d_marks / 5 →
  a_marks = 360 →
  d_marks = (4 : ℕ) * full_marks / 5 →
  b_marks - c_marks = c_marks / 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_problem_l2612_261201


namespace NUMINAMATH_CALUDE_badminton_probability_l2612_261240

theorem badminton_probability (p : ℝ) (n : ℕ) : 
  p = 3/4 → n = 3 → 
  (1 - p)^n = 1/64 → 
  n.choose 1 * p * (1 - p)^(n-1) = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_badminton_probability_l2612_261240


namespace NUMINAMATH_CALUDE_similar_right_triangles_l2612_261227

theorem similar_right_triangles (y : ℝ) : 
  -- First triangle with legs 15 and 12
  let a₁ : ℝ := 15
  let b₁ : ℝ := 12
  -- Second triangle with legs y and 9
  let a₂ : ℝ := y
  let b₂ : ℝ := 9
  -- Triangles are similar (corresponding sides are proportional)
  a₁ / a₂ = b₁ / b₂ →
  -- The value of y is 11.25
  y = 11.25 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l2612_261227
