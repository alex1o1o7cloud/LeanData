import Mathlib

namespace NUMINAMATH_CALUDE_max_triangle_area_l471_47197

def parabola (x : ℝ) : ℝ := x^2 - 6*x + 9

theorem max_triangle_area :
  let A : ℝ × ℝ := (0, 9)
  let B : ℝ × ℝ := (6, 9)
  ∀ p q : ℝ,
    1 ≤ p → p ≤ 6 →
    q = parabola p →
    let C : ℝ × ℝ := (p, q)
    let area := abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)) / 2
    area ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l471_47197


namespace NUMINAMATH_CALUDE_student_weight_l471_47126

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight + sister_weight = 116)
  (h2 : student_weight - 5 = 2 * sister_weight) : 
  student_weight = 79 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l471_47126


namespace NUMINAMATH_CALUDE_percent_relation_l471_47120

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  4 * b / a = 20 / 9 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l471_47120


namespace NUMINAMATH_CALUDE_slower_train_speed_calculation_l471_47150

/-- The speed of the faster train in kilometers per hour -/
def faster_train_speed : ℝ := 162

/-- The length of the faster train in meters -/
def faster_train_length : ℝ := 1320

/-- The time taken by the faster train to cross a man in the slower train, in seconds -/
def crossing_time : ℝ := 33

/-- The speed of the slower train in kilometers per hour -/
def slower_train_speed : ℝ := 18

theorem slower_train_speed_calculation :
  let relative_speed := (faster_train_speed - slower_train_speed) * 1000 / 3600
  faster_train_length = relative_speed * crossing_time →
  slower_train_speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_slower_train_speed_calculation_l471_47150


namespace NUMINAMATH_CALUDE_train_crossing_time_l471_47187

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 375 →
  train_speed_kmh = 90 →
  crossing_time = 15 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l471_47187


namespace NUMINAMATH_CALUDE_rosie_pie_production_l471_47131

/-- Given that Rosie can make 3 pies out of 12 apples, prove that she can make 9 pies with 36 apples. -/
theorem rosie_pie_production (apples_per_batch : ℕ) (pies_per_batch : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : total_apples = 36) :
  (total_apples / (apples_per_batch / pies_per_batch)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pie_production_l471_47131


namespace NUMINAMATH_CALUDE_three_distinct_zeros_l471_47177

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem: For f to have three distinct real zeros, a must be in (1/4, +∞) -/
theorem three_distinct_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  a > (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_l471_47177


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l471_47103

open Real

theorem trigonometric_equation_solution (z : ℝ) :
  cos z ≠ 0 →
  sin z ≠ 0 →
  (5.38 * (1 / (cos z)^4) = 160/9 - (2 * ((cos (2*z) / sin (2*z)) * (cos z / sin z) + 1)) / (sin z)^2) →
  ∃ k : ℤ, z = (π/6) * (3 * k + 1) ∨ z = (π/6) * (3 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l471_47103


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l471_47119

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  5 * a * b^2 - 3 * a * b^2 + (1/3) * a * b^2 = (7/3) * a * b^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) :
  (7 * m^2 * n - 5 * m) - (4 * m^2 * n - 5 * m) = 3 * m^2 * n := by sorry

-- Problem 3
theorem simplify_and_evaluate_expression (x y : ℝ) 
  (hx : x = -1/4) (hy : y = 2) :
  2 * x^2 * y - 2 * (x * y^2 + 2 * x^2 * y) + 2 * (x^2 * y - 3 * x * y^2) = 8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_and_evaluate_expression_l471_47119


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l471_47171

theorem newberg_airport_passengers :
  let on_time : ℕ := 14507
  let late : ℕ := 213
  on_time + late = 14720 :=
by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l471_47171


namespace NUMINAMATH_CALUDE_cylinder_radius_l471_47193

/-- The original radius of a cylinder satisfying specific conditions -/
theorem cylinder_radius : ∃ (r : ℝ), r > 0 ∧ 
  (∀ (y : ℝ), 
    (2 * π * ((r + 6)^2 - r^2) = y) ∧ 
    (6 * π * r^2 = y)) → 
  r = 6 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l471_47193


namespace NUMINAMATH_CALUDE_election_invalid_votes_percentage_l471_47165

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (votes_B : ℕ) 
  (h_total : total_votes = 9720)
  (h_B : votes_B = 3159)
  (h_difference : ∃ (votes_A : ℕ), votes_A = votes_B + (15 * total_votes) / 100) :
  (total_votes - (votes_B + (votes_B + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
sorry

end NUMINAMATH_CALUDE_election_invalid_votes_percentage_l471_47165


namespace NUMINAMATH_CALUDE_min_value_quadratic_l471_47142

theorem min_value_quadratic :
  ∃ (min_z : ℝ), min_z = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ min_z :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l471_47142


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l471_47156

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream speed_along_stream : ℝ) 
  (against_stream speed_against_stream : ℝ) :
  along_stream = 9 → against_stream = 5 →
  speed_along_stream = along_stream / 1 →
  speed_against_stream = against_stream / 1 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = speed_along_stream ∧
    boat_speed - stream_speed = speed_against_stream ∧
    boat_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l471_47156


namespace NUMINAMATH_CALUDE_min_value_A_min_value_A_equality_l471_47118

theorem min_value_A (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  let A := (Real.sqrt (3 * x^4 + y) + Real.sqrt (3 * y^4 + z) + Real.sqrt (3 * z^4 + x) - 3) / (x * y + y * z + z * x)
  A ≥ 1 := by
  sorry

theorem min_value_A_equality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  let A := (Real.sqrt (3 * x^4 + y) + Real.sqrt (3 * y^4 + z) + Real.sqrt (3 * z^4 + x) - 3) / (x * y + y * z + z * x)
  (A = 1) ↔ (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_A_min_value_A_equality_l471_47118


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l471_47107

theorem quadratic_rational_solutions (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l471_47107


namespace NUMINAMATH_CALUDE_cog_production_90_workers_2_hours_l471_47164

/-- Represents the production capabilities of workers in a factory --/
structure ProductionRate where
  gears_per_hour : ℝ
  cogs_per_hour : ℝ

/-- Calculates the total production for a given number of workers, hours, and production rate --/
def total_production (workers : ℝ) (hours : ℝ) (rate : ProductionRate) : ProductionRate :=
  { gears_per_hour := workers * hours * rate.gears_per_hour,
    cogs_per_hour := workers * hours * rate.cogs_per_hour }

/-- Theorem stating the production of cogs by 90 workers in 2 hours --/
theorem cog_production_90_workers_2_hours 
  (rate : ProductionRate)
  (h1 : total_production 150 1 rate = { gears_per_hour := 450, cogs_per_hour := 300 })
  (h2 : total_production 100 1.5 rate = { gears_per_hour := 300, cogs_per_hour := 375 })
  (h3 : (total_production 90 2 rate).gears_per_hour = 360) :
  (total_production 90 2 rate).cogs_per_hour = 180 := by
  sorry

#check cog_production_90_workers_2_hours

end NUMINAMATH_CALUDE_cog_production_90_workers_2_hours_l471_47164


namespace NUMINAMATH_CALUDE_circle_center_parabola_focus_l471_47145

/-- The value of p for which the center of the circle x^2 + y^2 - 6x = 0 
    is exactly the focus of the parabola y^2 = 2px (p > 0) -/
theorem circle_center_parabola_focus (p : ℝ) : p > 0 → 
  (∃ (x y : ℝ), x^2 + y^2 - 6*x = 0 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), x^2 + y^2 - 6*x = 0 → x = 3 ∧ y = 0) →
  (∀ (x y : ℝ), y^2 = 2*p*x → x = p/2 ∧ y = 0) →
  p = 6 := by sorry

end NUMINAMATH_CALUDE_circle_center_parabola_focus_l471_47145


namespace NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l471_47185

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_symmetry_implies_ordering (b c : ℝ) 
  (h : ∀ t : ℝ, f (2 + t) b c = f (2 - t) b c) : 
  f 2 b c < f 1 b c ∧ f 1 b c < f 4 b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_implies_ordering_l471_47185


namespace NUMINAMATH_CALUDE_triangle_height_ratio_l471_47158

theorem triangle_height_ratio (a b c : ℝ) (ha hb hc : a > 0 ∧ b > 0 ∧ c > 0) 
  (side_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  ∃ (h₁ h₂ h₃ : ℝ), h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ 
    (a * h₁ = b * h₂) ∧ (b * h₂ = c * h₃) ∧
    h₁ / 20 = h₂ / 15 ∧ h₂ / 15 = h₃ / 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_height_ratio_l471_47158


namespace NUMINAMATH_CALUDE_travel_time_calculation_l471_47190

theorem travel_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 300 →
  speed1 = 30 →
  speed2 = 25 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l471_47190


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l471_47144

/-- Given a line segment with midpoint (-1, 4) and one endpoint (3, -1), 
    the other endpoint is (-5, 9). -/
theorem other_endpoint_of_line_segment (m x₁ y₁ x₂ y₂ : ℝ) : 
  m = (-1 : ℝ) ∧ 
  (4 : ℝ) = (y₁ + y₂) / 2 ∧ 
  x₁ = (3 : ℝ) ∧ 
  y₁ = (-1 : ℝ) ∧ 
  m = (x₁ + x₂) / 2 → 
  x₂ = (-5 : ℝ) ∧ y₂ = (9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l471_47144


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l471_47175

/-- The maximum area of a rectangle with a perimeter of 16 meters -/
theorem max_area_rectangle_with_fixed_perimeter : 
  ∀ (length width : ℝ), 
  length > 0 → width > 0 → 
  2 * (length + width) = 16 → 
  length * width ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l471_47175


namespace NUMINAMATH_CALUDE_sin_150_degrees_l471_47162

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l471_47162


namespace NUMINAMATH_CALUDE_cube_of_product_l471_47189

theorem cube_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l471_47189


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l471_47154

theorem quadratic_discriminant_zero_implies_geometric_progression
  (k a b c : ℝ) (h1 : k ≠ 0) :
  4 * k^2 * (b^2 - a*c) = 0 →
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l471_47154


namespace NUMINAMATH_CALUDE_log_comparison_l471_47143

theorem log_comparison : Real.log 7 / Real.log 5 > Real.log 17 / Real.log 13 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l471_47143


namespace NUMINAMATH_CALUDE_race_start_distance_l471_47123

theorem race_start_distance (speed_a speed_b : ℝ) (total_distance : ℝ) (start_distance : ℝ) : 
  speed_a = (5 / 3) * speed_b →
  total_distance = 200 →
  total_distance / speed_a = (total_distance - start_distance) / speed_b →
  start_distance = 80 := by
sorry

end NUMINAMATH_CALUDE_race_start_distance_l471_47123


namespace NUMINAMATH_CALUDE_solution_to_equation_l471_47147

theorem solution_to_equation : ∃ x : ℝ, (2 / (x + 5) = 1 / x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l471_47147


namespace NUMINAMATH_CALUDE_opposite_seats_theorem_l471_47100

/-- Represents a circular seating arrangement -/
structure CircularArrangement where
  total_seats : ℕ
  is_valid : total_seats > 0

/-- Checks if two positions are opposite in a circular arrangement -/
def are_opposite (c : CircularArrangement) (pos1 pos2 : ℕ) : Prop :=
  pos1 ≤ c.total_seats ∧ pos2 ≤ c.total_seats ∧
  (pos2 - pos1) % c.total_seats = c.total_seats / 2

/-- The main theorem stating that if positions 10 and 29 are opposite, 
    the total number of seats is 38 -/
theorem opposite_seats_theorem :
  ∀ c : CircularArrangement, are_opposite c 10 29 → c.total_seats = 38 :=
by sorry

end NUMINAMATH_CALUDE_opposite_seats_theorem_l471_47100


namespace NUMINAMATH_CALUDE_mika_stickers_total_l471_47180

/-- The total number of stickers Mika has after receiving stickers from various sources -/
theorem mika_stickers_total : 
  let initial : ℝ := 20.5
  let bought : ℝ := 26.3
  let birthday : ℝ := 19.75
  let sister : ℝ := 6.25
  let mother : ℝ := 57.65
  let cousin : ℝ := 15.8
  initial + bought + birthday + sister + mother + cousin = 146.25 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_total_l471_47180


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l471_47109

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, a * c^2 ≤ b * c^2 :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l471_47109


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_7_l471_47159

theorem remainder_11_pow_2023_mod_7 : 11^2023 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_7_l471_47159


namespace NUMINAMATH_CALUDE_missing_number_proof_l471_47163

theorem missing_number_proof : ∃ (x : ℤ), |7 - 8 * (x - 12)| - |5 - 11| = 73 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l471_47163


namespace NUMINAMATH_CALUDE_power_sum_to_quadratic_expression_l471_47108

theorem power_sum_to_quadratic_expression (x : ℝ) :
  5 * (3 : ℝ)^x = 243 →
  (x + 2) * (x - 2) = 21 - 10 * (Real.log 5 / Real.log 3) + (Real.log 5 / Real.log 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_to_quadratic_expression_l471_47108


namespace NUMINAMATH_CALUDE_january_first_day_l471_47176

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  tuesdays : Nat
  saturdays : Nat

/-- Returns the day of the week for the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

theorem january_first_day (m : Month) 
  (h1 : m.days = 31)
  (h2 : m.tuesdays = 4)
  (h3 : m.saturdays = 4) :
  firstDayOfMonth m = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_january_first_day_l471_47176


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l471_47178

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l471_47178


namespace NUMINAMATH_CALUDE_number_ordering_l471_47168

theorem number_ordering : (1 : ℚ) / 5 < (25 : ℚ) / 100 ∧ (25 : ℚ) / 100 < (42 : ℚ) / 100 ∧ (42 : ℚ) / 100 < (1 : ℚ) / 2 ∧ (1 : ℚ) / 2 < (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l471_47168


namespace NUMINAMATH_CALUDE_division_problem_l471_47183

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 100 →
  divisor = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l471_47183


namespace NUMINAMATH_CALUDE_average_pen_price_l471_47192

/-- Given the purchase of pens and pencils, prove the average price of a pen. -/
theorem average_pen_price 
  (total_cost : ℝ)
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : avg_pencil_price = 2) :
  (total_cost - num_pencils * avg_pencil_price) / num_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_pen_price_l471_47192


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l471_47166

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l471_47166


namespace NUMINAMATH_CALUDE_gcd_7429_12345_is_1_l471_47136

theorem gcd_7429_12345_is_1 : Nat.gcd 7429 12345 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7429_12345_is_1_l471_47136


namespace NUMINAMATH_CALUDE_paint_room_time_l471_47155

/-- The time required for Doug, Dave, and Diana to paint a room together -/
theorem paint_room_time (t : ℝ) 
  (hDoug : (1 : ℝ) / 5 * t = 1)  -- Doug can paint the room in 5 hours
  (hDave : (1 : ℝ) / 7 * t = 1)  -- Dave can paint the room in 7 hours
  (hDiana : (1 : ℝ) / 6 * t = 1) -- Diana can paint the room in 6 hours
  (hLunch : ℝ) (hLunchTime : hLunch = 2) -- 2-hour lunch break
  : ((1 : ℝ) / 5 + 1 / 7 + 1 / 6) * (t - hLunch) = 1 :=
by sorry

end NUMINAMATH_CALUDE_paint_room_time_l471_47155


namespace NUMINAMATH_CALUDE_actual_score_calculation_l471_47182

/-- Given the following conditions:
  * The passing threshold is 30% of the maximum score
  * The maximum possible score is 790
  * The actual score falls short of the passing threshold by 25 marks
  Prove that the actual score is 212 marks -/
theorem actual_score_calculation (passing_threshold : Real) (max_score : Nat) (shortfall : Nat) :
  passing_threshold = 0.30 →
  max_score = 790 →
  shortfall = 25 →
  ⌊passing_threshold * max_score⌋ - shortfall = 212 := by
  sorry

end NUMINAMATH_CALUDE_actual_score_calculation_l471_47182


namespace NUMINAMATH_CALUDE_condition_implies_right_triangle_l471_47161

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b)^2 = t.c^2 + 2*t.a*t.b

-- Define what it means for a triangle to be a right triangle
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- State the theorem
theorem condition_implies_right_triangle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_right_triangle_l471_47161


namespace NUMINAMATH_CALUDE_intersection_M_N_l471_47191

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l471_47191


namespace NUMINAMATH_CALUDE_bug_probability_after_seven_steps_l471_47181

-- Define the probability function
def probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => 1/3 * (1 - probability m)

-- State the theorem
theorem bug_probability_after_seven_steps :
  probability 7 = 182 / 729 :=
sorry

end NUMINAMATH_CALUDE_bug_probability_after_seven_steps_l471_47181


namespace NUMINAMATH_CALUDE_magnitude_v_l471_47172

/-- Given complex numbers u and v, prove that |v| = 5.2 under the given conditions -/
theorem magnitude_v (u v : ℂ) : 
  u * v = 24 - 10 * Complex.I → Complex.abs u = 5 → Complex.abs v = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l471_47172


namespace NUMINAMATH_CALUDE_current_speed_l471_47106

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 20)
  (h2 : speed_against_current = 14) :
  ∃ (current_speed : ℝ), current_speed = 3 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l471_47106


namespace NUMINAMATH_CALUDE_terms_before_five_l471_47199

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_five (a₁ : ℤ) (d : ℤ) :
  a₁ = 105 ∧ d = -5 →
  ∃ n : ℕ, 
    arithmetic_sequence a₁ d n = 5 ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > 5) ∧
    n - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_five_l471_47199


namespace NUMINAMATH_CALUDE_probability_both_white_probability_at_least_one_white_l471_47111

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | White

/-- Represents the outcome of drawing two balls -/
def DrawOutcome := BallColor × BallColor

/-- The set of all possible outcomes when drawing two balls with replacement -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The set of outcomes where both balls are white -/
def bothWhite : Finset DrawOutcome := sorry

/-- The set of outcomes where at least one ball is white -/
def atLeastOneWhite : Finset DrawOutcome := sorry

/-- The probability of an event occurring -/
def probability (event : Finset DrawOutcome) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem probability_both_white :
  probability bothWhite = 4/9 := by sorry

theorem probability_at_least_one_white :
  probability atLeastOneWhite = 8/9 := by sorry

end NUMINAMATH_CALUDE_probability_both_white_probability_at_least_one_white_l471_47111


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l471_47114

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of their difference is 2√5 -/
theorem parallel_vectors_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h_a : a = (1, 2)) 
  (h_b : b.2 = 6) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l471_47114


namespace NUMINAMATH_CALUDE_tangent_line_equation_l471_47184

/-- The equation of a line tangent to a unit circle that intersects a specific ellipse -/
theorem tangent_line_equation (k b : ℝ) (h_b_pos : b > 0) 
  (h_tangent : b^2 = k^2 + 1)
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 2 + y₁^2 = 1 ∧ 
    x₂^2 / 2 + y₂^2 = 1)
  (h_dot_product : 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      y₁ = k * x₁ + b → 
      y₂ = k * x₂ + b → 
      x₁^2 / 2 + y₁^2 = 1 → 
      x₂^2 / 2 + y₂^2 = 1 → 
      x₁ * x₂ + y₁ * y₂ = 2/3) :
  (k = 1 ∧ b = Real.sqrt 2) ∨ (k = -1 ∧ b = Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l471_47184


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l471_47160

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h1 : tens ≥ 1 ∧ tens ≤ 9
  h2 : ones ≥ 0 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h1 : hundreds ≥ 1 ∧ hundreds ≤ 9
  h2 : tens ≥ 0 ∧ tens ≤ 9
  h3 : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def twoDigitToNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a ThreeDigitNumber to its numerical value -/
def threeDigitToNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem smallest_three_digit_number 
  (ab : TwoDigitNumber) 
  (aab : ThreeDigitNumber) 
  (h1 : ab.tens = aab.hundreds ∧ ab.tens = aab.tens)
  (h2 : ab.ones = aab.ones)
  (h3 : ab.tens ≠ ab.ones)
  (h4 : twoDigitToNat ab = (threeDigitToNat aab) / 9) :
  225 ≤ threeDigitToNat aab :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l471_47160


namespace NUMINAMATH_CALUDE_roots_sum_sixth_power_l471_47122

theorem roots_sum_sixth_power (u v : ℝ) : 
  u^2 - 3 * u * Real.sqrt 3 + 3 = 0 →
  v^2 - 3 * v * Real.sqrt 3 + 3 = 0 →
  u^6 + v^6 = 178767 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_sixth_power_l471_47122


namespace NUMINAMATH_CALUDE_area_of_wxuv_l471_47105

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the division of a rectangle into four smaller rectangles -/
structure RectangleDivision where
  pqxw : Rectangle
  qrsx : Rectangle
  xstu : Rectangle
  wxuv : Rectangle

theorem area_of_wxuv (div : RectangleDivision)
  (h1 : div.pqxw.area = 9)
  (h2 : div.qrsx.area = 10)
  (h3 : div.xstu.area = 15) :
  div.wxuv.area = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_wxuv_l471_47105


namespace NUMINAMATH_CALUDE_expansion_coefficients_l471_47101

theorem expansion_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, (x + 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₁ = 7 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 127) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l471_47101


namespace NUMINAMATH_CALUDE_f_sum_equals_half_point_five_l471_47195

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(x+1) = -f(x) for all x -/
axiom f_period (x : ℝ) : f (x + 1) = -f x

/-- f(x) = x for x in (-1, 1) -/
axiom f_identity (x : ℝ) (h : x > -1 ∧ x < 1) : f x = x

/-- The main theorem to prove -/
theorem f_sum_equals_half_point_five : f 3 + f (-7.5) = 0.5 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_half_point_five_l471_47195


namespace NUMINAMATH_CALUDE_total_money_proof_l471_47124

def sam_money : ℕ := 75

def billy_money (sam : ℕ) : ℕ := 2 * sam - 25

def total_money (sam : ℕ) : ℕ := sam + billy_money sam

theorem total_money_proof : total_money sam_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l471_47124


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l471_47173

theorem largest_inscribed_triangle_area (r : ℝ) (hr : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l471_47173


namespace NUMINAMATH_CALUDE_distance_inequality_l471_47130

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the vertices of the quadrilateral
variable (A B C D : V)

-- Define the condition that all sides are equal
variable (h : ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - D‖ ∧ ‖C - D‖ = ‖D - A‖)

-- State the theorem
theorem distance_inequality (P : V) : 
  ‖P - A‖ < ‖P - B‖ + ‖P - C‖ + ‖P - D‖ := by sorry

end NUMINAMATH_CALUDE_distance_inequality_l471_47130


namespace NUMINAMATH_CALUDE_simplify_fraction_l471_47153

theorem simplify_fraction : 45 * (14 / 25) * (1 / 18) * (5 / 11) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l471_47153


namespace NUMINAMATH_CALUDE_expression_evaluation_l471_47157

theorem expression_evaluation :
  let x : ℚ := -1/2
  (x - 3)^2 + (x + 3)*(x - 3) - 2*x*(x - 2) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l471_47157


namespace NUMINAMATH_CALUDE_y48y_divisible_by_24_l471_47128

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (y : ℕ) : ℕ := y * 1000 + 480 + y

theorem y48y_divisible_by_24 :
  ∃! (y : ℕ), y < 10 ∧ is_divisible_by (four_digit_number y) 24 :=
sorry

end NUMINAMATH_CALUDE_y48y_divisible_by_24_l471_47128


namespace NUMINAMATH_CALUDE_intersection_points_count_l471_47125

/-- A triangle with sides divided into p equal segments, where p is an odd prime -/
structure DividedTriangle where
  p : ℕ
  is_odd_prime : Nat.Prime p ∧ p % 2 = 1

/-- The number of intersection points in a divided triangle -/
def intersection_points (t : DividedTriangle) : ℕ := 3 * (t.p - 1)^2

/-- Theorem: The number of intersection points in a divided triangle is 3(p-1)^2 -/
theorem intersection_points_count (t : DividedTriangle) : 
  intersection_points t = 3 * (t.p - 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_count_l471_47125


namespace NUMINAMATH_CALUDE_pictures_on_front_l471_47179

theorem pictures_on_front (total : ℕ) (on_back : ℕ) (h1 : total = 15) (h2 : on_back = 9) :
  total - on_back = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_front_l471_47179


namespace NUMINAMATH_CALUDE_smallest_perimeter_l471_47110

/-- Represents the side lengths of the squares in the rectangle --/
structure SquareSides where
  a : ℕ
  b : ℕ

/-- Calculates the perimeter of the rectangle given the square sides --/
def rectanglePerimeter (s : SquareSides) : ℕ :=
  2 * ((2 * s.a + 3 * s.b) + (3 * s.a + 4 * s.b))

/-- The theorem stating the smallest possible perimeter --/
theorem smallest_perimeter :
  ∃ (s : SquareSides), 
    (5 * s.a + 2 * s.b = 20 * s.a - 3 * s.b) ∧
    (∀ (t : SquareSides), rectanglePerimeter s ≤ rectanglePerimeter t) ∧
    rectanglePerimeter s = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l471_47110


namespace NUMINAMATH_CALUDE_quadratic_factorization_l471_47102

theorem quadratic_factorization (x y : ℝ) : 5*x^2 + 6*x*y - 8*y^2 = (x + 2*y)*(5*x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l471_47102


namespace NUMINAMATH_CALUDE_distribute_volunteers_count_l471_47140

/-- The number of ways to distribute 5 volunteers into 4 groups -/
def distribute_volunteers : ℕ :=
  Nat.choose 5 2 * Nat.factorial 4

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribute_volunteers_count : distribute_volunteers = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_volunteers_count_l471_47140


namespace NUMINAMATH_CALUDE_max_value_constraint_l471_47135

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  3*x + 4*y + 5*z ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l471_47135


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l471_47149

theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ < 2 ∧ 
    r₁^2 + (2*m - 1)*r₁ + 4 - 2*m = 0 ∧
    r₂^2 + (2*m - 1)*r₂ + 4 - 2*m = 0) →
  m < -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l471_47149


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l471_47117

theorem complex_sum_of_parts (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  (z * Complex.mk 1 2 = 5) → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l471_47117


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_smallest_primes_l471_47174

def smallest_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n % m = 0

theorem largest_four_digit_divisible_by_smallest_primes :
  ∀ n : Nat, n ≤ 9999 → n ≥ 1000 →
  is_divisible_by_all n smallest_primes →
  n ≤ 9240 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_smallest_primes_l471_47174


namespace NUMINAMATH_CALUDE_max_playground_area_l471_47196

/-- Represents a rectangular playground --/
structure Playground where
  length : ℝ
  width : ℝ

/-- The perimeter of the playground is 400 feet --/
def perimeterConstraint (p : Playground) : Prop :=
  2 * p.length + 2 * p.width = 400

/-- The length of the playground is at least 100 feet --/
def lengthConstraint (p : Playground) : Prop :=
  p.length ≥ 100

/-- The width of the playground is at least 50 feet --/
def widthConstraint (p : Playground) : Prop :=
  p.width ≥ 50

/-- The area of the playground --/
def area (p : Playground) : ℝ :=
  p.length * p.width

/-- The maximum area of the playground satisfying all constraints is 10000 square feet --/
theorem max_playground_area :
  ∃ (p : Playground),
    perimeterConstraint p ∧
    lengthConstraint p ∧
    widthConstraint p ∧
    area p = 10000 ∧
    ∀ (q : Playground),
      perimeterConstraint q →
      lengthConstraint q →
      widthConstraint q →
      area q ≤ area p :=
by
  sorry


end NUMINAMATH_CALUDE_max_playground_area_l471_47196


namespace NUMINAMATH_CALUDE_triangle_side_length_l471_47138

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 4 → b = 6 → C = 2 * π / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l471_47138


namespace NUMINAMATH_CALUDE_average_score_is_76_point_8_l471_47134

def class_size : ℕ := 50

def first_group_scores : List ℕ := [90, 85, 88, 92, 80, 94, 89, 91, 84, 87]

def second_group_scores : List ℕ := 
  [85, 80, 83, 87, 75, 89, 84, 86, 79, 82, 77, 74, 81, 78, 70]

def third_group_scores : List ℕ := 
  [40, 62, 58, 70, 72, 68, 64, 66, 74, 76, 60, 78, 80, 82, 84, 86, 88, 61, 63, 65, 67, 69, 71, 73, 75]

def total_score : ℕ := 
  (first_group_scores.sum + second_group_scores.sum + third_group_scores.sum)

theorem average_score_is_76_point_8 :
  (total_score : ℚ) / class_size = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_76_point_8_l471_47134


namespace NUMINAMATH_CALUDE_xy_is_zero_l471_47151

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l471_47151


namespace NUMINAMATH_CALUDE_simplify_expression_l471_47198

theorem simplify_expression (x y : ℝ) (m : ℤ) :
  (x + y) ^ (2 * m + 1) / (x + y) ^ (m - 1) = (x + y) ^ (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l471_47198


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l471_47170

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l471_47170


namespace NUMINAMATH_CALUDE_inequality_range_l471_47152

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l471_47152


namespace NUMINAMATH_CALUDE_flagpole_height_l471_47167

/-- Given a tree and a flagpole with known measurements, calculate the height of the flagpole -/
theorem flagpole_height
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (flagpole_shadow : ℝ)
  (h_tree_height : tree_height = 3.6)
  (h_tree_shadow : tree_shadow = 0.6)
  (h_flagpole_shadow : flagpole_shadow = 1.5) :
  ∃ (flagpole_height : ℝ), flagpole_height = 9 ∧
    tree_height / tree_shadow = flagpole_height / flagpole_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l471_47167


namespace NUMINAMATH_CALUDE_log_simplification_l471_47112

theorem log_simplification (a b c d x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) : 
  Real.log (a^2 / b) + Real.log (b / c^2) + Real.log (c / d) - Real.log (a^2 * y / (d^3 * x)) = 
  Real.log (d^2 * x / y) := by
sorry

end NUMINAMATH_CALUDE_log_simplification_l471_47112


namespace NUMINAMATH_CALUDE_solve_for_c_l471_47141

theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 5) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 27 →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l471_47141


namespace NUMINAMATH_CALUDE_election_margin_theorem_l471_47104

/-- Represents an election with two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  winner_percentage : ℚ

/-- Calculates the margin of victory in an election -/
def margin_of_victory (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the margin of victory for the given election scenario -/
theorem election_margin_theorem (e : Election) 
  (h1 : e.winner_percentage = 65 / 100)
  (h2 : e.winner_votes = 650) :
  margin_of_victory e = 300 := by
sorry

#eval margin_of_victory { total_votes := 1000, winner_votes := 650, winner_percentage := 65 / 100 }

end NUMINAMATH_CALUDE_election_margin_theorem_l471_47104


namespace NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l471_47129

-- Define the propositions A and B
variable (A B : Prop)

-- Define what it means for B to be a sufficient condition for A
def sufficient (B A : Prop) : Prop := B → A

-- Define what it means for A to be a necessary condition for B
def necessary (A B : Prop) : Prop := B → A

-- Theorem: If B is sufficient for A, it doesn't necessarily mean A is necessary for B
theorem sufficient_not_imply_necessary (h : sufficient B A) : 
  ¬ (∀ A B, sufficient B A → necessary A B) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_imply_necessary_l471_47129


namespace NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l471_47133

theorem smallest_x_with_given_remainders :
  ∃ x : ℕ,
    x > 0 ∧
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y ∧
    x = 167 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l471_47133


namespace NUMINAMATH_CALUDE_rectangle_area_l471_47121

/-- A rectangle with length thrice its breadth and perimeter 88 meters has an area of 363 square meters. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 88 → l * b = 363 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l471_47121


namespace NUMINAMATH_CALUDE_five_point_thirty_five_million_equals_scientific_notation_l471_47137

-- Define 5.35 million
def five_point_thirty_five_million : ℝ := 5.35 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 5.35 * (10 ^ 6)

-- Theorem to prove equality
theorem five_point_thirty_five_million_equals_scientific_notation : 
  five_point_thirty_five_million = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_five_point_thirty_five_million_equals_scientific_notation_l471_47137


namespace NUMINAMATH_CALUDE_election_winner_percentage_l471_47113

theorem election_winner_percentage :
  ∀ (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ),
    winner_votes = 806 →
    margin = 312 →
    total_votes = winner_votes + (winner_votes - margin) →
    (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l471_47113


namespace NUMINAMATH_CALUDE_negation_of_implication_l471_47188

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a + 1 > b) ↔ (a ≤ b → a + 1 ≤ b) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l471_47188


namespace NUMINAMATH_CALUDE_common_root_equations_l471_47169

theorem common_root_equations (p : ℤ) (x : ℚ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

#check common_root_equations

end NUMINAMATH_CALUDE_common_root_equations_l471_47169


namespace NUMINAMATH_CALUDE_roots_sum_square_value_l471_47148

theorem roots_sum_square_value (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → n^2 + 3*n - 1 = 0 → m^2 + 4*m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_square_value_l471_47148


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_prob_l471_47194

/-- Represents a set of symmetrical dice -/
structure DiceSet where
  /-- The number of dice in the set -/
  num_dice : ℕ
  /-- The maximum number of points on each die -/
  max_points : ℕ
  /-- The probability of getting a sum of 2022 -/
  prob_2022 : ℝ
  /-- Assumption that the probability is positive -/
  pos_prob : prob_2022 > 0
  /-- Assumption that 2022 is achievable with these dice -/
  sum_2022 : num_dice * max_points = 2022

/-- 
Theorem: Given a set of symmetrical dice where a sum of 2022 is possible 
with probability p > 0, the smallest sum possible with the same probability p is 337.
-/
theorem smallest_sum_with_same_prob (d : DiceSet) : 
  d.num_dice = 337 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_prob_l471_47194


namespace NUMINAMATH_CALUDE_circle_area_when_radius_equals_three_times_reciprocal_circumference_l471_47186

theorem circle_area_when_radius_equals_three_times_reciprocal_circumference :
  ∀ r : ℝ, r > 0 →
  (3 * (1 / (2 * π * r)) = r) →
  (π * r^2 = 3/2) := by
sorry

end NUMINAMATH_CALUDE_circle_area_when_radius_equals_three_times_reciprocal_circumference_l471_47186


namespace NUMINAMATH_CALUDE_complex_power_problem_l471_47146

theorem complex_power_problem : (((1 - Complex.I) / (1 + Complex.I)) ^ 10 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l471_47146


namespace NUMINAMATH_CALUDE_coin_collection_values_l471_47139

/-- Represents a collection of coins -/
structure CoinCollection where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Defines the conditions for the coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.quarters = c.nickels / 2 ∧ c.half_dollars = 2 * c.quarters

/-- Calculates the total value of the coin collection in cents -/
def total_value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars

/-- Theorem stating that there exist valid collections with total values of $67.50 and $135.00 -/
theorem coin_collection_values : 
  ∃ (c1 c2 : CoinCollection), 
    valid_collection c1 ∧ valid_collection c2 ∧ 
    total_value c1 = 6750 ∧ total_value c2 = 13500 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_values_l471_47139


namespace NUMINAMATH_CALUDE_problem_statement_l471_47132

theorem problem_statement (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l471_47132


namespace NUMINAMATH_CALUDE_brets_nap_time_l471_47127

/-- Represents the duration of Bret's train journey and activities --/
structure TrainJourney where
  totalTime : ℝ
  readingTime : ℝ
  eatingTime : ℝ
  movieTime : ℝ
  chattingTime : ℝ
  browsingTime : ℝ
  waitingTime : ℝ
  workingTime : ℝ

/-- Calculates the remaining time for napping given a TrainJourney --/
def remainingTimeForNap (journey : TrainJourney) : ℝ :=
  journey.totalTime - (journey.readingTime + journey.eatingTime + journey.movieTime + 
    journey.chattingTime + journey.browsingTime + journey.waitingTime + journey.workingTime)

/-- Theorem stating that for Bret's specific journey, the remaining time for napping is 4.75 hours --/
theorem brets_nap_time (journey : TrainJourney) 
  (h1 : journey.totalTime = 15)
  (h2 : journey.readingTime = 2)
  (h3 : journey.eatingTime = 1)
  (h4 : journey.movieTime = 3)
  (h5 : journey.chattingTime = 1)
  (h6 : journey.browsingTime = 0.75)
  (h7 : journey.waitingTime = 0.5)
  (h8 : journey.workingTime = 2) :
  remainingTimeForNap journey = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_brets_nap_time_l471_47127


namespace NUMINAMATH_CALUDE_star_A_B_equals_result_l471_47116

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | y ≥ 1}

-- Define the operation *
def star (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem star_A_B_equals_result : star A B = {x | (0 ≤ x ∧ x < 1) ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_star_A_B_equals_result_l471_47116


namespace NUMINAMATH_CALUDE_inverse_statement_is_false_l471_47115

-- Define the set S of all non-zero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Define what it means for an element to be an inverse under *
def is_inverse (a b : ℝ) : Prop := star a b = 1/3 ∧ star b a = 1/3

-- The theorem to be proved
theorem inverse_statement_is_false :
  ∀ a ∈ S, ¬(is_inverse a (1/(3*a))) := by
  sorry

end NUMINAMATH_CALUDE_inverse_statement_is_false_l471_47115
