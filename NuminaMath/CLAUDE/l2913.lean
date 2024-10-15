import Mathlib

namespace NUMINAMATH_CALUDE_basketball_game_scores_l2913_291377

/-- Represents the scores of a team in a four-quarter basketball game -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the given scores form an increasing geometric sequence -/
def isGeometricSequence (scores : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    scores.q2 = scores.q1 * r ∧
    scores.q3 = scores.q2 * r ∧
    scores.q4 = scores.q3 * r

/-- Checks if the given scores form an increasing arithmetic sequence -/
def isArithmeticSequence (scores : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

/-- The main theorem representing the basketball game scenario -/
theorem basketball_game_scores 
  (tigers lions : TeamScores)
  (h1 : tigers.q1 = lions.q1)  -- Tied at the end of first quarter
  (h2 : isGeometricSequence tigers)
  (h3 : isArithmeticSequence lions)
  (h4 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 = 
        lions.q1 + lions.q2 + lions.q3 + lions.q4 + 4)  -- Tigers won by 4 points
  (h5 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 120)  -- Max score constraint for Tigers
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 120)  -- Max score constraint for Lions
  : tigers.q1 + tigers.q2 + lions.q1 + lions.q2 = 23 :=
by
  sorry  -- Proof omitted as per instructions


end NUMINAMATH_CALUDE_basketball_game_scores_l2913_291377


namespace NUMINAMATH_CALUDE_f_max_value_l2913_291343

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l2913_291343


namespace NUMINAMATH_CALUDE_crackers_sales_total_l2913_291353

theorem crackers_sales_total (friday_sales : ℕ) 
  (h1 : friday_sales = 30) 
  (h2 : ∃ saturday_sales : ℕ, saturday_sales = 2 * friday_sales) 
  (h3 : ∃ sunday_sales : ℕ, sunday_sales = saturday_sales - 15) : 
  friday_sales + 2 * friday_sales + (2 * friday_sales - 15) = 135 := by
  sorry

end NUMINAMATH_CALUDE_crackers_sales_total_l2913_291353


namespace NUMINAMATH_CALUDE_inequality_proof_l2913_291346

theorem inequality_proof (a b c d e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) :
  (a/b)^4 + (b/c)^4 + (c/d)^4 + (d/e)^4 + (e/a)^4 ≥ a/b + b/c + c/d + d/e + e/a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2913_291346


namespace NUMINAMATH_CALUDE_distance_between_cities_l2913_291334

/-- The distance between city A and city B in miles -/
def distance : ℝ := sorry

/-- The time taken for the trip from A to B in hours -/
def time_AB : ℝ := 3

/-- The time taken for the trip from B to A in hours -/
def time_BA : ℝ := 2.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The speed for the round trip if time was saved, in miles per hour -/
def speed_with_savings : ℝ := 80

theorem distance_between_cities :
  distance = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2913_291334


namespace NUMINAMATH_CALUDE_total_balls_l2913_291362

/-- Given 2 boxes, each containing 3 balls, the total number of balls is 6. -/
theorem total_balls (num_boxes : ℕ) (balls_per_box : ℕ) (h1 : num_boxes = 2) (h2 : balls_per_box = 3) :
  num_boxes * balls_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l2913_291362


namespace NUMINAMATH_CALUDE_sales_price_ratio_l2913_291340

/-- Proves the ratio of percent increase in units sold to combined percent decrease in price -/
theorem sales_price_ratio (P : ℝ) (U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let price_decrease := 0.20
  let additional_discount := 0.10
  let new_price := P * (1 - price_decrease)
  let new_units := U / (1 - price_decrease)
  let final_price := new_price * (1 - additional_discount)
  let percent_increase_units := (new_units - U) / U
  let percent_decrease_price := (P - final_price) / P
  (percent_increase_units / percent_decrease_price) = 1 / 1.12 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_price_ratio_l2913_291340


namespace NUMINAMATH_CALUDE_angle_side_ratio_angle_sine_relation_two_solutions_l2913_291348

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem 1
theorem angle_side_ratio (t : Triangle) :
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3 →
  t.a / t.b = 1 / Real.sqrt 3 ∧ t.b / t.c = Real.sqrt 3 / 2 := by sorry

-- Theorem 2
theorem angle_sine_relation (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by sorry

-- Theorem 3
theorem two_solutions (t : Triangle) :
  t.A = π / 6 ∧ t.a = 3 ∧ t.b = 4 →
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧
    t1.A = t.A ∧ t1.a = t.a ∧ t1.b = t.b ∧
    t2.A = t.A ∧ t2.a = t.a ∧ t2.b = t.b := by sorry

end NUMINAMATH_CALUDE_angle_side_ratio_angle_sine_relation_two_solutions_l2913_291348


namespace NUMINAMATH_CALUDE_max_page_number_l2913_291331

/-- The number of '2's available -/
def available_twos : ℕ := 34

/-- The number of '2's used in numbers from 1 to 99 -/
def twos_in_1_to_99 : ℕ := 19

/-- The number of '2's used in numbers from 100 to 199 -/
def twos_in_100_to_199 : ℕ := 10

/-- The highest page number that can be reached with the available '2's -/
def highest_page_number : ℕ := 199

theorem max_page_number :
  available_twos = twos_in_1_to_99 + twos_in_100_to_199 + 5 ∧
  highest_page_number = 199 :=
sorry

end NUMINAMATH_CALUDE_max_page_number_l2913_291331


namespace NUMINAMATH_CALUDE_hotel_problem_l2913_291337

theorem hotel_problem (n : ℕ) : n = 9 :=
  let total_spent : ℚ := 29.25
  let standard_meal_cost : ℚ := 3
  let standard_meal_count : ℕ := 8
  let extra_cost : ℚ := 2

  have h1 : n > 0 := by sorry
  have h2 : (n : ℚ) * (total_spent / n) = total_spent := by sorry
  have h3 : standard_meal_count * standard_meal_cost + (total_spent / n + extra_cost) = total_spent := by sorry

  sorry

end NUMINAMATH_CALUDE_hotel_problem_l2913_291337


namespace NUMINAMATH_CALUDE_reservoir_D_largest_l2913_291310

-- Define the initial amount of water (same for all reservoirs)
variable (a : ℝ)

-- Define the final amounts of water in each reservoir
def final_amount_A : ℝ := a * (1 + 0.10) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

-- Theorem stating that Reservoir D has the largest amount of water
theorem reservoir_D_largest (a : ℝ) (h : a > 0) : 
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end NUMINAMATH_CALUDE_reservoir_D_largest_l2913_291310


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l2913_291385

theorem min_value_of_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y : ℝ, y = 5 * x^2 - 10 * x + 14 → y ≥ min_y :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l2913_291385


namespace NUMINAMATH_CALUDE_hemisphere_with_disk_surface_area_l2913_291311

/-- Given a hemisphere with base area 144π and an attached circular disk of radius 5,
    the total exposed surface area is 313π. -/
theorem hemisphere_with_disk_surface_area :
  ∀ (r : ℝ) (disk_radius : ℝ),
    r > 0 →
    disk_radius > 0 →
    π * r^2 = 144 * π →
    disk_radius = 5 →
    2 * π * r^2 + π * disk_radius^2 = 313 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_with_disk_surface_area_l2913_291311


namespace NUMINAMATH_CALUDE_sum_of_integers_l2913_291306

theorem sum_of_integers : (-9) + 18 + 2 + (-1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2913_291306


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2913_291333

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  neither = 9 →
  football + tennis - (total - neither) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2913_291333


namespace NUMINAMATH_CALUDE_ribbon_parts_l2913_291327

theorem ribbon_parts (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) :
  total_length = 30 ∧ used_parts = 4 ∧ unused_length = 10 →
  ∃ (n : ℕ), n > 0 ∧ n * (total_length - unused_length) / used_parts = total_length / n :=
by sorry

end NUMINAMATH_CALUDE_ribbon_parts_l2913_291327


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2913_291375

/-- The range of m for which the line y = kx + 2 (k ∈ ℝ) always intersects the ellipse x² + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), ∃ (x y : ℝ), 
    y = k * x + 2 ∧ 
    x^2 + y^2 / m = 1 ↔ 
    m ∈ Set.Ici (4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2913_291375


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2913_291314

def calculate_selling_price (purchase_price repair_cost transport_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := (total_cost * profit_percentage) / 100
  total_cost + profit

theorem selling_price_calculation :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2913_291314


namespace NUMINAMATH_CALUDE_triangle_side_length_l2913_291315

theorem triangle_side_length (a b : ℝ) (C : ℝ) (S : ℝ) : 
  a = 3 * Real.sqrt 2 →
  Real.cos C = 1 / 3 →
  S = 4 * Real.sqrt 3 →
  S = 1 / 2 * a * b * Real.sin C →
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2913_291315


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2913_291366

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2913_291366


namespace NUMINAMATH_CALUDE_certain_number_problem_l2913_291389

theorem certain_number_problem : 
  ∃ x : ℝ, (0.1 * x + 0.15 * 50 = 10.5) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2913_291389


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2913_291347

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2913_291347


namespace NUMINAMATH_CALUDE_sector_central_angle_l2913_291349

theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 3 * π / 8) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2913_291349


namespace NUMINAMATH_CALUDE_train_passengers_l2913_291364

theorem train_passengers (initial : ℕ) 
  (first_off first_on second_off second_on final : ℕ) : 
  first_off = 29 → 
  first_on = 17 → 
  second_off = 27 → 
  second_on = 35 → 
  final = 116 → 
  initial = 120 → 
  initial - first_off + first_on - second_off + second_on = final :=
by sorry

end NUMINAMATH_CALUDE_train_passengers_l2913_291364


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l2913_291397

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The theorem statement -/
theorem fixed_point_coordinates :
  (∃ M : Point, ∀ k : ℝ, lineEquation k M) →
  ∃ M : Point, M.x = -1 ∧ M.y = 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l2913_291397


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l2913_291399

def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

def in_first_quadrant (θ : Real) : Prop :=
  0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2

def in_third_quadrant (θ : Real) : Prop :=
  Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2

theorem angle_in_first_or_third_quadrant (α : Real) (k : Int) 
  (h_acute : is_acute α) :
  in_first_quadrant (k * Real.pi + α) ∨ in_third_quadrant (k * Real.pi + α) := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l2913_291399


namespace NUMINAMATH_CALUDE_rafael_earnings_l2913_291359

def hours_monday : ℕ := 10
def hours_tuesday : ℕ := 8
def hours_left : ℕ := 20
def hourly_rate : ℕ := 20

theorem rafael_earnings : 
  (hours_monday + hours_tuesday + hours_left) * hourly_rate = 760 := by
  sorry

end NUMINAMATH_CALUDE_rafael_earnings_l2913_291359


namespace NUMINAMATH_CALUDE_daves_phone_files_l2913_291383

theorem daves_phone_files :
  let initial_apps : ℕ := 15
  let initial_files : ℕ := 24
  let final_apps : ℕ := 21
  let app_file_difference : ℕ := 17
  let files_left : ℕ := final_apps - app_file_difference
  files_left = 4 := by sorry

end NUMINAMATH_CALUDE_daves_phone_files_l2913_291383


namespace NUMINAMATH_CALUDE_michael_remaining_yards_l2913_291381

/-- Represents the length of an ultra-marathon in miles and yards -/
structure UltraMarathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def yards_per_mile : ℕ := 1760

def ultra_marathon : UltraMarathon := ⟨50, 800⟩

def michael_marathons : ℕ := 5

theorem michael_remaining_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    (michael_marathons * ultra_marathon.miles * yards_per_mile + 
     michael_marathons * ultra_marathon.yards) = 
    (m * yards_per_mile + y) ∧
    y = 480 := by
  sorry

end NUMINAMATH_CALUDE_michael_remaining_yards_l2913_291381


namespace NUMINAMATH_CALUDE_savings_fraction_is_one_seventh_l2913_291378

/-- A worker's monthly financial situation -/
structure WorkerFinances where
  P : ℝ  -- Monthly take-home pay
  S : ℝ  -- Fraction of take-home pay saved
  E : ℝ  -- Fraction of take-home pay for expenses
  T : ℝ  -- Monthly taxes
  h_positive_pay : 0 < P
  h_valid_fractions : 0 ≤ S ∧ 0 ≤ E ∧ S + E ≤ 1

/-- The theorem stating that if total yearly savings equals twice the monthly amount not saved,
    then the savings fraction is 1/7 -/
theorem savings_fraction_is_one_seventh (w : WorkerFinances) 
    (h_savings_equality : 12 * w.P * w.S = 2 * w.P * (1 - w.S)) : 
    w.S = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_savings_fraction_is_one_seventh_l2913_291378


namespace NUMINAMATH_CALUDE_symmetric_function_a_value_inequality_condition_a_range_l2913_291326

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + 2*a

-- Theorem 1
theorem symmetric_function_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = f a (3 - x)) → a = -3 := by sorry

-- Theorem 2
theorem inequality_condition_a_range (a : ℝ) :
  (∃ x : ℝ, f a x ≤ -|2*x - 1| + a) → a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_a_value_inequality_condition_a_range_l2913_291326


namespace NUMINAMATH_CALUDE_rabbit_run_time_l2913_291317

/-- The time taken for a rabbit to run from the end to the front of a moving line and back -/
theorem rabbit_run_time (line_length : ℝ) (line_speed : ℝ) (rabbit_speed : ℝ) : 
  line_length = 40 →
  line_speed = 3 →
  rabbit_speed = 5 →
  (line_length / (rabbit_speed - line_speed)) + (line_length / (rabbit_speed + line_speed)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_run_time_l2913_291317


namespace NUMINAMATH_CALUDE_oranges_picked_total_l2913_291329

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l2913_291329


namespace NUMINAMATH_CALUDE_container_weight_container_weight_proof_l2913_291320

/-- Given a container with weights p and q when three-quarters and one-third full respectively,
    the total weight when completely full is (8p - 3q) / 5 -/
theorem container_weight (p q : ℝ) : ℝ :=
  let three_quarters_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 3 * q) / 5
  full_weight

/-- Proof of the container weight theorem -/
theorem container_weight_proof (p q : ℝ) :
  container_weight p q = (8 * p - 3 * q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_container_weight_container_weight_proof_l2913_291320


namespace NUMINAMATH_CALUDE_fisherman_catch_l2913_291391

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (salmon : ℕ) (pike : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  blue_gill = 2 * bass →
  salmon = bass + bass / 3 →
  pike = (bass + trout + blue_gill + salmon) / 5 →
  bass + trout + blue_gill + salmon + pike = 138 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l2913_291391


namespace NUMINAMATH_CALUDE_downstream_distance_proof_l2913_291322

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distance_downstream (boat_speed stream_speed time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 7 hours, with a speed of 24 km/hr in still water
    and a stream speed of 4 km/hr, travels 196 km -/
theorem downstream_distance_proof :
  distance_downstream 24 4 7 = 196 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_proof_l2913_291322


namespace NUMINAMATH_CALUDE_initial_number_proof_l2913_291309

theorem initial_number_proof : ∃ (n : ℕ), n = 427398 ∧ 
  (∃ (k : ℕ), n - 6 = 14 * k) ∧ 
  (∀ (m : ℕ), m < 6 → ¬∃ (j : ℕ), n - m = 14 * j) :=
by sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2913_291309


namespace NUMINAMATH_CALUDE_only_coin_toss_is_random_l2913_291396

-- Define the type for events
inductive Event
  | CoinToss : Event
  | ChargeAttraction : Event
  | WaterFreeze : Event

-- Define a predicate for random events
def is_random_event : Event → Prop :=
  fun e => match e with
    | Event.CoinToss => True
    | _ => False

-- Theorem statement
theorem only_coin_toss_is_random :
  (is_random_event Event.CoinToss) ∧
  (¬ is_random_event Event.ChargeAttraction) ∧
  (¬ is_random_event Event.WaterFreeze) :=
by sorry

end NUMINAMATH_CALUDE_only_coin_toss_is_random_l2913_291396


namespace NUMINAMATH_CALUDE_cone_base_radius_l2913_291360

/-- A cone formed by a semicircle with radius 2 cm has a base circle with radius 1 cm -/
theorem cone_base_radius (r : ℝ) (h : r = 2) : 
  (2 * Real.pi * r / 2) / (2 * Real.pi) = 1 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2913_291360


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2913_291358

def ellipse_equation (e : ℝ) (l : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 / 32 + y^2 / 16 = 1

theorem ellipse_standard_equation (e l : ℝ) 
  (h1 : e = Real.sqrt 2 / 2) 
  (h2 : l = 8) : 
  ellipse_equation e l = fun (x, y) => x^2 / 32 + y^2 / 16 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2913_291358


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2913_291373

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(53*b)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2913_291373


namespace NUMINAMATH_CALUDE_triangle_existence_l2913_291321

theorem triangle_existence (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a > c) ∧ (a > b) ∧ (a < b + c) := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l2913_291321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2913_291324

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2913_291324


namespace NUMINAMATH_CALUDE_arun_age_is_sixty_l2913_291336

/-- Given the ages of Arun, Gokul, and Madan, prove that Arun's age is 60 years. -/
theorem arun_age_is_sixty (arun_age gokul_age madan_age : ℕ) : 
  ((arun_age - 6) / 18 = gokul_age) →
  (gokul_age = madan_age - 2) →
  (madan_age = 5) →
  arun_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_is_sixty_l2913_291336


namespace NUMINAMATH_CALUDE_range_of_a_l2913_291342

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2913_291342


namespace NUMINAMATH_CALUDE_complex_magnitude_and_argument_l2913_291302

theorem complex_magnitude_and_argument :
  ∃ (t : ℝ), t > 0 ∧ 
  (Complex.abs (9 + t * Complex.I) = 13 ↔ t = Real.sqrt 88) ∧
  Complex.arg (9 + t * Complex.I) ≠ π / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_and_argument_l2913_291302


namespace NUMINAMATH_CALUDE_marble_ratio_l2913_291330

theorem marble_ratio (total : ℕ) (white : ℕ) (removed : ℕ) (remaining : ℕ)
  (h1 : total = 50)
  (h2 : white = 20)
  (h3 : removed = 2 * (white - (total - white - (total - removed - white))))
  (h4 : remaining = 40)
  (h5 : total = remaining + removed) :
  (total - removed - white) = (total - white - (total - removed - white)) :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_l2913_291330


namespace NUMINAMATH_CALUDE_battery_usage_difference_l2913_291335

theorem battery_usage_difference (flashlights remote_controllers wall_clock wireless_mouse toys : ℝ) 
  (h1 : flashlights = 3.5)
  (h2 : remote_controllers = 7.25)
  (h3 : wall_clock = 4.8)
  (h4 : wireless_mouse = 3.4)
  (h5 : toys = 15.75) :
  toys - (flashlights + remote_controllers + wall_clock + wireless_mouse) = -3.2 := by
  sorry

end NUMINAMATH_CALUDE_battery_usage_difference_l2913_291335


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2913_291305

theorem solution_satisfies_system :
  let solutions : List (Int × Int) := [(-3, -1), (-1, -3), (1, 3), (3, 1)]
  ∀ (x y : Int), (x, y) ∈ solutions →
    (x^2 - x*y + y^2 = 7 ∧ x^4 + x^2*y^2 + y^4 = 91) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2913_291305


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2913_291356

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = Complex.abs (1/2 - Complex.I * (Real.sqrt 3 / 2)) →
  z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2913_291356


namespace NUMINAMATH_CALUDE_min_quadratic_expression_l2913_291352

theorem min_quadratic_expression :
  ∃ (x : ℝ), ∀ (y : ℝ), 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_min_quadratic_expression_l2913_291352


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2913_291338

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = 1 / (a + 3)) ↔ a ≠ -3 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2913_291338


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2913_291332

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2913_291332


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2913_291350

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 11 = 12 / 77 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2913_291350


namespace NUMINAMATH_CALUDE_sum_of_sequences_l2913_291390

def sequence1 : List ℕ := [1, 12, 23, 34, 45]
def sequence2 : List ℕ := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum) = 265 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l2913_291390


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2913_291312

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2913_291312


namespace NUMINAMATH_CALUDE_exists_special_number_l2913_291384

/-- A function that checks if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that swaps two digits in a natural number at given positions -/
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

theorem exists_special_number :
  ∃ (N : ℕ),
    N % 2020 = 0 ∧
    has_distinct_digits N ∧
    num_digits N = 6 ∧
    ∀ (i j : ℕ), i ≠ j → (swap_digits N i j) % 2020 ≠ 0 ∧
    ∀ (M : ℕ), M % 2020 = 0 → has_distinct_digits M →
      (∀ (i j : ℕ), i ≠ j → (swap_digits M i j) % 2020 ≠ 0) →
      num_digits M ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_exists_special_number_l2913_291384


namespace NUMINAMATH_CALUDE_log4_20_approximation_l2913_291376

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.300
def log10_5_approx : ℝ := 0.699

-- Define the target approximation
def target_approx : ℚ := 13/6

-- State the theorem
theorem log4_20_approximation : 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x y : ℝ), 
    |x - log10_2_approx| < δ ∧ 
    |y - log10_5_approx| < δ → 
    |((1 + x) / (2 * x)) - target_approx| < ε) :=
sorry

end NUMINAMATH_CALUDE_log4_20_approximation_l2913_291376


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l2913_291304

theorem candy_mixture_cost (candy1_weight : ℝ) (candy1_cost : ℝ) (total_weight : ℝ) (mixture_cost : ℝ) :
  candy1_weight = 30 →
  candy1_cost = 8 →
  total_weight = 90 →
  mixture_cost = 6 →
  ∃ candy2_cost : ℝ,
    candy2_cost = 5 ∧
    candy1_weight * candy1_cost + (total_weight - candy1_weight) * candy2_cost = total_weight * mixture_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l2913_291304


namespace NUMINAMATH_CALUDE_cubic_equation_third_root_l2913_291365

theorem cubic_equation_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (2*b - 4*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (2*b - 4*a) * 4 + (10 - a) = 0)
  : ∃ (x : ℚ), x = -62/19 ∧ 
    a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_third_root_l2913_291365


namespace NUMINAMATH_CALUDE_seventy_five_days_after_wednesday_is_monday_l2913_291341

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def days_after (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (days_after start m)

theorem seventy_five_days_after_wednesday_is_monday :
  days_after DayOfWeek.Wednesday 75 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_seventy_five_days_after_wednesday_is_monday_l2913_291341


namespace NUMINAMATH_CALUDE_max_value_theorem_l2913_291388

theorem max_value_theorem (x y : ℝ) (h : x^2/4 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y/(x + 2*y - 2) → z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2913_291388


namespace NUMINAMATH_CALUDE_problem_solution_l2913_291387

def set_product (A B : Set ℝ) : Set ℝ :=
  {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

def A : Set ℝ := {0, 2}
def B : Set ℝ := {1, 3}
def C : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem problem_solution :
  (set_product A B) ∩ (set_product B C) = {2, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2913_291387


namespace NUMINAMATH_CALUDE_max_value_g_on_interval_l2913_291325

def g (x : ℝ) : ℝ := x * (x^2 - 1)

theorem max_value_g_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 1 → g y ≤ g x ∧
  g x = 0 :=
sorry

end NUMINAMATH_CALUDE_max_value_g_on_interval_l2913_291325


namespace NUMINAMATH_CALUDE_soccer_goals_product_l2913_291303

def first_ten_games : List Nat := [2, 5, 3, 6, 2, 4, 2, 5, 1, 3]

def total_first_ten : Nat := first_ten_games.sum

theorem soccer_goals_product (g11 g12 : Nat) : 
  g11 < 8 → 
  g12 < 8 → 
  (total_first_ten + g11) % 11 = 0 → 
  (total_first_ten + g11 + g12) % 12 = 0 → 
  g11 * g12 = 49 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goals_product_l2913_291303


namespace NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l2913_291300

theorem sqrt_450_equals_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_equals_15_sqrt_2_l2913_291300


namespace NUMINAMATH_CALUDE_train_passing_jogger_l2913_291372

/-- Theorem: Train passing jogger
  Given:
  - Jogger's speed: 9 kmph
  - Train's speed: 45 kmph
  - Train's length: 120 meters
  - Initial distance between jogger and train engine: 240 meters
  Prove: The time for the train to pass the jogger is 36 seconds
-/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_distance : Real) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 120) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_l2913_291372


namespace NUMINAMATH_CALUDE_triangle_left_side_value_l2913_291374

/-- Given a triangle with sides L, R, and B satisfying certain conditions, prove that L = 12 -/
theorem triangle_left_side_value (L R B : ℝ) 
  (h1 : L + R + B = 50)
  (h2 : R = L + 2)
  (h3 : B = 24) : 
  L = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_left_side_value_l2913_291374


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2913_291313

/-- Given two points P and Q symmetric about the x-axis, prove that a + b = -1 -/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ P Q : ℝ × ℝ, 
    P = (a - 1, 5) ∧ 
    Q = (2, b - 1) ∧ 
    P.1 = Q.1 ∧ 
    P.2 = -Q.2) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2913_291313


namespace NUMINAMATH_CALUDE_lice_check_time_l2913_291301

/-- The total number of hours required for lice checks -/
def total_hours (kindergarteners first_graders second_graders third_graders : ℕ) 
  (minutes_per_check : ℕ) : ℚ :=
  (kindergarteners + first_graders + second_graders + third_graders) * minutes_per_check / 60

/-- Theorem stating that the total time for lice checks is 3 hours -/
theorem lice_check_time : 
  total_hours 26 19 20 25 2 = 3 := by sorry

end NUMINAMATH_CALUDE_lice_check_time_l2913_291301


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l2913_291395

theorem floor_sum_inequality (x y : ℝ) :
  (⌊x⌋ : ℝ) + ⌊y⌋ ≤ ⌊x + y⌋ ∧ ⌊x + y⌋ ≤ (⌊x⌋ : ℝ) + ⌊y⌋ + 1 ∧
  ((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∨ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) ∧
  ¬((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∧ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) :=
by sorry


end NUMINAMATH_CALUDE_floor_sum_inequality_l2913_291395


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l2913_291361

def john_scores : List ℕ := [85, 88, 90, 92, 83]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (score : ℕ) : 
  (john_scores.sum + score) / num_quizzes = target_mean ↔ score = 102 := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l2913_291361


namespace NUMINAMATH_CALUDE_inflection_point_is_center_of_symmetry_l2913_291367

/-- Represents a cubic function of the form ax³ + bx² + cx + d -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_nonzero : a ≠ 0

/-- The given cubic function x³ - 3x² + 3x -/
def f : CubicFunction := {
  a := 1
  b := -3
  c := 3
  d := 0
  a_nonzero := by norm_num
}

/-- Evaluates a cubic function at a given x -/
def evaluate (f : CubicFunction) (x : ℝ) : ℝ :=
  f.a * x^3 + f.b * x^2 + f.c * x + f.d

/-- Computes the second derivative of a cubic function -/
def secondDerivative (f : CubicFunction) (x : ℝ) : ℝ :=
  6 * f.a * x + 2 * f.b

/-- An inflection point of a cubic function -/
structure InflectionPoint (f : CubicFunction) where
  x : ℝ
  y : ℝ
  is_inflection : secondDerivative f x = 0
  on_curve : y = evaluate f x

theorem inflection_point_is_center_of_symmetry :
  ∃ (p : InflectionPoint f), p.x = 1 ∧ p.y = 1 := by sorry

end NUMINAMATH_CALUDE_inflection_point_is_center_of_symmetry_l2913_291367


namespace NUMINAMATH_CALUDE_greatest_third_side_length_l2913_291368

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 10 cm. -/
theorem greatest_third_side_length : ℕ :=
  let a : ℝ := 7
  let b : ℝ := 10
  let c : ℝ := 16
  have triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b := by sorry
  have c_less_than_sum : c < a + b := by sorry
  have c_greatest_integer : ∀ n : ℕ, (n : ℝ) > c → (n : ℝ) ≥ a + b := by sorry
  16


end NUMINAMATH_CALUDE_greatest_third_side_length_l2913_291368


namespace NUMINAMATH_CALUDE_first_day_charge_l2913_291398

/-- Represents the charge and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℝ
  day2_charge : ℝ
  day3_charge : ℝ
  attendance_ratio : Fin 3 → ℝ
  average_charge : ℝ

/-- Theorem stating the charge on the first day given the show data -/
theorem first_day_charge (s : ShowData)
  (h1 : s.day2_charge = 7.5)
  (h2 : s.day3_charge = 2.5)
  (h3 : s.attendance_ratio 0 = 2)
  (h4 : s.attendance_ratio 1 = 5)
  (h5 : s.attendance_ratio 2 = 13)
  (h6 : s.average_charge = 5)
  (h7 : (s.attendance_ratio 0 * s.day1_charge + 
         s.attendance_ratio 1 * s.day2_charge + 
         s.attendance_ratio 2 * s.day3_charge) / 
        (s.attendance_ratio 0 + s.attendance_ratio 1 + s.attendance_ratio 2) = s.average_charge) :
  s.day1_charge = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_day_charge_l2913_291398


namespace NUMINAMATH_CALUDE_remainder_8673_mod_7_l2913_291323

theorem remainder_8673_mod_7 : 8673 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_8673_mod_7_l2913_291323


namespace NUMINAMATH_CALUDE_student_number_problem_l2913_291379

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2913_291379


namespace NUMINAMATH_CALUDE_twice_product_of_sum_and_difference_l2913_291393

theorem twice_product_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 80) 
  (diff_eq : x - y = 10) : 
  2 * x * y = 3150 := by
sorry

end NUMINAMATH_CALUDE_twice_product_of_sum_and_difference_l2913_291393


namespace NUMINAMATH_CALUDE_function_equality_implies_n_value_l2913_291345

/-- The function f(x) = 2x^2 - 3x + n -/
def f (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + n

/-- The function g(x) = 2x^2 - 3x + 5n -/
def g (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + 5 * n

/-- Theorem stating that if 3f(3) = 2g(3), then n = 9/7 -/
theorem function_equality_implies_n_value :
  ∀ n : ℚ, 3 * (f n 3) = 2 * (g n 3) → n = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_n_value_l2913_291345


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2913_291316

/-- Given a rectangle ABCD and a square EFGH, if the rectangle shares 60% of its area with the square,
    and the square shares 30% of its area with the rectangle, then the ratio of the rectangle's length
    to its width is 8. -/
theorem rectangle_square_overlap_ratio :
  ∀ (rect_area square_area overlap_area : ℝ) (rect_length rect_width : ℝ),
    rect_area > 0 →
    square_area > 0 →
    overlap_area > 0 →
    rect_length > 0 →
    rect_width > 0 →
    rect_area = rect_length * rect_width →
    overlap_area = 0.6 * rect_area →
    overlap_area = 0.3 * square_area →
    rect_length / rect_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l2913_291316


namespace NUMINAMATH_CALUDE_intersection_point_y_axis_l2913_291386

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem intersection_point_y_axis :
  ∃ (y : ℝ), f 0 = y ∧ (0, y) = (0, -2) := by sorry

end NUMINAMATH_CALUDE_intersection_point_y_axis_l2913_291386


namespace NUMINAMATH_CALUDE_correct_ages_unique_solution_l2913_291339

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  brother : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.father + 15 = 3 * (ages.father - 25)) ∧
  (ages.father + 15 = 2 * (ages.son + 15)) ∧
  (ages.brother = (ages.father + 15) / 2 + 7)

/-- Theorem stating that the ages 45, 15, and 37 satisfy the problem conditions -/
theorem correct_ages : satisfiesConditions { father := 45, son := 15, brother := 37 } := by
  sorry

/-- Theorem stating the uniqueness of the solution -/
theorem unique_solution (ages : FamilyAges) :
  satisfiesConditions ages → ages = { father := 45, son := 15, brother := 37 } := by
  sorry

end NUMINAMATH_CALUDE_correct_ages_unique_solution_l2913_291339


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2913_291370

theorem cos_sum_of_complex_exponentials (α β : ℝ) : 
  Complex.exp (Complex.I * α) = (4:ℝ)/5 + Complex.I * (3:ℝ)/5 →
  Complex.exp (Complex.I * β) = -(5:ℝ)/13 + Complex.I * (12:ℝ)/13 →
  Real.cos (α + β) = -(7:ℝ)/13 := by sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2913_291370


namespace NUMINAMATH_CALUDE_log_expression_equals_four_l2913_291319

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_four :
  4 * log10 2 + 3 * log10 5 - log10 (1/5) = 4 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_four_l2913_291319


namespace NUMINAMATH_CALUDE_line_relations_l2913_291363

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle_of_inclination : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_relations (l1 l2 : Line) (h_distinct : l1 ≠ l2) :
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle_of_inclination = l2.angle_of_inclination) ∧
  (l1.angle_of_inclination = l2.angle_of_inclination → parallel l1 l2) := by
  sorry

end NUMINAMATH_CALUDE_line_relations_l2913_291363


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l2913_291357

theorem max_sum_of_roots (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 8) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
  ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 8 →
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) ≤
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ∧
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 3 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l2913_291357


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l2913_291351

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

def has_16_divisors (n : ℕ+) : Prop := divisor_count n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ+), has_16_divisors n ∧ ∀ (m : ℕ+), has_16_divisors m → n ≤ m :=
by
  use 216
  sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l2913_291351


namespace NUMINAMATH_CALUDE_armans_sister_age_l2913_291354

theorem armans_sister_age (arman_age sister_age : ℚ) : 
  arman_age = 6 * sister_age →
  arman_age + 4 = 40 →
  sister_age - 4 = 16 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_armans_sister_age_l2913_291354


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l2913_291355

theorem arithmetic_geometric_harmonic_means (p q r : ℝ) : 
  ((p + q) / 2 = 10) →
  (Real.sqrt (p * q) = 12) →
  ((q + r) / 2 = 26) →
  (2 / (1 / p + 1 / r) = 8) →
  (r - p = 32) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l2913_291355


namespace NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2913_291394

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2913_291394


namespace NUMINAMATH_CALUDE_kyro_debt_payment_percentage_l2913_291380

/-- Proves that Kyro paid 80% of her debt to Fernanda given the problem conditions -/
theorem kyro_debt_payment_percentage (aryan_debt : ℝ) (kyro_debt : ℝ) 
  (aryan_payment_percentage : ℝ) (initial_savings : ℝ) (final_savings : ℝ) :
  aryan_debt = 1200 →
  aryan_debt = 2 * kyro_debt →
  aryan_payment_percentage = 0.6 →
  initial_savings = 300 →
  final_savings = 1500 →
  (kyro_debt - (final_savings - initial_savings - aryan_payment_percentage * aryan_debt)) / kyro_debt = 0.2 := by
  sorry

#check kyro_debt_payment_percentage

end NUMINAMATH_CALUDE_kyro_debt_payment_percentage_l2913_291380


namespace NUMINAMATH_CALUDE_total_price_calculation_l2913_291328

def jewelry_original_price : ℝ := 30
def painting_original_price : ℝ := 100
def jewelry_price_increase : ℝ := 10
def painting_price_increase_percentage : ℝ := 0.20
def jewelry_sales_tax : ℝ := 0.06
def painting_sales_tax : ℝ := 0.08
def discount_percentage : ℝ := 0.10
def discount_min_amount : ℝ := 800
def jewelry_quantity : ℕ := 2
def painting_quantity : ℕ := 5

def jewelry_new_price : ℝ := jewelry_original_price + jewelry_price_increase
def painting_new_price : ℝ := painting_original_price * (1 + painting_price_increase_percentage)

def jewelry_price_with_tax : ℝ := jewelry_new_price * (1 + jewelry_sales_tax)
def painting_price_with_tax : ℝ := painting_new_price * (1 + painting_sales_tax)

def total_price : ℝ := jewelry_price_with_tax * jewelry_quantity + painting_price_with_tax * painting_quantity

theorem total_price_calculation :
  total_price = 732.80 ∧ total_price < discount_min_amount :=
sorry

end NUMINAMATH_CALUDE_total_price_calculation_l2913_291328


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l2913_291318

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 : ℝ) - 7 * (x^3 : ℝ) = 54 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l2913_291318


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2913_291307

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : 
  min x y = 8 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2913_291307


namespace NUMINAMATH_CALUDE_flour_cost_l2913_291308

/-- Represents the cost of ingredients and cake slices --/
structure CakeCost where
  flour : ℝ
  sugar : ℝ
  butter : ℝ
  eggs : ℝ
  total : ℝ
  sliceCount : ℕ
  sliceCost : ℝ
  dogAteCost : ℝ

/-- Theorem stating that given the total cost of ingredients and the cost of what the dog ate, 
    the cost of flour is $4 --/
theorem flour_cost (c : CakeCost) 
  (h1 : c.sugar = 2)
  (h2 : c.butter = 2.5)
  (h3 : c.eggs = 0.5)
  (h4 : c.total = c.flour + c.sugar + c.butter + c.eggs)
  (h5 : c.sliceCount = 6)
  (h6 : c.sliceCost = c.total / c.sliceCount)
  (h7 : c.dogAteCost = 6)
  (h8 : c.dogAteCost = 4 * c.sliceCost) :
  c.flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_flour_cost_l2913_291308


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2913_291392

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3*(a - b)^2) / (a * b * (1 - a - b)) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2913_291392


namespace NUMINAMATH_CALUDE_solution_set_properties_l2913_291344

-- Define the set M
def M : Set ℝ := {x | 3 - 2*x < 0}

-- Theorem statement
theorem solution_set_properties :
  0 ∉ M ∧ 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_solution_set_properties_l2913_291344


namespace NUMINAMATH_CALUDE_nested_function_evaluation_l2913_291369

-- Define the functions a and b
def a (k : ℕ) : ℕ := (k + 1) ^ 2
def b (k : ℕ) : ℕ := k ^ 3 - 2 * k + 1

-- State the theorem
theorem nested_function_evaluation :
  b (a (a (a (a 1)))) = 95877196142432 :=
by sorry

end NUMINAMATH_CALUDE_nested_function_evaluation_l2913_291369


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2913_291382

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1/2 : ℚ) + n/9 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2913_291382


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l2913_291371

/-- The number of children in the family -/
def num_children : ℕ := 6

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The probability of having an unequal number of sons and daughters -/
def unequal_gender_prob : ℚ := 11/16

theorem unequal_gender_probability :
  (1 : ℚ) - (Nat.choose num_children (num_children / 2) : ℚ) / (2 ^ num_children) = unequal_gender_prob :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l2913_291371
