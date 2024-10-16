import Mathlib

namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3033_303397

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3033_303397


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l3033_303319

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_N_complement_M : N ∩ (Mᶜ) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l3033_303319


namespace NUMINAMATH_CALUDE_certain_number_proof_l3033_303302

theorem certain_number_proof (x : ℝ) : 
  (3 - (1/5) * x) - (4 - (1/7) * 210) = 114 → x = -425 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3033_303302


namespace NUMINAMATH_CALUDE_basketball_players_l3033_303366

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 6)
  (h3 : total = 11)
  (h4 : total = cricket + basketball - both) :
  basketball = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l3033_303366


namespace NUMINAMATH_CALUDE_fourth_group_frequency_l3033_303314

theorem fourth_group_frequency 
  (groups : Fin 6 → ℝ) 
  (first_three_sum : (groups 0) + (groups 1) + (groups 2) = 0.65)
  (last_two_sum : (groups 4) + (groups 5) = 0.32)
  (all_sum_to_one : (groups 0) + (groups 1) + (groups 2) + (groups 3) + (groups 4) + (groups 5) = 1) :
  groups 3 = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_frequency_l3033_303314


namespace NUMINAMATH_CALUDE_floor_distinctness_iff_range_l3033_303326

variable (N M : ℕ)
variable (a : ℝ)

-- Define the property that floor values of ka are distinct
def distinctFloorMultiples : Prop :=
  ∀ k l, k ≠ l → k ≤ N → l ≤ N → ⌊k * a⌋ ≠ ⌊l * a⌋

-- Define the property that floor values of k/a are distinct
def distinctFloorDivisions : Prop :=
  ∀ k l, k ≠ l → k ≤ M → l ≤ M → ⌊k / a⌋ ≠ ⌊l / a⌋

theorem floor_distinctness_iff_range (hN : N > 1) (hM : M > 1) :
  (distinctFloorMultiples N a ∧ distinctFloorDivisions M a) ↔
  ((N - 1 : ℝ) / N ≤ a ∧ a ≤ M / (M - 1 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_floor_distinctness_iff_range_l3033_303326


namespace NUMINAMATH_CALUDE_hash_composition_20_l3033_303372

-- Define the # operation
def hash (N : ℝ) : ℝ := (0.5 * N)^2 + 1

-- State the theorem
theorem hash_composition_20 : hash (hash (hash 20)) = 1627102.64 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_20_l3033_303372


namespace NUMINAMATH_CALUDE_bookstore_sales_amount_l3033_303338

theorem bookstore_sales_amount (total_calculators : ℕ) (price1 price2 : ℕ) (quantity1 : ℕ) :
  total_calculators = 85 →
  price1 = 15 →
  price2 = 67 →
  quantity1 = 35 →
  (quantity1 * price1 + (total_calculators - quantity1) * price2 : ℕ) = 3875 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_amount_l3033_303338


namespace NUMINAMATH_CALUDE_triangle_problem_l3033_303375

/-- Given a triangle ABC with the specified properties, prove AC = 5 and ∠A = 120° --/
theorem triangle_problem (A B C : ℝ) (BC AB AC : ℝ) (angleA : ℝ) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ angleA = 120 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3033_303375


namespace NUMINAMATH_CALUDE_basketball_team_lineups_l3033_303394

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid lineups for the basketball team -/
def validLineups : ℕ :=
  choose 15 7 - choose 13 5

theorem basketball_team_lineups :
  validLineups = 5148 := by sorry

end NUMINAMATH_CALUDE_basketball_team_lineups_l3033_303394


namespace NUMINAMATH_CALUDE_positive_solution_x_l3033_303349

theorem positive_solution_x (x y z : ℝ) : 
  x * y = 8 - 2 * x - 3 * y →
  y * z = 8 - 4 * y - 2 * z →
  x * z = 40 - 5 * x - 3 * z →
  x > 0 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3033_303349


namespace NUMINAMATH_CALUDE_problem_solution_l3033_303378

theorem problem_solution (a : ℝ) : a = 1 / (Real.sqrt 2 - 1) → 4 * a^2 - 8 * a + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3033_303378


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3033_303323

/-- The magnitude of the complex number z = 1 / (2 + i) is equal to √3 / 3 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 1 / (2 + i)
  Complex.abs z = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3033_303323


namespace NUMINAMATH_CALUDE_initial_average_price_l3033_303354

/-- The price of an apple in cents -/
def apple_price : ℕ := 40

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits Mary initially selects -/
def total_fruits : ℕ := 10

/-- The number of oranges Mary puts back -/
def oranges_removed : ℕ := 5

/-- The average price of remaining fruits after removing oranges, in cents -/
def remaining_avg_price : ℕ := 48

theorem initial_average_price (a o : ℕ) 
  (h1 : a + o = total_fruits)
  (h2 : (apple_price * a + orange_price * o) / total_fruits = 54)
  (h3 : (apple_price * a + orange_price * (o - oranges_removed)) / (total_fruits - oranges_removed) = remaining_avg_price) :
  (apple_price * a + orange_price * o) / total_fruits = 54 :=
sorry

end NUMINAMATH_CALUDE_initial_average_price_l3033_303354


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3033_303362

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →  -- N is a two-digit number
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →  -- Property condition
  (N = 32 ∨ N = 64 ∨ N = 96) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3033_303362


namespace NUMINAMATH_CALUDE_line_segments_proportion_l3033_303340

theorem line_segments_proportion : 
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 2
  let d : ℝ := 4
  (a / b = c / d) := by sorry

end NUMINAMATH_CALUDE_line_segments_proportion_l3033_303340


namespace NUMINAMATH_CALUDE_f_positive_iff_in_intervals_l3033_303307

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff_in_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_in_intervals_l3033_303307


namespace NUMINAMATH_CALUDE_stick_pieces_l3033_303341

def stick_length : ℕ := 60

def marks_10 : List ℕ := [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
def marks_12 : List ℕ := [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
def marks_15 : List ℕ := [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

def all_marks : List ℕ := marks_10 ++ marks_12 ++ marks_15

theorem stick_pieces : 
  (all_marks.toFinset.card) + 1 = 28 := by sorry

end NUMINAMATH_CALUDE_stick_pieces_l3033_303341


namespace NUMINAMATH_CALUDE_multiply_subtract_equal_computation_result_l3033_303300

theorem multiply_subtract_equal (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem computation_result : 65 * 1515 - 25 * 1515 = 60600 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_equal_computation_result_l3033_303300


namespace NUMINAMATH_CALUDE_difference_in_group_l3033_303393

theorem difference_in_group (partition : Finset (Finset Nat)) : 
  (partition.card = 2) →
  (partition.biUnion id = Finset.range 5) →
  (∃ (group : Finset Nat) (a b c : Nat), 
    group ∈ partition ∧ 
    a ∈ group ∧ b ∈ group ∧ c ∈ group ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a - b = c) := by
  sorry

end NUMINAMATH_CALUDE_difference_in_group_l3033_303393


namespace NUMINAMATH_CALUDE_fill_time_both_pipes_l3033_303336

-- Define the time it takes for Pipe A to fill the tank
def pipeA_time : ℝ := 12

-- Define the rate at which Pipe B fills the tank relative to Pipe A
def pipeB_rate_multiplier : ℝ := 3

-- Theorem stating the time it takes to fill the tank with both pipes open
theorem fill_time_both_pipes (pipeA_time : ℝ) (pipeB_rate_multiplier : ℝ) 
  (h1 : pipeA_time > 0) (h2 : pipeB_rate_multiplier > 0) :
  (1 / (1 / pipeA_time + pipeB_rate_multiplier / pipeA_time)) = 3 := by
  sorry

#check fill_time_both_pipes

end NUMINAMATH_CALUDE_fill_time_both_pipes_l3033_303336


namespace NUMINAMATH_CALUDE_math_score_proof_l3033_303385

theorem math_score_proof (a b c : ℕ) : 
  (a + b + c = 288) →  -- Sum of scores is 288
  (∃ k : ℕ, a = 2*k ∧ b = 2*k + 2 ∧ c = 2*k + 4) →  -- Consecutive even numbers
  b = 96  -- Mathematics score is 96
:= by sorry

end NUMINAMATH_CALUDE_math_score_proof_l3033_303385


namespace NUMINAMATH_CALUDE_inclination_angle_60_degrees_l3033_303352

def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

theorem inclination_angle_60_degrees :
  line (Real.sqrt 3) 4 →
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_60_degrees_l3033_303352


namespace NUMINAMATH_CALUDE_equation_solution_l3033_303345

theorem equation_solution : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 10) + 1 / (x^2 - 11*x - 10)
  ∀ x : ℝ, f x = 0 ↔ 
    x = (-15 + Real.sqrt 265) / 2 ∨ 
    x = (-15 - Real.sqrt 265) / 2 ∨ 
    x = (6 + Real.sqrt 76) / 2 ∨ 
    x = (6 - Real.sqrt 76) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3033_303345


namespace NUMINAMATH_CALUDE_f_greater_than_log_over_x_minus_one_l3033_303325

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1) + 1 / x

theorem f_greater_than_log_over_x_minus_one (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  f x > Real.log x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_log_over_x_minus_one_l3033_303325


namespace NUMINAMATH_CALUDE_intersection_equal_angles_not_always_perpendicular_l3033_303305

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (angle_with : Line → Plane → ℝ)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem intersection_equal_angles_not_always_perpendicular
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane),
    (intersect α β = m) →
    (angle_with n α = angle_with n β) →
    (perpendicular m n)) :=
sorry

end NUMINAMATH_CALUDE_intersection_equal_angles_not_always_perpendicular_l3033_303305


namespace NUMINAMATH_CALUDE_problem_solution_l3033_303342

theorem problem_solution (a b : ℤ) : 
  (5 + a = 6 - b) → (6 + b = 9 + a) → (5 - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3033_303342


namespace NUMINAMATH_CALUDE_sum_of_factors_l3033_303370

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  a + b + c + d + e = 39 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3033_303370


namespace NUMINAMATH_CALUDE_friends_meeting_movie_and_games_l3033_303389

theorem friends_meeting_movie_and_games 
  (total : ℕ) 
  (movie : ℕ) 
  (picnic : ℕ) 
  (games : ℕ) 
  (movie_and_picnic : ℕ) 
  (picnic_and_games : ℕ) 
  (all_three : ℕ) 
  (h1 : total = 31)
  (h2 : movie = 10)
  (h3 : picnic = 20)
  (h4 : games = 5)
  (h5 : movie_and_picnic = 4)
  (h6 : picnic_and_games = 0)
  (h7 : all_three = 2) : 
  ∃ (movie_and_games : ℕ), 
    total = movie + picnic + games - movie_and_picnic - movie_and_games - picnic_and_games + all_three ∧ 
    movie_and_games = 2 := by
  sorry

end NUMINAMATH_CALUDE_friends_meeting_movie_and_games_l3033_303389


namespace NUMINAMATH_CALUDE_sqrt_36_div_6_l3033_303357

theorem sqrt_36_div_6 : Real.sqrt 36 / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_36_div_6_l3033_303357


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l3033_303330

theorem max_side_length_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Different side lengths
  a + b + c = 24 →  -- Perimeter is 24
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 →  -- Maximum side length is 11
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  ∃ (x y z : ℕ), x + y + z = 24 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x ≤ 11 ∧ y ≤ 11 ∧ z ≤ 11 ∧ 
    (x + y > z ∧ y + z > x ∧ x + z > y) ∧
    (∀ w : ℕ, w > 11 → ¬(∃ u v : ℕ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
      u + v + w = 24 ∧ u + v > w ∧ v + w > u ∧ u + w > v)) :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l3033_303330


namespace NUMINAMATH_CALUDE_binomial_12_11_squared_l3033_303359

theorem binomial_12_11_squared : (Nat.choose 12 11)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_binomial_12_11_squared_l3033_303359


namespace NUMINAMATH_CALUDE_correct_operation_l3033_303388

theorem correct_operation (x : ℝ) : 3 * x - 2 * x = x := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3033_303388


namespace NUMINAMATH_CALUDE_inequality_proof_l3033_303348

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3033_303348


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3033_303310

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ := fun _ ↦ 1

/-- The main theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y) ∧ (f 0 = 1) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3033_303310


namespace NUMINAMATH_CALUDE_total_tickets_is_56_l3033_303315

/-- The total number of tickets spent during three trips to the arcade -/
def total_tickets : ℕ :=
  let first_trip := 2 + 10 + 2
  let second_trip := 3 + 7 + 5
  let third_trip := 8 + 15 + 4
  first_trip + second_trip + third_trip

/-- Theorem stating that the total number of tickets spent is 56 -/
theorem total_tickets_is_56 : total_tickets = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_56_l3033_303315


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l3033_303353

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l3033_303353


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3033_303391

theorem framed_painting_ratio : 
  ∀ (y : ℝ),
  y > 0 →
  (15 + 2*y) * (20 + 6*y) = 2 * 15 * 20 →
  (min (15 + 2*y) (20 + 6*y)) / (max (15 + 2*y) (20 + 6*y)) = 4/7 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3033_303391


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l3033_303324

theorem correct_fraction_proof (x y : ℕ) (h : x > 0 ∧ y > 0) :
  (5 : ℚ) / 6 * 480 = x / y * 480 + 250 → x / y = (5 : ℚ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l3033_303324


namespace NUMINAMATH_CALUDE_clea_escalator_ride_time_l3033_303316

/-- Represents the escalator scenario for Clea -/
structure EscalatorScenario where
  /-- Time (in seconds) for Clea to walk down a stationary escalator -/
  stationary_time : ℝ
  /-- Time (in seconds) for Clea to walk down a moving escalator -/
  moving_time : ℝ
  /-- Slowdown factor for the escalator during off-peak hours -/
  slowdown_factor : ℝ

/-- Calculates the time for Clea to ride the slower escalator without walking -/
def ride_time (scenario : EscalatorScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that given the specific scenario, the ride time is 60 seconds -/
theorem clea_escalator_ride_time :
  let scenario : EscalatorScenario :=
    { stationary_time := 80
      moving_time := 30
      slowdown_factor := 0.8 }
  ride_time scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_clea_escalator_ride_time_l3033_303316


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l3033_303308

theorem solution_satisfies_equations : ∃ x : ℚ, 8 * x^3 = 125 ∧ 4 * (x - 1)^2 = 9 := by
  use 5/2
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l3033_303308


namespace NUMINAMATH_CALUDE_school_classrooms_l3033_303318

theorem school_classrooms 
  (total_students : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : total_students = 58)
  (h2 : seats_per_bus = 2)
  (h3 : buses_needed = 29)
  (h4 : total_students = buses_needed * seats_per_bus)
  (h5 : ∃ (students_per_class : ℕ), total_students % students_per_class = 0) :
  ∃ (num_classrooms : ℕ), num_classrooms = 2 ∧ 
    total_students / num_classrooms = buses_needed := by
  sorry

end NUMINAMATH_CALUDE_school_classrooms_l3033_303318


namespace NUMINAMATH_CALUDE_triangle_side_range_l3033_303321

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- acute triangle
  A + B + C = π ∧ -- sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧ -- positive sides
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- given equation
  a = Real.sqrt 3 -- given value of a
  →
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3033_303321


namespace NUMINAMATH_CALUDE_car_race_distance_l3033_303343

theorem car_race_distance (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) : 
  karen_speed = 75 →
  tom_speed = 50 →
  karen_delay = 7 / 60 →
  win_margin = 5 →
  (karen_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) - 
   tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay)) = win_margin →
  tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) = 27.5 :=
by sorry

end NUMINAMATH_CALUDE_car_race_distance_l3033_303343


namespace NUMINAMATH_CALUDE_triangle_side_sharing_l3033_303312

/-- A point on a circle -/
structure Point

/-- A triangle formed by three points -/
structure Triangle (Point : Type) where
  p1 : Point
  p2 : Point
  p3 : Point

/-- A side of a triangle -/
structure Side (Point : Type) where
  p1 : Point
  p2 : Point

/-- Definition of 8 points on a circle -/
def circle_points : Finset Point := sorry

/-- Definition of all possible triangles formed by the 8 points -/
def all_triangles : Finset (Triangle Point) := sorry

/-- Definition of all possible sides formed by the 8 points -/
def all_sides : Finset (Side Point) := sorry

/-- Function to get the sides of a triangle -/
def triangle_sides (t : Triangle Point) : Finset (Side Point) := sorry

theorem triangle_side_sharing :
  ∀ (triangles : Finset (Triangle Point)),
    triangles ⊆ all_triangles →
    triangles.card = 9 →
    ∃ (t1 t2 : Triangle Point) (s : Side Point),
      t1 ∈ triangles ∧ t2 ∈ triangles ∧ t1 ≠ t2 ∧
      s ∈ triangle_sides t1 ∧ s ∈ triangle_sides t2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sharing_l3033_303312


namespace NUMINAMATH_CALUDE_work_completion_time_l3033_303347

theorem work_completion_time (b_time : ℕ) (b_worked : ℕ) (a_remaining : ℕ) : 
  b_time = 15 → b_worked = 10 → a_remaining = 3 → 
  ∃ (a_time : ℕ), a_time = 9 ∧ 
    (b_worked : ℚ) / b_time + a_remaining / a_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3033_303347


namespace NUMINAMATH_CALUDE_garden_perimeter_l3033_303380

theorem garden_perimeter : 
  ∀ (width length : ℝ),
  length = 3 * width + 2 →
  length = 38 →
  2 * length + 2 * width = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3033_303380


namespace NUMINAMATH_CALUDE_fraction_equality_l3033_303309

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) :
  (a + b) / (a - b) = -1001 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3033_303309


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3033_303398

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l3033_303398


namespace NUMINAMATH_CALUDE_grading_implications_l3033_303379

-- Define the type for grades
inductive Grade : Type
| A : Grade
| B : Grade
| C : Grade
| D : Grade
| F : Grade

-- Define the ordering on grades
instance : LE Grade where
  le := λ g₁ g₂ => match g₁, g₂ with
    | Grade.F, _ => true
    | Grade.D, Grade.D | Grade.D, Grade.C | Grade.D, Grade.B | Grade.D, Grade.A => true
    | Grade.C, Grade.C | Grade.C, Grade.B | Grade.C, Grade.A => true
    | Grade.B, Grade.B | Grade.B, Grade.A => true
    | Grade.A, Grade.A => true
    | _, _ => false

instance : LT Grade where
  lt := λ g₁ g₂ => g₁ ≤ g₂ ∧ g₁ ≠ g₂

-- Define the grading function
def grading_function (score : ℚ) : Grade :=
  if score ≥ 90 then Grade.B
  else if score < 70 then Grade.C
  else Grade.C  -- Default case, can be any grade between B and C

-- State the theorem
theorem grading_implications :
  (∀ (score : ℚ) (grade : Grade),
    (grading_function score = grade → 
      (grade < Grade.B → score < 90) ∧
      (grade > Grade.C → score ≥ 70))) :=
sorry

end NUMINAMATH_CALUDE_grading_implications_l3033_303379


namespace NUMINAMATH_CALUDE_parallelogram_EFGH_area_l3033_303363

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.base * p.height

/-- Theorem: The area of parallelogram EFGH is 18 square units -/
theorem parallelogram_EFGH_area :
  let p : Parallelogram := { base := 6, height := 3 }
  area p = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_EFGH_area_l3033_303363


namespace NUMINAMATH_CALUDE_working_partner_receives_6000_l3033_303331

/-- Calculates the amount received by the working partner in a business partnership --/
def amount_received_by_working_partner (total_profit management_fee_percentage a_capital b_capital : ℚ) : ℚ :=
  let management_fee := management_fee_percentage * total_profit
  let remaining_profit := total_profit - management_fee
  let total_capital := a_capital + b_capital
  let a_share := (a_capital / total_capital) * remaining_profit
  management_fee + a_share

/-- Theorem stating that the working partner receives 6000 Rs given the specified conditions --/
theorem working_partner_receives_6000 :
  let total_profit : ℚ := 9600
  let management_fee_percentage : ℚ := 1/10
  let a_capital : ℚ := 3500
  let b_capital : ℚ := 2500
  amount_received_by_working_partner total_profit management_fee_percentage a_capital b_capital = 6000 := by
  sorry

end NUMINAMATH_CALUDE_working_partner_receives_6000_l3033_303331


namespace NUMINAMATH_CALUDE_acute_and_less_than_90_subset_l3033_303373

-- Define the sets
def A : Set ℝ := {x | ∃ k : ℤ, k * 360 < x ∧ x < k * 360 + 90}
def B : Set ℝ := {x | 0 < x ∧ x < 90}
def C : Set ℝ := {x | x < 90}

-- Theorem statement
theorem acute_and_less_than_90_subset :
  B ∪ C ⊆ C := by sorry

end NUMINAMATH_CALUDE_acute_and_less_than_90_subset_l3033_303373


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l3033_303303

def circle1_center : ℝ × ℝ := (3, 3)
def circle2_center : ℝ × ℝ := (15, 10)
def circle1_radius : ℝ := 5
def circle2_radius : ℝ := 10

theorem common_external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧ 
  (∀ (x y : ℝ), y = m * x + b → 
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 = circle1_radius^2 ∨
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) →
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 > circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 > circle2_radius^2) ∨
    ((x - circle1_center.1)^2 + (y - circle1_center.2)^2 < circle1_radius^2 ∧
     (x - circle2_center.1)^2 + (y - circle2_center.2)^2 < circle2_radius^2)) ∧
  b = 446 / 95 := by
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l3033_303303


namespace NUMINAMATH_CALUDE_reach_probability_is_15_1024_l3033_303384

/-- Represents a point in a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of a single step in any direction --/
def stepProbability : Rat := 1 / 4

/-- The starting point --/
def start : Point := ⟨0, 0⟩

/-- The target point --/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : Nat := 8

/-- Calculates the probability of reaching the target from the start in at most maxSteps --/
def reachProbability (start : Point) (target : Point) (maxSteps : Nat) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem reach_probability_is_15_1024 : 
  reachProbability start target maxSteps = 15 / 1024 := by sorry

end NUMINAMATH_CALUDE_reach_probability_is_15_1024_l3033_303384


namespace NUMINAMATH_CALUDE_root_inequality_l3033_303355

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) : f a < f 1 ∧ f 1 < f b := by
  sorry

end

end NUMINAMATH_CALUDE_root_inequality_l3033_303355


namespace NUMINAMATH_CALUDE_modular_product_equivalence_l3033_303376

theorem modular_product_equivalence (n : ℕ) : 
  (507 * 873) % 77 = n ∧ 0 ≤ n ∧ n < 77 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_modular_product_equivalence_l3033_303376


namespace NUMINAMATH_CALUDE_farm_roosters_l3033_303333

theorem farm_roosters (initial_hens : ℕ) (initial_roosters : ℕ) : 
  initial_roosters = initial_hens + 6 →
  initial_hens + 8 = 20 →
  initial_roosters + 4 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_roosters_l3033_303333


namespace NUMINAMATH_CALUDE_unique_base_for_1024_l3033_303344

theorem unique_base_for_1024 : ∃! b : ℕ, 4 ≤ b ∧ b ≤ 12 ∧ 1024 % b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_for_1024_l3033_303344


namespace NUMINAMATH_CALUDE_mean_of_solutions_l3033_303329

-- Define the polynomial
def f (x : ℝ) := x^3 + 5*x^2 - 14*x

-- Define the set of solutions
def solutions := {x : ℝ | f x = 0}

-- State the theorem
theorem mean_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solutions ∧ s.card = 3 ∧ (s.sum id) / s.card = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_solutions_l3033_303329


namespace NUMINAMATH_CALUDE_distance_between_points_l3033_303361

/-- The distance between two points A(-1, 2) and B(-4, 6) is 5. -/
theorem distance_between_points : 
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (-4, 6)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3033_303361


namespace NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l3033_303360

theorem square_sum_from_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_diff_and_product_l3033_303360


namespace NUMINAMATH_CALUDE_triangle_inequality_for_specific_triangle_l3033_303311

/-- A triangle with sides of length 3, 4, and x is valid if and only if 1 < x < 7 -/
theorem triangle_inequality_for_specific_triangle (x : ℝ) :
  (3 + 4 > x ∧ 3 + x > 4 ∧ 4 + x > 3) ↔ (1 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_specific_triangle_l3033_303311


namespace NUMINAMATH_CALUDE_spider_eyes_l3033_303367

theorem spider_eyes (spider_count : ℕ) (ant_count : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  ant_count = 50 →
  ant_eyes = 2 →
  total_eyes = 124 →
  total_eyes = spider_count * (total_eyes - ant_count * ant_eyes) / spider_count →
  (total_eyes - ant_count * ant_eyes) / spider_count = 8 :=
by sorry

end NUMINAMATH_CALUDE_spider_eyes_l3033_303367


namespace NUMINAMATH_CALUDE_inequality_proof_l3033_303377

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) :
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3033_303377


namespace NUMINAMATH_CALUDE_semi_circle_radius_is_27_l3033_303392

/-- A rectangle with a semi-circle inscribed, where the diameter of the semi-circle
    lies on the length of the rectangle. -/
structure SemiCircleRectangle where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  radius : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  diameter_eq_length : 2 * radius = length
  width_eq_radius : width = 2 * radius

/-- The theorem stating that for a rectangle with perimeter 216 and an inscribed semi-circle,
    the radius of the semi-circle is 27. -/
theorem semi_circle_radius_is_27 (rect : SemiCircleRectangle) 
    (h : rect.perimeter = 216) : rect.radius = 27 := by
  sorry


end NUMINAMATH_CALUDE_semi_circle_radius_is_27_l3033_303392


namespace NUMINAMATH_CALUDE_largest_common_term_under_1000_l3033_303395

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 7 + 11 * m

/-- A common term of both sequences -/
def commonTerm (n m : ℕ) : Prop := seq1 n = seq2 m

/-- The largest common term less than 1000 -/
theorem largest_common_term_under_1000 :
  ∃ (n m : ℕ), commonTerm n m ∧ seq1 n = 974 ∧ 
  (∀ (k l : ℕ), commonTerm k l → seq1 k < 1000 → seq1 k ≤ 974) :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_under_1000_l3033_303395


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l3033_303374

def scores : List ℕ := [95, 97, 96, 97, 99, 98]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

theorem scores_mode_and_median :
  mode scores = 97 ∧ median scores = 97 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l3033_303374


namespace NUMINAMATH_CALUDE_randy_spends_two_dollars_per_trip_l3033_303339

/-- Calculates the amount spent per store trip -/
def amount_per_trip (initial_amount final_amount trips_per_month months : ℕ) : ℚ :=
  (initial_amount - final_amount : ℚ) / (trips_per_month * months : ℚ)

/-- Theorem: Randy spends $2 per store trip -/
theorem randy_spends_two_dollars_per_trip :
  amount_per_trip 200 104 4 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_spends_two_dollars_per_trip_l3033_303339


namespace NUMINAMATH_CALUDE_adam_wall_area_l3033_303350

/-- The total area of four rectangular walls with given dimensions -/
def totalWallArea (w1_width w1_height w2_width w2_height w3_width w3_height w4_width w4_height : ℝ) : ℝ :=
  w1_width * w1_height + w2_width * w2_height + w3_width * w3_height + w4_width * w4_height

/-- Theorem: The total area of the walls with the given dimensions is 160 square feet -/
theorem adam_wall_area :
  totalWallArea 4 8 6 8 4 8 6 8 = 160 := by
  sorry

#eval totalWallArea 4 8 6 8 4 8 6 8

end NUMINAMATH_CALUDE_adam_wall_area_l3033_303350


namespace NUMINAMATH_CALUDE_jackson_souvenirs_l3033_303387

theorem jackson_souvenirs :
  let hermit_crabs : ℕ := 45
  let shells_per_crab : ℕ := 3
  let starfish_per_shell : ℕ := 2
  let total_shells : ℕ := hermit_crabs * shells_per_crab
  let total_starfish : ℕ := total_shells * starfish_per_shell
  let total_souvenirs : ℕ := hermit_crabs + total_shells + total_starfish
  total_souvenirs = 450 := by
sorry

end NUMINAMATH_CALUDE_jackson_souvenirs_l3033_303387


namespace NUMINAMATH_CALUDE_aa_existence_l3033_303335

theorem aa_existence : ∃ aa : ℕ, 1 ≤ aa ∧ aa ≤ 9 ∧ (7 * aa^3) % 100 ≥ 10 ∧ (7 * aa^3) % 100 < 20 :=
by sorry

end NUMINAMATH_CALUDE_aa_existence_l3033_303335


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3033_303317

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 8 = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3033_303317


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3033_303301

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 210 > 0 ∧ x + x + 210 = 360 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3033_303301


namespace NUMINAMATH_CALUDE_box_surface_area_l3033_303351

/-- Proves that a rectangular box with given edge sum and diagonal length has a specific surface area -/
theorem box_surface_area (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 168) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1139 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l3033_303351


namespace NUMINAMATH_CALUDE_maggots_eaten_second_feeding_correct_l3033_303306

/-- Given the total number of maggots served, the number of maggots laid out and eaten in the first feeding,
    and the number laid out in the second feeding, calculate the number of maggots eaten in the second feeding. -/
def maggots_eaten_second_feeding (total_served : ℕ) (first_feeding_laid_out : ℕ) (first_feeding_eaten : ℕ) (second_feeding_laid_out : ℕ) : ℕ :=
  total_served - first_feeding_eaten - second_feeding_laid_out

/-- Theorem stating that the number of maggots eaten in the second feeding is correct -/
theorem maggots_eaten_second_feeding_correct
  (total_served : ℕ)
  (first_feeding_laid_out : ℕ)
  (first_feeding_eaten : ℕ)
  (second_feeding_laid_out : ℕ)
  (h1 : total_served = 20)
  (h2 : first_feeding_laid_out = 10)
  (h3 : first_feeding_eaten = 1)
  (h4 : second_feeding_laid_out = 10) :
  maggots_eaten_second_feeding total_served first_feeding_laid_out first_feeding_eaten second_feeding_laid_out = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_maggots_eaten_second_feeding_correct_l3033_303306


namespace NUMINAMATH_CALUDE_laptop_price_proof_l3033_303365

/-- The original sticker price of the laptop -/
def original_price : ℝ := 500

/-- The price at Store A after discount and rebate -/
def store_a_price (x : ℝ) : ℝ := 0.82 * x - 100

/-- The price at Store B after discount -/
def store_b_price (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the original price satisfies the given conditions -/
theorem laptop_price_proof :
  store_a_price original_price = store_b_price original_price - 40 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l3033_303365


namespace NUMINAMATH_CALUDE_james_run_duration_l3033_303327

/-- Calculates the duration of James' run in minutes -/
def run_duration (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) (cal_per_min : ℕ) (excess_cal : ℕ) : ℕ :=
  let total_oz := bags * oz_per_bag
  let total_cal := total_oz * cal_per_oz
  let cal_to_burn := total_cal - excess_cal
  cal_to_burn / cal_per_min

/-- Proves that James' run duration is 40 minutes given the problem conditions -/
theorem james_run_duration :
  run_duration 3 2 150 12 420 = 40 := by
  sorry

end NUMINAMATH_CALUDE_james_run_duration_l3033_303327


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3033_303356

/-- Represents the number of books of each type -/
structure BookCounts where
  chinese : Nat
  english : Nat
  math : Nat

/-- Represents the arrangement constraints -/
structure ArrangementConstraints where
  chinese_adjacent : Bool
  english_adjacent : Bool
  math_not_adjacent : Bool

/-- Calculates the number of valid book arrangements -/
def count_arrangements (counts : BookCounts) (constraints : ArrangementConstraints) : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  let counts : BookCounts := ⟨2, 2, 3⟩
  let constraints : ArrangementConstraints := ⟨true, true, true⟩
  count_arrangements counts constraints = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3033_303356


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3033_303390

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: In an arithmetic sequence, if a₈ = 20 and S₇ = 56, then a₁₂ = 32 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.a 8 = 20)
    (h₂ : seq.S 7 = 56) :
  seq.a 12 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3033_303390


namespace NUMINAMATH_CALUDE_jungkook_has_bigger_number_l3033_303304

theorem jungkook_has_bigger_number :
  let yoongi_collected : ℕ := 4
  let jungkook_collected : ℕ := 6 + 3
  jungkook_collected > yoongi_collected :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_bigger_number_l3033_303304


namespace NUMINAMATH_CALUDE_minimum_students_l3033_303386

theorem minimum_students (b g : ℕ) : 
  b > 0 → 
  g > 0 → 
  2 * (b / 2) = g * 2 / 3 → 
  ∀ b' g', b' > 0 → g' > 0 → 2 * (b' / 2) = g' * 2 / 3 → b' + g' ≥ b + g →
  b + g = 5 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l3033_303386


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l3033_303382

theorem farmer_land_calculation (total_land : ℝ) : 
  (0.05 * 0.9 * total_land + 0.05 * 0.9 * total_land = 90) → total_land = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l3033_303382


namespace NUMINAMATH_CALUDE_ellipse_equation_l3033_303371

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b
  h4 : a^2 = b^2 + c^2
  h5 : (4 : ℝ) / a^2 + 3 / b^2 = 1

/-- The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic progression. -/
def is_arithmetic_progression (e : Ellipse) : Prop :=
  ∃ (d : ℝ), 2 * e.a = 4 * e.c ∧ e.c > 0

theorem ellipse_equation (e : Ellipse) (h : is_arithmetic_progression e) :
  e.a = 2 * Real.sqrt 2 ∧ e.b = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3033_303371


namespace NUMINAMATH_CALUDE_smallest_distance_circle_ellipse_l3033_303369

/-- The smallest distance between a point on the unit circle and a point on a specific ellipse -/
theorem smallest_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 9) = 1}
  (∃ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ circle ∧ B ∈ ellipse ∧
    ∀ (C : ℝ × ℝ) (D : ℝ × ℝ), C ∈ circle → D ∈ ellipse →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_distance_circle_ellipse_l3033_303369


namespace NUMINAMATH_CALUDE_no_numbers_seven_times_digit_sum_l3033_303313

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_seven_times_digit_sum : 
  ∀ n : ℕ, n > 0 ∧ n < 10000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_numbers_seven_times_digit_sum_l3033_303313


namespace NUMINAMATH_CALUDE_nell_card_collection_l3033_303399

/-- Represents the number and types of cards Nell has --/
structure CardCollection where
  baseball : ℕ
  ace : ℕ
  pokemon : ℕ

/-- Represents the initial state of Nell's card collection --/
def initial_collection : CardCollection := {
  baseball := 438,
  ace := 18,
  pokemon := 312
}

/-- Represents the state of Nell's card collection after giving away cards --/
def after_giveaway (c : CardCollection) : CardCollection := {
  baseball := c.baseball - c.baseball / 2,
  ace := c.ace - c.ace / 3,
  pokemon := c.pokemon
}

/-- Represents the final state of Nell's card collection after trading --/
def final_collection (c : CardCollection) : CardCollection := {
  baseball := c.baseball,
  ace := c.ace + 37,
  pokemon := c.pokemon - 52
}

/-- The main theorem to prove --/
theorem nell_card_collection :
  let final := final_collection (after_giveaway initial_collection)
  (final.baseball - final.ace = 170) ∧
  (final.baseball : ℚ) / 219 = (final.ace : ℚ) / 49 ∧
  (final.ace : ℚ) / 49 = (final.pokemon : ℚ) / 260 := by
  sorry


end NUMINAMATH_CALUDE_nell_card_collection_l3033_303399


namespace NUMINAMATH_CALUDE_division_decimal_l3033_303381

theorem division_decimal : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_division_decimal_l3033_303381


namespace NUMINAMATH_CALUDE_sequence_general_term_l3033_303358

theorem sequence_general_term (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = a n + 2 * n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3033_303358


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l3033_303364

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_of_B_in_A :
  (A \ B) = {0, 2, 6, 10} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l3033_303364


namespace NUMINAMATH_CALUDE_jennys_pen_cost_l3033_303383

/-- Proves that the cost of each pen is $1.50 given the conditions of Jenny's purchase --/
theorem jennys_pen_cost 
  (print_cost : ℚ) 
  (copies : ℕ) 
  (pages : ℕ) 
  (num_pens : ℕ) 
  (payment : ℚ) 
  (change : ℚ)
  (h1 : print_cost = 1 / 10)
  (h2 : copies = 7)
  (h3 : pages = 25)
  (h4 : num_pens = 7)
  (h5 : payment = 40)
  (h6 : change = 12) :
  (payment - change - print_cost * copies * pages) / num_pens = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jennys_pen_cost_l3033_303383


namespace NUMINAMATH_CALUDE_basketball_team_starters_l3033_303332

theorem basketball_team_starters : Nat.choose 16 8 = 12870 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l3033_303332


namespace NUMINAMATH_CALUDE_solve_equation_l3033_303337

theorem solve_equation (y : ℝ) : (4 / 7) * (1 / 5) * y - 2 = 10 → y = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3033_303337


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3033_303396

theorem cubic_roots_sum_of_squares_reciprocal :
  ∀ a b c : ℝ,
  (a^3 - 6*a^2 + 11*a - 6 = 0) →
  (b^3 - 6*b^2 + 11*b - 6 = 0) →
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3033_303396


namespace NUMINAMATH_CALUDE_grid_flip_theorem_l3033_303368

/-- Represents a 4x4 binary grid -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents a flip operation on the grid -/
inductive FlipOperation
| row : Fin 4 → FlipOperation
| column : Fin 4 → FlipOperation
| diagonal : Bool → FlipOperation  -- True for main diagonal, False for anti-diagonal

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if the grid is all zeros -/
def isAllZeros (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Initial configurations -/
def initialGrid1 : Grid :=
  ![![false, true,  true,  false],
    ![true,  true,  false, true ],
    ![false, false, true,  true ],
    ![false, false, true,  true ]]

def initialGrid2 : Grid :=
  ![![false, true,  false, false],
    ![true,  true,  false, true ],
    ![false, false, false, true ],
    ![true,  false, true,  true ]]

def initialGrid3 : Grid :=
  ![![false, false, false, false],
    ![true,  true,  false, false],
    ![false, true,  false, true ],
    ![true,  false, false, true ]]

/-- Main theorem -/
theorem grid_flip_theorem :
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid1)) ∧
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid2)) ∧
  (∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid3)) :=
sorry

end NUMINAMATH_CALUDE_grid_flip_theorem_l3033_303368


namespace NUMINAMATH_CALUDE_inequality_proof_l3033_303328

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3033_303328


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3033_303320

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3033_303320


namespace NUMINAMATH_CALUDE_function_properties_l3033_303334

/-- Given a function f with parameter ω, prove properties about its graph -/
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (ω * x) + 2 * (Real.sin (ω * x / 2))^2
  -- Assume the graph has exactly three symmetric centers on [0, π]
  (∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ f y = f (2 * x₃ - y))) ∧
    (∀ (z : ℝ), 0 ≤ z ∧ z ≤ π → (z = x₁ ∨ z = x₂ ∨ z = x₃ ∨ f z ≠ f (2 * z - z)))) →
  -- Then prove:
  (13/6 ≤ ω ∧ ω < 19/6) ∧  -- 1. Range of ω
  (∃ (n : ℕ), n = 2 ∨ n = 3 ∧  -- 2. Number of axes of symmetry
    ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ (n = 3 → x₂ < x₃) ∧ x₃ < π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ (n = 3 → f y = f (2 * x₃ - y))))) ∧
  (∃ (x : ℝ), 0 < x ∧ x < π/4 ∧ f x = 3) ∧  -- 3. Maximum value on (0, π/4)
  (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < π/6 → f x < f y)  -- 4. Increasing on (0, π/6)
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3033_303334


namespace NUMINAMATH_CALUDE_infinitely_many_heinersch_triples_l3033_303322

/-- A positive integer is heinersch if it can be written as the sum of a positive square and positive cube. -/
def IsHeinersch (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^3 ∧ a > 0 ∧ b > 0

/-- The main theorem stating the existence of infinitely many heinersch numbers h such that h-1 and h+1 are also heinersch. -/
theorem infinitely_many_heinersch_triples :
  ∀ N : ℕ, ∃ t : ℕ, t > N ∧
    let h := ((9*t^4)^3 - 1) / 2
    IsHeinersch h ∧
    IsHeinersch (h-1) ∧
    IsHeinersch (h+1) := by
  sorry

/-- Helper lemma for the identity used in the proof -/
lemma cube_identity (t : ℕ) :
  (9*t^3 - 1)^3 + (9*t^4 - 3*t)^3 = (9*t^4)^3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_heinersch_triples_l3033_303322


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l3033_303346

-- Define the polynomial Q(x)
def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

-- State the theorem
theorem polynomial_coefficient_b (d b e : ℝ) :
  -- Conditions
  (∀ x, Q x d b e = 0 → -d/3 = x) ∧  -- Mean of zeros
  (∀ x y z, Q x d b e = 0 ∧ Q y d b e = 0 ∧ Q z d b e = 0 → x*y*z = -e) ∧  -- Product of zeros
  (-d/3 = 1 + d + b + e) ∧  -- Sum of coefficients equals mean of zeros
  (e = 6)  -- y-intercept is 6
  →
  b = -31 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l3033_303346
