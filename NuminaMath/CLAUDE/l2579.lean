import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_l2579_257950

/-- A circle C with center (a,b) and radius 1 -/
structure Circle where
  a : ℝ
  b : ℝ
  radius : ℝ := 1

/-- The circle C is in the first quadrant -/
def in_first_quadrant (C : Circle) : Prop :=
  C.a > 0 ∧ C.b > 0

/-- The circle C is tangent to the line 4x-3y=0 -/
def tangent_to_line (C : Circle) : Prop :=
  abs (4 * C.a - 3 * C.b) / 5 = C.radius

/-- The circle C is tangent to the x-axis -/
def tangent_to_x_axis (C : Circle) : Prop :=
  C.b = C.radius

/-- The standard equation of the circle -/
def standard_equation (C : Circle) : Prop :=
  ∀ x y : ℝ, (x - C.a)^2 + (y - C.b)^2 = C.radius^2

theorem circle_equation (C : Circle) 
  (h1 : in_first_quadrant C)
  (h2 : tangent_to_line C)
  (h3 : tangent_to_x_axis C) :
  standard_equation { a := 2, b := 1, radius := 1 } :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2579_257950


namespace NUMINAMATH_CALUDE_truck_speed_problem_l2579_257917

/-- Proves that the speed of Truck Y is 53 miles per hour given the problem conditions -/
theorem truck_speed_problem (initial_distance : ℝ) (truck_x_speed : ℝ) (overtake_time : ℝ) (final_lead : ℝ) :
  initial_distance = 13 →
  truck_x_speed = 47 →
  overtake_time = 3 →
  final_lead = 5 →
  (initial_distance + truck_x_speed * overtake_time + final_lead) / overtake_time = 53 := by
  sorry

#check truck_speed_problem

end NUMINAMATH_CALUDE_truck_speed_problem_l2579_257917


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2579_257944

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2579_257944


namespace NUMINAMATH_CALUDE_video_game_expense_is_correct_l2579_257914

def total_allowance : ℚ := 50

def book_fraction : ℚ := 1/2
def toy_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/10

def video_game_expense : ℚ := total_allowance - (book_fraction * total_allowance + toy_fraction * total_allowance + snack_fraction * total_allowance)

theorem video_game_expense_is_correct : video_game_expense = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expense_is_correct_l2579_257914


namespace NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l2579_257951

-- Part 1
theorem no_polynomial_satisfies_conditions :
  ¬(∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, (deriv P) x > (deriv (deriv P)) x ∧ P x > (deriv (deriv P)) x)) :=
sorry

-- Part 2
theorem exists_polynomial_satisfies_modified_conditions :
  ∃ P : ℝ → ℝ, (∀ x : ℝ, Differentiable ℝ P ∧ Differentiable ℝ (deriv P)) ∧
    (∀ x : ℝ, P x > (deriv P) x ∧ P x > (deriv (deriv P)) x) :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_satisfies_conditions_exists_polynomial_satisfies_modified_conditions_l2579_257951


namespace NUMINAMATH_CALUDE_monitor_length_is_14_l2579_257937

/-- Represents the dimensions of a rectangular monitor. -/
structure Monitor where
  width : ℝ
  length : ℝ
  circumference : ℝ

/-- The circumference of a rectangle is equal to twice the sum of its length and width. -/
def circumference_formula (m : Monitor) : Prop :=
  m.circumference = 2 * (m.length + m.width)

/-- Theorem: A monitor with width 9 cm and circumference 46 cm has a length of 14 cm. -/
theorem monitor_length_is_14 :
  ∃ (m : Monitor), m.width = 9 ∧ m.circumference = 46 ∧ circumference_formula m → m.length = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_monitor_length_is_14_l2579_257937


namespace NUMINAMATH_CALUDE_gold_families_count_l2579_257938

def fundraiser (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) : Prop :=
  let bronze_donation := 25
  let silver_donation := 50
  let gold_donation := 100
  let total_goal := 750
  let final_day_goal := 50
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation = 
  total_goal - final_day_goal

theorem gold_families_count : 
  ∃! gold_families : ℕ, fundraiser 10 7 gold_families :=
sorry

end NUMINAMATH_CALUDE_gold_families_count_l2579_257938


namespace NUMINAMATH_CALUDE_max_visible_cubes_12_10_9_l2579_257990

/-- Represents a rectangular block formed by unit cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point for a given block -/
def max_visible_cubes (b : Block) : ℕ :=
  b.length * b.width + b.width * b.height + b.length * b.height -
  (b.length + b.width + b.height) + 1

/-- The theorem stating that for a 12 × 10 × 9 block, the maximum number of visible unit cubes is 288 -/
theorem max_visible_cubes_12_10_9 :
  max_visible_cubes ⟨12, 10, 9⟩ = 288 := by
  sorry

#eval max_visible_cubes ⟨12, 10, 9⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12_10_9_l2579_257990


namespace NUMINAMATH_CALUDE_solve_for_a_l2579_257947

theorem solve_for_a : ∃ a : ℝ, (2 : ℝ) - a * (1 : ℝ) = 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2579_257947


namespace NUMINAMATH_CALUDE_sports_club_members_l2579_257982

theorem sports_club_members (B T Both Neither : ℕ) 
  (hB : B = 17)
  (hT : T = 17)
  (hBoth : Both = 6)
  (hNeither : Neither = 2) :
  B + T - Both + Neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l2579_257982


namespace NUMINAMATH_CALUDE_remainder_after_division_l2579_257948

theorem remainder_after_division (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 5) → n % 8 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_division_l2579_257948


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l2579_257958

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) (h : S > 0) : 
  S = (S - (R/100) * S) * (1 + 25/100) → R = 20 :=
by sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l2579_257958


namespace NUMINAMATH_CALUDE_carlys_running_schedule_l2579_257977

/-- Carly's running schedule over four weeks -/
theorem carlys_running_schedule (x : ℝ) : 
  (∃ week2 week3 : ℝ,
    week2 = 2*x + 3 ∧ 
    week3 = (9/7) * week2 ∧ 
    week3 - 5 = 4) → 
  x = 2 := by sorry

end NUMINAMATH_CALUDE_carlys_running_schedule_l2579_257977


namespace NUMINAMATH_CALUDE_brothers_ages_l2579_257988

theorem brothers_ages (x y : ℕ) : 
  x + y = 16 → 
  2 * (x + 4) = y + 4 → 
  ∃ (younger older : ℕ), younger = x ∧ older = y ∧ younger < older :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l2579_257988


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_l2579_257993

theorem fixed_point_quadratic (p : ℝ) : 
  9 * (5 : ℝ)^2 + p * 5 - 5 * p = 225 := by sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_l2579_257993


namespace NUMINAMATH_CALUDE_martin_bell_ringing_l2579_257955

theorem martin_bell_ringing (small big : ℕ) : 
  small = (big / 3) + 4 →  -- Condition 1
  small + big = 52 →      -- Condition 2
  big = 36 :=             -- Conclusion
by sorry

end NUMINAMATH_CALUDE_martin_bell_ringing_l2579_257955


namespace NUMINAMATH_CALUDE_sawyer_coaching_fee_l2579_257928

/-- Calculate the total coaching fee for Sawyer --/
theorem sawyer_coaching_fee :
  let start_date : Nat := 1  -- January 1
  let end_date : Nat := 307  -- November 3
  let daily_fee : ℚ := 39
  let discount_days : Nat := 50
  let discount_rate : ℚ := 0.1

  let full_price_days : Nat := min discount_days (end_date - start_date + 1)
  let discounted_days : Nat := (end_date - start_date + 1) - full_price_days
  let discounted_fee : ℚ := daily_fee * (1 - discount_rate)

  let total_fee : ℚ := (full_price_days : ℚ) * daily_fee + (discounted_days : ℚ) * discounted_fee

  total_fee = 10967.7 := by
    sorry

end NUMINAMATH_CALUDE_sawyer_coaching_fee_l2579_257928


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2579_257933

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2579_257933


namespace NUMINAMATH_CALUDE_globe_division_count_l2579_257906

/-- The number of parts a globe's surface is divided into, given the number of parallels and meridians -/
def globe_divisions (parallels : ℕ) (meridians : ℕ) : ℕ :=
  meridians * (parallels + 1)

/-- Theorem: A globe with 17 parallels and 24 meridians is divided into 432 parts -/
theorem globe_division_count : globe_divisions 17 24 = 432 := by
  sorry

end NUMINAMATH_CALUDE_globe_division_count_l2579_257906


namespace NUMINAMATH_CALUDE_area_of_three_sectors_l2579_257961

/-- The area of a figure formed by three sectors of a circle,
    where each sector subtends an angle of 40° at the center
    and the circle has a radius of 15. -/
theorem area_of_three_sectors (r : ℝ) (angle : ℝ) (n : ℕ) :
  r = 15 →
  angle = 40 * π / 180 →
  n = 3 →
  n * (angle / (2 * π) * π * r^2) = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_three_sectors_l2579_257961


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l2579_257984

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 6.5 →
  added_water = 3.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 17 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l2579_257984


namespace NUMINAMATH_CALUDE_car_comparison_l2579_257902

-- Define the speeds and times for both cars
def speed_M : ℝ := 1  -- Arbitrary unit speed for Car M
def speed_N : ℝ := 3 * speed_M
def start_time_M : ℝ := 0
def start_time_N : ℝ := 2
def total_time : ℝ := 3  -- From the solution, but not directly given in the problem

-- Define the distance traveled by each car
def distance_M (t : ℝ) : ℝ := speed_M * t
def distance_N (t : ℝ) : ℝ := speed_N * (t - start_time_N)

-- Theorem statement
theorem car_comparison :
  ∃ (t : ℝ), t > start_time_N ∧
  distance_M t = distance_N t ∧
  speed_N = 3 * speed_M ∧
  start_time_N - start_time_M = 2 := by
  sorry

end NUMINAMATH_CALUDE_car_comparison_l2579_257902


namespace NUMINAMATH_CALUDE_point_Q_coordinate_l2579_257987

theorem point_Q_coordinate (Q : ℝ) : (|Q - 0| = 3) → (Q = 3 ∨ Q = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_Q_coordinate_l2579_257987


namespace NUMINAMATH_CALUDE_sum_due_calculation_l2579_257954

/-- Given a banker's discount and true discount, calculate the sum due -/
def sum_due (bankers_discount true_discount : ℚ) : ℚ :=
  (true_discount^2) / (bankers_discount - true_discount)

/-- Theorem: The sum due is 2400 when banker's discount is 576 and true discount is 480 -/
theorem sum_due_calculation : sum_due 576 480 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_sum_due_calculation_l2579_257954


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l2579_257995

/-- The area of a regular hexagon inscribed in a circle with area 100π square units -/
theorem inscribed_hexagon_area :
  let circle_area : ℝ := 100 * Real.pi
  let hexagon_area : ℝ := 150 * Real.sqrt 3
  (∃ (r : ℝ), r > 0 ∧ circle_area = Real.pi * r^2) →
  (∃ (s : ℝ), s > 0 ∧ hexagon_area = 6 * (s^2 * Real.sqrt 3 / 4)) →
  hexagon_area = 150 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l2579_257995


namespace NUMINAMATH_CALUDE_power_product_reciprocal_equals_one_l2579_257932

theorem power_product_reciprocal_equals_one :
  (4 : ℝ)^7 * (0.25 : ℝ)^7 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_power_product_reciprocal_equals_one_l2579_257932


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2579_257975

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.a = 2 * t.c)

-- Theorem statement
theorem isosceles_triangle_base_angles 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ t.b = 45) ∨ (t.b = 72 ∧ t.c = 72) ∨ (t.a = 72 ∧ t.c = 72) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2579_257975


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2579_257946

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2579_257946


namespace NUMINAMATH_CALUDE_rectangle_problem_l2579_257929

/-- Given a rectangle with length 4x + 1 and width x + 7, where the area is equal to twice the perimeter,
    the positive value of x is (-9 + √481) / 8. -/
theorem rectangle_problem (x : ℝ) : 
  (4*x + 1) * (x + 7) = 2 * (2*(4*x + 1) + 2*(x + 7)) → 
  x > 0 → 
  x = (-9 + Real.sqrt 481) / 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l2579_257929


namespace NUMINAMATH_CALUDE_negation_equivalence_l2579_257942

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2579_257942


namespace NUMINAMATH_CALUDE_inequality_solution_l2579_257978

noncomputable def f (x : ℝ) : ℝ := x^4 + Real.exp (abs x)

theorem inequality_solution :
  let S := {t : ℝ | 2 * f (Real.log t) - f (Real.log (1 / t)) ≤ f 2}
  S = {t : ℝ | Real.exp (-2) ≤ t ∧ t ≤ Real.exp 2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2579_257978


namespace NUMINAMATH_CALUDE_exists_monthly_increase_factor_l2579_257913

/-- The marathon distance in miles -/
def marathon_distance : ℝ := 26.3

/-- The initial running distance in miles -/
def initial_distance : ℝ := 3

/-- The number of months of training -/
def training_months : ℕ := 5

/-- Theorem stating the existence of a monthly increase factor -/
theorem exists_monthly_increase_factor :
  ∃ x : ℝ, x > 1 ∧ initial_distance * x^(training_months - 1) = marathon_distance :=
sorry

end NUMINAMATH_CALUDE_exists_monthly_increase_factor_l2579_257913


namespace NUMINAMATH_CALUDE_fraction_simplification_l2579_257920

theorem fraction_simplification (y : ℝ) (h : y = 3) :
  (y^8 + 10*y^4 + 25) / (y^4 + 5) = 86 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2579_257920


namespace NUMINAMATH_CALUDE_square_area_ratio_l2579_257962

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := 4 * s₂
  (s₁ * s₁) / (s₂ * s₂) = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2579_257962


namespace NUMINAMATH_CALUDE_min_k_value_l2579_257960

def is_valid (n k : ℕ) : Prop :=
  ∀ i ∈ Finset.range (k - 1), n % (i + 2) = i + 1

theorem min_k_value :
  ∃ (k : ℕ), k > 0 ∧
  (∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n k ∧
    ∀ (m : ℕ), m < n → ¬(is_valid m k)) ∧
  ∀ (j : ℕ), j < k →
    ¬(∃ (n : ℕ), n > 2000 ∧ n < 3000 ∧ is_valid n j ∧
      ∀ (m : ℕ), m < n → ¬(is_valid m j)) ∧
  k = 9 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l2579_257960


namespace NUMINAMATH_CALUDE_boat_distance_is_105_l2579_257907

/-- Given a boat traveling downstream, calculate the distance covered. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance covered by the boat downstream is 105 km. -/
theorem boat_distance_is_105 :
  let boat_speed : ℝ := 16
  let stream_speed : ℝ := 5
  let time : ℝ := 5
  distance_downstream boat_speed stream_speed time = 105 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_is_105_l2579_257907


namespace NUMINAMATH_CALUDE_population_change_l2579_257900

/-- The initial population of a village that underwent several population changes --/
def initial_population : ℕ :=
  -- Define the initial population (to be proved)
  6496

/-- The final population after a series of events --/
def final_population : ℕ :=
  -- Given final population
  4555

/-- Theorem stating the relationship between initial and final population --/
theorem population_change (P : ℕ) :
  P = initial_population →
  (1.10 : ℝ) * ((0.75 : ℝ) * ((0.85 : ℝ) * P)) = final_population := by
  sorry


end NUMINAMATH_CALUDE_population_change_l2579_257900


namespace NUMINAMATH_CALUDE_first_player_can_draw_l2579_257930

/-- Represents a chess position -/
def ChessPosition : Type := Unit

/-- Represents a chess move -/
def ChessMove : Type := Unit

/-- Represents a strategy in double chess -/
def DoubleChessStrategy : Type := ChessPosition → ChessMove × ChessMove

/-- The initial chess position -/
def initialPosition : ChessPosition := sorry

/-- Applies a move to a position, returning the new position -/
def applyMove (pos : ChessPosition) (move : ChessMove) : ChessPosition := sorry

/-- Applies two consecutive moves to a position, returning the new position -/
def applyDoubleMoves (pos : ChessPosition) (moves : ChessMove × ChessMove) : ChessPosition := sorry

/-- Determines if a position is a win for the current player -/
def isWinningPosition (pos : ChessPosition) : Prop := sorry

/-- A knight move that doesn't change the position -/
def neutralKnightMove : ChessMove := sorry

/-- Theorem: The first player in double chess can always force at least a draw -/
theorem first_player_can_draw :
  ∀ (secondPlayerStrategy : DoubleChessStrategy),
  ∃ (firstPlayerStrategy : DoubleChessStrategy),
  ¬(isWinningPosition (applyDoubleMoves (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove)) (secondPlayerStrategy (applyDoubleMoves initialPosition (neutralKnightMove, neutralKnightMove))))) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_draw_l2579_257930


namespace NUMINAMATH_CALUDE_two_digit_square_last_two_digits_l2579_257986

theorem two_digit_square_last_two_digits (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ x^2 % 100 = x % 100 ↔ x = 25 ∨ x = 76 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_square_last_two_digits_l2579_257986


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_PQ_l2579_257973

theorem multiplicative_inverse_of_PQ (P Q : ℕ) (M : ℕ) : 
  P = 123321 → 
  Q = 246642 → 
  M = 69788 → 
  (P * Q * M) % 1000003 = 1 := by
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_PQ_l2579_257973


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2579_257964

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2579_257964


namespace NUMINAMATH_CALUDE_point_translation_l2579_257999

def translate_point (x y dx dy : Int) : (Int × Int) := (x + dx, y + dy)

theorem point_translation :
  let P : (Int × Int) := (-5, 1)
  let P1 := translate_point P.1 P.2 2 0
  let P2 := translate_point P1.1 P1.2 0 (-4)
  P2 = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l2579_257999


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2579_257931

/-- Given a rectangle with length to width ratio of 5:4 and diagonal d, 
    its area A can be expressed as A = (20/41)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 4 ∧ l^2 + w^2 = d^2 ∧ l * w = (20/41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2579_257931


namespace NUMINAMATH_CALUDE_oreilly_triple_8_49_l2579_257925

/-- Definition of an O'Reilly triple -/
def is_oreilly_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) + (b.val : ℝ)^(1/2) = x.val

/-- Theorem: If (8,49,x) is an O'Reilly triple, then x = 9 -/
theorem oreilly_triple_8_49 (x : ℕ+) :
  is_oreilly_triple 8 49 x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_oreilly_triple_8_49_l2579_257925


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2579_257991

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2579_257991


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l2579_257904

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the union of complement of A and B
theorem union_complement_A_B : (Aᶜ) ∪ B = {x : ℝ | 0 < x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_l2579_257904


namespace NUMINAMATH_CALUDE_common_terms_count_l2579_257949

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  start : ℝ
  diff : ℝ
  length : ℕ

/-- Returns the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.start + (n - 1 : ℝ) * seq.diff

/-- Counts the number of common terms between two arithmetic sequences -/
def countCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  (seq1.length).min seq2.length

theorem common_terms_count (seq1 seq2 : ArithmeticSequence) 
  (h1 : seq1.start = 5 ∧ seq1.diff = 3 ∧ seq1.length = 100)
  (h2 : seq2.start = 3 ∧ seq2.diff = 5 ∧ seq2.length = 100) :
  countCommonTerms seq1 seq2 = 20 := by
  sorry

#check common_terms_count

end NUMINAMATH_CALUDE_common_terms_count_l2579_257949


namespace NUMINAMATH_CALUDE_september_first_is_wednesday_l2579_257935

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- The number of lessons for each day of the week -/
def lessonsPerDay (d : DayOfWeek) : Nat :=
  match d with
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 0
  | .Sunday => 0

/-- The total number of lessons in a week -/
def lessonsPerWeek : Nat :=
  (lessonsPerDay .Monday) +
  (lessonsPerDay .Tuesday) +
  (lessonsPerDay .Wednesday) +
  (lessonsPerDay .Thursday) +
  (lessonsPerDay .Friday) +
  (lessonsPerDay .Saturday) +
  (lessonsPerDay .Sunday)

/-- The function to determine the day of the week for September 1 -/
def septemberFirstDay (totalLessons : Nat) : DayOfWeek :=
  sorry

/-- The theorem stating that September 1 falls on a Wednesday -/
theorem september_first_is_wednesday :
  septemberFirstDay 64 = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_september_first_is_wednesday_l2579_257935


namespace NUMINAMATH_CALUDE_milk_carton_volume_l2579_257974

/-- The volume of a rectangular prism with given dimensions -/
def rectangular_prism_volume (width length height : ℝ) : ℝ :=
  width * length * height

/-- Theorem: The volume of a milk carton with given dimensions is 252 cubic centimeters -/
theorem milk_carton_volume :
  rectangular_prism_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_milk_carton_volume_l2579_257974


namespace NUMINAMATH_CALUDE_triangle_side_difference_minimum_l2579_257919

theorem triangle_side_difference_minimum (x : ℝ) : 
  (5/3 < x) →
  (x < 11/3) →
  (x + 6 + (4*x - 1) > x + 10) →
  (x + 6 + (x + 10) > 4*x - 1) →
  ((4*x - 1) + (x + 10) > x + 6) →
  (x + 10 > x + 6) →
  (x + 10 > 4*x - 1) →
  (x + 10) - (x + 6) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_minimum_l2579_257919


namespace NUMINAMATH_CALUDE_fraction_addition_l2579_257968

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2579_257968


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2579_257998

theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : platform_length = 380.04)
  (h3 : crossing_time = 25) :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  ∃ ε > 0, abs (speed_kmh - 72.01) < ε :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2579_257998


namespace NUMINAMATH_CALUDE_divisor_count_not_25323_or_25322_l2579_257970

def sequential_number (n : ℕ) : ℕ :=
  -- Definition of the number formed by writing integers from 1 to n sequentially
  sorry

def count_divisors (n : ℕ) : ℕ :=
  -- Definition to count the number of divisors of n
  sorry

theorem divisor_count_not_25323_or_25322 :
  let N := sequential_number 1975
  (count_divisors N ≠ 25323) ∧ (count_divisors N ≠ 25322) := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_not_25323_or_25322_l2579_257970


namespace NUMINAMATH_CALUDE_third_car_year_l2579_257916

def first_car_year : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_year :
  first_car_year + years_between_first_and_second + years_between_second_and_third = 2000 :=
by sorry

end NUMINAMATH_CALUDE_third_car_year_l2579_257916


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2579_257952

theorem right_triangle_side_length 
  (X Y Z : ℝ) 
  (hypotenuse : ℝ) 
  (right_angle : X = 90) 
  (hyp_length : Y - Z = hypotenuse) 
  (hyp_value : hypotenuse = 13) 
  (tan_cos_relation : Real.tan Z = 3 * Real.cos Y) : 
  X - Y = (2 * Real.sqrt 338) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2579_257952


namespace NUMINAMATH_CALUDE_red_balls_count_l2579_257908

theorem red_balls_count (total_balls : ℕ) (red_probability : ℚ) (h1 : total_balls = 20) (h2 : red_probability = 1/4) :
  (red_probability * total_balls : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2579_257908


namespace NUMINAMATH_CALUDE_equation_solution_l2579_257966

theorem equation_solution : ∃ x : ℝ, 6*x - 4*x = 380 - 10*(x + 2) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2579_257966


namespace NUMINAMATH_CALUDE_smallest_special_integer_l2579_257939

theorem smallest_special_integer (N : ℕ) : N = 793 ↔ 
  N > 1 ∧
  (∀ M : ℕ, M > 1 → 
    (M ≡ 1 [ZMOD 8] ∧
     M ≡ 1 [ZMOD 9] ∧
     (∃ k : ℕ, 8^k ≤ M ∧ M < 2 * 8^k) ∧
     (∃ m : ℕ, 9^m ≤ M ∧ M < 2 * 9^m)) →
    N ≤ M) ∧
  N ≡ 1 [ZMOD 8] ∧
  N ≡ 1 [ZMOD 9] ∧
  (∃ k : ℕ, 8^k ≤ N ∧ N < 2 * 8^k) ∧
  (∃ m : ℕ, 9^m ≤ N ∧ N < 2 * 9^m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_integer_l2579_257939


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2579_257936

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hab : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2579_257936


namespace NUMINAMATH_CALUDE_ratio_equality_l2579_257941

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ c / 4 ≠ 0) : 
  (a + b) / c = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2579_257941


namespace NUMINAMATH_CALUDE_chocolate_chip_per_recipe_l2579_257963

/-- Given that 23 recipes require 46 cups of chocolate chips in total,
    prove that the number of cups of chocolate chips needed for one recipe is 2. -/
theorem chocolate_chip_per_recipe :
  let total_recipes : ℕ := 23
  let total_chips : ℕ := 46
  (total_chips / total_recipes : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_chocolate_chip_per_recipe_l2579_257963


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2579_257926

theorem coefficient_x_squared_expansion : 
  let p : Polynomial ℤ := (X + 1)^5 * (X - 2)
  p.coeff 2 = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l2579_257926


namespace NUMINAMATH_CALUDE_sunnydale_walk_home_fraction_l2579_257956

/-- The fraction of students who walk home at Sunnydale Middle School -/
theorem sunnydale_walk_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/5
  let bike_fraction : ℚ := 1/8
  let walk_fraction : ℚ := 1 - (bus_fraction + auto_fraction + bike_fraction)
  walk_fraction = 41/120 := by
  sorry

end NUMINAMATH_CALUDE_sunnydale_walk_home_fraction_l2579_257956


namespace NUMINAMATH_CALUDE_square_hexagon_area_l2579_257996

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hexagon_area : ℝ) :
  square_area = Real.sqrt 3 →
  square_area = s ^ 2 →
  hexagon_area = 3 * Real.sqrt 3 * s ^ 2 / 2 →
  hexagon_area = 9 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_hexagon_area_l2579_257996


namespace NUMINAMATH_CALUDE_intersection_A_B_l2579_257983

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Define set B
def B : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2579_257983


namespace NUMINAMATH_CALUDE_total_bill_sum_l2579_257924

-- Define the variables for each person's bill
variable (alice_bill : ℝ) (bob_bill : ℝ) (charlie_bill : ℝ)

-- Define the conditions
axiom alice_tip : 0.15 * alice_bill = 3
axiom bob_tip : 0.25 * bob_bill = 5
axiom charlie_tip : 0.20 * charlie_bill = 4

-- Theorem statement
theorem total_bill_sum :
  alice_bill + bob_bill + charlie_bill = 60 :=
sorry

end NUMINAMATH_CALUDE_total_bill_sum_l2579_257924


namespace NUMINAMATH_CALUDE_one_third_percent_of_150_l2579_257910

theorem one_third_percent_of_150 : (1 / 3 * 1 / 100) * 150 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_150_l2579_257910


namespace NUMINAMATH_CALUDE_set_partition_real_line_l2579_257909

theorem set_partition_real_line (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) → (A ∩ B = ∅) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_partition_real_line_l2579_257909


namespace NUMINAMATH_CALUDE_exists_divisibility_property_l2579_257901

theorem exists_divisibility_property (n : ℕ+) : ∃ (a b : ℕ+), n ∣ (4 * a^2 + 9 * b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisibility_property_l2579_257901


namespace NUMINAMATH_CALUDE_losing_teams_total_score_l2579_257976

/-- Represents a basketball game between two teams -/
structure Game where
  team1_score : ℕ
  team2_score : ℕ

/-- The total score of a game -/
def Game.total_score (g : Game) : ℕ := g.team1_score + g.team2_score

/-- The margin of victory in a game -/
def Game.margin (g : Game) : ℤ := g.team1_score - g.team2_score

theorem losing_teams_total_score (game1 game2 : Game) 
  (h1 : game1.total_score = 150)
  (h2 : game1.margin = 10)
  (h3 : game2.total_score = 140)
  (h4 : game2.margin = -20) :
  game1.team2_score + game2.team1_score = 130 := by
sorry

end NUMINAMATH_CALUDE_losing_teams_total_score_l2579_257976


namespace NUMINAMATH_CALUDE_ten_person_tournament_matches_l2579_257965

/-- Calculate the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin chess tournament has 45 matches. -/
theorem ten_person_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end NUMINAMATH_CALUDE_ten_person_tournament_matches_l2579_257965


namespace NUMINAMATH_CALUDE_min_value_expression_l2579_257918

theorem min_value_expression (a b c k : ℝ) 
  (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  ∃ (min : ℝ), min = k^2/3 + 2 ∧ 
  ∀ (x : ℝ), ((k*c - a)^2 + (a + c)^2 + (c - a)^2) / c^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2579_257918


namespace NUMINAMATH_CALUDE_convex_curve_properties_l2579_257959

/-- Represents a convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields and properties for a convex curve
  -- This is a simplified representation

/-- Defines the reflection of a curve about a point -/
def reflect (K : ConvexCurve) (O : Point) : ConvexCurve :=
  sorry

/-- Defines the arithmetic mean of two curves -/
def arithmeticMean (K1 K2 : ConvexCurve) : ConvexCurve :=
  sorry

/-- Checks if a curve has a center of symmetry -/
def hasCenterOfSymmetry (K : ConvexCurve) : Prop :=
  sorry

/-- Calculates the diameter of a curve -/
def diameter (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the width of a curve -/
def width (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the length of a curve -/
def length (K : ConvexCurve) : ℝ :=
  sorry

/-- Calculates the area enclosed by a curve -/
def area (K : ConvexCurve) : ℝ :=
  sorry

theorem convex_curve_properties (K : ConvexCurve) (O : Point) :
  let K' := reflect K O
  let K_star := arithmeticMean K K'
  (hasCenterOfSymmetry K_star) ∧
  (diameter K_star = diameter K) ∧
  (width K_star = width K) ∧
  (length K_star = length K) ∧
  (area K_star ≥ area K) :=
by
  sorry

end NUMINAMATH_CALUDE_convex_curve_properties_l2579_257959


namespace NUMINAMATH_CALUDE_janet_initial_lives_l2579_257997

theorem janet_initial_lives :
  ∀ (initial : ℕ),
  (initial - 16 + 32 = 54) →
  initial = 38 :=
by sorry

end NUMINAMATH_CALUDE_janet_initial_lives_l2579_257997


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_3_4_2012_l2579_257957

def arithmetic_sequence_count (a₁ : ℕ) (d : ℕ) (max : ℕ) : ℕ :=
  (max - a₁) / d + 1

theorem arithmetic_sequence_count_3_4_2012 :
  arithmetic_sequence_count 3 4 2012 = 502 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_3_4_2012_l2579_257957


namespace NUMINAMATH_CALUDE_train_length_l2579_257921

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 6 → 
  ∃ (length_m : ℝ), abs (length_m - 100.02) < 0.01 := by sorry

end NUMINAMATH_CALUDE_train_length_l2579_257921


namespace NUMINAMATH_CALUDE_dual_colored_cubes_count_l2579_257980

/-- Represents a cube with colored faces -/
structure ColoredCube where
  size : ℕ
  blue_faces : Fin 3
  red_faces : Fin 3

/-- Counts the number of smaller cubes with both colors when a colored cube is sliced -/
def count_dual_colored_cubes (cube : ColoredCube) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4x4 cube with two opposite blue faces and four red faces
    will have exactly 24 smaller cubes with both colors when sliced into 1x1x1 cubes -/
theorem dual_colored_cubes_count :
  let cube : ColoredCube := ⟨4, 2, 4⟩
  count_dual_colored_cubes cube = 24 := by sorry

end NUMINAMATH_CALUDE_dual_colored_cubes_count_l2579_257980


namespace NUMINAMATH_CALUDE_sequence_problem_l2579_257985

theorem sequence_problem (x : ℕ → ℝ) 
  (h_distinct : ∀ n m, n ≥ 2 → m ≥ 2 → n ≠ m → x n ≠ x m)
  (h_relation : ∀ n, n ≥ 2 → x n = (x (n-1) + 398 * x n + x (n+1)) / 400) :
  Real.sqrt ((x 2023 - x 2) / 2021 * (2022 / (x 2023 - x 1))) + 2021 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2579_257985


namespace NUMINAMATH_CALUDE_bread_sharing_theorem_l2579_257943

/-- Calculates the number of slices each friend eats when sharing bread equally -/
def slices_per_friend (slices_per_loaf : ℕ) (num_friends : ℕ) (num_loaves : ℕ) : ℕ :=
  (slices_per_loaf * num_loaves) / num_friends

/-- Proves that under the given conditions, each friend eats 6 slices of bread -/
theorem bread_sharing_theorem :
  let slices_per_loaf : ℕ := 15
  let num_friends : ℕ := 10
  let num_loaves : ℕ := 4
  slices_per_friend slices_per_loaf num_friends num_loaves = 6 := by
  sorry

end NUMINAMATH_CALUDE_bread_sharing_theorem_l2579_257943


namespace NUMINAMATH_CALUDE_polynomial_sum_l2579_257994

theorem polynomial_sum (h k : ℝ → ℝ) :
  (∀ x, h x + k x = -3 + 2 * x) →
  (∀ x, h x = x^3 - 3 * x^2 - 2) →
  (∀ x, k x = -x^3 + 3 * x^2 + 2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2579_257994


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2579_257992

theorem right_triangle_sides (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k) 
  (h_area : a * b / 2 = 24) :
  a = 6 ∧ b = 8 ∧ c = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2579_257992


namespace NUMINAMATH_CALUDE_product_of_ab_l2579_257915

theorem product_of_ab (a b : ℝ) (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ab_l2579_257915


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2579_257971

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hR : R = 0.6 * P)
  (hN : N = 0.75 * R)
  (hP : P ≠ 0) :
  M / N = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2579_257971


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l2579_257945

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n : ℝ) : ℝ × ℝ := (m + 4, n + 2)

-- Theorem statement
theorem x_coordinate_of_first_point (m n : ℝ) :
  line_equation m n ∧ line_equation (m + 4) (n + 2) → m = 2 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l2579_257945


namespace NUMINAMATH_CALUDE_problem_statement_l2579_257972

theorem problem_statement (a b c m : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2579_257972


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2579_257912

def total_profit : ℝ := 500
def share_difference : ℝ := 100

theorem profit_share_ratio :
  ∀ (x y : ℝ),
  x + y = total_profit →
  x - y = share_difference →
  x / total_profit = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2579_257912


namespace NUMINAMATH_CALUDE_min_value_theorem_l2579_257905

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2579_257905


namespace NUMINAMATH_CALUDE_fourth_root_unity_sum_l2579_257969

theorem fourth_root_unity_sum (ζ : ℂ) (h : ζ^4 = 1) (h_nonreal : ζ ≠ 1 ∧ ζ ≠ -1) :
  (1 - ζ + ζ^3)^4 + (1 + ζ - ζ^3)^4 = -14 - 48 * I :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_unity_sum_l2579_257969


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2579_257934

theorem polynomial_expansion (x : ℝ) : 
  (1 - x^3) * (1 + x^4 - x^5) = 1 - x^3 + x^4 - x^5 - x^7 + x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2579_257934


namespace NUMINAMATH_CALUDE_q_components_l2579_257923

/-- The rank of a rational number -/
def rank (q : ℚ) : ℕ :=
  sorry

/-- The largest rational number less than 1/4 with rank 3 -/
def q : ℚ :=
  sorry

/-- The components of q when expressed as a sum of three unit fractions -/
def a₁ : ℕ := sorry
def a₂ : ℕ := sorry
def a₃ : ℕ := sorry

/-- q is less than 1/4 -/
axiom q_lt_quarter : q < 1/4

/-- q has rank 3 -/
axiom q_rank : rank q = 3

/-- q is the largest such number -/
axiom q_largest (r : ℚ) : r < 1/4 → rank r = 3 → r ≤ q

/-- q is expressed as the sum of three unit fractions -/
axiom q_sum : q = 1/a₁ + 1/a₂ + 1/a₃

/-- Each aᵢ is the smallest positive integer satisfying the condition -/
axiom a₁_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/n → n ≥ a₁
axiom a₂_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/n → n ≥ a₂
axiom a₃_minimal : ∀ n : ℕ, n > 0 → q ≥ 1/a₁ + 1/a₂ + 1/n → n ≥ a₃

theorem q_components : a₁ = 5 ∧ a₂ = 21 ∧ a₃ = 421 :=
  sorry

end NUMINAMATH_CALUDE_q_components_l2579_257923


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_slope_sum_l2579_257911

-- Define the trapezoid ABCD
structure IsoscelesTrapezoid where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ
  D : ℤ × ℤ

-- Define the conditions
def validTrapezoid (t : IsoscelesTrapezoid) : Prop :=
  t.A = (15, 15) ∧
  t.D = (16, 20) ∧
  t.B.1 ≠ t.A.1 ∧ t.B.2 ≠ t.A.2 ∧  -- No horizontal or vertical sides
  t.C.1 ≠ t.D.1 ∧ t.C.2 ≠ t.D.2 ∧
  (t.B.2 - t.A.2) * (t.D.1 - t.C.1) = (t.B.1 - t.A.1) * (t.D.2 - t.C.2) ∧  -- AB || CD
  (t.C.2 - t.B.2) * (t.D.1 - t.A.1) ≠ (t.C.1 - t.B.1) * (t.D.2 - t.A.2) ∧  -- BC not || AD
  (t.D.2 - t.A.2) * (t.C.1 - t.B.1) ≠ (t.D.1 - t.A.1) * (t.C.2 - t.B.2)    -- CD not || AB

-- Define the slope of AB
def slopeAB (t : IsoscelesTrapezoid) : ℚ :=
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1)

-- Define the theorem
theorem isosceles_trapezoid_slope_sum (t : IsoscelesTrapezoid) 
  (h : validTrapezoid t) : 
  ∃ (slopes : List ℚ), (∀ s ∈ slopes, ∃ t' : IsoscelesTrapezoid, validTrapezoid t' ∧ slopeAB t' = s) ∧
                       slopes.sum = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_slope_sum_l2579_257911


namespace NUMINAMATH_CALUDE_three_people_seven_steps_l2579_257979

def staircase_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 3 * (Nat.choose n 3 + Nat.choose 3 1 * Nat.choose n 2)

theorem three_people_seven_steps :
  staircase_arrangements 7 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_people_seven_steps_l2579_257979


namespace NUMINAMATH_CALUDE_class_size_l2579_257903

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2579_257903


namespace NUMINAMATH_CALUDE_french_students_count_l2579_257953

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → german = 22 → both = 9 → neither = 24 → 
  ∃ french : ℕ, french = 41 ∧ french + german - both + neither = total :=
sorry

end NUMINAMATH_CALUDE_french_students_count_l2579_257953


namespace NUMINAMATH_CALUDE_inequality_range_l2579_257940

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2579_257940


namespace NUMINAMATH_CALUDE_fraction_zero_l2579_257922

theorem fraction_zero (x : ℝ) : x = 1/2 → (2*x - 1) / (x + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l2579_257922


namespace NUMINAMATH_CALUDE_sin_difference_equality_l2579_257981

theorem sin_difference_equality : 
  Real.sin (70 * π / 180) * Real.sin (65 * π / 180) - 
  Real.sin (20 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_difference_equality_l2579_257981


namespace NUMINAMATH_CALUDE_sequence_max_term_l2579_257967

def a (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem sequence_max_term :
  ∃ (k : ℕ), k = 7 ∧ a k = 108 ∧ ∀ (n : ℕ), a n ≤ a k :=
sorry

end NUMINAMATH_CALUDE_sequence_max_term_l2579_257967


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l2579_257927

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 2023) → (abs b = 2022) → (a > b) → ((a + b = 1) ∨ (a + b = 4045)) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l2579_257927


namespace NUMINAMATH_CALUDE_tiles_needed_l2579_257989

/-- Given a rectangular room and tiling specifications, calculate the number of tiles needed --/
theorem tiles_needed (room_length room_width tile_size fraction_to_tile : ℝ) 
  (h1 : room_length = 12)
  (h2 : room_width = 20)
  (h3 : tile_size = 1)
  (h4 : fraction_to_tile = 1/6) :
  (room_length * room_width * fraction_to_tile) / tile_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_l2579_257989
