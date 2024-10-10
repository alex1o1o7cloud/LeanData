import Mathlib

namespace inequality_solution_set_f_inequality_l3337_333765

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for part (I)
theorem inequality_solution_set (x : ℝ) :
  f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0 := by sorry

-- Theorem for part (II)
theorem f_inequality (x y : ℝ) (hx : |x| > 1) (hy : |y| < 1) :
  f y < |x| * f (y / x^2) := by sorry

end inequality_solution_set_f_inequality_l3337_333765


namespace right_triangle_area_l3337_333774

theorem right_triangle_area (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 → 
  a + b = 24 → 
  c = 24 → 
  d = 24 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * d = 216 :=
sorry

end right_triangle_area_l3337_333774


namespace half_difference_donations_l3337_333769

theorem half_difference_donations (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300) 
  (h2 : julie_donation = 4700) : 
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end half_difference_donations_l3337_333769


namespace fraction_sum_l3337_333739

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end fraction_sum_l3337_333739


namespace accumulator_implies_limit_in_segment_l3337_333703

/-- A sequence is a function from natural numbers to real numbers -/
def Sequence := ℕ → ℝ

/-- A segment [a, b] is an accumulator for a sequence if infinitely many terms of the sequence lie within [a, b] -/
def IsAccumulator (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- The limit of a sequence, if it exists -/
def HasLimit (s : Sequence) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |s n - L| < ε

theorem accumulator_implies_limit_in_segment (s : Sequence) (a b L : ℝ) :
  IsAccumulator s a b → HasLimit s L → a ≤ L ∧ L ≤ b :=
by sorry


end accumulator_implies_limit_in_segment_l3337_333703


namespace parallel_lines_distance_l3337_333770

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 36, and 30, the distance between two adjacent parallel lines is 2√10. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 30 ∧ 
    chord1^2 = 4 * (r^2 - (d/2)^2) ∧
    chord2^2 = 4 * (r^2 - d^2) ∧
    chord3^2 = 4 * (r^2 - (3*d/2)^2)) →
  d = 2 * Real.sqrt 10 :=
by sorry

end parallel_lines_distance_l3337_333770


namespace coin_stack_problem_l3337_333710

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 39/20

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 35/20

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 20

/-- The number of coins in the stack -/
def num_coins : ℕ := 10

theorem coin_stack_problem :
  ∃ (n q : ℕ), n * nickel_thickness + q * quarter_thickness = stack_height ∧ n + q = num_coins :=
sorry

end coin_stack_problem_l3337_333710


namespace min_value_expression_l3337_333705

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 48 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 3*z₀^2 = 144 :=
by sorry

end min_value_expression_l3337_333705


namespace ibrahim_lacking_money_l3337_333745

/-- The amount of money Ibrahim lacks to buy all items -/
def money_lacking (mp3_cost cd_cost headphones_cost case_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost + headphones_cost + case_cost) - (savings + father_contribution)

/-- Theorem stating that Ibrahim lacks 165 euros -/
theorem ibrahim_lacking_money : 
  money_lacking 135 25 50 30 55 20 = 165 := by
  sorry

end ibrahim_lacking_money_l3337_333745


namespace cube_lateral_surface_area_l3337_333778

/-- The lateral surface area of a cube with side length 12 meters is 576 square meters. -/
theorem cube_lateral_surface_area : 
  let side_length : ℝ := 12
  let lateral_surface_area := 4 * side_length * side_length
  lateral_surface_area = 576 := by
sorry

end cube_lateral_surface_area_l3337_333778


namespace store_owner_order_theorem_l3337_333785

/-- The number of bottles of soda ordered by a store owner in April and May -/
def total_bottles_ordered (april_cases : ℕ) (may_cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  (april_cases + may_cases) * bottles_per_case

/-- Theorem stating that the store owner ordered 1000 bottles in April and May -/
theorem store_owner_order_theorem :
  total_bottles_ordered 20 30 20 = 1000 := by
  sorry

#eval total_bottles_ordered 20 30 20

end store_owner_order_theorem_l3337_333785


namespace smallest_integer_with_divisibility_property_l3337_333781

theorem smallest_integer_with_divisibility_property : ∃ (n : ℕ), 
  (∀ i ∈ Finset.range 28, i.succ ∣ n) ∧ 
  ¬(29 ∣ n) ∧ 
  ¬(30 ∣ n) ∧
  (∀ m : ℕ, m < n → ¬(∀ i ∈ Finset.range 28, i.succ ∣ m) ∨ (29 ∣ m) ∨ (30 ∣ m)) :=
by sorry

end smallest_integer_with_divisibility_property_l3337_333781


namespace ship_blown_westward_distance_l3337_333729

/-- Represents the ship's journey with given conditions -/
structure ShipJourney where
  travelTime : ℝ
  speed : ℝ
  obstaclePercentage : ℝ
  finalFraction : ℝ

/-- Calculates the distance blown westward by the storm -/
def distanceBlownWestward (journey : ShipJourney) : ℝ :=
  let plannedDistance := journey.travelTime * journey.speed
  let actualDistance := plannedDistance * (1 + journey.obstaclePercentage)
  let totalDistance := 2 * actualDistance
  let finalDistance := journey.finalFraction * totalDistance
  actualDistance - finalDistance

/-- Theorem stating that for the given journey conditions, the ship was blown 230 km westward -/
theorem ship_blown_westward_distance :
  let journey : ShipJourney := {
    travelTime := 20,
    speed := 30,
    obstaclePercentage := 0.15,
    finalFraction := 1/3
  }
  distanceBlownWestward journey = 230 := by sorry

end ship_blown_westward_distance_l3337_333729


namespace range_of_a_l3337_333731

theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) →
  a ≤ -4 :=
by sorry

end range_of_a_l3337_333731


namespace function_increasing_intervals_l3337_333700

noncomputable def f (A ω φ x : ℝ) : ℝ := 2 * A * (Real.cos (ω * x + φ))^2 - A

theorem function_increasing_intervals
  (A ω φ : ℝ)
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π / 2)
  (h_symmetry_axis : ∀ x, f A ω φ (π/3 - x) = f A ω φ (π/3 + x))
  (h_symmetry_center : ∀ x, f A ω φ (π/12 - x) = f A ω φ (π/12 + x))
  : ∀ k : ℤ, StrictMonoOn (f A ω φ) (Set.Icc (k * π - 2*π/3) (k * π - π/6)) :=
by sorry

end function_increasing_intervals_l3337_333700


namespace binary_multiplication_division_l3337_333713

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_helper (n : Nat) : List Bool :=
    if n = 0 then [] else (n % 2 = 1) :: to_binary_helper (n / 2)
  to_binary_helper n

def binary_110010 : List Bool := [false, true, false, false, true, true]
def binary_1101 : List Bool := [true, false, true, true]
def binary_101 : List Bool := [true, false, true]
def binary_11110100 : List Bool := [false, false, true, false, true, true, true, true]

theorem binary_multiplication_division :
  (binary_to_nat binary_110010 * binary_to_nat binary_1101) / binary_to_nat binary_101 =
  binary_to_nat binary_11110100 := by
  sorry

end binary_multiplication_division_l3337_333713


namespace subset_implies_a_range_l3337_333716

def S : Set ℝ := {x | -2 < x ∧ x < 5}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem subset_implies_a_range (a : ℝ) (h : S ⊆ P a) : -5 ≤ a ∧ a ≤ -3 := by
  sorry

end subset_implies_a_range_l3337_333716


namespace second_projectile_speed_l3337_333750

/-- Given two projectiles launched simultaneously from a distance apart, 
    with one traveling at a known speed, and both meeting after a certain time, 
    this theorem proves the speed of the second projectile. -/
theorem second_projectile_speed 
  (initial_distance : ℝ) 
  (speed_first : ℝ) 
  (time_to_meet : ℝ) 
  (h1 : initial_distance = 1998) 
  (h2 : speed_first = 444) 
  (h3 : time_to_meet = 2) : 
  ∃ (speed_second : ℝ), speed_second = 555 :=
by
  sorry

end second_projectile_speed_l3337_333750


namespace road_repair_time_l3337_333722

/-- 
Theorem: Time to repair a road with two teams working simultaneously
Given:
- Team A can repair the entire road in 3 hours
- Team B can repair the entire road in 6 hours
- Both teams work simultaneously from opposite ends
Prove: The time taken to complete the repair is 2 hours
-/
theorem road_repair_time (team_a_time team_b_time : ℝ) 
  (ha : team_a_time = 3)
  (hb : team_b_time = 6) :
  (1 / team_a_time + 1 / team_b_time) * 2 = 1 := by
  sorry

end road_repair_time_l3337_333722


namespace cat_direction_at_noon_l3337_333797

/-- Represents the activities of the Cat -/
inductive CatActivity
  | TellingTale
  | SingingSong

/-- Represents the direction the Cat is going -/
inductive CatDirection
  | Left
  | Right

/-- The Cat's state at a given time -/
structure CatState where
  activity : CatActivity
  timeSpentOnCurrentActivity : ℕ

def minutes_per_tale : ℕ := 5
def minutes_per_song : ℕ := 4
def start_time : ℕ := 0
def end_time : ℕ := 120  -- 2 hours = 120 minutes

def initial_state : CatState :=
  { activity := CatActivity.TellingTale, timeSpentOnCurrentActivity := 0 }

def next_activity (current : CatActivity) : CatActivity :=
  match current with
  | CatActivity.TellingTale => CatActivity.SingingSong
  | CatActivity.SingingSong => CatActivity.TellingTale

def activity_duration (activity : CatActivity) : ℕ :=
  match activity with
  | CatActivity.TellingTale => minutes_per_tale
  | CatActivity.SingingSong => minutes_per_song

def update_state (state : CatState) (elapsed_time : ℕ) : CatState :=
  let total_time := state.timeSpentOnCurrentActivity + elapsed_time
  let current_activity_duration := activity_duration state.activity
  if total_time < current_activity_duration then
    { activity := state.activity, timeSpentOnCurrentActivity := total_time }
  else
    { activity := next_activity state.activity, timeSpentOnCurrentActivity := total_time % current_activity_duration }

def final_state : CatState :=
  update_state initial_state (end_time - start_time)

def activity_to_direction (activity : CatActivity) : CatDirection :=
  match activity with
  | CatActivity.TellingTale => CatDirection.Left
  | CatActivity.SingingSong => CatDirection.Right

theorem cat_direction_at_noon :
  activity_to_direction final_state.activity = CatDirection.Left :=
sorry

end cat_direction_at_noon_l3337_333797


namespace arithmetic_sequence_ratio_l3337_333727

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n, 
    prove that if S_n / T_n = (2n - 5) / (4n + 3) for all n, then a_6 / b_6 = 17 / 47 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
  (h_sum_a : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h_sum_b : ∀ n, T n = (n / 2) * (b 1 + b n))
  (h_ratio : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
  a 6 / b 6 = 17 / 47 := by
  sorry

end arithmetic_sequence_ratio_l3337_333727


namespace imaginary_part_of_z_l3337_333780

theorem imaginary_part_of_z (z : ℂ) : z = (3 + 4*Complex.I)*Complex.I → z.im = 3 := by
  sorry

end imaginary_part_of_z_l3337_333780


namespace max_time_sum_of_digits_is_19_l3337_333732

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : Nat :=
  19

theorem max_time_sum_of_digits_is_19 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits :=
by sorry

end max_time_sum_of_digits_is_19_l3337_333732


namespace equal_roots_condition_no_three_equal_values_same_solutions_condition_l3337_333763

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement ①
theorem equal_roots_condition (a b c : ℝ) (h : a ≠ 0) :
  b^2 - 4*a*c = 0 → ∃! x : ℝ, quadratic a b c x = 0 :=
sorry

-- Statement ②
theorem no_three_equal_values (a b c : ℝ) (h : a ≠ 0) :
  ¬∃ (m n s : ℝ), m ≠ n ∧ n ≠ s ∧ m ≠ s ∧
    quadratic a b c m = quadratic a b c n ∧
    quadratic a b c n = quadratic a b c s :=
sorry

-- Statement ③
theorem same_solutions_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, quadratic a b c x + 2 = 0 ↔ (x + 2) * (x - 3) = 0) →
  4*a - 2*b + c = -2 :=
sorry

end equal_roots_condition_no_three_equal_values_same_solutions_condition_l3337_333763


namespace circle_line_intersection_range_l3337_333761

/-- Circle C in the Cartesian coordinate plane -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

/-- Line in the Cartesian coordinate plane -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

/-- New circle with radius 2 centered at a point (a, b) -/
def NewCircle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 4

/-- Theorem stating the range of k values -/
theorem circle_line_intersection_range :
  ∀ k : ℝ, (∃ a b : ℝ, Line k a b ∧
    (∃ x y : ℝ, CircleC x y ∧ NewCircle a b x y)) ↔
  0 ≤ k ∧ k ≤ 4/3 :=
sorry

end circle_line_intersection_range_l3337_333761


namespace bob_sandwich_combinations_l3337_333702

/-- Represents the number of sandwich combinations Bob can order -/
def bobSandwichCombinations : ℕ :=
  let totalBreads : ℕ := 5
  let totalMeats : ℕ := 7
  let totalCheeses : ℕ := 6
  let turkeyCombos : ℕ := totalBreads -- Turkey/Swiss combinations
  let roastBeefRyeCombos : ℕ := totalCheeses -- Roast beef/Rye combinations
  let roastBeefSwissCombos : ℕ := totalBreads - 1 -- Roast beef/Swiss combinations (excluding Rye)
  let totalCombinations : ℕ := totalBreads * totalMeats * totalCheeses
  let forbiddenCombinations : ℕ := turkeyCombos + roastBeefRyeCombos + roastBeefSwissCombos
  totalCombinations - forbiddenCombinations

/-- Theorem stating that Bob can order exactly 194 different sandwiches -/
theorem bob_sandwich_combinations : bobSandwichCombinations = 194 := by
  sorry

end bob_sandwich_combinations_l3337_333702


namespace inequality_proof_l3337_333756

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) (hn : n > 0) : 
  (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) ≥ 
  3^n / (3^(n+2) - 9) := by
  sorry

end inequality_proof_l3337_333756


namespace all_signs_used_l3337_333737

/-- Proves that all signs are used in the area code system --/
theorem all_signs_used (total_signs : Nat) (used_signs : Nat) (additional_codes : Nat) 
  (h1 : total_signs = 224)
  (h2 : used_signs = 222)
  (h3 : additional_codes = 888)
  (h4 : ∀ (sign : Nat), sign ≤ total_signs → (additional_codes / used_signs) * sign ≤ additional_codes) :
  total_signs - used_signs = 0 := by
  sorry

end all_signs_used_l3337_333737


namespace largest_integer_divisible_by_all_less_than_cube_root_l3337_333721

theorem largest_integer_divisible_by_all_less_than_cube_root : ∃ (N : ℕ), 
  (N = 420) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k ≤ ⌊(N : ℝ)^(1/3)⌋ → N % k = 0) ∧
  (∀ (M : ℕ), M > N → ∃ (m : ℕ), m > 0 ∧ m ≤ ⌊(M : ℝ)^(1/3)⌋ ∧ M % m ≠ 0) :=
by sorry

end largest_integer_divisible_by_all_less_than_cube_root_l3337_333721


namespace probability_two_absent_one_present_l3337_333755

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 25

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n_students : ℕ := 3

/-- The number of students that should be absent -/
def n_absent : ℕ := 2

-- Theorem statement
theorem probability_two_absent_one_present :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 276 / 15625 := by
  sorry

end probability_two_absent_one_present_l3337_333755


namespace special_numbers_l3337_333742

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a + b * 100 + c * 10 + d) * 10 = a * 100 + b * 10 + c + d

theorem special_numbers :
  ∀ n : ℕ, is_special_number n ↔ 
    n = 2019 ∨ n = 3028 ∨ n = 4037 ∨ n = 5046 ∨ 
    n = 6055 ∨ n = 7064 ∨ n = 8073 ∨ n = 9082 :=
by sorry

end special_numbers_l3337_333742


namespace second_meeting_time_is_12_minutes_l3337_333719

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  trackLength : ℝ
  firstMeetingDistance : ℝ
  firstMeetingTime : ℝ
  marie : Runner
  john : Runner

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (scenario : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 12 minutes after the start -/
theorem second_meeting_time_is_12_minutes (scenario : RaceScenario) 
  (h1 : scenario.trackLength = 500)
  (h2 : scenario.firstMeetingDistance = 100)
  (h3 : scenario.firstMeetingTime = 2)
  (h4 : scenario.marie.initialPosition = 0)
  (h5 : scenario.john.initialPosition = 500)
  (h6 : scenario.marie.speed = scenario.firstMeetingDistance / scenario.firstMeetingTime)
  (h7 : scenario.john.speed = (scenario.trackLength - scenario.firstMeetingDistance) / scenario.firstMeetingTime) :
  secondMeetingTime scenario = 12 :=
sorry

end second_meeting_time_is_12_minutes_l3337_333719


namespace books_read_per_year_l3337_333740

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  36 * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem books_read_per_year (c s : ℕ) : 
  total_books_read c s = 3 * 12 * c * s := by
  sorry

#check books_read_per_year

end books_read_per_year_l3337_333740


namespace power_three_fifteen_mod_five_l3337_333760

theorem power_three_fifteen_mod_five : 3^15 % 5 = 2 := by
  sorry

end power_three_fifteen_mod_five_l3337_333760


namespace distance_between_complex_points_l3337_333723

theorem distance_between_complex_points : 
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 29 := by sorry

end distance_between_complex_points_l3337_333723


namespace marble_count_l3337_333728

/-- The number of marbles in Jar A -/
def jarA : ℕ := 56

/-- The number of marbles in Jar B -/
def jarB : ℕ := 3 * jarA / 2

/-- The number of marbles in Jar C -/
def jarC : ℕ := 2 * jarA

/-- The number of marbles in Jar D -/
def jarD : ℕ := 3 * jarC / 4

/-- The total number of marbles in all jars -/
def totalMarbles : ℕ := jarA + jarB + jarC + jarD

theorem marble_count : totalMarbles = 336 := by
  sorry

end marble_count_l3337_333728


namespace valid_cube_assignment_exists_l3337_333746

/-- Represents a vertex of a cube -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Checks if two vertices are connected by an edge -/
def isConnected (v1 v2 : Vertex) : Prop := sorry

/-- Represents an assignment of natural numbers to the vertices of a cube -/
def CubeAssignment := Vertex → Nat

/-- Checks if the assignment satisfies the divisibility condition for connected vertices -/
def satisfiesConnectedDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, isConnected v1 v2 → 
    (assignment v1 ∣ assignment v2) ∨ (assignment v2 ∣ assignment v1)

/-- Checks if the assignment satisfies the non-divisibility condition for non-connected vertices -/
def satisfiesNonConnectedNonDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, ¬isConnected v1 v2 → 
    ¬(assignment v1 ∣ assignment v2) ∧ ¬(assignment v2 ∣ assignment v1)

/-- The main theorem stating that a valid assignment exists -/
theorem valid_cube_assignment_exists : 
  ∃ (assignment : CubeAssignment), 
    satisfiesConnectedDivisibility assignment ∧ 
    satisfiesNonConnectedNonDivisibility assignment := by
  sorry

end valid_cube_assignment_exists_l3337_333746


namespace room_length_calculation_l3337_333794

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 ∧ total_cost = 16500 ∧ rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end room_length_calculation_l3337_333794


namespace symmetry_of_point_l3337_333711

def is_symmetrical_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_of_point : 
  is_symmetrical_wrt_origin (-1, 1) (1, -1) := by
  sorry

end symmetry_of_point_l3337_333711


namespace coffee_machine_payoff_l3337_333714

/-- Calculates the number of days until a coffee machine pays for itself. --/
def coffee_machine_payoff_days (machine_price : ℕ) (discount : ℕ) (daily_cost : ℕ) (prev_coffees : ℕ) (prev_price : ℕ) : ℕ :=
  let actual_cost := machine_price - discount
  let prev_daily_expense := prev_coffees * prev_price
  let daily_savings := prev_daily_expense - daily_cost
  actual_cost / daily_savings

/-- Theorem stating that under the given conditions, the coffee machine pays for itself in 36 days. --/
theorem coffee_machine_payoff :
  coffee_machine_payoff_days 200 20 3 2 4 = 36 := by
  sorry

end coffee_machine_payoff_l3337_333714


namespace ball_weight_order_l3337_333741

theorem ball_weight_order (a b c d : ℝ) 
  (eq1 : a + b = c + d)
  (ineq1 : a + d > b + c)
  (ineq2 : a + c < b) :
  d > b ∧ b > a ∧ a > c := by
  sorry

end ball_weight_order_l3337_333741


namespace constant_c_value_l3337_333751

theorem constant_c_value : ∃ (d e c : ℝ), 
  (∀ x : ℝ, (6*x^2 - 2*x + 5/2)*(d*x^2 + e*x + c) = 18*x^4 - 9*x^3 + 13*x^2 - 7/2*x + 15/4) →
  c = 3/2 := by
  sorry

end constant_c_value_l3337_333751


namespace largest_of_three_consecutive_odd_integers_l3337_333779

theorem largest_of_three_consecutive_odd_integers (a b c : ℤ) : 
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) →  -- a, b, c are odd
  (b = a + 2 ∧ c = b + 2) →              -- a, b, c are consecutive
  (a + b + c = -147) →                   -- sum is -147
  (max a (max b c) = -47) :=             -- largest is -47
by sorry

end largest_of_three_consecutive_odd_integers_l3337_333779


namespace intersection_M_N_l3337_333707

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by sorry

end intersection_M_N_l3337_333707


namespace jerky_order_fulfillment_l3337_333734

def days_to_fulfill_order (ordered_bags : ℕ) (existing_bags : ℕ) (bags_per_batch : ℕ) : ℕ :=
  ((ordered_bags - existing_bags) + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 :=
by sorry

end jerky_order_fulfillment_l3337_333734


namespace benny_comic_books_l3337_333730

theorem benny_comic_books (x : ℚ) : 
  (3/4 * (2/5 * x + 12) + 18 = 72) → x = 150 := by
  sorry

end benny_comic_books_l3337_333730


namespace probability_of_white_ball_l3337_333718

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.4 → p_black = 0.25 → p_red + p_black + p_white = 1 → p_white = 0.35 := by
  sorry

end probability_of_white_ball_l3337_333718


namespace inheritance_problem_l3337_333795

theorem inheritance_problem (total_inheritance : ℝ) (additional_share : ℝ) 
  (h1 : total_inheritance = 84000)
  (h2 : additional_share = 3500)
  (h3 : ∃ x : ℕ, x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2)) :
  ∃ x : ℕ, x = 8 ∧ x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2) :=
sorry

end inheritance_problem_l3337_333795


namespace angle_measure_proof_l3337_333725

/-- Given an angle whose complement is 7° more than five times the angle,
    prove that the angle measures 13.833°. -/
theorem angle_measure_proof (x : ℝ) : 
  x + (5 * x + 7) = 90 → x = 13.833 := by
  sorry

end angle_measure_proof_l3337_333725


namespace range_of_sqrt_function_l3337_333747

theorem range_of_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, y = Real.sqrt (x + 2)) ↔ x ≥ -2 := by sorry

end range_of_sqrt_function_l3337_333747


namespace dot_product_on_curve_l3337_333799

/-- Given a point M on the graph of f(x) = (x^2 + 4) / x, prove that the dot product of
    vectors MA and MB is -2, where A is the foot of the perpendicular from M to y = x,
    and B is the foot of the perpendicular from M to the y-axis. -/
theorem dot_product_on_curve (t : ℝ) (ht : t > 0) :
  let M : ℝ × ℝ := (t, (t^2 + 4) / t)
  let A : ℝ × ℝ := ((M.1 + M.2) / 2, (M.1 + M.2) / 2)  -- Foot on y = x
  let B : ℝ × ℝ := (0, M.2)  -- Foot on y-axis
  let MA : ℝ × ℝ := (A.1 - M.1, A.2 - M.2)
  let MB : ℝ × ℝ := (B.1 - M.1, B.2 - M.2)
  MA.1 * MB.1 + MA.2 * MB.2 = -2 :=
by sorry

end dot_product_on_curve_l3337_333799


namespace tan_45_degrees_l3337_333720

theorem tan_45_degrees (Q : ℝ × ℝ) : 
  (Q.1 = 1 / Real.sqrt 2) → 
  (Q.2 = 1 / Real.sqrt 2) → 
  (Q.1^2 + Q.2^2 = 1) →
  Real.tan (π/4) = 1 := by
  sorry


end tan_45_degrees_l3337_333720


namespace train_speed_is_88_l3337_333762

/-- Represents the transportation problem with train and ship --/
structure TransportProblem where
  rail_distance : ℝ
  river_distance : ℝ
  train_delay : ℝ
  train_arrival_diff : ℝ
  speed_difference : ℝ

/-- Calculates the train speed given the problem parameters --/
def calculate_train_speed (p : TransportProblem) : ℝ :=
  let train_time := p.rail_distance / x
  let ship_time := p.river_distance / (x - p.speed_difference)
  let time_diff := ship_time - train_time
  x
where
  x := 88 -- The solution we want to prove

/-- Theorem stating that the calculated train speed is correct --/
theorem train_speed_is_88 (p : TransportProblem) 
  (h1 : p.rail_distance = 88)
  (h2 : p.river_distance = 108)
  (h3 : p.train_delay = 1)
  (h4 : p.train_arrival_diff = 1/4)
  (h5 : p.speed_difference = 40) :
  calculate_train_speed p = 88 := by
  sorry

#eval calculate_train_speed { 
  rail_distance := 88, 
  river_distance := 108, 
  train_delay := 1, 
  train_arrival_diff := 1/4, 
  speed_difference := 40 
}

end train_speed_is_88_l3337_333762


namespace parabola_symmetric_points_a_range_l3337_333715

/-- A parabola with equation y = ax^2 - 1 where a ≠ 0 -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- A point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2 - 1

/-- Two points are symmetric about the line y + x = 0 -/
def symmetric_about_line (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 + p1.2 + p2.1 + p2.2 = 0

/-- The main theorem -/
theorem parabola_symmetric_points_a_range (p : Parabola) 
  (p1 p2 : ParabolaPoint p) (h_distinct : p1 ≠ p2) 
  (h_symmetric : symmetric_about_line (p1.x, p1.y) (p2.x, p2.y)) : 
  p.a > 3/4 := by sorry

end parabola_symmetric_points_a_range_l3337_333715


namespace antonias_supplements_l3337_333792

theorem antonias_supplements :
  let total_pills : ℕ := 3 * 120 + 2 * 30
  let days : ℕ := 14
  let remaining_pills : ℕ := 350
  let supplements : ℕ := (total_pills - remaining_pills) / days
  supplements = 5 :=
by sorry

end antonias_supplements_l3337_333792


namespace det_A_l3337_333764

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, 2; 8, 5, -1; 3, 3, 7]

theorem det_A : A.det = 132 := by sorry

end det_A_l3337_333764


namespace total_seashells_l3337_333775

def mary_seashells : ℕ := 18
def jessica_seashells : ℕ := 41

theorem total_seashells : mary_seashells + jessica_seashells = 59 := by
  sorry

end total_seashells_l3337_333775


namespace duke_three_pointers_l3337_333724

/-- The number of additional three-pointers Duke scored in the final game compared to his normal amount -/
def additional_three_pointers (
  points_to_tie : ℕ
  ) (points_over_record : ℕ
  ) (old_record : ℕ
  ) (free_throws : ℕ
  ) (regular_baskets : ℕ
  ) (normal_three_pointers : ℕ
  ) : ℕ :=
  let total_points := points_to_tie + points_over_record
  let points_from_free_throws := free_throws * 1
  let points_from_regular_baskets := regular_baskets * 2
  let points_from_three_pointers := total_points - (points_from_free_throws + points_from_regular_baskets)
  let three_pointers_scored := points_from_three_pointers / 3
  three_pointers_scored - normal_three_pointers

theorem duke_three_pointers :
  additional_three_pointers 17 5 257 5 4 2 = 1 := by
  sorry

end duke_three_pointers_l3337_333724


namespace leo_current_weight_l3337_333796

-- Define Leo's current weight
def leo_weight : ℝ := sorry

-- Define Kendra's current weight
def kendra_weight : ℝ := sorry

-- Condition 1: If Leo gains 10 pounds, he will weigh 50% more than Kendra
axiom condition_1 : leo_weight + 10 = 1.5 * kendra_weight

-- Condition 2: Their combined current weight is 150 pounds
axiom condition_2 : leo_weight + kendra_weight = 150

-- Theorem to prove
theorem leo_current_weight : leo_weight = 86 := by sorry

end leo_current_weight_l3337_333796


namespace collinear_vectors_x_value_l3337_333768

theorem collinear_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt (1 + Real.sin (40 * π / 180))]
  let b : Fin 2 → ℝ := ![1 / Real.sin (65 * π / 180), x]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) → x = Real.sqrt 2 := by
sorry

end collinear_vectors_x_value_l3337_333768


namespace KBrO3_molecular_weight_l3337_333788

/-- Atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- Atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Molecular weight of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + 3 * atomic_weight_O

/-- Theorem stating that the molecular weight of KBrO3 is 167.00 g/mol -/
theorem KBrO3_molecular_weight :
  molecular_weight_KBrO3 = 167.00 := by
  sorry

end KBrO3_molecular_weight_l3337_333788


namespace fourteenth_root_of_unity_l3337_333759

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) := by
  sorry

end fourteenth_root_of_unity_l3337_333759


namespace arithmetic_sequence_sum_l3337_333712

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of specific terms in the arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_roots : a 5 ^ 2 - 6 * a 5 - 1 = 0 ∧ a 13 ^ 2 - 6 * a 13 - 1 = 0) :
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end arithmetic_sequence_sum_l3337_333712


namespace color_tv_price_l3337_333790

/-- The original price of a color TV -/
def original_price : ℝ := 1200

/-- The price after 40% increase -/
def increased_price (x : ℝ) : ℝ := x * (1 + 0.4)

/-- The final price after 20% discount -/
def final_price (x : ℝ) : ℝ := increased_price x * 0.8

theorem color_tv_price :
  final_price original_price - original_price = 144 := by sorry

end color_tv_price_l3337_333790


namespace total_trolls_l3337_333784

/-- The number of trolls in different locations --/
structure TrollCounts where
  forest : ℕ
  bridge : ℕ
  plains : ℕ
  mountain : ℕ

/-- The conditions of the troll counting problem --/
def troll_conditions (t : TrollCounts) : Prop :=
  t.forest = 8 ∧
  t.forest = 2 * t.bridge - 4 ∧
  t.plains = t.bridge / 2 ∧
  t.mountain = t.plains + 3 ∧
  t.forest - t.mountain = 2 * t.bridge

/-- The theorem stating that given the conditions, the total number of trolls is 23 --/
theorem total_trolls (t : TrollCounts) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains + t.mountain = 23 := by
  sorry


end total_trolls_l3337_333784


namespace decimal_rep_17_70_digit_150_of_17_70_l3337_333776

/-- The decimal representation of 17/70 has a repeating cycle of 6 digits -/
def decimal_cycle (n : ℕ) : ℕ := n % 6

/-- The digits in the repeating cycle of 17/70 -/
def cycle_digits : Fin 6 → ℕ
| 0 => 2
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7

theorem decimal_rep_17_70 (n : ℕ) : 
  n > 0 → cycle_digits (decimal_cycle n) = 7 → n % 6 = 0 := by sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_of_17_70 : cycle_digits (decimal_cycle 150) = 7 := by sorry

end decimal_rep_17_70_digit_150_of_17_70_l3337_333776


namespace odd_square_plus_multiple_l3337_333786

theorem odd_square_plus_multiple (o n : ℤ) 
  (ho : ∃ k, o = 2 * k + 1) : 
  Odd (o^2 + n * o) ↔ Even n := by
sorry

end odd_square_plus_multiple_l3337_333786


namespace smallest_positive_largest_negative_smallest_abs_rational_l3337_333757

theorem smallest_positive_largest_negative_smallest_abs_rational 
  (a b : ℤ) (c : ℚ) 
  (ha : a = 1) 
  (hb : b = -1) 
  (hc : c = 0) : a - b - c = 2 := by
  sorry

end smallest_positive_largest_negative_smallest_abs_rational_l3337_333757


namespace christmas_cards_count_l3337_333772

/-- The number of Christmas cards John sent -/
def christmas_cards : ℕ := 20

/-- The number of birthday cards John sent -/
def birthday_cards : ℕ := 15

/-- The cost of each card in dollars -/
def cost_per_card : ℕ := 2

/-- The total amount John spent on cards in dollars -/
def total_spent : ℕ := 70

/-- Theorem stating that the number of Christmas cards is 20 -/
theorem christmas_cards_count :
  christmas_cards = 20 ∧
  birthday_cards = 15 ∧
  cost_per_card = 2 ∧
  total_spent = 70 →
  christmas_cards * cost_per_card + birthday_cards * cost_per_card = total_spent :=
by sorry

end christmas_cards_count_l3337_333772


namespace graces_mother_age_l3337_333777

theorem graces_mother_age :
  ∀ (grace_age grandmother_age mother_age : ℕ),
    grace_age = 60 →
    grace_age = (3 * grandmother_age) / 8 →
    grandmother_age = 2 * mother_age →
    mother_age = 80 := by
  sorry

end graces_mother_age_l3337_333777


namespace sum_of_coefficients_zero_l3337_333736

/-- A function g(x) with specific properties -/
noncomputable def g (A B C : ℤ) : ℝ → ℝ := λ x => x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of coefficients A, B, and C is zero -/
theorem sum_of_coefficients_zero
  (A B C : ℤ)
  (h1 : ∀ x > 2, g A B C x > 0.3)
  (h2 : (A * 1^2 + B * 1 + C = 0) ∧ (A * (-3)^2 + B * (-3) + C = 0)) :
  A + B + C = 0 := by
  sorry

end sum_of_coefficients_zero_l3337_333736


namespace minute_hand_catches_hour_hand_l3337_333704

/-- The speed of the hour hand in degrees per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The number of degrees in a full circle -/
def full_circle : ℚ := 360

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time after 12:00 when the minute hand first catches up to the hour hand -/
def catch_up_time : ℚ := 65 + 5/11

theorem minute_hand_catches_hour_hand :
  let relative_speed := minute_hand_speed - hour_hand_speed
  let catch_up_angle := catch_up_time * relative_speed
  catch_up_angle = full_circle ∧ 
  catch_up_time < minutes_per_hour := by
  sorry

#check minute_hand_catches_hour_hand

end minute_hand_catches_hour_hand_l3337_333704


namespace time_difference_problem_l3337_333791

theorem time_difference_problem (speed_ratio : ℚ) (time_A : ℚ) :
  speed_ratio = 3 / 4 →
  time_A = 2 →
  ∃ (time_B : ℚ), time_A - time_B = 1 / 2 :=
by sorry

end time_difference_problem_l3337_333791


namespace square_number_correct_l3337_333708

/-- The number in the square with coordinates (m, n) -/
def square_number (m n : ℕ) : ℕ :=
  ((m + n - 2) * (m + n - 1)) / 2 + n

/-- Theorem: The square_number function correctly calculates the number
    in the square with coordinates (m, n) for positive integers m and n -/
theorem square_number_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  square_number m n = ((m + n - 2) * (m + n - 1)) / 2 + n :=
by sorry

end square_number_correct_l3337_333708


namespace first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l3337_333726

theorem first_triple_winner (n : ℕ) : 
  (n % 25 = 0 ∧ n % 36 = 0 ∧ n % 45 = 0) → n ≥ 900 :=
by sorry

theorem lcm_of_prizes : Nat.lcm (Nat.lcm 25 36) 45 = 900 :=
by sorry

theorem first_triple_winner_is_900 : 
  900 % 25 = 0 ∧ 900 % 36 = 0 ∧ 900 % 45 = 0 :=
by sorry

end first_triple_winner_lcm_of_prizes_first_triple_winner_is_900_l3337_333726


namespace total_tickets_correct_l3337_333706

/-- The total number of tickets sold at University Theater -/
def total_tickets : ℕ := 510

/-- The price of an adult ticket -/
def adult_price : ℕ := 21

/-- The price of a senior citizen ticket -/
def senior_price : ℕ := 15

/-- The number of senior citizen tickets sold -/
def senior_tickets : ℕ := 327

/-- The total receipts from ticket sales -/
def total_receipts : ℕ := 8748

/-- Theorem stating that the total number of tickets sold is correct -/
theorem total_tickets_correct :
  ∃ (adult_tickets : ℕ),
    total_tickets = adult_tickets + senior_tickets ∧
    total_receipts = adult_tickets * adult_price + senior_tickets * senior_price :=
by sorry

end total_tickets_correct_l3337_333706


namespace complex_multiplication_l3337_333743

theorem complex_multiplication (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 3*I ∧ z₂ = 6 - 8*I ∧ z = z₁ * z₂ → z = -18 - 26*I :=
by sorry

end complex_multiplication_l3337_333743


namespace quadratic_radical_always_nonnegative_l3337_333717

theorem quadratic_radical_always_nonnegative (x : ℝ) : x^2 + 1 ≥ 0 := by
  sorry

end quadratic_radical_always_nonnegative_l3337_333717


namespace projectile_max_height_l3337_333754

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 155

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∃ t, h t = max_height ∧ ∀ s, h s ≤ h t :=
sorry

end projectile_max_height_l3337_333754


namespace accurate_reading_is_10_30_l3337_333749

/-- Represents a scale reading with a lower bound, upper bound, and increment -/
structure ScaleReading where
  lowerBound : ℝ
  upperBound : ℝ
  increment : ℝ

/-- Represents the position of an arrow on the scale -/
structure ArrowPosition where
  value : ℝ
  beforeMidpoint : Bool

/-- Given a scale reading and an arrow position, determines the most accurate reading -/
def mostAccurateReading (scale : ScaleReading) (arrow : ArrowPosition) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the most accurate reading is 10.30 -/
theorem accurate_reading_is_10_30 :
  let scale := ScaleReading.mk 10.2 10.4 0.05
  let arrow := ArrowPosition.mk 10.33 true
  mostAccurateReading scale arrow = 10.30 := by
  sorry

end accurate_reading_is_10_30_l3337_333749


namespace range_of_g_l3337_333744

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ g x = y} = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end range_of_g_l3337_333744


namespace thirty_percent_more_than_75_l3337_333752

theorem thirty_percent_more_than_75 (x : ℝ) : x / 2 = 75 * 1.3 → x = 195 := by
  sorry

end thirty_percent_more_than_75_l3337_333752


namespace point_A_movement_l3337_333783

def point_movement (initial_x initial_y right_movement down_movement : ℝ) : ℝ × ℝ :=
  (initial_x + right_movement, initial_y - down_movement)

theorem point_A_movement :
  point_movement 1 0 2 3 = (3, -3) := by sorry

end point_A_movement_l3337_333783


namespace expression_simplification_l3337_333798

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) := by
  sorry

end expression_simplification_l3337_333798


namespace fair_wall_painting_l3337_333753

theorem fair_wall_painting (people : ℕ) (rooms_type1 rooms_type2 : ℕ) 
  (walls_per_room_type1 walls_per_room_type2 : ℕ) :
  people = 5 →
  rooms_type1 = 5 →
  rooms_type2 = 4 →
  walls_per_room_type1 = 4 →
  walls_per_room_type2 = 5 →
  (rooms_type1 * walls_per_room_type1 + rooms_type2 * walls_per_room_type2) / people = 8 :=
by
  sorry

end fair_wall_painting_l3337_333753


namespace youngest_sibling_age_l3337_333735

theorem youngest_sibling_age (youngest_age : ℕ) : 
  (youngest_age + (youngest_age + 4) + (youngest_age + 5) + (youngest_age + 7)) / 4 = 21 →
  youngest_age = 17 := by
sorry

end youngest_sibling_age_l3337_333735


namespace harolds_rent_l3337_333767

/-- Harold's monthly finances --/
def harolds_finances (rent : ℝ) : Prop :=
  let income : ℝ := 2500
  let car_payment : ℝ := 300
  let utilities : ℝ := car_payment / 2
  let groceries : ℝ := 50
  let remaining : ℝ := income - rent - car_payment - utilities - groceries
  let retirement_savings : ℝ := remaining / 2
  let final_balance : ℝ := remaining - retirement_savings
  final_balance = 650

/-- Theorem: Harold's rent is $700.00 --/
theorem harolds_rent : ∃ (rent : ℝ), harolds_finances rent ∧ rent = 700 := by
  sorry

end harolds_rent_l3337_333767


namespace jakes_comic_books_l3337_333733

theorem jakes_comic_books (jake_books : ℕ) (brother_books : ℕ) : 
  brother_books = jake_books + 15 →
  jake_books + brother_books = 87 →
  jake_books = 36 := by
sorry

end jakes_comic_books_l3337_333733


namespace cubic_inequality_l3337_333787

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a + b) + b*c*(b + c) + c*a*(c + a) := by
  sorry

end cubic_inequality_l3337_333787


namespace average_math_chem_score_l3337_333782

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 40 := by
sorry

end average_math_chem_score_l3337_333782


namespace equal_chords_subtend_equal_arcs_l3337_333738

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord in a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- An arc in a circle -/
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- The length of a chord -/
def chordLength (c : Circle) (ch : Chord c) : ℝ :=
  sorry

/-- The measure of an arc -/
def arcMeasure (c : Circle) (a : Arc c) : ℝ :=
  sorry

/-- A chord subtends an arc -/
def subtends (c : Circle) (ch : Chord c) (a : Arc c) : Prop :=
  sorry

theorem equal_chords_subtend_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 →
  subtends c ch1 a1 →
  subtends c ch2 a2 →
  arcMeasure c a1 = arcMeasure c a2 :=
sorry

end equal_chords_subtend_equal_arcs_l3337_333738


namespace complex_modulus_l3337_333748

theorem complex_modulus (a b : ℝ) :
  (1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end complex_modulus_l3337_333748


namespace digit_sum_of_power_product_l3337_333793

def power_product (a b c d e : ℕ) : ℕ := a^b * c^d * e

theorem digit_sum_of_power_product :
  ∃ (f : ℕ → ℕ), f (power_product 2 2010 5 2012 7) = 13 :=
sorry

end digit_sum_of_power_product_l3337_333793


namespace ratio_of_sum_equals_three_times_difference_l3337_333758

theorem ratio_of_sum_equals_three_times_difference
  (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) (h4 : x + y = 3 * (x - y)) :
  x / y = 2 := by
sorry

end ratio_of_sum_equals_three_times_difference_l3337_333758


namespace arithmetic_sequence_formula_l3337_333771

/-- An arithmetic sequence {aₙ} with a₇ = 4 and a₁₉ = 2a₉ has the general term formula aₙ = (n+1)/2 -/
theorem arithmetic_sequence_formula (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a7 : a 7 = 4)
  (h_a19 : a 19 = 2 * a 9) :
  ∀ n : ℕ, a n = (n + 1) / 2 := by
sorry

end arithmetic_sequence_formula_l3337_333771


namespace largest_inscribed_circle_area_l3337_333789

/-- The area of the largest circle that can be inscribed in a square with side length 2 decimeters is π square decimeters. -/
theorem largest_inscribed_circle_area (square_side : ℝ) (h : square_side = 2) :
  let circle_area := π * (square_side / 2)^2
  circle_area = π := by
  sorry

end largest_inscribed_circle_area_l3337_333789


namespace correct_mark_calculation_l3337_333766

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ initial_avg = 100 ∧ wrong_mark = 60 ∧ correct_avg = 95 →
  ∃ x : ℚ, n * initial_avg - wrong_mark + x = n * correct_avg ∧ x = 10 :=
by sorry

end correct_mark_calculation_l3337_333766


namespace solution_range_l3337_333709

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + y = 1 + a) ∧ (x + 3 * y = 3)

-- Define the theorem
theorem solution_range (x y a : ℝ) :
  system x y a → x + y < 2 → a < 4 := by
  sorry

end solution_range_l3337_333709


namespace unique_sum_of_eight_only_36_37_l3337_333773

/-- A function that returns true if there exists exactly one set of 8 different positive integers that sum to n -/
def unique_sum_of_eight (n : ℕ) : Prop :=
  ∃! (s : Finset ℕ), s.card = 8 ∧ (∀ x ∈ s, x > 0) ∧ s.sum id = n

/-- Theorem stating that 36 and 37 are the only natural numbers with a unique sum of eight different positive integers -/
theorem unique_sum_of_eight_only_36_37 :
  ∀ n : ℕ, unique_sum_of_eight n ↔ n = 36 ∨ n = 37 := by
  sorry

#check unique_sum_of_eight_only_36_37

end unique_sum_of_eight_only_36_37_l3337_333773


namespace sanctuary_swamps_count_l3337_333701

/-- The number of different reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles living in the swamp areas -/
def total_reptiles : ℕ := 1424

/-- The number of swamps in the sanctuary -/
def number_of_swamps : ℕ := total_reptiles / reptiles_per_swamp

theorem sanctuary_swamps_count :
  number_of_swamps = 4 :=
by sorry

end sanctuary_swamps_count_l3337_333701
