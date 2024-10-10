import Mathlib

namespace binomial_10_9_l695_69522

theorem binomial_10_9 : Nat.choose 10 9 = 10 := by
  sorry

end binomial_10_9_l695_69522


namespace geometric_sequence_sum_l695_69568

theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (a 2 + a 4 + a 6 = 3) →       -- Given condition
  (a 5 + a 7 + a 9 = 24) :=     -- Conclusion to prove
by
  sorry

end geometric_sequence_sum_l695_69568


namespace walker_speed_l695_69531

-- Define the speed of person B
def speed_B : ℝ := 3

-- Define the number of crossings
def num_crossings : ℕ := 5

-- Define the time period in hours
def time_period : ℝ := 1

-- Theorem statement
theorem walker_speed (speed_A : ℝ) : 
  (num_crossings : ℝ) / (speed_A + speed_B) = time_period → 
  speed_A = 2 := by
sorry

end walker_speed_l695_69531


namespace largest_negative_integer_l695_69544

theorem largest_negative_integer : ∀ n : ℤ, n < 0 → n ≤ -1 := by
  sorry

end largest_negative_integer_l695_69544


namespace quadratic_inequality_solution_l695_69561

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ -1 }
  else if a > 0 then { x | -1 ≤ x ∧ x ≤ 2/a }
  else if -2 < a ∧ a < 0 then { x | x ≤ 2/a ∨ x ≥ -1 }
  else if a < -2 then { x | x ≤ -1 ∨ x ≥ 2/a }
  else Set.univ

-- State the theorem
theorem quadratic_inequality_solution (a : ℝ) :
  { x : ℝ | f a x ≤ 0 } = solution_set a :=
sorry

end quadratic_inequality_solution_l695_69561


namespace power_division_rule_l695_69513

theorem power_division_rule (m : ℝ) : m^7 / m^3 = m^4 := by sorry

end power_division_rule_l695_69513


namespace summer_camp_boys_l695_69541

theorem summer_camp_boys (total : ℕ) (teachers : ℕ) (boy_ratio girl_ratio : ℕ) :
  total = 65 →
  teachers = 5 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls + teachers = total ∧
    boys * girl_ratio = girls * boy_ratio ∧
    boys = 26 :=
by sorry

end summer_camp_boys_l695_69541


namespace system_of_equations_l695_69580

theorem system_of_equations (a : ℝ) :
  let x := 2 * a + 3
  let y := -a - 2
  (x > 0 ∧ y ≥ 0) →
  ((-3 < a ∧ a ≤ -2) ∧
   (a = -5/3 → x = y) ∧
   (a = -2 → x + y = 5 + a)) :=
by sorry

end system_of_equations_l695_69580


namespace min_value_theorem_l695_69581

noncomputable section

variables (a m n : ℝ)

-- Define the function f
def f (x : ℝ) := a^(x - 1) - 2

-- State the conditions
axiom a_pos : a > 0
axiom a_neq_one : a ≠ 1
axiom m_pos : m > 0
axiom n_pos : n > 0

-- Define the fixed point A
def A : ℝ × ℝ := (1, -1)

-- State that A lies on the line mx - ny - 1 = 0
axiom A_on_line : m * A.1 - n * A.2 - 1 = 0

-- State the theorem to be proved
theorem min_value_theorem : 
  (∀ x : ℝ, f x = f (A.1) → x = A.1) → 
  (∃ (m' n' : ℝ), m' > 0 ∧ n' > 0 ∧ m' * A.1 - n' * A.2 - 1 = 0 ∧ 1/m' + 2/n' < 1/m + 2/n) → 
  1/m + 2/n ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end

end min_value_theorem_l695_69581


namespace sqrt_sum_inequality_l695_69543

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 11 < Real.sqrt 3 + Real.sqrt 10 := by
  sorry

end sqrt_sum_inequality_l695_69543


namespace eighth_fibonacci_term_l695_69500

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term : fibonacci 7 = 21 := by
  sorry

end eighth_fibonacci_term_l695_69500


namespace system_solution_l695_69536

theorem system_solution (x y : ℝ) : 
  (6 * (1 - x)^2 = 1 / y ∧ 6 * (1 - y)^2 = 1 / x) ↔ 
  ((x = 3/2 ∧ y = 2/3) ∨ 
   (x = 2/3 ∧ y = 3/2) ∨ 
   (x = (1/6) * (4 + 2^(2/3) + 2^(4/3)) ∧ y = (1/6) * (4 + 2^(2/3) + 2^(4/3)))) :=
by sorry

end system_solution_l695_69536


namespace baseball_cost_l695_69592

/-- The cost of a baseball given the cost of a football, total payment, and change received. -/
theorem baseball_cost (football_cost change_received total_payment : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : change_received = 4.05)
  (h3 : total_payment = 20) : 
  total_payment - change_received - football_cost = 6.81 := by
  sorry

end baseball_cost_l695_69592


namespace window_width_calculation_l695_69596

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3
def total_cost : ℝ := 2718

theorem window_width_calculation (W : ℝ) :
  (2 * (room_length * room_height + room_width * room_height) -
   door_length * door_width - num_windows * W * window_height) * cost_per_sqft = total_cost →
  W = 4 := by sorry

end window_width_calculation_l695_69596


namespace inequality_solution_l695_69533

theorem inequality_solution (x : ℝ) : 2 ≤ (3*x)/(3*x-7) ∧ (3*x)/(3*x-7) < 6 ↔ 7/3 < x ∧ x < 42/15 := by
  sorry

end inequality_solution_l695_69533


namespace ellipse_sum_is_twelve_l695_69507

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ  -- x-coordinate of center
  k : ℝ  -- y-coordinate of center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- The sum of center coordinates and semi-axes lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: The sum of center coordinates and semi-axes lengths for the given ellipse is 12 -/
theorem ellipse_sum_is_twelve : 
  let e : Ellipse := { h := 3, k := -2, a := 7, b := 4 }
  ellipse_sum e = 12 := by
  sorry

end ellipse_sum_is_twelve_l695_69507


namespace custom_polynomial_value_l695_69591

/-- Custom multiplication operation -/
def star_mult (x y : ℕ) : ℕ := (x + 1) * (y + 1)

/-- Custom squaring operation -/
def star_square (x : ℕ) : ℕ := star_mult x x

/-- The main theorem to prove -/
theorem custom_polynomial_value :
  3 * (star_square 2) - 2 * 2 + 1 = 32 := by sorry

end custom_polynomial_value_l695_69591


namespace solve_windows_problem_l695_69519

def windows_problem (installed : ℕ) (hours_per_window : ℕ) (remaining_hours : ℕ) : Prop :=
  let remaining := remaining_hours / hours_per_window
  installed + remaining = 9

theorem solve_windows_problem :
  windows_problem 6 6 18 := by
  sorry

end solve_windows_problem_l695_69519


namespace intersection_in_first_quadrant_l695_69560

/-- Two lines intersect in the first quadrant if and only if k is in the open interval (-2/3, 2) -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + k + 2 ∧ y = -2 * x + 4) ↔ 
  -2/3 < k ∧ k < 2 :=
sorry

end intersection_in_first_quadrant_l695_69560


namespace b_joined_after_ten_months_l695_69530

/-- Represents the business scenario --/
structure Business where
  a_investment : ℕ
  b_investment : ℕ
  profit_ratio_a : ℕ
  profit_ratio_b : ℕ
  total_duration : ℕ

/-- Calculates the number of months after which B joined the business --/
def months_before_b_joined (b : Business) : ℕ :=
  b.total_duration - (b.a_investment * b.total_duration * b.profit_ratio_b) / 
    (b.b_investment * b.profit_ratio_a)

/-- Theorem stating that B joined after 10 months --/
theorem b_joined_after_ten_months (b : Business) 
  (h1 : b.a_investment = 3500)
  (h2 : b.b_investment = 31500)
  (h3 : b.profit_ratio_a = 2)
  (h4 : b.profit_ratio_b = 3)
  (h5 : b.total_duration = 12) :
  months_before_b_joined b = 10 := by
  sorry

end b_joined_after_ten_months_l695_69530


namespace jacob_needs_26_more_fish_l695_69562

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier : ℕ) (alex_loss : ℕ) : ℕ :=
  let alex_initial := jacob_initial * alex_multiplier
  let alex_final := alex_initial - alex_loss
  let jacob_target := alex_final + 1
  jacob_target - jacob_initial

theorem jacob_needs_26_more_fish :
  fishing_tournament 8 7 23 = 26 := by
  sorry

end jacob_needs_26_more_fish_l695_69562


namespace fraction_problem_l695_69518

theorem fraction_problem : ∃ x : ℝ, x * (5/9) * (1/2) = 0.11111111111111112 ∧ x = 0.4 := by
  sorry

end fraction_problem_l695_69518


namespace jimmy_stair_climbing_time_jimmy_total_time_l695_69594

/-- The sum of an arithmetic sequence with 5 terms, first term 20, and common difference 5 -/
def arithmetic_sum : ℕ := by sorry

/-- The number of flights Jimmy climbs -/
def num_flights : ℕ := 5

/-- The time taken to climb the first flight -/
def first_flight_time : ℕ := 20

/-- The increase in time for each subsequent flight -/
def time_increase : ℕ := 5

theorem jimmy_stair_climbing_time :
  arithmetic_sum = num_flights * (2 * first_flight_time + (num_flights - 1) * time_increase) / 2 :=
by sorry

theorem jimmy_total_time : arithmetic_sum = 150 := by sorry

end jimmy_stair_climbing_time_jimmy_total_time_l695_69594


namespace geometry_propositions_l695_69587

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def subset (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (α β : Plane), 
    subset m α → perpendicular m β → perpendicular_planes α β) ∧
  (∃ (m : Line) (α β : Plane) (n : Line), 
    subset m α ∧ intersect α β n ∧ perpendicular_planes α β ∧ ¬(perpendicular m n)) ∧
  (∃ (m n : Line) (α β : Plane), 
    subset m α ∧ subset n β ∧ parallel_planes α β ∧ ¬(parallel_lines m n)) ∧
  (∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → subset m β → intersect α β n → parallel_lines m n) := by
  sorry


end geometry_propositions_l695_69587


namespace ship_supplies_l695_69566

/-- Calculates the remaining supplies on a ship given initial amount and usage rates --/
theorem ship_supplies (initial_supply : ℚ) (first_day_usage : ℚ) (next_days_usage : ℚ) :
  initial_supply = 400 ∧ 
  first_day_usage = 2/5 ∧ 
  next_days_usage = 3/5 →
  initial_supply * (1 - first_day_usage) * (1 - next_days_usage) = 96 := by
  sorry

end ship_supplies_l695_69566


namespace lap_time_improvement_is_12_seconds_l695_69597

-- Define the initial condition
def initial_laps : ℕ := 25
def initial_time : ℕ := 50

-- Define the later condition
def later_laps : ℕ := 30
def later_time : ℕ := 54

-- Define the function to calculate lap time in seconds
def lap_time_seconds (laps : ℕ) (time : ℕ) : ℚ :=
  (time * 60) / laps

-- Define the improvement in lap time
def lap_time_improvement : ℚ :=
  lap_time_seconds initial_laps initial_time - lap_time_seconds later_laps later_time

-- Theorem statement
theorem lap_time_improvement_is_12_seconds :
  lap_time_improvement = 12 := by sorry

end lap_time_improvement_is_12_seconds_l695_69597


namespace rectangle_area_l695_69512

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 :=
by sorry

end rectangle_area_l695_69512


namespace total_answer_key_ways_l695_69552

/-- Represents a sequence of true-false answers -/
def TFSequence := List Bool

/-- Represents a sequence of multiple-choice answers -/
def MCSequence := List Nat

/-- Checks if a TFSequence is valid (no more than 3 consecutive true or false answers) -/
def isValidTFSequence (seq : TFSequence) : Bool :=
  sorry

/-- Checks if a MCSequence is valid (no consecutive answers are the same) -/
def isValidMCSequence (seq : MCSequence) : Bool :=
  sorry

/-- Counts the number of valid TFSequences of length 10 -/
def countValidTFSequences : Nat :=
  sorry

/-- Counts the number of valid MCSequences of length 5 with 6 choices each -/
def countValidMCSequences : Nat :=
  sorry

/-- The main theorem stating the total number of ways to write the answer key -/
theorem total_answer_key_ways :
  (countValidTFSequences * countValidMCSequences) =
  (countValidTFSequences * 3750) :=
by
  sorry

end total_answer_key_ways_l695_69552


namespace farm_corn_cobs_l695_69558

theorem farm_corn_cobs (field1_rows field1_cobs_per_row : ℕ)
                       (field2_rows field2_cobs_per_row : ℕ)
                       (field3_rows field3_cobs_per_row : ℕ)
                       (field4_rows field4_cobs_per_row : ℕ)
                       (h1 : field1_rows = 13 ∧ field1_cobs_per_row = 8)
                       (h2 : field2_rows = 16 ∧ field2_cobs_per_row = 12)
                       (h3 : field3_rows = 9 ∧ field3_cobs_per_row = 10)
                       (h4 : field4_rows = 20 ∧ field4_cobs_per_row = 6) :
  field1_rows * field1_cobs_per_row +
  field2_rows * field2_cobs_per_row +
  field3_rows * field3_cobs_per_row +
  field4_rows * field4_cobs_per_row = 506 := by
  sorry

end farm_corn_cobs_l695_69558


namespace coeff_x4_is_negative_30_l695_69508

/-- The coefficient of x^4 in the expansion of (4x^2-2x-5)(x^2+1)^5 -/
def coeff_x4 : ℤ :=
  4 * (Nat.choose 5 3) - 5 * (Nat.choose 5 1)

/-- Theorem stating that the coefficient of x^4 is -30 -/
theorem coeff_x4_is_negative_30 : coeff_x4 = -30 := by
  sorry

end coeff_x4_is_negative_30_l695_69508


namespace cubic_factorization_sum_l695_69547

theorem cubic_factorization_sum (a b c d e : ℤ) : 
  (∀ x, 1728 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 132 := by
sorry

end cubic_factorization_sum_l695_69547


namespace longest_side_of_triangle_l695_69511

/-- 
Given a triangle with sides in the ratio 5 : 6 : 7 and a perimeter of 720 cm,
prove that the longest side has a length of 280 cm.
-/
theorem longest_side_of_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
  sorry

end longest_side_of_triangle_l695_69511


namespace sum_of_roots_equals_seven_l695_69556

theorem sum_of_roots_equals_seven : 
  ∀ (x y : ℝ), x^2 - 7*x + 12 = 0 ∧ y^2 - 7*y + 12 = 0 ∧ x ≠ y → x + y = 7 := by
  sorry

end sum_of_roots_equals_seven_l695_69556


namespace remainder_theorem_l695_69585

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 7

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  q D E F 2 = 5 → q D E F (-2) = 5 := by
  sorry

end remainder_theorem_l695_69585


namespace gear_rotation_l695_69589

/-- Represents a gear in the system -/
structure Gear where
  angle : Real

/-- Represents a system of two meshed gears -/
structure GearSystem where
  left : Gear
  right : Gear

/-- Rotates the left gear by a given angle -/
def rotateLeft (system : GearSystem) (θ : Real) : GearSystem :=
  { left := { angle := system.left.angle + θ },
    right := { angle := system.right.angle - θ } }

/-- Theorem stating that rotating the left gear by θ results in the right gear rotating by -θ -/
theorem gear_rotation (system : GearSystem) (θ : Real) :
  (rotateLeft system θ).right.angle = system.right.angle - θ :=
by sorry

end gear_rotation_l695_69589


namespace unique_half_value_l695_69546

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + 2 * f x

/-- The theorem stating that f(1/2) has only one possible value, which is 1 -/
theorem unique_half_value (f : ℝ → ℝ) (hf : special_function f) : 
  ∃! v : ℝ, f (1/2) = v ∧ v = 1 :=
sorry

end unique_half_value_l695_69546


namespace total_discount_percentage_l695_69574

-- Define the discounts
def initial_discount : ℝ := 0.3
def clearance_discount : ℝ := 0.2

-- Theorem statement
theorem total_discount_percentage : 
  (1 - (1 - initial_discount) * (1 - clearance_discount)) * 100 = 44 := by
  sorry

end total_discount_percentage_l695_69574


namespace math_team_probability_l695_69527

theorem math_team_probability : 
  let team_sizes : List Nat := [6, 8, 9]
  let num_teams : Nat := 3
  let num_cocaptains : Nat := 3
  let prob_select_team : Rat := 1 / num_teams
  let prob_select_cocaptains (n : Nat) : Rat := 6 / (n * (n - 1) * (n - 2))
  (prob_select_team * (team_sizes.map prob_select_cocaptains).sum : Rat) = 1 / 70 := by
  sorry

end math_team_probability_l695_69527


namespace x_percent_of_2x_is_10_l695_69563

theorem x_percent_of_2x_is_10 (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * (2 * x) = 10) : 
  x = 10 * Real.sqrt 5 := by
sorry

end x_percent_of_2x_is_10_l695_69563


namespace quadratic_common_root_l695_69555

-- Define the quadratic functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem quadratic_common_root (a b c : ℝ) :
  (∃! x, f a b c x + g a b c x = 0) →
  (∃ x, f a b c x = 0 ∧ g a b c x = 0) :=
by sorry

end quadratic_common_root_l695_69555


namespace carlo_friday_practice_time_l695_69565

/-- Represents Carlo's practice times for each day of the week -/
structure PracticeTimes where
  M : ℕ  -- Monday
  T : ℕ  -- Tuesday
  W : ℕ  -- Wednesday
  Th : ℕ -- Thursday
  F : ℕ  -- Friday

/-- Conditions for Carlo's practice schedule -/
def valid_practice_schedule (pt : PracticeTimes) : Prop :=
  pt.M = 2 * pt.T ∧
  pt.T = pt.W - 10 ∧
  pt.W = pt.Th + 5 ∧
  pt.Th = 50 ∧
  pt.M + pt.T + pt.W + pt.Th + pt.F = 300

/-- Theorem stating that given the conditions, Carlo should practice 60 minutes on Friday -/
theorem carlo_friday_practice_time (pt : PracticeTimes) 
  (h : valid_practice_schedule pt) : pt.F = 60 := by
  sorry

end carlo_friday_practice_time_l695_69565


namespace unique_number_satisfying_conditions_l695_69559

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_square_sum (n : ℕ) : ℕ := (n / 10)^2 + (n % 10)^2

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
            n / digit_sum n = 3 ∧
            n % digit_sum n = 7 ∧
            digit_square_sum n - digit_product n = n :=
by
  sorry

end unique_number_satisfying_conditions_l695_69559


namespace triangle_area_l695_69582

/-- The area of a triangle with one side of length 12 cm and an adjacent angle of 30° is 36 square centimeters. -/
theorem triangle_area (BC : ℝ) (angle_C : ℝ) : 
  BC = 12 → angle_C = 30 * (π / 180) → 
  (1/2) * BC * (BC * Real.sin angle_C) = 36 := by
  sorry

end triangle_area_l695_69582


namespace nearest_integer_to_power_l695_69553

theorem nearest_integer_to_power : ∃ n : ℤ, 
  n = 3707 ∧ 
  ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end nearest_integer_to_power_l695_69553


namespace average_score_l695_69578

def scores : List ℕ := [65, 67, 76, 82, 85]

theorem average_score : (scores.sum / scores.length : ℚ) = 75 := by
  sorry

end average_score_l695_69578


namespace integral_root_iff_odd_l695_69549

theorem integral_root_iff_odd (n : ℕ) :
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ Odd n :=
sorry

end integral_root_iff_odd_l695_69549


namespace students_in_line_l695_69564

theorem students_in_line (total : ℕ) (behind : ℕ) (h1 : total = 25) (h2 : behind = 13) :
  total - (behind + 1) = 11 := by
  sorry

end students_in_line_l695_69564


namespace min_value_expression_l695_69517

theorem min_value_expression (x : ℝ) (h : x > 0) : 
  4 * x + 1 / x^2 ≥ 5 ∧ ∃ y > 0, 4 * y + 1 / y^2 = 5 := by
  sorry

end min_value_expression_l695_69517


namespace rational_root_count_l695_69506

def polynomial (a₁ : ℤ) (x : ℚ) : ℚ := 12 * x^3 - 4 * x^2 + a₁ * x + 18

def is_possible_root (x : ℚ) : Prop :=
  ∃ (p q : ℤ), x = p / q ∧ 
  (p ∣ 18 ∨ p = 0) ∧ 
  (q ∣ 12 ∧ q ≠ 0)

theorem rational_root_count :
  ∃! (roots : Finset ℚ), 
    (∀ x ∈ roots, is_possible_root x) ∧
    (∀ x, is_possible_root x → x ∈ roots) ∧
    roots.card = 20 :=
sorry

end rational_root_count_l695_69506


namespace max_value_of_g_l695_69537

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end max_value_of_g_l695_69537


namespace specific_polygon_perimeter_l695_69540

/-- The perimeter of a polygon consisting of a rectangle and a right triangle -/
def polygon_perimeter (rect_side1 rect_side2 triangle_hypotenuse : ℝ) : ℝ :=
  2 * (rect_side1 + rect_side2) - rect_side2 + triangle_hypotenuse

/-- Theorem: The perimeter of the specific polygon is 21 units -/
theorem specific_polygon_perimeter :
  polygon_perimeter 6 4 5 = 21 := by
  sorry

#eval polygon_perimeter 6 4 5

end specific_polygon_perimeter_l695_69540


namespace triangle_division_l695_69505

/-- A quadrilateral that is both inscribed in a circle and circumscribed about a circle -/
structure BicentricQuadrilateral where
  -- We don't need to define the structure completely, just its existence
  mk :: (dummy : Unit)

/-- Represents a division of a triangle into bicentric quadrilaterals -/
def TriangleDivision (n : ℕ) := 
  { division : List BicentricQuadrilateral // division.length = n }

/-- The main theorem: any triangle can be divided into n bicentric quadrilaterals for n ≥ 3 -/
theorem triangle_division (n : ℕ) (h : n ≥ 3) : 
  ∃ (division : TriangleDivision n), True :=
sorry

end triangle_division_l695_69505


namespace train_length_l695_69575

/-- The length of a train that crosses a platform of equal length in one minute at 54 km/hr -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 54 → -- speed in km/hr
  time = 1 / 60 → -- time in hours (1 minute = 1/60 hour)
  length = speed * time / 2 → -- distance formula, divided by 2 due to equal lengths
  length = 450 / 1000 -- length in km (450m = 0.45km)
  := by sorry

end train_length_l695_69575


namespace subset_M_l695_69504

def M : Set ℝ := {x | x + 1 > 0}

theorem subset_M : {0} ⊆ M := by sorry

end subset_M_l695_69504


namespace rectangles_4x4_grid_l695_69598

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles_4x4 : ℕ :=
  let horizontal_lines := 5
  let vertical_lines := 5
  (horizontal_lines.choose 2) * (vertical_lines.choose 2)

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_4x4_grid :
  num_rectangles_4x4 = 100 := by
  sorry

end rectangles_4x4_grid_l695_69598


namespace simplify_fraction_product_l695_69569

theorem simplify_fraction_product : (222 : ℚ) / 999 * 111 = 74 := by sorry

end simplify_fraction_product_l695_69569


namespace square_of_binomial_l695_69520

theorem square_of_binomial (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 9 * x^2 + 24 * x + b = (3 * x + c)^2) → b = 16 := by
sorry

end square_of_binomial_l695_69520


namespace tony_money_left_l695_69539

/-- The amount of money Tony has left after purchases at a baseball game. -/
def money_left (initial_amount ticket_cost hot_dog_cost drink_cost cap_cost : ℕ) : ℕ :=
  initial_amount - ticket_cost - hot_dog_cost - drink_cost - cap_cost

/-- Theorem stating that Tony has $13 left after his purchases. -/
theorem tony_money_left : 
  money_left 50 16 5 4 12 = 13 := by
  sorry

end tony_money_left_l695_69539


namespace sphere_cylinder_volume_ratio_l695_69515

theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * r^3) / (Real.pi * r^2 * (2 * r)) = 2 / 3 := by
  sorry

end sphere_cylinder_volume_ratio_l695_69515


namespace parabola_roots_and_point_below_axis_l695_69571

/-- A parabola with a point below the x-axis has two distinct real roots, and the x-coordinate of the point is between these roots. -/
theorem parabola_roots_and_point_below_axis 
  (p q x₀ : ℝ) 
  (h_below : x₀^2 + p*x₀ + q < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (x₁ < x₀) ∧ 
    (x₀ < x₂) ∧ 
    (x₁ ≠ x₂) := by
  sorry

end parabola_roots_and_point_below_axis_l695_69571


namespace complex_power_sum_l695_69501

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^97 + z^98 * z^99 * z^100 + z^101 + z^102 + z^103 = -1 := by
  sorry

end complex_power_sum_l695_69501


namespace boys_on_playground_l695_69503

theorem boys_on_playground (total_children girls : ℕ) 
  (h1 : total_children = 62) 
  (h2 : girls = 35) : 
  total_children - girls = 27 := by
sorry

end boys_on_playground_l695_69503


namespace happy_boys_count_l695_69548

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) (happy_boys_exist : Prop) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 19 →
  total_girls = 41 →
  sad_girls = 4 →
  neutral_boys = 7 →
  happy_boys_exist →
  ∃ (happy_boys : ℕ), happy_boys = 6 ∧ 
    happy_boys + (sad_children - sad_girls) + neutral_boys = total_boys :=
by sorry

end happy_boys_count_l695_69548


namespace hostel_cost_23_days_l695_69542

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 11
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- The cost of staying for 23 days in the student youth hostel is $302.00. -/
theorem hostel_cost_23_days : hostelCost 23 = 302 := by
  sorry

#eval hostelCost 23

end hostel_cost_23_days_l695_69542


namespace sum_of_roots_quadratic_l695_69526

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 3 = 0) → (x₂^2 + 5*x₂ - 3 = 0) → (x₁ + x₂ = -5) :=
by sorry

end sum_of_roots_quadratic_l695_69526


namespace circumradius_area_ratio_not_always_equal_l695_69545

/-- Isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  perimeter : ℝ
  area : ℝ
  circumradius : ℝ

/-- Given two isosceles triangles with distinct sides, prove that the ratio of their circumradii
is not always equal to the ratio of their areas -/
theorem circumradius_area_ratio_not_always_equal
  (I II : IsoscelesTriangle)
  (h_distinct_base : I.base ≠ II.base)
  (h_distinct_side : I.side ≠ II.side) :
  ¬ ∀ (I II : IsoscelesTriangle),
    I.circumradius / II.circumradius = I.area / II.area :=
sorry

end circumradius_area_ratio_not_always_equal_l695_69545


namespace probability_wait_two_minutes_expected_wait_time_l695_69554

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def suitcase_interval : ℕ := 2  -- seconds

-- Part a
theorem probability_wait_two_minutes :
  (Nat.choose 59 9 : ℚ) / (Nat.choose total_suitcases business_suitcases) =
  ↑(Nat.choose 59 9) / ↑(Nat.choose total_suitcases business_suitcases) := by sorry

-- Part b
theorem expected_wait_time :
  (4020 : ℚ) / 11 = 2 * (business_suitcases * (total_suitcases + 1) / (business_suitcases + 1)) := by sorry

end probability_wait_two_minutes_expected_wait_time_l695_69554


namespace max_mn_for_exponential_intersection_max_mn_achieved_l695_69579

/-- The maximum value of mn for a line mx + ny = 1 that intersects
    the graph of y = a^(x-1) at a fixed point, where a > 0 and a ≠ 1 -/
theorem max_mn_for_exponential_intersection (a : ℝ) (m n : ℝ) 
  (ha : a > 0) (ha_ne_one : a ≠ 1) : 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) → m*n ≤ 1/4 := by
  sorry

/-- The maximum value of mn is achieved when m = n = 1/2 -/
theorem max_mn_achieved (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (m n : ℝ), m*n = 1/4 ∧ 
  (∃ (x y : ℝ), y = a^(x-1) ∧ m*x + n*y = 1) := by
  sorry

end max_mn_for_exponential_intersection_max_mn_achieved_l695_69579


namespace apples_given_to_neighbor_l695_69516

def initial_apples : ℕ := 127
def remaining_apples : ℕ := 39

theorem apples_given_to_neighbor :
  initial_apples - remaining_apples = 88 :=
by sorry

end apples_given_to_neighbor_l695_69516


namespace one_sheet_removal_median_l695_69532

/-- Represents a collection of notes with pages and sheets -/
structure Notes where
  total_pages : ℕ
  total_sheets : ℕ
  last_sheet_pages : ℕ
  mk_notes_valid : total_pages = 2 * (total_sheets - 1) + last_sheet_pages

/-- Calculates the median page number after removing sheets -/
def median_after_removal (notes : Notes) (sheets_removed : ℕ) : ℕ :=
  (notes.total_pages - 2 * sheets_removed + 1) / 2

/-- Theorem stating that removing one sheet results in a median of 36 -/
theorem one_sheet_removal_median (notes : Notes)
  (h1 : notes.total_pages = 65)
  (h2 : notes.total_sheets = 33)
  (h3 : notes.last_sheet_pages = 1) :
  median_after_removal notes 1 = 36 := by
  sorry

#check one_sheet_removal_median

end one_sheet_removal_median_l695_69532


namespace sum_squares_50_rings_l695_69551

/-- The number of squares in the nth ring of a square array -/
def squares_in_ring (n : ℕ) : ℕ := 8 * n

/-- The sum of squares from the 1st to the nth ring -/
def sum_squares (n : ℕ) : ℕ := 
  (List.range n).map squares_in_ring |>.sum

/-- Theorem stating that the sum of squares in the first 50 rings is 10200 -/
theorem sum_squares_50_rings : sum_squares 50 = 10200 := by
  sorry

end sum_squares_50_rings_l695_69551


namespace distance_XY_proof_l695_69534

/-- The distance between points X and Y -/
def distance_XY : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The distance Bob has walked when they meet -/
def bob_distance : ℝ := 28

theorem distance_XY_proof :
  distance_XY = yolanda_speed * (bob_distance / bob_speed + time_difference) + bob_distance :=
by sorry

end distance_XY_proof_l695_69534


namespace equation_solution_l695_69583

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
sorry

end equation_solution_l695_69583


namespace common_zero_condition_l695_69573

/-- The first polynomial -/
def P (k : ℝ) (x : ℝ) : ℝ := 1988 * x^2 + k * x + 8891

/-- The second polynomial -/
def Q (k : ℝ) (x : ℝ) : ℝ := 8891 * x^2 + k * x + 1988

/-- Theorem stating the condition for common zeros -/
theorem common_zero_condition (k : ℝ) :
  (∃ x : ℝ, P k x = 0 ∧ Q k x = 0) ↔ (k = 10879 ∨ k = -10879) := by sorry

end common_zero_condition_l695_69573


namespace quadratic_equation_solution_l695_69523

theorem quadratic_equation_solution (x : ℝ) : x^2 - 2*x - 8 = 0 → x = 4 ∨ x = -2 := by
  sorry

end quadratic_equation_solution_l695_69523


namespace filled_circles_in_2009_l695_69538

/-- Represents the cumulative number of circles (both filled and empty) after n filled circles -/
def s (n : ℕ) : ℕ := (n^2 + n) / 2

/-- Represents the pattern where the nth filled circle is followed by n empty circles -/
def circle_pattern (n : ℕ) : ℕ := n + 1

theorem filled_circles_in_2009 : 
  ∃ k : ℕ, k = 63 ∧ s k ≤ 2009 ∧ s (k + 1) > 2009 :=
sorry

end filled_circles_in_2009_l695_69538


namespace spurs_team_size_l695_69535

theorem spurs_team_size :
  ∀ (num_players : ℕ) (basketballs_per_player : ℕ) (total_basketballs : ℕ),
    basketballs_per_player = 11 →
    total_basketballs = 242 →
    total_basketballs = num_players * basketballs_per_player →
    num_players = 22 := by
  sorry

end spurs_team_size_l695_69535


namespace fraction_addition_l695_69525

theorem fraction_addition : (5 / (8/13)) + (4/7) = 487/56 := by
  sorry

end fraction_addition_l695_69525


namespace sum_lent_calculation_l695_69584

-- Define the interest rate and time period
def interest_rate : ℚ := 3 / 100
def time_period : ℕ := 3

-- Define the theorem
theorem sum_lent_calculation (P : ℚ) : 
  P * interest_rate * time_period = P - 1820 → P = 2000 := by
  sorry

end sum_lent_calculation_l695_69584


namespace x_less_than_negative_one_sufficient_not_necessary_l695_69521

theorem x_less_than_negative_one_sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end x_less_than_negative_one_sufficient_not_necessary_l695_69521


namespace cone_volume_l695_69567

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 8) :
  (1 / 3 : ℝ) * π * (slant_height ^ 2 - height ^ 2) * height = 429 * (1 / 3 : ℝ) * π :=
by sorry

end cone_volume_l695_69567


namespace triangle_shape_determination_l695_69550

structure Triangle where
  -- Define a triangle structure
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the different sets of data
def ratioSideToAngleBisector (t : Triangle) : ℝ := sorry
def ratiosOfAngleBisectors (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def midpointsOfSides (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry
def twoSidesAndOppositeAngle (t : Triangle) : (ℝ × ℝ × ℝ) := sorry
def ratioOfTwoAngles (t : Triangle) : ℝ := sorry

-- Define what it means for a set of data to uniquely determine a triangle
def uniquelyDetermines (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

theorem triangle_shape_determination :
  (¬ uniquelyDetermines ratioSideToAngleBisector) ∧
  (uniquelyDetermines ratiosOfAngleBisectors) ∧
  (¬ uniquelyDetermines midpointsOfSides) ∧
  (uniquelyDetermines twoSidesAndOppositeAngle) ∧
  (uniquelyDetermines ratioOfTwoAngles) := by sorry

end triangle_shape_determination_l695_69550


namespace absolute_value_of_negative_three_equals_three_l695_69595

theorem absolute_value_of_negative_three_equals_three : |(-3 : ℝ)| = 3 := by
  sorry

end absolute_value_of_negative_three_equals_three_l695_69595


namespace min_value_z_l695_69529

/-- The function z(x) = 5x^2 + 10x + 20 has a minimum value of 15 -/
theorem min_value_z (x : ℝ) : ∀ y : ℝ, 5 * x^2 + 10 * x + 20 ≥ 15 := by
  sorry

end min_value_z_l695_69529


namespace quadratic_point_ordering_l695_69588

/-- 
Given a quadratic function y = ax² + 6ax - 5 where a > 0, 
and points A(-4, y₁), B(-3, y₂), and C(1, y₃) on this function's graph,
prove that y₂ < y₁ < y₃.
-/
theorem quadratic_point_ordering (a y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (hA : y₁ = a * (-4)^2 + 6 * a * (-4) - 5)
  (hB : y₂ = a * (-3)^2 + 6 * a * (-3) - 5)
  (hC : y₃ = a * 1^2 + 6 * a * 1 - 5) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end quadratic_point_ordering_l695_69588


namespace gwen_spent_nothing_l695_69572

/-- Represents the amount of money Gwen received from her mom -/
def mom_money : ℤ := 8

/-- Represents the amount of money Gwen received from her dad -/
def dad_money : ℤ := 5

/-- Represents the difference in money Gwen has from her mom compared to her dad after spending -/
def difference_after_spending : ℤ := 3

/-- Represents the amount of money Gwen spent -/
def money_spent : ℤ := 0

theorem gwen_spent_nothing :
  (mom_money - money_spent) - (dad_money - money_spent) = difference_after_spending :=
sorry

end gwen_spent_nothing_l695_69572


namespace system_solution_fractional_solution_l695_69577

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  3 * x + y = 7 ∧ 2 * x - y = 3

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ -1 ∧ x ≠ 1 ∧ 1 / (x + 1) = 1 / (x^2 - 1)

-- Theorem for the system of equations
theorem system_solution :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 1 := by
  sorry

-- Theorem for the fractional equation
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = 2 := by
  sorry

end system_solution_fractional_solution_l695_69577


namespace total_shingle_area_l695_69590

/-- Calculate the total square footage of shingles required for a house with a main roof and a porch roof. -/
theorem total_shingle_area (main_roof_base main_roof_height porch_roof_length porch_roof_upper_base porch_roof_lower_base porch_roof_height : ℝ) : 
  main_roof_base = 20.5 →
  main_roof_height = 25 →
  porch_roof_length = 6 →
  porch_roof_upper_base = 2.5 →
  porch_roof_lower_base = 4.5 →
  porch_roof_height = 3 →
  (main_roof_base * main_roof_height + (porch_roof_upper_base + porch_roof_lower_base) * porch_roof_height * 2) = 554.5 := by
  sorry

#check total_shingle_area

end total_shingle_area_l695_69590


namespace friends_average_age_l695_69586

def average_age (m : ℝ) : ℝ := 1.05 * m + 21.6

theorem friends_average_age (m : ℝ) :
  let john := 1.5 * m
  let mary := m
  let tonya := 60
  let sam := 0.8 * tonya
  let carol := 2.75 * m
  (john + mary + tonya + sam + carol) / 5 = average_age m := by
  sorry

end friends_average_age_l695_69586


namespace sum_of_squares_l695_69593

theorem sum_of_squares (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 120) : 
  x^2 + y^2 = 2424 / 49 := by
sorry

end sum_of_squares_l695_69593


namespace inequality_system_solution_l695_69570

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x + 7 > 3*x + 2 ∧ 2*x - 2 < 2*m) ↔ x < 5) →
  m ≥ 4 :=
by sorry

end inequality_system_solution_l695_69570


namespace right_triangle_ab_length_l695_69528

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The length of AC is 225
- The slope of line segment AC is 4/3
Prove that the length of AB is 180.
-/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- ∠B = 90°
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 225) -- Length of AC is 225
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4 / 3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 180 := by
  sorry

end right_triangle_ab_length_l695_69528


namespace inequality_range_l695_69502

theorem inequality_range (P : ℝ) (h : 0 ≤ P ∧ P ≤ 4) :
  (∀ x : ℝ, x^2 + P*x > 4*x + P - 3) ↔ (∀ x : ℝ, x < -1 ∨ x > 3) :=
sorry

end inequality_range_l695_69502


namespace picture_frame_area_l695_69524

theorem picture_frame_area (x y : ℤ) 
  (x_gt_one : x > 1) 
  (y_gt_one : y > 1) 
  (frame_area : (2*x + 4)*(y + 2) - x*y = 45) : 
  x*y = 15 := by
sorry

end picture_frame_area_l695_69524


namespace min_value_expr_l695_69509

theorem min_value_expr (x y : ℝ) (h1 : x > 0) (h2 : y > -1) (h3 : x + y = 1) :
  (x^2 + 3) / x + y^2 / (y + 1) ≥ 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > -1 ∧ x₀ + y₀ = 1 ∧
    (x₀^2 + 3) / x₀ + y₀^2 / (y₀ + 1) = 2 + Real.sqrt 3 :=
by sorry

end min_value_expr_l695_69509


namespace quadratic_inequality_solution_l695_69514

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a = 5 ∧ b = -6 := by
sorry

end quadratic_inequality_solution_l695_69514


namespace opposite_of_negative_five_l695_69599

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end opposite_of_negative_five_l695_69599


namespace train_speed_calculation_l695_69510

/-- Given a train of length 360 meters passing a bridge of length 240 meters in 4 minutes,
    prove that the speed of the train is 2.5 m/s. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (time_minutes : ℝ) :
  train_length = 360 →
  bridge_length = 240 →
  time_minutes = 4 →
  (train_length + bridge_length) / (time_minutes * 60) = 2.5 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l695_69510


namespace hall_breadth_calculation_l695_69576

/-- Proves that given a hall of length 36 meters, paved with 1350 stones each measuring 8 dm by 5 dm, the breadth of the hall is 15 meters. -/
theorem hall_breadth_calculation (hall_length : ℝ) (stone_length : ℝ) (stone_width : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  stone_length = 0.8 →
  stone_width = 0.5 →
  num_stones = 1350 →
  (num_stones * stone_length * stone_width) / hall_length = 15 :=
by sorry

end hall_breadth_calculation_l695_69576


namespace inverse_proportion_percentage_change_l695_69557

theorem inverse_proportion_percentage_change (x y a b : ℝ) (k : ℝ) : 
  x > 0 → y > 0 → 
  (x * y = k) → 
  ((1 + a / 100) * x) * ((1 - b / 100) * y) = k → 
  b = |100 * a / (100 + a)| := by
sorry

end inverse_proportion_percentage_change_l695_69557
