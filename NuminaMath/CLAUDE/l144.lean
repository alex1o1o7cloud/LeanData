import Mathlib

namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l144_14449

-- Define the streets
inductive Street
| Batman
| Robin
| Joker

-- Define the properties for each street
def termite_ridden_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 1/3
  | Street.Robin => 3/7
  | Street.Joker => 1/2

def collapsing_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 7/10
  | Street.Robin => 4/5
  | Street.Joker => 3/8

-- Theorem to prove
theorem termite_ridden_not_collapsing (s : Street) :
  (termite_ridden_fraction s) * (1 - collapsing_fraction s) =
    match s with
    | Street.Batman => 1/10
    | Street.Robin => 3/35
    | Street.Joker => 5/16
    := by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l144_14449


namespace NUMINAMATH_CALUDE_oranges_picked_sum_l144_14483

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_sum :
  total_oranges = 55 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_sum_l144_14483


namespace NUMINAMATH_CALUDE_element_value_l144_14498

theorem element_value (a : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) → 
  a ∉ ({0, 1, 2} : Set ℕ) → 
  a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_element_value_l144_14498


namespace NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_equality_condition_l144_14478

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 0 < x ∧ x < 2/3}

-- Theorem for the solution set of the inequality
theorem inequality_solution : 
  {x : ℝ | f x + |x + 1| < 2} = solution_set :=
sorry

-- Theorem for the minimum value of (4/m) + (1/n)
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n ≥ 9/2) :=
sorry

-- Theorem for the equality condition
theorem equality_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n = 9/2 ↔ m = 4/3 ∧ n = 2/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_equality_condition_l144_14478


namespace NUMINAMATH_CALUDE_least_K_inequality_l144_14430

theorem least_K_inequality (K : ℝ) : (∀ x y : ℝ, (1 + 20 * x^2) * (1 + 19 * y^2) ≥ K * x * y) ↔ K ≤ 8 * Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_least_K_inequality_l144_14430


namespace NUMINAMATH_CALUDE_exponent_problem_l144_14451

theorem exponent_problem (a : ℝ) (m n : ℤ) 
  (h1 : a ^ m = 2) (h2 : a ^ n = 3) : 
  a ^ (m + n) = 6 ∧ a ^ (m - 2*n) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l144_14451


namespace NUMINAMATH_CALUDE_norris_money_left_l144_14480

def savings_september : ℕ := 29
def savings_october : ℕ := 25
def savings_november : ℕ := 31
def spending_game : ℕ := 75

theorem norris_money_left : 
  savings_september + savings_october + savings_november - spending_game = 10 := by
  sorry

end NUMINAMATH_CALUDE_norris_money_left_l144_14480


namespace NUMINAMATH_CALUDE_prize_points_l144_14408

/-- The number of chocolate bunnies sold -/
def chocolate_bunnies : ℕ := 8

/-- The points per chocolate bunny -/
def points_per_bunny : ℕ := 100

/-- The number of Snickers bars needed -/
def snickers_bars : ℕ := 48

/-- The points per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The total points needed for the prize -/
def total_points : ℕ := 2000

theorem prize_points :
  chocolate_bunnies * points_per_bunny + snickers_bars * points_per_snickers = total_points :=
by sorry

end NUMINAMATH_CALUDE_prize_points_l144_14408


namespace NUMINAMATH_CALUDE_certain_number_calculation_l144_14431

theorem certain_number_calculation (x y : ℝ) :
  x = 77.7 ∧ x = y + 0.11 * y → y = 77.7 / 1.11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l144_14431


namespace NUMINAMATH_CALUDE_M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l144_14469

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M = {x | 0 < x < 2}
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ⊆ N if and only if a ∈ [-2, 2]
theorem M_subset_N_iff_a_in_range : 
  ∀ a : ℝ, M a ⊆ N ↔ a ∈ Set.Icc (-2) 2 := by sorry

-- Additional helper theorems to establish the relationship
theorem N_explicit : N = Set.Icc (-1) 3 := by sorry

theorem M_cases (a : ℝ) : 
  (a < -1 → M a = {x | a + 1 < x ∧ x < 0}) ∧
  (a = -1 → M a = ∅) ∧
  (a > -1 → M a = {x | 0 < x ∧ x < a + 1}) := by sorry

end NUMINAMATH_CALUDE_M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l144_14469


namespace NUMINAMATH_CALUDE_arrange_40521_eq_96_l144_14432

/-- The number of ways to arrange the digits of 40,521 to form a 5-digit number -/
def arrange_40521 : ℕ :=
  let digits : List ℕ := [4, 0, 5, 2, 1]
  let n : ℕ := digits.length
  let non_zero_digits : ℕ := (digits.filter (· ≠ 0)).length
  (n - 1) * Nat.factorial (n - 1)

theorem arrange_40521_eq_96 : arrange_40521 = 96 := by
  sorry

end NUMINAMATH_CALUDE_arrange_40521_eq_96_l144_14432


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l144_14416

def S : Set ℝ := {x | 2 * x + 1 > 0}
def T : Set ℝ := {x | 3 * x - 5 < 0}

theorem set_intersection_theorem :
  S ∩ T = {x : ℝ | -1/2 < x ∧ x < 5/3} :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l144_14416


namespace NUMINAMATH_CALUDE_brick_width_calculation_l144_14427

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := 700 * 600 * 22.5

/-- The number of bricks required -/
def num_bricks : ℕ := 5600

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

theorem brick_width_calculation : 
  wall_volume = (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l144_14427


namespace NUMINAMATH_CALUDE_train_passing_time_l144_14436

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_speed : ℝ) (train_length : ℝ) (man_speed : ℝ) :
  train_speed = 60 →
  train_length = 110 →
  man_speed = 6 →
  ∃ t : ℝ, t > 0 ∧ t < 7 ∧
  t = train_length / ((train_speed + man_speed) * (1000 / 3600)) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l144_14436


namespace NUMINAMATH_CALUDE_ones_digit_of_11_to_46_l144_14492

theorem ones_digit_of_11_to_46 : (11^46 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_11_to_46_l144_14492


namespace NUMINAMATH_CALUDE_bus_passing_theorem_l144_14456

/-- Represents the time in minutes since midnight -/
def Time := ℕ

/-- Represents the direction of the bus -/
inductive Direction
| Austin2SanAntonio
| SanAntonio2Austin

/-- Represents a bus schedule -/
structure BusSchedule where
  start : Time
  interval : ℕ
  direction : Direction

/-- Calculates the number of buses passed during a journey -/
def count_passed_buses (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) (journey_time : ℕ) : ℕ :=
  sorry

/-- Converts time from hour:minute format to minutes since midnight -/
def time_to_minutes (hour : ℕ) (minute : ℕ) : Time :=
  hour * 60 + minute

theorem bus_passing_theorem (sa_schedule : BusSchedule) (austin_schedule : BusSchedule) :
  sa_schedule.start = time_to_minutes 12 15 ∧
  sa_schedule.interval = 30 ∧
  sa_schedule.direction = Direction.SanAntonio2Austin ∧
  austin_schedule.start = time_to_minutes 12 0 ∧
  austin_schedule.interval = 45 ∧
  austin_schedule.direction = Direction.Austin2SanAntonio →
  count_passed_buses sa_schedule austin_schedule (6 * 60) = 9 :=
sorry

end NUMINAMATH_CALUDE_bus_passing_theorem_l144_14456


namespace NUMINAMATH_CALUDE_sum_of_x_satisfying_condition_l144_14477

def X : Finset ℕ := {0, 1, 2}

def g : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 1
| _ => 0

def f : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 0
| _ => 0

theorem sum_of_x_satisfying_condition : 
  (X.filter (fun x => f (g x) > g (f x))).sum id = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_satisfying_condition_l144_14477


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l144_14434

/-- The analysis method in mathematical proofs -/
def analysis_method (method : String) : Prop :=
  method = "starts from the conclusion to be proven and progressively seeks conditions that make the conclusion valid"

/-- The type of conditions sought by a proof method -/
inductive ConditionType
  | Sufficient
  | Necessary
  | NecessaryAndSufficient
  | Equivalent

/-- The conditions sought by a proof method -/
def seeks_conditions (method : String) (condition_type : ConditionType) : Prop :=
  analysis_method method → condition_type = ConditionType.Sufficient

theorem analysis_method_seeks_sufficient_conditions :
  ∀ (method : String),
  analysis_method method →
  seeks_conditions method ConditionType.Sufficient :=
by
  sorry


end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_conditions_l144_14434


namespace NUMINAMATH_CALUDE_triangle_area_angle_relation_l144_14450

theorem triangle_area_angle_relation (a b c : ℝ) (A : ℝ) (S : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (S = (1/4) * (b^2 + c^2 - a^2)) →
  (S = (1/2) * b * c * Real.sin A) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (A = π/4) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_angle_relation_l144_14450


namespace NUMINAMATH_CALUDE_reflection_line_sum_l144_14442

/-- Given a line y = mx + b, if the reflection of point (1,2) across this line is (7,6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 7 ∧ y = 6 ∧ 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧
    (y - 2) = m * (x - 1) ∧
    y = m * x + b) →
  m + b = 8.5 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l144_14442


namespace NUMINAMATH_CALUDE_fish_tank_ratio_l144_14475

/-- Given 3 fish tanks with a total of 100 fish, where one tank has 20 fish
    and the other two have an equal number of fish, prove that the ratio of fish
    in each of the other two tanks to the first tank is 2:1 -/
theorem fish_tank_ratio :
  ∀ (fish_in_other_tanks : ℕ),
  3 * 20 + 2 * fish_in_other_tanks = 100 →
  fish_in_other_tanks = 2 * 20 :=
by sorry

end NUMINAMATH_CALUDE_fish_tank_ratio_l144_14475


namespace NUMINAMATH_CALUDE_calculate_sales_11_to_12_l144_14418

/-- Sales data for a shopping mall during National Day Golden Week promotion -/
structure SalesData where
  sales_9_to_10 : ℝ
  height_ratio_11_to_12 : ℝ

/-- Theorem: Given the sales from 9:00 to 10:00 and the height ratio of the 11:00 to 12:00 bar,
    calculate the sales from 11:00 to 12:00 -/
theorem calculate_sales_11_to_12 (data : SalesData)
    (h1 : data.sales_9_to_10 = 25000)
    (h2 : data.height_ratio_11_to_12 = 4) :
    data.sales_9_to_10 * data.height_ratio_11_to_12 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_sales_11_to_12_l144_14418


namespace NUMINAMATH_CALUDE_computer_preference_ratio_l144_14497

theorem computer_preference_ratio : 
  ∀ (total mac no_pref equal : ℕ),
    total = 210 →
    mac = 60 →
    no_pref = 90 →
    equal = total - (mac + no_pref) →
    equal = mac →
    (equal : ℚ) / mac = 1 := by
  sorry

end NUMINAMATH_CALUDE_computer_preference_ratio_l144_14497


namespace NUMINAMATH_CALUDE_inequality_proof_l144_14470

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l144_14470


namespace NUMINAMATH_CALUDE_subsets_with_three_adjacent_chairs_l144_14443

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs
    arranged in a circle that contain at least three adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged
    in a circle that contain at least three adjacent chairs is 1634 -/
theorem subsets_with_three_adjacent_chairs :
  subsets_with_adjacent_chairs n = 1634 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_three_adjacent_chairs_l144_14443


namespace NUMINAMATH_CALUDE_ratio_composition_l144_14407

theorem ratio_composition (a b c : ℚ) 
  (hab : a / b = 11 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_composition_l144_14407


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l144_14462

/-- Proves that the rate of interest is 4% per annum given the conditions of the problem -/
theorem interest_rate_is_four_percent (principal : ℝ) (simple_interest : ℝ) (time : ℝ) 
  (h1 : simple_interest = principal - 2080)
  (h2 : principal = 2600)
  (h3 : time = 5)
  (h4 : simple_interest = (principal * rate * time) / 100) : rate = 4 := by
  sorry

#check interest_rate_is_four_percent

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l144_14462


namespace NUMINAMATH_CALUDE_monotone_increasing_constraint_l144_14479

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem monotone_increasing_constraint (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) →
  -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_constraint_l144_14479


namespace NUMINAMATH_CALUDE_min_face_sum_l144_14490

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- The sum of any three numbers on the same face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- The sum of numbers on one face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The minimum possible sum of numbers on one face is 16 -/
theorem min_face_sum :
  ∀ (arr : CubeArrangement), ValidArrangement arr →
    ∃ (face : Fin 6), FaceSum arr face = 16 ∧
      ∀ (other_face : Fin 6), FaceSum arr other_face ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_face_sum_l144_14490


namespace NUMINAMATH_CALUDE_thirteen_bead_necklace_l144_14452

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def arrangements (n : ℕ) : ℕ :=
  fibonacci (n + 2) - fibonacci (n - 2)

def circular_arrangements (n : ℕ) : ℕ :=
  (arrangements n - 1) / n + 1

theorem thirteen_bead_necklace :
  circular_arrangements 13 = 41 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_bead_necklace_l144_14452


namespace NUMINAMATH_CALUDE_six_less_than_twice_square_of_four_l144_14404

theorem six_less_than_twice_square_of_four : (2 * 4^2) - 6 = 26 := by
  sorry

end NUMINAMATH_CALUDE_six_less_than_twice_square_of_four_l144_14404


namespace NUMINAMATH_CALUDE_shape_reassembly_l144_14467

/-- Represents a geometric shape with an area -/
structure Shape :=
  (area : ℝ)

/-- Represents the original rectangle -/
def rectangle : Shape :=
  { area := 1 }

/-- Represents the square -/
def square : Shape :=
  { area := 0.5 }

/-- Represents the triangle with a hole -/
def triangleWithHole : Shape :=
  { area := 0.5 }

/-- Represents the two parts after cutting the rectangle -/
def part1 : Shape :=
  { area := 0.5 }

def part2 : Shape :=
  { area := 0.5 }

theorem shape_reassembly :
  (rectangle.area = part1.area + part2.area) ∧
  (square.area = part1.area) ∧
  (triangleWithHole.area = part2.area) := by
  sorry

#check shape_reassembly

end NUMINAMATH_CALUDE_shape_reassembly_l144_14467


namespace NUMINAMATH_CALUDE_quiz_total_points_l144_14405

/-- Represents a quiz with a specified number of questions, where each question after
    the first is worth a fixed number of points more than the preceding question. -/
structure Quiz where
  num_questions : ℕ
  point_increment : ℕ
  third_question_points : ℕ

/-- Calculates the total points for a given quiz. -/
def total_points (q : Quiz) : ℕ :=
  let first_question_points := q.third_question_points - 2 * q.point_increment
  let last_question_points := first_question_points + (q.num_questions - 1) * q.point_increment
  (first_question_points + last_question_points) * q.num_questions / 2

/-- Theorem stating that a quiz with 8 questions, where each question after the first
    is worth 4 points more than the preceding question, and the third question is
    worth 39 points, has a total of 360 points. -/
theorem quiz_total_points :
  ∀ (q : Quiz), q.num_questions = 8 ∧ q.point_increment = 4 ∧ q.third_question_points = 39 →
  total_points q = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_total_points_l144_14405


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l144_14466

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt5 : p > 5) :
  ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ 
  (∀ (r : ℕ), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l144_14466


namespace NUMINAMATH_CALUDE_euclid_middle_school_contest_l144_14473

/-- The number of distinct students preparing for the math contest at Euclid Middle School -/
def total_students (euler_students fibonacci_students gauss_students overlap : ℕ) : ℕ :=
  euler_students + fibonacci_students + gauss_students - overlap

theorem euclid_middle_school_contest :
  let euler_students := 12
  let fibonacci_students := 10
  let gauss_students := 11
  let overlap := 3
  total_students euler_students fibonacci_students gauss_students overlap = 27 := by
  sorry

#eval total_students 12 10 11 3

end NUMINAMATH_CALUDE_euclid_middle_school_contest_l144_14473


namespace NUMINAMATH_CALUDE_jeanne_needs_eight_tickets_l144_14421

/-- The number of tickets needed for the Ferris wheel -/
def ferris_wheel_tickets : ℕ := 5

/-- The number of tickets needed for the roller coaster -/
def roller_coaster_tickets : ℕ := 4

/-- The number of tickets needed for the bumper cars -/
def bumper_cars_tickets : ℕ := 4

/-- The number of tickets Jeanne already has -/
def jeanne_tickets : ℕ := 5

/-- The total number of tickets needed for all three rides -/
def total_tickets_needed : ℕ := ferris_wheel_tickets + roller_coaster_tickets + bumper_cars_tickets

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets_needed : ℕ := total_tickets_needed - jeanne_tickets

theorem jeanne_needs_eight_tickets : additional_tickets_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_needs_eight_tickets_l144_14421


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l144_14454

/-- Represents the worth of apples in terms of pears -/
def apple_worth (apples : ℚ) (pears : ℚ) : Prop :=
  apples = pears

theorem apple_pear_equivalence :
  apple_worth (3/4 * 12) 9 →
  apple_worth (2/3 * 6) 4 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l144_14454


namespace NUMINAMATH_CALUDE_min_distance_to_midpoint_l144_14425

/-- Given a line segment AB with length 4 and a point P satisfying |PA| - |PB| = 3,
    where O is the midpoint of AB, the minimum value of |OP| is 3/2. -/
theorem min_distance_to_midpoint (A B P O : EuclideanSpace ℝ (Fin 2)) :
  dist A B = 4 →
  O = midpoint ℝ A B →
  dist P A - dist P B = 3 →
  ∃ (min_dist : ℝ), min_dist = 3/2 ∧ ∀ Q, dist P A - dist Q B = 3 → dist O Q ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_midpoint_l144_14425


namespace NUMINAMATH_CALUDE_triangle_properties_l144_14493

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define geometric elements
def angleBisector (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def median (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry
def altitude (t : Triangle) : Fin 3 → Set (ℝ × ℝ) := sorry

-- Define properties
def isInside (t : Triangle) (s : Set (ℝ × ℝ)) : Prop := sorry
def isRightTriangle (t : Triangle) : Prop := sorry
def isLine (s : Set (ℝ × ℝ)) : Prop := sorry
def isRay (s : Set (ℝ × ℝ)) : Prop := sorry
def isLineSegment (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem triangle_properties :
  ∃ (t : Triangle),
    (¬ (∀ i : Fin 3, isInside t (angleBisector t i) ∧ isInside t (median t i) ∧ isInside t (altitude t i))) ∧
    (isRightTriangle t → ∃ i j : Fin 3, i ≠ j ∧ altitude t i ≠ altitude t j) ∧
    (∃ i : Fin 3, isInside t (altitude t i)) ∧
    (¬ (∀ i : Fin 3, isLine (altitude t i) ∧ isRay (angleBisector t i) ∧ isLineSegment (median t i))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l144_14493


namespace NUMINAMATH_CALUDE_jamal_cart_books_l144_14445

def books_in_cart (history : ℕ) (fiction : ℕ) (children : ℕ) (children_misplaced : ℕ) 
  (science : ℕ) (science_misplaced : ℕ) (biography : ℕ) (remaining : ℕ) : ℕ :=
  history + fiction + (children - children_misplaced) + (science - science_misplaced) + biography + remaining

theorem jamal_cart_books :
  books_in_cart 15 22 10 5 8 3 12 20 = 79 := by
  sorry

end NUMINAMATH_CALUDE_jamal_cart_books_l144_14445


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_naturals_l144_14441

theorem negation_of_universal_nonnegative_naturals :
  (¬ ∀ (x : ℕ), x ≥ 0) ↔ (∃ (x : ℕ), x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_naturals_l144_14441


namespace NUMINAMATH_CALUDE_cosine_identity_73_47_l144_14460

theorem cosine_identity_73_47 :
  let α : Real := 73 * π / 180
  let β : Real := 47 * π / 180
  (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos α) * (Real.cos β) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_73_47_l144_14460


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l144_14409

/-- 
Given a quantity y divided into three parts proportional to 1, 3, and 5,
the smallest part is equal to y/9.
-/
theorem smallest_part_of_proportional_division (y : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    x₁ + x₂ + x₃ = y ∧ 
    x₂ = 3 * x₁ ∧ 
    x₃ = 5 * x₁ ∧ 
    x₁ = y / 9 ∧
    x₁ ≤ x₂ ∧ 
    x₁ ≤ x₃ := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l144_14409


namespace NUMINAMATH_CALUDE_num_squares_6x6_l144_14415

/-- A square on a grid --/
structure GridSquare where
  size : ℕ
  rotation : Bool  -- False for regular, True for diagonal

/-- The set of all possible non-congruent squares on a 6x6 grid --/
def squares_6x6 : Finset GridSquare := sorry

/-- The number of non-congruent squares on a 6x6 grid --/
theorem num_squares_6x6 : Finset.card squares_6x6 = 75 := by sorry

end NUMINAMATH_CALUDE_num_squares_6x6_l144_14415


namespace NUMINAMATH_CALUDE_mets_fans_count_l144_14435

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def baseball_town (fans : FanCounts) : Prop :=
  fans.yankees * 2 = fans.mets * 3 ∧
  fans.mets * 5 = fans.redsox * 4 ∧
  fans.yankees + fans.mets + fans.redsox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fans : FanCounts) :
  baseball_town fans → fans.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l144_14435


namespace NUMINAMATH_CALUDE_point_minimizing_distance_sum_l144_14487

/-- The point that minimizes the sum of distances to two fixed points on a line --/
theorem point_minimizing_distance_sum 
  (M N P : ℝ × ℝ) 
  (hM : M = (1, 2)) 
  (hN : N = (4, 6)) 
  (hP : P.2 = P.1 - 1) 
  (h_min : ∀ Q : ℝ × ℝ, Q.2 = Q.1 - 1 → 
    dist P M + dist P N ≤ dist Q M + dist Q N) : 
  P = (17/5, 12/5) := by
  sorry

-- where
-- dist : ℝ × ℝ → ℝ × ℝ → ℝ 
-- represents the Euclidean distance between two points

end NUMINAMATH_CALUDE_point_minimizing_distance_sum_l144_14487


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l144_14481

theorem rectangle_perimeter (square_perimeter : ℝ) (num_squares : ℕ) : 
  square_perimeter = 24 →
  num_squares = 3 →
  let square_side := square_perimeter / 4
  let rectangle_length := square_side * num_squares
  let rectangle_width := square_side
  2 * (rectangle_length + rectangle_width) = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l144_14481


namespace NUMINAMATH_CALUDE_battery_price_is_56_l144_14485

/-- The price of a battery given the total cost of four tires and one battery, and the cost of each tire. -/
def battery_price (total_cost : ℕ) (tire_price : ℕ) : ℕ :=
  total_cost - 4 * tire_price

/-- Theorem stating that the battery price is $56 given the conditions. -/
theorem battery_price_is_56 :
  battery_price 224 42 = 56 := by
  sorry

end NUMINAMATH_CALUDE_battery_price_is_56_l144_14485


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l144_14494

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The condition that the function must satisfy for all positive integers x and y -/
def SatisfiesCondition (f : PositiveIntFunction) : Prop :=
  ∀ x y : ℕ+, ∃ k : ℕ, (x : ℤ)^2 - (y : ℤ)^2 + 2*(y : ℤ)*((f x : ℤ) + (f y : ℤ)) = (k : ℤ)^2

/-- The theorem stating that the identity function is the only function satisfying the condition -/
theorem unique_satisfying_function :
  ∃! f : PositiveIntFunction, SatisfiesCondition f ∧ ∀ n : ℕ+, f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l144_14494


namespace NUMINAMATH_CALUDE_hill_climbing_time_l144_14433

theorem hill_climbing_time 
  (descent_time : ℝ) 
  (average_speed_total : ℝ) 
  (average_speed_climbing : ℝ) 
  (h1 : descent_time = 2)
  (h2 : average_speed_total = 3.5)
  (h3 : average_speed_climbing = 2.625) :
  ∃ (climb_time : ℝ), 
    climb_time = 4 ∧ 
    average_speed_total = (2 * average_speed_climbing * climb_time) / (climb_time + descent_time) := by
  sorry

end NUMINAMATH_CALUDE_hill_climbing_time_l144_14433


namespace NUMINAMATH_CALUDE_largest_value_l144_14403

theorem largest_value (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  (a - b > a) ∧ (a - b > a + b) ∧ (a - b > a * b) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l144_14403


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt26_l144_14417

theorem consecutive_integers_around_sqrt26 (n m : ℤ) : 
  (n + 1 = m) → (n < Real.sqrt 26) → (Real.sqrt 26 < m) → (m + n = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt26_l144_14417


namespace NUMINAMATH_CALUDE_intersection_point_on_median_l144_14428

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the necessary functions
def intersectionPoint (ω₁ ω₂ : Circle) : Point :=
  sorry

def sameSideOfLine (p₁ p₂ : Point) (l : Line) : Prop :=
  sorry

def liesOnMedian (p : Point) (a b c : Point) : Prop :=
  sorry

-- Theorem statement
theorem intersection_point_on_median 
  (A B C Y : Point) (ω₁ ω₂ : Circle) (AC : Line) (BM : Line) :
  Y = intersectionPoint ω₁ ω₂ →
  sameSideOfLine Y B AC →
  liesOnMedian Y A B C :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_median_l144_14428


namespace NUMINAMATH_CALUDE_eighteen_percent_of_700_is_126_l144_14401

theorem eighteen_percent_of_700_is_126 : (18 / 100) * 700 = 126 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_700_is_126_l144_14401


namespace NUMINAMATH_CALUDE_sqrt_simplification_l144_14400

theorem sqrt_simplification :
  Real.sqrt (49 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l144_14400


namespace NUMINAMATH_CALUDE_least_three_digit_seven_heavy_l144_14455

def is_seven_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_seven_heavy : 
  (∀ n : ℕ, is_three_digit n → is_seven_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_seven_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_seven_heavy_l144_14455


namespace NUMINAMATH_CALUDE_school_teachers_count_l144_14476

/-- The number of departments in the school -/
def num_departments : ℕ := 7

/-- The number of teachers in each department -/
def teachers_per_department : ℕ := 20

/-- The total number of teachers in the school -/
def total_teachers : ℕ := num_departments * teachers_per_department

theorem school_teachers_count : total_teachers = 140 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l144_14476


namespace NUMINAMATH_CALUDE_pyramid_volume_l144_14426

/-- Given a pyramid with a square base ABCD and vertex P, prove its volume. -/
theorem pyramid_volume (base_area : ℝ) (triangle_ABP_area : ℝ) (triangle_BCP_area : ℝ) (triangle_ADP_area : ℝ)
  (h_base : base_area = 256)
  (h_ABP : triangle_ABP_area = 128)
  (h_BCP : triangle_BCP_area = 80)
  (h_ADP : triangle_ADP_area = 128) :
  ∃ (volume : ℝ), volume = (2048 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l144_14426


namespace NUMINAMATH_CALUDE_extra_marks_for_second_candidate_l144_14458

/-- The total number of marks in the exam -/
def T : ℝ := 300

/-- The passing marks -/
def P : ℝ := 120

/-- The percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the second candidate -/
def second_candidate_percentage : ℝ := 0.45

/-- The number of marks by which the first candidate fails -/
def failing_margin : ℝ := 30

theorem extra_marks_for_second_candidate : 
  second_candidate_percentage * T - P = 15 := by sorry

end NUMINAMATH_CALUDE_extra_marks_for_second_candidate_l144_14458


namespace NUMINAMATH_CALUDE_unique_triple_solution_l144_14453

theorem unique_triple_solution (p q : Nat) (n : Nat) (h_p : Nat.Prime p) (h_q : Nat.Prime q)
    (h_n : n > 1) (h_p_odd : Odd p) (h_q_odd : Odd q)
    (h_cong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
    (h_cong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
    p = 3 ∧ q = 3 := by
  sorry

#check unique_triple_solution

end NUMINAMATH_CALUDE_unique_triple_solution_l144_14453


namespace NUMINAMATH_CALUDE_cubic_quadratic_inequality_l144_14437

theorem cubic_quadratic_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2*y + x^2*z + y^2*z + y^2*x + z^2*x + z^2*y :=
by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_inequality_l144_14437


namespace NUMINAMATH_CALUDE_largest_valid_number_l144_14457

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number :
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l144_14457


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l144_14459

/-- Given a circle with equation (x-2)^2 + y^2 = 4, prove its center and radius -/
theorem circle_center_and_radius :
  let equation := (fun (x y : ℝ) => (x - 2)^2 + y^2 = 4)
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l144_14459


namespace NUMINAMATH_CALUDE_stamp_theorem_l144_14440

/-- Represents the ability to form a value using given stamp denominations -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * n + b * (n + 2)

/-- Theorem stating that for n = 3, any value k ≥ 8 can be formed using stamps of denominations 3 and 5 -/
theorem stamp_theorem :
  ∀ k : ℕ, k ≥ 8 → can_form 3 k :=
by sorry

end NUMINAMATH_CALUDE_stamp_theorem_l144_14440


namespace NUMINAMATH_CALUDE_square_difference_l144_14461

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l144_14461


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l144_14484

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = (5/2 : ℝ))
  (h₃ : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 + Complex.I * Real.sqrt 3 / 5 ∨
  z₁ / z₂ = -1/5 - Complex.I * Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l144_14484


namespace NUMINAMATH_CALUDE_water_bucket_addition_l144_14439

theorem water_bucket_addition (initial_water : ℝ) (added_water : ℝ) :
  initial_water = 3 → added_water = 6.8 → initial_water + added_water = 9.8 :=
by
  sorry

end NUMINAMATH_CALUDE_water_bucket_addition_l144_14439


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l144_14472

theorem fraction_product_simplification :
  let fractions : List Rat := 
    (7 / 3) :: 
    (List.range 124).map (fun n => ((8 * (n + 1) + 7) : ℚ) / (8 * (n + 1) - 1))
  (fractions.prod : ℚ) = 333 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l144_14472


namespace NUMINAMATH_CALUDE_cereal_box_servings_l144_14402

/-- Calculates the number of servings in a cereal box -/
def servings_in_box (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

/-- Theorem: The number of servings in a cereal box that holds 18 cups,
    with each serving being 2 cups, is 9. -/
theorem cereal_box_servings :
  servings_in_box 18 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_servings_l144_14402


namespace NUMINAMATH_CALUDE_sin_315_degrees_l144_14482

theorem sin_315_degrees : 
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l144_14482


namespace NUMINAMATH_CALUDE_number_puzzle_l144_14420

theorem number_puzzle (x y : ℝ) : x = 265 → (x / 5) + y = 61 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l144_14420


namespace NUMINAMATH_CALUDE_total_cost_approx_636_38_l144_14446

def membership_fee (initial_fee : ℝ) (increase_rates : List ℝ) (discount_rates : List ℝ) : ℝ :=
  let fees := List.scanl (λ acc rate => acc * (1 + rate)) initial_fee increase_rates
  let discounted_fees := List.zipWith (λ fee discount => fee * (1 - discount)) fees discount_rates
  discounted_fees.sum

def total_cost : ℝ :=
  membership_fee 80 [0.1, 0.12, 0.14, 0.15, 0.15, 0.15] [0, 0, 0, 0, 0.1, 0.05]

theorem total_cost_approx_636_38 : 
  ∃ ε > 0, abs (total_cost - 636.38) < ε :=
sorry

end NUMINAMATH_CALUDE_total_cost_approx_636_38_l144_14446


namespace NUMINAMATH_CALUDE_total_money_divided_l144_14413

/-- Proves that the total amount of money divided is 1600, given the specified conditions. -/
theorem total_money_divided (x : ℝ) (T : ℝ) : 
  x + (T - x) = T →  -- The money is divided into two parts
  0.06 * x + 0.05 * (T - x) = 85 →  -- The whole annual interest from both parts is 85
  T - x = 1100 →  -- 1100 was lent at approximately 5%
  T = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_money_divided_l144_14413


namespace NUMINAMATH_CALUDE_sound_distance_at_10C_l144_14411

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the table

-- Theorem statement
theorem sound_distance_at_10C (temp : Int) (time : Int) :
  temp = 10 ∧ time = 4 → speed_of_sound temp * time = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sound_distance_at_10C_l144_14411


namespace NUMINAMATH_CALUDE_brothers_reading_percentage_l144_14422

theorem brothers_reading_percentage
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (difference : ℕ)
  (h1 : total_books = 20)
  (h2 : peter_percentage = 2/5)
  (h3 : difference = 6)
  : (↑(peter_percentage * total_books - difference) / total_books : ℚ) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_brothers_reading_percentage_l144_14422


namespace NUMINAMATH_CALUDE_sammy_score_l144_14447

theorem sammy_score (sammy_score : ℕ) (gab_score : ℕ) (cher_score : ℕ) (opponent_score : ℕ) :
  gab_score = 2 * sammy_score →
  cher_score = 2 * gab_score →
  opponent_score = 85 →
  sammy_score + gab_score + cher_score = opponent_score + 55 →
  sammy_score = 20 := by
sorry

end NUMINAMATH_CALUDE_sammy_score_l144_14447


namespace NUMINAMATH_CALUDE_original_speed_before_training_l144_14412

/-- Represents the skipping speed of a person -/
structure SkippingSpeed :=
  (skips : ℕ)
  (minutes : ℕ)

/-- Calculates the skips per minute -/
def skipsPerMinute (speed : SkippingSpeed) : ℚ :=
  speed.skips / speed.minutes

theorem original_speed_before_training
  (after_training : SkippingSpeed)
  (h_doubles : after_training.skips = 700 ∧ after_training.minutes = 5) :
  let before_training := SkippingSpeed.mk (after_training.skips / 2) after_training.minutes
  skipsPerMinute before_training = 70 := by
sorry

end NUMINAMATH_CALUDE_original_speed_before_training_l144_14412


namespace NUMINAMATH_CALUDE_functional_equation_solution_l144_14406

-- Define the property that f must satisfy
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x*u - y*v) + f (x*v + y*u)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_property f →
    (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = (1/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l144_14406


namespace NUMINAMATH_CALUDE_paula_meal_combinations_l144_14486

/-- The number of meat options available --/
def meat_options : ℕ := 3

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Paula must choose --/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The total number of meal combinations Paula can construct --/
def total_meals : ℕ :=
  meat_options * choose vegetable_options vegetables_to_choose * dessert_options

theorem paula_meal_combinations :
  total_meals = 150 :=
sorry

end NUMINAMATH_CALUDE_paula_meal_combinations_l144_14486


namespace NUMINAMATH_CALUDE_simplify_expression_l144_14474

theorem simplify_expression (x : ℝ) : 1 - (2 * (1 - (1 + (1 - (3 - x))))) = -3 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l144_14474


namespace NUMINAMATH_CALUDE_chocolate_chip_cups_per_batch_l144_14495

/-- Given that 23.0 cups of chocolate chips can make 11.5 batches of cookies,
    prove that 2 cups of chocolate chips are needed for one batch. -/
theorem chocolate_chip_cups_per_batch : 
  ∀ (total_cups batches cups_per_batch : ℝ), 
    total_cups = 23.0 → 
    batches = 11.5 → 
    total_cups = batches * cups_per_batch →
    cups_per_batch = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cups_per_batch_l144_14495


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l144_14410

theorem solution_set_of_inequality (x : ℝ) :
  (8 * x^2 + 6 * x ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l144_14410


namespace NUMINAMATH_CALUDE_f_g_zero_range_l144_14438

def f (x : ℝ) : ℝ := sorry

def g (x : ℝ) : ℝ := f x

theorem f_g_zero_range (π : ℝ) (h_π : π > 0) :
  (∀ x ∈ Set.Icc (1 / π) π, f x = f (1 / x)) →
  (∀ x ∈ Set.Icc (1 / π) 1, f x = Real.log x) →
  (∃ x ∈ Set.Icc (1 / π) π, g x = 0) →
  Set.Icc (-π * Real.log π) 0 = {a | g a = 0} := by sorry

end NUMINAMATH_CALUDE_f_g_zero_range_l144_14438


namespace NUMINAMATH_CALUDE_det_specific_matrix_l144_14499

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -1; 0, 3, 2; 5, -1, 3]
  Matrix.det A = 77 := by
sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l144_14499


namespace NUMINAMATH_CALUDE_unique_solution_abc_squared_l144_14468

theorem unique_solution_abc_squared (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_squared_l144_14468


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l144_14491

/-- Given a rectangle with perimeter 60 and a triangle with height 60, 
    if their areas are equal, then the base of the triangle is 20/3 -/
theorem rectangle_triangle_equal_area (rect_width rect_height tri_base : ℝ) : 
  rect_width > 0 → 
  rect_height > 0 → 
  tri_base > 0 → 
  rect_width + rect_height = 30 → 
  rect_width * rect_height = 30 * tri_base → 
  tri_base = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l144_14491


namespace NUMINAMATH_CALUDE_blue_shoes_count_l144_14463

theorem blue_shoes_count (total : ℕ) (purple : ℕ) (h1 : total = 1250) (h2 : purple = 355) :
  ∃ (blue green : ℕ), blue + green + purple = total ∧ green = purple ∧ blue = 540 := by
  sorry

end NUMINAMATH_CALUDE_blue_shoes_count_l144_14463


namespace NUMINAMATH_CALUDE_is_stratified_sampling_l144_14444

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  dividePopulation : Bool
  sampleFromParts : Bool
  proportionalSampling : Bool
  combineSamples : Bool

/-- Definition of stratified sampling -/
def stratifiedSampling : SamplingMethod :=
  { name := "Stratified Sampling",
    dividePopulation := true,
    sampleFromParts := true,
    proportionalSampling := true,
    combineSamples := true }

/-- Theorem stating that a sampling method with specific characteristics is stratified sampling -/
theorem is_stratified_sampling
  (method : SamplingMethod)
  (h1 : method.dividePopulation = true)
  (h2 : method.sampleFromParts = true)
  (h3 : method.proportionalSampling = true)
  (h4 : method.combineSamples = true) :
  method = stratifiedSampling := by
  sorry

#check is_stratified_sampling

end NUMINAMATH_CALUDE_is_stratified_sampling_l144_14444


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l144_14464

/-- Represents the school with students and their pet ownership. -/
structure School where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ
  rabbit_owners : ℕ
  h_no_multiple_pets : cat_owners + dog_owners + rabbit_owners ≤ total_students

/-- Calculates the percentage of students owning at least one pet. -/
def percentage_pet_owners (s : School) : ℚ :=
  (s.cat_owners + s.dog_owners + s.rabbit_owners : ℚ) / s.total_students * 100

/-- Theorem stating that in the given school, 48% of students own at least one pet. -/
theorem pet_ownership_percentage (s : School) 
    (h_total : s.total_students = 500)
    (h_cats : s.cat_owners = 80)
    (h_dogs : s.dog_owners = 120)
    (h_rabbits : s.rabbit_owners = 40) : 
    percentage_pet_owners s = 48 := by sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l144_14464


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l144_14423

/-- Given a function f(x) = ax + b/x + 5 where a ≠ 0 and b ≠ 0, if f(2) = 3, then f(-2) = 7 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = a * x + b / x + 5)
  (h2 : f 2 = 3) :
  f (-2) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l144_14423


namespace NUMINAMATH_CALUDE_paco_cookie_consumption_l144_14424

/-- Represents the number of sweet cookies Paco ate -/
def sweet_cookies_eaten : ℕ := 15

/-- Represents the initial number of sweet cookies Paco had -/
def initial_sweet_cookies : ℕ := 40

/-- Represents the initial number of salty cookies Paco had -/
def initial_salty_cookies : ℕ := 25

/-- Represents the number of salty cookies Paco ate -/
def salty_cookies_eaten : ℕ := 28

theorem paco_cookie_consumption :
  sweet_cookies_eaten = 15 ∧
  initial_sweet_cookies = 40 ∧
  initial_salty_cookies = 25 ∧
  salty_cookies_eaten = 28 ∧
  salty_cookies_eaten = sweet_cookies_eaten + 13 :=
by sorry

end NUMINAMATH_CALUDE_paco_cookie_consumption_l144_14424


namespace NUMINAMATH_CALUDE_shopping_trip_cost_l144_14414

/-- Calculates the total cost of a shopping trip including discounts, taxes, and fees -/
def calculate_total_cost (items : List (ℕ × ℚ)) (discount_rate : ℚ) (sales_tax_rate : ℚ) (local_tax_rate : ℚ) (sustainability_fee : ℚ) : ℚ :=
  let total_before_discount := (items.map (λ (q, p) => q * p)).sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_tax_rate := sales_tax_rate + local_tax_rate
  let tax_amount := discounted_total * total_tax_rate
  let total_with_tax := discounted_total + tax_amount
  total_with_tax + sustainability_fee

theorem shopping_trip_cost :
  let items := [(3, 18), (2, 11), (4, 22), (6, 9), (5, 14), (2, 30), (3, 25)]
  let discount_rate := 0.15
  let sales_tax_rate := 0.05
  let local_tax_rate := 0.02
  let sustainability_fee := 5
  calculate_total_cost items discount_rate sales_tax_rate local_tax_rate sustainability_fee = 389.72 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_l144_14414


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inequality_general_tetrahedron_inequality_l144_14419

/-- Represents a tetrahedron with a triangle inside it -/
structure Tetrahedron where
  /-- Areas of the triangle's projections on the four faces -/
  P : Fin 4 → ℝ
  /-- Areas of the tetrahedron's faces -/
  S : Fin 4 → ℝ
  /-- Condition that all areas are non-negative -/
  all_non_neg : ∀ i, P i ≥ 0 ∧ S i ≥ 0

/-- Theorem for regular tetrahedron -/
theorem regular_tetrahedron_inequality (t : Tetrahedron) (h_regular : ∀ i j, t.S i = t.S j) :
  t.P 0 ≤ t.P 1 + t.P 2 + t.P 3 :=
sorry

/-- Theorem for any tetrahedron -/
theorem general_tetrahedron_inequality (t : Tetrahedron) :
  t.P 0 * t.S 0 ≤ t.P 1 * t.S 1 + t.P 2 * t.S 2 + t.P 3 * t.S 3 :=
sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inequality_general_tetrahedron_inequality_l144_14419


namespace NUMINAMATH_CALUDE_num_ways_to_achieve_18_with_5_dice_l144_14488

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 5

/-- The target sum we're aiming for -/
def targetSum : ℕ := 18

/-- A function that calculates the number of ways to achieve the target sum -/
def numWaysToAchieveSum (faces : ℕ) (dice : ℕ) (sum : ℕ) : ℕ := sorry

theorem num_ways_to_achieve_18_with_5_dice : 
  numWaysToAchieveSum numFaces numDice targetSum = 651 := by sorry

end NUMINAMATH_CALUDE_num_ways_to_achieve_18_with_5_dice_l144_14488


namespace NUMINAMATH_CALUDE_erica_money_l144_14496

/-- Given that Sam and Erica have $91 together and Sam has $38,
    prove that Erica has $53. -/
theorem erica_money (total : ℕ) (sam : ℕ) (erica : ℕ) 
    (h1 : total = 91)
    (h2 : sam = 38)
    (h3 : total = sam + erica) : 
  erica = 53 := by
  sorry

end NUMINAMATH_CALUDE_erica_money_l144_14496


namespace NUMINAMATH_CALUDE_total_spent_l144_14465

-- Define the amounts spent by each person
variable (A B C : ℝ)

-- Define the relationships between spending amounts
axiom alice_bella : A = (13/10) * B
axiom clara_bella : C = (4/5) * B
axiom alice_clara : A = C + 15

-- Theorem to prove
theorem total_spent : A + B + C = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_l144_14465


namespace NUMINAMATH_CALUDE_total_ways_to_place_balls_l144_14448

/-- The number of ways to place four distinct colored balls into two boxes -/
def place_balls : ℕ :=
  let box1_with_1_ball := Nat.choose 4 1
  let box1_with_2_balls := Nat.choose 4 2
  box1_with_1_ball + box1_with_2_balls

/-- Theorem stating that there are 10 ways to place the balls -/
theorem total_ways_to_place_balls : place_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_to_place_balls_l144_14448


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l144_14471

-- Define the number of flowers
def num_flowers : ℕ := 4

-- Define the number of gems
def num_gems : ℕ := 6

-- Define the number of invalid combinations
def num_invalid : ℕ := 3

-- Theorem statement
theorem wizard_elixir_combinations :
  (num_flowers * num_gems) - num_invalid = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l144_14471


namespace NUMINAMATH_CALUDE_money_ratio_l144_14489

/-- Jake's feeding allowance in dollars -/
def feeding_allowance : ℚ := 4

/-- Cost of one candy in dollars -/
def candy_cost : ℚ := 1/5

/-- Number of candies Jake's friend can purchase -/
def candies_purchased : ℕ := 5

/-- Amount of money Jake gave to his friend in dollars -/
def money_given : ℚ := candy_cost * candies_purchased

/-- Theorem stating the ratio of money given to feeding allowance -/
theorem money_ratio : money_given / feeding_allowance = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_l144_14489


namespace NUMINAMATH_CALUDE_max_candy_pieces_l144_14429

theorem max_candy_pieces (n : ℕ) (avg : ℕ) (min_pieces : ℕ) :
  n = 30 →
  avg = 7 →
  min_pieces = 1 →
  ∃ (max_pieces : ℕ), max_pieces = n * avg - (n - 1) * min_pieces ∧
                       max_pieces = 181 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l144_14429
