import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_cubes_l2871_287186

theorem root_sum_cubes (a b c : ℝ) : 
  (a^3 + 14*a^2 + 49*a + 36 = 0) → 
  (b^3 + 14*b^2 + 49*b + 36 = 0) → 
  (c^3 + 14*c^2 + 49*c + 36 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 686 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l2871_287186


namespace NUMINAMATH_CALUDE_expression_value_l2871_287136

theorem expression_value
  (a b x y : ℝ)
  (m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * |m| - 2 * x * y = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2871_287136


namespace NUMINAMATH_CALUDE_journey_distance_l2871_287143

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_time = 40)
  (h2 : speed1 = 20)
  (h3 : speed2 = 30)
  (h4 : total_time = (distance / 2) / speed1 + (distance / 2) / speed2) :
  distance = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2871_287143


namespace NUMINAMATH_CALUDE_correct_proposition_l2871_287145

theorem correct_proposition :
  let p := ∀ x : ℝ, 2 * x < 3 * x
  let q := ∃ x : ℝ, x^3 = 1 - x^2
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l2871_287145


namespace NUMINAMATH_CALUDE_initial_number_of_boys_l2871_287114

theorem initial_number_of_boys (B : ℝ) : 
  (1.2 * B) + B + (2.4 * B) = 51 → B = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_boys_l2871_287114


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2871_287181

theorem increasing_function_inequality (f : ℝ → ℝ) (h_increasing : Monotone f) :
  (∀ x : ℝ, f 4 < f (2^x)) → {x : ℝ | x > 2}.Nonempty := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2871_287181


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2871_287169

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the non-coincident property for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincident property for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_a_perp_α : perp_line_plane a α)
  (h_b_perp_β : perp_line_plane b β)
  (h_α_perp_β : perp_plane_plane α β) :
  perp_line_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l2871_287169


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2871_287125

theorem polynomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  10 * p^9 * q = 120 * p^7 * q^3 → 
  p = Real.sqrt (12/13) := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2871_287125


namespace NUMINAMATH_CALUDE_fifth_term_value_l2871_287178

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem fifth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l2871_287178


namespace NUMINAMATH_CALUDE_grace_pumpkin_pie_fraction_l2871_287146

theorem grace_pumpkin_pie_fraction :
  let total_pies : ℕ := 4
  let sold_pies : ℕ := 1
  let given_pies : ℕ := 1
  let slices_per_pie : ℕ := 6
  let remaining_slices : ℕ := 4
  
  let remaining_pies : ℕ := total_pies - sold_pies - given_pies
  let total_slices : ℕ := remaining_pies * slices_per_pie
  let eaten_slices : ℕ := total_slices - remaining_slices
  
  (eaten_slices : ℚ) / total_slices = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grace_pumpkin_pie_fraction_l2871_287146


namespace NUMINAMATH_CALUDE_average_of_numbers_is_one_l2871_287117

def numbers : List Int := [-5, -2, 0, 4, 8]

theorem average_of_numbers_is_one :
  (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_one_l2871_287117


namespace NUMINAMATH_CALUDE_jessica_payment_l2871_287134

/-- Calculates the payment for a given hour based on the repeating pattern --/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 6 with
  | 0 => 2
  | 1 => 4
  | 2 => 6
  | 3 => 8
  | 4 => 10
  | 5 => 12
  | _ => 0  -- This case should never occur due to the modulo operation

/-- Calculates the total payment for a given number of hours --/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem jessica_payment : total_payment 45 = 306 := by
  sorry


end NUMINAMATH_CALUDE_jessica_payment_l2871_287134


namespace NUMINAMATH_CALUDE_points_per_treasure_l2871_287159

theorem points_per_treasure (total_treasures : ℕ) (total_score : ℕ) (points_per_treasure : ℕ) : 
  total_treasures = 7 → total_score = 63 → points_per_treasure * total_treasures = total_score → points_per_treasure = 9 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l2871_287159


namespace NUMINAMATH_CALUDE_find_n_l2871_287138

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ) : ℚ := 2 * n

/-- Theorem stating that given the conditions, n must equal 10 -/
theorem find_n : ∃ (n : ℕ), n > 0 ∧ a^2 - (b n)^2 = 0 ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_find_n_l2871_287138


namespace NUMINAMATH_CALUDE_parabola_constant_term_l2871_287151

theorem parabola_constant_term (b c : ℝ) : 
  (2 = 2*(1^2) + b*1 + c) ∧ (2 = 2*(3^2) + b*3 + c) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_constant_term_l2871_287151


namespace NUMINAMATH_CALUDE_sqrt_c_value_l2871_287155

theorem sqrt_c_value (a b c : ℝ) :
  (a^2 + 2020 * a + c = 0) →
  (b^2 + 2020 * b + c = 0) →
  (a / b + b / a = 98) →
  Real.sqrt c = 202 := by
sorry

end NUMINAMATH_CALUDE_sqrt_c_value_l2871_287155


namespace NUMINAMATH_CALUDE_decimal_numbers_less_than_one_infinite_l2871_287135

theorem decimal_numbers_less_than_one_infinite :
  Set.Infinite {x : ℝ | x < 1 ∧ ∃ (n : ℕ), x = ↑n / (10 ^ n)} :=
sorry

end NUMINAMATH_CALUDE_decimal_numbers_less_than_one_infinite_l2871_287135


namespace NUMINAMATH_CALUDE_inequality_proof_l2871_287148

theorem inequality_proof (k n : ℕ) (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2871_287148


namespace NUMINAMATH_CALUDE_remainder_proof_l2871_287163

theorem remainder_proof : (7 * 10^20 + 1^20) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2871_287163


namespace NUMINAMATH_CALUDE_chord_rotation_in_unit_circle_l2871_287132

/-- Chord rotation in a unit circle -/
theorem chord_rotation_in_unit_circle :
  -- Define the circle
  let circle_radius : ℝ := 1
  -- Define the chord length (side of inscribed equilateral triangle)
  let chord_length : ℝ := Real.sqrt 3
  -- Define the rotation angle (90 degrees in radians)
  let rotation_angle : ℝ := π / 2
  -- Define the area of the full circle
  let circle_area : ℝ := π * circle_radius ^ 2

  -- Statement 1: Area swept by chord during 90° rotation
  let area_swept : ℝ := (7 * π / 16) - 1 / 4

  -- Statement 2: Angle to sweep half of circle's area
  let angle_half_area : ℝ := (4 * π + 6 * Real.sqrt 3) / 9

  -- Prove the following:
  True →
    -- 1. The area swept by the chord during a 90° rotation
    (area_swept = (7 * π / 16) - 1 / 4) ∧
    -- 2. The angle required to sweep exactly half of the circle's area
    (angle_half_area = (4 * π + 6 * Real.sqrt 3) / 9) ∧
    -- Additional verification: the swept area at angle_half_area is indeed half the circle's area
    (2 * ((angle_half_area / (2 * π)) * circle_area - 
     (Real.sqrt (1 - (chord_length / 2) ^ 2) * (chord_length / 2))) = circle_area) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_rotation_in_unit_circle_l2871_287132


namespace NUMINAMATH_CALUDE_complex_power_result_l2871_287177

theorem complex_power_result : ∃ (i : ℂ), i^2 = -1 ∧ ((1 + i) / i)^2014 = 2^1007 * i := by sorry

end NUMINAMATH_CALUDE_complex_power_result_l2871_287177


namespace NUMINAMATH_CALUDE_mady_balls_after_2023_steps_l2871_287115

/-- Converts a natural number to its septenary (base 7) representation -/
def to_septenary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Sums the digits in a list of natural numbers -/
def sum_digits (l : List ℕ) : ℕ :=
  l.sum

/-- Represents Mady's ball placement process -/
def mady_process (steps : ℕ) : ℕ :=
  sum_digits (to_septenary steps)

theorem mady_balls_after_2023_steps :
  mady_process 2023 = 13 := by sorry

end NUMINAMATH_CALUDE_mady_balls_after_2023_steps_l2871_287115


namespace NUMINAMATH_CALUDE_regression_unit_increase_food_expenditure_increase_l2871_287162

/-- Represents a linear regression equation ŷ = ax + b -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- Calculates the predicted value for a given x -/
def LinearRegression.predict (reg : LinearRegression) (x : ℝ) : ℝ :=
  reg.a * x + reg.b

/-- The increase in ŷ when x increases by 1 is equal to the coefficient a -/
theorem regression_unit_increase (reg : LinearRegression) :
  reg.predict (x + 1) - reg.predict x = reg.a :=
by sorry

/-- The specific regression equation from the problem -/
def food_expenditure_regression : LinearRegression :=
  { a := 0.254, b := 0.321 }

/-- The increase in food expenditure when income increases by 1 is 0.254 -/
theorem food_expenditure_increase :
  food_expenditure_regression.predict (x + 1) - food_expenditure_regression.predict x = 0.254 :=
by sorry

end NUMINAMATH_CALUDE_regression_unit_increase_food_expenditure_increase_l2871_287162


namespace NUMINAMATH_CALUDE_pharmacy_work_hours_l2871_287130

/-- Proves that given the conditions of the pharmacy problem, 
    the number of hours worked by Ann and Becky is 8 --/
theorem pharmacy_work_hours : 
  ∀ (h : ℕ), 
  (7 * h + 7 * h + 7 * 6 = 154) → 
  h = 8 := by
sorry

end NUMINAMATH_CALUDE_pharmacy_work_hours_l2871_287130


namespace NUMINAMATH_CALUDE_equality_condition_l2871_287196

theorem equality_condition (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (20 - x) + Real.sqrt (20 * x - x^3) = 20 ↔
  x = 20 ∨ x^2 + x - 20 = 0 := by
sorry

end NUMINAMATH_CALUDE_equality_condition_l2871_287196


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l2871_287192

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l2871_287192


namespace NUMINAMATH_CALUDE_weekly_distance_is_1760_l2871_287110

/-- Calculates the total distance traveled by a driver in a week -/
def weekly_distance : ℕ :=
  let weekday_speed1 : ℕ := 30
  let weekday_time1 : ℕ := 3
  let weekday_speed2 : ℕ := 25
  let weekday_time2 : ℕ := 4
  let weekday_speed3 : ℕ := 40
  let weekday_time3 : ℕ := 2
  let weekday_days : ℕ := 6
  let sunday_speed : ℕ := 35
  let sunday_time : ℕ := 5
  let sunday_breaks : ℕ := 2
  let break_duration : ℕ := 30

  let weekday_distance := (weekday_speed1 * weekday_time1 + 
                           weekday_speed2 * weekday_time2 + 
                           weekday_speed3 * weekday_time3) * weekday_days
  let sunday_distance := sunday_speed * (sunday_time - sunday_breaks * break_duration / 60)
  
  weekday_distance + sunday_distance

theorem weekly_distance_is_1760 : weekly_distance = 1760 := by
  sorry

end NUMINAMATH_CALUDE_weekly_distance_is_1760_l2871_287110


namespace NUMINAMATH_CALUDE_pq_is_one_eighth_of_rs_l2871_287107

-- Define the line segment RS and points P and Q on it
structure LineSegment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the problem setup
def problem (RS : LineSegment) (P Q : Point) : Prop :=
  -- P and Q lie on RS
  0 ≤ P.position ∧ P.position ≤ RS.length ∧
  0 ≤ Q.position ∧ Q.position ≤ RS.length ∧
  -- RP is 3 times PS
  P.position = (3/4) * RS.length ∧
  -- RQ is 7 times QS
  Q.position = (7/8) * RS.length

-- Theorem to prove
theorem pq_is_one_eighth_of_rs (RS : LineSegment) (P Q : Point) 
  (h : problem RS P Q) : 
  abs (Q.position - P.position) = (1/8) * RS.length :=
sorry

end NUMINAMATH_CALUDE_pq_is_one_eighth_of_rs_l2871_287107


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l2871_287127

def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := probability_k_heads 5 3
  let p4 := probability_k_heads 5 4
  |p3 - p4| = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l2871_287127


namespace NUMINAMATH_CALUDE_sequence_lower_bound_l2871_287112

/-- Given a sequence of positive integers satisfying certain conditions, 
    the last element is greater than or equal to 2n² - 1 -/
theorem sequence_lower_bound (n : ℕ) (a : ℕ → ℕ) : n > 1 →
  (∀ i, 1 ≤ i → i < n → a i < a (i + 1)) →
  (∀ i, 1 ≤ i → i < n → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) →
  a n ≥ 2 * n ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_lower_bound_l2871_287112


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2871_287165

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2871_287165


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2871_287184

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 10 → left = 4 → new = 42 → initial - left + new = 48 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2871_287184


namespace NUMINAMATH_CALUDE_jerky_order_theorem_l2871_287154

/-- Calculates the total number of jerky bags for a customer order -/
def customer_order_bags (production_rate : ℕ) (initial_inventory : ℕ) (production_days : ℕ) : ℕ :=
  production_rate * production_days + initial_inventory

/-- Theorem stating that given the specific conditions, the customer order is 60 bags -/
theorem jerky_order_theorem :
  let production_rate := 10
  let initial_inventory := 20
  let production_days := 4
  customer_order_bags production_rate initial_inventory production_days = 60 := by
  sorry

#eval customer_order_bags 10 20 4  -- Should output 60

end NUMINAMATH_CALUDE_jerky_order_theorem_l2871_287154


namespace NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l2871_287170

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (non_coincident : Line → Line → Prop)
variable (plane_non_coincident : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_non_coincident : non_coincident m n)
  (h_plane_non_coincident : plane_non_coincident α β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l2871_287170


namespace NUMINAMATH_CALUDE_a_minus_b_equals_negative_seven_l2871_287128

theorem a_minus_b_equals_negative_seven
  (a b : ℝ)
  (h1 : |a| = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0) :
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_negative_seven_l2871_287128


namespace NUMINAMATH_CALUDE_train_time_difference_l2871_287116

def distance : ℝ := 425.80645161290323
def speed_slow : ℝ := 44
def speed_fast : ℝ := 75

theorem train_time_difference :
  (distance / speed_slow) - (distance / speed_fast) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_time_difference_l2871_287116


namespace NUMINAMATH_CALUDE_average_senior_visitors_l2871_287174

/-- Represents the categories of visitors -/
inductive VisitorCategory
  | Adult
  | Student
  | Senior

/-- Represents the types of days -/
inductive DayType
  | Sunday
  | Other

/-- Average number of visitors for each day type -/
def averageVisitors (d : DayType) : ℕ :=
  match d with
  | DayType.Sunday => 150
  | DayType.Other => 120

/-- Ratio of visitors for each category on each day type -/
def visitorRatio (c : VisitorCategory) (d : DayType) : ℕ :=
  match d with
  | DayType.Sunday =>
    match c with
    | VisitorCategory.Adult => 5
    | VisitorCategory.Student => 3
    | VisitorCategory.Senior => 2
  | DayType.Other =>
    match c with
    | VisitorCategory.Adult => 4
    | VisitorCategory.Student => 3
    | VisitorCategory.Senior => 3

def daysInMonth : ℕ := 30
def sundaysInMonth : ℕ := 5
def otherDaysInMonth : ℕ := daysInMonth - sundaysInMonth

theorem average_senior_visitors :
  (sundaysInMonth * averageVisitors DayType.Sunday * visitorRatio VisitorCategory.Senior DayType.Sunday +
   otherDaysInMonth * averageVisitors DayType.Other * visitorRatio VisitorCategory.Senior DayType.Other) /
  daysInMonth = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_senior_visitors_l2871_287174


namespace NUMINAMATH_CALUDE_point_movement_l2871_287124

def initial_point : ℝ × ℝ := (-2, -3)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem point_movement :
  let p := initial_point
  let p' := move_left p 1
  let p'' := move_up p' 3
  p'' = (-3, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_l2871_287124


namespace NUMINAMATH_CALUDE_customized_notebook_combinations_l2871_287199

/-- The number of different notebook designs available. -/
def notebook_designs : ℕ := 12

/-- The number of different pen types available. -/
def pen_types : ℕ := 3

/-- The number of different sticker varieties available. -/
def sticker_varieties : ℕ := 5

/-- The total number of possible combinations for a customized notebook package. -/
def total_combinations : ℕ := notebook_designs * pen_types * sticker_varieties

/-- Theorem stating that the total number of combinations is 180. -/
theorem customized_notebook_combinations :
  total_combinations = 180 := by sorry

end NUMINAMATH_CALUDE_customized_notebook_combinations_l2871_287199


namespace NUMINAMATH_CALUDE_employed_females_percentage_l2871_287142

theorem employed_females_percentage
  (total_employable : ℝ)
  (total_employed : ℝ)
  (employed_males : ℝ)
  (h1 : total_employed = 1.2 * total_employable)
  (h2 : employed_males = 0.8 * total_employable)
  : (total_employed - employed_males) / total_employed = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l2871_287142


namespace NUMINAMATH_CALUDE_popcorn_servings_proof_l2871_287141

/-- The number of pieces of popcorn in one serving -/
def serving_size : ℕ := 30

/-- The number of pieces of popcorn Jared can eat -/
def jared_consumption : ℕ := 90

/-- The number of Jared's friends -/
def num_friends : ℕ := 3

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_consumption : ℕ := 60

/-- The total number of servings needed for Jared and his friends -/
def total_servings : ℕ := 9

theorem popcorn_servings_proof :
  (jared_consumption + num_friends * friend_consumption) / serving_size = total_servings :=
sorry

end NUMINAMATH_CALUDE_popcorn_servings_proof_l2871_287141


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l2871_287140

/-- Given a triangle EFG and an inscribed rectangle ABCD, prove the area of ABCD -/
theorem inscribed_rectangle_area (EG AD AB : ℝ) (altitude : ℝ) : 
  EG = 15 →
  altitude = 10 →
  AB = (1 / 3) * AD →
  AD ≤ EG →
  AD * AB = 100 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l2871_287140


namespace NUMINAMATH_CALUDE_jarek_calculation_l2871_287175

theorem jarek_calculation (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jarek_calculation_l2871_287175


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_inequality_l2871_287139

theorem least_positive_integer_satisfying_inequality : 
  ∀ n : ℕ+, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 ↔ n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_inequality_l2871_287139


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l2871_287150

theorem arithmetic_sequence_sum_times_three : 
  ∀ (a l d n : ℕ), 
    a = 50 → 
    l = 95 → 
    d = 3 → 
    n * d = l - a + d → 
    3 * (n / 2 * (a + l)) = 3480 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l2871_287150


namespace NUMINAMATH_CALUDE_complement_intersection_l2871_287173

theorem complement_intersection (I A B : Set ℕ) : 
  I = {1, 2, 3, 4, 5} →
  A = {1, 2} →
  B = {1, 3, 5} →
  (I \ A) ∩ B = {3, 5} := by
sorry

end NUMINAMATH_CALUDE_complement_intersection_l2871_287173


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l2871_287185

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < (y : ℚ) / 17 → y ≥ 13 :=
by
  sorry

theorem thirteen_satisfies (y : ℤ) : (8 : ℚ) / 11 < (13 : ℚ) / 17 :=
by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < (z : ℚ) / 17 → z ≥ y) ∧ y = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l2871_287185


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_length_l2871_287166

def repeating_decimal_length (n m : ℕ) : ℕ :=
  sorry

theorem seven_thirteenths_repeating_length :
  repeating_decimal_length 7 13 = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_length_l2871_287166


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2871_287194

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2871_287194


namespace NUMINAMATH_CALUDE_divisors_of_2018_or_2019_l2871_287160

theorem divisors_of_2018_or_2019 (h1 : Nat.Prime 673) (h2 : Nat.Prime 1009) :
  (Finset.filter (fun n => n ∣ 2018 ∨ n ∣ 2019) (Finset.range 2020)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2018_or_2019_l2871_287160


namespace NUMINAMATH_CALUDE_headphone_cost_l2871_287149

/-- The cost of the headphone set given Amanda's shopping scenario -/
theorem headphone_cost (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  cassette_cost = 9 →
  num_cassettes = 2 →
  remaining_amount = 7 →
  initial_amount - (num_cassettes * cassette_cost) - remaining_amount = 25 := by
sorry

end NUMINAMATH_CALUDE_headphone_cost_l2871_287149


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l2871_287158

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l2871_287158


namespace NUMINAMATH_CALUDE_second_item_cost_price_l2871_287157

/-- Given two items sold together for 432 yuan, where one item is sold at a 20% loss
    and the combined sale results in a 20% profit, prove that the cost price of the second item is 90 yuan. -/
theorem second_item_cost_price (total_selling_price : ℝ) (loss_percentage : ℝ) (profit_percentage : ℝ) 
  (h1 : total_selling_price = 432)
  (h2 : loss_percentage = 0.20)
  (h3 : profit_percentage = 0.20) :
  ∃ (cost_price_1 cost_price_2 : ℝ),
    cost_price_1 * (1 - loss_percentage) = total_selling_price / 2 ∧
    total_selling_price = (cost_price_1 + cost_price_2) * (1 + profit_percentage) ∧
    cost_price_2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_second_item_cost_price_l2871_287157


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2871_287189

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).im ≠ 0 ∧ 
  (Complex.mk (m^2 - 3*m) (m^2 - 5*m + 6)).re = 0 → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2871_287189


namespace NUMINAMATH_CALUDE_subset_intersection_bound_l2871_287100

/-- Given a set S of n elements and a family of b subsets of S, each containing k elements,
    with the property that any two subsets intersect in at most one element,
    the number of subsets b is bounded above by ⌊(n/k)⌊(n-1)/(k-1)⌋⌋. -/
theorem subset_intersection_bound (n k b : ℕ) (S : Finset (Fin n)) (B : Fin b → Finset (Fin n)) 
  (h1 : ∀ i, (B i).card = k)
  (h2 : ∀ i j, i < j → (B i ∩ B j).card ≤ 1)
  (h3 : k > 0)
  (h4 : n > 0)
  : b ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_bound_l2871_287100


namespace NUMINAMATH_CALUDE_expand_product_l2871_287183

theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2871_287183


namespace NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l2871_287156

theorem sqrt_x6_plus_x4 (x : ℝ) : Real.sqrt (x^6 + x^4) = |x|^2 * Real.sqrt (x^2 + 1) := by sorry

end NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l2871_287156


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l2871_287195

/-- Given a large rectangle of dimensions A × B and a small rectangle of dimensions a × b,
    where the small rectangle is entirely contained within the large rectangle,
    this theorem proves that the absolute difference between the total area of the parts
    of the small rectangle outside the large rectangle and the area of the large rectangle
    not covered by the small rectangle is equal to 572, given specific dimensions. -/
theorem rectangle_area_difference (A B a b : ℝ) 
    (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7)
    (h5 : a ≤ A ∧ b ≤ B) : 
    |0 - (A * B - a * b)| = 572 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l2871_287195


namespace NUMINAMATH_CALUDE_scientific_notation_of_2720000_l2871_287109

theorem scientific_notation_of_2720000 :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    2720000 = a * (10 : ℝ) ^ n ∧
    a = 2.72 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2720000_l2871_287109


namespace NUMINAMATH_CALUDE_solve_for_a_l2871_287182

theorem solve_for_a (a : ℝ) : (1 + 2 * a = -3) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2871_287182


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2871_287168

theorem circle_area_ratio (A B : Real) (rA rB : ℝ) (hA : A = 2 * π * rA) (hB : B = 2 * π * rB)
  (h_arc : (60 / 360) * A = (40 / 360) * B) :
  π * rA^2 / (π * rB^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2871_287168


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2871_287105

theorem exponential_equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (16 : ℝ) ^ (x - 1) = (512 : ℝ) ^ (x + 1) ∧ x = -15/8 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2871_287105


namespace NUMINAMATH_CALUDE_fourth_hexagon_dots_l2871_287103

/-- Calculates the number of dots in the nth layer of the hexagonal pattern. -/
def layerDots (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n % 2 = 0 then 7 * (n - 1)
  else 7 * n

/-- Calculates the total number of dots in the nth hexagon of the sequence. -/
def totalDots (n : ℕ) : ℕ :=
  (List.range n).map layerDots |> List.sum

/-- The fourth hexagon in the sequence contains 50 dots. -/
theorem fourth_hexagon_dots : totalDots 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fourth_hexagon_dots_l2871_287103


namespace NUMINAMATH_CALUDE_min_value_of_f_l2871_287118

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2871_287118


namespace NUMINAMATH_CALUDE_project_work_time_difference_l2871_287147

theorem project_work_time_difference (x : ℝ) 
  (h1 : x > 0)
  (h2 : 2*x + 3*x + 4*x = 90) : 4*x - 2*x = 20 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l2871_287147


namespace NUMINAMATH_CALUDE_simplify_sqrt_neg_five_squared_l2871_287121

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_neg_five_squared_l2871_287121


namespace NUMINAMATH_CALUDE_second_number_is_seventeen_l2871_287191

theorem second_number_is_seventeen (first_number second_number third_number : ℕ) :
  first_number = 16 →
  third_number = 20 →
  3 * first_number + 3 * second_number + 3 * third_number + 11 = 170 →
  second_number = 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_seventeen_l2871_287191


namespace NUMINAMATH_CALUDE_school_tournament_games_l2871_287193

/-- The number of games in a round-robin tournament for n teams -/
def roundRobinGames (n : ℕ) : ℕ := n.choose 2

/-- The total number of games in a multi-grade round-robin tournament -/
def totalGames (grade1 grade2 grade3 : ℕ) : ℕ :=
  roundRobinGames grade1 + roundRobinGames grade2 + roundRobinGames grade3

theorem school_tournament_games :
  totalGames 5 8 3 = 41 := by sorry

end NUMINAMATH_CALUDE_school_tournament_games_l2871_287193


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l2871_287167

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l2871_287167


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2871_287126

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2871_287126


namespace NUMINAMATH_CALUDE_fathers_age_multiplier_l2871_287131

theorem fathers_age_multiplier (father_age son_age : ℕ) (h_sum : father_age + son_age = 75)
  (h_son : son_age = 27) (h_father : father_age = 48) :
  ∃ (M : ℕ), M * (son_age - (father_age - son_age)) = father_age ∧ M = 8 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_multiplier_l2871_287131


namespace NUMINAMATH_CALUDE_root_in_interval_l2871_287171

/-- The function f(x) = ln x + 3x - 7 has a root in the interval (2, 3) -/
theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + 3 * x - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2871_287171


namespace NUMINAMATH_CALUDE_plant_growth_mean_l2871_287111

theorem plant_growth_mean (measurements : List ℝ) 
  (h1 : measurements.length = 15)
  (h2 : (measurements.filter (λ x => 10 ≤ x ∧ x < 20)).length = 3)
  (h3 : (measurements.filter (λ x => 20 ≤ x ∧ x < 30)).length = 7)
  (h4 : (measurements.filter (λ x => 30 ≤ x ∧ x < 40)).length = 5)
  (h5 : measurements.sum = 401) :
  measurements.sum / measurements.length = 401 / 15 := by
sorry

end NUMINAMATH_CALUDE_plant_growth_mean_l2871_287111


namespace NUMINAMATH_CALUDE_min_value_theorem_l2871_287161

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_x : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ (Set.Ioo x₁ x₂)) :
  ∃ (min_val : ℝ), 
    (∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y) ∧ 
    (x₁ + x₂ + a / (x₁ * x₂) = y ↔ y = 4 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2871_287161


namespace NUMINAMATH_CALUDE_function_composition_equality_l2871_287119

theorem function_composition_equality (a : ℝ) (h_pos : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x + 1 / Real.sqrt 2
  f (f (1 / Real.sqrt 2)) = f 0 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l2871_287119


namespace NUMINAMATH_CALUDE_fraction_condition_l2871_287144

theorem fraction_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a / b > 1 → b / a < 1) ∧
  (∃ a b, b / a < 1 ∧ a / b ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_condition_l2871_287144


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2871_287123

theorem z_in_first_quadrant (z : ℂ) (h : (3 + 2*I)*z = 13*I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2871_287123


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_all_even_digits_l2871_287176

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  100000000 > n ∧ n ≥ 10000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

def largest_eight_digit_with_even_digits : Nat := 99986420

theorem largest_eight_digit_with_all_even_digits :
  is_eight_digit largest_eight_digit_with_even_digits ∧
  contains_all_even_digits largest_eight_digit_with_even_digits ∧
  ∀ n, is_eight_digit n → contains_all_even_digits n →
    n ≤ largest_eight_digit_with_even_digits :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_all_even_digits_l2871_287176


namespace NUMINAMATH_CALUDE_xy_value_l2871_287198

theorem xy_value (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2871_287198


namespace NUMINAMATH_CALUDE_trees_needed_l2871_287108

/-- Represents a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the planting scheme for trees -/
structure PlantingScheme where
  treeSpacing : ℕ
  alternateTrees : Bool

/-- Calculates the total number of trees needed for a given playground and planting scheme -/
def totalTrees (p : Playground) (scheme : PlantingScheme) : ℕ :=
  (perimeter p) / scheme.treeSpacing

/-- Theorem stating the total number of trees required for the given playground and planting scheme -/
theorem trees_needed (p : Playground) (scheme : PlantingScheme) :
  p.length = 150 ∧ p.width = 60 ∧ scheme.treeSpacing = 10 ∧ scheme.alternateTrees = true →
  totalTrees p scheme = 42 := by
  sorry

end NUMINAMATH_CALUDE_trees_needed_l2871_287108


namespace NUMINAMATH_CALUDE_total_handshakes_l2871_287188

/-- The total number of handshakes in a group of boys with specific conditions -/
theorem total_handshakes (n : ℕ) (l : ℕ) (f : ℕ) (h : ℕ) : 
  n = 15 → l = 5 → f = 3 → h = 2 → 
  (n * (n - 1)) / 2 - (l * (n - l)) - f * h = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l2871_287188


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2871_287179

theorem quadratic_two_roots (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (4 * a)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2871_287179


namespace NUMINAMATH_CALUDE_infinitely_many_primes_in_differences_l2871_287137

/-- Definition of the sequence a_n -/
def a (k : ℕ) : ℕ → ℕ
  | n => if n < k then 0  -- arbitrary value for n < k
         else if n = k then 2 * k
         else if Nat.gcd (a k (n-1)) n = 1 then a k (n-1) + 1
         else 2 * n

/-- The theorem statement -/
theorem infinitely_many_primes_in_differences (k : ℕ) (h : k ≥ 3) :
  ∀ M : ℕ, ∃ n > k, ∃ p : ℕ, p.Prime ∧ p > M ∧ p ∣ (a k n - a k (n-1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_in_differences_l2871_287137


namespace NUMINAMATH_CALUDE_triangle_side_length_l2871_287197

/-- Given a triangle DEF with sides d, e, and f, where d = 7, e = 3, and cos(D - E) = 39/40,
    prove that the length of side f is equal to √(9937)/10. -/
theorem triangle_side_length (D E F : ℝ) (d e f : ℝ) : 
  d = 7 → 
  e = 3 → 
  Real.cos (D - E) = 39 / 40 → 
  f = Real.sqrt 9937 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2871_287197


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2871_287106

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(0 < a ∧ a < b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2871_287106


namespace NUMINAMATH_CALUDE_complement_of_M_wrt_U_l2871_287102

def U : Finset Int := {1, -2, 3, -4, 5, -6}
def M : Finset Int := {1, -2, 3, -4}

theorem complement_of_M_wrt_U :
  U \ M = {5, -6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_wrt_U_l2871_287102


namespace NUMINAMATH_CALUDE_set_union_problem_l2871_287190

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {0, a}
  let B : Set ℕ := {2^a, b}
  A ∪ B = {0, 1, 2} → b = 0 ∨ b = 1 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l2871_287190


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2871_287180

/-- Given a class of students where half the number of girls equals one-fifth of the total number of students, prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h : 2 * G = S) :
  (S - G) / G = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2871_287180


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l2871_287172

/-- Given complex numbers p, q, and r forming an equilateral triangle with side length 24,
    if |p + q + r| = 48, then |pq + pr + qr| = 768 -/
theorem equilateral_triangle_sum_product (p q r : ℂ) :
  (Complex.abs (p - q) = 24) →
  (Complex.abs (q - r) = 24) →
  (Complex.abs (r - p) = 24) →
  (Complex.abs (p + q + r) = 48) →
  Complex.abs (p*q + q*r + r*p) = 768 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l2871_287172


namespace NUMINAMATH_CALUDE_s_value_l2871_287153

theorem s_value (n : ℝ) (s : ℝ) (h1 : n ≠ 0) 
  (h2 : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1/n)) : s = 1/4 := by
sorry

end NUMINAMATH_CALUDE_s_value_l2871_287153


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2871_287133

theorem arithmetic_mean_problem (y : ℝ) : 
  (7 + y + 22 + 8 + 18) / 5 = 15 → y = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2871_287133


namespace NUMINAMATH_CALUDE_tan_30_degrees_l2871_287113

theorem tan_30_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l2871_287113


namespace NUMINAMATH_CALUDE_dima_walking_speed_l2871_287129

/-- Represents the time in hours and minutes -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hour * 60 + t2.minute) - (t1.hour * 60 + t1.minute)

/-- Represents the problem setup -/
structure ProblemSetup where
  scheduledArrival : Time
  actualArrival : Time
  carSpeed : Nat
  earlyArrivalTime : Nat

/-- Calculates Dima's walking speed -/
def calculateWalkingSpeed (setup : ProblemSetup) : Rat :=
  sorry

theorem dima_walking_speed (setup : ProblemSetup) 
  (h1 : setup.scheduledArrival = ⟨18, 0⟩)
  (h2 : setup.actualArrival = ⟨17, 5⟩)
  (h3 : setup.carSpeed = 60)
  (h4 : setup.earlyArrivalTime = 10) :
  calculateWalkingSpeed setup = 6 := by
  sorry

end NUMINAMATH_CALUDE_dima_walking_speed_l2871_287129


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2871_287187

theorem complex_equation_solution (Z : ℂ) : (3 + Z) * Complex.I = 1 → Z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2871_287187


namespace NUMINAMATH_CALUDE_rational_function_value_l2871_287152

-- Define the property for the rational function f
def satisfies_equation (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 3 * f (1 / x) + 2 * f x / x = x^2

-- State the theorem
theorem rational_function_value :
  ∀ f : ℚ → ℚ, satisfies_equation f → f (-2) = 67 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l2871_287152


namespace NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l2871_287122

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Definition of similarity for triangles -/
def similar_triangles (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

/-- Theorem: All equilateral triangles are similar -/
theorem all_equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar_triangles t1 t2 :=
sorry

end NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l2871_287122


namespace NUMINAMATH_CALUDE_profit_function_max_profit_profit_2400_l2871_287164

-- Define the cost price
def cost_price : ℝ := 80

-- Define the sales quantity function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 320

-- Define the valid price range
def valid_price (x : ℝ) : Prop := 80 ≤ x ∧ x ≤ 160

-- Define the daily profit function
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_quantity x

-- Theorem statements
theorem profit_function (x : ℝ) (h : valid_price x) :
  daily_profit x = -2 * x^2 + 480 * x - 25600 := by sorry

theorem max_profit (x : ℝ) (h : valid_price x) :
  daily_profit x ≤ 3200 ∧ daily_profit 120 = 3200 := by sorry

theorem profit_2400 :
  ∃ x, valid_price x ∧ daily_profit x = 2400 ∧
  ∀ y, valid_price y → daily_profit y = 2400 → x ≤ y := by sorry

end NUMINAMATH_CALUDE_profit_function_max_profit_profit_2400_l2871_287164


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l2871_287101

theorem smallest_distance_between_complex_numbers
  (z w : ℂ)
  (hz : Complex.abs (z + 2 + 4 * Complex.I) = 2)
  (hw : Complex.abs (w - 6 - 7 * Complex.I) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 185 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4 * Complex.I) = 2 →
      Complex.abs (w' - 6 - 7 * Complex.I) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l2871_287101


namespace NUMINAMATH_CALUDE_andrew_payment_l2871_287104

/-- The total amount Andrew paid for grapes and mangoes -/
def total_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1055 for his purchase -/
theorem andrew_payment : total_paid 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l2871_287104


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2871_287120

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 4*x = 5 ↔ x = 1 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2871_287120
