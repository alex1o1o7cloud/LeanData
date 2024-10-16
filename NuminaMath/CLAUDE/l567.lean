import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_proof_l567_56761

theorem average_marks_proof (total_subjects : Nat) 
                             (avg_five_subjects : ℝ) 
                             (sixth_subject_marks : ℝ) : 
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 50 →
  ((avg_five_subjects * 5 + sixth_subject_marks) / total_subjects : ℝ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l567_56761


namespace NUMINAMATH_CALUDE_additional_track_length_l567_56714

/-- Calculates the additional track length required to reduce the grade of a railroad line --/
theorem additional_track_length (rise : ℝ) (initial_grade : ℝ) (final_grade : ℝ) :
  rise = 800 →
  initial_grade = 0.04 →
  final_grade = 0.015 →
  ∃ (additional_length : ℝ), 
    33333 ≤ additional_length ∧ 
    additional_length < 33334 ∧
    additional_length = (rise / final_grade) - (rise / initial_grade) :=
by sorry

end NUMINAMATH_CALUDE_additional_track_length_l567_56714


namespace NUMINAMATH_CALUDE_smallest_n_for_polygon_cuts_l567_56749

theorem smallest_n_for_polygon_cuts : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (m - 2) % 31 = 0 ∧ (m - 2) % 65 = 0 → n ≤ m) ∧
  (n - 2) % 31 = 0 ∧ 
  (n - 2) % 65 = 0 ∧ 
  n = 2017 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_polygon_cuts_l567_56749


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l567_56736

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x^2 - x > 0) ↔ (x < 0 ∨ x > 1/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l567_56736


namespace NUMINAMATH_CALUDE_square_side_length_l567_56780

theorem square_side_length (s AF DH BG AE : ℝ) (area_EFGH : ℝ) 
  (h1 : AF = 7)
  (h2 : DH = 4)
  (h3 : BG = 5)
  (h4 : AE = 1)
  (h5 : area_EFGH = 78)
  (h6 : s > 0)
  (h7 : s * s = ((area_EFGH - (AF - DH) * (BG - AE)) * 2) + area_EFGH) :
  s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l567_56780


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l567_56706

/-- Represents Jane's babysitting career and age information -/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  start_age : ℕ

/-- Calculates the maximum age of a child Jane could babysit at a given time -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/-- Theorem stating the current age of the oldest person Jane could have babysat -/
theorem oldest_babysat_age (jane : BabysittingCareer)
  (h1 : jane.current_age = 34)
  (h2 : jane.years_since_stopped = 10)
  (h3 : jane.start_age = 18) :
  jane.current_age - jane.years_since_stopped - max_child_age (jane.current_age - jane.years_since_stopped) + jane.years_since_stopped = 22 :=
by
  sorry

#check oldest_babysat_age

end NUMINAMATH_CALUDE_oldest_babysat_age_l567_56706


namespace NUMINAMATH_CALUDE_height_c_smallest_integer_l567_56796

/-- Represents a right-angled triangle with given altitudes --/
structure RightTriangle where
  -- The height from A to BC
  height_a : ℝ
  -- The height from B to AC
  height_b : ℝ
  -- Ensures the heights are positive
  height_a_pos : height_a > 0
  height_b_pos : height_b > 0

/-- 
Given a right-angled triangle ABC with �angle C = 90°, 
if the height from A to BC is 5 and the height from B to AC is 15,
then the smallest integer greater than or equal to the height from C to AB is 5.
-/
theorem height_c_smallest_integer (t : RightTriangle) 
  (h1 : t.height_a = 5) 
  (h2 : t.height_b = 15) : 
  ∃ (h_c : ℝ), h_c > 0 ∧ 
  (∀ (x : ℝ), x > 0 → x * t.height_a * t.height_b = 2 * t.height_a * t.height_b → x = h_c) ∧
  (Nat.ceil h_c : ℝ) = 5 :=
sorry

end NUMINAMATH_CALUDE_height_c_smallest_integer_l567_56796


namespace NUMINAMATH_CALUDE_fourth_task_end_time_l567_56784

-- Define the start time of the first task
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes

-- Define the end time of the second task
def end_second_task : Nat := 10 * 60 + 20  -- 10:20 AM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem fourth_task_end_time :
  let total_time := end_second_task - start_time
  let task_duration := total_time / 2
  let end_time := end_second_task + task_duration * 2
  end_time = 12 * 60 + 40  -- 12:40 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_fourth_task_end_time_l567_56784


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_and_z_l567_56743

theorem x_in_terms_of_y_and_z (x y z : ℝ) :
  1 / (x + y) + 1 / (x - y) = z / (x - y) → x = z / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_and_z_l567_56743


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023_l567_56708

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_sequence_2023 (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  a 1 = 2 →
  GeometricSequence (a 1) (a 3) (a 7) →
  a 2023 = 2024 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023_l567_56708


namespace NUMINAMATH_CALUDE_charlie_snowballs_l567_56715

theorem charlie_snowballs (lucy_snowballs : ℕ) (charlie_snowballs : ℕ) : 
  lucy_snowballs = 19 → 
  charlie_snowballs = lucy_snowballs + 31 → 
  charlie_snowballs = 50 := by
  sorry

end NUMINAMATH_CALUDE_charlie_snowballs_l567_56715


namespace NUMINAMATH_CALUDE_ryan_weekly_commute_l567_56711

/-- Represents the different routes Ryan can take --/
inductive Route
| A
| B

/-- Represents the days of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the different transportation methods --/
inductive TransportMethod
| Bike
| Bus
| FriendRide
| Walk

/-- Function to calculate biking time based on the route --/
def bikingTime (route : Route) : ℕ :=
  match route with
  | Route.A => 30
  | Route.B => 40

/-- Function to calculate bus time based on the day --/
def busTime (day : Day) : ℕ :=
  match day with
  | Day.Tuesday => 50
  | _ => 40

/-- Function to calculate friend's ride time based on the day --/
def friendRideTime (day : Day) : ℕ :=
  match day with
  | Day.Wednesday => 25
  | _ => 10

/-- Function to calculate walking time --/
def walkingTime : ℕ := 90

/-- Function to calculate total weekly commuting time --/
def totalWeeklyCommutingTime : ℕ :=
  (bikingTime Route.A + bikingTime Route.B) +
  (busTime Day.Monday + busTime Day.Tuesday + busTime Day.Wednesday) +
  friendRideTime Day.Wednesday +
  walkingTime

/-- Theorem stating that Ryan's total weekly commuting time is 315 minutes --/
theorem ryan_weekly_commute : totalWeeklyCommutingTime = 315 := by
  sorry

end NUMINAMATH_CALUDE_ryan_weekly_commute_l567_56711


namespace NUMINAMATH_CALUDE_teds_age_l567_56750

theorem teds_age (t s : ℕ) : t = 3 * s - 20 → t + s = 76 → t = 52 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l567_56750


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l567_56756

-- Define a geometric sequence
def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_properties
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b) :
  (is_geometric_sequence (λ n => a n * b n)) ∧
  ¬(∀ x y : ℕ → ℝ, is_geometric_sequence x → is_geometric_sequence y →
    is_geometric_sequence (λ n => x n + y n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l567_56756


namespace NUMINAMATH_CALUDE_hyperbola_property_l567_56775

/-- The hyperbola with equation x²/9 - y²/4 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) - (p.2^2 / 4) = 1}

/-- Left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the right branch of the hyperbola -/
def A : ℝ × ℝ := sorry

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Point P such that 2 * OP = OA + OF₁ -/
def P : ℝ × ℝ := sorry

/-- Point Q such that 2 * OQ = OA + OF₂ -/
def Q : ℝ × ℝ := sorry

theorem hyperbola_property (h₁ : A ∈ Hyperbola)
    (h₂ : 2 • (P - O) = (A - O) + (F₁ - O))
    (h₃ : 2 • (Q - O) = (A - O) + (F₂ - O)) :
  ‖Q - O‖ - ‖P - O‖ = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_property_l567_56775


namespace NUMINAMATH_CALUDE_sum_of_squares_l567_56783

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 5)
  (h2 : a + b + c = 20) :
  a^2 + b^2 + c^2 = 390 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l567_56783


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l567_56753

theorem inequality_system_solutions : 
  {x : ℕ | 5 * x - 6 ≤ 2 * (x + 3) ∧ (x : ℚ) / 4 - 1 < (x - 2 : ℚ) / 3} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l567_56753


namespace NUMINAMATH_CALUDE_complex_real_condition_l567_56754

theorem complex_real_condition (a : ℝ) :
  (Complex.I * (a - 1) = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l567_56754


namespace NUMINAMATH_CALUDE_intersection_M_N_l567_56735

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l567_56735


namespace NUMINAMATH_CALUDE_angle_equation_solution_l567_56722

theorem angle_equation_solution (x : Real) (h_acute : 0 < x ∧ x < π / 2) 
  (h_eq : Real.sin x ^ 3 + Real.cos x ^ 3 = Real.sqrt 2 / 2) : 
  x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_solution_l567_56722


namespace NUMINAMATH_CALUDE_lead_percentage_in_mixture_l567_56759

/-- Proves that the percentage of lead in a mixture is 25% given the specified conditions -/
theorem lead_percentage_in_mixture
  (cobalt_percent : Real)
  (copper_percent : Real)
  (lead_weight : Real)
  (copper_weight : Real)
  (h1 : cobalt_percent = 0.15)
  (h2 : copper_percent = 0.60)
  (h3 : lead_weight = 5)
  (h4 : copper_weight = 12)
  : (lead_weight / (copper_weight / copper_percent)) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_lead_percentage_in_mixture_l567_56759


namespace NUMINAMATH_CALUDE_yoongis_subtraction_mistake_l567_56798

theorem yoongis_subtraction_mistake (A B : ℕ) : 
  A ≥ 1 ∧ A ≤ 9 ∧ B = 9 ∧ 
  (10 * A + 6) - 57 = 39 →
  10 * A + B = 99 := by
sorry

end NUMINAMATH_CALUDE_yoongis_subtraction_mistake_l567_56798


namespace NUMINAMATH_CALUDE_sum_inequality_l567_56787

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l567_56787


namespace NUMINAMATH_CALUDE_even_function_implies_cubic_l567_56704

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m+1)x^2 + (m-2)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m + 1) * x^2 + (m - 2) * x

theorem even_function_implies_cubic (m : ℝ) :
  IsEven (f m) → f m = fun x ↦ 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_cubic_l567_56704


namespace NUMINAMATH_CALUDE_molecular_weight_difference_l567_56797

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

-- Define molecular weights of compounds
def molecular_weight_A : ℝ := atomic_weight_N + 4 * atomic_weight_H + atomic_weight_Br
def molecular_weight_B : ℝ := 2 * atomic_weight_O + atomic_weight_C + 3 * atomic_weight_H

-- Theorem statement
theorem molecular_weight_difference :
  molecular_weight_A - molecular_weight_B = 50.91 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_difference_l567_56797


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l567_56769

theorem a_can_be_any_real : ∀ (a b c d : ℝ), 
  b * (3 * d + 2) ≠ 0 → 
  a / b < -c / (3 * d + 2) → 
  ∃ (a₁ a₂ a₃ : ℝ), a₁ > 0 ∧ a₂ < 0 ∧ a₃ = 0 ∧ 
    (a₁ / b < -c / (3 * d + 2)) ∧ 
    (a₂ / b < -c / (3 * d + 2)) ∧ 
    (a₃ / b < -c / (3 * d + 2)) := by
  sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l567_56769


namespace NUMINAMATH_CALUDE_fold_sum_theorem_l567_56727

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold of a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The sum of the x and y coordinates of the fourth point in a fold -/
def fourthPointSum (f : Fold) : ℝ :=
  f.p4.x + f.p4.y

/-- Theorem stating that for the given fold, the sum of x and y coordinates of the fourth point is 13 -/
theorem fold_sum_theorem (f : Fold) 
  (h1 : f.p1 = ⟨0, 4⟩) 
  (h2 : f.p2 = ⟨5, 0⟩) 
  (h3 : f.p3 = ⟨9, 6⟩) : 
  fourthPointSum f = 13 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_theorem_l567_56727


namespace NUMINAMATH_CALUDE_line_parameterization_l567_56785

/-- Given a line y = 5x - 7 parameterized by [x; y] = [s; 2] + t[3; h], 
    prove that s = 9/5 and h = 15 -/
theorem line_parameterization (x y s h t : ℝ) : 
  y = 5 * x - 7 ∧ 
  ∃ (v : ℝ × ℝ), v.1 = x ∧ v.2 = y ∧ v = (s, 2) + t • (3, h) →
  s = 9/5 ∧ h = 15 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l567_56785


namespace NUMINAMATH_CALUDE_constant_c_value_l567_56707

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, 4 * (x + 2) * (x + b) = x^2 + c*x + 12) →
  c = 14 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l567_56707


namespace NUMINAMATH_CALUDE_wilson_number_l567_56731

theorem wilson_number (x : ℚ) : x - (1/3) * x = 16/3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_wilson_number_l567_56731


namespace NUMINAMATH_CALUDE_translation_of_sine_graph_l567_56737

open Real

theorem translation_of_sine_graph (θ φ : ℝ) : 
  (abs θ < π / 2) →
  (0 < φ) →
  (φ < π) →
  (sin θ = 1 / 2) →
  (sin (θ - 2 * φ) = 1 / 2) →
  (φ = 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_sine_graph_l567_56737


namespace NUMINAMATH_CALUDE_range_of_m_for_two_zeros_l567_56717

/-- Given a function f and a real number m, g is defined as their sum -/
def g (f : ℝ → ℝ) (m : ℝ) : ℝ → ℝ := λ x ↦ f x + m

/-- The main theorem -/
theorem range_of_m_for_two_zeros (ω : ℝ) (h_ω_pos : ω > 0) 
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 2 * (Real.cos (ω * x / 2))^2) 
  (h_period : ∀ x, f (x + 2 * Real.pi / 3) = f x) :
  {m : ℝ | ∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ z₁ ∈ Set.Icc 0 (Real.pi / 3) ∧ z₂ ∈ Set.Icc 0 (Real.pi / 3) ∧ 
    g f m z₁ = 0 ∧ g f m z₂ = 0 ∧ ∀ z ∈ Set.Icc 0 (Real.pi / 3), g f m z = 0 → z = z₁ ∨ z = z₂} = 
  Set.Ioc (-3) (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_two_zeros_l567_56717


namespace NUMINAMATH_CALUDE_clockwise_rotation_240_l567_56758

/-- The angle formed by rotating a ray clockwise around its endpoint -/
def clockwise_rotation (angle : ℝ) : ℝ := -angle

/-- Theorem: The angle formed by rotating a ray 240° clockwise around its endpoint is -240° -/
theorem clockwise_rotation_240 : clockwise_rotation 240 = -240 := by
  sorry

end NUMINAMATH_CALUDE_clockwise_rotation_240_l567_56758


namespace NUMINAMATH_CALUDE_value_of_a_l567_56745

theorem value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l567_56745


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l567_56738

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / m) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l567_56738


namespace NUMINAMATH_CALUDE_u_2002_equals_2_l567_56705

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 2
| 5 => 1
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 3
| (n + 1) => g (u n)

-- State the theorem
theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_u_2002_equals_2_l567_56705


namespace NUMINAMATH_CALUDE_men_on_first_road_calculation_l567_56742

/-- The number of men who worked on the first road -/
def men_on_first_road : ℕ := 30

/-- The length of the first road in kilometers -/
def first_road_length : ℕ := 1

/-- The number of days spent working on the first road -/
def days_on_first_road : ℕ := 12

/-- The number of hours worked per day on the first road -/
def hours_per_day_first_road : ℕ := 8

/-- The number of men working on the second road -/
def men_on_second_road : ℕ := 20

/-- The number of days spent working on the second road -/
def days_on_second_road : ℕ := 32

/-- The number of hours worked per day on the second road -/
def hours_per_day_second_road : ℕ := 9

/-- The length of the second road in kilometers -/
def second_road_length : ℕ := 2

theorem men_on_first_road_calculation :
  men_on_first_road * days_on_first_road * hours_per_day_first_road =
  (men_on_second_road * days_on_second_road * hours_per_day_second_road) / 2 :=
by sorry

end NUMINAMATH_CALUDE_men_on_first_road_calculation_l567_56742


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_problem_l567_56760

theorem arithmetic_sequence_sin_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_problem_l567_56760


namespace NUMINAMATH_CALUDE_hike_attendance_l567_56773

/-- The number of cars used for the hike -/
def num_cars : ℕ := 7

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 10

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 4

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The number of buses used for the hike -/
def num_buses : ℕ := 3

/-- The number of people in each bus -/
def people_per_bus : ℕ := 20

/-- The number of minibuses used for the hike -/
def num_minibuses : ℕ := 2

/-- The number of people in each minibus -/
def people_per_minibus : ℕ := 8

/-- The total number of people who went on the hike -/
def total_people : ℕ := 
  num_cars * people_per_car + 
  num_taxis * people_per_taxi + 
  num_vans * people_per_van + 
  num_buses * people_per_bus + 
  num_minibuses * people_per_minibus

theorem hike_attendance : total_people = 184 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l567_56773


namespace NUMINAMATH_CALUDE_exists_containing_quadrilateral_l567_56774

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- A point in 2D space -/
def Point := Real × Real

/-- Check if a point is inside a convex polygon -/
def is_inside (p : Point) (poly : ConvexPolygon) : Bool := sorry

/-- Check if four points form a quadrilateral -/
def is_quadrilateral (a b c d : Point) : Bool := sorry

/-- Check if a quadrilateral contains a point -/
def quadrilateral_contains (a b c d : Point) (p : Point) : Bool := sorry

theorem exists_containing_quadrilateral 
  (poly : ConvexPolygon) (p1 p2 : Point) 
  (h1 : is_inside p1 poly) (h2 : is_inside p2 poly) :
  ∃ (a b c d : Point), 
    a ∈ poly.vertices ∧ 
    b ∈ poly.vertices ∧ 
    c ∈ poly.vertices ∧ 
    d ∈ poly.vertices ∧
    is_quadrilateral a b c d ∧
    quadrilateral_contains a b c d p1 ∧
    quadrilateral_contains a b c d p2 := by
  sorry

end NUMINAMATH_CALUDE_exists_containing_quadrilateral_l567_56774


namespace NUMINAMATH_CALUDE_greatest_difference_multiple_of_five_l567_56766

theorem greatest_difference_multiple_of_five : ∀ a b : ℕ,
  (a < 10) →
  (b < 10) →
  (700 + 10 * a + b) % 5 = 0 →
  ((a + b) % 5 = 0) →
  ∃ c d : ℕ,
    (c < 10) ∧
    (d < 10) ∧
    (700 + 10 * c + d) % 5 = 0 ∧
    ((c + d) % 5 = 0) ∧
    (∀ e f : ℕ,
      (e < 10) →
      (f < 10) →
      (700 + 10 * e + f) % 5 = 0 →
      ((e + f) % 5 = 0) →
      (a + b) - (c + d) ≤ (e + f) - (c + d)) ∧
    (a + b) - (c + d) = 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_difference_multiple_of_five_l567_56766


namespace NUMINAMATH_CALUDE_expression_values_l567_56710

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l567_56710


namespace NUMINAMATH_CALUDE_expression_evaluation_l567_56777

theorem expression_evaluation :
  ((2^2009)^2 - (2^2007)^2) / ((2^2008)^2 - (2^2006)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l567_56777


namespace NUMINAMATH_CALUDE_monica_savings_l567_56733

theorem monica_savings (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_per_cycle : ℚ) 
  (h1 : weeks_per_cycle = 60)
  (h2 : num_cycles = 5)
  (h3 : total_per_cycle = 4500) :
  (num_cycles * total_per_cycle) / (num_cycles * weeks_per_cycle) = 75 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l567_56733


namespace NUMINAMATH_CALUDE_jorge_age_proof_l567_56793

/-- Jorge's age in 2005 -/
def jorge_age_2005 : ℕ := 16

/-- Simon's age in 2010 -/
def simon_age_2010 : ℕ := 45

/-- Age difference between Simon and Jorge -/
def age_difference : ℕ := 24

/-- Years between 2005 and 2010 -/
def years_difference : ℕ := 5

theorem jorge_age_proof :
  jorge_age_2005 = simon_age_2010 - years_difference - age_difference :=
by sorry

end NUMINAMATH_CALUDE_jorge_age_proof_l567_56793


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l567_56755

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 90 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 35 := by
sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l567_56755


namespace NUMINAMATH_CALUDE_linear_function_properties_l567_56741

-- Define the linear function
def f (k x : ℝ) : ℝ := (3 - k) * x - 2 * k^2 + 18

theorem linear_function_properties :
  -- Part 1: The function passes through (0, -2) when k = ±√10
  (∃ k : ℝ, k^2 = 10 ∧ f k 0 = -2) ∧
  -- Part 2: The function is parallel to y = -x when k = 4
  (f 4 1 - f 4 0 = -1) ∧
  -- Part 3: The function decreases as x increases when k > 3
  (∀ k : ℝ, k > 3 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l567_56741


namespace NUMINAMATH_CALUDE_product_of_roots_l567_56720

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 22 → 
  ∃ y : ℝ, (y + 3) * (y - 5) = 22 ∧ x * y = -37 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l567_56720


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l567_56734

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a₀ + a₁*(x + 3) + a₂*(x + 3)^2 + a₃*(x + 3)^3 + 
    a₄*(x + 3)^4 + a₅*(x + 3)^5 + a₆*(x + 3)^6 + a₇*(x + 3)^7 + a₈*(x + 3)^8 + 
    a₉*(x + 3)^9 + a₁₀*(x + 3)^10) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l567_56734


namespace NUMINAMATH_CALUDE_problem_solution_l567_56763

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := k * a^x - a^(-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 4 * (f a 1 x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a 1
  ∀ x : ℝ, f x = -f (-x) →
  f 1 > 0 →
  f 1 = 3/2 →
  (∀ x : ℝ, f (x^2 + 2*x) + f (x - 4) > 0 ↔ x < -4 ∨ x > 1) ∧
  (∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x ≥ 1 → g a x ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l567_56763


namespace NUMINAMATH_CALUDE_classroom_seating_arrangements_l567_56767

/-- Represents a seating arrangement of students in a classroom -/
structure SeatingArrangement where
  rows : Nat
  cols : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of valid seating arrangements -/
def validArrangements (s : SeatingArrangement) : Nat :=
  2 * (Nat.factorial s.boys) * (Nat.factorial s.girls)

/-- Theorem stating the number of valid seating arrangements
    for the given classroom configuration -/
theorem classroom_seating_arrangements :
  let s : SeatingArrangement := {
    rows := 5,
    cols := 6,
    boys := 15,
    girls := 15
  }
  validArrangements s = 2 * (Nat.factorial 15) * (Nat.factorial 15) :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_seating_arrangements_l567_56767


namespace NUMINAMATH_CALUDE_measles_cases_1987_l567_56768

/-- Calculates the number of measles cases in a given year assuming a linear decrease --/
def measlesCases (initialYear finalYear targetYear : ℕ) (initialCases finalCases : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let targetYears := targetYear - initialYear
  let totalDecrease := initialCases - finalCases
  let decrease := (targetYears * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of measles cases in 1987 would be 112,875 --/
theorem measles_cases_1987 :
  measlesCases 1960 1996 1987 450000 500 = 112875 := by
  sorry

#eval measlesCases 1960 1996 1987 450000 500

end NUMINAMATH_CALUDE_measles_cases_1987_l567_56768


namespace NUMINAMATH_CALUDE_lighthouse_lights_sum_l567_56795

theorem lighthouse_lights_sum (n : ℕ) (a₁ : ℝ) (q : ℝ) : 
  n = 7 → a₁ = 1 → q = 2 → 
  a₁ * (1 - q^n) / (1 - q) = 127 := by
sorry

end NUMINAMATH_CALUDE_lighthouse_lights_sum_l567_56795


namespace NUMINAMATH_CALUDE_min_value_theorem_l567_56702

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log 2 * (2^a) + Real.log 2 * (8^b) = Real.log 2 → 
    1/x + 1/(3*y) ≤ 1/a + 1/(3*b)) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log 2 * (2^x) + Real.log 2 * (8^y) = Real.log 2 ∧ 
    1/x + 1/(3*y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l567_56702


namespace NUMINAMATH_CALUDE_base12_addition_l567_56721

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Digit12 to its corresponding natural number --/
def digit12ToNat (d : Digit12) : ℕ :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11
  | Digit12.C => 12

/-- Represents a number in base 12 --/
def Number12 := List Digit12

/-- Converts a Number12 to its corresponding natural number --/
def number12ToNat (n : Number12) : ℕ :=
  n.foldr (fun d acc => digit12ToNat d + 12 * acc) 0

/-- The theorem to be proved --/
theorem base12_addition :
  let n1 : Number12 := [Digit12.C, Digit12.D9, Digit12.D7]
  let n2 : Number12 := [Digit12.D2, Digit12.D6, Digit12.A]
  let result : Number12 := [Digit12.D3, Digit12.D4, Digit12.D1, Digit12.B]
  number12ToNat n1 + number12ToNat n2 = number12ToNat result := by
  sorry

end NUMINAMATH_CALUDE_base12_addition_l567_56721


namespace NUMINAMATH_CALUDE_resulting_shape_is_option_A_l567_56794

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the shape resulting from the folding and cutting process
structure ResultingShape where
  vertices : List Point
  is_symmetric : Bool

-- Function to perform the folding and cutting process
def fold_and_cut (triangle : EquilateralTriangle) : ResultingShape :=
  sorry

-- Theorem stating that the resulting shape has the properties of option A
theorem resulting_shape_is_option_A (triangle : EquilateralTriangle) :
  let shape := fold_and_cut triangle
  shape.is_symmetric ∧ shape.vertices.length = 6 :=
sorry

end NUMINAMATH_CALUDE_resulting_shape_is_option_A_l567_56794


namespace NUMINAMATH_CALUDE_cos_2alpha_from_tan_alpha_plus_pi_4_l567_56788

theorem cos_2alpha_from_tan_alpha_plus_pi_4 (α : Real) 
  (h : Real.tan (α + π/4) = 2) : 
  Real.cos (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_from_tan_alpha_plus_pi_4_l567_56788


namespace NUMINAMATH_CALUDE_expand_and_simplify_l567_56719

theorem expand_and_simplify (a b : ℝ) : (3*a + b) * (a - b) = 3*a^2 - 2*a*b - b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l567_56719


namespace NUMINAMATH_CALUDE_cube_root_problem_l567_56786

theorem cube_root_problem (a b c : ℝ) : 
  (3 * a + 21) ^ (1/3) = 3 → 
  (4 * a - b - 1) ^ (1/2) = 2 → 
  c ^ (1/2) = c → 
  a = 2 ∧ b = 3 ∧ c = 0 ∧ (3 * a + 10 * b + c) ^ (1/2) = 6 ∨ (3 * a + 10 * b + c) ^ (1/2) = -6 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_problem_l567_56786


namespace NUMINAMATH_CALUDE_elizabeths_husband_weight_l567_56725

/-- Represents a married couple -/
structure Couple where
  husband_weight : ℝ
  wife_weight : ℝ

/-- The problem setup -/
def cannibal_problem (couples : Fin 3 → Couple) : Prop :=
  let wives_weights := (couples 0).wife_weight + (couples 1).wife_weight + (couples 2).wife_weight
  let total_weight := (couples 0).husband_weight + (couples 0).wife_weight +
                      (couples 1).husband_weight + (couples 1).wife_weight +
                      (couples 2).husband_weight + (couples 2).wife_weight
  ∃ (leon victor maurice : Fin 3),
    leon ≠ victor ∧ leon ≠ maurice ∧ victor ≠ maurice ∧
    wives_weights = 171 ∧
    ¬ ∃ n : ℤ, total_weight = n ∧
    (couples leon).husband_weight = (couples leon).wife_weight ∧
    (couples victor).husband_weight = 1.5 * (couples victor).wife_weight ∧
    (couples maurice).husband_weight = 2 * (couples maurice).wife_weight ∧
    (couples 0).wife_weight = (couples 1).wife_weight + 10 ∧
    (couples 1).wife_weight = (couples 2).wife_weight - 5 ∧
    (couples victor).husband_weight = 85.5

/-- The main theorem to prove -/
theorem elizabeths_husband_weight (couples : Fin 3 → Couple) :
  cannibal_problem couples → ∃ i : Fin 3, (couples i).husband_weight = 85.5 :=
sorry

end NUMINAMATH_CALUDE_elizabeths_husband_weight_l567_56725


namespace NUMINAMATH_CALUDE_erased_number_proof_l567_56744

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 1 →
  (n : ℝ) * ((n : ℝ) + 21) / 2 - x = 23 * ((n : ℝ) - 1) →
  x = 36 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l567_56744


namespace NUMINAMATH_CALUDE_f_sum_zero_l567_56765

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_property (x : ℝ) : f (2 - x) + f x = 0

-- State the theorem
theorem f_sum_zero : f 2022 + f 2023 = 0 := by sorry

end NUMINAMATH_CALUDE_f_sum_zero_l567_56765


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l567_56703

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 12 * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l567_56703


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l567_56779

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-1/5 : ℝ) 3 ∪ {3} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l567_56779


namespace NUMINAMATH_CALUDE_tower_height_property_l567_56792

/-- The height of a tower that appears under twice the angle from 18 meters as it does from 48 meters -/
def tower_height : ℝ := 24

/-- The distance from which the tower appears under twice the angle -/
def closer_distance : ℝ := 18

/-- The distance from which the tower appears under the original angle -/
def farther_distance : ℝ := 48

/-- The angle at which the tower appears from the farther distance -/
noncomputable def base_angle (h : ℝ) : ℝ := Real.arctan (h / farther_distance)

/-- The theorem stating the property of the tower's height -/
theorem tower_height_property : 
  base_angle (2 * tower_height) = 2 * base_angle tower_height := by sorry

end NUMINAMATH_CALUDE_tower_height_property_l567_56792


namespace NUMINAMATH_CALUDE_fraction_equality_l567_56776

theorem fraction_equality : (1632^2 - 1625^2) / (1645^2 - 1612^2) = 7/33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l567_56776


namespace NUMINAMATH_CALUDE_saras_team_games_l567_56712

/-- The total number of games played by Sara's high school basketball team -/
def total_games (won_games defeated_games : ℕ) : ℕ :=
  won_games + defeated_games

/-- Theorem stating that for Sara's team, the total number of games
    is equal to the sum of won games and defeated games -/
theorem saras_team_games :
  total_games 12 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_saras_team_games_l567_56712


namespace NUMINAMATH_CALUDE_age_problem_l567_56770

/-- Mr. Li's current age -/
def mr_li_age : ℕ := 23

/-- Xiao Ming's current age -/
def xiao_ming_age : ℕ := 10

/-- The age difference between Mr. Li and Xiao Ming -/
def age_difference : ℕ := 13

theorem age_problem :
  (mr_li_age - 6 = xiao_ming_age + 7) ∧
  (mr_li_age + 4 + xiao_ming_age - 5 = 32) ∧
  (mr_li_age = xiao_ming_age + age_difference) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l567_56770


namespace NUMINAMATH_CALUDE_system_solution_l567_56751

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (x * y + y * z + z * x = a^2 - x^2) ∧ 
  (x * y + y * z + z * x = b^2 - y^2) ∧ 
  (x * y + y * z + z * x = c^2 - z^2) →
  ((x = (|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = (|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = (-|a*b/c| + |a*c/b| + |b*c/a|) / 2) ∨
   (x = -(|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = -(|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = -(-|a*b/c| + |a*c/b| + |b*c/a|) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l567_56751


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l567_56700

-- Define what it means for two quadratic radicals to be of the same type
def same_type_quadratic_radical (a b : ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a = k * x^2 ∧ b = k * y^2)

-- State the theorem
theorem quadratic_radical_equivalence :
  same_type_quadratic_radical (m + 1) 8 → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l567_56700


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l567_56726

theorem sum_of_x_and_y_is_three (x y : ℝ) 
  (hx : (x - 1)^2003 + 2002*(x - 1) = -1)
  (hy : (y - 2)^2003 + 2002*(y - 2) = 1) : 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_three_l567_56726


namespace NUMINAMATH_CALUDE_enclosed_area_semicircles_l567_56747

/-- Given a semicircle with radius R and its diameter divided into parts 2r and 2(R-r),
    the area enclosed between the three semicircles (the original and two smaller ones)
    is equal to π r(R-r) -/
theorem enclosed_area_semicircles (R r : ℝ) (h1 : 0 < R) (h2 : 0 < r) (h3 : r < R) :
  let original_area := π * R^2 / 2
  let small_area1 := π * r^2 / 2
  let small_area2 := π * (R-r)^2 / 2
  original_area - small_area1 - small_area2 = π * r * (R-r) :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_semicircles_l567_56747


namespace NUMINAMATH_CALUDE_common_chord_theorem_l567_56713

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def common_chord_line (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem common_chord_theorem :
  ∃ (x y : ℝ), 
    (C1 x y ∧ C2 x y) → 
    (common_chord_line x y ∧ 
     ∃ (x1 y1 x2 y2 : ℝ), 
       C1 x1 y1 ∧ C2 x1 y1 ∧ 
       C1 x2 y2 ∧ C2 x2 y2 ∧ 
       common_chord_line x1 y1 ∧ 
       common_chord_line x2 y2 ∧ 
       ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 24/5) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_theorem_l567_56713


namespace NUMINAMATH_CALUDE_reflect_L_shape_is_mirrored_l567_56791

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a vertical line -/
structure VerticalLine :=
  (x : ℝ)

/-- Defines an L-like shape -/
def LLikeShape : Shape :=
  sorry

/-- Defines a mirrored L-like shape -/
def MirroredLLikeShape : Shape :=
  sorry

/-- Reflects a point across a vertical line -/
def reflectPoint (p : ℝ × ℝ) (line : VerticalLine) : ℝ × ℝ :=
  (2 * line.x - p.1, p.2)

/-- Reflects a shape across a vertical line -/
def reflectShape (s : Shape) (line : VerticalLine) : Shape :=
  ⟨s.points.image (λ p => reflectPoint p line)⟩

/-- Theorem: Reflecting an L-like shape across a vertical line results in a mirrored L-like shape -/
theorem reflect_L_shape_is_mirrored (line : VerticalLine) :
  reflectShape LLikeShape line = MirroredLLikeShape :=
sorry

end NUMINAMATH_CALUDE_reflect_L_shape_is_mirrored_l567_56791


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_twenty_l567_56748

def sequence_term (n : ℕ) : ℤ := (-1)^(n+1) * n * (n+1)

theorem fourth_term_is_negative_twenty : sequence_term 4 = -20 := by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_twenty_l567_56748


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l567_56724

theorem geometric_series_ratio (a r : ℝ) (hr : 0 < r) (hr1 : r < 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l567_56724


namespace NUMINAMATH_CALUDE_art_supplies_solution_l567_56723

/-- Represents the cost and quantity of art supplies -/
structure ArtSupplies where
  brush_cost : ℕ
  canvas_cost : ℕ
  brush_quantity : ℕ
  canvas_quantity : ℕ

/-- Defines the conditions of the art supplies problem -/
def art_supplies_conditions (s : ArtSupplies) : Prop :=
  s.brush_cost * 2 + s.canvas_cost * 4 = 94 ∧
  s.brush_cost * 4 + s.canvas_cost * 2 = 98 ∧
  s.brush_quantity + s.canvas_quantity = 10 ∧
  s.brush_cost * s.brush_quantity + s.canvas_cost * s.canvas_quantity ≤ 157

/-- Theorem stating the solution to the art supplies problem -/
theorem art_supplies_solution (s : ArtSupplies) 
  (h : art_supplies_conditions s) : 
  s.brush_cost = 17 ∧ 
  s.canvas_cost = 15 ∧ 
  (∀ m : ℕ, m < 7 → s.brush_cost * (10 - m) + s.canvas_cost * m > 157) ∧
  s.brush_cost * 2 + s.canvas_cost * 8 < s.brush_cost * 3 + s.canvas_cost * 7 := by
  sorry


end NUMINAMATH_CALUDE_art_supplies_solution_l567_56723


namespace NUMINAMATH_CALUDE_number_guessing_game_l567_56728

theorem number_guessing_game (a b c : ℕ) : 
  a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ c > 0 ∧ c < 10 →
  ((2 * a + 2) * 5 + b) * 10 + c = 567 →
  a = 4 ∧ b = 6 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_number_guessing_game_l567_56728


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l567_56782

/-- Given a = e^0.1 - 1, b = sin 0.1, and c = ln 1.1, prove that c < b < a -/
theorem ordering_of_expressions :
  let a : ℝ := Real.exp 0.1 - 1
  let b : ℝ := Real.sin 0.1
  let c : ℝ := Real.log 1.1
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l567_56782


namespace NUMINAMATH_CALUDE_tangent_at_one_minimum_a_l567_56799

noncomputable section

def f (x : ℝ) := (1/6) * x^3 + (1/2) * x - x * Real.log x

def domain : Set ℝ := {x | x > 0}

def interval : Set ℝ := {x | 1/Real.exp 1 < x ∧ x < Real.exp 1}

theorem tangent_at_one (x : ℝ) (hx : x ∈ domain) :
  (f x - f 1) = 0 * (x - 1) := by sorry

theorem minimum_a :
  ∃ a : ℝ, (∀ x ∈ interval, f x < a) ∧
  (∀ b : ℝ, (∀ x ∈ interval, f x < b) → a ≤ b) ∧
  a = (1/6) * (Real.exp 1)^3 - (1/2) * (Real.exp 1) := by sorry

end

end NUMINAMATH_CALUDE_tangent_at_one_minimum_a_l567_56799


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_18_27_36_l567_56781

theorem gcf_lcm_sum_18_27_36 : 
  let X := Nat.gcd 18 (Nat.gcd 27 36)
  let Y := Nat.lcm 18 (Nat.lcm 27 36)
  X + Y = 117 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_18_27_36_l567_56781


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l567_56739

theorem consecutive_odd_numbers_multiple (k : ℕ) : 
  let a := 7
  let b := a + 2
  let c := b + 2
  k * a = 3 * c + (2 * b + 5) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l567_56739


namespace NUMINAMATH_CALUDE_part_probabilities_l567_56730

/-- Given two machines producing parts with known quantities of standard parts,
    this theorem proves the probabilities of selecting a standard part overall
    and conditionally based on which machine produced it. -/
theorem part_probabilities
  (total_parts_1 : ℕ) (standard_parts_1 : ℕ)
  (total_parts_2 : ℕ) (standard_parts_2 : ℕ)
  (h1 : total_parts_1 = 200)
  (h2 : standard_parts_1 = 190)
  (h3 : total_parts_2 = 300)
  (h4 : standard_parts_2 = 280) :
  let total_parts := total_parts_1 + total_parts_2
  let total_standard := standard_parts_1 + standard_parts_2
  let p_A := total_standard / total_parts
  let p_A_given_B := standard_parts_1 / total_parts_1
  let p_A_given_not_B := standard_parts_2 / total_parts_2
  p_A = 47/50 ∧ p_A_given_B = 19/20 ∧ p_A_given_not_B = 14/15 :=
by sorry

end NUMINAMATH_CALUDE_part_probabilities_l567_56730


namespace NUMINAMATH_CALUDE_five_g_base_stations_scientific_notation_l567_56729

theorem five_g_base_stations_scientific_notation :
  (819000 : ℝ) = 8.19 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_five_g_base_stations_scientific_notation_l567_56729


namespace NUMINAMATH_CALUDE_no_integer_solution_l567_56771

theorem no_integer_solution : ¬∃ (x y : ℤ), 
  (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l567_56771


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l567_56701

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l567_56701


namespace NUMINAMATH_CALUDE_unknown_number_problem_l567_56789

theorem unknown_number_problem (x : ℚ) : (2 / 3) * x + 6 = 10 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l567_56789


namespace NUMINAMATH_CALUDE_all_days_happy_l567_56772

theorem all_days_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_all_days_happy_l567_56772


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l567_56790

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  (1 / (a + 3 * b)) + (1 / (b + 3 * c)) + (1 / (c + 3 * a)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l567_56790


namespace NUMINAMATH_CALUDE_dog_division_theorem_l567_56709

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def ways_to_divide_dogs (n : ℕ) (sizes : List ℕ) : ℕ :=
  sorry

theorem dog_division_theorem :
  ways_to_divide_dogs number_of_dogs group_sizes = 4200 :=
by sorry

end NUMINAMATH_CALUDE_dog_division_theorem_l567_56709


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l567_56746

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l567_56746


namespace NUMINAMATH_CALUDE_abc_maximum_l567_56762

theorem abc_maximum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c^2 = (a + c) * (b + c)) (h2 : a + b + c = 3) :
  a * b * c ≤ 1 ∧ ∃ (a' b' c' : ℝ), a' * b' * c' = 1 ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a' * b' + c'^2 = (a' + c') * (b' + c') ∧ 
    a' + b' + c' = 3 :=
by sorry

end NUMINAMATH_CALUDE_abc_maximum_l567_56762


namespace NUMINAMATH_CALUDE_eldest_boy_age_l567_56718

/-- Given three boys whose ages are in proportion 3 : 5 : 7 and have an average age of 15 years,
    the age of the eldest boy is 21 years. -/
theorem eldest_boy_age (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 45 →  -- average age is 15
  ∃ (k : ℕ), age1 = 3 * k ∧ age2 = 5 * k ∧ age3 = 7 * k →  -- ages are in proportion 3 : 5 : 7
  age3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_eldest_boy_age_l567_56718


namespace NUMINAMATH_CALUDE_equation_solution_range_l567_56732

-- Define the equation
def equation (a : ℝ) (ξ : ℝ) (x : ℝ) : Prop :=
  4 * a * x^2 - ξ * x + a * ξ = 0

-- Define the set of possible ξ values
def ξ_values : Set ℝ := {4, 9}

-- Define the interval for x
def x_interval : Set ℝ := Set.Icc 1 2

-- Define the range of a
def a_range : Set ℝ := Set.union (Set.Icc (8/5) 2) (Set.Icc (36/13) 3)

-- Theorem statement
theorem equation_solution_range :
  ∀ a : ℝ, (∃ ξ ∈ ξ_values, ∃ x ∈ x_interval, equation a ξ x) ↔ a ∈ a_range :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l567_56732


namespace NUMINAMATH_CALUDE_slips_with_three_l567_56740

theorem slips_with_three (total_slips : ℕ) (value_a value_b : ℕ) (expected_value : ℚ) : 
  total_slips = 15 →
  value_a = 3 →
  value_b = 8 →
  expected_value = 5 →
  ∃ (slips_with_a : ℕ),
    slips_with_a ≤ total_slips ∧
    (slips_with_a : ℚ) / total_slips * value_a + 
    ((total_slips - slips_with_a) : ℚ) / total_slips * value_b = expected_value ∧
    slips_with_a = 9 := by
sorry

end NUMINAMATH_CALUDE_slips_with_three_l567_56740


namespace NUMINAMATH_CALUDE_median_distance_product_sum_l567_56752

/-- Given a triangle with medians of lengths s₁, s₂, s₃ and a point P with 
    distances d₁, d₂, d₃ to these medians respectively, prove that 
    s₁d₁ + s₂d₂ + s₃d₃ = 0 -/
theorem median_distance_product_sum (s₁ s₂ s₃ d₁ d₂ d₃ : ℝ) : 
  s₁ * d₁ + s₂ * d₂ + s₃ * d₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_median_distance_product_sum_l567_56752


namespace NUMINAMATH_CALUDE_problem_solution_l567_56757

theorem problem_solution : 
  ((-54 : ℚ) * (-1/2 + 2/3 - 4/9) = 15) ∧ 
  (-2 / (4/9) * (-2/3)^2 = -2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l567_56757


namespace NUMINAMATH_CALUDE_smallest_multiple_l567_56764

theorem smallest_multiple (x : ℕ) : x = 40 ↔ 
  (x > 0 ∧ 
   800 ∣ (360 * x) ∧ 
   ∀ y : ℕ, y > 0 → 800 ∣ (360 * y) → x ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l567_56764


namespace NUMINAMATH_CALUDE_perfect_square_condition_l567_56716

theorem perfect_square_condition (m : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 4 * x^2 - m * x * y + 9 * y^2 = k^2) →
  m = 12 ∨ m = -12 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l567_56716


namespace NUMINAMATH_CALUDE_juanico_age_30_years_from_now_l567_56778

/-- Juanico's age 30 years from now, given the conditions in the problem -/
def juanico_future_age (gladys_current_age : ℕ) (juanico_current_age : ℕ) : ℕ :=
  juanico_current_age + 30

/-- The theorem stating Juanico's age 30 years from now -/
theorem juanico_age_30_years_from_now :
  ∀ (gladys_current_age : ℕ) (juanico_current_age : ℕ),
    gladys_current_age + 10 = 40 →
    juanico_current_age = gladys_current_age / 2 - 4 →
    juanico_future_age gladys_current_age juanico_current_age = 41 :=
by
  sorry

#check juanico_age_30_years_from_now

end NUMINAMATH_CALUDE_juanico_age_30_years_from_now_l567_56778
