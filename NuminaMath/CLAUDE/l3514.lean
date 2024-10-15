import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_vertex_and_symmetry_l3514_351427

/-- Given a quadratic function f(x) = -x^2 - 4x + 2, 
    its vertex is at (-2, 6) and its axis of symmetry is x = -2 -/
theorem quadratic_vertex_and_symmetry :
  let f : ℝ → ℝ := λ x ↦ -x^2 - 4*x + 2
  ∃ (vertex : ℝ × ℝ) (axis : ℝ),
    vertex = (-2, 6) ∧
    axis = -2 ∧
    (∀ x, f x = f (2 * axis - x)) ∧
    (∀ x, f x ≤ f axis) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_and_symmetry_l3514_351427


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l3514_351449

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 2) : 
  Real.tan α = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l3514_351449


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3514_351474

theorem quadratic_always_positive : ∀ x : ℝ, 15 * x^2 - 8 * x + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3514_351474


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3514_351414

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x1 x2 : ℝ), x1 > 0 ∧ x2 < 0 ∧ 
  (a * x1^2 + 2 * x1 + 1 = 0) ∧ 
  (a * x2^2 + 2 * x2 + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3514_351414


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3514_351466

theorem polynomial_difference_divisibility 
  (a b c d : ℤ) (x y : ℤ) (h : x ≠ y) :
  ∃ k : ℤ, (x - y) * k = 
    (a * x^3 + b * x^2 + c * x + d) - (a * y^3 + b * y^2 + c * y + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3514_351466


namespace NUMINAMATH_CALUDE_calculate_expression_l3514_351485

theorem calculate_expression : ((-2 : ℤ)^2 : ℝ) - |(-5 : ℤ)| - Real.sqrt 144 = -13 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3514_351485


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3514_351430

theorem tangent_line_at_negative_one (x y : ℝ) :
  y = 2*x - x^3 → 
  let tangent_point := (-1, 2*(-1) - (-1)^3)
  let tangent_slope := -3*(-1)^2 + 2
  (x + y + 2 = 0) = 
    ((y - tangent_point.2) = tangent_slope * (x - tangent_point.1)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3514_351430


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3514_351467

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- The slope of the line that the tangent should be parallel to -/
def m : ℝ := 4

theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = -1 ∧ p.2 = -4 ∨ p.1 = 1 ∧ p.2 = 0} =
  {p : ℝ × ℝ | p.2 = f p.1 ∧ f' p.1 = m} :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3514_351467


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3514_351473

/-- Given an ellipse with equation 9(x-1)^2 + y^2 = 36, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√10 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), 9 * (x - 1)^2 + y^2 = 36 → 
      ((x = A.1 ∧ y = A.2) ∨ (x = -A.1 ∧ y = -A.2)) ∨ 
      ((x = B.1 ∧ y = B.2) ∨ (x = -B.1 ∧ y = -B.2))) →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l3514_351473


namespace NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l3514_351470

/-- Given a parabola y^2 = 4px where p > 0, with two perpendicular chords OA and OB
    drawn from the vertex O(0,0), the locus of the projection of O onto AB
    is a circle with equation (x - 2p)^2 + y^2 = 4p^2 -/
theorem parabola_perpendicular_chords_locus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*p*x}
  let O := (0 : ℝ × ℝ)
  let perpendicular_chords := {(OA, OB) : (ℝ × ℝ) × (ℝ × ℝ) |
    O.1 = 0 ∧ O.2 = 0 ∧
    OA ∈ parabola ∧ OB ∈ parabola ∧
    (OA.2 - O.2) * (OB.2 - O.2) = -(OA.1 - O.1) * (OB.1 - O.1)}
  let projection := {M : ℝ × ℝ | ∃ (OA OB : ℝ × ℝ), (OA, OB) ∈ perpendicular_chords ∧
    (M.2 - O.2) * (OA.1 - OB.1) = (M.1 - O.1) * (OA.2 - OB.2)}
  projection = {(x, y) : ℝ × ℝ | (x - 2*p)^2 + y^2 = 4*p^2} :=
by sorry


end NUMINAMATH_CALUDE_parabola_perpendicular_chords_locus_l3514_351470


namespace NUMINAMATH_CALUDE_mike_practice_hours_l3514_351489

/-- Calculates the number of hours Mike practices every weekday -/
def weekday_practice_hours (days_in_week : ℕ) (practice_days_per_week : ℕ) 
  (saturday_hours : ℕ) (total_weeks : ℕ) (total_practice_hours : ℕ) : ℕ :=
  let total_practice_days := practice_days_per_week * total_weeks
  let total_saturdays := total_weeks
  let saturday_practice_hours := saturday_hours * total_saturdays
  let weekday_practice_hours := total_practice_hours - saturday_practice_hours
  let total_weekdays := (practice_days_per_week - 1) * total_weeks
  weekday_practice_hours / total_weekdays

/-- Theorem stating that Mike practices 3 hours every weekday -/
theorem mike_practice_hours : 
  weekday_practice_hours 7 6 5 3 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mike_practice_hours_l3514_351489


namespace NUMINAMATH_CALUDE_fib_80_mod_7_l3514_351484

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The period of the Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_80_mod_7 :
  fib 80 % 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fib_80_mod_7_l3514_351484


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_ten_l3514_351465

theorem ac_plus_bd_equals_negative_ten
  (a b c d : ℝ)
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 6) :
  a * c + b * d = -10 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_ten_l3514_351465


namespace NUMINAMATH_CALUDE_product_of_primes_minus_one_l3514_351496

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 0 ∧ m < n → n % m = 0 → m = 1

axiom every_nat_is_product_of_primes :
  ∀ n : Nat, n > 1 → ∃ (factors : List Nat), n = factors.prod ∧ ∀ p ∈ factors, isPrime p

theorem product_of_primes_minus_one (h : isPrime 11 ∧ isPrime 19) :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
by sorry

end NUMINAMATH_CALUDE_product_of_primes_minus_one_l3514_351496


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l3514_351448

theorem scavenger_hunt_items (tanya samantha lewis : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 → 
  lewis = 20 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l3514_351448


namespace NUMINAMATH_CALUDE_pairings_count_l3514_351403

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The number of possible pairings when choosing one bowl and one glass -/
def num_pairings : ℕ := num_bowls * num_glasses

/-- Theorem stating that the number of possible pairings is 25 -/
theorem pairings_count : num_pairings = 25 := by
  sorry

end NUMINAMATH_CALUDE_pairings_count_l3514_351403


namespace NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3514_351491

/-- Represents the work schedule of a tutor -/
structure TutorSchedule where
  cycle : ℕ

/-- Represents the lab schedule -/
structure LabSchedule where
  openDays : Fin 7 → Bool

/-- Calculates the next day all tutors work together -/
def nextJointWorkDay (emma noah olivia liam : TutorSchedule) (lab : LabSchedule) : ℕ :=
  sorry

theorem next_joint_work_day_is_360 :
  let emma : TutorSchedule := { cycle := 5 }
  let noah : TutorSchedule := { cycle := 8 }
  let olivia : TutorSchedule := { cycle := 9 }
  let liam : TutorSchedule := { cycle := 10 }
  let lab : LabSchedule := { openDays := fun d => d < 5 }
  nextJointWorkDay emma noah olivia liam lab = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3514_351491


namespace NUMINAMATH_CALUDE_arrangement_count_is_180_l3514_351499

/-- The number of ways to select 4 students from 5 and assign them to 3 subjects --/
def arrangement_count : ℕ := 180

/-- The total number of students --/
def total_students : ℕ := 5

/-- The number of students to be selected --/
def selected_students : ℕ := 4

/-- The number of subjects --/
def subject_count : ℕ := 3

/-- Theorem stating that the number of arrangements is 180 --/
theorem arrangement_count_is_180 :
  arrangement_count = 
    subject_count * 
    (Nat.choose total_students 2) * 
    (Nat.choose (total_students - 2) 1) * 
    (Nat.choose (total_students - 3) 1) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_180_l3514_351499


namespace NUMINAMATH_CALUDE_quadratic_extremum_l3514_351492

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2 / (3a),
    prove that the graph of y = f(x) has a maximum if a < 0 and a minimum if a > 0 -/
theorem quadratic_extremum (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - b^2 / (3 * a)
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_extremum_l3514_351492


namespace NUMINAMATH_CALUDE_sequence_sum_l3514_351419

def a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * a (n + 2) - 4 * (n + 3) * a (n + 1) + (4 * (n + 3) - 8) * a n

theorem sequence_sum (n : ℕ) : a n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3514_351419


namespace NUMINAMATH_CALUDE_james_tin_collection_l3514_351488

/-- The number of tins James collects in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collects on the first day -/
def first_day_tins : ℕ := 50

/-- The number of tins James collects on the second day -/
def second_day_tins : ℕ := 3 * first_day_tins

/-- The number of tins James collects on each of the remaining days (4th to 7th) -/
def remaining_days_tins : ℕ := 50

/-- The total number of tins James collects on the remaining days (4th to 7th) -/
def total_remaining_days_tins : ℕ := 4 * remaining_days_tins

/-- The number of tins James collects on the third day -/
def third_day_tins : ℕ := total_tins - first_day_tins - second_day_tins - total_remaining_days_tins

theorem james_tin_collection :
  second_day_tins - third_day_tins = 50 :=
sorry

end NUMINAMATH_CALUDE_james_tin_collection_l3514_351488


namespace NUMINAMATH_CALUDE_odd_function_property_l3514_351441

-- Define odd functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := 3 * f x + 5 * g x + 2

-- Theorem statement
theorem odd_function_property (f g : ℝ → ℝ) (a : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g) (hFa : F f g a = 3) : 
  F f g (-a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3514_351441


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_necessary_but_not_sufficient_condition_l3514_351439

-- Define set P
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}

-- Define set S as a function of m
def S (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for part I
theorem no_m_exists_for_equality :
  ¬ ∃ m : ℝ, P = S m :=
sorry

-- Theorem for part II
theorem necessary_but_not_sufficient_condition :
  ∀ m : ℝ, m ≤ 3 → (P ⊆ S m ∧ P ≠ S m) :=
sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_necessary_but_not_sufficient_condition_l3514_351439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3514_351407

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a₁ : ℝ)
  (d : ℝ)
  (h1 : arithmeticSequence a₁ d 1 + arithmeticSequence a₁ d 3 + arithmeticSequence a₁ d 5 = 15)
  (h2 : arithmeticSequence a₁ d 4 = 3) :
  d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3514_351407


namespace NUMINAMATH_CALUDE_tiling_ways_eq_fib_l3514_351463

/-- The number of ways to tile a 2 × n strip with 1 × 2 or 2 × 1 bricks -/
def tiling_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => tiling_ways (k + 1) + tiling_ways k

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => fib (k + 1) + fib k

theorem tiling_ways_eq_fib (n : ℕ) : tiling_ways n = fib n := by
  sorry

end NUMINAMATH_CALUDE_tiling_ways_eq_fib_l3514_351463


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3514_351476

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 4 is 1 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3514_351476


namespace NUMINAMATH_CALUDE_exists_n_for_m_l3514_351417

-- Define the function f(n) as the sum of n and its digits
def f (n : ℕ) : ℕ :=
  n + (Nat.digits 10 n).sum

-- Theorem statement
theorem exists_n_for_m (m : ℕ) :
  m > 0 → ∃ n : ℕ, f n = m ∨ f n = m + 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_m_l3514_351417


namespace NUMINAMATH_CALUDE_factor_expression_l3514_351486

theorem factor_expression (a : ℝ) : 198 * a^2 + 36 * a + 54 = 18 * (11 * a^2 + 2 * a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3514_351486


namespace NUMINAMATH_CALUDE_second_class_average_l3514_351480

/-- Proves that the average mark of the second class is 69.83 given the conditions of the problem -/
theorem second_class_average (students_class1 : ℕ) (students_class2 : ℕ) 
  (avg_class1 : ℝ) (total_avg : ℝ) :
  students_class1 = 39 →
  students_class2 = 35 →
  avg_class1 = 45 →
  total_avg = 56.75 →
  let total_students := students_class1 + students_class2
  let avg_class2 := (total_avg * total_students - avg_class1 * students_class1) / students_class2
  avg_class2 = 69.83 := by
sorry

end NUMINAMATH_CALUDE_second_class_average_l3514_351480


namespace NUMINAMATH_CALUDE_batsman_average_runs_l3514_351477

/-- The average runs scored by a batsman in a series of cricket matches. -/
def AverageRuns (first_10_avg : ℝ) (total_matches : ℕ) (overall_avg : ℝ) : Prop :=
  let first_10_total := first_10_avg * 10
  let total_runs := overall_avg * total_matches
  let next_10_total := total_runs - first_10_total
  let next_10_avg := next_10_total / 10
  next_10_avg = 30

/-- Theorem stating that given the conditions, the average runs scored in the next 10 matches is 30. -/
theorem batsman_average_runs : AverageRuns 40 20 35 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_runs_l3514_351477


namespace NUMINAMATH_CALUDE_total_hot_dogs_is_fifteen_l3514_351461

/-- Represents the number of hot dogs served at each meal -/
structure HotDogMeals where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that the total number of hot dogs served is 15 given the conditions -/
theorem total_hot_dogs_is_fifteen (h : HotDogMeals) :
  (h.breakfast = 2 * h.dinner) →
  (h.lunch = 9) →
  (h.lunch = h.breakfast + h.dinner + 3) →
  (h.breakfast + h.lunch + h.dinner = 15) := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dogs_is_fifteen_l3514_351461


namespace NUMINAMATH_CALUDE_miles_travel_time_l3514_351435

/-- Proves that if the distance between two cities is 57 miles and Miles takes 40 hours
    to complete 4 round trips, then Miles takes 10 hours for one round trip. -/
theorem miles_travel_time 
  (distance : ℝ) 
  (total_time : ℝ) 
  (num_round_trips : ℕ) 
  (h1 : distance = 57) 
  (h2 : total_time = 40) 
  (h3 : num_round_trips = 4) : 
  (total_time / num_round_trips) = 10 := by
  sorry


end NUMINAMATH_CALUDE_miles_travel_time_l3514_351435


namespace NUMINAMATH_CALUDE_product_of_special_reals_l3514_351401

/-- Given two real numbers a and b satisfying certain conditions, 
    their product is approximately 17.26 -/
theorem product_of_special_reals (a b : ℝ) 
  (sum_eq : a + b = 8)
  (fourth_power_sum : a^4 + b^4 = 272) :
  ∃ ε > 0, |a * b - 17.26| < ε :=
sorry

end NUMINAMATH_CALUDE_product_of_special_reals_l3514_351401


namespace NUMINAMATH_CALUDE_arithmetic_seq_product_l3514_351426

/-- An increasing arithmetic sequence of integers -/
def is_increasing_arithmetic_seq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_seq_product (b : ℕ → ℤ) 
  (h_seq : is_increasing_arithmetic_seq b)
  (h_prod : b 4 * b 5 = 15) :
  b 3 * b 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_product_l3514_351426


namespace NUMINAMATH_CALUDE_soccer_team_girls_l3514_351423

theorem soccer_team_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  present = 18 →
  boys + girls = total →
  present = (2 / 3 : ℚ) * boys + girls →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_girls_l3514_351423


namespace NUMINAMATH_CALUDE_arrangeStudents_eq_288_l3514_351411

/-- The number of ways to arrange 6 students in a 2x3 grid with constraints -/
def arrangeStudents : ℕ :=
  let totalStudents : ℕ := 6
  let rows : ℕ := 2
  let columns : ℕ := 3
  let positionsForA : ℕ := totalStudents
  let positionsForB : ℕ := 2  -- Not in same row or column as A
  let remainingStudents : ℕ := totalStudents - 2
  positionsForA * positionsForB * Nat.factorial remainingStudents

theorem arrangeStudents_eq_288 : arrangeStudents = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangeStudents_eq_288_l3514_351411


namespace NUMINAMATH_CALUDE_cubic_inequality_iff_open_interval_l3514_351424

theorem cubic_inequality_iff_open_interval :
  ∀ x : ℝ, x * (x^2 - 9) < 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_iff_open_interval_l3514_351424


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l3514_351420

/-- Represents the number of emails Jack received at different times of the day. -/
structure EmailCount where
  morning : Nat
  total : Nat

/-- Calculates the number of emails Jack received in the afternoon. -/
def afternoon_emails (e : EmailCount) : Nat :=
  e.total - e.morning

/-- Theorem: Jack received 1 email in the afternoon. -/
theorem jack_afternoon_emails :
  let e : EmailCount := { morning := 4, total := 5 }
  afternoon_emails e = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l3514_351420


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l3514_351450

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_two : 2 * log 5 10 + log 5 0.25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l3514_351450


namespace NUMINAMATH_CALUDE_S_description_l3514_351402

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let x := p.1
    let y := p.2
    ((5 = x + 3 ∧ (y - 6 ≤ 5 ∨ y - 6 = 5 / 2)) ∨
     (5 = y - 6 ∧ (x + 3 ≤ 5 ∨ x + 3 = 5 / 2)) ∨
     (x + 3 = y - 6 ∧ 5 = (x + 3) / 2))}

-- Define what it means to be parts of a right triangle
def isPartsOfRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a.1 = b.1 ∧ b.2 = c.2 ∧
    (c.1 - a.1) * (b.2 - a.2) = 0

-- Define what it means to have a separate point
def hasSeparatePoint (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ S ∧ ∀ q ∈ S, q ≠ p → ‖p - q‖ > 0

-- Theorem statement
theorem S_description :
  isPartsOfRightTriangle S ∧ hasSeparatePoint S :=
sorry

end NUMINAMATH_CALUDE_S_description_l3514_351402


namespace NUMINAMATH_CALUDE_new_students_average_age_l3514_351479

/-- Calculates the average age of new students joining a class --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_new_strength := original_strength + new_students
  let total_age_after := (original_strength + new_students) * new_average
  let total_age_before := original_strength * original_average
  let total_age_new_students := total_age_after - total_age_before
  (total_age_new_students / new_students : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3514_351479


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3514_351429

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 ≤ 16) ∧ 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a^2 - 1 = 16) → 
  a = 3 ∨ a = -3 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3514_351429


namespace NUMINAMATH_CALUDE_reflections_composition_is_translation_l3514_351405

/-- Four distinct points on a circle -/
structure CirclePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2 ∧
    (D.1 - center.1)^2 + (D.2 - center.2)^2 = radius^2

/-- Reflection across a line defined by two points -/
def reflect (p q : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translation of a point -/
def translate (v : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := (x.1 + v.1, x.2 + v.2)

/-- The main theorem stating that the composition of reflections is a translation -/
theorem reflections_composition_is_translation (points : CirclePoints) :
  ∃ (v : ℝ × ℝ), ∀ (x : ℝ × ℝ),
    reflect points.D points.A (reflect points.C points.D (reflect points.B points.C (reflect points.A points.B x))) = translate v x :=
sorry

end NUMINAMATH_CALUDE_reflections_composition_is_translation_l3514_351405


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3514_351415

/-- 
Proves that an arithmetic sequence with first term 2, common difference 3, 
and last term 2014 has 671 terms.
-/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 2 → d = 3 → last = 2014 → 
    last = a + (n - 1) * d → n = 671 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3514_351415


namespace NUMINAMATH_CALUDE_contrapositive_evenness_l3514_351482

theorem contrapositive_evenness (a b : ℤ) : 
  (Odd (a + b) → Odd a ∨ Odd b) = False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_evenness_l3514_351482


namespace NUMINAMATH_CALUDE_f_composition_l3514_351446

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 5

-- State the theorem
theorem f_composition (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (2 * x - 3) = 4 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l3514_351446


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l3514_351437

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 + 2*x ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l3514_351437


namespace NUMINAMATH_CALUDE_sidney_monday_jj_l3514_351434

/-- The number of jumping jacks Sidney did on Monday -/
def monday_jj : ℕ := sorry

/-- The number of jumping jacks Sidney did on Tuesday -/
def tuesday_jj : ℕ := 36

/-- The number of jumping jacks Sidney did on Wednesday -/
def wednesday_jj : ℕ := 40

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jj : ℕ := 50

/-- The total number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

/-- Theorem stating that Sidney did 20 jumping jacks on Monday -/
theorem sidney_monday_jj : monday_jj = 20 := by
  sorry

end NUMINAMATH_CALUDE_sidney_monday_jj_l3514_351434


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3514_351453

theorem fish_population_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) 
  (h1 : initial_marked = 100)
  (h2 : second_catch = 200)
  (h3 : marked_in_second = 25) :
  (initial_marked * second_catch) / marked_in_second = 800 :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l3514_351453


namespace NUMINAMATH_CALUDE_circle_outside_triangle_percentage_l3514_351490

theorem circle_outside_triangle_percentage
  (A : ℝ) -- Total area
  (A_intersection : ℝ) -- Area of intersection
  (A_triangle_outside : ℝ) -- Area of triangle outside circle
  (h1 : A > 0) -- Total area is positive
  (h2 : A_intersection = 0.45 * A) -- Intersection is 45% of total area
  (h3 : A_triangle_outside = 0.4 * A) -- Triangle outside is 40% of total area
  : (A - A_intersection - A_triangle_outside) / (A_intersection + (A - A_intersection - A_triangle_outside)) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_outside_triangle_percentage_l3514_351490


namespace NUMINAMATH_CALUDE_rectangular_wall_area_l3514_351404

theorem rectangular_wall_area : 
  ∀ (width length area : ℝ),
    width = 5.4 →
    length = 2.5 →
    area = width * length →
    area = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_wall_area_l3514_351404


namespace NUMINAMATH_CALUDE_bmw_sales_l3514_351408

theorem bmw_sales (total : ℕ) (mercedes_percent : ℚ) (nissan_percent : ℚ) (ford_percent : ℚ) (chevrolet_percent : ℚ) 
  (h_total : total = 300)
  (h_mercedes : mercedes_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_ford : ford_percent = 10 / 100)
  (h_chevrolet : chevrolet_percent = 18 / 100) :
  ↑(total - (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).num * total / (mercedes_percent + nissan_percent + ford_percent + chevrolet_percent).den) = 81 := by
  sorry


end NUMINAMATH_CALUDE_bmw_sales_l3514_351408


namespace NUMINAMATH_CALUDE_cubic_root_product_l3514_351410

theorem cubic_root_product (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + a - 5 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 5 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 5 = 0) → 
  a * b * c = 5/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3514_351410


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3514_351468

theorem necessary_not_sufficient :
  (∀ x : ℝ, -1 ≤ x ∧ x < 2 → -1 ≤ x ∧ x < 3) ∧
  ¬(∀ x : ℝ, -1 ≤ x ∧ x < 3 → -1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3514_351468


namespace NUMINAMATH_CALUDE_water_jar_problem_l3514_351455

theorem water_jar_problem (c1 c2 c3 : ℝ) (h1 : c1 > 0) (h2 : c2 > 0) (h3 : c3 > 0) 
  (h4 : c1 < c2) (h5 : c2 < c3) 
  (h6 : c1 / 6 = c2 / 5) (h7 : c2 / 5 = c3 / 7) : 
  (c1 / 6 + c2 / 5) / c3 = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_water_jar_problem_l3514_351455


namespace NUMINAMATH_CALUDE_max_value_x_minus_y_l3514_351413

theorem max_value_x_minus_y (θ : Real) (x y : Real)
  (h1 : x = Real.sin θ)
  (h2 : y = Real.cos θ)
  (h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi)
  (h4 : (x^2 + y^2)^2 = x + y) :
  ∃ (θ_max : Real), 
    0 ≤ θ_max ∧ θ_max ≤ 2 * Real.pi ∧
    ∀ (θ' : Real), 0 ≤ θ' ∧ θ' ≤ 2 * Real.pi →
      Real.sin θ' - Real.cos θ' ≤ Real.sin θ_max - Real.cos θ_max ∧
      Real.sin θ_max - Real.cos θ_max = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_y_l3514_351413


namespace NUMINAMATH_CALUDE_race_heartbeats_l3514_351460

/-- Calculates the total number of heartbeats during a race with varying heart rates. -/
def total_heartbeats (base_rate : ℕ) (distance : ℕ) (pace : ℕ) (rate_increase : ℕ) (increase_start : ℕ) : ℕ :=
  let total_time := distance * pace
  let base_beats := base_rate * total_time
  let increased_distance := distance - increase_start
  let increased_beats := increased_distance * (increased_distance + 1) * rate_increase / 2
  base_beats + increased_beats

/-- Theorem stating the total number of heartbeats during a 20-mile race 
    with specific heart rate conditions. -/
theorem race_heartbeats : 
  total_heartbeats 160 20 6 5 10 = 11475 :=
sorry

end NUMINAMATH_CALUDE_race_heartbeats_l3514_351460


namespace NUMINAMATH_CALUDE_inscribed_triangle_radius_l3514_351458

theorem inscribed_triangle_radius 
  (S : ℝ) 
  (α : ℝ) 
  (h1 : S > 0) 
  (h2 : 0 < α ∧ α < 2 * Real.pi) : 
  ∃ R : ℝ, R > 0 ∧ 
    R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_radius_l3514_351458


namespace NUMINAMATH_CALUDE_difference_of_squares_l3514_351472

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3514_351472


namespace NUMINAMATH_CALUDE_ratio_AB_BC_l3514_351497

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The diagram configuration -/
structure Diagram where
  rectangles : Fin 5 → Rectangle
  x : ℝ
  h1 : ∀ i, (rectangles i).width = x
  h2 : ∀ i, (rectangles i).length = 3 * x

/-- AB is the sum of two widths and one length -/
def AB (d : Diagram) : ℝ := 2 * d.x + 3 * d.x

/-- BC is the length of one rectangle -/
def BC (d : Diagram) : ℝ := 3 * d.x

theorem ratio_AB_BC (d : Diagram) : AB d / BC d = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_AB_BC_l3514_351497


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3514_351451

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 8*x + 1 = 0) ↔ ((x - 4)^2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3514_351451


namespace NUMINAMATH_CALUDE_probability_of_three_pointing_l3514_351440

/-- The number of people in the room -/
def n : ℕ := 5

/-- The probability of one person pointing at two specific others -/
def p : ℚ := 1 / 6

/-- The number of ways to choose 2 people out of n -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of a group of three all pointing at each other -/
def prob_three_pointing : ℚ := p^3

/-- The main theorem: probability of having a group of three all pointing at each other -/
theorem probability_of_three_pointing :
  (choose_two n : ℚ) * prob_three_pointing = 5 / 108 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_pointing_l3514_351440


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3514_351444

theorem polynomial_evaluation : (4 : ℝ)^4 + (4 : ℝ)^3 + (4 : ℝ)^2 + (4 : ℝ) + 1 = 341 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3514_351444


namespace NUMINAMATH_CALUDE_last_four_digits_of_7_to_5000_l3514_351447

theorem last_four_digits_of_7_to_5000 (h : 7^250 ≡ 1 [ZMOD 1250]) : 
  7^5000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_7_to_5000_l3514_351447


namespace NUMINAMATH_CALUDE_factor_expression_l3514_351432

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3514_351432


namespace NUMINAMATH_CALUDE_product_of_numbers_l3514_351422

theorem product_of_numbers (x y : ℝ) : x + y = 24 → x - y = 8 → x * y = 128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3514_351422


namespace NUMINAMATH_CALUDE_rectangle_width_l3514_351443

/-- Given a rectangle where the length is 2 cm shorter than the width and the perimeter is 16 cm, 
    the width of the rectangle is 5 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * w + 2 * (w - 2) = 16) : w = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3514_351443


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3514_351406

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + 2*α - 2005 = 0) → 
  (β^2 + 2*β - 2005 = 0) → 
  (α^2 + 3*α + β = 2003) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3514_351406


namespace NUMINAMATH_CALUDE_ladder_length_proof_l3514_351494

theorem ladder_length_proof (ladder_length wall_height : ℝ) : 
  wall_height = ladder_length + 8/3 →
  ∃ (ladder_base ladder_top : ℝ),
    ladder_base = 3/5 * ladder_length ∧
    ladder_top = 2/5 * wall_height ∧
    ladder_length^2 = ladder_base^2 + ladder_top^2 →
  ladder_length = 8/3 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_proof_l3514_351494


namespace NUMINAMATH_CALUDE_square_of_number_doubled_exceeds_fifth_l3514_351428

theorem square_of_number_doubled_exceeds_fifth : ∃ x : ℝ, 2 * x = (1/5) * x + 9 ∧ x^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_doubled_exceeds_fifth_l3514_351428


namespace NUMINAMATH_CALUDE_average_difference_l3514_351438

def number_of_students : ℕ := 120
def number_of_teachers : ℕ := 4
def class_sizes : List ℕ := [60, 30, 20, 10]

def t : ℚ := (List.sum class_sizes) / number_of_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / number_of_students

theorem average_difference : t - s = -11663/1000 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3514_351438


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3514_351495

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Mᶜ) = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3514_351495


namespace NUMINAMATH_CALUDE_problem_1_l3514_351431

theorem problem_1 (x y : ℝ) : (x - 2*y)^2 - x*(x + 3*y) - 4*y^2 = -7*x*y := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3514_351431


namespace NUMINAMATH_CALUDE_cosine_equality_in_range_l3514_351493

theorem cosine_equality_in_range (n : ℤ) :
  100 ≤ n ∧ n ≤ 300 ∧ Real.cos (n * π / 180) = Real.cos (140 * π / 180) → n = 220 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_in_range_l3514_351493


namespace NUMINAMATH_CALUDE_emma_savings_l3514_351462

theorem emma_savings (initial_savings withdrawal deposit final_savings : ℕ) : 
  initial_savings = 230 →
  final_savings = 290 →
  deposit = 2 * withdrawal →
  final_savings = initial_savings - withdrawal + deposit →
  withdrawal = 60 := by
sorry

end NUMINAMATH_CALUDE_emma_savings_l3514_351462


namespace NUMINAMATH_CALUDE_problem_statement_l3514_351487

theorem problem_statement : 
  let p := ∀ a b c : ℝ, a > b → a * c^2 > b * c^2
  let q := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0
  (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3514_351487


namespace NUMINAMATH_CALUDE_divisible_by_27_l3514_351400

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(5*n + 1) + 5^(n + 2) = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l3514_351400


namespace NUMINAMATH_CALUDE_estimate_above_120_l3514_351452

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  prob_100_to_110 : ℝ

/-- Estimates the number of students scoring above a given threshold -/
def estimate_students_above (sd : ScoreDistribution) (threshold : ℝ) : ℕ := sorry

/-- The main theorem to prove -/
theorem estimate_above_120 (sd : ScoreDistribution) 
  (h1 : sd.total_students = 50)
  (h2 : sd.mean = 110)
  (h3 : sd.std_dev = 10)
  (h4 : sd.prob_100_to_110 = 0.36) :
  estimate_students_above sd 120 = 7 := by sorry

end NUMINAMATH_CALUDE_estimate_above_120_l3514_351452


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3514_351469

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 9 * x + 6 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3514_351469


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l3514_351433

theorem loan_principal_calculation (principal : ℝ) : 
  (principal * 0.08 * 10 = principal - 1540) → principal = 7700 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l3514_351433


namespace NUMINAMATH_CALUDE_optimal_price_increase_maximizes_profit_l3514_351471

/-- Represents the daily profit function for a meal set -/
structure MealSet where
  baseProfit : ℝ
  baseSales : ℝ
  salesDecreaseRate : ℝ

/-- Calculate the daily profit for a meal set given a price increase -/
def dailyProfit (set : MealSet) (priceIncrease : ℝ) : ℝ :=
  (set.baseProfit + priceIncrease) * (set.baseSales - set.salesDecreaseRate * priceIncrease)

/-- The optimal price increase for meal set A maximizes the total profit -/
theorem optimal_price_increase_maximizes_profit 
  (setA setB : MealSet)
  (totalPriceIncrease : ℝ)
  (hA : setA = { baseProfit := 8, baseSales := 90, salesDecreaseRate := 4 })
  (hB : setB = { baseProfit := 10, baseSales := 70, salesDecreaseRate := 2 })
  (hTotal : totalPriceIncrease = 10) :
  ∃ (x : ℝ), x = 4 ∧ 
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ totalPriceIncrease →
      dailyProfit setA x + dailyProfit setB (totalPriceIncrease - x) ≥
      dailyProfit setA y + dailyProfit setB (totalPriceIncrease - y) :=
by sorry


end NUMINAMATH_CALUDE_optimal_price_increase_maximizes_profit_l3514_351471


namespace NUMINAMATH_CALUDE_sqrt_twelve_over_sqrt_two_equals_sqrt_six_l3514_351459

theorem sqrt_twelve_over_sqrt_two_equals_sqrt_six : 
  (Real.sqrt 12) / (Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_over_sqrt_two_equals_sqrt_six_l3514_351459


namespace NUMINAMATH_CALUDE_girls_fraction_l3514_351436

theorem girls_fraction (T G B : ℝ) (x : ℝ) 
  (h1 : x * G = (1 / 5) * T)  -- Some fraction of girls is 1/5 of total
  (h2 : B / G = 1.5)          -- Ratio of boys to girls is 1.5
  (h3 : T = B + G)            -- Total is sum of boys and girls
  : x = 1 / 2 := by 
  sorry

end NUMINAMATH_CALUDE_girls_fraction_l3514_351436


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3514_351457

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (5 * x^2 + 4) = Real.sqrt 29}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l3514_351457


namespace NUMINAMATH_CALUDE_sin_square_inequality_l3514_351409

theorem sin_square_inequality (n : ℕ+) (x : ℝ) :
  n * (Real.sin x)^2 ≥ Real.sin x * Real.sin (n * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_square_inequality_l3514_351409


namespace NUMINAMATH_CALUDE_range_of_m_l3514_351421

theorem range_of_m (α : ℝ) (m : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sqrt 3 * Real.sin α + Real.cos α = m) :
  m ∈ Set.Ioo 1 2 ∪ {2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3514_351421


namespace NUMINAMATH_CALUDE_problem_statement_l3514_351454

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x - 2*a*x + 2*a

theorem problem_statement (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 1/4) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ →
    |g a x₁ - g a x₂| < 2*a*|1/x₁ - 1/x₂|) →
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3514_351454


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l3514_351416

theorem binomial_coefficient_x_cubed_in_x_plus_one_to_sixth : 
  (Finset.range 7).sum (fun k => Nat.choose 6 k * X^k) = 
    X^6 + 6*X^5 + 15*X^4 + 20*X^3 + 15*X^2 + 6*X + 1 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_cubed_in_x_plus_one_to_sixth_l3514_351416


namespace NUMINAMATH_CALUDE_boys_camp_total_l3514_351483

theorem boys_camp_total (total_boys : ℕ) : 
  (total_boys : ℝ) * 0.2 * 0.7 = 21 → total_boys = 150 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3514_351483


namespace NUMINAMATH_CALUDE_ballet_arrangement_l3514_351412

/-- The number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- The distance between each girl and her two assigned boys (in meters) -/
def distance : ℕ := 5

/-- The maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := 20

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem ballet_arrangement (n : ℕ) (d : ℕ) (m : ℕ) 
  (h1 : n = num_boys) 
  (h2 : d = distance) 
  (h3 : m = max_girls) :
  m = n * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ballet_arrangement_l3514_351412


namespace NUMINAMATH_CALUDE_randolph_is_55_l3514_351498

def sherry_age : ℕ := 25

def sydney_age : ℕ := 2 * sherry_age

def randolph_age : ℕ := sydney_age + 5

theorem randolph_is_55 : randolph_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_randolph_is_55_l3514_351498


namespace NUMINAMATH_CALUDE_cos_sin_shift_l3514_351425

theorem cos_sin_shift (x : ℝ) : 
  Real.cos (x/2 - Real.pi/4) = Real.sin (x/2 + Real.pi/4) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_shift_l3514_351425


namespace NUMINAMATH_CALUDE_pyramid_side_length_l3514_351442

/-- Represents a pyramid with a rectangular base ABCD and vertex E above A -/
structure Pyramid where
  -- Base side lengths
  AB : ℝ
  BC : ℝ
  -- Angles
  BCE : ℝ
  ADE : ℝ

/-- Theorem: In a pyramid with given conditions, BC = 2√2 -/
theorem pyramid_side_length (p : Pyramid)
  (h_AB : p.AB = 4)
  (h_BCE : p.BCE = Real.pi / 3)  -- 60 degrees in radians
  (h_ADE : p.ADE = Real.pi / 4)  -- 45 degrees in radians
  : p.BC = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_side_length_l3514_351442


namespace NUMINAMATH_CALUDE_joaos_chocolates_l3514_351478

theorem joaos_chocolates :
  ∃! n : ℕ, 30 < n ∧ n < 100 ∧ n % 7 = 1 ∧ n % 10 = 2 ∧ n = 92 := by
  sorry

end NUMINAMATH_CALUDE_joaos_chocolates_l3514_351478


namespace NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l3514_351464

/-- A regular dodecagon is a polygon with 12 sides. -/
def RegularDodecagon : Type := Unit

/-- The sum of exterior angles of a polygon. -/
def SumOfExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360°. -/
theorem sum_exterior_angles_dodecagon :
  SumOfExteriorAngles RegularDodecagon = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l3514_351464


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3514_351445

theorem problem_1 : (-1/2)⁻¹ + (3 - Real.pi)^0 + (-3)^2 = 8 := by sorry

theorem problem_2 (a : ℝ) : a^2 * a^4 - (-2*a^2)^3 - 3*a^2 + a^2 = 9*a^6 - 2*a^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3514_351445


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3514_351481

theorem simplify_trig_expression (x : ℝ) : 
  ((1 + Real.sin x) / Real.cos x) * (Real.sin (2 * x) / (2 * (Real.cos (π/4 - x/2))^2)) = 2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3514_351481


namespace NUMINAMATH_CALUDE_unique_non_divisible_by_3_l3514_351418

def is_divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_non_divisible_by_3 :
  let numbers : List ℕ := [3543, 3555, 3567, 3573, 3581]
  ∀ n ∈ numbers, ¬(is_divisible_by_3 n) → n = 3581 ∧ units_digit n + tens_digit n = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_non_divisible_by_3_l3514_351418


namespace NUMINAMATH_CALUDE_factorial_division_l3514_351475

theorem factorial_division (h : Nat.factorial 7 = 5040) :
  Nat.factorial 7 / Nat.factorial 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3514_351475


namespace NUMINAMATH_CALUDE_unique_be_length_l3514_351456

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a unit square ABCD -/
def UnitSquare : (Point × Point × Point × Point) :=
  (⟨0, 0⟩, ⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 1⟩)

/-- Definition of perpendicularity between two line segments -/
def Perpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

/-- Theorem: In a unit square ABCD, with points E on BC, F on CD, and G on DA,
    if AE ⊥ EF, EF ⊥ FG, and GA = 404/1331, then BE = 9/11 -/
theorem unique_be_length (A B C D E F G : Point)
  (square : (A, B, C, D) = UnitSquare)
  (e_on_bc : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = ⟨1, t⟩)
  (f_on_cd : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = ⟨1 - t, 1⟩)
  (g_on_da : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = ⟨0, 1 - t⟩)
  (ae_perp_ef : Perpendicular A E E F)
  (ef_perp_fg : Perpendicular E F F G)
  (ga_length : (G.x - A.x)^2 + (G.y - A.y)^2 = (404/1331)^2) :
  (E.x - B.x)^2 + (E.y - B.y)^2 = (9/11)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_be_length_l3514_351456
