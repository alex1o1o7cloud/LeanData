import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_18_and_640_l692_69211

theorem smallest_n_divisible_by_18_and_640 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(18 ∣ m^2) ∨ ¬(640 ∣ m^3))) ∧ 
  (18 ∣ n^2) ∧ (640 ∣ n^3) :=
by
  use 120
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_18_and_640_l692_69211


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l692_69201

/-- Given a polynomial x^4 + 5x^3 + 6x^2 + 5x + 1 with complex roots, 
    the sum of the cubes of its roots is -54 -/
theorem sum_of_cubes_of_roots : 
  ∀ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁^4 + 5*x₁^3 + 6*x₁^2 + 5*x₁ + 1 = 0) →
    (x₂^4 + 5*x₂^3 + 6*x₂^2 + 5*x₂ + 1 = 0) →
    (x₃^4 + 5*x₃^3 + 6*x₃^2 + 5*x₃ + 1 = 0) →
    (x₄^4 + 5*x₄^3 + 6*x₄^2 + 5*x₄ + 1 = 0) →
    x₁^3 + x₂^3 + x₃^3 + x₄^3 = -54 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l692_69201


namespace NUMINAMATH_CALUDE_darnel_jogging_distance_l692_69255

theorem darnel_jogging_distance (sprint_distance : Real) (extra_sprint : Real) : 
  sprint_distance = 0.875 →
  extra_sprint = 0.125 →
  sprint_distance = (sprint_distance - extra_sprint) + extra_sprint →
  (sprint_distance - extra_sprint) = 0.750 := by
sorry

end NUMINAMATH_CALUDE_darnel_jogging_distance_l692_69255


namespace NUMINAMATH_CALUDE_factorial_10_trailing_zeros_base_15_l692_69276

/-- The number of trailing zeros in the base 15 representation of a natural number -/
def trailingZerosBase15 (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in the base 15 representation of 10! is 2 -/
theorem factorial_10_trailing_zeros_base_15 : 
  trailingZerosBase15 (factorial 10) = 2 := by sorry

end NUMINAMATH_CALUDE_factorial_10_trailing_zeros_base_15_l692_69276


namespace NUMINAMATH_CALUDE_scalene_triangles_count_l692_69268

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 20 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem scalene_triangles_count :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S → is_valid_scalene_triangle t.1 t.2.1 t.2.2) ∧
    S.card > 7 :=
sorry

end NUMINAMATH_CALUDE_scalene_triangles_count_l692_69268


namespace NUMINAMATH_CALUDE_novel_reading_ratio_l692_69299

theorem novel_reading_ratio :
  ∀ (jordan alexandre : ℕ),
    jordan = 120 →
    jordan = alexandre + 108 →
    (alexandre : ℚ) / jordan = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_novel_reading_ratio_l692_69299


namespace NUMINAMATH_CALUDE_bianca_cupcakes_l692_69250

/-- The number of cupcakes Bianca initially made -/
def initial_cupcakes : ℕ := 14

/-- The number of cupcakes Bianca sold -/
def sold_cupcakes : ℕ := 6

/-- The number of additional cupcakes Bianca made -/
def additional_cupcakes : ℕ := 17

/-- The final number of cupcakes Bianca had -/
def final_cupcakes : ℕ := 25

theorem bianca_cupcakes : 
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_bianca_cupcakes_l692_69250


namespace NUMINAMATH_CALUDE_eulerian_path_implies_at_most_two_odd_vertices_l692_69232

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def hasOddDegree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph. -/
def hasEulerianPath (G : Graph V) : Prop := sorry

/-- The main theorem: If a graph has an Eulerian path, 
    then the number of vertices with odd degree is at most 2. -/
theorem eulerian_path_implies_at_most_two_odd_vertices 
  (V : Type*) (G : Graph V) : 
  hasEulerianPath G → 
  ∃ (n : ℕ), n ≤ 2 ∧ (∃ (S : Finset V), S.card = n ∧ 
    ∀ v, v ∈ S ↔ hasOddDegree G v) := by
  sorry

end NUMINAMATH_CALUDE_eulerian_path_implies_at_most_two_odd_vertices_l692_69232


namespace NUMINAMATH_CALUDE_max_area_triangle_line_circle_l692_69200

/-- The maximum area of a triangle formed by the origin and two intersection points of a line and a unit circle --/
theorem max_area_triangle_line_circle : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line (k : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 1}
  let intersectionPoints (k : ℝ) := circle ∩ line k
  let triangleArea (A B : ℝ × ℝ) := (1/2) * abs (A.1 * B.2 - A.2 * B.1)
  ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersectionPoints k → B ∈ intersectionPoints k → A ≠ B →
    triangleArea A B ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_line_circle_l692_69200


namespace NUMINAMATH_CALUDE_min_sum_squares_l692_69277

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  x^2 + y^2 + z^2 ≥ 18/7 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l692_69277


namespace NUMINAMATH_CALUDE_train_speed_calculation_l692_69222

/-- Proves that given the conditions of two trains passing each other, the speed of the first train is 72 kmph -/
theorem train_speed_calculation (length_train1 length_train2 speed_train2 time_to_cross : ℝ) 
  (h1 : length_train1 = 380)
  (h2 : length_train2 = 540)
  (h3 : speed_train2 = 36)
  (h4 : time_to_cross = 91.9926405887529)
  : (length_train1 + length_train2) / time_to_cross * 3.6 + speed_train2 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l692_69222


namespace NUMINAMATH_CALUDE_show_episodes_l692_69271

/-- Calculates the number of episodes in a show given the watching conditions -/
def num_episodes (days : ℕ) (episode_length : ℕ) (hours_per_day : ℕ) : ℕ :=
  (days * hours_per_day * 60) / episode_length

/-- Proves that the number of episodes in the show is 20 -/
theorem show_episodes : num_episodes 5 30 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_l692_69271


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l692_69225

theorem fraction_sum_simplification : 1 / 462 + 23 / 42 = 127 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l692_69225


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l692_69291

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ k : ℕ+, k < n → ¬(1023 * k.val ≡ 2147 * k.val [ZMOD 30])) ∧ 
  (1023 * n.val ≡ 2147 * n.val [ZMOD 30]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l692_69291


namespace NUMINAMATH_CALUDE_range_of_a_l692_69240

def M : Set ℝ := {x | |x| < 1}
def N (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : M ∪ N a = M) : a ∈ Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l692_69240


namespace NUMINAMATH_CALUDE_train_length_l692_69221

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, 
  (abs (length - 129.96) < 0.01) ∧ (length = speed * 1000 / 3600 * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l692_69221


namespace NUMINAMATH_CALUDE_complex_numbers_with_extreme_arguments_l692_69285

open Complex

/-- The complex numbers with smallest and largest arguments satisfying |z - 5 - 5i| = 5 -/
theorem complex_numbers_with_extreme_arguments :
  ∃ (z₁ z₂ : ℂ),
    (∀ z : ℂ, abs (z - (5 + 5*I)) = 5 →
      arg z₁ ≤ arg z ∧ arg z ≤ arg z₂) ∧
    z₁ = 5 ∧
    z₂ = 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_with_extreme_arguments_l692_69285


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l692_69259

/-- Given two vectors a and b in R², prove that if k*a + b is perpendicular to a - b, then k = 1. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 1))
  (h2 : b = (-1, 1))
  (h3 : (k * a.1 + b.1) * (a.1 - b.1) + (k * a.2 + b.2) * (a.2 - b.2) = 0) :
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l692_69259


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l692_69275

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * x * Real.log x ≥ -x^2 + a*x - 3) → 
  a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l692_69275


namespace NUMINAMATH_CALUDE_expression_simplification_l692_69295

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2) * (x - 3)) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l692_69295


namespace NUMINAMATH_CALUDE_fourth_root_of_390820584961_l692_69205

theorem fourth_root_of_390820584961 :
  let n : ℕ := 390820584961
  let expansion : ℕ := 1 * 75^4 + 4 * 75^3 + 6 * 75^2 + 4 * 75 + 1
  n = expansion →
  (n : ℝ) ^ (1/4 : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_390820584961_l692_69205


namespace NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l692_69263

/-- Represents the five points on the circle -/
inductive CirclePoint
  | one
  | two
  | three
  | four
  | five

/-- Determines if a point is odd-numbered -/
def isOdd (p : CirclePoint) : Bool :=
  match p with
  | CirclePoint.one => true
  | CirclePoint.two => false
  | CirclePoint.three => true
  | CirclePoint.four => false
  | CirclePoint.five => true

/-- Determines the next point based on the current point -/
def nextPoint (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.one => CirclePoint.two
  | CirclePoint.two => CirclePoint.four
  | CirclePoint.three => CirclePoint.four
  | CirclePoint.four => CirclePoint.one
  | CirclePoint.five => CirclePoint.one

/-- Calculates the position after a given number of jumps -/
def positionAfterJumps (start : CirclePoint) (jumps : Nat) : CirclePoint :=
  match jumps with
  | 0 => start
  | n + 1 => nextPoint (positionAfterJumps start n)

theorem bug_position_after_1995_jumps :
  positionAfterJumps CirclePoint.five 1995 = CirclePoint.four := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l692_69263


namespace NUMINAMATH_CALUDE_multiple_of_number_l692_69230

theorem multiple_of_number (n : ℝ) (h : n = 6) : ∃ k : ℝ, 3 * n - 6 = k * n ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_number_l692_69230


namespace NUMINAMATH_CALUDE_candy_game_solution_l692_69288

/-- The number of questions Vanya answered correctly in the candy game -/
def correct_answers : ℕ := 15

/-- The total number of questions asked in the game -/
def total_questions : ℕ := 50

/-- The number of candies gained for a correct answer -/
def correct_reward : ℕ := 7

/-- The number of candies lost for an incorrect answer -/
def incorrect_penalty : ℕ := 3

theorem candy_game_solution :
  correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty :=
by sorry

end NUMINAMATH_CALUDE_candy_game_solution_l692_69288


namespace NUMINAMATH_CALUDE_sufficient_condition_for_hyperbola_not_necessary_condition_for_hyperbola_l692_69212

/-- Represents a hyperbola equation with parameter m -/
structure HyperbolaEquation (m : ℝ) :=
  (x y : ℝ)
  (eq : x^2 / (2 + m) - y^2 / (1 + m) = 1)

/-- Predicate to check if the equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  (2 + m ≠ 0) ∧ (1 + m ≠ 0) ∧ ((2 + m) * (1 + m) > 0)

theorem sufficient_condition_for_hyperbola (m : ℝ) :
  m > -1 → is_hyperbola m :=
sorry

theorem not_necessary_condition_for_hyperbola :
  ∃ m : ℝ, is_hyperbola m ∧ ¬(m > -1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_hyperbola_not_necessary_condition_for_hyperbola_l692_69212


namespace NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l692_69231

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l692_69231


namespace NUMINAMATH_CALUDE_sets_inclusion_l692_69243

-- Define the sets M, N, and P
def M : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}
def N : Set ℝ := {x | Real.cos (2 * x) = 0}
def P : Set ℝ := {a | Real.sin (2 * a) = 1}

-- State the theorem
theorem sets_inclusion : P ⊆ N ∧ N ⊆ M := by sorry

end NUMINAMATH_CALUDE_sets_inclusion_l692_69243


namespace NUMINAMATH_CALUDE_inequality_equivalence_l692_69278

theorem inequality_equivalence (x : ℝ) : 
  (|x + 3| + |1 - x|) / (x + 2016) < 1 ↔ x < -2016 ∨ (-1009 < x ∧ x < 1007) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l692_69278


namespace NUMINAMATH_CALUDE_function_equality_l692_69237

theorem function_equality (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 4 * x^2 + 2 / x + 2) → 
  (∀ x, g x = x^2 - 3 * x + m) → 
  f 3 - g 3 = 1 → 
  m = 113 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_equality_l692_69237


namespace NUMINAMATH_CALUDE_abs_inequality_l692_69242

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l692_69242


namespace NUMINAMATH_CALUDE_lindsay_daily_income_l692_69283

/-- Represents Doctor Lindsay's work schedule and patient fees --/
structure DoctorSchedule where
  adult_patients_per_hour : ℕ
  child_patients_per_hour : ℕ
  adult_fee : ℕ
  child_fee : ℕ
  hours_per_day : ℕ

/-- Calculates Doctor Lindsay's daily income based on her schedule --/
def daily_income (schedule : DoctorSchedule) : ℕ :=
  (schedule.adult_patients_per_hour * schedule.adult_fee +
   schedule.child_patients_per_hour * schedule.child_fee) *
  schedule.hours_per_day

/-- Theorem stating Doctor Lindsay's daily income --/
theorem lindsay_daily_income :
  ∃ (schedule : DoctorSchedule),
    schedule.adult_patients_per_hour = 4 ∧
    schedule.child_patients_per_hour = 3 ∧
    schedule.adult_fee = 50 ∧
    schedule.child_fee = 25 ∧
    schedule.hours_per_day = 8 ∧
    daily_income schedule = 2200 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_daily_income_l692_69283


namespace NUMINAMATH_CALUDE_train_length_train_length_specific_l692_69251

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) : Real :=
  let relative_speed := train_speed + man_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 114.99 kmph passing a man walking at 5 kmph in the opposite direction in 6 seconds has a length of approximately 199.98 meters. -/
theorem train_length_specific : 
  ∃ (ε : Real), ε > 0 ∧ abs (train_length 114.99 5 6 - 199.98) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_specific_l692_69251


namespace NUMINAMATH_CALUDE_unique_solution_l692_69213

/-- The system of equations and constraint -/
def system (x y z w : ℝ) : Prop :=
  x = Real.sin (z + w + z * w * x) ∧
  y = Real.sin (w + x + w * x * y) ∧
  z = Real.sin (x + y + x * y * z) ∧
  w = Real.sin (y + z + y * z * w) ∧
  Real.cos (x + y + z + w) = 1

/-- There exists exactly one solution to the system -/
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l692_69213


namespace NUMINAMATH_CALUDE_sum_53_to_100_l692_69290

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_53_to_100 (h : sum_range 51 100 = 3775) : sum_range 53 100 = 3672 := by
  sorry

end NUMINAMATH_CALUDE_sum_53_to_100_l692_69290


namespace NUMINAMATH_CALUDE_number_of_happy_arrangements_l692_69280

/-- Represents the types of chains --/
inductive Chain : Type
| Silver : Chain
| Gold : Chain
| Iron : Chain

/-- Represents the types of stones --/
inductive Stone : Type
| CubicZirconia : Stone
| Emerald : Stone
| Quartz : Stone

/-- Represents the types of pendants --/
inductive Pendant : Type
| Star : Pendant
| Sun : Pendant
| Moon : Pendant

/-- Represents a piece of jewelry --/
structure Jewelry :=
  (chain : Chain)
  (stone : Stone)
  (pendant : Pendant)

/-- Represents an arrangement of three pieces of jewelry --/
structure Arrangement :=
  (left : Jewelry)
  (middle : Jewelry)
  (right : Jewelry)

/-- Predicate to check if an arrangement satisfies Polina's conditions --/
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr.middle.chain = Chain.Iron ∧ arr.middle.pendant = Pendant.Sun) ∧
  ((arr.left.chain = Chain.Gold ∧ arr.right.chain = Chain.Silver) ∨
   (arr.left.chain = Chain.Silver ∧ arr.right.chain = Chain.Gold)) ∧
  (arr.left.stone ≠ arr.middle.stone ∧ arr.left.stone ≠ arr.right.stone ∧ arr.middle.stone ≠ arr.right.stone) ∧
  (arr.left.pendant ≠ arr.middle.pendant ∧ arr.left.pendant ≠ arr.right.pendant ∧ arr.middle.pendant ≠ arr.right.pendant) ∧
  (arr.left.chain ≠ arr.middle.chain ∧ arr.left.chain ≠ arr.right.chain ∧ arr.middle.chain ≠ arr.right.chain)

/-- The theorem to be proved --/
theorem number_of_happy_arrangements :
  ∃! (n : ℕ), ∃ (arrangements : Finset Arrangement),
    arrangements.card = n ∧
    (∀ arr ∈ arrangements, satisfiesConditions arr) ∧
    (∀ arr : Arrangement, satisfiesConditions arr → arr ∈ arrangements) :=
sorry

end NUMINAMATH_CALUDE_number_of_happy_arrangements_l692_69280


namespace NUMINAMATH_CALUDE_complement_union_theorem_l692_69210

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 2, 4}

theorem complement_union_theorem : (U \ B) ∪ A = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l692_69210


namespace NUMINAMATH_CALUDE_tens_digit_of_4032_pow_4033_minus_4036_l692_69273

theorem tens_digit_of_4032_pow_4033_minus_4036 :
  (4032^4033 - 4036) % 100 / 10 = 9 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_4032_pow_4033_minus_4036_l692_69273


namespace NUMINAMATH_CALUDE_polynomial_division_result_l692_69267

-- Define the polynomials f and d
def f (x : ℝ) : ℝ := 3 * x^4 - 9 * x^3 + 6 * x^2 + 2 * x - 5
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

-- State the theorem
theorem polynomial_division_result :
  ∃ (q r : ℝ → ℝ), 
    (∀ x, f x = q x * d x + r x) ∧ 
    (∀ x, r x = 14) ∧
    (q 1 + r (-1) = 17) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l692_69267


namespace NUMINAMATH_CALUDE_man_speed_against_current_l692_69249

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific speeds, the man's speed against the current is 20 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l692_69249


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l692_69245

theorem p_sufficient_not_necessary_for_q :
  ∃ (p q : Prop),
    (p ↔ (∃ x : ℝ, x = 2)) ∧
    (q ↔ (∃ x : ℝ, (x - 2) * (x + 3) = 0)) ∧
    (p → q) ∧
    ¬(q → p) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l692_69245


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_l692_69216

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board :=
  (side : ℕ)
  (pieces : ℕ)

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  let piece_area := b.side * b.side / b.pieces
  let piece_perimeter := if piece_area = 4 then 10 else 8
  (b.pieces * piece_perimeter - 4 * b.side) / 2

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30 :
  max_cut_length { side := 30, pieces := 225 } = 1065 :=
sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_l692_69216


namespace NUMINAMATH_CALUDE_complex_product_real_implies_m_equals_negative_one_l692_69292

theorem complex_product_real_implies_m_equals_negative_one (m : ℂ) : 
  (∃ (r : ℝ), (m^2 + Complex.I) * (1 + m * Complex.I) = r) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_m_equals_negative_one_l692_69292


namespace NUMINAMATH_CALUDE_hyperbola_equation_l692_69203

/-- Given an ellipse and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b c m n c' : ℝ) (e e' : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + y^2 = 2) →  -- Ellipse equation
  (a^2 = 2 ∧ b^2 = 1) →             -- Semi-major and semi-minor axes of ellipse
  (c = (a^2 - b^2).sqrt) →          -- Focal length of ellipse
  (e = c / a) →                     -- Eccentricity of ellipse
  (m = a) →                         -- Semi-major axis of hyperbola
  (e' * e = 1) →                    -- Product of eccentricities
  (c' = m * e') →                   -- Focal length of hyperbola
  (n^2 = c'^2 - m^2) →              -- Semi-minor axis of hyperbola
  (∀ x y : ℝ, y^2 / n^2 - x^2 / m^2 = 1) →  -- Standard form of hyperbola
  (∀ x y : ℝ, y^2 - x^2 = 2) :=     -- Desired hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l692_69203


namespace NUMINAMATH_CALUDE_expression_simplification_l692_69264

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) = (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l692_69264


namespace NUMINAMATH_CALUDE_emmas_room_width_l692_69286

theorem emmas_room_width (w : ℝ) : 
  w > 0 → -- width is positive
  40 = (1/6) * (w * 20) → -- 40 sq ft of tiles cover 1/6 of the room
  w = 12 := by
sorry

end NUMINAMATH_CALUDE_emmas_room_width_l692_69286


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l692_69298

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 31*x^2 - 34*x - 7

-- Theorem statement
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 31*x^2 + (-34)*x + (-7)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d e : ℚ, ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧
  -- 3+√2 is a root
  p (3 + Real.sqrt 2) = 0 ∧
  -- 2-√5 is a root
  p (2 - Real.sqrt 5) = 0 :=
sorry

end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l692_69298


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l692_69253

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+1)*x + a = 0}

-- Define the complement of B in ℝ
def C_ℝB (a : ℝ) : Set ℝ := {x : ℝ | x ∉ B a}

-- Statement 1
theorem intersection_A_complement_B_when_a_2 :
  A ∩ C_ℝB 2 = {-3} :=
sorry

-- Statement 2
theorem a_values_when_A_union_B_equals_A :
  {a : ℝ | A ∪ B a = A} = {-3, 1} :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_a_2_a_values_when_A_union_B_equals_A_l692_69253


namespace NUMINAMATH_CALUDE_password_theorem_triangle_password_theorem_factorization_theorem_l692_69265

-- Part 1
def password_generator (x y : ℤ) : List ℤ :=
  [x * 10000 + (x - y) * 100 + (x + y),
   x * 10000 + (x + y) * 100 + (x - y),
   (x - y) * 10000 + x * 100 + (x + y)]

theorem password_theorem :
  password_generator 21 7 = [211428, 212814, 142128] :=
sorry

-- Part 2
def right_triangle_password (x y : ℝ) : ℝ :=
  x * y * (x^2 + y^2)

theorem triangle_password_theorem :
  ∀ x y : ℝ,
  x + y + 13 = 30 →
  x^2 + y^2 = 13^2 →
  right_triangle_password x y = 10140 :=
sorry

-- Part 3
def polynomial_factorization (m n : ℤ) : Prop :=
  ∀ x : ℤ,
  x^3 + (m - 3*n)*x^2 - n*x - 21 = (x - 3)*(x + 1)*(x + 7)

theorem factorization_theorem :
  polynomial_factorization 56 17 :=
sorry

end NUMINAMATH_CALUDE_password_theorem_triangle_password_theorem_factorization_theorem_l692_69265


namespace NUMINAMATH_CALUDE_intersection_dot_product_l692_69294

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define the dot product of vectors OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- The theorem statement
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points A B l → dot_product A B = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l692_69294


namespace NUMINAMATH_CALUDE_boat_distance_proof_l692_69248

/-- The speed of the boat in still water (mph) -/
def boat_speed : ℝ := 15.6

/-- The time taken for the trip against the current (hours) -/
def time_against : ℝ := 8

/-- The time taken for the return trip with the current (hours) -/
def time_with : ℝ := 5

/-- The speed of the current (mph) -/
def current_speed : ℝ := 3.6

/-- The distance traveled by the boat (miles) -/
def distance : ℝ := 96

theorem boat_distance_proof :
  distance = (boat_speed - current_speed) * time_against ∧
  distance = (boat_speed + current_speed) * time_with :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l692_69248


namespace NUMINAMATH_CALUDE_gillians_phone_bill_l692_69241

theorem gillians_phone_bill (x : ℝ) : 
  (12 * (x * 1.1) = 660) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_gillians_phone_bill_l692_69241


namespace NUMINAMATH_CALUDE_saline_drip_duration_l692_69274

/-- Calculates the duration of a saline drip treatment -/
theorem saline_drip_duration 
  (drop_rate : ℕ) 
  (drops_per_ml : ℚ) 
  (total_volume : ℚ) : 
  drop_rate = 20 →
  drops_per_ml = 100 / 5 →
  total_volume = 120 →
  (total_volume * drops_per_ml / drop_rate) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_saline_drip_duration_l692_69274


namespace NUMINAMATH_CALUDE_andrew_work_hours_l692_69206

/-- The number of days Andrew worked on his Science report -/
def days_worked : ℝ := 3

/-- The number of hours Andrew worked each day -/
def hours_per_day : ℝ := 2.5

/-- The total number of hours Andrew worked -/
def total_hours : ℝ := days_worked * hours_per_day

theorem andrew_work_hours : total_hours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l692_69206


namespace NUMINAMATH_CALUDE_simplify_expression_l692_69215

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (4 + 7*x) = 1 - 9*x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l692_69215


namespace NUMINAMATH_CALUDE_hiking_trip_calculation_l692_69258

structure HikingSegment where
  distance : Float
  speed : Float

def total_distance (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance) |> List.sum

def total_time (segments : List HikingSegment) : Float :=
  segments.map (λ s => s.distance / s.speed) |> List.sum

def hiking_segments : List HikingSegment := [
  { distance := 0.5, speed := 3.0 },
  { distance := 1.2, speed := 2.5 },
  { distance := 0.8, speed := 2.0 },
  { distance := 0.6, speed := 2.8 }
]

theorem hiking_trip_calculation :
  total_distance hiking_segments = 3.1 ∧
  (total_time hiking_segments * 60).round = 76 := by
  sorry

#eval total_distance hiking_segments
#eval (total_time hiking_segments * 60).round

end NUMINAMATH_CALUDE_hiking_trip_calculation_l692_69258


namespace NUMINAMATH_CALUDE_square_boundary_length_l692_69261

/-- The total length of the boundary created by quarter-circle arcs and straight segments
    in a square with area 144, where each side is divided into thirds and quarters. -/
theorem square_boundary_length : ∃ (l : ℝ),
  l = 12 * Real.pi + 16 ∧ 
  (∃ (s : ℝ), s^2 = 144 ∧ 
    l = 4 * (2 * Real.pi * (s / 3) / 4 + Real.pi * (s / 6) / 4) + 4 * (s / 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_boundary_length_l692_69261


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l692_69284

/-- A quadratic function f(x) = (x + 1)² has a symmetry axis of x = -1 -/
theorem quadratic_symmetry_axis (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x + 1)^2
  ∀ y : ℝ, f (-1 - y) = f (-1 + y) := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l692_69284


namespace NUMINAMATH_CALUDE_partnership_profit_l692_69282

theorem partnership_profit (john_investment mike_investment : ℚ)
  (equal_share_ratio investment_ratio : ℚ)
  (john_extra_profit : ℚ) :
  john_investment = 700 →
  mike_investment = 300 →
  equal_share_ratio = 1/3 →
  investment_ratio = 2/3 →
  john_extra_profit = 800 →
  ∃ (total_profit : ℚ),
    total_profit * equal_share_ratio / 2 +
    total_profit * investment_ratio * (john_investment / (john_investment + mike_investment)) -
    (total_profit * equal_share_ratio / 2 +
     total_profit * investment_ratio * (mike_investment / (john_investment + mike_investment)))
    = john_extra_profit ∧
    total_profit = 3000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l692_69282


namespace NUMINAMATH_CALUDE_sin_30_degrees_l692_69229

/-- Sine of 30 degrees is 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l692_69229


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l692_69235

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  all_faces_painted : Bool

/-- Calculates the number of 1x1x1 subcubes with at least two painted faces in a painted cube -/
def subcubes_with_two_or_more_painted_faces (c : Cube 4) : ℕ :=
  if c.all_faces_painted then
    -- Corner cubes (3 faces painted)
    8 +
    -- Edge cubes without corners (2 faces painted)
    (12 * 2) +
    -- Middle-edge face cubes (2 faces painted)
    (6 * 4)
  else
    0

/-- Theorem: In a 4x4x4 cube with all faces painted, there are 56 subcubes with at least two painted faces -/
theorem painted_subcubes_count (c : Cube 4) (h : c.all_faces_painted = true) :
  subcubes_with_two_or_more_painted_faces c = 56 := by
  sorry

#check painted_subcubes_count

end NUMINAMATH_CALUDE_painted_subcubes_count_l692_69235


namespace NUMINAMATH_CALUDE_fraction_males_first_class_l692_69252

/-- Given a flight with passengers, prove the fraction of males in first class -/
theorem fraction_males_first_class
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_percentage : ℚ)
  (females_in_coach : ℕ)
  (h_total : total_passengers = 120)
  (h_female : female_percentage = 30 / 100)
  (h_first_class : first_class_percentage = 10 / 100)
  (h_females_coach : females_in_coach = 28) :
  (↑(total_passengers * first_class_percentage.num - (total_passengers * female_percentage.num - females_in_coach)) /
   ↑(total_passengers * first_class_percentage.num) : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_males_first_class_l692_69252


namespace NUMINAMATH_CALUDE_range_of_a_l692_69239

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, p x ↔ (3*x - 1)/(x - 2) ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a*(a + 1) < 0) →
  (∀ x, ¬(q x) → ¬(p x)) →
  (∃ x, ¬(q x) ∧ p x) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l692_69239


namespace NUMINAMATH_CALUDE_inverse_125_mod_79_l692_69236

theorem inverse_125_mod_79 (h : (5⁻¹ : ZMod 79) = 39) : (125⁻¹ : ZMod 79) = 69 := by
  sorry

end NUMINAMATH_CALUDE_inverse_125_mod_79_l692_69236


namespace NUMINAMATH_CALUDE_next_roll_for_average_three_l692_69219

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three (rolls : List Nat) : 
  rolls.length = 10 → 
  rolls.sum = 31 → 
  ∃ (next_roll : Nat), 
    (rolls.sum + next_roll) / (rolls.length + 1 : Nat) = 3 ∧ 
    next_roll = 2 := by
  sorry

#check next_roll_for_average_three rolls

end NUMINAMATH_CALUDE_next_roll_for_average_three_l692_69219


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l692_69238

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l692_69238


namespace NUMINAMATH_CALUDE_brick_weight_l692_69247

theorem brick_weight : ∃ x : ℝ, x = 2 + x / 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_brick_weight_l692_69247


namespace NUMINAMATH_CALUDE_time_after_adding_seconds_l692_69254

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (4:45:00 a.m.) -/
def initialTime : Time :=
  { hours := 4, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 12345

/-- The resulting time after adding seconds -/
def resultTime : Time :=
  { hours := 8, minutes := 30, seconds := 45 }

theorem time_after_adding_seconds :
  addSeconds initialTime secondsToAdd = resultTime := by
  sorry

end NUMINAMATH_CALUDE_time_after_adding_seconds_l692_69254


namespace NUMINAMATH_CALUDE_trapezium_area_l692_69214

theorem trapezium_area (a b h : ℝ) (ha : a = 28) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l692_69214


namespace NUMINAMATH_CALUDE_vector_angle_problem_l692_69208

theorem vector_angle_problem (α β : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (Real.cos α, Real.sin α))
  (h2 : b = (Real.cos β, Real.sin β))
  (h3 : ‖a - b‖ = (2 / 5) * Real.sqrt 5)
  (h4 : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h5 : Real.sin β = -5/13) :
  Real.cos (α - β) = 3/5 ∧ Real.sin α = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l692_69208


namespace NUMINAMATH_CALUDE_rectangle_width_l692_69218

theorem rectangle_width (length width : ℝ) : 
  length / width = 6 / 5 → length = 24 → width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l692_69218


namespace NUMINAMATH_CALUDE_statement_D_is_false_l692_69202

-- Define the set A_k
def A (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 4 * n + k}

-- State the theorem
theorem statement_D_is_false : ¬ (∀ a b : ℤ, (a + b) ∈ A 3 → a ∈ A 1 ∧ b ∈ A 2) := by
  sorry

end NUMINAMATH_CALUDE_statement_D_is_false_l692_69202


namespace NUMINAMATH_CALUDE_expression_evaluation_l692_69209

theorem expression_evaluation : (1 + (3 * 5)) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l692_69209


namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l692_69246

theorem stratified_sampling_proof (total_population : ℕ) (female_students : ℕ) 
  (sampled_female : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 1200)
  (h2 : female_students = 500)
  (h3 : sampled_female = 40)
  (h4 : (sample_size : ℚ) / total_population = (sampled_female : ℚ) / female_students) :
  sample_size = 96 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l692_69246


namespace NUMINAMATH_CALUDE_triangle_special_x_values_l692_69269

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle (a b c R : ℝ) : Prop where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius : R = (a * b * c) / (4 * area)
  area_positive : 0 < area

/-- The main theorem -/
theorem triangle_special_x_values
  (a b c : ℝ)
  (h_triangle : Triangle a b c 2)
  (h_angle : a^2 + c^2 ≤ b^2)
  (h_polynomial : ∃ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 = 0) :
  ∃ x : ℝ, x = -1/2 * (Real.sqrt 6 + Real.sqrt 2) ∨ x = -1/2 * (Real.sqrt 6 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_special_x_values_l692_69269


namespace NUMINAMATH_CALUDE_sqrt_21000_l692_69279

theorem sqrt_21000 (h : Real.sqrt 2.1 = 1.449) : Real.sqrt 21000 = 144.9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21000_l692_69279


namespace NUMINAMATH_CALUDE_problem_statement_l692_69226

theorem problem_statement (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) :
  b < a := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l692_69226


namespace NUMINAMATH_CALUDE_solve_record_problem_l692_69266

def record_problem (initial_records : ℕ) (bought_records : ℕ) (days_per_record : ℕ) (total_days : ℕ) : Prop :=
  let total_records := total_days / days_per_record
  let friends_records := total_records - (initial_records + bought_records)
  friends_records = 12

theorem solve_record_problem :
  record_problem 8 30 2 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_record_problem_l692_69266


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l692_69270

/-- Given a curve in polar coordinates defined by r = 2 / (2sin θ - cos θ),
    prove that it represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∀ θ r : ℝ,
  r = 2 / (2 * Real.sin θ - Real.cos θ) →
  ∃ m c : ℝ, ∀ x y : ℝ,
  x = r * Real.cos θ ∧ y = r * Real.sin θ →
  y = m * x + c :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l692_69270


namespace NUMINAMATH_CALUDE_paris_hair_theorem_l692_69228

theorem paris_hair_theorem (population : ℕ) (max_hair_count : ℕ) 
  (h1 : population > 2000000) 
  (h2 : max_hair_count = 150000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hair_count ∧ 
  (∃ (group : Finset (Fin population)), group.card ≥ 14 ∧ 
  ∀ i ∈ group, hair_count = (i : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_paris_hair_theorem_l692_69228


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l692_69262

theorem cube_volume_doubling (v : ℝ) (h : v = 27) :
  let new_volume := (2 * v^(1/3))^3
  new_volume = 216 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l692_69262


namespace NUMINAMATH_CALUDE_div_point_five_by_point_zero_twenty_five_l692_69257

theorem div_point_five_by_point_zero_twenty_five : (0.5 : ℚ) / 0.025 = 20 := by
  sorry

end NUMINAMATH_CALUDE_div_point_five_by_point_zero_twenty_five_l692_69257


namespace NUMINAMATH_CALUDE_supermarket_spending_l692_69244

theorem supermarket_spending (total : ℚ) (candy : ℚ) : 
  total = 24 →
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + candy = total →
  candy = 8 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l692_69244


namespace NUMINAMATH_CALUDE_perimeter_ratio_triangles_l692_69287

theorem perimeter_ratio_triangles :
  let small_triangle_sides : Fin 3 → ℝ := ![4, 8, 4 * Real.sqrt 3]
  let large_triangle_sides : Fin 3 → ℝ := ![8, 8, 8 * Real.sqrt 2]
  let small_perimeter := (Finset.univ.sum small_triangle_sides)
  let large_perimeter := (Finset.univ.sum large_triangle_sides)
  small_perimeter / large_perimeter = (4 + 8 + 4 * Real.sqrt 3) / (8 + 8 + 8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_triangles_l692_69287


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l692_69293

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) ≥ 0 ↔ (x ≥ -1 ∨ x < -2) ∧ x ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l692_69293


namespace NUMINAMATH_CALUDE_nested_sqrt_problem_l692_69234

theorem nested_sqrt_problem (n : ℕ) :
  (∃ k : ℕ, (n * (n * n^(1/2))^(1/2))^(1/2) = k) ∧ 
  (n * (n * n^(1/2))^(1/2))^(1/2) < 2217 →
  n = 256 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_problem_l692_69234


namespace NUMINAMATH_CALUDE_twelve_lines_theorem_l692_69256

/-- A line in a plane. -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle in a plane. -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The distance from a point to a line. -/
def distance_to_line (p : Point) (l : Line) : ℝ :=
  sorry

/-- Check if the distances from three points to a line are in one of the specified ratios. -/
def valid_ratio (A B C : Point) (l : Line) : Prop :=
  let dA := distance_to_line A l
  let dB := distance_to_line B l
  let dC := distance_to_line C l
  (dA = dB / 2 ∧ dA = dC / 2) ∨
  (dB = dA / 2 ∧ dB = dC / 2) ∨
  (dC = dA / 2 ∧ dC = dB / 2)

/-- The main theorem: there are exactly 12 lines satisfying the distance ratio condition for any triangle. -/
theorem twelve_lines_theorem (t : Triangle) : 
  ∃! (s : Finset Line), s.card = 12 ∧ ∀ l ∈ s, valid_ratio t.A t.B t.C l :=
sorry

end NUMINAMATH_CALUDE_twelve_lines_theorem_l692_69256


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l692_69204

theorem elevator_weight_problem (initial_count : ℕ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  new_person_weight = 97 →
  new_average = 151 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_person_weight = new_average * (initial_count + 1) ∧
    initial_average = 160 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l692_69204


namespace NUMINAMATH_CALUDE_complex_equation_solution_l692_69233

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (2 : ℂ) - 3 * i * z = (4 : ℂ) + 5 * i * z ∧ z = i / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l692_69233


namespace NUMINAMATH_CALUDE_two_sin_plus_cos_value_cos_and_tan_values_l692_69207

-- Define the angle α
variable (α : Real)

-- Define the point P
structure Point where
  x : Real
  y : Real

-- Theorem 1
theorem two_sin_plus_cos_value (P : Point) (h1 : P.x = 4) (h2 : P.y = -3) :
  2 * Real.sin α + Real.cos α = -2/5 := by sorry

-- Theorem 2
theorem cos_and_tan_values (P : Point) (m : Real) 
  (h1 : P.x = -Real.sqrt 3) (h2 : P.y = m) (h3 : m ≠ 0)
  (h4 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  Real.cos α = -Real.sqrt 3 / 5 ∧ 
  (Real.tan α = -Real.sqrt 10 / 3 ∨ Real.tan α = Real.sqrt 10 / 3) := by sorry

end NUMINAMATH_CALUDE_two_sin_plus_cos_value_cos_and_tan_values_l692_69207


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_t_equals_one_l692_69297

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then t = 1 -/
theorem perpendicular_vectors_imply_t_equals_one (a b c : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (-Real.sqrt 3, t) →
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 →
  t = 1 := by
  sorry

#check perpendicular_vectors_imply_t_equals_one

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_t_equals_one_l692_69297


namespace NUMINAMATH_CALUDE_smallest_value_l692_69289

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ 3*x ∧ x^3 ≤ x^(1/3) ∧ x^3 ≤ 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l692_69289


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l692_69272

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f (-4) x ≥ 6 ↔ x ≤ 0 ∨ x ≥ 6 :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → a ∈ Set.Icc (-1) 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l692_69272


namespace NUMINAMATH_CALUDE_min_distance_M_to_F₂_l692_69296

-- Define the rectangle and its properties
def Rectangle (a b : ℝ) := a > b ∧ a > 0 ∧ b > 0

-- Define the points on the sides of the rectangle
def Points (n : ℕ) (a b : ℝ) := n ≥ 5

-- Define the ellipse F₁
def F₁ (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola F₂
def F₂ (x y a b : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point M on F₁
def M (b : ℝ) := (0, b)

-- Theorem statement
theorem min_distance_M_to_F₂ (n : ℕ) (a b : ℝ) :
  Rectangle a b →
  Points n a b →
  ∀ (x y : ℝ), F₂ x y a b →
  Real.sqrt ((x - 0)^2 + (y - b)^2) ≥ a * Real.sqrt ((a^2 + 2*b^2) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_M_to_F₂_l692_69296


namespace NUMINAMATH_CALUDE_statement_a_is_false_l692_69260

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  isotopes : List (ℕ × ℝ)  -- List of (mass number, abundance) pairs

/-- Calculates the relative atomic mass of an element -/
def relative_atomic_mass (e : Element) : ℝ :=
  (e.isotopes.map (λ (mass, abundance) => mass * abundance)).sum

/-- Represents a single atom of an element -/
structure Atom where
  protons : ℕ
  neutrons : ℕ

/-- The statement we want to prove false -/
def statement_a (e : Element) (a : Atom) : Prop :=
  relative_atomic_mass e = a.protons + a.neutrons

/-- Theorem stating that the statement is false -/
theorem statement_a_is_false :
  ∃ (e : Element) (a : Atom), ¬(statement_a e a) :=
sorry

end NUMINAMATH_CALUDE_statement_a_is_false_l692_69260


namespace NUMINAMATH_CALUDE_constant_d_value_l692_69224

theorem constant_d_value (x y d : ℝ) 
  (h1 : x / (2 * y) = d / 2)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 25) :
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l692_69224


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_2b_for_tangent_line_l692_69227

/-- Given a line ax + by = 1 (where a > 0, b > 0) tangent to the circle x² + y² = 1,
    the maximum value of a + 2b is √5. -/
theorem max_value_of_a_plus_2b_for_tangent_line :
  ∀ a b : ℝ,
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 = 1) →
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∀ c : ℝ, c ≥ a + 2*b → c ≥ Real.sqrt 5) ∧
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    (∀ x y : ℝ, a' * x + b' * y = 1 → x^2 + y^2 = 1) ∧
    (∃ x y : ℝ, a' * x + b' * y = 1 ∧ x^2 + y^2 = 1) ∧
    a' + 2*b' = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_2b_for_tangent_line_l692_69227


namespace NUMINAMATH_CALUDE_contradiction_assumption_l692_69220

theorem contradiction_assumption (x y : ℝ) (h : x < y) :
  (¬ (x^3 < y^3)) ↔ (x^3 = y^3 ∨ x^3 > y^3) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l692_69220


namespace NUMINAMATH_CALUDE_coins_left_l692_69223

def pennies : ℕ := 42
def nickels : ℕ := 36
def dimes : ℕ := 15
def donated : ℕ := 66

theorem coins_left : pennies + nickels + dimes - donated = 27 := by
  sorry

end NUMINAMATH_CALUDE_coins_left_l692_69223


namespace NUMINAMATH_CALUDE_log7_2400_rounded_to_nearest_integer_l692_69281

-- Define the logarithm base 7 function
noncomputable def log7 (x : ℝ) : ℝ := Real.log x / Real.log 7

-- Theorem statement
theorem log7_2400_rounded_to_nearest_integer :
  ⌊log7 2400 + 0.5⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_log7_2400_rounded_to_nearest_integer_l692_69281


namespace NUMINAMATH_CALUDE_tangent_slope_condition_l692_69217

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^5 - a*(x + 1)

/-- The derivative of the curve function -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 5*x^4 - a

theorem tangent_slope_condition (a : ℝ) :
  (f_derivative a 1 > 1) ↔ (a < 4) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_condition_l692_69217
