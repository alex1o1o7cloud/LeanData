import Mathlib

namespace NUMINAMATH_CALUDE_tom_finishes_30_min_after_anna_l3624_362495

/-- Represents the race scenario with given parameters -/
structure RaceScenario where
  distance : ℝ
  anna_speed : ℝ
  tom_speed : ℝ

/-- Calculates the finish time difference between Tom and Anna -/
def finishTimeDifference (race : RaceScenario) : ℝ :=
  race.distance * (race.tom_speed - race.anna_speed)

/-- Theorem stating that in the given race scenario, Tom finishes 30 minutes after Anna -/
theorem tom_finishes_30_min_after_anna :
  let race : RaceScenario := {
    distance := 15,
    anna_speed := 7,
    tom_speed := 9
  }
  finishTimeDifference race = 30 := by sorry

end NUMINAMATH_CALUDE_tom_finishes_30_min_after_anna_l3624_362495


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_intercept_l3624_362462

/-- Given a line that intersects a circle centered at the origin, 
    prove that the line forms an isosceles right triangle with the origin 
    if and only if the absolute value of its y-intercept equals √2. -/
theorem isosceles_right_triangle_intercept (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.1 - A.2 + a = 0 ∧ 
    B.1 - B.2 + a = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2 ∧ 
    (A.1 - 0) * (B.1 - 0) + (A.2 - 0) * (B.2 - 0) = 0) ↔ 
  |a| = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_intercept_l3624_362462


namespace NUMINAMATH_CALUDE_vector_parallel_solution_l3624_362491

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_solution :
  ∃ (x : ℝ), parallel (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_solution_l3624_362491


namespace NUMINAMATH_CALUDE_two_digit_number_solution_l3624_362478

/-- A two-digit number with unit digit greater than tens digit by 2 and less than 30 -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 10 = (n / 10) + 2 ∧  -- unit digit greater than tens digit by 2
  n < 30  -- less than 30

theorem two_digit_number_solution :
  ∀ n : ℕ, TwoDigitNumber n → (n = 13 ∨ n = 24) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_solution_l3624_362478


namespace NUMINAMATH_CALUDE_parabola_vertex_l3624_362453

/-- A parabola is defined by the equation y = (x - 2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x y : ℝ) : Prop :=
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex 2 (-1) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3624_362453


namespace NUMINAMATH_CALUDE_squared_roots_polynomial_l3624_362482

theorem squared_roots_polynomial (x : ℝ) : 
  let f (x : ℝ) := x^3 + x^2 - 2*x - 1
  let g (x : ℝ) := x^3 - 5*x^2 + 6*x - 1
  ∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2 :=
by sorry

end NUMINAMATH_CALUDE_squared_roots_polynomial_l3624_362482


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3624_362433

/-- The number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem: Given the conditions, 130 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 420 325 175 50 = 130 := by
  sorry

#eval students_play_both 420 325 175 50

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l3624_362433


namespace NUMINAMATH_CALUDE_platform_length_l3624_362428

-- Define the train's properties
variable (l : ℝ) -- length of the train
variable (t : ℝ) -- time to pass a pole
variable (v : ℝ) -- velocity of the train

-- Define the platform
variable (p : ℝ) -- length of the platform

-- State the theorem
theorem platform_length 
  (h1 : v = l / t) -- velocity when passing the pole
  (h2 : v = (l + p) / (5 * t)) -- velocity when passing the platform
  : p = 4 * l := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3624_362428


namespace NUMINAMATH_CALUDE_unique_x_value_l3624_362418

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a + c, b - d)

/-- The theorem stating the unique value of x -/
theorem unique_x_value : ∃! x : ℤ, ∃ y : ℤ, 
  star (x, y) (3, 3) = star (5, 4) (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l3624_362418


namespace NUMINAMATH_CALUDE_birthday_ratio_l3624_362421

def peters_candles : ℕ := 10
def ruperts_candles : ℕ := 35

def age_ratio (x y : ℕ) : ℚ := (x : ℚ) / (y : ℚ)

theorem birthday_ratio : 
  age_ratio ruperts_candles peters_candles = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_birthday_ratio_l3624_362421


namespace NUMINAMATH_CALUDE_smallest_value_operation_l3624_362475

theorem smallest_value_operation (a b : ℤ) (h1 : a = -3) (h2 : b = -6) :
  a + b ≤ min (a - b) (min (a * b) (a / b)) := by sorry

end NUMINAMATH_CALUDE_smallest_value_operation_l3624_362475


namespace NUMINAMATH_CALUDE_negation_of_implication_l3624_362450

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3624_362450


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3624_362415

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3624_362415


namespace NUMINAMATH_CALUDE_button_probability_l3624_362480

/-- Represents a jar containing buttons of different colors -/
structure Jar :=
  (green : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Represents the state of both jars after the transfer -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

def initial_jar_A : Jar := ⟨6, 3, 9⟩

def transfer (x : ℕ) : JarState :=
  ⟨⟨initial_jar_A.green - x, initial_jar_A.red, initial_jar_A.blue - 2*x⟩,
   ⟨x, 0, 2*x⟩⟩

def half_buttons (js : JarState) : Prop :=
  js.jarA.green + js.jarA.red + js.jarA.blue = 
    (initial_jar_A.green + initial_jar_A.red + initial_jar_A.blue) / 2

def prob_blue_A (js : JarState) : ℚ :=
  js.jarA.blue / (js.jarA.green + js.jarA.red + js.jarA.blue)

def prob_green_B (js : JarState) : ℚ :=
  js.jarB.green / (js.jarB.green + js.jarB.blue)

theorem button_probability :
  ∃ x : ℕ, 
    let js := transfer x
    half_buttons js ∧ 
    prob_blue_A js * prob_green_B js = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_button_probability_l3624_362480


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3624_362420

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point M on the left half of x-axis
def point_M (a : ℝ) : Prop := a < 0

-- Define the tangent line from M to circle O
def tangent_line (a x y : ℝ) : Prop := 
  ∃ (t : ℝ), x = a * (1 - t^2) / (1 + t^2) ∧ y = 2 * a * t / (1 + t^2)

-- Define points A, B, and C
def point_A (a x y : ℝ) : Prop := circle_O x y ∧ tangent_line a x y
def point_B (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y
def point_C (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y ∧ ¬(point_B a x y)

-- Define the condition AB = BC
def equal_segments (a : ℝ) : Prop := 
  ∀ (xa ya xb yb xc yc : ℝ), 
    point_A a xa ya → point_B a xb yb → point_C a xc yc →
    (xa - xb)^2 + (ya - yb)^2 = (xb - xc)^2 + (yb - yc)^2

-- Theorem statement
theorem point_M_coordinates : 
  ∀ (a : ℝ), point_M a → equal_segments a → a = -4 :=
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l3624_362420


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3624_362446

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 4) (hb : |b| = 7) (hab : a < b) :
  a + b = 3 ∨ a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3624_362446


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3624_362460

theorem inequality_system_solution (x : ℝ) : 
  (2*x - 1)/3 - (5*x + 1)/2 ≤ 1 → 
  5*x - 1 < 3*(x + 1) → 
  -1 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3624_362460


namespace NUMINAMATH_CALUDE_ya_interval_neg_sqrt_seven_value_of_c_l3624_362406

-- Definition of Ya interval
def ya_interval (T : ℝ) : Set ℝ :=
  {x | ∃ m n : ℤ, m < T ∧ T < n ∧ x ∈ Set.Ioo (↑m : ℝ) (↑n : ℝ) ∧
    ∀ k : ℤ, (k : ℝ) ≤ T → k ≤ m}

-- Theorem 1: Ya interval of -√7
theorem ya_interval_neg_sqrt_seven :
  ya_interval (-Real.sqrt 7) = Set.Ioo (-3 : ℝ) (-2 : ℝ) := by sorry

-- Theorem 2: Value of c in the equation
theorem value_of_c (m n : ℕ) (h1 : ya_interval (Real.sqrt n - m) = Set.Ioo (↑m : ℝ) (↑n : ℝ))
  (h2 : 0 < m + Real.sqrt n) (h3 : m + Real.sqrt n < 12)
  (h4 : ∃ (x y : ℕ), x = m ∧ y^2 = n ∧ m*x - n*y = c) :
  c = 1 ∨ c = 37 := by sorry

end NUMINAMATH_CALUDE_ya_interval_neg_sqrt_seven_value_of_c_l3624_362406


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3624_362494

theorem triangle_inequalities (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : A > π/2) :
  (1 + Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) < Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2)) ∧
  (1 - Real.cos A + Real.sin B + Real.sin C < Real.sin A + Real.cos B + Real.cos C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3624_362494


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3624_362463

def vector_a : ℝ × ℝ := (-5, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_x_value :
  perpendicular vector_a (vector_b x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3624_362463


namespace NUMINAMATH_CALUDE_first_agency_mile_rate_calculation_l3624_362440

-- Define the constants
def first_agency_daily_rate : ℝ := 20.25
def second_agency_daily_rate : ℝ := 18.25
def second_agency_mile_rate : ℝ := 0.22
def crossover_miles : ℝ := 25.0

-- Define the theorem
theorem first_agency_mile_rate_calculation :
  ∃ (x : ℝ),
    first_agency_daily_rate + crossover_miles * x =
    second_agency_daily_rate + crossover_miles * second_agency_mile_rate ∧
    x = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_first_agency_mile_rate_calculation_l3624_362440


namespace NUMINAMATH_CALUDE_triangle_problem_l3624_362474

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) (h1 : t.a * Real.cos t.C - t.c * Real.sin t.A = 0)
    (h2 : t.b = 4) (h3 : (1/2) * t.a * t.b * Real.sin t.C = 6) :
    t.C = π/3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3624_362474


namespace NUMINAMATH_CALUDE_smallest_n_with_specific_decimal_periods_l3624_362414

/-- A function that checks if a fraction has a repeating decimal representation with a given period -/
def hasRepeatingDecimalPeriod (numerator : ℕ) (denominator : ℕ) (period : ℕ) : Prop :=
  ∃ k : ℕ, (10^period - 1) * numerator = k * denominator

/-- The smallest positive integer n less than 1000 such that 1/n has a repeating decimal
    representation with a period of 5 and 1/(n+7) has a repeating decimal representation
    with a period of 4 is 266 -/
theorem smallest_n_with_specific_decimal_periods : 
  ∃ n : ℕ, n > 0 ∧ n < 1000 ∧
    hasRepeatingDecimalPeriod 1 n 5 ∧
    hasRepeatingDecimalPeriod 1 (n + 7) 4 ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(hasRepeatingDecimalPeriod 1 m 5 ∧ hasRepeatingDecimalPeriod 1 (m + 7) 4)) ∧
    n = 266 :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_n_with_specific_decimal_periods_l3624_362414


namespace NUMINAMATH_CALUDE_square_difference_l3624_362419

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3624_362419


namespace NUMINAMATH_CALUDE_medical_team_selection_l3624_362431

theorem medical_team_selection (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 6) (h2 : female_doctors = 5) :
  Nat.choose male_doctors 2 * Nat.choose female_doctors 1 = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l3624_362431


namespace NUMINAMATH_CALUDE_stacys_current_height_l3624_362485

/-- Prove Stacy's current height given the conditions of the problem -/
theorem stacys_current_height 
  (S J M S' J' M' : ℕ) 
  (h1 : S = 50)
  (h2 : S' = J' + 6)
  (h3 : J' = J + 1)
  (h4 : M' = M + 2 * (J' - J))
  (h5 : S + J + M = 128)
  (h6 : S' + J' + M' = 140) :
  S' = 59 := by
  sorry

end NUMINAMATH_CALUDE_stacys_current_height_l3624_362485


namespace NUMINAMATH_CALUDE_chris_pennies_l3624_362437

theorem chris_pennies (a c : ℕ) : 
  (c + 2 = 4 * (a - 2)) → 
  (c - 2 = 3 * (a + 2)) → 
  c = 62 := by
sorry

end NUMINAMATH_CALUDE_chris_pennies_l3624_362437


namespace NUMINAMATH_CALUDE_triangle_sum_equals_58_l3624_362468

/-- The triangle operation that takes three numbers and returns the sum of their squares -/
def triangle (a b c : ℝ) : ℝ := a^2 + b^2 + c^2

/-- Theorem stating that the sum of triangle(2,3,6) and triangle(1,2,2) equals 58 -/
theorem triangle_sum_equals_58 : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_58_l3624_362468


namespace NUMINAMATH_CALUDE_cu_cn2_formation_l3624_362496

-- Define the chemical species
inductive Species
| HCN
| CuSO4
| CuCN2
| H2SO4

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (Species × ℕ)
  products : List (Species × ℕ)

-- Define the balanced equation
def balancedEquation : Reaction :=
  { reactants := [(Species.HCN, 2), (Species.CuSO4, 1)]
  , products := [(Species.CuCN2, 1), (Species.H2SO4, 1)] }

-- Define the initial amounts of reactants
def initialHCN : ℕ := 2
def initialCuSO4 : ℕ := 1

-- Theorem statement
theorem cu_cn2_formation
  (reaction : Reaction)
  (hreaction : reaction = balancedEquation)
  (hHCN : initialHCN = 2)
  (hCuSO4 : initialCuSO4 = 1) :
  ∃ (amount : ℕ), amount = 1 ∧ 
  (Species.CuCN2, amount) ∈ reaction.products :=
sorry

end NUMINAMATH_CALUDE_cu_cn2_formation_l3624_362496


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3624_362426

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3624_362426


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l3624_362430

theorem unique_prime_with_prime_successors :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l3624_362430


namespace NUMINAMATH_CALUDE_inequality_proof_l3624_362409

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (y + z)) * (b + c) + (y / (z + x)) * (c + a) + (z / (x + y)) * (a + b) ≥ 
  Real.sqrt (3 * (a * b + b * c + c * a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3624_362409


namespace NUMINAMATH_CALUDE_junior_rabbit_toys_l3624_362497

def toys_per_rabbit (num_rabbits : ℕ) (monday_toys : ℕ) : ℕ :=
  let wednesday_toys := 2 * monday_toys
  let friday_toys := 4 * monday_toys
  let saturday_toys := wednesday_toys / 2
  let total_toys := monday_toys + wednesday_toys + friday_toys + saturday_toys
  total_toys / num_rabbits

theorem junior_rabbit_toys :
  toys_per_rabbit 16 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_junior_rabbit_toys_l3624_362497


namespace NUMINAMATH_CALUDE_linear_function_properties_l3624_362469

-- Define the linear function
def f (x : ℝ) : ℝ := x + 2

-- Theorem stating the properties of the function
theorem linear_function_properties :
  (f 1 = 3) ∧ 
  (f (-2) = 0) ∧ 
  (∃ x > 2, f x ≥ 4) ∧
  (∀ x y, f x = y → (x > 0 → y > 0)) :=
by sorry

#check linear_function_properties

end NUMINAMATH_CALUDE_linear_function_properties_l3624_362469


namespace NUMINAMATH_CALUDE_consecutive_points_length_l3624_362455

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (c - a = 11) →           -- ac = 11
  (e - a = 21) →           -- ae = 21
  (b - a = 5) :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l3624_362455


namespace NUMINAMATH_CALUDE_expression_evaluation_l3624_362479

theorem expression_evaluation : 
  let b : ℚ := 4/3
  (6 * b^2 - 8 * b + 3) * (3 * b - 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3624_362479


namespace NUMINAMATH_CALUDE_mrs_hilt_nickels_l3624_362487

/-- Represents the number of coins of each type a person has -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents given a CoinCount -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Mrs. Hilt's coin count, with unknown number of nickels -/
def mrsHilt (n : ℕ) : CoinCount :=
  { pennies := 2, nickels := n, dimes := 2 }

/-- Jacob's coin count -/
def jacob : CoinCount :=
  { pennies := 4, nickels := 1, dimes := 1 }

/-- Theorem stating that Mrs. Hilt must have 2 nickels -/
theorem mrs_hilt_nickels :
  ∃ n : ℕ, totalValue (mrsHilt n) - totalValue jacob = 13 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_nickels_l3624_362487


namespace NUMINAMATH_CALUDE_problem_solution_l3624_362443

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + 2

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2*x + k/x

theorem problem_solution :
  -- Part 1: Average rate of change
  (f 2 - f 0) / 2 = 2 ∧
  -- Part 2: Parallel tangent lines
  (∃ k : ℝ, (deriv f 1 = deriv (g k) 1) → k = 1) ∧
  -- Part 3: Tangent line equation
  (∃ a b : ℝ, (∀ x : ℝ, a*x + b = 10*x - 14) ∧
              f 2 = a*2 + b ∧
              deriv f 2 = a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3624_362443


namespace NUMINAMATH_CALUDE_rocking_chair_legs_count_l3624_362402

/-- Represents the number of legs on the rocking chair -/
def rocking_chair_legs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of four-legged tables -/
def four_leg_tables : ℕ := 4

/-- Represents the number of sofas -/
def sofas : ℕ := 1

/-- Represents the number of four-legged chairs -/
def four_leg_chairs : ℕ := 2

/-- Represents the number of three-legged tables -/
def three_leg_tables : ℕ := 3

/-- Represents the number of one-legged tables -/
def one_leg_tables : ℕ := 1

theorem rocking_chair_legs_count : 
  rocking_chair_legs = 
    total_legs - 
    (4 * four_leg_tables + 
     4 * sofas + 
     4 * four_leg_chairs + 
     3 * three_leg_tables + 
     1 * one_leg_tables) :=
by sorry

end NUMINAMATH_CALUDE_rocking_chair_legs_count_l3624_362402


namespace NUMINAMATH_CALUDE_kyle_track_laps_l3624_362465

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := 1.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := total_laps - pe_laps

theorem kyle_track_laps : track_laps = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_kyle_track_laps_l3624_362465


namespace NUMINAMATH_CALUDE_least_boxes_for_candy_packing_l3624_362422

/-- Given that N is a non-zero perfect cube and 45 is a factor of N,
    prove that the least number of boxes needed to pack N pieces of candy,
    with 45 pieces per box, is 75. -/
theorem least_boxes_for_candy_packing (N : ℕ) : 
  N ≠ 0 ∧ 
  (∃ k : ℕ, N = k^3) ∧ 
  (∃ m : ℕ, N = 45 * m) ∧
  (∀ M : ℕ, M ≠ 0 ∧ (∃ j : ℕ, M = j^3) ∧ (∃ n : ℕ, M = 45 * n) → N ≤ M) →
  N / 45 = 75 := by
sorry

end NUMINAMATH_CALUDE_least_boxes_for_candy_packing_l3624_362422


namespace NUMINAMATH_CALUDE_seating_probability_l3624_362456

-- Define the number of boys in the class
def num_boys : ℕ := 9

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the derangement function for 4 elements
def derangement_4 : ℕ := 9

-- Define the probability we want to prove
def target_probability : ℚ := 1 / 32

-- Theorem statement
theorem seating_probability :
  (choose num_boys 3 * choose (num_boys - 3) 2 * derangement_4) / (Nat.factorial num_boys) = target_probability := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l3624_362456


namespace NUMINAMATH_CALUDE_rectangular_envelope_foldable_l3624_362483

-- Define a rectangular envelope
structure RectangularEnvelope where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define a tetrahedron
structure Tetrahedron where
  surface_area : ℝ
  surface_area_positive : surface_area > 0

-- Define the property of being able to fold into two congruent tetrahedrons
def can_fold_into_congruent_tetrahedrons (env : RectangularEnvelope) : Prop :=
  ∃ (t : Tetrahedron), 
    t.surface_area = (env.length * env.width) / 2 ∧ 
    env.length ≠ env.width

-- State the theorem
theorem rectangular_envelope_foldable (env : RectangularEnvelope) :
  can_fold_into_congruent_tetrahedrons env :=
sorry

end NUMINAMATH_CALUDE_rectangular_envelope_foldable_l3624_362483


namespace NUMINAMATH_CALUDE_coordinates_wrt_symmetric_point_l3624_362488

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

-- Theorem statement
theorem coordinates_wrt_symmetric_point (A B : Point2D) :
  A.x = -5 ∧ A.y = 2 ∧ symmetricAboutYAxis A B →
  (A.x - B.x = 5 ∧ A.y - B.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_symmetric_point_l3624_362488


namespace NUMINAMATH_CALUDE_committee_count_is_738_l3624_362432

/-- Represents a department in the university's science division -/
inductive Department
| Physics
| Chemistry
| Biology

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents a professor with their department and gender -/
structure Professor :=
  (dept : Department)
  (gender : Gender)

/-- The total number of professors in each department for each gender -/
def professors_per_dept_gender : Nat := 3

/-- The total number of professors in the committee -/
def committee_size : Nat := 7

/-- The number of male professors required in the committee -/
def required_males : Nat := 4

/-- The number of female professors required in the committee -/
def required_females : Nat := 3

/-- The number of professors required from the physics department -/
def required_physics : Nat := 3

/-- The number of professors required from each of chemistry and biology departments -/
def required_chem_bio : Nat := 2

/-- Calculates the number of possible committees given the conditions -/
def count_committees (professors : List Professor) : Nat :=
  sorry

/-- Theorem stating that the number of possible committees is 738 -/
theorem committee_count_is_738 (professors : List Professor) : 
  count_committees professors = 738 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_is_738_l3624_362432


namespace NUMINAMATH_CALUDE_eraser_cost_mary_eraser_cost_l3624_362457

/-- The cost of each eraser given Mary's school supplies purchase --/
theorem eraser_cost (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℚ) (pencil_cost : ℚ) (total_spent : ℚ) 
  (paint_cost : ℚ) : ℚ :=
  let folders := classes * folders_per_class
  let pencils := classes * pencils_per_class
  let erasers := pencils / pencils_per_eraser
  let folder_total := folders * folder_cost
  let pencil_total := pencils * pencil_cost
  let eraser_total := total_spent - folder_total - pencil_total - paint_cost
  eraser_total / erasers

/-- The cost of each eraser in Mary's specific purchase is $1 --/
theorem mary_eraser_cost : 
  eraser_cost 6 1 3 6 6 2 80 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_eraser_cost_mary_eraser_cost_l3624_362457


namespace NUMINAMATH_CALUDE_barbaras_selling_price_l3624_362401

/-- Proves that Barbara's selling price for each stuffed animal is $2 --/
theorem barbaras_selling_price : 
  ∀ (barbara_price : ℚ),
  (9 : ℚ) * barbara_price + (2 * 9 : ℚ) * (3/2 : ℚ) = 45 →
  barbara_price = 2 := by
sorry

end NUMINAMATH_CALUDE_barbaras_selling_price_l3624_362401


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l3624_362408

theorem systematic_sampling_removal (total_students sample_size : ℕ) 
  (h1 : total_students = 1252)
  (h2 : sample_size = 50) :
  ∃ (removed : ℕ), 
    removed = 2 ∧ 
    (total_students - removed) % sample_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l3624_362408


namespace NUMINAMATH_CALUDE_product_satisfies_X_l3624_362439

/-- Condition X: every positive integer less than m is a sum of distinct divisors of m -/
def condition_X (m : ℕ+) : Prop :=
  ∀ k < m, ∃ (S : Finset ℕ), (∀ d ∈ S, d ∣ m) ∧ (Finset.sum S id = k)

theorem product_satisfies_X (m n : ℕ+) (hm : condition_X m) (hn : condition_X n) :
  condition_X (m * n) :=
sorry

end NUMINAMATH_CALUDE_product_satisfies_X_l3624_362439


namespace NUMINAMATH_CALUDE_ryan_learning_days_l3624_362416

def daily_english_hours : ℕ := 6
def daily_chinese_hours : ℕ := 7
def total_hours : ℕ := 65

theorem ryan_learning_days : 
  total_hours / (daily_english_hours + daily_chinese_hours) = 5 := by
sorry

end NUMINAMATH_CALUDE_ryan_learning_days_l3624_362416


namespace NUMINAMATH_CALUDE_tangent_and_fixed_line_l3624_362407

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Define a line through M with non-zero slope
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersect_C₁_line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C₁ p.1 p.2 ∧ line_through_M k p.1 p.2}

-- Define perpendicular bisector
def perp_bisector (p₁ p₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - (p₁.1 + p₂.1) / 2) * (p₂.1 - p₁.1) + (y - (p₁.2 + p₂.2) / 2) * (p₂.2 - p₁.2) = 0

theorem tangent_and_fixed_line 
  (k : ℝ) 
  (hk : k ≠ 0) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ intersect_C₁_line k) 
  (hB : B ∈ intersect_C₁_line k) 
  (hAB : A ≠ B) :
  (∃ (P : ℝ × ℝ), 
    (∀ (x y : ℝ), perp_bisector A N x y → C₂ x y) ∧ 
    (∀ (x y : ℝ), perp_bisector B N x y → C₂ x y) ∧
    perp_bisector A N P.1 P.2 ∧ 
    perp_bisector B N P.1 P.2 ∧
    P.1 = -4) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_fixed_line_l3624_362407


namespace NUMINAMATH_CALUDE_equation_one_solution_l3624_362412

theorem equation_one_solution (x : ℝ) : x^2 = -4*x → x = 0 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l3624_362412


namespace NUMINAMATH_CALUDE_vector_equation_l3624_362438

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (O A B C : V) :
  (A - O) - (B - O) + (C - A) = C - B := by sorry

end NUMINAMATH_CALUDE_vector_equation_l3624_362438


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_divisible_by_8_l3624_362466

theorem consecutive_odd_squares_difference_divisible_by_8 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_divisible_by_8_l3624_362466


namespace NUMINAMATH_CALUDE_profit_increase_l3624_362490

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > cost_price)
  (h3 : (selling_price - cost_price) / cost_price = a / 100)
  (h4 : (selling_price - cost_price * 0.95) / (cost_price * 0.95) = (a + 15) / 100) :
  a = 185 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l3624_362490


namespace NUMINAMATH_CALUDE_sum_of_numbers_less_than_three_tenths_l3624_362442

def numbers : List ℚ := [8/10, 1/2, 9/10, 2/10, 1/3]

theorem sum_of_numbers_less_than_three_tenths :
  (numbers.filter (λ x => x < 3/10)).sum = 2/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_less_than_three_tenths_l3624_362442


namespace NUMINAMATH_CALUDE_fish_sales_hours_l3624_362445

/-- The number of hours fish are sold for, given peak and low season sales rates,
    price per pack, and daily revenue difference between seasons. -/
theorem fish_sales_hours 
  (peak_rate : ℕ) 
  (low_rate : ℕ) 
  (price_per_pack : ℕ) 
  (daily_revenue_diff : ℕ) 
  (h_peak_rate : peak_rate = 6)
  (h_low_rate : low_rate = 4)
  (h_price : price_per_pack = 60)
  (h_revenue_diff : daily_revenue_diff = 1800) :
  (peak_rate - low_rate) * price_per_pack * h = daily_revenue_diff → h = 15 :=
by sorry

end NUMINAMATH_CALUDE_fish_sales_hours_l3624_362445


namespace NUMINAMATH_CALUDE_jamie_flyer_earnings_l3624_362444

/-- Calculates Jamie's earnings from delivering flyers --/
def jamies_earnings (hourly_rate : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) (total_weeks : ℕ) : ℕ :=
  hourly_rate * days_per_week * hours_per_day * total_weeks

/-- Proves that Jamie's earnings after 6 weeks will be $360 --/
theorem jamie_flyer_earnings :
  jamies_earnings 10 2 3 6 = 360 := by
  sorry

#eval jamies_earnings 10 2 3 6

end NUMINAMATH_CALUDE_jamie_flyer_earnings_l3624_362444


namespace NUMINAMATH_CALUDE_class_height_ratio_l3624_362473

theorem class_height_ratio :
  ∀ (x y : ℕ),
  x > 0 → y > 0 →
  149 * x + 144 * y = 147 * (x + y) →
  (x : ℚ) / y = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_class_height_ratio_l3624_362473


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3624_362476

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3624_362476


namespace NUMINAMATH_CALUDE_max_m_value_l3624_362467

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l3624_362467


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3624_362452

theorem ceiling_product_equation (x : ℝ) : 
  ⌈x⌉ * x = 198 ↔ x = 13.2 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3624_362452


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3624_362459

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  a = 5 + 2 * Real.sqrt 6 →           -- given value of a
  c = 5 - 2 * Real.sqrt 6 →           -- given value of c
  b = 1 ∨ b = -1 :=                   -- conclusion
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3624_362459


namespace NUMINAMATH_CALUDE_sock_pairs_count_l3624_362477

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  white * brown + brown * blue + white * blue

/-- Theorem: Given 5 white socks, 4 brown socks, and 3 blue socks,
    there are 47 ways to choose a pair of socks of different colors -/
theorem sock_pairs_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l3624_362477


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3624_362458

def geometric_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

theorem geometric_sequence_proof (b : ℝ) 
  (h₁ : geometric_sequence 150 b (60/36)) 
  (h₂ : b > 0) : 
  b = 5 * Real.sqrt 10 ∧ ¬ arithmetic_sequence 150 b (60/36) := by
  sorry

#check geometric_sequence_proof

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3624_362458


namespace NUMINAMATH_CALUDE_closest_point_and_area_l3624_362403

def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

def line_l (x y : ℝ) : Prop := y = -x - 2

def point_P : ℝ × ℝ := (-2, 1)

def focus_C : ℝ × ℝ := (0, 1)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem closest_point_and_area :
  ∀ (A B : ℝ × ℝ),
    parabola_C point_P.1 point_P.2 →
    (∀ (Q : ℝ × ℝ), parabola_C Q.1 Q.2 →
      ∃ (d_P d_Q : ℝ),
        d_P = abs (point_P.2 + point_P.1 + 2) / Real.sqrt 2 ∧
        d_Q = abs (Q.2 + Q.1 + 2) / Real.sqrt 2 ∧
        d_P ≤ d_Q) →
    parabola_C A.1 A.2 →
    parabola_C B.1 B.2 →
    is_centroid focus_C point_P A B →
    ∃ (area : ℝ), area = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_closest_point_and_area_l3624_362403


namespace NUMINAMATH_CALUDE_expression_percentage_l3624_362498

theorem expression_percentage (x : ℝ) (h : x > 0) : 
  (x / 50 + x / 25 - x / 10 + x / 5) / x = 16 / 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_l3624_362498


namespace NUMINAMATH_CALUDE_car_speed_proof_l3624_362425

/-- Proves that given the conditions of the car journey, the initial speed must be 75 km/hr -/
theorem car_speed_proof (v : ℝ) : 
  v > 0 →
  (320 / (160 / v + 160 / 80) = 77.4193548387097) →
  v = 75 := by
sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3624_362425


namespace NUMINAMATH_CALUDE_part_one_part_two_l3624_362471

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + 3 * x

theorem part_one :
  let f₁ := f 1
  (∀ x, f₁ x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

theorem part_two (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3624_362471


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3624_362449

theorem trigonometric_equation_solution :
  ∀ x : Real,
    0 < x →
    x < 180 →
    Real.tan ((150 : Real) * Real.pi / 180 - x * Real.pi / 180) = 
      (Real.sin ((150 : Real) * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
      (Real.cos ((150 : Real) * Real.pi / 180) - Real.cos (x * Real.pi / 180)) →
    x = 120 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3624_362449


namespace NUMINAMATH_CALUDE_horner_rule_evaluation_l3624_362470

/-- Horner's Rule evaluation for a polynomial --/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ --/
def f : List ℤ := [12, 35, -8, 79, 6, 5, 3]

/-- Theorem: The value of f(-4) using Horner's Rule is 220 --/
theorem horner_rule_evaluation :
  horner_eval f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_evaluation_l3624_362470


namespace NUMINAMATH_CALUDE_smallest_height_of_special_triangle_l3624_362486

/-- Given a scalene triangle with integer side lengths a, b, c satisfying 
    the relation (a^2/c) - (a-c)^2 = (b^2/c) - (b-c)^2, 
    the smallest height of the triangle is 12/5. -/
theorem smallest_height_of_special_triangle (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hrelation : (a^2 : ℚ)/c - (a-c)^2 = (b^2 : ℚ)/c - (b-c)^2) :
  ∃ h : ℚ, h = 12/5 ∧ h = min (2 * (a * b) / (2 * a)) (min (2 * (b * c) / (2 * b)) (2 * (a * c) / (2 * c))) :=
sorry

end NUMINAMATH_CALUDE_smallest_height_of_special_triangle_l3624_362486


namespace NUMINAMATH_CALUDE_equal_triangle_areas_l3624_362411

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)

-- Define a point inside a polygon
def PointInside (P : ℝ × ℝ) (polygon : List (ℝ × ℝ)) : Prop := sorry

-- Define parallel lines
def Parallel (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : Prop :=
  (P₁.2 - Q₁.2) / (P₁.1 - Q₁.1) = (P₂.2 - Q₂.2) / (P₂.1 - Q₂.1)

-- Define the area of a triangle
def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (A B C D M N : ℝ × ℝ) 
  (trap : Trapezoid A B C D)
  (m_inside : PointInside M [A, B, C, D])
  (n_inside : PointInside N [B, M, C])
  (am_cn_parallel : Parallel A M C N)
  (bm_dn_parallel : Parallel B M D N) :
  TriangleArea A B N = TriangleArea C D M := by
  sorry

end NUMINAMATH_CALUDE_equal_triangle_areas_l3624_362411


namespace NUMINAMATH_CALUDE_polynomial_intersection_l3624_362492

-- Define the polynomials f and h
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def h (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem polynomial_intersection (a b p q : ℝ) : 
  -- f and h are distinct polynomials
  (∃ x, f a b x ≠ h p q x) →
  -- The x-coordinate of the vertex of f is a root of h
  h p q (-a/2) = 0 →
  -- The x-coordinate of the vertex of h is a root of f
  f a b (-p/2) = 0 →
  -- Both f and h have the same minimum value
  (∃ y, f a b (-a/2) = y ∧ h p q (-p/2) = y) →
  -- The graphs of f and h intersect at the point (50, -50)
  f a b 50 = -50 ∧ h p q 50 = -50 →
  -- Conclusion: a + p = 0
  a + p = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l3624_362492


namespace NUMINAMATH_CALUDE_greatest_possible_award_l3624_362427

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 800)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award) :
  ∃ (max_award : ℕ), max_award = 420 ∧ 
    (∀ (award : ℕ), award > max_award → 
      ¬(∃ (awards : List ℕ), awards.length = num_winners ∧ 
        awards.sum = total_prize ∧ 
        (∀ x ∈ awards, x ≥ min_award) ∧
        award ∈ awards)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l3624_362427


namespace NUMINAMATH_CALUDE_basketball_scores_second_half_total_l3624_362424

/-- Represents the score of a team in a quarter -/
structure QuarterScore :=
  (score : ℕ)

/-- Represents the scores of a team for all four quarters -/
structure GameScore :=
  (q1 : QuarterScore)
  (q2 : QuarterScore)
  (q3 : QuarterScore)
  (q4 : QuarterScore)

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℚ), b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ (diff : ℤ), b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem statement -/
theorem basketball_scores_second_half_total
  (eagles : GameScore)
  (lions : GameScore)
  (h1 : eagles.q1.score = lions.q1.score)
  (h2 : eagles.q1.score + eagles.q2.score = lions.q1.score + lions.q2.score)
  (h3 : isGeometricSequence eagles.q1.score eagles.q2.score eagles.q3.score eagles.q4.score)
  (h4 : isArithmeticSequence lions.q1.score lions.q2.score lions.q3.score lions.q4.score)
  (h5 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score = 
        lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score + 1)
  (h6 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score ≤ 100)
  (h7 : lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score ≤ 100) :
  eagles.q3.score + eagles.q4.score + lions.q3.score + lions.q4.score = 109 :=
sorry

end NUMINAMATH_CALUDE_basketball_scores_second_half_total_l3624_362424


namespace NUMINAMATH_CALUDE_sixth_grade_students_l3624_362435

/-- The number of students in the sixth grade -/
def total_students : ℕ := 147

/-- The number of books available -/
def total_books : ℕ := 105

/-- The number of boys in the sixth grade -/
def num_boys : ℕ := 84

/-- The number of girls in the sixth grade -/
def num_girls : ℕ := 63

theorem sixth_grade_students :
  (total_students = num_boys + num_girls) ∧
  (total_books = 105) ∧
  (num_boys + (num_girls / 3) = total_books) ∧
  (num_girls + (num_boys / 2) = total_books) :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_students_l3624_362435


namespace NUMINAMATH_CALUDE_target_primes_are_5_13_17_29_l3624_362417

/-- The set of prime numbers less than 30 -/
def primes_less_than_30 : Set ℕ :=
  {p | p < 30 ∧ Nat.Prime p}

/-- A function that checks if a number becomes a multiple of 4 after adding 3 -/
def becomes_multiple_of_4 (n : ℕ) : Prop :=
  (n + 3) % 4 = 0

/-- The set of prime numbers less than 30 that become multiples of 4 after adding 3 -/
def target_primes : Set ℕ :=
  {p ∈ primes_less_than_30 | becomes_multiple_of_4 p}

theorem target_primes_are_5_13_17_29 : target_primes = {5, 13, 17, 29} := by
  sorry

end NUMINAMATH_CALUDE_target_primes_are_5_13_17_29_l3624_362417


namespace NUMINAMATH_CALUDE_isosceles_triangles_perimeter_l3624_362493

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def PQ : ℝ := sorry
def PR : ℝ := sorry
def QR : ℝ := sorry
def PS : ℝ := sorry
def SR : ℝ := sorry

-- Define x
def x : ℝ := sorry

-- State the theorem
theorem isosceles_triangles_perimeter (P Q R S : ℝ × ℝ) (PQ PR QR PS SR x : ℝ) :
  PQ = PR →                           -- Triangle PQR is isosceles
  PS = SR →                           -- Triangle PRS is isosceles
  PS = x →                            -- PS = x
  SR = x →                            -- SR = x
  PQ + QR + PR = 22 →                 -- Perimeter of Triangle PQR is 22
  PR + PS + SR = 22 →                 -- Perimeter of Triangle PRS is 22
  PQ + QR + SR + PS = 24 →            -- Perimeter of quadrilateral PQRS is 24
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_perimeter_l3624_362493


namespace NUMINAMATH_CALUDE_problem_statement_l3624_362434

theorem problem_statement (n : ℕ+) 
  (h1 : ∃ a : ℕ+, (3 * n + 1 : ℕ) = a ^ 2)
  (h2 : ∃ b : ℕ+, (5 * n - 1 : ℕ) = b ^ 2) :
  (∃ p q : ℕ+, p * q = 7 * n + 13 ∧ p ≠ 1 ∧ q ≠ 1) ∧
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3624_362434


namespace NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3624_362454

-- Define atomic weights
def atomic_weight_H : ℝ := 1
def atomic_weight_Br : ℝ := 79.9
def atomic_weight_O : ℝ := 16

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 129

-- Define the number of atoms for H and Br
def num_H : ℕ := 1
def num_Br : ℕ := 1

-- Theorem to prove
theorem oxygen_atoms_in_compound :
  ∃ (n : ℕ), n * atomic_weight_O = molecular_weight - (num_H * atomic_weight_H + num_Br * atomic_weight_Br) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3624_362454


namespace NUMINAMATH_CALUDE_correct_calculation_l3624_362472

theorem correct_calculation (a : ℝ) : 4 * a - (-7 * a) = 11 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3624_362472


namespace NUMINAMATH_CALUDE_period_of_trigonometric_function_l3624_362413

/-- The period of the function y = 3sin(x) + 4cos(x - π/6) is 2π. -/
theorem period_of_trigonometric_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos (x - π/6)
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, (0 < S ∧ S < T) → ∃ x : ℝ, f (x + S) ≠ f x ∧ T = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_period_of_trigonometric_function_l3624_362413


namespace NUMINAMATH_CALUDE_mixture_volume_l3624_362451

theorem mixture_volume (initial_water_percent : Real) 
                       (added_water : Real) 
                       (final_water_percent : Real) :
  initial_water_percent = 0.20 →
  added_water = 13.333333333333334 →
  final_water_percent = 0.25 →
  (∃ (initial_volume : Real),
    initial_volume * initial_water_percent + added_water = 
    (initial_volume + added_water) * final_water_percent ∧
    initial_volume = 200) :=
by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l3624_362451


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3624_362410

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd (2^a.val - 1) (2^b.val - 1) = 2^(Nat.gcd a.val b.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3624_362410


namespace NUMINAMATH_CALUDE_time_after_3577_minutes_l3624_362448

/-- Represents a time of day -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat
  deriving Repr

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : TimeOfDay
  deriving Repr

def startDateTime : DateTime := {
  date := { year := 2020, month := 12, day := 31 }
  time := { hours := 18, minutes := 0 }
}

def minutesElapsed : Nat := 3577

def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

theorem time_after_3577_minutes :
  addMinutes startDateTime minutesElapsed = {
    date := { year := 2021, month := 1, day := 3 }
    time := { hours := 5, minutes := 37 }
  } := by sorry

end NUMINAMATH_CALUDE_time_after_3577_minutes_l3624_362448


namespace NUMINAMATH_CALUDE_remainder_problem_l3624_362429

theorem remainder_problem (N : ℕ) : 
  (∃ R, N = 5 * 2 + R ∧ R < 5) → 
  (∃ Q, N = 4 * Q + 2) → 
  (∃ R, N = 5 * 2 + R ∧ R = 4) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3624_362429


namespace NUMINAMATH_CALUDE_volume_is_zero_l3624_362400

def S : Set (ℝ × ℝ) := {(x, y) | |6 - x| + y ≤ 8 ∧ 2*y - x ≥ 10}

def revolution_axis : Set (ℝ × ℝ) := {(x, y) | 2*y - x = 10}

def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_is_zero :
  volume_of_revolution S revolution_axis = 0 := by sorry

end NUMINAMATH_CALUDE_volume_is_zero_l3624_362400


namespace NUMINAMATH_CALUDE_f_properties_l3624_362484

def f (x : ℝ) := x^3 + 2*x^2 - 4*x + 5

theorem f_properties :
  (f (-2) = 13) ∧
  (HasDerivAt f 0 (-2)) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≤ 13) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 13) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3624_362484


namespace NUMINAMATH_CALUDE_relationship_functions_l3624_362499

-- Define the relationships
def relationA (x : ℝ) : ℝ := 180 - x
def relationB (x : ℝ) : ℝ := 60 + 3 * x
def relationC (x : ℝ) : ℝ := x ^ 2
def relationD (x : ℝ) : Set ℝ := {y | y ^ 2 = x ∧ x ≥ 0}

-- Theorem stating that A, B, and C are functions, while D is not
theorem relationship_functions :
  (∀ x : ℝ, ∃! y : ℝ, y = relationA x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationB x) ∧
  (∀ x : ℝ, ∃! y : ℝ, y = relationC x) ∧
  ¬(∀ x : ℝ, ∃! y : ℝ, y ∈ relationD x) :=
by sorry

end NUMINAMATH_CALUDE_relationship_functions_l3624_362499


namespace NUMINAMATH_CALUDE_min_disks_for_problem_l3624_362405

/-- Represents a file with its size in MB -/
structure File where
  size : Float

/-- Represents a disk with its capacity in MB -/
structure Disk where
  capacity : Float

/-- Function to calculate the minimum number of disks needed -/
def min_disks_needed (files : List File) (disk_capacity : Float) : Nat :=
  sorry

/-- Theorem stating the minimum number of disks needed for the given problem -/
theorem min_disks_for_problem : 
  let files : List File := 
    (List.replicate 5 ⟨1.0⟩) ++ 
    (List.replicate 15 ⟨0.6⟩) ++ 
    (List.replicate 25 ⟨0.3⟩)
  let disk_capacity : Float := 1.44
  min_disks_needed files disk_capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_for_problem_l3624_362405


namespace NUMINAMATH_CALUDE_average_and_difference_l3624_362441

theorem average_and_difference (x : ℝ) : 
  (23 + x) / 2 = 27 → |x - 23| = 8 := by
sorry

end NUMINAMATH_CALUDE_average_and_difference_l3624_362441


namespace NUMINAMATH_CALUDE_eccentricity_range_lower_bound_l3624_362436

/-- The common foci of an ellipse and a hyperbola -/
structure CommonFoci :=
  (F₁ F₂ : ℝ × ℝ)

/-- An ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse :=
  (a b : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)

/-- A hyperbola with equation x²/m² - y²/n² = 1 -/
structure Hyperbola :=
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)

/-- A point in the first quadrant -/
structure FirstQuadrantPoint :=
  (P : ℝ × ℝ)
  (h_x_pos : P.1 > 0)
  (h_y_pos : P.2 > 0)

/-- The main theorem -/
theorem eccentricity_range_lower_bound
  (cf : CommonFoci)
  (e : Ellipse)
  (h : Hyperbola)
  (P : FirstQuadrantPoint)
  (h_common_point : P.P ∈ {x : ℝ × ℝ | x.1^2 / e.a^2 + x.2^2 / e.b^2 = 1} ∩
                            {x : ℝ × ℝ | x.1^2 / h.m^2 - x.2^2 / h.n^2 = 1})
  (h_orthogonal : (cf.F₂.1 - P.P.1, cf.F₂.2 - P.P.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) +
                  (cf.F₂.1 - cf.F₁.1, cf.F₂.2 - cf.F₁.2) • (P.P.1 - cf.F₁.1, P.P.2 - cf.F₁.2) = 0)
  (e₁ : ℝ)
  (h_e₁ : e₁ = Real.sqrt (1 - e.b^2 / e.a^2))
  (e₂ : ℝ)
  (h_e₂ : e₂ = Real.sqrt (1 + h.n^2 / h.m^2)) :
  (4 + e₁ * e₂) / (2 * e₁) ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_eccentricity_range_lower_bound_l3624_362436


namespace NUMINAMATH_CALUDE_added_classes_l3624_362489

theorem added_classes (initial_classes : ℕ) (students_per_class : ℕ) (new_total_students : ℕ)
  (h1 : initial_classes = 15)
  (h2 : students_per_class = 20)
  (h3 : new_total_students = 400) :
  (new_total_students - initial_classes * students_per_class) / students_per_class = 5 := by
sorry

end NUMINAMATH_CALUDE_added_classes_l3624_362489


namespace NUMINAMATH_CALUDE_P_less_than_Q_l3624_362447

theorem P_less_than_Q : ∀ x : ℝ, (x - 2) * (x - 4) < (x - 3)^2 := by sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l3624_362447


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3624_362404

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 - 1 = 5 * x ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -5 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3624_362404


namespace NUMINAMATH_CALUDE_f_symmetry_solutions_l3624_362464

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = x^3 + 6

theorem f_symmetry_solutions (f : ℝ → ℝ) (hf : f_condition f) :
  {x : ℝ | x ≠ 0 ∧ f x = f (-x)} = {(1/2)^(1/6), -(1/2)^(1/6)} := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_solutions_l3624_362464


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_square_l3624_362481

theorem right_triangle_inscribed_square (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) :
  (a^2 + b^2 = c^2) ∧ 
  (let s := c / Real.sqrt 2;
   s^2 = 760.5) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_square_l3624_362481


namespace NUMINAMATH_CALUDE_candy_box_problem_l3624_362461

theorem candy_box_problem (milk_chocolate dark_chocolate milk_almond : ℕ) 
  (h1 : milk_chocolate = 25)
  (h2 : dark_chocolate = 25)
  (h3 : milk_almond = 25)
  (h4 : ∀ chocolate_type, chocolate_type = milk_chocolate ∨ 
                          chocolate_type = dark_chocolate ∨ 
                          chocolate_type = milk_almond ∨ 
                          chocolate_type = white_chocolate →
        chocolate_type = (milk_chocolate + dark_chocolate + milk_almond + white_chocolate) / 4) :
  white_chocolate = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_box_problem_l3624_362461


namespace NUMINAMATH_CALUDE_weight_replacement_l3624_362423

theorem weight_replacement (n : ℕ) (avg_increase w_new : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  w_new = 119.4 →
  ∃ w_old : ℝ, w_old = w_new - n * avg_increase ∧ w_old = 76 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l3624_362423
