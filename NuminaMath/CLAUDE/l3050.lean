import Mathlib

namespace NUMINAMATH_CALUDE_monomial_combination_l3050_305047

theorem monomial_combination (m n : ℤ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 3 * a^4 * b^(n+2) = 5 * a^(m-1) * b^(2*n+3)) → 
  m + n = 4 := by
sorry

end NUMINAMATH_CALUDE_monomial_combination_l3050_305047


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3050_305059

open Set

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 ≤ y} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3050_305059


namespace NUMINAMATH_CALUDE_prob_red_after_transfer_l3050_305012

-- Define the initial contents of bags A and B
def bag_A : Finset (Fin 3) := {0, 1, 2}
def bag_B : Finset (Fin 3) := {0, 1, 2}

-- Define the number of balls of each color in bag A
def red_A : ℕ := 3
def white_A : ℕ := 2
def black_A : ℕ := 5

-- Define the number of balls of each color in bag B
def red_B : ℕ := 3
def white_B : ℕ := 3
def black_B : ℕ := 4

-- Define the total number of balls in each bag
def total_A : ℕ := red_A + white_A + black_A
def total_B : ℕ := red_B + white_B + black_B

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_B : ℚ := 3 / 10

-- State the theorem
theorem prob_red_after_transfer : 
  (red_A * (red_B + 1) + white_A * red_B + black_A * red_B) / 
  (total_A * (total_B + 1)) = prob_red_B := by sorry

end NUMINAMATH_CALUDE_prob_red_after_transfer_l3050_305012


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3050_305004

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6}
def B : Set ℕ := {1,3,5,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3050_305004


namespace NUMINAMATH_CALUDE_equal_sum_of_squares_8_9_larger_sum_of_squares_12_11_l3050_305037

-- Define the polynomial f(x) = 4x³ - 18x² + 24x
def f (x : ℝ) : ℝ := 4 * x^3 - 18 * x^2 + 24 * x

-- Define a function to calculate the sum of squares of roots
def sum_of_squares_of_roots (a b c d : ℝ) : ℝ := sorry

-- Theorem 1: Equal sum of squares of roots for f(x) = 8 and f(x) = 9
theorem equal_sum_of_squares_8_9 :
  sum_of_squares_of_roots 4 (-18) 24 (-8) = sum_of_squares_of_roots 4 (-18) 24 (-9) := by sorry

-- Theorem 2: Larger sum of squares of roots for f(x) = 12 compared to f(x) = 11
theorem larger_sum_of_squares_12_11 :
  sum_of_squares_of_roots 4 (-18) 24 (-12) > sum_of_squares_of_roots 4 (-18) 24 (-11) := by sorry

end NUMINAMATH_CALUDE_equal_sum_of_squares_8_9_larger_sum_of_squares_12_11_l3050_305037


namespace NUMINAMATH_CALUDE_earnings_difference_is_250_l3050_305000

/-- Calculates the difference between B's earnings and A's earnings given investment ratios, return ratios, and total earnings -/
def earnings_difference (investment_ratio_a investment_ratio_b investment_ratio_c : ℚ)
                        (return_ratio_a return_ratio_b return_ratio_c : ℚ)
                        (total_earnings : ℚ) : ℚ :=
  let total_ratio := investment_ratio_a * return_ratio_a + 
                     investment_ratio_b * return_ratio_b + 
                     investment_ratio_c * return_ratio_c
  let earnings_a := (investment_ratio_a * return_ratio_a / total_ratio) * total_earnings
  let earnings_b := (investment_ratio_b * return_ratio_b / total_ratio) * total_earnings
  earnings_b - earnings_a

/-- Theorem stating that the difference between B's earnings and A's earnings is 250 -/
theorem earnings_difference_is_250 :
  earnings_difference 3 4 5 6 5 4 7250 = 250 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_is_250_l3050_305000


namespace NUMINAMATH_CALUDE_cone_symmetry_properties_l3050_305057

-- Define the types of cones
inductive ConeType
  | Bounded
  | UnboundedSingleNapped
  | UnboundedDoubleNapped

-- Define symmetry properties
structure SymmetryProperties where
  hasAxis : Bool
  hasPlaneBundleThroughAxis : Bool
  hasCentralSymmetry : Bool
  hasPerpendicularPlane : Bool

-- Function to determine symmetry properties based on cone type
def symmetryPropertiesForCone (coneType : ConeType) : SymmetryProperties :=
  match coneType with
  | ConeType.Bounded => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedSingleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedDoubleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := true,
      hasPerpendicularPlane := true
    }

theorem cone_symmetry_properties (coneType : ConeType) :
  (coneType = ConeType.Bounded ∨ coneType = ConeType.UnboundedSingleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = false ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = false
  ∧
  (coneType = ConeType.UnboundedDoubleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = true ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = true :=
by sorry

end NUMINAMATH_CALUDE_cone_symmetry_properties_l3050_305057


namespace NUMINAMATH_CALUDE_real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l3050_305093

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : Prop := x^2 - 4*x + 2*k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic k x

-- Define the second quadratic equation
def quadratic2 (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m - 1 = 0

-- Theorem for part 1
theorem real_roots_iff_k_le_2 :
  ∀ k : ℝ, has_real_roots k ↔ k ≤ 2 :=
sorry

-- Theorem for part 2
theorem m_eq_3_and_other_root_4 :
  ∃ x : ℝ, quadratic 2 x ∧ quadratic2 3 x ∧ quadratic2 3 4 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l3050_305093


namespace NUMINAMATH_CALUDE_parallel_transitive_infinite_perpendicular_to_skew_l3050_305001

-- Define the concept of a line in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define the concept of parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Define the concept of perpendicular lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define the concept of skew lines
def skew (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem 1: Transitivity of parallel lines
theorem parallel_transitive (a b c : Line3D) :
  parallel a b → parallel b c → parallel a c :=
sorry

-- Theorem 2: Infinitely many perpendicular lines to skew lines
theorem infinite_perpendicular_to_skew (a b : Line3D) (h : skew a b) :
  ∃ (S : Set Line3D), (∀ l ∈ S, perpendicular l a ∧ perpendicular l b) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_infinite_perpendicular_to_skew_l3050_305001


namespace NUMINAMATH_CALUDE_aluminium_hydroxide_weight_l3050_305030

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of moles of Aluminium hydroxide -/
def num_moles : ℝ := 4

/-- The molecular weight of Aluminium hydroxide (Al(OH)₃) in g/mol -/
def molecular_weight_AlOH3 : ℝ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- The total weight of the given number of moles of Aluminium hydroxide in grams -/
def total_weight : ℝ := num_moles * molecular_weight_AlOH3

theorem aluminium_hydroxide_weight :
  total_weight = 312.04 := by sorry

end NUMINAMATH_CALUDE_aluminium_hydroxide_weight_l3050_305030


namespace NUMINAMATH_CALUDE_complex_trajectory_line_l3050_305049

theorem complex_trajectory_line (z : ℂ) :
  Complex.abs (z + 1) = Complex.abs (1 + Complex.I * z) →
  (z.re : ℝ) + z.im = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_line_l3050_305049


namespace NUMINAMATH_CALUDE_set_operations_l3050_305013

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x | x > 1}) ∧
  (A ∩ (Set.univ \ B) = {x | 1 < x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3050_305013


namespace NUMINAMATH_CALUDE_sqrt_cubed_equals_64_l3050_305011

theorem sqrt_cubed_equals_64 (x : ℝ) : (Real.sqrt x)^3 = 64 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cubed_equals_64_l3050_305011


namespace NUMINAMATH_CALUDE_smallest_odd_four_digit_number_l3050_305072

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * b + 10 * a + d

theorem smallest_odd_four_digit_number (n : ℕ) : 
  is_four_digit n ∧ 
  n % 2 = 1 ∧
  swap_digits n - n = 5940 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, is_four_digit m ∧ m % 2 = 1 ∧ swap_digits m - m = 5940 ∧ m % 9 = 8 → n ≤ m) →
  n = 1979 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_four_digit_number_l3050_305072


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3050_305075

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc 1 2 = {x | (x - 1) * (x - 2) ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3050_305075


namespace NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l3050_305014

theorem smallest_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → a / b = 5 / 4 → min a b = 40 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l3050_305014


namespace NUMINAMATH_CALUDE_root_equation_m_value_l3050_305097

theorem root_equation_m_value :
  ∀ m : ℝ, ((-4)^2 + m * (-4) - 20 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_m_value_l3050_305097


namespace NUMINAMATH_CALUDE_expression_simplification_and_ratio_l3050_305009

theorem expression_simplification_and_ratio :
  let expr := (6 * m + 4 * n + 12) / 4
  let a := 3/2
  let b := 1
  let c := 3
  expr = a * m + b * n + c ∧ (a + b + c) / c = 11/6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_ratio_l3050_305009


namespace NUMINAMATH_CALUDE_cos_minus_sin_value_l3050_305099

theorem cos_minus_sin_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2) : 
  Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_value_l3050_305099


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3050_305056

theorem quadratic_inequality : ∀ x : ℝ, 2*x^2 + 5*x + 3 > x^2 + 4*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3050_305056


namespace NUMINAMATH_CALUDE_expression_factorization_l3050_305045

theorem expression_factorization (x : ℝ) : 
  (21 * x^4 + 90 * x^3 + 40 * x - 10) - (7 * x^4 + 6 * x^3 + 8 * x - 6) = 
  2 * x * (7 * x^3 + 42 * x^2 + 16) - 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3050_305045


namespace NUMINAMATH_CALUDE_derivative_f_l3050_305071

noncomputable def f (x : ℝ) : ℝ :=
  (4^x * (Real.log 4 * Real.sin (4*x) - 4 * Real.cos (4*x))) / (16 + Real.log 4^2)

theorem derivative_f (x : ℝ) :
  deriv f x = 4^x * Real.sin (4*x) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_l3050_305071


namespace NUMINAMATH_CALUDE_middle_speed_calculation_l3050_305063

/-- Represents the speed and duration of a part of the journey -/
structure JourneyPart where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (part : JourneyPart) : ℝ := part.speed * part.duration

theorem middle_speed_calculation (total_distance : ℝ) (first_part last_part middle_part : JourneyPart) 
  (h1 : total_distance = 800)
  (h2 : first_part.speed = 80 ∧ first_part.duration = 6)
  (h3 : last_part.speed = 40 ∧ last_part.duration = 2)
  (h4 : middle_part.duration = 4)
  (h5 : total_distance = distance first_part + distance middle_part + distance last_part) :
  middle_part.speed = 60 := by
sorry

end NUMINAMATH_CALUDE_middle_speed_calculation_l3050_305063


namespace NUMINAMATH_CALUDE_min_chestnuts_is_253_l3050_305035

/-- Represents the process of a monkey dividing chestnuts -/
def monkey_divide (n : ℕ) : ℕ :=
  if n % 4 = 1 then (3 * (n - 1)) / 4 else 0

/-- Represents the process of all four monkeys dividing chestnuts -/
def all_monkeys_divide (n : ℕ) : ℕ :=
  monkey_divide (monkey_divide (monkey_divide (monkey_divide n)))

/-- Theorem stating that 253 is the minimum number of chestnuts that satisfies the problem conditions -/
theorem min_chestnuts_is_253 :
  (∀ m : ℕ, m < 253 → all_monkeys_divide m ≠ 0) ∧ all_monkeys_divide 253 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_chestnuts_is_253_l3050_305035


namespace NUMINAMATH_CALUDE_ivan_fate_l3050_305082

structure Animal where
  name : String
  always_truth : Bool
  alternating : Bool
  deriving Repr

def Statement := (Bool × Bool)

theorem ivan_fate (bear fox wolf : Animal)
  (h_bear : bear.always_truth = true ∧ bear.alternating = false)
  (h_fox : fox.always_truth = false ∧ fox.alternating = false)
  (h_wolf : wolf.always_truth = false ∧ wolf.alternating = true)
  (statement1 statement2 statement3 : Statement)
  (h_distinct : bear ≠ fox ∧ bear ≠ wolf ∧ fox ≠ wolf)
  : ∃ (animal1 animal2 animal3 : Animal),
    animal1 = fox ∧ animal2 = wolf ∧ animal3 = bear ∧
    (¬statement1.1 ∧ ¬statement1.2) ∧
    (statement2.1 ∧ ¬statement2.2) ∧
    (statement3.1 ∧ statement3.2) :=
by sorry

#check ivan_fate

end NUMINAMATH_CALUDE_ivan_fate_l3050_305082


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3050_305044

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 9) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3050_305044


namespace NUMINAMATH_CALUDE_ceiling_sqrt_162_l3050_305076

theorem ceiling_sqrt_162 : ⌈Real.sqrt 162⌉ = 13 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_162_l3050_305076


namespace NUMINAMATH_CALUDE_cyclist_speed_l3050_305090

-- Define the parameters of the problem
def first_distance : ℝ := 8
def second_distance : ℝ := 10
def second_speed : ℝ := 8
def total_average_speed : ℝ := 8.78

-- Define the theorem
theorem cyclist_speed (v : ℝ) (h : v > 0) :
  (first_distance + second_distance) / ((first_distance / v) + (second_distance / second_speed)) = total_average_speed →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l3050_305090


namespace NUMINAMATH_CALUDE_alice_win_probability_l3050_305087

-- Define the game types
inductive Move
| Rock
| Paper
| Scissors

-- Define the player types
inductive Player
| Alice
| Bob
| Other

-- Define the tournament structure
def TournamentSize : Nat := 8
def NumRounds : Nat := 3

-- Define the rules of the game
def beats (m1 m2 : Move) : Bool :=
  match m1, m2 with
  | Move.Rock, Move.Scissors => true
  | Move.Scissors, Move.Paper => true
  | Move.Paper, Move.Rock => true
  | _, _ => false

-- Define the strategy for each player
def playerMove (p : Player) : Move :=
  match p with
  | Player.Alice => Move.Rock
  | Player.Bob => Move.Paper
  | Player.Other => Move.Scissors

-- Define the probability of Alice winning
def aliceWinProbability : Rat := 6/7

-- Theorem statement
theorem alice_win_probability :
  (TournamentSize = 8) →
  (NumRounds = 3) →
  (∀ p, playerMove p = match p with
    | Player.Alice => Move.Rock
    | Player.Bob => Move.Paper
    | Player.Other => Move.Scissors) →
  (∀ m1 m2, beats m1 m2 = match m1, m2 with
    | Move.Rock, Move.Scissors => true
    | Move.Scissors, Move.Paper => true
    | Move.Paper, Move.Rock => true
    | _, _ => false) →
  aliceWinProbability = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_alice_win_probability_l3050_305087


namespace NUMINAMATH_CALUDE_notebook_duration_is_seven_l3050_305046

/-- Represents the number of weeks John's notebooks last -/
def notebook_duration (
  num_notebooks : ℕ
  ) (pages_per_notebook : ℕ
  ) (math_pages_per_day : ℕ
  ) (math_days_per_week : ℕ
  ) (science_pages_per_day : ℕ
  ) (science_days_per_week : ℕ
  ) (history_pages_per_day : ℕ
  ) (history_days_per_week : ℕ
  ) : ℕ :=
  let total_pages := num_notebooks * pages_per_notebook
  let pages_per_week := 
    math_pages_per_day * math_days_per_week +
    science_pages_per_day * science_days_per_week +
    history_pages_per_day * history_days_per_week
  total_pages / pages_per_week

theorem notebook_duration_is_seven :
  notebook_duration 5 40 4 3 5 2 6 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_is_seven_l3050_305046


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3050_305048

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 2) / (x - 3)) ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3050_305048


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_6m_l3050_305073

theorem factorization_3m_squared_minus_6m (m : ℝ) : 3 * m^2 - 6 * m = 3 * m * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_6m_l3050_305073


namespace NUMINAMATH_CALUDE_nikita_produces_two_per_hour_l3050_305027

-- Define the productivity of Ivan and Nikita
def ivan_productivity : ℝ := sorry
def nikita_productivity : ℝ := sorry

-- Define the conditions from the problem
axiom monday_condition : 3 * ivan_productivity + 2 * nikita_productivity = 7
axiom tuesday_condition : 5 * ivan_productivity + 3 * nikita_productivity = 11

-- Theorem to prove
theorem nikita_produces_two_per_hour : nikita_productivity = 2 := by
  sorry

end NUMINAMATH_CALUDE_nikita_produces_two_per_hour_l3050_305027


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l3050_305083

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ (n : ℕ) (k : ℕ), pascal n k = m

theorem smallest_four_digit_in_pascal : 
  (∀ m : ℕ, m < 1000 → ¬(inPascalTriangle m ∧ m ≥ 1000)) ∧ 
  inPascalTriangle 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l3050_305083


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3050_305051

/-- Given a polynomial f(x) = ax^4 + bx^3 + cx^2 + dx + e where f(-3) = -5,
    prove that 8a - 4b + 2c - d + e = -5 -/
theorem polynomial_value_theorem (a b c d e : ℝ) :
  (fun x : ℝ ↦ a * x^4 + b * x^3 + c * x^2 + d * x + e) (-3) = -5 →
  8 * a - 4 * b + 2 * c - d + e = -5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3050_305051


namespace NUMINAMATH_CALUDE_second_quadrant_complex_number_range_l3050_305095

theorem second_quadrant_complex_number_range (m : ℝ) : 
  let z : ℂ := m - 1 + (m + 2) * I
  (z.re < 0 ∧ z.im > 0) ↔ -2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_number_range_l3050_305095


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l3050_305043

/-- Given a function f with period π and its left-translated version g, 
    prove the interval of monotonic increase for g. -/
theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (h_f_def : ∀ x, f x = Real.sin (ω * x - π / 4))
  (h_f_period : ∀ x, f (x + π) = f x)
  (g : ℝ → ℝ)
  (h_g_def : ∀ x, g x = f (x + π / 4)) :
  ∀ k : ℤ, StrictMonoOn g (Set.Icc (-3 * π / 8 + k * π) (π / 8 + k * π)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l3050_305043


namespace NUMINAMATH_CALUDE_line_BC_equation_triangle_ABC_area_l3050_305021

-- Define the points of the triangle
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (5, -2)
def C : ℝ × ℝ := (3, 5)

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := A, B := B, C := C }

-- Theorem for the equation of line BC
theorem line_BC_equation (t : Triangle) (l : Line) : 
  t = ABC → l.a = 7 ∧ l.b = 2 ∧ l.c = -31 → 
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
  l.a * t.C.1 + l.b * t.C.2 + l.c = 0 :=
sorry

-- Theorem for the area of triangle ABC
theorem triangle_ABC_area (t : Triangle) : 
  t = ABC → (1/2) * |t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2)| = 29/2 :=
sorry

end NUMINAMATH_CALUDE_line_BC_equation_triangle_ABC_area_l3050_305021


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3050_305053

-- Define the quadratic function
def f (b c x : ℝ) := x^2 + b*x + c

-- Define the solution set condition
def solution_set (b c : ℝ) : Prop :=
  ∀ x, f b c x > 0 ↔ (x > 2 ∨ x < 1)

-- Theorem statement
theorem quadratic_inequality (b c : ℝ) (h : solution_set b c) :
  (b = -3 ∧ c = 2) ∧
  (∀ x, 2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3050_305053


namespace NUMINAMATH_CALUDE_absolute_value_and_square_l3050_305032

theorem absolute_value_and_square (x : ℝ) : 
  (x < 0 → abs x > x) ∧ (x > 2 → x^2 > 4) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_square_l3050_305032


namespace NUMINAMATH_CALUDE_strictly_decreasing_exponential_range_l3050_305015

theorem strictly_decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → a ∈ Set.Ioo (1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_strictly_decreasing_exponential_range_l3050_305015


namespace NUMINAMATH_CALUDE_first_term_values_l3050_305089

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_values (a b : ℕ) :
  fibonacci_like_sequence a b 2 = 7 ∧
  fibonacci_like_sequence a b 2013 % 4 = 1 →
  a = 1 ∨ a = 5 := by
sorry

end NUMINAMATH_CALUDE_first_term_values_l3050_305089


namespace NUMINAMATH_CALUDE_circle_fit_theorem_l3050_305068

/-- Represents a square with unit side length -/
structure UnitSquare where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The main theorem statement -/
theorem circle_fit_theorem (rect : Rectangle) (squares : Finset UnitSquare) :
  rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
  ∃ (cx cy : ℝ), cx ∈ Set.Icc 0.5 19.5 ∧ cy ∈ Set.Icc 0.5 24.5 ∧
    ∀ (s : UnitSquare), s ∈ squares →
      (cx - s.x) ^ 2 + (cy - s.y) ^ 2 > 0.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_fit_theorem_l3050_305068


namespace NUMINAMATH_CALUDE_medication_dosage_range_l3050_305022

theorem medication_dosage_range 
  (daily_min : ℝ) 
  (daily_max : ℝ) 
  (num_doses : ℕ) 
  (h1 : daily_min = 60) 
  (h2 : daily_max = 120) 
  (h3 : num_doses = 4) :
  ∃ x_min x_max : ℝ, 
    x_min = daily_min / num_doses ∧ 
    x_max = daily_max / num_doses ∧ 
    x_min = 15 ∧ 
    x_max = 30 ∧ 
    ∀ x : ℝ, (x_min ≤ x ∧ x ≤ x_max) ↔ (15 ≤ x ∧ x ≤ 30) :=
by sorry

end NUMINAMATH_CALUDE_medication_dosage_range_l3050_305022


namespace NUMINAMATH_CALUDE_square_less_than_self_for_unit_interval_l3050_305007

theorem square_less_than_self_for_unit_interval (x : ℝ) : 0 < x → x < 1 → x^2 < x := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_self_for_unit_interval_l3050_305007


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_l3050_305084

/-- For a parabola with equation y² = ax, if the distance from its focus to its directrix is 2, then a = ±4 -/
theorem parabola_focus_directrix (a : ℝ) : 
  (∃ (y x : ℝ), y^2 = a*x) →  -- parabola equation
  (∃ (p : ℝ), p = 2) →        -- distance from focus to directrix
  (a = 4 ∨ a = -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_l3050_305084


namespace NUMINAMATH_CALUDE_shortest_diagonal_probability_l3050_305098

/-- The number of sides in the regular polygon -/
def n : ℕ := 11

/-- The total number of diagonals in the polygon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in the polygon -/
def shortest_diagonals : ℕ := n / 2

/-- The probability of selecting a shortest diagonal -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem shortest_diagonal_probability :
  probability = 5 / 44 := by sorry

end NUMINAMATH_CALUDE_shortest_diagonal_probability_l3050_305098


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3050_305058

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse length
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3050_305058


namespace NUMINAMATH_CALUDE_ellipse_to_hyperbola_l3050_305019

/-- Given an ellipse with equation x²/8 + y²/5 = 1 where its foci are its vertices,
    prove that the equation of the hyperbola with foci at the vertices of the ellipse
    is x²/3 - y²/5 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (x^2 / 8 + y^2 / 5 = 1) ∧
    (c^2 = a^2 + b^2) ∧
    (c = 2 * a)) →
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (x^2 / 3 - y^2 / 5 = 1) ∧
    (c'^2 = a'^2 + b'^2) ∧
    (c' = 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_to_hyperbola_l3050_305019


namespace NUMINAMATH_CALUDE_daps_to_dups_l3050_305006

-- Define the units
variable (dap dop dip dup : ℚ)

-- Define the exchange rates
axiom rate1 : 5 * dap = 4 * dop
axiom rate2 : 3 * dop = 9 * dip
axiom rate3 : 5 * dip = 2 * dup

-- Theorem to prove
theorem daps_to_dups : 37.5 * dap = 36 * dup := by
  sorry

end NUMINAMATH_CALUDE_daps_to_dups_l3050_305006


namespace NUMINAMATH_CALUDE_computer_price_difference_computer_price_difference_is_500_l3050_305062

/-- The price difference between an enhanced computer and a basic computer -/
theorem computer_price_difference : ℝ → Prop :=
  fun difference =>
    let basic_price := 2000
    let printer_price := 2500 - basic_price
    let enhanced_price := 6 * printer_price
    difference = enhanced_price - basic_price

/-- Proof that the price difference is $500 -/
theorem computer_price_difference_is_500 : computer_price_difference 500 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_difference_computer_price_difference_is_500_l3050_305062


namespace NUMINAMATH_CALUDE_inequality_proof_l3050_305028

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) ≥ 5/2 ∧
  ((a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) = 5/2 ↔ a = b ∧ b = c) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3050_305028


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3050_305010

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n - 2) * 180 = 900 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l3050_305010


namespace NUMINAMATH_CALUDE_drama_ticket_revenue_l3050_305039

theorem drama_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) : ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
  full_price + half_price = total_tickets ∧
  full_price * price + half_price * (price / 2) = total_revenue ∧
  full_price * price = 1600 :=
sorry

end NUMINAMATH_CALUDE_drama_ticket_revenue_l3050_305039


namespace NUMINAMATH_CALUDE_complex_expression_equals_five_l3050_305069

theorem complex_expression_equals_five :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1/3))^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_five_l3050_305069


namespace NUMINAMATH_CALUDE_evaluate_expression_l3050_305055

theorem evaluate_expression : 3^(1^(2^8)) + ((3^1)^2)^4 = 6564 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3050_305055


namespace NUMINAMATH_CALUDE_steve_bike_time_l3050_305031

/-- Given that Steve biked 5 miles in the same time Jordan biked 3 miles,
    and Jordan took 18 minutes to bike 3 miles,
    prove that Steve will take 126/5 minutes to bike 7 miles. -/
theorem steve_bike_time (steve_distance : ℝ) (jordan_distance : ℝ) (jordan_time : ℝ) (steve_new_distance : ℝ) :
  steve_distance = 5 →
  jordan_distance = 3 →
  jordan_time = 18 →
  steve_new_distance = 7 →
  (steve_new_distance / (steve_distance / jordan_time)) = 126 / 5 := by
  sorry

end NUMINAMATH_CALUDE_steve_bike_time_l3050_305031


namespace NUMINAMATH_CALUDE_jason_shopping_total_l3050_305040

theorem jason_shopping_total (jacket_cost shorts_cost : ℚ) 
  (h1 : jacket_cost = 4.74)
  (h2 : shorts_cost = 9.54) :
  jacket_cost + shorts_cost = 14.28 := by
  sorry

end NUMINAMATH_CALUDE_jason_shopping_total_l3050_305040


namespace NUMINAMATH_CALUDE_flag_combinations_l3050_305024

def available_colors : ℕ := 6
def stripes : ℕ := 3

theorem flag_combinations : (available_colors * (available_colors - 1) * (available_colors - 2)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_flag_combinations_l3050_305024


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l3050_305029

theorem grape_juice_mixture (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 50 →
  initial_percentage = 0.1 →
  added_volume = 10 →
  let initial_grape_juice := initial_volume * initial_percentage
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  let final_percentage := total_grape_juice / final_volume
  final_percentage = 0.25 := by sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l3050_305029


namespace NUMINAMATH_CALUDE_base_8_digit_product_l3050_305085

def base_10_num : ℕ := 7890

def to_base_8 (n : ℕ) : List ℕ :=
  sorry

def digit_product (digits : List ℕ) : ℕ :=
  sorry

theorem base_8_digit_product :
  digit_product (to_base_8 base_10_num) = 84 :=
sorry

end NUMINAMATH_CALUDE_base_8_digit_product_l3050_305085


namespace NUMINAMATH_CALUDE_max_profit_selling_price_l3050_305067

-- Define the profit function
def profit (x : ℝ) : ℝ := 10 * (-x^2 + 140*x - 4000)

-- Define the theorem
theorem max_profit_selling_price :
  -- Given conditions
  let cost_price : ℝ := 40
  let initial_price : ℝ := 50
  let initial_sales : ℝ := 500
  let price_sensitivity : ℝ := 10

  -- Theorem statement
  ∃ (max_price max_profit : ℝ),
    -- The maximum price is greater than the cost price
    max_price > cost_price ∧
    -- The maximum profit occurs at the maximum price
    profit max_price = max_profit ∧
    -- The maximum profit is indeed the maximum
    ∀ x > cost_price, profit x ≤ max_profit ∧
    -- The specific values for maximum price and profit
    max_price = 70 ∧ max_profit = 9000 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_l3050_305067


namespace NUMINAMATH_CALUDE_pyramid_volume_l3050_305018

/-- The volume of a pyramid with an equilateral triangular base of side length 1/√2 and height 1 is √3/24 -/
theorem pyramid_volume : 
  let base_side : ℝ := 1 / Real.sqrt 2
  let height : ℝ := 1
  let base_area : ℝ := (Real.sqrt 3 / 4) * base_side ^ 2
  let volume : ℝ := (1 / 3) * base_area * height
  volume = Real.sqrt 3 / 24 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3050_305018


namespace NUMINAMATH_CALUDE_imaginary_part_of_symmetrical_complex_ratio_l3050_305066

theorem imaginary_part_of_symmetrical_complex_ratio :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  Complex.im (z₂ / z₁) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_symmetrical_complex_ratio_l3050_305066


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l3050_305060

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ¬ (∀ x y z : ℝ, 
    (x = (a + 1/a) * (b + 1/b) ∧ 
     y = (Real.sqrt (a * b) + 1 / Real.sqrt (a * b))^2 ∧ 
     z = ((a + b)/2 + 2/(a + b))^2) →
    (x > y ∧ x > z) ∨ (y > x ∧ y > z) ∨ (z > x ∧ z > y)) :=
by sorry


end NUMINAMATH_CALUDE_comparison_of_expressions_l3050_305060


namespace NUMINAMATH_CALUDE_complement_of_union_equals_four_l3050_305003

universe u

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_equals_four :
  (M ∪ N)ᶜ = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_four_l3050_305003


namespace NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l3050_305017

/-- The function f(x) = x³ - x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem function_minimum_and_tangent_line (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a 1 ≤ f a x) →
  a = -1 ∧
  (∃ x₀ : ℝ, f a x₀ - (-1) = f' a x₀ * (x₀ - (-1)) ∧
              (x₀ = 1 ∨ 4 * x₀ - f a x₀ + 3 = 0)) :=
sorry

end NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l3050_305017


namespace NUMINAMATH_CALUDE_cross_shaded_area_equality_l3050_305061

-- Define the rectangle and shaded area properties
def rectangle_length : ℝ := 9
def rectangle_width : ℝ := 8
def shaded_rect1_width : ℝ := 3

-- Define the shaded area as a function of X
def shaded_area (x : ℝ) : ℝ :=
  shaded_rect1_width * rectangle_width + rectangle_length * x - shaded_rect1_width * x

-- Define the total area of the rectangle
def total_area : ℝ := rectangle_length * rectangle_width

-- State the theorem
theorem cross_shaded_area_equality (x : ℝ) :
  shaded_area x = (1 / 2) * total_area → x = 2 := by sorry

end NUMINAMATH_CALUDE_cross_shaded_area_equality_l3050_305061


namespace NUMINAMATH_CALUDE_ellipse_properties_l3050_305025

-- Define the ellipse C₁
def ellipse_C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola C₂
def hyperbola_C₂ (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

-- Define the line that intersects C₁
def intersecting_line (x y : ℝ) : Prop := x + y = 1

-- Define the perimeter of triangle PF₁F₂
def triangle_perimeter (a c : ℝ) : Prop := 2 * a + 2 * c = 2 * Real.sqrt 3 + 2

-- Define the eccentricity range
def eccentricity_range (e : ℝ) : Prop := Real.sqrt 3 / 3 ≤ e ∧ e ≤ Real.sqrt 2 / 2

-- Define the condition for circle passing through origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem ellipse_properties (a b c : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : c = 1) -- Foci of C₁ are vertices of C₂
  (h₄ : triangle_perimeter a c) :
  -- 1. Equation of C₁
  (∀ x y, ellipse_C₁ x y a b ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- 2. Length of chord AB
  (∃ x₁ y₁ x₂ y₂, 
    ellipse_C₁ x₁ y₁ a b ∧ 
    ellipse_C₁ x₂ y₂ a b ∧ 
    intersecting_line x₁ y₁ ∧ 
    intersecting_line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (8 * Real.sqrt 3 / 5)^2) ∧
  -- 3. Range of major axis length
  (∀ e, eccentricity_range e →
    (∃ x₁ y₁ x₂ y₂, 
      ellipse_C₁ x₁ y₁ a b ∧ 
      ellipse_C₁ x₂ y₂ a b ∧ 
      intersecting_line x₁ y₁ ∧ 
      intersecting_line x₂ y₂ ∧
      circle_through_origin x₁ y₁ x₂ y₂) →
    Real.sqrt 5 ≤ 2 * a ∧ 2 * a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3050_305025


namespace NUMINAMATH_CALUDE_repair_time_for_14_people_l3050_305033

/-- Represents the time needed for a group of people to repair a dam -/
structure RepairTime where
  people : ℕ
  minutes : ℕ

/-- The theorem stating the time needed for 14 people to repair the dam -/
theorem repair_time_for_14_people 
  (repair1 : RepairTime) 
  (repair2 : RepairTime)
  (h1 : repair1.people = 10 ∧ repair1.minutes = 45)
  (h2 : repair2.people = 20 ∧ repair2.minutes = 20) :
  ∃ (repair3 : RepairTime), repair3.people = 14 ∧ repair3.minutes = 30 :=
sorry

end NUMINAMATH_CALUDE_repair_time_for_14_people_l3050_305033


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3050_305016

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def C : Nat := 12
def D : Nat := 13

theorem base_conversion_sum :
  let base_8_num := base_to_decimal [5, 3, 7] 8
  let base_14_num := base_to_decimal [5, C, D] 14
  base_8_num + base_14_num = 1512 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3050_305016


namespace NUMINAMATH_CALUDE_opposite_and_absolute_value_l3050_305042

theorem opposite_and_absolute_value (x y : ℤ) :
  (- x = 3 ∧ |y| = 5) → (x + y = 2 ∨ x + y = -8) :=
by sorry

end NUMINAMATH_CALUDE_opposite_and_absolute_value_l3050_305042


namespace NUMINAMATH_CALUDE_max_value_of_f_l3050_305005

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y a ≥ f x a) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ f y a) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 43) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3050_305005


namespace NUMINAMATH_CALUDE_ali_remaining_money_l3050_305086

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let after_food := initial_amount * (1 - 3/8)
  let after_glasses := after_food * (1 - 2/5)
  let after_gift := after_glasses * (1 - 1/4)
  after_gift

theorem ali_remaining_money :
  calculate_remaining_money 480 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ali_remaining_money_l3050_305086


namespace NUMINAMATH_CALUDE_symmetric_function_theorem_l3050_305026

/-- A function is symmetric to another function with respect to the origin -/
def SymmetricToOrigin (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-x) = -y

/-- The main theorem -/
theorem symmetric_function_theorem (f : ℝ → ℝ) :
  SymmetricToOrigin f (λ x ↦ 3 - 2*x) → ∀ x, f x = -2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_theorem_l3050_305026


namespace NUMINAMATH_CALUDE_gum_distribution_l3050_305038

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) : 
  cousins = 4 → total_gum = 20 → gum_per_cousin = total_gum / cousins → gum_per_cousin = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l3050_305038


namespace NUMINAMATH_CALUDE_max_dot_product_on_ellipses_l3050_305050

def ellipse_C1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

def ellipse_C2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem max_dot_product_on_ellipses :
  ∀ x1 y1 x2 y2 : ℝ,
  ellipse_C1 x1 y1 → ellipse_C2 x2 y2 →
  dot_product x1 y1 x2 y2 ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_ellipses_l3050_305050


namespace NUMINAMATH_CALUDE_difference_of_squares_601_599_l3050_305052

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_599_l3050_305052


namespace NUMINAMATH_CALUDE_paint_distribution_l3050_305080

theorem paint_distribution (total_paint : ℝ) (num_colors : ℕ) (paint_per_color : ℝ) :
  total_paint = 15 →
  num_colors = 3 →
  paint_per_color * num_colors = total_paint →
  paint_per_color = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_distribution_l3050_305080


namespace NUMINAMATH_CALUDE_intersection_characterization_l3050_305091

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def matches_x_squared_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

def has_two_distinct_intersections (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ + a ∧ f x₂ = x₂ + a

theorem intersection_characterization (f : ℝ → ℝ) (a : ℝ) :
  is_even_function f ∧ has_period_two f ∧ matches_x_squared_on_unit_interval f →
  has_two_distinct_intersections f a ↔ ∃ n : ℤ, a = 2 * n ∨ a = 2 * n - 1/4 :=
sorry

end NUMINAMATH_CALUDE_intersection_characterization_l3050_305091


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l3050_305034

theorem right_triangle_acute_angle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  c = 90 →           -- One angle is a right angle (90°)
  a = 55 →           -- One acute angle is 55°
  b = 35             -- The other acute angle is 35°
:= by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l3050_305034


namespace NUMINAMATH_CALUDE_f_at_2_l3050_305065

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_at_2 : f 2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3050_305065


namespace NUMINAMATH_CALUDE_drugstore_inventory_theorem_l3050_305008

def bottles_delivered (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) : ℕ :=
  let total_sales := monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 5)
  let remaining_before_delivery := initial_inventory - (monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 4))
  final_inventory - remaining_before_delivery

theorem drugstore_inventory_theorem (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) 
  (h1 : initial_inventory = 4500)
  (h2 : monday_sales = 2445)
  (h3 : tuesday_sales = 900)
  (h4 : daily_sales_wed_to_sun = 50)
  (h5 : final_inventory = 1555) :
  bottles_delivered initial_inventory monday_sales tuesday_sales daily_sales_wed_to_sun final_inventory = 600 := by
  sorry

end NUMINAMATH_CALUDE_drugstore_inventory_theorem_l3050_305008


namespace NUMINAMATH_CALUDE_series_sum_l3050_305064

noncomputable def series_term (n : ℕ) : ℝ :=
  (2^n : ℝ) / (3^(2^n) + 1)

theorem series_sum : ∑' (n : ℕ), series_term n = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3050_305064


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3050_305078

theorem smallest_whole_number_above_sum : ∃ (n : ℕ), 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) ∧ 
  n = 19 ∧ 
  ∀ (m : ℕ), m < n → (m : ℚ) ≤ (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) :=
by sorry

#check smallest_whole_number_above_sum

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3050_305078


namespace NUMINAMATH_CALUDE_wine_price_increase_l3050_305096

/-- The additional cost for 5 bottles of wine after a 25% price increase -/
theorem wine_price_increase (current_price : ℝ) (num_bottles : ℕ) (price_increase_percent : ℝ) :
  current_price = 20 →
  num_bottles = 5 →
  price_increase_percent = 0.25 →
  num_bottles * current_price * price_increase_percent = 25 :=
by sorry

end NUMINAMATH_CALUDE_wine_price_increase_l3050_305096


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l3050_305020

theorem shirt_price_calculation (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 105 ∧ 
  discount1 = 19.954259576901087 ∧ 
  discount2 = 12.55 →
  ∃ (original_price : ℝ), 
    original_price = 150 ∧ 
    final_price = original_price * (1 - discount1 / 100) * (1 - discount2 / 100) :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l3050_305020


namespace NUMINAMATH_CALUDE_wickets_before_last_match_value_l3050_305041

/-- Represents the bowling statistics of a cricket player -/
structure BowlingStats where
  initial_average : ℝ
  initial_wickets : ℕ
  new_wickets : ℕ
  new_runs : ℕ
  average_decrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wickets_before_last_match (stats : BowlingStats) : ℕ :=
  stats.initial_wickets

/-- Theorem stating the number of wickets taken before the last match -/
theorem wickets_before_last_match_value (stats : BowlingStats) 
  (h1 : stats.initial_average = 12.4)
  (h2 : stats.new_wickets = 3)
  (h3 : stats.new_runs = 26)
  (h4 : stats.average_decrease = 0.4)
  (h5 : stats.initial_wickets = wickets_before_last_match stats) :
  wickets_before_last_match stats = 25 := by
  sorry

#eval wickets_before_last_match { 
  initial_average := 12.4, 
  initial_wickets := 25, 
  new_wickets := 3, 
  new_runs := 26, 
  average_decrease := 0.4 
}

end NUMINAMATH_CALUDE_wickets_before_last_match_value_l3050_305041


namespace NUMINAMATH_CALUDE_revenue_change_l3050_305092

theorem revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.19) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.0685 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l3050_305092


namespace NUMINAMATH_CALUDE_alpha_value_l3050_305054

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_alpha_value_l3050_305054


namespace NUMINAMATH_CALUDE_misread_weight_l3050_305077

/-- Proves that the misread weight is 56 kg given the conditions of the problem -/
theorem misread_weight (n : ℕ) (initial_avg correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 59 ∧ 
  correct_weight = 68 →
  ∃ x : ℝ, x = 56 ∧ n * correct_avg - n * initial_avg = correct_weight - x :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_l3050_305077


namespace NUMINAMATH_CALUDE_circular_well_diameter_l3050_305074

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem circular_well_diameter 
  (depth : ℝ) 
  (volume : ℝ) 
  (h_depth : depth = 8) 
  (h_volume : volume = 25.132741228718345) : 
  2 * (volume / (Real.pi * depth))^(1/2 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_circular_well_diameter_l3050_305074


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_gcd_fermat_numbers_l3050_305070

-- Part (a)
theorem gcd_power_minus_one (a m n : ℕ) (ha : a > 1) (hm : m ≠ n) :
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 := by
sorry

-- Part (b)
def fermat (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (n m : ℕ) (h : n ≠ m) :
  Nat.gcd (fermat n) (fermat m) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_gcd_fermat_numbers_l3050_305070


namespace NUMINAMATH_CALUDE_expression_value_at_two_l3050_305036

theorem expression_value_at_two :
  let a : ℝ := 2
  (2 * a⁻¹ + 3 * a^2) / a = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l3050_305036


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l3050_305002

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l3050_305002


namespace NUMINAMATH_CALUDE_power_five_137_mod_8_l3050_305023

theorem power_five_137_mod_8 : 5^137 % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_137_mod_8_l3050_305023


namespace NUMINAMATH_CALUDE_modulus_of_z_l3050_305094

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = 2 + 4 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3050_305094


namespace NUMINAMATH_CALUDE_probability_increases_l3050_305079

/-- The probability of player A winning a game of 2n rounds -/
noncomputable def P (n : ℕ) : ℝ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℝ) / 2^(2*n))

/-- Theorem stating that the probability of winning increases with the number of rounds -/
theorem probability_increases (n : ℕ) : P (n+1) > P n := by
  sorry

end NUMINAMATH_CALUDE_probability_increases_l3050_305079


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l3050_305081

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem sets_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l3050_305081


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l3050_305088

/-- A type representing questions that can be asked -/
inductive Question
| NumberOfQuestions : Question
| CurrentTime : Question

/-- A type representing the state of the world at a given moment -/
structure WorldState where
  questionsAsked : Nat
  currentTime : Nat

/-- A function that gives the truthful answer to a question given the world state -/
def truthfulAnswer (q : Question) (w : WorldState) : Nat :=
  match q with
  | Question.NumberOfQuestions => w.questionsAsked
  | Question.CurrentTime => w.currentTime

/-- Theorem stating that there exists a question that can have different truthful answers at different times -/
theorem exists_question_with_different_answers :
  ∃ (q : Question) (w1 w2 : WorldState), w1 ≠ w2 → truthfulAnswer q w1 ≠ truthfulAnswer q w2 := by
  sorry


end NUMINAMATH_CALUDE_exists_question_with_different_answers_l3050_305088
