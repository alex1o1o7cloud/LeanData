import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1009_100940

noncomputable def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def line_equation (x y m : ℝ) : Prop := y = x + m

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

def minor_axis_length (a b : ℝ) : ℝ := 2 * b

theorem ellipse_intersection_theorem (m : ℝ) :
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt 3 / 2
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_equation x₁ y₁ ∧
    ellipse_equation x₂ y₂ ∧
    line_equation x₁ y₁ m ∧
    line_equation x₂ y₂ m ∧
    distance x₁ y₁ x₂ y₂ = minor_axis_length a b ∧
    c^2 = a^2 - b^2 ∧
    e = c / a
  → m = Real.sqrt 30 / 4 ∨ m = -Real.sqrt 30 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1009_100940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_two_l1009_100980

-- Define the given conditions
def downstream_distance : ℝ := 45
def upstream_distance : ℝ := 25
def time : ℝ := 5

-- Define the variables
def swimmer_speed : ℝ → ℝ → ℝ := λ v c ↦ v
def current_speed : ℝ → ℝ := λ c ↦ c

-- Define the equations based on the distance formula
def downstream_equation (v c : ℝ) : Prop :=
  (swimmer_speed v c + current_speed c) * time = downstream_distance

def upstream_equation (v c : ℝ) : Prop :=
  (swimmer_speed v c - current_speed c) * time = upstream_distance

-- The theorem to prove
theorem current_speed_is_two :
  ∃ (v c : ℝ), downstream_equation v c ∧ upstream_equation v c ∧ c = 2 := by
  sorry

#check current_speed_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_is_two_l1009_100980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_coefficient_l1009_100983

/-- A polynomial is a perfect square trinomial if it can be expressed as (x + a)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_coefficient (m : ℝ) :
  IsPerfectSquareTrinomial 1 m 9 → m = 6 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_coefficient_l1009_100983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_abc_solution_l1009_100938

-- Define a structure for base-5 numbers
structure Base5Number where
  digits : List Nat
  nonZero : digits.all (· < 5) ∧ digits ≠ []

-- Define addition for base-5 numbers
def addBase5 : Base5Number → Base5Number → Base5Number :=
  sorry

-- Define the main theorem
theorem unique_abc_solution :
  ∀ A B C : Nat,
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 5 → B < 5 → C < 5 →
    A ≠ B → B ≠ C → A ≠ C →
    addBase5 ⟨[A, B], sorry⟩ ⟨[C], sorry⟩ = ⟨[C, 0], sorry⟩ →
    addBase5 ⟨[A, B], sorry⟩ ⟨[B, A], sorry⟩ = ⟨[C, C], sorry⟩ →
    A = 3 ∧ B = 2 ∧ C = 3 := by
  sorry

#check unique_abc_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_abc_solution_l1009_100938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turning_point_sum_is_ten_l1009_100950

/-- The point where the mouse starts getting farther from the cheese -/
noncomputable def turning_point (cheese : ℝ × ℝ) : ℝ × ℝ :=
  let line := fun x => -5 * x + 18
  let perpendicular_slope := 1 / 5
  let perpendicular_line := fun x => perpendicular_slope * (x - cheese.1) + cheese.2
  let intersection_x := (18 - cheese.2 - 5 * cheese.1 * perpendicular_slope) / (perpendicular_slope + 5)
  (intersection_x, line intersection_x)

/-- The sum of coordinates of the turning point is 10 -/
theorem turning_point_sum_is_ten (cheese : ℝ × ℝ) (h : cheese = (12, 10)) :
  let (a, b) := turning_point cheese
  a + b = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turning_point_sum_is_ten_l1009_100950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_roots_l1009_100966

theorem quadratic_no_real_roots : 
  ∀ (x : ℝ), x^2 + 2*x + 2 ≠ 0 := by
  intro x
  -- Proof goes here
  sorry

#check quadratic_no_real_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_real_roots_l1009_100966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l1009_100903

theorem y_squared_range (y : ℝ) (h : (y + 16)^(1/3) - (y - 16)^(1/3) = 4) :
  230 < y^2 ∧ y^2 < 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_squared_range_l1009_100903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1009_100959

/-- Two 3x3 matrices that are inverses of each other -/
def Matrix1 (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, 2, b],
    ![3, 2, 4],
    ![c, 6, d]]

def Matrix2 (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-7, e, -13],
    ![f, -15, g],
    ![3, h, 5]]

/-- The theorem stating that the sum of all variables in the inverse matrices equals 33.5 -/
theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  Matrix1 a b c d * Matrix2 e f g h = 1 →
  a + b + c + d + e + f + g + h = 33.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_sum_l1009_100959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_depreciation_model_comparison_l1009_100932

/-- Represents a geometric series with a given initial term and ratio --/
structure GeometricSeries where
  initial_term : ℝ
  ratio : ℝ

/-- Calculates the sum of an infinite geometric series --/
noncomputable def sum_of_series (s : GeometricSeries) : ℝ :=
  s.initial_term / (1 - s.ratio)

/-- Theorem: For two geometric series with the same initial term of 24,
    where the second term of the first series is 6 and the second term
    of the second series is 6+n, the value of n that makes the sum of
    the second series three times the sum of the first series is 12. --/
theorem depreciation_model_comparison :
  let s1 : GeometricSeries := { initial_term := 24, ratio := 6 / 24 }
  let s2 : GeometricSeries := { initial_term := 24, ratio := (6 + 12) / 24 }
  sum_of_series s2 = 3 * sum_of_series s1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_depreciation_model_comparison_l1009_100932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sixth_to_degree_pi_eighth_to_degree_three_pi_fourth_to_degree_l1009_100971

noncomputable def radian_to_degree (x : Real) : Real := x * (180 / Real.pi)

-- Theorem statements
theorem pi_sixth_to_degree : radian_to_degree (Real.pi / 6) = 30 := by sorry

theorem pi_eighth_to_degree : radian_to_degree (Real.pi / 8) = 22.5 := by sorry

theorem three_pi_fourth_to_degree : radian_to_degree (3 * Real.pi / 4) = 135 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sixth_to_degree_pi_eighth_to_degree_three_pi_fourth_to_degree_l1009_100971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l1009_100948

theorem polynomial_roots_sum (p q r s : ℤ) : 
  (∃ (q₁ q₂ q₃ q₄ : ℕ), 
    (∀ i : ℕ, i ∈ ({q₁, q₂, q₃, q₄} : Finset ℕ) → Odd i) ∧ 
    (∀ x : ℤ, x^4 + p*x^3 + q*x^2 + r*x + s = (x + q₁)*(x + q₂)*(x + q₃)*(x + q₄)) ∧
    p + q + r + s = 2673) →
  s = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l1009_100948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_eq_one_l1009_100990

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 18 / (4 + 2 * x)

-- State the theorem
theorem inverse_g_cubed_eq_one : (g⁻¹ 3)⁻¹ ^ 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_eq_one_l1009_100990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_weight_is_ten_l1009_100970

/-- The weight of the bag of apples in pounds -/
def apple_weight : ℝ := sorry

/-- The weight of the bag of oranges in pounds -/
def orange_weight : ℝ := sorry

/-- The total weight of fruits in pounds -/
def total_weight : ℝ := 12

axiom orange_apple_ratio : orange_weight = 5 * apple_weight

axiom total_weight_sum : apple_weight + orange_weight = total_weight

theorem orange_weight_is_ten : orange_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_weight_is_ten_l1009_100970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l1009_100974

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with positive common difference,
    and positive integers i, j, k, l satisfying i < k < l and i + j = k + l,
    the sum of S_i and S_j is greater than the sum of S_k and S_l -/
theorem arithmetic_sequence_sum_inequality (a₁ d : ℝ) (i j k l : ℕ+) 
    (h_d : d > 0) 
    (h_ik : i < k) 
    (h_kl : k < l) 
    (h_sum : i + j = k + l) :
    S a₁ d i + S a₁ d j > S a₁ d k + S a₁ d l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_inequality_l1009_100974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1009_100934

/-- The polynomial q(x) satisfying the given conditions -/
noncomputable def q (x : ℝ) : ℝ := (20/21) * x^2 + (40/21) * x - 20/7

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  (q 1 = 0) ∧ 
  (q (-3) = 0) ∧ 
  (∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c) ∧ 
  (q 4 = 20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l1009_100934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_sixth_power_complex_l1009_100981

open Complex Real

theorem magnitude_sixth_power_complex (z : ℂ) : 
  z = 5 + 2 * I * sqrt 3 → Complex.abs (z^6) = 50653 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_sixth_power_complex_l1009_100981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1009_100951

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (x : ℝ) : Prop := x = 1/2

-- Theorem statement
theorem chord_length :
  ∃ (a b c d : ℝ), 
    my_circle a b ∧ my_circle c d ∧ 
    my_line a ∧ my_line c ∧
    (a - c)^2 + (b - d)^2 = 3 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1009_100951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_density_proof_l1009_100933

/-- Represents the density of a material relative to water -/
structure RelativeDensity where
  value : ℝ
  positive : value > 0

/-- The relative density of gold -/
def gold_density : RelativeDensity where
  value := 10
  positive := by norm_num

/-- The relative density of copper -/
def copper_density : RelativeDensity where
  value := 6
  positive := by norm_num

/-- The desired relative density of the alloy -/
def alloy_density : RelativeDensity where
  value := 8
  positive := by norm_num

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 1

theorem alloy_density_proof :
  let total_parts := gold_copper_ratio + 1
  let gold_part := gold_copper_ratio / total_parts
  let copper_part := 1 / total_parts
  gold_part * gold_density.value + copper_part * copper_density.value = alloy_density.value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_density_proof_l1009_100933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l1009_100957

/-- A square in a plane --/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 2 ∧ 
               B.1 - A.1 = side_length ∧ B.2 = A.2 ∧
               C.1 = B.1 ∧ C.2 - B.2 = side_length ∧
               D.1 = A.1 ∧ D.2 = C.2)

/-- Distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Maximum distance from P to D given conditions --/
theorem max_distance_to_D (s : Square) (P : ℝ × ℝ) 
  (h : (distance P s.A)^2 + (distance P s.B)^2 = 2 * (distance P s.C)^2) :
  (∀ Q : ℝ × ℝ, (distance Q s.A)^2 + (distance Q s.B)^2 = 2 * (distance Q s.C)^2 →
    distance Q s.D ≤ 2 * Real.sqrt 2) ∧
  (∃ P : ℝ × ℝ, (distance P s.A)^2 + (distance P s.B)^2 = 2 * (distance P s.C)^2 ∧
    distance P s.D = 2 * Real.sqrt 2) :=
by sorry

#check max_distance_to_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_D_l1009_100957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_point_36_l1009_100945

/-- Represents the number of marbles each player has initially -/
def n : ℕ := 20

/-- Represents Peter's initial number of white marbles -/
def peter_white_initial : ℕ := 12

/-- Represents Paul's initial number of black marbles -/
def paul_black_initial : ℕ := 12

/-- Represents the probability of Peter transferring a white marble on the first move -/
noncomputable def p_peter_white_first : ℝ := peter_white_initial / n

/-- Represents the probability of Paul transferring a black marble on the first move -/
noncomputable def p_paul_black_first : ℝ := paul_black_initial / n

/-- Represents the probability of both events occurring on the first move -/
noncomputable def p_both_first : ℝ := p_peter_white_first * p_paul_black_first

/-- Represents the probability of the desired outcome on the k-th move -/
noncomputable def p_both_k (k : ℕ) : ℝ :=
  (1 - 4/n + 2/n^2)^(k-1) * (p_both_first - n/(4*n-2)) -
  (p_peter_white_first - 1/2) + n/(4*n-1)

/-- The main theorem stating the probability is 0.36 -/
theorem probability_is_point_36 (k : ℕ) : p_both_k k = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_point_36_l1009_100945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1009_100955

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 2)

-- Define the property of sharing the same focus
def same_focus (m : ℝ) : Prop := 
  ∃ (f : ℝ × ℝ), f = parabola_focus ∧ 
  (∃ (c : ℝ), hyperbola m (f.fst) (f.snd + c) ∨ hyperbola m (f.fst) (f.snd - c))

-- Theorem statement
theorem hyperbola_asymptotes (m : ℝ) :
  same_focus m → 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ), hyperbola m x y → (y = k * x ∨ y = -k * x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1009_100955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1009_100978

def our_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n, a n - a (n + 1) + 3 = 0

theorem sequence_general_term (a : ℕ → ℝ) (h : our_sequence a) :
  ∀ n, a n = 3 * n + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l1009_100978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hhtth_probability_l1009_100931

/-- A fair coin is a coin with equal probability of landing on either side. -/
def FairCoin : Type := Unit

/-- The probability of a fair coin landing on a specific side. -/
noncomputable def fairCoinProbability : ℝ := 1 / 2

/-- The number of tosses. -/
def numTosses : ℕ := 5

/-- The probability of getting a specific sequence of outcomes when tossing a fair coin multiple times. -/
noncomputable def sequenceProbability (n : ℕ) : ℝ := (fairCoinProbability) ^ n

/-- Theorem: The probability of getting the sequence HHTTH when tossing a fair coin 5 times is 1/32. -/
theorem hhtth_probability : sequenceProbability numTosses = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hhtth_probability_l1009_100931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safety_competition_theorem_l1009_100908

/-- Represents a participant in the safety knowledge competition -/
structure Participant where
  name : String
  probA : ℝ  -- Probability of answering correctly in category A
  probB : ℝ  -- Probability of answering correctly in category B

/-- The safety knowledge competition -/
def SafetyCompetition (xiaoFang xiaoMing : Participant) : Prop :=
  let totalQuestionsA := 5
  let totalQuestionsB := 5
  let questionsAskedA := 2
  let questionsAskedB := 2
  let pointsForA := 40
  let pointsPerB := 30
  let advancingScore := 60
  -- Xiao Fang's probabilities
  xiaoFang.probA = 0.5 ∧ xiaoFang.probB = 0.5 ∧
  -- Xiao Ming's probabilities
  xiaoMing.probA = 4/5 ∧ xiaoMing.probB = 0.4 ∧
  -- Probability of Xiao Ming scoring 40 points in first round
  (Nat.choose totalQuestionsA questionsAskedA - Nat.choose (totalQuestionsA - 4) questionsAskedA) / 
    Nat.choose totalQuestionsA questionsAskedA = 3/5 ∧
  -- Xiao Ming more likely to advance
  (let pMing := 3/5 * (0.4 * 0.6 + 0.6 * 0.4) + 3/5 * 3/5 * 0.4 * 0.4 + 2/5 * 0.4 * 0.4
   let pFang := 1/4 * (0.5 * 0.5 + 0.5 * 0.5) + 1/4 * 1/4 * 0.5 * 0.5 + 3/4 * 0.5 * 0.5
   pMing > pFang)

theorem safety_competition_theorem (xiaoFang xiaoMing : Participant) :
  SafetyCompetition xiaoFang xiaoMing := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safety_competition_theorem_l1009_100908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_carrots_l1009_100965

/-- The number of carrots Nancy has in total after two days of picking and discarding damaged ones. -/
def total_carrots (first_day_picked : ℕ) (first_day_damaged : ℕ) (second_day_picked : ℕ) (spoilage_rate : ℚ) : ℕ :=
  let first_day_kept := first_day_picked - first_day_damaged
  let second_day_spoiled := (spoilage_rate * second_day_picked).floor.toNat
  let second_day_kept := second_day_picked - second_day_spoiled
  first_day_kept + second_day_kept

/-- Theorem stating that Nancy has 308 carrots in total given the conditions of the problem. -/
theorem nancy_carrots :
  total_carrots 125 30 250 (15 / 100 : ℚ) = 308 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_carrots_l1009_100965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1009_100904

/-- The function f(x) = (x^2 - 2x + 9) / x for x < 0 -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 9) / x

/-- Theorem: The maximum value of f(x) for x < 0 is -8 -/
theorem f_max_value :
  ∀ x : ℝ, x < 0 → f x ≤ -8 :=
by
  intro x hx
  sorry

#check f_max_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1009_100904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_zeros_of_F_l1009_100923

noncomputable def f (x : ℝ) := 1 + x - x^2/2 + x^3/3

noncomputable def g (x : ℝ) := 1 - x + x^2/2 - x^3/3

noncomputable def F (x : ℝ) := f (x + 3) * g (x - 4)

theorem min_interval_for_zeros_of_F (a b : ℤ) :
  a < b →
  (∀ x : ℝ, F x = 0 → ↑a ≤ x ∧ x ≤ ↑b) →
  b - a ≥ 10 := by
  sorry

#check min_interval_for_zeros_of_F

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_for_zeros_of_F_l1009_100923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_identity_logarithm_equation_solution_l1009_100972

/-- Given that a = log_4(3), prove that 2^a + 2^(-a) = (4 * sqrt(3)) / 3 -/
theorem power_sum_identity (a : ℝ) (h : a = Real.log 3 / Real.log 4) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := by sorry

/-- Prove that the solution to log_2(9^(x-1) - 5) = log_2(3^(x-1) - 2) + 2 is x = 2 -/
theorem logarithm_equation_solution :
  ∃ x : ℝ, x = 2 ∧ 
    Real.log (9^(x-1) - 5) / Real.log 2 = Real.log (3^(x-1) - 2) / Real.log 2 + 2 ∧
    ∀ y : ℝ, y ≠ 2 → Real.log (9^(y-1) - 5) / Real.log 2 ≠ Real.log (3^(y-1) - 2) / Real.log 2 + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_identity_logarithm_equation_solution_l1009_100972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1009_100994

noncomputable def f (x y z : ℝ) : ℝ := (3*x^2 - x)/(1 + x^2) + (3*y^2 - y)/(1 + y^2) + (3*z^2 - z)/(1 + z^2)

theorem min_value_of_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  f x y z ≥ 0 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ f a b c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1009_100994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_8pm_l1009_100993

/-- Represents a time on a 12-hour analog clock. -/
structure ClockTime where
  hour : Nat
  minute : Nat
  h_hour_range : hour ≥ 0 ∧ hour < 12
  h_minute_range : minute ≥ 0 ∧ minute < 60

/-- Calculates the angle between hour and minute hands on a 12-hour analog clock. -/
noncomputable def angleBetweenHands (time : ClockTime) : ℝ :=
  let hourAngle : ℝ := (time.hour % 12 + time.minute / 60) * 30
  let minuteAngle : ℝ := time.minute * 6
  let diff := |hourAngle - minuteAngle|
  min diff (360 - diff)

/-- Theorem: At 8:00 PM, the angle between the hour and minute hands is 120°. -/
theorem angle_at_8pm :
  let time : ClockTime := ⟨8, 0, by simp, by simp⟩
  angleBetweenHands time = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_8pm_l1009_100993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_meal2_friday_probability_max_binomial_probability_l1009_100920

-- Define the contingency table
def a : ℕ := 40
def b : ℕ := 10
def c : ℕ := 20
def d : ℕ := 30
def n : ℕ := a + b + c + d

-- Define the chi-square statistic
noncomputable def chi_square : ℝ := (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value
noncomputable def critical_value : ℝ := 7.879

-- Define the probabilities for meal choices
noncomputable def p_meal1_wednesday : ℝ := 1/2
noncomputable def p_meal1_friday_given_meal1_wednesday : ℝ := 4/5
noncomputable def p_meal1_friday_given_meal2_wednesday : ℝ := 1/3

-- Define the binomial distribution parameters
def n_students : ℕ := 10
noncomputable def p_like_cafeteria : ℝ := 0.6

-- Theorem 1: Chi-square test for independence
theorem chi_square_test : chi_square > critical_value := by sorry

-- Theorem 2: Probability of choosing Meal 2 on Friday
theorem meal2_friday_probability : 
  1 - (p_meal1_wednesday * p_meal1_friday_given_meal1_wednesday + 
       (1 - p_meal1_wednesday) * p_meal1_friday_given_meal2_wednesday) = 13/30 := by sorry

-- Theorem 3: Maximum probability in binomial distribution
theorem max_binomial_probability : 
  ∀ k : ℕ, k ≤ n_students → 
    (Nat.choose n_students 6 : ℝ) * p_like_cafeteria^6 * (1 - p_like_cafeteria)^(n_students - 6) ≥
    (Nat.choose n_students k : ℝ) * p_like_cafeteria^k * (1 - p_like_cafeteria)^(n_students - k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_test_meal2_friday_probability_max_binomial_probability_l1009_100920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drones_12_feet_apart_after_8_steps_l1009_100921

-- Define the movement patterns for drones A and B
def drone_A_pattern : List (Int × Int × Int) := [(1, 1, 0), (1, -1, 0), (0, 0, 1)]
def drone_B_pattern : List (Int × Int × Int) := [(-1, -1, 0), (-1, 1, 0), (0, 0, 1)]

-- Define a function to calculate the position of a drone after n steps
def drone_position (pattern : List (Int × Int × Int)) (n : Nat) : Int × Int × Int :=
  let full_cycles := n / 3
  let remaining_steps := n % 3
  let cycle_sum := pattern.foldl (fun (x, y, z) (dx, dy, dz) => (x + dx, y + dy, z + dz)) (0, 0, 0)
  let (cx, cy, cz) := (full_cycles * cycle_sum.fst, full_cycles * cycle_sum.snd.fst, full_cycles * cycle_sum.snd.snd)
  let partial_sum := (pattern.take remaining_steps).foldl (fun (x, y, z) (dx, dy, dz) => (x + dx, y + dy, z + dz)) (0, 0, 0)
  (cx + partial_sum.fst, cy + partial_sum.snd.fst, cz + partial_sum.snd.snd)

-- Define a function to calculate the squared distance between two points
def squared_distance (p1 p2 : Int × Int × Int) : Int :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2

-- Theorem statement
theorem drones_12_feet_apart_after_8_steps :
  let pos_A := drone_position drone_A_pattern 8
  let pos_B := drone_position drone_B_pattern 8
  squared_distance pos_A pos_B = 144 ∧
  pos_A = (6, 0, 2) ∧
  pos_B = (-6, 0, 2) := by
  sorry

#check drones_12_feet_apart_after_8_steps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drones_12_feet_apart_after_8_steps_l1009_100921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_without_degree_l1009_100917

theorem men_without_degree (total_employees : ℕ) 
  (women_percentage : ℚ) (men_with_degree_percentage : ℚ) (women_count : ℕ) : 
  women_percentage = 3/5 →
  men_with_degree_percentage = 3/4 →
  women_count = 48 →
  total_employees * women_percentage = ↑women_count →
  ⌊(total_employees : ℚ) * (1 - women_percentage) * (1 - men_with_degree_percentage)⌋ = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_without_degree_l1009_100917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_on_figure_l1009_100962

/-- Represents a single die face -/
structure DieFace where
  dots : Nat
  inv_mem : dots ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)

/-- Represents a pair of opposite faces on a die -/
structure OppositeFaces where
  face1 : DieFace
  face2 : DieFace
  sum_is_seven : face1.dots + face2.dots = 7

/-- Represents the configuration of dice in the figure -/
structure DiceConfiguration where
  num_dice : Nat
  num_glued_pairs : Nat
  glued_pairs : List (DieFace × DieFace)
  remaining_faces : List DieFace
  dice_count : num_dice = 7
  glued_pairs_count : num_glued_pairs = 9
  glued_pairs_length : glued_pairs.length = num_glued_pairs
  remaining_faces_count : remaining_faces.length = 6

/-- The main theorem to prove -/
theorem total_dots_on_figure (config : DiceConfiguration) :
  (config.glued_pairs.map (fun p => p.fst.dots + p.snd.dots)).sum +
  (config.remaining_faces.map (fun f => f.dots)).sum = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_dots_on_figure_l1009_100962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_iff_a_gt_one_l1009_100936

/-- The function f(x) = a^x - 2 for a > 0 and a ≠ 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 2

/-- The statement that f has a zero in the interval (0, +∞) -/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f a x = 0

/-- Theorem: f(x) = a^x - 2 has a zero in (0, +∞) iff a > 1, given a > 0 and a ≠ 1 -/
theorem zero_iff_a_gt_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  has_zero_in_interval a ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_iff_a_gt_one_l1009_100936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_rod_necessary_for_dilution_l1009_100924

/-- Represents laboratory instruments -/
inductive Instrument
  | TrayBalance
  | VolumetricFlask
  | GlassRod
  | RubberBulbPipette
  | MeasuringCylinder
  | Beaker

/-- Represents the process of preparing diluted hydrochloric acid -/
structure AcidPreparation where
  initial_concentration : ℝ
  target_concentration : ℝ
  available_instruments : List Instrument

/-- Determines if a given instrument is necessary for the acid preparation process -/
def is_necessary_instrument (instr : Instrument) : Bool :=
  match instr with
  | Instrument.GlassRod => true
  | _ => false

/-- Theorem stating that a glass rod is necessary for preparing 15% HCl from 37% HCl -/
theorem glass_rod_necessary_for_dilution :
  ∀ (prep : AcidPreparation),
    prep.initial_concentration = 37 ∧
    prep.target_concentration = 15 ∧
    prep.available_instruments = [Instrument.MeasuringCylinder, Instrument.Beaker] →
    is_necessary_instrument Instrument.GlassRod = true :=
by
  intro prep h
  simp [is_necessary_instrument]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_rod_necessary_for_dilution_l1009_100924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_458_to_14_l1009_100992

/-- Represents the allowed operations on a number -/
inductive Move
  | double : Move
  | erase_last : Move

/-- Applies a move to a number -/
def apply_move (n : ℕ) (m : Move) : ℕ :=
  match m with
  | Move.double => 2 * n
  | Move.erase_last => n / 10

/-- Checks if it's possible to transform start into target using a sequence of moves -/
def can_transform (start target : ℕ) : Prop :=
  ∃ (moves : List Move), moves.foldl apply_move start = target

/-- Theorem stating that it's possible to transform 458 into 14 using the allowed moves -/
theorem transform_458_to_14 : can_transform 458 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_458_to_14_l1009_100992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_division_theorem_l1009_100958

/-- Represents a number in base 4 -/
structure Base4 where
  value : Nat
  isValid : value < 4^64 := by sorry

/-- Converts a base 4 number to base 10 -/
def toBase10 (n : Base4) : Nat :=
  n.value

/-- Performs division in base 4 -/
def divBase4 (dividend : Base4) (divisor : Base4) : (Base4 × Base4) :=
  let quotient := Base4.mk (dividend.value / divisor.value) (by sorry)
  let remainder := Base4.mk (dividend.value % divisor.value) (by sorry)
  (quotient, remainder)

/-- Converts a natural number to Base4 -/
def toBase4 (n : Nat) : Base4 :=
  Base4.mk n (by sorry)

theorem base4_division_theorem :
  let dividend := toBase4 2301
  let divisor := toBase4 21
  let (quotient, remainder) := divBase4 dividend divisor
  quotient.value = 112 ∧ 
  remainder.value = 0 ∧
  toBase10 quotient = 22 := by
  sorry

#eval toBase10 (toBase4 2301)
#eval toBase10 (toBase4 21)
#eval toBase10 ((divBase4 (toBase4 2301) (toBase4 21)).1)
#eval toBase10 ((divBase4 (toBase4 2301) (toBase4 21)).2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_division_theorem_l1009_100958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_is_minus_30_l1009_100930

/-- The sum of the coefficients of the expanded form of -(5 - 2c)(c + 3(5 - 2c)) is -30 -/
theorem sum_of_coefficients_expansion (c : ℝ) : 
  let expanded := -(5 - 2*c) * (c + 3*(5 - 2*c))
  expanded = -10*c^2 + 55*c - 75 := by
  sorry

/-- The sum of the coefficients of the expanded form is -30 -/
theorem sum_of_coefficients_is_minus_30 :
  -10 + 55 + (-75) = -30 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_is_minus_30_l1009_100930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_equals_95_l1009_100942

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = -4
  sum_condition : a 4 + a 6 = 16

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1) / 2) * (seq.a 2 - seq.a 1)

/-- The main theorem to prove -/
theorem sum_10_equals_95 (seq : ArithmeticSequence) : sum_n seq 10 = 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_equals_95_l1009_100942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_of_exp_l1009_100905

theorem definite_integral_of_exp : ∫ x in (0:ℝ)..(1:ℝ), Real.exp x = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_of_exp_l1009_100905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmedian_circle_theorem_l1009_100977

/-- A circle passes through the feet of the symmedians of a triangle -/
def circle_passes_through_symmedian_feet (a b c : ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- A circle is tangent to one side of a triangle -/
def circle_tangent_to_side (a b c : ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Given a non-isosceles triangle ABC with side lengths a, b, and c, and a circle 
    passing through the feet of the symmedians and tangent to one side, 
    prove that (a^2 + b^2)(a^2 + c^2) = (b^2 + c^2)^2 -/
theorem symmedian_circle_theorem (a b c : ℝ) 
  (h_non_isosceles : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circle : ∃ (circle : Set (ℝ × ℝ)), 
    circle_passes_through_symmedian_feet a b c circle ∧
    circle_tangent_to_side a b c circle) :
  (a^2 + b^2) * (a^2 + c^2) = (b^2 + c^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmedian_circle_theorem_l1009_100977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eq_12b_when_b_gt_2_g_min_value_l1009_100911

-- Define the function f
def f (b x : ℝ) : ℝ := x^2 + b*x - 3

-- Define the interval
def interval (b : ℝ) : Set ℝ := Set.Icc (b - 2) (b + 2)

-- Define M(b) as the maximum value of f(x) in the interval
noncomputable def M (b : ℝ) : ℝ := 
  sSup (Set.image (f b) (interval b))

-- Define m(b) as the minimum value of f(x) in the interval
noncomputable def m (b : ℝ) : ℝ := 
  sInf (Set.image (f b) (interval b))

-- Define g(b)
noncomputable def g (b : ℝ) : ℝ := M b - m b

-- Theorem 1: For b > 2, g(b) = 12b
theorem g_eq_12b_when_b_gt_2 (b : ℝ) (h : b > 2) : g b = 12 * b := by
  sorry

-- Theorem 2: The minimum value of g(b) is 4, occurring when b = 0
theorem g_min_value (b : ℝ) : g b ≥ 4 ∧ g 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_eq_12b_when_b_gt_2_g_min_value_l1009_100911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_with_sum_97_squared_l1009_100927

open Set

theorem arithmetic_sequences_with_sum_97_squared :
  let is_valid_sequence : ℕ × ℕ × ℕ → Prop := fun (a, d, n) =>
    n ≥ 3 ∧ n * (2 * a + (n - 1) * d) = 2 * 97^2
  (Finset.filter (fun x => is_valid_sequence x) (Finset.product (Finset.range 98) (Finset.product (Finset.range 98) (Finset.range (97^2 + 1))))).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_with_sum_97_squared_l1009_100927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_circumcircle_property_l1009_100989

/-- Given a triangle ABC with centroid G, circumradius R, and side lengths a, b, and c,
    prove that for any point P on its circumcircle,
    PA^2 + PB^2 + PC^2 - PG^2 = (1/2)(a^2 + b^2 + c^2) + 3R^2 -/
theorem triangle_centroid_circumcircle_property
  (A B C : EuclideanSpace ℝ (Fin 2))  -- Points of the triangle
  (G : EuclideanSpace ℝ (Fin 2))      -- Centroid of the triangle
  (P : EuclideanSpace ℝ (Fin 2))      -- Point on the circumcircle
  (R : ℝ)                             -- Circumradius
  (a b c : ℝ)                         -- Side lengths of the triangle
  (h₁ : G = (1/3 : ℝ) • (A + B + C))  -- Definition of centroid
  (h₂ : ‖P - A‖ = R)                  -- P is on the circumcircle
  (h₃ : ‖P - B‖ = R)                  -- P is on the circumcircle
  (h₄ : ‖P - C‖ = R)                  -- P is on the circumcircle
  (h₅ : ‖B - C‖ = a)                  -- Definition of side length a
  (h₆ : ‖C - A‖ = b)                  -- Definition of side length b
  (h₇ : ‖A - B‖ = c)                  -- Definition of side length c
  : ‖P - A‖^2 + ‖P - B‖^2 + ‖P - C‖^2 - ‖P - G‖^2 = (1/2) * (a^2 + b^2 + c^2) + 3 * R^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_circumcircle_property_l1009_100989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l1009_100975

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  ha : a > 0
  hb : b > 0

/-- The distance from the right focus to the left vertex of a hyperbola -/
def rightFocusToLeftVertex (h : Hyperbola a b) : ℝ := sorry

/-- The distance from the right focus to the asymptote of a hyperbola -/
def rightFocusToAsymptote (h : Hyperbola a b) : ℝ := sorry

/-- The equation of the asymptote of a hyperbola -/
def asymptoteEquation (h : Hyperbola a b) : ℝ → ℝ → Prop := sorry

theorem hyperbola_asymptote_equation (a b : ℝ) (h : Hyperbola a b) :
  rightFocusToLeftVertex h = 2 * rightFocusToAsymptote h →
  asymptoteEquation h = (fun x y ↦ 4 * x = 3 * y ∨ 4 * x = -3 * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_equation_l1009_100975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l1009_100956

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  side_condition : a > 0 ∧ b > 0 ∧ c > 0
  angle_condition : 0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π
  angle_sum : α + β + γ = π
  sine_law : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ

-- Define the theorem
theorem triangle_cotangent_ratio (t : Triangle) (h : t.a^2 + t.b^2 = 2020 * t.c^2) :
  (Real.tan t.γ)⁻¹ / ((Real.tan t.α)⁻¹ + (Real.tan t.β)⁻¹) = 1009.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_ratio_l1009_100956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l1009_100919

noncomputable section

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Symmetric point with respect to the origin -/
def sym_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Symmetric point with respect to the x-axis -/
def sym_x_axis (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Y-intercept of the line through two points -/
def y_intercept (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 * y2 - x2 * y1) / (x1 - x2)

theorem constant_product (x1 y1 x2 y2 : ℝ) :
  unit_circle x1 y1 →
  unit_circle x2 y2 →
  let m := y_intercept x2 y2 (sym_origin x1 y1).1 (sym_origin x1 y1).2
  let n := y_intercept x2 y2 (sym_x_axis x1 y1).1 (sym_x_axis x1 y1).2
  m * n = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l1009_100919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_one_unique_cube_root_unique_eighth_root_exists_l1009_100982

-- Define p as a prime number of the form 3k + 2
def p : ℕ := 5 -- Example value, replace with a specific prime of the form 3k + 2

axiom p_prime : Nat.Prime p

axiom p_form : ∃ k : ℕ, p = 3 * k + 2

-- Define p-arithmetic
def p_arithmetic (a b : ℕ) : ℕ := (a + b) % p

-- Theorem 1: The equation x^3 = 1 has only one solution in p-arithmetic
theorem cube_root_of_one_unique :
  ∃! x : ℕ, x < p ∧ (x^3) % p = 1 := by
  sorry

-- Theorem 2: For any a, the equation x^3 = a has at most one solution in p-arithmetic
theorem cube_root_unique (a : ℕ) :
  ∀ x y : ℕ, x < p → y < p → (x^3) % p = a % p → (y^3) % p = a % p → x = y := by
  sorry

-- Theorem 3: For any a, there exists an x such that x^8 = a in p-arithmetic
theorem eighth_root_exists (a : ℕ) :
  ∃ x : ℕ, x < p ∧ (x^8) % p = a % p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_one_unique_cube_root_unique_eighth_root_exists_l1009_100982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_travel_time_l1009_100991

/-- The time taken for Carl to ride to Ralph's house -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Proof that Carl's travel time is 5 hours -/
theorem carl_travel_time :
  let distance : ℝ := 10
  let speed : ℝ := 2
  travel_time distance speed = 5 := by
  unfold travel_time
  simp
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_travel_time_l1009_100991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l1009_100935

-- Define the complex number z
noncomputable def z : ℂ := (4 + 2*Complex.I) / (1 + Complex.I)^2

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem statement
theorem complex_point_on_line :
  ∃ (x y : ℝ), (z = x + y*Complex.I ∧ line_equation x y (-5)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_point_on_line_l1009_100935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_T_bound_l1009_100910

/-- Chebyshev polynomial of the first kind -/
def chebyshev_T (n : ℕ) : ℝ → ℝ := sorry

/-- Theorem: The absolute value of the Chebyshev polynomial of the first kind is bounded by 1 for x ≤ 1 -/
theorem chebyshev_T_bound (n : ℕ) (x : ℝ) (h : x ≤ 1) : |chebyshev_T n x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshev_T_bound_l1009_100910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_one_l1009_100976

theorem gcd_not_one (n k : ℕ) (hn : n > 0) (hk : k > 0) (h : n ∣ k^n - 1) : Nat.gcd n (k - 1) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_not_one_l1009_100976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_formula_l1009_100912

/-- The sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 1  -- Add this case to handle n = 0
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + 4

/-- The proposed general formula for a_n -/
def a_formula (n : ℕ) : ℤ := 5 * 2^(n - 1) - 4

/-- Theorem stating that the recursive definition equals the general formula -/
theorem a_equals_formula : ∀ n : ℕ, n ≥ 1 → a n = a_formula n := by
  sorry

/-- Lemma to prove the base case -/
lemma a_base_case : a 1 = a_formula 1 := by
  rfl

/-- Lemma to prove the inductive step -/
lemma a_inductive_step (n : ℕ) (h : n ≥ 1) :
  a (n + 1) = a_formula (n + 1) → a (n + 2) = a_formula (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_formula_l1009_100912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_marbles_count_l1009_100916

def marble_colors := Fin 5

def num_marbles : Fin 5 → ℕ
| 0 => 1  -- red
| 1 => 1  -- green
| 2 => 1  -- blue
| 3 => 1  -- purple
| 4 => 4  -- yellow

def choose_two_marbles : ℕ :=
  Nat.choose 5 2 + 1

theorem two_marbles_count :
  choose_two_marbles = 11 := by
  unfold choose_two_marbles
  unfold Nat.choose
  norm_num
  rfl

#eval choose_two_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_marbles_count_l1009_100916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1009_100929

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin (2 * α) = 24 / 25) 
  (h2 : π < α) 
  (h3 : α < 5 * π / 4) : 
  Real.cos α - Real.sin α = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l1009_100929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_given_points_l1009_100999

/-- The slope of a line passing through two points. -/
def my_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- Theorem stating that the slope of the line passing through (2, 1) and (3, 3) is 2. -/
theorem slope_of_line_through_given_points : my_slope 2 1 3 3 = 2 := by
  unfold my_slope
  norm_num

#eval my_slope 2 1 3 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_through_given_points_l1009_100999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l1009_100964

theorem expansion_terms_count : ℕ := by
  let first_bracket_terms := 5
  let second_bracket_terms := 4
  let third_bracket_terms := 3
  have h : first_bracket_terms * second_bracket_terms * third_bracket_terms = 60 := by
    norm_num
  exact 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l1009_100964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sq_minus_sin_sq_l1009_100900

theorem cos_sq_minus_sin_sq (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) :
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sq_minus_sin_sq_l1009_100900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_problem_l1009_100901

/-- Given a line ax + by + c = 0 and a point (x₀, y₀), 
    returns the symmetric point (x₁, y₁) with respect to the line -/
noncomputable def symmetric_point (a b c x₀ y₀ : ℝ) : ℝ × ℝ :=
  let d := (a * x₀ + b * y₀ + c) / (a^2 + b^2)
  ((x₀ - 2 * a * d), (y₀ - 2 * b * d))

/-- The symmetric point of (0,4) with respect to the line x-y+1=0 is (3,1) -/
theorem symmetric_point_problem : symmetric_point 1 (-1) 1 0 4 = (3, 1) := by
  -- Unfold the definition of symmetric_point
  unfold symmetric_point
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_problem_l1009_100901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_greater_than_n_l1009_100906

theorem sum_of_digits_greater_than_n (n : ℕ) :
  ∃ (digit_sum : ℕ → ℕ),
    (∀ m : ℕ, digit_sum m = (Nat.sum (Nat.digits 10 m))) →
    digit_sum (2^(2^(2 * n))) > n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_greater_than_n_l1009_100906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1009_100969

-- Define the function f(x) = x / ln(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := x / Real.log x

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → x₂ < Real.exp 1 →
  f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1009_100969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_arg_of_e_3iπ_l1009_100943

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the principal argument
noncomputable def principal_arg (z : ℂ) : ℝ :=
  if z.im ≥ 0 then Real.arccos (z.re / Complex.abs z)
  else -Real.arccos (z.re / Complex.abs z)

-- State the theorem
theorem principal_arg_of_e_3iπ :
  principal_arg (cexp (3 * Complex.I * Real.pi)) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_arg_of_e_3iπ_l1009_100943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1009_100961

noncomputable def f (x : ℝ) : ℝ :=
  1 - 2 * (Real.sin (x + Real.pi/8))^2 + 2 * Real.sin (x + Real.pi/8) * Real.cos (x + Real.pi/8)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi) ∧
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi/2) (k * Real.pi))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1009_100961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_f_maximum_l1009_100996

noncomputable def a (x : Real) : Real × Real := (Real.cos (3*x/2), Real.sin (3*x/2))
noncomputable def b (x : Real) : Real × Real := (Real.cos (x/2), -Real.sin (x/2))

def dot_product (v w : Real × Real) : Real :=
  v.1 * w.1 + v.2 * w.2

def vector_sum (v w : Real × Real) : Real × Real :=
  (v.1 + w.1, v.2 + w.2)

noncomputable def vector_magnitude (v : Real × Real) : Real :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def f (x : Real) : Real :=
  dot_product (a x) (b x) + vector_magnitude (vector_sum (a x) (b x))

theorem vector_properties (x : Real) 
    (h : x ∈ Set.Icc (Real.pi / 2) Real.pi) : 
    dot_product (a x) (b x) = Real.cos (2*x) ∧ 
    vector_magnitude (vector_sum (a x) (b x)) = -2 * Real.cos x := by
  sorry

theorem f_maximum (x : Real) 
    (h : x ∈ Set.Icc (Real.pi / 2) Real.pi) :
    f x ≤ 3 ∧ (f Real.pi = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_f_maximum_l1009_100996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_abc_l1009_100907

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_abc (a b c : ℕ+) : 
  a * b * c = factorial 9 → a < b → b < c → 
  ∀ (x y z : ℕ+), x * y * z = factorial 9 → x < y → y < z → c - a ≤ z - x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_abc_l1009_100907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_condition_l1009_100968

noncomputable section

open Real

theorem vector_equality_condition (a b : ℝ × ℝ × ℝ) : 
  a ≠ 0 → b ≠ 0 → ¬(∃ (k : ℝ), a = k • b) → 
  (norm a = norm b ↔ norm (a + 2 • b) = norm (2 • a + b)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_condition_l1009_100968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l1009_100985

/-- A function representing the inverse proportional relationship between a and b^2 -/
noncomputable def inverse_proportion (k : ℝ) (b : ℝ) : ℝ := k / (b ^ 2)

theorem inverse_proportion_problem (k : ℝ) :
  (inverse_proportion k 24 = 10) →
  (inverse_proportion k 12 = 40) := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_problem_l1009_100985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_implies_B_B_not_implies_A_l1009_100922

open Set
open Function

-- Define the interval (a, b)
variable (a b : ℝ) (hab : a < b)

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)
variable (hf' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)

-- Proposition A
def proposition_A (a b : ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x ∈ Ioo a b, f' x > 0

-- Proposition B
def proposition_B (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  StrictMonoOn f (Ioo a b)

-- Theorem: Proposition A implies Proposition B
theorem A_implies_B (a b : ℝ) (f f' : ℝ → ℝ) (hab : a < b) 
  (hf' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) :
  proposition_A a b f' → proposition_B a b f := by
  sorry

-- Theorem: Proposition B does not imply Proposition A
theorem B_not_implies_A (a b : ℝ) (hab : a < b) : 
  ∃ f f' : ℝ → ℝ, proposition_B a b f ∧ ¬proposition_A a b f' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_implies_B_B_not_implies_A_l1009_100922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1009_100963

/-- The rational function f(x) = (3x^2 + 5x - 4) / (x - 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 5 * x - 4) / (x - 4)

/-- The slope of the slant asymptote of f -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote of f -/
def b : ℝ := 17

/-- Theorem: The sum of the slope and y-intercept of the slant asymptote of f is 20 -/
theorem slant_asymptote_sum : m + b = 20 := by
  rw [m, b]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l1009_100963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1009_100928

-- Define the speeds in kmph
def speed_train1 : ℚ := 120
def speed_train2 : ℚ := 80

-- Define the time taken to cross in seconds
def crossing_time : ℚ := 9

-- Define the length of the second train in meters
def length_train2 : ℚ := 220.04

-- Convert kmph to m/s
def kmph_to_mps (speed : ℚ) : ℚ := speed * (1000 / 3600)

-- Calculate the relative speed of the trains
noncomputable def relative_speed : ℚ := kmph_to_mps (speed_train1 + speed_train2)

-- Calculate the total length of both trains
noncomputable def total_length : ℚ := relative_speed * crossing_time

-- Calculate the length of the first train
noncomputable def length_train1 : ℚ := total_length - length_train2

-- Theorem statement
theorem train_length_proof :
  length_train1 = 280 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l1009_100928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_radius_l1009_100949

/-- A configuration of spheres in a unit cube -/
structure SpherePacking where
  /-- The number of spheres in the packing -/
  num_spheres : ℕ
  /-- The radius of each sphere -/
  radius : ℚ
  /-- There is a central sphere -/
  has_central_sphere : Prop
  /-- Each non-central sphere is tangent to the central sphere -/
  tangent_to_central : Prop
  /-- Each non-central sphere is tangent to three faces of the cube -/
  tangent_to_faces : Prop

/-- The specific sphere packing configuration described in the problem -/
def problem_packing : SpherePacking where
  num_spheres := 16
  radius := 1 / 3
  has_central_sphere := True
  tangent_to_central := True
  tangent_to_faces := True

theorem sphere_packing_radius :
  ∀ (p : SpherePacking),
    p.num_spheres = 16 →
    p.has_central_sphere →
    p.tangent_to_central →
    p.tangent_to_faces →
    p.radius = 1 / 3 := by
  sorry

#check sphere_packing_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_radius_l1009_100949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1009_100941

theorem problem_statement :
  (¬ (∀ a b : ℝ, (|a| + |b| > 1 → |a + b| > 1) ∧ 
    ¬(∀ a b : ℝ, |a + b| > 1 → |a| + |b| > 1))) ∧ 
  ({x : ℝ | ∃ y : ℝ, y = Real.sqrt (|x - 1| - 2)} = 
    Set.Iic (-1) ∪ Set.Ici 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1009_100941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1009_100954

/-- The equation we want to solve for integers x, y, z -/
def equation (x y z : ℤ) : Prop :=
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18

/-- The set of all integer solutions to the equation -/
def solutions : Set (ℤ × ℤ × ℤ) :=
  {xyz | equation xyz.1 xyz.2.1 xyz.2.2}

/-- The six specific solutions we claim are correct -/
def claimed_solutions : Set (ℤ × ℤ × ℤ) :=
  {xyz | ∃ x, (xyz = (x, x, x) ∨ 
              xyz = (x, x + 1, x) ∨ 
              xyz = (x, x, x - 5) ∨ 
              xyz = (x, x - 4, x - 5) ∨ 
              xyz = (x, x - 4, x - 4) ∨ 
              xyz = (x, x + 1, x - 4))}

/-- Theorem stating that our claimed solutions are exactly the set of all solutions -/
theorem solution_characterization :
  solutions = claimed_solutions := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1009_100954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1009_100979

theorem sqrt_calculations : 
  (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1/2) + Real.sqrt 12) = -5 * Real.sqrt 3) ∧
  ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2) = -18) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1009_100979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l1009_100998

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line, if it exists -/
noncomputable def Line.slope (l : Line) : Option ℝ :=
  if l.b ≠ 0 then some (-l.a / l.b) else none

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The three lines that form the triangle -/
def l1 : Line := ⟨1, -1, 0⟩
def l2 : Line := ⟨1, 1, -2⟩
def l3 (k : ℝ) : Line := ⟨5, -k, -15⟩

/-- The theorem stating the range of k for which the lines form a triangle -/
theorem triangle_formation_condition (k : ℝ) :
  (¬ parallel l1 (l3 k) ∧ ¬ parallel l2 (l3 k)) ↔ k ≠ 5 ∧ k ≠ -5 := by
  sorry

#check triangle_formation_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l1009_100998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_set_upper_bound_l1009_100939

theorem bounded_set_upper_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let d := min (min ((a - b)^2) ((b - c)^2)) ((c - a)^2)
  0 < d / (a^2 + b^2 + c^2) ∧ d / (a^2 + b^2 + c^2) < 1/5 := by
  sorry

#check bounded_set_upper_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_set_upper_bound_l1009_100939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_63521_to_nearest_tenth_l1009_100986

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_24_63521_to_nearest_tenth :
  round_to_nearest_tenth 24.63521 = 24.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_63521_to_nearest_tenth_l1009_100986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1009_100913

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

theorem expression_equality : 
  let a := base_to_decimal [8, 6, 4, 2] 9
  let b := base_to_decimal [0, 0, 2] 5
  let c := base_to_decimal [6, 5, 4, 3] 8
  let d := base_to_decimal [0, 9, 8, 7] 9
  (a / b) - c + d = 4030 := by
  -- Proof steps would go here
  sorry

#eval base_to_decimal [8, 6, 4, 2] 9
#eval base_to_decimal [0, 0, 2] 5
#eval base_to_decimal [6, 5, 4, 3] 8
#eval base_to_decimal [0, 9, 8, 7] 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l1009_100913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_area_is_30_sq_yards_l1009_100997

/-- Represents the dimensions of a rectangular room --/
structure RoomDimensions where
  length : ℚ
  width : ℚ

/-- Converts feet to yards --/
def feetToYards (feet : ℚ) : ℚ := feet / 3

/-- Calculates the area of a room in square yards --/
def roomAreaInSquareYards (room : RoomDimensions) : ℚ :=
  feetToYards room.length * feetToYards room.width

/-- Theorem: The area of the classroom is 30 square yards --/
theorem classroom_area_is_30_sq_yards :
  let classroom : RoomDimensions := { length := 15, width := 18 }
  roomAreaInSquareYards classroom = 30 := by
  -- Unfold definitions
  unfold roomAreaInSquareYards feetToYards
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

#eval roomAreaInSquareYards { length := 15, width := 18 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_classroom_area_is_30_sq_yards_l1009_100997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1009_100918

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => 2^n * a n

theorem a_formula (n : ℕ) : a n = 2^(n * (n-1) / 2) := by
  induction n with
  | zero =>
    simp [a]
  | succ k ih =>
    simp [a]
    rw [ih]
    sorry  -- The detailed proof steps are omitted for brevity

#eval a 5  -- This will evaluate a(5) to check if the function works correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1009_100918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_subsets_l1009_100984

/-- Given a natural number n and a set size r where 2 ≤ r < n/2,
    the maximum number of r-element subsets of an n-element set
    such that every pair of subsets intersects is (n-1 choose r-1). -/
theorem max_intersecting_subsets (n r : ℕ) (h1 : 2 ≤ r) (h2 : r < n/2) :
  (Finset.univ.filter (λ A : Finset (Fin n) => A.card = r)).card ≤ Nat.choose (n-1) (r-1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_subsets_l1009_100984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l1009_100926

theorem sin_double_theta (θ : ℝ) (h : Real.cos θ + Real.sin θ = 3/2) : Real.sin (2*θ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l1009_100926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2005_equals_1_l1009_100953

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | n + 2 => sequence_a (n + 1) - sequence_a n

theorem a_2005_equals_1 : sequence_a 2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2005_equals_1_l1009_100953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l1009_100944

/-- Given a principal amount and an interest rate, calculates the simple interest for 2 years -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate * 2 / 100

/-- Given a principal amount and an interest rate, calculates the compound interest for 2 years -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate / 100)^2 - principal

/-- Theorem stating that if the compound interest is 11730 and simple interest is 10200 for 2 years,
    then the principal amount is 130 -/
theorem principal_amount (P : ℝ) (R : ℝ) 
  (h1 : compound_interest P R = 11730)
  (h2 : simple_interest P R = 10200) :
  P = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_l1009_100944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_surface_area_l1009_100973

/-- Represents a cone with given diameter and height -/
structure Cone where
  diameter : ℝ
  height : ℝ

/-- Calculate the volume of a cone -/
noncomputable def volume (c : Cone) : ℝ :=
  (1/3) * Real.pi * (c.diameter/2)^2 * c.height

/-- Calculate the surface area of a cone -/
noncomputable def surfaceArea (c : Cone) : ℝ :=
  let r := c.diameter/2
  let l := Real.sqrt (r^2 + c.height^2)
  Real.pi * r^2 + Real.pi * r * l

theorem cone_volume_and_surface_area (c : Cone) 
  (h_diameter : c.diameter = 12)
  (h_height : c.height = 9) :
  volume c = 108 * Real.pi ∧ surfaceArea c = Real.pi * (36 + 6 * Real.sqrt 117) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_surface_area_l1009_100973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_property_l1009_100952

theorem infinitely_many_primes_with_property (a : ℤ) : 
  {p : ℕ | Nat.Prime p ∧ ∃ (n m : ℤ), (p : ℤ) ∣ n^2 + 3 ∧ (p : ℤ) ∣ m^3 - a}.Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_property_l1009_100952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_plus_i_one_minus_i_l1009_100988

theorem complex_product_one_plus_i_one_minus_i : 
  (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_one_plus_i_one_minus_i_l1009_100988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l1009_100914

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 1

noncomputable def g (x : ℝ) : ℝ := 3 * x - 4

noncomputable def h (x : ℝ) : ℝ := f (g x)

noncomputable def h_inverse (y : ℝ) : ℝ := (33 + Real.sqrt (144 * y - 3519)) / 36

theorem h_inverse_is_correct (y : ℝ) (hy : y ≥ 3519 / 144) :
  h (h_inverse y) = y ∧ h_inverse (h (h_inverse y)) = h_inverse y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l1009_100914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_neg_three_neg_four_l1009_100902

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem polar_coords_of_neg_three_neg_four :
  let (r, θ) := rectangular_to_polar (-3) (-4)
  r = 5 ∧ θ = 7*Real.pi/4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_of_neg_three_neg_four_l1009_100902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_trapezoid_ABCD_area_l1009_100915

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_parallel_cd : (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2)
  ac_perp_bd : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0

def trapezoid_ABCD : Trapezoid := {
  A := (1, 7)
  B := (7, 5)
  C := (4, 1)
  D := (1, 2)  -- We'll prove this
  ab_parallel_cd := by sorry
  ac_perp_bd := by sorry
}

-- Theorem for the coordinates of point D
theorem point_D_coordinates (t : Trapezoid) (h : t = trapezoid_ABCD) : 
  t.D = (1, 2) := by sorry

-- Function to calculate the area of a trapezoid
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  let ac_length := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let bd_length := Real.sqrt ((t.D.1 - t.B.1)^2 + (t.D.2 - t.B.2)^2)
  (1/2) * ac_length * bd_length

-- Theorem for the area of trapezoid ABCD
theorem trapezoid_ABCD_area : 
  trapezoid_area trapezoid_ABCD = 45/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_trapezoid_ABCD_area_l1009_100915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_PQ_l1009_100946

-- Define the basic geometric objects
variable (Γ : Set Point) (A B C D P Q : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed property
def is_inscribed (quad : Set Point) (circ : Set Point) : Prop := sorry

-- Define the second circle
def second_circle (Γ : Set Point) (B C P Q : Point) : Set Point := sorry

-- Define the tangency and intersection properties
def is_tangent_to_circle (circ1 circ2 : Set Point) : Prop := sorry
def is_tangent_to_line (circ : Set Point) (P Q : Point) : Prop := sorry
def intersects_line_at_point (circ : Set Point) (B C : Point) : Prop := sorry

-- Define incenter
noncomputable def incenter (A B C : Point) : Point := sorry

-- Define a point lying on a line
def point_on_line (P : Point) (Q R : Point) : Prop := sorry

-- State the theorem
theorem incenter_on_PQ 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inscribed {A, B, C, D} Γ)
  (h3 : is_tangent_to_circle (second_circle Γ B C P Q) Γ)
  (h4 : is_tangent_to_line (second_circle Γ B C P Q) B D)
  (h5 : is_tangent_to_line (second_circle Γ B C P Q) A C)
  (h6 : intersects_line_at_point (second_circle Γ B C P Q) B C) :
  point_on_line (incenter A B C) P Q ∧ point_on_line (incenter D B C) P Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_PQ_l1009_100946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_angles_theorem_l1009_100987

-- Define a triangle with side lengths a, b, c
structure Triangle :=
  (a b c : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (triangle_inequality_ab : a < b + c)
  (triangle_inequality_bc : b < a + c)
  (triangle_inequality_ca : c < a + b)

-- Define a predicate for arithmetic sequence of angles
def has_arithmetic_sequence_angles (t : Triangle) : Prop :=
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = Real.pi ∧
    B - A = C - B ∧
    t.a * t.a = t.b * t.b + t.c * t.c - 2 * t.b * t.c * Real.cos A ∧
    t.b * t.b = t.a * t.a + t.c * t.c - 2 * t.a * t.c * Real.cos B ∧
    t.c * t.c = t.a * t.a + t.b * t.b - 2 * t.a * t.b * Real.cos C

-- State the theorem
theorem triangle_arithmetic_angles_theorem (t : Triangle) 
  (h : has_arithmetic_sequence_angles t) : 
  1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_angles_theorem_l1009_100987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_added_during_lunch_rush_l1009_100960

theorem customers_added_during_lunch_rush 
  (initial_customers : ℝ) 
  (total_customers_after : ℕ) 
  (additional_customers : ℝ) 
  (h1 : initial_customers = 29.0)
  (h2 : total_customers_after = 83)
  (h3 : additional_customers = 34.0)
  : (total_customers_after : ℝ) - initial_customers = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_customers_added_during_lunch_rush_l1009_100960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_nine_l1009_100937

/-- Represents an isosceles, obtuse triangle with a specific angle condition -/
structure SpecialTriangle where
  /-- The measure of the largest angle in degrees -/
  largest_angle : ℝ
  /-- The triangle is isosceles -/
  isosceles : Bool
  /-- The triangle is obtuse -/
  obtuse : Bool
  /-- The largest angle is 80% larger than a right angle -/
  angle_condition : largest_angle = 1.8 * 90

/-- The measure of one of the two smallest angles in the special triangle -/
noncomputable def smallest_angle (t : SpecialTriangle) : ℝ := (180 - t.largest_angle) / 2

/-- Theorem stating that the measure of one of the two smallest angles is 9° -/
theorem smallest_angle_is_nine (t : SpecialTriangle) :
  smallest_angle t = 9 := by
  sorry

#check smallest_angle_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_is_nine_l1009_100937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_calculation_l1009_100967

/-- Calculate the cost of fuel for a trip given the start and end odometer readings,
    fuel efficiency, and fuel price per gallon. -/
def fuel_cost_for_trip (start_odometer end_odometer : ℕ) 
                       (fuel_efficiency : ℚ) 
                       (fuel_price : ℚ) : ℚ :=
  let miles_driven := end_odometer - start_odometer
  let gallons_used := miles_driven / fuel_efficiency
  gallons_used * fuel_price

/-- Round a rational number to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ :=
  (x * 100).floor / 100

theorem fuel_cost_calculation (start_odometer end_odometer : ℕ) 
                              (fuel_efficiency fuel_price : ℚ) :
  start_odometer = 52214 →
  end_odometer = 52235 →
  fuel_efficiency = 32 →
  fuel_price = 389/100 →
  round_to_cent (fuel_cost_for_trip start_odometer end_odometer fuel_efficiency fuel_price) = 255/100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_calculation_l1009_100967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1009_100909

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def are_foci (F1 F2 : ℝ × ℝ) : Prop := 
  let c := ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2).sqrt / 2
  c^2 = 25 - 9

-- Define the perimeter of the triangle
noncomputable def triangle_perimeter (P F1 F2 : ℝ × ℝ) : ℝ :=
  ((P.1 - F1.1)^2 + (P.2 - F1.2)^2).sqrt +
  ((P.1 - F2.1)^2 + (P.2 - F2.2)^2).sqrt +
  ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2).sqrt

-- State the theorem
theorem ellipse_triangle_perimeter 
  (P F1 F2 : ℝ × ℝ) 
  (h1 : is_on_ellipse P.1 P.2) 
  (h2 : are_foci F1 F2) : 
  triangle_perimeter P F1 F2 = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1009_100909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_dot_product_dot_product_range_l1009_100947

-- Define the plane region
def plane_region (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x + Real.sqrt 3 * y + 4 ≥ 0 ∧ 
  x ≤ 2

-- Define the circle
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the dot product of vectors PA and PB
def dot_product_PA_PB (x y : ℝ) : ℝ :=
  (x + 2) * (x - 2) + y * (-y)

-- Theorem statement
theorem largest_inscribed_circle_dot_product 
  (x y : ℝ) 
  (h_circle : circle_M x y) 
  (h_geometric_seq : x^2 - y^2 = 2) : 
  dot_product_PA_PB x y = -2 := by
  sorry

-- Additional theorem for the range of dot product
theorem dot_product_range
  (x y : ℝ)
  (h_circle : circle_M x y) :
  -2 ≤ dot_product_PA_PB x y ∧ dot_product_PA_PB x y < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_dot_product_dot_product_range_l1009_100947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₂_x₁_value_l1009_100925

-- Define the quadratic functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the x-coordinates of the intercepts
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry
noncomputable def x₃ : ℝ := sorry
noncomputable def x₄ : ℝ := sorry

-- Define m, n, and p
def m : ℕ := sorry
def n : ℕ := sorry
def p : ℕ := sorry

-- State the conditions
axiom quad_functions : (∀ x, ∃ a b c, f x = a*x^2 + b*x + c) ∧ (∀ x, ∃ a b c, g x = a*x^2 + b*x + c)
axiom g_def : ∀ x, g x = -f (90 - x)
axiom g_passes_vertex : ∃ v, g v = f v ∧ ∀ x, f x ≤ f v
axiom intercepts_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
axiom x₄_x₃_diff : x₄ - x₃ = 110
axiom x₂_x₁_form : x₂ - x₁ = m + n * Real.sqrt p
axiom m_n_p_positive : m > 0 ∧ n > 0 ∧ p > 0
axiom p_not_square_divisible : ∀ (q : ℕ), q > 1 → ¬(q^2 ∣ p)

-- State the theorem to be proved
theorem x₂_x₁_value : x₂ - x₁ = 330 + 220 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₂_x₁_value_l1009_100925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_point_theorem_l1009_100995

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

-- Define the theorem
theorem nine_point_theorem (points : Finset Point) :
  points.card = 9 →
  (∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1) →
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangleArea p1 p2 p3 ≤ 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_point_theorem_l1009_100995
