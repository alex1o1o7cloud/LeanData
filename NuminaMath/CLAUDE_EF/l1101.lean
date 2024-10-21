import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l1101_110144

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the principal amount given the interest difference -/
theorem principal_from_interest_difference (rate : ℝ) (time : ℝ) (difference : ℝ) :
  rate = 10 →
  time = 2 →
  difference = 61 →
  ∃ (principal : ℝ), 
    compoundInterest principal rate time - simpleInterest principal rate time = difference ∧
    principal = 6100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_from_interest_difference_l1101_110144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_numbers_count_l1101_110188

/-- Count of occurrences of a digit in a range of consecutive integers -/
def countDigit (digit : Nat) (start : Nat) (stop : Nat) : Nat :=
  (List.range (stop - start + 1)).map (· + start)
    |> List.filter (fun n => n.repr.any (fun c => c.toString == digit.repr))
    |> List.length

theorem house_numbers_count (digit : Nat) :
  (digit = 9 → countDigit digit 1 100 = 10) ∧
  (digit = 1 → countDigit digit 1 100 = 28) := by
  sorry

#eval countDigit 9 1 100  -- Expected: 10
#eval countDigit 1 1 100  -- Expected: 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_numbers_count_l1101_110188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ex_radii_cubic_root_sum_sqrt_l1101_110126

/-- The ex-radii of the triangle --/
def ex_radii : Fin 3 → ℝ := ![10.5, 12, 14]

/-- The cubic equation coefficients --/
structure CubicCoeffs where
  p : ℤ
  q : ℤ
  r : ℤ

/-- The sides of the triangle are roots of the cubic equation --/
def is_cubic_root (sides : Fin 3 → ℝ) (coeffs : CubicCoeffs) : Prop :=
  ∀ i : Fin 3, (sides i)^3 - coeffs.p * (sides i)^2 + coeffs.q * (sides i) - coeffs.r = 0

/-- Helper function to sum over Fin 3 --/
def sum_fin3 (f : Fin 3 → ℝ) : ℝ :=
  f 0 + f 1 + f 2

/-- The main theorem --/
theorem ex_radii_cubic_root_sum_sqrt (sides : Fin 3 → ℝ) (coeffs : CubicCoeffs) :
  (sum_fin3 ex_radii / 3 = sum_fin3 sides / 3) →
  is_cubic_root sides coeffs →
  Int.floor (Real.sqrt (coeffs.p + coeffs.q + coeffs.r : ℝ)) + 1 = 43 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ex_radii_cubic_root_sum_sqrt_l1101_110126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l1101_110179

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def total_arrangements : ℕ := Nat.choose (num_red_lamps + num_blue_lamps) num_blue_lamps
def total_on_configurations : ℕ := Nat.choose (num_red_lamps + num_blue_lamps) num_lamps_on

def favorable_color_arrangements : ℕ := Nat.choose 5 3
def favorable_on_configurations : ℕ := Nat.choose 5 2

def favorable_outcomes : ℕ := favorable_color_arrangements * favorable_on_configurations

theorem lava_lamp_probability :
  (favorable_outcomes : ℚ) / ((total_arrangements * total_on_configurations) : ℚ) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l1101_110179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_b_l1101_110115

noncomputable def u (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Add case for 0
  | 1 => b
  | n + 2 => -2 / (u b (n + 1) + 1)

theorem u_16_equals_b (b : ℝ) (h : b > 1) : u b 16 = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_16_equals_b_l1101_110115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_is_82_inches_tall_l1101_110195

/-- Sara's height in inches -/
def sara_height : ℝ := sorry

/-- Joe's height in inches -/
def joe_height : ℝ := sorry

/-- The combined height of Sara and Joe is 120 inches -/
axiom combined_height : sara_height + joe_height = 120

/-- Joe's height is 6 inches more than double Sara's height -/
axiom joe_height_relation : joe_height = 2 * sara_height + 6

/-- Theorem: Joe's height is 82 inches -/
theorem joe_is_82_inches_tall : joe_height = 82 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_is_82_inches_tall_l1101_110195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_curvilinear_trapezoid_l1101_110110

open MeasureTheory Interval Set

theorem area_curvilinear_trapezoid
  (f : ℝ → ℝ) (a b : ℝ) (h₁ : ContinuousOn f (Icc a b)) (h₂ : a < b)
  (h₃ : ∀ x ∈ Icc a b, f x ≤ 0) :
  (∫ (x : ℝ) in a..b, -f x) = ∫ (x : ℝ) in a..b, |f x| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_curvilinear_trapezoid_l1101_110110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l1101_110177

/-- Represents the time it takes Clea to travel on an escalator under different conditions -/
structure EscalatorTime where
  stationary : ℝ  -- Time to walk down stationary escalator
  moving : ℝ      -- Time to walk down moving escalator
  riding : ℝ      -- Time to ride without walking

/-- Calculates the time it takes to ride the escalator without walking -/
noncomputable def calculate_riding_time (et : EscalatorTime) : ℝ :=
  (et.stationary * et.moving) / (et.stationary - et.moving)

/-- Theorem stating that given the specific times for walking down the escalator,
    the time to ride without walking is 40 seconds -/
theorem escalator_ride_time (et : EscalatorTime) 
  (h1 : et.stationary = 60)
  (h2 : et.moving = 24) :
  calculate_riding_time et = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l1101_110177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_7982_to_hundredth_l1101_110108

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_2_7982_to_hundredth :
  roundToHundredth 2.7982 = 2.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_7982_to_hundredth_l1101_110108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_tree_height_l1101_110184

noncomputable def tree_heights (h1 : ℝ) : Fin 4 → ℝ
| 0 => h1
| 1 => h1 / 2
| 2 => h1 / 2
| 3 => h1 + 200

theorem average_tree_height (h1 : ℝ) (h1_pos : h1 > 0) (h1_eq : h1 = 1000) :
  (Finset.sum Finset.univ (tree_heights h1)) / 4 = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_tree_height_l1101_110184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l1101_110173

theorem dot_product_of_unit_vectors (a b c d : ℝ × ℝ × ℝ) : 
  ‖a‖ = 1 → ‖b‖ = 1 → ‖c‖ = 1 → ‖d‖ = 1 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  a • b = -1/7 → a • c = -1/7 → b • c = -1/7 → 
  b • d = -1/7 → c • d = -1/7 →
  a • d = -19/21 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_of_unit_vectors_l1101_110173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_with_highest_second_l1101_110152

variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
def prob_order (p₁ p₂ p₃ : ℝ) : Prop := 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ ≤ 1

-- Define the probability of winning two consecutive games for each scenario
def P_A (p₁ p₂ p₃ : ℝ) : ℝ := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B (p₁ p₂ p₃ : ℝ) : ℝ := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C (p₁ p₂ p₃ : ℝ) : ℝ := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem: The probability is maximized when playing against C (highest probability) in the second game
theorem max_prob_with_highest_second (h : prob_order p₁ p₂ p₃) : 
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_with_highest_second_l1101_110152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_count_exceeds_even_sum_count_l1101_110120

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define a function to check if a number has odd sum of digits
def hasOddDigitSum (n : ℕ) : Bool :=
  sumOfDigits n % 2 = 1

-- Define the set of numbers from 1 to 1,000,000
def numberSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 1000000}

-- Define the set of numbers with odd digit sums
def oddSumSet : Set ℕ := {n ∈ numberSet | hasOddDigitSum n = true}

-- Define the set of numbers with even digit sums
def evenSumSet : Set ℕ := {n ∈ numberSet | hasOddDigitSum n = false}

-- Theorem statement
theorem odd_sum_count_exceeds_even_sum_count : 
  Finset.card (Finset.filter (λ n => hasOddDigitSum n) (Finset.range 1000000)) = 
  Finset.card (Finset.filter (λ n => ¬hasOddDigitSum n) (Finset.range 1000000)) + 1 :=
by sorry

#eval Finset.card (Finset.filter (λ n => hasOddDigitSum n) (Finset.range 1000000))
#eval Finset.card (Finset.filter (λ n => ¬hasOddDigitSum n) (Finset.range 1000000))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_count_exceeds_even_sum_count_l1101_110120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_duration_l1101_110151

noncomputable section

/-- The duration (in years) that repaired shoes will last -/
def repaired_duration : ℝ := 14.50 / (16.00 / 1.10344827586206897)

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 14.50

/-- The cost of new shoes -/
def new_shoes_cost : ℝ := 32.00

/-- The duration (in years) that new shoes will last -/
def new_shoes_duration : ℝ := 2

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 10.344827586206897

/-- The repaired shoes last approximately 1 year -/
theorem repaired_shoes_duration :
  ∃ ε > 0, |repaired_duration - 1| < ε :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repaired_shoes_duration_l1101_110151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_is_four_l1101_110100

/-- Represents a cricket player's performance -/
structure CricketPlayer where
  initialInnings : ℕ
  initialAverage : ℚ
  nextInningsRuns : ℕ

/-- Calculates the increase in average after the next innings -/
noncomputable def averageIncrease (player : CricketPlayer) : ℚ :=
  let totalInitialRuns := player.initialAverage * player.initialInnings
  let newTotalRuns := totalInitialRuns + player.nextInningsRuns
  let newAverage := newTotalRuns / (player.initialInnings + 1)
  newAverage - player.initialAverage

/-- Theorem: The player's average increases by 4 runs -/
theorem average_increase_is_four (player : CricketPlayer) 
  (h1 : player.initialInnings = 10)
  (h2 : player.initialAverage = 15)
  (h3 : player.nextInningsRuns = 59) :
  averageIncrease player = 4 := by
  sorry

#check average_increase_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_is_four_l1101_110100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_l1101_110189

theorem smallest_integer_satisfying_inequality : 
  ∀ x : ℕ, x > 0 → x^2 > 50 → x ≥ 8 :=
by sorry

theorem eight_satisfies_inequality : 
  8^2 > 50 :=
by sorry

theorem eight_is_smallest : 
  ∀ x : ℕ, x > 0 → (x^2 > 50 ↔ x ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_l1101_110189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1101_110118

/-- Parabola and line intersection -/
structure Intersection where
  k : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  parabola_eq : y₁^2 = 2*x₁ ∧ y₂^2 = 2*x₂
  line_eq : y₁ = k*(x₁-2) ∧ y₂ = k*(x₂-2)
  distinct : (x₁, y₁) ≠ (x₂, y₂)

/-- Theorem about perpendicularity and area -/
theorem intersection_properties (i : Intersection) :
  (i.x₁ * i.x₂ + i.y₁ * i.y₂ = 0) ∧
  (i.k = Real.sqrt 2 → (1/2 : ℝ) * Real.sqrt (i.x₁^2 + i.y₁^2) * Real.sqrt (i.x₂^2 + i.y₂^2) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1101_110118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_200_l1101_110125

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

/-- Collinearity condition for three points -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B.1 = t * A.1 + (1 - t) * C.1 ∧ B.2 = t * A.2 + (1 - t) * C.2

theorem arithmetic_sequence_sum_200
  (a : ℕ → ℝ)
  (O A B C : ℝ × ℝ)
  (h_arith : arithmetic_sequence a)
  (h_collinear : collinear A B C)
  (h_vector : B = (a 1 • A) + (a 200 • C)) :
  S a 200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_200_l1101_110125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1101_110112

/-- Given vectors a, b, and c in ℝ², prove that if λa + b is collinear with c, then λ = 2. -/
theorem collinear_vectors_lambda (a b c : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (2, 3)) (h_c : c = (-4, -7)) :
  (∃ l : ℝ, ∃ k : ℝ, k ≠ 0 ∧ (l • a + b) = k • c) → (∃ l : ℝ, l • a + b = 2 • a + b ∧ l = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1101_110112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1101_110182

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * ω * x) - 2 * (Real.sin (ω * x))^2

-- Define the theorem
theorem triangle_side_length 
  (ω : ℝ) 
  (A B C : ℝ) 
  (h1 : ∀ x, f ω (x + 3 * Real.pi) = f ω x)  -- Minimum positive period is 3π
  (h2 : f ω C = 1)  -- f(C) = 1
  (h3 : 0 < A ∧ A < Real.pi)  -- A is in (0, π)
  (h4 : 0 < B ∧ B < Real.pi)  -- B is in (0, π)
  (h5 : 0 < C ∧ C < Real.pi)  -- C is in (0, π)
  (h6 : A + B + C = Real.pi)  -- Sum of angles in a triangle
  (h7 : 2 = 2 * Real.sin A * Real.sin C / Real.sin B)  -- AB = 2 (using sine law)
  (h8 : 2 * (Real.sin B)^2 = Real.cos B + Real.cos (A - C))  -- Given condition
  : 2 * Real.sin A = Real.sqrt 5 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1101_110182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l1101_110186

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab : ℝ
  cd : ℝ

/-- Calculates the length of BC in a trapezoid with given properties -/
noncomputable def calculate_bc (t : Trapezoid) : ℝ :=
  20 - 0.5 * (Real.sqrt 69 + Real.sqrt 189)

/-- Theorem stating that for a trapezoid with the given properties, 
    the length of BC is equal to the calculated value -/
theorem trapezoid_bc_length (t : Trapezoid) 
    (h1 : t.area = 200) 
    (h2 : t.altitude = 10) 
    (h3 : t.ab = 13) 
    (h4 : t.cd = 17) : 
  calculate_bc t = 20 - 0.5 * (Real.sqrt 69 + Real.sqrt 189) := by
  sorry

#check trapezoid_bc_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l1101_110186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1101_110198

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧  -- Acute triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  1 + Real.tan B / Real.tan A = 2 * c / (Real.sqrt 3 * a) ∧  -- Given relation
  a = 2 →  -- Given side length
  -- Conclusions
  B = π/6 ∧
  Real.sqrt 3 / 2 < (1/2 * a * c * Real.sin B) ∧ 
  (1/2 * a * c * Real.sin B) < 2 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1101_110198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_equals_two_min_value_f_xgx_plus_t_l1101_110183

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) := x^2 - a*x
def g (x : ℝ) := Real.log x

-- Define the conditions
axiom a_nonzero : ∃ a : ℝ, a ≠ 0

axiom intersection_M (a : ℝ) : 
  ∃ M : ℝ, M ≠ 0 ∧ f a M = 0

axiom intersection_N : 
  ∃ N : ℝ, g (N - 1) = 0

axiom parallel_tangents (a : ℝ) :
  ∃ M N : ℝ, M ≠ 0 ∧ g (N - 1) = 0 ∧ 
  (2 * M - a) = 1 / (N - 1)

-- Theorem for part (I)
theorem f_two_equals_two (a : ℝ) (h1 : ∃ a : ℝ, a ≠ 0) 
  (h2 : ∃ M N : ℝ, M ≠ 0 ∧ g (N - 1) = 0 ∧ (2 * M - a) = 1 / (N - 1)) :
  f a 2 = 2 := by sorry

-- Theorem for part (II)
theorem min_value_f_xgx_plus_t (a : ℝ) (t : ℝ) 
  (h1 : ∃ a : ℝ, a ≠ 0)
  (h2 : ∃ M N : ℝ, M ≠ 0 ∧ g (N - 1) = 0 ∧ (2 * M - a) = 1 / (N - 1))
  (h3 : ∃ x : ℝ, x ∈ Set.Icc 1 (Real.exp 1) ∧
    ∀ y ∈ Set.Icc 1 (Real.exp 1), 
      f a (y * g y + t) ≥ f a (x * g x + t)) :
  ∃ x : ℝ, x ∈ Set.Icc 1 (Real.exp 1) ∧ f a (x * g x + t) = -1/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_equals_two_min_value_f_xgx_plus_t_l1101_110183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_complex_l1101_110139

theorem isosceles_triangle_complex (ω : ℂ) (h_norm : Complex.abs ω = 3) :
  ∃ l : ℝ, l > 1 ∧ Complex.abs (ω - ω^2) = Complex.abs (ω - l • ω) ∧ l = 1 - 4 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_complex_l1101_110139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_final_average_l1101_110104

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  averageIncrease : ℚ
  lastInningScore : ℕ

/-- Calculates the average score of a batsman after their latest inning -/
noncomputable def finalAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore : ℚ) / b.innings

/-- Theorem: Given the conditions, the batsman's final average is 49 -/
theorem batsman_final_average (b : Batsman) 
  (h1 : b.innings = 21)
  (h2 : b.lastInningScore = 89)
  (h3 : b.averageIncrease = 2)
  : finalAverage b = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_final_average_l1101_110104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110123

noncomputable def f (x m : ℝ) := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) + m

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x m : ℝ), f (x + T) m = f x m) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x m : ℝ), f (x + T') m = f x m) → T' ≥ T)) ∧
  -- If max value in [0, π/2] is 6, then min value in [0, π/2] is 3
  (∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x m = 6) →
    (∃ y : ℝ, y ∈ Set.Icc 0 (Real.pi / 2) ∧
      f y m = 3 ∧ ∀ z ∈ Set.Icc 0 (Real.pi / 2), f z m ≥ 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_composite_for_all_n_l1101_110146

theorem infinitely_many_k_composite_for_all_n : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ (k : ℕ), k ∈ S → ∀ (n : ℕ), ∃ (d : ℕ), d ≠ 1 ∧ d ∣ (k * 2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_composite_for_all_n_l1101_110146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2016_equals_negative_four_l1101_110196

def sequence_b : ℕ → ℤ
  | 0 => 1  -- We define b₀ as 1 to handle the zero case
  | 1 => 1
  | 2 => 5
  | (n + 3) => sequence_b (n + 2) - sequence_b (n + 1)

theorem b_2016_equals_negative_four : sequence_b 2016 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2016_equals_negative_four_l1101_110196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l1101_110171

/-- The length of the train in meters -/
def train_length : ℝ := 50

/-- The time taken for the train to cross the electric pole in seconds -/
def crossing_time : ℝ := 0.49996000319974404

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / crossing_time

/-- Theorem stating that the train's speed is approximately 100.008 m/s -/
theorem train_speed_approx : 
  |train_speed - 100.008| < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l1101_110171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_food_calculation_l1101_110131

/-- Given a farm with sheep and horses, calculate the total amount of horse food needed per day. -/
def farm_horse_food_needed (num_sheep : ℕ) (sheep_to_horse_ratio : ℚ) (food_per_horse : ℕ) : ℕ :=
  let num_horses := (num_sheep : ℚ) / sheep_to_horse_ratio
  (num_horses.ceil.toNat * food_per_horse)

/-- Theorem: The farm needs 12,880 ounces of horse food per day. -/
theorem farm_food_calculation :
  farm_horse_food_needed 32 (4/7) 230 = 12880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_food_calculation_l1101_110131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1101_110114

/-- Sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℝ := 3
  let r : ℝ := -2
  let n : ℕ := 10
  geometricSum a r n = -1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1101_110114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_turns_in_triangular_city_l1101_110160

/-- A triangular city graph -/
structure TriangularCity where
  vertices : Finset (ℕ × ℕ)
  edges : Finset ((ℕ × ℕ) × (ℕ × ℕ))
  faces : Finset (Finset (ℕ × ℕ))

/-- A path in the city -/
def CityPath (city : TriangularCity) := List (ℕ × ℕ)

/-- Predicate to check if a path is Hamiltonian -/
def IsHamiltonian (city : TriangularCity) (path : CityPath city) : Prop :=
  path.Nodup ∧ path.toFinset = city.vertices

/-- Count the number of 120° turns in a path -/
def CountTurns (city : TriangularCity) (path : CityPath city) : ℕ := sorry

/-- Main theorem -/
theorem min_turns_in_triangular_city (city : TriangularCity) 
  (h1 : city.vertices.card = 15)
  (h2 : city.faces.card = 16)
  (path : CityPath city)
  (h3 : IsHamiltonian city path) :
  CountTurns city path ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_turns_in_triangular_city_l1101_110160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l1101_110150

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_number_not_in_range_of_g 
  (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h17 : g p q r s 17 = 17) 
  (h89 : g p q r s 89 = 89)
  (hinv : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∃! y, (∀ x, g p q r s x ≠ y) ∧ y = 53 :=
by
  sorry

#check unique_number_not_in_range_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_of_g_l1101_110150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l1101_110166

/-- Given an angle α where sin α < 0 and tan α > 0, prove that the terminal side of α is in the third quadrant. -/
theorem terminal_side_in_third_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  α ∈ Set.Ioo (Real.pi) (3 * Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l1101_110166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110157

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos (2 * x)

-- State the theorem
theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S)) ∧
  (∀ y ∈ Set.range f ∩ Set.Icc (-1) 2,
    ∃ x ∈ Set.Icc 0 (2 * Real.pi / 3), f x = y) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), -1 ≤ f x ∧ f x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1101_110132

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

-- State the theorem
theorem f_has_two_zeros : 
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l1101_110132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_exists_hyperbola_with_ecc_2_l1101_110197

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: If the eccentricity of a hyperbola is 2, then the line y = 2x has no common points with the hyperbola -/
theorem no_common_points (h : Hyperbola) (h_ecc : eccentricity h = 2) : 
  ∀ x y : ℝ, y = 2*x → x^2/h.a^2 - y^2/h.b^2 ≠ 1 := by
  sorry

/-- Theorem: There exists a hyperbola with eccentricity 2 -/
theorem exists_hyperbola_with_ecc_2 : 
  ∃ h : Hyperbola, eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_exists_hyperbola_with_ecc_2_l1101_110197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_purchased_theorem_l1101_110194

def mystery_price : ℚ := 4
def scifi_price : ℚ := 6
def romance_price : ℚ := 5
def scifi_discount : ℚ := 4/5
def total_spent : ℚ := 90

def cost_function (x : ℚ) : ℚ :=
  (2/3 * x * mystery_price) + (x * scifi_price * scifi_discount) + (x * romance_price)

theorem books_purchased_theorem :
  ∃ (x : ℚ), (cost_function x = total_spent) ∧ 
  (3 * (Int.floor x).toNat : ℕ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_purchased_theorem_l1101_110194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1101_110155

/-- The number of solutions to the equation |x| + √(t - x²) = √2 for t > 0 -/
noncomputable def num_solutions (t : ℝ) : ℕ :=
  if 0 < t ∧ t < 1 then 0
  else if t = 1 then 2
  else if 1 < t ∧ t < 2 then 4
  else if t = 2 then 3
  else 0

theorem equation_solutions (t : ℝ) (h : t > 0) :
  let f := fun x : ℝ => |x| + Real.sqrt (t - x^2)
  (∃ x, f x = Real.sqrt 2) ↔ num_solutions t ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1101_110155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_waste_percentage_l1101_110133

/-- Represents the percentage of copper in different states of ore processing -/
structure OreComposition where
  extracted : ℚ
  enriched : ℚ
  waste : ℚ

/-- Represents the distribution of ore after processing -/
structure OreDistribution where
  waste_ratio : ℚ

/-- Calculates the percentage of copper in waste given ore composition and distribution -/
noncomputable def copper_in_waste (comp : OreComposition) (dist : OreDistribution) : ℚ :=
  (comp.extracted - (1 - dist.waste_ratio) * comp.enriched) / dist.waste_ratio

/-- Theorem stating that given the problem conditions, the copper percentage in waste is 5% -/
theorem copper_waste_percentage (comp : OreComposition) (dist : OreDistribution) :
  comp.extracted = 21 →
  comp.enriched = 45 →
  dist.waste_ratio = 3/5 →
  copper_in_waste comp dist = 5 := by
  sorry

-- Example usage (commented out as it's not computable)
-- #eval copper_in_waste { extracted := 21, enriched := 45, waste := 0 } { waste_ratio := 3/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_copper_waste_percentage_l1101_110133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_diff_identity_l1101_110161

theorem cos_diff_identity (a b : ℝ) : 
  Real.cos (a + b) - Real.cos (a - b) = -2 * Real.sin a * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_diff_identity_l1101_110161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l1101_110121

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_equation_solution (a b : ℝ) :
  (∀ n : ℕ+, a * floor (b * ↑n) = b * floor (a * ↑n)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m k : ℤ, a = ↑m ∧ b = ↑k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l1101_110121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutation_sum_l1101_110102

def is_valid_digit (d : Nat) : Prop := 0 < d ∧ d ≤ 9

def distinct_digits (x y z t : Nat) : Prop :=
  is_valid_digit x ∧ is_valid_digit y ∧ is_valid_digit z ∧ is_valid_digit t ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t

def four_digit_number (x y z t : Nat) : Nat := 1000 * x + 100 * y + 10 * z + t

def sum_of_permutations (x y z t : Nat) : Nat :=
  6666 * (x + y + z + t)

theorem four_digit_permutation_sum (x y z t : Nat) :
  distinct_digits x y z t →
  sum_of_permutations x y z t = 10 * four_digit_number x x x x →
  x = 9 ∧ ({y, z, t} : Finset Nat) = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_permutation_sum_l1101_110102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l1101_110137

/-- The number of significant digits in a real number -/
def significantDigits (x : ℝ) : ℕ :=
  sorry -- Implementation not provided for brevity

/-- A square with a given area -/
structure Square where
  area : ℝ
  area_positive : area > 0

/-- The side length of a square -/
noncomputable def Square.side (s : Square) : ℝ :=
  Real.sqrt s.area

/-- Approximation relation for reals -/
def approx (x y : ℝ) : Prop :=
  sorry -- Implementation not provided for brevity

/-- Precision of a real number measurement -/
def precision (x : ℝ) : ℝ :=
  sorry -- Implementation not provided for brevity

theorem square_side_significant_digits (s : Square) 
  (h : approx s.area 2.4896) -- area is approximately 2.4896
  (h_precision : precision s.area ≤ 0.00005) -- precision to nearest ten-thousandth
  : significantDigits s.side = 5 :=
sorry

#check square_side_significant_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l1101_110137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pollution_index_l1101_110164

-- Define the function f(x, a)
noncomputable def f (x a : ℝ) : ℝ := |x / (x^2 + 1) + 1/3 - a| + 2*a

-- Define the function M(a)
noncomputable def M (a : ℝ) : ℝ :=
  if 0 ≤ a ∧ a < 7/12 then
    a + 5/6
  else if 7/12 ≤ a ∧ a ≤ 3/4 then
    3*a - 1/3
  else
    0  -- This case should never occur given our constraints

-- State the theorem
theorem max_pollution_index :
  ∀ a : ℝ, 0 ≤ a → a ≤ 3/4 →
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 24 ∧
  f x a ≤ M a ∧
  M a ≤ 23/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pollution_index_l1101_110164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_value_l1101_110129

-- Define the triangle XYZ
noncomputable def triangle_XYZ (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  xy = 13 ∧ xz = 8 ∧ yz = 15

-- Define the diameter of the inscribed circle
noncomputable def inscribed_circle_diameter (X Y Z : ℝ × ℝ) : ℝ :=
  let s := (13 + 8 + 15) / 2
  let area := Real.sqrt (s * (s - 13) * (s - 8) * (s - 15))
  2 * area / s

-- Theorem statement
theorem inscribed_circle_diameter_value {X Y Z : ℝ × ℝ} 
  (h : triangle_XYZ X Y Z) : 
  inscribed_circle_diameter X Y Z = 10 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_value_l1101_110129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_range_l1101_110111

-- Define the hyperbola
def hyperbola (x y m : ℝ) : Prop := x^2 - y^2/m^2 = 1

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + (y+2)^2 = 1

-- Define the asymptotes of the hyperbola
def asymptotes (x y m : ℝ) : Prop := y = m*x ∨ y = -m*x

-- Define the condition that asymptotes do not intersect the circle
def no_intersection (m : ℝ) : Prop := 
  ∀ x y, asymptotes x y m → ¬circle_equation x y

-- Define the focal length of the hyperbola
noncomputable def focal_length (m : ℝ) : ℝ := 2 * Real.sqrt (1 + m^2)

-- State the theorem
theorem hyperbola_focal_length_range (m : ℝ) :
  m > 0 → no_intersection m → 2 < focal_length m ∧ focal_length m < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_range_l1101_110111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_solutions_l1101_110180

theorem triplet_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x^2 + y^2 = 3 * 2016^z + 77} =
  {(8, 4, 0), (4, 8, 0), (77, 14, 1), (14, 77, 1), (70, 35, 1), (35, 70, 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_solutions_l1101_110180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_equals_15_l1101_110159

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angled : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the length of a side
noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the angle at Q
noncomputable def angle_at_Q (P Q R : ℝ × ℝ) : ℝ :=
  Real.arccos ((length Q P)^2 + (length Q R)^2 - (length P R)^2) / (2 * length Q P * length Q R)

theorem pq_equals_15 (P Q R : ℝ × ℝ) 
  (triangle : Triangle P Q R) 
  (pr_length : length P R = 15)
  (angle_q : angle_at_Q P Q R = π/4) : 
  length P Q = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_equals_15_l1101_110159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_a_capacity_percentage_of_tank_b_l1101_110128

-- Define the properties of the tanks
def tank_a_height : ℝ := 10
def tank_a_circumference : ℝ := 6
def tank_b_height : ℝ := 6
def tank_b_circumference : ℝ := 10

-- Define the volume of a cylinder
noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

-- Theorem statement
theorem tank_a_capacity_percentage_of_tank_b :
  (cylinder_volume tank_a_height tank_a_circumference) /
  (cylinder_volume tank_b_height tank_b_circumference) = 0.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_a_capacity_percentage_of_tank_b_l1101_110128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_equation_solution_l1101_110148

-- Define new operations
noncomputable def new_add (a b : ℝ) := a * b
noncomputable def new_sub (a b : ℝ) := a + b
noncomputable def new_mul (a b : ℝ) := a / b
noncomputable def new_div (a b : ℝ) := a - b

-- Theorem statement
theorem special_equation_solution :
  ∃ x : ℝ, new_sub 6 (new_add 9 (new_div (new_mul x 3) 25)) = 5 ∧ x = 8 := by
  -- Proof goes here
  sorry

#check special_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_equation_solution_l1101_110148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_certain_l1101_110124

-- Define the events as propositions
def event_A : Prop := ∃ (outcome : Bool × Bool), outcome.1 = true ∨ outcome.2 = true
def event_B : Prop := ∃ (days : Fin 365 → Bool), ∀ (d : Fin 365), days d = false
def event_C : Prop := ∀ (water : ℝ × ℝ × ℝ) (t : ℝ), water.2.2 < water.2.2 + t
def event_D : Prop := ∃ (ball : Nat), ball = 0 ∧ ball > 0

-- Define what it means for an event to be certain
def is_certain (e : Prop) : Prop := ∀ (outcome : Prop), outcome → e

-- Theorem statement
theorem only_C_is_certain :
  is_certain event_C ∧ 
  ¬is_certain event_A ∧ 
  ¬is_certain event_B ∧ 
  ¬is_certain event_D :=
by
  sorry -- Skipping the proof as requested


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_certain_l1101_110124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110168

noncomputable def f (x : ℝ) := Real.sin (2 * x + 3 * Real.pi / 2)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ Real.pi) ∧
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_result_l1101_110175

def v : ℝ × ℝ × ℝ := (3, -1, 4)
def w : ℝ × ℝ × ℝ := (-2, 3, 1)

def scalar_mult (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (s * v.1, s * v.2.1, s * v.2.2)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_result :
  cross_product v (scalar_mult 2 w) = (-26, -22, 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_result_l1101_110175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_distance_l1101_110167

/-- The distance swum by an athlete between two bridges --/
noncomputable def distance_swum (l k : ℝ) : ℝ :=
  l * (3 * k + 1) / (k + 3)

/-- The theorem stating the distance swum by the athlete --/
theorem athlete_distance (l k : ℝ) (hl : l > 0) (hk : k > 1) :
  let d := distance_swum l k
  ∃ (y : ℝ), 0 < y ∧ y < l ∧
    (l - y) / (k * y) = (l + y) / (2 * k * y) + y / ((k - 1) * y) ∧
    d = (2 * l + y) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_distance_l1101_110167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1101_110134

/-- The focus of the parabola y^2 = 8x has coordinates (2,0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (2, 0) ∧ 
    ∀ (p : ℝ × ℝ), p ∈ parabola → dist p f = dist p (0, 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1101_110134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1101_110127

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    a line passing through the left focus F₁ touches the circle x² + y² = a²
    at point P and intersects the right branch of C at Q.
    If F₁P = PQ, then the equation of the asymptote of C is y = ±2x. -/
theorem hyperbola_asymptote (a b c : ℝ) (ha : a > 0) (hb : b > 0) 
  (C : Set (ℝ × ℝ)) (F₁ P Q : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ C ↔ x^2/a^2 - y^2/b^2 = 1) →
  (P.1^2 + P.2^2 = a^2) →
  (Q = F₁ + 2 • (P - F₁)) →
  (∃ s : ℝ, P = F₁ + s • (Q - F₁) ∧ 0 < s ∧ s < 1) →
  Q ∈ C →
  F₁ = (-c, 0) →
  b = 2*a →
  (∀ x y, (x, y) ∈ C → y = 2*x ∨ y = -2*x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1101_110127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_fx_x_l1101_110130

theorem gcd_fx_x (x : ℤ) (h : 7200 ∣ x) :
  Nat.gcd (((5*x+6)*(8*x+3)*(11*x+9)*(4*x+12)).natAbs) x.natAbs = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_fx_x_l1101_110130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l1101_110178

-- Define the series A
noncomputable def A : ℝ := ∑' n, if (n % 4 = 2 ∨ n % 4 = 3) then ((-1)^((n - 2) / 2)) / n^2 else 0

-- Define the series B
noncomputable def B : ℝ := ∑' n, if (n % 4 = 0) then ((-1)^(n / 4 - 1)) / n^2 else 0

-- Theorem statement
theorem A_div_B_eq_17 : A / B = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l1101_110178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1101_110199

-- Define the slope k
def k : Set ℝ := Set.Ioo 0 (Real.sqrt 3)

-- Define the inclination angle α
noncomputable def α (x : ℝ) : ℝ := Real.arctan x

-- Theorem statement
theorem inclination_angle_range :
  ∀ x ∈ k, α x ∈ Set.Ioo 0 (Real.pi / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1101_110199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_acute_angle_probability_l1101_110136

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of the curve
noncomputable def f_derivative (x : ℝ) : ℝ := 2*x + 1

-- Define the condition for acute angle of inclination
def is_acute_angle (a : ℝ) : Prop := f_derivative a > 0

-- Define the probability calculation
noncomputable def probability_acute_angle : ℝ :=
  (1 - (-1/2)) / (1 - (-1))

-- The main theorem
theorem tangent_acute_angle_probability :
  probability_acute_angle = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_acute_angle_probability_l1101_110136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1101_110122

/-- Represents an ellipse with equation y²/4 + x² = 1 -/
def Ellipse : Set (ℝ × ℝ) := {p | p.2^2/4 + p.1^2 = 1}

/-- Represents a line with equation y = kx + √3 -/
def Line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + Real.sqrt 3}

/-- The intersection points of the ellipse and the line -/
def Intersection (k : ℝ) : Set (ℝ × ℝ) := Ellipse ∩ Line k

/-- Checks if a circle with diameter AB passes through the origin -/
def CircleThroughOrigin (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem ellipse_line_intersection (k : ℝ) :
  (∃ A B, A ∈ Intersection k ∧ B ∈ Intersection k ∧ CircleThroughOrigin A B) ↔ 
  k = Real.sqrt (11/4) ∨ k = -Real.sqrt (11/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1101_110122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1101_110143

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) :
  seq.a 3 = 4 → S seq 9 - S seq 6 = 27 → S seq 10 = 65 := by
  sorry

#check arithmetic_sequence_sum_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1101_110143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_and_expression_l1101_110192

theorem cubic_root_and_expression : ∃ (x : ℝ), 
  (27 * x^3 - 24 * x^2 - 6 * x - 2 = 0) ∧ 
  (x = 1/3) ∧
  (∃ (a b c : ℕ), x = (a^(1/3 : ℝ) + b^(1/3 : ℝ) + 1) / c ∧ 
    a = 0 ∧ b = 0 ∧ c = 3) :=
by
  sorry

#check cubic_root_and_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_and_expression_l1101_110192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1101_110113

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x + 2*x + m else -(2^(-x) + 2*(-x) + m)

-- State the theorem
theorem odd_function_value : ∃ m : ℝ, f m (-1) = -3 :=
by
  -- We'll prove this exists by constructing the specific m that works
  use -1
  -- Now we need to show that f (-1) (-1) = -3
  -- This is where we'd typically put our proof steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1101_110113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_construction_altitude_construction_l1101_110105

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def onCircumcircle (p : Point) (c : Circle) : Prop := sorry

def angleBisectorIntersectsCircumcircle (t t' : Triangle) (c : Circle) : Prop := sorry

def altitudeIntersectsCircumcircle (t t' : Triangle) (c : Circle) : Prop := sorry

def constructedFromPerpendiculars (t t' : Triangle) : Prop := sorry

def constructedFromExtendedAltitudes (t t' : Triangle) : Prop := sorry

-- State the theorem for angle bisectors
theorem angle_bisector_construction 
  (ABC A'B'C' : Triangle) (C : Circle) : 
  isAcute ABC ∧ isAcute A'B'C' ∧
  onCircumcircle ABC.A C ∧ onCircumcircle ABC.B C ∧ onCircumcircle ABC.C C ∧
  onCircumcircle A'B'C'.A C ∧ onCircumcircle A'B'C'.B C ∧ onCircumcircle A'B'C'.C C ∧
  angleBisectorIntersectsCircumcircle ABC A'B'C' C →
  constructedFromPerpendiculars ABC A'B'C' := by sorry

-- State the theorem for altitudes
theorem altitude_construction 
  (ABC A'B'C' : Triangle) (C : Circle) : 
  isAcute ABC ∧ isAcute A'B'C' ∧
  onCircumcircle ABC.A C ∧ onCircumcircle ABC.B C ∧ onCircumcircle ABC.C C ∧
  onCircumcircle A'B'C'.A C ∧ onCircumcircle A'B'C'.B C ∧ onCircumcircle A'B'C'.C C ∧
  altitudeIntersectsCircumcircle ABC A'B'C' C →
  constructedFromExtendedAltitudes ABC A'B'C' := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_construction_altitude_construction_l1101_110105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_base4_conversion_l1101_110187

theorem binary_to_base4_conversion : 
  (11001111 : Nat) = Nat.ofDigits 4 [3, 3, 0, 3] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_base4_conversion_l1101_110187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1101_110138

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/3) : Real.sin (2*α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1101_110138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1101_110149

noncomputable def f (x : ℝ) : ℝ := Real.cos x - x / 2

def point : ℝ × ℝ := (0, 1)

def tangent_line (x y : ℝ) : Prop := x + 2 * y - 2 = 0

theorem tangent_line_is_correct :
  let (x₀, y₀) := point
  let m := -Real.sin x₀ - 1/2  -- Slope at the point
  tangent_line x₀ y₀ ∧
  f x₀ = y₀ ∧
  (∀ x y, tangent_line x y ↔ y - y₀ = m * (x - x₀)) := by
  sorry

#check tangent_line_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l1101_110149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_birthday_l1101_110163

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Determines if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Calculates the day of the week for July 3rd in a given year -/
def dayOfWeekJuly3 (startYear : ℕ) (startDay : DayOfWeek) (year : ℕ) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2009 when July 3rd falls on a Monday is 2017 -/
theorem next_monday_birthday (startYear : ℕ) (startDay : DayOfWeek) :
  startYear = 2009 →
  startDay = DayOfWeek.Thursday →
  ¬isLeapYear startYear →
  (∀ y, startYear < y → y < 2017 → dayOfWeekJuly3 startYear startDay y ≠ DayOfWeek.Monday) →
  dayOfWeekJuly3 startYear startDay 2017 = DayOfWeek.Monday :=
by
  sorry

#check next_monday_birthday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_birthday_l1101_110163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_of_special_ellipse_l1101_110142

-- Define the ellipse structure
structure Ellipse where
  center : ℝ × ℝ
  major_axis : ℝ
  minor_axis : ℝ

-- Define the properties of our specific ellipse
def special_ellipse : Ellipse :=
  { center := (4, 1)
  , major_axis := 8
  , minor_axis := 2 }

-- Theorem statement
theorem foci_distance_of_special_ellipse :
  let e := special_ellipse
  Real.sqrt ((e.major_axis ^ 2) - (e.minor_axis ^ 2)) = 2 * Real.sqrt 15 := by
  sorry

-- Additional definitions to establish the ellipse properties
axiom tangent_to_x_axis : special_ellipse.center.1 = 4
axiom tangent_to_y_axis : special_ellipse.center.2 = 1
axiom axes_parallel_to_coordinate_axes : True  -- This is implied by the structure of special_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_of_special_ellipse_l1101_110142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_approx_l1101_110158

/-- The length of the common chord of two overlapping circles -/
noncomputable def common_chord_length (r : ℝ) (d : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (d/2)^2)

/-- Theorem: The length of the common chord of two overlapping circles
    with radius 15 cm and centers 25 cm apart is approximately 17 cm -/
theorem common_chord_length_approx :
  ∃ ε > 0, |common_chord_length 15 25 - 17| < ε :=
by
  -- We'll use ε = 0.01 as our error bound
  use 0.01
  -- Split the goal into two parts
  constructor
  · -- Prove ε > 0
    norm_num
  · -- Prove |common_chord_length 15 25 - 17| < ε
    -- This part requires computation and approximation
    -- For now, we'll use sorry to skip the detailed proof
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_approx_l1101_110158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_not_isosceles_l1101_110185

-- Define a set of points
variable (S : Set (ℝ × ℝ))

-- Define a predicate for forming a triangle
def forms_triangle (a b c : ℝ × ℝ) : Prop := sorry

-- Define a predicate for isosceles triangle
def is_isosceles (a b c : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem distinct_elements_not_isosceles (a b c : ℝ × ℝ) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  forms_triangle a b c →
  ¬(is_isosceles a b c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_elements_not_isosceles_l1101_110185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l1101_110135

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (oldValue newValue : ℝ) : ℝ :=
  ((newValue - oldValue) / oldValue) * 100

theorem interest_rate_increase : 
  let lastYear : ℝ := 9.90990990990991
  let thisYear : ℝ := 11
  abs (percentageIncrease lastYear thisYear - 11) < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_increase_l1101_110135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_implies_a_bound_l1101_110147

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a*x - a * Real.log (2*x)

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x - 2*a - a/x

theorem monotonically_decreasing_implies_a_bound :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 1 2, f_deriv a x ≤ 0) → a ≥ 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_implies_a_bound_l1101_110147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payback_ratio_is_one_fourth_l1101_110140

/-- Represents the financial transaction between Greg and Tessa --/
structure Transaction where
  initial_loan : ℚ
  additional_loan : ℚ
  remaining_debt : ℚ

/-- Calculates the ratio of the amount paid back to the initial debt --/
def payback_ratio (t : Transaction) : ℚ :=
  let amount_paid_back := t.initial_loan + t.additional_loan - t.remaining_debt
  amount_paid_back / t.initial_loan

/-- Theorem stating that the payback ratio is 1:4 given the problem conditions --/
theorem payback_ratio_is_one_fourth (t : Transaction) 
  (h1 : t.initial_loan = 40)
  (h2 : t.additional_loan = 10)
  (h3 : t.remaining_debt = 30) : 
  payback_ratio t = 1 / 4 := by
  sorry

#eval payback_ratio ⟨40, 10, 30⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payback_ratio_is_one_fourth_l1101_110140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l1101_110106

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 8y + 24 and the point (-3, 4) is 10. -/
theorem circle_center_distance : ∃ (center : ℝ × ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x - 8*y + 24 ↔ (x - center.1)^2 + (y - center.2)^2 = 49) ∧ 
  Real.sqrt ((center.1 - (-3))^2 + (center.2 - 4)^2) = 10 := by
  -- Define the center of the circle
  let center : ℝ × ℝ := (3, -4)
  
  -- Provide the center and prove the theorem
  use center
  constructor
  
  -- Prove the circle equation equivalence
  · intro x y
    sorry  -- Proof of circle equation equivalence omitted
  
  -- Prove the distance is 10
  · norm_num
    sorry  -- Proof of distance calculation omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l1101_110106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_difference_l1101_110153

theorem arithmetic_geometric_mean_difference (x y : ℕ) (a b c : ℕ) :
  (x ≠ y) →
  (x > 0) →
  (y > 0) →
  (a ≠ 0) →
  ((x + y) / 2 = 100 * a + 10 * b + c) →
  (c < 10) →
  (Nat.sqrt (x * y) = 10 * b + a) →
  (|Int.ofNat x - Int.ofNat y| = 99) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_difference_l1101_110153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110103

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
  ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), g x = g (-x)) ∧
  (∀ (x₁ x₂ : ℝ), x₁ + x₂ = 5 * Real.pi / 6 → f x₁ + f x₂ = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1101_110174

theorem problem_solution :
  (∀ x y : ℝ, x = Real.sqrt 2 ∧ y = Real.sqrt 3 →
      |x - y| + 2*x = y + x) ∧
  (∀ z : ℝ, z = Real.sqrt 5 →
      z*(z - 1/z) = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1101_110174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_projection_theorem_final_theorem_l1101_110169

-- Define the sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the projection point
noncomputable def projectionPoint : ℝ × ℝ × ℝ := (0, 0, 1/2)

-- Define the properties of the sphere and its projection
def sphereProperties (s : Sphere) : Prop :=
  -- The sphere is tangent to the xy-plane
  s.center.2.2 = s.radius ∧
  -- The center has a positive z-coordinate
  s.center.2.2 > 0 ∧
  -- The projection forms the conic section y=x^2
  ∀ (x y : ℝ), y = x^2 → 
    ∃ (p : ℝ × ℝ × ℝ), p.1 = x ∧ p.2.1 = y ∧ p.2.2 = 0 ∧
    ∃ (q : ℝ × ℝ × ℝ), (q.1 - s.center.1)^2 + (q.2.1 - s.center.2.1)^2 + (q.2.2 - s.center.2.2)^2 = s.radius^2 ∧
                        (projectionPoint.1 - q.1) / (projectionPoint.2.2 - q.2.2) = (x - q.1) / (-q.2.2) ∧
                        (projectionPoint.2.1 - q.2.1) / (projectionPoint.2.2 - q.2.2) = (y - q.2.1) / (-q.2.2)

-- The theorem to prove
theorem sphere_projection_theorem (s : Sphere) :
  sphereProperties s → projectionPoint.2.2 = 1/2 := by
  sorry

-- Define a and b
def a : ℚ := 1/2
def b : ℚ := 0

-- The final theorem
theorem final_theorem : a.num + a.den = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_projection_theorem_final_theorem_l1101_110169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_three_points_l1101_110109

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define non-collinearity for three points
def NonCollinear (A B C : Point3D) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define membership of a point in a plane
def PointInPlane (P : Point3D) (π : Plane) : Prop :=
  π.a * P.x + π.b * P.y + π.c * P.z + π.d = 0

-- Theorem statement
theorem unique_plane_through_three_points (A B C : Point3D) 
  (h : NonCollinear A B C) : ∃! π : Plane, PointInPlane A π ∧ PointInPlane B π ∧ PointInPlane C π :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_three_points_l1101_110109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_p_value_l1101_110141

noncomputable def f (x : ℝ) := Real.exp x

theorem max_p_value (m n p : ℝ) 
  (h1 : f (m + n) = f m + f n)
  (h2 : f (m + n + p) = f m + f n + f p) :
  p ≤ 2 * Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_p_value_l1101_110141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1101_110162

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | 2 * x^2 - 2 * x * y + 9 * x + y = 2} =
  {(1, 9), (2, 8), (0, 2), (-1, 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1101_110162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110172

noncomputable def f (x : ℝ) := Real.cos x * (Real.sin x + Real.cos x) - 1/2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (T > 0) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (-3 * Real.pi / 8 + k * Real.pi) (Real.pi / 8 + k * Real.pi))) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 2), f x ≤ Real.sqrt 2 / 2) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 2), f x = Real.sqrt 2 / 2) ∧
    (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 2), f x ≥ -1/2) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 2), f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1101_110172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_probability_l1101_110101

noncomputable def p : ℝ := 2/3

noncomputable def q : ℝ := 1/3

def choose (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def prob_win_in (n : ℕ) : ℝ :=
  if n < 5 then 0
  else (choose (n-1) (n-4) : ℝ) * p^4 * q^(n-4)

noncomputable def total_prob : ℝ :=
  prob_win_in 5 + prob_win_in 6 + prob_win_in 7

theorem lakers_win_probability :
  total_prob = 864/2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lakers_win_probability_l1101_110101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_for_given_condition_l1101_110170

theorem function_value_for_given_condition (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = 1/5) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi) : 
  (Real.cos (Real.pi/2 + α) * Real.cos (2*Real.pi - α)) / (Real.cos (-Real.pi - α) * Real.sin (3*Real.pi/2 + α)) = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_for_given_condition_l1101_110170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1101_110193

/-- A function that represents x³(a·2ˣ - 2⁻ˣ) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * Real.exp (x * Real.log 2) - Real.exp (-x * Real.log 2))

/-- Definition of an even function -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem even_function_implies_a_equals_one (a : ℝ) :
  IsEven (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1101_110193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_one_arithmetic_sqrt3_div3_minus2_sequence_lambda_3_sequences_existence_l1101_110176

/-- Definition of a λ-k sequence -/
def is_lambda_k_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda k : ℝ) : Prop :=
  ∀ n : ℕ, S (n + 1) ^ (1 / k) - S n ^ (1 / k) = lambda * (a (n + 1)) ^ (1 / k)

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem 1: For an arithmetic λ-1 sequence, λ = 1 -/
theorem lambda_one_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) (lambda : ℝ) :
  is_arithmetic_sequence a →
  is_lambda_k_sequence a S lambda 1 →
  a 1 = 1 →
  lambda = 1 := by sorry

/-- Theorem 2: Formula for √3/3-2 sequence with positive terms -/
theorem sqrt3_div3_minus2_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_lambda_k_sequence a S (Real.sqrt 3 / 3) 2 →
  (∀ n, a n > 0) →
  (a 1 = 1 ∧ ∀ n ≥ 2, a n = 3 * 4^(n - 2)) := by sorry

/-- Theorem 3: Existence of three different λ-3 sequences -/
theorem lambda_3_sequences_existence (lambda : ℝ) :
  (∃ a₁ a₂ a₃ : ℕ → ℝ, a₁ ≠ a₂ ∧ a₂ ≠ a₃ ∧ a₁ ≠ a₃ ∧
    (∀ n, a₁ n ≥ 0) ∧ (∀ n, a₂ n ≥ 0) ∧ (∀ n, a₃ n ≥ 0) ∧
    (∃ S₁ S₂ S₃ : ℕ → ℝ,
      is_lambda_k_sequence a₁ S₁ lambda 3 ∧
      is_lambda_k_sequence a₂ S₂ lambda 3 ∧
      is_lambda_k_sequence a₃ S₃ lambda 3)) ↔
  (0 < lambda ∧ lambda < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_one_arithmetic_sqrt3_div3_minus2_sequence_lambda_3_sequences_existence_l1101_110176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_sum_7_11_l1101_110119

def triangle_third_side_sum (a b : ℕ) : ℕ :=
  let possible_lengths := List.range (a + b) |>.filter (λ m => m > max a b - min a b ∧ m < a + b)
  possible_lengths.sum

theorem triangle_third_side_sum_7_11 :
  triangle_third_side_sum 7 11 = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_sum_7_11_l1101_110119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shorten_by_one_expected_new_length_l1101_110145

/-- A sequence of digits where each digit is 0 or 9 -/
def DigitSequence := List (Fin 2)

/-- The length of the original sequence -/
def originalLength : ℕ := 2015

/-- The probability of a digit being the same as the previous one -/
def sameDigitProb : ℝ := 0.1

/-- The probability of a digit being different from the previous one -/
def differentDigitProb : ℝ := 0.9

/-- The number of Bernoulli trials (number of digits that can be removed) -/
def numTrials : ℕ := originalLength - 1

/-- Function to calculate the probability of exactly one digit being removed -/
def probExactlyOneRemoved (n : ℕ) (p : ℝ) : ℝ :=
  (n : ℝ) * p * (1 - p) ^ (n - 1)

/-- Function to calculate the expected number of digits removed -/
def expectedRemoved (n : ℕ) (p : ℝ) : ℝ :=
  (n : ℝ) * p

/-- Theorem stating the probability of shortening by exactly one digit -/
theorem prob_shorten_by_one :
  ∃ ε > 0, |probExactlyOneRemoved numTrials sameDigitProb - 1.564e-90| < ε := by
  sorry

/-- Theorem stating the expected length of the new sequence -/
theorem expected_new_length :
  ∃ ε > 0, |originalLength - expectedRemoved numTrials sameDigitProb - 1813.6| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shorten_by_one_expected_new_length_l1101_110145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_correct_l1101_110191

def overall_average (students : List ℚ) (marks : List ℚ) : ℚ :=
  (List.sum (List.zipWith (· * ·) students marks)) / (List.sum students)

theorem overall_average_correct (students : List ℚ) (marks : List ℚ) 
  (h1 : students.length = 8)
  (h2 : marks.length = 8)
  (h3 : students = [65, 70, 80, 75, 90, 85, 60, 55])
  (h4 : marks = [50, 60, 75, 85, 70, 90, 65, 45]) :
  overall_average students marks = 69.22413793103448 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_correct_l1101_110191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1101_110154

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.log (x^2 - 5*x + 6)

-- Theorem statement
theorem f_strictly_increasing :
  ∀ x y, 3 < x ∧ x < y → f x < f y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1101_110154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_square_circle_area_ratio_l1101_110156

/-- Given a wire of length 2l cut into two equal pieces, where one piece forms a square
    and the other forms a circle, the ratio of the area of the square to the area of the circle is π/4. -/
theorem wire_square_circle_area_ratio (l : ℝ) (h : l > 0) :
  (l / 4) ^ 2 / (Real.pi * (l / (2 * Real.pi)) ^ 2) = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_square_circle_area_ratio_l1101_110156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_phi_l1101_110107

theorem least_positive_angle_phi : 
  ∃ φ : Real, 
    (φ > 0) ∧ 
    (φ ≤ π) ∧ 
    (Real.cos (10 * π / 180) = Real.sin (50 * π / 180) + Real.sin φ) ∧
    (∀ ψ : Real, (ψ > 0) ∧ (ψ < φ) → Real.cos (10 * π / 180) ≠ Real.sin (50 * π / 180) + Real.sin ψ) ∧
    φ = 10 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_angle_phi_l1101_110107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_interval_l1101_110181

theorem sin_increasing_interval (ω φ : ℝ) (h_ω_pos : ω > 0)
  (h_symmetry : (π / 2) * ω = π)
  (h_max : ∀ x, Real.cos (ω * x + φ) ≤ Real.cos (-7 * π / 8 * ω + φ)) :
  ∀ x ∈ Set.Icc (-π/8) (3*π/8), 
    ∀ y ∈ Set.Icc (-π/8) (3*π/8),
      x < y → Real.sin (ω * x + φ) < Real.sin (ω * y + φ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_interval_l1101_110181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_6_or_8_l1101_110165

def is_multiple_of_6_or_8 (n : ℕ) : Bool := n % 6 = 0 || n % 8 = 0

theorem probability_multiple_6_or_8 : 
  (Finset.filter (λ n => is_multiple_of_6_or_8 n) (Finset.range 100)).card / 100 = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_6_or_8_l1101_110165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_f_l1101_110116

def f (n m : ℕ+) : ℤ :=
  let rec a : ℕ → ℤ
    | 0 => m
    | k+1 => (a k)^2 % n
  (List.range 2001).foldl (λ sum i => sum + (-1)^i * a i) 0

theorem exists_positive_f (n : ℕ+) :
  (n ≥ 5) → (∃ m : ℕ+, 2 ≤ m ∧ m ≤ (n : ℕ) / 2 ∧ f n m > 0) ↔ (n = 6 ∨ n = 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_positive_f_l1101_110116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_order_l1101_110117

theorem sine_order : Real.sin 4 < Real.sin 3 ∧ Real.sin 3 < Real.sin 1 ∧ Real.sin 1 < Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_order_l1101_110117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_lower_bound_l1101_110190

/-- The curve W defined by y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle with three vertices on W -/
structure RectangleOnW where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  hA : A ∈ W
  hB : B ∈ W
  hC : C ∈ W
  is_rectangle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- The perimeter of a rectangle -/
noncomputable def perimeter (r : RectangleOnW) : ℝ :=
  2 * (((r.A.1 - r.B.1)^2 + (r.A.2 - r.B.2)^2).sqrt +
       ((r.B.1 - r.C.1)^2 + (r.B.2 - r.C.2)^2).sqrt)

/-- Theorem: The perimeter of any rectangle with three vertices on W is greater than 3√3 -/
theorem perimeter_lower_bound (r : RectangleOnW) : perimeter r > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_lower_bound_l1101_110190
