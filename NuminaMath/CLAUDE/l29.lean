import Mathlib

namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l29_2936

/-- The number of different hamburger combinations -/
def hamburger_combinations (num_buns num_condiments num_patty_choices : ℕ) : ℕ :=
  num_buns * (2 ^ num_condiments) * num_patty_choices

/-- Theorem stating the total number of different hamburger combinations -/
theorem total_hamburger_combinations :
  hamburger_combinations 3 9 3 = 4608 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l29_2936


namespace NUMINAMATH_CALUDE_number_of_observations_l29_2929

theorem number_of_observations (original_mean corrected_mean wrong_value correct_value : ℝ) 
  (h1 : original_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 36.5) :
  ∃ n : ℕ, (n : ℝ) * original_mean + (correct_value - wrong_value) = n * corrected_mean ∧ n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l29_2929


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l29_2958

theorem smallest_a_for_nonprime_polynomial :
  ∃ (a : ℕ+), (∀ (x : ℤ), ∃ (p q : ℤ), p > 1 ∧ q > 1 ∧ x^4 + (a + 4)^2 = p * q) ∧
  (∀ (b : ℕ+), b < a → ∃ (y : ℤ), ∀ (p q : ℤ), (p > 1 ∧ q > 1 → y^4 + (b + 4)^2 ≠ p * q)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_polynomial_l29_2958


namespace NUMINAMATH_CALUDE_climb_10_steps_in_8_moves_l29_2931

/-- The number of ways to climb n steps in exactly k moves, where each move can be either 1 or 2 steps. -/
def climbWays (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem states that there are 28 ways to climb 10 steps in exactly 8 moves. -/
theorem climb_10_steps_in_8_moves : climbWays 10 8 = 28 := by sorry

end NUMINAMATH_CALUDE_climb_10_steps_in_8_moves_l29_2931


namespace NUMINAMATH_CALUDE_smallest_common_divisor_l29_2962

theorem smallest_common_divisor (n : ℕ) (h1 : n = 627) :
  let m := n + 3
  let k := Nat.minFac (Nat.gcd m (Nat.gcd 4590 105))
  k = 105 := by sorry

end NUMINAMATH_CALUDE_smallest_common_divisor_l29_2962


namespace NUMINAMATH_CALUDE_mishas_current_money_l29_2904

/-- Misha's current amount of money in dollars -/
def current_money : ℕ := sorry

/-- The amount Misha needs to earn in dollars -/
def money_to_earn : ℕ := 13

/-- The total amount Misha will have after earning more money, in dollars -/
def total_money : ℕ := 47

/-- Theorem stating Misha's current amount of money -/
theorem mishas_current_money : current_money = 34 := by sorry

end NUMINAMATH_CALUDE_mishas_current_money_l29_2904


namespace NUMINAMATH_CALUDE_runners_meeting_time_l29_2914

/-- Represents a runner with their lap time in minutes -/
structure Runner where
  name : String
  lapTime : Nat

/-- Calculates the earliest time (in minutes) when all runners meet at the starting point -/
def earliestMeetingTime (runners : List Runner) : Nat :=
  sorry

theorem runners_meeting_time :
  let runners : List Runner := [
    { name := "Laura", lapTime := 5 },
    { name := "Maria", lapTime := 8 },
    { name := "Charlie", lapTime := 10 },
    { name := "Zoe", lapTime := 2 }
  ]
  earliestMeetingTime runners = 40 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l29_2914


namespace NUMINAMATH_CALUDE_triangle_expression_negative_l29_2994

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- Theorem statement
theorem triangle_expression_negative (t : Triangle) : (t.a - t.c)^2 - t.b^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_negative_l29_2994


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l29_2945

/-- The probability of a specific pairing in a class with random pairings. -/
theorem specific_pairing_probability
  (total_students : ℕ)
  (non_participating : ℕ)
  (h1 : total_students = 32)
  (h2 : non_participating = 1)
  : (1 : ℚ) / (total_students - non_participating - 1) = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_specific_pairing_probability_l29_2945


namespace NUMINAMATH_CALUDE_goldbach_refutation_l29_2991

theorem goldbach_refutation (n : ℕ) : 
  (∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) → 
  ¬(∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
by sorry

end NUMINAMATH_CALUDE_goldbach_refutation_l29_2991


namespace NUMINAMATH_CALUDE_lcm_of_5_8_12_20_l29_2972

theorem lcm_of_5_8_12_20 : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 12 20)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_8_12_20_l29_2972


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l29_2948

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_inequality (ABCD : Quadrilateral) :
  length ABCD.A ABCD.D = length ABCD.B ABCD.C →
  angle_measure ABCD.A ABCD.D ABCD.C > angle_measure ABCD.B ABCD.C ABCD.D →
  length ABCD.A ABCD.C > length ABCD.B ABCD.D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l29_2948


namespace NUMINAMATH_CALUDE_david_widget_production_difference_l29_2959

/-- Given David's widget production rates and hours worked, prove the difference between Monday and Tuesday production. -/
theorem david_widget_production_difference 
  (t : ℕ) -- Number of hours worked on Monday
  (w : ℕ) -- Widgets produced per hour on Monday
  (h1 : w = 2 * t) -- Relation between w and t
  : w * t - (w + 5) * (t - 3) = t + 15 := by
  sorry

end NUMINAMATH_CALUDE_david_widget_production_difference_l29_2959


namespace NUMINAMATH_CALUDE_median_to_longest_side_l29_2905

/-- Given a triangle with side lengths 10, 24, and 26, the length of the median to the longest side is 13. -/
theorem median_to_longest_side (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  let m := (1/2) * Real.sqrt (2 * a^2 + 2 * b^2 - c^2)
  m = 13 := by sorry

end NUMINAMATH_CALUDE_median_to_longest_side_l29_2905


namespace NUMINAMATH_CALUDE_system_solution_proof_l29_2998

theorem system_solution_proof (x y : ℚ) : 
  (3 * x - y = 4 ∧ 6 * x - 3 * y = 10) ↔ (x = 2/3 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l29_2998


namespace NUMINAMATH_CALUDE_apple_distribution_l29_2967

/-- The number of ways to distribute n items among k people with a minimum of m items each -/
def distribution_ways (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 253 ways to distribute 30 apples among 3 people with at least 3 apples each -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l29_2967


namespace NUMINAMATH_CALUDE_fraction_ordering_l29_2981

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l29_2981


namespace NUMINAMATH_CALUDE_divide_by_fraction_main_proof_l29_2909

theorem divide_by_fraction (a b c : ℝ) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem main_proof : (5 : ℝ) / ((7 : ℝ) / 3) = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_main_proof_l29_2909


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l29_2902

theorem units_digit_of_quotient (n : ℕ) (h : 5 ∣ (2^1993 + 3^1993)) :
  (2^1993 + 3^1993) / 5 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l29_2902


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l29_2928

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup factor applied to the wholesale price -/
def markup_factor : ℝ := 1.8

theorem wholesale_price_calculation :
  wholesale_price * markup_factor = retail_price :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l29_2928


namespace NUMINAMATH_CALUDE_cubic_function_range_l29_2943

/-- A cubic function f(x) = ax³ + bx² + cx + d satisfying given conditions -/
structure CubicFunction where
  f : ℝ → ℝ
  cubic : ∃ (a b c d : ℝ), ∀ x, f x = a * x^3 + b * x^2 + c * x + d
  cond1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2
  cond2 : 1 ≤ f 1 ∧ f 1 ≤ 3
  cond3 : 2 ≤ f 2 ∧ f 2 ≤ 4
  cond4 : -1 ≤ f 3 ∧ f 3 ≤ 1

/-- The value of f(4) is always within the range [-21¾, 1] for any CubicFunction -/
theorem cubic_function_range (cf : CubicFunction) :
  -21.75 ≤ cf.f 4 ∧ cf.f 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l29_2943


namespace NUMINAMATH_CALUDE_no_odd_3digit_div5_without5_l29_2934

theorem no_odd_3digit_div5_without5 : 
  ¬∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- 3-digit number
    n % 2 = 1 ∧           -- odd
    n % 5 = 0 ∧           -- divisible by 5
    ∀ d : ℕ, d < 3 → (n / 10^d) % 10 ≠ 5  -- does not contain digit 5
    := by sorry

end NUMINAMATH_CALUDE_no_odd_3digit_div5_without5_l29_2934


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_39_and_120_l29_2942

theorem no_square_divisible_by_six_between_39_and_120 :
  ¬∃ (x : ℕ), ∃ (y : ℕ), x = y^2 ∧ 6 ∣ x ∧ 39 < x ∧ x < 120 :=
by
  sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_39_and_120_l29_2942


namespace NUMINAMATH_CALUDE_triangle_properties_l29_2982

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following statements under the given conditions. -/
theorem triangle_properties (A B C a b c : Real) :
  -- Given conditions
  (4 * Real.sin (A / 2 - B / 2) ^ 2 + 4 * Real.sin A * Real.sin B = 2 + Real.sqrt 2) →
  (b = 4) →
  (1 / 2 * a * b * Real.sin C = 6) →
  -- Triangle inequality and angle sum
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (A + B + C = π) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  -- Statements to prove
  (C = π / 4 ∧
   c = Real.sqrt 10 ∧
   Real.tan (2 * B - C) = 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l29_2982


namespace NUMINAMATH_CALUDE_quiz_true_false_count_l29_2978

theorem quiz_true_false_count :
  ∀ n : ℕ,
  (2^n - 2) * 16 = 224 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_true_false_count_l29_2978


namespace NUMINAMATH_CALUDE_a_range_l29_2990

theorem a_range (p : ∀ x > 0, x + 1/x ≥ a^2 - a) 
                (q : ∃ x : ℝ, x + |x - 1| = 2*a) : 
  a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l29_2990


namespace NUMINAMATH_CALUDE_modulus_of_z_l29_2986

-- Define the complex number z
def z : ℂ := sorry

-- State the given equation
axiom z_equation : z^2 + z = 1 - 3*Complex.I

-- Define the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l29_2986


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l29_2960

theorem opposite_of_negative_2023 : -(-(2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l29_2960


namespace NUMINAMATH_CALUDE_rebecca_groups_l29_2999

def egg_count : Nat := 75
def banana_count : Nat := 99
def marble_count : Nat := 48
def apple_count : Nat := 6 * 12  -- 6 dozen
def orange_count : Nat := 6  -- half dozen

def egg_group_size : Nat := 4
def banana_group_size : Nat := 5
def marble_group_size : Nat := 6
def apple_group_size : Nat := 12
def orange_group_size : Nat := 2

def total_groups : Nat :=
  (egg_count + egg_group_size - 1) / egg_group_size +
  (banana_count + banana_group_size - 1) / banana_group_size +
  marble_count / marble_group_size +
  apple_count / apple_group_size +
  orange_count / orange_group_size

theorem rebecca_groups : total_groups = 54 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_groups_l29_2999


namespace NUMINAMATH_CALUDE_total_ad_cost_is_66000_l29_2930

/-- Represents an advertisement with its duration and cost per minute -/
structure Advertisement where
  duration : ℕ
  costPerMinute : ℕ

/-- Calculates the total cost of an advertisement -/
def adCost (ad : Advertisement) : ℕ := ad.duration * ad.costPerMinute

/-- The list of advertisements shown during the race -/
def raceAds : List Advertisement := [
  ⟨2, 3500⟩,
  ⟨3, 4500⟩,
  ⟨3, 3000⟩,
  ⟨2, 4000⟩,
  ⟨5, 5500⟩
]

/-- The theorem stating that the total cost of advertisements is $66000 -/
theorem total_ad_cost_is_66000 :
  (raceAds.map adCost).sum = 66000 := by sorry

end NUMINAMATH_CALUDE_total_ad_cost_is_66000_l29_2930


namespace NUMINAMATH_CALUDE_first_degree_function_characterization_l29_2984

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem first_degree_function_characterization
  (f : ℝ → ℝ) 
  (h1 : FirstDegreeFunction f)
  (h2 : ∀ x : ℝ, f (f x) = 4 * x + 6) :
  (∀ x : ℝ, f x = 2 * x + 2) ∨ (∀ x : ℝ, f x = -2 * x - 6) :=
sorry

end NUMINAMATH_CALUDE_first_degree_function_characterization_l29_2984


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l29_2923

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 7/49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l29_2923


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_properties_l29_2995

/-- Represents a sequence of seven consecutive even numbers -/
structure ConsecutiveEvenNumbers where
  middle : ℤ
  sum : ℤ
  sum_eq : sum = 7 * middle

/-- Properties of the sequence of consecutive even numbers -/
theorem consecutive_even_numbers_properties (seq : ConsecutiveEvenNumbers)
  (h : seq.sum = 686) :
  let smallest := seq.middle - 6
  let median := seq.middle
  let mean := seq.sum / 7
  (smallest = 92) ∧ (median = 98) ∧ (mean = 98) := by
  sorry

#check consecutive_even_numbers_properties

end NUMINAMATH_CALUDE_consecutive_even_numbers_properties_l29_2995


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l29_2953

theorem sqrt_difference_approximation : 
  ∃ ε > 0, |Real.sqrt (49 + 16) - Real.sqrt (36 - 9) - 2.8661| < ε :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l29_2953


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l29_2952

theorem opposite_of_negative_one_half : 
  (-(-(1/2 : ℚ))) = (1/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l29_2952


namespace NUMINAMATH_CALUDE_l_shaped_area_l29_2912

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (a b c d : ℕ) (h1 : a = 7) (h2 : b = 2) (h3 : c = 2) (h4 : d = 3) : 
  a^2 - (b^2 + c^2 + d^2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l29_2912


namespace NUMINAMATH_CALUDE_triangle_double_inequality_left_equality_condition_right_equality_condition_l29_2988

/-- A triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

/-- The double inequality for triangles -/
theorem triangle_double_inequality (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) ≤ (t.a + t.b + t.c)^2 ∧
  (t.a + t.b + t.c)^2 < 4 * (t.a * t.b + t.b * t.c + t.c * t.a) := by
  sorry

/-- Equality condition for the left inequality -/
theorem left_equality_condition (t : Triangle) :
  3 * (t.a * t.b + t.b * t.c + t.c * t.a) = (t.a + t.b + t.c)^2 ↔ t.a = t.b ∧ t.b = t.c := by
  sorry

/-- Equality condition for the right inequality -/
theorem right_equality_condition (t : Triangle) :
  (t.a + t.b + t.c)^2 = 4 * (t.a * t.b + t.b * t.c + t.c * t.a) ↔
  t.a + t.b = t.c ∨ t.b + t.c = t.a ∨ t.c + t.a = t.b := by
  sorry

end NUMINAMATH_CALUDE_triangle_double_inequality_left_equality_condition_right_equality_condition_l29_2988


namespace NUMINAMATH_CALUDE_bee_legs_count_l29_2965

/-- Given 8 bees with a total of 48 legs, prove that each bee has 6 legs. -/
theorem bee_legs_count :
  let total_bees : ℕ := 8
  let total_legs : ℕ := 48
  total_legs / total_bees = 6 := by sorry

end NUMINAMATH_CALUDE_bee_legs_count_l29_2965


namespace NUMINAMATH_CALUDE_complement_intersection_equals_three_l29_2935

universe u

def U : Finset (Fin 5) := {0, 1, 2, 3, 4}
def M : Finset (Fin 5) := {0, 1, 2}
def N : Finset (Fin 5) := {2, 3}

theorem complement_intersection_equals_three :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_three_l29_2935


namespace NUMINAMATH_CALUDE_binomial_12_11_l29_2987

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l29_2987


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l29_2963

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem infinitely_many_divisible_pairs :
  ∀ k : ℕ, ∃ m n : ℕ+,
    (m : ℕ) ∣ (n : ℕ)^2 + 1 ∧
    (n : ℕ) ∣ (m : ℕ)^2 + 1 ∧
    (m : ℕ) = fib (2 * k + 1) ∧
    (n : ℕ) = fib (2 * k + 3) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_pairs_l29_2963


namespace NUMINAMATH_CALUDE_equation_solution_l29_2900

theorem equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ↔ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l29_2900


namespace NUMINAMATH_CALUDE_proposition_analysis_l29_2906

theorem proposition_analysis (m n : ℝ) : 
  (¬ (((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) ∧ 
  ((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧
  (((m > 0) ∧ (n > 0)) → (m + n > 0)) ∧
  (¬ ((m + n > 0) → ((m > 0) ∧ (n > 0)))) ∧
  (((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧ ¬(((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l29_2906


namespace NUMINAMATH_CALUDE_water_tank_fill_time_l29_2920

/-- Represents the time (in hours) it takes to fill a water tank -/
def fill_time : ℝ → ℝ → ℝ → Prop :=
  λ T leak_empty_time leak_fill_time =>
    (1 / T - 1 / leak_empty_time = 1 / leak_fill_time) ∧
    (leak_fill_time = T + 1)

theorem water_tank_fill_time :
  ∃ (T : ℝ), T > 0 ∧ fill_time T 30 (T + 1) ∧ T = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_fill_time_l29_2920


namespace NUMINAMATH_CALUDE_sunday_necklace_production_l29_2980

/-- The number of necklaces made by the first machine -/
def first_machine_necklaces : ℕ := 45

/-- The ratio of necklaces made by the second machine compared to the first -/
def second_machine_ratio : ℚ := 2.4

/-- The total number of necklaces made on Sunday -/
def total_necklaces : ℕ := 153

theorem sunday_necklace_production :
  (first_machine_necklaces : ℚ) + (first_machine_necklaces : ℚ) * second_machine_ratio = (total_necklaces : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sunday_necklace_production_l29_2980


namespace NUMINAMATH_CALUDE_lex_apple_count_l29_2949

/-- The total number of apples Lex picked -/
def total_apples : ℕ := 85

/-- The number of apples with worms -/
def wormy_apples : ℕ := total_apples / 5

/-- The number of bruised apples -/
def bruised_apples : ℕ := total_apples / 5 + 9

/-- The number of apples left to eat raw -/
def raw_apples : ℕ := 42

theorem lex_apple_count :
  wormy_apples + bruised_apples + raw_apples = total_apples :=
by sorry

end NUMINAMATH_CALUDE_lex_apple_count_l29_2949


namespace NUMINAMATH_CALUDE_car_distances_l29_2993

/-- Represents the possible distances between two cars after one hour, given their initial distance and speeds. -/
def possible_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Set ℝ :=
  { d | ∃ (direction1 direction2 : Bool),
      d = |initial_distance + (if direction1 then speed1 else -speed1) - (if direction2 then speed2 else -speed2)| }

/-- Theorem stating the possible distances between two cars after one hour. -/
theorem car_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
    (h_initial : initial_distance = 200)
    (h_speed1 : speed1 = 60)
    (h_speed2 : speed2 = 80) :
    possible_distances initial_distance speed1 speed2 = {60, 340, 180, 220} := by
  sorry

end NUMINAMATH_CALUDE_car_distances_l29_2993


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l29_2940

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l29_2940


namespace NUMINAMATH_CALUDE_inequality_solution_set_l29_2964

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l29_2964


namespace NUMINAMATH_CALUDE_max_quotient_value_l29_2977

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l29_2977


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l29_2911

/-- 
Given a quadratic equation 6x² - 1 = 3x, when converted to the general form ax² + bx + c = 0,
the coefficient of x² (a) is 6 and the coefficient of x (b) is -3.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 6 * x^2 - 1 = 3 * x ↔ a * x^2 + b * x + c = 0) ∧
    a = 6 ∧ 
    b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l29_2911


namespace NUMINAMATH_CALUDE_simplify_expression_l29_2908

theorem simplify_expression : (5^7 + 3^6) * (1^5 - (-1)^4)^10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l29_2908


namespace NUMINAMATH_CALUDE_system_of_inequalities_l29_2937

theorem system_of_inequalities (x : ℝ) : 
  3 * (x + 1) < 4 * x + 5 → 2 * x > (x + 6) / 2 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l29_2937


namespace NUMINAMATH_CALUDE_plane_count_l29_2997

theorem plane_count (total_wings : ℕ) (wings_per_plane : ℕ) (h1 : total_wings = 108) (h2 : wings_per_plane = 2) :
  total_wings / wings_per_plane = 54 := by
  sorry

end NUMINAMATH_CALUDE_plane_count_l29_2997


namespace NUMINAMATH_CALUDE_fraction_sum_l29_2932

theorem fraction_sum (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l29_2932


namespace NUMINAMATH_CALUDE_constant_term_exists_l29_2944

/-- Represents the derivative of a function q with respect to some variable -/
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The equation q' = 3q - 3 -/
def equation (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x - 3

/-- The value of (4')' is 72 -/
def condition (q : ℝ → ℝ) : Prop :=
  derivative (derivative q) 4 = 72

/-- There exists a constant term in the equation -/
theorem constant_term_exists (q : ℝ → ℝ) (h1 : equation q) (h2 : condition q) :
  ∃ c : ℝ, ∀ x, derivative q x = 3 * q x + c :=
sorry

end NUMINAMATH_CALUDE_constant_term_exists_l29_2944


namespace NUMINAMATH_CALUDE_product_with_decimals_l29_2938

theorem product_with_decimals (a b c : ℚ) (h : (125 : ℕ) * 384 = 48000) :
  a = 0.125 ∧ b = 3.84 ∧ c = 0.48 → a * b = c := by sorry

end NUMINAMATH_CALUDE_product_with_decimals_l29_2938


namespace NUMINAMATH_CALUDE_sin_600_degrees_l29_2975

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l29_2975


namespace NUMINAMATH_CALUDE_slab_rate_per_sq_meter_l29_2957

/-- Prove that the rate per square meter for paving a rectangular room is 900 Rs. -/
theorem slab_rate_per_sq_meter (length width total_cost : ℝ) : 
  length = 6 →
  width = 4.75 →
  total_cost = 25650 →
  total_cost / (length * width) = 900 := by
sorry

end NUMINAMATH_CALUDE_slab_rate_per_sq_meter_l29_2957


namespace NUMINAMATH_CALUDE_tetrahedralContactsFormula_l29_2921

/-- The number of contact points in a tetrahedral stack of spheres -/
def tetrahedralContacts (n : ℕ) : ℕ := n^3 - n

/-- Theorem: The number of contact points in a tetrahedral stack of spheres
    with n spheres along each edge is n³ - n -/
theorem tetrahedralContactsFormula (n : ℕ) :
  tetrahedralContacts n = n^3 - n := by
  sorry

end NUMINAMATH_CALUDE_tetrahedralContactsFormula_l29_2921


namespace NUMINAMATH_CALUDE_investment_profit_ratio_l29_2989

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit ratio of two partners -/
def profitRatio (p q : Partner) : ℚ × ℚ :=
  let pProfit := p.investment * p.time
  let qProfit := q.investment * q.time
  (pProfit, qProfit)

theorem investment_profit_ratio :
  let p : Partner := ⟨7, 5⟩
  let q : Partner := ⟨5, 14⟩
  profitRatio p q = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_ratio_l29_2989


namespace NUMINAMATH_CALUDE_range_of_a_l29_2907

-- Define the function f(x) for any real a
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 + a) * x^2 - a * x + 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > 0) ↔ ((-4/3 < a ∧ a < -1) ∨ a = 0) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l29_2907


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l29_2919

/-- The quadratic equation x^2 - 4mx + 3m^2 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

theorem quadratic_equation_properties :
  ∀ m : ℝ,
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation x1 m ∧ quadratic_equation x2 m) ∧
  (∀ x1 x2 : ℝ, x1 > x2 → quadratic_equation x1 m → quadratic_equation x2 m → x1 - x2 = 2 → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l29_2919


namespace NUMINAMATH_CALUDE_sum_norms_gt_sum_pairwise_norms_l29_2917

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

/-- Given four pairwise non-parallel vectors whose sum is zero, 
    the sum of their norms is greater than the sum of the norms of their pairwise sums with the first vector -/
theorem sum_norms_gt_sum_pairwise_norms (a b c d : V) 
    (h_sum : a + b + c + d = 0)
    (h_ab : ¬ ∃ (k : ℝ), b = k • a)
    (h_ac : ¬ ∃ (k : ℝ), c = k • a)
    (h_ad : ¬ ∃ (k : ℝ), d = k • a)
    (h_bc : ¬ ∃ (k : ℝ), c = k • b)
    (h_bd : ¬ ∃ (k : ℝ), d = k • b)
    (h_cd : ¬ ∃ (k : ℝ), d = k • c) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ > ‖a + b‖ + ‖a + c‖ + ‖a + d‖ := by
  sorry

end NUMINAMATH_CALUDE_sum_norms_gt_sum_pairwise_norms_l29_2917


namespace NUMINAMATH_CALUDE_tan_equation_solution_l29_2992

theorem tan_equation_solution (θ : Real) :
  2 * Real.tan θ - Real.tan (θ + π/4) = 7 → Real.tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l29_2992


namespace NUMINAMATH_CALUDE_unique_rectangle_with_given_perimeter_and_area_l29_2926

theorem unique_rectangle_with_given_perimeter_and_area : 
  ∃! (w h : ℕ+), (2 * (w + h) = 80) ∧ (w * h = 400) :=
by sorry

end NUMINAMATH_CALUDE_unique_rectangle_with_given_perimeter_and_area_l29_2926


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l29_2946

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l29_2946


namespace NUMINAMATH_CALUDE_red_points_centroid_theorem_l29_2956

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line in a 2D grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- Definition of a grid -/
structure Grid where
  size : Nat
  horizontal_lines : List GridLine
  vertical_lines : List GridLine

/-- Definition of a triangle -/
structure Triangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Calculates the centroid of a triangle -/
def centroid (t : Triangle) : GridPoint :=
  { x := (t.a.x + t.b.x + t.c.x) / 3,
    y := (t.a.y + t.b.y + t.c.y) / 3 }

/-- Theorem statement -/
theorem red_points_centroid_theorem (m : Nat) (grid : Grid)
  (h1 : grid.size = 4 * m + 2)
  (h2 : grid.horizontal_lines.length = 2 * m + 1)
  (h3 : grid.vertical_lines.length = 2 * m + 1) :
  ∃ (A B C D E F : GridPoint),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    centroid {a := A, b := B, c := C} = {x := 0, y := 0} ∧
    centroid {a := D, b := E, c := F} = {x := 0, y := 0} :=
  sorry

end NUMINAMATH_CALUDE_red_points_centroid_theorem_l29_2956


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l29_2939

theorem polygon_interior_angles (n : ℕ) : 
  180 * (n - 2) = 1440 → n = 10 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l29_2939


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_means_l29_2927

/-- A linear regression line for a set of data points -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample means of a dataset -/
structure SampleMeans where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem stating that two regression lines for the same dataset intersect at the sample means -/
theorem regression_lines_intersect_at_means 
  (m n : RegressionLine) (means : SampleMeans) : 
  ∃ (x y : ℝ), 
    x = means.x_mean ∧ 
    y = means.y_mean ∧ 
    y = m.slope * x + m.intercept ∧ 
    y = n.slope * x + n.intercept := by
  sorry


end NUMINAMATH_CALUDE_regression_lines_intersect_at_means_l29_2927


namespace NUMINAMATH_CALUDE_largest_power_divisor_l29_2985

theorem largest_power_divisor (m n : ℕ) (h1 : m = 1991^1992) (h2 : n = 1991^1990) :
  ∃ k : ℕ, k = 1991^1990 ∧ 
  k ∣ (1990*m + 1992*n) ∧ 
  ∀ l : ℕ, l > k → l = 1991^(1990 + (l.log 1991 - 1990)) → ¬(l ∣ (1990*m + 1992*n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_divisor_l29_2985


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l29_2903

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l29_2903


namespace NUMINAMATH_CALUDE_typing_service_cost_l29_2968

/-- Typing service cost calculation -/
theorem typing_service_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) :
  total_pages = 100 →
  revised_once = 20 →
  revised_twice = 30 →
  revision_cost = 5 →
  total_cost = 1400 →
  ∃ (first_time_cost : ℚ),
    first_time_cost * total_pages + 
    revision_cost * (revised_once + 2 * revised_twice) = total_cost ∧
    first_time_cost = 10 := by
  sorry


end NUMINAMATH_CALUDE_typing_service_cost_l29_2968


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l29_2933

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 120 → x = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l29_2933


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l29_2925

def multiples_of_10 : List ℕ := List.range 30 |>.map (fun n => 10 * (n + 1))

theorem average_of_multiples_of_10 :
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 155 := by
  sorry

#eval (List.sum multiples_of_10) / (List.length multiples_of_10)

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l29_2925


namespace NUMINAMATH_CALUDE_nicki_running_mileage_nicki_second_half_mileage_l29_2969

/-- Calculates the weekly mileage for the second half of the year given the conditions -/
theorem nicki_running_mileage (total_weeks : ℕ) (first_half_weeks : ℕ) 
  (first_half_weekly_miles : ℕ) (total_annual_miles : ℕ) : ℕ :=
  let second_half_weeks := total_weeks - first_half_weeks
  let first_half_total_miles := first_half_weekly_miles * first_half_weeks
  let second_half_total_miles := total_annual_miles - first_half_total_miles
  second_half_total_miles / second_half_weeks

/-- Proves that Nicki ran 30 miles per week in the second half of the year -/
theorem nicki_second_half_mileage :
  nicki_running_mileage 52 26 20 1300 = 30 := by
  sorry

end NUMINAMATH_CALUDE_nicki_running_mileage_nicki_second_half_mileage_l29_2969


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l29_2922

-- Define the structure for a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the property of three lines being parallel
def parallel (l₁ l₂ l₃ : Line2D) : Prop :=
  ∃ k₁ k₂ : ℝ, l₁.a = k₁ * l₂.a ∧ l₁.b = k₁ * l₂.b ∧
              l₁.a = k₂ * l₃.a ∧ l₁.b = k₂ * l₃.b

-- Define when a point is on a line
def on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define an equilateral triangle
def is_equilateral (A B C : Point2D) : Prop :=
  let d_AB := (A.x - B.x)^2 + (A.y - B.y)^2
  let d_BC := (B.x - C.x)^2 + (B.y - C.y)^2
  let d_CA := (C.x - A.x)^2 + (C.y - A.y)^2
  d_AB = d_BC ∧ d_BC = d_CA

-- State the theorem
theorem equilateral_triangle_on_parallel_lines 
  (d₁ d₂ d₃ : Line2D) 
  (h : parallel d₁ d₂ d₃) : 
  ∃ (A : Point2D) (B : Point2D) (C : Point2D),
    on_line A d₂ ∧ on_line B d₁ ∧ on_line C d₃ ∧
    is_equilateral A B C := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parallel_lines_l29_2922


namespace NUMINAMATH_CALUDE_same_color_probability_l29_2954

def red_plates : ℕ := 7
def blue_plates : ℕ := 5
def green_plates : ℕ := 3

def total_plates : ℕ := red_plates + blue_plates + green_plates

def same_color_pairs : ℕ := (red_plates.choose 2) + (blue_plates.choose 2) + (green_plates.choose 2)
def total_pairs : ℕ := total_plates.choose 2

theorem same_color_probability :
  (same_color_pairs : ℚ) / total_pairs = 34 / 105 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l29_2954


namespace NUMINAMATH_CALUDE_number_of_shoppers_l29_2966

theorem number_of_shoppers (isabella sam giselle : ℕ) (shoppers : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / shoppers = 115 →
  shoppers = 3 := by
sorry

end NUMINAMATH_CALUDE_number_of_shoppers_l29_2966


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l29_2974

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical shift to a parabola --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 2 ∧ p.h = 1 ∧ p.k = 3 →
  let p' := shift_parabola p 2 (-1)
  p'.a = 2 ∧ p'.h = -1 ∧ p'.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l29_2974


namespace NUMINAMATH_CALUDE_seminar_attendance_l29_2973

/-- The total number of people who attended the seminars given the attendance for math and music seminars -/
theorem seminar_attendance (math_attendees music_attendees both_attendees : ℕ) 
  (h1 : math_attendees = 75)
  (h2 : music_attendees = 61)
  (h3 : both_attendees = 12) :
  math_attendees + music_attendees - both_attendees = 124 := by
  sorry

#check seminar_attendance

end NUMINAMATH_CALUDE_seminar_attendance_l29_2973


namespace NUMINAMATH_CALUDE_sin_four_arcsin_l29_2961

theorem sin_four_arcsin (x : ℝ) (h : x ∈ Set.Icc (-1 : ℝ) 1) :
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_four_arcsin_l29_2961


namespace NUMINAMATH_CALUDE_deck_cost_is_32_l29_2979

/-- Calculates the total cost of Tom's deck of cards. -/
def deck_cost : ℝ :=
  let rare_count : ℕ := 19
  let uncommon_count : ℕ := 11
  let common_count : ℕ := 30
  let rare_cost : ℝ := 1
  let uncommon_cost : ℝ := 0.5
  let common_cost : ℝ := 0.25
  rare_count * rare_cost + uncommon_count * uncommon_cost + common_count * common_cost

/-- Proves that the total cost of Tom's deck is $32. -/
theorem deck_cost_is_32 : deck_cost = 32 := by
  sorry

end NUMINAMATH_CALUDE_deck_cost_is_32_l29_2979


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l29_2950

/-- A sequence where each term is twice the previous term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : a 1 + a 4 = 2) : 
  a 5 + a 8 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l29_2950


namespace NUMINAMATH_CALUDE_exists_functions_with_even_product_l29_2918

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define what it means for a function to be neither odd nor even
def NeitherOddNorEven (f : RealFunction) : Prop :=
  ¬(IsEven f) ∧ ¬(IsOdd f)

-- State the theorem
theorem exists_functions_with_even_product :
  ∃ (f g : RealFunction),
    NeitherOddNorEven f ∧
    NeitherOddNorEven g ∧
    IsEven (fun x ↦ f x * g x) :=
by sorry

end NUMINAMATH_CALUDE_exists_functions_with_even_product_l29_2918


namespace NUMINAMATH_CALUDE_m_less_than_n_l29_2955

theorem m_less_than_n (x : ℝ) : (x + 2) * (x + 3) < 2 * x^2 + 5 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l29_2955


namespace NUMINAMATH_CALUDE_unique_number_l29_2996

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- The first digit is 2
  (n / 100000 = 2) ∧
  -- Moving the first digit to the last position results in a number that is three times the original number
  (n % 100000 * 10 + 2 = 3 * n)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 285714 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l29_2996


namespace NUMINAMATH_CALUDE_max_value_problem_l29_2947

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l29_2947


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l29_2983

theorem longest_side_of_triangle (y : ℝ) : 
  10 + (y + 6) + (3*y + 2) = 45 →
  max 10 (max (y + 6) (3*y + 2)) = 22.25 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l29_2983


namespace NUMINAMATH_CALUDE_no_integer_roots_l29_2951

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l29_2951


namespace NUMINAMATH_CALUDE_nancy_savings_l29_2901

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (nancy_quarters : ℕ) (h1 : nancy_quarters = dozen) : 
  (nancy_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l29_2901


namespace NUMINAMATH_CALUDE_empty_pencil_cases_l29_2971

theorem empty_pencil_cases (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) (empty : ℕ) : 
  total = 10 ∧ pencils = 5 ∧ pens = 4 ∧ both = 2 → empty = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_empty_pencil_cases_l29_2971


namespace NUMINAMATH_CALUDE_final_salary_ratio_l29_2910

/-- Represents the sequence of salary adjustments throughout the year -/
def salary_adjustments : List (ℝ → ℝ) := [
  (· * 1.20),       -- 20% increase after 2 months
  (· * 0.90),       -- 10% decrease in 3rd month
  (· * 1.12),       -- 12% increase in 4th month
  (· * 0.92),       -- 8% decrease in 5th month
  (· * 1.12),       -- 12% increase in 6th month
  (· * 0.92),       -- 8% decrease in 7th month
  (· * 1.08),       -- 8% bonus in 8th month
  (· * 0.50),       -- 50% decrease due to financial crisis
  (· * 0.90),       -- 10% decrease in 9th month
  (· * 1.15),       -- 15% increase in 10th month
  (· * 0.90),       -- 10% decrease in 11th month
  (· * 1.50)        -- 50% increase in last month
]

/-- Applies a list of functions sequentially to an initial value -/
def apply_adjustments (adjustments : List (ℝ → ℝ)) (initial : ℝ) : ℝ :=
  adjustments.foldl (λ acc f => f acc) initial

/-- Theorem stating the final salary ratio after adjustments -/
theorem final_salary_ratio (S : ℝ) (hS : S > 0) :
  let initial_after_tax := 0.70 * S
  let final_salary := apply_adjustments salary_adjustments initial_after_tax
  ∃ ε > 0, abs (final_salary / initial_after_tax - 0.8657) < ε :=
sorry

end NUMINAMATH_CALUDE_final_salary_ratio_l29_2910


namespace NUMINAMATH_CALUDE_function_parity_and_ranges_l29_2913

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2*a*x - a

def F (x : ℝ) : ℝ := x * f a x

def g (x : ℝ) : ℝ := -Real.exp x

theorem function_parity_and_ranges :
  (∀ x, f a x = f a (-x) ↔ a = 0) ∧
  (∃ m₁ m₂, m₁ = -16/3 ∧ m₂ = 112/729 ∧
    ∀ m, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      F a x₁ = m ∧ F a x₂ = m ∧ F a x₃ = m) ↔ m₁ < m ∧ m < m₂) ∧
  (∃ a₁ a₂, a₁ = Real.log 2 - 1 ∧ a₂ = 1/2 ∧
    (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.exp 1 ∧ x₁ > x₂ →
      |f a x₁ - f a x₂| < |g x₁ - g x₂|) ↔ a₁ ≤ a ∧ a ≤ a₂) :=
by sorry

end

end NUMINAMATH_CALUDE_function_parity_and_ranges_l29_2913


namespace NUMINAMATH_CALUDE_function_simplification_l29_2915

theorem function_simplification (x : ℝ) :
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) +
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l29_2915


namespace NUMINAMATH_CALUDE_integer_sum_problem_l29_2970

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 12 → x * y = 45 → x + y = 18 := by sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l29_2970


namespace NUMINAMATH_CALUDE_special_action_figure_value_prove_special_figure_value_l29_2941

theorem special_action_figure_value 
  (total_figures : Nat) 
  (regular_figure_value : Nat) 
  (regular_figure_count : Nat) 
  (discount : Nat) 
  (total_earnings : Nat) : Nat :=
  let special_figure_count := total_figures - regular_figure_count
  let regular_figures_earnings := regular_figure_count * (regular_figure_value - discount)
  let special_figure_earnings := total_earnings - regular_figures_earnings
  special_figure_earnings + discount

theorem prove_special_figure_value :
  special_action_figure_value 5 15 4 5 55 = 20 := by
  sorry

end NUMINAMATH_CALUDE_special_action_figure_value_prove_special_figure_value_l29_2941


namespace NUMINAMATH_CALUDE_prob_two_even_correct_l29_2924

/-- The total number of balls -/
def total_balls : ℕ := 17

/-- The number of even-numbered balls -/
def even_balls : ℕ := 8

/-- The probability of drawing two even-numbered balls without replacement -/
def prob_two_even : ℚ := 7 / 34

theorem prob_two_even_correct :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = prob_two_even := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_correct_l29_2924


namespace NUMINAMATH_CALUDE_binomial_coefficient_bound_l29_2916

theorem binomial_coefficient_bound (k : ℕ) (p : ℕ) (x : ℤ) :
  Prime p →
  p = 4 * k + 1 →
  |x| ≤ (p - 1) / 2 →
  Nat.choose (2 * k) k ≡ x [ZMOD p] →
  |x| ≤ 2 * Real.sqrt p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_bound_l29_2916


namespace NUMINAMATH_CALUDE_oil_price_reduction_l29_2976

/-- Calculates the percentage reduction in oil price given the conditions --/
theorem oil_price_reduction (total_cost : ℝ) (additional_kg : ℝ) (reduced_price : ℝ) : 
  total_cost = 1100 ∧ 
  additional_kg = 5 ∧ 
  reduced_price = 55 →
  (((total_cost / (total_cost / reduced_price - additional_kg)) - reduced_price) / 
   (total_cost / (total_cost / reduced_price - additional_kg))) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l29_2976
