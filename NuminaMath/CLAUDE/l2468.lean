import Mathlib

namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_four_l2468_246812

-- Define the Fibonacci sequence modulo 7
def fibMod7 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fibMod7 n + fibMod7 (n + 1)) % 7

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod7 k = d

-- Define a function to check if all digits from 0 to 6 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 6 → digitAppeared d n

-- The main theorem
theorem last_digit_to_appear_is_four :
  ∃ n, allDigitsAppeared n ∧ ¬(digitAppeared 4 (n - 1)) :=
sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_four_l2468_246812


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2468_246872

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 6) → (abs b = 4) → (a * b < 0) → (a + b = 2 ∨ a + b = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2468_246872


namespace NUMINAMATH_CALUDE_escalator_length_l2468_246805

/-- The length of an escalator given two people walking in opposite directions -/
theorem escalator_length 
  (time_A : ℝ) 
  (time_B : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (h1 : time_A = 100) 
  (h2 : time_B = 300) 
  (h3 : speed_A = 3) 
  (h4 : speed_B = 2) : 
  (speed_A - speed_B) / (1 / time_A - 1 / time_B) = 150 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l2468_246805


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2468_246800

/-- Given a geometric sequence {a_n} with common ratio q = -2 and a_3 * a_7 = 4 * a_4,
    prove that the arithmetic mean of a_8 and a_11 is -56. -/
theorem geometric_sequence_arithmetic_mean 
  (a : ℕ → ℝ) -- The geometric sequence
  (h1 : ∀ n, a (n + 1) = a n * (-2)) -- Common ratio q = -2
  (h2 : a 3 * a 7 = 4 * a 4) -- Given condition
  : (a 8 + a 11) / 2 = -56 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2468_246800


namespace NUMINAMATH_CALUDE_C_formula_l2468_246809

/-- 
C(n, p) represents the number of decompositions of n into sums of powers of p, 
where each power p^k appears at most p^2 - 1 times
-/
def C (n p : ℕ) : ℕ := sorry

/-- Theorem stating the formula for C(n, p) -/
theorem C_formula (n p : ℕ) (hp : p > 1) : C n p = n / p + 1 := by sorry

end NUMINAMATH_CALUDE_C_formula_l2468_246809


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2468_246875

theorem smallest_candy_count : ∃ n : ℕ, 
  (n ≥ 100) ∧ (n < 1000) ∧ 
  ((n + 7) % 6 = 0) ∧ 
  ((n - 5) % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < n ∧ m < 1000 → 
    ((m + 7) % 6 ≠ 0) ∨ ((m - 5) % 9 ≠ 0)) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2468_246875


namespace NUMINAMATH_CALUDE_vanessa_record_score_l2468_246818

/-- Vanessa's record-setting basketball score --/
theorem vanessa_record_score (total_team_score : ℕ) (other_players : ℕ) (other_players_avg : ℚ)
  (h1 : total_team_score = 48)
  (h2 : other_players = 6)
  (h3 : other_players_avg = 3.5) :
  total_team_score - (other_players : ℚ) * other_players_avg = 27 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_record_score_l2468_246818


namespace NUMINAMATH_CALUDE_password_equation_l2468_246822

theorem password_equation : ∃ (A B C P Q R : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ P < 10 ∧ Q < 10 ∧ R < 10) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
   B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
   C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
   P ≠ Q ∧ P ≠ R ∧
   Q ≠ R) ∧
  3 * (100000 * A + 10000 * B + 1000 * C + 100 * P + 10 * Q + R) =
  4 * (100000 * P + 10000 * Q + 1000 * R + 100 * A + 10 * B + C) :=
by sorry

end NUMINAMATH_CALUDE_password_equation_l2468_246822


namespace NUMINAMATH_CALUDE_august_tips_fraction_l2468_246803

theorem august_tips_fraction (total_months : ℕ) (other_months : ℕ) (august_multiplier : ℕ) :
  total_months = other_months + 1 →
  august_multiplier = 8 →
  (august_multiplier : ℚ) / (august_multiplier + other_months : ℚ) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_fraction_l2468_246803


namespace NUMINAMATH_CALUDE_mentor_fraction_is_one_seventh_l2468_246885

/-- Represents the mentorship program in a school --/
structure MentorshipProgram where
  seventh_graders : ℕ
  tenth_graders : ℕ
  mentored_seventh : ℕ
  mentoring_tenth : ℕ

/-- Conditions of the mentorship program --/
def valid_program (p : MentorshipProgram) : Prop :=
  p.mentoring_tenth = p.mentored_seventh ∧
  4 * p.mentoring_tenth = p.tenth_graders ∧
  3 * p.mentored_seventh = p.seventh_graders

/-- The fraction of students with a mentor --/
def mentor_fraction (p : MentorshipProgram) : ℚ :=
  p.mentored_seventh / (p.seventh_graders + p.tenth_graders)

/-- Theorem stating that the fraction of students with a mentor is 1/7 --/
theorem mentor_fraction_is_one_seventh (p : MentorshipProgram) 
  (h : valid_program p) : mentor_fraction p = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mentor_fraction_is_one_seventh_l2468_246885


namespace NUMINAMATH_CALUDE_tangent_square_area_l2468_246860

/-- A square with two vertices on a circle and two on its tangent -/
structure TangentSquare where
  /-- The radius of the circle -/
  R : ℝ
  /-- The side length of the square -/
  x : ℝ
  /-- Two vertices of the square lie on the circle -/
  vertices_on_circle : x ≤ 2 * R
  /-- Two vertices of the square lie on the tangent -/
  vertices_on_tangent : x^2 / 4 = R^2 - (x - R)^2

/-- The area of a TangentSquare with radius 5 is 64 -/
theorem tangent_square_area (s : TangentSquare) (h : s.R = 5) : s.x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_tangent_square_area_l2468_246860


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2468_246898

theorem inequality_equivalence (x y : ℝ) : (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2468_246898


namespace NUMINAMATH_CALUDE_cube_sum_digits_equals_square_l2468_246825

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem cube_sum_digits_equals_square (n : ℕ) :
  n > 0 ∧ n < 1000 ∧ (sum_of_digits n)^3 = n^2 ↔ n = 1 ∨ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_digits_equals_square_l2468_246825


namespace NUMINAMATH_CALUDE_feline_sanctuary_count_l2468_246829

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 2
  lions + tigers + cougars = 39 :=
by sorry

end NUMINAMATH_CALUDE_feline_sanctuary_count_l2468_246829


namespace NUMINAMATH_CALUDE_lemon_juice_test_point_l2468_246838

theorem lemon_juice_test_point (lower_bound upper_bound : ℝ) 
  (h_lower : lower_bound = 500)
  (h_upper : upper_bound = 1500)
  (golden_ratio : ℝ) 
  (h_golden : golden_ratio = 0.618) : 
  let x₁ := lower_bound + golden_ratio * (upper_bound - lower_bound)
  let x₂ := upper_bound + lower_bound - x₁
  x₂ = 882 := by
sorry

end NUMINAMATH_CALUDE_lemon_juice_test_point_l2468_246838


namespace NUMINAMATH_CALUDE_product_of_sums_powers_l2468_246883

theorem product_of_sums_powers : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^6 + 1^6) = 2394400 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_l2468_246883


namespace NUMINAMATH_CALUDE_inscribed_prism_lateral_area_l2468_246876

theorem inscribed_prism_lateral_area (sphere_surface_area : ℝ) (prism_height : ℝ) :
  sphere_surface_area = 24 * Real.pi →
  prism_height = 4 →
  ∃ (prism_lateral_area : ℝ),
    prism_lateral_area = 32 ∧
    prism_lateral_area = 4 * prism_height * (Real.sqrt ((4 * sphere_surface_area / Real.pi) / 4 - prism_height^2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_prism_lateral_area_l2468_246876


namespace NUMINAMATH_CALUDE_nested_multiplication_l2468_246807

theorem nested_multiplication : 3 * (3 * (3 * (3 * (3 * (3 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end NUMINAMATH_CALUDE_nested_multiplication_l2468_246807


namespace NUMINAMATH_CALUDE_range_of_a_l2468_246808

theorem range_of_a (x a : ℝ) : 
  (∀ x, (-3 ≤ x ∧ x ≤ 3) ↔ x < a) →
  a ∈ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2468_246808


namespace NUMINAMATH_CALUDE_divisibility_condition_l2468_246874

theorem divisibility_condition (n : ℕ+) : 
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2468_246874


namespace NUMINAMATH_CALUDE_f_properties_l2468_246831

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 then Real.log (x^2 - 2*x + 2)
  else Real.log (x^2 + 2*x + 2)

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f x = f (-x)) ∧  -- f is even
  (∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2*x + 2)) ∧  -- expression for x < 0
  (StrictMonoOn f (Set.Ioo (-1) 0) ∧ StrictMonoOn f (Set.Ioi 1)) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2468_246831


namespace NUMINAMATH_CALUDE_rhombus_area_l2468_246813

/-- The area of a rhombus with side length 4 and an angle of 45 degrees between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → 
  angle = Real.pi / 4 → 
  (side * side * Real.sin angle : ℝ) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2468_246813


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l2468_246806

theorem correct_calculation : 2 * Real.sqrt 5 * Real.sqrt 5 = 10 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 2 + Real.sqrt 5 = Real.sqrt 7) :=
by sorry

theorem incorrect_calculation_B : ¬(2 * Real.sqrt 3 - Real.sqrt 3 = 2) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt (3^2 - 2^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l2468_246806


namespace NUMINAMATH_CALUDE_g_invertible_interval_l2468_246878

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- State the theorem
theorem g_invertible_interval :
  ∃ (a : ℝ), a ≤ 2 ∧
  (∀ x y, a ≤ x ∧ x < y → g x < g y) ∧
  (∀ b, b < a → ∃ x y, b ≤ x ∧ x < y ∧ y < a ∧ g x ≥ g y) :=
sorry

end NUMINAMATH_CALUDE_g_invertible_interval_l2468_246878


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2468_246861

/-- The number of games played in a chess tournament. -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a chess tournament with 25 players, where every player plays
    three times with each of their opponents, the total number of games is 900. -/
theorem chess_tournament_games :
  num_games 25 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2468_246861


namespace NUMINAMATH_CALUDE_solve_equation_binomial_identity_l2468_246852

-- Define A_x as the falling factorial
def A (x : ℕ) (n : ℕ) : ℕ := 
  if n ≤ x then
    (x - n + 1).factorial / (x - n).factorial
  else 0

-- Define binomial coefficient
def C (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then
    n.factorial / (k.factorial * (n - k).factorial)
  else 0

theorem solve_equation : ∃ x : ℕ, x > 3 ∧ 3 * A x 3 = 2 * A (x + 1) 2 + 6 * A x 2 ∧ x = 5 := by
  sorry

theorem binomial_identity (n k : ℕ) (h : k ≤ n) : k * C n k = n * C (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_binomial_identity_l2468_246852


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2468_246893

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : a 1 + a 3 + a 5 = 21) 
    (h3 : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2468_246893


namespace NUMINAMATH_CALUDE_james_hourly_rate_l2468_246833

/-- Represents the car rental scenario for James --/
structure CarRental where
  hours_per_day : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculates the hourly rate for car rental --/
def hourly_rate (rental : CarRental) : ℚ :=
  rental.weekly_income / (rental.hours_per_day * rental.days_per_week)

/-- Theorem stating that James' hourly rate is $20 --/
theorem james_hourly_rate :
  let james_rental : CarRental := {
    hours_per_day := 8,
    days_per_week := 4,
    weekly_income := 640
  }
  hourly_rate james_rental = 20 := by sorry

end NUMINAMATH_CALUDE_james_hourly_rate_l2468_246833


namespace NUMINAMATH_CALUDE_z_value_z_value_proof_l2468_246886

theorem z_value : ℝ → ℝ → ℝ → Prop :=
  fun x y z =>
    x = 40 * (1 + 0.2) →
    y = x * (1 - 0.35) →
    z = (x + y) / 2 →
    z = 39.6

-- Proof
theorem z_value_proof : ∃ x y z : ℝ, z_value x y z := by
  sorry

end NUMINAMATH_CALUDE_z_value_z_value_proof_l2468_246886


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l2468_246848

theorem no_two_cubes_between_squares : ¬∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l2468_246848


namespace NUMINAMATH_CALUDE_h_x_equality_l2468_246873

theorem h_x_equality (x : ℝ) (h : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 + h x = 7 * x^3 - 5 * x^2 + 9 * x + 3) → 
  (h x = -2 * x^5 + 3 * x^3 - 5 * x^2 + 9 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_h_x_equality_l2468_246873


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l2468_246842

theorem a_minus_b_equals_two (a b : ℝ) 
  (eq1 : 4 * a + 3 * b = 8) 
  (eq2 : 3 * a + 4 * b = 6) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l2468_246842


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l2468_246819

def arrangement_count : ℕ := 
  (Finset.range 3).sum (λ i =>
    Nat.choose 4 i * Nat.choose 6 (i + 2) * Nat.choose 5 i)

theorem correct_arrangement_count : arrangement_count = 1315 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l2468_246819


namespace NUMINAMATH_CALUDE_increase_data_effect_l2468_246867

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- Represents the operation of increasing each data point by a fixed value -/
def increase_data (d : DataSet) (inc : ℝ) : DataSet :=
  { average := d.average + inc, variance := d.variance }

/-- Theorem stating the effect of increasing each data point on the average and variance -/
theorem increase_data_effect (d : DataSet) (inc : ℝ) :
  d.average = 2 ∧ d.variance = 3 ∧ inc = 60 →
  (increase_data d inc).average = 62 ∧ (increase_data d inc).variance = 3 := by
  sorry

end NUMINAMATH_CALUDE_increase_data_effect_l2468_246867


namespace NUMINAMATH_CALUDE_bills_age_l2468_246821

/-- Proves Bill's age given the conditions of the problem -/
theorem bills_age :
  ∀ (bill_age caroline_age : ℕ),
    bill_age = 2 * caroline_age - 1 →
    bill_age + caroline_age = 26 →
    bill_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_bills_age_l2468_246821


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_two_thirds_l2468_246896

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a given LargeCube -/
def white_surface_fraction (c : LargeCube) : ℚ :=
  -- The actual calculation is not implemented here
  0

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , small_cube_count := 64
  , white_cube_count := 30
  , black_cube_count := 34 }

theorem white_surface_fraction_is_two_thirds :
  white_surface_fraction problem_cube = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_two_thirds_l2468_246896


namespace NUMINAMATH_CALUDE_stating_b_joined_after_five_months_l2468_246884

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment -/
def aInvestment : ℕ := 3500

/-- Represents B's investment -/
def bInvestment : ℕ := 9000

/-- Represents the profit ratio for A -/
def aProfitRatio : ℕ := 2

/-- Represents the profit ratio for B -/
def bProfitRatio : ℕ := 3

/-- 
Theorem stating that B joined 5 months after A started the business,
given the conditions of the problem.
-/
theorem b_joined_after_five_months :
  ∀ (x : ℕ),
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = aProfitRatio / bProfitRatio →
  x = 5 := by
  sorry


end NUMINAMATH_CALUDE_stating_b_joined_after_five_months_l2468_246884


namespace NUMINAMATH_CALUDE_no_negative_roots_l2468_246862

theorem no_negative_roots : ∀ x : ℝ, x < 0 → 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l2468_246862


namespace NUMINAMATH_CALUDE_business_investment_proof_l2468_246824

/-- Praveen's initial investment in the business -/
def praveenInvestment : ℕ := 35280

/-- Hari's investment in the business -/
def hariInvestment : ℕ := 10080

/-- Praveen's investment duration in months -/
def praveenDuration : ℕ := 12

/-- Hari's investment duration in months -/
def hariDuration : ℕ := 7

/-- Praveen's share in the profit ratio -/
def praveenShare : ℕ := 2

/-- Hari's share in the profit ratio -/
def hariShare : ℕ := 3

theorem business_investment_proof :
  praveenInvestment * praveenDuration * hariShare = 
  hariInvestment * hariDuration * praveenShare := by
  sorry

end NUMINAMATH_CALUDE_business_investment_proof_l2468_246824


namespace NUMINAMATH_CALUDE_function_value_at_sqrt_two_l2468_246854

/-- Given a function f : ℝ → ℝ satisfying the equation 2 * f x + f (x^2 - 1) = 1 for all real x,
    prove that f(√2) = 1/3 -/
theorem function_value_at_sqrt_two (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x + f (x^2 - 1) = 1) : 
    f (Real.sqrt 2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_sqrt_two_l2468_246854


namespace NUMINAMATH_CALUDE_pear_count_theorem_l2468_246890

/-- Represents the types of fruits on the table -/
inductive Fruit
  | Apple
  | Pear
  | Orange

/-- Represents the state of the table -/
structure TableState where
  apples : Nat
  pears : Nat
  oranges : Nat

/-- Defines the order in which fruits are taken -/
def nextFruit : Fruit → Fruit
  | Fruit.Apple => Fruit.Pear
  | Fruit.Pear => Fruit.Orange
  | Fruit.Orange => Fruit.Apple

/-- Determines if a fruit can be taken from the table -/
def canTakeFruit (state : TableState) (fruit : Fruit) : Bool :=
  match fruit with
  | Fruit.Apple => state.apples > 0
  | Fruit.Pear => state.pears > 0
  | Fruit.Orange => state.oranges > 0

/-- Takes a fruit from the table -/
def takeFruit (state : TableState) (fruit : Fruit) : TableState :=
  match fruit with
  | Fruit.Apple => { state with apples := state.apples - 1 }
  | Fruit.Pear => { state with pears := state.pears - 1 }
  | Fruit.Orange => { state with oranges := state.oranges - 1 }

/-- Checks if the table is empty -/
def isTableEmpty (state : TableState) : Bool :=
  state.apples = 0 && state.pears = 0 && state.oranges = 0

/-- Main theorem: The number of pears must be either 99 or 100 -/
theorem pear_count_theorem (initialPears : Nat) :
  let initialState : TableState := { apples := 100, pears := initialPears, oranges := 99 }
  (∃ (finalState : TableState), 
    isTableEmpty finalState ∧
    (∀ fruit : Fruit, canTakeFruit initialState fruit →
      ∃ nextState : TableState, 
        nextState = takeFruit initialState fruit ∧
        (isTableEmpty nextState ∨ 
          canTakeFruit nextState (nextFruit fruit)))) →
  initialPears = 99 ∨ initialPears = 100 := by
  sorry


end NUMINAMATH_CALUDE_pear_count_theorem_l2468_246890


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2468_246891

theorem quadratic_equation_result (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : 
  (8 * y - 4)^2 = 202 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2468_246891


namespace NUMINAMATH_CALUDE_rotated_line_equation_l2468_246855

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a line 90 degrees counterclockwise around a given point -/
def rotateLine90 (l : Line) (p : Point) : Line :=
  sorry

/-- The original line l₀ -/
def l₀ : Line :=
  { slope := 1, yIntercept := 1 }

/-- The point P around which the line is rotated -/
def P : Point :=
  { x := 3, y := 1 }

/-- The resulting line l after rotation -/
def l : Line :=
  rotateLine90 l₀ P

theorem rotated_line_equation :
  l.slope * P.x + l.yIntercept = P.y ∧ l.slope = -1 →
  ∀ x y, y + x - 4 = 0 ↔ y = l.slope * x + l.yIntercept :=
sorry

end NUMINAMATH_CALUDE_rotated_line_equation_l2468_246855


namespace NUMINAMATH_CALUDE_division_problem_l2468_246881

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2468_246881


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_six_l2468_246888

theorem arctan_sum_equals_pi_over_six (b : ℝ) :
  (4/3 : ℝ) * (b + 1) = 3/2 →
  Real.arctan (1/3) + Real.arctan b = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_six_l2468_246888


namespace NUMINAMATH_CALUDE_refrigerator_cash_savings_l2468_246877

/-- Calculates the savings when buying a refrigerator with cash instead of installments. -/
theorem refrigerator_cash_savings 
  (cash_price : ℕ) 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (h1 : cash_price = 8000)
  (h2 : deposit = 3000)
  (h3 : num_installments = 30)
  (h4 : installment_amount = 300) :
  deposit + num_installments * installment_amount - cash_price = 4000 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_cash_savings_l2468_246877


namespace NUMINAMATH_CALUDE_slope_angle_of_line_through_origin_and_unit_point_l2468_246836

/-- The slope angle of a line passing through (0,0) and (1,1) is π/4 -/
theorem slope_angle_of_line_through_origin_and_unit_point :
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (1, 1)
  let slope : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_through_origin_and_unit_point_l2468_246836


namespace NUMINAMATH_CALUDE_opposite_number_any_real_l2468_246830

theorem opposite_number_any_real (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x + y = 0) → 
  (∃ b : ℝ, a + b = 0 ∧ b = -a) → 
  True :=
by sorry

end NUMINAMATH_CALUDE_opposite_number_any_real_l2468_246830


namespace NUMINAMATH_CALUDE_math_marks_calculation_l2468_246802

theorem math_marks_calculation (english physics chemistry biology : ℕ)
  (average : ℕ) (total_subjects : ℕ) (h1 : english = 73)
  (h2 : physics = 92) (h3 : chemistry = 64) (h4 : biology = 82)
  (h5 : average = 76) (h6 : total_subjects = 5) :
  average * total_subjects - (english + physics + chemistry + biology) = 69 :=
by sorry

end NUMINAMATH_CALUDE_math_marks_calculation_l2468_246802


namespace NUMINAMATH_CALUDE_barneys_inventory_l2468_246839

/-- The number of items left in Barney's grocery store -/
def items_left (restocked : ℕ) (sold : ℕ) (in_storeroom : ℕ) : ℕ :=
  (restocked - sold) + in_storeroom

/-- Theorem stating the total number of items left in Barney's grocery store -/
theorem barneys_inventory : items_left 4458 1561 575 = 3472 := by
  sorry

end NUMINAMATH_CALUDE_barneys_inventory_l2468_246839


namespace NUMINAMATH_CALUDE_triangle_area_l2468_246866

/-- The area of a triangle with vertices at (5, -2), (5, 8), and (12, 8) is 35 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (5, -2)
  let v2 : ℝ × ℝ := (5, 8)
  let v3 : ℝ × ℝ := (12, 8)
  let area := (1/2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))
  area = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2468_246866


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2468_246897

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : team_avg_age = 21)
  (h4 : ∃ (remaining_avg_age : ℕ), remaining_avg_age = team_avg_age - 1 ∧ 
    (team_size - 2) * remaining_avg_age + captain_age + (captain_age + x) = team_size * team_avg_age) :
  x = 3 :=
sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2468_246897


namespace NUMINAMATH_CALUDE_min_value_fraction_l2468_246857

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem min_value_fraction (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, a m * a n = 16 * (a 1)^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  (∃ m n : ℕ, 1 / m + 9 / n = 11 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2468_246857


namespace NUMINAMATH_CALUDE_distinct_sums_l2468_246835

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  hd : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * seq.a 1 + (n * (n - 1) : ℚ) / 2 * seq.d

theorem distinct_sums (seq : ArithmeticSequence) 
  (h_sum_5 : S seq 5 = 0) : 
  Finset.card (Finset.image (S seq) (Finset.range 100)) = 98 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_l2468_246835


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l2468_246851

/-- Calculates the percentage decrease in salary after an initial increase -/
theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (initial_increase_percentage : ℝ) 
  (final_salary : ℝ) 
  (h1 : original_salary = 1000.0000000000001)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 1045) :
  let increased_salary := original_salary * (1 + initial_increase_percentage / 100)
  let decrease_percentage := (1 - final_salary / increased_salary) * 100
  decrease_percentage = 5 := by
sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l2468_246851


namespace NUMINAMATH_CALUDE_chess_tournament_players_l2468_246892

theorem chess_tournament_players :
  ∀ (num_girls : ℕ) (total_points : ℕ),
    (num_girls > 0) →
    (total_points = 2 * num_girls * (6 * num_girls - 1)) →
    (2 * num_girls * (6 * num_girls - 1) = (num_girls^2 + 9*num_girls) + 2*(5*num_girls)*(5*num_girls - 1)) →
    (num_girls + 5*num_girls = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l2468_246892


namespace NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l2468_246827

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, a^2 + 2*a + |1 + x| > f x} = {a : ℝ | a < -3 ∨ a > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_4_range_of_a_l2468_246827


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2468_246856

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ y : ℝ, 2 * x^2 + 6 * x - 5 ≥ 2 * min_x^2 + 6 * min_x - 5 ∧ min_x = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2468_246856


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2468_246804

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (f = fun x ↦ x - 1) ∨ (f = fun x ↦ -x - 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2468_246804


namespace NUMINAMATH_CALUDE_set_difference_M_N_range_of_a_l2468_246801

-- Define set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - 1)}
def N : Set ℝ := {y | ∃ x, y = 1 - x^2}

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < a*x - 1 ∧ a*x - 1 ≤ 5}
def B : Set ℝ := {y | -1/2 < y ∧ y ≤ 2}

-- Theorem 1
theorem set_difference_M_N : set_difference M N = {x | x > 1} := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : set_difference (A a) B = ∅ → a < -12 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_set_difference_M_N_range_of_a_l2468_246801


namespace NUMINAMATH_CALUDE_bob_marathon_preparation_l2468_246859

/-- The total miles Bob runs in 3 days -/
def total_miles : ℝ := 70

/-- Miles run on day one -/
def day_one_miles : ℝ := 0.2 * total_miles

/-- Miles run on day two -/
def day_two_miles : ℝ := 0.5 * (total_miles - day_one_miles)

/-- Miles run on day three -/
def day_three_miles : ℝ := 28

theorem bob_marathon_preparation :
  day_one_miles + day_two_miles + day_three_miles = total_miles :=
by sorry

end NUMINAMATH_CALUDE_bob_marathon_preparation_l2468_246859


namespace NUMINAMATH_CALUDE_quadratic_factoring_l2468_246837

/-- 
Given a quadratic function y = x^2 - 2x + 3, 
prove that it is equivalent to y = (x - 1)^2 + 2 
when factored into the form y = (x - h)^2 + k
-/
theorem quadratic_factoring (x : ℝ) : 
  x^2 - 2*x + 3 = (x - 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factoring_l2468_246837


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2468_246844

/-- Proves that given the conditions of ticket prices, total revenue, and total tickets sold,
    the number of adult tickets sold is 22. -/
theorem adult_tickets_sold (adult_price child_price total_revenue total_tickets : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end NUMINAMATH_CALUDE_adult_tickets_sold_l2468_246844


namespace NUMINAMATH_CALUDE_f_equiv_g_l2468_246817

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

-- Theorem stating that f and g are equivalent for all real numbers
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equiv_g_l2468_246817


namespace NUMINAMATH_CALUDE_gcd_f_x_eq_one_l2468_246871

def f (x : ℤ) : ℤ := (3*x+4)*(8*x+5)*(15*x+11)*(x+14)

theorem gcd_f_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 54321 * k) :
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_f_x_eq_one_l2468_246871


namespace NUMINAMATH_CALUDE_complex_square_second_quadrant_l2468_246865

/-- Given that (1+2i)^2 = a+bi where a and b are real numbers,
    prove that the point P(a,b) lies in the second quadrant. -/
theorem complex_square_second_quadrant :
  ∃ (a b : ℝ), (1 + 2 * Complex.I) ^ 2 = a + b * Complex.I ∧
  a < 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_second_quadrant_l2468_246865


namespace NUMINAMATH_CALUDE_class_size_l2468_246810

theorem class_size (N M S : ℕ) 
  (h1 : N - M = 10)
  (h2 : N - S = 15)
  (h3 : N - (M + S - 7) = 2)
  (h4 : M + S = N + 7) : N = 34 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2468_246810


namespace NUMINAMATH_CALUDE_tree_height_calculation_l2468_246863

/-- Given a tree and a pole with their respective shadows, calculate the height of the tree -/
theorem tree_height_calculation (tree_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ) :
  tree_shadow = 30 →
  pole_height = 1.5 →
  pole_shadow = 3 →
  (tree_shadow * pole_height) / pole_shadow = 15 :=
by sorry

end NUMINAMATH_CALUDE_tree_height_calculation_l2468_246863


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2468_246846

theorem no_positive_integer_solutions :
  ¬ ∃ (x y z : ℕ+), x^4004 + y^4004 = z^2002 :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2468_246846


namespace NUMINAMATH_CALUDE_twenty_people_handshakes_l2468_246895

/-- The number of handshakes when n people shake hands with each other exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 people, the total number of handshakes is 190. -/
theorem twenty_people_handshakes :
  handshakes 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_twenty_people_handshakes_l2468_246895


namespace NUMINAMATH_CALUDE_average_weight_increase_l2468_246858

/-- Proves that replacing a person weighing 65 kg with a person weighing 105 kg
    in a group of 8 people increases the average weight by 5 kg. -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 105
  let new_average := new_total / 8
  new_average - initial_average = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2468_246858


namespace NUMINAMATH_CALUDE_polynomials_common_factor_l2468_246845

def p1 (x : ℝ) : ℝ := 16 * x^5 - x
def p2 (x : ℝ) : ℝ := (x - 1)^2 - 4 * (x - 1) + 4
def p3 (x : ℝ) : ℝ := (x + 1)^2 - 4 * x * (x + 1) + 4 * x^2
def p4 (x : ℝ) : ℝ := -4 * x^2 - 1 + 4 * x

theorem polynomials_common_factor :
  ∃ (f : ℝ → ℝ) (g1 g4 : ℝ → ℝ),
    (∀ x, p1 x = f x * g1 x) ∧
    (∀ x, p4 x = f x * g4 x) ∧
    (∀ x, f x ≠ 0) ∧
    (∀ x, f x ≠ 1) ∧
    (∀ x, f x ≠ -1) ∧
    (∀ (h2 h3 : ℝ → ℝ),
      (∀ x, p2 x ≠ f x * h2 x) ∧
      (∀ x, p3 x ≠ f x * h3 x)) :=
by sorry

end NUMINAMATH_CALUDE_polynomials_common_factor_l2468_246845


namespace NUMINAMATH_CALUDE_division_remainder_l2468_246811

theorem division_remainder (j : ℕ) (h1 : j > 0) (h2 : 132 % (j^2) = 12) : 250 % j = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2468_246811


namespace NUMINAMATH_CALUDE_bus_problem_solution_l2468_246823

/-- Represents the problem of distributing passengers among buses --/
structure BusProblem where
  m : ℕ  -- Initial number of buses
  n : ℕ  -- Number of passengers per bus after redistribution
  initialPassengers : ℕ  -- Initial number of passengers per bus
  maxCapacity : ℕ  -- Maximum capacity of each bus

/-- The conditions of the bus problem --/
def validBusProblem (bp : BusProblem) : Prop :=
  bp.m ≥ 2 ∧
  bp.initialPassengers = 22 ∧
  bp.maxCapacity = 32 ∧
  bp.n ≤ bp.maxCapacity ∧
  bp.initialPassengers * bp.m + 1 = bp.n * (bp.m - 1)

/-- The theorem stating the solution to the bus problem --/
theorem bus_problem_solution (bp : BusProblem) (h : validBusProblem bp) :
  bp.m = 24 ∧ bp.n * (bp.m - 1) = 529 := by
  sorry

#check bus_problem_solution

end NUMINAMATH_CALUDE_bus_problem_solution_l2468_246823


namespace NUMINAMATH_CALUDE_no_integer_solution_l2468_246826

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 2 * x + 6 * y = 91 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2468_246826


namespace NUMINAMATH_CALUDE_odd_divisibility_l2468_246869

theorem odd_divisibility (n : ℕ) (h : Odd n) : n ∣ (2^(n.factorial) - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_divisibility_l2468_246869


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2468_246828

theorem magnitude_of_z (z : ℂ) (h : z^2 = 3 - 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2468_246828


namespace NUMINAMATH_CALUDE_prob_rain_sunday_and_monday_l2468_246815

-- Define the probabilities
def prob_rain_saturday : ℝ := 0.8
def prob_rain_sunday : ℝ := 0.3
def prob_rain_monday_if_sunday : ℝ := 0.5
def prob_rain_monday_if_not_sunday : ℝ := 0.1

-- Define the independence of Saturday and Sunday
axiom saturday_sunday_independent : True

-- Theorem to prove
theorem prob_rain_sunday_and_monday : 
  prob_rain_sunday * prob_rain_monday_if_sunday = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_sunday_and_monday_l2468_246815


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l2468_246894

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes for assigning 3 people to 7 communities with at most 2 people per community. -/
def totalAllocationSchemes : ℕ := A 7 3 + C 3 2 * C 1 1 * A 7 2

theorem allocation_schemes_count :
  totalAllocationSchemes = 336 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l2468_246894


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l2468_246832

/-- The price of the book in cents -/
def book_price : ℕ := 4250

/-- The number of $10 bills Jane has -/
def ten_dollar_bills : ℕ := 4

/-- The number of quarters Jane has -/
def quarters : ℕ := 5

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The minimum number of nickels Jane needs to afford the book -/
def min_nickels : ℕ := 25

theorem minimum_nickels_needed :
  ∀ n : ℕ,
  (ten_dollar_bills * 1000 + quarters * 25 + n * nickel_value ≥ book_price) →
  (n ≥ min_nickels) :=
sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l2468_246832


namespace NUMINAMATH_CALUDE_smallest_even_natural_with_properties_l2468_246843

def is_smallest_even_natural_with_properties (a : ℕ) : Prop :=
  Even a ∧
  (∃ k₁, a + 1 = 3 * k₁) ∧
  (∃ k₂, a + 2 = 5 * k₂) ∧
  (∃ k₃, a + 3 = 7 * k₃) ∧
  (∃ k₄, a + 4 = 11 * k₄) ∧
  (∃ k₅, a + 5 = 13 * k₅) ∧
  (∀ b < a, ¬(is_smallest_even_natural_with_properties b))

theorem smallest_even_natural_with_properties : 
  is_smallest_even_natural_with_properties 788 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_natural_with_properties_l2468_246843


namespace NUMINAMATH_CALUDE_quadrilateral_inscribed_circle_l2468_246889

-- Define the types for points and circles
variable (Point : Type) (Circle : Type)

-- Define the necessary geometric predicates
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (on_segment : Point → Point → Point → Prop)
variable (intersection : Point → Point → Point → Point → Point → Prop)
variable (has_inscribed_circle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem quadrilateral_inscribed_circle 
  (A B C D E F G H P : Point) :
  is_convex_quadrilateral A B C D →
  on_segment A B E →
  on_segment B C F →
  on_segment C D G →
  on_segment D A H →
  intersection E G F H P →
  has_inscribed_circle H A E P →
  has_inscribed_circle E B F P →
  has_inscribed_circle F C G P →
  has_inscribed_circle G D H P →
  has_inscribed_circle A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inscribed_circle_l2468_246889


namespace NUMINAMATH_CALUDE_smoothie_ingredients_sum_l2468_246868

/-- The amount of strawberries used in cups -/
def strawberries : ℝ := 0.2

/-- The amount of yogurt used in cups -/
def yogurt : ℝ := 0.1

/-- The amount of orange juice used in cups -/
def orange_juice : ℝ := 0.2

/-- The total amount of ingredients used for the smoothies -/
def total_ingredients : ℝ := strawberries + yogurt + orange_juice

theorem smoothie_ingredients_sum :
  total_ingredients = 0.5 := by sorry

end NUMINAMATH_CALUDE_smoothie_ingredients_sum_l2468_246868


namespace NUMINAMATH_CALUDE_sequence_equality_l2468_246814

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l2468_246814


namespace NUMINAMATH_CALUDE_min_value_problem_l2468_246820

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2468_246820


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2468_246880

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2468_246880


namespace NUMINAMATH_CALUDE_polynomial_divisibility_condition_l2468_246849

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Definition of divisibility for integers -/
def divides (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

/-- Definition of an odd prime number -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

/-- The main theorem -/
theorem polynomial_divisibility_condition (f : IntPolynomial) :
  (∀ p : ℕ, is_odd_prime p → divides (f.eval p) ((p - 3).factorial + (p + 1) / 2)) →
  (f = Polynomial.X) ∨ (f = -Polynomial.X) ∨ (f = Polynomial.C 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_condition_l2468_246849


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l2468_246834

theorem trig_sum_equals_one :
  let angle_to_real (θ : ℤ) : ℝ := (θ % 360 : ℝ) * Real.pi / 180
  Real.sin (angle_to_real (-120)) * Real.cos (angle_to_real 1290) +
  Real.cos (angle_to_real (-1020)) * Real.sin (angle_to_real (-1050)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l2468_246834


namespace NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2468_246847

/-- Calculates the total number of dogwood trees in the park after a given number of years -/
def total_dogwood_trees (initial_trees : ℕ) (trees_today : ℕ) (trees_tomorrow : ℕ) 
  (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + 
  (trees_today + growth_rate_today * years) + 
  (trees_tomorrow + growth_rate_tomorrow * years)

/-- Theorem stating that the total number of dogwood trees after 5 years is 130 -/
theorem dogwood_trees_after_five_years : 
  total_dogwood_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval total_dogwood_trees 39 41 20 2 4 5

end NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l2468_246847


namespace NUMINAMATH_CALUDE_sara_red_balloons_l2468_246899

def total_red_balloons : ℕ := 55
def sandy_red_balloons : ℕ := 24

theorem sara_red_balloons : ∃ (sara_balloons : ℕ), 
  sara_balloons + sandy_red_balloons = total_red_balloons ∧ sara_balloons = 31 := by
  sorry

end NUMINAMATH_CALUDE_sara_red_balloons_l2468_246899


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2468_246864

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(5 * m : ℤ) ≡ 1978 [ZMOD 26]) ∧ 
  (5 * n : ℤ) ≡ 1978 [ZMOD 26] ↔ 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2468_246864


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l2468_246870

theorem triangle_inequality_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  3 ≤ (a / (b + c - a)).sqrt + (b / (a + c - b)).sqrt + (c / (a + b - c)).sqrt :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l2468_246870


namespace NUMINAMATH_CALUDE_computer_price_proof_l2468_246841

theorem computer_price_proof (P : ℝ) : 
  1.20 * P = 351 → 2 * P = 585 → P = 292.50 := by sorry

end NUMINAMATH_CALUDE_computer_price_proof_l2468_246841


namespace NUMINAMATH_CALUDE_exists_k_greater_than_two_l2468_246882

/-- Given a linear function y = (k-2)x + 3 that is increasing,
    prove that there exists a value of k greater than 2. -/
theorem exists_k_greater_than_two (k : ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k - 2) * x₁ + 3 < (k - 2) * x₂ + 3) : 
  ∃ k' : ℝ, k' > 2 := by
sorry

end NUMINAMATH_CALUDE_exists_k_greater_than_two_l2468_246882


namespace NUMINAMATH_CALUDE_tan_equation_solutions_l2468_246879

open Real

noncomputable def S (x : ℝ) := tan x + x

theorem tan_equation_solutions :
  let a := arctan 500
  ∃ (sols : Finset ℝ), (∀ x ∈ sols, 0 ≤ x ∧ x ≤ a ∧ tan x = tan (S x)) ∧ Finset.card sols = 160 :=
sorry

end NUMINAMATH_CALUDE_tan_equation_solutions_l2468_246879


namespace NUMINAMATH_CALUDE_circle_no_intersection_probability_l2468_246887

/-- A rectangle with width 15 and height 36 -/
structure Rectangle where
  width : ℝ := 15
  height : ℝ := 36

/-- A circle with radius 1 -/
structure Circle where
  radius : ℝ := 1

/-- The probability that a circle doesn't intersect the diagonal of a rectangle -/
def probability_no_intersection (r : Rectangle) (c : Circle) : ℚ :=
  375 / 442

/-- Theorem stating the probability of no intersection -/
theorem circle_no_intersection_probability (r : Rectangle) (c : Circle) :
  probability_no_intersection r c = 375 / 442 := by
  sorry

#check circle_no_intersection_probability

end NUMINAMATH_CALUDE_circle_no_intersection_probability_l2468_246887


namespace NUMINAMATH_CALUDE_initial_bench_press_weight_l2468_246850

/-- The initial bench press weight before injury -/
def W : ℝ := 500

/-- The bench press weight after injury -/
def after_injury : ℝ := 0.2 * W

/-- The bench press weight after training -/
def after_training : ℝ := 3 * after_injury

/-- The final bench press weight -/
def final_weight : ℝ := 300

theorem initial_bench_press_weight :
  W = 500 ∧ after_injury = 0.2 * W ∧ after_training = 3 * after_injury ∧ after_training = final_weight := by
  sorry

end NUMINAMATH_CALUDE_initial_bench_press_weight_l2468_246850


namespace NUMINAMATH_CALUDE_volume_Q4_l2468_246840

/-- Represents the volume of the i-th polyhedron in the sequence --/
def Q (i : ℕ) : ℝ :=
  sorry

/-- The volume difference between consecutive polyhedra --/
def ΔQ (i : ℕ) : ℝ :=
  sorry

theorem volume_Q4 :
  Q 0 = 8 →
  (∀ i : ℕ, ΔQ (i + 1) = (1 / 2) * ΔQ i) →
  ΔQ 1 = 4 →
  Q 4 = 15.5 :=
by
  sorry

end NUMINAMATH_CALUDE_volume_Q4_l2468_246840


namespace NUMINAMATH_CALUDE_definite_integral_result_l2468_246853

theorem definite_integral_result : 
  ∫ x in -Real.arcsin (2 / Real.sqrt 5)..π/4, (2 - Real.tan x) / (Real.sin x + 3 * Real.cos x)^2 = 15/4 - Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_result_l2468_246853


namespace NUMINAMATH_CALUDE_max_p_value_l2468_246816

theorem max_p_value (p q r : ℝ) (sum_eq : p + q + r = 10) (prod_sum_eq : p*q + p*r + q*r = 25) :
  p ≤ 20/3 ∧ ∃ q r : ℝ, p = 20/3 ∧ p + q + r = 10 ∧ p*q + p*r + q*r = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_p_value_l2468_246816
