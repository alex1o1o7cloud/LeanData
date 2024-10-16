import Mathlib

namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l2040_204047

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  initial_balance > 0 →
  let remaining_balance := initial_balance - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 →
  400 / initial_balance = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l2040_204047


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2040_204026

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 7 + 1
  let expr := (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3))
  expr = x - 2 ∧ expr = Real.sqrt 7 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2040_204026


namespace NUMINAMATH_CALUDE_function_value_at_two_l2040_204045

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 + 1) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2040_204045


namespace NUMINAMATH_CALUDE_crystal_lake_trail_length_l2040_204057

/-- Represents the Crystal Lake Trail hike --/
structure CrystalLakeTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- Conditions of the Crystal Lake Trail hike --/
def hikingConditions (hike : CrystalLakeTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day3 + hike.day4 + hike.day5 = 42 ∧
  hike.day1 + hike.day4 = 30

/-- Theorem stating that the total length of the Crystal Lake Trail is 70 miles --/
theorem crystal_lake_trail_length 
  (hike : CrystalLakeTrail) 
  (h : hikingConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_crystal_lake_trail_length_l2040_204057


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2040_204070

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 ≥ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → 2 ≤ x ∧ x ≤ 3 :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, q x ∧ ¬(p x a)) → 1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2040_204070


namespace NUMINAMATH_CALUDE_gcd_of_factorials_l2040_204023

theorem gcd_of_factorials : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_factorials_l2040_204023


namespace NUMINAMATH_CALUDE_product_modulo_300_l2040_204055

theorem product_modulo_300 : (2025 * 1233) % 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_300_l2040_204055


namespace NUMINAMATH_CALUDE_g_of_4_equals_18_l2040_204025

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_4_equals_18 : g 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_4_equals_18_l2040_204025


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2040_204061

def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2040_204061


namespace NUMINAMATH_CALUDE_sequence_always_terminates_l2040_204019

def last_digit (n : ℕ) : ℕ := n % 10

def next_term (n : ℕ) : ℕ :=
  if last_digit n ≤ 5 then n / 10 else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate next_term k a₀) = 0

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

end NUMINAMATH_CALUDE_sequence_always_terminates_l2040_204019


namespace NUMINAMATH_CALUDE_inequality_proof_l2040_204066

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a / (b^2 + 4)) + (b / (a^2 + 4)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2040_204066


namespace NUMINAMATH_CALUDE_pi_estimation_l2040_204038

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate := 4 * (m / n + 1 / 2)
  π_estimate = 47 / 15 := by
  sorry

end NUMINAMATH_CALUDE_pi_estimation_l2040_204038


namespace NUMINAMATH_CALUDE_stone_bucket_probability_l2040_204012

/-- The probability of having exactly k stones in the bucket after n seconds -/
def f (n k : ℕ) : ℚ :=
  (↑(Nat.floor ((n - k : ℤ) / 2)) : ℚ) / 2^n

/-- The main theorem stating the probability of having 1337 stones after 2017 seconds -/
theorem stone_bucket_probability : f 2017 1337 = 340 / 2^2017 := by sorry

end NUMINAMATH_CALUDE_stone_bucket_probability_l2040_204012


namespace NUMINAMATH_CALUDE_parabola_equation_l2040_204069

/-- A parabola with vertex at the origin and focus on the positive x-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : 0 < p
  h_focus : focus = (p / 2, 0)

/-- Two points on the parabola -/
structure ParabolaPoints (C : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola : A.2^2 = 2 * C.p * A.1 ∧ B.2^2 = 2 * C.p * B.1
  h_line_through_focus : ∃ k : ℝ, A.2 = k * (A.1 - C.p / 2) ∧ B.2 = k * (B.1 - C.p / 2)

/-- The dot product condition -/
def dot_product_condition (C : Parabola) (P : ParabolaPoints C) : Prop :=
  P.A.1 * P.B.1 + P.A.2 * P.B.2 = -12

/-- The main theorem -/
theorem parabola_equation (C : Parabola) (P : ParabolaPoints C)
  (h_dot : dot_product_condition C P) :
  C.p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2040_204069


namespace NUMINAMATH_CALUDE_positive_function_from_condition_l2040_204049

theorem positive_function_from_condition (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x : ℝ, f x + x * deriv f x > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_function_from_condition_l2040_204049


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2040_204073

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2040_204073


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l2040_204078

theorem candle_burning_theorem (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ n * k = n * (n + 1) / 2) → Odd n :=
by
  sorry

#check candle_burning_theorem

end NUMINAMATH_CALUDE_candle_burning_theorem_l2040_204078


namespace NUMINAMATH_CALUDE_complex_modulus_l2040_204094

theorem complex_modulus (z : ℂ) (h : (z - 2) * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2040_204094


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2040_204054

theorem quadratic_coefficient (x : ℝ) : 
  (3 * x^2 = 8 * x + 10) → 
  ∃ a b c : ℝ, (a * x^2 + b * x + c = 0 ∧ b = -8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2040_204054


namespace NUMINAMATH_CALUDE_expression_value_l2040_204008

theorem expression_value (p q : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ p * x^3 + q * x - 1 = 13) → 
  (∃ y : ℝ, y = -3 ∧ p * y^3 + q * y - 1 = -15) :=
sorry

end NUMINAMATH_CALUDE_expression_value_l2040_204008


namespace NUMINAMATH_CALUDE_lcm_18_28_l2040_204075

theorem lcm_18_28 : Nat.lcm 18 28 = 252 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_28_l2040_204075


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2040_204081

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2040_204081


namespace NUMINAMATH_CALUDE_number_divided_by_three_l2040_204052

theorem number_divided_by_three : ∃ x : ℝ, x / 3 = 3 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l2040_204052


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l2040_204021

theorem sin_alpha_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = -4) →
  Real.sin α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l2040_204021


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l2040_204003

/-- Given a nine-digit number B, returns the number A obtained by moving the last digit of B to the first place -/
def getA (B : ℕ) : ℕ :=
  (B % 10) * 10^8 + B / 10

/-- Checks if two natural numbers are coprime -/
def isCoprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

theorem largest_and_smallest_A :
  ∃ (A_max A_min : ℕ),
    (∀ A B : ℕ,
      B > 22222222 →
      isCoprime B 18 →
      A = getA B →
      A ≤ A_max ∧ A ≥ A_min) ∧
    A_max = 999999998 ∧
    A_min = 122222224 := by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l2040_204003


namespace NUMINAMATH_CALUDE_base_seven_digits_of_4300_l2040_204090

theorem base_seven_digits_of_4300 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 4300 ∧ 4300 < 7^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_4300_l2040_204090


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2040_204079

/-- The equation is quadratic if and only if the exponent of x in the first term is 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The coefficient of the highest degree term should not be zero -/
def coeff_nonzero (m : ℝ) : Prop := m ≠ 2

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ coeff_nonzero m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2040_204079


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2040_204035

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + y = 2 ∧ 5*x - 2*(x + y) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2040_204035


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2040_204046

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 8 →
  m > -1/2 →
  m ≠ 0 →
  m = (6 + 2*Real.sqrt 33) / 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2040_204046


namespace NUMINAMATH_CALUDE_vote_difference_is_40_l2040_204029

-- Define the committee and voting scenario
def CommitteeVoting (total_members : ℕ) (initial_for initial_against revote_for revote_against : ℕ) : Prop :=
  -- Total members condition
  total_members = initial_for + initial_against ∧
  total_members = revote_for + revote_against ∧
  -- Initially rejected condition
  initial_against > initial_for ∧
  -- Re-vote margin condition
  (revote_for - revote_against) = 3 * (initial_against - initial_for) ∧
  -- Re-vote for vs initial against condition
  revote_for * 12 = initial_against * 13

-- Theorem statement
theorem vote_difference_is_40 :
  ∀ (initial_for initial_against revote_for revote_against : ℕ),
    CommitteeVoting 500 initial_for initial_against revote_for revote_against →
    revote_for - initial_for = 40 := by
  sorry

end NUMINAMATH_CALUDE_vote_difference_is_40_l2040_204029


namespace NUMINAMATH_CALUDE_mikes_toy_expenses_l2040_204086

/-- The total amount Mike spent on toys -/
def total_spent (marbles_cost football_cost baseball_cost : ℚ) : ℚ :=
  marbles_cost + football_cost + baseball_cost

/-- Theorem stating the total amount Mike spent on toys -/
theorem mikes_toy_expenses :
  total_spent 9.05 4.95 6.52 = 20.52 := by sorry

end NUMINAMATH_CALUDE_mikes_toy_expenses_l2040_204086


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2040_204020

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem: The equation of the tangent line to f(x) at (1, 1) is 2x - y - 1 = 0
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2040_204020


namespace NUMINAMATH_CALUDE_pentagon_side_length_l2040_204015

/-- Given an equilateral triangle with side length 9/20 cm, prove that a regular pentagon with the same perimeter has side length 27/100 cm. -/
theorem pentagon_side_length (triangle_side : ℝ) (pentagon_side : ℝ) : 
  triangle_side = 9/20 → 
  3 * triangle_side = 5 * pentagon_side → 
  pentagon_side = 27/100 := by sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l2040_204015


namespace NUMINAMATH_CALUDE_problem_solution_l2040_204034

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 36/((x - 3)^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2040_204034


namespace NUMINAMATH_CALUDE_remainder_of_power_plus_two_l2040_204089

theorem remainder_of_power_plus_two : (3^87 + 2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_plus_two_l2040_204089


namespace NUMINAMATH_CALUDE_second_group_size_l2040_204028

theorem second_group_size (sum_first : ℕ) (count_first : ℕ) (avg_second : ℚ) (avg_total : ℚ) 
  (h1 : sum_first = 84)
  (h2 : count_first = 7)
  (h3 : avg_second = 21)
  (h4 : avg_total = 18) :
  ∃ (count_second : ℕ), 
    (sum_first + count_second * avg_second) / (count_first + count_second) = avg_total ∧ 
    count_second = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l2040_204028


namespace NUMINAMATH_CALUDE_derivative_of_exp_sin_l2040_204031

theorem derivative_of_exp_sin (x : ℝ) :
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_sin_l2040_204031


namespace NUMINAMATH_CALUDE_sin_B_value_l2040_204051

-- Define a right triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2

-- Define the given triangle
def given_triangle : RightTriangle where
  a := 3
  b := 4
  c := 5
  right_angle := by norm_num

-- Theorem to prove
theorem sin_B_value (triangle : RightTriangle) (h1 : triangle.a = 3) (h2 : triangle.b = 4) :
  Real.sin (Real.arcsin (triangle.b / triangle.c)) = 4/5 := by
  sorry

#check sin_B_value given_triangle rfl rfl

end NUMINAMATH_CALUDE_sin_B_value_l2040_204051


namespace NUMINAMATH_CALUDE_option1_cheapest_l2040_204082

/-- Regular ticket price -/
def regular_price : ℕ → ℕ := λ x => 40 * x

/-- Platinum card (Option 1) price -/
def platinum_price : ℕ → ℕ := λ x => 200 + 20 * x

/-- Diamond card (Option 2) price -/
def diamond_price : ℕ → ℕ := λ _ => 1000

/-- Theorem: For 8 < x < 40, Option 1 is the cheapest -/
theorem option1_cheapest (x : ℕ) (h1 : 8 < x) (h2 : x < 40) :
  platinum_price x < regular_price x ∧ platinum_price x < diamond_price x :=
by sorry

end NUMINAMATH_CALUDE_option1_cheapest_l2040_204082


namespace NUMINAMATH_CALUDE_initial_number_relation_l2040_204036

/-- The game sequence for Professor Célia's number game -/
def game_sequence (n : ℤ) : Vector ℤ 4 :=
  let c := 2 * (n + 1)
  let m := 3 * (c - 1)
  let a := 4 * (m + 1)
  ⟨[n, c, m, a], rfl⟩

/-- Theorem stating the relationship between the initial number and Ademar's number -/
theorem initial_number_relation (n x : ℤ) : 
  (game_sequence n).get 3 = x → n = (x - 16) / 24 :=
sorry

end NUMINAMATH_CALUDE_initial_number_relation_l2040_204036


namespace NUMINAMATH_CALUDE_raspberry_ratio_l2040_204017

theorem raspberry_ratio (total_berries : ℕ) (blackberries : ℕ) (blueberries : ℕ) :
  total_berries = 42 →
  blackberries = total_berries / 3 →
  blueberries = 7 →
  (total_berries - blackberries - blueberries) * 2 = total_berries := by
  sorry

end NUMINAMATH_CALUDE_raspberry_ratio_l2040_204017


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2040_204065

/-- Given a geometric sequence with positive terms where a₁, ½a₃, 2a₂ form an arithmetic sequence,
    the ratio (a₁₃ + a₁₄) / (a₁₄ + a₁₅) equals √2 - 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n) 
    (h_arith : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 13 + a 14) / (a 14 + a 15) = Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2040_204065


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2040_204016

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁ ≠ x₂) →
  (x₁^2 + a*x₁ + 2 = 0) →
  (x₂^2 + a*x₂ + 2 = 0) →
  (x₁^3 + 14/x₂^2 = x₂^3 + 14/x₁^2) →
  (a = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2040_204016


namespace NUMINAMATH_CALUDE_rug_profit_calculation_l2040_204001

theorem rug_profit_calculation (buying_price selling_price num_rugs tax_rate transport_fee : ℚ) 
  (h1 : buying_price = 40)
  (h2 : selling_price = 60)
  (h3 : num_rugs = 20)
  (h4 : tax_rate = 1/10)
  (h5 : transport_fee = 5) :
  let total_cost := buying_price * num_rugs + transport_fee * num_rugs
  let total_revenue := selling_price * num_rugs * (1 + tax_rate)
  let profit := total_revenue - total_cost
  profit = 420 := by sorry

end NUMINAMATH_CALUDE_rug_profit_calculation_l2040_204001


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2040_204064

theorem three_numbers_sum : ∀ (a b c : ℝ),
  (a ≤ b ∧ b ≤ c) →                             -- a, b, c are in ascending order
  ((a + b + c) / 3 = a + 15) →                  -- mean is 15 more than smallest
  ((a + b + c) / 3 = c - 20) →                  -- mean is 20 less than largest
  (b = 7) →                                     -- median is 7
  (a + b + c = 36) :=                           -- sum is 36
by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2040_204064


namespace NUMINAMATH_CALUDE_min_sum_squares_l2040_204091

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
    (h_sum : 3 * y₁ + 2 * y₂ + y₃ = 30) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 450/7 ∧ ∃ y₁' y₂' y₃', y₁'^2 + y₂'^2 + y₃'^2 = 450/7 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 3 * y₁' + 2 * y₂' + y₃' = 30 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_squares_l2040_204091


namespace NUMINAMATH_CALUDE_matches_arrangement_count_l2040_204088

/-- The number of ways to arrange matches for n players with some interchangeable players -/
def arrangeMatches (n : ℕ) (interchangeablePairs : ℕ) : ℕ :=
  Nat.factorial n * (2 ^ interchangeablePairs)

/-- Theorem: For 7 players with 3 pairs of interchangeable players, there are 40320 ways to arrange matches -/
theorem matches_arrangement_count :
  arrangeMatches 7 3 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_matches_arrangement_count_l2040_204088


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l2040_204085

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.38 →
  p_white = 0.34 →
  p_red + p_white + p_black = 1 →
  p_black = 0.28 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l2040_204085


namespace NUMINAMATH_CALUDE_repeating_decimal_calculation_l2040_204011

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem to prove -/
theorem repeating_decimal_calculation :
  let x : ℚ := RepeatingDecimal 5 4
  let y : ℚ := RepeatingDecimal 1 8
  (x / y) * (1 / 2) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_calculation_l2040_204011


namespace NUMINAMATH_CALUDE_parabola_properties_l2040_204084

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about the slope of line AB and the equation of the parabola -/
theorem parabola_properties (para : Parabola) (F H A B : Point) :
  -- Conditions
  (A.y^2 = 2 * para.p * A.x) →  -- A is on the parabola
  (B.y^2 = 2 * para.p * B.x) →  -- B is on the parabola
  (H.x = -para.p/2 ∧ H.y = 0) →  -- H is on the x-axis at (-p/2, 0)
  (F.x = para.p/2 ∧ F.y = 0) →  -- F is the focus at (p/2, 0)
  ((B.x - F.x)^2 + (B.y - F.y)^2 = 4 * ((A.x - F.x)^2 + (A.y - F.y)^2)) →  -- |BF| = 2|AF|
  -- Conclusions
  let slope := (B.y - A.y) / (B.x - A.x)
  (slope = 2*Real.sqrt 2/3 ∨ slope = -2*Real.sqrt 2/3) ∧
  (((B.x - A.x) * (B.y + A.y) / 2 = Real.sqrt 2) → para.p = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2040_204084


namespace NUMINAMATH_CALUDE_solve_equation_l2040_204076

/-- A function representing the non-standard addition in the sequence -/
def nonStandardAdd (a b : ℕ) : ℕ := a + b - 1

/-- The theorem stating that if 8 + x = 16 in the non-standard addition, then x = 9 -/
theorem solve_equation (x : ℕ) : nonStandardAdd 8 x = 16 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2040_204076


namespace NUMINAMATH_CALUDE_triangle_side_length_l2040_204071

theorem triangle_side_length (A B C : Real) (tanA : Real) (angleC : Real) (BC : Real) :
  tanA = 1 / 3 →
  angleC = 150 * π / 180 →
  BC = 1 →
  let sinA := Real.sqrt (1 - 1 / (1 + tanA^2))
  let AB := BC * Real.sin angleC / sinA
  AB = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2040_204071


namespace NUMINAMATH_CALUDE_rotated_ellipse_sum_l2040_204067

/-- Represents an ellipse rotated 90 degrees around its center. -/
structure RotatedEllipse where
  h' : ℝ  -- x-coordinate of the center
  k' : ℝ  -- y-coordinate of the center
  a' : ℝ  -- length of the semi-major axis
  b' : ℝ  -- length of the semi-minor axis

/-- Theorem stating the sum of parameters for a specific rotated ellipse. -/
theorem rotated_ellipse_sum (e : RotatedEllipse) 
  (center_x : e.h' = 3) 
  (center_y : e.k' = -5) 
  (major_axis : e.a' = 4) 
  (minor_axis : e.b' = 2) : 
  e.h' + e.k' + e.a' + e.b' = 4 := by
  sorry

end NUMINAMATH_CALUDE_rotated_ellipse_sum_l2040_204067


namespace NUMINAMATH_CALUDE_research_institute_reward_allocation_l2040_204005

theorem research_institute_reward_allocation :
  let n : ℕ := 10
  let a₁ : ℚ := 2
  let r : ℚ := 2
  let S := (a₁ * (1 - r^n)) / (1 - r)
  S = 2046 := by
sorry

end NUMINAMATH_CALUDE_research_institute_reward_allocation_l2040_204005


namespace NUMINAMATH_CALUDE_john_chores_time_l2040_204068

/-- Calculates the number of minutes of chores John has to do based on his cartoon watching time -/
def chores_minutes (cartoon_hours : ℕ) : ℕ :=
  let cartoon_minutes := cartoon_hours * 60
  let chore_blocks := cartoon_minutes / 10
  chore_blocks * 8

/-- Theorem: John has to do 96 minutes of chores when he watches 2 hours of cartoons -/
theorem john_chores_time : chores_minutes 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_john_chores_time_l2040_204068


namespace NUMINAMATH_CALUDE_initial_cows_count_l2040_204027

theorem initial_cows_count (initial_pigs : ℕ) (initial_goats : ℕ) 
  (added_cows : ℕ) (added_pigs : ℕ) (added_goats : ℕ) (total_after : ℕ) :
  initial_pigs = 3 →
  initial_goats = 6 →
  added_cows = 3 →
  added_pigs = 5 →
  added_goats = 2 →
  total_after = 21 →
  ∃ initial_cows : ℕ, initial_cows = 2 ∧ 
    initial_cows + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_after :=
by sorry

end NUMINAMATH_CALUDE_initial_cows_count_l2040_204027


namespace NUMINAMATH_CALUDE_product_xyz_l2040_204018

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 3) : 
  x * y * z = 1/11 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l2040_204018


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2040_204044

theorem unique_integer_satisfying_conditions : ∃! (n : ℤ), n + 15 > 16 ∧ -3*n > -9 :=
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2040_204044


namespace NUMINAMATH_CALUDE_smallest_operation_between_sqrt18_and_sqrt8_l2040_204022

theorem smallest_operation_between_sqrt18_and_sqrt8 :
  let a := Real.sqrt 18
  let b := Real.sqrt 8
  (a - b < a + b) ∧ (a - b < a * b) ∧ (a - b < a / b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_operation_between_sqrt18_and_sqrt8_l2040_204022


namespace NUMINAMATH_CALUDE_gcd_153_68_l2040_204013

theorem gcd_153_68 : Nat.gcd 153 68 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_68_l2040_204013


namespace NUMINAMATH_CALUDE_minimum_handshakes_l2040_204080

theorem minimum_handshakes (n : ℕ) (h : ℕ) (hn : n = 30) (hh : h = 3) :
  (n * h) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l2040_204080


namespace NUMINAMATH_CALUDE_point_c_transformation_l2040_204010

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ := (p.1 + t.1, p.2 + t.2)

theorem point_c_transformation :
  let c : ℝ × ℝ := (3, 3)
  let c' := translate (reflect_x (reflect_y c)) (3, -4)
  c' = (0, -7) := by sorry

end NUMINAMATH_CALUDE_point_c_transformation_l2040_204010


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2040_204007

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2040_204007


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l2040_204042

theorem sum_of_squares_bound (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l2040_204042


namespace NUMINAMATH_CALUDE_books_left_to_read_l2040_204060

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 14 → read_books = 8 → total_books - read_books = 6 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l2040_204060


namespace NUMINAMATH_CALUDE_pipe_C_rate_l2040_204072

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 80
def rate_combined : ℚ := 1 / 40

-- Define the rate of pipe C
def rate_C : ℚ := rate_A + rate_B - rate_combined

-- Theorem statement
theorem pipe_C_rate : rate_C = 1 / 240 := by
  sorry

end NUMINAMATH_CALUDE_pipe_C_rate_l2040_204072


namespace NUMINAMATH_CALUDE_lifesaving_test_percentage_l2040_204048

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed : ℝ := 30

theorem lifesaving_test_percentage :
  let total_members : ℕ := 60
  let not_passed_with_course : ℕ := 12
  let not_passed_without_course : ℕ := 30
  percentage_passed = 30 ∧
  percentage_passed = (total_members - (not_passed_with_course + not_passed_without_course)) / total_members * 100 :=
by sorry

end NUMINAMATH_CALUDE_lifesaving_test_percentage_l2040_204048


namespace NUMINAMATH_CALUDE_div_chain_equals_four_l2040_204040

theorem div_chain_equals_four : (((120 / 5) / 3) / 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_div_chain_equals_four_l2040_204040


namespace NUMINAMATH_CALUDE_min_rental_cost_is_2860_l2040_204032

/-- Represents a rental plan for cars --/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid for transporting the given amount of goods --/
def isValidPlan (plan : RentalPlan) (totalGoods : ℕ) : Prop :=
  3 * plan.typeA + 4 * plan.typeB = totalGoods

/-- Calculates the rental cost for a given plan --/
def rentalCost (plan : RentalPlan) : ℕ :=
  300 * plan.typeA + 320 * plan.typeB

/-- Theorem stating that the minimum rental cost to transport 35 tons of goods is 2860 yuan --/
theorem min_rental_cost_is_2860 :
  ∃ (plan : RentalPlan),
    isValidPlan plan 35 ∧
    rentalCost plan = 2860 ∧
    ∀ (otherPlan : RentalPlan), isValidPlan otherPlan 35 → rentalCost plan ≤ rentalCost otherPlan :=
sorry

end NUMINAMATH_CALUDE_min_rental_cost_is_2860_l2040_204032


namespace NUMINAMATH_CALUDE_repetend_5_17_l2040_204098

def repetend_of_5_17 : List Nat := [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9]

theorem repetend_5_17 :
  ∃ (k : ℕ), (5 : ℚ) / 17 = (k : ℚ) / 10^18 + 
  (List.sum (List.zipWith (λ (d i : ℕ) => (d : ℚ) / 10^(i+1)) repetend_of_5_17 (List.range 18))) *
  (1 / (1 - 1 / 10^18)) :=
by
  sorry

end NUMINAMATH_CALUDE_repetend_5_17_l2040_204098


namespace NUMINAMATH_CALUDE_supplement_of_beta_l2040_204097

def complementary_angles (α β : Real) : Prop := α + β = 90

theorem supplement_of_beta (α β : Real) 
  (h1 : complementary_angles α β) 
  (h2 : α = 30) : 
  180 - β = 120 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_beta_l2040_204097


namespace NUMINAMATH_CALUDE_work_completion_time_l2040_204024

theorem work_completion_time 
  (john_time : ℝ) 
  (rose_time : ℝ) 
  (h1 : john_time = 320) 
  (h2 : rose_time = 480) : 
  1 / (1 / john_time + 1 / rose_time) = 192 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2040_204024


namespace NUMINAMATH_CALUDE_investment_plans_count_l2040_204056

/-- The number of ways to distribute 3 distinct projects among 5 cities, 
    with no more than 2 projects per city -/
def investmentPlans : ℕ := 120

/-- The number of candidate cities -/
def numCities : ℕ := 5

/-- The number of projects to be distributed -/
def numProjects : ℕ := 3

/-- The maximum number of projects allowed in a single city -/
def maxProjectsPerCity : ℕ := 2

theorem investment_plans_count :
  investmentPlans = 
    (numCities.choose numProjects) + 
    (numProjects.choose 2) * numCities * (numCities - 1) := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l2040_204056


namespace NUMINAMATH_CALUDE_projection_a_on_b_is_sqrt_5_l2040_204039

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 4]

theorem projection_a_on_b_is_sqrt_5 :
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  dot_product / magnitude_b = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_is_sqrt_5_l2040_204039


namespace NUMINAMATH_CALUDE_least_divisor_power_l2040_204074

theorem least_divisor_power (a : ℕ) (h1 : a > 1) (h2 : Odd a) :
  (∃ n : ℕ, n > 0 ∧ (2^2000 : ℕ) ∣ (a^n - 1)) ∧
  (∀ m : ℕ, 0 < m → m < 2^1998 → ¬((2^2000 : ℕ) ∣ (a^m - 1))) ∧
  ((2^2000 : ℕ) ∣ (a^(2^1998) - 1)) :=
sorry

end NUMINAMATH_CALUDE_least_divisor_power_l2040_204074


namespace NUMINAMATH_CALUDE_largest_number_l2040_204099

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 99 / 100)
  (hb : b = 9099 / 10000)
  (hc : c = 9 / 10)
  (hd : d = 909 / 1000)
  (he : e = 9009 / 10000) :
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l2040_204099


namespace NUMINAMATH_CALUDE_corrected_mean_l2040_204033

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 46 →
  ((n : ℚ) * original_mean - wrong_value + correct_value) / n = 36.46 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l2040_204033


namespace NUMINAMATH_CALUDE_no_real_roots_geometric_sequence_l2040_204037

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c = 0 has no real solutions -/
theorem no_real_roots_geometric_sequence (a b c : ℝ) (h1 : a ≠ 0) (h2 : b^2 = a*c) (h3 : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_geometric_sequence_l2040_204037


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l2040_204096

/-- Given a compound where 7 moles have a total molecular weight of 854 grams,
    prove that the molecular weight of 1 mole is 122 grams/mole. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 854)
  (h2 : num_moles = 7) :
  total_weight / num_moles = 122 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l2040_204096


namespace NUMINAMATH_CALUDE_work_completion_time_l2040_204009

/-- Represents the amount of work one man can do in one day -/
def man_work : ℝ := sorry

/-- Represents the amount of work one boy can do in one day -/
def boy_work : ℝ := sorry

/-- The number of days it takes 6 men and 8 boys to complete the work -/
def x : ℝ := sorry

theorem work_completion_time :
  (6 * man_work + 8 * boy_work) * x = (26 * man_work + 48 * boy_work) * 2 ∧
  (6 * man_work + 8 * boy_work) * x = (15 * man_work + 20 * boy_work) * 4 →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2040_204009


namespace NUMINAMATH_CALUDE_special_linear_function_properties_l2040_204043

/-- A linear function y = mx + c, where m is the slope and c is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (2a-4)x + (3-b) -/
def specialLinearFunction (a b : ℝ) : LinearFunction where
  slope := 2*a - 4
  intercept := 3 - b

theorem special_linear_function_properties (a b : ℝ) :
  let f := specialLinearFunction a b
  (∃ k : ℝ, ∀ x, f.slope * x = k * x) ↔ (a ≠ 2 ∧ b = 3) ∧
  (f.slope < 0 ∧ f.intercept ≤ 0) ↔ (a < 2 ∧ b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_special_linear_function_properties_l2040_204043


namespace NUMINAMATH_CALUDE_fraction_repeating_block_length_l2040_204083

/-- The number of digits in the smallest repeating block of the decimal expansion of 5/7 -/
def repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 5 / 7

theorem fraction_repeating_block_length :
  repeating_block_length = 6 ∧ 
  ∀ n : ℕ, n < repeating_block_length → 
    ∃ k : ℕ, fraction * 10^repeating_block_length - fraction * 10^n = k :=
sorry

end NUMINAMATH_CALUDE_fraction_repeating_block_length_l2040_204083


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l2040_204006

theorem sin_alpha_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -Real.sqrt 3)
  (∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.sin α = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l2040_204006


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l2040_204059

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l2040_204059


namespace NUMINAMATH_CALUDE_curve_and_intersection_l2040_204000

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 3))^2) +
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statement
theorem curve_and_intersection :
  ∃ (k : ℝ),
    (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
    (∃ x₁ y₁ x₂ y₂,
      C x₁ y₁ ∧ C x₂ y₂ ∧
      intersecting_line k x₁ y₁ ∧
      intersecting_line k x₂ y₂ ∧
      perpendicular x₁ y₁ x₂ y₂ ∧
      (k = 1/2 ∨ k = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_curve_and_intersection_l2040_204000


namespace NUMINAMATH_CALUDE_three_possible_values_l2040_204095

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def construct_number (a b : ℕ) : ℕ := a * 10000 + 3750 + b

theorem three_possible_values :
  ∃ (s : Finset ℕ), s.card = 3 ∧
    (∀ a ∈ s, is_single_digit a ∧
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0)) ∧
    (∀ a, is_single_digit a →
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0) →
      a ∈ s) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_l2040_204095


namespace NUMINAMATH_CALUDE_complex_number_problem_l2040_204077

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = 1 → 
  a = 1 ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2040_204077


namespace NUMINAMATH_CALUDE_inscribed_circle_exists_l2040_204092

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the area of a polygon
def area (p : ConvexPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : ConvexPolygon) : ℝ := sorry

-- Define a point inside a polygon
def PointInside (p : ConvexPolygon) : Type := sorry

-- Define the distance from a point to a side of the polygon
def distanceToSide (point : PointInside p) (side : sorry) : ℝ := sorry

-- Theorem statement
theorem inscribed_circle_exists (p : ConvexPolygon) (h : area p > 0) :
  ∃ (center : PointInside p), ∀ (side : sorry),
    distanceToSide center side ≥ (area p) / (perimeter p) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_exists_l2040_204092


namespace NUMINAMATH_CALUDE_set_inclusion_range_l2040_204058

theorem set_inclusion_range (a : ℝ) : 
  let P : Set ℝ := {x | |x - 1| > 2}
  let S : Set ℝ := {x | x^2 - (a + 1)*x + a > 0}
  (P ⊆ S) → ((-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3)) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_range_l2040_204058


namespace NUMINAMATH_CALUDE_roof_area_calculation_l2040_204087

def roof_area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

theorem roof_area_calculation :
  ∀ w l : ℝ,
  l = 4 * w →
  l - w = 36 →
  roof_area w l = 576 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_area_calculation_l2040_204087


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l2040_204050

variable (a : ℝ)
variable (h : a ≠ 0)

theorem simplest_common_denominator : 
  lcm (2 * a) (a ^ 2) = 2 * (a ^ 2) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l2040_204050


namespace NUMINAMATH_CALUDE_spiral_similarity_composition_l2040_204004

open Real

/-- A spiral similarity (also known as a rotational homothety) -/
structure SpiralSimilarity where
  center : ℝ × ℝ
  angle : ℝ
  coefficient : ℝ

/-- Composition of two spiral similarities -/
def compose (P₁ P₂ : SpiralSimilarity) : SpiralSimilarity :=
  sorry

/-- Rotation -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Check if a spiral similarity is a rotation -/
def isRotation (P : SpiralSimilarity) : Prop :=
  sorry

/-- The angle between two vectors -/
def vectorAngle (v₁ v₂ : ℝ × ℝ) : ℝ :=
  sorry

theorem spiral_similarity_composition
  (P₁ P₂ : SpiralSimilarity)
  (h₁ : P₁.angle = P₂.angle)
  (h₂ : P₁.coefficient * P₂.coefficient = 1)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hN : N = sorry) -- N = P₁(M)
  : 
  let P := compose P₂ P₁
  ∃ (R : Rotation), 
    isRotation P ∧ 
    P.center = R.center ∧
    R.center.fst = P₁.center.fst ∧ R.center.snd = P₂.center.snd ∧
    R.angle = 2 * vectorAngle (M.fst - P₁.center.fst, M.snd - P₁.center.snd) (N.fst - M.fst, N.snd - M.snd) :=
sorry

end NUMINAMATH_CALUDE_spiral_similarity_composition_l2040_204004


namespace NUMINAMATH_CALUDE_square_difference_equality_l2040_204093

theorem square_difference_equality : (19 + 15)^2 - (19 - 15)^2 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2040_204093


namespace NUMINAMATH_CALUDE_matrix_power_sum_l2040_204062

theorem matrix_power_sum (b m : ℤ) : 
  let C : Matrix (Fin 3) (Fin 3) ℤ := !![1, 3, b; 0, 1, 5; 0, 0, 1]
  (C ^ m = !![1, 27, 3008; 0, 1, 45; 0, 0, 1]) → b + m = 283 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l2040_204062


namespace NUMINAMATH_CALUDE_appended_number_cube_sum_l2040_204053

theorem appended_number_cube_sum (a b c : ℕ) : 
  b ≥ 10 ∧ b < 100 ∧ c ≥ 10 ∧ c < 100 →
  10000 * a + 100 * b + c = (a + b + c)^3 →
  a = 9 ∧ b = 11 ∧ c = 25 :=
by sorry

end NUMINAMATH_CALUDE_appended_number_cube_sum_l2040_204053


namespace NUMINAMATH_CALUDE_final_value_16_l2040_204002

/-- A function that simulates the loop behavior --/
def loop_iteration (b : ℕ) : ℕ := b + 3

/-- The loop condition --/
def loop_condition (b : ℕ) : Prop := b < 16

/-- The theorem statement --/
theorem final_value_16 :
  ∃ (n : ℕ), 
    let b := 10
    let final_b := (loop_iteration^[n] b)
    (∀ k < n, loop_condition ((loop_iteration^[k]) b)) ∧
    ¬(loop_condition final_b) ∧
    final_b = 16 :=
sorry

end NUMINAMATH_CALUDE_final_value_16_l2040_204002


namespace NUMINAMATH_CALUDE_multiplication_error_factors_l2040_204030

theorem multiplication_error_factors : ∃ (x y z : ℕ), 
  x = y + 10 ∧ 
  x * y = z + 40 ∧ 
  z = 39 * y + 22 ∧ 
  x = 41 ∧ 
  y = 31 := by
sorry

end NUMINAMATH_CALUDE_multiplication_error_factors_l2040_204030


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l2040_204014

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  ∀ z, z ∈ Set.Ioo (-18 : ℝ) 10 ↔ ∃ (x' y' : ℝ), 
    30 < x' ∧ x' < 42 ∧ 
    16 < y' ∧ y' < 24 ∧ 
    z = x' - 2*y' :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l2040_204014


namespace NUMINAMATH_CALUDE_first_three_valid_numbers_l2040_204063

def is_sum_of_consecutive (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = k * a

def is_valid_number (n : ℕ) : Prop :=
  is_sum_of_consecutive n 5 ∧ is_sum_of_consecutive n 7

theorem first_three_valid_numbers :
  (is_valid_number 35 ∧ 
   is_valid_number 70 ∧ 
   is_valid_number 105) ∧ 
  (∀ m : ℕ, m < 35 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 35 < m ∧ m < 70 → ¬is_valid_number m) ∧
  (∀ m : ℕ, 70 < m ∧ m < 105 → ¬is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_first_three_valid_numbers_l2040_204063


namespace NUMINAMATH_CALUDE_jack_email_difference_l2040_204041

theorem jack_email_difference : 
  ∀ (morning_emails afternoon_emails morning_letters afternoon_letters : ℕ),
  morning_emails = 10 →
  afternoon_emails = 3 →
  morning_letters = 12 →
  afternoon_letters = 44 →
  morning_emails - afternoon_emails = 7 :=
by sorry

end NUMINAMATH_CALUDE_jack_email_difference_l2040_204041
