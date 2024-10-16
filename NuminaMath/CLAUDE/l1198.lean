import Mathlib

namespace NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l1198_119821

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 242 - 14 * Real.sqrt 193 :=
by sorry

theorem equality_condition (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 242 - 14 * Real.sqrt 193 ↔ 
  α = 0 ∧ β = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_equality_condition_l1198_119821


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l1198_119867

/-- Represents the number of carrots harvested from each bed -/
structure CarrotHarvest where
  bed1 : ℕ
  bed2 : ℕ
  bed3 : ℕ

/-- Calculates the total weight of carrots in pounds -/
def totalWeight (harvest : CarrotHarvest) (carrotsPerPound : ℕ) : ℕ :=
  (harvest.bed1 + harvest.bed2 + harvest.bed3) / carrotsPerPound

/-- Theorem stating that Kelly's carrot harvest weighs 39 pounds -/
theorem kelly_carrot_harvest :
  let harvest := CarrotHarvest.mk 55 101 78
  let carrotsPerPound := 6
  totalWeight harvest carrotsPerPound = 39 := by
  sorry


end NUMINAMATH_CALUDE_kelly_carrot_harvest_l1198_119867


namespace NUMINAMATH_CALUDE_certain_number_problem_l1198_119874

theorem certain_number_problem (n x : ℝ) : 
  (n - 4) / x = 7 + (8 / x) → x = 6 → n = 54 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1198_119874


namespace NUMINAMATH_CALUDE_april_savings_l1198_119806

def savings_pattern (month : Nat) : Nat :=
  2^month

theorem april_savings : savings_pattern 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_april_savings_l1198_119806


namespace NUMINAMATH_CALUDE_sam_chewing_gums_l1198_119861

theorem sam_chewing_gums (total : ℕ) (mary : ℕ) (sue : ℕ) (h1 : total = 30) (h2 : mary = 5) (h3 : sue = 15) : 
  total - mary - sue = 10 := by
  sorry

end NUMINAMATH_CALUDE_sam_chewing_gums_l1198_119861


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l1198_119840

theorem youngest_sibling_age 
  (siblings : Fin 4 → ℕ) 
  (age_differences : ∀ i : Fin 4, siblings i = siblings 0 + [0, 2, 7, 11].get i) 
  (average_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 25) : 
  siblings 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l1198_119840


namespace NUMINAMATH_CALUDE_fraction_simplification_l1198_119881

theorem fraction_simplification :
  (240 : ℚ) / 20 * 6 / 150 * 12 / 5 = 48 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1198_119881


namespace NUMINAMATH_CALUDE_triangle_inequality_third_stick_length_l1198_119876

theorem triangle_inequality (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (a < b + c ∧ b < c + a ∧ c < a + b) :=
sorry

theorem third_stick_length (a b : ℝ) (ha : a = 20) (hb : b = 30) :
  ∃ c, c = 30 ∧ 
       (a + b > c ∧ b + c > a ∧ c + a > b) ∧
       ¬(a + b > 10 ∧ b + 10 > a ∧ 10 + a > b) ∧
       ¬(a + b > 50 ∧ b + 50 > a ∧ 50 + a > b) ∧
       ¬(a + b > 70 ∧ b + 70 > a ∧ 70 + a > b) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_stick_length_l1198_119876


namespace NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1198_119812

theorem square_minus_self_divisible_by_two (a : ℤ) : ∃ k : ℤ, a^2 - a = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_divisible_by_two_l1198_119812


namespace NUMINAMATH_CALUDE_intersection_A_B_l1198_119836

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 4) ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1198_119836


namespace NUMINAMATH_CALUDE_unit_digit_4137_pow_1289_l1198_119817

/-- The unit digit of a number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The unit digit pattern for powers of 7 repeats every 4 steps -/
def unitDigitPattern : Fin 4 → ℕ
  | 0 => 7
  | 1 => 9
  | 2 => 3
  | 3 => 1

theorem unit_digit_4137_pow_1289 :
  unitDigit ((4137 : ℕ) ^ 1289) = 7 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_4137_pow_1289_l1198_119817


namespace NUMINAMATH_CALUDE_chris_breath_holding_start_l1198_119885

def breath_holding_progression (start : ℕ) (days : ℕ) : ℕ :=
  start + 10 * (days - 1)

theorem chris_breath_holding_start :
  ∃ (start : ℕ),
    breath_holding_progression start 2 = 20 ∧
    breath_holding_progression start 6 = 90 ∧
    start = 10 := by
  sorry

end NUMINAMATH_CALUDE_chris_breath_holding_start_l1198_119885


namespace NUMINAMATH_CALUDE_equation_solutions_l1198_119842

def is_solution (m n r k : ℕ+) : Prop :=
  m * n + n * r + m * r = k * (m + n + r)

theorem equation_solutions :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = 7 ∧ 
    (∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 2) ∧
    (∀ x : ℕ+ × ℕ+ × ℕ+, is_solution x.1 x.2.1 x.2.2 2 → x ∈ s)) ∧
  (∀ k : ℕ+, k > 1 → 
    ∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card ≥ 3 * k + 1 ∧ 
      ∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 k) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1198_119842


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_88200_l1198_119839

def sum_of_prime_factors (n : ℕ) : ℕ := (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_88200 :
  sum_of_prime_factors 88200 = 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_88200_l1198_119839


namespace NUMINAMATH_CALUDE_eliot_account_balance_l1198_119829

theorem eliot_account_balance :
  ∀ (A E : ℝ),
  A > E →
  A - E = (1 / 12) * (A + E) →
  1.1 * A = 1.2 * E + 21 →
  E = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l1198_119829


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1198_119890

theorem fraction_equality_sum (M N : ℚ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1198_119890


namespace NUMINAMATH_CALUDE_integral_even_odd_functions_l1198_119857

open Set
open Interval
open MeasureTheory
open Measure

/-- A function f is even on [-a,a] -/
def IsEven (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = f x

/-- A function f is odd on [-a,a] -/
def IsOdd (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = -f x

theorem integral_even_odd_functions (f : ℝ → ℝ) (a : ℝ) :
  (IsEven f a → ∫ x in Icc (-a) a, f x = 2 * ∫ x in Icc 0 a, f x) ∧
  (IsOdd f a → ∫ x in Icc (-a) a, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_integral_even_odd_functions_l1198_119857


namespace NUMINAMATH_CALUDE_men_per_table_l1198_119833

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90) : 
  (total_customers - num_tables * women_per_table) / num_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_men_per_table_l1198_119833


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1198_119834

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 400) + (Real.sqrt 98 / Real.sqrt 56) = (3 + 2 * Real.sqrt 7) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1198_119834


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l1198_119850

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 12 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l1198_119850


namespace NUMINAMATH_CALUDE_sixth_term_value_l1198_119847

/-- A sequence of positive integers where each term after the first is 1/4 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) : ℚ) = (1 / 4) * ((a n : ℚ) + (a (n + 2) : ℚ))

theorem sixth_term_value (a : ℕ → ℕ+) (h : SpecialSequence a) (h1 : a 1 = 3) (h5 : a 5 = 43) :
  a 6 = 129 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1198_119847


namespace NUMINAMATH_CALUDE_functional_equation_identity_l1198_119899

theorem functional_equation_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f m + n) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l1198_119899


namespace NUMINAMATH_CALUDE_probability_is_one_l1198_119804

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

def probability_less_than_or_equal_to_9 : ℚ :=
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card

theorem probability_is_one :
  probability_less_than_or_equal_to_9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_l1198_119804


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1198_119827

theorem negation_of_proposition_is_true : 
  (∃ a : ℝ, a > 2 ∧ a^2 ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1198_119827


namespace NUMINAMATH_CALUDE_y_derivative_l1198_119800

noncomputable def y (x : ℝ) : ℝ := 
  (1/2) * Real.log ((1 + Real.cos x) / (1 - Real.cos x)) - 1 / Real.cos x - 1 / (3 * (Real.cos x)^3)

theorem y_derivative (x : ℝ) (h : Real.cos x ≠ 0) (h' : Real.sin x ≠ 0) : 
  deriv y x = -1 / (Real.sin x * (Real.cos x)^4) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1198_119800


namespace NUMINAMATH_CALUDE_chocolate_chip_recipe_l1198_119818

theorem chocolate_chip_recipe (total_recipes : ℕ) (total_cups : ℕ) 
  (h1 : total_recipes = 23) (h2 : total_cups = 46) :
  total_cups / total_recipes = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_recipe_l1198_119818


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1198_119878

theorem square_rectangle_area_ratio :
  let rectangle_width : ℝ := 3
  let rectangle_length : ℝ := 5
  let square_side : ℝ := 1
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_width * rectangle_length
  square_area / rectangle_area = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l1198_119878


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_eight_l1198_119884

theorem sum_of_solutions_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 7 ∧ N₂ * (N₂ - 8) = 7 ∧ N₁ + N₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_eight_l1198_119884


namespace NUMINAMATH_CALUDE_original_cost_price_l1198_119877

/-- Calculates the original cost price given a series of transactions and the final price --/
theorem original_cost_price 
  (profit_ab profit_bc discount_cd profit_de final_price : ℝ) :
  let original_price := 
    final_price / ((1 + profit_ab/100) * (1 + profit_bc/100) * (1 - discount_cd/100) * (1 + profit_de/100))
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    (profit_ab = 20 ∧ 
     profit_bc = 25 ∧ 
     discount_cd = 15 ∧ 
     profit_de = 30 ∧ 
     final_price = 289.1) →
    (142.8 - ε ≤ original_price ∧ original_price ≤ 142.8 + ε) :=
by sorry

end NUMINAMATH_CALUDE_original_cost_price_l1198_119877


namespace NUMINAMATH_CALUDE_mandy_toys_count_l1198_119869

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end NUMINAMATH_CALUDE_mandy_toys_count_l1198_119869


namespace NUMINAMATH_CALUDE_mabel_transactions_l1198_119816

theorem mabel_transactions : ∃ M : ℕ,
  let A := (11 * M) / 10  -- Anthony's transactions
  let C := (2 * A) / 3    -- Cal's transactions
  let J := C + 15         -- Jade's transactions
  J = 81 ∧ M = 90 := by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l1198_119816


namespace NUMINAMATH_CALUDE_problem_solution_l1198_119815

theorem problem_solution (a b n : ℤ) : 
  a % 50 = 24 →
  b % 50 = 95 →
  150 ≤ n ∧ n ≤ 200 →
  (a - b) % 50 = n % 50 →
  n % 4 = 3 →
  n = 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1198_119815


namespace NUMINAMATH_CALUDE_exists_d_for_equation_l1198_119831

theorem exists_d_for_equation : ∃ d : ℕ, 3^2 + 6*d = 735 := by sorry

end NUMINAMATH_CALUDE_exists_d_for_equation_l1198_119831


namespace NUMINAMATH_CALUDE_sequence_count_mod_l1198_119852

def sequence_count (n : ℕ) (max : ℕ) : ℕ :=
  let m := Nat.choose (max - n + n) n
  m / 3

theorem sequence_count_mod (n : ℕ) (max : ℕ) : 
  sequence_count n max % 1000 = 662 :=
sorry

#check sequence_count_mod 10 2018

end NUMINAMATH_CALUDE_sequence_count_mod_l1198_119852


namespace NUMINAMATH_CALUDE_red_beans_proposition_l1198_119802

-- Define a type for lines in the poem
inductive PoemLine
| A : PoemLine
| B : PoemLine
| C : PoemLine
| D : PoemLine

-- Define what a proposition is
def isProposition (line : PoemLine) : Prop :=
  match line with
  | PoemLine.A => true  -- "Red beans grow in the southern country" is a proposition
  | _ => false          -- Other lines are not propositions for this problem

-- Theorem statement
theorem red_beans_proposition :
  isProposition PoemLine.A :=
by sorry

end NUMINAMATH_CALUDE_red_beans_proposition_l1198_119802


namespace NUMINAMATH_CALUDE_biased_coin_theorem_l1198_119853

def biased_coin_prob (h : ℚ) : Prop :=
  (15 : ℚ) * h^2 * (1 - h)^4 = (20 : ℚ) * h^3 * (1 - h)^3

theorem biased_coin_theorem :
  ∀ h : ℚ, 0 < h → h < 1 → biased_coin_prob h →
  (15 : ℚ) * h^4 * (1 - h)^2 = 40 / 243 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_theorem_l1198_119853


namespace NUMINAMATH_CALUDE_partridge_family_allowance_l1198_119814

/-- The total weekly allowance for the Partridge family children -/
theorem partridge_family_allowance : 
  ∀ (younger_children older_children : ℕ) 
    (younger_allowance older_allowance : ℚ),
  younger_children = 3 →
  older_children = 2 →
  younger_allowance = 8 →
  older_allowance = 13 →
  (younger_children : ℚ) * younger_allowance + (older_children : ℚ) * older_allowance = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_partridge_family_allowance_l1198_119814


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l1198_119893

/-- 
Theorem: The angles -675° and -315° are the only angles in the range [-720°, 0°) 
that have the same terminal side as 45°.
-/
theorem same_terminal_side_angles : 
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 → 
  (∃ k : ℤ, θ = 45 + 360 * k) ↔ (θ = -675 ∨ θ = -315) := by
  sorry


end NUMINAMATH_CALUDE_same_terminal_side_angles_l1198_119893


namespace NUMINAMATH_CALUDE_parabola_intersection_l1198_119810

/-- The function f(x) = x² --/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: Given two points on the parabola y = x², with the first point at x = 1
    and the second at x = 4, if we trisect the line segment between these points
    and draw a horizontal line through the first trisection point (closer to the first point),
    then this line intersects the parabola at x = -2. --/
theorem parabola_intersection (x₁ x₂ x₃ : ℝ) (y₁ y₂ y₃ : ℝ) :
  x₁ = 1 →
  x₂ = 4 →
  y₁ = f x₁ →
  y₂ = f x₂ →
  let xc := (2 * x₁ + x₂) / 3
  let yc := f xc
  y₃ = yc →
  y₃ = f x₃ →
  x₃ ≠ xc →
  x₃ = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1198_119810


namespace NUMINAMATH_CALUDE_survival_probability_estimate_l1198_119851

/-- Represents the survival data for a sample of seedlings -/
structure SeedlingSample where
  transplanted : ℕ
  survived : ℕ
  survivalRate : ℚ

/-- The data set of seedling survival samples -/
def seedlingData : List SeedlingSample := [
  ⟨20, 15, 75/100⟩,
  ⟨40, 33, 33/40⟩,
  ⟨100, 78, 39/50⟩,
  ⟨200, 158, 79/100⟩,
  ⟨400, 321, 801/1000⟩,
  ⟨1000, 801, 801/1000⟩
]

/-- Estimates the overall probability of seedling survival -/
def estimateSurvivalProbability (data : List SeedlingSample) : ℚ :=
  sorry

/-- Theorem stating that the estimated survival probability is approximately 0.80 -/
theorem survival_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateSurvivalProbability seedlingData - 4/5| < ε :=
sorry

end NUMINAMATH_CALUDE_survival_probability_estimate_l1198_119851


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l1198_119855

theorem lawn_mowing_earnings 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 17) 
  (h2 : unmowed_lawns = 9) 
  (h3 : total_earnings = 32) : 
  (total_earnings : ℚ) / ((total_lawns - unmowed_lawns) : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l1198_119855


namespace NUMINAMATH_CALUDE_six_students_solved_only_B_l1198_119860

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  d : ℕ  -- A and B
  e : ℕ  -- A and C
  f : ℕ  -- B and C
  g : ℕ  -- A, B, and C

/-- The conditions of the math competition problem -/
def MathCompetitionConditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.d + s.e + s.f + s.g = 25 ∧
  -- Among students who didn't solve A, number solving B is twice the number solving C
  s.b + s.f = 2 * (s.c + s.f) ∧
  -- Number of students solving only A is one more than number of students solving A among remaining students
  s.a = s.d + s.e + s.g + 1 ∧
  -- Among students solving only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_students_solved_only_B (s : ProblemSolvers) 
  (h : MathCompetitionConditions s) : s.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_solved_only_B_l1198_119860


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1198_119844

theorem polynomial_factorization (a b m n : ℝ) 
  (h : |m - 4| + (n^2 - 8*n + 16) = 0) : 
  a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1198_119844


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1198_119882

theorem quadratic_equation_properties :
  ∀ s t : ℝ, 2 * s^2 + 3 * s - 1 = 0 → 2 * t^2 + 3 * t - 1 = 0 → s ≠ t →
  (s + t = -3/2) ∧
  (s * t = -1/2) ∧
  (s^2 + t^2 = 13/4) ∧
  (|1/s - 1/t| = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1198_119882


namespace NUMINAMATH_CALUDE_probability_two_white_balls_correct_l1198_119866

/-- The probability of having two white balls in an urn, given the conditions of the problem -/
def probability_two_white_balls (n : ℕ) : ℚ :=
  (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n)

/-- The theorem stating the probability of having two white balls in the urn -/
theorem probability_two_white_balls_correct (n : ℕ) :
  let total_balls : ℕ := 4
  let draws : ℕ := 2 * n
  let white_draws : ℕ := n
  probability_two_white_balls n = (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n) :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_correct_l1198_119866


namespace NUMINAMATH_CALUDE_sum_squared_l1198_119883

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) :
  (x + y)^2 = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l1198_119883


namespace NUMINAMATH_CALUDE_pythagorean_theorem_geometric_dissection_l1198_119849

/-- Pythagorean theorem using geometric dissection -/
theorem pythagorean_theorem_geometric_dissection 
  (a b c : ℝ) 
  (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_hypotenuse : c = max a b)
  (h_inner_square : ∃ (s : ℝ), s = |b - a| ∧ s^2 = (b - a)^2)
  (h_area_equality : c^2 = 2*a*b + (b - a)^2) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_geometric_dissection_l1198_119849


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_product_l1198_119863

theorem distinct_prime_factors_of_product : ∃ (s : Finset Nat), 
  (∀ p ∈ s, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (86 * 88 * 90 * 92) ↔ p ∈ s)) ∧ 
  Finset.card s = 6 := by
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_product_l1198_119863


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l1198_119848

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 4|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2) 2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≤ m - m^2) → m ∈ Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l1198_119848


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1198_119801

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))) ∧
    vector P N = (-2 : ℝ) • (vector P M) ∧
    P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1198_119801


namespace NUMINAMATH_CALUDE_tim_initial_books_l1198_119841

/-- The number of books Sandy has -/
def sandy_books : ℕ := 10

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books they have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Tim's initial number of books -/
def tim_books : ℕ := 33

theorem tim_initial_books : 
  sandy_books + tim_books - benny_lost = remaining_books :=
by sorry

end NUMINAMATH_CALUDE_tim_initial_books_l1198_119841


namespace NUMINAMATH_CALUDE_dave_apps_remaining_l1198_119895

/-- Calculates the number of apps remaining after deletion -/
def apps_remaining (initial : Nat) (deleted : Nat) : Nat :=
  initial - deleted

theorem dave_apps_remaining :
  apps_remaining 16 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_remaining_l1198_119895


namespace NUMINAMATH_CALUDE_pizza_production_l1198_119835

theorem pizza_production (craig_day1 : ℕ) (craig_increase : ℕ) (heather_decrease : ℕ)
  (h1 : craig_day1 = 40)
  (h2 : craig_increase = 60)
  (h3 : heather_decrease = 20) :
  let heather_day1 := 4 * craig_day1
  let craig_day2 := craig_day1 + craig_increase
  let heather_day2 := craig_day2 - heather_decrease
  heather_day1 + craig_day1 + heather_day2 + craig_day2 = 380 := by
  sorry


end NUMINAMATH_CALUDE_pizza_production_l1198_119835


namespace NUMINAMATH_CALUDE_roberts_pencils_l1198_119870

-- Define the price of a pencil in cents
def pencil_price : ℕ := 20

-- Define the number of pencils Tolu wants
def tolu_pencils : ℕ := 3

-- Define the number of pencils Melissa wants
def melissa_pencils : ℕ := 2

-- Define the total amount spent by all students in cents
def total_spent : ℕ := 200

-- Theorem to prove Robert's number of pencils
theorem roberts_pencils : 
  ∃ (robert_pencils : ℕ), 
    pencil_price * (tolu_pencils + melissa_pencils + robert_pencils) = total_spent ∧
    robert_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_roberts_pencils_l1198_119870


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l1198_119888

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 300) :
  length * width * rate = 6187.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l1198_119888


namespace NUMINAMATH_CALUDE_crystal_beads_cost_l1198_119828

/-- The cost of one set of crystal beads -/
def crystal_cost : ℝ := sorry

/-- The cost of one set of metal beads -/
def metal_cost : ℝ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends -/
def total_spent : ℝ := 29

theorem crystal_beads_cost :
  crystal_cost = 9 :=
by
  have h1 : crystal_cost + metal_cost * metal_sets = total_spent := sorry
  sorry

end NUMINAMATH_CALUDE_crystal_beads_cost_l1198_119828


namespace NUMINAMATH_CALUDE_no_function_satisfies_equation_l1198_119859

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
sorry

end NUMINAMATH_CALUDE_no_function_satisfies_equation_l1198_119859


namespace NUMINAMATH_CALUDE_shooting_probability_l1198_119887

theorem shooting_probability (accuracy : ℝ) (consecutive_hits : ℝ) 
  (h1 : accuracy = 9/10) 
  (h2 : consecutive_hits = 1/2) : 
  consecutive_hits / accuracy = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_l1198_119887


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1198_119896

theorem trig_expression_simplification :
  let num := Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + Real.sin (60 * π / 180) + Real.sin (80 * π / 180) +
             Real.sin (100 * π / 180) + Real.sin (120 * π / 180) + Real.sin (140 * π / 180) + Real.sin (160 * π / 180)
  let den := Real.cos (15 * π / 180) * Real.cos (30 * π / 180) * Real.cos (45 * π / 180)
  num / den = (16 * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.cos (10 * π / 180)) /
              (Real.cos (15 * π / 180) * Real.cos (30 * π / 180) * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_trig_expression_simplification_l1198_119896


namespace NUMINAMATH_CALUDE_problem_solution_l1198_119830

theorem problem_solution (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1198_119830


namespace NUMINAMATH_CALUDE_f_range_l1198_119845

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.log (1 + x) + x^2
  else -x * Real.log (1 - x) + x^2

theorem f_range (a : ℝ) : f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l1198_119845


namespace NUMINAMATH_CALUDE_greatest_common_measure_l1198_119880

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l1198_119880


namespace NUMINAMATH_CALUDE_unique_five_step_palindrome_l1198_119826

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := n = reverseNum n

/-- Performs one step of the reverse-and-add process -/
def reverseAndAdd (n : ℕ) : ℕ := n + reverseNum n

/-- Counts the number of steps required to reach a palindrome -/
def stepsToParindrome (n : ℕ) : ℕ := sorry

theorem unique_five_step_palindrome :
  ∃! n : ℕ, 200 ≤ n ∧ n < 300 ∧ ¬isPalindrome n ∧ stepsToParindrome n = 5 ∧ n = 237 := by sorry

end NUMINAMATH_CALUDE_unique_five_step_palindrome_l1198_119826


namespace NUMINAMATH_CALUDE_neil_final_three_prob_l1198_119805

/-- A 3-sided die with numbers 1, 2, and 3 -/
inductive Die : Type
| one : Die
| two : Die
| three : Die

/-- The probability of rolling each number on the die -/
def prob_roll (d : Die) : ℚ := 1/3

/-- The event of Neil's final number being 3 -/
def neil_final_three : Set (Die × Die) := {(j, n) | n = Die.three}

/-- The probability space of all possible outcomes (Jerry's roll, Neil's final roll) -/
def prob_space : Set (Die × Die) := Set.univ

/-- The theorem stating the probability of Neil's final number being 3 -/
theorem neil_final_three_prob :
  ∃ (P : Set (Die × Die) → ℚ),
    P prob_space = 1 ∧
    P neil_final_three = 11/18 :=
sorry

end NUMINAMATH_CALUDE_neil_final_three_prob_l1198_119805


namespace NUMINAMATH_CALUDE_chair_carrying_trips_l1198_119843

/-- Proves that given 5 students, each carrying 5 chairs per trip, and a total of 250 chairs moved, the number of trips each student made is 10 -/
theorem chair_carrying_trips 
  (num_students : ℕ) 
  (chairs_per_trip : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5)
  (h2 : chairs_per_trip = 5)
  (h3 : total_chairs = 250) :
  (total_chairs / (num_students * chairs_per_trip) : ℕ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chair_carrying_trips_l1198_119843


namespace NUMINAMATH_CALUDE_opposite_numbers_quotient_l1198_119819

theorem opposite_numbers_quotient (p q : ℝ) (h1 : p ≠ 0) (h2 : p + q = 0) : |q| / p = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_quotient_l1198_119819


namespace NUMINAMATH_CALUDE_remainder_theorem_l1198_119854

theorem remainder_theorem (x : ℤ) : x % 66 = 14 → x % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1198_119854


namespace NUMINAMATH_CALUDE_fraction_of_savings_used_for_bills_l1198_119871

def weekly_savings : ℚ := 25
def weeks_saved : ℕ := 6
def dad_contribution : ℚ := 70
def coat_cost : ℚ := 170

theorem fraction_of_savings_used_for_bills :
  let total_savings := weekly_savings * weeks_saved
  let remaining_cost := coat_cost - dad_contribution
  let amount_for_bills := total_savings - remaining_cost
  amount_for_bills / total_savings = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_savings_used_for_bills_l1198_119871


namespace NUMINAMATH_CALUDE_intersection_condition_l1198_119897

open Set Real

def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
def B (b : ℝ) : Set ℝ := {x | (x - b)^2 < 1}

theorem intersection_condition (b : ℝ) : A ∩ B b ≠ ∅ ↔ -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1198_119897


namespace NUMINAMATH_CALUDE_dogwood_trees_theorem_l1198_119837

def dogwood_trees_problem (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow

theorem dogwood_trees_theorem (initial_trees planted_today planted_tomorrow : ℕ) :
  dogwood_trees_problem initial_trees planted_today planted_tomorrow =
  initial_trees + planted_today + planted_tomorrow :=
by
  sorry

#eval dogwood_trees_problem 7 5 4

end NUMINAMATH_CALUDE_dogwood_trees_theorem_l1198_119837


namespace NUMINAMATH_CALUDE_min_positive_period_sin_l1198_119864

/-- The minimum positive period of the function y = 3 * sin(2x + π/4) is π -/
theorem min_positive_period_sin (x : ℝ) : 
  let f := fun x => 3 * Real.sin (2 * x + π / 4)
  ∃ p : ℝ, p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
    ∀ q : ℝ, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_min_positive_period_sin_l1198_119864


namespace NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l1198_119823

/-- A parabola with vertex (h, k) has the general form y = a(x - h)² + k, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) (h k a : ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k ∧ a ≠ 0

/-- The vertex of a parabola f is the point (h, k) -/
def has_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∃ a : ℝ, is_parabola f h k a

theorem parabola_with_vertex_two_three (f : ℝ → ℝ) :
  has_vertex f 2 3 → (∀ x, f x = -(x - 2)^2 + 3) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_with_vertex_two_three_l1198_119823


namespace NUMINAMATH_CALUDE_school_play_tickets_l1198_119822

/-- Calculates the total number of tickets sold for a school play. -/
def total_tickets (adult_tickets : ℕ) : ℕ :=
  adult_tickets + 2 * adult_tickets

/-- Theorem: Given 122 adult tickets and student tickets being twice the number of adult tickets,
    the total number of tickets sold is 366. -/
theorem school_play_tickets : total_tickets 122 = 366 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l1198_119822


namespace NUMINAMATH_CALUDE_max_sum_squares_and_products_l1198_119808

def S : Finset ℕ := {2, 4, 6, 8}

theorem max_sum_squares_and_products (f g h j : ℕ) 
  (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S) (hj : j ∈ S)
  (hsum : f + g + h + j = 20) :
  (∃ (f' g' h' j' : ℕ), f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧ j' ∈ S ∧ 
    f' + g' + h' + j' = 20 ∧
    f'^2 + g'^2 + h'^2 + j'^2 ≤ 120 ∧
    (f'^2 + g'^2 + h'^2 + j'^2 = 120 → 
      f' * g' + g' * h' + h' * j' + f' * j' = 100)) ∧
  f^2 + g^2 + h^2 + j^2 ≤ 120 ∧
  (f^2 + g^2 + h^2 + j^2 = 120 → 
    f * g + g * h + h * j + f * j = 100) :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_and_products_l1198_119808


namespace NUMINAMATH_CALUDE_john_rejection_rate_proof_l1198_119872

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.7

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.75

/-- The ratio of products Jane inspected compared to John -/
def jane_inspection_ratio : ℝ := 1.25

/-- John's rejection rate -/
def john_rejection_rate : ℝ := 0.8125

theorem john_rejection_rate_proof :
  let total_products := 1 + jane_inspection_ratio
  jane_rejection_rate * jane_inspection_ratio + john_rejection_rate = total_rejection_rate * total_products :=
by sorry

end NUMINAMATH_CALUDE_john_rejection_rate_proof_l1198_119872


namespace NUMINAMATH_CALUDE_prime_sequence_multiple_of_six_l1198_119889

theorem prime_sequence_multiple_of_six (a d : ℤ) : 
  (Prime a ∧ a > 3) ∧ 
  (Prime (a + d) ∧ (a + d) > 3) ∧ 
  (Prime (a + 2*d) ∧ (a + 2*d) > 3) → 
  ∃ k : ℤ, d = 6 * k :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_multiple_of_six_l1198_119889


namespace NUMINAMATH_CALUDE_boosters_club_average_sales_l1198_119813

/-- Calculates the average monthly sales for the Boosters Club --/
theorem boosters_club_average_sales
  (sales : List ℝ)
  (refund : ℝ)
  (h1 : sales = [90, 75, 55, 130, 110, 85])
  (h2 : refund = 25)
  (h3 : sales.length = 6) :
  (sales.sum - refund) / sales.length = 86.67 := by
  sorry

end NUMINAMATH_CALUDE_boosters_club_average_sales_l1198_119813


namespace NUMINAMATH_CALUDE_pool_volume_l1198_119898

/-- The volume of a cylindrical pool minus a central cylindrical pillar -/
theorem pool_volume (pool_diameter : ℝ) (pool_depth : ℝ) (pillar_diameter : ℝ) (pillar_depth : ℝ)
  (h1 : pool_diameter = 20)
  (h2 : pool_depth = 5)
  (h3 : pillar_diameter = 4)
  (h4 : pillar_depth = 5) :
  (π * (pool_diameter / 2)^2 * pool_depth) - (π * (pillar_diameter / 2)^2 * pillar_depth) = 480 * π := by
  sorry

#check pool_volume

end NUMINAMATH_CALUDE_pool_volume_l1198_119898


namespace NUMINAMATH_CALUDE_parallel_lines_plane_count_l1198_119858

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the specifics of a line for this problem

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Function to count the number of planes determined by three lines -/
def count_planes (l1 l2 l3 : Line3D) : ℕ :=
  sorry

/-- Theorem: The number of planes determined by three mutually parallel lines is either 1 or 3 -/
theorem parallel_lines_plane_count (l1 l2 l3 : Line3D) 
  (h1 : are_parallel l1 l2) 
  (h2 : are_parallel l2 l3) 
  (h3 : are_parallel l1 l3) : 
  count_planes l1 l2 l3 = 1 ∨ count_planes l1 l2 l3 = 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_plane_count_l1198_119858


namespace NUMINAMATH_CALUDE_monomial_degree_implies_a_value_l1198_119894

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, prove that a = -2 -/
theorem monomial_degree_implies_a_value (a : ℤ) : 
  (∃ (x y : ℚ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) ∧ 
  (2 + (|a| + 1) = 5) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_monomial_degree_implies_a_value_l1198_119894


namespace NUMINAMATH_CALUDE_harriet_miles_l1198_119868

theorem harriet_miles (total_miles : ℕ) (katarina_miles : ℕ) 
  (h1 : total_miles = 195)
  (h2 : katarina_miles = 51)
  (h3 : ∃ x : ℕ, x * 3 + katarina_miles = total_miles) :
  ∃ harriet_miles : ℕ, harriet_miles = 48 ∧ 
    harriet_miles * 3 + katarina_miles = total_miles := by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_l1198_119868


namespace NUMINAMATH_CALUDE_evaluate_P_l1198_119892

/-- The polynomial P(x) = x^3 - 6x^2 - 5x + 4 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

/-- Theorem stating that under given conditions, P(y) = -22 -/
theorem evaluate_P (y z : ℝ) (h : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P y = -22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_P_l1198_119892


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_8_l1198_119803

theorem binomial_coefficient_20_8 :
  let n : ℕ := 20
  let k : ℕ := 8
  let binomial := Nat.choose
  binomial 18 5 = 8568 →
  binomial 18 7 = 31824 →
  binomial n k = 83656 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_8_l1198_119803


namespace NUMINAMATH_CALUDE_characterization_of_valid_n_l1198_119891

def floor_sqrt (n : ℕ) : ℕ := Nat.sqrt n

def is_valid (n : ℕ) : Prop :=
  (n > 0) ∧
  (∃ k₁ : ℕ, n - 4 = k₁ * (floor_sqrt n - 2)) ∧
  (∃ k₂ : ℕ, n + 4 = k₂ * (floor_sqrt n + 2))

def special_set : Set ℕ := {2, 4, 11, 20, 31, 36, 44}

def general_form (a : ℕ) : ℕ := a^2 + 2*a - 4

theorem characterization_of_valid_n :
  ∀ n : ℕ, is_valid n ↔ (n ∈ special_set ∨ ∃ a : ℕ, a > 2 ∧ n = general_form a) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_valid_n_l1198_119891


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_C_subset_B_iff_l1198_119873

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 2) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

theorem C_subset_B_iff (a : ℝ) : C a ⊆ B ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_C_subset_B_iff_l1198_119873


namespace NUMINAMATH_CALUDE_linear_system_solution_l1198_119807

theorem linear_system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y + z = 0 →
  3*x + 6*y + 5*z = 0 →
  (k = 90/41 ∧ y*z/x^2 = 41/30) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1198_119807


namespace NUMINAMATH_CALUDE_average_problem_k_problem_point_problem_quadratic_problem_l1198_119811

-- Question 1
theorem average_problem (p q r t : ℝ) :
  (p + q + r) / 3 = 12 ∧ (p + q + r + t + 2*t) / 5 = 15 → t = 13 := by sorry

-- Question 2
theorem k_problem (k s : ℝ) :
  k^4 + 1/k^4 = 14 ∧ s = k^2 + 1/k^2 → s = 4 := by sorry

-- Question 3
theorem point_problem (a b s : ℝ) :
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (11, 7)
  let P : ℝ × ℝ := (a, b)
  P.1 = (1 * N.1 + s * M.1) / (1 + s) ∧
  P.2 = (1 * N.2 + s * M.2) / (1 + s) ∧
  s = 4 → a = 3 := by sorry

-- Question 4
theorem quadratic_problem (a c : ℝ) :
  a = 3 ∧ (∃ x : ℝ, a * x^2 + 12 * x + c = 0 ∧
    ∀ y : ℝ, y ≠ x → a * y^2 + 12 * y + c ≠ 0) → c = 12 := by sorry

end NUMINAMATH_CALUDE_average_problem_k_problem_point_problem_quadratic_problem_l1198_119811


namespace NUMINAMATH_CALUDE_x_equals_5y_when_squared_difference_equal_l1198_119886

theorem x_equals_5y_when_squared_difference_equal
  (x y : ℕ) -- x and y are natural numbers
  (h : x^2 - 3*x = 25*y^2 - 15*y) -- given equation
  : x = 5*y := by
sorry

end NUMINAMATH_CALUDE_x_equals_5y_when_squared_difference_equal_l1198_119886


namespace NUMINAMATH_CALUDE_complex_quadrant_problem_l1198_119862

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_quadrant_problem (a : ℝ) :
  is_purely_imaginary (Complex.mk (a^2 - 3*a - 4) (a - 4)) →
  a = -1 ∧ a < 0 ∧ -a > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_quadrant_problem_l1198_119862


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1198_119820

def teaching_problem (v a d e n : ℕ) : Prop :=
  v + a + d + e + n = 225 ∧
  v = a + 9 ∧
  v = d - 15 ∧
  e = a - 3 ∧
  e = n + 7

theorem dennis_teaching_years :
  ∀ v a d e n : ℕ, teaching_problem v a d e n → d = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1198_119820


namespace NUMINAMATH_CALUDE_measuring_cup_size_proof_l1198_119825

/-- The size of the measuring cup in cups -/
def measuring_cup_size : ℚ := 1/4

theorem measuring_cup_size_proof (total_flour : ℚ) (flour_needed : ℚ) (scoops_to_remove : ℕ) 
  (h1 : total_flour = 8)
  (h2 : flour_needed = 6)
  (h3 : scoops_to_remove = 8)
  (h4 : total_flour - scoops_to_remove * measuring_cup_size = flour_needed) : 
  measuring_cup_size = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_measuring_cup_size_proof_l1198_119825


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_l1198_119832

theorem square_triangle_perimeter (square_perimeter : ℝ) :
  square_perimeter = 160 →
  let side_length := square_perimeter / 4
  let diagonal_length := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal_length
  triangle_perimeter = 80 + 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_perimeter_l1198_119832


namespace NUMINAMATH_CALUDE_rectangular_plot_theorem_l1198_119875

/-- Represents a rectangular plot with given properties --/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Theorem stating the properties of the rectangular plot --/
theorem rectangular_plot_theorem (plot : RectangularPlot) : 
  plot.length = plot.breadth + plot.length_breadth_difference ∧
  plot.total_fencing_cost = plot.fencing_cost_per_meter * (2 * plot.length + 2 * plot.breadth) ∧
  plot.length = 65 ∧
  plot.fencing_cost_per_meter = 26.5 ∧
  plot.total_fencing_cost = 5300 →
  plot.length_breadth_difference = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_theorem_l1198_119875


namespace NUMINAMATH_CALUDE_fraction_calculation_l1198_119846

theorem fraction_calculation : (5 / 6 : ℚ) * (1 / ((7 / 8 : ℚ) - (3 / 4 : ℚ))) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1198_119846


namespace NUMINAMATH_CALUDE_philip_paintings_l1198_119838

/-- The number of paintings a painter will have after a certain number of days -/
def total_paintings (initial_paintings : ℕ) (paintings_per_day : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings : total_paintings 20 2 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_l1198_119838


namespace NUMINAMATH_CALUDE_billy_carnival_tickets_l1198_119856

theorem billy_carnival_tickets : ∀ (ferris_rides bumper_rides ticket_per_ride : ℕ),
  ferris_rides = 7 →
  bumper_rides = 3 →
  ticket_per_ride = 5 →
  (ferris_rides + bumper_rides) * ticket_per_ride = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_carnival_tickets_l1198_119856


namespace NUMINAMATH_CALUDE_three_top_numbers_count_l1198_119809

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if three consecutive numbers satisfy the "Three Top Numbers" conditions -/
def isThreeTopNumbers (n : ℕ) : Prop :=
  isTwoDigit n ∧ isTwoDigit (n + 1) ∧ isTwoDigit (n + 2) ∧
  isTwoDigit (n + (n + 1) + (n + 2)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit n) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 1)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 2))

/-- The theorem stating that there are exactly 5 sets of "Three Top Numbers" -/
theorem three_top_numbers_count :
  ∃! (s : Finset ℕ), (∀ n ∈ s, isThreeTopNumbers n) ∧ s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_three_top_numbers_count_l1198_119809


namespace NUMINAMATH_CALUDE_lowest_dropped_score_l1198_119865

theorem lowest_dropped_score (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  (a + b + c + d) / 4 = 45 →
  d ≤ a ∧ d ≤ b ∧ d ≤ c →
  (a + b + c) / 3 = 50 →
  d = 30 := by
sorry

end NUMINAMATH_CALUDE_lowest_dropped_score_l1198_119865


namespace NUMINAMATH_CALUDE_total_tickets_l1198_119824

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys : ℕ := 12

/-- The number of tickets Dave used to buy clothes -/
def tickets_for_clothes : ℕ := 7

/-- The difference between tickets used for toys and clothes -/
def tickets_difference : ℕ := 5

/-- Theorem: Given the conditions, Dave won 19 tickets in total -/
theorem total_tickets : 
  tickets_for_toys + tickets_for_clothes = 19 ∧
  tickets_for_toys = tickets_for_clothes + tickets_difference :=
sorry

end NUMINAMATH_CALUDE_total_tickets_l1198_119824


namespace NUMINAMATH_CALUDE_min_exponent_sum_l1198_119879

theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (h1 : A = 2^α * 3^β * 5^γ)
  (h2 : ∃ (k : ℕ), A / 2 = k^2)
  (h3 : ∃ (m : ℕ), A / 3 = m^3)
  (h4 : ∃ (n : ℕ), A / 5 = n^5) :
  α + β + γ ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l1198_119879
