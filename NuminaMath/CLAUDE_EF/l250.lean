import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_length_problem_l250_25000

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b : V)

-- State the theorem
theorem vector_length_problem 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖a + b‖ = Real.sqrt 7) 
  (h3 : inner a b = Real.pi / 3) : 
  ‖b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_length_problem_l250_25000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l250_25071

-- Define the line l as a function of m
def line_l (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m - 2) * x + (m + 1) * y - 3 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 4)

-- Theorem statement
theorem line_l_properties :
  (∀ m : ℝ, ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ line_l m x y) ∧
  (line_l (1/2) ≠ λ x y ↦ y = Real.tan (3 * π / 4) * x + 0) ∧
  (line_l 1 ≠ λ x y ↦ x + 2*y - 3 = 0) ∧
  (∀ m : ℝ, ∃ d : ℝ, d > 3*Real.sqrt 2 ∧
    (∀ x y : ℝ, line_l m x y →
      (x - point_P.1)^2 + (y - point_P.2)^2 ≤ d^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l250_25071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l250_25096

def g : ℕ → ℕ
| n => if n < 12 then n + 12 else g (n - 7)

theorem g_max_value : (∀ n : ℕ, g n ≤ 23) ∧ (∃ m : ℕ, g m = 23) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l250_25096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l250_25085

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 6 →
  num_girls = 4 →
  avg_weight_boys = 150 →
  avg_weight_girls = 120 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 138 := by
  intro h_boys h_girls h_avg_boys h_avg_girls
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_of_children_l250_25085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zibo_barbecue_pricing_l250_25038

/-- Represents the barbecue shop's sales and pricing model -/
structure BarbecueShop where
  cost_per_skewer : ℚ
  base_price : ℚ
  base_sales : ℚ
  price_change : ℚ
  sales_change : ℚ
  max_allowed_price : ℚ

/-- Calculates the number of skewers sold at a given price -/
noncomputable def skewers_sold (shop : BarbecueShop) (price : ℚ) : ℚ :=
  shop.base_sales + shop.sales_change * (shop.base_price - price) / shop.price_change

/-- Calculates the profit at a given price -/
noncomputable def profit (shop : BarbecueShop) (price : ℚ) : ℚ :=
  (skewers_sold shop price) * (price - shop.cost_per_skewer)

/-- The Zibo barbecue shop -/
def zibo_shop : BarbecueShop :=
  { cost_per_skewer := 3
  , base_price := 10
  , base_sales := 300
  , price_change := 1/2
  , sales_change := 30
  , max_allowed_price := 8 }

theorem zibo_barbecue_pricing :
  (∃ (max_price : ℚ), max_price ≤ 9 ∧ max_price > 0 ∧
    ∀ (price : ℚ), price > max_price → skewers_sold zibo_shop price < 360) ∧
  (∃ (optimal_price : ℚ), optimal_price = 7 ∧
    profit zibo_shop optimal_price = 1920 ∧
    optimal_price ≤ zibo_shop.max_allowed_price) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zibo_barbecue_pricing_l250_25038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l250_25045

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (9 - x^2) / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ∈ Set.Icc (-3) 3 ∧ x ≠ 1}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y ∧ (9 - x^2 ≥ 0) ∧ (x ≠ 1)} = domain_f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l250_25045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_is_40_minutes_l250_25066

/-- The time (in hours) it takes P to finish the job alone -/
noncomputable def p_time : ℝ := 3

/-- The time (in hours) it takes Q to finish the job alone -/
noncomputable def q_time : ℝ := 18

/-- The time (in hours) P and Q work together -/
noncomputable def together_time : ℝ := 2

/-- The portion of the job completed in one hour when P and Q work together -/
noncomputable def combined_rate : ℝ := 1 / p_time + 1 / q_time

/-- The portion of the job completed when P and Q work together -/
noncomputable def completed_portion : ℝ := combined_rate * together_time

/-- The remaining portion of the job after P and Q work together -/
noncomputable def remaining_portion : ℝ := 1 - completed_portion

/-- The additional time (in hours) it takes P to finish the remaining portion -/
noncomputable def additional_time : ℝ := remaining_portion * p_time

theorem additional_time_is_40_minutes : 
  additional_time * 60 = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_is_40_minutes_l250_25066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sum_limit_fractional_part_diff_limit_counterexample_l250_25094

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem fractional_part_sum_limit (x y : ℕ → ℝ)
  (hx : ∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (x n) < ε)
  (hy : ∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (y n) < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (x n + y n) < ε := by
  sorry

theorem fractional_part_diff_limit_counterexample :
  ∃ x y : ℕ → ℝ,
    (∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (x n) < ε) ∧
    (∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (y n) < ε) ∧
    ¬(∀ ε > 0, ∃ N, ∀ n ≥ N, fractional_part (x n - y n) < ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_part_sum_limit_fractional_part_diff_limit_counterexample_l250_25094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calcium_hydroxide_pH_l250_25065

/-- Represents the concentration of calcium hydroxide in mol/L -/
def C_CaOH2 : ℝ := sorry

/-- Represents the concentration of hydroxide ions in mol/L -/
def OH_conc : ℝ := 2 * C_CaOH2

/-- Calculates pH based on hydroxide ion concentration -/
noncomputable def pH : ℝ := 14 + Real.log OH_conc

/-- Theorem stating that the pH of the calcium hydroxide solution is 12.6 -/
theorem calcium_hydroxide_pH : pH = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calcium_hydroxide_pH_l250_25065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l250_25023

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 3 < x ∧ x < 9}

-- Define set C parameterized by a
def C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 1}

-- State the theorem
theorem sets_theorem :
  (Set.univ \ (A ∩ B) = {x : ℝ | x ≤ 3 ∨ x ≥ 6}) ∧
  ((Set.univ \ B) ∪ A = {x : ℝ | x < 6 ∨ x ≥ 9}) ∧
  (∀ a : ℝ, C a ⊆ B → 3 ≤ a ∧ a ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_theorem_l250_25023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_prime_factor_of_2021_pow_8_plus_1_l250_25029

theorem smallest_odd_prime_factor_of_2021_pow_8_plus_1 :
  ∃ (p : ℕ), p = 17 ∧ Nat.Prime p ∧ p ∣ (2021^8 + 1) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (2021^8 + 1) → Odd q → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_prime_factor_of_2021_pow_8_plus_1_l250_25029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l250_25086

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 1)^2 + 2

-- Define what we mean by an extreme value
def ExtremeValue (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem extreme_values_of_f :
  ∀ x : ℝ, ExtremeValue f x → (x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_f_l250_25086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_schemes_l250_25022

def number_of_distribution_schemes (n k m : ℕ) : ℕ := 
  Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m

theorem student_distribution_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 12) 
  (h2 : k = 3) 
  (h3 : m = 4) 
  (h4 : n = k * m) : 
  number_of_distribution_schemes n k m = 
  Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m :=
by
  unfold number_of_distribution_schemes
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_distribution_schemes_l250_25022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_equality_l250_25059

-- Define the universal set S
variable (S : Type)

-- Define subsets A and B of S
variable (A B : Set S)

-- State the theorem
theorem subset_intersection_equality
  (h1 : B ⊆ A)
  (h2 : A ⊆ Set.univ) :
  A ∩ B = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_equality_l250_25059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leroy_balance_correct_adjustments_sum_to_zero_l250_25088

/-- The amount LeRoy should give to balance the costs of a camping trip -/
noncomputable def leroy_balance (X Y Z : ℝ) : ℝ :=
  (Y + Z - 2 * X) / 3

/-- Theorem stating that LeRoy's balance amount is correct -/
theorem leroy_balance_correct (X Y Z : ℝ) :
  let total := X + Y + Z
  let equal_share := total / 3
  leroy_balance X Y Z = equal_share - X := by
  sorry

/-- Theorem stating that the sum of all adjustments is zero -/
theorem adjustments_sum_to_zero (X Y Z : ℝ) :
  let total := X + Y + Z
  let equal_share := total / 3
  (leroy_balance X Y Z) + (equal_share - Y) + (equal_share - Z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leroy_balance_correct_adjustments_sum_to_zero_l250_25088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l250_25068

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 2)^2 - (1/2)

theorem quadratic_function_properties :
  (f (-1) = 4 ∧ f 1 = 0 ∧ f 3 = 0) ∧
  (∀ x : ℝ, f x ≥ -(1/2)) ∧
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 4 → f x₁ = f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l250_25068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l250_25003

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

-- State the theorem
theorem f_strictly_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l250_25003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_2147_l250_25063

/-- The number of digits in the base-8 representation of a positive integer -/
noncomputable def num_digits_base8 (n : ℕ+) : ℕ :=
  Nat.floor (Real.log n / Real.log 8) + 1

/-- Theorem: The number of digits in the base-8 representation of 2147 is 4 -/
theorem base8_digits_of_2147 : num_digits_base8 2147 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_digits_of_2147_l250_25063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_Z_at_zero_Z_purely_imaginary_Z_in_second_quadrant_l250_25024

-- Define the complex number Z as a function of m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 3*m - 4) (m^2 - 10*m + 9)

-- Theorem 1: Modulus of Z when m = 0
theorem modulus_Z_at_zero : Complex.abs (Z 0) = Real.sqrt 97 := by sorry

-- Theorem 2: Z is purely imaginary iff m = -4
theorem Z_purely_imaginary (m : ℝ) : Z m = Complex.I * Complex.im (Z m) ↔ m = -4 := by sorry

-- Theorem 3: Z lies in the second quadrant iff -4 < m < 1
theorem Z_in_second_quadrant (m : ℝ) : 
  Complex.re (Z m) < 0 ∧ Complex.im (Z m) > 0 ↔ -4 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_Z_at_zero_Z_purely_imaginary_Z_in_second_quadrant_l250_25024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l250_25077

theorem magnitude_relationship : 
  ∃ (a b c : ℝ), 
    a = (1/2)^(1/3) ∧ 
    b = Real.log 3 / Real.log 2 ∧ 
    c = Real.log 7 / Real.log 4 ∧ 
    a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l250_25077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_equals_2CP_squared_iff_midpoint_l250_25009

/-- A 30-60-90 triangle ABC with hypotenuse AB -/
structure Triangle30_60_90 where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_30_60_90 : Prop
  AB_hypotenuse : Prop

/-- Point P on line AB or its extension -/
def P (t : Triangle30_60_90) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The squared distance s -/
def s (t : Triangle30_60_90) : ℝ :=
  (distance (P t) t.A)^2 + (distance (P t) t.B)^2

/-- The altitude CP -/
def CP (t : Triangle30_60_90) : ℝ := sorry

/-- Midpoint of a line segment -/
def is_midpoint (p a b : ℝ × ℝ) : Prop := sorry

theorem s_equals_2CP_squared_iff_midpoint (t : Triangle30_60_90) :
  s t = 2 * (CP t)^2 ↔ is_midpoint (P t) t.A t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_equals_2CP_squared_iff_midpoint_l250_25009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l250_25027

theorem sin_alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : π/2 < β ∧ β < π)
  (h3 : Real.sin (α + β) = 3/5)
  (h4 : Real.cos β = -4/5) :
  Real.sin α = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l250_25027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_bound_on_triples_l250_25075

/-- Given a set of unordered pairs of positive integers, prove a lower bound on the number of triples forming a cycle. -/
theorem lower_bound_on_triples (n m : ℕ) (S : Finset (ℕ × ℕ)) 
  (h1 : ∀ (p : ℕ × ℕ), p ∈ S → 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n)
  (h2 : ∀ (p : ℕ × ℕ), p ∈ S → p.1 ≠ p.2)
  (h3 : S.card = m) :
  (∃ T : Finset (ℕ × ℕ × ℕ), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ T → (t.1, t.2.1) ∈ S ∧ (t.2.1, t.2.2) ∈ S ∧ (t.2.2, t.1) ∈ S) ∧
    T.card ≥ (4 * m : ℚ) / (3 * n) * (m - n^2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_bound_on_triples_l250_25075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_equality_l250_25073

theorem gcd_lcm_equality (a b c : ℕ+) :
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd a c * Nat.gcd b c) =
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm a c * Nat.lcm b c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_equality_l250_25073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_2_not_sqrt_30_l250_25016

-- Define the polynomial P(x)
def P (a x : ℝ) : ℝ := x^3 + a*x + 1

-- State the theorem
theorem P_2_not_sqrt_30 (a : ℝ) :
  (∃! x, x ∈ Set.Icc (-2) 0 ∧ P a x = 0) ∧
  (∃! x, x ∈ Set.Ioo 0 1 ∧ P a x = 0) →
  P a 2 ≠ Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_2_not_sqrt_30_l250_25016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equation_l250_25097

theorem sine_sum_equation (α : ℝ) :
  Real.sin (π/3 + α) + Real.sin α = 4 * Real.sqrt 3 / 5 →
  Real.sin (α + 7 * π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_equation_l250_25097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_property_l250_25048

noncomputable def θ : ℝ := Real.arccos (3 * Real.sqrt 10 / 10)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 10 * Real.sin (2 * x - θ)

theorem roots_property (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g x₁ = a) (h₂ : g x₂ = a) 
  (h₃ : 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2) 
  (h₄ : 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2) :
  Real.sin (2 * x₁ + 2 * x₂) = -3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_property_l250_25048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equality_l250_25019

/-- Definition of binomial coefficient for real numbers and nonnegative integers -/
noncomputable def binomial (x : ℝ) (k : ℕ) : ℝ :=
  (Finset.range k).prod (fun i => (x - ↑i) / (↑k - ↑i))

/-- Theorem stating the equality of the given expression -/
theorem binomial_equality :
  (binomial (1/2) 1007 * 8^1007) / binomial 2014 1007 = -8 * 2^2014 / Nat.factorial 2014 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equality_l250_25019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l250_25025

theorem die_roll_probability : (35 : ℚ) / 972 = 
  let n : ℕ := 8  -- Total number of rolls
  let k : ℕ := 6  -- Number of odd rolls
  let m : ℕ := 2  -- Number of 3s among odd rolls
  let p_odd : ℚ := 1/2  -- Probability of rolling an odd number
  let p_three : ℚ := 1/3  -- Probability of rolling a 3 given an odd roll

  (Nat.choose n k : ℚ) * p_odd^k * (1 - p_odd)^(n - k) *
  (Nat.choose k m : ℚ) * p_three^m * (1 - p_three)^(k - m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_die_roll_probability_l250_25025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_eq_one_third_expression_eq_sqrt_ten_over_ten_l250_25043

noncomputable section

variable (α : Real)

-- α is an acute angle
axiom α_acute : 0 < α ∧ α < Real.pi / 2

-- tan(π/4 + α) = 2
axiom tan_sum_eq_two : Real.tan (Real.pi / 4 + α) = 2

-- Theorem 1: tan α = 1/3
theorem tan_α_eq_one_third : Real.tan α = 1 / 3 := by sorry

-- Theorem 2: (sin 2α cos α - sin α) / cos 2α = √10/10
theorem expression_eq_sqrt_ten_over_ten :
  (Real.sin (2 * α) * Real.cos α - Real.sin α) / Real.cos (2 * α) = Real.sqrt 10 / 10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_α_eq_one_third_expression_eq_sqrt_ten_over_ten_l250_25043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l250_25037

/-- Inverse variation function with a constant multiplicative factor -/
noncomputable def inverse_variation (k : ℝ) (x : ℝ) : ℝ := 3 * k / x

theorem inverse_variation_problem (k : ℝ) :
  inverse_variation k 4 = 8 →
  inverse_variation k (-16) = -6 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check inverse_variation_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l250_25037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_integers_inequality_l250_25055

theorem nine_integers_inequality (S : Finset ℕ) :
  S.card = 9 →
  (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 9000) →
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → x ≠ y) →
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    4 + d ≤ a + b + c ∧ a + b + c ≤ 4 * d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_integers_inequality_l250_25055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_range_theorem_l250_25062

-- Define the conditions
def condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (2:ℝ)^a * (4:ℝ)^b = 2

-- Part I: Minimum value theorem
theorem min_value_theorem (a b : ℝ) (h : condition a b) :
  (∀ a' b' : ℝ, condition a' b' → 2/a' + 1/b' ≥ 2/a + 1/b) → 2/a + 1/b = 8 :=
by sorry

-- Part II: Range theorem
theorem range_theorem (x : ℝ) :
  (∃ a b : ℝ, condition a b ∧ |x-1| + |2*x-3| ≥ 2/a + 1/b) ↔ 
  (x ≤ -4/3 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_range_theorem_l250_25062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l250_25018

/-- Represents the length of a candle stub after burning for a given time. -/
noncomputable def stubLength (initialLength : ℝ) (burnTime : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialLength * (burnTime - elapsedTime) / burnTime

/-- Proves that the candles must be lit at 10:40 AM to satisfy the given conditions. -/
theorem candle_lighting_time 
  (initialLength : ℝ) 
  (burnTimeA : ℝ) 
  (burnTimeB : ℝ) 
  (elapsedTime : ℝ) 
  (h1 : burnTimeA = 360)
  (h2 : burnTimeB = 480)
  (h3 : stubLength initialLength burnTimeB elapsedTime = 3 * stubLength initialLength burnTimeA elapsedTime) :
  elapsedTime = 320 := by
  sorry

#check candle_lighting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l250_25018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_y_coordinate_l250_25052

theorem angle_terminal_side_y_coordinate 
  (θ : Real) 
  (P : Real × Real) 
  (h1 : P.1 = 4) 
  (h2 : Real.sin θ = -2 * Real.sqrt 5 / 5) : 
  P.2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_y_coordinate_l250_25052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_theorem_l250_25020

/-- The total distance traveled by a squirrel on two concentric circles -/
noncomputable def squirrel_path_distance (r₁ r₂ : ℝ) : ℝ :=
  (1/4 * 2 * Real.pi * r₁) +  -- Quarter of smaller circle
  (r₂ - r₁) +                 -- Radial movement
  (1/2 * 2 * Real.pi * r₂) +  -- Half of larger circle
  (2 * r₁)                    -- Diameter of smaller circle

/-- Theorem: The squirrel's path on concentric circles with radii 15 and 25 meters -/
theorem squirrel_path_theorem :
  squirrel_path_distance 15 25 = 32.5 * Real.pi + 40 := by
  -- Unfold the definition of squirrel_path_distance
  unfold squirrel_path_distance
  
  -- Simplify the expression
  simp [Real.pi]
  
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_path_theorem_l250_25020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_when_parallel_range_of_f_l250_25072

-- Define the vectors
noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def p : ℝ × ℝ := (2 * Real.sqrt 3, 1)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem dot_product_when_parallel (x : ℝ) : 
  parallel (m x) p → dot_product (m x) (n x) = (2 * Real.sqrt 3 + 1) / 5 := by sorry

-- Theorem 2
theorem range_of_f (x : ℝ) : 
  x ∈ Set.Ioo 0 (Real.pi / 3) → f x ∈ Set.Icc 1 (3 / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_when_parallel_range_of_f_l250_25072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_sister_l250_25007

-- Define the participants
inductive Participant
  | Father
  | Sister
  | Son
  | Daughter

-- Define the skill level
inductive SkillLevel
  | Best
  | Worst

-- Define the sex
inductive Sex
  | Male
  | Female

-- Define the functions
def skill_level : Participant → SkillLevel → Prop := sorry
def age : Participant → ℕ := sorry
def sex : Participant → Sex := sorry

-- Define the conditions
axiom four_participants :
  ∀ p : Participant, p = Participant.Father ∨ p = Participant.Sister ∨ p = Participant.Son ∨ p = Participant.Daughter

axiom twins_skill :
  ∃ p1 p2 : Participant, p1 ≠ p2 ∧ skill_level p1 SkillLevel.Best ∧ skill_level p2 SkillLevel.Worst

axiom age_difference :
  ∃ p1 p2 : Participant, skill_level p1 SkillLevel.Best ∧ skill_level p2 SkillLevel.Worst ∧ age p1 = age p2 + 1

axiom different_sex :
  ∃ p1 p2 : Participant, skill_level p1 SkillLevel.Best ∧ skill_level p2 SkillLevel.Worst ∧ sex p1 ≠ sex p2

-- The theorem to prove
theorem worst_player_is_sister :
  skill_level Participant.Sister SkillLevel.Worst :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_sister_l250_25007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l250_25005

-- Define the set of colors for shorts and jerseys
def shorts_colors : Finset String := {"black", "gold", "red"}
def jersey_colors : Finset String := {"black", "white", "green"}

-- Define the total number of possible combinations
def total_combinations : ℕ := shorts_colors.card * jersey_colors.card

-- Define the number of combinations with different colors
def different_color_combinations : ℕ := 
  shorts_colors.card * jersey_colors.card - (shorts_colors ∩ jersey_colors).card

-- Theorem statement
theorem different_color_probability :
  (different_color_combinations : ℚ) / total_combinations = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l250_25005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_sum_l250_25021

theorem sin_cos_ratio_sum (u v : ℝ) 
  (h1 : Real.sin u / Real.sin v = 4)
  (h2 : Real.cos u / Real.cos v = 1/3) :
  Real.sin (2*u) / Real.sin (2*v) + Real.cos (2*u) / Real.cos (2*v) = 19/381 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_ratio_sum_l250_25021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_determines_q_l250_25046

/-- Given a triangle ABX where:
  A has coordinates (3, 10)
  B has coordinates (15, 0)
  X has coordinates (0, q)
  The area of triangle ABX is 58
  Then q = 43/6 -/
theorem triangle_area_determines_q : ∀ q : ℝ,
  let a : ℝ × ℝ := (3, 10)
  let b : ℝ × ℝ := (15, 0)
  let x : ℝ × ℝ := (0, q)
  let area_abx := 
    (1/2 : ℝ) * abs ((a.1 - x.1) * (b.2 - x.2) - (b.1 - x.1) * (a.2 - x.2))
  area_abx = 58 → q = 43/6 := by
  sorry

#check triangle_area_determines_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_determines_q_l250_25046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_ABP_l250_25014

-- Define the line l: 3x - 4y + 5 = 0
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define the circle C: x^2 + y^2 - 10x = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 10 * x = 0

-- Define points A and B as intersections of l and C
noncomputable def point_A : ℝ × ℝ := (1/5, 7/5)
noncomputable def point_B : ℝ × ℝ := (5, 5)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of triangle ABP
noncomputable def perimeter_ABP (P : ℝ × ℝ) : ℝ :=
  distance point_A point_B + distance point_A P + distance point_B P

-- Theorem: The minimum perimeter of triangle ABP is 14
theorem min_perimeter_ABP :
  ∀ P : ℝ × ℝ, P.2 = 0 → perimeter_ABP P ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_ABP_l250_25014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_is_integer_l250_25032

/-- Arithmetic progression -/
def arithmetic_progression (a d : ℚ) : ℕ → ℚ :=
  λ n => a + (n - 1 : ℚ) * d

/-- Geometric progression -/
def geometric_progression (b q : ℚ) : ℕ → ℚ :=
  λ n => b * q^(n - 1 : ℕ)

theorem geometric_ratio_is_integer
  (a d b q : ℚ)
  (h : ∀ n : ℕ, ∃ k : ℕ, geometric_progression b q n = arithmetic_progression a d k) :
  ∃ z : ℤ, q = z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_is_integer_l250_25032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_routes_existence_l250_25058

theorem bus_routes_existence :
  ∃ (S : Type) (R : Finset (Finset S)), 
    Finset.card R = 10 ∧ 
    (∀ (T : Finset (Finset S)), Finset.card T = 8 → T ⊆ R → 
      ∃ s : S, ∀ r ∈ T, s ∉ r) ∧
    (∀ (T : Finset (Finset S)), Finset.card T = 9 → T ⊆ R → 
      ∀ s : S, ∃ r ∈ T, s ∈ r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_routes_existence_l250_25058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l250_25060

/-- An ellipse with specific properties -/
structure Ellipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The focal distance -/
  c : ℝ
  /-- The vertex is at (a,0) -/
  vertex_condition : a = 2
  /-- The chord length condition -/
  chord_condition : 2 * b^2 / a = 1
  /-- The relationship between a, b, and c in an ellipse -/
  ellipse_condition : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating the eccentricity of the specific ellipse -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l250_25060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l250_25013

theorem triangle_side_length (A B : ℝ) (a b : ℝ) :
  A = π/4 → B = π/6 → a = 2 →
  Real.sin A / a = Real.sin B / b →
  b = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l250_25013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_identity_g_zero_point_range_h_equality_range_l250_25015

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 3*m + 3) * x^(2 - m^2)

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k-3)*x + 1

-- Define the functions h₁, h₂, and h₃
def h₁ (x : ℝ) : ℝ := |x^2 - 3|
def h₂ (x : ℝ) : ℝ := |h₁ x - 3|
def h₃ (x : ℝ) : ℝ := |h₂ x - 3|

-- Theorem 1
theorem f_is_identity (m : ℝ) : 
  (∀ x y : ℝ, x < y → f m x < f m y) → 
  (∀ x : ℝ, f m x = x) :=
by sorry

-- Theorem 2
theorem g_zero_point_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ g k x = 0) →
  k ≤ 1 :=
by sorry

-- Theorem 3
theorem h_equality_range :
  {x : ℝ | h₃ x = h₁ x} = {x : ℝ | -Real.sqrt 6 ≤ x ∧ x ≤ Real.sqrt 6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_identity_g_zero_point_range_h_equality_range_l250_25015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l250_25039

-- Define the coordinates of points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the distance function as noncomputable
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the theorem
theorem trajectory_of_C (C : ℝ × ℝ) :
  distance B C - distance A C = (1/2) * distance A B →
  C.1 < -2 →
  C.1^2 / 4 - C.2^2 / 12 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_C_l250_25039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_grid_half_direction_l250_25034

/-- Represents a direction in a hexagonal grid -/
inductive HexDirection
| A
| B
| C
deriving BEq, Repr

/-- Represents a path in a hexagonal grid -/
structure HexPath where
  length : ℕ
  moves : List HexDirection
deriving Repr

/-- Axiom: The sum of movements in all directions is zero for a closed path -/
axiom hex_closed_path_sum_zero (p : HexPath) : 
  p.moves.count HexDirection.A = p.moves.count HexDirection.B + p.moves.count HexDirection.C

/-- Definition of a shortest path in a hexagonal grid -/
def is_shortest_path (p : HexPath) : Prop :=
  ∀ q : HexPath, q.length = p.length → q.moves.length ≥ p.moves.length

/-- Main theorem: For any shortest path of length 100, there exists a direction with at least 50 moves -/
theorem hex_grid_half_direction (p : HexPath) 
  (h_length : p.length = 100) 
  (h_shortest : is_shortest_path p) : 
  ∃ d : HexDirection, p.moves.count d ≥ 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_grid_half_direction_l250_25034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l250_25064

-- Define the universe of students
variable (Student : Type)

-- Define predicates for being an honors student and attending the school
variable (isHonorsStudent : Student → Prop)
variable (attendsThisSchool : Student → Prop)

-- Define the original statement
def noHonorsStudentsAttend (Student : Type) (isHonorsStudent attendsThisSchool : Student → Prop) : Prop :=
  ¬∃ (s : Student), isHonorsStudent s ∧ attendsThisSchool s

-- Define the negation
def someHonorsStudentsAttend (Student : Type) (isHonorsStudent attendsThisSchool : Student → Prop) : Prop :=
  ∃ (s : Student), isHonorsStudent s ∧ attendsThisSchool s

-- Theorem stating that the negation of "No honors students attend this school"
-- is equivalent to "Some honors students attend this school"
theorem negation_equivalence (Student : Type) (isHonorsStudent attendsThisSchool : Student → Prop) :
  ¬(noHonorsStudentsAttend Student isHonorsStudent attendsThisSchool) ↔ 
  someHonorsStudentsAttend Student isHonorsStudent attendsThisSchool :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l250_25064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandcastle_tower_total_l250_25081

/-- The combined total number of sandcastles and towers on Mark's and Jeff's beaches -/
theorem sandcastle_tower_total 
  (mark_castles : ℕ) 
  (mark_towers_per_castle : ℕ) 
  (jeff_castles_multiplier : ℕ) 
  (jeff_towers_per_castle : ℕ) :
  mark_castles = 20 →
  mark_towers_per_castle = 10 →
  jeff_castles_multiplier = 3 →
  jeff_towers_per_castle = 5 →
  (mark_castles + jeff_castles_multiplier * mark_castles + 
   mark_castles * mark_towers_per_castle + 
   jeff_castles_multiplier * mark_castles * jeff_towers_per_castle) = 580 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandcastle_tower_total_l250_25081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_cost_theorem_l250_25076

/-- The cost of eel in dollars -/
noncomputable def eel_cost : ℝ := 180

/-- The ratio of eel cost to jellyfish cost -/
noncomputable def eel_to_jellyfish_ratio : ℝ := 9

/-- The cost of jellyfish in dollars -/
noncomputable def jellyfish_cost : ℝ := eel_cost / eel_to_jellyfish_ratio

/-- The combined cost of one order each of jellyfish and eel -/
noncomputable def combined_cost : ℝ := jellyfish_cost + eel_cost

theorem sushi_cost_theorem : combined_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_cost_theorem_l250_25076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_AB_distances_l250_25089

/-- An ellipse with focus F and vertices A and B -/
structure Ellipse where
  F : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  FA_dist : Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2) = 3
  FB_dist : Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 2

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the possible values of |AB| -/
theorem ellipse_AB_distances (Γ : Ellipse) :
  {d | d = distance Γ.A Γ.B} = {5, Real.sqrt 7, Real.sqrt 17} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_AB_distances_l250_25089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l250_25041

-- Define the set of real numbers that satisfy the inequality
def solution_set : Set ℝ :=
  {x | (2 * x) / (x + 2) ≤ 3}

-- State the theorem
theorem inequality_solution :
  solution_set = Set.Iic (-6) ∪ Set.Ioi (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l250_25041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_curved_sides_l250_25008

/-- A configuration of intersecting circles -/
structure CircleConfiguration where
  n : ℕ
  circles : Fin n → Circle
  h_n_ge_2 : n ≥ 2

/-- A curved side of the intersection figure -/
structure CurvedSide (config : CircleConfiguration) where
  circle : Fin config.n
  start_point : Point
  end_point : Point

/-- The intersection figure formed by the circles -/
noncomputable def IntersectionFigure (config : CircleConfiguration) : Set Point :=
  sorry

/-- The set of curved sides of the intersection figure -/
noncomputable def curvedSides (config : CircleConfiguration) : Finset (CurvedSide config) :=
  sorry

/-- The maximum number of curved sides is 2n - 2 -/
theorem max_curved_sides (config : CircleConfiguration) :
  (curvedSides config).card ≤ 2 * config.n - 2 ∧
  ∃ (config' : CircleConfiguration), config'.n = config.n ∧ (curvedSides config').card = 2 * config.n - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_curved_sides_l250_25008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_points_iff_a_in_range_l250_25017

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the condition for a
def valid_a (a : ℝ) : Prop := a > 0 ∧ a ≠ 1

-- Define the property of having two common points
def has_two_common_points (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = g a x ∧ f a y = g a y

-- State the theorem
theorem two_common_points_iff_a_in_range (a : ℝ) :
  valid_a a → (has_two_common_points a ↔ 1 < a ∧ a < Real.exp (1 / Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_common_points_iff_a_in_range_l250_25017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_typhoon_problem_l250_25049

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The problem statement -/
theorem ship_typhoon_problem (ship port typhoon : Point)
  (h1 : typhoon.x = ship.x - 70)  -- typhoon is 70 km west of ship
  (h2 : typhoon.y = ship.y)       -- typhoon is due west of ship
  (h3 : port.x = typhoon.x)       -- port is due north of typhoon
  (h4 : port.y = typhoon.y + 40)  -- port is 40 km north of typhoon
  (h5 : distance ship port = distance typhoon port) -- ship is on a straight line to port
  : distance ship typhoon > 30 := by
  sorry

#check ship_typhoon_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_typhoon_problem_l250_25049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l250_25057

/-- Represents the time it takes for a worker to complete the entire task alone -/
def WorkTime := ℝ

/-- Represents the total duration of the work -/
def TotalDuration := ℝ

/-- Represents the time when the second worker joins -/
def JoinTime := ℝ

theorem work_completion_time 
  (x_time y_time total_duration join_time : ℝ) : 
  x_time = 20 →
  y_time = 12 →
  total_duration = 10 →
  join_time * (1 / x_time) + (total_duration - join_time) * (1 / x_time + 1 / y_time) = 1 →
  join_time = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l250_25057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_theorem_l250_25082

-- Define complex number operations using the existing Complex type
open Complex

-- State the theorem
theorem complex_division_theorem : 
  (1 - I) / (1 + I) = -I :=
by
  -- Multiply numerator and denominator by the conjugate of the denominator
  have h1 : (1 - I) / (1 + I) = (1 - I) * (1 - I) / ((1 + I) * (1 - I)) := by
    sorry
  
  -- Expand the numerator and denominator
  have h2 : (1 - I) * (1 - I) / ((1 + I) * (1 - I)) = (1 - 2*I - 1) / (1^2 - I^2) := by
    sorry
  
  -- Simplify using I^2 = -1
  have h3 : (1 - 2*I - 1) / (1^2 - I^2) = (-2*I) / 2 := by
    sorry
  
  -- Simplify the fraction
  have h4 : (-2*I) / 2 = -I := by
    sorry
  
  -- Combine all steps
  calc
    (1 - I) / (1 + I) = (1 - I) * (1 - I) / ((1 + I) * (1 - I)) := h1
    _ = (1 - 2*I - 1) / (1^2 - I^2) := h2
    _ = (-2*I) / 2 := h3
    _ = -I := h4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_theorem_l250_25082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l250_25051

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x

-- State the theorem
theorem max_value_of_f (θ : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x ≤ f θ) →
  (4 * Real.tan θ = m) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l250_25051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_5_exists_l250_25002

noncomputable def c : ℕ → ℝ
  | 0 => 0
  | n + 1 => (7/4) * c n + (3/4) * Real.sqrt (9^n - (c n)^2)

theorem c_5_exists : ∃ x : ℝ, c 5 = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_5_exists_l250_25002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_combination_l250_25004

/-- Represents the denominations of coins available -/
inductive Coin
  | kopek_20 : Coin
  | kopek_15 : Coin
  | kopek_10 : Coin

/-- Represents the exchange rules for each coin -/
def exchange_rule : Coin → List ℕ
  | Coin.kopek_20 => [15, 2, 2, 1]
  | Coin.kopek_15 => [10, 2, 2, 1]
  | Coin.kopek_10 => [3, 3, 2, 2]

/-- The total amount to be exchanged in kopecks -/
def total_amount : ℕ := 125

/-- A function that checks if a given combination of coins can be exchanged for the total amount -/
def is_valid_combination (coins : List Coin) : Prop :=
  (coins.map exchange_rule).join.sum = total_amount

/-- The theorem stating that the only valid combination is one 15-kopek coin and eleven 10-kopek coins -/
theorem unique_combination : 
  ∀ (coins : List Coin), 
    is_valid_combination coins ↔ 
    coins = [Coin.kopek_15] ++ List.replicate 11 Coin.kopek_10 :=
by
  sorry

#check unique_combination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_combination_l250_25004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l250_25054

/-- Represents a point on the circle -/
structure Point where
  index : Fin 10

/-- Represents the state of grasshoppers on the circle -/
structure CircleState where
  positions : Fin 10 → Point

/-- Represents a jump of a grasshopper -/
def jump (state : CircleState) : CircleState := sorry

/-- The initial state of the grasshoppers -/
def initialState : CircleState := sorry

/-- Checks if a state is valid according to the rules -/
def isValidState (state : CircleState) : Prop := sorry

/-- Checks if a state is the final state described in the problem -/
def isFinalState (state : CircleState) : Prop := sorry

/-- Checks if two points are neighbors on the circle -/
def areNeighbors (p1 p2 : Point) : Prop := sorry

/-- Represents the symmetry of points with respect to the center -/
def areSymmetric (p1 p2 : Point) : Prop := sorry

theorem grasshopper_theorem (state : CircleState) :
  isValidState state →
  isFinalState state →
  ∃ (n : ℕ), state = (n.iterate jump initialState) →
  state.positions 9 = Point.mk 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_theorem_l250_25054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_vector_of_h_collinear_unit_vector_tan_2x0_range_l250_25091

-- Define companion function
noncomputable def companion_function (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

-- Define h(x)
noncomputable def h (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (x + Real.pi / 6) + 3 * Real.cos (Real.pi / 3 - x)

theorem companion_vector_of_h :
  ∃ (a b : ℝ), ∀ x, h x = companion_function a b x ∧ a = 3 ∧ b = Real.sqrt 3 := by
  sorry

theorem collinear_unit_vector :
  let v := (3, Real.sqrt 3)
  let u := (Real.sqrt 3 / 2, 1 / 2)
  ∃ (k : ℝ), k * v.1 = u.1 ∧ k * v.2 = u.2 ∧ u.1^2 + u.2^2 = 1 := by
  sorry

theorem tan_2x0_range (a b : ℝ) (h1 : 0 < b/a) (h2 : b/a ≤ Real.sqrt 3) :
  let f := companion_function a b
  let x0 := Real.arctan (a/b)
  Real.tan (2*x0) < 0 ∨ Real.tan (2*x0) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_vector_of_h_collinear_unit_vector_tan_2x0_range_l250_25091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l250_25031

theorem root_difference_quadratic : 
  ∃ (r₁ r₂ : ℝ), (r₁^2 + 44*r₁ + 352 = -16) ∧ 
                 (r₂^2 + 44*r₂ + 352 = -16) ∧ 
                 |r₁ - r₂| = 12 ∧ 
                 ∀ (s₁ s₂ : ℝ), (s₁^2 + 44*s₁ + 352 = -16) → 
                                (s₂^2 + 44*s₂ + 352 = -16) → 
                                |s₁ - s₂| ≤ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l250_25031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l250_25050

theorem tan_value_third_quadrant (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : α ∈ Set.Icc π (3*π/2)) :
  Real.tan α = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_third_quadrant_l250_25050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l250_25093

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- State the theorem
theorem tangent_line_and_range_of_a :
  (∀ x : ℝ, x > 0 → 
    (∃ a : ℝ, a > 0 ∧ f a x > 0)) ↔
  (∃ tangent_line : ℝ → ℝ → Prop, 
    (tangent_line = λ x y ↦ x + y - 2 = 0) ∧
    (∀ a : ℝ, a > 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l250_25093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l250_25056

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem ellipse_properties (e : Ellipse) 
  (hA : on_ellipse e 1 (3/2))
  (hF : ∃ x₁ x₂ : ℝ, 
    distance 1 (3/2) x₁ 0 + distance 1 (3/2) x₂ 0 = 4 ∧
    on_ellipse e x₁ 0 ∧ on_ellipse e x₂ 0) :
  (∀ x y : ℝ, on_ellipse e x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1 ∧ 
    on_ellipse e x₁ 0 ∧ on_ellipse e x₂ 0) ∧
  (∀ x₁ y₁ : ℝ, on_ellipse e x₁ y₁ → 
    ∀ x y : ℝ, (x + 1/2)^2 + 4*y^2/3 = 1 ↔ 
    x = (x₁ - 1)/2 ∧ y = y₁/2) ∧
  (∀ m n x y : ℝ, on_ellipse e m n → on_ellipse e (-m) (-n) → on_ellipse e x y →
    (y - n)/(x - m) * (y + n)/(x + m) = -e.b^2/e.a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l250_25056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_l250_25087

/-- The final value of a baseball card after three years of depreciation -/
theorem baseball_card_value (X : ℝ) (X_pos : X > 0) :
  X * (1 - 0.40) * (1 - 0.10) * (1 - 0.20) = X * 0.432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_l250_25087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_from_complex_equation_l250_25079

/-- The minimum area of a triangle formed by three distinct solutions of (z-4)^6 = 64 -/
theorem min_area_triangle_from_complex_equation :
  ∃ (area : ℝ), area ≥ 3/4 ∧
  ∀ (z D E F : ℂ),
  (z - 4) ^ 6 = 64 →
  (D - 4) ^ 6 = 64 →
  (E - 4) ^ 6 = 64 →
  (F - 4) ^ 6 = 64 →
  D ≠ E ∧ E ≠ F ∧ D ≠ F →
  area ≤ abs ((E - D).im * (F - D).re - (E - D).re * (F - D).im) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_from_complex_equation_l250_25079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_iterated_f_l250_25098

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the composition of f with itself n times
def f_n (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem roots_of_iterated_f :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f_n 2013 x = 1/2) ∧ (Finset.card S = 4030) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_iterated_f_l250_25098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_properties_l250_25078

noncomputable section

variable (b : ℝ)

def f (a b x : ℝ) : ℝ := a * x - b * (x + 1) * Real.log (x + 1) + 1

def tangent_line (b x y : ℝ) : Prop := x - y + b = 0

theorem function_and_tangent_line_properties :
  (∃ a b : ℝ, (∀ x y : ℝ, f a b 0 = y → tangent_line b x y) ∧
   a = 2 ∧ b = 1) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 0 → f 2 1 x ≥ k * x^2 + x + 1) → k ≤ -1/2) ∧
  (∀ x : ℝ, x ≥ 0 → f 2 1 x ≥ -1/2 * x^2 + x + 1) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_tangent_line_properties_l250_25078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_obtuse_l250_25053

noncomputable def f (x : ℝ) := Real.exp x * Real.sin x

theorem tangent_inclination_obtuse :
  let slope := (Real.exp 4) * (Real.sin 4 + Real.cos 4)
  slope < 0 ∧ 
  ∀ θ, θ = Real.arctan slope → π/2 < θ ∧ θ < π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_obtuse_l250_25053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brocard_and_steiner_coordinates_l250_25070

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- Trilinear coordinates -/
structure Trilinear where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vertices of the Brocard triangle -/
noncomputable def brocardTriangleVertices (t : Triangle) : Trilinear :=
  { x := t.a ^ 2
    y := t.c ^ 2
    z := t.b ^ 2 }

/-- Steiner point -/
noncomputable def steinerPoint (t : Triangle) : Trilinear :=
  { x := 1 / (t.a * (t.b ^ 2 - t.c ^ 2))
    y := 1 / (t.b * (t.c ^ 2 - t.a ^ 2))
    z := 1 / (t.c * (t.a ^ 2 - t.b ^ 2)) }

theorem brocard_and_steiner_coordinates (t : Triangle) :
  (brocardTriangleVertices t = { x := t.a ^ 2, y := t.c ^ 2, z := t.b ^ 2 }) ∧
  (steinerPoint t = { x := 1 / (t.a * (t.b ^ 2 - t.c ^ 2)),
                      y := 1 / (t.b * (t.c ^ 2 - t.a ^ 2)),
                      z := 1 / (t.c * (t.a ^ 2 - t.b ^ 2)) }) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brocard_and_steiner_coordinates_l250_25070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_path_count_l250_25028

/-- Represents the pyramid-shaped grid of letters --/
structure Grid :=
  (letters : List (List Char))

/-- Represents a valid move in the grid --/
inductive Move
  | Vertical
  | Diagonal

/-- A path in the grid is a list of moves --/
def ContestPath := List Move

/-- Checks if a path forms the word "CONTEST" in the given grid --/
def formsContest (g : Grid) (p : ContestPath) : Prop :=
  sorry -- Implementation details omitted

/-- The main theorem stating that there is exactly one path forming "CONTEST" --/
theorem contest_path_count (g : Grid) : 
  (∃! p : ContestPath, formsContest g p ∧ p.head? = some Move.Vertical) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_path_count_l250_25028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_product_l250_25044

theorem max_cos_product (α β γ : ℝ) (h_acute : α ∈ Set.Ioo 0 (π/2) ∧ β ∈ Set.Ioo 0 (π/2) ∧ γ ∈ Set.Ioo 0 (π/2)) 
  (h_sin_sum : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  Real.cos α * Real.cos β * Real.cos γ ≤ 2 * Real.sqrt 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_product_l250_25044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_for_g_eq_6_l250_25074

noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 15 * x + 25 else 3 * x - 9

theorem sum_of_x_values_for_g_eq_6 :
  ∃ (x₁ x₂ : ℝ), g x₁ = 6 ∧ g x₂ = 6 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 56 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_for_g_eq_6_l250_25074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l250_25026

/-- Calculates the distance travelled downstream by a boat given its speed in still water,
    the current speed, and the time of travel. -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_speed : ℝ) (time_minutes : ℝ) : ℝ :=
  (boat_speed + current_speed) * (time_minutes / 60)

/-- Theorem stating that a boat with a speed of 30 km/hr in still water,
    travelling in a current of 7 km/hr for 36 minutes, will cover 22.2 km downstream. -/
theorem boat_downstream_distance :
  distance_downstream 30 7 36 = 22.2 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l250_25026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_cube_dot_path_length_l250_25092

/-- Predicate to represent a valid rolling cube path (this would need to be defined based on the problem constraints) -/
def is_valid_rolling_cube_path (cube_edge : ℝ) (path_length : ℝ) : Prop :=
  ∃ (num_rotations : ℕ),
    path_length = 
      (2 * num_rotations * Real.sqrt 5 * Real.pi / 4) + 
      (2 * num_rotations * Real.pi / 4)

/-- Predicate to represent that a given length is the path followed by the dot on a rolling cube -/
def is_path_length_of_rolling_cube_dot (cube_edge : ℝ) (path_length : ℝ) : Prop :=
  cube_edge > 0 ∧
  path_length = (Real.sqrt 5 + 1) * Real.pi ∧
  is_valid_rolling_cube_path cube_edge path_length

/-- The length of the path followed by a dot on a rolling cube -/
theorem rolling_cube_dot_path_length :
  ∀ (cube_edge : ℝ) (dot_path : ℝ),
    cube_edge = 2 →
    dot_path = (Real.sqrt 5 + 1) * Real.pi →
    is_path_length_of_rolling_cube_dot cube_edge dot_path :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rolling_cube_dot_path_length_l250_25092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_fin_area_proof_l250_25033

/-- The area of the "shark's fin falcata" region -/
noncomputable def sharks_fin_area : ℝ := 2 * Real.pi

/-- The radius of the larger circle -/
def large_radius : ℝ := 4

/-- The radius of the smaller circle -/
def small_radius : ℝ := 2

/-- The center of the smaller circle -/
def small_circle_center : ℝ × ℝ := (0, 2)

/-- The area of a quarter circle with radius r -/
noncomputable def quarter_circle_area (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := (1/2) * Real.pi * r^2

theorem sharks_fin_area_proof :
  sharks_fin_area = quarter_circle_area large_radius - semicircle_area small_radius :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_fin_area_proof_l250_25033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l250_25047

/-- A structure representing a rational function with specific properties -/
structure RationalFunction where
  m : ℝ → ℝ
  n : ℝ → ℝ
  is_quadratic_m : ∃ a b c : ℝ, ∀ x, m x = a * x^2 + b * x + c
  is_quadratic_n : ∃ a b c : ℝ, ∀ x, n x = a * x^2 + b * x + c
  horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x > M, |m x / n x + 3| < ε
  vertical_asymptote : n 3 = 0
  hole : m (-4) = 0 ∧ n (-4) = 0

/-- Theorem stating that for a rational function satisfying the given conditions, m(-1)/n(-1) = 9/4 -/
theorem rational_function_value (f : RationalFunction) : f.m (-1) / f.n (-1) = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l250_25047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_with_large_s_l250_25036

-- Define s(k) as the number of ways to express k as the sum of distinct 2012th powers
def s (k : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_n_with_large_s (c : ℝ) : ∃ n : ℕ, (s n : ℝ) > c * (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_with_large_s_l250_25036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l250_25001

/-- Represents a rectangular garden with flower beds and a playground -/
structure Garden where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- Calculates the area of an isosceles right triangle -/
noncomputable def isoscelesRightTriangleArea (leg : ℝ) : ℝ := (1/2) * leg^2

/-- Calculates the area of the flower beds -/
noncomputable def flowerBedsArea (g : Garden) : ℝ :=
  2 * isoscelesRightTriangleArea ((g.trapezoid_long_side - g.trapezoid_short_side) / 2)

/-- Calculates the total area of the garden -/
noncomputable def gardenArea (g : Garden) : ℝ := g.length * g.width

/-- Theorem: The fraction of the garden occupied by flower beds is 4/23 -/
theorem flower_beds_fraction (g : Garden) 
    (h1 : g.trapezoid_short_side = 30)
    (h2 : g.trapezoid_long_side = 46)
    (h3 : g.width = (g.trapezoid_long_side - g.trapezoid_short_side) / 2)
    (h4 : g.length = g.trapezoid_long_side) :
    flowerBedsArea g / gardenArea g = 4 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l250_25001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_donation_theorem_l250_25080

def cloth_donation (cloth1_area cloth2_area cloth3_area : ℝ)
  (cloth1_cuts cloth2_cuts cloth3_cuts : List ℝ) : ℝ :=
  let donated1 := cloth1_area * (cloth1_cuts[1]! + cloth1_cuts[2]! + cloth1_cuts[3]!)
  let donated2 := cloth2_area * (cloth2_cuts[1]! + cloth2_cuts[2]!)
  let donated3 := cloth3_area * (cloth3_cuts[1]! + cloth3_cuts[2]! + cloth3_cuts[3]!)
  donated1 + donated2 + donated3

theorem cloth_donation_theorem :
  cloth_donation 100 65 48
    [0.375, 0.255, 0.22, 0.15]
    [0.487, 0.323, 0.19]
    [0.295, 0.272, 0.238, 0.195] = 129.685 := by
  sorry

#eval cloth_donation 100 65 48
    [0.375, 0.255, 0.22, 0.15]
    [0.487, 0.323, 0.19]
    [0.295, 0.272, 0.238, 0.195]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_donation_theorem_l250_25080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_problem_l250_25069

noncomputable section

open Real

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

def law_of_sines (a b c : ℝ) (A B C : ℝ) : Prop :=
  a / sin A = b / sin B ∧
  b / sin B = c / sin C

def area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * a * c * sin B

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_law_of_sines : law_of_sines a b c A B C)
  (h_condition : a * sin (2*B) = b * sin A)
  (h_b : b = 3 * sqrt 2)
  (h_area : area a b c A B C = (3 * sqrt 3) / 2) :
  B = Real.pi/3 ∧ a + b + c = 6 + 3 * sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_problem_l250_25069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_interior_point_and_locus_l250_25010

/-- A sequence of points in a plane --/
def PointSequence := ℕ → EuclideanSpace ℝ (Fin 2)

/-- Predicate to check if a point is interior to a triangle --/
def IsInterior (p : EuclideanSpace ℝ (Fin 2)) (a b c : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Predicate to check if a point is on an arc --/
def IsOnArc (p : EuclideanSpace ℝ (Fin 2)) (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) (startAngle endAngle : ℝ) : Prop := sorry

/-- Predicate to check if three points form a right angle --/
def IsRightAngle (a b c : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Predicate to check if four points form a perpendicular --/
def IsPerpendicular (a b c d : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Function to calculate the midpoint of two points --/
def MidPoint (a b : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

/-- Function to calculate the distance between two points --/
def Distance (a b : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- The theorem statement --/
theorem unique_interior_point_and_locus 
  (A : PointSequence) 
  (right_angle : IsRightAngle (A 1) (A 2) (A 3)) 
  (perpendicular : ∀ n ≥ 3, IsPerpendicular (A n) (A (n+1)) (A (n-2)) (A (n-1))) :
  ∃! P : EuclideanSpace ℝ (Fin 2), 
    (∀ n ≥ 3, IsInterior P (A (n-2)) (A (n-1)) (A n)) ∧ 
    IsOnArc P (MidPoint (A 1) (A 3)) (Distance (A 1) (A 3) / 2) 0 (2 * Real.arctan (1/2)) ∧
    P ≠ A 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_interior_point_and_locus_l250_25010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_final_collection_l250_25012

/-- Represents a stamp collection --/
structure StampCollection where
  nature : ℕ
  architecture : ℕ
  animals : ℕ
  vehicles : ℕ
  famous_people : ℕ

/-- Calculates the total number of stamps in a collection --/
def total_stamps (sc : StampCollection) : ℕ :=
  sc.nature + sc.architecture + sc.animals + sc.vehicles + sc.famous_people

/-- Represents a stamp transaction between two collectors --/
structure StampTransaction where
  nature : ℤ
  architecture : ℤ
  animals : ℤ
  vehicles : ℤ
  famous_people : ℤ

/-- Applies a transaction to a stamp collection --/
def apply_transaction (sc : StampCollection) (st : StampTransaction) : StampCollection where
  nature := (sc.nature : ℤ) + st.nature |> Int.toNat
  architecture := (sc.architecture : ℤ) + st.architecture |> Int.toNat
  animals := (sc.animals : ℤ) + st.animals |> Int.toNat
  vehicles := (sc.vehicles : ℤ) + st.vehicles |> Int.toNat
  famous_people := (sc.famous_people : ℤ) + st.famous_people |> Int.toNat

theorem anna_final_collection
  (initial_anna : StampCollection)
  (transaction1 : StampTransaction)
  (transaction2 : StampTransaction)
  (transaction3 : StampTransaction)
  (transaction4 : StampTransaction)
  (transaction5 : StampTransaction)
  (transaction6 : StampTransaction)
  (h_initial : initial_anna = { nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4 })
  (h_t1 : transaction1 = { nature := 4, architecture := 5, animals := 5, vehicles := 2, famous_people := 1 })
  (h_t2 : transaction2 = { nature := 2, architecture := 0, animals := -1, vehicles := 0, famous_people := 0 })
  (h_t3 : transaction3 = { nature := 0, architecture := 3, animals := -5, vehicles := 0, famous_people := 0 })
  (h_t4 : transaction4 = { nature := 7, architecture := 0, animals := -4, vehicles := 0, famous_people := 0 })
  (h_t5 : transaction5 = { nature := 5, architecture := 0, animals := 0, vehicles := -2, famous_people := 0 })
  (h_t6 : transaction6 = { nature := 0, architecture := 0, animals := 0, vehicles := 3, famous_people := -3 }) :
  let final_anna := apply_transaction (apply_transaction (apply_transaction (apply_transaction (apply_transaction (apply_transaction initial_anna transaction1) transaction2) transaction3) transaction4) transaction5) transaction6
  total_stamps final_anna = 69 ∧
  final_anna.nature = 28 ∧
  final_anna.architecture = 23 ∧
  final_anna.animals = 7 ∧
  final_anna.vehicles = 9 ∧
  final_anna.famous_people = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anna_final_collection_l250_25012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacterial_growth_ratio_approx_l250_25090

/-- The ratio of bacterial growth between two time periods -/
noncomputable def bacterial_growth_ratio (initial : ℝ) (middle : ℝ) (final : ℝ) : ℝ :=
  let first_increase := middle - initial
  let second_increase := final - middle
  first_increase / second_increase

theorem bacterial_growth_ratio_approx :
  ∀ (ε : ℝ),
  ε > 0 →
  ∃ (initial middle final : ℝ),
  initial > 0 ∧ middle > initial ∧ final > middle ∧
  abs (initial - 10.0) < ε ∧ abs (middle - 17.0) < ε ∧ abs (final - 28.9) < ε ∧
  abs (bacterial_growth_ratio initial middle final - (1 / 1.7)) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacterial_growth_ratio_approx_l250_25090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_triangle_l250_25067

/-- A triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17
  right_angle : a^2 + b^2 = c^2

/-- The radius of the circumcircle of a right triangle -/
noncomputable def circumradius (t : RightTriangle) : ℝ := t.c / 2

/-- Theorem: The radius of the circle passing through the vertices of the given triangle is 8.5 -/
theorem circumradius_of_triangle : 
  ∀ (t : RightTriangle), circumradius t = 8.5 := by
  intro t
  unfold circumradius
  rw [t.hc]
  norm_num

#check circumradius_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_triangle_l250_25067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l250_25011

/-- The simple interest formula -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem interest_rate_problem (principal interest time : ℝ) 
  (h_principal : principal = 4000)
  (h_interest : interest = 320)
  (h_time : time = 2)
  (h_simple_interest : simple_interest principal (4 : ℝ) time = interest) :
  (4 : ℝ) = (interest * 100) / (principal * time) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l250_25011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_purity_Y_Z_l250_25084

/-- Represents a mineral deposit with a certain amount and purity -/
structure Deposit where
  amount : ℝ
  purity : ℝ

/-- Calculate the average purity of two combined deposits -/
noncomputable def averagePurity (d1 d2 : Deposit) : ℝ :=
  (d1.amount * d1.purity + d2.amount * d2.purity) / (d1.amount + d2.amount)

theorem max_purity_Y_Z (X Y Z : Deposit)
  (h1 : X.purity = 0.3)
  (h2 : Y.purity = 0.6)
  (h3 : averagePurity X Y = 0.5)
  (h4 : averagePurity X Z = 0.45) :
  averagePurity Y Z ≤ 0.65 := by
  sorry

#check max_purity_Y_Z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_purity_Y_Z_l250_25084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_tip_fraction_l250_25099

/-- Represents the income structure of a waiter -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income that comes from tips -/
def tipFraction (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

theorem waiter_tip_fraction :
  ∀ (income : WaiterIncome),
  income.tips = (5 : ℚ) / 2 * income.salary →
  tipFraction income = 5 / 7 := by
  intro income h
  simp [tipFraction]
  -- The rest of the proof would go here
  sorry

#eval tipFraction { salary := 2, tips := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waiter_tip_fraction_l250_25099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_equation_l250_25095

/-- A circle C with radius 1, whose center lies on the line 3x - y = 0 and is tangent to the line 4x - 3y = 0 -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : 3 * center.1 - center.2 = 0
  radius_is_one : radius = 1
  tangent_to_line : |4 * center.1 - 3 * center.2| / Real.sqrt (16 + 9) = radius

/-- The standard equation of circle C is either (x - 1)² + (y - 3)² = 1 or (x + 1)² + (y + 3)² = 1 -/
def standard_equation (c : CircleC) : Prop :=
  ∀ x y : ℝ, ((x - 1)^2 + (y - 3)^2 = 1 ∨ (x + 1)^2 + (y + 3)^2 = 1) ↔
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_c_equation (c : CircleC) : standard_equation c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_c_equation_l250_25095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_3_sqrt_17_l250_25083

/-- The straight-line distance traveled after walking 5 miles east and then 8 miles at a 45-degree angle northward -/
noncomputable def distance_traveled : ℝ :=
  3 * Real.sqrt 17

/-- Theorem stating that the distance traveled is equal to 3√17 miles -/
theorem distance_is_3_sqrt_17 (east_distance : ℝ) (angle_turn : ℝ) (angled_distance : ℝ)
  (h1 : east_distance = 5)
  (h2 : angle_turn = 45)
  (h3 : angled_distance = 8) :
  Real.sqrt ((east_distance + angled_distance / Real.sqrt 2) ^ 2 + (angled_distance / Real.sqrt 2) ^ 2) = distance_traveled :=
by sorry

#check distance_is_3_sqrt_17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_3_sqrt_17_l250_25083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l250_25061

/-- Represents the growth rate in the second and third quarters -/
def x : ℝ := sorry

/-- Represents the production in the first quarter (in tens of thousands) -/
def first_quarter : ℝ := 10

/-- Represents the total production over three quarters (in tens of thousands) -/
def total_production : ℝ := 36.4

/-- Theorem stating that the equation correctly represents the total production -/
theorem production_equation : 
  first_quarter + first_quarter * (1 + x) + first_quarter * (1 + x)^2 = total_production := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l250_25061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l250_25006

/-- Represents the train journey from New York to Chicago -/
structure TrainJourney where
  distance : ℝ
  departureTimeNY : ℝ
  arrivalTimeChicago : ℝ
  timeDifference : ℝ

/-- Calculates the average speed of the train -/
noncomputable def averageSpeed (journey : TrainJourney) : ℝ :=
  journey.distance / (journey.arrivalTimeChicago - (journey.departureTimeNY - journey.timeDifference))

/-- Theorem stating that the average speed of the train is 60 mph -/
theorem train_average_speed :
  let journey : TrainJourney := {
    distance := 480,
    departureTimeNY := 10,
    arrivalTimeChicago := 17,
    timeDifference := 1
  }
  averageSpeed journey = 60 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l250_25006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_eccentricity_l250_25042

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

-- Define the two hyperbolas
def hyperbola1 (m : ℝ) (x y : ℝ) : Prop := x^3 / m - y^2 / 3 = 1
def hyperbola2 (x y : ℝ) : Prop := x^3 / 8 - y^2 / 4 = 1

-- State the theorem
theorem hyperbolas_same_eccentricity (m : ℝ) :
  (∃ e : ℝ, eccentricity (Real.sqrt m) (Real.sqrt 3) = e ∧
            eccentricity 2 (2 * Real.sqrt 2) = e) →
  m = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_eccentricity_l250_25042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_tangency_l250_25035

-- Define the circle C in polar coordinates
noncomputable def circle_C (a : ℝ) (θ : ℝ) : ℝ := 2 * a * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (3 * t + 1, 4 * t + 3)

-- State the theorem
theorem circle_line_tangency (a : ℝ) :
  a ≠ 0 ∧ a < 1 →
  (∃! p : ℝ × ℝ, 
    (∃ θ : ℝ, (circle_C a θ * Real.cos θ, circle_C a θ * Real.sin θ) = p) ∧
    (∃ t : ℝ, line_l t = p)) →
  4 * a + 5 = 5 * a ∨ 4 * a + 5 = -5 * a :=
by
  sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_tangency_l250_25035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_max_at_14_l250_25030

/-- The sequence a_n defined as n / (n^2 + 196) -/
noncomputable def a_n (n : ℝ) : ℝ := n / (n^2 + 196)

/-- Theorem stating that the sequence a_n reaches its maximum when n = 14 -/
theorem a_n_max_at_14 :
  ∀ n : ℝ, n > 0 → a_n n ≤ a_n 14 := by
  sorry

#check a_n_max_at_14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_max_at_14_l250_25030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_l250_25040

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L that contains point P
noncomputable def line_L (x y : ℝ) : Prop := x + 2*y - 9 = 0

-- Define the fixed point F
noncomputable def fixed_point : ℝ × ℝ := (4/9, 8/9)

-- State the theorem
theorem tangent_line_passes_through_fixed_point :
  ∀ (P A B : ℝ × ℝ),
  line_L P.1 P.2 →
  circle_C A.1 A.2 →
  circle_C B.1 B.2 →
  (∃ (t : ℝ), A = (t * P.1, t * P.2)) →
  (∃ (s : ℝ), B = (s * P.1, s * P.2)) →
  ∃ (k : ℝ),
    fixed_point.1 = k * A.1 + (1 - k) * B.1 ∧
    fixed_point.2 = k * A.2 + (1 - k) * B.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_fixed_point_l250_25040
