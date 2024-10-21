import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_product_equals_140_l571_57166

/-- Round a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- The original expression -/
def original_expression : ℝ := 2.5 * (56.2 + 0.15)

/-- The rounded result -/
noncomputable def rounded_result : ℤ := round_to_nearest 2.5 * round_to_nearest (56.2 + 0.15)

theorem rounded_product_equals_140 : rounded_result = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounded_product_equals_140_l571_57166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_l571_57126

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + m) * Real.exp x

-- Theorem for part 1
theorem monotonicity_intervals (x : ℝ) :
  let f₁ := f 1
  (∀ x < -2, (deriv f₁) x > 0) ∧
  (∀ x ∈ Set.Ioo (-2) (-1), (deriv f₁) x < 0) ∧
  (∀ x > -1, (deriv f₁) x > 0) := by
  sorry

-- Theorem for part 2
theorem max_value_condition (m : ℝ) :
  m < 2 →
  (∃ x, f m x = 10 * Real.exp (-2)) →
  (∀ y, f m y ≤ 10 * Real.exp (-2)) →
  m = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_l571_57126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_distance_l571_57109

/-- The acceleration due to gravity in mm/s² -/
noncomputable def g : ℝ := 9800

/-- The height of the cliff in mm -/
noncomputable def h : ℝ := 300000

/-- The initial distance fallen by the first particle in mm -/
noncomputable def s₁ : ℝ := 1/1000

/-- The time it takes for the first particle to reach the base of the cliff -/
noncomputable def t : ℝ := Real.sqrt (2 * h / g)

/-- The time delay before the second particle starts falling -/
noncomputable def t₁ : ℝ := Real.sqrt (2 * s₁ / g)

/-- The distance between the two particles when the first reaches the base -/
noncomputable def d : ℝ := 2 * Real.sqrt (h * s₁) - s₁

theorem particle_distance : ‖d - 34.6‖ < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_distance_l571_57109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_l571_57152

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci
variable (F₁ F₂ : ℝ × ℝ)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- Define the distance function
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem hyperbola_distance (h : hyperbola P.1 P.2) (d : distance P F₁ = 3) :
  distance P F₂ = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_distance_l571_57152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_binomial_properties_l571_57140

/-- The negative binomial distribution -/
def negative_binomial_dist (p : ℝ) (r k : ℕ) : ℝ :=
  (Nat.choose (k - 1) (r - 1) : ℝ) * p^r * (1 - p)^(k - r)

/-- The expected value of the negative binomial distribution -/
noncomputable def negative_binomial_expectation (p : ℝ) (r : ℕ) : ℝ := r / p

theorem negative_binomial_properties (p : ℝ) (r k : ℕ) 
    (hp : 0 < p ∧ p < 1) (hr : r > 0) (hk : k ≥ r) :
  negative_binomial_dist p r k = (Nat.choose (k - 1) (r - 1) : ℝ) * p^r * (1 - p)^(k - r) ∧
  negative_binomial_expectation p r = r / p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_binomial_properties_l571_57140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l571_57195

-- Define the equations
def equation_A (x : ℝ) : Prop := x^2 + 1/x^2 = 0
def equation_B (x : ℝ) : Prop := x^2 - 2*x = x^2 + 1
def equation_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def equation_D (x y : ℝ) : Prop := 3*x - 2*x*y - 5*y = 0

-- Define what it means for an equation to be quadratic in x
def is_quadratic_in_x (f : ℝ → Prop) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x ↔ a*x^2 + b*x + c = 0

-- Theorem statement
theorem quadratic_equation_identification :
  is_quadratic_in_x equation_C ∧
  ¬is_quadratic_in_x equation_A ∧
  ¬is_quadratic_in_x equation_B ∧
  ¬is_quadratic_in_x (λ x ↦ ∃ y, equation_D x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l571_57195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_yield_calculation_l571_57168

/-- Represents a chemical species in a reaction -/
structure Species where
  name : String
  moles : ℝ

/-- Represents a balanced chemical equation -/
structure BalancedEquation where
  reactants : List Species
  products : List Species

/-- Calculates the limiting reactant based on the balanced equation and initial moles -/
noncomputable def limitingReactant (equation : BalancedEquation) (initialMoles : List Species) : Species :=
  sorry

/-- Calculates the theoretical yield of a product based on the limiting reactant -/
noncomputable def theoreticalYield (equation : BalancedEquation) (limitingReactant : Species) (product : Species) : ℝ :=
  sorry

/-- Calculates the actual yield based on the theoretical yield and yield percentage -/
def actualYield (theoreticalYield : ℝ) (yieldPercentage : ℝ) : ℝ :=
  theoreticalYield * yieldPercentage

theorem water_yield_calculation (naHSO3 : Species) (hCl : Species) (h2O : Species) (equation : BalancedEquation) 
    (h_naHSO3_moles : naHSO3.moles = 3)
    (h_hCl_moles : hCl.moles = 4)
    (h_balanced : equation = BalancedEquation.mk [naHSO3, hCl] [h2O, Species.mk "Na2SO3" 0, hCl])
    (h_yield_percentage : ℝ) 
    (h_yield_percentage_value : h_yield_percentage = 0.8) :
  actualYield (theoreticalYield equation (limitingReactant equation [naHSO3, hCl]) h2O) h_yield_percentage = 2.4 := by
  sorry

#eval 3 * 0.8  -- This should output 2.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_yield_calculation_l571_57168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_l571_57182

/-- The sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) : ℝ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series with first term 10, last term 30, 
    and common difference 0.5 is equal to 820 -/
theorem arithmetic_series_sum : 
  arithmetic_sum 10 30 0.5 = 820 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_series_sum_l571_57182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_P_or_Q_l571_57132

/-- The function f parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * m * x + m + 4/3

/-- Proposition P -/
def P (m : ℝ) : Prop := -3 ≤ m - 5 ∧ m - 5 ≤ 3

/-- Proposition Q -/
def Q (m : ℝ) : Prop := ∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0

/-- The main theorem -/
theorem range_of_m_for_P_or_Q (m : ℝ) : 
  (P m ∨ Q m) ↔ (m ≥ 2 ∨ m < -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_P_or_Q_l571_57132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l571_57190

-- Define the angle θ
variable (θ : Real)

-- Define the point on the terminal side of the angle
def terminal_point : ℝ × ℝ := (4, -3)

-- Define the distance from the point to the origin
noncomputable def distance_to_origin : ℝ := Real.sqrt (terminal_point.1^2 + terminal_point.2^2)

-- Theorem statement
theorem cos_theta_value : Real.cos θ = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l571_57190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l571_57158

open Real MeasureTheory

-- Define the parametric curve
noncomputable def x (t : ℝ) : ℝ := 2 * (cos t) ^ 3
noncomputable def y (t : ℝ) : ℝ := 2 * (sin t) ^ 3

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

-- Theorem statement
theorem curve_arc_length :
  arcLength 0 (π / 4) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_arc_length_l571_57158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_house_library_park_l571_57145

noncomputable def run_time_to_park : ℝ := 30
noncomputable def distance_to_park : ℝ := 4
noncomputable def library_position : ℝ := distance_to_park / 2

theorem run_time_house_library_park (constant_speed : ℝ → ℝ → ℝ) :
  constant_speed library_position (run_time_to_park / 2) +
  constant_speed (distance_to_park - library_position) (run_time_to_park / 2) =
  constant_speed distance_to_park run_time_to_park :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_time_house_library_park_l571_57145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l571_57102

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log (Real.sqrt (x^2 + 1) + x)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  f ((a + 1) / (a - 1)) - Real.log (Real.sqrt 2 - 1) < -1 → 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l571_57102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l571_57116

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (max min : ℝ),
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 → f x ≤ max) ∧
    (∀ (x : ℝ), -π/6 ≤ x ∧ x ≤ -π/12 → min ≤ f x) ∧
    (∃ (x₁ x₂ : ℝ), -π/6 ≤ x₁ ∧ x₁ ≤ -π/12 ∧ -π/6 ≤ x₂ ∧ x₂ ≤ -π/12 ∧
      f x₁ = max ∧ f x₂ = min) ∧
    max + min = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l571_57116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_composition_l571_57127

open Function

def g : Fin 5 → Fin 6 :=
  fun x => match x with
  | 0 => 3  -- corresponds to g(1) = 4
  | 1 => 1  -- corresponds to g(2) = 2
  | 2 => 5  -- corresponds to g(3) = 6
  | 3 => 4  -- corresponds to g(4) = 5
  | 4 => 2  -- corresponds to g(5) = 3

-- Assumption that g is bijective (which implies g^(-1) exists)
axiom g_bijective : Bijective g

-- Define g^(-1) using the fact that g is bijective
noncomputable def g_inv : Fin 6 → Fin 5 := invFun g

theorem g_inv_composition :
  g_inv (g_inv (g_inv (g_inv 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inv_composition_l571_57127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_MNP_l571_57148

-- Define the constants
noncomputable def M : ℝ := 0.3^5
noncomputable def N : ℝ := Real.log 5 / Real.log 0.3
noncomputable def P : ℝ := Real.log 5 / Real.log 3

-- State the theorem
theorem relationship_MNP : N < M ∧ M < P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_MNP_l571_57148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l571_57169

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

theorem f_min_max :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 →
  (∀ y : ℝ, 0 ≤ y → y ≤ 2 → f y ≥ 1/2) ∧
  (∃ z : ℝ, 0 ≤ z ∧ z ≤ 2 ∧ f z = 1/2) ∧
  (∀ y : ℝ, 0 ≤ y → y ≤ 2 → f y ≤ 5/2) ∧
  (∃ w : ℝ, 0 ≤ w ∧ w ≤ 2 ∧ f w = 5/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l571_57169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l571_57100

/-- The circumference of the base of a right circular cone formed from a circular sector --/
theorem cone_base_circumference (r θ : ℝ) (h_r : r = 5) (h_θ : θ = 300 * π / 180) :
  (2 * π - θ) / (2 * π) * (2 * π * r) = 25 / 3 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l571_57100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_l571_57121

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := -1 / x
noncomputable def curve2 (x : ℝ) : ℝ := Real.log x

-- Define the function f(x) = ln x - 1/2 - 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 1/2 - 1/x

-- Theorem statement
theorem unique_common_tangent :
  ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

#check unique_common_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_l571_57121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l571_57187

/-- The function f(x) = (3x^2 - 4x - 8) / (2x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 4 * x - 8) / (2 * x + 3)

/-- The proposed oblique asymptote y = (3/2)x - 5 -/
noncomputable def asymptote (x : ℝ) : ℝ := (3/2) * x - 5

/-- Theorem: The oblique asymptote of f(x) is y = (3/2)x - 5 -/
theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - asymptote x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l571_57187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_l571_57163

/-- Given a quadratic function f(x) = ax² + bx where a > 0 and b > 0,
    and its tangent line at x = 1 has a slope of 1,
    the minimum value of (8a + b) / (ab) is 18. -/
theorem quadratic_function_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let f := fun x : ℝ => a * x^2 + b * x
  let f' := fun x : ℝ => 2 * a * x + b
  (f' 1 = 1) → 
  (∀ x : ℝ, (8 * a + b) / (a * b) ≥ 18) ∧ 
  (∃ x : ℝ, (8 * a + b) / (a * b) = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_minimum_l571_57163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l571_57197

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3) + 1

theorem symmetry_about_line (x : ℝ) :
  f (-Real.pi/12 + x) = f (-Real.pi/12 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l571_57197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_one_l571_57164

theorem least_number_with_remainder_one (n : ℕ) : n = 2521 ↔ 
  (n > 100) ∧ 
  (∀ d ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 100 → (∀ d ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), m % d = 1) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_one_l571_57164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_value_l571_57115

theorem cubic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) →
  (∀ n : ℕ, n ∈ ({1, 2, 3, 4} : Set ℕ) → p (n : ℝ) = 1 / (n : ℝ)^2) →
  p 5 = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_value_l571_57115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l571_57165

theorem power_equation_solution (n b : ℝ) (h1 : n = 2^(0.15 : ℝ)) (h2 : n^b = 32) : b = 5 / 0.15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l571_57165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l571_57171

/-- The time taken for a train to pass a pole -/
noncomputable def time_to_pass_pole (train_length : ℝ) (platform_length : ℝ) (time_to_pass_platform : ℝ) : ℝ :=
  train_length / (train_length + platform_length) * time_to_pass_platform

/-- Theorem: The time taken for the train to pass the pole is 20 seconds -/
theorem train_passing_pole_time :
  let train_length : ℝ := 100
  let platform_length : ℝ := 100
  let time_to_pass_platform : ℝ := 40
  time_to_pass_pole train_length platform_length time_to_pass_platform = 20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l571_57171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_weight_system_l571_57135

/-- Represents the weight of a sparrow in liang -/
def x : ℝ := sorry

/-- Represents the weight of a swallow in liang -/
def y : ℝ := sorry

/-- The total weight of 5 sparrows and 6 swallows is 16 liang -/
axiom total_weight : 5 * x + 6 * y = 16

/-- Sparrows are heavier than swallows -/
axiom sparrow_heavier : x > y

/-- If one sparrow is exchanged with one swallow, the two groups weigh the same -/
axiom exchange_weight : 4 * x + y = x + 5 * y

/-- The system of equations representing the bird weight problem -/
theorem bird_weight_system : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = x + 5 * y) := by
  constructor
  · exact total_weight
  · exact exchange_weight

#check bird_weight_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_weight_system_l571_57135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_jordan_pairing_probability_l571_57157

/-- Represents a class of students -/
structure StudentClass where
  size : ℕ
  alex_in_class : Prop
  jordan_in_class : Prop

/-- The probability of a specific pairing in a random pairing of students -/
def pairingProbability (c : StudentClass) : ℚ :=
  1 / (c.size - 1)

/-- Theorem: In a class of 40 students, the probability of Alex being paired with Jordan is 1/39 -/
theorem alex_jordan_pairing_probability 
  (c : StudentClass)
  (h_size : c.size = 40)
  (h_alex : c.alex_in_class)
  (h_jordan : c.jordan_in_class)
  (h_distinct : c.alex_in_class ≠ c.jordan_in_class) :
  pairingProbability c = 1 / 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_jordan_pairing_probability_l571_57157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_without_repetition_four_digit_numbers_divisible_by_25_without_repetition_four_digit_numbers_greater_than_4032_without_repetition_l571_57101

-- Define the set of digits
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

-- Define a function to check if a number has repeated digits
def hasRepeatedDigits (n : Nat) : Bool := sorry

-- Define a function to check if a number is four-digit
def isFourDigit (n : Nat) : Bool := sorry

-- Define a function to count numbers satisfying certain conditions
def countNumbers (pred : Nat → Bool) : Nat := sorry

theorem four_digit_numbers_without_repetition :
  countNumbers (fun n => isFourDigit n ∧ ¬hasRepeatedDigits n ∧ n ∈ Finset.image id digits) = 300 := by sorry

theorem four_digit_numbers_divisible_by_25_without_repetition :
  countNumbers (fun n => isFourDigit n ∧ ¬hasRepeatedDigits n ∧ n % 25 = 0 ∧ n ∈ Finset.image id digits) = 21 := by sorry

theorem four_digit_numbers_greater_than_4032_without_repetition :
  countNumbers (fun n => isFourDigit n ∧ ¬hasRepeatedDigits n ∧ n > 4032 ∧ n ∈ Finset.image id digits) = 112 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_without_repetition_four_digit_numbers_divisible_by_25_without_repetition_four_digit_numbers_greater_than_4032_without_repetition_l571_57101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l571_57177

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The area of a rectangle given its width and height -/
def rectangle_area (width height : ℝ) : ℝ :=
  width * height

theorem triangle_fraction_of_grid : 
  let triangle_area := triangle_area 2 1 7 1 5 5
  let grid_area := rectangle_area 8 6
  triangle_area / grid_area = 5 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l571_57177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_income_at_175_l571_57112

/-- Represents the travel company's sightseeing car rental system -/
structure RentalSystem where
  totalCars : ℕ
  managementFee : ℝ
  rentalFee : ℝ

/-- Calculates the number of cars rented based on the rental fee -/
noncomputable def carsRented (sys : RentalSystem) : ℝ :=
  if sys.rentalFee ≤ 100 then sys.totalCars
  else sys.totalCars - (sys.rentalFee - 100) / 5

/-- Calculates the net income based on the rental fee -/
noncomputable def netIncome (sys : RentalSystem) : ℝ :=
  sys.rentalFee * carsRented sys - sys.managementFee

/-- Theorem stating that the maximum net income occurs at a rental fee of 175 yuan -/
theorem max_net_income_at_175 (sys : RentalSystem) :
  sys.totalCars = 50 ∧ sys.managementFee = 1100 →
  ∀ x, x > 0 → netIncome { sys with rentalFee := x } ≤ netIncome { sys with rentalFee := 175 } := by
  sorry

#check max_net_income_at_175

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_income_at_175_l571_57112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_max_area_l571_57178

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the relationship between P and Q
def Q_relation (xp yp xq yq : ℝ) : Prop := xq = xp / 3 ∧ yq = yp / 3

-- Define the line l
def line_l (k n x y : ℝ) : Prop := y = k * x + n

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4 / 9

-- Define the trajectory M
def M (x y : ℝ) : Prop := x^2 / (4 / 9) + y^2 / (2 / 9) = 1

-- Define the area of triangle ABO
noncomputable def area_ABO (k n : ℝ) : ℝ := 
  let d := |n| / Real.sqrt (k^2 + 1)
  Real.sqrt ((4 / 9 - d^2) * d^2)

-- State the theorem
theorem ellipse_trajectory_and_max_area :
  ∀ (xp yp xq yq k n : ℝ),
  E xp yp →
  Q_relation xp yp xq yq →
  (∀ x y, M x y → (line_l k n x y → x = xq ∧ y = yq)) →
  (∃ xa ya xb yb, line_l k n xa ya ∧ line_l k n xb yb ∧ C xa ya ∧ C xb yb) →
  (M xq yq ∧ area_ABO k n ≤ 2 / 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_and_max_area_l571_57178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l571_57196

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (t : ℝ) : ℝ × ℝ := (0, t^2 + 1)

noncomputable def distance (t : ℝ) : ℝ :=
  Real.sqrt ((1 - 0)^2 + (Real.sqrt 3 - t)^2)

theorem distance_range :
  ∀ t ∈ Set.Icc (-Real.sqrt 3) 2,
    1 ≤ distance t ∧ distance t ≤ Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l571_57196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_bisects_opposite_angles_parallelogram_diagonal_not_necessarily_bisects_opposite_angles_l571_57172

-- Define a point
structure Point :=
  (x y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a parallelogram
class Parallelogram (Q : Quadrilateral) :=
  (opposite_sides_parallel : Prop)
  (opposite_sides_equal : Prop)
  (opposite_angles_equal : Prop)
  (diagonals_bisect : Prop)

-- Define additional properties of a rhombus
class Rhombus (Q : Quadrilateral) extends Parallelogram Q :=
  (diagonals_perpendicular : Prop)

-- Define the property of diagonals bisecting opposite angles
def DiagonalsBisectOppositeAngles (Q : Quadrilateral) : Prop :=
  ∃ (p1 p2 : Prop), p1 ∧ p2

-- Theorem statement
theorem rhombus_diagonal_bisects_opposite_angles 
  (Q : Quadrilateral) (h : Rhombus Q) : 
  DiagonalsBisectOppositeAngles Q :=
sorry

-- Theorem statement for parallelogram
theorem parallelogram_diagonal_not_necessarily_bisects_opposite_angles 
  (Q : Quadrilateral) (h : Parallelogram Q) : 
  ¬ (∀ (Q : Quadrilateral), Parallelogram Q → DiagonalsBisectOppositeAngles Q) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_bisects_opposite_angles_parallelogram_diagonal_not_necessarily_bisects_opposite_angles_l571_57172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l571_57137

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - 1) * f a x

theorem min_value_g (a : ℝ) :
  f a 2 = 1/2 →
  ∃ x₀ ∈ Set.Icc (1/2 : ℝ) 2, ∀ x ∈ Set.Icc (1/2 : ℝ) 2, g a x₀ ≤ g a x ∧ g a x₀ = -1 :=
by
  intro h
  sorry

#check min_value_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l571_57137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l571_57107

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^(Real.sqrt (2*x - x^2 + 3))}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log ((x + 3) / (2 - x))}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = Set.Ioo (-3 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l571_57107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_from_sin_plus_cos_l571_57110

theorem tan_A_from_sin_plus_cos (A : ℝ) (h : Real.sin A + Real.cos A = -4/3) : Real.tan A = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_A_from_sin_plus_cos_l571_57110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custard_slice_price_is_6_verify_total_revenue_l571_57123

/-- Represents the price of a slice of custard pie -/
def custard_slice_price : ℚ := 6

/-- Represents the number of slices in a pumpkin pie -/
def pumpkin_slices : ℕ := 8

/-- Represents the number of slices in a custard pie -/
def custard_slices : ℕ := 6

/-- Represents the price of a slice of pumpkin pie -/
def pumpkin_slice_price : ℚ := 5

/-- Represents the number of pumpkin pies sold -/
def pumpkin_pies_sold : ℕ := 4

/-- Represents the number of custard pies sold -/
def custard_pies_sold : ℕ := 5

/-- Represents the total revenue -/
def total_revenue : ℚ := 340

/-- Theorem stating that the price of a custard pie slice is 6 dollars -/
theorem custard_slice_price_is_6 :
  custard_slice_price = 6 :=
by
  -- The proof goes here
  sorry

/-- Theorem verifying that the total revenue is correct -/
theorem verify_total_revenue :
  (pumpkin_slices * pumpkin_pies_sold * pumpkin_slice_price) +
  (custard_slices * custard_pies_sold * custard_slice_price) = total_revenue :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custard_slice_price_is_6_verify_total_revenue_l571_57123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_efficiency_l571_57150

noncomputable def work_efficiency (days : ℝ) : ℝ := 1 / days

theorem combined_work_efficiency 
  (days_A : ℝ) 
  (h1 : days_A > 0) : 
  work_efficiency 18 + work_efficiency 9 = 1 / 6 := by
  -- Proof steps would go here
  sorry

#check combined_work_efficiency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_efficiency_l571_57150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l571_57129

noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

def points : List (ℝ × ℝ) := [(2, 3), (4, -1), (-3, 4), (0, -7), (5, 0), (-6, 2)]

theorem farthest_point : 
  ∀ (p : ℝ × ℝ), p ∈ points → distance_from_origin 0 (-7) ≥ distance_from_origin p.1 p.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l571_57129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mitchells_goal_is_30_l571_57155

/-- Donovan Mitchell's goal for average points per game -/
noncomputable def mitchells_goal (
  current_average : ℝ) 
  (games_played : ℕ) 
  (total_games : ℕ) 
  (required_average_remaining : ℝ) : ℝ :=
  (current_average * (games_played : ℝ) + 
   required_average_remaining * ((total_games - games_played) : ℝ)) / (total_games : ℝ)

theorem mitchells_goal_is_30 :
  mitchells_goal 26 15 20 42 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mitchells_goal_is_30_l571_57155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l571_57185

theorem cosine_inequality (x y : ℝ) (h : x^2 + y^2 ≤ π/2) :
  Real.cos x + Real.cos y ≤ 1 + Real.cos (x*y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_l571_57185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l571_57118

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (4, -3), (-1, 2), and (2, -7) is 15 -/
theorem specific_triangle_area :
  triangle_area (4, -3) (-1, 2) (2, -7) = 15 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp [abs_of_nonneg]
  -- The proof is completed
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l571_57118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_bailing_rate_prevents_sinking_l571_57159

/-- Represents the fishing scenario with Alex and Jamie --/
structure FishingScenario where
  distance_to_shore : ℝ
  leak_rate : ℝ
  rain_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to prevent the boat from sinking --/
noncomputable def min_bailing_rate (scenario : FishingScenario) : ℝ :=
  let time_to_shore := scenario.distance_to_shore / scenario.rowing_speed
  let total_water_intake := (scenario.leak_rate + scenario.rain_rate) * time_to_shore
  (total_water_intake - scenario.max_water_capacity) / time_to_shore

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_for_scenario :
  let scenario := FishingScenario.mk 2 15 5 60 3
  min_bailing_rate scenario = 18.5 := by
  sorry

/-- Prove that the calculated bailing rate is sufficient to prevent sinking --/
theorem bailing_rate_prevents_sinking (scenario : FishingScenario) :
  let rate := min_bailing_rate scenario
  let time := scenario.distance_to_shore / scenario.rowing_speed
  (scenario.leak_rate + scenario.rain_rate - rate) * time ≤ scenario.max_water_capacity := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_for_scenario_bailing_rate_prevents_sinking_l571_57159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_election_problem_l571_57193

def election_problem (total_votes : ℚ) : Prop :=
  let initial_A := 0.40 * total_votes
  let initial_B := 0.30 * total_votes
  let initial_C := 0.20 * total_votes
  let initial_D := 0.10 * total_votes
  let shifted_votes : ℚ := 3000
  let new_A := initial_A - shifted_votes
  let new_B := initial_B + shifted_votes
  
  -- A wins initially with 10% margin
  initial_A = initial_B + 0.10 * total_votes ∧
  -- B wins after shift with 20% margin
  new_B = new_A + 0.20 * total_votes ∧
  -- C has 10% more votes than D
  initial_C = initial_D + 0.10 * total_votes

theorem solve_election_problem : 
  ∃ (total_votes : ℚ), election_problem total_votes ∧ total_votes = 20000 := by
  sorry

#eval (20000 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_election_problem_l571_57193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l571_57160

theorem expansion_coefficient_sum :
  ∀ (f : ℝ → ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x, f x = (x^2 - x - 6)^3 * (x^2 + x - 6)^3) →
  (∀ x, f x = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + 
              a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  a₁ + a₅ + a₉ = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l571_57160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l571_57125

theorem least_sum_m_n (m n : ℕ) : 
  (Nat.gcd (m + n) 330 = 1) →
  (∃ k : ℕ, m^m = k * n^n) →
  (¬ ∃ l : ℕ, m = l * n) →
  (∀ p q : ℕ, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^p = k * q^q) → 
    (¬ ∃ l : ℕ, p = l * q) → 
    (p + q : ℕ) ≥ (m + n : ℕ)) →
  m + n = 119 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l571_57125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_f_l571_57188

-- Define the function g
variable (g : ℝ → ℝ)

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

-- State the theorem
theorem tangent_line_f (g : ℝ → ℝ) (hg : (9 : ℝ) * 1 + g 1 - 1 = 0) :
  ∃ (m : ℝ), m * 1 + f g 1 = 0 ∧ 
  ∀ (x : ℝ), m * x + f g x = (f g 1 + m * (x - 1)) :=
by
  -- Prove the existence of m
  use 7
  constructor
  
  -- Prove m * 1 + f g 1 = 0
  · sorry
  
  -- Prove ∀ (x : ℝ), m * x + f g x = (f g 1 + m * (x - 1))
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_f_l571_57188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l571_57128

theorem logarithm_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^4) / Real.log (x^7)) * (Real.log (x^5) / Real.log (y^6)) *
  (Real.log (y^6) / Real.log (x^5)) * (Real.log (x^7) / Real.log (y^4)) =
  (1/4) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l571_57128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l571_57154

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed_kmph = 54)
  (h3 : bridge_length = 660) :
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 52 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l571_57154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_for_longest_seq_l571_57124

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- The sequence defined in the problem -/
def seq (x : ℕ) : ℕ → ℤ
| 0 => 1000  -- Added case for 0
| 1 => 1000
| 2 => x
| 3 => 1000 - x
| (n + 4) => seq x (n + 2) - seq x (n + 3)

/-- The theorem stating the maximum value of x for the longest non-negative sequence -/
theorem max_x_for_longest_seq : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ 618 ∧ 
  (∀ (y : ℕ), y > x → ∃ (n : ℕ), seq y n < 0 ∧ ∀ (m : ℕ), m < n → seq y m ≥ 0) ∧
  (∀ (n : ℕ), seq x n ≥ 0) :=
by sorry

#check max_x_for_longest_seq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_for_longest_seq_l571_57124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_option_difference_l571_57114

noncomputable def loan_amount : ℝ := 12000
noncomputable def compound_rate : ℝ := 0.08
noncomputable def simple_rate : ℝ := 0.10
noncomputable def loan_term : ℝ := 12
noncomputable def compound_frequency : ℝ := 2
noncomputable def partial_payment_time : ℝ := 6
noncomputable def partial_payment_fraction : ℝ := 1/3

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

noncomputable def compound_option_total : ℝ :=
  let first_period := compound_interest loan_amount compound_rate compound_frequency partial_payment_time
  let partial_payment := first_period * partial_payment_fraction
  let remaining_balance := first_period - partial_payment
  let second_period := compound_interest remaining_balance compound_rate compound_frequency partial_payment_time
  partial_payment + second_period

noncomputable def simple_option_total : ℝ :=
  simple_interest loan_amount simple_rate loan_term

theorem loan_option_difference :
  ‖simple_option_total - compound_option_total - 815‖ < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_option_difference_l571_57114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l571_57175

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b : ℝ × ℝ := (Real.sqrt 3, 1)

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  Real.arccos cos_theta = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l571_57175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l571_57108

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, c^2 * (f (x + y)) = f x * f y

theorem functional_equation_solution
  (f : ℝ → ℝ)
  (c : ℝ)
  (hf : Continuous f)
  (hc : c > 0)
  (heq : FunctionalEquation f c)
  (hf1 : f 1 = c) :
  ∀ x : ℝ, f x = c^x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l571_57108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l571_57104

-- Define the max function
noncomputable def max' (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Define M as a function of x and y
noncomputable def M (x y : ℝ) : ℝ := max' (|x - y^2 + 4|) (|2*y^2 - x + 8|)

-- State the theorem
theorem range_of_m : 
  (∀ (x y : ℝ), M x y ≥ 6) ∧ 
  (∀ m : ℝ, (∀ (x y : ℝ), M x y ≥ m^2 - 2*m) ↔ 1 - Real.sqrt 7 ≤ m ∧ m ≤ 1 + Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l571_57104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_common_terms_l571_57147

/-- Sequence A defined by A(n) = 5n - 2 -/
def A : ℕ → ℤ := λ n => 5*n - 2

/-- Sequence B defined by B(1) = 7 and B(k) = B(k-1) + d for k ≥ 2 -/
def B (d : ℤ) : ℕ → ℤ
  | 0 => 7
  | k + 1 => B d k + d

/-- A natural number n such that B(n) = A(m) for some m -/
def CommonTerm (d : ℤ) (n : ℕ) : Prop :=
  ∃ m : ℕ, B d n = A m

theorem infinite_common_terms :
  ∃ S : Set ℤ, (Set.Infinite S) ∧ 
    ∀ d ∈ S, Set.Infinite {n : ℕ | CommonTerm d n} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_common_terms_l571_57147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_function_a_range_l571_57186

/-- Given a function f(x) = log_a(x^3 - ax) that is monotonically increasing 
    in the interval (-1/2, 0), where a > 0 and a ≠ 1, prove that a ∈ [3/4, 1). -/
theorem monotonic_log_function_a_range (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (f : ℝ → ℝ)
  (h3 : f = λ x => Real.log (x^3 - a*x) / Real.log a)
  (h4 : StrictMonoOn f (Set.Ioo (-1/2) 0)) :
  a ∈ Set.Icc (3/4) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_log_function_a_range_l571_57186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l571_57181

/-- Calculate the length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) :
  train_length = 130 →
  crossing_time = 27.997760179185665 →
  train_speed_kmph = 36 →
  ∃ bridge_length : ℝ, abs (bridge_length - 149.97760179185665) < 0.01 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l571_57181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l571_57105

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (x / 2) - (1 / 2) * sin x

-- State the theorem
theorem f_properties :
  ∃ max_x ∈ Set.Icc 0 (2 * π),
    (∀ x ∈ Set.Icc 0 π, ∀ y ∈ Set.Icc 0 π, x ≤ y → f x ≤ f y) ∧
    (∀ y ∈ Set.Icc 0 (2 * π), f y ≤ f max_x) ∧
    f max_x = (3 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l571_57105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l571_57192

theorem largest_prime_factor_of_expression : 
  (Nat.factors (21^3 + 14^4 - 7^5)).foldl Nat.max 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l571_57192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_false_l571_57111

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Predicate: four points are not coplanar -/
def not_coplanar (A B C D : Point3D) : Prop := sorry

/-- Predicate: three points are not collinear -/
def not_collinear (P Q R : Point3D) : Prop := sorry

/-- The original proposition -/
def original_proposition (A B C D : Point3D) : Prop :=
  not_coplanar A B C D → 
    not_collinear A B C ∧ not_collinear A B D ∧ not_collinear A C D ∧ not_collinear B C D

/-- The inverse proposition -/
def inverse_proposition (A B C D : Point3D) : Prop :=
  (not_collinear A B C ∧ not_collinear A B D ∧ not_collinear A C D ∧ not_collinear B C D) →
    not_coplanar A B C D

/-- Theorem: The inverse proposition is false -/
theorem inverse_proposition_is_false : ∃ A B C D : Point3D, ¬(inverse_proposition A B C D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_false_l571_57111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_six_numbers_l571_57141

theorem existence_of_six_numbers : ∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (∀ (x y : ℕ), x ∈ ({a, b, c, d, e, f} : Set ℕ) → y ∈ ({a, b, c, d, e, f} : Set ℕ) → x ≠ y →
    ¬((x * y) ∣ (a + b + c + d + e + f))) ∧
  (∀ (x y z : ℕ), x ∈ ({a, b, c, d, e, f} : Set ℕ) → y ∈ ({a, b, c, d, e, f} : Set ℕ) → z ∈ ({a, b, c, d, e, f} : Set ℕ) →
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    ((x * y * z) ∣ (a + b + c + d + e + f))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_six_numbers_l571_57141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_is_546_l571_57144

/-- The least common multiple of 2, 3, and 7 -/
def lcm_2_3_7 : ℕ := 42

/-- The approximate count of numbers divisible by 2, 3, and 7 in the range -/
def approx_count : ℝ := 11.9

/-- The lower bound of the range -/
def lower_bound : ℕ := 100

/-- The upper bound of the range -/
def upper_bound : ℕ := 546

/-- Theorem stating that the upper bound is 546 given the conditions -/
theorem upper_bound_is_546 :
  ∃ (n : ℕ), n ≥ lower_bound ∧ n ≤ upper_bound ∧
  (n % lcm_2_3_7 = 0) ∧
  (Int.floor ((upper_bound - lower_bound) / lcm_2_3_7 : ℝ) = Int.floor approx_count) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upper_bound_is_546_l571_57144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_digit_assignment_l571_57149

/-- Represents a point in the hexagon (either a vertex or the center) -/
inductive HexagonPoint
| A | B | C | D | E | F | J

/-- Represents an assignment of digits to the hexagon points -/
def Assignment := HexagonPoint → Fin 7

/-- Checks if an assignment is valid (each digit used once) -/
def is_valid_assignment (a : Assignment) : Prop :=
  Function.Injective a

/-- Checks if the sums on the specified lines are equal -/
def has_equal_sums (a : Assignment) : Prop :=
  a HexagonPoint.A + a HexagonPoint.J + a HexagonPoint.C =
  a HexagonPoint.B + a HexagonPoint.J + a HexagonPoint.D ∧
  a HexagonPoint.B + a HexagonPoint.J + a HexagonPoint.D =
  a HexagonPoint.C + a HexagonPoint.J + a HexagonPoint.E

/-- The main theorem stating there are 144 valid assignments -/
theorem hexagon_digit_assignment :
  ∃ s : Finset Assignment, (∀ a ∈ s, is_valid_assignment a ∧ has_equal_sums a) ∧ s.card = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_digit_assignment_l571_57149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l571_57133

theorem coefficient_x_squared (m n : ℕ) 
  (h : 2 * m + 5 * n = 16) : 
  (3 * m * (m - 1) + 10 * n * (n - 1)) / 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l571_57133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circles_l571_57146

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 6*y + 9 = 0

-- Define the number of tangent lines
def num_tangent_lines : ℕ := 4

-- Theorem statement
theorem tangent_lines_to_circles :
  ∃! n : ℕ, n = num_tangent_lines ∧
  (∃ (lines : Finset (Set (ℝ × ℝ))), lines.card = n ∧
    (∀ l ∈ lines, 
      (∃ x y : ℝ, (x, y) ∈ l ∧ C₁ x y) ∧
      (∃ x y : ℝ, (x, y) ∈ l ∧ C₂ x y) ∧
      (∀ x y : ℝ, (x, y) ∈ l → (C₁ x y ∨ C₂ x y) ∧ 
        (¬(C₁ x y ∧ C₂ x y))))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circles_l571_57146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_and_function_l571_57113

-- Define the angle α and the parameter t
variable (α : Real) (t : Real)

-- Define the point P on the terminal side of α
def P : Real × Real := (3*t, 4*t)

-- Define the function f
noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi/2) * Real.cos (3*Real.pi/2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

-- State the theorem
theorem angle_values_and_function (h : t > 0) :
  Real.sin α = 4/5 ∧ 
  Real.cos α = 3/5 ∧ 
  Real.tan α = 4/3 ∧
  f α = -Real.cos α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_and_function_l571_57113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l571_57173

noncomputable section

/-- Curve C in polar coordinates -/
def curve_C (θ : ℝ) : ℝ := 3 / (2 - Real.cos θ)

/-- Line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ := (3 + t, 2 + 2*t)

/-- Intersection points of curve C and line l -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ θ t, 0 ≤ θ ∧ θ < 2*Real.pi ∧ 
       curve_C θ * Real.cos θ = p.1 ∧
       curve_C θ * Real.sin θ = p.2 ∧
       line_l t = p}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_distance :
  ∀ (A B : ℝ × ℝ), A ∈ intersection_points → B ∈ intersection_points → A ≠ B → distance A B = 60/19 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l571_57173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_hours_for_book_l571_57138

/-- The number of additional hours needed to finish reading a book -/
noncomputable def additional_hours_needed (total_pages : ℕ) (pages_per_hour : ℕ) (monday_hours : ℝ) (tuesday_hours : ℝ) : ℝ :=
  let pages_read := (monday_hours + tuesday_hours) * (pages_per_hour : ℝ)
  let pages_left := (total_pages : ℝ) - pages_read
  pages_left / (pages_per_hour : ℝ)

/-- Theorem stating the additional hours needed to finish the book -/
theorem additional_hours_for_book : 
  additional_hours_needed 387 12 3 6.5 = 22.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_hours_for_book_l571_57138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_count_expansion_terms_count_l571_57131

/-- Theorem about the number of distinct terms in the expansion of (x + 2y)^12 -/
theorem binomial_expansion_terms_count : ∃ (n : ℕ), n = 13 ∧ 
  n = Finset.card (Finset.range 13) :=
by
  -- Define the number of terms in the expansion
  let term_count := 13

  -- Prove that the number of terms is equal to term_count
  have h : Finset.card (Finset.range 13) = term_count := by rfl

  -- Construct the existence proof
  use term_count
  constructor
  · rfl
  · exact h

/-- Helper lemma: Each term in the expansion is unique -/
lemma unique_terms (k : ℕ) (hk : k ≤ 12) : 
  ∃! t : ℕ → ℕ → ℕ, ∃ c : ℕ, t = λ x y => c * x^(12 - k) * y^k :=
sorry

/-- The main theorem stating that the number of distinct terms is 13 -/
theorem expansion_terms_count : 
  (∀ k : ℕ, k ≤ 12 → ∃! t : ℕ → ℕ → ℕ, ∃ c : ℕ, t = λ x y => c * x^(12 - k) * y^k) → 
  (∃ n : ℕ, n = 13 ∧ n = Finset.card (Finset.range 13)) :=
by
  intro h
  exact binomial_expansion_terms_count


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_terms_count_expansion_terms_count_l571_57131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l571_57153

theorem trigonometric_inequality (x : ℝ) (n : ℕ+) (h : ∃ y, Real.tan x = y) :
  (Finset.sum (Finset.range n) (λ i => Real.sin x ^ (2 * (i + 1)))) ≤ Real.tan x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l571_57153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_for_odd_cosine_l571_57103

/-- A function representing y = 3cos(2x+φ) -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (2 * x + φ)

/-- The property of being an odd function -/
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem min_abs_phi_for_odd_cosine :
  ∃ φ₀ : ℝ, (is_odd_function (f φ₀)) ∧
    (∀ φ : ℝ, is_odd_function (f φ) → abs φ₀ ≤ abs φ) ∧
    abs φ₀ = π / 2 := by
  sorry

#check min_abs_phi_for_odd_cosine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_for_odd_cosine_l571_57103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l571_57122

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l571_57122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rectangle_ratio_l571_57151

/-- A rectangle with sides a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- A rhombus with diagonals d₁ and d₂. -/
structure Rhombus where
  d₁ : ℝ
  d₂ : ℝ
  d₁_pos : 0 < d₁
  d₂_pos : 0 < d₂

/-- A rhombus is inscribed in a rectangle if each vertex of the rhombus
    lies on a side of the rectangle. -/
def isInscribed (rect : Rectangle) (rhom : Rhombus) : Prop :=
  ∃ (θ : ℝ), 0 < θ ∧ θ < Real.pi/2 ∧
    rhom.d₁ = 2 * rect.a / Real.cos θ ∧
    rhom.d₂ = 2 * rect.b / Real.cos θ

/-- The theorem stating that for a rhombus inscribed in a rectangle,
    the ratio of the rhombus diagonals equals the ratio of the rectangle sides. -/
theorem rhombus_rectangle_ratio (rect : Rectangle) (rhom : Rhombus)
    (h : isInscribed rect rhom) :
    rhom.d₁ / rhom.d₂ = rect.a / rect.b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_rectangle_ratio_l571_57151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l571_57180

/-- Line l defined parametrically -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t/2, Real.sqrt 3 * t/2)

/-- Parabola C -/
def parabola_C (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Point M -/
def point_M : ℝ × ℝ := (1, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    t₁ ≠ t₂ ∧
    parabola_C (line_l t₁) ∧
    parabola_C (line_l t₂) ∧
    distance (line_l t₁) point_M + distance (line_l t₂) point_M = 16/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l571_57180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l571_57167

def a : ℝ × ℝ × ℝ := (5, 0, -2)
def b : ℝ × ℝ × ℝ := (6, 4, 3)

def c₁ : ℝ × ℝ × ℝ := (5 * a.1 - 3 * b.1, 5 * a.2.1 - 3 * b.2.1, 5 * a.2.2 - 3 * b.2.2)
def c₂ : ℝ × ℝ × ℝ := (6 * b.1 - 10 * a.1, 6 * b.2.1 - 10 * a.2.1, 6 * b.2.2 - 10 * a.2.2)

theorem vectors_collinear : ∃ (k : ℝ), c₂ = (k * c₁.1, k * c₁.2.1, k * c₁.2.2) := by
  use -2
  simp [c₁, c₂, a, b]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_collinear_l571_57167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_decrease_l571_57184

/-- Represents the fuel efficiency of a car at different speeds -/
structure CarEfficiency where
  mpg_at_45: ℚ  -- Miles per gallon at 45 mph
  miles_at_60: ℚ  -- Miles traveled at 60 mph
  gallons_at_60: ℚ  -- Gallons used at 60 mph

/-- Calculates the percentage decrease in fuel efficiency -/
def efficiency_decrease (car: CarEfficiency) : ℚ :=
  let mpg_at_60 := car.miles_at_60 / car.gallons_at_60
  let decrease := car.mpg_at_45 - mpg_at_60
  (decrease / car.mpg_at_45) * 100

/-- Theorem stating that the efficiency decrease for Car Z is 20% -/
theorem car_z_efficiency_decrease :
  let car_z : CarEfficiency := {
    mpg_at_45 := 51,
    miles_at_60 := 408,
    gallons_at_60 := 10
  }
  efficiency_decrease car_z = 20 := by
  sorry

#eval efficiency_decrease { mpg_at_45 := 51, miles_at_60 := 408, gallons_at_60 := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_z_efficiency_decrease_l571_57184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l571_57176

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the first n terms of a sequence. -/
def SequenceProduct (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (fun acc i => acc * a (i + 1)) 1

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 10 * a 11 = 2 →
  SequenceProduct a 20 = 1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l571_57176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_17_l571_57179

noncomputable def f (n : ℝ) : ℝ :=
  if n < 3 then n^2 + 1
  else if n < 6 then 3*n + 2
  else 2*n - 1

theorem f_composition_equals_17 : f (f (f 1)) = 17 := by
  -- Evaluate f(1)
  have h1 : f 1 = 2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(1)) = f(2)
  have h2 : f (f 1) = 5 := by
    rw [h1]
    simp [f]
    norm_num
  
  -- Evaluate f(f(f(1))) = f(5)
  rw [h2]
  simp [f]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_17_l571_57179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_impossibility_l571_57143

/-- Represents a coloring of smaller triangles in a divided equilateral triangle -/
def Coloring (n : ℕ) := Fin (n^2) → Bool

/-- Represents a recoloring operation along a line parallel to a side -/
def Recolor (n : ℕ) (c : Coloring n) (line : Fin n) : Coloring n :=
  sorry

/-- Predicate to check if all triangles are white -/
def AllWhite (n : ℕ) (c : Coloring n) : Prop :=
  ∀ i, c i = false

/-- Predicate to check if exactly one triangle is black -/
def OneBlack (n : ℕ) (c : Coloring n) : Prop :=
  ∃! i, c i = true

theorem recoloring_impossibility (n : ℕ) (h : n > 2) :
  ∀ (initial : Coloring n),
    OneBlack n initial →
    ¬∃ (steps : List (Fin n)),
      AllWhite n (steps.foldl (fun c line => Recolor n c line) initial) :=
by
  sorry

#check recoloring_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recoloring_impossibility_l571_57143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l571_57170

/-- The inequality function -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/2 * x^2 + 2*x - m*x

/-- The solution set of the inequality -/
noncomputable def solution_set (m : ℝ) : Set ℝ := {x | f m x > 0}

/-- The theorem statement -/
theorem inequality_solution (m : ℝ) : solution_set m = Set.Ioo 0 2 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l571_57170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l571_57117

noncomputable section

variable (f : ℝ → ℝ)

-- f is increasing on (0, +∞)
axiom f_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- f is defined on (0, +∞)
axiom f_domain : ∀ x, 0 < x → f x ∈ Set.range f

-- f(2) = 1
axiom f_2 : f 2 = 1

-- For any a, b ∈ (0, +∞), f(a) - f(b) = f(a/b)
axiom f_property : ∀ a b, 0 < a → 0 < b → f a - f b = f (a / b)

-- Main theorem
theorem main_theorem :
  (f 8 = 3) ∧
  (∀ x, 0 < x → (f (x + 2) - f (1 / (2 * x)) < 1 + f (x^2 + 4) ↔ 0 < x ∧ x < 2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l571_57117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l571_57156

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 1) / (3 * x^2 - 4)

-- Theorem statement
theorem f_at_3 : f 3 = 13 / 23 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the numerator and denominator
  simp [pow_two, mul_add, add_mul]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_3_l571_57156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fifteen_angle_l571_57194

noncomputable def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let hour_angle : ℝ := (hour % 12 + minute / 60 : ℝ) * 30
  let minute_angle : ℝ := minute * 6
  min (abs (hour_angle - minute_angle)) (360 - abs (hour_angle - minute_angle))

theorem three_fifteen_angle :
  clock_angle 3 15 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fifteen_angle_l571_57194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l571_57199

-- Define the two functions
noncomputable def f (x : ℝ) := x^2
noncomputable def g (x : ℝ) := Real.sqrt x

-- Define the area as the integral of the difference between g and f
noncomputable def area : ℝ := ∫ x in Set.Icc 0 1, g x - f x

-- Theorem statement
theorem area_between_curves : area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l571_57199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_equality_squared_l571_57106

theorem multiples_equality_squared (a b : ℕ) : 
  (a = (Finset.filter (λ x => x % 12 = 0 ∧ x > 0) (Finset.range 60)).card) →
  (b = (Finset.filter (λ x => x % 4 = 0 ∧ x % 6 = 0 ∧ x > 0) (Finset.range 60)).card) →
  (a - b)^2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_equality_squared_l571_57106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l571_57136

/-- Represents the number of sides in a convex polygon -/
def n : ℕ := 12

/-- Represents the common difference in the arithmetic progression of interior angles -/
def common_difference : ℝ := 4

/-- Represents the largest interior angle of the polygon -/
def largest_angle : ℝ := 170

/-- Represents the sum of interior angles of the polygon -/
def sum_of_angles : ℝ := 180 * (n - 2)

/-- Represents the smallest interior angle of the polygon -/
def smallest_angle : ℝ := largest_angle - (n - 1) * common_difference

/-- Theorem stating that n must equal 12 given the conditions -/
theorem polygon_sides_count : n = 12 := by
  -- The proof goes here
  sorry

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l571_57136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_12_l571_57183

/-- An arithmetic progression with first term a and common difference d -/
structure ArithmeticProgression where
  a : ℝ
  d : ℝ

/-- The nth term of an arithmetic progression -/
noncomputable def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def ArithmeticProgression.sum (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

theorem arithmetic_progression_sum_12 (ap : ArithmeticProgression) 
  (h : ap.nthTerm 4 + ap.nthTerm 12 = 20) :
  ap.sum 12 = 120 - 18 * ap.d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_12_l571_57183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l571_57162

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the line
def my_line (x y m : ℝ) : Prop := x + m*y - m - 2 = 0

-- Theorem statement
theorem circle_line_properties :
  ∃ (m : ℝ),
  (∀ x y, my_circle x y → (x - 1)^2 + (y - 2)^2 = 4) ∧
  (my_line 2 1 m) ∧
  (∃ x₁ y₁ x₂ y₂, 
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧ 
    my_line x₁ y₁ m ∧ my_line x₂ y₂ m ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 8) ∧
  (∃ x y, my_circle x y ∧ my_line x y m ∧ (x - 1)^2 + (y - 2)^2 < 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l571_57162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tip_percentage_l571_57119

theorem minimum_tip_percentage 
  (meal_cost : ℝ) 
  (total_paid : ℝ) 
  (tip_percentage : ℝ) 
  (h1 : meal_cost = 37.25)
  (h2 : total_paid = 40.975)
  (h3 : tip_percentage < 15)
  (h4 : total_paid = meal_cost + (tip_percentage / 100) * meal_cost) :
  tip_percentage ≥ 10 := by
  sorry

#check minimum_tip_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tip_percentage_l571_57119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_45_equals_1_l571_57174

/-- Tangent subtraction formula -/
axiom tan_sub (A B : ℝ) : Real.tan (A - B) = (Real.tan A - Real.tan B) / (1 + Real.tan A * Real.tan B)

/-- Given values for tangent of specific angles -/
axiom tan_60 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3
axiom tan_15 : Real.tan (15 * Real.pi / 180) = 2 - Real.sqrt 3

/-- 45 degrees equals 60 degrees minus 15 degrees -/
axiom angle_equality : 45 * Real.pi / 180 = 60 * Real.pi / 180 - 15 * Real.pi / 180

/-- Theorem: tangent of 45 degrees equals 1 -/
theorem tan_45_equals_1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_45_equals_1_l571_57174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPB_l571_57198

-- Define the curve C
def C (ρ θ : ℝ) : Prop :=
  ρ^2 * (1 + 3 * Real.sin θ^2) = 4

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define point A as the intersection of C with positive x-axis
def A : ℝ × ℝ := (2, 0)

-- Define point B as the intersection of C with positive y-axis
def B : ℝ × ℝ := (0, 1)

-- Define a point P on the curve C in the first quadrant
noncomputable def P (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, Real.sin φ)

-- Define the area of quadrilateral OAPB
noncomputable def area_OAPB (φ : ℝ) : ℝ :=
  Real.cos φ + Real.sin φ

-- State the theorem
theorem max_area_OAPB :
  ∃ (max_area : ℝ), max_area = Real.sqrt 2 ∧
  ∀ (φ : ℝ), 0 < φ ∧ φ < Real.pi / 2 →
  area_OAPB φ ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPB_l571_57198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_shop_earnings_l571_57142

/-- Represents the types of pies sold in the shop -/
inductive PieType
  | Custard
  | Apple
  | Blueberry

/-- Calculates the total earnings from selling all slices of pies -/
def total_earnings (price_per_slice : PieType → ℕ) 
  (slices_per_pie : PieType → ℕ) (whole_pies : PieType → ℕ) : ℕ :=
  (price_per_slice PieType.Custard * slices_per_pie PieType.Custard * whole_pies PieType.Custard) +
  (price_per_slice PieType.Apple * slices_per_pie PieType.Apple * whole_pies PieType.Apple) +
  (price_per_slice PieType.Blueberry * slices_per_pie PieType.Blueberry * whole_pies PieType.Blueberry)

/-- The main theorem stating that the total earnings is $608 -/
theorem pie_shop_earnings : 
  total_earnings 
    (fun | PieType.Custard => 3 | PieType.Apple => 4 | PieType.Blueberry => 5)
    (fun | PieType.Custard => 10 | PieType.Apple => 8 | PieType.Blueberry => 12)
    (fun | PieType.Custard => 6 | PieType.Apple => 4 | PieType.Blueberry => 5)
  = 608 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_shop_earnings_l571_57142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l571_57139

noncomputable def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 4) / x

theorem max_a_value (a : ℝ) : 
  (∀ s t : ℝ, s > 0 → t > 0 → g a s ≥ f t) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l571_57139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_reals_l571_57189

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := Real.log (a * x^2 - b * x - c) / Real.log 10

-- State the theorem
theorem range_of_f_is_reals (a b c : ℝ) (ha : a > 0) :
  (Set.range (f a b c) = Set.univ) ↔ (∃ x : ℝ, a * x^2 ≤ b * x + c) := by
  sorry

-- You can add more helper lemmas or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_reals_l571_57189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l571_57161

def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 7

theorem quadratic_function_properties :
  (∀ x : ℝ, (deriv (deriv f)) x > 0) ∧ 
  (∃ x : ℝ, deriv f x = 0 ∧ x = 2) ∧
  f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l571_57161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l571_57130

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x - 1)^2 * (x + 1) / (x - 2))

def domain : Set ℝ := Set.Iic (-1) ∪ {1} ∪ Set.Ioi 2

theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (∃ y : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l571_57130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l571_57191

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_problem (a : ℝ) :
  (∃ k : ℝ, ∃ m : ℝ, ∃ n : ℝ,
    k * m = f m ∧
    k * n = g (a * n) ∧
    k = (f m - 0) / (m - 0) ∧
    k = (g (a * n) - 0) / (n - 0)) →
  a = 1 / Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l571_57191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l571_57120

/-- Parabola type representing y^2 = -4x --/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = -4*x

/-- Focus of the parabola --/
noncomputable def focus : ℝ × ℝ := (1/4, 0)

/-- Point A --/
def point_A : ℝ × ℝ := (-2, 1)

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating the minimum value of |PF| + |PA| --/
theorem min_distance_sum (P : Parabola) :
  ∃ (min_val : ℝ), min_val = 3 ∧
  ∀ (Q : Parabola), distance (Q.x, Q.y) focus + distance (Q.x, Q.y) point_A ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l571_57120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l571_57134

theorem trig_identity (α φ : ℝ) : 
  (Real.cos φ) ^ 2 + (Real.cos (α - φ)) ^ 2 - 2 * (Real.cos α) * (Real.cos φ) * (Real.cos (α - φ)) = (Real.sin α) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l571_57134
