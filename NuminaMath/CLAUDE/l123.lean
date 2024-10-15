import Mathlib

namespace NUMINAMATH_CALUDE_smallest_apocalyptic_number_l123_12349

/-- A number is apocalyptic if it has 6 different positive divisors that sum to 3528 -/
def IsApocalyptic (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧
    d₅ ≠ d₆ ∧
    d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧ d₄ > 0 ∧ d₅ > 0 ∧ d₆ > 0 ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧
    d₁ + d₂ + d₃ + d₄ + d₅ + d₆ = 3528

theorem smallest_apocalyptic_number :
  IsApocalyptic 1440 ∧ ∀ m : ℕ, m < 1440 → ¬IsApocalyptic m := by
  sorry

end NUMINAMATH_CALUDE_smallest_apocalyptic_number_l123_12349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_neither_necessary_nor_sufficient_l123_12339

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_neither_necessary_nor_sufficient :
  ∃ (a : ℕ → ℝ) (m n p q : ℕ),
    arithmetic_sequence a ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧
    (a m + a n > a p + a q ∧ m + n ≤ p + q) ∧
    (m + n > p + q ∧ a m + a n ≤ a p + a q) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_neither_necessary_nor_sufficient_l123_12339


namespace NUMINAMATH_CALUDE_twenty_fourth_digit_is_8_l123_12375

-- Define the decimal representations of 1/7 and 1/9
def decimal_1_7 : ℚ := 1 / 7
def decimal_1_9 : ℚ := 1 / 9

-- Define the sum of the decimal representations
def sum_decimals : ℚ := decimal_1_7 + decimal_1_9

-- Function to get the nth digit after the decimal point
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem twenty_fourth_digit_is_8 :
  nth_digit_after_decimal sum_decimals 24 = 8 := by sorry

end NUMINAMATH_CALUDE_twenty_fourth_digit_is_8_l123_12375


namespace NUMINAMATH_CALUDE_same_terminal_side_330_neg_30_l123_12372

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- The angle -30° -/
def angle_neg_30 : ℝ := -30

/-- The angle 330° -/
def angle_330 : ℝ := 330

/-- Theorem: 330° has the same terminal side as -30° -/
theorem same_terminal_side_330_neg_30 :
  same_terminal_side angle_330 angle_neg_30 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_330_neg_30_l123_12372


namespace NUMINAMATH_CALUDE_finite_n_with_prime_factors_in_A_l123_12314

theorem finite_n_with_prime_factors_in_A (A : Finset Nat) (a : Nat) 
  (h_A : ∀ p ∈ A, Nat.Prime p) (h_a : a ≥ 2) :
  ∃ S : Finset Nat, ∀ n : Nat, (∀ p : Nat, p ∣ (a^n - 1) → p ∈ A) → n ∈ S :=
by sorry

end NUMINAMATH_CALUDE_finite_n_with_prime_factors_in_A_l123_12314


namespace NUMINAMATH_CALUDE_line_parallel_perp_implies_planes_perp_l123_12303

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def planesPerpendicular (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is parallel to one plane and perpendicular to another,
    then the two planes are perpendicular -/
theorem line_parallel_perp_implies_planes_perp
  (c : Line3D) (α β : Plane3D)
  (h1 : parallel c α)
  (h2 : perpendicular c β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perp_implies_planes_perp_l123_12303


namespace NUMINAMATH_CALUDE_problem_solution_l123_12370

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + m
def h (m : ℝ) (x : ℝ) : ℝ := 2^(2*x) + m - 2*m*2^x

-- Define the minimum value function
def H (m : ℝ) : ℝ :=
  if m < 1 then 1 - m
  else if m ≤ 2 then m - m^2
  else 4 - 3*m

theorem problem_solution :
  ∀ (a m : ℝ),
  (a > 0 ∧ a ≠ 1 ∧ m > 0) →
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f a x ≤ 5/2 ∧ f a x ≥ 0) →
  (f a 1 + f a (-1) = 5/2) →
  (a = 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → h m x ≥ H m) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → |1 - m*(2^x + m/2^x)| ≤ 1 → m ∈ Set.Icc 0 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l123_12370


namespace NUMINAMATH_CALUDE_power_inequality_l123_12380

theorem power_inequality (a : ℝ) (n : ℕ) :
  (a > 1 → a^n > 1) ∧ (a < 1 → a^n < 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l123_12380


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l123_12310

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
def hyperbola_condition := a > 0 ∧ b > 0
def focus_condition := ∃ (y : ℝ), (6^2 : ℝ) = a^2 + b^2
def asymptote_condition := b / a = Real.sqrt 3

-- State the theorem
theorem hyperbola_parameters
  (h1 : hyperbola_condition a b)
  (h2 : focus_condition a b)
  (h3 : asymptote_condition a b) :
  a^2 = 9 ∧ b^2 = 27 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l123_12310


namespace NUMINAMATH_CALUDE_not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l123_12391

-- 1. A function f: ℝ → ℝ that satisfies f(-2) = f(2) is not necessarily an even function
theorem not_necessarily_even (f : ℝ → ℝ) (h : f (-2) = f 2) :
  ¬ ∀ x, f (-x) = f x :=
sorry

-- 2. If f: ℝ → ℝ is monotonically increasing on (-∞, 0] and [0, +∞), then f is increasing on ℝ
theorem increasing_on_reals (f : ℝ → ℝ)
  (h1 : ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≤ f y)
  (h2 : ∀ x y, x ≤ y → 0 ≤ x → 0 ≤ y → f x ≤ f y) :
  ∀ x y, x ≤ y → f x ≤ f y :=
sorry

-- 3. If f: [a, b] → ℝ (where a < c < b) is increasing on [a, c) and decreasing on [c, b],
--    then f(c) is the maximum value of f on [a, b]
theorem max_at_turning_point {a b c : ℝ} (h : a < c ∧ c < b) (f : ℝ → ℝ)
  (h1 : ∀ x y, a ≤ x → x < y → y < c → f x ≤ f y)
  (h2 : ∀ x y, c < x → x < y → y ≤ b → f y ≤ f x) :
  ∀ x, a ≤ x → x ≤ b → f x ≤ f c :=
sorry

-- 4. For f(x) = √x and any x₁, x₂ ∈ (0, +∞), (f(x₁) + f(x₂))/2 ≤ f((x₁ + x₂)/2)
theorem sqrt_convexity (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) :
  (Real.sqrt x₁ + Real.sqrt x₂) / 2 ≤ Real.sqrt ((x₁ + x₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l123_12391


namespace NUMINAMATH_CALUDE_polynomial_factorization_l123_12330

theorem polynomial_factorization (x : ℝ) : 
  x^5 + x^4 + 1 = (x^2 + x + 1) * (x^3 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l123_12330


namespace NUMINAMATH_CALUDE_white_balls_count_l123_12324

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l123_12324


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l123_12355

theorem like_terms_exponent_relation (m n : ℕ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 3 * x^(3*m) * y^2 = k * x^6 * y^n) → m^n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l123_12355


namespace NUMINAMATH_CALUDE_zero_in_A_l123_12379

def A : Set ℝ := {x | x * (x - 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_zero_in_A_l123_12379


namespace NUMINAMATH_CALUDE_constant_chord_length_l123_12360

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the satellite circle -/
def satellite_circle (x y a b : ℝ) : Prop := x^2 + y^2 = a^2 + b^2

/-- Theorem statement -/
theorem constant_chord_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h_ecc : (a^2 - b^2) / a^2 = 1/2)
  (h_point : ellipse 2 (Real.sqrt 2) a b)
  (h_sat : satellite_circle 2 (Real.sqrt 2) a b) :
  ∃ (M N : ℝ × ℝ),
    ∀ (P : ℝ × ℝ), satellite_circle P.1 P.2 a b →
      ∃ (l₁ l₂ : ℝ → ℝ),
        (∀ x, (l₁ x - P.2) * (l₂ x - P.2) = -(x - P.1)^2) ∧
        (∃! x₁, ellipse x₁ (l₁ x₁) a b) ∧
        (∃! x₂, ellipse x₂ (l₂ x₂) a b) ∧
        satellite_circle M.1 M.2 a b ∧
        satellite_circle N.1 N.2 a b ∧
        (M.1 - N.1)^2 + (M.2 - N.2)^2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_chord_length_l123_12360


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l123_12358

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l123_12358


namespace NUMINAMATH_CALUDE_molecular_weight_NH4_correct_l123_12361

/-- The molecular weight of NH4 in grams per mole -/
def molecular_weight_NH4 : ℝ := 18

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 7

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 126

/-- Theorem stating that the molecular weight of NH4 is correct given the sample information -/
theorem molecular_weight_NH4_correct :
  molecular_weight_NH4 * sample_moles = sample_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_NH4_correct_l123_12361


namespace NUMINAMATH_CALUDE_product_inequality_l123_12396

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l123_12396


namespace NUMINAMATH_CALUDE_chord_length_l123_12387

/-- The length of the chord intercepted by the circle x^2 + y^2 = 4 on the line x - √3y + 2√3 = 0 is 2. -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 4) → (x - Real.sqrt 3 * y + 2 * Real.sqrt 3 = 0) → 
  ∃ (a b c d : ℝ), (a^2 + b^2 = 4) ∧ (c^2 + d^2 = 4) ∧ 
  (a - Real.sqrt 3 * b + 2 * Real.sqrt 3 = 0) ∧ 
  (c - Real.sqrt 3 * d + 2 * Real.sqrt 3 = 0) ∧ 
  ((a - c)^2 + (b - d)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_chord_length_l123_12387


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l123_12311

/-- Given that the solution set of ax^2 + (a-5)x - 2 > 0 is {x | -2 < x < -1/4},
    prove the following statements. -/
theorem quadratic_inequality_problem (a : ℝ) 
  (h : ∀ x, ax^2 + (a-5)*x - 2 > 0 ↔ -2 < x ∧ x < -1/4) :
  /- 1. a = -4 -/
  (a = -4) ∧ 
  /- 2. The solution set of 2x^2 + (2-a)x - a > 0 is (-∞, -2) ∪ (-1, ∞) -/
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -2 ∨ x > -1) ∧
  /- 3. The range of b such that -ax^2 + bx + 3 ≥ 0 for all real x 
        is [-4√3, 4√3] -/
  (∀ b, (∀ x, -a*x^2 + b*x + 3 ≥ 0) ↔ -4*Real.sqrt 3 ≤ b ∧ b ≤ 4*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l123_12311


namespace NUMINAMATH_CALUDE_test_passing_difference_l123_12323

theorem test_passing_difference (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arithmetic_correct : ℚ) (algebra_correct : ℚ) (geometry_correct : ℚ) (passing_grade : ℚ)
  (h1 : total = 90)
  (h2 : arithmetic = 20)
  (h3 : algebra = 40)
  (h4 : geometry = 30)
  (h5 : arithmetic_correct = 60 / 100)
  (h6 : algebra_correct = 50 / 100)
  (h7 : geometry_correct = 70 / 100)
  (h8 : passing_grade = 65 / 100)
  (h9 : total = arithmetic + algebra + geometry) :
  ⌈total * passing_grade⌉ - (⌊arithmetic * arithmetic_correct⌋ + ⌊algebra * algebra_correct⌋ + ⌊geometry * geometry_correct⌋) = 6 := by
  sorry

end NUMINAMATH_CALUDE_test_passing_difference_l123_12323


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l123_12338

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule where
  hireYear : ℕ
  hireAge : ℕ
  retirementYear : ℕ

/-- Calculates the required total of age and years of employment for retirement -/
def requiredTotal (rule : RetirementRule) : ℕ :=
  let ageAtRetirement := rule.hireAge + (rule.retirementYear - rule.hireYear)
  let yearsOfEmployment := rule.retirementYear - rule.hireYear
  ageAtRetirement + yearsOfEmployment

/-- Theorem stating that the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
    (h1 : rule.hireYear = 1989)
    (h2 : rule.hireAge = 32)
    (h3 : rule.retirementYear = 2008) :
  requiredTotal rule = 70 := by
  sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l123_12338


namespace NUMINAMATH_CALUDE_simplify_expression_l123_12398

theorem simplify_expression : (2^8 + 5^3) * (2^2 - (-1)^5)^7 = 29765625 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l123_12398


namespace NUMINAMATH_CALUDE_machine_values_after_two_years_l123_12390

def machineValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (initialValue * depreciationRate * years)

def combinedValue (valueA valueB valueC : ℝ) : ℝ :=
  valueA + valueB + valueC

theorem machine_values_after_two_years :
  let machineA := machineValue 8000 0.20 2
  let machineB := machineValue 10000 0.15 2
  let machineC := machineValue 12000 0.10 2
  combinedValue machineA machineB machineC = 21400 := by
  sorry

end NUMINAMATH_CALUDE_machine_values_after_two_years_l123_12390


namespace NUMINAMATH_CALUDE_division_problem_l123_12325

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 176 →
  quotient = 12 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l123_12325


namespace NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l123_12322

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_gt_beta_squared_l123_12322


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l123_12384

theorem largest_gcd_of_sum_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1008 ∧ Nat.gcd x y = 504 ∧ 
  ∀ (c d : ℕ), c > 0 → d > 0 → c + d = 1008 → Nat.gcd c d ≤ 504 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l123_12384


namespace NUMINAMATH_CALUDE_spinner_probability_l123_12367

theorem spinner_probability (p : ℝ) (n : ℕ) : 
  p = 3/4 → (p^n = 0.5625) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l123_12367


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l123_12335

theorem fixed_point_of_exponential_function (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-2) + 4
  f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l123_12335


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l123_12376

/-- A quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- A linear function -/
def LinearFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ m * x + n

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c m n : ℝ) :
  let f := QuadraticFunction a b c
  let g := LinearFunction m n
  (f (-1) = 2) ∧ (g (-1) = 2) ∧ (f 2 = 5) ∧ (g 2 = 5) ∧
  (∃ x₀, ∀ x, f x₀ ≤ f x) ∧ (f x₀ = 1) →
  (f = λ x ↦ x^2 + 1) ∨ (f = λ x ↦ (1/9) * x^2 + (8/9) * x + 25/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l123_12376


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l123_12347

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_a3 : a 3 = 7) :
  a 8 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l123_12347


namespace NUMINAMATH_CALUDE_train_travel_theorem_l123_12357

/-- Represents the distance traveled by a train -/
def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_travel_theorem (initial_distance initial_time final_time : ℝ) 
  (h1 : initial_distance = 300)
  (h2 : initial_time = 20)
  (h3 : final_time = 600) : 
  train_distance (initial_distance / initial_time) final_time = 9000 := by
  sorry

#check train_travel_theorem

end NUMINAMATH_CALUDE_train_travel_theorem_l123_12357


namespace NUMINAMATH_CALUDE_diagonal_intersection_l123_12332

/-- A regular 18-sided polygon -/
structure RegularPolygon18 where
  vertices : Fin 18 → ℝ × ℝ
  is_regular : ∀ i j : Fin 18, 
    dist (vertices i) (vertices ((i + 1) % 18)) = 
    dist (vertices j) (vertices ((j + 1) % 18))

/-- A diagonal of the polygon -/
def diagonal (p : RegularPolygon18) (i j : Fin 18) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • p.vertices i + t • p.vertices j}

/-- The statement to be proved -/
theorem diagonal_intersection (p : RegularPolygon18) :
  ∃ x : ℝ × ℝ, x ∈ diagonal p 1 11 ∩ diagonal p 7 17 ∩ diagonal p 4 15 ∧
  (∀ i : Fin 18, x ∉ diagonal p i ((i + 9) % 18)) :=
sorry

end NUMINAMATH_CALUDE_diagonal_intersection_l123_12332


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l123_12302

theorem quadratic_equation_proof (m : ℝ) (h1 : m < 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + m
  (∃ x : ℝ, f x = 0 ∧ x = -1) →
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) ∧  -- two distinct real roots
  m = -3 ∧                                  -- value of m
  (∃ x : ℝ, f x = 0 ∧ x = 3)                -- other root
:= by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l123_12302


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l123_12381

def is_valid_assignment (K O A L V D : ℕ) : Prop :=
  K ≠ O ∧ K ≠ A ∧ K ≠ L ∧ K ≠ V ∧ K ≠ D ∧
  O ≠ A ∧ O ≠ L ∧ O ≠ V ∧ O ≠ D ∧
  A ≠ L ∧ A ≠ V ∧ A ≠ D ∧
  L ≠ V ∧ L ≠ D ∧
  V ≠ D ∧
  K < 10 ∧ O < 10 ∧ A < 10 ∧ L < 10 ∧ V < 10 ∧ D < 10

def satisfies_equation (K O A L V D : ℕ) : Prop :=
  1000 * K + 100 * O + 10 * K + A +
  1000 * K + 100 * O + 10 * L + A =
  1000 * V + 100 * O + 10 * D + A

theorem alphametic_puzzle_solution :
  ∃! (K O A L V D : ℕ), 
    is_valid_assignment K O A L V D ∧
    satisfies_equation K O A L V D ∧
    K = 3 ∧ O = 9 ∧ A = 0 ∧ L = 8 ∧ V = 7 ∧ D = 1 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l123_12381


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l123_12383

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_four_digit_number (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : n % 10 ≠ 0)
  (h3 : 2014 % (first_two_digits n) = 0)
  (h4 : 2014 % ((first_two_digits n) * (last_two_digits n)) = 0) :
  n ≤ 5376 ∧ ∃ m : ℕ, m = 5376 ∧ 
    is_four_digit m ∧ 
    m % 10 ≠ 0 ∧ 
    2014 % (first_two_digits m) = 0 ∧ 
    2014 % ((first_two_digits m) * (last_two_digits m)) = 0 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l123_12383


namespace NUMINAMATH_CALUDE_min_value_quadratic_l123_12354

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 + 8 * x + 18 → y ≥ min_y ∧ min_y = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l123_12354


namespace NUMINAMATH_CALUDE_max_socks_pulled_correct_l123_12353

/-- Represents the state of socks in the drawer and pulled out -/
structure SockState where
  white_in_drawer : ℕ
  black_in_drawer : ℕ
  white_pulled : ℕ
  black_pulled : ℕ

/-- The initial state of socks -/
def initial_state : SockState :=
  { white_in_drawer := 8
  , black_in_drawer := 15
  , white_pulled := 0
  , black_pulled := 0 }

/-- Predicate to check if more black socks than white socks have been pulled -/
def more_black_than_white (state : SockState) : Prop :=
  state.black_pulled > state.white_pulled

/-- The maximum number of socks that can be pulled -/
def max_socks_pulled : ℕ := 17

/-- Theorem stating the maximum number of socks that can be pulled -/
theorem max_socks_pulled_correct :
  ∀ (state : SockState),
    state.white_in_drawer + state.black_in_drawer + state.white_pulled + state.black_pulled = 23 →
    state.white_pulled + state.black_pulled ≤ max_socks_pulled →
    ¬(more_black_than_white state) :=
  sorry

#check max_socks_pulled_correct

end NUMINAMATH_CALUDE_max_socks_pulled_correct_l123_12353


namespace NUMINAMATH_CALUDE_log_product_equals_one_l123_12394

theorem log_product_equals_one :
  Real.log 5 / Real.log 2 * Real.log 2 / Real.log 3 * Real.log 3 / Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l123_12394


namespace NUMINAMATH_CALUDE_parabola_p_value_l123_12300

/-- Given a parabola with equation y^2 = 2px and axis of symmetry x = -1, prove that p = 2 -/
theorem parabola_p_value (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) → 
  (∀ y : ℝ, y^2 = -2*p) → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_p_value_l123_12300


namespace NUMINAMATH_CALUDE_power_zero_fraction_l123_12333

theorem power_zero_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a / b) ^ (0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_fraction_l123_12333


namespace NUMINAMATH_CALUDE_johnson_family_seating_l123_12365

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the number of sons and daughters
def num_sons : ℕ := 5
def num_daughters : ℕ := 4

-- Define the total number of children
def total_children : ℕ := num_sons + num_daughters

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ :=
  factorial total_children - (factorial num_sons * factorial num_daughters)

-- Theorem statement
theorem johnson_family_seating :
  seating_arrangements = 360000 :=
sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l123_12365


namespace NUMINAMATH_CALUDE_min_value_theorem_l123_12312

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 10) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 14 ∧
  ((x + 10) / Real.sqrt (x - 4) = 2 * Real.sqrt 14 ↔ x = 22) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l123_12312


namespace NUMINAMATH_CALUDE_solution_sets_equality_l123_12307

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x : ℝ, |8*x + 9| < 7 ↔ a*x^2 + b*x > 2) → 
  (a = -4 ∧ b = -9) := by
sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l123_12307


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l123_12341

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l123_12341


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_factor_l123_12397

theorem polynomial_divisibility_and_factor :
  let p (x : ℝ) := 6 * x^3 - 18 * x^2 + 24 * x - 24
  let q (x : ℝ) := x - 2
  let r (x : ℝ) := 6 * x^2 + 4
  (∃ (s : ℝ → ℝ), p = q * s) ∧ (∃ (t : ℝ → ℝ), p = r * t) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_factor_l123_12397


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l123_12336

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  (a₀ + a₁) + (a₀ + a₂) + (a₀ + a₃) + (a₀ + a₄) + (a₀ + a₅) + (a₀ + a₆) + (a₀ + a₇) + (a₀ + a₈) + (a₀ + a₉) + (a₀ + a₁₀) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l123_12336


namespace NUMINAMATH_CALUDE_solution_bounds_and_expression_l123_12389

def system_of_equations (x y m : ℝ) : Prop :=
  3 * (x + 1) / 2 + y = 2 ∧ 3 * x - m = 2 * y

theorem solution_bounds_and_expression (x y m : ℝ) 
  (h_system : system_of_equations x y m) 
  (h_x_bound : x ≤ 1) 
  (h_y_bound : y ≤ 1) : 
  (-3 ≤ m ∧ m ≤ 5) ∧ 
  |x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_bounds_and_expression_l123_12389


namespace NUMINAMATH_CALUDE_expected_value_of_winnings_l123_12392

def fair_10_sided_die : Finset ℕ := Finset.range 10

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 0 then roll else 0

theorem expected_value_of_winnings :
  (Finset.sum fair_10_sided_die (λ roll => (1 : ℚ) / 10 * winnings roll)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_winnings_l123_12392


namespace NUMINAMATH_CALUDE_inequality_solution_set_l123_12317

theorem inequality_solution_set (x : ℝ) :
  (2 < (1 / (x - 1)) ∧ (1 / (x - 1)) < 3 ∧ 0 < x - 1) ↔ (4/3 < x ∧ x < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l123_12317


namespace NUMINAMATH_CALUDE_infinite_points_in_region_l123_12366

theorem infinite_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 > 0 ∧ p.2 > 0) ∧ 
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 + p.2 ≤ 7) ∧
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 ≥ 1) ∧
    Set.Infinite S :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_points_in_region_l123_12366


namespace NUMINAMATH_CALUDE_students_in_all_events_l123_12363

theorem students_in_all_events 
  (total_students : ℕ) 
  (event_A_participants : ℕ) 
  (event_B_participants : ℕ) 
  (h1 : total_students = 45)
  (h2 : event_A_participants = 39)
  (h3 : event_B_participants = 28)
  (h4 : event_A_participants + event_B_participants - total_students ≤ event_A_participants)
  (h5 : event_A_participants + event_B_participants - total_students ≤ event_B_participants) :
  event_A_participants + event_B_participants - total_students = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_in_all_events_l123_12363


namespace NUMINAMATH_CALUDE_infinitely_many_special_triangles_l123_12364

/-- A triangle with integer area formed by square roots of distinct non-square integers -/
structure SpecialTriangle where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃
  not_squares : ¬∃ m : ℕ, a₁ = m^2 ∧ ¬∃ n : ℕ, a₂ = n^2 ∧ ¬∃ k : ℕ, a₃ = k^2
  triangle_inequality : Real.sqrt a₁.val + Real.sqrt a₂.val > Real.sqrt a₃.val ∧
                        Real.sqrt a₁.val + Real.sqrt a₃.val > Real.sqrt a₂.val ∧
                        Real.sqrt a₂.val + Real.sqrt a₃.val > Real.sqrt a₁.val
  integer_area : ∃ S : ℕ, 16 * S^2 = (a₁ + a₂ + a₃)^2 - 2 * (a₁^2 + a₂^2 + a₃^2)

/-- There exist infinitely many SpecialTriangles -/
theorem infinitely_many_special_triangles : 
  ∀ n : ℕ, ∃ (triangles : Fin n → SpecialTriangle), 
    ∀ i j : Fin n, i ≠ j → 
      ¬∃ (k : ℚ), (k * (triangles i).a₁ : ℚ) = (triangles j).a₁ ∧ 
                   (k * (triangles i).a₂ : ℚ) = (triangles j).a₂ ∧ 
                   (k * (triangles i).a₃ : ℚ) = (triangles j).a₃ :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_triangles_l123_12364


namespace NUMINAMATH_CALUDE_det_A_eq_48_l123_12362

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_48_l123_12362


namespace NUMINAMATH_CALUDE_function_positive_iff_a_greater_half_l123_12373

/-- The function f(x) = ax² - 2x + 2 is positive for all x in (1, 4) if and only if a > 1/2 -/
theorem function_positive_iff_a_greater_half (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 4 → a * x^2 - 2*x + 2 > 0) ↔ a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_function_positive_iff_a_greater_half_l123_12373


namespace NUMINAMATH_CALUDE_remainder_theorem_l123_12309

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l123_12309


namespace NUMINAMATH_CALUDE_arctan_sum_tan_l123_12343

theorem arctan_sum_tan (x y : Real) :
  x = 45 * π / 180 →
  y = 30 * π / 180 →
  Real.arctan (Real.tan x + 2 * Real.tan y) = 75 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_tan_l123_12343


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l123_12340

/-- The number of remaining movies to watch in the 'crazy silly school' series -/
def remaining_movies (total : ℕ) (watched : ℕ) : ℕ :=
  total - watched

theorem crazy_silly_school_movies : 
  remaining_movies 17 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l123_12340


namespace NUMINAMATH_CALUDE_problem_1_l123_12315

theorem problem_1 : Real.sqrt 8 - 4 * Real.sin (45 * π / 180) + (1/3)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l123_12315


namespace NUMINAMATH_CALUDE_ice_cream_cones_l123_12304

theorem ice_cream_cones (cost_per_cone : ℕ) (total_cost : ℕ) (h1 : cost_per_cone = 99) (h2 : total_cost = 198) :
  total_cost / cost_per_cone = 2 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cones_l123_12304


namespace NUMINAMATH_CALUDE_total_savings_calculation_l123_12337

-- Define the original prices and discount rates
def chlorine_price : ℝ := 10
def chlorine_discount : ℝ := 0.20
def soap_price : ℝ := 16
def soap_discount : ℝ := 0.25

-- Define the quantities
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5

-- Theorem statement
theorem total_savings_calculation :
  let chlorine_savings := chlorine_price * chlorine_discount * chlorine_quantity
  let soap_savings := soap_price * soap_discount * soap_quantity
  chlorine_savings + soap_savings = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_calculation_l123_12337


namespace NUMINAMATH_CALUDE_club_count_l123_12371

theorem club_count (total : ℕ) (black : ℕ) (red : ℕ) (spades : ℕ) (diamonds : ℕ) (hearts : ℕ) (clubs : ℕ) :
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end NUMINAMATH_CALUDE_club_count_l123_12371


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l123_12378

theorem quadratic_function_minimum (a b c : ℝ) (x₀ : ℝ) (ha : a > 0) (hx₀ : 2 * a * x₀ + b = 0) :
  ¬ (∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l123_12378


namespace NUMINAMATH_CALUDE_complex_solution_l123_12348

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- State the theorem
theorem complex_solution :
  ∃ z : ℂ, det 2 (-1) z (z * Complex.I) = 1 + Complex.I ∧ z = 3/5 - 1/5 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_solution_l123_12348


namespace NUMINAMATH_CALUDE_pencils_given_eq_difference_l123_12320

/-- The number of pencils Jesse gave to Joshua -/
def pencils_given : ℕ := sorry

/-- The initial number of pencils Jesse had -/
def initial_pencils : ℕ := 78

/-- The remaining number of pencils Jesse has -/
def remaining_pencils : ℕ := 34

/-- Theorem stating that the number of pencils given is equal to the difference between initial and remaining pencils -/
theorem pencils_given_eq_difference : 
  pencils_given = initial_pencils - remaining_pencils := by sorry

end NUMINAMATH_CALUDE_pencils_given_eq_difference_l123_12320


namespace NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l123_12306

/-- A geometric sequence with positive integer terms -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (r : ℚ), r > 0 ∧ ∀ n, a (n + 1) = a n * ⌊r⌋

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℕ) (b : ℕ → ℤ)
  (h_geo : geometric_sequence a)
  (h_arith : arithmetic_sequence b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_inequality_l123_12306


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l123_12305

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  a 3 * a 7 = 8 → a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l123_12305


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l123_12329

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters_in_mathematics : ℕ := 8
  let probability : ℚ := unique_letters_in_mathematics / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l123_12329


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l123_12319

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 2 = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ x + (a - 1) * y + 1 = 0

/-- The lines are parallel -/
def areParallel (lines : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), lines.line1 (x, y) ↔ lines.line2 (k * x + lines.a, k * y + 2))

/-- The main theorem -/
theorem parallel_iff_a_eq_neg_one (lines : TwoLines) :
  areParallel lines ↔ lines.a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l123_12319


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l123_12352

theorem triangle_inequality_variant (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + 3*y^2) + Real.sqrt (x^2 + z^2 + x*z) > Real.sqrt (z^2 + 3*y^2 + 3*y*z) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l123_12352


namespace NUMINAMATH_CALUDE_unique_two_digit_multiple_l123_12393

theorem unique_two_digit_multiple : ∃! s : ℕ, 
  10 ≤ s ∧ s < 100 ∧ (13 * s) % 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_multiple_l123_12393


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_multiple_roots_l123_12321

def has_integral_multiple_roots (m : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ (10 * x^2 - m * x + 360 = 0) ∧ 
             (10 * y^2 - m * y + 360 = 0) ∧
             (x ∣ y ∨ y ∣ x)

theorem smallest_m_for_integral_multiple_roots :
  (has_integral_multiple_roots 120) ∧
  (∀ m : ℕ, m > 0 ∧ m < 120 → ¬(has_integral_multiple_roots m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_multiple_roots_l123_12321


namespace NUMINAMATH_CALUDE_fraction_doubling_l123_12399

theorem fraction_doubling (x y : ℝ) (h : x + y ≠ 0) :
  (2*x)^2 / (2*x + 2*y) = 2 * (x^2 / (x + y)) :=
sorry

end NUMINAMATH_CALUDE_fraction_doubling_l123_12399


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l123_12385

/-- Represents the number of available colors for triangles -/
def total_colors : ℕ := 8

/-- Represents the number of colors available for corner triangles -/
def corner_colors : ℕ := total_colors - 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def distinguishable_triangles : ℕ :=
  corner_colors +  -- All corners same color
  (corner_colors * (corner_colors - 1)) +  -- Two corners same color
  choose corner_colors 3  -- All corners different colors

theorem distinguishable_triangles_count :
  distinguishable_triangles = 84 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l123_12385


namespace NUMINAMATH_CALUDE_equalize_buses_l123_12388

def students_first_bus : ℕ := 57
def students_second_bus : ℕ := 31

def students_to_move : ℕ := 13

theorem equalize_buses :
  (students_first_bus - students_to_move = students_second_bus + students_to_move) ∧
  (students_first_bus - students_to_move > 0) ∧
  (students_second_bus + students_to_move > 0) :=
by sorry

end NUMINAMATH_CALUDE_equalize_buses_l123_12388


namespace NUMINAMATH_CALUDE_green_chips_count_l123_12331

theorem green_chips_count (total : ℕ) (blue_fraction : ℚ) (red : ℕ) : 
  total = 60 →
  blue_fraction = 1 / 6 →
  red = 34 →
  (total : ℚ) * blue_fraction + red + (total - (total : ℚ) * blue_fraction - red) = total →
  total - (total : ℚ) * blue_fraction - red = 16 :=
by sorry

end NUMINAMATH_CALUDE_green_chips_count_l123_12331


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l123_12308

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions -/
def perimeter (d : Dimensions) : ℕ := 2 * (d.length + d.width)

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tiling pattern of the large rectangle -/
structure TilingPattern where
  inner : Dimensions
  redTiles : ℕ

theorem large_rectangle_perimeter 
  (pattern : TilingPattern)
  (h1 : pattern.redTiles = 2900) :
  ∃ (large : Dimensions), 
    area large = area pattern.inner + 2900 + 2 * area { length := pattern.inner.length + 20, width := pattern.inner.width + 20 } ∧ 
    perimeter large = 350 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l123_12308


namespace NUMINAMATH_CALUDE_fence_poles_count_l123_12351

def side_length : ℝ := 150
def pole_spacing : ℝ := 30

theorem fence_poles_count :
  let perimeter := 4 * side_length
  let poles_count := perimeter / pole_spacing
  poles_count = 20 := by sorry

end NUMINAMATH_CALUDE_fence_poles_count_l123_12351


namespace NUMINAMATH_CALUDE_sum_possibilities_l123_12326

theorem sum_possibilities (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  b / c = 1 →
  4^a + 3^b + 2^c + 1^d = 10 ∨ 
  4^a + 3^b + 2^c + 1^d = 22 ∨ 
  4^a + 3^b + 2^c + 1^d = 70 :=
by sorry

end NUMINAMATH_CALUDE_sum_possibilities_l123_12326


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l123_12395

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) :
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l123_12395


namespace NUMINAMATH_CALUDE_sum_ratio_equals_55_49_l123_12356

theorem sum_ratio_equals_55_49 : 
  let sum_n (n : ℕ) := n * (n + 1) / 2
  let sum_squares (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6
  let sum_cubes (n : ℕ) := (sum_n n) ^ 2
  (sum_n 10 * sum_cubes 10) / (sum_squares 10) ^ 2 = 55 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_55_49_l123_12356


namespace NUMINAMATH_CALUDE_fib_150_mod_7_l123_12359

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_7 : fib 150 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_7_l123_12359


namespace NUMINAMATH_CALUDE_part_one_part_two_l123_12346

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a|

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x, f a x ≥ |2*x + 3| ↔ x ∈ Set.Icc (-3) (-1)) →
  a = 0 :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, f a x + |x - a| ≥ a^2 - 2*a) →
  0 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l123_12346


namespace NUMINAMATH_CALUDE_charles_discount_l123_12313

/-- The discount given to a customer, given the total cost before discount and the amount paid after discount. -/
def discount (total_cost : ℝ) (amount_paid : ℝ) : ℝ :=
  total_cost - amount_paid

/-- Theorem: The discount given to Charles is $2. -/
theorem charles_discount : discount 45 43 = 2 := by
  sorry

end NUMINAMATH_CALUDE_charles_discount_l123_12313


namespace NUMINAMATH_CALUDE_total_money_calculation_l123_12316

theorem total_money_calculation (total_notes : ℕ) 
  (denominations : Fin 3 → ℕ) 
  (h1 : total_notes = 75) 
  (h2 : denominations 0 = 1 ∧ denominations 1 = 5 ∧ denominations 2 = 10) : 
  (total_notes / 3) * (denominations 0 + denominations 1 + denominations 2) = 400 :=
by
  sorry

#check total_money_calculation

end NUMINAMATH_CALUDE_total_money_calculation_l123_12316


namespace NUMINAMATH_CALUDE_complex_equation_solution_l123_12350

theorem complex_equation_solution (z : ℂ) : z + z * Complex.I = 1 + 5 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l123_12350


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l123_12377

theorem complex_number_real_condition (a : ℝ) :
  (∃ (z : ℂ), z = (a + 1) + (a^2 - 1) * I ∧ z.im = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l123_12377


namespace NUMINAMATH_CALUDE_monotonic_quadratic_condition_l123_12382

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_condition :
  ∀ a : ℝ, IsMonotonic (f a) 2 3 ↔ (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_condition_l123_12382


namespace NUMINAMATH_CALUDE_tangent_segments_area_l123_12334

theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 4) :
  let inner_radius := r
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * inner_radius^2) = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l123_12334


namespace NUMINAMATH_CALUDE_board_cut_theorem_l123_12327

theorem board_cut_theorem (total_length : ℝ) (x : ℝ) 
  (h1 : total_length = 120)
  (h2 : x = 1.5) : 
  let shorter_piece := total_length / (1 + (2 * x + 1/3))
  let longer_piece := shorter_piece * (2 * x + 1/3)
  longer_piece = 92 + 4/13 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l123_12327


namespace NUMINAMATH_CALUDE_blueberry_picking_l123_12374

theorem blueberry_picking (annie kathryn ben : ℕ) 
  (h1 : kathryn = annie + 2)
  (h2 : ben = kathryn - 3)
  (h3 : annie + kathryn + ben = 25) :
  annie = 8 := by
sorry

end NUMINAMATH_CALUDE_blueberry_picking_l123_12374


namespace NUMINAMATH_CALUDE_power_equation_solution_l123_12369

theorem power_equation_solution (n : ℕ) : 2^n = 2 * 4^2 * 16^3 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l123_12369


namespace NUMINAMATH_CALUDE_jo_alan_sum_equal_l123_12328

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def alan_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem jo_alan_sum_equal :
  jo_sum 120 = alan_sum 120 :=
sorry

end NUMINAMATH_CALUDE_jo_alan_sum_equal_l123_12328


namespace NUMINAMATH_CALUDE_prob_even_sum_is_31_66_l123_12345

/-- A set of twelve prime numbers including two even primes -/
def prime_set : Finset ℕ := sorry

/-- The number of prime numbers in the set -/
def n : ℕ := 12

/-- The number of even prime numbers in the set -/
def even_primes : ℕ := 2

/-- The number of primes to be selected -/
def k : ℕ := 5

/-- Predicate to check if a set of natural numbers has an even sum -/
def has_even_sum (s : Finset ℕ) : Prop := Even (s.sum id)

/-- The probability of selecting k primes from prime_set with two even primes such that their sum is even -/
def prob_even_sum : ℚ := sorry

theorem prob_even_sum_is_31_66 : prob_even_sum = 31 / 66 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_31_66_l123_12345


namespace NUMINAMATH_CALUDE_parallelogram_area_l123_12386

theorem parallelogram_area (base height : ℝ) (h1 : base = 26) (h2 : height = 16) : 
  base * height = 416 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l123_12386


namespace NUMINAMATH_CALUDE_cauliflower_increase_l123_12301

theorem cauliflower_increase (n : ℕ) (h : n^2 = 12544) : n^2 - (n-1)^2 = 223 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_increase_l123_12301


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l123_12344

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l123_12344


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l123_12342

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola a b) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- Predicate to check if four points form a parallelogram -/
def is_parallelogram (p q r s : ℝ × ℝ) : Prop := sorry

/-- The area of a quadrilateral given by four points -/
def quadrilateral_area (p q r s : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity_theorem (a b c : ℝ) (h : Hyperbola a b) 
  (m n : ℝ × ℝ) (hm : on_hyperbola h m) (hn : on_hyperbola h n)
  (hpara : is_parallelogram (0, 0) (right_focus h) m n)
  (harea : quadrilateral_area (0, 0) (right_focus h) m n = Real.sqrt 3 * b * c) :
  eccentricity h = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l123_12342


namespace NUMINAMATH_CALUDE_not_all_isosceles_congruent_l123_12318

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

/-- Congruence of triangles -/
def are_congruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side1 = t2.side1 ∧ t1.side2 = t2.side2 ∧ t1.base = t2.base

/-- Theorem: Not all isosceles triangles are congruent -/
theorem not_all_isosceles_congruent : 
  ∃ t1 t2 : IsoscelesTriangle, ¬(are_congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_isosceles_congruent_l123_12318


namespace NUMINAMATH_CALUDE_rectangle_perimeter_change_l123_12368

theorem rectangle_perimeter_change (a b : ℝ) (h : 2 * (1.3 * a + 0.8 * b) = 2 * (a + b)) :
  2 * (0.8 * a + 1.3 * b) = 1.1 * (2 * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_change_l123_12368
