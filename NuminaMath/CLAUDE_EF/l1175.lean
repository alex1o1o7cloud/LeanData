import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1175_117540

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ ¬(x^2 - 4*a*x + 3*a^2 < 0)) →
  a < 0 →
  a ∈ Set.Iic (-4 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1175_117540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_slope_constant_l1175_117504

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_positive : 0 < r

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Point on an ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Tangent lines from a point to a circle -/
def tangent_lines (c : Circle) (px py : ℝ) : Prop :=
  (px - c.h)^2 + (py - c.k)^2 > c.r^2

/-- Theorem statement -/
theorem ellipse_tangent_slope_constant 
  (e : Ellipse) 
  (c : Circle) :
  eccentricity e = 1/2 →
  on_ellipse e 1 (3/2) →
  c.h = 1 ∧ c.k = 0 ∧ c.r < 3/2 →
  tangent_lines c 1 (3/2) →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_ellipse e x₁ y₁ ∧
    on_ellipse e x₂ y₂ ∧
    (y₁ - 3/2) / (x₁ - 1) = -(y₂ - 3/2) / (x₂ - 1) ∧
    (y₂ - y₁) / (x₂ - x₁) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_slope_constant_l1175_117504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_is_ten_l1175_117589

/-- The sum of positive integer values of m for which the intersection of y = x and y = mx - 4 has positive integer coordinates --/
def intersection_sum : ℕ := 
  (Finset.filter (fun m : ℕ => 
    let x : ℚ := 4 / (m - 1)
    x > 0 ∧ x.isInt ∧ m > 1) (Finset.range 100)).sum id

/-- The theorem stating that the sum is 10 --/
theorem intersection_sum_is_ten : intersection_sum = 10 := by
  sorry

#eval intersection_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_is_ten_l1175_117589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_complex_numbers_l1175_117518

-- Define the complex numbers
noncomputable def z1 : ℂ := 1 + Complex.I * Real.sqrt 3
noncomputable def z2 : ℂ := -Real.sqrt 3 + Complex.I

-- Define the angle between two complex numbers
noncomputable def angle (a b : ℂ) : ℝ := Real.arccos ((a.re * b.re + a.im * b.im) / (Complex.abs a * Complex.abs b))

-- Theorem statement
theorem angle_between_complex_numbers : angle z1 z2 = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_complex_numbers_l1175_117518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_count_l1175_117515

/-- Given an arithmetic sequence starting with 6, ending with 202, and having a common difference of 4,
    the number of terms in the sequence that are divisible by 4 is 49. -/
theorem divisible_by_four_count (seq : List ℕ) : 
  (seq.head? = some 6) →
  (seq.get? (seq.length - 1) = some 202) →
  (∀ i, i < seq.length - 1 → seq.get? (i+1) = (seq.get? i).map (· + 4)) →
  (seq.filter (λ x => x % 4 = 0)).length = 49 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_count_l1175_117515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_sum_l1175_117535

noncomputable def log10 (n : ℕ) : ℝ := Real.log n / Real.log 10

theorem prime_factors_sum (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (eq1 : log10 x + 2 * log10 (Nat.gcd x y) = 12)
  (eq2 : log10 y + 2 * log10 (Nat.lcm x y) = 18) :
  (Nat.factors x).length + (Nat.factors y).length = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_sum_l1175_117535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_square_l1175_117519

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rhombus -/
structure Rhombus where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a square -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Constructs squares on the sides of a rhombus -/
noncomputable def constructSquaresOnRhombus (r : Rhombus) : Square :=
  sorry

/-- Predicate to check if four points form a square -/
def isSquare (P Q R S : Point) : Prop :=
  sorry

/-- Theorem: The centers of squares constructed on the sides of a rhombus form a square -/
theorem centers_form_square (r : Rhombus) : 
  let s := constructSquaresOnRhombus r
  isSquare s.P s.Q s.R s.S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_square_l1175_117519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l1175_117538

-- Define a triangle inscribed in a unit circle
structure InscribedTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  inscribed : A + B + C = π
  unit_circle : ∀ θ, 0 ≤ θ ∧ θ ≤ 2*π → ∃ (x y : ℝ), x^2 + y^2 = 1

-- Define the condition (1+tan A)(1+tan B) = 2
def condition (t : InscribedTriangle) : Prop :=
  (1 + Real.tan t.A) * (1 + Real.tan t.B) = 2

-- Define the area of the triangle
noncomputable def area (t : InscribedTriangle) : ℝ :=
  (1/2) * Real.sin t.A * Real.sin t.B * Real.sin t.C

-- Theorem statement
theorem inscribed_triangle_properties (t : InscribedTriangle) 
  (h : condition t) : 
  t.C = (3*π)/4 ∧ 
  ∀ (s : InscribedTriangle), condition s → area s ≤ (Real.sqrt 2 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_properties_l1175_117538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_not_24_pow_2022_l1175_117580

-- Define the sequence type
def Sequence := List Nat

-- Define the property of adjacent elements in the sequence
def ValidSequence (s : Sequence) : Prop :=
  ∀ i, i < s.length - 1 →
    (s.get! (i + 1) = 9 * s.get! i ∨ s.get! i = 2 * s.get! (i + 1))

-- Define the theorem
theorem sequence_sum_not_24_pow_2022 (s : Sequence) 
  (h1 : s.length = 200) 
  (h2 : ValidSequence s) : 
  s.sum ≠ 24^2022 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_not_24_pow_2022_l1175_117580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_complete_task_is_eight_l1175_117579

/-- The number of days needed to complete a task given the following conditions:
  * There are 5 workers
  * 735 total parts need to be processed
  * 135 parts have been processed in the first two days
  * One worker took one day off during these two days
  * Each worker processes the same number of parts per day
  * No more days off going forward
-/
def days_to_complete_task : ℕ :=
  let total_parts : ℕ := 735
  let processed_parts : ℕ := 135
  let num_workers : ℕ := 5
  let initial_days : ℕ := 2
  let days_off : ℕ := 1
  
  let remaining_parts : ℕ := total_parts - processed_parts
  let worker_days : ℕ := num_workers * initial_days - days_off
  let parts_per_worker_day : ℕ := processed_parts / worker_days
  let daily_output : ℕ := num_workers * parts_per_worker_day
  
  remaining_parts / daily_output

#eval days_to_complete_task

theorem days_to_complete_task_is_eight :
  days_to_complete_task = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_complete_task_is_eight_l1175_117579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l1175_117594

theorem tan_2alpha_value (α : Real) 
  (h1 : Real.sin (2 * α) = -Real.sqrt 3 * Real.cos α) 
  (h2 : α ∈ Set.Ioo (-Real.pi/2) 0) : 
  Real.tan (2 * α) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_value_l1175_117594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_intersection_l1175_117571

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + x

-- State the theorem
theorem extreme_points_intersection (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ →
  (∃ (y₁ y₂ : ℝ), y₁ = f a x₁ ∧ y₂ = f a x₂) →
  (∃ (m b : ℝ), ∀ (x : ℝ), m * x + b = f a x → x = 0) →
  (∃ (x : ℝ), x ≠ x₁ ∧ x ≠ x₂ ∧ (deriv (f a)) x = 0) →
  a = Real.sqrt 6 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_intersection_l1175_117571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1175_117599

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ) - Real.sqrt 3 * Real.cos (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π / 2) 
  (h_symmetry : ∀ x, f ω φ x = f ω φ (-x) ∧ f ω φ (x + π/2) = f ω φ (-x + π/2)) :
  (∃ T > 0, (∀ x, f ω φ (x + T) = f ω φ x) ∧ 
   ∀ T' > 0, (∀ x, f ω φ (x + T') = f ω φ x) → T ≤ T') ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f ω φ x < f ω φ y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1175_117599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l1175_117545

noncomputable def recurring_decimal : ℚ := 2.5081081081081

theorem recurring_decimal_fraction :
  ∃ (m n : ℕ), 
    (recurring_decimal : ℚ) = m / n ∧ 
    m = 61727 ∧ 
    n = 24690 ∧ 
    Nat.Coprime m n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l1175_117545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_properties_l1175_117508

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ)

noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ (x - π/3)

theorem periodic_sine_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 < φ ∧ φ < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_value : f ω φ (π/4) = 1) :
  (ω = 2 ∧ φ = 0) ∧
  (∀ x, f ω φ x + g ω φ x ≤ 1) ∧
  (∀ k : ℤ, f ω φ (k * π + 5*π/12) + g ω φ (k * π + 5*π/12) = 1) := by
  sorry

#check periodic_sine_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_properties_l1175_117508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1175_117587

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  2 * sqrt 3 * (cos x)^2 - 2 * (sin (π/4 - x))^2 - sqrt 3

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonically increasing intervals
  (∀ k : ℤ, ∀ x y : ℝ,
    k * π - 5 * π / 12 ≤ x ∧ x < y ∧ y ≤ k * π + π / 12 →
    f x < f y) ∧
  -- Part 2: Maximum value on [0, π/6]
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 6 → f x ≤ 1) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 6 ∧ f x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1175_117587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1175_117595

-- Define the arithmetic sequence and its sum
def arithmeticSequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sumArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_sum : sumArithmeticSequence a 6 > sumArithmeticSequence a 7 ∧
           sumArithmeticSequence a 7 > sumArithmeticSequence a 5) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d < 0) ∧
  sumArithmeticSequence a 11 > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1175_117595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_sum_l1175_117590

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  Real.sqrt 3 * Real.cos x = 1 / Real.tan y ∧
  2 * Real.cos y = Real.tan z ∧
  Real.cos z = 2 / Real.tan x

-- State the theorem
theorem min_sin_sum (x y z : ℝ) (h : system x y z) :
  ∃ (m : ℝ), m = -7 * Real.sqrt 2 / 6 ∧
  Real.sin x + Real.sin z ≥ m ∧
  ∃ (x' y' z' : ℝ), system x' y' z' ∧ Real.sin x' + Real.sin z' = m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin_sum_l1175_117590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_l1175_117575

theorem second_day_distance (total_distance : ℝ) (num_days : ℕ) (ratio : ℝ) :
  total_distance = 378 →
  num_days = 6 →
  ratio = 1 / 2 →
  let sequence := λ n ↦ (total_distance * (1 - ratio^num_days) / (1 - ratio)) * ratio^(n-1)
  sequence 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_day_distance_l1175_117575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fraction_sum_l1175_117541

/-- Given a natural number n > 1, find the maximum value of a/b + c/d where a, b, c, d are natural numbers,
    a/b + c/d < 1, and a + c ≤ n -/
theorem max_fraction_sum (n : ℕ) (hn : n > 1) :
  let max_value := 1 - 1 / ((⌊(2 * n / 3 : ℚ) + 7 / 6⌋ : ℚ) * ((n : ℚ) - ⌊(2 * n / 3 : ℚ) + 1 / 6⌋) + 1)
  ∀ (a b c d : ℕ), (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) < 1 → a + c ≤ n →
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) ≤ max_value :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fraction_sum_l1175_117541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_approx_l1175_117551

/-- Triangle with inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of DP, where P is the tangent point on DE -/
  dp : ℝ
  /-- Length of PE, where P is the tangent point on DE -/
  pe : ℝ

/-- Calculate the perimeter of a triangle with an inscribed circle -/
noncomputable def perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  2 * (t.dp + t.pe + t.r * (t.dp + t.pe) / (t.dp * t.pe))

/-- Theorem: The perimeter of the given triangle is approximately 81.018 -/
theorem triangle_perimeter_approx (t : TriangleWithInscribedCircle)
    (h1 : t.r = 15)
    (h2 : t.dp = 19)
    (h3 : t.pe = 31) :
    ∃ ε > 0, |perimeter t - 81.018| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_approx_l1175_117551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l1175_117583

/-- Given vectors a and b, find λ such that a + 2b is parallel to 3a + λb -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 3)) (h2 : b = (2, 1)) :
  ∃ l : ℝ, l = 6 ∧ ∃ k : ℝ, k • (a + 2 • b) = 3 • a + l • b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l1175_117583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l1175_117517

/-- The equation of a hyperbola in the form ax^2 + bx + cy^2 + dy + e = 0 -/
structure HyperbolaEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a conic section -/
structure ConicCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the hyperbola given by 9x^2 - 54x - 36y^2 + 288y + 72 = 0 is (3, 4) -/
theorem hyperbola_center (h : HyperbolaEquation) 
  (h_eq : h = ⟨9, -54, -36, 288, 72⟩) : 
  ConicCenter.mk 3 4 = ⟨3, 4⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l1175_117517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l1175_117543

/-- Calculates the percentage increase between two amounts -/
noncomputable def percentageIncrease (originalAmount newAmount : ℝ) : ℝ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem john_salary_increase :
  let originalSalary : ℝ := 65
  let newSalary : ℝ := 72
  let increase := percentageIncrease originalSalary newSalary
  ∃ ε > 0, |increase - 10.77| < ε :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_salary_increase_l1175_117543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangeMultipleOfSeven_l1175_117501

def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem arrangeMultipleOfSeven (n : ℕ) (h : n ≥ 2) :
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    7 ∣ (fromDigits (List.ofFn (λ i => (p i).val + 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangeMultipleOfSeven_l1175_117501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1175_117593

theorem solve_exponential_equation :
  ∃ y : ℝ, 5 * (2:ℝ)^y = 160 ∧ y = 5 := by
  use 5
  constructor
  · norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1175_117593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l1175_117548

/-- Represents the bus driver's pay structure and work details -/
structure BusDriverPay where
  regularRate : ℚ
  regularHours : ℚ
  totalHours : ℚ
  totalCompensation : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
noncomputable def overtimeRateIncrease (pay : BusDriverPay) : ℚ :=
  let overtimeHours := pay.totalHours - pay.regularHours
  let regularEarnings := pay.regularRate * pay.regularHours
  let overtimeEarnings := pay.totalCompensation - regularEarnings
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - pay.regularRate) / pay.regularRate) * 100

/-- Theorem stating the overtime rate increase for the given scenario -/
theorem bus_driver_overtime_rate_increase :
  let pay := BusDriverPay.mk 16 40 57 1116
  overtimeRateIncrease pay = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_overtime_rate_increase_l1175_117548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l1175_117576

noncomputable def a : ℝ × ℝ := (-1/3, -2/3)

theorem perpendicular_unit_vectors :
  let b₁ : ℝ × ℝ := (2*Real.sqrt 5/5, -Real.sqrt 5/5)
  let b₂ : ℝ × ℝ := (-2*Real.sqrt 5/5, Real.sqrt 5/5)
  (∀ b : ℝ × ℝ, (b.1^2 + b.2^2 = 1 ∧ b.1 * a.1 + b.2 * a.2 = 0) → (b = b₁ ∨ b = b₂)) ∧
  (b₁.1^2 + b₁.2^2 = 1 ∧ b₁.1 * a.1 + b₁.2 * a.2 = 0) ∧
  (b₂.1^2 + b₂.2^2 = 1 ∧ b₂.1 * a.1 + b₂.2 * a.2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vectors_l1175_117576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1175_117597

theorem triangle_side_ratio_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle is acute
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →  -- Sine law
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →  -- Cosine law
  Real.sqrt 3 * Real.sin C * Real.cos C + Real.cos C^2 = 1 →  -- Given condition
  3 ≤ (a^2 + b^2 + c^2) / (a * b) ∧ (a^2 + b^2 + c^2) / (a * b) < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_range_l1175_117597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1175_117584

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x - 5)

-- Define the domain of the function
def domain_f : Set ℝ := {x | x ≠ 5}

-- Theorem statement
theorem domain_of_f : 
  domain_f = Set.Iio 5 ∪ Set.Ioi 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1175_117584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_difference_of_squares_l1175_117586

theorem power_of_two_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 1) 
  (h2 : x - y = 3) : 
  (2 : ℝ)^(x^2 - y^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_difference_of_squares_l1175_117586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1175_117591

/-- The function g(x) defined as the sum of cubes of arctan and arccot -/
noncomputable def g (x : ℝ) : ℝ := (Real.arctan x)^3 + (Real.arctan (1/x))^3

/-- The theorem stating the range of g(x) -/
theorem range_of_g :
  Set.range g = Set.Icc (Real.pi^3 / 32) (3 * Real.pi^3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1175_117591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l1175_117503

/-- The minimum positive period of y = tan(3x - π/4) is π/3 -/
theorem tan_period (x : ℝ) : 
  ∃ T : ℝ, T > 0 ∧ T = π / 3 ∧ 
  (∀ t : ℝ, Real.tan (3 * (x + T) - π / 4) = Real.tan (3 * x - π / 4)) ∧
  (∀ S : ℝ, S > 0 → S < T → 
    ∃ s : ℝ, Real.tan (3 * (x + S) - π / 4) ≠ Real.tan (3 * x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_l1175_117503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pieces_theorem_six_pieces_theorem_l1175_117506

-- Define the board as a 4x4 grid
def Board : Type := Fin 4 → Fin 4 → Bool

-- Define a valid configuration of pieces
def ValidConfig (n : Nat) (board : Board) : Prop :=
  (∃ (pieces : Finset (Fin 4 × Fin 4)), pieces.card = n ∧
    ∀ (p : Fin 4 × Fin 4), p ∈ pieces ↔ board p.1 p.2 = true) ∧
  ∀ (i j : Fin 4), board i j → ∀ (i' j' : Fin 4), board i' j' → (i = i' ∧ j = j') ∨ (i ≠ i' ∨ j ≠ j')

-- Define the removal of rows and columns
def RemoveRowsColumns (rows columns : Finset (Fin 4)) (board : Board) : Board :=
  λ i j ↦ if i ∉ rows ∧ j ∉ columns then board i j else false

-- Theorem for part (a)
theorem seven_pieces_theorem :
  ∃ (board : Board), ValidConfig 7 board ∧
    ∀ (rows columns : Finset (Fin 4)),
      rows.card = 2 ∧ columns.card = 2 →
      ∃ (i j : Fin 4), RemoveRowsColumns rows columns board i j = true := by
  sorry

-- Theorem for part (b)
theorem six_pieces_theorem :
  ∀ (board : Board), ValidConfig 6 board →
    ∃ (rows columns : Finset (Fin 4)),
      rows.card = 2 ∧ columns.card = 2 ∧
      ∀ (i j : Fin 4), RemoveRowsColumns rows columns board i j = false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pieces_theorem_six_pieces_theorem_l1175_117506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_hundred_sixty_thousand_scientific_notation_l1175_117585

-- Define scientific notation
noncomputable def scientific_notation (n : ℝ) (m : ℤ) : ℝ := n * (10 : ℝ) ^ m

-- Theorem statement
theorem six_hundred_sixty_thousand_scientific_notation :
  (660000 : ℝ) = scientific_notation 6.6 5 := by
  -- Expand the definition of scientific_notation
  unfold scientific_notation
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_hundred_sixty_thousand_scientific_notation_l1175_117585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_long_sides_l1175_117572

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  sides : Fin 6 → ℝ
  convex : True -- Placeholder for convexity property
  two_lengths : ∃ (a b : ℝ), (∀ i, sides i = a ∨ sides i = b) ∧ a ≠ b

/-- The perimeter of a hexagon -/
def perimeter (h : ConvexHexagon) : ℝ :=
  (Finset.sum Finset.univ (λ i => h.sides i))

theorem three_long_sides
  (h : ConvexHexagon)
  (ab_length : h.sides 0 = 5)
  (bc_length : h.sides 1 = 7)
  (perim : perimeter h = 40) :
  (Finset.sum Finset.univ (λ i => if h.sides i = 7 then 1 else 0) : ℕ) = 3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_long_sides_l1175_117572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_M_is_four_l1175_117525

def sequence_a : ℕ → ℚ
  | 0 => 5/2
  | n + 1 => (sequence_a n)^2 - 2

noncomputable def M : ℤ := ⌊sequence_a 2022 + 1/2⌋

theorem last_digit_of_M_is_four :
  M % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_M_is_four_l1175_117525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_l1175_117527

theorem smallest_factor_difference (n : ℕ) (hn : n = 1794) :
  ∃ (a b : ℕ), a * b = n ∧ a > 0 ∧ b > 0 ∧
  ∀ (c d : ℕ), c * d = n → c > 0 → d > 0 → (Int.natAbs (a - b) : ℤ) ≤ Int.natAbs (c - d) ∧ Int.natAbs (a - b) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_difference_l1175_117527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_average_distance_l1175_117552

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculate the distance from a point to a vertical line -/
noncomputable def distToVertical (p : Point) (x : ℝ) : ℝ :=
  |p.x - x|

/-- Calculate the distance from a point to a horizontal line -/
noncomputable def distToHorizontal (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- Calculate the endpoint after moving along the diagonal -/
noncomputable def moveAlongDiagonal (rect : Rectangle) (distance : ℝ) : Point :=
  let diag := Real.sqrt (rect.length^2 + rect.width^2)
  { x := distance * rect.length / diag,
    y := distance * rect.width / diag }

/-- Calculate the endpoint after moving perpendicular to the diagonal -/
def movePerpendicular (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance,
    y := p.y }

/-- Main theorem -/
theorem rabbit_average_distance (rect : Rectangle) (diagonalDist : ℝ) (perpDist : ℝ) :
  rect.length = 12 →
  rect.width = 8 →
  diagonalDist = 8 →
  perpDist = 3 →
  let p1 := moveAlongDiagonal rect diagonalDist
  let p2 := movePerpendicular p1 perpDist
  let avgDist := (distToVertical p2 0 + distToVertical p2 rect.length +
                  distToHorizontal p2 0 + distToHorizontal p2 rect.width) / 4
  avgDist = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_average_distance_l1175_117552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_through_P_at_distance_2_line_through_P_at_max_distance_max_distance_from_origin_no_line_at_distance_6_l1175_117573

/-- Point P with coordinates (2, -1) -/
def P : ℝ × ℝ := (2, -1)

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

/-- Distance from the origin to a line -/
noncomputable def distanceOriginToLine (a b c : ℝ) : ℝ :=
  abs c / Real.sqrt (a^2 + b^2)

theorem lines_through_P_at_distance_2 :
  (∀ x y, x = 2 ∨ 3*x - 4*y - 10 = 0) ↔
  (distanceOriginToLine 1 0 (-2) = 2 ∧ distancePointToLine P 1 0 (-2) = 0) ∨
  (distanceOriginToLine 3 (-4) (-10) = 2 ∧ distancePointToLine P 3 (-4) (-10) = 0) := by
  sorry

theorem line_through_P_at_max_distance :
  (∀ x y, 2*x - y - 5 = 0) ↔
  (distancePointToLine P 2 (-1) (-5) = 0 ∧
   ∀ a b c, distancePointToLine P a b c = 0 →
            distanceOriginToLine a b c ≤ distanceOriginToLine 2 (-1) (-5)) := by
  sorry

theorem max_distance_from_origin :
  (∀ a b c, distancePointToLine P a b c = 0 →
            distanceOriginToLine a b c ≤ Real.sqrt 5) ∧
  (∃ a b c, distancePointToLine P a b c = 0 ∧
            distanceOriginToLine a b c = Real.sqrt 5) := by
  sorry

theorem no_line_at_distance_6 :
  ¬ ∃ a b c, distancePointToLine P a b c = 0 ∧ distanceOriginToLine a b c = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_through_P_at_distance_2_line_through_P_at_max_distance_max_distance_from_origin_no_line_at_distance_6_l1175_117573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1175_117511

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (1 - Real.sin x ^ 2)) / Real.cos x + (Real.sqrt (1 - Real.cos x ^ 2)) / Real.sin x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 ∧ f x = y) ↔ y ∈ ({-2, 0, 2} : Set ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1175_117511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_a_result_a_continued_fraction_b_result_b_continued_fraction_c_result_c_l1175_117553

noncomputable def continuedFraction (a : ℕ) (b : List ℕ) : ℝ := sorry

theorem continued_fraction_a (α : ℝ) : 
  (α = 10 + 1 / (1 + 2 / (2 + 1 / (1 + 1 / α)))) → α = 5 + Real.sqrt 33 :=
by sorry

theorem result_a : continuedFraction 5 [1,2,1,10] = Real.sqrt 33 :=
by sorry

theorem continued_fraction_b (α : ℝ) :
  (α = 10 + 1 / (1 + 4 / (4 + 1 / (1 + 1 / α)))) → α = 5 + Real.sqrt 34 :=
by sorry

theorem result_b : continuedFraction 5 [1,4,1,10] = Real.sqrt 34 :=
by sorry

theorem continued_fraction_c (α : ℝ) :
  (α = 3 + 1 / (1 + 1 / (1 + 1 / α))) → α = (1 + Real.sqrt 17) / 2 :=
by sorry

theorem result_c : continuedFraction 2 [1,1,3] = (1 + Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_a_result_a_continued_fraction_b_result_b_continued_fraction_c_result_c_l1175_117553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1175_117554

noncomputable def f (x : ℝ) := (Real.sqrt (-x^2 + 2*x + 15)) / (x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-3) 1 ∪ Set.Ioc 1 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1175_117554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_panacea_permutations_l1175_117505

def word : List Char := ['P', 'A', 'A', 'A', 'N', 'C', 'E']

def letter_count (c : Char) (l : List Char) : Nat :=
  l.filter (· = c) |>.length

def total_permutations (l : List Char) : Nat :=
  Nat.factorial l.length / (l.dedup.map (letter_count · l)).prod

def permutations_with_three_as_together (l : List Char) : Nat :=
  Nat.factorial (l.length - 2)

theorem panacea_permutations :
  total_permutations word - permutations_with_three_as_together word = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_panacea_permutations_l1175_117505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1175_117556

/-- Given points A, B, and C on a line, where AB = 4 and BC = 3, prove that AC is either 7 or 1 -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3) → (|C - A| = 7 ∨ |C - A| = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l1175_117556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l1175_117563

/-- Represents a point in the coordinate plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- The number of steps each object takes -/
def steps : ℕ := 10

/-- The starting point of object A -/
def start_A : Point := ⟨0, 0⟩

/-- The starting point of object B -/
def start_B : Point := ⟨10, 10⟩

/-- The probability of a single step for each object -/
def step_prob : ℚ := 1/2

/-- The total number of possible paths for each object -/
def total_paths : ℕ := 2^steps

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of A and B meeting -/
noncomputable def meet_probability : ℚ :=
  (Finset.sum (Finset.range (steps + 1)) (fun x => (binomial steps x * binomial steps (steps - x)))) / ((total_paths * total_paths) : ℚ)

/-- Theorem stating the probability of A and B meeting -/
theorem probability_of_meeting :
  meet_probability = 369512 / 1048576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_meeting_l1175_117563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l1175_117520

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 2 + Real.sqrt 6 ∧ x₂ = 2 - Real.sqrt 6) ∧
  (∀ x : ℝ, x = x₁ ∨ x = x₂ → 
    (3 * x^2) / (x - 2) - (3 * x + 4) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l1175_117520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_and_rational_numbers_l1175_117560

-- Define the numbers
noncomputable def pi_half : ℝ := Real.pi / 2
def frac_22_7 : ℚ := 22 / 7
def sqrt_4 : ℚ := 2
def decimal : ℚ := 101001000 / 1000000000

-- State the theorem
theorem irrational_and_rational_numbers :
  (¬ ∃ (q : ℚ), (↑q : ℝ) = pi_half) ∧
  (∃ (q : ℚ), (↑q : ℝ) = frac_22_7) ∧
  (∃ (q : ℚ), (↑q : ℝ) = sqrt_4) ∧
  (∃ (q : ℚ), (↑q : ℝ) = decimal) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_and_rational_numbers_l1175_117560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1175_117516

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π/4 - α)) : 
  Real.sin (2 * α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1175_117516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_form_with_finite_P_l1175_117514

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The set of primes that divide f(2023^j) for some j -/
def P (f : IntPolynomial) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ j : ℕ, ∃ k : ℤ, f.eval (↑(2023 ^ j)) = k * ↑p}

/-- Theorem stating the form of f(x) given the conditions -/
theorem polynomial_form_with_finite_P (f : IntPolynomial) 
    (h_finite : Set.Finite (P f)) :
    ∃ (c : ℤ) (n : ℕ), c ≠ 0 ∧ f = c • (Polynomial.X : IntPolynomial) ^ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_form_with_finite_P_l1175_117514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1175_117524

-- Define the triangle
structure Triangle (α : Type*) [LinearOrderedField α] where
  AB : α
  AC : α
  angle_A : α
  h_positive : 0 < AB ∧ 0 < AC
  h_right_angle : angle_A = 90

-- Define the area calculation function
def area {α : Type*} [LinearOrderedField α] (base height : α) : α :=
  (1/2) * base * height

-- Theorem statement
theorem right_triangle_area {α : Type*} [LinearOrderedField α] (t : Triangle α) 
  (h_AB : t.AB = 35)
  (h_AC : t.AC = 15) :
  area t.AB t.AC = 262.5 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1175_117524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l1175_117565

/-- Represents the speed of a particle at the nth mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  if n = 1 then 1 else 1 / (2 * (n - 1)^2)

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ :=
  if n = 1 then 1 else 1 / speed n

theorem particle_travel_time (n : ℕ) (h : n ≥ 2) :
  time n = 2 * (n - 1)^2 := by
  sorry

#check particle_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_travel_time_l1175_117565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_carol_optimal_is_half_l1175_117555

/-- Alice's choice is a random number in [0, 1] -/
def alice_choice : Set ℝ := Set.Icc 0 1

/-- Bob's choice is a random number in [0.4, 0.6] -/
def bob_choice : Set ℝ := Set.Icc 0.4 0.6

/-- Carol's winning probability given her choice c -/
noncomputable def carol_prob (c : ℝ) : ℝ :=
  if c < 0.4 then c
  else if c > 0.6 then 1 - c
  else -4 * c^2 + 4 * c - 1

/-- Carol's optimal choice maximizes her winning probability -/
theorem carol_optimal_choice :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 1 → carol_prob c ≥ carol_prob x :=
by sorry

/-- Carol's optimal choice is 0.5 -/
theorem carol_optimal_is_half :
  ∃ (c : ℝ), c = 0.5 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 1 → carol_prob c ≥ carol_prob x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carol_optimal_choice_carol_optimal_is_half_l1175_117555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_p_l1175_117530

theorem subset_sum_divisible_by_p (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (Finset.filter (λ A : Finset ℕ ↦
    A.card = p ∧ (A.sum id) % p = 0) (Finset.powerset S)).card =
  (1 / p : ℚ) * ((Nat.choose (2 * p) p : ℚ) - 2) + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_p_l1175_117530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1175_117562

/-- Represents the time (in hours) it takes Taimour to paint the fence alone -/
noncomputable def taimour_time : ℝ := 9

/-- Represents the time (in hours) it takes Jamshid to paint the fence alone -/
noncomputable def jamshid_time : ℝ := taimour_time / 2

/-- Represents the time (in hours) it takes Jamshid and Taimour to paint the fence together -/
noncomputable def combined_time : ℝ := 3

theorem taimour_paint_time :
  (1 / taimour_time + 1 / jamshid_time = 1 / combined_time) →
  taimour_time = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taimour_paint_time_l1175_117562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_at_half_time_l1175_117523

/-- The height of a candle after burning for a given time -/
def candleHeight (initialHeight : ℕ) (burnTime : ℕ) : ℕ :=
  initialHeight - (Finset.filter (fun k => 5 * k * (k + 1) / 2 ≤ burnTime) (Finset.range initialHeight)).card

/-- The total burning time for a candle -/
def totalBurnTime (height : ℕ) : ℕ :=
  5 * height * (height + 1) / 2

theorem candle_height_at_half_time (initialHeight : ℕ) (h : initialHeight = 60) :
  candleHeight initialHeight (totalBurnTime initialHeight / 2) = 17 := by
  sorry

#eval candleHeight 60 (totalBurnTime 60 / 2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_at_half_time_l1175_117523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_equals_three_subset_complement_implies_m_range_l1175_117509

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_three (m : ℝ) : A ∩ B m = Set.Icc 1 3 → m = 3 := by
  sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) : A ⊆ (B m)ᶜ → m < -3 ∨ m > 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_equals_three_subset_complement_implies_m_range_l1175_117509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_seattle_atlanta_l1175_117507

-- Define the points on the complex plane
noncomputable def seattle : ℂ := 0
noncomputable def atlanta : ℂ := 900 + 1200 * Complex.I

-- Theorem statement
theorem distance_seattle_atlanta : Complex.abs (atlanta - seattle) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_seattle_atlanta_l1175_117507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1175_117577

/-- Represents a hyperbola with one asymptote y - x = 0 and passing through (√5, 1) -/
structure Hyperbola where
  -- The standard equation of the hyperbola is x² - y² = a
  a : ℝ
  -- The hyperbola passes through (√5, 1)
  point_condition : 5 - 1 = a

/-- The line y = kx - 1 intersects the hyperbola at only one point -/
def single_intersection (h : Hyperbola) (k : ℝ) : Prop :=
  (k = 1 ∨ k = -1) ∨ (k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2)

theorem hyperbola_properties (h : Hyperbola) :
  h.a = 4 ∧ 
  (∀ k : ℝ, single_intersection h k ↔ 
    (k = 1 ∨ k = -1 ∨ k = Real.sqrt 5 / 2 ∨ k = -Real.sqrt 5 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1175_117577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_positive_max_value_on_interval_l1175_117559

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Theorem for the tangent line equation
theorem tangent_line_at_zero (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ x + y - 1 = 0 :=
by sorry

-- Theorem for f(x) > 0 when a = 2
theorem f_positive (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f a x > 0 :=
by sorry

-- Theorem for maximum value of f(x) on [0, a] when a > 1
theorem max_value_on_interval (a : ℝ) (h : a > 1) :
  ∃ max_val : ℝ, max_val = f a a ∧ 
  ∀ x : ℝ, x ∈ Set.Icc 0 a → f a x ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_positive_max_value_on_interval_l1175_117559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_total_candy_l1175_117502

def bobby_candy_count (initial additional : ℕ) : ℕ :=
  initial + additional

theorem bobby_total_candy :
  let initial : ℕ := 26
  let additional : ℕ := 17
  bobby_candy_count initial additional = 43 := by
  unfold bobby_candy_count
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_total_candy_l1175_117502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1175_117596

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) + 9 / (1 + abs x)

-- Define the domain of the function
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0}

-- Theorem: f is an even function on its domain
theorem f_is_even : ∀ x ∈ domain, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l1175_117596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_percentage_is_ten_percent_l1175_117534

/-- The original selling price of the product -/
noncomputable def original_selling_price : ℝ := 439.99999999999966

/-- The additional profit if the product was purchased for 10% less and sold at 30% profit -/
def additional_profit : ℝ := 28

/-- Calculates the original purchase price based on the given conditions -/
noncomputable def original_purchase_price : ℝ := 
  (original_selling_price + additional_profit) / (1.3 * 0.9)

/-- Calculates the original profit -/
noncomputable def original_profit : ℝ := 
  original_selling_price - original_purchase_price

/-- Calculates the original profit percentage -/
noncomputable def original_profit_percentage : ℝ := 
  (original_profit / original_purchase_price) * 100

/-- Theorem stating that the original profit percentage is 10% -/
theorem original_profit_percentage_is_ten_percent : 
  original_profit_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_percentage_is_ten_percent_l1175_117534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_in_sequence_l1175_117528

/-- Definition of the sequence a_n -/
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- We define a_1 as 1 to start the sequence
  | n + 1 => Real.sqrt (a n + 1)

/-- The theorem statement -/
theorem exists_irrational_in_sequence :
  ∃ (n : ℕ), Irrational (a n) :=
by
  sorry

/-- Helper lemma: a_n is positive for all n -/
lemma a_positive (n : ℕ) : a n > 0 :=
by
  sorry

/-- Helper lemma: the recurrence relation holds for n ≥ 1 -/
lemma a_recurrence (n : ℕ) : (a (n + 1))^2 = a n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_irrational_in_sequence_l1175_117528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1175_117537

/-- Given two predicates p and q on real numbers x and m, 
    where q is a necessary but not sufficient condition for p,
    prove that m ≥ 9. -/
theorem range_of_m (x m : ℝ) 
  (hp : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
  (hq : (x+m-1)*(x-m-1) ≤ 0)
  (hm : m > 0)
  (hnec : ∀ x, (-2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2) → (x+m-1)*(x-m-1) ≤ 0)
  (hnsuff : ∃ x, (x+m-1)*(x-m-1) ≤ 0 ∧ ¬(-2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)) :
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1175_117537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1175_117566

theorem calculate_expression : 
  Real.sqrt 12 + 2 * Real.sin (π / 3) - abs (1 - Real.sqrt 3) - (2023 - Real.pi) ^ 0 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l1175_117566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_l1175_117542

theorem coefficient_x4 : 
  ∃ (q : Polynomial ℝ), 
    2*(X^4 - 2*X^3 + 3*X^2) + 4*(2*X^4 + X^3 - X^2 + 2*X^5) - 7*(3 + 2*X^2 - 5*X^4) 
    = 45 * X^4 + q ∧ 
    (∀ n : ℕ, n ≠ 4 → q.coeff n = (2*(X^4 - 2*X^3 + 3*X^2) + 4*(2*X^4 + X^3 - X^2 + 2*X^5) - 7*(3 + 2*X^2 - 5*X^4)).coeff n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_l1175_117542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_of_f_l1175_117570

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc 1 4

-- Theorem statement
theorem max_min_of_f :
  (∃ (x : ℝ), x ∈ domain ∧ ∀ (y : ℝ), y ∈ domain → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ domain ∧ ∀ (y : ℝ), y ∈ domain → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ domain → f x ≤ 6) ∧
  (∀ (x : ℝ), x ∈ domain → 2 ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_of_f_l1175_117570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_in_reciprocal_sequence_l1175_117564

theorem arithmetic_progression_in_reciprocal_sequence (k : ℕ) (h : k ≥ 3) :
  ∃ (a d : ℚ) (f : ℕ → ℕ),
    StrictMono f ∧
    (∀ i, i < k → (1 : ℚ) / (f i : ℚ) = a + i * d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_in_reciprocal_sequence_l1175_117564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_l1175_117547

/-- Represents a convex 2n-gon -/
structure ConvexPolygon (n : ℕ) where
  n_pos : n > 0

/-- Represents a diagonal of the polygon -/
inductive Diagonal (polygon : ConvexPolygon n) where
  | mk : Diagonal polygon

/-- Represents a side of the polygon -/
inductive Side (polygon : ConvexPolygon n) where
  | mk : Side polygon

/-- Predicate for parallel lines -/
def Parallel (polygon : ConvexPolygon n) (d : Diagonal polygon) (s : Side polygon) : Prop :=
  sorry

/-- 
  Theorem: In a convex 2n-gon, there exists a diagonal that is not parallel to any of its sides.
-/
theorem exists_non_parallel_diagonal {n : ℕ} (polygon : ConvexPolygon n) : 
  ∃ (d : Diagonal polygon), ∀ (s : Side polygon), ¬ Parallel polygon d s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_l1175_117547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_parallel_l1175_117533

-- Define the points A, B, and C
def A : Fin 3 → ℝ := ![0, 1, 0]
def B : Fin 3 → ℝ := ![1, 1, 1]
def C : Fin 3 → ℝ := ![0, 2, 1]

-- Define the normal vector of plane β
def n : Fin 3 → ℝ := ![2, 2, -2]

-- Define a function to calculate the vector between two points
def vector (p q : Fin 3 → ℝ) : Fin 3 → ℝ := fun i => q i - p i

-- Define a function to calculate the cross product of two vectors
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![v 1 * w 2 - v 2 * w 1, v 2 * w 0 - v 0 * w 2, v 0 * w 1 - v 1 * w 0]

-- Define a function to check if two vectors are parallel
def are_parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), v = fun i => k * w i

-- Theorem: Plane α is parallel to plane β
theorem planes_are_parallel :
  let AB := vector A B
  let AC := vector A C
  let m := cross_product AB AC
  are_parallel m n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_are_parallel_l1175_117533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_percentage_increase_l1175_117574

/-- Represents the bike ride details -/
structure BikeRide where
  first_hour : ℝ
  second_hour : ℝ
  third_hour : ℝ
  total_distance : ℝ

/-- Calculates the percentage increase between two values -/
noncomputable def percentage_increase (x y : ℝ) : ℝ :=
  (y - x) / x * 100

/-- Theorem stating the conditions and the result to be proved -/
theorem bike_ride_percentage_increase :
  ∀ (ride : BikeRide),
    ride.second_hour = 18 →
    ride.third_hour = ride.second_hour * 1.25 →
    ride.total_distance = 55.5 →
    ride.total_distance = ride.first_hour + ride.second_hour + ride.third_hour →
    percentage_increase ride.first_hour ride.second_hour = 20 := by
  sorry

#check bike_ride_percentage_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_percentage_increase_l1175_117574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_is_nine_fourths_l1175_117521

/-- Represents a parabola opening rightward with equation y² = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a parabola -/
def lies_on (pt : Point) (para : Parabola) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Calculate the distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (pt : Point) (para : Parabola) : ℝ :=
  pt.x + para.p / 2

theorem distance_to_directrix_is_nine_fourths 
  (C : Parabola) (A : Point) 
  (h1 : A.x = 1) 
  (h2 : A.y = Real.sqrt 5) 
  (h3 : lies_on A C) : 
  distance_to_directrix A C = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_is_nine_fourths_l1175_117521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1175_117581

/-- Ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_positive : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := sorry

/-- Angle between three points -/
noncomputable def angle (p q r : Point) : ℝ := sorry

/-- Perimeter of a triangle formed by three points -/
noncomputable def triangle_perimeter (p q r : Point) : ℝ := sorry

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := sorry

theorem ellipse_properties (a b : ℝ) (e : Ellipse a b) (F₁ F₂ A B : Point) :
  let d := distance
  (d F₁ A = 3 * d F₁ B) →
  (d A B = 4) →
  (triangle_perimeter A B F₂ = 16) →
  (d A F₂ = 5) ∧
  (Real.cos (angle A F₂ B) = 3/5 → eccentricity e = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1175_117581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_tetrahedron_volume_correct_l1175_117582

/-- A tetrahedron with a right-angled triangular base -/
structure Tetrahedron where
  /-- Length of the hypotenuse of the base triangle -/
  c : ℝ
  /-- One angle of the base triangle is 30° -/
  base_angle : c > 0 → 30 * π / 180 = Real.arcsin (1 / 2)
  /-- Lateral edges are inclined at 45° to the base plane -/
  lateral_angle : c > 0 → 45 * π / 180 = Real.arctan 1

/-- The volume of the tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := Real.sqrt 3 / 48 * t.c^3

theorem tetrahedron_volume (t : Tetrahedron) :
  volume t = Real.sqrt 3 / 48 * t.c^3 := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl

-- The main theorem stating that our calculation is correct
theorem tetrahedron_volume_correct (t : Tetrahedron) (h : t.c > 0) :
  volume t = Real.sqrt 3 / 48 * t.c^3 := by
  -- Apply the previous theorem
  exact tetrahedron_volume t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_tetrahedron_volume_correct_l1175_117582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l1175_117539

noncomputable section

-- Define the ellipse parameters
def a : ℝ := Real.sqrt 3
def b : ℝ := 1
def e : ℝ := Real.sqrt 6 / 3

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the distance between left focus and minor axis endpoint
def focus_distance : ℝ := Real.sqrt 3

-- Define the fixed point E
def E : ℝ × ℝ := (-1, 0)

-- Define the line equation
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Theorem statement
theorem ellipse_and_circle_intersection :
  -- Part 1: Standard equation of the ellipse
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 3 + y^2 = 1) ∧
  -- Part 2: Existence of k
  (∃ k : ℝ, k = 7/6 ∧
    ∀ A B : ℝ × ℝ,
      (ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line k A.1 A.2 ∧ line k B.1 B.2) →
      (∃ C : ℝ × ℝ, C = E ∧ 
        (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_intersection_l1175_117539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1175_117578

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + (Real.sin x + Real.cos x)^2 ≥ 6 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (Real.tan y + 1 / Real.tan y)^2 + (Real.sin y + Real.cos y)^2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l1175_117578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1175_117526

def p₁ : Prop := ∀ x y : ℝ, x < y → (2:ℝ)^x - (2:ℝ)^(-x) < (2:ℝ)^y - (2:ℝ)^(-y)

def p₂ : Prop := ∀ x y : ℝ, x < y → (2:ℝ)^x + (2:ℝ)^(-x) > (2:ℝ)^y + (2:ℝ)^(-y)

def q₁ : Prop := p₁ ∨ p₂

def q₂ : Prop := p₁ ∧ p₂

def q₃ : Prop := (¬p₁) ∨ p₂

def q₄ : Prop := p₁ ∨ (¬p₂)

theorem problem_solution : q₁ ∧ q₄ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1175_117526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1175_117588

-- Define the fixed cost
noncomputable def fixed_cost : ℝ := 250

-- Define the additional cost function
noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Define the selling price per product
noncomputable def selling_price : ℝ := 0.05

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  selling_price * 1000 * x - C x - fixed_cost

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x > 0, L x ≤ 1000 ∧ L 100 = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_l1175_117588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_added_to_reach_new_average_problem_solution_l1175_117561

theorem value_added_to_reach_new_average (n : ℕ) (initial_avg final_avg : ℚ) : 
  n > 0 → 
  initial_avg < final_avg → 
  (n : ℚ) * (final_avg - initial_avg) = n * final_avg - n * initial_avg := by
  sorry

theorem problem_solution : 
  let n : ℕ := 15
  let initial_avg : ℚ := 40
  let final_avg : ℚ := 50
  let x : ℚ := final_avg - initial_avg
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_added_to_reach_new_average_problem_solution_l1175_117561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3theta_over_sin_theta_l1175_117569

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3theta_over_sin_theta_l1175_117569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1175_117500

theorem solution_set_inequality : 
  {x : ℝ | x * (x - 1) ≤ 0} = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l1175_117500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_needed_l1175_117544

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Calculates the perimeter of an irregular quadrilateral -/
def irregularPerimeter (side1 side2 side3 side4 : ℝ) : ℝ := side1 + side2 + side3 + side4

/-- Calculates the circumference of a circle -/
noncomputable def circleCircumference (diameter : ℝ) : ℝ := Real.pi * diameter

/-- Approximates the perimeter of an ellipse using Ramanujan's formula -/
noncomputable def ellipsePerimeter (majorAxis minorAxis : ℝ) : ℝ :=
  let a := majorAxis / 2
  let b := minorAxis / 2
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

/-- Calculates the total width of gates -/
def totalGateWidth (gates : List ℝ) : ℝ := gates.sum

/-- Theorem: The total fencing needed for Bob's garden is approximately 1191.4 feet -/
theorem total_fencing_needed (rectangleLength rectangleWidth : ℝ)
    (irregularSide1 irregularSide2 irregularSide3 irregularSide4 : ℝ)
    (treeDiameter : ℝ) (pondMajorAxis pondMinorAxis : ℝ) (gates : List ℝ) :
    rectangleLength = 225 → rectangleWidth = 125 →
    irregularSide1 = 75 → irregularSide2 = 150 → irregularSide3 = 45 → irregularSide4 = 120 →
    treeDiameter = 6 → pondMajorAxis = 20 → pondMinorAxis = 12 →
    gates = [3, 10, 4, 7, 2.5, 5] →
    abs (rectanglePerimeter rectangleLength rectangleWidth +
         irregularPerimeter irregularSide1 irregularSide2 irregularSide3 irregularSide4 +
         circleCircumference treeDiameter +
         ellipsePerimeter pondMajorAxis pondMinorAxis +
         totalGateWidth gates - 1191.4) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fencing_needed_l1175_117544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1175_117557

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  (Real.cos A) / (Real.cos B) = (2 * c - a) / b →
  a + c = 3 * Real.sqrt 3 →
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 →
  -- Conclusions
  B = Real.pi / 3 ∧ b = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1175_117557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1175_117512

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi ∧
  b = 5/8 * a ∧
  A = 2 * B →
  Real.cos A = 7/25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1175_117512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_game_theorem_l1175_117513

/-- Represents the state of the card arrangement game -/
def GameState (n : ℕ) := Fin (n + 1) → Fin (n + 1)

/-- Represents a valid move in the game -/
def ValidMove (n : ℕ) (state : GameState n) (h k : Fin (n + 1)) : Prop :=
  h < k ∧ state h = k ∧ ∀ i, h < i ∧ i < k → state i < k

/-- The game is solved when each card is in its correct position -/
def IsSolved (n : ℕ) (state : GameState n) : Prop :=
  ∀ i : Fin (n + 1), state i = i

/-- The number of moves required to solve the game from a given state -/
noncomputable def MovesToSolve (n : ℕ) (state : GameState n) : ℕ := sorry

/-- The maximum number of moves required for any initial state -/
def MaxMoves (n : ℕ) : ℕ := 2 * n - 1

/-- The initial state requiring the maximum number of moves -/
def WorstInitialState (n : ℕ) : GameState n :=
  fun i => if i.val = n then 0 else i.val.succ

theorem card_arrangement_game_theorem (n : ℕ) :
  (∀ state : GameState n, MovesToSolve n state ≤ MaxMoves n) ∧
  (∀ state : GameState n, MovesToSolve n state = MaxMoves n → state = WorstInitialState n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_arrangement_game_theorem_l1175_117513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1175_117510

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = -1/5) 
  (h2 : π/2 < α ∧ α < π) : 
  (Real.sin (π/2 + α) * Real.cos (π/2 - α) = -12/25) ∧ 
  (1 / Real.sin (π - α) + 1 / Real.cos (π - α) = 35/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1175_117510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5632_49999999_l1175_117550

/-- Rounds a real number to the nearest integer -/
noncomputable def myRound (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem round_5632_49999999 :
  myRound 5632.49999999 = 5632 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_5632_49999999_l1175_117550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_rectangle_area_l1175_117546

/-- The area of a composite shape consisting of three rectangles -/
theorem composite_rectangle_area : 
  (let rect1_height : ℕ := 8
   let rect1_width : ℕ := 6
   let rect2_height : ℕ := 3
   let rect2_width : ℕ := 4
   let rect3_height : ℕ := 2
   let rect3_width : ℕ := 5
   let rect1_area := rect1_height * rect1_width
   let rect2_area := rect2_height * rect2_width
   let rect3_area := rect3_height * rect3_width
   let total_area := rect1_area + rect2_area + rect3_area
   total_area) = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_rectangle_area_l1175_117546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_theorem_l1175_117532

/-- The Radical Axis Theorem -/
theorem radical_axis_theorem (x₁ x₂ r₁ r₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : r₁ > 0) (h₄ : r₂ > 0) :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    (x + x₁)^2 + y^2 - r₁^2 = (x - x₂)^2 + y^2 - r₂^2 → x = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_theorem_l1175_117532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unitDigitSumSquaresOdd2023_l1175_117549

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitDigitSumSquaresOdd (n : ℕ) : ℕ :=
  (Finset.sum (Finset.range n) (fun i => ((2 * i + 1)^2) % 10)) % 10

/-- The theorem stating that the units digit of the sum of the squares
    of the first 2023 odd, positive integers is 5 -/
theorem unitDigitSumSquaresOdd2023 : unitDigitSumSquaresOdd 2023 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unitDigitSumSquaresOdd2023_l1175_117549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_approx_l1175_117558

/-- The molar mass of sodium (Na) in g/mol -/
noncomputable def molar_mass_Na : ℝ := 22.99

/-- The molar mass of chlorine (Cl) in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- The molar mass of oxygen (O) in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of sodium hypochlorite (NaClO) in g/mol -/
noncomputable def molar_mass_NaClO : ℝ := molar_mass_Na + molar_mass_Cl + molar_mass_O

/-- The mass percentage of chlorine in sodium hypochlorite (NaClO) -/
noncomputable def mass_percentage_Cl : ℝ := (molar_mass_Cl / molar_mass_NaClO) * 100

/-- Theorem stating that the mass percentage of chlorine in NaClO is approximately 47.61% -/
theorem mass_percentage_Cl_approx :
  |mass_percentage_Cl - 47.61| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cl_approx_l1175_117558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1175_117598

/-- Calculates the length of a train given the speeds of two trains, time to clear, and length of the other train -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (clear_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * clear_time
  total_distance - other_train_length

/-- The theorem stating the length of the first train -/
theorem first_train_length :
  let speed1 : ℝ := 75
  let speed2 : ℝ := 65
  let clear_time : ℝ := 7.353697418492236
  let second_train_length : ℝ := 165
  let first_train_length := calculate_train_length speed1 speed2 clear_time second_train_length
  ∃ ε > 0, |first_train_length - 121.11| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1175_117598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l1175_117567

def is_valid_partition (m n : ℕ) (partition : ℕ → Fin 4) : Prop :=
  ∀ (i : Fin 4) (x y : ℕ),
    partition x = i ∧ partition y = i →
    (x : ℤ) - (y : ℤ) ≠ m ∧
    (x : ℤ) - (y : ℤ) ≠ n ∧
    (x : ℤ) - (y : ℤ) ≠ m + n

theorem partition_exists (m n : ℕ) :
  ∃ (partition : ℕ → Fin 4), is_valid_partition m n partition :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_exists_l1175_117567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_17_5_percent_l1175_117568

/-- Given a principal amount, simple interest, and time, calculate the interest rate. -/
noncomputable def calculate_interest_rate (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem: Given the specified conditions, the interest rate is 17.5% -/
theorem interest_rate_is_17_5_percent 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 140) 
  (h3 : time = 2) : 
  calculate_interest_rate principal simple_interest time = 17.5 := by
  sorry

-- Use #eval only for computable functions
def computable_calculate_interest_rate (principal : Float) (simple_interest : Float) (time : Float) : Float :=
  (simple_interest * 100) / (principal * time)

#eval computable_calculate_interest_rate 400 140 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_17_5_percent_l1175_117568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1175_117536

/-- Proves that the eccentricity of a specific ellipse is 1/2 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let ellipse := λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1
  let circle := λ x y ↦ x^2 + y^2 = b^2
  let line := λ x ↦ Real.sqrt 3 * (x - a)
  (∀ x y, circle x y → line x ≠ y) →  -- line is tangent to circle
  (∃ x, x ≠ a ∧ circle x (line x)) →  -- line intersects circle
  let c := Real.sqrt (a^2 - b^2)
  c / a = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1175_117536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l1175_117531

noncomputable def g (n : ℕ) (x : ℝ) : ℝ := Real.sin x ^ n + Real.cos x ^ n

theorem equation_solutions_count :
  ∃ (S : Finset ℝ), S.card = 5 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
  (∀ x ∈ S, 8 * g 5 x - 5 * g 3 x = 3 * g 1 x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 8 * g 5 x - 5 * g 3 x = 3 * g 1 x → x ∈ S) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l1175_117531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_necessary_not_sufficient_condition_main_theorem_l1175_117592

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (2 + x) + Real.log (4 - x)

-- Define the domain A
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- State the theorem
theorem range_of_m_for_necessary_not_sufficient_condition :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ 
            (∃ x : ℝ, x ∈ A ∧ x ∉ B m) ↔ 
            m < 5/2 := by
  sorry

-- Define the main theorem that encapsulates both parts of the problem
theorem main_theorem :
  (A = {x : ℝ | -2 ≤ x ∧ x < 4}) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ 
            (∃ x : ℝ, x ∈ A ∧ x ∉ B m) ↔ 
            m < 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_necessary_not_sufficient_condition_main_theorem_l1175_117592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_common_sign_vector_l1175_117522

/-- Given a 3x3 matrix A with positive diagonal entries and negative off-diagonal entries,
    there exist strictly positive c₁, c₂, c₃ such that Ac is either all positive, all negative, or all zero. -/
theorem existence_of_common_sign_vector (A : Matrix (Fin 3) (Fin 3) ℝ)
    (h_diag : ∀ i, A i i > 0)
    (h_off_diag : ∀ i j, i ≠ j → A i j < 0) :
    ∃ c : Fin 3 → ℝ, (∀ i, c i > 0) ∧
    ((∀ i, (A.mulVec c) i > 0) ∨
     (∀ i, (A.mulVec c) i < 0) ∨
     (∀ i, (A.mulVec c) i = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_common_sign_vector_l1175_117522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1175_117529

def sequence_a : ℕ → ℤ
  | 0 => 33
  | (n + 1) => sequence_a n + 2 * (n + 1)

theorem sequence_a_formula (n : ℕ) : sequence_a n = n^2 + n + 33 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [sequence_a]
    rw [ih]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1175_117529
