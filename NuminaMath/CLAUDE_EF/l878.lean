import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l878_87886

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x^2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 0 (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l878_87886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_seventeen_l878_87846

theorem ceiling_sum_equals_seventeen :
  ⌈(Real.sqrt (16/5:ℝ))⌉ + ⌈(16/5:ℝ)⌉ + ⌈((16/5:ℝ)^2)⌉ = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_seventeen_l878_87846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_sum_of_k_l878_87817

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- A triangle with vertices (0, 0), (2, 2), and (8k, 0) is divided into two equal areas by the line y = 2kx. The sum of all possible values of k is -1/4. -/
theorem triangle_division_sum_of_k : 
  ∃ k₁ k₂ : ℝ,
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 2)
  let C : ℝ × ℝ := (8*k₁, 0)
  let dividing_line (x : ℝ) := 2*k₁*x
  (∃ D : ℝ × ℝ, D.1 ∈ Set.Icc 2 (8*k₁) ∧ D.2 = dividing_line D.1 ∧ 
   area_triangle A B D = area_triangle C D B) ∧
  let C' : ℝ × ℝ := (8*k₂, 0)
  let dividing_line' (x : ℝ) := 2*k₂*x
  (∃ D' : ℝ × ℝ, D'.1 ∈ Set.Icc 2 (8*k₂) ∧ D'.2 = dividing_line' D'.1 ∧ 
   area_triangle A B D' = area_triangle C' D' B) ∧
  k₁ + k₂ = -1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_division_sum_of_k_l878_87817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l878_87838

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 + x) + (Real.sin (Real.pi / 2 + x))^2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/4 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l878_87838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wenchuan_earthquake_donation_l878_87835

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation with a given number of significant figures -/
noncomputable def toScientificNotation (x : ℝ) (significantFigures : ℕ) : ScientificNotation :=
  sorry

/-- The problem statement -/
theorem wenchuan_earthquake_donation :
  let original := 43.681 * 1000000000 -- 43.681 billion
  let scientificNotation := toScientificNotation original 3
  scientificNotation.coefficient = 0.437 ∧ scientificNotation.exponent = 12 := by
  sorry

#eval 43.681 * 1000000000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wenchuan_earthquake_donation_l878_87835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l878_87818

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧ 
  (Real.sin (2 * t.A + t.B)) / (Real.sin t.A) = 2 * (1 - Real.cos t.C)

-- Define the area condition
def area_condition (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : area_condition t) : 
  t.b = 2 ∧ (t.c = Real.sqrt 3 ∨ t.c = Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l878_87818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_inequality_l878_87858

theorem cos_sin_inequality (a : Real) (h : a ∈ Set.Ioo (π/4) (π/2)) :
  (Real.cos a) ^ (Real.sin a) < (Real.cos a) ^ (Real.cos a) ∧ 
  (Real.cos a) ^ (Real.cos a) < (Real.sin a) ^ (Real.cos a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_inequality_l878_87858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_decreasing_power_of_16_l878_87866

/-- A natural number is decreasing if each digit in its decimal representation, 
    except for the first, is less than or equal to the previous one. -/
def is_decreasing (n : ℕ) : Prop := 
  ∀ i j, 0 < i ∧ i < j ∧ j < (Nat.digits 10 n).length → 
    (Nat.digits 10 n).get ⟨i, by sorry⟩ ≥ (Nat.digits 10 n).get ⟨j, by sorry⟩

/-- There does not exist a natural number n such that 16^n is decreasing. -/
theorem no_decreasing_power_of_16 : ¬∃ n : ℕ, is_decreasing (16^n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_decreasing_power_of_16_l878_87866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_triangle_side_c_l878_87836

-- Define the function f and vectors m and n
noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Define the interval of monotonic increase
def monotonic_increase_interval (k : ℤ) : Set ℝ := 
  Set.Icc (-Real.pi/3 + ↑k * Real.pi) (Real.pi/6 + ↑k * Real.pi)

-- Theorem for the interval of monotonic increase
theorem f_monotonic_increase (k : ℤ) : 
  StrictMonoOn f (monotonic_increase_interval k) := 
sorry

-- Define triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (area : ℝ)

-- Theorem for the value of c in triangle ABC
theorem triangle_side_c (abc : Triangle) (h1 : f abc.A = 2) (h2 : abc.b = 1) 
  (h3 : abc.area = Real.sqrt 3 / 2) : abc.c = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increase_triangle_side_c_l878_87836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l878_87850

/-- The function f(x) as described in the problem -/
noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (2 * x + φ) - 1/2

theorem range_of_m (A : ℝ) (φ : ℝ) (h1 : A > 0) (h2 : 0 < φ) (h3 : φ < Real.pi/2)
  (h4 : f A φ 0 = 1)  -- f(x) intercepts y-axis at 1
  (h5 : ∀ x, f A φ x = f A φ (Real.pi/6 - x))  -- f(x) is symmetric about x = π/12
  (h6 : ∀ x ∈ Set.Icc 0 (Real.pi/2), ∀ m : ℝ, m^2 - 3*m ≤ f A φ x) :
  ∀ m : ℝ, m ∈ Set.Icc 1 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l878_87850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_of_g_7_l878_87839

-- Define the functions r and g
noncomputable def r (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - r x

-- State the theorem
theorem r_of_g_7 : r (g 7) = Real.sqrt (37 - 5 * Real.sqrt 37) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_of_g_7_l878_87839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_fourth_eq_zero_l878_87863

/-- Given a function f(x) = f'(π/2) * sin(x) + cos(x), prove that f(π/4) = 0 -/
theorem f_pi_fourth_eq_zero (f : ℝ → ℝ) (h : ∀ x, f x = (deriv f) (π / 2) * Real.sin x + Real.cos x) :
  f (π / 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_fourth_eq_zero_l878_87863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solution_l878_87887

-- Define the function f(x)
noncomputable def f (a x : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x - a / 4 - 1 / 2

-- Define the maximum value function M(a)
noncomputable def M (a : ℝ) : ℝ :=
  if 0 < a ∧ a ≤ 2 then
    1 / 4 * a ^ 2 - 1 / 4 * a + 1 / 2
  else if a > 2 then
    3 / 4 * a - 1 / 2
  else
    0  -- This case should not occur given a > 0, but we need to define it for completeness

-- Theorem statement
theorem max_value_and_solution (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ M a) ∧
  (M (10 / 3) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_solution_l878_87887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l878_87864

/-- Given a triangle PQR with side lengths satisfying certain conditions, 
    its maximum possible area is 3565. -/
theorem triangle_max_area :
  ∀ (PQ QR PR : ℝ),
  PQ = 13 →
  QR / PR = 50 / 51 →
  0 < PQ ∧ 0 < QR ∧ 0 < PR →
  PQ + QR > PR ∧ PQ + PR > QR ∧ QR + PR > PQ →
  (∀ A : ℝ, A = Real.sqrt ((PQ + QR + PR) / 2 * 
              ((PQ + QR + PR) / 2 - PQ) * 
              ((PQ + QR + PR) / 2 - QR) * 
              ((PQ + QR + PR) / 2 - PR)) → A ≤ 3565) ∧
  (∃ A : ℝ, A = Real.sqrt ((PQ + QR + PR) / 2 * 
              ((PQ + QR + PR) / 2 - PQ) * 
              ((PQ + QR + PR) / 2 - QR) * 
              ((PQ + QR + PR) / 2 - PR)) ∧ A = 3565) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l878_87864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_m2_n2_l878_87844

theorem min_value_sqrt_m2_n2 (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  ∃ (k : ℝ), k = Real.sqrt (m^2 + n^2) ∧ k ≥ Real.sqrt 3 ∧ 
  ∀ (m' n' : ℝ), m'*a + n'*b = 3 → Real.sqrt (m'^2 + n'^2) ≥ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_m2_n2_l878_87844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_lambda_bound_l878_87804

/-- Arithmetic sequence sum -/
def S (n : ℕ+) (lambda : ℝ) : ℝ := n^2 + (lambda + 1) * n

/-- The sequence {S_n} is increasing -/
def is_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, S (n + 1) lambda > S n lambda

theorem arithmetic_sequence_lambda_bound (lambda : ℝ) :
  is_increasing lambda → lambda > -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_lambda_bound_l878_87804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_containing_zero_up_to_2050_l878_87895

def containsZero (n : Nat) : Bool :=
  n.repr.any (· = '0')

def countContainingZero (upperBound : Nat) : Nat :=
  (List.range upperBound).filter containsZero |> List.length

theorem count_containing_zero_up_to_2050 :
  countContainingZero 2051 = 502 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_containing_zero_up_to_2050_l878_87895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_iff_gamma_range_l878_87859

/-- Represents a triangle with sides a, b, c, obtuse angle γ, and area T. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  γ : ℝ
  T : ℝ
  hₐ : a > 0
  hₑ : b > 0
  hᵤ : c > 0
  hₒ : π / 2 < γ ∧ γ < π
  hₜ : T > 0

/-- The inequality condition for the triangle -/
def triangle_inequality (t : Triangle) : Prop :=
  t.T / Real.sqrt (t.a^2 * t.b^2 - 4 * t.T^2) +
  t.T / Real.sqrt (t.b^2 * t.c^2 - 4 * t.T^2) +
  t.T / Real.sqrt (t.c^2 * t.a^2 - 4 * t.T^2) ≥ (3 * Real.sqrt 3) / 2

/-- The theorem stating the equivalence between the inequality and the range of γ -/
theorem triangle_inequality_iff_gamma_range (t : Triangle) :
  triangle_inequality t ↔ (π / 2 < t.γ ∧ t.γ ≤ 105.248 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_iff_gamma_range_l878_87859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_positive_integer_l878_87868

theorem fraction_is_positive_integer (p : ℕ+) : 
  (∃ (k : ℤ), k > 0 ∧ ((4 * p.val + 11) : ℤ) = k * ((2 * p.val - 7) : ℤ)) ↔ 
  p.val = 4 ∨ p.val = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_is_positive_integer_l878_87868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l878_87891

-- Define a type for ten-digit numbers with all digits different
def TenDigitNumber := {n : Fin 10 → Fin 10 // Function.Injective n}

-- Define the transformation function
def transform (n : TenDigitNumber) : Fin 10 → Fin 10 :=
  fun i => (if i > 0 && n.val (i-1) < n.val i then 1 else 0) +
           (if i < 9 && n.val (i+1) < n.val i then 1 else 0)

-- Define the sequences we want to check
def seq1 : Fin 10 → Fin 10 := fun i => [1,1,0,1,1,1,1,1,1,1].get ⟨i.val, by simp⟩
def seq2 : Fin 10 → Fin 10 := fun i => [1,2,0,1,2,0,1,0,2,0].get ⟨i.val, by simp⟩
def seq3 : Fin 10 → Fin 10 := fun i => [1,0,2,1,0,2,1,0,2,0].get ⟨i.val, by simp⟩
def seq4 : Fin 10 → Fin 10 := fun i => [0,1,1,2,1,0,2,0,1,1].get ⟨i.val, by simp⟩

-- The theorem to prove
theorem transformation_result :
  (∃ n : TenDigitNumber, transform n = seq1) ∧
  (∃ n : TenDigitNumber, transform n = seq3) ∧
  (∃ n : TenDigitNumber, transform n = seq4) ∧
  ¬(∃ n : TenDigitNumber, transform n = seq2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l878_87891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l878_87821

noncomputable section

open Real

/-- The function f(x) = tan(ωx + π/4) with ω > 0 and least positive period 2π -/
def f (ω : ℝ) (x : ℝ) : ℝ := tan (ω * x + π / 4)

/-- Theorem stating the properties of the function f -/
theorem function_properties (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + 2*π) = f ω x) :
  ω = 1/2 ∧ f ω (π/6) = sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l878_87821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_minus_one_minus_three_l878_87894

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x+2) passes through (-1,3)
def passes_through_minus_one_three (f : ℝ → ℝ) : Prop :=
  f 1 = 3

-- Define symmetry about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem passes_through_minus_one_minus_three
  (h1 : passes_through_minus_one_three f)
  (h2 : symmetric_about_origin f) :
  f (-1) = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_minus_one_minus_three_l878_87894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_difference_l878_87805

noncomputable def S (r : ℝ) : ℝ := 10 / (1 - r)

theorem geometric_series_difference (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : S b ^ 2 - S (-b) ^ 2 = 7840) : 
  S b - S (-b) = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_difference_l878_87805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l878_87883

noncomputable def work_time_together (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

theorem work_completion_time (time_a time_b : ℝ) (ha : time_a > 0) (hb : time_b > 0) :
  work_time_together time_a time_b = 24 ↔ time_a = 40 ∧ time_b = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l878_87883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_approx_one_mph_l878_87897

/-- Calculates the speed of a car in miles per hour given distance in yards and time in minutes -/
noncomputable def carSpeed (distance_yards : ℝ) (time_minutes : ℝ) : ℝ :=
  let yards_per_mile : ℝ := 1760
  let minutes_per_hour : ℝ := 60
  let distance_miles : ℝ := distance_yards / yards_per_mile
  let time_hours : ℝ := time_minutes / minutes_per_hour
  distance_miles / time_hours

/-- Theorem stating that a car traveling 106 yards in 3.61 minutes has a speed of approximately 1 mph -/
theorem car_speed_approx_one_mph :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |carSpeed 106 3.61 - 1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_approx_one_mph_l878_87897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_ratio_l878_87803

-- Define the points
noncomputable def A : ℝ × ℝ := (1, 2)
noncomputable def B : ℝ × ℝ := (2, 1)
noncomputable def C : ℝ × ℝ := (3, -3)
noncomputable def D : ℝ × ℝ := (0, 0)

-- Define the diagonal intersection point
noncomputable def M : ℝ × ℝ := (1, 1/2)

-- Theorem statement
theorem diagonal_intersection_ratio :
  let AM := (M.1 - A.1, M.2 - A.2)
  let MC := (C.1 - M.1, C.2 - M.2)
  AM.1 * 3 = MC.1 ∧ AM.2 * 3 = MC.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_ratio_l878_87803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_f_properties_l878_87816

/-- The function f(x) given θ -/
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

/-- f is an odd function when θ = 2π/3 -/
theorem f_is_odd (x : ℝ) : f (2 * Real.pi / 3) (-x) = -(f (2 * Real.pi / 3) x) := by sorry

/-- f is decreasing on [0, π/4] when θ = 2π/3 -/
theorem f_is_decreasing : 
  StrictMonoOn (fun x => -(f (2 * Real.pi / 3) x)) (Set.Icc 0 (Real.pi / 4)) := by sorry

/-- The main theorem combining both properties -/
theorem f_properties : 
  (∀ x, f (2 * Real.pi / 3) (-x) = -(f (2 * Real.pi / 3) x)) ∧ 
  StrictMonoOn (fun x => -(f (2 * Real.pi / 3) x)) (Set.Icc 0 (Real.pi / 4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_f_properties_l878_87816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l878_87892

/-- Represents a hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b ^ 2 / h.a ^ 2)

/-- The x-coordinate of the right focus of a hyperbola -/
noncomputable def right_focus_x (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a ^ 2 + h.b ^ 2)

/-- Theorem: Given a hyperbola E with equation (x^2 / a^2) - (y^2 / b^2) = 1 (a > 0, b > 0),
    if a line perpendicular to the x-axis through its right focus F intersects E at points B and C,
    and if triangle ABC (where A is the left vertex) is a right-angled triangle,
    then the eccentricity of E is 2. -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola)
  (h_right_angle : ∃ (B C : ℝ × ℝ),
    (B.1 = right_focus_x h ∧ (B.1^2 / h.a^2 - B.2^2 / h.b^2 = 1)) ∧
    (C.1 = right_focus_x h ∧ (C.1^2 / h.a^2 - C.2^2 / h.b^2 = 1)) ∧
    ((-h.a - B.1) * (C.2 - B.2) = (B.2 - (-h.a)) * (C.1 - B.1))) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l878_87892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_number_l878_87848

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧ 
  (let digits := n.digits 10;
   digits.length = 4 ∧ digits.toFinset.card = 4)

def count_valid_numbers : ℕ := sorry

theorem probability_of_valid_number : 
  (count_valid_numbers : ℚ) / (9000 : ℚ) = 9 / 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_number_l878_87848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_x_in_original_interval_l878_87857

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x y : ℝ,
    x ∈ Set.Ioo 0 (5 * Real.pi / 12) →
    y ∈ Set.Ioo 0 (5 * Real.pi / 12) →
    x < y →
    f x < f y :=
by
  sorry

-- Additional hypothesis to ensure x is in the original interval (0, π/2)
theorem x_in_original_interval :
  ∀ x : ℝ,
    x ∈ Set.Ioo 0 (5 * Real.pi / 12) →
    x ∈ Set.Ioo 0 (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_x_in_original_interval_l878_87857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l878_87865

/-- The speed of a train in kilometers per hour, given its distance traveled and time taken -/
noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 60

theorem train_speed_is_90 (distance : ℝ) (time : ℝ) 
  (h1 : distance = 7.5) 
  (h2 : time = 5) : 
  train_speed distance time = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_90_l878_87865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_domain_of_G_l878_87833

-- Define the function f with domain [0, 1]
def f : Set ℝ → Set ℝ := fun S ↦ S ∩ Set.Icc 0 1

-- Define F(x) = f(x^2)
def F (f : Set ℝ → Set ℝ) : Set ℝ := 
  {x | ∃ y ∈ f (Set.Icc 0 1), y = x^2}

-- Define G(x) = f(x + a) + f(x - a)
def G (f : Set ℝ → Set ℝ) (a : ℝ) : Set ℝ := 
  {x | ∃ y ∈ f (Set.Icc 0 1), (y = x + a ∨ y = x - a)}

-- Theorem for the domain of F
theorem domain_of_F (f : Set ℝ → Set ℝ) : 
  F f = Set.Icc (-1) 1 := by sorry

-- Theorem for the domain of G
theorem domain_of_G (f : Set ℝ → Set ℝ) (a : ℝ) : 
  G f a = if a < -1/2 ∨ a > 1/2 then ∅ 
          else if -1/2 ≤ a ∧ a ≤ 0 then Set.Icc (-a) (1 + a)
          else if 0 < a ∧ a ≤ 1/2 then Set.Icc a (1 - a)
          else ∅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_F_domain_of_G_l878_87833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l878_87888

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (15 * x^2 - 13 * x - 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≤ -2/5 ∨ x ≥ 4/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l878_87888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l878_87867

noncomputable def f (x : ℝ) := x + 1/x + (x + 1/x)^2

theorem min_value_of_f :
  (∀ x > 0, f x ≥ 6) ∧ (∃ x > 0, f x = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l878_87867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l878_87875

/-- The phase shift of a sine function y = a * sin(b * x + c) is given by -c/b -/
noncomputable def phase_shift (a b c : ℝ) : ℝ := -c / b

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (3 * x - Real.pi / 4)

theorem phase_shift_of_f :
  phase_shift 3 3 (-Real.pi / 4) = Real.pi / 12 ∧ 
  phase_shift 3 3 (-Real.pi / 4) > 0 := by
  sorry

#check phase_shift_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l878_87875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_is_one_l878_87860

variable (α β γ : ℝ)
variable (a₁ a₂ a₃ : ℝ)
variable (i j k a : Euclidean E)

/-- Three mutually perpendicular unit vectors -/
axiom orthonormal_basis : 
  i.dot j = 0 ∧ i.dot k = 0 ∧ j.dot k = 0 ∧ 
  i.norm = 1 ∧ j.norm = 1 ∧ k.norm = 1

/-- Definition of vector a -/
axiom a_def : a = a₁ • i + a₂ • j + a₃ • k

/-- a is non-zero -/
axiom a_nonzero : a ≠ 0

/-- Angles between a and i, j, k -/
axiom angle_α : Real.cos α = a.dot i / (a.norm * i.norm)
axiom angle_β : Real.cos β = a.dot j / (a.norm * j.norm)
axiom angle_γ : Real.cos γ = a.dot k / (a.norm * k.norm)

/-- The theorem to be proved -/
theorem cos_squared_sum_is_one : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_sum_is_one_l878_87860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_condition_l878_87820

theorem ln_inequality_condition (x : ℝ) : 
  (∀ y : ℝ, Real.log (y + 1) < 0 → y < 0) ∧ 
  (∃ z : ℝ, z < 0 ∧ Real.log (z + 1) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_condition_l878_87820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l878_87884

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = Real.log 4 / Real.log 3 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, (3 : ℝ)^(2*x) - 13 * (3 : ℝ)^x + 36 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l878_87884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l878_87893

theorem sin_cos_equation_solution (x : ℝ) : 
  (Real.sin x * Real.cos x * Real.cos (2 * x) * Real.cos (8 * x) = (1 / 4) * Real.sin (12 * x)) ↔ 
  ∃ k : ℤ, x = (k * Real.pi) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_solution_l878_87893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lamps_on_road_l878_87847

/-- Represents a street lamp with its position on the road -/
structure Lamp where
  position : ℚ
  deriving Repr, BEq

/-- Represents a road with lamps -/
structure Road where
  length : ℚ
  lamps : List Lamp
  deriving Repr

/-- Checks if a road is fully illuminated -/
def isFullyIlluminated (road : Road) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ road.length → ∃ lamp ∈ road.lamps, 
    lamp.position - 1/2 ≤ x ∧ x ≤ lamp.position + 1/2

/-- Checks if removing any lamp results in incomplete illumination -/
def isMaximallyRedundant (road : Road) : Prop :=
  ∀ lamp ∈ road.lamps, ¬isFullyIlluminated ⟨road.length, road.lamps.erase lamp⟩

/-- The main theorem to be proved -/
theorem max_lamps_on_road :
  ∃ (road : Road), road.length = 1000 ∧ 
    isFullyIlluminated road ∧ 
    isMaximallyRedundant road ∧
    road.lamps.length = 1998 ∧
    (∀ (road' : Road), road'.length = 1000 → 
      isFullyIlluminated road' → 
      isMaximallyRedundant road' → 
      road'.lamps.length ≤ 1998) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lamps_on_road_l878_87847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l878_87869

-- Define the differential equation
noncomputable def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv^[2] y) t - 2 * (deriv y) t - 3 * y t = Real.exp (3 * t)

-- Define the initial conditions
def initial_conditions (y : ℝ → ℝ) : Prop :=
  y 0 = 0 ∧ (deriv y) 0 = 0

-- Define the solution function
noncomputable def solution (t : ℝ) : ℝ :=
  (1/4) * t * Real.exp (3*t) - (1/16) * Real.exp (3*t) + (1/16) * Real.exp (-t)

-- Theorem statement
theorem solution_satisfies_diff_eq_and_initial_conditions :
  diff_eq solution ∧ initial_conditions solution := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_and_initial_conditions_l878_87869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_money_fraction_l878_87881

-- Define the friends
inductive Friend
| Loki
| Moe
| Nick
| Ott
deriving Repr, DecidableEq

-- Define the initial conditions and money transfers
def initial_money (f : Friend) : ℚ :=
  match f with
  | Friend.Ott => 1
  | _ => 0  -- We don't know the initial amounts for others, so we set them to 0

def gave_fraction (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/6
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Ott => 0

def ott_return_fraction : ℚ := 1/10

-- Define the amount each friend (except Ott) gave to Ott
def amount_given (f : Friend) (x : ℚ) : ℚ :=
  if f = Friend.Ott then 0 else x

-- Define Ott's final money
def ott_final_money (x : ℚ) : ℚ :=
  1 + (3 * x * (1 - ott_return_fraction))

-- Define the total money of the group
def total_money (x : ℚ) : ℚ :=
  15 * x + 1

-- The theorem to prove
theorem ott_money_fraction (x : ℚ) :
  ott_final_money x / total_money x = (10 + 27*x) / (150*x + 10) := by
  -- Expand the definitions
  unfold ott_final_money total_money
  -- Simplify the fraction
  simp [add_mul, mul_add, mul_assoc, mul_comm]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ott_money_fraction_l878_87881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_am_length_l878_87809

-- Define the triangle ABC
structure Triangle (α : Type*) [LinearOrderedField α] where
  A : α × α
  B : α × α
  C : α × α

-- Define point M on BC
def M {α : Type*} [LinearOrderedField α] (t : Triangle α) : α × α :=
  sorry

-- Define the distance function
def dist {α : Type*} [LinearOrderedField α] (p q : α × α) : α :=
  sorry

-- Define the perimeter function
def perimeter {α : Type*} [LinearOrderedField α] (p q r : α × α) : α :=
  dist p q + dist q r + dist r p

theorem isosceles_triangle_am_length
  {α : Type*} [LinearOrderedField α] (t : Triangle α) :
  -- Triangle ABC is isosceles
  dist t.A t.B = dist t.A t.C →
  -- M is on BC such that BM = MC
  dist t.B (M t) = dist (M t) t.C →
  -- Perimeter of triangle ABC is 64
  perimeter t.A t.B t.C = 64 →
  -- Perimeter of triangle ABM is 40
  perimeter t.A t.B (M t) = 40 →
  -- Then AM = 8
  dist t.A (M t) = 8 := by
  sorry

#check isosceles_triangle_am_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_am_length_l878_87809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertical_coordinate_P_l878_87823

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define the symmetry condition for Q and Q₁
def symmetric_points (Q Q₁ : ℝ × ℝ) : Prop :=
  Q.1 = Q₁.1 ∧ Q.2 = -Q₁.2

-- Define the slope product condition
def slope_product (Q Q₁ : ℝ × ℝ) : Prop :=
  let slope_AQ := (Q.2 - A.2) / (Q.1 - A.1)
  let slope_BQ₁ := (Q₁.2 - B.2) / (Q₁.1 - B.1)
  slope_AQ * slope_BQ₁ = -1/4

-- Define the point P on the line x=1
def P (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the theorem
theorem max_vertical_coordinate_P :
  ∃ (Q : ℝ × ℝ) (Q₁ : ℝ × ℝ),
    C Q.1 Q.2 ∧
    symmetric_points Q Q₁ ∧
    slope_product Q Q₁ ∧
    (∃ (y : ℝ), 
      (∀ (y' : ℝ), abs (A.1 - (P y').1) ≤ abs (A.1 - (P y).1)) →
      abs y ≤ Real.sqrt 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertical_coordinate_P_l878_87823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_fc_l878_87879

theorem triangle_similarity_fc (DC CB : ℝ) (AB ED AD : ℝ) (h1 : DC = 9) (h2 : CB = 6)
  (h3 : AB = (1/3) * AD) (h4 : ED = (2/3) * AD) : 
  let CA := CB + AB
  let FC := (ED * CA) / AD
  FC = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_fc_l878_87879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_combinations_l878_87852

/-- The number of ways to set n traffic lights, where k lights have three states and (n-k) lights have two states. -/
def number_of_ways_to_set_lights (n k : ℕ) : ℕ :=
  3^k * 2^(n-k)

theorem traffic_light_combinations (n k : ℕ) : 
  number_of_ways_to_set_lights n k = 3^k * 2^(n-k) :=
by
  -- Unfold the definition of number_of_ways_to_set_lights
  unfold number_of_ways_to_set_lights
  -- The equality now holds by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_combinations_l878_87852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_pow_7_times_7_pow_4_l878_87827

-- Define the number of digits in a positive integer
noncomputable def num_digits (n : ℕ+) : ℕ :=
  Nat.floor (Real.log n.val / Real.log 10) + 1

-- State the theorem
theorem digits_of_3_pow_7_times_7_pow_4 :
  num_digits (3^7 * 7^4 : ℕ+) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_pow_7_times_7_pow_4_l878_87827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l878_87831

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x

-- State the theorem
theorem tangent_line_and_inequality (a b : ℝ) :
  (∀ x : ℝ, x > 0 → (deriv (f a b)) x = a / x + b) →
  (f a b 1 = -1/2) →
  ((deriv (f a b)) 1 = 1/2) →
  (∀ x : ℝ, x > 1 → (∃ k : ℝ, f a b x + k / x < 0 ↔ k ≤ 1/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l878_87831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l878_87898

theorem cos_sin_equation (x : ℝ) : 
  Real.cos x - 3 * Real.sin x = 2 → Real.sin x + 3 * Real.cos x = -26/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l878_87898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_with_area_div_by_20_l878_87872

/-- Triangle with consecutive integer sides -/
structure ConsecutiveTriangle where
  a : ℕ
  h1 : a > 0

/-- Area of a triangle given its sides -/
noncomputable def triangleArea (t : ConsecutiveTriangle) : ℝ :=
  let s := (t.a + (t.a + 1) + (t.a + 2)) / 2
  Real.sqrt (s * (s - t.a) * (s - (t.a + 1)) * (s - (t.a + 2)))

/-- Predicate for a triangle with area divisible by 20 -/
def hasAreaDivisibleBy20 (t : ConsecutiveTriangle) : Prop :=
  ∃ k : ℕ, triangleArea t = 20 * k

/-- The smallest triangle with consecutive integer sides and area divisible by 20 -/
def smallestValidTriangle : ConsecutiveTriangle :=
  ⟨2701, by norm_num⟩

theorem smallest_triangle_with_area_div_by_20 :
  hasAreaDivisibleBy20 smallestValidTriangle ∧
  ∀ t : ConsecutiveTriangle, t.a < smallestValidTriangle.a → ¬hasAreaDivisibleBy20 t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_with_area_div_by_20_l878_87872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l878_87877

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Define the foci
def focus1 : ℝ × ℝ := (2, 2)
def focus2 : ℝ × ℝ := (-2, -2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem hyperbola_foci_distance :
  distance focus1 focus2 = 4 * Real.sqrt 2 := by
  -- Unfold definitions
  unfold distance focus1 focus2
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l878_87877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l878_87845

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with two foci -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Given an ellipse with specific foci and one x-intercept, prove the other x-intercept -/
theorem ellipse_x_intercept (e : Ellipse) (x_intercept : Point) :
  e.focus1 = ⟨0, 3⟩ →
  e.focus2 = ⟨4, 0⟩ →
  x_intercept = ⟨0, 0⟩ →
  ∃ (other_intercept : Point),
    other_intercept.y = 0 ∧
    other_intercept.x = 56 / 11 ∧
    distance x_intercept e.focus1 + distance x_intercept e.focus2 =
    distance other_intercept e.focus1 + distance other_intercept e.focus2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l878_87845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l878_87890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + 1

theorem min_value_of_f (a : ℝ) :
  (∀ x, f a x ≤ 3) → (∃ x, f a x = 3) →
  (∃ x, f a x = -1) ∧ (∀ x, f a x ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l878_87890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_area_ratio_l878_87862

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Area of triangle ABF -/
noncomputable def area_ABF (a b : ℝ) : ℝ := b * Real.sqrt (a^2 - b^2) / 2

/-- Theorem about the ellipse properties and area ratio -/
theorem ellipse_properties_and_area_ratio :
  ∀ (a b : ℝ) (h1 : a > b) (h2 : b > 0),
  eccentricity a b = Real.sqrt 2 / 2 →
  area_ABF a b = 1 →
  (∀ (x y : ℝ), ellipse x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  ∃ (lower upper : ℝ),
    lower = 3 - 2 * Real.sqrt 2 ∧
    upper = 1 ∧
    ∀ (m : ℝ), m^2 > 1 →
      ∃ (t : ℝ), lower < t ∧ t < upper ∧
        ∃ (y₁ y₂ : ℝ),
          y₁ ≠ y₂ ∧
          ellipse (m * y₁ + 2) y₁ a b ∧
          ellipse (m * y₂ + 2) y₂ a b ∧
          t = abs y₁ / abs y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_area_ratio_l878_87862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_log10_81_l878_87802

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem floor_log10_81 : floor (log10 81) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_log10_81_l878_87802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_vs_simple_interest_l878_87807

/-- Calculate compound interest given principal, rate, time, and compounding frequency -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (frequency : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time) - principal

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem comparing compound and simple interest -/
theorem compound_vs_simple_interest :
  ∀ (principal : ℝ),
    simple_interest principal 0.2 2 = 400 →
    ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |compound_interest principal 0.2 2 4 - 477.46| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_vs_simple_interest_l878_87807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l878_87834

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 4)

-- State the theorem
theorem inverse_f_undefined_at_one :
  ∀ x : ℝ, f x = 1 → x = 3 ∨ x = 4 :=
by
  intro x
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

-- Note: We're proving that f(x) = 1 implies x = 3 or x = 4,
-- which is equivalent to saying f^(-1)(1) is undefined.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l878_87834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l878_87829

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_diameter : ℝ)
  (h_width : rectangle_width = 20)
  (h_height : rectangle_height = 15)
  (h_diameter : circle_diameter = 4)
  (h_positive : rectangle_width > 0 ∧ rectangle_height > 0 ∧ circle_diameter > 0) :
  Real.sqrt ((rectangle_width - circle_diameter) ^ 2 + (rectangle_height - circle_diameter) ^ 2) = Real.sqrt 377 :=
by
  -- Replace this line with the actual proof when ready
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_circle_centers_l878_87829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_tangency_l878_87841

-- Define the types for line and hyperbola
structure Line where
  -- Add necessary fields for a line

structure Hyperbola where
  -- Add necessary fields for a hyperbola

-- Define the predicates
def have_one_common_point (l : Line) (h : Hyperbola) : Prop := sorry

def are_tangent (l : Line) (h : Hyperbola) : Prop := sorry

-- State the theorem
theorem line_hyperbola_tangency 
  (l : Line) (h : Hyperbola) : 
  (are_tangent l h → have_one_common_point l h) ∧ 
  ¬(have_one_common_point l h → are_tangent l h) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_tangency_l878_87841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l878_87808

/-- Calculates the selling price of a bond given its face value, interest rate, tax rate, inflation rate, and net interest percentage. -/
noncomputable def bondSellingPrice (faceValue : ℝ) (interestRate : ℝ) (taxRate : ℝ) (inflationRate : ℝ) (netInterestPercentage : ℝ) : ℝ :=
  let interestEarned := faceValue * interestRate
  let taxPaid := interestEarned * taxRate
  let netInterestAfterTax := interestEarned - taxPaid
  let realValueOfNetInterest := netInterestAfterTax / (1 + inflationRate)
  realValueOfNetInterest / netInterestPercentage

/-- Theorem stating that the selling price of a bond with given parameters is approximately $11,272.22 -/
theorem bond_selling_price_approx :
  let faceValue : ℝ := 10000
  let interestRate : ℝ := 0.07
  let taxRate : ℝ := 0.25
  let inflationRate : ℝ := 0.035
  let netInterestPercentage : ℝ := 0.045
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |bondSellingPrice faceValue interestRate taxRate inflationRate netInterestPercentage - 11272.22| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l878_87808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_for_normal_uric_acid_l878_87899

/-- The blood uric acid concentration model -/
noncomputable def uric_acid_model (U₀ K t : ℝ) : ℝ := -U₀ * Real.log (K * t)

/-- The theorem stating the minimum number of days to reach normal uric acid levels -/
theorem min_days_for_normal_uric_acid (U₀ K : ℝ) (h1 : U₀ = 20) 
  (h2 : uric_acid_model U₀ K 50 = 15) : 
  ∃ t' : ℕ, t' = 75 ∧ 
  (∀ t : ℕ, t < 75 → uric_acid_model U₀ K (t : ℝ) > 7) ∧
  uric_acid_model U₀ K (t' : ℝ) ≤ 7 := by
  sorry

#check min_days_for_normal_uric_acid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_days_for_normal_uric_acid_l878_87899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l878_87830

theorem complex_real_condition (m : ℂ) : 
  (m^2 + Complex.I) * (1 + m) ∈ Set.range (Complex.ofReal) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l878_87830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l878_87851

noncomputable def tank_capacity : ℝ := 5000
noncomputable def pipe_a_rate : ℝ := 200
noncomputable def pipe_b_rate : ℝ := 50
noncomputable def pipe_c_rate : ℝ := 25

noncomputable def cycle_fill_amount : ℝ := pipe_a_rate * 1 + pipe_b_rate * 2 - pipe_c_rate * 2

noncomputable def cycles_needed : ℝ := tank_capacity / cycle_fill_amount

noncomputable def cycle_duration : ℝ := 5  -- 1 min (A) + 2 min (B) + 2 min (C)

theorem tank_fill_time :
  cycles_needed * cycle_duration = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l878_87851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l878_87855

/-- Regular tetrahedron with side length 2 -/
structure RegularTetrahedron :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Plane parallel to a face of the tetrahedron -/
structure IntersectingPlane :=
  (tetrahedron : RegularTetrahedron)
  (is_parallel : Bool)
  (height_ratio : ℝ)
  (at_half_height : height_ratio = 1/2)

/-- Triangle formed by the intersection of the plane and the tetrahedron -/
def intersection_triangle (p : IntersectingPlane) : Set (ℝ × ℝ) := sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem intersection_triangle_area (t : RegularTetrahedron) (p : IntersectingPlane) :
  p.tetrahedron = t → p.is_parallel = true → 
  area (intersection_triangle p) = Real.sqrt 3 / 4 := by
  sorry

#check intersection_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_triangle_area_l878_87855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l878_87873

theorem exponential_inequality : (4 : ℝ)^(0.6 : ℝ) > (8 : ℝ)^(0.34 : ℝ) ∧ (8 : ℝ)^(0.34 : ℝ) > (1/2 : ℝ)^(-(0.9 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l878_87873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_after_exclusion_l878_87832

theorem class_average_after_exclusion
  (total_students : ℕ)
  (original_average : ℚ)
  (excluded_students : ℕ)
  (excluded_average : ℚ)
  (h1 : total_students = 13)
  (h2 : original_average = 72)
  (h3 : excluded_students = 5)
  (h4 : excluded_average = 40) :
  (total_students * original_average - excluded_students * excluded_average) /
    (total_students - excluded_students) = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_after_exclusion_l878_87832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l878_87878

/-- A linear function g satisfying g(c+1) - g(c) = 5 for all real c -/
noncomputable def g : ℝ → ℝ := sorry

/-- g is a linear function -/
axiom g_linear : IsLinearMap ℝ g

/-- g satisfies g(c+1) - g(c) = 5 for all real c -/
axiom g_increment (c : ℝ) : g (c + 1) - g c = 5

/-- Theorem: g(2) - g(7) = -25 -/
theorem g_difference : g 2 - g 7 = -25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l878_87878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_inequality_l878_87849

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (x - 1) else -(-x * (-x - 1))

-- State the theorem
theorem solution_set_for_inequality (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f (-x) = -f x) →  -- f is odd on [-1, 1]
  (∀ x < 0, f x = x * (x - 1)) →    -- definition of f for x < 0
  (f (1 - m) + f (1 - m^2) < 0 ↔ m ∈ Set.Ico 0 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_inequality_l878_87849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_surface_area_around_unit_sphere_l878_87896

-- Define the surface area of a cone as a function of its base radius
noncomputable def coneSurfaceArea (r : ℝ) : ℝ :=
  2 * Real.pi * r^4 / (r^2 - 1)

-- State the theorem
theorem min_cone_surface_area_around_unit_sphere :
  ∃ (r : ℝ), r > 1 ∧ coneSurfaceArea r = Real.pi * 8 ∧
  ∀ (s : ℝ), s > 1 → coneSurfaceArea s ≥ Real.pi * 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cone_surface_area_around_unit_sphere_l878_87896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisible_by_eight_l878_87814

def f (x : ℤ) : ℤ := x^3 + 2*x^2 + 3*x + 4

def T : Set ℤ := {t : ℤ | 0 ≤ t ∧ t ≤ 50}

theorem no_divisible_by_eight :
  ∀ t ∈ T, ¬(8 ∣ f t) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisible_by_eight_l878_87814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weak_multiple_exists_l878_87825

/-- A positive integer n is weak with respect to coprime positive integers a and b
    if there do not exist nonnegative integers x and y such that ax + by = n -/
def IsWeak (a b n : ℕ) : Prop :=
  ∀ x y : ℕ, a * x + b * y ≠ n

theorem weak_multiple_exists (a b n : ℕ) (h_coprime : Nat.Coprime a b)
    (h_weak : IsWeak a b n) (h_bound : n < a * b / 6) (h_pos : 0 < n) :
    ∃ k : ℕ, k ≥ 2 ∧ IsWeak a b (k * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weak_multiple_exists_l878_87825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_circle_not_consecutive_l878_87819

/-- Given a circle of 10^1000 natural numbers, it is impossible for the LCMs
    of adjacent pairs to form 10^1000 consecutive numbers. -/
theorem lcm_circle_not_consecutive : ∀ (a : Fin (10^1000) → ℕ+),
  ¬∃ (π : Equiv.Perm (Fin (10^1000))),
    ∀ (i : Fin (10^1000)),
      π i.succ = π i + 1 ∨ π i = π i.succ + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_circle_not_consecutive_l878_87819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l878_87889

open Real MeasureTheory Interval

theorem definite_integral_value : 
  ∫ x in (Set.Icc 0 1), (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_value_l878_87889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l878_87815

noncomputable def complex_number_z : ℂ := 1 / (1 - Complex.I)

theorem z_in_first_quadrant :
  complex_number_z.re > 0 ∧ complex_number_z.im > 0 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l878_87815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_volume_l878_87837

/-- A double pyramid formed by two regular tetrahedra connected by two faces -/
structure DoublePyramid (a : ℝ) where
  edge_length : a > 0

/-- A right triangular prism whose vertices are the centers of the six lateral faces of a double pyramid -/
structure TriangularPrism (a : ℝ) where
  base_pyramid : DoublePyramid a

/-- The volume of a TriangularPrism -/
noncomputable def volume (a : ℝ) (prism : TriangularPrism a) : ℝ :=
  (Real.sqrt 2 * a^3) / 54

/-- The volume of the triangular prism is (√2 * a³) / 54 -/
theorem triangular_prism_volume (a : ℝ) (prism : TriangularPrism a) :
  volume a prism = (Real.sqrt 2 * a^3) / 54 := by
  sorry

#check triangular_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_volume_l878_87837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l878_87810

/-- The fraction of a 6x6 grid covered by an equilateral triangle -/
theorem triangle_area_fraction : 
  let grid_size : ℕ := 6
  let vertex1 : Fin grid_size × Fin grid_size := (2, 2)
  let vertex2 : Fin grid_size × Fin grid_size := (4, 2)
  let vertex3 : Fin grid_size × Fin grid_size := (3, 4)
  let triangle_area := Real.sqrt 3
  let grid_area := (grid_size : ℝ) ^ 2
  (triangle_area / grid_area : ℝ) = Real.sqrt 3 / 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_fraction_l878_87810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l878_87854

/-- Given a function f: ℤ → ℤ satisfying certain properties, prove that f(0) = -2016 -/
theorem f_zero_value (f : ℤ → ℤ) 
  (h1 : ∀ n, f (f n) + f n = 2*n + 3) 
  (h2 : f 2016 = 2017) : 
  f 0 = -2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_value_l878_87854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l878_87876

-- Define the theorem
theorem sin_two_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α - Real.cos α = Real.sqrt 2) : 
  Real.sin (2 * α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_value_l878_87876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_result_l878_87871

def B : ℂ := 6 - 5*Complex.I
def N : ℂ := 3 + 2*Complex.I
def T : ℂ := -2*Complex.I
def Q : ℝ := 5

theorem complex_subtraction_result : B - N + T - Q = -2 - 9*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_subtraction_result_l878_87871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l878_87882

theorem square_measurement_error (x : ℝ) (e : ℝ) (h : (x + e)^2 = 1.21 * x^2) :
  abs (e / x - 0.105) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l878_87882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decode_sequence_l878_87885

/-- Represents the Russian alphabet encoding --/
def RussianEncoding : Char → Nat
| 'А' => 1
| 'Б' => 2
| 'Й' => 11
| 'К' => 12
| 'У' => 21
| 'Ф' => 22
| _ => 0  -- Default case for other characters

/-- Decodes a single number into a character using the Russian alphabet encoding --/
def decodeChar (n : Nat) : Char :=
  match n with
  | 1 => 'А'
  | 2 => 'Б'
  | 11 => 'Й'
  | 12 => 'К'
  | 21 => 'У'
  | 22 => 'Ф'
  | _ => ' '  -- Default case for other numbers

/-- Decodes a sequence of numbers into a string using the Russian alphabet encoding --/
def decode (s : List Nat) : String :=
  s.map decodeChar |>.foldl (· ++ ·.toString) ""

/-- The main theorem stating that the given sequence decodes to ФУФАЙКА --/
theorem decode_sequence :
  decode [22, 21, 22, 11, 12] = "ФУФАЙКА" := by
  sorry

#eval decode [22, 21, 22, 11, 12]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decode_sequence_l878_87885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_distances_l878_87880

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through a point with given inclination angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Curve defined by x^2 + y^2 - 6y = 0 -/
def Curve (p : Point) : Prop :=
  p.x^2 + p.y^2 - 6*p.y = 0

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem min_reciprocal_distances (P : Point) (l : Line) (A B : Point) :
  P.x = 1 ∧ P.y = 2 →
  l.point = P →
  Curve A ∧ Curve B →
  (∃ t : ℝ, A.x = P.x + t * Real.cos l.angle ∧ A.y = P.y + t * Real.sin l.angle) →
  (∃ t : ℝ, B.x = P.x + t * Real.cos l.angle ∧ B.y = P.y + t * Real.sin l.angle) →
  (1 / distance P A + 1 / distance P B) ≥ 2 * Real.sqrt 7 / 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_reciprocal_distances_l878_87880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_with_property_T_l878_87856

def hasPropertyT (B : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ B → b ∈ B → c ∈ B → a + b > c ∧ b + c > a ∧ c + a > b

theorem max_set_with_property_T :
  ∃ (B : Finset ℕ), B ⊆ Finset.range 2018 ∧ hasPropertyT B ∧ B.card = 1009 ∧
  ∀ (C : Finset ℕ), C ⊆ Finset.range 2018 → hasPropertyT C → C.card ≤ 1009 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_set_with_property_T_l878_87856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_achieves_max_l878_87806

/-- The function f(x) that takes the minimum of x^2, 6-x, and 2x+15 -/
noncomputable def f (x : ℝ) : ℝ := min (x^2) (min (6 - x) (2*x + 15))

/-- Theorem stating that the maximum value of f(x) is 9 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 9 := by
  intro x
  -- Proof steps would go here
  sorry

/-- Theorem stating that there exists an x for which f(x) = 9 -/
theorem f_achieves_max : ∃ x : ℝ, f x = 9 := by
  -- We can use x = -3 here
  use -3
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_achieves_max_l878_87806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bea_lemonade_price_l878_87826

/-- The price of Bea's lemonade per glass in cents -/
def B : ℕ := sorry

/-- The price of Dawn's lemonade per glass in cents -/
def D : ℕ := 28

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- The difference in earnings between Bea and Dawn in cents -/
def earnings_difference : ℕ := 26

theorem bea_lemonade_price :
  B * bea_glasses = D * dawn_glasses + earnings_difference →
  B = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bea_lemonade_price_l878_87826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l878_87822

def a (n : ℕ) : ℚ := 1 / ((n + 2) ^ 2 : ℚ)

def b : ℕ → ℚ
  | 0 => 1 - a 0
  | n + 1 => b n * (1 - a (n + 1))

theorem b_formula (n : ℕ) : b n = (n + 2 : ℚ) / (2 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l878_87822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l878_87840

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q : ℝ := sorry

/-- The first term of the geometric sequence -/
def a₁ : ℝ := sorry

theorem geometric_sequence_sum (h1 : S 4 = 24) (h2 : S 8 = 36) : S 12 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l878_87840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_bound_l878_87828

theorem polynomial_degree_bound (t : ℝ) (f : Polynomial ℝ) (n : ℕ) 
  (h_t : t ≥ 3) 
  (h_f : ∀ k : ℕ, k ≤ n → |f.eval (k : ℝ) - t^k| < 1) : 
  Polynomial.degree f ≥ n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_degree_bound_l878_87828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oxen_count_l878_87812

/-- Represents the number of oxen and months for each person -/
structure Grazing where
  oxen : ℕ
  months : ℕ

/-- The problem setup -/
def pasture_problem (a : Grazing) : Prop :=
  let total_rent : ℚ := 210
  let b := Grazing.mk 12 5
  let c := Grazing.mk 15 3
  let c_rent : ℚ := 54  -- Rounded from 53.99999999999999 for simplicity
  a.months = 7 ∧
  c_rent / total_rent = (c.oxen * c.months : ℚ) / ((a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) : ℚ)

/-- The theorem to prove -/
theorem oxen_count :
  ∃ (a : Grazing), pasture_problem a ∧ a.oxen = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oxen_count_l878_87812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_generation_l878_87874

/-- The probability density function f(x, y) = 6y -/
def f (x y : ℝ) : ℝ := 6 * y

/-- The region bounded by y=0, y=x, and x=1 -/
def R : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.2 ∧ p.2 ≤ p.1 ∧ p.1 ≤ 1}

/-- Uniform distribution on [0, 1] -/
def Uniform : Type := {r : ℝ // 0 ≤ r ∧ r ≤ 1}

theorem random_variable_generation 
  (X : Uniform → ℝ) 
  (Y : Uniform → Uniform → ℝ)
  (hX : ∀ u : Uniform, X u = u.val ^ (1/3))
  (hY : ∀ u v : Uniform, Y u v = (X u) * Real.sqrt v.val) :
  ∀ (u v : Uniform), (X u, Y u v) ∈ R ∧ 
  (∀ x y : ℝ, (x, y) ∈ R → f x y = 6 * y) :=
by
  sorry

#check random_variable_generation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_generation_l878_87874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_is_pi_half_l878_87853

open Real

/-- The function f(x) = sin(x + π/6) -/
noncomputable def f (x : ℝ) : ℝ := sin (x + π/6)

/-- The condition for symmetry about the x-axis after shifting -/
def is_symmetric (φ : ℝ) : Prop :=
  ∀ x, sin (x + π/6 + φ) = -sin (x + π/6 - φ)

/-- The minimum positive value of φ that satisfies the symmetry condition is π/2 -/
theorem min_phi_is_pi_half :
  ∃ φ, φ > 0 ∧ is_symmetric φ ∧ ∀ ψ, ψ > 0 → is_symmetric ψ → φ ≤ ψ :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_is_pi_half_l878_87853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_eq_neg_sqrt_three_l878_87811

/-- A function f is odd if f(-x) = -f(x) for all x in its domain. -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The given function f(x) = √3 cos(3x - θ) - sin(3x - θ) -/
noncomputable def f (θ : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.cos (3 * x - θ) - Real.sin (3 * x - θ)

theorem tan_theta_eq_neg_sqrt_three (θ : ℝ) 
    (h : IsOddFunction (f θ)) : Real.tan θ = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_eq_neg_sqrt_three_l878_87811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generadora_not_multiple_of_seven_l878_87801

def generadora_sequence : ℕ → ℕ
  | 0 => 2
  | 1 => 22
  | (n + 2) => 10^(n + 2) + 80

theorem generadora_not_multiple_of_seven :
  ∀ n : ℕ, ¬(7 ∣ generadora_sequence n) := by
  intro n
  cases n with
  | zero =>
    simp [generadora_sequence]
    norm_num
  | succ m =>
    cases m with
    | zero =>
      simp [generadora_sequence]
      norm_num
    | succ k =>
      simp [generadora_sequence]
      sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generadora_not_multiple_of_seven_l878_87801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l878_87861

noncomputable section

def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1)
def g (a : ℝ) (x : ℝ) := (1/3) * x^3 - a * x
def h (x : ℝ) := Real.exp x - 1
def F (a : ℝ) (x : ℝ) := h x - g a x

theorem problem_statement :
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≤ h x) ↔ a ≤ 1) ∧
  (∀ a : ℝ, (a ≤ -1 → ∀ x, x < 0 → F a x ≠ 0) ∧
            (a > -1 → ∃! x, x < 0 ∧ F a x = 0)) ∧
  (1095/1000 < Real.exp (1/10) ∧ Real.exp (1/10) < 3000/2699) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l878_87861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_pdf_l878_87843

/-- The function f(x) = a * exp(-|x|) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (-abs x)

/-- The integral of f(x) over the entire real line -/
noncomputable def integral (a : ℝ) : ℝ := ∫ (x : ℝ), f a x

/-- Theorem stating that f(x) is a probability density function when a = 1/2 -/
theorem f_is_pdf : 
  ∃ (a : ℝ), a > 0 ∧ integral a = 1 ∧ ∀ (x : ℝ), f a x ≥ 0 := by
  sorry

#check f_is_pdf

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_pdf_l878_87843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l878_87870

noncomputable def f (x : ℝ) : ℝ := min (4*x + 1) (min (x + 2) (-2*x + 4))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 8/3 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l878_87870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_period_scaling_monotonic_intervals_l878_87800

open Real

/-- Given functions f and g, prove that φ = π/6 when their graphs intersect at x = π/6 -/
theorem intersection_point (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = sin x - 1/2) →
  (∀ x, g x = cos (2*x + φ)) →
  0 ≤ φ ∧ φ < π/2 →
  f (π/6) = g (π/6) →
  φ = π/6 := by sorry

/-- Given function h obtained by scaling f, prove that ω = 2 when h has period π -/
theorem period_scaling (f h : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = sin x - 1/2) →
  (∀ x, h x = sin (ω*x) - 1/2) →
  ω > 0 →
  (∃ T > 0, ∀ x, h (x + T) = h x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, h (y + S) ≠ h y) →
  (π = T) →
  ω = 2 := by sorry

/-- Prove the intervals of monotonic increase for h(x) = sin(2x) - 1/2 -/
theorem monotonic_intervals (h : ℝ → ℝ) :
  (∀ x, h x = sin (2*x) - 1/2) →
  ∀ n : ℤ, StrictMonoOn h (Set.Icc (n*π - π/4) (n*π + π/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_period_scaling_monotonic_intervals_l878_87800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_twelfth_l878_87842

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := x^3

-- Define the area of the closed figure
noncomputable def area_between_curves : ℝ := ∫ x in (0)..(1), curve1 x - curve2 x

-- Theorem statement
theorem area_is_one_twelfth : area_between_curves = 1/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_twelfth_l878_87842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decorative_window_area_ratio_l878_87824

/-- Represents the dimensions and areas of a decorative window. -/
structure DecorativeWindow where
  /-- Width of the rectangle (AB) -/
  width : ℝ
  /-- Length of the rectangle (AD) -/
  length : ℝ
  /-- Ratio of length to width -/
  lengthWidthRatio : ℝ
  /-- Area of the rectangle -/
  rectangleArea : ℝ
  /-- Area of the semicircles -/
  semicirclesArea : ℝ
  /-- Area of the equilateral triangle -/
  triangleArea : ℝ

/-- Theorem about the ratio of areas in a decorative window -/
theorem decorative_window_area_ratio
  (window : DecorativeWindow)
  (h1 : window.width = 20)
  (h2 : window.lengthWidthRatio = 3 / 2)
  (h3 : window.length = window.width * window.lengthWidthRatio)
  (h4 : window.rectangleArea = window.length * window.width)
  (h5 : window.semicirclesArea = π * (window.width / 2)^2)
  (h6 : window.triangleArea = (Real.sqrt 3 / 4) * window.width^2) :
  window.rectangleArea / (window.semicirclesArea + window.triangleArea) = 6 / (π + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decorative_window_area_ratio_l878_87824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l878_87813

noncomputable section

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ  -- base angles
  b : ℝ  -- vertex angle
  sum_angles : a + a + b = 180
  positive_angles : 0 < a ∧ 0 < b

-- Define the interior angle positive value
def interior_angle_positive_value (t : IsoscelesTriangle) : ℝ :=
  max t.a t.b - min t.a t.b

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : interior_angle_positive_value t = 45) : 
  t.b = 90 ∨ t.b = 30 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_vertex_angle_l878_87813
