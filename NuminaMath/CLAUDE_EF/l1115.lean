import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_sum_l1115_111592

/-- Given a sequence of 5 consecutive odd numbers with a sum of 275,
    prove that the numbers are 51, 53, 55, 57, and 59. -/
theorem consecutive_odd_numbers_sum (a b c d e : ℕ) : 
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧  -- All numbers are odd
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2 ∧  -- Consecutive
  a + b + c + d + e = 275 →  -- Sum is 275
  a = 51 ∧ b = 53 ∧ c = 55 ∧ d = 57 ∧ e = 59 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_odd_numbers_sum_l1115_111592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l1115_111539

theorem quadratic_equation_roots (m : ℝ) :
  (∀ x : ℝ, m * x^2 + 3 * x + 2 = 0 → x ∈ Set.univ) →
  (m ≤ 9/8 ∧ m ≠ 0) ∧
  (m ∈ Set.Ioi 0 ∩ Set.Icc 1 1 → m = 1) ∧
  (m = 1 → ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -2 ∧ m * x₁^2 + 3 * x₁ + 2 = 0 ∧ m * x₂^2 + 3 * x₂ + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l1115_111539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guess_probability_secret_number_count_l1115_111554

/-- A secret number between 200 and 300 with an even tens digit and an odd units digit -/
def SecretNumber : Type :=
  { n : ℕ // 200 ≤ n ∧ n < 300 ∧ (n / 10) % 2 = 0 ∧ n % 2 = 1 }

/-- The number of valid secret numbers -/
def numSecretNumbers : ℕ := 10

/-- The probability of guessing the correct secret number -/
theorem guess_probability : (numSecretNumbers : ℚ)⁻¹ = (10 : ℚ)⁻¹ := by
  -- Proof goes here
  sorry

/-- The set of all valid secret numbers -/
def allSecretNumbers : Finset ℕ :=
  Finset.filter (fun n => 200 ≤ n ∧ n < 300 ∧ (n / 10) % 2 = 0 ∧ n % 2 = 1) (Finset.range 300)

/-- The cardinality of the set of all valid secret numbers is 10 -/
theorem secret_number_count : allSecretNumbers.card = 10 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_guess_probability_secret_number_count_l1115_111554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l1115_111506

/-- The number of integer triples satisfying the given conditions -/
def solution_count : ℕ := 4

/-- Predicate defining the conditions for a triple (a, b, c) -/
def satisfies_conditions (a b c : ℤ) : Prop :=
  (a + b).natAbs + c = 21 ∧ a * b + c.natAbs = 99

/-- Theorem stating that there are exactly 4 triples satisfying the conditions -/
theorem count_solutions :
  (∃! (s : Finset (ℤ × ℤ × ℤ)), s.card = solution_count ∧ 
    ∀ (t : ℤ × ℤ × ℤ), t ∈ s ↔ satisfies_conditions t.1 t.2.1 t.2.2) :=
by sorry

#check count_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l1115_111506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1115_111563

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry  -- We don't know the full definition of f
def g (m : ℝ) : ℝ → ℝ := λ x ↦ x^2 - 2*x + m

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x) →  -- f is odd on [-2, 2]
  (∀ x ∈ Set.Ioo 0 2, f x = 2^x - 1) →  -- f(x) = 2^x - 1 for x ∈ (0, 2]
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, g m x₂ = f x₁) →  -- Condition on g and f
  m ∈ Set.Icc (-5 : ℝ) (-2) :=
by
  sorry  -- The proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1115_111563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_expressions_count_l1115_111545

theorem equal_expressions_count (x : ℝ) (h : x > 0) : 
  (∃! n : ℕ, n = (if 2 * x^x = x^x + x^x then 1 else 0) + 
              (if x^(x+1) = x^x + x^x then 1 else 0) + 
              (if (x+1)^x = x^x + x^x then 1 else 0) + 
              (if (2*x)^(2*x) = x^x + x^x then 1 else 0)) ∧ 
  (∃! expr : ℝ → ℝ, (∀ y > 0, expr y = y^y + y^y) ∧ 
    ((expr = λ y => 2 * y^y) ∨ 
     (expr = λ y => y^(y+1)) ∨ 
     (expr = λ y => (y+1)^y) ∨ 
     (expr = λ y => (2*y)^(2*y)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_expressions_count_l1115_111545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1115_111537

theorem repeating_decimal_sum (a b : ℕ+) (h1 : (56 : ℚ) / 99 = a / b) (h2 : Nat.gcd a.val b.val = 1) : 
  a.val + b.val = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1115_111537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_max_sum_of_products_l1115_111510

/-- Represents a labeling of a cube's faces -/
structure CubeLabeling where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  valid : Finset.toSet {a, b, c, d, e, f} = {3, 4, 5, 6, 7, 8}
  opposite_small : (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)

/-- Computes the sum of products for a given cube labeling -/
def sumOfProducts (l : CubeLabeling) : Nat :=
  (l.a + l.b) * (l.c + l.d) * (l.e + l.f)

/-- The maximum sum of products for any valid cube labeling -/
def maxSumOfProducts : Nat := 1183

theorem cube_max_sum_of_products :
  ∀ l : CubeLabeling, sumOfProducts l ≤ maxSumOfProducts := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_max_sum_of_products_l1115_111510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_convergence_l1115_111561

/-- The infinite series ∑_{n=1}^∞ (n^3 + 2n^2 - 3) / ((n+3)!) converges to 1/4 -/
theorem infinite_series_convergence :
  ∑' n : ℕ+, (((n : ℝ)^3 + 2*(n : ℝ)^2 - 3) / (Nat.factorial (n + 3) : ℝ)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_convergence_l1115_111561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_volume_l1115_111514

-- Define the parameters of the problem
noncomputable def cylinder_radius : ℝ := 8
noncomputable def cylinder_height : ℝ := 16
noncomputable def wedge_angle : ℝ := 45
noncomputable def cone_radius : ℝ := 8
noncomputable def cone_height : ℝ := 16

-- Define the volume of the cylindrical wedge
noncomputable def cylindrical_wedge_volume : ℝ := 
  (cylinder_radius^2 * cylinder_height * Real.pi) / 2

-- Define the volume of the conical cap
noncomputable def conical_cap_volume : ℝ := 
  (1/3) * Real.pi * cone_radius^2 * cone_height

-- Theorem statement
theorem combined_solid_volume :
  cylindrical_wedge_volume + conical_cap_volume = (2560/3) * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_solid_volume_l1115_111514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_approx_dihedral_angle_exact_l1115_111574

-- Define the dihedral angle
noncomputable def dihedral_angle (r1 r2 : ℝ) (angle : ℝ) : ℝ :=
  Real.arccos (Real.sqrt ((1 + 1 / Real.sqrt 2) / 2))

-- State the theorem
theorem dihedral_angle_approx :
  ∀ (r1 r2 : ℝ), r1 > 0 → r2 > 0 → r1 = 1.5 * r2 →
  let angle := π / 4  -- 45 degrees in radians
  let result := dihedral_angle r1 r2 angle
  abs (Real.cos result - 0.84) < 0.01 := by
  sorry

-- Additional lemma to show the exact value
theorem dihedral_angle_exact :
  ∀ (r1 r2 : ℝ), r1 > 0 → r2 > 0 → r1 = 1.5 * r2 →
  let angle := π / 4  -- 45 degrees in radians
  let result := dihedral_angle r1 r2 angle
  Real.cos result = Real.sqrt ((1 + 1 / Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_approx_dihedral_angle_exact_l1115_111574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_third_quadrant_l1115_111532

theorem cosine_value_in_third_quadrant (α : Real) 
  (h1 : Real.tan α = 3 / 4) 
  (h2 : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.cos α = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_in_third_quadrant_l1115_111532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_diagonals_l1115_111599

/-- The interior angle of a vertex in a convex polygon -/
def interior_angle (n : ℕ) (i : Fin n) : ℝ := sorry

/-- Predicate indicating if a polygon with n sides is convex -/
def convex_polygon (n : ℕ) : Prop := sorry

/-- Function to calculate the number of diagonals in a polygon with n sides -/
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with interior angles of 150° has 54 diagonals -/
theorem convex_polygon_diagonals : 
  ∀ (n : ℕ), 
  (n > 2) → 
  (∀ i : Fin n, interior_angle n i = 150) → 
  convex_polygon n → 
  number_of_diagonals n = 54 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_diagonals_l1115_111599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2023rd_term_l1115_111560

def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => (mySequence n + mySequence (n + 1)) % 10

theorem last_digit_of_2023rd_term : mySequence 2022 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2023rd_term_l1115_111560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_l1115_111577

theorem prime_sequence (p : ℕ) (f : ℕ → ℕ) 
  (h_def : ∀ x, f x = x^2 + x + p)
  (h_prime : ∀ k : ℕ, k ≤ ⌊Real.sqrt (p / 3)⌋ → Nat.Prime (f k)) :
  ∀ n : ℕ, n < p - 1 → Nat.Prime (f n) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_l1115_111577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_angle_range_l1115_111523

noncomputable def angle_between_vectors (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem point_on_circle_angle_range (x₀ : ℝ) :
  (∃ (N : ℝ × ℝ), N.1^2 + N.2^2 = 1 ∧ 
    angle_between_vectors (N.1, N.2) (x₀, 1) = 30 * π / 180) →
  x₀ ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_angle_range_l1115_111523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_real_and_rational_l1115_111516

theorem circle_equation_real_and_rational :
  (∃ S : Set (ℝ × ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ 
      0 ≤ x ∧ x ≤ Real.sqrt 3 ∧ 
      0 ≤ y ∧ y ≤ Real.sqrt 3 ∧ 
      x * Real.sqrt (3 - y^2) + y * Real.sqrt (3 - x^2) = 3) ∧
    Set.Infinite S) ∧
  (∀ (x y : ℚ), 
    ¬(0 ≤ (x : ℝ) ∧ (x : ℝ) ≤ Real.sqrt 3 ∧ 
      0 ≤ (y : ℝ) ∧ (y : ℝ) ≤ Real.sqrt 3 ∧ 
      (x : ℝ) * Real.sqrt (3 - (y : ℝ)^2) + (y : ℝ) * Real.sqrt (3 - (x : ℝ)^2) = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_real_and_rational_l1115_111516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_solution_system_l1115_111522

theorem prime_solution_system (a b c : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c →
  (2 * a - b + 7 * c = 1826) ∧ 
  (3 * a + 5 * b + 7 * c = 2007) →
  a = 7 ∧ b = 29 ∧ c = 263 := by
  sorry

#check prime_solution_system

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_solution_system_l1115_111522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_proof_l1115_111546

noncomputable def trapezium_area (a b l : ℝ) (θ : ℝ) : ℝ :=
  (1/2) * (a + b) * (l * Real.sin θ)

theorem trapezium_area_proof (a b l θ : ℝ) 
  (ha : a = 20)
  (hb : b = 10)
  (hl : l = 10)
  (hθ : θ = 30 * π / 180) :
  trapezium_area a b l θ = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_proof_l1115_111546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nonneg_necessary_not_sufficient_l1115_111591

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x in the domain of f -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = a*exp(x) + ln(x^2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp x + Real.log (x^2)

/-- Theorem stating that a ≥ 0 is a necessary but not sufficient condition for f to be even -/
theorem a_nonneg_necessary_not_sufficient :
  (∃ a : ℝ, a ≥ 0 ∧ ¬IsEven (f a)) ∧
  (∀ a : ℝ, IsEven (f a) → a ≥ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_nonneg_necessary_not_sufficient_l1115_111591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_neg_three_min_area_triangle_OAB_l1115_111579

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := -1/3 * x^3 + 2*x^2 - 3*x + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- Theorem for tangent lines with slope -3 -/
theorem tangent_lines_slope_neg_three :
  ∃ (x₁ x₂ : ℝ), 
    f' x₁ = -3 ∧ f' x₂ = -3 ∧ 
    (3*x₁ + f x₁ - 1 = 0 ∨ 9*x₁ + 3*(f x₁) - 35 = 0) ∧
    (3*x₂ + f x₂ - 1 = 0 ∨ 9*x₂ + 3*(f x₂) - 35 = 0) ∧
    x₁ ≠ x₂ :=
by sorry

/-- Theorem for minimum area of triangle OAB -/
theorem min_area_triangle_OAB :
  ∃ (x : ℝ), f' x = 1 ∧ 
    (∀ k < 0, 
      let xA := x - 1/(3*k)
      let yB := f x - k*x
      (1/2 * xA * yB) ≥ 4/3) ∧
    (∃ k < 0, 
      let xA := x - 1/(3*k)
      let yB := f x - k*x
      (1/2 * xA * yB) = 4/3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_slope_neg_three_min_area_triangle_OAB_l1115_111579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_EFGH_l1115_111578

/-- Rectangle EFGH with given coordinates -/
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  H : ℝ × ℝ

/-- The rectangle EFGH with given coordinates -/
def rectangleEFGH : Rectangle where
  E := (-4, 3)
  F := (996, 43)
  H := (-2, -47)

/-- The area of a rectangle given two adjacent sides -/
def rectangleArea (side1 side2 : ℝ) : ℝ :=
  side1 * side2

/-- Theorem stating that the area of rectangle EFGH is 50050 -/
theorem area_of_rectangle_EFGH :
  rectangleArea
    (((rectangleEFGH.F.1 - rectangleEFGH.E.1)^2 + (rectangleEFGH.F.2 - rectangleEFGH.E.2)^2).sqrt)
    (((rectangleEFGH.H.1 - rectangleEFGH.E.1)^2 + (rectangleEFGH.H.2 - rectangleEFGH.E.2)^2).sqrt) = 50050 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_rectangle_EFGH_l1115_111578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_preserving_map_properties_l1115_111595

open Matrix
open LinearMap

variable {n : ℕ}

/-- A linear map on complex matrices preserving determinants -/
def DetPreservingMap (T : Matrix (Fin n) (Fin n) ℂ →ₗ[ℂ] Matrix (Fin n) (Fin n) ℂ) : Prop :=
  ∀ A : Matrix (Fin n) (Fin n) ℂ, Complex.abs (det A) = Complex.abs (det (T A))

theorem det_preserving_map_properties
  (T : Matrix (Fin n) (Fin n) ℂ →ₗ[ℂ] Matrix (Fin n) (Fin n) ℂ)
  (h : DetPreservingMap T) :
  (∀ A : Matrix (Fin n) (Fin n) ℂ, T A = 0 → A = 0) ∧
  (∀ A : Matrix (Fin n) (Fin n) ℂ, Matrix.rank A = Matrix.rank (T A)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_preserving_map_properties_l1115_111595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_pricing_theorem_l1115_111596

/- Define the tiered electricity pricing system -/
def tier1_limit : ℝ := 240
def tier2_limit : ℝ := 400

noncomputable def electricity_price (a : ℝ) (usage : ℝ) : ℝ :=
  if usage ≤ tier1_limit then
    a * usage
  else if usage ≤ tier2_limit then
    a * tier1_limit + 0.65 * (usage - tier1_limit)
  else
    a * tier1_limit + 0.65 * (tier2_limit - tier1_limit) + (a + 0.3) * (usage - tier2_limit)

/- Define Mr. Li's electricity usage data -/
def october_usage : ℝ := 200
def october_payment : ℝ := 120
def september_payment : ℝ := 157
def august_avg_price : ℝ := 0.7

/- Theorem to prove -/
theorem electricity_pricing_theorem :
  ∃ (a : ℝ) (september_usage : ℝ) (august_usage : ℝ),
    electricity_price a october_usage = october_payment ∧
    electricity_price a september_usage = september_payment ∧
    electricity_price a august_usage / august_usage = august_avg_price ∧
    a = 0.6 ∧
    september_usage = 260 ∧
    august_usage = 560 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_pricing_theorem_l1115_111596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_of_squares_l1115_111534

variable (p q r s : ℝ)

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2*p, 2*q; 2*r, 2*s]

theorem matrix_sum_of_squares (h : Matrix.transpose B = 4 * B⁻¹) :
  p^2 + q^2 + r^2 + s^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_of_squares_l1115_111534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1115_111567

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  Real.cos B / Real.cos C = -b / (2*a + c) →
  b = Real.sqrt 13 →
  a + c = 4 →
  B = 2*π/3 ∧ 
  (1/2) * a * c * Real.sin B = 3*Real.sqrt 3/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1115_111567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l1115_111549

/-- Given a rectangle ABCD with side lengths AB = 2 and AD = 4, and a point E on AD with AE = 3,
    the cosine of the dihedral angle formed by folding triangle ABE and triangle DCE along BE and CE
    respectively (so that D falls on AE) is 7/8. -/
theorem dihedral_angle_cosine (A B C D E : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let AE := Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2)
  AB = 2 ∧ AD = 4 ∧ AE = 3 ∧ E ∈ Set.Icc A D →
  let dihedral_angle := sorry -- Definition of dihedral angle in terms of A, B, C, D, E
  Real.cos dihedral_angle = 7/8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_cosine_l1115_111549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1115_111508

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (n + 2) => (Real.sqrt 6 - Real.sqrt 2) / 2 * sequence_a (n + 1) - sequence_a n

theorem a_2019_value : sequence_a 2019 = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1115_111508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_product_magnitude_l1115_111552

/-- Vector product of two 2D vectors -/
def vector_product (a b : ℝ × ℝ) : ℝ := sorry

/-- Magnitude of a 2D vector -/
def magnitude (v : ℝ × ℝ) : ℝ := sorry

/-- Angle between two 2D vectors -/
noncomputable def angle (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_product_magnitude :
  let u : ℝ × ℝ := (2, 0)
  let v : ℝ × ℝ := (1, Real.sqrt 3)
  (∀ a b : ℝ × ℝ, vector_product a b = magnitude a * magnitude b * Real.sin (angle a b)) →
  magnitude (u + v) = 2 * Real.sqrt 3 ∧ vector_product u (u + v) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_product_magnitude_l1115_111552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rhombus_diagonal_l1115_111547

-- Define the function for the diagonal of the largest inscribed rhombus
def diagonal_of_largest_inscribed_rhombus (r : ℝ) : ℝ := 2 * r

theorem largest_rhombus_diagonal (r : ℝ) (h : r = 10) : 
  diagonal_of_largest_inscribed_rhombus r = 2 * r :=
by
  -- Unfold the definition of diagonal_of_largest_inscribed_rhombus
  unfold diagonal_of_largest_inscribed_rhombus
  -- The goal is now to prove 2 * r = 2 * r, which is true by reflexivity
  rfl

#check largest_rhombus_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rhombus_diagonal_l1115_111547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1115_111588

/-- The principal amount for a compound interest calculation -/
noncomputable def principal (A : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  A / (1 + r) ^ t

/-- The given amount after interest -/
def amount : ℝ := 1792

/-- The given annual interest rate -/
def rate : ℝ := 0.05

/-- The given time period in years -/
def time : ℝ := 2.4

theorem principal_calculation :
  abs (principal amount rate time - 1590.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1115_111588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_radius_l1115_111571

theorem swimming_pool_radius (r : ℝ) :
  (r > 0) →
  (π * ((r + 4)^2 - r^2) = (11/25) * π * r^2) →
  r = 20 := by
  intros h_pos h_eq
  -- Proof steps would go here
  sorry

#check swimming_pool_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_radius_l1115_111571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l1115_111565

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Checks if a natural number has no perfect cube factors -/
def no_perfect_cube_factors (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k^3 ∣ n → k = 1

theorem sphere_diameter_theorem :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ no_perfect_cube_factors b ∧
  (∃ (r : ℝ), sphere_volume r = 3 * sphere_volume 6 ∧
              2 * r = a * Real.rpow b (1/3)) ∧
  a + b = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l1115_111565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1115_111582

/-- Given vectors a, b, and c in ℝ², prove properties about b and c -/
theorem vector_properties (a b c : ℝ × ℝ) : 
  (a = (1, 2)) →
  (‖b‖ = 3 * Real.sqrt 5) →
  (∃ (k : ℝ), b = k • a) →
  (a.1 * c.1 + a.2 * c.2 = -(Real.sqrt 5 / 10) * ‖a‖ * ‖c‖) →
  ((a.1 + c.1, a.2 + c.2) • (a.1 - 9 * c.1, a.2 - 9 * c.2) = 0) →
  ((b = (3, 6) ∨ b = (-3, -6)) ∧ ‖c‖ = 1) := by
  sorry

#check vector_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1115_111582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_exists_l1115_111507

/-- Proves the existence of an annual interest rate satisfying given compound interest conditions -/
theorem compound_interest_rate_exists : ∃ (r : ℝ) (P : ℝ), 
  2420 = P * (1 + r / 12) ^ (12 * 2) ∧ 
  3146 = P * (1 + r / 12) ^ (12 * 3) ∧ 
  |r - 0.2676| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_exists_l1115_111507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_of_n_l1115_111515

def n : ℕ := 2^3 * 3^2 * 5 * 7

theorem count_even_factors_of_n :
  (Finset.filter (fun x => x ∣ n ∧ Even x) (Finset.range (n + 1))).card = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_factors_of_n_l1115_111515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1115_111544

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3 + φ)

theorem function_transformation (φ : ℝ) 
  (h1 : |φ| < Real.pi / 2) 
  (h2 : ∀ x, f (x + Real.pi / 2) φ = f (-x) φ) : 
  ∀ x, f x φ = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1115_111544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1115_111521

/-- Given two circles C₁ and C₂, and a point P(a, b) that forms equal tangent segments to both circles,
    prove that the minimum value of a² + b² - 6a - 4b + 13 is 8/5 -/
theorem min_value_of_expression : 
  ∃ (a b : ℝ), 
  let C₁ := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let C₂ := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 4}
  let P := (a, b)
  ∃ (M N : ℝ × ℝ), M ∈ C₁ ∧ N ∈ C₂ ∧
    (dist P M = dist P N) ∧ 
    (∀ (a' b' : ℝ), a'^2 + b'^2 - 6*a' - 4*b' + 13 ≥ 8/5) ∧
    (a^2 + b^2 - 6*a - 4*b + 13 = 8/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1115_111521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_change_of_base_l1115_111583

theorem log_change_of_base (N a b : ℝ) 
  (hN : N > 0) (ha : a > 0) (hb : b > 0) (ha_neq_1 : a ≠ 1) (hb_neq_1 : b ≠ 1) : 
  Real.log N / Real.log b = Real.log N / Real.log a / (Real.log b / Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_change_of_base_l1115_111583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_highest_second_l1115_111527

variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
def prob_ordered := 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ ≤ 1

-- Define the probability of winning two consecutive games for each scenario
def P_A (p₁ p₂ p₃ : ℝ) := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B (p₁ p₂ p₃ : ℝ) := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C (p₁ p₂ p₃ : ℝ) := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- State the theorem
theorem max_prob_highest_second (h : prob_ordered p₁ p₂ p₃) :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_highest_second_l1115_111527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l1115_111512

theorem triangle_angle_inequalities (α β γ : Real) (R r : Real) 
  (h_triangle : α + β + γ = Real.pi)
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_circumradius : R > 0)
  (h_inradius : r > 0)
  (h_cosine_sum : Real.cos α + Real.cos β + Real.cos γ = (R + r) / R)
  (h_radius_inequality : r ≤ R / 2) : 
  (1 < Real.cos α + Real.cos β + Real.cos γ ∧ Real.cos α + Real.cos β + Real.cos γ ≤ 3/2) ∧
  (1 < Real.sin (α/2) + Real.sin (β/2) + Real.sin (γ/2) ∧ 
   Real.sin (α/2) + Real.sin (β/2) + Real.sin (γ/2) ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequalities_l1115_111512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1115_111597

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x

-- State the theorem
theorem unique_a_value (a : ℝ) : 
  (∀ x > 0, x * (exp x - a - 1) - f a x ≥ 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l1115_111597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_proof_l1115_111584

/-- Represents the relationship between number of men and days for a construction task -/
structure ConstructionTask where
  men : ℕ
  days : ℝ
  work : ℝ

/-- The work is inversely proportional between men and days -/
axiom inverse_proportion (t1 t2 : ConstructionTask) : 
  t1.work = t2.work → t1.men * t1.days = t2.men * t2.days

/-- Given initial conditions -/
def initial_task : ConstructionTask := { men := 18, days := 6, work := 18 * 6 }

/-- The task we want to prove -/
def target_task : ConstructionTask := { men := 30, days := 3.6, work := 18 * 6 }

/-- Theorem stating that 30 men will take 3.6 days to complete the task -/
theorem construction_time_proof : 
  target_task.days = 3.6 := by
  have h : initial_task.work = target_task.work := rfl
  have inv_prop := inverse_proportion initial_task target_task h
  calc
    target_task.days = (initial_task.men * initial_task.days) / target_task.men := by sorry
    _ = (18 * 6) / 30 := by sorry
    _ = 108 / 30 := by sorry
    _ = 3.6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_time_proof_l1115_111584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l1115_111526

/-- The area of a triangle in the complex plane given by three points --/
noncomputable def triangleArea (z1 z2 z3 : ℂ) : ℝ :=
  (1/2) * abs (z1.re * z2.im + z2.re * z3.im + z3.re * z1.im -
               z1.im * z2.re - z2.im * z3.re - z3.im * z1.re)

/-- The smallest positive integer n such that the area of the triangle
    formed by (n + i), (n + i)³, and (n + i)⁴ is greater than 3000 --/
theorem smallest_n_for_triangle_area : 
  (∀ k : ℕ, k > 0 → k < 10 → 
    triangleArea (k + I) ((k + I)^3) ((k + I)^4) ≤ 3000) ∧ 
  triangleArea (10 + I) ((10 + I)^3) ((10 + I)^4) > 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l1115_111526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_seventh_l1115_111524

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Define the expression
noncomputable def expression : ℚ :=
  (ceiling ((43 : ℝ) / 13 - ceiling ((45 : ℝ) / 29))) /
  (ceiling ((56 : ℝ) / 13 + ceiling (((13 : ℝ) * 29) / 45)))

-- Theorem statement
theorem expression_equals_one_seventh : expression = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_seventh_l1115_111524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1115_111589

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : 2 * abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B + abc.a * Real.sin abc.B)
  (h2 : abc.b = Real.sqrt 13)
  (h3 : abc.a + abc.c = 5) :
  abc.B = π / 3 ∧ Triangle.area abc = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1115_111589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_between_circles_and_x_axis_l1115_111528

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
noncomputable def areaOfRegion (c1 c2 : Circle) : ℝ :=
  25 - 8.5 * Real.pi

/-- Theorem stating that the area of the region bound by the given circles and the x-axis
    is equal to 25 - 8.5π square units -/
theorem area_of_region_between_circles_and_x_axis :
  let c1 : Circle := { center := (5, 5), radius := 3 }
  let c2 : Circle := { center := (10, 5), radius := 5 }
  areaOfRegion c1 c2 = 25 - 8.5 * Real.pi :=
by
  sorry

#check area_of_region_between_circles_and_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_between_circles_and_x_axis_l1115_111528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1115_111500

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_point_x_coordinate 
  (x y : ℝ) (h1 : parabola x y) (h2 : distance (x, y) focus = 20) : 
  x = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1115_111500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_f_solutions_in_interval_l1115_111568

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

-- Theorem for the increasing intervals
theorem f_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := by
  sorry

-- Theorem for the solutions of f(x) = 0 in (0, π]
theorem f_solutions_in_interval :
  {x : ℝ | x ∈ Set.Ioo 0 Real.pi ∧ f x = 0} = {5 * Real.pi / 12, 11 * Real.pi / 12} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_f_solutions_in_interval_l1115_111568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l1115_111598

theorem count_integers_in_pi_range : 
  (Finset.range (Int.toNat (Int.floor (12 * Real.pi) + 1 - Int.ceil (-6 * Real.pi)))).card = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_pi_range_l1115_111598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l1115_111518

theorem units_digit_of_sum : (42^4 + 24^4) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_l1115_111518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_twice_min_chord_line_l1115_111566

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def lineL (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the intersection of a line and a circle
def intersects (m : ℝ) : Prop := ∃ (x y : ℝ), circleC x y ∧ lineL m x y

-- Theorem 1: The line always intersects the circle at two points
theorem line_intersects_circle_twice :
  ∀ m : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    circleC p1.1 p1.2 ∧ circleC p2.1 p2.2 ∧ 
    lineL m p1.1 p1.2 ∧ lineL m p2.1 p2.2 :=
by sorry

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b

-- Theorem 2: The slope-intercept form of the line with minimum chord length
theorem min_chord_line :
  ∃! (m b : ℝ), (∀ x y : ℝ, slope_intercept_form m b x y ↔ lineL ((m - 1) / (2*m + 1)) x y) ∧
    m = 2 ∧ b = -5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_twice_min_chord_line_l1115_111566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_game_prob_diff_l1115_111556

/-- Casey's random walk game --/
structure CaseyGame where
  /-- The current number --/
  current : ℕ
  /-- The probability of reaching a specific number --/
  prob_reach : ℕ → ℝ

/-- The probability of reaching number n in Casey's game --/
noncomputable def prob_reach (n : ℕ) : ℝ :=
  1 + (1 / 3) * (-1 / 2) ^ n

/-- Theorem stating the difference between probabilities of reaching 20 and 15 --/
theorem casey_game_prob_diff :
  prob_reach 20 - prob_reach 15 = 11 / 2^20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_game_prob_diff_l1115_111556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_irreducible_l1115_111580

theorem sum_irreducible (a b : ℕ) (h : Nat.Coprime a b) (ha : a > 0) (hb : b > 0) :
  let sum := (2 * a + b : ℚ) / (a * (a + b))
  Irreducible sum := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_irreducible_l1115_111580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_coefficient_sum_l1115_111557

/-- Given a function g(x) with vertical asymptotes, find the sum of coefficients a and b -/
theorem asymptote_coefficient_sum :
  ∀ (a b : ℝ),
  let g := λ x : ℝ ↦ (x + 5) / (x^2 + a*x + b)
  (∀ x ≠ 2, g x ≠ 0) →
  (∀ x ≠ -3, g x ≠ 0) →
  (∀ x, x ≠ 2 ∧ x ≠ -3 → g x ≠ 0) →
  a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_coefficient_sum_l1115_111557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_second_half_time_l1115_111542

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  initialSpeed : ℝ
  halfwayPoint : ℝ
  secondHalfExtraTime : ℝ

/-- Calculates the time taken for the second half of the journey --/
noncomputable def secondHalfTime (journey : RunnerJourney) : ℝ :=
  journey.halfwayPoint / (journey.initialSpeed / 2)

/-- Theorem stating the conditions and the result to be proved --/
theorem runner_second_half_time (journey : RunnerJourney) 
  (h1 : journey.totalDistance = 40)
  (h2 : journey.halfwayPoint = journey.totalDistance / 2)
  (h3 : journey.halfwayPoint / journey.initialSpeed + journey.secondHalfExtraTime = 
        journey.halfwayPoint / (journey.initialSpeed / 2))
  (h4 : journey.secondHalfExtraTime = 8) :
  secondHalfTime journey = 16 := by
  sorry

#check runner_second_half_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_second_half_time_l1115_111542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bowl_weight_correct_l1115_111586

/-- Calculates the weight of an empty cat bowl given the filling schedule, daily food amount, and weight after some food has been eaten. -/
def empty_bowl_weight
  (fill_days : ℕ)             -- Number of days between fills
  (daily_food : ℕ)            -- Grams of food given per day
  (total_weight : ℕ)          -- Weight of bowl with remaining food
  (eaten_food : ℕ)            -- Grams of food eaten
  : ℕ :=
  let total_food := fill_days * daily_food
  let remaining_food := total_food - eaten_food
  total_weight - remaining_food

theorem empty_bowl_weight_correct
  (fill_days : ℕ)
  (daily_food : ℕ)
  (total_weight : ℕ)
  (eaten_food : ℕ)
  (h1 : fill_days = 3)        -- Bowl is filled every 3 days
  (h2 : daily_food = 60)      -- 60 grams of food per day
  (h3 : total_weight = 586)   -- Bowl weighs 586 grams after some food was eaten
  (h4 : eaten_food = 14)      -- Cat ate 14 grams
  : empty_bowl_weight fill_days daily_food total_weight eaten_food = 420 := by
  sorry

#eval empty_bowl_weight 3 60 586 14  -- Should output 420

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bowl_weight_correct_l1115_111586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_integers_l1115_111590

open Real

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ := (2 * Real.sqrt 2 * Real.cos (25 * π / 180) - 1) * Real.tan (x * π / 180)

-- Define the right-hand side of the equation
noncomputable def rhs (x : ℝ) : ℝ := (2 * Real.sqrt 2 * Real.sin (25 * π / 180) - 1) * Real.tan (3 * x * π / 180)

-- State the theorem
theorem solution_integers (x : ℝ) :
  lhs x = rhs x →
  (∃ k : ℤ, x = 180 * ↑k) ∨ (∃ n : ℤ, x = 180 * ↑n + 25 ∨ x = 180 * ↑n - 25) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_integers_l1115_111590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_three_nonparallel_lines_perpendicular_to_three_hexagon_sides_l1115_111520

-- Define the basic structures
structure Line : Type
structure Plane : Type

-- Define the perpendicular relation
def perpendicular : Line → Plane → Prop := sorry

-- Define the perpendicular relation between two lines
def perpendicularLines : Line → Line → Prop := sorry

-- Define the condition of being within a plane
def within : Line → Plane → Prop := sorry

-- Define the parallel relation between lines
def parallel : Line → Line → Prop := sorry

-- Define a regular hexagon
structure RegularHexagon : Type where
  plane : Plane
  sides : Fin 6 → Line

-- Theorem 1
theorem perpendicular_to_three_nonparallel_lines
  (l : Line) (p : Plane) (l1 l2 l3 : Line)
  (h1 : within l1 p ∧ within l2 p ∧ within l3 p)
  (h2 : ¬ parallel l1 l2 ∧ ¬ parallel l2 l3 ∧ ¬ parallel l1 l3)
  (h3 : perpendicularLines l l1 ∧ perpendicularLines l l2 ∧ perpendicularLines l l3) :
  perpendicular l p :=
sorry

-- Theorem 2
theorem perpendicular_to_three_hexagon_sides
  (l : Line) (h : RegularHexagon) (s1 s2 s3 : Fin 6) (p : Plane)
  (h1 : h.plane = p)
  (h2 : s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3)
  (h3 : perpendicularLines l (h.sides s1) ∧ perpendicularLines l (h.sides s2) ∧ perpendicularLines l (h.sides s3)) :
  perpendicular l p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_to_three_nonparallel_lines_perpendicular_to_three_hexagon_sides_l1115_111520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hm_length_approx_l1115_111570

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB BC CA : ℝ)
  (ab_positive : AB > 0)
  (bc_positive : BC > 0)
  (ca_positive : CA > 0)
  (triangle_inequality : AB + BC > CA ∧ BC + CA > AB ∧ CA + AB > BC)

/-- Point on a side of the triangle -/
structure TrianglePoint (T : Triangle) :=
  (X Y : ℝ)
  (on_side : (X = 0 ∧ 0 ≤ Y ∧ Y ≤ T.BC) ∨ 
             (Y = 0 ∧ 0 ≤ X ∧ X ≤ T.CA) ∨ 
             (X + Y = T.AB ∧ X ≥ 0 ∧ Y ≥ 0))

/-- Midpoint of a side -/
def is_midpoint (T : Triangle) (P : TrianglePoint T) : Prop :=
  (P.X = T.AB / 2 ∧ P.Y = 0) ∨ 
  (P.X = 0 ∧ P.Y = T.BC / 2) ∨ 
  (P.X = T.CA / 2 ∧ P.Y = T.AB - T.CA / 2)

/-- Foot of the altitude -/
def altitude_foot (T : Triangle) (P : TrianglePoint T) : Prop :=
  P.X = 0 ∧ 0 < P.Y ∧ P.Y < T.BC

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Approximate equality -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop :=
  abs (x - y) < ε

/-- Main theorem -/
theorem hm_length_approx (T : Triangle) 
  (h_ab : T.AB = 16) (h_bc : T.BC = 17) (h_ca : T.CA = 18)
  (M : TrianglePoint T) (h_m : is_midpoint T M)
  (H : TrianglePoint T) (h_h : altitude_foot T H) :
  ∃ (hm : ℝ), approx_equal (distance (H.X, H.Y) (M.X, M.Y)) 6.75 0.01 ∧ 0 < hm := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hm_length_approx_l1115_111570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_iff_a_in_range_l1115_111593

/-- The function f(x) = a*ln(x) - x^2 + (1/2)*a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + (1/2) * a

/-- The theorem stating that f(x) ≤ 0 for all x > 0 if and only if 0 ≤ a ≤ 2 -/
theorem f_nonpositive_iff_a_in_range (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_iff_a_in_range_l1115_111593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1115_111535

noncomputable def f (x : ℝ) := -3 / x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x > 3 ∧ f x = y) ↔ -1 < y ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1115_111535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charging_bull_rounds_is_35_l1115_111502

/-- The number of rounds the "Racing Magic" completes in one hour -/
def racing_magic_rounds : ℕ := 30

/-- The time (in minutes) it takes for both cars to meet at the starting point for the second time -/
def meeting_time : ℕ := 12

/-- The number of rounds the "Racing Magic" completes before the second meeting -/
def racing_magic_rounds_at_meeting : ℕ := meeting_time / 2

/-- The number of rounds the "Charging Bull" completes in one hour -/
def charging_bull_rounds_per_hour : ℕ :=
  let n : ℕ := 1  -- The minimum number of additional rounds the "Charging Bull" completes
  5 * (racing_magic_rounds_at_meeting + n)

theorem charging_bull_rounds_is_35 : charging_bull_rounds_per_hour = 35 := by
  -- Expand the definition of charging_bull_rounds_per_hour
  unfold charging_bull_rounds_per_hour
  -- Expand the definition of racing_magic_rounds_at_meeting
  unfold racing_magic_rounds_at_meeting
  -- Evaluate the expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charging_bull_rounds_is_35_l1115_111502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1115_111543

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 →
  b > 0 →
  c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a^2 + b^2 = 4 - Real.cos C^2 →
  a * b = 2 →
  S = (1/2) * a * b * Real.sin C →
  S = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l1115_111543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1115_111531

-- Define terminal_side as a function
def terminal_side (θ : Real) : Set (Real × Real) :=
  {P | ∃ (r : Real), r > 0 ∧ P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ}

theorem angle_problem (θ : Real) (m : Real) (n : Real) :
  (∃ (P : Real × Real), P.1 = -Real.sqrt 3 ∧ P.2 = m ∧ P ∈ terminal_side θ) →
  Real.sin θ = Real.sqrt 10 / 10 →
  n = Real.tan (θ + Real.pi / 4) →
  m > 0 →
  m^2 + n^2 = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l1115_111531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2017_l1115_111576

noncomputable section

variables (f : ℝ → ℝ)

axiom f_def : ∀ x, f x = (1/2) * x^2 + 2 * x * (deriv f 2017) - 2017 * Real.log x

theorem f_derivative_at_2017 : 
  (deriv f) 2017 = -2016 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2017_l1115_111576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l1115_111505

/-- Represents the total distance of a journey in kilometers. -/
noncomputable def total_distance : ℝ := 425 / 27

/-- Represents the fraction of the journey covered by train. -/
def train_fraction : ℚ := 3 / 5

/-- Represents the fraction of the journey covered by bus. -/
def bus_fraction : ℚ := 7 / 20

/-- Represents the fraction of the journey covered by bicycle. -/
def bicycle_fraction : ℚ := 3 / 10

/-- Represents the fraction of the journey covered by taxi. -/
def taxi_fraction : ℚ := 1 / 50

/-- Represents the distance walked in kilometers. -/
noncomputable def walking_distance : ℝ := 4.25

/-- Theorem stating that the sum of all journey parts equals the total distance. -/
theorem journey_distance : 
  (train_fraction : ℝ) * total_distance + 
  (bus_fraction : ℝ) * total_distance + 
  (bicycle_fraction : ℝ) * total_distance + 
  (taxi_fraction : ℝ) * total_distance + 
  walking_distance = total_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l1115_111505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l1115_111541

noncomputable def square_pyramid_volume (base_edge : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

theorem frustum_volume_fraction (base_edge height : ℝ) :
  base_edge > 0 → height > 0 →
  (let original_volume := square_pyramid_volume base_edge height
   let smaller_height := height / 3
   let smaller_base_edge := base_edge / 3
   let smaller_volume := square_pyramid_volume smaller_base_edge smaller_height
   let frustum_volume := original_volume - smaller_volume
   frustum_volume / original_volume) = 13 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l1115_111541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l1115_111529

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -x + 1/x

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ -1/9 }

-- Theorem statement
theorem max_value_of_f_on_interval :
  ∃ (m : ℝ), m = 3/2 ∧ ∀ x ∈ interval, f x ≤ m := by
  -- We'll use 3/2 as our maximum value
  let m := 3/2
  
  -- Prove that m satisfies the conditions
  have h_m_eq : m = 3/2 := by rfl
  
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l1115_111529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_six_equals_43_over_16_l1115_111575

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 9) / 4  -- This is f⁻¹(x)
  3 * y^2 + 4 * y - 2

-- State the theorem
theorem g_of_neg_six_equals_43_over_16 : g (-6) = 43 / 16 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_six_equals_43_over_16_l1115_111575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plus_signs_count_l1115_111564

theorem plus_signs_count (total : ℕ) (plus_count : ℕ) (minus_count : ℕ) :
  total = 23 →
  plus_count + minus_count = total →
  (∀ (selected : Finset ℕ), selected.card = 10 → ∃ (i : ℕ), i ∈ selected ∧ i ≤ plus_count) →
  (∀ (selected : Finset ℕ), selected.card = 15 → ∃ (i : ℕ), i ∈ selected ∧ plus_count < i ∧ i ≤ total) →
  plus_count = 14 :=
by
  intros h_total h_sum h_plus h_minus
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plus_signs_count_l1115_111564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_five_l1115_111517

/-- Two lines intersecting at point P(2, 5) with slopes 1/2 and 3 respectively -/
structure IntersectingLines where
  P : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ
  h_P : P = (2, 5)
  h_slope1 : slope1 = 1/2
  h_slope2 : slope2 = 3

/-- Points Q and R where the lines intersect the y-axis -/
noncomputable def get_y_intercepts (lines : IntersectingLines) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let Q := (0, lines.slope1 * (-2) + 5)
  let R := (0, lines.slope2 * (-2) + 5)
  (Q, R)

/-- The area of triangle PQR -/
noncomputable def triangle_area (lines : IntersectingLines) : ℝ :=
  let (Q, R) := get_y_intercepts lines
  let base := abs (Q.2 - R.2)
  let height := abs (lines.P.1 - 0)
  (1/2) * base * height

/-- Theorem stating that the area of triangle PQR is 5 -/
theorem area_is_five (lines : IntersectingLines) : triangle_area lines = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_five_l1115_111517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pay_per_mile_l1115_111511

/-- Calculates the pay per mile for a truck driver given gas cost, truck efficiency, profit, and trip distance. -/
noncomputable def pay_per_mile (gas_cost : ℝ) (truck_efficiency : ℝ) (profit : ℝ) (trip_distance : ℝ) : ℝ :=
  let gas_used := trip_distance / truck_efficiency
  let gas_expense := gas_used * gas_cost
  let total_earnings := profit + gas_expense
  total_earnings / trip_distance

/-- Proves that James' pay per mile is $0.50 given the specified conditions. -/
theorem james_pay_per_mile :
  pay_per_mile 4 20 180 600 = 0.5 := by
  -- Unfold the definition of pay_per_mile
  unfold pay_per_mile
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pay_per_mile_l1115_111511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_Y_at_pi_over_3_l1115_111501

-- Define the function Y as noncomputable
noncomputable def Y (x : ℝ) : ℝ := (Real.sin x - Real.cos x) / (2 * Real.cos x)

-- State the theorem
theorem derivative_Y_at_pi_over_3 :
  deriv Y (π/3) = 2 := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_Y_at_pi_over_3_l1115_111501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_sin_x_l1115_111558

open Real Set

theorem max_value_x_minus_sin_x :
  let f : ℝ → ℝ := fun x ↦ x - sin x
  (∀ x ∈ Icc (π/2) (3*π/2), f x ≤ f (3*π/2)) ∧ f (3*π/2) = 3*π/2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_sin_x_l1115_111558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_sequence_prime_count_l1115_111569

def sequenceNum (n : ℕ) : ℕ :=
  if n = 1 then 29
  else 29 * (((10 ^ n - 1) / 9))

theorem only_first_prime :
  ∀ n : ℕ, n > 1 → ¬(Nat.Prime (sequenceNum n)) := by
  sorry

theorem sequence_prime_count :
  (Finset.filter (λ n => Nat.Prime (sequenceNum n)) (Finset.range ω)).card = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_sequence_prime_count_l1115_111569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1115_111540

noncomputable def ellipse_problem : Prop :=
  ∀ (a b : ℝ) (F A B : ℝ × ℝ),
    a > b ∧ b > 0 →
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) →
    F.1 = -a/2 ∧ F.2 = 0 →
    A.1 = 0 ∧ A.2 = b →
    B.2 = 0 →
    (A.2 - F.2) * (B.1 - A.1) = -(A.1 - F.1) * (B.2 - A.2) →
    (∃ (c : ℝ × ℝ) (r : ℝ), 
      (c.1 - F.1)^2 + (c.2 - F.2)^2 = r^2 ∧
      (c.1 - A.1)^2 + (c.2 - A.2)^2 = r^2 ∧
      (c.1 - B.1)^2 + (c.2 - B.2)^2 = r^2 ∧
      |c.1 + Real.sqrt 3 * c.2 + 3| / Real.sqrt 4 = r) →
    (a^2 - b^2) / a^2 = 1/4 →
    (∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) ∧
    (∀ (k : ℝ), k ≠ 0 →
      ¬∃ (M N P Q : ℝ × ℝ),
        M ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} ∧
        N ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} ∧
        Q ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1} ∧
        M.2 - F.2 = k * (M.1 - F.1) ∧
        N.2 - F.2 = k * (N.1 - F.1) ∧
        P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
        Q.1 - P.1 = P.1 ∧ Q.2 - P.2 = P.2 ∧
        M.1 + N.1 = Q.1 ∧ M.2 + N.2 = Q.2)

theorem ellipse_theorem : ellipse_problem := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1115_111540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_F_l1115_111573

-- Define the function f
variable (a : ℝ) (ha : a > 0)
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_defined_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, x ∈ Set.Icc (-a) a → ∃ y, f x = y

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem sum_of_max_min_F (hf_odd : is_odd f) (hf_defined : is_defined_on_interval f a) :
  (⨆ x ∈ Set.Icc (-a) a, F f x) + (⨅ x ∈ Set.Icc (-a) a, F f x) = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_F_l1115_111573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_derivative_reasoning_methods_math_proof_methods_l1115_111594

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, |x - x₀| < 1 → f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define inductive and deductive reasoning
def InductiveReasoning : Prop :=
  ∃ (specific general : Prop), specific → general

def DeductiveReasoning : Prop :=
  ∃ (general specific : Prop), general → specific

-- Define synthetic and analytic methods
def SyntheticMethod : Prop :=
  ∃ (effect cause : Prop), effect → cause

def AnalyticMethod : Prop :=
  ∃ (result cause : Prop), result → cause

-- Theorem statements
theorem extremum_derivative (x₀ : ℝ) (h : IsExtremum f x₀) :
  deriv f x₀ = 0 := by sorry

theorem reasoning_methods :
  InductiveReasoning ∧ DeductiveReasoning := by sorry

theorem math_proof_methods :
  SyntheticMethod ∧ AnalyticMethod := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_derivative_reasoning_methods_math_proof_methods_l1115_111594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terry_spending_l1115_111550

/-- The amount Terry spent on breakfast over three days -/
def breakfast_spending : ℕ → ℕ := sorry

/-- The total amount Terry spent over three days -/
def total_spending : ℕ := sorry

/-- Theorem stating the total amount Terry spent over three days -/
theorem terry_spending :
  (breakfast_spending 1 = 6) →
  (breakfast_spending 2 = 2 * breakfast_spending 1) →
  (breakfast_spending 3 = 2 * (breakfast_spending 1 + breakfast_spending 2)) →
  (total_spending = breakfast_spending 1 + breakfast_spending 2 + breakfast_spending 3) →
  total_spending = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terry_spending_l1115_111550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_eq_solution_set_l1115_111536

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- Theorem for the solution of f(x) = 1/4
theorem solution_eq (x : ℝ) : f x = 1/4 ↔ x = Real.sqrt 2 := by sorry

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set : {x : ℝ | f x ≤ 2} = Set.Icc (-1) 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_eq_solution_set_l1115_111536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_from_incircle_and_complex_points_l1115_111559

/-- Given a triangle ABC with an incircle of radius r centered at the origin,
    and complex numbers t₁, t₂, t₃ representing the points where the incircle
    touches the sides BC, CA, and AB respectively, prove that the circumradius R
    of triangle ABC is equal to 2r⁴ / |((t₁ + t₂)(t₂ + t₃)(t₃ + t₁))| -/
theorem circumradius_from_incircle_and_complex_points
  (r : ℝ) (t₁ t₂ t₃ : ℂ) :
  let R := (2 * r^4) / Complex.abs ((t₁ + t₂) * (t₂ + t₃) * (t₃ + t₁))
  ∃ (A B C : ℂ),
    -- The incircle touches BC, CA, and AB at t₁, t₂, and t₃ respectively
    Complex.abs (B - t₁) = r ∧ Complex.abs (C - t₁) = r ∧
    Complex.abs (C - t₂) = r ∧ Complex.abs (A - t₂) = r ∧
    Complex.abs (A - t₃) = r ∧ Complex.abs (B - t₃) = r ∧
    -- R is the circumradius of triangle ABC
    Complex.abs (A - (B + C) / 2) = R ∧
    Complex.abs (B - (A + C) / 2) = R ∧
    Complex.abs (C - (A + B) / 2) = R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_from_incircle_and_complex_points_l1115_111559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1115_111587

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := |Real.sin x| + |Real.cos x|

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, f (x + Real.pi/2) = f x) ∧
  (∀ x : ℝ, 1 ≤ f x ∧ f x ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1115_111587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_after_moves_l1115_111503

/-- Represents a move in the game, where numbers are paired and replaced by their sum and difference -/
def gameMove (numbers : List Int) : List Int :=
  sorry

/-- Checks if a list of integers consists of 2n consecutive integers -/
def isConsecutive (numbers : List Int) : Bool :=
  sorry

/-- Theorem stating that after any number of moves, 2n consecutive integers will never reappear -/
theorem no_consecutive_after_moves (n : Nat) (initial : List Int) :
  isConsecutive initial → initial.length = 2 * n →
  ∀ (moves : Nat), ¬isConsecutive (Nat.rec initial (fun _ => gameMove) moves) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_after_moves_l1115_111503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_distance_calculation_l1115_111553

/-- Calculates the distance to an event given the total cost, duration, and cost per mile --/
noncomputable def distance_to_event (total_cost : ℝ) (duration_days : ℕ) (cost_per_mile : ℝ) : ℝ :=
  total_cost / (2 * (duration_days : ℝ) * cost_per_mile)

/-- Theorem stating that given the specified conditions, the distance to the event is 200 miles --/
theorem event_distance_calculation :
  let total_cost : ℝ := 7000
  let duration_days : ℕ := 7
  let cost_per_mile : ℝ := 2.5
  distance_to_event total_cost duration_days cost_per_mile = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_distance_calculation_l1115_111553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_side_length_l1115_111555

-- Define the rectangle
def Rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ y ≥ x

-- Define the diagonal of the rectangle
noncomputable def Diagonal (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

-- State the theorem
theorem shorter_side_length (x y : ℝ) :
  Rectangle x y →
  y = 9 →
  x + y - Diagonal x y = y / 3 →
  x = 15 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_side_length_l1115_111555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1115_111538

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) + 1 / (x - 1)

theorem domain_of_f : 
  {x : ℝ | (x + 2 ≥ 0 ∧ x ≠ 1)} = {x : ℝ | x ≥ -2 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1115_111538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_kangaroo_distance_l1115_111551

def hop_distance (n : ℕ) : ℚ :=
  (1/4) * (3/4)^(n-1) * 2

theorem kevin_kangaroo_distance :
  (Finset.range 6).sum (λ i => hop_distance (i+1)) = 1321 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_kangaroo_distance_l1115_111551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_or_not_q_l1115_111504

-- Define proposition p
def p : Prop := ∀ (a x : ℝ), a > 1 → a^x > Real.log x / Real.log a

-- Define arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define proposition q
def q : Prop := ∀ (a : ℕ → ℝ) (m n p q : ℕ),
  isArithmeticSequence a →
  (m + n = p + q → a n + a m = a p + a q) ∧
  ¬(a n + a m = a p + a q → m + n = p + q)

-- Theorem to prove
theorem not_p_or_not_q : ¬p ∨ ¬q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_or_not_q_l1115_111504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_purchase_problem_l1115_111548

/-- Represents the quantity of high-quality wine in dou -/
def x : ℝ := sorry

/-- Represents the quantity of ordinary wine in dou -/
def y : ℝ := sorry

/-- The cost of high-quality wine per dou -/
def high_quality_cost : ℝ := 50

/-- The cost of ordinary wine per dou -/
def ordinary_cost : ℝ := 10

/-- The total quantity of wine purchased -/
def total_quantity : ℝ := 2

/-- The total cost of the wine purchase -/
def total_cost : ℝ := 30

/-- Theorem stating that the system of equations correctly represents the wine purchase problem -/
theorem wine_purchase_problem :
  (x + y = total_quantity) ∧ (high_quality_cost * x + ordinary_cost * y = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_purchase_problem_l1115_111548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1115_111519

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) :
  (seq.a 4)^2 = seq.a 3 * seq.a 7 →  -- a_4 is geometric mean of a_3 and a_7
  sum_n seq 8 = 32 →                 -- S_8 = 32
  sum_n seq 10 = 60 :=                -- S_10 = 60
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1115_111519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_marbles_count_l1115_111525

/-- The number of yellow marbles in the bag -/
def yellow_marbles : ℕ := 12

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 10

/-- The number of black marbles in the bag -/
def black_marbles : ℕ := 1

/-- The probability of drawing a black marble -/
def black_probability : ℚ := 1 / 28

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := 28

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := total_marbles - (yellow_marbles + blue_marbles + black_marbles)

theorem green_marbles_count :
  black_probability = black_marbles / total_marbles → green_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_marbles_count_l1115_111525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1115_111585

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x * Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem f_inequality : f (1/4 : ℝ) < f (-1/3 : ℝ) ∧ f (-1/3 : ℝ) < f (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1115_111585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quadratic_roots_l1115_111572

theorem sum_of_quadratic_roots : 
  ∃ r s : ℝ, (-11 * r^2 + 19 * r + 63 = 0) ∧ 
             (-11 * s^2 + 19 * s + 63 = 0) ∧ 
             (r + s = 19 / 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_quadratic_roots_l1115_111572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eternal_channel_classification_l1115_111530

noncomputable section

/-- A function has a channel of width d within D if there exist two parallel lines
    with distance d that bound the function within D. -/
def has_channel (f : ℝ → ℝ) (d : ℝ) (D : Set ℝ) : Prop :=
  ∃ (k m₁ m₂ : ℝ), ∀ x ∈ D, k * x + m₁ ≤ f x ∧ f x ≤ k * x + m₂ ∧ m₂ - m₁ = d

/-- A function has an eternal channel at positive infinity if for any positive ε,
    there exists an x₀ such that the function has a channel of width ε within [x₀, ∞). -/
def has_eternal_channel (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ x₀ : ℝ, has_channel f ε {x | x ≥ x₀}

noncomputable def f₁ : ℝ → ℝ := λ x ↦ Real.log x
noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.sin x / x
noncomputable def f₃ : ℝ → ℝ := λ x ↦ Real.sqrt (x^2 - 1)
def f₄ : ℝ → ℝ := λ x ↦ x^2
noncomputable def f₅ : ℝ → ℝ := λ x ↦ Real.exp (-x)

theorem eternal_channel_classification :
  (¬ has_eternal_channel f₁) ∧
  (has_eternal_channel f₂) ∧
  (has_eternal_channel f₃) ∧
  (¬ has_eternal_channel f₄) ∧
  (has_eternal_channel f₅) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eternal_channel_classification_l1115_111530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l1115_111533

-- Define the types for our geometric objects
def Circle : Type := ℝ × ℝ × ℝ  -- (center_x, center_y, radius)
def Point : Type := ℝ × ℝ

-- Define the main geometric setup
structure GeometricSetup where
  Ω : Circle
  ω : Fin 3 → Circle
  γ : Fin 3 → Circle
  O : Fin 3 → Point
  T : Fin 3 → Point
  C : Fin 3 → Point
  S : Fin 3 → Point

-- Define auxiliary functions (these would need proper implementations)
def circles_tangent (c1 c2 : Circle) : Prop := sorry
def circles_inside (c1 c2 : Circle) : Prop := sorry
def is_center (p : Point) (c : Circle) : Prop := sorry
def is_tangent_point (p : Point) (c1 c2 : Circle) : Prop := sorry
def line (p1 p2 : Point) : Set Point := sorry

-- Define the conditions
def satisfies_conditions (setup : GeometricSetup) : Prop :=
  ∀ i : Fin 3,
    (circles_tangent (setup.Ω) (setup.ω i)) ∧
    (circles_inside (setup.ω i) setup.Ω) ∧
    (circles_tangent (setup.γ i) setup.Ω) ∧
    (circles_tangent (setup.γ i) (setup.ω ((i + 1) % 3))) ∧
    (circles_tangent (setup.γ i) (setup.ω ((i + 2) % 3))) ∧
    (is_center (setup.O i) (setup.ω i)) ∧
    (is_tangent_point (setup.T i) (setup.ω i) setup.Ω) ∧
    (is_center (setup.C i) (setup.γ i)) ∧
    (is_tangent_point (setup.S i) (setup.γ i) setup.Ω)

-- Define concurrency
def are_concurrent (l₁ l₂ l₃ : Point × Point) : Prop :=
  ∃ p : Point, p ∈ line l₁.1 l₁.2 ∧ p ∈ line l₂.1 l₂.2 ∧ p ∈ line l₃.1 l₃.2

-- State the theorem
theorem circles_concurrency (setup : GeometricSetup) 
  (h : satisfies_conditions setup) :
  (are_concurrent (setup.T 0, setup.C 0) (setup.T 1, setup.C 1) (setup.T 2, setup.C 2)) ∧
  (are_concurrent (setup.O 0, setup.C 0) (setup.O 1, setup.C 1) (setup.O 2, setup.C 2)) ∧
  (are_concurrent (setup.O 0, setup.S 0) (setup.O 1, setup.S 1) (setup.O 2, setup.S 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_concurrency_l1115_111533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_in_cube_l1115_111562

def count_digit (d : Nat) (x : Nat) : Nat :=
  (x.repr.toList.filter (· == d.repr.toList.head!)).length

theorem count_nines_in_cube : 
  let n : Nat := 10^20 - 1
  count_digit 9 (n^3) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_nines_in_cube_l1115_111562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_for_Q_less_than_half_l1115_111509

/-- Q(N) is the probability that at least 4/7 of the yellow balls are on the same side of the blue ball -/
def Q (N : ℕ) : ℚ :=
  (Int.floor (3 * N / 7 : ℚ) + 1 + (N - Int.ceil (4 * N / 7 : ℚ) + 1)) / (N + 1 : ℚ)

/-- N is a positive multiple of 3 -/
def is_valid_N (N : ℕ) : Prop :=
  N > 0 ∧ N % 3 = 0

theorem smallest_N_for_Q_less_than_half :
  (∀ N, is_valid_N N → N < 21 → Q N ≥ 1/2) ∧
  (is_valid_N 21 ∧ Q 21 < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_for_Q_less_than_half_l1115_111509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_palindrome_8_less_square_l1115_111581

/-- A function to check if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function to check if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop :=
  let s := String.mk (List.reverse (String.toList (toString n)))
  toString n = s

/-- A function to check if a number is 8 less than a perfect square -/
def is8LessThanPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n + 8 = m * m

/-- The main theorem -/
theorem smallest_prime_palindrome_8_less_square :
  ∀ n : ℕ, n > 0 →
    (isPrime n ∧ isPalindrome n ∧ is8LessThanPerfectSquare n) →
    n ≥ 17 :=
by
  sorry

#check smallest_prime_palindrome_8_less_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_palindrome_8_less_square_l1115_111581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l1115_111513

open Real

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem f_min_max_on_interval :
  ∃ (min max : ℝ), min = -3*π/2 ∧ max = π/2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2*π), f x ≥ min ∧ f x ≤ max) ∧
  (∃ x₁ ∈ Set.Icc 0 (2*π), f x₁ = min) ∧
  (∃ x₂ ∈ Set.Icc 0 (2*π), f x₂ = max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l1115_111513
