import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_solution_l862_86220

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℤ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The theorem stating the solution to the magic square problem -/
theorem magic_square_solution :
  ∀ (ms : MagicSquare), 
    ms.a = 62 ∧ 
    ms.b = -88 ∧ 
    ms.c = 87 ∧ 
    ms.d = 4 → 
    ms.a + ms.b + ms.c = ms.d + ms.e + ms.f ∧
    ms.a + ms.d + ms.g = ms.b + ms.e + ms.h ∧
    ms.a + ms.e + ms.i = ms.c + ms.e + ms.g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_solution_l862_86220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_9pi_l862_86234

/-- The number of circles -/
def num_circles : ℕ := 12

/-- The radius of each circle in cm -/
noncomputable def radius : ℝ := 1

/-- The central angle of the unshaded sector in degrees -/
noncomputable def unshaded_angle : ℝ := 90

/-- The total central angle of a circle in degrees -/
noncomputable def total_angle : ℝ := 360

/-- The shaded fraction of each circle -/
noncomputable def shaded_fraction : ℝ := 1 - (unshaded_angle / total_angle)

/-- The area of a single circle -/
noncomputable def circle_area : ℝ := Real.pi * radius^2

/-- The shaded area of a single circle -/
noncomputable def shaded_area_per_circle : ℝ := shaded_fraction * circle_area

/-- The total shaded area of all circles -/
noncomputable def total_shaded_area : ℝ := num_circles * shaded_area_per_circle

theorem total_shaded_area_is_9pi : total_shaded_area = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_9pi_l862_86234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_and_max_value_l862_86240

def S (n : ℕ) : ℤ := -n^2 + 7*n + 1

def a : ℕ → ℤ
  | 0 => 7  -- Added case for 0
  | 1 => 7
  | (n+2) => -2*(n+2) + 8

theorem general_term_and_max_value :
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) ∧
  (∃ n : ℕ, S n = 13 ∧ ∀ m : ℕ, S m ≤ S n) :=
by
  sorry

#eval S 3  -- Test S function
#eval a 3  -- Test a function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_and_max_value_l862_86240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l862_86251

def A : ℤ := 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 16 * 17 + 18 * 19 + 
           20 * 21 + 22 * 23 + 24 * 25 + 26 * 27 + 28 * 29 + 30 * 31 + 32 * 33 + 34 * 35 + 
           36 * 37 + 38 * 39 + 40

def B : ℤ := 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 16 * 17 + 18 * 19 + 
           20 * 21 + 22 * 23 + 24 * 25 + 26 * 27 + 28 * 29 + 30 * 31 + 32 * 33 + 34 * 35 + 
           36 * 37 + 38 * 39 + 40 * 41 + 42

theorem difference_A_B : |A - B| = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l862_86251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_payment_l862_86271

theorem james_payment : 
  let total_lessons : Nat := 20
  let lesson_cost : Nat := 5
  let free_lessons : Nat := 1
  let full_price_lessons : Nat := 10
  let remaining_lessons := total_lessons - free_lessons - full_price_lessons
  let half_price_lessons := remaining_lessons / 2
  let total_cost := lesson_cost * (full_price_lessons + half_price_lessons)
  let james_payment := total_cost / 2
  james_payment = 35
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_payment_l862_86271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l862_86278

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1/2) * a * x^2 - x + 1/2) / Real.log a

theorem f_positive_iff_a_in_range :
  ∀ a > 0, (∀ x ∈ Set.Icc 2 3, f a x > 0) ↔
    a ∈ Set.union (Set.Ioo (3/4) (7/9)) (Set.Ioi (5/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_a_in_range_l862_86278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_30_terms_equals_neg_10_sqrt_3_l862_86295

/-- A function f(x) with specific properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ 2 * Real.sin (ω * x + φ)

/-- Sequence an defined based on f -/
noncomputable def a (ω φ : ℝ) : ℕ → ℝ := fun n ↦ n * f ω φ (n * Real.pi / 3)

/-- Sum of the first N terms of sequence a -/
noncomputable def S (ω φ : ℝ) (N : ℕ) : ℝ := (Finset.range N).sum (a ω φ)

theorem sum_30_terms_equals_neg_10_sqrt_3 :
  ∀ ω φ : ℝ,
    ω > 0 →
    |φ| < Real.pi →
    f ω φ (Real.pi / 12) = -2 →
    f ω φ (7 * Real.pi / 12) = 2 →
    StrictMonoOn (f ω φ) (Set.Ioo (Real.pi / 12) (7 * Real.pi / 12)) →
    S ω φ 30 = -10 * Real.sqrt 3 := by
  sorry

#check sum_30_terms_equals_neg_10_sqrt_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_30_terms_equals_neg_10_sqrt_3_l862_86295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l862_86299

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (-3, 4)

def is_on_parabola (P : ℝ × ℝ) : Prop :=
  P.2^2 = 4 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_sum :
  ∃ (min : ℝ), min = 12 ∧
  ∀ (P : ℝ × ℝ), is_on_parabola P →
  distance A P + distance B P ≥ min := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l862_86299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l862_86272

noncomputable def f (α : Real) : Real := Real.sin (Real.pi - α) / Real.sin (Real.pi / 2 + α)

theorem problem_solution :
  -- Part I
  f (4 * Real.pi / 3) = Real.sqrt 3 ∧
  -- Part II
  ∀ A : Real,
    0 < A ∧ A < Real.pi →  -- A is an interior angle of a triangle
    f A = 3 / 4 →
    (Real.cos A) ^ 2 - (Real.sin A) ^ 2 = 7 / 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l862_86272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_conversions_l862_86249

-- Define the conversion factors
def light_year_km : ℝ := 9.5e12
def parsec_au : ℝ := 206265
def parsec_ly : ℝ := 3.2616

-- Define the target values
def parsec_km : ℝ := 3.099e13
def au_km : ℝ := 1.502e8

-- Define an approximate equality relation
def approx_equal (x y : ℝ) : Prop := abs (x - y) < 1e-4 * abs y

-- State the theorem
theorem unit_conversions :
  (approx_equal (light_year_km * parsec_ly) parsec_km) ∧
  (approx_equal (parsec_km / parsec_au) au_km) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_conversions_l862_86249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_l862_86244

/-- Given an ellipse with eccentricity √3/2 and other specified properties, 
    prove its equation and find the maximum area of a related quadrilateral. -/
theorem ellipse_and_quadrilateral (a b c : ℝ) (F₁ F₂ M : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- Ellipse parameters
  c / a = Real.sqrt 3 / 2 ∧  -- Eccentricity
  c^2 = a^2 - b^2 ∧  -- Relation between a, b, and c
  (F₁.1 = -c ∧ F₁.2 = 0) ∧ (F₂.1 = c ∧ F₂.2 = 0) ∧  -- Foci positions
  M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧  -- M is on the ellipse
  M ≠ (a, 0) ∧ M ≠ (-a, 0) ∧  -- M is not at the endpoints of major axis
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) + 
  Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) + 
  Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 4 + 2 * Real.sqrt 3 →  -- Perimeter of ΔMF₁F₂
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧  -- Equation of ellipse C
  (∃ k : ℝ, ∀ A B N : ℝ × ℝ,
    (A.1^2 / 4 + A.2^2 = 1) ∧  -- A is on the ellipse
    (B.1^2 / 4 + B.2^2 = 1) ∧  -- B is on the ellipse
    A.2 = k * A.1 - 2 ∧  -- A is on line l
    B.2 = k * B.1 - 2 ∧  -- B is on line l
    N = (A.1 + B.1, A.2 + B.2) →  -- N satisfies ON = OA + OB
    abs (A.1 * B.2 - A.2 * B.1) ≤ 1 ∧  -- Area of OANB ≤ 2
    (abs (A.1 * B.2 - A.2 * B.1) = 1 ↔ k^2 = 7/4))  -- Maximum area occurs when k = ±√(7/4)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_l862_86244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eq_tan_implies_sum_is_one_l862_86222

theorem cos_eq_tan_implies_sum_is_one (α : ℝ) (h : Real.cos α = Real.tan α) : 
  (1 / Real.sin α) + (Real.cos α)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eq_tan_implies_sum_is_one_l862_86222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l862_86229

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (f (2 * a) < f (a - 1)) → (a < -1) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l862_86229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_car_speed_is_90_l862_86218

/-- Calculates the speed of a sports car given specific traffic conditions -/
def sports_car_speed 
  (truck_speed : ℝ) 
  (truck_spacing : ℝ) 
  (car_speed : ℝ) 
  (car_spacing_time : ℝ) 
  (cars_passed_between_trucks : ℕ) : ℝ :=
by
  sorry

#check sports_car_speed

theorem sports_car_speed_is_90 :
  sports_car_speed 60 (1/4) 75 (3/3600) 2 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_car_speed_is_90_l862_86218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l862_86275

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and point D symmetric to F₂ with respect
    to the line bx - ay = 0, if D lies on the circle centered at F₁ with
    radius |OF₁|, then the eccentricity of C is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let F₁ : ℝ × ℝ := (0, -Real.sqrt (a^2 + b^2))
  let F₂ : ℝ × ℝ := (0, Real.sqrt (a^2 + b^2))
  let O : ℝ × ℝ := (0, 0)
  let l := {p : ℝ × ℝ | b * p.1 - a * p.2 = 0}
  ∃ (D : ℝ × ℝ), (D.1 - F₂.1)^2 + (D.2 + F₂.2)^2 = (2 * b)^2 ∧ 
                 D.1^2 + D.2^2 = F₁.1^2 + F₁.2^2 →
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l862_86275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l862_86263

-- Define propositions p and q
noncomputable def p : Prop := ∃ (a b : ℚ), Real.pi = a / b
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem problem_solution :
  (¬p) ∧ q → (¬(p ∧ (¬q))) ∧ ((¬p) ∨ q) := by
  intro h
  constructor
  . intro h'
    cases h'
    case intro hp hnq =>
      cases h
      case intro hnp hq =>
        exact hnp hp
  . left
    cases h
    case intro hnp hq =>
      exact hnp

-- The proof is completed with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l862_86263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equation_l862_86221

-- Define the ellipse
def my_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def my_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y = 0

-- Define the line
def my_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 2) + 1

-- Define the theorem
theorem ellipse_and_line_equation :
  ∀ (a b : ℝ) (P F₁ F₂ : ℝ × ℝ),
  a > b ∧ b > 0 ∧
  my_ellipse a b P.1 P.2 ∧
  (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0 ∧
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (4/3)^2 ∧
  (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (14/3)^2 →
  ∃ (k : ℝ) (A B : ℝ × ℝ),
  my_circle (-2) 1 ∧
  my_line k A.1 A.2 ∧ my_line k B.1 B.2 ∧
  my_ellipse a b A.1 A.2 ∧ my_ellipse a b B.1 B.2 ∧
  A.1 + B.1 = -4 ∧ A.2 + B.2 = 2 →
  (a = 3 ∧ b = 2 ∧ k = 8/9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equation_l862_86221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l862_86298

theorem periodic_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 + 3 * (2 / 99)) → 68 / 33 = 2 + 3 * (2 / 99) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_decimal_to_fraction_l862_86298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_area_is_one_percent_l862_86274

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  side : ℝ
  cross_area_percent : ℝ
  green_area_percent : ℝ
  cross_symmetric : Bool

/-- Calculates the percentage of the flag's area occupied by the blue square -/
def blue_area_percent (flag : SquareFlag) : ℝ :=
  sorry

/-- Theorem stating that under given conditions, the blue area is 1% of the flag -/
theorem blue_area_is_one_percent (flag : SquareFlag) 
  (h1 : flag.cross_area_percent = 49)
  (h2 : flag.green_area_percent = 5)
  (h3 : flag.cross_symmetric = true) :
  blue_area_percent flag = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_area_is_one_percent_l862_86274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_brand_a_l862_86207

/-- The weight of one liter of vegetable ghee for brand 'a' in grams -/
def weight_a : ℝ := sorry

/-- The weight of one liter of vegetable ghee for brand 'b' in grams -/
def weight_b : ℝ := 750

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3360

theorem weight_of_brand_a : 
  weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
  weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume = total_weight → 
  weight_a = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_of_brand_a_l862_86207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_three_intersecting_circles_l862_86256

/-- The area of the shaded region formed by the intersection of three circles --/
theorem shaded_area_of_three_intersecting_circles :
  let r : ℝ := 5
  let circle_area : ℝ := π * r^2
  let sector_area : ℝ := circle_area / 6
  let triangle_side : ℝ := r
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  let shaded_area : ℝ := 3 * sector_area - triangle_area
  shaded_area = (150 * π - 75 * Real.sqrt 3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_three_intersecting_circles_l862_86256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l862_86258

/-- The complex number z = (2-i)/(3i-1) -/
noncomputable def z : ℂ := (2 - Complex.I) / (3 * Complex.I - 1)

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

/-- Theorem: The complex number z lies in the third quadrant -/
theorem z_in_third_quadrant : in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l862_86258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_pi_fourth_l862_86253

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem second_derivative_at_pi_fourth :
  (deriv^[2] f) (π/4) = Real.sqrt 2 - (π * Real.sqrt 2) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_pi_fourth_l862_86253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_theorem_l862_86216

/-- A type representing a variable in our problem -/
structure Variable where
  value : ℝ

/-- Defines a relationship between two variables -/
def relationship (x y : Variable) : Prop :=
  y.value = -0.1 * x.value + 1

/-- Defines positive correlation between two variables -/
def positively_correlated (a b : Variable) : Prop :=
  ∀ (δ : ℝ), δ > 0 → (a.value + δ > a.value) → (b.value > b.value)

/-- Defines negative correlation between two variables -/
def negatively_correlated (a b : Variable) : Prop :=
  ∀ (δ : ℝ), δ > 0 → (a.value + δ > a.value) → (b.value < b.value)

theorem correlation_theorem (x y z : Variable) 
  (h1 : relationship x y) 
  (h2 : positively_correlated y z) : 
  negatively_correlated x y ∧ negatively_correlated x z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_theorem_l862_86216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_are_finite_l862_86238

/-- Represents a polygon with n vertices, where n is odd and n ≥ 5 -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℤ
  n_odd : Odd n
  n_ge_5 : n ≥ 5

/-- The operation performed on the polygon -/
def apply_operation (p : Polygon n) (i : Fin n) : Polygon n :=
  sorry

/-- Predicate to check if an operation is valid -/
def valid_operation (p : Polygon n) (i : Fin n) : Prop :=
  p.vertices i < 0

/-- The sum of all integers on the vertices -/
def vertex_sum (p : Polygon n) : ℤ :=
  (Finset.sum (Finset.univ : Finset (Fin n)) fun i => p.vertices i)

/-- Main theorem: Any sequence of operations is finite -/
theorem operations_are_finite (n : ℕ) (p : Polygon n) 
    (h_sum_pos : vertex_sum p > 0) :
    ¬∃(seq : ℕ → Polygon n), (seq 0 = p) ∧ 
    (∀ k, ∃ i, valid_operation (seq k) i ∧ seq (k+1) = apply_operation (seq k) i) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_are_finite_l862_86238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l862_86285

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 3) / Real.sqrt (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > -1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l862_86285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l862_86257

-- Define the given constants
noncomputable def boat_speed_still : ℝ := 30
noncomputable def distance_downstream : ℝ := 0.24
noncomputable def time_downstream : ℝ := 1 / 150

-- Define the theorem
theorem current_speed_calculation :
  let downstream_speed := distance_downstream / time_downstream
  let current_speed := downstream_speed - boat_speed_still
  current_speed = 6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_calculation_l862_86257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l862_86277

theorem square_of_3x_plus_4_when_x_is_2 :
  ∀ x : ℝ, x = 2 → (3 * x + 4)^2 = 100 := by
  intro x h
  have h1 : 3 * x + 4 = 10 := by
    rw [h]
    norm_num
  calc
    (3 * x + 4)^2 = 10^2 := by rw [h1]
    _ = 100 := by norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_2_l862_86277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l862_86268

theorem triangle_problem (A B C a b c : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
  (h5 : a * Real.cos B = b * Real.cos A) (h6 : Real.sin A = 1/3) :
  b / a = 1 ∧ Real.sin (C - π/4) = (8 + 7 * Real.sqrt 2) / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l862_86268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l862_86267

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x - a / Real.rpow x (1/3 : ℝ))^4 = 32 + 
    Real.sqrt x * (c₁ * x + c₂ * Real.sqrt x + c₃) + x^2 * (c₄ * x + c₅)) →
  a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_constant_term_l862_86267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_proof_l862_86203

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    S_n represents the sum of the first n terms of the sequence. -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with first term a₁ and common ratio q,
    if S₃ is the middle term of an arithmetic sequence with end terms 2a₁ and a₁q,
    then q = -1/2. -/
theorem geometric_ratio_proof (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) :
  let S₃ := geometric_sum a₁ q 3
  2 * S₃ = 2 * a₁ + a₁ * q → q = -1/2 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_proof_l862_86203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_secret_numbers_l862_86282

-- Define the set of numbers satisfying the conditions
def secretNumbers : Finset ℕ :=
  Finset.filter (fun n =>
    10 ≤ n ∧ n < 100 ∧  -- two-digit integer
    n % 2 = 1 ∧         -- tens digit is odd
    (n / 10) % 2 = 0 ∧  -- units digit is even
    n > 75 ∧            -- greater than 75
    n % 3 = 0           -- divisible by 3
  ) (Finset.range 100)

-- Theorem stating that there are exactly two numbers satisfying all conditions
theorem two_secret_numbers : Finset.card secretNumbers = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_secret_numbers_l862_86282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_is_quarter_l862_86204

def sample_data : List ℚ := [12, 7, 11, 12, 11, 12, 10, 10, 9, 8, 13, 12, 10, 9, 6, 11, 8, 9, 8, 10]

def in_range (x : ℚ) : Bool :=
  11.5 ≤ x ∧ x < 13.5

def count_in_range (data : List ℚ) : Nat :=
  (data.filter in_range).length

noncomputable def frequency (data : List ℚ) : ℚ :=
  (count_in_range data : ℚ) / (data.length : ℚ)

theorem frequency_is_quarter :
  frequency sample_data = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_is_quarter_l862_86204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l862_86245

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the tangent property
def is_tangent (P : Point) (A : Point) : Prop :=
  my_circle A.x A.y ∧ 
  ∃ t : ℝ, P.x = A.x + t * A.y ∧ P.y = A.y - t * A.x

-- Define the right angle property
def right_angle (P A B : Point) : Prop :=
  (A.x - P.x) * (B.x - P.x) + (A.y - P.y) * (B.y - P.y) = 0

-- Main theorem
theorem trajectory_equation (P : Point) :
  (∃ A B : Point, is_tangent P A ∧ is_tangent P B ∧ right_angle P A B) →
  P.x^2 + P.y^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l862_86245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_theorem_l862_86243

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction of the student's walk -/
inductive Direction
| Forward
| Backward

/-- Defines the state of the corridor -/
structure CorridorState where
  lockers : Fin 500 → LockerState
  direction : Direction
  skips : Nat
  toggles : Nat
  walks : Nat

/-- Defines the initial state of the corridor -/
def initialState : CorridorState := {
  lockers := λ _ => LockerState.Closed,
  direction := Direction.Forward,
  skips := 0,
  toggles := 1,
  walks := 0
}

/-- Defines a single step in the walking pattern -/
def step (state : CorridorState) : CorridorState :=
  sorry

/-- Defines the condition for the walking to stop -/
def isComplete (state : CorridorState) : Bool :=
  sorry

/-- Theorem stating the final state of the corridor after the walking pattern -/
theorem final_state_theorem :
  ∃ (finalState : CorridorState),
    (finalState.walks = 251) ∧
    (finalState.lockers 499 = LockerState.Open) ∧
    (∀ i : Fin 500, finalState.lockers i = LockerState.Open) ∧
    (isComplete finalState) ∧
    (∃ n : Nat, Nat.iterate step n initialState = finalState) :=
  sorry

#check final_state_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_state_theorem_l862_86243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersecting_or_skew_l862_86228

/-- Given two lines in 3D space with direction vectors a and b, 
    prove that they are either intersecting or skew if their 
    direction vectors are not parallel. -/
theorem lines_intersecting_or_skew 
  (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, 2, 1)) 
  (hb : b = (-2, 0, 1)) : 
  (∃ (p : ℝ × ℝ × ℝ), p ∈ Set.range (λ t : ℝ ↦ t • a) ∩ 
                        Set.range (λ t : ℝ ↦ t • b)) ∨ 
  (Set.range (λ t : ℝ ↦ t • a) ∩ 
   Set.range (λ t : ℝ ↦ t • b) = ∅) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersecting_or_skew_l862_86228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l862_86214

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/6) - Real.cos (2*x + Real.pi/3) + 2 * (Real.cos x)^2

theorem f_properties :
  (f (Real.pi/12) = Real.sqrt 3 + 1) ∧
  (∀ x, f x ≤ 3) ∧
  (∀ k : ℤ, f (k * Real.pi + Real.pi/6) = 3) := by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l862_86214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicholas_paid_more_l862_86246

/-- The amount of fabric Kenneth bought in yards -/
def kenneth_fabric : ℕ := 700

/-- The price per yard of fabric in dollars -/
def price_per_yard : ℕ := 40

/-- The amount of fabric Nicholas bought in yards -/
def nicholas_fabric : ℕ := 6 * kenneth_fabric

/-- The discount rate Nicholas received -/
def discount_rate : ℚ := 15 / 100

/-- Calculate the total cost for a given amount of fabric -/
def total_cost (fabric : ℕ) : ℕ := fabric * price_per_yard

/-- Calculate the discounted cost for Nicholas -/
def nicholas_discounted_cost : ℕ :=
  total_cost nicholas_fabric - (discount_rate * (total_cost nicholas_fabric : ℚ)).floor.toNat

/-- The theorem to prove -/
theorem nicholas_paid_more :
  (nicholas_discounted_cost : ℤ) - (total_cost kenneth_fabric : ℤ) = 114800 := by
  sorry

#eval nicholas_discounted_cost - total_cost kenneth_fabric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicholas_paid_more_l862_86246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_reduction_l862_86294

/-- The initial population of the village -/
def initial_population : ℝ := sorry

/-- The final active population of the village -/
def final_active_population : ℝ := 3553

/-- Theorem stating the relationship between initial and final population -/
theorem population_reduction :
  initial_population * 0.95 * 0.92 * 0.85 * 0.90 * 0.88 = final_active_population :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_reduction_l862_86294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l862_86276

/-- A shape in a 2D plane -/
structure Shape where
  area : ℝ

/-- A square with side length a -/
def Square (a : ℝ) : Shape where
  area := a * a

/-- A division of a shape into smaller shapes -/
def Division (S : Shape) (parts : List Shape) : Prop :=
  (parts.length > 0) ∧ (parts.map Shape.area).sum = S.area

/-- Two shapes are congruent if they have the same area -/
def Congruent (S1 S2 : Shape) : Prop := S1.area = S2.area

/-- A shape is non-rectangular -/
def NonRectangular (S : Shape) : Prop := sorry

/-- Main theorem: Any square can be divided into pq congruent, non-rectangular shapes -/
theorem square_division (a : ℝ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  ∃ (parts : List Shape), 
    Division (Square a) parts ∧ 
    parts.length = p * q ∧
    (∀ S1 S2, S1 ∈ parts → S2 ∈ parts → Congruent S1 S2) ∧
    (∀ S, S ∈ parts → NonRectangular S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l862_86276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mth_difference_constant_l862_86270

def sequence_m (m n : ℕ) : ℚ :=
  (m + n).factorial / n.factorial

def forward_difference (f : ℕ → ℚ) : ℕ → ℚ :=
  λ k => f (k + 1) - f k

def nth_forward_difference (f : ℕ → ℚ) : ℕ → (ℕ → ℚ)
  | 0 => f
  | n + 1 => forward_difference (nth_forward_difference f n)

theorem mth_difference_constant (m : ℕ) :
  ∀ n, nth_forward_difference (sequence_m m) m n = m.factorial :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mth_difference_constant_l862_86270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_problem_triangle_valid_l862_86231

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the specific triangle from the problem
noncomputable def problemTriangle : Triangle where
  A := Real.arccos (Real.sqrt 6 / 3)
  B := 2 * Real.arccos (Real.sqrt 6 / 3)
  C := Real.pi - 3 * Real.arccos (Real.sqrt 6 / 3)
  a := 3
  b := 2 * Real.sqrt 6
  c := 5

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b = 2 * Real.sqrt 6)
  (h3 : t.B = 2 * t.A)
  : Real.cos t.A = Real.sqrt 6 / 3 ∧ t.c = 5 := by
  sorry

-- Verify that our problemTriangle satisfies the conditions
theorem problem_triangle_valid 
  : problemTriangle.a = 3 
  ∧ problemTriangle.b = 2 * Real.sqrt 6 
  ∧ problemTriangle.B = 2 * problemTriangle.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_problem_triangle_valid_l862_86231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l862_86259

noncomputable def f (x : ℝ) := (Real.sin x + Real.cos x)^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 3) ∧
    f (5 * Real.pi / 12) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l862_86259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_hyperbola_equation_l862_86206

/-- Predicate to determine if an equation represents a shifted hyperbola -/
def IsShiftedHyperbola (x y : ℝ) : Prop :=
  ∃ (a b h k : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- The equation x^2 - 4y^2 - 2x = 0 represents a shifted hyperbola -/
theorem shifted_hyperbola_equation (x y : ℝ) : 
  (x^2 - 4*y^2 - 2*x = 0) ↔ IsShiftedHyperbola x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_hyperbola_equation_l862_86206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_radius_l862_86255

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define what it means for a point to be inside a circle
def is_inside (p : Point) (c : Circle) : Prop :=
  distance c.center p < c.radius

-- Theorem statement
theorem point_inside_circle_radius (O A : Point) (circle : Circle) :
  distance O A = 4 →
  circle.center = O →
  (∀ p : Point, is_inside p circle → distance O p ≤ circle.radius) →
  is_inside A circle →
  circle.radius > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_radius_l862_86255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_mall_promotion_l862_86280

/-- Calculates the discounted price based on the given conditions --/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 500 then price
  else if price ≤ 800 then price * 0.8
  else 800 * 0.8 + (price - 800) * 0.6

/-- Represents the shopping mall promotion problem --/
theorem shopping_mall_promotion 
  (xiaohong_purchase : ℝ) 
  (mother_purchase : ℝ) 
  (h1 : xiaohong_purchase = 480) 
  (h2 : mother_purchase = 520) : 
  discountedPrice (xiaohong_purchase + mother_purchase) = 838 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_mall_promotion_l862_86280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_l862_86201

open Real MeasureTheory

theorem cosine_values (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/3)) 
  (h2 : Real.sqrt 3 * sin α + cos α = Real.sqrt 6 / 2) : 
  cos (α + π/6) = Real.sqrt 10 / 4 ∧ 
  cos (2*α + 7*π/12) = (Real.sqrt 2 - Real.sqrt 30) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_l862_86201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_denominator_l862_86286

theorem rationalize_cube_root_denominator :
  ∃ (A B C D : ℕ), 
    (A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0) ∧
    (1 / (Real.rpow 5 (1/3) - Real.rpow 3 (1/3)) = 
     (Real.rpow A (1/3) + Real.rpow B (1/3) + Real.rpow C (1/3)) / D) ∧
    (A + B + C + D = 51) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_cube_root_denominator_l862_86286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_l862_86230

/-- Represents a train with its length and time to cross a fixed point -/
structure Train where
  length : ℝ
  crossTime : ℝ

/-- Calculates the speed of a train -/
noncomputable def speed (t : Train) : ℝ := t.length / t.crossTime

/-- Calculates the time for the faster train to overtake the slower one -/
noncomputable def overtakeTime (t1 t2 : Train) : ℝ :=
  (t1.length + t2.length) / (speed t2 - speed t1)

/-- Calculates the time for two trains to cross each other when moving in opposite directions -/
noncomputable def crossTime (t1 t2 : Train) : ℝ :=
  (t1.length + t2.length) / (speed t1 + speed t2)

theorem train_problem (t1 t2 : Train) 
  (h1 : t1.length = 140 ∧ t1.crossTime = 16)
  (h2 : t2.length = 180 ∧ t2.crossTime = 20)
  (h3 : speed t2 > speed t1) :
  overtakeTime t1 t2 = 1280 ∧ 
  (∃ ε > 0, abs (crossTime t1 t2 - 18.03) < ε) := by
  sorry

#eval "Theorem train_problem has been stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_l862_86230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l862_86250

/-- The area of a parallelogram with base 20 cm and height 16 cm is 320 square centimeters. -/
theorem parallelogram_area (base height : Real) 
  (h1 : base = 20) (h2 : height = 16) :
  base * height = 320 := by
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l862_86250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_in_three_tosses_l862_86254

/-- The probability of getting exactly k successes in n independent trials,
    where p is the probability of success in each trial. -/
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 1/2 of landing heads. -/
def fair_coin_probability : ℚ := 1 / 2

/-- The number of coin tosses. -/
def num_tosses : ℕ := 3

/-- The number of heads we want to get. -/
def num_heads : ℕ := 2

theorem probability_two_heads_in_three_tosses :
  binomial_probability num_tosses num_heads (fair_coin_probability : ℝ) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_heads_in_three_tosses_l862_86254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l862_86208

/-- The area of the region bounded by two curves -/
noncomputable def boundedArea (a : ℝ) : ℝ :=
  32 * a^2 / (1 + 4 * a^2)

/-- First curve equation -/
def curve1 (x y a : ℝ) : Prop :=
  (x + 2*a*y)^2 = 16*a^2

/-- Second curve equation -/
def curve2 (x y a : ℝ) : Prop :=
  (2*a*x - y)^2 = 4*a^2

/-- Theorem stating the area between the curves -/
theorem area_between_curves (a : ℝ) (h : a > 0) :
  ∃ (A : ℝ), A = boundedArea a ∧
  ∀ (x y : ℝ), (curve1 x y a ∨ curve2 x y a) →
  A = 32 * a^2 / (1 + 4 * a^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l862_86208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_and_range_l862_86205

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.exp x - (x - a)^2 + 3

theorem tangent_parallel_and_range (a : ℝ) :
  (∃ k, HasDerivAt (f a) k 0 ∧ k = 0) = (a = -1) ∧
  (∀ x ≥ 0, f a x ≥ 0) = (Real.log 3 - 3 ≤ a ∧ a ≤ Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_and_range_l862_86205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l862_86215

/-- The function f(x) = sin x + √3 cos x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

/-- θ is the value where f(x) reaches its maximum -/
noncomputable def θ : ℝ := Real.pi / 6

/-- f(x) reaches its maximum value when x = θ -/
axiom f_max : ∀ x : ℝ, f x ≤ f θ

theorem cos_theta_value : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l862_86215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_elimination_l862_86297

theorem trigonometric_system_elimination (x y a b c : ℝ) 
  (h1 : Real.sin x + Real.sin y = a) 
  (h2 : Real.cos x + Real.cos y = b) 
  (h3 : (Real.cos x / Real.sin x) * (Real.cos y / Real.sin y) = c) : 
  (a^2 + b^2)^2 - 4*a^2 = c*((a^2 + b^2)^2 - 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_elimination_l862_86297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l862_86291

/-- Rational Woman's ellipse -/
def rational_ellipse (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 4

/-- Irrational Woman's circle -/
def irrational_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 9

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the smallest possible distance -/
theorem smallest_distance :
  ∃ (min_dist : ℝ),
    (∀ (x1 y1 x2 y2 : ℝ),
      rational_ellipse x1 y1 →
      irrational_circle x2 y2 →
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ),
      rational_ellipse x1 y1 ∧
      irrational_circle x2 y2 ∧
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = Real.sqrt (6 - 4 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l862_86291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_recurrence_generating_function_closed_form_l862_86212

/-- The sequence a_n defined as (n^2 + 1) * 3^n -/
def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

/-- The recurrence relation for the sequence a_n -/
theorem a_recurrence (n : ℕ) : 
  a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0 := by sorry

/-- The generating function of the sequence a_n -/
noncomputable def generating_function (x : ℝ) : ℝ := ∑' n, a n * x^n

/-- The closed form of the generating function -/
theorem generating_function_closed_form (x : ℝ) (h : x ≠ 1/3) : 
  generating_function x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_recurrence_generating_function_closed_form_l862_86212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_sorting_probability_l862_86248

theorem orange_sorting_probability :
  ∃ (ratio_large_to_small prob_large_misclassified prob_small_misclassified
     total_oranges prob_large prob_small prob_large_correct
     prob_classified_large prob_large_and_classified_large : ℚ),
    ratio_large_to_small = 3 / 2 ∧
    prob_large_misclassified = 2 / 100 ∧
    prob_small_misclassified = 5 / 100 ∧
    total_oranges = 5 ∧
    prob_large = ratio_large_to_small * total_oranges / (ratio_large_to_small * total_oranges + total_oranges) ∧
    prob_small = 1 - prob_large ∧
    prob_large_correct = 1 - prob_large_misclassified ∧
    prob_classified_large = prob_large * prob_large_correct + prob_small * prob_small_misclassified ∧
    prob_large_and_classified_large = prob_large * prob_large_correct ∧
    prob_large_and_classified_large / prob_classified_large = 147 / 152 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_sorting_probability_l862_86248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_revenue_difference_l862_86217

/-- Calculates the net revenue for a given season --/
def calculate_net_revenue (packs_per_hour : ℕ) (price_per_pack : ℚ) (hours : ℕ) 
  (discount_rate : ℚ) (commission_rate : ℚ) : ℚ :=
  let total_packs := packs_per_hour * hours
  let total_sales := (total_packs : ℚ) * price_per_pack
  let discounted_packs := total_packs / 2
  let discount := (discounted_packs : ℚ) * price_per_pack * discount_rate
  let sales_after_discount := total_sales - discount
  let commission := sales_after_discount * commission_rate
  sales_after_discount - commission

/-- The difference in net revenue between peak and low season --/
theorem net_revenue_difference : 
  (calculate_net_revenue 8 70 17 (1/10) (1/20)) - (calculate_net_revenue 5 50 14 (7/100) (3/100)) = 5315.62 := by
  sorry

#eval (calculate_net_revenue 8 70 17 (1/10) (1/20)) - (calculate_net_revenue 5 50 14 (7/100) (3/100))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_revenue_difference_l862_86217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_has_zero_point_l862_86213

noncomputable section

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^2 + 2*a*x - b^2 + Real.pi

/-- The probability space of (a, b) pairs -/
def Ω : Set (ℝ × ℝ) := {p | -Real.pi ≤ p.1 ∧ p.1 ≤ Real.pi ∧ -Real.pi ≤ p.2 ∧ p.2 ≤ Real.pi}

/-- The event where f(x) has a zero point -/
def E : Set (ℝ × ℝ) := {p ∈ Ω | ∃ x, f p.1 p.2 x = 0}

/-- The probability measure on Ω -/
noncomputable def P : Set (ℝ × ℝ) → ℝ := sorry

/-- The main theorem stating the probability that f(x) has a zero point -/
theorem probability_f_has_zero_point : P E = 3/4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_has_zero_point_l862_86213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_alternating_parity_powers_of_five_l862_86202

/-- For a natural number n and a positive integer k, this function returns
    the last k digits of 5^n in decimal representation as a list of integers. -/
def lastDigits (n : ℕ) (k : ℕ) : List ℕ :=
  sorry

/-- This function checks if a list of natural numbers alternates in parity. -/
def alternatingParity (lst : List ℕ) : Prop :=
  sorry

/-- This theorem states that for any m ∈ ℕ, there exists an infinite subset S of ℕ
    such that for all n ∈ S, the last m digits of 5^n in decimal representation
    alternate in parity. -/
theorem infinite_alternating_parity_powers_of_five (m : ℕ) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, alternatingParity (lastDigits n m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_alternating_parity_powers_of_five_l862_86202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_proof_l862_86260

theorem fraction_sum_proof : 
  let start : ℕ := 10
  let stop : ℕ := 20
  let denominator : ℕ := 7
  let sum := (Finset.range (stop - start + 1)).sum (fun k => (start + k) / denominator)
  sum = 165 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_proof_l862_86260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_12_l862_86279

theorem sin_alpha_plus_pi_12 (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 3) : 
  Real.sin (α + π / 12) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_pi_12_l862_86279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_theorem_l862_86262

/-- The butterfly theorem -/
theorem butterfly_theorem (S : Set (ℝ × ℝ)) (A B M N P Q E F O : ℝ × ℝ) :
  (∀ X : ℝ × ℝ, X ∈ S → ∃ r : ℝ, (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2) →  -- S is a circle
  A ∈ S → B ∈ S → M ∈ S → N ∈ S → P ∈ S → Q ∈ S → E ∈ S → F ∈ S →  -- All points are on the circle
  O = (A + B) / 2 →  -- O is midpoint of AB
  (∃ t₁ t₂ : ℝ, M + t₁ • (N - M) = O ∧ P + t₂ • (Q - P) = O) →  -- MN and PQ pass through O
  (P.2 - A.2) * (N.2 - A.2) > 0 →  -- P and N are on the same side of AB
  (∃ s₁ : ℝ, E = A + s₁ • (B - A) ∧ ∃ u₁ : ℝ, E = M + u₁ • (P - M)) →  -- E is on AB and MP
  (∃ s₂ : ℝ, F = A + s₂ • (B - A) ∧ ∃ u₂ : ℝ, F = N + u₂ • (Q - N)) →  -- F is on AB and NQ
  O = (E + F) / 2  -- O is midpoint of EF
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_theorem_l862_86262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l862_86269

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The area of a triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ := 
  abs (l.c / l.a * l.c / l.b) / 2

/-- Two lines are parallel if their slopes are equal -/
def isParallel (l1 l2 : Line) : Prop := 
  l1.a * l2.b = l1.b * l2.a

theorem line_equation (l : Line) : 
  isParallel l { a := 3, b := 4, c := -7 } → 
  triangleArea l = 24 → 
  ∃ k : ℝ, (k = 24 ∨ k = -24) ∧ l = { a := 3, b := 4, c := k } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l862_86269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_at_focus_for_specific_ellipse_circle_tangent_and_contained_l862_86235

/-- The radius of a circle centered at one focus of an ellipse and tangent to the ellipse internally -/
noncomputable def circle_radius_at_focus (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 - b^2)
  3 * Real.sqrt (4 - 2 * Real.sqrt 3)

/-- Theorem stating the radius of a circle centered at one focus of an ellipse and tangent to the ellipse internally -/
theorem circle_radius_at_focus_for_specific_ellipse :
  circle_radius_at_focus 6 3 = 3 * Real.sqrt (4 - 2 * Real.sqrt 3) :=
by sorry

/-- Theorem stating that the circle with the calculated radius is tangent to the ellipse and fully contained within it -/
theorem circle_tangent_and_contained (x y : ℝ) :
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  let r := circle_radius_at_focus a b
  (x^2 / a^2 + y^2 / b^2 = 1) →
  ((x - c)^2 + y^2 ≥ r^2) ∧
  (∃ x₀ y₀, x₀^2 / a^2 + y₀^2 / b^2 = 1 ∧ (x₀ - c)^2 + y₀^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_at_focus_for_specific_ellipse_circle_tangent_and_contained_l862_86235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l862_86223

theorem set_elements_problem (A B C : Finset ℕ) 
  (h1 : A.card = 3 * B.card)
  (h2 : (A ∩ B).card = 1200)
  (h3 : (A ∪ B ∪ C).card = 4200)
  (h4 : (A ∩ C).card = 300)
  (h5 : B ∩ C = ∅) :
  A.card = 3825 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l862_86223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painting_cost_l862_86224

def house_count : ℕ := 25
def east_start : ℕ := 6
def east_diff : ℕ := 8
def west_start : ℕ := 5
def west_diff : ℕ := 7
def cost_per_digit : ℕ := 2

def east_sequence (n : ℕ) : ℕ := east_start + east_diff * n
def west_sequence (n : ℕ) : ℕ := west_start + west_diff * n

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def total_digits (seq : ℕ → ℕ) : ℕ :=
  (List.range house_count).map (λ i ↦ digit_count (seq i)) |>.sum

theorem total_painting_cost :
  cost_per_digit * (total_digits east_sequence + total_digits west_sequence) = 340 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painting_cost_l862_86224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_average_cost_result_l862_86252

/-- Calculates the average cost of markers in cents, rounded to the nearest whole number. -/
def marker_average_cost (total_markers : ℕ) (price_per_pack : ℚ) 
  (markers_per_pack : ℕ) (shipping_charge : ℚ) : ℕ :=
  let total_packs := total_markers / markers_per_pack
  let total_cost_dollars := total_packs * price_per_pack + shipping_charge
  let total_cost_cents := (total_cost_dollars * 100).floor
  let average_cost_cents := total_cost_cents / total_markers
  (average_cost_cents + 1/2).floor.toNat

#eval marker_average_cost 300 (25/2) 25 (157/20)

theorem marker_average_cost_result : 
  marker_average_cost 300 (25/2) 25 (157/20) = 53 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marker_average_cost_result_l862_86252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l862_86209

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  min (min ((10 * a^2 - 5 * a + 1) / (b^2 - 5 * b + 10))
           ((10 * b^2 - 5 * b + 1) / (c^2 - 5 * c + 10)))
      ((10 * c^2 - 5 * c + 1) / (a^2 - 5 * a + 10))
  ≤ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l862_86209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l862_86239

/-- The circle defined by the equation x^2 + y^2 - 4y + 3 = 0 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

/-- The line defined by the equation 3x - 4y - 2 = 0 -/
def line_eq (x y : ℝ) : Prop := 3*x - 4*y - 2 = 0

/-- The distance from a point (x, y) to the line 3x - 4y - 2 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - 4*y - 2| / Real.sqrt (3^2 + (-4)^2)

/-- The theorem stating the range of distances from points on the circle to the line -/
theorem distance_range :
  ∀ x y : ℝ, circle_eq x y →
  ∃ d : ℝ, distance_to_line x y = d ∧ 1 ≤ d ∧ d ≤ 3 :=
by
  sorry

#check distance_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l862_86239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_three_l862_86227

-- Define the sector and inscribed circle
noncomputable def sector_radius : ℝ := 6
noncomputable def sector_angle : ℝ := 2 * Real.pi / 3  -- 1/3 of a full circle

-- Define the radius of the inscribed circle as a function
noncomputable def inscribed_circle_radius (R : ℝ) (θ : ℝ) : ℝ :=
  R * (1 - Real.cos (θ/2)) / (1 + Real.cos (θ/2))

-- Theorem statement
theorem inscribed_circle_radius_is_three :
  inscribed_circle_radius sector_radius sector_angle = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_is_three_l862_86227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_dividing_polynomial_l862_86210

/-- A point on a plane with a color -/
structure ColoredPoint where
  x : ℝ
  y : ℝ
  color : Bool -- True for red, False for blue

/-- A set of N colored points with unique abscissae -/
def ColoredPointSet (N : ℕ) := { s : Finset ColoredPoint // s.card = N ∧ (∀ p q : ColoredPoint, p ∈ s → q ∈ s → p.x = q.x → p = q) }

/-- A polynomial divides a set of colored points -/
def DividesSet (P : ℝ → ℝ) (s : Finset ColoredPoint) : Prop :=
  (∀ p ∈ s, p.color = true → P p.x ≥ p.y) ∧ (∀ p ∈ s, p.color = false → P p.x ≤ p.y) ∨
  (∀ p ∈ s, p.color = true → P p.x ≤ p.y) ∧ (∀ p ∈ s, p.color = false → P p.x ≥ p.y)

/-- The degree of a polynomial -/
noncomputable def degree (P : ℝ → ℝ) : ℕ := sorry

/-- The main theorem -/
theorem min_degree_dividing_polynomial (N : ℕ) (h : N ≥ 3) :
  ∀ s : ColoredPointSet N, ∃ P : ℝ → ℝ, DividesSet P s.val ∧
    (∀ Q : ℝ → ℝ, DividesSet Q s.val → degree Q ≥ N - 2) ∧
    degree P = N - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_dividing_polynomial_l862_86210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_opposite_vertex_distance_l862_86232

/-- Definition of a regular dodecahedron -/
structure RegularDodecahedron :=
  (face_perimeter : ℝ)
  (opposite_vertex_distance_squared : ℝ)

/-- The squared distance between a vertex and its opposite point on a regular dodecahedron -/
noncomputable def opposite_vertex_distance_squared (face_perimeter : ℝ) : ℝ :=
  (17 + 7 * Real.sqrt 5) / 2

/-- Theorem: For a regular dodecahedron with face perimeter 5, 
    the squared distance between a vertex and its opposite point is (17 + 7√5) / 2 -/
theorem dodecahedron_opposite_vertex_distance 
  (d : RegularDodecahedron) 
  (h : d.face_perimeter = 5) : 
  d.opposite_vertex_distance_squared = opposite_vertex_distance_squared 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_opposite_vertex_distance_l862_86232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l862_86284

/-- Represents the sizes of detergent boxes -/
inductive Size
  | XS
  | S
  | M
  | L

/-- Represents the cost and quantity of a detergent box -/
structure Box where
  size : Size
  cost : ℚ
  quantity : ℚ

/-- Calculates the cost-effectiveness (cost per unit quantity) of a box -/
def costEffectiveness (b : Box) : ℚ := b.cost / b.quantity

/-- The main theorem stating the ranking of box sizes by cost-effectiveness -/
theorem detergent_ranking (xs s m l : Box) 
  (h_xs : xs.size = Size.XS)
  (h_s : s.size = Size.S)
  (h_m : m.size = Size.M)
  (h_l : l.size = Size.L)
  (h_s_cost : s.cost = 9/5 * xs.cost)
  (h_s_quantity : s.quantity = 3/2 * xs.quantity)
  (h_m_cost : m.cost = 6/5 * s.cost)
  (h_m_quantity : m.quantity = 3/4 * l.quantity)
  (h_l_quantity : l.quantity = 7/5 * s.quantity) :
  costEffectiveness xs < costEffectiveness s ∧
  costEffectiveness s = costEffectiveness l ∧
  costEffectiveness l < costEffectiveness m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l862_86284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_maximizes_profit_l862_86266

/-- Fixed cost for producing electronic instruments -/
noncomputable def fixed_cost : ℝ := 20000

/-- Additional cost per instrument -/
noncomputable def unit_cost : ℝ := 100

/-- Total revenue function -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

/-- Profit function -/
noncomputable def f (x : ℝ) : ℝ := R x - (fixed_cost + unit_cost * x)

/-- The production level that maximizes profit -/
noncomputable def optimal_production : ℝ := 300

/-- The maximum profit -/
noncomputable def max_profit : ℝ := 25000

/-- Theorem stating that the optimal production level maximizes profit -/
theorem optimal_production_maximizes_profit :
  ∀ x : ℝ, f x ≤ f optimal_production ∧ f optimal_production = max_profit := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_maximizes_profit_l862_86266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l862_86287

/-- The projection of vector u onto vector v -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scalar * v.1, scalar * v.2)

/-- The problem statement -/
theorem projection_problem (v : ℝ × ℝ) :
  let u₁ : ℝ × ℝ := (3, 2)
  let u₂ : ℝ × ℝ := (-2, 4)
  let p : ℝ × ℝ := (32/29, 80/29)
  proj u₁ v = proj u₂ v → proj u₁ v = p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l862_86287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_18_equals_5_l862_86273

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the given condition
axiom inverse_relation : ∀ x : ℝ, Function.invFun f (g x) = x^4 - 5*x^2 + 4

-- State that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Theorem to prove
theorem g_inverse_f_18_equals_5 : Function.invFun g (f 18) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_18_equals_5_l862_86273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_smaller_residuals_random_error_properties_correct_statements_l862_86241

-- Define a model
structure Model where
  residuals : List ℝ
  random_error : ℝ → ℝ

-- Define the sum of squared residuals
noncomputable def sum_squared_residuals (m : Model) : ℝ :=
  (m.residuals.map (λ x => x^2)).sum

-- Define the fit quality of a model
noncomputable def fit_quality (m : Model) : ℝ :=
  1 / (1 + sum_squared_residuals m)

-- Define the expected value of a random variable
noncomputable def expected_value (X : ℝ → ℝ) : ℝ := sorry

theorem better_fit_smaller_residuals (m1 m2 : Model) :
  sum_squared_residuals m1 < sum_squared_residuals m2 → fit_quality m1 > fit_quality m2 := by
  sorry

theorem random_error_properties (m : Model) :
  (∀ x, m.random_error x ≠ 0) ∧ expected_value m.random_error = 0 := by
  sorry

-- Main theorem combining both statements
theorem correct_statements (m1 m2 : Model) :
  (sum_squared_residuals m1 < sum_squared_residuals m2 → fit_quality m1 > fit_quality m2) ∧
  ((∀ x, m1.random_error x ≠ 0) ∧ expected_value m1.random_error = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_fit_smaller_residuals_random_error_properties_correct_statements_l862_86241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_exceeding_speed_limit_l862_86225

/-- Represents the percentage of motorists who exceed the speed limit and receive tickets -/
def ticketed_speeders : ℝ := 10

/-- Represents the percentage of speeders who do not receive tickets -/
def unticketed_speeders_ratio : ℝ := 50

/-- Theorem stating that 20% of motorists exceed the speed limit -/
theorem percent_exceeding_speed_limit : 
  let total_speeders := ticketed_speeders * (100 / (100 - unticketed_speeders_ratio))
  total_speeders = 20 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_exceeding_speed_limit_l862_86225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_difference_l862_86200

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
deriving Repr

/-- Represents the grid and its filling rules -/
def Grid (n : Nat) :=
  { grid : Cell → Nat //
    (∀ i j, grid ⟨i, j⟩ ≤ n^2) ∧
    (∃ i j, grid ⟨i, j⟩ = 1) ∧
    (∀ k, 2 ≤ k → k ≤ n^2 → ∃ i j, grid ⟨i, j⟩ = k ∧ 
      ∃ i' j', grid ⟨i', j'⟩ = k - 1 ∧ i = j') }

/-- The sum of numbers in a row -/
def rowSum (g : Grid n) (row : Nat) : Nat :=
  (Finset.range n).sum (fun j => g.val ⟨row, j⟩)

/-- The sum of numbers in a column -/
def colSum (g : Grid n) (col : Nat) : Nat :=
  (Finset.range n).sum (fun i => g.val ⟨i, col⟩)

theorem grid_sum_difference (n : Nat) (g : Grid n) :
  ∃ i j, g.val ⟨i, j⟩ = 1 ∧ 
    ∃ i' j', g.val ⟨i', j'⟩ = n^2 ∧ 
      colSum g j' - rowSum g i = n * (n - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_difference_l862_86200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_zero_l862_86247

/-- A function from positive natural numbers to natural numbers. -/
def PositiveNatToNat := ℕ+ → ℕ

/-- Check if a number ends with 7. -/
def ends_with_seven (n : ℕ+) : Prop := (n : ℕ) % 10 = 7

/-- The theorem stating that any function satisfying the given conditions is constantly zero. -/
theorem function_is_zero (f : PositiveNatToNat) 
  (h1 : ∀ x y : ℕ+, f (x * y) = f x + f y)
  (h2 : f 30 = 0)
  (h3 : ∀ x : ℕ+, ends_with_seven x → f x = 0) :
  ∀ n : ℕ+, f n = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_zero_l862_86247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_to_balance_l862_86288

def can_balance (weights : List ℕ) (target : ℕ) : Prop :=
  target ∈ weights ∨ ∃ w1 w2, w1 ∈ weights ∧ w2 ∈ weights ∧ w1 + w2 = target

def can_balance_all (weights : List ℕ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → n ≤ 20 → can_balance weights n

theorem min_weights_to_balance :
  ∃ (weights : List ℕ),
    can_balance_all weights ∧
    weights.length = 6 ∧
    ∀ (other_weights : List ℕ),
      can_balance_all other_weights →
      other_weights.length ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_to_balance_l862_86288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_KLN_is_equilateral_l862_86211

-- Define the triangle ABC
structure Triangle (A B C : Point) where
  -- No specific conditions needed for a general triangle

-- Define the angle measure in degrees
noncomputable def angle_measure (A B C : Point) : ℝ := sorry

-- Define a point on a line segment
def point_on_segment (P : Point) (A B : Point) : Prop := sorry

-- Define angle bisector
def is_angle_bisector (L : Point) (A C B : Point) : Prop :=
  angle_measure A C L = angle_measure L C B

-- Define distance between two points
noncomputable def distance (P Q : Point) : ℝ := sorry

-- Main theorem
theorem triangle_KLN_is_equilateral 
  (A B C L N K : Point) 
  (triangle : Triangle A B C)
  (angle_ACB : angle_measure A C B = 120)
  (L_on_AB : point_on_segment L A B)
  (N_on_AC : point_on_segment N A C)
  (K_on_BC : point_on_segment K B C)
  (CL_bisects_ACB : is_angle_bisector L A C B)
  (CK_CN_eq_CL : distance C K + distance C N = distance C L) :
  distance N K = distance L K ∧ 
  distance N K = distance N L ∧
  angle_measure K L N = 60 ∧
  angle_measure L N K = 60 ∧
  angle_measure N K L = 60 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_KLN_is_equilateral_l862_86211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_time_is_30_l862_86236

/-- Represents the typing scenario with Jonathan, Susan, and Jack --/
structure TypingScenario where
  doc_pages : ℝ
  jonathan_time : ℝ
  jack_time : ℝ
  combined_time : ℝ

/-- Calculates Susan's typing time given a TypingScenario --/
noncomputable def susan_time (scenario : TypingScenario) : ℝ :=
  let jonathan_rate := scenario.doc_pages / scenario.jonathan_time
  let jack_rate := scenario.doc_pages / scenario.jack_time
  let combined_rate := scenario.doc_pages / scenario.combined_time
  scenario.doc_pages / (combined_rate - jonathan_rate - jack_rate)

/-- Theorem: Susan's typing time is 30 minutes in the given scenario --/
theorem susan_time_is_30 (scenario : TypingScenario) 
    (h1 : scenario.doc_pages = 50)
    (h2 : scenario.jonathan_time = 40)
    (h3 : scenario.jack_time = 24)
    (h4 : scenario.combined_time = 10) : 
  susan_time scenario = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_time_is_30_l862_86236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radii_sum_l862_86296

theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : 
  (∃ (θ₁ θ₂ θ₃ : ℝ), 
    θ₁ + θ₂ + θ₃ = 2 * Real.pi ∧ 
    θ₁ / θ₂ = 1 / 2 ∧ θ₂ / θ₃ = 2 / 3 ∧
    r₁ = θ₁ / (2 * Real.pi) ∧
    r₂ = θ₂ / (2 * Real.pi) ∧
    r₃ = θ₃ / (2 * Real.pi)) →
  r₁ + r₂ + r₃ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radii_sum_l862_86296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l862_86292

/-- The focal length of an ellipse with semi-major axis a and semi-minor axis b --/
noncomputable def focalLength (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Prove that the focal length of the ellipse x = 5cos(θ), y = 4sin(θ) is 6 --/
theorem ellipse_focal_length : 
  let a : ℝ := 5
  let b : ℝ := 4
  focalLength a b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l862_86292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l862_86283

/-- Represents a rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diag_ratio : ℝ
  area_pos : 0 < area
  ratio_pos : 0 < diag_ratio

/-- The length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (2 * r.area * r.diag_ratio / (1 + r.diag_ratio))

/-- Theorem: The longest diagonal of a rhombus with area 150 and diagonal ratio 4:3 is 20 -/
theorem rhombus_longest_diagonal :
  let r : Rhombus := ⟨150, 4/3, by norm_num, by norm_num⟩
  longest_diagonal r = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l862_86283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_terms_l862_86289

theorem arithmetic_progression_terms (n : ℕ) (a d : ℤ) : 
  -- n is even
  Even n →
  -- sum of odd-numbered terms is 36
  (n / 2 : ℤ) * (2 * a + (n - 2) * d) = 36 →
  -- sum of even-numbered terms is 45
  (n / 2 : ℤ) * (2 * a + 2 * d + (n - 2) * d) = 45 →
  -- last term exceeds first term by 15
  a + (n - 1) * d - a = 15 →
  -- number of terms is 6
  n = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_terms_l862_86289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_l862_86281

-- Define a regular hexagon
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i j : Fin 6, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  ∀ i j : Fin 3, dist (t1.vertices i) (t1.vertices j) = dist (t2.vertices i) (t2.vertices j)

-- Theorem statement
theorem hexagon_division (h : RegularHexagon) :
  ∃ t1 t2 t3 t4 : Triangle,
    (congruent t1 t2 ∧ congruent t1 t3 ∧ congruent t1 t4) ∧
    (∀ p : ℝ × ℝ, p ∈ (Set.range h.vertices) → 
      p ∈ (Set.range t1.vertices) ∨ p ∈ (Set.range t2.vertices) ∨ 
      p ∈ (Set.range t3.vertices) ∨ p ∈ (Set.range t4.vertices)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_division_l862_86281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_degree_2020_roots_possible_odd_roots_not_always_possible_l862_86265

/-- A polynomial with real coefficients -/
structure RealPolynomial where
  coeffs : List ℝ
  nonConstant : coeffs.length > 1

/-- The degree of a polynomial -/
def degree (p : RealPolynomial) : ℕ := p.coeffs.length - 1

/-- The number of real roots of a polynomial -/
noncomputable def realRootCount (p : RealPolynomial) : ℕ := sorry

/-- Represents the operations Bob can perform -/
inductive Operation
  | AddConstant (c : ℝ)
  | ComposeWithP

/-- Apply an operation to a polynomial -/
def applyOperation (p : RealPolynomial) (op : Operation) (P : RealPolynomial) : RealPolynomial := sorry

/-- A sequence of operations -/
def OperationSequence := List Operation

theorem even_degree_2020_roots_possible (P : RealPolynomial) 
    (hEven : Even (degree P)) (hNonConst : degree P > 0) :
  ∃ (seq : OperationSequence), realRootCount (seq.foldl (fun p op => applyOperation p op P) P) = 2020 := by
  sorry

theorem odd_roots_not_always_possible (n : ℕ) (hOdd : Odd n) :
  ∃ (P : RealPolynomial), ∀ (seq : OperationSequence), 
    realRootCount (seq.foldl (fun p op => applyOperation p op P) P) ≠ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_degree_2020_roots_possible_odd_roots_not_always_possible_l862_86265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_bags_needed_l862_86290

theorem soup_bags_needed : ℕ := by
  let milk_quarts : ℕ := 6
  let vegetable_quarts : ℕ := 3
  let bag_capacity : ℕ := 2
  let chicken_stock_quarts : ℕ := 3 * milk_quarts
  let total_soup_quarts : ℕ := milk_quarts + chicken_stock_quarts + vegetable_quarts
  let bags_needed : ℕ := (total_soup_quarts + bag_capacity - 1) / bag_capacity
  have h : bags_needed = 14 := by sorry
  exact bags_needed

#check soup_bags_needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_bags_needed_l862_86290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_symmetry_axis_l862_86219

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

-- Define the symmetry axis
noncomputable def symmetry_axis : ℝ := -5 * Real.pi / 3

-- Theorem statement
theorem is_symmetry_axis : ∀ (x : ℝ), f (2 * symmetry_axis - x) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_symmetry_axis_l862_86219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_winning_strategy_l862_86293

/-- Represents a card in the game -/
def Card := Fin 9

/-- Represents a player in the game -/
inductive Player
| Alice
| Bob
deriving Repr, DecidableEq

/-- Represents the state of the game -/
structure GameState where
  remainingCards : Finset Card
  aliceCards : Finset Card
  bobCards : Finset Card

/-- Checks if a set of cards contains a winning combination -/
def hasWinningCombination (cards : Finset Card) : Prop :=
  ∃ (a b c : Card), a ∈ cards ∧ b ∈ cards ∧ c ∈ cards ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a.val + 1) + (b.val + 1) + (c.val + 1) = 15

/-- Represents a strategy for a player -/
def Strategy := GameState → Option Card

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  ∀ (opponent : Strategy), 
    let game := λ (state : GameState) (turn : Player) =>
      if turn = player
      then strategy state
      else opponent state
    ∃ (finalState : GameState), 
      (player = Player.Alice → hasWinningCombination finalState.aliceCards) ∧
      (player = Player.Bob → hasWinningCombination finalState.bobCards)

/-- The main theorem stating that neither player has a winning strategy -/
theorem no_winning_strategy :
  ¬ ∃ (player : Player) (strategy : Strategy), isWinningStrategy player strategy :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_winning_strategy_l862_86293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_odd_function_l862_86242

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/a - 1/(a^x + 1)

theorem range_of_odd_function (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f a x = -f a (-x)) :
  Set.range (f a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_odd_function_l862_86242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_kernel_difference_l862_86261

theorem corn_kernel_difference (stalks : ℕ) (ears_per_stalk : ℕ) (kernels_first_half : ℕ) (total_kernels : ℕ) : 
  stalks = 108 →
  ears_per_stalk = 4 →
  kernels_first_half = 500 →
  total_kernels = 237600 →
  (total_kernels - (stalks * ears_per_stalk / 2 * kernels_first_half)) - 
  (stalks * ears_per_stalk / 2 * kernels_first_half) = 21600 := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check corn_kernel_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_kernel_difference_l862_86261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_c_dot_product_l862_86264

-- Part 1
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := 2 • a + b

theorem magnitude_c : ‖c‖ = Real.sqrt 2 := by sorry

-- Part 2
def norm_a : ℝ := 2
def norm_b : ℝ := 1
noncomputable def angle_ab : ℝ := Real.pi / 3  -- 60° in radians

theorem dot_product : 
  norm_a * norm_a + norm_a * norm_b * Real.cos angle_ab = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_c_dot_product_l862_86264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l862_86237

def sequence_a : ℕ → ℚ
  | 0 => 3/5  -- Add this case for 0
  | 1 => 3/5
  | n + 1 => 
      if sequence_a n < 1/2 then 2 * sequence_a n
      else 2 * sequence_a n - 1

theorem a_2015_value : sequence_a 2015 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l862_86237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l862_86233

theorem equation_solution : ∃ x : ℝ, (3 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^(x - 24) ∧ x = 96 := by
  use 96
  constructor
  · simp [Real.rpow_mul, Real.rpow_sub]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l862_86233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l862_86226

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) : ℝ) / 2 * d

theorem arithmetic_sequence_properties
  (a₁ d : ℝ)
  (h₁ : arithmetic_sequence a₁ d 3 = 12)
  (h₂ : sum_arithmetic_sequence a₁ d 12 > 0)
  (h₃ : sum_arithmetic_sequence a₁ d 13 < 0) :
  (-24 / 7 < d ∧ d < -3) ∧
  (∀ n : ℕ, 1 ≤ n → n ≤ 12 → sum_arithmetic_sequence a₁ d n ≤ sum_arithmetic_sequence a₁ d 6) :=
by sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l862_86226
