import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_visit_eiffel_tower_l1221_122135

-- Define the universe
variable {U : Type}

-- Define the sets
variable (A : Set U) -- American women from Minnesota
variable (H : Set U) -- Women wearing hats with flowers
variable (E : Set U) -- Visitors to the Eiffel Tower

-- Define the conditions
axiom condition1 : A ⊆ (H ∩ E)
axiom condition2 : (H ∩ E) ⊆ A

-- State the theorem
theorem not_all_visit_eiffel_tower : ¬(A ∩ H ⊆ E) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_visit_eiffel_tower_l1221_122135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_rearrangement_bounds_l1221_122126

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def is_coprime_with_24 (n : ℕ) : Prop := Nat.Coprime n 24

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * 100000000 + (n / 10)

theorem nine_digit_rearrangement_bounds :
  ∀ B : ℕ,
  is_nine_digit B →
  is_coprime_with_24 B →
  B > 666666666 →
  ∃ A : ℕ,
  A = move_last_to_first B ∧
  166666667 ≤ A ∧ A ≤ 999999998 :=
by
  intro B h_nine_digit h_coprime h_lower_bound
  use move_last_to_first B
  constructor
  · rfl
  constructor
  · sorry  -- Proof for lower bound
  · sorry  -- Proof for upper bound


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_rearrangement_bounds_l1221_122126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_25_l1221_122140

/-- The total cost function for producing x units -/
noncomputable def total_cost (x : ℝ) : ℝ := 1200 + (2/75) * x^3

/-- The relationship between price P and quantity x -/
def price_quantity_relation (k : ℝ) (x : ℝ) (P : ℝ) : Prop := P^2 = k / x

/-- The constant k in the price-quantity relation -/
def k : ℝ := 25 * 10^4

/-- The profit function -/
noncomputable def profit (x : ℝ) : ℝ := x * (500 / Real.sqrt x) - total_cost x

/-- Theorem stating that profit is maximized at x = 25 -/
theorem profit_maximized_at_25 :
  ∃ (x_max : ℝ), x_max > 0 ∧ 
  (∀ (x : ℝ), x > 0 → profit x ≤ profit x_max) ∧
  x_max = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_25_l1221_122140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_intersection_circle_and_line_intersection_DE_on_AB_AD_equals_BE_parabola_equation_and_line_equation_l1221_122107

/-- Parabola with vertex at origin and focus at (1/2, 0) -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- Circle M with equation (x - 3/2)² + (y - 8)² = 49 -/
def circle_M (x y : ℝ) : Prop := (x - 3/2)^2 + (y - 8)^2 = 49

/-- Line l passing through F(1/2, 0) -/
def line_l (x y : ℝ) (t : ℝ) : Prop := x = t*y + 1/2

theorem parabola_and_line_intersection :
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    (∃ t : ℝ, line_l A.1 A.2 t ∧ line_l B.1 B.2 t) :=
by sorry

theorem circle_and_line_intersection :
  ∃ (D E : ℝ × ℝ), 
    circle_M D.1 D.2 ∧ 
    circle_M E.1 E.2 ∧ 
    (∃ t : ℝ, line_l D.1 D.2 t ∧ line_l E.1 E.2 t) :=
by sorry

theorem DE_on_AB :
  ∀ (A B D E : ℝ × ℝ),
    (∃ t : ℝ, line_l A.1 A.2 t ∧ line_l B.1 B.2 t ∧ line_l D.1 D.2 t ∧ line_l E.1 E.2 t) →
    (D.1 - A.1) * (B.1 - D.1) + (D.2 - A.2) * (B.2 - D.2) ≥ 0 ∧
    (E.1 - A.1) * (B.1 - E.1) + (E.2 - A.2) * (B.2 - E.2) ≥ 0 :=
by sorry

theorem AD_equals_BE :
  ∀ (A B D E : ℝ × ℝ),
    (∃ t : ℝ, line_l A.1 A.2 t ∧ line_l B.1 B.2 t ∧ line_l D.1 D.2 t ∧ line_l E.1 E.2 t) →
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = (B.1 - E.1)^2 + (B.2 - E.2)^2 :=
by sorry

theorem parabola_equation_and_line_equation :
  (∀ x y : ℝ, parabola x y ↔ y^2 = 2*x) ∧
  (∃ t : ℝ, ∀ x y : ℝ, line_l x y t ↔ 2*x - 4*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_intersection_circle_and_line_intersection_DE_on_AB_AD_equals_BE_parabola_equation_and_line_equation_l1221_122107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1221_122152

noncomputable section

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi/4) = Real.sqrt 2

/-- Curve C in parametric form -/
def curve_C (t x y : ℝ) : Prop :=
  x = t ∧ y = t^2

/-- Cartesian form of line l -/
def line_l_cartesian (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- Cartesian form of curve C -/
def curve_C_cartesian (x y : ℝ) : Prop :=
  y = x^2

/-- Intersection points of line l and curve C -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  line_l_cartesian x₁ y₁ ∧ curve_C_cartesian x₁ y₁ ∧
  line_l_cartesian x₂ y₂ ∧ curve_C_cartesian x₂ y₂ ∧
  x₁ ≠ x₂

/-- Distance between two points -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_distance (A B : ℝ × ℝ) :
  intersection_points A B → distance A B = 3 * Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1221_122152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadrilateral_area_l1221_122148

-- Define the function g on a finite domain
def g : Fin 4 → ℝ := sorry

-- Define the area of a quadrilateral given by four points
def quadrilateralArea (p₁ p₂ p₃ p₄ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem transformed_quadrilateral_area 
  (h : quadrilateralArea (1, g 1) (2, g 2) (3, g 3) (4, g 4) = 50) :
  quadrilateralArea (1/3, 3 * g 1) (2/3, 3 * g 2) (1, 3 * g 3) (4/3, 3 * g 4) = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadrilateral_area_l1221_122148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_correct_l1221_122163

/-- Conversion from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (7, Real.pi / 3, -3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (3.5, 7 * Real.sqrt 3 / 2, -3)

/-- Theorem stating that the conversion is correct -/
theorem cylindrical_to_rectangular_correct :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_correct_l1221_122163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1221_122177

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < Real.pi) ∧
  (0 < B) ∧ (B < Real.pi) ∧
  (0 < C) ∧ (C < Real.pi) ∧
  (2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a) ∧
  (Real.cos B = 3/5) →
  (A = Real.pi/3) ∧ (Real.sin (B - C) = (7 * Real.sqrt 3 - 24) / 50) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1221_122177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vikki_hours_worked_l1221_122128

/-- Calculates the number of hours worked given the take-home pay and deduction rates -/
noncomputable def hours_worked (hourly_rate : ℝ) (tax_rate : ℝ) (insurance_rate : ℝ) (union_dues : ℝ) (take_home_pay : ℝ) : ℝ :=
  (take_home_pay + union_dues) / (hourly_rate * (1 - tax_rate - insurance_rate))

/-- Proves that given the specified conditions, the number of hours worked is 42 -/
theorem vikki_hours_worked :
  let hourly_rate : ℝ := 10
  let tax_rate : ℝ := 0.20
  let insurance_rate : ℝ := 0.05
  let union_dues : ℝ := 5
  let take_home_pay : ℝ := 310
  hours_worked hourly_rate tax_rate insurance_rate union_dues take_home_pay = 42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vikki_hours_worked_l1221_122128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1221_122184

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2*θ) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l1221_122184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1221_122145

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x - 2) / (x + 1 : ℚ) ≤ 0 ∧ x + 1 ≠ 0}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1221_122145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_digit_agreement_l1221_122192

/-- Given that 3^18, 3^19, and 3^20 have the same digit in the 10^2 place,
    this theorem proves that:
    1) There exist three other consecutive powers of 3 with the same digit in the 10^2 place
    2) There exist more than three consecutive powers of 3 with the same digit in the 10^2 place -/
theorem power_of_three_digit_agreement :
  ∃ (n : ℕ) (d : ℕ),
    (n ≠ 18) ∧
    (d < 10) ∧
    (∀ k : ℕ, k ∈ Finset.range 3 → (3^(n+k) / 100 : ℕ) % 10 = d) ∧
    (∃ (m : ℕ), m > 3 ∧ ∀ k : ℕ, k ∈ Finset.range m → (3^(n+k) / 100 : ℕ) % 10 = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_digit_agreement_l1221_122192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1221_122161

-- Define the constants
noncomputable def a : ℝ := Real.log 4 / Real.log 0.3
noncomputable def b : ℝ := Real.log 0.2 / Real.log 0.3
noncomputable def c : ℝ := (1 / Real.exp 1) ^ Real.pi

-- State the theorem
theorem order_of_abc : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1221_122161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_j_value_l1221_122144

/-- Two lines in ℝ³ are coplanar if and only if their direction vectors and the vector connecting their points are linearly dependent. -/
def are_coplanar (p₁ p₂ v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
    a • v₁ + b • v₂ = c • (p₂ - p₁)

theorem coplanar_lines_j_value (j : ℝ) :
  let p₁ : ℝ × ℝ × ℝ := (3, 2, 5)
  let p₂ : ℝ × ℝ × ℝ := (2, 5, 6)
  let v₁ : ℝ × ℝ × ℝ := (2, -1, j)
  let v₂ : ℝ × ℝ × ℝ := (j, 3, 2)
  are_coplanar p₁ p₂ v₁ v₂ → j = -6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_lines_j_value_l1221_122144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1221_122106

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  c * Real.sin A = a * Real.cos C

theorem triangle_ABC_properties
  (A B C : ℝ) (a b c : ℝ)
  (h : triangle_ABC A B C a b c) :
  C = Real.pi/4 ∧
  (∃ (max : ℝ), max = 2 ∧
    ∀ A' B' : ℝ, triangle_ABC A' B' (Real.pi/4) a b c →
      Real.sqrt 3 * Real.sin A' - Real.cos (B' + Real.pi/4) ≤ max) ∧
  Real.sqrt 3 * Real.sin (Real.pi/3) - Real.cos (5*Real.pi/12 + Real.pi/4) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1221_122106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_eq_one_fifth_l1221_122132

/-- The function f as defined in the problem -/
noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2) - 2*a)^2

/-- The theorem statement -/
theorem exists_x0_implies_a_eq_one_fifth :
  ∀ a : ℝ, (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ a ≤ 4/5) → a = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x0_implies_a_eq_one_fifth_l1221_122132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_approx_cube_edge_length_frustum_surface_area_exact_l1221_122170

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  edge_length : a > 0

/-- Represents the frustum ABCA₁FK in the cube -/
structure Frustum (a : ℝ) where
  cube : Cube a
  volume : ℝ
  volume_eq : volume = 13608

/-- Calculates the surface area of the frustum -/
noncomputable def surface_area (f : Frustum a) : ℝ :=
  (11 + 6 * Real.sqrt 2 + 3 * Real.sqrt 5) * a^2 / 8

/-- Theorem stating the surface area of the frustum is approximately 4243 square units -/
theorem frustum_surface_area_approx (a : ℝ) (f : Frustum a) :
  ∃ ε > 0, |surface_area f - 4243| < ε := by
  sorry

/-- Theorem stating the edge length of the cube is 36 -/
theorem cube_edge_length (a : ℝ) (f : Frustum a) : a = 36 := by
  sorry

/-- Theorem stating the exact surface area of the frustum -/
theorem frustum_surface_area_exact (a : ℝ) (f : Frustum a) :
  surface_area f = (11 + 6 * Real.sqrt 2 + 3 * Real.sqrt 5) * 36^2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_surface_area_approx_cube_edge_length_frustum_surface_area_exact_l1221_122170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_one_eq_zero_l1221_122179

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.rpow (a * x + b) (1/3)

-- State the theorem
theorem inverse_f_one_eq_zero (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Function.invFun f 1 = 0) ↔ ((1 - b) / a = 0) := by
  sorry

-- Additional helper lemma
lemma f_at_zero (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  f a b ((1 - b) / a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_one_eq_zero_l1221_122179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_drawing_area_ratio_l1221_122150

/-- Represents a trapezoid with upper base a, lower base b, and height h -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ := (t.a + t.b) / 2 * t.h

/-- The area of the intuitive diagram of a trapezoid drawn using the oblique drawing method -/
noncomputable def intuitiveDiagramArea (t : Trapezoid) : ℝ := (t.a + t.b) / 2 * (Real.sqrt 2 * t.h / 4)

theorem oblique_drawing_area_ratio (t : Trapezoid) :
  intuitiveDiagramArea t / trapezoidArea t = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_drawing_area_ratio_l1221_122150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_triple_l1221_122160

theorem unique_solution_triple : ∃! (x y z : ℝ),
  (1/3 * min x y + 2/3 * max x y = 2017) ∧
  (1/3 * min y z + 2/3 * max y z = 2018) ∧
  (1/3 * min z x + 2/3 * max z x = 2019) ∧
  x = 2019 ∧ y = 2016 ∧ z = 2019 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_triple_l1221_122160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_rotation_and_scaling_l1221_122155

noncomputable section

def rotation_angle : ℝ := Real.pi / 6  -- 30 degrees in radians
def scale_factor : ℝ := 2

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 3, -1],
    ![1, Real.sqrt 3]]

theorem transformation_is_rotation_and_scaling :
  transformation_matrix = scale_factor • 
    ![![Real.cos rotation_angle, -Real.sin rotation_angle],
      ![Real.sin rotation_angle, Real.cos rotation_angle]] := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_is_rotation_and_scaling_l1221_122155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1221_122171

theorem triangle_problem (A B C a b c : ℝ) : 
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  2 * Real.cos A * (c * Real.cos B + b * Real.cos C) = a ∧
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 ∧
  c^2 + a * b * Real.cos C + a^2 = 4 →
  A = Real.pi/3 ∧ a = Real.sqrt 21 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1221_122171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l1221_122173

/-- Calculates the monthly salary given savings rate, expense increase, and final savings --/
noncomputable def calculate_salary (initial_savings_rate : ℝ) (expense_increase_rate : ℝ) (final_savings : ℝ) : ℝ :=
  final_savings / (initial_savings_rate - initial_savings_rate * expense_increase_rate)

/-- Theorem stating the relationship between savings, expense increase, and salary --/
theorem salary_calculation 
  (initial_savings_rate : ℝ) 
  (expense_increase_rate : ℝ) 
  (final_savings : ℝ) 
  (h1 : initial_savings_rate = 0.10)
  (h2 : expense_increase_rate = 0.05)
  (h3 : final_savings = 400) :
  ∃ ε > 0, |calculate_salary initial_savings_rate expense_increase_rate final_savings - 7272.73| < ε := by
  sorry

-- Using #eval with noncomputable functions is not possible
-- Instead, we can state a lemma about the approximate value
lemma salary_approx :
  ∃ ε > 0, |calculate_salary 0.10 0.05 400 - 7272.73| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_calculation_l1221_122173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1221_122123

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16

-- Define the midpoint of the chord
def chord_midpoint : ℝ × ℝ := (1, -1)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := x - 4*y - 5 = 0

-- Theorem statement
theorem chord_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧ 
    chord_midpoint = ((x₁ + x₂)/2, (y₁ + y₂)/2) →
    ∀ (x y : ℝ), line_equation x y ↔ 
      ∃ (t : ℝ), x = (1-t)*x₁ + t*x₂ ∧ y = (1-t)*y₁ + t*y₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l1221_122123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_minus_cot_l1221_122143

theorem cos_double_minus_cot (α : Real) : 
  0 < α ∧ α < π/2 → 
  Real.cos α = 2 * Real.sqrt 5 / 5 → 
  Real.cos (2 * α) - Real.cos α / Real.sin α = -7/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_minus_cot_l1221_122143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octopus_family_size_l1221_122139

/-- Represents the number of octopus children of each color initially -/
def initial_count : ℕ := sorry

/-- Represents the number of blue octopuses that became striped -/
def changed_count : ℕ := sorry

theorem octopus_family_size :
  -- Equal initial counts
  initial_count = initial_count ∧ initial_count = initial_count →
  -- Count of blue and white after change
  (initial_count - changed_count) + initial_count = 10 →
  -- Count of white and striped after change
  initial_count + (initial_count + changed_count) = 18 →
  -- Total number of children
  3 * initial_count = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octopus_family_size_l1221_122139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_l1221_122109

-- Define the point in rectangular coordinates
noncomputable def point_rectangular : ℝ × ℝ × ℝ := (4, 4 * Real.sqrt 3, -3)

-- Define the point in cylindrical coordinates
noncomputable def point_cylindrical : ℝ × ℝ × ℝ := (8, Real.pi / 3, -3)

-- Theorem statement
theorem rectangular_to_cylindrical :
  let (x, y, z) := point_rectangular
  let (r, θ, z') := point_cylindrical
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ ∧
  z = z' :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_l1221_122109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1221_122147

theorem sum_of_repeating_decimals :
  (∃ (x y : ℚ), x = 2/3 ∧ y = 7/9 ∧ x + y = 13/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1221_122147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_max_y_diff_l1221_122186

/-- The maximum difference between the y-coordinates of the intersection points of
    y = 4 - 2x^2 + x^3 and y = 2 + 2x^2 + x^3 is √2/2. -/
theorem intersection_max_y_diff :
  let f (x : ℝ) := 4 - 2 * x^2 + x^3
  let g (x : ℝ) := 2 + 2 * x^2 + x^3
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (a b : ℝ), a ∈ intersection_points ∧ b ∈ intersection_points ∧
    |f a - f b| = Real.sqrt 2 / 2 ∧
    ∀ (x y : ℝ), x ∈ intersection_points → y ∈ intersection_points →
      |f x - f y| ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_max_y_diff_l1221_122186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_37_l1221_122112

theorem sum_of_divisors_37 : (Finset.sum (Nat.divisors 37) id) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_37_l1221_122112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_rate_calculation_l1221_122191

/-- Given a boat with speed in still water and its travel time and distance downstream,
    calculate the rate of the stream. -/
theorem stream_rate_calculation (boat_speed travel_time distance : ℝ) 
    (h1 : boat_speed = 16)
    (h2 : travel_time = 5)
    (h3 : distance = 105) : 
  distance / travel_time - boat_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_rate_calculation_l1221_122191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_pies_exist_l1221_122174

/-- Represents the filling of a pie -/
inductive Filling
  | Apple
  | Cherry

/-- Represents the preparation method of a pie -/
inductive Preparation
  | Fried
  | Baked

/-- Represents a pie with a filling and preparation method -/
structure Pie where
  filling : Filling
  preparation : Preparation

/-- Theorem: If there are at least three different types of pies available,
    then it is possible to select two pies that differ in both filling and preparation method -/
theorem two_different_pies_exist (pies : Finset Pie) 
    (h : Finset.card pies ≥ 3) :
    ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ 
    p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_pies_exist_l1221_122174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l1221_122167

/-- Calculates the initial distance between two trains given their speeds and meeting time -/
noncomputable def initial_distance (speed1 speed2 : ℝ) (meeting_time : ℝ) : ℝ :=
  (speed1 + speed2) * meeting_time

/-- Converts km/h to m/s -/
noncomputable def km_per_hour_to_m_per_s (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem train_distance_theorem (length1 length2 : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) :
  length1 = 100 →
  length2 = 200 →
  speed1 = 54 →
  speed2 = 72 →
  meeting_time = 23.99808015358771 →
  ∃ (ε : ℝ), ε > 0 ∧ 
    |initial_distance (km_per_hour_to_m_per_s speed1) (km_per_hour_to_m_per_s speed2) meeting_time - 839.93| < ε := by
  sorry

#check train_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l1221_122167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_4_eq_half_l1221_122185

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if abs x ≤ 1 then Real.sqrt x else 1 / x

-- State the theorem
theorem f_of_f_4_eq_half : f (f 4) = 1/2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_4_eq_half_l1221_122185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1221_122146

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

-- Define curve C₁
def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

-- Define curve C₂ in polar coordinates
def curve_C2_polar (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- Define the intersection points
def point_A : ℝ × ℝ := curve_C1 (2 * Real.pi / 3)
def point_B : ℝ × ℝ := line_l (2 / Real.sqrt 3)

-- State the theorem
theorem intersection_distance :
  Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2) = 4 - Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1221_122146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_discontinuities_l1221_122116

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def HasInverseNegative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = -x

-- Define the theorem
theorem infinite_discontinuities
  (h : HasInverseNegative f) :
  ∃ S : Set ℝ, (Set.Infinite S) ∧ (∀ x ∈ S, ¬ContinuousAt f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_discontinuities_l1221_122116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strip_ratio_l1221_122153

/-- A strip in an n × n grid square crossword -/
structure Strip (n : ℕ) where
  direction : Bool  -- True for horizontal, False for vertical
  position : Fin n  -- Position of the strip
  length : Fin n    -- Length of the strip

/-- The crossword puzzle -/
structure Crossword (n : ℕ) where
  strips : List (Strip n)

/-- The minimum number of strips needed to cover the crossword -/
def min_cover_strips (c : Crossword n) : ℕ :=
  sorry  -- Implementation details omitted

/-- The ratio of total strips to minimum covering strips -/
noncomputable def strip_ratio (c : Crossword n) : ℚ :=
  (c.strips.length : ℚ) / (min_cover_strips c : ℚ)

/-- The maximum strip ratio for any n × n crossword -/
theorem max_strip_ratio (n : ℕ) :
  (∀ c : Crossword n, strip_ratio c ≤ 1 + (n : ℚ) / 2) ∧
  (∃ c : Crossword n, strip_ratio c = 1 + (n : ℚ) / 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strip_ratio_l1221_122153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_tax_rate_calculation_l1221_122172

noncomputable def john_tax_rate : ℚ := 30 / 100
noncomputable def ingrid_tax_rate : ℚ := 40 / 100
def john_income : ℚ := 58000
def ingrid_income : ℚ := 72000

noncomputable def combined_tax_rate : ℚ := 
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)

theorem combined_tax_rate_calculation :
  (combined_tax_rate * 10000).floor / 10000 = 3554 / 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_tax_rate_calculation_l1221_122172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1221_122105

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 49}
def P : ℝ × ℝ := (8, 6)
def S (k : ℝ) : ℝ × ℝ := (0, k)
def Q : ℝ × ℝ := (7, 0)
def R : ℝ × ℝ := (10, 0)

-- State the theorem
theorem find_k :
  P ∈ larger_circle ∧
  (∃ k, S k ∈ smaller_circle) ∧
  ‖R - Q‖ = 3 →
  ∃ k, S k ∈ smaller_circle ∧ k = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l1221_122105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1221_122151

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define line AB passing through F
def line_AB (t : ℝ) (x y : ℝ) : Prop := x = t * y + 1

-- Define points A, B, and E on the ellipse
def point_on_ellipse_and_line (t : ℝ) (x y : ℝ) : Prop :=
  ellipse x y ∧ line_AB t x y

-- Define the vector sum condition
def vector_sum_condition (xA yA xB yB xE yE : ℝ) : Prop :=
  xE = xA + xB ∧ yE = yA + yB

-- Define perpendicular condition for QE and AB
def perpendicular_condition (t : ℝ) : Prop := t * (-1/t) = -1

-- Define parallel condition for EF and FQ
def parallel_condition (xE yE xQ yQ : ℝ) : Prop :=
  (xE - 1) * (yQ - 0) = (yE - 0) * (xQ - 1)

-- Theorem statement
theorem ellipse_theorem (t : ℝ) (xA yA xB yB xE yE xQ yQ : ℝ) 
  (h1 : point_on_ellipse_and_line t xA yA)
  (h2 : point_on_ellipse_and_line t xB yB)
  (h3 : ellipse xE yE)
  (h4 : vector_sum_condition xA yA xB yB xE yE)
  (h5 : perpendicular_condition t)
  (h6 : parallel_condition xE yE xQ yQ)
  (h7 : t > 0) : 
  (t = Real.sqrt 2 / 2) ∧ 
  (∃ (S : ℝ), S ≤ 16/9 ∧ S = (abs (xB - xA) * abs (xQ - xE)) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1221_122151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_slope_l1221_122122

def circle1_center : ℝ × ℝ := (20, 100)
def circle2_center : ℝ × ℝ := (25, 85)
def circle3_center : ℝ × ℝ := (28, 95)
def circle_radius : ℝ := 4

def line_point : ℝ × ℝ := (25, 85)

-- Helper functions (not implemented, just for context)
noncomputable def area_left_of_line (center : ℝ × ℝ) (radius : ℝ) (line : ℝ × ℝ × ℝ) : ℝ := sorry
noncomputable def area_right_of_line (center : ℝ × ℝ) (radius : ℝ) (line : ℝ × ℝ × ℝ) : ℝ := sorry

theorem equal_area_line_slope :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y - m * (x - 25) - 85 = 0 → 
      (area_left_of_line circle1_center circle_radius (m, 25, 85) +
       area_left_of_line circle2_center circle_radius (m, 25, 85) +
       area_left_of_line circle3_center circle_radius (m, 25, 85) =
       area_right_of_line circle1_center circle_radius (m, 25, 85) +
       area_right_of_line circle2_center circle_radius (m, 25, 85) +
       area_right_of_line circle3_center circle_radius (m, 25, 85))) ∧
    abs m = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_slope_l1221_122122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1221_122101

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem f_range : 
  {y : ℝ | ∃ x > -1, f x = y} = Set.Ici 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1221_122101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l1221_122195

open Complex

theorem complex_fraction_evaluation : 
  let z : ℂ := exp (I * (π / 12))
  ((z ^ 30 + 1) / (I - 1) = -I) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l1221_122195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_shape_properties_l1221_122194

noncomputable section

structure Rectangle (a b : ℝ) where
  ab_pos : a > 0
  ad_pos : b > 0

noncomputable def v_shape_area (a b x : ℝ) : ℝ := (b * x * (4 * a - 5 * x)) / (2 * (a - x))

theorem v_shape_properties (a b x : ℝ) (rect : Rectangle a b) (h_x : 0 < x ∧ x < a / 2) :
  let q := (a - x) / (2 * a)
  (1/4 : ℝ) < q ∧ q < (1/2 : ℝ) ∧
  v_shape_area a b x = (b * x * (4 * a - 5 * x)) / (2 * (a - x)) ∧
  (v_shape_area a b ((a * (5 - Real.sqrt 5)) / 10) = (a * b) / 2) ∧
  (∃ (x_max x_min : ℝ), 0 < x_max ∧ x_max < a / 2 ∧
                        0 < x_min ∧ x_min < a / 2 ∧
                        ∀ (y : ℝ), 0 < y ∧ y < a / 2 →
                          v_shape_area a b y ≤ v_shape_area a b x_max ∧
                          v_shape_area a b x_min ≤ v_shape_area a b y) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_shape_properties_l1221_122194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_w_l1221_122154

noncomputable def vector_w : ℝ × ℝ := sorry

theorem smallest_norm_w :
  (‖vector_w + (4, -2)‖ = 10) →
  (∀ v : ℝ × ℝ, ‖v + (4, -2)‖ = 10 → ‖vector_w‖ ≤ ‖v‖) →
  ‖vector_w‖ = 10 - 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_norm_w_l1221_122154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1221_122102

theorem train_speed_problem (train1_length train1_speed train2_length distance_between time_to_cross : ℝ) 
  (h1 : train1_length = 100)
  (h2 : train1_speed = 10)
  (h3 : train2_length = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  let total_distance := train1_length + train2_length + distance_between
  let train2_speed := total_distance / time_to_cross
  train2_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1221_122102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1221_122118

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (150 * Real.pi)

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1/3) ^ x

-- Define the intersection points
def intersection_points : Set ℝ := {x ∈ interval | f x = g x}

-- State the theorem
theorem intersection_count : ∃ (S : Finset ℝ), S.card = 150 ∧ ∀ x ∈ S, x ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1221_122118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1221_122104

noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) + Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) : ℝ := f (x - 3 * Real.pi / 4) (Real.pi / 6)

theorem min_value_of_g (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, f x φ = f (Real.pi / 6 - x) φ) :
  ∃ x₀ ∈ Set.Icc (-Real.pi / 4) (Real.pi / 6), 
    ∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 6), g x₀ ≤ g x ∧ g x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1221_122104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_min_value_is_four_min_value_achieved_l1221_122117

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1 / (a * x)

/-- The slope of the tangent line to f(x) at x = 1 -/
noncomputable def k (a : ℝ) : ℝ := 3 * a + 1 / a

theorem min_slope_tangent_line (a : ℝ) (h : a ≥ 1) :
  ∀ b ≥ 1, k a ≤ k b := by
  sorry

theorem min_value_is_four (a : ℝ) (h : a ≥ 1) :
  k a ≥ 4 := by
  sorry

theorem min_value_achieved (h : ∃ a ≥ 1, k a = 4) :
  ∃ a ≥ 1, ∀ b ≥ 1, k a ≤ k b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_line_min_value_is_four_min_value_achieved_l1221_122117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cost_is_60_l1221_122193

/-- Given a cube with edge length, paper coverage rate, and total expenditure, 
    calculate the cost of paper per kg. -/
noncomputable def paper_cost_per_kg (edge_length : ℝ) (coverage_rate : ℝ) (total_expenditure : ℝ) : ℝ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_rate
  total_expenditure / paper_needed

/-- Theorem: The cost of paper per kg for a cube with given specifications is 60 Rs. -/
theorem paper_cost_is_60 :
  paper_cost_per_kg 10 20 1800 = 60 := by
  unfold paper_cost_per_kg
  -- The proof steps would go here, but for now we'll use sorry
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cost_is_60_l1221_122193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1221_122108

/-- Calculates the length of a second train given the parameters of two trains passing each other --/
theorem second_train_length 
  (length_train1 : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (time_to_cross : ℝ) 
  (h1 : length_train1 = 200)
  (h2 : speed_train1 = 20)
  (h3 : speed_train2 = 10)
  (h4 : time_to_cross = 49.9960003199744)
  (h5 : speed_train1 > speed_train2) :
  length_train1 + (speed_train1 - speed_train2) * time_to_cross - length_train1 = 299.960003199744 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1221_122108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1221_122110

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

-- Define the property of being an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Theorem statement
theorem odd_function_values :
  ∀ a b : ℝ, (is_odd (f a b)) → (a = -1/2 ∧ b = Real.log 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_l1221_122110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1221_122133

open Real Set

noncomputable def f (x : ℝ) := x * sin x + cos x

theorem f_increasing_on_interval :
  ∀ x ∈ Ioo (3 * π / 2) (5 * π / 2),
    x ∈ Ioo π (3 * π) →
    ∀ y ∈ Ioo (3 * π / 2) (5 * π / 2),
      x < y → f x < f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1221_122133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1221_122165

noncomputable section

/-- The distance between Alice and Bob in miles -/
def distance_AB : ℝ := 12

/-- The angle of elevation from Alice's position in radians -/
def angle_Alice : ℝ := Real.pi / 4

/-- The angle of elevation from Bob's position in radians -/
def angle_Bob : ℝ := Real.pi / 6

/-- The altitude of the airplane -/
def altitude : ℝ := Real.sqrt (144/7)

theorem airplane_altitude :
  ∃ (x y : ℝ),
    x^2 + y^2 = distance_AB^2 ∧
    x = altitude * Real.tan angle_Alice ∧
    y = altitude * Real.tan angle_Bob ∧
    altitude = Real.sqrt (144/7) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l1221_122165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_around_sqrt30_l1221_122138

theorem consecutive_integers_around_sqrt30 (a b : ℕ) :
  (min (Real.sqrt 30) (a : ℝ) = a) →
  (min (Real.sqrt 30) (b : ℝ) = Real.sqrt 30) →
  (b = a + 1) →
  (2 * a - b = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_around_sqrt30_l1221_122138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1221_122183

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1221_122183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_cos_equality_l1221_122156

theorem smallest_positive_cos_equality : 
  ∃ (x : ℝ), x > 0 ∧ 
  Real.cos x = Real.cos (3 * x) ∧ 
  (∀ (y : ℝ), y > 0 → Real.cos y = Real.cos (3 * y) → x ≤ y) ∧
  x = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_cos_equality_l1221_122156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_intersection_points_l1221_122130

/-- Represents the number of distinct intersection points for a given configuration of five lines -/
def IntersectionPoints : Type := Nat

/-- The set of all possible numbers of distinct intersection points for five lines -/
def PossibleIntersections : Finset Nat :=
  {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}

/-- The number of distinct lines -/
def NumLines : Nat := 5

/-- Theorem stating that the sum of all possible numbers of distinct intersection points is 53 -/
theorem sum_intersection_points :
  (PossibleIntersections.sum id) = 53 := by
  sorry

#eval PossibleIntersections.sum id

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_intersection_points_l1221_122130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_iff_perfect_square_l1221_122134

theorem odd_divisors_iff_perfect_square (n : ℕ) :
  1 ≤ n ∧ n ≤ 100 →
  (∃ k : ℕ, n = k^2) ↔ (Finset.card (Finset.filter (λ d : ℕ => d ∣ n) (Finset.range (n + 1))) % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_divisors_iff_perfect_square_l1221_122134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_30_8_equals_formula_l1221_122175

-- Define the logarithm base 30 of 8
noncomputable def log_30_8 : ℝ := Real.log 8 / Real.log 30

-- Define 'a' as the base 10 logarithm of 5
noncomputable def a : ℝ := Real.log 5 / Real.log 10

-- Define 'b' as the base 10 logarithm of 3
noncomputable def b : ℝ := Real.log 3 / Real.log 10

-- Theorem statement
theorem log_30_8_equals_formula : log_30_8 = (3 * (1 - a)) / (1 + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_30_8_equals_formula_l1221_122175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheets_calculation_total_sheets_used_l1221_122103

def total_sheets (first_three_classes last_three_classes students_first_three students_last_three
  sheets_study_guide_first sheets_study_guide_last : ℕ) : ℕ :=
  ((students_first_three * sheets_study_guide_first + 10 * sheets_study_guide_first +
    students_first_three * 2) * first_three_classes) +
  ((students_last_three * sheets_study_guide_last + 10 * sheets_study_guide_last +
    students_last_three * 2) * last_three_classes)

axiom total_classes : 3 + 3 = 6
axiom students_first : 22 = 22
axiom students_last : 18 = 18
axiom study_guide_first : 6 = 6
axiom study_guide_last : 4 = 4
axiom extra_copies_def : 10 = 10
axiom handout_sheets : 2 = 2

theorem sheets_calculation (first_three_classes last_three_classes students_first_three students_last_three
    sheets_study_guide_first sheets_study_guide_last : ℕ) :
  total_sheets first_three_classes last_three_classes students_first_three students_last_three
    sheets_study_guide_first sheets_study_guide_last =
  ((students_first_three * sheets_study_guide_first + 10 * sheets_study_guide_first +
    students_first_three * 2) * first_three_classes) +
  ((students_last_three * sheets_study_guide_last + 10 * sheets_study_guide_last +
    students_last_three * 2) * last_three_classes) :=
by sorry

theorem total_sheets_used :
  total_sheets 3 3 22 18 6 4 = 1152 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheets_calculation_total_sheets_used_l1221_122103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1221_122141

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (1 + Real.sin (2 * x)) + Real.cos (2 * x)

-- State the theorem
theorem function_properties :
  ∃ (m : ℝ),
    (f m (π / 4) = 2) ∧
    (m = 1) ∧
    (∀ x, f m x ≥ 1 - Real.sqrt 2) ∧
    (∀ k : ℤ, f m (k * π - 3 * π / 8) = 1 - Real.sqrt 2) ∧
    (∀ x, f m x = 1 - Real.sqrt 2 → ∃ k : ℤ, x = k * π - 3 * π / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1221_122141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterclockwise_rotation_90_degrees_l1221_122180

-- Define rotation direction
inductive RotationDirection
  | Clockwise
  | Counterclockwise

-- Define rotation in degrees
def rotationDegrees (direction : RotationDirection) (degrees : Int) : Int :=
  match direction with
  | RotationDirection.Clockwise => -degrees
  | RotationDirection.Counterclockwise => degrees

-- Theorem statement
theorem counterclockwise_rotation_90_degrees :
  rotationDegrees RotationDirection.Counterclockwise 90 = 90 :=
by
  -- Unfold the definition of rotationDegrees
  unfold rotationDegrees
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterclockwise_rotation_90_degrees_l1221_122180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l1221_122196

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N = ![![3, 2], ![3, -3]] ∧
  N.vecMul ![1, 1] = ![5, 0] ∧
  N.vecMul ![0, 3] = ![6, -9] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l1221_122196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_45_days_later_l1221_122137

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to advance a day by one -/
def advanceDayOne (start : DayOfWeek) : DayOfWeek :=
  match start with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to advance a day by a given number of days -/
def advanceDay (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDay (advanceDayOne start) n

/-- Theorem: 45 days after a Monday is a Thursday -/
theorem birthday_45_days_later (start : DayOfWeek) :
  start = DayOfWeek.Monday → advanceDay start 45 = DayOfWeek.Thursday :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_45_days_later_l1221_122137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l1221_122149

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := 2016 * x^2 - Real.sin x + b + 2

-- State the theorem
theorem odd_function_sum_zero (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 4) (2 * a - 2), f x a b = -f (-x) a b) →
  f a a b + f b a b = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l1221_122149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_buses_is_eight_l1221_122187

-- Define the variables
noncomputable def num_vans : ℝ := 6.0
noncomputable def people_per_van : ℝ := 6.0
noncomputable def people_per_bus : ℝ := 18.0
noncomputable def extra_people_in_buses : ℝ := 108.0

-- Define the function to calculate the number of buses
noncomputable def calculate_num_buses : ℝ :=
  (num_vans * people_per_van + extra_people_in_buses) / people_per_bus

-- Theorem statement
theorem num_buses_is_eight :
  calculate_num_buses = 8.0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_buses_is_eight_l1221_122187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_in_interval_l1221_122121

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x + Real.pi / 3)

theorem g_range_in_interval :
  let a := 0
  let b := 5 * Real.pi / 24
  ∀ y ∈ Set.Icc (-1) 2, ∃ x ∈ Set.Icc a b, g x = y ∧
  ∀ x ∈ Set.Icc a b, -1 ≤ g x ∧ g x ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_in_interval_l1221_122121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_iff_submodular_l1221_122129

universe u

variable {S : Type u} [Finite S]

def MonotoneDecreasing (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, X ⊆ Y → f Y ≤ f X

theorem monotone_decreasing_iff_submodular (f : Set S → ℝ) :
  (∀ X Y : Set S, f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y) ↔
  (∀ a : S, MonotoneDecreasing (fun X : Set S ↦ f (X ∪ {a}) - f X)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_iff_submodular_l1221_122129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_schemes_l1221_122176

/-- The number of ways to select 5 courses out of 8, with at most one of two specific courses -/
theorem course_selection_schemes : 
  (let total_courses : Nat := 8
   let courses_to_select : Nat := 5
   let special_courses : Nat := 2

   /- Number of ways to select 5 out of 6 courses (excluding both special courses) -/
   let scenario1 : Nat := Nat.choose (total_courses - special_courses) courses_to_select

   /- Number of ways to select 1 out of 2 special courses and 4 out of 6 remaining courses -/
   let scenario2 : Nat := Nat.choose special_courses 1 * Nat.choose (total_courses - special_courses) (courses_to_select - 1)

   /- Total number of course selection schemes -/
   scenario1 + scenario2) = 36 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_selection_schemes_l1221_122176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_fraction_l1221_122136

noncomputable def total_distance : ℝ := 280
noncomputable def first_stop_fraction : ℝ := 1/2
noncomputable def final_leg_distance : ℝ := 105

theorem maria_trip_fraction :
  let remaining_after_first_stop := total_distance * (1 - first_stop_fraction)
  let fraction_between_stops := (remaining_after_first_stop - final_leg_distance) / remaining_after_first_stop
  fraction_between_stops = 1/4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maria_trip_fraction_l1221_122136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_from_intersecting_circles_l1221_122162

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def Line (p q : Point) := {r : Point | ∃ t : ℝ, r.x = p.x + t * (q.x - p.x) ∧ r.y = p.y + t * (q.y - p.y)}

def Chord (p q : Point) := Line p q

-- Define the given conditions
def intersecting_circles (c1 c2 : Circle) : Prop := sorry

def common_chord (c1 c2 : Circle) (a b : Point) : Prop := sorry

def point_on_line (p : Point) (l : Set Point) : Prop := p ∈ l

def chord_through_point (c : Circle) (p q r : Point) : Prop := sorry

def cyclic_quadrilateral (k l m n : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_from_intersecting_circles 
  (c1 c2 : Circle) (a b p k l m n : Point) :
  intersecting_circles c1 c2 →
  common_chord c1 c2 a b →
  point_on_line p (Line a b) →
  chord_through_point c1 k m p →
  chord_through_point c2 l n p →
  cyclic_quadrilateral k l m n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_from_intersecting_circles_l1221_122162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_growth_rate_is_ten_percent_l1221_122115

/-- The average monthly growth rate from January to March, given initial and final profits -/
noncomputable def average_monthly_growth_rate (initial_profit final_profit : ℝ) : ℝ :=
  Real.sqrt (final_profit / initial_profit) - 1

theorem profit_growth_rate_is_ten_percent :
  let initial_profit : ℝ := 3000
  let final_profit : ℝ := 3630
  let growth_rate := average_monthly_growth_rate initial_profit final_profit
  growth_rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_growth_rate_is_ten_percent_l1221_122115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1221_122113

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define the chord passing through left focus
def passes_through_left_focus (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
  left_focus.1 = (1 - t) * x₁ + t * x₂ ∧
  left_focus.2 = (1 - t) * y₁ + t * y₂

-- Define the incircle length condition
def incircle_length_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let a := 3
  let perimeter := 2 * a + 2 * a
  perimeter = 2 * Real.pi * 1  -- Incircle length is 2π, so radius is 1

theorem ellipse_chord_theorem (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ ∧ 
  is_on_ellipse x₂ y₂ ∧
  passes_through_left_focus x₁ y₁ x₂ y₂ ∧
  incircle_length_condition x₁ y₁ x₂ y₂ →
  |y₂ - y₁| = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l1221_122113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1221_122158

-- Define the ellipse properties
noncomputable def major_axis_length : ℝ := 12
noncomputable def eccentricity : ℝ := 1/3

-- Define the standard form of an ellipse equation
def is_standard_ellipse_equation (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1

-- Theorem statement
theorem ellipse_equation :
  let a := major_axis_length / 2
  let c := a * eccentricity
  let b := Real.sqrt (a^2 - c^2)
  (is_standard_ellipse_equation (a^2) (b^2) (λ x y ↦ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)) ∨
  (is_standard_ellipse_equation (b^2) (a^2) (λ x y ↦ (x^2 / (b^2)) + (y^2 / (a^2)) = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1221_122158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l1221_122190

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 0) (h3 : f a (-1) = 1) :
  Function.invFun (f a) 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l1221_122190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1221_122199

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π / 2 + α) * Real.sin (π + α) * Real.tan (3 * π + α)) /
  (Real.cos (3 * π / 2 + α) * Real.sin (-α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1221_122199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_graph_l1221_122111

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
noncomputable def area_enclosed : ENNReal := 96

/-- The equation of the graph -/
def graph_equation (p : ℝ × ℝ) : Prop := abs p.1 + abs (3 * p.2) = 12

/-- Theorem stating that the area enclosed by the graph is 96 square units -/
theorem area_of_graph :
  ∃ (A : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ A ↔ graph_equation p) ∧
    (MeasureTheory.volume A = area_enclosed) := by
  sorry

#check area_of_graph

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_graph_l1221_122111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_a_wins_match_l1221_122157

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_Wins
| B_Wins

/-- Represents the state of the match -/
structure MatchState :=
  (games_played : ℕ)
  (a_wins : ℕ)
  (b_wins : ℕ)

/-- The probability of player A winning a single game -/
noncomputable def p_a_wins : ℝ := 2/3

/-- The probability of player B winning a single game -/
noncomputable def p_b_wins : ℝ := 1/3

/-- The maximum number of games in a match -/
def max_games : ℕ := 6

/-- Determines if the match is over given the current state -/
def match_over (state : MatchState) : Bool :=
  state.a_wins ≥ state.b_wins + 2 ∨ 
  state.b_wins ≥ state.a_wins + 2 ∨ 
  state.games_played = max_games

/-- Determines the winner of the match given the final state -/
def match_winner (state : MatchState) : Option GameOutcome :=
  if state.a_wins ≥ state.b_wins + 2 then some GameOutcome.A_Wins
  else if state.b_wins ≥ state.a_wins + 2 then some GameOutcome.B_Wins
  else if state.games_played = max_games then
    if state.a_wins > state.b_wins then some GameOutcome.A_Wins
    else if state.b_wins > state.a_wins then some GameOutcome.B_Wins
    else none  -- Tie after 6 games, use first game winner (not modeled here)
  else none

/-- The main theorem to prove -/
theorem probability_a_wins_match : 
  p_a_wins = 74/81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_a_wins_match_l1221_122157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_sum_l1221_122164

/-- A quartic polynomial Q with specific values at 0, 1, and -1 -/
def QuarticPolynomial (k : ℝ) (Q : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), 
    (∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + 4 * k) ∧
    Q 0 = 4 * k ∧
    Q 1 = 5 * k ∧
    Q (-1) = 9 * k

theorem quartic_sum (k : ℝ) (Q : ℝ → ℝ) (hQ : QuarticPolynomial k Q) :
  Q 2 + Q (-2) = 48 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_sum_l1221_122164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equal_roots_ten_trials_probability_finite_decimal_five_trials_l1221_122114

-- Define the set of numbers
def numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to check if roots are equal
def roots_equal (a b c : ℕ) : Prop := b^2 = 4*a*c

-- Define the function to check if the product of roots is a finite decimal less than 1
def roots_product_finite_decimal_less_than_one (a c : ℕ) : Prop :=
  c < a ∧ (a = 2 ∨ a = 4 ∨ a = 5 ∨ a = 8 ∨ (a = 6 ∧ c = 3))

-- Theorem for part (a)
theorem probability_equal_roots_ten_trials :
  ∀ (a b c : ℕ), a ∈ numbers → b ∈ numbers → c ∈ numbers →
  a ≠ b → b ≠ c → a ≠ c →
  (1 - (251/252)^10 : ℚ) = 0.039 := by
  sorry

-- Theorem for part (b)
theorem probability_finite_decimal_five_trials :
  ∀ (a c : ℕ), a ∈ numbers → c ∈ numbers → a ≠ c →
  (5 * (2/9) * (7/9)^4 : ℚ) = 0.407 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equal_roots_ten_trials_probability_finite_decimal_five_trials_l1221_122114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1221_122181

theorem angle_sum_theorem (α β : ℝ) : 
  α ∈ Set.Ioo 0 π → β ∈ Set.Ioo 0 π → Real.cos α + Real.cos β - Real.cos (α + β) = 3/2 → 2*α + β = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1221_122181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1221_122169

/-- Given two workers p and q who can complete a task individually in 15 and 10 days respectively,
    prove that they can complete the task together in 6 days. -/
theorem work_completion_time (work : ℝ) (h_work_pos : work > 0) : 
  (work / 15 + work / 10) * 6 = work := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1221_122169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_parallel_false_l1221_122166

-- Define a 2D plane
structure Plane :=
  (points : Type*)
  (vectors : Type*)
  (lines : Type*)

-- Define perpendicularity for vectors
def perpendicular_vectors (p : Plane) (a b : p.vectors) : Prop := sorry

-- Define perpendicularity for a vector and a line
def perpendicular_vector_line (p : Plane) (v : p.vectors) (l : p.lines) : Prop := sorry

-- Define parallel lines
def parallel_lines (p : Plane) (l1 l2 : p.lines) : Prop := sorry

-- Define a function to convert a vector to a line
def vector_to_line (p : Plane) (v : p.vectors) : p.lines := sorry

-- The statement to be proven false
theorem perpendicular_implies_parallel_false (p : Plane) :
  ¬ (∀ (a b : p.vectors) (c : p.lines),
    perpendicular_vectors p a b → 
    perpendicular_vector_line p a c → 
    parallel_lines p (vector_to_line p b) c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_parallel_false_l1221_122166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_scale_ratio_l1221_122188

/-- Represents the height of the actual statue in feet -/
noncomputable def actual_statue_height : ℝ := 90

/-- Represents the height of the scale model in inches -/
noncomputable def scale_model_height : ℝ := 6

/-- Represents the ratio of the actual statue height to the scale model height in feet per inch -/
noncomputable def statue_to_model_ratio : ℝ := actual_statue_height / scale_model_height

theorem statue_scale_ratio :
  statue_to_model_ratio = 15 := by
  -- Unfold the definitions
  unfold statue_to_model_ratio actual_statue_height scale_model_height
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_scale_ratio_l1221_122188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_proof_l1221_122100

-- Define the cost of milk tea
noncomputable def milk_tea_cost : ℚ := 2.40

-- Define the cost of a slice of cake
noncomputable def cake_slice_cost : ℚ := (3/4) * milk_tea_cost

-- Define the total cost function
noncomputable def total_cost (cake_slices : ℕ) (milk_tea_cups : ℕ) : ℚ :=
  (cake_slices : ℚ) * cake_slice_cost + (milk_tea_cups : ℚ) * milk_tea_cost

-- Theorem statement
theorem total_cost_proof :
  total_cost 2 1 = 6 := by
  -- Expand the definition of total_cost
  unfold total_cost
  -- Simplify the expression
  simp [cake_slice_cost, milk_tea_cost]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_proof_l1221_122100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteProduct_value_l1221_122189

/-- The infinite product representation --/
noncomputable def infiniteProduct : ℝ := 1/2 * ∏' n, Real.sqrt (1/2 + 1/2 * Real.cos (2 * Real.pi / (3 * 2^n)))

/-- Theorem stating that the infinite product equals 3/(4π) --/
theorem infiniteProduct_value : infiniteProduct = 3 / (4 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteProduct_value_l1221_122189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_cosine_l1221_122125

/-- A triangle with sides that are consecutive even integers and where the largest angle is thrice the smallest angle -/
structure SpecialTriangle where
  /-- The smallest side length divided by 2 -/
  n : ℕ
  /-- The smallest angle of the triangle -/
  smallest_angle : ℝ
  /-- The largest angle of the triangle -/
  largest_angle : ℝ
  /-- The sides are consecutive even integers -/
  sides_consecutive : List ℕ := [2*n, 2*n+2, 2*n+4]
  /-- The largest angle is thrice the smallest angle -/
  angle_ratio : largest_angle = 3 * smallest_angle
  /-- The angles form a valid triangle -/
  angle_sum : smallest_angle + largest_angle + (π - smallest_angle - largest_angle) = π
  /-- The sides form a valid triangle (triangle inequality) -/
  triangle_inequality : (2*n) + (2*n+2) > (2*n+4) ∧ (2*n) + (2*n+4) > (2*n+2) ∧ (2*n+2) + (2*n+4) > (2*n)

/-- The cosine of the smallest angle in a SpecialTriangle is 1/2 -/
theorem special_triangle_cosine (t : SpecialTriangle) : Real.cos t.smallest_angle = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_cosine_l1221_122125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_parallel_l1221_122198

-- Define structures for Plane and Line
structure Plane where
  -- Placeholder for plane properties
  dummy : Unit

structure Line where
  -- Placeholder for line properties
  dummy : Unit

-- Define relations between lines and planes
def perpendicular (l m : Line) : Prop :=
  sorry -- Definition to be implemented

def parallel_to_plane (l : Line) (α : Plane) : Prop :=
  sorry -- Definition to be implemented

def perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  sorry -- Definition to be implemented

-- Theorem statement
theorem line_plane_perpendicular_parallel (l m : Line) (α : Plane) :
  l ≠ m →
  perpendicular_to_plane m α →
  (parallel_to_plane l α → perpendicular l m) ∧
  ¬(perpendicular l m → parallel_to_plane l α) :=
by
  sorry -- Proof to be implemented

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_perpendicular_parallel_l1221_122198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_range_l1221_122119

theorem subset_implies_a_range (a : ℝ) : 
  let A : Set ℝ := {x | |x - 1| ≤ 2}
  let B : Set ℝ := {x | 2^x ≥ a}
  A ⊆ B → a ∈ Set.Iic (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_range_l1221_122119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_two_implies_x_equals_one_l1221_122142

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (2 : ℝ) ^ x else -x

-- State the theorem
theorem f_equals_two_implies_x_equals_one :
  ∀ x : ℝ, f x = 2 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_two_implies_x_equals_one_l1221_122142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_coordinate_l1221_122127

open Real

theorem smallest_x_coordinate (n : ℕ) : 
  (∀ k : ℕ, k < n → (k + 1) * (k + 2)^2 / (k^2 * (k + 4 : ℝ)) ≠ 163/162) ∧ 
  (n + 1) * (n + 2)^2 / (n^2 * (n + 4 : ℝ)) = 163/162 →
  n = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_coordinate_l1221_122127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1221_122178

/-- The radius of the circumscribed sphere of a regular tetrahedron -/
noncomputable def circumscribed_sphere_radius (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

/-- Theorem: The radius of the circumscribed sphere of a regular tetrahedron
    with base side length a and lateral edge angle α with the base plane
    is equal to a / (3 * sin(α)) -/
theorem circumscribed_sphere_radius_formula
  (a : ℝ) (α : ℝ)
  (h_a : a > 0)
  (h_α : 0 < α ∧ α < π / 2) :
  circumscribed_sphere_radius a α = a / (3 * Real.sin α) := by
  -- Unfold the definition of circumscribed_sphere_radius
  unfold circumscribed_sphere_radius
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_formula_l1221_122178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l1221_122168

-- Define the function f(x) = ln x - 2/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- Theorem statement
theorem zero_point_existence :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  -- The proof would go here, but we'll skip it
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_existence_l1221_122168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extremum_implies_a_geq_one_l1221_122197

/-- A function f: ℝ → ℝ has no extremum points if for all x in ℝ, 
    there exists a neighborhood of x where f is strictly monotonic -/
def has_no_extremum_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, (∀ y z, y ∈ Set.Ioo (x - ε) (x + ε) → z ∈ Set.Ioo (x - ε) (x + ε) → y < z → f y < f z) ∨
                    (∀ y z, y ∈ Set.Ioo (x - ε) (x + ε) → z ∈ Set.Ioo (x - ε) (x + ε) → y < z → f y > f z)

/-- The main theorem -/
theorem no_extremum_implies_a_geq_one (a : ℝ) :
  has_no_extremum_points (fun x => (1/3) * x^3 + x^2 + a*x) → a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extremum_implies_a_geq_one_l1221_122197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_distance_ratio_l1221_122131

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the theorem
theorem five_points_distance_ratio (p1 p2 p3 p4 p5 : Point) :
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 →
  let distances := [
    distance p1 p2, distance p1 p3, distance p1 p4, distance p1 p5,
    distance p2 p3, distance p2 p4, distance p2 p5,
    distance p3 p4, distance p3 p5,
    distance p4 p5
  ]
  let max_distance := List.maximum? distances
  let min_distance := List.minimum? distances
  ∀ max min, max_distance = some max → min_distance = some min →
    max / min ≥ 2 * Real.sin (54 * π / 180) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_distance_ratio_l1221_122131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1221_122124

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2)) (h2 : Real.tan α = 4 / 3) : Real.sin α = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1221_122124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangency_l1221_122182

noncomputable section

/-- Curve C₁: y² = px where y > 0 and p > 0 -/
def C₁ (p : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = p * x ∧ y > 0 ∧ p > 0}

/-- Curve C₂: y = e^(x+1) - 1 -/
def C₂ : Set (ℝ × ℝ) :=
  {(x, y) | y = Real.exp (x + 1) - 1}

/-- Point M on curve C₁ -/
def M (p : ℝ) : ℝ × ℝ := (4 / p, 2)

/-- Assume the existence of tangent lines for any point on the curves -/
axiom tangent_line_exists (C : Set (ℝ × ℝ)) (point : ℝ × ℝ) : 
  ∃ (line : ℝ × ℝ → ℝ), ∀ (x : ℝ × ℝ), x ∈ C → line x = 0

/-- Tangent line at point M on C₁ is also tangent to C₂ -/
def tangent_condition (p : ℝ) : Prop :=
  ∃ (t : ℝ × ℝ), t ∈ C₂ ∧ 
    (Classical.choose (tangent_line_exists (C₁ p) (M p))) t = 
    (Classical.choose (tangent_line_exists C₂ t)) t

theorem curve_tangency (p : ℝ) :
  (M p) ∈ C₁ p → tangent_condition p → (1/2) * p * Real.log ((4 * Real.exp 2) / p) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangency_l1221_122182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_equal_foci_l1221_122120

-- Define the two ellipses
def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

def ellipse2 (x y k : ℝ) : Prop := x^2 / (9 - k) + y^2 / (25 - k) = 1

-- Define the condition on k
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9

-- Define the foci distance for an ellipse
noncomputable def foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem ellipses_equal_foci :
  ∀ k, k_condition k →
  foci_distance 5 3 = foci_distance (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_equal_foci_l1221_122120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gabriel_paint_area_l1221_122159

theorem gabriel_paint_area : 
  (let num_bedrooms : ℕ := 4
   let length : ℕ := 14
   let width : ℕ := 12
   let height : ℕ := 9
   let unpaintable_area : ℕ := 80
   let wall_area_per_room : ℕ := 2 * (length * height + width * height)
   let paintable_area_per_room : ℕ := wall_area_per_room - unpaintable_area
   let total_paintable_area : ℕ := num_bedrooms * paintable_area_per_room
   total_paintable_area) = 1552 := by
  -- Expand definitions
  simp_arith
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gabriel_paint_area_l1221_122159
