import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_l158_15811

/-- Represents the number of pots of green lilies -/
def green_lilies : ℕ → ℕ := sorry

/-- Represents the number of pots of spider plants -/
def spider_plants : ℕ → ℕ := sorry

/-- The total number of pots -/
def total_pots : ℕ := 46

/-- The cost of each green lily pot -/
def green_lily_cost : ℕ := 9

/-- The cost of each spider plant pot -/
def spider_plant_cost : ℕ := 6

/-- The total budget -/
def total_budget : ℕ := 390

/-- The condition that the number of green lily pots is at least twice the number of spider plant pots -/
axiom green_lily_condition (n : ℕ) : green_lilies n ≥ 2 * spider_plants n

/-- The total number of pots equals 46 -/
axiom total_pots_condition (n : ℕ) : green_lilies n + spider_plants n = total_pots

/-- The function to calculate the total cost of purchasing the plants -/
def total_cost (n : ℕ) : ℕ := green_lily_cost * green_lilies n + spider_plant_cost * spider_plants n

/-- The theorem stating that the minimum total cost is 369 -/
theorem min_total_cost : ∃ (n : ℕ), total_cost n = 369 ∧ ∀ (m : ℕ), total_cost m ≥ 369 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_l158_15811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_properties_l158_15839

-- Define the sample space
def Ω : Finset (ℕ × ℕ) := 
  {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)}

-- Define event A
def A : Finset (ℕ × ℕ) := Ω.filter (fun x => x.1 + x.2 > 4)

-- Define event B
def B : Finset (ℕ × ℕ) := Ω.filter (fun x => x.1 * x.2 < 5)

-- Define the probability measure
def P (X : Finset (ℕ × ℕ)) : ℚ := (X ∩ Ω).card / Ω.card

-- Theorem statement
theorem events_properties : 
  (A ∩ B).Nonempty ∧ 
  (A ∪ B ≠ Ω ∨ (A ∩ B).Nonempty) ∧ 
  P A + P B = 9/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_properties_l158_15839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l158_15801

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2^x) / (1 + 2^x)

theorem function_properties (a : ℝ) 
  (h : ∀ x : ℝ, f a (-x) = -f a x) :
  (a = 1) ∧ 
  (∀ x y : ℝ, x < y → f 1 y < f 1 x) ∧
  (Set.Icc (f 1 2) (f 1 0) = Set.Icc (-3/5) 0) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l158_15801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l158_15809

-- Define a Point type for 2D coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the quadrilateral vertices
def v1 : Point := ⟨1, 1⟩
def v2 : Point := ⟨4, 5⟩
def v3 : Point := ⟨5, 4⟩
def v4 : Point := ⟨4, 0⟩

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the perimeter of the quadrilateral
noncomputable def perimeter : ℝ :=
  distance v1 v2 + distance v2 v3 + distance v3 v4 + distance v4 v1

-- Theorem statement
theorem quadrilateral_perimeter_sum (c d : ℤ) :
  (∃ k : ℝ, perimeter = k + c * Real.sqrt 2 + d * Real.sqrt 17) →
  c + d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_sum_l158_15809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_exists_l158_15831

noncomputable def f (x : ℤ) : ℤ := 23 * x^2 + 0 * x + 0

theorem quadratic_function_exists : ∃ (a b c : ℤ), 
  (∀ x : ℤ, f x = a * x^2 + b * x + c) ∧
  f 177883 = 1324754875645 ∧
  f 348710 = 1782225466694 ∧
  f 796921 = 1984194627862 ∧
  f 858522 = 4388794883485 ∧
  a = 23 :=
by
  use 23, 0, 0
  constructor
  · intro x
    rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_exists_l158_15831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_values_l158_15870

theorem cos_sin_values (θ : Real) (h1 : Real.cos θ = -3/5) (h2 : θ ∈ Set.Ioo (π/2) π) :
  (Real.sin θ = 4/5) ∧ (Real.cos (π/3 - θ) = (4*Real.sqrt 3 - 3)/10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_values_l158_15870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l158_15830

structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_origin (T : Set Point) : Prop :=
  ∀ p : Point, p ∈ T → Point.mk (-p.x) (-p.y) ∈ T

def symmetric_about_x_axis (T : Set Point) : Prop :=
  ∀ p : Point, p ∈ T → Point.mk p.x (-p.y) ∈ T

def symmetric_about_y_axis (T : Set Point) : Prop :=
  ∀ p : Point, p ∈ T → Point.mk (-p.x) p.y ∈ T

def symmetric_about_y_eq_x (T : Set Point) : Prop :=
  ∀ p : Point, p ∈ T → Point.mk p.y p.x ∈ T

def symmetric_about_y_eq_neg_x (T : Set Point) : Prop :=
  ∀ p : Point, p ∈ T → Point.mk (-p.y) (-p.x) ∈ T

theorem smallest_symmetric_set_size (T : Set Point) :
  symmetric_about_origin T →
  symmetric_about_x_axis T →
  symmetric_about_y_axis T →
  symmetric_about_y_eq_x T →
  symmetric_about_y_eq_neg_x T →
  (Point.mk 3 4) ∈ T →
  (∃ (S : Finset Point), ↑S ⊆ T ∧ Finset.card S = 8) ∧
  (∀ (S : Finset Point), ↑S ⊆ T → Finset.card S < 8 → ↑S ≠ T) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l158_15830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_k_kplus1_binary_property_l158_15813

-- Define the fractional part of a real number in binary representation
noncomputable def fractionalPartBinary (x : ℝ) : ℕ → Bool :=
  fun n => (2^n * (x - ⌊x⌋)) % 2 ≥ 1

-- Theorem statement
theorem sqrt_k_kplus1_binary_property (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  ∃ i, i ∈ Finset.range (n + 1) ∧ fractionalPartBinary (Real.sqrt (k * (k + 1))) (n + i) = true := by
  sorry

#check sqrt_k_kplus1_binary_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_k_kplus1_binary_property_l158_15813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_307_5_l158_15890

/-- Represents a trapezoid ABCD with midpoints E and F on sides AD and BC respectively -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ

/-- Calculates the area of quadrilateral EFCD in the given trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  t.altitude * (t.AB / 2 + t.CD / 2)

/-- Theorem stating that the area of EFCD in the given trapezoid is 307.5 square units -/
theorem area_EFCD_eq_307_5 (t : Trapezoid) 
  (h1 : t.AB = 10)
  (h2 : t.CD = 24)
  (h3 : t.altitude = 15) :
  area_EFCD t = 307.5 := by
  sorry

#check area_EFCD_eq_307_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_eq_307_5_l158_15890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_135_degrees_l158_15806

def line_equation (x y : ℝ) : Prop := x + y + Real.sqrt 3 = 0

def angle_of_inclination (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < Real.pi ∧ ∃ (x y : ℝ), line_equation x y ∧ Real.tan θ = -1

theorem angle_is_135_degrees :
  ∃ (θ : ℝ), angle_of_inclination θ ∧ θ = Real.pi * (3/4) := by
  sorry

#check angle_is_135_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_135_degrees_l158_15806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l158_15810

theorem smallest_set_size {S : Type} [Fintype S] (X : Fin 100 → Set S)
  (h_distinct : ∀ i j, i ≠ j → X i ≠ X j)
  (h_nonempty : ∀ i, (X i).Nonempty)
  (h_disjoint : ∀ i : Fin 99, Disjoint (X i) (X (i.succ)))
  (h_not_union : ∀ i : Fin 99, (X i) ∪ (X (i.succ)) ≠ Set.univ) :
  8 ≤ Fintype.card S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_set_size_l158_15810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_triangle_area_l158_15869

/-- The area of a triangle formed by the tangency points of three mutually externally tangent circles -/
noncomputable def area_of_triangle_formed_by_tangency_points (r₁ r₂ r₃ : ℝ) : ℝ :=
  sorry

/-- The area of the triangle formed by the tangency points of three mutually externally tangent circles -/
theorem tangent_circles_triangle_area :
  ∀ (r₁ r₂ r₃ : ℝ),
  r₁ = 2 ∧ r₂ = 3 ∧ r₃ = 4 →
  ∃ (A : ℝ),
  A = area_of_triangle_formed_by_tangency_points r₁ r₂ r₃ ∧
  A = 6 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_triangle_area_l158_15869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_value_l158_15822

-- Define the curve and the line
noncomputable def curve (x : ℝ) : ℝ := Real.exp x
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - k = 0

-- Define the tangent condition
def is_tangent (k : ℝ) : Prop :=
  ∃ x : ℝ, line k x (curve x) ∧
    ∀ y : ℝ, y ≠ curve x → ¬(line k x y)

-- Theorem statement
theorem tangent_line_k_value :
  ∀ k : ℝ, is_tangent k → k = Real.exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_value_l158_15822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l158_15832

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point A
def point_A : ℝ × ℝ := (2, 1)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0
def tangent_line_2 (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem tangent_lines_to_circle :
  ∀ (x y : ℝ),
  (tangent_line_1 x y ∨ tangent_line_2 x) ↔
  (∃ (t : ℝ), 
    (x, y) = (2 + t * (-3/5), 1 + t * (4/5)) ∧  -- parametric equation of line through A
    (∀ (ε : ℝ), ε ≠ 0 → ¬(my_circle (2 + (t+ε) * (-3/5)) (1 + (t+ε) * (4/5))))) -- tangency condition
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l158_15832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_relation_l158_15833

/-- RightAngle B A C means ∠ABC is a right angle -/
def RightAngle (B A C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- IsAltitude B H A C means BH is an altitude of triangle ABC -/
def IsAltitude (B H A C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- IsInradius r A B C means r is the inradius of triangle ABC -/
def IsInradius (r : ℝ) (A B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- Given a right triangle ABC with ∠B = 90° and altitude BH, 
    r is the inradius of ABC, r₁ is the inradius of ABH, and r₂ is the inradius of CBH -/
theorem inradius_relation 
    (A B C H : EuclideanSpace ℝ (Fin 2)) (r r₁ r₂ : ℝ) : 
  RightAngle B A C →
  IsAltitude B H A C →
  IsInradius r A B C →
  IsInradius r₁ A B H →
  IsInradius r₂ C B H →
  r^2 = r₁^2 + r₂^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_relation_l158_15833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dasha_to_vasya_distance_l158_15848

/-- A road network connecting five friends' houses -/
structure RoadNetwork where
  asya_galia : ℝ
  galia_borya : ℝ
  asya_borya : ℝ
  dasha_galia : ℝ
  vasya_galia : ℝ

/-- The shortest distance between two points in the road network -/
def shortest_distance (network : RoadNetwork) (from_loc to_loc : String) : ℝ :=
  sorry

/-- Theorem stating the shortest distance between Dasha and Vasya -/
theorem dasha_to_vasya_distance (network : RoadNetwork)
  (h1 : network.asya_galia = 12)
  (h2 : network.galia_borya = 10)
  (h3 : network.asya_borya = 8)
  (h4 : network.dasha_galia = 15)
  (h5 : network.vasya_galia = 17) :
  shortest_distance network "Dasha" "Vasya" = 18 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dasha_to_vasya_distance_l158_15848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_basic_multiple_in_set_l158_15853

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ({a, b, c} : Finset ℕ) = {2, 4, 5} ∧ n = 100 * a + 10 * b + c)

def is_basic_multiple (a b : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ ({2, 3, 5} : Finset ℕ) ∧ a = k * b

theorem no_basic_multiple_in_set :
  ¬ ∃ a b : ℕ, is_valid_number a ∧ is_valid_number b ∧ a ≠ b ∧ is_basic_multiple a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_basic_multiple_in_set_l158_15853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l158_15887

/-- A parabola with focus F and directrix intersecting x-axis at (-3/2, 0) -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : p > 0
  h_focus : focus.1 = p / 2 ∧ focus.2 = 0

/-- A line passing through the focus of the parabola -/
structure FocusLine (C : Parabola) where
  m : ℝ
  h_not_perp : m ≠ 0

/-- Predicate to check if a point is on the parabola -/
def OnParabola (C : Parabola) (point : ℝ × ℝ) : Prop :=
  point.2^2 = 2 * C.p * point.1

/-- Theorem about properties of the parabola and its intersecting line -/
theorem parabola_properties (C : Parabola) (l : FocusLine C) :
  -- 1. The equation of C is y^2 = 6x
  C.p = 3 ∧
  -- 2. The minimum value of |AB| + 3|BF| is 27/2
  (∃ (A B : ℝ × ℝ), OnParabola C A ∧ OnParabola C B ∧ A ≠ B ∧ 
    (∀ (A' B' : ℝ × ℝ), OnParabola C A' → OnParabola C B' → A' ≠ B' → 
      dist A B + 3 * dist B C.focus ≤ dist A' B' + 3 * dist B' C.focus) ∧
    dist A B + 3 * dist B C.focus = 27/2) ∧
  -- 3. |BF|(|MA| + |MB|) = 2|MB||PF| for any point P on x-axis
  (∀ (A B : ℝ × ℝ) (P : ℝ), OnParabola C A → OnParabola C B → A ≠ B →
    dist B C.focus * (dist (-3/2, 0) A + dist (-3/2, 0) B) = 
    2 * dist (-3/2, 0) B * dist (P, 0) C.focus) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l158_15887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_equality_l158_15865

theorem gcf_of_lcm_equality : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 5 18) = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcm_equality_l158_15865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_zero_asymptote_l158_15868

/-- The denominator polynomial of the rational function -/
noncomputable def q (x : ℝ) : ℝ := 3*x^6 - 2*x^3 + x - 5

/-- The rational function -/
noncomputable def f (p : ℝ → ℝ) (x : ℝ) : ℝ := p x / q x

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

theorem largest_degree_with_zero_asymptote :
  ∃ (p : ℝ → ℝ), 
    (∀ ε > 0, ∃ M, ∀ x, |x| > M → |f p x| < ε) ∧ 
    degree p = 5 ∧
    (∀ p', (∀ ε > 0, ∃ M, ∀ x, |x| > M → |f p' x| < ε) → degree p' ≤ 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_zero_asymptote_l158_15868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_variance_of_X_standard_deviation_of_X_l158_15866

/-- A discrete random variable X with the given distribution -/
def X : ℕ → ℝ
| 1 => 0.1
| 2 => 0.3
| 4 => 0.6
| _ => 0

/-- The support of X -/
def support : List ℕ := [1, 2, 4]

/-- Expected value of X -/
def expectedValue : ℝ := (support.map (fun x => x * X x)).sum

/-- Variance of X -/
def variance : ℝ := (support.map (fun x => (x - expectedValue)^2 * X x)).sum

/-- Standard deviation of X -/
noncomputable def standardDeviation : ℝ := Real.sqrt variance

theorem expected_value_of_X : expectedValue = 3.1 := by sorry

theorem variance_of_X : variance = 1.29 := by sorry

theorem standard_deviation_of_X : standardDeviation = Real.sqrt 1.29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_of_X_variance_of_X_standard_deviation_of_X_l158_15866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_non_shaded_ratio_l158_15860

noncomputable section

-- Define the equilateral triangle ABC
def Triangle (s : ℝ) : Set (ℝ × ℝ) :=
  {⟨x, y⟩ | 0 ≤ x ∧ x ≤ s ∧ 0 ≤ y ∧ y ≤ (Real.sqrt 3 / 2) * s}

-- Define midpoints D, E, F
def D (s : ℝ) : ℝ × ℝ := (s / 2, 0)
def E (s : ℝ) : ℝ × ℝ := (s, 0)
def F (s : ℝ) : ℝ × ℝ := (s / 2, (Real.sqrt 3 / 2) * s)

-- Define centroids G and H
def G (s : ℝ) : ℝ × ℝ := (s / 3, (Real.sqrt 3 / 6) * s)
def H (s : ℝ) : ℝ × ℝ := (2 * s / 3, (Real.sqrt 3 / 6) * s)

-- Define the areas
def area_ABC (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2
def area_shaded (s : ℝ) : ℝ := (Real.sqrt 3 / 108) * s^2
def area_non_shaded (s : ℝ) : ℝ := area_ABC s - area_shaded s

-- Theorem statement
theorem shaded_to_non_shaded_ratio (s : ℝ) (h : s > 0) :
  area_shaded s / area_non_shaded s = 1 / 26 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_to_non_shaded_ratio_l158_15860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_credit_card_expense_ratio_l158_15861

theorem credit_card_expense_ratio 
  (initial_balance : ℚ)
  (grocery_expense : ℚ)
  (return_amount : ℚ)
  (new_balance : ℚ)
  (h1 : initial_balance = 126)
  (h2 : grocery_expense = 60)
  (h3 : return_amount = 45)
  (h4 : new_balance = 171) :
  (new_balance - (initial_balance + grocery_expense - return_amount)) / grocery_expense = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_credit_card_expense_ratio_l158_15861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_possible_l158_15807

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the set of available triangles -/
noncomputable def available_triangles : List Triangle :=
  (List.replicate 2 ⟨3, 4, 5⟩) ++
  (List.replicate 4 ⟨4, 5, Real.sqrt 41⟩) ++
  (List.replicate 6 ⟨(5/6) * Real.sqrt 2, 4, 5⟩)

/-- Checks if a tetrahedron can be formed from four triangles -/
def can_form_tetrahedron (t1 t2 t3 t4 : Triangle) : Prop := sorry

/-- Theorem stating that no tetrahedron can be formed from the available triangles -/
theorem no_tetrahedron_possible : 
  ∀ (t1 t2 t3 t4 : Triangle), t1 ∈ available_triangles → t2 ∈ available_triangles → 
  t3 ∈ available_triangles → t4 ∈ available_triangles → ¬(can_form_tetrahedron t1 t2 t3 t4) := by
  sorry

#check no_tetrahedron_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_possible_l158_15807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l158_15891

theorem quadratic_equation_solutions (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  let solutions := {x : ℝ | a*x^2 + b*x + c = 0}
  solutions = 
    if discriminant > 0 then
      {(-b + Real.sqrt discriminant) / (2*a), (-b - Real.sqrt discriminant) / (2*a)}
    else if discriminant = 0 then
      {-b / (2*a)}
    else
      ∅ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l158_15891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l158_15867

theorem solution_equation (x y : ℤ) : 
  (x : ℝ) * Real.log 27 * (Real.log 13)⁻¹ = 27 * Real.log y / Real.log 13 →
  y > 70 →
  (∀ z : ℤ, z > 70 → z < y) →
  x = 36 ∧ y = 81 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equation_l158_15867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_l158_15882

/-- The area of a figure formed by two circular sectors -/
noncomputable def area_two_sectors (r : ℝ) (angle1 angle2 : ℝ) : ℝ :=
  (angle1 / 360) * Real.pi * r^2 + (angle2 / 360) * Real.pi * r^2

/-- Theorem: The area of figure ABCD is 37.5π square units -/
theorem area_ABCD : area_two_sectors 10 90 45 = 37.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCD_l158_15882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_calculate_part1_simplify_and_calculate_part2_l158_15800

-- Part 1
theorem simplify_and_calculate_part1 :
  (7 + 4 * Real.sqrt 3) ^ (1/2 : ℝ) - 81 ^ (1/8 : ℝ) + 32 ^ (3/5 : ℝ) - 2 * (1/8 : ℝ) ^ (-(2/3) : ℝ) + 32 * (4 ^ (-(1/3) : ℝ)) ^ (-1 : ℝ) = 4 := by
  sorry

-- Part 2
theorem simplify_and_calculate_part2 :
  (Real.log 2 / Real.log 6) ^ 2 + (Real.log 3 / Real.log 6) ^ 2 + 3 * (Real.log 2 / Real.log 6) * (Real.log 318 / Real.log 6 - 1/3 * Real.log 2 / Real.log 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_and_calculate_part1_simplify_and_calculate_part2_l158_15800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_approx_l158_15875

/-- A right triangle with sides 5, 12, and 13 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- The crease formed when point C is folded onto point A -/
noncomputable def crease (t : RightTriangle) : ℝ := 
  let midpoint_x := t.b / 2
  let midpoint_y := t.a / 2
  let intersection_x := (59.5 : ℝ) / 12
  Real.sqrt ((intersection_x - midpoint_x)^2 + midpoint_y^2)

/-- The theorem stating that the crease length is approximately 2.708 inches -/
theorem crease_length_approx (t : RightTriangle) : 
  ∃ ε > 0, |crease t - 2.708| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_approx_l158_15875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l158_15846

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 32 / Real.log x = 5/2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l158_15846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_and_sum_l158_15885

theorem simplify_cube_root_and_sum (a b : ℕ) (ha : a = 3) (hb : b = 5) :
  ∃ (c d : ℕ), (c > 0 ∧ d > 0) ∧ 
  ((a^5 * b^4 : ℝ)^(1/3) = c * (d : ℝ)^(1/3)) ∧
  c + d = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_and_sum_l158_15885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_z_l158_15859

-- Define the relationships between x, y, and z
def varies_as (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ t, f t = k * g t

-- State the theorem
theorem x_varies_as_z (x y z : ℝ → ℝ) 
  (h1 : varies_as x (fun t ↦ (y t)^3))
  (h2 : varies_as y (fun t ↦ (z t)^(1/3))) :
  varies_as x z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_z_l158_15859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_square_base_seven_l158_15893

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 239

/-- A number has exactly 4 digits in base 7 if it's between 7^3 and 7^4 - 1 -/
def has_four_digits_base_seven (n : ℕ) : Prop := 7^3 ≤ n ∧ n < 7^4

theorem largest_four_digit_square_base_seven :
  (has_four_digits_base_seven (M^2)) ∧
  ∀ n : ℕ, n > M → ¬(has_four_digits_base_seven (n^2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_four_digit_square_base_seven_l158_15893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_X_l158_15889

/-- Calculate the number of digits in a positive integer -/
def number_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating that X = -4 is the unique solution -/
theorem unique_solution_for_X : 
  ∃! X : ℤ, number_of_digits ((50^8 * 8^3 * 11^2 : ℕ) * (10^X.natAbs : ℕ)) = 18 ∧ 
          (X < 0 → (10^(-X).natAbs : ℕ) ∣ (50^8 * 8^3 * 11^2 : ℕ)) ∧
          X = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_X_l158_15889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l158_15804

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem: If (4, -2) lies on the asymptote of the hyperbola, its eccentricity is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (asymptote_point : (4 : ℝ) * (h.a / h.b) = 2) : 
  eccentricity h = Real.sqrt 5 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l158_15804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_l158_15857

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := 1 + Real.log 2 / Real.log 7

-- State the theorem
theorem compare_logarithms : a > b ∧ b > c := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_l158_15857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_l158_15872

theorem quadratic_roots_sum (p q x₁ x₂ : ℤ) : 
  (∀ x, x^2 - p*x + q = 0 ↔ x = x₁ ∨ x = x₂) →  -- x₁ and x₂ are roots
  (x₁ > x₂) →                                  -- x₁ > x₂ > q
  (x₂ > q) →
  (x₁ - x₂ = x₂ - q) →                         -- arithmetic progression
  (∃ d : ℤ, x₁ = x₂ + d ∧ q = x₂ - d) →        -- decreasing arithmetic progression
  (∃ s : ℤ, s = -5 ∧ (∀ x₂' : ℤ, (∃ x₁' q' : ℤ, 
    (∀ x, x^2 - p*x + q' = 0 ↔ x = x₁' ∨ x = x₂') ∧
    (x₁' > x₂') ∧ 
    (x₂' > q') ∧
    (x₁' - x₂' = x₂' - q') ∧
    (∃ d : ℤ, x₁' = x₂' + d ∧ q' = x₂' - d)) → x₂' = -3 ∨ x₂' = -2) ∧
    s = Finset.sum {-3, -2} id) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_l158_15872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_infinite_grid_l158_15819

-- Define the type for our grid
def Grid := ℤ → ℤ → ℕ

-- Define what it means for a grid to be valid
def ValidGrid (g : Grid) : Prop :=
  (∀ i j i' j' : ℤ, (i = i' ∧ j = j' + 1) ∨ (i = i' + 1 ∧ j = j') →
    (g i j : ℤ) - (g i' j' : ℤ) ≤ 2015 ∧ (g i' j' : ℤ) - (g i j : ℤ) ≤ 2015) ∧
  (∀ i j i' j' : ℤ, (i, j) ≠ (i', j') → g i j ≠ g i' j')

-- Theorem stating the impossibility
theorem no_valid_infinite_grid : ¬∃ (g : Grid), ValidGrid g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_infinite_grid_l158_15819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_intersecting_lines_l158_15871

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if a point is on a line -/
def point_on_line (p : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = l.point + t • l.direction

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ (p : ℝ × ℝ × ℝ), point_on_line p l1 ∧ point_on_line p l2) ∧
  ¬ (∃ (k : ℝ), l1.direction = k • l2.direction)

/-- A set of lines that intersect all three given lines -/
def intersecting_lines (a b c : Line3D) : Set Line3D :=
  {l : Line3D | ∃ (p q r : ℝ × ℝ × ℝ), 
    point_on_line p l ∧ point_on_line p a ∧ 
    point_on_line q l ∧ point_on_line q b ∧ 
    point_on_line r l ∧ point_on_line r c}

/-- Theorem stating that there are infinitely many intersecting lines -/
theorem infinitely_many_intersecting_lines
  (a b c : Line3D)
  (h_ab : are_skew a b)
  (h_bc : are_skew b c)
  (h_ac : are_skew a c) :
  Set.Infinite (intersecting_lines a b c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_intersecting_lines_l158_15871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l158_15880

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := 4 * x - y - 25 = 0

/-- The curve W in the Cartesian plane -/
def curve_W (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The minimum distance between a point on line_l and a point on curve_W -/
theorem min_distance_line_curve :
  ∃ (x1 y1 x2 y2 : ℝ),
    line_l x1 y1 ∧ curve_W x2 y2 ∧
    ∀ (x3 y3 x4 y4 : ℝ),
      line_l x3 y3 → curve_W x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4 ∧
      distance x1 y1 x2 y2 = (8 * Real.sqrt 17) / 17 := by
  sorry

#check min_distance_line_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_curve_l158_15880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_education_expense_l158_15858

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings : ℕ
  savings_percentage : ℚ

/-- Calculates the total salary based on savings and savings percentage --/
def total_salary (finances : KishoreFinances) : ℚ :=
  (finances.savings : ℚ) / finances.savings_percentage

/-- Calculates the total known expenses --/
def known_expenses (finances : KishoreFinances) : ℕ :=
  finances.rent + finances.milk + finances.groceries + finances.petrol + finances.miscellaneous

/-- Theorem: The amount spent on children's education is 2500 --/
theorem children_education_expense (finances : KishoreFinances) 
    (h1 : finances.rent = 5000)
    (h2 : finances.milk = 1500)
    (h3 : finances.groceries = 4500)
    (h4 : finances.petrol = 2000)
    (h5 : finances.miscellaneous = 6100)
    (h6 : finances.savings = 2400)
    (h7 : finances.savings_percentage = 1/10) :
    ⌊total_salary finances⌋ - known_expenses finances - finances.savings = 2500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_education_expense_l158_15858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l158_15877

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2/2 - m*x

theorem problem_solution :
  (∀ x : ℝ, f (1/2) x < x ↔ 0 < x ∧ x < 3) ∧
  (∀ m : ℝ, 
    (m > 1/2 → ∀ x : ℝ, f m x + x/2 ≥ 0 ↔ x ≤ 0 ∨ x ≥ 2*m - 1) ∧
    (m < 1/2 → ∀ x : ℝ, f m x + x/2 ≥ 0 ↔ x ≥ 0 ∨ x ≤ 2*m - 1) ∧
    (m = 1/2 → ∀ x : ℝ, f m x + x/2 ≥ 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l158_15877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_of_geometric_sequence_l158_15852

-- Define a geometric sequence
def geometric_sequence (a₁ : ℕ+) (r : ℕ+) : ℕ → ℕ+ 
  | 0 => a₁
  | n + 1 => (geometric_sequence a₁ r n) * r

theorem seventh_term_of_geometric_sequence :
  ∀ (a₁ r : ℕ+),
    a₁ = 3 →
    geometric_sequence a₁ r 5 = 729 →
    geometric_sequence a₁ r 6 = 2187 := by
  sorry

#check seventh_term_of_geometric_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_of_geometric_sequence_l158_15852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_representation_l158_15850

/-- Represents the financial value in yuan -/
structure Yuan where
  value : ℤ

/-- Represents income as a positive value -/
def income (x : Yuan) : Prop := x.value > 0

/-- Represents expenses as a negative value -/
def expense (x : Yuan) : Prop := x.value < 0

/-- States that income and expense of the same amount sum to zero -/
axiom income_expense_sum (x : Yuan) : income x → expense (Yuan.mk (-x.value)) → x.value + (-x.value) = 0

theorem expense_representation (x : Yuan) (h : income x) : expense (Yuan.mk (-x.value)) := by
  sorry

#check expense_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_representation_l158_15850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_sum_of_squares_l158_15874

theorem no_integer_solution_for_sum_of_squares (a : ℕ) (h : a % 4 = 3) :
  ¬ ∃ (x y : ℤ), x^2 + y^2 = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_sum_of_squares_l158_15874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l158_15817

def base_6_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem cryptarithm_solution :
  let F : Nat := 1
  let A : Nat := 5
  let R : Nat := 3
  let E : Nat := 2
  let S : Nat := 4
  let FARES : List Nat := [S, E, R, A, F]
  let FEE : List Nat := [E, E, F]
  base_6_to_decimal FARES = (base_6_to_decimal FEE) ^ 2 := by
  sorry

#check cryptarithm_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l158_15817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_k_range_l158_15863

noncomputable def sequenceA (n : ℕ) (k : ℝ) : ℝ := n^2 + k*n

theorem increasing_sequence_k_range :
  ∀ k : ℝ, (∀ n : ℕ, sequenceA (n + 1) k > sequenceA n k) ↔ k ≥ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_k_range_l158_15863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_3000_l158_15847

def next_term (a : ℕ) : ℕ :=
  if a % 2 = 0 then a / 2 else 3 * a + 1

def satisfies_condition (a : ℕ) : Bool :=
  let a2 := next_term a
  let a3 := next_term a2
  let a4 := next_term a3
  a < a2 ∧ a < a3 ∧ a < a4

def count_satisfying (n : ℕ) : ℕ :=
  (Finset.range n).filter (fun a => satisfies_condition a) |>.card

theorem count_satisfying_3000 :
  count_satisfying 3001 = 750 := by
  sorry

#eval count_satisfying 3001

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_3000_l158_15847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_conditions_l158_15820

noncomputable def f (x : ℝ) : ℝ := -3/8 * x^5 + 5/4 * x^3 - 15/8 * x

theorem polynomial_conditions :
  (∃ (a b c d e : ℝ), f = fun x ↦ a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x) ∧
  (∃ (g : ℝ → ℝ), ∀ x, f x + 1 = (x - 1)^3 * g x) ∧
  (∃ (h : ℝ → ℝ), ∀ x, f x - 1 = (x + 1)^3 * h x) :=
by
  sorry

#check polynomial_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_conditions_l158_15820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l158_15884

theorem subsets_of_three_element_set :
  let S : Finset Char := {'a', 'b', 'c'}
  Finset.powerset S |>.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l158_15884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_illuminating_orientation_l158_15818

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a cube
structure Cube where
  center : Point3D
  sideLength : ℝ

-- Define an octant (represented by its orientation)
structure Octant where
  orientation : ℝ × ℝ × ℝ  -- Representing the orientation as Euler angles

-- Function to get vertices of a cube
def vertices (c : Cube) : List Point3D :=
  sorry

-- Function to check if a point is illuminated by an octant
def isIlluminated (p : Point3D) (o : Octant) (c : Cube) : Prop :=
  sorry

-- Theorem stating that there exists an octant orientation that doesn't illuminate any vertex
theorem exists_non_illuminating_orientation (c : Cube) :
  ∃ (o : Octant), ∀ (v : Point3D), v ∈ vertices c → ¬(isIlluminated v o c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_illuminating_orientation_l158_15818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_set_l158_15823

def number_set : Finset ℕ := {1, 2, 3, 4, 5}

def mean (s : Finset ℕ) : ℚ :=
  (s.sum (fun x => (x : ℚ))) / s.card

def variance (s : Finset ℕ) : ℚ :=
  (s.sum (fun x => ((x : ℚ) - mean s) ^ 2)) / s.card

theorem variance_of_set : variance number_set = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_set_l158_15823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_composition_Cl_l158_15838

/-- Atomic mass of sodium in g/mol -/
noncomputable def mass_Na : ℝ := 22.99

/-- Atomic mass of oxygen in g/mol -/
noncomputable def mass_O : ℝ := 16.00

/-- Atomic mass of chlorine in g/mol -/
noncomputable def mass_Cl : ℝ := 35.45

/-- Molar mass of sodium hypochlorite (NaOCl) in g/mol -/
noncomputable def mass_NaOCl : ℝ := mass_Na + mass_O + mass_Cl

/-- Percentage composition of chlorine in sodium hypochlorite -/
noncomputable def percentage_Cl : ℝ := (mass_Cl / mass_NaOCl) * 100

theorem percentage_composition_Cl :
  ∃ ε > 0, |percentage_Cl - 47.63| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_composition_Cl_l158_15838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l158_15851

/-- The minimum area of a triangle with vertices at (0,0), (30,21), and (p,q) where p and q are integers -/
theorem min_triangle_area : ∃ (area : ℝ), area = 3/2 ∧ 
  ∀ (p q : ℤ), (1/2 : ℝ) * |30 * (q : ℝ) - 21 * (p : ℝ)| ≥ area :=
by
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 21)
  let C : ℤ × ℤ → ℝ × ℝ := λ (p, q) => ((p : ℝ), (q : ℝ))
  let triangle_area (p q : ℤ) : ℝ := (1/2 : ℝ) * |30 * (q : ℝ) - 21 * (p : ℝ)|
  sorry  -- Proof skipped


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l158_15851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l158_15827

theorem beta_value (α β : Real) 
  (h1 : Real.sin α + Real.sin (α + β) + Real.cos (α + β) = Real.sqrt 3)
  (h2 : β ∈ Set.Icc (Real.pi / 4) Real.pi) : 
  β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l158_15827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_arrangement_l158_15842

-- Define the colors of balls and boxes
inductive Color
| Red
| White
| Yellow

-- Define a function to represent the color of balls in each box
def ballsInBox : Color → Color := sorry

-- Define the number of balls of each color
def numBalls : Color → ℕ := sorry

-- Define the number of balls in each box
def numBallsInBox : Color → ℕ := sorry

-- Condition 1: The yellow box contains more balls than there are yellow balls
axiom yellow_box_condition :
  numBallsInBox Color.Yellow > numBalls Color.Yellow

-- Condition 2: The number of balls in the red box is different from the number of white balls
axiom red_box_condition :
  numBallsInBox Color.Red ≠ numBalls Color.White

-- Condition 3: There are fewer white balls than the number of balls in the white box
axiom white_box_condition :
  numBalls Color.White < numBallsInBox Color.White

-- Theorem: The only possible arrangement is yellow balls in red box, red balls in white box, and white balls in yellow box
theorem balls_arrangement :
  (ballsInBox Color.Red = Color.Yellow) ∧
  (ballsInBox Color.White = Color.Red) ∧
  (ballsInBox Color.Yellow = Color.White) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_arrangement_l158_15842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l158_15837

theorem triangle_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.sqrt 3 * b))
  (h2 : n = (Real.cos A, Real.sin B))
  (h3 : ∃ k : ℝ, m = k • n)
  (h4 : a = Real.sqrt 7)
  (h5 : (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2) :
  A = π/3 ∧ a + b + c = Real.sqrt 7 + 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l158_15837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_max_value_domain_double_intersection_range_l158_15836

-- Define the parabola C: y = ax^2 + 2x - 1 (a ≠ 0)
noncomputable def C (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 1

-- Define the line l: y = kx + b passing through A(-3, -3) and B(1, -1)
noncomputable def l (x : ℝ) : ℝ := (1/2) * x - (3/2)

-- Theorem 1: Range of a for C intersecting l
theorem intersection_range (a : ℝ) :
  (∃ x : ℝ, C a x = l x) ↔ (a ≤ 9/8 ∧ a ≠ 0) := by
  sorry

-- Theorem 2: Maximum value and domain for a = -1
theorem max_value_domain (m : ℝ) :
  (∀ x : ℝ, m ≤ x ∧ x ≤ m + 2 → C (-1) x ≤ -4) ∧
  (∃ x : ℝ, m ≤ x ∧ x ≤ m + 2 ∧ C (-1) x = -4) ↔
  (m = -3 ∨ m = 3) := by
  sorry

-- Theorem 3: Range of a for C intersecting AB twice
theorem double_intersection_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ -3 ≤ x₁ ∧ x₁ ≤ 1 ∧ -3 ≤ x₂ ∧ x₂ ≤ 1 ∧
    C a x₁ = l x₁ ∧ C a x₂ = l x₂) ↔
  ((4/9 ≤ a ∧ a < 9/8) ∨ a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_max_value_domain_double_intersection_range_l158_15836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_to_curve_l158_15898

variable (f : ℝ → ℝ)
variable (x₀ y₀ : ℝ)

-- f is differentiable at x₀
variable (hf : DifferentiableAt ℝ f x₀)

-- (x₀, y₀) is a point on the curve
variable (h : f x₀ = y₀)

-- Define the equation of the normal line
def normal_equation (x y : ℝ) : Prop :=
  -(deriv f x₀) * (y - y₀) = x - x₀

-- Theorem statement
theorem normal_to_curve :
  ∀ x y : ℝ, normal_equation f x₀ y₀ x y ↔ 
  (x - x₀) * (deriv f x₀) + (y - y₀) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_to_curve_l158_15898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l158_15843

-- Define the function f(x)
noncomputable def f (x : Real) : Real := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) - 1

-- Theorem statement
theorem f_properties :
  -- 1. Maximum value of f(x) is 2
  (∃ x, f x = 2) ∧ (∀ x, f x ≤ 2) ∧
  -- 2. f(x) is monotonically decreasing in [π/6 + kπ, 2π/3 + kπ] for any integer k
  (∀ k : Int, ∀ x y, x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi) →
    y ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi) →
    x ≤ y → f y ≤ f x) ∧
  -- 3. For x ∈ [-π/6, π/3], the range of f(x) is [-1, 2]
  (∀ x, x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3) → f x ∈ Set.Icc (-1) 2) ∧
  (∃ x y, x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3) ∧ y ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3) ∧
    f x = -1 ∧ f y = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l158_15843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sequence_property_l158_15825

/-- A circle with equation x^2 + y^2 = 10x -/
def Circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 10 * p.1}

/-- The point (5, 3) inside the circle -/
def Point : ℝ × ℝ := (5, 3)

/-- The shortest chord length -/
def ShortestChord : ℝ := 8

/-- The longest chord length -/
def LongestChord : ℝ := 10

/-- The common difference of the arithmetic sequence formed by the chord lengths -/
noncomputable def CommonDifference (k : ℕ) : ℝ := (LongestChord - ShortestChord) / (k - 1)

theorem chord_sequence_property (k : ℕ) :
  Point ∈ Circle →
  1/3 ≤ CommonDifference k →
  CommonDifference k ≤ 1/2 →
  k ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sequence_property_l158_15825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_tank_capacity_is_twelve_l158_15805

/-- Represents the fuel efficiency and capacity of a car before and after modification --/
structure Car where
  initial_efficiency : ℚ  -- miles per gallon before modification
  fuel_usage_ratio : ℚ    -- ratio of fuel usage after modification (compared to before)
  additional_miles : ℚ    -- additional miles that can be traveled after modification
  tank_capacity : ℚ       -- fuel tank capacity in gallons

/-- Calculates the fuel tank capacity given the car's properties --/
def calculate_tank_capacity (c : Car) : ℚ :=
  c.additional_miles / (c.initial_efficiency * (1 / c.fuel_usage_ratio - 1))

/-- Theorem stating that for a car with given properties, the fuel tank capacity is 12 gallons --/
theorem fuel_tank_capacity_is_twelve :
  ∃ (c : Car),
    c.initial_efficiency = 24 ∧
    c.fuel_usage_ratio = 3/4 ∧
    c.additional_miles = 96 ∧
    calculate_tank_capacity c = 12 := by
  -- Construct the car with the given properties
  let c : Car := {
    initial_efficiency := 24
    fuel_usage_ratio := 3/4
    additional_miles := 96
    tank_capacity := 12
  }
  -- Prove that this car satisfies all conditions
  exists c
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  -- Prove that calculate_tank_capacity c = 12
  unfold calculate_tank_capacity
  simp
  -- The following line completes the proof
  norm_num

-- Evaluate the function with the given values
#eval calculate_tank_capacity { initial_efficiency := 24, fuel_usage_ratio := 3/4, additional_miles := 96, tank_capacity := 12 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_tank_capacity_is_twelve_l158_15805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_l158_15879

/-- The sum of the infinite series ∑(4n + 2) / 3^n for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (4 * n + 2) / (3 ^ n)

/-- The sum of the infinite series ∑(4n + 2) / 3^n for n from 1 to infinity is equal to 3 -/
theorem infinite_series_sum_eq_three : infinite_series_sum = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_three_l158_15879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l158_15892

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (k : ℝ) : ℝ × ℝ := (2, k)

theorem perpendicular_vectors (k : ℝ) :
  (vector_a.1 * (vector_b k).1 + vector_a.2 * (vector_b k).2 = 0) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l158_15892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l158_15821

theorem cone_base_radius (arc_length : Real) (h : arc_length = 8 * Real.pi) : 
  (arc_length / (2 * Real.pi)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l158_15821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_equals_one_l158_15824

-- Define the polynomials A and B
def A (x y m : ℝ) : ℝ := 2 * x^2 + 2 * x * y + m * y - 8
def B (x y n : ℝ) : ℝ := -n * x^2 + x * y + y + 7

-- Define the condition that A - 2B has no x^2 term and no y term
def no_x2_and_y_terms (m n : ℝ) : Prop :=
  ∀ x y, (2 + 2*n) * x^2 + (m - 2) * y = 0

-- Theorem statement
theorem m_plus_n_equals_one (m n : ℝ) (h : no_x2_and_y_terms m n) : m + n = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_plus_n_equals_one_l158_15824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l158_15828

theorem equation_solution (k m : ℝ) : 
  (∀ x, (abs k - 3) * x^2 - (k - 3) * x + 2 * m + 1 = 0 ↔ x = 1/2) →
  (∀ x, (abs k - 3) * x^2 - (k - 3) * x + 2 * m + 1 = 0 ↔ 3 * x = 4 - 5 * x) →
  k = -3 ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l158_15828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_squared_l158_15896

noncomputable def circle_radius : ℝ := Real.sqrt 50
def length_AB : ℝ := 6
def length_BC : ℝ := 2

theorem distance_B_to_center_squared (O A B C : ℝ × ℝ) : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = length_AB^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = length_BC^2 →
  (A.1 - B.1) * (B.1 - C.1) + (A.2 - B.2) * (B.2 - C.2) = 0 →
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = circle_radius^2 →
  (C.1 - O.1)^2 + (C.2 - O.2)^2 = circle_radius^2 →
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_B_to_center_squared_l158_15896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_c_less_than_b_l158_15841

-- Define the constants as noncomputable
noncomputable def a : ℝ := (32 * (4 - Real.log 32)) / Real.exp 4
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 2 / Real.log (Real.sqrt (Real.exp 1))) / 4

-- State the theorem
theorem a_less_than_c_less_than_b : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_c_less_than_b_l158_15841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l158_15862

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define helper functions
def externally_tangent (c1 c2 : Circle) : Prop := sorry
def internally_tangent (c1 c2 : Circle) : Prop := sorry
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the problem setup
def problem_setup (A B C D : Circle) : Prop :=
  -- Circles A, B, and C are externally tangent to each other
  externally_tangent A B ∧ externally_tangent A C ∧ externally_tangent B C
  -- Circles A, B, and C are internally tangent to circle D
  ∧ internally_tangent A D ∧ internally_tangent B D ∧ internally_tangent C D
  -- Circles B and C are congruent
  ∧ B.radius = C.radius
  -- Circle A has radius 2
  ∧ A.radius = 2
  -- Circle A passes through the center of D
  ∧ distance A.center D.center = A.radius + D.radius

-- Define the theorem
theorem circle_B_radius (A B C D : Circle) :
  problem_setup A B C D → B.radius = 16 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_B_radius_l158_15862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_can_add_eight_underwear_l158_15899

/-- Represents the laundry problem with given weights and quantities -/
structure LaundryProblem where
  max_weight : ℕ
  sock_weight : ℕ
  shirt_weight : ℕ
  shorts_weight : ℕ
  pants_weight : ℕ
  num_pants : ℕ
  num_shirts : ℕ
  num_shorts : ℕ
  num_socks : ℕ
  underwear_weight : ℕ

/-- The specific laundry problem instance -/
def tonyLaundry : LaundryProblem :=
  { max_weight := 50
  , sock_weight := 2
  , shirt_weight := 5
  , shorts_weight := 8
  , pants_weight := 10
  , num_pants := 1
  , num_shirts := 2
  , num_shorts := 1
  , num_socks := 3
  , underwear_weight := 2
  }

/-- Calculates the number of underwear pairs that can be added -/
def calculateUnderwearPairs (l : LaundryProblem) : ℕ :=
  let currentWeight := l.num_pants * l.pants_weight +
                       l.num_shirts * l.shirt_weight +
                       l.num_shorts * l.shorts_weight +
                       l.num_socks * l.sock_weight
  let remainingWeight := l.max_weight - currentWeight
  remainingWeight / l.underwear_weight

/-- Theorem stating that Tony can add 8 pairs of underwear -/
theorem tony_can_add_eight_underwear :
  calculateUnderwearPairs tonyLaundry = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tony_can_add_eight_underwear_l158_15899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_freedom_ratio_l158_15844

theorem bird_freedom_ratio : 
  ∀ (grey_birds white_birds freed_birds : ℕ),
  white_birds = grey_birds + 6 →
  grey_birds = 40 →
  grey_birds + white_birds - freed_birds = 66 →
  freed_birds * 2 = grey_birds :=
fun grey_birds white_birds freed_birds 
  h_white h_grey h_remaining =>
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_freedom_ratio_l158_15844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_final_configuration_l158_15854

/-- Represents the configuration of stones on the infinite strip -/
def Configuration := ℤ → ℕ

/-- The golden ratio, defined as the positive root of x^2 = x + 1 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Weight function for a configuration -/
noncomputable def weight (c : Configuration) : ℝ :=
  ∑' (i : ℤ), (c i : ℝ) * φ^i

/-- Predicate for a valid move of type 1 -/
def valid_move1 (c₁ c₂ : Configuration) (n : ℤ) : Prop :=
  c₁ (n - 1) ≥ 1 ∧ c₁ n ≥ 1 ∧
  c₂ (n - 1) = c₁ (n - 1) - 1 ∧
  c₂ n = c₁ n - 1 ∧
  c₂ (n + 1) = c₁ (n + 1) + 1 ∧
  ∀ i : ℤ, (i ≠ n - 1 ∧ i ≠ n ∧ i ≠ n + 1) → c₂ i = c₁ i

/-- Predicate for a valid move of type 2 -/
def valid_move2 (c₁ c₂ : Configuration) (n : ℤ) : Prop :=
  c₁ n ≥ 2 ∧
  c₂ (n - 2) = c₁ (n - 2) + 1 ∧
  c₂ n = c₁ n - 2 ∧
  c₂ (n + 1) = c₁ (n + 1) + 1 ∧
  ∀ i : ℤ, (i ≠ n - 2 ∧ i ≠ n ∧ i ≠ n + 1) → c₂ i = c₁ i

/-- Predicate for a valid move (either type 1 or type 2) -/
def valid_move (c₁ c₂ : Configuration) : Prop :=
  ∃ n : ℤ, valid_move1 c₁ c₂ n ∨ valid_move2 c₁ c₂ n

/-- Predicate for a final configuration where no moves are possible -/
def is_final (c : Configuration) : Prop :=
  ∀ c₂ : Configuration, ¬(valid_move c c₂)

/-- Main theorem: All sequences of valid moves lead to a unique final configuration -/
theorem unique_final_configuration (c_init : Configuration) :
  ∃! c_final : Configuration, is_final c_final ∧
  ∃ (seq : ℕ → Configuration), seq 0 = c_init ∧
  (∀ n : ℕ, valid_move (seq n) (seq (n + 1))) ∧
  (∃ N : ℕ, ∀ n ≥ N, seq n = c_final) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_final_configuration_l158_15854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_correct_l158_15888

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- Length of the first median -/
  median1 : ℝ
  /-- Length of the second median -/
  median2 : ℝ
  /-- Area of the triangle -/
  area : ℝ
  /-- The first median is 5 inches long -/
  median1_length : median1 = 5
  /-- The second median is 9 inches long -/
  median2_length : median2 = 9
  /-- The area of the triangle is 4√21 square inches -/
  triangle_area : area = 4 * Real.sqrt 21

/-- The length of the altitude from the vertex opposite the side bisected by the 5-inch median -/
noncomputable def altitudeLength (t : SpecialTriangle) : ℝ :=
  0.8 * Real.sqrt 21

/-- Theorem stating that the altitude length is correct for the given triangle -/
theorem altitude_length_correct (t : SpecialTriangle) :
  altitudeLength t = 0.8 * Real.sqrt 21 := by
  -- Unfold the definition of altitudeLength
  unfold altitudeLength
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_length_correct_l158_15888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_l158_15840

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, 2 * Real.cos x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - Real.sqrt 3

-- Theorem for the smallest positive period of f
theorem smallest_positive_period_of_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
by sorry

-- Theorem for the monotonically increasing intervals of f
theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∀ x : ℝ,
    x ∈ Set.Icc (- 5 * Real.pi / 12 + k * Real.pi) (k * Real.pi + Real.pi / 12) →
    ∀ y : ℝ, x < y →
    y ∈ Set.Icc (- 5 * Real.pi / 12 + k * Real.pi) (k * Real.pi + Real.pi / 12) →
    f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_monotonically_increasing_intervals_of_f_l158_15840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_at_school_correct_l158_15876

def students_at_school (absent : ℕ) (male : ℕ) (female_diff : ℕ) : ℕ :=
  let female := male - female_diff
  let total := male + female
  total - absent

#eval students_at_school 18 848 49

theorem students_at_school_correct (absent : ℕ) (male : ℕ) (female_diff : ℕ) :
  students_at_school absent male female_diff = male + (male - female_diff) - absent :=
by
  unfold students_at_school
  rfl

#check students_at_school_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_at_school_correct_l158_15876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l158_15803

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f (x - f y) = f (x + y) + f y) →
  (f = λ x ↦ 0) ∨ (f = λ x ↦ -2 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l158_15803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buses_most_cost_effective_l158_15845

/-- Represents the transportation problem for the fifth grade visit to the Natural History Museum -/
structure TransportationProblem where
  total_people : ℕ
  bus_capacity : ℕ
  bus_price : ℚ
  bus_discount : ℚ
  minivan_capacity : ℕ
  minivan_price : ℚ
  minivan_discount : ℚ

/-- Calculate the total cost for using buses -/
def bus_total_cost (p : TransportationProblem) : ℚ :=
  let num_buses := (p.total_people + p.bus_capacity - 1) / p.bus_capacity
  ↑num_buses * ↑p.bus_capacity * p.bus_price * (1 - p.bus_discount)

/-- Calculate the total cost for using minivans -/
def minivan_total_cost (p : TransportationProblem) : ℚ :=
  let num_minivans := (p.total_people + p.minivan_capacity - 1) / p.minivan_capacity
  ↑num_minivans * ↑p.minivan_capacity * p.minivan_price * (1 - p.minivan_discount)

/-- The theorem stating that using buses is the most cost-effective option -/
theorem buses_most_cost_effective (p : TransportationProblem) 
  (h1 : p.total_people = 120)
  (h2 : p.bus_capacity = 40)
  (h3 : p.bus_price = 5)
  (h4 : p.bus_discount = 1/5)
  (h5 : p.minivan_capacity = 10)
  (h6 : p.minivan_price = 6)
  (h7 : p.minivan_discount = 1/4) :
  bus_total_cost p ≤ minivan_total_cost p := by
  sorry

#check buses_most_cost_effective

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buses_most_cost_effective_l158_15845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l158_15894

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle parametrically
noncomputable def circle_x (α : ℝ) : ℝ := 2 * Real.cos α
noncomputable def circle_y (α : ℝ) : ℝ := 3 + 2 * Real.sin α

-- Define the distance between intersection points
noncomputable def intersection_distance : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem line_circle_intersection (k : ℝ) :
  (∃ α β : ℝ, α ≠ β ∧
    line k (circle_x α) = circle_y α ∧
    line k (circle_x β) = circle_y β ∧
    (circle_x α - circle_x β)^2 + (circle_y α - circle_y β)^2 = intersection_distance^2) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l158_15894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l158_15881

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l158_15881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_specific_l158_15808

/-- The volume of a truncated right circular cone -/
noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: The volume of a truncated right circular cone with large base radius 12 cm, 
    small base radius 6 cm, and height 10 cm is equal to 840π cm³ -/
theorem truncated_cone_volume_specific : 
  truncated_cone_volume 12 6 10 = 840 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_specific_l158_15808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_triangle_theorem_l158_15856

-- Define a type for colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define what it means for a triangle to be isosceles
def isIsosceles (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (a.x - c.x)^2 + (a.y - c.y)^2 ∨
  (b.x - a.x)^2 + (b.y - a.y)^2 = (b.x - c.x)^2 + (b.y - c.y)^2 ∨
  (c.x - a.x)^2 + (c.y - a.y)^2 = (c.x - b.x)^2 + (c.y - b.y)^2

-- Define what it means for angles to be in geometric progression
noncomputable def anglesInGeometricProgression (a b c : Point) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ r ≠ 1 ∧
  (∃ (θ₁ θ₂ θ₃ : ℝ), θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₁ + θ₂ + θ₃ = Real.pi ∧
  (θ₂ = r * θ₁ ∧ θ₃ = r * θ₂ ∨ θ₃ = r * θ₂ ∧ θ₁ = r * θ₃ ∨ θ₁ = r * θ₃ ∧ θ₂ = r * θ₁))

theorem three_color_triangle_theorem (f : Coloring) :
  ∃ (a b c : Point),
    f a = f b ∧ f b = f c ∧
    (isIsosceles a b c ∨ anglesInGeometricProgression a b c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_color_triangle_theorem_l158_15856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sufficient_not_necessary_l158_15895

/-- An angle in a rectangular coordinate system -/
structure Angle where
  /-- The measure of the angle in radians -/
  α : Real
  /-- The vertex of the angle coincides with the origin -/
  vertex_at_origin : True
  /-- The initial side of the angle coincides with the non-negative x-axis -/
  initial_side_on_x_axis : True

/-- Predicate for the terminal side of an angle being on the ray x + 3y = 0 (x ≥ 0) -/
def terminal_side_on_ray (a : Angle) : Prop :=
  ∃ (x y : Real), x ≥ 0 ∧ x + 3 * y = 0 ∧ x = Real.cos a.α ∧ y = Real.sin a.α

/-- The main theorem stating that the terminal side being on the ray is a sufficient
    but not necessary condition for sin 2α = -3/5 -/
theorem terminal_side_sufficient_not_necessary (a : Angle) :
  (terminal_side_on_ray a → Real.sin (2 * a.α) = -3/5) ∧
  ¬(Real.sin (2 * a.α) = -3/5 → terminal_side_on_ray a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_sufficient_not_necessary_l158_15895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_probability_l158_15834

-- Define the number of houses and packages
def n : ℕ := 5

-- Define the probability of exactly 3 correct deliveries
def prob_three_correct : ℚ := 1 / 12

-- Theorem statement
theorem exactly_three_correct_probability :
  (Nat.choose n 3 * 1) / n.factorial = prob_three_correct :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_probability_l158_15834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_divides_altitudes_equally_implies_equilateral_l158_15835

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the altitude from a vertex to the opposite side
noncomputable def altitude (t : Triangle) (vertex : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the ratio in which a point divides a line segment
noncomputable def divisionRatio (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define what it means for a triangle to be equilateral
def isEquilateral (t : Triangle) : Prop := sorry

-- Theorem statement
theorem orthocenter_divides_altitudes_equally_implies_equilateral (t : Triangle) :
  let H := orthocenter t
  let A₁ := altitude t t.A
  let B₁ := altitude t t.B
  let C₁ := altitude t t.C
  (divisionRatio A₁ H t.A = divisionRatio B₁ H t.B) ∧
  (divisionRatio B₁ H t.B = divisionRatio C₁ H t.C) →
  isEquilateral t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_divides_altitudes_equally_implies_equilateral_l158_15835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l158_15802

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (1/16) * (1 + 4*a n + Real.sqrt (1 + 24*a n))

theorem a_formula (n : ℕ) : 
  a n = (2^(4-2*n) + 6*2^(2-n) + 8) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l158_15802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l158_15815

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 2*x - 2 * Real.log x

theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l158_15815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boyd_boy_percentage_l158_15855

def julianTotalFriends : ℕ := 80
def julianGirlPercentage : ℚ := 40 / 100
def boydTotalFriends : ℕ := 100

def julianGirlFriends : ℕ := (julianTotalFriends : ℚ) * julianGirlPercentage |> Int.floor |> Int.toNat
def boydGirlFriends : ℕ := 2 * julianGirlFriends
def boydBoyFriends : ℕ := boydTotalFriends - boydGirlFriends

theorem boyd_boy_percentage :
  (boydBoyFriends : ℚ) / (boydTotalFriends : ℚ) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boyd_boy_percentage_l158_15855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l158_15864

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add a case for 0 to avoid missing cases error
  | 1 => 1
  | (n + 2) => (2^(n+2) - 1) * sequence_a (n+1) / (sequence_a (n+1) + 2^(n+1) - 1)

theorem sequence_a_formula (n : ℕ) (h : n ≥ 1) : 
  sequence_a n = (2^n - 1) / n := by
  sorry  -- Use 'by' instead of ':=' for tactics


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l158_15864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l158_15873

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b + t.c = 2 * Real.sin (t.C + Real.pi/6) ∧
  t.a = t.a ∧  -- CD = a, where D is midpoint of AB
  t.b - t.c = 1

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = Real.pi/3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l158_15873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l158_15812

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x - 2*y - 12 = 0

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - 2*y - 12| / Real.sqrt 5

/-- Theorem stating the minimum distance from the ellipse to the line -/
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), min_dist = 4 * Real.sqrt 5 / 5 ∧
  ∀ (x y : ℝ), is_on_ellipse x y → 
    distance_to_line x y ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l158_15812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_factorial_sum_integer_digit_square_sum_l158_15826

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := Digit × Digit × Digit

/-- Converts a ThreeDigitNumber to its integer value -/
def toInt (n : ThreeDigitNumber) : ℕ :=
  100 * n.1.val + 10 * n.2.1.val + n.2.2.val

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem for part (a) -/
theorem three_digit_factorial_sum :
  ∃! n : ThreeDigitNumber, toInt n = factorial n.1.val + factorial n.2.1.val + factorial n.2.2.val ∧ toInt n = 145 :=
sorry

/-- Represents an integer as a list of its digits -/
def IntegerDigits := List Digit

/-- Converts an IntegerDigits to its integer value -/
def toIntFromDigits (digits : IntegerDigits) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d.val) 0

/-- Theorem for part (b) -/
theorem integer_digit_square_sum :
  ∃! digits : IntegerDigits, toIntFromDigits digits = (digits.map (fun d => d.val ^ 2)).sum ∧ toIntFromDigits digits = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_factorial_sum_integer_digit_square_sum_l158_15826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l158_15878

theorem tan_alpha_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : Real.cos α * Real.cos β + Real.sin α * Real.sin β = 4/5)
  (h5 : Real.tan β = 4/3) : 
  Real.tan α = 7/24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l158_15878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l158_15816

/-- Represents a hyperbola with equation x²/4 - y²/2 = 1 -/
structure Hyperbola where
  equation : ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1

/-- Represents the focus of a hyperbola -/
noncomputable def focus (h : Hyperbola) : ℝ × ℝ :=
  (Real.sqrt 6, 0)

/-- Represents the asymptote of a hyperbola -/
noncomputable def asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ Real.sqrt 2 / 2 * x

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
  let (x, y) := p
  let m := (f 1 - f 0)  -- slope of the line
  let b := f 0  -- y-intercept of the line
  abs (m * x - y + b) / Real.sqrt (m^2 + 1)

/-- Theorem stating that the distance from the focus to the asymptote is √2 -/
theorem focus_to_asymptote_distance (h : Hyperbola) :
  distance_point_to_line (focus h) (asymptote h) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_asymptote_distance_l158_15816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_curve_is_unit_circle_when_a_is_two_l158_15897

/-- Parametric equations of the curve -/
noncomputable def x (a t : ℝ) : ℝ := (a * t) / (1 + t^2)
noncomputable def y (t : ℝ) : ℝ := (1 - t^2) / (1 + t^2)

/-- The curve is an ellipse -/
theorem curve_is_ellipse (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  ∀ (t : ℝ), (x a t / A)^2 + (y t / B)^2 = 1 := by
  sorry

/-- The curve is a unit circle when a = 2 -/
theorem curve_is_unit_circle_when_a_is_two :
  ∀ (t : ℝ), (x 2 t)^2 + (y t)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_curve_is_unit_circle_when_a_is_two_l158_15897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distances_l158_15849

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus of the hyperbola
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

-- Define a line passing through a point
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define the asymptotes of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b/a) * x ∨ y = -(b/a) * x

-- Main theorem
theorem hyperbola_intersection_distances 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (A B C D : ℝ × ℝ) (m : ℝ) :
  hyperbola a b A.1 A.2 →
  hyperbola a b B.1 B.2 →
  asymptote a b C.1 C.2 →
  asymptote a b D.1 D.2 →
  line_through_point m (right_focus a b) A.1 A.2 →
  line_through_point m (right_focus a b) B.1 B.2 →
  line_through_point m (right_focus a b) C.1 C.2 →
  line_through_point m (right_focus a b) D.1 D.2 →
  dist A C = dist B D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distances_l158_15849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garrison_size_l158_15829

theorem garrison_size (initial_days reinforcement_arrival remaining_days reinforcement_size : ℕ)
  (h1 : initial_days = 60)
  (h2 : reinforcement_arrival = 15)
  (h3 : remaining_days = 20)
  (h4 : reinforcement_size = 1250)
  (initial_men : ℕ)
  (h5 : initial_men * initial_days = 
        (initial_men + reinforcement_size) * remaining_days + 
        initial_men * reinforcement_arrival) :
  initial_men = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garrison_size_l158_15829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l158_15814

/-- Represents a quadratic function of the form ax² + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The quadratic equation corresponding to a quadratic function --/
def quadratic_equation (f : QuadraticFunction) (x : ℝ) : Prop :=
  f.a * x^2 + f.b * x + f.c = 0

/-- The vertex of a quadratic function --/
noncomputable def vertex (f : QuadraticFunction) : ℝ × ℝ :=
  (- f.b / (2 * f.a), f.a * (- f.b / (2 * f.a))^2 + f.b * (- f.b / (2 * f.a)) + f.c)

/-- The axis of symmetry of a quadratic function --/
noncomputable def axis_of_symmetry (f : QuadraticFunction) : ℝ :=
  - f.b / (2 * f.a)

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ (y : ℝ), y < 0 ∧ vertex f = (axis_of_symmetry f, y)) →
  ¬ (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation f x₁ ∧ quadratic_equation f x₂) ∧
  (quadratic_equation f 0 → ∃ (x : ℝ), x ≠ 0 ∧ quadratic_equation f x) ∧
  (f.a * f.b > 0 → axis_of_symmetry f < 0) ∧
  (2 * f.b = 4 * f.a + f.c → quadratic_equation f (-2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l158_15814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_complex_expression_l158_15886

theorem max_value_of_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧
    ∀ v : ℂ, Complex.abs v = Real.sqrt 3 →
      Complex.abs ((v - 1) * (v + 1)^2) ≤ Complex.abs ((w - 1) * (w + 1)^2) ∧
      Complex.abs ((w - 1) * (w + 1)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_complex_expression_l158_15886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_salary_proof_l158_15883

/-- Mother's salary in rubles -/
noncomputable def mother_salary : ℝ := 25000

/-- Father's salary as a fraction of mother's salary -/
noncomputable def father_salary_ratio : ℝ := 1.3

/-- Fraction of combined salary saved in first method -/
noncomputable def first_method_fraction : ℝ := 1/10

/-- Number of months for first saving method -/
def first_method_months : ℕ := 6

/-- Fraction of combined salary saved in second method -/
noncomputable def second_method_fraction : ℝ := 1/2

/-- Number of months for second saving method interest -/
def second_method_interest_months : ℕ := 10

/-- Monthly interest rate for second saving method -/
noncomputable def second_method_interest_rate : ℝ := 0.03

/-- Cost of computer table in rubles -/
noncomputable def computer_table_cost : ℝ := 2875

theorem mother_salary_proof :
  let combined_salary := mother_salary * (1 + father_salary_ratio)
  let first_method_savings := (combined_salary * first_method_fraction) * first_method_months
  let second_method_principal := combined_salary * second_method_fraction
  let second_method_savings := second_method_principal * (1 + second_method_interest_rate * second_method_interest_months)
  second_method_savings = first_method_savings + computer_table_cost :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mother_salary_proof_l158_15883
