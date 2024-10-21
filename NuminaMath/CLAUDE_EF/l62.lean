import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l62_6201

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x + 6 else x + 6

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < 1) ∨ x > 3

-- Theorem statement
theorem f_inequality_solution_set :
  ∀ x : ℝ, f x > f 1 ↔ solution_set x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_l62_6201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminate_plane_l62_6232

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a spotlight
structure Spotlight where
  position : Point2D
  angle : ℝ
  direction : ℝ

-- Helper function to determine if a point is illuminated by a spotlight
def illuminated (s : Spotlight) (p : Point2D) : Prop :=
  sorry  -- Definition of illumination based on spotlight properties and point position

-- Define the theorem
theorem illuminate_plane (points : Fin 4 → Point2D) : 
  ∃ (spotlights : Fin 4 → Spotlight), 
    (∀ i : Fin 4, (spotlights i).angle = 90 ∧ (spotlights i).position = points i) ∧ 
    (∀ p : Point2D, ∃ i : Fin 4, illuminated (spotlights i) p) :=
by
  sorry  -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminate_plane_l62_6232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_math_tournament_clothing_l62_6203

/-- At the Rice Mathematics Tournament:
  * 80% of contestants wear blue jeans
  * 70% of contestants wear tennis shoes
  * 80% of those who wear blue jeans also wear tennis shoes
This theorem proves that the fraction of people wearing tennis shoes
who are also wearing blue jeans is 32/35. -/
theorem rice_math_tournament_clothing (N : ℝ) (h : N > 0) :
  let blue_jeans := 0.8 * N
  let tennis_shoes := 0.7 * N
  let both := 0.8 * blue_jeans
  (both / tennis_shoes) = 32 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_math_tournament_clothing_l62_6203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_asymptote_angle_l62_6254

-- Define the hyperbolas
structure Hyperbola where
  a : ℝ
  b : ℝ

-- Define the equation of a hyperbola
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Define the slope angle of an asymptote
noncomputable def asymptote_slope_angle (h : Hyperbola) : ℝ :=
  Real.arctan (h.b / h.a)

-- Theorem statement
theorem hyperbola_equation_from_asymptote_angle 
  (C₁ C₂ : Hyperbola)
  (h_foci : C₁.a^2 + C₁.b^2 = C₂.a^2 + C₂.b^2)  -- Coincident foci
  (h_C₁_eq : C₁.a^2 = 3 ∧ C₁.b^2 = 1)  -- Equation of C₁
  (h_angle : asymptote_slope_angle C₂ = 2 * asymptote_slope_angle C₁)
  (x y : ℝ) :
  hyperbola_equation C₂ x y ↔ x^2 - y^2/3 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_from_asymptote_angle_l62_6254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_irregular_set_l62_6264

/-- A set A is irregular if for any different elements x and y in A,
    there is no element of the form x + k(y - x) different from x and y,
    where k is an integer. -/
def Irregular (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → ∀ k : ℤ, x + k * (y - x) ∈ A → x + k * (y - x) = x ∨ x + k * (y - x) = y

/-- There exists an infinite irregular set of integers. -/
theorem exists_infinite_irregular_set : ∃ A : Set ℤ, Set.Infinite A ∧ Irregular A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_irregular_set_l62_6264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l62_6277

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The distance from the center to a focus -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem about the ellipse with given conditions -/
theorem ellipse_theorem (e : Ellipse) 
    (h_eccentricity : e.eccentricity = Real.sqrt 3 / 3)
    (h_intercept : 2 * Real.sqrt (e.b^2 - e.focal_distance^2) = 4 * Real.sqrt 3 / 3) :
  e.a = Real.sqrt 3 ∧ e.b = Real.sqrt 2 ∧
  (∃ (k : ℝ), k^2 = 10 ∧
    let x1 := -3 * k^2 / (2 + 3 * k^2)
    let x2 := (3 * k^2 - 6) / ((2 + 3 * k^2) * x1)
    let y1 := k * (x1 + 1)
    let y2 := k * (x2 + 1)
    (x1 + Real.sqrt 3) * (Real.sqrt 3 - x2) + y1 * (-y2) +
    (x2 + Real.sqrt 3) * (Real.sqrt 3 - x1) + y2 * (-y1) = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l62_6277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_condition_minimum_value_l62_6226

/-- The function f(x) = ln x - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem monotone_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, MonotoneOn (f a) (Set.Icc 2 3)) ↔ a ≥ (1 / 2) := by sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ min (-a) (Real.log 2 - 2 * a)) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = min (-a) (Real.log 2 - 2 * a)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_condition_minimum_value_l62_6226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equal_and_irrational_l62_6223

-- Define the quadratic equation
def quadratic_equation (x c : ℝ) : Prop :=
  3 * x^2 - 6 * x * Real.sqrt 3 + c = 0

-- Define the discriminant
noncomputable def discriminant (c : ℝ) : ℝ :=
  (-6 * Real.sqrt 3)^2 - 4 * 3 * c

-- Theorem statement
theorem roots_equal_and_irrational :
  ∃ c : ℝ, discriminant c = 0 ∧
  (∃ x : ℝ, quadratic_equation x c ∧ Real.sqrt 3 = x) ∧
  (∀ y : ℝ, quadratic_equation y c → y = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equal_and_irrational_l62_6223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l62_6222

theorem integer_sum_problem (a b : ℕ+) : 
  a * b + a + b = 156 →
  Nat.gcd a b = 1 →
  a < 25 →
  b = 2 * a →
  a + b = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sum_problem_l62_6222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l62_6239

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  -- We'll use 2 as our maximum value
  use 2
  constructor
  · -- Prove M = 2
    rfl
  · -- Prove ∀ (x : ℝ), f x ≤ M
    intro x
    cases' le_or_gt x 1 with h1 h2
    · -- Case: x ≤ 1
      rw [f]
      simp [h1]
      sorry -- Complete the proof for this case
    · -- Case: x > 1
      rw [f]
      simp [h2]
      sorry -- Complete the proof for this case


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l62_6239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l62_6215

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then |x - 1| else 3^x

-- Theorem statement
theorem f_properties :
  (f (f (-2)) = 27) ∧ (∀ a : ℝ, f a = 2 ↔ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l62_6215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_to_cover_hexagon_l62_6296

/-- The side length of the small equilateral triangle -/
noncomputable def small_triangle_side : ℝ := 2

/-- The side length of the regular hexagon -/
noncomputable def hexagon_side : ℝ := 10

/-- The area of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- The area of a regular hexagon given its side length -/
noncomputable def regular_hexagon_area (side : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side^2

/-- The minimum number of small triangles needed to cover the hexagon -/
noncomputable def min_triangles : ℕ := 
  Int.toNat ⌈(regular_hexagon_area hexagon_side) / (equilateral_triangle_area small_triangle_side)⌉

theorem min_triangles_to_cover_hexagon : 
  min_triangles = 150 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangles_to_cover_hexagon_l62_6296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_ticket_cost_is_20_l62_6263

/-- The cost of a children's ticket to an aquarium -/
def childrens_ticket_cost (total_cost adult_cost num_children : ℕ) : ℕ :=
  (total_cost - adult_cost) / num_children

theorem childrens_ticket_cost_is_20 :
  childrens_ticket_cost 155 35 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_ticket_cost_is_20_l62_6263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l62_6286

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin x * Real.cos (x + Real.pi/6) +
  Real.cos x * Real.sin (x + Real.pi/3) +
  Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∀ x ∈ Set.Ioo 0 (Real.pi/2), f x ∈ Set.Ioc (-Real.sqrt 3) 2) ∧
  (∀ α β : ℝ,
    α ∈ Set.Ioo (Real.pi/12) (Real.pi/3) →
    β ∈ Set.Ioo (-Real.pi/6) (Real.pi/12) →
    f α = 6/5 →
    f β = 10/13 →
    Real.cos (2*α - 2*β) = -33/65) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l62_6286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l62_6261

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (1/2) * (a + b) * h

/-- Theorem: For a trapezium with parallel sides of lengths 20 cm and 18 cm, and an area of 247 square centimeters, the distance between the parallel sides is 13 cm. -/
theorem trapezium_height_calculation :
  let a := (20 : ℝ)
  let b := (18 : ℝ)
  let area := (247 : ℝ)
  ∃ h : ℝ, trapezium_area a b h = area ∧ h = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_calculation_l62_6261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karens_speed_l62_6219

/-- Proves that Karen's average speed is 60 mph given the race conditions -/
theorem karens_speed (karen_start_delay : ℝ) (karen_win_margin : ℝ) 
  (tom_speed : ℝ) (tom_distance : ℝ) 
  (h1 : karen_start_delay = 4 / 60)  -- 4 minutes in hours
  (h2 : karen_win_margin = 4)        -- miles
  (h3 : tom_speed = 45)              -- mph
  (h4 : tom_distance = 24)           -- miles
  : karen_speed = 60 := by
  sorry

where
  karen_speed : ℝ := 
    (tom_distance + karen_win_margin) / 
    (tom_distance / tom_speed - karen_start_delay)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karens_speed_l62_6219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_spherical_to_rectangular_example_l62_6273

/-- Modified spherical coordinates to rectangular coordinates conversion --/
noncomputable def modified_spherical_to_rectangular (ρ θ φ k : ℝ) : ℝ × ℝ × ℝ :=
  (k * ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- Theorem stating the equivalence of the given modified spherical coordinates to rectangular coordinates --/
theorem modified_spherical_to_rectangular_example :
  let ρ : ℝ := 5
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let k : ℝ := 1.5
  modified_spherical_to_rectangular ρ θ φ k = (0, -5 * Real.sqrt 3 / 2, 5 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_spherical_to_rectangular_example_l62_6273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_equal_volume_l62_6225

theorem sphere_cone_equal_volume (r s h R : ℝ) : 
  r > 0 → s > 0 → h > 0 → R > 0 →
  π * r * s = 80 * π →
  π * r^2 + π * r * s = 144 * π →
  (1/3) * π * r^2 * h = (4/3) * π * R^3 →
  R = (96 : ℝ)^(1/3) := by
  sorry

#check sphere_cone_equal_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cone_equal_volume_l62_6225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_property_l62_6212

/-- An ellipse C in the xy-plane -/
def Ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

/-- A point E on the x-axis -/
noncomputable def E : ℝ × ℝ := (Real.sqrt 3, 0)

/-- The squared distance between two points -/
def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: For any chord AB of ellipse C passing through E, 1/EA^2 + 1/EB^2 = 2 -/
theorem ellipse_chord_property :
  ∀ A B : ℝ × ℝ,
  Ellipse A.1 A.2 → Ellipse B.1 B.2 →
  (B.2 - A.2) * E.1 = (B.1 - A.1) * E.2 →  -- AB passes through E
  1 / dist_squared E A + 1 / dist_squared E B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_property_l62_6212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l62_6249

theorem tan_theta_minus_pi_over_four (θ : Real) 
  (h1 : π < θ) (h2 : θ < 3*π/2) (h3 : Real.sin θ = -3/5) : 
  Real.tan (θ - π/4) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l62_6249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l62_6295

/-- Represents a triangle with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The triangle is acute -/
def Triangle.isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

/-- The point (a, b) lies on the given line -/
def Triangle.pointOnLine (t : Triangle) : Prop :=
  t.a * (Real.sin t.A - Real.sin t.B) + t.b * Real.sin t.B = t.c * Real.sin t.C

/-- The triangle satisfies the given equation -/
def Triangle.satisfiesEquation (t : Triangle) (m : ℝ) : Prop :=
  m / Real.tan t.C = 1 / Real.tan t.A + 1 / Real.tan t.B

theorem triangle_properties (t : Triangle) (m : ℝ)
  (h1 : t.isAcute)
  (h2 : t.pointOnLine)
  (h3 : t.satisfiesEquation m) :
  t.C = Real.pi/3 ∧ ∃ (m_min : ℝ), m_min = 2 ∧ ∀ (m' : ℝ), t.satisfiesEquation m' → m_min ≤ m' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l62_6295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l62_6209

theorem subset_intersection_theorem (n : ℕ) (X : Finset ℕ) (S : Finset (Finset ℕ)) 
  (h1 : n ≥ 5)
  (h2 : Finset.card X = n)
  (h3 : Finset.card S = n + 1)
  (h4 : ∀ s ∈ S, Finset.card s = 3)
  (h5 : ∀ s ∈ S, s ⊆ X) :
  ∃ s1 s2, s1 ∈ S ∧ s2 ∈ S ∧ s1 ≠ s2 ∧ Finset.card (s1 ∩ s2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l62_6209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l62_6224

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin (Real.pi - x) * Real.cos (2*Real.pi - x) * Real.tan (-x + Real.pi)) / Real.cos (-Real.pi/2 + x)

theorem f_value : f (-31*Real.pi/3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_l62_6224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l62_6271

/-- Rectangle ABCD with given dimensions and midpoints -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  h_AB : B.1 - A.1 = 3
  h_BC : C.2 - B.2 = 4
  h_E_midpoint : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_F_midpoint : F = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  h_G_midpoint : G = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
  h_H_midpoint : H = ((G.1 + E.1) / 2, (G.2 + E.2) / 2)

/-- Semicircle with diameter AD -/
def semicircle (r : Rectangle) : Set (ℝ × ℝ) :=
  {p | (p.1 - r.A.1)^2 + (p.2 - r.A.2)^2 ≤ (r.D.1 - r.A.1)^2 + (r.D.2 - r.A.2)^2 ∧ p.2 ≥ r.A.2}

/-- Quadrilateral EHGF -/
def quadrilateral (r : Rectangle) : Set (ℝ × ℝ) :=
  {p | ∃ (t₁ t₂ : ℝ), 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧
    p = (t₁ * r.E.1 + (1 - t₁) * r.H.1, t₂ * r.G.1 + (1 - t₂) * r.F.1)}

/-- The main theorem -/
theorem intersection_area (r : Rectangle) : 
  MeasureTheory.volume (semicircle r ∩ quadrilateral r) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l62_6271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_midpoint_distance_l62_6233

/-- Represents a parabola with equation y² = 6x -/
structure Parabola where
  equation : ∀ x y, y^2 = 6*x

/-- Represents a line that passes through the focus of the parabola -/
structure FocusLine where
  passesThroughFocus : Bool

/-- Represents two points of intersection between the line and the parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The main theorem -/
theorem parabola_intersection_midpoint_distance
  (p : Parabola)
  (l : FocusLine)
  (i : Intersection)
  (h : l.passesThroughFocus = true)
  (d : abs (i.A.1 - i.B.1) + abs (i.A.2 - i.B.2) = 9) :
  let midpoint := ((i.A.1 + i.B.1) / 2, (i.A.2 + i.B.2) / 2)
  ∃ directrix : ℝ → ℝ × ℝ,
    abs (midpoint.1 - (directrix 0).1) + abs (midpoint.2 - (directrix 0).2) = 9/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_midpoint_distance_l62_6233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_recipe_scaling_l62_6244

/-- Given a recipe for muffins, calculate the required ingredients for a different batch size. -/
theorem muffin_recipe_scaling (original_muffins original_flour original_sugar target_muffins : ℚ)
  (h1 : original_muffins = 48)
  (h2 : original_flour = 3)
  (h3 : original_sugar = 2)
  (h4 : target_muffins = 72) :
  (target_muffins / original_muffins) * original_flour = 4.5 ∧
  (target_muffins / original_muffins) * original_sugar = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_recipe_scaling_l62_6244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l62_6266

noncomputable section

open Real

theorem triangle_property (a b c A B C : ℝ) :
  -- Given conditions
  (sin C * sin (A - B) = sin B * sin (C - A)) →
  -- Part 1: Prove 2a² = b² + c²
  (2 * a^2 = b^2 + c^2) ∧
  -- Part 2: Prove perimeter is 14 when a = 5 and cos A = 25/31
  (a = 5 ∧ cos A = 25/31 → a + b + c = 14) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l62_6266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l62_6279

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def line (x y : ℝ) : Prop := y = x - 1

-- Define the chord length
noncomputable def chord_length (p : ℝ) : ℝ := 2 * Real.sqrt 6

-- Define the triangle area
noncomputable def triangle_area (x₀ : ℝ) : ℝ := 5 * Real.sqrt 3

theorem parabola_intersection_theorem (p : ℝ) (h₁ : p > 0) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (chord_length p)^2) →
  p = 1 ∧
  (∃ x₀ : ℝ, (x₀ = -4 ∨ x₀ = 6) ∧
    triangle_area x₀ = (1/2) * chord_length p * |x₀ - 1| / Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l62_6279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_radius_correct_l62_6258

/-- The area of the circular sector -/
noncomputable def sector_area : ℝ := 100

/-- The perimeter of the circular sector as a function of radius -/
noncomputable def sector_perimeter (r : ℝ) : ℝ := 2 * r + 200 / r

/-- The radius that minimizes the perimeter -/
noncomputable def min_perimeter_radius : ℝ := 10

/-- Theorem stating that the min_perimeter_radius minimizes the perimeter -/
theorem min_perimeter_radius_correct :
  ∀ r : ℝ, r > 0 → sector_perimeter min_perimeter_radius ≤ sector_perimeter r := by
  sorry

#check min_perimeter_radius_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_radius_correct_l62_6258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_area_ratio_cd_equals_twelve_l62_6278

-- Define a regular octagon
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the circle inscribed in the octagon
structure InscribedCircle (octagon : RegularOctagon) where
  radius : ℝ
  radius_pos : radius > 0
  touches_midpoints : radius = octagon.side_length * Real.sqrt (2 + Real.sqrt 2) / 2

-- Theorem statement
theorem octagon_circle_area_ratio 
  (octagon : RegularOctagon) 
  (circle : InscribedCircle octagon) : 
  (Real.pi * circle.radius ^ 2) / (2 * (1 + Real.sqrt 2) * octagon.side_length ^ 2) = 1 / 12 :=
sorry

-- Define the product cd
def cd : ℕ := 12

-- Theorem to prove that cd = 12
theorem cd_equals_twelve : cd = 12 :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circle_area_ratio_cd_equals_twelve_l62_6278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l62_6257

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define a point on the line
def P : Type := {p : ℝ × ℝ // line_eq p.1 p.2}

-- Define the circle
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | circle_eq p.1 p.2}

-- Define tangent points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define area of quadrilateral (placeholder function)
noncomputable def area_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_area_PACB : 
  ∀ (p : P), 
  ∃ (a b : ℝ × ℝ), 
  a ∈ C ∧ b ∈ C ∧ 
  (∀ (q : ℝ × ℝ), q ∈ C → (p.val.1 - q.1)^2 + (p.val.2 - q.2)^2 ≥ (p.val.1 - a.1)^2 + (p.val.2 - a.2)^2) ∧
  (∀ (q : ℝ × ℝ), q ∈ C → (p.val.1 - q.1)^2 + (p.val.2 - q.2)^2 ≥ (p.val.1 - b.1)^2 + (p.val.2 - b.2)^2) ∧
  (area_quadrilateral p.val A B (1, 1) ≥ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l62_6257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_chord_length_l62_6267

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line through the focus of a parabola with slope angle π/3 -/
structure FocusLine (para : Parabola) where
  slope_angle : ℝ
  h_slope : slope_angle = π / 3

/-- Represents the intersection points of the focus line and the parabola -/
structure IntersectionPoints (para : Parabola) (line : FocusLine para) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_parabola : (A.2)^2 = 2 * para.p * A.1 ∧ (B.2)^2 = 2 * para.p * B.1
  h_on_line : A.2 = Real.sqrt 3 * (A.1 - para.p / 2) ∧ B.2 = Real.sqrt 3 * (B.1 - para.p / 2)
  h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

/-- The main theorem statement -/
theorem unique_chord_length (para : Parabola) (line : FocusLine para) 
  (points : IntersectionPoints para line) : 
  ∃! chord : (ℝ × ℝ) × (ℝ × ℝ), 
    (chord.1.2)^2 = 2 * para.p * chord.1.1 ∧ 
    (chord.2.2)^2 = 2 * para.p * chord.2.1 ∧ 
    Real.sqrt ((chord.1.1 - chord.2.1)^2 + (chord.1.2 - chord.2.2)^2) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_chord_length_l62_6267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_360_l62_6272

theorem count_divisors_of_360 : 
  Finset.card (Nat.divisors 360) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_360_l62_6272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_collinearity_l62_6251

/-- Two circles in a plane -/
structure IntersectingCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : A ∈ S₁ ∩ S₂
  h₂ : B ∈ S₁ ∩ S₂

/-- Rotational homothety centered at a point -/
structure RotationalHomothety where
  center : ℝ × ℝ
  source : Set (ℝ × ℝ)
  target : Set (ℝ × ℝ)
  map : (ℝ × ℝ) → (ℝ × ℝ)
  h : ∀ x ∈ source, map x ∈ target

/-- Collinearity of three points -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

/-- Main theorem -/
theorem homothety_collinearity 
  (C : IntersectingCircles) 
  (P : RotationalHomothety) 
  (h_center : P.center = C.A)
  (h_map : P.source = C.S₁ ∧ P.target = C.S₂)
  (M₁ : ℝ × ℝ) 
  (h_M₁ : M₁ ∈ C.S₁)
  (M₂ : ℝ × ℝ) 
  (h_M₂ : M₂ = P.map M₁) :
  collinear M₁ C.B M₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homothety_collinearity_l62_6251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_product_l62_6289

theorem factorial_nine_equals_product (m : ℕ) : 2^3 * 3^3 * m = Nat.factorial 9 ↔ m = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_nine_equals_product_l62_6289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_fg_monotone_increasing_l62_6242

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 4)
noncomputable def g (x : ℝ) : ℝ := cos (2 * x + π / 4)

-- Theorem for the distance between intersection points
theorem intersection_distance :
  ∀ x y : ℝ, f x = g x → f y = g y → x ≠ y → |x - y| = π / 2 ∨ |x - y| > π / 2 :=
sorry

-- Theorem for the monotonicity of f * g
theorem fg_monotone_increasing :
  StrictMonoOn (fun x => f x * g x) (Set.Ioo (π / 4) (π / 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_fg_monotone_increasing_l62_6242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_45_degrees_l62_6229

/-- The curve function f(x) = x³ - 2x + 4 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The slope of the tangent line at x = 1 -/
def tangent_slope : ℝ := f' 1

/-- The slope angle of the tangent line in radians -/
noncomputable def slope_angle : ℝ := Real.arctan tangent_slope

theorem tangent_slope_angle_45_degrees :
  slope_angle = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_45_degrees_l62_6229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_less_than_e_l62_6243

/-- The function f(x) = x(1 - ln(x)) -/
noncomputable def f (x : ℝ) : ℝ := x * (1 - Real.log x)

/-- Theorem: If x₁ and x₂ are distinct positive real numbers such that f(x₁) = f(x₂), 
    then x₁ + x₂ < e -/
theorem sum_less_than_e (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) 
    (h₄ : f x₁ = f x₂) : x₁ + x₂ < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_less_than_e_l62_6243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l62_6291

theorem division_problem (x y : ℕ) (h1 : x % y = 12) (h2 : (x : ℝ) / (y : ℝ) = 75.12) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l62_6291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l62_6216

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Theorem stating that the maximum value of f on the interval is 21
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ interval ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ f c ∧ f c = 21 := by
  sorry

#check max_value_of_f_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l62_6216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_in_fourth_quadrant_l62_6211

def z₁ : ℂ := 1 - 3 * Complex.I
def z₂ : ℂ := 3 - 2 * Complex.I

theorem z₁_div_z₂_in_fourth_quadrant :
  let w := z₁ / z₂
  0 < w.re ∧ w.im < 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_div_z₂_in_fourth_quadrant_l62_6211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_on_playground_l62_6280

theorem girls_on_playground (total_students : ℕ) 
  (h1 : total_students = 20) 
  (students_on_playground : ℕ)
  (h2 : (total_students / 4 : ℚ) = (total_students - students_on_playground : ℚ)) 
  (boys_on_playground : ℕ)
  (h3 : (students_on_playground / 3 : ℚ) = (boys_on_playground : ℚ)) : 
  students_on_playground - boys_on_playground = 10 :=
by
  sorry

#check girls_on_playground

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_on_playground_l62_6280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_placements_l62_6253

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- Represents a 1x3 rectangle --/
structure Rectangle where
  start_row : Fin 5
  start_col : Fin 5
  is_vertical : Bool

/-- Checks if a rectangle is valid (within grid bounds) --/
def is_valid_rectangle (r : Rectangle) : Bool :=
  if r.is_vertical then
    r.start_row.val + 2 < 5
  else
    r.start_col.val + 2 < 5

/-- Checks if a rectangle overlaps with pre-blackened squares or shares an edge/vertex --/
def is_not_overlapping (grid : Grid) (r : Rectangle) : Bool :=
  sorry

/-- Counts the number of valid placements for a 1x3 rectangle --/
def count_valid_placements (grid : Grid) : Nat :=
  sorry

theorem eight_placements (grid : Grid) 
  (h1 : ∃ (a b c : Fin 5 × Fin 5), grid a.fst a.snd ∧ grid b.fst b.snd ∧ grid c.fst c.snd) 
  (h2 : ∀ (i j : Fin 5), grid i j → (i, j) ∈ [a, b, c]) :
  count_valid_placements grid = 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_placements_l62_6253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l62_6204

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x + b / x

theorem f_monotone_increasing (b : ℝ) :
  (∃ c ∈ Set.Ioo 1 2, (deriv (f b)) c = 0) →
  StrictMonoOn (f b) (Set.Ioi 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l62_6204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l62_6269

def U : Set Int := {-1, 0, 1}
def A : Set Int := {0, 1}

theorem complement_of_A_in_U : Set.compl A ∩ U = {-1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l62_6269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivias_siblings_l62_6220

-- Define the characteristics
inductive EyeColor | Green | Gray
inductive HairColor | Red | Brown
inductive Age | Ten | Twelve

-- Define a child's characteristics
structure Child where
  name : String
  eye : EyeColor
  hair : HairColor
  age : Age

def children : List Child := [
  ⟨"Olivia", EyeColor.Green, HairColor.Red, Age.Twelve⟩,
  ⟨"Henry", EyeColor.Gray, HairColor.Brown, Age.Twelve⟩,
  ⟨"Lucas", EyeColor.Green, HairColor.Red, Age.Ten⟩,
  ⟨"Emma", EyeColor.Green, HairColor.Brown, Age.Twelve⟩,
  ⟨"Mia", EyeColor.Gray, HairColor.Red, Age.Ten⟩,
  ⟨"Noah", EyeColor.Gray, HairColor.Brown, Age.Twelve⟩
]

-- Define a function to check if two children share at least one characteristic
def shareCharacteristic (c1 c2 : Child) : Prop :=
  c1.eye = c2.eye ∨ c1.hair = c2.hair ∨ c1.age = c2.age

-- Define a function to check if three children are siblings
def areSiblings (c1 c2 c3 : Child) : Prop :=
  shareCharacteristic c1 c2 ∧ shareCharacteristic c2 c3 ∧ shareCharacteristic c1 c3

-- Theorem to prove
theorem olivias_siblings :
  ∃ (c1 c2 : Child),
    c1.name = "Lucas" ∧
    c2.name = "Emma" ∧
    (∃ (olivia : Child), olivia.name = "Olivia" ∧ areSiblings olivia c1 c2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivias_siblings_l62_6220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trams_to_add_l62_6284

/-- The number of trams needed to reduce intervals by one-fifth -/
def additional_trams (initial_trams : ℕ) (interval_reduction : ℚ) : ℕ :=
  let new_trams := (initial_trams : ℚ) / (1 - interval_reduction)
  (Int.ceil new_trams).toNat - initial_trams

/-- Theorem stating that 3 additional trams are needed -/
theorem trams_to_add : additional_trams 12 (1/5) = 3 := by
  sorry

#eval additional_trams 12 (1/5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trams_to_add_l62_6284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l62_6293

-- Define the function f(x) = x - 2 log(1/√x) - 3
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log (1 / Real.sqrt x) - 3

-- Theorem statement
theorem solution_exists_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l62_6293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₂_not_convex_l62_6236

open Real

/-- A function is convex on an interval if its second derivative is negative on that interval -/
def IsConvexOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x ∧ deriv (deriv f) x < 0

/-- The four functions to consider -/
noncomputable def f₁ : ℝ → ℝ := fun x ↦ sin x + cos x
noncomputable def f₂ : ℝ → ℝ := fun x ↦ -x * exp (-x)
def f₃ : ℝ → ℝ := fun x ↦ -x^3 + 2*x - 1
noncomputable def f₄ : ℝ → ℝ := fun x ↦ log x - 2*x

/-- The theorem stating that only f₂ is not convex on (0, π/2) -/
theorem only_f₂_not_convex :
  IsConvexOn f₁ 0 (π/2) ∧
  ¬IsConvexOn f₂ 0 (π/2) ∧
  IsConvexOn f₃ 0 (π/2) ∧
  IsConvexOn f₄ 0 (π/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₂_not_convex_l62_6236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l62_6230

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + a

theorem function_properties (a : ℝ) :
  f a (Real.pi / 6) = 1 →
  (a = -1/2) ∧
  (∀ x : ℝ, f a (x + Real.pi) = f a x) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ Real.pi / 2 → f a x ≥ -1/2) ∧
  (f a (Real.pi / 2) = -1/2) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l62_6230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l62_6206

/-- An isosceles triangle with specific properties -/
structure SpecialTriangle where
  -- The triangle is isosceles
  isIsosceles : Bool
  -- The altitude to the base
  altitude : ℝ
  -- The perimeter
  perimeter : ℝ
  -- One of the angles in degrees
  oneAngle : ℝ

/-- The area of the special triangle -/
noncomputable def triangleArea (t : SpecialTriangle) : ℝ :=
  (100 * Real.sqrt 3) / 3

/-- Theorem stating the area of the special triangle -/
theorem special_triangle_area (t : SpecialTriangle) 
  (h1 : t.isIsosceles = true)
  (h2 : t.altitude = 10)
  (h3 : t.perimeter = 40)
  (h4 : t.oneAngle = 60) :
  triangleArea t = (100 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l62_6206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_equation_l62_6276

theorem no_function_satisfies_equation :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) - x^2 + x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_function_satisfies_equation_l62_6276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_halves_l62_6256

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / x

-- State the theorem
theorem inverse_f_at_three_halves :
  ∃ (f_inv : ℝ → ℝ), Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f ∧ f_inv (3/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_three_halves_l62_6256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_l62_6283

/-- The ellipse C defined by x^2/5 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- The foci of the ellipse C -/
noncomputable def F₁ : ℝ × ℝ := (2, 0)
noncomputable def F₂ : ℝ × ℝ := (-2, 0)

/-- A point P on the ellipse C -/
noncomputable def P : ℝ × ℝ := sorry

/-- The dot product of vectors PF₁ and PF₂ is zero -/
axiom vectors_orthogonal : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

/-- P is on the ellipse C -/
axiom P_on_C : P ∈ C

/-- The theorem to be proved -/
theorem product_of_distances :
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) *
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_l62_6283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_specific_l62_6252

/-- The volume of a regular hexagonal pyramid -/
noncomputable def regular_hexagonal_pyramid_volume (base_edge : ℝ) (side_edge : ℝ) : ℝ :=
  let base_area := 3 * base_edge^2 * Real.sqrt 3 / 2
  let height := Real.sqrt (side_edge^2 - base_edge^2)
  (1/3) * base_area * height

/-- Theorem: The volume of a regular hexagonal pyramid with base edge length 3 and side edge length 5 is 18√3 -/
theorem regular_hexagonal_pyramid_volume_specific : 
  regular_hexagonal_pyramid_volume 3 5 = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagonal_pyramid_volume_specific_l62_6252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l62_6287

/-- In an acute triangle ABC, side lengths opposite to angles A, B, C are a, b, c respectively. -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
  opposite : True  -- This is a placeholder for the relationship between sides and angles

/-- The theorem stating the range of b in the given acute triangle -/
theorem b_range (t : AcuteTriangle) (h1 : t.a = 1) (h2 : t.B = 2 * t.A) :
  t.b > Real.sqrt 2 ∧ t.b < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l62_6287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_ratio_l62_6255

/-- Prove the ratio of half-maximum donors to maximum donors --/
theorem donation_ratio :
  let max_donation : ℚ := 1200
  let max_donors : ℕ := 500
  let total_raised : ℚ := 3750000
  let donation_percentage : ℚ := 40 / 100
  let half_max_donation : ℚ := max_donation / 2
  let max_donation_total : ℚ := max_donation * max_donors
  let donation_sum : ℚ := total_raised * donation_percentage
  let half_max_total : ℚ := donation_sum - max_donation_total
  let half_max_donors : ℚ := half_max_total / half_max_donation
  half_max_donors / max_donors = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_ratio_l62_6255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_card_difference_l62_6281

def chris_cards : ℕ := 18
def charlie_cards : ℕ := 32
def diana_cards : ℕ := 25
def ethan_cards : ℕ := 40

theorem total_card_difference : 
  (charlie_cards - chris_cards) + (diana_cards - chris_cards) + (ethan_cards - chris_cards) = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_card_difference_l62_6281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_comparison_l62_6274

theorem exponential_comparison (b : ℝ) (h1 : 0 < b) (h2 : b < 1) :
  b^((-0.1 : ℝ)) > b^((0.1 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_comparison_l62_6274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_expenditure_l62_6285

/-- Given a person's income and expenditure patterns, calculate their house rent expenditure -/
theorem house_rent_expenditure (income : ℝ) (petrol_percentage : ℝ) (house_rent_percentage : ℝ)
  (petrol_expenditure : ℝ) : 
  petrol_percentage = 0.30 →
  house_rent_percentage = 0.20 →
  petrol_expenditure = 300 →
  petrol_expenditure = petrol_percentage * income →
  house_rent_percentage * (income - petrol_expenditure) = 140 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_expenditure_l62_6285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l62_6299

def is_prime (n : ℕ) : Prop := Nat.Prime n

def f_conditions (f : ℕ → ℕ) : Prop :=
  (f 1 = 1) ∧
  (∀ p : ℕ, is_prime p → f p = 1 + f (p - 1)) ∧
  (∀ u : ℕ, ∀ p : List ℕ, (∀ pi ∈ p, is_prime pi) →
    f (p.prod) = (p.map f).sum)

theorem f_bounds (f : ℕ → ℕ) (h : f_conditions f) :
  ∀ n : ℕ, n ≥ 2 → 2^(f n) ≤ n^3 ∧ n^3 ≤ 3^(f n) := by
  sorry

#check f_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l62_6299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kedlaya_fractional_parts_l62_6245

/-- Fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem kedlaya_fractional_parts (p s : ℕ) (hp : Nat.Prime p) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧
    frac ((s * m : ℝ) / p) < frac ((s * n : ℝ) / p) ∧ frac ((s * n : ℝ) / p) < (s : ℝ) / p) ↔
  ¬(s ∣ (p - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kedlaya_fractional_parts_l62_6245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_token_game_ends_in_13_rounds_l62_6208

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : Fin 4 → Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game is over (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Theorem: The token distribution game ends after exactly 13 rounds -/
theorem token_game_ends_in_13_rounds :
  let initialState : GameState := {
    players := λ i => match i with
      | 0 => ⟨18⟩  -- Player A
      | 1 => ⟨16⟩  -- Player B
      | 2 => ⟨17⟩  -- Player C
      | 3 => ⟨15⟩  -- Player D
    rounds := 0
  }
  ∃ (finalState : GameState),
    finalState.rounds = 13 ∧
    isGameOver finalState ∧
    (∀ n < 13, ¬isGameOver (Nat.iterate playRound n initialState)) := by
  sorry

#check token_game_ends_in_13_rounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_token_game_ends_in_13_rounds_l62_6208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_after_volume_change_l62_6241

/-- Represents a right cone -/
structure RightCone where
  circumference : ℝ
  height : ℝ

/-- Calculate the volume of a right cone -/
noncomputable def volume (cone : RightCone) : ℝ :=
  (1/3) * Real.pi * (cone.circumference / (2 * Real.pi))^2 * cone.height

/-- Theorem stating the ratio of new height to original height -/
theorem height_ratio_after_volume_change (original : RightCone) (new_height : ℝ) :
  original.circumference = 20 * Real.pi ∧
  original.height = 40 ∧
  volume { circumference := original.circumference, height := new_height } = 400 * Real.pi →
  new_height / original.height = 3 / 10 := by
  sorry

#check height_ratio_after_volume_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_ratio_after_volume_change_l62_6241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_difference_l62_6247

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits -/
def digit_size : ℕ := 10

/-- The number of letters in an Ohio license plate -/
def ohio_letters : ℕ := 4

/-- The number of digits in an Ohio license plate -/
def ohio_digits : ℕ := 3

/-- The number of letters in a Montana license plate -/
def montana_letters : ℕ := 5

/-- The number of digits in a Montana license plate -/
def montana_digits : ℕ := 2

/-- The difference between the number of possible Montana license plates and Ohio license plates -/
theorem license_plate_difference : 
  (alphabet_size ^ montana_letters * digit_size ^ montana_digits) - 
  (alphabet_size ^ ohio_letters * digit_size ^ ohio_digits) = 731161600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_difference_l62_6247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_3k_is_imaginary_l62_6205

/-- The complex number z raised to the power 3k is purely imaginary for any positive integer k -/
theorem z_power_3k_is_imaginary (k : ℕ+) : 
  let z : ℂ := (3 / (3/2 + (Real.sqrt 3 / 2) * Complex.I))^(3 * k.val)
  ∃ (y : ℝ), z = y * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_3k_is_imaginary_l62_6205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_angle_l62_6214

/-- The number of degrees at each point of a star formed by extending the sides of a regular polygon -/
noncomputable def star_point_angle (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- Theorem stating that for a regular polygon with n sides (n > 4), when its sides are extended to form a star,
    the number of degrees at each point of the star is (n-2)180/n -/
theorem star_polygon_angle (n : ℕ) (h : n > 4) :
  star_point_angle n = (n - 2 : ℝ) * 180 / n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_angle_l62_6214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l62_6227

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity (h : Hyperbola) (p : PointOnHyperbola h)
    (f₁_x f₁_y f₂_x f₂_y : ℝ) -- Coordinates of foci F₁ and F₂
    (h_f₁f₂_distance : distance f₁_x f₁_y f₂_x f₂_y = 12)
    (h_pf₂_distance : distance p.x p.y f₂_x f₂_y = 5)
    (h_pf₂_perpendicular : p.y = f₂_y) -- PF₂ ⊥ x-axis
    : eccentricity h = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l62_6227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_trip_gas_cost_l62_6270

/-- Calculates the cost of gas for Carla's trip given the car's efficiency, gas price, and distances. -/
noncomputable def gas_cost_for_trip (efficiency : ℝ) (gas_price : ℝ) (grocery_dist : ℝ) (school_dist : ℝ) (soccer_dist : ℝ) : ℝ :=
  let total_distance := grocery_dist + school_dist + soccer_dist + 2 * soccer_dist
  let gallons_used := total_distance / efficiency
  gallons_used * gas_price

/-- Proves that the cost of gas for Carla's trip is $5.00 given the specified conditions. -/
theorem carla_trip_gas_cost :
  gas_cost_for_trip 25 2.5 8 6 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_trip_gas_cost_l62_6270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l62_6237

/-- The repeating decimal 0.00003769 where 3769 repeats indefinitely -/
noncomputable def x : ℚ := 3769.3769 / 9999

/-- The theorem stating the value of (10^8 - 10^4)(0.00003769) -/
theorem value_of_expression : (10^8 - 10^4 : ℚ) * x = 3765230.6231 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l62_6237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l62_6246

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - focus.2 = m * (x - focus.1)

-- Define two perpendicular lines
def perpendicular_lines (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the intersection points
def intersection_points (A B D E : ℝ × ℝ) (m1 m2 : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola D.1 D.2 ∧ parabola E.1 E.2 ∧
  line_through_focus m1 A.1 A.2 ∧ line_through_focus m1 B.1 B.2 ∧
  line_through_focus m2 D.1 D.2 ∧ line_through_focus m2 E.1 E.2

-- Define the area of the quadrilateral
noncomputable def area_quadrilateral (A B D E : ℝ × ℝ) : ℝ :=
  abs ((A.1 - D.1) * (B.2 - E.2) - (B.1 - E.1) * (A.2 - D.2)) / 2

-- Theorem statement
theorem min_area_quadrilateral :
  ∀ (A B D E : ℝ × ℝ) (m1 m2 : ℝ),
  perpendicular_lines m1 m2 →
  intersection_points A B D E m1 m2 →
  ∃ (min_area : ℝ), min_area = 128 ∧
  ∀ (A' B' D' E' : ℝ × ℝ) (m1' m2' : ℝ),
  perpendicular_lines m1' m2' →
  intersection_points A' B' D' E' m1' m2' →
  area_quadrilateral A' B' D' E' ≥ min_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_quadrilateral_l62_6246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l62_6228

def circle' (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 3

def ellipse' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem max_distance_circle_ellipse :
  ∃ (d : ℝ),
    d = (7 * Real.sqrt 3) / 3 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      circle' x₁ y₁ → ellipse' x₂ y₂ →
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ d ∧
      ∃ (x₁' y₁' x₂' y₂' : ℝ),
        circle' x₁' y₁' ∧ ellipse' x₂' y₂' ∧
        Real.sqrt ((x₁' - x₂')^2 + (y₁' - y₂')^2) = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_ellipse_l62_6228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_mean_inequality_l62_6218

theorem power_mean_inequality (a b : ℝ) (n : ℕ) (ha : a ≥ 0) (hb : b ≥ 0) (hn : n > 0) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2)^n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_mean_inequality_l62_6218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_speed_is_0_62_l62_6221

/-- Represents the travel details of Youngjin's family trip -/
structure TravelDetails where
  total_distance : ℝ
  total_time : ℝ
  bus_time : ℝ
  bus_fuel_efficiency_distance : ℝ
  bus_fuel_efficiency_fuel : ℝ
  bus_fuel_used : ℝ

/-- Calculates the subway travel speed given the travel details -/
noncomputable def subway_speed (travel : TravelDetails) : ℝ :=
  let bus_distance := (travel.bus_fuel_used * travel.bus_fuel_efficiency_distance) / travel.bus_fuel_efficiency_fuel
  let subway_distance := travel.total_distance - bus_distance
  let subway_time := travel.total_time - travel.bus_time
  subway_distance / subway_time

/-- Theorem stating that the subway speed is 0.62 km/min given the specific travel details -/
theorem subway_speed_is_0_62 :
  subway_speed {
    total_distance := 120,
    total_time := 110,
    bus_time := 70,
    bus_fuel_efficiency_distance := 40.8,
    bus_fuel_efficiency_fuel := 6,
    bus_fuel_used := 14
  } = 0.62 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_speed_is_0_62_l62_6221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l62_6290

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1

theorem sequence_fifth_term (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 2 + a 4 + a 6 = 18) :
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l62_6290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l62_6231

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b c d e f : ℝ) : ℝ :=
  2 * Real.sqrt (25.75 - 4.12)

/-- Theorem: The distance between the foci of the ellipse 25x^2 - 100x + 4y^2 + 8y + 1 = 0 is 2√21.63 -/
theorem ellipse_foci_distance :
  distance_between_foci 25 (-100) 4 8 1 0 = 2 * Real.sqrt 21.63 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l62_6231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_exists_l62_6250

/-- Represents a 1×2 rectangle (domino) --/
structure Domino :=
  (x : Fin 5) (y : Fin 199)

/-- Represents the 5×200 grid --/
def Grid := Fin 5 → Fin 200 → Bool

/-- Checks if a row has an odd number of dominoes --/
def rowHasOddDominoes (g : Grid) (row : Fin 5) : Prop :=
  (Finset.filter (λ col => g row col) (Finset.univ : Finset (Fin 200))).card % 2 = 1

/-- Checks if a column has an odd number of dominoes --/
def colHasOddDominoes (g : Grid) (col : Fin 200) : Prop :=
  (Finset.filter (λ row => g row col) (Finset.univ : Finset (Fin 5))).card % 2 = 1

/-- The main theorem statement --/
theorem domino_arrangement_exists : 
  ∃ (arrangement : Finset Domino) (g : Grid),
    arrangement.card = 500 ∧ 
    (∀ row : Fin 5, rowHasOddDominoes g row) ∧
    (∀ col : Fin 200, colHasOddDominoes g col) := by
  sorry

#check domino_arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_arrangement_exists_l62_6250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l62_6262

/-- Conic curve C -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Left focus F₁ -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Right focus F₂ -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Point A -/
noncomputable def A : ℝ × ℝ := (0, -Real.sqrt 3)

/-- Line passing through F₁ and parallel to AF₂ -/
noncomputable def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 + 1)}

/-- Intersection points of L and C -/
def intersectionPoints : Set (ℝ × ℝ) :=
  C ∩ L

theorem intersection_product :
  ∃ (M N : ℝ × ℝ), M ∈ intersectionPoints ∧ N ∈ intersectionPoints ∧
    M ≠ N ∧ Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) *
           Real.sqrt ((N.1 - F₁.1)^2 + (N.2 - F₁.2)^2) = 12/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l62_6262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_BC_distribution_max_profit_l62_6238

/-- Represents the characteristics of a vegetable type -/
structure VegetableType where
  tons_per_truck : ℝ
  profit_per_ton : ℝ

/-- The problem setup -/
structure VegetableProblem where
  type_A : VegetableType
  type_B : VegetableType
  type_C : VegetableType
  total_trucks : ℕ
  total_tons : ℝ
  min_A_trucks : ℕ
  max_A_trucks : ℕ

def problem : VegetableProblem :=
  { type_A := ⟨2, 5⟩
  , type_B := ⟨1, 7⟩
  , type_C := ⟨2.5, 4⟩
  , total_trucks := 30
  , total_tons := 48
  , min_A_trucks := 1
  , max_A_trucks := 10 }

/-- Part 1: Optimal distribution of trucks for B and C -/
theorem optimal_BC_distribution :
  ∃ (b c : ℕ), b + c = 14 ∧ 
    b * problem.type_B.tons_per_truck + c * problem.type_C.tons_per_truck = 17 ∧
    b = 12 ∧ c = 2 := by sorry

/-- Part 2: Maximum profit calculation -/
theorem max_profit :
  ∃ (a b c : ℕ), a + b + c = problem.total_trucks ∧
    a * problem.type_A.tons_per_truck + b * problem.type_B.tons_per_truck + c * problem.type_C.tons_per_truck = problem.total_tons ∧
    problem.min_A_trucks ≤ a ∧ a ≤ problem.max_A_trucks ∧
    ∀ (a' b' c' : ℕ), a' + b' + c' = problem.total_trucks →
      a' * problem.type_A.tons_per_truck + b' * problem.type_B.tons_per_truck + c' * problem.type_C.tons_per_truck = problem.total_tons →
      problem.min_A_trucks ≤ a' ∧ a' ≤ problem.max_A_trucks →
      (a * problem.type_A.tons_per_truck * problem.type_A.profit_per_ton +
       b * problem.type_B.tons_per_truck * problem.type_B.profit_per_ton +
       c * problem.type_C.tons_per_truck * problem.type_C.profit_per_ton) ≥
      (a' * problem.type_A.tons_per_truck * problem.type_A.profit_per_ton +
       b' * problem.type_B.tons_per_truck * problem.type_B.profit_per_ton +
       c' * problem.type_C.tons_per_truck * problem.type_C.profit_per_ton) ∧
    (a * problem.type_A.tons_per_truck * problem.type_A.profit_per_ton +
     b * problem.type_B.tons_per_truck * problem.type_B.profit_per_ton +
     c * problem.type_C.tons_per_truck * problem.type_C.profit_per_ton) = 255 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_BC_distribution_max_profit_l62_6238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_range_l62_6234

noncomputable def P : ℝ × ℝ := (1/2, Real.sqrt 3/2)

noncomputable def rotate (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

noncomputable def f (x : ℝ) : ℝ := P.1 * (rotate (Real.pi/3 + x) P).1 + P.2 * (rotate (Real.pi/3 + x) P).2

noncomputable def g (x : ℝ) : ℝ := f x * f (x + Real.pi/3)

theorem rotation_and_range :
  (rotate (Real.pi/4) P = ((Real.sqrt 2 - Real.sqrt 6)/4, (Real.sqrt 6 + Real.sqrt 2)/4)) ∧
  (∀ x, -1/4 ≤ g x ∧ g x ≤ 3/4) ∧
  (∃ x y, g x = -1/4 ∧ g y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_and_range_l62_6234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_statements_set1_compound_statements_set2_l62_6282

-- Definitions for the first set of propositions
def has_real_roots (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 0

def equal_roots (f : ℝ → ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ |x| = |y|

-- Definitions for the second set of propositions
def isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c

def equal_base_angles (a b c : ℝ) : Prop := isosceles_triangle a b c → 
  ∃ α β γ : ℝ, α + β + γ = Real.pi ∧ ((a = b ∧ α = β) ∨ (b = c ∧ β = γ) ∨ (a = c ∧ α = γ))

def acute_triangle (a b c : ℝ) : Prop := 
  ∃ α β γ : ℝ, α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2

-- Theorem statements
theorem compound_statements_set1 :
  let p := has_real_roots (λ x => x^2 + 1)
  let q := equal_roots (λ x => x^2 - 1)
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p := by sorry

theorem compound_statements_set2 :
  ∀ a b c : ℝ,
  let p := equal_base_angles a b c
  let q := acute_triangle a b c
  (p ∨ q) ∧ ¬(p ∧ q) ∧ p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_statements_set1_compound_statements_set2_l62_6282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_genuine_certain_at_least_one_genuine_event_certain_l62_6248

-- Define the total number of products
def total_products : ℕ := 12

-- Define the number of genuine products
def genuine_products : ℕ := 10

-- Define the number of defective products
def defective_products : ℕ := 2

-- Define the number of products selected
def selected_products : ℕ := 3

-- Define a function to calculate the probability of selecting all defective products
noncomputable def prob_all_defective : ℚ :=
  (defective_products.choose selected_products : ℚ) / (total_products.choose selected_products : ℚ)

-- Theorem: The probability of selecting at least one genuine product is 1
theorem at_least_one_genuine_certain :
  1 - prob_all_defective = 1 := by
  sorry

-- Theorem: The event "At least 1 is genuine" is certain to happen
theorem at_least_one_genuine_event_certain :
  prob_all_defective = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_genuine_certain_at_least_one_genuine_event_certain_l62_6248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l62_6260

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 1}

-- Define the line l: x + y = 1
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1}

-- Define the point P
def point_P : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem circle_and_tangent_lines :
  -- Given conditions
  (∀ p ∈ circle_C, (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 1) →
  (∃ p q, p ∈ line_l ∩ circle_C ∧ q ∈ line_l ∩ circle_C ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 2) →
  -- Conclusions
  (∀ p ∈ circle_C, (p.1 - 1)^2 + (p.2 - 1)^2 = 1) ∧
  (∀ p ∈ circle_C, p.1 = 2 ∨ 3 * p.1 - 4 * p.2 + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l62_6260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bd_length_l62_6259

noncomputable section

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = x^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = y^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = z^2

-- Define a right angle
def RightAngle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the length of a line segment
noncomputable def SegmentLength (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem triangle_bd_length 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_right_angle : RightAngle A C B)
  (h_ac_length : SegmentLength A C = 9)
  (h_bc_length : SegmentLength B C = 12)
  (h_d_on_ab : PointOnSegment D A B)
  (h_e_on_bc : PointOnSegment E B C)
  (h_bed_right : RightAngle B E D)
  (h_de_length : SegmentLength D E = 6) :
  SegmentLength B D = 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bd_length_l62_6259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l62_6207

/-- The complex number in cis form -/
noncomputable def z : ℂ := (4 * Complex.exp (Complex.I * Real.pi * 25 / 180)) * (-3 * Complex.exp (Complex.I * Real.pi * 48 / 180))

/-- The magnitude of the result -/
noncomputable def r : ℝ := 12

/-- The angle of the result in radians -/
noncomputable def θ : ℝ := 253 * Real.pi / 180

/-- Theorem stating that the complex number z is equal to r * cis(θ) -/
theorem complex_multiplication :
  z = r * Complex.exp (Complex.I * θ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_l62_6207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_remainder_one_l62_6213

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, i + 3 < arr.length →
    (arr[i]! % 3 ≠ arr[i + 2]! % 3) ∧
    (arr[i]! % 3 ≠ arr[i + 3]! % 3) ∧
    (arr[i + 2]! % 3 ≠ arr[i + 3]! % 3)

theorem first_number_remainder_one
  (arr : List Nat)
  (h1 : arr.length = 2023)
  (h2 : ∀ n, n ∈ arr ↔ 1 ≤ n ∧ n ≤ 2023)
  (h3 : is_valid_arrangement arr) :
  arr[0]! % 3 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_remainder_one_l62_6213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometricSequenceRatio_l62_6292

/-- Geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ := λ n ↦ a * q^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometricSum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometricSequenceRatio (a : ℝ) (q : ℝ) (h_a : a ≠ 0) :
  (geometricSum a q 3 + 3 * geometricSum a q 2 = 0) → q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometricSequenceRatio_l62_6292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_and_sean_money_l62_6202

theorem rick_and_sean_money (fritz_money : ℕ) (h1 : fritz_money = 40) : 
  let sean_money := fritz_money / 2 + 4
  let rick_money := sean_money * 3
  sean_money + rick_money = 96 := by
  -- Introduce the local definitions
  let sean_money := fritz_money / 2 + 4
  let rick_money := sean_money * 3

  -- State the goal
  have goal : sean_money + rick_money = 96

  -- Proof steps would go here
  sorry

  -- Return the goal
  exact goal


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rick_and_sean_money_l62_6202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_minimum_a_l62_6235

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

theorem tangent_and_minimum_a :
  (∃ (m b : ℝ), ∀ x, m * x + b = (deriv f 1) * (x - 1) + f 1 ∧ m = -15 ∧ b = 1) ∧
  (∀ a : ℤ, (∀ x : ℝ, x > 0 → f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1) →
    a ≥ 1) ∧
  (∃ a : ℤ, a = 1 ∧ ∀ x : ℝ, x > 0 → f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_minimum_a_l62_6235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_prism_lateral_area_l62_6275

/-- Given a right prism with a rhombus base, this theorem proves that
    the lateral surface area is 160 cm² when the diagonals of the rhombus
    are 9 cm and 15 cm, and the height of the prism is 5 cm. -/
theorem rhombus_prism_lateral_area :
  ∀ (diagonal1 diagonal2 height : ℝ),
    diagonal1 = 9 →
    diagonal2 = 15 →
    height = 5 →
    4 * Real.sqrt ((diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2) * height = 160 := by
  sorry

#check rhombus_prism_lateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_prism_lateral_area_l62_6275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_120_degrees_l62_6268

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given equation
def given_equation (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 2 * t.a * t.b

-- Define the angle opposite side c in radians
noncomputable def angle_C (t : Triangle) : ℝ :=
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- Theorem statement
theorem angle_C_is_120_degrees (t : Triangle) (h : given_equation t) :
  angle_C t = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_120_degrees_l62_6268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_fractional_part_bounds_l62_6265

theorem sqrt_fractional_part_bounds (n : ℕ) (h : n > 100) :
  ∃ (x : ℝ), x = Real.sqrt ((n : ℝ)^2 + 3*(n : ℝ) + 1) - (n : ℝ) - 1 ∧ 0.49 < x ∧ x < 0.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_fractional_part_bounds_l62_6265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guessing_strategy_exists_l62_6297

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_correct_guess (target guess : ℕ) : Prop :=
  is_two_digit target ∧ is_two_digit guess ∧
  ((target / 10 = guess / 10 ∧ Int.natAbs (target % 10 - guess % 10) ≤ 1) ∨
   (target % 10 = guess % 10 ∧ Int.natAbs (target / 10 - guess / 10) ≤ 1))

def covers_all_numbers (guesses : List ℕ) : Prop :=
  ∀ n, is_two_digit n → ∃ g ∈ guesses, is_correct_guess n g

theorem guessing_strategy_exists :
  ∃ guesses : List ℕ, guesses.length = 22 ∧ covers_all_numbers guesses := by
  sorry

#check guessing_strategy_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guessing_strategy_exists_l62_6297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l62_6240

theorem determinant_transformation (x y z w : ℝ) : 
  Matrix.det !![x, y; z, w] = 6 → Matrix.det !![x, 5*x + 4*y; z, 5*z + 4*w] = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_transformation_l62_6240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_pond_estimation_l62_6288

theorem fish_pond_estimation : 
  let initial_marked : ℕ := 40
  let second_catch : ℕ := 100
  let marked_in_second : ℕ := 5
  let estimated_total : ℕ := (second_catch * initial_marked) / marked_in_second
  estimated_total = 800 := by
  -- Proof goes here
  sorry

#eval (100 * 40) / 5  -- This will evaluate to 800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_pond_estimation_l62_6288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l62_6294

/-- The standard equation of a circle with center at (-5, 4) and tangent to the x-axis -/
theorem circle_equation : ∃ (C : Set (ℝ × ℝ)), 
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 + 5)^2 + (p.2 - 4)^2 = 16) ∧ 
  (∃ (x : ℝ), (x, 0) ∈ C) ∧
  (∀ (p : ℝ × ℝ), p ∈ C → (p.1 + 5)^2 + (p.2 - 4)^2 = 16) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l62_6294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l62_6298

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 10 * k + 7 ∧ 
  7 * (10 ^ (Nat.log n 10 + 1)) + k = 5 * n

theorem least_valid_number : 
  is_valid_number 142857 ∧ 
  ∀ m : ℕ, m < 142857 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l62_6298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_length_is_6_sqrt_59_l62_6210

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ab_parallel_dc : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  ac_perp_dc : (C.1 - A.1) * (D.1 - C.1) + (C.2 - A.2) * (D.2 - C.2) = 0
  dc_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 18
  tan_c : (C.2 - A.2) / (C.1 - A.1) = 2
  tan_b : (B.2 - A.2) / (B.1 - A.1) = 1.25

/-- The length of BD in the trapezoid -/
noncomputable def bd_length (t : Trapezoid) : ℝ :=
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2)

/-- Theorem stating that BD length is 6√59 -/
theorem bd_length_is_6_sqrt_59 (t : Trapezoid) : bd_length t = 6 * Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_length_is_6_sqrt_59_l62_6210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l62_6200

/-- The value of the infinite nested square root √(18 + √(18 + √(18 + ...))) --/
noncomputable def nested_sqrt : ℝ := 
  Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt 18)))

/-- Theorem: The value of √(18 + √(18 + √(18 + ...))) is 6 --/
theorem nested_sqrt_value : nested_sqrt = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l62_6200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_l62_6217

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def sum_of_digits_range (start : ℕ) (end_ : ℕ) : ℕ :=
  List.sum (List.map sum_of_digits (List.range (end_ - start + 1) |>.map (· + start)))

theorem sum_of_digits_0_to_99 : sum_of_digits_range 0 99 = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_99_l62_6217
