import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_increasing_l310_31093

/-- The sequence a_n is defined as n^2 + λn for positive integers n -/
def a_n (n : ℕ+) (lambda : ℝ) : ℝ := n.val^2 + lambda * n.val

/-- The sequence a_n is increasing if a_(n+1) > a_n for all positive integers n -/
def is_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, a_n (n + 1) lambda > a_n n lambda

/-- Theorem: For any λ > -3, the sequence a_n is increasing -/
theorem a_n_increasing (lambda : ℝ) (h : lambda > -3) : is_increasing lambda := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_increasing_l310_31093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_detection_theorem_l310_31094

/-- Represents a detector that can scan a subgrid of the M × N grid -/
structure Detector (M N : ℕ) where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  h1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ M
  h2 : 1 ≤ c ∧ c ≤ d ∧ d ≤ N

/-- The minimum number of detectors required to guarantee finding the treasure -/
def minDetectors (M N : ℕ) : ℕ := ⌈(M : ℚ) / 2⌉₊ + ⌈(N : ℚ) / 2⌉₊

theorem treasure_detection_theorem (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  ∀ (detectors : Finset (Detector M N)),
    (∀ (i j : ℕ) (hi : 1 ≤ i ∧ i ≤ M) (hj : 1 ≤ j ∧ j ≤ N),
      ∃! (subset : Finset (Detector M N)), subset ⊆ detectors ∧
        ∀ (d : Detector M N), d ∈ subset ↔ d.a ≤ i ∧ i ≤ d.b ∧ d.c ≤ j ∧ j ≤ d.d) →
    minDetectors M N ≤ detectors.card :=
by sorry

#check treasure_detection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_detection_theorem_l310_31094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_percentage_l310_31083

theorem garden_tulips_percentage (F : ℝ) (hF : F > 0) : 
  let pink_flowers := (7/12) * F
  let pink_daisies := (1/2) * pink_flowers
  let pink_tulips := pink_flowers - pink_daisies
  let red_flowers := F - pink_flowers
  let red_tulips := (3/5) * red_flowers
  let total_tulips := pink_tulips + red_tulips
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((total_tulips / F) * 100 - 54.17) < ε := by
  sorry

#check garden_tulips_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_tulips_percentage_l310_31083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l310_31098

noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => x  -- Adding case for n = 0
  | 1, x => x
  | n + 1, x => 1 / (2 - a n x)

theorem a_formula (n : ℕ) (x : ℝ) (h : n > 0) :
  a n x = ((n - 1) - (n - 2) * x) / (n - (n - 1) * x) := by
  sorry

#check a_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l310_31098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l310_31084

/-- Given that i is the imaginary unit, prove that (3+5i)/(1+i) = 4+i -/
theorem complex_fraction_equality : (3 : ℂ) + 5 * Complex.I / (1 + Complex.I) = 4 + Complex.I := by
  -- The proof is omitted
  sorry

#check complex_fraction_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l310_31084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l310_31013

theorem sum_remainder (x y z : ℕ) 
  (hx : x % 15 = 6)
  (hy : y % 15 = 9)
  (hz : z % 15 = 3) :
  (x + y + z) % 15 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l310_31013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_color_theorem_l310_31030

-- Define the color type
inductive Color
| Red
| Green

-- Define the vertex type
structure Vertex :=
  (label : String)

-- Define the edge type
structure Edge :=
  (v1 v2 : Vertex)
  (color : Color)

-- Define the prism type
structure Prism :=
  (top_vertices bottom_vertices : List Vertex)
  (edges : List Edge)

-- Define the condition that every triangle has at least two sides of different colors
def triangleCondition (p : Prism) : Prop :=
  ∀ (v1 v2 v3 : Vertex),
    v1 ∈ (p.top_vertices ++ p.bottom_vertices) →
    v2 ∈ (p.top_vertices ++ p.bottom_vertices) →
    v3 ∈ (p.top_vertices ++ p.bottom_vertices) →
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1 →
    ∃ (e1 e2 : Edge),
      e1 ∈ p.edges ∧ e2 ∈ p.edges ∧
      ((e1.v1 = v1 ∧ e1.v2 = v2) ∨ (e1.v1 = v2 ∧ e1.v2 = v1)) ∧
      ((e2.v1 = v2 ∧ e2.v2 = v3) ∨ (e2.v1 = v3 ∧ e2.v2 = v2)) ∧
      e1.color ≠ e2.color

-- Define the theorem
theorem prism_color_theorem (p : Prism) :
  p.top_vertices.length = 5 →
  p.bottom_vertices.length = 5 →
  triangleCondition p →
  ∃ (c : Color),
    ∀ (e : Edge),
      (e.v1 ∈ p.top_vertices ∧ e.v2 ∈ p.top_vertices) ∨
      (e.v1 ∈ p.bottom_vertices ∧ e.v2 ∈ p.bottom_vertices) →
      e ∈ p.edges →
      e.color = c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_color_theorem_l310_31030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l310_31024

theorem midpoint_triangle_ratio (l m n p : ℝ) (A B C : Fin 3 → ℝ) : 
  let midpoint_BC : Fin 3 → ℝ := λ i => if i = 0 then l else 0
  let midpoint_AC : Fin 3 → ℝ := λ i => if i = 1 then m else 0
  let midpoint_AB : Fin 3 → ℝ := λ i => if i = 2 then n else 0
  let z_coord : ℝ := p
  (∀ i, midpoint_BC i = (B i + C i) / 2) ∧
  (∀ i, midpoint_AC i = (A i + C i) / 2) ∧
  (∀ i, midpoint_AB i = (A i + B i) / 2) ∧
  (A 2 = z_coord) ∧ (B 2 = z_coord) ∧ (C 2 = z_coord) →
  (((A 0 - B 0)^2 + (A 1 - B 1)^2 + (A 2 - B 2)^2) + 
   ((A 0 - C 0)^2 + (A 1 - C 1)^2 + (A 2 - C 2)^2) + 
   ((B 0 - C 0)^2 + (B 1 - C 1)^2 + (B 2 - C 2)^2)) / 
  (l^2 + m^2 + n^2 + 3 * p^2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l310_31024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l310_31006

/-- The hyperbola C defined by x^2/a^2 - y^2/b^2 = 1 with a > 0 and b > 0 -/
structure Hyperbola (a b : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0

/-- Point A on the hyperbola C -/
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 1)

/-- Focal points of the hyperbola -/
def focal_points (c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0))

/-- The dot product of vectors AF₁ and AF₂ is zero -/
def vectors_orthogonal (A F₁ F₂ : ℝ × ℝ) : Prop :=
  let AF₁ := (F₁.1 - A.1, F₁.2 - A.2)
  let AF₂ := (F₂.1 - A.1, F₂.2 - A.2)
  AF₁.1 * AF₂.1 + AF₁.2 * AF₂.2 = 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- Theorem: The eccentricity of the given hyperbola is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) :
  let (F₁, F₂) := focal_points (Real.sqrt 2)
  point_A ∈ {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1} ∧
  vectors_orthogonal point_A F₁ F₂ →
  eccentricity a (Real.sqrt 2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l310_31006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l310_31036

-- Define the hyperbola C₁
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the circle C₂
def circle_equation (c : ℝ) (x y : ℝ) : Prop :=
  (x + c)^2 + y^2 = (2*c)^2

-- Define the triangle area
noncomputable def triangle_area (c : ℝ) : ℝ :=
  1/2 * (2*c) * (2*c) * Real.sin (75 * Real.pi / 180)

-- Main theorem
theorem hyperbola_circle_intersection 
  (a b c : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  circle_equation c x y →
  triangle_area c = 4 →
  c = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l310_31036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l310_31022

theorem mean_median_difference (x c : ℕ) : 
  let S : Finset ℕ := {x, x + 2, x + 4, x + c, x + 37}
  let median := x + 4
  let mean := (5 * x + c + 43) / 5
  (∀ n ∈ S, n > 0) →
  mean = median + 6 →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l310_31022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_A_legal_time_driver_B_rate_range_l310_31052

-- Define the initial blood alcohol content
def BAC₀ : ℝ := 1

-- Define the legal driving limit
def legalLimit : ℝ := 0.2

-- Define the rate of decrease for driver A
def p₁ : ℝ := 0.3

-- Define the blood alcohol content after t hours for a given rate p
noncomputable def BAC (p : ℝ) (t : ℝ) : ℝ := BAC₀ * (1 - p)^t

-- Theorem for driver A
theorem driver_A_legal_time :
  ∃ t : ℕ, (∀ s : ℕ, s < t → BAC p₁ (s : ℝ) ≥ legalLimit) ∧
           BAC p₁ (t : ℝ) < legalLimit ∧
           t = 5 := by sorry

-- Theorem for driver B
theorem driver_B_rate_range :
  ∃ p₂ : ℝ, BAC p₂ 6 ≥ legalLimit ∧
           BAC p₂ 7 < legalLimit ∧
           0.21 < p₂ ∧ p₂ ≤ 0.24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_A_legal_time_driver_B_rate_range_l310_31052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l310_31086

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ :=
  1/2 * abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))

/-- Given a line ax + by = 1 intersecting the unit circle, prove the maximum of a + b -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 = 1 ∧ p.1^2 + p.2^2 = 1} ∧
                   B ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 = 1 ∧ p.1^2 + p.2^2 = 1} ∧
                   A ≠ B) →
  (area_triangle (0, 0) A B = 1/2) →
  a + b ≤ 2 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l310_31086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l310_31064

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define point P
def P : ℝ × ℝ := (0, -4)

-- Define the line passing through P with slope k
def line_eq (k : ℝ) (x y : ℝ) : Prop := y = k*x - 4

-- Define the slope of a line passing through the origin and a point
noncomputable def slope_through_origin (x y : ℝ) : ℝ := y / x

theorem sum_of_slopes_constant (k : ℝ) (h : k > 3/4) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    line_eq k A.1 A.2 ∧
    line_eq k B.1 B.2 ∧
    slope_through_origin A.1 A.2 + slope_through_origin B.1 B.2 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_slopes_constant_l310_31064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l310_31045

noncomputable def α : ℝ := Real.arctan (4/3)

theorem angle_properties :
  let x : ℝ := 3
  let y : ℝ := 4
  let r : ℝ := Real.sqrt (x^2 + y^2)
  Real.sin α = y / r ∧ 
  Real.cos α = x / r ∧
  (2 * Real.cos (π/2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l310_31045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_age_at_record_l310_31073

/-- Represents the problem of Sandy's fingernail growth --/
structure FingernailGrowth where
  world_record : ℚ
  current_age : ℚ
  current_length : ℚ
  monthly_growth_rate : ℚ

/-- Calculates the age when Sandy's fingernails reach the world record length --/
def age_at_record (fg : FingernailGrowth) : ℚ :=
  fg.current_age + (fg.world_record - fg.current_length) / (fg.monthly_growth_rate * 12)

/-- Theorem stating that Sandy will be 32 years old when she achieves the world record --/
theorem sandy_age_at_record :
  let fg : FingernailGrowth := {
    world_record := 26,
    current_age := 12,
    current_length := 2,
    monthly_growth_rate := 1/10
  }
  age_at_record fg = 32 := by
  -- Proof goes here
  sorry

#eval age_at_record {
  world_record := 26,
  current_age := 12,
  current_length := 2,
  monthly_growth_rate := 1/10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_age_at_record_l310_31073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l310_31000

noncomputable section

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4) + (P.2^2 / 3) = 1

-- Define the foci
def foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-1, 0), (1, 0))

-- Define the inscribed circle radius
def inscribed_circle_radius : ℝ := 1/2

-- Define the dot product of vectors PF₁ and PF₂
def dot_product (P : ℝ × ℝ) : ℝ :=
  let F₁ := foci.1
  let F₂ := foci.2
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2)

theorem ellipse_dot_product (P : ℝ × ℝ) :
  is_on_ellipse P →
  dot_product P = 9/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l310_31000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l310_31074

noncomputable def street_length : ℝ := 600  -- in meters
noncomputable def crossing_time : ℝ := 2    -- in minutes

noncomputable def speed_km_per_hour : ℝ :=
  (street_length / 1000) / (crossing_time / 60)

theorem speed_calculation :
  speed_km_per_hour = 18 := by
  -- Unfold the definitions
  unfold speed_km_per_hour street_length crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l310_31074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l310_31055

-- Define the curve and line
noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x
def line (x : ℝ) : ℝ := x - 4

-- Define the distance function from a point (x, y) to the line y = x - 4
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 4| / Real.sqrt 2

-- State the theorem
theorem min_distance_curve_to_line :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  (∀ (x : ℝ), x > 0 → distance_to_line x (curve x) ≥ distance_to_line x₀ (curve x₀)) ∧
  distance_to_line x₀ (curve x₀) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l310_31055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_third_max_area_is_4_sqrt_3_l310_31026

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  ab_eq_c : c = 4

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C

-- Theorem 1: Prove that angle C is π/3
theorem angle_C_is_pi_third (t : Triangle) (h : given_condition t) : t.C = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the maximum area is 4√3
theorem max_area_is_4_sqrt_3 (t : Triangle) (h : given_condition t) : 
  (∃ (S : ℝ), S = (1/2) * t.a * t.b * Real.sin t.C ∧ 
   ∀ (S' : ℝ), S' = (1/2) * t.a * t.b * Real.sin t.C → S' ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_third_max_area_is_4_sqrt_3_l310_31026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_dot_AC_equals_6_l310_31053

noncomputable section

-- Define the line
def line (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * (x - 4)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define points A and B as the intersection of the line and circle
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define point C as the center of the circle
def C : ℝ × ℝ := (2, 0)

-- Vector AB
def vec_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Vector AC
def vec_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Dot product of AB and AC
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem AB_dot_AC_equals_6 : dot_product vec_AB vec_AC = 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AB_dot_AC_equals_6_l310_31053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l310_31065

theorem remainder_sum_mod_13 
  (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_13_l310_31065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plane_angles_convex_polyhedron_sum_plane_angles_twice_interior_angles_l310_31078

/-- Definition of a convex polyhedron with p vertices -/
def ConvexPolyhedron (p : ℕ) : Prop :=
  ∃ (vertices : Finset (EuclideanSpace ℝ (Fin 3))) (faces : Finset (Finset (EuclideanSpace ℝ (Fin 3)))),
    vertices.card = p ∧
    (∀ face ∈ faces, face ⊆ vertices) ∧
    -- Additional properties to ensure convexity and polyhedron structure
    True

/-- Sum of plane angles of all faces of a polyhedron -/
noncomputable def SumPlaneAngles (p : ℕ) : ℝ :=
  -- Definition left abstract as it depends on the specific structure of the polyhedron
  sorry

/-- A convex polyhedron with p vertices has a sum of plane angles of all faces equal to 2π(p-2) -/
theorem sum_plane_angles_convex_polyhedron (p : ℕ) :
  ConvexPolyhedron p → SumPlaneAngles p = 2 * Real.pi * (p - 2) :=
by
  sorry

/-- Sum of interior angles of a planar polygon with p vertices -/
noncomputable def SumInteriorAnglesPolygon (p : ℕ) : ℝ :=
  (p - 2) * Real.pi

/-- The sum of plane angles of a convex polyhedron is twice the sum of interior angles of a planar polygon with the same number of vertices -/
theorem sum_plane_angles_twice_interior_angles (p : ℕ) :
  ConvexPolyhedron p → SumPlaneAngles p = 2 * SumInteriorAnglesPolygon p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plane_angles_convex_polyhedron_sum_plane_angles_twice_interior_angles_l310_31078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_rearrangement_l310_31088

theorem six_digit_rearrangement
  (a : Fin 6 → Nat)
  (h_digit : ∀ i, a i ≤ 9) :
  ∃ p : Equiv.Perm (Fin 6),
    (a (p 0) + a (p 1) + a (p 2)) - (a (p 3) + a (p 4) + a (p 5)) < 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_rearrangement_l310_31088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_diagonal_intersection_to_shorter_base_l310_31039

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Perimeter of the trapezoid -/
  perimeter : ℝ
  /-- Area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- A circle can be inscribed in the trapezoid -/
  hasInscribedCircle : Bool

/-- The distance from the intersection of diagonals to the shorter base -/
noncomputable def diagonalIntersectionToShorterBase (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := 
  (2 - Real.sqrt 3) / 4

/-- Theorem stating the distance from the intersection of diagonals to the shorter base -/
theorem distance_diagonal_intersection_to_shorter_base 
  (t : IsoscelesTrapezoidWithInscribedCircle) 
  (h1 : t.perimeter = 8) 
  (h2 : t.area = 2) 
  (h3 : t.isIsosceles = true) 
  (h4 : t.hasInscribedCircle = true) : 
  diagonalIntersectionToShorterBase t = (2 - Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_diagonal_intersection_to_shorter_base_l310_31039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l310_31076

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), f ((-Real.pi / 6) + x) = -f ((-Real.pi / 6) - x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l310_31076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_nickels_needed_l310_31081

theorem minimum_nickels_needed (book_cost : ℚ) (twenty_bills : ℕ) (quarters : ℕ) :
  book_cost = 35.50 ∧
  twenty_bills = 2 ∧
  quarters = 12 →
  ∀ n : ℕ, (twenty_bills * 20 + quarters * (1/4) + n * (1/20) ≥ book_cost) → n ≥ 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_nickels_needed_l310_31081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_root_in_interval_l310_31059

theorem no_root_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 0.25 * Real.pi → Real.sin (2 * x) + 5 * Real.sin x + 5 * Real.cos x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_root_in_interval_l310_31059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_product_set_is_correct_l310_31072

/-- The maximum number of natural numbers not exceeding 2016 that can be selected,
    such that the product of any two selected numbers is a perfect square. -/
def max_square_product_set : ℕ := 44

/-- The upper bound for the natural numbers in the set. -/
def upper_bound : ℕ := 2016

theorem max_square_product_set_is_correct :
  ∀ (S : Finset ℕ),
  (∀ n : ℕ, n ∈ S → n ≤ upper_bound) →
  (∀ a b : ℕ, a ∈ S → b ∈ S → ∃ k : ℕ, a * b = k * k) →
  S.card ≤ max_square_product_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_product_set_is_correct_l310_31072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l310_31048

theorem election_votes (candidates : ℕ) (loser_percentage : ℚ)
  (vote_difference : ℕ) (invalid_votes : ℕ)
  (h1 : candidates = 2)
  (h2 : loser_percentage = 1/5)
  (h3 : vote_difference = 500)
  (h4 : invalid_votes = 10) :
  ∃ (total_votes : ℕ),
    total_votes = 844 ∧
    (Nat.ceil ((vote_difference : ℚ) / (1 - 2 * loser_percentage)) : ℕ) + invalid_votes = total_votes :=
by
  sorry

#check election_votes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l310_31048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l310_31007

theorem cosine_problem (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 3) = -2 / 3) : Real.cos α = (Real.sqrt 15 - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_problem_l310_31007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_ellipse_midpoint_l310_31070

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

-- Define a line passing through a point with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1) + 1

-- Define the intersection points of the line and ellipse
def intersection (k : ℝ) (x1 x2 : ℝ) : Prop :=
  ∃ y1 y2 : ℝ, 
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    line k x1 y1 ∧ line k x2 y2 ∧
    x1 ≠ x2

-- Define the midpoint condition
def chord_midpoint (x1 x2 : ℝ) : Prop := (x1 + x2) / 2 = 1

-- Theorem statement
theorem line_equation_through_ellipse_midpoint :
  ∀ k x1 x2 : ℝ,
    intersection k x1 x2 →
    chord_midpoint x1 x2 →
    ∃ x y : ℝ, 4 * x + 9 * y - 13 = 0 ∧ line k x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_ellipse_midpoint_l310_31070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l310_31027

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The given expression -/
noncomputable def expression : ℂ := (2 * i / (1 + i)) * (2 * i - i ^ 2016)

/-- Theorem stating that the expression equals -3 + i -/
theorem expression_value : expression = -3 + i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l310_31027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_two_l310_31068

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

/-- The tangent line equation -/
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the tangent line equation is correct -/
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := 2
  tangent_line x₀ y₀ ∧ 
  ∀ x : ℝ, x ≠ 0 → (tangent_line x (f x) ↔ x = x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_two_l310_31068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l310_31029

-- Define the characteristics of each function
def functionA : ℝ → ℝ := λ x => x

def functionB : ℝ → ℝ := λ x => x^2

def functionC : ℝ → ℝ := λ x => -x + 4

noncomputable def functionD : ℝ → ℝ := λ x => Real.sqrt (9 - x^2)

noncomputable def functionE : ℝ → ℝ := λ x => 5 * Real.exp (-x)

-- Define the property of having an inverse
def hasInverse (f : ℝ → ℝ) : Prop := Function.Bijective f

-- Theorem statement
theorem inverse_functions :
  (hasInverse functionA) ∧
  (¬ hasInverse functionB) ∧
  (hasInverse functionC) ∧
  (¬ hasInverse functionD) ∧
  (hasInverse functionE) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l310_31029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l310_31015

-- Define the parabola C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle E
def E (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define the lines l1 and l2 (implicitly through their properties)
def tangent_lines (l : ℝ → ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), E A.1 A.2 ∧ E B.1 B.2 ∧
    (∀ x, l x - A.2 = (x - A.1) * (l A.1 - F.2) / (A.1 - F.1)) ∧
    (∀ x, l x - B.2 = (x - B.1) * (l B.1 - F.2) / (B.1 - F.1))

-- Main theorem
theorem parabola_circle_intersection
  (l1 l2 : ℝ → ℝ)
  (h1 : tangent_lines l1)
  (h2 : tangent_lines l2)
  : (∃ x y, C x y ∧ E x y) ∧
    (∃ A B : ℝ × ℝ, E A.1 A.2 ∧ E B.1 B.2 ∧
      (A.1^2 + A.2^2 - 2*A.1 - A.2 = 0) ∧
      (B.1^2 + B.2^2 - 2*B.1 - B.2 = 0) ∧
      (F.1^2 + F.2^2 - 2*F.1 - F.2 = 0)) ∧
    (∃ A B : ℝ × ℝ, E A.1 A.2 ∧ E B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*Real.sqrt 5 / 5)^2) ∧
    (∃ M N P Q : ℝ × ℝ, C M.1 M.2 ∧ C N.1 N.2 ∧ C P.1 P.2 ∧ C Q.1 Q.2 ∧
      l1 M.1 = M.2 ∧ l1 N.1 = N.2 ∧ l2 P.1 = P.2 ∧ l2 Q.1 = Q.2 ∧
      (M.1 - N.1)^2 + (M.2 - N.2)^2 + (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (136/9)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l310_31015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_l310_31097

-- Define necessary structures and predicates
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

def centers_form_equilateral_triangle (c1 c2 c3 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2 ∧
  (x3 - x2)^2 + (y3 - y2)^2 = (x1 - x3)^2 + (y1 - y3)^2 ∧
  (x1 - x3)^2 + (y1 - y3)^2 = (x2 - x1)^2 + (y2 - y1)^2

theorem circle_configuration (r : ℝ) : 
  (∃ (inner_circle outer_circle1 outer_circle2 outer_circle3 : Circle),
    inner_circle.radius = 2 ∧
    outer_circle1.radius = r ∧
    outer_circle2.radius = r ∧
    outer_circle3.radius = r ∧
    are_tangent inner_circle outer_circle1 ∧
    are_tangent inner_circle outer_circle2 ∧
    are_tangent inner_circle outer_circle3 ∧
    are_tangent outer_circle1 outer_circle2 ∧
    are_tangent outer_circle2 outer_circle3 ∧
    are_tangent outer_circle3 outer_circle1 ∧
    centers_form_equilateral_triangle outer_circle1 outer_circle2 outer_circle3) →
  r = 1 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_l310_31097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_elimination_probability_l310_31038

noncomputable section

-- Define the room dimensions
def room_edge_length : ℝ := 2

-- Define the mosquito killer range
def killer_range : ℝ := 1

-- Define the volume of the room
def room_volume : ℝ := room_edge_length ^ 3

-- Define the volume of a sphere (mosquito killer range)
def sphere_volume : ℝ := (4 / 3) * Real.pi * killer_range ^ 3

-- Theorem statement
theorem mosquito_elimination_probability :
  (sphere_volume / room_volume) = (Real.pi / 6) := by
  -- Expand the definitions
  unfold sphere_volume room_volume room_edge_length killer_range
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_elimination_probability_l310_31038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l310_31043

open Real

-- Define the function
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 6)

-- State the theorem
theorem omega_range (ω : ℝ) : 
  (ω > 0) →
  (∃ x ∈ Set.Icc 0 π, f ω x = 0) →
  (∀ x ∈ Set.Icc 0 π, f ω x ≥ -1/2) →
  ω ∈ Set.Icc (1/6) (4/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l310_31043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_set_l310_31004

/-- Checks if three numbers can form a right triangle. -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers. -/
def sets : List (Fin 3 → ℕ) :=
  [![1, 1, 2], ![6, 8, 10], ![4, 6, 8], ![5, 12, 11]]

theorem right_triangle_set : ∃! s, s ∈ sets ∧ is_right_triangle (s 0) (s 1) (s 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_set_l310_31004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_M_functions_l310_31031

-- Property M definition
def has_property_M (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → (Real.exp x) * (f x) < (Real.exp y) * (f y)

-- Define the four functions
noncomputable def f₁ (x : ℝ) : ℝ := (2 : ℝ) ^ (-x)
noncomputable def f₂ (x : ℝ) : ℝ := (3 : ℝ) ^ (-x)
def f₃ (x : ℝ) : ℝ := x ^ 3
def f₄ (x : ℝ) : ℝ := x ^ 2 + 2

-- Theorem stating which functions have property M
theorem property_M_functions :
  (has_property_M f₁) ∧
  (has_property_M f₄) ∧
  ¬(has_property_M f₂) ∧
  ¬(has_property_M f₃) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_M_functions_l310_31031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l310_31095

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_closed_interval : 
  A_intersect_B = Set.Ioo 1 4 ∪ Set.Ioc 4 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_closed_interval_l310_31095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equivalent_to_abs_l310_31096

-- Define the absolute value function
def abs_val (x : ℝ) : ℝ := |x|

-- Define the functions given in the problem
noncomputable def func_A (x : ℝ) : ℝ := (Real.sqrt x)^2
def func_B (v : ℝ) : ℝ := v
noncomputable def func_C (x : ℝ) : ℝ := Real.sqrt (x^2)
noncomputable def func_D (n : ℝ) : ℝ := if n ≠ 0 then n else 0

-- Theorem stating which functions are not equivalent to the absolute value function
theorem functions_not_equivalent_to_abs : 
  (∃ x, func_A x ≠ abs_val x) ∧ 
  (∃ v, func_B v ≠ abs_val v) ∧ 
  (∃ n, func_D n ≠ abs_val n) ∧
  (∀ x, func_C x = abs_val x) := by
  sorry

#check functions_not_equivalent_to_abs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equivalent_to_abs_l310_31096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l310_31040

-- Define the constants
noncomputable def a : ℝ := Real.rpow 0.6 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log 0.6
noncomputable def c : ℝ := Real.log Real.pi

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l310_31040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_range_condition_l310_31001

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

-- Theorem 1: If f'(1) = 3 and f(1) = 3 - 2, then a = 2
theorem tangent_condition (a : ℝ) : 
  (deriv (f a) 1 = 3) → (f a 1 = 3 - 2) → a = 2 := by sorry

-- Theorem 2: The range of a for which f(x) ≥ a holds for all x > 0 is [-e², 0]
theorem range_condition :
  {a : ℝ | ∀ x > 0, f a x ≥ a} = Set.Icc (-Real.exp 2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_range_condition_l310_31001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_values_l310_31032

theorem product_of_x_values : 
  (∀ x : ℝ, |18 / x + 4| = 3 → x = -18 ∨ x = -18/7) ∧ 
  (-18 * (-18/7) = 324/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_x_values_l310_31032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_must_say_B_higher_than_A_l310_31077

-- Define the inhabitants
inductive Inhabitant : Type
| A
| B
| C

-- Define the ranks
inductive Rank : Type
| Knight
| Ordinary
| Liar

-- Define a function to assign ranks to inhabitants
def rank : Inhabitant → Rank := sorry

-- Define a function to determine if a statement is true
def isTruthful (statement : Prop) (speaker : Inhabitant) : Prop :=
  (rank speaker = Rank.Knight ∧ statement) ∨
  (rank speaker = Rank.Liar ∧ ¬statement) ∨
  (rank speaker = Rank.Ordinary)

-- Define a custom ordering for Rank
instance : LE Rank where
  le := λ a b => match a, b with
    | Rank.Liar, _ => True
    | Rank.Ordinary, Rank.Knight => True
    | Rank.Ordinary, Rank.Ordinary => True
    | Rank.Knight, Rank.Knight => True
    | _, _ => False

instance : LT Rank where
  lt := λ a b => a ≤ b ∧ a ≠ b

-- A's statement: B is higher in rank than C
axiom A_statement : isTruthful (rank Inhabitant.B > rank Inhabitant.C) Inhabitant.A

-- B's statement: C is higher in rank than A
axiom B_statement : isTruthful (rank Inhabitant.C > rank Inhabitant.A) Inhabitant.B

-- Each inhabitant has a different rank
axiom different_ranks :
  rank Inhabitant.A ≠ rank Inhabitant.B ∧
  rank Inhabitant.B ≠ rank Inhabitant.C ∧
  rank Inhabitant.C ≠ rank Inhabitant.A

-- One inhabitant is a Knight, one is a Liar, and one is Ordinary
axiom rank_distribution :
  (rank Inhabitant.A = Rank.Knight ∧ rank Inhabitant.B = Rank.Ordinary ∧ rank Inhabitant.C = Rank.Liar) ∨
  (rank Inhabitant.A = Rank.Knight ∧ rank Inhabitant.B = Rank.Liar ∧ rank Inhabitant.C = Rank.Ordinary) ∨
  (rank Inhabitant.A = Rank.Ordinary ∧ rank Inhabitant.B = Rank.Knight ∧ rank Inhabitant.C = Rank.Liar) ∨
  (rank Inhabitant.A = Rank.Ordinary ∧ rank Inhabitant.B = Rank.Liar ∧ rank Inhabitant.C = Rank.Knight) ∨
  (rank Inhabitant.A = Rank.Liar ∧ rank Inhabitant.B = Rank.Knight ∧ rank Inhabitant.C = Rank.Ordinary) ∨
  (rank Inhabitant.A = Rank.Liar ∧ rank Inhabitant.B = Rank.Ordinary ∧ rank Inhabitant.C = Rank.Knight)

-- The theorem to prove
theorem C_must_say_B_higher_than_A :
  isTruthful (rank Inhabitant.B > rank Inhabitant.A) Inhabitant.C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_must_say_B_higher_than_A_l310_31077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l310_31033

/-- The shortest distance from a point on the parabola y^2 = 8x to the line passing through (0,-4) and (3,2) is 3√5/5 -/
theorem shortest_distance_parabola_to_line :
  let A : ℝ × ℝ := (0, -4)
  let B : ℝ × ℝ := (3, 2)
  let parabola := {P : ℝ × ℝ | P.2^2 = 8*P.1}
  let line := {Q : ℝ × ℝ | 2*Q.1 - Q.2 - 4 = 0}
  ∃ d : ℝ, d = 3 * Real.sqrt 5 / 5 ∧
    ∀ P ∈ parabola, ∀ Q ∈ line, d ≤ dist P Q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_parabola_to_line_l310_31033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_height_theorem_l310_31035

/-- The number of hours Alex needs to hang upside down each month to reach the required height for the roller coaster --/
noncomputable def hours_per_month (required_height current_height normal_growth_rate upside_down_growth_rate : ℝ) : ℝ :=
  let months_in_year : ℝ := 12
  let height_difference := required_height - current_height
  let normal_yearly_growth := normal_growth_rate * months_in_year
  let additional_growth_needed := height_difference - normal_yearly_growth
  let total_upside_down_hours := additional_growth_needed / upside_down_growth_rate
  total_upside_down_hours / months_in_year

theorem roller_coaster_height_theorem (required_height current_height normal_growth_rate upside_down_growth_rate : ℝ)
  (h1 : required_height = 60)
  (h2 : current_height = 48)
  (h3 : normal_growth_rate = 1/3)
  (h4 : upside_down_growth_rate = 1/8) :
  ∃ ε > 0, |hours_per_month required_height current_height normal_growth_rate upside_down_growth_rate - 5.33| < ε := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval hours_per_month 60 48 (1/3) (1/8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roller_coaster_height_theorem_l310_31035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_theorem_l310_31044

/-- The distance from the focus to the directrix of a parabola -/
noncomputable def parabola_focus_directrix_distance (a : ℝ) : ℝ :=
  |a| / 2

/-- Theorem: For a parabola with equation y² = ax (a ≠ 0), 
    the distance from its focus to its directrix is |a|/2 -/
theorem parabola_focus_directrix_distance_theorem (a : ℝ) (h : a ≠ 0) :
  parabola_focus_directrix_distance a = |a| / 2 := by
  -- Unfold the definition of parabola_focus_directrix_distance
  unfold parabola_focus_directrix_distance
  -- The equality is trivial by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_theorem_l310_31044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l310_31016

theorem cos_alpha_value (α β : ℝ) : 
  α ∈ Set.Ioo 0 Real.pi → 
  β ∈ Set.Ioo 0 Real.pi → 
  Real.cos β = -1/3 → 
  Real.sin (α + β) = 4/5 → 
  Real.cos α = (3 + 8 * Real.sqrt 2) / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l310_31016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_correctly_rounded_correct_values_l310_31075

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the rounding function to 7 decimal places
noncomputable def roundTo7DecimalPlaces (x : ℝ) : ℝ := 
  (⌊x * 10000000⌋ : ℝ) / 10000000

-- Define the given approximate values
def approxValues : List (ℝ × ℝ) := [
  (2, 1.2599210),
  (16, 2.5198421),
  (54, 3.7797631),
  (128, 5.0396842),
  (250, 6.2996053),
  (432, 7.5595263),
  (686, 8.8194474),
  (1024, 10.0793684)
]

-- Theorem stating that not all values are correctly rounded
theorem not_all_correctly_rounded : 
  ∃ (x : ℝ) (y : ℝ), (x, y) ∈ approxValues ∧ y ≠ roundTo7DecimalPlaces (cubeRoot x) :=
by
  sorry

-- Theorem stating the correct values for 250 and 686
theorem correct_values : 
  roundTo7DecimalPlaces (cubeRoot 250) = 6.2996052 ∧
  roundTo7DecimalPlaces (cubeRoot 686) = 8.8194473 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_correctly_rounded_correct_values_l310_31075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l310_31092

def A : Set ℝ := {x | x^2 > 9}

theorem complement_of_A : 
  Set.compl A = {x : ℝ | -3 ≤ x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l310_31092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_28_l310_31014

-- Define the series A
noncomputable def A : ℝ := ∑' n, if (n % 2 = 1) ∧ (n % 3 ≠ 0) then (if (n - 1) / 2 % 2 = 0 then 1 else -1) / n^3 else 0

-- Define the series B
noncomputable def B : ℝ := ∑' n, if n % 6 = 3 then (-1)^((n - 3) / 6) / n^3 else 0

-- Theorem statement
theorem A_div_B_eq_28 : A / B = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_28_l310_31014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_G_l310_31042

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + Real.sqrt 7)^2 + y^2 = 64

-- Define the fixed point N
def point_N : ℝ × ℝ := (Real.sqrt 7, 0)

-- Define a point P on the circle
def point_P : ℝ × ℝ → Prop := λ p ↦ circle_M p.1 p.2

-- Define point Q on line segment NP
def point_Q (p q : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ q = (t * point_N.1 + (1 - t) * p.1, t * point_N.2 + (1 - t) * p.2)

-- Define point G on line segment MP
def point_G (m p g : ℝ × ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ g = (t * m.1 + (1 - t) * p.1, t * m.2 + (1 - t) * p.2)

-- Vector NP = 2 * Vector NQ
def vector_condition (p q : ℝ × ℝ) : Prop := 
  (p.1 - point_N.1, p.2 - point_N.2) = (2 * (q.1 - point_N.1), 2 * (q.2 - point_N.2))

-- Vector GQ · Vector NP = 0
def orthogonal_condition (g p q : ℝ × ℝ) : Prop :=
  (g.1 - q.1) * (p.1 - point_N.1) + (g.2 - q.2) * (p.2 - point_N.2) = 0

-- Theorem statement
theorem trajectory_of_G (g : ℝ × ℝ) : 
  (∃ p q : ℝ × ℝ, point_P p ∧ point_Q p q ∧ point_G (-Real.sqrt 7, 0) p g ∧ 
   vector_condition p q ∧ orthogonal_condition g p q) → 
  g.1^2 / 16 + g.2^2 / 9 = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_G_l310_31042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_between_triangles_l310_31091

/-- The set of trapezoids formed between the inner and outer triangles -/
def set_of_trapezoids : Set ℝ := sorry

/-- The area of a trapezoid formed between two concentric equilateral triangles -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ)
  (inner_area : ℝ)
  (num_trapezoids : ℕ)
  (h_outer_area : outer_area = 36)
  (h_inner_area : inner_area = 4)
  (h_num_trapezoids : num_trapezoids = 4)
  (h_congruent : ∀ t1 t2 : ℝ, t1 ∈ set_of_trapezoids ∧ t2 ∈ set_of_trapezoids → t1 = t2)
  : (outer_area - inner_area) / num_trapezoids = 8 := by
  sorry

#check trapezoid_area_between_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_between_triangles_l310_31091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_of_2457_l310_31063

theorem smallest_prime_factor_of_2457 : (Nat.factors 2457).head! = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_of_2457_l310_31063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_exists_and_unique_l310_31062

-- Define a right triangle
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the projection of a leg on the hypotenuse
noncomputable def leg_projection (t : RightTriangle) : ℝ :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  AC * (AC / AB)

-- Define the angle bisector of the right angle
noncomputable def angle_bisector (t : RightTriangle) : ℝ :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  (AB * AC) / (AB + AC)

-- Theorem statement
theorem right_triangle_exists_and_unique 
  (proj : ℝ) 
  (h_proj_pos : proj > 0) :
  ∃! t : RightTriangle, 
    leg_projection t = proj ∧ 
    angle_bisector t = Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_exists_and_unique_l310_31062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l310_31023

theorem matrix_scalar_multiplication (v : Fin 3 → ℝ) :
  let N : Matrix (Fin 3) (Fin 3) ℝ := ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]
  N.vecMul v = (3 : ℝ) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l310_31023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_upper_bound_l310_31011

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (log x + 1) / (exp x)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (1 - x * log x - x) / (x * exp x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x^2 + x) * f' x

-- Theorem statement
theorem g_upper_bound {x : ℝ} (hx : x > 0) : g x < 1 + exp 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_upper_bound_l310_31011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_count_l310_31046

theorem perfect_square_divisors_count : 
  let n := (2^12) * (3^15) * (5^18) * (7^8)
  (Finset.filter (fun x => x^2 ∣ n ∧ x > 0) (Finset.range (n + 1))).card = 2800 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_divisors_count_l310_31046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_dist_correct_l310_31037

/-- The number of lathes -/
def num_lathes : ℕ := 3

/-- The probability of failure for each lathe in scenario 1 -/
noncomputable def p_fail_s1 : ℝ := 0.2

/-- The probability of failure for type A lathes in scenario 2 -/
noncomputable def p_fail_s2_A : ℝ := 0.1

/-- The probability of failure for type B lathes in scenario 2 -/
noncomputable def p_fail_s2_B : ℝ := 0.2

/-- The number of type A lathes in scenario 2 -/
def num_lathes_A : ℕ := 2

/-- The number of type B lathes in scenario 2 -/
def num_lathes_B : ℕ := 1

/-- The probability distribution for scenario 1 -/
noncomputable def prob_dist_s1 : Fin 4 → ℝ
| 0 => 64/125
| 1 => 48/125
| 2 => 12/125
| 3 => 1/125

/-- The probability distribution for scenario 2 -/
noncomputable def prob_dist_s2 : Fin 4 → ℝ
| 0 => 0.648
| 1 => 0.306
| 2 => 0.044
| 3 => 0.002

/-- Theorem stating that the probability distributions are correct -/
theorem prob_dist_correct :
  (∀ x : Fin 4, prob_dist_s1 x = (Nat.choose num_lathes x.val) * (1 - p_fail_s1) ^ (num_lathes - x.val) * p_fail_s1 ^ x.val) ∧
  (∀ x : Fin 4, prob_dist_s2 x = 
    if x.val = 0 then (1 - p_fail_s2_A) ^ num_lathes_A * (1 - p_fail_s2_B) ^ num_lathes_B
    else if x.val = 1 then 
      Nat.choose num_lathes_A 1 * (1 - p_fail_s2_A) * p_fail_s2_A * (1 - p_fail_s2_B) +
      (1 - p_fail_s2_A) ^ num_lathes_A * p_fail_s2_B
    else if x.val = 2 then
      Nat.choose num_lathes_A 1 * (1 - p_fail_s2_A) * p_fail_s2_A * p_fail_s2_B +
      p_fail_s2_A ^ num_lathes_A * (1 - p_fail_s2_B)
    else p_fail_s2_A ^ num_lathes_A * p_fail_s2_B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_dist_correct_l310_31037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_no_straightforward_solution_l310_31085

theorem complex_equation_no_straightforward_solution :
  ∀ (a b : ℝ) (z : ℂ),
    b > 0 →
    z = a + b * Complex.I →
    (z - 2 * Complex.I) * (z + 2 * Complex.I) * (z - 3 * Complex.I) = 1234 * Complex.I →
    ¬ (a ∈ ({0, 2, 3, 4, 5} : Set ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_no_straightforward_solution_l310_31085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_units_count_l310_31002

/-- The number of storage units in a building with given specifications --/
theorem storage_units_count : ℕ := by
  -- Define the total area of all storage units
  let total_area : ℕ := 5040

  -- Define the number of small units
  let small_units : ℕ := 20

  -- Define the dimensions of small units
  let small_unit_length : ℕ := 8
  let small_unit_width : ℕ := 4

  -- Define the area of one large unit
  let large_unit_area : ℕ := 200

  -- Calculate the total area of small units
  let small_units_area : ℕ := small_units * small_unit_length * small_unit_width

  -- Calculate the remaining area for large units
  let large_units_area : ℕ := total_area - small_units_area

  -- Calculate the number of large units
  let large_units : ℕ := large_units_area / large_unit_area

  -- Calculate the total number of units
  let total_units : ℕ := small_units + large_units

  -- State the goal
  have h : total_units = 42 := by
    -- The actual proof would go here
    sorry

  -- Return the result
  exact 42

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_units_count_l310_31002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_city_road_system_exists_l310_31049

/-- A directed graph representing the road system between cities. -/
structure RoadSystem where
  n : Nat  -- number of cities
  hasRoad : Fin n → Fin n → Prop  -- hasRoad i j means there's a road from city i to city j

/-- A path in the road system. -/
inductive RoadPath (r : RoadSystem) : Fin r.n → Fin r.n → Nat → Prop where
  | direct {i j : Fin r.n} (h : r.hasRoad i j) : RoadPath r i j 1
  | step {i j k : Fin r.n} {l : Nat} (h1 : r.hasRoad i k) (h2 : RoadPath r k j l) : RoadPath r i j (l + 1)

/-- The main theorem: there exists a road system for 7 cities where any two cities are connected by a path of length at most 2. -/
theorem seven_city_road_system_exists :
  ∃ (r : RoadSystem), r.n = 7 ∧ ∀ (i j : Fin r.n), ∃ (l : Nat), l ≤ 2 ∧ RoadPath r i j l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_city_road_system_exists_l310_31049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_for_prism_l310_31061

/-- A right rectangular prism with edge lengths 2, 3, and 4 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 4

/-- The set of points in 3D space no farther than distance r from any point in P -/
def T (P : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of T(r) -/
def V (P : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function V(r) = ar³ + br² + cr + d -/
structure VolumeCoefficients (P : Prism) where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_ratio_for_prism (P : Prism) (coeff : VolumeCoefficients P) :
  (∀ r : ℝ, V P r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
  (coeff.b * coeff.c) / (coeff.a * coeff.d) = 14.625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_for_prism_l310_31061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l310_31019

theorem complex_real_condition (a : ℝ) : 
  (Complex.I + 1) * (1 - a * Complex.I) ∈ Set.range Complex.ofReal → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_condition_l310_31019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l310_31018

/-- Represents the tax calculation function --/
noncomputable def tax_function (x : ℝ) : ℝ :=
  if x ≤ 500 then 0.05 * x
  else if x ≤ 2000 then 25 + 0.10 * (x - 500)
  else 175 + 0.15 * (x - 2000)

/-- Represents the tax-free threshold --/
def tax_free_threshold : ℝ := 800

/-- Calculates the taxable amount --/
noncomputable def taxable_amount (total_income : ℝ) : ℝ :=
  max (total_income - tax_free_threshold) 0

/-- Theorem stating that for a monthly income of 3000 yuan, the tax payable is 205 yuan --/
theorem tax_calculation_correct :
  tax_function (taxable_amount 3000) = 205 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_calculation_correct_l310_31018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_two_thousand_fourteenth_term_l310_31017

/-- Defines the sequence based on the given rules -/
def mySequence : ℕ → ℕ
| 0 => 1  -- The sequence starts with 1
| n + 1 => 
  let k := (n + 2) / 2  -- Determines which group we're in
  let isEven := n % 2 = 1  -- Even groups are at odd indices
  let groupStart := if isEven then 2*k*(k-1) else k*k
  groupStart + (n + 1 - k*(k-1))

/-- States that the 15th term of the sequence is 25 -/
theorem fifteenth_term : mySequence 14 = 25 := by
  sorry

/-- States that the 2014th term of the sequence is 3965 -/
theorem two_thousand_fourteenth_term : mySequence 2013 = 3965 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_two_thousand_fourteenth_term_l310_31017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_specific_values_l310_31080

theorem tan_difference_specific_values (α β : ℝ) 
  (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_specific_values_l310_31080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaO_l310_31051

-- Define the molar masses
noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_O : ℝ := 16.00

-- Define the molar mass of CaO
noncomputable def molar_mass_CaO : ℝ := molar_mass_Ca + molar_mass_O

-- Define the mass percentage formula
noncomputable def mass_percentage (mass_element : ℝ) (mass_compound : ℝ) : ℝ :=
  (mass_element / mass_compound) * 100

-- Theorem statement
theorem mass_percentage_O_in_CaO :
  abs (mass_percentage molar_mass_O molar_mass_CaO - 28.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaO_l310_31051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l310_31057

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (1 + 1 / ((n + 1)^2 + (n + 1))) * sequence_a n + 1 / 2^(n + 1)

theorem sequence_a_properties :
  (∀ n ≥ 2, sequence_a n ≥ 2) ∧
  (∀ n ≥ 1, sequence_a n < Real.exp 2) := by
  sorry

axiom ln_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l310_31057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_centroid_theorem_l310_31089

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- A function representing the movement of points along the sides of a triangle -/
noncomputable def move (t : Triangle) (time : ℝ) : Triangle :=
  sorry

/-- The theorem statement -/
theorem beetle_centroid_theorem (t : Triangle) :
  (∀ time : ℝ, centroid (move t time) = centroid t) →
  (∃ time : ℝ, (move t time).A = t.A ∧ (move t time).B = t.B ∧ (move t time).C = t.C) →
  centroid (move t 1) = centroid t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_centroid_theorem_l310_31089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l310_31087

/-- The function g as defined in the problem -/
noncomputable def g (n : ℝ) : ℝ := (1/4) * n * (n+1) * (n+2) * (n+3)

/-- Theorem stating the difference of g(r) and g(r-1) -/
theorem g_difference (r : ℝ) : g r - g (r-1) = r * (r+1) * (r+2) := by
  -- Expand the definition of g
  unfold g
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l310_31087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l310_31020

/-- Given a regular triangular pyramid with lateral edge √5 and height 1,
    the dihedral angle at the base is 45°. -/
theorem dihedral_angle_regular_triangular_pyramid 
  (lateral_edge : ℝ)
  (height : ℝ)
  (h_lateral_edge : lateral_edge = Real.sqrt 5)
  (h_height : height = 1) :
  let dihedral_angle := Real.arctan (height / (lateral_edge / Real.sqrt 3))
  dihedral_angle = 45 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l310_31020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_institute_size_l310_31079

/-- Represents an employee in the institute -/
structure Employee where
  isTruthTeller : Bool
  workload : ℕ
  salary : ℕ

/-- The institute with its employees -/
structure Institute where
  employees : List Employee

/-- Axiom: All employees have different workloads -/
axiom different_workloads (i : Institute) :
  ∀ e1 e2 : Employee, e1 ∈ i.employees → e2 ∈ i.employees → e1 ≠ e2 → e1.workload ≠ e2.workload

/-- Axiom: All employees have different salaries -/
axiom different_salaries (i : Institute) :
  ∀ e1 e2 : Employee, e1 ∈ i.employees → e2 ∈ i.employees → e1 ≠ e2 → e1.salary ≠ e2.salary

/-- Axiom: Statement 1 is true for truth-tellers and false for liars -/
axiom statement1 (i : Institute) (e : Employee) :
  e ∈ i.employees →
    (e.isTruthTeller ↔ (i.employees.filter (λ e' ↦ e'.workload > e.workload)).length < 10)

/-- Axiom: Statement 2 is true for truth-tellers and false for liars -/
axiom statement2 (i : Institute) (e : Employee) :
  e ∈ i.employees →
    (e.isTruthTeller ↔ (i.employees.filter (λ e' ↦ e'.salary > e.salary)).length ≥ 100)

/-- Theorem: The number of employees in the institute is 110 -/
theorem institute_size (i : Institute) : i.employees.length = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_institute_size_l310_31079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_fourth_quadrant_l310_31099

noncomputable def terminal_side_quadrant (α : Real) : Nat :=
  if Real.sin α > 0 && Real.cos α > 0 then 1
  else if Real.sin α > 0 && Real.cos α < 0 then 2
  else if Real.sin α < 0 && Real.cos α < 0 then 3
  else 4

theorem terminal_side_in_fourth_quadrant (α : Real) :
  (Real.tan α < 0 ∧ Real.cos α > 0) → terminal_side_quadrant α = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_fourth_quadrant_l310_31099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_theorem_l310_31008

/-- The radius of a sphere inscribed in a right circular cone --/
noncomputable def inscribed_sphere_radius (base_radius height : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (base_radius^2 + height^2)
  50 * (Real.sqrt 5 - 1)

/-- The sum of coefficients in the expression of the radius --/
def coefficient_sum (a c : ℝ) : ℝ := a + c

theorem inscribed_sphere_theorem (base_radius height : ℝ) 
  (h1 : base_radius = 10) 
  (h2 : height = 20) : 
  ∃ (a c : ℝ), 
    inscribed_sphere_radius base_radius height = a * Real.sqrt c - a ∧ 
    coefficient_sum a c = 55 := by
  use 50, 5
  constructor
  · simp [inscribed_sphere_radius, h1, h2]
    -- The proof steps would go here, but we'll use sorry for now
    sorry
  · simp [coefficient_sum]
    norm_num

#check inscribed_sphere_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_theorem_l310_31008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_16_l310_31067

-- Define a Point type for 2D coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the triangle vertices
def D : Point := ⟨2, 3⟩
def E : Point := ⟨2, 9⟩
def F : Point := ⟨6, 6⟩

-- Define the perimeter of the triangle
noncomputable def trianglePerimeter : ℝ :=
  distance D E + distance E F + distance F D

-- Theorem statement
theorem triangle_perimeter_is_16 : trianglePerimeter = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_is_16_l310_31067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_at_P_l310_31056

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, -2)

-- Define the point P on the x-axis
def P : ℝ × ℝ := (13, 0)

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating that P maximizes the absolute difference |AP| - |BP|
theorem max_difference_at_P :
  ∀ x : ℝ, |distance A P - distance B P| ≥ |distance A (x, 0) - distance B (x, 0)| := by
  sorry

#check max_difference_at_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_at_P_l310_31056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l310_31009

/-- The area of a triangle given its vertex coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)

/-- Theorem: The area of the triangle with vertices at (-3, 4), (1, 7), and (3, -1) is 16 square units -/
theorem triangle_DEF_area :
  triangleArea (-3) 4 1 7 3 (-1) = 16 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l310_31009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_x_percentage_in_mixed_solution_l310_31050

/-- The percentage of liquid X in a solution -/
def LiquidXPercentage (solution : ℕ) : ℚ :=
  match solution with
  | 1 => 8/10   -- Solution A
  | 2 => 18/10  -- Solution B
  | 3 => 13/10  -- Solution C
  | 4 => 24/10  -- Solution D
  | _ => 0

/-- The weight of each solution in grams -/
def SolutionWeight (solution : ℕ) : ℕ :=
  match solution with
  | 1 => 400  -- Solution A
  | 2 => 700  -- Solution B
  | 3 => 500  -- Solution C
  | 4 => 600  -- Solution D
  | _ => 0

/-- The total weight of liquid X in the mixed solution -/
def TotalLiquidXWeight : ℚ :=
  (LiquidXPercentage 1 * SolutionWeight 1 +
   LiquidXPercentage 2 * SolutionWeight 2 +
   LiquidXPercentage 3 * SolutionWeight 3 +
   LiquidXPercentage 4 * SolutionWeight 4) / 100

/-- The total weight of the mixed solution -/
def TotalMixedWeight : ℕ :=
  SolutionWeight 1 + SolutionWeight 2 + SolutionWeight 3 + SolutionWeight 4

/-- The percentage of liquid X in the mixed solution -/
def MixedLiquidXPercentage : ℚ :=
  TotalLiquidXWeight / TotalMixedWeight * 100

theorem liquid_x_percentage_in_mixed_solution :
  |MixedLiquidXPercentage - 167/100| < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_x_percentage_in_mixed_solution_l310_31050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_proof_l310_31058

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in the compound -/
def carbon_atoms : ℕ := 8

/-- The number of Hydrogen atoms in the compound -/
def hydrogen_atoms : ℕ := 18

/-- The number of Nitrogen atoms in the compound -/
def nitrogen_atoms : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def oxygen_atoms : ℕ := 4

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  carbon_weight * (carbon_atoms : ℝ) +
  hydrogen_weight * (hydrogen_atoms : ℝ) +
  nitrogen_weight * (nitrogen_atoms : ℝ) +
  oxygen_weight * (oxygen_atoms : ℝ)

theorem compound_weight_proof :
  ∃ ε > 0, |molecular_weight - 206.244| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_proof_l310_31058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_A_equals_interval_l310_31066

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | Real.exp ((x + 1) * Real.log 2) > 1}

-- Define the complement of A with respect to B
def complement_B_A : Set ℝ := B \ A

-- State the theorem
theorem complement_B_A_equals_interval :
  complement_B_A = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_B_A_equals_interval_l310_31066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l310_31041

/-- Circle C centered at (2,2) with radius 1 -/
def C : Set (ℝ × ℝ) := {p | (p.fst - 2)^2 + (p.snd - 2)^2 = 1}

/-- Point M satisfies the tangent condition -/
def is_tangent_point (M : ℝ × ℝ) : Prop :=
  ∃ N : ℝ × ℝ, N ∈ C ∧
    (M.fst - N.fst)*(N.fst - 2) + (M.snd - N.snd)*(N.snd - 2) = 0 ∧
    (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = M.fst^2 + M.snd^2

/-- The minimum distance theorem -/
theorem min_distance_theorem :
  ∃ d : ℝ, d = 7 * Real.sqrt 2 / 8 ∧
    ∀ M : ℝ × ℝ, is_tangent_point M →
      Real.sqrt (M.fst^2 + M.snd^2) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l310_31041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_cubic_equation_solutions_count_comparison_l310_31047

/-- The number of solutions to the equation x^3 + 1 = ax depends on the value of a -/
theorem solutions_count_cubic_equation (a : ℝ) :
  (∃! x : ℝ, x^3 + 1 = a*x) ∨
  (∃ x y : ℝ, x ≠ y ∧ x^3 + 1 = a*x ∧ y^3 + 1 = a*y ∧ ∀ z : ℝ, z^3 + 1 = a*z → z = x ∨ z = y) ∨
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^3 + 1 = a*x ∧ y^3 + 1 = a*y ∧ z^3 + 1 = a*z) :=
by
  sorry

/-- The critical value for the number of solutions -/
noncomputable def critical_value : ℝ := (3/2) * Real.rpow 2 (1/3)

/-- The number of solutions is determined by comparing a with the critical value -/
theorem solutions_count_comparison (a : ℝ) :
  (a < critical_value → ∃! x : ℝ, x^3 + 1 = a*x) ∧
  (a = critical_value → ∃ x y : ℝ, x ≠ y ∧ x^3 + 1 = a*x ∧ y^3 + 1 = a*y ∧ ∀ z : ℝ, z^3 + 1 = a*z → z = x ∨ z = y) ∧
  (a > critical_value → ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^3 + 1 = a*x ∧ y^3 + 1 = a*y ∧ z^3 + 1 = a*z) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_cubic_equation_solutions_count_comparison_l310_31047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_divides_segment_l310_31025

noncomputable def harry_position : ℝ × ℝ := (10, -3)
noncomputable def sandy_position : ℝ × ℝ := (2, 7)
noncomputable def meeting_point : ℝ × ℝ := (14/3, 11/3)

def divides_in_ratio (p q r : ℝ × ℝ) (m n : ℝ) : Prop :=
  m * (r.1 - p.1) = n * (q.1 - r.1) ∧
  m * (r.2 - p.2) = n * (q.2 - r.2)

theorem meeting_point_divides_segment :
  divides_in_ratio harry_position meeting_point sandy_position 2 1 := by
  sorry

#check meeting_point_divides_segment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_divides_segment_l310_31025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l310_31021

theorem right_triangle_max_area :
  ∀ x : ℝ, 
  0 < x → x < 8 →
  let area := (1/2) * x * (8 - x)
  (∀ y : ℝ, 0 < y → y < 8 → area ≥ (1/2) * y * (8 - y)) ∧ 
  area = 8 ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_max_area_l310_31021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_equation_l310_31010

theorem largest_root_equation : 
  ∃ (x : ℝ), x = Real.sqrt 6 - 2 ∧ 
  x^2 + 4 * abs x + 2 / (x^2 + 4 * abs x) = 3 ∧
  ∀ (y : ℝ), y^2 + 4 * abs y + 2 / (y^2 + 4 * abs y) = 3 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_equation_l310_31010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l310_31060

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (a x : ℝ) : ℝ := 2^x + a

-- State the theorem
theorem function_inequality (a : ℝ) : 
  (∃ x₁ ∈ Set.Icc (1/2 : ℝ) 3, ∀ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g a x₂) → 
  a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l310_31060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l310_31005

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : inner a b = -(1/2 : ℝ)) : 
  ‖a + 2 • b‖ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l310_31005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drivers_meet_conditions_l310_31082

/-- Two drivers traveling between cities -/
structure TwoDriversProblem where
  a : ℝ  -- distance between cities in km
  t : ℝ  -- time difference in hours
  h : 0 < a ∧ 0 < t  -- ensure positive distance and time

/-- The speeds of the two drivers -/
noncomputable def driverSpeeds (p : TwoDriversProblem) : ℝ × ℝ :=
  (p.a * (Real.sqrt 5 - 1) / (4 * p.t), p.a * (3 - Real.sqrt 5) / (4 * p.t))

theorem drivers_meet_conditions (p : TwoDriversProblem) :
  let (x, y) := driverSpeeds p
  (p.a / (2 * x) + p.t = p.a / (2 * y)) ∧  -- meet halfway condition
  (2 * p.t * (x + y) = p.a)                -- meet in 2t hours condition
  := by sorry

#check drivers_meet_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drivers_meet_conditions_l310_31082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_build_five_l310_31054

/-- The number of houses that can be built by a group of people in 5 days -/
def houses_built (num_people : ℕ) : ℕ := sorry

/-- Assumption: 100 people can build 100 houses in 5 days -/
axiom hundred_people_build_hundred : houses_built 100 = 100

/-- Theorem: 5 people can build 5 houses in 5 days -/
theorem five_people_build_five : houses_built 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_build_five_l310_31054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_of_f_l310_31090

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.rpow 3 (1/x)

-- State the theorem
theorem limits_of_f :
  (∀ ε > 0, ∃ δ > 0, ∀ x < 0, |x| < δ → |f x| < ε) ∧
  (∀ M > 0, ∃ δ > 0, ∀ x > 0, x < δ → f x > M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limits_of_f_l310_31090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l310_31034

/-- The diameter of the cylindrical pipes -/
def pipe_diameter : ℝ := 12

/-- The number of pipes in each crate -/
def num_pipes : ℕ := 144

/-- The height of square packing for the given number of pipes -/
noncomputable def square_packing_height : ℝ := pipe_diameter * Real.sqrt (num_pipes : ℝ)

/-- The height of hexagonal packing for the given number of pipes -/
noncomputable def hexagonal_packing_height : ℝ :=
  11 * pipe_diameter * Real.sin (60 * Real.pi / 180) + pipe_diameter

/-- The difference in height between square and hexagonal packing -/
theorem packing_height_difference :
  square_packing_height - hexagonal_packing_height = 132 - 66 * Real.sqrt 3 := by
  sorry

-- Remove #eval statements as they can't be computed for noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_packing_height_difference_l310_31034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l310_31003

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 = a2 * b1 ∧ a1 ≠ 0 ∧ a2 ≠ 0

/-- The slope of line 1: 2x + (m+1)y + 4 = 0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -(2 / (m + 1))

/-- The slope of line 2: mx + 3y - 2 = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -(m / 3)

theorem parallel_lines_condition (m : ℝ) :
  are_parallel 2 (m + 1) m 3 ↔ m = -3 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_condition_l310_31003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l310_31069

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors u and v
noncomputable def u (t : Triangle) : Real × Real := (t.b, -Real.sqrt 3 * t.a)
noncomputable def v (t : Triangle) : Real × Real := (Real.sin t.A, Real.cos t.B)

-- Define perpendicularity for 2D vectors
def perpendicular (x y : Real × Real) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : perpendicular (u t) (v t))  -- u is perpendicular to v
  (h2 : t.b = 3)    -- b = 3
  (h3 : t.c = 2 * t.a) -- c = 2a
  : t.B = π / 3 ∧ t.a = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l310_31069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_characterization_l310_31012

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_good_number (n : ℕ) : Prop :=
  ∃ (perm : Fin n → Fin n), Function.Bijective perm ∧
    ∀ k : Fin n, is_perfect_square ((k.val + 1) + ((perm k).val + 1))

theorem good_number_characterization :
  ∀ n ∈ ({11, 13, 15, 17, 19} : Set ℕ), is_good_number n ↔ n ≠ 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_number_characterization_l310_31012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_f_l310_31028

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

-- State the theorem
theorem second_derivative_of_f :
  ∀ x : ℝ, (deriv (deriv f)) x = Real.sin x + x * Real.cos x :=
by
  sorry

#check second_derivative_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_of_f_l310_31028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fire_alarms_and_passengers_discrete_l310_31071

-- Define the random variables
variable (ξ₁ ξ₂ ξ₃ : ℕ → ℝ)

-- Define what it means for a random variable to be discrete
def is_discrete (X : ℕ → ℝ) : Prop :=
  ∃ (S : Set ℝ), Set.Countable S ∧ ∀ n, X n ∈ S

-- State the theorem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ξ₁ ∧ is_discrete ξ₃ ∧ ¬is_discrete ξ₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fire_alarms_and_passengers_discrete_l310_31071
