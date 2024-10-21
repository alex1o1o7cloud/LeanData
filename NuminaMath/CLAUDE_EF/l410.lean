import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_boat_round_trip_speed_l410_41085

/-- Calculates the average speed for a round trip given upstream and downstream speeds -/
noncomputable def averageRoundTripSpeed (upstreamSpeed downstreamSpeed : ℝ) : ℝ :=
  2 * upstreamSpeed * downstreamSpeed / (upstreamSpeed + downstreamSpeed)

theorem river_boat_round_trip_speed :
  let upstreamSpeed : ℝ := 3
  let downstreamSpeed : ℝ := 7
  averageRoundTripSpeed upstreamSpeed downstreamSpeed = 4.2 := by
  -- Proof steps would go here
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_boat_round_trip_speed_l410_41085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l410_41011

/-- Pyramid with rectangular base -/
structure Pyramid where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

/-- Calculate the volume of a pyramid -/
noncomputable def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- The main theorem about the volume of the specific pyramid -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : ‖p.A - p.B‖ = 10) -- AB = 10
  (h2 : ‖p.B - p.C‖ = 5)  -- BC = 5
  (h3 : (p.P - p.A) • (p.A - p.D) = 0) -- PA ⟂ AD
  (h4 : (p.P - p.A) • (p.A - p.B) = 0) -- PA ⟂ AB
  (h5 : ‖p.P - p.B‖ = 20) -- PB = 20
  : pyramidVolume p = 500 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l410_41011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_spheres_volume_ratio_l410_41071

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  a_pos : 0 < a

/-- A sphere that touches every edge of a regular tetrahedron -/
structure TetrahedronSphere where
  tetrahedron : RegularTetrahedron
  radius : ℝ
  radius_pos : 0 < radius
  touches_edges : True  -- This is a placeholder for the touching condition

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The theorem stating the ratio of volumes of two spheres touching a regular tetrahedron -/
theorem tetrahedron_spheres_volume_ratio 
  (t : RegularTetrahedron) 
  (s1 s2 : TetrahedronSphere) 
  (h1 : s1.tetrahedron = t) 
  (h2 : s2.tetrahedron = t) 
  (h3 : s1.radius < s2.radius) :
  sphere_volume s1.radius / sphere_volume s2.radius = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_spheres_volume_ratio_l410_41071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_equivalence_l410_41006

/-- The solution set for the inequality (2^(x^2-6) - 4 * 2^(x+4)) * log_(cos(πx))(x^2 - 2x + 1) ≥ 0 -/
def solution_set : Set ℝ :=
  {x | x ∈ Set.Ioo (-2.5) (-2) ∪ Set.Ioo (-2) (-1.5) ∪ Set.Ioo (-0.5) 0 ∪ Set.Ioo 2 2.5 ∪ Set.Ioo 3.5 4}

/-- The inequality function -/
noncomputable def inequality (x : ℝ) : ℝ :=
  (2^(x^2 - 6) - 4 * 2^(x + 4)) * (Real.log (x^2 - 2*x + 1) / Real.log (Real.cos (Real.pi * x)))

theorem inequality_solution_equivalence :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_equivalence_l410_41006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l410_41072

def set_A : Set ℝ := {x | 4 - x^2 ≥ 0}

def set_B : Set ℝ := {x | ∃ y, y = Real.log (x + 1)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l410_41072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_trig_2013_l410_41048

theorem compare_trig_2013 : 
  Real.tan (2013 * π / 180) > Real.sin (2013 * π / 180) ∧ 
  Real.sin (2013 * π / 180) > Real.cos (2013 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_trig_2013_l410_41048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadruples_eq_40_l410_41008

/-- The number of quadruples (a, b, c, d) of natural numbers satisfying a * b * c * d = 98 -/
def count_quadruples : Nat :=
  (Finset.filter (fun quad => 
    quad.fst * quad.snd.fst * quad.snd.snd.fst * quad.snd.snd.snd = 98) 
    (Finset.product (Finset.range 99) 
      (Finset.product (Finset.range 99) 
        (Finset.product (Finset.range 99) (Finset.range 99))))).card

/-- Theorem stating that the number of quadruples (a, b, c, d) of natural numbers 
    satisfying a * b * c * d = 98 is equal to 40 -/
theorem count_quadruples_eq_40 : count_quadruples = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quadruples_eq_40_l410_41008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l410_41055

/-- The area of a triangle with vertices (0, 3, 5), (-2, 3, 1), and (1, 6, 1) is 3√13 -/
theorem triangle_area : 
  let A : Fin 3 → ℝ := ![0, 3, 5]
  let B : Fin 3 → ℝ := ![-2, 3, 1]
  let C : Fin 3 → ℝ := ![1, 6, 1]
  let area := Real.sqrt 13 * 3
  ∃ (triangleArea : ℝ), triangleArea = area ∧ 
    triangleArea = (1 / 2) * Real.sqrt (
      ((C 0 - A 0) * (B 1 - A 1) - (B 0 - A 0) * (C 1 - A 1))^2 +
      ((C 1 - A 1) * (B 2 - A 2) - (B 1 - A 1) * (C 2 - A 2))^2 +
      ((C 2 - A 2) * (B 0 - A 0) - (B 2 - A 2) * (C 0 - A 0))^2
    ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l410_41055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l410_41076

/-- Triangle with inscribed circle -/
structure InscribedCircleTriangle where
  /-- Side lengths of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Distance from vertex D to tangent point P on side DE -/
  dp : ℝ
  /-- Distance from vertex E to tangent point P on side DE -/
  pe : ℝ
  /-- The triangle inequality holds for the side lengths -/
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  /-- The radius is positive -/
  radius_positive : r > 0
  /-- The sum of dp and pe equals side c -/
  tangent_sum : dp + pe = c

/-- The perimeter of the triangle is approximately 340 -/
theorem inscribed_circle_triangle_perimeter
  (t : InscribedCircleTriangle)
  (h1 : t.r = 15)
  (h2 : t.dp = 36)
  (h3 : t.pe = 29) :
  ∃ (p : ℝ), abs (t.a + t.b + t.c - p) < 1 ∧ p = 340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l410_41076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_ratio_sum_l410_41087

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_ratio_sum (a b : V) (h1 : ‖a + b‖ = 3) (h2 : ‖a - b‖ = 1) :
  ‖a‖ / (‖b‖ * Real.cos (Real.arccos (inner a b / (‖a‖ * ‖b‖)))) + 
  ‖b‖ / (‖a‖ * Real.cos (Real.arccos (inner a b / (‖a‖ * ‖b‖)))) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_ratio_sum_l410_41087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l410_41070

noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ := 
  Real.arccos ((v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)))

theorem ellipse_dot_product (P F₁ F₂ : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  (F₁.1 = -1 ∧ F₁.2 = 0) →
  (F₂.1 = 1 ∧ F₂.2 = 0) →
  angle_between (F₁.1 - P.1, F₁.2 - P.2) (F₂.1 - P.1, F₂.2 - P.2) = π / 3 →
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_l410_41070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_track_inner_circumference_l410_41030

/-- The circumference of a circle with radius r -/
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

/-- The inner circumference of a circular race track -/
noncomputable def inner_circumference (outer_radius width : ℝ) : ℝ :=
  circumference (outer_radius - width)

theorem race_track_inner_circumference :
  let outer_radius : ℝ := 84.02817496043394
  let width : ℝ := 14
  let expected_inner_circumference : ℝ := 439.82297150253
  abs (inner_circumference outer_radius width - expected_inner_circumference) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_track_inner_circumference_l410_41030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l410_41096

/-- Properties of roots for the quadratic equation 8x^2 - mx + (m - 6) = 0 -/
theorem quadratic_roots_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => 8 * x^2 - m * x + (m - 6)
  let roots := {x : ℝ | f x = 0}
  ((∀ x, x ∈ roots → x > 0) ↔ (6 < m ∧ m ≤ 8) ∨ m ≥ 24) ∧
  (¬∃ m : ℝ, ∀ x, x ∈ roots → x < 0) ∧
  ((∃ x y, x ∈ roots ∧ y ∈ roots ∧ x * y < 0) ↔ m < 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l410_41096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_consecutive_zeros_l410_41084

/-- Represents a random 33-bit string -/
def RandomBitString : Type := Fin 33 → Bool

/-- The probability of a single bit being 0 -/
noncomputable def p_zero : ℝ := 1 / 2

/-- The probability of two consecutive bits being 0 -/
noncomputable def p_two_zeros : ℝ := p_zero * p_zero

/-- The number of possible locations for a two-bit substring in a 33-bit string -/
def num_locations : ℕ := 32

/-- The expected number of occurrences of two consecutive 0's in a random 33-bit string -/
noncomputable def expected_occurrences : ℝ := num_locations * p_two_zeros

theorem expected_consecutive_zeros (s : RandomBitString) : 
  expected_occurrences = 8 := by
  sorry

#eval num_locations -- This will evaluate to 32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_consecutive_zeros_l410_41084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l410_41043

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)) ∧ 
                     s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l410_41043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_f_l410_41057

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 - x^2) * (log x)^2

-- State the theorem
theorem third_derivative_f :
  ∀ x : ℝ, x > 0 → (deriv^[3] f) x = (-4 * log x - 9) / x - 6 / x^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_f_l410_41057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_price_l410_41004

/-- Represents the coffee mixture problem -/
structure CoffeeMixture where
  price1 : ℚ  -- Price of first coffee type per pound
  price2 : ℚ  -- Price of second coffee type per pound
  weight1 : ℚ  -- Weight of first coffee type in pounds
  weight2 : ℚ  -- Weight of second coffee type in pounds

/-- Calculates the selling price per pound of the coffee mixture -/
def sellingPrice (mix : CoffeeMixture) : ℚ :=
  (mix.price1 * mix.weight1 + mix.price2 * mix.weight2) / (mix.weight1 + mix.weight2)

/-- Theorem stating that the selling price of the given coffee mixture is $7.20 per pound -/
theorem coffee_mixture_price :
  let mix : CoffeeMixture := {
    price1 := 728/100,
    price2 := 642/100,
    weight1 := 6825/100,
    weight2 := 7
  }
  sellingPrice mix = 720/100 := by
  -- The proof goes here
  sorry

#eval sellingPrice {
  price1 := 728/100,
  price2 := 642/100,
  weight1 := 6825/100,
  weight2 := 7
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_mixture_price_l410_41004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_rotation_result_l410_41010

structure LineSegment where
  start : ℝ
  finish : ℝ
  length : ℝ

def rotate180 (segment : LineSegment) (center : ℝ) : LineSegment :=
  { start := 2 * center - segment.finish,
    finish := 2 * center - segment.start,
    length := segment.length }

theorem double_rotation_result (initialSegment : LineSegment) :
  initialSegment.start = 1 ∧
  initialSegment.finish = 6 ∧
  initialSegment.length = 5 →
  let afterFirstRotation := rotate180 initialSegment 2
  let finalSegment := rotate180 afterFirstRotation 1
  finalSegment.start = -1 ∧ finalSegment.finish = 4 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_rotation_result_l410_41010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_max_value_l410_41089

theorem cos_sin_max_value (x : ℝ) : 
  Real.cos x + Real.sin x + Real.cos x * Real.sin x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_max_value_l410_41089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l410_41095

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : ∃ (d : ℝ), d = 0 := by
  -- Define the first circle equation
  let circle1 : ℝ → ℝ → Prop := λ x y ↦ x^2 - 6*x + y^2 + 10*y + 9 = 0
  
  -- Define the second circle equation
  let circle2 : ℝ → ℝ → Prop := λ x y ↦ x^2 + 8*x + y^2 - 2*y + 16 = 0
  
  -- The shortest distance between the circles
  let shortest_distance : ℝ := 0

  -- Proof that the shortest distance is 0
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l410_41095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l410_41047

/-- The line equation -/
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - Real.sqrt 3 = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A is on the line and parabola, and above x-axis -/
def point_A (x y : ℝ) : Prop := line x y ∧ parabola x y ∧ y > 0

/-- Point B is on the line and parabola -/
def point_B (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point F is where the line intersects the x-axis -/
def point_F (x : ℝ) : Prop := line x 0

/-- Vector equation -/
def vector_equation (l m : ℝ) (xF yF xA yA xB yB : ℝ) : Prop :=
  (xF, yF) = l • (xA, yA) + m • (xB, yB)

theorem line_parabola_intersection (xA yA xB yB xF : ℝ) (l m : ℝ) :
  point_A xA yA → point_B xB yB → point_F xF →
  vector_equation l m xF 0 xA yA xB yB →
  m^2 - l^2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l410_41047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l410_41079

-- Define the function f
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^(-2*m^2 + m + 3)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (f 1 x - a*x) / Real.log a

-- State the theorem
theorem function_properties :
  ∃ (m : ℤ),
    (∀ x, f m x = f m (-x)) ∧  -- f is even
    (∀ x y, 0 < x ∧ x < y → f m x < f m y) ∧  -- f increases on (0, +∞)
    (∀ x, f m x = x^2) ∧  -- f(x) = x^2
    (∀ a, 0 < a ∧ a ≠ 1 →
      (∀ x y, 2 ≤ x ∧ x < y ∧ y ≤ 3 → g a x < g a y) →  -- g increases on [2, 3]
      1 < a ∧ a < 2) :=  -- range of a is (1, 2)
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l410_41079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_on_leg_l410_41059

/-- An isosceles right triangle with a circle touching its hypotenuse -/
structure IsoscelesRightTriangleWithCircle where
  /-- The length of the hypotenuse -/
  hypotenuseLength : ℝ
  /-- The radius of the circle -/
  circleRadius : ℝ
  /-- The circle touches the hypotenuse at its midpoint -/
  circleTouchesHypotenuseMidpoint : Bool

/-- The theorem statement -/
theorem segment_length_on_leg 
  (triangle : IsoscelesRightTriangleWithCircle)
  (h1 : triangle.hypotenuseLength = 40)
  (h2 : triangle.circleRadius = 9)
  (h3 : triangle.circleTouchesHypotenuseMidpoint = true) :
  ∃ (segmentLength : ℝ), segmentLength = Real.sqrt 82 :=
by
  sorry

/-- A comment explaining the meaning of the segment length -/
def segment_length_explanation : String :=
  "The segment length is the length of the segment cut off by the circle on one of the legs of the triangle."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_on_leg_l410_41059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_plane_angle_l410_41031

/-- A regular 2n-sided prism with a fixed base circumcircle radius -/
structure RegularPrism (n : ℕ) (R : ℝ) where
  vertices : Fin (4 * n) → ℝ × ℝ × ℝ
  is_regular : Bool
  base_radius : ℝ

/-- The angle between a diagonal and a plane in the prism -/
noncomputable def diagonal_plane_angle (n : ℕ) (R : ℝ) (p : RegularPrism n R) : ℝ := 
  sorry

/-- The height of the prism -/
noncomputable def prism_height (n : ℕ) (R : ℝ) (p : RegularPrism n R) : ℝ := 
  sorry

/-- The optimal prism configuration -/
def optimal_prism (n : ℕ) (R : ℝ) : RegularPrism n R :=
  ⟨λ i => sorry, sorry, sorry⟩

theorem max_diagonal_plane_angle {n : ℕ} (hn : n > 1) (R : ℝ) :
  ∀ p : RegularPrism n R,
    diagonal_plane_angle n R p ≤ diagonal_plane_angle n R (optimal_prism n R) ∧
    prism_height n R (optimal_prism n R) = 2 * R * Real.cos (π / (2 * n : ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonal_plane_angle_l410_41031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l410_41042

-- Define the function f(x) = 8/x
noncomputable def f (x : ℝ) : ℝ := 8 / x

-- Define the theorem
theorem inverse_proportion_inequality 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : x₁ < x₂) (h2 : x₂ < 0) (h3 : 0 < x₃)
  (h4 : y₁ = f x₁) (h5 : y₂ = f x₂) (h6 : y₃ = f x₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l410_41042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_value_l410_41005

theorem largest_k_value (X : Finset ℕ) (S : ℕ → Finset ℕ) (k : ℕ) :
  (X.card = 1983) →
  (∀ i j l, i ≠ j ∧ j ≠ l ∧ i ≠ l → i ≤ k ∧ j ≤ k ∧ l ≤ k → (S i ∪ S j ∪ S l) = X) →
  (∀ i j, i ≠ j → i ≤ k ∧ j ≤ k → (S i ∪ S j).card ≤ 1979) →
  k ≤ 31 ∧ ∃ X' : Finset ℕ, ∃ S' : ℕ → Finset ℕ, ∃ k' : ℕ, k' = 31 ∧
    (X'.card = 1983) ∧
    (∀ i j l, i ≠ j ∧ j ≠ l ∧ i ≠ l → i ≤ k' ∧ j ≤ k' ∧ l ≤ k' → (S' i ∪ S' j ∪ S' l) = X') ∧
    (∀ i j, i ≠ j → i ≤ k' ∧ j ≤ k' → (S' i ∪ S' j).card ≤ 1979) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_value_l410_41005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_lower_bound_l410_41067

/-- The set of altitudes of a triangle -/
def altitudes (T : Set (ℝ × ℝ)) : Set ℝ :=
sorry

/-- The area of a triangle -/
noncomputable def area (T : Set (ℝ × ℝ)) : ℝ :=
sorry

/-- Given a triangle with two altitudes greater than 1, its area is greater than 1/2 -/
theorem triangle_area_lower_bound (T : Set (ℝ × ℝ)) 
  (h1 : ∃ a ∈ altitudes T, a > 1) 
  (h2 : ∃ b ∈ altitudes T, b > 1 ∧ b ≠ a) : 
  area T > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_lower_bound_l410_41067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l410_41056

-- Define the sales revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 20 then 500 - 2*x
  else 370 + 2140/x - 6250/(x^2)

-- Define the profit function
noncomputable def S (x : ℝ) : ℝ :=
  x * R x - 380*x - 150

-- Theorem statement
theorem max_profit_at_25 :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → S y ≤ S x) ∧
  x = 25 ∧ S x = 1490 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_25_l410_41056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_intervals_max_m_for_one_positive_one_negative_root_l410_41018

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * |x^2 - a|

theorem f_monotone_decreasing_intervals (a : ℝ) (h : a = 1) :
  ∃ (i1 i2 : Set ℝ), 
    StrictMonoOn (f a) i1 ∧ 
    StrictMonoOn (f a) i2 ∧ 
    i1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ 
    i2 = Set.Icc (-1 + Real.sqrt 2) 1 := by sorry

theorem max_m_for_one_positive_one_negative_root (a : ℝ) (h : a ≥ 0) :
  ∃ (m : ℝ), m = 4 / Real.exp 2 ∧
    (∀ m' > m, ¬∃! (x : ℝ), x > 0 ∧ f a x = m') ∧
    (∀ m' > m, ¬∃! (x : ℝ), x < 0 ∧ f a x = m') := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_intervals_max_m_for_one_positive_one_negative_root_l410_41018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l410_41086

/-- A monic polynomial of degree 3 satisfying specific conditions -/
noncomputable def f (x : ℝ) : ℝ := x^3 + (1/2)*x^2 + (11/2)*x + 6

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying the given conditions -/
theorem unique_polynomial : 
  (∀ x, f x = x^3 + (1/2)*x^2 + (11/2)*x + 6) ∧ 
  f 0 = 6 ∧ 
  f 1 = 12 ∧ 
  f (-1) = 0 ∧
  (∀ g : ℝ → ℝ, (∃ a b c, ∀ x, g x = x^3 + a*x^2 + b*x + c) → 
    g 0 = 6 → g 1 = 12 → g (-1) = 0 → g = f) := by
  sorry

#check unique_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l410_41086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l410_41092

theorem min_roots_trigonometric_equation 
  (k₀ k₁ k₂ : ℕ) (A₁ A₂ : ℝ) 
  (h₁ : k₀ < k₁) (h₂ : k₁ < k₂) : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x < 2 * π ∧ 
      Real.sin (k₀ * x) + A₁ * Real.sin (k₁ * x) + A₂ * Real.sin (k₂ * x) = 0) ∧ 
    S.card ≥ 2 * k₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roots_trigonometric_equation_l410_41092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l410_41009

-- Define the vector operation
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

-- Define the vectors m and n
noncomputable def m : ℝ × ℝ := (2, 1/2)
noncomputable def n : ℝ × ℝ := (Real.pi/3, 0)

-- Define the function for point P
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, Real.sin x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (vector_op m (P ((x - Real.pi/3)/2)) + n).2

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l410_41009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_is_2_sqrt_5_l410_41003

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length

/-- The given ellipse from the problem -/
def problemEllipse : Ellipse :=
  { center := { x := 3, y := -2 }
    a := 4
    b := 2 }

/-- An endpoint of the major axis -/
def C : Point :=
  { x := problemEllipse.center.x + problemEllipse.a
    y := problemEllipse.center.y }

/-- An endpoint of the minor axis -/
def D : Point :=
  { x := problemEllipse.center.x
    y := problemEllipse.center.y + problemEllipse.b }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating that the distance between C and D is 2√5 -/
theorem distance_CD_is_2_sqrt_5 : distance C D = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_CD_is_2_sqrt_5_l410_41003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_25_l410_41032

noncomputable def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 + 6*x + 9 else x - 20

theorem g_composition_25 : g (g (g 25)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_25_l410_41032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l410_41015

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + 2*y - 1 = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ (m : ℝ), (∀ x y : ℝ, y = m*x → hyperbola a b x y) ∧ 
                   (∀ x y : ℝ, y = m*x → line x y → m * (-1/2) = -1)) :
  eccentricity a b = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l410_41015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l410_41020

/-- Helper function to represent the focus of a parabola -/
noncomputable def focus_of_parabola (left right : ℝ → ℝ) : ℝ × ℝ :=
  sorry

/-- The focus of the parabola x² = 16y is at the point (0, 4) -/
theorem parabola_focus (x y : ℝ) : 
  (x^2 = 16*y) → (0, 4) = focus_of_parabola (λ x => x^2) (λ y => 16*y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l410_41020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l410_41022

/-- The area of a rhombus with one diagonal of 6cm and side length equal to the positive root of x^2 - 2x - 15 = 0 is 24 cm^2. -/
theorem rhombus_area (side : ℝ) (h1 : side^2 - 2*side - 15 = 0) (h2 : side > 0) : 
  (1/2) * 6 * (2 * Real.sqrt (side^2 - 9)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l410_41022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l410_41040

/-- A line in 3D space -/
structure Line3D where
  point : (ℝ × ℝ × ℝ)
  direction : (ℝ × ℝ × ℝ)

/-- A plane in 3D space -/
structure Plane3D where
  point : (ℝ × ℝ × ℝ)
  normal : (ℝ × ℝ × ℝ)

/-- Predicate to check if a line is parallel to a plane -/
def isParallel (l : Line3D) (p : Plane3D) : Prop :=
  let (dx, dy, dz) := l.direction
  let (nx, ny, nz) := p.normal
  dx * nx + dy * ny + dz * nz = 0

/-- Predicate to check if a point lies on a line -/
def pointOnLine (point : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  ∃ t : ℝ, 
    let (px, py, pz) := point
    let (lx, ly, lz) := l.point
    let (dx, dy, dz) := l.direction
    px = lx + t * dx ∧ py = ly + t * dy ∧ pz = lz + t * dz

/-- Predicate to check if a point lies on a plane -/
def pointOnPlane (point : ℝ × ℝ × ℝ) (p : Plane3D) : Prop :=
  let (px, py, pz) := point
  let (qx, qy, qz) := p.point
  let (nx, ny, nz) := p.normal
  (px - qx) * nx + (py - qy) * ny + (pz - qz) * nz = 0

/-- Theorem stating that if a line is not parallel to a plane, then they have a point in common -/
theorem line_plane_intersection (l : Line3D) (p : Plane3D) :
  ¬(isParallel l p) → ∃ point : ℝ × ℝ × ℝ, pointOnLine point l ∧ pointOnPlane point p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l410_41040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_l410_41041

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_time : ∃ n : ℕ, 
  simpleInterest 1272.000000000001 10 5 = 
  (1/2) * compoundInterest 5000 12 (n : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_l410_41041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l410_41075

theorem sine_cosine_inequality : 
  (Real.sin (14 * π / 180) + Real.cos (14 * π / 180)) < 
  (Real.sqrt 6 / 2) ∧
  (Real.sqrt 6 / 2) < 
  (Real.sin (16 * π / 180) + Real.cos (16 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l410_41075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_equals_six_l410_41000

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line 3y - 2x + 6 = 0 -/
noncomputable def m1 : ℝ := 2/3

/-- The slope of the line 4y + bx + 3 = 0 in terms of b -/
noncomputable def m2 (b : ℝ) : ℝ := -b/4

/-- Theorem: If the lines 3y - 2x + 6 = 0 and 4y + bx + 3 = 0 are perpendicular, then b = 6 -/
theorem perpendicular_lines_b_equals_six :
  ∀ b : ℝ, perpendicular m1 (m2 b) → b = 6 :=
by
  intro b h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_b_equals_six_l410_41000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_value_is_mean_l410_41083

/-- A symmetric distribution with specific properties -/
structure SymmetricDistribution where
  /-- The central value of the distribution -/
  central_value : ℝ
  /-- The standard deviation of the distribution -/
  std_dev : ℝ
  /-- The distribution is symmetric about the central value -/
  symmetric : Bool
  /-- 68% of the distribution lies within one standard deviation of the central value -/
  within_one_std_dev : Float
  /-- 84% of the distribution is less than the central value plus one standard deviation -/
  less_than_one_std_dev : Float

/-- The central value of a symmetric distribution with specific properties is the mean -/
theorem central_value_is_mean (d : SymmetricDistribution) 
  (h1 : d.symmetric = true) 
  (h2 : d.within_one_std_dev = 0.68) 
  (h3 : d.less_than_one_std_dev = 0.84) : 
  d.central_value = d.central_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_value_is_mean_l410_41083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_l410_41016

-- Define a triangle ABC with sides a, b, c
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the theorem
theorem angle_C_is_60_degrees (t : Triangle) 
  (h : (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b) : 
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_60_degrees_l410_41016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_in_pyramid_l410_41078

def pyramid_books (n : ℕ) (top_level_books : ℕ) (ratio : ℚ) : ℕ :=
  let level_books := fun i => (top_level_books : ℚ) * ratio^(n - i - 1)
  (List.range n).map level_books |> List.map (fun x => Int.toNat x.floor) |> List.sum

theorem total_books_in_pyramid :
  pyramid_books 6 48 (3/4) = 662 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_books_in_pyramid_l410_41078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l410_41064

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def stretched_function (x : ℝ) : ℝ := original_function (x / 2)

noncomputable def final_function (x : ℝ) : ℝ := stretched_function (x - Real.pi / 4)

theorem function_transformation :
  ∀ x : ℝ, final_function x = Real.sin (x - Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l410_41064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_pure_imaginary_z_is_zero_z_in_third_quadrant_l410_41054

/-- Definition of the complex number z in terms of real number m -/
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) - m * (3 + Complex.I) - 6 * Complex.I

/-- z is a real number if and only if m = 3 or m = -2 -/
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 3 ∨ m = -2 := by sorry

/-- z is a complex number if and only if m ≠ 3 and m ≠ -2 -/
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 3 ∧ m ≠ -2 := by sorry

/-- z is a pure imaginary number if and only if m = 0 -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 0 := by sorry

/-- z is zero if and only if m = 3 -/
theorem z_is_zero (m : ℝ) : z m = 0 ↔ m = 3 := by sorry

/-- z corresponds to a point in the third quadrant if and only if 0 < m < 3 -/
theorem z_in_third_quadrant (m : ℝ) : 
  (z m).re < 0 ∧ (z m).im < 0 ↔ 0 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_pure_imaginary_z_is_zero_z_in_third_quadrant_l410_41054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l410_41093

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 6 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M ∧ ∃ (x₀ : ℝ), x₀ ≥ 0 ∧ f x₀ = M :=
by
  -- The proof goes here
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l410_41093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l410_41069

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m : ℝ) : ℝ := m

/-- The slope of a line in the form ax + by = c, where b ≠ 0, is -a/b -/
noncomputable def slope_of_general_line (a b c : ℝ) (h : b ≠ 0) : ℝ := -a / b

/-- The condition for the line ax + 2y = 0 to be parallel to y = 1 + x -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → y = 1 + x) ↔ a = -2 := by
  sorry

#check parallel_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l410_41069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equality_l410_41063

/-- Given two functions f and g where f⁻¹(g(x)) = x^4 - 4 and g has an inverse,
    prove that g⁻¹(f(15)) = ∜19 -/
theorem inverse_composition_equality (f g : ℝ → ℝ) 
    (h1 : ∀ x, f⁻¹ (g x) = x^4 - 4)
    (h2 : Function.Bijective g) :
    g⁻¹ (f 15) = (19 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_equality_l410_41063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_power_minus_three_l410_41027

theorem mod_power_minus_three (n : ℕ) : 
  n < 29 → 
  (5 * n) % 29 = 1 → 
  (2^n)^3 % 29 - 3 % 29 = 10 % 29 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_power_minus_three_l410_41027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_speed_proof_l410_41037

noncomputable section

-- Define the total number of pages
def total_pages : ℝ := 72

-- Define the ratio of pages typed by the first typist to the second typist
def typing_ratio : ℝ := 6 / 5

-- Define the time difference in hours
def time_difference : ℝ := 1.5

-- Define the typing speed of the first typist
def first_typist_speed : ℝ := 9.6

-- Define the typing speed of the second typist
def second_typist_speed : ℝ := 8

-- Theorem statement
theorem typing_speed_proof :
  (total_pages / first_typist_speed = 
   total_pages / second_typist_speed - time_difference) ∧
  (first_typist_speed / second_typist_speed = typing_ratio) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_speed_proof_l410_41037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_at_distance_l410_41021

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem two_points_same_color_at_distance (c : Coloring) (x : ℝ) :
  ∃ (p1 p2 : Point), c p1 = c p2 ∧ distance p1 p2 = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_points_same_color_at_distance_l410_41021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l410_41044

open Real

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  b = 2 * sqrt 6 / 3 →
  Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A) →
  B = π / 4 ∧ (1 / 2) * b * c * Real.sin A = 2 * sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l410_41044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_floor_fraction_l410_41074

theorem last_two_digits_of_floor_fraction : ∃ n : ℕ, 
  (n * 100 + 8 : ℕ) = (Int.toNat ⌊(10^93 : ℝ) / ((10^31 : ℝ) + 3)⌋) % 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_floor_fraction_l410_41074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l410_41001

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def is_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def triangle_area (t : Triangle) : ℝ :=
  Real.sqrt 3

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_arithmetic_sequence t.A t.B t.C)
  (h2 : is_geometric_sequence t.a t.b t.c)
  (h3 : triangle_area t = Real.sqrt 3) :
  t.B = 60 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l410_41001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_for_two_hours_l410_41046

noncomputable def parking_cost (initial_cost : ℝ) : ℝ → ℝ := fun hours =>
  if hours ≤ 2 then initial_cost
  else initial_cost + 1.75 * (hours - 2)

noncomputable def average_cost (initial_cost : ℝ) (hours : ℝ) : ℝ :=
  parking_cost initial_cost hours / hours

theorem parking_cost_for_two_hours : 
  ∃ (initial_cost : ℝ), 
    (average_cost initial_cost 9 = 3.0277777777777777) ∧ 
    (initial_cost = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_for_two_hours_l410_41046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l410_41034

/-- The expression we want to bound -/
noncomputable def f (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem f_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
    (h : a * b + b * c + c * a = 1) :
  27 / 16 ≤ f a b c ∧ f a b c ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l410_41034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_myopia_focal_length_relationship_l410_41061

-- Define the data points
def data_points : List (Float × Float) := [
  (1.00, 100),
  (0.50, 200),
  (0.25, 400),
  (0.20, 500),
  (0.10, 1000)
]

-- Define the function relating focal length to degree of myopia
def myopia_function (x : Float) : Float := 100 / x

-- Theorem statement
theorem myopia_focal_length_relationship :
  (∀ (point : Float × Float), point ∈ data_points → myopia_function point.1 = point.2) ∧
  (myopia_function (100 / 250) = 250) := by
  sorry

#check myopia_focal_length_relationship

end NUMINAMATH_CALUDE_ERRORFEEDBACK_myopia_focal_length_relationship_l410_41061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_movie_cost_ratio_l410_41024

/-- The cost of a DVD player in dollars -/
noncomputable def dvd_cost : ℝ := 81

/-- The difference in cost between a DVD player and a movie in dollars -/
noncomputable def cost_difference : ℝ := 63

/-- The cost of a movie in dollars -/
noncomputable def movie_cost : ℝ := dvd_cost - cost_difference

/-- The ratio of the cost of a DVD player to the cost of a movie -/
noncomputable def cost_ratio : ℝ := dvd_cost / movie_cost

theorem dvd_movie_cost_ratio :
  cost_ratio = 4.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_movie_cost_ratio_l410_41024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ketchup_distribution_l410_41039

/-- Given 84.6 grams of ketchup distributed equally among 12 hot dogs,
    each hot dog receives 7.05 grams of ketchup. -/
theorem ketchup_distribution (total_ketchup : ℚ) (num_hotdogs : ℕ) 
  (ketchup_per_hotdog : ℚ) : 
  total_ketchup = 846/10 ∧ num_hotdogs = 12 ∧ 
  ketchup_per_hotdog = total_ketchup / num_hotdogs → 
  ketchup_per_hotdog = 141/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ketchup_distribution_l410_41039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_C1_C2_l410_41017

/-- Curve C1 in parametric form -/
noncomputable def C1 (t : ℝ) : ℝ × ℝ :=
  (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

/-- Curve C2 in polar form -/
noncomputable def C2 (θ : ℝ) : ℝ :=
  2 * Real.sin θ

/-- Convert from polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Theorem stating the intersection points of C1 and C2 -/
theorem intersection_points_C1_C2 :
  ∃ (θ₁ θ₂ : ℝ),
    (polar_to_cartesian (Real.sqrt 2) (π/4) = C1 θ₁ ∧
     C2 (π/4) = Real.sqrt 2) ∧
    (polar_to_cartesian 2 (π/2) = C1 θ₂ ∧
     C2 (π/2) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_C1_C2_l410_41017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_geq_one_l410_41091

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 1 < x → x < y → f a x < f a y

theorem monotone_increasing_implies_a_geq_one (a : ℝ) :
  is_monotone_increasing a → a ≥ 1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_geq_one_l410_41091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l410_41058

/-- The area of a regular hexadecagon inscribed in a circle -/
theorem hexadecagon_area (r : ℝ) (r_pos : r > 0) : 
  (16 : ℝ) * ((1/2) * r^2 * Real.sin (2 * Real.pi / 32)) = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_l410_41058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_winning_strategy_l410_41090

/-- Represents a position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
deriving Repr, BEq

/-- Represents a move in the game -/
inductive Move where
  | Up : Nat → Move
  | Right : Nat → Move
deriving Repr

/-- The game state -/
structure GameState where
  board_size : Position
  current_position : Position
  is_jacob_turn : Bool
deriving Repr

/-- Checks if a position is on the diagonal -/
def is_on_diagonal (pos : Position) : Bool :=
  pos.x = pos.y

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.Up n => state.current_position.y + n ≤ state.board_size.y
  | Move.Right n => state.current_position.x + n ≤ state.board_size.x

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Up n => { state with 
      current_position := ⟨state.current_position.x, state.current_position.y + n⟩,
      is_jacob_turn := ¬state.is_jacob_turn }
  | Move.Right n => { state with 
      current_position := ⟨state.current_position.x + n, state.current_position.y⟩,
      is_jacob_turn := ¬state.is_jacob_turn }

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Bool :=
  state.current_position == state.board_size

/-- The main theorem to be proved -/
theorem jacob_winning_strategy (m n : Nat) :
  (∃ (strategy : GameState → Move),
    (∀ (state : GameState),
      state.board_size = ⟨m, n⟩ →
      state.current_position = ⟨0, 0⟩ →
      state.is_jacob_turn = true →
      is_valid_move state (strategy state) ∧
      (is_game_over (apply_move state (strategy state)) ∨
       ∀ (opponent_move : Move),
         is_valid_move (apply_move state (strategy state)) opponent_move →
         ¬is_game_over (apply_move (apply_move state (strategy state)) opponent_move)))) ↔
  m ≠ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_winning_strategy_l410_41090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_30_l410_41023

/-- The angle between clock hands at 2:30 -/
noncomputable def clock_angle (total_degrees : ℝ) (hour_marks : ℕ) (hour : ℕ) (minute : ℕ) : ℝ :=
  let degrees_per_hour := total_degrees / hour_marks
  let hour_angle := (↑hour % 12 + ↑minute / 60) * degrees_per_hour
  let minute_angle := ↑minute * (total_degrees / 60)
  min (abs (hour_angle - minute_angle)) (total_degrees - abs (hour_angle - minute_angle))

/-- Theorem: The smaller angle formed by the hands of a clock at 2:30 is 105° -/
theorem clock_angle_at_2_30 :
  clock_angle 360 12 2 30 = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_30_l410_41023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_foreign_students_l410_41019

/-- Calculates the number of new foreign students given the total number of students,
    the current percentage of foreign students, and the future number of foreign students. -/
theorem new_foreign_students
  (total_students : ℕ)
  (current_foreign_percentage : ℚ)
  (future_foreign_students : ℕ)
  (h1 : total_students = 1800)
  (h2 : current_foreign_percentage = 30 / 100)
  (h3 : future_foreign_students = 740) :
  future_foreign_students - (current_foreign_percentage * ↑total_students).floor = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_foreign_students_l410_41019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oak_tree_probability_l410_41033

/-- The probability of no two oak trees being adjacent when planting trees in a row -/
theorem oak_tree_probability (maple oak birch : ℕ) (h_maple : maple = 2) (h_oak : oak = 5) (h_birch : birch = 6) :
  let total := maple + oak + birch
  let non_oak := maple + birch
  let favorable_arrangements := Nat.choose (non_oak + 1) oak
  let total_arrangements := Nat.choose total oak * Nat.factorial maple * Nat.factorial birch
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oak_tree_probability_l410_41033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l410_41062

noncomputable def angle (A B C : ℝ × ℝ) : ℝ := 
  Real.arccos (((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)))

theorem right_triangle_side_length 
  (D E F : ℝ × ℝ)  -- Points in 2D plane
  (is_right_triangle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0)  -- Right angle at D
  (sin_E : Real.sin (angle D E F) = 8 * Real.sqrt 145 / 145)
  (de_length : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = Real.sqrt 145) :
  Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l410_41062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l410_41088

theorem complex_fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a ≠ b) :
  (b^(-(1/6 : ℝ)) * (a^3 * b)^(1/2 : ℝ) * (a^3 * b)^(1/3 : ℝ) - (a^3 * b^2)^(1/2 : ℝ) * b^(2/3 : ℝ)) /
  ((2*a^2 - b^2 - a*b) * (a^9 * b^4)^(1/6 : ℝ)) /
  (3*a^3 / (2*a^2 - a*b - b^2) - a*b / (a - b)) =
  1 / (a * (3*a + b)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l410_41088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_between_black_cells_l410_41068

/-- Represents a square board with odd side length -/
structure Board (n : ℕ) where
  side : n % 2 = 1

/-- Represents a cell on the board -/
structure Cell (n : ℕ) where
  x : Fin n
  y : Fin n

/-- Determines if a cell is black based on the checkerboard pattern -/
def is_black (n : ℕ) (c : Cell n) : Prop :=
  (c.x.val + c.y.val) % 2 = 0

/-- Determines if two cells are adjacent -/
def are_adjacent (n : ℕ) (c1 c2 : Cell n) : Prop :=
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y)

/-- A path on the board is a list of cells -/
def BoardPath (n : ℕ) := List (Cell n)

/-- Checks if a path is valid (adjacent moves and visits each cell once) -/
def is_valid_path (n : ℕ) (p : BoardPath n) : Prop :=
  p.length = n * n ∧
  p.Nodup ∧
  ∀ i, i < p.length - 1 → are_adjacent n (p.get ⟨i, by sorry⟩) (p.get ⟨i + 1, by sorry⟩)

/-- The main theorem: there always exists a valid path between any two black cells -/
theorem exists_valid_path_between_black_cells (n : ℕ) (b : Board n) 
  (start finish : Cell n) (h_start : is_black n start) (h_finish : is_black n finish) :
  ∃ (p : BoardPath n), is_valid_path n p ∧ p.head? = some start ∧ p.getLast? = some finish := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_path_between_black_cells_l410_41068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_blocks_and_budget_l410_41014

/-- Represents a team in a block -/
structure Team where
  workers : Nat
  giftValue : Nat

/-- Represents a block in the company -/
structure Block where
  teams : List Team
deriving Inhabited

def totalBudget : Nat := 4000

def block1 : Block := {
  teams := [
    { workers := 70, giftValue := 3 },
    { workers := 60, giftValue := 4 }
  ]
}

def block2 : Block := {
  teams := [
    { workers := 40, giftValue := 5 },
    { workers := 50, giftValue := 6 },
    { workers := 30, giftValue := 7 }
  ]
}

def block3 : Block := {
  teams := [
    { workers := 100, giftValue := 4 }
  ]
}

def blockCost (b : Block) : Nat :=
  b.teams.foldl (fun acc team => acc + team.workers * team.giftValue) 0

def knownBlocksCost : Nat := blockCost block1 + blockCost block2 + blockCost block3

theorem company_blocks_and_budget :
  (∃ (X : Nat), X = totalBudget - knownBlocksCost) ∧
  (∃ (blocks : List Block), blocks.length = 4 ∧ 
    blocks.take 3 = [block1, block2, block3] ∧
    blockCost (blocks.get! 3) = totalBudget - knownBlocksCost) := by
  sorry

#eval knownBlocksCost
#eval totalBudget - knownBlocksCost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_blocks_and_budget_l410_41014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_million_expressions_l410_41077

theorem one_million_expressions :
  -- First expression
  999999 + (9/9) = 1000000 ∧
  -- Second expression
  (999999 * 9 + 9) / 9 = 1000000 ∧
  -- Third expression
  ((9 : ℝ) + 9/9)^(9 - Real.sqrt 9) = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_million_expressions_l410_41077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_implies_at_most_one_root_l410_41050

theorem inverse_function_implies_at_most_one_root
  {X Y : Type} [LinearOrder X] [LinearOrder Y]
  (f : X → Y) (hf : Function.Injective f) (a : Y) :
  (∃! x, f x = a) ∨ ¬∃ x, f x = a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_implies_at_most_one_root_l410_41050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_relationship_l410_41051

theorem negation_relationship : True := by
  -- Define condition p and q as functions
  let p (x : ℝ) := x ≥ 1
  let q (x : ℝ) := 1/x < 1
  -- Define the negation of p
  let not_p (x : ℝ) := ¬(p x)
  
  -- Statement: not_p is neither sufficient nor necessary for q
  have h1 : ¬(∀ x : ℝ, not_p x → q x) := by sorry
  have h2 : ¬(∀ x : ℝ, q x → not_p x) := by sorry
  
  -- The proof is completed
  trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_relationship_l410_41051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slowly_increasing_interval_l410_41053

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x + 3/2

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

noncomputable def g (x : ℝ) : ℝ := f x / x

theorem slowly_increasing_interval :
  (is_increasing f 1 (Real.sqrt 3)) ∧
  (is_decreasing g 1 (Real.sqrt 3)) ∧
  (∀ a b : ℝ, a < 1 ∨ b > Real.sqrt 3 →
    ¬(is_increasing f a b ∧ is_decreasing g a b)) := by
  sorry

#check slowly_increasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slowly_increasing_interval_l410_41053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_already_added_marys_cake_recipe_l410_41028

/-- Given a recipe that requires a total amount of flour and the amount still needed to be added,
    calculate the amount of flour already put in. -/
theorem flour_already_added (total : ℕ) (to_add : ℕ) (h : total ≥ to_add) : 
  total - to_add = total - (total - (total - to_add)) := by
  sorry

/-- Mary's cake recipe problem -/
theorem marys_cake_recipe : 
  let total := 8
  let to_add := 6
  total - to_add = 2 := by
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_already_added_marys_cake_recipe_l410_41028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_one_forty_second_greater_l410_41038

open Real

/-- The decimal representation of 1/42 -/
noncomputable def one_forty_second : ℝ := 1 / 42

/-- The period of the repeating decimal representation of 1/42 -/
def period : ℕ := 6

/-- The 1997th digit after the decimal point in the decimal representation of 1/42 -/
def digit_1997 : ℕ := 1997 % period

/-- The function that removes the 1997th digit and shifts subsequent digits -/
noncomputable def modified_number (x : ℝ) : ℝ :=
  sorry

theorem modified_one_forty_second_greater :
  modified_number one_forty_second > one_forty_second :=
by sorry

#check modified_one_forty_second_greater

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_one_forty_second_greater_l410_41038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_tshirt_purchase_l410_41082

/-- The total amount spent by friends buying discounted t-shirts -/
noncomputable def total_spent (num_friends : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_friends * original_price * (1 - discount_percent / 100)

/-- Theorem: 4 friends buying t-shirts originally priced at $20 with a 50% discount spend $40 in total -/
theorem friends_tshirt_purchase :
  total_spent 4 20 50 = 40 := by
  -- Unfold the definition of total_spent
  unfold total_spent
  -- Simplify the arithmetic
  simp [Rat.mul_def, Rat.add_def]
  -- The proof is complete
  rfl

-- This will not be evaluated at compile-time due to noncomputable definition
-- #eval total_spent 4 20 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_tshirt_purchase_l410_41082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_return_speed_l410_41094

/-- Calculates the return speed given total journey time, distance, and outbound speed -/
noncomputable def return_speed (total_time : ℝ) (distance : ℝ) (outbound_speed : ℝ) : ℝ :=
  distance / (total_time - distance / outbound_speed)

/-- Theorem stating that for the given journey parameters, the return speed is 2 km/h -/
theorem journey_return_speed :
  let total_time : ℝ := 5.8
  let distance : ℝ := 10
  let outbound_speed : ℝ := 12.5
  return_speed total_time distance outbound_speed = 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_return_speed_l410_41094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l410_41065

theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) : 
  (a^3 + 2*a^2 + 4*a + 6 = 0) →
  (b^3 + 2*b^2 + 4*b + 6 = 0) →
  (c^3 + 2*c^2 + 4*c + 6 = 0) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  (P a = b + c) →
  (P b = a + c) →
  (P c = a + b) →
  (P (a + b + c) = -18) →
  (∀ x, P x = 9/4*x^3 + 5/2*x^2 + 7*x + 15/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_problem_l410_41065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_distribution_bacteria_never_equal_l410_41060

/-- Represents the state of the grid -/
structure GridState where
  white_count : ℕ  -- Number of bacteria in white cells
  black_count : ℕ  -- Number of bacteria in black cells

/-- Represents a move of a bacterium -/
inductive Move
  | WhiteToBlack  -- A bacterium moves from a white cell to a black cell
  | BlackToWhite  -- A bacterium moves from a black cell to a white cell

/-- Applies a move to the grid state -/
def apply_move (state : GridState) (move : Move) : GridState :=
  match move with
  | Move.WhiteToBlack => ⟨state.white_count - 1, state.black_count + 2⟩
  | Move.BlackToWhite => ⟨state.white_count + 2, state.black_count - 1⟩

/-- Applies a list of moves to the grid state -/
def apply_moves (initial_state : GridState) (moves : List Move) : GridState :=
  moves.foldl apply_move initial_state

/-- The main theorem to prove -/
theorem bacteria_distribution (initial_state : GridState) (moves : List Move) : 
  ∃ k : ℤ, (apply_moves initial_state moves).white_count - 
           (apply_moves initial_state moves).black_count = 1 + 3 * k ∨
           (apply_moves initial_state moves).white_count - 
           (apply_moves initial_state moves).black_count = -1 + 3 * k :=
  sorry

/-- Corollary: The difference between white and black cell counts is never zero -/
theorem bacteria_never_equal (initial_state : GridState) (moves : List Move) : 
  (apply_moves initial_state moves).white_count ≠ 
  (apply_moves initial_state moves).black_count :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_distribution_bacteria_never_equal_l410_41060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l410_41036

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 := by sorry

-- Part 2
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l410_41036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l410_41097

/-- An ellipse with major axis twice the length of minor axis passing through (2,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  /-- The ellipse passes through point (2,0) -/
  passes_through_2_0 : x^2 / a^2 + y^2 / b^2 = 1 → (x = 2 ∧ y = 0)
  /-- The major axis is twice the length of the minor axis -/
  major_twice_minor : a = 2 * b
  /-- a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  (∀ x y, x^2 / 4 + y^2 = 1) ∨ (∀ x y, y^2 / 16 + x^2 / 4 = 1)

/-- Theorem: The standard equation of the ellipse is correct -/
theorem ellipse_equation (e : Ellipse) : standard_equation e := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l410_41097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_congruent_to_one_mod_seven_l410_41026

theorem smallest_square_congruent_to_one_mod_seven :
  ∃ (x : ℕ), x > 1 ∧ x^2 % 7 = 1 ∧ ∀ (y : ℕ), y > 1 ∧ y < x → y^2 % 7 ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_congruent_to_one_mod_seven_l410_41026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l410_41045

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l410_41045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l410_41012

/-- Given a hyperbola and a circle with specific properties, prove that m = 2 -/
theorem hyperbola_circle_intersection (a b m : ℝ) : 
  a > 0 → b > 0 → m > 0 →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → ∃ c : ℝ, c = Real.sqrt 2 * a) →
  (∃ x y : ℝ, (x - m)^2 + y^2 = 4) →
  (∃ x y x' y' : ℝ, y = x ∧ (x - m)^2 + y^2 = 4 ∧ 
    y' = x' ∧ (x' - m)^2 + y'^2 = 4 ∧ 
    (x - x')^2 + (y - y')^2 = 8) →
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l410_41012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l410_41066

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The condition for the ellipse equation -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 3 + y^2 / (m + 9) = 1

/-- The theorem stating the possible values of m for the given ellipse -/
theorem ellipse_m_values (m : ℝ) :
  (∃ x y, ellipse_equation x y m) ∧ 
  (∃ a b, a > b ∧ b > 0 ∧ eccentricity a b = 1/2) →
  m = -9/4 ∨ m = 3 := by
  sorry

#check ellipse_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l410_41066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_12_l410_41052

def sequence_to_number (start : ℕ) (stop : ℕ) : ℕ :=
  sorry

def append_to_sequence (seq n : ℕ) : ℕ :=
  sorry

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem smallest_divisible_by_12 :
  ∃ N : ℕ,
    N ≥ 82 ∧
    is_divisible_by_12 (append_to_sequence (sequence_to_number 71 81) N) ∧
    (∀ k : ℕ, 82 ≤ k ∧ k < N →
      ¬is_divisible_by_12 (append_to_sequence (sequence_to_number 71 81) k)) ∧
    N = 84 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_12_l410_41052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_problem_l410_41073

theorem set_average_problem (T : Finset ℕ) (m : ℕ) (h_nonempty : T.Nonempty) :
  (∃ (b₁ bₘ : ℕ), b₁ ∈ T ∧ bₘ ∈ T ∧ 
   (∀ x ∈ T, b₁ ≤ x ∧ x ≤ bₘ) ∧
   (Finset.sum (T.erase bₘ) (fun x => (x : ℚ))) / (m - 1 : ℚ) = 40 ∧
   (Finset.sum (T.erase b₁ ∪ T.erase bₘ) (fun x => (x : ℚ))) / (m - 2 : ℚ) = 43 ∧
   (Finset.sum (T.erase b₁) (fun x => (x : ℚ))) / (m - 1 : ℚ) = 47 ∧
   bₘ = b₁ + 84 ∧
   T.card = m) →
  (Finset.sum T (fun x => (x : ℚ))) / (m : ℚ) = 885 / 20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_average_problem_l410_41073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_needs_six_more_correct_answers_l410_41002

/-- Represents Alex's biology exam results and passing criteria -/
structure BiologyExam where
  total_questions : ℕ
  genetics_questions : ℕ
  ecology_questions : ℕ
  evolution_questions : ℕ
  genetics_correct_rate : ℚ
  ecology_correct_rate : ℚ
  evolution_correct_rate : ℚ
  passing_rate : ℚ

/-- Calculates the number of additional correct answers needed to pass the exam -/
def additional_correct_answers_needed (exam : BiologyExam) : ℕ :=
  let total_correct := (exam.genetics_questions : ℚ) * exam.genetics_correct_rate +
                       (exam.ecology_questions : ℚ) * exam.ecology_correct_rate +
                       (exam.evolution_questions : ℚ) * exam.evolution_correct_rate
  let passing_score := (exam.total_questions : ℚ) * exam.passing_rate
  (passing_score - total_correct).ceil.toNat

/-- Theorem stating that Alex needs 6 more correct answers to pass the exam -/
theorem alex_needs_six_more_correct_answers :
  let exam : BiologyExam := {
    total_questions := 120,
    genetics_questions := 20,
    ecology_questions := 50,
    evolution_questions := 50,
    genetics_correct_rate := 3/5,
    ecology_correct_rate := 1/2,
    evolution_correct_rate := 7/10,
    passing_rate := 13/20
  }
  additional_correct_answers_needed exam = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_needs_six_more_correct_answers_l410_41002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_stereo_fraction_l410_41049

/-- Represents the production of stereos by Company S --/
structure StereoProduction where
  basic : ℝ  -- number of basic stereos
  deluxe : ℝ  -- number of deluxe stereos
  basicTime : ℝ  -- time to produce one basic stereo

/-- The conditions of the stereo production problem --/
class ProductionConditions (p : StereoProduction) where
  deluxe_time : p.basicTime * 1.6 = (p.deluxe * p.basicTime * 1.6) / (p.basic * p.basicTime + p.deluxe * p.basicTime * 1.6)

/-- The theorem stating the fraction of basic stereos produced --/
theorem basic_stereo_fraction (p : StereoProduction) [ProductionConditions p] :
  (p.basic / (p.basic + p.deluxe)) = 8 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_stereo_fraction_l410_41049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_value_l410_41035

theorem smallest_z_value (z w x y : ℕ) 
  (h1 : w^4 + x^4 + y^4 = z^4)
  (h2 : w^4 ≠ x^4 ∧ w^4 ≠ y^4 ∧ w^4 ≠ z^4 ∧ x^4 ≠ y^4 ∧ x^4 ≠ z^4 ∧ y^4 ≠ z^4)
  (h3 : ∃ (n : ℕ), w = 2*n - 2 ∧ x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4)
  (h4 : w < x ∧ x < y ∧ y < z)
  (h5 : ∀ (z' : ℕ), z' < z → ¬(∃ (w' x' y' : ℕ), w'^4 + x'^4 + y'^4 = z'^4 ∧ 
       w'^4 ≠ x'^4 ∧ w'^4 ≠ y'^4 ∧ w'^4 ≠ z'^4 ∧ x'^4 ≠ y'^4 ∧ x'^4 ≠ z'^4 ∧ y'^4 ≠ z'^4 ∧
       (∃ (n' : ℕ), w' = 2*n' - 2 ∧ x' = 2*n' ∧ y' = 2*n' + 2 ∧ z' = 2*n' + 4) ∧
       w' < x' ∧ x' < y' ∧ y' < z'))
  : z = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_value_l410_41035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeep_speed_time_fraction_l410_41025

/-- Proves that given a distance of 440 km, an original time of 3 hours, and a new speed of 293.3333333333333 kmph, the fraction of the previous time needed to cover the same distance is 1/2. -/
theorem jeep_speed_time_fraction (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 440)
  (h2 : original_time = 3)
  (h3 : new_speed = 293.3333333333333) :
  (distance / original_time) / new_speed = 1 / 2 := by
  -- Calculate the original speed
  let original_speed := distance / original_time
  
  -- Prove that the fraction of time is 1/2
  have fraction_eq : original_speed / new_speed = 1 / 2 := by
    -- Insert the proof steps here
    sorry
  
  -- Apply the equality
  exact fraction_eq


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeep_speed_time_fraction_l410_41025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l410_41013

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem axis_of_symmetry : ∀ (x : ℝ), f (-Real.pi/12 + x) = f (-Real.pi/12 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l410_41013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_condition_l410_41098

theorem triangle_angle_condition (A B C : ℝ) : 
  -- Triangle conditions
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi →
  -- The sum condition
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 →
  -- The conclusion
  C = Real.pi / 5 ∨ C = 3 * Real.pi / 5 := by
  sorry

#check triangle_angle_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_condition_l410_41098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_eq_93324_l410_41081

open Nat Real

/-- The number of n-well numbers for a given n -/
def f (n : ℕ) : ℕ :=
  (n.sqrt : ℕ) + (n / 1 : ℕ) - (n / (n.sqrt : ℕ) : ℕ)

/-- The sum of f(n) from 1 to 9999 -/
def sum_f : ℕ :=
  (List.range 9999).map (fun i => f (i + 1)) |>.sum

theorem sum_f_eq_93324 : sum_f = 93324 := by
  sorry

#eval sum_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_eq_93324_l410_41081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_theorem_l410_41007

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define points E, N, F, M on the sides of ABCD
variable (E : ℝ × ℝ)
variable (N : ℝ × ℝ)
variable (F : ℝ × ℝ)
variable (M : ℝ × ℝ)

-- Define scalar values x and y
variable (x y : ℝ)

-- Define the intersection point P
variable (P : ℝ × ℝ)

-- Define vector operations
def vec (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

-- State the conditions
variable (h1 : vec A M = x • (vec M D))
variable (h2 : vec B N = x • (vec N C))
variable (h3 : vec A E = y • (vec E B))
variable (h4 : vec D F = y • (vec F C))

-- State that P is the intersection of MN and EF
variable (h5 : ∃ t1 t2 : ℝ, 
  P = M + t1 • (vec M N) ∧ 
  P = E + t2 • (vec E F))

-- State the theorem to be proved
theorem quadrilateral_intersection_theorem :
  vec E P = x • (vec P F) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_theorem_l410_41007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_of_ice_l410_41080

/-- Converts temperature from Celsius to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

/-- Converts temperature from Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius (f : ℝ) : ℝ := (f - 32) * (5/9)

theorem melting_point_of_ice (boiling_point_c : ℝ) (boiling_point_f : ℝ) 
  (melting_point_c : ℝ) (water_temp_c : ℝ) (water_temp_f : ℝ) :
  boiling_point_c = 100 →
  boiling_point_f = 212 →
  melting_point_c = 0 →
  water_temp_c = 45 →
  water_temp_f = 113 →
  celsius_to_fahrenheit melting_point_c = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_melting_point_of_ice_l410_41080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_binomial_coefficient_l410_41099

theorem middle_term_binomial_coefficient :
  let n : ℕ := 10
  let expansion := (1 - X : Polynomial ℚ) ^ n
  let middle_term_index := (n + 1) / 2
  expansion.coeff middle_term_index = (-1) ^ middle_term_index * n.choose middle_term_index := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_binomial_coefficient_l410_41099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increases_l410_41029

/-- Efficiency function -/
noncomputable def E (p Q m k : ℝ) : ℝ := p * k / (Q + m * k)

theorem efficiency_increases (p Q m : ℝ) (hp : p > 0) (hQ : Q > 0) (hm : m > 0) :
  ∀ k > 0, deriv (E p Q m) k > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increases_l410_41029
