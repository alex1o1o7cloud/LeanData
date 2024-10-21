import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_implies_r_equals_one_l1206_120636

-- Define the cones and sphere
structure Cone where
  radius : ℝ
  height : ℝ

structure Sphere where
  radius : ℝ

-- Define the configuration
structure Configuration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphere : Sphere
  r : ℝ
  cone_heights_equal : cone1.height = cone2.height ∧ cone2.height = cone3.height
  cone_radii : cone1.radius = 2 * r ∧ cone2.radius = 3 * r ∧ cone3.radius = 10 * r
  sphere_radius : sphere.radius = 2
  cones_touch : Prop
  sphere_touches_cones : Prop
  center_equidistant : Prop

-- Theorem statement
theorem configuration_implies_r_equals_one (config : Configuration) : config.r = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_implies_r_equals_one_l1206_120636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1206_120634

/-- Curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 9

/-- Line l in polar coordinates -/
def line_l (ρ θ m : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (θ - Real.pi/4) = m

/-- Distance from a point to a line in rectangular coordinates -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem curve_and_line_intersection :
  ∀ m : ℝ,
  (∃ x y : ℝ, curve_C x y) →
  (∃ ρ θ : ℝ, line_l ρ θ m) →
  (distance_point_to_line 1 (-2) 1 (-1) m = 2) →
  (m = -3 + 2 * Real.sqrt 2 ∨ m = -3 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1206_120634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1206_120619

theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, Real.sqrt 3 * Real.sin x - Real.cos x = 4 - m) ↔ 2 ≤ m ∧ m ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1206_120619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BD4_hex_to_base4_l1206_120650

/-- Converts a single hexadecimal digit to base 4 -/
def hex_to_base4 (hex_digit : Nat) : Nat × Nat :=
  (hex_digit / 4, hex_digit % 4)

/-- Converts a hexadecimal number to base 4 -/
def hex_to_base4_number (hex : List Nat) : List Nat :=
  hex.bind (fun d => let (q, r) := hex_to_base4 d; [q, r])

theorem BD4_hex_to_base4 :
  hex_to_base4_number [11, 13, 4] = [2, 3, 3, 1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BD4_hex_to_base4_l1206_120650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_locus_characterization_l1206_120687

/-- A convex quadrilateral in a 2D plane -/
structure ConvexQuadrilateral where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  D : EuclideanSpace ℝ (Fin 2)
  convex : Convex ℝ {A, B, C, D}

/-- The area of a triangle formed by three points -/
noncomputable def triangleArea (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- A conic section in a 2D plane -/
structure Conic where
  -- Define the conic (e.g., by its equation or other means)
  points : Set (EuclideanSpace ℝ (Fin 2))

/-- The locus of points satisfying the area condition in a convex quadrilateral -/
def areaConditionLocus (quad : ConvexQuadrilateral) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {P | P ∈ interior {quad.A, quad.B, quad.C, quad.D} ∧
       (triangleArea P quad.A quad.B) * (triangleArea P quad.C quad.D) =
       (triangleArea P quad.B quad.C) * (triangleArea P quad.D quad.A)}

/-- The theorem stating the locus is the union of a conic and two lines -/
theorem area_condition_locus_characterization (quad : ConvexQuadrilateral) :
  ∃ (conic : Conic),
    areaConditionLocus quad =
      {P | P ∈ conic.points ∨ P ∈ affineSpan ℝ {quad.A, quad.C} ∨ P ∈ affineSpan ℝ {quad.B, quad.D}} ∧
    quad.A ∈ conic.points ∧ quad.B ∈ conic.points ∧ quad.C ∈ conic.points ∧ quad.D ∈ conic.points :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_condition_locus_characterization_l1206_120687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1206_120603

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def line (x y m : ℝ) : Prop := y = x + m

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem ellipse_line_intersection :
  ∀ (x₁ y₁ x₂ y₂ m : ℝ),
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧  -- A and B are on the ellipse
  line x₁ y₁ m ∧ line x₂ y₂ m ∧    -- A and B are on the line
  distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 2 / 3 →  -- |AB| = 4√2/3
  m = 1 ∨ m = -1  -- The y-intercept of the line is either 1 or -1
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1206_120603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explosion_height_approx_l1206_120607

/-- The height at which a projectile explodes given initial conditions --/
noncomputable def explosion_height (initial_velocity : ℝ) (sound_speed : ℝ) (gravity : ℝ) (time_heard : ℝ) : ℝ :=
  let t_explosion := ((-initial_velocity - sound_speed) + 
    Real.sqrt ((initial_velocity + sound_speed)^2 + 2 * gravity * sound_speed * time_heard)) / 
    (-gravity)
  sound_speed * (time_heard - t_explosion)

/-- Theorem stating that the explosion height is approximately 431.51 meters --/
theorem explosion_height_approx :
  let c := 99  -- initial velocity in m/s
  let v_sound := 333  -- speed of sound in m/s
  let g := 9.806  -- acceleration due to gravity in m/s²
  let t := 5  -- time when explosion is heard in seconds
  abs (explosion_height c v_sound g t - 431.51) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explosion_height_approx_l1206_120607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l1206_120654

/-- The semi-major axis of the ellipse -/
def a : ℝ := 5

/-- The semi-minor axis of the ellipse -/
def b : ℝ := 4

/-- The distance from the center to a focus of the ellipse -/
noncomputable def c : ℝ := Real.sqrt (a^2 - b^2)

/-- The x-coordinate of the point where the circle touches the ellipse -/
def x : ℝ := a

/-- The radius of the circle -/
noncomputable def r : ℝ := x - c

/-- Theorem stating that the radius of the tangent circle is 2 -/
theorem ellipse_tangent_circle_radius : r = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_radius_l1206_120654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_zero_at_pi_over_four_l1206_120675

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - (Real.cos x ^ 2 - Real.sin x ^ 2) - 1

-- Theorem for the minimum value and period of f
theorem f_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x) ∧
  (∀ (x : ℝ), f x = f (x + π)) ∧
  (∀ (p : ℝ), 0 < p → p < π → ∃ (x : ℝ), f x ≠ f (x + p)) :=
sorry

-- Theorem for the value of C when f(C) = 0
theorem f_zero_at_pi_over_four :
  f (π / 4) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_zero_at_pi_over_four_l1206_120675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_for_y_symmetric_angles_l1206_120638

-- Define a property for angles being symmetric with respect to the y-axis
def symmetric_to_y_axis (α β : ℝ) : Prop :=
  ∃ (p q : ℝ × ℝ),
    p.1 = Real.cos α ∧ p.2 = Real.sin α ∧
    q.1 = Real.cos β ∧ q.2 = Real.sin β ∧
    p.1 = -q.1 ∧ p.2 = q.2

theorem sine_equality_for_y_symmetric_angles (α β : ℝ) :
  symmetric_to_y_axis α β → Real.sin α = Real.sin β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_for_y_symmetric_angles_l1206_120638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l1206_120629

/-- Proves that given a discount of 16% and a gain of 31.25% after the discount,
    the cost price is 64% of the marked price. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : MP > 0) (h2 : CP > 0) : 
  let discount : ℝ := 0.16
  let gain : ℝ := 0.3125
  let SP : ℝ := MP * (1 - discount)
  CP * (1 + gain) = SP
  → CP / MP = 0.64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l1206_120629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_current_age_l1206_120623

-- Define the ages of the people
def maya_age : ℕ := sorry
def drew_age : ℕ := sorry
def peter_age : ℕ := sorry
def john_age : ℕ := sorry
def jacob_age : ℕ := sorry

-- Define the conditions
axiom drew_maya : drew_age = maya_age + 5
axiom peter_drew : peter_age = drew_age + 4
axiom john_maya : john_age = 30 ∧ john_age = 2 * maya_age
axiom jacob_peter_future : jacob_age + 2 = (peter_age + 2) / 2

-- Theorem to prove
theorem jacob_current_age : jacob_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacob_current_age_l1206_120623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_l1206_120645

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum : i^10 + i^20 + i^(-30 : ℤ) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_l1206_120645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l1206_120632

/-- An equilateral triangle with a specific interior point -/
structure SpecialTriangle where
  -- D, E, F are the vertices of the equilateral triangle
  D : EuclideanSpace ℝ (Fin 2)
  E : EuclideanSpace ℝ (Fin 2)
  F : EuclideanSpace ℝ (Fin 2)
  -- Q is the interior point
  Q : EuclideanSpace ℝ (Fin 2)
  -- The triangle is equilateral
  equilateral : dist D E = dist D F ∧ dist D F = dist E F
  -- Distances from Q to the vertices
  dist_QD : dist Q D = 9
  dist_QE : dist Q E = 12
  dist_QF : dist Q F = 15

/-- The area of the special triangle -/
noncomputable def area (t : SpecialTriangle) : ℝ :=
  (225 * Real.sqrt 3) / 4

/-- Theorem stating that the area of the special triangle is as defined -/
theorem special_triangle_area (t : SpecialTriangle) : 
  area t = (225 * Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_l1206_120632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parallelogram_points_l1206_120622

noncomputable section

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 3/2)

-- Define the condition that P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the condition about the sum of distances from P to foci
axiom foci_distance_sum : ∃ (f1 f2 : ℝ × ℝ), 
  Real.sqrt ((P.1 - f1.1)^2 + (P.2 - f1.2)^2) + 
  Real.sqrt ((P.1 - f2.1)^2 + (P.2 - f2.2)^2) = 4

-- Define a function to check if POMN is a parallelogram
def is_parallelogram (M N : ℝ × ℝ) : Prop :=
  M.1 + N.1 = P.1 ∧ M.2 + N.2 = P.2

-- The main theorem
theorem ellipse_parallelogram_points : 
  ∃ M N : ℝ × ℝ, 
    ellipse M.1 M.2 ∧
    ellipse N.1 N.2 ∧
    is_parallelogram M N ∧
    ((M = (-1, 3/2) ∧ N = (-2, 0)) ∨ (M = (1, 9/2) ∧ N = (2, 6))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parallelogram_points_l1206_120622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1206_120667

/-- The time it takes for A to complete the work alone -/
def A : ℝ := 8

/-- The time it takes for B to complete the work alone -/
def B : ℝ := 6

/-- The time it takes for C to complete the work alone -/
def C : ℝ := 4.8

/-- The time it takes for A, B, and C to complete the work together -/
def ABC : ℝ := 2

theorem work_completion_time : A = 8 := by
  -- Define the equation
  have h1 : 1/A + 1/B + 1/C = 1/ABC := by
    -- Proof of the equation
    sorry
  
  -- Solve the equation for A
  have h2 : 1/A = 1/ABC - 1/B - 1/C := by
    -- Algebraic manipulation
    sorry
  
  -- Show that A = 8 satisfies the equation
  have h3 : 1/8 = 1/2 - 1/6 - 1/4.8 := by
    -- Numerical calculation
    sorry
  
  -- Conclude that A = 8
  exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1206_120667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1206_120620

/-- Curve C in the xy-plane -/
def C : Set (ℝ × ℝ) :=
  {p | p.1^2 / 5 + p.2^2 = 1}

/-- Line l in the xy-plane -/
def l : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - 2}

/-- Point P -/
def P : ℝ × ℝ := (0, -2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem to be proved -/
theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    A ∈ C ∧ A ∈ l ∧
    B ∈ C ∧ B ∈ l ∧
    A ≠ B ∧
    distance P A + distance P B = 10 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1206_120620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_double_sum_zero_l1206_120680

open Real

theorem cos_sin_sum_zero_implies_cos_double_sum_zero
  (x y z : ℝ)
  (h1 : Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0)
  (h2 : Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_double_sum_zero_l1206_120680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1206_120676

theorem max_value_theorem (k : ℕ) (a b c : ℝ) 
  (h_pos_k : k > 0)
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3 * k) :
  a^(3*k-1)*b + b^(3*k-1)*c + c^(3*k-1)*a + k^2*a^k*b^k*c^k ≤ (3*k-1)^(3*k-1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1206_120676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_30_value_l1206_120671

-- Define the selling price for 40% profit
noncomputable def selling_price_40 : ℝ := 2412.31

-- Define the profit percentages
def profit_40 : ℝ := 0.40
def profit_30 : ℝ := 0.30

-- Define the cost of the computer
noncomputable def cost : ℝ := selling_price_40 / (1 + profit_40)

-- Define the selling price for 30% profit
noncomputable def selling_price_30 : ℝ := cost * (1 + profit_30)

-- Theorem to prove
theorem selling_price_30_value : 
  ∃ ε > 0, |selling_price_30 - 2240.00| < ε :=
by
  -- We'll use ε = 0.01 as our error margin
  use 0.01
  -- Split the goal into two parts: ε > 0 and |selling_price_30 - 2240.00| < ε
  constructor
  · -- Prove ε > 0
    norm_num
  · -- Prove |selling_price_30 - 2240.00| < ε
    -- This step would require actual computation, which is complex in Lean
    -- For now, we'll use sorry to skip the proof
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_30_value_l1206_120671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_weight_l1206_120662

theorem textbook_weight (bookcase_limit : ℝ) (num_hardcover : ℕ) (hardcover_weight : ℝ)
  (num_textbooks : ℕ) (num_knickknacks : ℕ) (knickknack_weight : ℝ) (over_limit : ℝ)
  (textbook_weight : ℝ)
  (h1 : bookcase_limit = 80)
  (h2 : num_hardcover = 70)
  (h3 : hardcover_weight = 0.5)
  (h4 : num_textbooks = 30)
  (h5 : num_knickknacks = 3)
  (h6 : knickknack_weight = 6)
  (h7 : over_limit = 33)
  (h8 : num_hardcover * hardcover_weight + num_textbooks * textbook_weight + num_knickknacks * knickknack_weight = bookcase_limit + over_limit) :
  textbook_weight = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_textbook_weight_l1206_120662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boundedArea_eq_pi_squared_div_four_minus_two_l1206_120633

open Real MeasureTheory

/-- The area of the figure bounded by y = x^2 * cos(x), y = 0, and 0 ≤ x ≤ π/2 -/
noncomputable def boundedArea : ℝ :=
  ∫ x in (0)..(π/2), x^2 * cos x

/-- The theorem stating that the bounded area is equal to π^2/4 - 2 -/
theorem boundedArea_eq_pi_squared_div_four_minus_two :
  boundedArea = π^2/4 - 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boundedArea_eq_pi_squared_div_four_minus_two_l1206_120633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l1206_120652

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 10*x - 12*y + 45 = 0

/-- The given line -/
def given_line (x y : ℝ) : Prop :=
  y = 3*x

/-- The first tangent line -/
def tangent1 (x y : ℝ) : Prop :=
  y = 3*x - 9 + 4*Real.sqrt 10

/-- The second tangent line -/
def tangent2 (x y : ℝ) : Prop :=
  y = 3*x - 9 - 4*Real.sqrt 10

/-- Theorem stating that the two lines are tangent to the circle and parallel to the given line -/
theorem tangent_lines_theorem :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
    tangent1 x1 y1 ∧ tangent2 x2 y2 ∧
    (∀ (x y : ℝ), tangent1 x y → ∃ (k : ℝ), y = 3*x + k) ∧
    (∀ (x y : ℝ), tangent2 x y → ∃ (k : ℝ), y = 3*x + k) :=
by
  sorry

/-- Helper lemma: The tangent lines are indeed tangent to the circle -/
lemma tangent_touch_circle :
  ∀ (x y : ℝ),
    (tangent1 x y → circle_equation x y) ∧
    (tangent2 x y → circle_equation x y) :=
by
  sorry

/-- Helper lemma: The tangent lines are parallel to the given line -/
lemma tangents_parallel_to_given :
  ∀ (x y : ℝ),
    (tangent1 x y → ∃ (k : ℝ), y = 3*x + k) ∧
    (tangent2 x y → ∃ (k : ℝ), y = 3*x + k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l1206_120652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_point_f_at_neg_two_l1206_120648

/-- An exponential function that passes through (1, 1/2) -/
noncomputable def f (x : ℝ) : ℝ := (1/2)^x

/-- The function passes through (1, 1/2) -/
theorem f_passes_through_point : f 1 = 1/2 := by
  -- Proof steps would go here
  sorry

/-- The value of f(-2) is 4 -/
theorem f_at_neg_two : f (-2) = 4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_point_f_at_neg_two_l1206_120648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_neg_i_l1206_120669

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_expression_equals_neg_i :
  i^3 + i^11 + i^(-17 : ℤ) + 2*i = -i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_neg_i_l1206_120669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1206_120647

noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_calculation : 
  diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l1206_120647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_touches_both_curves_l1206_120694

open Real

-- Define the original curve
noncomputable def f (x : ℝ) : ℝ := x + log x

-- Define the second curve with parameter a
def g (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

-- Define the tangent line to f at x = 1
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem tangent_line_touches_both_curves : 
  ∃ a : ℝ, a = 8 ∧ 
  (∀ x : ℝ, g a x = tangent_line x → x = 1) ∧
  (tangent_line 1 = f 1) ∧
  ((deriv f) 1 = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_touches_both_curves_l1206_120694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1206_120624

/-- The distance from a point (x₀, y₀) to a line ax + by + c = 0 --/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1206_120624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_of_integer_convex_polyhedron_l1206_120665

-- Define a 3D integer point
def IntegerPoint := ℤ × ℤ × ℤ

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : Finset IntegerPoint
  is_convex : Bool
  no_other_integer_points : Bool

-- Theorem statement
theorem max_vertices_of_integer_convex_polyhedron 
  (P : ConvexPolyhedron) 
  (h1 : P.is_convex = true) 
  (h2 : P.no_other_integer_points = true) : 
  P.vertices.card ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_of_integer_convex_polyhedron_l1206_120665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mount_xianv_measurements_l1206_120681

noncomputable def initial_temp : ℝ := 14
def temp_changes : List ℝ := [-3.8, 1.4, -3.3, -2.9, 1.5, -3.1]
noncomputable def temp_decrease_per_100m : ℝ := 0.5

noncomputable def temp_at_second_measurement : ℝ := initial_temp + temp_changes[0]! + temp_changes[1]!

noncomputable def total_temp_change : ℝ := initial_temp - (initial_temp + temp_changes.sum)

noncomputable def vertical_height : ℝ := (total_temp_change / temp_decrease_per_100m) * 100

theorem mount_xianv_measurements :
  temp_at_second_measurement = 11.6 ∧ vertical_height = 2040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mount_xianv_measurements_l1206_120681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_triangle_area_l1206_120692

-- Define necessary predicates and functions
def is_square (s : Set (ℝ × ℝ)) : Prop := sorry
def inscribed (inner outer : Set (ℝ × ℝ)) : Prop := sorry
def is_triangle (t : Set (ℝ × ℝ)) : Prop := sorry
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem inscribed_squares_triangle_area :
  ∀ (outer_square inner_square : Set (ℝ × ℝ)) 
    (outer_perimeter inner_perimeter : ℝ),
  is_square outer_square →
  is_square inner_square →
  inscribed inner_square outer_square →
  outer_perimeter = 28 →
  inner_perimeter = 20 →
  ∃ (triangle : Set (ℝ × ℝ)),
    is_triangle triangle ∧
    triangle ⊆ (outer_square \ inner_square) ∧
    area triangle = 6 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_triangle_area_l1206_120692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l1206_120641

-- Define the necessary geometric objects
structure GeometricObjects where
  S1 : Set (ℝ × ℝ)  -- Circle S1
  S2 : Set (ℝ × ℝ)  -- Circle S2
  A : ℝ × ℝ         -- Point A
  l : Set (ℝ × ℝ)   -- Line l

-- Define what it means for a line to be tangent to a circle at a point
def isTangent (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Prop :=
  point ∈ line ∧ point ∈ circle ∧ 
  ∀ p ∈ line, p ≠ point → p ∉ circle

-- Define what it means for two circles to be tangent
def areCirclesTangent (circle1 : Set (ℝ × ℝ)) (circle2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧
  ∀ q : ℝ × ℝ, q ≠ p → (q ∈ circle1 → q ∉ circle2) ∧ (q ∈ circle2 → q ∉ circle1)

-- Theorem statement
theorem tangent_circle_existence (obj : GeometricObjects) 
  (h1 : obj.A ∈ obj.S1)
  (h2 : isTangent obj.l obj.S1 obj.A) :
  ∃ S : Set (ℝ × ℝ), 
    isTangent obj.l S obj.A ∧ 
    areCirclesTangent S obj.S1 ∧ 
    areCirclesTangent S obj.S2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_existence_l1206_120641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l1206_120611

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 4 →
  ∃ k b : ℝ, ∀ x y : ℝ,
    y = k * (x - 1) + f 4 1 ↔ 2 * x + y - 2 = 0 :=
by sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x > 0) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l1206_120611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_for_y_one_third_l1206_120672

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 3) / (3 * x + 2)

-- State the theorem
theorem no_x_for_y_one_third :
  ∀ x : ℝ, x ≠ -2/3 → f x ≠ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_for_y_one_third_l1206_120672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_theorem_l1206_120600

noncomputable def possible_values (A : ℝ) (n : ℕ+) : Set ℝ :=
  let series := {x : ℕ → ℝ | ∀ i, x i > 0 ∧ HasSum x A}
  let sum_of_powers (x : ℕ → ℝ) := ∑' i, (x i) ^ (n : ℝ)
  Set.Icc 0 (A ^ (n : ℝ))

theorem possible_values_theorem (A : ℝ) (n : ℕ+) (h : A > 0) :
  possible_values A n = if n = 1 then {A} else Set.Ioo 0 (A ^ (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_theorem_l1206_120600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120688

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x - 1) - x

noncomputable def g (x : ℝ) : ℝ := (1 / Real.exp 1 - 1) * x

theorem f_properties :
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f z1 = 0 ∧ f z2 = 0) ∧
  (∀ x, f x ≥ g x) ∧
  (∀ a x1 x2, x1 < x2 → f x1 = a → f x2 = a → |x1 - x2| ≤ ((1 - 2 * Real.exp 1) * a) / (1 - Real.exp 1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalue_sum_l1206_120684

/-- Given a 2x2 matrix M with eigenvalue 2 and eigenvector [2, 1], prove that the sum of its off-diagonal elements is 6. -/
theorem eigenvalue_sum (a b : ℝ) : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; -1, b]
  (M.vecMul (![2, 1] : (Fin 2 → ℝ)) = (2 : ℝ) • (![2, 1] : (Fin 2 → ℝ))) →
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eigenvalue_sum_l1206_120684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1206_120664

noncomputable section

/-- Definition of the function f -/
noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - Real.pi) * Real.cos (2*Real.pi - α) * Real.sin (-α + 3*Real.pi/2) * Real.sin (5*Real.pi/2 + α)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

/-- Theorem stating the simplification of f and its value under a specific condition -/
theorem f_simplification_and_value :
  (∀ α, f α = -Real.cos (2*α)) ∧
  (∀ α, Real.cos (5*Real.pi/6 + 2*α) = 1/3 → f (Real.pi/12 - α) = -1/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_and_value_l1206_120664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_a_statement_b_statement_c_statement_d_not_always_true_l1206_120602

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Statement A
theorem statement_a (t : Triangle) : t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Statement B
def acute_triangle (t : Triangle) : Prop := t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

theorem statement_b (t : Triangle) : acute_triangle t → Real.sin t.A > Real.cos t.B := by
  sorry

-- Statement C
theorem statement_c (t : Triangle) : t.a^2 + t.b^2 < t.c^2 → t.C > Real.pi/2 := by
  sorry

-- Statement D (counterexample)
theorem statement_d_not_always_true : ¬ ∀ (t : Triangle), Real.sin (2*t.A) = Real.sin (2*t.B) → t.a = t.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_a_statement_b_statement_c_statement_d_not_always_true_l1206_120602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_price_two_pounds_non_organic_is_21_6_l1206_120614

/-- The regular price of two pounds of non-organic chicken -/
noncomputable def regular_price_two_pounds_non_organic (discounted_price_organic : ℝ) 
  (discount_percentage : ℝ) (non_organic_discount : ℝ) : ℝ :=
  let regular_price_organic := discounted_price_organic / (1 - discount_percentage)
  let price_non_organic := regular_price_organic * (1 - non_organic_discount)
  2 * price_non_organic

/-- Theorem stating the regular price of two pounds of non-organic chicken -/
theorem regular_price_two_pounds_non_organic_is_21_6 :
  regular_price_two_pounds_non_organic 9 0.25 0.1 = 21.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_price_two_pounds_non_organic_is_21_6_l1206_120614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_theorem_l1206_120625

/-- Represents the water depth after placing an iron block in a container. -/
noncomputable def water_depth_after_block (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 9 then (10/9) * a
  else if 9 ≤ a ∧ a < 49 then a + 1
  else if 49 ≤ a ∧ a ≤ 50 then 50
  else 0  -- undefined for other values of a

/-- Theorem stating the water depth after placing an iron block in a container. -/
theorem water_depth_theorem (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≤ 50) 
  (container_length : ℝ) (container_width : ℝ) (container_height : ℝ)
  (block_edge : ℝ)
  (h3 : container_length = 40)
  (h4 : container_width = 25)
  (h5 : container_height = 50)
  (h6 : block_edge = 10) :
  water_depth_after_block a = 
    if 0 < a ∧ a < 9 then (10/9) * a
    else if 9 ≤ a ∧ a < 49 then a + 1
    else 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_theorem_l1206_120625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1206_120679

-- Define the equation
def equation (x : ℝ) : Prop :=
  ((10 * x - 1) ^ (1/3 : ℝ) + (8 * x + 1) ^ (1/3 : ℝ) = 3 * x ^ (1/3 : ℝ))

-- Define the set of solutions
noncomputable def solution_set : Set ℝ :=
  {0, (2 + 6 * Real.sqrt 6) / 106, (2 - 6 * Real.sqrt 6) / 106}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1206_120679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l1206_120615

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 5 →
  e = -2*a - c →
  a + b*Complex.I + c + d*Complex.I + e + f*Complex.I = 3 + 2*Complex.I →
  d + 3*f = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_problem_l1206_120615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1206_120695

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point P inside triangle ABC -/
def Point := ℝ × ℝ

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The minimum value of (a·PA + b·PB + c·PC) / S is 4 -/
theorem min_value_theorem (t : Triangle) (P : Point) :
  let S := area t
  let PA := distance P t.A
  let PB := distance P t.B
  let PC := distance P t.C
  (t.a * PA + t.b * PB + t.c * PC) / S ≥ 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1206_120695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l1206_120609

/-- Given a cube and a regular tetrahedron sharing 4 vertices, 
    the ratio of their surface areas is √3 : 1 -/
theorem cube_tetrahedron_surface_area_ratio :
  ∀ (cube_edge : ℝ) (cube_area tetra_area : ℝ),
  cube_edge > 0 →
  cube_area = 6 * cube_edge^2 →
  tetra_area = 2 * Real.sqrt 3 * cube_edge^2 →
  (cube_area : ℝ) / tetra_area = Real.sqrt 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_tetrahedron_surface_area_ratio_l1206_120609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tradesman_gain_percentage_l1206_120682

/-- Represents the percentage of fraud applied by the tradesman when buying or selling goods -/
noncomputable def fraudPercentage : ℝ := 20

/-- Calculates the actual price paid by the tradesman when buying goods -/
noncomputable def actualBuyingPrice (trueValue : ℝ) : ℝ :=
  trueValue * (1 - fraudPercentage / 100)

/-- Calculates the selling price of goods by the tradesman -/
noncomputable def sellingPrice (trueValue : ℝ) : ℝ :=
  trueValue * (1 + fraudPercentage / 100)

/-- Calculates the gain percentage on the outlay -/
noncomputable def gainPercentage (buyingPrice sellingPrice : ℝ) : ℝ :=
  ((sellingPrice - buyingPrice) / buyingPrice) * 100

/-- Theorem stating that the tradesman's gain percentage is 50% -/
theorem tradesman_gain_percentage (trueValue : ℝ) (h : trueValue > 0) :
  gainPercentage (actualBuyingPrice trueValue) (sellingPrice trueValue) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tradesman_gain_percentage_l1206_120682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l1206_120610

/-- Calculates the overall average marks per student for multiple sections -/
def overallAverage (students : List ℕ) (means : List ℚ) : ℚ :=
  (List.sum (List.zipWith (λ s m => (s : ℚ) * m) students means)) / (List.sum students : ℚ)

/-- Theorem: The overall average for the given sections matches the expected result -/
theorem chemistry_class_average :
  let students : List ℕ := [40, 35, 45, 42]
  let means : List ℚ := [50, 60, 55, 45]
  overallAverage students means = 8465 / 162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_class_average_l1206_120610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_150_l1206_120630

-- Define the total revenue function
noncomputable def H (x : ℝ) : ℝ :=
  if x ≤ 200 then 400 * x - x^2 else 40000

-- Define the total cost function
def T (x : ℝ) : ℝ := 10000 + 100 * x

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := H x - T x

-- Theorem statement
theorem max_profit_at_150 :
  ∃ (max_profit : ℝ), max_profit = 12500 ∧
  ∀ (x : ℝ), x ≥ 0 → f x ≤ max_profit ∧
  f 150 = max_profit := by
  sorry

#check max_profit_at_150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_150_l1206_120630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1206_120668

-- Define the function g
noncomputable def g (A : ℝ) : ℝ := (Real.cos A ^ 2 + 2) ^ 2 / (Real.cos A ^ 2 - 2 * Real.sin A ^ 4)

-- State the theorem
theorem range_of_g :
  ∀ A : ℝ, (∀ n : ℤ, A ≠ n * Real.pi / 2) →
  ∃ y ∈ Set.Ioo 4 5, y = g A :=
by
  sorry

-- Alternatively, you can use this more concise syntax:
-- theorem range_of_g :
--   ∀ A : ℝ, (∀ n : ℤ, A ≠ n * Real.pi / 2) →
--   ∃ y ∈ Set.Ioo 4 5, y = g A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l1206_120668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1206_120621

-- Define the ellipse C
structure Ellipse where
  center : ℝ × ℝ
  foci_distance : ℝ
  major_minor_ratio : ℝ

-- Define the line l
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define the triangle ABP
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ

def ellipse_equation (e : Ellipse) : Prop :=
  e.center = (0, 0) ∧
  e.foci_distance = 2 ∧
  e.major_minor_ratio = Real.sqrt 2

noncomputable def triangle_area (t : Triangle) : ℝ :=
  Real.sqrt 10 / 2

theorem ellipse_and_line_theorem (C : Ellipse) (l : Line) (ABP : Triangle) :
  ellipse_equation C →
  ABP.P = (2, 0) →
  triangle_area ABP = Real.sqrt 10 / 2 →
  (∃ (x y : ℝ), x^2 / 2 + y^2 = 1) ∧
  (l.slope = 2 ∨ l.slope = -2) ∧
  l.y_intercept = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1206_120621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1206_120683

noncomputable def a : ℝ := (5 : ℝ) ^ (0.8 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ) ^ (5 : ℝ)
noncomputable def c : ℝ := Real.log 0.8 / Real.log 5

theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1206_120683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_and_decreasing_f_l1206_120651

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x - (1/3) * x

-- State the theorem
theorem max_and_decreasing_f {x₀ : ℝ} (h₀ : x₀ ∈ Set.Icc 0 π) (h₁ : cos x₀ = 1/3) :
  (∀ x ∈ Set.Icc 0 π, f x ≤ f x₀) ∧
  (∀ x ∈ Set.Icc x₀ π, ∀ y ∈ Set.Icc x₀ π, x ≤ y → f x ≥ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_and_decreasing_f_l1206_120651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120649

noncomputable section

/-- The function f(x) = sin(2x + π/6) - 2cos²x --/
def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) - 2 * (Real.cos x) ^ 2

/-- The interval [-π/3, π/6] --/
def I : Set ℝ := {x | -Real.pi/3 ≤ x ∧ x ≤ Real.pi/6}

theorem f_properties :
  f (Real.pi/6) = -1/2 ∧
  (∀ x ∈ I, f x ≤ 0) ∧
  (∃ x ∈ I, f x = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1206_120617

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

-- Theorem for part (I)
theorem part_one (α : ℝ) (h : f α = 2/3) :
  f (α - π/12) = (2 * sqrt 3 + sqrt 5) / 6 ∨
  f (α - π/12) = (2 * sqrt 3 - sqrt 5) / 6 :=
sorry

-- Theorem for part (II)
theorem part_two (A B C : ℝ) (h1 : f A = sqrt 3 / 2) (h2 : B = π/4) (h3 : 2 = cos A * 2) :
  (3 - sqrt 3) / 2 = 1/2 * 2 * (sqrt 3 - 1) * sqrt 3 / 2 ∨
  2 = 1/2 * 2 * sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1206_120617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_b_c_values_l1206_120612

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ Real.cos t.B = 3/5

-- Theorem 1
theorem sin_A_value (t : Triangle) (h : triangle_conditions t) (hb : t.b = 4) :
  Real.sin t.A = 2/5 := by sorry

-- Theorem 2
theorem b_c_values (t : Triangle) (h : triangle_conditions t) (harea : (1/2) * t.a * t.c * Real.sin t.B = 4) :
  t.c = 5 ∧ t.b = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_A_value_b_c_values_l1206_120612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_l1206_120698

/-- The function representing x^4 + 2/x^2 -/
noncomputable def f (x : ℝ) : ℝ := x^4 + 2/x^2

/-- The system of equations -/
def system (n : ℕ) (p : ℝ) (x : ℕ → ℝ) : Prop :=
  n ≥ 2 ∧ ∀ i, i ∈ Finset.range n → f (x i) = p * x ((i + 1) % n)

/-- The theorem stating the condition for multiple solutions -/
theorem multiple_solutions (n : ℕ) :
  n ≥ 2 →
  (∃ p : ℝ, ∃ x y : ℕ → ℝ, x ≠ y ∧ system n p x ∧ system n p y) ↔ 
  (∃ p : ℝ, p < -2 * Real.sqrt 2 ∨ p > 2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_l1206_120698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cut_length_squared_l1206_120627

/-- Represents a right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

/-- The process of cutting a right triangle along its altitude to the hypotenuse -/
def cut_process (t : RightTriangle) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sorry  -- Definition of the cutting process

/-- The expected value of the total length of cuts -/
noncomputable def expected_cut_length (t : RightTriangle) : ℝ := sorry

theorem expected_cut_length_squared (t : RightTriangle) 
  (h1 : t.a = 3) (h2 : t.b = 4) (h3 : t.c = 5) : 
  (expected_cut_length t)^2 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_cut_length_squared_l1206_120627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_expression_l1206_120601

theorem min_value_of_exponential_expression :
  (∀ x : ℝ, (16 : ℝ)^x - (4 : ℝ)^x - (4 : ℝ)^(x+1) + 3 ≥ -4) ∧
  (∃ x : ℝ, (16 : ℝ)^x - (4 : ℝ)^x - (4 : ℝ)^(x+1) + 3 = -4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_expression_l1206_120601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_tangent_l1206_120631

/-- The value of n for which the ellipse 4x^2 + y^2 = 4 and the hyperbola x^2 - n(y - 1)^2 = 1 are tangent -/
noncomputable def tangent_n : ℝ := 3/2

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

/-- Definition of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := x^2 - n * (y - 1)^2 = 1

/-- Theorem stating that the ellipse and hyperbola are tangent when n = 3/2 -/
theorem ellipse_hyperbola_tangent :
  ∃ (x y : ℝ), is_on_ellipse x y ∧ is_on_hyperbola x y tangent_n ∧
  (∀ (x' y' : ℝ), is_on_ellipse x' y' ∧ is_on_hyperbola x' y' tangent_n → (x', y') = (x, y)) := by
  sorry

#check ellipse_hyperbola_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_hyperbola_tangent_l1206_120631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120642

noncomputable def f (x : ℝ) : ℝ := Real.tan (1/3 * x - Real.pi/6)

theorem f_properties :
  let T := 3 * Real.pi
  (∀ x : ℝ, f (x + T) = f x) ∧  -- smallest positive period
  f (3 * Real.pi / 2) = Real.sqrt 3 ∧
  (∀ α : ℝ, f (3 * α + 7 * Real.pi / 2) = -1/2 →
    (Real.sin (Real.pi - α) + Real.cos (α - Real.pi)) / (Real.sqrt 2 * Real.sin (α + Real.pi / 4)) = -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1206_120642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l1206_120696

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (2018 + log x)

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) : 
  (deriv f x₀ = 2019) → x₀ = 1 := by
  intro h_deriv
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l1206_120696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_event_probabilities_l1206_120628

/-- Probability of an event occurring -/
noncomputable def prob (A : Prop) : ℝ := sorry

/-- Independence of events -/
def independent (A B : Prop) : Prop := sorry

axiom prob_complement (A : Prop) : prob (¬A) = 1 - prob A

axiom prob_or_independent (A B : Prop) (h : independent A B) :
  prob (A ∨ B) = prob A + prob B - prob A * prob B

theorem three_event_probabilities 
  (A1 A2 A3 : Prop)
  (h_ind12 : independent A1 A2)
  (h_ind13 : independent A1 A3)
  (h_ind23 : independent A2 A3)
  (h_prob1 : prob A1 = 0.8)
  (h_prob2 : prob A2 = 0.7)
  (h_prob3 : prob A3 = 0.6) :
  (prob (A1 ∨ A2 ∨ A3) = 0.28) ∧
  (prob ((A1 ∧ A2) ∨ (A1 ∧ A3) ∨ (A2 ∧ A3)) = 0.54) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_event_probabilities_l1206_120628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_is_linear_l1206_120686

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Sequence defined by a_0 = n and a_k = P(a_{k-1}) -/
def sequenceP (P : IntPolynomial) (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => (P (sequenceP P n k)).toNat

/-- P is non-constant -/
def NonConstant (P : IntPolynomial) : Prop :=
  ∃ x y, P x ≠ P y

theorem polynomial_is_linear
  (P : IntPolynomial)
  (h_non_const : NonConstant P)
  (h_sequence : ∀ (n : ℕ),
    ∀ (b : ℕ), b > 0 →
      ∃ (k : ℕ) (t : ℕ), t ≥ 2 ∧ sequenceP P n k = t ^ b) :
  ∃ (a b : ℤ), ∀ x, P x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_is_linear_l1206_120686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_l1206_120673

noncomputable def G : ℂ := (1/2 : ℂ) + (1/2 : ℂ) * Complex.I

theorem reciprocal_of_G :
  let recip_G := G⁻¹
  (recip_G.re = 1) ∧ 
  (recip_G.im = -1) ∧ 
  (Complex.abs recip_G > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_G_l1206_120673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coords_where_f_equals_2_2_l1206_120608

-- Define a piecewise linear function with five segments
noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then -3*x - 5
  else if x < -1 then -x - 3
  else if x < 1 then 2*x
  else if x < 2 then -x + 3
  else 2*x - 3

-- Define the proposition to be proved
theorem sum_of_x_coords_where_f_equals_2_2 : 
  ∃ (x₁ x₂ : ℝ), f x₁ = 2.2 ∧ f x₂ = 2.2 ∧ x₁ + x₂ = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coords_where_f_equals_2_2_l1206_120608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cheburashkas_is_eleven_l1206_120697

/-- Represents the number of Cheburashkas in a row -/
def num_cheburashkas_per_row : ℕ → ℕ := sorry

/-- Represents the total number of characters in a row before erasure -/
def total_characters_per_row (n : ℕ) : ℕ :=
  2 * num_cheburashkas_per_row n - 1 + num_cheburashkas_per_row n

/-- The total number of Krakozyabras after erasure -/
def total_krakozyabras : ℕ := 29

/-- Theorem stating that the total number of Cheburashkas is 11 -/
theorem total_cheburashkas_is_eleven :
  ∃ n : ℕ, 
    num_cheburashkas_per_row n ≥ 1 ∧
    num_cheburashkas_per_row (n + 1) ≥ 1 ∧
    total_characters_per_row n + total_characters_per_row (n + 1) - 2 = total_krakozyabras ∧
    num_cheburashkas_per_row n + num_cheburashkas_per_row (n + 1) = 11 := by
  sorry

#check total_cheburashkas_is_eleven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cheburashkas_is_eleven_l1206_120697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1206_120637

/-- The radius of an inscribed circle in a sector that is one-third of a larger circle -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  ∃ (inscribed_radius : ℝ), 
    inscribed_radius = R * (Real.sqrt 2 - 1) ∧
    inscribed_radius > 0 ∧
    inscribed_radius < R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l1206_120637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1206_120635

/-- Represents a triangle with side lengths k, c, and d --/
structure Triangle where
  k : ℝ
  c : ℝ
  d : ℝ

/-- Represents a figure composed of squares and triangles --/
structure Figure where
  perimeter : ℝ

/-- The problem setup with three figures --/
structure ProblemSetup where
  fig1 : Figure
  fig2 : Figure
  fig3 : Figure
  triangle : Triangle

/-- The theorem statement --/
theorem triangle_side_lengths 
  (setup : ProblemSetup) 
  (h1 : setup.fig1.perimeter = 26)
  (h2 : setup.fig2.perimeter = 32)
  (h3 : setup.fig3.perimeter = 30)
  (h4 : setup.fig2.perimeter - setup.fig1.perimeter = 2 * setup.triangle.k)
  (h5 : setup.fig3.perimeter = 2 * setup.triangle.k + 6 * setup.triangle.c)
  (h6 : setup.fig1.perimeter = 2 * (setup.triangle.c + setup.triangle.d)) :
  setup.triangle.k = 3 ∧ setup.triangle.c = 4 ∧ setup.triangle.d = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_lengths_l1206_120635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apartment_size_marks_max_apartment_size_l1206_120691

/-- The maximum apartment size that can be rented given a rental rate and budget -/
theorem max_apartment_size (rate : ℝ) (budget : ℝ) (h_rate : rate > 0) (h_budget : budget > 0) :
  let max_size := budget / rate
  (max_size * rate = budget ∧ ∀ s : ℝ, s * rate ≤ budget → s ≤ max_size) := by
  sorry

/-- The specific case for Mark's apartment search -/
theorem marks_max_apartment_size :
  let rate : ℝ := 1.20
  let budget : ℝ := 720
  let max_size := budget / rate
  (max_size = 600 ∧ ∀ s : ℝ, s * rate ≤ budget → s ≤ max_size) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apartment_size_marks_max_apartment_size_l1206_120691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1206_120657

/-- The equation of a hyperbola given its asymptotes and foci -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) : 
  (∃ k : ℝ, ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 / 16 - x^2 / 9 = k) →
  (∀ (x y : ℝ), (x, y) ∈ C → (y = 4/3 * x ∨ y = -4/3 * x)) →
  (∃ (a b : ℝ), a^2 + b^2 = 25 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C → (x - a)^2 / 16 + (y - b)^2 / 9 = 1 ∧
                              (x + a)^2 / 16 + (y + b)^2 / 9 = 1) →
  ∀ (x y : ℝ), (x, y) ∈ C ↔ y^2 / 16 - x^2 / 9 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1206_120657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_comparison_l1206_120670

-- Define the quadrilaterals
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the concept of corresponding sides being equal
def corresponding_sides_equal (q1 q2 : Quadrilateral) : Prop :=
  (dist q1.A q1.B = dist q2.A q2.B) ∧
  (dist q1.B q1.C = dist q2.B q2.C) ∧
  (dist q1.C q1.D = dist q2.C q2.D) ∧
  (dist q1.D q1.A = dist q2.D q2.A)

-- Define the angle measurement function
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define convexity for a quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem quadrilateral_angle_comparison
  (q1 q2 : Quadrilateral)
  (h_convex1 : is_convex q1)
  (h_convex2 : is_convex q2)
  (h_sides : corresponding_sides_equal q1 q2)
  (h_angle_A : angle q1.B q1.A q1.D > angle q2.B q2.A q2.D) :
  (angle q1.A q1.B q1.C < angle q2.A q2.B q2.C) ∧
  (angle q1.B q1.C q1.D > angle q2.B q2.C q2.D) ∧
  (angle q1.C q1.D q1.A < angle q2.C q2.D q2.A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_comparison_l1206_120670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l1206_120616

/-- The function f(x) defined as a*ln(x) + 2x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 * x

/-- The theorem stating the range of a for which the minimum value of f(x) is not less than -a --/
theorem min_value_condition (a : ℝ) : 
  (a ≠ 0 ∧ ∀ x > 0, f a x ≥ -a) ↔ -2 ≤ a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l1206_120616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1206_120678

noncomputable def f (a b c x : ℝ) : ℝ := (b * x) / (a * x^2 + c)

theorem function_properties (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : (deriv (f a b c)) 0 = 9)
  (h3 : b + c = 10) :
  b = 9 ∧ c = 1 ∧ 
  ∀ x : ℝ, 0 < a → a ≤ 1 → x > 1 → (x^3 + 1) * f a b c x > 9 + Real.log x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1206_120678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1206_120606

-- Define the inequality function
def f (x : ℝ) : Prop :=
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4

-- Define the solution set
noncomputable def solution_set : Set ℝ :=
  Set.Ioo (-2) (-1) ∪ Set.Ioo 2 ((13 - Real.sqrt 57) / 8) ∪ Set.Ioi ((13 + Real.sqrt 57) / 8)

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x} = solution_set := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1206_120606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l1206_120646

-- Define the point P
def P : ℝ × ℝ := (-1, 4)

-- Define line l₂
def l₂ (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Define line l₁ (parallel to l₂ and passing through P)
def l₁ (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define line l₃
def l₃ (x y m : ℝ) : Prop := 4 * x - 2 * y + m = 0

-- Distance between two parallel lines
noncomputable def distance_parallel_lines (a b c d : ℝ) : ℝ := 
  |c - d| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem line_distance_theorem (m : ℝ) : 
  (P.1 ∈ Set.range (λ x => x)) → 
  (P.2 ∈ Set.range (λ y => y)) → 
  (∀ x y, l₁ x y ↔ l₂ (x + 3.5) (y + 0.5)) → 
  (distance_parallel_lines 4 (-2) 12 m = 2 * Real.sqrt 5) → 
  (m = -8 ∨ m = 32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_distance_theorem_l1206_120646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_Ω_l1206_120659

/-- The body Ω -/
def Ω : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    z ≤ Real.sqrt (1 - x^2 - y^2) ∧
                    z ≥ Real.sqrt ((x^2 + y^2) / 4)}

/-- The density function μ -/
def μ : ℝ × ℝ × ℝ → ℝ
  | (_, _, z) => 20 * z

/-- The mass of the body Ω -/
noncomputable def mass (S : Set (ℝ × ℝ × ℝ)) (ρ : ℝ × ℝ × ℝ → ℝ) : ℝ :=
  ∫ p in S, ρ p

/-- The theorem stating that the mass of Ω is 4π -/
theorem mass_of_Ω : mass Ω μ = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_of_Ω_l1206_120659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coal_cheaper_from_B_equal_expense_point_l1206_120685

/-- Represents the coal pricing and transportation scenario -/
structure CoalScenario where
  s : ℝ  -- Road length between A and B in kilometers
  q : ℝ  -- Coal price at A in rubles per ton
  p : ℝ  -- Percentage increase in coal price at B
  r : ℝ  -- Transportation cost in rubles per ton per kilometer
  s_pos : 0 < s
  q_pos : 0 < q
  p_pos : 0 < p
  r_pos : 0 < r

/-- The cost of coal from point A for a given distance x from B -/
noncomputable def cost_from_A (scenario : CoalScenario) (x : ℝ) : ℝ :=
  scenario.r * (scenario.s - x) + scenario.q

/-- The cost of coal from point B for a given distance x from B -/
noncomputable def cost_from_B (scenario : CoalScenario) (x : ℝ) : ℝ :=
  scenario.r * x + scenario.q * (1 + scenario.p / 100)

/-- Theorem stating the condition for when coal from B is cheaper or equal in cost -/
theorem coal_cheaper_from_B (scenario : CoalScenario) (x : ℝ) :
  cost_from_B scenario x ≤ cost_from_A scenario x ↔ 
  x ≥ scenario.s / 2 - scenario.q * scenario.p / (200 * scenario.r) := by
  sorry

/-- Theorem stating the point of equal expense -/
theorem equal_expense_point (scenario : CoalScenario) :
  ∃ x : ℝ, cost_from_A scenario x = cost_from_B scenario x ∧ 
  x = scenario.s / 2 - scenario.q * scenario.p / (200 * scenario.r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coal_cheaper_from_B_equal_expense_point_l1206_120685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1206_120644

/-- Represents a parallelogram with heights h_a and h_b, and angle γ between adjacent sides -/
structure Parallelogram where
  h_a : ℝ
  h_b : ℝ
  γ : ℝ
  h_a_pos : 0 < h_a
  h_b_pos : 0 < h_b
  γ_pos : 0 < γ
  γ_lt_pi : γ < π

/-- The area of a parallelogram -/
noncomputable def area (p : Parallelogram) : ℝ := p.h_a * p.h_b / Real.sin p.γ

/-- Theorem stating that the area of a parallelogram is equal to the product of its two heights
    divided by the sine of the angle between them -/
theorem parallelogram_area (p : Parallelogram) : 
  area p = p.h_a * p.h_b / Real.sin p.γ := by
  -- Unfold the definition of area
  unfold area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1206_120644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1206_120699

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem axis_of_symmetry :
  ∃ (k : ℤ), g ((k : ℝ) * Real.pi / 2 + 5 * Real.pi / 12) = 1 ∨ g ((k : ℝ) * Real.pi / 2 + 5 * Real.pi / 12) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1206_120699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l1206_120618

noncomputable section

open Real

variable (t : ℝ)

def x (t : ℝ) : ℝ := sin t - t * cos t
def y (t : ℝ) : ℝ := cos t + t * sin t

def y_xx_second_derivative (t : ℝ) : ℝ := -1 / (t * (sin t)^3)

-- We use 'deriv' instead of '∂' for derivatives in Lean
theorem second_derivative_parametric_function (t : ℝ) :
  deriv (deriv (y ∘ x⁻¹)) (x t) = y_xx_second_derivative t :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_function_l1206_120618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_diminished_number_l1206_120639

theorem smallest_diminished_number (n : ℕ) : 
  (∀ d ∈ ({12, 16, 18, 21, 28} : Set ℕ), (1011 - n) % d = 0) ∧
  (∀ m : ℕ, m < n → ∃ d ∈ ({12, 16, 18, 21, 28} : Set ℕ), (1011 - m) % d ≠ 0) →
  n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_diminished_number_l1206_120639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1206_120656

/-- An exponential function passing through (1, 2010) -/
noncomputable def f (x : ℝ) : ℝ := 2010^x

/-- The inverse function of f -/
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2010

theorem inverse_function_proof :
  (∀ x > 0, f (g x) = x) ∧ (∀ x, g (f x) = x) ∧ f 1 = 2010 := by
  sorry

#check inverse_function_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1206_120656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_is_zero_l1206_120666

theorem g_100_is_zero
  (g : ℝ → ℝ)
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x^2 / y)) :
  g 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_is_zero_l1206_120666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l1206_120658

/-- Reflection of a point across a line --/
def reflect (m b : ℚ) : ℚ × ℚ → ℚ × ℚ := sorry

/-- The theorem stating that if (2, -3) is reflected to (4, 5) across y = mx + b, then m + b = 3/2 --/
theorem reflection_sum (m b : ℚ) : 
  reflect m b (2, -3) = (4, 5) → m + b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_sum_l1206_120658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_present_age_l1206_120604

-- Define Henry and Jill's ages as natural numbers
def henry_age : ℕ → Prop := λ h => True
def jill_age : ℕ → Prop := λ j => True

-- Define the conditions
axiom sum_of_ages : ∀ h j, henry_age h ∧ jill_age j → h + j = 40
axiom past_age_relation : ∀ h j, henry_age h ∧ jill_age j → h - 11 = 2 * (j - 11)

-- Theorem to prove
theorem henry_present_age :
  ∃ h, henry_age h ∧ h = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_present_age_l1206_120604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_measure_four_messzely_l1206_120643

/-- Represents the state of the two containers -/
structure ContainerState where
  small : ℕ  -- Amount in the 3-messzely container
  large : ℕ  -- Amount in the 5-messzely container

/-- Represents a pouring action -/
inductive PouringAction where
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies an action to a container state -/
def applyAction (state : ContainerState) (action : PouringAction) : ContainerState :=
  match action with
  | PouringAction.FillSmall => ⟨3, state.large⟩
  | PouringAction.FillLarge => ⟨state.small, 5⟩
  | PouringAction.EmptySmall => ⟨0, state.large⟩
  | PouringAction.EmptyLarge => ⟨state.small, 0⟩
  | PouringAction.PourSmallToLarge =>
      let amount := min state.small (5 - state.large)
      ⟨state.small - amount, state.large + amount⟩
  | PouringAction.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      ⟨state.small + amount, state.large - amount⟩

/-- Theorem stating that it's possible to measure 4 messzely -/
theorem can_measure_four_messzely :
  ∃ (actions : List PouringAction), 
    (actions.foldl applyAction ⟨0, 0⟩).small = 0 ∧ 
    (actions.foldl applyAction ⟨0, 0⟩).large = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_measure_four_messzely_l1206_120643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_polar_curves_l1206_120689

/-- The area enclosed by the curves represented by the polar equations
    θ = π/3 (ρ > 0), θ = 2π/3 (ρ > 0), and ρ = 4 is equal to 8π/3. -/
theorem area_enclosed_by_polar_curves : 
  (1 / 2 : Real) * 4^2 * (2 * π / 3 - π / 3) = 8 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_polar_curves_l1206_120689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l1206_120655

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  hq : q ≠ 1 -- Assumption that q ≠ 1

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def S (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_5 (g : GeometricSequence) 
  (h2 : S g 2 = 3)
  (h6 : S g 6 = 63) :
  S g 5 = 31 ∨ S g 5 = -33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l1206_120655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_sliced_prism_l1206_120640

/-- A right prism with equilateral triangular bases -/
structure RightPrism :=
  (height : ℝ)
  (baseSideLength : ℝ)

/-- A point in 3D space -/
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- The solid BGPR formed by slicing the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (B : Point3D)
  (G : Point3D)
  (P : Point3D)
  (Q : Point3D)  -- Added Q point
  (R : Point3D)

/-- The surface area of the sliced solid BGPR -/
noncomputable def surfaceArea (solid : SlicedSolid) : ℝ := sorry

theorem surface_area_of_sliced_prism (solid : SlicedSolid) 
  (h1 : solid.prism.height = 20)
  (h2 : solid.prism.baseSideLength = 10)
  (h3 : solid.P = Point3D.mk 5 0 0)  -- Midpoint of AB
  (h4 : solid.Q = Point3D.mk 5 0 10) -- Midpoint of BG
  (h5 : solid.R = Point3D.mk 0 5 20) -- Midpoint of DG
  : surfaceArea solid = 50 + 25 * Real.sqrt 3 / 4 + 5 * Real.sqrt 118.75 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_sliced_prism_l1206_120640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_l1206_120677

-- Define the sequence a_n
noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then
    2 / (n * (n + 2))
  else
    Real.log ((n + 2) / n)

-- State the theorem
theorem sum_first_10_terms :
  Finset.sum (Finset.range 10) (fun i => a (i + 1)) = 10 / 11 + Real.log 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_terms_l1206_120677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_fifth_l1206_120653

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 1 ∧ 
  ∀ n ≥ 2, (a n * a (n-1)) / (a (n-1) - a n) = (a n * a (n+1)) / (a n - a (n+1))

theorem tenth_term_is_one_fifth (a : ℕ → ℚ) (h : my_sequence a) : a 10 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_one_fifth_l1206_120653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1206_120613

noncomputable section

-- Define the point P
def P : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Define the line l passing through P with slope -1
def l (t : ℝ) : ℝ × ℝ := (-t / Real.sqrt 2, Real.sqrt 3 + t / Real.sqrt 2)

-- Define the intersection points A and B
def A : ℝ × ℝ := l (Real.sqrt 2 - Real.sqrt 8)
def B : ℝ × ℝ := l (Real.sqrt 2 + Real.sqrt 8)

-- Theorem statement
theorem intersection_ratio :
  C A.1 A.2 ∧ C B.1 B.2 →
  (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) / Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) +
  (Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) / Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)) = 8/3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l1206_120613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_sum_equals_black_l1206_120693

noncomputable def white_number (a b : ℤ) : ℝ := Real.sqrt (a + b * Real.sqrt 2)
noncomputable def black_number (c d : ℤ) : ℝ := Real.sqrt (c + d * Real.sqrt 7)

theorem white_sum_equals_black (a b c d : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  a = 3 ∧ b = 1 ∧ c = 6 ∧ d = 2 →
  white_number a b + white_number a (-b) = black_number c d :=
by
  intro h
  sorry

#check white_sum_equals_black

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_sum_equals_black_l1206_120693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_of_fraction_l1206_120626

theorem greatest_integer_of_fraction : 
  ⌊(4^100 + 3^100 : ℝ) / (4^98 + 3^98 : ℝ)⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_of_fraction_l1206_120626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_microdegrees_in_seventeenth_of_circle_l1206_120661

/-- The number of degrees in a full circle -/
noncomputable def full_circle_degrees : ℚ := 360

/-- The fraction of a circle we're considering -/
noncomputable def circle_fraction : ℚ := 1 / 17

/-- The conversion factor from degrees to microdegrees -/
noncomputable def degrees_to_microdegrees : ℚ := 1000

/-- The number of microdegrees in a fraction of a circle -/
noncomputable def microdegrees_in_fraction (fraction : ℚ) : ℚ :=
  fraction * full_circle_degrees * degrees_to_microdegrees

theorem microdegrees_in_seventeenth_of_circle :
  microdegrees_in_fraction circle_fraction = (360 / 17) * 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_microdegrees_in_seventeenth_of_circle_l1206_120661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_valid_set_l1206_120674

def is_valid_set (S : Finset Nat) : Prop :=
  S.card = 8 ∧
  (∀ x, x ∈ S → x ≥ 1 ∧ x ≤ 15) ∧
  (∀ a b, a ∈ S → b ∈ S → a < b → ¬(b % a = 0))

theorem least_element_in_valid_set :
  ∃ S : Finset Nat, is_valid_set S ∧
  (∀ x, x ∈ S → x ≥ 4) ∧
  (∀ T : Finset Nat, is_valid_set T → ∃ y, y ∈ T ∧ y ≥ 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_element_in_valid_set_l1206_120674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l1206_120690

noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem farthest_point_from_origin :
  let points : List (ℝ × ℝ) := [(0, 5), (1, 2), (3, -4), (6, 0), (-1, -2)]
  (6, 0) ∈ points ∧
  ∀ p ∈ points, distance_from_origin 6 0 ≥ distance_from_origin p.1 p.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_from_origin_l1206_120690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speeds_equal_l1206_120663

/-- The speed of train A in km/h -/
noncomputable def speed_A : ℝ := 70

/-- The time train A takes to reach its destination after meeting train B, in hours -/
noncomputable def time_A_after_meeting : ℝ := 9

/-- The time train B takes to reach its destination after meeting train A, in hours -/
noncomputable def time_B_after_meeting : ℝ := 4

/-- The speed of train B in km/h -/
noncomputable def speed_B : ℝ := speed_A * time_A_after_meeting / time_B_after_meeting

theorem train_speeds_equal : speed_B = speed_A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speeds_equal_l1206_120663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1206_120605

/-- A type representing the arrangement of chips on an 8x8 board -/
def BoardArrangement := Fin 8 → Fin 8 → Bool

/-- The number of chips in a given column -/
def column_sum (b : BoardArrangement) (j : Fin 8) : Nat :=
  (Finset.range 8).sum (fun i => if b i j then 1 else 0)

/-- The number of chips in a given row -/
def row_sum (b : BoardArrangement) (i : Fin 8) : Nat :=
  (Finset.range 8).sum (fun j => if b i j then 1 else 0)

/-- The main theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (b : BoardArrangement),
  (∀ j k : Fin 8, column_sum b j = column_sum b k) ∧
  (∀ i l : Fin 8, i ≠ l → row_sum b i ≠ row_sum b l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l1206_120605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_graph_l1206_120660

/-- Given a function g : ℝ → ℝ such that g(8) = 6, prove that (8/3, 1) is on the graph of 3y = (g(3x) + 3) / 3 and the sum of its coordinates is 11/3 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 8 = 6) :
  let f (x y : ℝ) := 3 * y = (g (3 * x) + 3) / 3
  f (8/3) 1 ∧ 8/3 + 1 = 11/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_transformed_graph_l1206_120660
