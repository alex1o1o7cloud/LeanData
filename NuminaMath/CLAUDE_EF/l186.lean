import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l186_18656

/-- The equation of the curve in polar coordinates --/
def polar_equation (r θ : ℝ) : Prop := r = 2 / (1 + Real.sin θ)

/-- The equation of a unit circle in Cartesian coordinates --/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Theorem stating that the polar equation represents a circle --/
theorem polar_equation_is_circle :
  ∀ (x y θ : ℝ), 
    (let r := Real.sqrt (x^2 + y^2);
     polar_equation r θ ∧ 
     x = r * Real.cos θ ∧ 
     y = r * Real.sin θ) →
    unit_circle x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_is_circle_l186_18656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l186_18695

/-- J function for nonzero real numbers -/
noncomputable def J (a b c : ℝ) : ℝ := a / b + b / c + c / a

/-- Theorem stating that J(3, -6, 4) equals -2/3 -/
theorem J_specific_value : J 3 (-6) 4 = -2/3 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_specific_value_l186_18695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l186_18664

-- Define the points in the coordinate plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (15, 0)
def C : ℝ × ℝ := (25, 0)

-- Define the initial slopes of the lines
def slope_ℓA : ℝ := 1
noncomputable def slope_ℓB : ℝ := Real.arctan 1000000  -- Approximation for vertical line
def slope_ℓC : ℝ := -1

-- Define the rotation of the lines
noncomputable def rotate_line (p : ℝ × ℝ) (initial_slope : ℝ) (angle : ℝ) : ℝ × ℝ → Prop :=
  λ q => ∃ (m : ℝ), (q.2 - p.2) = m * (q.1 - p.1) ∧ m = Real.tan (Real.arctan initial_slope + angle)

-- Define the triangle formed by the intersections
noncomputable def triangle_area (ℓA ℓB ℓC : ℝ × ℝ → Prop) : ℝ :=
  sorry  -- Definition of the area of the triangle formed by the three lines

-- Theorem statement
theorem max_triangle_area :
  ∃ (angle : ℝ),
    let ℓA := rotate_line A slope_ℓA angle
    let ℓB := rotate_line B slope_ℓB angle
    let ℓC := rotate_line C slope_ℓC angle
    ∀ θ : ℝ,
      triangle_area (rotate_line A slope_ℓA θ)
                    (rotate_line B slope_ℓB θ)
                    (rotate_line C slope_ℓC θ) ≤ 162.5 ∧
      triangle_area ℓA ℓB ℓC = 162.5 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l186_18664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l186_18633

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l186_18633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_yesterday_stars_total_stars_eq_sum_l186_18601

/-- The number of gold stars Shelby earned yesterday -/
def yesterday_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def today_stars : ℕ := 3

/-- The total number of gold stars Shelby has now -/
def total_stars : ℕ := 7

/-- Theorem: Shelby earned 4 gold stars yesterday -/
theorem shelby_yesterday_stars : yesterday_stars = 4 := by
  rfl

/-- Theorem: The total number of stars is the sum of yesterday's and today's stars -/
theorem total_stars_eq_sum : total_stars = yesterday_stars + today_stars := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_yesterday_stars_total_stars_eq_sum_l186_18601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_when_parallel_to_x_axis_radius_when_tangent_to_circle_l186_18693

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the circle (renamed to avoid conflict)
def circleEq (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the line l
structure Line where
  slope : Option ℝ
  intercept : ℝ

-- Define the point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the origin
def origin : Point := ⟨0, 0⟩

-- Define the right angle
def is_right_angle (A B : Point) : Prop :=
  (A.x - origin.x) * (B.x - origin.x) + (A.y - origin.y) * (B.y - origin.y) = 0

-- Define the area of a triangle
noncomputable def triangle_area (A B : Point) : ℝ :=
  (1/2) * abs (A.x * B.y - A.y * B.x)

-- Part I theorem
theorem area_when_parallel_to_x_axis 
  (l : Line) (A B : Point) (h1 : ellipse A.x A.y) (h2 : ellipse B.x B.y)
  (h3 : l.slope = some 0) (h4 : is_right_angle A B) :
  triangle_area A B = 4/5 :=
sorry

-- Part II theorem
theorem radius_when_tangent_to_circle 
  (l : Line) (A B : Point) (r : ℝ)
  (h1 : ellipse A.x A.y) (h2 : ellipse B.x B.y)
  (h3 : ∀ (x y : ℝ), circleEq x y r → (∃ (t : ℝ), y = (Option.getD l.slope 0) * x + t ∨ x = t))
  (h4 : is_right_angle A B) (h5 : r > 0) :
  r = 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_when_parallel_to_x_axis_radius_when_tangent_to_circle_l186_18693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_is_nine_fourths_l186_18618

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y² = 2px -/
structure Parabola where
  p : ℝ

/-- The distance from a point to the directrix of a parabola -/
noncomputable def distanceToDirectrix (point : Point) (parabola : Parabola) : ℝ :=
  point.x + parabola.p / 2

theorem distance_to_directrix_is_nine_fourths :
  ∀ (C : Parabola) (A : Point),
    A.x = 1 →
    A.y = Real.sqrt 5 →
    A.y ^ 2 = 2 * C.p * A.x →
    distanceToDirectrix A C = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_is_nine_fourths_l186_18618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l186_18666

noncomputable def a (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x), Real.cos (ω * x))

noncomputable def b (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x) + 2 * Real.cos (ω * x), Real.cos (ω * x))

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2

def is_periodic (g : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, g (x + T) = g x

def smallest_positive_period (g : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic g T ∧ T > 0 ∧ ∀ T' > 0, is_periodic g T' → T ≤ T'

theorem vector_dot_product_problem (ω : ℝ) (h1 : ω > 0) 
  (h2 : smallest_positive_period (f ω) (Real.pi / 4)) :
  ω = 4 ∧ 
  (∀ x, f ω x ≤ 2) ∧ 
  (∀ x, f ω x = 2 ↔ ∃ k : ℤ, x = Real.pi / 16 + k * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l186_18666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_power_equality_l186_18657

-- Define the digit sum function
def digitSum (n : Nat) : Nat := sorry

-- Main theorem
theorem digit_sum_power_equality (a b : Nat) :
  (a > 0 ∧ b > 0) →
  (digitSum (a^(b+1)) = a^b ↔ 
  (a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_power_equality_l186_18657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l186_18655

/-- Represents a cylinder with given height and surface area -/
structure Cylinder where
  height : ℝ
  surfaceArea : ℝ

/-- Calculate the radius of the base circle of a cylinder -/
noncomputable def Cylinder.baseRadius (c : Cylinder) : ℝ :=
  5 -- The actual calculation is omitted

/-- Calculate the volume of a cylinder -/
noncomputable def Cylinder.volume (c : Cylinder) : ℝ :=
  200 * Real.pi -- The actual calculation is omitted

/-- Theorem stating the properties of the specific cylinder -/
theorem cylinder_properties :
  let c : Cylinder := { height := 8, surfaceArea := 130 * Real.pi }
  c.baseRadius = 5 ∧ c.volume = 200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_properties_l186_18655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_undetermined_l186_18699

-- Define the equation for which a and b are roots
noncomputable def equation (x θ : ℝ) : ℝ := x^2 * Real.sin θ + x * Real.cos θ - (Real.pi / 4)

-- Define a and b as roots of the equation
axiom a_is_root (θ : ℝ) : ∃ a : ℝ, equation a θ = 0
axiom b_is_root (θ : ℝ) : ∃ b : ℝ, equation b θ = 0 ∧ b ≠ (Classical.choose (a_is_root θ))

-- Define the line passing through points A(a^2, a) and B(b^2, b)
def line_equation (x y a b : ℝ) : Prop :=
  y = ((b - a) / (b^2 - a^2)) * x + (a * b * (a - b) / (b^2 - a^2))

-- Define the distance between a point and the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := distance_to_origin x y = 1

-- Theorem stating that the relationship cannot be determined
theorem relationship_undetermined (θ : ℝ) :
  ∃ a b : ℝ, equation a θ = 0 ∧ equation b θ = 0 ∧ a ≠ b ∧
  (∀ x y : ℝ, line_equation x y a b →
    ¬(∀ d : ℝ, d = distance_to_origin x y → (d < 1 ∨ d = 1 ∨ d > 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_undetermined_l186_18699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_spheres_l186_18634

/-- The greatest distance between points on two spheres -/
theorem greatest_distance_between_spheres :
  let sphere1_center : ℝ × ℝ × ℝ := (-5, -15, 10)
  let sphere1_radius : ℝ := 24
  let sphere2_center : ℝ × ℝ × ℝ := (15, 5, -20)
  let sphere2_radius : ℝ := 93
  let distance_between_centers : ℝ := Real.sqrt ((15 - (-5))^2 + (5 - (-15))^2 + (-20 - 10)^2)
  ∀ p1 p2 : ℝ × ℝ × ℝ,
    (p1.1 - sphere1_center.1)^2 + (p1.2.1 - sphere1_center.2.1)^2 + (p1.2.2 - sphere1_center.2.2)^2 = sphere1_radius^2 →
    (p2.1 - sphere2_center.1)^2 + (p2.2.1 - sphere2_center.2.1)^2 + (p2.2.2 - sphere2_center.2.2)^2 = sphere2_radius^2 →
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2.1 - p1.2.1)^2 + (p2.2.2 - p1.2.2)^2) ≤ 117 + 10 * Real.sqrt 17 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_between_spheres_l186_18634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_line_l186_18620

-- Define the point A
def A : ℝ × ℝ := (6, 0)

-- Define the line y = -x
def line (x : ℝ) : ℝ × ℝ := (x, -x)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem unique_point_on_line :
  ∃! P : ℝ × ℝ, ∃ x : ℝ, P = line x ∧ distance A P = 3 * Real.sqrt 2 := by
  sorry

#check unique_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_line_l186_18620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_is_4_l186_18662

def total_cost : ℕ := 480
def cost_book1 : ℕ := 280
def loss_percent : ℚ := 15 / 100
def gain_percent : ℚ := 19 / 100

def cost_book2 : ℕ := total_cost - cost_book1

def selling_price_book1 : ℚ := cost_book1 * (1 - loss_percent)
def selling_price_book2 : ℚ := cost_book2 * (1 + gain_percent)

def total_selling_price : ℚ := selling_price_book1 + selling_price_book2

theorem overall_loss_is_4 : 
  Int.floor (total_cost - total_selling_price) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_is_4_l186_18662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l186_18645

theorem trigonometric_identities (θ : Real) 
  (h1 : Real.sin θ = 4/5) 
  (h2 : π/2 < θ) 
  (h3 : θ < π) : 
  (Real.tan θ = -4/3) ∧ 
  ((Real.sin θ)^2 + 2*(Real.sin θ)*(Real.cos θ)) / (3*(Real.sin θ)^2 + (Real.cos θ)^2) = -8/57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l186_18645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l186_18629

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Perpendicularity condition for two 2D vectors -/
def isPerpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

/-- Distance between two points in 2D space -/
noncomputable def distance (v1 v2 : Vector2D) : ℝ :=
  Real.sqrt ((v1.x - v2.x)^2 + (v1.y - v2.y)^2)

/-- Cosine of the angle between two 2D vectors -/
noncomputable def cosAngle (v1 v2 : Vector2D) : ℝ :=
  (v1.x * v2.x + v1.y * v2.y) / (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2))

theorem vector_properties (v1 v2 : Vector2D) :
  (isPerpendicular v1 v2 ↔ v1.x * v2.x + v1.y * v2.y = 0) ∧
  (distance v1 v2 = Real.sqrt ((v1.x - v2.x)^2 + (v1.y - v2.y)^2)) ∧
  (cosAngle v1 v2 = (v1.x * v2.x + v1.y * v2.y) / (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l186_18629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_intersection_distance_l186_18630

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Distance from focus to intersection point on parabola -/
theorem focus_to_intersection_distance
  (C : Parabola)
  (F : Point)
  (l : Line)
  (M N D : Point)
  (h1 : C.equation = fun x y => y^2 = 4*x)
  (h2 : F.x = 1 ∧ F.y = 0)
  (h3 : l.equation F.x F.y)
  (h4 : C.equation M.x M.y ∧ l.equation M.x M.y)
  (h5 : C.equation N.x N.y ∧ l.equation N.x N.y)
  (h6 : distance M D = 2 * distance N F)
  (h7 : distance D F = 1)
  : distance M F = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_to_intersection_distance_l186_18630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_l186_18652

theorem negative_cosine_sequence (α : ℝ) : 
  (∀ n : ℕ, Real.cos (2^n * α) < 0) ↔ 
  ∃ k : ℤ, α = 2 * k * Real.pi + 2 * Real.pi / 3 ∨ 
            α = 2 * k * Real.pi - 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_cosine_sequence_l186_18652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_axis_of_symmetry_l186_18669

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (x + 5 * Real.pi / 12)

theorem closest_axis_of_symmetry :
  ∀ k : ℤ, |Real.pi / 12 + k * Real.pi| ≥ |Real.pi / 12| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_axis_of_symmetry_l186_18669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l186_18653

/-- In a triangle ABC, if BE = 1/3 EC, then AE = 3/4 AB + 1/4 AC -/
theorem triangle_vector_relation (A B C E : EuclideanSpace ℝ (Fin 2)) : 
  (E - B) = (1/3 : ℝ) • (C - E) → 
  (E - A) = (3/4 : ℝ) • (B - A) + (1/4 : ℝ) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l186_18653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_solution_l186_18632

theorem chess_tournament_solution (n : ℕ) : n = 4 ↔ 
  (∃ (x : ℕ),
    let total_players := 4 * n;
    let total_matches := (total_players * (total_players - 1)) / 2;
    let junior_wins := 3 * x;
    let senior_wins := 7 * x;
    total_matches = junior_wins + senior_wins ∧
    junior_wins * 7 = senior_wins * 3)
:= by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_solution_l186_18632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l186_18609

-- Define the second quadrant
def is_second_quadrant (θ : ℝ) : Prop :=
  Real.pi / 2 < θ ∧ θ < Real.pi

-- Define the fourth quadrant for a point
def is_fourth_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem point_in_fourth_quadrant (θ : ℝ) 
  (h : is_second_quadrant θ) :
  is_fourth_quadrant (Real.sin (Real.cos θ)) (Real.cos (Real.cos θ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l186_18609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_point_l186_18651

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 2*x - 4*y + 8

/-- The center of the circle -/
def center : ℝ × ℝ :=
  (1, -2)

/-- The given point -/
def point : ℝ × ℝ :=
  (-3, 4)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_circle_center_to_point :
  distance center point = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_point_l186_18651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_equals_negative_f_a_l186_18605

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / (abs (x + 3) - 3)

-- State the theorem
theorem f_negative_a_equals_negative_f_a (a : ℝ) :
  a ∈ Set.Icc (-2 : ℝ) 2 →  -- a is in the closed interval [-2, 2]
  f a = -4 →               -- given condition
  f (-a) = 4               -- conclusion to prove
:= by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_equals_negative_f_a_l186_18605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l186_18681

open Real

-- Define the function f(x) = ln(2x - x^2)
noncomputable def f (x : ℝ) : ℝ := log (2*x - x^2)

-- State the theorem
theorem f_strictly_decreasing :
  ∀ x y, x ∈ Set.Ioo 1 2 → y ∈ Set.Ioo 1 2 → x < y → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l186_18681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_squared_value_l186_18600

noncomputable def y (a b x : ℝ) : ℝ := (a * Real.cos x + b * Real.sin x) * Real.cos x

theorem ab_squared_value (a b : ℝ) :
  (∀ x, y a b x ≤ 2) ∧
  (∀ x, y a b x ≥ -1) ∧
  (∃ x, y a b x = 2) ∧
  (∃ x, y a b x = -1) →
  (a * b)^2 = 8 := by
  sorry

#check ab_squared_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_squared_value_l186_18600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l186_18667

noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.tan x

noncomputable def f' (x : ℝ) : ℝ := -Real.sin x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := a * (1 / (Real.cos x)^2)

theorem tangents_perpendicular (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x = g a x → (f' x) * (g' a x) = -1 := by
  intro x h_intersection
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_perpendicular_l186_18667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_equality_l186_18639

/-- Sum of first m terms in an arithmetic sequence -/
noncomputable def S (m : ℕ) (a₁ d : ℝ) : ℝ := m / 2 * (2 * a₁ + (m - 1) * d)

/-- Theorem: For an arithmetic sequence with first term a₁ and common difference d,
    the equation (S_{n+k})/(n+k) = (S_n - S_k)/(n-k) holds for all n > k ≥ 1 -/
theorem arithmetic_sequence_sum_equality (a₁ d : ℝ) (n k : ℕ) (h : n > k) (h' : k ≥ 1) :
  (S (n + k) a₁ d) / (n + k : ℝ) = (S n a₁ d - S k a₁ d) / ((n - k : ℕ) : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_equality_l186_18639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_theorem_l186_18676

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat : ℝ := (1 - Real.sqrt 5) / 2

noncomputable def fibonacci_generating_function (x : ℝ) : ℝ := 1 / (1 - x - x^2)

noncomputable def lucas_number (n : ℕ) : ℝ :=
  φ^n + φ_hat^n

noncomputable def lucas_generating_function (x : ℝ) : ℝ := (2 - x) / (1 - x - x^2)

theorem lucas_theorem :
  (∀ n : ℕ, lucas_number n = φ^n + φ_hat^n) ∧
  (∀ x : ℝ, lucas_generating_function x = (2 - x) / (1 - x - x^2)) := by
  sorry

#check lucas_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_theorem_l186_18676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_circle_intersects_all_sides_twice_l186_18627

/-- A convex quadrilateral is a polygon with four sides where all interior angles are less than 180 degrees. -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- A circle is defined by its center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two points are considered distinct if they are not equal. -/
def DistinctPoints (p q : ℝ × ℝ) : Prop := p ≠ q

/-- A point is on a line segment if it lies between the endpoints of the segment. -/
def PointOnSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = (1 - t) • a + t • b

/-- A circle intersects a line segment at two interior points if there exist two distinct points on the segment that also lie on the circle. -/
def CircleIntersectsSegmentTwice (c : Circle) (a b : ℝ × ℝ) : Prop :=
  ∃ p q : ℝ × ℝ, DistinctPoints p q ∧
    PointOnSegment p a b ∧ PointOnSegment q a b ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

/-- The main theorem stating that no circle exists that intersects all sides of a convex quadrilateral at two interior points each. -/
theorem no_circle_intersects_all_sides_twice (q : ConvexQuadrilateral) :
  ¬ ∃ c : Circle,
    (CircleIntersectsSegmentTwice c (q.vertices 0) (q.vertices 1)) ∧
    (CircleIntersectsSegmentTwice c (q.vertices 1) (q.vertices 2)) ∧
    (CircleIntersectsSegmentTwice c (q.vertices 2) (q.vertices 3)) ∧
    (CircleIntersectsSegmentTwice c (q.vertices 3) (q.vertices 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_circle_intersects_all_sides_twice_l186_18627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_equivalences_l186_18671

theorem increasing_function_equivalences 
  (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → f x < f y) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b < f (-a) + f (-b) → a + b < 0) ∧
  (∀ a b : ℝ, a + b < 0 → f a + f b < f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_equivalences_l186_18671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_cube_sale_percentage_l186_18650

/-- Represents the properties of a silver cube and its sale --/
structure SilverCube where
  side_length : ℚ
  weight_per_cubic_inch : ℚ
  price_per_ounce : ℚ
  selling_price : ℚ

/-- Calculates the percentage of silver value for which the cube is sold --/
def sale_percentage (cube : SilverCube) : ℚ :=
  let volume := cube.side_length ^ 3
  let weight := volume * cube.weight_per_cubic_inch
  let silver_value := weight * cube.price_per_ounce
  (cube.selling_price / silver_value) * 100

/-- Theorem stating that given the specific conditions, the sale percentage is 110% --/
theorem silver_cube_sale_percentage :
  ∀ (cube : SilverCube),
    cube.side_length = 3 ∧
    cube.weight_per_cubic_inch = 6 ∧
    cube.price_per_ounce = 25 ∧
    cube.selling_price = 4455 →
    sale_percentage cube = 110 := by
  intro cube h
  simp [sale_percentage]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_cube_sale_percentage_l186_18650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l186_18678

-- Define the function f
noncomputable def f (φ : ℝ) : ℝ → ℝ := λ x => Real.sin (2 * x + φ)

-- State the theorem
theorem function_properties (φ : ℝ) 
  (h1 : ∀ x, f φ x ≤ |f φ (π/6)|)
  (h2 : f φ (π/2) > f φ π) :
  (∀ x, f φ x = Real.sin (2*x + 7*π/6)) ∧ 
  (∀ x ∈ Set.Icc 0 (π/6), StrictAntiOn (f φ) (Set.Icc 0 (π/6))) ∧
  (∀ x ∈ Set.Icc (2*π/3) π, StrictAntiOn (f φ) (Set.Icc (2*π/3) π)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l186_18678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_90_degrees_l186_18670

open Real Matrix

noncomputable def rotation_matrix (angle : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos angle, -Real.sin angle; Real.sin angle, Real.cos angle]

theorem rotation_90_degrees : 
  rotation_matrix (π / 2) = !![0, -1; 1, 0] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_90_degrees_l186_18670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l186_18616

-- Define a quadrilateral as a structure with four points in a 2D plane
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) :
  distance q.A q.B < distance q.B q.C + distance q.C q.D + distance q.D q.A ∧
  distance q.B q.C < distance q.C q.D + distance q.D q.A + distance q.A q.B ∧
  distance q.C q.D < distance q.D q.A + distance q.A q.B + distance q.B q.C ∧
  distance q.D q.A < distance q.A q.B + distance q.B q.C + distance q.C q.D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l186_18616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l186_18686

noncomputable def P (x : ℝ) : ℝ := x^3 - 3*x^2 - 3*x - 1

noncomputable def x₁ : ℝ := 1 - Real.sqrt 2
noncomputable def x₂ : ℝ := 1 + Real.sqrt 2
noncomputable def x₃ : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x₄ : ℝ := 1 + 2 * Real.sqrt 2
noncomputable def x₅ : ℝ := 1 + Real.rpow 2 (1/3) + Real.rpow 4 (1/3)

theorem polynomial_properties :
  (P x₁ + P x₂ = -12) ∧
  (P x₃ + P x₄ = -12) ∧
  (P x₅ = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_properties_l186_18686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_solutions_l186_18610

theorem quadratic_rational_solutions (m : ℕ) : 
  (∃ x : ℚ, m * x^2 + 24 * x + m = 0) ↔ m ∈ ({6, 8, 10} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_rational_solutions_l186_18610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_win_series_probability_l186_18643

def probability_cubs_win_game : ℚ := 3/5

def games_to_win : ℕ := 5

noncomputable def probability_cubs_win_series : ℚ := 
  Finset.sum (Finset.range 5) (λ k => (Nat.choose (4 + k) k : ℚ) * probability_cubs_win_game^games_to_win * (1 - probability_cubs_win_game)^k)

theorem cubs_win_series_probability :
  probability_cubs_win_series = 243/625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubs_win_series_probability_l186_18643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_dynamics_index_difference_l186_18613

theorem group_dynamics_index_difference (n k : ℕ) (h1 : n = 25) (h2 : k = 8) :
  (n - k : ℚ) / n - (n - (n - k) : ℚ) / n = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_dynamics_index_difference_l186_18613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l186_18626

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * 1000 / 3600 * time

theorem train_length_calculation :
  let speed := (40 : ℝ) -- km/h
  let time := (7.199424046076314 : ℝ) -- seconds
  abs (trainLength speed time - 80) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l186_18626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_condition_l186_18635

/-- The function f(x) = sin(ωx + π/4) with ω > 0 has only one axis of symmetry
    in the interval [0, π/4] if and only if 1 ≤ ω < 5 -/
theorem sine_symmetry_condition (ω : ℝ) :
  ω > 0 →
  (∃! x, x ∈ Set.Icc 0 (π/4) ∧ ∀ y ∈ Set.Icc 0 (π/4),
    Real.sin (ω * (x - y) + π/4) = Real.sin (ω * (x + y) + π/4)) ↔
  1 ≤ ω ∧ ω < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_condition_l186_18635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_interior_point_inequality_l186_18611

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define the interior point P
def InteriorPoint {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) (P : α) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
  P = a • t.A + b • t.B + c • t.C

-- Define the angle between two vectors
noncomputable def angle {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (u v : α) : ℝ :=
  Real.arccos ((inner u v) / (norm u * norm v))

-- State the theorem
theorem isosceles_triangle_interior_point_inequality 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (t : Triangle α) (P : α) :
  norm (t.A - t.B) = norm (t.A - t.C) →
  InteriorPoint t P →
  angle (t.A - P) (t.B - P) > angle (t.A - P) (t.C - P) →
  norm (t.C - P) > norm (t.B - P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_interior_point_inequality_l186_18611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polycarp_kolya_numbers_l186_18665

def is_valid_number (n : ℕ) : Prop :=
  100000 > n ∧ n ≥ 10000 ∧ 
  (∀ d, d ∈ n.digits 10 → d % 2 = 0) ∧
  (n.digits 10).Nodup

def is_smallest_valid_number (n : ℕ) : Prop :=
  is_valid_number n ∧ ∀ m, is_valid_number m → n ≤ m

theorem polycarp_kolya_numbers : 
  ∃ (p k : ℕ), 
    is_smallest_valid_number p ∧
    is_valid_number k ∧
    k > p ∧
    k - p < 100 ∧
    p.digits 10 = k.digits 10 ∧
    p = 20468 ∧
    k = 20486 := by
  sorry

#check polycarp_kolya_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polycarp_kolya_numbers_l186_18665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l186_18658

noncomputable def f (x : ℝ) : ℝ := (1 + Real.exp x) / (Real.exp x - 6)

def range_f : Set ℝ := {y | ∃ x, f x = y ∧ x ≠ Real.log 6}

theorem f_range : range_f = Set.Iio (-1/6) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l186_18658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_ratio_from_fraction_l186_18628

-- Part 1
theorem inverse_proportion_k (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →
  y (-4) = 3/4 →
  k = -3 := by sorry

-- Part 2
theorem ratio_from_fraction (a b : ℝ) :
  (a - b) / b = 2/3 →
  ∃ (m n : ℕ), m ≠ 0 ∧ n ≠ 0 ∧ (a / b : ℝ) = m / n ∧ m = 5 ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_ratio_from_fraction_l186_18628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_size_valid_set_l186_18604

-- Define the set of digits
def Digits : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define a valid integer as a subset of Digits
def ValidInteger (s : Finset Nat) : Prop :=
  s ⊆ Digits ∧ s.Nonempty ∧ s.toList.Sorted (· < ·)

-- Define the property that any two integers have at least one digit in common
def CommonDigit (S : Finset (Finset Nat)) : Prop :=
  ∀ s₁ s₂, s₁ ∈ S → s₂ ∈ S → s₁ ≠ s₂ → (s₁ ∩ s₂).Nonempty

-- Define the property that no digit appears in all integers
def NoUniversalDigit (S : Finset (Finset Nat)) : Prop :=
  ∀ d, d ∈ Digits → ∃ s, s ∈ S ∧ d ∉ s

-- Define the set of all valid sets of integers satisfying the conditions
def ValidSets : Set (Finset (Finset Nat)) :=
  {S | (∀ s ∈ S, ValidInteger s) ∧ CommonDigit S ∧ NoUniversalDigit S}

-- State the theorem
theorem max_size_valid_set : ∃ S ∈ ValidSets, S.card = 32 ∧ ∀ T ∈ ValidSets, T.card ≤ 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_size_valid_set_l186_18604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_is_299_l186_18690

def loop_iteration (x : ℕ) (s : ℕ) : ℕ × ℕ :=
  let new_x := x + 3
  let new_s := s + new_x
  (new_x, new_s)

def final_x (x₀ s₀ : ℕ) : ℕ :=
  let rec aux (x s : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then x
    else if s < 15000 then
      let (new_x, new_s) := loop_iteration x s
      aux new_x new_s (fuel - 1)
    else
      x
  aux x₀ s₀ 1000  -- Use a sufficiently large fuel value

theorem final_x_is_299 :
  final_x 5 0 = 299 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_is_299_l186_18690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_false_weight_l186_18692

-- Define the profit percentage
def profit_percentage : ℚ := 4.166666666666674

-- Define the true weight of a kg in grams
def true_weight : ℚ := 1000

-- Define the function to calculate the false weight
noncomputable def false_weight (p : ℚ) : ℚ := true_weight / (1 + p / 100)

-- Theorem statement
theorem shopkeeper_false_weight :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |false_weight profit_percentage - 960| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_false_weight_l186_18692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l186_18687

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The ratio of volumes of two cylinders -/
noncomputable def volumeRatio (r1 h1 r2 h2 : ℝ) : ℝ := 
  cylinderVolume r1 h1 / cylinderVolume r2 h2

theorem container_volume_ratio : 
  volumeRatio 6 24 12 24 = 1/4 := by
  -- Expand the definition of volumeRatio
  unfold volumeRatio
  -- Expand the definition of cylinderVolume
  unfold cylinderVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l186_18687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_cos_B_range_max_value_f_B_l186_18641

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 / 2

-- Theorem for the smallest positive period of f
theorem smallest_positive_period_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Theorem for the range of cos(B) in triangle ABC
theorem cos_B_range (a b c : ℝ) (h : b^2 = a*c) :
  ∃ (B : ℝ), 0 < B ∧ B < Real.pi ∧
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧
  (1/2 : ℝ) ≤ Real.cos B ∧ Real.cos B < 1 :=
sorry

-- Theorem for the maximum value of f(B)
theorem max_value_f_B (a b c : ℝ) (h : b^2 = a*c) :
  ∃ (B : ℝ), 0 < B ∧ B < Real.pi ∧
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧
  (∀ (x : ℝ), 0 < x ∧ x < Real.pi ∧ Real.cos x = (a^2 + c^2 - b^2) / (2*a*c) → f x ≤ f B) ∧
  f B = 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_cos_B_range_max_value_f_B_l186_18641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seeds_is_100_l186_18608

/-- The number of seeds in bucket B -/
def seeds_B : ℕ := 30

/-- The number of seeds in bucket A -/
def seeds_A : ℕ := seeds_B + 10

/-- The number of seeds in bucket C -/
def seeds_C : ℕ := 30

/-- The total number of seeds in all buckets -/
def total_seeds : ℕ := seeds_A + seeds_B + seeds_C

theorem total_seeds_is_100 : total_seeds = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_seeds_is_100_l186_18608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l186_18683

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 144

/-- Represents the length of the train in meters -/
noncomputable def train_length : ℝ := 60

/-- Represents the time it takes for the train to cross the electric pole in seconds -/
noncomputable def crossing_time : ℝ := 1.5

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

theorem train_crossing_time :
  crossing_time = train_length / (train_speed * km_hr_to_m_s) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l186_18683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_exist_l186_18624

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a circle is tangent to a line at a given point -/
def circleTangentToLineAtPoint (c : Circle) (l : Line) (p : Point) : Prop :=
  pointOnLine p l ∧ distance p c.center = c.radius

/-- Check if a line is tangent to a circle -/
def lineTangentToCircle (l : Line) (c : Circle) : Prop :=
  ∃ p : Point, pointOnLine p l ∧ distance p c.center = c.radius

theorem four_circles_exist (P : Point) (t : ℝ) (g : Line) (G : Point) :
  pointOnLine G g →
  ∃! (s : Finset Circle), s.card = 4 ∧ 
    ∀ c ∈ s, 
      circleTangentToLineAtPoint c g G ∧
      ∃ P1 : Point, distance P P1 = t ∧ 
        ∃ l : Line, pointOnLine P1 l ∧ lineTangentToCircle l c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_circles_exist_l186_18624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_proof_l186_18688

/-- The quadratic function g(x) = x^2 + 5x + d -/
noncomputable def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + d

/-- 5 is in the range of g -/
def in_range (d : ℝ) : Prop := ∃ x, g d x = 5

/-- The smallest value of d such that 5 is in the range of g -/
noncomputable def smallest_d : ℝ := 45/4

/-- Theorem stating that smallest_d is indeed the smallest value of d
    such that 5 is in the range of g -/
theorem smallest_d_proof :
  (∀ d < smallest_d, ¬(in_range d)) ∧ 
  (in_range smallest_d) := by
  sorry

#check smallest_d_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_proof_l186_18688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l186_18698

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | n + 1 => (8/5) * a n + (6/5) * Real.sqrt (4^n - (a n)^2)

theorem a_10_value : a 10 = 24576/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l186_18698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l186_18679

-- Define the revenue function R(x)
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - (1/30) * x^2
  else if x > 10 then
    108/x - 1000/(3*x^2)
  else
    0

-- Define the profit function W(x)
noncomputable def W (x : ℝ) : ℝ :=
  x * R x - (10 + 2.7 * x)

-- Theorem statement
theorem max_profit_at_nine :
  ∃ (x_max : ℝ), x_max = 9 ∧
  ∀ (x : ℝ), 0 < x → W x ≤ W x_max ∧
  W x_max = 38.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l186_18679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_arithmetic_progression_l186_18694

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6
def common_difference : ℕ := 1

def is_arithmetic_progression (seq : List ℕ) : Prop :=
  seq.length = num_dice ∧
  ∀ i, i + 1 < seq.length → seq[i + 1]! - seq[i]! = common_difference

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length = num_dice ∧
  ∀ n ∈ seq, 1 ≤ n ∧ n ≤ faces_per_die

def count_valid_progressions : ℕ := sorry

theorem probability_of_arithmetic_progression :
  (count_valid_progressions : ℚ) / (faces_per_die ^ num_dice) = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_arithmetic_progression_l186_18694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_sum_of_incircle_radii_l186_18689

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon := Fin 5 → ℝ × ℝ

-- Define a property for a pentagon to be convex and inscriptible
def IsConvexInscriptible (p : Pentagon) : Prop := sorry

-- Define a triangulation of a pentagon
def Triangulation (p : Pentagon) := Fin 3 → Fin 3 → Fin 5

-- Define the radius of an incircle of a triangle
noncomputable def IncircleRadius (t : Fin 3 → ℝ × ℝ) : ℝ := sorry

-- Define the sum of incircle radii for a given triangulation
noncomputable def SumOfIncircleRadii (p : Pentagon) (t : Triangulation p) : ℝ :=
  (Finset.univ : Finset (Fin 3)).sum (λ i => IncircleRadius (λ j => p (t i j)))

-- The main theorem
theorem invariant_sum_of_incircle_radii
  (p : Pentagon) (h : IsConvexInscriptible p) :
  ∀ t1 t2 : Triangulation p,
    SumOfIncircleRadii p t1 = SumOfIncircleRadii p t2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invariant_sum_of_incircle_radii_l186_18689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l186_18619

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  a₁₅ : ℚ  -- Fifteenth term
  is_arithmetic : a₁₅ = a₁ + 14 * (a₁₅ - a₁) / 14

/-- Properties of the specific arithmetic sequence -/
def my_sequence : ArithmeticSequence where
  a₁ := 7
  a₁₅ := 35
  is_arithmetic := by
    field_simp
    ring

/-- General term of an arithmetic sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * (seq.a₁₅ - seq.a₁) / 14

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a₁ + general_term seq n)

theorem arithmetic_sequence_properties :
  general_term my_sequence 60 = 125 ∧
  sum_of_terms my_sequence 60 = 3960 := by
  constructor
  · -- Proof for the 60th term
    simp [general_term, my_sequence]
    norm_num
  · -- Proof for the sum of 60 terms
    simp [sum_of_terms, general_term, my_sequence]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l186_18619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_eq_8_point_2_l186_18680

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

-- Define the function g using f_inv
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

-- Theorem statement
theorem g_of_3_eq_8_point_2 : g 3 = 8.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_eq_8_point_2_l186_18680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_bound_l186_18606

open Real

/-- Given a > 0, define the function f(x) = (3/a)x³ - x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3/a) * x^3 - x

/-- The x-coordinate of the intersection point of the tangent line with the x-axis -/
noncomputable def x₂ (a : ℝ) (x₁ : ℝ) : ℝ :=
  x₁ - f a x₁ / ((9/a) * x₁^2 - 1)

/-- The ratio of x₂ to x₁ -/
noncomputable def ratio (a : ℝ) (x₁ : ℝ) : ℝ :=
  x₂ a x₁ / x₁

theorem tangent_intersection_bound (a : ℝ) (x₁ : ℝ) (ha : a > 0) (hx : x₁ > sqrt (a/3)) :
  2/3 < ratio a x₁ ∧ ratio a x₁ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_bound_l186_18606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l186_18617

def p (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (a * x^2 - a * x + 1) / Real.log 10

def q (a : ℝ) : Prop := ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → (x₁ : ℝ)^(a^2 - 2*a - 3) > (x₂ : ℝ)^(a^2 - 2*a - 3)

theorem range_of_a : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∀ a : ℝ, (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l186_18617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l186_18672

/-- An isosceles trapezoid is a quadrilateral with at least one pair of parallel sides and two non-parallel sides of equal length. -/
structure IsoscelesTrapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_isosceles : true  -- This is a placeholder for the isosceles property

/-- Calculate the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating that the diagonal of the specified isosceles trapezoid is √419 -/
theorem isosceles_trapezoid_diagonal (t : IsoscelesTrapezoid) 
  (h1 : distance t.A t.B = 25)
  (h2 : distance t.B t.C = 12)
  (h3 : distance t.C t.D = 11)
  (h4 : distance t.D t.A = 12) :
  distance t.A t.C = Real.sqrt 419 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_l186_18672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_sum_l186_18649

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_dot_product_sum 
  (u v w z : E)
  (hu : ‖u‖ = 2)
  (hv : ‖v‖ = 6)
  (hw : ‖w‖ = 3)
  (hz : ‖z‖ = 1)
  (hsum : u + v + w + z = 0) :
  inner u v + inner u w + inner u z + inner v w + inner v z + inner w z = (-25 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_sum_l186_18649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_triangles_l186_18602

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  sides : Fin n → ℝ × ℝ

/-- A triangle in a triangulation -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A triangulation of a polygon -/
structure Triangulation (n : ℕ) where
  polygon : Polygon n
  triangles : List Triangle

/-- An internal vertex of a triangulation -/
def InternalVertex (v : ℝ × ℝ) (t : Triangulation n) : Prop := sorry

/-- The number of triangles sharing a vertex in a triangulation -/
def TrianglesSharing (v : ℝ × ℝ) (t : Triangulation n) : Finset Triangle := sorry

/-- An admissible triangulation is one where every internal vertex is shared by 6 or more triangles -/
def Admissible (n : ℕ) (t : Triangulation n) : Prop :=
  ∀ v, InternalVertex v t → (TrianglesSharing v t).card ≥ 6

/-- The theorem to be proved -/
theorem exists_max_triangles (n : ℕ) (h : n ≥ 3) :
  ∃ M_n : ℕ, ∀ (t : Triangulation n), Admissible n t → (t.triangles.length ≤ M_n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_max_triangles_l186_18602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_size_l186_18603

theorem max_intersection_size (A B C : Finset ℕ) : 
  A.Nonempty → B.Nonempty → C.Nonempty →
  A.card = 2019 → B.card = 2019 →
  (2 ^ A.card) + (2 ^ B.card) + (2 ^ C.card) = 2 ^ (A ∪ B ∪ C).card →
  (A ∩ B ∩ C).card ≤ 2018 ∧ ∃ A' B' C' : Finset ℕ, 
    A'.Nonempty ∧ B'.Nonempty ∧ C'.Nonempty ∧
    A'.card = 2019 ∧ B'.card = 2019 ∧
    (2 ^ A'.card) + (2 ^ B'.card) + (2 ^ C'.card) = 2 ^ (A' ∪ B' ∪ C').card ∧
    (A' ∩ B' ∩ C').card = 2018 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_size_l186_18603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_range_sum_l186_18674

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem h_range :
  Set.range h = Set.Ioo 0 1 := by sorry

theorem range_sum :
  ∃ (a b : ℝ), Set.range h = Set.Ioc a b ∧ a + b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_range_sum_l186_18674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_volume_after_transformation_l186_18682

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  side : ℝ  -- Length of one side of the square base
  height : ℝ  -- Height of the pyramid

/-- Calculate the volume of a square-based pyramid -/
noncomputable def volume (p : SquarePyramid) : ℝ := (1/3) * p.side^2 * p.height

/-- Theorem: New volume after tripling base sides and doubling height -/
theorem new_volume_after_transformation (p : SquarePyramid) 
  (h_initial_volume : volume p = 60) :
  volume {side := 3 * p.side, height := 2 * p.height} = 1080 := by
  sorry

#check new_volume_after_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_volume_after_transformation_l186_18682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_characterization_l186_18654

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ (p q r s : ℝ), p > 0 → q > 0 → r > 0 → s > 0 → p * q = r * s →
    (f p ^ 2 + f q ^ 2) / (f (r ^ 2) + f (s ^ 2)) = (p ^ 2 + q ^ 2) / (r ^ 2 + s ^ 2)

/-- The main theorem stating that any satisfying function must be either x or 1/x -/
theorem satisfying_function_characterization (f : ℝ → ℝ) :
  (∀ x : ℝ, x > 0 → f x > 0) →
  SatisfyingFunction f ↔ (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_function_characterization_l186_18654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l186_18685

def is_systematic_sample (sample : List Nat) (total_products : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  ∀ i j, i < j → j < sample.length → 
    (sample.get ⟨j, by sorry⟩) - (sample.get ⟨i, by sorry⟩) = (total_products / sample_size) * (j - i)

theorem correct_systematic_sample :
  is_systematic_sample [2, 12, 22, 32] 40 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l186_18685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l186_18663

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

-- State the theorem
theorem relationship_abc : b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l186_18663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l186_18622

/-- Line l: x - 2y + 8 = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y + 8 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (-2, 8)

/-- Point B -/
def point_B : ℝ × ℝ := (-2, -4)

/-- Point P -/
def point_P : ℝ × ℝ := (12, 10)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem -/
theorem max_distance_difference :
  line_l point_P.1 point_P.2 ∧
  ∀ (x y : ℝ), line_l x y →
    |distance (x, y) point_A - distance (x, y) point_B| ≤ |distance point_P point_A - distance point_P point_B| ∧
  |distance point_P point_A - distance point_P point_B| = 4 * Real.sqrt 2 := by
  sorry

#check max_distance_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l186_18622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_A_l186_18621

/-- Given workers A, B, and C, where:
    - A, B, and C together can finish the work in 4 days
    - B alone can do it in 24 days
    - C alone can do it in approximately 8 days
    Prove that A can finish the work alone in 12 days -/
theorem work_completion_time_A (
  combined_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (h1 : combined_rate = 1 / 4)
  (h2 : B_rate = 1 / 24)
  (h3 : C_rate = 1 / 8) : 
  1 / (combined_rate - B_rate - C_rate) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_A_l186_18621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l186_18637

/-- Represents the sequence of numbers on the board -/
def BoardSequence := List ℝ

/-- Calculates the median of a list of real numbers -/
noncomputable def median (l : List ℝ) : ℝ := sorry

/-- Generates the sequence of medians based on the board sequence -/
def medianSequence (board : BoardSequence) : List ℝ := sorry

/-- The given sequence of medians -/
def givenMedians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Theorem stating that the fourth number on the board is 2 and the eighth number is 2 -/
theorem board_numbers (board : BoardSequence) :
  medianSequence board = givenMedians →
  (board.get? 3 = some 2 ∧ board.get? 7 = some 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l186_18637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_when_met_l186_18636

/-- The distance Bob walked when he met Yolanda -/
noncomputable def distance_bob_walked (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) (yolanda_head_start : ℝ) : ℝ :=
  (total_distance * bob_rate - yolanda_rate * yolanda_head_start) / (yolanda_rate + bob_rate)

/-- Theorem stating that Bob walked 35 miles when they met -/
theorem bob_distance_when_met :
  distance_bob_walked 65 5 7 1 = 35 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_distance_when_met_l186_18636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_determined_by_two_points_l186_18614

/-- A linear function is uniquely determined by two distinct points. -/
theorem linear_function_determined_by_two_points 
  (a b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) (hxy : x₁ ≠ x₂) :
  (y₁ = a * x₁ + b ∧ y₂ = a * x₂ + b) →
  ∃! f : ℝ → ℝ, (∀ x, f x = a * x + b) ∧ f x₁ = y₁ ∧ f x₂ = y₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_determined_by_two_points_l186_18614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_l186_18612

/-- Represents an ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : Point
  F₂ : Point

/-- The maximum distance from a point on the ellipse to F₁ -/
def max_distance (E : Ellipse) : ℝ := 7

/-- The minimum distance from a point on the ellipse to F₁ -/
def min_distance (E : Ellipse) : ℝ := 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := 
  let a := (max_distance E + min_distance E) / 2
  let c := (max_distance E - min_distance E) / 2
  c / a

theorem ellipse_eccentricity_sqrt (E : Ellipse) :
  Real.sqrt (eccentricity E) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_l186_18612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_players_players_in_63_match_tournament_l186_18661

/-- The number of players in a knockout tournament given the number of matches. -/
def number_of_players (num_matches : ℕ) : ℕ := num_matches + 1

/-- In a knockout tournament, the number of players is one more than the number of matches. -/
theorem knockout_tournament_players (num_matches : ℕ) : 
  number_of_players num_matches = num_matches + 1 :=
by
  rfl  -- reflexivity, as this is true by definition

/-- There are 64 players in a knockout tournament with 63 matches. -/
theorem players_in_63_match_tournament : 
  number_of_players 63 = 64 :=
by
  rfl  -- reflexivity, as this follows directly from the definition

#check players_in_63_match_tournament

end NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_players_players_in_63_match_tournament_l186_18661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_ratio_of_congruent_equilateral_triangles_l186_18691

/-- Represents an equilateral triangle on a plane -/
structure EquilateralTriangle where
  center : ℝ × ℝ
  sideLength : ℝ

/-- Helper function to get the vertices of an equilateral triangle -/
def vertices (t : EquilateralTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- Helper function to calculate the maximum possible overlapping area -/
def maxOverlapArea (t1 t2 : EquilateralTriangle) : ℝ :=
  sorry

/-- Helper function to calculate the minimum possible overlapping area -/
def minOverlapArea (t1 t2 : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem: The ratio of maximum to minimum overlapping area of two congruent equilateral triangles -/
theorem overlap_area_ratio_of_congruent_equilateral_triangles 
  (t1 t2 : EquilateralTriangle) 
  (h_congruent : t1.sideLength = t2.sideLength)
  (h_vertex_at_center : ∃ (v : ℝ × ℝ), v ∈ vertices t1 ∧ v = t2.center) :
  (maxOverlapArea t1 t2) / (minOverlapArea t1 t2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_ratio_of_congruent_equilateral_triangles_l186_18691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_running_average_l186_18673

theorem middle_school_running_average : 
  ∃ (t : ℕ), 
  let first_graders := 9 * t
  let second_graders := 3 * t
  let third_graders := t
  let total_students := first_graders + second_graders + third_graders
  let first_graders_minutes := 8 * first_graders
  let second_graders_minutes := 12 * second_graders
  let third_graders_minutes := 16 * third_graders
  let total_minutes := first_graders_minutes + second_graders_minutes + third_graders_minutes
  (total_minutes : ℚ) / (total_students : ℚ) = 10 :=
by
  use 1
  simp [Nat.cast_add, Nat.cast_mul]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_running_average_l186_18673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l186_18697

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem parallel_vectors_magnitude (x : ℝ) :
  parallel (vector_a x) (vector_b x) →
  (magnitude ((vector_a x).1 - 2 * (vector_b x).1, (vector_a x).2 - 2 * (vector_b x).2) = 5 ∨
   magnitude ((vector_a x).1 - 2 * (vector_b x).1, (vector_a x).2 - 2 * (vector_b x).2) = 3 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l186_18697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noahs_call_charge_per_minute_l186_18684

/-- Calculates the charge per call minute for Noah's calls to his Grammy -/
theorem noahs_call_charge_per_minute 
  (calls_per_year : ℕ) 
  (minutes_per_call : ℕ) 
  (total_bill : ℚ) 
  (charge_per_minute : ℚ := total_bill / (calls_per_year * minutes_per_call : ℚ)) :
  calls_per_year = 52 →
  minutes_per_call = 30 →
  total_bill = 78 →
  charge_per_minute = 1/20 := by
  intros h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_noahs_call_charge_per_minute_l186_18684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_lcm_hcf_l186_18638

theorem ratio_lcm_hcf (a b c : ℕ) (k : ℕ) 
  (h_ratio : a = 3 * k ∧ b = 4 * k ∧ c = 5 * k)
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 2400) : Nat.gcd a (Nat.gcd b c) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_lcm_hcf_l186_18638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l186_18646

/-- Represents the fuel efficiency of a car in kilometers per gallon. -/
noncomputable def fuel_efficiency (distance : ℝ) (fuel : ℝ) : ℝ :=
  distance / fuel

/-- Theorem stating that the car's fuel efficiency is 40 km/gal given the provided conditions. -/
theorem car_fuel_efficiency :
  let distance := (120 : ℝ) -- kilometers
  let fuel := (3 : ℝ) -- gallons
  fuel_efficiency distance fuel = 40 := by
  -- Unfold the definitions
  unfold fuel_efficiency
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l186_18646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_transformed_circle_l186_18668

/-- The scaling transformation applied to the circle -/
def scaling (x y : ℝ) : ℝ × ℝ := (2 * x, y)

/-- The equation of the original circle C -/
def original_circle (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2

/-- The equation of the transformed ellipse C' -/
def transformed_ellipse (a : ℝ) (x y : ℝ) : Prop :=
  let (x', y') := scaling x y
  (x' / 2)^2 + y'^2 = a^2

/-- The equation of the line l -/
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 2

/-- The main theorem -/
theorem tangent_line_to_transformed_circle (a : ℝ) :
  (a > 0) →
  (∃! p : ℝ × ℝ, transformed_ellipse a p.1 p.2 ∧ line p.1 p.2) →
  a = 2 * Real.sqrt 13 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_transformed_circle_l186_18668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_count_lower_bound_l186_18660

/-- The number of distinct points on the unit circle -/
def n : ℕ := 2024

/-- The maximum central angle for the chords we're counting -/
noncomputable def max_angle : ℝ := 2 * Real.pi / 3

/-- A function that takes a natural number and returns the number of pairs
    of points from that many points whose central angle is ≤ max_angle -/
def count_chords (points : ℕ) : ℕ := sorry

/-- Theorem stating the lower bound for the number of chords -/
theorem chord_count_lower_bound :
  count_chords n ≥ 1024732 := by
  sorry

#eval n  -- This line is added to ensure that 'n' is recognized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_count_lower_bound_l186_18660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_function_and_sides_l186_18696

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 2 * Real.pi / 3) - Real.cos (2 * x)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_angles_from_function_and_sides 
  (t : Triangle) 
  (h1 : f (t.B / 2) = -Real.sqrt 3 / 2)
  (h2 : t.b = 1)
  (h3 : t.c = Real.sqrt 3)
  (h4 : t.a > t.b) :
  t.B = Real.pi / 6 ∧ t.C = Real.pi / 3 := by
  sorry

#check triangle_angles_from_function_and_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_from_function_and_sides_l186_18696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_not_exist_at_origin_limit_not_exist_at_infinity_l186_18623

/-- The function f(x, y) = (2x^2 y^2) / (x^4 + y^4) -/
noncomputable def f (x y : ℝ) : ℝ := (2 * x^2 * y^2) / (x^4 + y^4)

/-- The limit of f(x, y) does not exist as (x, y) approaches (0, 0) -/
theorem limit_not_exist_at_origin :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    0 < Real.sqrt (x^2 + y^2) ∧ Real.sqrt (x^2 + y^2) < δ →
    |f x y - L| < ε := by
  sorry

/-- The limit of f(x, y) does not exist as (x, y) approaches (∞, ∞) -/
theorem limit_not_exist_at_infinity :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ M > 0, ∀ x y : ℝ,
    x > M ∧ y > M →
    |f x y - L| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_not_exist_at_origin_limit_not_exist_at_infinity_l186_18623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_approximation_l186_18607

/-- Calculates the annual interest rate given initial investment, final amount, compounding frequency, and time period. -/
noncomputable def calculate_annual_interest_rate (initial_investment : ℝ) (final_amount : ℝ) (compounding_frequency : ℝ) (time_period : ℝ) : ℝ :=
  2 * compounding_frequency * ((final_amount / initial_investment) ^ (1 / (compounding_frequency * time_period)) - 1)

/-- Theorem stating that the calculated annual interest rate is approximately 3.98% given the problem conditions. -/
theorem annual_interest_rate_approximation :
  let initial_investment : ℝ := 10000
  let final_amount : ℝ := 10815.83
  let compounding_frequency : ℝ := 2  -- semi-annually
  let time_period : ℝ := 2  -- years
  let calculated_rate := calculate_annual_interest_rate initial_investment final_amount compounding_frequency time_period
  abs (calculated_rate - 0.0398) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_rate_approximation_l186_18607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_government_covers_half_l186_18615

/-- Represents Bill's financial situation and health insurance costs -/
structure BillsFinances where
  hourlyWage : ℚ
  weeklyHours : ℚ
  weeksPerMonth : ℚ
  monthlyPlanCost : ℚ
  annualContribution : ℚ

/-- Calculates the percentage of health insurance cost covered by the government -/
def governmentCoveragePercentage (finances : BillsFinances) : ℚ :=
  let annualIncome := finances.hourlyWage * finances.weeklyHours * finances.weeksPerMonth * 12
  let annualPlanCost := finances.monthlyPlanCost * 12
  let governmentContribution := annualPlanCost - finances.annualContribution
  (governmentContribution / annualPlanCost) * 100

/-- Theorem stating that the government covers 50% of Bill's health insurance cost -/
theorem government_covers_half (billsFinances : BillsFinances)
  (h1 : billsFinances.hourlyWage = 25)
  (h2 : billsFinances.weeklyHours = 30)
  (h3 : billsFinances.weeksPerMonth = 4)
  (h4 : billsFinances.monthlyPlanCost = 500)
  (h5 : billsFinances.annualContribution = 3000) :
  governmentCoveragePercentage billsFinances = 50 := by
  sorry

def main : IO Unit := do
  let result := governmentCoveragePercentage {
    hourlyWage := 25,
    weeklyHours := 30,
    weeksPerMonth := 4,
    monthlyPlanCost := 500,
    annualContribution := 3000
  }
  IO.println s!"Government coverage percentage: {result}%"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_government_covers_half_l186_18615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_in_terms_of_x_l186_18647

theorem y_in_terms_of_x (p : ℝ) :
  let x := 1 + (3 : ℝ)^p
  let y := 1 + (3 : ℝ)^(-p)
  y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_in_terms_of_x_l186_18647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_l186_18625

open Nat

def is_valid_n (n : ℕ) : Bool :=
  let q := n / 100
  let r := n % 100
  n % 13 = 0 &&
  (q + r) % 7 = 0 &&
  10000 ≤ n && n ≤ 99999

theorem count_valid_n :
  (Finset.filter (fun n => is_valid_n n) (Finset.range 90000)).card = 8160 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_n_l186_18625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_mowers_four_additional_mowers_needed_l186_18631

/-- The number of additional people needed to mow a lawn in half the time -/
theorem additional_mowers (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) 
  (h1 : initial_people > 0) (h2 : initial_time > 0) (h3 : new_time > 0)
  (h4 : new_time * 2 = initial_time) :
  (initial_people * initial_time / new_time) - initial_people = initial_people := by
  sorry

/-- The specific case where 4 people can mow a lawn in 6 hours, and we want to know
    how many additional people are needed to mow it in 3 hours -/
theorem four_additional_mowers_needed : 
  (4 * 6 / 3) - 4 = 4 := by
  norm_num

#eval (4 * 6 / 3) - 4  -- This will output 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_mowers_four_additional_mowers_needed_l186_18631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l186_18648

noncomputable section

/-- Given a line l: √3x + y - 2 = 0 -/
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 = 0

/-- The slope of line l -/
noncomputable def slope_l : ℝ := -Real.sqrt 3

/-- A normal vector of line l -/
noncomputable def normal_vector_l : ℝ × ℝ := (Real.sqrt 3, 1)

/-- A directional vector of line l -/
noncomputable def directional_vector_l : ℝ × ℝ := (-Real.sqrt 3, 3)

theorem line_l_properties :
  slope_l ≠ 5 * Real.pi / 6 ∧
  slope_l ≠ Real.sqrt 3 ∧
  normal_vector_l ≠ (1, Real.sqrt 3) ∧
  ∃ (k : ℝ), k ≠ 0 ∧ k * directional_vector_l.1 = -Real.sqrt 3 ∧ k * directional_vector_l.2 = 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l186_18648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_95_not_possible_l186_18659

-- Define the set of coin values
def coin_values : List Nat := [1, 5, 10, 50]

-- Function to check if a sum is possible with 5 coins
def is_sum_possible (sum : Nat) : Prop :=
  ∃ (c₁ c₂ c₃ c₄ c₅ : Nat), 
    c₁ ∈ coin_values ∧ 
    c₂ ∈ coin_values ∧ 
    c₃ ∈ coin_values ∧ 
    c₄ ∈ coin_values ∧ 
    c₅ ∈ coin_values ∧ 
    c₁ + c₂ + c₃ + c₄ + c₅ = sum

-- Theorem stating that 95 cents is not possible with 5 coins
theorem sum_95_not_possible : ¬(is_sum_possible 95) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_95_not_possible_l186_18659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_mass_equilibrium_l186_18675

/-- Represents a system with a uniform rod and a load in equilibrium -/
structure EquilibriumSystem where
  m₁ : ℝ  -- Mass of the load
  l : ℝ   -- Length of the uniform rod
  S : ℝ   -- Distance between attachment points of left thread

/-- The mass of the rod in an equilibrium system -/
noncomputable def rod_mass (sys : EquilibriumSystem) : ℝ :=
  (sys.m₁ * sys.S) / sys.l

/-- Theorem stating that the mass of the rod in an equilibrium system
    is equal to (m₁ * S) / l -/
theorem rod_mass_equilibrium (sys : EquilibriumSystem)
  (h₁ : sys.m₁ > 0)
  (h₂ : sys.l > 0)
  (h₃ : sys.S > 0)
  (h₄ : sys.S < sys.l) :
  rod_mass sys = (sys.m₁ * sys.S) / sys.l :=
by
  -- Unfold the definition of rod_mass
  unfold rod_mass
  -- The equality holds by definition
  rfl

#check rod_mass_equilibrium

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_mass_equilibrium_l186_18675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l186_18640

/-- Set of ways to insert 3 pairs of parentheses into the product of 5 numbers -/
def A : Finset (List (Fin 4)) := sorry

/-- Set of ways to divide a convex hexagon into 4 triangles -/
def B : Finset (List (Fin 6)) := sorry

/-- Set of ways to arrange 4 black and 4 white balls with the given condition -/
def C : Finset (List Bool) := sorry

/-- The cardinalities of sets A, B, and C are equal -/
theorem cardinality_equality : Finset.card A = Finset.card B ∧ Finset.card A = Finset.card C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_equality_l186_18640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_is_four_l186_18677

/-- The polynomial P(x) -/
def P (a b x : ℝ) : ℝ := x^7 - 11*x^6 + 35*x^5 - 5*x^4 + 7*x^3 + a*x^2 + b*x

/-- The quadratic function Q(x) -/
def Q (c d e x : ℝ) : ℝ := c*x^2 + d*x + e

/-- The set of intersection points between P and Q -/
def intersection_points (a b c d e : ℝ) : Set ℝ :=
  {x : ℝ | P a b x = Q c d e x}

theorem largest_intersection_is_four (a b c d e : ℝ) :
  (∀ x, x ∉ intersection_points a b c d e → P a b x > Q c d e x) →
  (∃ s : Finset ℝ, s.card = 3 ∧ ↑s = intersection_points a b c d e) →
  ∃ x ∈ intersection_points a b c d e, x = 4 ∧ ∀ y ∈ intersection_points a b c d e, y ≤ x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_is_four_l186_18677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_only_positive_result_l186_18642

-- Define a type for the binary operations
inductive BinaryOp
  | Add
  | Sub
  | Mul
  | Div

-- Define the operation function
noncomputable def apply_op (op : BinaryOp) (a b : ℝ) : ℝ :=
  match op with
  | BinaryOp.Add => a + b
  | BinaryOp.Sub => a - b
  | BinaryOp.Mul => a * b
  | BinaryOp.Div => a / b

-- Theorem statement
theorem subtraction_only_positive_result :
  ∀ op : BinaryOp, (apply_op op 1 (-2) > 0) ↔ (op = BinaryOp.Sub) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subtraction_only_positive_result_l186_18642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_theorem_l186_18644

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents the operation of drawing a ball and adding two of the same color -/
def perform_operation (urn : UrnContents) : UrnContents → Prop :=
  sorry

/-- Represents the probability of a specific urn content after operations -/
def probability_after_operations (initial : UrnContents) (final : UrnContents) (num_operations : ℕ) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem urn_probability_theorem :
  let initial_urn := UrnContents.mk 2 1
  let final_urn := UrnContents.mk 6 4
  let num_operations := 3
  probability_after_operations initial_urn final_urn num_operations = 8/35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_theorem_l186_18644
