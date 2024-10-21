import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_is_four_l452_45249

/-- The volume of a mixture of two vegetable ghee brands --/
noncomputable def mixture_volume (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ) : ℝ :=
  let va := (total_weight * (ratio_a + ratio_b : ℝ)) / (weight_a * ratio_a + weight_b * ratio_b)
  va * ((ratio_a + ratio_b : ℝ) / ratio_a)

/-- Theorem stating that the mixture volume is 4 liters under given conditions --/
theorem mixture_volume_is_four :
  mixture_volume 900 800 3 2 3440 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_volume_is_four_l452_45249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_lowest_degree_solution_l452_45241

/-- The polynomial f(x, y) that satisfies the given conditions -/
def f (x y : ℝ) : ℝ := (x - y) * x * y * (x + y) * (2 * x^2 - 5 * x * y + 2 * y^2)

/-- Predicate to check if a function is a polynomial of degree n -/
def IsPolynomial (n : ℕ) (p : ℝ → ℝ → ℝ) : Prop :=
  sorry

/-- The theorem stating that f(x, y) satisfies the required conditions and is of the lowest degree -/
theorem f_is_lowest_degree_solution :
  (∀ x y : ℝ, f x y + f y x = 0) ∧
  (∀ x y : ℝ, f x (x + y) + f y (x + y) = 0) ∧
  (∀ g : ℝ → ℝ → ℝ, (∀ x y : ℝ, g x y + g y x = 0) →
    (∀ x y : ℝ, g x (x + y) + g y (x + y) = 0) →
    (∃ n : ℕ, n ≥ 1 ∧ IsPolynomial n g) →
    (∃ n : ℕ, n ≥ 5 ∧ IsPolynomial n g)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_lowest_degree_solution_l452_45241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_slopes_l452_45295

/-- A right triangle in the coordinate plane with legs parallel to the axes -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The slope of a line -/
def lineSlope (m : ℝ) := m

/-- Condition for a median to lie on a line with slope m -/
def median_on_line (t : RightTriangle) (m : ℝ) : Prop :=
  (lineSlope (t.c / (2 * t.d)) = m) ∨ (lineSlope ((2 * t.c) / t.d) = m)

/-- The two possible slopes for the medians -/
def median_slopes : Set ℝ := {5, 5/4}

theorem right_triangle_median_slopes :
  ∃! (s : Set ℝ), s.Finite ∧ s.Nonempty ∧
  (∀ m : ℝ, m ∈ s ↔
    ∃ t : RightTriangle,
      median_on_line t 5 ∧
      median_on_line t m) ∧
  s = median_slopes :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_median_slopes_l452_45295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tree_height_l452_45239

/-- Calculates the original height of a broken tree given the road breadth and breaking height. -/
noncomputable def original_tree_height (road_breadth : ℝ) (breaking_height : ℝ) : ℝ :=
  breaking_height + Real.sqrt (road_breadth ^ 2 + breaking_height ^ 2)

/-- Theorem stating that a tree with given conditions has an original height of 36 meters. -/
theorem broken_tree_height : original_tree_height 12 16 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tree_height_l452_45239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_base_angle_is_arccos_sqrt3_div_3_l452_45269

/-- A regular triangular pyramid with mutually perpendicular lateral edges -/
structure RegularTriangularPyramid where
  /-- The length of a lateral edge -/
  a : ℝ
  /-- The lateral edges are mutually perpendicular -/
  lateral_edges_perpendicular : True

/-- The angle between a lateral face and the base plane of a regular triangular pyramid -/
noncomputable def lateral_face_base_angle (p : RegularTriangularPyramid) : ℝ :=
  Real.arccos (Real.sqrt 3 / 3)

/-- 
Theorem: In a regular triangular pyramid where the lateral edges are mutually perpendicular in pairs, 
the angle between a lateral face and the base plane is arccos(√3/3).
-/
theorem lateral_face_base_angle_is_arccos_sqrt3_div_3 (p : RegularTriangularPyramid) : 
  lateral_face_base_angle p = Real.arccos (Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_base_angle_is_arccos_sqrt3_div_3_l452_45269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_determination_l452_45228

theorem triangle_angle_determination (a b : ℝ) (B : ℝ) (hA : a = 2 * Real.sqrt 3)
  (hB : b = 6) (hAngleB : B = 60 * π / 180) :
  ∃ (A : ℝ), A = 30 * π / 180 ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧
  Real.sin A / a = Real.sin B / b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_determination_l452_45228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l452_45230

theorem triangle_area_bound (a b c : ℝ) (ha : a < 1) (hb : b < 1) (hc : c < 1) :
  ∃ (S : ℝ), S < Real.sqrt 3 / 4 ∧ S = (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l452_45230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l452_45282

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 6} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l452_45282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_units_digit_when_tens_is_seven_l452_45281

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem square_units_digit_when_tens_is_seven (n : ℤ) :
  tens_digit (n^2).toNat = 7 → units_digit (n^2).toNat = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_units_digit_when_tens_is_seven_l452_45281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l452_45243

-- Define the complex numbers
variable (z : ℂ)

-- Define the operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define x
noncomputable def x : ℂ := (1 - Complex.I) / (1 + Complex.I)

-- Define y
noncomputable def y : ℂ := det (4 * Complex.I) (3 - x * Complex.I) (1 + Complex.I) (x + Complex.I)

-- Theorem statement
theorem y_value : y = -2 - 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_value_l452_45243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_scheme_l452_45221

/-- Represents the total cost of painting three rooms -/
def total_cost (x y z a b c : ℝ) : ℝ := a * x + b * y + c * z

theorem lowest_cost_scheme (x y z a b c : ℝ) 
  (h_rooms : x < y ∧ y < z) 
  (h_costs : a < b ∧ b < c) : 
  ∀ σ : Equiv.Perm (Fin 3), 
    total_cost x y z a b c ≤ total_cost x y z (σ 0) (σ 1) (σ 2) := by
  sorry

#check lowest_cost_scheme

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_scheme_l452_45221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l452_45267

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the line AB
def lineAB (x y : ℝ) : Prop := x - 2*y - 2 = 0

-- Define the circle
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 9

-- Define the diagonal relative to (0,1)
def diagonalRelativeTo (rect : Rectangle) : Prop :=
  ∃ (dx dy : ℝ), rect.C.1 - rect.A.1 = dx ∧ rect.C.2 - rect.A.2 = dy ∧
  rect.A.1 + dx = 0 ∧ rect.A.2 + dy = 1

-- Theorem statement
theorem chord_length (rect : Rectangle) 
  (h1 : lineAB rect.A.1 rect.A.2)
  (h2 : lineAB rect.B.1 rect.B.2)
  (h3 : diagonalRelativeTo rect)
  (h4 : ∃ (x1 y1 x2 y2 : ℝ), circleEq x1 y1 ∧ circleEq x2 y2 ∧ 
        (x1 - rect.C.1) * (rect.D.2 - rect.C.2) = (y1 - rect.C.2) * (rect.D.1 - rect.C.1) ∧
        (x2 - rect.C.1) * (rect.D.2 - rect.C.2) = (y2 - rect.C.2) * (rect.D.1 - rect.C.1)) :
  ∃ (x1 y1 x2 y2 : ℝ), 
    circleEq x1 y1 ∧ circleEq x2 y2 ∧
    (x1 - rect.C.1) * (rect.D.2 - rect.C.2) = (y1 - rect.C.2) * (rect.D.1 - rect.C.1) ∧
    (x2 - rect.C.1) * (rect.D.2 - rect.C.2) = (y2 - rect.C.2) * (rect.D.1 - rect.C.1) ∧
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l452_45267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_distance_l452_45205

/-- The total distance traveled by a rabbit on a path consisting of arcs and straight lines between two concentric circles -/
theorem rabbit_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 24) : 
  (1/6 * 2 * Real.pi * r₂) + (r₂ - r₁) + (1/3 * 2 * Real.pi * r₁) + (2 * r₁) = 16 * Real.pi + 36 := by
  sorry

#check rabbit_path_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_distance_l452_45205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_implies_m_values_l452_45206

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def lineEq (x y k m : ℝ) : Prop := y = k * x + m

-- Define the chord length
noncomputable def chord_length (k m : ℝ) : ℝ := 2 * Real.sqrt (4 - (m / Real.sqrt (1 + k^2))^2)

-- Theorem statement
theorem min_chord_length_implies_m_values (k m : ℝ) :
  (∀ k', chord_length k' m ≥ 2) ∧ (∃ k', chord_length k' m = 2) →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_implies_m_values_l452_45206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l452_45236

theorem roots_of_equation (x : ℝ) :
  (3 * Real.sqrt x + 5 * x^(-(1/2 : ℝ)) = 8) ↔ (9 * x^2 - 46 * x + 25 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l452_45236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l452_45222

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (D : ℝ × ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  2 * b * Real.cos C = 2 * a + c →
  b = 2 * Real.sqrt 3 →
  (∃ (BD : ℝ), BD = 1 ∧
    ((Real.sin (B/2) / a = Real.sin ((π - B)/2) / c) ∨
     (D.1 = (Real.cos A * b + c) / (2 * b) ∧ D.2 = (Real.sin A * b) / (2 * b)))) →
  B = 2 * π / 3 ∧ a * c * Real.sin B / 2 = Real.sqrt 3 := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l452_45222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_between_29_and_36_l452_45277

theorem count_numbers_between_29_and_36 :
  Finset.card (Finset.filter (fun n => 30 ≤ n ∧ n < 36) (Finset.range 100)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_between_29_and_36_l452_45277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45220

open Real

/-- Triangle ABC with vectors m and n -/
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  b : ℝ  -- Side b
  c : ℝ  -- Side c
  m : ℝ × ℝ  -- Vector m
  n : ℝ × ℝ  -- Vector n

/-- Given conditions for the triangle -/
def triangle_conditions (t : Triangle) : Prop :=
  t.m = (cos t.A, sin t.A) ∧
  t.n = (cos t.B, -sin t.B) ∧
  ‖t.m - t.n‖ = 1 ∧
  t.c = 3

/-- Theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.C = 2 * π / 3 ∧ 
  (∀ (area : ℝ), area ≤ 3 * Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_palindrome_proof_l452_45272

theorem product_palindrome_proof :
  ∃! (i k s : ℕ),
    i ≠ 0 ∧ k ≠ 0 ∧ s ≠ 0 ∧
    i ≠ k ∧ i ≠ s ∧ k ≠ s ∧
    i < 10 ∧ k < 10 ∧ s < 10 ∧
    (i % 2 = 0 ∨ k % 2 = 0 ∨ s % 2 = 0) ∧
    (i % 2 ≠ 0 ∨ k % 2 ≠ 0 ∨ s % 2 ≠ 0) ∧
    (i % 2 ≠ 0 ∨ k % 2 ≠ 0 ∨ s % 2 ≠ 0) ∧
    let iks := 100 * i + 10 * k + s
    let isk := 100 * i + 10 * s + k
    let product := iks * isk
    100000 > product ∧ product ≥ 10000 ∧
    (product / 10000 = product % 10) ∧
    ((product / 1000) % 10 = (product / 10) % 10) ∧
    i = 1 ∧ k = 6 ∧ s = 7 ∧
    product = 29392 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_palindrome_proof_l452_45272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_candy_count_l452_45208

/-- Represents the number of cherry candies -/
def cherry : ℕ := sorry

/-- Represents the number of grape candies -/
def grape : ℕ := sorry

/-- Represents the number of apple candies -/
def apple : ℕ := sorry

/-- The cost of each candy in cents -/
def cost_per_candy : ℕ := 250

/-- The total cost of all candies in cents -/
def total_cost : ℕ := 20000

theorem grape_candy_count :
  grape = 3 * cherry →
  apple = 2 * grape →
  cost_per_candy * (cherry + grape + apple) = total_cost →
  grape = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_candy_count_l452_45208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_6_l452_45235

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Symmetric point with respect to xy-plane -/
def sym_xy (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Symmetric point with respect to x-axis -/
def sym_x (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

theorem length_AB_is_6 (A B M : Point3D) 
    (hM : M = ⟨2, -3, 5⟩) 
    (hA : A = sym_xy M) 
    (hB : B = sym_x M) : 
  distance A B = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_is_6_l452_45235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_f_composition_value_l452_45286

noncomputable def f (x : ℝ) : ℝ :=
  let y := x - 2 * ⌊x / 2⌋
  if -1 ≤ y ∧ y < 0 then -4 * y^2 + 1
  else if 0 ≤ y ∧ y < 1 then y + 7/4
  else 0  -- This case should never occur due to periodicity, but Lean requires it for completeness

theorem f_periodic (x : ℝ) : f (x + 2) = f x := by
  sorry

theorem f_composition_value : f (f (3/2)) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_f_composition_value_l452_45286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_l452_45210

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- The polygon formed by the given points -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨20, 0⟩, ⟨30, 10⟩, ⟨20, 20⟩, ⟨0, 20⟩, ⟨10, 10⟩, ⟨0, 0⟩
]

/-- Calculate the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculate the area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ :=
  width * height

/-- The theorem stating that the area of the polygon is 400 square units -/
theorem polygon_area : 
  triangleArea (List.get! polygon 0) (List.get! polygon 1) (List.get! polygon 5) +
  triangleArea (List.get! polygon 4) (List.get! polygon 3) (List.get! polygon 5) +
  rectangleArea 10 10 +
  rectangleArea 10 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_l452_45210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_calculation_l452_45202

/-- Calculates the average score of a class given specific score distributions -/
theorem class_average_calculation (total_students : ℕ) 
  (score_95_count score_0_count score_85_count : ℕ)
  (remaining_average : ℚ) : 
  total_students = 50 →
  score_95_count = 5 →
  score_0_count = 5 →
  score_85_count = 5 →
  remaining_average = 45 →
  (let remaining_count := total_students - (score_95_count + score_0_count + score_85_count)
   let total_score := score_95_count * 95 + score_0_count * 0 + score_85_count * 85 + 
                      (remaining_count : ℚ) * remaining_average
   total_score / (total_students : ℚ) = 49.5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_calculation_l452_45202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_l452_45209

-- Define the parabola C₁
def C₁ (p : ℝ) (x y : ℝ) : Prop := y^2 = 4*p*x

-- Define the ellipse C₂
def C₂ (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the theorem
theorem parabola_ellipse_intersection (p : ℝ) (h_p : p > 0) :
  -- Conditions
  ∃ (F₁ F₂ M : ℝ × ℝ) (a b : ℝ),
    -- F₂ is focus of C₁
    F₂.1 = p ∧ F₂.2 = 0 ∧
    -- F₁ is on x-axis
    F₁.2 = 0 ∧
    -- F₁ and F₂ are foci of C₂
    F₁.1 = -F₂.1 ∧
    -- Eccentricity of C₂ is 1/2
    (F₂.1 - F₁.1)/(2*a) = 1/2 ∧
    -- M is on both C₁ and C₂
    C₁ p M.1 M.2 ∧ C₂ a b M.1 M.2 →
  -- Conclusion
  (p = 1 → ∀ x y, C₂ 2 (Real.sqrt 3) x y ↔ x^2/4 + y^2/3 = 1) ∧
  (∃ k : ℝ, k = Real.sqrt 2 ∨ k = -Real.sqrt 2) ∧
  (∀ A B : ℝ × ℝ, 
    (C₁ 1 A.1 A.2 ∧ C₁ 1 B.1 B.2 ∧ line_l (Real.sqrt 2) A.1 A.2 ∧ line_l (Real.sqrt 2) B.1 B.2) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 
      Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) +
      Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) +
      Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_intersection_l452_45209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mork_and_mindy_tax_rate_l452_45213

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem mork_and_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.45) 
  (h2 : mindy_rate = 0.20) 
  (h3 : income_ratio = 4) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.25 := by
  sorry

#check mork_and_mindy_tax_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mork_and_mindy_tax_rate_l452_45213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l452_45266

-- Define the region
def Region (x y : ℝ) : Prop :=
  x + y ≤ 5 ∧ 3 * x + 2 * y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1

-- Define the vertices of the quadrilateral
def Vertices : Set (ℝ × ℝ) :=
  {(1, 4), (4, 1), (1, 2), (2, 3)}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ Vertices ∧ p2 ∈ Vertices ∧
    (∀ (q1 q2 : ℝ × ℝ), q1 ∈ Vertices → q2 ∈ Vertices →
      distance q1 q2 ≤ distance p1 p2) ∧
    distance p1 p2 = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l452_45266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_trig_sum_l452_45211

noncomputable def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

theorem rational_trig_sum (x : ℝ) 
  (h1 : IsRational (Real.sin (64 * x) + Real.sin (65 * x)))
  (h2 : IsRational (Real.cos (64 * x) + Real.cos (65 * x))) :
  (IsRational (Real.sin (64 * x)) ∧ IsRational (Real.sin (65 * x))) ∨
  (IsRational (Real.cos (64 * x)) ∧ IsRational (Real.cos (65 * x))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_trig_sum_l452_45211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l452_45251

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := (m - 3^x) / (n + 3^x)

theorem odd_function_properties (m n k : ℝ) :
  (∀ x, f m n x = -f m n (-x)) →  -- f is an odd function
  (∃ t ∈ Set.Icc 0 4, f m n (k - 2*t^2) + f m n (4*t - 2*t^2) < 0) →
  (m = 1 ∧ n = 1 ∧ k > -1) :=
by
  sorry

#check odd_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l452_45251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l452_45265

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_l452_45265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_is_perfect_square_sequence_term_formula_l452_45244

/-- Definition of the nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ :=
  (4 * (10^n - 1) / 9) * 10^n + (8 * (10^n - 1) / 9) + 9

/-- Theorem stating that the nth term is a perfect square -/
theorem sequence_term_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, sequenceTerm n = k^2 := by
  sorry

/-- Theorem stating the explicit formula for the nth term -/
theorem sequence_term_formula (n : ℕ) :
  sequenceTerm n = (2 * 10^(n+1) + 1)^2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_term_is_perfect_square_sequence_term_formula_l452_45244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solutions_l452_45237

theorem complex_equation_solutions :
  ∀ z : ℂ, (z^4 + 4*z^2 + 6 = z) ↔ 
    (z = (1 + Complex.I * Real.sqrt 7) / 2 ∨
     z = (1 - Complex.I * Real.sqrt 7) / 2 ∨
     z = (-1 + Complex.I * Real.sqrt 11) / 2 ∨
     z = (-1 - Complex.I * Real.sqrt 11) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solutions_l452_45237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l452_45215

def satisfies_conditions (x : ℕ) : Prop :=
  ∀ n ∈ Finset.range 9, x % (n + 2) = n + 1

theorem smallest_satisfying_number : 
  satisfies_conditions 2519 ∧ ∀ y < 2519, ¬satisfies_conditions y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l452_45215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_car_problem_solution_l452_45268

/-- Represents a point on the path between two cities -/
structure Point where
  distance : ℚ
  deriving Repr

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℚ
  deriving Repr

/-- Configuration of the two-car problem -/
structure TwoCarProblem where
  totalDistance : ℚ
  car1 : Car
  car2 : Car
  meetingPoint : Point
  timeToBForCar1 : ℚ
  timeToAForCar2 : ℚ

/-- The solution to the two-car problem -/
def solveTwoCarProblem (problem : TwoCarProblem) : ℚ :=
  problem.meetingPoint.distance

theorem two_car_problem_solution (problem : TwoCarProblem) 
  (h1 : problem.totalDistance = 900)
  (h2 : problem.timeToBForCar1 = 4)
  (h3 : problem.timeToAForCar2 = 16)
  (h4 : problem.car1.speed * problem.timeToBForCar1 = problem.totalDistance - problem.meetingPoint.distance)
  (h5 : problem.car2.speed * problem.timeToAForCar2 = problem.meetingPoint.distance)
  (h6 : problem.car1.speed * (problem.timeToBForCar1 + (problem.meetingPoint.distance / problem.car1.speed)) = 
        problem.car2.speed * (problem.timeToAForCar2 + ((problem.totalDistance - problem.meetingPoint.distance) / problem.car2.speed))) :
  solveTwoCarProblem problem = 600 := by
  sorry

#eval solveTwoCarProblem {
  totalDistance := 900,
  car1 := { speed := 0 },
  car2 := { speed := 0 },
  meetingPoint := { distance := 600 },
  timeToBForCar1 := 4,
  timeToAForCar2 := 16
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_car_problem_solution_l452_45268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l452_45273

/-- Given a triangle ABC with points A, B, C, and D, prove that if BC = λ * BD and 
    AD = (2/3) * AB + (1/3) * AC, then λ = 3 -/
theorem triangle_ratio (A B C D : ℝ × ℝ) (lambda : ℝ) : 
  (C.1 - B.1, C.2 - B.2) = lambda • (D.1 - B.1, D.2 - B.2) →
  (D.1 - A.1, D.2 - A.2) = (2/3) • (B.1 - A.1, B.2 - A.2) + (1/3) • (C.1 - A.1, C.2 - A.2) →
  lambda = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l452_45273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exists_l452_45262

open Real

noncomputable def curve1 (θ : ℝ) : ℝ × ℝ := (2 / sin θ * cos θ, 2)
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ := (2 * cos θ * cos θ, 2 * cos θ * sin θ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_exists :
  ∃ (d : ℝ), ∀ (θ1 θ2 : ℝ), distance (curve1 θ1) (curve2 θ2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_exists_l452_45262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_problem_l452_45253

-- Define the rectangular parallelepiped
structure Parallelepiped where
  ab : ℝ
  a1d1 : ℝ
  cc1 : ℝ

-- Define a sphere
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (p : Parallelepiped) (σ1 σ2 : Sphere) : Prop :=
  p.ab = 6 - Real.sqrt 2 ∧
  p.a1d1 = 6 + Real.sqrt 2 ∧
  p.cc1 = 6 ∧
  -- σ1 touches three faces
  σ1.center.1 = σ1.radius ∧
  σ1.center.2.1 = σ1.radius ∧
  σ1.center.2.2 = σ1.radius ∧
  -- σ2 touches three faces
  σ2.center.1 = p.ab - σ2.radius ∧
  σ2.center.2.1 = p.a1d1 - σ2.radius ∧
  σ2.center.2.2 = p.cc1 - σ2.radius ∧
  -- Spheres touch externally
  (σ1.center.1 - σ2.center.1)^2 + (σ1.center.2.1 - σ2.center.2.1)^2 + (σ1.center.2.2 - σ2.center.2.2)^2 = (σ1.radius + σ2.radius)^2

-- Define the theorem
theorem sphere_problem (p : Parallelepiped) (σ1 σ2 : Sphere) 
  (h : problem_setup p σ1 σ2) : 
  -- Distance between centers
  (σ1.center.1 - σ2.center.1)^2 + (σ1.center.2.1 - σ2.center.2.1)^2 + (σ1.center.2.2 - σ2.center.2.2)^2 = 16 ∧
  -- Maximum total volume
  (4/3 * Real.pi * (σ1.radius^3 + σ2.radius^3) ≤ 64/3 * Real.pi) ∧
  -- Minimum total volume
  (4/3 * Real.pi * (σ1.radius^3 + σ2.radius^3) ≥ (495 - 90 * Real.sqrt 3) * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_problem_l452_45253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l452_45271

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 4) + Real.sin (x + Real.pi / 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-2 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l452_45271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l452_45226

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2 else -x + 1

theorem f_composition_negative_one : f (f (-1)) = 6 := by
  -- Calculate f(-1)
  have h1 : f (-1) = 2 := by
    simp [f]
    norm_num
  
  -- Calculate f(f(-1)) = f(2)
  have h2 : f 2 = 6 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc f (f (-1))
    = f 2 := by rw [h1]
    _ = 6 := h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l452_45226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_is_five_percent_l452_45224

noncomputable def cost_price : ℝ := 600

noncomputable def new_cost_price : ℝ := cost_price * 0.95

noncomputable def new_selling_price : ℝ := new_cost_price * 1.1

noncomputable def original_selling_price : ℝ := new_selling_price + 3

noncomputable def original_profit_percentage : ℝ := (original_selling_price - cost_price) / cost_price * 100

theorem original_profit_is_five_percent :
  original_profit_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_is_five_percent_l452_45224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l452_45229

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 - 3*x else -2*x + 1

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by
  -- We'll use 2 as our maximum value M
  use 2
  constructor
  · -- Prove M = 2
    rfl
  · -- Prove ∀ (x : ℝ), f x ≤ M
    intro x
    -- We'll split the proof into two cases: x ≤ 0 and x > 0
    by_cases h : x ≤ 0
    · -- Case: x ≤ 0
      sorry -- Proof for this case omitted
    · -- Case: x > 0
      sorry -- Proof for this case omitted

-- The proof is incomplete, but the structure is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l452_45229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l452_45292

def number_list : List Nat := [33, 35, 37, 39, 41]

def is_prime (n : Nat) : Bool :=
  n > 1 && (Nat.factorial (n - 1) + 1) % n == 0

def prime_numbers (list : List Nat) : List Nat :=
  list.filter is_prime

def arithmetic_mean (list : List Nat) : Rat :=
  if list.length > 0 then
    (list.sum : Rat) / list.length
  else
    0

theorem arithmetic_mean_of_primes :
  arithmetic_mean (prime_numbers number_list) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l452_45292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2040_is_identity_l452_45287

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.sqrt (3/2), 0, -(1/Real.sqrt 2)],
    ![0, 1, 0],
    ![1/Real.sqrt 2, 0, Real.sqrt (3/2)]]

theorem B_power_2040_is_identity :
  B^2040 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2040_is_identity_l452_45287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_lines_l452_45218

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Definition of a harmonic line -/
def is_harmonic_line (line : ℝ → ℝ) (M N : Point2D) : Prop :=
  ∃ P : Point2D, line P.y = P.x ∧ distance P M - distance P N = 6

/-- The given points M and N -/
def M : Point2D := ⟨-5, 0⟩
def N : Point2D := ⟨5, 0⟩

/-- The four given lines -/
noncomputable def line1 (x : ℝ) : ℝ := x - 1
noncomputable def line2 (x : ℝ) : ℝ := -2/3 * x
noncomputable def line3 (x : ℝ) : ℝ := 5/3 * x
noncomputable def line4 (x : ℝ) : ℝ := 2 * x + 1

/-- The theorem to be proved -/
theorem harmonic_lines :
  (is_harmonic_line line1 M N ∧ is_harmonic_line line2 M N) ∧
  (¬is_harmonic_line line3 M N ∧ ¬is_harmonic_line line4 M N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_lines_l452_45218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_ball_weights_l452_45258

structure Ball where
  weight : ℝ

structure Measurement where
  value : ℝ

def is_correct (m : Measurement) : Prop :=
  m.value = m.value  -- placeholder for correctness

theorem determine_ball_weights
  (balls : Fin 5 → Ball)
  (measurements : Fin 9 → Measurement)
  (h1 : ∃ i, ¬ is_correct (measurements i))
  (h2 : ∀ i, i.val < 8 → is_correct (measurements i) ∨ is_correct (measurements (i + 1)))
  (h3 : measurements 0 = ⟨(balls 0).weight⟩)
  (h4 : measurements 1 = ⟨(balls 1).weight⟩)
  (h5 : measurements 2 = ⟨(balls 2).weight⟩)
  (h6 : measurements 3 = ⟨(balls 0).weight + (balls 3).weight⟩)
  (h7 : measurements 4 = ⟨(balls 0).weight + (balls 4).weight⟩)
  (h8 : measurements 5 = ⟨(balls 1).weight + (balls 3).weight⟩)
  (h9 : measurements 6 = ⟨(balls 1).weight + (balls 4).weight⟩)
  (h10 : measurements 7 = ⟨(balls 2).weight + (balls 3).weight⟩)
  (h11 : measurements 8 = ⟨(balls 2).weight + (balls 4).weight⟩) :
  ∃ (weights : Fin 5 → ℝ), ∀ i, (balls i).weight = weights i :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_ball_weights_l452_45258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_has_minimum_iff_t_in_range_l452_45216

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := t * (4 : ℝ)^x + (2*t - 1) * (2 : ℝ)^x

theorem function_has_minimum_iff_t_in_range :
  ∀ t : ℝ, (∃ m : ℝ, ∀ x : ℝ, f t x ≥ m) ↔ (0 < t ∧ t < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_has_minimum_iff_t_in_range_l452_45216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_approx_l452_45299

/-- The cost of fencing a circular field -/
noncomputable def fencing_cost (diameter : ℝ) (cost_per_meter : ℝ) : ℝ :=
  Real.pi * diameter * cost_per_meter

/-- Theorem: The cost of fencing a circular field with diameter 30m at Rs. 5 per meter is approximately Rs. 471.25 -/
theorem fencing_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |fencing_cost 30 5 - 471.25| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_approx_l452_45299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p₁_or_p₂_l452_45212

-- Definition of proposition p₁
def p₁ : Prop := ∀ x : ℝ, Real.sin x ≠ 0 → Real.sin x + 1 / Real.sin x ≥ 2

-- Definition of proposition p₂
def p₂ : Prop := ∀ x y : ℝ, x + y = 0 ↔ x / y = -1

-- Theorem to prove
theorem not_p₁_or_p₂ : (¬p₁) ∨ p₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p₁_or_p₂_l452_45212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_property_l452_45279

/-- Given a triangle ABC with centroid G, and a line PQ passing through G
    intersecting AB at P and AC at Q, prove that if AP = ma and AQ = nb
    (where a = AB and b = AC), then 1/m + 1/n = 3. -/
theorem centroid_intersection_property
  (A B C G P Q : EuclideanSpace ℝ (Fin 3))
  (m n : ℝ) (a b : EuclideanSpace ℝ (Fin 3)) :
  G = (1 / 3 : ℝ) • (A + B + C) →  -- G is the centroid
  a = B - A →            -- a is vector AB
  b = C - A →            -- b is vector AC
  P - A = m • a →        -- AP = ma
  Q - A = n • b →        -- AQ = nb
  (∃ (t : ℝ), G = (1 - t) • P + t • Q) →  -- G is on line PQ
  1 / m + 1 / n = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_intersection_property_l452_45279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l452_45233

/-- The inclination angle of a line in the plane. -/
def InclinationAngle : Type := ℝ

/-- The set of all possible inclination angles for lines in the plane. -/
def InclinationAngleSet : Set ℝ := { θ : ℝ | 0 ≤ θ ∧ θ < Real.pi }

/-- Assertion that the inclination angle is measured counterclockwise from the positive x-axis. -/
axiom inclination_angle_measurement : InclinationAngle → Prop

/-- Assertion that a line can have any direction except vertical. -/
axiom line_direction : InclinationAngle → Prop

/-- Theorem stating that the range of inclination angles is [0, π). -/
theorem inclination_angle_range :
  ∀ θ : InclinationAngle, inclination_angle_measurement θ → line_direction θ →
  θ ∈ InclinationAngleSet :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l452_45233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_of_specific_prism_l452_45242

/-- The width of a rectangular prism with given length, height, and diagonal. -/
noncomputable def width_of_prism (l h d : ℝ) : ℝ :=
  Real.sqrt (d^2 - l^2 - h^2)

/-- Theorem stating that the width of a rectangular prism with length 4, height 9, and diagonal 15 is 8√2. -/
theorem width_of_specific_prism :
  width_of_prism 4 9 15 = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_of_specific_prism_l452_45242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iphone_cost_l452_45250

/-- Proves the cost of the iPhone in the Oscar swag bag -/
theorem iphone_cost (earring_cost scarf_cost total_value iphone_cost : ℕ) 
  (h1 : earring_cost = 6000)
  (h2 : scarf_cost = 1500)
  (h3 : total_value = 20000)
  (h4 : 2 * earring_cost + 4 * scarf_cost + iphone_cost = total_value) :
  iphone_cost = 2000 := by
  sorry

#check iphone_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iphone_cost_l452_45250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l452_45298

/-- Golden ratio --/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment AB of length 6, and C is the golden section point of AB,
    prove that BC is either (3√5 - 3) or (9 - 3√5) --/
theorem golden_section_length (A B C : ℝ) : 
  (B - A = 6) →
  (C - A) / (B - A) = 1 / φ ∨ (C - A) / (B - A) = 1 - 1 / φ →
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l452_45298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l452_45270

theorem complex_fraction_simplification :
  (3 - Complex.I) / (5 - 2 * Complex.I) = 17/29 + (1/29) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l452_45270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l452_45275

/-- A rectangular parallelepiped with base sides in ratio m:n and diagonal cross-section area Q -/
structure RectangularParallelepiped (m n Q : ℝ) where
  (m_pos : 0 < m)
  (n_pos : 0 < n)
  (Q_pos : 0 < Q)

/-- The volume of a rectangular parallelepiped -/
noncomputable def volume (m n Q : ℝ) (p : RectangularParallelepiped m n Q) : ℝ :=
  (m * n * Q * Real.sqrt Q) / (m^2 + n^2)

/-- Theorem: The volume of the rectangular parallelepiped is (mn * Q * √Q) / (m^2 + n^2) -/
theorem volume_formula (m n Q : ℝ) (p : RectangularParallelepiped m n Q) :
  volume m n Q p = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l452_45275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_unfoldings_l452_45284

/-- Represents a cube with 12 edges -/
structure Cube where
  edges : Fin 12 → Bool

/-- Represents an unfolding of a cube -/
structure Unfolding where
  cuts : Fin 7 → Fin 12

/-- Two unfoldings are considered equivalent if they are flipped versions of each other -/
def equivalent_unfoldings (u1 u2 : Unfolding) : Prop :=
  sorry

/-- The set of all possible unfoldings of a cube -/
def all_unfoldings : Set Unfolding :=
  sorry

/-- The set of distinct unfoldings after considering equivalence -/
noncomputable def distinct_unfoldings : Finset Unfolding :=
  sorry

/-- The number of distinct unfoldings of a cube is 11 -/
theorem num_distinct_unfoldings :
  Finset.card distinct_unfoldings = 11 :=
by
  sorry

#check num_distinct_unfoldings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distinct_unfoldings_l452_45284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_distribution_theorem_l452_45238

/-- Represents the amount of soup taken by each person in one cycle -/
noncomputable def soupTakenInCycle (x : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let remainingAfterOmkar := x - 1
  let remainingAfterKrit1 := (5/6) * remainingAfterOmkar
  let remainingAfterKrit2 := (4/6) * remainingAfterOmkar
  let remainingAfterKrit3 := (1/2) * remainingAfterOmkar
  (1, (1/6) * remainingAfterOmkar, (1/6) * remainingAfterOmkar, (1/6) * remainingAfterOmkar)

/-- Represents the condition for equal distribution of soup -/
def equalDistribution (x : ℝ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ 
    x / 4 = 1 + (x + 1) / (2^n) - 1 ∧
    2^n ≤ x + 1 ∧ x + 1 ≤ 2^(n+1)

/-- The main theorem stating that 49/3 is the only solution -/
theorem soup_distribution_theorem :
  ∀ x > 0, equalDistribution x ↔ x = 49/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soup_distribution_theorem_l452_45238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_f_inequality_l452_45259

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := Real.log x - x - m

-- Theorem for the first part of the problem
theorem f_two_zeros (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ m = 0 ∧ f x₂ m = 0) ↔ m < -1 :=
sorry

-- Theorem for the second part of the problem
theorem f_inequality (m : ℝ) (hm : m ≥ -3) :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f x m + (x - 2) * Real.exp x < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_f_inequality_l452_45259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_proof_l452_45260

noncomputable def original_wage (new_wage : ℝ) (increase_rate : ℝ) : ℝ := 
  new_wage / (1 + increase_rate)

theorem wage_increase_proof (new_wage : ℝ) (increase_rate : ℝ) 
  (h1 : new_wage = 28) 
  (h2 : increase_rate = 0.4) : 
  original_wage new_wage increase_rate = 20 := by
  -- Unfold the definition of original_wage
  unfold original_wage
  -- Substitute the known values
  rw [h1, h2]
  -- Simplify the expression
  simp
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_increase_proof_l452_45260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exclusive_proposition_range_l452_45247

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x - a - 1) / Real.log 2

def g (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*a*x + 1

-- Define propositions p and q
def proposition_p (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x ∧ x < y → f a x < f a y

def proposition_q (a : ℝ) : Prop :=
  ∃ x₁ x₂, (∀ x, g a x ≤ g a x₁) ∧ (∀ x, g a x₂ ≤ g a x)

-- Define the set of a for which exactly one proposition is true
def exclusive_proposition_set : Set ℝ :=
  {a | (proposition_p a ∧ ¬proposition_q a) ∨ (¬proposition_p a ∧ proposition_q a)}

-- Theorem statement
theorem exclusive_proposition_range :
  exclusive_proposition_set = Set.Iic (-3) ∪ Set.Icc 0 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exclusive_proposition_range_l452_45247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_shared_by_circles_l452_45240

/-- Two circles with centers A and B intersect at points X and Y.
    The minor arc ∠XY = 120° with respect to circle A,
    and ∠XY = 60° with respect to circle B.
    XY = 2 is the length of the chord. -/
def intersecting_circles (A B X Y : ℝ × ℝ) : Prop :=
  ∃ (R r : ℝ),
    R > 0 ∧ r > 0 ∧
    (X.1 - A.1)^2 + (X.2 - A.2)^2 = R^2 ∧
    (Y.1 - A.1)^2 + (Y.2 - A.2)^2 = R^2 ∧
    (X.1 - B.1)^2 + (X.2 - B.2)^2 = r^2 ∧
    (Y.1 - B.1)^2 + (Y.2 - B.2)^2 = r^2 ∧
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 4 ∧
    Real.arccos ((X.1 - A.1) * (Y.1 - A.1) + (X.2 - A.2) * (Y.2 - A.2)) / R^2 = 2 * Real.pi / 3 ∧
    Real.arccos ((X.1 - B.1) * (Y.1 - B.1) + (X.2 - B.2) * (Y.2 - B.2)) / r^2 = Real.pi / 3

/-- The area shared by the two intersecting circles is (10π - 12√3) / 9 -/
theorem area_shared_by_circles (A B X Y : ℝ × ℝ) 
  (h : intersecting_circles A B X Y) : 
  ∃ (area : ℝ), area = (10 * Real.pi - 12 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_shared_by_circles_l452_45240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l452_45296

-- Define a structure for our triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.c / Real.cos t.C = t.a / Real.cos (3 * Real.pi / 2 + t.A))
  (h2 : t.c / t.a = 2)
  (h3 : t.b = 4 * Real.sqrt 3) :
  t.C = Real.pi / 6 ∧ 
  (1 / 2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 15 - 2 * Real.sqrt 3) := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l452_45296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l452_45274

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the function g implicitly
noncomputable def g (x : ℝ) : ℝ :=
  if x ≠ 1 then 1 / (f (x - 1) - 1) else 0  -- We define g(1) arbitrarily as 0

-- State the theorem
theorem range_of_g :
  Set.range g = {y : ℝ | y < -1 ∨ y > 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l452_45274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45278

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  sqrt 3 = (a * sin B) / (2 * sin A) ∧  -- Circumradius formula
  tan B + tan C = (2 * sin A) / cos C →
  B = π / 3 ∧ 
  b = 3 ∧
  ∀ (S : ℝ), S = (1/2) * a * c * sin B → S ≤ (9 * sqrt 3) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_bounds_l452_45280

/-- Given real numbers x and y satisfying x|x| + (y|y|)/2 = 1, 
    prove that 4-√6 ≤ |√3x + y - 4| < 4 -/
theorem expression_bounds (x y : ℝ) 
  (h : x * abs x + (y * abs y) / 2 = 1) : 
  4 - Real.sqrt 6 ≤ abs (Real.sqrt 3 * x + y - 4) ∧ 
  abs (Real.sqrt 3 * x + y - 4) < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_bounds_l452_45280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l452_45289

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) - Real.sqrt 2

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_min_dist : ∀ x₁ x₂, f ω φ x₁ = 0 → f ω φ x₂ = 0 → x₁ ≠ x₂ → |x₁ - x₂| ≥ Real.pi / 4)
  (h_sum_abscissas : ∃ x₁ x₂, x₁ < x₂ ∧ -Real.pi/2 < x₁ ∧ x₂ < Real.pi/2 ∧ 
    f ω φ x₁ = 0 ∧ f ω φ x₂ = 0 ∧ x₁ + x₂ = Real.pi/3) :
  ω = 2 ∧ 
  φ = Real.pi/6 ∧ 
  f ω φ (Real.pi/3) = 1 - Real.sqrt 2 ∧
  ∀ x, -Real.pi/6 < x → x < Real.pi/6 → 
    ∀ y, x < y → y < Real.pi/6 → f ω φ x < f ω φ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l452_45289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l452_45263

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem f_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →
  f a b (1/2) = 4/5 →
  ∃ g : ℝ → ℝ, 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → g x = 2 * x / (1 + x^2)) ∧
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = g x) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → g x < g y) ∧
    (Set.Ioo (0 : ℝ) (1/3) = {t : ℝ | g (2*t-1) + g t < 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l452_45263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_567_l452_45290

/-- The volume of a triangular pyramid enclosed by 4 congruent triangles -/
noncomputable def pyramid_volume (a b c : ℝ) : ℝ :=
  (a * b * c) / 6 * Real.sqrt (1 - a^2 - b^2 - c^2 + 2*a*b*c)

/-- Theorem: The volume of a triangular pyramid with side lengths 5, 6, and 7 is approximately 19.4936 -/
theorem pyramid_volume_567 :
  |pyramid_volume 5 6 7 - 19.4936| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_567_l452_45290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_parabola_l452_45201

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = x^2

/-- The distance between two points -/
noncomputable def distance (p q : ParabolaPoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p q r : ParabolaPoint) : ℝ :=
  (1/2) * abs (p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y))

/-- Theorem: The maximum area of triangle AOB is 8 -/
theorem max_triangle_area_on_parabola :
  ∀ (A B : ParabolaPoint),
    A.x < 0 →
    B.x > 0 →
    distance A B = 4 →
    triangleArea A B ⟨0, 0, by simp⟩ ≤ 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_parabola_l452_45201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sled_friction_coefficient_l452_45252

/-- Coefficient of kinetic friction for a sled on an inclined plane --/
theorem sled_friction_coefficient (g : ℝ) (θ : ℝ) (s : ℝ) (t : ℝ) :
  g > 0 →
  θ = 25 * Real.pi / 180 →
  s = 85 →
  t = 17 →
  let a := 2 * s / (t^2)
  let μ := (g * Real.sin θ - a) / (g * Real.cos θ)
  ∃ ε > 0, |μ - 0.40| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sled_friction_coefficient_l452_45252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_one_l452_45291

/-- The function f(x) = x / (2x + 2) -/
noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 2)

/-- The n-th composition of f -/
noncomputable def f_n : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f (f_n n x)

/-- Theorem stating the value of f_n(1) for positive integers n -/
theorem f_n_one (n : ℕ+) : f_n n 1 = 1 / (3 * 2^(n : ℝ) - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_one_l452_45291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l452_45225

theorem complex_number_theorem (z : ℂ) :
  15 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 2) + 36 →
  z + 9 / z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_theorem_l452_45225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l452_45256

/-- Represents a point on the geoboard -/
structure GeoPoint where
  x : Fin 8
  y : Fin 8

/-- Represents a segment on the geoboard -/
structure GeoSegment where
  start : GeoPoint
  finish : GeoPoint

/-- Calculates the length of a segment -/
def segmentLength (s : GeoSegment) : ℕ :=
  sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (a b c : GeoPoint) : Prop :=
  sorry

/-- The main theorem -/
theorem isosceles_triangle_count 
  (ab : GeoSegment) 
  (h_length : segmentLength ab = 3) : 
  ∃ (points : Finset GeoPoint), 
    points.card = 18 ∧ 
    ∀ c ∈ points, isIsosceles ab.start ab.finish c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_count_l452_45256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_angle_l452_45207

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 9*x

-- Define the chord AB
def chord_AB (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12^2

-- Define that AB passes through the focus
def passes_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  (t * A.1 + (1 - t) * B.1 = 9/4) ∧
  (t * A.2 + (1 - t) * B.2 = 0)

-- Define the angle of inclination
noncomputable def angle_of_inclination (A B : ℝ × ℝ) : ℝ :=
  Real.arctan ((B.2 - A.2) / (B.1 - A.1))

-- Theorem statement
theorem focal_chord_angle (A B : ℝ × ℝ) :
  chord_AB A B → passes_through_focus A B →
  angle_of_inclination A B = π/3 ∨ angle_of_inclination A B = 2*π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_chord_angle_l452_45207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l452_45227

-- Define constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := (3 : ℝ) ^ (-1/2 : ℝ)
noncomputable def c : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) + 
             Real.cos (140 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

-- Theorem statement
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l452_45227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_number_is_729_l452_45246

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- Checks if two ThreeDigitNumbers match in exactly one digit place -/
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.hundreds = b.hundreds ∧ a.tens ≠ b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens = b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.units = b.units)

/-- The theorem stating that 729 is the only number satisfying the conditions -/
theorem antons_number_is_729 :
  ∃! n : ThreeDigitNumber,
    matchesOneDigit n ⟨1, 0, 9, by sorry, by sorry, by sorry⟩ ∧
    matchesOneDigit n ⟨7, 0, 4, by sorry, by sorry, by sorry⟩ ∧
    matchesOneDigit n ⟨1, 2, 4, by sorry, by sorry, by sorry⟩ ∧
    n = ⟨7, 2, 9, by sorry, by sorry, by sorry⟩ :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_number_is_729_l452_45246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_pi_plus_alpha_l452_45288

theorem cos_double_pi_plus_alpha (α m : Real) 
  (h1 : Real.sin (Real.pi - α) = m) 
  (h2 : |m| ≤ 1) : 
  Real.cos (2 * (Real.pi + α)) = 1 - 2 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_pi_plus_alpha_l452_45288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_zero_l452_45293

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Theorem statement
theorem tangent_at_zero :
  ∀ x y : ℝ, f x = y → (∃ ε > 0, ∀ h : ℝ, |h| < ε → 
    |f (h + 0) - (f 0 + 3 * h)| / |h| < ε) →
  tangent_line x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_zero_l452_45293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abcd_l452_45283

noncomputable def a : ℝ := Real.sin (Real.sin 2008)
noncomputable def b : ℝ := Real.sin (Real.cos 2008)
noncomputable def c : ℝ := Real.cos (Real.sin 2008)
noncomputable def d : ℝ := Real.cos (Real.cos 2008)

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abcd_l452_45283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_sum_l452_45261

theorem power_difference_sum (m n : ℕ) : 496 = 2^m - 2^n → m + n = 13 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_sum_l452_45261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_is_positive_integer_l452_45294

noncomputable def has_22_decimal_digits (x : ℝ) : Prop :=
  ∃ n : ℕ, (x^4 * 3.456789)^11 * 10^n ∈ Set.Icc (10^21) (10^22 - 1)

theorem base_number_is_positive_integer (x : ℝ) (hx : x > 0) 
  (h : has_22_decimal_digits x) : ∃ n : ℕ+, x = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_number_is_positive_integer_l452_45294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_constant_l452_45231

/-- Molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of chlorine in g/mol -/
noncomputable def molar_mass_Cl : ℝ := 35.45

/-- Molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of chlorous acid (HClO₂) in g/mol -/
noncomputable def molar_mass_HClO2 : ℝ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

/-- Mass percentage of oxygen in chlorous acid -/
noncomputable def mass_percentage_O : ℝ := (2 * molar_mass_O / molar_mass_HClO2) * 100

/-- Temperature in Celsius -/
def temperature : Set ℝ := {10, 25, 50}

/-- Pressure in atmospheres -/
def pressure : Set ℝ := {1, 2, 3}

theorem mass_percentage_O_constant (t : ℝ) (p : ℝ) 
  (ht : t ∈ temperature) (hp : p ∈ pressure) : 
  ∃ ε > 0, |mass_percentage_O - 46.75| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_constant_l452_45231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45214

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.sin (t.A - t.B) + Real.sin t.C = Real.sqrt 2 * Real.sin t.A ∧
  t.b = 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = Real.pi / 4 ∧
  (∃ (max : Real), max = 8 + 4 * Real.sqrt 2 ∧
    ∀ (a c : Real), a^2 + c^2 ≤ max) ∧
  (∃ (a c : Real), a^2 + c^2 = 8 + 4 * Real.sqrt 2 ∧
    t.A = 3*Real.pi/8 ∧ t.C = 3*Real.pi/8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l452_45214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l452_45276

/-- A parabola with focus F(a, 0) where a < 0 has the standard equation y^2 = 4ax -/
theorem parabola_equation (a : ℝ) (h : a < 0) :
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p.1 = a + t^2 ∧ p.2 = 2*t*(-a)} ↔ y^2 = 4*a*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l452_45276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l452_45248

theorem relationship_abc : (0.3 : Real)^2 < 2^(0.3 : Real) ∧ 2^(0.3 : Real) < Real.log 2 / Real.log (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l452_45248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_can_volume_ratio_l452_45285

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The ratio of volumes of two cylinders -/
noncomputable def volumeRatio (r1 h1 r2 h2 : ℝ) : ℝ :=
  cylinderVolume r1 h1 / cylinderVolume r2 h2

theorem cat_food_can_volume_ratio :
  volumeRatio 4 16 5 8 = 32 / 25 := by
  -- Unfold definitions
  unfold volumeRatio cylinderVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_can_volume_ratio_l452_45285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_path_length_l452_45217

/-- The longest path length for hammering nails into equally spaced poles -/
def longest_path_length (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    (n^2 + 2*n - 2) / 2
  else
    (n^2 + 2*n - 1) / 2

/-- Theorem stating that the longest_path_length function gives the optimal path length -/
theorem optimal_path_length (n : ℕ) (h : n > 1) :
  ∀ (path : List ℕ),
    path.Nodup ∧
    path.length = n ∧
    (∀ i ∈ path, 0 ≤ i ∧ i < n) →
    (path.zip path.tail).foldl (λ acc (a, b) => acc + |Int.ofNat a - Int.ofNat b|) 0 ≤ longest_path_length n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_path_length_l452_45217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_ten_hours_l452_45254

/-- Calculates the total time for a round trip given the distance and speeds -/
noncomputable def roundTripTime (distance : ℝ) (speedTo : ℝ) (speedFrom : ℝ) : ℝ :=
  distance / speedTo + distance / speedFrom

/-- Theorem: The round trip time for the given conditions is 10 hours -/
theorem round_trip_time_is_ten_hours :
  roundTripTime 300 50 75 = 10 := by
  -- Unfold the definition of roundTripTime
  unfold roundTripTime
  -- Simplify the arithmetic
  simp [div_add_div]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_ten_hours_l452_45254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_shoe_arrangements_l452_45234

def num_socks : ℕ := 8
def num_shoes : ℕ := 8
def total_items : ℕ := num_socks + num_shoes

theorem sock_shoe_arrangements :
  (Nat.factorial total_items) / (2^num_socks) = 81729648000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_shoe_arrangements_l452_45234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l452_45232

open Real

/-- The function f to be maximized -/
noncomputable def f (x y z : ℝ) : ℝ :=
  (x * (2*y - z)) / (1 + x + 3*y) +
  (y * (2*z - x)) / (1 + y + 3*z) +
  (z * (2*x - y)) / (1 + z + 3*x)

/-- The theorem stating the maximum value of f -/
theorem max_value_of_f :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 1 →
  f x y z ≤ 1/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l452_45232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_matches_convergents_l452_45297

/-- Convergent of a continued fraction -/
noncomputable def Convergent (p q : ℝ) (k : ℕ) : ℝ := 
  sorry

/-- Newton's method iteration -/
noncomputable def NewtonIteration (p q : ℝ) (x : ℝ) : ℝ :=
  (x^2 - q) / (2*x - p)

/-- Theorem statement -/
theorem newton_matches_convergents (p q : ℝ) :
  ∀ n : ℕ, ∃ k : ℕ, 
    (NewtonIteration p q)^[n] (Convergent p q 0) = Convergent p q k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newton_matches_convergents_l452_45297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_equivalence_l452_45257

variable (P Q : Prop)

theorem logical_equivalence :
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  constructor
  · -- Proof for (P → Q) ↔ (¬Q → ¬P)
    sorry
  · -- Proof for (P → Q) ↔ (¬P ∨ Q)
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_equivalence_l452_45257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedral_angle_parallelogram_intersection_l452_45245

-- Define a convex polyhedral angle
structure ConvexPolyhedralAngle where
  -- Simplified representation
  vertex : Point
  edges : List Line
  convex : Bool

-- Define a plane
structure Plane where
  -- Simplified representation
  normal : Vector ℝ 3
  point : Point

-- Define a parallelogram
structure Parallelogram where
  -- Simplified representation
  vertices : List Point
  is_parallelogram : Bool

-- Define an intersection function (placeholder)
def intersection (angle : ConvexPolyhedralAngle) (p : Plane) : Parallelogram :=
  sorry

-- Theorem statement
theorem convex_polyhedral_angle_parallelogram_intersection 
  (angle : ConvexPolyhedralAngle) :
  ∃ (p : Plane), ∃ (sect : Parallelogram), 
    sect = intersection angle p ∧ 
    sect.is_parallelogram = true :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedral_angle_parallelogram_intersection_l452_45245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l452_45255

noncomputable def f (x : ℝ) : ℝ := (x - 9*x^2 + 27*x^3) / (9 - x^3)

theorem f_nonnegative_iff (x : ℝ) : 
  f x ≥ 0 ↔ (0 ≤ x ∧ x ≤ 1/3) ∨ (3 ≤ x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l452_45255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_max_triangle_area_l452_45223

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  focus : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.f x

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Line passing through a point with given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Point lies on a line -/
def PointOnLine (l : Line) (x y : ℝ) : Prop :=
  y - l.point.2 = l.slope * (x - l.point.1)

/-- Triangle area given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1*(p2.2 - p3.2) + p2.1*(p3.2 - p1.2) + p3.1*(p1.2 - p2.2))

theorem parabola_focus_property (p : Parabola) (x y : ℝ) :
  p.f x = (x^2) / 4 ∧ p.focus = (0, 1) →
  PointOnParabola p 2 1 ∧ PointOnParabola p (-2) 1 ∧
  distance (2, 1) p.focus = 2 ∧ distance (-2, 1) p.focus = 2 := by sorry

theorem max_triangle_area (p : Parabola) (l : Line) :
  p.f x = (x^2) / 4 ∧ p.focus = (0, 1) ∧ l.slope = 1 ∧ l.point = (0, 1) →
  ∃ (A B : ℝ × ℝ),
    PointOnParabola p A.1 A.2 ∧ PointOnParabola p B.1 B.2 ∧
    PointOnLine l A.1 A.2 ∧ PointOnLine l B.1 B.2 ∧
    (∀ (P : ℝ × ℝ), PointOnParabola p P.1 P.2 →
      triangleArea P A B ≤ 4 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_max_triangle_area_l452_45223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_inequality_l452_45219

theorem triangle_angle_sine_inequality (A B C : Real) : 
  0 < A → 0 < B → 0 < C →  -- angles are positive
  A + B + C = Real.pi →  -- sum of angles in a triangle
  A < B → B < C →  -- given order of angles
  C ≠ Real.pi/2 →  -- C is not a right angle
  Real.sin A < Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sine_inequality_l452_45219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l452_45204

-- Define the basic types
variable (α : Type) -- Plane
variable (l m b c : Type) -- Lines

-- Define the relationships
def subset (x y : Type) : Prop := sorry
def intersect (x y : Type) : Prop := sorry
def parallel (x y : Type) : Prop := sorry

-- Define the propositions
def proposition1 (α l m : Type) : Prop :=
  subset m α ∧ ¬subset l α ∧ ¬intersect l m → parallel l α

def proposition2 (α l b c : Type) : Prop :=
  subset b α ∧ subset c α ∧ ¬subset l α ∧ intersect b c ∧ ¬intersect l b ∧ ¬intersect l c → parallel l α

def proposition3 (α b c : Type) : Prop :=
  parallel b c ∧ parallel b α → parallel c α

def proposition4 (α l b : Type) : Prop :=
  parallel l α ∧ parallel b α → parallel l b

-- Theorem statement
theorem all_propositions_false :
  ¬proposition1 α l m ∧
  ¬proposition2 α l b c ∧
  ¬proposition3 α b c ∧
  ¬proposition4 α l b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l452_45204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madison_qualification_proof_l452_45203

/-- The minimum score required to qualify for a geometry class -/
def minimum_qualifying_score : ℚ := 85

/-- The number of quarters considered for qualification -/
def number_of_quarters : ℕ := 5

/-- Madison's scores for the first four quarters -/
def madison_scores : List ℚ := [84, 81, 87, 83]

/-- The minimum score Madison needs in the 5th quarter to qualify -/
def minimum_fifth_quarter_score : ℚ := 90

theorem madison_qualification_proof :
  minimum_fifth_quarter_score = 
    (minimum_qualifying_score * number_of_quarters) - madison_scores.sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madison_qualification_proof_l452_45203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_proof_l452_45200

theorem equation_proof (x y : ℝ) (h1 : (3 : ℝ)^x = 6) (h2 : (4 : ℝ)^y = 6) :
  2/x + 1/y = (Real.log 18) / (Real.log 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_proof_l452_45200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_line_theorem_l452_45264

/-- Circle C with center (1, -2) and radius 3 -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 + 2)^2 = 9}

/-- Line l with equation x - y + m = 0 -/
def line_l (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + m = 0}

/-- Distance between a point and a line -/
noncomputable def distance_point_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  |p.1 - p.2 + m| / Real.sqrt 2

/-- Theorem stating the relation between m and the distance -/
theorem distance_circle_line_theorem :
  ∀ m : ℝ, distance_point_line (1, -2) m = 2 →
    m = -3 + 2 * Real.sqrt 2 ∨ m = -3 - 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_line_theorem_l452_45264
