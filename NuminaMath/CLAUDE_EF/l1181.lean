import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_pie_degrees_l1181_118142

theorem lemon_pie_degrees (total_students : ℕ) (chocolate_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ) (raspberry_pie : ℕ) :
  total_students = 40 →
  chocolate_pie = 15 →
  apple_pie = 10 →
  blueberry_pie = 7 →
  raspberry_pie = 3 →
  (total_students - (chocolate_pie + apple_pie + blueberry_pie + raspberry_pie)) % 2 = 0 →
  (total_students - (chocolate_pie + apple_pie + blueberry_pie + raspberry_pie)) / 2 * 360 / total_students = 45/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_pie_degrees_l1181_118142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_a_l1181_118136

theorem max_prime_factors_a (a b : ℕ) 
  (h1 : (Nat.gcd a b).factors.length = 7)
  (h2 : (Nat.lcm a b).factors.length = 28)
  (h3 : (a.factors.length < b.factors.length)) :
  a.factors.length ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_a_l1181_118136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowans_rate_l1181_118175

/-- Represents the rate of a rower in still water -/
noncomputable def still_water_rate (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) : ℝ :=
  (3 * downstream_distance) / (downstream_time * upstream_time)

/-- Theorem stating that Rowan's rate in still water is 8⅔ km/h given the problem conditions -/
theorem rowans_rate :
  let downstream_distance : ℝ := 26
  let downstream_time : ℝ := 2
  let upstream_time : ℝ := 4
  still_water_rate downstream_distance downstream_time upstream_time = 26 / 3 := by
  sorry

#eval (26 : ℚ) / 3  -- To verify the result is indeed 8⅔

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowans_rate_l1181_118175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_congruent_triangle_l1181_118155

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : Real.sqrt 7 = dist A B)
  (BC_length : Real.sqrt 13 = dist B C)
  (CA_length : Real.sqrt 19 = dist C A)

-- Define the circles
noncomputable def circle_A (t : Triangle) := {p : ℝ × ℝ | dist p t.A = 1/3}
noncomputable def circle_B (t : Triangle) := {p : ℝ × ℝ | dist p t.B = 2/3}
noncomputable def circle_C (t : Triangle) := {p : ℝ × ℝ | dist p t.C = 1}

-- Define congruence between triangles
def congruent (t1 t2 : Triangle) : Prop :=
  dist t1.A t1.B = dist t2.A t2.B ∧
  dist t1.B t1.C = dist t2.B t2.C ∧
  dist t1.C t1.A = dist t2.C t2.A

-- Theorem statement
theorem existence_of_congruent_triangle (t : Triangle) :
  ∃ A' B' C',
    A' ∈ circle_A t ∧
    B' ∈ circle_B t ∧
    C' ∈ circle_C t ∧
    congruent t (Triangle.mk A' B' C' sorry sorry sorry) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_congruent_triangle_l1181_118155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1181_118111

theorem power_equation_solution (n b : ℝ) : n = 2^(15/100) → n^b = 64 → b = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1181_118111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1181_118127

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (x + 1)
noncomputable def g (a x : ℝ) : ℝ := a * x + 5 - 2 * a

-- State the theorem
theorem range_of_a :
  ∀ (a : ℝ), 
    (a > 0) →
    (∀ x₁ ∈ Set.Icc 0 1, ∃ x₀ ∈ Set.Icc 0 1, g a x₀ = f x₁) →
    (a ∈ Set.Icc (5/2) 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1181_118127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1181_118107

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define point M
def M : ℝ × ℝ := (4, -8)

-- Define the condition that M is outside the circle
def M_outside_circle : Prop := ¬ circle_eq M.1 M.2

-- Define the length of chord AB
def AB_length : ℝ := 4

-- Define the possible equations of line AB
def line_AB_eq1 (x y : ℝ) : Prop := 45*x + 28*y + 44 = 0
def line_AB_eq2 (x : ℝ) : Prop := x = 4

-- Theorem statement
theorem line_AB_equation :
  ∀ (A B : ℝ × ℝ),
  circle_eq A.1 A.2 →
  circle_eq B.1 B.2 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB_length^2 →
  (∃ t : ℝ, A = M + t • (B - M)) →
  (line_AB_eq1 A.1 A.2 ∧ line_AB_eq1 B.1 B.2) ∨ (line_AB_eq2 A.1 ∧ line_AB_eq2 B.1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1181_118107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_to_flour_ratio_l1181_118169

-- Define the variables
def sugar : ℝ := 3000
def flour : ℝ := sorry
def bakingSoda : ℝ := sorry

-- Define the conditions
axiom flour_to_bakingSoda : flour = 10 * bakingSoda
axiom flour_to_bakingSoda_plus60 : flour = 8 * (bakingSoda + 60)

-- Define the theorem
theorem sugar_to_flour_ratio : sugar / flour = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_to_flour_ratio_l1181_118169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_l1181_118160

theorem det_rotation_matrix (θ : ℝ) : 
  let R : Matrix (Fin 2) (Fin 2) ℝ := !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]
  Matrix.det R = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_l1181_118160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1181_118108

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem f_properties :
  let a := (2 : ℝ)
  let b := (4 : ℝ)
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (f a) (f b)) ∧
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x < f y) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ f a) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ f b) ∧
  f a = 2/3 ∧
  f b = 4/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1181_118108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1181_118129

def sequence_a : ℕ → ℕ
| 0 => 3
| n + 1 => sequence_a n + 3

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n = 3 * n.succ) ∧
  sequence_a 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1181_118129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maggies_purchase_cost_l1181_118154

theorem maggies_purchase_cost (plant_books fish_books magazines : ℕ)
  (plant_book_price fish_book_price magazine_price discount_rate : ℚ) :
  plant_books = 20 →
  fish_books = 7 →
  magazines = 25 →
  plant_book_price = 25 →
  fish_book_price = 30 →
  magazine_price = 5 →
  discount_rate = 1/10 →
  (plant_books * plant_book_price + fish_books * fish_book_price + magazines * magazine_price) * (1 - discount_rate) = 751.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maggies_purchase_cost_l1181_118154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_range_l1181_118186

noncomputable section

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*a*y = 0

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point T on the circle
def point_T_on_circle (a : ℝ) (t : ℝ × ℝ) : Prop :=
  circle_M a t.1 t.2

-- Define the angle MAT
noncomputable def angle_MAT (a : ℝ) (t : ℝ × ℝ) : ℝ :=
  Real.arctan ((t.2 - 2) / (t.1 - 0)) - Real.arctan ((a - 2) / a)

theorem circle_tangent_range (a : ℝ) :
  a > 0 →
  (∃ t : ℝ × ℝ, point_T_on_circle a t ∧ angle_MAT a t = π/4) →
  a ≥ Real.sqrt 3 - 1 ∧ a < 1 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_range_l1181_118186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totalSurfaceArea_formula_l1181_118104

/-- A pyramid with an isosceles trapezoid base -/
structure IsoscelesTrapezoidPyramid where
  a : ℝ -- lateral side of the trapezoid base
  α : ℝ -- acute angle of the trapezoid base
  β : ℝ -- angle between lateral faces and base

/-- The total surface area of the pyramid -/
noncomputable def totalSurfaceArea (p : IsoscelesTrapezoidPyramid) : ℝ :=
  (2 * p.a^2 * Real.sin p.α * (Real.cos (p.β / 2))^2) / Real.cos p.β

theorem totalSurfaceArea_formula (p : IsoscelesTrapezoidPyramid) :
  totalSurfaceArea p = (2 * p.a^2 * Real.sin p.α * (Real.cos (p.β / 2))^2) / Real.cos p.β :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_totalSurfaceArea_formula_l1181_118104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1181_118103

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b
  h' : b > 0
  eccentricity : ℝ
  h_ecc : eccentricity = Real.sqrt 2 / 2
  max_area : ℝ
  h_area : max_area = 4

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  k : ℝ
  t : ℝ
  h : t ≠ 0

/-- The theorem stating the properties of the ellipse and the fixed point -/
theorem ellipse_and_fixed_point (e : Ellipse) (l : IntersectingLine) :
  (∀ x y : ℝ, x^2 / 8 + y^2 / 4 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
  (l.t = 4/3 → ∃ Q : ℝ × ℝ, Q.1 = 0 ∧ Q.2 = 13/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1181_118103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l1181_118184

/-- Represents the number of sides in a convex polygon --/
def n : ℕ := 9

/-- The largest interior angle of the polygon in degrees --/
noncomputable def largest_angle : ℝ := 160

/-- The common difference of the arithmetic progression of interior angles in degrees --/
noncomputable def common_difference : ℝ := 5

/-- The sum of interior angles of an n-sided polygon --/
noncomputable def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The sum of the arithmetic sequence of interior angles --/
noncomputable def angle_sequence_sum (n : ℕ) (largest : ℝ) (diff : ℝ) : ℝ :=
  n * largest - diff * (n * (n - 1) / 2)

theorem polygon_sides_count :
  angle_sequence_sum n largest_angle common_difference = interior_angle_sum n ∧ n = 9 := by
  sorry

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l1181_118184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l1181_118134

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1/8)

-- Define the directrix of the parabola
def directrix (y : ℝ) : Prop := y = -1/8

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_point_ordinate (M : ℝ × ℝ) :
  parabola M.1 M.2 →
  distance M focus = 1 →
  M.2 = 7/8 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_ordinate_l1181_118134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_even_l1181_118145

theorem exactly_one_even (a b c : ℕ) : 
  ¬(a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∧ 
  ¬(∃ (x y : ℕ), x ≠ y ∧ ({x, y} : Set ℕ) ⊆ {a, b, c} ∧ x % 2 = 0 ∧ y % 2 = 0) →
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧
  ¬(∃ (x y : ℕ), x ≠ y ∧ ({x, y} : Set ℕ) ⊆ {a, b, c} ∧ x % 2 = 0 ∧ y % 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_even_l1181_118145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_any_sequence_reducible_to_one_l1181_118173

/-- Represents the operation of replacing two adjacent numbers with their sum according to the rule -/
def replace_operation (a b : ℕ) : ℕ := a + 2 * b

/-- Predicate to check if a sequence can be reduced to 1 using the replace operation -/
def can_reduce_to_one (seq : List ℕ) : Prop :=
  ∃ (steps : ℕ), ∃ (final_seq : List ℕ),
    (final_seq.length = 1) ∧ 
    (final_seq.head? = some 1) ∧
    (∃ (intermediate_seqs : List (List ℕ)), 
      intermediate_seqs.length = steps ∧
      intermediate_seqs.head? = some seq ∧
      intermediate_seqs.getLast? = some final_seq ∧
      ∀ i < steps, 
        ∃ (a b : ℕ), ∃ (rest : List ℕ),
          (intermediate_seqs.get! i = a :: b :: rest) ∧
          (intermediate_seqs.get! (i+1) = replace_operation a b :: rest))

/-- Theorem stating that any finite sequence of natural numbers can be reduced to 1 -/
theorem any_sequence_reducible_to_one (seq : List ℕ) : can_reduce_to_one seq := by
  sorry

#check any_sequence_reducible_to_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_any_sequence_reducible_to_one_l1181_118173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_cost_difference_l1181_118196

def grant_base_cost : ℝ := 200
def grant_discount_rate : ℝ := 0.1
def juanita_mon_wed_cost : ℝ := 0.5
def juanita_thu_sat_cost : ℝ := 0.75
def juanita_sun_cost : ℝ := 2.5
def juanita_sun_coupon : ℝ := 0.25
def weeks_per_year : ℝ := 52
def months_per_year : ℝ := 12

def grant_annual_cost : ℝ := grant_base_cost * (1 - grant_discount_rate)

def juanita_annual_cost : ℝ :=
  (juanita_mon_wed_cost * 3 + juanita_thu_sat_cost * 3 + juanita_sun_cost) * weeks_per_year
  - juanita_sun_coupon * months_per_year

theorem newspaper_cost_difference :
  juanita_annual_cost - grant_annual_cost = 142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_cost_difference_l1181_118196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1181_118165

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - 1

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := -1 / Real.exp x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (Real.exp x₀ * x + y + 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1181_118165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l1181_118197

/-- The percentage of fiber absorbed by koalas -/
noncomputable def fiber_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
noncomputable def fiber_absorbed : ℝ := 12

/-- The total amount of fiber eaten by the koala in one day (in ounces) -/
noncomputable def total_fiber_eaten : ℝ := fiber_absorbed / fiber_absorption_rate

theorem koala_fiber_consumption : total_fiber_eaten = 40 := by
  -- Unfold the definitions
  unfold total_fiber_eaten fiber_absorbed fiber_absorption_rate
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l1181_118197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_dot_product_bounds_and_constant_distance_l1181_118112

/-- The trajectory of point M(x, y) given the conditions -/
noncomputable def trajectory (x y : ℝ) : Prop :=
  x^2/4 + y^2/3 = 1

/-- The dot product of vectors PF₁ and PF₂ -/
noncomputable def dot_product (x y : ℝ) : ℝ :=
  x^2/4 + 2

/-- The distance from origin O to a line intersecting the trajectory -/
noncomputable def distance_to_line : ℝ :=
  2 * Real.sqrt 21 / 7

theorem trajectory_and_dot_product_bounds_and_constant_distance 
  (x y : ℝ) (a b : ℝ × ℝ) :
  a = (x - 1, y) → b = (x + 1, y) → Real.sqrt ((x - 1)^2 + y^2) + Real.sqrt ((x + 1)^2 + y^2) = 4 →
  (trajectory x y ∧
   (∀ x₀ y₀, trajectory x₀ y₀ → 2 ≤ dot_product x₀ y₀ ∧ dot_product x₀ y₀ ≤ 3) ∧
   (∀ l : Set (ℝ × ℝ), (∃ A B : ℝ × ℝ, A ∈ l ∧ B ∈ l ∧ trajectory A.1 A.2 ∧ trajectory B.1 B.2) →
     ∃ d : ℝ, d = distance_to_line)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_dot_product_bounds_and_constant_distance_l1181_118112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1181_118198

-- Define the given values
noncomputable def amount : ℝ := 1120
noncomputable def interestRate : ℝ := 0.06
noncomputable def time : ℝ := 2.4

-- Define the function to calculate the principal
noncomputable def calculatePrincipal (a r t : ℝ) : ℝ :=
  a / (1 + r * t)

-- State the theorem
theorem principal_calculation :
  abs (calculatePrincipal amount interestRate time - 979.02) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1181_118198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_ratio_l1181_118168

/-- An equilateral triangle with specific points on its sides -/
structure TriangleWithPoints where
  -- The equilateral triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Point D on AB
  D : ℝ × ℝ
  -- Point E on BC
  E : ℝ × ℝ
  -- Point F on AC
  F : ℝ × ℝ
  -- Conditions
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  D_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  E_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  F_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)
  D_ratio : (A.1 - D.1)^2 + (A.2 - D.2)^2 = 1/16 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  E_ratio : (B.1 - E.1)^2 + (B.2 - E.2)^2 = 4/9 * ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  no_rotation : ∃ O : ℝ × ℝ, (O.1 - D.1) * (A.2 - B.2) = (O.2 - D.2) * (A.1 - B.1) ∧
                              (O.1 - E.1) * (B.2 - C.2) = (O.2 - E.2) * (B.1 - C.1) ∧
                              (O.1 - F.1) * (C.2 - A.2) = (O.2 - F.2) * (C.1 - A.1)

/-- The third point F divides AC in the ratio 5:7 from A -/
theorem third_point_ratio (t : TriangleWithPoints) : 
  (t.A.1 - t.F.1)^2 + (t.A.2 - t.F.2)^2 = 25/144 * ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_ratio_l1181_118168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1181_118153

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def lateralSurfaceArea (r₁ r₂ h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (r₂ - r₁)^2)
  Real.pi * (r₁ + r₂) * s

/-- Theorem: The lateral surface area of a frustum of a right circular cone
    with an upper base radius of 4 inches, a lower base radius of 7 inches,
    and a vertical height of 6 inches is equal to 33π√5 square inches. -/
theorem frustum_lateral_surface_area :
  lateralSurfaceArea 4 7 6 = 33 * Real.pi * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1181_118153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_sum_l1181_118176

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧ digits.Nodup

def sum_of_three_digit_combinations (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (List.sum (List.map (λ x => 1332 * x) digits))

theorem unique_five_digit_sum : ∃! n : ℕ,
  n ≥ 10000 ∧ n < 100000 ∧
  all_digits_different n ∧
  (∀ d, d ∈ n.digits 10 → is_valid_digit d) ∧
  n = sum_of_three_digit_combinations n ∧
  n = 35964 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_sum_l1181_118176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_l1181_118106

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Tetrahedron with vertices A, B, C, and S -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  S : Point3D

/-- Function to get the symmetric point with respect to a line -/
noncomputable def symmetricPoint (p : Point3D) (line : Point3D → Point3D → Point3D) : Point3D :=
  sorry

/-- Function to get the perpendicular bisector of two points -/
noncomputable def perpendicularBisector (p1 p2 : Point3D) : Point3D → Point3D → Point3D :=
  sorry

/-- Function to check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Main theorem -/
theorem tetrahedron_intersection (t : Tetrahedron) 
  (h_no_equal_edges : ∀ e1 e2 : Point3D × Point3D, e1 ≠ e2 → distance e1.1 e1.2 ≠ distance e2.1 e2.2) :
  ∃ (p : Point3D),
    let A' := symmetricPoint t.S (perpendicularBisector t.B t.C)
    let B' := symmetricPoint t.S (perpendicularBisector t.C t.A)
    let C' := symmetricPoint t.S (perpendicularBisector t.A t.B)
    let planeABC := Plane3D.mk 0 0 0 0  -- Placeholder, actual plane equation needed
    let planeAB'C' := Plane3D.mk 0 0 0 0  -- Placeholder, actual plane equation needed
    let planeA'BC' := Plane3D.mk 0 0 0 0  -- Placeholder, actual plane equation needed
    let planeA'B'C := Plane3D.mk 0 0 0 0  -- Placeholder, actual plane equation needed
    pointOnPlane p planeABC ∧
    pointOnPlane p planeAB'C' ∧
    pointOnPlane p planeA'BC' ∧
    pointOnPlane p planeA'B'C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_l1181_118106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l1181_118191

noncomputable def v (x : ℝ) : ℝ := 1 / (x^2 + Real.sqrt x)

theorem v_domain : Set.Ioi 0 = {x : ℝ | v x ≠ 0 ∧ x^2 + Real.sqrt x ≠ 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_domain_l1181_118191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_beta_pi_fourth_l1181_118130

theorem sin_alpha_beta_pi_fourth (α β : ℝ) 
  (h1 : Real.sin (π/3 + α/6) = -3/5)
  (h2 : Real.cos (π/12 - β/2) = -12/13)
  (h3 : -5*π < α ∧ α < -2*π)
  (h4 : -11*π/6 < β ∧ β < π/6) :
  Real.sin (α/6 + β/2 + π/4) = 16/65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_beta_pi_fourth_l1181_118130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_decreasing_function_condition_l1181_118170

-- Define the domain
def D : Set ℝ := Set.Icc (-2) 2

-- Define the properties
def isOddOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S → f (-x) = -f x

def isDecreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x < y → f x > f y

def isMonotonicOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  (∀ x y, x ∈ S → y ∈ S → x < y → f x < f y) ∨ 
  (∀ x y, x ∈ S → y ∈ S → x < y → f x > f y)

-- State the theorems
theorem odd_function_condition (f : ℝ → ℝ) :
  (∀ x, x ∈ D → f (-x) + f x = 0) → isOddOn f D := by
  sorry

theorem decreasing_function_condition (f : ℝ → ℝ) :
  isMonotonicOn f D → f 0 > f 1 → isDecreasingOn f D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_decreasing_function_condition_l1181_118170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_los_angeles_distance_l1181_118140

/-- The distance between two points on a complex plane -/
noncomputable def complexDistance (z₁ z₂ : ℂ) : ℝ :=
  Complex.abs (z₁ - z₂)

/-- Theorem: The distance between 0 and 900 + 1200i on the complex plane is 1500 -/
theorem miami_los_angeles_distance :
  complexDistance 0 (900 + 1200 * Complex.I) = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miami_los_angeles_distance_l1181_118140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1181_118159

noncomputable def f (x : ℝ) := Real.log ((2 - x) / (2 + x))

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1181_118159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cucumber_equivalence_l1181_118114

-- Define the types for our fruits
structure Fruit where
  count : ℕ

def Apple := Fruit
def Banana := Fruit
def Cucumber := Fruit

-- Define the cost relationship between fruits
def cost_equivalent (a b c : Fruit) : Prop :=
  ∃ (price : ℚ), (a.count : ℚ) * price = (b.count : ℚ) * price ∧ (b.count : ℚ) * price = (c.count : ℚ) * price

-- State the theorem
theorem apple_cucumber_equivalence :
  cost_equivalent ⟨12⟩ ⟨6⟩ ⟨0⟩ →  -- 12 apples cost the same as 6 bananas
  cost_equivalent ⟨0⟩ ⟨3⟩ ⟨4⟩ →   -- 3 bananas cost the same as 4 cucumbers
  cost_equivalent ⟨24⟩ ⟨0⟩ ⟨16⟩   -- 24 apples cost the same as 16 cucumbers
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cucumber_equivalence_l1181_118114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1181_118147

theorem trig_identity (α : ℝ) : 
  1 - (1/4) * (Real.sin (2*α))^2 + Real.cos (2*α) = Real.cos α^2 + Real.cos α^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1181_118147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l1181_118131

/-- Represents an ellipse with center O, major axis AB, minor axis CD, and focus F. -/
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  minorAxis : ℝ
  focus : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The diameter of the inscribed circle of a right triangle with legs a and b -/
noncomputable def inscribedCircleDiameter (a b : ℝ) : ℝ :=
  2 * (a + b - Real.sqrt (a^2 + b^2)) / 2

theorem ellipse_product (e : Ellipse) 
    (h1 : distance e.center e.focus = 8)
    (h2 : inscribedCircleDiameter (e.minorAxis/2) (distance e.center e.focus) = 4) :
  e.majorAxis * e.minorAxis = 240 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l1181_118131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_probability_l1181_118109

/-- The probability of arranging 4 specific items consecutively in a random permutation of n items -/
def consecutive_probability (n : ℕ) (k : ℕ) : ℚ :=
  if n < k then 0
  else (n - k + 1 : ℚ) * (Nat.factorial (n - k)) / (Nat.factorial n)

/-- The probability of arranging 4 Escher prints consecutively in a random arrangement of 12 art pieces -/
theorem escher_prints_probability :
  consecutive_probability 12 4 = 1 / 1320 := by
  sorry

#eval consecutive_probability 12 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escher_prints_probability_l1181_118109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_circle_with_diameter_AB_l1181_118181

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 + y^2/2 = 1

-- Define the foci of the ellipse
noncomputable def F₁ : ℝ × ℝ := (0, -1)
noncomputable def F₂ : ℝ × ℝ := (0, 1)

-- Define the point P that the ellipse passes through
noncomputable def P : ℝ × ℝ := (Real.sqrt 2 / 2, 1)

-- Define the point S through which line l passes
noncomputable def S : ℝ × ℝ := (-1/3, 0)

-- Define the fixed point T
noncomputable def T : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem fixed_point_on_circle_with_diameter_AB :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  ∃ (k : ℝ), A.2 - S.2 = k * (A.1 - S.1) ∧ B.2 - S.2 = k * (B.1 - S.1) →
  (T.1 - A.1) * (T.1 - B.1) + (T.2 - A.2) * (T.2 - B.2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_circle_with_diameter_AB_l1181_118181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_three_implications_l1181_118121

theorem tan_negative_three_implications (θ : Real) (h : Real.tan θ = -3) :
  (Real.sin θ + 2 * Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -1/10 ∧
  Real.sin θ ^ 2 - Real.sin θ * Real.cos θ = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_three_implications_l1181_118121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_l1181_118190

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else x^2 + 3 * x - 18

-- Theorem statement
theorem solutions_of_g (x : ℝ) : g x = 2 ↔ x = -8/5 ∨ x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_l1181_118190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solution_l1181_118124

open Real

-- Define the problem parameters
def isIsoscelesTriangle (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), a = Real.cos x ∧ b = Real.cos x ∧ c = Real.cos (7 * x) ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the vertex angle condition
def hasVertexAngle2x (x : ℝ) : Prop :=
  ∃ (angle : ℝ), angle = 2 * x ∧ 0 < angle ∧ angle < π

-- Define the acute angle condition
def isAcute (x : ℝ) : Prop := 0 < x ∧ x < π/2

-- Main theorem
theorem isosceles_triangle_solution :
  ∀ x : ℝ, isIsoscelesTriangle x ∧ hasVertexAngle2x x ∧ isAcute x →
  x = π/18 ∨ x = 5*π/18 ∨ x = π/3.33333 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_solution_l1181_118124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_exists_tangent_line_l1181_118141

-- Define the circle M
def circleM (q : ℝ) (x y : ℝ) : Prop :=
  (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1

-- Define the line l
def lineL (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

-- Statement B
theorem line_intersects_circle :
  ∀ (k q : ℝ), ∃ (x y : ℝ), circleM q x y ∧ lineL k x y :=
by
  sorry

-- Statement D
theorem exists_tangent_line :
  ∀ (k : ℝ), ∃ (q : ℝ), ∃! (x y : ℝ), circleM q x y ∧ lineL k x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_exists_tangent_line_l1181_118141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1181_118113

open Real

noncomputable def f (x : ℝ) : ℝ := log x

theorem problem_statement (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) :
  let x₀ := (x₁ + x₂) / 2
  let kAB := (f x₂ - f x₁) / (x₂ - x₁)
  ∀ a : ℝ,
    (∀ x > -1, MonotoneOn (λ x ↦ f (x + 1) - a * x) (Set.Ioi (-1)) ∨
      ∃ c > -1, MonotoneOn (λ x ↦ f (x + 1) - a * x) (Set.Ioc (-1) c) ∧
                AntitoneOn (λ x ↦ f (x + 1) - a * x) (Set.Ioi c)) ∧
    kAB > (deriv f x₀) ∧
    ∀ x > 1, exp x / (x + 1) > (x - 1) / log x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1181_118113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1181_118125

-- Define the set of positive integers
def PositiveInt := { n : ℕ // n > 0 }

-- Define the function type
def FunctionType := PositiveInt → ℤ

-- Define the divisibility relation
def divides (a b : PositiveInt) : Prop := ∃ k : ℕ, b.val = a.val * k

-- Define the prime factorization type
def PrimeFactorization := List (Nat × Nat)

-- Define a function to get prime factors of the form 4k+3
noncomputable def primesOf4kPlus3Form : PrimeFactorization → List PositiveInt := sorry

-- Define the conditions for the function
def satisfiesConditions (f : FunctionType) : Prop :=
  (∀ a b : PositiveInt, divides a b → f a ≥ f b) ∧
  (∀ a b : PositiveInt, f ⟨a.val * b.val, sorry⟩ + f ⟨a.val^2 + b.val^2, sorry⟩ = f a + f b)

-- Define the structure of the function as described in the solution
def solutionForm (f : FunctionType) : Prop :=
  ∃ (fPrime : PositiveInt → ℤ) (factors : PrimeFactorization),
    (∀ n : PositiveInt,
      f n = (primesOf4kPlus3Form factors).foldl
        (λ acc q => acc + fPrime q) 0 -
        ((primesOf4kPlus3Form factors).length - 1) * f ⟨1, sorry⟩) ∧
    (∀ q : PositiveInt, q ∈ primesOf4kPlus3Form factors → fPrime q ≤ f ⟨1, sorry⟩)

-- State the theorem
theorem function_characterization (f : FunctionType) :
  satisfiesConditions f → solutionForm f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l1181_118125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_increasing_l1181_118172

-- Define the function f(x) = 2/x
noncomputable def f (x : ℝ) : ℝ := 2 / x

-- State the theorem
theorem f_not_increasing :
  ¬(∀ x y : ℝ, 0 < x → 0 < y → x < y → f x ≤ f y) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_increasing_l1181_118172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equality_l1181_118149

theorem smallest_angle_trigonometric_equality :
  let y : Real := Real.pi / 14
  ∀ θ : Real, θ > 0 → (Real.sin (3 * θ) * Real.sin (4 * θ) = Real.cos (3 * θ) * Real.cos (4 * θ)) → θ ≥ y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_trigonometric_equality_l1181_118149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_squares_even_cubes_difference_l1181_118157

theorem odd_squares_even_cubes_difference : ∃ (diff : ℤ), diff = -799700002 := by
  let n : ℕ := 1001
  let sum_odd_squares : ℕ := n * (2 * n - 1) * (2 * n + 1) / 3
  let sum_even_cubes : ℕ := n^2 * (n + 1)^2
  let diff : ℤ := (sum_even_cubes : ℤ) - (sum_odd_squares : ℤ)
  use diff
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_squares_even_cubes_difference_l1181_118157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1181_118135

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Define acute triangle
def AcuteTriangle (a b c : ℝ) : Prop :=
  Triangle a b c ∧ a^2 < b^2 + c^2 ∧ b^2 < a^2 + c^2 ∧ c^2 < a^2 + b^2

-- Define the main theorem
theorem triangle_cosine_theorem 
  (a b c : ℝ) 
  (h_acute : AcuteTriangle a b c) 
  (h_ratio : b * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) / 
             (c * Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))) = 
             (1 + Real.cos (2 * Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) / 
             (1 + Real.cos (2 * Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))))) :
  (Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) = 1/3 → 
   Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))) = Real.sqrt 3 / 3) ∧
  (b = 5 → Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) = 3/4 → 
   a = 5 * Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1181_118135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_103_97_l1181_118116

theorem abs_diff_squares_103_97 : |((103 : ℤ)^2 - 97^2)| = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_squares_103_97_l1181_118116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l1181_118101

theorem cos_alpha_second_quadrant (α : Real) 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h2 : Real.sin α = 5 / 13) : Real.cos α = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l1181_118101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equation_l1181_118171

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := Real.cos θ ^ 2
noncomputable def y (θ : ℝ) : ℝ := 2 * Real.sin θ ^ 2

-- State the theorem
theorem parametric_to_ordinary_equation :
  ∀ θ : ℝ, 2 * x θ + y θ - 2 = 0 ∧ x θ ∈ Set.Icc 0 1 := by
  intro θ
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_equation_l1181_118171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_type_percentage_bounds_l1181_118123

/-- Represents the stock market scenario with three types of stocks -/
structure StockMarket where
  k : ℝ  -- number of shares of first type
  m : ℝ  -- number of shares of second type
  n : ℝ  -- number of shares of third type
  x : ℝ  -- price per share of first type
  y : ℝ  -- price per share of second type
  z : ℝ  -- price per share of third type

/-- Conditions for the stock market scenario -/
def validStockMarket (s : StockMarket) : Prop :=
  s.k > 0 ∧ s.m > 0 ∧ s.n > 0 ∧
  s.k + s.m = s.n ∧
  s.k * s.x + s.m * s.y = s.n * s.z ∧
  s.m * s.y = 4 * s.k * s.x ∧
  1.6 ≤ s.y - s.x ∧ s.y - s.x ≤ 2 ∧
  4.2 ≤ s.z ∧ s.z ≤ 6

/-- Percentage of total shares for the first type of stock -/
noncomputable def firstTypePercentage (s : StockMarket) : ℝ :=
  (s.k / (s.k + s.m + s.n)) * 100

/-- Theorem stating the bounds on the percentage of first type stock -/
theorem first_type_percentage_bounds (s : StockMarket) 
  (h : validStockMarket s) : 
  12.5 ≤ firstTypePercentage s ∧ firstTypePercentage s ≤ 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_type_percentage_bounds_l1181_118123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1181_118167

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 3) + 2

-- State the theorem
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-3) = 3 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l1181_118167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l1181_118187

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 1) / (x + a)

-- State the theorem
theorem odd_function_and_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 0 ∧ ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_monotonicity_l1181_118187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1181_118185

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define z
noncomputable def z : ℂ := 1 / (1 + i) + i

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1181_118185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_permutations_l1181_118156

theorem eight_digit_permutations : ∃ n : ℕ, n = 2520 := by
  -- Define the set of digits
  let digits : Multiset ℕ := {2, 2, 5, 5, 7, 7, 9, 9}

  -- Define the number of positions
  let positions : ℕ := 8

  -- Define the number of permutations
  let permutations : ℕ := Nat.factorial positions / (Nat.factorial 2)^4

  -- Assert and prove that the number of permutations is 2520
  have h : permutations = 2520 := by
    -- Proof steps would go here
    sorry

  -- Prove the theorem by showing the existence of such a natural number
  exact ⟨permutations, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_permutations_l1181_118156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_proof_l1181_118164

/-- Family of curves parameterized by θ -/
def family_of_curves (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

/-- The line y = 2x -/
def line (x y : ℝ) : Prop := y = 2 * x

/-- The maximum chord length -/
noncomputable def max_chord_length : ℝ := 8 * Real.sqrt 5

theorem max_chord_length_proof :
  ∃ (θ : ℝ) (x y : ℝ),
    family_of_curves θ x y ∧ 
    line x y ∧
    ∀ (θ' : ℝ) (x' y' : ℝ),
      family_of_curves θ' x' y' → 
      line x' y' → 
      (x' - 0)^2 + (y' - 0)^2 ≤ (x - 0)^2 + (y - 0)^2 ∧
      (x - 0)^2 + (y - 0)^2 = max_chord_length^2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_proof_l1181_118164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1181_118120

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - 5*x₁ - 3 = 0 ∧ x₂^2 - 5*x₂ - 3 = 0) :
  (x₁^2 + x₂^2 = 31) ∧ 
  ((1/x₁ - 1/x₂ = Real.sqrt 37 / (-3)) ∨ 
   (1/x₁ - 1/x₂ = -Real.sqrt 37 / (-3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_properties_l1181_118120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_l1181_118178

theorem max_inequality (a b c : ℂ) (m n : ℝ) 
  (h1 : Complex.abs (a + b) = m)
  (h2 : Complex.abs (a - b) = n)
  (h3 : m * n ≠ 0) :
  max (Complex.abs (a * c + b)) (Complex.abs (a + b * c)) ≥ m * n / Real.sqrt (m^2 + n^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_l1181_118178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_l1181_118158

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4 * x + 1 / x + a * Real.log x

-- State the theorem
theorem f_greater_than_four (x : ℝ) (a : ℝ) (hx : x > 0) (ha : -3 < a ∧ a < 0) :
  f x a > 4 := by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_four_l1181_118158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probable_deviation_l1181_118138

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (a σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - a)^2 / (2 * σ^2))

/-- The property that Φ(0.675) = 0.75 -/
axiom Φ_0675 : Φ 0.675 = 0.75

/-- Theorem: For a normally distributed random variable X with mean a and 
    standard deviation σ, the value E that satisfies P(|X-a| < E) = 50% 
    is equal to 0.675σ -/
theorem probable_deviation (a σ : ℝ) (σ_pos : σ > 0) :
  ∃ E : ℝ, E = 0.675 * σ ∧ 
  (∫ (x : ℝ) in Set.Ioo (a - E) (a + E), normal_pdf a σ x) = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probable_deviation_l1181_118138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_movie_count_l1181_118146

/-- Calculates the number of movies Emily watched during her flight --/
theorem emily_movie_count : ℕ := by
  -- Flight duration in minutes
  let flight_duration : ℕ := 10 * 60
  -- TV episode duration in minutes
  let tv_episode_duration : ℕ := 25
  -- Number of TV episodes watched
  let tv_episodes_watched : ℕ := 3
  -- Sleep duration in minutes
  let sleep_duration : ℕ := (9 * 60) / 2
  -- Movie duration in minutes
  let movie_duration : ℕ := 60 + 45
  -- Remaining time after all activities in minutes
  let remaining_time : ℕ := 45

  -- Calculate the number of movies watched
  let num_movies : ℕ := 
    (flight_duration - 
     (tv_episode_duration * tv_episodes_watched + sleep_duration + remaining_time)) / 
    movie_duration

  -- Prove that the number of movies watched is 2
  have h : num_movies = 2 := by sorry
  exact 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_movie_count_l1181_118146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_decreasing_function_l1181_118194

def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem solution_set_of_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_f_0 : f 0 = 1)
  (h_f_1 : f 1 = 0) :
  {x : ℝ | f x > 0} = Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_decreasing_function_l1181_118194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_010101_l1181_118199

def sequenceA : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (sequenceA n + sequenceA (n + 1) + sequenceA (n + 2) + 
              sequenceA (n + 3) + sequenceA (n + 4) + sequenceA (n + 5)) % 10

theorem no_consecutive_010101 : 
  ¬ ∃ j : ℕ, sequenceA j = 0 ∧ sequenceA (j + 1) = 1 ∧ 
           sequenceA (j + 2) = 0 ∧ sequenceA (j + 3) = 1 ∧ 
           sequenceA (j + 4) = 0 ∧ sequenceA (j + 5) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_010101_l1181_118199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l1181_118189

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2:ℝ)^a * (2:ℝ)^b = 4^2 → 
  (∀ x y : ℝ, x > 0 → y > 0 → (2:ℝ)^x * (2:ℝ)^y = 4^2 → 1/a + 1/b ≤ 1/x + 1/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2:ℝ)^x * (2:ℝ)^y = 4^2 ∧ 1/x + 1/y = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l1181_118189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_line_angle_theorem_l1181_118143

/-- A line in a plane -/
structure Line where
  -- We don't need to define the internal structure of a line for this problem

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  sorry -- We don't need to implement this for the statement

/-- A configuration of 7 non-parallel lines in a plane -/
structure SevenLineConfig where
  lines : Fin 7 → Line
  not_parallel : ∀ i j, i ≠ j → ¬ (lines i).isParallelTo (lines j)

/-- The angle between two lines -/
noncomputable def angleBetween (l1 l2 : Line) : ℝ :=
  sorry -- We don't need to implement this for the statement

/-- Main theorem: In any configuration of 7 non-parallel lines, 
    there exists a pair of lines with an angle less than 26° between them -/
theorem seven_line_angle_theorem (config : SevenLineConfig) : 
  ∃ i j, i < j ∧ angleBetween (config.lines i) (config.lines j) < 26 * π / 180 :=
by
  sorry -- We don't need to provide the proof for this statement


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_line_angle_theorem_l1181_118143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_properties_l1181_118195

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (s1 s2 angle : ℝ) : ℝ :=
  (1/2) * s1 * s2 * Real.sin angle

theorem triangle_side_and_area_properties (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.C = π/4)
  (h3 : t.c = Real.sqrt 2)
  (h4 : t.b + t.c = 2 * Real.sqrt 2) : 
  (t.b = Real.sqrt 2) ∧ 
  (∀ (s : Triangle), s.a = 2 ∧ s.b + s.c = 2 * Real.sqrt 2 → 
    triangleArea s.b s.c s.A ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_properties_l1181_118195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_vendor_problem_l1181_118128

/-- Ice cream vendor problem -/
theorem ice_cream_vendor_problem 
  (chocolate_initial : ℕ) 
  (mango_initial : ℕ) 
  (chocolate_sold_fraction : ℚ) 
  (mango_sold_fraction : ℚ) 
  (total_unsold : ℕ) :
  chocolate_initial = 50 →
  chocolate_sold_fraction = 3/5 →
  mango_sold_fraction = 2/3 →
  total_unsold = 38 →
  chocolate_initial + mango_initial - 
    (Nat.floor (↑chocolate_initial * chocolate_sold_fraction)) - 
    (Nat.floor (↑mango_initial * mango_sold_fraction)) = total_unsold →
  mango_initial = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_vendor_problem_l1181_118128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_divides_l1181_118188

noncomputable def y (n : ℕ) : ℝ := 
  ((2 + Real.sqrt 3) ^ (n + 1) - (2 - Real.sqrt 3) ^ (n + 1)) / (2 * Real.sqrt 3)

theorem y_divides (m r : ℕ) (hm : m ≥ 2) (hr : r ≥ 1) : 
  ∃ k : ℤ, y (r * m - 1) = k * y (m - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_divides_l1181_118188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_same_domain_range_l1181_118174

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.log x)

-- State the theorem
theorem f_g_same_domain_range : 
  (∀ x : ℝ, x > 0 → f x ∈ Set.Ioi 0) ∧ 
  (∀ x : ℝ, x > 0 → g x ∈ Set.Ioi 0) ∧
  (∀ y : ℝ, y > 0 → ∃ x > 0, f x = y) ∧
  (∀ y : ℝ, y > 0 → ∃ x > 0, g x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_same_domain_range_l1181_118174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_configuration_l1181_118182

-- Define a square on a plane
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  color : Bool  -- True for blue, False for red

-- Define intersection between two squares
def intersect (s1 s2 : Square) : Prop :=
  sorry  -- Definition of intersection

-- The main theorem
theorem square_intersection_configuration :
  ∃ (squares : List Square),
    squares.length = 6 ∧
    (∃ blue_squares red_squares : List Square,
      squares = blue_squares ++ red_squares ∧
      blue_squares.length = 3 ∧
      red_squares.length = 3 ∧
      (∀ b ∈ blue_squares, ∀ r ∈ red_squares, intersect b r) ∧
      (∀ b1 b2 : Square, b1 ∈ blue_squares → b2 ∈ blue_squares → b1 ≠ b2 → ¬ intersect b1 b2) ∧
      (∀ r1 r2 : Square, r1 ∈ red_squares → r2 ∈ red_squares → r1 ≠ r2 → ¬ intersect r1 r2)) :=
by
  sorry

#check square_intersection_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_intersection_configuration_l1181_118182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_expected_value_l1181_118132

/-- The number of envelopes in the game -/
def num_envelopes : ℕ := 3

/-- The minimum amount of money in an envelope -/
def min_amount : ℝ := 0

/-- The maximum amount of money in an envelope -/
def max_amount : ℝ := 1000

/-- The probability of an envelope containing an amount between a and b -/
noncomputable def probability_between (a b : ℝ) : ℝ := (b - a) / (max_amount - min_amount)

/-- The distribution of money in each envelope is uniform -/
axiom uniform_distribution : ∀ (a b : ℝ), min_amount ≤ a → a < b → b ≤ max_amount →
  (probability_between a b = (b - a) / (max_amount - min_amount))

/-- The expected value function for the optimal strategy -/
noncomputable def expected_value : ℕ → ℝ
| 0 => 0
| n + 1 => sorry -- Placeholder for the actual implementation

/-- The optimal strategy achieves the maximum expected value -/
axiom optimal_strategy : ∀ (n : ℕ), n ≤ num_envelopes →
  expected_value n = expected_value n

/-- The theorem to be proved -/
theorem optimal_expected_value :
  expected_value num_envelopes = 695.3125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_expected_value_l1181_118132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trianglePerimeterSumIs9s_l1181_118177

/-- The sum of the perimeters of all triangles formed by the described process --/
noncomputable def trianglePerimeterSum (s : ℝ) : ℝ := 3 * s / (1 - 2/3)

/-- The theorem stating that the sum of perimeters equals 9s --/
theorem trianglePerimeterSumIs9s (s : ℝ) (h : s > 0) : 
  trianglePerimeterSum s = 9 * s := by
  unfold trianglePerimeterSum
  field_simp
  ring

#check trianglePerimeterSumIs9s

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trianglePerimeterSumIs9s_l1181_118177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_triangle_ABC_l1181_118105

/-- Helper function to calculate the area of a triangle given three points in 3D space -/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let (x3, y3, z3) := p3
  let v1 := (x2 - x1, y2 - y1, z2 - z1)
  let v2 := (x3 - x1, y3 - y1, z3 - z1)
  let cross_product := (
    v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,
    v1.1 * v2.2.2 - v1.2.2 * v2.1,
    v1.1 * v2.2.1 - v1.2.1 * v2.1
  )
  (Real.sqrt (cross_product.1^2 + cross_product.2.1^2 + cross_product.2.2^2)) / 2

/-- The smallest area of triangle ABC -/
theorem smallest_area_triangle_ABC :
  let A : ℝ × ℝ × ℝ := (-1, 1, 2)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ → ℝ × ℝ × ℝ := fun t ↦ (t, 2, t)
  ∃ (min_area : ℝ), min_area = Real.sqrt 10 / 2 ∧
    ∀ t : ℝ, area_triangle A B (C t) ≥ min_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_triangle_ABC_l1181_118105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_l1181_118148

-- Define the centers and radii of the circles
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def radius_A : ℝ := 8
def radius_B : ℝ := 5

-- Define the distance between centers
noncomputable def AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem stating all conditions are possible
theorem all_conditions_possible : 
  (∃ (A B : ℝ × ℝ), AB = 3) ∧ 
  (∃ (A B : ℝ × ℝ), AB = 13) ∧ 
  (∃ (A B : ℝ × ℝ), AB > 13) ∧ 
  (∃ (A B : ℝ × ℝ), AB > 3 ∧ AB ≠ 13) :=
by
  constructor
  · sorry -- Proof for AB = 3
  constructor
  · sorry -- Proof for AB = 13
  constructor
  · sorry -- Proof for AB > 13
  · sorry -- Proof for AB > 3 and AB ≠ 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_l1181_118148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_slope_intercept_sum_l1181_118161

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- The sum of slope and y-intercept of the line through Q that bisects the area of triangle PQR -/
theorem triangle_bisector_slope_intercept_sum (P Q R : ℝ × ℝ) : 
  P = (-3, 9) → Q = (4, -2) → R = (10, -2) →
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), y = m * x + b → 
      (x = 4 ∧ y = -2) ∨ 
      (area_triangle P Q (x, y) = area_triangle P R (x, y))) →
    m + b = 31 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_slope_intercept_sum_l1181_118161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_in_range_l1181_118163

theorem fractional_parts_in_range (m n : ℕ) (hm : m > 0) (hn : n > 0) (hdist : m ≠ n) :
  ∃ x : ℝ, (1/3 : ℝ) ≤ (x * ↑n - ⌊x * ↑n⌋) ∧
            (x * ↑n - ⌊x * ↑n⌋) ≤ (2/3 : ℝ) ∧
            (1/3 : ℝ) ≤ (x * ↑m - ⌊x * ↑m⌋) ∧
            (x * ↑m - ⌊x * ↑m⌋) ≤ (2/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_in_range_l1181_118163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_value_l1181_118119

def b : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | 2 => 1
  | n + 3 => (2 - b (n + 2)) / (3 * b (n + 1))

def c : ℕ → ℚ := λ n => 3 * b n - 2

theorem b_120_value : b 120 = -194/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_120_value_l1181_118119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1181_118162

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def coplanar (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def right_angle_D (A D C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def AC_length : ℝ := 10
def AB_length : ℝ := 17
def DC_length : ℝ := 6
def DE_length : ℝ := 8
def E_on_DC_extended (D C E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the area function for a triangle
noncomputable def triangle_area (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC (A B C D E : EuclideanSpace ℝ (Fin 2)) :
  coplanar A B C D E →
  right_angle_D A D C →
  ‖A - C‖ = AC_length →
  ‖A - B‖ = AB_length →
  ‖D - C‖ = DC_length →
  ‖D - E‖ = DE_length →
  E_on_DC_extended D C E →
  triangle_area A B C = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l1181_118162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_three_l1181_118100

def sequence_a : ℕ → ℚ
  | 0 => 2  -- We define a₀ = 2 to handle the zero case
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_2018_equals_negative_three :
  sequence_a 2018 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_three_l1181_118100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_diagonal_cells_good_l1181_118144

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 13
  col : Fin 13

/-- Represents the table -/
def Table := Fin 13 → Fin 13 → Fin 25

/-- The set of cells in the same row, column, and the cell itself -/
def crossSet (c : Cell) : Set Cell :=
  {c' : Cell | c'.row = c.row ∨ c'.col = c.col}

/-- A cell is good if all numbers in its cross set are unique -/
def isGood (t : Table) (c : Cell) : Prop :=
  ∀ c1 c2, c1 ∈ crossSet c → c2 ∈ crossSet c → c1 ≠ c2 → t c1.row c1.col ≠ t c2.row c2.col

/-- The main diagonal of the table -/
def mainDiagonal : Set Cell :=
  {c : Cell | c.row = c.col}

/-- The theorem stating that not all cells on the main diagonal can be good -/
theorem not_all_diagonal_cells_good (t : Table) : 
  ¬(∀ c, c ∈ mainDiagonal → isGood t c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_diagonal_cells_good_l1181_118144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_count_l1181_118118

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the set T
def T : Set ℕ := {n : ℕ | sumOfDigits n = 15 ∧ n < 10^6}

-- Define p as the number of elements in T
noncomputable def p : ℕ := Finset.card (Finset.filter (λ n => sumOfDigits n = 15) (Finset.range (10^6)))

-- Theorem statement
theorem sum_of_digits_of_count : sumOfDigits p = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_count_l1181_118118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_seven_dividing_factorial_l1181_118192

theorem largest_power_of_seven_dividing_factorial : 
  ∃ (w : ℕ), (w = 16 ∧ 
    (∀ k : ℕ, 7^k ∣ Nat.factorial 100 → k ≤ w) ∧ 
    7^w ∣ Nat.factorial 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_seven_dividing_factorial_l1181_118192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l1181_118110

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := 2 - a * Complex.I

-- Statement of the theorem
theorem a_eq_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → Complex.abs (z a) = 2 * Real.sqrt 2) ∧
  (∃ a : ℝ, a ≠ 2 ∧ Complex.abs (z a) = 2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_2_sufficient_not_necessary_l1181_118110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l1181_118179

/-- Prove that the speed of a man rowing downstream is 25 kmph -/
theorem downstream_speed (upstream_speed still_water_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 5)
  (h2 : still_water_speed = 15)
  (h3 : still_water_speed = (upstream_speed + downstream_speed) / 2) :
  downstream_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_speed_l1181_118179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_a_b_c_l1181_118166

-- Define the constants
noncomputable def a : ℝ := 3^(-0.1 : ℝ)
noncomputable def b : ℝ := -Real.log 5 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 2 / Real.log (Real.sqrt 3)

-- State the theorem
theorem order_of_a_b_c : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_a_b_c_l1181_118166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l1181_118193

theorem solution_to_equation (x : ℝ) : (9 : ℝ)^x + (3 : ℝ)^x - 2 = 0 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l1181_118193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_inscribed_circle_theorem_l1181_118150

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the first leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle -/
  leg2 : ℝ
  /-- Length of the hypotenuse of the right triangle -/
  hypotenuse : ℝ
  /-- Radius of the inscribed circle -/
  inradius : ℝ
  /-- Distance from the right angle vertex to the center of the inscribed circle -/
  center_distance : ℝ
  /-- The point of tangency divides the hypotenuse in this ratio -/
  hypotenuse_ratio : ℝ × ℝ
  /-- The triangle is a right triangle -/
  right_angle : leg1^2 + leg2^2 = hypotenuse^2
  /-- The circle is inscribed in the triangle -/
  inscribed_circle : inradius = (leg1 + leg2 - hypotenuse) / 2
  /-- The point of tangency divides the hypotenuse in the given ratio -/
  tangency_ratio : hypotenuse_ratio.1 / (hypotenuse_ratio.1 + hypotenuse_ratio.2) * hypotenuse = leg1 - inradius
  /-- The center distance is correct -/
  center_distance_correct : center_distance = Real.sqrt 8

/-- The main theorem -/
theorem right_triangle_with_inscribed_circle_theorem (t : RightTriangleWithInscribedCircle) 
  (h_ratio : t.hypotenuse_ratio = (2, 3)) :
  t.leg1 = 6 ∧ t.leg2 = 8 ∧ t.hypotenuse = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_inscribed_circle_theorem_l1181_118150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equal_two_implies_cos_triangle_condition_implies_f_range_l1181_118152

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3 * Real.sin (x / 4), 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), Real.cos (x / 4) ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (m x) (n x)

theorem dot_product_equal_two_implies_cos (x : ℝ) :
  dot_product (m x) (n x) = 2 → Real.cos (x + π/3) = 1/2 := by sorry

theorem triangle_condition_implies_f_range (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  ∃ (l u : ℝ), l = 2 ∧ u = 3 ∧ ∀ y ∈ Set.Ioo l u, ∃ A', f A' = y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equal_two_implies_cos_triangle_condition_implies_f_range_l1181_118152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_approximately_16_79_percent_l1181_118126

/-- Represents the cost and price increase of an item --/
structure Item where
  initialCost : ℝ
  priceIncrease : ℝ

/-- Calculates the total percentage increase given a list of items --/
noncomputable def totalPercentageIncrease (items : List Item) : ℝ :=
  let initialTotal := items.foldl (fun acc item => acc + item.initialCost) 0
  let newTotal := items.foldl (fun acc item => acc + item.initialCost * (1 + item.priceIncrease)) 0
  (newTotal - initialTotal) / initialTotal * 100

/-- Theorem stating that the given items result in approximately 16.79% increase --/
theorem price_increase_approximately_16_79_percent :
  let items : List Item := [
    { initialCost := 200, priceIncrease := 0.15 },  -- Scooter
    { initialCost := 60,  priceIncrease := 0.20 },  -- Safety gear
    { initialCost := 20,  priceIncrease := 0.25 }   -- Maintenance kit
  ]
  abs (totalPercentageIncrease items - 16.79) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_approximately_16_79_percent_l1181_118126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_5_l1181_118122

noncomputable def g (x : ℝ) : ℝ := 25 / (2 + 5 * x)

theorem inverse_g_at_5 : (g⁻¹ 5)⁻¹ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_5_l1181_118122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_divisibility_l1181_118183

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ), x > 0 → c > 0 →
  ∃ (k : ℕ), 
  (15 : ℕ) * k = (Finset.range 15).sum (λ i ↦ x + i * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_divisibility_l1181_118183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1181_118137

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Define the line
def line_AB (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line segment
def in_line_segment (A B : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t • A.1 + (1 - t) • B.1, t • A.2 + (1 - t) • B.2)

-- Theorem statement
theorem intersection_line_equation :
  ∀ A B : ℝ × ℝ,
  (circle1 A.1 A.2 ∧ circle2 A.1 A.2) →
  (circle1 B.1 B.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  (∀ x y : ℝ, in_line_segment A B (x, y) ↔ line_AB x y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l1181_118137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l1181_118115

/-- The total number of students in a class, given that 1/4 are girls and there are 120 boys -/
theorem total_students (girls_ratio : Rat) (boys_count : Nat) : Nat :=
  let total_count := 160
  have h1 : girls_ratio = 1 / 4 := by sorry
  have h2 : boys_count = 120 := by sorry
  have h3 : total_count = 160 := by sorry
  total_count


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_l1181_118115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_congruence_solutions_l1181_118102

theorem two_digit_congruence_solutions : 
  let two_digit_solutions := {x : ℕ | 10 ≤ x ∧ x < 100 ∧ (3269 * x + 532) % 17 = 875 % 17}
  Finset.card (Finset.filter (fun x => 10 ≤ x ∧ x < 100 ∧ (3269 * x + 532) % 17 = 875 % 17) (Finset.range 100)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_congruence_solutions_l1181_118102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_number_l1181_118133

theorem existence_of_divisible_number : ∃ (n : ℕ) (m : ℕ), ∃ (k : ℕ),
  (k * 2022 = (2021 : ℕ) * (10^(4*n + 4) - 1) / 9 * 10^m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_number_l1181_118133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1181_118180

-- Define proposition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

-- Define proposition q
def q (x : ℝ) : Prop := ∃ m : ℝ, m ∈ Set.Ioo 1 2 ∧ x = (1/2)^(m-1)

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p x (1/4) ∧ q x → x ∈ Set.Ioo (1/2) (3/4) := by sorry

-- Part 2
theorem part_two : 
  {a : ℝ | ∀ x : ℝ, ¬(q x) → ¬(p x a) ∧ ∃ y : ℝ, ¬(q y) ∧ p y a} = Set.Icc (1/3) (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1181_118180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rank_of_remaining_weight_l1181_118117

/-- Represents a set of weights with their masses -/
structure WeightSet (n : ℕ) where
  masses : Fin (2^n) → ℝ
  distinct : ∀ i j, i ≠ j → masses i ≠ masses j

/-- Represents the process of weighing and selecting weights -/
noncomputable def weighing_process (n : ℕ) (w : WeightSet n) : Fin (2^n) :=
  sorry

theorem minimum_rank_of_remaining_weight (n : ℕ) (w : WeightSet n) :
  ∃ (rank : ℕ), rank ≥ 2^n - n ∧
    ∀ (i : Fin (2^n)),
      (w.masses i ≥ w.masses (weighing_process n w)) →
        (Finset.card (Finset.filter (λ j => w.masses j ≥ w.masses i) (Finset.univ)) ≤ rank) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rank_of_remaining_weight_l1181_118117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sum_digits_l1181_118151

theorem binary_sum_digits : ∃ (n : ℕ), 
  (Nat.log2 (350 + 1500) + 1 : ℕ) = n ∧ n = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sum_digits_l1181_118151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_to_rhombus_equal_area_l1181_118139

/-- Given a parallelogram with side lengths a and b, prove that a rhombus with side length √(a * b) has the same area. -/
theorem parallelogram_to_rhombus_equal_area 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) :
  a * (b * Real.sin (π / 3)) = (Real.sqrt (a * b))^2 * Real.sin (π / 2) :=
by
  -- Define the areas
  let parallelogram_area := a * (b * Real.sin (π / 3))
  let rhombus_side := Real.sqrt (a * b)
  let rhombus_area := rhombus_side^2 * Real.sin (π / 2)
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_to_rhombus_equal_area_l1181_118139
