import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_solution_l614_61423

/-- Two lines are symmetric with respect to y = x if and only if
    their slopes are reciprocals and their y-intercepts sum to 0 -/
axiom symmetry_condition (m₁ m₂ b₁ b₂ : ℝ) :
  (∀ x y, y = m₁ * x + b₁) ↔ (∀ x y, y = m₂ * x + b₂) ∧ m₁ * m₂ = 1 ∧ b₁ + b₂ = 0

/-- The problem statement -/
theorem symmetric_lines_solution (a b : ℝ) :
  (∀ x y, y = a * x - 4) ↔ (∀ x y, y = 8 * x - b) →
  a = 1/8 ∧ b = -32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_lines_solution_l614_61423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_on_positive_measure_set_l614_61479

open MeasureTheory Measure RealInnerProductSpace

theorem function_lower_bound_on_positive_measure_set
  (f : ℝ → ℝ) (n : ℕ) (h_integrable : Integrable f volume) :
  (∀ m < n, ∫ x in (0:ℝ)..1, (x ^ m) * f x = 0) →
  (∫ x in (0:ℝ)..1, (x ^ n) * f x = 1) →
  ∃ (s : Set ℝ), MeasurableSet s ∧ volume s > 0 ∧ ∀ x ∈ s, |f x| ≥ 2 * n * (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_on_positive_measure_set_l614_61479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l614_61444

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
noncomputable def line1 : Line2D :=
  { point := (3, 2),
    direction := (3, -4) }

/-- The second line -/
noncomputable def line2 : Line2D :=
  { point := (7, -5),
    direction := (6, 3) }

/-- A point is on a line if there exists a parameter t such that
    the point equals the line's point plus t times the direction vector -/
def on_line (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧
            p.2 = l.point.2 + t * l.direction.2

/-- The intersection point of the two lines -/
noncomputable def intersection_point : ℝ × ℝ := (87/11, -50/11)

/-- Theorem stating that the intersection_point is on both lines and is unique -/
theorem intersection_point_is_unique :
  on_line intersection_point line1 ∧
  on_line intersection_point line2 ∧
  ∀ p : ℝ × ℝ, on_line p line1 ∧ on_line p line2 → p = intersection_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_unique_l614_61444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l614_61439

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = Real.sqrt 2) ∧
  (∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f (x + p) = f x) → p ≥ Real.pi) ∧
  (∀ (x : ℝ), f x ≥ 1 ↔ ∃ (k : ℤ), k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l614_61439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l614_61459

theorem candy_problem (x : ℕ) : x ∈ ({92, 96, 100} : Set ℕ) →
  (x % 4 = 0) ∧
  (92 ≤ x ∧ x ≤ 102) ∧
  (∃ k : ℕ, 10 ≤ k ∧ k ≤ 15 ∧ x / 2 - 36 = k) ∧
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 6 ∧ (3 * x / 4) * 2 / 3 - 36 - n = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_problem_l614_61459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l614_61468

theorem smallest_positive_solution_tan_sec_equation :
  let f : ℝ → ℝ := λ x => Real.tan (3 * x) + Real.tan (4 * x) - 1 / Real.cos (4 * x)
  ∃ (x : ℝ), x > 0 ∧ f x = 0 ∧ ∀ (y : ℝ), y > 0 → f y = 0 → x ≤ y ∧ x = π / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l614_61468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l614_61421

/-- The minimum distance from a point on the parabola y = x^2 to the line 2x - y - 10 = 0 is 9√5/5 -/
theorem min_distance_parabola_to_line : 
  ∃ (d : ℝ), d = (9 * Real.sqrt 5) / 5 ∧ 
  ∀ (x : ℝ), 
    (|2*x - x^2 - 10| / Real.sqrt 5) ≥ d := by
  -- Let d be the minimum distance
  let d := (9 * Real.sqrt 5) / 5
  
  -- We claim that this d satisfies our conditions
  use d
  
  constructor
  
  -- First part: show that d equals (9 * √5) / 5
  · rfl
  
  -- Second part: show that for all x, the distance is greater than or equal to d
  · intro x
    -- The distance formula
    let dist := |2*x - x^2 - 10| / Real.sqrt 5
    
    -- We need to prove: dist ≥ d
    sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l614_61421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l614_61458

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

-- Define the original triangle
def original_triangle : Triangle := {
  angle1 := 20,
  angle2 := 20,
  angle3 := 140,
  sum_180 := by norm_num
}

-- Define a function to check if a triangle is similar to the original
def is_similar_to_original (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t.angle1 = k * original_triangle.angle1 ∧
    t.angle2 = k * original_triangle.angle2 ∧
    t.angle3 = k * original_triangle.angle3

-- Define a function to represent a single cut along an angle bisector
noncomputable def cut_along_bisector (t : Triangle) : Triangle × Triangle :=
  sorry -- Implementation details omitted

-- Theorem stating that it's impossible to obtain a similar triangle
theorem no_similar_triangle_after_cuts :
  ∀ (n : ℕ) (cuts : ℕ → Triangle → Triangle × Triangle),
  (∀ i, cuts i = cut_along_bisector) →
  ∀ (result : Triangle),
  (∃ (sequence : ℕ → Triangle), 
    sequence 0 = original_triangle ∧
    (∀ i : ℕ, i > 0 → i ≤ n → 
      sequence i ∈ [((cuts (i - 1) (sequence (i - 1))).1),
                    ((cuts (i - 1) (sequence (i - 1))).2)]) ∧
    result = sequence n) →
  ¬(is_similar_to_original result) :=
by
  sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l614_61458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_area_theorem_l614_61485

/-- Represents the speed of the vehicle in different terrains -/
structure VehicleSpeed where
  roadSpeed : ℝ
  sandSpeed : ℝ

/-- Represents the time limit for the vehicle's travel -/
noncomputable def timeLimit : ℝ := 8 / 60 -- 8 minutes converted to hours

/-- Helper function to calculate the area of the reachable region -/
noncomputable def area_of_reachable_region (speed : VehicleSpeed) (time : ℝ) : ℝ :=
  sorry

/-- The theorem stating the area of the reachable region -/
theorem reachable_area_theorem (speed : VehicleSpeed) 
  (h1 : speed.roadSpeed = 60)
  (h2 : speed.sandSpeed = 10) :
  ∃ (area : ℝ), area = 10874 / 225 ∧ 
  area * 225 = area_of_reachable_region speed timeLimit :=
by
  sorry

#check reachable_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachable_area_theorem_l614_61485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l614_61497

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 - (1 - Complex.I) / Complex.I : ℂ) = Complex.mk a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l614_61497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_equal_227_l614_61493

/-- Represents the types of fruits sold by Wendy -/
inductive Fruit
  | Apple
  | Orange
  | Banana

/-- Represents the time of day for sales -/
inductive TimeOfDay
  | Morning
  | Afternoon

/-- Calculates the price of a fruit at a given time of day -/
def price (f : Fruit) (t : TimeOfDay) : ℚ :=
  match f, t with
  | Fruit.Apple, TimeOfDay.Morning => 3/2
  | Fruit.Orange, TimeOfDay.Morning => 1
  | Fruit.Banana, TimeOfDay.Morning => 3/4
  | Fruit.Apple, TimeOfDay.Afternoon => 27/20
  | Fruit.Orange, TimeOfDay.Afternoon => 9/10
  | Fruit.Banana, TimeOfDay.Afternoon => 27/40

/-- Calculates the sales for a specific fruit at a given time -/
def sales (f : Fruit) (t : TimeOfDay) : ℕ :=
  match f, t with
  | Fruit.Apple, TimeOfDay.Morning => 40
  | Fruit.Orange, TimeOfDay.Morning => 30
  | Fruit.Banana, TimeOfDay.Morning => 10
  | Fruit.Apple, TimeOfDay.Afternoon => 50
  | Fruit.Orange, TimeOfDay.Afternoon => 40
  | Fruit.Banana, TimeOfDay.Afternoon => 20

/-- Calculates the number of unsold fruits -/
def unsold (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 0
  | Fruit.Orange => 10
  | Fruit.Banana => 20

/-- Calculates the total sales for the day, including anticipated revenue from unsold fruits -/
def totalSales : ℚ :=
  let fruits := [Fruit.Apple, Fruit.Orange, Fruit.Banana]
  fruits.foldl
    (fun acc f =>
      acc +
      (sales f TimeOfDay.Morning) * (price f TimeOfDay.Morning) +
      (sales f TimeOfDay.Afternoon) * (price f TimeOfDay.Afternoon) +
      (unsold f) * (price f TimeOfDay.Morning) / 2)
    0

/-- The main theorem stating that the total sales is equal to $227 -/
theorem total_sales_equal_227 : totalSales = 227 := by
  sorry

#eval totalSales

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_equal_227_l614_61493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l614_61430

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi

-- Define the dot product of vectors
def dotProduct (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : 2 * (dotProduct t.b t.c) = t.a^2 - (t.b + t.c)^2) :
  t.A = 2 * Real.pi / 3 ∧ 
  (∃ (max : ℝ), max = Real.sqrt 3 / 8 ∧ 
    ∀ x y z, x + y + z = Real.pi → 
      Real.sin x * Real.sin y * Real.sin z ≤ max) ∧
  (Real.sin t.A * Real.sin t.B * Real.sin t.C = Real.sqrt 3 / 8 → 
    t.B = Real.pi / 6 ∧ t.C = Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l614_61430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l614_61418

noncomputable def f (x : ℝ) := Real.sqrt (2 * x - 3)

theorem f_domain : Set.Ici (3/2 : ℝ) = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l614_61418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l614_61475

theorem triangle_side_count : ∃ (count : ℕ), count = 29 ∧
  count = (Finset.filter (λ x : ℕ => 
    x > 0 ∧ x + 15 > 40 ∧ x + 40 > 15 ∧ 15 + 40 > x) 
    (Finset.range 100)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l614_61475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_with_ten_points_l614_61433

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- A frame is a 4x4 square without its central 2x2 square -/
structure Frame where
  outerSquare : Square

/-- Checks if a point is inside a frame -/
def isPointInFrame (p : Point) (f : Frame) : Prop := sorry

/-- Counts the number of points in a frame -/
def countPointsInFrame (points : Finset Point) (f : Frame) : ℕ := sorry

/-- The main theorem -/
theorem frame_with_ten_points 
  (square : Square) 
  (points : Finset Point) : 
  square.sideLength = 96 → 
  points.card = 7501 → 
  (∀ p ∈ points, p.x ≥ square.bottomLeft.x ∧ p.x ≤ square.bottomLeft.x + 96 ∧ 
                 p.y ≥ square.bottomLeft.y ∧ p.y ≤ square.bottomLeft.y + 96) →
  ∃ (f : Frame), countPointsInFrame points f ≥ 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_with_ten_points_l614_61433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l614_61478

noncomputable section

-- Define the initial deposits
def alice_deposit : ℝ := 15000
def bob_deposit : ℝ := 17000

-- Define the interest rates
def alice_rate : ℝ := 0.06
def bob_rate : ℝ := 0.08

-- Define the time period
def years : ℝ := 20

-- Define the compounding frequency for Alice's account
def compounding_frequency : ℝ := 2

-- Function to calculate Alice's balance (compound interest)
noncomputable def alice_balance : ℝ := 
  alice_deposit * (1 + alice_rate / compounding_frequency) ^ (compounding_frequency * years)

-- Function to calculate Bob's balance (simple interest)
noncomputable def bob_balance : ℝ := 
  bob_deposit * (1 + bob_rate * years)

-- Theorem to prove
theorem balance_difference_approx : 
  abs (bob_balance - alice_balance - 7791) < 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_difference_approx_l614_61478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l614_61457

def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 = 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem possible_m_values :
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ∈ ({-1, -1/3, 0} : Set ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_m_values_l614_61457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_numbers_divisibility_l614_61403

theorem seventeen_numbers_divisibility (S : Finset ℕ) :
  S.card = 17 →
  (∃ a b c d e, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ∣ d ∧ d ∣ c ∧ c ∣ b ∧ b ∣ a) ∨
  (∃ a b c d e, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧
    ¬(a ∣ b) ∧ ¬(a ∣ c) ∧ ¬(a ∣ d) ∧ ¬(a ∣ e) ∧
    ¬(b ∣ a) ∧ ¬(b ∣ c) ∧ ¬(b ∣ d) ∧ ¬(b ∣ e) ∧
    ¬(c ∣ a) ∧ ¬(c ∣ b) ∧ ¬(c ∣ d) ∧ ¬(c ∣ e) ∧
    ¬(d ∣ a) ∧ ¬(d ∣ b) ∧ ¬(d ∣ c) ∧ ¬(d ∣ e) ∧
    ¬(e ∣ a) ∧ ¬(e ∣ b) ∧ ¬(e ∣ c) ∧ ¬(e ∣ d)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventeen_numbers_divisibility_l614_61403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l614_61499

theorem problem_solution (x y : ℕ) (h1 : Nat.lcm x y = 36) (h2 : Nat.gcd x y = 2) (h3 : y = 18) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l614_61499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l614_61486

/-- Given a sphere O with radius R and a plane perpendicular to a radius OP at its midpoint M,
    intersecting the sphere to form circle O₁, the volume ratio of the sphere with O₁ as its
    great circle to sphere O is (3/8)√3. -/
theorem sphere_volume_ratio (R : ℝ) (hR : R > 0) : 
  (4/3 * Real.pi * ((Real.sqrt ((3/4) * R^2))^3)) / (4/3 * Real.pi * R^3) = (3/8) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l614_61486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l614_61456

-- Define the quadrants
inductive Quadrant
| I
| II
| III
| IV

-- Define a function to determine the quadrant of an angle
noncomputable def terminalSideQuadrant (θ : ℝ) : Quadrant :=
  sorry

-- Theorem statement
theorem terminal_side_in_third_quadrant (θ : ℝ) :
  Real.sin (π - θ) < 0 → Real.tan (π - θ) < 0 → terminalSideQuadrant θ = Quadrant.III := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_third_quadrant_l614_61456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l614_61463

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2012 + b^2013 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l614_61463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nineteen_numbers_with_same_digit_sum_summing_to_1999_l614_61437

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10 + digit_sum (n / 10))

theorem no_nineteen_numbers_with_same_digit_sum_summing_to_1999 :
  ¬ (∃ (S : Finset ℕ), 
    (Finset.card S = 19) ∧ 
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → True) ∧
    (∃ (d : ℕ), ∀ n, n ∈ S → digit_sum n = d) ∧
    (Finset.sum S id = 1999)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nineteen_numbers_with_same_digit_sum_summing_to_1999_l614_61437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_range_l614_61420

theorem z_range (x y z : ℝ) 
  (eq1 : x + y + 2*z = 5) 
  (eq2 : x*y + y*z + z*x = 3) : 
  -Real.sqrt 13 / 2 ≤ z ∧ z ≤ Real.sqrt 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_range_l614_61420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inscribed_circle_radius_obtuse_triangle_inscribed_circle_radius_counterexample_l614_61452

/-- A triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- The circle in which the triangle is inscribed -/
  circle : Set (ℝ × ℝ)
  /-- The three vertices of the triangle -/
  vertices : Fin 3 → ℝ × ℝ
  /-- Proof that the vertices lie on the circle -/
  on_circle : ∀ i, vertices i ∈ circle

/-- The radius of a circle -/
noncomputable def circle_radius (c : Set (ℝ × ℝ)) : ℝ := sorry

/-- The circumradius of a triangle -/
noncomputable def triangle_circumradius (v : Fin 3 → ℝ × ℝ) : ℝ := sorry

/-- Checks if a triangle is acute -/
def is_acute_triangle (v : Fin 3 → ℝ × ℝ) : Prop := sorry

/-- Checks if a triangle is obtuse -/
def is_obtuse_triangle (v : Fin 3 → ℝ × ℝ) : Prop := sorry

/-- Theorem: For any acute triangle inscribed in a circle, 
    the radius of the circle is greater than or equal to the circumradius of the triangle -/
theorem acute_triangle_inscribed_circle_radius 
  (t : InscribedTriangle) (h : is_acute_triangle t.vertices) : 
  circle_radius t.circle ≥ triangle_circumradius t.vertices := by sorry

/-- Theorem: The statement does not necessarily hold for obtuse triangles -/
theorem obtuse_triangle_inscribed_circle_radius_counterexample :
  ∃ (t : InscribedTriangle), is_obtuse_triangle t.vertices ∧ 
  circle_radius t.circle < triangle_circumradius t.vertices := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inscribed_circle_radius_obtuse_triangle_inscribed_circle_radius_counterexample_l614_61452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_set_is_spherical_surface_l614_61488

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A spherical surface in 3D space -/
def SphericalSurface (center : Point3D) (radius : ℝ) : Set Point3D :=
  {p : Point3D | distance p center = radius}

/-- Theorem: The set of all points in 3D space whose distance from a fixed point
    is equal to a constant forms a spherical surface -/
theorem constant_distance_set_is_spherical_surface 
  (center : Point3D) (r : ℝ) (h : r > 0) :
  {p : Point3D | distance p center = r} = SphericalSurface center r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_set_is_spherical_surface_l614_61488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l614_61426

noncomputable def f (a b x : ℝ) : ℝ :=
  if x > 3 then a * x + 2
  else if x ≥ -3 then x - 6
  else 2 * x - b

theorem continuous_piecewise_function_sum (a b : ℝ) :
  (∀ x, ContinuousAt (f a b) x) → a + b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_sum_l614_61426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l614_61482

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

theorem min_shift_for_symmetry :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (ψ : ℝ), ψ > 0 → 
    (∀ (x : ℝ), f (x - ψ) = f (-(x - ψ))) → 
    φ ≤ ψ) ∧
  (∀ (x : ℝ), f (x - φ) = f (-(x - φ))) ∧
  φ = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l614_61482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_unit_l614_61410

theorem divisibility_implies_unit (a b c d : ℤ) 
  (h : ∀ x ∈ ({a, b, c, d} : Set ℤ), (a * d + b * c) ∣ x) : 
  a * d + b * c = 1 ∨ a * d + b * c = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_unit_l614_61410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_cord_length_l614_61442

theorem dog_cord_length (arc_length : ℝ) (h : arc_length = 30) : 
  ∃ (cord_length : ℝ), abs (cord_length - 9.55) < 0.01 ∧ arc_length = π * cord_length := by
  sorry

#check dog_cord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_cord_length_l614_61442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transition_l614_61400

/-- Represents the color of a bead -/
inductive Color where
  | Black
  | Blue
  | Green
deriving Repr, DecidableEq

/-- Represents the state of the beads -/
structure BeadState where
  total : Nat
  black : Nat
  blue : Nat
  green : Nat

/-- The replacement rule for beads -/
def replaceRule (left right : Color) : Color :=
  if left = right then left else Color.Blue

/-- Theorem stating the impossibility of the transition -/
theorem impossible_transition (initial : BeadState) :
  initial.total = 2016 ∧ 
  initial.black = 1000 ∧ 
  initial.green = 1016 ∧ 
  initial.blue = 0 →
  ¬∃ (steps : Nat), ∃ (final : BeadState),
    final.total = 2016 ∧
    final.blue = 2016 ∧
    final.black = 0 ∧
    final.green = 0 :=
by
  sorry

#check impossible_transition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transition_l614_61400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_first_and_twelfth_intersection_l614_61477

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 2) * cos (x - π / 2)

-- Define the intersection points
def intersection_points : Set ℝ := {x | f x = -1/2 ∧ x > 0}

-- Define the nth intersection point
noncomputable def nth_intersection (n : ℕ) : ℝ := 
  if n = 0 then 0 else
  (7 * π) / 12 + ((n - 1) / 2 : ℝ) * π + (if n % 2 = 0 then π / 3 else 0)

-- Theorem statement
theorem distance_between_first_and_twelfth_intersection :
  |nth_intersection 12 - nth_intersection 1| = 16 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_first_and_twelfth_intersection_l614_61477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_success_probability_is_one_eighth_l614_61417

/-- The number of lily pads -/
def n : ℕ := 8

/-- The probability of the frog successfully landing on lily pad i+1 when jumping from lily pad i -/
def p (i : ℕ) : ℚ := i / (i + 1)

/-- The probability of the frog successfully reaching lily pad 8 from lily pad 1 -/
def frog_success_probability : ℚ :=
  Finset.prod (Finset.range (n - 1)) p

theorem frog_success_probability_is_one_eighth :
  frog_success_probability = 1 / 8 := by
  sorry

#eval 100 * frog_success_probability.num + frog_success_probability.den

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_success_probability_is_one_eighth_l614_61417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l614_61496

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, divided by π -/
noncomputable def cone_volume_divided_by_pi : ℝ :=
  let sector_angle : ℝ := 270
  let circle_radius : ℝ := 20
  let base_radius : ℝ := (3/4) * circle_radius
  let cone_height : ℝ := Real.sqrt (circle_radius^2 - base_radius^2)
  (1/3) * base_radius^2 * cone_height

theorem cone_volume_proof : cone_volume_divided_by_pi = 375 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l614_61496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l614_61432

-- Define the decimal number
def x : ℚ := 2.71717171

-- Define the fractional representation
def frac : ℚ := 269 / 99

-- Theorem statement
theorem sum_of_numerator_and_denominator :
  x = frac ∧ (frac.num.natAbs + frac.den = 368) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_l614_61432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l614_61446

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x : ℕ | x < 2}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l614_61446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_red_prob_first_prize_prob_second_prize_prob_third_prize_expectation_winners_lottery_theorem_l614_61495

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White
| Yellow

/-- Represents the lottery box -/
def LotteryBox : Finset BallColor := sorry

/-- The total number of balls in the box -/
def totalBalls : ℕ := 10

/-- The number of red balls -/
def redBalls : ℕ := 2

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of yellow balls -/
def yellowBalls : ℕ := 5

/-- Theorem stating the probability of drawing a red ball second -/
theorem prob_second_red : ℚ :=
  1 / 5

/-- Theorem stating the probability of winning first prize -/
theorem prob_first_prize : ℚ :=
  2 / 15

/-- Theorem stating the probability of winning second prize -/
theorem prob_second_prize : ℚ :=
  1 / 45

/-- Theorem stating the probability of winning third prize -/
theorem prob_third_prize : ℚ :=
  1 / 15

/-- The number of independent draws -/
def numDraws : ℕ := 3

/-- The probability of winning any prize in a single draw -/
def probWin : ℚ := 2 / 9

/-- Represents the probability distribution of the number of winners -/
noncomputable def probDistribution (x : ℕ) : ℚ :=
  (numDraws.choose x : ℚ) * (probWin ^ x) * ((1 - probWin) ^ (numDraws - x))

/-- Theorem stating the expectation of the number of winners -/
theorem expectation_winners : ℚ :=
  2 / 3

/-- Main theorem combining all results -/
theorem lottery_theorem :
  (prob_second_red = 1 / 5) ∧
  (prob_first_prize = 2 / 15) ∧
  (prob_second_prize = 1 / 45) ∧
  (prob_third_prize = 1 / 15) ∧
  (∀ x, x ≤ numDraws → probDistribution x = (numDraws.choose x : ℚ) * (probWin ^ x) * ((1 - probWin) ^ (numDraws - x))) ∧
  (expectation_winners = 2 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_second_red_prob_first_prize_prob_second_prize_prob_third_prize_expectation_winners_lottery_theorem_l614_61495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l614_61416

-- Define the ellipse
def Ellipse (h k a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  let F₁ : ℝ × ℝ := (0, 2)
  let F₂ : ℝ × ℝ := (6, 2)
  (∀ p : ℝ × ℝ, p ∈ Ellipse h k a b ↔ distance p F₁ + distance p F₂ = 10) →
  h + k + a + b = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l614_61416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_even_condition_l614_61451

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The sine function with phase shift φ -/
noncomputable def SineWithPhase (φ : ℝ) : ℝ → ℝ :=
  fun x ↦ Real.sin (x + φ)

/-- φ = -π/2 is sufficient but not necessary for SineWithPhase φ to be even -/
theorem sine_phase_shift_even_condition :
  (∃ φ, φ ≠ -π/2 ∧ IsEven (SineWithPhase φ)) ∧
  IsEven (SineWithPhase (-π/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_phase_shift_even_condition_l614_61451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incentive_scheme_constraints_l614_61467

noncomputable def incentive_function (a : ℝ) (x : ℝ) : ℝ := (15 * x - a) / (x + 8)

theorem incentive_scheme_constraints (a : ℝ) : 
  (∀ x ∈ Set.Icc 50 500, 
    incentive_function a x ≥ 7 ∧ 
    incentive_function a x ≤ 0.15 * x ∧ 
    incentive_function a x ≤ incentive_function a (x + 1)) ↔ 
  a ∈ Set.Icc 315 344 := by
  sorry

#check incentive_scheme_constraints

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incentive_scheme_constraints_l614_61467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_talents_count_l614_61419

/-- Represents the talents of students at a summer camp -/
inductive Talent
  | Sing
  | Dance
  | Act

/-- Represents a student at the summer camp -/
structure Student where
  talents : Finset Talent

/-- The summer camp scenario -/
structure SummerCamp where
  students : Finset Student
  total_students : Nat
  cannot_sing : Nat
  cannot_dance : Nat
  cannot_act : Nat

/-- Properties of the summer camp -/
axiom camp_properties (camp : SummerCamp) :
  camp.total_students = 120 ∧
  camp.cannot_sing = 50 ∧
  camp.cannot_dance = 75 ∧
  camp.cannot_act = 40

/-- Each student has at least one talent -/
axiom student_has_talent (camp : SummerCamp) (s : Student) :
  s ∈ camp.students → s.talents.Nonempty

/-- No student has all three talents -/
axiom no_student_has_all_talents (camp : SummerCamp) (s : Student) :
  s ∈ camp.students → s.talents.card < 3

/-- Count students with exactly two talents -/
def count_two_talents (camp : SummerCamp) : Nat :=
  (camp.students.filter (fun s => s.talents.card = 2)).card

/-- Main theorem: The number of students with exactly two talents is 75 -/
theorem two_talents_count (camp : SummerCamp) :
  count_two_talents camp = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_talents_count_l614_61419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_and_variance_of_Z_l614_61466

-- Define the probability distributions
def prob_X : ℕ → ℝ
| 1 => 0.1
| 2 => 0.6
| 3 => 0.3
| _ => 0

def prob_Y : ℕ → ℝ
| 0 => 0.2
| 1 => 0.8
| _ => 0

-- Define the random variables X and Y
def X : ℕ → ℝ := prob_X
def Y : ℕ → ℝ := prob_Y

-- Define Z as the sum of X and Y
def Z (x y : ℕ) : ℝ := X x + Y y

-- State the theorem
theorem expectation_and_variance_of_Z :
  (∑' x, x * prob_X x) + (∑' y, y * prob_Y y) = 3 ∧
  (∑' x, x^2 * prob_X x) - (∑' x, x * prob_X x)^2 +
  (∑' y, y^2 * prob_Y y) - (∑' y, y * prob_Y y)^2 = 0.52 := by
  sorry

-- Note: The above theorem states that the expected value of Z is 3
-- and the variance of Z is 0.52, using the properties of expectation
-- and variance for independent random variables.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expectation_and_variance_of_Z_l614_61466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l614_61471

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x + 2| + |x| + |x - 2| = 4 :=
by
  -- The unique solution is x = 0
  use 0
  constructor
  · -- Prove that x = 0 satisfies the equation
    simp
    norm_num
  · -- Prove uniqueness
    intro y hy
    -- We'll use 'sorry' to skip the detailed proof for now
    sorry

#check absolute_value_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_l614_61471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_correct_l614_61443

/-- Represents a class with students and their marks -/
structure ClassInfo where
  studentCount : Nat
  initialAverage : Float
  studentAOldMark : Float
  studentANewMark : Float
  studentBOldMark : Float
  studentBNewMark : Float
  studentCOldMark : Float
  studentCNewMark : Float

/-- Calculates the new average mark after correcting three students' marks -/
def calculateNewAverage (c : ClassInfo) : Float :=
  let oldTotal := c.initialAverage * c.studentCount.toFloat
  let correction := (c.studentANewMark - c.studentAOldMark) + 
                    (c.studentBNewMark - c.studentBOldMark) + 
                    (c.studentCNewMark - c.studentCOldMark)
  let newTotal := oldTotal + correction
  newTotal / c.studentCount.toFloat

/-- Theorem stating that the new average mark is 85.214 given the specified conditions -/
theorem new_average_is_correct (c : ClassInfo) 
  (h1 : c.studentCount = 50)
  (h2 : c.initialAverage = 85.4)
  (h3 : c.studentAOldMark = 73.6)
  (h4 : c.studentANewMark = 63.5)
  (h5 : c.studentBOldMark = 92.4)
  (h6 : c.studentBNewMark = 96.7)
  (h7 : c.studentCOldMark = 55.3)
  (h8 : c.studentCNewMark = 51.8) :
  calculateNewAverage c = 85.214 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_correct_l614_61443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notation_for_3_in_row_5_l614_61445

/-- Represents an element in a row -/
structure ElementInRow where
  element : ℕ
  row : ℕ

/-- The notation for an element in a row -/
def elementNotation (e : ElementInRow) : ℕ × ℕ := (e.element, e.row)

/-- Given condition: "8 in row 4" is denoted as (8,4) -/
axiom example_notation : elementNotation ⟨8, 4⟩ = (8, 4)

/-- Theorem: The notation for "3 in row 5" is (3,5) -/
theorem notation_for_3_in_row_5 : 
  elementNotation ⟨3, 5⟩ = (3, 5) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notation_for_3_in_row_5_l614_61445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_9_plus_4pi_over_8_l614_61434

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 2
  else if x > 2 ∧ x ≤ 4 then Real.sqrt (1 - (x - 3)^2)
  else 0

-- State the theorem
theorem integral_f_equals_9_plus_4pi_over_8 :
  ∫ x in (1/2)..4, f x = (9 + 4 * Real.pi) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_9_plus_4pi_over_8_l614_61434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_solution_l614_61453

noncomputable section

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_triangle : Triangle where
  A := 60 * Real.pi / 180  -- Convert to radians
  a := 4 * Real.sqrt 3
  b := 4 * Real.sqrt 2
  B := 45 * Real.pi / 180  -- Convert to radians
  C := 75 * Real.pi / 180  -- Convert to radians
  c := 2 * Real.sqrt 2 + 2 * Real.sqrt 6

-- Theorem statement
theorem triangle_abc_solution (t : Triangle) (h1 : t.A = given_triangle.A)
    (h2 : t.a = given_triangle.a) (h3 : t.b = given_triangle.b) :
    t.B = given_triangle.B ∧ t.C = given_triangle.C ∧ t.c = given_triangle.c := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_solution_l614_61453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_result_l614_61469

/-- Scaling transformation T -/
def T : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (4 * x, 3 * y)

/-- Original curve -/
def original_curve (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Transformed curve -/
def transformed_curve (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1

/-- Original line -/
def original_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- Transformed line -/
def transformed_line (x y : ℝ) : Prop := 3*x - 8*y + 12 = 0

theorem scaling_transformation_result :
  (∀ x y, original_curve x y ↔ transformed_curve ((T (x, y)).1) ((T (x, y)).2)) →
  (∀ x y, original_line x y → transformed_line ((T (x, y)).1) ((T (x, y)).2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_result_l614_61469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l614_61412

/-- Given a square with vertices (-a, -a), (a, -a), (-a, a), (a, a) cut by the line y = 2x,
    the perimeter of one of the resulting quadrilaterals divided by a equals 6 + 2√5. -/
theorem square_cut_perimeter (a : ℝ) (h : a > 0) :
  let square_vertices := [(- a, - a), (a, - a), (- a, a), (a, a)]
  let cutting_line := fun x : ℝ => 2 * x
  let intersection_points := [(- a, cutting_line (- a)), (a, cutting_line a)]
  let quadrilateral_vertices := [(- a, - a), (a, - a), (a, cutting_line a), (- a, cutting_line (- a))]
  let perimeter := ((a - (-a)) + (cutting_line a - (-a)) + 
                    Real.sqrt ((2*a)^2 + (4*a)^2) + ((-a) - cutting_line (-a)))
  perimeter / a = 6 + 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_perimeter_l614_61412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l614_61472

theorem count_integers_in_range : 
  (Finset.range 38 \ Finset.range 11).card = 27 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l614_61472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l614_61448

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + Real.sin x

/-- The translated function -/
noncomputable def f_translated (θ : ℝ) (x : ℝ) : ℝ := f (x - θ)

/-- Symmetry condition about y-axis -/
def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

theorem min_translation_for_symmetry :
  ∀ θ : ℝ, θ > 0 →
  is_symmetric_about_y_axis (f_translated θ) →
  ∀ φ : ℝ, φ > 0 → is_symmetric_about_y_axis (f_translated φ) →
  θ ≤ φ →
  θ = 5 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_symmetry_l614_61448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_triple_l614_61440

theorem unique_integer_triple : ∀ t a b : ℕ,
  t > 0 → a > 0 → b > 0 →
  (((t^(a+b) : ℕ) + 1) / ((t^a : ℕ) + (t^b : ℕ) + 1) : ℚ).isInt →
  t = 2 ∧ a = 1 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_triple_l614_61440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l614_61498

/-- Represents a geometric sequence -/
noncomputable def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def GeometricSum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_8 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  q > 1 →
  a 1 + a 4 = 18 →
  a 2 * a 3 = 32 →
  GeometricSum a q 8 = 510 := by
  sorry

#check geometric_sequence_sum_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l614_61498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_n_value_l614_61490

def quadratic_equation (x : ℝ) : Prop :=
  3 * x^2 - 4 * x - 7 = 0

def root_form (x m n p : ℝ) : Prop :=
  x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

def are_integers (m n p : ℝ) : Prop :=
  ∃ (mi ni pi : ℤ), m = ↑mi ∧ n = ↑ni ∧ p = ↑pi

def are_relatively_prime (m n p : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs m) (Int.natAbs n)) (Int.natAbs p) = 1

theorem quadratic_roots_n_value :
  ∃ (m n p : ℝ),
    (∀ x, quadratic_equation x → root_form x m n p) ∧
    are_integers m n p ∧
    (∀ (mi ni pi : ℤ), m = ↑mi → n = ↑ni → p = ↑pi → are_relatively_prime mi ni pi) →
    n = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_n_value_l614_61490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_half_pi_l614_61480

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

-- State the theorem
theorem enclosed_area_half_pi :
  (∫ x in Set.Icc (-1) 1, f x - Real.sqrt (1 - (f x)^2)) = π / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_half_pi_l614_61480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_know_everyone_l614_61406

/-- A type representing a person in the gathering -/
def Person : Type := Fin 1982

/-- The total number of people in the gathering -/
def total_people : ℕ := 1982

/-- A function that returns true if person a knows person b -/
def knows (a b : Person) : Prop := sorry

/-- The property that among any 4 people, at least 1 knows the other 3 -/
axiom four_person_property :
  ∀ (a b c d : Person),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (knows a b ∧ knows a c ∧ knows a d) ∨
    (knows b a ∧ knows b c ∧ knows b d) ∨
    (knows c a ∧ knows c b ∧ knows c d) ∨
    (knows d a ∧ knows d b ∧ knows d c)

/-- A person who knows everyone -/
def knows_everyone (p : Person) : Prop :=
  ∀ (q : Person), p ≠ q → knows p q

/-- The theorem to be proved -/
theorem min_people_know_everyone :
  ∃ (S : Finset Person),
    (∀ (p : Person), p ∈ S ↔ knows_everyone p) ∧
    S.card = 1979 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_know_everyone_l614_61406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_correct_negation_correct_l614_61474

-- Define the original proposition
def original_prop (a b : ℝ) : Prop := a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1

-- Define the contrapositive
def contrapositive (a b : ℝ) : Prop := a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1

-- Define the universal proposition
def universal_prop : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- Define the negation of the universal proposition
def negation_prop : Prop := ∃ x : ℝ, Real.sin x > 1

-- Theorem stating that the contrapositive is correct
theorem contrapositive_correct :
  ∀ a b : ℝ, (original_prop a b ↔ contrapositive a b) := by
  sorry

-- Theorem stating that the negation is correct
theorem negation_correct :
  ¬universal_prop ↔ negation_prop := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_correct_negation_correct_l614_61474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quintic_root_sum_l614_61435

/-- The quintic polynomial equation -/
def quintic_equation (x : ℝ) : Prop :=
  16 * x^5 - 4 * x^4 - 4 * x - 1 = 0

/-- The form of the real root -/
def root_form (a b c : ℕ+) (x : ℝ) : Prop :=
  x = (((a : ℝ)^(1/5 : ℝ) + (b : ℝ)^(1/5 : ℝ) + 1) / (c : ℝ))

/-- The theorem stating the sum of a, b, and c -/
theorem quintic_root_sum (a b c : ℕ+) (x : ℝ) :
  quintic_equation x → root_form a b c x → a + b + c = 69648 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quintic_root_sum_l614_61435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equilateral_triangle_l614_61422

/-- Three parallel lines in a plane -/
structure ParallelLines where
  line1 : Set (ℝ × ℝ)
  line2 : Set (ℝ × ℝ)
  line3 : Set (ℝ × ℝ)
  parallel : (∀ x y, x ∈ line1 → y ∈ line2 → (x.1 - y.1) * (x.2 - y.2) = 0) ∧
             (∀ x y, x ∈ line2 → y ∈ line3 → (x.1 - y.1) * (x.2 - y.2) = 0) ∧
             (∀ x y, x ∈ line1 → y ∈ line3 → (x.1 - y.1) * (x.2 - y.2) = 0)

/-- An equilateral triangle -/
structure EquilateralTriangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ
  eq_sides : dist p1 p2 = dist p2 p3 ∧ dist p2 p3 = dist p3 p1

/-- Given three parallel lines, there exist three points, one on each line,
    which form an equilateral triangle -/
theorem parallel_lines_equilateral_triangle (pl : ParallelLines) :
  ∃ (t : EquilateralTriangle), t.p1 ∈ pl.line1 ∧ t.p2 ∈ pl.line2 ∧ t.p3 ∈ pl.line3 :=
by
  sorry

#check parallel_lines_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equilateral_triangle_l614_61422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l614_61402

theorem tangent_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 25)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 30) :
  Real.tan (x + y) = 150 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_l614_61402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l614_61450

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_equivalence :
  ∀ x : ℝ, (4 * (floor x)^2 - 12 * (floor x) + 5 ≤ 0) ↔ (1 ≤ x ∧ x < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_equivalence_l614_61450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_PA_PB_product_l614_61407

-- Define the parametric equations for line l
noncomputable def line_l (t θ : ℝ) : ℝ × ℝ := (1 + t * Real.cos θ, t * Real.sin θ)

-- Define the parametric equations for curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define point P
def point_P : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem range_of_PA_PB_product (θ : ℝ) :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ α₁ α₂ : ℝ),
    A = line_l t₁ θ ∧ 
    B = line_l t₂ θ ∧
    A = curve_C α₁ ∧
    B = curve_C α₂ ∧
    2/3 ≤ (dist point_P A) * (dist point_P B) ∧
    (dist point_P A) * (dist point_P B) ≤ 2 := by
  sorry

#check range_of_PA_PB_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_PA_PB_product_l614_61407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l614_61465

/-- Represents the speed of a train in km/hr -/
noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  (2 * train_length * 60) / (1000 * crossing_time)

/-- Theorem stating that a train of length 1500 meters crossing a platform
    of equal length in one minute has a speed of 180 km/hr -/
theorem train_speed_calculation :
  train_speed 1500 1 = 180 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l614_61465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_usage_l614_61481

/-- Represents a fraction of a roll of wrapping paper -/
structure WrappingPaperFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- The amount of wrapping paper used for all presents -/
def total_used : WrappingPaperFraction := {
  numerator := 2,
  denominator := 5,
  denominator_pos := by norm_num
}

/-- The number of presents wrapped -/
def num_presents : ℕ := 5

theorem wrapping_paper_usage :
  ∃ (paper_per_present remaining : WrappingPaperFraction),
    /- Each present uses 2/25 of a roll -/
    paper_per_present.numerator = 2 ∧ paper_per_present.denominator = 25 ∧
    /- 3/5 of the roll remains -/
    remaining.numerator = 3 ∧ remaining.denominator = 5 ∧
    /- The sum of paper used and remaining is a whole roll -/
    (num_presents * ((paper_per_present.numerator : ℚ) / paper_per_present.denominator) +
     (remaining.numerator : ℚ) / remaining.denominator = 1) :=
by
  -- Define paper_per_present
  let paper_per_present : WrappingPaperFraction := {
    numerator := 2,
    denominator := 25,
    denominator_pos := by norm_num
  }
  -- Define remaining
  let remaining : WrappingPaperFraction := {
    numerator := 3,
    denominator := 5,
    denominator_pos := by norm_num
  }
  -- Prove the existence
  exists paper_per_present, remaining
  -- Prove the conditions
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  -- Prove the sum equals 1
  -- This step would require actual calculation, which we'll skip for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrapping_paper_usage_l614_61481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l614_61424

theorem larger_circle_radius (r : ℝ) (R : ℝ) : 
  r = 2 → -- radius of smaller circles
  -- conditions for tangency and arrangement are implicit in the geometry
  R = r * (1 + Real.sqrt 3) → -- radius of larger circle
  R = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_radius_l614_61424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_speed_theorem_l614_61476

/-- The speed of a plane in still air, given its distances traveled with and against wind -/
noncomputable def plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) : ℝ :=
  let p := (distance_with_wind * (distance_against_wind + wind_speed * distance_with_wind)) /
            (distance_with_wind - distance_against_wind)
  p - wind_speed

/-- Theorem stating that the plane's speed in still air is 253 mph -/
theorem plane_speed_theorem :
  plane_speed 420 350 23 = 253 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_speed_theorem_l614_61476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_guests_acquainted_l614_61470

-- Define the type for guests
variable {Guest : Type}

-- Define the relation for acquaintance
variable (acquainted : Guest → Guest → Prop)

-- Define the property of sitting at a round table
variable (sits_at_round_table : Set Guest → Prop)

-- Define the property of equal intervals for acquaintances
variable (equal_intervals : Guest → Set Guest → Prop)

-- Define the theorem
theorem all_guests_acquainted
  (guests : Set Guest)
  (h_round_table : sits_at_round_table guests)
  (h_mutual : ∀ a b, a ∈ guests → b ∈ guests → acquainted a b → acquainted b a)
  (h_equal_intervals : ∀ g, g ∈ guests → equal_intervals g {x | x ∈ guests ∧ acquainted g x})
  (h_common_acquaintance : ∀ a b, a ∈ guests → b ∈ guests → 
    ∃ c, c ∈ guests ∧ acquainted a c ∧ acquainted b c) :
  ∀ a b, a ∈ guests → b ∈ guests → acquainted a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_guests_acquainted_l614_61470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l614_61428

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the slope angle of line l
noncomputable def slope_angle : ℝ := 2 * Real.pi / 3

-- Define the circle equation
noncomputable def circle_equation (θ : ℝ) : ℝ := 2 * Real.cos (θ + Real.pi / 3)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (P.1 + t * Real.cos slope_angle, P.2 + t * Real.sin slope_angle)

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_product : 
  let PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let PN := Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2)
  PM * PN = 6 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l614_61428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_powers_l614_61447

theorem divisibility_of_powers (a p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_powers_l614_61447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l614_61425

/-- A parabola with equation y^2 = 8x -/
structure Parabola where
  equation : ∀ x y, y^2 = 8*x

/-- The focus of a parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- The distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem stating the relationship between the distance to y-axis and the distance to focus -/
theorem distance_to_focus (p : Parabola) (m : PointOnParabola p) 
  (h : m.x = 3) : 
  distance (m.x, m.y) focus = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l614_61425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_equals_third_l614_61491

/-- Given natural numbers a, b, c, d that are pairwise coprime and satisfy
    ab + cd = ac - 10bd, there exist x, y, z among these numbers such that x + y = z. -/
theorem sum_of_two_equals_third (a b c d : ℕ) 
  (coprime_ab : Nat.Coprime a b)
  (coprime_ac : Nat.Coprime a c)
  (coprime_ad : Nat.Coprime a d)
  (coprime_bc : Nat.Coprime b c)
  (coprime_bd : Nat.Coprime b d)
  (coprime_cd : Nat.Coprime c d)
  (h : a * b + c * d = a * c - 10 * b * d) :
  ∃ x y z, x ∈ ({a, b, c, d} : Set ℕ) ∧ y ∈ ({a, b, c, d} : Set ℕ) ∧ z ∈ ({a, b, c, d} : Set ℕ) ∧ x + y = z :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_two_equals_third_l614_61491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l614_61494

/-- Given vectors AB, AC, and BC in ℝ², prove that m + n = 8 -/
theorem vector_sum (m n : ℝ) : 
  let AB : Fin 2 → ℝ := ![m, 5]
  let AC : Fin 2 → ℝ := ![4, n]
  let BC : Fin 2 → ℝ := ![7, 6]
  (∀ i : Fin 2, BC i = AC i - AB i) →
  m + n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_l614_61494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l614_61427

/-- Given a line ax + by = 1 intersecting a circle x^2 + y^2 = 1, 
    prove that the point P(a, b) lies outside the circle. -/
theorem point_outside_circle (a b : ℝ) : 
  (∃ x y : ℝ, a*x + b*y = 1 ∧ x^2 + y^2 = 1) →  -- line intersects circle
  a^2 + b^2 > 1  -- point (a, b) is outside the circle
:= by
  intro h
  sorry  -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l614_61427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_invariant_impossibility_theorem_possibility_theorem_l614_61492

-- Define the game rules
def game_move : ℕ → ℕ → ℕ → ℕ :=
  sorry -- Implementation details omitted for brevity

-- Define a function to calculate the sum of digits
def sum_of_digits : ℕ → ℕ :=
  sorry -- Implementation details omitted for brevity

-- Theorem 1: The sum of digits remains constant
theorem sum_of_digits_invariant (n i j : ℕ) :
  sum_of_digits n = sum_of_digits (game_move n i j) :=
by sorry

-- Theorem 2: Impossibility of transforming 324561 to 434434
theorem impossibility_theorem :
  ¬∃ (moves : List (ℕ × ℕ)), moves.foldl (λ acc (i, j) => game_move acc i j) 324561 = 434434 :=
by sorry

-- Theorem 3: Possibility of obtaining a number > 800000000 from 123456789
theorem possibility_theorem :
  ∃ (moves : List (ℕ × ℕ)), 
    moves.foldl (λ acc (i, j) => game_move acc i j) 123456789 > 800000000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_invariant_impossibility_theorem_possibility_theorem_l614_61492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l614_61483

/-- A point in R^2 -/
def Point := ℝ × ℝ

/-- The Euclidean distance between two points in R^2 -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the existence of a set A with the required properties -/
theorem exists_special_set :
  ∃ (A : Set Point), A.Finite ∧
    ∀ X ∈ A, ∃ (Y : Fin 1993 → Point),
      (∀ i, Y i ∈ A) ∧
      (∀ i, distance X (Y i) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_set_l614_61483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shifted_through_origin_l614_61408

/-- Represents a parabola of the form y = x^2 + kx - k^2 -/
structure Parabola where
  k : ℝ

/-- The axis of symmetry of the parabola -/
noncomputable def Parabola.axisOfSymmetry (p : Parabola) : ℝ := -p.k / 2

/-- The condition that the axis of symmetry is to the right of the y-axis -/
def Parabola.axisToRight (p : Parabola) : Prop := p.axisOfSymmetry > 0

/-- The shifted parabola equation -/
def Parabola.shiftedEquation (p : Parabola) (x y : ℝ) : Prop :=
  y = (x - 3 + p.k/2)^2 - (5*p.k^2)/4 + 1

/-- The main theorem -/
theorem parabola_shifted_through_origin (p : Parabola) :
  p.axisToRight →
  p.shiftedEquation 0 0 →
  p.k = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_shifted_through_origin_l614_61408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_product_l614_61449

/-- Given a line passing through M(-2,0) with slope K₁ ≠ 0 intersecting the hyperbola x²/4 - y² = 1
    at points A and B, with P as the midpoint of AB, and K₂ as the slope of OP where O is the origin,
    prove that K₁K₂ = 1/4 -/
theorem intersection_slope_product (K₁ K₂ : ℝ) (A B P : ℝ × ℝ) (h₁ : K₁ ≠ 0) :
  let M : ℝ × ℝ := (-2, 0)
  let O : ℝ × ℝ := (0, 0)
  let hyperbola := λ (x y : ℝ) => x^2/4 - y^2 = 1
  (∃ t : ℝ, A.1 = M.1 + t * 1 ∧ A.2 = M.2 + t * K₁) →
  (∃ s : ℝ, B.1 = M.1 + s * 1 ∧ B.2 = M.2 + s * K₁) →
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →
  K₂ = (P.2 - O.2) / (P.1 - O.1) →
  K₁ * K₂ = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_product_l614_61449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_path_count_l614_61438

/-- Represents a grid with dimensions height and width -/
structure Grid where
  height : Nat
  width : Nat

/-- Represents a path on the grid -/
structure GridPath (g : Grid) where
  steps : Nat
  step_size : Nat

/-- Calculates the number of distinct paths on a grid -/
def count_paths (g : Grid) (p : GridPath g) : Nat :=
  Nat.choose (g.height + g.width) g.width

/-- Theorem statement for the domino path problem -/
theorem domino_path_count (g : Grid) (p : GridPath g) 
  (h1 : g.height = 7) 
  (h2 : g.width = 5) 
  (h3 : p.steps = 6) 
  (h4 : p.step_size = 2) :
  count_paths g p = 792 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domino_path_count_l614_61438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l614_61401

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_intersection_distance 
  (A B : PointOnParabola) 
  (h_line : ∃ (m b : ℝ), A.y = m * A.x + b ∧ B.y = m * B.x + b ∧ 0 = m * 1 + b) 
  (h_AF_distance : distance (A.x, A.y) focus = 2) :
  distance (B.x, B.y) focus = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l614_61401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l614_61461

/-- The time required for A and B together to complete the job -/
def time_AB : ℝ := 3

/-- The time required for B and C together to complete the job -/
def time_BC : ℝ := 6

/-- The time required for A, B, and C together to complete the job -/
def time_ABC : ℝ := 2

/-- The time required for A alone to complete the job -/
def time_A : ℝ := 3

/-- The daily work rate of A -/
noncomputable def rate_A : ℝ := 1 / time_A

/-- The daily work rate of B -/
noncomputable def rate_B : ℝ := 1 / time_AB - rate_A

/-- The daily work rate of C -/
noncomputable def rate_C : ℝ := 1 / time_ABC - rate_A - rate_B

theorem job_completion_time :
  rate_A + rate_B = 1 / time_AB ∧
  rate_B + rate_C = 1 / time_BC ∧
  rate_A + rate_B + rate_C = 1 / time_ABC ∧
  time_A = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l614_61461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_simplification_l614_61436

theorem sqrt_simplification :
  (Real.sqrt 8 = 2 * Real.sqrt 2) ∧
  (Real.sqrt (3/2) = (Real.sqrt 6) / 2) ∧
  ((2 * Real.sqrt 3)^2 = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_simplification_l614_61436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelogram_law_l614_61413

/-- Given four points in a plane, prove that AC + BD = BC + AD -/
theorem vector_parallelogram_law {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]
  (A B C D : n) :
  (C - A) + (D - B) = (C - B) + (D - A) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallelogram_law_l614_61413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l614_61455

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 2√3/3 under the given conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  c = 2 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l614_61455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_monotonicity_l614_61404

noncomputable def S (t : ℝ) : ℝ :=
  if 0 < t ∧ t < 1/2 then 2 * (1 - t + t^2 - t^3)
  else if t ≥ 1/2 then 1/2 * (t + 1/t)
  else 0  -- undefined for t ≤ 0

noncomputable def S' (t : ℝ) : ℝ :=
  if 0 < t ∧ t < 1/2 then 2 * (-1 + 2*t - 3*t^2)
  else if t ≥ 1/2 then 1/2 * (1 - 1/t^2)
  else 0  -- undefined for t ≤ 0

theorem S_monotonicity :
  (∀ t ∈ Set.Ioo 0 1, HasDerivAt S (S' t) t ∧ S' t < 0) ∧
  (∀ t ∈ Set.Ici 1, HasDerivAt S (S' t) t ∧ S' t > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_monotonicity_l614_61404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l614_61429

/-- The volume of the solid formed by rotating the region bounded by y = ax² and y = bx about the line y = bx -/
noncomputable def revolution_volume (a b : ℝ) : ℝ :=
  Real.pi * b^5 / (30 * a^3 * Real.sqrt (b^2 + 1))

/-- Theorem stating the volume of the described solid of revolution -/
theorem volume_of_revolution (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  revolution_volume a b = Real.pi * b^5 / (30 * a^3 * Real.sqrt (b^2 + 1)) :=
by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_l614_61429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_product_l614_61464

noncomputable section

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x : ℝ) : ℝ := p * x - x^2
def hyperbola (q : ℝ) (x : ℝ) : ℝ := q / x

-- Define the intersection points
structure IntersectionPoint (p q : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola p x
  on_hyperbola : x * y = q

-- Define the theorem
theorem intersection_points_product (p q : ℝ) 
  (A B C : IntersectionPoint p q)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (sum_of_squares : 
    (A.x - B.x)^2 + (B.x - C.x)^2 + (C.x - A.x)^2 +
    (A.y - B.y)^2 + (B.y - C.y)^2 + (C.y - A.y)^2 = 324)
  (centroid_distance : ((A.x + B.x + C.x) / 3)^2 + ((A.y + B.y + C.y) / 3)^2 = 4) :
  p * q = 42 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_product_l614_61464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l614_61441

noncomputable def vector_a : ℝ × ℝ := (-3, 2)
noncomputable def vector_b : ℝ × ℝ := (4, -1)

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let magnitude_squared := v.1 * v.1 + v.2 * v.2
  ((dot_product / magnitude_squared) * v.1, (dot_product / magnitude_squared) * v.2)

theorem projection_equality (v : ℝ × ℝ) :
  projection vector_a v = projection vector_b v →
  projection vector_a v = (15/58, 35/58) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l614_61441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equations_solutions_l614_61473

theorem trigonometric_equations_solutions :
  (∀ x : ℝ, Real.sin x + Real.sin (2*x) + Real.sin (3*x) = 0 ↔
    (∃ k : ℤ, x = k * Real.pi / 2) ∨
    (∃ n : ℤ, x = 2*Real.pi/3 + n * 2*Real.pi) ∨
    (∃ n : ℤ, x = 4*Real.pi/3 + n * 2*Real.pi)) ∧
  (∀ x : ℝ, Real.cos x + Real.cos (2*x) + Real.cos (3*x) = 0 ↔
    (∃ n : ℤ, x = Real.pi/4 + n * Real.pi/2) ∨
    (∃ n : ℤ, x = 2*Real.pi/3 + n * 2*Real.pi) ∨
    (∃ n : ℤ, x = 4*Real.pi/3 + n * 2*Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equations_solutions_l614_61473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l614_61415

-- Define the quadrilateral ABCD
noncomputable def AB : ℝ × ℝ := (6, 1)
noncomputable def BC (x y : ℝ) : ℝ × ℝ := (x, y)
noncomputable def CD : ℝ × ℝ := (-2, -3)

-- Define the parallel and perpendicular conditions
def BC_parallel_DA (x y : ℝ) : Prop :=
  x * (-y + 2) - y * (-x - 4) = 0

def AC_perpendicular_BD (x y : ℝ) : Prop :=
  (x + 6) * (x - 2) + (y + 1) * (y - 3) = 0

-- Define the area of the quadrilateral
noncomputable def area_ABCD (x y : ℝ) : ℝ :=
  abs ((x + 6) * (y - 3) - (y + 1) * (x - 2)) / 2

-- State the theorem
theorem quadrilateral_properties (x y : ℝ) :
  BC_parallel_DA x y → AC_perpendicular_BD x y →
  (x + 2 * y = 0) ∧
  ((x = -6 ∧ y = 3) ∨ (x = 2 ∧ y = -1)) ∧
  area_ABCD x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_properties_l614_61415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_non_parallel_line_planes_l614_61411

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (notParallel : Line → Line → Prop)
variable (notSubset : Line → Plane → Prop)
variable (notParallelPlane : Line → Plane → Prop)

-- Theorem for proposition ①
theorem perpendicular_planes 
  (α β : Plane) (m : Line) 
  (h1 : perpendicular m α) 
  (h2 : subset m β) : 
  perpendicularPlanes α β :=
sorry

-- Theorem for proposition ④
theorem non_parallel_line_planes 
  (α β : Plane) (m n : Line)
  (h1 : intersect α β m)
  (h2 : notParallel n m)
  (h3 : notSubset n α)
  (h4 : notSubset n β) :
  notParallelPlane n α ∧ notParallelPlane n β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_non_parallel_line_planes_l614_61411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l614_61431

-- Define the circle C
noncomputable def circle_C (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

-- Define the polar equation of line l
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi/3) = 3 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop :=
  θ = Real.pi/3

-- Define the point P on circle C
noncomputable def point_P : ℝ × ℝ :=
  (1, Real.sqrt 3 / 2)

-- Define the point Q on line l
noncomputable def point_Q : ℝ × ℝ :=
  (3 * Real.cos (Real.pi/3), 3 * Real.sin (Real.pi/3))

-- Theorem statement
theorem length_PQ_is_two :
  let p := point_P
  let q := point_Q
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_two_l614_61431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l614_61487

/-- A sequence a_n is monotonically decreasing if a_(n+1) < a_n for all n -/
def MonotonicallyDecreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) < a n

/-- The sequence a_n defined as -n^3 + λ(n^2 + n) -/
def a (lambda : ℝ) (n : ℕ) : ℝ :=
  -n^3 + lambda * (n^2 + n)

theorem lambda_range :
  ∀ lambda : ℝ, (MonotonicallyDecreasing (a lambda)) ↔ lambda < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l614_61487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l614_61484

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * (2 : ℝ)^x + 3

-- State the theorem
theorem range_of_f :
  let domain := Set.Icc (-1 : ℝ) 2
  ∃ (range : Set ℝ), range = Set.Icc (9/2) 15 ∧
    (∀ y ∈ range, ∃ x ∈ domain, f x = y) ∧
    (∀ x ∈ domain, f x ∈ range) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l614_61484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_a_cubic_equation_b_cubic_equation_c_l614_61409

theorem cubic_equation_a (x : ℝ) :
  x^3 - 3*x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
sorry

theorem cubic_equation_b (x : ℝ) :
  x^3 - 19*x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
sorry

theorem cubic_equation_c (x : ℝ) :
  x^3 + 4*x^2 + 6*x + 4 = 0 ↔ x = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_a_cubic_equation_b_cubic_equation_c_l614_61409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l614_61460

theorem complex_inequality (a b c : ℂ) (h : ℕ+) (k l m : ℤ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → (abs k + abs l + abs m : ℝ) ≥ 2007 →
  Complex.abs (↑k * a + ↑l * b + ↑m * c) > (1 : ℝ) / h.val := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l614_61460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_on_diagonal_l614_61489

/-- Represents a square table filled with numbers -/
def SquareTable (n : ℕ) := Fin n → Fin n → Fin n

/-- Predicate to check if a table is valid according to the problem conditions -/
def is_valid_table (n : ℕ) (table : SquareTable n) : Prop :=
  (∀ i j : Fin n, (table i j).val < n) ∧
  (∀ i : Fin n, Function.Injective (table i)) ∧
  (∀ j : Fin n, Function.Injective (λ i => table i j))

/-- Predicate to check if a table is symmetric with respect to the main diagonal -/
def is_symmetric (n : ℕ) (table : SquareTable n) : Prop :=
  ∀ i j : Fin n, table i j = table j i

/-- Theorem statement -/
theorem all_numbers_on_diagonal
  (n : ℕ) (hn : Odd n) (table : SquareTable n)
  (hvalid : is_valid_table n table) (hsym : is_symmetric n table) :
  ∀ k : Fin n, ∃ i : Fin n, table i i = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_numbers_on_diagonal_l614_61489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_none_of_conditions_false_impossible_l614_61414

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circleA : Circle := { center := (0, 0), radius := 8 }
def circleB : Circle := { center := (0, 0), radius := 5 }

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating all conditions are possible
theorem all_conditions_possible : 
  ∃ (x y : ℝ), 
    (distance circleA.center (x, y) = 3) ∧
    (∃ (x' y' : ℝ), distance circleA.center (x', y') = 13) ∧
    (∃ (x'' y'' : ℝ), distance circleA.center (x'', y'') > 13) ∧
    (distance circleA.center (x, y) > 3) :=
by
  sorry

-- Theorem stating that none of the conditions being false is impossible
theorem none_of_conditions_false_impossible :
  ¬(∀ (x y : ℝ),
    (distance circleA.center (x, y) ≠ 3) ∧
    (distance circleA.center (x, y) ≠ 13) ∧
    (distance circleA.center (x, y) ≤ 13) ∧
    (distance circleA.center (x, y) ≤ 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_none_of_conditions_false_impossible_l614_61414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_P_l614_61454

-- Define the set of natural numbers including 0
def ℕ₀ : Set ℕ := {n : ℕ | n ≥ 0}

-- Define set M
def M : Set ℕ := {x ∈ ℕ₀ | (x - 1)^2 < 4}

-- Define set P as a set of naturals
def P : Set ℕ := {0, 1, 2, 3}

-- Theorem statement
theorem intersection_M_P : M ∩ P = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_P_l614_61454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_zero_l614_61462

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else 3 * x - 18

-- Theorem statement
theorem solutions_of_f_eq_zero :
  {x : ℝ | f x = 0} = {-2, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_zero_l614_61462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emmas_missing_coins_l614_61405

theorem emmas_missing_coins (x : ℚ) (h : x > 0) : 
  (x - ((2/3 : ℚ) * x + (1/4 : ℚ) * x - (3/5 : ℚ) * ((2/3 : ℚ) * x + (1/4 : ℚ) * x) + 
  (1/2 : ℚ) * ((3/5 : ℚ) * ((2/3 : ℚ) * x + (1/4 : ℚ) * x)))) / x = 29/40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emmas_missing_coins_l614_61405
