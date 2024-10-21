import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44345

-- Define the function f(x) = x + 4/x
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Define the interval [1, 2]
def I : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem f_properties :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x > f y) ∧
  (∀ x, x ∈ I → f x ≤ 5) ∧
  (∀ x, x ∈ I → f x ≥ 4) ∧
  (∃ x, x ∈ I ∧ f x = 5) ∧
  (∃ x, x ∈ I ∧ f x = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_5_or_6_l443_44329

-- Define a function to check if a number in base 8 contains 5 or 6
def contains_5_or_6_base_8 (n : ℕ) : Bool :=
  let rec aux (m : ℕ) : Bool :=
    if m = 0 then false
    else 
      let digit := m % 8
      if digit = 5 || digit = 6 then true
      else aux (m / 8)
  aux n

-- Define the set of numbers from 1 to 512 that contain 5 or 6 in base 8
def numbers_with_5_or_6 : Finset ℕ :=
  Finset.filter (λ n => contains_5_or_6_base_8 n) (Finset.range 512)

-- State the theorem
theorem count_numbers_with_5_or_6 :
  Finset.card numbers_with_5_or_6 = 296 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_5_or_6_l443_44329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edges_after_cuts_modifiedCubeHas12Edges_l443_44373

/-- Represents a polyhedron --/
structure Polyhedron :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

/-- A cube is a polyhedron with 8 vertices, 12 edges, and 6 faces --/
def Cube : Polyhedron :=
  { vertices := 8, edges := 12, faces := 6 }

/-- Function that represents the operation of cutting off all vertices of a polyhedron --/
def cutAllVertices (p : Polyhedron) : Polyhedron :=
  { vertices := p.vertices, edges := p.edges, faces := p.faces + p.vertices }

/-- Theorem stating that cutting all vertices of a cube doesn't change the number of edges --/
theorem cube_edges_after_cuts (c : Polyhedron) (h : c = Cube) : 
  (cutAllVertices c).edges = c.edges := by
  rw [h]
  rfl

/-- The number of edges in the modified cube --/
def modifiedCubeEdges : ℕ := (cutAllVertices Cube).edges

/-- Proof that the modified cube has 12 edges --/
theorem modifiedCubeHas12Edges : modifiedCubeEdges = 12 := by
  rfl

#eval modifiedCubeEdges -- This will output 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edges_after_cuts_modifiedCubeHas12Edges_l443_44373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_elements_in_set_l443_44387

theorem existence_of_divisible_elements_in_set (p : Nat) (hp : Prime p) :
  ∃ (a b : Int), a ∈ {x : Int | ∃ (n : Nat), x = p - n^2 ∧ n^2 < p} ∧
                 b ∈ {x : Int | ∃ (n : Nat), x = p - n^2 ∧ n^2 < p} ∧
                 a > 1 ∧ a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_elements_in_set_l443_44387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l443_44391

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point :=
  ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For a parabola y^2 = 4x, if a line passing through the focus
    intersects the parabola at points A and B, and the x-coordinate of
    the midpoint of AB is 2, then |AF| + |BF| = 6 -/
theorem parabola_intersection_distance (A B : Point)
  (h1 : A ∈ Parabola)
  (h2 : B ∈ Parabola)
  (h3 : ∃ (t : ℝ), A = ⟨focus.x + t * (A.x - focus.x), focus.y + t * (A.y - focus.y)⟩)
  (h4 : ∃ (s : ℝ), B = ⟨focus.x + s * (B.x - focus.x), focus.y + s * (B.y - focus.y)⟩)
  (h5 : (A.x + B.x) / 2 = 2) :
  distance A focus + distance B focus = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l443_44391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l443_44328

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := Real.log x + 1

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 0)

/-- The slope of the tangent line at P -/
noncomputable def k : ℝ := f_derivative P.1

/-- The y-intercept of the tangent line -/
noncomputable def b : ℝ := P.2 - k * P.1

/-- The x-intercept of the tangent line -/
noncomputable def x_intercept : ℝ := -b / k

/-- The y-intercept of the tangent line -/
noncomputable def y_intercept : ℝ := b

theorem tangent_line_triangle_area :
  (1/2) * x_intercept * y_intercept = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l443_44328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l443_44379

theorem circle_tangent_to_ellipse (r : ℝ) : 
  r = 2 * Real.sqrt 5 / 3 ↔ 
  ∃ (x y : ℝ), 4 * x^2 + 9 * y^2 = 36 ∧ 
               (x - r)^2 + y^2 = r^2 ∧
               (x + r)^2 + y^2 = r^2 :=
sorry

#check circle_tangent_to_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l443_44379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_is_vertical_line_l443_44372

/-- The radical axis of two circles is a vertical line. -/
theorem radical_axis_is_vertical_line 
  (x₁ x₂ r₁ r₂ : ℝ) (h : x₁ + x₂ ≠ 0) :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    ((x + x₁)^2 + y^2 - r₁^2) = ((x - x₂)^2 + y^2 - r₂^2) ↔ 
    x = (1 / (2 * (x₁ + x₂))) * (r₁^2 - r₂^2 + x₂^2 - x₁^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_is_vertical_line_l443_44372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_600_eq_24_l443_44353

/-- The number of divisors of 600 -/
def num_divisors_600 : ℕ := (Nat.divisors 600).card

/-- Theorem stating that the number of divisors of 600 is 24 -/
theorem num_divisors_600_eq_24 : num_divisors_600 = 24 := by
  -- Unfold the definition of num_divisors_600
  unfold num_divisors_600
  -- Evaluate the expression
  norm_num
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_600_eq_24_l443_44353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_product_equals_l443_44386

-- Define a structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a predicate to check if four points are collinear
def collinear (A B C D : Point) : Prop :=
  ∃ (m b : ℝ), (B.y = m * B.x + b) ∧ (C.y = m * C.x + b) ∧ (D.y = m * D.x + b)

-- Define a predicate to check if four points are concyclic
def concyclic (A B C D : Point) : Prop :=
  ∃ (center : Point) (r : ℝ), 
    distance center A = r ∧ 
    distance center B = r ∧ 
    distance center C = r ∧ 
    distance center D = r

-- Define the theorem
theorem concyclic_iff_product_equals (A B C D M : Point) :
  ¬collinear A B C D →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = Point.mk (A.x * (1 - t) + B.x * t) (A.y * (1 - t) + B.y * t)) →
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ M = Point.mk (C.x * (1 - s) + D.x * s) (C.y * (1 - s) + D.y * s)) →
  (concyclic A B C D ↔ distance M A * distance M B = distance M C * distance M D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_iff_product_equals_l443_44386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l443_44305

noncomputable def g (x : ℝ) : ℝ := Real.exp x - x + (1/2) * x^2

theorem min_value_of_m (m : ℝ) :
  (∃ (x₀ : ℝ), 2 * m - 1 ≥ g x₀) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_m_l443_44305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_correct_l443_44359

-- Define the ellipse parameters
noncomputable def a : ℝ := 10 + 2 * Real.sqrt 10
noncomputable def b : ℝ := Real.sqrt (101 + 20 * Real.sqrt 10)
def h : ℝ := 4
def k : ℝ := 0

-- Define the foci and the point on the ellipse
def focus1 : ℝ × ℝ := (1, 0)
def focus2 : ℝ × ℝ := (7, 0)
def point_on_ellipse : ℝ × ℝ := (9, 6)

-- Theorem statement
theorem ellipse_equation_correct :
  -- The equation of the ellipse is correct
  (point_on_ellipse.1 - h)^2 / a^2 + (point_on_ellipse.2 - k)^2 / b^2 = 1 ∧
  -- The sum of distances from the point to the foci equals 2a
  Real.sqrt ((point_on_ellipse.1 - focus1.1)^2 + (point_on_ellipse.2 - focus1.2)^2) +
  Real.sqrt ((point_on_ellipse.1 - focus2.1)^2 + (point_on_ellipse.2 - focus2.2)^2) = 2 * a ∧
  -- The center of the ellipse is at (h, k)
  h = (focus1.1 + focus2.1) / 2 ∧
  k = (focus1.2 + focus2.2) / 2 ∧
  -- a and b are positive
  a > 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_correct_l443_44359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_sequence_properties_l443_44321

/-- Definition of a companion sequence -/
def is_companion_sequence (n : ℕ) (A B : ℕ → ℝ) : Prop :=
  B 1 = A n ∧ 
  ∀ k, 2 ≤ k ∧ k ≤ n → B k + B (k-1) = A k + A (k-1)

/-- Helper definition for arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem companion_sequence_properties 
  (n : ℕ) (A B : ℕ → ℝ) (h : is_companion_sequence n A B) :
  (n = 9 → is_arithmetic_sequence (B 9) (A 9) (A 1)) ∧
  (Even n → B n = A 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_sequence_properties_l443_44321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cost_effective_restaurant_l443_44357

/-- Represents a restaurant with its pricing and offers -/
structure Restaurant where
  hamburger_price : ℝ
  milkshake_price : ℝ
  tax_rate : ℝ
  special_offer : ℕ → ℕ → ℝ → ℝ

/-- Calculates the total cost for a given restaurant -/
def total_cost (r : Restaurant) (hamburgers milkshakes : ℕ) : ℝ :=
  let subtotal := r.special_offer hamburgers milkshakes (hamburgers * r.hamburger_price + milkshakes * r.milkshake_price)
  subtotal * (1 + r.tax_rate)

/-- Restaurant A's special offer -/
def restaurant_a_offer (hamburgers milkshakes : ℕ) (subtotal : ℝ) : ℝ :=
  let free_hamburgers := hamburgers / 5
  subtotal - (free_hamburgers * 4 : ℝ)

/-- Restaurant B's special offer -/
def restaurant_b_offer (hamburgers milkshakes : ℕ) (subtotal : ℝ) : ℝ :=
  if milkshakes ≥ 3 then subtotal * 0.9 else subtotal

/-- Restaurant C's special offer -/
def restaurant_c_offer (hamburgers milkshakes : ℕ) (subtotal : ℝ) : ℝ :=
  let discounted_milkshakes := milkshakes / 3
  subtotal - (discounted_milkshakes * 2 : ℝ)

/-- The three restaurants -/
def restaurant_a : Restaurant := ⟨4, 5, 0.08, restaurant_a_offer⟩
def restaurant_b : Restaurant := ⟨3.5, 6, 0.06, restaurant_b_offer⟩
def restaurant_c : Restaurant := ⟨5, 4, 0.1, restaurant_c_offer⟩

/-- Annie's budget and requirements -/
def annie_budget : ℝ := 120
def hamburgers_wanted : ℕ := 8
def milkshakes_wanted : ℕ := 6

theorem most_cost_effective_restaurant :
  let costs := [total_cost restaurant_a hamburgers_wanted milkshakes_wanted,
                total_cost restaurant_b hamburgers_wanted milkshakes_wanted,
                total_cost restaurant_c hamburgers_wanted milkshakes_wanted]
  let min_cost := List.minimum? costs
  ∀ mc, min_cost = some mc →
    (mc = total_cost restaurant_b hamburgers_wanted milkshakes_wanted) ∧
    (annie_budget - mc = 58.94) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cost_effective_restaurant_l443_44357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l443_44384

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (2, 0)

-- Define the function to calculate the area of a triangle given its vertices
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem triangle_ABC_area :
  triangleArea A B C = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l443_44384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_leak_rate_is_zero_l443_44318

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  total_distance : ℝ
  total_time : ℝ
  initial_water : ℝ
  remaining_water : ℝ
  last_mile_consumption : ℝ
  first_six_miles_rate : ℝ

/-- Calculates the leak rate of the canteen per hour -/
noncomputable def calculate_leak_rate (scenario : HikeScenario) : ℝ :=
  let total_water_loss := scenario.initial_water - scenario.remaining_water
  let water_consumed := scenario.last_mile_consumption + 
                        scenario.first_six_miles_rate * (scenario.total_distance - 1)
  (total_water_loss - water_consumed) / scenario.total_time

/-- Theorem stating that the canteen leak rate is zero -/
theorem canteen_leak_rate_is_zero (scenario : HikeScenario) 
  (h1 : scenario.total_distance = 7)
  (h2 : scenario.total_time = 2)
  (h3 : scenario.initial_water = 9)
  (h4 : scenario.remaining_water = 3)
  (h5 : scenario.last_mile_consumption = 2)
  (h6 : scenario.first_six_miles_rate = 0.6666666666666666) :
  calculate_leak_rate scenario = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canteen_leak_rate_is_zero_l443_44318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l443_44342

noncomputable section

-- Define the hyperbola C
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Define the eccentricity
def Eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > a) 
  (F : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) :
  F.1 = c ∧ F.2 = 0 →  -- F is the right focus
  A.1 = a ∧ A.2 = 0 →  -- A is the right vertex
  B ∈ Hyperbola a b →  -- B is on the hyperbola
  B.1 = c →            -- BF is perpendicular to x-axis
  (B.2 - A.2) / (B.1 - A.1) = 3 →  -- Slope of AB is 3
  Eccentricity a c = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l443_44342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_real_roots_l443_44361

theorem cubic_equation_real_roots 
  (p q r s : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (h_pqrs : p * s = q * r) :
  let f : ℝ → ℝ := λ x => p * x^3 - q * x^2 - r * x + s
  (∀ x, f x = 0 → x ∈ Set.univ) ∧
  (∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ (x = y ↔ ∃ t, p = t^3 ∧ q = t^2 ∧ r = t ∧ s = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_real_roots_l443_44361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_60_l443_44308

/-- A function that generates a number with five consecutive descending digits -/
def consecutiveDescendingDigits (n : ℕ) : ℕ :=
  10000 * (n + 4) + 1000 * (n + 3) + 100 * (n + 2) + 10 * (n + 1) + n

/-- The set of possible lowest digits -/
def possibleLowestDigits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

/-- The theorem to be proved -/
theorem sum_of_remainders_is_60 :
  (possibleLowestDigits.sum fun n => (consecutiveDescendingDigits n) % 13) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_60_l443_44308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_dot_product_constant_line_equation_when_origin_dot_product_l443_44333

/-- A line passing through point A(0,1) with slope k -/
structure Line where
  k : ℝ

/-- Circle C: (x-2)^2 + (y-3)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

/-- Points where the line intersects the circle -/
def IntersectionPoints (l : Line) : Set (ℝ × ℝ) :=
  {p | p ∈ Circle ∧ p.2 = l.k * p.1 + 1}

theorem line_circle_intersection (l : Line) :
  (∃ M N, M ∈ IntersectionPoints l ∧ N ∈ IntersectionPoints l ∧ M ≠ N) ↔
    (4 - Real.sqrt 7) / 3 < l.k ∧ l.k < (4 + Real.sqrt 7) / 3 :=
sorry

theorem dot_product_constant (l : Line) (M N : ℝ × ℝ) 
    (hM : M ∈ IntersectionPoints l) (hN : N ∈ IntersectionPoints l) (hMN : M ≠ N) :
  (M.1 - 0) * (N.1 - 0) + (M.2 - 1) * (N.2 - 1) = 7 :=
sorry

theorem line_equation_when_origin_dot_product (l : Line) (M N : ℝ × ℝ) 
    (hM : M ∈ IntersectionPoints l) (hN : N ∈ IntersectionPoints l) (hMN : M ≠ N) :
  M.1 * N.1 + M.2 * N.2 = 12 → l.k = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_dot_product_constant_line_equation_when_origin_dot_product_l443_44333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l443_44336

noncomputable def f (ω φ x : ℝ) := Real.sqrt 3 * Real.sin (ω * x + φ) + 2 * Real.sin ((ω * x + φ) / 2) ^ 2 - 1

noncomputable def g (x : ℝ) := 2 * Real.sin (4 * x - Real.pi / 3)

theorem function_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  -- f is an odd function
  (∀ x, f ω φ (-x) = -f ω φ x) ∧
  -- Distance between adjacent axes of symmetry is π/2
  (∀ x, f ω φ (x + Real.pi / 2) = f ω φ x) ∧
  -- f simplifies to 2 * sin(2x)
  (∀ x, f ω φ x = 2 * Real.sin (2 * x)) ∧
  -- f is decreasing on [-π/2, π/4]
  (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 4), ∀ y ∈ Set.Icc (-Real.pi / 2) (Real.pi / 4), x < y → f ω φ y < f ω φ x) ∧
  -- g has the specified range on [-π/12, π/6]
  (∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 6), -2 ≤ g x ∧ g x ≤ Real.sqrt 3) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l443_44336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l443_44317

theorem angle_in_second_quadrant (α : ℝ) 
  (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
  α ∈ Set.Ioo (Real.pi / 2) Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l443_44317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_problem_l443_44334

/-- Fruit shop problem -/
theorem fruit_shop_problem 
  (peach_quantity : ℕ) 
  (apple_quantity : ℕ) 
  (peach_sell_price : ℝ) 
  (apple_sell_price : ℝ) 
  (total_profit : ℝ) 
  (new_peach_quantity : ℕ) 
  (new_peach_sell_price : ℝ) 
  (a : ℝ) 
  (price_reduction : ℝ) 
  (unsold_peaches : ℕ) 
  (new_batch_profit : ℝ)
  (h1 : peach_quantity = 100)
  (h2 : apple_quantity = 50)
  (h3 : peach_sell_price = 16)
  (h4 : apple_sell_price = 20)
  (h5 : total_profit = 1800)
  (h6 : new_peach_quantity = 300)
  (h7 : new_peach_sell_price = 17)
  (h8 : price_reduction = 0.1 * a)
  (h9 : unsold_peaches = 20)
  (h10 : new_batch_profit = 2980) :
  ∃ (peach_cost_price : ℝ),
    peach_cost_price = 5 ∧
    (let apple_cost_price := 1.2 * peach_cost_price
     (peach_sell_price - peach_cost_price) * ↑peach_quantity + 
     (apple_sell_price - apple_cost_price) * ↑apple_quantity = total_profit) ∧
    a = 25 ∧
    new_peach_sell_price * (8 * a) + 
    (peach_sell_price - price_reduction) * (↑new_peach_quantity - 8 * a - ↑unsold_peaches) - 
    peach_cost_price * ↑new_peach_quantity = new_batch_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_shop_problem_l443_44334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_expression_two_equality_l443_44392

-- Define the constant e (Euler's number)
noncomputable def e : ℝ := Real.exp 1

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement for the first expression
theorem expression_one_equality :
  1 / (Real.sqrt 2 - 1) - (3/5:ℝ)^(0:ℝ) + (9/4:ℝ)^(-(1/2):ℝ) + 4 * (Real.sqrt 2 - e)^4 =
  2/3 + Real.sqrt 2 + 4 * (Real.sqrt 2 - e)^4 :=
by sorry

-- Statement for the second expression
theorem expression_two_equality :
  log10 500 + log10 (8/5) - 1/2 * log10 64 + 50 * (log10 2 + log10 5)^2 = 52 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equality_expression_two_equality_l443_44392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_840_l443_44381

theorem divisibility_by_840 (n : ℕ) : 
  (∃ k : ℤ, n^3 + 6*n^2 - 4*n - 24 = 840 * k) ↔ 
  (n % 70 ∈ ({2, 8, 12, 22, 44, 54, 58, 64, 68} : Finset ℕ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_840_l443_44381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l443_44316

/-- Given a circle C with center (2, π/3) in polar coordinates and radius √5,
    its polar equation is ρ² - 2ρcosθ - 2√3ρsinθ - 1 = 0. -/
theorem circle_polar_equation (ρ θ : ℝ) :
  let center_r : ℝ := 2
  let center_θ : ℝ := π / 3
  let radius : ℝ := Real.sqrt 5
  (ρ * Real.cos θ - center_r * Real.cos center_θ)^2 + 
  (ρ * Real.sin θ - center_r * Real.sin center_θ)^2 = radius^2 ↔ 
  ρ^2 - 2*ρ*Real.cos θ - 2*Real.sqrt 3*ρ*Real.sin θ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l443_44316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roaming_area_approx_l443_44393

/-- Represents a triangular shed -/
structure TriangularShed :=
  (side_length : ℝ)

/-- Represents a dog tied to the shed -/
structure TiedDog :=
  (rope_length : ℝ)
  (tie_position : ℝ)  -- Distance from a vertex of the shed

/-- Calculates the approximate maximum area a dog can roam -/
noncomputable def max_roaming_area (shed : TriangularShed) (dog : TiedDog) : ℝ :=
  Real.pi * dog.rope_length^2

/-- Theorem stating the maximum roaming area for the given conditions -/
theorem max_roaming_area_approx 
  (shed : TriangularShed) 
  (dog : TiedDog) 
  (h1 : shed.side_length = 20) 
  (h2 : dog.rope_length = 10) 
  (h3 : dog.tie_position = 10) : 
  ∃ (ε : ℝ), ε > 0 ∧ |max_roaming_area shed dog - 100 * Real.pi| < ε :=
by
  sorry

#check max_roaming_area_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roaming_area_approx_l443_44393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qr_length_l443_44326

/-- Given a triangle PQR with PR = 5, PQ = 7, N as the midpoint of QR, and PN = 4, prove that QR = 2√21 -/
theorem triangle_qr_length (P Q R N : EuclideanSpace ℝ (Fin 2)) : 
  dist P R = 5 →
  dist P Q = 7 →
  N = midpoint ℝ Q R →
  dist P N = 4 →
  dist Q R = 2 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qr_length_l443_44326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l443_44380

theorem sum_of_reciprocals (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) : 
  1/a + 1/b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l443_44380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_to_circle_l443_44337

/-- The greatest distance from the origin to any point on a circle -/
theorem greatest_distance_to_circle (center : ℝ × ℝ) (radius : ℝ) :
  let origin : ℝ × ℝ := (0, 0)
  let distance_to_center := Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  let max_distance : ℝ := distance_to_center + radius
  center = (6, -2) →
  radius = 12 →
  max_distance = 12 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_to_circle_l443_44337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_derivative_at_one_l443_44303

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_differentiable : Differentiable ℝ f
axiom g_differentiable : Differentiable ℝ g
axiom f_at_one : f 1 = -1
axiom f_deriv_at_one : deriv f 1 = 2
axiom g_at_one : g 1 = -2
axiom g_deriv_at_one : deriv g 1 = 1

-- Define the product function
noncomputable def h (x : ℝ) : ℝ := f x * g x

-- State the theorem
theorem product_derivative_at_one :
  deriv h 1 = -5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_derivative_at_one_l443_44303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_non_divisor_l443_44388

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_first_n_odd (n : ℕ) : ℕ := n^2

noncomputable def product_of_first_n_odd (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (λ k => 2*k + 1)

theorem greatest_three_digit_non_divisor :
  ∃ (n : ℕ), is_three_digit n ∧
    ¬(sum_of_first_n_odd n ∣ product_of_first_n_odd n) ∧
    ∀ m, is_three_digit m ∧ m > n →
      (sum_of_first_n_odd m ∣ product_of_first_n_odd m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_three_digit_non_divisor_l443_44388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l443_44356

/-- The function f(x) = (x^2 - 2x + 5) / (x - 1) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 5) / (x - 1)

/-- The closed interval [2, 9] -/
def I : Set ℝ := Set.Icc 2 9

theorem sum_of_max_and_min_f : 
  ∃ (m M : ℝ), (∀ x ∈ I, m ≤ f x ∧ f x ≤ M) ∧ m + M = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l443_44356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brass_alloy_mixture_zinc_amount_l443_44362

/-- Represents an alloy with a given ratio of copper to zinc and total weight -/
structure Alloy where
  copper_ratio : ℚ
  zinc_ratio : ℚ
  total_weight : ℚ

/-- Calculates the amount of zinc in an alloy -/
def zinc_amount (a : Alloy) : ℚ :=
  (a.zinc_ratio / (a.copper_ratio + a.zinc_ratio)) * a.total_weight

/-- The problem statement -/
theorem brass_alloy_mixture_zinc_amount :
  let alloy_a : Alloy := ⟨3, 5, 80⟩
  let alloy_b : Alloy := ⟨4, 9, 120⟩
  let total_zinc : ℚ := zinc_amount alloy_a + zinc_amount alloy_b
  ∃ ε > 0, |total_zinc - 133.07| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brass_alloy_mixture_zinc_amount_l443_44362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_exist_and_unique_l443_44348

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Calculate the area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.a * l.b) / (2 * abs l.a * abs l.b)

/-- The main theorem stating the existence and uniqueness of the two lines -/
theorem two_lines_exist_and_unique :
  ∃! (l1 l2 : Line),
    (l1.contains (-5) (-4) ∧
     l2.contains (-5) (-4) ∧
     triangleArea l1 = 5 ∧
     triangleArea l2 = 5 ∧
     ((l1.a = 8 ∧ l1.b = -5 ∧ l1.c = 20) ∨
      (l1.a = 2 ∧ l1.b = -5 ∧ l1.c = -10)) ∧
     ((l2.a = 8 ∧ l2.b = -5 ∧ l2.c = 20) ∨
      (l2.a = 2 ∧ l2.b = -5 ∧ l2.c = -10)) ∧
     l1 ≠ l2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_exist_and_unique_l443_44348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l443_44309

/-- The incenter of a triangle -/
noncomputable def incenter (x₁ y₁ x₂ y₂ x₃ y₃ a b c : ℝ) : ℝ × ℝ :=
  ((a*x₁ + b*x₂ + c*x₃) / (a+b+c), (a*y₁ + b*y₂ + c*y₃) / (a+b+c))

/-- Theorem: The coordinates of the incenter of a triangle -/
theorem incenter_coordinates
  (x₁ y₁ x₂ y₂ x₃ y₃ a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : (x₁, y₁) ≠ (x₂, y₂))
  (h₅ : (x₂, y₂) ≠ (x₃, y₃))
  (h₆ : (x₃, y₃) ≠ (x₁, y₁)) :
  ∃ I : ℝ × ℝ, I = incenter x₁ y₁ x₂ y₂ x₃ y₃ a b c ∧ 
  I.1 = (a*x₁ + b*x₂ + c*x₃) / (a+b+c) ∧
  I.2 = (a*y₁ + b*y₂ + c*y₃) / (a+b+c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l443_44309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_cosine_l443_44341

noncomputable def nested_sqrt (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | k + 1 => Real.sqrt (2 + nested_sqrt k)

theorem nested_sqrt_cosine (n : ℕ) :
  n > 0 → nested_sqrt n = 2 * Real.cos (π / (2^(n+1))) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_cosine_l443_44341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l443_44343

open Real Set

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * sin x + cos x

-- Define the domain
def domain : Set ℝ := Ioo (-π) π

-- Theorem statement
theorem f_strictly_increasing :
  ∀ x ∈ domain,
    (x ∈ Ioo (-π) (-π/2) ∨ x ∈ Ioo 0 (π/2)) →
    ∃ δ > 0, ∀ y ∈ domain, y ∈ Ioo x (x + δ) → f y > f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l443_44343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l443_44397

/-- A monic quartic polynomial satisfying specific conditions -/
noncomputable def p : ℝ → ℝ := sorry

/-- p is a monic quartic polynomial -/
axiom p_monic_quartic : ∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (p 0)

/-- Conditions for p -/
axiom p_cond_1 : p 1 = 1
axiom p_cond_2 : p 2 = 9
axiom p_cond_3 : p 3 = 28
axiom p_cond_4 : p 4 = 65

/-- Theorem: p(5) = 126 -/
theorem p_value_at_5 : p 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l443_44397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_savings_theorem_l443_44354

/-- Calculates the total savings from a betting scenario -/
noncomputable def total_savings (initial_winnings : ℝ) : ℝ :=
  let first_savings := initial_winnings / 2
  let second_bet := initial_winnings / 2
  let second_earnings := second_bet + (second_bet * 0.6)
  let second_savings := second_earnings / 2
  first_savings + second_savings

/-- Theorem stating that the total savings from the given scenario is $90.00 -/
theorem betting_savings_theorem :
  total_savings 100 = 90 := by
  -- Unfold the definition of total_savings
  unfold total_savings
  -- Simplify the arithmetic expressions
  simp [add_div, mul_div, div_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_savings_theorem_l443_44354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l443_44365

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l443_44365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_squared_l443_44378

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  AB : ℚ
  CD : ℚ
  x : ℚ
  h_AB : AB = 92
  h_CD : CD = 19
  h_sides : x ≥ (AB - CD) / 2 -- This ensures the trapezoid is valid

/-- The squared length of the non-parallel side of the trapezoid -/
def side_length_squared (t : IsoscelesTrapezoid) : ℚ :=
  (t.AB - t.CD)^2 / 4 + t.CD * (t.AB - t.CD) / 4

/-- The theorem stating the minimum value of x^2 -/
theorem min_side_length_squared (t : IsoscelesTrapezoid) : 
  side_length_squared t ≥ 1679 ∧ 
  (∃ t' : IsoscelesTrapezoid, side_length_squared t' = 1679) := by
  sorry

#eval side_length_squared ⟨92, 19, 41, rfl, rfl, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_length_squared_l443_44378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_completion_time_l443_44315

/-- Represents the typing job with given parameters and calculates Jack's completion time -/
noncomputable def typing_job (total_time : ℝ) (john_work_time : ℝ) (jack_rate_ratio : ℝ) : ℝ :=
  let john_rate := 1 / total_time
  let john_work := john_rate * john_work_time
  let remaining_work := 1 - john_work
  let jack_rate := jack_rate_ratio * john_rate
  remaining_work / jack_rate

/-- Theorem stating that Jack will take 5 hours to complete the remaining work -/
theorem jack_completion_time :
  typing_job 5 3 (2/5) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_completion_time_l443_44315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l443_44313

/-- A function y in terms of x and m -/
noncomputable def y (x m : ℝ) : ℝ := (m - 3) * (x^(m^2 - 2*m - 1)) + 2*x + 1

/-- Condition for y to be a quadratic function in x -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2*m - 1 = 2 ∧ m ≠ 3

/-- Theorem stating that m = -1 is the only value satisfying the conditions -/
theorem unique_m_value : ∃! m : ℝ, is_quadratic m ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l443_44313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reeyas_weighted_average_l443_44382

def scores : List ℝ := [50, 60, 70, 80, 80]
def weights : List ℝ := [3, 2, 4, 1, 3]

theorem reeyas_weighted_average :
  (List.sum (List.zipWith (· * ·) scores weights)) / (List.sum weights) = 66.92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reeyas_weighted_average_l443_44382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l443_44363

/-- Defines the sequence a_k based on the given conditions -/
def sequence_a (n : ℕ+) : ℕ → ℕ
| 0 => n
| (k+1) => (List.range (k+2)).sum % (k+2)

/-- States that the sequence a_k eventually becomes constant -/
theorem sequence_eventually_constant (n : ℕ+) :
  ∃ (K : ℕ) (m : ℕ), ∀ k ≥ K, sequence_a n k = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_constant_l443_44363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_distributed_first_five_days_l443_44367

def workers_on_day (n : ℕ) : ℕ := 64 + 7 * (n - 1)

def total_workers_up_to_day (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => workers_on_day (i + 1))

def rice_distributed_up_to_day (n : ℕ) : ℕ :=
  3 * total_workers_up_to_day n

theorem rice_distributed_first_five_days :
  rice_distributed_up_to_day 5 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_distributed_first_five_days_l443_44367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition1_proposition2_proposition3_not_always_true_proposition4_not_always_true_l443_44350

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (linePlaneParallel : Line → Plane → Prop)
variable (linePlanePerpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Non-intersection of lines m and n
variable (nonIntersecting : ¬ (m = n))

-- Distinctness of planes
variable (distinctPlanes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Theorem statements
theorem proposition1 : 
  linePlanePerpendicular m α → linePlaneParallel n α → perpendicular m n := by
  sorry

theorem proposition2 : 
  planeParallel α β → planeParallel β γ → linePlanePerpendicular m α → 
  linePlanePerpendicular m γ := by
  sorry

theorem proposition3_not_always_true : 
  ¬(∀ m n α, linePlaneParallel m α → linePlaneParallel n α → parallel m n) := by
  sorry

theorem proposition4_not_always_true : 
  ¬(∀ α β γ, planePerpendicular α γ → planePerpendicular β γ → planeParallel α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition1_proposition2_proposition3_not_always_true_proposition4_not_always_true_l443_44350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_pest_control_l443_44332

def bugs_eaten_per_spider (initial_bugs : ℕ) (spray_reduction : ℚ) 
  (num_spiders : ℕ) (final_bugs : ℕ) : ℚ :=
  let after_spray := (initial_bugs : ℚ) * spray_reduction
  let bugs_eaten := after_spray - final_bugs
  bugs_eaten / num_spiders

theorem garden_pest_control : 
  bugs_eaten_per_spider 400 (4/5) 12 236 = 7 := by
  -- Unfold the definition of bugs_eaten_per_spider
  unfold bugs_eaten_per_spider
  -- Simplify the arithmetic
  simp [Nat.cast_mul, Nat.cast_sub]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_pest_control_l443_44332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l443_44347

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x + 1) * (a * x + 2 * a - 2)

-- State the theorem
theorem a_upper_bound (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ f a x - 2 < 0) → a < 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_upper_bound_l443_44347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l443_44311

theorem divisor_problem (d : ℕ) : 
  d = 10^23 - 10 ↔ (10^23 - 7) % d = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l443_44311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solvability_l443_44324

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- State the theorem
theorem triangle_solvability (t : Triangle) :
  -- Case 1: Two angles and one side are known
  ((t.α ≠ 0 ∧ t.β ≠ 0 ∧ t.a ≠ 0) →
    ∃! (t' : Triangle), t'.α = t.α ∧ t'.β = t.β ∧ t'.a = t.a) ∧
  -- Case 2: Two sides and one angle are known
  ((t.a ≠ 0 ∧ t.b ≠ 0 ∧ t.α ≠ 0) →
    ∃! (t' : Triangle), t'.a = t.a ∧ t'.b = t.b ∧ t'.α = t.α) :=
by sorry

-- Define the sum of angles in a triangle
axiom angle_sum (t : Triangle) : t.α + t.β + t.γ = Real.pi

-- Define the Law of Sines
axiom law_of_sines (t : Triangle) :
  t.a / Real.sin t.α = t.b / Real.sin t.β

-- Define the Law of Cosines
axiom law_of_cosines (t : Triangle) :
  t.c ^ 2 = t.a ^ 2 + t.b ^ 2 - 2 * t.a * t.b * Real.cos t.γ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solvability_l443_44324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l443_44395

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 3)

theorem monotonic_increasing_interval_of_f :
  ∀ k : ℤ, StrictMonoOn f (Set.Ioo (- π / 12 + k * π / 2) (5 * π / 12 + k * π / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l443_44395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_sin_l443_44302

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

-- Define the transformation functions
noncomputable def shift (f : ℝ → ℝ) (s : ℝ) : ℝ → ℝ := λ x => f (x - s)
noncomputable def double_x (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x / 2)

-- Define the resulting function g after transformations
noncomputable def g : ℝ → ℝ := double_x (shift f (Real.pi / 3))

-- Theorem statement
theorem g_equals_sin : g = Real.sin := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_sin_l443_44302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l443_44369

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between (a b : V) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem magnitude_of_vector_sum (a b : V)
  (angle_eq : angle_between V a b = π / 3)
  (norm_a : norm a = 1)
  (norm_b : norm b = 2) :
  norm (2 • a + b) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l443_44369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_problem_l443_44346

theorem remainder_division_problem : ∃ (n : ℕ), (1225 * 1227 * n) % 12 = 3 ∧ ∀ (m : ℕ), (1225 * 1227 * m) % 12 = 3 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_division_problem_l443_44346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_plus_one_l443_44338

/-- A positive integer is perfect if the sum of all its positive divisors (including 1 and itself) is equal to twice the number. -/
def IsPerfect (m : ℕ) : Prop :=
  (Finset.sum (Finset.filter (· ∣ m) (Finset.range (m + 1))) id) = 2 * m

/-- n^n + 1 is a perfect number if and only if n = 3 -/
theorem perfect_power_plus_one (n : ℕ) : IsPerfect (n^n + 1) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_power_plus_one_l443_44338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_problem_l443_44383

/-- Given a matrix A and conditions, prove the values of θ and k -/
theorem matrix_inverse_problem (θ : ℝ) (k : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 6; 10, 3 * Real.cos θ]
  Real.cos θ ≠ 0 →
  A⁻¹ = k • A →
  θ = Real.arccos (1/3) ∧ k = -1/56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_inverse_problem_l443_44383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_proof_l443_44306

/-- Compound interest calculation --/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * ((1 + r/n)^(n*t) - 1)

/-- Total amount calculation --/
noncomputable def total_amount (P : ℝ) (CI : ℝ) : ℝ :=
  P + CI

/-- Theorem: Given the conditions, prove that the total amount returned is 19828.80 --/
theorem total_amount_proof (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) (CI : ℝ) :
  r = 0.08 → n = 1 → t = 2 → CI = 2828.80 →
  compound_interest P r n t = CI →
  total_amount P CI = 19828.80 :=
by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues
-- #eval total_amount_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_proof_l443_44306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_coefficient_l443_44312

theorem factor_implies_coefficient (d : ℚ) : 
  (∃ (P : ℚ → ℚ), ∀ x, x^3 - 4*x^2 + d*x - 8 = (x + 3) * P x) → 
  d = -71/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_coefficient_l443_44312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_iff_zero_normals_l443_44325

/-- A convex k-gon in a 2D Euclidean space. -/
structure ConvexKGon (k : ℕ) where
  vertices : Fin k → ℝ × ℝ
  is_convex : Prop

/-- The unit outward normal vector for a side of a k-gon. -/
def unitOutwardNormal (kg : ConvexKGon k) (i : Fin k) : ℝ × ℝ := sorry

/-- The distance from a point to a side of the k-gon. -/
def distanceToSide (kg : ConvexKGon k) (p : ℝ × ℝ) (i : Fin k) : ℝ := sorry

/-- The sum of distances from a point to all sides of the k-gon. -/
def sumOfDistances (kg : ConvexKGon k) (p : ℝ × ℝ) : ℝ :=
  Finset.sum (Finset.range k) (fun i => distanceToSide kg p ⟨i, sorry⟩)

/-- The sum of all unit outward normal vectors of the k-gon. -/
def sumOfNormals (kg : ConvexKGon k) : ℝ × ℝ :=
  Finset.sum (Finset.range k) (fun i => unitOutwardNormal kg ⟨i, sorry⟩)

/-- The theorem stating the equivalence of constant sum of distances and zero sum of normals. -/
theorem constant_sum_iff_zero_normals (kg : ConvexKGon k) :
  (∀ p q : ℝ × ℝ, sumOfDistances kg p = sumOfDistances kg q) ↔ sumOfNormals kg = (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_iff_zero_normals_l443_44325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l443_44344

-- Define the given parameters
noncomputable def train_length : ℝ := 250  -- in meters
noncomputable def train_speed : ℝ := 120   -- in km/h
noncomputable def man_speed : ℝ := 18      -- in km/h

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the relative speed in m/s
noncomputable def relative_speed : ℝ := (train_speed - man_speed) * kmh_to_ms

-- Define the time taken for the train to pass the man
noncomputable def time_to_pass : ℝ := train_length / relative_speed

-- State the theorem
theorem train_passing_time_approx :
  abs (time_to_pass - 8.82) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l443_44344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_of_30_l443_44375

def factors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_of_30_l443_44375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l443_44314

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) :
  (∀ x, f ω (x + π/6) = Real.cos (ω * x)) →
  ω ≥ 3 ∧ ∃ (ω₀ : ℝ), ω₀ = 3 ∧ (∀ x, f ω₀ (x + π/6) = Real.cos (ω₀ * x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l443_44314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l443_44377

-- Define the function f with domain [0,4]
def f : Set ℝ → Set ℝ := sorry

-- Define the property that f has domain [0,4]
def has_domain_zero_to_four (f : Set ℝ → Set ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 4 ↔ x ∈ f (Set.Icc 0 4)

-- Theorem statement
theorem domain_of_composite_function
  (f : Set ℝ → Set ℝ)
  (h : has_domain_zero_to_four f) :
  ∀ x, x ∈ Set.Icc (-2) 2 ↔ x^2 ∈ f (Set.Icc 0 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l443_44377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_l443_44399

theorem count_multiples : ∃ S : Finset ℕ, 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 2500 ∧ (4 ∣ n ∨ 5 ∣ n) ∧ ¬(15 ∣ n)) ∧
  Finset.card S = 959 := by
  -- Define the set of numbers we're interested in
  let S : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 2500 ∧ (4 ∣ n ∨ 5 ∣ n) ∧ ¬(15 ∣ n)) (Finset.range 2501)
  
  -- State that the cardinality of this set is 959
  use S
  
  sorry -- The proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_l443_44399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_value_l443_44364

-- Define m and n as noncomputable real numbers
noncomputable def m : ℝ := 2 + Real.sqrt 3
noncomputable def n : ℝ := 2 - Real.sqrt 3

-- State the theorem
theorem square_root_value : Real.sqrt (m^2 + n^2 - 3*m*n) = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_value_l443_44364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l443_44366

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * x - a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.sin x₀ ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc (Real.exp (-1) - 1) (Real.exp 1 + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l443_44366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_minimum_l443_44330

/-- The minimum distance between a point on y = 2^x and a point on y = log_2 x -/
noncomputable def min_distance : ℝ := Real.sqrt 2 * (1 + Real.log (Real.log 2)) / Real.log 2

/-- A point on the curve y = 2^x -/
structure PointP where
  x : ℝ
  y : ℝ
  on_curve : y = 2^x

/-- A point on the curve y = log_2 x -/
structure PointQ where
  x : ℝ
  y : ℝ
  on_curve : y = Real.log x / Real.log 2

/-- The distance between two points -/
noncomputable def distance (p : PointP) (q : PointQ) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem stating that min_distance is the minimum distance between points on the two curves -/
theorem min_distance_is_minimum :
  ∀ (p : PointP) (q : PointQ), distance p q ≥ min_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_minimum_l443_44330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l443_44300

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to cross the person. -/
noncomputable def trainLength (trainSpeed manSpeed : ℝ) (crossingTime : ℝ) : ℝ :=
  let relativeSpeed := trainSpeed - manSpeed
  let relativeSpeedMS := relativeSpeed * (1000 / 3600)
  relativeSpeedMS * crossingTime

/-- The length of the train is approximately 1199.9 meters when the train speed is 63 km/h, 
    the man's walking speed is 3 km/h, and it takes 23.998 seconds for the train to cross the man. -/
theorem train_length_calculation :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (trainLen : ℝ),
    abs (trainLen - 1199.9) < ε ∧
    trainLen = trainLength 63 3 23.998 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l443_44300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l443_44370

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

-- Define the line l passing through (0,2)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 2

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem line_equation_proof :
  ∃ k : ℝ, (k = 4/3 ∨ k = 0) ∧
  ∀ x y : ℝ, line_l k x y →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l443_44370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l443_44349

theorem largest_lambda : 
  (∃ (l : ℝ), ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b^2 + l*b^2*c + c^2*d) ∧ 
  (∀ (l : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b^2 + l*b^2*c + c^2*d) → l ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l443_44349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCM_is_40_l443_44368

/-- A 10-sided polygon with alternating side lengths and right angles -/
structure DecagonRightAlternating :=
  (vertices : Fin 10 → ℝ × ℝ)
  (alternating_sides : ∀ i : Fin 10, 
    let j := i.succ
    dist (vertices i) (vertices j) = if i % 2 = 0 then 5 else 3)
  (right_angles : ∀ i : Fin 10,
    let j := i.succ
    (vertices j).1 - (vertices i).1 = 0 ∨ (vertices j).2 - (vertices i).2 = 0)

/-- The intersection point of two line segments -/
noncomputable def intersection_point (p1 p2 q1 q2 : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The area of a quadrilateral given its four vertices -/
noncomputable def area_quadrilateral (a b c d : ℝ × ℝ) : ℝ :=
  sorry

/-- The area of quadrilateral ABCM in the given decagon -/
noncomputable def area_ABCM (d : DecagonRightAlternating) : ℝ :=
  let A := d.vertices 0
  let B := d.vertices 1
  let C := d.vertices 2
  let E := d.vertices 4
  let G := d.vertices 6
  let M := intersection_point A E C G
  area_quadrilateral A B C M

/-- Theorem stating that the area of ABCM is 40 -/
theorem area_ABCM_is_40 (d : DecagonRightAlternating) :
  area_ABCM d = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABCM_is_40_l443_44368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l443_44335

theorem geometric_sequence_common_ratio (a : ℝ) : 
  let seq := λ n : ℕ => a + Real.log 3 / Real.log (2^(2^n))
  ∃ q : ℝ, q = 1/3 ∧ ∀ n : ℕ, seq (n+1) / seq n = q :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l443_44335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l443_44339

def spinner_numbers : List ℕ := [2, 4, 7, 9, 3, 10, 11, 8]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

theorem spinner_prime_probability :
  let total_sectors := spinner_numbers.length
  let prime_sectors := (spinner_numbers.filter is_prime).length
  (prime_sectors : ℚ) / total_sectors = 1 / 2 := by
    sorry

#eval spinner_numbers.filter is_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_l443_44339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44355

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

-- State the theorem
theorem f_properties (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f a x > f a y) ∧
  (a > 0 → ∀ x y, x < y → 
    ((x < Real.log (1/a) ∧ y < Real.log (1/a)) → f a x > f a y) ∧
    ((x > Real.log (1/a) ∧ y > Real.log (1/a)) → f a x < f a y)) ∧
  (∃ x y, x < y ∧ f a x = 0 ∧ f a y = 0 ↔ 0 < a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_is_2_sqrt_2_l443_44398

/-- Circle with equation x^2 + (y-4)^2 = 4 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 4}

/-- Point P(1,3) -/
def P : ℝ × ℝ := (1, 3)

/-- Minimum chord length through P -/
noncomputable def min_chord_length : ℝ := 2 * Real.sqrt 2

/-- Theorem: The minimum chord length through P is 2√2 -/
theorem min_chord_length_is_2_sqrt_2 : 
  ∀ (chord : Set (ℝ × ℝ)), 
    (∀ p ∈ chord, p ∈ C) → 
    P ∈ chord → 
    (∃ (a b : ℝ × ℝ), a ≠ b ∧ a ∈ chord ∧ b ∈ chord ∧ 
      ‖a - b‖ ≥ min_chord_length) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_length_is_2_sqrt_2_l443_44398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_number_l443_44374

/-- Represents the sequence of numbers said by Jo and Blair. -/
def sequenceJoBlair : ℕ → ℕ
| 0 => 1  -- Jo starts with 1
| 1 => 2  -- Blair says 2
| n + 2 => 
  if n % 3 = 1  -- Every third turn for Jo
  then sequenceJoBlair (n + 1) + 2  -- Jo adds 2
  else sequenceJoBlair (n + 1) + 1  -- Normal increment

/-- The 53rd number in the sequence is 72. -/
theorem fifty_third_number : sequenceJoBlair 52 = 72 := by
  sorry

#eval sequenceJoBlair 52  -- This will compute and print the actual 53rd number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_number_l443_44374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44323

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.sqrt 3 * (Real.sin (x / 2))^2

-- State the theorem
theorem f_properties :
  -- The smallest positive period is 2π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum value on [0, 2π/3] is -√3
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → f x ≥ -Real.sqrt 3) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 ∧ f x = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l443_44323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_constant_l443_44327

def b (n : ℕ) : ℤ := (10^n - 9) / 3

theorem gcd_b_constant (n : ℕ) (h : n ≥ 1) : Int.gcd (b n) (b (n + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_constant_l443_44327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l443_44340

theorem complex_division_result : 
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l443_44340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sequence_l443_44390

def b (n : ℕ) : ℤ := n - 35

def a : ℕ → ℕ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | n + 1 => a n + 2^n

theorem max_value_sequence (n : ℕ) :
  n ≥ 1 → a n = 2^n ∧ (∀ m : ℕ, m ≥ 1 → (b m : ℚ) / (a m : ℚ) ≤ 1 / 2^36) :=
by
  sorry

#eval a 5  -- Adding an evaluation to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sequence_l443_44390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_proof_l443_44320

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (4 * x + Real.pi / 4)

noncomputable def stretched_function (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

noncomputable def final_function (x : ℝ) : ℝ := Real.sin (2 * x)

theorem symmetry_center_proof :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = final_function x) ∧ 
    (∀ x, f (Real.pi / 2 - x) = f (Real.pi / 2 + x)) ∧
    (f (Real.pi / 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_proof_l443_44320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l443_44351

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := f (x/2)

theorem f_properties_and_g_range :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∃ (m : ℝ), m = -(2 + Real.sqrt 3)/2 ∧ ∀ (x : ℝ), f x ≥ m) ∧
  (∀ (y : ℝ), y ∈ Set.Icc ((1 - Real.sqrt 3)/2) ((2 - Real.sqrt 3)/2) ↔
    ∃ (x : ℝ), x ∈ Set.Icc (Real.pi/2) Real.pi ∧ g x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l443_44351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rescue_time_l443_44310

/-- The minimum time for Person A to reach Person B given the specified conditions. -/
theorem minimum_rescue_time (BC angle_BAC swimming_speed : ℝ) 
  (h1 : BC = 30)
  (h2 : angle_BAC = 15 * π / 180)
  (h3 : swimming_speed = 3) :
  ∃ (t : ℝ), t = 45 * Real.sqrt 2 + 15 * Real.sqrt 6 ∧ 
    ∀ (t' : ℝ), t' ≥ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_rescue_time_l443_44310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_cost_range_l443_44376

/-- Represents the cost range for a menu item -/
structure CostRange where
  min : Float
  max : Float

/-- Represents a menu item with its cost range and discount or tax rate -/
structure MenuItem where
  name : String
  costRange : CostRange
  rate : Float
  isTax : Bool

/-- Calculates the total cost for an order -/
noncomputable def calculateTotalCost (items : List (MenuItem × Nat)) : CostRange :=
  let minTotal := items.foldl (fun acc (item, quantity) =>
    let itemCost := item.costRange.min * quantity.toFloat
    let adjustedCost := if item.isTax
      then itemCost * (1 + item.rate)
      else itemCost * (1 - item.rate)
    acc + adjustedCost
  ) 0
  let maxTotal := items.foldl (fun acc (item, quantity) =>
    let itemCost := item.costRange.max * quantity.toFloat
    let adjustedCost := if item.isTax
      then itemCost * (1 + item.rate)
      else itemCost * (1 - item.rate)
    acc + adjustedCost
  ) 0
  { min := minTotal, max := maxTotal }

/-- The main theorem stating the range of possible costs for the given order -/
theorem order_cost_range :
  let cake := { name := "Cake", costRange := { min := 1.80, max := 2.40 }, rate := 0.12, isTax := false }
  let milkTea := { name := "Milk Tea", costRange := { min := 1.20, max := 3.00 }, rate := 0.08, isTax := false }
  let brownie := { name := "Chocolate Brownie", costRange := { min := 2.00, max := 3.60 }, rate := 0.15, isTax := true }
  let order := [(cake, 5), (milkTea, 3), (brownie, 2)]
  let result := calculateTotalCost order
  (result.min - 15.832).abs < 0.001 ∧ result.max = 27.12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_cost_range_l443_44376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_equals_4_99_l443_44371

/-- A sequence of real numbers defined recursively -/
def b : ℕ → ℝ
  | 0 => 1  -- Add a case for 0 to cover all natural numbers
  | 1 => 1
  | n + 1 => (64 * (b n)^3)^(1/3)

/-- Theorem stating that the 100th term of the sequence equals 4^99 -/
theorem b_100_equals_4_99 : b 100 = 4^99 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_equals_4_99_l443_44371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sequence_sum_l443_44396

/-- The unique real root of x^3 + 2x - 1 = 0 between 0.4 and 0.5 -/
noncomputable def r : ℝ := Real.sqrt (1/3)

/-- The sequence a_n = 3n - 2 -/
def a : ℕ → ℕ
  | n => 3 * n - 2

theorem cubic_root_sequence_sum (hr1 : r^3 + 2*r - 1 = 0) (hr2 : 0.4 < r) (hr3 : r < 0.5) :
  (∃! f : ℕ → ℕ, StrictMono f ∧ (∑' n, r^(f n) = 1/2)) ∧
  (∀ n, a n = 3 * n - 2) ∧
  StrictMono a ∧
  ∑' n, r^(a n) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sequence_sum_l443_44396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l443_44307

theorem triangle_area (A B C : ℝ) (h1 : A = 30 * Real.pi / 180) 
  (h2 : Real.sin A = 1/2) (h3 : AB = Real.sqrt 3) (h4 : AC = 1) :
  (1/2 * AB * AC * Real.sin A) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l443_44307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l443_44385

def setA : Set ℝ := {x : ℝ | x^2 - 4*x < 0}
def setB : Set ℝ := {y : ℝ | ∃ (n : ℤ), y = n}

theorem intersection_of_A_and_B :
  (setA ∩ setB : Set ℝ) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l443_44385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordans_score_l443_44304

theorem jordans_score (total_students : ℕ) (students_graded : ℕ) (initial_average : ℚ) (final_average : ℚ) (jordans_score : ℚ) :
  total_students = 20 →
  students_graded = 19 →
  initial_average = 74 →
  final_average = 76 →
  (students_graded * initial_average + (total_students - students_graded) * jordans_score) / total_students = final_average →
  jordans_score = 114 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordans_score_l443_44304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rugby_team_lineup_count_l443_44389

def choose (n k : ℕ) : ℕ := 
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def waysToChoosePlayers (totalPlayers forwardPlayers : ℕ) : ℕ :=
  let nonForwardPlayers := totalPlayers - forwardPlayers
  let remainingPlayers := totalPlayers - 1
  Finset.sum (Finset.range (min forwardPlayers 13 - 2)) (λ k =>
    choose forwardPlayers (k + 3) * choose nonForwardPlayers (12 - (k + 3)))

theorem rugby_team_lineup_count :
  let totalPlayers := 22
  let forwardPlayers := 8
  let waysToChooseCaptain := totalPlayers
  let waysToChooseOthers := waysToChoosePlayers totalPlayers forwardPlayers
  waysToChooseCaptain * waysToChooseOthers = 6478132 := by
  sorry

#eval waysToChoosePlayers 22 8 * 22

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rugby_team_lineup_count_l443_44389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l443_44319

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (2 : ℝ) ^ x > 1}

-- Define set B
def B : Set ℝ := Set.Icc (-1 : ℝ) 5

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = Set.Icc (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l443_44319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l443_44301

open Real

noncomputable def f (x : ℝ) : ℝ := 3 * sin (x / 2 + Real.pi / 3)

noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f ((x - Real.pi / 3) / 2)

noncomputable def g (x : ℝ) : ℝ := 3 * sin (x / 4 + Real.pi / 6)

theorem transformation_result :
  ∀ x : ℝ, transform f x = g x :=
by
  intro x
  simp [transform, f, g]
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l443_44301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l443_44331

/-- Triangle structure with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Angle bisector of a triangle -/
noncomputable def angle_bisector (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- Median of a triangle -/
noncomputable def median (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- Semiperimeter of a triangle -/
noncomputable def semiperimeter (t : Triangle) : ℝ :=
  (t.a + t.b + t.c) / 2

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

theorem triangle_inequality (t : Triangle) :
  (angle_bisector t 0) * (angle_bisector t 1) * (angle_bisector t 2) ≤
  (semiperimeter t) * (area t) ∧
  (semiperimeter t) * (area t) ≤
  (median t 0) * (median t 1) * (median t 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l443_44331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_circle_l443_44358

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- Define the curve
def curve (x y : ℝ) : Prop := y = 2/x ∧ x > 0

-- Define a circle with center (a, b) and radius r
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define tangency between a circle and the line
def is_tangent (a b r : ℝ) : Prop :=
  (2*a + b + 1)^2 / 5 = r^2

-- Theorem statement
theorem min_area_circle :
  ∃ (a b r : ℝ),
    curve a b ∧
    is_tangent a b r ∧
    (∀ (a' b' r' : ℝ), curve a' b' → is_tangent a' b' r' → r ≤ r') ∧
    circle_eq x y a b r ↔ circle_eq x y 1 2 (Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_circle_l443_44358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_high_diver_velocity_l443_44352

/-- The height function for a high diver -/
noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

/-- The instantaneous velocity of the high diver at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv h t

theorem high_diver_velocity : instantaneous_velocity 0.5 = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_high_diver_velocity_l443_44352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_is_thirteen_l443_44360

noncomputable def rectangular_prism_width (l h d : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (d^2 - l^2 - h^2))

theorem width_is_thirteen :
  rectangular_prism_width 5 10 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_is_thirteen_l443_44360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_savings_time_l443_44394

/-- The number of months required to save for a vehicle --/
def months_to_save (monthly_income : ℚ) (savings_fraction : ℚ) (vehicle_cost : ℚ) : ℚ :=
  vehicle_cost / (monthly_income * savings_fraction)

theorem vehicle_savings_time : 
  ⌈months_to_save 4000 (1/2) 16000⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_savings_time_l443_44394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l443_44322

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (4 - k) / x

-- Define the theorem
theorem inverse_proportion_k_range 
  (k : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ < 0) 
  (h2 : 0 < x₂) 
  (h3 : y₁ < y₂) 
  (h4 : y₁ = inverse_proportion k x₁) 
  (h5 : y₂ = inverse_proportion k x₂) :
  k < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_l443_44322
