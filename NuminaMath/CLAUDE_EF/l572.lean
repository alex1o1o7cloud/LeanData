import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_chocolate_bars_l572_57233

/-- The number of chocolate bars Matilda's father has left after all transactions -/
def chocolate_bars_left (num_people : ℕ) (bars_given_to_father : ℕ) (bars_given_away : ℕ) : ℕ :=
  (num_people * bars_given_to_father) - bars_given_away

/-- Theorem stating that Matilda's father ends up with 4 chocolate bars -/
theorem father_chocolate_bars : 
  chocolate_bars_left 7 2 10 = 4 := by
  rfl

#eval chocolate_bars_left 7 2 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_chocolate_bars_l572_57233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_sum_modulo_l572_57282

theorem prime_set_sum_modulo (p : ℕ) (hp : Nat.Prime p) (A B : Finset ℕ) 
  (hA : ∀ a ∈ A, a < p) (hB : ∀ b ∈ B, b < p) :
  let sum_set := (A.product B).image (fun (x : ℕ × ℕ) => (x.1 + x.2) % p)
  (A.card + B.card > p → sum_set.card = p) ∧
  (A.card + B.card ≤ p → sum_set.card ≥ A.card + B.card - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_sum_modulo_l572_57282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_speed_is_pi_over_three_l572_57267

/-- Represents a track with straight sides and semicircular ends -/
structure Track where
  width : ℝ
  timeDifference : ℝ

/-- Calculates the speed of a walker on the given track -/
noncomputable def walkSpeed (track : Track) : ℝ :=
  (2 * Real.pi * track.width) / track.timeDifference

theorem track_speed_is_pi_over_three (track : Track) 
  (h1 : track.width = 4)
  (h2 : track.timeDifference = 24) :
  walkSpeed track = Real.pi / 3 := by
  sorry

#check track_speed_is_pi_over_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_speed_is_pi_over_three_l572_57267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sqrt_four_only_l572_57252

theorem rational_sqrt_four_only (x : ℝ) : x ∈ ({Real.sqrt 2, Real.sqrt 4, Real.sqrt 6, Real.sqrt 8} : Set ℝ) → 
  (∃ (a b : ℤ), x = a / b ∧ b ≠ 0) ↔ x = Real.sqrt 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_sqrt_four_only_l572_57252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l572_57226

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -2)
def O : ℝ × ℝ := (0, 0)

-- Define the condition for point C
def C (α β : ℝ) : ℝ × ℝ := (α, -2*β)

-- Define the constraint for α and β
def αβ_constraint (α β : ℝ) : Prop := α - 2*β = 1

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the eccentricity condition
def eccentricity_condition (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ (b^2/a^2 - 1) ≤ 3

-- Theorem statement
theorem a_range :
  ∀ a b : ℝ,
  (∃ α β x y : ℝ,
    C α β = (x, y) ∧
    αβ_constraint α β ∧
    hyperbola a b x y ∧
    eccentricity_condition a b) →
  0 < a ∧ a ≤ 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l572_57226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_fixed_point_l572_57208

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetry_point : ℝ × ℝ := (2, 1)

/-- Line l₁ with equation y = k(x-4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Predicate to check if a point is on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.slope * (p.1 - l.point.1) + l.point.2

/-- Theorem: If l₁ is symmetric to l₂ about (2,1), then l₂ passes through (0,2) -/
theorem symmetric_line_passes_through_fixed_point (k : ℝ) (l₂ : Line) :
  (∀ (x y : ℝ), on_line (x, y) (l₁ k) ↔ on_line (4 - x + 2, 2 - y + 1) l₂) →
  on_line (0, 2) l₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_fixed_point_l572_57208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_left_on_beach_l572_57212

/-- The number of people left relaxing on the beach after some leave to wade in the water. -/
theorem people_left_on_beach : ℕ := by
  let initial_first_row : ℕ := 24
  let people_left_first_row : ℕ := 3
  let initial_second_row : ℕ := 20
  let people_left_second_row : ℕ := 5
  let third_row : ℕ := 18
  have : (initial_first_row - people_left_first_row) + 
         (initial_second_row - people_left_second_row) + 
         third_row = 54 := by
    rfl
  exact 54


end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_left_on_beach_l572_57212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l572_57243

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3

-- Define the function h
noncomputable def h (b c : ℝ) (x : ℝ) : ℝ := b * f x + c * x^2 + 5/3

-- Define the statement
theorem problem_statement 
  (b c : ℝ) 
  (h1 : h b c 1 = 1)  -- (1,1) lies on h(x)
  (h2 : (3*b*2 + 2*c = 0))  -- (1,1) is center of symmetry
  (a : ℝ) 
  (h3 : a > 0 ∧ a ≠ 1)
  (x1 x2 x3 : ℝ)
  (h4 : x1 < x2 ∧ x2 < 0 ∧ 0 < x3)
  (h5 : ∀ x, |f x| = a^x → (x = x1 ∨ x = x2 ∨ x = x3)) :
  (∃ (x0 : ℝ), 
    (h b c x0 - h b c 1 = (3*b*x0^2 + 2*c*x0)*(x0 - 1) ∧ 
     (3*b*x0^2 + 2*c*x0)*(-1 - x0) + h b c x0 = 1/3) ∨
    (h b c x0 - h b c 1 = (3*b*x0^2 + 2*c*x0)*(x0 - 1) ∧ 
     h b c x0 = 1/3)) ∧
  (Real.exp (-3/Real.exp 1) < a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l572_57243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l572_57292

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2*x + 1) / (x + 1)

-- Define the interval [1, +∞)
def I : Set ℝ := { x | x ≥ 1 }

-- Define the subinterval [1, 4]
def J : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem f_properties :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧  -- f is monotonically increasing on [1, +∞)
  (∀ x, x ∈ J → f x ≥ 3/2) ∧                    -- Minimum value on [1, 4] is 3/2
  (∀ x, x ∈ J → f x ≤ 9/5) ∧                    -- Maximum value on [1, 4] is 9/5
  (∃ x, x ∈ J ∧ f x = 3/2) ∧                    -- Minimum value is attained
  (∃ x, x ∈ J ∧ f x = 9/5) :=                   -- Maximum value is attained
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l572_57292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_rect_equation_C₁_equation_with_three_common_points_l572_57289

-- Define the polar coordinate system
noncomputable def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the curve C₂ in polar coordinates
def C₂ (ρ θ : ℝ) : Prop := ρ^2 + 2*ρ*(Real.cos θ) - 3 = 0

-- Theorem to prove
theorem C₂_rect_equation :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, C₂ ρ θ ∧ polar_to_rect ρ θ = (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

-- Define the curve C₁
def C₁ (k : ℝ) (x : ℝ) : ℝ := k * |x| + 2

-- Theorem for the equation of C₁ when it has exactly three common points with C₂
theorem C₁_equation_with_three_common_points :
  ∃ k : ℝ, (∀ x : ℝ, C₁ k x = -4/3 * |x| + 2) ∧
    (∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
      (C₁ k x₁)^2 + (x₁ + 1)^2 = 4 ∧
      (C₁ k x₂)^2 + (x₂ + 1)^2 = 4 ∧
      (C₁ k x₃)^2 + (x₃ + 1)^2 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_rect_equation_C₁_equation_with_three_common_points_l572_57289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l572_57264

theorem triangle_problem (AB : ℝ) (cosA sinC : ℝ) 
  (h1 : AB = 30)
  (h2 : cosA = 4/5)
  (h3 : sinC = 1/5) :
  ∃ AD BD BC DC : ℝ,
    AD = AB * cosA ∧
    BD = Real.sqrt (AB^2 - AD^2) ∧
    BC = BD / sinC ∧
    DC = Real.sqrt (BC^2 - BD^2) ∧
    DC = 88 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l572_57264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l572_57248

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 12) / (2*x - 6)

/-- The function g(x) with parameters b and c -/
noncomputable def g (b c : ℝ) (x : ℝ) : ℝ := (b*x^2 + 4*x + c) / (x - 3)

/-- The theorem stating the point of intersection -/
theorem intersection_point (b c : ℝ) :
  (∃ (x : ℝ), f x = g b c x ∧ x = 1) →  -- Intersection at x = 1
  (∀ (x : ℝ), x ≠ 3 → f x = g b c x) →  -- Same vertical asymptote
  (∃ (m₁ m₂ k₁ k₂ : ℝ), m₁ * m₂ = -1 ∧  -- Perpendicular oblique asymptotes
    (∀ (x : ℝ), |x| > 1 → |f x - (m₁ * x + k₁)| < 1) ∧
    (∀ (x : ℝ), |x| > 1 → |g b c x - (m₂ * x + k₂)| < 1)) →
  (∃ (x : ℝ), x ≠ 1 ∧ f x = g b c x ∧ x = -1/3 ∧ f x = -73/60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l572_57248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_8_l572_57254

/-- Represents a cistern with given dimensions and water level. -/
structure Cistern where
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the length of a cistern given its dimensions and wet surface area. -/
noncomputable def cisternLength (c : Cistern) : ℝ :=
  (c.wetSurfaceArea - 2 * c.width * c.depth) / (c.width + 2 * c.depth)

/-- Theorem stating that a cistern with width 6m, depth 1.25m, and wet surface area 83m² has a length of 8m. -/
theorem cistern_length_is_8 :
  let c : Cistern := ⟨6, 1.25, 83⟩
  cisternLength c = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_length_is_8_l572_57254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_count_primes_in_sequence_l572_57240

def sequenceN (n : ℕ) : ℕ :=
  match n with
  | 0 => 61
  | n + 1 => 100 * sequenceN n + 61

theorem only_first_term_prime :
  ∀ n : ℕ, n > 0 → ¬ Nat.Prime (sequenceN n) :=
by
  sorry

theorem count_primes_in_sequence :
  (Finset.filter (λ n => Nat.Prime (sequenceN n)) (Finset.range ω)).card = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_count_primes_in_sequence_l572_57240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_range_of_a_l572_57239

-- Define the conditions
def condition_p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a ≠ 0
def condition_q (x : ℝ) : Prop := 8 < (2 : ℝ)^(x + 1) ∧ (2 : ℝ)^(x + 1) ≤ 16

-- Define the theorem for question 1
theorem range_of_x (x : ℝ) : 
  condition_p x 1 ∧ condition_q x → x ∈ Set.Ioo 2 3 := by sorry

-- Define the theorem for question 2
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, condition_q x → condition_p x a) ∧ 
  (∃ x : ℝ, condition_p x a ∧ ¬condition_q x) → 
  a ∈ Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_range_of_a_l572_57239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_increase_l572_57274

/-- Given a line equation and two points on the line, prove the y-coordinate increase --/
theorem y_coordinate_increase (m n : ℝ) : 
  let line_eq (x y : ℝ) := x = (y / 5) - (2 / 5)
  let point1 := (m, n)
  let point2 := (m + 3, n + 15)
  (line_eq m n ∧ line_eq (m + 3) (n + 15)) → 15 = point2.2 - point1.2 := by
  intro h
  -- Proof steps would go here
  sorry

#check y_coordinate_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_increase_l572_57274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l572_57273

/-- Given a cone whose lateral surface unfolds into a sector with radius 3 and central angle 120°,
    the height of the cone is equal to √(9 - (3π/2)²). -/
theorem cone_height (r l h : ℝ) (angle : ℝ) : 
  l = 3 → 
  angle = 2 * π / 3 → 
  2 * π * r = angle * l → 
  h = Real.sqrt (l^2 - r^2) →
  h = Real.sqrt (9 - (3 * π / 2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l572_57273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_optimal_production_quantity_l572_57221

-- Define the total cost function
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the sales revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := R x - G x

-- Theorem statement
theorem max_profit_at_400_units :
  ∃ (max_profit : ℝ), max_profit = 3.6 ∧
  ∀ (x : ℝ), x ≥ 0 → f x ≤ max_profit ∧
  f 4 = max_profit := by
  sorry

-- Theorem to prove the optimal production quantity
theorem optimal_production_quantity :
  ∃ (x : ℝ), x = 4 ∧
  ∀ (y : ℝ), y ≥ 0 → f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_optimal_production_quantity_l572_57221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_proof_l572_57269

/-- The total surface area of a hemisphere with radius 10 cm, including its circular base -/
noncomputable def hemisphere_surface_area : ℝ := 300 * Real.pi

/-- The surface area of a sphere with radius r -/
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem: The total surface area of a hemisphere with radius 10 cm, including its circular base, is 300π cm² -/
theorem hemisphere_surface_area_proof :
  hemisphere_surface_area = sphere_surface_area 10 / 2 + Real.pi * 10^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_proof_l572_57269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l572_57299

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Final amount after applying simple interest -/
noncomputable def final_amount (principal rate time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem stating the relationship between principal, rate, time, and final amount -/
theorem principal_calculation (rate time final : ℝ) (h1 : rate = 11.67) (h2 : time = 5) (h3 : final = 950) :
  ∃ (principal : ℝ), (principal ≥ 599.5 ∧ principal ≤ 600.5) ∧ final_amount principal rate time = final := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l572_57299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l572_57219

/-- Triangle ABC with given angles and side lengths -/
structure Triangle where
  angleA : Real
  angleB : Real
  angleC : Real
  sideA : Real
  sideB : Real

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (t : Triangle) : Real :=
  0.5 * t.sideA * t.sideB * Real.sin (t.angleC * Real.pi / 180)

/-- Theorem stating that the area of the given triangle is approximately 51.702 -/
theorem triangle_area_approx :
  let t := Triangle.mk 80 60 40 15 7
  ∃ ε > 0, abs (triangleArea t - 51.702) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l572_57219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_from_circles_and_secant_l572_57256

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define membership for points in lines and circles
instance : Membership (ℝ × ℝ) Line where
  mem p l := ∃ t : ℝ, p = (1 - t) • l.point1 + t • l.point2

instance : Membership (ℝ × ℝ) Circle where
  mem p c := dist p c.center = c.radius

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line) : Prop :=
  let v1 := l1.point2 - l1.point1
  let v2 := l2.point2 - l2.point1
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the given conditions
def common_secant (circle1 circle2 : Circle) (C D : ℝ × ℝ) : Prop :=
  ∃ (l : Line), C ∈ l ∧ D ∈ l ∧ C ∈ circle1 ∧ D ∈ circle2

def perpendicular_line (l1 l2 : Line) (B : ℝ × ℝ) : Prop :=
  B ∈ l2 ∧ l1.point1 ∈ l2 ∧ l1.point2 ∈ l2 ∧ perpendicular l1 l2

def intersecting_points (l : Line) (circle1 circle2 : Circle) (E F : ℝ × ℝ) : Prop :=
  E ∈ l ∧ F ∈ l ∧ E ∈ circle1 ∧ F ∈ circle2

-- Define what it means to be a rhombus
def is_rhombus (C E D F : ℝ × ℝ) : Prop :=
  let CE := dist C E
  let ED := dist E D
  let DF := dist D F
  let FC := dist F C
  CE = ED ∧ ED = DF ∧ DF = FC

-- State the theorem
theorem rhombus_from_circles_and_secant 
  (circle1 circle2 : Circle) (C D E F B : ℝ × ℝ) (l1 l2 : Line) :
  common_secant circle1 circle2 C D →
  perpendicular_line l1 l2 B →
  intersecting_points l2 circle1 circle2 E F →
  is_rhombus C E D F := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_from_circles_and_secant_l572_57256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l572_57272

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Given points A and B, find the value of k that minimizes AC + BC where C is (0, k) -/
theorem minimize_distance_sum (A B : ℝ × ℝ) (h_A : A = (6, 3)) (h_B : B = (3, -2)) :
  ∃ k : ℝ, k = -1 ∧
  ∀ t : ℝ, distance A (0, t) + distance B (0, t) ≥ distance A (0, k) + distance B (0, k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l572_57272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polygon_with_interior_point_no_side_visible_exists_polygon_with_exterior_point_no_side_visible_l572_57257

/-- A polygon in 2D space --/
structure Polygon where
  vertices : List (Real × Real)
  is_closed : vertices.length ≥ 3

/-- Checks if a line segment is completely visible from a point --/
def is_completely_visible (segment : (Real × Real) × (Real × Real)) (point : Real × Real) : Prop :=
  sorry

/-- Helper function to get the edges of a polygon --/
def edges (P : Polygon) : List ((Real × Real) × (Real × Real)) :=
  sorry

/-- Helper function to check if a point is inside a polygon --/
def is_inside (P : Polygon) (point : Real × Real) : Prop :=
  sorry

/-- Theorem: There exists a polygon and an interior point such that no side is completely visible --/
theorem exists_polygon_with_interior_point_no_side_visible :
  ∃ (P : Polygon) (O : Real × Real),
    (∀ p, p ∈ P.vertices → is_inside P O) ∧
    (∀ side, side ∈ edges P → ¬is_completely_visible side O) := by
  sorry

/-- Theorem: There exists a polygon and an exterior point such that no side is completely visible --/
theorem exists_polygon_with_exterior_point_no_side_visible :
  ∃ (P : Polygon) (O : Real × Real),
    (∀ p, p ∈ P.vertices → ¬is_inside P O) ∧
    (∀ side, side ∈ edges P → ¬is_completely_visible side O) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polygon_with_interior_point_no_side_visible_exists_polygon_with_exterior_point_no_side_visible_l572_57257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l572_57234

/-- Given a circle and a point that is the midpoint of a chord, 
    prove the equation of the line containing the chord. -/
theorem chord_equation :
  ∀ (x y : ℝ), (x - 3)^2 + y^2 = 9 →  -- Circle equation
  (1, 1) ∈ Set.univ →             -- P(1,1) exists in the plane
  (∃ M N : ℝ × ℝ, (1, 1) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧ 
    (M.1 - 3)^2 + M.2^2 = 9 ∧ (N.1 - 3)^2 + N.2^2 = 9) →  -- P is midpoint of chord MN
  ∀ (x y : ℝ), 2*x - y - 1 = 0 ↔ (x, y) ∈ Set.univ ∧ 
    ∃ t : ℝ, x = 1 + t ∧ y = 1 + 2*t :=  -- Line equation
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l572_57234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_at_15_minutes_l572_57220

-- Define the passenger function P(t)
noncomputable def P (t : ℝ) : ℝ :=
  if 20 ≤ t ∧ t ≤ 30 then 500
  else if 10 ≤ t ∧ t < 20 then 500 - 4 * (20 - t)^2
  else 0

-- Define the net revenue function Q(t)
noncomputable def Q (t : ℝ) : ℝ := (2 * P t + 400) / t

theorem max_revenue_at_15_minutes :
  ∀ t : ℝ, 10 ≤ t ∧ t ≤ 30 →
  P 12 = 244 →
  Q 15 = 80 ∧ ∀ s : ℝ, 10 ≤ s ∧ s ≤ 30 → Q s ≤ Q 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_at_15_minutes_l572_57220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_max_distance_l572_57286

/-- Represents the duration of the journey in hours -/
def journey_duration : ℕ := 6

/-- Represents the observation duration of each scientist in hours -/
def observation_duration : ℕ := 1

/-- Represents the distance covered during each scientist's observation in meters -/
def distance_per_observation : ℕ := 1

/-- Represents the maximum possible distance the snail can travel -/
def max_distance : ℕ := 10

/-- Helper function to represent that a scientist observes at a given time -/
def observes (scientist : ℕ) (t : ℝ) : Prop := sorry

/-- Helper function to represent that a scientist observes for a given duration -/
def observes_for (scientist : ℕ) (duration : ℕ) : Prop := sorry

/-- Helper function to represent that a scientist reports a certain distance -/
def reports (scientist : ℕ) (distance : ℕ) : Prop := sorry

/-- Theorem stating the maximum distance the snail can travel given the conditions -/
theorem snail_max_distance :
  ∀ (actual_distance : ℕ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ journey_duration → ∃ scientist : ℕ, observes scientist t) →
  (∀ scientist : ℕ, observes_for scientist observation_duration) →
  (∀ scientist : ℕ, reports scientist distance_per_observation) →
  actual_distance ≤ max_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snail_max_distance_l572_57286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_full_range_iff_a_ge_four_f_full_range_complete_range_l572_57202

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 4 * x + a - 3)

/-- The range of f is all real numbers -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The theorem stating the range of a for which f has full range -/
theorem f_full_range_iff_a_ge_four :
  ∀ a : ℝ, has_full_range a ↔ a ≥ 4 :=
by
  sorry

/-- The theorem stating the complete range of a for which f has full range -/
theorem f_full_range_complete_range :
  ∀ a : ℝ, has_full_range a ↔ 4 ≤ a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_full_range_iff_a_ge_four_f_full_range_complete_range_l572_57202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_ratio_is_2_3_l572_57229

/-- Represents the business partnership between Praveen and Hari --/
structure Partnership where
  praveen_investment : ℤ
  hari_investment : ℤ
  praveen_duration : ℤ
  hari_duration : ℤ

/-- Calculates the ratio of profit shares given a partnership --/
def profit_share_ratio (p : Partnership) : ℚ × ℚ :=
  let praveen_contribution := p.praveen_investment * p.praveen_duration
  let hari_contribution := p.hari_investment * p.hari_duration
  let gcd := Int.gcd praveen_contribution hari_contribution
  (praveen_contribution / gcd, hari_contribution / gcd)

/-- The specific partnership described in the problem --/
def problem_partnership : Partnership where
  praveen_investment := 3360
  hari_investment := 8640
  praveen_duration := 12
  hari_duration := 7

theorem profit_share_ratio_is_2_3 :
  profit_share_ratio problem_partnership = (2, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_share_ratio_is_2_3_l572_57229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l572_57276

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between a plane and a point -/
noncomputable def distance (p : Plane) (pt : Point) : ℝ :=
  (abs ((p.a : ℝ) * pt.x + (p.b : ℝ) * pt.y + (p.c : ℝ) * pt.z + (p.d : ℝ))) /
  Real.sqrt ((p.a^2 : ℝ) + (p.b^2 : ℝ) + (p.c^2 : ℝ))

/-- Check if a plane contains a line defined by two other planes -/
def contains_line (p : Plane) (p1 : Plane) (p2 : Plane) : Prop :=
  ∃ (k l : ℝ), k * (p1.a : ℝ) + l * (p2.a : ℝ) = p.a ∧
                k * (p1.b : ℝ) + l * (p2.b : ℝ) = p.b ∧
                k * (p1.c : ℝ) + l * (p2.c : ℝ) = p.c ∧
                k * (p1.d : ℝ) + l * (p2.d : ℝ) = p.d

/-- The greatest common divisor of four integers -/
def gcd4 (a b c d : ℤ) : ℕ :=
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)

theorem plane_equation :
  ∃ (p : Plane),
    -- p contains the line of intersection
    contains_line p ⟨2, -1, 1, -4⟩ ⟨1, 3, -1, -5⟩ ∧
    -- p is distinct from the given planes
    p ≠ ⟨2, -1, 1, -4⟩ ∧ p ≠ ⟨1, 3, -1, -5⟩ ∧
    -- distance from p to the point (1, -2, 0) is 3/√5
    distance p ⟨1, -2, 0⟩ = 3 / Real.sqrt 5 ∧
    -- p.a > 0
    p.a > 0 ∧
    -- gcd of coefficients is 1
    gcd4 p.a p.b p.c p.d = 1 ∧
    -- p has the equation 8x - y + 7z - 10 = 0
    p = ⟨8, -1, 7, -10⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l572_57276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_in_S_l572_57251

/-- Given an integer d, S is the set of numbers of the form m^2 + dn^2 where m and n are integers -/
def S (d : ℤ) : Set ℤ := {x | ∃ m n : ℤ, x = m^2 + d*n^2}

/-- Main theorem: If p and q are in S, p is prime, and p divides q, then q/p is also in S -/
theorem quotient_in_S (d : ℤ) (p q : ℤ) (hp : Nat.Prime p.natAbs) (hpq : p ∣ q) 
  (hp_in_S : p ∈ S d) (hq_in_S : q ∈ S d) : 
  q / p ∈ S d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_in_S_l572_57251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winning_strategy_l572_57284

/-- Represents a card in the deck -/
structure Card where
  id : Nat

/-- Represents the game state -/
structure GameState where
  player1_deck : List Card
  player2_deck : List Card

/-- Represents a move in the game -/
inductive Move
  | Player1Wins : Move
  | Player2Wins : Move

/-- Defines the beating relation between cards -/
def beats (c1 c2 : Card) : Prop := sorry

/-- Applies a move to the current game state -/
def apply_move (state : GameState) (move : Move) : GameState := sorry

/-- Checks if the game has ended (one player has all cards) -/
def is_final_state (state : GameState) : Prop := 
  state.player1_deck.isEmpty ∨ state.player2_deck.isEmpty

/-- Theorem: There exists a sequence of moves that leads to a final state -/
theorem exists_winning_strategy 
  (n : Nat) 
  (initial_state : GameState) 
  (h_total_cards : initial_state.player1_deck.length + initial_state.player2_deck.length = n)
  (h_distinct_cards : ∀ c1 c2 : Card, c1 ∈ initial_state.player1_deck ∨ c1 ∈ initial_state.player2_deck → 
                                      c2 ∈ initial_state.player1_deck ∨ c2 ∈ initial_state.player2_deck → 
                                      c1 ≠ c2 → c1.id ≠ c2.id) 
  (h_beats_relation : ∀ c1 c2 : Card, c1 ∈ initial_state.player1_deck ∨ c1 ∈ initial_state.player2_deck → 
                                      c2 ∈ initial_state.player1_deck ∨ c2 ∈ initial_state.player2_deck → 
                                      c1 ≠ c2 → beats c1 c2 ∨ beats c2 c1) :
  ∃ (moves : List Move), is_final_state (moves.foldl apply_move initial_state) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_winning_strategy_l572_57284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l572_57232

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

-- State the theorem
theorem f_extrema_on_interval :
  let a : ℝ := -3/4
  let b : ℝ := 1/4
  (∀ x ∈ Set.Icc a b, f x ≥ Real.log 2 + 1/4) ∧
  (∃ x ∈ Set.Icc a b, f x = Real.log 2 + 1/4) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 1/16 + Real.log (7/2)) ∧
  (∃ x ∈ Set.Icc a b, f x = 1/16 + Real.log (7/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l572_57232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_distinct_sums_l572_57244

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a * b = 36

def sum_of_pair (a b : ℕ) : ℕ :=
  a + b

def distinct_sums : List ℕ :=
  [12, 13, 15, 20, 37]

theorem average_of_distinct_sums :
  (List.sum distinct_sums : ℚ) / (distinct_sums.length : ℚ) = 97 / 5 := by
  sorry

#eval (List.sum distinct_sums : ℚ) / (distinct_sums.length : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_distinct_sums_l572_57244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bottles_equation_l572_57230

/-- The number of soda bottles Debby brought to the potluck lunch -/
def total_bottles : ℕ := 10

/-- The number of sodas consumed at the potluck lunch -/
def consumed_sodas : ℕ := 8

/-- The number of bottles Debby took back home -/
def bottles_taken_home : ℕ := 2

/-- Theorem stating that the total number of bottles is equal to
    the sum of consumed sodas and bottles taken home -/
theorem total_bottles_equation : total_bottles = consumed_sodas + bottles_taken_home := by
  rfl

#check total_bottles_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bottles_equation_l572_57230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_count_correct_l572_57203

/-- Given the ratio of pencils : pens : exercise books and the number of pencils and exercise books, calculate the number of pens -/
def pen_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
  (pencil_count : ℕ) (book_count : ℕ) : ℕ :=
  let ratio_factor := pencil_count / pencil_ratio
  pen_ratio * ratio_factor

theorem pen_count_correct (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
  (pencil_count : ℕ) (book_count : ℕ) : 
  pen_count pencil_ratio pen_ratio book_ratio pencil_count book_count = 40 :=
by
  -- Assume the given ratio
  have h1 : pencil_ratio = 14 ∧ pen_ratio = 4 ∧ book_ratio = 3 := by sorry
  -- Assume the given counts
  have h2 : pencil_count = 140 ∧ book_count = 30 := by sorry
  -- Define the ratio factor
  let ratio_factor := pencil_count / pencil_ratio
  -- Prove that the ratio factor is consistent for both pencils and books
  have h3 : ratio_factor = pencil_count / pencil_ratio ∧ 
            ratio_factor = book_count / book_ratio := by sorry
  -- Calculate the number of pens
  have h4 : pen_count pencil_ratio pen_ratio book_ratio pencil_count book_count = pen_ratio * ratio_factor := by rfl
  -- Prove that the result is 40
  sorry

#eval pen_count 14 4 3 140 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_count_correct_l572_57203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_flip_probability_l572_57224

-- Define a fair coin
def fair_coin : Fin 2 → ℚ
| 0 => 1/2  -- Probability of heads
| 1 => 1/2  -- Probability of tails

-- Define a sequence of 8 coin flips
def eight_flips : Fin 8 → Fin 2
| _ => 0  -- We don't need to specify the actual sequence for this proof

-- Theorem statement
theorem ninth_flip_probability :
  (∀ (i : Fin 8), fair_coin (eight_flips i) = 1/2) →
  fair_coin 0 = 1/2 :=
by
  intro h
  exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_flip_probability_l572_57224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l572_57223

theorem problem_solution (a b : ℝ) (h : Set.toFinset {a, b/a, 1} = Set.toFinset {a^2, a+b, 0}) : 
  a^2004 + b^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l572_57223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l572_57209

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 10

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log (3/4))
noncomputable def b : ℝ := f (1/4)
noncomputable def c : ℝ := f (Real.tan (1/3))

-- Theorem statement
theorem order_of_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l572_57209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_union_elements_l572_57270

def A : Finset ℕ := {2, 0, 1, 8}

def B : Finset ℕ := A.image (· * 2)

theorem sum_of_union_elements : (A ∪ B).sum id = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_union_elements_l572_57270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_cube_root_of_four_l572_57278

-- Define the floor function (integer part)
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the equation
def equation (x : ℝ) : Prop := x^3 - (floor x : ℝ) = 3

-- State the theorem
theorem solution_is_cube_root_of_four :
  ∃ (x : ℝ), equation x ∧ x = Real.rpow 4 (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_cube_root_of_four_l572_57278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l572_57271

-- Define the function for the expansion
noncomputable def f (x : ℝ) : ℝ := (x.sqrt / 2 - 2 / x) ^ 7

-- Define a as the coefficient of x^2 in the expansion of f(x)
-- We'll use a specific value for 'a' based on the problem solution
noncomputable def a : ℝ := -7 / 32

-- State the theorem
theorem integral_value : 
  ∫ x in (1)..((-32) * a), (Real.exp x - 1 / x) = Real.exp 7 - Real.log 7 - Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l572_57271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l572_57245

theorem triangle_problem (A B C a b c : ℝ) : 
  b = 3 → c = 1 → Real.sin A = Real.sin (2 * B) → 
  (∃ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →
  (a / Real.sin A = b / Real.sin B) →
  a = 2 * Real.sqrt 3 ∧ 
  Real.sin (A + Real.pi/4) = (4 - Real.sqrt 2) / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l572_57245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_from_area_and_radius_l572_57215

/-- Given a circle with radius 18 meters and a sector with an area of 118.8 square meters,
    the central angle of the sector is approximately 42.048 degrees. -/
theorem sector_angle_from_area_and_radius :
  let r : ℝ := 18
  let area : ℝ := 118.8
  let θ : ℝ := (area * 360) / (π * r^2)
  ∃ ε > 0, abs (θ - 42.048) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_from_area_and_radius_l572_57215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equation_solutions_l572_57283

theorem product_equation_solutions : 
  {(x, y, z) : ℕ+ × ℕ+ × ℕ+ | (1 + 1 / x.val) * (1 + 1 / y.val) * (1 + 1 / z.val) = 2} = 
  {(2, 4, 15), (2, 5, 9), (2, 6, 7), (3, 3, 8), (3, 4, 5)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equation_solutions_l572_57283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_two_thirds_l572_57241

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * x - 2

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 2) / 3

-- Theorem statement
theorem sum_of_solutions_is_two_thirds :
  ∃ (x₁ x₂ : ℝ), 
    (g_inv x₁ = g (x₁^2)) ∧ 
    (g_inv x₂ = g (x₂^2)) ∧ 
    (∀ x, g_inv x = g (x^2) → x = x₁ ∨ x = x₂) ∧
    (x₁ + x₂ = 2/3) := by
  sorry

#check sum_of_solutions_is_two_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_two_thirds_l572_57241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l572_57259

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 7 * Real.pi / 12 ∧ b = 13 * Real.pi / 12 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ c d, (∀ x y, c ≤ x ∧ x < y ∧ y ≤ d → f x < f y) →
    (b - a) ≤ (d - c) ∨ (∃ k : ℤ, c = a + k * Real.pi ∧ d = b + k * Real.pi)) :=
by
  sorry

#check monotonic_increasing_interval_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l572_57259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_length_PQ_is_sqrt_3_l572_57205

/-- The length of the line segment PQ is √3, where P is the intersection of the circle ρ = 2sin(θ) 
    and the ray θ = π/3, and Q is the intersection of the curve ρsin(θ) = 3√3/(ρcos(θ)) and the 
    ray θ = π/3 -/
theorem length_PQ : ℝ → Prop :=
  fun length =>
    ∃ (ρ₁ ρ₂ : ℝ),
      -- P is on the circle ρ = 2sin(θ) at θ = π/3
      ρ₁ = 2 * Real.sin (Real.pi/3) ∧
      -- Q is on the curve ρsin(θ) = 3√3/(ρcos(θ)) at θ = π/3
      ρ₂ * Real.sin (Real.pi/3) = 3 * Real.sqrt 3 / (ρ₂ * Real.cos (Real.pi/3)) ∧
      -- The length of PQ is the absolute difference of their ρ values
      length = abs (ρ₁ - ρ₂) ∧
      -- The length is equal to √3
      length = Real.sqrt 3

-- Proof
theorem length_PQ_is_sqrt_3 : length_PQ (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_length_PQ_is_sqrt_3_l572_57205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_is_two_l572_57213

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The common ratio of a geometric sequence -/
def geometric_ratio (a b c : ℚ) : ℚ := b / a

/-- Theorem stating that under given conditions, the geometric ratio is 2 -/
theorem geometric_ratio_is_two (seq : ArithmeticSequence) 
  (h_geom : (seq.a 1) * (seq.a 4) = (seq.a 2) ^ 2) :
  geometric_ratio (seq.a 1) (seq.a 2) (seq.a 4) = 2 := by
  sorry

#check geometric_ratio_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_ratio_is_two_l572_57213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l572_57281

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (θ : ℝ) (h_d : d = 16) (h_θ : θ = π/4) :
  let r := d / (2 * Real.sqrt 2)
  (4/3) * Real.pi * r^3 = (4096 * Real.sqrt 2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l572_57281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l572_57206

theorem matrix_vector_computation 
  (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (a b : Fin 2 → ℝ) 
  (h1 : N.mulVec a = ![3, 2]) 
  (h2 : N.mulVec b = ![4, 1]) : 
  N.mulVec (2 • a - 4 • b) = ![-10, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l572_57206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_cube_volume_ratio_l572_57225

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- A cube whose vertices are the centers of the faces of a regular tetrahedron -/
structure CentroidCube where
  tetrahedron : RegularTetrahedron

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedronVolume (t : RegularTetrahedron) : ℝ :=
  (t.sideLength ^ 3 * Real.sqrt 2) / 12

/-- The volume of a centroid cube -/
noncomputable def centroidCubeVolume (c : CentroidCube) : ℝ :=
  (c.tetrahedron.sideLength ^ 3 * Real.sqrt 2) / 4

/-- The theorem stating that the ratio of the volume of a regular tetrahedron
    to the volume of its centroid cube is 1/3 -/
theorem tetrahedron_centroid_cube_volume_ratio :
    ∀ (t : RegularTetrahedron),
      tetrahedronVolume t / centroidCubeVolume ⟨t⟩ = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_centroid_cube_volume_ratio_l572_57225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transformation_l572_57288

-- Define the triangles
def triangle_DEF : List (ℝ × ℝ) := [(0, 0), (0, 10), (14, 0)]
def triangle_DEF_prime : List (ℝ × ℝ) := [(20, 20), (30, 20), (20, 8)]

-- Define the rotation function
noncomputable def rotate (θ : ℝ) (p q : ℝ) (x y : ℝ) : ℝ × ℝ :=
  let x' := (x - p) * Real.cos θ - (y - q) * Real.sin θ + p
  let y' := (x - p) * Real.sin θ + (y - q) * Real.cos θ + q
  (x', y')

-- Define the theorem
theorem rotation_transformation (n p q : ℝ) :
  0 < n → n < 180 →
  (∀ (x y : ℝ), (x, y) ∈ triangle_DEF →
    rotate (n * π / 180) p q x y ∈ triangle_DEF_prime) →
  n + p + q = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transformation_l572_57288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l572_57249

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A = 3 * t.C ∧
  t.c = 6 ∧
  (2 * t.a - t.c) * Real.cos t.B - t.b * Real.cos t.C = 0

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem area_of_triangle (t : Triangle) 
  (h : triangle_conditions t) : 
  triangle_area t = 18 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l572_57249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l572_57200

theorem min_omega_value (ω φ T : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < π)
  (h4 : T > 0)
  (h5 : ∀ x, Real.cos (ω * x + φ) = Real.cos (ω * (x + T) + φ))
  (h6 : ∀ t, t > 0 ∧ (∀ x, Real.cos (ω * x + φ) = Real.cos (ω * (x + t) + φ)) → T ≤ t)
  (h7 : Real.cos (ω * T + φ) = Real.sqrt 3 / 2)
  (h8 : Real.cos (ω * π / 9 + φ) = 0) :
  ω ≥ 3 ∧ ∃ ω₀, ω₀ = 3 ∧ ω₀ > 0 ∧
    ∃ φ₀ T₀, 0 < φ₀ ∧ φ₀ < π ∧ T₀ > 0 ∧
    (∀ x, Real.cos (ω₀ * x + φ₀) = Real.cos (ω₀ * (x + T₀) + φ₀)) ∧
    (∀ t, t > 0 ∧ (∀ x, Real.cos (ω₀ * x + φ₀) = Real.cos (ω₀ * (x + t) + φ₀)) → T₀ ≤ t) ∧
    Real.cos (ω₀ * T₀ + φ₀) = Real.sqrt 3 / 2 ∧
    Real.cos (ω₀ * π / 9 + φ₀) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l572_57200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_3_to_5_is_22_l572_57216

/-- The velocity function of the body -/
def velocity (t : ℝ) : ℝ := 2 * t + 3

/-- The distance traveled by the body between two time points -/
noncomputable def distance_traveled (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, velocity t

/-- Theorem stating that the distance traveled between 3 and 5 seconds is 22 meters -/
theorem distance_3_to_5_is_22 :
  distance_traveled 3 5 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_3_to_5_is_22_l572_57216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40pi_l572_57211

-- Define the radii of the two smaller circles
def r₁ : ℝ := 4
def r₂ : ℝ := 5

-- Define the radius of the larger circle
def R : ℝ := r₁ + r₂

-- Define the area of the shaded region
noncomputable def shaded_area : ℝ := Real.pi * R^2 - Real.pi * r₁^2 - Real.pi * r₂^2

theorem shaded_area_is_40pi :
  shaded_area = 40 * Real.pi := by
  -- Expand the definition of shaded_area
  unfold shaded_area
  -- Simplify the expression
  simp [R, r₁, r₂]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40pi_l572_57211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_intervals_max_value_on_interval_l572_57235

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x - Real.pi / 6)

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
  ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T' := by
  sorry

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 3 → 
  ∀ y : ℝ, k * Real.pi - Real.pi / 6 ≤ y ∧ y < x → f y < f x := by
  sorry

-- Theorem for maximum value on [0, π/2]
theorem max_value_on_interval :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_monotonically_increasing_intervals_max_value_on_interval_l572_57235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l572_57238

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  -- Base dimensions
  ef : ℝ
  fg : ℝ
  -- Height from vertex P to base
  pe : ℝ
  -- Distance from P to a corner of the base
  pf : ℝ

/-- Calculates the volume of a pyramid with a rectangular base -/
noncomputable def pyramidVolume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.ef * p.fg * p.pe

/-- The main theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  ∃ (p : RectangularBasePyramid),
    p.ef = 10 ∧
    p.fg = 6 ∧
    p.pf = 20 ∧
    p.pe = 10 * Real.sqrt 3 ∧
    pyramidVolume p = 200 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l572_57238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l572_57260

/-- The function f(x) = 1 / sqrt(-x^2 + 2x + 3) + ln(x^2 - 1) -/
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (-x^2 + 2*x + 3) + Real.log (x^2 - 1)

/-- The domain of f is {x | 1 < x < 3} -/
theorem domain_of_f :
  Set.Ioo 1 3 = {x : ℝ | -x^2 + 2*x + 3 > 0 ∧ x^2 - 1 > 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l572_57260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_negative_460_l572_57250

/-- The set of angles with the same terminal side as -460° -/
def SameTerminalSide : Set ℝ :=
  {θ | ∃ k : ℤ, θ = k * 360 + 260}

/-- Angles with the same terminal side differ by an integer multiple of 360° -/
axiom same_terminal_side (α β : ℝ) :
  (∃ k : ℤ, α = β + k * 360) ↔ α ∈ SameTerminalSide ∧ β ∈ SameTerminalSide

theorem terminal_side_negative_460 :
  SameTerminalSide = {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_negative_460_l572_57250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l572_57258

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 + 5*x - 2) / (x^2 - 5*x + 6)

theorem h_domain :
  {x : ℝ | ∃ y, h x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l572_57258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l572_57242

/-- Inverse proportion function passing through (-1, -2) -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k - 1) / x

theorem inverse_proportion_k_value :
  ∃ k : ℝ, inverse_proportion k (-1) = -2 ∧ k = 3 := by
  -- We'll use 3 as the value of k
  use 3
  constructor
  · -- Prove that inverse_proportion 3 (-1) = -2
    simp [inverse_proportion]
    ring
  · -- Prove that k = 3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l572_57242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l572_57295

/-- Circle O with center (0,0) and radius 2 -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Circle C with center (3,3) and radius √10 -/
def circle_C (x y : ℝ) : Prop := (x-3)^2 + (y-3)^2 = 10

/-- The distance between the centers of circles O and C -/
noncomputable def distance_OC : ℝ := Real.sqrt 18

/-- The length of the common chord of circles O and C -/
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 2

theorem circles_properties :
  distance_OC = 3 * Real.sqrt 2 ∧
  common_chord_length = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l572_57295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_implies_m_is_sqrt_two_sqrt_two_satisfies_conditions_l572_57280

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 1) * x^m

-- Define what it means for f to be a power function
def is_power_function (m : ℝ) : Prop := m^2 - 1 = 1

-- Define what it means for f to be increasing on (0, +∞)
def is_increasing_on_positive_reals (m : ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f m x < f m y

-- State the theorem
theorem power_function_increasing_implies_m_is_sqrt_two (m : ℝ) :
  is_power_function m → is_increasing_on_positive_reals m → m = Real.sqrt 2 := by
  sorry

-- Prove that √2 satisfies the conditions
theorem sqrt_two_satisfies_conditions :
  is_power_function (Real.sqrt 2) ∧ is_increasing_on_positive_reals (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_implies_m_is_sqrt_two_sqrt_two_satisfies_conditions_l572_57280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_fifteen_solution_l572_57247

theorem square_root_fifteen_solution :
  ∃ x : ℝ, (x = Real.sqrt 15 ∨ x = -Real.sqrt 15) ∧ 45 - 3 * x^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_fifteen_solution_l572_57247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_product_constant_l572_57287

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Predicate to check if a point is on the ellipse -/
def IsOnEllipse (E : Ellipse) (P : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is not an endpoint of the major axis -/
def NotEndpoint (E : Ellipse) (P : ℝ × ℝ) : Prop := sorry

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  P : ℝ × ℝ
  on_ellipse : IsOnEllipse E P
  not_endpoint : NotEndpoint E P

/-- The angle PF₁F₂ -/
noncomputable def angle_PF₁F₂ (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- The angle PF₂F₁ -/
noncomputable def angle_PF₂F₁ (E : Ellipse) (P : PointOnEllipse E) : ℝ := sorry

/-- The theorem statement -/
theorem ellipse_angle_product_constant (E : Ellipse) :
  ∃ (c : ℝ), ∀ (P : PointOnEllipse E),
    Real.tan (angle_PF₁F₂ E P / 2) * Real.tan (angle_PF₂F₁ E P / 2) = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_product_constant_l572_57287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_triple_matrix_l572_57231

def double_triple_transform (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    M * A = !![2 * A 0 0, 2 * A 0 1;
                3 * A 1 0, 3 * A 1 1]

theorem unique_double_triple_matrix :
  ∃! M : Matrix (Fin 2) (Fin 2) ℝ, double_triple_transform M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_triple_matrix_l572_57231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_equation_represents_hyperbola_l572_57294

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 5 * Real.tan θ * (1 / Real.cos θ) + 2

-- Define the Cartesian equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^4 + 2*x^2*y^2 + y^4 = 25*y^2

-- Theorem stating the equivalence of the two equations
theorem polar_to_cartesian_equivalence :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ → 
    cartesian_equation x y :=
by
  sorry

-- Theorem stating that the equation represents a hyperbola
theorem equation_represents_hyperbola :
  ∀ (x y : ℝ),
    cartesian_equation x y →
    ∃ (a b : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_equivalence_equation_represents_hyperbola_l572_57294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inheritance_tax_problem_l572_57210

theorem inheritance_tax_problem :
  ∃ (inheritance : ℕ),
    (let federal_tax_rate : ℚ := 25 / 100;
     let state_tax_rate : ℚ := 15 / 100;
     let federal_tax := (inheritance : ℚ) * federal_tax_rate;
     let remaining_after_federal := (inheritance : ℚ) - federal_tax;
     let state_tax := remaining_after_federal * state_tax_rate;
     let total_tax := federal_tax + state_tax;
     total_tax = 15000 ∧ inheritance = 41379) := by
  use 41379
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inheritance_tax_problem_l572_57210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l572_57253

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The number of terms in the geometric series ending with 6561 -/
def numTerms : ℕ := 9

theorem geometric_series_sum :
  let a : ℝ := 1
  let r : ℝ := -3
  let lastTerm : ℝ := 6561
  r^(numTerms - 1) = lastTerm →
  geometricSum a r numTerms = 4921.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l572_57253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l572_57290

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, deriv (f a b c) x = 0 → x = 2) →
  f a b c 2 = c - 16 →
  (∃ x : ℝ, f a b c x = 28 ∧ ∀ y : ℝ, f a b c y ≤ 28) →
  (a = 1 ∧ b = -12) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f a b c x ≥ -4) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f a b c x = -4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l572_57290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x60_is_11_l572_57204

/-- The polynomial P(x) = (x - 1)(x^2 - 2)(x^3 - 3) ... (x^10 - 10)(x^11 - 11) -/
def P (x : ℝ) : ℝ := (Finset.range 11).prod (fun i => x^(i+1) - (i+1))

/-- The coefficient of x^60 in the expansion of P(x) -/
noncomputable def coeff_x60 (P : ℝ → ℝ) : ℝ :=
  (deriv^[60] P 0) / Nat.factorial 60

theorem coeff_x60_is_11 : coeff_x60 P = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coeff_x60_is_11_l572_57204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l572_57217

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 6 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 7 - Real.sqrt 6

-- Theorem statement
theorem order_of_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l572_57217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_one_hour_l572_57266

/-- Calculates the time taken for a round trip given the rowing speed, river speed, and total distance. -/
noncomputable def roundTripTime (rowingSpeed riverSpeed totalDistance : ℝ) : ℝ :=
  let upstreamSpeed := rowingSpeed - riverSpeed
  let downstreamSpeed := rowingSpeed + riverSpeed
  let oneWayDistance := totalDistance / 2
  (oneWayDistance / upstreamSpeed) + (oneWayDistance / downstreamSpeed)

/-- Proves that given the specific conditions, the round trip time is 1 hour. -/
theorem round_trip_time_is_one_hour :
  roundTripTime 6 2 (5333333333333333 / 1000000000000000) = 1 := by
  -- Unfold the definition of roundTripTime
  unfold roundTripTime
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_one_hour_l572_57266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_theorem_l572_57279

/-- Represents a pile of stones -/
structure Pile where
  size : ℕ

/-- Represents the state of the game -/
structure GameState where
  piles : List Pile

/-- Checks if all piles in the game state satisfy the size constraint -/
def validState (state : GameState) : Prop :=
  ∀ p1 p2, p1 ∈ state.piles → p2 ∈ state.piles →
    p1.size < 2 * p2.size ∧ p2.size < 2 * p1.size

/-- Checks if the total number of stones in all piles is correct -/
def correctTotalStones (state : GameState) : Prop :=
  (state.piles.map Pile.size).sum = 660

/-- Defines a valid final state of the game -/
def validFinalState (state : GameState) : Prop :=
  validState state ∧ correctTotalStones state

/-- The maximum number of piles that can be formed -/
def maxPiles : ℕ := 30

/-- The main theorem to be proved -/
theorem max_piles_theorem :
  ∀ state : GameState,
    validFinalState state →
    state.piles.length ≤ maxPiles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_theorem_l572_57279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_price_l572_57227

/-- Calculates the final price of an item for an employee purchase given the wholesale cost, store markup percentage, and employee discount percentage. -/
noncomputable def employee_purchase_price (wholesale_cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + markup_percent / 100)
  retail_price * (1 - discount_percent / 100)

/-- Theorem stating that for a video recorder with a $200 wholesale cost, 20% markup, and 5% employee discount, the final price is $228. -/
theorem video_recorder_price :
  employee_purchase_price 200 20 5 = 228 := by
  -- Unfold the definition of employee_purchase_price
  unfold employee_purchase_price
  -- Simplify the expression
  simp
  -- The proof is incomplete, so we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_recorder_price_l572_57227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_area_l572_57285

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is right-angled -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  (distance p1 p2)^2 + (distance p2 p3)^2 = (distance p1 p3)^2

/-- Calculate the area of a triangle given its three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

theorem ellipse_right_triangle_area 
  (e : Ellipse) 
  (f1 f2 p : Point) 
  (h1 : e.a = 4 ∧ e.b = 3) 
  (h2 : isOnEllipse e p) 
  (h3 : isRightTriangle p f1 f2) : 
  triangleArea p f1 f2 = 9 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_area_l572_57285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_false_l572_57268

-- Define a point in 2D space
structure Point :=
  (x y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of having equal diagonals
def has_equal_diagonals (q : Quadrilateral) : Prop :=
  (q.A.x - q.C.x)^2 + (q.A.y - q.C.y)^2 = (q.B.x - q.D.x)^2 + (q.B.y - q.D.y)^2

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  -- A rectangle is a quadrilateral with four right angles
  -- This is a simplified definition for the purpose of this problem
  (q.B.x - q.A.x) * (q.C.x - q.B.x) + (q.B.y - q.A.y) * (q.C.y - q.B.y) = 0 ∧
  (q.C.x - q.B.x) * (q.D.x - q.C.x) + (q.C.y - q.B.y) * (q.D.y - q.C.y) = 0 ∧
  (q.D.x - q.C.x) * (q.A.x - q.D.x) + (q.D.y - q.C.y) * (q.A.y - q.D.y) = 0 ∧
  (q.A.x - q.D.x) * (q.B.x - q.A.x) + (q.A.y - q.D.y) * (q.B.y - q.A.y) = 0

-- Theorem stating that the inverse proposition is false
theorem inverse_proposition_is_false :
  ¬(∀ q : Quadrilateral, has_equal_diagonals q → is_rectangle q) :=
by
  sorry -- The proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_is_false_l572_57268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l572_57214

/-- The function g(x) with parameter c -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - 5*x + 6)

/-- Theorem stating that g(x) has exactly one vertical asymptote iff c = 0 or c = 2 -/
theorem g_has_one_vertical_asymptote (c : ℝ) :
  (∃! x : ℝ, ¬∃ y : ℝ, g c x = y) ↔ c = 0 ∨ c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_one_vertical_asymptote_l572_57214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_l572_57293

/-- The function f(x) defined on (0, +∞) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (x - 1) + a * (x - 1)

/-- The theorem stating the condition for monotonicity of f(x) -/
theorem f_monotonic_iff (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) ↔ a ≥ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_l572_57293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l572_57291

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 * Real.tan x + Real.cos x ^ 4 * (1 / Real.tan x)

theorem range_of_f :
  Set.range f = {y : ℝ | y ≤ -1/2 ∨ y ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l572_57291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l572_57263

/-- Given three points A, B, and C in a 2D plane, and a point P satisfying certain conditions,
    prove that the dot product of AB and AC is 4, and the sum of lambda and mu is 1/3. -/
theorem vector_problem (A B C P : ℝ × ℝ) (lambda mu : ℝ) : 
  A = (1, -1) → 
  B = (3, 0) → 
  C = (2, 1) → 
  P.1 - A.1 = lambda * (B.1 - A.1) + mu * (C.1 - A.1) → 
  P.2 - A.2 = lambda * (B.2 - A.2) + mu * (C.2 - A.2) → 
  (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 → 
  (P.1 - A.1) * (C.1 - A.1) + (P.2 - A.2) * (C.2 - A.2) = 3 → 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 4 ∧ lambda + mu = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l572_57263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_tank_capacity_l572_57296

/-- Represents the capacity of a gas tank in gallons -/
def TankCapacity (capacity : ℝ) : Prop := capacity > 0

/-- Represents the fraction of a full tank -/
def FractionOfTank (fraction : ℝ) : Prop := 0 ≤ fraction ∧ fraction ≤ 1

theorem gas_tank_capacity 
  (distance_to_work : ℝ)
  (fuel_efficiency : ℝ)
  (remaining_fraction : ℝ)
  (h1 : distance_to_work = 10)
  (h2 : fuel_efficiency = 5)
  (h3 : FractionOfTank remaining_fraction)
  (h4 : remaining_fraction = 2/3)
  : TankCapacity 12 :=
by
  sorry

#check gas_tank_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_tank_capacity_l572_57296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_chips_remain_even_not_one_chip_after_even_turns_min_turns_for_1000_chips_l572_57201

/-- Represents the state of the chip game -/
structure GameState where
  total_chips : ℕ
  black_chips : ℕ
  white_chips : ℕ
  deriving Repr

/-- Represents a player's turn in the game -/
inductive Turn
  | FirstPlayer
  | SecondPlayer

/-- Defines the game rules and state transitions -/
def next_state (state : GameState) (turn : Turn) : GameState :=
  match turn with
  | Turn.FirstPlayer => 
      { total_chips := state.white_chips
      , black_chips := 0  -- All black chips adjacent to white are removed
      , white_chips := state.white_chips }
  | Turn.SecondPlayer => 
      { total_chips := state.black_chips
      , black_chips := state.black_chips
      , white_chips := 0 }  -- All white chips adjacent to black are removed

/-- Applies the next_state function n times to the initial state -/
def apply_turns (initial_state : GameState) (n : ℕ) : GameState :=
  match n with
  | 0 => initial_state
  | n + 1 => next_state (apply_turns initial_state n) (if n % 2 = 0 then Turn.FirstPlayer else Turn.SecondPlayer)

/-- Theorem: After an even number of turns, the total number of chips remains even if it started even -/
theorem even_chips_remain_even 
  (initial_state : GameState) 
  (n : ℕ) 
  (h_even_initial : Even initial_state.total_chips) 
  (h_even_turns : Even n) :
  Even (apply_turns initial_state n).total_chips :=
by sorry

/-- Theorem: It's impossible to have one chip remaining after an even number of turns if started with an even number -/
theorem not_one_chip_after_even_turns 
  (initial_state : GameState) 
  (n : ℕ) 
  (h_even_initial : Even initial_state.total_chips) 
  (h_even_turns : Even n) :
  (apply_turns initial_state n).total_chips ≠ 1 :=
by sorry

/-- Theorem: The minimum number of turns to reach one chip from 1000 chips is 8 -/
theorem min_turns_for_1000_chips :
  ∃ (initial_state : GameState),
    initial_state.total_chips = 1000 ∧
    (apply_turns initial_state 8).total_chips = 1 ∧
    ∀ (m : ℕ), m < 8 → (apply_turns initial_state m).total_chips > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_chips_remain_even_not_one_chip_after_even_turns_min_turns_for_1000_chips_l572_57201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l572_57265

def our_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a n * a (n + 1) = -2

theorem eighth_term_value (a : ℕ → ℤ) (h : our_sequence a) : a 8 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_value_l572_57265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_numbers_sum_l572_57237

def is_permutation (p : List ℕ) : Prop :=
  p.length = 12 ∧ p.toFinset = Finset.range 12

def circular_consecutive_sums (p : List ℕ) : List ℕ :=
  List.zipWith3 (fun a b c => a + b + c) p (p.rotateLeft 1) (p.rotateLeft 2)

theorem clock_numbers_sum (p : List ℕ) (h : is_permutation p) :
  ∃ s ∈ circular_consecutive_sums p, s > 20 := by
  sorry

#check clock_numbers_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_numbers_sum_l572_57237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l572_57255

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := log10 (x + 3)
noncomputable def g (x : ℝ) : ℝ := (10 : ℝ) ^ x

-- Define the transformation
noncomputable def transform (x y : ℝ) : ℝ × ℝ := (y, log10 (x + 3))

theorem graph_transformation :
  ∀ x y : ℝ, f x = y ↔ transform (g⁻¹ (y + 3)) y = (x, y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l572_57255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l572_57228

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = Real.sqrt 2) ∧
  (∀ (x : ℝ), f (x + Real.pi / 6) = f (-x + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l572_57228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_a_amount_l572_57262

noncomputable def liquid_a : ℝ → ℝ := λ x => 4 * x

noncomputable def liquid_b : ℝ → ℝ := λ x => x

noncomputable def total_mixture : ℝ → ℝ := λ x => liquid_a x + liquid_b x

noncomputable def removed_a : ℝ := 4 / 5 * 40

noncomputable def removed_b : ℝ := 1 / 5 * 40

noncomputable def new_a : ℝ → ℝ := λ x => liquid_a x - removed_a

noncomputable def new_b : ℝ → ℝ := λ x => liquid_b x - removed_b + 40

theorem initial_liquid_a_amount : 
  ∃ x : ℝ, 
    (liquid_a x) / (liquid_b x) = 4 / 1 ∧ 
    (new_a x) / (new_b x) = 2 / 3 ∧ 
    liquid_a x = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_a_amount_l572_57262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_four_implies_m_four_l572_57298

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The focal length of an ellipse -/
noncomputable def focal_length (e : Ellipse) : ℝ := 2 * Real.sqrt (e.a^2 - e.b^2)

/-- The ellipse from the problem -/
noncomputable def problem_ellipse (m : ℝ) : Ellipse where
  a := (10 - m).sqrt
  b := (m - 2).sqrt
  h_positive := by
    sorry

theorem ellipse_focal_length_four_implies_m_four :
  ∀ m : ℝ, 2 < m ∧ m < 10 →
  focal_length (problem_ellipse m) = 4 →
  m = 4 := by
    sorry

#check ellipse_focal_length_four_implies_m_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_four_implies_m_four_l572_57298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l572_57275

theorem vector_difference_magnitude
  (a b : ℝ × ℝ)
  (angle_condition : a.1 * b.1 + a.2 * b.2 = -2)
  (magnitude_a : a.1^2 + a.2^2 = 4)
  (magnitude_b : b.1^2 + b.2^2 = 4) :
  (a.1 - 3*b.1)^2 + (a.2 - 3*b.2)^2 = 52 := by
  sorry

#check vector_difference_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l572_57275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l572_57222

theorem not_prime_sum_product (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬(Nat.Prime (Int.natAbs (a * b + c * d))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_sum_product_l572_57222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_product_property_l572_57236

/-- The property that for any partition of S_n into two subsets, 
    there exists a subset containing three numbers a, b, c such that ab = c -/
def has_product_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n + 1) → A ∩ B = ∅ →
    ∃ (a b c : ℕ), ((a ∈ A ∧ b ∈ A ∧ c ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ c ∈ B)) ∧
      a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ a * b = c

/-- 243 is the smallest natural number greater than 3 with the product property -/
theorem smallest_n_with_product_property :
  (∀ n : ℕ, n > 3 → n < 243 → ¬(has_product_property n)) ∧
  has_product_property 243 := by
  sorry

#check smallest_n_with_product_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_product_property_l572_57236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_0123_l572_57277

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our function f(x) = ⌊x⌋ + 1
noncomputable def f (x : ℝ) : ℤ := floor x + 1

-- Define the domain
def domain : Set ℝ := {x | -0.5 < x ∧ x < 2.5}

-- Define the range
def range : Set ℤ := {0, 1, 2, 3}

-- Theorem statement
theorem f_range_is_0123 : 
  {y | ∃ x ∈ domain, f x = y} = range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_0123_l572_57277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_finish_work_l572_57261

/-- The number of days it takes for p to finish the remaining work -/
noncomputable def days_for_p_to_finish (p q r : ℝ) : ℝ :=
  let work_rate_q : ℝ := 1 / q
  let work_rate_r : ℝ := 1 / r
  let work_done_by_qr : ℝ := (work_rate_q + work_rate_r) * 3
  let remaining_work : ℝ := 1 - work_done_by_qr
  remaining_work * p

theorem days_to_finish_work 
  (p q r : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hr : r > 0) 
  (hp_rate : p = 24) 
  (hq_rate : q = 9) 
  (hr_rate : r = 12) : 
  days_for_p_to_finish p q r = 10 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_finish_work_l572_57261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_crosses_one_l572_57207

noncomputable def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => a n + (a n)^2 / 2012

theorem sequence_crosses_one :
  ∃ k : ℕ, k = 2012 ∧ a k < 1 ∧ 1 < a (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_crosses_one_l572_57207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_1_greater_f_9_2_l572_57246

/-- Recursive definition of functions f_n -/
noncomputable def f : ℕ → ℝ → ℝ
| 0, x => x
| (n + 1), x => if n % 2 = 0 then Real.rpow 3 (f (n/2) x) else Real.rpow 2 (f (n/2) x)

/-- Theorem stating that f_10(1) > f_9(2) -/
theorem f_10_1_greater_f_9_2 : f 10 1 > f 9 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_1_greater_f_9_2_l572_57246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_types_65400_words_per_week_l572_57297

/-- Represents Carl's typing schedule for a day --/
structure TypingDay where
  speed : Nat  -- Words per minute
  hours : Nat  -- Total hours before breaks
  rest : Nat   -- Minutes of rest after each hour
  breaks : Nat -- Number of breaks

/-- Calculates the total words typed in a day --/
def wordsTypedInDay (day : TypingDay) : Nat :=
  let actualMinutes := day.hours * 60 - day.rest * day.breaks
  actualMinutes * day.speed

/-- Carl's weekly typing schedule --/
def carlsWeek : List TypingDay := [
  -- Monday, Wednesday, Friday
  { speed := 50, hours := 5, rest := 10, breaks := 4 },
  { speed := 50, hours := 5, rest := 10, breaks := 4 },
  { speed := 50, hours := 5, rest := 10, breaks := 4 },
  -- Tuesday, Thursday
  { speed := 40, hours := 3, rest := 15, breaks := 2 },
  { speed := 40, hours := 3, rest := 15, breaks := 2 },
  -- Saturday
  { speed := 60, hours := 4, rest := 0, breaks := 0 },
  -- Sunday
  { speed := 0, hours := 0, rest := 0, breaks := 0 }
]

/-- Theorem: Carl types 65,400 words in a week --/
theorem carl_types_65400_words_per_week : 
  (carlsWeek.map wordsTypedInDay).sum = 65400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_types_65400_words_per_week_l572_57297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_arithmetic_sum_l572_57218

theorem closest_perfect_square_to_arithmetic_sum : ∃ (n : ℕ), n > 0 ∧ 
  (let first_term := 8
   let common_diff := 8
   let last_term := 8040
   let num_terms := (last_term - first_term) / common_diff + 1
   let sum := (num_terms * (first_term + last_term)) / 2
   (n^2 : ℤ) = sum + 1 ∧ 
   ∀ (m : ℕ), m > 0 → m ≠ n → |((m^2 : ℤ) - sum)| > |((n^2 : ℤ) - sum)|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_arithmetic_sum_l572_57218
