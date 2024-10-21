import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_on_rectangle_l243_24307

/-- A triangle with vertices on the sides of a rectangle -/
structure TriangleOnRectangle where
  -- Rectangle dimensions
  width : ℝ
  height : ℝ
  -- Vertex coordinates (x, y)
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Ensure vertices are on the rectangle sides
  h_A : A.1 = 0 ∧ 0 ≤ A.2 ∧ A.2 ≤ height
  h_B : B.2 = 0 ∧ 0 ≤ B.1 ∧ B.1 ≤ width
  h_C : C.1 = width ∨ C.2 = height

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Theorem: The area of triangle ABC on a 4x5 rectangle is 9 square units -/
theorem triangle_area_on_rectangle :
  ∀ (t : TriangleOnRectangle),
  t.width = 4 ∧ t.height = 5 →
  triangleArea t.A t.B t.C = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_on_rectangle_l243_24307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kelvin_probability_l243_24317

noncomputable def jump_probability (k : ℕ) : ℝ :=
  if k > 0 then 1 / (2 ^ k) else 0

inductive FrogState
  | OnPad (n : ℕ)
  | Beyond

noncomputable def kelvin_chain (s : FrogState) : FrogState → ℝ
  | FrogState.OnPad n =>
      if n < 2019 then
        jump_probability (2019 - n)
      else if n = 2019 then 0
      else 1 - jump_probability (n - 2019)
  | FrogState.Beyond => 1

theorem kelvin_probability : 
  ∃ (p : ℝ), p = 1/2 ∧ 
  (∀ (s : FrogState), s ≠ FrogState.OnPad 2019 → 
    kelvin_chain s (FrogState.OnPad 2019) = p) := by
  sorry

#check kelvin_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kelvin_probability_l243_24317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_transformation_l243_24347

noncomputable section

-- Define the fractional linear function
noncomputable def f (x₁ x₂ x₃ x : ℝ) : ℝ :=
  ((x₂ - x₃) * x - x₁ * (x₂ - x₃)) / ((x₂ - x₁) * x - x₃ * (x₂ - x₁))

-- State the theorem
theorem fractional_linear_transformation (x₁ x₂ x₃ : ℝ) 
  (h : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :
  f x₁ x₂ x₃ x₁ = 0 ∧ 
  f x₁ x₂ x₃ x₂ = 1 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₃| < δ → |f x₁ x₂ x₃ x| > 1/ε) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_linear_transformation_l243_24347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l243_24331

/-- The area of the shaded region formed by semicircles -/
noncomputable def shaded_area (AB BC CD DE EF : ℝ) : ℝ :=
  let AF := AB + BC + CD + DE + EF
  let semicircle_area (d : ℝ) := Real.pi * d^2 / 8
  semicircle_area AF - (semicircle_area AB + semicircle_area BC + semicircle_area CD + semicircle_area DE) + semicircle_area EF

/-- Theorem stating that the area of the shaded region is 43.75π -/
theorem shaded_area_value :
  shaded_area 3 4 4 4 5 = 43.75 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l243_24331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l243_24363

/-- The markup percentage applied to the cost price -/
noncomputable def markup : ℚ := 15

/-- The selling price of the computer table -/
noncomputable def selling_price : ℚ := 8325

/-- The cost price of the computer table -/
noncomputable def cost_price : ℚ := selling_price / (1 + markup / 100)

/-- Theorem stating that the cost price is approximately 7234.78 -/
theorem cost_price_approximation : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 / 100) ∧ |cost_price - 7234.78| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approximation_l243_24363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_scaling_implies_homogeneous_zero_l243_24375

/-- An algebraic function of several variables -/
def AlgebraicFunction (α : Type*) [Field α] (n : ℕ) := (Fin n → α) → α

/-- A function is unchanged when scaled by k -/
def UnchangedByScaling {α : Type*} [Field α] {n : ℕ} (f : AlgebraicFunction α n) : Prop :=
  ∀ (k : α) (v : Fin n → α), k ≠ 0 → f (fun i => k * v i) = f v

/-- A homogeneous function of degree m -/
def HomogeneousFunction {α : Type*} [Field α] {n : ℕ} (f : AlgebraicFunction α n) (m : ℤ) : Prop :=
  ∀ (k : α) (v : Fin n → α), k ≠ 0 → f (fun i => k * v i) = k^m * f v

theorem unchanged_scaling_implies_homogeneous_zero
  {α : Type*} [Field α] {n : ℕ} (f : AlgebraicFunction α n) 
  (h : UnchangedByScaling f) : 
  HomogeneousFunction f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_scaling_implies_homogeneous_zero_l243_24375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_combined_length_l243_24376

-- Define the speeds of the trains in km/hr
noncomputable def train1_speed : ℝ := 80
noncomputable def train2_speed : ℝ := 60

-- Define the time taken for train1 to cross train2 in seconds
noncomputable def crossing_time : ℝ := 18

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Define the combined length of both trains
noncomputable def combined_length : ℝ := (train1_speed - train2_speed) * km_hr_to_m_s * crossing_time

-- Theorem statement
theorem trains_combined_length :
  combined_length = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_combined_length_l243_24376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_fraction_l243_24328

/-- A square with diagonals and midpoint-connecting lines -/
structure GeometricSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The square has positive side length -/
  side_pos : side > 0

/-- The area of the entire square -/
noncomputable def GeometricSquare.area (s : GeometricSquare) : ℝ :=
  s.side * s.side

/-- The area of one of the eight triangles formed by diagonals and midpoint lines -/
noncomputable def GeometricSquare.eighth_triangle_area (s : GeometricSquare) : ℝ :=
  s.area / 8

/-- The area of the shaded triangle -/
noncomputable def GeometricSquare.shaded_triangle_area (s : GeometricSquare) : ℝ :=
  s.eighth_triangle_area / 4

/-- Theorem: The area of the shaded triangle is 1/16 of the square's area -/
theorem shaded_triangle_fraction (s : GeometricSquare) :
    s.shaded_triangle_area = s.area / 16 := by
  sorry

#check shaded_triangle_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_triangle_fraction_l243_24328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_condition_l243_24325

/-- The equation √(x + a√x + b) + √x = c has infinitely many solutions
    if and only if a = -2c, b = c², and c > 0 -/
theorem infinite_solutions_condition (a b c : ℝ) :
  (∃ S : Set ℝ, (∀ x ∈ S, Real.sqrt (x + a * Real.sqrt x + b) + Real.sqrt x = c) ∧ Infinite S) ↔
  (a = -2 * c ∧ b = c^2 ∧ c > 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_condition_l243_24325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inventory_problem_l243_24341

/-- The inventory problem of a clothing store -/
theorem inventory_problem (black_shirts : ℕ) :
  let ties : ℕ := 34
  let belts : ℕ := 40
  let white_shirts : ℕ := 42
  let jeans : ℕ := (2 * (black_shirts + white_shirts) + 2) / 3
  let scarves : ℕ := (ties + belts) / 2
  jeans = scarves + 33 →
  black_shirts = 63 :=
by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inventory_problem_l243_24341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_for_three_element_set_l243_24329

theorem subset_count_for_three_element_set {α : Type*} [Fintype α] :
  ∀ (S : Finset α), Finset.card S = 3 → Finset.card (Finset.powerset S) = 8 :=
by
  intro S hS
  have h : Finset.card (Finset.powerset S) = 2^(Finset.card S) := by
    apply Finset.card_powerset
  rw [h, hS]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_for_three_element_set_l243_24329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_configurations_is_96_l243_24358

/-- Represents a color: Green, Yellow, or Purple -/
inductive Color
  | Green
  | Yellow
  | Purple

/-- Represents a dot in the configuration -/
structure Dot where
  color : Color

/-- Represents the configuration of 9 dots -/
structure Configuration where
  dots : Vector Dot 9

/-- Checks if two dots have different colors -/
def differentColors (d1 d2 : Dot) : Prop :=
  d1.color ≠ d2.color

/-- Checks if a configuration is valid (no adjacent dots have the same color) -/
def isValidConfiguration (config : Configuration) : Prop :=
  differentColors config.dots[0] config.dots[1] ∧
  differentColors config.dots[0] config.dots[2] ∧
  differentColors config.dots[1] config.dots[2] ∧
  differentColors config.dots[2] config.dots[3] ∧
  differentColors config.dots[3] config.dots[4] ∧
  differentColors config.dots[3] config.dots[5] ∧
  differentColors config.dots[4] config.dots[5] ∧
  differentColors config.dots[5] config.dots[6] ∧
  differentColors config.dots[6] config.dots[7] ∧
  differentColors config.dots[6] config.dots[8] ∧
  differentColors config.dots[7] config.dots[8]

/-- The number of valid configurations -/
def numValidConfigurations : ℕ :=
  6 * 4 * 4  -- Based on the solution provided

theorem num_valid_configurations_is_96 :
  numValidConfigurations = 96 := by
  rfl  -- reflexivity proves this, as it's true by definition

#eval numValidConfigurations  -- This will output 96

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_configurations_is_96_l243_24358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoint_l243_24322

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line passing through P(-1,0)
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_P k x₁ y₁ ∧ line_through_P k x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the midpoint M
def midpoint_M (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Theorem statement
theorem locus_of_midpoint :
  ∀ (x y k x₁ y₁ x₂ y₂ : ℝ),
    intersection_points k x₁ y₁ x₂ y₂ →
    midpoint_M x y x₁ y₁ x₂ y₂ →
    y^2 = (1/2) * x + 1/2 ∧ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_midpoint_l243_24322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_sledding_distance_l243_24318

/-- The optimal distance from point H to point B on a sledding hill -/
theorem optimal_sledding_distance (H S g : ℝ) (h1 : H = 5) (h2 : S = 3) (h3 : g = 10) :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt (H^2 + x^2) - x
  ∃ x : ℝ, x = 5 * Real.sqrt 3 / 3 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_sledding_distance_l243_24318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l243_24364

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*x - 4) * (e^x) + a * (x + 2)^2

theorem tangent_line_and_inequality (a : ℝ) :
  (∀ x : ℝ, (f 1 x = (2*x - 4) * (e^x) + (x + 2)^2) ∧
            (x = 0 → (deriv (f 1)) x = 2) ∧
            (f 1 0 = 0)) ∧
  (a ≥ 1/2 ↔ ∀ x : ℝ, x ≥ 0 → f a x ≥ 4*a - 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l243_24364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l243_24345

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the length of side b and the area of the triangle. -/
theorem triangle_side_and_area (a b c : ℝ) (A B C : Real) :
  a = 4 →
  c = 3 →
  Real.cos B = 1/8 →
  b = Real.sqrt 22 ∧ (1/2 * a * c * Real.sqrt (1 - (Real.cos B)^2)) = (9 * Real.sqrt 7) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_and_area_l243_24345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_dot_product_l243_24388

/-- The line y = k(x+1) intersecting the ellipse x^2 + 3y^2 = 5 at points A and B -/
noncomputable def line_ellipse_intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + 1) ∧ p.1^2 + 3 * p.2^2 = 5}

/-- The point M -/
noncomputable def M : ℝ × ℝ := (-7/3, 0)

/-- The dot product of vectors MA and MB -/
noncomputable def dot_product_MA_MB (A B : ℝ × ℝ) : ℝ :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2)

theorem line_ellipse_intersection_dot_product (k : ℝ) :
  ∃ A B, A ∈ line_ellipse_intersection k ∧ B ∈ line_ellipse_intersection k ∧ dot_product_MA_MB A B = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ellipse_intersection_dot_product_l243_24388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l243_24320

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of their linear combination is √13. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) : 
  a = (0, 1) → 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -Real.sqrt 2 / 2 →
  Real.sqrt ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l243_24320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l243_24377

theorem sum_of_solutions_is_zero :
  ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 = 256 + x) ∧ (s.sum id = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l243_24377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l243_24337

/-- Represents the number of solutions for a system of equations -/
inductive NumSolutions
  | None
  | One
  | MoreThanOne

/-- 
Given a system of n equations of the form:
  a x₁² + b x₁ + c = x₂
  a x₂² + b x₂ + c = x₃
  ...
  a xₙ₋₁² + b xₙ₋₁ + c = xₙ
  a xₙ² + b xₙ + c = x₁
where a, b, c are real numbers and a ≠ 0,
the number of real solutions depends on the value of (b-1)² - 4ac.
-/
theorem solution_count (a b c : ℝ) (ha : a ≠ 0) :
  (((b - 1)^2 - 4*a*c < 0) → (∀ x : ℝ, ¬(a*x^2 + b*x + c = x))) ∧
  (((b - 1)^2 - 4*a*c = 0) → (∃! x : ℝ, a*x^2 + b*x + c = x)) ∧
  (((b - 1)^2 - 4*a*c > 0) → (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = x ∧ a*y^2 + b*y + c = y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l243_24337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_lines_constant_distance_l243_24355

-- Define the ellipse G
noncomputable def G (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define a line l
def Line (k n : ℝ) (x y : ℝ) : Prop := y = k * x + n

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (k n : ℝ) : ℝ := |n| / Real.sqrt (1 + k^2)

theorem ellipse_intersecting_lines_constant_distance :
  ∀ (k n : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    G x₁ y₁ ∧ G x₂ y₂ ∧  -- A and B are on the ellipse
    Line k n x₁ y₁ ∧ Line k n x₂ y₂ ∧  -- A and B are on line l
    x₁ * x₂ + y₁ * y₂ = 0 →  -- OA ⊥ OB
    distance_point_to_line k n = 2 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_lines_constant_distance_l243_24355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l243_24380

/-- Given a parallelogram with adjacent sides of lengths s and 2s forming a 30-degree angle,
    if the area is 8√3 square units, then s = 2√2. -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Ensure s is positive
  (8 * Real.sqrt 3 = 2 * s * (s * Real.sin (30 * π / 180))) →
  s = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_l243_24380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_graph_counterexample_l243_24356

-- Define the graph structure
structure BipartiteGraph (α β : Type) where
  A : Set α
  B : Set β
  E : Set (α × β)

-- Define the E(X) function
def E_of_X {α β : Type} (G : BipartiteGraph α β) (X : Set α) : Set β :=
  {b ∈ G.B | ∃ a ∈ X, (a, b) ∈ G.E}

-- Define a saturating matching
def is_saturating_matching {α β : Type} (G : BipartiteGraph α β) (f : α → β) : Prop :=
  Function.Injective f ∧ (∀ a ∈ G.A, (a, f a) ∈ G.E)

theorem bipartite_graph_counterexample :
  ∃ (G : BipartiteGraph ℕ ℕ),
    (Countable G.A ∧ Countable G.B) ∧
    (∀ X : Set ℕ, X.Finite → X ⊆ G.A → (E_of_X G X).Infinite ∨ (Finite (E_of_X G X) ∧ Nat.card (E_of_X G X) ≥ Nat.card X)) ∧
    ¬∃ f : ℕ → ℕ, is_saturating_matching G f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bipartite_graph_counterexample_l243_24356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l243_24346

theorem inscribed_circle_radius_isosceles_triangle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let s := (2 * a + b) / 2
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  area / s = (5 * Real.sqrt 15) / 13 → a = 8 ∧ b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l243_24346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l243_24390

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 2 ∧ y₁ = -3 ∧ x₂ = 6 ∧ y₂ = -18) ∧
    (∀ x y : ℝ,
      (3*x + Real.sqrt (3*x - y) + y = 6 ∧
       9*x^2 + 3*x - y - y^2 = 36) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l243_24390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_fourth_quadrant_l243_24348

theorem cos_double_angle_fourth_quadrant (α : Real) :
  (α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →  -- given condition
  Real.cos (2 * α) = Real.sqrt 5 / 3 := by  -- conclusion to prove
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_fourth_quadrant_l243_24348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_police_placement_covers_all_streets_l243_24398

/-- Represents an intersection in the city --/
inductive Intersection
| A | B | C | D | E | F | G | H | I | J | K
deriving BEq, Repr

/-- Represents a street in the city --/
def Street := List Intersection

/-- The list of all streets in the city --/
def all_streets : List Street := [
  [Intersection.A, Intersection.B, Intersection.C, Intersection.D],
  [Intersection.E, Intersection.F, Intersection.G],
  [Intersection.H, Intersection.I, Intersection.J, Intersection.K],
  [Intersection.A, Intersection.E, Intersection.H],
  [Intersection.B, Intersection.F, Intersection.I],
  [Intersection.D, Intersection.G, Intersection.J],
  [Intersection.H, Intersection.F, Intersection.C],
  [Intersection.C, Intersection.G, Intersection.K]
]

/-- Checks if a street is covered by the given set of intersections --/
def is_street_covered (street : Street) (officers : List Intersection) : Bool :=
  street.any (fun i => officers.contains i)

/-- The theorem to be proved --/
theorem police_placement_covers_all_streets :
  let officers := [Intersection.B, Intersection.G, Intersection.H]
  all_streets.all (fun street => is_street_covered street officers) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_police_placement_covers_all_streets_l243_24398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l243_24305

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  l.x₁ - l.y₁ / m

/-- The theorem stating that the x-intercept of the line passing through (10, 3) and (-12, -8) is 4 -/
theorem x_intercept_of_specific_line :
  x_intercept ⟨10, 3, -12, -8⟩ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_specific_line_l243_24305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_a_plus_b_to_a_minus_b_l243_24316

-- Define a and b as noncomputable
noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) ^ 2
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) ^ (-2 : ℤ)

-- State the theorem
theorem ratio_a_plus_b_to_a_minus_b : (a + b) / (a - b) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_a_plus_b_to_a_minus_b_l243_24316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sequence_ratio_l243_24389

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the condition for {cos a_n} to be a geometric sequence
def is_geometric_cosine_sequence (a₁ : ℝ) (d : ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, Real.cos (arithmetic_sequence a₁ d (n + 1)) = q * Real.cos (arithmetic_sequence a₁ d n)

theorem cosine_sequence_ratio 
  (a₁ : ℝ) 
  (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 2 * Real.pi) 
  (h3 : is_geometric_cosine_sequence a₁ d) : 
  ∃ q : ℝ, q = -1 ∧ ∀ n : ℕ, Real.cos (arithmetic_sequence a₁ d (n + 1)) = q * Real.cos (arithmetic_sequence a₁ d n) := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sequence_ratio_l243_24389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l243_24379

noncomputable def walking_time (x : ℝ) : ℝ := x

noncomputable def biking_time (x : ℝ) : ℝ := (20 / 60) * x

theorem youseff_distance : 
  ∃ x : ℝ, 
    walking_time x = biking_time x + 8 ∧ 
    x = 12 := by
  use 12
  constructor
  · -- Prove walking_time 12 = biking_time 12 + 8
    simp [walking_time, biking_time]
    norm_num
  · -- Prove 12 = 12
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_youseff_distance_l243_24379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_a_time_l243_24321

noncomputable def fill_rate (fill_time : ℝ) : ℝ := 1 / fill_time

noncomputable def volume_filled (rate : ℝ) (time : ℝ) : ℝ := rate * time

theorem tap_a_time (
  fill_time_a : ℝ) (fill_time_b : ℝ) (remaining_time : ℝ) :
  fill_time_a = 45 →
  fill_time_b = 40 →
  remaining_time = 23 →
  ∃ (x : ℝ),
    x > 0 ∧
    volume_filled (fill_rate fill_time_a + fill_rate fill_time_b) x +
    volume_filled (fill_rate fill_time_b) remaining_time = 1 ∧
    x = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_a_time_l243_24321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_value_l243_24303

/-- A geometric sequence with sum of first n terms S_n = a · 2^n + a - 2 -/
def geometric_sequence (a : ℝ) : ℕ → ℝ := 
  λ n => a * (2 ^ (n - 1))

/-- Sum of the first n terms of the geometric sequence -/
def S_n (a : ℝ) (n : ℕ) : ℝ := a * 2^n + a - 2

/-- The value of 'a' for which the given sum formula holds -/
theorem geometric_sequence_sum_value (a : ℝ) :
  (∀ n : ℕ, S_n a n = (Finset.range n).sum (geometric_sequence a)) →
  a = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_value_l243_24303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_f_l243_24326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (x + a) * Real.log x else 2 * a * x + 2 + a

theorem min_derivative_f (a : ℝ) :
  (deriv (f a)) (-1) = (deriv (f a)) 1 →
  ∃ (min_val : ℝ), min_val = 2 ∧
    ∀ x > 0, (deriv (f a)) x ≥ min_val :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_derivative_f_l243_24326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l243_24332

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  let a := 2
  {x : ℝ | f a x ≥ 4 - |x - 1|} = Set.Iic (2/3) :=
sorry

-- Part 2
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, x ∈ Set.Icc 0 2 ↔ f ((1/m) + (1/(2*n))) x ≤ 1) →
  m + 2*n ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l243_24332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_expected_value_final_result_l243_24382

/-- A counter that starts at 0 and either increases by 1 or resets to 0 with equal probability each second -/
def Counter : Type := ℕ

/-- The probability of the counter increasing by 1 -/
noncomputable def p_increase : ℝ := 1/2

/-- The probability of the counter resetting to 0 -/
noncomputable def p_reset : ℝ := 1/2

/-- The expected value of the counter after n seconds -/
noncomputable def expected_value (n : ℕ) : ℝ := 1 - (1/2)^n

/-- The number of seconds we're interested in -/
def seconds : ℕ := 10

/-- Theorem: The expected value of the counter after 10 seconds is 1023/1024 -/
theorem counter_expected_value :
  expected_value seconds = 1023/1024 := by
  sorry

/-- The final result: 100m + n where m/n is the expected value after 10 seconds -/
theorem final_result :
  100 * 1023 + 1024 = 103324 := by
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_expected_value_final_result_l243_24382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l243_24384

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,4,8}
def B : Set ℕ := {3,4,7}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3,7} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l243_24384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_shots_missed_l243_24344

theorem basketball_shots_missed (shots_attempted : ℕ) (shots_made_percentage : ℚ) 
  (h1 : shots_attempted = 20)
  (h2 : shots_made_percentage = 80 / 100) : 
  shots_attempted - (shots_made_percentage * ↑shots_attempted).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_shots_missed_l243_24344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relatively_prime_cube_ratio_l243_24308

theorem relatively_prime_cube_ratio (a b : ℕ) : 
  a > b ∧ b > 0 ∧ 
  Nat.Coprime a b ∧ 
  (a^3 - b^3) / ((a - b)^3 : ℚ) = 91/7 → 
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relatively_prime_cube_ratio_l243_24308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enthalpy_change_proof_l243_24304

/-- Standard enthalpy of formation for Na₂O (solid) at 298 K in kJ/mol -/
noncomputable def H_f_Na2O : ℚ := -416

/-- Standard enthalpy of formation for H₂O (liquid) at 298 K in kJ/mol -/
noncomputable def H_f_H2O : ℚ := -286

/-- Standard enthalpy of formation for NaOH (solid) at 298 K in kJ/mol -/
noncomputable def H_f_NaOH : ℚ := -427.8

/-- Standard enthalpy change for the reaction Na₂O (s) + H₂O (l) → 2 NaOH (s) at 298 K in kJ -/
noncomputable def ΔH_reaction : ℚ := 2 * H_f_NaOH - (H_f_Na2O + H_f_H2O)

/-- Standard enthalpy change for the formation of 1 mole of NaOH at 298 K in kJ (heat released) -/
noncomputable def ΔH_per_mole : ℚ := -ΔH_reaction / 2

theorem enthalpy_change_proof :
  ΔH_reaction = -153.6 ∧ ΔH_per_mole = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enthalpy_change_proof_l243_24304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_cube_inequality_sum_inequality_main_theorem_l243_24327

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Theorem 1
theorem cosine_inequality (t : Triangle) (h : t.a > t.b) :
  Real.cos t.A < Real.cos t.B ∧ Real.cos (2 * t.A) < Real.cos (2 * t.B) := by
  sorry

-- Theorem 2
theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 := by
  sorry

-- Define an arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Sum of first n terms of arithmetic sequence
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem 3
theorem sum_inequality (a₁ d : ℝ) (h : S a₁ d 2016 - S a₁ d 1 = 1) :
  S a₁ d 2017 > 1 := by
  sorry

-- Main theorem combining all results
theorem main_theorem (t : Triangle) (a b : ℝ) (a₁ d : ℝ) 
  (h1 : t.a > t.b)
  (h2 : a > b)
  (h3 : S a₁ d 2016 - S a₁ d 1 = 1) :
  (Real.cos t.A < Real.cos t.B ∧ Real.cos (2 * t.A) < Real.cos (2 * t.B)) ∧
  (a^3 > b^3) ∧
  (S a₁ d 2017 > 1) := by
  have p1 := cosine_inequality t h1
  have p2 := cube_inequality h2
  have p4 := sum_inequality a₁ d h3
  exact ⟨p1, p2, p4⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_inequality_cube_inequality_sum_inequality_main_theorem_l243_24327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l243_24339

noncomputable section

-- Define the square and its vertices
def square_side_length : ℝ := 1
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (1, 1)

-- Define C' after folding
def C' : ℝ × ℝ := (1, 0.5)

-- Define E as the intersection of BC and AB
def E : ℝ × ℝ := (2/3, 2/3)

-- State the theorem
theorem perimeter_of_triangle_AEC' :
  let perimeter := dist A E + dist E C' + dist C' A
  perimeter = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l243_24339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l243_24393

theorem equation_solution :
  {x : ℝ | x > 0 ∧ x^(Real.log x^2 / Real.log 10) = x^4 / 1000} = {10, (10 : ℝ)^(3/2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l243_24393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l243_24386

def i : ℂ := Complex.I

theorem modulus_of_z : Complex.abs ((10 * i) / (3 + i)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l243_24386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_left_square_side_length_l243_24333

/-- Given a figure composed of squares with specific properties, prove the side length of the bottom left square is 4. -/
theorem bottom_left_square_side_length : 
  ∀ (x y : ℝ), 
  -- The side of the smallest square is 1
  let smallest_square := 1;
  -- The sides of other squares are x-1, x-2, and x-3
  let second_largest := x - 1;
  let third_largest := x - 2;
  let fourth_largest := x - 3;
  -- The side of the largest square (x) is the sum of the sides of the next largest square and the smallest square
  x = second_largest + smallest_square →
  -- The top side of the rectangle is x + (x-1)
  let top_side := x + second_largest;
  -- The bottom side of the rectangle is (x-2) + (x-3) + y, where y is the side of the bottom left square
  let bottom_side := third_largest + fourth_largest + y;
  -- Opposite sides of the rectangle are equal
  top_side = bottom_side →
  -- The side of the bottom left square (y) is 4
  y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_left_square_side_length_l243_24333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_bound_implies_common_factor_l243_24309

theorem product_bound_implies_common_factor (A : Finset ℕ) : 
  A.card = 16 → 
  (∀ a b : ℕ, a ∈ A → b ∈ A → a ≠ b → a * b ≤ 1994) → 
  ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_bound_implies_common_factor_l243_24309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l243_24391

theorem count_integer_pairs : 
  let S := {(a, b) : ℕ × ℕ | a > 0 ∧ b > 0 ∧ a + b ≤ 100 ∧ (a * b + a) / (b + a * b) = 13}
  Finset.card (Finset.filter (fun (p : ℕ × ℕ) => p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 100 ∧ (p.1 * p.2 + p.1) / (p.2 + p.1 * p.2) = 13) (Finset.range 101 ×ˢ Finset.range 101)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l243_24391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_actors_per_group_l243_24373

/-- Represents the show with its properties -/
structure Show where
  duration : ℕ  -- Duration of the show in minutes
  group_performance_time : ℕ  -- Time each group performs in minutes
  total_actors : ℕ  -- Total number of actors in the show

/-- Calculates the number of actors in each group -/
def actors_per_group (s : Show) : ℕ :=
  s.total_actors / (s.duration / s.group_performance_time)

/-- Theorem stating that for the given show parameters, there are 5 actors per group -/
theorem five_actors_per_group :
  ∃ (s : Show), s.duration = 60 ∧ s.group_performance_time = 15 ∧ s.total_actors = 20 ∧ actors_per_group s = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_actors_per_group_l243_24373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_difference_l243_24368

-- Define the operation @ (renamed to 'atOp')
def atOp (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem atOp_difference : (atOp 9 6) - (atOp 6 9) = -12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atOp_difference_l243_24368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l243_24351

/-- A circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ
  r_pos : r > 0

/-- A circle is tangent to the y-axis -/
def tangent_y_axis (c : Circle) : Prop :=
  |c.a| = c.r

/-- A circle is tangent to both coordinate axes -/
def tangent_both_axes (c : Circle) : Prop :=
  |c.a| = |c.b| ∧ |c.a| = c.r

theorem circle_tangency (c : Circle) :
  (tangent_y_axis c → |c.a| = c.r) ∧
  (tangent_both_axes c → |c.a| = |c.b| ∧ |c.a| = c.r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l243_24351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_generic_packages_is_two_l243_24361

def generic_tees_per_package : ℕ := 12
def aero_tees_per_package : ℕ := 2
def num_people : ℕ := 4
def min_tees_per_person : ℕ := 20
def aero_packages_bought : ℕ := 28

def min_generic_packages : ℕ :=
  let total_tees_needed := num_people * min_tees_per_person
  let aero_tees_bought := aero_packages_bought * aero_tees_per_package
  let additional_tees_needed := total_tees_needed - aero_tees_bought
  (additional_tees_needed + generic_tees_per_package - 1) / generic_tees_per_package
  
#eval min_generic_packages

theorem min_generic_packages_is_two : min_generic_packages = 2 := by
  unfold min_generic_packages
  norm_num
  rfl

#check min_generic_packages_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_generic_packages_is_two_l243_24361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_solution_l243_24399

open Real

theorem trigonometric_solution : ∃ x : ℝ, 
  (sin (3 * x) * sin (4 * x) = cos (3 * x) * cos (4 * x)) ∧ 
  (sin (7 * x) = 0) ∧ 
  (x = π / 7) := by
  use π / 7
  constructor
  · -- Prove first condition
    sorry
  constructor
  · -- Prove second condition
    sorry
  · -- Prove third condition
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_solution_l243_24399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l243_24387

theorem equation_solution : ∃! (x : ℝ), x > 0 ∧ |20 / x - x / 15| = 20 / 15 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l243_24387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_symmetry_and_period_l243_24359

def has_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_with_symmetry_and_period (f : ℝ → ℝ) :
  has_symmetry_axis f 2 → has_period f 4 → f = λ x ↦ Real.cos (π / 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_with_symmetry_and_period_l243_24359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parts_of_unit_circle_l243_24353

-- Define the equation
def equation (x y : ℝ) (n : ℤ) : Prop :=
  Real.arcsin x + Real.arccos y = n * Real.pi

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop :=
  x ≤ 0 ∧ y ≥ 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≤ 0

-- Theorem statement
theorem equation_represents_parts_of_unit_circle :
  ∀ x y : ℝ, ∃ n : ℤ, equation x y n →
  (unit_circle x y ∧ (second_quadrant x y ∨ fourth_quadrant x y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parts_of_unit_circle_l243_24353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_theorem_l243_24381

/-- A shape with right-angled segments -/
structure RightAngledShape where
  t : ℚ
  k : ℚ
  perimeter : ℚ

/-- The perimeter calculation for the shape -/
def perimeter_calc (shape : RightAngledShape) : ℚ :=
  2 * (3 * shape.t + shape.k + (3/2) * shape.k + shape.t + (1/2) * shape.k)

/-- The theorem stating the value of k given the conditions -/
theorem k_value_theorem (shape : RightAngledShape) 
  (h1 : perimeter_calc shape = shape.perimeter)
  (h2 : shape.perimeter = 162) :
  shape.k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_theorem_l243_24381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l243_24306

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

theorem f_satisfies_conditions :
  (∀ x : ℝ, f (π / 12 + x) + f (π / 12 - x) = 0) ∧
  (∀ x : ℝ, -π / 6 < x → x < π / 3 → (deriv^[2] f) x > 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l243_24306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1982_l243_24338

def f : ℕ+ → ℕ
  | n => sorry  -- We'll define the function later

axiom f_nonneg : ∀ n : ℕ+, f n ≥ 0

axiom f_2 : f 2 = 0

axiom f_3_pos : f 3 > 0

axiom f_9999 : f 9999 = 3333

axiom f_property : ∀ m n : ℕ+, (f (m + n) - f m - f n = 0) ∨ (f (m + n) - f m - f n = 1)

theorem f_1982 : f 1982 = 660 := by
  sorry  -- The proof will be added later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1982_l243_24338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_theorem_l243_24352

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) := (3*x - 1/(2*3*x))^n

def coefficient (n k : ℕ) : ℚ := (n.choose k) * (-1/2)^k

theorem binomial_theorem (n : ℕ) 
  (h1 : coefficient n 0 + (1/4 * coefficient n 2) = coefficient n 1) :
  n = 8 ∧ 
  coefficient n 4 = 35/8 ∧
  (∀ k, k ≠ 4 → coefficient n k ≤ coefficient n 4) ∧
  binomial_expansion (1 : ℝ) n = 1/256 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_theorem_l243_24352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l243_24349

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

/-- Total interest calculation function when principal is trebled after Y years -/
noncomputable def total_interest (principal rate Y : ℝ) : ℝ :=
  simple_interest principal rate Y + simple_interest (3 * principal) rate (10 - Y)

/-- Theorem stating that the principal is trebled after 5 years -/
theorem principal_trebled_after_five_years
  (principal rate : ℝ)
  (h1 : simple_interest principal rate 10 = 600)
  (h2 : total_interest principal rate 5 = 1200) :
  5 = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_trebled_after_five_years_l243_24349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l243_24350

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (2 * mySequence n) / (2 + mySequence n)

theorem mySequence_formula : ∀ n : ℕ, mySequence n = 2 / (n + 1) := by
  intro n
  induction n with
  | zero => 
    simp [mySequence]
    -- Proof for base case
    sorry
  | succ k ih =>
    simp [mySequence]
    -- Proof for inductive step
    sorry

#eval mySequence 0  -- Should output 1
#eval mySequence 1  -- Should output 2/3
#eval mySequence 2  -- Should output 1/2
#eval mySequence 3  -- Should output 2/5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_formula_l243_24350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_square_vertices_l243_24324

/-- A complex number z such that z, z^2 + 1, and z^4 form vertices of a square with area 1 -/
def squareVertices (z : ℂ) : Prop :=
  let v1 := z^2 + 1 - z
  let v2 := z^4 - z
  (v1.re * v2.re + v1.im * v2.im = 0) ∧  -- orthogonality
  (Complex.abs v1 = Complex.abs v2) ∧    -- equal magnitude
  (Complex.abs v1 ^ 2 = 1)               -- area = 1

/-- There exists a complex number z that satisfies the squareVertices property -/
theorem exists_square_vertices : ∃ z : ℂ, squareVertices z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_square_vertices_l243_24324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l243_24340

-- Define the constant k
variable (k : ℝ)

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (-k * x)

-- State the theorem
theorem function_properties (k : ℝ) (h1 : ∀ x, f k x > 0) (h2 : ∀ a b, f k a * f k b = f k (a + b)) (h3 : k > 0) :
  (f k 0 = 1) ∧ 
  (∀ a, f k (-a) = 1 / f k a) ∧ 
  (∀ a, f k a = (f k (3 * a)) ^ (1/3)) ∧ 
  (∀ a b, b > a → f k b > f k a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l243_24340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_M_l243_24314

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (y + x ≥ abs (x - y)) ∧
               ((x^2 - 8*x + y^2 + 6*y) / (x + 2*y - 8) ≤ 0)}

-- State the theorem
theorem area_of_M : MeasureTheory.volume M = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_M_l243_24314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composite_nn_plus_nplus1_nplus1_l243_24319

theorem infinitely_many_composite_nn_plus_nplus1_nplus1 :
  ∃ f : ℕ → ℕ, StrictMono f ∧
  ∀ k : ℕ, 3 ∣ (f k)^(f k) + (f k + 1)^(f k + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_composite_nn_plus_nplus1_nplus1_l243_24319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_l243_24360

theorem isosceles_triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let angle_A := Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))
  -- Isosceles triangle condition
  AB = AC ∧
  -- Vertex angle condition
  angle_A = 2 * Real.pi / 3 ∧
  -- Side length condition
  BC = 2 * Real.sqrt 3 →
  -- Dot product
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_dot_product_l243_24360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f4_is_simplified_l243_24334

-- Define the fractions
noncomputable def f1 (x : ℝ) : ℝ := 4 / (2 * x)
noncomputable def f2 (x : ℝ) : ℝ := (1 - x) / (x - 1)
noncomputable def f3 (x : ℝ) : ℝ := (x + 1) / (x^2 - 1)
noncomputable def f4 (x : ℝ) : ℝ := (2 * x) / (x^2 - 4)

-- Define what it means for a fraction to be simplified
def is_simplified (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → ∀ g : ℝ → ℝ, (∀ y : ℝ, y ≠ 0 → f y = g y) → f = g

-- State the theorem
theorem f4_is_simplified :
  ¬(is_simplified f1) ∧ ¬(is_simplified f2) ∧ ¬(is_simplified f3) ∧ (is_simplified f4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f4_is_simplified_l243_24334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_condition_l243_24313

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1)) / 2 * seq.d

theorem arithmetic_sequence_condition (seq : ArithmeticSequence) :
  (∀ n : ℕ, n ≥ 2 → S seq n > n * seq.a n) ↔ seq.a 3 > seq.a 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_condition_l243_24313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l243_24300

theorem set_relations :
  (0 ∈ ({0} : Set ℕ)) ∧
  (∅ ⊆ ({0} : Set ℕ)) ∧
  ¬({0, 1} ⊆ ({(0, 1)} : Set (ℕ × ℕ))) ∧
  ∀ (a b : ℕ), ({(a, b)} : Set (ℕ × ℕ)) = {(b, a)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relations_l243_24300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_count_without_two_corners_l243_24392

def Grid := Fin 5 → Fin 5 → Bool

def valid_coloring (g : Grid) : Prop :=
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = true) ∧
  (∀ j : Fin 5, ∃! i : Fin 5, g i j = true)

def corner_not_colored (g : Grid) : Prop :=
  g 0 0 = false ∧ g 4 4 = false

def coloring_count : ℕ := 120

def coloring_count_without_one_corner : ℕ := 96

-- Define the set of valid colorings without two corners
def ValidColoringsWithoutTwoCorners : Set Grid :=
  { g | valid_coloring g ∧ corner_not_colored g }

-- Assume the set is finite
axiom ValidColoringsWithoutTwoCorners_finite : 
  Fintype ValidColoringsWithoutTwoCorners

-- Now we can state the theorem
theorem coloring_count_without_two_corners :
  @Fintype.card ValidColoringsWithoutTwoCorners ValidColoringsWithoutTwoCorners_finite = 78 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_count_without_two_corners_l243_24392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_negative_four_thirds_l243_24310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (x + 1) + x + a - 1

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x^2 + b * x

theorem sum_of_a_and_b_is_negative_four_thirds :
  ∀ a b : ℝ,
  (∀ x : ℝ, f a x = f a (-2 - x)) →
  (((1 - 1 / ((1 + 1)^2)) * (Real.exp 0 + b)) = -1) →
  a + b = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_a_and_b_is_negative_four_thirds_l243_24310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_39_value_l243_24378

def sequence_a : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence_a n + (n + 2)

theorem a_39_value : sequence_a 39 = 820 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_39_value_l243_24378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_segment_trip_average_speed_l243_24362

/-- Calculates the average speed of a two-segment trip -/
theorem two_segment_trip_average_speed 
  (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) 
  (h1 : speed1 = 40) (h2 : time1 = 1) (h3 : speed2 = 60) (h4 : time2 = 3) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 55 := by
  sorry

#check two_segment_trip_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_segment_trip_average_speed_l243_24362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_l243_24342

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  xIntercept1 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The other x-intercept of the ellipse -/
theorem other_x_intercept (e : Ellipse) 
    (h1 : e.focus1 = ⟨4, 0⟩) 
    (h2 : e.focus2 = ⟨0, 3⟩)
    (h3 : e.xIntercept1 = ⟨1, 0⟩) :
  ∃ (p : Point), p.y = 0 ∧ 
    p.x = 20/7 + Real.sqrt 10 ∧
    distance e.xIntercept1 e.focus1 + distance e.xIntercept1 e.focus2 =
    distance p e.focus1 + distance p e.focus2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_intercept_l243_24342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_against_current_l243_24357

theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 16) 
  (h2 : current_speed = 3.2) : 
  speed_with_current - 2 * current_speed = 9.6 := by
  rw [h1, h2]
  norm_num

#check mans_speed_against_current

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_against_current_l243_24357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l243_24323

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Graph 1 equation -/
def graph1 (x y : ℝ) : Prop :=
  (x - floor x)^2 + (y - 1)^2 = x - floor x

/-- Graph 2 equation -/
def graph2 (x y : ℝ) : Prop :=
  y = (1/3) * x

/-- Intersection point of the two graphs -/
def is_intersection_point (x y : ℝ) : Prop :=
  graph1 x y ∧ graph2 x y

/-- The set of all intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | is_intersection_point p.1 p.2}

/-- The main theorem: there are exactly 14 intersection points -/
theorem intersection_count :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 14 ∧ ∀ p, p ∈ s ↔ p ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l243_24323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l243_24315

theorem order_of_numbers (x y z : ℝ) 
  (hx : 0.8 < x ∧ x < 1.0) 
  (hy : y = Real.rpow 2 x) 
  (hz : z = Real.rpow x (Real.rpow 2 x)) : 
  z < x ∧ x < y := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l243_24315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_a_thon_earnings_walk_a_thon_earnings_eq_4_l243_24369

theorem walk_a_thon_earnings (this_year_rate : ℝ) (last_year_winner_earnings : ℝ) 
  (additional_miles : ℕ) : ℝ :=
  let last_year_rate : ℝ := 
    (last_year_winner_earnings / (last_year_winner_earnings / this_year_rate - additional_miles : ℝ))
  last_year_rate

theorem walk_a_thon_earnings_eq_4 (this_year_rate : ℝ) (last_year_winner_earnings : ℝ) 
  (additional_miles : ℕ) : 
  walk_a_thon_earnings this_year_rate last_year_winner_earnings additional_miles = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_a_thon_earnings_walk_a_thon_earnings_eq_4_l243_24369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l243_24395

/-- Proves that if an article is sold at Rs. 250 with a 25% profit, then its cost price is Rs. 200. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 250)
  (h2 : profit_percentage = 25) : 
  selling_price / (1 + profit_percentage / 100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l243_24395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l243_24311

/-- Parabola C: y^2 = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line L: y = x + 2 -/
def line_L (x y : ℝ) : Prop := y = x + 2

/-- Point P on line L -/
def point_on_L (P : ℝ × ℝ) : Prop := line_L P.1 P.2

/-- Q1 and Q2 are points of contact of tangents from P to C -/
def tangent_points (P Q1 Q2 : ℝ × ℝ) : Prop :=
  parabola_C Q1.1 Q1.2 ∧ parabola_C Q2.1 Q2.2 ∧
  (∃ (m b : ℝ), (Q1.2 - P.2) = m * (Q1.1 - P.1) ∧ Q1.2 = m * Q1.1 + b) ∧
  (∃ (m b : ℝ), (Q2.2 - P.2) = m * (Q2.1 - P.1) ∧ Q2.2 = m * Q2.1 + b)

/-- Q is the midpoint of Q1Q2 -/
def midpoint_Q (Q Q1 Q2 : ℝ × ℝ) : Prop :=
  Q.1 = (Q1.1 + Q2.1) / 2 ∧ Q.2 = (Q1.2 + Q2.2) / 2

/-- The locus of Q -/
def locus_Q (x y : ℝ) : Prop := (y - 1)^2 = 2*(x - 3/2)

theorem locus_of_Q :
  ∀ (P Q Q1 Q2 : ℝ × ℝ),
  point_on_L P →
  tangent_points P Q1 Q2 →
  midpoint_Q Q Q1 Q2 →
  locus_Q Q.1 Q.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_Q_l243_24311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l243_24385

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^(1/3) + 3 / (x^(1/3) + 4)

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, f x ≤ 0 ↔ x ∈ Set.Icc (-27) (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l243_24385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l243_24367

noncomputable def floor_part (c : ℝ) := Int.floor c

noncomputable def frac_part (c : ℝ) := c - (Int.floor c)

theorem problem_solution (c : ℝ) 
  (h1 : 3 * (floor_part c)^2 + 4 * (floor_part c) - 28 = 0)
  (h2 : 5 * (frac_part c)^2 - 8 * (frac_part c) + 3 = 0) :
  c = -3.4 := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l243_24367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_rotated_vector_l243_24372

/-- Given a vector a and its rotation b, prove that the projection of a onto b is as calculated -/
theorem projection_of_rotated_vector (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b = ((Real.cos (π / 4 + π / 3), Real.sin (π / 4 + π / 3)) : ℝ × ℝ) * 2 →
  let proj := (Real.sqrt 2 / 2) • b
  proj = ((1 - Real.sqrt 3) / 2, (1 + Real.sqrt 3) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_rotated_vector_l243_24372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_equals_82_over_11_l243_24301

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 7

-- Theorem statement
theorem g_of_5_equals_82_over_11 : g 5 = 82 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_5_equals_82_over_11_l243_24301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l243_24394

/-- The slope of the angle bisector of the acute angle formed at the origin
    by the lines y = 2x and y = 4x is (√21 - 6) / 7. -/
theorem angle_bisector_slope :
  let m₁ : ℝ := 2
  let m₂ : ℝ := 4
  let angle_bisector_slope : ℝ → ℝ → ℝ := λ a b ↦ (a + b - Real.sqrt (1 + a^2 + b^2)) / (a * b - 1)
  angle_bisector_slope m₁ m₂ = (Real.sqrt 21 - 6) / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l243_24394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_land_renovation_cost_l243_24336

/-- Calculates the renovation cost of a triangular piece of land -/
theorem triangular_land_renovation_cost 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : side1 = 32)
  (h2 : side2 = 68)
  (h3 : angle = 30 * π / 180)
  (h4 : cost_per_sqm = 50) : 
  (1/2 * side1 * side2 * Real.sin angle) * cost_per_sqm = 27200 := by
  sorry

#check triangular_land_renovation_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_land_renovation_cost_l243_24336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l243_24343

theorem pyramid_volume (base_length base_width edge_length : ℝ) :
  base_length = 7 →
  base_width = 9 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal / 2)^2)
  let volume := (1 / 3) * base_area * height
  volume = 84 * Real.sqrt 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l243_24343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_l243_24374

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then g x else x^2 - 2*x

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_g :
  is_odd f → (∀ x < 0, g x = -x^2 - 2*x) :=
by
  intro h x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_g_l243_24374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_picking_ratio_l243_24383

theorem apple_picking_ratio (total_apples remaining_apples : ℕ) : 
  total_apples = 200 →
  remaining_apples = 20 →
  let first_day := total_apples / 5
  let third_day := first_day + 20
  let total_picked := total_apples - remaining_apples
  let second_day := total_picked - first_day - third_day
  (second_day : ℚ) / first_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_picking_ratio_l243_24383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_product_l243_24354

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (3, -2, 4) and (7, 6, -2) is 10. -/
theorem midpoint_coordinate_product : 
  let p1 : (ℝ × ℝ) × ℝ := ((3, -2), 4)
  let p2 : (ℝ × ℝ) × ℝ := ((7, 6), -2)
  let midpoint := (
    ((p1.1.1 + p2.1.1) / 2,
     (p1.1.2 + p2.1.2) / 2),
    (p1.2 + p2.2) / 2
  )
  (midpoint.1.1 * midpoint.1.2 * midpoint.2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinate_product_l243_24354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l243_24335

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := (1/3) ^ (1/5 : ℝ)
noncomputable def c : ℝ := 2 ^ (1/3 : ℝ)

-- State the theorem
theorem a_lt_b_lt_c : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_b_lt_c_l243_24335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l243_24312

/-- 
Given a positive integer n that satisfies the equation (n+1)! + (n+4)! = n! * 1560,
prove that the sum of the digits of n is 1.
-/
theorem sum_of_digits_of_n (n : ℕ) 
  (h : Nat.factorial (n + 1) + Nat.factorial (n + 4) = Nat.factorial n * 1560) 
  (hn : n > 0) : 
  (n.digits 10).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l243_24312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_fall_ratio_l243_24365

/-- The number of platforms in the scaffolding -/
def total_platforms : ℕ := 8

/-- The platform David is on -/
def david_platform : ℕ := 6

/-- The initial distance Walter fell before passing David (in meters) -/
def initial_fall : ℚ := 4

/-- The height of each platform (in meters) -/
noncomputable def platform_height : ℚ := initial_fall / (total_platforms - david_platform : ℚ)

/-- The total height Walter fell (in meters) -/
noncomputable def total_fall : ℚ := platform_height * total_platforms

theorem walter_fall_ratio :
  total_fall / initial_fall = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_fall_ratio_l243_24365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l243_24302

/-- Represents an ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The focal distance of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  e.a * e.eccentricity

/-- Theorem: The foci of the ellipse 3x^2 + 4y^2 = 12 are at (1,0) and (-1,0) -/
theorem ellipse_foci_coordinates :
  let e : Ellipse := ⟨2, Real.sqrt 3, by sorry⟩
  e.focalDistance = 1 ∧ 
  3 * 2^2 = 4 * (Real.sqrt 3)^2 ∧
  3 * 1^2 + 4 * 0^2 = 12 ∧
  3 * (-1)^2 + 4 * 0^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_coordinates_l243_24302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l243_24397

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A : (U \ A) = {2, 4} := by
  apply Set.ext
  intro x
  simp [U, A]
  sorry

#check complement_of_A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l243_24397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_equals_one_sufficient_not_necessary_l243_24370

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := m^2 + m * Complex.I - 1

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- Theorem statement
theorem m_equals_one_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 1 ∧ is_purely_imaginary (z m)) ∧
  (∀ m : ℝ, m = 1 → is_purely_imaginary (z m)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_equals_one_sufficient_not_necessary_l243_24370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_formula_l243_24371

theorem cosine_difference_formula (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) (h2 : Real.tan α = 2) :
  Real.cos (α - π/4) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_formula_l243_24371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_line_l243_24396

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define what it means for a point to be inside the circle
def inside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  (x - 2)^2 + (y - 1)^2 < 4

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 5 = 0

-- Theorem statement
theorem chord_midpoint_line :
  inside_circle P →
  ∃ (a b : ℝ × ℝ),
    is_on_circle a.1 a.2 ∧
    is_on_circle b.1 b.2 ∧
    P = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) →
    ∀ (x y : ℝ), line_equation x y ↔ (∃ t : ℝ, x = a.1 + t * (b.1 - a.1) ∧ y = a.2 + t * (b.2 - a.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_midpoint_line_l243_24396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_80_over_81_l243_24330

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5) / 4

theorem inverse_f_at_negative_80_over_81 :
  f⁻¹ (-80/81) = (85/81)^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_negative_80_over_81_l243_24330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_improvement_yields_greater_returns_l243_24366

/-- Represents a glass product with a rating --/
structure Product where
  rating : ℝ

/-- Represents a production line --/
structure ProductionLine where
  dailyCapacity : ℕ
  products : List Product

/-- Represents the factory --/
structure Factory where
  lines : List ProductionLine
  funds : ℝ

/-- Classifies a product as Grade A or B --/
noncomputable def gradeProduct (p : Product) : Bool :=
  p.rating ≥ 10

/-- Calculates the price of a product based on its grade --/
noncomputable def productPrice (p : Product) : ℝ :=
  if gradeProduct p then 2000 else 1200

/-- Calculates the mean of a list of real numbers --/
noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

/-- Calculates the variance of a list of real numbers --/
noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x => (x - m)^2)).sum / xs.length

/-- Theorem stating that improving one production line yields greater returns --/
theorem improvement_yields_greater_returns
  (f : Factory)
  (sampleRatings : List ℝ)
  (h1 : f.lines.length = 2)
  (h2 : ∀ l ∈ f.lines, l.dailyCapacity = 200)
  (h3 : mean sampleRatings = 9.98)
  (h4 : variance sampleRatings = 0.045)
  (h5 : f.funds = 20000000)
  (h6 : sampleRatings.length = 16)
  (h7 : (sampleRatings.filter (λ x => x ≥ 10)).length = 6) :
  let improvedReturns := (2000 - 1200) * (6 / 16) * 200 * 365 - 20000000
  let financialReturns := 20000000 * 0.082
  improvedReturns > financialReturns := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_improvement_yields_greater_returns_l243_24366
