import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_min_phi_symmetry_min_phi_value_l292_29286

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.cos (2 * x)

-- The period of f(x) is π
theorem f_period : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
  ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T' := by
  sorry

-- The minimum positive φ for y-axis symmetry is 3π/8
theorem min_phi_symmetry : 
  ∃ (φ : ℝ), φ > 0 ∧ (∀ (x : ℝ), f (x - φ) = f (-x - φ)) ∧ 
  ∀ (φ' : ℝ), φ' > 0 ∧ (∀ (x : ℝ), f (x - φ') = f (-x - φ')) → φ ≤ φ' := by
  sorry

-- The value of the minimum positive φ is 3π/8
theorem min_phi_value (h : ∃ (φ : ℝ), φ > 0 ∧ (∀ (x : ℝ), f (x - φ) = f (-x - φ)) ∧ 
  ∀ (φ' : ℝ), φ' > 0 ∧ (∀ (x : ℝ), f (x - φ') = f (-x - φ')) → φ ≤ φ') : 
  ∃ (φ : ℝ), φ = 3 * Real.pi / 8 ∧ (∀ (x : ℝ), f (x - φ) = f (-x - φ)) ∧ 
  ∀ (φ' : ℝ), φ' > 0 ∧ (∀ (x : ℝ), f (x - φ') = f (-x - φ')) → φ ≤ φ' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_min_phi_symmetry_min_phi_value_l292_29286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_is_one_l292_29247

/-- Represents the cost of fruits and the discount policy -/
structure FruitStore where
  apple_cost : ℚ
  orange_cost : ℚ
  banana_cost : ℚ
  discount_per_five : ℚ

/-- Represents a customer's purchase -/
structure Purchase where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ
  total_paid : ℚ

/-- Calculates the total cost before discount -/
def total_cost_before_discount (store : FruitStore) (purchase : Purchase) : ℚ :=
  store.apple_cost * purchase.apples +
  store.orange_cost * purchase.oranges +
  store.banana_cost * purchase.bananas

/-- Calculates the discount amount -/
def discount_amount (store : FruitStore) (purchase : Purchase) : ℚ :=
  ((purchase.apples + purchase.oranges + purchase.bananas) / 5 : ℚ).floor * store.discount_per_five

/-- The main theorem: proving the cost of an apple -/
theorem apple_cost_is_one (store : FruitStore) (purchase : Purchase) : store.apple_cost = 1 :=
  by
  have h1 : store.orange_cost = 2 := sorry
  have h2 : store.banana_cost = 3 := sorry
  have h3 : store.discount_per_five = 1 := sorry
  have h4 : purchase.apples = 5 := sorry
  have h5 : purchase.oranges = 3 := sorry
  have h6 : purchase.bananas = 2 := sorry
  have h7 : purchase.total_paid = 15 := sorry
  
  have total_cost : total_cost_before_discount store purchase - discount_amount store purchase = purchase.total_paid := sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_is_one_l292_29247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l292_29222

-- Define the function
noncomputable def f (a b : ℝ) : ℝ := -1/(2*a) - 2/b

-- State the theorem
theorem supremum_of_f :
  ∃ M : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → f a b ≤ M) ∧
  (∀ ε > 0, ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ f a b > -9/2 - ε) :=
by
  -- We claim that M = -9/2 is the supremum
  use (-9/2)
  constructor

  -- Part 1: Show that f a b ≤ -9/2 for all valid a and b
  · intros a b ha hb hab
    sorry  -- Proof omitted

  -- Part 2: Show that for any ε > 0, there exist a and b such that f a b > -9/2 - ε
  · intros ε hε
    -- We can use a = 1/3 and b = 2/3 to get arbitrarily close to -9/2
    use (1/3), (2/3)
    sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l292_29222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visit_all_cities_in_196_flights_l292_29280

/-- A graph is a set of vertices and a set of edges between them. -/
structure Graph (V : Type) where
  edges : V → V → Prop

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def GraphPath (G : Graph V) (start finish : V) : List V → Prop
  | [] => start = finish
  | [v] => start = v ∧ v = finish
  | (v :: w :: rest) => G.edges v w ∧ GraphPath G w finish (w :: rest)

/-- A graph is connected if there is a path between any two vertices. -/
def Connected (G : Graph V) : Prop :=
  ∀ u v : V, ∃ p : List V, GraphPath G u v p

theorem visit_all_cities_in_196_flights 
  (G : Graph (Fin 100)) 
  (h : Connected G) : 
  ∃ (start : Fin 100) (route : List (Fin 100)), 
    (∀ v : Fin 100, v ∈ route) ∧ 
    route.length ≤ 196 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_visit_all_cities_in_196_flights_l292_29280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l292_29269

/-- The function f(x) = (1/3)^x - (1/5)^x -/
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - (1/5)^x

/-- Theorem stating the properties of function f -/
theorem f_properties (x₁ x₂ : ℝ) (h1 : x₁ ≥ 1) (h2 : x₂ ≥ 1) (h3 : x₁ < x₂) :
  f x₁ > f x₂ ∧ f (Real.sqrt (x₁ * x₂)) > Real.sqrt (f x₁ * f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l292_29269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l292_29268

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t + 2 * x t + y t = Real.sin t) ∧
       (deriv y t - 4 * x t - 2 * y t = Real.cos t)

-- Define the initial conditions
def initial_conditions (x y : ℝ → ℝ) : Prop :=
  x Real.pi = 1 ∧ y Real.pi = 2

-- Define the proposed solution
noncomputable def x_solution (t : ℝ) : ℝ := 1 - 2 * (t - Real.pi) + 2 * Real.sin t
noncomputable def y_solution (t : ℝ) : ℝ := 4 * (t - Real.pi) - 2 * Real.cos t - 3 * Real.sin t

-- Theorem statement
theorem solution_satisfies_system_and_conditions :
  system x_solution y_solution ∧ initial_conditions x_solution y_solution := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l292_29268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l292_29255

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the hexagon -/
def hexagonVertices : List Point := [
  ⟨0, 0⟩, ⟨1, 1⟩, ⟨3, 1⟩, ⟨4, 0⟩, ⟨3, -1⟩, ⟨1, -1⟩
]

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of the hexagon -/
noncomputable def hexagonPerimeter : ℝ :=
  let pairs := List.zip hexagonVertices (hexagonVertices.rotateLeft 1)
  (pairs.map (fun (p1, p2) => distance p1 p2)).sum

/-- The perimeter in the form a + b√2 + c√5 -/
noncomputable def perimeterForm (a b c : ℤ) : ℝ :=
  a + b * Real.sqrt 2 + c * Real.sqrt 5

theorem hexagon_perimeter_theorem :
  hexagonPerimeter = perimeterForm 4 4 0 ∧
  4 + 4 + 0 = 8 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval hexagonPerimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l292_29255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_117_l292_29254

/-- Represents a cube with numbered faces -/
structure NumberedCube where
  faces : Fin 6 → ℕ
  consecutive_multiples : ∀ i : Fin 6, faces i = 12 + 3 * i.val
  opposite_faces_sum_equal : faces 0 + faces 5 = faces 1 + faces 4 ∧ faces 1 + faces 4 = faces 2 + faces 3

/-- The sum of all numbers on the cube is 117 -/
theorem cube_sum_is_117 (cube : NumberedCube) : 
  (Finset.univ.sum cube.faces) = 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_is_117_l292_29254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l292_29298

noncomputable def book_widths : List ℝ := [4, 3/4, 1.25, 3, 2, 7, 5.5]

theorem average_book_width :
  (List.sum book_widths) / (List.length book_widths) = 23.5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l292_29298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubeRoot_of_negative_27_l292_29297

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_27 : cubeRoot (-27) = -3 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubeRoot_of_negative_27_l292_29297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_rate_constant_approx_l292_29209

/-- The cooling rate constant k for an object cooling in air --/
noncomputable def cooling_rate_constant (θ₁ θ₀ θ t : ℝ) : ℝ :=
  (1 / (3 * t)) * Real.log ((θ₁ - θ₀) / (θ - θ₀))

/-- Theorem stating that the cooling rate constant k is approximately 0.17 --/
theorem cooling_rate_constant_approx :
  let θ₁ : ℝ := 60  -- Initial object temperature
  let θ₀ : ℝ := 15  -- Air temperature
  let θ  : ℝ := 42  -- Final object temperature
  let t  : ℝ := 3   -- Time in minutes
  let k := cooling_rate_constant θ₁ θ₀ θ t
  abs (k - 0.17) < 0.005 := by
  sorry

#check cooling_rate_constant_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooling_rate_constant_approx_l292_29209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l292_29261

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 2
  | n + 1 => 1 - sequenceA n

def sum_sequenceA (n : ℕ) : ℤ :=
  (List.range n).map sequenceA |>.foldl (· + ·) 0

theorem sequence_sum_property : 
  sum_sequenceA 2006 - 2 * sum_sequenceA 2007 + sum_sequenceA 2008 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_property_l292_29261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l292_29249

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2 * x + 1

-- Define the derivative of the curve function
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x + 2

-- Theorem statement
theorem tangent_line_at_zero_one :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (λ x ↦ m * (x - x₀) + y₀) = (λ x ↦ 3 * x + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l292_29249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_proof_l292_29263

/-- The circle with center (3, 0) and radius 3 -/
def my_circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- Point P is the midpoint of chord MN -/
def is_midpoint (xm ym xn yn : ℝ) : Prop := (1 : ℝ) = (xm + xn) / 2 ∧ (1 : ℝ) = (ym + yn) / 2

/-- The equation of the line on which chord MN lies -/
def chord_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem chord_equation_proof (xm ym xn yn : ℝ) 
  (h1 : my_circle xm ym) (h2 : my_circle xn yn) (h3 : is_midpoint xm ym xn yn) :
  ∀ x y, chord_equation x y ↔ (∃ t : ℝ, x = xm * (1 - t) + xn * t ∧ y = ym * (1 - t) + yn * t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_proof_l292_29263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_20164_l292_29248

/-- The number of integer pairs (x, y) where 1 ≤ x, y ≤ 1000 and x^2 + y^2 is divisible by 7 -/
def count_pairs : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ 1000 ∧ 1 ≤ p.2 ∧ p.2 ≤ 1000 ∧ (p.1^2 + p.2^2) % 7 = 0)
    (Finset.product (Finset.range 1000) (Finset.range 1000)))

/-- Theorem stating that the count of pairs is 20164 -/
theorem count_pairs_eq_20164 : count_pairs = 20164 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_20164_l292_29248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_property_l292_29229

theorem unique_b_property (b : ℕ) :
  (∃ (a k l : ℕ), k ≠ l ∧ k > 0 ∧ l > 0 ∧
    (b^(k+l) ∣ a^k + b^l) ∧
    (b^(k+l) ∣ a^l + b^k)) →
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_property_l292_29229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_trigonometric_simplification_l292_29271

-- Part 1
theorem trigonometric_equality (α : Real) 
  (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) :
  (Real.sin (2*π - α) * Real.tan (π + α) * Real.cos (-π + α)) / 
  (Real.sin (π / 2 - α) * Real.cos (π / 2 + α)) = 4 / 3 := by
sorry

-- Part 2
theorem trigonometric_simplification (α : Real) (n : Int) :
  (Real.sin (α + n * π) + Real.sin (α - n * π)) / 
  (Real.sin (α + n * π) * Real.cos (α - n * π)) = 
  if n % 2 = 0 then 2 / Real.cos α else -2 / Real.cos α := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_trigonometric_simplification_l292_29271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_assignment_l292_29203

/-- A polyhedron represented as a set of vertices and edges --/
structure Polyhedron where
  vertices : Set ℕ
  edges : Set (ℕ × ℕ)

/-- A function that assigns positive integers to vertices --/
def VertexAssignment (p : Polyhedron) := p.vertices → ℕ+

/-- Two vertices are adjacent if there's an edge between them --/
def adjacent (p : Polyhedron) (v w : ℕ) : Prop :=
  (v, w) ∈ p.edges ∨ (w, v) ∈ p.edges

/-- Two positive integers are relatively prime --/
def relativePrime (a b : ℕ+) : Prop :=
  Nat.gcd a.val b.val = 1

/-- The main theorem --/
theorem polyhedron_vertex_assignment (p : Polyhedron) :
  ∃ (f : VertexAssignment p),
    ∀ (v w : p.vertices),
      (relativePrime (f v) (f w) ↔ adjacent p v.val w.val) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_vertex_assignment_l292_29203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l292_29289

-- Define the slope of the line
variable (k : ℝ)

-- Define the inclination angle of the line
variable (α : ℝ)

-- Theorem statement
theorem inclination_angle_range (h : -1 < k ∧ k < Real.sqrt 3) :
  (0 ≤ α ∧ α < Real.pi / 3) ∨ (3 * Real.pi / 4 < α ∧ α < Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l292_29289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_equals_arctan_41_13_l292_29238

theorem arctan_sum_equals_arctan_41_13 (a b : ℝ) : 
  a = 3/4 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b = Real.arctan (41/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_sum_equals_arctan_41_13_l292_29238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_involution_implies_equal_coefficients_l292_29241

-- Define the function g as noncomputable
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (2*a*x - b) / (c*x - 2*d)

-- State the theorem
theorem involution_implies_equal_coefficients
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : d ≠ 0)
  (h5 : ∀ x, x ≠ 2*d/c → g a b c d (g a b c d x) = x) :
  2*a - 2*d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_involution_implies_equal_coefficients_l292_29241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l292_29208

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 9 * x - 17

/-- The y-coordinate of the directrix of the parabola -/
noncomputable def directrix_y : ℝ := -31/3

/-- Theorem: The directrix of the parabola y = -3x^2 + 9x - 17 is y = -31/3 -/
theorem parabola_directrix : 
  ∀ x : ℝ, ∃ y : ℝ, parabola_equation x = y ∧ 
  (∃ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - directrix_y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l292_29208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_properties_l292_29288

noncomputable def oplus (x y : ℝ) : ℝ := (x * y + x + y) / (x + y + 1)

theorem oplus_properties :
  (∀ x y : ℝ, x > 0 → y > 0 → oplus x y = oplus y x) ∧
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ oplus (oplus x y) z ≠ oplus x (oplus y z)) ∧
  (oplus 2 3 = 11 / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_properties_l292_29288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_singleton_l292_29234

universe u

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equals_singleton :
  N ∩ (U \ M) = {3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_singleton_l292_29234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_5_range_of_a_no_solution_l292_29210

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for part I
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = Set.Ioi 2 ∪ Set.Iic (-4/3) := by sorry

-- Theorem for part II
theorem range_of_a_no_solution :
  {a : ℝ | ∀ x, f x - 4 ≠ 1/a} = Set.Ioo (-2/3) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_greater_than_5_range_of_a_no_solution_l292_29210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l292_29251

-- Define the constants
noncomputable def a : ℝ := 5^(1/5)
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log (Real.sin ((2/3) * Real.pi)) / Real.log 5

-- State the theorem
theorem order_of_constants : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l292_29251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_min_area_rhombus_l292_29213

/-- Definition of an ellipse -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ
  not_circle : semi_major_axis ≠ semi_minor_axis

/-- Definition of a rhombus -/
structure Rhombus where
  center : ℝ × ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- A rhombus is tangent to an ellipse at all four of its sides -/
def is_tangent (r : Rhombus) (e : Ellipse) : Prop :=
  sorry

/-- The area of a rhombus -/
noncomputable def rhombus_area (r : Rhombus) : ℝ :=
  (1 / 2) * r.diagonal1 * r.diagonal2

/-- Theorem: Unique minimum area rhombus tangent to a non-circular ellipse -/
theorem unique_min_area_rhombus (e : Ellipse) :
  ∃! r : Rhombus, is_tangent r e ∧
    (∀ r' : Rhombus, is_tangent r' e → rhombus_area r ≤ rhombus_area r') ∧
    r.center = e.center :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_min_area_rhombus_l292_29213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_time_l292_29211

/-- The time (in years) for a machine to depreciate from initial value to final value -/
noncomputable def depreciation_time (initial_value : ℝ) (final_value : ℝ) (rate : ℝ) : ℝ :=
  Real.log (final_value / initial_value) / Real.log (1 - rate)

/-- Theorem stating that the depreciation time for the given conditions is approximately 2 years -/
theorem machine_depreciation_time :
  let initial_value : ℝ := 1200
  let final_value : ℝ := 972
  let rate : ℝ := 0.10
  abs (depreciation_time initial_value final_value rate - 2) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval depreciation_time 1200 972 0.10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_time_l292_29211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_9874_l292_29242

-- Define the rounding function as noncomputable
noncomputable def round_to_precision (x : ℝ) (precision : ℕ) : ℝ :=
  (⌊x * 10^precision + 0.5⌋ : ℝ) / 10^precision

-- Theorem statement
theorem rounding_9874 :
  (round_to_precision 9.874 0 = 10) ∧
  (round_to_precision 9.874 1 = 9.9) ∧
  (round_to_precision 9.874 2 = 9.87) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_9874_l292_29242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l292_29262

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := n

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(n-1)

-- Define S_n (sum of first n terms of a_n)
noncomputable def S (n : ℕ) : ℝ := n * (n + 1) / 2

-- Define T_n (sum of first n terms of a_n * b_n)
noncomputable def T (n : ℕ) : ℝ := n - 2 - (n - 1) * 2^n

theorem arithmetic_geometric_sequence_properties :
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (b 1 = 1) ∧
  (b 2 * S 2 = 6) ∧
  (b 2 + S 3 = 8) ∧
  (∀ n, a n = n) ∧
  (∀ n, b n = 2^(n-1)) ∧
  (∀ n, T n = n - 2 - (n - 1) * 2^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l292_29262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_l292_29206

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    Finset.toSet {a, b, c, d, e} = Finset.toSet {1, 2, 3, 6, 0}

def abc (n : ℕ) : ℕ := n / 100

def bcd (n : ℕ) : ℕ := (n / 10) % 1000

def cde (n : ℕ) : ℕ := n % 1000

theorem five_digit_divisibility (n : ℕ) :
  is_valid_number n →
  abc n % 3 = 0 →
  bcd n % 4 = 0 →
  cde n % 5 = 0 →
  n / 10000 = 1 := by
  sorry

#check five_digit_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_l292_29206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinach_volume_percentage_l292_29283

/-- Calculates the percentage of cooked spinach volume relative to its initial raw volume -/
noncomputable def cookedSpinachPercentage (initialRawSpinach totalQuiche creamCheese eggs : ℝ) : ℝ :=
  let cookedSpinach := totalQuiche - (creamCheese + eggs)
  (cookedSpinach / initialRawSpinach) * 100

/-- Proves that the cooked spinach is 20% of its initial volume given the problem conditions -/
theorem spinach_volume_percentage :
  cookedSpinachPercentage 40 18 6 4 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinach_volume_percentage_l292_29283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_unity_cubic_l292_29201

/-- A complex number z is a root of unity if there exists a positive integer n such that z^n = 1 -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

/-- The equation z^3 + p*z + q = 0 where p and q are real numbers -/
def cubic_equation (p q : ℝ) (z : ℂ) : Prop :=
  z^3 + p*z + q = 0

/-- The theorem stating that the maximum number of roots of unity 
    that can be roots of z^3 + pz + q = 0 (where p and q are real) is 3 -/
theorem max_roots_of_unity_cubic (p q : ℝ) :
  ∃ (roots : Finset ℂ), 
    (∀ z ∈ roots, is_root_of_unity z ∧ cubic_equation p q z) ∧
    (∀ z, is_root_of_unity z → cubic_equation p q z → z ∈ roots) ∧
    roots.card ≤ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_of_unity_cubic_l292_29201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_club_male_diff_is_24_l292_29230

/-- Represents the student body with given conditions -/
structure StudentBody where
  total : ℕ
  juniors_percent : ℚ
  not_sophomores_percent : ℚ
  seniors : ℕ
  freshmen_male_percent : ℚ
  sophomore_male_percent : ℚ
  junior_male_percent : ℚ
  senior_male_percent : ℚ
  freshmen_science_percent : ℚ
  sophomore_drama_percent : ℚ
  junior_debate_percent : ℚ

/-- Calculates the difference between male freshmen in Science Club and male sophomores in Drama Club -/
def club_male_diff (sb : StudentBody) : ℤ :=
  let freshmen := sb.total - (↑sb.total * sb.juniors_percent).floor - sb.seniors - (↑sb.total * (1 - sb.not_sophomores_percent)).floor
  let freshmen_science := (↑freshmen * sb.freshmen_science_percent).floor
  let male_freshmen_science := (↑freshmen_science * sb.freshmen_male_percent).floor
  let sophomores := (↑sb.total * (1 - sb.not_sophomores_percent)).floor
  let sophomore_drama := (↑sophomores * sb.sophomore_drama_percent).floor
  let male_sophomore_drama := (↑sophomore_drama * sb.sophomore_male_percent).floor
  male_freshmen_science - male_sophomore_drama

/-- The main theorem to prove -/
theorem club_male_diff_is_24 (sb : StudentBody) 
  (h1 : sb.total = 1200)
  (h2 : sb.juniors_percent = 23 / 100)
  (h3 : sb.not_sophomores_percent = 70 / 100)
  (h4 : sb.seniors = 160)
  (h5 : sb.freshmen_male_percent = 55 / 100)
  (h6 : sb.sophomore_male_percent = 60 / 100)
  (h7 : sb.junior_male_percent = 48 / 100)
  (h8 : sb.senior_male_percent = 52 / 100)
  (h9 : sb.freshmen_science_percent = 30 / 100)
  (h10 : sb.sophomore_drama_percent = 20 / 100)
  (h11 : sb.junior_debate_percent = 25 / 100) :
  club_male_diff sb = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_club_male_diff_is_24_l292_29230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_inv_ln2_l292_29259

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x * (2 : ℝ) ^ x

-- State the theorem
theorem f_min_at_neg_inv_ln2 :
  ∃ (x_min : ℝ), x_min = -1 / Real.log 2 ∧
  ∀ (x : ℝ), f x_min ≤ f x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_inv_ln2_l292_29259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_determinant_l292_29295

/-- Definition of an equilateral triangle using its angles -/
def IsEquilateralTriangle (A B C : ℝ) : Prop :=
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3

/-- The determinant of a specific matrix involving sines of angles in an equilateral triangle -/
theorem equilateral_triangle_determinant (A B C : ℝ) : 
  IsEquilateralTriangle A B C →
  Matrix.det !![Real.sin A, 1, 1; 1, Real.sin B, 1; 1, 1, Real.sin C] = -Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_determinant_l292_29295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l292_29274

/-- The speed of a train given its length and time to cross a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time * 3.6

/-- Theorem stating the speed of a train with given parameters -/
theorem train_speed_calculation :
  let train_length : ℝ := 2500
  let crossing_time : ℝ := 35
  let calculated_speed := train_speed train_length crossing_time
  ∃ (ε : ℝ), ε > 0 ∧ |calculated_speed - 257.14| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l292_29274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuo_lake_crab_analysis_l292_29236

structure CrabData where
  total_crabs : ℕ
  sample_size : ℕ
  quality_groups : List (ℝ × ℝ)
  frequencies : List ℕ
  x_y_relation : ℝ → ℝ

def tuo_lake_crabs : CrabData := {
  total_crabs := 1000,
  sample_size := 50,
  quality_groups := [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350)],
  frequencies := [2, 4, 6, 12, 8, 14, 4],
  x_y_relation := λ x => 2 * x
}

theorem tuo_lake_crab_analysis (data : CrabData) :
  data.frequencies[2]! = 6 ∧ 
  data.frequencies[3]! = 12 ∧
  200 ≤ (data.quality_groups[4]!.1) ∧ (data.quality_groups[4]!.2) < 250 ∧
  (((data.frequencies[4]! + data.frequencies[5]! + data.frequencies[6]!) / data.sample_size) * data.total_crabs : ℕ) = 520 := by
  sorry

#check tuo_lake_crab_analysis tuo_lake_crabs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuo_lake_crab_analysis_l292_29236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_alpha_l292_29253

theorem power_function_alpha (α : ℝ) : 2 = (8 : ℝ)^α → α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_alpha_l292_29253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l292_29235

theorem matrix_similarity_theorem (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (h : A ^ 2 + 5 • (1 : Matrix (Fin 2) (Fin 2) ℤ) = 0) : 
  ∃ (C : Matrix (Fin 2) (Fin 2) ℤ), 
    IsUnit (Matrix.det C) ∧ 
    (C * A * C⁻¹ = !![1, 2; -3, -1] ∨ C * A * C⁻¹ = !![0, 1; -5, 0]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_similarity_theorem_l292_29235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l292_29240

/-- Calculates the time (in seconds) for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance_m / train_speed_ms

/-- Theorem: A train with speed 72 km/h and length 230 m takes 26 seconds to cross a 290 m long platform -/
theorem train_crossing_time_example :
  train_crossing_time 72 230 290 = 26 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_example_l292_29240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_4_l292_29287

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_4_or_less : ℕ := 6

/-- The probability of getting a sum greater than 4 when two fair dice are tossed -/
theorem prob_sum_greater_than_4 : (1 : ℚ) - (outcomes_4_or_less : ℚ) / (total_outcomes : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_4_l292_29287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_k_l292_29272

/-- The interior angle of an equiangular decagon in degrees -/
def decagon_angle : ℝ := 144

/-- The number of sides of the first polygon P₁ -/
def n₁ : ℕ := 10

/-- The function to calculate the number of sides of the second polygon P₂ given k -/
def n₂ (k : ℕ) : ℕ := n₁ * k

/-- The function to calculate the interior angle of P₂ in degrees given k -/
def p₂_angle (k : ℕ) : ℝ := k * decagon_angle

/-- A polygon is valid if it has at least 3 sides -/
def is_valid_polygon (n : ℕ) : Prop := n ≥ 3

/-- The smallest k such that P₂ is a valid polygon -/
theorem smallest_valid_k : (∃ k : ℕ, k > 1 ∧ is_valid_polygon (n₂ k) ∧ 
  ∀ j, j > 1 → j < k → ¬is_valid_polygon (n₂ j)) → 
  (∃ k : ℕ, k > 1 ∧ is_valid_polygon (n₂ k) ∧ 
  ∀ j, j > 1 → j < k → ¬is_valid_polygon (n₂ j)) ∧ k = 2 := by
  intro h
  apply And.intro h
  sorry

#check smallest_valid_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_k_l292_29272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l292_29221

theorem inequality_problem (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (if |a| > |b| then 1 else 0) + 
  (if a + b > a * b then 1 else 0) + 
  (if (a / b) + (b / a) > 2 then 1 else 0) + 
  (if (a^2 / b) < 2 * a - b then 1 else 0) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l292_29221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l292_29282

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * (x - 1) * Real.exp x + x^2

def g (k : ℝ) (x : ℝ) : ℝ := x^2 + (k + 2) * x

noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := k * x * Real.exp x + 2 * x

theorem problem_solution (k : ℝ) :
  (k = 1/2 → ∃! x, x > 0 ∧ g k x = f_derivative k x) ∧
  (k ≤ -1 → ∃ m : ℝ, m = 1 ∧ ∀ x ∈ Set.Icc k 1, f k x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l292_29282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l292_29227

open Real

theorem trig_simplification (x : ℝ) 
  (h : 1 + cos x + cos (2*x) ≠ 0) : 
  (sin x + sin (2*x)) / (1 + cos x + cos (2*x)) = tan x :=
by
  -- Assuming the following trigonometric identities:
  have sin_double : sin (2*x) = 2 * sin x * cos x := by exact sin_two_mul x
  have cos_double : cos (2*x) = 2 * cos x ^ 2 - 1 := by exact cos_two_mul x
  have tan_def : tan x = sin x / cos x := by exact tan_eq_sin_div_cos x

  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l292_29227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_reading_progress_l292_29237

/-- Represents the number of pages in Sophia's book -/
noncomputable def total_pages : ℚ := 270

/-- Represents the number of pages Sophia has finished reading -/
noncomputable def pages_read : ℚ := (total_pages + 90) / 2

/-- Represents the fraction of the book Sophia has finished -/
noncomputable def fraction_finished : ℚ := pages_read / total_pages

/-- Proves that the fraction of the book Sophia finished is 2/3 -/
theorem sophia_reading_progress : fraction_finished = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sophia_reading_progress_l292_29237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_player_a_winning_strategy_integer_coordinates_l292_29218

/-- Represents a point on the plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a color (white or black) -/
inductive Color
| White
| Black

/-- Represents the game state -/
structure GameState where
  marked_points : List (Point × Color)

/-- Represents a line on the plane -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Checks if a point is on the left side of a line -/
def isLeftOfLine (p : Point) (l : Line) : Prop :=
  p.y < l.slope * p.x + l.intercept

/-- Represents a winning configuration for Player B -/
def isWinningConfigurationForB (gs : GameState) : Prop :=
  ∃ l : Line, ∀ p : Point × Color,
    p ∈ gs.marked_points →
      (p.2 = Color.Black → isLeftOfLine p.1 l) ∧
      (p.2 = Color.White → ¬isLeftOfLine p.1 l)

/-- Player A's strategy function (simplified) -/
noncomputable def playerAStrategy (gs : GameState) : Point :=
  ⟨0, 0⟩  -- Placeholder implementation

/-- Theorem: Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∀ gs : GameState, ¬isWinningConfigurationForB (GameState.mk (gs.marked_points ++ [(playerAStrategy gs, Color.White)] ++ [(playerAStrategy gs, Color.Black)])) :=
by
  sorry

/-- Theorem: Player A has a winning strategy even when restricted to integer coordinates -/
theorem player_a_winning_strategy_integer_coordinates :
  ∀ gs : GameState, ∃ p : Point,
    (∃ n : ℤ, p.x = n) ∧ (∃ m : ℤ, p.y = m) ∧
    ¬isWinningConfigurationForB (GameState.mk (gs.marked_points ++ [(p, Color.White)] ++ [(p, Color.Black)])) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_player_a_winning_strategy_integer_coordinates_l292_29218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_properties_l292_29294

variable {A B : Type*}
variable (f : A → B)

theorem bijection_properties :
  Function.Bijective f →
  (∀ a₁ a₂ : A, a₁ ≠ a₂ → f a₁ ≠ f a₂) ∧
  (∀ b : B, ∃ a : A, f a = b) ∧
  (Set.range f = Set.univ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bijection_properties_l292_29294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_and_k_range_l292_29217

/-- Regular tetrahedral pyramid with a plane through lateral edge and height -/
structure RegularTetrahedralPyramid where
  /-- Ratio of cross-section area to total surface area -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- Dihedral angle at the base of the pyramid -/
noncomputable def dihedral_angle (p : RegularTetrahedralPyramid) : ℝ :=
  2 * Real.arctan (2 * p.k * Real.sqrt 3)

/-- Theorem stating the dihedral angle and permissible values of k -/
theorem dihedral_angle_and_k_range (p : RegularTetrahedralPyramid) :
  dihedral_angle p = 2 * Real.arctan (2 * p.k * Real.sqrt 3) ∧
  p.k < Real.sqrt 3 / 6 := by
  sorry

#check dihedral_angle_and_k_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_and_k_range_l292_29217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_f_four_roots_l292_29202

/-- A function f defined as f(x) = x^2 + 4x + c for all real x -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The theorem stating that f(f(x)) has exactly 4 distinct real roots iff c ∈ (-1, 3) -/
theorem f_comp_f_four_roots (c : ℝ) :
  (∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, f_comp_f c x = 0) ↔ -1 < c ∧ c < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comp_f_four_roots_l292_29202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_degree_polynomial_roots_l292_29292

theorem odd_degree_polynomial_roots (n : ℕ) (hn : Odd n) 
  (P : Polynomial ℝ) (hP : P.degree = n) : 
  (∃ (k : ℕ), Odd k ∧ k > 0 ∧ (∃ (roots : Finset ℝ), roots.card = k ∧ ∀ r ∈ roots, P.eval r = 0)) ∧
  (∃ (r : ℝ), P.eval r = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_degree_polynomial_roots_l292_29292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l292_29223

/-- The minimum distance from the origin to the line 4x + 4y - 7 = 0 is 7√2/8 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | 4 * p.1 + 4 * p.2 = 7}
  let min_dist_to_origin := Real.sqrt (7^2 / (4^2 + 4^2))
  (∀ p ∈ line, Real.sqrt (p.1^2 + p.2^2) ≥ min_dist_to_origin) ∧ 
  (∃ q ∈ line, Real.sqrt (q.1^2 + q.2^2) = min_dist_to_origin) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l292_29223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_equal_area_l292_29284

/-- Given an isosceles right triangle with a square inscribed such that two sides
    coincide with the legs of the triangle, prove that another square inscribed
    with sides parallel to the hypotenuse has the same area. -/
theorem inscribed_squares_equal_area (leg : ℝ) (h : leg > 0) :
  let first_square_area := (leg / Real.sqrt 2) ^ 2
  let second_square_area := (leg / Real.sqrt 2) ^ 2
  first_square_area = 784 → second_square_area = 784 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_equal_area_l292_29284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_alpha_l292_29243

-- Define the angle α and point P
variable (α : Real)
variable (a : Real)
def P : Real × Real := (3 * a, -4 * a)

-- State the theorem
theorem sin_plus_cos_alpha (h1 : a < 0) : Real.sin α + Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_alpha_l292_29243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_at_least_twice_inradius_l292_29220

/-- Helper function to calculate the semi-perimeter of a triangle -/
noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2

/-- Helper function to calculate the area of a triangle using Heron's formula -/
noncomputable def area (a b c : ℝ) : ℝ := 
  Real.sqrt ((s a b c) * (s a b c - a) * (s a b c - b) * (s a b c - c))

/-- For any triangle with side lengths a, b, and c, where R is the radius of its 
    circumscribed circle and r is the radius of its inscribed circle, R ≥ 2r holds, 
    with equality if and only if the triangle is equilateral. -/
theorem circumradius_at_least_twice_inradius 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (R : ℝ) (r : ℝ)
  (hR : R = (a * b * c) / (4 * area a b c))
  (hr : r = (2 * area a b c) / (a + b + c)) :
  R ≥ 2 * r ∧ (R = 2 * r ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_at_least_twice_inradius_l292_29220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_specific_octagonal_prism_eval_lateral_surface_area_l292_29256

/-- The lateral surface area of a regular octagonal prism -/
noncomputable def lateral_surface_area_octagonal_prism (volume : ℝ) (height : ℝ) : ℝ :=
  16 * Real.sqrt (2.2 * (Real.sqrt 2 - 1))

/-- Theorem: The lateral surface area of a regular octagonal prism with volume 8 cubic meters and height 2.2 meters -/
theorem lateral_surface_area_specific_octagonal_prism :
  lateral_surface_area_octagonal_prism 8 2.2 = 16 * Real.sqrt (2.2 * (Real.sqrt 2 - 1)) := by
  -- The proof is omitted
  sorry

/-- Evaluate the lateral surface area for the specific case -/
theorem eval_lateral_surface_area :
  ∃ (x : ℝ), lateral_surface_area_octagonal_prism 8 2.2 = x ∧ x > 0 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_specific_octagonal_prism_eval_lateral_surface_area_l292_29256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_points_l292_29290

/-- Represents a point on the circle -/
structure Point where
  color : Bool  -- True for red, False for blue
  connections : Nat

/-- The problem setup -/
structure CircleProblem where
  total_points : Nat
  points : List Point
  no_duplicate_red_connections : ∀ p1 p2 : Point, p1 ∈ points → p2 ∈ points → 
    p1.color = true → p2.color = true → p1 ≠ p2 → p1.connections ≠ p2.connections

theorem max_red_points (setup : CircleProblem) (h1 : setup.total_points = 100) : 
  (setup.points.filter (λ p => p.color)).length ≤ 50 := by
  sorry

#check max_red_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_red_points_l292_29290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l292_29279

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def PerpLines (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Given two lines l₁: ax + y + 3 = 0 and l₂: x + (2a-3)y = 4 that are perpendicular, prove a = 1 -/
theorem perpendicular_lines_a_equals_one (a : ℝ) :
  PerpLines (-a) (1/(2*a-3)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_equals_one_l292_29279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_l292_29239

/-- A group with female and male members -/
structure GroupMembers where
  total : ℕ
  female : ℕ
  male : ℕ

/-- The probability of selecting k items from n items -/
def selectProb (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (Nat.choose n.succ k : ℚ)

/-- The expected value of X (number of female members selected) -/
def expectedX (g : GroupMembers) (selected : ℕ) : ℚ :=
  (0 : ℚ) * selectProb g.male selected +
  (1 : ℚ) * selectProb g.female 1 * selectProb g.male (selected - 1) +
  (2 : ℚ) * selectProb g.female selected

/-- Theorem: The expected value of X is 6/5 for the given group -/
theorem expected_value_X (g : GroupMembers) (h1 : g.total = 5) (h2 : g.female = 3) (h3 : g.male = 2) :
  expectedX g 2 = 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_X_l292_29239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l292_29277

-- Define the original parabola
noncomputable def original_parabola (x : ℝ) : ℝ := (1/2) * x^2 + 1

-- Define the rotation transformation
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem parabola_rotation :
  ∀ x : ℝ, 
  let (x', y') := rotate_180 (x, original_parabola x)
  y' = -(1/2) * x'^2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l292_29277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l292_29219

def α_set : Set ℝ := {-2, -1, -1/2, 2}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

theorem power_function_properties (α : ℝ) (hα : α ∈ α_set) :
  is_even_function (fun x ↦ x^α) ∧
  is_decreasing_on (fun x ↦ x^α) (Set.Ioi 0) →
  α = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l292_29219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_F_value_when_f_eq_2f_l292_29225

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := f x * f' x + (f x)^2

-- Theorem for the smallest positive period of F
theorem smallest_positive_period_F : ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), F (x + T) = F x) ∧ (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), F (x + T') = F x) → T' ≥ T) ∧ T = Real.pi := by
  sorry

-- Theorem for the value of the expression when f(x) = 2f'(x)
theorem value_when_f_eq_2f' : 
  (∀ (x : ℝ), f x = 2 * f' x) → 
  (∀ (x : ℝ), (1 + Real.sin x^2) / (Real.cos x^2 - Real.sin x * Real.cos x) = 11/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_F_value_when_f_eq_2f_l292_29225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_parallel_lines_l292_29250

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- The slope of a line ax + by = c is -a/b -/
def line_slope (a b : ℚ) : ℚ := -a / b

/-- The x-intercept of a line ax + by = c is c/a when y = 0 -/
def x_intercept (a c : ℚ) : ℚ := c / a

/-- Given that line l₁: (a+2)x+3y=5 is parallel to line l₂: (a-1)x+2y=6,
    prove that the x-intercept of line l₁ is 5/9 -/
theorem x_intercept_of_parallel_lines (a : ℚ) :
  parallel (line_slope (a + 2) 3) (line_slope (a - 1) 2) →
  x_intercept 9 5 = 5 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_parallel_lines_l292_29250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l292_29215

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (1/2 * x - Real.pi/6)

theorem sine_function_properties :
  let amplitude : ℝ := 3
  let period : ℝ := 4 * Real.pi
  let frequency : ℝ := 1 / (4 * Real.pi)
  let phase (x : ℝ) : ℝ := 1/2 * x - Real.pi/6
  let initial_phase : ℝ := -Real.pi/6
  (∀ x, f x = amplitude * Real.sin (phase x)) ∧
  (∀ x, f (x + period) = f x) ∧
  (frequency = 1 / period) ∧
  (∀ x, phase x = 1/2 * x + initial_phase) :=
by
  sorry

#check sine_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l292_29215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l292_29258

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem matrix_power_eight :
  A^8 = !![(-128 : ℝ), -128 * Real.sqrt 3; 128 * Real.sqrt 3, (-128 : ℝ)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_eight_l292_29258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_arrangement_exists_l292_29267

/-- Represents a circular cylinder in 3D space -/
structure Cylinder where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  height : ℝ

/-- The set of points that make up a cylinder -/
def set_of_cylinder (c : Cylinder) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Checks if two cylinders have a common point -/
def have_common_point (c1 c2 : Cylinder) : Prop :=
  ∃ (p : ℝ × ℝ × ℝ), p ∈ set_of_cylinder c1 ∧ p ∈ set_of_cylinder c2

/-- Theorem stating that there exists an arrangement of six cylinders
    where each cylinder has a common point with every other cylinder -/
theorem cylinder_arrangement_exists :
  ∃ (d h : ℝ) (cylinders : Fin 6 → Cylinder),
    (∀ i, (cylinders i).radius = d ∧ (cylinders i).height = h) ∧
    (∀ i j, i ≠ j → have_common_point (cylinders i) (cylinders j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_arrangement_exists_l292_29267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_composition_l292_29214

theorem decimal_composition (hundred_thousands ten_thousands thousands hundreds tens ones tenths hundredths : ℕ) :
  let decimal : ℚ := 100000 * hundred_thousands + 10000 * ten_thousands + 1000 * thousands + 100 * hundreds + 10 * tens + ones + (1 / 10 : ℚ) * tenths + (1 / 100 : ℚ) * hundredths
  hundred_thousands = 5 ∧ ten_thousands = 0 ∧ thousands = 0 ∧ hundreds = 6 ∧ tens = 3 ∧ ones = 0 ∧ tenths = 0 ∧ hundredths = 6 →
  decimal = 500630.06 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_composition_l292_29214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_on_interval_l292_29281

theorem sin_increasing_on_interval :
  ∀ x y : ℝ, -π/2 ≤ x ∧ x < y ∧ y ≤ π/2 → Real.sin x < Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_increasing_on_interval_l292_29281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l292_29200

/-- The length of a train given its speed and the time it takes to cross a platform of known length. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 280 → 
  crossing_time = 26 → 
  train_speed * crossing_time - platform_length = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l292_29200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l292_29212

theorem triangle_area_proof (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 4) 
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) : 
  Real.sqrt 3 * 8 = Complex.abs (z₁.im * z₂.re - z₁.re * z₂.im) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l292_29212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_crossing_time_approx_15_seconds_l292_29244

/-- Represents the properties of a train and its movement --/
structure TrainMovement where
  train_length : ℝ
  tunnel_length : ℝ
  tunnel_crossing_time : ℝ
  platform_length : ℝ

/-- Calculates the time taken for the train to cross the platform --/
noncomputable def platform_crossing_time (tm : TrainMovement) : ℝ :=
  let total_tunnel_distance := tm.train_length + tm.tunnel_length
  let train_speed := total_tunnel_distance / tm.tunnel_crossing_time
  let total_platform_distance := tm.train_length + tm.platform_length
  total_platform_distance / train_speed

/-- Theorem stating that the train will take approximately 15 seconds to cross the platform --/
theorem platform_crossing_time_approx_15_seconds 
  (tm : TrainMovement) 
  (h1 : tm.train_length = 330)
  (h2 : tm.tunnel_length = 1200)
  (h3 : tm.tunnel_crossing_time = 45)
  (h4 : tm.platform_length = 180) :
  ∃ ε > 0, |platform_crossing_time tm - 15| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_crossing_time_approx_15_seconds_l292_29244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_proof_l292_29260

/-- Given two lines and a solution point, prove the intersection point -/
theorem intersection_point_proof 
  (b m : ℝ) 
  (h1 : m = 3 * (-1) + 2) 
  (h2 : m = -(-1) + b) : 
  ∃ (x y : ℝ), x = -1 ∧ y = -1 ∧ y = -x + b ∧ y = 3*x + 2 := by
  sorry

#check intersection_point_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_proof_l292_29260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l292_29273

theorem min_omega_value (α : ℝ) (ω : ℝ) :
  Real.tan α = Real.sqrt 2 - 1 →
  0 < α →
  α < π / 2 →
  ω > 0 →
  (∃ k : ℤ, ω = -(3/2) + 3*k - (6/π)*α) →
  (∀ x : ℝ, Real.sin (ω * (x - π/3) - 2*α) = Real.sin (ω * (-x - π/3) - 2*α)) →
  ω ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l292_29273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_a_is_two_a_value_when_A_equals_B_l292_29207

-- Define the set A
noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 3*a - 1) < 0}

-- Define the function y
noncomputable def y (a x : ℝ) : ℝ := Real.log ((2*a - x) / (x - (a^2 + 1)))

-- Define the domain B of the function y
noncomputable def B (a : ℝ) : Set ℝ := {x | (2*a - x) / (x - (a^2 + 1)) > 0}

-- Theorem 1: When a=2, B is the open interval (4,5)
theorem domain_when_a_is_two : B 2 = Set.Ioo 4 5 := by sorry

-- Theorem 2: When A = B, a = -1
theorem a_value_when_A_equals_B : ∃ (a : ℝ), A a = B a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_when_a_is_two_a_value_when_A_equals_B_l292_29207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l292_29270

def repeating_decimal : ℚ := 78 / 99

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 59 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fraction_parts_l292_29270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coloring_cells_l292_29299

/-- A coloring of a 5x5 grid -/
def Coloring := Fin 5 → Fin 5 → Bool

/-- Checks if a 2x3 or 3x2 rectangle contains a colored cell -/
def valid_rectangle (c : Coloring) (i j : Fin 5) (horizontal : Bool) : Prop :=
  ∃ (x y : Fin 5), (x < i + (if horizontal then 2 else 3)) ∧ 
                   (y < j + (if horizontal then 3 else 2)) ∧ 
                   c x y = true

/-- A coloring is valid if every 2x3 or 3x2 rectangle contains a colored cell -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ (i j : Fin 5) (horizontal : Bool), 
    (i + (if horizontal then 2 else 3) ≤ 5) → 
    (j + (if horizontal then 3 else 2) ≤ 5) → 
    valid_rectangle c i j horizontal

/-- Counts the number of colored cells in a coloring -/
def colored_cells (c : Coloring) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) fun i =>
   Finset.sum (Finset.univ : Finset (Fin 5)) fun j =>
   if c i j then 1 else 0)

/-- The main theorem stating that the minimum number of cells to color is 4 -/
theorem min_coloring_cells :
  (∃ (c : Coloring), valid_coloring c ∧ colored_cells c = 4) ∧
  (∀ (c : Coloring), valid_coloring c → colored_cells c ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coloring_cells_l292_29299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l292_29276

/-- An arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence starting with 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sequence c_n defined as the sum of a_n and b_n -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  c_seq d r (k - 1) = 200 ∧ c_seq d r (k + 1) = 1200 → c_seq d r k = 423 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l292_29276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l292_29231

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α - Real.sin β = 1 - Real.sqrt 3 / 2)
  (h2 : Real.cos α - Real.cos β = 1 / 2) : 
  Real.cos (α - β) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_beta_l292_29231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_simon_separation_l292_29293

/-- The time it takes for two people traveling perpendicular to each other to be 60 miles apart -/
noncomputable def separation_time (speed_east speed_south : ℝ) : ℝ :=
  Real.sqrt (3600 / (speed_east^2 + speed_south^2))

/-- Theorem stating that Adam and Simon will be 60 miles apart after 6 hours -/
theorem adam_simon_separation :
  separation_time 8 6 = 6 := by
  -- Unfold the definition of separation_time
  unfold separation_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adam_simon_separation_l292_29293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_after_e_l292_29245

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem f_decreasing_after_e (a b : ℝ) (h1 : Real.exp 1 < b) (h2 : b < a) :
  f a < f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_after_e_l292_29245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l292_29233

noncomputable def f (x φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

noncomputable def f_shifted (x φ : ℝ) : ℝ := f (x + Real.pi/6) φ

theorem symmetry_implies_phi_value (φ : ℝ) 
  (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∀ x, f_shifted x φ = f_shifted (-x) φ) :
  |φ| = 5*Real.pi/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l292_29233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_first_13_even_numbers_l292_29296

/-- The nth even number -/
def evenNumber (n : ℕ) : ℕ := 2 * n

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := 
  (List.range n).map (fun i => evenNumber (i + 1)) |>.sum

/-- The average of the first n even numbers -/
noncomputable def averageFirstEvenNumbers (n : ℕ) : ℚ := 
  (sumFirstEvenNumbers n : ℚ) / n

theorem average_first_13_even_numbers : 
  averageFirstEvenNumbers 13 = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_first_13_even_numbers_l292_29296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_with_large_prime_divisor_l292_29246

/-- The set of positive integers n such that n^2 + 1 has a prime divisor greater than 2n + √(2n) is infinite. -/
theorem infinite_n_with_large_prime_divisor :
  {n : ℕ+ | ∃ p : ℕ, Nat.Prime p ∧ p ∣ (n : ℕ)^2 + 1 ∧ (p : ℝ) > 2*(n : ℝ) + Real.sqrt (2*(n : ℝ))}.Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_n_with_large_prime_divisor_l292_29246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_l292_29285

theorem q_value (Q : ℝ) (h : Real.sqrt (2 * Q^3) = 64 * (32 : ℝ)^(1/16)) : Q = 2^(123/24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_l292_29285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_l292_29266

/-- Represents a material point with mass and position -/
structure MaterialPoint where
  mass : ℝ
  position : ℝ

/-- Calculates the center of mass of a system of material points -/
noncomputable def centerOfMass (points : List MaterialPoint) : ℝ :=
  (points.map (λ p => p.mass * p.position)).sum / (points.map (λ p => p.mass)).sum

/-- The problem statement -/
theorem center_of_mass_distance (m₁ m₂ m₃ d₁₂ d₂₃ : ℝ) 
  (h₁ : m₁ = 2)
  (h₂ : m₂ = 3)
  (h₃ : m₃ = 4)
  (h₄ : d₁₂ = 25)
  (h₅ : d₂₃ = 75) :
  let points := [
    { mass := m₁, position := 0 },
    { mass := m₂, position := d₁₂ },
    { mass := m₃, position := d₁₂ + d₂₃ }
  ]
  abs (centerOfMass points - 52.8) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_distance_l292_29266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_separation_condition_l292_29264

theorem root_separation_condition (m n p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
  (∃ x₃ x₄ : ℝ, x₃ ≠ x₄ ∧ x₃^2 + p*x₃ + q = 0 ∧ x₄^2 + p*x₄ + q = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ, (x₁ < x₃ ∧ x₃ < x₂ ∧ x₄ < x₁) ∨ 
                      (x₁ < x₄ ∧ x₄ < x₂ ∧ x₃ < x₁) ∨
                      (x₁ < x₃ ∧ x₃ < x₂ ∧ x₂ < x₄) ∨ 
                      (x₁ < x₄ ∧ x₄ < x₂ ∧ x₂ < x₃)) →
  (n - q)^2 + (m - p)*(m*q - n*p) < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_separation_condition_l292_29264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_seven_l292_29204

theorem tan_alpha_equals_seven (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 4) = -3 / 5) : 
  Real.tan α = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equals_seven_l292_29204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l292_29252

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (1 < x ∧ x < 3) → (3 : ℝ)^x > 1) ∧
  (∃ x : ℝ, (3 : ℝ)^x > 1 ∧ ¬(1 < x ∧ x < 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l292_29252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_b_bottles_l292_29265

/-- Proves that Company B sold 350 bottles given the conditions of the problem -/
theorem company_b_bottles : ℕ := by
  -- Define the given conditions
  let company_a_price : ℚ := 4
  let company_b_price : ℚ := 7/2
  let company_a_bottles : ℕ := 300
  let revenue_difference : ℚ := 25

  -- Define the number of bottles sold by Company B
  let company_b_bottles : ℕ := 350

  -- State the theorem
  have h : (company_b_price * company_b_bottles : ℚ) = 
           (company_a_price * company_a_bottles : ℚ) + revenue_difference := by
    sorry

  -- Return the number of bottles sold by Company B
  exact company_b_bottles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_b_bottles_l292_29265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l292_29205

theorem trigonometric_expression_value (α : Real) 
  (h : 5 * (Real.sin α)^2 - 7 * (Real.sin α) - 6 = 0) :
  (Real.sin (-α - 3/2 * Real.pi) * Real.sin (3/2 * Real.pi - α) * Real.tan (2 * Real.pi - α)^2) / 
  (Real.cos (Real.pi/2 - α) * Real.cos (Real.pi/2 + α) * Real.cos (Real.pi - α)^2) = 25/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l292_29205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_for_500_miles_l292_29291

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  gallons_per_1000_miles : ℚ

/-- Calculates the fuel needed for a given distance -/
def fuel_needed (car : CarFuelEfficiency) (distance : ℚ) : ℚ :=
  (car.gallons_per_1000_miles * distance) / 1000

/-- Theorem: The fuel needed for a 500-mile trip is 20 gallons -/
theorem fuel_for_500_miles (car : CarFuelEfficiency) 
  (h : car.gallons_per_1000_miles = 40) : 
  fuel_needed car 500 = 20 := by
  sorry

#check fuel_for_500_miles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_for_500_miles_l292_29291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l292_29226

/-- Given a complex number z = 2/3 + 5/6i, prove that |z^8| = 41^4 / 1679616 -/
theorem complex_power_magnitude (z : ℂ) : 
  z = (2/3 : ℂ) + (5/6 : ℂ) * Complex.I → Complex.abs (z^8) = (41^4 : ℝ) / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_magnitude_l292_29226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_contains_two_points_of_L_l292_29224

-- Define the set L
def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41*x + 2*y, 59*x + 15*y)}

-- Define a parallelogram centered at the origin
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  center_at_origin : vertices 0 + vertices 2 = (0, 0)
  area : ℝ

-- Theorem statement
theorem parallelogram_contains_two_points_of_L 
  (P : Parallelogram) 
  (h_area : P.area = 1990) : 
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ L ∧ p2 ∈ L ∧ p1 ≠ p2 ∧ 
  (↑p1.1, ↑p1.2) ∈ Set.range P.vertices ∧ 
  (↑p2.1, ↑p2.2) ∈ Set.range P.vertices :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_contains_two_points_of_L_l292_29224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_qst_l292_29275

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if three points form a right angle -/
def isRightAngle (p q r : Point3D) : Prop :=
  (distance p q)^2 + (distance q r)^2 = (distance p r)^2

/-- Check if two line segments are parallel -/
def isParallel (p q r s : Point3D) : Prop :=
  (q.x - p.x) * (s.y - r.y) = (q.y - p.y) * (s.x - r.x) ∧
  (q.x - p.x) * (s.z - r.z) = (q.z - p.z) * (s.x - r.x)

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point3D) : ℝ :=
  let a := distance p q
  let b := distance q r
  let c := distance p r
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_qst (p q r s t : Point3D) :
  distance p q = 3 →
  distance q r = 3 →
  distance r s = 3 →
  distance s t = 3 →
  distance t p = 3 →
  isRightAngle p q r →
  isRightAngle r s t →
  isRightAngle s t p →
  isParallel p q s t →
  triangleArea q s t = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_qst_l292_29275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_progression_l292_29232

theorem coprime_arithmetic_progression :
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i, i < 100 → a i < a (i + 1)) ∧ 
    (∀ i j, i < j → j < 100 → a j - a i = d * (j - i)) ∧
    (∀ i j, i < j → j < 100 → Nat.Coprime (a i) (a j)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_progression_l292_29232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_north_eastland_population_increase_l292_29257

noncomputable def hours_per_day : ℝ := 24
noncomputable def days_per_year : ℝ := 365
noncomputable def hours_per_birth : ℝ := 6
noncomputable def hours_per_death : ℝ := 36

noncomputable def births_per_day : ℝ := hours_per_day / hours_per_birth
noncomputable def deaths_per_day : ℝ := hours_per_day / hours_per_death

noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase : ℝ := net_increase_per_day * days_per_year

theorem north_eastland_population_increase :
  Int.floor (annual_increase + 0.5) = 1200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_north_eastland_population_increase_l292_29257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l292_29216

theorem trig_problem (α β : ℝ) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α - β) = 13/14)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π/2) :
  Real.tan (2*α) = -(8/47) * Real.sqrt 3 ∧ Real.cos β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l292_29216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_5_l292_29228

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem min_sum_at_5 (seq : ArithmeticSequence) :
  seq.a 3 + seq.a 9 > 0 → S seq 9 < 0 →
  ∃ n, (∀ m, S seq n ≤ S seq m) ∧ n = 5 := by
  sorry

#check min_sum_at_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_at_5_l292_29228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l292_29278

def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : IsArithmeticSequence a) 
  (sum_456 : a 4 + a 5 + a 6 = 450) : a 2 + a 8 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l292_29278
