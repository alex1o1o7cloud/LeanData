import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_solution_A_is_84_l847_84751

/-- Represents a saline solution with concentration and weight -/
structure SalineSolution where
  concentration : ℝ
  weight : ℝ

/-- Finds the maximum and minimum amounts of solution A that can be used -/
def find_max_min_solution_A (
  solution_A : SalineSolution)
  (solution_B : SalineSolution)
  (solution_C : SalineSolution)
  (target_concentration : ℝ)
  (target_weight : ℝ) : ℝ × ℝ :=
  sorry

/-- The sum of maximum and minimum amounts of solution A is 84 -/
theorem sum_max_min_solution_A_is_84 :
  let solution_A : SalineSolution := { concentration := 0.05, weight := 60 }
  let solution_B : SalineSolution := { concentration := 0.08, weight := 60 }
  let solution_C : SalineSolution := { concentration := 0.09, weight := 47 }
  let target_concentration : ℝ := 0.07
  let target_weight : ℝ := 100
  let (max_A, min_A) := find_max_min_solution_A solution_A solution_B solution_C target_concentration target_weight
  max_A + min_A = 84 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_solution_A_is_84_l847_84751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_parabola_in_cone_l847_84785

theorem largest_parabola_in_cone (r l : ℝ) (hr : r > 0) (hl : l > 0) :
  ∃ (t : ℝ), t = (l * r * Real.sqrt 3) / 2 ∧ 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ r → 
    (4 * l * x * Real.sqrt (x * (2 * r - x))) / (3 * r) ≤ t := by
  sorry

#check largest_parabola_in_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_parabola_in_cone_l847_84785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l847_84764

def T : ℕ → ℕ
  | 0 => 11
  | n + 1 => 11^(T n)

theorem t_100_mod_7 : T 100 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l847_84764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l847_84749

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ := (a * 2 + b) / 2

-- Theorem statement
theorem odot_calculation : (odot (odot 4 6) 8) = 11 := by
  -- Expand the definition of odot
  unfold odot
  -- Simplify the expression
  simp [mul_add, add_mul, mul_div_cancel]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_calculation_l847_84749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_tangent_l847_84701

theorem contrapositive_tangent (α : Real) : 
  (α = π/4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_tangent_l847_84701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_hexagon_area_is_36_l847_84760

/-- The area of a hexagon after extending each side by 1 unit -/
noncomputable def extended_hexagon_area (original_area : ℝ) : ℝ :=
  original_area + 6 * (1/2 * 2 * 3)

/-- Theorem stating that the area of the extended hexagon is 36 square units -/
theorem extended_hexagon_area_is_36 :
  extended_hexagon_area 18 = 36 := by
  unfold extended_hexagon_area
  norm_num

#check extended_hexagon_area_is_36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_hexagon_area_is_36_l847_84760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_result_equals_target_l847_84777

/-- Round a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ := 
  ⌊(100 * x + 0.5)⌋ / 100

/-- The result of the given arithmetic operations -/
noncomputable def result : ℝ := roundToHundredth ((78.652 + 24.3981) - 0.025)

/-- Theorem stating that the result is equal to 103.03 -/
theorem result_equals_target : result = 103.03 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_result_equals_target_l847_84777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l847_84796

-- Define the sequences
noncomputable def a_n (a b n : ℝ) : ℝ := (a * n^2 + 3) / (b * n^2 - 2 * n + 2)
noncomputable def b_n (a b n : ℝ) : ℝ := b - a * (1/3)^(n-1)

-- State the theorem
theorem find_c (a b c : ℝ) : 
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a_n a b n - 3| < ε) →  -- limit of a_n is 3
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |b_n a b n + 1/4| < ε) →  -- limit of b_n is -1/4
  (2 * b = a + c) →  -- a, b, c form an arithmetic sequence
  c = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l847_84796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l847_84769

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EG to GF -/
  segment_ratio : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- segment_ratio is equal to 2 -/
  segment_ratio_eq_two : segment_ratio = 2

/-- The length of the base of the isosceles triangle -/
noncomputable def base_length (t : IsoscelesTriangle) : ℝ :=
  t.side_length * Real.sqrt 3

/-- Theorem: In an isosceles triangle DEF where DE = DF = 5, with altitude DG from D to EF, 
    and EG = 2(GF), the length of EF is 5√3 -/
theorem isosceles_triangle_base_length 
  (t : IsoscelesTriangle) 
  (h : t.side_length = 5) : 
  base_length t = 5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l847_84769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_runner_stop_time_l847_84792

/-- Represents a runner in the race -/
structure Runner where
  pace : ℚ  -- pace in minutes per mile (using rational numbers)

/-- Represents the race scenario -/
structure Race where
  distance : ℚ  -- total race distance in miles
  runner1 : Runner
  runner2 : Runner
  stop_time : ℚ  -- time when runner2 stops

/-- Calculates the distance covered by a runner in a given time -/
def distance_covered (r : Runner) (t : ℚ) : ℚ :=
  t / r.pace

/-- Calculates the time needed for a runner to cover a given distance -/
def time_to_cover (r : Runner) (d : ℚ) : ℚ :=
  d * r.pace

/-- Theorem: The second runner can remain stopped for 8 minutes before the first runner catches up -/
theorem second_runner_stop_time (race : Race) 
  (h1 : race.distance = 10)
  (h2 : race.runner1.pace = 8)
  (h3 : race.runner2.pace = 7)
  (h4 : race.stop_time = 56) :
  time_to_cover race.runner1 (distance_covered race.runner2 race.stop_time - distance_covered race.runner1 race.stop_time) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_runner_stop_time_l847_84792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l847_84746

theorem converse_of_implication (p q : Prop) : 
  (q → p) = (q → p) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l847_84746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l847_84762

noncomputable def a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin ((ω / 2) * x + φ), 1)
noncomputable def b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos ((ω / 2) * x + φ))

noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a_vec := a ω φ x
  let b_vec := b ω φ x
  ((a_vec.1 + b_vec.1) * (a_vec.1 - b_vec.1) + (a_vec.2 + b_vec.2) * (a_vec.2 - b_vec.2))

theorem vector_function_properties 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi / 4) 
  (h_period : ∀ x, f ω φ (x + 4) = f ω φ x) 
  (h_point : f ω φ 1 = 1 / 2) : 
  ω = Real.pi / 2 ∧ 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f ω φ x ≥ -1) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f ω φ x ≤ 1 / 2) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f ω φ x = -1) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f ω φ x = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l847_84762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_relations_l847_84703

/-- A trihedral angle with vertex S and edges SA, SB, SC -/
structure TrihedralAngle where
  /-- Plane angle opposite to edge SA -/
  α : Real
  /-- Plane angle opposite to edge SB -/
  β : Real
  /-- Plane angle opposite to edge SC -/
  γ : Real
  /-- Dihedral angle at edge SA -/
  A : Real
  /-- Dihedral angle at edge SB -/
  B : Real
  /-- Dihedral angle at edge SC -/
  C : Real

/-- Theorem relating plane angles and dihedral angles in a trihedral angle -/
theorem trihedral_angle_relations (t : TrihedralAngle) :
  (Real.cos t.α = (Real.cos t.A + Real.cos t.B * Real.cos t.C) / (Real.sin t.B * Real.sin t.C)) ∧
  (Real.cos t.β = (Real.cos t.B + Real.cos t.A * Real.cos t.C) / (Real.sin t.A * Real.sin t.C)) ∧
  (Real.cos t.γ = (Real.cos t.C + Real.cos t.A * Real.cos t.B) / (Real.sin t.A * Real.sin t.B)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_relations_l847_84703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_operations_l847_84787

theorem natural_number_operations (a b : ℕ) : 
  (∃ c : ℕ, a + b = c) ∧ (∃ d : ℕ, a * b = d) ∧ (∃ e : ℕ, a ^ b = e) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_operations_l847_84787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l847_84713

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem point_b_coordinates (A B : Point) (h1 : A.x = -4) (h2 : A.y = 3) 
    (h3 : A.x = B.x) (h4 : distance A B = 6) : 
  (B.x = -4 ∧ B.y = -3) ∨ (B.x = -4 ∧ B.y = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l847_84713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l847_84727

theorem system_has_solution (a b : ℝ) :
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ Real.cos x₁ = a * x₁ + b ∧ Real.cos x₂ = a * x₂ + b) →
  ∃ x : ℝ, Real.cos x = a * x + b ∧ Real.sin x + a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l847_84727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_point_coordinates_l847_84793

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the initial point P
noncomputable def point_P : ℝ × ℝ := (1/2, Real.sqrt 3/2)

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := Real.pi/3

-- Define the function to rotate a point counterclockwise
noncomputable def rotate (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x * Real.cos angle - y * Real.sin angle,
   x * Real.sin angle + y * Real.cos angle)

-- Statement to prove
theorem rotated_point_coordinates :
  unit_circle point_P.1 point_P.2 →
  let point_Q := rotate point_P rotation_angle
  point_Q = (-1/2, Real.sqrt 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_point_coordinates_l847_84793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ranking_arrangements_l847_84719

def num_arrangements (n : Nat) (k : Nat) (m : Nat) : Nat :=
  (n - k) * (n - m) * Nat.factorial (n - 2)

theorem ranking_arrangements :
  ∀ (n : Nat), n = 5 →
  ∀ (k m : Nat), k = 2 ∧ m = 1 →
  num_arrangements n k m = 36 := by
  sorry

#eval num_arrangements 5 2 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ranking_arrangements_l847_84719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l847_84759

-- Define the circle C
def circle_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 3 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := θ = Real.pi / 3

-- Define point P as the intersection of circle C and ray OM
def point_P (ρ : ℝ) : Prop := circle_C ρ (Real.pi / 3) ∧ ray_OM (Real.pi / 3)

-- Define point Q as the intersection of line l and ray OM
def point_Q (ρ : ℝ) : Prop := line_l ρ (Real.pi / 3) ∧ ray_OM (Real.pi / 3)

-- Theorem statement
theorem length_PQ : 
  ∀ (ρ_P ρ_Q : ℝ), point_P ρ_P → point_Q ρ_Q → |ρ_P - ρ_Q| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_l847_84759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_one_half_l847_84756

theorem reciprocal_of_one_half : (1 / 2 : ℝ)⁻¹ = 2 := by
  -- Convert the fraction to a real number
  have h : (1 / 2 : ℝ) = (1 : ℝ) / (2 : ℝ) := by norm_num
  -- Rewrite using the definition of reciprocal
  rw [h, inv_div]
  -- Simplify
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_one_half_l847_84756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_perfect_square_product_l847_84742

theorem exist_non_perfect_square_product (d : ℕ) 
  (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) (h4 : d > 0) :
  ∃ a b : ℕ, a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧
  ¬∃ k : ℕ, a * b - 1 = k^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_non_perfect_square_product_l847_84742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l847_84700

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- The directrix of a parabola -/
noncomputable def directrix (c : Parabola) : ℝ := -c.p / 2

/-- A line y = -√3(x-1) -/
noncomputable def line (x : ℝ) : ℝ := -Real.sqrt 3 * (x - 1)

/-- Checks if a point is on the parabola -/
def isOnParabola (c : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * c.p * p.x

/-- Checks if a point is on the line -/
def isOnLine (p : Point) : Prop :=
  p.y = line p.x

/-- Theorem statement -/
theorem parabola_line_intersection (c : Parabola) 
    (hf : isOnLine (focus c))
    (M N : Point) 
    (hM : isOnParabola c M ∧ isOnLine M)
    (hN : isOnParabola c N ∧ isOnLine N) :
  c.p = 2 ∧ 
  ∃ (O : Point), (O.x = (M.x + N.x) / 2 ∧ O.y = (M.y + N.y) / 2) ∧
    Real.sqrt ((O.x - M.x)^2 + (O.y - M.y)^2) = O.x - directrix c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l847_84700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_height_is_three_l847_84747

/-- Calculates the height of barbed wire given the area of a square field, cost per meter of wire,
    gate width, number of gates, and total cost. -/
noncomputable def barbed_wire_height (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (gate_width * (num_gates : ℝ))
  total_cost / (wire_length * cost_per_meter)

/-- Theorem stating that for the given conditions, the height of the barbed wire is 3 meters. -/
theorem barbed_wire_height_is_three :
  barbed_wire_height 3136 1.5 1 2 999 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_height_is_three_l847_84747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l847_84788

theorem trigonometric_equation_solution :
  ∃ (A φ b : ℝ) (ω : ℝ),
    A > 0 ∧
    0 < φ ∧ φ < π ∧
    (∀ x, 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) ∧
    A = Real.sqrt 2 ∧
    φ = π / 4 ∧
    b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l847_84788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alternating_tunnel_construction_l847_84715

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a closed self-intersecting line --/
structure SelfIntersectingLine where
  points : List Point
  is_closed : points.head? = points.head?  -- Changed from last? to head? as a placeholder
  has_single_intersection : ∀ p q : Point, p ∈ points → q ∈ points → p ≠ q → 
    (∃! r : Point, r ∈ points ∧ r ≠ p ∧ r ≠ q ∧ 
      (∃ t : ℝ, 0 < t ∧ t < 1 ∧ r = ⟨p.x * (1 - t) + q.x * t, p.y * (1 - t) + q.y * t⟩))
  no_self_touching : ∀ p q r : Point, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → r ≠ p → 
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ r = ⟨p.x * (1 - t) + q.x * t, p.y * (1 - t) + q.y * t⟩) → 
    (∀ s : Point, s ∈ points → s = p ∨ s = q ∨ s = r)

/-- Represents the construction of a tunnel --/
structure TunnelConstruction where
  line : SelfIntersectingLine
  alternating_path : List Bool  -- True for over, False for under

/-- Theorem: There exists a tunnel construction for any valid self-intersecting line --/
theorem exists_alternating_tunnel_construction (line : SelfIntersectingLine) :
  ∃ (construction : TunnelConstruction), construction.line = line ∧
    construction.alternating_path.length = line.points.length ∧
    (∀ i j : Nat, i < j → j < line.points.length →
      construction.alternating_path[i]? ≠ construction.alternating_path[j]?) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alternating_tunnel_construction_l847_84715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l847_84780

-- Define the ⊗ operation
noncomputable def otimes (x y z : ℝ) : ℝ := x / (y - z)

-- State the theorem
theorem otimes_calculation :
  otimes (otimes 1 3 2) (otimes 2 1 3) (otimes 3 2 1) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l847_84780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_eight_l847_84763

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 * x else 2^x

theorem f_sum_equals_negative_eight :
  f (-3) + f 2 = -8 :=
by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expressions
  simp
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_eight_l847_84763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l847_84761

/-- Definition of the ellipse C -/
def Ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Left focus F₁ -/
def F₁ (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- Right focus F₂ -/
def F₂ (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Point B -/
def B (b : ℝ) : ℝ × ℝ := (0, b)

/-- Point D on negative x-axis -/
def D (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)

/-- Main theorem -/
theorem ellipse_theorem (a b c : ℝ) (h : a > b ∧ b > 0) :
  let C := Ellipse a b h
  let f₁ := F₁ c
  let f₂ := F₂ c
  let b_point := B b
  let d := D (-b^2/c)
  (∀ p ∈ C, (p.1 - f₁.1)^2 + (p.2 - f₁.2)^2 + (p.1 - f₂.1)^2 + (p.2 - f₂.2)^2 = 4*a^2) →
  (b_point.1 - d.1) * (b_point.1 - f₂.1) + (b_point.2 - d.2) * (b_point.2 - f₂.2) = 0 →
  2 * (f₂.1 - f₁.1) + (d.1 - f₂.1) = 0 ∧ 2 * (f₂.2 - f₁.2) + (d.2 - f₂.2) = 0 →
  (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ C → y = k * (x - 1)) →
  (-- 1. Triangle BF₁F₂ is equilateral
   (b_point.1 - f₁.1)^2 + (b_point.2 - f₁.2)^2 = (b_point.1 - f₂.1)^2 + (b_point.2 - f₂.2)^2 ∧
   (b_point.1 - f₁.1)^2 + (b_point.2 - f₁.2)^2 = (f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) ∧
  -- 2. Equation of ellipse C is x²/4 + y²/3 = 1
  (a = 2 ∧ b = Real.sqrt 3) ∧
  -- 3. Fixed point N(4, 0) exists with the described property
  (∃ (N : ℝ × ℝ), N = (4, 0) ∧
   ∀ (P Q : ℝ × ℝ), P ∈ C → Q ∈ C →
   ∃ (k : ℝ), k ≠ 0 ∧ Q.2 - P.2 = k * (Q.1 - P.1) →
   ∃ (t : ℝ), (1 - t) * Q.1 + t * P.1 = N.1 ∧ (1 - t) * Q.2 - t * P.2 = N.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l847_84761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_N_disjoint_l847_84745

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 1)^2 = 0}

def N : Set ℝ := {-3, 1}

theorem M_N_disjoint : M ∩ (N.prod N) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_N_disjoint_l847_84745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l847_84711

theorem line_inclination_angle (α : Real) (h1 : 0 ≤ α) (h2 : α < Real.pi) :
  (Real.tan α = -1) → α = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l847_84711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l847_84797

theorem divisibility_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ α : ℤ, (p : ℤ) ∣ (α * (α - 1) + 3)) ↔ (∃ β : ℤ, (p : ℤ) ∣ (β * (β - 1) + 25)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_equivalence_l847_84797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l847_84789

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := ((2 * x + 5) / 5) ^ (1/3 : ℝ)

-- State the theorem
theorem h_equation_solution :
  ∃ (x : ℝ), h (3 * x) = 3 * h x ∧ x = -65/24 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equation_solution_l847_84789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_from_interest_and_discount_l847_84767

-- Define the simple interest and true discount functions
noncomputable def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

noncomputable def true_discount (P R T : ℝ) : ℝ := 
  let SI := simple_interest P R T
  (SI * P) / (P + SI)

-- State the theorem
theorem sum_from_interest_and_discount (P R T : ℝ) 
  (h1 : simple_interest P R T = 88)
  (h2 : true_discount P R T = 80) :
  P = 880 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_from_interest_and_discount_l847_84767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_f_is_odd_l847_84794

-- Define the function f
noncomputable def f (b : ℝ) : ℝ → ℝ := 
  λ x => if x ≥ 0 then (2:ℝ)^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

-- State the theorem
theorem odd_function_value (b : ℝ) : f b (-1) = -3 := by
  sorry

-- Define the property of being an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State that f is an odd function
theorem f_is_odd (b : ℝ) : is_odd (f b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_f_is_odd_l847_84794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l847_84735

-- Define the curves
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (4 * Real.sin θ * Real.cos θ, 4 * Real.sin θ * Real.sin θ)

noncomputable def C₃ (α ρ : ℝ) : ℝ × ℝ := (ρ * Real.cos α, ρ * Real.sin α)

-- Define the theorem
theorem intersection_angle (α : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi)
  (h2 : ∃ ρ₁ ρ₂ : ℝ, 
    C₃ α ρ₁ ≠ (0, 0) ∧ 
    C₃ α ρ₂ ≠ (0, 0) ∧
    (∃ φ : ℝ, C₃ α ρ₁ = C₁ φ) ∧
    C₃ α ρ₂ = C₂ α ∧
    (ρ₁ - ρ₂) ^ 2 = 32) :
  α = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l847_84735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l847_84772

theorem angle_in_second_quadrant (α : Real) 
  (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ (x y : Real), x < 0 ∧ y > 0 ∧ x = Real.cos α ∧ y = Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l847_84772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_perp_beta_sufficient_not_necessary_l847_84774

-- Define the basic structures
structure Plane : Type
structure Line : Type

-- Define the relationships
def Line.liesIn (l : Line) (p : Plane) : Prop := sorry
def Plane.intersects (p1 p2 : Plane) (l : Line) : Prop := sorry
def Line.perpendicular (l1 l2 : Line) : Prop := sorry
def Plane.perpendicular (p1 p2 : Plane) : Prop := sorry

-- Define the variables
variable (α β : Plane) (m : Line) (a b : Line)

-- Define the conditions
axiom intersect : Plane.intersects α β m
axiom a_in_α : Line.liesIn a α
axiom b_in_β : Line.liesIn b β
axiom b_perp_m : Line.perpendicular b m

-- Define the theorem
theorem alpha_perp_beta_sufficient_not_necessary :
  (∀ a b, Plane.perpendicular α β → Line.perpendicular a b) ∧
  (∃ a b, Line.perpendicular a b ∧ ¬Plane.perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_perp_beta_sufficient_not_necessary_l847_84774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_special_arithmetic_l847_84717

noncomputable def special_arithmetic_sequence (a : ℝ) : ℕ → ℝ
  | 1 => a + 8
  | 4 => a + 2
  | 6 => a - 2
  | n => a + 8 - 2 * (n - 1)

noncomputable def sum_special_arithmetic (a : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * (a + 8) + (n - 1 : ℝ) * -2)

theorem max_sum_special_arithmetic :
  ∃ (a : ℝ), (special_arithmetic_sequence a 1) * (special_arithmetic_sequence a 6) =
    (special_arithmetic_sequence a 4)^2 ∧
  ∃ (n : ℕ), ∀ (m : ℕ), sum_special_arithmetic a m ≤ sum_special_arithmetic a n ∧
  sum_special_arithmetic a n = 90 := by
  sorry

#check max_sum_special_arithmetic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_special_arithmetic_l847_84717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_count_l847_84784

/-- Proves that the total number of pets is 19 given the conditions about their relative numbers --/
theorem pet_count (cats : ℕ) : 
  cats + (cats + 6) + (cats - 1) + 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_count_l847_84784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_f_abs_less_than_half_l847_84741

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2*x - 1
  else if 0 < x ∧ x ≤ 1 then -2*x + 1
  else 0  -- undefined for other x values

-- Theorem for f(f(-1))
theorem f_of_f_neg_one : f (f (-1)) = -1 := by sorry

-- Define the solution set
def solution_set : Set ℝ := {x | -3/4 < x ∧ x < -1/4} ∪ {x | 1/4 < x ∧ x < 3/4}

-- Theorem for the solution set
theorem f_abs_less_than_half (x : ℝ) : 
  x ∈ solution_set ↔ (x ∈ Set.Icc (-1) 1 ∧ |f x| < 1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_neg_one_f_abs_less_than_half_l847_84741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_values_l847_84740

/-- Given a triangle ABC where a + c = 2b and A - C = π/2, 
    prove that sin A = (√7 + 1)/4, sin B = √3/4, and sin C = (√7 - 1)/4. -/
theorem triangle_sine_values (a b c A B C : ℝ) : 
  a + c = 2 * b →
  A - C = π / 2 →
  Real.sin A = (Real.sqrt 7 + 1) / 4 ∧ 
  Real.sin B = Real.sqrt 3 / 4 ∧ 
  Real.sin C = (Real.sqrt 7 - 1) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_values_l847_84740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_problem_l847_84707

theorem log_equation_problem (x : ℝ) (h1 : x > 1) 
  (h2 : (Real.log x)^2 - Real.log (x^2) = 18) : 
  (Real.log x)^4 - Real.log (x^4) = 472 + 76 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_problem_l847_84707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l847_84705

def z : ℂ := 1 - 2 * Complex.I

theorem distance_to_origin : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l847_84705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_squares_l847_84706

/-- A triangle DEF inscribed in a semicircle -/
structure InscribedTriangle where
  R : ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  is_inscribed : (D.1 - E.1)^2 + (D.2 - E.2)^2 = (2*R)^2
  on_semicircle : F.1^2 + F.2^2 = R^2
  not_coincide : F ≠ D ∧ F ≠ E

/-- The sum of distances from F to D and E -/
noncomputable def t (triangle : InscribedTriangle) : ℝ :=
  Real.sqrt ((triangle.D.1 - triangle.F.1)^2 + (triangle.D.2 - triangle.F.2)^2) +
  Real.sqrt ((triangle.E.1 - triangle.F.1)^2 + (triangle.E.2 - triangle.F.2)^2)

theorem inscribed_triangle_sum_squares (triangle : InscribedTriangle) :
  (t triangle)^2 = 8 * triangle.R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_sum_squares_l847_84706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l847_84730

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.a 2 + seq.a 6 = seq.a 8) :
  sum_n seq 5 / seq.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l847_84730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l847_84718

-- Define the sequence a_n
def a : ℕ+ → ℝ := sorry

-- Define the constant m
def m : ℝ := sorry

-- Define the property of the sequence
def sequence_property (a : ℕ+ → ℝ) (m : ℝ) :=
  ∀ n : ℕ+, n ≥ 2 → a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- Part 1
theorem part_one (a : ℕ+ → ℝ) (m : ℝ) (d : ℝ) (hd : d ≠ 0) :
  sequence_property a m →
  (∀ n : ℕ+, a (n + 1) = a n + d) →
  m = 1 := by
  sorry

-- Part 2
theorem part_two (a : ℕ+ → ℝ) (m : ℝ) :
  sequence_property a m →
  a 1 = 1 →
  a 2 = 2 →
  a 3 = 4 →
  (∃ p : ℝ, p ∈ Set.Icc 3 5 ∧ ∀ n : ℕ+, ∃ t : ℝ, t * a n + p ≥ n) →
  ∃ t : ℝ, t = 1 / 32 ∧ ∀ t' : ℝ, (∃ p : ℝ, p ∈ Set.Icc 3 5 ∧ ∀ n : ℕ+, t' * a n + p ≥ n) → t ≤ t' := by
  sorry

-- Part 3
theorem part_three (a : ℕ+ → ℝ) (m : ℝ) :
  sequence_property a m →
  m ≠ 0 →
  (∃ n₁ n₂ : ℕ+, a n₁ ≠ a n₂) →
  (∃ T : ℕ+, ∀ n : ℕ+, a (n + T) = a n) →
  ∃ T : ℕ+, T = 3 ∧ (∀ n : ℕ+, a (n + T) = a n) ∧ ∀ T' : ℕ+, (∀ n : ℕ+, a (n + T') = a n) → T ≤ T' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l847_84718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l847_84737

/-- A configuration of nine squares in a rectangle --/
structure SquareConfiguration where
  b : Fin 9 → ℕ  -- Sizes of the 9 squares
  l : ℕ          -- Length of the rectangle
  w : ℕ          -- Width of the rectangle

/-- The conditions for a valid configuration --/
def isValidConfiguration (c : SquareConfiguration) : Prop :=
  c.b 1 + c.b 2 = c.b 3 ∧
  c.b 1 + c.b 3 = c.b 4 ∧
  c.b 3 + c.b 4 = c.b 6 ∧
  c.b 2 + c.b 3 + c.b 4 = c.b 5 ∧
  c.b 2 + c.b 5 = c.b 7 ∧
  c.b 1 + c.b 6 = c.b 8 ∧
  c.b 6 + c.b 8 = c.b 5 + c.b 7 ∧
  c.b 7 + c.b 8 = c.l ∧
  c.b 6 + c.b 7 = c.w ∧
  Nat.Coprime c.l c.w

theorem rectangle_perimeter (c : SquareConfiguration) 
  (h : isValidConfiguration c) : 
  2 * (c.l + c.w) = 266 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l847_84737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_a_closed_l847_84791

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => a n + 3^(n + 1)

/-- The closed form of the sequence -/
def a_closed (n : ℕ) : ℚ := (3^(n + 1) - 7) / 2

/-- Theorem stating that the recursive and closed forms are equivalent -/
theorem a_eq_a_closed : ∀ n : ℕ, a n = a_closed n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_a_closed_l847_84791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_sum_max_sin_cos_sum_achievable_l847_84755

theorem max_sin_cos_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  Real.sin θ₁ * Real.cos θ₂ + Real.sin θ₂ * Real.cos θ₃ + Real.sin θ₃ * Real.cos θ₄ + 
  Real.sin θ₄ * Real.cos θ₅ + Real.sin θ₅ * Real.cos θ₁ ≤ 5/2 :=
by sorry

theorem max_sin_cos_sum_achievable :
  ∃ θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ,
    Real.sin θ₁ * Real.cos θ₂ + Real.sin θ₂ * Real.cos θ₃ + Real.sin θ₃ * Real.cos θ₄ + 
    Real.sin θ₄ * Real.cos θ₅ + Real.sin θ₅ * Real.cos θ₁ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_cos_sum_max_sin_cos_sum_achievable_l847_84755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l847_84743

theorem prime_product_divisors (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (Nat.divisors (p^6 * q^m)).card = 56 → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_product_divisors_l847_84743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l847_84765

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => -1 / sequenceA n

theorem sequence_periodic : ∀ n : ℕ, sequenceA (n + 2) = sequenceA n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l847_84765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_l847_84783

/-- Given a function f(x) = sin(ωx) + cos(ωx) where ω > 0, 
    if the adjacent axes of symmetry of f(x) are separated by a distance of π/3, 
    then ω = 3 -/
theorem symmetry_implies_omega (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, ∃ y : ℝ, Real.sin (ω * x) + Real.cos (ω * x) = Real.sin (ω * y) + Real.cos (ω * y) ∧ |x - y| = π / 3) : 
  ω = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_omega_l847_84783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l847_84732

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.cos (ω * x) * (Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x))

theorem min_omega (ω : ℝ) (h1 : ω > 0) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f ω x₀ ≤ f ω x ∧ f ω x ≤ f ω (x₀ + 2016 * Real.pi)) →
  ω ≥ 1 / 4032 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l847_84732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_500_l847_84720

/-- The length of a train that passes a pole in 20 seconds and a 500 m tunnel in 40 seconds -/
noncomputable def train_length : ℝ := 500

/-- The time taken for the train to pass a pole -/
noncomputable def pole_time : ℝ := 20

/-- The length of the tunnel -/
noncomputable def tunnel_length : ℝ := 500

/-- The time taken for the train to pass through the tunnel -/
noncomputable def tunnel_time : ℝ := 40

/-- The speed of the train -/
noncomputable def train_speed : ℝ := train_length / pole_time

theorem train_length_is_500 : train_length = 500 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_500_l847_84720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l847_84758

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x^2 - 1)
def domain_f_x2_minus_1 : Set ℝ := Set.Icc 0 3

-- Theorem stating the domain of f(2x - 1)
theorem domain_f_2x_minus_1 (h : ∀ x ∈ domain_f_x2_minus_1, f (x^2 - 1) = f (x^2 - 1)) :
  {x : ℝ | ∃ y ∈ domain_f_x2_minus_1, 2 * x - 1 = y^2 - 1} = Set.Icc 0 (9/2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l847_84758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_coffee_consumption_l847_84753

-- Define the types for sleep hours and coffee gallons as aliases for ℝ
abbrev SleepHours := ℝ
abbrev CoffeeGallons := ℝ

-- Define the inverse proportionality relationship
def inverseProportion (k : ℝ) (h : SleepHours) (g : CoffeeGallons) : Prop :=
  g * h = k

-- Define the conditions
axiom wednesday_sleep : SleepHours
axiom wednesday_coffee : CoffeeGallons
axiom wednesday_relation : inverseProportion 24 wednesday_sleep wednesday_coffee
axiom wednesday_sleep_value : wednesday_sleep = (8 : ℝ)
axiom wednesday_coffee_value : wednesday_coffee = (3 : ℝ)

axiom saturday_sleep : SleepHours
axiom saturday_coffee : CoffeeGallons
axiom saturday_relation : inverseProportion 48 saturday_sleep saturday_coffee
axiom saturday_sleep_value : saturday_sleep = (4 : ℝ)

-- State the theorem
theorem saturday_coffee_consumption :
  saturday_coffee = (12 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saturday_coffee_consumption_l847_84753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l847_84723

noncomputable section

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The equation of the ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The eccentricity of the ellipse -/
def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- A point on the ellipse -/
structure Point (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : ellipse_equation E x y

/-- The left vertex of the ellipse -/
def left_vertex (E : Ellipse) : Point E where
  x := -E.a
  y := 0
  h := by sorry

/-- The slope of a line through two points -/
def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- The main theorem -/
theorem ellipse_properties (E : Ellipse) 
  (h_ecc : eccentricity E = 2/3)
  (C : Point E)
  (h_C : C.x = 2 ∧ C.y = 5/3)
  (B : Point E)
  (h_B : ∃ (k : ℝ), B.x = k * C.x ∧ B.y = k * C.y ∧ k = 1/2) :
  (E.a^2 = 9 ∧ E.b^2 = 5) ∧ 
  line_slope (-E.a, 0) (B.x, B.y) = 5 * Real.sqrt 3 / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l847_84723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_sum_l847_84702

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x : ℝ, x^2 + 18*x + 80 = (x + ↑d) * (x + ↑e)) →
  (∀ x : ℝ, x^2 - 20*x + 96 = (x - ↑e) * (x - ↑f)) →
  d + e + f = 30 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_sum_l847_84702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_digit_not_five_l847_84776

def Digit := Fin 10

structure Envelope :=
  (digit : Digit)

def Statement (e : Envelope) : Fin 5 → Prop
  | 0 => e.digit = ⟨1, by norm_num⟩
  | 1 => e.digit ≠ ⟨2, by norm_num⟩
  | 2 => e.digit ≠ ⟨1, by norm_num⟩
  | 3 => e.digit = ⟨5, by norm_num⟩
  | 4 => e.digit ≠ ⟨3, by norm_num⟩

theorem envelope_digit_not_five (e : Envelope) 
  (h : ∃ (i : Fin 5), ¬Statement e i ∧ (∀ (j : Fin 5), j ≠ i → Statement e j)) :
  e.digit ≠ ⟨5, by norm_num⟩ := by
  sorry

#check envelope_digit_not_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_digit_not_five_l847_84776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_imply_a_range_l847_84754

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 - 2 * x + 1)

-- State the theorem
theorem domain_and_inequality_imply_a_range (a : ℝ) : 
  (∀ x, ∃ y, f a x = y) →  -- Domain of f is ℝ
  (∀ x ∈ Set.Icc (1/2) 2, x + 1/x > a) →  -- Inequality holds for x in [1/2, 2]
  1 < a ∧ a < 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_inequality_imply_a_range_l847_84754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l847_84716

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : f (-2) = 3) 
  (h2 : ∀ x : ℝ, (deriv (deriv f)) x < 3) :
  ∀ x : ℝ, f x < 3 * x + 9 ↔ x > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l847_84716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_day_of_historical_figure_l847_84712

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 400 == 0) || (year % 4 == 0 && year % 100 ≠ 0)

/-- Counts leap years between two years, inclusive -/
def countLeapYears (startYear endYear : ℕ) : ℕ :=
  (List.range (endYear - startYear + 1)).filter (fun i => isLeapYear (startYear + i)) |>.length

/-- Calculates the day of the week given a start day and number of days passed -/
def calculateDayOfWeek (startDay : DayOfWeek) (daysPassed : ℕ) : DayOfWeek :=
  sorry

theorem birth_day_of_historical_figure :
  let startYear : ℕ := 1772
  let endYear : ℕ := 2022
  let yearSpan : ℕ := endYear - startYear
  let leapYears : ℕ := countLeapYears startYear endYear
  let regularYears : ℕ := yearSpan - leapYears
  let totalDayShift : ℕ := regularYears + 2 * leapYears
  calculateDayOfWeek DayOfWeek.Monday (7 - (totalDayShift % 7)) = DayOfWeek.Saturday :=
by sorry

#eval countLeapYears 1772 2022

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_day_of_historical_figure_l847_84712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l847_84714

/-- The ellipse with equation x^2/25 + y^2/9 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Point A -/
def A : ℝ × ℝ := (4, 0)

/-- Point B -/
def B : ℝ × ℝ := (2, 2)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum value of |MA| + |MB| for any point M on the ellipse -/
theorem max_distance_sum :
  ∃ (max : ℝ), max = 10 + 2 * Real.sqrt 10 ∧
  ∀ (M : ℝ × ℝ), ellipse M.1 M.2 →
    distance M A + distance M B ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l847_84714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannon_probabilities_l847_84779

structure Cannon where
  short_prob : ℝ
  hit_prob : ℝ
  long_prob : ℝ

def cannon1 : Cannon := ⟨0.3, 0.2, 0.5⟩
def cannon2 : Cannon := ⟨0.6, 0.1, 0.3⟩

noncomputable def prob_at_least_one_pair_hit (n : ℕ) : ℝ := sorry

noncomputable def prob_at_least_one_hit_each_pair (n : ℕ) : ℝ := sorry

noncomputable def prob_at_most_k_short_shots (n k : ℕ) : ℝ := sorry

noncomputable def min_shots_for_simple_hit (p : ℝ) : ℕ := sorry

theorem cannon_probabilities :
  (abs (prob_at_least_one_pair_hit 20 - 0.333) < 0.001) ∧
  (abs (prob_at_least_one_hit_each_pair 3 - 0.022) < 0.001) ∧
  (abs (prob_at_most_k_short_shots 2 3 - 0.968) < 0.001) ∧
  (min_shots_for_simple_hit 0.9 = 8) := by
  sorry

#check cannon_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannon_probabilities_l847_84779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l847_84766

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem segment_length : distance (4, 6) (11, 22) = Real.sqrt 305 := by
  -- Expand the definition of distance
  unfold distance
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l847_84766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l847_84781

/-- A subset of integers from 1 to 150 where no member is 4 times another -/
def ValidSubset (S : Set ℕ) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, 1 ≤ x ∧ x ≤ 150 ∧ 1 ≤ y ∧ y ≤ 150 ∧ (x ≠ 4 * y ∧ y ≠ 4 * x)

/-- The maximum cardinality of a valid subset is 146 -/
theorem max_valid_subset_size :
  ∃ S : Finset ℕ, ValidSubset (↑S) ∧ S.card = 146 ∧
  ∀ T : Finset ℕ, ValidSubset (↑T) → T.card ≤ 146 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_subset_size_l847_84781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l847_84786

noncomputable def f (x : ℝ) : ℝ := 9 / (1 + Real.cos (2 * x)) + 25 / (1 - Real.cos (2 * x))

theorem f_minimum_value :
  ∀ x : ℝ, f x ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l847_84786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_travel_time_is_eight_hours_l847_84725

-- Define the given constants
def distance : ℚ := 360
def scheduledTime : ℚ := 9
def speedIncrease : ℚ := 5

-- Define the actual travel time function
def actualTravelTime (d : ℚ) (t : ℚ) (s : ℚ) : ℚ :=
  d / ((d / t) + s)

-- Theorem statement
theorem actual_travel_time_is_eight_hours :
  actualTravelTime distance scheduledTime speedIncrease = 8 := by
  -- Unfold the definition of actualTravelTime
  unfold actualTravelTime
  -- Perform the calculation
  simp [distance, scheduledTime, speedIncrease]
  -- The proof is complete
  rfl

#eval actualTravelTime distance scheduledTime speedIncrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_travel_time_is_eight_hours_l847_84725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_brokerage_percentage_l847_84798

/-- Calculates the brokerage percentage given the cash realized and total amount before brokerage -/
noncomputable def brokerage_percentage (cash_realized : ℝ) (total_amount : ℝ) : ℝ :=
  (total_amount - cash_realized) / total_amount * 100

/-- Theorem stating that the brokerage percentage for the given stock sale is approximately 0.2358% -/
theorem stock_sale_brokerage_percentage :
  let cash_realized : ℝ := 106.25
  let total_amount : ℝ := 106
  abs (brokerage_percentage cash_realized total_amount - 0.2358) < 0.0001 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_sale_brokerage_percentage_l847_84798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_18_pounds_l847_84708

/-- The cost of bananas with a quantity-based discount -/
noncomputable def banana_cost (base_price : ℝ) (base_quantity : ℝ) (discount_rate : ℝ) (discount_threshold : ℝ) (purchase_quantity : ℝ) : ℝ :=
  let regular_price := (purchase_quantity / base_quantity) * base_price
  let discount := if purchase_quantity > discount_threshold then discount_rate * regular_price else 0
  regular_price - discount

/-- Theorem: The cost of 18 pounds of bananas is $16.20 -/
theorem banana_cost_18_pounds :
  banana_cost 3 3 0.1 10 18 = 16.20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_18_pounds_l847_84708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_right_angles_implies_rectangle_l847_84748

/-- A quadrilateral with coordinates in ℝ² -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- A right angle is π/2 radians -/
def isRightAngle (θ : ℝ) : Prop := θ = Real.pi / 2

/-- A rectangle is a quadrilateral with four right angles -/
def isRectangle (q : Quadrilateral) : Prop :=
  isRightAngle (angle q.A q.B q.C) ∧
  isRightAngle (angle q.B q.C q.D) ∧
  isRightAngle (angle q.C q.D q.A) ∧
  isRightAngle (angle q.D q.A q.B)

/-- If a quadrilateral has three right angles, it is a rectangle -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) :
  isRightAngle (angle q.A q.B q.C) →
  isRightAngle (angle q.B q.C q.D) →
  isRightAngle (angle q.C q.D q.A) →
  isRectangle q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_right_angles_implies_rectangle_l847_84748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l847_84739

def b : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 4/5
  | n+3 => (b (n+1) * b (n+2)) / (3 * b (n+1) - 2 * b (n+2))

theorem b_2023_value : b 2023 = 2 / 10109 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l847_84739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l847_84757

-- Define the circles and points
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def point_A : ℝ × ℝ := (2, 2)
def point_B : ℝ × ℝ := (1, 1)

-- Define the constant lambda
noncomputable def lambda : ℝ := Real.sqrt 2

-- Define the function for the ratio of distances
noncomputable def distance_ratio (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) /
  Real.sqrt ((P.1 - point_B.1)^2 + (P.2 - point_B.2)^2)

-- Define the line l
def line_l (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ k, p = (2 + k, t + k)}

-- Define the midpoint condition
def is_midpoint (M N E : ℝ × ℝ) : Prop :=
  M.1 = (N.1 + E.1) / 2 ∧ M.2 = (N.2 + E.2) / 2

-- State the theorem
theorem circle_intersection_theorem :
  (∀ P ∈ circle_O, distance_ratio P = lambda) ∧
  (∀ t ∈ Set.Icc (-(Real.sqrt 5)) (Real.sqrt 5),
    ∃ M N, M ∈ circle_C ∧ N ∈ circle_C ∧ M ∈ line_l t ∧ N ∈ line_l t ∧
    is_midpoint M N (2, t)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l847_84757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_c_properties_l847_84704

/-- Ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eccentricity : (a^2 - b^2) / a^2 = 1/2
  h_point : 1/a^2 + (Real.sqrt 2/2)^2/b^2 = 1

/-- The equation of ellipse C -/
def ellipse_equation (e : EllipseC) : Prop :=
  ∀ x y : ℝ, x^2/2 + y^2 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- Fixed points on x-axis -/
def fixed_points : Prop :=
  ∃ (M₁ M₂ : ℝ × ℝ), M₁ = (1, 0) ∧ M₂ = (-1, 0) ∧
    ∀ (k m : ℝ), (∃! x : ℝ, (2*k^2 + 1)*x^2 + 4*k*m*x + 2*m^2 - 2 = 0) →
      (|((M₁.1 * k + m) * (M₂.1 * k + m))| / (k^2 + 1) = 1)

/-- Main theorem combining both parts of the problem -/
theorem ellipse_c_properties (e : EllipseC) :
  ellipse_equation e ∧ fixed_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_c_properties_l847_84704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l847_84773

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_bound : x ≥ -1/2)
  (y_bound : y ≥ -2)
  (z_bound : z ≥ -3) :
  ∃ (m : ℝ), m = 3 * Real.sqrt 15 ∧
  Real.sqrt (6*x + 3) + Real.sqrt (6*y + 12) + Real.sqrt (6*z + 18) ≤ m ∧
  ∃ (x' y' z' : ℝ), 
    x' + y' + z' = 2 ∧
    x' ≥ -1/2 ∧ y' ≥ -2 ∧ z' ≥ -3 ∧
    Real.sqrt (6*x' + 3) + Real.sqrt (6*y' + 12) + Real.sqrt (6*z' + 18) = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l847_84773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_run_distance_l847_84722

noncomputable def total_distance_flash (k d a : ℝ) : ℝ :=
  k * d / (k - 1) + a

theorem flash_run_distance (k d a v : ℝ) (hk : k > 1) :
  let ace_speed := v
  let flash_speed := k * v
  let head_start := d
  let additional_distance := a
  total_distance_flash k d a = flash_speed * (head_start / (flash_speed - ace_speed)) + additional_distance := by
  simp [total_distance_flash]
  field_simp
  ring
  sorry

#check flash_run_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_run_distance_l847_84722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march14_is_tuesday_l847_84799

-- Define a type for days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving BEq, Repr

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to get the day of the week after a given number of days
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem march14_is_tuesday (feb14 : DayOfWeek) 
  (h : feb14 = DayOfWeek.Tuesday) : 
  dayAfter feb14 28 = DayOfWeek.Tuesday := by
  rw [h]
  -- The proof goes here
  sorry

#eval dayAfter DayOfWeek.Tuesday 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march14_is_tuesday_l847_84799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l847_84734

def sequence_a : ℕ → ℤ
  | 0 => -1
  | n + 1 => 2 * sequence_a n + 3

def sequence_b : ℕ → ℤ := λ n => n + 1

def S : ℕ → ℤ := λ n => n * (2 * sequence_b 1 + (n - 1) * (sequence_b 2 - sequence_b 1)) / 2

def sequence_c (n : ℕ) : ℤ := (sequence_a n + 3 * sequence_b n) * ((-1 : ℤ) ^ n)

def T : ℕ → ℤ := λ n => (List.range n).map sequence_c |>.sum

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n = 2^n - 3) ∧
  (sequence_b 1 = 2 ∧ 2 * sequence_b 3 + S 5 = 28) ∧
  (∀ n : ℕ, sequence_b n = n + 1) ∧
  ¬ (∃ m : ℕ, m > 0 ∧ T m = 2023) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l847_84734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l847_84733

theorem simplify_trig_expression (θ : Real) 
  (h1 : Real.sin θ < 0) 
  (h2 : Real.tan θ > 0) : 
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l847_84733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_counterexample_l847_84744

theorem cosine_equation_counterexample : 
  ∃ x : ℝ, (Real.cos x + Real.cos (2*x) + Real.cos (4*x) = 0) ∧ 
           (Real.cos (2*x) + Real.cos (4*x) + Real.cos (8*x) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_counterexample_l847_84744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmys_pizza_cost_per_slice_l847_84775

/-- Represents the cost and number of slices for a pizza size -/
structure PizzaSize where
  cost : ℚ
  slices : ℕ

/-- Represents the cost of different topping categories -/
structure ToppingCosts where
  categoryA : ℚ
  categoryB : ℚ
  categoryC : ℚ

/-- Calculates the cost per slice for a pizza with given toppings -/
def costPerSlice (size : PizzaSize) (toppings : ToppingCosts) 
  (numA numB numC : ℕ) : ℚ :=
  (size.cost + numA * toppings.categoryA + numB * toppings.categoryB + numC * toppings.categoryC) / size.slices

/-- Theorem stating that the cost per slice for Jimmy's pizza is $2.10 -/
theorem jimmys_pizza_cost_per_slice :
  let mediumPizza : PizzaSize := ⟨12, 10⟩
  let toppingCosts : ToppingCosts := ⟨2, 1, (1/2)⟩
  costPerSlice mediumPizza toppingCosts 2 3 4 = (21/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmys_pizza_cost_per_slice_l847_84775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l847_84728

-- Define the complex number ω
def ω : ℂ := Complex.mk 5 3

-- State the theorem
theorem complex_absolute_value : 
  Complex.abs (ω^2 + 8*ω + 72) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l847_84728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_exists_l847_84771

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The left focus of the ellipse -/
noncomputable def leftFocus (e : Ellipse) : Point :=
  ⟨-Real.sqrt (e.a^2 - e.b^2), 0⟩

/-- Theorem: There exists a unique ellipse satisfying the given conditions -/
theorem unique_ellipse_exists :
  ∃! (e : Ellipse),
    ∃ (p a b : Point),
      isOnEllipse p e ∧
      isOnEllipse a e ∧
      isOnEllipse b e ∧
      a.y = 0 ∧
      a.x > 0 ∧
      b.x = 0 ∧
      b.y > 0 ∧
      (b.y - p.y) * a.x = (a.x - p.x) * b.y ∧
      p.x = (leftFocus e).x ∧
      distance (leftFocus e) a = Real.sqrt 10 + Real.sqrt 5 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_exists_l847_84771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_walking_speed_l847_84778

/-- Represents the scenario of Kolya's journey to the store -/
structure JourneyScenario where
  total_distance : ℝ
  initial_speed : ℝ
  doubled_speed : ℝ
  store_closing_time : ℝ

/-- Calculates Kolya's walking speed given the journey scenario -/
noncomputable def calculate_walking_speed (scenario : JourneyScenario) : ℝ :=
  20 / 3

/-- Theorem stating that Kolya's walking speed is 20/3 km/h given the conditions -/
theorem kolya_walking_speed (scenario : JourneyScenario) 
  (h1 : scenario.initial_speed = 10)
  (h2 : scenario.doubled_speed = 2 * scenario.initial_speed)
  (h3 : scenario.store_closing_time = scenario.total_distance / scenario.initial_speed)
  (h4 : scenario.total_distance > 0) :
  calculate_walking_speed scenario = 20 / 3 := by
  sorry

#check kolya_walking_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_walking_speed_l847_84778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_polar_axis_form_l847_84726

/-- A line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- The polar axis -/
def polarAxis : PolarLine :=
  { equation := λ ρ θ ↦ θ = 0 ∨ θ = Real.pi }

/-- A line is parallel to another line if they never intersect -/
def parallel (l₁ l₂ : PolarLine) : Prop :=
  ∀ ρ θ, ¬(l₁.equation ρ θ ∧ l₂.equation ρ θ)

/-- The theorem stating the form of a line parallel to the polar axis -/
theorem parallel_to_polar_axis_form (l : PolarLine) 
  (h : parallel l polarAxis) : 
  ∃ c : ℝ, l.equation = λ ρ θ ↦ ρ * Real.sin θ = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_polar_axis_form_l847_84726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_selections_l847_84736

/-- Represents the number of pairs of socks -/
def num_pairs : ℕ := 20

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of patterns -/
def num_patterns : ℕ := 4

/-- Represents a valid sock selection -/
structure SockSelection where
  color1 : Fin num_colors
  color2 : Fin num_colors
  pattern1 : Fin num_patterns
  pattern2 : Fin num_patterns
  different_colors : color1 ≠ color2
  different_patterns : pattern1 ≠ pattern2

/-- Instance of Fintype for SockSelection -/
instance : Fintype SockSelection :=
  sorry

/-- The theorem stating the number of valid sock selections -/
theorem num_valid_selections : Fintype.card SockSelection = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_selections_l847_84736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_number_of_committees_committee_structure_consistency_l847_84724

/-- Represents the structure of a committee --/
inductive CommitteeStructure
  | FiveSenators
  | FourSenatorsAndFourAides
  | TwoSenatorsAndTwelveAides

/-- The number of senators --/
def numSenators : ℕ := 100

/-- The number of aides per senator --/
def aidesPerSenator : ℕ := 4

/-- The number of committees each senator serves on --/
def committeesPerSenator : ℕ := 5

/-- The number of committees each aide serves on --/
def committeesPerAide : ℕ := 3

/-- The total number of committees --/
def totalCommittees : ℕ := 160

/-- Theorem stating that the total number of committees is 160 --/
theorem correct_number_of_committees :
  totalCommittees = (numSenators * committeesPerSenator + 
    numSenators * aidesPerSenator * committeesPerAide) / 5 := by
  sorry

/-- All committee structures contribute 5 points --/
theorem committee_structure_consistency (s : CommitteeStructure) :
  (match s with
  | CommitteeStructure.FiveSenators => 5
  | CommitteeStructure.FourSenatorsAndFourAides => 4 + 4 / 4
  | CommitteeStructure.TwoSenatorsAndTwelveAides => 2 + 12 / 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_number_of_committees_committee_structure_consistency_l847_84724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_f_eq_sum_d_l847_84790

/-- The function f(n) as defined in the problem statement. -/
def f (n : ℕ+) : ℤ :=
  if n = 1 then 45
  else if n = 2 then 2025
  else 0

/-- The theorem stating the values of f(n) for all positive integers n. -/
theorem f_values (n : ℕ+) : 
  (n = 1 → f n = 45) ∧ 
  (n = 2 → f n = 2025) ∧ 
  (n ≥ 3 → f n = 0) := by
  sorry

/-- Helper function to construct the matrix from a number k with n^2 digits. -/
def construct_matrix (k : ℕ) (n : ℕ+) : Matrix (Fin n) (Fin n) ℤ :=
  sorry

/-- The determinant function d(k) as defined in the problem statement. -/
def d (k : ℕ) (n : ℕ+) : ℤ :=
  Matrix.det (construct_matrix k n)

/-- The sum of d(k) over all valid k for a given n. -/
noncomputable def sum_d (n : ℕ+) : ℤ :=
  sorry

/-- Theorem stating that f(n) equals the sum of d(k) over all valid k. -/
theorem f_eq_sum_d (n : ℕ+) : f n = sum_d n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_f_eq_sum_d_l847_84790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trout_weight_l847_84770

/-- Represents the weight of trout caught by James in pounds -/
def trout : ℝ := sorry

/-- The total weight of fish caught by James in pounds -/
def total_weight : ℝ := 1100

/-- The relation between the weights of different fish types -/
axiom fish_relations :
  total_weight = trout + (1.5 * trout) + (2 * trout)

/-- The theorem stating the approximate weight of trout caught -/
theorem trout_weight : ⌊trout⌋ = 244 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trout_weight_l847_84770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_increase_l847_84729

/-- The cost of a krumblik -/
noncomputable def C_c : ℝ → ℝ := λ x => x

/-- The cost of a kryamblik -/
noncomputable def C_k (c : ℝ) : ℝ := 0.4 * c

/-- The value of coupons received from buying a kryamblik -/
noncomputable def coupon_value (c : ℝ) : ℝ := 0.5 * C_k c

/-- The amount paid for krumblik using coupons -/
noncomputable def coupon_payment (c : ℝ) : ℝ := 0.2 * c

/-- The total expense for buying both krumblik and kryamblik -/
noncomputable def total_expense (c : ℝ) : ℝ := C_k c + (c - coupon_payment c)

/-- The percentage increase in expenses -/
noncomputable def percentage_increase (c : ℝ) : ℝ := (total_expense c - c) / c * 100

theorem expense_increase (c : ℝ) (h : c > 0) : percentage_increase c = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_increase_l847_84729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_on_purchasing_power_l847_84731

theorem price_increase_effect_on_purchasing_power 
  (price_old : ℝ) 
  (price_new : ℝ) 
  (salary : ℝ) 
  (quantity_old : ℝ) 
  (quantity_new : ℝ) 
  (h1 : price_new = price_old * (1 + 0.25)) 
  (h2 : quantity_old = salary / price_old) 
  (h3 : quantity_new = salary / price_new) : 
  quantity_new / quantity_old = 0.8 := by
  sorry

#check price_increase_effect_on_purchasing_power

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_effect_on_purchasing_power_l847_84731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_theorem_l847_84738

/-- Represents a right quadrilateral prism -/
structure RightQuadrilateralPrism where
  base_side : ℝ
  height : ℝ

/-- The angle between non-intersecting diagonals of adjacent lateral faces -/
noncomputable def diagonal_angle (prism : RightQuadrilateralPrism) (α : ℝ) : ℝ :=
  2 * Real.arctan (Real.cos α)

/-- 
Theorem: In a right quadrilateral prism, the angle between non-intersecting diagonals 
of two adjacent lateral faces is 2 arctan(cos α), where α is the angle between 
the plane containing the diagonals and the base plane.
-/
theorem diagonal_angle_theorem (prism : RightQuadrilateralPrism) (α : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi / 2) : 
  diagonal_angle prism α = 2 * Real.arctan (Real.cos α) := by
  sorry

#check diagonal_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_angle_theorem_l847_84738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_a_range_l847_84710

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4 - a) * ((x^2 - 2*x - 2) * Real.exp x - a * x^3 + 12 * a * x)

-- State the theorem
theorem local_max_implies_a_range :
  ∀ a : ℝ, (∃ ε > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < ε → f a x ≤ f a 2) ∧ f a 2 = 3
  → 4 < a ∧ a < (1/3) * Real.exp 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_a_range_l847_84710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l847_84750

/-- The probability of selecting both A and B when randomly choosing 3 students from a group of 5 students (including A and B) -/
theorem probability_select_A_and_B (n k : ℕ) : 
  n = 5 → k = 3 → (Nat.choose n k : ℚ)⁻¹ * (Nat.choose (n - 2) (k - 2) : ℚ) = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_select_A_and_B_l847_84750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l847_84795

def translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (w : ℂ) :
  translation w (1 + 3*Complex.I) = 4 + 7*Complex.I →
  translation w (2 - Complex.I) = 5 + 3*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l847_84795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_l847_84752

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the cosine of half angle A
noncomputable def cosHalfA (t : Triangle) : ℝ := 
  Real.sqrt ((t.b + t.c) / (2 * t.c))

-- State the theorem
theorem right_angled_triangle (t : Triangle) 
  (h : cosHalfA t ^ 2 = (t.b + t.c) / (2 * t.c)) : 
  t.a^2 = t.b^2 + t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangle_l847_84752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l847_84709

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - (1/3)

-- Statement to prove
theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l847_84709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l847_84768

-- Define the vectors
def a : ℝ × ℝ := sorry
def b : ℝ × ℝ := (3, 4)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define perpendicular vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_properties :
  (magnitude a = 10) →
  (magnitude b = 5) ∧
  (parallel a b → (a = (6, 8) ∨ a = (-6, -8))) ∧
  (perpendicular a b → magnitude (a.1 - b.1, a.2 - b.2) = 5 * Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l847_84768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delegates_seating_probability_sum_of_m_and_n_l847_84721

-- Define the number of delegates and countries
def total_delegates : ℕ := 12
def num_countries : ℕ := 3
def delegates_per_country : ℕ := 4

-- Define the probability as a rational number
def probability : ℚ := 36 / 385

-- State the theorem
theorem delegates_seating_probability :
  (((total_delegates.factorial / (delegates_per_country.factorial ^ num_countries) : ℚ) - 7188) /
   (total_delegates.factorial / (delegates_per_country.factorial ^ num_countries) : ℚ)) = probability := by
  sorry

-- Prove that m + n = 421
theorem sum_of_m_and_n : 36 + 385 = 421 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_delegates_seating_probability_sum_of_m_and_n_l847_84721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_24_divisible_by_4_l847_84782

theorem count_divisors_of_24_divisible_by_4 :
  let b_set := {b : ℕ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}
  Finset.card (Finset.filter (λ b => b > 0 ∧ 4 ∣ b ∧ b ∣ 24) (Finset.range 25)) = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_24_divisible_by_4_l847_84782
