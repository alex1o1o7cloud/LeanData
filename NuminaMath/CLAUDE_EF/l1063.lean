import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sphere_with_one_rational_point_l1063_106340

/-- A point in 3D space is rational if all its coordinates are rational numbers. -/
def RationalPoint (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y z : ℚ), p = (↑x, ↑y, ↑z)

/-- A sphere in 3D space -/
def Sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2.1 - center.2.1)^2 + (p.2.2 - center.2.2)^2 = radius^2}

/-- Theorem: There exists a sphere with exactly one rational point -/
theorem exists_sphere_with_one_rational_point :
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ),
    ∃! p, p ∈ Sphere center radius ∧ RationalPoint p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sphere_with_one_rational_point_l1063_106340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_polar_curve_l1063_106337

theorem max_y_coordinate_polar_curve :
  ∃ (max_y : ℝ), 
    (∀ θ : ℝ, (Real.cos (2 * θ) * Real.sin θ) ≤ max_y) ∧ 
    (∃ θ₀ : ℝ, Real.cos (2 * θ₀) * Real.sin θ₀ = max_y) ∧ 
    max_y = 2/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_polar_curve_l1063_106337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_existence_l1063_106333

theorem subset_sum_existence (M m n : ℕ) (A : Finset ℕ) :
  0 < m → 0 < n → m ≤ n →
  0 < M → M ≤ m * (m + 1) / 2 →
  A ⊆ Finset.range n →
  A.card = m →
  ∃ B : Finset ℕ, B ⊆ A ∧ 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_existence_l1063_106333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1063_106334

/-- Parabola structure -/
structure Parabola where
  eq : ℝ → ℝ → Prop

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: Distance between points on parabola with equal distance to focus -/
theorem parabola_distance_theorem (C : Parabola) (F A B : Point) : 
  C.eq A.x A.y →  -- A lies on the parabola
  B.x = 3 ∧ B.y = 0 →  -- B is at (3,0)
  C.eq = fun x y => y^2 = 4*x →  -- Parabola equation is y^2 = 4x
  F.x = 1 ∧ F.y = 0 →  -- F is the focus of the parabola
  distance A F = distance B F →  -- |AF| = |BF|
  distance A B = 2 * Real.sqrt 2 :=  -- |AB| = 2√2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1063_106334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106309

-- Define the function f on [-1,1]
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 4^x) - (1 / 2^x)
  else 2^x - 4^x

-- State the theorem
theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x) ∧  -- f is odd on [-1,1]
  (∀ x ∈ Set.Icc 0 1, f x = 2^x - 4^x) ∧       -- f(x) = 2^x - 4^x for x in [0,1]
  (∀ x ∈ Set.Icc 0 1, f x ≤ 0) ∧               -- Maximum value is 0
  (∃ x ∈ Set.Icc 0 1, f x = 0) ∧               -- Maximum value is attained
  (∀ x ∈ Set.Icc 0 1, f x ≥ -2) ∧              -- Minimum value is -2
  (∃ x ∈ Set.Icc 0 1, f x = -2) :=              -- Minimum value is attained
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1063_106325

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |1 - 1/x|

-- Part I
theorem part_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : a * b > 1 := by
  sorry

-- Part II
theorem part_two (x₀ : ℝ) (h : 0 < x₀ ∧ x₀ < 1) : 
  let P : ℝ × ℝ := (x₀, f x₀)
  let tangent_line := λ (x : ℝ) => f x₀ - (1/x₀^2) * (x - x₀)
  let x_intercept := x₀ * (2 - x₀)
  let y_intercept := (2 - x₀) / x₀
  (1/2) * x_intercept * y_intercept = (1/2) * (2 - x₀)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1063_106325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_partition_bound_l1063_106367

/-- Definition of an A-partition --/
def is_A_partition (A : Finset ℕ) (n : ℕ) (P : List ℕ) : Prop :=
  (∀ x ∈ P, x ∈ A) ∧ P.sum = n

/-- Definition of the number of different parts in a partition --/
def num_different_parts (P : List ℕ) : ℕ :=
  (P.toFinset).card

/-- Definition of an optimal A-partition --/
def is_optimal_A_partition (A : Finset ℕ) (n : ℕ) (P : List ℕ) : Prop :=
  is_A_partition A n P ∧
  ∀ Q : List ℕ, is_A_partition A n Q → Q.length ≥ P.length

/-- Main theorem statement --/
theorem optimal_partition_bound (n : ℕ) (A : Finset ℕ) (P : List ℕ) :
  n > 0 →
  A ⊆ Finset.range (n + 1) →
  is_optimal_A_partition A n P →
  (num_different_parts P : ℝ) ≤ Real.sqrt (Real.sqrt (6 * n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_partition_bound_l1063_106367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_integers_l1063_106358

theorem min_distinct_integers (a : Fin 2006 → ℕ+) 
  (h : ∀ i j, i < 2005 → j < 2005 → i ≠ j → (a i : ℚ) / (a (i+1)) ≠ (a j : ℚ) / (a (j+1))) : 
  (Finset.univ.filter (λ i : Fin 2006 => ∃ j, a i = a j)).card ≥ 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_integers_l1063_106358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_a_value_l1063_106397

/-- A triangle with sides a, a+1, and a+2, where the smallest angle is half of the largest angle -/
structure SpecialTriangle where
  a : ℝ
  side1 : ℝ := a
  side2 : ℝ := a + 1
  side3 : ℝ := a + 2
  angle_smallest : ℝ
  angle_largest : ℝ
  angle_middle : ℝ
  angle_sum : angle_smallest + angle_middle + angle_largest = Real.pi
  angle_relation : angle_smallest = angle_largest / 2
  side_angle_relation : Real.sin angle_smallest / side1 = Real.sin angle_largest / side3

/-- The value of 'a' in the SpecialTriangle is 4 -/
theorem special_triangle_a_value (t : SpecialTriangle) : t.a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_a_value_l1063_106397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1063_106396

/-- Curve C₁ -/
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- Curve C₂ -/
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 1}

/-- The square of the distance between two points -/
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The minimum distance between curves C₁ and C₂ -/
theorem min_distance_C₁_C₂ : 
  ∃ (d : ℝ), d = Real.sqrt (7/4) ∧ 
  ∀ (p q : ℝ × ℝ), p ∈ C₁ → q ∈ C₂ → d^2 ≤ dist_sq p q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₁_C₂_l1063_106396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_five_l1063_106370

theorem opposite_of_five : 
  (∀ x : ℤ, x + (-x) = 0) → -5 = -5 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_five_l1063_106370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_units_l1063_106316

/-- Calculates the number of units in a building given the occupancy rate, monthly rent, and annual income. -/
def calculate_units (occupancy_rate : ℚ) (monthly_rent : ℕ) (annual_income : ℕ) : ℕ :=
  (annual_income * 4 / (occupancy_rate.num * monthly_rent * 12)).toNat

/-- Theorem: Given the specified conditions, the building has 100 units. -/
theorem building_units :
  let occupancy_rate : ℚ := 3/4
  let monthly_rent : ℕ := 400
  let annual_income : ℕ := 360000
  calculate_units occupancy_rate monthly_rent annual_income = 100 := by
  sorry

#eval calculate_units (3/4) 400 360000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_units_l1063_106316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1063_106390

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_properties (a₁ d : ℝ) :
  (S a₁ d 5) * (S a₁ d 6) + 15 = 0 ∧ S a₁ d 5 ≠ 5 →
  S a₁ d 6 = -3 ∧ a₁ = 7 ∧ (d ≤ -2 * Real.sqrt 2 ∨ d ≥ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1063_106390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compressor_stations_configuration_l1063_106342

-- Define the structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the distance function between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Theorem statement
theorem compressor_stations_configuration 
  (A B C : Point) 
  (not_collinear : ¬(∃ (t : ℝ), B = ⟨A.x + t * (C.x - A.x), A.y + t * (C.y - A.y)⟩))
  (condition1 : distance A B + distance B C = 4 * distance A C)
  (condition2 : distance A C + distance C B = distance A B + a)
  (condition3 : distance A C + distance C B = 85)
  : 60.71 < a ∧ a < 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compressor_stations_configuration_l1063_106342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1063_106384

/-- The tangent function f(x) = tan(ωx + π/4) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x + Real.pi / 4)

/-- Theorem stating the maximum value of ω given the conditions -/
theorem max_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : StrictMonoOn (f ω) (Set.Ioo (-Real.pi/3) (Real.pi/2))) :
  ω ≤ 1/2 ∧ ∃ (ω_max : ℝ), ω_max = 1/2 ∧ 
  StrictMonoOn (f ω_max) (Set.Ioo (-Real.pi/3) (Real.pi/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1063_106384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_stone_loss_l1063_106344

/-- Calculates the loss when a precious stone breaks -/
noncomputable def stone_break_loss (total_weight : ℝ) (total_value : ℝ) (ratio_small : ℝ) (ratio_large : ℝ) : ℝ :=
  let price_per_gram_squared := total_value / (total_weight ^ 2)
  let small_weight := (ratio_small / (ratio_small + ratio_large)) * total_weight
  let large_weight := (ratio_large / (ratio_small + ratio_large)) * total_weight
  let small_value := price_per_gram_squared * (small_weight ^ 2)
  let large_value := price_per_gram_squared * (large_weight ^ 2)
  total_value - (small_value + large_value)

/-- Theorem stating the loss incurred when a specific stone breaks -/
theorem specific_stone_loss :
  stone_break_loss 35 12250 2 5 = 5000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_stone_loss_l1063_106344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_8_three_even_one_odd_l1063_106346

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_three_even_one_odd (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (λ d => d % 2 = 0)).length = 3 ∧ (digits.filter (λ d => d % 2 = 1)).length = 1

theorem smallest_four_digit_divisible_by_8_three_even_one_odd :
  ∀ n : ℕ, is_four_digit n → n % 8 = 0 → has_three_even_one_odd n → n ≥ 2016 :=
by sorry

#check smallest_four_digit_divisible_by_8_three_even_one_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_8_three_even_one_odd_l1063_106346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_on_ruled_paper_l1063_106389

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  /-- The number of sides is at least 3 -/
  sides_ge_three : n ≥ 3

/-- A sheet of paper with evenly spaced parallel lines -/
structure RuledPaper where

/-- Predicate indicating whether a vertex is on a line of the ruled paper -/
def vertex_on_line (r : RuledPaper) (vertex : ℝ × ℝ) : Prop :=
  sorry -- We'll leave this undefined for now

/-- Predicate indicating whether a regular polygon can be placed on ruled paper
    with all vertices on the lines -/
def can_place_on_ruled_paper (n : ℕ) : Prop :=
  ∃ (p : RegularPolygon n) (r : RuledPaper),
    ∀ vertex, vertex_on_line r vertex

/-- Theorem stating that a regular polygon can be placed on ruled paper
    if and only if it has 3, 4, or 6 sides -/
theorem regular_polygon_on_ruled_paper (n : ℕ) :
  can_place_on_ruled_paper n ↔ n = 3 ∨ n = 4 ∨ n = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_on_ruled_paper_l1063_106389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sum_calculation_l1063_106345

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Problem statement --/
theorem original_sum_calculation (rate : ℝ) (time : ℕ) (interest : ℝ) 
  (h_rate : rate = 0.11)
  (h_time : time = 3)
  (h_interest : interest = 14705.24)
  (h_compound : ∀ p, compound_interest p rate time = interest) :
  ∃ p : ℝ, (p ≥ 39999 ∧ p ≤ 40001) ∧ compound_interest p rate time = interest :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_sum_calculation_l1063_106345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_theorem_l1063_106330

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem setup
def TangentProblem (circle : Circle) (A B C : Point) (l : Line) : Prop :=
  -- A and B are points on the circle
  ((A.x - circle.center.1)^2 + (A.y - circle.center.2)^2 = circle.radius^2) ∧
  ((B.x - circle.center.1)^2 + (B.y - circle.center.2)^2 = circle.radius^2) ∧
  -- C is the intersection of tangents through A and B
  ((C.x - A.x)*(B.y - A.y) = (C.y - A.y)*(B.x - A.x)) ∧
  -- l is tangent to the circle
  ((l.a * circle.center.1 + l.b * circle.center.2 + l.c)^2 = 
    (l.a^2 + l.b^2) * circle.radius^2) ∧
  -- l does not pass through A or B
  (l.a * A.x + l.b * A.y + l.c ≠ 0) ∧
  (l.a * B.x + l.b * B.y + l.c ≠ 0)

-- Define the distances
noncomputable def u (A : Point) (l : Line) : ℝ := 
  |l.a * A.x + l.b * A.y + l.c| / Real.sqrt (l.a^2 + l.b^2)

noncomputable def v (B : Point) (l : Line) : ℝ := 
  |l.a * B.x + l.b * B.y + l.c| / Real.sqrt (l.a^2 + l.b^2)

noncomputable def w (C : Point) (l : Line) : ℝ := 
  |l.a * C.x + l.b * C.y + l.c| / Real.sqrt (l.a^2 + l.b^2)

-- Define the angle ACB
noncomputable def angle_ACB (A B C : Point) : ℝ :=
  Real.arccos ((A.x - C.x)*(B.x - C.x) + (A.y - C.y)*(B.y - C.y)) / 
    (Real.sqrt ((A.x - C.x)^2 + (A.y - C.y)^2) * Real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2))

-- State the theorem
theorem tangent_ratio_theorem 
  (circle : Circle) (A B C : Point) (l : Line) 
  (h : TangentProblem circle A B C l) :
  (u A l * v B l) / (w C l)^2 = Real.sin (angle_ACB A B C / 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_theorem_l1063_106330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_50_factorial_l1063_106369

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The set of prime divisors of n -/
def primeDivisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d => Nat.Prime d ∧ d ∣ n) (Finset.range (n + 1))

/-- The number of prime divisors of 50! is 15 -/
theorem prime_divisors_50_factorial :
  (primeDivisors (factorial 50)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_50_factorial_l1063_106369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hamiltonian_cycle_exists_after_edge_removal_l1063_106348

/-- Represents a hypercube graph -/
def HypercubeGraph (n : ℕ) := Fin (2^n)

/-- Represents an edge in the hypercube graph -/
def Edge (n : ℕ) := Fin (2^n) × Fin (2^n)

/-- Represents a Hamiltonian cycle in a graph -/
def HamiltonianCycle (G : Type) := List G

/-- The number of cities in Country K -/
def num_cities : ℕ := 1024

/-- The dimension of the hypercube graph representing Country K -/
def hypercube_dim : ℕ := 10

/-- The number of roads closed in Country K -/
def num_closed_roads : ℕ := 8

theorem hamiltonian_cycle_exists_after_edge_removal 
  (G : HypercubeGraph hypercube_dim) 
  (removed_edges : Finset (Edge hypercube_dim)) 
  (h1 : num_cities = 2^hypercube_dim) 
  (h2 : Finset.card removed_edges = num_closed_roads) :
  ∃ (cycle : HamiltonianCycle (HypercubeGraph hypercube_dim)), True :=
by
  sorry

#check hamiltonian_cycle_exists_after_edge_removal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hamiltonian_cycle_exists_after_edge_removal_l1063_106348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_1_problem_statement_2_l1063_106319

-- Define a as a real number
variable (a : ℝ)

-- Define the complex number in the problem
noncomputable def complex_number (a : ℝ) : ℂ := (a^2 - 1) + (a + 1) * Complex.I

-- Define z as given in the problem
noncomputable def z (a : ℝ) : ℂ := (a + Complex.I * Real.sqrt 3) / (a * Complex.I)

-- Statement 1: If the complex number is pure imaginary, then a = 1
theorem problem_statement_1 : 
  (complex_number a).re = 0 → a = 1 := by sorry

-- Statement 2: If a = 1, then |z| = 2
theorem problem_statement_2 : 
  a = 1 → Complex.abs (z a) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_1_problem_statement_2_l1063_106319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_another_unfoldable_polyhedron_l1063_106301

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Add necessary fields and properties to define a convex polyhedron
  mk :: -- This allows the structure to be defined without specifying fields for now

/-- Represents the property of a polyhedron being unfoldable to a triangle without internal cuts. -/
def unfoldableToTriangle (p : ConvexPolyhedron) : Prop :=
  -- Define the property of being unfoldable to a triangle without internal cuts
  True -- Placeholder, replace with actual definition when available

/-- A triangular pyramid with opposite edges pairwise equal. -/
def TriangularPyramidWithEqualOppositeEdges : ConvexPolyhedron :=
  -- Define a triangular pyramid with opposite edges pairwise equal
  ConvexPolyhedron.mk

/-- Theorem stating the existence of another convex polyhedron that can be unfolded to a triangle. -/
theorem exists_another_unfoldable_polyhedron :
  ∃ (p : ConvexPolyhedron), p ≠ TriangularPyramidWithEqualOppositeEdges ∧ unfoldableToTriangle p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_another_unfoldable_polyhedron_l1063_106301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_condition_l1063_106339

theorem max_sin_a_given_condition :
  (∀ a b : ℝ, Real.sin (a - b) = Real.sin a - Real.sin b) →
  (∃ a : ℝ, Real.sin a = 1 ∧ ∀ a' : ℝ, Real.sin a' ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_condition_l1063_106339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_problem_l1063_106347

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow
deriving DecidableEq

/-- Represents the initial state of the bag -/
def initial_bag : Multiset BallColor :=
  Multiset.replicate 1 BallColor.Red + Multiset.replicate 3 BallColor.Yellow

/-- Calculates the probability of drawing a red ball from the initial bag -/
noncomputable def prob_red_initial : ℚ :=
  (Multiset.count BallColor.Red initial_bag) / (Multiset.card initial_bag)

/-- Calculates the probability of drawing two balls of the same color after adding a ball -/
noncomputable def prob_same_color (add_color : BallColor) : ℚ :=
  let new_bag := initial_bag + Multiset.replicate 1 add_color
  let total_draws := (Multiset.card new_bag) * (Multiset.card new_bag - 1)
  let same_color_draws :=
    (Multiset.count BallColor.Red new_bag) * (Multiset.count BallColor.Red new_bag - 1) +
    (Multiset.count BallColor.Yellow new_bag) * (Multiset.count BallColor.Yellow new_bag - 1)
  same_color_draws / total_draws

theorem lottery_problem :
  (prob_red_initial = 1/4) ∧
  (prob_same_color BallColor.Yellow > prob_same_color BallColor.Red) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_problem_l1063_106347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_max_min_l1063_106321

theorem sin_cos_max_min (α β : ℝ) (h : Real.sin α + Real.sin β = 1/3) :
  let y := Real.sin β - (Real.cos α)^2
  ∃ (y_max y_min : ℝ), (∀ y', y ≤ y_max ∧ y_min ≤ y') ∧ y_max = 4/9 ∧ y_min = -11/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_max_min_l1063_106321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1063_106303

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the domain
def D : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ D, f x = y} = Set.Icc 1 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1063_106303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_l1063_106354

-- Define the polynomial f
variable (f : ℝ → ℝ)

-- Define the condition that f must satisfy
axiom h : ∀ x : ℝ, f (x^2 + 1) - f (x^2 - 1) = 4*x^2 + 6

-- State the theorem to be proved
theorem polynomial_difference (x : ℝ) : f (x^2 + 1) - f (x^2) = 2*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_l1063_106354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1063_106317

open Real

-- Define the set of angles whose terminal sides fall on the y-axis
def y_axis_angles : Set ℝ := {θ | ∃ n : ℤ, θ = n * π + π / 2}

-- Define the function for which we're finding the center of symmetry
noncomputable def f (x : ℝ) : ℝ := 2 * cos (x - π / 4)

-- Define what it means for a point to be a center of symmetry
def is_center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

-- Define the first quadrant
def first_quadrant : Set ℝ := {x | 0 < x ∧ x < π / 2}

theorem problem_statement :
  (y_axis_angles = {θ | ∃ n : ℤ, θ = n * π + π / 2}) ∧
  (is_center_of_symmetry f (3 * π / 4, 0)) ∧
  (∃ α β : ℝ, α ∈ first_quadrant ∧ β ∈ first_quadrant ∧ α < β ∧ tan α > tan β) ∧
  (∀ x, sin (2 * x - π / 3) = sin (2 * (x - π / 6))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1063_106317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_length_l1063_106366

/-- Represents a square divided into two trapezoids and a triangle -/
structure DividedSquare where
  side_length : ℝ
  trapezoid_longer_side : ℝ
  trapezoid_shorter_side : ℝ
  side_length_positive : side_length > 0
  trapezoid_longer_side_positive : trapezoid_longer_side > 0
  trapezoid_shorter_side_positive : trapezoid_shorter_side > 0
  trapezoid_shorter_side_is_half : trapezoid_shorter_side = side_length / 2
  shapes_equal_area : trapezoid_area = triangle_area

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (d : DividedSquare) : ℝ :=
  (d.trapezoid_longer_side + d.trapezoid_shorter_side) * (d.side_length / 2) / 2

/-- Calculates the area of the triangle -/
noncomputable def triangle_area (d : DividedSquare) : ℝ :=
  d.side_length * d.side_length / 3

/-- Theorem stating that for a square with side length 2, the longer side of each trapezoid is 5/3 -/
theorem trapezoid_longer_side_length (d : DividedSquare) 
  (h : d.side_length = 2) : d.trapezoid_longer_side = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_length_l1063_106366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1063_106315

/-- The constant term in the expansion of (2x - 1/x^2)^6 is 240 -/
theorem constant_term_expansion : ∃ (c : ℤ), c = 240 ∧ 
  c = (Finset.range 7).sum (λ r ↦ 
    (Nat.choose 6 r) * (2^(6-r)) * ((-1:ℤ)^r) * 
    (if 6 - 3*r = 0 then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1063_106315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1063_106356

noncomputable def f (x : ℝ) := 2 * Real.sin (x / 3 + Real.pi / 6)

noncomputable def g (x : ℝ) := 2 * Real.sin (4 * x / 3 - 5 * Real.pi / 18)

theorem function_properties :
  let A := 2
  let ω := 1 / 3
  let φ := Real.pi / 6
  (A > 0) ∧
  (ω > 0) ∧
  (abs φ < Real.pi / 2) ∧
  (f Real.pi = 2) ∧
  (f (4 * Real.pi) = -2) ∧
  (∀ x, g x = f (4 * (x - Real.pi / 3))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1063_106356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106329

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x) + 1 / 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) ∧
  (∀ (k : ℤ), f (Real.pi / 12 + k * Real.pi / 2) = 1 / 2) ∧
  (¬ ∀ (x y : ℝ), Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 5 * Real.pi / 12 → f x < f y) ∧
  (∀ (x : ℝ), f x = g (x - Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l1063_106335

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) (result : ℕ) : 
  initial = 80 → percentage = 150 / 100 → result = initial + (initial * percentage).floor → result = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increase_by_percentage_l1063_106335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_condition2_condition3_condition4_l1063_106362

/-- The number of ways to select 5 subject representatives from a group of 5 boys and 3 girls -/
def selectRepresentatives (totalBoys totalGirls numRepresentatives : ℕ) (ways : ℕ) : Prop :=
  totalBoys = 5 ∧ totalGirls = 3 ∧ numRepresentatives = 5 ∧ ways > 0

/-- Condition 1: A specific girl must be the Chinese representative -/
theorem condition1 (totalBoys totalGirls numRepresentatives : ℕ) :
  selectRepresentatives totalBoys totalGirls numRepresentatives 840 := by
  sorry

/-- Condition 2: There are girls in the selection, but fewer than boys -/
theorem condition2 (totalBoys totalGirls numRepresentatives : ℕ) :
  selectRepresentatives totalBoys totalGirls numRepresentatives 5400 := by
  sorry

/-- Condition 3: A specific boy must be included, but he cannot be the Math representative -/
theorem condition3 (totalBoys totalGirls numRepresentatives : ℕ) :
  selectRepresentatives totalBoys totalGirls numRepresentatives 3360 := by
  sorry

/-- Condition 4: A specific girl must be the Chinese representative, and a specific boy must be a subject representative, but not for Math -/
theorem condition4 (totalBoys totalGirls numRepresentatives : ℕ) :
  selectRepresentatives totalBoys totalGirls numRepresentatives 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition1_condition2_condition3_condition4_l1063_106362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1063_106326

def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1063_106326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_at_most_two_is_at_least_three_main_statement_l1063_106374

/-- Represents the number of solutions to some unspecified problem. -/
structure NumSolutions where
  value : ℕ

/-- Predicate representing "at most two solutions" -/
def AtMostTwo (n : NumSolutions) : Prop := n.value ≤ 2

/-- Predicate representing "at least three solutions" -/
def AtLeastThree (n : NumSolutions) : Prop := n.value ≥ 3

/-- Theorem stating that the negation of "at most two solutions" is equivalent to "at least three solutions" -/
theorem negation_of_at_most_two_is_at_least_three :
  ∀ n : NumSolutions, ¬(AtMostTwo n) ↔ AtLeastThree n :=
by
  intro n
  apply Iff.intro
  · intro h
    exact Nat.not_le.mp h
  · intro h
    exact Nat.not_le.mpr h

/-- Proof of the main statement -/
theorem main_statement : ∀ n : NumSolutions, ¬(AtMostTwo n) ↔ AtLeastThree n :=
negation_of_at_most_two_is_at_least_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_at_most_two_is_at_least_three_main_statement_l1063_106374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l1063_106382

/-- The swimming speed of a person in still water -/
noncomputable def swimming_speed : ℝ → ℝ → ℝ → ℝ 
  := λ water_speed distance time => water_speed + distance / time

/-- Theorem: Given the conditions, prove that the swimming speed in still water is 4 km/h -/
theorem swimming_speed_calculation 
  (water_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 12) 
  (h3 : time = 6) : 
  swimming_speed water_speed distance time = 4 := by
  sorry

#check swimming_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_speed_calculation_l1063_106382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1063_106300

def A : Fin 3 → ℝ := ![(-1), 0, 0]
def B : Fin 3 → ℝ := ![1, 2, -2]
def C : Fin 3 → ℝ := ![0, 0, -2]
def D : Fin 3 → ℝ := ![2, 2, -4]
def O : Fin 3 → ℝ := ![0, 0, 0]

def vec (p q : Fin 3 → ℝ) : Fin 3 → ℝ := fun i => q i - p i

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

noncomputable def distance_point_to_line (p q r : Fin 3 → ℝ) : ℝ :=
  let v := vec q r
  let w := vec q p
  magnitude (![v 1 * w 2 - v 2 * w 1, v 2 * w 0 - v 0 * w 2, v 0 * w 1 - v 1 * w 0]) / magnitude v

theorem problem_solution :
  (dot_product (vec A C) (vec A B) = 6) ∧
  (∃ (a b c d : ℝ), a * A 0 + b * B 0 + c * C 0 + d * D 0 = 0 ∧
                     a * A 1 + b * B 1 + c * C 1 + d * D 1 = 0 ∧
                     a * A 2 + b * B 2 + c * C 2 + d * D 2 = 0 ∧
                     a + b + c + d = 0) ∧
  (dot_product (vec A C) (vec A B) / (magnitude (vec A C) * magnitude (vec A B)) = Real.sqrt 15 / 5) ∧
  (distance_point_to_line O A B = Real.sqrt 6 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1063_106300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_divisible_by_five_l1063_106387

def p (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_divisible_by_five (n : ℕ) : 
  5 ∣ p n ↔ n % 4 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_divisible_by_five_l1063_106387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_pentagon_angle_sum_l1063_106377

/-- A cyclic pentagon is a pentagon inscribed in a circle. -/
structure CyclicPentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- The angle CEB in a cyclic pentagon -/
noncomputable def angle_CEB (p : CyclicPentagon) : ℝ := sorry

/-- The sum of angles CDE and EAB in a cyclic pentagon -/
noncomputable def sum_angles_CDE_EAB (p : CyclicPentagon) : ℝ := sorry

/-- Theorem: In a cyclic pentagon ABCDE, if ∠CEB = 17°, then ∠CDE + ∠EAB = 163° -/
theorem cyclic_pentagon_angle_sum (p : CyclicPentagon) :
  angle_CEB p = 17 → sum_angles_CDE_EAB p = 163 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_pentagon_angle_sum_l1063_106377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_implies_a_leq_one_l1063_106381

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - 1 + Real.log x

/-- The theorem statement -/
theorem exists_solution_implies_a_leq_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ ≤ 0) → a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_solution_implies_a_leq_one_l1063_106381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_selection_probability_l1063_106353

theorem seed_selection_probability : 
  let total_seeds : ℕ := 10
  let selected_seeds : ℕ := 3
  let specific_seeds : ℕ := 3
  let probability : ℚ := 17 / 24
  (1 - (Nat.choose (total_seeds - specific_seeds) selected_seeds : ℚ) / 
   (Nat.choose total_seeds selected_seeds : ℚ)) = probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_selection_probability_l1063_106353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_table_profit_l1063_106379

/-- Calculates the profit from selling tables made from chopped trees -/
theorem tree_table_profit
  (num_trees : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ)
  (h1 : num_trees = 30)
  (h2 : planks_per_tree = 25)
  (h3 : planks_per_table = 15)
  (h4 : price_per_table = 300)
  (h5 : labor_cost = 3000)
  : (num_trees * planks_per_tree / planks_per_table) * price_per_table - labor_cost = 12000 := by
  sorry

-- Remove the #eval line as it's causing issues
-- #eval tree_table_profit 30 25 15 300 3000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_table_profit_l1063_106379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_equals_9a_div_2_l1063_106355

/-- The sum of perimeters of a sequence of triangles -/
noncomputable def perimeter_sum (a : ℝ) : ℝ :=
  let initial_perimeter := 3 * a
  let ratio := 1 / 3
  initial_perimeter / (1 - ratio)

/-- The theorem stating the sum of perimeters of the triangle sequence -/
theorem perimeter_sum_equals_9a_div_2 (a : ℝ) (h : a > 0) :
  perimeter_sum a = 9 * a / 2 := by
  sorry

#check perimeter_sum_equals_9a_div_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_equals_9a_div_2_l1063_106355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1063_106376

-- Use the built-in Complex number type
open Complex

-- Theorem statement
theorem complex_fraction_simplification :
  (7 + 10 * I) / (3 - 4 * I) = -19/25 + 58/25 * I := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1063_106376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l1063_106341

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, 1 + Real.sin θ)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the intersection points in polar coordinates
def intersection_points : Set (ℝ × ℝ) := {(1, 0), (1, Real.pi / 2)}

-- Theorem statement
theorem intersection_of_C₁_and_C₂ :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, C₁ θ = p) ∧ (∃ φ : ℝ, C₂ φ = p) ↔ p ∈ intersection_points := by
  sorry

-- Helper lemmas (if needed)
lemma C₁_cartesian_equation (x y : ℝ) :
  (∃ θ : ℝ, C₁ θ = (x, y)) ↔ (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

lemma C₂_cartesian_equation (x y : ℝ) :
  (∃ θ : ℝ, C₂ θ = (x, y)) ↔ x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l1063_106341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_decrease_l1063_106318

/-- Calculates the net change in weekly income given a raise and changes in deductions -/
def netIncomeChange (
  hourlyRaise : ℚ)
  (weeklyHours : ℚ)
  (monthlyHousingBenefitReduction : ℚ)
  (oldFederalTaxRate : ℚ)
  (newFederalTaxRate : ℚ)
  (stateTaxRate : ℚ)
  (socialSecurityRate : ℚ)
  (medicareRate : ℚ)
  (oldContributionRate : ℚ)
  (newContributionRate : ℚ) : ℚ :=
  sorry

/-- The net change in weekly income is negative $2.33 -/
theorem net_income_decrease :
  netIncomeChange 0.5 40 60 0.15 0.20 0.05 0.062 0.0145 0.03 0.04 = -2.33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_income_decrease_l1063_106318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_l1063_106359

/-- A function f: ℝ → ℝ satisfying the given recurrence relation -/
noncomputable def f : ℝ → ℝ := sorry

/-- The recurrence relation for f -/
axiom f_rec (x : ℝ) : f (x + 1) = 1/2 + Real.sqrt (f x - (f x)^2)

/-- The initial condition for f -/
axiom f_init : f (-1) = 1/2

/-- The theorem stating that f(2015) = 1/2 -/
theorem f_2015 : f 2015 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_l1063_106359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_and_parallelepiped_l1063_106305

/-- Represents a tetrahedron with mutually perpendicular edges from vertex S -/
structure Tetrahedron where
  a : ℝ  -- Length of edge SA
  b : ℝ  -- Length of edge SB
  c : ℝ  -- Length of edge SC
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The side length of the largest cube with vertex S inside the tetrahedron -/
noncomputable def largest_cube_side (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)

/-- The dimensions of the largest rectangular parallelepiped with vertex S inside the tetrahedron -/
noncomputable def largest_parallelepiped_dims (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  (t.a / 3, t.b / 3, t.c / 3)

theorem largest_cube_and_parallelepiped (t : Tetrahedron) :
  (largest_cube_side t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)) ∧
  (largest_parallelepiped_dims t = (t.a / 3, t.b / 3, t.c / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_and_parallelepiped_l1063_106305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_than_six_solutions_l1063_106364

theorem more_than_six_solutions :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 > 0 ∧ p.2 > 0) ∧
    (∀ (p : ℕ × ℕ), p ∈ S → (6 : ℚ) / p.1 + (3 : ℚ) / p.2 = 1) ∧
    S.card > 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_than_six_solutions_l1063_106364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l1063_106350

/-- The area of an equilateral triangle with side length 1 -/
noncomputable def equilateral_triangle_area : ℝ := Real.sqrt 3 / 4

/-- The ratio of the area of the orthographic projection to the area of the original triangle -/
noncomputable def projection_ratio : ℝ := Real.sqrt 2 / 4

/-- The theorem stating the area of the original triangle -/
theorem original_triangle_area (projection : ℝ) (h1 : projection = equilateral_triangle_area)
  (h2 : projection / projection_ratio = Real.sqrt 6 / 2) :
  projection / projection_ratio = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l1063_106350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_interval_l1063_106357

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def sideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the angle bisector
def angleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the angle bisector condition
  True

-- Theorem statement
theorem triangle_side_length_interval (t : Triangle) (D : ℝ × ℝ) :
  sideLength t.A t.B = 16 →
  angleBisector t D →
  sideLength t.C D = 4 →
  ∃ m n : ℝ, m < n ∧ 
    (∀ x : ℝ, m < x ∧ x < n ↔ ∃ t' : Triangle, t'.A = t.A ∧ t'.B = t.B ∧ sideLength t'.A t'.C = x) ∧
    m + n = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_interval_l1063_106357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_commerce_students_percentage_l1063_106361

theorem local_commerce_students_percentage 
  (total_arts : ℕ) 
  (total_science : ℕ) 
  (total_commerce : ℕ) 
  (local_arts_percentage : ℚ) 
  (local_science_percentage : ℚ) 
  (total_local_students : ℕ) 
  (h1 : total_arts = 400)
  (h2 : total_science = 100)
  (h3 : total_commerce = 120)
  (h4 : local_arts_percentage = 1/2)
  (h5 : local_science_percentage = 1/4)
  (h6 : total_local_students = 327)
  : ∃ local_commerce_percentage : ℚ, local_commerce_percentage = 85/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_commerce_students_percentage_l1063_106361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_broadly_convex_broadly_concave_inequalities_l1063_106368

variable {α : Type*} [AddCommMonoid α] [SMul ℝ α]
variable (g f g' : α → ℝ)
variable (q₁ q₂ : ℝ) (x₁ x₂ : α)

/-- A function is concave if and only if it satisfies the given inequality -/
def IsConcave (g : α → ℝ) : Prop :=
  ∀ (q₁ q₂ : ℝ) (x₁ x₂ : α), q₁ + q₂ = 1 → 
    g (q₁ • x₁ + q₂ • x₂) > q₁ * g x₁ + q₂ * g x₂

/-- A function is broadly convex if and only if it satisfies the given inequality -/
def IsBroadlyConvex (f : α → ℝ) : Prop :=
  ∀ (q₁ q₂ : ℝ) (x₁ x₂ : α), q₁ + q₂ = 1 → 
    f (q₁ • x₁ + q₂ • x₂) ≤ q₁ * f x₁ + q₂ * f x₂

/-- A function is broadly concave if and only if it satisfies the given inequality -/
def IsBroadlyConcave (g' : α → ℝ) : Prop :=
  ∀ (q₁ q₂ : ℝ) (x₁ x₂ : α), q₁ + q₂ = 1 → 
    g' (q₁ • x₁ + q₂ • x₂) ≥ q₁ * g' x₁ + q₂ * g' x₂

theorem concave_broadly_convex_broadly_concave_inequalities 
  (hq : q₁ + q₂ = 1) : 
  IsConcave g ∧ IsBroadlyConvex f ∧ IsBroadlyConcave g' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concave_broadly_convex_broadly_concave_inequalities_l1063_106368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1063_106394

noncomputable def projection (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  (dot_product / norm_squared * w.1, dot_product / norm_squared * w.2)

theorem projection_problem :
  let v₁ : ℝ × ℝ := (6, 4)
  let w₁ : ℝ × ℝ := (72/13, 48/13)
  let v₂ : ℝ × ℝ := (-3, 1)
  let w₂ : ℝ × ℝ := (-21/13, -14/13)
  projection v₁ (3, 2) = w₁ →
  projection v₂ (3, 2) = w₂ :=
by
  sorry

#check projection_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1063_106394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1063_106349

theorem min_abs_difference (x y : ℤ) (h : x * y - 4 * x + 3 * y = 204) :
  (∀ a b : ℤ, a * b - 4 * a + 3 * b = 204 →
    |x - y| ≤ |a - b|) ∧ |x - y| = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1063_106349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shifts_l1063_106324

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2) + 2

theorem cosine_shifts :
  (∃ (phase_shift : ℝ), phase_shift = Real.pi / 4 ∧
    ∀ (x : ℝ), f x = Real.cos (2 * (x + phase_shift))) ∧
  (∃ (vertical_shift : ℝ), vertical_shift = 2 ∧
    ∀ (x : ℝ), f x = Real.cos (2 * x + Real.pi / 2) + vertical_shift) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shifts_l1063_106324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_l1063_106323

theorem ladder_slide (ladder_length initial_distance slip_distance : ℝ) :
  ladder_length = 30 →
  initial_distance = 6 →
  slip_distance = 6 →
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height := initial_height - slip_distance
  let new_distance := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 18 := by
  sorry

#check ladder_slide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_l1063_106323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_12_divisors_l1063_106365

def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_with_12_divisors :
  ∃ (n : ℕ), n > 0 ∧ count_divisors n = 12 ∧ ∀ (m : ℕ), 0 < m → m < n → count_divisors m ≠ 12 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_12_divisors_l1063_106365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_negative_l1063_106386

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_symmetry (x : ℝ) : f (-x) = -f (x + 4)
axiom f_increasing {x y : ℝ} : 2 < x → x < y → f x < f y

-- Define the properties for x1 and x2
def x1_lt_2_lt_x2 (x1 x2 : ℝ) : Prop := x1 < 2 ∧ 2 < x2
def x1_plus_x2_lt_4 (x1 x2 : ℝ) : Prop := x1 + x2 < 4

-- State the theorem
theorem f_sum_negative (x1 x2 : ℝ) :
  x1_lt_2_lt_x2 x1 x2 → x1_plus_x2_lt_4 x1 x2 → f x1 + f x2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_negative_l1063_106386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_extrema_l1063_106310

open Real

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - k*x + log x

-- Define the derivative of f
noncomputable def f_deriv (k : ℝ) (x : ℝ) : ℝ := x - k + 1/x

theorem monotonicity_and_extrema (k : ℝ) :
  (∀ x, x > 0 → f_deriv k x > 0) ∨
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ f_deriv k x₁ = 0 ∧ f_deriv k x₂ = 0 ∧ 
    |f k x₁ - f k x₂| < k^2/2 - 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_extrema_l1063_106310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1063_106328

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.rpow 0.8 0.7
noncomputable def b : ℝ := Real.rpow 0.8 0.9
noncomputable def c : ℝ := Real.rpow 1.2 0.8

-- State the theorem
theorem ascending_order : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l1063_106328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1063_106393

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x - 3) / (x^2 + 1)

-- Define the solution set
def S : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x < 0} = S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1063_106393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_theorem_l1063_106388

theorem binomial_coefficient_theorem (a : ℝ) :
  (∃ c : ℝ, c = 9/4 ∧ 
   c = (Nat.choose 9 8) * a * (-1)^8 * (1/2)^4) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_theorem_l1063_106388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_width_calculation_l1063_106363

/-- Calculates the width of a tank given its dimensions and plastering cost. -/
theorem tank_width_calculation (length depth : ℝ) (cost rate : ℚ) : 
  length = 25 →
  depth = 6 →
  cost = 409.20 →
  rate = 55 / 100 →
  ∃ width : ℝ, width = 12 ∧ 
    (length * width + 2 * length * depth + 2 * width * depth) * (rate : ℝ) = cost := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_width_calculation_l1063_106363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106308

noncomputable def f (x : ℝ) := Real.sin (2 * x) + Real.cos (2 * x)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧  -- Smallest positive period is π
  (∀ x, f (x - π/4) + f (-x) = 0) ∧  -- Property 2
  (∀ x ∈ Set.Ioo (π/4) (π/2), ∀ y ∈ Set.Ioo (π/4) (π/2), x < y → f y < f x) :=  -- Decreasing on (π/4, π/2)
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_problem_l1063_106371

/-- Given an import tax rate, a tax threshold, and the amount of tax paid, 
    calculate the total value of an item. -/
noncomputable def calculate_total_value (tax_rate : ℝ) (threshold : ℝ) (tax_paid : ℝ) : ℝ :=
  (tax_paid / tax_rate) + threshold

/-- Prove that given the specified conditions, the total value of the item is $2,560. -/
theorem import_tax_problem :
  let tax_rate : ℝ := 0.07
  let threshold : ℝ := 1000
  let tax_paid : ℝ := 109.20
  calculate_total_value tax_rate threshold tax_paid = 2560 := by
  sorry

-- Use #eval with a function that returns a rational number instead
def calculate_total_value_rat (tax_rate : ℚ) (threshold : ℚ) (tax_paid : ℚ) : ℚ :=
  (tax_paid / tax_rate) + threshold

#eval calculate_total_value_rat (7/100) 1000 (2773/25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_problem_l1063_106371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_correct_l1063_106304

/-- Molar mass of Carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of Hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.008

/-- Molar mass of Butane (C4H10) in g/mol -/
def molar_mass_butane : ℝ := 4 * molar_mass_C + 10 * molar_mass_H

/-- Molar mass of Propane (C3H8) in g/mol -/
def molar_mass_propane : ℝ := 3 * molar_mass_C + 8 * molar_mass_H

/-- Molar mass of Methane (CH4) in g/mol -/
def molar_mass_methane : ℝ := molar_mass_C + 4 * molar_mass_H

/-- Total weight of the mixture in grams -/
def total_weight : ℝ := 8 * molar_mass_butane + 5 * molar_mass_propane + 3 * molar_mass_methane

/-- Theorem stating that the total weight is approximately equal to 733.556 grams -/
theorem total_weight_is_correct : ∃ ε > 0, |total_weight - 733.556| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_is_correct_l1063_106304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l1063_106302

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan (2 * x + π / 4)

-- Define the set of x for which f(x) ≥ √3
def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, π / 24 + k * π / 2 ≤ x ∧ x < π / 8 + k * π / 2}

-- Theorem statement
theorem tan_inequality_solution :
  {x : ℝ | f x ≥ sqrt 3} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l1063_106302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equality_n_317_l1063_106352

theorem cos_equality_n_317 (n : ℝ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equality_n_317_l1063_106352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1063_106398

/-- The initial investment that grows to a specified amount over a given time period with compound interest. -/
noncomputable def initial_investment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / (1 + interest_rate) ^ years

/-- Theorem stating that the initial investment of $600 / (1.08^5) grows to $600 after 5 years at 8% annual compound interest. -/
theorem investment_growth (initial : ℝ) (h : initial = initial_investment 600 0.08 5) :
  initial * (1 + 0.08) ^ 5 = 600 := by
  sorry

/-- Compute an approximate value for the initial investment -/
def approximate_initial_investment : ℚ :=
  600 / (1 + 0.08) ^ 5

#eval approximate_initial_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1063_106398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l1063_106314

/-- The radius of a circle inscribed in a rhombus with given diagonals --/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / (4 * Real.sqrt ((d1/2)^2 + (d2/2)^2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 14 and 30 is 105 / (2 * √274) --/
theorem inscribed_circle_radius_specific : 
  inscribed_circle_radius 14 30 = 105 / (2 * Real.sqrt 274) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l1063_106314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1063_106392

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.b / (Real.sqrt 3 * Real.cos t.B) = t.c / Real.sin t.C) :
  t.B = π / 3 ∧ 
  (t.b = 6 → 6 < t.a + t.c ∧ t.a + t.c ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1063_106392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_pq_distance_l1063_106395

noncomputable def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 - b^2), 0)

noncomputable def LineIntersection (m n : ℝ) (p : ℝ × ℝ) (f : ℝ × ℝ) : ℝ × ℝ :=
  let k := (n * (f.1 - p.1)) / (m * (f.2 - p.2) - n * (f.1 - p.1))
  (p.1 + k * (f.1 - p.1), p.2 + k * (f.2 - p.2))

theorem ellipse_pq_distance :
  ∀ (m n : ℝ),
    (m, n) ∈ Ellipse 2 (Real.sqrt 3) →
    let f := RightFocus 2 (Real.sqrt 3)
    let q := LineIntersection m n (m, n) f
    (q.1 - m)^2 + (q.2 - n)^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_pq_distance_l1063_106395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1063_106372

theorem power_equation_solution (y : ℝ) : (16 : ℝ)^y = (4 : ℝ)^16 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1063_106372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_other_unfoldable_polyhedron_l1063_106378

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  is_convex : Bool

/-- Represents the property of being unfoldable into a triangle without internal cuts -/
def unfoldable_to_triangle (p : ConvexPolyhedron) : Prop :=
  sorry -- Define the property

/-- A tetrahedral pyramid with equal opposite edges -/
def tetrahedral_pyramid_equal_edges : ConvexPolyhedron :=
  { is_convex := true }

/-- Main theorem: There exists a convex polyhedron, other than a tetrahedral pyramid 
    with equal opposite edges, that can be unfolded into a triangle without internal cuts -/
theorem exists_other_unfoldable_polyhedron : 
  ∃ (p : ConvexPolyhedron), p ≠ tetrahedral_pyramid_equal_edges ∧ 
  p.is_convex ∧ unfoldable_to_triangle p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_other_unfoldable_polyhedron_l1063_106378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_is_17_35_l1063_106385

/-- A regular hexagon -/
structure RegularHexagon where
  /-- The set of all sides and diagonals -/
  T : Finset (ℕ × ℕ)
  /-- There are 6 sides -/
  sides_count : (T.filter (fun s => s.1 = 1)).card = 6
  /-- There are 9 diagonals -/
  diagonals_count : (T.filter (fun s => s.1 = 2)).card = 9
  /-- Each element in T is either a side or a diagonal -/
  all_segments : ∀ s ∈ T, s.1 = 1 ∨ s.1 = 2

/-- The probability of selecting two segments of the same length -/
def prob_same_length (h : RegularHexagon) : ℚ :=
  17 / 35

/-- The theorem stating the probability of selecting two segments of the same length -/
theorem prob_same_length_is_17_35 (h : RegularHexagon) :
  prob_same_length h = 17 / 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_length_is_17_35_l1063_106385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106373

def f (x : ℝ) : ℝ := x^2 - 2*x + 7

theorem f_properties :
  (f 2 = 7) ∧
  (∀ x, f (x - 1) = x^2 - 4*x + 10) ∧
  (∀ x, f (x + 1) = x^2 + 6) ∧
  (∀ y, y ∈ Set.range (λ x ↦ f (x + 1)) ↔ y ≥ 6) :=
by
  constructor
  · -- Proof for f 2 = 7
    sorry
  constructor
  · -- Proof for ∀ x, f (x - 1) = x^2 - 4*x + 10
    sorry
  constructor
  · -- Proof for ∀ x, f (x + 1) = x^2 + 6
    sorry
  · -- Proof for ∀ y, y ∈ Set.range (λ x ↦ f (x + 1)) ↔ y ≥ 6
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1063_106373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_classification_l1063_106338

-- Define the type for points in the coordinate plane
def Point := ℤ × ℤ

-- Define the type for lines in the coordinate plane
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- Define membership for a point in a line
instance : Membership Point Line where
  mem p l := l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the set of all lines
def I : Set Line := Set.univ

-- Define the set of lines passing through exactly one integer point
def M : Set Line := {l : Line | ∃! p : Point, p ∈ l}

-- Define the set of lines passing through no integer points
def N : Set Line := {l : Line | ∀ p : Point, p ∉ l}

-- Define the set of lines passing through infinitely many integer points
def P : Set Line := {l : Line | ∃ f : ℕ → Point, Function.Injective f ∧ ∀ n, f n ∈ l}

-- State the theorem
theorem line_classification :
  (M ∪ N ∪ P = I) ∧
  (N ≠ ∅) ∧
  (M ≠ ∅) ∧
  (P ≠ ∅) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_classification_l1063_106338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_inscribed_rectangles_l1063_106391

/-- A triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Determines if a triangle is acute -/
noncomputable def isAcute (t : Triangle) : Bool := sorry

/-- The locus of centers of inscribed rectangles around a triangle -/
def locusOfCenters (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The set of points forming a curvilinear triangle within the given triangle -/
def curvilinearTriangle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The set of points forming two arcs on the shorter midlines of the triangle -/
def twoArcs (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Theorem stating the locus of centers for inscribed rectangles around a triangle -/
theorem locus_of_centers_inscribed_rectangles (t : Triangle) :
  locusOfCenters t = if isAcute t then curvilinearTriangle t else twoArcs t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_inscribed_rectangles_l1063_106391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1063_106312

/-- The hyperbola C with equation x²/m² - y²/(m²-1) = 1 -/
def Hyperbola (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / m^2) - (p.2^2 / (m^2 - 1)) = 1}

/-- The left focus of the hyperbola -/
noncomputable def LeftFocus (m : ℝ) : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
noncomputable def RightFocus (m : ℝ) : ℝ × ℝ := sorry

/-- Definition of perpendicular vectors -/
def Perpendicular (v w : ℝ × ℝ) : Prop := sorry

/-- The area of a triangle given by three points -/
noncomputable def TriangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def Eccentricity (m : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity (m : ℝ) :
  ∃ (p : ℝ × ℝ),
    p ∈ Hyperbola m ∧
    Perpendicular (p.1 - (LeftFocus m).1, p.2 - (LeftFocus m).2)
                  (p.1 - (RightFocus m).1, p.2 - (RightFocus m).2) ∧
    TriangleArea p (LeftFocus m) (RightFocus m) = 3 →
  Eccentricity m = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1063_106312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_1_best_best_prices_l1063_106327

/-- Represents the discount amount for a given coupon and price -/
noncomputable def discount (coupon : Nat) (price : ℝ) : ℝ :=
  match coupon with
  | 1 => if price ≥ 60 then 0.12 * price else 0
  | 2 => if price ≥ 120 then 25 else 0
  | 3 => 0.2 * (price - 120)
  | _ => 0

/-- Theorem stating when Coupon 1 offers the best discount -/
theorem coupon_1_best (price : ℝ) :
  (discount 1 price > discount 2 price ∧ discount 1 price > discount 3 price) ↔
  (208.33 < price ∧ price < 300) :=
by sorry

/-- Checks if a given price satisfies the condition for Coupon 1 being the best -/
noncomputable def is_coupon_1_best (price : ℝ) : Bool :=
  208.33 < price ∧ price < 300

/-- List of prices to check -/
def prices : List ℝ := [189.95, 209.95, 229.95, 249.95, 269.95]

/-- Theorem stating which prices in the list satisfy the condition -/
theorem best_prices :
  prices.filter is_coupon_1_best = [209.95, 229.95, 249.95, 269.95] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_1_best_best_prices_l1063_106327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_generating_set_l1063_106343

/-- A set of integers that can generate all residues modulo 100 through exponentiation -/
def GeneratingSet (A : Set ℤ) : Prop :=
  ∀ m : ℤ, ∃ (a : ℤ) (n : ℕ), a ∈ A ∧ n > 0 ∧ a^n ≡ m [ZMOD 100]

/-- The theorem stating the smallest cardinality of a generating set -/
theorem smallest_generating_set :
  ∃ (A : Finset ℤ), GeneratingSet (A : Set ℤ) ∧ A.card = 41 ∧
  (∀ (B : Finset ℤ), GeneratingSet (B : Set ℤ) → A.card ≤ B.card) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_generating_set_l1063_106343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_rolling_problem_l1063_106360

theorem cone_rolling_problem (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  h / r = 3 * Real.sqrt 133 →
  let m : ℕ := 3
  let n : ℕ := 133
  m + n = 136 := by
  intro h_eq
  -- Proof goes here
  sorry

#check cone_rolling_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_rolling_problem_l1063_106360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l1063_106313

/-- The speed of a train traveling between two stations -/
noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The problem statement -/
theorem first_train_speed 
  (distance_AB : ℝ) 
  (start_time_A start_time_B meeting_time : ℝ) 
  (speed_B : ℝ) :
  distance_AB = 20 →
  start_time_A = 7 →
  start_time_B = 8 →
  meeting_time = 8 →
  speed_B = 25 →
  train_speed distance_AB (meeting_time - start_time_A) = 20 :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l1063_106313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_is_6_25_l1063_106351

/-- Calculates the volume of a cylinder submerged in water --/
noncomputable def cylinder_volume (initial_level : ℝ) (submerged_level : ℝ) (reference_point : ℝ) (total_scale : ℝ) : ℝ :=
  let displaced_volume := submerged_level - initial_level
  let scale_ratio := (total_scale - reference_point) / (submerged_level - reference_point)
  scale_ratio * displaced_volume

/-- Theorem stating that the volume of the cylinder is 6.25 ml --/
theorem cylinder_volume_is_6_25 :
  let initial_level : ℝ := 35
  let submerged_level : ℝ := 40
  let reference_point : ℝ := 20
  let total_scale : ℝ := 45
  cylinder_volume initial_level submerged_level reference_point total_scale = 6.25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_is_6_25_l1063_106351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_proof_l1063_106331

theorem cube_sum_proof (a b x y z : ℝ) : 
  a^2 = 16/44 →
  b^2 = (2 + Real.sqrt 5)^2 / 11 →
  a < 0 →
  b > 0 →
  (a + b)^3 = x * Real.sqrt y / z →
  x > 0 →
  y > 0 →
  z > 0 →
  ∃ (n : ℤ), x = n →
  ∃ (m : ℤ), y = m →
  ∃ (k : ℤ), z = k →
  x + y + z = 181 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_proof_l1063_106331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_place_value_l1063_106332

/-- Represents a decimal number with three decimal places -/
structure Decimal where
  whole : ℕ
  tenths : ℕ
  hundredths : ℕ
  thousandths : ℕ
  inv_tenths : tenths < 10
  inv_hundredths : hundredths < 10
  inv_thousandths : thousandths < 10

/-- Converts a Decimal to a real number -/
noncomputable def Decimal.toReal (d : Decimal) : ℝ :=
  d.whole + (d.tenths : ℝ) / 10 + (d.hundredths : ℝ) / 100 + (d.thousandths : ℝ) / 1000

theorem decimal_place_value (d : Decimal) (h : d.toReal = 8.063) :
  d.thousandths = 3 ∧ 
  (d.thousandths : ℝ) / 1000 = 3 * 0.001 ∧ 
  0.48 = 48 * 0.01 := by
  sorry

#check decimal_place_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_place_value_l1063_106332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l1063_106383

noncomputable def sequenceTerm (n : ℕ) : ℚ :=
  (-1)^(n-1) * (3^(n-1) : ℚ) / (2*n - 1)

theorem sequence_first_four_terms :
  sequenceTerm 1 = -1 ∧
  sequenceTerm 2 = 1 ∧
  sequenceTerm 3 = -9/5 ∧
  sequenceTerm 4 = 27/7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l1063_106383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_functions_properties_l1063_106399

/-- Given two parallel vectors m and n, prove properties about functions f and g --/
theorem vector_functions_properties (x : ℝ) (f : ℝ → ℝ) :
  let m : ℝ × ℝ := (f x, 2 * Real.cos x)
  let n : ℝ × ℝ := (Real.sin x + Real.cos x, 1)
  let g (x : ℝ) := Real.sqrt 2 * Real.sin (4 * x + π / 4)
  (∃ k : ℝ, m = k • n) →
  (f x = Real.sqrt 2 * Real.sin (2 * x + π / 4) + 1) ∧
  (∀ y ∈ Set.Icc 0 (π / 8), g y ≤ Real.sqrt 2) ∧
  (g (π / 16) = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_functions_properties_l1063_106399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasek_solved_18_l1063_106380

/-- The number of problems solved by Majka -/
def m : ℕ := sorry

/-- The number of problems solved by Vašek -/
def v : ℕ := sorry

/-- The number of problems solved by Zuzka -/
def z : ℕ := sorry

/-- Majka and Vašek solved a total of 25 problems -/
axiom majka_vasek_total : m + v = 25

/-- Zuzka and Vašek solved a total of 32 problems -/
axiom zuzka_vasek_total : z + v = 32

/-- Zuzka solved twice as many problems as Majka -/
axiom zuzka_double_majka : z = 2 * m

theorem vasek_solved_18 : v = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasek_solved_18_l1063_106380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1063_106320

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + 1 else 1

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, f (1 - x^2) > f (2*x) ↔ -1 < x ∧ x < Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1063_106320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1063_106311

-- Define the set S as real numbers excluding 2/3
noncomputable def S : Set ℝ := {x : ℝ | x ≠ 2/3}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1992 * x * (x - 1) / (3 * x - 2)

-- State the theorem
theorem function_satisfies_equation :
  ∀ x ∈ S, 2 * f x + f (2 * x / (3 * x - 2)) = 996 * x := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l1063_106311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l1063_106307

/-- The capacity of a pool given specific filling conditions -/
noncomputable def pool_capacity (fill_time_both : ℝ) (fill_time_first : ℝ) (rate_difference : ℝ) : ℝ :=
  let first_valve_rate := 1 / fill_time_first
  let second_valve_rate := first_valve_rate + rate_difference
  let combined_rate := 1 / fill_time_both
  (combined_rate - first_valve_rate) / (second_valve_rate - first_valve_rate) * 
  (fill_time_both * fill_time_first * rate_difference)

/-- Theorem stating the capacity of the pool under given conditions -/
theorem pool_capacity_calculation :
  pool_capacity (48 / 60) 2 (50 / 60) = 12000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_calculation_l1063_106307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_of_sample_min_variance_value_min_variance_equals_three_l1063_106375

noncomputable def sample_average (x y : ℝ) : ℝ := (x + 1 + y + 5) / 4

noncomputable def sample_variance (x y : ℝ) : ℝ := 
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4

theorem min_variance_of_sample (x y : ℝ) 
  (h : sample_average x y = 2) : 
  ∀ a b : ℝ, sample_average a b = 2 → sample_variance x y ≤ sample_variance a b :=
by sorry

theorem min_variance_value (x y : ℝ) 
  (h : sample_average x y = 2) : 
  sample_variance x y ≥ 3 :=
by sorry

theorem min_variance_equals_three : 
  ∃ x y : ℝ, sample_average x y = 2 ∧ sample_variance x y = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_of_sample_min_variance_value_min_variance_equals_three_l1063_106375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_to_charlie_l1063_106322

/-- The centroid of three points in the plane -/
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

/-- The vertical distance between two points -/
noncomputable def vertical_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  p2.2 - p1.2

theorem meeting_point_to_charlie : 
  let annie : ℝ × ℝ := (6, -20)
  let barbara : ℝ × ℝ := (1, 14)
  let david : ℝ × ℝ := (0, -6)
  let charlie : ℝ × ℝ := (7/2, 2)
  let meeting_point := centroid annie barbara david
  vertical_distance meeting_point charlie = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_to_charlie_l1063_106322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_calculations_l1063_106336

theorem square_root_calculations :
  (Real.sqrt 9 = 3) ∧
  (-Real.sqrt 0.49 = -0.7) ∧
  (∀ (x : ℝ), x = Real.sqrt (64/81) ∨ x = -Real.sqrt (64/81) ↔ x = 8/9 ∨ x = -8/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_calculations_l1063_106336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l1063_106306

-- Define the function f(x) = lg((1-x)/(1+x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Define the domain of the function
def domain : Set ℝ := { x | -1 < x ∧ x < 1 }

-- Theorem statement
theorem f_odd_and_decreasing :
  (∀ x, x ∈ domain → f (-x) = -f x) ∧
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l1063_106306
