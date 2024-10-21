import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l27_2764

-- Define the set S
def S : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + 1/2*Complex.I) * z

-- Theorem statement
theorem transform_stays_in_S : ∀ z ∈ S, transform z ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l27_2764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_is_square_l27_2752

theorem subset_product_is_square (S : Finset ℕ) :
  S.card = 26 →
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 100) →
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → a ≠ b) →
  ∃ T : Finset ℕ, T.Nonempty ∧ T ⊆ S ∧ ∃ m : ℕ, (T.prod id = m ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_product_is_square_l27_2752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_painted_faces_count_total_painted_faces_l27_2747

/-- Represents a smaller cube within a 3x3x3 larger cube --/
structure SmallCube where
  x : Fin 3
  y : Fin 3
  z : Fin 3
deriving Fintype, DecidableEq

/-- Counts the number of painted faces on a small cube --/
def paintedFaces (c : SmallCube) : Nat :=
  (if c.x = 0 || c.x = 2 then 1 else 0) +
  (if c.y = 0 || c.y = 2 then 1 else 0) +
  (if c.z = 0 || c.z = 2 then 1 else 0)

/-- The set of all small cubes in the 3x3x3 larger cube --/
def allCubes : Finset SmallCube :=
  Finset.univ

/-- The set of small cubes with exactly two painted faces --/
def twoPaintedFaces : Finset SmallCube :=
  allCubes.filter (fun c => paintedFaces c = 2)

theorem two_painted_faces_count :
  twoPaintedFaces.card = 12 := by
  sorry

theorem total_painted_faces :
  (twoPaintedFaces.card * 2 : Nat) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_painted_faces_count_total_painted_faces_l27_2747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l27_2784

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem geometric_sequence_theorem (a : ℕ → ℝ) (k : ℝ) :
  geometric_sequence a ∧
  increasing_sequence a ∧
  a 1 * a 2 * a 3 = 8 ∧
  arithmetic_sequence (λ n => if n = 1 then a 1 + 1 else if n = 2 then a 2 + 2 else if n = 3 then a 3 + 2 else 0) ∧
  (∀ n : ℕ+, a n^2 + (2:ℝ)^(n:ℝ) * a n - k ≥ 0) →
  (∀ n : ℕ, a n = (2:ℝ)^((n:ℝ)-1)) ∧
  k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_theorem_l27_2784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_over_four_l27_2779

theorem tan_alpha_plus_pi_over_four (α : Real) :
  Real.tan (α + π/4) = 1/2 →
  α > -π/2 →
  α < 0 →
  (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (α - π/4) = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_over_four_l27_2779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l27_2729

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin x + 2 * x

-- Define the interval
def I : Set ℝ := Set.Icc (-π) π

-- State the theorem
theorem min_t_value (t : ℝ) :
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ t) ↔ t ≥ 4 * π :=
sorry

#check min_t_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l27_2729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l27_2705

/-- The nested radical function defined recursively --/
noncomputable def nestedRadical (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => Real.sqrt (4^2008 * x + 3)
  | n + 1 => Real.sqrt (4^n * x + nestedRadical n x)

/-- The theorem statement --/
theorem nested_radical_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (x + nestedRadical 2008 x) * Real.sqrt x = 1 ∧ x = 1 / 2^4018 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equation_solution_l27_2705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l27_2769

noncomputable def star (a b : ℝ) : ℝ :=
  if a < b then b else a

noncomputable def f (x : ℝ) : ℝ :=
  star (Real.log x / Real.log 2) (-Real.log x / Real.log 2)

-- Theorem statement
theorem f_range :
  Set.range f = Set.Ici (0 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l27_2769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_foci_vertices_ratio_l27_2721

-- Define the parabolas P and Q
noncomputable def P : Set (ℝ × ℝ) := {(x, y) | y = -x^2}
noncomputable def Q : Set (ℝ × ℝ) := {(x, y) | ∃ a b : ℝ, (a, -a^2) ∈ P ∧ (b, -b^2) ∈ P ∧ a * b = 1 ∧ x = (a + b) / 2 ∧ y = -(a + b)^2 / 2 - 1}

-- Define the vertices and foci
noncomputable def V1 : ℝ × ℝ := (0, 0)
noncomputable def F1 : ℝ × ℝ := (0, -1/4)
noncomputable def V2 : ℝ × ℝ := (0, -1)
noncomputable def F2 : ℝ × ℝ := (0, -9/8)

-- State the theorem
theorem parabola_foci_vertices_ratio : 
  let d1 := dist F1 F2
  let d2 := dist V1 V2
  d1 / d2 = 7/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_foci_vertices_ratio_l27_2721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larinjaitis_lifespan_l27_2762

/-- Represents a year in the BC/AD calendar system -/
structure Year where
  value : Int

/-- Converts a BC/AD year to its integer representation -/
def to_int (y : Year) : Int :=
  if y.value ≤ 0 then y.value else y.value - 1

/-- Calculates the number of years lived given birth and death years -/
def years_lived (birth death : Year) : Nat :=
  (to_int death - to_int birth).toNat + 1

theorem larinjaitis_lifespan :
  years_lived (Year.mk (-30)) (Year.mk 30) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larinjaitis_lifespan_l27_2762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l27_2709

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, IsRegular (f x) ↔ x ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l27_2709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l27_2755

/-- The polynomial in question -/
noncomputable def p (a b : ℝ) : ℝ := -3 * a^2 * b + (5/2) * a^2 * b^3 - a * b + 1

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ → ℝ) : ℕ :=
  sorry -- Definition of degree would go here

/-- Theorem: The degree of the polynomial p is 5 -/
theorem degree_of_p : degree p = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_p_l27_2755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l27_2794

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := y^2/16 + x^2/9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 3*x - 4*y = 24

/-- Distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - 4*y - 24| / Real.sqrt (3^2 + (-4)^2)

/-- The maximum distance from any point on the ellipse to the line -/
theorem max_distance_ellipse_to_line :
  ∃ (max_d : ℝ), max_d = (12/5)*(2 + Real.sqrt 2) ∧
  ∀ (x y : ℝ), ellipse x y → distance_to_line x y ≤ max_d := by
  sorry

#check max_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l27_2794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l27_2772

def f (x : ℤ) : ℤ := x^3 - x^2 + 2*x + 2013

theorem gcd_f_100_101 : Int.gcd (f 100) (f 101) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l27_2772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l27_2703

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x)

theorem sine_function_property (A ω a : ℝ) (h_pos : ω > 0) :
  (∀ x, f A ω (x - 1/2) = f A ω (x + 1/2)) →
  f A ω (-1/4) = -a →
  f A ω (9/4) = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_property_l27_2703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_distance_theorem_l27_2754

-- Define a partition of space into three disjoint sets
def Partition (α : Type) := 
  ∃ (A B C : Set α), 
    (Disjoint A B) ∧ 
    (Disjoint A C) ∧ 
    (Disjoint B C) ∧ 
    (A ∪ B ∪ C = Set.univ)

-- Define the theorem
theorem partition_distance_theorem {α : Type} [MetricSpace α] (p : Partition α) (d : ℝ) (h : d > 0) :
  ∃ (S : Set α), ∃ (x y : α), x ∈ S ∧ y ∈ S ∧ dist x y = d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_distance_theorem_l27_2754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfourth_decimal_of_35_36_l27_2778

/-- The 24th decimal place in the decimal expansion of 35/36 is 2 -/
theorem twentyfourth_decimal_of_35_36 : ∃ (n : ℕ),
  (35 : ℚ) / 36 = (n : ℚ) / 10^24 + 2 / 10^25 + (35 / 36 - ((n : ℚ) / 10^24 + 2 / 10^25)) ∧
  (35 / 36 - ((n : ℚ) / 10^24)) * 10^25 < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfourth_decimal_of_35_36_l27_2778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_standard_deviation_l27_2701

noncomputable def apple_weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def standard_deviation (xs : List ℝ) : ℝ :=
  let n := xs.length
  let mean := xs.sum / n
  Real.sqrt ((xs.map (λ x => (x - mean) ^ 2)).sum / n)

theorem apple_weights_standard_deviation :
  standard_deviation apple_weights = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_standard_deviation_l27_2701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_minimum_P_minimizes_dot_product_l27_2761

-- Define the points A and B
def A : Fin 3 → ℝ := ![1, 2, 0]
def B : Fin 3 → ℝ := ![0, 1, -1]

-- Define a function for the dot product of AP and BP
def dot_product (x : ℝ) : ℝ :=
  let P : Fin 3 → ℝ := ![x, 0, 0]
  let AP : Fin 3 → ℝ := λ i => P i - A i
  let BP : Fin 3 → ℝ := λ i => P i - B i
  (AP 0 * BP 0) + (AP 1 * BP 1) + (AP 2 * BP 2)

-- Theorem: The dot product is minimized when x = 1/2
theorem dot_product_minimum :
  ∀ x : ℝ, dot_product (1/2) ≤ dot_product x := by
  sorry

-- Theorem: The point P(1/2, 0, 0) minimizes the dot product
theorem P_minimizes_dot_product :
  let P : Fin 3 → ℝ := ![1/2, 0, 0]
  ∀ Q : Fin 3 → ℝ, Q 1 = 0 ∧ Q 2 = 0 →
    dot_product (P 0) ≤ dot_product (Q 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_minimum_P_minimizes_dot_product_l27_2761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ice_given_ski_value_l27_2797

structure School where
  ice_skating : ℝ
  skiing : ℝ
  either : ℝ

noncomputable def prob_ice_given_ski (s : School) : ℝ :=
  (s.ice_skating + s.skiing - s.either) / s.skiing

theorem prob_ice_given_ski_value (s : School) 
  (h1 : s.ice_skating = 0.6)
  (h2 : s.skiing = 0.5)
  (h3 : s.either = 0.7) :
  prob_ice_given_ski s = 0.8 := by
  sorry

#check prob_ice_given_ski_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_ice_given_ski_value_l27_2797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_healthy_male_child_l27_2753

-- Define the genes
inductive Gene
| P  -- Normal allele for phenylketonuria
| p  -- Mutated allele for phenylketonuria
| A  -- Normal allele for alkaptonuria
| a  -- Mutated allele for alkaptonuria
| XH -- Normal X chromosome for hemophilia
| Xh -- X chromosome with hemophilia mutation
| Y  -- Y chromosome
deriving BEq, Repr

-- Define a genotype as a list of genes
def Genotype := List Gene

-- Define the individuals
def I3 : Genotype := [Gene.P, Gene.p, Gene.A, Gene.a, Gene.XH, Gene.Y]
def II3 : Genotype := [Gene.P, Gene.P, Gene.A, Gene.A, Gene.XH, Gene.XH]

-- Define a function to check if a genotype is healthy
def is_healthy (g : Genotype) : Bool :=
  (g.count Gene.p < 2) && (g.count Gene.a < 2) && 
  (if g.contains Gene.Y then g.contains Gene.XH else g.count Gene.Xh < 2)

-- Define a function to simulate child genotype inheritance
noncomputable def child_genotype (mother : Genotype) (father : Genotype) : Genotype :=
  sorry -- Implementation details omitted for brevity

-- Theorem statement
theorem probability_of_healthy_male_child :
  let male_children := List.replicate 1000 (child_genotype II3 [Gene.P, Gene.P, Gene.A, Gene.A, Gene.XH, Gene.Y])
  (male_children.filter is_healthy).length / male_children.length = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_healthy_male_child_l27_2753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l27_2760

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) := x^2 + 1
noncomputable def g (x : ℝ) := log x

-- State the theorem
theorem min_difference_f_g :
  ∃ (m : ℝ), m = (3/2 : ℝ) + (1/2 : ℝ) * log 2 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_f_g_l27_2760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l27_2780

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type with slope k passing through (1, 0) -/
structure Line where
  k : ℝ
  x : ℝ
  y : ℝ
  eq : y = k*(x - 1)

/-- Intersection point of parabola and line -/
def intersection (p : Parabola) (l : Line) : Prop :=
  p.x = l.x ∧ p.y = l.y

/-- Vector from point M to a given point -/
def vector_MA (x y : ℝ) : ℝ × ℝ := (x + 1, y - 1)

theorem parabola_line_intersection :
  ∃ (k : ℝ) (A B : Parabola),
    (intersection A (Line.mk k A.x A.y (by sorry))) ∧
    (intersection B (Line.mk k B.x B.y (by sorry))) ∧
    A ≠ B ∧
    (vector_MA A.x A.y).1 * (vector_MA B.x B.y).1 +
    (vector_MA A.x A.y).2 * (vector_MA B.x B.y).2 = 0 →
    k = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l27_2780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_formation_theorem_l27_2708

/-- Represents a cube --/
structure Cube where
  edgeLength : ℕ
  deriving Repr

/-- Represents a wire --/
structure Wire where
  length : ℕ
  deriving Repr

/-- Represents the number of edges in a cube --/
def cubeEdges : ℕ := 12

/-- Represents the number of vertices in a cube --/
def cubeVertices : ℕ := 8

/-- Represents the degree of each vertex in a cube --/
def cubeVertexDegree : ℕ := 3

/-- Function to check if a wire can form a cube without breaking --/
def canFormCubeWithoutBreaking (w : Wire) (c : Cube) : Prop :=
  w.length ≥ c.edgeLength * cubeEdges ∧
  ∃ (path : List ℕ), path.length = cubeEdges ∧ path.toFinset.card = cubeVertices

/-- Function to calculate the minimum number of breaks needed --/
def minBreaksNeeded (w : Wire) (c : Cube) : ℕ :=
  if w.length < c.edgeLength * cubeEdges then
    0  -- Not enough wire to form the cube
  else
    (cubeVertices / 2) - 1

/-- Theorem stating the impossibility of forming the cube without breaking and the minimum breaks needed --/
theorem cube_formation_theorem (w : Wire) (c : Cube) 
    (h1 : w.length = 120)
    (h2 : c.edgeLength = 10) :
    ¬(canFormCubeWithoutBreaking w c) ∧ minBreaksNeeded w c = 3 := by
  sorry

-- Remove the #eval line as it's not necessary for building


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_formation_theorem_l27_2708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_gt_one_eighth_l27_2789

-- Define the function f
noncomputable def f (a l x : ℝ) : ℝ := x^2 - x + l + a * Real.log x

-- State the theorem
theorem monotone_increasing_implies_a_gt_one_eighth
  (h : ∀ l : ℝ, StrictMono (f a l)) :
  a > 1/8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_gt_one_eighth_l27_2789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_floor_is_two_l27_2771

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- Theorem statement
theorem root_floor_is_two (x₀ : ℝ) (h : f x₀ = 0) : floor x₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_floor_is_two_l27_2771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_shore_probability_l27_2713

/-- Represents the probability of a bridge surviving an earthquake --/
noncomputable def p : ℝ := 1/2

/-- Represents the probability of reaching the shore from the first island 
    given an infinite sequence of islands connected by bridges, 
    where each bridge has a probability p of surviving an earthquake --/
noncomputable def probability_reach_shore (p : ℝ) : ℝ :=
  p / (1 - p * (1 - p))

/-- Theorem stating that the probability of reaching the shore is 2/3 --/
theorem reach_shore_probability :
  probability_reach_shore p = 2/3 := by
  sorry

#check reach_shore_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_shore_probability_l27_2713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l27_2749

/-- The volume of a right triangular prism with specific conditions -/
theorem right_triangular_prism_volume : 
  ∀ (base_edge height : ℝ), 
    base_edge > 0 →
    height > 0 →
    height = 2 * base_edge →
    3 * base_edge * height = 18 →
    (Real.sqrt 3 / 4) * base_edge^2 * height = 9/2 :=
by
  intros base_edge height h1 h2 h3 h4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangular_prism_volume_l27_2749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l27_2712

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- State the theorem
theorem f_composition (x : ℝ) (h : -1 < x ∧ x < 1) : 
  f ((2 * x + x^2) / (1 + 2 * x)) = 2 * f x := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_l27_2712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l27_2733

-- Define set P
def P : Set ℝ := {x : ℝ | |x| ≥ 3}

-- Define set Q
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = Set.Iic (-3) ∪ Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l27_2733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l27_2736

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 9) / (x - 3)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 3 → (f x > 0 ↔ (x > -3 ∧ x < 3) ∨ x > 3) :=
by
  -- The proof is skipped using 'sorry'
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l27_2736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l27_2751

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides
  (h_angles : A + B + C = Real.pi)  -- sum of angles is π
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)  -- sides are positive

-- Define the circumcenter O
noncomputable def circumcenter (t : Triangle) : Real × Real := sorry

theorem triangle_properties (t : Triangle) :
  let O := circumcenter t
  -- Statement 1
  (Real.cos t.B * Real.cos t.C > Real.sin t.B * Real.sin t.C → t.A > Real.pi/2) ∧
  -- Statement 2 (negation)
  (t.a * Real.cos t.A = t.b * Real.cos t.B → ¬(t.a = t.b ∨ t.b = t.c ∨ t.c = t.a)) ∧
  -- Statement 3
  (let a : Real × Real := (Real.tan t.A + Real.tan t.B, Real.tan t.C)
   let b : Real × Real := (1, 1)
   (a.1 * b.1 + a.2 * b.2 > 0) → (t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2)) ∧
  -- Statement 4
  (let BC : Real × Real := (t.c * Real.cos t.B - t.b * Real.cos t.C, t.c * Real.sin t.B - t.b * Real.sin t.C)
   let AO : Real × Real := ((O.1 - t.c * Real.cos t.B), (O.2 - t.c * Real.sin t.B))
   AO.1 * BC.1 + AO.2 * BC.2 = (t.b^2 - t.c^2) / 2) ∧
  -- Statement 5
  (Real.sin t.A^2 + Real.sin t.B^2 = Real.sin t.C^2 ∧
   ∃ (OA OB OC : Real × Real), OA + OB + OC = (0, 0) →
   (OA.1^2 + OA.2^2 + OB.1^2 + OB.2^2) / (OC.1^2 + OC.2^2) = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l27_2751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l27_2770

-- Define the rectangle
def Rectangle (length width : ℝ) : Prop :=
  length > 0 ∧ width > 0

-- Define the perimeter of a rectangle
def perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

-- Define the area of a rectangle
def area (length width : ℝ) : ℝ :=
  length * width

-- Define the diagonal of a rectangle
noncomputable def diagonal (length width : ℝ) : ℝ :=
  Real.sqrt (length^2 + width^2)

theorem rectangle_diagonal : 
  ∃ (length width : ℝ), 
    Rectangle length width ∧ 
    perimeter length width = 10 ∧ 
    area length width = 6 ∧ 
    diagonal length width = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l27_2770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l27_2714

/-- Represents the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem speed_difference_proof (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h1 : distance = 200)
  (h2 : time_heavy = 5)
  (h3 : time_no = 4) :
  average_speed distance time_no - average_speed distance time_heavy = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_proof_l27_2714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l27_2746

/-- The conversion factor from degrees to radians -/
noncomputable def degToRad : ℝ := Real.pi / 180

/-- Converts an angle from degrees to radians -/
noncomputable def degreesToRadians (degrees : ℝ) : ℝ := degrees * degToRad

theorem negative_150_degrees_to_radians :
  degreesToRadians (-150) = -(5 * Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l27_2746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l27_2739

/-- Calculates the length of a train given its speed and time to pass a stationary object. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Proves that a train traveling at 180 km/hr and passing a stationary object in 10 seconds has a length of 500 meters. -/
theorem train_length_proof (speed : ℝ) (time : ℝ) 
  (h1 : speed = 180) 
  (h2 : time = 10) : 
  trainLength speed time = 500 := by
  sorry

-- Use #eval only for computable functions
def trainLengthNat (speed : Nat) (time : Nat) : Nat :=
  speed * 1000 * time / 3600

#eval trainLengthNat 180 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l27_2739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l27_2710

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (n + 4) * (n - 9) ≤ 0) ∧ Finset.card S = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l27_2710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erica_catch_ratio_l27_2722

/-- Represents the fish catching scenario for Erica --/
structure FishScenario where
  price_per_kg : ℚ
  past_four_months_catch : ℚ
  total_earnings : ℚ

/-- Calculates the ratio of fish caught today to fish caught in the past four months --/
noncomputable def catch_ratio (scenario : FishScenario) : ℚ :=
  let today_earnings := scenario.total_earnings - scenario.price_per_kg * scenario.past_four_months_catch
  let today_catch := today_earnings / scenario.price_per_kg
  today_catch / scenario.past_four_months_catch

/-- Theorem stating that the catch ratio is 2:1 for Erica's scenario --/
theorem erica_catch_ratio :
  let scenario := FishScenario.mk 20 80 4800
  catch_ratio scenario = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erica_catch_ratio_l27_2722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l27_2781

/-- The ratio of the area of an equilateral triangle to the area of a pentagon formed by
    placing the equilateral triangle atop a square with equal side lengths -/
theorem triangle_to_pentagon_area_ratio : ℝ := by
  -- Define the side length of the square and equilateral triangle
  let s : ℝ := 1  -- We can use any positive real number, 1 for simplicity
  -- Define the area of the square
  let square_area := s^2
  -- Define the area of the equilateral triangle
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  -- Define the area of the pentagon
  let pentagon_area := square_area + triangle_area
  -- The ratio we want to prove
  let ratio := triangle_area / pentagon_area
  -- Assert that the ratio equals (4√3 - 3) / 13
  have h : ratio = (4 * Real.sqrt 3 - 3) / 13 := by
    -- The proof steps would go here
    sorry
  -- Return the ratio
  exact (4 * Real.sqrt 3 - 3) / 13


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l27_2781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l27_2720

theorem equality_condition (a b c p q r : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^p + b^q + c^r = a^q + b^r + c^p ∧
  a^q + b^r + c^p = a^r + b^p + c^q →
  (a = b ∧ b = c) ∨ (p = q ∧ q = r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l27_2720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_24_sided_polygon_approx_96_l27_2700

/-- Represents a square sheet of paper -/
structure Sheet where
  side_length : ℝ
  rotation : ℝ

/-- Calculates the approximate area of the 24-sided polygon formed by overlapping three square sheets -/
noncomputable def approximate_area_24_sided_polygon (sheets : List Sheet) : ℝ :=
  sorry

/-- Theorem stating the approximate area of the 24-sided polygon -/
theorem area_24_sided_polygon_approx_96 :
  let sheets := [
    { side_length := 8, rotation := 0 },
    { side_length := 8, rotation := 45 },
    { side_length := 8, rotation := 75 }
  ]
  ∃ ε > 0, |approximate_area_24_sided_polygon sheets - 96| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_24_sided_polygon_approx_96_l27_2700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_kw_price_l27_2726

/-- The price of Company KW as a percentage of the combined assets of Companies A and B -/
noncomputable def price_percentage : ℚ := 8888888888888889 / 10000000000000000

/-- The price of Company KW relative to Company B's assets -/
def price_to_b_ratio : ℚ := 2

theorem company_kw_price (a b : ℚ) (ha : a > 0) (hb : b > 0) :
  let p := price_percentage * (a + b)
  (p / a - 1) * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_kw_price_l27_2726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_increase_l27_2765

/-- Given a circular grazing area with initial radius 12 meters,
    if the radius is increased such that the area increases by 933.4285714285714 square meters,
    then the new radius is approximately 21 meters. -/
theorem grazing_area_increase (π : ℝ) (initial_radius new_radius : ℝ) : 
  initial_radius = 12 →
  π * new_radius^2 - π * initial_radius^2 = 933.4285714285714 →
  ∃ ε > 0, |new_radius - 21| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grazing_area_increase_l27_2765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_propositions_true_l27_2788

-- Define a custom IsPerpendicularTo relation for lines
def IsPerpendicularTo (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d : ℝ, (∀ x y, l1 x y ↔ a*x + b*y + 1 = 0) ∧
                 (∀ x y, l2 x y ↔ c*x + d*y - 3 = 0) ∧
                 a*c + b*d = 0

theorem four_propositions_true :
  -- Proposition 1
  (¬ ∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3*x) ∧
  -- Proposition 2
  (∃ m : ℝ, m = -2 → 
    IsPerpendicularTo (fun x y => (m + 2)*x + m*y + 1 = 0) (fun x y => (m - 2)*x + (m + 2)*y - 3 = 0) ∧
    ¬ (IsPerpendicularTo (fun x y => (m + 2)*x + m*y + 1 = 0) (fun x y => (m - 2)*x + (m + 2)*y - 3 = 0) → m = -2)) ∧
  -- Proposition 3
  (∀ D E F x₁ x₂ y₁ y₂ : ℝ,
    D^2 + E^2 - 4*F > 0 →
    (x₁^2 + D*x₁ + F = 0) →
    (x₂^2 + D*x₂ + F = 0) →
    (y₁^2 + E*y₁ + F = 0) →
    (y₂^2 + E*y₂ + F = 0) →
    x₁*x₂ - y₁*y₂ = 0) ∧
  -- Proposition 4
  (∀ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 3| ≥ m) → m ≤ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_propositions_true_l27_2788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_center_l27_2783

open Real

-- Define the tangent function
noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 3)

-- Define what it means for a point to be a symmetry center
def is_symmetry_center (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

-- Theorem statement
theorem tan_symmetry_center :
  is_symmetry_center f (5 * π / 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_symmetry_center_l27_2783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_through_point_l27_2742

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between a point and a line given by ax + by + c = 0 -/
noncomputable def distancePointToLine (p : Point) (a b c : ℝ) : ℝ :=
  (abs (a * p.x + b * p.y + c)) / Real.sqrt (a^2 + b^2)

/-- The theorem stating the equations of lines passing through P and equidistant from A and B -/
theorem equidistant_lines_through_point (P A B : Point) 
  (h_P : P = ⟨1, 2⟩) 
  (h_A : A = ⟨2, 3⟩) 
  (h_B : B = ⟨0, -5⟩) : 
  (∃ (k : ℝ), k = 4 ∧ P.y = k * P.x - 2 ∧ 
    distancePointToLine A k (-1) 2 = distancePointToLine B k (-1) 2) ∨
  (P.x = 1 ∧ distancePointToLine A 1 0 (-1) = distancePointToLine B 1 0 (-1)) := by
  sorry

#check equidistant_lines_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_lines_through_point_l27_2742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l27_2716

/-- Represents the time (in minutes) it takes for pipe A to fill the tank alone -/
noncomputable def time_A : ℝ := 60

/-- Represents the time (in minutes) it takes for pipe B to fill the tank alone -/
def time_B : ℝ := 40

/-- Represents the total time (in minutes) it takes to fill the tank using the given strategy -/
def total_time : ℝ := 30

/-- Represents the fraction of the tank filled by pipe B in the first half of the total time -/
noncomputable def fraction_filled_B : ℝ := (total_time / 2) / time_B

/-- Represents the fraction of the tank filled by pipes A and B together in the second half of the total time -/
noncomputable def fraction_filled_AB : ℝ := 1 - fraction_filled_B

theorem pipe_A_fill_time :
  (1 / time_A + 1 / time_B) * (total_time / 2) = fraction_filled_AB →
  time_A = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l27_2716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l27_2774

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical objects into k distinct boxes,
    with each box containing at least one object -/
def distributeNonempty (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

theorem ball_distribution_theorem (n k : ℕ) (hn : n = 8) (hk : k = 4) :
  distributeNonempty n k = 35 ∧
  distribute n k - distributeNonempty n k = 130 ∧
  k * distributeNonempty (n - 1) (k - 1) = 21 ∧
  (distributeNonempty (n - 2) (k - 1) +
   distributeNonempty (n - 3) (k - 1) +
   distributeNonempty (n - 4) (k - 1) +
   distributeNonempty (n - 5) (k - 1)) = 20 := by
  sorry

#eval distribute 8 4
#eval distributeNonempty 8 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l27_2774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l27_2782

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

-- Theorem statement
theorem f_is_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_increasing_l27_2782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l27_2707

/-- Calculates the average speed for a two-part trip -/
noncomputable def averageSpeed (totalDistance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  totalDistance / ((distance1 / speed1) + (distance2 / speed2))

/-- Theorem: The average speed for the given trip is 20 mph -/
theorem trip_average_speed :
  let totalDistance : ℝ := 80
  let distance1 : ℝ := 40
  let speed1 : ℝ := 15
  let distance2 : ℝ := 40
  let speed2 : ℝ := 30
  averageSpeed totalDistance distance1 speed1 distance2 speed2 = 20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_average_speed_l27_2707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l27_2750

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def total_cost : ℝ := book_price + (gov_tax_rate * book_price) + shipping_fee - (seller_discount_rate * book_price)

noncomputable def profit : ℝ := selling_price - total_cost

noncomputable def profit_percentage : ℝ := (profit / total_cost) * 100

theorem book_profit_percentage :
  abs (profit_percentage + 1.27) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l27_2750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alpha_n_less_than_3_16_exists_alpha_all_greater_than_7_40_l27_2718

-- Define the sequence α_n
noncomputable def alpha_sequence (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => min (2 * alpha_sequence α n) (1 - 2 * alpha_sequence α n)

-- Statement 1
theorem exists_alpha_n_less_than_3_16 (α : ℝ) (h_irrational : Irrational α) (h_bound : 0 < α ∧ α < 1/2) :
  ∃ n : ℕ, alpha_sequence α n < 3/16 := by sorry

-- Statement 2
theorem exists_alpha_all_greater_than_7_40 :
  ∃ α : ℝ, Irrational α ∧ 0 < α ∧ α < 1/2 ∧ ∀ n : ℕ, alpha_sequence α n > 7/40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_alpha_n_less_than_3_16_exists_alpha_all_greater_than_7_40_l27_2718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l27_2799

-- Define the propositions
def vertically_opposite_angles_equal : Prop := 
  ∀ (a b c d : Real), a = c ∧ b = d → a = b

def perpendicular_segments_shortest : Prop := 
  ∀ (p q r : Real × Real) (l : Set (Real × Real)), 
    (∃ (v : Real × Real), v ∈ l ∧ (p.1 - v.1) * (q.1 - v.1) + (p.2 - v.2) * (q.2 - v.2) = 0) → 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2)

def unique_perpendicular_line : Prop := 
  ∀ (p : Real × Real) (l : Set (Real × Real)), p ∉ l → 
    ∃! (m : Set (Real × Real)), (∃ (v : Real × Real), v ∈ l ∧ v ∈ m) ∧ 
      (∀ (u v : Real × Real), u ∈ l ∧ v ∈ m → (u.1 - v.1) * (p.1 - v.1) + (u.2 - v.2) * (p.2 - v.2) = 0)

def parallel_lines_supplementary_angles : Prop := 
  ∀ (l1 l2 t : Set (Real × Real)), 
    (∀ (p q : Real × Real), p ∈ l1 ∧ q ∈ l2 → (p.2 - q.2) = 0) → 
    ∀ (a b : Real), (∃ (v : Real × Real), v ∈ t ∧ v ∈ l1) ∧ (∃ (w : Real × Real), w ∈ t ∧ w ∈ l2) → 
      a + b = Real.pi

-- Theorem statement
theorem all_propositions_true : 
  vertically_opposite_angles_equal ∧ 
  perpendicular_segments_shortest ∧ 
  unique_perpendicular_line ∧ 
  parallel_lines_supplementary_angles := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_true_l27_2799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l27_2702

open Real Set

noncomputable def f_A (a b x : ℝ) : ℝ :=
  ((x / a) + (b / x) - 1)^2 - (2 * b / a) + 1

theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  let A := Icc a b
  ∃ (min max : ℝ),
    (∀ x ∈ A, f_A a b x ≥ min) ∧
    (∃ x ∈ A, f_A a b x = min) ∧
    (∀ x ∈ A, f_A a b x ≤ max) ∧
    (∃ x ∈ A, f_A a b x = max) ∧
    min = 2 * (sqrt (b / a) - 1)^2 ∧
    max = ((b / a) - 1)^2 ∧
  (∀ m > 5/2, ∃ (k : ℕ) (x1 x2 : ℝ),
    x1 ∈ Icc (k^2 : ℝ) ((k+1)^2 : ℝ) ∧
    x2 ∈ Icc ((k+1)^2 : ℝ) ((k+2)^2 : ℝ) ∧
    f_A (k^2 : ℝ) ((k+1)^2 : ℝ) x1 + f_A ((k+1)^2 : ℝ) ((k+2)^2 : ℝ) x2 < m) ∧
  (∀ x1 x2 x3, x1 ∈ A → x2 ∈ A → x3 ∈ A →
    (sqrt (f_A a b x1) + sqrt (f_A a b x2) > sqrt (f_A a b x3) ↔
    1 < b / a ∧ b / a < (2 * sqrt 2 - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l27_2702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_Y_l27_2757

/-- A function that checks if a natural number only contains digits 0 and 1 -/
def only_zero_and_one (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The smallest positive integer S composed of only 0s and 1s that is divisible by 18 -/
def S : ℕ := 111111111000

/-- Y is defined as S divided by 18 -/
def Y : ℕ := S / 18

theorem smallest_Y :
  Y = 6172839500 ∧
  (∀ n : ℕ, n > 0 → only_zero_and_one n → n % 18 = 0 → n / 18 ≥ Y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_Y_l27_2757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_volume_and_circumference_l27_2745

/-- Prove that a right cone with a base circumference of 24π inches and a volume of 288π cubic inches has a height of 6 inches. -/
theorem cone_height_from_volume_and_circumference :
  ∀ (h : ℝ), 
  (1 / 3) * Real.pi * (12 ^ 2) * h = 288 * Real.pi →
  h = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_volume_and_circumference_l27_2745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_inequality_l27_2711

-- Define the triangle ABC
variable (A B C : EuclideanPlane)

-- Define the inscribed circle center I
variable (I : EuclideanPlane)

-- Define a point P inside the triangle
variable (P : EuclideanPlane)

-- Define the condition that P is inside the triangle
def is_inside_triangle (P A B C : EuclideanPlane) : Prop := sorry

-- Define the angle measurement function
noncomputable def angle (A B C : EuclideanPlane) : ℝ := sorry

-- Define the distance function
noncomputable def distance (X Y : EuclideanPlane) : ℝ := sorry

-- Define the inscribed circle center function
noncomputable def inscribed_circle_center (A B C : EuclideanPlane) : EuclideanPlane := sorry

-- State the theorem
theorem triangle_inscribed_circle_inequality 
  (h_triangle : is_inside_triangle P A B C)
  (h_I : I = inscribed_circle_center A B C)
  (h_angle : angle P B A + angle P C A ≥ angle P B C + angle P C B) :
  distance A P ≥ distance A I ∧ 
  (distance A P = distance A I ↔ P = I) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_inequality_l27_2711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l27_2792

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The coordinates of the vertices and focus of the ellipse -/
def upper_vertex (e : Ellipse) : ℝ × ℝ := (0, e.b)
def lower_vertex (e : Ellipse) : ℝ × ℝ := (0, -e.b)
def right_vertex (e : Ellipse) : ℝ × ℝ := (e.a, 0)
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- Vector from lower vertex to right focus -/
noncomputable def vector_BF (e : Ellipse) : ℝ × ℝ :=
  ((right_focus e).1 - (lower_vertex e).1, (right_focus e).2 - (lower_vertex e).2)

/-- Vector from upper vertex to right vertex -/
def vector_AC (e : Ellipse) : ℝ × ℝ :=
  ((right_vertex e).1 - (upper_vertex e).1, (right_vertex e).2 - (upper_vertex e).2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem ellipse_eccentricity (e : Ellipse) :
  dot_product (vector_BF e) (vector_AC e) = 0 →
  eccentricity e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l27_2792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_approx_l27_2759

/-- Represents a right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  slantEdge : ℝ

/-- Calculates the volume of a smaller pyramid formed by cutting a right square pyramid -/
noncomputable def smallerPyramidVolume (p : RightSquarePyramid) (cutHeight : ℝ) : ℝ :=
  let fullHeight := Real.sqrt (p.slantEdge ^ 2 - (p.baseEdge / Real.sqrt 2) ^ 2)
  let newHeight := fullHeight - cutHeight
  let ratio := newHeight / fullHeight
  (1 / 3) * (ratio ^ 3) * p.baseEdge ^ 2 * fullHeight

/-- The theorem statement -/
theorem smaller_pyramid_volume_approx :
  let p : RightSquarePyramid := { baseEdge := 10 * Real.sqrt 2, slantEdge := 12 }
  let cutHeight := 4
  abs (smallerPyramidVolume p cutHeight - 67.15) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_approx_l27_2759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l27_2748

theorem cosine_identity (α : ℝ) : 
  (Real.cos α) ^ 2 + (Real.cos (α + 60 * π / 180)) ^ 2 - 
  (Real.cos α) * (Real.cos (α + 60 * π / 180)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l27_2748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_count_l27_2730

/-- The total number of candies in the bag -/
def N : ℕ := 60

/-- The number of candies Carlson ate in the first 10 minutes -/
def initial_eaten : ℚ := (1/5) * N

/-- The number of caramel candies Carlson ate in the first 10 minutes -/
def initial_caramel : ℚ := (1/4) * initial_eaten

/-- The number of chocolate candies Carlson ate in the first 10 minutes -/
def initial_chocolate : ℚ := initial_eaten - initial_caramel

/-- The total number of candies Carlson ate -/
def total_eaten : ℚ := initial_eaten + 3

/-- The proportion of caramels among the candies Carlson had eaten after eating 3 more chocolates -/
def final_caramel_proportion : ℚ := (1/5)

theorem candy_count : N = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_count_l27_2730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_more_than_A_l27_2785

/-- The price of Company KW -/
def P : ℝ := sorry

/-- The assets of Company A -/
def A : ℝ := sorry

/-- The assets of Company B -/
def B : ℝ := sorry

/-- The price of Company KW is 100% more than Company B's assets -/
axiom price_B : P = 2 * B

/-- If Companies A and B merge, the price of Company KW would be 78.78787878787878% of their combined assets -/
axiom price_combined : P = 0.7878787878787878 * (A + B)

/-- Theorem: The price of Company KW is 30% more than Company A's assets -/
theorem price_more_than_A : P = 1.3 * A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_more_than_A_l27_2785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_examples_l27_2725

-- Definition of a closed function
def is_closed_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → (f x < f y ∨ f x > f y)) ∧
  (∃ a b, a < b ∧ Set.range (fun x ↦ f x) = Set.Icc a b)

-- The function y = -x^3
noncomputable def f (x : ℝ) : ℝ := -x^3

-- The function f(x) = x^2/2 - x + 1
noncomputable def g (x : ℝ) : ℝ := x^2/2 - x + 1

-- Theorem statement
theorem closed_function_examples :
  is_closed_function f ∧ ¬is_closed_function g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_function_examples_l27_2725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_standard_parabola_l27_2735

/-- The focus of a parabola y = x^2 -/
noncomputable def focus_of_parabola : ℝ × ℝ := (0, 1/4)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop := y = x^2

/-- Theorem stating that the focus of the standard parabola y = x^2 is (0, 1/4) -/
theorem focus_of_standard_parabola :
  ∀ x y : ℝ, parabola_equation x y →
  focus_of_parabola = (0, 1/4) := by
  sorry

#check focus_of_standard_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_standard_parabola_l27_2735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_750_l27_2768

/-- Represents a trapezoid ABCD with given side lengths and altitude --/
structure Trapezoid where
  ad : ℝ
  ab : ℝ
  bc : ℝ
  altitude : ℝ

/-- Calculates the area of a trapezoid --/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.ab + (t.ad + t.bc)) / 2 * t.altitude

theorem trapezoid_area_is_750 (t : Trapezoid) 
  (h1 : t.ad = 15)
  (h2 : t.ab = 50)
  (h3 : t.bc = 20)
  (h4 : t.altitude = 12) : 
  trapezoid_area t = 750 := by
  sorry

#check trapezoid_area_is_750

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_750_l27_2768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l27_2744

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * Real.log (1 + x) + x^2
  else -x * Real.log (1 - x) + x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l27_2744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l27_2790

theorem triangle_sine_sum_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l27_2790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l27_2738

/-- Parabola type -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * para.p * x

/-- Focus of a parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Main theorem -/
theorem parabola_focus_distance (para : Parabola) 
  (P : PointOnParabola para) (h : P.x = 2) :
  distance (P.x, P.y) (focus para) = 4 → para.p = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l27_2738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l27_2787

-- Define the rectangle dimensions
def width : ℝ := 6
def length : ℝ := 7

-- Define the volumes of the two cylinders
noncomputable def volume1 : ℝ := (width^2 * length) / (4 * Real.pi)
noncomputable def volume2 : ℝ := (length^2 * width) / (4 * Real.pi)

-- Define the ratio of the larger volume to the smaller volume
noncomputable def volumeRatio : ℝ := max volume1 volume2 / min volume1 volume2

-- Theorem statement
theorem rectangle_cylinder_volume_ratio :
  volumeRatio = 7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cylinder_volume_ratio_l27_2787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_e_replacement_l27_2731

def alphabet_size : ℕ := 26

def char_to_index (c : Char) : ℕ :=
  (c.toNat - 'a'.toNat) % alphabet_size

def index_to_char (n : ℕ) : Char :=
  Char.ofNat ((n % alphabet_size) + 'a'.toNat)

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def message : String :=
  "Hello, can everyone check the event, please?"

def count_occurrences (c : Char) (s : String) : ℕ :=
  s.toList.filter (· = c) |>.length

theorem last_e_replacement :
  let e_count := count_occurrences 'e' message
  let total_shift := triangular_number e_count
  let new_index := (char_to_index 'e' + total_shift) % alphabet_size
  index_to_char new_index = 'o' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_e_replacement_l27_2731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l27_2798

/-- The radius of the inscribed circle in an isosceles triangle with two sides of length 8 and base of length 7 -/
theorem inscribed_circle_radius_isosceles_triangle : 
  ∀ (A B C : ℝ × ℝ) (r : ℝ),
  let a := 8
  let b := 8
  let c := 7
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (dist A B = a ∧ dist A C = b ∧ dist B C = c) →
  (area = r * s) →
  r = 23.75 / 11.5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l27_2798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_and_radii_sum_l27_2786

/-- Curve C₁ in polar coordinates -/
noncomputable def C₁ (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ + Real.pi/4)

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) (a : ℝ) : Prop := ρ * Real.sin θ = a

/-- Symmetry condition for C₁ about C₂ -/
def symmetric (a : ℝ) : Prop := ∀ θ, ∃ ρ, C₂ ρ θ a ∧ C₁ (θ - Real.pi/4) = ρ ∧ C₁ (θ + Real.pi/4) = ρ

theorem curve_symmetry_and_radii_sum (a : ℝ) (h₁ : a > 0) (h₂ : symmetric a) :
  a = 1 ∧ ∀ φ, C₁ φ * C₁ (φ - Real.pi/4) + C₁ (φ + Real.pi/4) * C₁ (φ + Real.pi/2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_and_radii_sum_l27_2786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_exponent_sum_l27_2741

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (x y z : ℕ), 
    (Real.rpow (72 * a^5 * b^9 * c^14) (1/3) = k * a^x * b^y * c^z) ∧ 
    (x + y + z = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_root_exponent_sum_l27_2741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_equality_l27_2743

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem irrational_equality (α β : ℝ) : 
  (α > 0) → 
  (β > 0) → 
  (∀ x : ℝ, x > 0 → floor (α * floor (β * x)) = floor (β * floor (α * x))) → 
  (¬ ∃ q : ℚ, α = ↑q) → 
  (¬ ∃ q : ℚ, β = ↑q) → 
  α = β :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_equality_l27_2743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l27_2763

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem johns_average_speed :
  let distance : ℝ := 246
  let time : ℝ := 5.75
  abs (average_speed distance time - 42.7826) < 0.00005 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#eval Float.abs (246 / 5.75 - 42.7826) -- To check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_average_speed_l27_2763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_a_equals_one_h_inequality_l27_2758

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) - x

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := Real.log (x + 1)

-- Theorem 1: Prove that a = 1 given the conditions
theorem unique_root_implies_a_equals_one (a : ℝ) (h₁ : a > 0) 
  (h₂ : ∃! x, f a x = 0) : a = 1 := by sorry

-- Theorem 2: Prove the inequality for h
theorem h_inequality (x₁ x₂ : ℝ) (h₁ : x₁ > -1) (h₂ : x₂ > -1) (h₃ : x₁ ≠ x₂) :
  (x₁ - x₂) / (h x₁ - h x₂) > Real.sqrt (x₁ * x₂ + x₁ + x₂ + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_implies_a_equals_one_h_inequality_l27_2758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_m_value_l27_2719

theorem collinear_vectors_m_value 
  (e₁ e₂ : ℝ × ℝ) 
  (h_non_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁) 
  (a b : ℝ × ℝ) 
  (h_a : a = (3 • e₁ + 5 • e₂)) 
  (h_b : ∃ m : ℝ, b = (m • e₁ - 3 • e₂)) 
  (h_collinear : ∃ l : ℝ, b = l • a) : 
  ∃ m : ℝ, b = (m • e₁ - 3 • e₂) ∧ m = -9/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_m_value_l27_2719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l27_2777

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

-- Define a point on the hyperbola
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Define the distances to asymptotes
noncomputable def distancesToAsymptotes (h : Hyperbola) (p : PointOnHyperbola h) : ℝ × ℝ :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ((|h.b * p.x - h.a * p.y|) / c, (|h.b * p.x + h.a * p.y|) / c)

-- Define the focal distance
noncomputable def focalDistance (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

-- State the theorem
theorem hyperbola_asymptotes (h : Hyperbola) (p : PointOnHyperbola h) :
  let (d₁, d₂) := distancesToAsymptotes h p
  (focalDistance h)^2 = 16 * d₁ * d₂ → h.a = h.b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l27_2777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_314_n_1000_has_314_smallest_n_is_1000_l27_2740

def contains_314 (m n : ℕ) : Prop :=
  ∃ k : ℕ, (1000 * m) / n = 314 + k ∧ k < n

theorem smallest_n_for_314 :
  ∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ contains_314 m n) →
    n ≥ 1000 :=
by sorry

theorem n_1000_has_314 :
  ∃ m : ℕ, m < 1000 ∧ Nat.Coprime m 1000 ∧ contains_314 m 1000 :=
by sorry

theorem smallest_n_is_1000 :
  (∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m < n ∧ Nat.Coprime m n ∧ contains_314 m n) →
    n ≥ 1000) ∧
  (∃ m : ℕ, m < 1000 ∧ Nat.Coprime m 1000 ∧ contains_314 m 1000) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_314_n_1000_has_314_smallest_n_is_1000_l27_2740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_circle_diameter_special_rectangle_sum_l27_2727

/-- A rectangle with 3-4-5 right triangles in corners and a tangent circle -/
structure SpecialRectangle where
  width : ℝ
  height : ℝ
  circle_diameter : ℝ
  width_eq : width = 8
  height_eq : height = 7
  tangent_circle : circle_diameter > 0

/-- The diameter of the circle in the special rectangle is 40/7 -/
theorem special_rectangle_circle_diameter (r : SpecialRectangle) : 
  r.circle_diameter = 40 / 7 := by
  sorry

/-- The sum of numerator and denominator of the circle diameter fraction is 47 -/
theorem special_rectangle_sum (r : SpecialRectangle) : 
  ∃ (m n : ℕ), r.circle_diameter = ↑m / ↑n ∧ Nat.Coprime m n ∧ m + n = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_circle_diameter_special_rectangle_sum_l27_2727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_rationals_characterization_l27_2723

theorem positive_rationals_characterization (X : Set ℚ) 
  (closed_add : ∀ (a b : ℚ), a ∈ X → b ∈ X → a + b ∈ X)
  (closed_mul : ∀ (a b : ℚ), a ∈ X → b ∈ X → a * b ∈ X)
  (zero_not_in : 0 ∉ X)
  (exactly_one : ∀ (x : ℚ), x ≠ 0 → (x ∈ X ↔ -x ∉ X)) :
  X = {x : ℚ | 0 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_rationals_characterization_l27_2723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l27_2737

-- Define the geometric sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum of the first n terms of a_n
def S : ℕ → ℝ := sorry

-- Define the sequence b_n
def b : ℕ → ℝ := sorry

-- Define the sum of the first n terms of b_n
def T : ℕ → ℝ := sorry

theorem geometric_sequence_properties :
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) →  -- S_n is the sum of first n terms
  (S 3 = a 4 - a 1) →  -- Given condition
  (2 * (a 3 + 1) = a 1 + a 4) →  -- a_1, a_3 + 1, a_4 form arithmetic sequence
  (∀ n : ℕ, a n * b n = n * 2^(n+1) + 1) →  -- Relationship between a_n and b_n
  (∀ n : ℕ, a n = 2^n) ∧  -- First conclusion
  (∀ n : ℕ, T n = n * (n + 1) + 1 - 1 / 2^n) :=  -- Second conclusion
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l27_2737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l27_2795

noncomputable def f (x : ℝ) : ℝ := 4 / (3 * x^4 - 7)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f]
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l27_2795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_triangles_have_three_equal_medians_l27_2717

/-- A triangle has three equal medians -/
def has_three_equal_medians (T : Type) : Prop := sorry

/-- The negation of "Some triangles have three equal medians" is equivalent to "All triangles do not have three equal medians" -/
theorem negation_of_some_triangles_have_three_equal_medians :
  (¬ ∃ (T : Type), has_three_equal_medians T) ↔
  (∀ (T : Type), ¬ has_three_equal_medians T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_triangles_have_three_equal_medians_l27_2717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_l27_2796

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a + 1) * Real.exp x

/-- The condition that f(x) + a > 0 for all x > 0 -/
def condition (a : ℝ) : Prop := ∀ x > 0, f a x + a > 0

/-- The theorem stating that 3 is the maximum integer value of a satisfying the condition -/
theorem max_integer_a : ∃ (n : ℤ), (n : ℝ) = 3 ∧ condition n ∧ ∀ m : ℤ, (m : ℝ) > 3 → ¬condition m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_a_l27_2796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_measure_l27_2773

-- Define the angles
variable (angle_ABC angle_ABD angle_CBD : ℝ)

-- State the theorem
theorem angle_ABC_measure :
  angle_CBD = 90 →
  angle_ABD = 110 →
  angle_ABC + angle_ABD + angle_CBD = 270 →
  angle_ABC = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABC_measure_l27_2773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l27_2756

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + x + 2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-1/7 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l27_2756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_and_triangle_perimeter_l27_2728

noncomputable section

/-- Area of a rectangle given its four vertices --/
def area_rectangle (A B C D : Fin 2 → ℝ) : ℝ := sorry

/-- Perimeter of a triangle given its three vertices --/
def perimeter_triangle (B C E : Fin 2 → ℝ) : ℝ := sorry

theorem rectangle_area_and_triangle_perimeter 
  (A B C D E : Fin 2 → ℝ)
  (hA : A = ![0, 0])
  (hB : B = ![Real.sqrt 2, 0])
  (hC : C = ![Real.sqrt 2, 1])
  (hD : D = ![0, 1])
  (hE : E = ![Real.sqrt 2, Real.sqrt 2]) :
  (area_rectangle A B C D = Real.sqrt 2) ∧ 
  (perimeter_triangle B C E = 1 + Real.sqrt (4 - 2 * Real.sqrt 2) + (Real.sqrt 2 - 1)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_and_triangle_perimeter_l27_2728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_smaller_divides_larger_l27_2734

def S : Finset ℕ := {1, 2, 3, 6}

def divisible_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (fun p => p.1 < p.2 ∧ p.2 % p.1 = 0)

def total_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (fun p => p.1 < p.2)

theorem probability_smaller_divides_larger :
  (divisible_pairs S).card / (total_pairs S).card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_smaller_divides_larger_l27_2734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_parabola_directrix_example_l27_2776

/-- Definition of the directrix of a parabola -/
noncomputable def directrix (f : ℝ → ℝ) : ℝ := sorry

/-- The directrix of a parabola y = ax² is given by y = -1/(4a) when a ≠ 0 -/
theorem parabola_directrix (a : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2
  directrix f = -1 / (4 * a) := by sorry

/-- For the parabola y = -6x², its directrix is y = 1/24 -/
theorem parabola_directrix_example :
  let f : ℝ → ℝ := λ x ↦ -6 * x^2
  directrix f = 1/24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_parabola_directrix_example_l27_2776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hour_hand_rotation_6_to_9_l27_2706

/-- The rotation of the hour hand in radians from 6 o'clock to 9 o'clock -/
noncomputable def hour_hand_rotation : ℝ := -Real.pi/2

/-- Theorem stating that the rotation of the hour hand from 6 o'clock to 9 o'clock is -π/2 radians -/
theorem hour_hand_rotation_6_to_9 : hour_hand_rotation = -Real.pi/2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hour_hand_rotation_6_to_9_l27_2706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_even_function_l27_2715

theorem sine_shift_even_function (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, Real.sin (2 * (x + π / 6) + φ) = Real.sin (2 * (-x + π / 6) + φ)) → φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_shift_even_function_l27_2715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_geometric_sequence_implies_alpha_l27_2704

theorem sine_geometric_sequence_implies_alpha (α : Real) :
  α ∈ Set.Ioo 0 (π/2) ∪ Set.Ioo (π/2) π →
  (∃ r : Real, Real.sin α * r = Real.sin (2 * α) ∧ Real.sin (2 * α) * r = Real.sin (4 * α)) →
  α = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_geometric_sequence_implies_alpha_l27_2704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sums_achievable_l27_2732

/-- Represents the sum of digits on three faces adjacent to a vertex -/
def VertexSum : Type := Fin 8

/-- The set of possible sums for each vertex -/
def ValidSums : Finset Nat :=
  {6, 7, 9, 10, 11, 12, 14, 15}

/-- A configuration of the cube is represented by 8 vertex sums -/
def CubeConfiguration := Fin 8 → VertexSum

/-- The sum of all digits on the faces of the cube -/
def TotalSum (config : CubeConfiguration) : Nat :=
  (Finset.sum Finset.univ fun i => (config i).val)

/-- Theorem stating that any integer in [48, 120] can be achieved as a total sum -/
theorem all_sums_achievable :
  ∀ s : Nat, 48 ≤ s ∧ s ≤ 120 →
  ∃ config : CubeConfiguration, (∀ i, (config i).val ∈ ValidSums) ∧ TotalSum config = s := by
  sorry

#check all_sums_achievable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sums_achievable_l27_2732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_perpendicular_lines_l27_2766

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℝ × ℝ)

-- Define the property that points are on the circle
def on_circle (p : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop := p ∈ circle

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the property of equal distances
def equal_distances (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : Prop :=
  distance A₁ A₂ = distance A₂ A₃ ∧
  distance A₂ A₃ = distance A₃ A₄ ∧
  distance A₃ A₄ = distance A₄ A₅

-- Define the property of diametrically opposite points
def diametrically_opposite (p q : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ), on_circle center circle ∧
    distance center p = distance center q ∧
    distance p q = 2 * distance center p

-- Define the intersection of two lines
noncomputable def intersection (p₁ q₁ p₂ q₂ : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of intersection point is complex and not necessary for this statement

-- Define perpendicularity of two lines
def perpendicular (p₁ q₁ p₂ q₂ : ℝ × ℝ) : Prop :=
  (q₁.1 - p₁.1) * (q₂.1 - p₂.1) + (q₁.2 - p₁.2) * (q₂.2 - p₂.2) = 0

-- The theorem statement
theorem circle_points_perpendicular_lines :
  on_circle A₁ circle ∧ on_circle A₂ circle ∧ on_circle A₃ circle ∧ 
  on_circle A₄ circle ∧ on_circle A₅ circle ∧
  equal_distances A₁ A₂ A₃ A₄ A₅ ∧
  diametrically_opposite A₂ A₆ circle ∧
  A₇ = intersection A₁ A₅ A₃ A₆ →
  perpendicular A₁ A₆ A₄ A₇ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_points_perpendicular_lines_l27_2766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l27_2775

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x
def g (x : ℝ) : ℝ := -2 * x + 3

theorem min_t_value (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ -1) :
  ∃ t : ℝ, t = 11/4 ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 →
    |f a x₁ - f a x₂| ≤ t * |g x₁ - g x₂|) ∧
  (∀ t' : ℝ, t' < t →
    ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 ∧
      |f a x₁ - f a x₂| > t' * |g x₁ - g x₂|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_value_l27_2775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_theorem_l27_2767

-- Define the curve in polar coordinates
noncomputable def curve (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the line in polar coordinates
def line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/3) = 4

-- Define the minimum distance function
noncomputable def min_distance : ℝ := 5/2

-- Theorem statement
theorem minimum_distance_theorem :
  ∀ (θ : ℝ), ∃ (d : ℝ),
    d ≥ min_distance ∧
    ∀ (ρ : ℝ), ρ = curve θ →
      ∀ (θ' : ℝ), line ρ θ' →
        d ≤ Real.sqrt ((ρ * Real.cos θ' - ρ * Real.cos θ)^2 + (ρ * Real.sin θ' - ρ * Real.sin θ)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_theorem_l27_2767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l27_2724

open Real

theorem relationship_xyz (α b x y z : ℝ) 
  (h_α : α ∈ Set.Ioo (π/4) (π/2))
  (h_b : b ∈ Set.Ioo 0 1)
  (h_x : log x = (log (sin α))^2 / log b)
  (h_y : log y = (log (cos α))^2 / log b)
  (h_z : log z = (log (sin α * cos α))^2 / log b) :
  x > y ∧ y > z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_xyz_l27_2724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_equals_4_l27_2791

theorem sin_2x_equals_4 (x : ℝ) 
  (h : Real.sin x + Real.cos x + 2 * Real.tan x + 2 * (1 / Real.tan x) + (1 / Real.cos x) + (1 / Real.sin x) = 9) : 
  Real.sin (2 * x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_equals_4_l27_2791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_helium_production_l27_2793

noncomputable def total_cost (x : ℝ) : ℝ :=
  if 4/3 ≤ x ∧ x < 4 then
    40 * (4*x + 16/x + 100)
  else if 4 ≤ x ∧ x ≤ 8 then
    40 * (9/x^2 - 3/x + 117)
  else
    0  -- Invalid range

theorem minimum_cost_helium_production :
  ∃ (x : ℝ), 4/3 ≤ x ∧ x ≤ 8 ∧
    (∀ (y : ℝ), 4/3 ≤ y ∧ y ≤ 8 → total_cost x ≤ total_cost y) ∧
    x = 2 ∧ total_cost x = 4640 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_helium_production_l27_2793
