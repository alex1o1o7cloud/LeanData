import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_consecutive_digits_l378_37861

def consecutive_digits (m n : ℕ) : Prop :=
  ∃ k, (1000 * (10^k * m % n)) / n = 347

theorem smallest_n_with_consecutive_digits : 
  (∃ n : ℕ, n > 0 ∧ 
    (∃ m : ℕ, m < n ∧ 
      Nat.Coprime m n ∧ 
      consecutive_digits m n)) ∧ 
  (∀ n : ℕ, n > 0 → 
    (∃ m : ℕ, m < n ∧ 
      Nat.Coprime m n ∧ 
      consecutive_digits m n) → 
    n ≥ 347) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_consecutive_digits_l378_37861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_R_l378_37801

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (10, 0)

-- Define the vertical line intersecting AC at R and BC at S
noncomputable def R : ℝ × ℝ := sorry
noncomputable def S : ℝ × ℝ := sorry

-- Define the area of triangle RSC
def area_RSC : ℝ := 15

-- State the theorem
theorem coordinate_difference_of_R :
  let x_R := R.1
  let y_R := R.2
  (S.2 = 0) →  -- S is on the x-axis
  (R.1 = S.1) →  -- R and S form a vertical line
  (area_RSC = 15) →
  (y_R - x_R = 2 * Real.sqrt 30 - 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_difference_of_R_l378_37801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_equivalence_l378_37878

open Real

theorem trigonometric_function_equivalence 
  (f : ℝ → ℝ) (ω φ : ℝ) 
  (h_f : ∀ x, f x = sin (2 * ω * x + φ) + cos (2 * ω * x + φ))
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_period : ∀ x, f (x + π) = f x)
  (h_odd : ∀ x, f (-x) = -f x) :
  ∀ x, f x = - Real.sqrt 2 * sin (2 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_function_equivalence_l378_37878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_decomposition_integer_edge_l378_37873

/-- Represents a rectangle in a 2D plane --/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- Represents a decomposition of a rectangle into sub-rectangles --/
structure RectangleDecomposition where
  main : Rectangle
  subs : List Rectangle
  mutually_exclusive : ∀ (i j : Fin subs.length), i ≠ j → 
    (subs.get i).width * (subs.get i).height + 
    (subs.get j).width * (subs.get j).height = 
    (subs.get i).width * (subs.get i).height
  parallel_edges : ∀ (i : Fin subs.length), 
    (subs.get i).width ≤ main.width ∧ (subs.get i).height ≤ main.height
  cover_main : main.width * main.height = 
    (subs.map (λ r => r.width * r.height)).sum
  integer_edge : ∀ (i : Fin subs.length), 
    (subs.get i).width.den = 1 ∨ (subs.get i).height.den = 1

theorem rectangle_decomposition_integer_edge 
  (d : RectangleDecomposition) : 
  d.main.width.den = 1 ∨ d.main.height.den = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_decomposition_integer_edge_l378_37873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sequence_difference_l378_37887

noncomputable section

open Real

theorem geometric_arithmetic_sequence_difference
  (p q r : ℝ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_geometric : ∃ (k : ℝ), q = p * k ∧ r = q * k)
  (h_arithmetic : ∃ (d : ℝ), logb r p + d = logb q r ∧ logb q r + d = logb p q) :
  ∃ (d : ℝ), d = 3/2 ∧ logb r p + d = logb q r ∧ logb q r + d = logb p q :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sequence_difference_l378_37887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l378_37851

-- Define the space
variable (S : Type)

-- Define lines and planes in the space
variable (m n : S → S → Prop)
variable (α β : S → Prop)

-- Define the perpendicular and parallel relations
variable (perp : (S → S → Prop) → (S → S → Prop) → Prop)
variable (para : (S → S → Prop) → (S → S → Prop) → Prop)
variable (perp_plane : (S → S → Prop) → (S → Prop) → Prop)
variable (para_plane : (S → S → Prop) → (S → Prop) → Prop)
variable (perp_planes : (S → Prop) → (S → Prop) → Prop)

-- Define propositions p and q
def p : Prop := ∀ (S : Type) (m n : S → S → Prop) (α : S → Prop) 
  (perp : (S → S → Prop) → (S → S → Prop) → Prop)
  (para : (S → S → Prop) → (S → Prop) → Prop)
  (perp_plane : (S → S → Prop) → (S → Prop) → Prop),
  (perp m n ∧ perp_plane m α) → para m α

def q : Prop := ∀ (S : Type) (m : S → S → Prop) (α β : S → Prop)
  (perp_plane : (S → S → Prop) → (S → Prop) → Prop)
  (para_plane : (S → S → Prop) → (S → Prop) → Prop)
  (perp_planes : (S → Prop) → (S → Prop) → Prop),
  (perp_plane m α ∧ para_plane m β) → perp_planes α β

-- State the theorem
theorem problem_statement : 
  m ≠ n → α ≠ β → p ∨ q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l378_37851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_path_implies_tree_l378_37814

-- Define a graph structure
structure Graph (V : Type) where
  edges : V → V → Prop

-- Define a path in a graph
def PathInGraph {V : Type} (G : Graph V) (start finish : V) : List V → Prop
  | [] => start = finish
  | [v] => start = v ∧ v = finish
  | (v :: w :: rest) => G.edges v w ∧ PathInGraph G w finish (w :: rest)

-- Define the property of having exactly one path between any two vertices
def HasUniquePath (V : Type) (G : Graph V) : Prop :=
  ∀ (u v : V), ∃! (p : List V), PathInGraph G u v p

-- Define a tree
def IsTree (V : Type) (G : Graph V) : Prop :=
  (∀ (u v : V), ∃ (p : List V), PathInGraph G u v p) ∧
  (∀ (u v : V) (p q : List V), PathInGraph G u v p → PathInGraph G u v q → p = q)

-- State the theorem
theorem unique_path_implies_tree {V : Type} (G : Graph V) :
  HasUniquePath V G → IsTree V G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_path_implies_tree_l378_37814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_101st_term_l378_37812

/-- An arithmetic sequence with first term 2 and common difference 1/2 -/
def arithmeticSequence : ℕ → ℚ
| 0 => 2
| n + 1 => arithmeticSequence n + 1/2

/-- The 101st term of the arithmetic sequence is 52 -/
theorem arithmetic_sequence_101st_term :
  arithmeticSequence 100 = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_101st_term_l378_37812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l378_37804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - (1/4) * x

theorem function_extrema_and_monotonicity 
  (a : ℝ) 
  (h_a : a > 0) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f (1/2) x ≤ 0) ∧ 
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f (1/2) x ≥ Real.log 2 - 1) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), MonotoneOn (g a) (Set.Icc 1 (Real.exp 1)) ↔ a ≥ 4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l378_37804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_point_l378_37846

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope k and y-intercept m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The number of intersection points between a line and an ellipse -/
noncomputable def intersection_count (e : Ellipse) (l : Line) : ℕ := sorry

/-- The main theorem stating that a line satisfying the given conditions
    intersects the ellipse at exactly one point -/
theorem one_intersection_point (e : Ellipse) (l : Line) :
  e.a = 2 ∧ 
  e.b = 1 ∧ 
  (∃ c : ℝ, c = Real.sqrt 3 ∧ c^2 + e.b^2 = e.a^2) ∧
  (∃ A B : ℝ × ℝ, 
    A.1 = 2 ∧ B.1 = -2 ∧
    A.2 = l.k * A.1 + l.m ∧ 
    B.2 = l.k * B.1 + l.m ∧
    (A.1 - Real.sqrt 3) * (B.1 - Real.sqrt 3) + A.2 * B.2 = 0) →
  intersection_count e l = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_intersection_point_l378_37846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_functional_equation_l378_37816

theorem no_solution_functional_equation :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (f x + 2 * y) = 3 * x + f (f (f y) - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_functional_equation_l378_37816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l378_37810

def fourth_quadrant (α : Real) : Prop :=
  Real.cos α > 0 ∧ Real.sin α < 0

theorem angle_properties (α : Real) 
  (h1 : fourth_quadrant α) 
  (h2 : Real.cos α = 3/5) : 
  Real.tan α = -4/3 ∧ 
  (Real.sin (3/2 * Real.pi - α) + 2 * Real.cos (α + Real.pi/2)) / 
  (Real.sin (α - Real.pi) - 3 * Real.cos (2 * Real.pi - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l378_37810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_YX_equals_XY_l378_37848

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]

def matrix_equality (X Y : Matrix n n ℚ) : Prop :=
  X + Y = X * Y - 1

theorem YX_equals_XY 
  (X Y : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : matrix_equality X Y)
  (h2 : X * Y = ![![17/3, 7/3], ![-5/3, 10/3]]) :
  Y * X = X * Y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_YX_equals_XY_l378_37848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_moles_required_l378_37862

/-- Represents the number of moles of a substance -/
def Moles : Type := ℕ

instance : OfNat Moles n where
  ofNat := n

/-- Represents a chemical reaction between Mg and CO2 to form MgO and C -/
structure ChemicalReaction where
  mg_reactant : Moles
  co2_reactant : Moles
  mgo_product : Moles
  c_product : Moles

/-- A balanced chemical reaction satisfies this property -/
def is_balanced (reaction : ChemicalReaction) : Prop :=
  reaction.mg_reactant = 2 ∧
  reaction.mgo_product = 2 ∧
  reaction.c_product = 1

theorem co2_moles_required (reaction : ChemicalReaction) 
  (h : is_balanced reaction) : reaction.co2_reactant = 1 := by
  sorry

#check co2_moles_required

end NUMINAMATH_CALUDE_ERRORFEEDBACK_co2_moles_required_l378_37862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2_minus_x_squared_l378_37850

open Set
open MeasureTheory
open Interval
open Real

theorem integral_sqrt_2_minus_x_squared : 
  ∫ x in (-sqrt 2)..sqrt 2, sqrt (2 - x^2) = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_2_minus_x_squared_l378_37850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l378_37881

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else x - 1

-- State the theorem
theorem three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x - a^2 + 2*a = 0 ∧
    f y - a^2 + 2*a = 0 ∧
    f z - a^2 + 2*a = 0) ↔
  (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_roots_l378_37881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l378_37803

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- A fixed point P -/
def P : ℝ × ℝ := (3, 1)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_min_distance :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (M : ℝ × ℝ), M ∈ Parabola →
    distance M P + distance M F ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l378_37803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l378_37806

-- Define the conditions
def condition_p (x : ℝ) : Prop := (1/4 : ℝ) < Real.rpow 2 x ∧ Real.rpow 2 x < 16

def condition_q (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the theorem
theorem a_range (a : ℝ) :
  (∀ x, condition_p x → condition_q x a) ∧
  (∃ x, ¬condition_p x ∧ condition_q x a) →
  a < -4 ∧ ∀ b, b < -4 → b ≤ a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l378_37806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_radius_l378_37856

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the tangent point
def tangent_point : ℝ × ℝ := (-1, 2)

-- Define the radius of the sought circle
noncomputable def sought_radius : ℝ := 2 * Real.sqrt 5

-- Define the equation of the sought circle
def sought_circle (x y : ℝ) : Prop := (x + 3)^2 + (y - 6)^2 = 20

-- Theorem statement
theorem circle_tangent_and_radius :
  -- The sought circle is externally tangent to the given circle at the tangent point
  (∀ x y : ℝ, given_circle x y ∧ sought_circle x y → (x, y) = tangent_point) ∧
  -- The sought circle has the specified radius
  (∀ x y : ℝ, sought_circle x y → 
    ((x - tangent_point.1)^2 + (y - tangent_point.2)^2) = sought_radius^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_radius_l378_37856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_properties_l378_37854

noncomputable def geometric_progression (n : ℕ) : ℝ :=
  (1/3) * (2.5^(n-1))

theorem geometric_progression_properties :
  ∀ n : ℕ,
  n > 0 →
  (geometric_progression n = (1/3) * (2.5^(n-1))) ∧
  (n > 1 → geometric_progression n = geometric_progression (n-1) * 2.5) ∧
  (geometric_progression 1 = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_properties_l378_37854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l378_37877

/-- Represents a block in the pyramid --/
structure Block where
  layer : Nat
  value : Nat

/-- Represents the pyramid structure --/
structure Pyramid where
  blocks : List Block
  bottomLayer : List Nat

/-- Checks if a pyramid is valid according to the problem conditions --/
def isValidPyramid (p : Pyramid) : Prop :=
  p.blocks.length = 30 ∧
  (p.blocks.filter (fun b => b.layer = 1)).length = 16 ∧
  (p.blocks.filter (fun b => b.layer = 2)).length = 9 ∧
  (p.blocks.filter (fun b => b.layer = 3)).length = 4 ∧
  (p.blocks.filter (fun b => b.layer = 4)).length = 1 ∧
  p.bottomLayer.length = 16 ∧
  p.bottomLayer = List.range 16

/-- Checks if the value of each block above the bottom layer is the sum of the four blocks below it --/
def hasValidBlockValues (p : Pyramid) : Prop :=
  ∀ b ∈ p.blocks, b.layer > 1 →
    ∃ b1 b2 b3 b4, b1 ∈ p.blocks ∧ b2 ∈ p.blocks ∧ b3 ∈ p.blocks ∧ b4 ∈ p.blocks ∧
      b1.layer = b.layer - 1 ∧
      b2.layer = b.layer - 1 ∧
      b3.layer = b.layer - 1 ∧
      b4.layer = b.layer - 1 ∧
      b.value = b1.value + b2.value + b3.value + b4.value

/-- The theorem stating that the minimum value of the top block is 40 --/
theorem min_top_block_value (p : Pyramid) :
  isValidPyramid p → hasValidBlockValues p →
  ∃ topBlock ∈ p.blocks, topBlock.layer = 4 ∧ topBlock.value ≥ 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_top_block_value_l378_37877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l378_37822

theorem min_value_sin_cos (x : ℝ) : Real.sin x ^ 4 + 2 * Real.cos x ^ 4 ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_l378_37822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_divisible_by_24_has_4_digits_l378_37889

theorem largest_number_divisible_by_24_has_4_digits : 
  let n : ℕ := 9984
  -- n is exactly divisible by 24
  (n % 24 = 0) →
  -- n is the largest number with its number of digits that is divisible by 24
  (∀ m : ℕ, m % 24 = 0 → Nat.digits 10 m = Nat.digits 10 n → m ≤ n) →
  -- n has 4 digits
  Nat.digits 10 n = [4, 8, 9, 9] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_divisible_by_24_has_4_digits_l378_37889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deng_facilitated_sez_l378_37897

-- Define the year of Deng Xiaoping's meeting
def meeting_year : ℕ := 1979

-- Define Deng Xiaoping's thoughts
def deng_thoughts : Prop := ∃ x : String, x = "We want developed, productive, and prosperous socialism"

-- Define Deng Xiaoping's proposition
def deng_proposition : Prop := ∃ x : String, x = "Socialism can also engage in a market economy"

-- Define the establishment of Special Economic Zones
def special_economic_zones : Prop := ∃ x : String, x = "Establishment and development of China's Special Economic Zones"

-- Theorem to prove
theorem deng_facilitated_sez (thoughts : deng_thoughts) (proposition : deng_proposition) :
  special_economic_zones := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deng_facilitated_sez_l378_37897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l378_37882

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := (1 + 3 * x^2) / (3 + x^2)

/-- The point of tangency -/
def x₀ : ℝ := 1

/-- Theorem: The tangent line to the curve f at x₀ is y = x -/
theorem tangent_line_at_x₀ :
  ∃ m b : ℝ, (∀ x : ℝ, m * x + b = f x₀ + (deriv f x₀) * (x - x₀)) ∧ m = 1 ∧ b = 0 := by
  sorry

#check tangent_line_at_x₀

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l378_37882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_256_equals_2_to_m_l378_37866

theorem cube_root_256_equals_2_to_m (m : ℝ) : (256 : ℝ) ^ (1/3 : ℝ) = 2 ^ m → m = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_256_equals_2_to_m_l378_37866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_score_calculation_l378_37894

noncomputable def interview_score : ℝ := 80
noncomputable def written_score : ℝ := 90
noncomputable def interview_weight : ℝ := 3
noncomputable def written_weight : ℝ := 2

noncomputable def weighted_average (x y w1 w2 : ℝ) : ℝ :=
  (x * w1 + y * w2) / (w1 + w2)

theorem final_score_calculation :
  weighted_average interview_score written_score interview_weight written_weight = 84 := by
  -- Unfold the definition of weighted_average
  unfold weighted_average
  -- Simplify the expression
  simp [interview_score, written_score, interview_weight, written_weight]
  -- The proof is completed by normalization
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_score_calculation_l378_37894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tyler_cd_count_l378_37823

theorem tyler_cd_count 
  (initial : ℕ)
  (given_to_sam : ℕ)
  (bought_first : ℕ)
  (given_to_jenny : ℕ)
  (bought_second : ℕ)
  (h1 : initial = 21)
  (h2 : given_to_sam = initial / 3)
  (h3 : bought_first = 8)
  (h4 : given_to_jenny = 2)
  (h5 : bought_second = 12)
  : initial - given_to_sam + bought_first - given_to_jenny + bought_second = 32 := by
  sorry

#check tyler_cd_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tyler_cd_count_l378_37823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_extreme_points_sum_exp_factorial_inequality_l378_37886

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + a * x) - (2 * x) / (x + 2)

-- Statement 1
theorem f_min_value (x : ℝ) : 
  x > -2 → f (1/2) x ≥ Real.log 2 - 1 := by sorry

-- Statement 2
theorem f_extreme_points_sum (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 1/2 → a < 1 → 
  (∀ y, f a y ≤ f a x₁) → (∀ y, f a y ≤ f a x₂) → 
  f a x₁ + f a x₂ > f a 0 := by sorry

-- Statement 3
theorem exp_factorial_inequality (n : ℕ) : 
  n ≥ 2 → Real.exp (n * (n - 1) / 2 : ℝ) > Nat.factorial n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_extreme_points_sum_exp_factorial_inequality_l378_37886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l378_37832

theorem min_value_of_expression (x : ℝ) (hx : x > 0) :
  4 * x^5 + 5 * x^(-4 : ℝ) ≥ 9 ∧ ∃ y : ℝ, y > 0 ∧ 4 * y^5 + 5 * y^(-4 : ℝ) = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l378_37832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l378_37852

theorem inequality_and_equality_condition
  (p q a b c d e : ℝ)
  (hp_pos : 0 < p)
  (hpq : p ≤ q)
  (ha : p ≤ a ∧ a ≤ q)
  (hb : p ≤ b ∧ b ≤ q)
  (hc : p ≤ c ∧ c ≤ q)
  (hd : p ≤ d ∧ d ≤ q)
  (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ∧
  (∃ (x : Fin 5 → ℝ), (∀ i, x i = p ∨ x i = q) ∧
    ((Finset.filter (λ i => x i = p) Finset.univ).card = 2 ∨ (Finset.filter (λ i => x i = p) Finset.univ).card = 3) ∧
    (Finset.sum Finset.univ x) = a + b + c + d + e ∧
    (Finset.sum Finset.univ (λ i => 1 / (x i))) = 1/a + 1/b + 1/c + 1/d + 1/e) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_equality_condition_l378_37852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_equilateral_triangle_l378_37835

/-- The radius of a circle inscribed in an equilateral triangle with side length 8 units -/
noncomputable def inscribedCircleRadius (side : ℝ) (h : side = 8) : ℝ :=
  4 * Real.sqrt 3 / 3

/-- Theorem: The radius of a circle inscribed in an equilateral triangle with side length 8 units is 4√3/3 -/
theorem inscribed_circle_radius_equilateral_triangle :
  ∀ (side : ℝ) (h : side = 8),
  inscribedCircleRadius side h = 4 * Real.sqrt 3 / 3 :=
by
  intro side h
  unfold inscribedCircleRadius
  rfl

#check inscribed_circle_radius_equilateral_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_equilateral_triangle_l378_37835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_is_zero_l378_37838

noncomputable def my_sequence (n : ℝ) : ℝ := n * (Real.sqrt (n^4 + 3) - Real.sqrt (n^4 - 2))

theorem my_sequence_limit_is_zero :
  Filter.Tendsto my_sequence Filter.atTop (nhds 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_is_zero_l378_37838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l378_37829

theorem quadratic_root_form (a b c m n p : ℤ) : 
  a > 0 ∧ b < 0 ∧ c > 0 ∧ 
  (∀ x : ℚ, a * x^2 + b * x + c = 0 ↔ ∃ (s : Int), s = 1 ∨ s = -1 ∧ x = (m + s * Int.sqrt n) / p) ∧
  m > 0 ∧ n > 0 ∧ p > 0 ∧ 
  Int.gcd m (Int.gcd n p) = 1 ∧
  a = 3 ∧ b = -8 ∧ c = 1 →
  n = 13 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l378_37829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_eleven_remainders_l378_37896

theorem at_least_eleven_remainders (A : Fin 100 → ℕ) (h_perm : Function.Bijective A) :
  ∃ (S : Finset ℕ), S.card ≥ 11 ∧ 
    S = Finset.image (λ k ↦ (Finset.sum (Finset.range k) (λ i ↦ A i)) % 100) (Finset.range 101) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_eleven_remainders_l378_37896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l378_37843

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y a b : ℝ) : Prop := b*x + a*y = a*b

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := (a + b - a*b)^2 = a^2 + b^2

-- Main theorem
theorem circle_tangent_line (a b : ℝ) 
  (ha : a > 2) (hb : b > 2) 
  (h_tangent : is_tangent a b) :
  -- Part 1
  (a - 2) * (b - 2) = 2 ∧
  -- Part 2
  (∀ x y : ℝ, x > 1 → y > 1 → x = a / 2 → y = b / 2 → (x - 1) * (y - 1) = 1 / 2) ∧
  -- Part 3
  (∃ min_area : ℝ, min_area = 3 + 2 * Real.sqrt 2 ∧
    ∀ area : ℝ, area = a * b / 2 → area ≥ min_area) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l378_37843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_segment_l378_37875

/-- The perpendicular bisector of a line segment is a line that is perpendicular to the segment and passes through its midpoint. -/
def is_perpendicular_bisector (l : Set (ℝ × ℝ)) (a b : ℝ × ℝ) : Prop :=
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ y - midpoint.2 = m * (x - midpoint.1)) ∧
    m * ((b.1 - a.1) / (b.2 - a.2)) = -1

/-- The theorem states that the line x - y - 2 = 0 is the perpendicular bisector of the line segment with endpoints (1,3) and (5,-1). -/
theorem perpendicular_bisector_of_segment : 
  is_perpendicular_bisector {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} (1, 3) (5, -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_segment_l378_37875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_or_coinciding_l378_37818

noncomputable def v₁ : Fin 3 → ℝ := ![1, 2, 3]
noncomputable def v₂ : Fin 3 → ℝ := ![-1/2, -1, -3/2]

def are_parallel_or_coinciding (v₁ v₂ : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ i : Fin 3, v₁ i = k * v₂ i

theorem lines_parallel_or_coinciding :
  are_parallel_or_coinciding v₁ v₂ := by
  use -2
  constructor
  · exact (by norm_num : (-2 : ℝ) ≠ 0)
  · intro i
    fin_cases i <;> simp [v₁, v₂]
    all_goals norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_parallel_or_coinciding_l378_37818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fund_growth_and_prize_money_l378_37893

noncomputable def initial_fund : ℝ := 21000
noncomputable def interest_rate : ℝ := 0.0624
noncomputable def growth_rate : ℝ := 1 + interest_rate / 2

noncomputable def fund_amount (n : ℕ) : ℝ :=
  initial_fund * growth_rate ^ (n - 1)

noncomputable def total_prize_money (start_year end_year : ℕ) : ℝ :=
  (interest_rate / 2) * (Finset.sum (Finset.range (end_year - start_year + 1)) (λ i => fund_amount (start_year + i)))

theorem fund_growth_and_prize_money :
  (∀ n : ℕ, n > 0 → fund_amount n = initial_fund * (1 + 0.0312) ^ (n - 1)) ∧
  (total_prize_money 2 11 = 7560) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fund_growth_and_prize_money_l378_37893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_row_middle_must_be_EDCF_l378_37826

-- Define the grid and letter types
def Grid := Fin 6 → Fin 6 → Fin 6
def Letter := Fin 6

-- Define the property that each row, column, and 2x3 rectangle has all different letters
def valid_arrangement (g : Grid) : Prop :=
  (∀ row : Fin 6, Function.Injective (λ col => g row col)) ∧
  (∀ col : Fin 6, Function.Injective (λ row => g row col)) ∧
  (∀ i j : Fin 2, Function.Injective (λ k => g (3*i + k/3) (3*j + k%3)))

-- Define the property for the middle four cells of the fourth row
def fourth_row_middle (g : Grid) : Prop :=
  g 3 1 = 4 ∧ g 3 2 = 3 ∧ g 3 3 = 2 ∧ g 3 4 = 5

-- The main theorem
theorem fourth_row_middle_must_be_EDCF (g : Grid) :
  valid_arrangement g → fourth_row_middle g :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_row_middle_must_be_EDCF_l378_37826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangent_intersection_l378_37888

noncomputable section

-- Define the curve C
def C (x : ℝ) : ℝ := x + 1/x

-- Define the line l
def l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the set of valid k values
def K : Set ℝ := {k | 3/4 < k ∧ k < 1}

-- Define the intersection points M and N
def M (k : ℝ) : ℝ × ℝ := sorry

def N (k : ℝ) : ℝ × ℝ := sorry

-- Define the tangent lines at M and N
def tangent_M (k : ℝ) (x : ℝ) : ℝ := sorry

def tangent_N (k : ℝ) (x : ℝ) : ℝ := sorry

-- Define the intersection point of the tangents
def B (k : ℝ) : ℝ × ℝ := sorry

-- The main theorem
theorem locus_of_tangent_intersection (k : ℝ) (h : k ∈ K) :
  B k = (2, 4 - 2*k) ∧ 2 < (B k).2 ∧ (B k).2 < 5/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangent_intersection_l378_37888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_height_l378_37808

open Real

variable (R : ℝ)
variable (h r : ℝ)

/-- The radius of the sphere --/
def sphere_radius := R

/-- The radius of the inscribed cylinder --/
def cylinder_radius := r

/-- The height of the inscribed cylinder --/
def cylinder_height := h

/-- The condition that the cylinder is inscribed in the sphere --/
def inscribed_condition := r^2 + (h/2)^2 = R^2

/-- The lateral surface area of the cylinder --/
noncomputable def lateral_surface_area := 2 * Real.pi * r * h

/-- Theorem: The height of the cylinder that maximizes the lateral surface area --/
theorem max_lateral_surface_height : 
  ∃ (h : ℝ), h > 0 ∧ h = R * sqrt 2 ∧ 
  ∀ (h' : ℝ), h' ≠ h → lateral_surface_area R h' < lateral_surface_area R h :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lateral_surface_height_l378_37808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3pi_plus_theta_l378_37815

theorem tan_3pi_plus_theta (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 π) (h2 : Real.sin θ - Real.cos θ = 1/5) :
  Real.tan (3*π + θ) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3pi_plus_theta_l378_37815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l378_37834

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 5 * x - 2 else 12 - 3 * x

-- State the theorem
theorem g_values : g (-4) = -22 ∧ g 6 = -6 := by
  -- Split the conjunction
  constructor
  -- Prove g(-4) = -22
  · simp [g]
    norm_num
  -- Prove g(6) = -6
  · simp [g]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l378_37834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_backyard_trees_l378_37865

/-- Calculates the total number of trees in Mark's backyard after all planting is complete. -/
def total_trees (initial_trees : ℕ) (removed_trees : ℕ) (bought_trees : ℕ) (initially_planted : ℕ) (additional_percentage : ℚ) : ℕ :=
  let remaining_trees := initial_trees - removed_trees
  let after_initial_planting := remaining_trees + initially_planted
  let additional_trees := (initially_planted : ℚ) * additional_percentage
  after_initial_planting + Int.toNat additional_trees.ceil

/-- The theorem stating the total number of trees in Mark's backyard. -/
theorem marks_backyard_trees :
  total_trees 13 3 18 12 (1/4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_backyard_trees_l378_37865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_factorial_remainders_l378_37836

/-- For a positive integer n, the numbers 1!, 2!, ..., n! give pairwise distinct remainders when divided by n if and only if n is 1, 2, or 3. -/
theorem distinct_factorial_remainders (n : ℕ) :
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → (Nat.factorial i : ℤ) % n ≠ (Nat.factorial j : ℤ) % n) ↔ n = 1 ∨ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_factorial_remainders_l378_37836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l378_37872

/-- A function f: ℝ → ℝ satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- The condition that f'(x) - f(x) < -3 for all x -/
axiom f_condition (x : ℝ) : f' x - f x < -3

/-- The condition that f(0) = 4 -/
axiom f_zero : f 0 = 4

/-- The main theorem to prove -/
theorem solution_set_equivalence (x : ℝ) : f x > Real.exp x + 3 ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equivalence_l378_37872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_theorem_l378_37820

noncomputable def point_position (t : ℝ) (initial_pos : ℝ) (speed : ℝ) : ℝ :=
  initial_pos + speed * t

noncomputable def A (t : ℝ) : ℝ := point_position t (-24) (-1)
noncomputable def B (t : ℝ) : ℝ := point_position t (-10) 3
noncomputable def C (t : ℝ) : ℝ := point_position t 10 7

noncomputable def AB (t : ℝ) : ℝ := |A t - B t|
noncomputable def BC (t : ℝ) : ℝ := |B t - C t|

noncomputable def P (t : ℝ) : ℝ := 
  if t ≤ 14 then point_position t (-24) 1
  else point_position (t - 14) (-10) 1

noncomputable def Q (t : ℝ) : ℝ := 
  if t ≤ 14 then -24
  else point_position (t - 14) (-24) 3

noncomputable def PQ (t : ℝ) : ℝ := |P t - Q t|

theorem points_theorem :
  (∀ t, (BC t - AB t) / 2 = 3) ∧
  (∀ t, t > 0 → t ≤ 14 → PQ t = t) ∧
  (∀ t, t > 14 → t ≤ 34 → PQ t = |42 - 2*t|) ∧
  (PQ 10 = 10 ∧ PQ 16 = 10 ∧ PQ 26 = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_theorem_l378_37820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_1_plus_x_squared_l378_37840

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 + x^2)

-- State the theorem
theorem derivative_of_sqrt_1_plus_x_squared :
  deriv f = fun x => x / Real.sqrt (1 + x^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sqrt_1_plus_x_squared_l378_37840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_in_five_rolls_l378_37860

noncomputable def roll_at_least_four : ℝ := 1/2

noncomputable def probability_at_least_four_times (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * (roll_at_least_four ^ k) * ((1 - roll_at_least_four) ^ (n - k))

theorem probability_at_least_four_in_five_rolls :
  probability_at_least_four_times 5 4 + probability_at_least_four_times 5 5 = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_in_five_rolls_l378_37860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_point_iff_collinear_intercepts_l378_37821

/-- A circle in the coordinate plane passing through the origin -/
structure CircleThroughOrigin where
  x : ℝ  -- x-intercept
  y : ℝ  -- y-intercept
  not_tangent_to_axes : x ≠ 0 ∧ y ≠ 0

/-- Three circles are pairwise not tangent -/
def NotPairwiseTangent (c₁ c₂ c₃ : CircleThroughOrigin) : Prop :=
  c₁.x ≠ c₂.x ∧ c₁.x ≠ c₃.x ∧ c₂.x ≠ c₃.x ∧
  c₁.y ≠ c₂.y ∧ c₁.y ≠ c₃.y ∧ c₂.y ≠ c₃.y

/-- Three points are collinear -/
def AreCollinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Three circles have a common point other than the origin -/
def HaveCommonPointNotOrigin (c₁ c₂ c₃ : CircleThroughOrigin) : Prop :=
  ∃ (p : ℝ × ℝ), p ≠ (0, 0) ∧ 
    (p.1 - c₁.x/2)^2 + (p.2 - c₁.y/2)^2 = (c₁.x/2)^2 + (c₁.y/2)^2 ∧
    (p.1 - c₂.x/2)^2 + (p.2 - c₂.y/2)^2 = (c₂.x/2)^2 + (c₂.y/2)^2 ∧
    (p.1 - c₃.x/2)^2 + (p.2 - c₃.y/2)^2 = (c₃.x/2)^2 + (c₃.y/2)^2

/-- The main theorem -/
theorem circles_common_point_iff_collinear_intercepts
  (c₁ c₂ c₃ : CircleThroughOrigin)
  (h : NotPairwiseTangent c₁ c₂ c₃) :
  HaveCommonPointNotOrigin c₁ c₂ c₃ ↔ 
  AreCollinear (c₁.x, c₁.y) (c₂.x, c₂.y) (c₃.x, c₃.y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_point_iff_collinear_intercepts_l378_37821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_3_implies_l378_37839

theorem tan_alpha_3_implies (α : ℝ) (h : Real.tan α = 3) :
  (Real.tan (α + π/4) = -2) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_3_implies_l378_37839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l378_37879

/-- The set of complex numbers w such that (5-3i)w has an imaginary part equal to twice its real part -/
def T : Set ℂ :=
  {w : ℂ | 2 * ((5 - 3*Complex.I) * w).re = ((5 - 3*Complex.I) * w).im}

/-- Theorem stating that T is a line in the complex plane -/
theorem T_is_line : ∃ (a b : ℝ) (c : ℂ), c ≠ 0 ∧ T = {w : ℂ | ∃ (t : ℝ), w = a + b*Complex.I + t*c} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l378_37879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l378_37874

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating the properties of the function and its minimum value -/
theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x, (deriv (f a b c)) x = 3*x^2 + 2*a*x + b) ∧
    (f a b c (-1) = 7) ∧
    (deriv (f a b c) (-1) = 0) ∧
    (deriv (f a b c) 3 = 0) ∧
    (a = -3 ∧ b = 6 ∧ c = 17) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b c x ≥ -15) ∧
    (f a b c (-2) = -15) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l378_37874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_quadrilateral_is_rectangle_l378_37825

-- Define the curves
def curve1 (x y : ℝ) : Prop := x * y = 18
def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 36

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- Define the quadrilateral formed by joining the intersection points
def quadrilateral : Set (ℝ × ℝ) :=
  {p | p ∈ intersection_points}

-- Define what it means for a set of points to form a rectangle
def IsRectangle (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ),
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2 ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (d.1 - a.1)^2 + (d.2 - a.2)^2 ∧
    (a.1 - b.1) * (b.1 - c.1) + (a.2 - b.2) * (b.2 - c.2) = 0

-- Theorem statement
theorem intersection_quadrilateral_is_rectangle :
  IsRectangle quadrilateral :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_quadrilateral_is_rectangle_l378_37825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_calculation_l378_37841

-- Define the circle and square
def circle_radius : ℝ := 3
def square_side : ℝ := 6

-- Define the areas
noncomputable def circle_area : ℝ := Real.pi * circle_radius^2
def square_area : ℝ := square_side^2

-- Define the function to calculate the difference in areas
noncomputable def area_difference : ℝ := sorry

-- Theorem statement
theorem area_difference_calculation :
  area_difference = (circle_area - (circle_area ⊓ square_area)) -
                    (square_area - (circle_area ⊓ square_area)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_calculation_l378_37841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_problem_l378_37837

theorem box_volume_problem (edge_length : ℝ) (num_boxes : ℕ) :
  edge_length = 5 ∧ num_boxes = 5 →
  num_boxes * edge_length^3 = 625 := by
  intro h
  have h1 : edge_length = 5 := h.left
  have h2 : num_boxes = 5 := h.right
  rw [h1, h2]
  norm_num

#check box_volume_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_volume_problem_l378_37837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_eighth_to_negative_one_third_equals_two_l378_37805

theorem one_eighth_to_negative_one_third_equals_two :
  (1 / 8 : ℝ) ^ (-(1 / 3 : ℝ)) = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_eighth_to_negative_one_third_equals_two_l378_37805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l378_37884

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = Real.sqrt 7 ∧ C = Real.pi / 3

-- Part I
theorem part_one (a b c A B C : ℝ) (h : Triangle a b c A B C) 
  (h1 : 2 * Real.sin A = 3 * Real.sin B) : 
  a = 3 ∧ b = 2 :=
by sorry

-- Part II
theorem part_two (a b c A B C : ℝ) (h : Triangle a b c A B C) 
  (h2 : Real.cos B = 5 * Real.sqrt 7 / 14) : 
  Real.sin (2 * A) = -3 * Real.sqrt 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l378_37884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l378_37880

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) + Real.sqrt (15 - 3*x)

-- Define the domain
def domain (x : ℝ) : Prop := 4 ≤ x ∧ x ≤ 5

-- Statement of the theorem
theorem f_extrema :
  (∃ x : ℝ, domain x ∧ f x = Real.sqrt 3) ∧
  (∀ x : ℝ, domain x → f x ≤ Real.sqrt 3) ∧
  (∃ x : ℝ, domain x ∧ f x = 1) ∧
  (∀ x : ℝ, domain x → f x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l378_37880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_eaten_ratio_pentagonal_prism_damage_ratio_l378_37842

/-- Represents a non-regular pentagonal prism -/
structure PentagonalPrism where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a regular quadrilateral pyramid -/
structure QuadrilateralPyramid where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- Volume of cheese eaten within 1 cm of a vertex -/
noncomputable def volume_eaten_at_vertex (angle : ℝ) : ℝ := 2 * angle / 3

/-- Total volume of cheese eaten from a pentagonal prism -/
noncomputable def volume_eaten_prism (prism : PentagonalPrism) : ℝ := 2 * Real.pi

/-- Total volume of cheese eaten from a quadrilateral pyramid -/
noncomputable def volume_eaten_pyramid (pyramid : QuadrilateralPyramid) : ℝ := 4 * Real.pi / 9

/-- The theorem stating the relationship between the volumes of cheese eaten -/
theorem cheese_eaten_ratio (prism : PentagonalPrism) (pyramid : QuadrilateralPyramid) :
  volume_eaten_prism prism = (9 / 2) * volume_eaten_pyramid pyramid := by
  sorry

/-- The main theorem proving the 4.5 times more damage for the pentagonal prism -/
theorem pentagonal_prism_damage_ratio (prism : PentagonalPrism) (pyramid : QuadrilateralPyramid) :
  ∃ (r : ℝ), r = 4.5 ∧ volume_eaten_prism prism = r * volume_eaten_pyramid pyramid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_eaten_ratio_pentagonal_prism_damage_ratio_l378_37842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_continuity_l378_37847

open MeasureTheory Topology Filter

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- Continuity of conditional probabilities -/
theorem conditional_probability_continuity
  (A B : Set Ω) (A_n B_n : ℕ → Set Ω)
  (hA : Tendsto (fun n ↦ A_n n) atTop (𝓝 A))
  (hB : Tendsto (fun n ↦ B_n n) atTop (𝓝 B))
  (hBn_pos : ∀ n, μ (B_n n) > 0)
  (hB_pos : μ B > 0) :
  Tendsto (fun n ↦ μ (A_n n ∩ B_n n) / μ (B_n n)) atTop (𝓝 (μ (A ∩ B) / μ B)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_continuity_l378_37847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_point_l378_37890

/-- The closed unit disc -/
def D : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

/-- The function to maximize -/
def f : ℝ × ℝ → ℝ := fun p ↦ p.1 + p.2

/-- The point where the maximum is attained -/
noncomputable def max_point : ℝ × ℝ := (1/Real.sqrt 2, 1/Real.sqrt 2)

theorem max_value_at_point :
  max_point ∈ D ∧ ∀ p ∈ D, f p ≤ f max_point := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_point_l378_37890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_tangent_at_one_g_nonnegative_range_l378_37898

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * f a x - x^2 - a^2

-- Part 1
theorem f_monotonicity (x : ℝ) :
  (∀ y < 0, f 1 y > f 1 x) ∧ (∀ z > 0, f 1 z > f 1 x) := by
  sorry

-- Part 1 (continued)
theorem f_tangent_at_one :
  ∃ k : ℝ, k = Real.exp 1 - 1 ∧
  ∀ x : ℝ, f 1 x = k * (x - 1) + (Real.exp 1 - 1) := by
  sorry

-- Part 2
theorem g_nonnegative_range :
  ∃ a_min a_max : ℝ,
    a_min = -Real.sqrt 2 ∧
    a_max = 2 - Real.log 2 ∧
    ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max ↔ ∀ x : ℝ, x ≥ 0 → g a x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_tangent_at_one_g_nonnegative_range_l378_37898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l378_37807

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 3 * a n / (a n + 3)

def b (n : ℕ) : ℚ := 1 / a n

theorem sequence_properties :
  (∀ n : ℕ, b (n + 1) - b n = 1/3) ∧
  (∀ n : ℕ, a n = 3 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l378_37807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l378_37871

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 4)

noncomputable def g (x : ℝ) := f (x + Real.pi / 8)

theorem g_monotone_increasing_interval :
  ∀ x y, 5 * Real.pi / 8 < x ∧ x < y ∧ y < 7 * Real.pi / 8 → g x < g y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_interval_l378_37871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_0359_to_hundredth_l378_37870

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

theorem round_2_0359_to_hundredth :
  round_to_hundredth 2.0359 = 2.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_0359_to_hundredth_l378_37870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l378_37828

noncomputable section

-- Define the general form of a quadratic equation
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the equations given in the problem
def eq_A (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def eq_B (x : ℝ) : ℝ := 1 / x^2 + x
def eq_C (x : ℝ) : ℝ := x * (x + 3)
def eq_D (x : ℝ) : ℝ := x * (x - 2)

-- Theorem stating that eq_D is quadratic while others are not necessarily quadratic
theorem quadratic_equation_identification :
  (is_quadratic eq_D) ∧
  (¬ ∀ a b c, is_quadratic (eq_A a b c)) ∧
  (¬ is_quadratic eq_B) ∧
  (¬ is_quadratic eq_C) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_identification_l378_37828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l378_37819

noncomputable def g (x m : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 + m

theorem g_properties :
  ∃ (m : ℝ),
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), g x m ≤ 6) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), g x m = 6) ∧
    m = 3 ∧
    (∀ x : ℝ, g x m ≥ 2) ∧
    (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 2 * Real.pi / 3) (k * Real.pi - Real.pi / 6),
      ∀ y ∈ Set.Icc (k * Real.pi - 2 * Real.pi / 3) (k * Real.pi - Real.pi / 6),
        x ≤ y → g (-x) m ≤ g (-y) m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l378_37819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l378_37867

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x - 4*y - 9 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (3, 0)

-- Define the radius of the circle
def radius : ℝ := 3

-- Theorem statement
theorem chord_length : 
  (∃ (x y : ℝ), circle_eq x y ∧ line_eq x y) → 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 36) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l378_37867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_l378_37809

/-- Represents the cost price of an article -/
noncomputable def cost_price : ℝ := sorry

/-- Represents the selling price of an article -/
noncomputable def selling_price : ℝ := sorry

/-- The condition that the cost price of 22 articles equals the selling price of 16 articles -/
axiom price_relation : 22 * cost_price = 16 * selling_price

/-- The profit percentage calculation -/
noncomputable def profit_percentage : ℝ := (selling_price - cost_price) / cost_price * 100

theorem merchant_profit :
  profit_percentage = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_profit_l378_37809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_root_zeroth_root_limit_one_and_zero_are_root_indicators_l378_37811

-- Define the n-th root function
noncomputable def nthRoot (a : ℝ) (n : ℝ) : ℝ := Real.rpow a (1 / n)

-- Theorem for the first root
theorem first_root (a : ℝ) (h : a > 0) : nthRoot a 1 = a := by sorry

-- Theorem for the limit behavior of the zeroth root
theorem zeroth_root_limit (a : ℝ) (h : a > 0) :
  (∀ ε > 0, ∃ δ > 0, ∀ n, 0 < n ∧ n < δ → 
    (a < 1 → nthRoot a n < ε) ∧ 
    (a = 1 → nthRoot a n = 1) ∧ 
    (a > 1 → nthRoot a n > 1/ε)) := by sorry

-- Main theorem stating that 1 and 0 can be indicators of roots
theorem one_and_zero_are_root_indicators (a : ℝ) (h : a > 0) :
  (nthRoot a 1 = a) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ n, 0 < n ∧ n < δ → 
    (a < 1 → nthRoot a n < ε) ∧ 
    (a = 1 → nthRoot a n = 1) ∧ 
    (a > 1 → nthRoot a n > 1/ε)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_root_zeroth_root_limit_one_and_zero_are_root_indicators_l378_37811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l378_37869

def sequence_property (a : ℕ+ → ℤ) (d : ℕ+ → ℤ) : Prop :=
  ∀ n : ℕ+, d n = a (n + 2) + a n - 2 * a (n + 1)

theorem sequence_properties (a : ℕ+ → ℤ) (d : ℕ+ → ℤ) :
  sequence_property a d ∧ a 1 = 1 →
  (∀ n : ℕ+, d n = a n ∧ a 2 = 2 → a n = (2 : ℤ)^(n.val - 1)) ∧
  (a 2 = -2 ∧ ∀ n : ℕ+, d n ≥ 1 → ∀ n : ℕ+, a n ≥ -5) ∧
  (∀ n : ℕ+, |d n| = 1 ∧ a 2 = 1 ∧ ∀ k : ℕ+, a (k + 4) = a k →
    (∀ n : ℕ+, d n = if n.val % 4 = 1 then 1 else if n.val % 4 = 0 then 1 else -1) ∨
    (∀ n : ℕ+, d n = if n.val % 4 = 1 then -1 else if n.val % 4 = 0 then -1 else 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l378_37869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l378_37831

/-- Regular quadrilateral pyramid with all edges of length a -/
structure RegularQuadPyramid (a : ℝ) where
  (a_pos : a > 0)

/-- The cross-section formed by a plane passing through a side of the base
    and the midpoint of the opposite lateral edge -/
noncomputable def cross_section_area (a : ℝ) (p : RegularQuadPyramid a) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (a : ℝ) (p : RegularQuadPyramid a) :
  cross_section_area a p = (3 * a^2 * Real.sqrt 11) / 16 := by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The definition and the right-hand side are identical, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l378_37831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_approx_l378_37853

/-- The cost price of a radio given its selling price and loss percentage -/
noncomputable def cost_price (selling_price : ℝ) (loss_percentage : ℝ) : ℝ :=
  selling_price / (1 - loss_percentage / 100)

/-- Theorem stating that the cost price of a radio is approximately 4500 
    given the selling price and loss percentage -/
theorem radio_cost_price_approx : 
  let selling_price := (3200 : ℝ)
  let loss_percentage := (28.888888888888886 : ℝ)
  abs (cost_price selling_price loss_percentage - 4500) < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_cost_price_approx_l378_37853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l378_37859

-- Define the function as noncomputable due to its dependence on real numbers
noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) + 1 / Real.sqrt (2 - x)

-- Define the domain
def domain : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem f_domain : {x : ℝ | ∃ y, f x = y} = domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l378_37859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l378_37895

/-- Proves that given a cricketer's scores over 5 matches, with known averages for 3 matches and overall, the average for the other 2 matches can be determined. -/
theorem cricketer_average_score 
  (total_matches : ℕ) 
  (known_matches : ℕ) 
  (unknown_matches : ℕ) 
  (known_average : ℝ) 
  (total_average : ℝ) : ℝ :=
  by
  have h1 : total_matches = 5 := by sorry
  have h2 : known_matches = 3 := by sorry
  have h3 : unknown_matches = 2 := by sorry
  have h4 : known_average = 30 := by sorry
  have h5 : total_average = 26 := by sorry
  
  -- The average score for the unknown matches
  let unknown_average := (total_matches * total_average - known_matches * known_average) / unknown_matches
  
  -- Prove that unknown_average equals 20
  have h6 : unknown_average = 20 := by sorry
  
  exact unknown_average

-- Remove the #eval statement as it's causing issues with compilation
-- #eval cricketer_average_score 5 3 2 30 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_score_l378_37895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_sum_l378_37863

open Real

theorem max_value_of_vector_sum (a b : Fin 3 → ℝ) : 
  ‖a‖ = 1 → ‖b‖ = 1 → 
  (∃ (x : Fin 3 → ℝ), ‖x‖ = 3 ∧ ∀ (y : Fin 3 → ℝ), ‖y‖ ≤ 3) → 
  (∃ (z : Fin 3 → ℝ), ‖a + 2 • b‖ = ‖z‖ ∧ ∀ (w : Fin 3 → ℝ), ‖a + 2 • b‖ ≤ ‖w‖) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_vector_sum_l378_37863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l378_37800

/-- Line C₁ with parametric equations x = 1 + t * cos α, y = t * sin α -/
def C₁ (α : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 1 + t * Real.cos α ∧ p.2 = t * Real.sin α}

/-- Curve C₂ with Cartesian equation x²/2 + y² = 1 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- Point P with coordinates (1, 0) -/
def P : ℝ × ℝ := (1, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the result to be proved -/
theorem intersection_reciprocal_sum (α : ℝ) :
  ∀ A B : ℝ × ℝ,
  A ∈ C₁ α ∩ C₂ → B ∈ C₁ α ∩ C₂ → A ≠ B →
  1 / distance P A + 1 / distance P B = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_reciprocal_sum_l378_37800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_average_rate_of_change_l378_37885

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (Δx : ℝ) : ℝ :=
  (f (x + Δx) - f x) / Δx

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := x^3
noncomputable def f4 (x : ℝ) : ℝ := 1/x

theorem greatest_average_rate_of_change :
  let x := (1 : ℝ)
  let Δx := (0.3 : ℝ)
  (average_rate_of_change f3 x Δx > average_rate_of_change f1 x Δx) ∧
  (average_rate_of_change f3 x Δx > average_rate_of_change f2 x Δx) ∧
  (average_rate_of_change f3 x Δx > average_rate_of_change f4 x Δx) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_average_rate_of_change_l378_37885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l378_37817

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the point P on the circle
def point_P : ℝ × ℝ := (-4, -3)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 4*x + 3*y + 25 = 0

-- Theorem statement
theorem tangent_line_at_P :
  my_circle point_P.1 point_P.2 →
  ∀ x y : ℝ, tangent_line x y ↔ 
    (∃ t : ℝ, x = point_P.1 + t * (-3) ∧ y = point_P.2 + t * 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l378_37817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_proof_m_plus_n_proof_l378_37827

/-- The area of a regular hexagon with side length 3 -/
noncomputable def regular_hexagon_area : ℝ := 27 * Real.sqrt 3 / 2

/-- Theorem: The area of a regular hexagon with side length 3 is 27√3/2 -/
theorem regular_hexagon_area_proof : 
  regular_hexagon_area = 27 * Real.sqrt 3 / 2 := by
  -- Unfold the definition of regular_hexagon_area
  unfold regular_hexagon_area
  -- The equality is now trivial
  rfl

/-- The sum of m and n in the expression 3√m + n representing the hexagon area -/
def m_plus_n : ℕ := 27

/-- Theorem: m + n = 27 in the expression 3√m + n representing the hexagon area -/
theorem m_plus_n_proof : m_plus_n = 27 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_proof_m_plus_n_proof_l378_37827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_right_ray_l378_37833

/-- A function f : ℝ → ℝ satisfying the given conditions -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x, f x = f (-x)

/-- The derivative of f is less than f -/
axiom f_deriv_lt : ∀ x, deriv f x < f x

/-- f has a specific symmetry property -/
axiom f_symmetry : ∀ x, f (x + 1) = f (3 - x)

/-- The value of f at 2015 is 2 -/
axiom f_2015 : f 2015 = 2

/-- The solution set of the inequality f(x) < 2e^(x-1) -/
def solution_set : Set ℝ := {x | f x < 2 * Real.exp (x - 1)}

/-- The main theorem stating that the solution set is (1, +∞) -/
theorem solution_set_is_open_right_ray : 
  solution_set = Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_right_ray_l378_37833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_valid_numbers_l378_37849

def valid_digits : List ℕ := [1, 2, 3, 7, 8, 9]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 ∈ [1, 2, 3]) ∧
  ((n / 10) % 10 ∈ valid_digits) ∧
  (n % 10 ∈ valid_digits)

def sum_of_valid_numbers (a b : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (∀ d ∈ valid_digits, (List.countP (· = d) (Nat.digits 10 a) + List.countP (· = d) (Nat.digits 10 b) = List.countP (· = d) valid_digits))

theorem smallest_sum_of_valid_numbers :
  ∀ a b : ℕ, sum_of_valid_numbers a b → a + b ≥ 417 :=
by
  sorry

#eval valid_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_valid_numbers_l378_37849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_polar_axis_l378_37899

/-- The value of 'a' for which the curves C₁ and C₂ intersect on the polar axis -/
theorem intersection_on_polar_axis (a : ℝ) (ha : a > 0) :
  (∃ θ : ℝ, (Real.sqrt 2 * Real.cos θ + Real.sin θ) * a = 1 ∧ Real.sin θ = 0) →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_polar_axis_l378_37899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_six_l378_37868

/-- The time when Maxwell and Brad meet, given their speeds and the distance between their homes -/
noncomputable def meeting_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) : ℝ :=
  (distance + brad_speed * brad_delay) / (maxwell_speed + brad_speed)

theorem meeting_time_is_six :
  let distance : ℝ := 54
  let maxwell_speed : ℝ := 4
  let brad_speed : ℝ := 6
  let brad_delay : ℝ := 1
  meeting_time distance maxwell_speed brad_speed brad_delay = 6 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_is_six_l378_37868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_construction_l378_37864

/-- A parabola is defined by its focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- A circle is defined by its center and a point on its circumference -/
structure Circle where
  center : ℝ × ℝ
  passthrough : ℝ × ℝ

/-- External tangent to two circles -/
noncomputable def external_tangent (c1 c2 : Circle) : ℝ → ℝ :=
  sorry

/-- Perpendicular line through a point -/
noncomputable def perpendicular_through (l : ℝ → ℝ) (p : ℝ × ℝ) : ℝ → ℝ :=
  sorry

/-- Check if a point is on a parabola -/
def on_parabola (p : ℝ × ℝ) (para : Parabola) : Prop :=
  let (x, y) := p
  let (fx, fy) := para.focus
  (x - fx)^2 + (y - fy)^2 = (para.directrix y - x)^2

theorem parabola_construction 
  (P₁ P₂ F : ℝ × ℝ) 
  (h : ∃ p : Parabola, on_parabola P₁ p ∧ on_parabola P₂ p ∧ F = p.focus) :
  ∃ (d : ℝ → ℝ) (t : ℝ → ℝ),
    d = external_tangent (Circle.mk P₁ F) (Circle.mk P₂ F) ∧
    t = perpendicular_through d F ∧
    ∃ p : Parabola, p.focus = F ∧ p.directrix = d ∧ on_parabola P₁ p ∧ on_parabola P₂ p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_construction_l378_37864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l378_37892

noncomputable section

-- Define the types for points and lines
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the intersection point of two lines
noncomputable def intersection (l1 l2 : Line) : Point :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the theorem
theorem line_and_circle_equations :
  ∀ (l1 l2 l3 : Line) (p1 p2 : Point),
    -- Line conditions
    (l1.a = 2 ∧ l1.b = 1 ∧ l1.c = -8) →
    (l2.a = 1 ∧ l2.b = -2 ∧ l2.c = -1) →
    (l3.a = 6 ∧ l3.b = -8 ∧ l3.c = 3) →
    -- Circle conditions
    (p1.x = -1 ∧ p1.y = 1) →
    (p2.x = 1 ∧ p2.y = 3) →
    -- Prove that:
    ∃ (result_line : Line) (result_circle : Circle),
      -- The result line passes through the intersection of l1 and l2
      (result_line.a * (intersection l1 l2).x + result_line.b * (intersection l1 l2).y + result_line.c = 0) ∧
      -- The result line is perpendicular to l3
      (perpendicular result_line l3) ∧
      -- The result circle passes through p1 and p2
      ((p1.x - result_circle.center.x)^2 + (p1.y - result_circle.center.y)^2 = result_circle.radius^2) ∧
      ((p2.x - result_circle.center.x)^2 + (p2.y - result_circle.center.y)^2 = result_circle.radius^2) ∧
      -- The circle's center is on the x-axis
      (result_circle.center.y = 0) ∧
      -- The resulting equations are as specified
      (result_line.a = 4 ∧ result_line.b = 3 ∧ result_line.c = -18) ∧
      (result_circle.center.x = 2 ∧ result_circle.radius^2 = 10) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l378_37892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_car_travel_distance_l378_37857

/-- The distance traveled by the electric car given the distance traveled by the diesel car and the percentage increase -/
noncomputable def electric_car_distance (diesel_distance : ℝ) (percentage_increase : ℝ) : ℝ :=
  diesel_distance * (1 + percentage_increase / 100)

/-- Theorem: The electric car travels 180 miles given the conditions -/
theorem electric_car_travel_distance :
  let diesel_distance : ℝ := 120
  let percentage_increase : ℝ := 50
  electric_car_distance diesel_distance percentage_increase = 180 := by
  -- Unfold the definition of electric_car_distance
  unfold electric_car_distance
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_car_travel_distance_l378_37857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l378_37830

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the problem statement
theorem problem_statement (a : ℝ) : floor (5 * a - 0.9) = ⌊3 * a + 0.7⌋ → a = 1.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l378_37830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_income_is_60000_l378_37802

/-- Represents the tax structure and Jordan's income --/
structure TaxSystem where
  q : ℝ  -- Base tax rate
  baseIncome : ℝ  -- Income threshold ($30000)
  deduction : ℝ  -- Flat tax deduction ($600)
  jordanIncome : ℝ  -- Jordan's annual income

/-- Calculates the tax amount before deduction --/
noncomputable def taxBeforeDeduction (ts : TaxSystem) : ℝ :=
  (ts.q / 100) * ts.baseIncome + 
  ((ts.q + 3) / 100) * (ts.jordanIncome - ts.baseIncome)

/-- Calculates the effective tax rate --/
noncomputable def effectiveTaxRate (ts : TaxSystem) : ℝ :=
  (taxBeforeDeduction ts - ts.deduction) / ts.jordanIncome

/-- Theorem stating that Jordan's income is $60000 --/
theorem jordan_income_is_60000 (ts : TaxSystem) 
  (h1 : ts.baseIncome = 30000)
  (h2 : ts.deduction = 600)
  (h3 : effectiveTaxRate ts = (ts.q + 0.5) / 100) :
  ts.jordanIncome = 60000 := by
  sorry

#check jordan_income_is_60000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_income_is_60000_l378_37802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_l378_37883

-- Define the ellipse parametrically
noncomputable def ellipse_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def ellipse_y (φ : ℝ) : ℝ := 4 * Real.sin φ

-- Theorem: The length of the major axis of the ellipse is 8
theorem major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (∃ φ, x = ellipse_x φ ∧ y = ellipse_y φ) ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  max a b = 4 := by
  sorry

#check major_axis_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_l378_37883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_of_decreasing_is_constant_count_sequences_with_convex_1339_l378_37845

-- Define a sequence of natural numbers
def Sequence := ℕ → ℕ

-- Define the convex sequence of a given sequence
def convexSequence (a : Sequence) : Sequence :=
  fun n => Finset.sup (Finset.range n) a

-- Define a decreasing sequence
def isDecreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≥ a (n + 1)

-- Define a constant sequence
def isConstant (b : Sequence) : Prop :=
  ∀ n m : ℕ, b n = b m

-- Theorem 1: The convex sequence of a decreasing sequence is constant
theorem convex_of_decreasing_is_constant (a : Sequence) (h : isDecreasing a) :
  isConstant (convexSequence a) :=
sorry

-- Define a predicate for sequences with convex sequence 1, 3, 3, 9
def hasConvexSequence1339 (a : Sequence) : Prop :=
  convexSequence a 0 = 1 ∧ convexSequence a 1 = 3 ∧ convexSequence a 2 = 3 ∧ convexSequence a 3 = 9

-- Theorem 2: There are exactly 3 sequences with convex sequence 1, 3, 3, 9
theorem count_sequences_with_convex_1339 :
  ∃! (s : Finset Sequence), (∀ a ∈ s, hasConvexSequence1339 a) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_of_decreasing_is_constant_count_sequences_with_convex_1339_l378_37845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_l378_37891

/-- A sequence of positive integers whose binary representations have exactly 9 ones -/
def T : ℕ → ℕ := sorry

/-- The 1500th number in the sequence T -/
def M : ℕ := T 1500

/-- The property that T is an increasing sequence -/
axiom T_increasing : ∀ n m, n < m → T n < T m

/-- The property that each number in T has exactly 9 ones in its binary representation -/
axiom T_binary_ones : ∀ n, (Nat.digits 2 (T n)).count 1 = 9

/-- The remainder when M is divided by 1500 -/
def remainder : ℕ := M % 1500

theorem M_remainder :
  ∃ r, remainder = r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_remainder_l378_37891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l378_37855

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 10)^2 + y^2 = 36
def C₂ (x y : ℝ) : Prop := (x + 15)^2 + y^2 = 81

-- Define the centers and radii of the circles
def center_C₁ : ℝ × ℝ := (10, 0)
def center_C₂ : ℝ × ℝ := (-15, 0)
def radius_C₁ : ℝ := 6
def radius_C₂ : ℝ := 9

-- Define a function to calculate the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Define the theorem
theorem shortest_tangent_length :
  ∃ (P Q : ℝ × ℝ),
    C₁ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧
    (∀ (P' Q' : ℝ × ℝ), C₁ P'.1 P'.2 → C₂ Q'.1 Q'.2 → 
      distance P Q ≤ distance P' Q') ∧
    distance P Q = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l378_37855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_correct_l378_37858

/-- The coefficient of x^2 in the expansion of (2x^3 - 1/x)^6 -/
def coefficient_x_squared : ℕ := 60

/-- The binomial expression (2x^3 - 1/x)^6 -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (2 * x^3 - 1/x)^6

/-- Theorem stating that the coefficient of x^2 in the expansion is correct -/
theorem coefficient_x_squared_correct :
  ∃ (f : ℝ → ℝ), ∀ x, x ≠ 0 → 
    binomial_expression x = coefficient_x_squared * x^2 + f x ∧ 
    Filter.Tendsto (fun y => f y / y^2) Filter.atTop (nhds 0) := by
  sorry

#check coefficient_x_squared_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_correct_l378_37858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l378_37824

theorem relationship_abc (a b c : ℝ) (ha : a = (2 : ℝ)^(3/10)) (hb : b = (2 : ℝ)^(1/10)) (hc : c = ((1/5) : ℝ)^(13/10)) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l378_37824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l378_37844

/-- Given a circle with radius 10 inches and two equal parallel chords 10 inches apart,
    the area between the chords is 50π - 25√3 square inches. -/
theorem area_between_chords (r : ℝ) (d : ℝ) (h1 : r = 10) (h2 : d = 10) : 
  let θ := 2 * Real.arccos (d / (2 * r))
  let chord_length := 2 * r * Real.sin (θ / 2)
  let sector_area := r^2 * θ / 2
  let triangle_area := d * chord_length / 4
  2 * (sector_area - triangle_area) = 50 * Real.pi - 25 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_l378_37844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_3_pow_2007_l378_37813

noncomputable def series_sum (n : ℕ) : ℝ :=
  (3^(n+1) * (1 - 2*n) + 3) / 4

theorem smallest_n_exceeding_3_pow_2007 :
  ∀ k : ℕ, k < 2000 → series_sum k ≤ 3^2007 ∧
  series_sum 2000 > 3^2007 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_3_pow_2007_l378_37813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l378_37876

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 49 + y^2 / 16 = 4

-- Define the distance between foci
noncomputable def foci_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  4 * Real.sqrt 33

-- Theorem statement
theorem ellipse_foci_distance :
  foci_distance ellipse_equation = 4 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l378_37876
