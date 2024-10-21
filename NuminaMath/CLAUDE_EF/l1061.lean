import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbleArrangements_l1061_106164

/-- The number of ways to arrange 5 distinct objects in a line. -/
def totalArrangements : ℕ := 120

/-- The number of ways to arrange 5 distinct objects in a line where two specific objects are adjacent. -/
def adjacentArrangements : ℕ := 48

/-- The number of ways to arrange 5 distinct objects in a line where two specific objects are not adjacent. -/
def nonAdjacentArrangements : ℕ := totalArrangements - adjacentArrangements

theorem marbleArrangements : nonAdjacentArrangements = 72 := by
  unfold nonAdjacentArrangements
  unfold totalArrangements
  unfold adjacentArrangements
  rfl

#eval nonAdjacentArrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbleArrangements_l1061_106164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_north_notation_walking_north_notation_l1061_106101

/-- Represents the direction of movement --/
inductive Direction
  | North
  | South

/-- Represents a distance with direction --/
structure DirectedDistance where
  distance : ℝ
  direction : Direction

/-- Notation for directed distances --/
def directedNotation (d : DirectedDistance) : ℝ :=
  match d.direction with
  | Direction.South => d.distance
  | Direction.North => -d.distance

theorem north_notation (d : ℝ) :
  directedNotation { distance := d, direction := Direction.North } = -d :=
by sorry

theorem walking_north_notation :
  directedNotation { distance := 32, direction := Direction.North } = -32 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_north_notation_walking_north_notation_l1061_106101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1061_106117

theorem triangle_side_length (T PU PT TU : ℝ) : 
  Real.cos T = 3/5 → PU = 13 → PT^2 + TU^2 = PU^2 → PT = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1061_106117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_arrangement_count_l1061_106148

/-- Represents the number of products -/
def n : ℕ := 5

/-- Represents the constraint that A and B must be adjacent -/
def adjacent_constraint : Prop := True

/-- Represents the constraint that C and D must not be adjacent -/
def not_adjacent_constraint : Prop := True

/-- The total number of arrangements -/
def total_arrangements : ℕ := 24

/-- Theorem stating that the number of valid arrangements is 24 -/
theorem shelf_arrangement_count :
  ∀ (products : Fin n → Type) 
    (is_adjacent : adjacent_constraint) 
    (not_adjacent : not_adjacent_constraint),
  (Fintype.card (Equiv.Perm (Fin n))) = total_arrangements := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_arrangement_count_l1061_106148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1061_106103

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.cos (α + π/4) = 4/5) : Real.cos α = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1061_106103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1061_106113

/-- The molecular weight of a compound given the total weight of a specific number of moles -/
noncomputable def molecular_weight (total_weight : ℚ) (num_moles : ℚ) : ℚ :=
  total_weight / num_moles

/-- Theorem: The molecular weight of a compound is 233 grams/mole -/
theorem compound_molecular_weight :
  molecular_weight 699 3 = 233 := by
  unfold molecular_weight
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1061_106113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l1061_106128

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l1061_106128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_l1061_106135

theorem sum_distinct_prime_factors : 
  (Finset.sum (Finset.filter Nat.Prime ((Nat.factors (7^6 - 7^4 + 11)).toFinset)) id) = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_l1061_106135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l1061_106121

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate the ratio of two numbers -/
noncomputable def ratio (a : ℝ) (b : ℝ) : ℝ × ℝ :=
  (a, b)

theorem interest_ratio :
  let si := simple_interest 1750 8 3
  let ci := compound_interest 4000 10 2
  ratio si ci = (420, 840) := by
  sorry

#eval (420 : ℚ) / 840  -- This should evaluate to 1/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l1061_106121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_polar_coords_intersections_valid_l1061_106185

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)
noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (Real.sqrt 5 - (Real.sqrt 2 / 2) * t, -(Real.sqrt 2 / 2) * t)

-- Define the intersection points in Cartesian coordinates
def intersection_points : Set (ℝ × ℝ) := {(0, -Real.sqrt 5), (Real.sqrt 5, 0)}

-- Define the polar coordinates of the intersection points
def polar_intersection_points : Set (ℝ × ℝ) := {(5, 3 * Real.pi / 2), (5, 0)}

-- Theorem stating that the intersection points in polar coordinates are correct
theorem intersection_points_polar_coords :
  ∀ (x y : ℝ), (x, y) ∈ intersection_points →
  ∃ (ρ θ : ℝ), (ρ, θ) ∈ polar_intersection_points ∧
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

-- Theorem stating that the intersection points are indeed intersections of C₁ and C₂
theorem intersections_valid :
  ∀ (x y : ℝ), (x, y) ∈ intersection_points →
  ∃ (θ t : ℝ), C₁ θ = (x, y) ∧ C₂ t = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_polar_coords_intersections_valid_l1061_106185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1061_106182

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

/-- Calculates the time taken for a train to pass a man moving in opposite directions -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := kmph_to_mps train_speed_kmph
  let man_speed_mps := kmph_to_mps man_speed_kmph
  let relative_speed := train_speed_mps + man_speed_mps
  train_length / relative_speed

theorem train_passing_man_time :
  let train_length : ℝ := 140
  let train_speed_kmph : ℝ := 77.993280537557
  let man_speed_kmph : ℝ := 6
  let calculated_time := train_passing_time train_length train_speed_kmph man_speed_kmph
  abs (calculated_time - 1.2727) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1061_106182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1061_106195

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.sqrt (1 + x^4) > 1 - 2*a*x + x^2) ↔ a > 1 - Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1061_106195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_behavior_l1061_106126

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

-- Define the second derivative of f
noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / (x^3)

-- State the theorem
theorem f_second_derivative_behavior 
  (x₀ : ℝ) 
  (h_x₀_pos : x₀ > 0) 
  (h_f''_x₀_zero : f'' x₀ = 0) 
  (m n : ℝ) 
  (h_m : 0 < m ∧ m < x₀) 
  (h_n : x₀ < n) : 
  f'' m < 0 ∧ f'' n > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_behavior_l1061_106126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1061_106160

noncomputable def solve_power_equation (a b c d : ℝ) : ℝ :=
  let lhs := a^(1.25 : ℝ) * c^(0.75 : ℝ)
  Real.log ((d / lhs) / b) / Real.log b

theorem power_equation_solution :
  let x := solve_power_equation 5 12 60 300
  abs (x - 0.026335) < 0.000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1061_106160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l1061_106174

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x + 2

-- State the theorem
theorem function_inequality_range (a : ℝ) :
  f (a^2) + f (a - 2) > 4 ↔ a < -2 ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l1061_106174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1061_106116

/-- Represents a train with a length and speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- The time it takes for a train to pass a stationary point -/
noncomputable def passingTime (train : Train) : ℝ := train.length / train.speed

/-- The time it takes for two trains to cross each other -/
noncomputable def crossingTime (train1 train2 : Train) : ℝ := (train1.length + train2.length) / (train1.speed + train2.speed)

theorem train_passing_time
  (train1 train2 : Train)
  (h1 : train1.speed = train2.speed)
  (h2 : passingTime train1 = 27)
  (h3 : crossingTime train1 train2 = 22) :
  passingTime train2 = 17 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1061_106116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_count_l1061_106110

/-- Represents a pair of integers (a, b) where a < b -/
structure IntPair where
  a : ℕ
  b : ℕ
  h : a < b

/-- The set of integers from which pairs are chosen -/
def IntSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2021}

/-- Predicate to check if two IntPairs have no common elements -/
def noCommonElement (p1 p2 : IntPair) : Prop :=
  p1.a ≠ p2.a ∧ p1.a ≠ p2.b ∧ p1.b ≠ p2.a ∧ p1.b ≠ p2.b

/-- Predicate to check if all sums in a list of IntPairs are distinct -/
def allSumsDistinct (pairs : List IntPair) : Prop :=
  ∀ i j, i ≠ j → (pairs.get i).a + (pairs.get i).b ≠ (pairs.get j).a + (pairs.get j).b

/-- Predicate to check if all sums in a list of IntPairs are ≤ 2021 -/
def allSumsInRange (pairs : List IntPair) : Prop :=
  ∀ p ∈ pairs, p.a + p.b ≤ 2021

/-- The main theorem statement -/
theorem max_pairs_count :
  ∀ (pairs : List IntPair),
    (∀ p ∈ pairs, p.a ∈ IntSet ∧ p.b ∈ IntSet) →
    (∀ i j, i ≠ j → noCommonElement (pairs.get i) (pairs.get j)) →
    allSumsDistinct pairs →
    allSumsInRange pairs →
    pairs.length ≤ 808 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pairs_count_l1061_106110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_dilation_matrix_3_l1061_106152

/-- A matrix representing a dilation centered at the origin with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- The determinant of a 3x3 dilation matrix with scale factor 3 is 27 -/
theorem det_dilation_matrix_3 :
  Matrix.det (dilationMatrix 3) = 27 := by
  -- Expand the definition of dilationMatrix
  unfold dilationMatrix
  -- Simplify the determinant calculation
  simp [Matrix.det_diagonal]
  -- Evaluate the product
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_dilation_matrix_3_l1061_106152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l1061_106138

/-- The difference of squares formula -/
def difference_of_squares (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Factorization using difference of squares is possible -/
def is_difference_of_squares (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ → ℝ), ∀ x y, f x y = difference_of_squares (g x y) (h x y)

/-- The four expressions from the problem -/
def expr1 : ℝ → ℝ → ℝ := λ a b => a^2 - b^2
def expr2 : ℝ → ℝ → ℝ := λ x y => 49*x^2 - y^2
def expr3 : ℝ → ℝ → ℝ := λ x y => -x^2 - y^2
def expr4 : ℝ → ℝ → ℝ := λ m n => 16*m^2 - 25*n^2

/-- Theorem: expr3 cannot be factorized using difference of squares, while others can -/
theorem difference_of_squares_factorization :
  is_difference_of_squares expr1 ∧
  is_difference_of_squares expr2 ∧
  ¬is_difference_of_squares expr3 ∧
  is_difference_of_squares expr4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l1061_106138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l1061_106188

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the upper vertex B
def B : ℝ × ℝ := (0, 1)

-- Define a point P on the ellipse
noncomputable def P (θ : ℝ) : ℝ × ℝ := (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_PB :
  ∃ (M : ℝ), M = 5/2 ∧ ∀ (θ : ℝ), distance (P θ) B ≤ M := by
  sorry

#check max_distance_PB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l1061_106188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l1061_106150

def myCircle (a x y : ℝ) : Prop := (x - a)^2 + y^2 = 4

def myRay (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∧ x ≥ 0

theorem no_common_points (a : ℝ) : 
  (∀ x y : ℝ, ¬(myCircle a x y ∧ myRay x y)) ↔ (a < -2 ∨ a > (4/3) * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l1061_106150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_paths_M_to_N_l1061_106140

-- Define the network structure
inductive Node : Type
  | M | A | B | C | D | E | N

-- Define the directed edges in the network
def hasEdge : Node → Node → Prop
  | Node.M, Node.A => True
  | Node.M, Node.B => True
  | Node.M, Node.E => True
  | Node.A, Node.C => True
  | Node.A, Node.D => True
  | Node.B, Node.N => True
  | Node.B, Node.C => True
  | Node.C, Node.N => True
  | Node.D, Node.N => True
  | Node.E, Node.B => True
  | Node.E, Node.D => True
  | _, _ => False

-- Define a path in the network
def ValidPath (start finish : Node) : List Node → Prop
  | [] => start = finish
  | (h :: t) => hasEdge start h ∧ ValidPath h finish t

-- Count the number of paths between two nodes
def countPaths (start finish : Node) : Nat :=
  sorry

-- Theorem to prove
theorem count_paths_M_to_N :
  countPaths Node.M Node.N = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_paths_M_to_N_l1061_106140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l1061_106124

/-- Given that the terminal side of angle α passes through point P(4, -3),
    prove that sin(3π/2 + α) = -4/5 -/
theorem sin_three_pi_half_plus_alpha (α : ℝ) : 
  (4 : ℝ) * Real.cos α = 4 ∧ (4 : ℝ) * Real.sin α = -3 → 
  Real.sin (3 * Real.pi / 2 + α) = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_pi_half_plus_alpha_l1061_106124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_packing_l1061_106118

/-- A circle with a given center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A square with a given side length and bottom-left corner. -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- A line segment with start and end points. -/
structure LineSegment where
  start : ℝ × ℝ
  endPoint : ℝ × ℝ

/-- Check if a line segment intersects a circle. -/
def intersects (l : LineSegment) (c : Circle) : Prop :=
  sorry

/-- Check if a point is inside a square. -/
def insideSquare (p : ℝ × ℝ) (s : Square) : Prop :=
  sorry

/-- The main theorem to be proved. -/
theorem circle_packing (s : Square) (circles : List Circle) :
  s.sideLength = 100 →
  (∀ c ∈ circles, c.radius = 1) →
  (∀ l : LineSegment, l.start.1 - l.endPoint.1 = 10 ∨ l.start.2 - l.endPoint.2 = 10) →
  (∀ l : LineSegment, insideSquare l.start s ∧ insideSquare l.endPoint s →
    ∃ c ∈ circles, intersects l c) →
  circles.length ≥ 400 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_packing_l1061_106118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_7776_l1061_106177

/-- Custom multiplication operation on nonzero real numbers -/
noncomputable def custom_mul : ℝ → ℝ → ℝ := sorry

/-- Axiom: The custom multiplication is associative -/
axiom custom_mul_assoc (a b c : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → custom_mul a (custom_mul b c) = custom_mul (custom_mul a b) c

/-- Axiom: Each nonzero real number is its own inverse under the custom multiplication -/
axiom custom_mul_self_inverse (a : ℝ) : a ≠ 0 → custom_mul a a = 1

/-- Theorem: The solution to x * 36 = 216 is 7776 -/
theorem solution_is_7776 : ∃ (x : ℝ), x ≠ 0 ∧ custom_mul x 36 = 216 ∧ x = 7776 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_7776_l1061_106177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l1061_106191

open Set
open Function

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) (h : a < b) :
  (∃ f : ℝ → ℝ, DifferentiableOn ℝ f (Ioo a b) ∧ 
    (∀ x ∈ Ioo a b, (deriv f) x < 0) → 
    StrictMonoOn f (Ioo a b)) ∧
  (∃ f : ℝ → ℝ, DifferentiableOn ℝ f (Ioo a b) ∧ 
    StrictMonoOn f (Ioo a b) ∧ 
    ∃ x ∈ Ioo a b, (deriv f) x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_condition_l1061_106191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1061_106119

/-- A race between two runners A and B -/
structure Race where
  distance : ℚ
  time_B : ℚ
  lead_A : ℚ

/-- Calculate the time taken by runner A to finish the race -/
def time_A (r : Race) : ℚ :=
  (r.distance - r.lead_A) / (r.distance / r.time_B)

/-- Theorem: In a 100-meter race where B takes 25 seconds and A beats B by 20 meters, A finishes in 20 seconds -/
theorem race_theorem (r : Race) 
  (h1 : r.distance = 100)
  (h2 : r.time_B = 25)
  (h3 : r.lead_A = 20) : 
  time_A r = 20 := by
  sorry

#eval time_A { distance := 100, time_B := 25, lead_A := 20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1061_106119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1061_106132

/-- The function g(x) = 4 / (5x^4 - 3) -/
noncomputable def g (x : ℝ) : ℝ := 4 / (5 * x^4 - 3)

/-- g(x) is an even function -/
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  simp [g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1061_106132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1061_106153

noncomputable def expansion (x n : ℝ) := (x^(1/2) - 1 / (2 * x^(1/4)))^n

noncomputable def coefficient (n r : ℕ) : ℝ := (1/2)^r * Nat.choose n r

def first_three_arithmetic (n : ℕ) : Prop :=
  coefficient n 0 + coefficient n 2 = 2 * coefficient n 1

noncomputable def x_coefficient (n : ℕ) : ℝ := (1/2)^4 * Nat.choose n 4

theorem expansion_properties :
  ∀ n : ℕ, first_three_arithmetic n → n = 8 ∧ x_coefficient n = 35/8 := by
  sorry

#check expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l1061_106153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l1061_106129

/-- The area of an equilateral triangle with perimeter 3a is (√3/4)a² -/
theorem equilateral_triangle_area (a : ℝ) (h : a > 0) :
  let perimeter := 3 * a
  let side := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side^2
  area = (Real.sqrt 3 / 4) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l1061_106129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1061_106145

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_decreasing :
  ∃ α : ℝ, (f α 2 = Real.sqrt 2 / 2) ∧ 
    (∀ x y : ℝ, x > 0 → y > 0 → x < y → f α x > f α y) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l1061_106145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cod_age_at_15kg_l1061_106131

-- Define the type for the cod's characteristics
structure Cod where
  age : ℝ
  mass : ℝ

-- Define the graph as a function from age to mass
def graph : ℝ → ℝ := sorry

-- Define Jeff's pet Atlantic cod
def jeffs_cod : Cod := sorry

-- Theorem statement
theorem cod_age_at_15kg : 
  graph jeffs_cod.age = 15 → jeffs_cod.age = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cod_age_at_15kg_l1061_106131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l1061_106197

/-- Two lines intersecting at a point P --/
structure IntersectingLines where
  P : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ

/-- Find x-intercept of a line given a point and slope --/
noncomputable def xIntercept (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  p.1 - p.2 / m

/-- Calculate area of a triangle given base and height --/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Theorem statement --/
theorem area_of_triangle (lines : IntersectingLines)
    (h1 : lines.P = (2, 5))
    (h2 : lines.slope1 = -1)
    (h3 : lines.slope2 = 3) :
    triangleArea (xIntercept lines.P lines.slope1 - xIntercept lines.P lines.slope2) lines.P.2 = 50 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l1061_106197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1061_106136

-- Define the individual completion times
noncomputable def sam_time : ℚ := 4
noncomputable def lisa_time : ℚ := 6
noncomputable def tom_time : ℚ := 2
noncomputable def jessica_time : ℚ := 3

-- Define the function to calculate the combined completion time
noncomputable def combined_time (t1 t2 t3 t4 : ℚ) : ℚ :=
  1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)

-- Theorem statement
theorem combined_work_time :
  combined_time sam_time lisa_time tom_time jessica_time = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l1061_106136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a_for_arithmetic_sequence_zeros_l1061_106157

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * exp x - a * (x^2 - 2*x) + 2

/-- Theorem stating the existence and uniqueness of a for arithmetic sequence zeros -/
theorem exists_unique_a_for_arithmetic_sequence_zeros :
  ∃! a : ℝ, 0 < a ∧ a < exp 1 / 2 ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    x₁ * x₃ < 0 ∧
    x₃ - x₂ = x₂ - x₁) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a_for_arithmetic_sequence_zeros_l1061_106157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_recurrence_relation_second_recurrence_relation_l1061_106179

/-- A sequence where the first 6 terms are 1, 0, 1, 0, 1, 0 -/
def a : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 0
  | 3 => 1
  | 4 => 0
  | 5 => 1
  | n + 6 => a n  -- This defines the pattern for n ≥ 6

/-- The first recurrence relation holds for all n ≥ 1 -/
theorem first_recurrence_relation : ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 1 := by
  sorry

/-- The second recurrence relation holds for all n ≥ 1 -/
theorem second_recurrence_relation : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = (-1)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_recurrence_relation_second_recurrence_relation_l1061_106179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l1061_106100

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

/-- Given complex number z = 2i - 1, prove it is in the second quadrant -/
theorem complex_in_second_quadrant :
  let z : ℂ := 2 * Complex.I - 1
  is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l1061_106100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_arrangement_l1061_106109

def arrange_numbers (a b c d : ℕ) (op1 op2 op3 op4 : ℕ → ℕ → ℕ) : ℕ :=
  op2 (op1 a b) (op3 c (op4 d 1))

def add : ℕ → ℕ → ℕ := (·+·)
def sub : ℕ → ℕ → ℕ := (·-·)
def mul : ℕ → ℕ → ℕ := (·*·)
def div : ℕ → ℕ → ℕ := (·/·)

theorem max_value_arrangement :
  ∀ (ops : Fin 4 → ℕ → ℕ → ℕ),
    (∃ (p : Equiv.Perm (Fin 4)), ops = fun i => [add, sub, mul, div].get (p i)) →
    arrange_numbers 17 17 17 17 
      (ops 0)
      (ops 1)
      (ops 2)
      (ops 3) ≤ 305 :=
by
  sorry

#check max_value_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_arrangement_l1061_106109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_fill_time_for_leaky_cistern_l1061_106193

/-- Represents the time it takes to fill a leaky cistern -/
noncomputable def fill_time_with_leak (normal_fill_time hours_to_empty : ℝ) : ℝ :=
  1 / (1 / normal_fill_time - 1 / hours_to_empty)

/-- Theorem stating the additional time needed to fill a leaky cistern -/
theorem additional_fill_time_for_leaky_cistern 
  (normal_fill_time : ℝ) 
  (hours_to_empty : ℝ) 
  (h1 : normal_fill_time = 6) 
  (h2 : hours_to_empty = 24) : 
  fill_time_with_leak normal_fill_time hours_to_empty - normal_fill_time = 2 := by
  sorry

/-- Computes the result using rational numbers for exact calculation -/
def compute_result : ℚ :=
  1 / (1 / 6 - 1 / 24) - 6

#eval compute_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_fill_time_for_leaky_cistern_l1061_106193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_triangle_problem_l1061_106154

noncomputable section

-- Define the vectors m and n
def m (x : ℝ) : ℝ × ℝ := (2 - Real.sin (2*x + Real.pi/6), -2)
def n (x : ℝ) : ℝ × ℝ := (1, Real.sin x ^ 2)

-- Define the function f
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem vector_triangle_problem (k : ℤ) (ABC : Triangle) 
  (h1 : f (ABC.B / 2) = 1)
  (h2 : ABC.b = 1)
  (h3 : ABC.c = Real.sqrt 3) :
  (∀ x ∈ Set.Icc (k * Real.pi - 2*Real.pi/3) (k * Real.pi - Real.pi/6), 
    ∀ y ∈ Set.Icc (k * Real.pi - 2*Real.pi/3) (k * Real.pi - Real.pi/6),
    x ≤ y → f x ≤ f y) ∧
  ABC.B = Real.pi/6 ∧
  (ABC.a = 1 ∨ ABC.a = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_triangle_problem_l1061_106154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_vertical_angles_not_hold_l1061_106173

-- Define the basic geometric concepts
structure Line : Type := (id : Nat)
structure Angle : Type := (measure : Real)
structure Triangle : Type := (id : Nat)
structure Point : Type := (x y : Real)

-- Define the geometric relations
def parallel (l1 l2 : Line) : Prop := sorry
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def vertical_angles (a1 a2 : Angle) : Prop := sorry
def congruent_triangles (t1 t2 : Triangle) : Prop := sorry
def corresponding_sides_equal (t1 t2 : Triangle) : Prop := sorry
def angle_bisector (l : Line) (a : Angle) : Prop := sorry
def equidistant_from_sides (p : Point) (a : Angle) : Prop := sorry

-- Define a membership relation for points on lines
def on_line (p : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem inverse_vertical_angles_not_hold :
  (∀ a1 a2 : Angle, vertical_angles a1 a2 → a1 = a2) ∧
  ¬(∀ a1 a2 : Angle, a1 = a2 → vertical_angles a1 a2) ∧
  (∀ l1 l2 : Line, (∀ a1 a2 : Angle, corresponding_angles a1 a2 l1 l2 → a1 = a2) → parallel l1 l2) ∧
  (∀ l1 l2 : Line, parallel l1 l2 → (∀ a1 a2 : Angle, corresponding_angles a1 a2 l1 l2 → a1 = a2)) ∧
  (∀ t1 t2 : Triangle, congruent_triangles t1 t2 → corresponding_sides_equal t1 t2) ∧
  (∀ t1 t2 : Triangle, corresponding_sides_equal t1 t2 → congruent_triangles t1 t2) ∧
  (∀ l : Line, ∀ a : Angle, angle_bisector l a → (∀ p : Point, on_line p l → equidistant_from_sides p a)) ∧
  (∀ l : Line, ∀ a : Angle, (∀ p : Point, on_line p l → equidistant_from_sides p a) → angle_bisector l a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_vertical_angles_not_hold_l1061_106173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_l1061_106149

/-- Represents the weight of green beans in kg -/
def green_beans_weight : ℝ := 60

/-- Represents the weight difference between green beans and rice in kg -/
def x : ℝ := 30

/-- Represents the weight of rice in kg -/
noncomputable def rice_weight : ℝ := green_beans_weight - x

/-- Represents the weight of sugar in kg -/
def sugar_weight : ℝ := green_beans_weight - 10

/-- Represents the remaining weight of rice after loss -/
noncomputable def remaining_rice : ℝ := (2/3) * rice_weight

/-- Represents the remaining weight of sugar after loss -/
noncomputable def remaining_sugar : ℝ := (4/5) * sugar_weight

/-- Theorem stating the weight difference between green beans and rice -/
theorem weight_difference : x = 30 := by
  -- Proof goes here
  sorry

#check weight_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_difference_l1061_106149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_name_decoding_correct_l1061_106141

/-- Represents a mapping from Russian alphabet letters to their positions --/
def russianAlphabetPosition : Char → Nat := sorry

/-- Represents the reverse mapping from positions to Russian alphabet letters --/
def russianAlphabetLetter : Nat → Char := sorry

/-- Encodes a string using Russian alphabet positions --/
def encodeRussian (name : String) : Nat := sorry

/-- Decodes a number into a string using Russian alphabet positions --/
def decodeRussian (code : Nat) : String := sorry

/-- The encoded number given in the problem --/
def encodedNumber : Nat := 2011533

/-- The name we want to prove as the correct decoding --/
def targetName : String := "Таня"

theorem name_decoding_correct : 
  decodeRussian encodedNumber = targetName ∧ 
  encodeRussian targetName = encodedNumber := by
  sorry

#eval encodedNumber
#eval targetName

end NUMINAMATH_CALUDE_ERRORFEEDBACK_name_decoding_correct_l1061_106141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_max_k_for_g_inequality_l1061_106194

-- Define the functions f and g
noncomputable def f (x : ℝ) := x - Real.log x - 2
noncomputable def g (x : ℝ) := x * Real.log x + x

-- Statement for part 1
theorem f_has_zero_in_interval : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = 0 := by sorry

-- Statement for part 2
theorem max_k_for_g_inequality :
  (∀ x : ℝ, x > 1 → g x > 3 * (x - 1)) ∧
  (∀ k : ℤ, k > 3 → ∃ x : ℝ, x > 1 ∧ g x ≤ ↑k * (x - 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_max_k_for_g_inequality_l1061_106194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangement_count_l1061_106176

theorem ring_arrangement_count : ℕ := by
  let total_rings : ℕ := 10
  let arranged_rings : ℕ := 6
  let fingers : ℕ := 4
  
  let choose_rings : ℕ := Nat.choose total_rings arranged_rings
  let order_rings : ℕ := Nat.factorial arranged_rings
  let distribute_rings : ℕ := Nat.choose (arranged_rings + fingers - 1) (fingers - 1)
  
  have h : choose_rings * order_rings * distribute_rings = 12660480 := by sorry
  
  exact 12660480

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangement_count_l1061_106176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_hyperbola_eccentricity_range_is_two_to_infinity_l1061_106166

theorem hyperbola_eccentricity_range (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ x^2 + y^2 = (c/2)^2) →
  c/a ≥ 2 :=
by
  intro h
  -- The proof would go here
  sorry

-- The main theorem that captures the problem statement
theorem hyperbola_eccentricity_range_is_two_to_infinity :
  {e : ℝ | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ x^2 + y^2 = (e*a/2)^2)} = {e : ℝ | e ≥ 2} :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_hyperbola_eccentricity_range_is_two_to_infinity_l1061_106166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1061_106133

/-- A function that checks if a positive integer contains the digit 0 --/
def containsZero (n : ℕ) : Prop := 
  ∃ k : ℕ, n / (10^k) % 10 = 0

/-- A function that checks if 10^n satisfies the property for a given n --/
def satisfiesProperty (n : ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a * b = 10^n → (containsZero a ∨ containsZero b)

/-- The theorem stating that 8 is the smallest n satisfying the property --/
theorem smallest_n_with_property :
  satisfiesProperty 8 ∧ ∀ m : ℕ, m < 8 → ¬satisfiesProperty m := by
  sorry

#check smallest_n_with_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1061_106133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equation_l1061_106151

/-- The sum of the series S = 1 + 3x + 5x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := (1 + x) / ((1 - x)^2)

/-- Theorem: If S(x) = 4, then x = (9 - √33) / 8 -/
theorem series_sum_equation (x : ℝ) (hx : x < 1) :
  S x = 4 → x = (9 - Real.sqrt 33) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equation_l1061_106151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_commute_length_l1061_106175

/-- Represents Liam's commute details -/
structure CommuteDetails where
  actualSpeed : ℚ  -- Actual speed in mph
  earlyArrival : ℚ  -- Early arrival time in hours
  speedReduction : ℚ  -- Speed reduction that would make him arrive on time

/-- Calculates the length of Liam's commute in miles -/
def calculateCommuteLength (details : CommuteDetails) : ℚ :=
  let onTimeSpeed := details.actualSpeed - details.speedReduction
  let onTimeTrip := details.actualSpeed * (1 - details.earlyArrival)
  onTimeSpeed * onTimeTrip / details.actualSpeed

/-- Theorem stating that Liam's commute is 10 miles long given the conditions -/
theorem liam_commute_length :
  let details : CommuteDetails := {
    actualSpeed := 30,
    earlyArrival := 1/15,  -- 4 minutes in hours (1/15 = 4/60)
    speedReduction := 5
  }
  calculateCommuteLength details = 10 := by
  sorry

#eval calculateCommuteLength {
  actualSpeed := 30,
  earlyArrival := 1/15,
  speedReduction := 5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liam_commute_length_l1061_106175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_number_properties_l1061_106114

theorem rational_number_properties (a b : ℚ) : 
  ((-a)^2 = a^2) ∧ 
  (a * b < 0 → |a + b| = abs (|a| - |b|)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_number_properties_l1061_106114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_sin_increasing_on_interval_exp2_strictly_increasing_l1061_106115

-- Define the function y = 2^(sin x)
noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x)

-- Define the interval [2kπ - π/2, 2kπ + π/2] for k ∈ ℤ
def monotonic_interval (k : ℤ) : Set ℝ := 
  Set.Icc (2 * (k : ℝ) * Real.pi - Real.pi / 2) (2 * (k : ℝ) * Real.pi + Real.pi / 2)

-- State the theorem
theorem f_monotonic_increasing :
  ∀ k : ℤ, StrictMonoOn f (monotonic_interval k) :=
by
  sorry

-- Additional lemma to show that sin x is increasing on the given interval
theorem sin_increasing_on_interval (k : ℤ) :
  StrictMonoOn Real.sin (monotonic_interval k) :=
by
  sorry

-- Lemma to show that 2^x is strictly increasing
theorem exp2_strictly_increasing :
  StrictMono (fun x => 2^x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_sin_increasing_on_interval_exp2_strictly_increasing_l1061_106115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1061_106106

-- Define the basic geometric objects
variable (Point Plane Line : Type)

-- Define the geometric relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (point_outside_plane : Point → Plane → Prop)
variable (line_outside_plane : Line → Plane → Prop)
variable (plane_through_point : Plane → Point → Prop)
variable (plane_through_line : Plane → Line → Prop)

-- State the theorem
theorem geometric_propositions :
  -- Proposition 1 is false
  ¬ (∀ (l1 l2 : Line) (p : Plane), 
    parallel_line_plane l1 p → parallel_line_plane l2 p → parallel_lines l1 l2) ∧
  -- Proposition 2 is false
  ¬ (∀ (l1 l2 l3 : Line),
    perpendicular_lines l1 l3 → perpendicular_lines l2 l3 → parallel_lines l1 l2) ∧
  -- Proposition 3 is true
  (∀ (pt : Point) (p : Plane),
    point_outside_plane pt p → 
    ∃! (p' : Plane), plane_through_point p' pt ∧ parallel_planes p' p) ∧
  -- Proposition 4 is false
  ¬ (∀ (l : Line) (p : Plane),
    line_outside_plane l p → 
    ∃ (p' : Plane), plane_through_line p' l ∧ parallel_planes p' p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1061_106106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_meal_capacity_l1061_106104

-- Define the meal capacity in terms of adults and children
def meal_capacity_adults : ℕ := 70
def meal_capacity_children : ℕ := 90

-- Define the number of adults who have eaten
def adults_eaten : ℕ := 21

-- Define the ratio of food consumed by an adult compared to a child
def adult_child_ratio : ℚ := 9 / 7

-- Theorem statement
theorem remaining_meal_capacity : 
  meal_capacity_children - (adults_eaten * (Rat.num adult_child_ratio / Rat.den adult_child_ratio)) = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_meal_capacity_l1061_106104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_principle_circular_mirror_l1061_106122

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An ellipse in 2D space -/
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)

/-- Determines if a point is inside a circle -/
def isInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

/-- Determines if a point is on the circumference of a circle -/
def isOnCircumference (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Determines if an ellipse is tangent to a circle at a point -/
def isTangent (e : Ellipse) (c : Circle) (p : ℝ × ℝ) : Prop :=
  sorry  -- Definition of tangency

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

/-- Determines if a point extremizes the sum of distances to two given points -/
def isExtremum (a b p : ℝ × ℝ) : Prop :=
  sorry  -- Definition of extremum condition

theorem fermat_principle_circular_mirror (c : Circle) (a b : ℝ × ℝ) 
    (ha : isInside c a) (hb : isInside c b) :
    ∃ p : ℝ × ℝ, isOnCircumference c p ∧ 
    isExtremum a b p ∧ 
    ∃ e : Ellipse, e.foci = (a, b) ∧ isTangent e c p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_principle_circular_mirror_l1061_106122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1061_106123

theorem cos_theta_value (θ : ℝ) (P : ℝ × ℝ) :
  P.1 = -3/5 →
  P.2 = 4/5 →
  P.1^2 + P.2^2 = 1 →
  Real.cos θ = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l1061_106123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_soda_consumption_l1061_106161

/-- Tom's daily soda consumption in cans -/
def daily_soda_cans : ℕ := 5

/-- Tom's daily water consumption in ounces -/
def daily_water_oz : ℕ := 64

/-- Tom's weekly fluid consumption in ounces -/
def weekly_fluid_oz : ℕ := 868

/-- Size of a soda can in ounces -/
def soda_can_oz : ℕ := 12

/-- Number of days in a week -/
def days_in_week : ℕ := 7

theorem tom_soda_consumption :
  daily_soda_cans * soda_can_oz * days_in_week + daily_water_oz * days_in_week = weekly_fluid_oz :=
by
  -- Convert all values to natural numbers
  have h1 : (5 : ℕ) * (12 : ℕ) * (7 : ℕ) + (64 : ℕ) * (7 : ℕ) = (868 : ℕ) := by norm_num
  -- Use the equality
  exact h1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_soda_consumption_l1061_106161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derived_from_g_f_eq_x_unique_root_a_monotone_increasing_l1061_106144

/- Given function g(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1/2

/- Parameter m -/
variable (m : ℝ)

/- Hypothesis on m -/
variable (hm : 0 < m ∧ m < 1/2)

/- Derived function f(x) -/
noncomputable def f (x : ℝ) : ℝ := m * Real.sin x + 1

/- Sequence definition -/
noncomputable def a : ℕ → ℝ
  | 0 => 0
  | n + 1 => f m (a n)

/- Theorem statements -/
theorem f_derived_from_g : ∃ (h : ℝ → ℝ), 
  (∀ x, h x = Real.sin (x/2)) ∧ 
  (∀ x, f m x = m * h (x - π/6) + 1) := by sorry

theorem f_eq_x_unique_root : ∃! x, f m x = x := by sorry

theorem a_monotone_increasing : ∀ n, n > 0 → a m n < a m (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derived_from_g_f_eq_x_unique_root_a_monotone_increasing_l1061_106144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_major_axis_is_50_l1061_106102

/-- The length of the major axis of an ellipse with given foci and y-axis tangency -/
theorem ellipse_major_axis_length (F1 F2 : ℝ × ℝ) (tangent_to_y_axis : Prop) : ℝ :=
  let major_axis_length : ℝ := 50
  major_axis_length

/-- First focus of the ellipse -/
def F1 : ℝ × ℝ := (5, 10)

/-- Second focus of the ellipse -/
def F2 : ℝ × ℝ := (35, 40)

/-- The ellipse is tangent to the y-axis -/
def tangent_to_y_axis : Prop := sorry

/-- Proof that the major axis length is 50 -/
theorem major_axis_is_50 : ellipse_major_axis_length F1 F2 tangent_to_y_axis = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_major_axis_is_50_l1061_106102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_l1061_106111

noncomputable section

def triangle_OPQ : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := (0, 0, 3, 0, 3, 3)

def angle_PQO : ℝ := Real.pi / 2
def angle_POQ : ℝ := Real.pi / 4

def rotation_angle : ℝ := Real.pi / 3
def translation_distance : ℝ := 4

def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

def translate_point (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

theorem triangle_transformation :
  let (Ox, Oy, Qx, Qy, Px, Py) := triangle_OPQ
  let P' := rotate_point (Px - Ox, Py - Oy) rotation_angle
  let P'' := translate_point P' translation_distance
  P'' = ((11 - 3 * Real.sqrt 3) / 2, (3 * Real.sqrt 3 + 3) / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_transformation_l1061_106111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_area_theorem_l1061_106169

/-- A triangle with its three midlines -/
structure TriangleWithMidlines where
  -- The triangle itself
  triangle : Set (ℝ × ℝ)
  -- The three midlines
  midline1 : Set (ℝ × ℝ)
  midline2 : Set (ℝ × ℝ)
  midline3 : Set (ℝ × ℝ)
  -- Condition that the midlines divide the triangle into four parts
  divides_into_four : triangle ∩ (midline1 ∪ midline2 ∪ midline3) = ∅

/-- The area of a set in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

/-- The theorem stating the relationship between the area of a part and the whole triangle -/
theorem midline_area_theorem (t : TriangleWithMidlines) (S : ℝ) :
  (∃ part : Set (ℝ × ℝ), part ⊆ t.triangle ∧ area part = S) →
  area t.triangle = 4 * S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_area_theorem_l1061_106169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_worked_l1061_106183

noncomputable def normal_wage : ℚ := 12
noncomputable def overtime_rate : ℚ := 3/2
noncomputable def regular_hours : ℚ := 40
noncomputable def total_paycheck : ℚ := 696

noncomputable def calculate_total_hours (normal_wage : ℚ) (overtime_rate : ℚ) (regular_hours : ℚ) (total_paycheck : ℚ) : ℚ :=
  let overtime_wage := normal_wage * overtime_rate
  let regular_pay := normal_wage * regular_hours
  let overtime_pay := total_paycheck - regular_pay
  let overtime_hours := overtime_pay / overtime_wage
  regular_hours + overtime_hours

theorem total_hours_worked :
  calculate_total_hours normal_wage overtime_rate regular_hours total_paycheck = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_hours_worked_l1061_106183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_count_l1061_106165

/-- The number of men required to complete a work in a given time -/
noncomputable def number_of_men (work : ℝ) (time : ℝ) (days : ℝ) : ℝ :=
  work / (time * days)

/-- Theorem stating that the number of men is 56 given the conditions -/
theorem men_count (work : ℝ) :
  ∃ M : ℝ, number_of_men work M 60 = number_of_men work (M - 8) 70 ∧ M = 56 := by
  -- We'll use 56 as our witness for M
  use 56
  -- Split the goal into two parts
  constructor
  -- Prove the equation holds
  · simp [number_of_men]
    -- Simplify the fractions
    field_simp
    -- The equation is true (can be checked numerically)
    sorry
  -- Prove M = 56 (trivial as we used 56)
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_count_l1061_106165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_real_l1061_106168

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) = ln(ae^x - x + 2a^2 - 3) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * Real.exp x - x + 2 * a^2 - 3)

/-- The theorem stating that the domain of f is ℝ iff a ∈ (-∞, 1] -/
theorem domain_of_f_is_real (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_real_l1061_106168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1061_106180

theorem sin_half_angle_special_case (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1061_106180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_profit_percentage_l1061_106127

noncomputable def tea_mixture (weight1 : ℝ) (cost1 : ℝ) (weight2 : ℝ) (cost2 : ℝ) (sale_price : ℝ) : ℝ :=
  let total_cost := weight1 * cost1 + weight2 * cost2
  let total_weight := weight1 + weight2
  let cost_price := total_cost / total_weight
  let profit_per_kg := sale_price - cost_price
  (profit_per_kg / cost_price) * 100

theorem tea_profit_percentage :
  tea_mixture 80 15 20 20 22.4 = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_profit_percentage_l1061_106127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_larger_than_sqrt5_plus_sqrt3_to_4th_l1061_106108

theorem smallest_integer_larger_than_sqrt5_plus_sqrt3_to_4th :
  ∃ n : ℤ, (n = 248 ∧ (∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^4 → m ≥ n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_larger_than_sqrt5_plus_sqrt3_to_4th_l1061_106108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1061_106184

universe u

-- Define the universe of discourse
variable (Student : Type u)

-- Define the predicates
variable (IsFromUniversity : Student → Prop)
variable (IsResident : Student → Prop)

-- Define the original statement
def AllNonResidents (Student : Type u) (IsFromUniversity IsResident : Student → Prop) : Prop :=
  ∀ s : Student, IsFromUniversity s → ¬(IsResident s)

-- Define the negation
def SomeResidents (Student : Type u) (IsFromUniversity IsResident : Student → Prop) : Prop :=
  ∃ s : Student, IsFromUniversity s ∧ IsResident s

-- Theorem statement
theorem negation_equivalence (Student : Type u) (IsFromUniversity IsResident : Student → Prop) :
  ¬(AllNonResidents Student IsFromUniversity IsResident) ↔ SomeResidents Student IsFromUniversity IsResident :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1061_106184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_purchase_l1061_106143

/-- Given a total amount of money and the price of a single item, 
    calculate the maximum number of whole items that can be purchased. -/
def maxPurchase (totalMoney : ℚ) (itemPrice : ℚ) : ℕ :=
  (totalMoney / itemPrice).floor.toNat

/-- Prove that with $9.00 and a card price of $0.95, 
    the maximum number of cards that can be purchased is 9. -/
theorem max_cards_purchase : maxPurchase 9 (95/100) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_purchase_l1061_106143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1061_106134

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 2

-- Part 1
theorem part1 (a : ℝ) : (deriv (f a)) 0 = 1 → a = 0 := by sorry

-- Part 2
def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y, x < y → 
    ((x < Real.log a ∧ y < Real.log a → f a x > f a y) ∧
     (x > Real.log a ∧ y > Real.log a → f a x < f a y)))

theorem part2 (a : ℝ) : monotonic_intervals a := by sorry

-- Part 3
def condition (k : ℤ) : Prop :=
  ∀ x : ℝ, x > 0 → (x - ↑k) * (Real.exp x - 1) + x + 1 > 0

theorem part3 : ∀ k : ℤ, condition k → k ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l1061_106134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_prices_and_packaging_l1061_106120

-- Define variables for unit prices
variable (red_bean_price : ℚ) (meat_price : ℚ)

-- Define variables for discounted prices
variable (discounted_red_bean_price : ℚ) (discounted_meat_price : ℚ)

-- Define variable for package composition
variable (m : ℕ)

-- Define the conditions
def condition_1 (red_bean_price meat_price : ℚ) : Prop := 
  10 * red_bean_price + 12 * meat_price = 136

def condition_2 (red_bean_price meat_price : ℚ) : Prop := 
  meat_price = 2 * red_bean_price

def condition_3 (discounted_red_bean_price discounted_meat_price : ℚ) : Prop := 
  20 * discounted_red_bean_price + 30 * discounted_meat_price = 270

def condition_4 (discounted_red_bean_price discounted_meat_price : ℚ) : Prop := 
  30 * discounted_red_bean_price + 20 * discounted_meat_price = 230

def condition_5 (m : ℕ) : Prop := 
  (3 * m + 7 * (40 - m)) * (80 - 4 * m) + 
  (3 * (40 - m) + 7 * m) * (4 * m + 8) = 17280

def condition_6 (m : ℕ) : Prop := m ≤ 20

-- Theorem statement
theorem zongzi_prices_and_packaging :
  ∀ (red_bean_price meat_price discounted_red_bean_price discounted_meat_price : ℚ) (m : ℕ),
  condition_1 red_bean_price meat_price →
  condition_2 red_bean_price meat_price →
  condition_3 discounted_red_bean_price discounted_meat_price →
  condition_4 discounted_red_bean_price discounted_meat_price →
  condition_5 m →
  condition_6 m →
  red_bean_price = 4 ∧ meat_price = 8 ∧ 
  discounted_red_bean_price = 3 ∧ discounted_meat_price = 7 ∧
  m = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_prices_and_packaging_l1061_106120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1061_106146

-- Sequence 1
noncomputable def sequence1 (n : ℕ) : ℝ := Real.sqrt (3 * n - 1)

-- Sequence 2
def sequence2 : ℕ → ℝ
| 0 => 3  -- Adding case for 0
| 1 => 3
| 2 => 6
| (n + 3) => sequence2 (n + 2) - sequence2 (n + 1)

-- Arithmetic sequence 1
def arithmeticSequence1 (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Arithmetic sequence 2
def arithmeticSequence2 (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem all_statements_correct :
  (∀ n, sequence1 n = Real.sqrt (3 * n - 1)) ∧
  (sequence2 5 = -6) ∧
  (∃ a d, arithmeticSequence1 a d 3 + arithmeticSequence1 a d 4 + arithmeticSequence1 a d 5 + 
          arithmeticSequence1 a d 6 + arithmeticSequence1 a d 7 = 450 ∧
          arithmeticSequence1 a d 2 + arithmeticSequence1 a d 8 = 180) ∧
  (∃ a d, arithmeticSequence2 a d 2 = 1 ∧ arithmeticSequence2 a d 4 = 5 ∧
          (arithmeticSequence2 a d 1 + arithmeticSequence2 a d 2 + arithmeticSequence2 a d 3 + 
           arithmeticSequence2 a d 4 + arithmeticSequence2 a d 5) = 15) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_correct_l1061_106146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenland_111_l1061_106178

/-- Represents the Evenland number system --/
structure EvenlandNumber where
  value : ℕ

/-- Converts a natural number to its Evenland representation --/
def toEvenland (n : ℕ) : EvenlandNumber :=
  { value := sorry }

/-- The base of the Evenland number system --/
def evenlandBase : ℕ := 5

/-- Theorem: The Evenland representation of 111 is 842 --/
theorem evenland_111 : (toEvenland 111).value = 842 := by
  sorry

/-- Instance for OfNat EvenlandNumber --/
instance : OfNat EvenlandNumber n where
  ofNat := ⟨n⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenland_111_l1061_106178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edge_length_l1061_106170

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  x : ℝ  -- side length of the base
  h : ℝ  -- height of the tetrahedron

/-- The conditions for our specific tetrahedron -/
def TetrahedronConditions (t : Tetrahedron) : Prop :=
  t.h = t.x + 1 ∧  -- height is 1 unit greater than base side
  Real.sqrt 3 * t.x^2 ≥ 100  -- surface area is at least 100 square units

/-- The edge length of the tetrahedron -/
def edgeLength (t : Tetrahedron) : ℝ := t.x + 1

/-- Theorem stating the minimum edge length of the tetrahedron -/
theorem min_edge_length : 
  ∀ t : Tetrahedron, TetrahedronConditions t → 
  edgeLength t ≥ 10 * Real.sqrt 3 / 3 + 1 := by
  sorry

#check min_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_edge_length_l1061_106170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l1061_106172

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance between the center and a focus of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

theorem ellipse_eccentricity_equilateral_triangle (e : Ellipse) :
  focal_distance e = e.a / 2 →
  eccentricity e = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_equilateral_triangle_l1061_106172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1061_106196

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line ax + y + 1 = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a

/-- The slope of the line (a+2)x - 3y - 2 = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := (a + 2) / 3

/-- The condition that a = 1 is sufficient but not necessary for perpendicularity -/
theorem sufficient_not_necessary : 
  (∀ a : ℝ, a = 1 → perpendicular (slope1 a) (slope2 a)) ∧ 
  (∃ a : ℝ, a ≠ 1 ∧ perpendicular (slope1 a) (slope2 a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1061_106196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l1061_106199

/-- Represents a sequence of positive integers (a₀, a₁, a₂, a₃) satisfying the given conditions -/
structure Solution where
  a₀ : ℕ+
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  h_order : a₀ > a₁ ∧ a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > 1
  h_equation : (1 - 1 / (a₁ : ℚ)) + (1 - 1 / (a₂ : ℚ)) + (1 - 1 / (a₃ : ℚ)) = 2 * (1 - 1 / (a₀ : ℚ))

/-- The only valid solutions are (24, 4, 3, 2) and (60, 5, 3, 2) -/
theorem unique_solutions :
  ∀ s : Solution, (s.a₀ = 24 ∧ s.a₁ = 4 ∧ s.a₂ = 3 ∧ s.a₃ = 2) ∨
                  (s.a₀ = 60 ∧ s.a₁ = 5 ∧ s.a₂ = 3 ∧ s.a₃ = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solutions_l1061_106199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1061_106190

/-- The polynomial function representing 7x^4 - 16x^3 + 3x^2 - 5x - 20 -/
def f (x : ℝ) : ℝ := 7*x^4 - 16*x^3 + 3*x^2 - 5*x - 20

/-- The divisor function representing 2x - 4 -/
def g (x : ℝ) : ℝ := 2*x - 4

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + (-34) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l1061_106190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l1061_106156

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (Complex.I) / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l1061_106156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_speed_difference_l1061_106171

/-- The difference in speed between two ferries --/
theorem ferry_speed_difference : ∀ (speed_p speed_q : ℝ),
  speed_p = 8 →                         -- Ferry P's speed is 8 km/h
  speed_p < speed_q →                   -- Ferry P is slower than Ferry Q
  speed_p * 3 * 2 = speed_q * 4 →       -- Q's route is twice as long and takes 1 hour more
  speed_q - speed_p = 4 := by
  intros speed_p speed_q h1 h2 h3
  -- Proof steps would go here
  sorry

#check ferry_speed_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_speed_difference_l1061_106171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_for_given_lawn_l1061_106162

/-- Calculates the time required to mow a rectangular lawn -/
noncomputable def mowing_time (lawn_length : ℝ) (lawn_width : ℝ) (effective_swath_width : ℝ) (mowing_speed : ℝ) : ℝ :=
  (lawn_length * lawn_width) / (effective_swath_width * mowing_speed)

theorem mowing_time_for_given_lawn :
  let lawn_length : ℝ := 100
  let lawn_width : ℝ := 200
  let effective_swath_width : ℝ := 2
  let mowing_speed : ℝ := 5000
  mowing_time lawn_length lawn_width effective_swath_width mowing_speed = 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_time_for_given_lawn_l1061_106162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_rate_at_30_l1061_106112

-- Define the evaporation rate function as noncomputable
noncomputable def evaporation_rate (a b : ℝ) (x : ℝ) : ℝ := Real.exp (a * x + b)

-- State the theorem
theorem evaporation_rate_at_30 (a b : ℝ) :
  evaporation_rate a b 10 = 0.2 →
  evaporation_rate a b 20 = 0.4 →
  evaporation_rate a b 30 = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_rate_at_30_l1061_106112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_perimeter_l1061_106137

/-- A regular decagon is a polygon with 10 sides of equal length -/
structure RegularDecagon (α : Type*) [LinearOrderedField α] where
  side_length : α

/-- The perimeter of a regular decagon is 10 times the length of one side -/
def perimeter {α : Type*} [LinearOrderedField α] (d : RegularDecagon α) : α := 10 * d.side_length

theorem regular_decagon_perimeter :
  ∀ (d : RegularDecagon ℝ), d.side_length = 2.5 → perimeter d = 25 := by
  intro d h
  unfold perimeter
  rw [h]
  norm_num
  
#check regular_decagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_perimeter_l1061_106137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_endpoint_l1061_106142

/-- Given two points A and B in a 2D plane, and a point C such that BC is half the length of AB,
    this theorem proves that C has specific coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (4, -4) → 
  B = (18, 6) → 
  ‖C - B‖ = (1/2) * ‖B - A‖ → 
  C = (25, 11) := by
  sorry

#check extended_segment_endpoint

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_endpoint_l1061_106142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l1061_106139

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^(1/3) * x^2
def g (x : ℝ) : ℝ := x

-- State the theorem
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by
  intro x
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp [Real.rpow_mul, Real.rpow_add]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equiv_g_l1061_106139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_even_integers_sum_300_l1061_106186

theorem max_product_of_even_integers_sum_300 : 
  ∃ (a b : ℤ), 
    Even a ∧ 
    Even b ∧ 
    a + b = 300 ∧ 
    a * b = 22500 ∧ 
    ∀ (x y : ℤ), Even x → Even y → x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_even_integers_sum_300_l1061_106186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l1061_106198

/-- Rotate a point around another point by a given angle (in degrees) -/
def rotate_point (P Q : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  sorry

theorem rotation_equivalence (P Q R : ℝ × ℝ) (y : ℝ) :
  (rotate_point P Q 780 = R) →
  (rotate_point P Q (-y) = R) →
  y < 360 →
  y = 300 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l1061_106198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_visited_LakeQinghai_l1061_106147

-- Define the set of places
inductive Place
  | LakeQinghai
  | RapeseedFlowerSea
  | TeaCardSkyMirror

-- Define the set of tourists
inductive Tourist
  | A
  | B
  | C

-- Define a function that represents whether a tourist has visited a place
def hasVisited : Tourist → Place → Prop := sorry

-- Axioms based on the given conditions
axiom A_more_than_B : ∃ (p : Place), hasVisited Tourist.A p ∧ ¬hasVisited Tourist.B p

axiom A_not_RapeseedFlowerSea : ¬hasVisited Tourist.A Place.RapeseedFlowerSea

axiom B_not_TeaCardSkyMirror : ¬hasVisited Tourist.B Place.TeaCardSkyMirror

axiom all_visited_same : ∃ (p : Place), hasVisited Tourist.A p ∧ hasVisited Tourist.B p ∧ hasVisited Tourist.C p

-- Theorem to prove
theorem B_visited_LakeQinghai : hasVisited Tourist.B Place.LakeQinghai := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_visited_LakeQinghai_l1061_106147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_literate_male_percentage_l1061_106167

theorem literate_male_percentage
  (total_inhabitants : ℕ)
  (male_percentage : ℚ)
  (total_literate_percentage : ℚ)
  (female_literate_percentage : ℚ)
  (h1 : total_inhabitants = 1000)
  (h2 : male_percentage = 60 / 100)
  (h3 : total_literate_percentage = 25 / 100)
  (h4 : female_literate_percentage = 325 / 1000) :
  let female_inhabitants := total_inhabitants - (male_percentage * ↑total_inhabitants).floor
  let total_literate := (total_literate_percentage * ↑total_inhabitants).floor
  let literate_females := (female_literate_percentage * ↑female_inhabitants).floor
  let literate_males := total_literate - literate_females
  let male_inhabitants := (male_percentage * ↑total_inhabitants).floor
  (literate_males : ℚ) / male_inhabitants = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_literate_male_percentage_l1061_106167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l1061_106187

theorem trigonometric_system_solution :
  ∀ (x y z : ℝ),
  (∀ (a : ℝ), abs (a + 1/a) ≥ 2) →
  (∀ (a : ℝ), abs (a + 1/a) = 2 ↔ abs a = 1) →
  (Real.tan x = 1 ∨ Real.tan x = -1) ∧ Real.sin y = 1 →
  Real.cos z = 0 →
  ∃ (n k l : ℤ),
    x = π/4 + π/2 * ↑n ∧
    y = π/2 + 2*π * ↑k ∧
    z = π/2 + π * ↑l :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l1061_106187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_ratio_l1061_106107

open Real Set

theorem max_sin_ratio (α β : ℝ) (h1 : α ∈ Icc (π/4) (π/3)) 
  (h2 : β ∈ Icc (π/2) π) (h3 : sin (α + β) - sin α = 2 * sin α * cos β) :
  ∃ M, M = Real.sqrt 2 ∧ ∀ x ∈ Icc (π/4) (π/3), sin (2*x) / sin (β - x) ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_ratio_l1061_106107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_false_iff_a_geq_sqrt2_l1061_106163

theorem proposition_false_iff_a_geq_sqrt2 (a : ℝ) :
  (¬ ∃ x₀ ∈ Set.Icc 0 (π / 4), Real.sin (2 * x₀) + Real.cos (2 * x₀) > a) ↔ a ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_false_iff_a_geq_sqrt2_l1061_106163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_to_binary_digits_l1061_106159

theorem hex_to_binary_digits (A B C D : Nat) : 
  A < 16 ∧ B < 16 ∧ C < 16 ∧ D < 16 →
  (A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0 ≥ 2^15) ∧
  (A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0 < 2^16) →
  Nat.log2 (A * 16^3 + B * 16^2 + C * 16^1 + D * 16^0) + 1 = 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_to_binary_digits_l1061_106159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l1061_106105

/-- Calculates the upstream distance given downstream distance, time, and still water speed -/
noncomputable def upstream_distance (downstream_distance : ℝ) (time : ℝ) (still_water_speed : ℝ) : ℝ :=
  (still_water_speed - (downstream_distance / time - still_water_speed)) * time

/-- Theorem: Given the conditions, the upstream distance is 18 km -/
theorem upstream_distance_is_18 :
  let downstream_distance : ℝ := 30
  let time : ℝ := 6
  let still_water_speed : ℝ := 4
  upstream_distance downstream_distance time still_water_speed = 18 := by
  -- Unfold the definition of upstream_distance
  unfold upstream_distance
  -- Simplify the expression
  simp
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l1061_106105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1061_106158

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 10*y + 13 = 0

-- Define the area of the region
noncomputable def region_area : ℝ := 16 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof steps would go here
  sorry

-- Additional lemma to show that the region is indeed a circle
lemma region_is_circle :
  ∃ (center_x center_y radius : ℝ),
    ∀ x y, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1061_106158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l1061_106155

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Proves that a train with given parameters takes 50 seconds to pass a bridge -/
theorem train_bridge_passage_time :
  time_to_pass_bridge 485 140 45 = 50 := by
  sorry

-- Use #eval only for computable functions
def approx_time_to_pass_bridge (train_length : Float) (bridge_length : Float) (train_speed_kmh : Float) : Float :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

#eval approx_time_to_pass_bridge 485 140 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l1061_106155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1061_106125

def f (b c x : ℝ) : ℝ := x^3 + b*x + c

theorem unique_root_in_interval (b c : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, ∀ y ∈ Set.Icc (-1) 1, x < y → f b c x < f b c y) →
  f b c (-1/2) * f b c (1/2) < 0 →
  ∃! x, x ∈ Set.Icc (-1) 1 ∧ f b c x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1061_106125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l1061_106181

def d (m : ℕ) : ℕ := (Nat.divisors m).card

def a (c : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => d (a c n) + c

theorem sequence_eventually_periodic (c : ℕ) :
  ∃ k : ℕ, ∃ p : ℕ+, ∀ n : ℕ, a c (k + n) = a c (k + n + p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l1061_106181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_swans_are_white_l1061_106192

-- Define Swan and White as variables of type (α → Prop)
variable {α : Type*}
variable (Swan : α → Prop)
variable (White : α → Prop)

-- Theorem statement
theorem negation_of_all_swans_are_white :
  (¬ ∀ x : α, Swan x → White x) ↔ (∃ x : α, Swan x ∧ ¬ White x) := by
  -- Proof is omitted using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_swans_are_white_l1061_106192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_l1061_106130

/-- Represents the capital invested by two business partners -/
structure BusinessCapital where
  pyarelal : ℚ
  ashok : ℚ
  ashok_ratio : ashok = pyarelal / 9

/-- Calculates the loss distribution between two investors -/
def loss_distribution (capital : BusinessCapital) (total_loss : ℚ) : ℚ :=
  (capital.pyarelal / (capital.pyarelal + capital.ashok)) * total_loss

/-- Theorem: Pyarelal's share of the loss is 900 when the total loss is 1000 -/
theorem pyarelal_loss (capital : BusinessCapital) :
  loss_distribution capital 1000 = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyarelal_loss_l1061_106130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_supporters_count_is_13_l1061_106189

def total_attendance : ℕ := 50
def first_team_percentage : ℚ := 40 / 100
def second_team_percentage : ℚ := 34 / 100

def first_team_supporters : ℕ := (first_team_percentage * total_attendance).floor.toNat
def second_team_supporters : ℕ := (second_team_percentage * total_attendance).floor.toNat

def non_supporters_count : ℕ := total_attendance - (first_team_supporters + second_team_supporters)

theorem non_supporters_count_is_13 : non_supporters_count = 13 := by
  -- Unfold definitions
  unfold non_supporters_count
  unfold first_team_supporters
  unfold second_team_supporters
  unfold first_team_percentage
  unfold second_team_percentage
  unfold total_attendance
  
  -- Evaluate expressions
  simp
  
  -- The proof steps would go here
  sorry

#eval non_supporters_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_supporters_count_is_13_l1061_106189
