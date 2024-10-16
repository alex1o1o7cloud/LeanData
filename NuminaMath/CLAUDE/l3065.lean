import Mathlib

namespace NUMINAMATH_CALUDE_linear_system_det_proof_l3065_306504

/-- Given a linear equation system represented by an augmented matrix,
    prove that the determinant of a specific matrix using the solution is -1 -/
theorem linear_system_det_proof (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  a₁ = 2 ∧ b₁ = 0 ∧ c₁ = 2 ∧ a₂ = 3 ∧ b₂ = 1 ∧ c₂ = 2 →
  ∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ →
  x * 2 - y * (-3) = -1 := by
sorry


end NUMINAMATH_CALUDE_linear_system_det_proof_l3065_306504


namespace NUMINAMATH_CALUDE_exists_graph_with_short_paths_l3065_306571

/-- A directed graph with n vertices -/
def DirectedGraph (n : ℕ) := Fin n → Fin n → Prop

/-- A path of length at most 2 exists between two vertices in a directed graph -/
def HasPathAtMost2 (G : DirectedGraph n) (u v : Fin n) : Prop :=
  G u v ∨ ∃ w, G u w ∧ G w v

/-- For any n > 4, there exists a directed graph with n vertices
    such that any two vertices have a path of length at most 2 between them -/
theorem exists_graph_with_short_paths (n : ℕ) (h : n > 4) :
  ∃ G : DirectedGraph n, ∀ u v : Fin n, HasPathAtMost2 G u v :=
sorry

end NUMINAMATH_CALUDE_exists_graph_with_short_paths_l3065_306571


namespace NUMINAMATH_CALUDE_special_polynomial_n_is_two_l3065_306555

/-- A polynomial of degree 2n satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) ∧
  (p (2 * n + 1) = -30)

/-- The theorem stating that n must be 2 for the given conditions -/
theorem special_polynomial_n_is_two :
  ∀ p : ℝ → ℝ, ∀ n : ℕ, SpecialPolynomial p n → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_n_is_two_l3065_306555


namespace NUMINAMATH_CALUDE_nicky_running_time_l3065_306593

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_length : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_length = 500)
  (h2 : cristina_speed > nicky_speed)
  (h3 : head_start = 12)
  (h4 : cristina_speed = 5)
  (h5 : nicky_speed = 3) :
  head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed) = 30 := by
  sorry

#check nicky_running_time

end NUMINAMATH_CALUDE_nicky_running_time_l3065_306593


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3065_306544

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- Define set B
def B : Set ℝ := {x | Real.log x < 0}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3065_306544


namespace NUMINAMATH_CALUDE_unique_solution_l3065_306548

/-- Represents a digit in the equation --/
def Digit := Fin 10

/-- The equation is valid if it satisfies all conditions --/
def is_valid_equation (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  100 * A.val + 10 * C.val + A.val + 
  100 * B.val + 10 * B.val + D.val = 
  1000 * A.val + 100 * B.val + 10 * C.val + D.val

/-- There exists a unique solution to the equation --/
theorem unique_solution : 
  ∃! (A B C D : Digit), is_valid_equation A B C D ∧ 
    A.val = 9 ∧ B.val = 8 ∧ C.val = 0 ∧ D.val = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3065_306548


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3065_306592

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ (a*x^2 + 5*x + c > 0)) → 
  (a = -6 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3065_306592


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3065_306502

theorem polygon_sides_count (n : ℕ) (exterior_angle : ℝ) : 
  (n ≥ 2) →
  (exterior_angle > 0) →
  (exterior_angle < 45) →
  (n * exterior_angle = 360) →
  (n ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3065_306502


namespace NUMINAMATH_CALUDE_existence_of_sequences_l3065_306505

theorem existence_of_sequences : ∃ (a b : ℕ → ℝ), 
  (∀ i : ℕ, 3 * Real.pi / 2 ≤ a i ∧ a i ≤ b i) ∧
  (∀ i : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a i * x) - Real.cos (b i * x) ≥ -1 / i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sequences_l3065_306505


namespace NUMINAMATH_CALUDE_circle_area_sum_l3065_306595

/-- The sum of the areas of all circles in an infinite sequence, where the radii form a geometric
    sequence with first term 10/3 and common ratio 4/9, is equal to 180π/13. -/
theorem circle_area_sum : 
  let r₁ : ℝ := 10 / 3  -- First term of the radii sequence
  let r : ℝ := 4 / 9    -- Common ratio of the radii sequence
  let area_sum := ∑' n, π * (r₁ * r ^ n) ^ 2  -- Sum of areas of all circles
  area_sum = 180 * π / 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_sum_l3065_306595


namespace NUMINAMATH_CALUDE_correct_ranking_l3065_306524

-- Define the cities
inductive City
| Dover
| Eden
| Fairview

-- Define the growth rate comparison relation
def higherGrowthRate : City → City → Prop := sorry

-- Define the statements
def statement1 : Prop := higherGrowthRate City.Dover City.Eden ∧ higherGrowthRate City.Dover City.Fairview
def statement2 : Prop := ¬(higherGrowthRate City.Eden City.Dover ∧ higherGrowthRate City.Eden City.Fairview)
def statement3 : Prop := ¬(higherGrowthRate City.Dover City.Fairview ∧ higherGrowthRate City.Eden City.Fairview)

-- Theorem stating the correct ranking
theorem correct_ranking :
  (statement1 ∨ statement2 ∨ statement3) ∧
  (statement1 → ¬statement2 ∧ ¬statement3) ∧
  (statement2 → ¬statement1 ∧ ¬statement3) ∧
  (statement3 → ¬statement1 ∧ ¬statement2) →
  higherGrowthRate City.Eden City.Dover ∧
  higherGrowthRate City.Dover City.Fairview :=
sorry

end NUMINAMATH_CALUDE_correct_ranking_l3065_306524


namespace NUMINAMATH_CALUDE_area_ratio_is_three_thirtyseconds_l3065_306598

/-- Triangle PQR with points X, Y, Z on its sides -/
structure TriangleWithPoints where
  -- Side lengths of triangle PQR
  pq : ℝ
  qr : ℝ
  rp : ℝ
  -- Ratios for points X, Y, Z
  x : ℝ
  y : ℝ
  z : ℝ
  -- Conditions
  pq_eq : pq = 7
  qr_eq : qr = 24
  rp_eq : rp = 25
  x_pos : x > 0
  y_pos : y > 0
  z_pos : z > 0
  sum_eq : x + y + z = 3/4
  sum_sq_eq : x^2 + y^2 + z^2 = 3/8

/-- The ratio of areas of triangle XYZ to triangle PQR -/
def areaRatio (t : TriangleWithPoints) : ℚ :=
  3/32

/-- Theorem stating that the area ratio is 3/32 -/
theorem area_ratio_is_three_thirtyseconds (t : TriangleWithPoints) :
  areaRatio t = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_thirtyseconds_l3065_306598


namespace NUMINAMATH_CALUDE_range_of_m_l3065_306589

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*m*x + 9 ≥ 0) → m ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3065_306589


namespace NUMINAMATH_CALUDE_weight_problem_l3065_306551

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 41 kg. -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 27 →
  (b + c) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l3065_306551


namespace NUMINAMATH_CALUDE_circles_are_disjoint_l3065_306579

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 + 2)^2 = 1}

-- Define the centers and radii
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, -2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_are_disjoint : 
  Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2) > radius₁ + radius₂ := by
  sorry

end NUMINAMATH_CALUDE_circles_are_disjoint_l3065_306579


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3065_306533

theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1 : area = 80)
  (h2 : d1 = 16)
  (h3 : area = (d1 * d2) / 2) :
  d2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3065_306533


namespace NUMINAMATH_CALUDE_not_divisible_by_61_l3065_306585

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_61_l3065_306585


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l3065_306517

/-- Represents the number of acrobats in the parade. -/
def num_acrobats : ℕ := 4

/-- Represents the number of elephants in the parade. -/
def num_elephants : ℕ := 8

/-- Represents the number of horses in the parade. -/
def num_horses : ℕ := 8

/-- The total number of legs in the parade. -/
def total_legs : ℕ := 72

/-- The total number of heads in the parade. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  num_acrobats * 2 + num_elephants * 4 + num_horses * 4 = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads :=
by sorry

end NUMINAMATH_CALUDE_acrobats_count_correct_l3065_306517


namespace NUMINAMATH_CALUDE_probability_b_draws_red_l3065_306586

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

theorem probability_b_draws_red :
  let prob_b_red : ℚ := 
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) +
    (white_balls : ℚ) / total_balls * (red_balls : ℚ) / (total_balls - 1)
  prob_b_red = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_b_draws_red_l3065_306586


namespace NUMINAMATH_CALUDE_larger_circle_radius_l3065_306582

theorem larger_circle_radius (r : ℝ) (R : ℝ) : 
  r = 2 →  -- radius of smaller circles
  R = r + r * Real.sqrt 3 →  -- radius of larger circle
  R = 2 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l3065_306582


namespace NUMINAMATH_CALUDE_inequalities_proof_l3065_306528

theorem inequalities_proof :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3065_306528


namespace NUMINAMATH_CALUDE_parabola_intersection_angles_l3065_306518

/-- Parabola C: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point on the directrix -/
def directrix_point : ℝ × ℝ := (-1, 0)

/-- Line passing through P(m,0) -/
def line (m k : ℝ) (x y : ℝ) : Prop := x = k*y + m

/-- Intersection points of line and parabola -/
def intersection_points (m k : ℝ) : Prop :=
  ∃ (A B : PointOnParabola), A ≠ B ∧ line m k A.x A.y ∧ line m k B.x B.y

/-- Angle between two vectors -/
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_angles (m : ℝ) 
  (h_intersect : ∀ k, intersection_points m k) : 
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - directrix_point.1, A.y - directrix_point.2) 
          (B.x - directrix_point.1, B.y - directrix_point.2) < π/2) ∧
  (m = 3 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x - focus.1, A.y - focus.2) 
          (B.x - focus.1, B.y - focus.2) > π/2) ∧
  (m = 4 → ∀ A B : PointOnParabola, 
    line m (sorry) A.x A.y → line m (sorry) B.x B.y → 
    angle (A.x, A.y) (B.x, B.y) = π/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_angles_l3065_306518


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l3065_306566

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l3065_306566


namespace NUMINAMATH_CALUDE_tens_digit_of_3_pow_405_l3065_306520

theorem tens_digit_of_3_pow_405 : 3^405 % 100 = 43 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_pow_405_l3065_306520


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3065_306587

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 2) / (x - 4) ≥ 3 ↔ x > 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3065_306587


namespace NUMINAMATH_CALUDE_principal_amount_proof_l3065_306575

/-- 
Given a principal amount P put at simple interest for 3 years,
if increasing the interest rate by 3% results in 81 more interest,
then P must equal 900.
-/
theorem principal_amount_proof (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l3065_306575


namespace NUMINAMATH_CALUDE_expression_simplification_l3065_306508

theorem expression_simplification (q : ℝ) : 
  ((7 * q - 4) - 3 * q * 2) * 4 + (5 - 2 / 2) * (8 * q - 12) = 36 * q - 64 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3065_306508


namespace NUMINAMATH_CALUDE_no_grasshopper_overlap_l3065_306510

/-- Represents the position of a grasshopper -/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents the state of all four grasshoppers -/
structure GrasshopperState where
  g1 : Position
  g2 : Position
  g3 : Position
  g4 : Position

/-- Calculates the center of mass of three positions -/
def centerOfMass (p1 p2 p3 : Position) : Position :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

/-- Calculates the symmetric position of a point with respect to another point -/
def symmetricPosition (p center : Position) : Position :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- Performs a single jump for one grasshopper -/
def jump (state : GrasshopperState) (jumper : Fin 4) : GrasshopperState :=
  match jumper with
  | 0 => { state with g1 := symmetricPosition state.g1 (centerOfMass state.g2 state.g3 state.g4) }
  | 1 => { state with g2 := symmetricPosition state.g2 (centerOfMass state.g1 state.g3 state.g4) }
  | 2 => { state with g3 := symmetricPosition state.g3 (centerOfMass state.g1 state.g2 state.g4) }
  | 3 => { state with g4 := symmetricPosition state.g4 (centerOfMass state.g1 state.g2 state.g3) }

/-- Checks if any two grasshoppers are at the same position -/
def hasOverlap (state : GrasshopperState) : Prop :=
  state.g1 = state.g2 ∨ state.g1 = state.g3 ∨ state.g1 = state.g4 ∨
  state.g2 = state.g3 ∨ state.g2 = state.g4 ∨
  state.g3 = state.g4

/-- Initial state of the grasshoppers on a square -/
def initialState (n : ℕ) : GrasshopperState :=
  { g1 := { x := 0,     y := 0 },
    g2 := { x := 3^n,   y := 0 },
    g3 := { x := 3^n,   y := 3^n },
    g4 := { x := 0,     y := 3^n } }

/-- The main theorem stating that no overlap occurs after any number of jumps -/
theorem no_grasshopper_overlap (n : ℕ) :
  ∀ (jumps : List (Fin 4)), ¬(hasOverlap (jumps.foldl jump (initialState n))) :=
sorry

end NUMINAMATH_CALUDE_no_grasshopper_overlap_l3065_306510


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3065_306572

theorem fraction_equivalence (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 ↔ n = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3065_306572


namespace NUMINAMATH_CALUDE_abs_func_no_opposite_signs_l3065_306559

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem abs_func_no_opposite_signs :
  ∀ (a b : ℝ), (abs_func a) * (abs_func b) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_func_no_opposite_signs_l3065_306559


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l3065_306535

/-- The area inside a circle of radius 1 but outside a square of side length 2, when both share the same center, is equal to π - 2. -/
theorem area_circle_outside_square :
  let circle_radius : ℝ := 1
  let square_side : ℝ := 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let area_difference : ℝ := circle_area - square_area
  area_difference = π - 2 := by sorry

end NUMINAMATH_CALUDE_area_circle_outside_square_l3065_306535


namespace NUMINAMATH_CALUDE_f_2002_eq_zero_l3065_306554

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_2002_eq_zero
  (h1 : is_even f)
  (h2 : f 2 = 0)
  (h3 : is_odd g)
  (h4 : ∀ x, g x = f (x - 1)) :
  f 2002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2002_eq_zero_l3065_306554


namespace NUMINAMATH_CALUDE_coffee_buyers_fraction_l3065_306543

theorem coffee_buyers_fraction (total : ℕ) (non_coffee : ℕ) 
  (h1 : total = 25) (h2 : non_coffee = 10) : 
  (total - non_coffee : ℚ) / total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_coffee_buyers_fraction_l3065_306543


namespace NUMINAMATH_CALUDE_grid_and_unshaded_area_sum_l3065_306516

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents an unshaded square -/
structure UnshadedSquare :=
  (side_length : ℝ)

/-- Calculates the total area of a grid -/
def grid_area (g : Grid) : ℝ :=
  (g.size * g.square_size) ^ 2

/-- Calculates the area of an unshaded square -/
def unshaded_square_area (u : UnshadedSquare) : ℝ :=
  u.side_length ^ 2

/-- The main theorem to prove -/
theorem grid_and_unshaded_area_sum :
  let g : Grid := { size := 6, square_size := 3 }
  let u : UnshadedSquare := { side_length := 1.5 }
  let num_unshaded : ℕ := 5
  grid_area g + (num_unshaded * unshaded_square_area u) = 335.25 := by
  sorry


end NUMINAMATH_CALUDE_grid_and_unshaded_area_sum_l3065_306516


namespace NUMINAMATH_CALUDE_transaction_fraction_l3065_306584

theorem transaction_fraction (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 81 →
  jade_transactions = cal_transactions + 15 →
  cal_transactions * 3 = anthony_transactions * 2 := by
sorry

end NUMINAMATH_CALUDE_transaction_fraction_l3065_306584


namespace NUMINAMATH_CALUDE_expression_equality_l3065_306522

theorem expression_equality : 
  Real.sqrt 32 + (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 4 - 6 * Real.sqrt (1/2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3065_306522


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3065_306576

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^2 + 2*b*c)/(b^2 + c^2) + (b^2 + 2*a*c)/(c^2 + a^2) + (c^2 + 2*a*b)/(a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3065_306576


namespace NUMINAMATH_CALUDE_expression_value_l3065_306519

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4*a^2*b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3065_306519


namespace NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l3065_306556

theorem orthogonal_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∀ i, i < 2 → a i * b i = 0) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l3065_306556


namespace NUMINAMATH_CALUDE_sum_of_four_real_numbers_l3065_306546

theorem sum_of_four_real_numbers (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_eq : (a^2 + b^2 - 1)*(a + b) = (b^2 + c^2 - 1)*(b + c) ∧ 
          (b^2 + c^2 - 1)*(b + c) = (c^2 + d^2 - 1)*(c + d)) : 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_real_numbers_l3065_306546


namespace NUMINAMATH_CALUDE_adults_cookie_fraction_l3065_306509

theorem adults_cookie_fraction (total_cookies : ℕ) (num_children : ℕ) (cookies_per_child : ℕ) :
  total_cookies = 120 →
  num_children = 4 →
  cookies_per_child = 20 →
  (total_cookies - num_children * cookies_per_child : ℚ) / total_cookies = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adults_cookie_fraction_l3065_306509


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3065_306538

/-- Geometric sequence with first three terms summing to 14 and common ratio 2 -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 = 14)

/-- The general term of the geometric sequence is 2^n -/
theorem geometric_sequence_general_term (a : ℕ+ → ℝ) 
  (h : GeometricSequence a) : 
  ∀ n : ℕ+, a n = 2^(n : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3065_306538


namespace NUMINAMATH_CALUDE_students_without_vision_assistance_l3065_306565

/-- Given a group of 40 students where 25% wear glasses and 40% wear contact lenses,
    prove that 14 students do not wear any vision assistance wear. -/
theorem students_without_vision_assistance (total_students : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ) :
  total_students = 40 →
  glasses_percent = 25 / 100 →
  contacts_percent = 40 / 100 →
  total_students - (glasses_percent * total_students + contacts_percent * total_students) = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_without_vision_assistance_l3065_306565


namespace NUMINAMATH_CALUDE_detergent_calculation_l3065_306560

/-- Calculates the total amount of detergent used for washing clothes -/
theorem detergent_calculation (total_clothes cotton_clothes woolen_clothes : ℝ)
  (cotton_detergent wool_detergent : ℝ) : 
  total_clothes = cotton_clothes + woolen_clothes →
  cotton_clothes = 4 →
  woolen_clothes = 5 →
  cotton_detergent = 2 →
  wool_detergent = 1.5 →
  cotton_clothes * cotton_detergent + woolen_clothes * wool_detergent = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_calculation_l3065_306560


namespace NUMINAMATH_CALUDE_triangle_side_count_l3065_306573

/-- The number of integer values for the third side of a triangle with sides 15 and 40 -/
def triangleSideCount : ℕ := by
  sorry

theorem triangle_side_count :
  triangleSideCount = 29 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_count_l3065_306573


namespace NUMINAMATH_CALUDE_power_inequality_l3065_306574

theorem power_inequality (x y : ℝ) 
  (h1 : x^5 > y^4) 
  (h2 : y^5 > x^4) : 
  x^3 > y^2 := by
sorry

end NUMINAMATH_CALUDE_power_inequality_l3065_306574


namespace NUMINAMATH_CALUDE_exists_unique_solution_l3065_306583

theorem exists_unique_solution : ∃! x : ℝ, 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_solution_l3065_306583


namespace NUMINAMATH_CALUDE_minimum_students_l3065_306567

theorem minimum_students (b g : ℕ) : 
  b > 0 → g > 0 → 
  (b / 2 : ℚ) = 2 * (2 * g / 3 : ℚ) → 
  b + g ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l3065_306567


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3065_306597

/-- The probability of selecting 2 non-defective pens from a box of 9 pens with 3 defective pens -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h_total : total_pens = 9) 
  (h_defective : defective_pens = 3) : 
  (Nat.choose (total_pens - defective_pens) 2 : ℚ) / (Nat.choose total_pens 2) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l3065_306597


namespace NUMINAMATH_CALUDE_cost_of_type_B_theorem_l3065_306558

/-- The cost of purchasing type B books given the total number of books,
    the number of type A books purchased, and the unit price of type B books. -/
def cost_of_type_B (total_books : ℕ) (type_A_books : ℕ) (unit_price_B : ℕ) : ℕ :=
  unit_price_B * (total_books - type_A_books)

/-- Theorem stating that the cost of purchasing type B books
    is equal to 8(100-x) given the specified conditions. -/
theorem cost_of_type_B_theorem (x : ℕ) (h : x ≤ 100) :
  cost_of_type_B 100 x 8 = 8 * (100 - x) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_type_B_theorem_l3065_306558


namespace NUMINAMATH_CALUDE_log_half_increasing_interval_l3065_306501

noncomputable def y (x a : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem log_half_increasing_interval (a : ℝ) (h : a > 0) :
  (∃ m, m > 0 ∧ is_increasing (y · a) 0 m) ↔
  ((0 < a ∧ a ≤ Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a) ∨
   (a > Real.sqrt 3 ∧ ∃ m, 0 < m ∧ m ≤ a - Real.sqrt (a^2 - 3))) :=
sorry

end NUMINAMATH_CALUDE_log_half_increasing_interval_l3065_306501


namespace NUMINAMATH_CALUDE_quadratic_equation_a_value_l3065_306561

theorem quadratic_equation_a_value (a : ℝ) : 
  (∀ x, x^2 - 8*x + a = 0 ↔ (x - 4)^2 = 1) → a = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_value_l3065_306561


namespace NUMINAMATH_CALUDE_a_33_mod_33_l3065_306503

/-- The integer obtained by writing all integers from 1 to n sequentially -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that a₃₃ mod 33 = 22 -/
theorem a_33_mod_33 : a 33 % 33 = 22 := by
  sorry

end NUMINAMATH_CALUDE_a_33_mod_33_l3065_306503


namespace NUMINAMATH_CALUDE_smallest_shift_l3065_306530

-- Define the function f with the given property
def f_periodic (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 12) = f x

-- Define the property for the shifted function
def shifted_f_equal (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 3) = f (x / 3)

-- Theorem statement
theorem smallest_shift (f : ℝ → ℝ) (h : f_periodic f) :
  (∃ a : ℝ, a > 0 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b) →
  ∃ a : ℝ, a = 36 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l3065_306530


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3065_306532

-- Define the function f(x) = x³ - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

theorem tangent_line_y_intercept (a : ℝ) :
  (f' a 1 = 1) →  -- Tangent line at x=1 is parallel to x - y - 1 = 0
  (∃ b c : ℝ, ∀ x : ℝ, b * x + c = f a 1 + f' a 1 * (x - 1)) →  -- Equation of tangent line
  (∃ y : ℝ, y = f a 1 + f' a 1 * (0 - 1) ∧ y = -2)  -- y-intercept is -2
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3065_306532


namespace NUMINAMATH_CALUDE_smallest_possible_c_l3065_306500

theorem smallest_possible_c (a b c : ℝ) : 
  1 < a → a < b → b < c →
  1 + a ≤ b →
  1 / a + 1 / b ≤ 1 / c →
  c ≥ (3 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_c_l3065_306500


namespace NUMINAMATH_CALUDE_surface_area_after_vertex_removal_l3065_306506

/-- The surface area of a cube after removing unit cubes from its vertices -/
theorem surface_area_after_vertex_removal (side_length : ℝ) (h : side_length = 4) :
  6 * side_length^2 = 6 * side_length^2 := by sorry

end NUMINAMATH_CALUDE_surface_area_after_vertex_removal_l3065_306506


namespace NUMINAMATH_CALUDE_cubic_gp_roots_iff_a_60_l3065_306553

/-- A cubic polynomial with parameter a -/
def cubic (a : ℝ) (x : ℝ) : ℝ := x^3 - 15*x^2 + a*x - 64

/-- Predicate for three distinct real roots in geometric progression -/
def has_three_distinct_gp_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    cubic a x₁ = 0 ∧ cubic a x₂ = 0 ∧ cubic a x₃ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ x₂ = x₁ * q ∧ x₃ = x₂ * q

/-- The main theorem -/
theorem cubic_gp_roots_iff_a_60 :
  ∀ a : ℝ, has_three_distinct_gp_roots a ↔ a = 60 := by sorry

end NUMINAMATH_CALUDE_cubic_gp_roots_iff_a_60_l3065_306553


namespace NUMINAMATH_CALUDE_same_terminal_side_as_60_degrees_l3065_306599

def has_same_terminal_side (α : ℤ) : Prop :=
  ∃ k : ℤ, α = k * 360 + 60

theorem same_terminal_side_as_60_degrees :
  has_same_terminal_side (-300) ∧
  ¬has_same_terminal_side (-60) ∧
  ¬has_same_terminal_side 600 ∧
  ¬has_same_terminal_side 1380 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_60_degrees_l3065_306599


namespace NUMINAMATH_CALUDE_max_episodes_l3065_306541

/-- Represents a character in the TV show -/
structure Character where
  id : Nat

/-- Represents the state of knowledge for each character -/
structure KnowledgeState where
  knows_mystery : Set Character
  knows_others_know : Set (Character × Character)
  knows_others_dont_know : Set (Character × Character)

/-- Represents an episode of the TV show -/
inductive Episode
  | LearnMystery (c : Character)
  | LearnSomeoneKnows (c1 c2 : Character)
  | LearnSomeoneDoesntKnow (c1 c2 : Character)

/-- The number of characters in the TV show -/
def num_characters : Nat := 20

/-- Theorem: The maximum number of unique episodes is 780 -/
theorem max_episodes :
  ∃ (episodes : List Episode),
    episodes.length = 780 ∧
    episodes.Nodup ∧
    (∀ e : Episode, e ∈ episodes) ∧
    (∀ c : List Character, c.length = num_characters →
      ∃ (initial_state : KnowledgeState),
        ∃ (final_state : KnowledgeState),
          episodes.foldl
            (fun state episode =>
              match episode with
              | Episode.LearnMystery c =>
                { state with knows_mystery := state.knows_mystery ∪ {c} }
              | Episode.LearnSomeoneKnows c1 c2 =>
                { state with knows_others_know := state.knows_others_know ∪ {(c1, c2)} }
              | Episode.LearnSomeoneDoesntKnow c1 c2 =>
                { state with knows_others_dont_know := state.knows_others_dont_know ∪ {(c1, c2)} })
            initial_state
          = final_state) :=
  sorry

end NUMINAMATH_CALUDE_max_episodes_l3065_306541


namespace NUMINAMATH_CALUDE_brick_height_calculation_l3065_306531

/-- The height of a brick given wall dimensions, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_width : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_width = 0.1 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_height : ℝ),
    brick_height = 0.008 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l3065_306531


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3065_306514

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3065_306514


namespace NUMINAMATH_CALUDE_aquarium_fish_problem_l3065_306512

theorem aquarium_fish_problem (initial_fish : ℕ) : 
  initial_fish > 0 → initial_fish + 3 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_problem_l3065_306512


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_perimeter_l3065_306527

/-- Represents a triangle with two known side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ

/-- The perimeter of a triangle is less than the sum of its two known sides plus 24 -/
def perimeter_upper_bound (t : Triangle) : ℝ := t.a + t.b + 24

/-- The smallest integer greater than the perimeter of a triangle with sides 5 and 19 is 48 -/
theorem smallest_integer_greater_than_perimeter :
  ∀ (t : Triangle), t.a = 5 ∧ t.b = 19 → 
  (∀ (x : ℤ), x > perimeter_upper_bound t → x ≥ 48) ∧
  (48 > perimeter_upper_bound t) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_perimeter_l3065_306527


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l3065_306594

/-- The total weight of Marco's and his dad's strawberries is 40 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 8
  let dad_weight : ℕ := 32
  marco_weight + dad_weight = 40 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l3065_306594


namespace NUMINAMATH_CALUDE_max_edges_bipartite_graph_l3065_306507

/-- 
Given a complete bipartite graph K_{m,n} where m and n are positive integers and m + n = 21,
prove that the maximum number of edges is 110.
-/
theorem max_edges_bipartite_graph : 
  ∀ m n : ℕ+, 
  m + n = 21 → 
  ∃ (max_edges : ℕ), 
    max_edges = m * n ∧ 
    ∀ k l : ℕ+, k + l = 21 → k * l ≤ max_edges :=
by
  sorry

end NUMINAMATH_CALUDE_max_edges_bipartite_graph_l3065_306507


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l3065_306596

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a predicate for an equilateral triangle
def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = 1 ∧ distance p2 p3 = 1 ∧ distance p3 p1 = 1

-- Main theorem
theorem equilateral_triangle_exists 
  (points : Finset Point) 
  (h1 : points.card = 6) 
  (h2 : ∃ (pairs : Finset (Point × Point)), 
    pairs.card = 8 ∧ 
    ∀ (pair : Point × Point), pair ∈ pairs → distance pair.1 pair.2 = 1) :
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    is_equilateral_triangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l3065_306596


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3065_306511

/-- The distance between the vertices of a hyperbola with equation x^2/36 - y^2/25 = 1 is 12 -/
theorem hyperbola_vertex_distance : 
  let a : ℝ := Real.sqrt 36
  let b : ℝ := Real.sqrt 25
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 36 - y^2 / 25 = 1
  2 * a = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3065_306511


namespace NUMINAMATH_CALUDE_truck_loading_capacity_correct_bag_count_l3065_306537

theorem truck_loading_capacity (truck_capacity : ℕ) 
                                (box_count box_weight : ℕ) 
                                (crate_count crate_weight : ℕ) 
                                (sack_count sack_weight : ℕ) 
                                (bag_weight : ℕ) : ℕ :=
  let total_loaded := box_count * box_weight + crate_count * crate_weight + sack_count * sack_weight
  let remaining_capacity := truck_capacity - total_loaded
  remaining_capacity / bag_weight

theorem correct_bag_count : 
  truck_loading_capacity 13500 100 100 10 60 50 50 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_truck_loading_capacity_correct_bag_count_l3065_306537


namespace NUMINAMATH_CALUDE_conic_section_type_l3065_306539

theorem conic_section_type (x y : ℝ) : 
  (9 * x^2 - 16 * y^2 = 0) → 
  ∃ (m₁ m₂ : ℝ), (∀ x y, (y = m₁ * x ∨ y = m₂ * x) ↔ 9 * x^2 - 16 * y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_type_l3065_306539


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l3065_306580

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l3065_306580


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3065_306578

theorem fraction_evaluation (x : ℝ) (h : x = 6) : 3 / (2 - 3 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3065_306578


namespace NUMINAMATH_CALUDE_rectangle_area_l3065_306563

/-- Given a rectangle with length four times its width and perimeter 200 cm, its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 →
  l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3065_306563


namespace NUMINAMATH_CALUDE_problem_solution_l3065_306534

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem problem_solution :
  (∀ m : ℝ, m > 0 → (∀ x : ℝ, p x → q x m) → m ≥ 4) ∧
  (∀ x : ℝ, (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → (x ∈ Set.Icc (-4 : ℝ) (-1) ∪ Set.Ioc 5 6)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3065_306534


namespace NUMINAMATH_CALUDE_modified_prism_surface_area_difference_l3065_306569

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the surface area added by removing a cube from the center of a face -/
def added_surface_area (cube_side : ℝ) : ℝ := 5 * cube_side^2

theorem modified_prism_surface_area_difference :
  let original_sa := surface_area 2 4 5
  let modified_sa := original_sa + added_surface_area 1
  modified_sa - original_sa = 5 := by sorry

end NUMINAMATH_CALUDE_modified_prism_surface_area_difference_l3065_306569


namespace NUMINAMATH_CALUDE_rectangle_problem_l3065_306526

theorem rectangle_problem (a b k l : ℕ) (h1 : k * l = 47 * (a + b)) 
  (h2 : a * k = b * l) : k = 2256 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3065_306526


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3065_306525

theorem consecutive_odd_integers_sum (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next consecutive odd integer
  c = b + 2 →               -- c is the next consecutive odd integer after b
  a + c = 150 →             -- sum of first and third is 150
  a + b + c = 225 :=        -- sum of all three is 225
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3065_306525


namespace NUMINAMATH_CALUDE_min_students_same_score_l3065_306515

theorem min_students_same_score (total_students : ℕ) (min_score max_score : ℕ) :
  total_students = 8000 →
  min_score = 30 →
  max_score = 83 →
  ∃ (score : ℕ), min_score ≤ score ∧ score ≤ max_score ∧
    (∃ (students_with_score : ℕ), students_with_score ≥ 149 ∧
      (∀ (s : ℕ), min_score ≤ s ∧ s ≤ max_score →
        (∃ (students : ℕ), students ≤ students_with_score))) :=
by sorry

end NUMINAMATH_CALUDE_min_students_same_score_l3065_306515


namespace NUMINAMATH_CALUDE_circle_center_correct_l3065_306562

/-- Definition of the circle C -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that circle_center is the center of the circle defined by circle_equation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3065_306562


namespace NUMINAMATH_CALUDE_product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l3065_306557

def first_n_primes (n : ℕ) : List ℕ :=
  sorry

def product_of_list (l : List ℕ) : ℕ :=
  sorry

def is_factor (a b : ℕ) : Prop :=
  ∃ k : ℕ, b = a * k

theorem product_of_first_1001_primes_factors (n : ℕ) :
  let P := product_of_list (first_n_primes 1001)
  (n = 2002 ∨ n = 3003 ∨ n = 5005 ∨ n = 6006) →
  is_factor n P :=
sorry

theorem product_of_first_1001_primes_not_factor :
  let P := product_of_list (first_n_primes 1001)
  ¬ is_factor 7007 P :=
sorry

end NUMINAMATH_CALUDE_product_of_first_1001_primes_factors_product_of_first_1001_primes_not_factor_l3065_306557


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3065_306564

/-- The quadratic inequality -2 + 3x - 2x^2 > 0 has an empty solution set -/
theorem quadratic_inequality_empty_solution : 
  ∀ x : ℝ, ¬(-2 + 3*x - 2*x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l3065_306564


namespace NUMINAMATH_CALUDE_sanctuary_bird_pairs_l3065_306540

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := 29

/-- The number of pairs of birds per species -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := num_species * pairs_per_species

theorem sanctuary_bird_pairs : total_pairs = 203 := by
  sorry

end NUMINAMATH_CALUDE_sanctuary_bird_pairs_l3065_306540


namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l3065_306568

/-- A sequence of integers satisfying the recurrence relation a_{n+2} = a_{n+1} - m * a_n -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  (a 1 ≠ 0 ∨ a 2 ≠ 0) ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property (m : ℤ) (a : ℕ → ℤ) 
    (hm : |m| ≥ 2) 
    (ha : RecurrenceSequence m a) 
    (r s : ℕ) 
    (hrs : r > s ∧ s ≥ 2) 
    (heq : a r = a s ∧ a s = a 1) : 
  r - s ≥ |m| := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l3065_306568


namespace NUMINAMATH_CALUDE_marking_implies_prime_f_1997_l3065_306570

/-- Represents the marking procedure on a 2N-gon -/
def mark_procedure (N : ℕ) : Set ℕ := sorry

/-- The function f(N) that counts non-marked vertices -/
def f (N : ℕ) : ℕ := sorry

/-- Main theorem: If f(N) = 0, then 2N + 1 is prime -/
theorem marking_implies_prime (N : ℕ) (h1 : N > 2) (h2 : f N = 0) : Nat.Prime (2 * N + 1) := by
  sorry

/-- Computation of f(1997) -/
theorem f_1997 : f 1997 = 3810 := by
  sorry

end NUMINAMATH_CALUDE_marking_implies_prime_f_1997_l3065_306570


namespace NUMINAMATH_CALUDE_rectangle_width_l3065_306542

/-- Given a square and a rectangle, if the area of the square is five times the area of the rectangle,
    the perimeter of the square is 800 cm, and the length of the rectangle is 125 cm,
    then the width of the rectangle is 64 cm. -/
theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) :
  square_perimeter = 800 ∧
  rectangle_length = 125 ∧
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * (64 : ℝ)) →
  64 = (square_perimeter / 4) ^ 2 / (5 * rectangle_length) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l3065_306542


namespace NUMINAMATH_CALUDE_imaginary_cube_plus_one_l3065_306590

theorem imaginary_cube_plus_one (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_cube_plus_one_l3065_306590


namespace NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l3065_306591

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 3)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_equals_sqrt_13_l3065_306591


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3065_306545

theorem logarithm_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3065_306545


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l3065_306581

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x y : ℝ, y = 3 * x^2 - 6 * x + 3 ↔ y = -2 * x^2 + x + 5 → x = a ∨ x = c) ∧
  c ≥ a ∧
  c - a = Real.sqrt 89 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l3065_306581


namespace NUMINAMATH_CALUDE_traveler_distance_l3065_306529

/-- Calculates the distance traveled given initial conditions and new travel parameters. -/
def distance_traveled (initial_distance : ℚ) (initial_days : ℕ) (initial_hours_per_day : ℕ)
                      (new_days : ℕ) (new_hours_per_day : ℕ) : ℚ :=
  let initial_total_hours : ℚ := initial_days * initial_hours_per_day
  let speed : ℚ := initial_distance / initial_total_hours
  let new_total_hours : ℚ := new_days * new_hours_per_day
  speed * new_total_hours

/-- The theorem states that given the initial conditions and new travel parameters,
    the traveler will cover 93 23/29 kilometers. -/
theorem traveler_distance : 
  distance_traveled 112 29 7 17 10 = 93 + 23 / 29 := by
  sorry

end NUMINAMATH_CALUDE_traveler_distance_l3065_306529


namespace NUMINAMATH_CALUDE_limit_cubic_fraction_l3065_306523

theorem limit_cubic_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |((x^3 - 1) / (x - 1)) - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_cubic_fraction_l3065_306523


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l3065_306552

/-- The number of y-intercepts of the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l3065_306552


namespace NUMINAMATH_CALUDE_function_properties_l3065_306588

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 4*a*x + b

theorem function_properties :
  ∃ (a b : ℝ),
    -- Condition: f(1) = 3
    f a b 1 = 3 ∧
    -- Condition: f'(1) = 1 (slope of tangent line at x=1)
    f_derivative a b 1 = 1 ∧
    -- Prove: a = 2 and b = 6
    a = 2 ∧ b = 6 ∧
    -- Prove: Range of f(x) on [-1, 4] is [-11, 24]
    (∀ x, -1 ≤ x ∧ x ≤ 4 → -11 ≤ f a b x ∧ f a b x ≤ 24) ∧
    f a b (-1) = -11 ∧ f a b 4 = 24 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3065_306588


namespace NUMINAMATH_CALUDE_initial_tagged_fish_count_l3065_306547

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged_fish : ℕ := 50

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

theorem initial_tagged_fish_count :
  initial_tagged_fish = 50 := by sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_count_l3065_306547


namespace NUMINAMATH_CALUDE_business_partnership_timing_l3065_306513

/-- Represents the number of months after A started that B joined the business. -/
def months_until_b_joined : ℕ → Prop :=
  fun x =>
    let a_investment := 3500 * 12
    let b_investment := 21000 * (12 - x)
    a_investment * 3 = b_investment * 2

theorem business_partnership_timing :
  ∃ x : ℕ, months_until_b_joined x ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_timing_l3065_306513


namespace NUMINAMATH_CALUDE_equation_solutions_l3065_306521

theorem equation_solutions :
  (∃ x : ℝ, x - 0.4 * x = 120 ∧ x = 200) ∧
  (∃ x : ℝ, 5 * x - 5 / 6 = 5 / 4 ∧ x = 5 / 12) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3065_306521


namespace NUMINAMATH_CALUDE_inequality_proof_l3065_306577

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3065_306577


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3065_306550

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x^2 - 1)) ↔ x ≠ 1 ∧ x ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3065_306550


namespace NUMINAMATH_CALUDE_sunglasses_hat_probability_l3065_306549

theorem sunglasses_hat_probability 
  (total_sunglasses : ℕ) 
  (total_hats : ℕ) 
  (hat_also_sunglasses_prob : ℚ) :
  total_sunglasses = 80 →
  total_hats = 45 →
  hat_also_sunglasses_prob = 1/3 →
  (total_hats * hat_also_sunglasses_prob : ℚ) / total_sunglasses = 3/16 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_hat_probability_l3065_306549


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3065_306536

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3065_306536
