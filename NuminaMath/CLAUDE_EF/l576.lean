import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_change_specific_change_l576_57606

/-- Represents the slope of a line -/
structure Slope where
  rise : ℚ
  run : ℚ
  run_nonzero : run ≠ 0

/-- Calculates the change in y given a change in x and a slope -/
def change_in_y (slope : Slope) (dx : ℚ) : ℚ :=
  (slope.rise / slope.run) * dx

theorem line_change (dx : ℚ) : 
  let slope : Slope := { rise := 5, run := 2, run_nonzero := by norm_num }
  change_in_y slope dx = (5 / 2) * dx := by sorry

theorem specific_change : 
  let slope : Slope := { rise := 5, run := 2, run_nonzero := by norm_num }
  change_in_y slope 8 = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_change_specific_change_l576_57606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l576_57658

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x : ℝ) : ℝ := x^2 + 2*p*x + 2*p - 2

-- Define the discriminant of the parabola
def discriminant (p : ℝ) : ℝ := 4*p^2 - 8*p + 8

-- Define the x-coordinates of intersection points
def x_intersections (p : ℝ) : ℝ × ℝ := (-p - Real.sqrt (discriminant p / 4), -p + Real.sqrt (discriminant p / 4))

-- Define the vertex of the parabola
def vertex (p : ℝ) : ℝ × ℝ := (-p, -(p-1)^2 - 1)

-- Define the area of triangle ABM
def triangle_area (p : ℝ) : ℝ := 
  let (x1, x2) := x_intersections p
  let (_, y_vertex) := vertex p
  1/2 * abs (x2 - x1) * abs y_vertex

theorem parabola_properties (p : ℝ) : 
  (∀ x : ℝ, parabola p x = 0 → x = (x_intersections p).1 ∨ x = (x_intersections p).2) ∧
  (∃ p_min : ℝ, p_min = 1 ∧ ∀ q : ℝ, triangle_area q ≥ triangle_area p_min) ∧
  triangle_area 1 = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l576_57658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l576_57697

/-- The time (in seconds) required for a train to pass a stationary point -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  train_length / (train_speed_kmph * 1000 / 3600)

/-- Theorem: A train of length 75 meters moving at 50 km/h takes approximately 5.4 seconds to pass a stationary point -/
theorem train_passing_telegraph_post :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_passing_time 75 50 - 5.4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_telegraph_post_l576_57697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l576_57619

/-- The number of solutions in positive integers for the Diophantine equation 2x + 3y = 780 -/
def num_solutions : ℕ := 130

/-- A solution to the Diophantine equation 2x + 3y = 780 -/
structure Solution where
  x : ℕ+
  y : ℕ+
  eq : 2 * x.val + 3 * y.val = 780

/-- The set of all solutions to the Diophantine equation 2x + 3y = 780 -/
def solution_set : Set Solution := {s : Solution | True}

/-- Assume the solution set is finite -/
instance : Fintype solution_set := sorry

/-- The theorem stating that the number of solutions is correct -/
theorem count_solutions :
  Fintype.card solution_set = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l576_57619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_horses_count_l576_57665

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_cost : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the number of horses a put in the pasture -/
def calculate_a_horses (pr : PastureRental) : ℕ :=
  (pr.total_cost - pr.b_payment - pr.c_horses * pr.c_months) / pr.a_months

/-- Theorem stating that a put in 18 horses given the problem conditions -/
theorem a_horses_count (pr : PastureRental) 
  (h1 : pr.total_cost = 435)
  (h2 : pr.a_months = 8)
  (h3 : pr.b_horses = 16)
  (h4 : pr.b_months = 9)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) :
  calculate_a_horses pr = 18 := by
  sorry

#eval calculate_a_horses { 
  total_cost := 435, 
  a_months := 8, 
  b_horses := 16, 
  b_months := 9, 
  c_horses := 18, 
  c_months := 6, 
  b_payment := 180 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_horses_count_l576_57665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l576_57672

/-- Calculates the time for a train to cross a platform -/
noncomputable def time_to_cross (train_length : ℝ) (platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

/-- Proves that a train of length 270m, crossing a 250m platform in 20s, takes 15s to cross a 120m platform -/
theorem train_crossing_time_theorem :
  let train_length : ℝ := 270
  let first_platform_length : ℝ := 120
  let second_platform_length : ℝ := 250
  let time_second_platform : ℝ := 20
  let speed : ℝ := (train_length + second_platform_length) / time_second_platform
  time_to_cross train_length first_platform_length speed = 15 := by
  -- Unfold definitions
  unfold time_to_cross
  -- Simplify the expression
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry

#check train_crossing_time_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l576_57672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l576_57646

theorem polynomial_existence (a n : ℕ) (ha : a > 1) (hn : n > 0) :
  ∃ (p : Polynomial ℤ),
    (Polynomial.degree p = n) ∧
    (∀ i : Fin (n + 1), ∃ k : ℕ+,
      p.eval (↑i.val : ℤ) = 2 * a ^ k.val + 3) ∧
    (∀ i j : Fin (n + 1), i ≠ j → p.eval (↑i.val : ℤ) ≠ p.eval (↑j.val : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l576_57646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_9_with_three_even_one_odd_l576_57656

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def count_even_digits (n : ℕ) : ℕ :=
  (n.digits 10).filter (λ d => d % 2 = 0) |>.length

def count_odd_digits (n : ℕ) : ℕ :=
  (n.digits 10).filter (λ d => d % 2 ≠ 0) |>.length

theorem smallest_four_digit_divisible_by_9_with_three_even_one_odd :
  ∀ n : ℕ,
    is_four_digit n →
    divisible_by_9 n →
    count_even_digits n = 3 →
    count_odd_digits n = 1 →
    2008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_9_with_three_even_one_odd_l576_57656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_perfect_square_l576_57611

theorem polynomial_perfect_square : 
  ∃ (p q : ℚ), (fun x : ℚ ↦ x^4 + x^3 + 2*x^2 + (7/8)*x + 49/64) = (fun x : ℚ ↦ (x^2 + p*x + q)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_perfect_square_l576_57611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_min_score_l576_57604

theorem first_player_min_score (n : ℕ) (h : n = 101) :
  ∃ (a b : ℕ), a ≤ n ∧ b ≤ n ∧ a ≠ b ∧
  (∀ (S : Finset ℕ), S.card = 99 → S ⊆ Finset.range (n + 1) →
  (a ∉ S ∧ b ∉ S → |Int.ofNat a - Int.ofNat b| ≥ 55)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_min_score_l576_57604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l576_57657

theorem max_sum_of_products (a b c d e : ℕ) : 
  a ∈ ({1, 2, 4, 5, 6} : Set ℕ) → 
  b ∈ ({1, 2, 4, 5, 6} : Set ℕ) → 
  c ∈ ({1, 2, 4, 5, 6} : Set ℕ) → 
  d ∈ ({1, 2, 4, 5, 6} : Set ℕ) → 
  e ∈ ({1, 2, 4, 5, 6} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → 
  b ≠ c → b ≠ d → b ≠ e → 
  c ≠ d → c ≠ e → 
  d ≠ e → 
  (a * b + b * c + c * d + d * e + e * a) ≤ 54 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_products_l576_57657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_R_equals_half_l576_57682

theorem prove_R_equals_half (m n : ℝ) (R : ℝ) 
  (h1 : (2 : ℝ)^m = 36) 
  (h2 : (3 : ℝ)^n = 36) 
  (h3 : R = 1/m + 1/n) : 
  R = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_R_equals_half_l576_57682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rate_is_500_l576_57629

/-- Represents the flow rate problem with a tank and drains. -/
structure TankProblem where
  tankCapacity : ℝ
  initialLevel : ℝ
  drain1Rate : ℝ
  drain2Rate : ℝ
  fillTime : ℝ

/-- Calculates the flow rate of the pipe filling the tank. -/
noncomputable def calculateFlowRate (problem : TankProblem) : ℝ :=
  let emptySpace := problem.tankCapacity - problem.initialLevel
  let drain1Loss := (problem.fillTime / 4) * 1000
  let drain2Loss := (problem.fillTime / 6) * 1000
  let totalWaterNeeded := emptySpace + drain1Loss + drain2Loss
  totalWaterNeeded / problem.fillTime

/-- Theorem stating that for the given problem, the flow rate is 500 liters per minute. -/
theorem flow_rate_is_500 (problem : TankProblem) 
    (h1 : problem.tankCapacity = 10000)
    (h2 : problem.initialLevel = 5000)
    (h3 : problem.drain1Rate = 1000 / 4)
    (h4 : problem.drain2Rate = 1000 / 6)
    (h5 : problem.fillTime = 60) :
    calculateFlowRate problem = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rate_is_500_l576_57629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l576_57613

/- Define lg as log base 10 -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/- Part 1 -/
theorem part_one : 
  Real.sqrt (25/9) - (8/27)^(1/3:ℝ) - (Real.pi + Real.exp 1)^(0:ℝ) + (1/4:ℝ)^(-1/2:ℝ) = 2 := by sorry

/- Part 2 -/
theorem part_two : 
  (lg 2)^2 + lg 2 * lg 5 + Real.sqrt ((lg 2)^2 - lg 4 + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l576_57613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l576_57695

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The x-intercept of a line, if it exists -/
noncomputable def Line.xIntercept (l : Line) : Option ℝ :=
  if l.b = 0 then none else some (-l.c / l.b)

/-- The y-intercept of a line, if it exists -/
noncomputable def Line.yIntercept (l : Line) : Option ℝ :=
  if l.a = 0 then none else some (-l.c / l.a)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.containsPoint (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_proof (l : Line) : 
  (l.a = 1 ∧ l.b = 1 ∧ l.c = 5) →
  l.containsPoint ⟨-3, -2⟩ ∧
  (∃ x, l.xIntercept = some x ∧ l.yIntercept = some x ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l576_57695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l576_57650

theorem sin_sum_of_angles (α β : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.cos (π - α) = 1/3) (h6 : Real.sin (π/2 + β) = 2/3) :
  Real.sin (α + β) = (4 * Real.sqrt 2 - Real.sqrt 5) / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_angles_l576_57650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_exponents_l576_57645

theorem relationship_between_exponents (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (2 : ℝ)^x = (3 : ℝ)^y) (h2 : (3 : ℝ)^y = (5 : ℝ)^z) : 3*y < 2*x ∧ 2*x < 5*z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_exponents_l576_57645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probability_l576_57607

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- The distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2 : ℝ)

/-- Predicate to check if a point is a valid move (distance √3 from current position) -/
def isValidMove (current next : Point3D) : Prop :=
  distance current next = Real.sqrt 3

/-- The set of all possible moves from a given point -/
def possibleMoves (p : Point3D) : Set Point3D :=
  { next | isValidMove p next }

/-- Predicate to check if a point is at distance 2√2 from the origin -/
def isTargetDistance (p : Point3D) : Prop :=
  distance origin p = 2 * Real.sqrt 2

/-- The probability of reaching a point at distance 2√2 from the origin after two moves -/
def probabilityAfterTwoMoves : ℚ := 3/8

theorem particle_movement_probability :
  ∀ (p1 p2 : Point3D),
    p1 ∈ possibleMoves origin →
    p2 ∈ possibleMoves p1 →
    isTargetDistance p2 →
    (probabilityAfterTwoMoves : ℝ) = 3/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probability_l576_57607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_l576_57653

/-- A rectangular box with side lengths a, b, and c -/
structure RectangularBox where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Properties of the rectangular box -/
def BoxProperties (box : RectangularBox) : Prop :=
  4 * (box.a + box.b + box.c) = 60 ∧
  2 * (box.a * box.b + box.b * box.c + box.c * box.a) = 150 ∧
  box.a * box.b * box.c = 216

/-- The sum of the lengths of all interior diagonals -/
noncomputable def InteriorDiagonalsSum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.a^2 + box.b^2 + box.c^2)

/-- Theorem: The sum of the lengths of all interior diagonals is 20√3 -/
theorem interior_diagonals_sum (box : RectangularBox) 
  (h : BoxProperties box) : InteriorDiagonalsSum box = 20 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_l576_57653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_implies_three_non_collinear_l576_57666

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of non-coplanar points -/
def NonCoplanar (A B C D : Point3D) : Prop := sorry

/-- Definition of non-collinear points -/
def NonCollinear (P Q R : Point3D) : Prop := sorry

/-- Theorem: Given four non-coplanar points, there exist three non-collinear points among them -/
theorem non_coplanar_implies_three_non_collinear 
  (A B C D : Point3D)
  (h : NonCoplanar A B C D) : 
  NonCollinear A B C ∨ NonCollinear A B D ∨ NonCollinear A C D ∨ NonCollinear B C D := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_coplanar_implies_three_non_collinear_l576_57666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l576_57620

theorem largest_power_of_18_dividing_30_factorial : 
  ∃ m : ℕ, m = 7 ∧ 
  (∀ k : ℕ, (18 ^ k : ℕ) ∣ Nat.factorial 30 → k ≤ m) ∧
  ((18 ^ m : ℕ) ∣ Nat.factorial 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l576_57620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l576_57614

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 6 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State that A and B are on both circles
axiom A_on_circles : circle1 A.1 A.2 ∧ circle2 A.1 A.2
axiom B_on_circles : circle1 B.1 B.2 ∧ circle2 B.1 B.2

-- Define the length of a chord
noncomputable def chord_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem common_chord_length :
  chord_length A B = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l576_57614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l576_57679

noncomputable def area_of_equilateral_triangle_with_perimeter (perimeter : ℝ) : ℝ :=
  let side_length := perimeter / 3
  (Real.sqrt 3 / 4) * side_length^2

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) : 
  ∃ (A : ℝ), A = (Real.sqrt 3 * p^2) / 9 ∧ 
  A = area_of_equilateral_triangle_with_perimeter (2 * p) := by
  let A := (Real.sqrt 3 * p^2) / 9
  use A
  constructor
  · rfl
  · unfold area_of_equilateral_triangle_with_perimeter
    simp [h]
    ring
    
#check equilateral_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l576_57679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_regular_gift_bags_needed_l576_57654

/-- Calculates the number of additional regular gift bags needed for an open house event. -/
theorem additional_regular_gift_bags_needed 
  (confirmed_guests : ℕ)
  (high_prob_guests : ℕ) (high_prob : ℚ)
  (low_prob_guests : ℕ) (low_prob : ℚ)
  (extravagant_bags : ℕ)
  (special_bags : ℕ)
  (regular_bags : ℕ) :
  confirmed_guests = 50 →
  high_prob_guests = 30 →
  high_prob = 7/10 →
  low_prob_guests = 15 →
  low_prob = 2/5 →
  extravagant_bags = 10 →
  special_bags = 25 →
  regular_bags = 20 →
  (Nat.ceil ((confirmed_guests : ℚ) + 
    (high_prob_guests : ℚ) * high_prob + 
    (low_prob_guests : ℚ) * low_prob) : ℕ) - 
  (extravagant_bags + special_bags) - 
  regular_bags = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_regular_gift_bags_needed_l576_57654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l576_57652

def permutation_sum (p : Equiv.Perm (Fin 12)) : ℚ :=
  |p 0 - p 1| + |p 2 - p 3| + |p 4 - p 5| + |p 6 - p 7| + |p 8 - p 9| + |p 10 - p 11|

theorem average_permutation_sum : 
  (Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin 12))) permutation_sum) / (Nat.factorial 12) = 286 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_permutation_sum_l576_57652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l576_57644

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

/-- Theorem: A triangle with base 123 meters and height 10 meters has an area of 615 square meters -/
theorem triangle_area_example : triangle_area 123 10 = 615 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- The result follows from basic arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l576_57644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l576_57667

/-- Given a triangle ABC, k is a real number satisfying the inequality for all such triangles -/
def satisfies_inequality (k : ℝ) : Prop :=
  ∀ (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi),
    k * (Real.sin B)^2 + Real.sin A * Real.sin C > 19 * Real.sin B * Real.sin C

/-- The minimum value of k that satisfies the inequality is 100 -/
theorem min_k_value : 
  (∀ k : ℝ, satisfies_inequality k → k ≥ 100) ∧ 
  satisfies_inequality 100 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l576_57667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_formula_l576_57641

theorem cos_sum_formula (α : ℝ) 
  (h1 : Real.cos α = 12 / 13) 
  (h2 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) : 
  Real.cos (α + Real.pi / 4) = 17 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_formula_l576_57641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_positions_allow_cube_folding_l576_57648

/-- Represents a position on the edge of the cross-shaped layout --/
inductive Position
| OuterEdge
| InnerEdge

/-- Represents the cross-shaped layout with 4 congruent squares --/
structure CrossLayout :=
(squares : Fin 4 → Unit)

/-- Represents the arrangement after attaching the fifth square --/
structure Arrangement :=
(base : CrossLayout)
(fifthSquarePosition : Position)

/-- Predicate to check if an arrangement can be folded into a cube with one face missing --/
def canFoldIntoCube (a : Arrangement) : Bool :=
  sorry

/-- The set of all possible arrangements --/
def allArrangements : Finset Arrangement :=
  sorry

/-- The theorem stating that exactly 8 positions allow folding into a cube --/
theorem eight_positions_allow_cube_folding :
  (allArrangements.filter (fun a => canFoldIntoCube a)).card = 8 :=
by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_positions_allow_cube_folding_l576_57648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l576_57691

/-- Sequence definition -/
noncomputable def a : ℕ → ℝ → ℝ
| 0, t => 2 * t - 3
| n + 1, t => ((2 * t^(n + 2) - 3) * a n t + 2 * (t - 1) * t^(n + 1) - 1) / (a n t + 2 * t^(n + 1) - 1)

/-- Function f definition -/
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 4)

theorem sequence_properties :
  ∀ t : ℝ, t ≠ 1 ∧ t ≠ -1 →
  (∀ n : ℕ, (2^(n + 1) - 1) / (a (n + 1) 2 + 1) - (2^n - 1) / (a n 2 + 1) = 1/2) ∧
  (t > 0 → ∀ n : ℕ, a (n + 1) t > a n t) ∧
  (∀ t' : ℕ, t' ≥ 3 → ∀ n : ℕ, f (a (n + 1) (t' : ℝ)) < f (a n (t' : ℝ))) ∧
  (∀ t' : ℕ, t' < 3 → ∃ n : ℕ, f (a (n + 1) (t' : ℝ)) ≥ f (a n (t' : ℝ))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l576_57691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l576_57635

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with S_3 = 2 and S_6 = 18, S_10 / S_5 = 33 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) :
  (geometric_sum a q 3 = 2) →
  (geometric_sum a q 6 = 18) →
  (geometric_sum a q 10) / (geometric_sum a q 5) = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l576_57635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_problem_l576_57677

theorem pythagorean_triple_problem (x y z : ℤ) 
  (h1 : x^2 + 12^2 = y^2) 
  (h2 : x^2 + 40^2 = z^2) : 
  x^2 + y^2 - z^2 = -1375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_triple_problem_l576_57677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_theorem_l576_57659

/-- The area of a curvilinear triangle formed by two externally tangent circles -/
noncomputable def curvilinear_triangle_area (d α : ℝ) : ℝ :=
  (d^2 / 8) * (4 * Real.cos (α/2) - Real.pi * (1 + Real.sin (α/2)^2) + 2 * α * Real.sin (α/2))

/-- Theorem: The area of the curvilinear triangle bounded by a segment of one tangent
    and two corresponding arcs of two externally tangent circles -/
theorem curvilinear_triangle_area_theorem
  (d α : ℝ)
  (h_d : d > 0)
  (h_α : 0 < α ∧ α < π) :
  ∃ (area : ℝ),
    area = curvilinear_triangle_area d α ∧
    area > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_triangle_area_theorem_l576_57659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_seventh_number_l576_57603

theorem twenty_seventh_number (numbers : List ℝ) 
  (h1 : numbers.length = 40)
  (h2 : numbers.sum / 40 = 55.8)
  (h3 : (numbers.take 15).sum / 15 = 53.2)
  (h4 : ((numbers.drop 15).take 12).sum / 12 = 52.1)
  (h5 : (numbers.drop 27).sum / 13 = 60.5) :
  numbers[26]! = 52.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_seventh_number_l576_57603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_marks_of_passed_boys_l576_57608

theorem exam_average_marks_of_passed_boys 
  (total_boys : ℕ) 
  (overall_average : ℝ) 
  (passed_boys : ℕ) 
  (failed_average : ℝ) :
  total_boys = 140 →
  overall_average = 40 →
  passed_boys = 125 →
  failed_average = 15 →
  ∃! P : ℝ, 
    (P * (passed_boys : ℝ) + failed_average * ((total_boys - passed_boys) : ℝ)) / (total_boys : ℝ) = overall_average :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_marks_of_passed_boys_l576_57608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l576_57633

/-- The time taken for a train to pass a platform -/
noncomputable def train_passing_time (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 360 m, traveling at 45 km/hr, takes 56 seconds to pass a platform of length 340 m -/
theorem train_passing_platform : train_passing_time 360 340 45 = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l576_57633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l576_57637

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def g (x : ℝ) : ℝ := (f x + f (-x)) / 2

noncomputable def h (x : ℝ) : ℝ := (f x - f (-x)) / 2

noncomputable def p (t m : ℝ) : ℝ := g (2 * Real.arsinh (t / 2)) + 2 * m * t + m^2 - m - 1

theorem range_of_m (hf : ∀ x, f x = g x + h x)
                   (hg : ∀ x, g (-x) = g x)
                   (hh : ∀ x, h (-x) = -h x)
                   (h_mono : StrictMono h)
                   (h_range : Set.range h = Set.Icc (3/2) (15/4)) :
  {m : ℝ | ∀ x ∈ Set.Icc 1 2, p (h x) m ≥ m^2 - m - 1} = Set.Ici (-17/12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l576_57637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l576_57661

-- Define the rectangle ABCD
noncomputable def AB : ℝ := 15 * Real.sqrt 2
noncomputable def BC : ℝ := 10 * Real.sqrt 2

-- Define the point P where diagonals intersect
noncomputable def P : ℝ × ℝ × ℝ := (0, 475 / (2 * Real.sqrt 425), 75 / Real.sqrt 185)

-- Define the base area of the pyramid
noncomputable def baseArea : ℝ := 15 * Real.sqrt 185

-- Define the height of the pyramid
noncomputable def pyramidHeight : ℝ := 75 / Real.sqrt 185

-- State that all faces of the pyramid are isosceles triangles
axiom all_faces_isosceles : True

-- Theorem: The volume of the pyramid is 375
theorem pyramid_volume : 
  (1/3 : ℝ) * baseArea * pyramidHeight = 375 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l576_57661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_cardinality_l576_57623

theorem subset_sum_cardinality (X Y : Finset ℝ) : 
  (∀ x ∈ X, 0 ≤ x ∧ x < 1) →
  (∀ y ∈ Y, 0 ≤ y ∧ y < 1) →
  (0 ∈ X ∩ Y) →
  (∀ x ∈ X, ∀ y ∈ Y, x + y ≠ 1) →
  Finset.card (Finset.image (fun (p : ℝ × ℝ) => p.1 + p.2 - ⌊p.1 + p.2⌋) (Finset.product X Y)) ≥ 
    Finset.card X + Finset.card Y - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_cardinality_l576_57623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_relation_l576_57624

theorem trigonometric_relation (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin β / Real.sin α = Real.cos (α + β)) :
  (Real.tan β = Real.sin (2 * α) / (3 - Real.cos (2 * α))) ∧
  (∀ γ : ℝ, 0 < γ ∧ γ < π / 2 → Real.tan γ ≤ Real.sqrt 2 / 4) ∧
  (∃ δ : ℝ, 0 < δ ∧ δ < π / 2 ∧ Real.tan δ = Real.sqrt 2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_relation_l576_57624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_parabola_final_equation_l576_57683

/-- A parabola with focus F and parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- Theorem: Given a parabola y² = 2px (p > 0) with focus F, if a line y = 4 intersects
    the y-axis at P and the parabola at Q, and |QF| = 5/4 * |PQ|, then p = 2 -/
theorem parabola_equation (C : Parabola) (F P Q : Point) :
  P.x = 0 → P.y = 4 →
  Q.y = 4 →
  Q.y^2 = 2 * C.p * Q.x →
  F.x = C.p / 2 → F.y = 0 →
  distance Q F = 5/4 * distance P Q →
  C.p = 2 := by
  sorry

/-- Corollary: The equation of the parabola is y² = 4x -/
theorem parabola_final_equation (C : Parabola) (F P Q : Point) :
  P.x = 0 → P.y = 4 →
  Q.y = 4 →
  Q.y^2 = 2 * C.p * Q.x →
  F.x = C.p / 2 → F.y = 0 →
  distance Q F = 5/4 * distance P Q →
  ∀ (x y : ℝ), y^2 = 4 * x ↔ y^2 = 2 * C.p * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_parabola_final_equation_l576_57683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l576_57618

-- Define the second quadrant
def second_quadrant (θ : Real) : Prop :=
  Real.pi / 2 < θ ∧ θ < Real.pi

-- Define a point in the fourth quadrant
def fourth_quadrant (x y : Real) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem point_in_fourth_quadrant (θ : Real) :
  second_quadrant θ → fourth_quadrant (Real.sin θ) (Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_fourth_quadrant_l576_57618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l576_57628

theorem expression_equals_one 
  (a b c : ℝ) 
  (h : (a - 16)^2 + Real.sqrt (b - 27) + |c - 2| = 0) : 
  (Real.sqrt a - (b ^ (1/3)))^c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l576_57628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l576_57600

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  first_term : a 1 = 1
  fifth_term : a 5 = 8 * a 2
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℚ :=
  (1 - (seq.a 2 / seq.a 1) ^ n) / (1 - seq.a 2 / seq.a 1)

/-- The main theorem -/
theorem geometric_sequence_sum (seq : GeometricSequence) :
  ∃ n : ℕ, sum_n seq n = 1023 ∧ n = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l576_57600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_rotation_l576_57696

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
noncomputable def P (ω t : ℝ) : ℝ × ℝ := (Real.cos (ω * t), Real.sin (ω * t))

-- Define the point Q based on P's coordinates
noncomputable def Q (ω t : ℝ) : ℝ × ℝ :=
  let (x, y) := P ω t
  (-2 * x * y, y^2 - x^2)

-- State the theorem
theorem Q_rotation (ω : ℝ) :
  (∀ t, unit_circle (P ω t).1 (P ω t).2) →
  (∀ t, unit_circle (Q ω t).1 (Q ω t).2) ∧
  (∀ t, Q ω t = (Real.cos (3/2 * π - 2 * ω * t), Real.sin (3/2 * π - 2 * ω * t))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_rotation_l576_57696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l576_57680

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 1 then x^2
  else if x > 1 ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside the given ranges

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_five_sixths_l576_57680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_theorem_l576_57627

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the condition |BC| = 2
noncomputable def BC_length (t : Triangle) : ℝ := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)

-- Define the condition |AB|/|AC| = m
def AB_AC_ratio (t : Triangle) (m : ℝ) : Prop :=
  let AB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let AC := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  AB / AC = m

-- Define the trajectory equations
def trajectory_equation (m : ℝ) : Set (ℝ × ℝ) :=
  if m = 1 then {p : ℝ × ℝ | p.1 = 0}
  else if m = 0 then {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 + 1 = 0}
  else {p : ℝ × ℝ | (p.1 + (1+m^2)/(1-m^2))^2 + p.2^2 = (2*m/(1-m^2))^2}

theorem trajectory_theorem (t : Triangle) (m : ℝ) :
  BC_length t = 2 →
  AB_AC_ratio t m →
  t.A ∈ trajectory_equation m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_theorem_l576_57627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_specific_distance_l576_57660

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_on_parabola_with_specific_distance (x y : ℝ) :
  parabola x y →
  distance (x, y) focus = 4 →
  x = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_specific_distance_l576_57660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l576_57621

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3) + b

theorem function_values (a b : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≥ -2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≤ Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = -2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a b x = Real.sqrt 3) →
  a = 2 ∧ b = Real.sqrt 3 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l576_57621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irie_product_theorem_l576_57670

/-- An irie number is of the form 1 + 1/k for some positive integer k. -/
def IsIrie (x : ℚ) : Prop := ∃ k : ℕ+, x = 1 + 1 / k

/-- 
Given an integer n ≥ 2 and an integer r ≥ n-1, 
there exist r distinct irie numbers whose product is n.
-/
theorem irie_product_theorem (n r : ℕ) (hn : n ≥ 2) (hr : r ≥ n - 1) :
  ∃ (irie_nums : Finset ℚ), 
    (∀ x ∈ irie_nums, IsIrie x) ∧ 
    irie_nums.card = r ∧
    (irie_nums.prod id) = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irie_product_theorem_l576_57670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l576_57601

-- Define the parabola
noncomputable def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the focus coordinates
noncomputable def focus_coordinates (a : ℝ) : ℝ × ℝ := (0, 1 / (4 * a) + 2)

-- Theorem statement
theorem parabola_focus_coordinates (a b : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, |parabola a b x| ≥ 2) :
  focus_coordinates a = (0, 1 / (4 * a) + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l576_57601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_ratio_l576_57681

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem at_hash_ratio : (at_op 7 3) / (hash_op 7 3) = -12/53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_hash_ratio_l576_57681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_l576_57647

/-- The function f(x) = x^2 --/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = ln x --/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The distance between points M and N --/
noncomputable def distance (t : ℝ) : ℝ := f t - g t

/-- The theorem stating that the distance |MN| is minimized when t = √2/2 --/
theorem distance_minimized :
  ∃ (t : ℝ), t > 0 ∧ ∀ (s : ℝ), s > 0 → distance t ≤ distance s ∧ t = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_minimized_l576_57647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l576_57673

open Real Set

noncomputable def f (x : ℝ) := 4 * sin (2 * x + π / 6)

theorem intersection_sum (m : ℝ) (x₁ x₂ x₃ : ℝ) :
  x₁ ∈ Icc 0 (7 * π / 6) →
  x₂ ∈ Icc 0 (7 * π / 6) →
  x₃ ∈ Icc 0 (7 * π / 6) →
  x₁ < x₂ →
  x₂ < x₃ →
  f x₁ = m →
  f x₂ = m →
  f x₃ = m →
  x₁ + 2 * x₂ + x₃ = 5 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l576_57673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_equals_forty_l576_57684

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem sum_five_equals_forty (seq : ArithmeticSequence) 
  (h : seq.a 1 + seq.a 5 = 16) : sum_n seq 5 = 40 := by
  sorry

#check sum_five_equals_forty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_five_equals_forty_l576_57684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_l576_57671

/-- The function that returns the integer closest to a given real number -/
noncomputable def closest_integer (x : ℝ) : ℤ :=
  ⌊x + 1/2⌋

/-- The n-th positive integer that is not a perfect square -/
def f (n : ℕ+) : ℕ :=
  sorry

theorem nth_non_square (n : ℕ+) : 
  f n = n + closest_integer (Real.sqrt n.val) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_non_square_l576_57671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l576_57694

/-- Given a function f(x) = 3sin(2x + φ) that is symmetric about x = 2π/3, 
    the minimum absolute value of φ is π/6 -/
theorem min_abs_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = 3 * Real.sin (2 * x + φ)) →
  (∀ x, f (2 * π / 3 - x) = f (2 * π / 3 + x)) →
  |φ| ≥ π / 6 ∧ ∃ φ₀, |φ₀| = π / 6 ∧ 
    (∀ x, 3 * Real.sin (2 * x + φ₀) = 3 * Real.sin (2 * (2 * π / 3 - x) + φ₀)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l576_57694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_is_sqrt_3_l576_57655

def A : Fin 3 → ℝ := ![1, 1, 1]
def B : Fin 3 → ℝ := ![2, 2, 2]
def C : Fin 3 → ℝ := ![3, 2, 4]

def vector_AB : Fin 3 → ℝ := λ i => B i - A i
def vector_AC : Fin 3 → ℝ := λ i => C i - A i

noncomputable def area_triangle_ABC : ℝ := 
  Real.sqrt (
    (((vector_AB 1 * vector_AC 2) - (vector_AB 2 * vector_AC 1))^2 +
     ((vector_AB 2 * vector_AC 0) - (vector_AB 0 * vector_AC 2))^2 +
     ((vector_AB 0 * vector_AC 1) - (vector_AB 1 * vector_AC 0))^2) / 2
  )

theorem area_triangle_ABC_is_sqrt_3 : area_triangle_ABC = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABC_is_sqrt_3_l576_57655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l576_57676

/-- Represents a triangle -/
structure Triangle where
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The area of the triangle -/
  area : ℝ

/-- Predicate indicating that a triangle is isosceles and right-angled -/
def IsoscelesRight (t : Triangle) : Prop :=
  sorry

/-- An isosceles right triangle with hypotenuse 6√2 has an area of 18 square units -/
theorem isosceles_right_triangle_area : 
  ∀ (t : Triangle), 
    IsoscelesRight t → 
    t.hypotenuse = 6 * Real.sqrt 2 → 
    t.area = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l576_57676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l576_57662

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + (9/2) * x^2 - 3*x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l576_57662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_four_digit_number_l576_57651

theorem least_positive_four_digit_number (x : ℤ) : x = 1068 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧ 
  (∀ y : ℤ, y ≥ 1000 ∧ y < 10000 → 
    ((3 * y) % 18 = 12 ∧ 
     (5 * y + 20) % 15 = 35 ∧ 
     (34 - (3 * y % 34)) % 34 = (2 * y) % 34) → 
    x ≤ y) ∧
  (3 * x) % 18 = 12 ∧ 
  (5 * x + 20) % 15 = 35 ∧ 
  (34 - (3 * x % 34)) % 34 = (2 * x) % 34 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_four_digit_number_l576_57651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l576_57622

noncomputable section

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of eccentricity for an ellipse -/
def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- Definition of dot product for 2D vectors -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (2^2 / a^2) + (3^2 / b^2) = 1) 
    (h4 : Eccentricity a b = 1/2) :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ Ellipse 4 (2 * Real.sqrt 3) ↔ (x, y) ∈ Ellipse a b) ∧ 
    (k = 1 ∨ k = -1) ∧
    (∃ (M N : ℝ × ℝ), 
      M ∈ Ellipse 4 (2 * Real.sqrt 3) ∧ 
      N ∈ Ellipse 4 (2 * Real.sqrt 3) ∧ 
      M.2 = k * M.1 + 4 ∧ 
      N.2 = k * N.1 + 4 ∧ 
      M ≠ N ∧
      DotProduct M N = 16/7) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l576_57622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_scaled_vector_l576_57642

def v1 : Fin 3 → ℝ := ![4, -5, 2]
def v2 : Fin 3 → ℝ := ![-3, 6, -4]
def scalar : ℝ := 3

theorem dot_product_scaled_vector :
  (scalar * v1 0 * v2 0 + scalar * v1 1 * v2 1 + scalar * v1 2 * v2 2) = -150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_scaled_vector_l576_57642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l576_57674

-- Define the ellipse F
def F (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line l
def l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the midpoint G
def midpoint_of (x₁ y₁ x₂ y₂ xG yG : ℝ) : Prop :=
  xG = (x₁ + x₂) / 2 ∧ yG = (y₁ + y₂) / 2

-- Define the condition OQ = 2OG
def ray_condition (xG yG xQ yQ : ℝ) : Prop :=
  xQ = 2 * xG ∧ yQ = 2 * yG

theorem ellipse_theorem
  (a b k m : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_D : F a b 2 0)
  (h_E : F a b 1 (Real.sqrt 3 / 2))
  (h_distinct : ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ F a b x₁ y₁ ∧ F a b x₂ y₂ ∧ l k m x₁ y₁ ∧ l k m x₂ y₂)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ xG yG : ℝ), midpoint_of x₁ y₁ x₂ y₂ xG yG ∧ F a b x₁ y₁ ∧ F a b x₂ y₂ ∧ l k m x₁ y₁ ∧ l k m x₂ y₂)
  (h_ray : ∃ (xG yG xQ yQ : ℝ), ray_condition xG yG xQ yQ ∧ F a b xQ yQ) :
  (a = 2 ∧ b = 1) ∧
  (4 * m^2 = 4 * k^2 + 1) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), F 2 1 x₁ y₁ ∧ F 2 1 x₂ y₂ ∧ l k m x₁ y₁ ∧ l k m x₂ y₂ ∧
    Real.sqrt 3 / 2 = abs (m * (x₂ - x₁)) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l576_57674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l576_57643

noncomputable def f (x : ℝ) := 2 * (Real.sin (x - Real.pi / 4))^2 - 1

theorem f_properties : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l576_57643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_g_l576_57678

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The function g is the 2009-fold composition of f -/
def g : ℝ → ℝ := (f^[2009])

/-- The coefficient of x^(2^2009 - 1) in the expansion of g(x) -/
def a : ℝ := 2^2009

/-- Theorem stating that the coefficient of x^(2^2009 - 1) in g(x) is 2^2009 -/
theorem coefficient_of_g : 
  ∃ (p : Polynomial ℝ), (g = λ x ↦ p.eval x) ∧ 
  p.coeff (2^2009 - 1) = a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_g_l576_57678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_pie_cost_l576_57632

/-- Represents the cost of crust ingredients -/
def crust_cost : ℚ := 2 + 1 + (3/2)

/-- Represents the cost of blueberries at Store A -/
def blueberry_cost_A : ℚ := 6 * (9/4)

/-- Represents the cost of blueberries at Store B -/
def blueberry_cost_B : ℚ := (2 * 7 * (4/5)) + 7

/-- Represents the cost of cherries at Store A -/
def cherry_cost_A : ℚ := 14

/-- Represents the cost of cherries at Store B -/
def cherry_cost_B : ℚ := 3 * 4

/-- Represents the total cost of the blueberry pie -/
noncomputable def blueberry_pie_cost : ℚ := crust_cost + min blueberry_cost_A blueberry_cost_B

/-- Represents the total cost of the cherry pie -/
noncomputable def cherry_pie_cost : ℚ := crust_cost + min cherry_cost_A cherry_cost_B

/-- Theorem stating that the minimum cost of making either pie is $16.5 -/
theorem cheapest_pie_cost : min blueberry_pie_cost cherry_pie_cost = 33/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheapest_pie_cost_l576_57632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_daily_sorting_is_220000_l576_57669

/-- Represents the daily sorting deviation in units of 10,000 packages -/
def SortingDeviation := List Int

/-- Calculates the average daily sorting volume given the planned volume and deviations -/
def averageDailySorting (plannedDaily : ℕ) (deviations : SortingDeviation) : ℕ :=
  let totalPlanned := plannedDaily * 7
  let totalDeviation := deviations.sum * 10000
  ((totalPlanned : Int) + totalDeviation).toNat / 7

/-- Theorem stating that the average daily sorting volume is 220,000 packages -/
theorem average_daily_sorting_is_220000 (plannedDaily : ℕ) (deviations : SortingDeviation) :
  plannedDaily = 200000 →
  deviations = [6, 4, -6, 8, -1, 7, -4] →
  averageDailySorting plannedDaily deviations = 220000 := by
  sorry

#eval averageDailySorting 200000 [6, 4, -6, 8, -1, 7, -4]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_daily_sorting_is_220000_l576_57669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_product_l576_57685

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x - x^2 else x^2 + 2 * x

-- State the theorem
theorem odd_function_range_product (a b : ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = 2 * x - x^2) →  -- definition for x ≥ 0
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) →  -- f maps [a,b] onto [1/b, 1/a]
  (∀ y ∈ Set.Icc (1/b) (1/a), ∃ x ∈ Set.Icc a b, f x = y) →
  a * b = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_product_l576_57685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_symmetrical_shapes_l576_57610

-- Define the set of shapes we're considering
inductive Shape
  | EquilateralTriangle
  | IsoscelesTrapezoid
  | Parallelogram
  | RegularPentagon

-- Define a property for being symmetrical about an axis
def isAxisSymmetrical (s : Shape) : Bool :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.IsoscelesTrapezoid => true
  | Shape.Parallelogram => false
  | Shape.RegularPentagon => true

-- Theorem statement
theorem axis_symmetrical_shapes :
  (List.filter isAxisSymmetrical [Shape.EquilateralTriangle, Shape.IsoscelesTrapezoid, Shape.Parallelogram, Shape.RegularPentagon]).length = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_symmetrical_shapes_l576_57610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_l576_57698

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the inclination angle of a line
noncomputable def inclinationAngle (l : Line) : ℝ :=
  if l.b = 0 then Real.pi / 2 else Real.arctan (- l.a / l.b)

-- The theorem to prove
theorem vertical_line_inclination :
  let l : Line := { a := 1, b := 0, c := 3 }
  inclinationAngle l = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_l576_57698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l576_57668

noncomputable def f (x : ℝ) : ℝ := (1/5) * Real.sin (x + Real.pi/3) + Real.cos (x - Real.pi/6)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 6/5 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l576_57668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l576_57699

/-- Calculates the true discount on a bill -/
noncomputable def true_discount (amount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (amount * rate * time) / (100 + (rate * time))

/-- Theorem: The true discount on a bill of Rs. 1960 due in 9 months with 16% annual interest is Rs. 210 -/
theorem bill_true_discount : 
  let amount : ℝ := 1960
  let rate : ℝ := 16
  let time : ℝ := 9 / 12
  true_discount amount rate time = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_true_discount_l576_57699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_distance_4_has_x_coord_3_l576_57616

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_on_parabola_with_distance_4_has_x_coord_3 
  (M : ℝ × ℝ) (h1 : parabola M.1 M.2) (h2 : distance M focus = 4) :
  M.1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_parabola_with_distance_4_has_x_coord_3_l576_57616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l576_57664

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x - 2)^2 + 4)

-- State the theorem
theorem f_minimum : 
  (∀ x : ℝ, f 2 ≤ f x) ∧ f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_l576_57664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_135_degrees_l576_57630

theorem cosecant_135_degrees : 1 / Real.sin (135 * π / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_135_degrees_l576_57630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_moments_of_inertia_l576_57638

/-- Moment of inertia of a solid ball -/
noncomputable def moment_of_inertia_solid_ball (mass : ℝ) (radius : ℝ) : ℝ :=
  (2/5) * mass * radius^2

/-- Moment of inertia of a hollow sphere -/
noncomputable def moment_of_inertia_hollow_sphere (mass : ℝ) (radius : ℝ) : ℝ :=
  (2/3) * mass * radius^2

/-- Theorem stating that the moments of inertia are different for objects with the same mass and radius -/
theorem different_moments_of_inertia (mass : ℝ) (radius : ℝ) 
  (h1 : mass > 0) (h2 : radius > 0) : 
  moment_of_inertia_solid_ball mass radius ≠ moment_of_inertia_hollow_sphere mass radius :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_moments_of_inertia_l576_57638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l576_57688

noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_of_h :
  ∀ x : ℝ, x ≠ 5 ↔ h x ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l576_57688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_theorem_l576_57631

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 2 * Real.sin (3 * x)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop := y' = Real.sin x'

-- Define the scaling transformation
def scaling_transform (x y x' y' a b : ℝ) : Prop :=
  x' = a * x ∧ y' = b * y ∧ a > 0 ∧ b > 0

-- Theorem statement
theorem scaling_transformation_theorem :
  ∃ (a b : ℝ), ∀ (x y x' y' : ℝ),
    original_curve x y →
    transformed_curve x' y' →
    scaling_transform x y x' y' a b →
    a = 3 ∧ b = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_theorem_l576_57631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_negative_example_frac_example_gaussian_range_frac_range_l576_57626

-- Define the Gaussian function
noncomputable def gaussian (x : ℝ) : ℤ := 
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := 
  x - (Int.floor x : ℝ)

-- Theorem statements
theorem gaussian_negative_example : gaussian (-4.1) = -5 := by sorry

theorem frac_example : frac 3.5 = 0.5 := by sorry

theorem gaussian_range (x : ℝ) : 
  gaussian x = -3 ↔ -3 ≤ x ∧ x < -2 := by sorry

theorem frac_range (x : ℝ) : 
  2.5 < x ∧ x ≤ 3.5 → 0 ≤ frac x ∧ frac x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gaussian_negative_example_frac_example_gaussian_range_frac_range_l576_57626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_20_days_l576_57693

/-- The time (in days) it takes for person A to complete the work alone -/
noncomputable def time_a : ℝ := 30

/-- The time (in days) it takes for person B to complete the work alone -/
noncomputable def time_b : ℝ := 15

/-- The number of days A and B work together before B leaves -/
noncomputable def days_together : ℝ := 5

/-- The work rate of person A per day -/
noncomputable def rate_a : ℝ := 1 / time_a

/-- The work rate of person B per day -/
noncomputable def rate_b : ℝ := 1 / time_b

/-- The combined work rate of A and B when working together -/
noncomputable def combined_rate : ℝ := rate_a + rate_b

/-- Theorem stating that the total time to complete the work is 20 days -/
theorem total_time_is_20_days : 
  days_together + (1 - days_together * combined_rate) / rate_a = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_is_20_days_l576_57693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_bases_l576_57625

theorem min_sum_of_bases (c d : ℕ) (hc : c > 0) (hd : d > 0) : 
  (5 * c + 7 = 7 * d + 5) → 
  (∀ c' d' : ℕ, c' > 0 → d' > 0 → 5 * c' + 7 = 7 * d' + 5 → c + d ≤ c' + d') → 
  c + d = 14 := by
  sorry

#check min_sum_of_bases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_bases_l576_57625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_identity_right_angled_triangle_condition_l576_57609

-- Part 1: Trigonometric identity
theorem cosine_difference_identity (A B : ℝ) :
  Real.cos A - Real.cos B = -2 * Real.sin ((A + B) / 2) * Real.sin ((A - B) / 2) := by sorry

-- Part 2: Triangle shape determination
theorem right_angled_triangle_condition (A B C : ℝ) 
  (h1 : A + B + C = Real.pi) -- Sum of angles in a triangle
  (h2 : 0 < A ∧ A < Real.pi) -- Angle constraints
  (h3 : 0 < B ∧ B < Real.pi)
  (h4 : 0 < C ∧ C < Real.pi)
  (h5 : Real.cos (2*A) - Real.cos (2*B) = 1 - Real.cos (2*C)) : -- Given condition
  B = Real.pi/2 := by sorry -- B is a right angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_difference_identity_right_angled_triangle_condition_l576_57609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_faucets_fill_time_l576_57649

-- Define the given conditions
def initial_tub_capacity : ℚ := 160
def initial_faucet_count : ℕ := 4
def initial_time : ℚ := 8
def final_tub_capacity : ℚ := 40
def final_faucet_count : ℕ := 8

-- Define the function to calculate filling time
def filling_time (tub_capacity : ℚ) (faucet_count : ℕ) (flow_rate : ℚ) : ℚ :=
  tub_capacity / (faucet_count * flow_rate)

-- Theorem to prove
theorem eight_faucets_fill_time :
  let flow_rate : ℚ := initial_tub_capacity / (initial_faucet_count * initial_time)
  filling_time final_tub_capacity final_faucet_count flow_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_faucets_fill_time_l576_57649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_expense_l576_57636

def monthly_salary : ℕ := 18000
def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def miscellaneous : ℕ := 700
def savings : ℕ := 1800
def savings_rate : ℚ := 1/10

theorem petrol_expense (petrol : ℕ) : 
  savings = (savings_rate * monthly_salary).floor ∧
  monthly_salary = rent + milk + groceries + education + petrol + miscellaneous + savings →
  petrol = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petrol_expense_l576_57636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobbys_remaining_gasoline_l576_57692

/-- Calculates the remaining gasoline after a trip -/
noncomputable def remaining_gasoline (initial_gasoline : ℝ) (consumption_rate : ℝ) (distance : ℝ) : ℝ :=
  initial_gasoline - distance / consumption_rate

/-- Theorem: Bobby's remaining gasoline after his trips -/
theorem bobbys_remaining_gasoline :
  let initial_gasoline : ℝ := 12
  let consumption_rate : ℝ := 2
  let distance : ℝ := 20
  remaining_gasoline initial_gasoline consumption_rate distance = 2 := by
  -- Unfold the definition of remaining_gasoline
  unfold remaining_gasoline
  -- Simplify the arithmetic
  simp [sub_eq_iff_eq_add]
  -- Prove the equality
  ring

#check bobbys_remaining_gasoline

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobbys_remaining_gasoline_l576_57692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l576_57639

theorem tan_roots_sum (α β : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : β ∈ Set.Ioo 0 π) 
  (h3 : (6 : ℝ) * (Real.tan α)^2 - 5 * (Real.tan α) + 1 = 0) 
  (h4 : (6 : ℝ) * (Real.tan β)^2 - 5 * (Real.tan β) + 1 = 0) : 
  α + β = π/4 ∨ α + β = 5*π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_roots_sum_l576_57639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_equals_3_l576_57663

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -3 then 3 * x - 7
  else -x^2 + 3 * x - 3

-- Theorem statement
theorem no_solutions_for_f_equals_3 :
  ¬ ∃ x : ℝ, f x = 3 := by
  sorry

-- Helper lemma for the case x < -3
lemma no_solution_left :
  ¬ ∃ x : ℝ, x < -3 ∧ f x = 3 := by
  sorry

-- Helper lemma for the case x ≥ -3
lemma no_solution_right :
  ¬ ∃ x : ℝ, x ≥ -3 ∧ f x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_equals_3_l576_57663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryce_received_15_raisins_l576_57686

-- Define the number of raisins each person received
def bryce_raisins : ℕ := sorry
def carter_raisins : ℕ := sorry
def april_raisins : ℕ := sorry

-- Define the conditions from the problem
axiom bryce_more_than_carter : bryce_raisins = carter_raisins + 10
axiom carter_one_third_of_bryce : carter_raisins = bryce_raisins / 3
axiom april_twice_carter : april_raisins = 2 * carter_raisins

-- Theorem to prove
theorem bryce_received_15_raisins : bryce_raisins = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryce_received_15_raisins_l576_57686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bound_l576_57615

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a 3 x 4 rectangle -/
def isInRectangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 3

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For any 4 points in a 3 x 4 rectangle, 
    there exist at least two points whose distance is not greater than 25/8 -/
theorem distance_bound (p1 p2 p3 p4 : Point) 
    (h1 : isInRectangle p1) (h2 : isInRectangle p2) 
    (h3 : isInRectangle p3) (h4 : isInRectangle p4) : 
  ∃ (i j : Fin 4), i ≠ j ∧ 
    distance (match i with 
              | 0 => p1 
              | 1 => p2 
              | 2 => p3 
              | 3 => p4) 
             (match j with 
              | 0 => p1 
              | 1 => p2 
              | 2 => p3 
              | 3 => p4) ≤ 25/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bound_l576_57615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_y_value_l576_57634

-- Define the start point of the line segment
def start : ℝ × ℝ := (1, -3)

-- Define the end point of the line segment
def end_point (y : ℝ) : ℝ × ℝ := (y, 4)

-- Define the length of the line segment
def segment_length : ℝ := 10

-- Define the property of being an isosceles triangle with base parallel to x-axis
def is_isosceles_with_horizontal_base (start end_point : ℝ × ℝ) : Prop :=
  ∃ (base_start base_end : ℝ × ℝ),
    base_start.2 = base_end.2 ∧
    (start.1 - base_start.1)^2 + (start.2 - base_start.2)^2 =
    (end_point.1 - base_end.1)^2 + (end_point.2 - base_end.2)^2

-- Theorem statement
theorem line_segment_y_value (y : ℝ) :
  (end_point y).1 - start.1 = y - 1 ∧
  (end_point y).2 - start.2 = 7 ∧
  (y - 1)^2 + 7^2 = segment_length^2 ∧
  is_isosceles_with_horizontal_base start (end_point y) →
  y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_y_value_l576_57634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_f_two_zeros_product_lt_one_l576_57675

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

-- Theorem 1: If f(x) ≥ 0 for all x > 0, then a ≤ e + 1
theorem f_nonnegative_implies_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) → a ≤ Real.exp 1 + 1 := by
  sorry

-- Theorem 2: If f has two positive zeros, their product is less than 1
theorem f_two_zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bound_f_two_zeros_product_lt_one_l576_57675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_withdrawal_is_zero_l576_57617

/-- Represents a bank account with associated operations and constraints. -/
structure BankAccount where
  /-- The initial balance of the account. -/
  initialBalance : ℚ
  /-- The amount of the initial withdrawal. -/
  initialWithdrawal : ℚ
  /-- The fraction of the initial balance that the withdrawal represents. -/
  withdrawalFraction : ℚ
  /-- The minimum balance that must be maintained. -/
  minBalance : ℚ
  /-- The fraction of the remaining balance that is deposited. -/
  depositFraction : ℚ

/-- Calculates the minimum amount that can be withdrawn while maintaining the minimum balance. -/
def minWithdrawal (account : BankAccount) : ℚ :=
  let remainingBalance := account.initialBalance - account.initialWithdrawal
  let depositAmount := account.depositFraction * remainingBalance
  let finalBalance := remainingBalance + depositAmount
  max (finalBalance - account.minBalance) 0

/-- Theorem stating that under the given conditions, the minimum withdrawal is zero. -/
theorem min_withdrawal_is_zero (account : BankAccount) 
  (h1 : account.initialWithdrawal = 400)
  (h2 : account.withdrawalFraction = 2/5)
  (h3 : account.minBalance = 300)
  (h4 : account.depositFraction = 1/4)
  (h5 : account.initialBalance * account.withdrawalFraction = account.initialWithdrawal) :
  minWithdrawal account = 0 := by
  sorry

def main : IO Unit := do
  let result := minWithdrawal { 
    initialBalance := 1000, 
    initialWithdrawal := 400, 
    withdrawalFraction := 2/5, 
    minBalance := 300, 
    depositFraction := 1/4 
  }
  IO.println s!"Minimum withdrawal: {result}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_withdrawal_is_zero_l576_57617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_equals_four_l576_57612

theorem fraction_power_equals_four : (1 / 16 : ℝ) ^ (-1 / 2 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_power_equals_four_l576_57612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l576_57687

/-- Represents the time (in hours) to fill a cistern -/
noncomputable def fill_time : ℝ := 4

/-- Represents the time (in hours) to empty a cistern -/
noncomputable def empty_time : ℝ := 5

/-- Represents the capacity of the cistern -/
noncomputable def cistern_capacity : ℝ := 1

/-- Calculates the time to fill the cistern when both taps are open -/
noncomputable def time_to_fill : ℝ :=
  cistern_capacity / (1 / fill_time - 1 / empty_time)

theorem cistern_fill_time :
  time_to_fill = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l576_57687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_march_rainfall_l576_57689

/-- Given the rainfall in April and the difference between March and April rainfall,
    prove that the rainfall in March was 0.81 inches. -/
theorem march_rainfall (april_rainfall : ℝ) (rainfall_difference : ℝ) (march_rainfall : ℝ)
  (h1 : april_rainfall = 0.46)
  (h2 : rainfall_difference = 0.35)
  (h3 : april_rainfall + rainfall_difference = march_rainfall) :
  march_rainfall = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_march_rainfall_l576_57689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_P_2004_equals_half_l576_57602

noncomputable def P (x : ℝ) : ℝ := x^3 - 3/2 * x^2 + x + 1/4

noncomputable def P_iter : ℕ → (ℝ → ℝ)
| 0 => id
| 1 => P
| (n+1) => P_iter n ∘ P

theorem integral_P_2004_equals_half :
  ∫ x in (0:ℝ)..1, P_iter 2004 x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_P_2004_equals_half_l576_57602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l576_57640

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x - 2)

theorem oblique_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - (3 * x - 2)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l576_57640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_l576_57605

/-- Represents a 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the walk described in the problem --/
noncomputable def walk (x : ℝ) : Point :=
  let northPoint := Point.mk 0 x
  let angle : ℝ := 120 * Real.pi / 180  -- 120 degrees in radians
  let finalX := northPoint.x - 4 * Real.sin angle
  let finalY := northPoint.y - 4 * Real.cos angle
  Point.mk finalX finalY

/-- The theorem to be proved --/
theorem walk_distance (x : ℝ) : 
  distance (Point.mk 0 0) (walk x) = 2 → x = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_distance_l576_57605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_sum_five_l576_57690

noncomputable section

-- Define the exponential sequence
def exponential_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Define the sum of the first n terms
def sum_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem exponential_sequence_sum_five
  (a₁ : ℝ) (q : ℝ)
  (h_pos : a₁ > 0 ∧ q > 0)
  (h_a₃ : exponential_sequence a₁ q 3 = 4)
  (h_a₂a₆ : exponential_sequence a₁ q 2 * exponential_sequence a₁ q 6 = 64) :
  sum_of_terms a₁ q 5 = 31 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_sum_five_l576_57690
