import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_definition_not_in_same_plane_skew_l263_26363

-- Define the basic concepts
structure Line

def parallel (l1 l2 : Line) : Prop := sorry

def intersecting (l1 l2 : Line) : Prop := sorry

def skew (l1 l2 : Line) : Prop := sorry

def in_same_plane (l1 l2 : Line) : Prop := sorry

-- State the theorems to be proved
theorem skew_lines_definition (l1 l2 : Line) :
  ¬(parallel l1 l2) ∧ ¬(intersecting l1 l2) → skew l1 l2 := by
  sorry

theorem not_in_same_plane_skew (l1 l2 : Line) :
  ¬(in_same_plane l1 l2) → skew l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_definition_not_in_same_plane_skew_l263_26363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_l263_26320

noncomputable def arithmetic_sequence (start : ℕ) (step : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => start + i * step)

noncomputable def Q (sequence : List ℕ) : ℝ :=
  sequence.map (λ x => Real.log 2 / Real.log 128 * x) |>.sum

theorem Q_value : 
  let sequence := arithmetic_sequence 3 2 47
  Q sequence = 329 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_l263_26320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l263_26366

/-- Sum of all positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Number of distinct prime factors of n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem count_special_integers : 
  (Finset.filter (fun i => 
    1 ≤ i ∧ i ≤ 5000 ∧ 
    sum_of_divisors i = 1 + Real.sqrt (i : ℝ) + i + num_distinct_prime_factors i
  ) (Finset.range 5001)).card = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l263_26366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l263_26300

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l263_26300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_238_guarantee_l263_26397

theorem product_238_guarantee (S : Finset ℕ) (h_range : ∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 200) 
  (h_card : S.card = 198) : 
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b = 238 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_238_guarantee_l263_26397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_or_parallel_lines_l263_26310

/-- A parabola defined by y^2 = 8x -/
def parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 8 * p.1}

/-- The point M with coordinates (2,4) -/
def M : ℝ × ℝ := (2, 4)

/-- A line that passes through point M and intersects the parabola at only one point -/
def tangent_or_parallel_line (l : Set (ℝ × ℝ)) : Prop :=
  M ∈ l ∧ (∃! p, p ∈ l ∩ parabola)

/-- Theorem stating that there are exactly two lines passing through M 
    that intersect the parabola at only one point -/
theorem two_tangent_or_parallel_lines :
  ∃! (l₁ l₂ : Set (ℝ × ℝ)), l₁ ≠ l₂ ∧ 
    tangent_or_parallel_line l₁ ∧ 
    tangent_or_parallel_line l₂ ∧
    ∀ l, tangent_or_parallel_line l → (l = l₁ ∨ l = l₂) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_or_parallel_lines_l263_26310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_cube_root_eq_l263_26373

theorem largest_x_cube_root_eq : ∃ (x : ℝ), x = Real.sqrt 3 / 8 ∧
  (∀ (y : ℝ), y ≥ 0 → (3 * y) ^ (1/3) = 4 * y → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_x_cube_root_eq_l263_26373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_playing_theater_l263_26357

/-- The number of Ocho's friends -/
def total_friends : ℕ := 8

/-- The number of Ocho's friends who are girls -/
def girl_friends : ℕ := total_friends / 2

/-- The number of Ocho's friends who are boys -/
def boy_friends : ℕ := total_friends - girl_friends

/-- Predicate to represent "plays theater with Ocho" -/
def plays_theater_with (b : ℕ) (Ocho : ℕ) : Prop := sorry

/-- All of Ocho's boy friends play theater with him -/
axiom all_boys_play : ∀ b, b ≤ boy_friends → plays_theater_with b 0

theorem boys_playing_theater : boy_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_playing_theater_l263_26357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l263_26386

/-- A parabola is defined by the equation y = ax^2 where a ≠ 0 -/
structure Parabola where
  a : ℝ
  hne : a ≠ 0

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (0, 1 / (4 * p.a))

/-- Theorem: The focus of the parabola y = 2x^2 is at (0, 1/8) -/
theorem focus_of_specific_parabola :
  let p : Parabola := { a := 2, hne := by norm_num }
  focus p = (0, 1/8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l263_26386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_at_100_max_profit_price_max_profit_value_l263_26379

/-- Represents the price of each set of books in yuan -/
def x : ℝ → ℝ := id

/-- Calculates the sales volume in ten thousand sets -/
def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x

/-- Calculates the supply price of each set of books in yuan -/
noncomputable def supply_price (x : ℝ) : ℝ := 30 + 10 / sales_volume x

/-- Calculates the profit per set of books in yuan -/
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x

/-- Calculates the total profit in ten thousand yuan -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

/-- Theorem stating the total profit when the price is 100 yuan -/
theorem total_profit_at_100 : total_profit 100 = 340 := by sorry

/-- Theorem stating the price that maximizes the profit per set -/
theorem max_profit_price : ∃ (max_x : ℝ), max_x = 140 ∧ 
  ∀ (y : ℝ), profit_per_set y ≤ profit_per_set max_x := by sorry

/-- Theorem stating the maximum profit per set -/
theorem max_profit_value : profit_per_set 140 = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_at_100_max_profit_price_max_profit_value_l263_26379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leveling_cost_approx_l263_26314

-- Define constants
def inner_radius : ℝ := 16
def walk_width : ℝ := 3
def cost_per_sq_meter : ℝ := 2

-- Define the function to calculate the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the function to calculate the cost of leveling the walk
noncomputable def leveling_cost : ℝ :=
  (circle_area (inner_radius + walk_width) - circle_area inner_radius) * cost_per_sq_meter

-- Theorem statement
theorem leveling_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |leveling_cost - 660| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leveling_cost_approx_l263_26314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_constant_l263_26362

-- Define the line y = (5/2)x + 4
noncomputable def line (x : ℝ) : ℝ := (5/2) * x + 4

-- Define a vector on the line
noncomputable def vector_on_line (a : ℝ) : ℝ × ℝ := (a, line a)

-- Define the projection vector w
noncomputable def w (d : ℝ) : ℝ × ℝ := (-(5/2) * d, d)

-- Define the projection formula
noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_squared := w.1 * w.1 + w.2 * w.2
  let scalar := dot_product / norm_squared
  (scalar * w.1, scalar * w.2)

-- The theorem to prove
theorem projection_is_constant (a d : ℝ) :
  proj (vector_on_line a) (w d) = (-40/29, 16/29) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_is_constant_l263_26362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l263_26382

/-- Circle with center (2, 3) and radius 5 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 25}

/-- Point P -/
def P : ℝ × ℝ := (-1, 7)

/-- Line L: 3x - 4y + 31 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 31 = 0}

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * p.1 + b * p.2 + c)) / Real.sqrt (a^2 + b^2)

theorem tangent_line_proof :
  (P ∈ Line) ∧
  (∀ p ∈ Circle, distancePointToLine p 3 (-4) 31 ≥ 5) ∧
  (∃ q ∈ Circle, distancePointToLine q 3 (-4) 31 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l263_26382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_at_10_point_8_l263_26391

/-- Represents the park and walking scenario -/
structure ParkScenario where
  parkSize : ℚ
  hectorSpeed : ℚ
  windStartDistance : ℚ
  janeSpeedReduction : ℚ

/-- Calculates the meeting point of Jane and Hector -/
noncomputable def meetingPoint (scenario : ParkScenario) : ℚ :=
  let totalDistance := 4 * scenario.parkSize
  let janeNormalSpeed := 2 * scenario.hectorSpeed
  let janeReducedSpeed := janeNormalSpeed * (1 - scenario.janeSpeedReduction)
  let timeBeforeWind := scenario.windStartDistance / janeNormalSpeed
  let remainingDistance := totalDistance - scenario.windStartDistance
  let timeAfterWind := remainingDistance / (scenario.hectorSpeed + janeReducedSpeed)
  let totalTime := timeBeforeWind + timeAfterWind
  let hectorDistance := scenario.hectorSpeed * totalTime
  hectorDistance - scenario.parkSize

/-- Theorem stating that Jane and Hector meet 10.8 blocks into the second side -/
theorem meet_at_10_point_8 (scenario : ParkScenario) :
  scenario.parkSize = 24 ∧
  scenario.windStartDistance = 12 ∧
  scenario.janeSpeedReduction = 1/4 →
  meetingPoint scenario = 27/2.5 := by
  sorry

#eval (27 : ℚ) / 2.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_at_10_point_8_l263_26391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l263_26378

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - (x + 3) / (x + 1))
noncomputable def g (a x : ℝ) : ℝ := Real.log ((x - a - 1) * (2 * a - x)) / Real.log 10

-- Define the domains A and B
def A : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) :
  a < 1 → (B a ⊆ A) → a ∈ Set.Iic (-2) ∪ Set.Ico (1/2) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l263_26378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l263_26370

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.b * Real.cos t.A = (2 * t.c + t.a) * Real.cos (Real.pi - t.B) ∧
  t.b = 4 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3

theorem triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = 2 * Real.pi / 3 ∧ t.a + t.b + t.c = 4 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l263_26370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_problem_l263_26336

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (n : ℝ) / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem arithmetic_sequence_sum_problem (a : ℕ → ℝ) (n : ℕ) :
  arithmetic_sequence a →
  sum_arithmetic_sequence a 6 = 36 →
  sum_arithmetic_sequence a n = 324 →
  n > 6 →
  sum_arithmetic_sequence a (n - 6) = 144 →
  n = 18 := by
  sorry

#check arithmetic_sequence_sum_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_problem_l263_26336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_properties_l263_26338

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-1)*x + a

noncomputable def g (x : ℝ) : ℝ := 2*x / (3-x)

theorem symmetric_function_properties (a : ℝ) :
  (∀ x, f a x + f a (2-x) = 4) →
  (∀ x ∈ Set.Icc 0 1, f a x = x^2 - (a-1)*x + a) →
  (f a 0 + f a (1/2) + f a 1 + f a (3/2) + f a 2 = 10) ∧
  (∀ x, g x + g (6-x) = -4) ∧
  (∀ x1 ∈ Set.Icc 0 2, ∃ x2 ∈ Set.Icc (-3) 2, f a x1 = g x2) ↔ a ∈ Set.Icc 0 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_properties_l263_26338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l263_26319

/-- The sequence a_n is not periodic for x > 1 and x not an integer -/
theorem sequence_not_periodic (x : ℝ) (hx1 : x > 1) (hx2 : ¬ Int.floor x = ↑(Int.floor x)) :
  ¬ ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), n > 0 →
    (⌊x^(n+1)⌋ - x * ⌊x^n⌋ = ⌊x^(n+p+1)⌋ - x * ⌊x^(n+p)⌋) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_l263_26319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l263_26328

theorem constant_term_expansion (n : ℕ) (h : (1 + 1)^n = 512) :
  ∃ (k : ℕ), k = 6 ∧ 
  (n.choose k) * (-1 : ℤ)^k = 84 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l263_26328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_p_l263_26334

def sequence_property (seq : List ℤ) : Prop :=
  ∀ i, i ≥ 2 → i < seq.length → seq[i]! = seq[i-1]! + seq[i-2]!

theorem find_p (seq : List ℤ) (h_prop : sequence_property seq) 
  (h_last : seq.take 4 = [-2, 5, 3, 8]) : 
  seq[seq.length - 5]! = -25 := by
  sorry

#check find_p

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_p_l263_26334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l263_26389

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 2, 3, 7, 8, 9}

/-- A valid arrangement of digits -/
structure Arrangement where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  a_in : a ∈ digits
  b_in : b ∈ digits
  c_in : c ∈ digits
  d_in : d ∈ digits
  e_in : e ∈ digits
  f_in : f ∈ digits
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
             b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
             c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
             d ≠ e ∧ d ≠ f ∧
             e ≠ f

/-- The sum of two 3-digit numbers formed by an arrangement -/
def sum (arr : Arrangement) : Nat :=
  100 * (arr.a + arr.d) + 10 * (arr.b + arr.e) + (arr.c + arr.f)

/-- The theorem stating that 417 is the smallest possible sum -/
theorem smallest_sum :
  ∀ arr : Arrangement, sum arr ≥ 417 := by
  sorry

#check smallest_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_l263_26389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l263_26348

-- Define the values
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 2 / Real.log 0.5
def c : ℝ := 0.5^2

-- State the theorem
theorem relationship_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l263_26348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l263_26359

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1)) + x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  f (1 + a) + f (1 - a^2) < 0 → (a < -1 ∨ a > 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l263_26359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_theorem_l263_26399

/-- Represents an ellipse in the 2D plane -/
structure Ellipse where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem: Given an ellipse with center at the origin, foci at (-4, 0) and (4, 0),
    and passing through the point (5, 0), its equation is x^2/25 + y^2/9 = 1 -/
theorem ellipse_equation_theorem (E : Ellipse) :
  E.center = (0, 0) →
  E.foci = ((-4, 0), (4, 0)) →
  E.point = (5, 0) →
  ∀ x y : ℝ, ellipse_equation 25 9 x y ↔ (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (5 * Real.cos t, 3 * Real.sin t)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_theorem_l263_26399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_false_l263_26374

-- Define the four statements
def statement1 : Prop := ∀ (p q : Prop), p ∨ q → p ∧ q

def statement2 : Prop := (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧
                         (∃ x : ℝ, x^2 - 4*x - 5 > 0 ∧ x ≤ 5)

def statement3 : Prop := (¬ ∀ x : ℝ, 2^x > x^2) ↔
                         (∃ x : ℝ, 2^x ≤ x^2)

def statement4 : Prop := ∃ x : ℝ, Real.exp x = 1 + x

-- Theorem stating that exactly two of the statements are false
theorem exactly_two_false :
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_false_l263_26374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rounds_is_four_l263_26343

/-- Represents the gender of a child: Boy or Girl -/
inductive Gender : Type
| Boy : Gender
| Girl : Gender

/-- The state of the circle with 4 positions -/
structure CircleState :=
  (positions : Vector Gender 4)

/-- The rules for transitioning between states -/
def transition (state : CircleState) : CircleState :=
  sorry

/-- Predicate to check if all positions are occupied by boys -/
def all_boys (state : CircleState) : Prop :=
  ∀ i, state.positions.get i = Gender.Boy

/-- The number of children of each gender -/
def initial_counts : Nat × Nat := (5, 6)

/-- The maximum number of rounds possible -/
def max_rounds : Nat := 4

/-- Main theorem: The maximum number of rounds is 4 -/
theorem max_rounds_is_four :
  ∀ initial_state : CircleState,
  (∃ i, initial_state.positions.get i = Gender.Girl) →
  ∃ n : Nat, n ≤ max_rounds ∧
    (∃ final_state : CircleState, 
      (Nat.iterate transition n initial_state = final_state) ∧
      all_boys final_state) ∧
    ∀ m : Nat, m < n →
      ¬(∃ final_state : CircleState, 
        (Nat.iterate transition m initial_state = final_state) ∧
        all_boys final_state) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rounds_is_four_l263_26343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equality_l263_26398

theorem scientific_notation_equality : 0.0000205 = 2.05 * (10 : ℝ)^((-5) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_notation_equality_l263_26398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l263_26368

/-- Represents a route with a given distance and speed -/
structure Route where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken to travel a route in hours -/
noncomputable def travelTime (r : Route) : ℝ := r.distance / r.speed

/-- Represents Route A -/
def routeA : Route := { distance := 8, speed := 40 }

/-- Represents the non-construction part of Route B -/
def routeB1 : Route := { distance := 5, speed := 50 }

/-- Represents the construction part of Route B -/
def routeB2 : Route := { distance := 1, speed := 10 }

/-- The total time for Route B is the sum of its two parts -/
noncomputable def routeBTime : ℝ := travelTime routeB1 + travelTime routeB2

/-- Theorem stating that the time difference between Route A and Route B is 0 minutes -/
theorem route_time_difference : (travelTime routeA - routeBTime) * 60 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l263_26368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1026_is_2008_l263_26330

/-- The sequence defined by the problem -/
def problem_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => if (n + 1) % (n / 2 + 2) = 0 then 1 else 2

/-- The sum of the first n terms of the sequence -/
def sequence_sum (n : ℕ) : ℕ :=
  (List.range n).map problem_sequence |>.sum

/-- The theorem stating that the sum of the first 1026 terms is 2008 -/
theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_1026_is_2008_l263_26330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_surface_area_difference_l263_26380

theorem inscribed_cylinder_surface_area_difference (r h : ℝ) :
  r > 0 → h > 0 → r^2 + h^2 = 16 →
  let sphere_surface_area := 4 * Real.pi * 16
  let cylinder_lateral_area := 2 * Real.pi * r * h
  (∀ r' h', r' > 0 → h' > 0 → r'^2 + h'^2 = 16 →
    cylinder_lateral_area ≥ 2 * Real.pi * r' * h') →
  sphere_surface_area - cylinder_lateral_area = 32 * Real.pi := by
  sorry

#check inscribed_cylinder_surface_area_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_surface_area_difference_l263_26380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l263_26384

/-- The number of teachers -/
def num_teachers : ℕ := 4

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of arrangements with students standing together -/
def arrangements_students_together : ℕ := (num_teachers + 1) * Nat.factorial num_students

/-- The number of arrangements with no two students next to each other -/
def arrangements_students_separated : ℕ := Nat.factorial num_teachers * Nat.factorial num_students * (num_teachers + 1)

/-- The number of arrangements with teachers and students alternating -/
def arrangements_alternating : ℕ := Nat.factorial num_teachers * Nat.factorial num_students

theorem photo_arrangements :
  arrangements_students_together = 2880 ∧
  arrangements_students_separated = 2880 ∧
  arrangements_alternating = 1152 := by
  sorry

#eval arrangements_students_together
#eval arrangements_students_separated
#eval arrangements_alternating

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l263_26384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l263_26315

/-- A hyperbola with equation (x-2)^2/7^2 - (y+5)^2/3^2 = 1 -/
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 7^2 - (y + 5)^2 / 3^2 = 1

/-- The x-coordinate of the focus with smaller x-coordinate -/
noncomputable def focus_x : ℝ := 2 - Real.sqrt 58

/-- The y-coordinate of the focus -/
def focus_y : ℝ := -5

/-- Theorem stating that (focus_x, focus_y) is the focus of the hyperbola with smaller x-coordinate -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola x y ∧ 
  (∀ (x' y' : ℝ), hyperbola x' y' → x ≤ x') ∧
  x = focus_x ∧ y = focus_y := by
  sorry

#check hyperbola_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l263_26315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_twice_l263_26323

/-- The reciprocal function -/
noncomputable def f (x : ℝ) : ℝ := 1 / x

/-- The initial number displayed on the calculator -/
def initial_number : ℝ := 50

/-- The theorem stating that applying the reciprocal function twice to 50 returns 50 -/
theorem reciprocal_twice (n : ℕ) :
  n = 2 ↔ f (f initial_number) = initial_number ∧ 
  ∀ m : ℕ, m < n → f^[m] initial_number ≠ initial_number :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_twice_l263_26323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetromino_coverage_sum_l263_26354

theorem tetromino_coverage_sum : 
  (Finset.filter (fun k => k ≤ 25 ∧ k % 4 = 0) (Finset.range 26)).sum id = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetromino_coverage_sum_l263_26354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_property_problem_solution_l263_26342

def greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a > r1) (hb : b > r2) : ℕ :=
  Finset.sup (Finset.filter (fun d => a % d = r1 ∧ b % d = r2) (Finset.range (min (a - r1) (b - r2) + 1))) id

theorem greatest_divisor_property (a b r1 r2 : ℕ) (ha : a > r1) (hb : b > r2) :
  let d := greatest_divisor_with_remainders a b r1 r2 ha hb
  d > 0 ∧ a % d = r1 ∧ b % d = r2 ∧
  ∀ k, k > 0 → a % k = r1 → b % k = r2 → k ≤ d :=
by sorry

theorem problem_solution :
  greatest_divisor_with_remainders 1657 2037 10 7 (by norm_num) (by norm_num) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_property_problem_solution_l263_26342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l263_26325

noncomputable def line_point (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3*s, 4 - s, 2 + 5*s)

def target_point : ℝ × ℝ × ℝ := (3, 2, 1)

noncomputable def closest_point : ℝ × ℝ × ℝ := (44/35, 137/35, 85/35)

def distance_squared (p q : ℝ × ℝ × ℝ) : ℝ :=
  let (px, py, pz) := p
  let (qx, qy, qz) := q
  (px - qx)^2 + (py - qy)^2 + (pz - qz)^2

theorem closest_point_on_line :
  ∀ s : ℝ, distance_squared (line_point s) target_point ≥ distance_squared closest_point target_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l263_26325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_a_amount_l263_26316

/-- Represents the mixture of liquids A and B -/
structure Mixture where
  a : ℝ  -- Amount of liquid A
  b : ℝ  -- Amount of liquid B

/-- Calculates the ratio of liquid A to liquid B in a mixture -/
noncomputable def ratio (m : Mixture) : ℝ := m.a / m.b

/-- Replaces a portion of the mixture with liquid B -/
noncomputable def replace (m : Mixture) (amount : ℝ) : Mixture :=
  let total := m.a + m.b
  let a_removed := (m.a / total) * amount
  let b_removed := (m.b / total) * amount
  { a := m.a - a_removed, b := m.b - b_removed + amount }

/-- The theorem to be proved -/
theorem initial_liquid_a_amount 
  (m : Mixture)  -- Initial mixture
  (h1 : ratio m = 4)  -- Initial ratio is 4:1
  (h2 : ratio (replace m 30) = 2/3)  -- New ratio after replacement is 2:3
  : m.a = 48  -- Initial amount of liquid A
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_a_amount_l263_26316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_octagon_l263_26307

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem interior_angle_regular_octagon : ℝ := by
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := 180 * (n - 2)
  let one_interior_angle : ℝ := sum_interior_angles / n
  have h : one_interior_angle = 135 := by
    -- Proof steps would go here
    sorry
  exact one_interior_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_octagon_l263_26307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l263_26352

/-- Circle C₁ -/
def circle_C1 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

/-- Parabola C₂ -/
def parabola_C2 (p x y : ℝ) : Prop := y^2 = 2 * p * x

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem parabola_equation (p : ℝ) 
  (h₁ : p > 0)
  (h₂ : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C1 x₁ y₁ ∧ circle_C1 x₂ y₂ ∧ 
    parabola_C2 p x₁ y₁ ∧ parabola_C2 p x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 8 * Real.sqrt 5 / 5) :
  p = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l263_26352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_gain_l263_26383

/-- Represents the fruit-seller's apple sale scenario -/
structure AppleSale where
  cost_price : ℚ  -- Cost price of one apple (using rational numbers)
  num_sold : ℕ    -- Number of apples sold
  gain_percent : ℚ -- Gain percentage

/-- Calculates the number of apples whose selling price equals the total gain -/
def gain_in_apples (sale : AppleSale) : ℚ :=
  let selling_price := sale.cost_price * (1 + sale.gain_percent / 100)
  let total_gain := sale.num_sold * (selling_price - sale.cost_price)
  total_gain / selling_price

/-- Theorem: The fruit-seller gains the selling price of 30 apples -/
theorem fruit_seller_gain (sale : AppleSale) 
  (h1 : sale.num_sold = 150)
  (h2 : sale.gain_percent = 25) :
  gain_in_apples sale = 30 := by
  sorry

/-- Example calculation -/
def example_sale : AppleSale := { 
  cost_price := 1, 
  num_sold := 150, 
  gain_percent := 25 
}

#eval gain_in_apples example_sale

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_seller_gain_l263_26383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_c_k_l263_26331

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_sequence d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_sequence (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_sequence r n * r

/-- Sum of arithmetic and geometric sequences -/
def c_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

/-- Theorem stating the existence and uniqueness of c_k -/
theorem exists_unique_c_k :
  ∃! c_k : ℕ, ∃ d r k : ℕ,
    c_sequence d r (k - 2) = 400 ∧
    c_sequence d r (k + 2) = 1600 ∧
    c_sequence d r k = 4 * c_sequence d r (k - 1) :=
by
  sorry

#check exists_unique_c_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_c_k_l263_26331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l263_26355

def jebb_main_dish : ℚ := 25
def jebb_appetizer : ℚ := 12
def jebb_dessert : ℚ := 7
def friend_main_dish : ℚ := 22
def friend_appetizer : ℚ := 10
def friend_dessert : ℚ := 6

def service_fee_rate (total_food_cost : ℚ) : ℚ :=
  if total_food_cost ≥ 70 then 15/100
  else if total_food_cost ≥ 50 then 12/100
  else if total_food_cost ≥ 30 then 10/100
  else 0

def tip_rate : ℚ := 18/100

theorem restaurant_bill_calculation :
  let jebb_total := jebb_main_dish + jebb_appetizer + jebb_dessert
  let friend_total := friend_main_dish + friend_appetizer + friend_dessert
  let total_food_cost := jebb_total + friend_total
  let service_fee := service_fee_rate total_food_cost * total_food_cost
  let bill_with_service := total_food_cost + service_fee
  let tip := tip_rate * bill_with_service
  let total_bill := bill_with_service + tip
  total_bill = 11127/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l263_26355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l263_26305

/-- Given a hyperbola with equation x²/4 - y²/m² = 1 where m > 0,
    if the eccentricity is √3, then m = 2√2 -/
theorem hyperbola_eccentricity (m : ℝ) (h₁ : m > 0) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / m^2 = 1) →
  (∃ e : ℝ, e = Real.sqrt 3 ∧ e^2 = 1 + m^2 / 4) →
  m = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l263_26305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniquely_3_colorable_edge_bound_l263_26346

/-- A graph is uniquely k-colorable if it has a k-coloring and there do not exist
    vertices that receive the same color in one k-coloring and different colors in another. -/
def UniquelyKColorable {α : Type*} [Fintype α] (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- The number of edges in a graph -/
def numEdges {α : Type*} [Fintype α] (G : SimpleGraph α) : ℕ := sorry

theorem uniquely_3_colorable_edge_bound {α : Type*} [Fintype α] (G : SimpleGraph α) (n : ℕ) :
  Fintype.card α = n →
  n ≥ 3 →
  UniquelyKColorable G 3 →
  numEdges G ≥ 2 * n - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniquely_3_colorable_edge_bound_l263_26346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l263_26341

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type representing a line passing through two points -/
structure Line where
  passes_through : (ℝ × ℝ) → (ℝ × ℝ) → Prop

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_intersection_length
  (para : Parabola)
  (l : Line)
  (P Q : ℝ × ℝ)
  (h1 : para.equation = fun x y => y^2 = 4*x)
  (h2 : para.focus = (1, 0))
  (h3 : l.passes_through para.focus P ∧ l.passes_through para.focus Q)
  (h4 : para.equation P.1 P.2 ∧ para.equation Q.1 Q.2)
  (h5 : distance P para.focus = 3)
  : distance Q para.focus = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l263_26341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l263_26309

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi/2) * Real.cos (x + Real.pi/4)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (5*Real.pi/8 + x) = f (5*Real.pi/8 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l263_26309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l263_26312

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | x < 0 ∨ x > 1} = {x : ℝ | ∃ y, f x = y} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l263_26312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_x_eq_8_l263_26339

/-- A point in 3D space represented by its coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given four points in 3D space, determines if they are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b : ℝ), 
    p4.x - p1.x = a * (p2.x - p1.x) + b * (p3.x - p1.x) ∧
    p4.y - p1.y = a * (p2.y - p1.y) + b * (p3.y - p1.y) ∧
    p4.z - p1.z = a * (p2.z - p1.z) + b * (p3.z - p1.z)

/-- The four points from the problem -/
def O : Point3D := ⟨0, 0, 0⟩
def A : Point3D := ⟨-2, 2, -2⟩
def B : Point3D := ⟨1, 4, -6⟩
def C (x : ℝ) : Point3D := ⟨x, -8, 8⟩

/-- The main theorem: the points are coplanar iff x = 8 -/
theorem coplanar_iff_x_eq_8 : 
  ∀ x, areCoplanar O A B (C x) ↔ x = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_iff_x_eq_8_l263_26339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_child_b_share_l263_26377

theorem child_b_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) (b_share : ℕ) : 
  total_money = 3600 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  b_share = total_money * ratio_b / (ratio_a + ratio_b + ratio_c) →
  b_share = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_child_b_share_l263_26377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_g_odd_g_upper_bound_lower_bound_l263_26345

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := (1/4)^x + (1/2)^x - 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - m * 2^x) / (1 + m * 2^x)

-- Statement 1: f(x) is not bounded on (-∞, 0)
theorem f_not_bounded : ¬ ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → |f x| ≤ M := by
  sorry

-- Statement 2: g(x) is an odd function when m = 1
theorem g_odd : ∀ (x : ℝ), g 1 (-x) = -(g 1 x) := by
  sorry

-- Statement 3: Lower bound of the upper bound of g(x) on [0, 1] when m ∈ (0, 1/2)
theorem g_upper_bound_lower_bound (m : ℝ) (h : 0 < m ∧ m < 1/2) :
  ∃ (G : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → g m x ≤ G) ∧ G ≥ (1 - m) / (1 + m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_bounded_g_odd_g_upper_bound_lower_bound_l263_26345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_sum_l263_26395

noncomputable def vec_op (a b : ℝ × ℝ) : ℝ := 
  (a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2)

theorem vector_operation_sum (a b : ℝ × ℝ) (θ : ℝ) : 
  θ ∈ Set.Ioo 0 (π/4) → 
  ∃ (k m : ℕ), k > 0 ∧ m > 0 ∧ 
    vec_op a b = k / 2 ∧ 
    vec_op b a = m / 2 → 
  vec_op a b + vec_op b a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operation_sum_l263_26395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_L_sets_l263_26390

/-- A color for an edge -/
inductive EdgeColor
| Red
| Blue
| White

/-- A complete graph with n vertices and colored edges -/
structure ColoredCompleteGraph (n : ℕ) where
  edgeColor : Fin n → Fin n → EdgeColor

/-- The set L(u,v) as defined in the problem -/
def L (G : ColoredCompleteGraph 2015) (V : Set (Fin 2015)) (u v : Fin 2015) : Set (Fin 2015) :=
  {u, v} ∪ {w | w ∈ V ∧ (G.edgeColor u w = EdgeColor.Red ∧ G.edgeColor v w = EdgeColor.Red) ∧
                       (G.edgeColor u v ≠ EdgeColor.Red)}

/-- The main theorem -/
theorem distinct_L_sets (G : ColoredCompleteGraph 2015) (V : Set (Fin 2015)) :
  ∃ S : Finset (Set (Fin 2015)), (∀ s ∈ S, ∃ u v, s = L G V u v) ∧ S.card ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_L_sets_l263_26390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimum_distance_l263_26358

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance from a point to a line ax + by + c = 0 -/
noncomputable def distanceToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

/-- Check if a circle satisfies the given conditions -/
def satisfiesConditions (c : Circle) : Prop :=
  -- Condition 1: The chord cut on the y-axis is 2 units long
  c.radius^2 = c.center.1^2 + 1 ∧
  -- Condition 2: The x-axis divides the circle into two arcs with a ratio of 3:1
  c.radius^2 = 2 * c.center.2^2

/-- The target line x - 2y = 0 -/
def targetLine : ℝ × ℝ × ℝ := (1, -2, 0)

/-- The theorem to prove -/
theorem circle_minimum_distance (c : Circle) :
  satisfiesConditions c →
  (∀ c' : Circle, satisfiesConditions c' →
    distanceToLine c.center targetLine.1 targetLine.2.1 targetLine.2.2 ≤
    distanceToLine c'.center targetLine.1 targetLine.2.1 targetLine.2.2) ↔
  ((c.center = (1, 1) ∧ c.radius = Real.sqrt 2) ∨
   (c.center = (-1, -1) ∧ c.radius = Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimum_distance_l263_26358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l263_26385

-- Define the function f(x) = sin(2x + π/3)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- State the theorem about the minimum positive period of f
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l263_26385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_yeast_operations_l263_26318

-- Define the set of all operations
def AllOperations : Set Nat := {1, 2, 3, 4, 5}

-- Define each operation as a proposition
def operation1 : Prop := True
def operation2 : Prop := False
def operation3 : Prop := False
def operation4 : Prop := True
def operation5 : Prop := True

-- Define the correctness of each operation
def isCorrect (n : Nat) : Prop :=
  match n with
  | 1 => operation1
  | 2 => operation2
  | 3 => operation3
  | 4 => operation4
  | 5 => operation5
  | _ => False

-- Theorem: The correct set of operations is {1, 4, 5}
theorem correct_yeast_operations :
  {n ∈ AllOperations | isCorrect n} = {1, 4, 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_yeast_operations_l263_26318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_assignment_l263_26340

structure Triangle where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ

def valid_triangle (t : Triangle) : Prop :=
  t.A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧ t.A ≠ t.E ∧ t.A ≠ t.F ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧ t.B ≠ t.E ∧ t.B ≠ t.F ∧
  t.C ≠ t.D ∧ t.C ≠ t.E ∧ t.C ≠ t.F ∧
  t.D ≠ t.E ∧ t.D ≠ t.F ∧
  t.E ≠ t.F ∧
  t.B + t.D + t.E = 14 ∧
  t.C + t.E + t.F = 12

theorem unique_triangle_assignment :
  ∃! t : Triangle, valid_triangle t ∧ 
    t.A = 1 ∧ t.B = 3 ∧ t.C = 2 ∧ t.D = 5 ∧ t.E = 6 ∧ t.F = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_assignment_l263_26340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l263_26337

-- Define the function f(x) = x + 2cos(x)
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- Define the domain of the function
def domain : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := Set.Ioo (Real.pi / 6) (5 * Real.pi / 6)

-- Theorem statement
theorem monotonic_decreasing_interval_of_f :
  ∀ x ∈ domain, (∀ y ∈ domain, x < y → f x > f y) ↔ x ∈ monotonic_decreasing_interval :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l263_26337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_existence_l263_26351

theorem library_books_existence : ∃ x : ℕ, 
  (3 * x + 2 * x + (3 * x / 2) + (3 * x / 5) + (4 * x / 5) > 10000) ∧ 
  (∃ n : ℕ, n = 3 * x + 2 * x + (3 * x / 2) + (3 * x / 5) + (4 * x / 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_books_existence_l263_26351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_values_l263_26367

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a₃ : ℝ
  q : ℝ
  sum_first_three : ℝ
  third_term_is_9 : a₃ = 9
  sum_is_27 : sum_first_three = 27
  first_three_terms : sum_first_three = a₃ / q^2 + a₃ / q + a₃

/-- The common ratio of the arithmetic progression is either 1 or -1/2 -/
theorem common_ratio_values (ap : ArithmeticProgression) : ap.q = 1 ∨ ap.q = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_values_l263_26367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_16_l263_26313

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (8 * t^2, 8 * t)

-- Define the line
def line (x : ℝ) : ℝ := x - 2

-- Define the point F
def point_F : ℝ × ℝ := (2, 0)

-- Define the inclination angle of the line
noncomputable def inclination_angle : ℝ := Real.pi / 4

-- Theorem statement
theorem length_of_AB_is_16 :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ : ℝ),
    A = curve_C t₁ ∧
    B = curve_C t₂ ∧
    A.2 = line A.1 ∧
    B.2 = line B.1 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_is_16_l263_26313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_with_parabolas_l263_26327

/-- The radius of a circle given specific parabola arrangements -/
theorem circle_radius_with_parabolas : ∃ (r : ℝ), r = (1/4 : ℝ) ∧
  (∃ (f : ℝ → ℝ),
    (∀ x, f x = x^2) ∧  -- parabola equation
    (∀ x y, f (x + Real.sqrt r) - r = y ↔ f x = y) ∧  -- symmetry about origin
    (∃ x, f x = r ∧ (deriv f) x = 0) ∧  -- vertex touches circle
    (∀ x, f x = x → ((deriv f) x = 1 → x = 1 - 2 * Real.sqrt r))) -- tangent along y = x
  := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_with_parabolas_l263_26327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l263_26356

/-- Given that m² + 12a² < 7am (a > 0) is a sufficient but not necessary condition
    for the equation x²/(m-1) + y²/(2-m) = 1 to represent an ellipse with foci on the y-axis,
    prove that 1/3 ≤ a ≤ 3/8 --/
theorem a_range (a m : ℝ) :
  (a > 0) →
  (m^2 + 12*a^2 < 7*a*m) →
  (∀ x y : ℝ, x^2/(m-1) + y^2/(2-m) = 1 → ∃ c : ℝ, c > 0 ∧ (x^2 + (y - c)^2 = (y + c)^2)) →
  (∀ m : ℝ, (1 < m ∧ m < 3/2) → (3*a < m ∧ m < 4*a)) →
  (1/3 ≤ a ∧ a ≤ 3/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l263_26356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_double_sum_zero_l263_26365

theorem cos_sin_sum_zero_implies_cos_double_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z + Real.cos (x - y) = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z + Real.sin (x - y) = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_zero_implies_cos_double_sum_zero_l263_26365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l263_26393

/-- The length of a train given its speed, the time to cross a man, and the man's speed. -/
noncomputable def train_length (train_speed : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : ℝ :=
  (train_speed + man_speed) * crossing_time * 1000 / 3600

/-- Theorem stating the length of the train under given conditions -/
theorem train_length_calculation :
  let train_speed := 69.994
  let crossing_time := 6
  let man_speed := 5
  ∃ ε > 0, |train_length train_speed crossing_time man_speed - 124.9866| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l263_26393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_exists_l263_26333

theorem sum_of_three_exists (S : Finset ℕ) (h_card : S.card = 68) (h_bound : ∀ n ∈ S, n < 100) :
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
             a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
             a = b + c + d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_exists_l263_26333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_mm_to_inches_l263_26344

-- Define the conversion factor
noncomputable def mm_per_inch : ℚ := 254/10

-- Define the function to convert millimeters to inches
noncomputable def mm_to_inches (mm : ℚ) : ℚ := mm / mm_per_inch

-- Define a function to round to the nearest hundredth
noncomputable def round_to_hundredth (x : ℚ) : ℚ := 
  (⌊x * 100 + 1/2⌋ : ℤ) / 100

-- Theorem statement
theorem two_thousand_mm_to_inches : 
  round_to_hundredth (mm_to_inches 2000) = 7874/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_thousand_mm_to_inches_l263_26344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_divisible_by_five_l263_26375

def original_number : Nat := 377353752

def digit_count (n : Nat) : Nat := 
  if n = 0 then 1 else Nat.log n 10 + 1

def is_divisible_by_five (n : Nat) : Prop :=
  n % 5 = 0

def last_digit (n : Nat) : Nat :=
  n % 10

def digits_list (n : Nat) : List Nat :=
  sorry

def count_occurrences (l : List Nat) (d : Nat) : Nat :=
  sorry

theorem permutations_divisible_by_five :
  let n := original_number
  let digits := digits_list n
  let last_dig := last_digit n
  (digit_count n = 9) →
  (last_dig = 5 ∨ last_dig = 0) →
  (count_occurrences digits 3 = 3) →
  (count_occurrences digits 7 = 3) →
  (count_occurrences digits 5 = 2) →
  (count_occurrences digits 2 = 1) →
  (∀ (m : Nat), m ∈ digits → m ∈ [2, 3, 5, 7]) →
  (Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 1 * Nat.factorial 1) = 1120) :=
by sorry

#eval digit_count original_number
#eval last_digit original_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutations_divisible_by_five_l263_26375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_0375_l263_26317

theorem simplest_fraction_of_0375 (a b : ℕ+) :
  (a : ℚ) / b = 375 / 1000 ∧ (a.val.gcd b.val = 1) → a.val + b.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_of_0375_l263_26317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_box_weight_is_155_l263_26326

noncomputable section

/-- The weight of the first box in kilograms -/
def first_box_weight : ℝ := 155

/-- The weight of the second box in kilograms -/
def second_box_weight : ℝ := 1.5 * first_box_weight

/-- The weight of the third box in kilograms -/
def third_box_weight : ℝ := 1.25 * first_box_weight

/-- The weight of the fourth box in kilograms -/
def fourth_box_weight : ℝ := 350

/-- The weight of the fifth box in kilograms -/
def fifth_box_weight : ℝ := fourth_box_weight / 0.7

theorem first_box_weight_is_155 :
  let heaviest_four_avg := (second_box_weight + third_box_weight + fourth_box_weight + fifth_box_weight) / 4
  let lightest_four_avg := (first_box_weight + second_box_weight + third_box_weight + fourth_box_weight) / 4
  heaviest_four_avg - lightest_four_avg = 75 ∧
  first_box_weight = 155 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_box_weight_is_155_l263_26326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alloy_weight_l263_26347

/-- The weight of the initial alloy in ounces -/
def initial_weight : ℝ := sorry

/-- The percentage of gold in the initial alloy -/
def initial_gold_percentage : ℝ := 0.5

/-- The amount of pure gold added in ounces -/
def added_gold : ℝ := 24

/-- The percentage of gold in the final alloy -/
def final_gold_percentage : ℝ := 0.8

theorem initial_alloy_weight :
  (initial_gold_percentage * initial_weight + added_gold) / (initial_weight + added_gold) = final_gold_percentage →
  initial_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alloy_weight_l263_26347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l263_26332

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_expression :
  (∀ x : ℝ, f (x + 1) = 3 * x + 2) →
  (∀ x : ℝ, f x = 3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l263_26332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_reduction_proof_l263_26360

/-- The number of collisions required to reduce the speed by a factor of 8 -/
def num_collisions : ℕ := 6

/-- The factor by which the speed is reduced -/
noncomputable def speed_reduction_factor : ℝ := 8

/-- The factor by which kinetic energy is reduced after each collision -/
noncomputable def energy_loss_factor : ℝ := 1/2

theorem speed_reduction_proof :
  (energy_loss_factor ^ num_collisions) * speed_reduction_factor^2 = 1 := by
  sorry

#check speed_reduction_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_reduction_proof_l263_26360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_unit_angle_l263_26394

/-- Given a circle where the chord length corresponding to a central angle of 1 radian is 2,
    prove that the area of the sector enclosed by this central angle is 1 / (2 * (sin (1/2))^2). -/
theorem sector_area_for_unit_angle (r : ℝ) : 
  r * Real.sin (1/2 : ℝ) = 1 → 
  (1/2 : ℝ) * 1 * r^2 = 1 / (2 * (Real.sin (1/2 : ℝ))^2) := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_unit_angle_l263_26394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l263_26361

/-- Given two lines in a 2D plane, where:
    - Line 1 passes through points (3, a) and (-2, 0)
    - Line 2 passes through point (3, -4) and has a slope of 1/2
    - The lines are perpendicular
    This theorem proves that the value of 'a' must be -10. -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  (((a - 0) / (3 - (-2))) * (1/2) = -1) → a = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l263_26361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l263_26311

noncomputable section

/-- Given a circle with radius and central angle, calculate the arc length of a sector --/
def arcLength (r : ℝ) (α : ℝ) : ℝ := r * α

/-- Calculate the area of a sector given its radius and arc length --/
def sectorArea (r : ℝ) (l : ℝ) : ℝ := (1/2) * r * l

/-- Calculate the perimeter of a sector given its radius and arc length --/
def sectorPerimeter (r : ℝ) (l : ℝ) : ℝ := l + 2 * r

theorem sector_properties :
  let r₁ : ℝ := 6
  let α₁ : ℝ := 150 * (π / 180)
  let l₁ : ℝ := arcLength r₁ α₁
  let perimeter : ℝ := 24
  let max_angle : ℝ := 2
  let max_area : ℝ := 36
  (l₁ = 5 * π) ∧
  (∀ r l, sectorPerimeter r l = perimeter →
    sectorArea r l ≤ max_area) ∧
  (∃ r l, sectorPerimeter r l = perimeter ∧
    arcLength r max_angle = l ∧
    sectorArea r l = max_area) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_properties_l263_26311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_moves_l263_26302

/-- Represents a wedge resting on a horizontal frictionless surface -/
structure Wedge where
  mass : ℝ
  angle : ℝ

/-- Represents a block that can slide on the wedge -/
structure Block where
  mass : ℝ

/-- Represents the system of a block and wedge -/
structure BlockWedgeSystem where
  block : Block
  wedge : Wedge

/-- Represents the position of the center of mass -/
structure CenterOfMass where
  x : ℝ
  y : ℝ

/-- Function to calculate the center of mass of the system at time t -/
noncomputable def centerOfMassAtTime (system : BlockWedgeSystem) (t : ℝ) : CenterOfMass :=
  sorry

/-- Theorem stating that the center of mass moves both horizontally and vertically -/
theorem center_of_mass_moves (system : BlockWedgeSystem) :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (centerOfMassAtTime system t₁).x ≠ (centerOfMassAtTime system t₂).x ∧
  (centerOfMassAtTime system t₁).y ≠ (centerOfMassAtTime system t₂).y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_moves_l263_26302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_concentrate_volume_approx_l263_26335

/-- Calculates the volume of orange concentrate in a cylindrical jug -/
noncomputable def orange_concentrate_volume (height : ℝ) (diameter : ℝ) (fill_ratio : ℝ) (concentrate_ratio : ℝ) : ℝ :=
  let radius := diameter / 2
  let juice_height := height * fill_ratio
  let juice_volume := Real.pi * radius^2 * juice_height
  juice_volume * concentrate_ratio

/-- Theorem stating the volume of orange concentrate in the jug -/
theorem orange_concentrate_volume_approx :
  let jug_height : ℝ := 8
  let jug_diameter : ℝ := 3
  let fill_ratio : ℝ := 3/4
  let concentrate_ratio : ℝ := 1/6
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ 
    |orange_concentrate_volume jug_height jug_diameter fill_ratio concentrate_ratio - 2.25| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_concentrate_volume_approx_l263_26335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l263_26392

noncomputable section

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The eccentricity of the ellipse -/
def eccentricity : ℝ := 1 / 2

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- A line through the right focus -/
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

/-- Projection onto the line x = 4 -/
def project_to_line (p : ℝ × ℝ) : ℝ × ℝ := (4, p.2)

/-- The fixed intersection point -/
def P : ℝ × ℝ := (5/2, 0)

theorem ellipse_intersection_fixed_point :
  ∀ m : ℝ,
  ∀ A B : ℝ × ℝ,
  C A.1 A.2 ∧ C B.1 B.2 ∧
  line_through_focus m A.1 A.2 ∧ line_through_focus m B.1 B.2 →
  ∃ t : ℝ,
    (1 - t) • A + t • (project_to_line B) = P ∧
    (1 - t) • B + t • (project_to_line A) = P :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l263_26392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_future_value_l263_26321

/-- Calculates the future value of a loan with daily compound interest -/
noncomputable def future_value (principal : ℝ) (days : ℕ) (annual_rate : ℝ) (days_in_year : ℕ) : ℝ :=
  principal * (1 + (days : ℝ) * annual_rate / (days_in_year : ℝ))

/-- Theorem: The future value of a 200,000 ruble loan for 73 days at 25% annual interest is 210,000 rubles -/
theorem loan_future_value : 
  let principal : ℝ := 200000
  let days : ℕ := 73
  let annual_rate : ℝ := 0.25
  let days_in_year : ℕ := 365
  future_value principal days annual_rate days_in_year = 210000 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval future_value 200000 73 0.25 365

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_future_value_l263_26321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_PQ_passes_through_fixed_point_l263_26308

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def l (x : ℝ) : Prop := x = 4

-- Define the points A₁ and A₂
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)

-- Define a point M on line l
noncomputable def M (t : ℝ) : ℝ × ℝ := (4, t)

-- Define the intersection points P and Q
noncomputable def P (t : ℝ) : ℝ × ℝ := ((72 - 2*t^2) / (36 + t^2), (24*t) / (36 + t^2))
noncomputable def Q (t : ℝ) : ℝ × ℝ := ((2*t^2 - 8) / (4 + t^2), (-8*t) / (4 + t^2))

-- Define the fixed point
def fixedPoint : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem line_PQ_passes_through_fixed_point :
  ∀ t : ℝ, ∃ k : ℝ, k * (P t).1 + (1 - k) * (Q t).1 = (fixedPoint).1 ∧
                    k * (P t).2 + (1 - k) * (Q t).2 = (fixedPoint).2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_PQ_passes_through_fixed_point_l263_26308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_geq_one_set_m_range_l263_26364

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem axis_of_symmetry (k : ℤ) :
  ∀ x, f x = f (k * π / 2 + 3 * π / 8 - x) := by sorry

theorem f_geq_one_set :
  {x : ℝ | f x ≥ 1} = {x : ℝ | ∃ k : ℤ, k * π + π / 4 ≤ x ∧ x ≤ k * π + π / 2} := by sorry

theorem m_range :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (π / 6) (π / 3), f x - m < 2) ↔ m > (Real.sqrt 3 - 5) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_f_geq_one_set_m_range_l263_26364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l263_26306

def full_rotation : ℝ := 360

theorem rotation_equivalence (y : ℝ) (h : y < full_rotation) :
  (450 % full_rotation = full_rotation - y) →
  y = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_equivalence_l263_26306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l263_26387

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-1 - 1/2 * t, 2 + Real.sqrt 3/2 * t)

-- Define the circle
noncomputable def circle_eq (θ : ℝ) : ℝ := 2 * Real.cos (θ + Real.pi/3)

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 2 * Real.pi / 3

-- Theorem statement
theorem intersection_product : 
  ∃ (M N : ℝ × ℝ), 
    (∃ (t_M t_N : ℝ), line_l t_M = M ∧ line_l t_N = N) ∧ 
    (∃ (θ_M θ_N : ℝ), 
      Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) = circle_eq θ_M ∧
      Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = circle_eq θ_N) ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) *
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 6 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l263_26387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_points_imply_a_range_l263_26301

noncomputable section

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x + a * (x - Real.log x)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (Real.exp x + a * x) * (x - 1) / (x^2)

theorem extremal_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 2 ∧
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧ f_deriv a x₃ = 0 ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  -2 * Real.sqrt (Real.exp 1) < a ∧ a < -Real.exp 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremal_points_imply_a_range_l263_26301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_five_days_avg_l263_26381

/-- Represents the average daily TV production for different periods in a month. -/
structure TVProduction where
  totalDays : ℕ
  firstPeriodDays : ℕ
  firstPeriodAvg : ℚ
  monthlyAvg : ℚ

/-- Calculates the average daily production for the remaining days of the month. -/
def remainingDaysAvg (p : TVProduction) : ℚ :=
  let totalProduction := p.monthlyAvg * p.totalDays
  let firstPeriodProduction := p.firstPeriodAvg * p.firstPeriodDays
  let remainingDays := p.totalDays - p.firstPeriodDays
  (totalProduction - firstPeriodProduction) / remainingDays

/-- Theorem stating that under given conditions, the average production for the last 5 days is 58 TVs per day. -/
theorem last_five_days_avg (p : TVProduction) 
  (h1 : p.totalDays = 30)
  (h2 : p.firstPeriodDays = 25)
  (h3 : p.firstPeriodAvg = 70)
  (h4 : p.monthlyAvg = 68) :
  remainingDaysAvg p = 58 := by
  sorry

#eval remainingDaysAvg { totalDays := 30, firstPeriodDays := 25, firstPeriodAvg := 70, monthlyAvg := 68 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_five_days_avg_l263_26381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_200_l263_26353

/-- Represents the simple interest calculation for a given principal, rate, and time period. -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Theorem stating that if increasing the interest rate by 5% results in Rs. 100 more interest
    over 10 years, then the principal amount must be Rs. 200. -/
theorem principal_is_200 (P : ℝ) (R : ℝ) :
  simpleInterest P (R + 5) 10 = simpleInterest P R 10 + 100 →
  P = 200 := by
  intro h
  -- The proof goes here
  sorry

#check principal_is_200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_200_l263_26353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painted_area_l263_26349

-- Define the dimensions of the painted areas
noncomputable def monday_length : ℝ := 8
noncomputable def monday_width : ℝ := 6
noncomputable def tuesday_length1 : ℝ := 12
noncomputable def tuesday_width1 : ℝ := 4
noncomputable def tuesday_length2 : ℝ := 6
noncomputable def tuesday_width2 : ℝ := 6
noncomputable def wednesday_base : ℝ := 10
noncomputable def wednesday_height : ℝ := 4

-- Define the function to calculate the area of a rectangle
noncomputable def rectangle_area (length width : ℝ) : ℝ := length * width

-- Define the function to calculate the area of a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

-- Theorem statement
theorem total_painted_area :
  rectangle_area monday_length monday_width +
  rectangle_area tuesday_length1 tuesday_width1 +
  rectangle_area tuesday_length2 tuesday_width2 +
  triangle_area wednesday_base wednesday_height = 152 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_painted_area_l263_26349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l263_26303

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Represents the folded paper -/
structure FoldedPaper where
  original : Paper
  foldedArea : ℝ

/-- Creates a rectangular paper with length twice its width -/
def createPaper (w : ℝ) : Paper :=
  { width := w
  , length := 2 * w
  , area := 2 * w^2 }

/-- Folds the paper as described in the problem -/
noncomputable def foldPaper (p : Paper) : FoldedPaper :=
  { original := p
  , foldedArea := p.area - p.width * Real.sqrt 2 }

/-- Theorem stating the ratio of folded area to original area -/
theorem folded_area_ratio (w : ℝ) (h : w > 0) :
  let p := createPaper w
  let fp := foldPaper p
  fp.foldedArea / p.area = 1 - Real.sqrt 2 / 2 := by
  sorry

#check folded_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l263_26303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_even_g_l263_26371

/-- The function f(x) = sin x + cos x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

/-- The function g(x) = sin(x-t) + cos(x-t), which is f(x) translated right by t units -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f (x - t)

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The theorem stating that the minimum positive t for which g is even is 3π/4 -/
theorem min_t_for_even_g :
  (∃ t₀ : ℝ, t₀ > 0 ∧ is_even (g t₀) ∧ ∀ t, t > 0 → is_even (g t) → t ≥ t₀) ∧
  (∃ t : ℝ, t > 0 ∧ is_even (g t)) →
  (Classical.epsilon (λ t ↦ t > 0 ∧ is_even (g t))) = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_t_for_even_g_l263_26371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l263_26324

theorem cos_difference_value (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1/2) 
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) : 
  Real.cos (α - β) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l263_26324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l263_26376

def total_candies : ℕ := 40
def red_candies : ℕ := 15
def blue_candies : ℕ := 15
def green_candies : ℕ := 10

def pick_size : ℕ := 3

theorem same_color_combination_probability :
  let probability_numerator := 118545
  let probability_denominator := 2192991
  (probability_numerator : ℚ) / probability_denominator =
    (Nat.choose red_candies pick_size * Nat.choose (red_candies - pick_size) pick_size +
     Nat.choose blue_candies pick_size * Nat.choose (blue_candies - pick_size) pick_size +
     Nat.choose green_candies pick_size * Nat.choose (green_candies - pick_size) pick_size +
     Nat.choose red_candies 1 * Nat.choose blue_candies 1 * Nat.choose green_candies 1 *
     Nat.choose (red_candies - 1) 1 * Nat.choose (blue_candies - 1) 1 * Nat.choose (green_candies - 1) 1) /
    (Nat.choose total_candies pick_size * Nat.choose (total_candies - pick_size) pick_size) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l263_26376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equals_interval_range_of_m_l263_26396

-- Define set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (3/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

-- Theorem 1: Set A is equal to [7/16, 2]
theorem set_A_equals_interval : A = Set.Icc (7/16 : ℝ) 2 := by sorry

-- Theorem 2: Range of m given A ⊆ B
theorem range_of_m (h : ∀ m : ℝ, A ⊆ B m) : 
  ∀ m : ℝ, m ≥ 3/4 ∨ m ≤ -3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_equals_interval_range_of_m_l263_26396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l263_26322

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane
  (α β : Plane) (a b : Line)
  (h1 : perpendicular α β)
  (h2 : intersect α β a)
  (h3 : contains β b)
  (h4 : perpendicularLines a b) :
  perpendicularLineToPlane b α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_to_plane_l263_26322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_x_squared_count_l263_26304

theorem ceiling_x_squared_count (x : ℝ) (h : ⌊x⌋ = -11) :
  ∃ (S : Finset ℤ), (∀ n ∈ S, ∃ y : ℝ, ⌊y⌋ = -11 ∧ ⌈y^2⌉ = n) ∧ S.card = 22 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_x_squared_count_l263_26304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karlson_candy_theorem_l263_26350

theorem karlson_candy_theorem :
  let initial_board : List Nat := List.replicate 26 1
  let num_operations : Nat := 25
  let candy_eaten (x y : Nat) : Nat := x * y
  ∀ (operation_sequence : List (Nat × Nat)),
    operation_sequence.length = num_operations →
    (∀ (pair : Nat × Nat), pair ∈ operation_sequence → 
      pair.1 ∈ initial_board ∨ pair.1 ∈ (operation_sequence.map (λ (x, y) => x + y))) →
    (∀ (pair : Nat × Nat), pair ∈ operation_sequence → 
      pair.2 ∈ initial_board ∨ pair.2 ∈ (operation_sequence.map (λ (x, y) => x + y))) →
    (operation_sequence.map (λ (x, y) => candy_eaten x y)).sum = 325 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karlson_candy_theorem_l263_26350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l263_26369

-- Define the general term of the series
def a (n : ℕ) : ℚ :=
  (3 * n^3 + 2 * n^2 - n + 1) / (n^6 + n^5 - n^4 + n^3 - n^2 + n)

-- Define the series
noncomputable def S : ℚ := ∑' n, if n ≥ 2 then a n else 0

-- State the theorem
theorem series_sum : S = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l263_26369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implies_m_value_complex_in_second_quadrant_implies_x_range_l263_26388

-- Part 1
theorem complex_real_implies_m_value (m : ℝ) : 
  (m^2 - 1 : ℂ) + (m + 1 : ℂ) * Complex.I = (m^2 - 1 : ℂ) → m = -1 := by sorry

-- Part 2
theorem complex_in_second_quadrant_implies_x_range (x : ℝ) :
  let z : ℂ := (Real.sqrt x - 1 : ℝ) + (x^2 - 3*x + 2 : ℝ) * Complex.I
  (Real.sqrt x - 1 < 0 ∧ x^2 - 3*x + 2 > 0) → x > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implies_m_value_complex_in_second_quadrant_implies_x_range_l263_26388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_10_sided_polygon_l263_26372

/-- The measure of an interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

/-- The measure of an exterior angle of a regular polygon with n sides -/
noncomputable def exterior_angle (n : ℕ) : ℝ := 180 - interior_angle n

theorem exterior_angle_10_sided_polygon :
  3 ≤ 10 ∧ 10 ≤ 10 →
  exterior_angle 10 = 36 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_10_sided_polygon_l263_26372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_theorem_l263_26329

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 1 else -2

theorem max_difference_theorem (x₁ x₂ : ℝ) 
  (h₁ : x₁ + (x₁ - 1) * f (x₁ + 1) ≤ 5) 
  (h₂ : x₂ + (x₂ - 1) * f (x₂ + 1) ≤ 5) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ y₁ y₂ : ℝ, 
    y₁ + (y₁ - 1) * f (y₁ + 1) ≤ 5 → 
    y₂ + (y₂ - 1) * f (y₂ + 1) ≤ 5 → 
    y₁ - y₂ ≤ m :=
by
  -- We'll use 6 as our maximum difference
  use 6
  constructor
  · -- Prove that m = 6
    rfl
  · -- Prove the universal quantification
    intros y₁ y₂ hy₁ hy₂
    -- The proof goes here, but we'll use sorry for now
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_theorem_l263_26329
