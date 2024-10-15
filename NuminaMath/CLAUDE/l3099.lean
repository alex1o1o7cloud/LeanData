import Mathlib

namespace NUMINAMATH_CALUDE_alice_spending_percentage_l3099_309983

theorem alice_spending_percentage (alice_initial : ℝ) (bob_initial : ℝ) (alice_final : ℝ)
  (h1 : bob_initial = 0.9 * alice_initial)
  (h2 : alice_final = 0.9 * bob_initial) :
  (alice_initial - alice_final) / alice_initial = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_alice_spending_percentage_l3099_309983


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l3099_309906

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 4)
  (h2 : parrots_per_cage = 8)
  (h3 : total_birds = 40)
  : (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l3099_309906


namespace NUMINAMATH_CALUDE_compute_expression_l3099_309911

theorem compute_expression : 3 * 3^4 - 27^60 / 27^58 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3099_309911


namespace NUMINAMATH_CALUDE_inequality_always_true_l3099_309930

theorem inequality_always_true (x : ℝ) : (7 / 20) + |3 * x - (2 / 5)| ≥ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3099_309930


namespace NUMINAMATH_CALUDE_smallest_three_digit_middle_ring_l3099_309927

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- Checks if a number can be expressed as a product of numbers from 1 to 26 -/
def is_expressible (n : ℕ) : Prop := ∃ (factors : List ℕ), (factors.all (λ x => 1 ≤ x ∧ x ≤ 26)) ∧ (factors.prod = n)

/-- The smallest three-digit middle ring number -/
def smallest_middle_ring : ℕ := 106

theorem smallest_three_digit_middle_ring :
  is_composite smallest_middle_ring ∧
  ¬(is_expressible smallest_middle_ring) ∧
  ∀ n < smallest_middle_ring, n ≥ 100 → is_composite n → is_expressible n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_middle_ring_l3099_309927


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3099_309977

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x - 5 > 0 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3099_309977


namespace NUMINAMATH_CALUDE_four_twos_polynomial_property_l3099_309937

/-- A polynomial that takes the value 2 for four different integer inputs -/
def FourTwosPolynomial (P : ℤ → ℤ) : Prop :=
  ∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  P a = 2 ∧ P b = 2 ∧ P c = 2 ∧ P d = 2

theorem four_twos_polynomial_property (P : ℤ → ℤ) 
  (h : FourTwosPolynomial P) :
  ∀ x : ℤ, P x ≠ 1 ∧ P x ≠ 3 ∧ P x ≠ 5 ∧ P x ≠ 7 ∧ P x ≠ 9 :=
sorry

end NUMINAMATH_CALUDE_four_twos_polynomial_property_l3099_309937


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3099_309943

/-- Given a triangle DEF with side lengths DE = 26, DF = 15, and EF = 17,
    the radius of its inscribed circle is 3√2. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3099_309943


namespace NUMINAMATH_CALUDE_inequality_proof_l3099_309986

theorem inequality_proof (a b : ℝ) (h : a ≠ b) : a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3099_309986


namespace NUMINAMATH_CALUDE_odd_power_difference_divisibility_l3099_309948

theorem odd_power_difference_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) :
  ∃ k : ℕ, (2^2018 ∣ b^k - a^2) ∨ (2^2018 ∣ a^k - b^2) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_difference_divisibility_l3099_309948


namespace NUMINAMATH_CALUDE_divisibility_problem_l3099_309940

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) 
  (hy : y ≠ -1) 
  (h_int : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3099_309940


namespace NUMINAMATH_CALUDE_triangle_equation_implies_right_triangle_l3099_309926

/-- A triangle with side lengths satisfying a certain equation is a right triangle -/
theorem triangle_equation_implies_right_triangle 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^3 + (Real.sqrt 2 / 4) * b^3 + (Real.sqrt 3 / 9) * c^3 - (Real.sqrt 6 / 2) * a * b * c = 0) :
  a^2 + b^2 = c^2 := by
  sorry

#check triangle_equation_implies_right_triangle

end NUMINAMATH_CALUDE_triangle_equation_implies_right_triangle_l3099_309926


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3099_309999

/-- A polynomial that takes integer values at integer points -/
def IntegerPolynomial := ℤ → ℤ

/-- Proposition: If a polynomial with integer coefficients takes the value 2 
    at three distinct integer points, it cannot take the value 3 at any integer point -/
theorem polynomial_value_theorem (P : IntegerPolynomial) 
  (h1 : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ¬∃ x : ℤ, P x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3099_309999


namespace NUMINAMATH_CALUDE_perimeter_increase_theorem_l3099_309998

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- Result of moving sides of a polygon outward -/
structure TransformedPolygon where
  original : ConvexPolygon
  distance : Real

/-- Perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real := sorry

/-- Perimeter increase after transformation -/
def perimeter_increase (tp : TransformedPolygon) : Real :=
  perimeter (ConvexPolygon.mk tp.original.vertices true) - perimeter tp.original

/-- Theorem: Perimeter increase is greater than 30 cm when sides are moved by 5 cm -/
theorem perimeter_increase_theorem (p : ConvexPolygon) :
  perimeter_increase (TransformedPolygon.mk p 5) > 30 := by sorry

end NUMINAMATH_CALUDE_perimeter_increase_theorem_l3099_309998


namespace NUMINAMATH_CALUDE_minimum_dimes_needed_l3099_309957

/-- The cost of the jacket in cents -/
def jacket_cost : ℕ := 4550

/-- The value of two $20 bills in cents -/
def bills_value : ℕ := 2 * 2000

/-- The value of five quarters in cents -/
def quarters_value : ℕ := 5 * 25

/-- The value of six nickels in cents -/
def nickels_value : ℕ := 6 * 5

/-- The value of one dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed -/
def min_dimes : ℕ := 40

theorem minimum_dimes_needed :
  ∀ n : ℕ, 
    n ≥ min_dimes → 
    bills_value + quarters_value + nickels_value + n * dime_value ≥ jacket_cost ∧
    ∀ m : ℕ, m < min_dimes → 
      bills_value + quarters_value + nickels_value + m * dime_value < jacket_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_dimes_needed_l3099_309957


namespace NUMINAMATH_CALUDE_equation_equivalence_l3099_309923

theorem equation_equivalence : ∀ x : ℝ, (2 * (x + 1) = x + 7) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3099_309923


namespace NUMINAMATH_CALUDE_root_equations_imply_m_n_values_l3099_309992

theorem root_equations_imply_m_n_values (m n : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    ((r1 + m) * (r1 + n) * (r1 + 8)) / ((r1 + 2)^2) = 0 ∧
    ((r2 + m) * (r2 + n) * (r2 + 8)) / ((r2 + 2)^2) = 0 ∧
    ((r3 + m) * (r3 + n) * (r3 + 8)) / ((r3 + 2)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*m) * (r + 4) * (r + 10)) / ((r + n) * (r + 8)) = 0) →
  m = 1 ∧ n = 4 ∧ 50*m + n = 54 := by
sorry

end NUMINAMATH_CALUDE_root_equations_imply_m_n_values_l3099_309992


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l3099_309993

/-- Represents the game state with n objects and maximum removal of m objects per turn -/
structure GameState where
  n : ℕ  -- Number of objects in the pile
  m : ℕ  -- Maximum number of objects that can be removed per turn

/-- Predicate to check if a player has a winning strategy -/
def has_winning_strategy (state : GameState) : Prop :=
  ¬(state.n + 1 ∣ state.m)

/-- Theorem stating the condition for Alice to have a winning strategy -/
theorem alice_winning_strategy (state : GameState) :
  has_winning_strategy state ↔ ¬(state.n + 1 ∣ state.m) :=
sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l3099_309993


namespace NUMINAMATH_CALUDE_four_numbers_theorem_l3099_309991

def satisfies_condition (x y z t : ℝ) : Prop :=
  x + y * z * t = 2 ∧
  y + x * z * t = 2 ∧
  z + x * y * t = 2 ∧
  t + x * y * z = 2

theorem four_numbers_theorem :
  ∀ x y z t : ℝ,
    satisfies_condition x y z t ↔
      ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
       (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = 3) ∨
       (x = -1 ∧ y = -1 ∧ z = 3 ∧ t = -1) ∨
       (x = -1 ∧ y = 3 ∧ z = -1 ∧ t = -1) ∨
       (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_theorem_l3099_309991


namespace NUMINAMATH_CALUDE_infinite_pairs_and_odd_sum_l3099_309976

theorem infinite_pairs_and_odd_sum :
  (∃ (S : Set (ℕ × ℕ)), Set.Infinite S ∧
    ∀ (p : ℕ × ℕ), p ∈ S →
      (⌊(4 + 2 * Real.sqrt 3) * p.1⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * p.2⌋) ∧
  (∀ (m n : ℕ), 
    (⌊(4 + 2 * Real.sqrt 3) * m⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * n⌋ →
    Odd (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_pairs_and_odd_sum_l3099_309976


namespace NUMINAMATH_CALUDE_root_equality_implies_b_equals_three_l3099_309996

theorem root_equality_implies_b_equals_three
  (a b c N : ℝ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (hN : N > 1)
  (h_int_a : ∃ k : ℤ, a = k)
  (h_int_b : ∃ k : ℤ, b = k)
  (h_int_c : ∃ k : ℤ, c = k)
  (h_eq : (N * (N^(1/b))^(1/c))^(1/a) = N^(25/36)) :
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_root_equality_implies_b_equals_three_l3099_309996


namespace NUMINAMATH_CALUDE_midpoint_square_area_l3099_309902

theorem midpoint_square_area (A B C D : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (1, 0) → 
  C = (1, 1) → 
  D = (0, 1) → 
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let Q := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_square_area_l3099_309902


namespace NUMINAMATH_CALUDE_increase_in_circumference_l3099_309978

/-- The increase in circumference when the diameter of a circle increases by 2π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + 2 * π)
  let increase_in_circumference := new_circumference - original_circumference
  increase_in_circumference = 2 * π^2 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_circumference_l3099_309978


namespace NUMINAMATH_CALUDE_polygon_angles_l3099_309920

theorem polygon_angles (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 + (360 / n) = 1500 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l3099_309920


namespace NUMINAMATH_CALUDE_trapezoid_xy_length_l3099_309931

-- Define the trapezoid and its properties
structure Trapezoid where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  wy_perp_zy : (W.1 - Y.1) * (Z.1 - Y.1) + (W.2 - Y.2) * (Z.2 - Y.2) = 0

-- Define the given conditions
def trapezoid_conditions (t : Trapezoid) : Prop :=
  let (_, y2) := t.Y
  let (_, z2) := t.Z
  let yz_length := Real.sqrt ((t.Y.1 - t.Z.1)^2 + (y2 - z2)^2)
  let tan_z := (t.W.2 - t.Z.2) / (t.W.1 - t.Z.1)
  let tan_x := (t.W.2 - t.X.2) / (t.X.1 - t.W.1)
  yz_length = 15 ∧ tan_z = 2 ∧ tan_x = 2.5

-- State the theorem
theorem trapezoid_xy_length (t : Trapezoid) (h : trapezoid_conditions t) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 6 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_xy_length_l3099_309931


namespace NUMINAMATH_CALUDE_quadratic_points_comparison_l3099_309980

theorem quadratic_points_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (-1)^2 - 6*(-1) + c) 
  (h2 : y₂ = 2^2 - 6*2 + c) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_comparison_l3099_309980


namespace NUMINAMATH_CALUDE_correct_answers_for_given_exam_l3099_309909

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  totalScore : ℤ

/-- Calculates the number of correctly answered questions. -/
def correctAnswers (result : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that given the specific exam conditions, 
    the number of correctly answered questions is 44. -/
theorem correct_answers_for_given_exam : 
  let exam : Exam := { totalQuestions := 60, correctScore := 4, wrongScore := -1 }
  let result : ExamResult := { exam := exam, totalScore := 160 }
  correctAnswers result = 44 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_exam_l3099_309909


namespace NUMINAMATH_CALUDE_perfect_square_factorization_l3099_309951

/-- Perfect square formula check -/
def isPerfectSquare (a b c : ℝ) : Prop :=
  ∃ (k : ℝ), a * c = (b / 2) ^ 2 ∧ a > 0

theorem perfect_square_factorization :
  ¬ isPerfectSquare 1 (1/4 : ℝ) (1/4) ∧
  ¬ isPerfectSquare 1 (2 : ℝ) (-1) ∧
  ¬ isPerfectSquare 1 (1 : ℝ) 1 ∧
  isPerfectSquare 4 (4 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factorization_l3099_309951


namespace NUMINAMATH_CALUDE_propositions_equivalent_l3099_309975

-- Define P as a set
variable (P : Set α)

-- Define the original proposition
def original_prop (a b : α) : Prop :=
  a ∈ P → b ∉ P

-- Define the equivalent proposition (option D)
def equivalent_prop (a b : α) : Prop :=
  b ∈ P → a ∉ P

-- Theorem stating the equivalence of the two propositions
theorem propositions_equivalent (a b : α) :
  original_prop P a b ↔ equivalent_prop P a b :=
sorry

end NUMINAMATH_CALUDE_propositions_equivalent_l3099_309975


namespace NUMINAMATH_CALUDE_three_large_five_small_capacity_l3099_309955

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℝ := sorry

/-- Represents the capacity of a small truck in tons -/
def small_truck_capacity : ℝ := sorry

/-- The total capacity of 2 large trucks and 3 small trucks is 15.5 tons -/
axiom condition1 : 2 * large_truck_capacity + 3 * small_truck_capacity = 15.5

/-- The total capacity of 5 large trucks and 6 small trucks is 35 tons -/
axiom condition2 : 5 * large_truck_capacity + 6 * small_truck_capacity = 35

/-- Theorem: 3 large trucks and 5 small trucks can transport 24.5 tons -/
theorem three_large_five_small_capacity : 
  3 * large_truck_capacity + 5 * small_truck_capacity = 24.5 := by sorry

end NUMINAMATH_CALUDE_three_large_five_small_capacity_l3099_309955


namespace NUMINAMATH_CALUDE_men_per_table_l3099_309974

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  women_per_table = 5 →
  total_customers = 40 →
  (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by sorry

end NUMINAMATH_CALUDE_men_per_table_l3099_309974


namespace NUMINAMATH_CALUDE_geometric_sequence_3_pow_l3099_309928

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem geometric_sequence_3_pow (a : ℕ → ℝ) :
  (∀ n, a n = 3^n) →
  geometric_sequence a ∧
  (∀ n, a (n + 1) > a n) ∧
  a 5^2 = a 10 ∧
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_3_pow_l3099_309928


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3099_309965

/-- Represents a 5x6 grid of integers -/
def Grid := Matrix (Fin 5) (Fin 6) ℕ

/-- Checks if a row in the grid has no repeating numbers -/
def rowNoRepeats (g : Grid) (row : Fin 5) : Prop :=
  ∀ i j : Fin 6, i ≠ j → g row i ≠ g row j

/-- Checks if a column in the grid has no repeating numbers -/
def colNoRepeats (g : Grid) (col : Fin 6) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i col ≠ g j col

/-- Checks if all numbers in the grid are between 1 and 6 -/
def validNumbers (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 6, 1 ≤ g i j ∧ g i j ≤ 6

/-- Checks if the sums of specific digits match the given constraints -/
def validSums (g : Grid) : Prop :=
  g 0 0 * 100 + g 0 1 * 10 + g 0 2 = 669 ∧
  g 0 3 * 10 + g 0 4 = 44

/-- The main theorem stating that 41244 satisfies all conditions -/
theorem solution_satisfies_conditions : ∃ (g : Grid),
  (∀ row : Fin 5, rowNoRepeats g row) ∧
  (∀ col : Fin 6, colNoRepeats g col) ∧
  validNumbers g ∧
  validSums g ∧
  g 0 0 = 4 ∧ g 0 1 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 4 ∧ g 0 4 = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3099_309965


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3099_309960

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2*x - 3

/-- Definition of a point on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the parabola with the y-axis -/
def intersection_point : ℝ × ℝ := (0, -3)

/-- Theorem stating that the intersection_point is on the parabola and the y-axis -/
theorem parabola_y_axis_intersection :
  let (x, y) := intersection_point
  parabola x y ∧ on_y_axis x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3099_309960


namespace NUMINAMATH_CALUDE_prob_heads_tails_tails_l3099_309990

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a fair coin flip -/
def prob_tails : ℚ := 1/2

/-- The number of coin flips -/
def num_flips : ℕ := 3

/-- Theorem: The probability of getting heads on the first flip and tails on the last two flips
    when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_tails : prob_heads * prob_tails * prob_tails = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_heads_tails_tails_l3099_309990


namespace NUMINAMATH_CALUDE_negative_two_equals_negative_abs_two_l3099_309935

theorem negative_two_equals_negative_abs_two : -2 = -|-2| := by
  sorry

end NUMINAMATH_CALUDE_negative_two_equals_negative_abs_two_l3099_309935


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l3099_309952

/-- The number of times Terrell lifts the weights -/
def usual_lifts : ℕ := 10

/-- The weight of each dumbbell Terrell usually uses (in pounds) -/
def usual_weight : ℕ := 25

/-- The weight of each new dumbbell Terrell wants to use (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℚ :=
  (total_weight usual_weight usual_lifts : ℚ) / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l3099_309952


namespace NUMINAMATH_CALUDE_quadratic_roots_l3099_309934

theorem quadratic_roots (a b c : ℝ) (h : (b^3)^2 - 4*(a^3)*(c^3) > 0) :
  (b^5)^2 - 4*(a^5)*(c^5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3099_309934


namespace NUMINAMATH_CALUDE_trenton_earnings_goal_l3099_309984

/-- Trenton's weekly earnings calculation --/
def weekly_earnings (base_pay : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  base_pay + commission_rate * sales

theorem trenton_earnings_goal :
  let base_pay : ℝ := 190
  let commission_rate : ℝ := 0.04
  let min_sales : ℝ := 7750
  let goal : ℝ := 500
  weekly_earnings base_pay commission_rate min_sales = goal := by
sorry

end NUMINAMATH_CALUDE_trenton_earnings_goal_l3099_309984


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3099_309981

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3099_309981


namespace NUMINAMATH_CALUDE_triangle_area_l3099_309941

/-- Given a triangle with perimeter 28 cm and inradius 2.0 cm, its area is 28 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2 → area = inradius * (perimeter / 2) → area = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3099_309941


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_8_is_452_l3099_309958

/-- The count of three-digit whole numbers containing at least one digit 6 or at least one digit 8 -/
def count_numbers_with_6_or_8 : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let digits_without_6_or_8 := 8  -- 0-5, 7, 9
  let first_digit_choices := 7    -- 1-5, 7, 9
  let numbers_without_6_or_8 := first_digit_choices * digits_without_6_or_8 * digits_without_6_or_8
  total_three_digit_numbers - numbers_without_6_or_8

theorem count_numbers_with_6_or_8_is_452 : count_numbers_with_6_or_8 = 452 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_8_is_452_l3099_309958


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l3099_309962

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), 
    (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 7*x - 18)) ∧ 
    (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 7*x - 18 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l3099_309962


namespace NUMINAMATH_CALUDE_function_properties_l3099_309924

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2 + 1

-- Define the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 2 ∨ a = -2) ∧
  (∀ x y : ℝ, x < y ∧ x ≤ 0 ∧ 0 < y → f a x > f a y) ∧
  (∀ x y : ℝ, x < y ∧ 0 < x → f a x < f a y) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3099_309924


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3099_309949

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x + y : ℕ) ≤ (a + b : ℕ)) ∧ (x + y : ℕ) = 49 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3099_309949


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3099_309961

/-- Given a geometric sequence {a_n} with all positive terms,
    if a_3, (1/2)a_5, a_4 form an arithmetic sequence,
    then (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (a 3 + a 4 = a 5) →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3099_309961


namespace NUMINAMATH_CALUDE_patio_surrounded_by_bushes_l3099_309905

/-- The side length of the square patio in feet -/
def patio_side_length : ℝ := 20

/-- The spacing between rose bushes in feet -/
def bush_spacing : ℝ := 2

/-- The number of rose bushes needed to surround the patio -/
def num_bushes : ℕ := 40

/-- Theorem stating that the number of rose bushes needed to surround the square patio is 40 -/
theorem patio_surrounded_by_bushes :
  (4 * patio_side_length) / bush_spacing = num_bushes := by sorry

end NUMINAMATH_CALUDE_patio_surrounded_by_bushes_l3099_309905


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l3099_309945

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l3099_309945


namespace NUMINAMATH_CALUDE_bakers_cakes_l3099_309950

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 8)
  (h2 : bought_cakes = 139)
  (h3 : sold_cakes = 145) :
  sold_cakes - bought_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3099_309950


namespace NUMINAMATH_CALUDE_cube_root_unity_polynomial_identity_l3099_309936

theorem cube_root_unity_polynomial_identity
  (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_polynomial_identity_l3099_309936


namespace NUMINAMATH_CALUDE_sum_of_nth_row_l3099_309932

/-- Represents the sum of numbers in the nth row of the triangular array -/
def row_sum (n : ℕ) : ℕ := 2^n

/-- The first row sum is 2 -/
axiom first_row : row_sum 1 = 2

/-- Each subsequent row sum is double the previous row sum -/
axiom double_previous (n : ℕ) : n ≥ 1 → row_sum (n + 1) = 2 * row_sum n

/-- The sum of numbers in the nth row of the triangular array is 2^n -/
theorem sum_of_nth_row (n : ℕ) : n ≥ 1 → row_sum n = 2^n := by sorry

end NUMINAMATH_CALUDE_sum_of_nth_row_l3099_309932


namespace NUMINAMATH_CALUDE_volume_central_region_is_one_sixth_l3099_309969

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  -- Add necessary fields/axioms for a unit cube

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- Represents the central region (regular octahedron) formed by intersecting planes -/
structure CentralRegion where
  cube : UnitCube
  intersecting_planes : List Plane
  -- Add necessary conditions to ensure the planes intersect at midpoints of edges

/-- Calculate the volume of the central region in a unit cube intersected by specific planes -/
def volume_central_region (region : CentralRegion) : ℝ :=
  sorry

/-- Theorem stating that the volume of the central region is 1/6 -/
theorem volume_central_region_is_one_sixth (region : CentralRegion) :
  volume_central_region region = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_volume_central_region_is_one_sixth_l3099_309969


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l3099_309918

-- Define the total number of employees
variable (E : ℝ)

-- Define the percentage of fair-haired employees who are women
def fair_haired_women_ratio : ℝ := 0.4

-- Define the percentage of employees who have fair hair
def fair_haired_ratio : ℝ := 0.8

-- Define the percentage of employees who are women with fair hair
def women_fair_hair_ratio : ℝ := fair_haired_women_ratio * fair_haired_ratio

-- Theorem statement
theorem women_fair_hair_percentage :
  women_fair_hair_ratio = 0.32 :=
sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l3099_309918


namespace NUMINAMATH_CALUDE_min_value_my_plus_nx_l3099_309964

theorem min_value_my_plus_nx (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∀ z : ℝ, m * y + n * x ≥ z → z ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_my_plus_nx_l3099_309964


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3099_309959

/-- Given a differentiable function f, prove that its derivative at x = 1 is 2,
    given that the tangent line equation at (1, f(1)) is 2x - y + 2 = 0. -/
theorem tangent_line_slope (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, x = 1 ∧ y = f 1 → 2 * x - y + 2 = 0) →
  deriv f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3099_309959


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3099_309901

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0, and eccentricity e = 2,
    the equation of its asymptotes is y = ±√3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3099_309901


namespace NUMINAMATH_CALUDE_slope_implies_y_value_l3099_309921

/-- Given two points A(4, y) and B(2, -3), if the slope of the line passing through these points is π/4, then y = -1 -/
theorem slope_implies_y_value (y : ℝ) :
  let A : ℝ × ℝ := (4, y)
  let B : ℝ × ℝ := (2, -3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = π / 4 → y = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_implies_y_value_l3099_309921


namespace NUMINAMATH_CALUDE_determinant_scaling_l3099_309914

theorem determinant_scaling {x y z w : ℝ} (h : Matrix.det !![x, y; z, w] = 3) :
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 27 := by sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3099_309914


namespace NUMINAMATH_CALUDE_polynomial_roots_l3099_309939

/-- The polynomial x^3 + x^2 - 4x - 2 --/
def f (x : ℂ) : ℂ := x^3 + x^2 - 4*x - 2

/-- The roots of the polynomial --/
def roots : List ℂ := [1, -1 + Complex.I, -1 - Complex.I]

theorem polynomial_roots :
  ∀ r ∈ roots, f r = 0 ∧ (∀ z : ℂ, f z = 0 → z ∈ roots) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3099_309939


namespace NUMINAMATH_CALUDE_absolute_value_integral_l3099_309966

theorem absolute_value_integral : ∫ x in (0:ℝ)..4, |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l3099_309966


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3099_309916

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b → a > b - 1) ∧
  (∃ a b : ℝ, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3099_309916


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3099_309954

theorem contrapositive_equivalence :
  (∀ x : ℝ, x > 10 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3099_309954


namespace NUMINAMATH_CALUDE_multiple_of_nine_squared_greater_than_80_less_than_30_l3099_309912

theorem multiple_of_nine_squared_greater_than_80_less_than_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 80)
  (h3 : x < 30) :
  x = 9 ∨ x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_squared_greater_than_80_less_than_30_l3099_309912


namespace NUMINAMATH_CALUDE_rhino_state_reachable_l3099_309910

/-- Represents the state of a Rhinoceros with folds on its skin -/
structure RhinoState :=
  (left_vertical : Nat)
  (left_horizontal : Nat)
  (right_vertical : Nat)
  (right_horizontal : Nat)

/-- Represents the direction of scratching -/
inductive ScratchDirection
  | Vertical
  | Horizontal

/-- Represents the side of the Rhinoceros being scratched -/
inductive Side
  | Left
  | Right

/-- Defines a single transition step for a Rhinoceros state -/
def transition (s : RhinoState) (dir : ScratchDirection) (side : Side) : RhinoState :=
  sorry

/-- Defines if a target state is reachable from an initial state -/
def is_reachable (initial : RhinoState) (target : RhinoState) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem rhino_state_reachable :
  is_reachable
    (RhinoState.mk 0 2 2 1)
    (RhinoState.mk 2 0 2 1) :=
  sorry

end NUMINAMATH_CALUDE_rhino_state_reachable_l3099_309910


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3099_309908

theorem area_between_concentric_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 60) 
  (h₂ : r₂ = 40) 
  (h₃ : chord_length = 100) 
  (h₄ : r₁ > r₂) 
  (h₅ : chord_length / 2 > r₂) : 
  (r₁^2 - r₂^2) * π = 2500 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3099_309908


namespace NUMINAMATH_CALUDE_bridget_apples_l3099_309988

theorem bridget_apples (x : ℕ) : 
  (2 * x) / 3 - 11 = 10 → x = 32 := by sorry

end NUMINAMATH_CALUDE_bridget_apples_l3099_309988


namespace NUMINAMATH_CALUDE_sequence_property_l3099_309919

def sequence_sum (n : ℕ) : ℚ := n * (3 * n - 1) / 2

def sequence_term (n : ℕ) : ℚ := 3 * n - 2

theorem sequence_property (m : ℕ) :
  (∀ n, sequence_sum n = n * (3 * n - 1) / 2) →
  (∀ n, sequence_term n = 3 * n - 2) →
  sequence_term 1 * sequence_term m = (sequence_term 4) ^ 2 →
  m = 34 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3099_309919


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l3099_309989

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 2) / 2

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l3099_309989


namespace NUMINAMATH_CALUDE_bombardment_percentage_l3099_309903

/-- Proves that the percentage of people who died by bombardment is 10% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4500)
  (h2 : final_population = 3240)
  (h3 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population)))) :
  ∃ x : ℝ, x = 10 ∧ 
    final_population = initial_population - (x / 100 * initial_population) - 
    (1/5 * (initial_population - (x / 100 * initial_population))) :=
by sorry

#check bombardment_percentage

end NUMINAMATH_CALUDE_bombardment_percentage_l3099_309903


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3099_309947

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 3 / Real.sqrt 1000 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3099_309947


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3099_309971

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the lines
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the chord length
def chord_length (k m : ℝ) : ℝ := 
  2 * Real.sqrt 2 * (Real.sqrt (1 + k^2) * Real.sqrt (2 * k^2 - m^2 + 1)) / (1 + 2 * k^2)

-- Define the area of the quadrilateral
def quad_area (k m₁ : ℝ) : ℝ := 
  4 * Real.sqrt 2 * Real.sqrt ((2 * k^2 - m₁^2 + 1) * m₁^2) / (1 + 2 * k^2)

-- State the theorem
theorem ellipse_intersection_theorem (k m₁ m₂ : ℝ) 
  (h₁ : m₁ ≠ m₂) 
  (h₂ : chord_length k m₁ = chord_length k m₂) : 
  m₁ + m₂ = 0 ∧ 
  ∀ m, quad_area k m ≤ 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3099_309971


namespace NUMINAMATH_CALUDE_determine_x_with_gcd_queries_l3099_309922

theorem determine_x_with_gcd_queries :
  ∀ X : ℕ+, X ≤ 100 →
  ∃ (queries : Fin 7 → ℕ+ × ℕ+),
    (∀ i, (queries i).1 < 100 ∧ (queries i).2 < 100) ∧
    ∀ Y : ℕ+, Y ≤ 100 →
      (∀ i, Nat.gcd (X + (queries i).1) (queries i).2 = Nat.gcd (Y + (queries i).1) (queries i).2) →
      X = Y := by
  sorry

end NUMINAMATH_CALUDE_determine_x_with_gcd_queries_l3099_309922


namespace NUMINAMATH_CALUDE_count_negative_rationals_l3099_309973

/-- The number of negative rational numbers in the given set is 3 -/
theorem count_negative_rationals : 
  let S : Finset ℚ := {-|(-2:ℚ)|, -(2:ℚ)^2019, -(-1:ℚ), 0, -(-2:ℚ)^2}
  (S.filter (λ x => x < 0)).card = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l3099_309973


namespace NUMINAMATH_CALUDE_max_min_difference_l3099_309985

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

-- Define the interval
def interval : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_difference (a : ℝ) :
  ∃ (M N : ℝ),
    (∀ x ∈ interval, f a x ≤ M) ∧
    (∃ x ∈ interval, f a x = M) ∧
    (∀ x ∈ interval, N ≤ f a x) ∧
    (∃ x ∈ interval, f a x = N) ∧
    M - N = 18 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l3099_309985


namespace NUMINAMATH_CALUDE_max_f_and_min_sum_l3099_309972

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 4|

-- Theorem statement
theorem max_f_and_min_sum :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∃ m : ℝ, m = 4 ∧
   ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = m →
   2/a + 9/b ≥ 8 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = m ∧ 2/a₀ + 9/b₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_f_and_min_sum_l3099_309972


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l3099_309963

/-- A parabola is tangent to a line if and only if the discriminant of their difference is zero -/
def is_tangent (a : ℝ) : Prop :=
  (4 : ℝ) - 12 * a = 0

/-- The value of a for which the parabola y = ax^2 + 6 is tangent to the line y = 2x + 3 -/
theorem parabola_tangent_line : ∃ (a : ℝ), is_tangent a ∧ a = (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l3099_309963


namespace NUMINAMATH_CALUDE_smallest_n_with_property_l3099_309915

def has_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n - 2) ⊔ {3, 4} → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c)

theorem smallest_n_with_property :
  (∀ k < 243, ¬ has_property k) ∧ has_property 243 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_property_l3099_309915


namespace NUMINAMATH_CALUDE_tank_fill_time_l3099_309929

/-- Represents the time it takes to fill a tank given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given specific pipe rates, the fill time is 20 minutes -/
theorem tank_fill_time :
  fill_time (1/18) (1/60) (-1/45) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3099_309929


namespace NUMINAMATH_CALUDE_bread_price_for_cash_register_l3099_309938

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  breadPrice : ℝ
  breadSold : ℕ
  cakesPrice : ℝ
  cakesSold : ℕ
  rentCost : ℝ
  electricityCost : ℝ

/-- Calculates the daily profit of the bakery --/
def dailyProfit (b : BakeryFinances) : ℝ :=
  b.breadPrice * b.breadSold + b.cakesPrice * b.cakesSold - b.rentCost - b.electricityCost

/-- The main theorem: The price of bread that allows Marie to buy the cash register in 8 days is $2 --/
theorem bread_price_for_cash_register (b : BakeryFinances) 
    (h1 : b.breadSold = 40)
    (h2 : b.cakesSold = 6)
    (h3 : b.cakesPrice = 12)
    (h4 : b.rentCost = 20)
    (h5 : b.electricityCost = 2)
    (h6 : 8 * dailyProfit b = 1040) : 
  b.breadPrice = 2 := by
  sorry

#check bread_price_for_cash_register

end NUMINAMATH_CALUDE_bread_price_for_cash_register_l3099_309938


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_total_volume_l3099_309997

-- Define the conversion factor
def meters_to_centimeters : ℝ := 100

-- Theorem 1: One cubic meter is equal to 1,000,000 cubic centimeters
theorem cubic_meter_to_cubic_centimeters : 
  (meters_to_centimeters ^ 3 : ℝ) = 1000000 := by sorry

-- Theorem 2: The sum of one cubic meter and 500 cubic centimeters is equal to 1,000,500 cubic centimeters
theorem total_volume (cubic_cm_to_add : ℝ) : 
  cubic_cm_to_add = 500 → 
  (meters_to_centimeters ^ 3 + cubic_cm_to_add : ℝ) = 1000500 := by sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_total_volume_l3099_309997


namespace NUMINAMATH_CALUDE_count_ones_digits_divisible_by_six_l3099_309968

/-- A number is divisible by 6 if and only if it is divisible by both 2 and 3 -/
axiom divisible_by_six (n : ℕ) : n % 6 = 0 ↔ n % 2 = 0 ∧ n % 3 = 0

/-- The set of possible ones digits in numbers divisible by 6 -/
def ones_digits_divisible_by_six : Finset ℕ :=
  {0, 2, 4, 6, 8}

/-- The number of possible ones digits in numbers divisible by 6 is 5 -/
theorem count_ones_digits_divisible_by_six :
  Finset.card ones_digits_divisible_by_six = 5 := by sorry

end NUMINAMATH_CALUDE_count_ones_digits_divisible_by_six_l3099_309968


namespace NUMINAMATH_CALUDE_equation_solution_l3099_309900

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3099_309900


namespace NUMINAMATH_CALUDE_tile_border_ratio_l3099_309995

theorem tile_border_ratio (s d : ℝ) (h1 : s > 0) (h2 : d > 0) : 
  (25 * s)^2 / ((25 * s + 2 * d)^2) = 0.81 → d / s = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l3099_309995


namespace NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l3099_309970

theorem gcd_n_cubed_plus_16_and_n_plus_3 (n : ℕ) (h : n > 8) :
  Nat.gcd (n^3 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l3099_309970


namespace NUMINAMATH_CALUDE_snow_probability_first_week_january_l3099_309946

def probability_of_snow (days : ℕ) (prob : ℚ) : ℚ :=
  1 - (1 - prob) ^ days

theorem snow_probability_first_week_january : 
  1 - (1 - probability_of_snow 3 (1/2)) * (1 - probability_of_snow 4 (1/3)) = 79/81 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_january_l3099_309946


namespace NUMINAMATH_CALUDE_cases_in_1990_l3099_309982

/-- Calculates the number of cases in a given year assuming linear decrease --/
def casesInYear (initialCases : ℕ) (finalCases : ℕ) (initialYear : ℕ) (finalYear : ℕ) (targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let yearsFromInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * yearsFromInitial)

/-- The number of cases in 1990 given linear decrease from 1970 to 2000 --/
theorem cases_in_1990 : 
  casesInYear 600000 200 1970 2000 1990 = 200133 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1990_l3099_309982


namespace NUMINAMATH_CALUDE_eggs_ratio_is_one_to_one_l3099_309956

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the total number of eggs Megan initially had --/
def initial_eggs : ℕ := 2 * dozen

/-- Represents the number of eggs Megan used for cooking --/
def used_eggs : ℕ := 2 + 4

/-- Represents the number of eggs Megan plans to use for her meals --/
def planned_meals_eggs : ℕ := 3 * 3

/-- Theorem stating that the ratio of eggs Megan gave to her aunt to the eggs she kept for herself is 1:1 --/
theorem eggs_ratio_is_one_to_one : 
  (initial_eggs - used_eggs - planned_meals_eggs) = planned_meals_eggs := by
  sorry

end NUMINAMATH_CALUDE_eggs_ratio_is_one_to_one_l3099_309956


namespace NUMINAMATH_CALUDE_special_triangle_angles_special_triangle_exists_l3099_309953

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Angles of the triangle
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Condition: The sum of angles in a triangle is 180°
  angle_sum : angleA + angleB + angleC = 180
  -- Condition: Angle C is a right angle
  right_angleC : angleC = 90
  -- Condition: Angle A is one-fourth of angle B
  angle_relation : angleA = angleB / 3

/-- Theorem stating the angles of the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angleA = 22.5 ∧ t.angleB = 67.5 ∧ t.angleC = 90 := by
  sorry

/-- The existence of such a triangle -/
theorem special_triangle_exists : ∃ t : SpecialTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angles_special_triangle_exists_l3099_309953


namespace NUMINAMATH_CALUDE_annual_interest_rate_is_eight_percent_l3099_309987

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem annual_interest_rate_is_eight_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (total : ℝ) 
  (time : ℕ) 
  (h1 : principal + interest = total)
  (h2 : interest = 2828.80)
  (h3 : total = 19828.80)
  (h4 : time = 2) :
  compound_interest principal 0.08 time = interest := by
  sorry

#check annual_interest_rate_is_eight_percent

end NUMINAMATH_CALUDE_annual_interest_rate_is_eight_percent_l3099_309987


namespace NUMINAMATH_CALUDE_lottery_winnings_l3099_309967

/-- Calculates the total money won in a lottery given the number of tickets, winning numbers per ticket, and value per winning number. -/
def total_money_won (num_tickets : ℕ) (winning_numbers_per_ticket : ℕ) (value_per_winning_number : ℕ) : ℕ :=
  num_tickets * winning_numbers_per_ticket * value_per_winning_number

/-- Proves that with 3 lottery tickets, 5 winning numbers per ticket, and $20 per winning number, the total money won is $300. -/
theorem lottery_winnings :
  total_money_won 3 5 20 = 300 := by
  sorry

#eval total_money_won 3 5 20

end NUMINAMATH_CALUDE_lottery_winnings_l3099_309967


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_geq_two_l3099_309979

theorem x_plus_reciprocal_geq_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_geq_two_l3099_309979


namespace NUMINAMATH_CALUDE_only_solutions_are_24_and_42_l3099_309994

/-- Reverses the digits of a natural number -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Computes the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number has no zeros in its decimal representation -/
def no_zeros (n : ℕ) : Prop := sorry

/-- The main theorem stating that 24 and 42 are the only solutions -/
theorem only_solutions_are_24_and_42 :
  {X : ℕ | no_zeros X ∧ X * (reverse_digits X) = 1000 + product_of_digits X} = {24, 42} :=
sorry

end NUMINAMATH_CALUDE_only_solutions_are_24_and_42_l3099_309994


namespace NUMINAMATH_CALUDE_solution_sets_equal_l3099_309942

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OneToOne (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = f y → x = y

def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | f (f x) = x}

theorem solution_sets_equal
  (f : ℝ → ℝ)
  (h_increasing : StrictlyIncreasing f)
  (h_onetoone : OneToOne f) :
  SolutionSetP f = SolutionSetQ f :=
sorry

end NUMINAMATH_CALUDE_solution_sets_equal_l3099_309942


namespace NUMINAMATH_CALUDE_cat_whiskers_ratio_l3099_309904

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem stating the relationship between the cats' whiskers and their ratio -/
theorem cat_whiskers_ratio (c : CatWhiskers) : 
  c.juniper = 12 →
  c.buffy = 40 →
  c.puffy = 3 * c.juniper →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  (Ratio.mk c.puffy c.scruffy) = (Ratio.mk 1 2) := by
  sorry


end NUMINAMATH_CALUDE_cat_whiskers_ratio_l3099_309904


namespace NUMINAMATH_CALUDE_min_rental_cost_l3099_309917

/-- Represents the rental arrangement for buses --/
structure RentalArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental arrangement is valid according to the given constraints --/
def is_valid_arrangement (arr : RentalArrangement) : Prop :=
  36 * arr.typeA + 60 * arr.typeB ≥ 900 ∧
  arr.typeA + arr.typeB ≤ 21 ∧
  arr.typeB - arr.typeA ≤ 7

/-- Calculates the total cost for a given rental arrangement --/
def total_cost (arr : RentalArrangement) : ℕ :=
  1600 * arr.typeA + 2400 * arr.typeB

/-- Theorem stating that the minimum rental cost is 36800 yuan --/
theorem min_rental_cost :
  ∃ (arr : RentalArrangement),
    is_valid_arrangement arr ∧
    total_cost arr = 36800 ∧
    ∀ (other : RentalArrangement),
      is_valid_arrangement other →
      total_cost other ≥ 36800 :=
sorry

end NUMINAMATH_CALUDE_min_rental_cost_l3099_309917


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_50_l3099_309933

/-- Represents a triangle with specific side lengths -/
structure Triangle where
  left_side : ℝ
  right_side : ℝ
  base : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  t.left_side + t.right_side + t.base

/-- Theorem: The perimeter of a triangle with given conditions is 50 cm -/
theorem triangle_perimeter_is_50 :
  ∀ t : Triangle,
    t.left_side = 12 →
    t.right_side = t.left_side + 2 →
    t.base = 24 →
    perimeter t = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_50_l3099_309933


namespace NUMINAMATH_CALUDE_dividend_calculation_l3099_309913

/-- Calculates the total dividends received over three years given an initial investment and dividend rates. -/
def total_dividends (initial_investment : ℚ) (share_face_value : ℚ) (initial_premium : ℚ) 
  (dividend_rate1 : ℚ) (dividend_rate2 : ℚ) (dividend_rate3 : ℚ) : ℚ :=
  let cost_per_share := share_face_value * (1 + initial_premium)
  let num_shares := initial_investment / cost_per_share
  let dividend1 := num_shares * share_face_value * dividend_rate1
  let dividend2 := num_shares * share_face_value * dividend_rate2
  let dividend3 := num_shares * share_face_value * dividend_rate3
  dividend1 + dividend2 + dividend3

/-- Theorem stating that the total dividends received is 2640 given the specified conditions. -/
theorem dividend_calculation :
  total_dividends 14400 100 (1/5) (7/100) (9/100) (6/100) = 2640 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3099_309913


namespace NUMINAMATH_CALUDE_always_close_piece_l3099_309944

-- Define the grid structure
structure Grid :=
  (points : Set (ℤ × ℤ))
  (adjacent : (ℤ × ℤ) → Set (ℤ × ℤ))
  (initial : ℤ × ℤ)

-- Define the grid distance
def gridDistance (g : Grid) (p : ℤ × ℤ) : ℕ :=
  sorry

-- Define the marking function
def mark (n : ℕ) : ℚ :=
  1 / 2^n

-- Define the sum of markings for pieces
def pieceSum (g : Grid) (pieces : Set (ℤ × ℤ)) : ℚ :=
  sorry

-- Define the sum of markings for points with grid distance ≥ 7
def distantSum (g : Grid) : ℚ :=
  sorry

-- Main theorem
theorem always_close_piece (g : Grid) (pieces : Set (ℤ × ℤ)) :
  pieceSum g pieces > distantSum g :=
sorry

end NUMINAMATH_CALUDE_always_close_piece_l3099_309944


namespace NUMINAMATH_CALUDE_babysitting_theorem_l3099_309925

def babysitting_earnings (initial_charge : ℝ) (hours : ℕ) : ℝ :=
  let rec calc_earnings (h : ℕ) (prev_charge : ℝ) (total : ℝ) : ℝ :=
    if h = 0 then
      total
    else
      calc_earnings (h - 1) (prev_charge * 1.5) (total + prev_charge)
  calc_earnings hours initial_charge 0

theorem babysitting_theorem :
  babysitting_earnings 4 4 = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_theorem_l3099_309925


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3099_309907

theorem students_liking_both_desserts
  (total_students : ℕ)
  (like_apple_pie : ℕ)
  (like_chocolate_cake : ℕ)
  (like_neither : ℕ)
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 10) :
  (like_apple_pie + like_chocolate_cake) - (total_students - like_neither) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3099_309907
