import Mathlib

namespace NUMINAMATH_CALUDE_three_common_points_l2135_213596

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point satisfies the first equation -/
def satisfiesEq1 (p : Point) : Prop :=
  (2 * p.x - 3 * p.y + 6) * (5 * p.x + 2 * p.y - 10) = 0

/-- Function to check if a point satisfies the second equation -/
def satisfiesEq2 (p : Point) : Prop :=
  (p.x - 2 * p.y + 1) * (3 * p.x - 4 * p.y + 8) = 0

/-- The main theorem stating that there are exactly 3 common points -/
theorem three_common_points :
  ∃ (p1 p2 p3 : Point),
    satisfiesEq1 p1 ∧ satisfiesEq2 p1 ∧
    satisfiesEq1 p2 ∧ satisfiesEq2 p2 ∧
    satisfiesEq1 p3 ∧ satisfiesEq2 p3 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (p : Point), satisfiesEq1 p ∧ satisfiesEq2 p → p = p1 ∨ p = p2 ∨ p = p3 :=
sorry


end NUMINAMATH_CALUDE_three_common_points_l2135_213596


namespace NUMINAMATH_CALUDE_min_total_time_for_three_students_l2135_213526

/-- Represents a student with their bucket filling time -/
structure Student where
  name : String
  fillTime : Real

/-- Calculates the minimum total time for students to fill their buckets -/
def minTotalTime (students : List Student) : Real :=
  sorry

/-- Theorem stating the minimum total time for the given scenario -/
theorem min_total_time_for_three_students :
  let students := [
    { name := "A", fillTime := 1.5 },
    { name := "B", fillTime := 0.5 },
    { name := "C", fillTime := 1.0 }
  ]
  minTotalTime students = 5 := by sorry

end NUMINAMATH_CALUDE_min_total_time_for_three_students_l2135_213526


namespace NUMINAMATH_CALUDE_find_numbers_l2135_213580

theorem find_numbers (A B C : ℕ) 
  (h1 : Nat.gcd A B = 2)
  (h2 : Nat.lcm A B = 60)
  (h3 : Nat.gcd A C = 3)
  (h4 : Nat.lcm A C = 42) :
  A = 6 ∧ B = 20 ∧ C = 21 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_l2135_213580


namespace NUMINAMATH_CALUDE_sqrt3_times_sqrt10_minus_sqrt3_bounds_l2135_213511

theorem sqrt3_times_sqrt10_minus_sqrt3_bounds :
  2 < Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) ∧
  Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) < 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt3_times_sqrt10_minus_sqrt3_bounds_l2135_213511


namespace NUMINAMATH_CALUDE_cubic_root_nature_l2135_213555

theorem cubic_root_nature :
  ∃ (p n1 n2 : ℝ), p > 0 ∧ n1 < 0 ∧ n2 < 0 ∧
  p^3 + 3*p^2 - 4*p - 12 = 0 ∧
  n1^3 + 3*n1^2 - 4*n1 - 12 = 0 ∧
  n2^3 + 3*n2^2 - 4*n2 - 12 = 0 ∧
  ∀ x : ℝ, x^3 + 3*x^2 - 4*x - 12 = 0 → x = p ∨ x = n1 ∨ x = n2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_nature_l2135_213555


namespace NUMINAMATH_CALUDE_problem_statement_l2135_213568

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_squares_nonzero : a^2 + b^2 + c^2 ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a^2 + b^2 + c^2)) = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2135_213568


namespace NUMINAMATH_CALUDE_power_station_output_scientific_notation_l2135_213542

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem power_station_output_scientific_notation :
  toScientificNotation 448000 = ScientificNotation.mk 4.48 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_power_station_output_scientific_notation_l2135_213542


namespace NUMINAMATH_CALUDE_oliver_gave_janet_ten_pounds_l2135_213525

/-- The amount of candy Oliver gave to Janet -/
def candy_given_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : ℕ :=
  initial_candy - remaining_candy

/-- Proof that Oliver gave Janet 10 pounds of candy -/
theorem oliver_gave_janet_ten_pounds :
  candy_given_to_janet 78 68 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_gave_janet_ten_pounds_l2135_213525


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_l2135_213517

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def primes_in_range (a b : ℕ) : Set ℕ :=
  {n : ℕ | a ≤ n ∧ n ≤ b ∧ is_prime n}

theorem sum_smallest_largest_prime :
  let P := primes_in_range 1 50
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 49 :=
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_l2135_213517


namespace NUMINAMATH_CALUDE_fraction_simplification_l2135_213592

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) :
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2135_213592


namespace NUMINAMATH_CALUDE_triangle_inequalities_and_side_relationships_l2135_213556

theorem triangle_inequalities_and_side_relationships (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∃ (x y z : ℝ), x^2 = a ∧ y^2 = b ∧ z^2 = c ∧ x + y > z ∧ y + z > x ∧ z + x > y) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_and_side_relationships_l2135_213556


namespace NUMINAMATH_CALUDE_gcd_1729_1768_l2135_213579

theorem gcd_1729_1768 : Nat.gcd 1729 1768 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1768_l2135_213579


namespace NUMINAMATH_CALUDE_jamie_yellow_balls_l2135_213554

/-- Proves that Jamie bought 32 yellow balls given the initial conditions and final total -/
theorem jamie_yellow_balls :
  let initial_red : ℕ := 16
  let initial_blue : ℕ := 2 * initial_red
  let lost_red : ℕ := 6
  let final_total : ℕ := 74
  let remaining_red : ℕ := initial_red - lost_red
  let yellow_balls : ℕ := final_total - (remaining_red + initial_blue)
  yellow_balls = 32 := by
  sorry

end NUMINAMATH_CALUDE_jamie_yellow_balls_l2135_213554


namespace NUMINAMATH_CALUDE_special_number_exists_l2135_213581

/-- A function that removes the trailing zero from a binary representation -/
def removeTrailingZero (n : ℕ) : ℕ := sorry

/-- A function that converts a natural number to its ternary representation -/
def toTernary (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of the special number -/
theorem special_number_exists : ∃ N : ℕ, 
  N % 2 = 0 ∧ 
  removeTrailingZero N = toTernary (N / 3) := by
  sorry

end NUMINAMATH_CALUDE_special_number_exists_l2135_213581


namespace NUMINAMATH_CALUDE_max_true_statements_l2135_213569

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (true_statements : List Bool), 
    true_statements.length ≤ 4 ∧ 
    (∀ i, i < statements.length → 
      (true_statements.get! i = true ↔ statements.get! i)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2135_213569


namespace NUMINAMATH_CALUDE_exponent_division_l2135_213584

theorem exponent_division (a : ℝ) : a ^ 3 / a = a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2135_213584


namespace NUMINAMATH_CALUDE_lassi_production_l2135_213587

theorem lassi_production (mangoes : ℕ) (lassis : ℕ) : 
  (3 * lassis = 13 * mangoes) → (15 * lassis = 65 * mangoes) :=
by sorry

end NUMINAMATH_CALUDE_lassi_production_l2135_213587


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2135_213505

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2135_213505


namespace NUMINAMATH_CALUDE_new_triangle_is_acute_l2135_213522

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the new triangle after increasing each side by x
def NewTriangle (t : RightTriangle) (x : ℝ) : Prop :=
  ∀ (h : 0 < x),
    let new_a := t.a + x
    let new_b := t.b + x
    let new_c := t.c + x
    (new_a^2 + new_b^2 - new_c^2) / (2 * new_a * new_b) > 0

-- Theorem statement
theorem new_triangle_is_acute (t : RightTriangle) :
  ∀ x, NewTriangle t x :=
sorry

end NUMINAMATH_CALUDE_new_triangle_is_acute_l2135_213522


namespace NUMINAMATH_CALUDE_largest_y_floor_div_l2135_213501

theorem largest_y_floor_div : 
  ∀ y : ℝ, (↑(⌊y⌋) / y = 8 / 9) → y ≤ 63 / 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_floor_div_l2135_213501


namespace NUMINAMATH_CALUDE_solve_for_C_l2135_213566

theorem solve_for_C : ∃ C : ℝ, (2 * C - 3 = 11) ∧ (C = 7) := by sorry

end NUMINAMATH_CALUDE_solve_for_C_l2135_213566


namespace NUMINAMATH_CALUDE_line_equation_l2135_213535

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y + 21 = 0

-- Define point A
def point_A : ℝ × ℝ := (-6, 7)

-- Define the property of being tangent to the circle
def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let center := (4, -3)
  let radius := 2
  abs (a * center.1 + b * center.2 + c) / Real.sqrt (a^2 + b^2) = radius

-- Theorem statement
theorem line_equation :
  ∃ (a b c : ℝ), 
    (a * point_A.1 + b * point_A.2 + c = 0) ∧
    is_tangent_to_circle a b c ∧
    ((a = 3 ∧ b = 4 ∧ c = -10) ∨ (a = 4 ∧ b = 3 ∧ c = 3)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2135_213535


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l2135_213557

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ × ℝ → Prop := λ p => a * p.1 + p.2 + 2 = 0
def l₂ (a : ℝ) : ℝ × ℝ → Prop := λ p => p.1 + (a - 2) * p.2 + 1 = 0

-- Define perpendicularity of lines
def perpendicular (f g : ℝ × ℝ → Prop) : Prop :=
  ∃ (m₁ m₂ : ℝ), (∀ p, f p ↔ p.2 = m₁ * p.1 + 0) ∧
                 (∀ p, g p ↔ p.2 = m₂ * p.1 + 0) ∧
                 m₁ * m₂ = -1

-- State the theorem
theorem perpendicular_lines_a_equals_one :
  perpendicular (l₁ a) (l₂ a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l2135_213557


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_third_l2135_213515

theorem trigonometric_expression_equals_one_third :
  let θ : Real := 30 * π / 180  -- 30 degrees in radians
  (Real.tan θ)^2 - (Real.cos θ)^2 = 1/3 * ((Real.tan θ)^2 * (Real.cos θ)^2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_third_l2135_213515


namespace NUMINAMATH_CALUDE_triangle_problem_l2135_213574

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition: 2c - a = 2b*cos(A) -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.c - t.a = 2 * t.b * Real.cos t.A

/-- Theorem stating the two parts of the problem -/
theorem triangle_problem (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.B = π / 3 ∧ 
  (t.a = 2 ∧ t.b = Real.sqrt 7 → t.c = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2135_213574


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2135_213594

/-- The sum of the lengths of sides of one face of a cube with side length 9 cm is 36 cm -/
theorem cube_face_perimeter (cube_side_length : ℝ) (h : cube_side_length = 9) : 
  4 * cube_side_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2135_213594


namespace NUMINAMATH_CALUDE_house_payment_theorem_l2135_213561

/-- Calculates the amount still owed on a house given the initial price,
    down payment percentage, and percentage paid by parents. -/
def amount_owed (price : ℝ) (down_payment_percent : ℝ) (parents_payment_percent : ℝ) : ℝ :=
  let remaining_after_down := price * (1 - down_payment_percent)
  remaining_after_down * (1 - parents_payment_percent)

/-- Theorem stating that for a $100,000 house with 20% down payment
    and 30% of the remaining balance paid by parents, $56,000 is still owed. -/
theorem house_payment_theorem :
  amount_owed 100000 0.2 0.3 = 56000 := by
  sorry

end NUMINAMATH_CALUDE_house_payment_theorem_l2135_213561


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2135_213551

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (3*x^2 - 1) = (x^2 - 50*x - 10)*(x^2 + 25*x + 5)) ∧ x = 25 + Real.sqrt 159 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2135_213551


namespace NUMINAMATH_CALUDE_unique_room_dimensions_l2135_213528

/-- A room with integer dimensions where the unpainted border area is four times the painted area --/
structure PaintedRoom where
  a : ℕ
  b : ℕ
  h1 : 0 < a
  h2 : 0 < b
  h3 : b > a
  h4 : 4 * ((a - 4) * (b - 4)) = a * b - (a - 4) * (b - 4)

/-- The only valid dimensions for the room are 6 by 30 feet --/
theorem unique_room_dimensions : 
  ∀ (room : PaintedRoom), room.a = 6 ∧ room.b = 30 :=
by sorry

end NUMINAMATH_CALUDE_unique_room_dimensions_l2135_213528


namespace NUMINAMATH_CALUDE_second_stop_off_is_two_l2135_213578

/-- Represents the number of passengers on the trolley at various stages --/
structure TrolleyPassengers where
  initial : Nat
  second_stop_off : Nat
  second_stop_on : Nat
  third_stop_off : Nat
  third_stop_on : Nat
  final : Nat

/-- The trolley problem with given conditions --/
def trolleyProblem : TrolleyPassengers where
  initial := 10
  second_stop_off := 2  -- This is what we want to prove
  second_stop_on := 20  -- Twice the initial number
  third_stop_off := 18
  third_stop_on := 2
  final := 12

/-- Theorem stating that the number of people who got off at the second stop is 2 --/
theorem second_stop_off_is_two (t : TrolleyPassengers) : 
  t.initial = 10 ∧ 
  t.second_stop_on = 2 * t.initial ∧ 
  t.third_stop_off = 18 ∧ 
  t.third_stop_on = 2 ∧ 
  t.final = 12 →
  t.second_stop_off = 2 := by
  sorry

#check second_stop_off_is_two trolleyProblem

end NUMINAMATH_CALUDE_second_stop_off_is_two_l2135_213578


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l2135_213538

theorem candy_box_price_increase : 
  ∀ (original_candy_price original_soda_price : ℝ),
  original_candy_price + original_soda_price = 20 →
  original_soda_price * 1.5 = 6 →
  original_candy_price * 1.25 = 20 →
  (20 - original_candy_price) / original_candy_price = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l2135_213538


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2135_213550

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 13
    let b : ℝ := 14
    let c : ℝ := 15
    let s : ℝ := (a + b + c) / 2
    area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ area = 84

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : ∃ area : ℝ, triangle_area area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2135_213550


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l2135_213583

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log (3 * x) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l2135_213583


namespace NUMINAMATH_CALUDE_infinite_possibilities_for_A_squared_l2135_213559

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    there are infinitely many possible matrices that A^2 can be. -/
theorem infinite_possibilities_for_A_squared 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  ∃ S : Set (Matrix (Fin 3) (Fin 3) ℝ), 
    (∀ B ∈ S, ∃ A : Matrix (Fin 3) (Fin 3) ℝ, A ^ 4 = 0 ∧ A ^ 2 = B) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_possibilities_for_A_squared_l2135_213559


namespace NUMINAMATH_CALUDE_union_of_sets_l2135_213533

-- Define the sets A and B
def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

-- Theorem statement
theorem union_of_sets (a b : ℕ) :
  (A a ∩ B a b = {2}) → (A a ∪ B a b = {1, 2, 3}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2135_213533


namespace NUMINAMATH_CALUDE_andrew_work_hours_l2135_213590

/-- The number of days Andrew worked on his Science report -/
def days_worked : ℝ := 3

/-- The number of hours Andrew worked each day -/
def hours_per_day : ℝ := 2.5

/-- The total number of hours Andrew worked on his Science report -/
def total_hours : ℝ := days_worked * hours_per_day

theorem andrew_work_hours : total_hours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l2135_213590


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2135_213591

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 6.2)
  (h2 : (c + d) / 2 = 6.1)
  (h3 : (e + f) / 2 = 6.9) :
  (a + b + c + d + e + f) / 6 = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2135_213591


namespace NUMINAMATH_CALUDE_school_play_girls_l2135_213558

/-- The number of girls in the school play -/
def num_girls : ℕ := 6

/-- The number of boys in the school play -/
def num_boys : ℕ := 8

/-- The total number of parents attending the premiere -/
def total_parents : ℕ := 28

/-- The number of parents per child -/
def parents_per_child : ℕ := 2

theorem school_play_girls :
  num_girls = 6 ∧
  num_boys * parents_per_child + num_girls * parents_per_child = total_parents :=
sorry

end NUMINAMATH_CALUDE_school_play_girls_l2135_213558


namespace NUMINAMATH_CALUDE_max_plus_min_equals_two_l2135_213520

noncomputable def f (x : ℝ) : ℝ := (2^x + 1)^2 / (2^x * x) + 1

def interval := {x : ℝ | (x ∈ Set.Icc (-2018) 0 ∧ x ≠ 0) ∨ (x ∈ Set.Ioc 0 2018)}

theorem max_plus_min_equals_two :
  ∃ (M N : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
               (∀ x ∈ interval, N ≤ f x) ∧
               (∃ x₁ ∈ interval, f x₁ = M) ∧
               (∃ x₂ ∈ interval, f x₂ = N) ∧
               M + N = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_two_l2135_213520


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2135_213523

/-- The equation of a hyperbola passing through a specific point with its asymptote tangent to a circle -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (16 / a^2 - 4 / b^2 = 1) →  -- Hyperbola passes through (4, 2)
  (|2 * Real.sqrt 2 * b| / Real.sqrt (b^2 + a^2) = Real.sqrt (8/3)) →  -- Asymptote tangent to circle
  (∀ x y : ℝ, x^2 / 8 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2135_213523


namespace NUMINAMATH_CALUDE_power_64_five_sixths_l2135_213524

theorem power_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_64_five_sixths_l2135_213524


namespace NUMINAMATH_CALUDE_matt_cookies_left_l2135_213562

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  flour_per_batch : ℕ        -- pounds of flour per batch
  cookies_per_batch : ℕ      -- number of cookies per batch
  flour_bags : ℕ             -- number of flour bags used
  flour_per_bag : ℕ          -- pounds of flour per bag
  cookies_eaten : ℕ          -- number of cookies eaten

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_batches := total_flour / scenario.flour_per_batch
  let total_cookies := total_batches * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in Matt's scenario -/
theorem matt_cookies_left :
  let matt_scenario : CookieScenario := {
    flour_per_batch := 2,
    cookies_per_batch := 12,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left matt_scenario = 105 := by
  sorry


end NUMINAMATH_CALUDE_matt_cookies_left_l2135_213562


namespace NUMINAMATH_CALUDE_lcm_5_6_8_18_l2135_213531

theorem lcm_5_6_8_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 18)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_18_l2135_213531


namespace NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l2135_213534

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] → n ≥ 2 :=
by sorry

theorem two_satisfies_congruence : 527 * 2 ≡ 1083 * 2 [ZMOD 30] :=
by sorry

theorem two_is_smallest : ∀ m : ℕ, m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ 2 :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] ∧ ∀ m : ℕ, (m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l2135_213534


namespace NUMINAMATH_CALUDE_f_is_even_g_is_odd_h_is_neither_l2135_213543

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

-- Define properties of even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statements
theorem f_is_even : is_even f := by sorry

theorem g_is_odd : is_odd g := by sorry

theorem h_is_neither : ¬(is_even h) ∧ ¬(is_odd h) := by sorry

end NUMINAMATH_CALUDE_f_is_even_g_is_odd_h_is_neither_l2135_213543


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2135_213516

theorem cube_sum_given_sum_and_product (x y : ℝ) :
  x + y = 10 → x * y = 15 → x^3 + y^3 = 550 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2135_213516


namespace NUMINAMATH_CALUDE_three_number_sum_l2135_213521

theorem three_number_sum : ∀ a b c : ℝ, 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 30 → 
  a + b + c = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l2135_213521


namespace NUMINAMATH_CALUDE_magnitude_of_8_minus_15i_l2135_213595

theorem magnitude_of_8_minus_15i : Complex.abs (8 - 15*I) = 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_8_minus_15i_l2135_213595


namespace NUMINAMATH_CALUDE_minor_premise_identification_l2135_213529

-- Define the basic propositions
def ship_departs_on_time : Prop := sorry
def ship_arrives_on_time : Prop := sorry

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Define our specific syllogism
def our_syllogism : Syllogism :=
  { major_premise := ship_departs_on_time → ship_arrives_on_time,
    minor_premise := ship_arrives_on_time,
    conclusion := ship_departs_on_time }

-- Theorem to prove
theorem minor_premise_identification :
  our_syllogism.minor_premise = ship_arrives_on_time :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l2135_213529


namespace NUMINAMATH_CALUDE_n_squared_plus_n_plus_one_properties_l2135_213536

theorem n_squared_plus_n_plus_one_properties (n : ℕ) :
  (Odd (n^2 + n + 1)) ∧ (¬ ∃ m : ℕ, n^2 + n + 1 = m^2) := by
  sorry

end NUMINAMATH_CALUDE_n_squared_plus_n_plus_one_properties_l2135_213536


namespace NUMINAMATH_CALUDE_favorite_subject_problem_l2135_213553

theorem favorite_subject_problem (total_students : ℕ) 
  (math_fraction : ℚ) (english_fraction : ℚ) (science_fraction : ℚ)
  (h_total : total_students = 30)
  (h_math : math_fraction = 1 / 5)
  (h_english : english_fraction = 1 / 3)
  (h_science : science_fraction = 1 / 7) :
  total_students - 
  (↑total_students * math_fraction).floor - 
  (↑total_students * english_fraction).floor - 
  ((↑total_students - (↑total_students * math_fraction).floor - (↑total_students * english_fraction).floor) * science_fraction).floor = 12 := by
sorry

end NUMINAMATH_CALUDE_favorite_subject_problem_l2135_213553


namespace NUMINAMATH_CALUDE_prob_two_red_before_three_green_is_two_sevenths_l2135_213519

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
| TwoRed
| ThreeGreen

/-- The probability of drawing 2 red chips before 3 green chips -/
def prob_two_red_before_three_green : ℚ :=
  2 / 7

/-- The number of red chips in the hat initially -/
def initial_red_chips : ℕ := 4

/-- The number of green chips in the hat initially -/
def initial_green_chips : ℕ := 3

/-- The total number of chips in the hat initially -/
def total_chips : ℕ := initial_red_chips + initial_green_chips

/-- Theorem stating that the probability of drawing 2 red chips before 3 green chips is 2/7 -/
theorem prob_two_red_before_three_green_is_two_sevenths :
  prob_two_red_before_three_green = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_before_three_green_is_two_sevenths_l2135_213519


namespace NUMINAMATH_CALUDE_round_table_seats_l2135_213540

/-- Represents a round table with equally spaced seats numbered clockwise -/
structure RoundTable where
  num_seats : ℕ

/-- Represents a seat at the round table -/
structure Seat where
  number : ℕ

/-- Two seats are opposite if they are half the table size apart -/
def are_opposite (t : RoundTable) (s1 s2 : Seat) : Prop :=
  (s2.number - s1.number) % t.num_seats = t.num_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Seat) :
  s1.number = 10 →
  s2.number = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seats_l2135_213540


namespace NUMINAMATH_CALUDE_larger_number_problem_l2135_213597

theorem larger_number_problem (x y : ℝ) : 3 * y = 4 * x → y - x = 8 → y = 32 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2135_213597


namespace NUMINAMATH_CALUDE_nancy_limes_l2135_213570

def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def total_limes : ℕ := 103

theorem nancy_limes : total_limes - (fred_limes + alyssa_limes) = 35 := by
  sorry

end NUMINAMATH_CALUDE_nancy_limes_l2135_213570


namespace NUMINAMATH_CALUDE_cubic_inequality_l2135_213552

theorem cubic_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a > b ∧ a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2135_213552


namespace NUMINAMATH_CALUDE_number_of_boys_l2135_213572

theorem number_of_boys (total_kids : ℕ) (girls : ℕ) (boys : ℕ) :
  total_kids = 9 →
  girls = 3 →
  total_kids = girls + boys →
  boys = 6 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l2135_213572


namespace NUMINAMATH_CALUDE_round_75_36_bar_l2135_213544

/-- Represents a number with a repeating decimal part -/
structure RepeatingDecimal where
  wholePart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 75.363636... -/
def number : RepeatingDecimal :=
  { wholePart := 75,
    nonRepeatingPart := 36,
    repeatingPart := 36 }

theorem round_75_36_bar : roundToHundredth number = 75.37 := by
  sorry

end NUMINAMATH_CALUDE_round_75_36_bar_l2135_213544


namespace NUMINAMATH_CALUDE_proportional_function_and_value_l2135_213599

/-- Given that y+3 is directly proportional to x and y=7 when x=2, prove:
    1. The function expression for y in terms of x
    2. The value of y when x = -1/2 -/
theorem proportional_function_and_value (y : ℝ → ℝ) (k : ℝ) 
    (h1 : ∀ x, y x + 3 = k * x)  -- y+3 is directly proportional to x
    (h2 : y 2 = 7)  -- when x=2, y=7
    : (∀ x, y x = 5*x - 3) ∧ (y (-1/2) = -11/2) := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_and_value_l2135_213599


namespace NUMINAMATH_CALUDE_function_periodicity_l2135_213506

/-- A function satisfying the given functional equation is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l2135_213506


namespace NUMINAMATH_CALUDE_leanna_leftover_money_l2135_213586

/-- Represents the amount of money Leanna has left after purchasing one CD and two cassettes --/
def money_left_over (total_money : ℕ) (cd_price : ℕ) : ℕ :=
  let cassette_price := total_money - 2 * cd_price
  total_money - (cd_price + 2 * cassette_price)

/-- Theorem stating that Leanna will have $5 left over if she chooses to buy one CD and two cassettes --/
theorem leanna_leftover_money : 
  money_left_over 37 14 = 5 := by
  sorry

end NUMINAMATH_CALUDE_leanna_leftover_money_l2135_213586


namespace NUMINAMATH_CALUDE_soccer_team_composition_l2135_213514

theorem soccer_team_composition (total_players goalies defenders : ℕ) 
  (h1 : total_players = 40)
  (h2 : goalies = 3)
  (h3 : defenders = 10)
  : total_players - (goalies + defenders + 2 * defenders) = 7 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_composition_l2135_213514


namespace NUMINAMATH_CALUDE_system_solution_l2135_213504

theorem system_solution (x y : ℝ) : 
  (x^x = y ∧ x^y = y^x) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 2 ∧ y = 4)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2135_213504


namespace NUMINAMATH_CALUDE_rectangle_area_at_stage_4_l2135_213509

/-- Represents the stage number of the rectangle formation process -/
def Stage : ℕ := 4

/-- The side length of each square added at each stage -/
def SquareSideLength : ℝ := 5

/-- The area of the rectangle at a given stage -/
def RectangleArea (stage : ℕ) : ℝ :=
  (stage : ℝ) * SquareSideLength * SquareSideLength

/-- Theorem stating that the area of the rectangle at Stage 4 is 100 square inches -/
theorem rectangle_area_at_stage_4 : RectangleArea Stage = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_at_stage_4_l2135_213509


namespace NUMINAMATH_CALUDE_smallest_m_congruence_l2135_213539

theorem smallest_m_congruence : ∃ m : ℕ+, 
  (∀ k : ℕ+, k < m → ¬(790 * k.val ≡ 1430 * k.val [ZMOD 30])) ∧ 
  (790 * m.val ≡ 1430 * m.val [ZMOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_congruence_l2135_213539


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2135_213532

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second_term : a 1 = 3)
  (h_fourth_term : a 3 = 12) :
  a 0 = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2135_213532


namespace NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2135_213588

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_squared_times_x_cubed_l2135_213588


namespace NUMINAMATH_CALUDE_system_of_equations_l2135_213585

theorem system_of_equations (x y c d : ℝ) : 
  x ≠ 0 →
  y ≠ 0 →
  d ≠ 0 →
  8 * x - 5 * y = c →
  10 * y - 12 * x = d →
  c / d = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l2135_213585


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2135_213598

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2135_213598


namespace NUMINAMATH_CALUDE_son_age_l2135_213548

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2135_213548


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l2135_213589

theorem quadratic_equation_sum (r s : ℝ) : 
  (∀ x, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + r)^2 = s) →
  r + s = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l2135_213589


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_sphere_area_l2135_213564

/-- Given a rectangle ABCD with area 8, when its perimeter is minimized
    and triangle ACD is folded along diagonal AC to form a pyramid D-ABC,
    the surface area of the circumscribed sphere of this pyramid is 16π. -/
theorem min_perimeter_rectangle_sphere_area :
  ∀ (x y : ℝ),
  x > 0 → y > 0 →
  x * y = 8 →
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 8 → 2*(x + y) ≤ 2*(a + b)) →
  16 * Real.pi = 4 * Real.pi * (2 : ℝ)^2 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_sphere_area_l2135_213564


namespace NUMINAMATH_CALUDE_continued_fraction_result_l2135_213567

-- Define the continued fraction representation of x
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2)))

-- State the theorem
theorem continued_fraction_result :
  1 / ((x + 1) * (x - 3)) = (3 + Real.sqrt 3) / (-6) :=
sorry

end NUMINAMATH_CALUDE_continued_fraction_result_l2135_213567


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2135_213508

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2135_213508


namespace NUMINAMATH_CALUDE_dessert_preference_l2135_213518

theorem dessert_preference (total : Nat) (apple : Nat) (chocolate : Nat) (pumpkin : Nat) (none : Nat)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 17)
  (h4 : pumpkin = 10)
  (h5 : none = 15)
  (h6 : total ≥ apple + chocolate + pumpkin - none) :
  ∃ x : Nat, x = 7 ∧ x ≤ apple ∧ x ≤ chocolate ∧ x ≤ pumpkin ∧
  apple + chocolate + pumpkin - 2*x = total - none :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l2135_213518


namespace NUMINAMATH_CALUDE_cattle_area_calculation_l2135_213527

def farm_length : ℝ := 3.6

theorem cattle_area_calculation (width : ℝ) (total_area : ℝ) (cattle_area : ℝ)
  (h1 : width = 2.5 * farm_length)
  (h2 : total_area = farm_length * width)
  (h3 : cattle_area = total_area / 2) :
  cattle_area = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_cattle_area_calculation_l2135_213527


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l2135_213577

theorem cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi 
  (α : Real) 
  (h : Real.cos (α + Real.pi) = -2/3) : 
  Real.sin (α + 3/2 * Real.pi) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l2135_213577


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_l2135_213510

theorem complex_modulus_sqrt (z : ℂ) (h : z^2 = -15 + 8*I) : Complex.abs z = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_l2135_213510


namespace NUMINAMATH_CALUDE_more_heads_probability_l2135_213507

def coin_prob : ℚ := 2/3

def num_flips : ℕ := 5

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def more_heads_prob : ℚ :=
  binomial_probability num_flips 3 coin_prob +
  binomial_probability num_flips 4 coin_prob +
  binomial_probability num_flips 5 coin_prob

theorem more_heads_probability :
  more_heads_prob = 64/81 := by sorry

end NUMINAMATH_CALUDE_more_heads_probability_l2135_213507


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2135_213582

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l2135_213582


namespace NUMINAMATH_CALUDE_right_triangle_tan_A_l2135_213546

theorem right_triangle_tan_A (A B C : Real) (sinB : Real) :
  -- ABC is a right triangle with angle C = 90°
  A + B + C = Real.pi →
  C = Real.pi / 2 →
  -- sin B = 3/5
  sinB = 3 / 5 →
  -- tan A = 4/3
  Real.tan A = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_A_l2135_213546


namespace NUMINAMATH_CALUDE_sequence_not_ap_gp_l2135_213571

theorem sequence_not_ap_gp (a b c d n : ℝ) : 
  a < b ∧ b < c ∧ a > 1 ∧ 
  b = a + d ∧ c = a + 2*d ∧ 
  n > 1 →
  ¬(∃r : ℝ, (Real.log n / Real.log b - Real.log n / Real.log a = r) ∧ 
             (Real.log n / Real.log c - Real.log n / Real.log b = r)) ∧
  ¬(∃r : ℝ, (Real.log n / Real.log b) / (Real.log n / Real.log a) = r ∧ 
             (Real.log n / Real.log c) / (Real.log n / Real.log b) = r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_ap_gp_l2135_213571


namespace NUMINAMATH_CALUDE_add_zero_eq_self_l2135_213547

theorem add_zero_eq_self (x : ℝ) : x + 0 = x := by
  sorry

end NUMINAMATH_CALUDE_add_zero_eq_self_l2135_213547


namespace NUMINAMATH_CALUDE_imaginary_number_real_part_l2135_213560

theorem imaginary_number_real_part (a : ℝ) : 
  let z : ℂ := a + (Complex.I / (1 - Complex.I))
  (∃ (b : ℝ), z = Complex.I * b) → a = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_real_part_l2135_213560


namespace NUMINAMATH_CALUDE_track_circumference_l2135_213545

/-- Represents the circular track and the runners' movement --/
structure TrackSystem where
  circumference : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The conditions of the problem --/
def satisfies_conditions (s : TrackSystem) : Prop :=
  s.speed_a > 0 ∧ s.speed_b > 0 ∧
  s.speed_a ≠ s.speed_b ∧
  (s.circumference / 2) / s.speed_b = 150 / s.speed_a ∧
  (s.circumference - 90) / s.speed_a = (s.circumference / 2 + 90) / s.speed_b

/-- The theorem to be proved --/
theorem track_circumference (s : TrackSystem) :
  satisfies_conditions s → s.circumference = 540 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l2135_213545


namespace NUMINAMATH_CALUDE_smallest_integer_l2135_213500

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 84) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 84 → b ≤ c → b = 35 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l2135_213500


namespace NUMINAMATH_CALUDE_jamie_oyster_collection_l2135_213537

/-- The proportion of oysters that have pearls -/
def pearl_ratio : ℚ := 1/4

/-- The number of dives Jamie makes -/
def num_dives : ℕ := 14

/-- The total number of pearls Jamie collects -/
def total_pearls : ℕ := 56

/-- The number of oysters Jamie can collect during each dive -/
def oysters_per_dive : ℕ := 16

theorem jamie_oyster_collection :
  oysters_per_dive = (total_pearls / num_dives) / pearl_ratio := by
  sorry

end NUMINAMATH_CALUDE_jamie_oyster_collection_l2135_213537


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l2135_213565

/-- The final length of a movie after cutting a scene -/
def final_movie_length (original_length scene_cut : ℕ) : ℕ :=
  original_length - scene_cut

/-- Theorem: The final length of the movie is 52 minutes -/
theorem movie_length_after_cut :
  final_movie_length 60 8 = 52 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l2135_213565


namespace NUMINAMATH_CALUDE_proportion_solution_l2135_213563

theorem proportion_solution :
  ∀ x : ℝ, (0.75 / x = 5 / 6) → x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2135_213563


namespace NUMINAMATH_CALUDE_least_integer_abs_inequality_l2135_213541

theorem least_integer_abs_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y + 5| ≤ 20 → x ≤ y) ∧ |3*x + 5| ≤ 20 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_abs_inequality_l2135_213541


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l2135_213549

def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_given_up : ℕ := 3

theorem carmen_pets_difference :
  initial_cats - cats_given_up - initial_dogs = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l2135_213549


namespace NUMINAMATH_CALUDE_triangle_sin_B_l2135_213502

theorem triangle_sin_B (a b : ℝ) (A : ℝ) :
  a = Real.sqrt 6 →
  b = 2 →
  A = π / 4 →
  Real.sin (Real.arcsin ((b * Real.sin A) / a)) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sin_B_l2135_213502


namespace NUMINAMATH_CALUDE_mod_sum_powers_seven_l2135_213576

theorem mod_sum_powers_seven : (45^1234 + 27^1234) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_seven_l2135_213576


namespace NUMINAMATH_CALUDE_turtle_combination_probability_l2135_213593

/- Define the number of initial turtles -/
def initial_turtles : ℕ := 2017

/- Define the number of combinations -/
def combinations : ℕ := 2015

/- Function to calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/- Probability that a specific turtle is never chosen -/
def prob_never_chosen : ℚ := 1 / (binomial initial_turtles 2)

/- Theorem statement -/
theorem turtle_combination_probability :
  (initial_turtles : ℚ) * prob_never_chosen = 1 / 1008 :=
sorry

end NUMINAMATH_CALUDE_turtle_combination_probability_l2135_213593


namespace NUMINAMATH_CALUDE_x_value_proof_l2135_213575

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2135_213575


namespace NUMINAMATH_CALUDE_angle_properties_l2135_213513

-- Define the angle θ
variable (θ : Real)

-- Define the condition that the terminal side of θ passes through (4, -3)
def terminal_side_condition : Prop := ∃ (k : Real), k > 0 ∧ k * Real.cos θ = 4 ∧ k * Real.sin θ = -3

-- Theorem statement
theorem angle_properties (h : terminal_side_condition θ) : 
  Real.tan θ = -3/4 ∧ 
  (Real.sin (θ + Real.pi/2) + Real.cos θ) / (Real.sin θ - Real.cos (θ - Real.pi)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l2135_213513


namespace NUMINAMATH_CALUDE_trick_deck_cost_l2135_213573

/-- The cost of a trick deck satisfies the given conditions -/
theorem trick_deck_cost : ∃ (cost : ℕ), 
  6 * cost + 2 * cost = 64 ∧ cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l2135_213573


namespace NUMINAMATH_CALUDE_circular_garden_fence_area_ratio_l2135_213512

theorem circular_garden_fence_area_ratio (r : ℝ) (h : r = 12) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_fence_area_ratio_l2135_213512


namespace NUMINAMATH_CALUDE_complement_of_union_l2135_213503

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union :
  (M ∪ N)ᶜ = {1, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2135_213503


namespace NUMINAMATH_CALUDE_expression_non_negative_lower_bound_achievable_l2135_213530

/-- The expression is always non-negative for real x and y -/
theorem expression_non_negative (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 := by
  sorry

/-- The lower bound of 0 is achievable -/
theorem lower_bound_achievable :
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_non_negative_lower_bound_achievable_l2135_213530
