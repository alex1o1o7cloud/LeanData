import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_is_17_4_l1605_160516

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to the y-axis at a given point -/
def isTangentToYAxis (c : Circle) (p : ℝ × ℝ) : Prop :=
  c.center.1 = c.radius ∧ p.1 = 0 ∧ p.2 = c.center.2

/-- Predicate to check if a given x-coordinate is an x-intercept of the circle -/
def isXIntercept (c : Circle) (x : ℝ) : Prop :=
  ∃ y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧ y = 0

theorem circle_radius_is_17_4 (c : Circle) :
  isTangentToYAxis c (0, 2) →
  isXIntercept c 8 →
  c.radius = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_17_4_l1605_160516


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l1605_160522

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  painted_size : ℕ
  total_cubes : ℕ
  painted_cubes : ℕ

/-- Theorem: In a 4x4x4 cube with 2x2 squares painted on each face, 56 unit cubes are unpainted -/
theorem unpainted_cubes_in_4x4x4_cube (c : PaintedCube) 
  (h_size : c.size = 4)
  (h_painted : c.painted_size = 2)
  (h_total : c.total_cubes = c.size ^ 3)
  (h_painted_count : c.painted_cubes = 8) :
  c.total_cubes - c.painted_cubes = 56 := by
  sorry

#check unpainted_cubes_in_4x4x4_cube

end NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l1605_160522


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1605_160592

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1605_160592


namespace NUMINAMATH_CALUDE_phil_coin_collection_l1605_160524

def coin_collection (initial : ℕ) (year1 : ℕ → ℕ) (year2 : ℕ) (year3 : ℕ) (year4 : ℕ) 
                    (year5 : ℕ → ℕ) (year6 : ℕ) (year7 : ℕ) (year8 : ℕ) (year9 : ℕ → ℕ) : ℕ :=
  let after_year1 := year1 initial
  let after_year2 := after_year1 + year2
  let after_year3 := after_year2 + year3
  let after_year4 := after_year3 + year4
  let after_year5 := year5 after_year4
  let after_year6 := after_year5 + year6
  let after_year7 := after_year6 + year7
  let after_year8 := after_year7 + year8
  year9 after_year8

theorem phil_coin_collection :
  coin_collection 1000 (λ x => x * 4) (7 * 52) (3 * 182) (2 * 52) 
                  (λ x => x - (x * 2 / 5)) (5 * 91) (20 * 12) (10 * 52)
                  (λ x => x - (x / 3)) = 2816 := by
  sorry

end NUMINAMATH_CALUDE_phil_coin_collection_l1605_160524


namespace NUMINAMATH_CALUDE_clerts_in_120_degrees_proof_l1605_160598

/-- Represents the number of clerts in a full circle for Martian angle measurement -/
def clerts_in_full_circle : ℕ := 800

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / 360) * clerts_in_full_circle

/-- The number of clerts in a 120° angle -/
def clerts_in_120_degrees : ℕ := 267

theorem clerts_in_120_degrees_proof : 
  ⌊degrees_to_clerts 120⌋ = clerts_in_120_degrees :=
sorry

end NUMINAMATH_CALUDE_clerts_in_120_degrees_proof_l1605_160598


namespace NUMINAMATH_CALUDE_expression_evaluation_l1605_160589

theorem expression_evaluation :
  (18^40 : ℕ) / (54^20) * 2^10 = 2^30 * 3^20 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1605_160589


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l1605_160597

/-- Given a two-digit number, return its tens digit -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Given a two-digit number, return its ones digit -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := (tens_digit n) * (ones_digit n)

/-- The sum of digits of a two-digit number -/
def digit_sum (n : ℕ) : ℕ := (tens_digit n) + (ones_digit n)

theorem two_digit_number_theorem (x : ℕ) 
  (h1 : is_two_digit x)
  (h2 : digit_product (x + 46) = 6)
  (h3 : digit_sum x = 14) :
  x = 77 ∨ x = 86 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l1605_160597


namespace NUMINAMATH_CALUDE_total_houses_l1605_160500

theorem total_houses (dogs cats both : ℕ) 
  (h_dogs : dogs = 40)
  (h_cats : cats = 30)
  (h_both : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l1605_160500


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1605_160537

/-- Simple interest calculation --/
theorem simple_interest_principal (interest_rate : ℚ) (time_months : ℕ) (interest_earned : ℕ) (principal : ℕ) : 
  interest_rate = 50 / 3 → 
  time_months = 9 → 
  interest_earned = 8625 →
  principal = 69000 →
  interest_earned * 1200 = principal * interest_rate * time_months := by
  sorry

#check simple_interest_principal

end NUMINAMATH_CALUDE_simple_interest_principal_l1605_160537


namespace NUMINAMATH_CALUDE_triangle_area_is_integer_l1605_160523

theorem triangle_area_is_integer (a b c : ℝ) (h1 : a^2 = 377) (h2 : b^2 = 153) (h3 : c^2 = 80)
  (h4 : ∃ (w h : ℤ), ∃ (x y z : ℝ),
    (x^2 + y^2 = w^2) ∧ (x^2 + z^2 = h^2) ∧
    (y + z = a ∨ y + z = b ∨ y + z = c) ∧
    (∃ (d1 d2 : ℤ), d1 ≥ 0 ∧ d2 ≥ 0 ∧ d1 + d2 + x = w ∧ d1 + d2 + y = h)) :
  ∃ (A : ℤ), A = 42 ∧ 16 * A^2 = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_integer_l1605_160523


namespace NUMINAMATH_CALUDE_solution_equivalence_l1605_160599

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {((1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), Real.sqrt 3 / Real.rpow 6 (1/6)),
   (-(1 : ℝ) / Real.rpow 6 (1/6), Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6)),
   ((1 : ℝ) / Real.rpow 6 (1/6), -Real.sqrt 2 / Real.rpow 6 (1/6), -Real.sqrt 3 / Real.rpow 6 (1/6))}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^3 * y^3 * z^3 = 1 ∧ x * y^5 * z^3 = 2 ∧ x * y^3 * z^5 = 3

theorem solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l1605_160599


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1605_160514

/-- Given the conditions of a student's exam performance, 
    prove that the maximum marks are 500. -/
theorem exam_maximum_marks :
  let pass_percentage : ℚ := 33 / 100
  let student_marks : ℕ := 125
  let fail_margin : ℕ := 40
  ∃ (max_marks : ℕ), 
    (pass_percentage * max_marks : ℚ) = (student_marks + fail_margin : ℕ) ∧ 
    max_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1605_160514


namespace NUMINAMATH_CALUDE_tax_discount_commute_ana_equals_bob_miltonville_market_problem_l1605_160526

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Ana's total (tax then discount) --/
def ana_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Bob's total (discount then tax) --/
def bob_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Ana's total equals Bob's total --/
theorem ana_equals_bob (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  ana_total price tax_rate discount_rate = bob_total price tax_rate discount_rate := by
  sorry

/-- Specific case for the problem --/
theorem miltonville_market_problem :
  ana_total 120 0.08 0.25 = bob_total 120 0.08 0.25 := by
  sorry

end NUMINAMATH_CALUDE_tax_discount_commute_ana_equals_bob_miltonville_market_problem_l1605_160526


namespace NUMINAMATH_CALUDE_johns_remaining_money_l1605_160580

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after subtracting flight cost --/
def remaining_money (savings : ℕ) (flight_cost : ℕ) : ℕ :=
  octal_to_decimal savings - flight_cost

/-- Theorem stating that John's remaining money is 1725 in decimal --/
theorem johns_remaining_money :
  remaining_money 5555 1200 = 1725 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l1605_160580


namespace NUMINAMATH_CALUDE_orange_sum_l1605_160520

theorem orange_sum : 
  let tree1 : ℕ := 80
  let tree2 : ℕ := 60
  let tree3 : ℕ := 120
  let tree4 : ℕ := 45
  let tree5 : ℕ := 25
  let tree6 : ℕ := 97
  tree1 + tree2 + tree3 + tree4 + tree5 + tree6 = 427 := by
sorry

end NUMINAMATH_CALUDE_orange_sum_l1605_160520


namespace NUMINAMATH_CALUDE_total_chips_amount_l1605_160545

def person1_chips : ℕ := 350
def person2_chips : ℕ := 268
def person3_chips : ℕ := 182

theorem total_chips_amount : person1_chips + person2_chips + person3_chips = 800 := by
  sorry

end NUMINAMATH_CALUDE_total_chips_amount_l1605_160545


namespace NUMINAMATH_CALUDE_leo_weight_l1605_160539

/-- Given that Leo's weight plus 10 pounds is 1.5 times Kendra's weight,
    and that their combined weight is 180 pounds,
    prove that Leo's current weight is 104 pounds. -/
theorem leo_weight (leo kendra : ℝ) 
  (h1 : leo + 10 = 1.5 * kendra) 
  (h2 : leo + kendra = 180) : 
  leo = 104 := by sorry

end NUMINAMATH_CALUDE_leo_weight_l1605_160539


namespace NUMINAMATH_CALUDE_tan_sin_product_l1605_160533

theorem tan_sin_product (A B : Real) (hA : A = 10 * Real.pi / 180) (hB : B = 35 * Real.pi / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
    1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) + 
    Real.tan A * (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_product_l1605_160533


namespace NUMINAMATH_CALUDE_race_probability_l1605_160572

theorem race_probability (pA pB pC pD pE : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/8) 
  (hC : pC = 1/12) 
  (hD : pD = 1/20) 
  (hE : pE = 1/30) : 
  pA + pB + pC + pD + pE = 65/120 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l1605_160572


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1605_160515

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1605_160515


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l1605_160534

/-- Converts a length in inches to miles, given a scale where 1 inch represents 1000 feet --/
def inches_to_miles (inches : ℚ) : ℚ :=
  inches * 1000 / 5280

/-- Theorem stating that 7.5 inches on the given scale represents 125/88 miles --/
theorem scale_drawing_conversion :
  inches_to_miles (7.5) = 125 / 88 := by
  sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l1605_160534


namespace NUMINAMATH_CALUDE_center_line_perpendicular_iff_arithmetic_progression_l1605_160546

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The line passing through two points. -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The angle bisectors of a triangle. -/
def angle_bisectors (t : Triangle) : List (Set (ℝ × ℝ)) := sorry

/-- Two lines are perpendicular. -/
def perpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- The sides of a triangle form an arithmetic progression. -/
def arithmetic_progression (t : Triangle) : Prop :=
  t.a - t.b = t.b - t.c

theorem center_line_perpendicular_iff_arithmetic_progression (t : Triangle) :
  ∃ (bisector : Set (ℝ × ℝ)), bisector ∈ angle_bisectors t ∧
    perpendicular (line_through (incenter t) (circumcenter t)) bisector
  ↔ arithmetic_progression t := by sorry

end NUMINAMATH_CALUDE_center_line_perpendicular_iff_arithmetic_progression_l1605_160546


namespace NUMINAMATH_CALUDE_bus_passenger_count_l1605_160561

/-- Calculates the total number of passengers transported by a bus. -/
def totalPassengers (numTrips : ℕ) (initialPassengers : ℕ) (passengerDecrease : ℕ) : ℕ :=
  (numTrips * (2 * initialPassengers - (numTrips - 1) * passengerDecrease)) / 2

/-- Proves that the total number of passengers transported is 1854. -/
theorem bus_passenger_count : totalPassengers 18 120 2 = 1854 := by
  sorry

#eval totalPassengers 18 120 2

end NUMINAMATH_CALUDE_bus_passenger_count_l1605_160561


namespace NUMINAMATH_CALUDE_factories_unchecked_l1605_160510

/-- The number of unchecked factories given the total number of factories and the number checked by two groups -/
def unchecked_factories (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  total - (group1 + group2)

/-- Theorem stating that 67 factories remain unchecked -/
theorem factories_unchecked :
  unchecked_factories 259 105 87 = 67 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l1605_160510


namespace NUMINAMATH_CALUDE_function_intersection_l1605_160518

/-- Two functions f and g have exactly one common point if and only if (a-c)(b-d) = 2,
    given that they are centrally symmetric with respect to the point ((b+d)/2, a+c) -/
theorem function_intersection (a b c d : ℝ) :
  let f (x : ℝ) := 2*a + 1/(x-b)
  let g (x : ℝ) := 2*c + 1/(x-d)
  let center : ℝ × ℝ := ((b+d)/2, a+c)
  (∃! p, f p = g p ∧ 
   ∀ x y, f x = y ↔ g (b+d-x) = 2*(a+c)-y) ↔ 
  (a-c)*(b-d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_l1605_160518


namespace NUMINAMATH_CALUDE_complement_of_union_l1605_160547

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union (h1 : U = {1, 2, 3, 4, 5}) 
                            (h2 : A = {1, 2, 3}) 
                            (h3 : B = {3, 4}) : 
  U \ (A ∪ B) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1605_160547


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1605_160508

-- Define the solution set
def solution_set : Set ℝ := Set.Ioi 4 ∪ Set.Iic 1

-- Define the inequality
def inequality (a b x : ℝ) : Prop := (x - a) / (x - b) > 0

theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a b x) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1605_160508


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1605_160577

theorem matrix_equation_solution :
  ∀ (a b c d : ℝ),
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; b, 1]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![c, 2; 0, d]
  M * N = !![2, 4; -2, 0] →
  a = 1 ∧ b = -1 ∧ c = 2 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1605_160577


namespace NUMINAMATH_CALUDE_problem_solution_l1605_160579

/-- Given M = 2x + y, N = 2x - y, P = xy, M = 4, and N = 2, prove that P = 1.5 -/
theorem problem_solution (x y M N P : ℝ) 
  (hM : M = 2*x + y)
  (hN : N = 2*x - y)
  (hP : P = x*y)
  (hM_val : M = 4)
  (hN_val : N = 2) :
  P = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1605_160579


namespace NUMINAMATH_CALUDE_m_value_l1605_160551

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : a + b = 2) : 
  m = 100 := by
sorry

end NUMINAMATH_CALUDE_m_value_l1605_160551


namespace NUMINAMATH_CALUDE_beths_school_students_l1605_160548

theorem beths_school_students (beth paul : ℕ) : 
  beth = 4 * paul →  -- Beth's school has 4 times as many students as Paul's
  beth + paul = 5000 →  -- Total students in both schools is 5000
  beth = 4000 :=  -- Prove that Beth's school has 4000 students
by
  sorry

end NUMINAMATH_CALUDE_beths_school_students_l1605_160548


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l1605_160511

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l1605_160511


namespace NUMINAMATH_CALUDE_blue_marble_difference_is_twenty_l1605_160532

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue_marbles - tom_blue_marbles

theorem blue_marble_difference_is_twenty : blue_marble_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_marble_difference_is_twenty_l1605_160532


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1605_160549

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) →
  (a 2 + a 6 = 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1605_160549


namespace NUMINAMATH_CALUDE_diagonal_difference_l1605_160562

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The difference between the number of diagonals in an octagon and a heptagon -/
theorem diagonal_difference : num_diagonals 8 - num_diagonals 7 = 6 := by sorry

end NUMINAMATH_CALUDE_diagonal_difference_l1605_160562


namespace NUMINAMATH_CALUDE_white_balls_count_l1605_160566

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_or_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 8 →
  red = 9 →
  purple = 3 →
  prob_not_red_or_purple = 88/100 →
  total - (green + yellow + red + purple) = 50 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l1605_160566


namespace NUMINAMATH_CALUDE_lemons_per_glass_l1605_160571

/-- Given that 9 glasses of lemonade can be made with 18 lemons,
    prove that the number of lemons needed per glass is 2. -/
theorem lemons_per_glass (total_glasses : ℕ) (total_lemons : ℕ) 
  (h1 : total_glasses = 9) (h2 : total_lemons = 18) :
  total_lemons / total_glasses = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemons_per_glass_l1605_160571


namespace NUMINAMATH_CALUDE_initial_blue_balls_l1605_160559

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) (initial_blue : ℕ) : 
  total = 18 →
  removed = 3 →
  prob = 1 / 5 →
  (initial_blue - removed : ℚ) / (total - removed) = prob →
  initial_blue = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l1605_160559


namespace NUMINAMATH_CALUDE_polygon_sides_l1605_160550

/-- A polygon with side length 7 and perimeter 42 has 6 sides -/
theorem polygon_sides (side_length : ℕ) (perimeter : ℕ) (h1 : side_length = 7) (h2 : perimeter = 42) :
  perimeter / side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1605_160550


namespace NUMINAMATH_CALUDE_contest_finish_orders_l1605_160586

def number_of_participants : ℕ := 3

theorem contest_finish_orders :
  (Nat.factorial number_of_participants) = 6 := by
  sorry

end NUMINAMATH_CALUDE_contest_finish_orders_l1605_160586


namespace NUMINAMATH_CALUDE_share_in_ratio_l1605_160507

theorem share_in_ratio (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 4320) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  let sum_ratio := ratio1 + ratio2 + ratio3
  let share1 := total * ratio1 / sum_ratio
  share1 = 720 := by
sorry

end NUMINAMATH_CALUDE_share_in_ratio_l1605_160507


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1605_160556

/-- Given a quadratic equation 6x^2 + 5x + 7, prove that the sum of the reciprocals of its roots is -5/7 -/
theorem sum_of_reciprocals_of_roots (x : ℝ) (γ δ : ℝ) :
  (6 * x^2 + 5 * x + 7 = 0) →
  (∃ p q : ℝ, 6 * p^2 + 5 * p + 7 = 0 ∧ 6 * q^2 + 5 * q + 7 = 0 ∧ γ = 1/p ∧ δ = 1/q) →
  γ + δ = -5/7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1605_160556


namespace NUMINAMATH_CALUDE_fourth_year_afforestation_l1605_160512

/-- Calculates the area afforested in a given year, given the initial area and annual increase rate. -/
def area_afforested (initial_area : ℝ) (annual_increase : ℝ) (year : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ (year - 1)

/-- Theorem stating that given an initial area of 10,000 acres and an annual increase of 20%,
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  area_afforested 10000 0.2 4 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_afforestation_l1605_160512


namespace NUMINAMATH_CALUDE_product_three_consecutive_divisibility_l1605_160519

theorem product_three_consecutive_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 60 * m) :=
by sorry

end NUMINAMATH_CALUDE_product_three_consecutive_divisibility_l1605_160519


namespace NUMINAMATH_CALUDE_quadratic_properties_l1605_160593

/-- A quadratic function of the form y = -x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  (b = 4 ∧ c = 3 →
    (∃ (x y : ℝ), (x = 2 ∧ y = 7) ∧ 
      ∀ (t : ℝ), -1 ≤ t ∧ t ≤ 3 → 
        -2 ≤ quadratic_function b c t ∧ quadratic_function b c t ≤ 7)) ∧
  ((∀ (x : ℝ), x ≤ 0 → quadratic_function b c x ≤ 2) ∧
   (∀ (x : ℝ), x > 0 → quadratic_function b c x ≤ 3) ∧
   (∃ (x₁ x₂ : ℝ), x₁ ≤ 0 ∧ x₂ > 0 ∧ 
     quadratic_function b c x₁ = 2 ∧ quadratic_function b c x₂ = 3) →
    b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1605_160593


namespace NUMINAMATH_CALUDE_equation_solution_l1605_160503

theorem equation_solution : ∃! x : ℝ, x ≠ 1 ∧ (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1605_160503


namespace NUMINAMATH_CALUDE_config_7_3_1_wins_for_second_player_l1605_160504

/-- Represents the nim-value of a wall of bricks. -/
def nimValue (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | _ => 0  -- Default case, though not used in this problem

/-- Calculates the nim-sum (XOR) of a list of natural numbers. -/
def nimSum : List ℕ → ℕ
  | [] => 0
  | (x::xs) => x ^^^ (nimSum xs)

/-- Represents a configuration of walls in the game. -/
structure GameConfig where
  walls : List ℕ

/-- Determines if a given game configuration is a winning position for the second player. -/
def isWinningForSecondPlayer (config : GameConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The theorem stating that the configuration (7, 3, 1) is a winning position for the second player. -/
theorem config_7_3_1_wins_for_second_player :
  isWinningForSecondPlayer ⟨[7, 3, 1]⟩ := by sorry

end NUMINAMATH_CALUDE_config_7_3_1_wins_for_second_player_l1605_160504


namespace NUMINAMATH_CALUDE_range_m_for_always_negative_range_m_for_bounded_interval_l1605_160554

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Theorem 1
theorem range_m_for_always_negative (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Theorem 2
theorem range_m_for_bounded_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 5) ↔ m < 6/7 :=
sorry

end NUMINAMATH_CALUDE_range_m_for_always_negative_range_m_for_bounded_interval_l1605_160554


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_l1605_160502

theorem dividend_divisor_quotient (x y z : ℚ) :
  x = y * z + 15 ∧ y = 25 ∧ 3 * x - 4 * y + 2 * z = 0 →
  x = 230 / 7 ∧ z = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_l1605_160502


namespace NUMINAMATH_CALUDE_sqrt_37_between_6_and_7_l1605_160542

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_37_between_6_and_7_l1605_160542


namespace NUMINAMATH_CALUDE_power_function_value_l1605_160587

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f (1/2) = 8) : f 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1605_160587


namespace NUMINAMATH_CALUDE_norris_game_spending_l1605_160558

/-- The amount of money Norris spent on the online game -/
def money_spent (september_savings october_savings november_savings money_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - money_left

/-- Theorem stating that Norris spent $75 on the online game -/
theorem norris_game_spending :
  money_spent 29 25 31 10 = 75 := by
  sorry

end NUMINAMATH_CALUDE_norris_game_spending_l1605_160558


namespace NUMINAMATH_CALUDE_fencing_cost_l1605_160560

-- Define the ratio of sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 8112

-- Define the cost per meter in rupees
def cost_per_meter : ℚ := 25 / 100

-- Theorem statement
theorem fencing_cost :
  let x : ℚ := (area / (ratio_length * ratio_width)) ^ (1/2)
  let length : ℚ := ratio_length * x
  let width : ℚ := ratio_width * x
  let perimeter : ℚ := 2 * (length + width)
  let total_cost : ℚ := perimeter * cost_per_meter
  total_cost = 91 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_l1605_160560


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l1605_160530

-- Define the polynomial function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem polynomial_symmetry (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l1605_160530


namespace NUMINAMATH_CALUDE_inequality_solution_l1605_160543

theorem inequality_solution (x : ℝ) : 3 * x^2 - x < 8 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1605_160543


namespace NUMINAMATH_CALUDE_third_number_proof_l1605_160525

theorem third_number_proof (x : ℕ) : 
  let second := 3 * x - 7
  let third := 2 * x + 2
  x + second + third = 168 →
  third = 60 := by
sorry

end NUMINAMATH_CALUDE_third_number_proof_l1605_160525


namespace NUMINAMATH_CALUDE_smallest_angle_trig_equation_l1605_160528

theorem smallest_angle_trig_equation : 
  (∃ (x : ℝ), x > 0 ∧ x < 10 * π / 180 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x)) ∨
  (∀ (x : ℝ), x > 0 ∧ Real.sin (4*x) * Real.sin (5*x) = Real.cos (4*x) * Real.cos (5*x) → x ≥ 10 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_trig_equation_l1605_160528


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1605_160581

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.0000037 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.7 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1605_160581


namespace NUMINAMATH_CALUDE_factor_expression_l1605_160573

theorem factor_expression (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1605_160573


namespace NUMINAMATH_CALUDE_midpoint_locus_of_intersection_l1605_160591

/-- Given an arithmetic sequence A, B, C, this function represents the line Ax + By + C = 0 --/
def line (A B C : ℝ) (x y : ℝ) : Prop :=
  A * x + B * y + C = 0

/-- The parabola y = -2x^2 --/
def parabola (x y : ℝ) : Prop :=
  y = -2 * x^2

/-- The locus of the midpoint --/
def midpoint_locus (x y : ℝ) : Prop :=
  y + 1 = -(2 * x - 1)^2

/-- The main theorem --/
theorem midpoint_locus_of_intersection
  (A B C : ℝ) -- A, B, C are real numbers
  (h_arithmetic : A - 2*B + C = 0) -- A, B, C form an arithmetic sequence
  (x y : ℝ) -- x and y are real numbers
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line A B C x₁ y₁ ∧ parabola x₁ y₁ ∧
    line A B C x₂ y₂ ∧ parabola x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) -- (x, y) is the midpoint of the chord of intersection
  : midpoint_locus x y :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_of_intersection_l1605_160591


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1605_160517

theorem x_minus_y_value (x y : ℝ) (h : |x + y + 1| + Real.sqrt (2 * x - y) = 0) : 
  x - y = 1/3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1605_160517


namespace NUMINAMATH_CALUDE_triangle_angle_y_l1605_160541

theorem triangle_angle_y (y : ℝ) : 
  (45 : ℝ) + 3 * y + y = 180 → y = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_y_l1605_160541


namespace NUMINAMATH_CALUDE_no_unique_solution_l1605_160535

/-- 
Given a system of two linear equations:
  3(3x + 4y) = 36
  kx + cy = 30
where k = 9, prove that the system does not have a unique solution when c = 12.
-/
theorem no_unique_solution (x y : ℝ) : 
  (3 * (3 * x + 4 * y) = 36) →
  (9 * x + 12 * y = 30) →
  ¬ (∃! (x y : ℝ), 3 * (3 * x + 4 * y) = 36 ∧ 9 * x + 12 * y = 30) :=
by sorry

end NUMINAMATH_CALUDE_no_unique_solution_l1605_160535


namespace NUMINAMATH_CALUDE_fraction_equality_l1605_160509

theorem fraction_equality (a b : ℝ) (h : a / b = 4 / 3) :
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1605_160509


namespace NUMINAMATH_CALUDE_f_properties_l1605_160584

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) :
  (f 0 a b c) * (f 1 a b c) < 0 ∧ (f 0 a b c) * (f 3 a b c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1605_160584


namespace NUMINAMATH_CALUDE_interval_intersection_l1605_160557

theorem interval_intersection (x : ℝ) : 
  (|4 - x| < 5 ∧ x^2 < 36) ↔ (-1 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l1605_160557


namespace NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l1605_160544

theorem obtuse_triangle_from_altitudes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = 13) (h5 : b = 11) (h6 : c = 5) :
  (c^2 + b^2 - a^2) / (2 * b * c) < 0 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l1605_160544


namespace NUMINAMATH_CALUDE_boris_fudge_amount_l1605_160536

-- Define the conversion rate from pounds to ounces
def poundsToOunces (pounds : ℝ) : ℝ := pounds * 16

-- Define the amount of fudge eaten by each person
def tomasFudge : ℝ := 1.5
def katyaFudge : ℝ := 0.5

-- Define the total amount of fudge eaten by all three friends in ounces
def totalFudgeOunces : ℝ := 64

-- Theorem to prove
theorem boris_fudge_amount :
  let borisFudgeOunces := totalFudgeOunces - (poundsToOunces tomasFudge + poundsToOunces katyaFudge)
  borisFudgeOunces / 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boris_fudge_amount_l1605_160536


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l1605_160531

theorem fraction_ratio_equality : 
  (15 / 8) / (2 / 5) = (3 / 8) / (1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l1605_160531


namespace NUMINAMATH_CALUDE_projection_property_l1605_160505

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  let p := projection
  p (3, 3) = (27/5, 9/5) →
  p (1, -1) = (3/5, 1/5) := by sorry

end NUMINAMATH_CALUDE_projection_property_l1605_160505


namespace NUMINAMATH_CALUDE_problem_statement_l1605_160590

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_prod : a * b * c = 1)
  (h_eq1 : a + 1 / c = 8)
  (h_eq2 : b + 1 / a = 20) :
  c + 1 / b = 10 / 53 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1605_160590


namespace NUMINAMATH_CALUDE_max_profit_l1605_160568

/-- Annual sales revenue function -/
noncomputable def Q (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then -x^2 + 1040*x + 1200
  else if x > 30 then 998*x - 2048/(x-2) + 1800
  else 0

/-- Annual total profit function (in million yuan) -/
noncomputable def W (x : ℝ) : ℝ :=
  (Q x - (1000*x + 600)) / 1000

/-- The maximum profit is 1068 million yuan -/
theorem max_profit :
  ∃ x : ℝ, x > 0 ∧ W x = 1068 ∧ ∀ y : ℝ, y > 0 → W y ≤ W x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l1605_160568


namespace NUMINAMATH_CALUDE_no_such_hexagon_exists_l1605_160555

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the condition that all sides are greater than 1
def all_sides_greater_than_one (h : Hexagon) : Prop :=
  ∀ i : Fin 6, dist (h i) (h ((i + 1) % 6)) > 1

-- Define the condition that the distance from M to any vertex is less than 1
def all_vertices_less_than_one_from_point (h : Hexagon) (m : ℝ × ℝ) : Prop :=
  ∀ i : Fin 6, dist (h i) m < 1

-- The main theorem
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    all_sides_greater_than_one h ∧
    all_vertices_less_than_one_from_point h m :=
sorry

end NUMINAMATH_CALUDE_no_such_hexagon_exists_l1605_160555


namespace NUMINAMATH_CALUDE_min_sum_of_system_l1605_160521

theorem min_sum_of_system (x y z : ℝ) 
  (eq1 : x + 3*y + 6*z = 1)
  (eq2 : x*y + 2*x*z + 6*y*z = -8)
  (eq3 : x*y*z = 2) :
  ∀ (a b c : ℝ), (a + 3*b + 6*c = 1 ∧ a*b + 2*a*c + 6*b*c = -8 ∧ a*b*c = 2) → 
  x + y + z ≤ a + b + c ∧ x + y + z = -8/3 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_system_l1605_160521


namespace NUMINAMATH_CALUDE_jake_arrival_delay_l1605_160583

/-- Represents the problem of Austin and Jake descending a building --/
structure DescentProblem where
  floors : ℕ               -- Number of floors to descend
  steps_per_floor : ℕ      -- Number of steps per floor
  jake_speed : ℕ           -- Jake's speed in steps per second
  elevator_time : ℕ        -- Time taken by elevator in seconds

/-- Calculates the time difference between Jake's arrival and the elevator's arrival --/
def time_difference (p : DescentProblem) : ℤ :=
  let total_steps := p.floors * p.steps_per_floor
  let jake_time := total_steps / p.jake_speed
  jake_time - p.elevator_time

/-- The main theorem stating that Jake will arrive 20 seconds after the elevator --/
theorem jake_arrival_delay (p : DescentProblem) 
  (h1 : p.floors = 8)
  (h2 : p.steps_per_floor = 30)
  (h3 : p.jake_speed = 3)
  (h4 : p.elevator_time = 60) : 
  time_difference p = 20 := by
  sorry

#eval time_difference ⟨8, 30, 3, 60⟩

end NUMINAMATH_CALUDE_jake_arrival_delay_l1605_160583


namespace NUMINAMATH_CALUDE_not_right_triangle_l1605_160565

theorem not_right_triangle (a b c : ℕ) (h : a = 7 ∧ b = 9 ∧ c = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1605_160565


namespace NUMINAMATH_CALUDE_room_area_square_inches_l1605_160582

-- Define the conversion rate from feet to inches
def inches_per_foot : ℕ := 12

-- Define the side length of the room in feet
def room_side_feet : ℕ := 10

-- Theorem to prove the area of the room in square inches
theorem room_area_square_inches : 
  (room_side_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_room_area_square_inches_l1605_160582


namespace NUMINAMATH_CALUDE_G_initial_conditions_G_recurrence_G_20_diamonds_l1605_160576

/-- The number of diamonds in the n-th figure of sequence G -/
def G (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else 2 * n * (n + 1)

/-- The sequence G satisfies the given initial conditions -/
theorem G_initial_conditions :
  G 1 = 1 ∧ G 2 = 5 ∧ G 3 = 17 := by sorry

/-- The recurrence relation for G_n, n ≥ 3 -/
theorem G_recurrence (n : ℕ) (h : n ≥ 3) :
  G n = G (n-1) + 4 * n := by sorry

/-- The main theorem: G_20 has 840 diamonds -/
theorem G_20_diamonds : G 20 = 840 := by sorry

end NUMINAMATH_CALUDE_G_initial_conditions_G_recurrence_G_20_diamonds_l1605_160576


namespace NUMINAMATH_CALUDE_problem_solution_l1605_160540

theorem problem_solution : ∃! x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1605_160540


namespace NUMINAMATH_CALUDE_smallest_sum_l1605_160529

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Represents the problem setup -/
def ProblemSetup (x y m : ℕ) : Prop :=
  TwoDigitInt x ∧
  TwoDigitInt y ∧
  y = reverseDigits x ∧
  x^2 + y^2 = m^2 ∧
  ∃ k, x + y = 9 * (2 * k + 1)

theorem smallest_sum (x y m : ℕ) (h : ProblemSetup x y m) :
  x + y + m ≥ 169 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_l1605_160529


namespace NUMINAMATH_CALUDE_problem_solution_l1605_160574

theorem problem_solution : 
  (99^2 = 9801) ∧ 
  ((-8)^2009 * (-1/8)^2008 = -8) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1605_160574


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1605_160538

/-- The constant k in the inverse variation relationship -/
def k : ℝ := 192

/-- The relationship between z and x -/
def relation (z x : ℝ) : Prop := 3 * z = k / (x^3)

theorem inverse_variation_problem (z₁ z₂ x₁ x₂ : ℝ) 
  (h₁ : relation z₁ x₁)
  (h₂ : z₁ = 8)
  (h₃ : x₁ = 2)
  (h₄ : x₂ = 4) :
  z₂ = 1 ∧ relation z₂ x₂ := by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_problem_l1605_160538


namespace NUMINAMATH_CALUDE_compound_vs_simple_interest_amount_l1605_160552

/-- The amount of money (in rupees) that results in a difference of 8.000000000000227
    between 8% compound interest and 4% simple interest over 2 years -/
theorem compound_vs_simple_interest_amount : ℝ := by
  -- Define the compound interest rate
  let compound_rate : ℝ := 0.08
  -- Define the simple interest rate
  let simple_rate : ℝ := 0.04
  -- Define the time period in years
  let time : ℝ := 2
  -- Define the difference between compound and simple interest amounts
  let difference : ℝ := 8.000000000000227

  -- Define the function for compound interest
  let compound_interest (p : ℝ) : ℝ := p * (1 + compound_rate) ^ time

  -- Define the function for simple interest
  let simple_interest (p : ℝ) : ℝ := p * (1 + simple_rate * time)

  -- The amount p that satisfies the condition
  let p : ℝ := difference / (compound_interest 1 - simple_interest 1)

  -- Assert that p is approximately equal to 92.59
  sorry


end NUMINAMATH_CALUDE_compound_vs_simple_interest_amount_l1605_160552


namespace NUMINAMATH_CALUDE_mika_stickers_left_l1605_160595

/-- The number of stickers Mika has left after various changes -/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika has 2 stickers left -/
theorem mika_stickers_left :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_left_l1605_160595


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1605_160575

/-- Given that x varies inversely as the square of y, prove that x = 1/9 when y = 6,
    given that y = 2 when x = 1. -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / (y^2)) 
  (h2 : 1 = k / (2^2)) : 
  (y = 6) → (x = 1/9) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1605_160575


namespace NUMINAMATH_CALUDE_cyclists_distance_l1605_160594

theorem cyclists_distance (a b : ℝ) : 
  (a = b^2) ∧ (a - 1 = 3 * (b - 1)) → (a - b = 0 ∨ a - b = 2) :=
by sorry

end NUMINAMATH_CALUDE_cyclists_distance_l1605_160594


namespace NUMINAMATH_CALUDE_volume_of_region_l1605_160563

-- Define the region
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   |x - y + z| + |x - y - z| ≤ 10 ∧
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : 
  MeasureTheory.volume Region = 125 := by sorry

end NUMINAMATH_CALUDE_volume_of_region_l1605_160563


namespace NUMINAMATH_CALUDE_complex_cube_root_product_l1605_160527

theorem complex_cube_root_product (w : ℂ) (hw : w^3 = 1) :
  (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_product_l1605_160527


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_negative_three_l1605_160578

theorem tan_alpha_two_implies_fraction_equals_negative_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_negative_three_l1605_160578


namespace NUMINAMATH_CALUDE_hyperbola_focal_coordinates_l1605_160506

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The focal coordinates of the hyperbola -/
def focal_coordinates : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

/-- Theorem: The focal coordinates of the hyperbola x^2/16 - y^2/9 = 1 are (-5, 0) and (5, 0) -/
theorem hyperbola_focal_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ focal_coordinates :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_coordinates_l1605_160506


namespace NUMINAMATH_CALUDE_probability_not_greater_than_2_78_l1605_160513

def digits : Finset ℕ := {7, 1, 8}

def valid_combinations : Finset (ℕ × ℕ) :=
  {(1, 7), (1, 8), (7, 1), (7, 8)}

theorem probability_not_greater_than_2_78 :
  (Finset.card valid_combinations) / (Finset.card (digits.product digits)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_2_78_l1605_160513


namespace NUMINAMATH_CALUDE_stock_value_order_l1605_160553

/-- Represents the value of a stock over time -/
structure Stock :=
  (initial : ℝ)
  (first_year_change : ℝ)
  (second_year_change : ℝ)

/-- Calculates the final value of a stock after two years -/
def final_value (s : Stock) : ℝ :=
  s.initial * (1 + s.first_year_change) * (1 + s.second_year_change)

/-- The three stocks: Alabama Almonds (AA), Boston Beans (BB), and California Cauliflower (CC) -/
def AA : Stock := ⟨100, 0.2, -0.2⟩
def BB : Stock := ⟨100, -0.25, 0.25⟩
def CC : Stock := ⟨100, 0, 0⟩

theorem stock_value_order :
  final_value BB < final_value AA ∧ final_value AA < final_value CC :=
sorry

end NUMINAMATH_CALUDE_stock_value_order_l1605_160553


namespace NUMINAMATH_CALUDE_binomial_15_3_l1605_160569

theorem binomial_15_3 : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_3_l1605_160569


namespace NUMINAMATH_CALUDE_largest_integer_from_averages_l1605_160588

theorem largest_integer_from_averages : 
  ∀ w x y z : ℤ,
  (w + x + y) / 3 = 32 →
  (w + x + z) / 3 = 39 →
  (w + y + z) / 3 = 40 →
  (x + y + z) / 3 = 44 →
  max w (max x (max y z)) = 59 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_from_averages_l1605_160588


namespace NUMINAMATH_CALUDE_intersection_distance_zero_l1605_160564

/-- The distance between the intersection points of x^2 + y^2 = 18 and x + y = 6 is 0 -/
theorem intersection_distance_zero : 
  let S := {p : ℝ × ℝ | p.1^2 + p.2^2 = 18 ∧ p.1 + p.2 = 6}
  ∃! p : ℝ × ℝ, p ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_zero_l1605_160564


namespace NUMINAMATH_CALUDE_output_for_15_l1605_160596

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 25 then step1 + 10 else step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by sorry

end NUMINAMATH_CALUDE_output_for_15_l1605_160596


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1605_160501

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1605_160501


namespace NUMINAMATH_CALUDE_louie_last_match_goals_l1605_160570

/-- The number of goals Louie scored in the last match -/
def last_match_goals : ℕ := sorry

/-- The number of seasons Louie's brother has played -/
def brothers_seasons : ℕ := 3

/-- The number of games in each season -/
def games_per_season : ℕ := 50

/-- The total number of goals scored by both brothers -/
def total_goals : ℕ := 1244

/-- The number of goals Louie scored in previous matches -/
def previous_goals : ℕ := 40

theorem louie_last_match_goals : 
  last_match_goals = 4 ∧
  brothers_seasons * games_per_season * (2 * last_match_goals) + 
  previous_goals + last_match_goals = total_goals :=
by sorry

end NUMINAMATH_CALUDE_louie_last_match_goals_l1605_160570


namespace NUMINAMATH_CALUDE_blue_bicycle_selection_count_l1605_160585

/-- The number of ways to select at least two blue shared bicycles -/
def select_blue_bicycles : ℕ :=
  (Nat.choose 4 2 * Nat.choose 6 2) + (Nat.choose 4 3 * Nat.choose 6 1) + Nat.choose 4 4

/-- Theorem stating that the number of ways to select at least two blue shared bicycles is 115 -/
theorem blue_bicycle_selection_count :
  select_blue_bicycles = 115 := by sorry

end NUMINAMATH_CALUDE_blue_bicycle_selection_count_l1605_160585


namespace NUMINAMATH_CALUDE_evaluate_expression_l1605_160567

theorem evaluate_expression (b : ℝ) : 
  let x := 2 * b + 9
  x - 2 * b + 5 = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1605_160567
