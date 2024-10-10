import Mathlib

namespace two_equal_intercept_lines_l3164_316488

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The intercept of the line on both axes -/
  intercept : ℝ
  /-- The line passes through (2, 3) -/
  passes_through : intercept - 2 = 3 * (intercept - intercept) / intercept

/-- There are exactly two lines passing through (2, 3) with equal intercepts on both axes -/
theorem two_equal_intercept_lines : 
  ∃! (s : Finset EqualInterceptLine), s.card = 2 ∧ 
  (∀ l : EqualInterceptLine, l ∈ s) ∧
  (∀ l : EqualInterceptLine, l ∈ s → l.intercept ≠ 0) :=
sorry

end two_equal_intercept_lines_l3164_316488


namespace bedevir_will_participate_l3164_316455

/-- The combat skill of the n-th opponent -/
def opponent_skill (n : ℕ) : ℚ := 1 / (2^(n+1) - 1)

/-- The probability of Sir Bedevir winning against the n-th opponent -/
def win_probability (n : ℕ) : ℚ := 1 / (1 + opponent_skill n)

/-- Theorem: Sir Bedevir's probability of winning is greater than 1/2 for any opponent -/
theorem bedevir_will_participate (k : ℕ) (h : k > 1) :
  ∀ n, n < k → win_probability n > 1/2 := by sorry

end bedevir_will_participate_l3164_316455


namespace expression_evaluation_l3164_316420

theorem expression_evaluation (a b : ℚ) (h1 : a = 1) (h2 : b = 1/2) :
  a * (a - 2*b) + (a + b) * (a - b) + (a - b)^2 = 1 := by
  sorry

end expression_evaluation_l3164_316420


namespace distribute_6_3_l3164_316428

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_6_3 : distribute 6 3 = 729 := by
  sorry

end distribute_6_3_l3164_316428


namespace lcm_5_7_10_14_l3164_316414

theorem lcm_5_7_10_14 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 14)) = 70 := by
  sorry

end lcm_5_7_10_14_l3164_316414


namespace carpooling_distance_ratio_l3164_316465

/-- Proves that the ratio of the distance driven between the second friend's house and work
    to the total distance driven to the first and second friend's houses is 3:1 -/
theorem carpooling_distance_ratio :
  let distance_to_first : ℝ := 8
  let distance_to_second : ℝ := distance_to_first / 2
  let distance_to_work : ℝ := 36
  let total_distance_to_friends : ℝ := distance_to_first + distance_to_second
  (distance_to_work / total_distance_to_friends) = 3
  := by sorry

end carpooling_distance_ratio_l3164_316465


namespace spade_calculation_l3164_316418

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 3 (spade 4 5) = -72 := by
  sorry

end spade_calculation_l3164_316418


namespace sunzi_wood_measurement_l3164_316438

/-- Given a piece of wood and a rope with unknown lengths, prove that the system of equations
    describing their relationship is correct based on the given measurements. -/
theorem sunzi_wood_measurement (x y : ℝ) : 
  (y - x = 4.5 ∧ y > x) →  -- Full rope measurement
  (x - y / 2 = 1 ∧ x > y / 2) →  -- Half rope measurement
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end sunzi_wood_measurement_l3164_316438


namespace jake_arrival_time_l3164_316446

-- Define the problem parameters
def floors : ℕ := 9
def steps_per_floor : ℕ := 30
def jake_steps_per_second : ℕ := 3
def elevator_time : ℕ := 60  -- in seconds

-- Calculate the total number of steps
def total_steps : ℕ := floors * steps_per_floor

-- Calculate Jake's descent time
def jake_time : ℕ := total_steps / jake_steps_per_second

-- Define the theorem
theorem jake_arrival_time :
  jake_time - elevator_time = 30 := by sorry

end jake_arrival_time_l3164_316446


namespace camp_children_count_l3164_316425

/-- Represents the number of children currently in the camp -/
def current_children : ℕ := sorry

/-- Represents the fraction of boys in the camp -/
def boy_fraction : ℚ := 9/10

/-- Represents the fraction of girls in the camp -/
def girl_fraction : ℚ := 1 - boy_fraction

/-- Represents the desired fraction of girls after adding more boys -/
def desired_girl_fraction : ℚ := 1/20

/-- Represents the number of additional boys to be added -/
def additional_boys : ℕ := 60

/-- Theorem stating that the current number of children in the camp is 60 -/
theorem camp_children_count : current_children = 60 := by
  sorry

end camp_children_count_l3164_316425


namespace puzzle_solution_l3164_316493

theorem puzzle_solution (triangle square : ℤ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 := by
sorry

end puzzle_solution_l3164_316493


namespace multiply_by_eleven_l3164_316480

theorem multiply_by_eleven (A B : Nat) (h1 : A < 10) (h2 : B < 10) (h3 : A + B < 10) :
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B := by
  sorry

end multiply_by_eleven_l3164_316480


namespace xiaolongs_dad_age_l3164_316496

theorem xiaolongs_dad_age (xiaolong_age : ℕ) : 
  xiaolong_age > 0 →
  (9 * xiaolong_age = 9 * xiaolong_age) →  -- Mom's age this year
  (9 * xiaolong_age + 3 = 9 * xiaolong_age + 3) →  -- Dad's age this year
  (9 * xiaolong_age + 4 = 8 * (xiaolong_age + 1)) →  -- Dad's age next year = 8 * Xiaolong's age next year
  9 * xiaolong_age + 3 = 39 := by
sorry

end xiaolongs_dad_age_l3164_316496


namespace min_value_x_plus_y_l3164_316442

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 4/y₀ = 1 ∧ x₀ + y₀ = 9 :=
sorry

end min_value_x_plus_y_l3164_316442


namespace vacation_miles_per_day_l3164_316482

theorem vacation_miles_per_day 
  (vacation_days : ℝ) 
  (total_miles : ℝ) 
  (h1 : vacation_days = 5.0) 
  (h2 : total_miles = 1250) : 
  total_miles / vacation_days = 250 := by
sorry

end vacation_miles_per_day_l3164_316482


namespace arrangement_counts_l3164_316448

/-- The number of singing programs -/
def num_singing : ℕ := 5

/-- The number of dance programs -/
def num_dance : ℕ := 4

/-- The total number of programs -/
def total_programs : ℕ := num_singing + num_dance

/-- Calculates the number of permutations of n items taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where no two dance programs are adjacent -/
def non_adjacent_arrangements : ℕ :=
  permutations num_singing num_singing * permutations (num_singing + 1) num_dance

/-- The number of arrangements with alternating singing and dance programs -/
def alternating_arrangements : ℕ :=
  permutations num_singing num_singing * permutations num_dance num_dance

theorem arrangement_counts :
  non_adjacent_arrangements = 43200 ∧ alternating_arrangements = 2880 := by
  sorry

end arrangement_counts_l3164_316448


namespace range_of_x_when_m_is_two_range_of_m_given_inequality_l3164_316470

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Theorem 1
theorem range_of_x_when_m_is_two :
  ∀ x : ℝ, f 2 x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Theorem 2
theorem range_of_m_given_inequality :
  (∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) → -8 ≤ m ∧ m ≤ 6 := by sorry

end range_of_x_when_m_is_two_range_of_m_given_inequality_l3164_316470


namespace vector_operation_l3164_316497

/-- Given two vectors a and b in R², prove that 2a - b equals (0,5) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, -1)) :
  (2 : ℝ) • a - b = (0, 5) := by sorry

end vector_operation_l3164_316497


namespace max_cards_from_poster_board_l3164_316485

/-- Represents the dimensions of a rectangular object in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of small rectangles that can fit into a larger square -/
def maxRectangles (square_side : ℕ) (card : Dimensions) : ℕ :=
  (square_side / card.length) * (square_side / card.width)

theorem max_cards_from_poster_board :
  let poster_board_side : ℕ := 12  -- 1 foot = 12 inches
  let card : Dimensions := { length := 2, width := 3 }
  maxRectangles poster_board_side card = 24 := by
sorry

end max_cards_from_poster_board_l3164_316485


namespace divisibility_property_implies_factor_of_99_l3164_316432

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_property_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  99 ∣ k :=
by sorry

end divisibility_property_implies_factor_of_99_l3164_316432


namespace unique_perpendicular_line_l3164_316461

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a plane type
structure Plane where
  points : Set Point

-- Define what it means for a point to be on a line
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be perpendicular
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (l : Line) (p : Point) 
  (h : ¬ p.onLine l) : 
  ∃! l_perp : Line, 
    l_perp.perpendicular l ∧ 
    p.onLine l_perp :=
  sorry

end unique_perpendicular_line_l3164_316461


namespace ending_number_of_range_problem_solution_l3164_316463

theorem ending_number_of_range (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_multiple := ((start + divisor - 1) / divisor) * divisor
  first_multiple + (count - 1) * divisor

theorem problem_solution : 
  ending_number_of_range 49 3 11 = 77 := by
  sorry

end ending_number_of_range_problem_solution_l3164_316463


namespace expression_evaluation_l3164_316489

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end expression_evaluation_l3164_316489


namespace expression_simplification_l3164_316402

theorem expression_simplification :
  80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end expression_simplification_l3164_316402


namespace even_product_probability_l3164_316464

-- Define the spinners
def spinner1 : List ℕ := [2, 4, 6, 8]
def spinner2 : List ℕ := [1, 3, 5, 7, 9]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability function
def probabilityEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := (s1.length * s2.length : ℚ)
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probabilityEvenProduct spinner1 spinner2 = 1 := by
  sorry

end even_product_probability_l3164_316464


namespace pqr_value_l3164_316443

theorem pqr_value (p q r : ℂ) 
  (eq1 : p * q + 5 * q = -20)
  (eq2 : q * r + 5 * r = -20)
  (eq3 : r * p + 5 * p = -20) : 
  p * q * r = 80 := by
sorry

end pqr_value_l3164_316443


namespace degree_relation_degree_bound_a2_zero_l3164_316487

/-- A real polynomial -/
def RealPolynomial := ℝ → ℝ

/-- The degree of a polynomial -/
noncomputable def degree (p : RealPolynomial) : ℕ := sorry

/-- Theorem for part (a) -/
theorem degree_relation (p : RealPolynomial) (h : degree p > 2) :
  degree p = 2 + degree (fun x => p (x + 1) + p (x - 1) - 2 * p x) := by sorry

/-- Theorem for part (b) -/
theorem degree_bound (p : RealPolynomial) (r s : ℝ)
  (h : ∀ x : ℝ, p (x + 1) + p (x - 1) - r * p x - s = 0) :
  degree p ≤ 2 := by sorry

/-- Theorem for part (c) -/
theorem a2_zero (p : RealPolynomial) (r : ℝ)
  (h : ∀ x : ℝ, p (x + 1) + p (x - 1) - r * p x = 0) :
  ∃ a₀ a₁, p = fun x => a₁ * x + a₀ := by sorry

end degree_relation_degree_bound_a2_zero_l3164_316487


namespace boxes_delivered_to_orphanage_l3164_316412

def total_lemon_cupcakes : ℕ := 53
def total_chocolate_cupcakes : ℕ := 76
def lemon_cupcakes_left_at_home : ℕ := 7
def chocolate_cupcakes_left_at_home : ℕ := 8
def cupcakes_per_box : ℕ := 5

def lemon_cupcakes_delivered : ℕ := total_lemon_cupcakes - lemon_cupcakes_left_at_home
def chocolate_cupcakes_delivered : ℕ := total_chocolate_cupcakes - chocolate_cupcakes_left_at_home

def total_cupcakes_delivered : ℕ := lemon_cupcakes_delivered + chocolate_cupcakes_delivered

theorem boxes_delivered_to_orphanage :
  (total_cupcakes_delivered / cupcakes_per_box : ℕ) +
  (if total_cupcakes_delivered % cupcakes_per_box > 0 then 1 else 0) = 23 :=
by sorry

end boxes_delivered_to_orphanage_l3164_316412


namespace min_product_of_reciprocal_sum_min_product_exists_l3164_316426

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 → x * y ≥ 98 := by
  sorry

theorem min_product_exists : 
  ∃ (x y : ℕ+), (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 7 ∧ x * y = 98 := by
  sorry

end min_product_of_reciprocal_sum_min_product_exists_l3164_316426


namespace f_inequality_l3164_316421

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition for f
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem f_inequality (h : strictly_increasing f) : f (-2) < f 1 ∧ f 1 < f 3 := by
  sorry

end f_inequality_l3164_316421


namespace problem_solution_l3164_316404

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end problem_solution_l3164_316404


namespace second_term_value_l3164_316483

/-- A geometric sequence with sum of first n terms Sn = a·3^n - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := sorry

/-- Sum of first n terms of the geometric sequence -/
def Sn (a : ℝ) (n : ℕ) : ℝ := a * 3^n - 2

/-- The second term of the sequence -/
def a2 (a : ℝ) : ℝ := GeometricSequence a 2

theorem second_term_value (a : ℝ) :
  a2 a = 12 :=
sorry

end second_term_value_l3164_316483


namespace tangent_line_at_one_l3164_316451

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x - 2*x + 1

theorem tangent_line_at_one (x y : ℝ) :
  let f' : ℝ → ℝ := λ t => 2*t * Real.log t + t - 2
  (x + y = 0) ↔ (y - f 1 = f' 1 * (x - 1)) :=
sorry

end tangent_line_at_one_l3164_316451


namespace age_sum_proof_l3164_316453

-- Define the son's current age
def son_age : ℕ := 36

-- Define the father's current age
def father_age : ℕ := 72

-- Theorem stating the conditions and the result to prove
theorem age_sum_proof :
  -- 18 years ago, father was 3 times as old as son
  (father_age - 18 = 3 * (son_age - 18)) ∧
  -- Now, father is twice as old as son
  (father_age = 2 * son_age) →
  -- The sum of their present ages is 108
  son_age + father_age = 108 := by
  sorry

end age_sum_proof_l3164_316453


namespace smallest_number_of_students_l3164_316445

/-- Represents the number of students in each grade --/
structure Students where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- The ratio of 8th-graders to 6th-graders is 5:3 --/
def ratio_8th_to_6th (s : Students) : Prop :=
  5 * s.sixth = 3 * s.eighth

/-- The ratio of 8th-graders to 7th-graders is 8:5 --/
def ratio_8th_to_7th (s : Students) : Prop :=
  8 * s.seventh = 5 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.eighth + s.seventh + s.sixth

/-- The main theorem: The smallest possible number of students is 89 --/
theorem smallest_number_of_students :
  ∃ (s : Students), ratio_8th_to_6th s ∧ ratio_8th_to_7th s ∧
  (∀ (t : Students), ratio_8th_to_6th t ∧ ratio_8th_to_7th t →
    total_students s ≤ total_students t) ∧
  total_students s = 89 := by
  sorry

end smallest_number_of_students_l3164_316445


namespace repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l3164_316413

/-- Represents a repeating decimal with a 3-digit repetend -/
def RepeatingDecimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_417_equals_fraction :
  RepeatingDecimal 4 1 7 = 46 / 111 := by sorry

theorem sum_of_numerator_and_denominator :
  46 + 111 = 157 := by sorry

end repeating_decimal_417_equals_fraction_sum_of_numerator_and_denominator_l3164_316413


namespace smallest_number_with_conditions_l3164_316490

/-- A function that checks if a natural number consists only of 2's and 7's in its decimal representation -/
def only_2_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

/-- A function that checks if a natural number has at least one 2 and one 7 in its decimal representation -/
def has_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

/-- The theorem stating the properties of the smallest number satisfying the given conditions -/
theorem smallest_number_with_conditions (m : ℕ) : 
  (∀ n : ℕ, n < m → ¬(n % 5 = 0 ∧ n % 8 = 0 ∧ only_2_and_7 n ∧ has_2_and_7 n)) →
  m % 5 = 0 ∧ m % 8 = 0 ∧ only_2_and_7 m ∧ has_2_and_7 m →
  m % 10000 = 7272 :=
by sorry

end smallest_number_with_conditions_l3164_316490


namespace pi_minus_2023_power_0_minus_one_third_power_neg_2_l3164_316467

theorem pi_minus_2023_power_0_minus_one_third_power_neg_2 :
  (π - 2023) ^ (0 : ℝ) - (1 / 3 : ℝ) ^ (-2 : ℝ) = -8 := by sorry

end pi_minus_2023_power_0_minus_one_third_power_neg_2_l3164_316467


namespace max_value_of_function_l3164_316462

theorem max_value_of_function (x : ℝ) (hx : x < 0) : 
  2 * x + 2 / x ≤ -4 ∧ 
  ∃ y : ℝ, y < 0 ∧ 2 * y + 2 / y = -4 :=
by sorry

end max_value_of_function_l3164_316462


namespace x_power_twelve_l3164_316471

theorem x_power_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 + 1/x^12 = 103682 := by
  sorry

end x_power_twelve_l3164_316471


namespace cubic_function_extreme_points_l3164_316423

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Predicate stating that f has exactly two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0 ∧
  ∀ z : ℝ, f' a z = 0 → z = x ∨ z = y

theorem cubic_function_extreme_points (a : ℝ) :
  has_two_extreme_points a → a < 0 :=
sorry

end cubic_function_extreme_points_l3164_316423


namespace always_positive_l3164_316458

theorem always_positive (x : ℝ) : x^2 + |x| + 1 > 0 := by
  sorry

end always_positive_l3164_316458


namespace january_salary_l3164_316419

/-- Represents the monthly salary structure --/
structure MonthlySalary where
  january : ℝ
  february : ℝ
  march : ℝ
  april : ℝ
  may : ℝ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : MonthlySalary) 
  (h1 : (s.january + s.february + s.march + s.april) / 4 = 8000)
  (h2 : (s.february + s.march + s.april + s.may) / 4 = 8450)
  (h3 : s.may = 6500) :
  s.january = 4700 := by
sorry

end january_salary_l3164_316419


namespace union_of_A_and_B_l3164_316411

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B : Set ℝ := {x | x^3 = x}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end union_of_A_and_B_l3164_316411


namespace negative_integer_equation_solution_l3164_316424

theorem negative_integer_equation_solution :
  ∀ N : ℤ, (N < 0) → (2 * N^2 + N = 15) → (N = -3) := by
  sorry

end negative_integer_equation_solution_l3164_316424


namespace cost_of_goods_l3164_316474

/-- The cost of goods problem -/
theorem cost_of_goods
  (mango_rice_ratio : ℝ)
  (flour_rice_ratio : ℝ)
  (flour_cost : ℝ)
  (h1 : 10 * mango_rice_ratio = 24)
  (h2 : 6 * flour_rice_ratio = 2)
  (h3 : flour_cost = 21) :
  4 * (24 / 10 * (2 / 6 * flour_cost)) + 3 * (2 / 6 * flour_cost) + 5 * flour_cost = 898.80 :=
by sorry

end cost_of_goods_l3164_316474


namespace negative_four_cubed_equality_l3164_316444

theorem negative_four_cubed_equality : (-4)^3 = -4^3 := by
  sorry

end negative_four_cubed_equality_l3164_316444


namespace book_club_single_people_count_l3164_316415

/-- Represents a book club with members and book selection turns. -/
structure BookClub where
  total_turns : ℕ  -- Total number of turns per year
  couple_count : ℕ  -- Number of couples in the club
  ron_turns : ℕ  -- Number of turns Ron gets per year

/-- Calculates the number of single people in the book club. -/
def single_people_count (club : BookClub) : ℕ :=
  club.total_turns - (club.couple_count + 1)

/-- Theorem stating that the number of single people in the given book club is 9. -/
theorem book_club_single_people_count :
  ∃ (club : BookClub),
    club.total_turns = 52 / 4 ∧
    club.couple_count = 3 ∧
    club.ron_turns = 4 ∧
    single_people_count club = 9 := by
  sorry

end book_club_single_people_count_l3164_316415


namespace expression_percentage_of_y_l3164_316498

theorem expression_percentage_of_y (y z : ℝ) (hy : y > 0) :
  ((2 * y + z) / 10 + (3 * y - z) / 10) / y = 1 / 2 := by
  sorry

end expression_percentage_of_y_l3164_316498


namespace unique_solution_equation_l3164_316400

theorem unique_solution_equation : ∃! x : ℝ, 3 * x - 8 - 2 = x := by sorry

end unique_solution_equation_l3164_316400


namespace expression_simplification_and_evaluation_l3164_316456

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 :=
by sorry

end expression_simplification_and_evaluation_l3164_316456


namespace equal_area_rectangles_l3164_316452

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles (carol jordan : Rectangle) 
  (h1 : carol.length = 15)
  (h2 : carol.width = 20)
  (h3 : jordan.length = 6)
  (h4 : area carol = area jordan) :
  jordan.width = 50 := by
  sorry

end equal_area_rectangles_l3164_316452


namespace quadratic_range_l3164_316447

theorem quadratic_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_range_l3164_316447


namespace division_remainder_l3164_316479

theorem division_remainder : 
  let dividend : ℕ := 220020
  let divisor : ℕ := 555 + 445
  let quotient : ℕ := 2 * (555 - 445)
  dividend % divisor = 20 :=
by
  sorry

end division_remainder_l3164_316479


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l3164_316472

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  ∃ (q : ℕ), 41 = 17 * q + 7 :=
by
  sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l3164_316472


namespace gcd_of_polynomials_l3164_316495

def is_even_multiple_of_5959 (b : ℤ) : Prop :=
  ∃ k : ℤ, b = 2 * 5959 * k

theorem gcd_of_polynomials (b : ℤ) (h : is_even_multiple_of_5959 b) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 := by
  sorry

end gcd_of_polynomials_l3164_316495


namespace max_value_x_1_minus_3x_l3164_316440

theorem max_value_x_1_minus_3x (x : ℝ) (h : 0 < x ∧ x < 1/3) :
  ∃ (max : ℝ), max = 1/12 ∧ ∀ y, 0 < y ∧ y < 1/3 → x * (1 - 3*x) ≤ max := by
  sorry

end max_value_x_1_minus_3x_l3164_316440


namespace hyperbola_focus_range_l3164_316494

theorem hyperbola_focus_range (a b : ℝ) : 
  a > 0 → 
  b > 0 → 
  a^2 + b^2 = 16 → 
  b^2 ≥ 3 * a^2 → 
  0 < a ∧ a ≤ 2 := by
  sorry

end hyperbola_focus_range_l3164_316494


namespace f_condition_iff_sum_less_two_l3164_316486

open Real

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The first condition: f(x) = f(2-x) -/
axiom f_symmetry (x : ℝ) : f x = f (2 - x)

/-- The second condition: f'(x)(x-1) > 0 -/
axiom f_derivative_condition (x : ℝ) : deriv f x * (x - 1) > 0

/-- The main theorem -/
theorem f_condition_iff_sum_less_two (x₁ x₂ : ℝ) (h : x₁ < x₂) :
  f x₁ > f x₂ ↔ x₁ + x₂ < 2 :=
sorry

end f_condition_iff_sum_less_two_l3164_316486


namespace quadratic_form_k_value_l3164_316408

theorem quadratic_form_k_value :
  ∀ (a h k : ℝ),
  (∀ x, x^2 - 7*x + 1 = a*(x - h)^2 + k) →
  k = -45/4 := by
sorry

end quadratic_form_k_value_l3164_316408


namespace point_coordinates_l3164_316407

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (M : Point) 
  (h1 : fourth_quadrant M)
  (h2 : distance_to_x_axis M = 3)
  (h3 : distance_to_y_axis M = 4) :
  M.x = 4 ∧ M.y = -3 := by
  sorry

end point_coordinates_l3164_316407


namespace duck_ratio_l3164_316409

theorem duck_ratio (total_birds : ℕ) (chicken_feed_cost : ℚ) (total_chicken_feed_cost : ℚ) :
  total_birds = 15 →
  chicken_feed_cost = 2 →
  total_chicken_feed_cost = 20 →
  (total_birds - (total_chicken_feed_cost / chicken_feed_cost)) / total_birds = 1/3 := by
  sorry

end duck_ratio_l3164_316409


namespace max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l3164_316436

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a line passing through F₁
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points P and Q
def Intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ Ellipse x y ∧ Line k x y}

-- State the theorems
theorem max_distance_F₂_to_l :
  ∃ (k : ℝ), ∀ (l : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), l x y ↔ Line k x y) →
    (∃ (d : ℝ), d = 4 ∧ ∀ (p : ℝ × ℝ), l p.1 p.2 → dist F₂ p ≤ d) :=
sorry

theorem max_value_PF₂_QF₂ :
  ∃ (P Q : ℝ × ℝ), P ∈ Intersection k ∧ Q ∈ Intersection k ∧
    ∀ (P' Q' : ℝ × ℝ), P' ∈ Intersection k → Q' ∈ Intersection k →
      dist P' F₂ + dist Q' F₂ ≤ 26/3 :=
sorry

theorem range_F₁P_dot_F₁Q :
  ∀ (k : ℝ), ∀ (P Q : ℝ × ℝ),
    P ∈ Intersection k → Q ∈ Intersection k →
    -5 ≤ (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ∧
    (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) ≤ -25/9 :=
sorry

end max_distance_F₂_to_l_max_value_PF₂_QF₂_range_F₁P_dot_F₁Q_l3164_316436


namespace inequality_solution_1_inequality_solution_2_inequality_solution_3_l3164_316491

-- Define the functions for each inequality
def f₁ (x : ℝ) := (x - 2)^11 * (x + 1)^22 * (x + 3)^33
def f₂ (x : ℝ) := (4*x + 3)^5 * (3*x + 2)^3 * (2*x + 1)
def f₃ (x : ℝ) := (x + 3) * (x + 1)^2 * (x - 2)^3 * (x - 4)

-- Define the solution sets
def S₁ : Set ℝ := {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo (-1) 2)}
def S₂ : Set ℝ := {x | x ∈ (Set.Iic (-3/4)) ∪ (Set.Icc (-2/3) (-1/2))}
def S₃ : Set ℝ := {x | x ∈ (Set.Iic (-3)) ∪ {-1} ∪ (Set.Icc 2 4)}

-- State the theorems
theorem inequality_solution_1 : {x : ℝ | f₁ x < 0} = S₁ := by sorry

theorem inequality_solution_2 : {x : ℝ | f₂ x ≤ 0} = S₂ := by sorry

theorem inequality_solution_3 : {x : ℝ | f₃ x ≤ 0} = S₃ := by sorry

end inequality_solution_1_inequality_solution_2_inequality_solution_3_l3164_316491


namespace rational_roots_quadratic_l3164_316468

theorem rational_roots_quadratic (m : ℤ) :
  (∃ x y : ℚ, m * x^2 - (m - 1) * x + 1 = 0 ∧ m * y^2 - (m - 1) * y + 1 = 0 ∧ x ≠ y) →
  m = 6 ∧ (1/2 : ℚ) * m - (m - 1) * (1/2) + 1 = 0 ∧ (1/3 : ℚ) * m - (m - 1) * (1/3) + 1 = 0 :=
by sorry

end rational_roots_quadratic_l3164_316468


namespace inequality_proof_l3164_316473

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end inequality_proof_l3164_316473


namespace divisible_by_132_iff_in_list_l3164_316427

def is_valid_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = 1000 * x + 100 * y + 90 + z

theorem divisible_by_132_iff_in_list (n : ℕ) :
  is_valid_number n ∧ n % 132 = 0 ↔ n ∈ [3696, 4092, 6996, 7392] := by
  sorry

end divisible_by_132_iff_in_list_l3164_316427


namespace min_a_for_f_nonpositive_l3164_316478

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_f_nonpositive :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), b < 1 - 1/Real.exp 1 → ∃ (x : ℝ), x ≥ -2 ∧ f b x > 0) :=
sorry

end min_a_for_f_nonpositive_l3164_316478


namespace fraction_sum_inequality_l3164_316449

theorem fraction_sum_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  b / a + a / b > 2 := by sorry

end fraction_sum_inequality_l3164_316449


namespace floor_abs_negative_real_l3164_316416

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end floor_abs_negative_real_l3164_316416


namespace expression_simplification_l3164_316401

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 5) * 4 + (5 - 2 / 4) * (8 * p - 12) = 4 * p - 42 := by
  sorry

end expression_simplification_l3164_316401


namespace root_square_transformation_l3164_316410

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

/-- The resulting polynomial g(x) -/
def g (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 16

theorem root_square_transformation (r : ℝ) : 
  f r = 0 → ∃ s, g s = 0 ∧ s = r^2 := by sorry

end root_square_transformation_l3164_316410


namespace bisecting_line_sum_of_squares_l3164_316431

/-- A line with slope 4 that bisects a 3x3 unit square into two equal areas -/
def bisecting_line (a b c : ℝ) : Prop :=
  -- The line has slope 4
  a / b = 4 ∧
  -- The line equation is of the form ax = by + c
  ∀ x y, a * x = b * y + c ↔ y = 4 * x ∧
  -- The line bisects the square into two equal areas
  ∃ x₁ y₁ x₂ y₂, 
    0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 ∧
    0 ≤ y₁ ∧ y₁ < y₂ ∧ y₂ ≤ 3 ∧
    a * x₁ = b * y₁ + c ∧
    a * x₂ = b * y₂ + c ∧
    (3 * y₁ + (3 - y₂) * 3) / 2 = 9 / 2

theorem bisecting_line_sum_of_squares (a b c : ℝ) :
  bisecting_line a b c → a^2 + b^2 + c^2 = 17 := by
  sorry

end bisecting_line_sum_of_squares_l3164_316431


namespace quadratic_completion_of_square_l3164_316441

theorem quadratic_completion_of_square (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 19 = (x + n)^2 - 6) → b > 0 → b = 10 := by
  sorry

end quadratic_completion_of_square_l3164_316441


namespace chi_square_relationship_certainty_l3164_316454

-- Define the Chi-square test result
def chi_square_result : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level
def critical_value : ℝ := 6.635

-- Define the certainty level
def certainty_level : ℝ := 0.99

-- Theorem statement
theorem chi_square_relationship_certainty :
  chi_square_result > critical_value →
  certainty_level = 0.99 :=
sorry

end chi_square_relationship_certainty_l3164_316454


namespace binomial_coefficient_divisibility_binomial_coefficient_extremes_l3164_316405

theorem binomial_coefficient_divisibility (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  ∃ m : ℕ, (Nat.choose p k) = m * p :=
sorry

theorem binomial_coefficient_extremes (p : ℕ) (hp : Nat.Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end binomial_coefficient_divisibility_binomial_coefficient_extremes_l3164_316405


namespace inverse_proportion_increasing_l3164_316484

/-- For an inverse proportion function y = (m-5)/x, if y increases as x increases on each branch of its graph, then m < 5 -/
theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ 0 → x₂ ≠ 0 → x₁ < x₂ → (m - 5) / x₁ < (m - 5) / x₂) → 
  m < 5 :=
sorry

end inverse_proportion_increasing_l3164_316484


namespace perpendicular_vectors_x_value_l3164_316430

/-- Given two perpendicular vectors a = (3, -1) and b = (x, -2), prove that x = -2/3 -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = -2/3 := by
sorry

end perpendicular_vectors_x_value_l3164_316430


namespace rectangle_area_diagonal_l3164_316466

/-- For a rectangle with length to width ratio of 5:2 and diagonal d, 
    the area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) 
    (h_diagonal : l^2 + w^2 = d^2) : l * w = (10/29) * d^2 := by
  sorry

end rectangle_area_diagonal_l3164_316466


namespace inscribed_rectangle_area_l3164_316492

/-- A circular segment with a 120° arc and height h -/
structure CircularSegment :=
  (h : ℝ)
  (arc_angle : ℝ)
  (arc_angle_eq : arc_angle = 120)

/-- A rectangle inscribed in a circular segment -/
structure InscribedRectangle (seg : CircularSegment) :=
  (AB : ℝ)
  (BC : ℝ)
  (ratio : AB / BC = 1 / 4)
  (BC_on_chord : True)  -- Represents that BC lies on the chord

/-- The area of an inscribed rectangle -/
def area (seg : CircularSegment) (rect : InscribedRectangle seg) : ℝ :=
  rect.AB * rect.BC

/-- Theorem: The area of the inscribed rectangle is 36h²/25 -/
theorem inscribed_rectangle_area (seg : CircularSegment) (rect : InscribedRectangle seg) :
  area seg rect = 36 * seg.h^2 / 25 := by
  sorry

end inscribed_rectangle_area_l3164_316492


namespace servant_months_worked_l3164_316475

/-- Calculates the number of months served given the annual salary and the amount paid -/
def months_served (annual_salary : ℚ) (amount_paid : ℚ) : ℚ :=
  (amount_paid * 12) / annual_salary

theorem servant_months_worked (annual_salary : ℚ) (amount_paid : ℚ) 
  (h1 : annual_salary = 90)
  (h2 : amount_paid = 75) :
  months_served annual_salary amount_paid = 10 := by
  sorry

end servant_months_worked_l3164_316475


namespace mary_initial_amount_l3164_316469

def marco_initial : ℕ := 24

theorem mary_initial_amount (mary_initial : ℕ) : mary_initial = 27 :=
  by
  have h1 : mary_initial + marco_initial / 2 > marco_initial / 2 := by sorry
  have h2 : mary_initial - 5 = marco_initial / 2 + 10 := by sorry
  sorry


end mary_initial_amount_l3164_316469


namespace integer_fraction_conditions_l3164_316417

theorem integer_fraction_conditions (p a b : ℕ) : 
  Prime p → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℤ, (4 * a + p : ℤ) / b + (4 * b + p : ℤ) / a = k) → 
  (∃ m : ℤ, (a^2 : ℤ) / b + (b^2 : ℤ) / a = m) → 
  a = b ∨ a = p * b :=
by sorry

end integer_fraction_conditions_l3164_316417


namespace scallop_cost_theorem_l3164_316499

def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) 
  (price_per_pound : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_scallops := people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  let initial_cost := pounds_needed * price_per_pound
  let discounted_cost := initial_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  final_cost

theorem scallop_cost_theorem :
  let result := scallop_cost 8 2 8 24 (1/10) (7/100)
  ⌊result * 100⌋ / 100 = 4622 / 100 := by sorry

end scallop_cost_theorem_l3164_316499


namespace orange_selling_loss_l3164_316422

def total_money : ℚ := 75
def ratio_sum : ℕ := 4 + 5 + 6
def cara_ratio : ℕ := 4
def janet_ratio : ℕ := 5
def selling_percentage : ℚ := 80 / 100

theorem orange_selling_loss :
  let cara_money := (cara_ratio : ℚ) / ratio_sum * total_money
  let janet_money := (janet_ratio : ℚ) / ratio_sum * total_money
  let combined_money := cara_money + janet_money
  let selling_price := selling_percentage * combined_money
  combined_money - selling_price = 9 := by sorry

end orange_selling_loss_l3164_316422


namespace z_in_first_quadrant_l3164_316433

-- Define the complex number
def z : ℂ := Complex.I * (1 - Complex.I)

-- Theorem statement
theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_first_quadrant_l3164_316433


namespace parametric_to_ordinary_equation_l3164_316435

theorem parametric_to_ordinary_equation (α : ℝ) :
  let x := Real.sin (α / 2) + Real.cos (α / 2)
  let y := Real.sqrt (2 + Real.sin α)
  (y ^ 2 - x ^ 2 = 1) ∧
  (|x| ≤ Real.sqrt 2) ∧
  (1 ≤ y) ∧
  (y ≤ Real.sqrt 3) := by
  sorry

end parametric_to_ordinary_equation_l3164_316435


namespace same_side_line_range_l3164_316406

theorem same_side_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 2) → 
    (a * x + 2 * y - 1) * (a * 3 + 2 * (-1) - 1) > 0) ↔ 
  a ∈ Set.Ioo 1 3 :=
sorry

end same_side_line_range_l3164_316406


namespace math_fun_books_count_l3164_316477

theorem math_fun_books_count : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 18 * x + 8 * y = 92 ∧ y = 7 := by
  sorry

end math_fun_books_count_l3164_316477


namespace ice_cream_stacking_l3164_316403

theorem ice_cream_stacking (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end ice_cream_stacking_l3164_316403


namespace max_horses_for_25_and_7_l3164_316434

/-- Given a total number of horses and a minimum number of races to determine the top 3 fastest,
    calculate the maximum number of horses that can race together at a time. -/
def max_horses_per_race (total_horses : ℕ) (min_races : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 25 horses and 7 minimum races, the maximum number of horses
    that can race together is 5. -/
theorem max_horses_for_25_and_7 :
  max_horses_per_race 25 7 = 5 := by
  sorry

end max_horses_for_25_and_7_l3164_316434


namespace farmers_field_planted_fraction_l3164_316481

theorem farmers_field_planted_fraction 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (leg_lengths : a = 5 ∧ b = 12) 
  (square_side : ℝ) 
  (square_distance_to_hypotenuse : ℝ) 
  (square_distance_condition : square_distance_to_hypotenuse = 3) 
  (square_tangent : square_side ≤ a ∧ square_side ≤ b) 
  (area_equation : (1/2) * c * square_distance_to_hypotenuse = (1/2) * a * b - square_side^2) :
  (((1/2) * a * b - square_side^2) / ((1/2) * a * b)) = 7/10 := by
  sorry

end farmers_field_planted_fraction_l3164_316481


namespace complex_product_real_l3164_316437

theorem complex_product_real (t : ℝ) : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := t + i
  (z₁ * z₂).im = 0 → t = -3/4 := by
  sorry

end complex_product_real_l3164_316437


namespace larger_number_proof_l3164_316459

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 3300) (h3 : a > b) : a = 300 := by
  sorry

end larger_number_proof_l3164_316459


namespace arithmetic_sequence_sum_l3164_316429

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₆ + a₁₁ = 3,
    prove that a₃ + a₉ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
  sorry

end arithmetic_sequence_sum_l3164_316429


namespace root_equation_value_l3164_316476

theorem root_equation_value (a : ℝ) : 
  a^2 - 3*a - 1011 = 0 → 2*a^2 - 6*a + 1 = 2023 := by
  sorry

end root_equation_value_l3164_316476


namespace circle_radius_l3164_316450

theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 - 6*x + y^2 + 2*y + 6 = 0) → 
  ∃ r : ℝ, r = 2 ∧ ∀ x y, (x - 3)^2 + (y + 1)^2 = r^2 := by
sorry

end circle_radius_l3164_316450


namespace figure_50_squares_l3164_316457

def square_count (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 1

theorem figure_50_squares :
  square_count 0 = 1 ∧
  square_count 1 = 6 ∧
  square_count 2 = 15 ∧
  square_count 3 = 28 →
  square_count 50 = 5151 := by
  sorry

end figure_50_squares_l3164_316457


namespace greatest_sum_consecutive_integers_l3164_316460

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) → 
  n + (n + 1) = 43 :=
sorry

end greatest_sum_consecutive_integers_l3164_316460


namespace set_equality_implies_difference_l3164_316439

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = {1, a+b, a} → b - a = 2 := by
sorry

end set_equality_implies_difference_l3164_316439
