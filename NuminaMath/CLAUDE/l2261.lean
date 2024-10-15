import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_seven_place_values_l2261_226103

/-- Given the number 87953.0727, this theorem states that the sum of the place values
    of the three 7's in this number is equal to 7,000.0707. -/
theorem sum_of_seven_place_values (n : ℝ) (h : n = 87953.0727) :
  7000 + 0.07 + 0.0007 = 7000.0707 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_place_values_l2261_226103


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2261_226104

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_cond : x ≥ -1)
  (y_cond : y ≥ -3/2)
  (z_cond : z ≥ -2) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    x₀ ≥ -1 ∧ 
    y₀ ≥ -3/2 ∧ 
    z₀ ≥ -2 ∧
    Real.sqrt (4 * x₀ + 2) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (4 * z₀ + 8) = 2 * Real.sqrt 30 ∧
    ∀ (x y z : ℝ), 
      x + y + z = 2 → 
      x ≥ -1 → 
      y ≥ -3/2 → 
      z ≥ -2 → 
      Real.sqrt (4 * x + 2) + Real.sqrt (4 * y + 6) + Real.sqrt (4 * z + 8) ≤ 2 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2261_226104


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l2261_226107

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    if the base of the triangle is 24 inches and the base of the trapezoid
    is half the length of the triangle's base, then the median of the
    trapezoid is 12 inches. -/
theorem trapezoid_median_length
  (triangle_area trapezoid_area : ℝ)
  (altitude : ℝ)
  (triangle_base trapezoid_base : ℝ)
  (trapezoid_median : ℝ) :
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base = triangle_base / 2 →
  triangle_area = (1 / 2) * triangle_base * altitude →
  trapezoid_area = trapezoid_median * altitude →
  trapezoid_median = (trapezoid_base + trapezoid_base) / 2 →
  trapezoid_median = 12 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_median_length_l2261_226107


namespace NUMINAMATH_CALUDE_sophias_book_length_l2261_226145

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (total_pages / 2 : ℚ) + 45 →
  total_pages = 4556 :=
by
  sorry

end NUMINAMATH_CALUDE_sophias_book_length_l2261_226145


namespace NUMINAMATH_CALUDE_marble_selection_probability_l2261_226162

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 2

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 2

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The number of yellow marbles in the bag -/
def yellow_marbles : ℕ := 1

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles + yellow_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 3

/-- The probability of selecting one red, one blue, and one green marble -/
def probability_red_blue_green : ℚ := 8 / 35

theorem marble_selection_probability :
  probability_red_blue_green = (red_marbles * blue_marbles * green_marbles : ℚ) / (total_marbles.choose selected_marbles) :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l2261_226162


namespace NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2261_226184

/-- Represents the cost of a text messaging plan -/
structure PlanCost where
  perMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : PlanCost) (messages : ℕ) : ℚ :=
  plan.perMessage * messages + plan.monthlyFee

/-- The two text messaging plans offered by the cell phone company -/
def planA : PlanCost := { perMessage := 0.25, monthlyFee := 9 }
def planB : PlanCost := { perMessage := 0.40, monthlyFee := 0 }

theorem equal_cost_at_60_messages :
  ∃ (messages : ℕ), messages = 60 ∧ totalCost planA messages = totalCost planB messages :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_60_messages_l2261_226184


namespace NUMINAMATH_CALUDE_fraction_product_equality_l2261_226160

theorem fraction_product_equality : (2/3)^4 * (1/5) * (3/4) = 4/135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l2261_226160


namespace NUMINAMATH_CALUDE_p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l2261_226178

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem stating that p is true
theorem p_is_true : p := sorry

-- Theorem stating that q is false
theorem q_is_false : ¬q := sorry

-- Theorem stating that the disjunction of p and q is true
theorem p_or_q_is_true : p ∨ q := sorry

-- Theorem stating that the conjunction of p and not q is true
theorem p_and_not_q_is_true : p ∧ ¬q := sorry

end NUMINAMATH_CALUDE_p_is_true_q_is_false_p_or_q_is_true_p_and_not_q_is_true_l2261_226178


namespace NUMINAMATH_CALUDE_divide_seven_friends_four_teams_l2261_226146

/-- The number of ways to divide n friends among k teams -/
def divideFriends (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: Dividing 7 friends among 4 teams results in 16384 ways -/
theorem divide_seven_friends_four_teams : 
  divideFriends 7 4 = 16384 := by
  sorry

end NUMINAMATH_CALUDE_divide_seven_friends_four_teams_l2261_226146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2261_226185

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : S seq 8 = 4 * seq.a 3)
  (h2 : seq.a 7 = -2) :
  seq.a 9 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2261_226185


namespace NUMINAMATH_CALUDE_janes_numbers_l2261_226131

def is_between (n : ℕ) (a b : ℕ) : Prop := a ≤ n ∧ n ≤ b

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_conditions (n : ℕ) : Prop :=
  is_between n 100 150 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  sum_of_digits n % 4 = 0

theorem janes_numbers : 
  {n : ℕ | satisfies_conditions n} = {112, 147} := by sorry

end NUMINAMATH_CALUDE_janes_numbers_l2261_226131


namespace NUMINAMATH_CALUDE_probability_of_selecting_specific_pair_l2261_226133

/-- Given a box of shoes with the following properties:
    - There are 20 pairs of shoes (40 shoes in total)
    - Each pair has a unique design
    Prove that the probability of randomly selecting both shoes
    from a specific pair (pair A) is 1/780. -/
theorem probability_of_selecting_specific_pair (total_shoes : Nat) (total_pairs : Nat)
    (h1 : total_shoes = 40)
    (h2 : total_pairs = 20)
    (h3 : total_shoes = 2 * total_pairs) :
  (1 : ℚ) / total_shoes * (1 : ℚ) / (total_shoes - 1) = 1 / 780 := by
  sorry

#check probability_of_selecting_specific_pair

end NUMINAMATH_CALUDE_probability_of_selecting_specific_pair_l2261_226133


namespace NUMINAMATH_CALUDE_pages_per_day_l2261_226193

/-- Given a book with 144 pages, prove that reading two-thirds of it in 12 days results in 8 pages read per day. -/
theorem pages_per_day (total_pages : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  total_pages = 144 → 
  days_read = 12 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages) / days_read = 8 := by
sorry

end NUMINAMATH_CALUDE_pages_per_day_l2261_226193


namespace NUMINAMATH_CALUDE_existence_equivalence_l2261_226154

/-- Proves the equivalence between the existence of x in [1, 2] satisfying 
    2x^2 - ax + 2 > 0 and a < 4 for any real number a -/
theorem existence_equivalence (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 4 := by
  sorry

#check existence_equivalence

end NUMINAMATH_CALUDE_existence_equivalence_l2261_226154


namespace NUMINAMATH_CALUDE_circle_equation_symmetric_center_l2261_226159

/-- A circle C in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle. -/
def standardEquation (c : Circle) : ℝ → ℝ → Prop :=
  λ x y => (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry of two points about the line y = x. -/
def symmetricAboutDiagonal (p q : ℝ × ℝ) : Prop :=
  p.1 + q.2 = p.2 + q.1 ∧ p.1 + p.2 = q.1 + q.2

theorem circle_equation_symmetric_center (c : Circle) :
  c.radius = 1 →
  symmetricAboutDiagonal c.center (1, 0) →
  standardEquation c = λ x y => x^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_symmetric_center_l2261_226159


namespace NUMINAMATH_CALUDE_paper_pallet_ratio_l2261_226120

theorem paper_pallet_ratio (total : ℕ) (towels tissues cups plates : ℕ) : 
  total = 20 → 
  towels = total / 2 → 
  tissues = total / 4 → 
  cups = 1 → 
  plates = total - (towels + tissues + cups) → 
  (plates : ℚ) / total = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_paper_pallet_ratio_l2261_226120


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2261_226172

-- Define set A
def A : Set ℝ := {x | |x| > 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2261_226172


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2261_226169

theorem exactly_one_greater_than_one (a b c : ℝ) : 
  a * b * c = 1 → a + b + c > 1/a + 1/b + 1/c → 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l2261_226169


namespace NUMINAMATH_CALUDE_average_difference_is_negative_13_point_5_l2261_226155

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculates the average number of students per teacher -/
def average_students_per_teacher (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculates the average number of students per student -/
def average_students_per_student (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to be proved -/
theorem average_difference_is_negative_13_point_5 (school : School) 
  (h1 : school.num_students = 100)
  (h2 : school.num_teachers = 5)
  (h3 : school.class_sizes = [50, 20, 20, 5, 5]) :
  average_students_per_teacher school - average_students_per_student school = -13.5 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_is_negative_13_point_5_l2261_226155


namespace NUMINAMATH_CALUDE_dogsled_speed_difference_l2261_226164

/-- Calculates the difference in average speed between two dogsled teams -/
theorem dogsled_speed_difference 
  (course_distance : ℝ) 
  (team_e_speed : ℝ) 
  (time_difference : ℝ) : 
  course_distance = 300 →
  team_e_speed = 20 →
  time_difference = 3 →
  (course_distance / (course_distance / team_e_speed - time_difference)) - team_e_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_dogsled_speed_difference_l2261_226164


namespace NUMINAMATH_CALUDE_product_evaluation_l2261_226128

theorem product_evaluation : (3 + 1) * (3^3 + 1^3) * (3^9 + 1^9) = 2878848 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2261_226128


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_108_l2261_226181

/-- Represents a triangle with sides a, b, c and incenter radius r -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  r : ℕ+
  isIsosceles : a = b
  incenterRadius : r = 8

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.a.val + t.b.val + t.c.val

/-- Theorem: The smallest possible perimeter of a triangle satisfying the given conditions is 108 -/
theorem smallest_perimeter_is_108 :
  ∀ t : Triangle, perimeter t ≥ 108 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_108_l2261_226181


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2261_226161

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = 1 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2261_226161


namespace NUMINAMATH_CALUDE_shenzhen_metro_growth_l2261_226189

/-- Represents the passenger growth of Shenzhen Metro Line 11 -/
theorem shenzhen_metro_growth (x : ℝ) : 
  (1.2 : ℝ) * (1 + x)^2 = 1.75 ↔ 
  120 * (1 + x)^2 = 175 := by sorry

#check shenzhen_metro_growth

end NUMINAMATH_CALUDE_shenzhen_metro_growth_l2261_226189


namespace NUMINAMATH_CALUDE_composite_product_ratio_l2261_226102

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product (l : List Nat) : Nat := l.foldl (· * ·) 1

theorem composite_product_ratio :
  (product first_six_composites : ℚ) / (product next_six_composites) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_ratio_l2261_226102


namespace NUMINAMATH_CALUDE_chocolate_theorem_l2261_226152

def chocolate_problem (total : ℕ) (typeA typeB typeC : ℕ) : Prop :=
  let typeD := 2 * typeA
  let typeE := 2 * typeB
  let typeF := typeA + 6
  let typeG := typeB + 6
  let typeH := typeC + 6
  let non_peanut := typeA + typeB + typeC + typeD + typeE + typeF + typeG + typeH
  let peanut := total - non_peanut
  (peanut : ℚ) / total = 3 / 10

theorem chocolate_theorem :
  chocolate_problem 100 5 6 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l2261_226152


namespace NUMINAMATH_CALUDE_parabola_equation_l2261_226117

/-- A parabola with its focus and a line passing through it -/
structure ParabolaWithLine where
  p : ℝ
  focus : ℝ × ℝ
  line : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The conditions of the problem -/
def problem_conditions (P : ParabolaWithLine) : Prop :=
  P.p > 0 ∧
  P.focus = (P.p / 2, 0) ∧
  P.focus ∈ P.line ∧
  P.A ∈ P.line ∧ P.B ∈ P.line ∧
  P.A.1 ^ 2 = 2 * P.p * P.A.2 ∧
  P.B.1 ^ 2 = 2 * P.p * P.B.2 ∧
  ((P.A.1 + P.B.1) / 2, (P.A.2 + P.B.2) / 2) = (3, 2)

/-- The theorem statement -/
theorem parabola_equation (P : ParabolaWithLine) :
  problem_conditions P →
  P.p = 2 ∨ P.p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2261_226117


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2261_226135

theorem x_plus_y_values (x y : ℝ) : 
  (|x| = 3) → (|y| = 2) → (|x - y| = y - x) → 
  (x + y = -1 ∨ x + y = -5) := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2261_226135


namespace NUMINAMATH_CALUDE_rachel_apple_trees_l2261_226153

/-- The total number of apples remaining on Rachel's trees -/
def total_apples_remaining (X : ℕ) : ℕ :=
  let first_four_trees := 10 + 40 + 15 + 22
  let remaining_trees := 48 * X
  first_four_trees + remaining_trees

/-- Theorem stating the total number of apples remaining on Rachel's trees -/
theorem rachel_apple_trees (X : ℕ) :
  total_apples_remaining X = 87 + 48 * X := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_trees_l2261_226153


namespace NUMINAMATH_CALUDE_meeting_point_distance_l2261_226111

/-- 
Given two people starting at opposite ends of a path, prove that they meet
when the slower person has traveled a specific distance.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_slow : ℝ) 
  (speed_fast : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed_slow = 3)
  (h3 : speed_fast = 6)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (meeting_distance : ℝ), 
    meeting_distance = total_distance * speed_slow / (speed_slow + speed_fast) ∧ 
    meeting_distance = 12 := by
sorry


end NUMINAMATH_CALUDE_meeting_point_distance_l2261_226111


namespace NUMINAMATH_CALUDE_min_value_theorem_l2261_226113

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) (hab : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
  ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
  1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2261_226113


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2261_226143

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, x = Real.sqrt y ∧ 
  (∀ z : ℝ, z > 0 → z ≠ y → ¬∃ (a b : ℝ), a^2 * b = y ∧ b > 0 ∧ b ≠ 1)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (Real.sqrt 6) ∧ 
  ¬is_simplest_quadratic_radical (Real.sqrt 27) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/4)) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2261_226143


namespace NUMINAMATH_CALUDE_expression_value_l2261_226148

theorem expression_value (x : ℝ) : 
  let a : ℝ := 2005 * x + 2009
  let b : ℝ := 2005 * x + 2010
  let c : ℝ := 2005 * x + 2011
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2261_226148


namespace NUMINAMATH_CALUDE_largest_divisible_digit_l2261_226110

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def number_with_digit (d : ℕ) : ℕ := 78120 + d

theorem largest_divisible_digit : 
  (∀ d : ℕ, d ≤ 9 → is_divisible_by_6 (number_with_digit d) → d ≤ 6) ∧ 
  is_divisible_by_6 (number_with_digit 6) :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_digit_l2261_226110


namespace NUMINAMATH_CALUDE_calculation_proof_l2261_226183

theorem calculation_proof : 
  Real.sqrt 5 * (-Real.sqrt 10) - (1/7)⁻¹ + |-(2^3)| = -5 * Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2261_226183


namespace NUMINAMATH_CALUDE_bicyclist_average_speed_l2261_226126

/-- The average speed of a bicyclist's trip -/
theorem bicyclist_average_speed :
  let total_distance : ℝ := 250
  let first_part_distance : ℝ := 100
  let first_part_speed : ℝ := 20
  let second_part_distance : ℝ := total_distance - first_part_distance
  let second_part_speed : ℝ := 15
  let average_speed : ℝ := total_distance / (first_part_distance / first_part_speed + second_part_distance / second_part_speed)
  average_speed = 250 / (100 / 20 + 150 / 15) :=
by
  sorry

#eval (250 : Float) / ((100 : Float) / 20 + (150 : Float) / 15)

end NUMINAMATH_CALUDE_bicyclist_average_speed_l2261_226126


namespace NUMINAMATH_CALUDE_missing_number_equation_l2261_226168

theorem missing_number_equation : ∃! x : ℝ, x + 3699 + 1985 - 2047 = 31111 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2261_226168


namespace NUMINAMATH_CALUDE_max_value_on_curve_l2261_226114

/-- Given a point (a,b) on the curve y = e^2 / x where a > 1 and b > 1,
    the maximum value of a^(ln b) is e. -/
theorem max_value_on_curve (a b : ℝ) : 
  a > 1 → b > 1 → b = Real.exp 2 / a → (Real.exp 1 : ℝ) ≥ a^(Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l2261_226114


namespace NUMINAMATH_CALUDE_f_odd_iff_a_b_zero_l2261_226197

/-- The function f defined with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ x * |x + a| + b

/-- f is an odd function if and only if a^2 + b^2 = 0 -/
theorem f_odd_iff_a_b_zero (a b : ℝ) :
  (∀ x, f a b (-x) = -(f a b x)) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_iff_a_b_zero_l2261_226197


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2261_226157

theorem sum_smallest_largest_prime_1_to_50 : ∃ (p q : Nat), 
  (p.Prime ∧ q.Prime) ∧ 
  (∀ r, r.Prime → 1 < r ∧ r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 49 := by
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2261_226157


namespace NUMINAMATH_CALUDE_horner_rule_v₃_l2261_226147

/-- Horner's Rule for a specific polynomial -/
def horner_polynomial (x : ℤ) : ℤ := (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (x : ℤ) : ℤ :=
  let v₀ := 1
  let v₁ := x - 5 * v₀
  let v₂ := x * v₁ + 6
  x * v₂ + 0

theorem horner_rule_v₃ :
  v₃ (-2) = -40 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v₃_l2261_226147


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l2261_226123

theorem bakery_rolls_combinations :
  let n : ℕ := 8  -- total number of rolls
  let k : ℕ := 4  -- number of roll types
  let remaining : ℕ := n - k  -- remaining rolls after putting one in each category
  (Nat.choose (remaining + k - 1) (k - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l2261_226123


namespace NUMINAMATH_CALUDE_hall_width_proof_l2261_226179

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 25 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) 
    (h1 : length = 20)
    (h2 : height = 5)
    (h3 : cost_per_sqm = 20)
    (h4 : total_cost = 19000) :
  ∃ (width : ℝ), 
    total_cost = cost_per_sqm * (length * width + 2 * length * height + 2 * width * height) ∧ 
    width = 25 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_proof_l2261_226179


namespace NUMINAMATH_CALUDE_part1_part2_l2261_226100

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the specific conditions of our triangle
def ourTriangle (t : Triangle) : Prop :=
  isValidTriangle t ∧
  t.B = Real.pi / 3 ∧  -- 60 degrees
  t.c = 8

-- Define the midpoint condition
def isMidpoint (M : ℝ × ℝ) (B C : ℝ × ℝ) : Prop :=
  M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2

-- Theorem for part 1
theorem part1 (t : Triangle) (M : ℝ × ℝ) (B C : ℝ × ℝ) :
  ourTriangle t →
  isMidpoint M B C →
  (Real.sqrt 3) * (Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2)) = 
    Real.sqrt ((t.A - M.1)^2 + (t.A - M.2)^2) →
  t.b = 8 :=
sorry

-- Theorem for part 2
theorem part2 (t : Triangle) :
  ourTriangle t →
  t.b = 12 →
  (1/2) * t.b * t.c * Real.sin t.A = 24 * Real.sqrt 2 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2261_226100


namespace NUMINAMATH_CALUDE_correct_calculation_l2261_226190

theorem correct_calculation (x : ℝ) (h : 2 * x = 22) : 20 * x + 3 = 223 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2261_226190


namespace NUMINAMATH_CALUDE_right_triangle_set_l2261_226127

theorem right_triangle_set : ∃! (a b c : ℕ), 
  ((a = 7 ∧ b = 24 ∧ c = 25) ∨ 
   (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 8 ∧ b = 15 ∧ c = 18)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2261_226127


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2261_226136

theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2261_226136


namespace NUMINAMATH_CALUDE_g_zero_l2261_226196

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero : g (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_l2261_226196


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2261_226101

theorem consecutive_integers_sum (n : ℚ) : 
  (n - 1) + (n + 1) + (n + 2) = 175 → n = 57 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2261_226101


namespace NUMINAMATH_CALUDE_family_adults_count_l2261_226194

/-- Represents the number of adults in a family visiting an amusement park. -/
def adults : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 22

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 7

/-- The number of children in the family. -/
def num_children : ℕ := 2

/-- The total cost for the family's admission in dollars. -/
def total_cost : ℕ := 58

/-- Theorem stating that the number of adults in the family is 2. -/
theorem family_adults_count : adults = 2 := by
  sorry

end NUMINAMATH_CALUDE_family_adults_count_l2261_226194


namespace NUMINAMATH_CALUDE_expression_simplification_l2261_226124

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 2) 
  (hb : b = Real.sqrt 3 - 2) : 
  (a^2 / (a^2 + 2*a*b + b^2) - a / (a + b)) / (a^2 / (a^2 - b^2) - b / (a - b) - 1) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2261_226124


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2261_226188

theorem quadratic_equation_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - (a+8)*x + 8*a - 1 = 0 ∧ y^2 - (a+8)*y + 8*a - 1 = 0) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2261_226188


namespace NUMINAMATH_CALUDE_dave_has_more_cats_l2261_226182

/-- The number of pets owned by Teddy, Ben, and Dave -/
structure PetOwnership where
  teddy_dogs : ℕ
  teddy_cats : ℕ
  ben_dogs : ℕ
  dave_dogs : ℕ
  dave_cats : ℕ

/-- The conditions of the pet ownership problem -/
def pet_problem (p : PetOwnership) : Prop :=
  p.teddy_dogs = 7 ∧
  p.teddy_cats = 8 ∧
  p.ben_dogs = p.teddy_dogs + 9 ∧
  p.dave_dogs = p.teddy_dogs - 5 ∧
  p.teddy_dogs + p.teddy_cats + p.ben_dogs + p.dave_dogs + p.dave_cats = 54

/-- The theorem stating that Dave has 13 more cats than Teddy -/
theorem dave_has_more_cats (p : PetOwnership) (h : pet_problem p) :
  p.dave_cats = p.teddy_cats + 13 := by
  sorry

end NUMINAMATH_CALUDE_dave_has_more_cats_l2261_226182


namespace NUMINAMATH_CALUDE_gardens_area_difference_l2261_226130

/-- Represents a rectangular garden with length and width -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the usable area of a garden with a path around the perimeter -/
def Garden.usableArea (g : Garden) (pathWidth : ℝ) : ℝ :=
  (g.length - 2 * pathWidth) * (g.width - 2 * pathWidth)

theorem gardens_area_difference : 
  let karlGarden : Garden := { length := 22, width := 50 }
  let makennaGarden : Garden := { length := 30, width := 46 }
  let pathWidth : ℝ := 1
  makennaGarden.usableArea pathWidth - karlGarden.area = 132 := by sorry

end NUMINAMATH_CALUDE_gardens_area_difference_l2261_226130


namespace NUMINAMATH_CALUDE_same_speed_problem_l2261_226137

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 9*x - 18
  let jill_distance := x^2 - 5*x - 66
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  (x ≠ -6) →
  (jack_speed = jill_speed) →
  jack_speed = -4 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_problem_l2261_226137


namespace NUMINAMATH_CALUDE_a_10_ends_with_1000_nines_l2261_226132

def a : ℕ → ℕ
  | 0 => 9
  | (n + 1) => 3 * (a n)^4 + 4 * (a n)^3

def ends_with_nines (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^k + (10^k - 1)

theorem a_10_ends_with_1000_nines : ends_with_nines (a 10) 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_10_ends_with_1000_nines_l2261_226132


namespace NUMINAMATH_CALUDE_cloth_loss_per_metre_l2261_226140

def cloth_problem (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : Prop :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  let loss_per_metre := total_loss / total_metres
  total_metres = 300 ∧ 
  total_selling_price = 18000 ∧ 
  cost_price_per_metre = 65 ∧
  loss_per_metre = 5

theorem cloth_loss_per_metre :
  ∃ (total_metres total_selling_price cost_price_per_metre : ℕ),
    cloth_problem total_metres total_selling_price cost_price_per_metre :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_loss_per_metre_l2261_226140


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_150_500_l2261_226177

theorem lcm_gcf_ratio_150_500 : Nat.lcm 150 500 / Nat.gcd 150 500 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_150_500_l2261_226177


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_greater_than_two_point_five_l2261_226198

theorem inequality_holds_iff_p_greater_than_two_point_five (p q : ℝ) (hq : q > 0) :
  (5 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q ↔ p > 2.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_greater_than_two_point_five_l2261_226198


namespace NUMINAMATH_CALUDE_house_price_calculation_l2261_226119

theorem house_price_calculation (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end NUMINAMATH_CALUDE_house_price_calculation_l2261_226119


namespace NUMINAMATH_CALUDE_power_five_mod_seven_l2261_226174

theorem power_five_mod_seven : 5^2010 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_seven_l2261_226174


namespace NUMINAMATH_CALUDE_correct_average_weight_l2261_226173

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_average = 58.4 ∧ 
  misread_weight = 56 ∧ 
  correct_weight = 68 →
  (n : ℝ) * initial_average + (correct_weight - misread_weight) = n * 59 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2261_226173


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2261_226187

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2261_226187


namespace NUMINAMATH_CALUDE_route_time_difference_l2261_226176

theorem route_time_difference (x : ℝ) (h : x > 0) : 
  10 / x - 7 / ((1 + 0.4) * x) = 10 / 60 :=
by
  sorry

#check route_time_difference

end NUMINAMATH_CALUDE_route_time_difference_l2261_226176


namespace NUMINAMATH_CALUDE_rosa_phone_calls_l2261_226171

theorem rosa_phone_calls (total_pages : ℝ) (pages_this_week : ℝ) 
  (h1 : total_pages = 18.8) 
  (h2 : pages_this_week = 8.6) : 
  total_pages - pages_this_week = 10.2 := by
sorry

end NUMINAMATH_CALUDE_rosa_phone_calls_l2261_226171


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2261_226138

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Set (ℕ × ℕ) := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in set T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1))

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2261_226138


namespace NUMINAMATH_CALUDE_sum_of_common_x_coords_l2261_226165

/-- Given two congruences modulo 16, find the sum of x-coordinates of common points -/
theorem sum_of_common_x_coords : ∃ (S : Finset ℕ),
  (∀ x ∈ S, ∃ y : ℕ, (y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16])) ∧
  (∀ x : ℕ, (∃ y : ℕ, y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16]) → x ∈ S) ∧
  (Finset.sum S id = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_x_coords_l2261_226165


namespace NUMINAMATH_CALUDE_spherical_cap_area_ratio_l2261_226122

/-- Given two concentric spheres and a spherical cap area on the smaller sphere,
    calculate the corresponding spherical cap area on the larger sphere. -/
theorem spherical_cap_area_ratio (R₁ R₂ A₁ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (R₂ / R₁)^2
  R₁ = 4 → R₂ = 6 → A₁ = 17 → A₂ = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_spherical_cap_area_ratio_l2261_226122


namespace NUMINAMATH_CALUDE_expression_values_l2261_226129

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -3 :=
sorry

end NUMINAMATH_CALUDE_expression_values_l2261_226129


namespace NUMINAMATH_CALUDE_cricket_game_overs_l2261_226191

/-- The number of initial overs in a cricket game -/
def initial_overs : ℕ := 20

/-- The initial run rate in runs per over -/
def initial_run_rate : ℚ := 46/10

/-- The target score in runs -/
def target_score : ℕ := 396

/-- The number of remaining overs -/
def remaining_overs : ℕ := 30

/-- The required run rate for the remaining overs -/
def required_run_rate : ℚ := 10133333333333333/1000000000000000

theorem cricket_game_overs :
  initial_overs * initial_run_rate + 
  remaining_overs * required_run_rate = target_score :=
sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l2261_226191


namespace NUMINAMATH_CALUDE_multiplication_of_powers_of_ten_l2261_226163

theorem multiplication_of_powers_of_ten : (2 * 10^3) * (8 * 10^3) = 1.6 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_of_ten_l2261_226163


namespace NUMINAMATH_CALUDE_infinitely_many_wrappers_l2261_226150

/-- A wrapper for a 1 × 1 painting is a rectangle with area 2 that can cover the painting on both sides. -/
def IsWrapper (width height : ℝ) : Prop :=
  width > 0 ∧ height > 0 ∧ width * height = 2 ∧ width ≥ 1 ∧ height ≥ 1

/-- There exist infinitely many wrappers for a 1 × 1 painting. -/
theorem infinitely_many_wrappers :
  ∃ f : ℕ → ℝ × ℝ, ∀ n : ℕ, IsWrapper (f n).1 (f n).2 ∧
    ∀ m : ℕ, m ≠ n → f m ≠ f n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_wrappers_l2261_226150


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l2261_226144

/-- Represents a configuration of lit buttons on a 3 × 2 grid -/
def ButtonGrid := Fin 3 → Fin 2 → Bool

/-- Returns true if at least one button in the grid is lit -/
def atLeastOneLit (grid : ButtonGrid) : Prop :=
  ∃ i j, grid i j = true

/-- Two grids are equivalent if one can be obtained from the other by translation -/
def equivalentGrids (grid1 grid2 : ButtonGrid) : Prop :=
  sorry

/-- The number of distinct observable arrangements -/
def distinctArrangements : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct arrangements is 44 -/
theorem distinct_arrangements_count :
  distinctArrangements = 44 :=
sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l2261_226144


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2261_226118

theorem cubic_root_sum (p q r : ℝ) : 
  p + q + r = 4 →
  p * q + p * r + q * r = 1 →
  p * q * r = -6 →
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 22 - 213 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2261_226118


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2261_226105

theorem complex_modulus_example : Complex.abs (-3 + (9/4)*Complex.I) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2261_226105


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2261_226158

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2261_226158


namespace NUMINAMATH_CALUDE_factorization_proof_l2261_226108

variable (x y b : ℝ)

theorem factorization_proof : 
  (-x^3 - 2*x^2 - x = -x*(x + 1)^2) ∧ 
  ((x - y) - 4*b^2*(x - y) = (x - y)*(1 + 2*b)*(1 - 2*b)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_proof_l2261_226108


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2261_226139

/-- The perimeter of the rectangle in feet -/
def perimeter : ℕ := 190

/-- The maximum area of the rectangle in square feet -/
def max_area : ℕ := 2256

/-- A function to calculate the area of a rectangle given one side length -/
def area (x : ℕ) : ℕ := x * (perimeter / 2 - x)

/-- Theorem stating that the maximum area of a rectangle with the given perimeter and integer side lengths is 2256 square feet -/
theorem max_rectangle_area :
  ∀ x : ℕ, x > 0 ∧ x < perimeter / 2 → area x ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2261_226139


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seat_capacity_l2261_226186

/-- Represents a Ferris wheel with a given number of seats and total capacity. -/
structure FerrisWheel where
  numSeats : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each seat in a Ferris wheel. -/
def seatCapacity (wheel : FerrisWheel) : ℕ :=
  wheel.totalCapacity / wheel.numSeats

theorem paradise_park_ferris_wheel_seat_capacity :
  let wheel : FerrisWheel := { numSeats := 14, totalCapacity := 84 }
  seatCapacity wheel = 6 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seat_capacity_l2261_226186


namespace NUMINAMATH_CALUDE_inequality_proof_l2261_226175

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2261_226175


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2261_226149

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 16) 
  (h2 : x + y = 8) : 
  x^2 + y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2261_226149


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l2261_226167

theorem sum_of_divisors_30 : (Finset.filter (· ∣ 30) (Finset.range 31)).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l2261_226167


namespace NUMINAMATH_CALUDE_conors_work_week_l2261_226141

/-- Conor's vegetable chopping problem -/
theorem conors_work_week (eggplants carrots potatoes total : ℕ) 
  (h1 : eggplants = 12)
  (h2 : carrots = 9)
  (h3 : potatoes = 8)
  (h4 : total = 116) : 
  total / (eggplants + carrots + potatoes) = 4 := by
  sorry

#check conors_work_week

end NUMINAMATH_CALUDE_conors_work_week_l2261_226141


namespace NUMINAMATH_CALUDE_all_propositions_false_l2261_226151

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem all_propositions_false :
  ∀ (m n : Line) (α : Plane),
    m ≠ n →
    (¬ (parallel_line_plane m α ∧ parallel_line_plane n α → parallel_lines m n)) ∧
    (¬ (parallel_lines m n ∧ line_in_plane n α → parallel_line_plane m α)) ∧
    (¬ (perpendicular_line_plane m α ∧ perpendicular_lines m n → parallel_line_plane n α)) ∧
    (¬ (parallel_line_plane m α ∧ perpendicular_lines m n → perpendicular_line_plane n α)) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l2261_226151


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2261_226116

theorem units_digit_of_product (a b c : ℕ) : 
  (2^104 * 5^205 * 11^302) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2261_226116


namespace NUMINAMATH_CALUDE_min_distance_A_D_l2261_226134

/-- Given points A, B, C, D, E in a metric space, prove that the minimum distance between A and D is 2 units -/
theorem min_distance_A_D (X : Type*) [MetricSpace X] (A B C D E : X) :
  dist A B = 12 →
  dist B C = 7 →
  dist C E = 2 →
  dist E D = 5 →
  ∃ (d : ℝ), d ≥ 2 ∧ ∀ (d' : ℝ), dist A D ≥ d' → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_min_distance_A_D_l2261_226134


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2261_226192

theorem complex_number_quadrant (z : ℂ) : z * Complex.I = 2 - Complex.I → z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2261_226192


namespace NUMINAMATH_CALUDE_three_additional_trams_needed_l2261_226166

/-- The number of trams needed to reduce intervals by one-fifth -/
def additional_trams (initial_trams : ℕ) : ℕ :=
  let total_distance := 60
  let initial_interval := total_distance / initial_trams
  let new_interval := initial_interval * 4 / 5
  let new_total_trams := total_distance / new_interval
  new_total_trams - initial_trams

/-- Theorem stating that 3 additional trams are needed -/
theorem three_additional_trams_needed :
  additional_trams 12 = 3 := by
  sorry

#eval additional_trams 12

end NUMINAMATH_CALUDE_three_additional_trams_needed_l2261_226166


namespace NUMINAMATH_CALUDE_solution_set_theorem_m_value_theorem_l2261_226106

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3|

-- Define the function g
def g (x m : ℝ) : ℝ := f (x + m) + f (x - m)

-- Theorem for the solution set of the inequality
theorem solution_set_theorem :
  {x : ℝ | f x > 5 - |x + 2|} = {x : ℝ | x < 0 ∨ x > 2} :=
sorry

-- Theorem for the value of m
theorem m_value_theorem (m : ℝ) :
  (∀ x, g x m ≥ 4) ∧ (∃ x, g x m = 4) → m = 1 ∨ m = -1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_m_value_theorem_l2261_226106


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l2261_226156

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number :
  triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l2261_226156


namespace NUMINAMATH_CALUDE_calculator_minimum_operations_l2261_226112

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The theorem to be proved --/
theorem calculator_minimum_operations :
  ∃ (ops : List Operation),
    isValidSequence ops 400 ∧
    ops.length = 10 ∧
    (∀ (other_ops : List Operation),
      isValidSequence other_ops 400 → other_ops.length ≥ 10) := by
  sorry


end NUMINAMATH_CALUDE_calculator_minimum_operations_l2261_226112


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_common_difference_l2261_226170

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  first_term : ℚ
  last_term : ℚ
  sum : ℚ
  is_arithmetic : Bool

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  sorry

/-- Theorem stating the common difference of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_common_difference :
  let seq := ArithmeticSequence.mk 3 28 186 true
  common_difference seq = 25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_common_difference_l2261_226170


namespace NUMINAMATH_CALUDE_inequality_preserved_subtraction_l2261_226121

theorem inequality_preserved_subtraction (a b : ℝ) (h : a < b) : a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_subtraction_l2261_226121


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2261_226195

theorem polynomial_simplification (x : ℝ) :
  x * (4 * x^3 + 3 * x^2 - 5) - 7 * (x^3 - 4 * x^2 + 2 * x - 6) =
  4 * x^4 - 4 * x^3 + 28 * x^2 - 19 * x + 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2261_226195


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l2261_226125

/-- Proves that the difference between Tom's and Dorothy's payments to equalize costs is 20 --/
theorem vacation_cost_difference (tom_paid dorothy_paid sammy_paid : ℕ) 
  (h1 : tom_paid = 105)
  (h2 : dorothy_paid = 125)
  (h3 : sammy_paid = 175) : 
  (((tom_paid + dorothy_paid + sammy_paid) / 3 - tom_paid) - 
   ((tom_paid + dorothy_paid + sammy_paid) / 3 - dorothy_paid)) = 20 := by
  sorry

#eval ((105 + 125 + 175) / 3 - 105) - ((105 + 125 + 175) / 3 - 125)

end NUMINAMATH_CALUDE_vacation_cost_difference_l2261_226125


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l2261_226115

theorem gold_coin_distribution (x y : ℕ) (k : ℕ) 
  (h1 : x + y = 16) 
  (h2 : x > y) 
  (h3 : x^2 - y^2 = k * (x - y)) : 
  k = 16 :=
sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l2261_226115


namespace NUMINAMATH_CALUDE_used_car_percentage_l2261_226109

theorem used_car_percentage (used_price original_price : ℝ) 
  (h1 : used_price = 15000)
  (h2 : original_price = 37500) :
  (used_price / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_used_car_percentage_l2261_226109


namespace NUMINAMATH_CALUDE_smallest_sum_solution_l2261_226142

theorem smallest_sum_solution : ∃ (a b c : ℕ), 
  (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ 
  (∀ (x y z : ℕ), (x * z + 2 * y * z + x + 2 * y = z^2 + z + 6) → 
    (a + b + c ≤ x + y + z)) ∧
  (a = 2 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_solution_l2261_226142


namespace NUMINAMATH_CALUDE_blue_segments_count_l2261_226180

/-- Set A of points (x, y) where x and y are natural numbers between 1 and 20 inclusive -/
def A : Set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20}

/-- Set B of points (x, y) where x and y are natural numbers between 2 and 19 inclusive -/
def B : Set (ℕ × ℕ) := {p | 2 ≤ p.1 ∧ p.1 ≤ 19 ∧ 2 ≤ p.2 ∧ p.2 ≤ 19}

/-- Color of a point in A -/
inductive Color
| Red
| Blue

/-- Coloring function for points in A -/
def coloring : A → Color := sorry

/-- Total number of red points in A -/
def total_red_points : ℕ := 219

/-- Number of red points in B -/
def red_points_in_B : ℕ := 180

/-- Corner points are blue -/
axiom corner_points_blue :
  coloring ⟨(1, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(1, 20), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 1), sorry⟩ = Color.Blue ∧
  coloring ⟨(20, 20), sorry⟩ = Color.Blue

/-- Number of black line segments of length 1 -/
def black_segments : ℕ := 237

/-- Theorem: The number of blue line segments of length 1 is 233 -/
theorem blue_segments_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2261_226180


namespace NUMINAMATH_CALUDE_weight_difference_l2261_226199

/-- The weight difference between two metal pieces -/
theorem weight_difference (iron_weight aluminum_weight : ℝ) 
  (h1 : iron_weight = 11.17)
  (h2 : aluminum_weight = 0.83) : 
  iron_weight - aluminum_weight = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2261_226199
