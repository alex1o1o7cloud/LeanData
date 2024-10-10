import Mathlib

namespace third_group_men_l4072_407294

/-- The work rate of a man -/
def man_rate : ℝ := sorry

/-- The work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

/-- The work rate of 3 men and 8 women equals the work rate of 6 men and 2 women -/
axiom work_rate_equality : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate

/-- The work rate of x men and 2 women is 0.7142857142857143 times the work rate of 3 men and 8 women -/
axiom work_rate_fraction : 
  x * man_rate + 2 * woman_rate = 0.7142857142857143 * (3 * man_rate + 8 * woman_rate)

/-- The number of men in the third group is 4 -/
theorem third_group_men : x = 4 := by sorry

end third_group_men_l4072_407294


namespace complex_number_intersection_l4072_407278

theorem complex_number_intersection (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z*i} → 
  N = {3, 4} → 
  M ∩ N = {4} → 
  i^2 = -1 →
  z = -4*i := by sorry

end complex_number_intersection_l4072_407278


namespace opposite_of_negative_fraction_l4072_407257

theorem opposite_of_negative_fraction (m : ℚ) : 
  m = -(-(-(1 / 3))) → m = -(1 / 3) := by
  sorry

end opposite_of_negative_fraction_l4072_407257


namespace shift_sine_graph_l4072_407222

theorem shift_sine_graph (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin (2 * x + π / 6)
  let period := 2 * π / 2
  let shift := period / 4
  let g (x : ℝ) := f (x - shift)
  g x = 2 * Real.sin (2 * x - π / 3) := by sorry

end shift_sine_graph_l4072_407222


namespace congruence_sufficient_not_necessary_for_similarity_l4072_407274

-- Define triangles
variable (T1 T2 : Type)

-- Define congruence and similarity relations
variable (congruent : T1 → T2 → Prop)
variable (similar : T1 → T2 → Prop)

-- Theorem: Triangle congruence is sufficient but not necessary for similarity
theorem congruence_sufficient_not_necessary_for_similarity :
  (∀ t1 : T1, ∀ t2 : T2, congruent t1 t2 → similar t1 t2) ∧
  ¬(∀ t1 : T1, ∀ t2 : T2, similar t1 t2 → congruent t1 t2) :=
sorry

end congruence_sufficient_not_necessary_for_similarity_l4072_407274


namespace hired_waiters_count_l4072_407280

/-- Represents the number of waiters hired to change the ratio of cooks to waiters -/
def waiters_hired (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) : ℕ :=
  let initial_waiters := (num_cooks * initial_ratio_waiters) / initial_ratio_cooks
  let total_new_waiters := (num_cooks * new_ratio_waiters) / new_ratio_cooks
  total_new_waiters - initial_waiters

/-- Theorem stating that given the conditions, the number of waiters hired is 12 -/
theorem hired_waiters_count :
  waiters_hired 3 8 1 4 9 = 12 :=
by sorry

end hired_waiters_count_l4072_407280


namespace smallest_twice_square_three_cube_l4072_407207

def is_twice_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k^2

def is_three_times_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m^3

theorem smallest_twice_square_three_cube :
  (∀ n : ℕ, n > 0 ∧ n < 648 → ¬(is_twice_perfect_square n ∧ is_three_times_perfect_cube n)) ∧
  (is_twice_perfect_square 648 ∧ is_three_times_perfect_cube 648) :=
sorry

end smallest_twice_square_three_cube_l4072_407207


namespace shirt_and_coat_cost_l4072_407264

/-- Given a shirt that costs $150 and is one-third the price of a coat,
    prove that the total cost of the shirt and coat is $600. -/
theorem shirt_and_coat_cost (shirt_cost : ℕ) (coat_cost : ℕ) : 
  shirt_cost = 150 → 
  shirt_cost * 3 = coat_cost →
  shirt_cost + coat_cost = 600 := by
  sorry

end shirt_and_coat_cost_l4072_407264


namespace round_trip_distance_l4072_407288

/-- Calculates the total distance of a round trip given the times for each direction and the average speed -/
theorem round_trip_distance 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (avg_speed : ℝ) 
  (h1 : time_to > 0) 
  (h2 : time_from > 0) 
  (h3 : avg_speed > 0) : 
  ∃ (distance : ℝ), distance = avg_speed * (time_to + time_from) / 60 := by
  sorry

#check round_trip_distance

end round_trip_distance_l4072_407288


namespace product_remainder_divisible_by_eight_l4072_407226

theorem product_remainder_divisible_by_eight :
  (1502 * 1786 * 1822 * 2026) % 8 = 0 := by
  sorry

end product_remainder_divisible_by_eight_l4072_407226


namespace symmedian_point_is_centroid_of_projections_l4072_407245

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Projects a point onto a line segment -/
def projectOntoSegment (P : Point) (A : Point) (B : Point) : Point :=
  sorry

/-- Calculates the centroid of a triangle -/
def centroid (T : Triangle) : Point :=
  sorry

/-- Determines if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculates the Symmedian Point of a triangle -/
def symmedianPoint (T : Triangle) : Point :=
  sorry

/-- Main theorem: The Symmedian Point is the unique point inside the triangle
    that is the centroid of its projections -/
theorem symmedian_point_is_centroid_of_projections (T : Triangle) :
  let S := symmedianPoint T
  isInside S T ∧
  ∀ P, isInside P T →
    (S = P ↔
      let X := projectOntoSegment P T.B T.C
      let Y := projectOntoSegment P T.C T.A
      let Z := projectOntoSegment P T.A T.B
      P = centroid ⟨X, Y, Z⟩) :=
  sorry

end symmedian_point_is_centroid_of_projections_l4072_407245


namespace line_vector_at_negative_two_l4072_407203

def line_vector (s : ℝ) : ℝ × ℝ := sorry

theorem line_vector_at_negative_two :
  line_vector 1 = (2, 5) →
  line_vector 4 = (8, -7) →
  line_vector (-2) = (-4, 17) := by sorry

end line_vector_at_negative_two_l4072_407203


namespace range_of_a_in_fourth_quadrant_l4072_407285

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define the property of being in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -2 < a ∧ a < 3 :=
by sorry

end range_of_a_in_fourth_quadrant_l4072_407285


namespace decimal_shift_difference_l4072_407272

theorem decimal_shift_difference (x : ℝ) : 10 * x - x / 10 = 23.76 → x = 2.4 := by
  sorry

end decimal_shift_difference_l4072_407272


namespace octagon_perimeter_l4072_407292

/-- Represents an eight-sided polygon that can be divided into a rectangle and a square --/
structure OctagonWithRectAndSquare where
  rectangle_area : ℕ
  square_area : ℕ
  sum_perimeter : ℕ
  h1 : square_area > rectangle_area
  h2 : square_area * rectangle_area = 98
  h3 : ∃ (a b : ℕ), rectangle_area = a * b ∧ a > 0 ∧ b > 0
  h4 : ∃ (s : ℕ), square_area = s * s ∧ s > 0

/-- The perimeter of the octagon is 32 --/
theorem octagon_perimeter (oct : OctagonWithRectAndSquare) : oct.sum_perimeter = 32 := by
  sorry

end octagon_perimeter_l4072_407292


namespace six_digit_number_theorem_l4072_407243

/-- A six-digit number represented as a list of its digits -/
def SixDigitNumber := List Nat

/-- Checks if a list represents a valid six-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ ∀ d ∈ n.toFinset, 0 ≤ d ∧ d ≤ 9

/-- Converts a six-digit number to its integer value -/
def toInt (n : SixDigitNumber) : ℕ :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Left-shifts the digits of a six-digit number -/
def leftShift (n : SixDigitNumber) : SixDigitNumber :=
  match n with
  | [a, b, c, d, e, f] => [f, a, b, c, d, e]
  | _ => []

/-- The condition that needs to be satisfied -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  isValidSixDigitNumber n ∧
  toInt (leftShift n) = n.head! * toInt n

theorem six_digit_number_theorem :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (n = [1, 1, 1, 1, 1, 1] ∨ n = [1, 0, 2, 5, 6, 4]) :=
sorry

end six_digit_number_theorem_l4072_407243


namespace terrell_workout_equivalence_l4072_407269

/-- Given Terrell's original workout and new weights, calculate the number of lifts needed to match the total weight. -/
theorem terrell_workout_equivalence (original_weight original_reps new_weight : ℕ) : 
  original_weight = 30 →
  original_reps = 10 →
  new_weight = 20 →
  (2 * new_weight * (600 / (2 * new_weight)) : ℕ) = 2 * original_weight * original_reps :=
by
  sorry

#check terrell_workout_equivalence

end terrell_workout_equivalence_l4072_407269


namespace indefinite_integral_proof_l4072_407230

noncomputable def f (x : ℝ) : ℝ := -1 / (x + 2) + (1 / 2) * Real.log (x^2 + 4) + (1 / 2) * Real.arctan (x / 2)

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ -2) : 
  deriv f x = (x^3 + 6*x^2 + 8*x + 8) / ((x + 2)^2 * (x^2 + 4)) :=
by sorry

end indefinite_integral_proof_l4072_407230


namespace unanswered_questions_l4072_407266

/-- Represents the scoring for a math competition participant --/
structure Scoring where
  correct : ℕ      -- number of correct answers
  incorrect : ℕ    -- number of incorrect answers
  unanswered : ℕ   -- number of unanswered questions

/-- Calculates the score using the first method --/
def score_method1 (s : Scoring) : ℕ :=
  5 * s.correct + 2 * s.unanswered

/-- Calculates the score using the second method --/
def score_method2 (s : Scoring) : ℕ :=
  39 + 3 * s.correct - s.incorrect

/-- Theorem stating the possible number of unanswered questions --/
theorem unanswered_questions (s : Scoring) :
  score_method1 s = 71 ∧ score_method2 s = 71 ∧ 
  s.correct + s.incorrect + s.unanswered = s.correct + s.incorrect →
  s.unanswered = 8 ∨ s.unanswered = 3 := by
  sorry


end unanswered_questions_l4072_407266


namespace seattle_seahawks_field_goals_l4072_407250

theorem seattle_seahawks_field_goals :
  ∀ (total_score touchdown_score field_goal_score touchdown_count : ℕ),
    total_score = 37 →
    touchdown_score = 7 →
    field_goal_score = 3 →
    touchdown_count = 4 →
    ∃ (field_goal_count : ℕ),
      total_score = touchdown_count * touchdown_score + field_goal_count * field_goal_score ∧
      field_goal_count = 3 :=
by
  sorry

end seattle_seahawks_field_goals_l4072_407250


namespace definite_integral_sin_plus_one_l4072_407297

theorem definite_integral_sin_plus_one (f : ℝ → ℝ) (h : ∀ x, f x = 1 + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 2 + 1 := by
  sorry

end definite_integral_sin_plus_one_l4072_407297


namespace vector_parallel_implies_x_equals_two_l4072_407246

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_implies_x_equals_two :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  parallel (a.1 + b.1, a.2 + b.2) (4 * b.1 - 2 * a.1, 4 * b.2 - 2 * a.2) →
  x = 2 := by
sorry

end vector_parallel_implies_x_equals_two_l4072_407246


namespace slope_value_l4072_407277

theorem slope_value (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 12 → m = -2 := by
sorry

end slope_value_l4072_407277


namespace find_divisor_l4072_407298

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  divisor = 12 := by
sorry

end find_divisor_l4072_407298


namespace remainders_of_65_powers_l4072_407208

theorem remainders_of_65_powers (n : ℕ) : 
  (65^(6*n) % 9 = 1) ∧ 
  (65^(6*n + 1) % 9 = 2) ∧ 
  (65^(6*n + 2) % 9 = 4) ∧ 
  (65^(6*n + 3) % 9 = 8) := by
sorry

end remainders_of_65_powers_l4072_407208


namespace bicycle_sale_percentage_prove_bicycle_sale_percentage_l4072_407258

/-- The percentage of the suggested retail price that John paid for a bicycle -/
theorem bicycle_sale_percentage : ℝ → ℝ → ℝ → Prop :=
  fun wholesale_price suggested_retail_price johns_price =>
    suggested_retail_price = wholesale_price * (1 + 0.4) →
    johns_price = suggested_retail_price / 3 →
    johns_price / suggested_retail_price = 1 / 3

/-- Proof of the bicycle sale percentage theorem -/
theorem prove_bicycle_sale_percentage :
  ∀ (wholesale_price suggested_retail_price johns_price : ℝ),
    bicycle_sale_percentage wholesale_price suggested_retail_price johns_price := by
  sorry

#check prove_bicycle_sale_percentage

end bicycle_sale_percentage_prove_bicycle_sale_percentage_l4072_407258


namespace tenth_fib_is_55_l4072_407238

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The 10th Fibonacci number is 55 -/
theorem tenth_fib_is_55 : fib 9 = 55 := by
  sorry

end tenth_fib_is_55_l4072_407238


namespace quadratic_discriminant_zero_not_harmonic_l4072_407227

/-- The discriminant of the quadratic equation 3ax^2 + bx + 2c = 0 is zero -/
def discriminant_zero (a b c : ℝ) : Prop :=
  b^2 = 24*a*c

/-- a, b, and c form a harmonic progression -/
def harmonic_progression (a b c : ℝ) : Prop :=
  2/b = 1/a + 1/c

theorem quadratic_discriminant_zero_not_harmonic
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  discriminant_zero a b c → ¬harmonic_progression a b c :=
by
  sorry

end quadratic_discriminant_zero_not_harmonic_l4072_407227


namespace relay_race_arrangements_l4072_407265

/-- The number of students in the class -/
def total_students : Nat := 8

/-- The number of students needed for the relay race -/
def relay_team_size : Nat := 4

/-- The number of students that must be selected (A and B) -/
def must_select : Nat := 2

/-- The number of positions where A and B can be placed (first or last) -/
def fixed_positions : Nat := 2

/-- The number of remaining positions to be filled -/
def remaining_positions : Nat := relay_team_size - must_select

/-- The number of remaining students to choose from -/
def remaining_students : Nat := total_students - must_select

theorem relay_race_arrangements :
  (fixed_positions.factorial) *
  (remaining_students.choose remaining_positions) *
  (remaining_positions.factorial) = 60 := by
  sorry

end relay_race_arrangements_l4072_407265


namespace expand_and_simplify_l4072_407268

theorem expand_and_simplify (a : ℝ) : (2*a - 3)^2 + (2*a + 3)*(2*a - 3) = 8*a^2 - 12*a := by
  sorry

end expand_and_simplify_l4072_407268


namespace largest_number_l4072_407261

theorem largest_number : 
  let numbers : List ℝ := [0.935, 0.9401, 0.9349, 0.9041, 0.9400]
  ∀ x ∈ numbers, x ≤ 0.9401 := by
  sorry

end largest_number_l4072_407261


namespace two_digit_square_with_square_digit_product_l4072_407295

/-- A function that returns true if a number is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the product of digits of a two-digit number --/
def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The main theorem to be proved --/
theorem two_digit_square_with_square_digit_product : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ is_square n ∧ is_square (digit_product n) :=
sorry

end two_digit_square_with_square_digit_product_l4072_407295


namespace smallest_linear_combination_divides_l4072_407263

theorem smallest_linear_combination_divides (a b x₀ y₀ : ℤ) 
  (h_not_zero : a ≠ 0 ∨ b ≠ 0)
  (h_smallest : ∀ x y : ℤ, a * x + b * y > 0 → a * x₀ + b * y₀ ≤ a * x + b * y) :
  ∀ x y : ℤ, ∃ k : ℤ, a * x + b * y = k * (a * x₀ + b * y₀) := by
sorry

end smallest_linear_combination_divides_l4072_407263


namespace arithmetic_sequence_fourth_term_l4072_407232

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term
  (a : ℝ)  -- Third term of the sequence
  (d : ℝ)  -- Common difference of the sequence
  (h : a + (a + 2*d) = 12)  -- Sum of third and fifth terms is 12
  : a + d = 6 :=  -- Fourth term is 6
by sorry

end arithmetic_sequence_fourth_term_l4072_407232


namespace type_b_first_is_better_l4072_407273

/-- Represents the score for a correct answer to a question type -/
def score (questionType : Bool) : ℝ :=
  if questionType then 80 else 20

/-- Represents the probability of correctly answering a question type -/
def probability (questionType : Bool) : ℝ :=
  if questionType then 0.6 else 0.8

/-- Calculates the expected score when choosing a specific question type first -/
def expectedScore (firstQuestionType : Bool) : ℝ :=
  let p1 := probability firstQuestionType
  let p2 := probability (!firstQuestionType)
  let s1 := score firstQuestionType
  let s2 := score (!firstQuestionType)
  p1 * s1 + p1 * p2 * s2

/-- Theorem stating that choosing type B questions first yields a higher expected score -/
theorem type_b_first_is_better :
  expectedScore true > expectedScore false :=
sorry

end type_b_first_is_better_l4072_407273


namespace fermat_like_theorem_l4072_407239

theorem fermat_like_theorem (x y z n : ℕ) (h : n ≥ z) : x^n + y^n ≠ z^n := by
  sorry

end fermat_like_theorem_l4072_407239


namespace certain_number_problem_l4072_407229

theorem certain_number_problem : ∃ x : ℕ, 
  220020 = (x + 445) * (2 * (x - 445)) + 20 ∧ x = 555 := by
  sorry

end certain_number_problem_l4072_407229


namespace altitude_length_l4072_407215

/-- An isosceles triangle with given side lengths and altitude --/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab = ac
  -- Altitude
  ad : ℝ
  -- Altitude meets base at midpoint
  isMidpoint : bd = bc / 2

/-- The theorem stating the length of the altitude in the given isosceles triangle --/
theorem altitude_length (t : IsoscelesTriangle) 
  (h1 : t.ab = 10) 
  (h2 : t.bc = 16) : 
  t.ad = 6 := by
  sorry

#check altitude_length

end altitude_length_l4072_407215


namespace smaller_angle_is_55_degrees_l4072_407235

/-- A parallelogram with specific angle properties -/
structure SpecialParallelogram where
  /-- The measure of the smaller angle in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger angle in degrees -/
  larger_angle : ℝ
  /-- The length of the parallelogram -/
  length : ℝ
  /-- The width of the parallelogram -/
  width : ℝ
  /-- The larger angle exceeds the smaller angle by 70 degrees -/
  angle_difference : larger_angle = smaller_angle + 70
  /-- Consecutive angles in a parallelogram are supplementary -/
  supplementary : smaller_angle + larger_angle = 180
  /-- The length is three times the width -/
  length_width_ratio : length = 3 * width

/-- Theorem: In a parallelogram where one angle exceeds the other by 70 degrees,
    the measure of the smaller angle is 55 degrees -/
theorem smaller_angle_is_55_degrees (p : SpecialParallelogram) : p.smaller_angle = 55 := by
  sorry

end smaller_angle_is_55_degrees_l4072_407235


namespace quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l4072_407209

-- Problem 1
theorem quadratic_inequality_solution_sets (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, c*x^2 - 2*x + a < 0 ↔ -2 < x ∧ x < 3) :=
sorry

-- Problem 2
theorem quadratic_inequality_parameter_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) ↔ m < 4 :=
sorry

end quadratic_inequality_solution_sets_quadratic_inequality_parameter_range_l4072_407209


namespace solve_for_b_l4072_407254

theorem solve_for_b (m a k c d b : ℝ) (h : m = (k * c * a * b) / (k * a - d)) :
  b = (m * k * a - m * d) / (k * c * a) := by
  sorry

end solve_for_b_l4072_407254


namespace base6_addition_theorem_l4072_407212

/-- Represents a number in base 6 as a list of digits (least significant first) -/
def Base6 := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Adds two base 6 numbers -/
noncomputable def base6_add (a b : Base6) : Base6 :=
  sorry

theorem base6_addition_theorem :
  let a : Base6 := [2, 3, 5, 4]  -- 4532₆
  let b : Base6 := [2, 1, 4, 3]  -- 3412₆
  let result : Base6 := [4, 1, 4, 0, 1]  -- 10414₆
  base6_add a b = result := by sorry

end base6_addition_theorem_l4072_407212


namespace collinear_points_m_value_l4072_407221

/-- Given three points A, B, and C in 2D space, determines if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(1,2), B(3,m), and C(7,m+6) are collinear, then m = 5 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (1, 2) (3, m) (7, m + 6) → m = 5 := by
  sorry

end collinear_points_m_value_l4072_407221


namespace consecutive_binomial_ratio_l4072_407220

theorem consecutive_binomial_ratio (n k : ℕ) : 
  n > k → 
  (n.choose k : ℚ) / (n.choose (k + 1)) = 1 / 3 →
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2)) = 1 / 2 →
  n + k = 13 := by
  sorry

end consecutive_binomial_ratio_l4072_407220


namespace inscribed_square_side_length_l4072_407256

/-- A right triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ
  right_triangle : ab^2 + bc^2 = ac^2
  ab_eq : ab = 6
  bc_eq : bc = 8
  ac_eq : ac = 10

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.ac
  on_ab : side_length ≤ t.ab
  on_bc : side_length ≤ t.bc

/-- The theorem stating that the side length of the inscribed square is 120/37 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 120 / 37 := by sorry

end inscribed_square_side_length_l4072_407256


namespace ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l4072_407270

theorem ending_number_divisible_by_three (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_divisible := start + (divisor - start % divisor) % divisor
  first_divisible + (count - 1) * divisor

theorem eleven_numbers_divisible_by_three : 
  ending_number_divisible_by_three 10 11 3 = 42 := by
  sorry

end ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l4072_407270


namespace probability_three_defective_l4072_407234

/-- Represents the probability of selecting a defective smartphone from a category. -/
structure CategoryProbability where
  total : ℕ
  defective : ℕ
  probability : ℚ
  valid : probability = defective / total

/-- Represents the data for the smartphone shipment. -/
structure ShipmentData where
  premium : CategoryProbability
  standard : CategoryProbability
  basic : CategoryProbability

/-- The probability of selecting three defective smartphones, one from each category. -/
def probabilityAllDefective (data : ShipmentData) : ℚ :=
  data.premium.probability * data.standard.probability * data.basic.probability

/-- The given shipment data. -/
def givenShipment : ShipmentData := {
  premium := { total := 120, defective := 26, probability := 26 / 120, valid := by norm_num }
  standard := { total := 160, defective := 68, probability := 68 / 160, valid := by norm_num }
  basic := { total := 60, defective := 30, probability := 30 / 60, valid := by norm_num }
}

/-- Theorem stating that the probability of selecting three defective smartphones
    is equal to 221 / 4800 for the given shipment data. -/
theorem probability_three_defective :
  probabilityAllDefective givenShipment = 221 / 4800 := by
  sorry

end probability_three_defective_l4072_407234


namespace circular_arrangement_size_l4072_407262

/-- Represents a circular arrangement of students and a teacher. -/
structure CircularArrangement where
  total_positions : ℕ
  teacher_position : ℕ

/-- Defines the property of two positions being opposite in the circle. -/
def is_opposite (c : CircularArrangement) (pos1 pos2 : ℕ) : Prop :=
  (pos2 - pos1) % c.total_positions = c.total_positions / 2

/-- The main theorem stating the total number of positions in the arrangement. -/
theorem circular_arrangement_size :
  ∀ (c : CircularArrangement),
    (is_opposite c 6 16) →
    (c.teacher_position ≤ c.total_positions) →
    (c.total_positions = 23) :=
by sorry

end circular_arrangement_size_l4072_407262


namespace quaternary_30012_to_decimal_l4072_407214

/-- Converts a list of digits in base 4 to its decimal representation -/
def quaternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary number 30012 -/
def quaternary_30012 : List Nat := [2, 1, 0, 0, 3]

theorem quaternary_30012_to_decimal :
  quaternary_to_decimal quaternary_30012 = 774 := by
  sorry

end quaternary_30012_to_decimal_l4072_407214


namespace line_perp_plane_condition_l4072_407281

-- Define the types for lines and planes
variable (L P : Type) [NormedAddCommGroup L] [NormedSpace ℝ L] [NormedAddCommGroup P] [NormedSpace ℝ P]

-- Define the perpendicular relation
variable (perpendicular : L → L → Prop)
variable (perpendicular_plane : L → P → Prop)

-- Define the subset relation
variable (subset : L → P → Prop)

-- Theorem statement
theorem line_perp_plane_condition (l m : L) (α : P) 
  (h_subset : subset m α) :
  (∀ l m α, perpendicular_plane l α → perpendicular l m) ∧ 
  (∃ l m α, perpendicular l m ∧ ¬perpendicular_plane l α) :=
sorry

end line_perp_plane_condition_l4072_407281


namespace inequality_proof_l4072_407248

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) ≥ x^2 + y^2 + 1 := by
  sorry

end inequality_proof_l4072_407248


namespace sum_of_first_n_naturals_l4072_407204

theorem sum_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / 2 = 3675 ↔ n = 81 := by sorry

end sum_of_first_n_naturals_l4072_407204


namespace part_one_part_two_l4072_407284

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part (1)
theorem part_one (m : ℝ) : 
  (Set.univ \ A) ∪ B m = Set.univ \ A ↔ m ≤ 4 := by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ (a b : ℝ), (Set.univ \ A) ∩ B m = {x | a ≤ x ∧ x ≤ b} ∧ b - a ≥ 1) ↔ 
  (3 ≤ m ∧ m ≤ 5) := by sorry

end part_one_part_two_l4072_407284


namespace binomial_square_coefficient_l4072_407241

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end binomial_square_coefficient_l4072_407241


namespace complex_fraction_equality_l4072_407276

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 41 / 20 :=
by sorry

end complex_fraction_equality_l4072_407276


namespace problem_statement_l4072_407296

theorem problem_statement (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by
sorry

end problem_statement_l4072_407296


namespace least_number_with_remainder_l4072_407259

theorem least_number_with_remainder (n : ℕ) : 
  (n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) →
  (∀ m : ℕ, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) →
  n = 130 := by
sorry

end least_number_with_remainder_l4072_407259


namespace friday_first_day_over_200_l4072_407213

/-- Represents the days of the week -/
inductive Day
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the number of days after Monday -/
def daysAfterMonday (d : Day) : Nat :=
  match d with
  | Day.monday => 0
  | Day.tuesday => 1
  | Day.wednesday => 2
  | Day.thursday => 3
  | Day.friday => 4
  | Day.saturday => 5
  | Day.sunday => 6

/-- Calculates the number of paperclips on a given day -/
def paperclipsOn (d : Day) : Nat :=
  4 * (3 ^ (daysAfterMonday d))

/-- Theorem: Friday is the first day with more than 200 paperclips -/
theorem friday_first_day_over_200 :
  (∀ d : Day, d ≠ Day.friday → paperclipsOn d ≤ 200) ∧
  paperclipsOn Day.friday > 200 :=
sorry

end friday_first_day_over_200_l4072_407213


namespace real_part_of_z_l4072_407242

theorem real_part_of_z (z : ℂ) (h : z - Complex.abs z = -8 + 12*I) : 
  Complex.re z = 5 := by
  sorry

end real_part_of_z_l4072_407242


namespace dave_initial_apps_l4072_407286

/-- The number of apps Dave had on his phone initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating the initial number of apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end dave_initial_apps_l4072_407286


namespace machine_selling_price_l4072_407293

/-- Calculates the selling price of a machine given its costs and desired profit percentage. -/
def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Proves that the selling price of the machine is 25500 Rs given the specified costs and profit percentage. -/
theorem machine_selling_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

#eval selling_price 11000 5000 1000 50

end machine_selling_price_l4072_407293


namespace rectangle_covers_curve_l4072_407216

/-- A plane curve is a continuous function from a closed interval to ℝ² -/
def PlaneCurve := Set.Icc 0 1 → ℝ × ℝ

/-- The length of a plane curve -/
def curveLength (γ : PlaneCurve) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle covers a curve -/
def covers (r : Rectangle) (γ : PlaneCurve) : Prop := sorry

/-- Main theorem: For any plane curve of length 1, there exists a rectangle of area 1/4 that covers it -/
theorem rectangle_covers_curve (γ : PlaneCurve) (h : curveLength γ = 1) :
  ∃ r : Rectangle, r.area = 1/4 ∧ covers r γ := by sorry

end rectangle_covers_curve_l4072_407216


namespace power_equality_implies_q_eight_l4072_407231

theorem power_equality_implies_q_eight : 16^4 = 4^q → q = 8 := by sorry

end power_equality_implies_q_eight_l4072_407231


namespace cost_of_soft_drink_l4072_407223

/-- The cost of a can of soft drink given the following conditions:
  * 5 boxes of pizza cost $50
  * 6 hamburgers cost $18
  * 20 cans of soft drinks were bought
  * Total spent is $106
-/
theorem cost_of_soft_drink :
  let pizza_cost : ℚ := 50
  let hamburger_cost : ℚ := 18
  let total_cans : ℕ := 20
  let total_spent : ℚ := 106
  let soft_drink_cost : ℚ := (total_spent - pizza_cost - hamburger_cost) / total_cans
  soft_drink_cost = 19/10 := by sorry

end cost_of_soft_drink_l4072_407223


namespace multiple_right_triangles_exist_l4072_407236

/-- A right triangle with a given hypotenuse length and one non-right angle -/
structure RightTriangle where
  hypotenuse : ℝ
  angle : ℝ
  hypotenuse_positive : 0 < hypotenuse
  angle_range : 0 < angle ∧ angle < π / 2

/-- Theorem stating that multiple right triangles can have the same hypotenuse and non-right angle -/
theorem multiple_right_triangles_exist (h : ℝ) (θ : ℝ) 
  (h_pos : 0 < h) (θ_range : 0 < θ ∧ θ < π / 2) :
  ∃ (t1 t2 : RightTriangle), t1 ≠ t2 ∧ 
    t1.hypotenuse = h ∧ t1.angle = θ ∧
    t2.hypotenuse = h ∧ t2.angle = θ :=
sorry

end multiple_right_triangles_exist_l4072_407236


namespace regular_polygon_perimeter_l4072_407228

/-- A regular polygon with side length 2 and interior angles measuring 135° has a perimeter of 16. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (interior_angle : ℝ) :
  n ≥ 3 ∧
  side_length = 2 ∧
  interior_angle = 135 ∧
  (n : ℝ) * (180 - interior_angle) = 360 →
  n * side_length = 16 :=
by sorry

end regular_polygon_perimeter_l4072_407228


namespace prime_even_intersection_l4072_407260

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end prime_even_intersection_l4072_407260


namespace triangular_array_sum_of_digits_l4072_407206

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_sum_of_digits :
  ∀ N : ℕ, N > 0 → N * (N + 1) / 2 = 3003 → sum_of_digits N = 14 :=
by
  sorry

end triangular_array_sum_of_digits_l4072_407206


namespace M_necessary_not_sufficient_for_N_l4072_407225

def M : Set ℝ := {x | |x + 1| < 4}
def N : Set ℝ := {x | x / (x - 3) < 0}

theorem M_necessary_not_sufficient_for_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ b : ℝ, b ∈ M ∧ b ∉ N) := by
  sorry

end M_necessary_not_sufficient_for_N_l4072_407225


namespace infinitely_many_mtrp_numbers_l4072_407282

/-- Sum of digits in decimal representation of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of MTRP-number -/
def is_mtrp_number (m n : ℕ) : Prop :=
  n > 0 ∧ n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_mtrp_numbers (m : ℕ) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_mtrp_number m n := by sorry

end infinitely_many_mtrp_numbers_l4072_407282


namespace ball_probabilities_l4072_407224

-- Define the sample space
def Ω : Type := Unit

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def red : Set Ω := sorry
def black : Set Ω := sorry
def white : Set Ω := sorry
def green : Set Ω := sorry

-- State the theorem
theorem ball_probabilities 
  (h1 : P red = 5/12)
  (h2 : P black = 1/3)
  (h3 : P white = 1/6)
  (h4 : P green = 1/12)
  (h5 : Disjoint red black)
  (h6 : Disjoint red white)
  (h7 : Disjoint red green)
  (h8 : Disjoint black white)
  (h9 : Disjoint black green)
  (h10 : Disjoint white green) :
  (P (red ∪ black) = 3/4) ∧ 
  (P (red ∪ black ∪ white) = 11/12) := by
  sorry

end ball_probabilities_l4072_407224


namespace seven_digit_divisible_by_11_l4072_407201

def is_divisible_by_11 (n : ℕ) : Prop :=
  ∃ k : ℤ, (945 * 10000 + n * 1000 + 631) = 11 * k

theorem seven_digit_divisible_by_11 (n : ℕ) (h : n < 10) :
  is_divisible_by_11 n → n = 3 := by
  sorry

end seven_digit_divisible_by_11_l4072_407201


namespace different_color_chip_probability_l4072_407237

theorem different_color_chip_probability : 
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let blue_chips : ℕ := 5
  let red_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_not_green : ℚ := (blue_chips + red_chips) / total_chips
  let prob_not_blue : ℚ := (green_chips + red_chips) / total_chips
  let prob_not_red : ℚ := (green_chips + blue_chips) / total_chips
  prob_green * prob_not_green + prob_blue * prob_not_blue + prob_red * prob_not_red = 148 / 225 :=
by sorry

end different_color_chip_probability_l4072_407237


namespace distance_between_harper_and_jack_l4072_407210

/-- Represents the distance between two runners at the end of a race. -/
def distance_between (race_length : ℕ) (jack_distance : ℕ) : ℕ :=
  race_length - jack_distance

/-- Proves that the distance between Harper and Jack at the end of the race is 848 meters. -/
theorem distance_between_harper_and_jack :
  let race_length_km : ℕ := 1
  let race_length_m : ℕ := race_length_km * 1000
  let jack_distance : ℕ := 152
  distance_between race_length_m jack_distance = 848 := by
  sorry

end distance_between_harper_and_jack_l4072_407210


namespace hex_sum_equals_451A5_l4072_407271

/-- Represents a hexadecimal digit --/
def HexDigit : Type := Fin 16

/-- Represents a hexadecimal number as a list of digits --/
def HexNumber := List HexDigit

/-- Convert a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : HexNumber := sorry

/-- Convert a hexadecimal number to its natural number representation --/
def fromHex (h : HexNumber) : ℕ := sorry

/-- Addition of hexadecimal numbers --/
def hexAdd (a b : HexNumber) : HexNumber := sorry

theorem hex_sum_equals_451A5 :
  let a := toHex 25  -- 19₁₆
  let b := toHex 12  -- C₁₆
  let c := toHex 432 -- 1B0₁₆
  let d := toHex 929 -- 3A1₁₆
  let e := toHex 47  -- 2F₁₆
  hexAdd a (hexAdd b (hexAdd c (hexAdd d e))) = toHex 283045 -- 451A5₁₆
  := by sorry

end hex_sum_equals_451A5_l4072_407271


namespace divisibility_by_three_l4072_407267

theorem divisibility_by_three (a b : ℤ) (h : 3 ∣ (a * b)) :
  ¬(¬(3 ∣ a) ∧ ¬(3 ∣ b)) := by
  sorry

end divisibility_by_three_l4072_407267


namespace rhombus_area_from_intersecting_strips_l4072_407283

/-- The area of a rhombus formed by two intersecting strips -/
theorem rhombus_area_from_intersecting_strips (α : ℝ) (h_α : 0 < α ∧ α < π) :
  let strip_width : ℝ := 1
  let rhombus_side : ℝ := strip_width / Real.sin α
  let rhombus_area : ℝ := rhombus_side * strip_width
  rhombus_area = 1 / Real.sin α :=
by sorry

end rhombus_area_from_intersecting_strips_l4072_407283


namespace athlete_heartbeats_l4072_407279

/-- The number of heartbeats during a race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proof that the athlete's heart beats 19200 times during the race -/
theorem athlete_heartbeats :
  heartbeats_during_race 160 6 20 = 19200 := by
  sorry

#eval heartbeats_during_race 160 6 20

end athlete_heartbeats_l4072_407279


namespace divisibility_problem_l4072_407218

theorem divisibility_problem (n : ℕ) (h : n = 856) :
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 8) = 24 * k₁ ∧ (n + 8) = 32 * k₂ ∧ (n + 8) = 36 * k₃ ∧ (n + 8) = 3 * k₄) :=
by sorry

end divisibility_problem_l4072_407218


namespace computer_selection_count_l4072_407255

def lenovo_count : ℕ := 4
def crsc_count : ℕ := 5
def total_selection : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem computer_selection_count :
  (choose lenovo_count 1 * choose crsc_count 2) + 
  (choose lenovo_count 2 * choose crsc_count 1) = 70 := by
  sorry

end computer_selection_count_l4072_407255


namespace negation_of_existence_l4072_407219

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - x + 1 = 0) ↔ (∀ x, x^2 - x + 1 ≠ 0) := by sorry

end negation_of_existence_l4072_407219


namespace swanson_class_avg_l4072_407200

/-- The average number of zits per kid in Ms. Swanson's class -/
def swanson_avg : ℝ := 5

/-- The number of kids in Ms. Swanson's class -/
def swanson_kids : ℕ := 25

/-- The number of kids in Mr. Jones' class -/
def jones_kids : ℕ := 32

/-- The average number of zits per kid in Mr. Jones' class -/
def jones_avg : ℝ := 6

/-- The difference in total zits between Mr. Jones' and Ms. Swanson's classes -/
def zit_difference : ℕ := 67

theorem swanson_class_avg : 
  swanson_avg * swanson_kids + zit_difference = jones_avg * jones_kids := by
  sorry

#check swanson_class_avg

end swanson_class_avg_l4072_407200


namespace max_acute_angles_convex_polygon_l4072_407249

-- Define a convex polygon
structure ConvexPolygon where
  n : ℕ  -- number of sides
  convex : Bool  -- property of being convex

-- Define the theorem
theorem max_acute_angles_convex_polygon (p : ConvexPolygon) : 
  p.convex = true →  -- the polygon is convex
  (∃ (sum_exterior_angles : ℝ), sum_exterior_angles = 360) →  -- sum of exterior angles is 360°
  (∀ (i : ℕ) (interior_angle exterior_angle : ℝ), 
    i < p.n → interior_angle + exterior_angle = 180) →  -- interior and exterior angles are supplementary
  (∃ (max_acute : ℕ), max_acute = 3 ∧ 
    ∀ (acute_count : ℕ), acute_count ≤ max_acute) :=
by sorry

end max_acute_angles_convex_polygon_l4072_407249


namespace polynomial_divisibility_l4072_407289

/-- Given polynomials P, Q, R, and S with integer coefficients satisfying the equation
    P(x^5) + x Q(x^5) + x^2 R(x^5) = (1 + x + x^2 + x^3 + x^4) S(x),
    prove that there exists a polynomial p such that P(x) = (x - 1) * p(x). -/
theorem polynomial_divisibility 
  (P Q R S : Polynomial ℤ) 
  (h : P.comp (X^5 : Polynomial ℤ) + X * Q.comp (X^5 : Polynomial ℤ) + X^2 * R.comp (X^5 : Polynomial ℤ) = 
       (1 + X + X^2 + X^3 + X^4) * S) : 
  ∃ p : Polynomial ℤ, P = (X - 1) * p := by
sorry

end polynomial_divisibility_l4072_407289


namespace train_speed_calculation_l4072_407240

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 45 := by sorry

end train_speed_calculation_l4072_407240


namespace max_distance_circle_to_line_l4072_407252

/-- The maximum distance from any point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8 * p.2}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (d : ℝ),
    d = 6 ∧
    ∀ p ∈ circle, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ d :=
by sorry


end max_distance_circle_to_line_l4072_407252


namespace z_power_sum_l4072_407291

theorem z_power_sum (z : ℂ) (h : z = (Real.sqrt 2) / (1 - Complex.I)) : 
  z^100 + z^50 + 1 = Complex.I := by
  sorry

end z_power_sum_l4072_407291


namespace inequality_proof_l4072_407202

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (a * b * c * d) ^ (1/4)) ^ 4 ≥ 
  16 * a * b * c * d * (1 + a) * (1 + b) * (1 + c) * (1 + d) := by
  sorry

end inequality_proof_l4072_407202


namespace stream_speed_calculation_l4072_407244

/-- Represents the speed of a boat in still water (in kmph) -/
def boat_speed : ℝ := 48

/-- Represents the speed of the stream (in kmph) -/
def stream_speed : ℝ := 16

/-- Represents the time ratio of upstream to downstream travel -/
def time_ratio : ℝ := 2

theorem stream_speed_calculation :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = time_ratio :=
by sorry

end stream_speed_calculation_l4072_407244


namespace max_sum_with_constraint_max_sum_achievable_l4072_407233

theorem max_sum_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
by sorry

theorem max_sum_achievable :
  ∃ (x y z : ℚ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 :=
by sorry

end max_sum_with_constraint_max_sum_achievable_l4072_407233


namespace complex_equality_sum_l4072_407205

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_equality_sum (a b : ℝ) (h : (1 + i) * i = a + b * i) : a + b = 0 := by
  sorry

end complex_equality_sum_l4072_407205


namespace tetrahedron_edge_length_is_2_sqrt_5_l4072_407253

-- Define the radius of each ball
def ball_radius : ℝ := 2

-- Define the arrangement of balls
structure BallArrangement where
  bottom_balls : Fin 4 → ℝ × ℝ × ℝ  -- Centers of the four bottom balls
  top_ball : ℝ × ℝ × ℝ              -- Center of the top ball

-- Define the properties of the arrangement
def valid_arrangement (arr : BallArrangement) : Prop :=
  -- Four bottom balls are mutually tangent
  ∀ i j, i ≠ j → ‖arr.bottom_balls i - arr.bottom_balls j‖ = 2 * ball_radius
  -- Top ball is tangent to all bottom balls
  ∧ ∀ i, ‖arr.top_ball - arr.bottom_balls i‖ = 2 * ball_radius

-- Define the tetrahedron circumscribed around the arrangement
def tetrahedron_edge_length (arr : BallArrangement) : ℝ :=
  ‖arr.top_ball - arr.bottom_balls 0‖

-- Theorem statement
theorem tetrahedron_edge_length_is_2_sqrt_5 (arr : BallArrangement) 
  (h : valid_arrangement arr) : 
  tetrahedron_edge_length arr = 2 * Real.sqrt 5 := by
  sorry

end tetrahedron_edge_length_is_2_sqrt_5_l4072_407253


namespace tan_difference_l4072_407211

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end tan_difference_l4072_407211


namespace inequality_proof_l4072_407251

theorem inequality_proof (n : ℕ) (h : n > 1) : (4^n : ℚ) / (n + 1) < (2*n).factorial / (n.factorial ^ 2) := by
  sorry

end inequality_proof_l4072_407251


namespace quadratic_function_properties_l4072_407247

/-- Given a quadratic function f(x) = ax^2 + bx + a satisfying certain conditions,
    prove its expression and range. -/
theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + a
  (∀ x, f (x + 7/4) = f (7/4 - x)) →
  (∃! x, f x = 7 * x + a) →
  (f = λ x ↦ -2 * x^2 + 7 * x - 2) ∧
  (Set.range f = Set.Iic (33/8)) := by
sorry

end quadratic_function_properties_l4072_407247


namespace math_competition_score_l4072_407217

theorem math_competition_score (total_questions n_correct n_wrong n_unanswered : ℕ) 
  (new_score old_score : ℕ) :
  total_questions = 50 ∧ 
  new_score = 150 ∧ 
  old_score = 118 ∧ 
  new_score = 6 * n_correct + 3 * n_unanswered ∧ 
  old_score = 40 + 5 * n_correct - 2 * n_wrong ∧ 
  total_questions = n_correct + n_wrong + n_unanswered →
  n_unanswered = 16 := by
sorry

end math_competition_score_l4072_407217


namespace trig_invariant_poly_characterization_l4072_407287

/-- A real polynomial that satisfies P(cos x) = P(sin x) for all real x -/
def TrigInvariantPoly (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (Real.cos x) = P (Real.sin x)

/-- The main theorem stating the existence of Q for a trig-invariant polynomial P -/
theorem trig_invariant_poly_characterization
  (P : ℝ → ℝ) (hP : TrigInvariantPoly P) :
  ∃ Q : ℝ → ℝ, ∀ X : ℝ, P X = Q (X^4 - X^2) := by
  sorry

end trig_invariant_poly_characterization_l4072_407287


namespace ball_game_bill_l4072_407275

theorem ball_game_bill (num_adults num_children : ℕ) 
  (adult_price child_price : ℚ) : 
  num_adults = 10 → 
  num_children = 11 → 
  adult_price = 8 → 
  child_price = 4 → 
  (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price = 124 := by
  sorry

end ball_game_bill_l4072_407275


namespace percentage_problem_l4072_407299

theorem percentage_problem : ∃ P : ℝ, P = (0.25 * 16 + 2) ∧ P = 6 := by
  sorry

end percentage_problem_l4072_407299


namespace min_value_of_f_in_interval_l4072_407290

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 2

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) (-1) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) (-1) → f y ≥ f x) ∧
  f x = -4 := by
  sorry

end min_value_of_f_in_interval_l4072_407290
