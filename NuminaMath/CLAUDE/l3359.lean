import Mathlib

namespace NUMINAMATH_CALUDE_least_clock_equivalent_l3359_335988

def clock_equivalent (n : ℕ) : Prop :=
  n > 5 ∧ (n^2 - n) % 12 = 0

theorem least_clock_equivalent : ∃ (n : ℕ), clock_equivalent n ∧ ∀ m, m < n → ¬ clock_equivalent m :=
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_l3359_335988


namespace NUMINAMATH_CALUDE_f_increasing_implies_f_1_ge_25_l3359_335978

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_increasing_implies_f_1_ge_25 (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_f_1_ge_25_l3359_335978


namespace NUMINAMATH_CALUDE_max_product_partition_l3359_335983

/-- Given positive integers k and n with k ≥ n, where k = nq + r (0 ≤ r < n),
    F(k) is the maximum product of n positive integers that sum to k. -/
def F (k n : ℕ+) (h : k ≥ n) : ℕ := by sorry

/-- The quotient when k is divided by n -/
def q (k n : ℕ+) : ℕ := k / n

/-- The remainder when k is divided by n -/
def r (k n : ℕ+) : ℕ := k % n

theorem max_product_partition (k n : ℕ+) (h : k ≥ n) :
  F k n h = (q k n) ^ (n - r k n) * ((q k n) + 1) ^ (r k n) := by sorry

end NUMINAMATH_CALUDE_max_product_partition_l3359_335983


namespace NUMINAMATH_CALUDE_cab_journey_time_l3359_335901

/-- Given a cab that arrives 12 minutes late when traveling at 5/6th of its usual speed,
    prove that its usual journey time is 60 minutes. -/
theorem cab_journey_time : ℝ := by
  -- Let S be the usual speed and T be the usual time
  let S : ℝ := 1  -- We can set S to any positive real number
  let T : ℝ := 60 -- This is what we want to prove

  -- Define the reduced speed
  let reduced_speed : ℝ := (5 / 6) * S

  -- Define the time taken at reduced speed
  let reduced_time : ℝ := T + 12

  -- Check if the speed-time relation holds
  have h : S * T = reduced_speed * reduced_time := by sorry

  -- Prove that T equals 60
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l3359_335901


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3359_335999

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, -1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-12, 5; 8, -3]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-0.8, -2.6; -2.0, 1.8]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3359_335999


namespace NUMINAMATH_CALUDE_no_x_term_iff_k_eq_two_l3359_335966

/-- The polynomial x^2 + (k-2)x - 3 does not contain the term with x if and only if k = 2 -/
theorem no_x_term_iff_k_eq_two (k : ℝ) : 
  (∀ x : ℝ, x^2 + (k-2)*x - 3 = x^2 - 3) ↔ k = 2 := by
sorry

end NUMINAMATH_CALUDE_no_x_term_iff_k_eq_two_l3359_335966


namespace NUMINAMATH_CALUDE_local_politics_coverage_l3359_335920

theorem local_politics_coverage (total_reporters : ℕ) 
  (h1 : total_reporters > 0) 
  (politics_coverage : ℝ) 
  (h2 : politics_coverage = 0.25) 
  (local_politics_non_coverage : ℝ) 
  (h3 : local_politics_non_coverage = 0.2) : 
  (politics_coverage * (1 - local_politics_non_coverage)) * total_reporters / total_reporters = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_local_politics_coverage_l3359_335920


namespace NUMINAMATH_CALUDE_integer_inequalities_result_l3359_335981

theorem integer_inequalities_result (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_integer_inequalities_result_l3359_335981


namespace NUMINAMATH_CALUDE_distinct_grade_assignments_l3359_335940

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- Theorem: The number of distinct ways to assign grades to all students -/
theorem distinct_grade_assignments :
  (num_grades : ℕ) ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_distinct_grade_assignments_l3359_335940


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3359_335918

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3359_335918


namespace NUMINAMATH_CALUDE_luca_drink_cost_l3359_335990

/-- The cost of Luca's lunch items and the total bill -/
structure LunchCost where
  sandwich : ℝ
  discount_rate : ℝ
  avocado : ℝ
  salad : ℝ
  total_bill : ℝ

/-- Calculate the cost of Luca's drink given his lunch costs -/
def drink_cost (lunch : LunchCost) : ℝ :=
  lunch.total_bill - (lunch.sandwich * (1 - lunch.discount_rate) + lunch.avocado + lunch.salad)

/-- Theorem: Given Luca's lunch costs, the cost of his drink is $2 -/
theorem luca_drink_cost :
  let lunch : LunchCost := {
    sandwich := 8,
    discount_rate := 0.25,
    avocado := 1,
    salad := 3,
    total_bill := 12
  }
  drink_cost lunch = 2 := by sorry

end NUMINAMATH_CALUDE_luca_drink_cost_l3359_335990


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l3359_335935

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a point lies on a line segment between two other points -/
def liesBetween (P Q R : Point) : Prop :=
  sorry

/-- Checks if a quadrilateral has an inscribed circle -/
def hasInscribedCircle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_theorem (A B C D E F G H P : Point) 
  (q : Quadrilateral) (h1 : q = Quadrilateral.mk A B C D) 
  (h2 : isConvex q)
  (h3 : liesBetween A E B)
  (h4 : liesBetween B F C)
  (h5 : liesBetween C G D)
  (h6 : liesBetween D H A)
  (h7 : P = sorry) -- P is the intersection of EG and FH
  (h8 : hasInscribedCircle (Quadrilateral.mk H A E P))
  (h9 : hasInscribedCircle (Quadrilateral.mk E B F P))
  (h10 : hasInscribedCircle (Quadrilateral.mk F C G P))
  (h11 : hasInscribedCircle (Quadrilateral.mk G D H P)) :
  hasInscribedCircle q :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l3359_335935


namespace NUMINAMATH_CALUDE_video_votes_total_l3359_335965

/-- Represents the voting system for a video -/
structure VideoVotes where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem: Given the conditions, the total number of votes is 240 -/
theorem video_votes_total (v : VideoVotes) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 120) :
  v.totalVotes = 240 := by
  sorry


end NUMINAMATH_CALUDE_video_votes_total_l3359_335965


namespace NUMINAMATH_CALUDE_repayment_plan_earnings_l3359_335941

def hourly_rate (hour : ℕ) : ℕ :=
  if hour % 8 = 0 then 8 else hour % 8

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem repayment_plan_earnings :
  total_earnings 50 = 219 :=
by sorry

end NUMINAMATH_CALUDE_repayment_plan_earnings_l3359_335941


namespace NUMINAMATH_CALUDE_product_of_conjugates_l3359_335936

theorem product_of_conjugates (x p q : ℝ) :
  (x + p / 2 - Real.sqrt (p^2 / 4 - q)) * (x + p / 2 + Real.sqrt (p^2 / 4 - q)) = x^2 + p * x + q :=
by sorry

end NUMINAMATH_CALUDE_product_of_conjugates_l3359_335936


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3359_335948

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log p / Real.log 8 = Real.log (p + 2*q) / Real.log 24) : 
  q / p = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3359_335948


namespace NUMINAMATH_CALUDE_inequality_proof_l3359_335928

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3359_335928


namespace NUMINAMATH_CALUDE_unique_number_l3359_335970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n / 10) % 10
  let u := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  h = 2 * t ∧           -- hundreds digit is twice the tens digit
  u = 2 * t^3 ∧         -- units digit is double the cube of tens digit
  is_prime (h + t + u)  -- sum of digits is prime

theorem unique_number : ∀ n : ℕ, satisfies_conditions n ↔ n = 212 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l3359_335970


namespace NUMINAMATH_CALUDE_system_solution_l3359_335910

theorem system_solution :
  ∃! (x y : ℚ), (7 * x = -10 - 3 * y) ∧ (4 * x = 5 * y - 32) ∧
  (x = -219 / 88) ∧ (y = 97 / 22) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3359_335910


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3359_335957

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3359_335957


namespace NUMINAMATH_CALUDE_prob_c_not_adjacent_to_ab_l3359_335938

/-- Represents the number of students in the group photo -/
def total_students : ℕ := 7

/-- Represents the probability that student C is not adjacent to student A or B,
    given that A and B stand together and C stands on the edge -/
def probability_not_adjacent : ℚ := 4/5

/-- Theorem stating the probability that student C is not adjacent to student A or B -/
theorem prob_c_not_adjacent_to_ab :
  probability_not_adjacent = 4/5 :=
sorry

end NUMINAMATH_CALUDE_prob_c_not_adjacent_to_ab_l3359_335938


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3359_335914

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3359_335914


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3359_335977

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 5*x ≥ 0}

-- Theorem statement
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Ioc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3359_335977


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3359_335960

theorem quadratic_transformation (p q r : ℤ) : 
  (∀ x, (p * x + q)^2 + r = 4 * x^2 - 16 * x + 15) → 
  p * q = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3359_335960


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3359_335932

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The statement of the problem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 ^ 2 + 6 * a 2 + 2 = 0 →
  a 16 ^ 2 + 6 * a 16 + 2 = 0 →
  (a 2 * a 16 / a 9 = Real.sqrt 2 ∨ a 2 * a 16 / a 9 = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3359_335932


namespace NUMINAMATH_CALUDE_bob_distance_when_met_l3359_335989

/-- The distance between X and Y in miles -/
def total_distance : ℝ := 60

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time difference in hours between Yolanda's and Bob's start -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 30 miles when they met -/
theorem bob_distance_when_met : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    yolanda_rate * (t + time_difference) + bob_rate * t = total_distance ∧ 
    bob_rate * t = 30 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_when_met_l3359_335989


namespace NUMINAMATH_CALUDE_complement_of_A_l3359_335974

-- Define the set A
def A : Set ℝ := {x : ℝ | (x + 1) / (x + 2) ≤ 0}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Ici (-1) ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_l3359_335974


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3359_335946

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3359_335946


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3359_335924

theorem reciprocal_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3359_335924


namespace NUMINAMATH_CALUDE_rational_function_value_l3359_335967

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  ∃ k : ℝ,
    (∀ x, q x = (x + 5) * (x - 1)) ∧
    (∀ x, p x = k * x) ∧
    (p 0 / q 0 = 0) ∧
    (p 4 / q 4 = -1/2)

/-- The main theorem -/
theorem rational_function_value (p q : ℝ → ℝ) 
  (h : rational_function p q) : p (-1) / q (-1) = 27/64 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3359_335967


namespace NUMINAMATH_CALUDE_lego_count_l3359_335944

theorem lego_count (initial : Nat) (lost : Nat) (remaining : Nat) : 
  initial = 380 → lost = 57 → remaining = initial - lost → remaining = 323 := by
  sorry

end NUMINAMATH_CALUDE_lego_count_l3359_335944


namespace NUMINAMATH_CALUDE_total_students_count_l3359_335922

/-- The number of students who wish to go on a scavenger hunting trip -/
def scavenger_students : ℕ := 4000

/-- The number of students who wish to go on a skiing trip -/
def skiing_students : ℕ := 2 * scavenger_students

/-- The total number of students who wish to go on either trip -/
def total_students : ℕ := scavenger_students + skiing_students

theorem total_students_count : total_students = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l3359_335922


namespace NUMINAMATH_CALUDE_domino_arrangements_count_l3359_335956

structure Grid :=
  (rows : Nat)
  (cols : Nat)

structure Domino :=
  (length : Nat)
  (width : Nat)

def count_arrangements (g : Grid) (d : Domino) (num_dominoes : Nat) : Nat :=
  Nat.choose (g.rows + g.cols - 2) (g.cols - 1)

theorem domino_arrangements_count (g : Grid) (d : Domino) (num_dominoes : Nat) :
  g.rows = 6 → g.cols = 4 → d.length = 2 → d.width = 1 → num_dominoes = 4 →
  count_arrangements g d num_dominoes = 126 := by
  sorry

end NUMINAMATH_CALUDE_domino_arrangements_count_l3359_335956


namespace NUMINAMATH_CALUDE_green_marbles_after_replacement_l3359_335971

/-- Represents the number of marbles of each color in a jar -/
structure MarbleJar where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  purple : ℕ
  white : ℕ

/-- Calculates the total number of marbles in the jar -/
def totalMarbles (jar : MarbleJar) : ℕ :=
  jar.red + jar.green + jar.blue + jar.yellow + jar.purple + jar.white

/-- Represents the percentage of each color in the jar -/
structure MarblePercentages where
  red : ℚ
  green : ℚ
  blue : ℚ
  yellow : ℚ
  purple : ℚ

/-- Theorem stating the final number of green marbles after replacement -/
theorem green_marbles_after_replacement (jar : MarbleJar) (percentages : MarblePercentages) :
  percentages.red = 25 / 100 →
  percentages.green = 15 / 100 →
  percentages.blue = 20 / 100 →
  percentages.yellow = 10 / 100 →
  percentages.purple = 15 / 100 →
  jar.white = 35 →
  (jar.red : ℚ) / (totalMarbles jar : ℚ) = percentages.red →
  (jar.green : ℚ) / (totalMarbles jar : ℚ) = percentages.green →
  (jar.blue : ℚ) / (totalMarbles jar : ℚ) = percentages.blue →
  (jar.yellow : ℚ) / (totalMarbles jar : ℚ) = percentages.yellow →
  (jar.purple : ℚ) / (totalMarbles jar : ℚ) = percentages.purple →
  (jar.white : ℚ) / (totalMarbles jar : ℚ) = 1 - (percentages.red + percentages.green + percentages.blue + percentages.yellow + percentages.purple) →
  jar.green + jar.red / 3 = 55 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_after_replacement_l3359_335971


namespace NUMINAMATH_CALUDE_coin_ratio_is_equal_l3359_335934

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in rupees -/
def coinValue (c : CoinType) : Rat :=
  match c with
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

/-- The number of coins of each type -/
def numCoins : Nat := 80

/-- The total value of all coins in rupees -/
def totalValue : Rat := 140

/-- Theorem stating that the ratio of coin counts is 1:1:1 -/
theorem coin_ratio_is_equal :
  let oneRupeeCount := numCoins
  let fiftyPaiseCount := numCoins
  let twentyFivePaiseCount := numCoins
  let totalCalculatedValue := oneRupeeCount * coinValue CoinType.OneRupee +
                              fiftyPaiseCount * coinValue CoinType.FiftyPaise +
                              twentyFivePaiseCount * coinValue CoinType.TwentyFivePaise
  totalCalculatedValue = totalValue →
  oneRupeeCount = fiftyPaiseCount ∧ fiftyPaiseCount = twentyFivePaiseCount :=
by
  sorry

#check coin_ratio_is_equal

end NUMINAMATH_CALUDE_coin_ratio_is_equal_l3359_335934


namespace NUMINAMATH_CALUDE_negation_or_false_implies_and_false_l3359_335903

theorem negation_or_false_implies_and_false (p q : Prop) : 
  ¬(¬(p ∨ q)) → ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_negation_or_false_implies_and_false_l3359_335903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3359_335975

/-- Arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term :
  arithmeticSequence 150 = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3359_335975


namespace NUMINAMATH_CALUDE_gemma_pizza_payment_l3359_335952

/-- The amount of money Gemma gave for her pizza order -/
def amount_given (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (change : ℕ) : ℕ :=
  num_pizzas * price_per_pizza + tip + change

/-- Proof that Gemma gave $50 for her pizza order -/
theorem gemma_pizza_payment :
  amount_given 4 10 5 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gemma_pizza_payment_l3359_335952


namespace NUMINAMATH_CALUDE_pam_miles_walked_l3359_335900

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_count : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps walked given a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_count * p.resets + p.final_reading + p.resets

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Pam --/
theorem pam_miles_walked :
  let p : Pedometer := { max_count := 49999, resets := 50, final_reading := 25000 }
  let steps_per_mile := 1500
  steps_to_miles (total_steps p) steps_per_mile = 1683 := by
  sorry


end NUMINAMATH_CALUDE_pam_miles_walked_l3359_335900


namespace NUMINAMATH_CALUDE_A_profit_share_l3359_335953

-- Define the investments and profit shares
def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500
def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100
def total_profit : ℕ := 12200

-- Theorem to prove A's share of the profit
theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end NUMINAMATH_CALUDE_A_profit_share_l3359_335953


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3359_335947

/-- Represents the number of fish in a sample, given the total population, 
    sample size, and the count of a specific type of fish in the population -/
def stratified_sample_count (population : ℕ) (sample_size : ℕ) (fish_count : ℕ) : ℕ :=
  (fish_count * sample_size) / population

/-- Proves that in a stratified sample of size 20 drawn from a population of 200 fish, 
    where silver carp make up 20 of the population and common carp make up 40 of the population, 
    the number of silver carp and common carp together in the sample is 6 -/
theorem stratified_sample_theorem (total_population : ℕ) (sample_size : ℕ) 
  (silver_carp_count : ℕ) (common_carp_count : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 20) 
  (h3 : silver_carp_count = 20) 
  (h4 : common_carp_count = 40) : 
  stratified_sample_count total_population sample_size silver_carp_count + 
  stratified_sample_count total_population sample_size common_carp_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3359_335947


namespace NUMINAMATH_CALUDE_tea_consumption_l3359_335998

/-- The total number of cups of tea consumed by three merchants -/
def total_cups (s o p : ℝ) : ℝ := s + o + p

/-- Theorem stating that the total cups of tea consumed is 19.5 -/
theorem tea_consumption (s o p : ℝ) 
  (h1 : s + o = 11) 
  (h2 : p + o = 15) 
  (h3 : p + s = 13) : 
  total_cups s o p = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_l3359_335998


namespace NUMINAMATH_CALUDE_olympiad_problem_l3359_335979

theorem olympiad_problem (a b c d : ℕ) 
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_olympiad_problem_l3359_335979


namespace NUMINAMATH_CALUDE_expected_product_1000_flips_l3359_335905

/-- The expected value of the product of heads and tails for n fair coin flips -/
def expected_product (n : ℕ) : ℚ := n * (n - 1) / 4

/-- Theorem: The expected value of the product of heads and tails for 1000 fair coin flips is 249750 -/
theorem expected_product_1000_flips : 
  expected_product 1000 = 249750 := by sorry

end NUMINAMATH_CALUDE_expected_product_1000_flips_l3359_335905


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_three_halves_l3359_335976

theorem sqrt_meaningful_iff_x_geq_three_halves (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_three_halves_l3359_335976


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3359_335907

theorem algebraic_expression_value (a b : ℝ) (h : 5*a + 3*b = -4) :
  -8 - 2*(a + b) - 4*(2*a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3359_335907


namespace NUMINAMATH_CALUDE_no_poly3_satisfies_conditions_l3359_335921

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  degree_three : a ≠ 0

/-- Evaluation of a Poly3 at a point -/
def Poly3.eval (p : Poly3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The conditions that the polynomial must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x, p.eval (x^2) = (p.eval x)^2 ∧
       p.eval (x^2) = p.eval (p.eval x) ∧
       p.eval 1 = 2

theorem no_poly3_satisfies_conditions :
  ¬∃ p : Poly3, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_no_poly3_satisfies_conditions_l3359_335921


namespace NUMINAMATH_CALUDE_centroid_property_l3359_335964

/-- The centroid of a triangle divides each median in the ratio 2:1 -/
def is_centroid (P Q R S : ℝ × ℝ) : Prop :=
  S.1 = (P.1 + Q.1 + R.1) / 3 ∧ S.2 = (P.2 + Q.2 + R.2) / 3

theorem centroid_property :
  let P : ℝ × ℝ := (2, 5)
  let Q : ℝ × ℝ := (9, 3)
  let R : ℝ × ℝ := (4, -4)
  let S : ℝ × ℝ := (x, y)
  is_centroid P Q R S → 9 * x + 4 * y = 151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_property_l3359_335964


namespace NUMINAMATH_CALUDE_three_sixes_probability_l3359_335968

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_2_prob_six : ℚ := 1 / 2
def biased_die_2_prob_other : ℚ := 1 / 10
def biased_die_3_prob_six : ℚ := 3 / 4
def biased_die_3_prob_other : ℚ := 1 / 5  -- (1 - 3/4) / 5

-- Define the probability of choosing each die
def choose_die_prob : ℚ := 1 / 3

-- Define the event of rolling two sixes
def two_sixes_prob (die_prob : ℚ) : ℚ := die_prob * die_prob

-- Define the theorem
theorem three_sixes_probability :
  let total_two_sixes := 
    choose_die_prob * two_sixes_prob fair_die_prob +
    choose_die_prob * two_sixes_prob biased_die_2_prob_six +
    choose_die_prob * two_sixes_prob biased_die_3_prob_six
  let prob_fair_given_two_sixes := 
    (choose_die_prob * two_sixes_prob fair_die_prob) / total_two_sixes
  let prob_biased_2_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_2_prob_six) / total_two_sixes
  let prob_biased_3_given_two_sixes := 
    (choose_die_prob * two_sixes_prob biased_die_3_prob_six) / total_two_sixes
  (prob_fair_given_two_sixes * fair_die_prob + 
   prob_biased_2_given_two_sixes * biased_die_2_prob_six + 
   prob_biased_3_given_two_sixes * biased_die_3_prob_six) = 109 / 148 := by
  sorry

end NUMINAMATH_CALUDE_three_sixes_probability_l3359_335968


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3359_335972

theorem inequality_and_equality_condition (x y : ℝ) :
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3359_335972


namespace NUMINAMATH_CALUDE_binary_1100_equals_12_l3359_335963

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1100_equals_12 :
  binary_to_decimal [false, false, true, true] = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_1100_equals_12_l3359_335963


namespace NUMINAMATH_CALUDE_total_cards_l3359_335939

def sallys_cards (initial : ℕ) (dans_gift : ℕ) (purchased : ℕ) : ℕ :=
  initial + dans_gift + purchased

theorem total_cards : sallys_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3359_335939


namespace NUMINAMATH_CALUDE_angle_value_proof_l3359_335913

theorem angle_value_proof (ABC : ℝ) (x : ℝ) 
  (h1 : ABC = 90)
  (h2 : ABC = 44 + x) : 
  x = 46 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_proof_l3359_335913


namespace NUMINAMATH_CALUDE_sum_is_composite_l3359_335931

theorem sum_is_composite (a b : ℕ) (h : 31 * a = 54 * b) : ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ a + b = k * m := by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3359_335931


namespace NUMINAMATH_CALUDE_cherry_olive_discount_l3359_335987

theorem cherry_olive_discount (cherry_price olives_price bags_count total_cost : ℝ) :
  cherry_price = 5 →
  olives_price = 7 →
  bags_count = 50 →
  total_cost = 540 →
  let original_cost := cherry_price * bags_count + olives_price * bags_count
  let discount_amount := original_cost - total_cost
  let discount_percentage := (discount_amount / original_cost) * 100
  discount_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_cherry_olive_discount_l3359_335987


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3359_335950

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3359_335950


namespace NUMINAMATH_CALUDE_tetragon_diagonals_l3359_335943

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A tetragon is a polygon with 4 sides -/
def tetragon_sides : ℕ := 4

/-- Theorem: The number of diagonals in a tetragon is 2 -/
theorem tetragon_diagonals : num_diagonals tetragon_sides = 2 := by
  sorry

end NUMINAMATH_CALUDE_tetragon_diagonals_l3359_335943


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l3359_335923

theorem price_ratio_theorem (CP : ℝ) (SP1 SP2 : ℝ) 
  (h1 : SP1 = CP * (1 + 0.2))
  (h2 : SP2 = CP * (1 - 0.2)) :
  SP2 / SP1 = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l3359_335923


namespace NUMINAMATH_CALUDE_max_product_of_three_numbers_l3359_335919

theorem max_product_of_three_numbers (n : ℕ+) :
  ∃ (a b c : ℕ), 
    a ∈ Finset.range (3*n + 2) ∧ 
    b ∈ Finset.range (3*n + 2) ∧ 
    c ∈ Finset.range (3*n + 2) ∧ 
    a + b + c = 3*n + 1 ∧
    a * b * c = n^3 + n^2 ∧
    ∀ (x y z : ℕ), 
      x ∈ Finset.range (3*n + 2) → 
      y ∈ Finset.range (3*n + 2) → 
      z ∈ Finset.range (3*n + 2) → 
      x + y + z = 3*n + 1 → 
      x * y * z ≤ n^3 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_three_numbers_l3359_335919


namespace NUMINAMATH_CALUDE_white_balls_count_l3359_335984

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 60 →
  green = 10 →
  yellow = 7 →
  red = 15 →
  purple = 6 →
  prob_not_red_purple = 13/20 →
  ∃ white : ℕ, white = 22 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l3359_335984


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l3359_335955

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 →
  (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l3359_335955


namespace NUMINAMATH_CALUDE_not_divisible_by_61_l3359_335994

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_61_l3359_335994


namespace NUMINAMATH_CALUDE_group_size_proof_l3359_335969

theorem group_size_proof (total_rupees : ℚ) (h1 : total_rupees = 72.25) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) = total_rupees * 100 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3359_335969


namespace NUMINAMATH_CALUDE_students_in_section_A_l3359_335982

/-- The number of students in section A -/
def students_A : ℕ := 26

/-- The number of students in section B -/
def students_B : ℕ := 34

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 50

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 30

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 38.67

theorem students_in_section_A : 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total := by
  sorry

end NUMINAMATH_CALUDE_students_in_section_A_l3359_335982


namespace NUMINAMATH_CALUDE_project_hours_difference_l3359_335908

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 198) 
  (h_pat_kate : ∃ k : ℕ, pat_hours = 2 * k ∧ kate_hours = k)
  (h_pat_mark : ∃ m : ℕ, mark_hours = 3 * pat_hours ∧ pat_hours = m)
  (h_sum : pat_hours + kate_hours + mark_hours = total_hours) :
  mark_hours - kate_hours = 110 :=
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3359_335908


namespace NUMINAMATH_CALUDE_joan_change_l3359_335973

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost payment : ℚ) : 
  cat_toy_cost = 877/100 →
  cage_cost = 1097/100 →
  payment = 20 →
  payment - (cat_toy_cost + cage_cost) = 26/100 := by
sorry

end NUMINAMATH_CALUDE_joan_change_l3359_335973


namespace NUMINAMATH_CALUDE_transaction_fraction_l3359_335993

theorem transaction_fraction (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 81 →
  jade_transactions = cal_transactions + 15 →
  cal_transactions * 3 = anthony_transactions * 2 := by
sorry

end NUMINAMATH_CALUDE_transaction_fraction_l3359_335993


namespace NUMINAMATH_CALUDE_joes_journey_time_l3359_335996

/-- Represents the problem of Joe's journey to school -/
theorem joes_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
  r_w > 0 →
  3 * r_w = 3 * d / 4 →
  (3 + 1 / 4 : ℝ) = 3 + (d / 4) / (4 * r_w) :=
by sorry

end NUMINAMATH_CALUDE_joes_journey_time_l3359_335996


namespace NUMINAMATH_CALUDE_exists_unique_solution_l3359_335992

theorem exists_unique_solution : ∃! x : ℝ, 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_solution_l3359_335992


namespace NUMINAMATH_CALUDE_unique_solution_is_201_l3359_335933

theorem unique_solution_is_201 : ∃! (n : ℕ+), 
  (Finset.sum (Finset.range n) (λ k => 4*k + 1)) / (Finset.sum (Finset.range n) (λ k => 4*(k + 1))) = 100 / 101 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_201_l3359_335933


namespace NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3359_335917

theorem x_minus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 2| = p) (h2 : x < 2) : x - p = 2 - 2*p := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_in_terms_of_p_l3359_335917


namespace NUMINAMATH_CALUDE_han_xin_counting_l3359_335926

theorem han_xin_counting (n : ℕ) : n ≥ 53 ∧ n % 3 = 2 ∧ n % 5 = 3 ∧ n % 7 = 4 →
  ∀ m : ℕ, m < 53 → ¬(m % 3 = 2 ∧ m % 5 = 3 ∧ m % 7 = 4) := by
  sorry

end NUMINAMATH_CALUDE_han_xin_counting_l3359_335926


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l3359_335912

theorem boys_ratio_in_class (p : ℝ) 
  (h1 : p ≥ 0 ∧ p ≤ 1) -- Probability is between 0 and 1
  (h2 : p = 3/4 * (1 - p)) -- Condition from the problem
  : p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l3359_335912


namespace NUMINAMATH_CALUDE_card_area_problem_l3359_335942

theorem card_area_problem (l w : ℝ) (h1 : l = 8) (h2 : w = 3) 
  (h3 : (l - 2) * w = 15) : (l * (w - 2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l3359_335942


namespace NUMINAMATH_CALUDE_total_students_l3359_335929

theorem total_students (group1_count : ℕ) (group1_avg : ℚ)
                       (group2_count : ℕ) (group2_avg : ℚ)
                       (total_avg : ℚ) :
  group1_count = 15 →
  group1_avg = 80 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  total_avg = 84 / 100 →
  group1_count + group2_count = 25 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3359_335929


namespace NUMINAMATH_CALUDE_first_hour_premium_l3359_335991

/-- A psychologist charges different rates for the first hour and additional hours of therapy. -/
structure TherapyRates where
  /-- The charge for the first hour of therapy -/
  first_hour : ℝ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℝ
  /-- The total charge for 5 hours of therapy is $375 -/
  five_hour_total : first_hour + 4 * additional_hour = 375
  /-- The total charge for 2 hours of therapy is $174 -/
  two_hour_total : first_hour + additional_hour = 174

/-- The difference between the first hour charge and additional hour charge is $40 -/
theorem first_hour_premium (rates : TherapyRates) : 
  rates.first_hour - rates.additional_hour = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_hour_premium_l3359_335991


namespace NUMINAMATH_CALUDE_matches_for_512_players_l3359_335915

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  matches_eliminate_one : Bool

/-- The number of matches needed to determine the winner in a single-elimination tournament -/
def matches_needed (t : Tournament) : ℕ :=
  t.num_players - 1

/-- Theorem stating the number of matches needed for a 512-player single-elimination tournament -/
theorem matches_for_512_players (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.matches_eliminate_one = true) : 
  matches_needed t = 511 := by
  sorry

end NUMINAMATH_CALUDE_matches_for_512_players_l3359_335915


namespace NUMINAMATH_CALUDE_sevenPointOneTwoThreeBar_eq_fraction_l3359_335961

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 7.123̄ -/
def sevenPointOneTwoThreeBar : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 123 }

theorem sevenPointOneTwoThreeBar_eq_fraction :
  RepeatingDecimal.toRational sevenPointOneTwoThreeBar = 2372 / 333 := by
  sorry

end NUMINAMATH_CALUDE_sevenPointOneTwoThreeBar_eq_fraction_l3359_335961


namespace NUMINAMATH_CALUDE_breaking_sequences_count_l3359_335925

/-- Represents the number of targets in each column -/
def targetDistribution : List Nat := [4, 3, 3]

/-- The total number of targets -/
def totalTargets : Nat := targetDistribution.sum

/-- Calculates the number of different sequences to break all targets -/
def breakingSequences (dist : List Nat) : Nat :=
  Nat.factorial totalTargets / (dist.map Nat.factorial).prod

theorem breaking_sequences_count : breakingSequences targetDistribution = 4200 := by
  sorry

end NUMINAMATH_CALUDE_breaking_sequences_count_l3359_335925


namespace NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l3359_335906

theorem unique_solution_sin_cos_equation :
  ∃! (n : ℕ+), Real.sin (π / (2 * n.val)) * Real.cos (π / (2 * n.val)) = n.val / 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sin_cos_equation_l3359_335906


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3359_335902

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 + Real.sqrt x) = 3 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3359_335902


namespace NUMINAMATH_CALUDE_sequence_seventh_term_l3359_335930

theorem sequence_seventh_term (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end NUMINAMATH_CALUDE_sequence_seventh_term_l3359_335930


namespace NUMINAMATH_CALUDE_petya_bus_catch_l3359_335949

/-- Represents the maximum distance between bus stops that allows Petya to always catch the bus -/
def max_bus_stop_distance (v_p : ℝ) : ℝ :=
  0.12

/-- Theorem stating the maximum distance between bus stops for Petya to always catch the bus -/
theorem petya_bus_catch (v_p : ℝ) (h_v_p : v_p > 0) :
  let v_b := 5 * v_p
  let max_observation_distance := 0.6
  ∀ d : ℝ, d > 0 → d ≤ max_bus_stop_distance v_p →
    (∀ t : ℝ, t ≥ 0 → 
      (v_p * t ≤ d ∧ v_b * t ≤ max_observation_distance) ∨
      (v_p * t ≤ 2 * d ∧ v_b * t ≤ d + max_observation_distance)) :=
by
  sorry

end NUMINAMATH_CALUDE_petya_bus_catch_l3359_335949


namespace NUMINAMATH_CALUDE_no_solutions_prime_factorial_inequality_l3359_335995

theorem no_solutions_prime_factorial_inequality :
  ¬ ∃ (n k : ℕ), Prime n ∧ n ≤ n! - k^n ∧ n! - k^n ≤ k * n :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_prime_factorial_inequality_l3359_335995


namespace NUMINAMATH_CALUDE_marias_candy_l3359_335962

/-- The number of candy pieces Maria ate -/
def pieces_eaten : ℕ := 64

/-- The number of candy pieces Maria has left -/
def pieces_left : ℕ := 3

/-- The initial number of candy pieces Maria had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem marias_candy : initial_pieces = 67 := by
  sorry

end NUMINAMATH_CALUDE_marias_candy_l3359_335962


namespace NUMINAMATH_CALUDE_range_of_m_l3359_335951

/-- Proposition p: The equation x^2+mx+1=0 has exactly two distinct negative roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  ∀ x : ℝ, x^2 + m*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)

/-- Proposition q: The inequality 3^x-m+1≤0 has a real solution -/
def q (m : ℝ) : Prop :=
  ∃ x : ℝ, 3^x - m + 1 ≤ 0

/-- The range of m given the conditions -/
theorem range_of_m :
  ∀ m : ℝ, (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (1 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3359_335951


namespace NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l3359_335945

theorem quadratic_roots_pure_imaginary (m : ℂ) (h : m.re = 0 ∧ m.im ≠ 0) :
  ∃ (z₁ z₂ : ℂ), z₁.re = 0 ∧ z₂.re = 0 ∧ 
  8 * z₁^2 + 4 * Complex.I * z₁ - m = 0 ∧
  8 * z₂^2 + 4 * Complex.I * z₂ - m = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l3359_335945


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_fourth_l3359_335959

theorem sin_alpha_minus_pi_fourth (α : Real) : 
  α ∈ Set.Icc (π) (3*π/2) →   -- α is in the third quadrant
  Real.tan (α + π/4) = -2 →   -- tan(α + π/4) = -2
  Real.sin (α - π/4) = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_fourth_l3359_335959


namespace NUMINAMATH_CALUDE_problem_statement_l3359_335904

theorem problem_statement (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 5) :
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3359_335904


namespace NUMINAMATH_CALUDE_seventh_house_number_l3359_335986

theorem seventh_house_number (k : ℕ) (p : ℕ) : 
  p = 5 →
  k * (p + k - 1) = 2021 →
  p + 2 * (7 - 1) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_house_number_l3359_335986


namespace NUMINAMATH_CALUDE_fraction_simplification_l3359_335980

theorem fraction_simplification (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3359_335980


namespace NUMINAMATH_CALUDE_sports_club_non_players_l3359_335954

theorem sports_club_non_players (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_non_players_l3359_335954


namespace NUMINAMATH_CALUDE_simplify_fraction_l3359_335958

theorem simplify_fraction : 18 * (8 / 12) * (1 / 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3359_335958


namespace NUMINAMATH_CALUDE_person_speed_l3359_335911

/-- Given a person crossing a street, calculate their speed in km/hr -/
theorem person_speed (distance : ℝ) (time : ℝ) (h1 : distance = 720) (h2 : time = 12) :
  distance / 1000 / (time / 60) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_person_speed_l3359_335911


namespace NUMINAMATH_CALUDE_vacuum_time_per_room_l3359_335916

theorem vacuum_time_per_room 
  (battery_life : ℕ) 
  (num_rooms : ℕ) 
  (additional_charges : ℕ) 
  (h1 : battery_life = 10)
  (h2 : num_rooms = 5)
  (h3 : additional_charges = 2) :
  (battery_life * (additional_charges + 1)) / num_rooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_vacuum_time_per_room_l3359_335916


namespace NUMINAMATH_CALUDE_conference_handshakes_l3359_335909

/-- The number of handshakes in a conference with specified conditions -/
def max_handshakes (total_participants : ℕ) (committee_members : ℕ) : ℕ :=
  let non_committee := total_participants - committee_members
  (non_committee * (non_committee - 1)) / 2

/-- Theorem stating the maximum number of handshakes in the given conference scenario -/
theorem conference_handshakes :
  max_handshakes 50 10 = 780 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3359_335909


namespace NUMINAMATH_CALUDE_tank_filling_time_l3359_335927

/-- The time taken to fill a tank with two pipes and a leak -/
theorem tank_filling_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 → 
  pipe2_time = 30 → 
  leak_fraction = 1/3 → 
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3359_335927


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l3359_335997

theorem cats_sold_during_sale 
  (initial_siamese : ℕ) 
  (initial_house : ℕ) 
  (remaining : ℕ) 
  (h1 : initial_siamese = 13) 
  (h2 : initial_house = 5) 
  (h3 : remaining = 8) : 
  initial_siamese + initial_house - remaining = 10 := by
sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l3359_335997


namespace NUMINAMATH_CALUDE_orange_count_l3359_335985

theorem orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + added = initial + added - thrown_away := by
  sorry

-- Example with given values
example : 31 - 9 + 38 = 60 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l3359_335985


namespace NUMINAMATH_CALUDE_hypotenuse_length_from_quadratic_roots_l3359_335937

theorem hypotenuse_length_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 6*a + 4 = 0) →
  (b^2 - 6*b + 4 = 0) →
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_from_quadratic_roots_l3359_335937
