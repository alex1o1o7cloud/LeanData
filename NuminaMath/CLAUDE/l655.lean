import Mathlib

namespace iwatch_price_l655_65598

theorem iwatch_price (iphone_price : ℝ) (iphone_discount : ℝ) (iwatch_discount : ℝ) 
  (cashback : ℝ) (total_cost : ℝ) :
  iphone_price = 800 ∧
  iphone_discount = 0.15 ∧
  iwatch_discount = 0.10 ∧
  cashback = 0.02 ∧
  total_cost = 931 →
  ∃ (iwatch_price : ℝ),
    iwatch_price = 300 ∧
    (1 - cashback) * ((1 - iphone_discount) * iphone_price + 
    (1 - iwatch_discount) * iwatch_price) = total_cost :=
by sorry

end iwatch_price_l655_65598


namespace equation_solution_l655_65529

theorem equation_solution (y : ℝ) : 
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 :=
by
  sorry

end equation_solution_l655_65529


namespace work_completion_time_l655_65504

theorem work_completion_time (x : ℝ) : 
  x > 0 → 
  (8 * (1 / x + 1 / 20) = 14 / 15) → 
  x = 15 := by
sorry

end work_completion_time_l655_65504


namespace solution_t_l655_65543

theorem solution_t (t : ℝ) : 
  Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4) → t = 37/10 := by
  sorry

end solution_t_l655_65543


namespace simple_interest_rate_calculation_l655_65550

theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 10000) 
  (h2 : time = 1) 
  (h3 : interest = 800) : 
  (interest / (principal * time)) * 100 = 8 := by
  sorry

end simple_interest_rate_calculation_l655_65550


namespace reporters_covering_local_politics_l655_65523

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (percent_not_covering_politics : ℚ)
  (h2 : percent_not_covering_politics = 1/2)
  (percent_not_covering_local_politics : ℚ)
  (h3 : percent_not_covering_local_politics = 3/10)
  : (↑total_reporters - (percent_not_covering_politics * ↑total_reporters) -
     (percent_not_covering_local_politics * (↑total_reporters - (percent_not_covering_politics * ↑total_reporters))))
    / ↑total_reporters = 7/20 :=
by sorry

end reporters_covering_local_politics_l655_65523


namespace seven_count_l655_65534

-- Define the range of integers
def IntRange := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Function to count occurrences of a digit in a number
def countDigit (d : ℕ) (n : ℕ) : ℕ := sorry

-- Function to count total occurrences of a digit in a range
def totalOccurrences (d : ℕ) (range : Set ℕ) : ℕ := sorry

-- Theorem statement
theorem seven_count :
  totalOccurrences 7 IntRange = 19 := by sorry

end seven_count_l655_65534


namespace triangle_cosine_problem_l655_65525

theorem triangle_cosine_problem (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given condition
  ((Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C) →
  -- Conclusion
  Real.cos A = Real.sqrt 3 / 3 := by
sorry

end triangle_cosine_problem_l655_65525


namespace car_motorcycle_transaction_result_l655_65564

theorem car_motorcycle_transaction_result :
  let car_selling_price : ℚ := 20000
  let motorcycle_selling_price : ℚ := 10000
  let car_loss_percentage : ℚ := 25 / 100
  let motorcycle_gain_percentage : ℚ := 25 / 100
  let car_cost : ℚ := car_selling_price / (1 - car_loss_percentage)
  let motorcycle_cost : ℚ := motorcycle_selling_price / (1 + motorcycle_gain_percentage)
  let total_cost : ℚ := car_cost + motorcycle_cost
  let total_selling_price : ℚ := car_selling_price + motorcycle_selling_price
  let transaction_result : ℚ := total_cost - total_selling_price
  transaction_result = 4667 / 1 := by sorry

end car_motorcycle_transaction_result_l655_65564


namespace three_squared_sum_equals_three_cubed_l655_65531

theorem three_squared_sum_equals_three_cubed (a : ℕ) :
  3^2 + 3^2 + 3^2 = 3^a → a = 3 := by
sorry

end three_squared_sum_equals_three_cubed_l655_65531


namespace fraction_power_calculation_l655_65571

theorem fraction_power_calculation (x y : ℚ) 
  (hx : x = 2/3) (hy : y = 3/2) : 
  (3/4) * x^8 * y^9 = 9/8 := by
  sorry

end fraction_power_calculation_l655_65571


namespace assignment_count_correct_l655_65593

/-- The number of ways to assign 5 students to 3 universities -/
def assignment_count : ℕ := 150

/-- The number of students to be assigned -/
def num_students : ℕ := 5

/-- The number of universities -/
def num_universities : ℕ := 3

/-- Theorem stating that the number of assignment methods is correct -/
theorem assignment_count_correct :
  (∀ (assignment : Fin num_students → Fin num_universities),
    (∀ u : Fin num_universities, ∃ s : Fin num_students, assignment s = u) →
    (∃ (unique_assignment : Fin num_students → Fin num_universities),
      unique_assignment = assignment)) →
  assignment_count = 150 := by
sorry

end assignment_count_correct_l655_65593


namespace fraction_value_given_equation_l655_65583

theorem fraction_value_given_equation (a b : ℝ) : 
  |5 - a| + (b + 3)^2 = 0 → b / a = -3 / 5 := by
sorry

end fraction_value_given_equation_l655_65583


namespace prime_sum_theorem_l655_65596

theorem prime_sum_theorem (a p q : ℕ) : 
  Nat.Prime a → Nat.Prime p → Nat.Prime q → a < p → a + p = q → a = 2 := by
sorry

end prime_sum_theorem_l655_65596


namespace divisibility_by_nine_l655_65507

theorem divisibility_by_nine : ∃ k : ℤ, 8 * 10^18 + 1^18 = 9 * k := by sorry

end divisibility_by_nine_l655_65507


namespace maximize_x_3_minus_3x_l655_65501

theorem maximize_x_3_minus_3x :
  ∀ x : ℝ, 0 < x → x < 1 → x * (3 - 3 * x) ≤ 3 / 4 ∧
  (x * (3 - 3 * x) = 3 / 4 ↔ x = 1 / 2) :=
by sorry

end maximize_x_3_minus_3x_l655_65501


namespace parabola_equation_l655_65509

/-- The equation of a parabola given its parametric form -/
theorem parabola_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t^2 - 7
  y = (5/9) * x^2 - (20/3) * x + 13 :=
by sorry

end parabola_equation_l655_65509


namespace trajectory_of_point_P_l655_65503

/-- The trajectory of point P given vertices A and B and slope product condition -/
theorem trajectory_of_point_P (x y : ℝ) :
  let A := (0, -Real.sqrt 2)
  let B := (0, Real.sqrt 2)
  let slope_PA := (y - A.2) / (x - A.1)
  let slope_PB := (y - B.2) / (x - B.1)
  x ≠ 0 →
  slope_PA * slope_PB = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end trajectory_of_point_P_l655_65503


namespace coefficient_x4_in_binomial_expansion_l655_65559

/-- The coefficient of x^4 in the binomial expansion of (2x^2 - 1/x)^5 is 80 -/
theorem coefficient_x4_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℚ → ℚ := λ x => 2 * x^2
  let b : ℚ → ℚ := λ x => -1/x
  let coeff : ℕ → ℚ := λ k => (-1)^k * 2^(n-k) * (n.choose k)
  (coeff 2) = 80 := by sorry

end coefficient_x4_in_binomial_expansion_l655_65559


namespace carpenter_needs_eight_more_logs_l655_65511

/-- Represents the carpenter's log and woodblock problem -/
def CarpenterProblem (total_woodblocks : ℕ) (initial_logs : ℕ) (woodblocks_per_log : ℕ) : Prop :=
  let initial_woodblocks := initial_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - initial_woodblocks
  remaining_woodblocks % woodblocks_per_log = 0 ∧
  remaining_woodblocks / woodblocks_per_log = 8

/-- The carpenter needs 8 more logs to reach the required 80 woodblocks -/
theorem carpenter_needs_eight_more_logs :
  CarpenterProblem 80 8 5 := by
  sorry

#check carpenter_needs_eight_more_logs

end carpenter_needs_eight_more_logs_l655_65511


namespace two_digit_multiple_of_35_l655_65538

theorem two_digit_multiple_of_35 (n : ℕ) (h1 : 10 ≤ n ∧ n < 100) (h2 : n % 35 = 0) : 
  n % 10 = 5 :=
sorry

end two_digit_multiple_of_35_l655_65538


namespace peach_count_l655_65533

/-- The number of peaches Sally had initially -/
def initial_peaches : ℕ := 13

/-- The number of peaches Sally picked -/
def picked_peaches : ℕ := 55

/-- The total number of peaches Sally has now -/
def total_peaches : ℕ := initial_peaches + picked_peaches

theorem peach_count : total_peaches = 68 := by
  sorry

end peach_count_l655_65533


namespace distance_center_to_point_l655_65561

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 5

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (8, -3)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_center_to_point :
  distance circle_center given_point = 6 * Real.sqrt 2 := by sorry

end distance_center_to_point_l655_65561


namespace bernoullis_inequality_l655_65520

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x ≥ -1) (h2 : n ≥ 1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end bernoullis_inequality_l655_65520


namespace pure_imaginary_complex_number_l655_65562

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := (x^2 - 1) + (x - 1) * Complex.I
  (∀ r : ℝ, z ≠ r) → x = -1 := by
  sorry

end pure_imaginary_complex_number_l655_65562


namespace roots_equation_value_l655_65576

theorem roots_equation_value (a b : ℝ) : 
  a^2 - a - 3 = 0 ∧ b^2 - b - 3 = 0 →
  2*a^3 + b^2 + 3*a^2 - 11*a - b + 5 = 23 :=
by sorry

end roots_equation_value_l655_65576


namespace element_in_set_given_complement_l655_65541

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_given_complement (M : Finset Nat) 
  (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_given_complement_l655_65541


namespace tangent_circle_rectangle_area_l655_65581

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ
  -- The radius of the tangent circle
  s : ℝ
  -- The circle is tangent to three sides
  tangent_to_sides : length = 2 * s ∧ width = s
  -- The circle passes through the midpoint of the diagonal
  passes_through_midpoint : s = Real.sqrt (s^2 + (length/2)^2)

/-- The area of a TangentCircleRectangle is 2s^2 -/
theorem tangent_circle_rectangle_area (r : TangentCircleRectangle) : 
  r.length * r.width = 2 * r.s^2 := by
  sorry

#check tangent_circle_rectangle_area

end tangent_circle_rectangle_area_l655_65581


namespace gold_coins_puzzle_l655_65512

theorem gold_coins_puzzle (n c : ℕ) 
  (h1 : n = 9 * (c - 2))  -- Condition 1: 9 coins per chest, 2 empty chests
  (h2 : n = 6 * c + 3)    -- Condition 2: 6 coins per chest, 3 coins leftover
  : n = 45 := by
  sorry

end gold_coins_puzzle_l655_65512


namespace two_digit_R_equals_R_plus_two_l655_65535

def R (n : ℕ) : ℕ := 
  (n % 2) + (n % 3) + (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8) + (n % 9)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_R_equals_R_plus_two :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    (∀ n ∈ s, is_two_digit n ∧ R n = R (n + 2)) ∧
    (∀ n, is_two_digit n → R n = R (n + 2) → n ∈ s) :=
sorry

end two_digit_R_equals_R_plus_two_l655_65535


namespace problem_statement_l655_65500

theorem problem_statement (m : ℝ) (h : |m| = m + 1) : (4 * m + 1)^2013 = -1 := by
  sorry

end problem_statement_l655_65500


namespace simplify_sqrt_expression_l655_65524

theorem simplify_sqrt_expression (y : ℝ) (h : y ≥ 5/2) :
  Real.sqrt (y + 2 + 3 * Real.sqrt (2 * y - 5)) - Real.sqrt (y - 2 + Real.sqrt (2 * y - 5)) = Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l655_65524


namespace store_comparison_l655_65567

/-- The number of soccer balls to be purchased -/
def soccer_balls : ℕ := 100

/-- The cost of each soccer ball in yuan -/
def soccer_ball_cost : ℕ := 200

/-- The cost of each basketball in yuan -/
def basketball_cost : ℕ := 80

/-- The cost function for Store A's discount plan -/
def cost_A (x : ℕ) : ℕ := 
  if x ≤ soccer_balls then soccer_balls * soccer_ball_cost
  else soccer_balls * soccer_ball_cost + basketball_cost * (x - soccer_balls)

/-- The cost function for Store B's discount plan -/
def cost_B (x : ℕ) : ℕ := 
  (soccer_balls * soccer_ball_cost + x * basketball_cost) * 4 / 5

theorem store_comparison (x : ℕ) :
  (x = 100 → cost_A x < cost_B x) ∧
  (x > 100 → cost_A x = 80 * x + 12000 ∧ cost_B x = 64 * x + 16000) ∧
  (x = 300 → min (cost_A x) (cost_B x) > 
    cost_A 100 + cost_B 200) := by sorry

#eval cost_A 100
#eval cost_B 100
#eval cost_A 300
#eval cost_B 300
#eval cost_A 100 + cost_B 200

end store_comparison_l655_65567


namespace tangerines_per_box_l655_65528

theorem tangerines_per_box
  (total : ℕ)
  (boxes : ℕ)
  (remaining : ℕ)
  (h1 : total = 29)
  (h2 : boxes = 8)
  (h3 : remaining = 5)
  : (total - remaining) / boxes = 3 :=
by
  sorry

end tangerines_per_box_l655_65528


namespace fraction_equality_l655_65526

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 1 / 3) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = 3 / 2 := by
  sorry

end fraction_equality_l655_65526


namespace simplify_and_substitute_l655_65592

theorem simplify_and_substitute :
  let expression (x : ℝ) := (1 + 1 / x) / ((x^2 - 1) / x)
  expression 2 = 1 := by
  sorry

end simplify_and_substitute_l655_65592


namespace hotel_room_charges_percentage_increase_l655_65532

/-- Proves that if the charge for a single room at hotel P is 70% less than hotel R
    and 10% less than hotel G, then the charge for a single room at hotel R is 170%
    greater than hotel G. -/
theorem hotel_room_charges (P R G : ℝ) 
    (h1 : P = R * 0.3)  -- P is 70% less than R
    (h2 : P = G * 0.9)  -- P is 10% less than G
    : R = G * 2.7 := by
  sorry

/-- Proves that if R = G * 2.7, then R is 170% greater than G. -/
theorem percentage_increase (R G : ℝ) (h : R = G * 2.7) 
    : (R - G) / G * 100 = 170 := by
  sorry

end hotel_room_charges_percentage_increase_l655_65532


namespace students_in_class_g_l655_65554

theorem students_in_class_g (total_students : ℕ) (class_a class_b class_c class_d class_e class_f class_g : ℕ) : 
  total_students = 1500 ∧
  class_a = 188 ∧
  class_b = 115 ∧
  class_c = class_b + 80 ∧
  class_d = 2 * class_b ∧
  class_e = class_a + class_b ∧
  class_f = (class_c + class_d) / 2 ∧
  class_g = total_students - (class_a + class_b + class_c + class_d + class_e + class_f) →
  class_g = 256 :=
by sorry

end students_in_class_g_l655_65554


namespace admission_charge_problem_l655_65578

/-- The admission charge problem -/
theorem admission_charge_problem (child_charge : ℚ) (total_charge : ℚ) (num_children : ℕ) 
  (h1 : child_charge = 3/4)
  (h2 : total_charge = 13/4)
  (h3 : num_children = 3) :
  total_charge - (↑num_children * child_charge) = 1 := by
  sorry

end admission_charge_problem_l655_65578


namespace arithmetic_progression_coverage_l655_65597

/-- Theorem: There exists an integer N = 12 and 11 infinite arithmetic progressions
    with differences 2, 3, 4, ..., 12 such that every natural number belongs to
    at least one of these progressions. -/
theorem arithmetic_progression_coverage : ∃ (N : ℕ) (progressions : Fin (N - 1) → Set ℕ),
  N = 12 ∧
  (∀ i : Fin (N - 1), ∃ d : ℕ, d ≥ 2 ∧ d ≤ N ∧
    progressions i = {n : ℕ | ∃ k : ℕ, n = d * k + (i : ℕ)}) ∧
  (∀ n : ℕ, ∃ i : Fin (N - 1), n ∈ progressions i) :=
sorry

end arithmetic_progression_coverage_l655_65597


namespace johns_weekly_sleep_l655_65517

/-- Calculates the total sleep John got in a week given the specified conditions --/
def johnsTotalSleep (daysInWeek : ℕ) (shortSleepDays : ℕ) (shortSleepHours : ℝ) 
  (recommendedSleep : ℝ) (percentOfRecommended : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - shortSleepDays
  let normalSleepHours := recommendedSleep * percentOfRecommended
  shortSleepDays * shortSleepHours + normalSleepDays * normalSleepHours

/-- Theorem stating that John's total sleep for the week equals 30 hours --/
theorem johns_weekly_sleep :
  johnsTotalSleep 7 2 3 8 0.6 = 30 := by
  sorry

#eval johnsTotalSleep 7 2 3 8 0.6

end johns_weekly_sleep_l655_65517


namespace gcd_1215_1995_l655_65589

theorem gcd_1215_1995 : Nat.gcd 1215 1995 = 15 := by
  sorry

end gcd_1215_1995_l655_65589


namespace parsley_sprigs_left_l655_65574

/-- Calculates the number of parsley sprigs left after decorating plates -/
theorem parsley_sprigs_left
  (initial_sprigs : ℕ)
  (whole_sprig_plates : ℕ)
  (half_sprig_plates : ℕ)
  (h1 : initial_sprigs = 25)
  (h2 : whole_sprig_plates = 8)
  (h3 : half_sprig_plates = 12) :
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 :=
by sorry

end parsley_sprigs_left_l655_65574


namespace book_arrangement_theorem_l655_65546

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 6 books with 3 identical copies results in 120 ways -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

#eval arrange_books 6 3

end book_arrangement_theorem_l655_65546


namespace neg_two_star_neg_one_l655_65522

/-- Custom binary operation ※ -/
def star (a b : ℤ) : ℤ := b^2 - a*b

/-- Theorem stating that (-2) ※ (-1) = -1 -/
theorem neg_two_star_neg_one : star (-2) (-1) = -1 := by
  sorry

end neg_two_star_neg_one_l655_65522


namespace star_3_5_l655_65536

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2 + 3*(a+b)

-- State the theorem
theorem star_3_5 : star 3 5 = 88 := by sorry

end star_3_5_l655_65536


namespace f_upper_bound_and_g_monotonicity_l655_65513

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.log x + 1

def g (a x : ℝ) : ℝ := (f x - f a) / (x - a)

theorem f_upper_bound_and_g_monotonicity :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) ∧
  (∀ c : ℝ, c < -1 → ∃ x : ℝ, x > 0 ∧ f x > 2 * x + c) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → g a x₁ > g a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂)) :=
sorry

end f_upper_bound_and_g_monotonicity_l655_65513


namespace ball_count_l655_65515

theorem ball_count (blue_count : ℕ) (prob_blue : ℚ) (green_count : ℕ) : 
  blue_count = 8 → 
  prob_blue = 1 / 5 → 
  prob_blue = blue_count / (blue_count + green_count) →
  green_count = 32 := by
  sorry

end ball_count_l655_65515


namespace y_value_l655_65588

theorem y_value (x : ℝ) : 
  Real.sqrt ((2008 * x + 2009) / (2010 * x - 2011)) + 
  Real.sqrt ((2008 * x + 2009) / (2011 - 2010 * x)) + 2010 = 2010 := by
  sorry

end y_value_l655_65588


namespace good_iff_mod_three_l655_65530

/-- A number n > 3 is "good" if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass. -/
def IsGood (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

theorem good_iff_mod_three (n : ℕ) : IsGood n ↔ n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end good_iff_mod_three_l655_65530


namespace monster_family_kids_eyes_l655_65573

/-- Represents a monster family with parents and kids -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  total_eyes : ℕ

/-- Calculates the number of eyes each kid has in a monster family -/
def eyes_per_kid (family : MonsterFamily) : ℕ :=
  (family.total_eyes - family.mom_eyes - family.dad_eyes) / family.num_kids

/-- Theorem: In the specific monster family, each kid has 4 eyes -/
theorem monster_family_kids_eyes :
  let family : MonsterFamily := {
    mom_eyes := 1,
    dad_eyes := 3,
    num_kids := 3,
    total_eyes := 16
  }
  eyes_per_kid family = 4 := by sorry

end monster_family_kids_eyes_l655_65573


namespace percent_difference_l655_65539

theorem percent_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.1 * y) : x - y = -10 := by
  sorry

end percent_difference_l655_65539


namespace factor_expression_l655_65590

theorem factor_expression (y : ℝ) : 3 * y^2 - 75 = 3 * (y - 5) * (y + 5) := by
  sorry

end factor_expression_l655_65590


namespace ratio_problem_l655_65545

theorem ratio_problem (p q r s : ℚ) 
  (h1 : p / q = 4)
  (h2 : q / r = 3)
  (h3 : r / s = 1 / 5) :
  s / p = 5 / 12 := by
  sorry

end ratio_problem_l655_65545


namespace cooking_participants_l655_65595

/-- The number of people who practice yoga -/
def yoga : ℕ := 35

/-- The number of people who study weaving -/
def weaving : ℕ := 15

/-- The number of people who study cooking only -/
def cooking_only : ℕ := 7

/-- The number of people who study both cooking and yoga -/
def cooking_and_yoga : ℕ := 5

/-- The number of people who participate in all curriculums -/
def all_curriculums : ℕ := 3

/-- The number of people who study both cooking and weaving -/
def cooking_and_weaving : ℕ := 5

/-- The total number of people who study cooking -/
def total_cooking : ℕ := cooking_only + (cooking_and_yoga - all_curriculums) + (cooking_and_weaving - all_curriculums) + all_curriculums

theorem cooking_participants : total_cooking = 14 := by
  sorry

end cooking_participants_l655_65595


namespace rectangle_area_l655_65552

/-- The area of a rectangle bounded by lines y = 2a, y = 3b, x = 4c, and x = 5d,
    where a, b, c, and d are positive numbers. -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (2 * a - 3 * b) * (5 * d - 4 * c) = 10 * a * d - 8 * a * c - 15 * b * d + 12 * b * c := by
  sorry

end rectangle_area_l655_65552


namespace eight_hash_four_eq_eighteen_l655_65587

-- Define the operation #
def hash (a b : ℚ) : ℚ := 2 * a + a / b

-- Theorem statement
theorem eight_hash_four_eq_eighteen : hash 8 4 = 18 := by
  sorry

end eight_hash_four_eq_eighteen_l655_65587


namespace exists_n_with_specific_digit_sums_l655_65575

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_n_with_specific_digit_sums :
  ∃ n : ℕ, sumOfDigits n = 100 ∧ sumOfDigits (n^3) = 1000000 := by sorry

end exists_n_with_specific_digit_sums_l655_65575


namespace box_surface_area_l655_65591

theorem box_surface_area (side_area1 side_area2 volume : ℝ) 
  (h1 : side_area1 = 120)
  (h2 : side_area2 = 72)
  (h3 : volume = 720) :
  ∃ (length width height : ℝ),
    length * width = side_area1 ∧
    length * height = side_area2 ∧
    length * width * height = volume ∧
    length * width = 120 :=
by sorry

end box_surface_area_l655_65591


namespace bikini_fraction_correct_l655_65502

/-- The fraction of garments that are bikinis at Lindsey's Vacation Wear -/
def bikini_fraction : ℝ := 0.38

/-- The fraction of garments that are trunks at Lindsey's Vacation Wear -/
def trunk_fraction : ℝ := 0.25

/-- The fraction of garments that are either bikinis or trunks at Lindsey's Vacation Wear -/
def bikini_or_trunk_fraction : ℝ := 0.63

/-- Theorem stating that the fraction of garments that are bikinis is correct -/
theorem bikini_fraction_correct :
  bikini_fraction + trunk_fraction = bikini_or_trunk_fraction :=
by sorry

end bikini_fraction_correct_l655_65502


namespace three_digit_divisibility_by_37_l655_65505

theorem three_digit_divisibility_by_37 (A B C : ℕ) (h_three_digit : 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000) (h_divisible : (100 * A + 10 * B + C) % 37 = 0) :
  ∃ M : ℕ, M = 100 * B + 10 * C + A ∧ 100 ≤ M ∧ M < 1000 ∧ M % 37 = 0 := by
  sorry

end three_digit_divisibility_by_37_l655_65505


namespace division_and_subtraction_l655_65516

theorem division_and_subtraction : (12 / (1/6)) - (1/3) = 215/3 := by
  sorry

end division_and_subtraction_l655_65516


namespace golden_state_points_l655_65584

/-- The total points scored by the Golden State Team -/
def golden_state_total (draymond curry kelly durant klay : ℕ) : ℕ :=
  draymond + curry + kelly + durant + klay

/-- Theorem stating the total points of the Golden State Team -/
theorem golden_state_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    golden_state_total draymond curry kelly durant klay = 69 := by
  sorry

end golden_state_points_l655_65584


namespace isosceles_right_triangle_area_l655_65569

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  -- AB and BC are the legs, AC is the hypotenuse
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Angle B is 90°
  angle_B_is_right : AB^2 + BC^2 = AC^2
  -- Triangle is isosceles (AB = BC)
  is_isosceles : AB = BC
  -- Altitude BD is 1 unit
  altitude_BD : ℝ
  altitude_is_one : altitude_BD = 1

-- Theorem statement
theorem isosceles_right_triangle_area
  (t : IsoscelesRightTriangle) : 
  (1/2) * t.AB * t.BC = 1 := by
  sorry

end isosceles_right_triangle_area_l655_65569


namespace solve_exponential_equation_l655_65537

theorem solve_exponential_equation :
  ∃ y : ℝ, (5 : ℝ)^9 = 25^y ∧ y = (9 : ℝ) / 2 := by
  sorry

end solve_exponential_equation_l655_65537


namespace oak_trees_after_planting_l655_65510

def initial_trees : ℕ := 237
def planting_factor : ℕ := 5

theorem oak_trees_after_planting :
  initial_trees + planting_factor * initial_trees = 1422 := by
  sorry

end oak_trees_after_planting_l655_65510


namespace train_speed_and_length_l655_65558

-- Define the bridge length
def bridge_length : ℝ := 1000

-- Define the time to completely cross the bridge
def cross_time : ℝ := 60

-- Define the time spent on the bridge
def bridge_time : ℝ := 40

-- Define the train's speed
def train_speed : ℝ := 20

-- Define the train's length
def train_length : ℝ := 200

theorem train_speed_and_length :
  bridge_length = 1000 ∧ 
  cross_time = 60 ∧ 
  bridge_time = 40 →
  train_speed * cross_time = bridge_length + train_length ∧
  train_speed * bridge_time = bridge_length ∧
  train_speed = 20 ∧
  train_length = 200 := by sorry

end train_speed_and_length_l655_65558


namespace test_score_calculation_l655_65555

theorem test_score_calculation (total_questions : ℕ) (first_half : ℕ) (second_half : ℕ)
  (first_correct_rate : ℚ) (second_correct_rate : ℚ)
  (h1 : total_questions = 80)
  (h2 : first_half = 40)
  (h3 : second_half = 40)
  (h4 : first_correct_rate = 9/10)
  (h5 : second_correct_rate = 19/20)
  (h6 : total_questions = first_half + second_half) :
  ⌊first_correct_rate * first_half⌋ + ⌊second_correct_rate * second_half⌋ = 74 := by
  sorry

end test_score_calculation_l655_65555


namespace pineapple_juice_theorem_l655_65586

/-- Represents the juice bar problem -/
structure JuiceBarProblem where
  total_spent : ℕ
  mango_price : ℕ
  pineapple_price : ℕ
  total_people : ℕ

/-- Calculates the amount spent on pineapple juice -/
def pineapple_juice_spent (problem : JuiceBarProblem) : ℕ :=
  let mango_people := problem.total_people - (problem.total_spent - problem.mango_price * problem.total_people) / (problem.pineapple_price - problem.mango_price)
  let pineapple_people := problem.total_people - mango_people
  pineapple_people * problem.pineapple_price

/-- Theorem stating that the amount spent on pineapple juice is $54 -/
theorem pineapple_juice_theorem (problem : JuiceBarProblem) 
  (h1 : problem.total_spent = 94)
  (h2 : problem.mango_price = 5)
  (h3 : problem.pineapple_price = 6)
  (h4 : problem.total_people = 17) :
  pineapple_juice_spent problem = 54 := by
  sorry

#eval pineapple_juice_spent { total_spent := 94, mango_price := 5, pineapple_price := 6, total_people := 17 }

end pineapple_juice_theorem_l655_65586


namespace herman_breakfast_cost_l655_65566

/-- Calculates the total amount spent on breakfast during a project --/
def total_breakfast_cost (team_size : ℕ) (days_per_week : ℕ) (meal_cost : ℚ) (project_duration : ℕ) : ℚ :=
  (team_size : ℚ) * (days_per_week : ℚ) * meal_cost * (project_duration : ℚ)

/-- Proves that Herman's total breakfast cost for the project is $1,280.00 --/
theorem herman_breakfast_cost :
  let team_size : ℕ := 4  -- Herman and 3 team members
  let days_per_week : ℕ := 5
  let meal_cost : ℚ := 4
  let project_duration : ℕ := 16
  total_breakfast_cost team_size days_per_week meal_cost project_duration = 1280 := by
  sorry

end herman_breakfast_cost_l655_65566


namespace expression_factorization_l655_65553

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 - 9) = 6 * x^4 * (2 * x^2 + 7) := by
  sorry

end expression_factorization_l655_65553


namespace vector_perpendicular_and_obtuse_angle_l655_65579

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define x and y as functions of k
def x (k : ℝ) : Fin 2 → ℝ := ![k - 3, 2*k + 2]
def y : Fin 2 → ℝ := ![10, -4]

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define the theorem
theorem vector_perpendicular_and_obtuse_angle (k : ℝ) :
  (dot_product (x k) y = 0 ↔ k = 19) ∧
  (dot_product (x k) y < 0 ↔ k < 19 ∧ k ≠ -1/3) :=
sorry

end vector_perpendicular_and_obtuse_angle_l655_65579


namespace susan_single_digit_in_ten_steps_l655_65551

/-- Represents a multi-digit number as a list of digits -/
def MultiDigitNumber := List Nat

/-- Represents a position where a plus sign can be inserted -/
def PlusPosition := Nat

/-- Represents a set of positions where plus signs are inserted -/
def PlusPositions := List PlusPosition

/-- Performs one step of Susan's operation -/
def performStep (n : MultiDigitNumber) (positions : PlusPositions) : MultiDigitNumber :=
  sorry

/-- Checks if a number is a single digit -/
def isSingleDigit (n : MultiDigitNumber) : Prop :=
  n.length = 1

/-- Main theorem: Susan can always obtain a single-digit number in at most ten steps -/
theorem susan_single_digit_in_ten_steps (n : MultiDigitNumber) :
  ∃ (steps : List PlusPositions),
    steps.length ≤ 10 ∧
    isSingleDigit (steps.foldl performStep n) :=
  sorry

end susan_single_digit_in_ten_steps_l655_65551


namespace fly_probabilities_l655_65521

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def fly_probability (n m : ℕ) : ℚ :=
  (binomial (n + m) n : ℚ) / (2 ^ (n + m))

def fly_probability_through_segment (n1 m1 n2 m2 : ℕ) : ℚ :=
  ((binomial (n1 + m1) n1 : ℚ) * (binomial (n2 + m2) n2)) / (2 ^ (n1 + m1 + n2 + m2 + 1))

def fly_probability_through_circle (n m r : ℕ) : ℚ :=
  let total_steps := n + m
  let mid_steps := total_steps / 2
  (2 * (binomial mid_steps 2 : ℚ) * (binomial mid_steps (mid_steps - 2)) +
   2 * (binomial mid_steps 3 : ℚ) * (binomial mid_steps (mid_steps - 3)) +
   (binomial mid_steps 4 : ℚ) * (binomial mid_steps (mid_steps - 4))) /
  (2 ^ total_steps)

theorem fly_probabilities :
  fly_probability 8 10 = (binomial 18 8 : ℚ) / (2^18) ∧
  fly_probability_through_segment 5 6 2 4 = ((binomial 11 5 : ℚ) * (binomial 6 2)) / (2^18) ∧
  fly_probability_through_circle 8 10 3 = 
    (2 * (binomial 9 2 : ℚ) * (binomial 9 6) + 
     2 * (binomial 9 3 : ℚ) * (binomial 9 5) + 
     (binomial 9 4 : ℚ) * (binomial 9 4)) / (2^18) := by
  sorry

end fly_probabilities_l655_65521


namespace sum_of_a_and_c_l655_65540

theorem sum_of_a_and_c (a b c r : ℝ) 
  (sum_eq : a + b + c = 114)
  (product_eq : a * b * c = 46656)
  (b_eq : b = a * r)
  (c_eq : c = a * r^2) :
  a + c = 78 := by sorry

end sum_of_a_and_c_l655_65540


namespace function_lower_bound_l655_65556

open Real

theorem function_lower_bound (x : ℝ) (h : x > 0) : Real.exp x - Real.log x > 2 := by
  sorry

end function_lower_bound_l655_65556


namespace shane_chewed_eleven_pieces_l655_65548

def elyse_initial_gum : ℕ := 100
def shane_remaining_gum : ℕ := 14

def rick_gum : ℕ := elyse_initial_gum / 2
def shane_initial_gum : ℕ := rick_gum / 2

def shane_chewed_gum : ℕ := shane_initial_gum - shane_remaining_gum

theorem shane_chewed_eleven_pieces : shane_chewed_gum = 11 := by
  sorry

end shane_chewed_eleven_pieces_l655_65548


namespace simplify_expression_l655_65568

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end simplify_expression_l655_65568


namespace sum_of_prime_factors_l655_65572

def n : ℕ := 240345

theorem sum_of_prime_factors (p : ℕ → Prop) 
  (h_prime : ∀ x, p x ↔ Nat.Prime x) : 
  ∃ (a b c : ℕ), 
    p a ∧ p b ∧ p c ∧ 
    n = a * b * c ∧ 
    a + b + c = 16011 := by
  sorry

end sum_of_prime_factors_l655_65572


namespace two_digit_number_puzzle_l655_65542

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem two_digit_number_puzzle (n : ℕ) :
  is_two_digit n ∧ 
  (digit_sum n) % 3 = 0 ∧ 
  n - 27 = reverse_digits n → 
  n = 63 ∨ n = 96 := by
sorry

end two_digit_number_puzzle_l655_65542


namespace prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l655_65557

/-- The probability of getting at least one head when tossing a fair coin three times -/
theorem prob_at_least_one_head_three_tosses : ℚ :=
  let S := Finset.powerset {1, 2, 3}
  let favorable_outcomes := S.filter (λ s => s.card > 0)
  favorable_outcomes.card / S.card

theorem prob_at_least_one_head_three_tosses_is_seven_eighths :
  prob_at_least_one_head_three_tosses = 7 / 8 := by
  sorry

end prob_at_least_one_head_three_tosses_prob_at_least_one_head_three_tosses_is_seven_eighths_l655_65557


namespace square_perimeters_sum_l655_65514

theorem square_perimeters_sum (x : ℝ) : 
  let area1 := x^2 + 8*x + 16
  let area2 := 4*x^2 - 12*x + 9
  let area3 := 9*x^2 - 6*x + 1
  let perimeter1 := 4 * Real.sqrt area1
  let perimeter2 := 4 * Real.sqrt area2
  let perimeter3 := 4 * Real.sqrt area3
  perimeter1 + perimeter2 + perimeter3 = 48 → x = 2 := by
  sorry

end square_perimeters_sum_l655_65514


namespace parabola_properties_l655_65577

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  h : eq = fun x ↦ a * x^2 + 2 * a * x - 1

/-- Points on the parabola -/
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.eq x = y

/-- Axis of symmetry -/
def AxisOfSymmetry (p : Parabola) : ℝ := -1

/-- Vertex on x-axis condition -/
def VertexOnXAxis (p : Parabola) : Prop :=
  p.a = -1

theorem parabola_properties (p : Parabola) (m y₁ y₂ : ℝ) 
  (hM : PointOnParabola p m y₁) 
  (hN : PointOnParabola p 2 y₂)
  (h_y : y₁ > y₂) :
  (AxisOfSymmetry p = -1) ∧
  (VertexOnXAxis p → p.eq = fun x ↦ -x^2 - 2*x - 1) ∧
  ((p.a > 0 → (m > 2 ∨ m < -4)) ∧ 
   (p.a < 0 → (-4 < m ∧ m < 2))) := by sorry

end parabola_properties_l655_65577


namespace peach_difference_l655_65544

/-- Given a basket of peaches with specific counts for each color, 
    prove the difference between green and red peaches. -/
theorem peach_difference (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end peach_difference_l655_65544


namespace quadratic_function_property_l655_65582

/-- The quadratic function f(x) = x^2 + 1774x + 235 satisfies f(f(x) + x) / f(x) = x^2 + 1776x + 2010 for all x. -/
theorem quadratic_function_property : ∀ x : ℝ,
  let f : ℝ → ℝ := λ x ↦ x^2 + 1774*x + 235
  (f (f x + x)) / (f x) = x^2 + 1776*x + 2010 := by
  sorry

end quadratic_function_property_l655_65582


namespace factor_polynomial_l655_65506

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) := by
  sorry

end factor_polynomial_l655_65506


namespace solve_email_problem_l655_65519

def email_problem (initial_delete : ℕ) (first_receive : ℕ) (second_delete : ℕ) (final_receive : ℕ) (final_count : ℕ) : Prop :=
  ∃ (x : ℕ), 
    initial_delete = 50 ∧
    first_receive = 15 ∧
    second_delete = 20 ∧
    final_receive = 10 ∧
    final_count = 30 ∧
    first_receive + x + final_receive = final_count ∧
    x = 5

theorem solve_email_problem :
  ∃ (initial_delete first_receive second_delete final_receive final_count : ℕ),
    email_problem initial_delete first_receive second_delete final_receive final_count :=
by
  sorry

end solve_email_problem_l655_65519


namespace number_puzzle_l655_65599

theorem number_puzzle (x : ℝ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end number_puzzle_l655_65599


namespace smallest_valid_integer_l655_65570

def decimal_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

def is_valid (n : ℕ) : Prop :=
  1000 < n ∧ n < 2000 ∧ decimal_sum n = binary_sum n

theorem smallest_valid_integer : 
  (∀ m, 1000 < m ∧ m < 1101 → ¬(is_valid m)) ∧ is_valid 1101 := by
  sorry

end smallest_valid_integer_l655_65570


namespace rationalize_denominator_l655_65585

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_denominator_l655_65585


namespace village_population_percentage_l655_65518

theorem village_population_percentage :
  let total_population : ℕ := 24000
  let part_population : ℕ := 23040
  let percentage : ℚ := (part_population : ℚ) / total_population * 100
  percentage = 96 := by
  sorry

end village_population_percentage_l655_65518


namespace farmer_tomatoes_l655_65560

theorem farmer_tomatoes (T : ℕ) : 
  T - 53 + 12 = 136 → T = 71 := by
  sorry

end farmer_tomatoes_l655_65560


namespace line_slope_angle_l655_65549

theorem line_slope_angle (x y : ℝ) : 
  x + Real.sqrt 3 * y = 0 → 
  Real.tan (150 * π / 180) = -(1 / Real.sqrt 3) :=
by sorry

end line_slope_angle_l655_65549


namespace bottom_is_red_l655_65527

/-- Represents the colors of the squares -/
inductive Color
  | R | B | O | Y | G | W | P

/-- Represents a face of the cube -/
structure Face where
  color : Color

/-- Represents the cube configuration -/
structure Cube where
  top : Face
  bottom : Face
  sides : List Face
  outward : Face

/-- Theorem: Given the cube configuration, the bottom face is Red -/
theorem bottom_is_red (cube : Cube)
  (h1 : cube.top.color = Color.W)
  (h2 : cube.outward.color = Color.P)
  (h3 : cube.sides.length = 4)
  (h4 : ∀ c : Color, c ≠ Color.P → c ∈ (cube.top :: cube.bottom :: cube.sides).map Face.color) :
  cube.bottom.color = Color.R :=
sorry

end bottom_is_red_l655_65527


namespace stratified_sampling_theorem_l655_65594

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people to be sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The total number of people in the population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- The total number of people in the sample -/
def totalSample (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportionally representative of the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * totalSample s ∧
  s.middleAged * totalPopulation p = p.middleAged * totalSample s ∧
  s.young * totalPopulation p = p.young * totalSample s

theorem stratified_sampling_theorem (p : Population) (s : Sample) :
  p.elderly = 27 →
  p.middleAged = 54 →
  p.young = 81 →
  totalSample s = 42 →
  isProportionalSample p s →
  s.elderly = 7 ∧ s.middleAged = 14 ∧ s.young = 21 := by
  sorry

end stratified_sampling_theorem_l655_65594


namespace circle_radius_from_perimeter_l655_65565

theorem circle_radius_from_perimeter (perimeter : ℝ) (radius : ℝ) :
  perimeter = 8 ∧ perimeter = 2 * Real.pi * radius → radius = 4 / Real.pi := by
  sorry

end circle_radius_from_perimeter_l655_65565


namespace perfect_square_proof_l655_65563

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end perfect_square_proof_l655_65563


namespace existence_of_m_n_l655_65580

theorem existence_of_m_n (p s : ℕ) (hp : Nat.Prime p) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧
    (m * s % p : ℚ) / p < (n * s % p : ℚ) / p ∧ (n * s % p : ℚ) / p < (s : ℚ) / p) ↔
  ¬(s ∣ p - 1) := by
sorry

end existence_of_m_n_l655_65580


namespace max_digits_product_4digit_3digit_l655_65508

theorem max_digits_product_4digit_3digit : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 → 
    100 ≤ b ∧ b < 1000 → 
    a * b < 10000000 :=
by sorry

end max_digits_product_4digit_3digit_l655_65508


namespace cone_base_radius_l655_65547

/-- Given a sector paper with a central angle of 90° and a radius of 20 cm
    used to form the lateral surface of a cone, the radius of the base of the cone is 5 cm. -/
theorem cone_base_radius (θ : Real) (R : Real) (r : Real) : 
  θ = 90 → R = 20 → 2 * π * r = (θ / 360) * 2 * π * R → r = 5 := by
  sorry

end cone_base_radius_l655_65547
