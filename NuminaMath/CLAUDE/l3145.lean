import Mathlib

namespace f_has_three_zeros_l3145_314569

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2017^x + Real.log x / Real.log 2017
  else if x < 0 then -(2017^(-x) + Real.log (-x) / Real.log 2017)
  else 0

theorem f_has_three_zeros :
  (∃! a b c : ℝ, a < 0 ∧ b = 0 ∧ c > 0 ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) :=
sorry

end f_has_three_zeros_l3145_314569


namespace simplification_condition_l3145_314552

theorem simplification_condition (x y k : ℝ) : 
  y = k * x →
  ((x - y) * (2 * x - y) - 3 * x * (2 * x - y) = 5 * x^2) ↔ (k = 3 ∨ k = -3) :=
by sorry

end simplification_condition_l3145_314552


namespace lg_sqrt5_plus_half_lg20_equals_1_l3145_314580

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt5_plus_half_lg20_equals_1 : lg (Real.sqrt 5) + (1/2) * lg 20 = 1 := by
  sorry

end lg_sqrt5_plus_half_lg20_equals_1_l3145_314580


namespace hundred_squared_plus_201_is_composite_l3145_314535

theorem hundred_squared_plus_201_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 100^2 + 201 = a * b := by
  sorry

end hundred_squared_plus_201_is_composite_l3145_314535


namespace expected_heads_value_l3145_314514

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The probability of a coin showing heads after a single flip -/
def prob_heads : ℚ := 1 / 2

/-- The maximum number of flips for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after at most four flips -/
def prob_heads_after_four_flips : ℚ :=
  1 - (1 - prob_heads) ^ max_flips

/-- The expected number of coins showing heads after the series of flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_four_flips

theorem expected_heads_value :
  expected_heads = 93.75 := by sorry

end expected_heads_value_l3145_314514


namespace karthik_weight_average_l3145_314599

def karthik_weight_lower_bound : ℝ := 56
def karthik_weight_upper_bound : ℝ := 57

theorem karthik_weight_average :
  let min_weight := karthik_weight_lower_bound
  let max_weight := karthik_weight_upper_bound
  (min_weight + max_weight) / 2 = 56.5 := by sorry

end karthik_weight_average_l3145_314599


namespace max_leftover_cookies_l3145_314503

theorem max_leftover_cookies (n : ℕ) (h : n > 0) : 
  ∃ (total : ℕ), total % n = n - 1 ∧ ∀ (m : ℕ), m % n ≤ n - 1 :=
by sorry

end max_leftover_cookies_l3145_314503


namespace calculate_expression_l3145_314519

theorem calculate_expression : |(-8 : ℝ)| + (-2011 : ℝ)^0 - 2 * Real.cos (π / 3) + (1 / 2)⁻¹ = 10 := by
  sorry

end calculate_expression_l3145_314519


namespace choose_three_from_eleven_l3145_314555

theorem choose_three_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 3 → Nat.choose n k = 165 := by
  sorry

end choose_three_from_eleven_l3145_314555


namespace complementary_events_l3145_314512

-- Define the sample space for two shots
inductive ShotOutcome
| HH  -- Hit-Hit
| HM  -- Hit-Miss
| MH  -- Miss-Hit
| MM  -- Miss-Miss

-- Define the event of missing both times
def missBoth : Set ShotOutcome := {ShotOutcome.MM}

-- Define the event of hitting at least once
def hitAtLeastOnce : Set ShotOutcome := {ShotOutcome.HH, ShotOutcome.HM, ShotOutcome.MH}

-- Theorem stating that hitAtLeastOnce is the complement of missBoth
theorem complementary_events :
  hitAtLeastOnce = missBoth.compl :=
sorry

end complementary_events_l3145_314512


namespace set_forms_triangle_l3145_314521

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (7, 15, 10) can form a triangle. -/
theorem set_forms_triangle : can_form_triangle 7 15 10 := by
  sorry


end set_forms_triangle_l3145_314521


namespace time_saved_two_pipes_l3145_314597

/-- Represents the time saved when using two pipes instead of one to fill a reservoir -/
theorem time_saved_two_pipes (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) :
  let time_saved := p - (a * p) / (a + b)
  time_saved = (b * p) / (a + b) :=
by sorry

end time_saved_two_pipes_l3145_314597


namespace min_value_expression_l3145_314531

theorem min_value_expression (x y : ℝ) : (x^2*y + x*y^2 - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end min_value_expression_l3145_314531


namespace power_of_128_l3145_314536

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by sorry

end power_of_128_l3145_314536


namespace intersection_and_perpendicular_line_equal_intercepts_lines_l3145_314524

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 4 * y - 15 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (3, 2)

-- Define the perpendicular line l1
def l1 (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the lines l2 with equal intercepts
def l2_1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def l2_2 (x y : ℝ) : Prop := x + y - 5 = 0

theorem intersection_and_perpendicular_line :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (∀ x y, l1 x y → (4 : ℝ) * 3 + 3 * 4 = 0) ∧
  l1 P.1 P.2 :=
sorry

theorem equal_intercepts_lines :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) ∧
  (l2_1 P.1 P.2 ∨ l2_2 P.1 P.2) ∧
  (∃ a ≠ 0, ∀ x y, l2_1 x y → x / a + y / a = 1) ∧
  (∃ a ≠ 0, ∀ x y, l2_2 x y → x / a + y / a = 1) :=
sorry

end intersection_and_perpendicular_line_equal_intercepts_lines_l3145_314524


namespace balance_theorem_l3145_314578

/-- Represents the balance of symbols -/
structure Balance :=
  (star : ℚ)
  (square : ℚ)
  (heart : ℚ)
  (club : ℚ)

/-- The balance equations from the problem -/
def balance_equations (b : Balance) : Prop :=
  3 * b.star + 4 * b.square + b.heart = 12 * b.club ∧
  b.star = b.heart + 2 * b.club

/-- The theorem to prove -/
theorem balance_theorem (b : Balance) :
  balance_equations b →
  3 * b.square + 2 * b.heart = (26 / 9) * b.square :=
by sorry

end balance_theorem_l3145_314578


namespace m_eq_2_necessary_not_sufficient_l3145_314505

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_necessary_not_sufficient :
  (∀ m : ℝ, A m ∩ B = {4} → m = 2 ∨ m = -2) ∧
  (∃ m : ℝ, m = 2 ∧ A m ∩ B = {4}) ∧
  (∃ m : ℝ, m = -2 ∧ A m ∩ B = {4}) :=
sorry

end m_eq_2_necessary_not_sufficient_l3145_314505


namespace present_age_of_b_l3145_314564

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →
  (a = b + 7) →
  b = 37 := by
sorry

end present_age_of_b_l3145_314564


namespace max_value_of_expression_l3145_314540

theorem max_value_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
    (a + b + c)^2 / (a^2 + b^2 + c^2) = 3) ∧ 
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 1 → 
    (p + q + r)^2 / (p^2 + q^2 + r^2) ≤ 3) := by
sorry

end max_value_of_expression_l3145_314540


namespace probability_sum_greater_than_third_roll_l3145_314547

-- Define a die roll as a number between 1 and 6
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of two die rolls
def SumTwoDice (roll1 roll2 : DieRoll) : ℕ := roll1.val + roll2.val

-- Define the probability space
def TotalOutcomes : ℕ := 6 * 6 * 6

-- Define the favorable outcomes
def FavorableOutcomes : ℕ := 51

-- The main theorem
theorem probability_sum_greater_than_third_roll :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 17 / 72 :=
sorry

end probability_sum_greater_than_third_roll_l3145_314547


namespace probability_of_a_l3145_314556

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.28571428571428575) :
  p a = 0.7142857142857143 :=
by sorry

end probability_of_a_l3145_314556


namespace parabola_c_value_l3145_314515

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-1) = 3 →  -- vertex condition
  p.x_coord (-2) = 1 →  -- point condition
  p.c = 1 := by
sorry

end parabola_c_value_l3145_314515


namespace square_root_7396_squared_l3145_314528

theorem square_root_7396_squared : (Real.sqrt 7396)^2 = 7396 := by sorry

end square_root_7396_squared_l3145_314528


namespace jacoby_lottery_winnings_l3145_314532

theorem jacoby_lottery_winnings :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let hours_worked : ℕ := 10
  let cookie_price : ℕ := 4
  let cookies_sold : ℕ := 24
  let lottery_ticket_cost : ℕ := 10
  let remaining_needed : ℕ := 3214
  let sister_gift : ℕ := 500
  let num_sisters : ℕ := 2

  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let total_earnings := job_earnings + cookie_earnings - lottery_ticket_cost
  let total_gifts := sister_gift * num_sisters
  let current_funds := total_earnings + total_gifts
  let lottery_winnings := trip_cost - remaining_needed - current_funds

  lottery_winnings = 500 := by
    sorry

end jacoby_lottery_winnings_l3145_314532


namespace triangle_third_side_range_l3145_314594

theorem triangle_third_side_range (a b x : ℕ) : 
  a = 7 → b = 10 → (∃ (s : ℕ), s = x ∧ 4 ≤ s ∧ s ≤ 16) ↔ 
  (a + b > x ∧ x + a > b ∧ x + b > a) := by sorry

end triangle_third_side_range_l3145_314594


namespace largest_nested_root_l3145_314568

theorem largest_nested_root : 
  let a := (7 : ℝ)^(1/4) * 8^(1/12)
  let b := 8^(1/2) * 7^(1/8)
  let c := 7^(1/2) * 8^(1/8)
  let d := 7^(1/3) * 8^(1/6)
  let e := 8^(1/3) * 7^(1/6)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by sorry

end largest_nested_root_l3145_314568


namespace gcf_of_1260_and_1440_l3145_314591

theorem gcf_of_1260_and_1440 : Nat.gcd 1260 1440 = 180 := by
  sorry

end gcf_of_1260_and_1440_l3145_314591


namespace max_students_is_nine_l3145_314537

/-- Represents the answer choices for each question -/
inductive Choice
| A
| B
| C

/-- Represents a student's answers to all questions -/
def StudentAnswers := Fin 4 → Choice

/-- The property that for any 3 students, there is at least one question where their answers differ -/
def DifferentAnswersExist (answers : Finset StudentAnswers) : Prop :=
  ∀ s1 s2 s3 : StudentAnswers, s1 ∈ answers → s2 ∈ answers → s3 ∈ answers →
    s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    ∃ q : Fin 4, s1 q ≠ s2 q ∧ s2 q ≠ s3 q ∧ s1 q ≠ s3 q

/-- The main theorem stating that the maximum number of students is 9 -/
theorem max_students_is_nine :
  ∃ (answers : Finset StudentAnswers),
    DifferentAnswersExist answers ∧
    answers.card = 9 ∧
    ∀ (larger_set : Finset StudentAnswers),
      larger_set.card > 9 →
      ¬DifferentAnswersExist larger_set :=
sorry

end max_students_is_nine_l3145_314537


namespace exp_2pi_3i_in_second_quadrant_l3145_314501

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (x * Complex.I) = Complex.mk (Real.cos x) (Real.sin x)

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem exp_2pi_3i_in_second_quadrant :
  second_quadrant (Complex.exp ((2 * Real.pi / 3) * Complex.I)) :=
sorry

end exp_2pi_3i_in_second_quadrant_l3145_314501


namespace smaller_number_in_ratio_l3145_314590

theorem smaller_number_in_ratio (n m d u x y : ℝ) : 
  0 < n → n < m → x > 0 → y > 0 → x / y = n / m → x + y + u = d → 
  min x y = n * (d - u) / (n + m) := by
sorry

end smaller_number_in_ratio_l3145_314590


namespace absolute_value_inequality_l3145_314543

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 5) ↔ (x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8) := by
  sorry

end absolute_value_inequality_l3145_314543


namespace water_remaining_l3145_314513

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by
sorry

end water_remaining_l3145_314513


namespace star_seven_three_l3145_314585

def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

theorem star_seven_three : star 7 3 = 16 := by sorry

end star_seven_three_l3145_314585


namespace half_abs_diff_squares_25_20_l3145_314559

theorem half_abs_diff_squares_25_20 : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end half_abs_diff_squares_25_20_l3145_314559


namespace no_real_roots_of_composition_l3145_314538

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_of_composition 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, quadratic a b c x ≠ x) :
  ∀ x : ℝ, quadratic a b c (quadratic a b c x) ≠ x := by
  sorry

end no_real_roots_of_composition_l3145_314538


namespace parabola_focus_coordinates_l3145_314502

/-- The focus of the parabola y² = 4x has coordinates (1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (1, 0) ∧ 
    (∀ (p : ℝ × ℝ), p ∈ parabola → (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - 0)^2 + (p.2 - 0)^2) :=
by
  sorry

end parabola_focus_coordinates_l3145_314502


namespace exam_scores_theorem_l3145_314554

/-- A type representing a student's scores in three tasks -/
structure StudentScores :=
  (task1 : Nat)
  (task2 : Nat)
  (task3 : Nat)

/-- A predicate that checks if all scores are between 0 and 7 -/
def validScores (s : StudentScores) : Prop :=
  0 ≤ s.task1 ∧ s.task1 ≤ 7 ∧
  0 ≤ s.task2 ∧ s.task2 ≤ 7 ∧
  0 ≤ s.task3 ∧ s.task3 ≤ 7

/-- A predicate that checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (s1 s2 : StudentScores) : Prop :=
  s1.task1 ≥ s2.task1 ∧ s1.task2 ≥ s2.task2 ∧ s1.task3 ≥ s2.task3

/-- The main theorem to be proved -/
theorem exam_scores_theorem (students : Finset StudentScores) 
    (h : students.card = 49)
    (h_valid : ∀ s ∈ students, validScores s) :
  ∃ s1 s2 : StudentScores, s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧ scoresGreaterOrEqual s1 s2 :=
sorry

end exam_scores_theorem_l3145_314554


namespace expression_evaluation_l3145_314598

theorem expression_evaluation :
  let a : ℝ := 3 + Real.sqrt 5
  let b : ℝ := 3 - Real.sqrt 5
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * ((a*b) / (a - b)) = 2/3 := by
  sorry

end expression_evaluation_l3145_314598


namespace quadratic_coefficients_l3145_314577

/-- Given a quadratic equation 3x^2 - 4 = -2x, prove that when rearranged 
    into the standard form ax^2 + bx + c = 0, the coefficients are a = 3, b = 2, and c = -4 -/
theorem quadratic_coefficients : 
  ∀ (x : ℝ), 3 * x^2 - 4 = -2 * x → 
  ∃ (a b c : ℝ), a * x^2 + b * x + c = 0 ∧ a = 3 ∧ b = 2 ∧ c = -4 :=
sorry

end quadratic_coefficients_l3145_314577


namespace inequality_implies_bounds_l3145_314570

/-- Custom operation ⊗ defined on ℝ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the relationship between the inequality and the bounds on a -/
theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end inequality_implies_bounds_l3145_314570


namespace no_valid_A_l3145_314575

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 81 % A = 0 ∧ (456200 + A * 10 + 4) % 8 = 0 := by
  sorry

end no_valid_A_l3145_314575


namespace teacher_age_l3145_314576

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 50 →
  student_avg_age = 14 →
  new_avg_age = 15 →
  (num_students * student_avg_age + (65 : ℝ)) / (num_students + 1) = new_avg_age :=
by sorry

end teacher_age_l3145_314576


namespace compute_expression_l3145_314534

theorem compute_expression : 12 * (216 / 3 + 36 / 6 + 16 / 8 + 2) = 984 := by
  sorry

end compute_expression_l3145_314534


namespace correct_arrangements_l3145_314545

/-- The number of different arrangements of representatives for 7 subjects -/
def num_arrangements (num_boys num_girls num_subjects : ℕ) : ℕ :=
  num_boys * num_girls * (Nat.factorial (num_subjects - 2))

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  num_arrangements 4 3 7 = 1440 := by
  sorry

end correct_arrangements_l3145_314545


namespace pinterest_group_average_pins_l3145_314582

/-- The average number of pins contributed per day by each member in a Pinterest group. -/
def average_pins_per_day (
  group_size : ℕ
  ) (
  initial_pins : ℕ
  ) (
  final_pins : ℕ
  ) (
  days : ℕ
  ) (
  deleted_pins_per_week_per_person : ℕ
  ) : ℚ :=
  let total_deleted_pins := (group_size * deleted_pins_per_week_per_person * (days / 7) : ℚ)
  let total_new_pins := (final_pins - initial_pins : ℚ) + total_deleted_pins
  total_new_pins / (group_size * days : ℚ)

/-- Theorem stating that the average number of pins contributed per day is 10. -/
theorem pinterest_group_average_pins :
  average_pins_per_day 20 1000 6600 30 5 = 10 := by
  sorry

end pinterest_group_average_pins_l3145_314582


namespace ellipse_equation_l3145_314510

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a = 2) (h2 : c = Real.sqrt 3) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 := by
sorry

end ellipse_equation_l3145_314510


namespace max_distinct_substrings_l3145_314573

/-- Represents the length of the string -/
def stringLength : ℕ := 66

/-- Represents the number of distinct letters in the string -/
def distinctLetters : ℕ := 4

/-- Calculates the sum of an arithmetic series -/
def arithmeticSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: The maximum number of distinct substrings in a string of length 66
    composed of 4 distinct letters is 2100 -/
theorem max_distinct_substrings :
  distinctLetters +
  distinctLetters^2 +
  (arithmeticSum (stringLength - 2) - arithmeticSum (distinctLetters - 1)) = 2100 := by
  sorry

end max_distinct_substrings_l3145_314573


namespace choose_five_three_l3145_314542

theorem choose_five_three (n : ℕ) (k : ℕ) : n = 5 ∧ k = 3 → Nat.choose n k = 10 := by
  sorry

end choose_five_three_l3145_314542


namespace probability_standard_deck_l3145_314539

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (diamond_cards : Nat)
  (spade_cards : Nat)

/-- A standard deck has 52 cards, 13 diamonds, and 13 spades -/
def standard_deck : Deck :=
  ⟨52, 13, 13⟩

/-- Calculates the probability of drawing a diamond first, then two spades -/
def probability_diamond_then_two_spades (d : Deck) : Rat :=
  (d.diamond_cards : Rat) / d.total_cards *
  (d.spade_cards : Rat) / (d.total_cards - 1) *
  ((d.spade_cards - 1) : Rat) / (d.total_cards - 2)

/-- Theorem: The probability of drawing a diamond first, then two spades from a standard deck is 13/850 -/
theorem probability_standard_deck :
  probability_diamond_then_two_spades standard_deck = 13 / 850 := by
  sorry

end probability_standard_deck_l3145_314539


namespace binomial_product_integer_l3145_314551

theorem binomial_product_integer (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2 * m) * Nat.factorial (2 * n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m + n)) := by
  sorry

end binomial_product_integer_l3145_314551


namespace range_of_f_l3145_314508

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem range_of_f : range = { y | -3 ≤ y ∧ y ≤ 5 } := by sorry

end range_of_f_l3145_314508


namespace lower_half_plane_inequality_l3145_314562

/-- Given a line l passing through points A(2,1) and B(-1,3), 
    the inequality 2x + 3y - 7 ≤ 0 represents the lower half-plane including line l. -/
theorem lower_half_plane_inequality (x y : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (2 - 3*t, 1 + 2*t)}
  (x, y) ∈ l ∨ (∃ p ∈ l, y < p.2) ↔ 2*x + 3*y - 7 ≤ 0 := by
  sorry

end lower_half_plane_inequality_l3145_314562


namespace arctan_equation_solution_l3145_314506

theorem arctan_equation_solution :
  ∀ y : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 → y = 1210 := by
  sorry

end arctan_equation_solution_l3145_314506


namespace c_leq_one_sufficient_not_necessary_l3145_314565

def is_increasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n < a (n + 1)

def sequence_a (c : ℝ) (n : ℕ+) : ℝ :=
  |n.val - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end c_leq_one_sufficient_not_necessary_l3145_314565


namespace three_cards_same_suit_count_l3145_314584

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- The number of ways to select three cards in order from the same suit -/
def ways_to_select_three_same_suit (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1) * (d.cards_per_suit - 2))

/-- Theorem stating the number of ways to select three cards from the same suit -/
theorem three_cards_same_suit_count (d : Deck) 
  (h2 : d.total_cards = 52) 
  (h3 : d.num_suits = 4) 
  (h4 : d.cards_per_suit = 13) : 
  ways_to_select_three_same_suit d = 6864 := by
  sorry

#eval ways_to_select_three_same_suit ⟨52, 4, 13, rfl⟩

end three_cards_same_suit_count_l3145_314584


namespace smallest_r_in_special_progression_l3145_314574

theorem smallest_r_in_special_progression (p q r : ℤ) : 
  p < q → q < r → 
  q^2 = p * r →  -- Geometric progression condition
  2 * q = p + r →  -- Arithmetic progression condition
  ∀ (p' q' r' : ℤ), p' < q' → q' < r' → q'^2 = p' * r' → 2 * q' = p' + r' → r ≤ r' →
  r = 4 := by
sorry

end smallest_r_in_special_progression_l3145_314574


namespace square_root_equality_l3145_314557

theorem square_root_equality (a b : ℝ) : 
  Real.sqrt (6 + a / b) = 6 * Real.sqrt (a / b) → a = 6 ∧ b = 35 := by
  sorry

end square_root_equality_l3145_314557


namespace integer_pairs_satisfying_equation_l3145_314566

theorem integer_pairs_satisfying_equation :
  ∀ x y : ℤ, y ≥ 0 → (x^2 + 2*x*y + Nat.factorial y.toNat = 131) ↔ ((x = 1 ∧ y = 5) ∨ (x = -11 ∧ y = 5)) :=
by sorry

end integer_pairs_satisfying_equation_l3145_314566


namespace rachel_reading_homework_l3145_314579

theorem rachel_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 7)
  (h2 : math_homework = reading_homework + 3) :
  reading_homework = 4 := by
  sorry

end rachel_reading_homework_l3145_314579


namespace product_of_numbers_with_given_sum_and_difference_l3145_314533

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l3145_314533


namespace rationalize_denominator_l3145_314523

theorem rationalize_denominator :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 := by
  sorry

end rationalize_denominator_l3145_314523


namespace brenda_lead_after_turn3_l3145_314567

/-- Represents the score of a player in a Scrabble game -/
structure ScrabbleScore where
  turn1 : ℕ
  turn2 : ℕ
  turn3 : ℕ

/-- Calculates the total score for a player -/
def totalScore (score : ScrabbleScore) : ℕ :=
  score.turn1 + score.turn2 + score.turn3

/-- Represents the Scrabble game between Brenda and David -/
structure ScrabbleGame where
  brenda : ScrabbleScore
  david : ScrabbleScore
  brenda_lead_before_turn3 : ℕ

/-- The Scrabble game instance based on the given problem -/
def game : ScrabbleGame :=
  { brenda := { turn1 := 18, turn2 := 25, turn3 := 15 }
  , david := { turn1 := 10, turn2 := 35, turn3 := 32 }
  , brenda_lead_before_turn3 := 22 }

/-- Theorem stating that Brenda is ahead by 5 points after the third turn -/
theorem brenda_lead_after_turn3 (g : ScrabbleGame) : 
  totalScore g.brenda - totalScore g.david = 5 :=
sorry

end brenda_lead_after_turn3_l3145_314567


namespace min_value_abc_l3145_314561

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 ≥ 18 ∧
  (a^2 + 4*a*b + 9*b^2 + 3*b*c + c^2 = 18 ↔ a = 3 ∧ b = 1/3 ∧ c = 1) :=
by sorry

end min_value_abc_l3145_314561


namespace intersection_circle_passes_through_zero_one_l3145_314500

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  a : ℝ
  b : ℝ
  distinct_intersections : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0 ∧ b ≠ 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (p : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  { point | ∃ (center : ℝ × ℝ) (radius : ℝ),
    (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2 ∧
    (0 - center.1)^2 + (p.b - center.2)^2 = radius^2 ∧
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    x₁^2 + p.a*x₁ + p.b = 0 ∧ x₂^2 + p.a*x₂ + p.b = 0 ∧
    (x₁ - center.1)^2 + (0 - center.2)^2 = radius^2 ∧
    (x₂ - center.1)^2 + (0 - center.2)^2 = radius^2 }

/-- The main theorem stating that the intersection circle passes through (0,1) -/
theorem intersection_circle_passes_through_zero_one (p : TripleIntersectingParabola) :
  (0, 1) ∈ intersection_circle p := by
  sorry

end intersection_circle_passes_through_zero_one_l3145_314500


namespace vector_sum_magnitude_l3145_314527

/-- Given real number m and vectors a and b in ℝ², prove that if a ⊥ b, then |a + b| = √34 -/
theorem vector_sum_magnitude (m : ℝ) (a b : ℝ × ℝ) : 
  a = (m + 2, 1) → 
  b = (1, -2*m) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 34 := by
  sorry


end vector_sum_magnitude_l3145_314527


namespace no_divisible_by_six_l3145_314587

theorem no_divisible_by_six : ∀ z : ℕ, z < 10 → ¬(35000 + z * 100 + 45) % 6 = 0 := by
  sorry

end no_divisible_by_six_l3145_314587


namespace fencing_cost_theorem_l3145_314541

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let breadth := length - 10
  let perimeter := 2 * (length + breadth)
  cost_per_meter * perimeter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  total_fencing_cost 55 26.50 = 5300 := by
  sorry

end fencing_cost_theorem_l3145_314541


namespace power_of_negative_square_l3145_314588

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end power_of_negative_square_l3145_314588


namespace rectangular_solid_volume_range_l3145_314507

theorem rectangular_solid_volume_range (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  2 * (a * b + b * c + a * c) = 48 →
  4 * (a + b + c) = 36 →
  16 ≤ a * b * c ∧ a * b * c ≤ 20 := by
  sorry

end rectangular_solid_volume_range_l3145_314507


namespace validArrangementCount_l3145_314530

/-- Represents a seating arrangement around a rectangular table. -/
structure SeatingArrangement where
  chairs : Fin 15 → Person
  satisfiesConditions : Bool

/-- Represents a person to be seated. -/
inductive Person
  | Man : Person
  | Woman : Person
  | AdditionalPerson : Person

/-- Checks if two positions are adjacent or opposite on the table. -/
def areAdjacentOrOpposite (pos1 pos2 : Fin 15) : Bool := sorry

/-- Checks if the seating arrangement satisfies all conditions. -/
def satisfiesAllConditions (arrangement : SeatingArrangement) : Bool := sorry

/-- Counts the number of valid seating arrangements. -/
def countValidArrangements : Nat := sorry

/-- Theorem stating the number of valid seating arrangements. -/
theorem validArrangementCount : countValidArrangements = 3265920 := by sorry

end validArrangementCount_l3145_314530


namespace monochromatic_right_triangle_exists_l3145_314592

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a color (either 0 or 1) -/
inductive Color where
  | zero : Color
  | one : Color

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is on the side of a triangle -/
def isOnSide (p : Point) (t : Triangle) : Prop := sorry

/-- Represents a coloring of points on the sides of a triangle -/
def Coloring (t : Triangle) := Point → Color

/-- The main theorem to be proved -/
theorem monochromatic_right_triangle_exists 
  (t : Triangle) 
  (h_equilateral : isEquilateral t) 
  (coloring : Coloring t) : 
  ∃ (p q r : Point), 
    isOnSide p t ∧ isOnSide q t ∧ isOnSide r t ∧
    isRightAngled ⟨p, q, r⟩ ∧
    coloring p = coloring q ∧ coloring q = coloring r :=
sorry

end monochromatic_right_triangle_exists_l3145_314592


namespace pudong_exemplifies_ideal_pattern_l3145_314526

-- Define the characteristics of city cluster development
structure CityClusterDevelopment where
  aggregation : Bool
  radiation : Bool
  mutualInfluence : Bool

-- Define the development pattern of Pudong, Shanghai
def pudongDevelopment : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Define the ideal world city cluster development pattern
def idealCityClusterPattern : CityClusterDevelopment :=
  { aggregation := true,
    radiation := true,
    mutualInfluence := true }

-- Theorem statement
theorem pudong_exemplifies_ideal_pattern :
  pudongDevelopment = idealCityClusterPattern :=
by sorry

end pudong_exemplifies_ideal_pattern_l3145_314526


namespace arithmetic_sequence_common_difference_l3145_314546

/-- An arithmetic sequence with first term 13 and fourth term 1 has common difference -4. -/
theorem arithmetic_sequence_common_difference :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence property
  a 1 = 13 →
  a 4 = 1 →
  a 2 - a 1 = -4 := by
sorry

end arithmetic_sequence_common_difference_l3145_314546


namespace no_such_function_exists_l3145_314520

theorem no_such_function_exists :
  ∀ (f : ℤ → Fin 3),
  ∃ (x y : ℤ), (|x - y| = 2 ∨ |x - y| = 3 ∨ |x - y| = 5) ∧ f x = f y :=
by sorry

end no_such_function_exists_l3145_314520


namespace smallest_number_with_given_remainders_l3145_314581

theorem smallest_number_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 1 ∧ 
  x % 4 = 2 ∧ 
  x % 7 = 3 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 1 → y % 4 = 2 → y % 7 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_number_with_given_remainders_l3145_314581


namespace polynomial_Q_value_l3145_314572

-- Define the polynomial
def polynomial (P Q R : ℤ) (z : ℝ) : ℝ :=
  z^5 - 15*z^4 + P*z^3 + Q*z^2 + R*z + 64

-- Define the roots
def roots : List ℤ := [8, 4, 1, 1, 1]

-- Theorem statement
theorem polynomial_Q_value (P Q R : ℤ) :
  (∀ r ∈ roots, polynomial P Q R r = 0) →
  (List.sum roots = 15) →
  (List.prod roots = 64) →
  (∀ r ∈ roots, r > 0) →
  Q = -45 := by
  sorry

end polynomial_Q_value_l3145_314572


namespace square_even_implies_even_l3145_314550

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end square_even_implies_even_l3145_314550


namespace colored_rectangle_iff_same_parity_l3145_314593

/-- Represents the four colors used to color the squares -/
inductive Color
  | Red
  | Yellow
  | Blue
  | Green

/-- Represents a unit square with colored sides -/
structure ColoredSquare where
  top : Color
  right : Color
  bottom : Color
  left : Color
  different_colors : top ≠ right ∧ top ≠ bottom ∧ top ≠ left ∧ 
                     right ≠ bottom ∧ right ≠ left ∧ 
                     bottom ≠ left

/-- Represents a rectangle formed by colored squares -/
structure ColoredRectangle where
  width : ℕ
  height : ℕ
  top_color : Color
  right_color : Color
  bottom_color : Color
  left_color : Color
  different_colors : top_color ≠ right_color ∧ top_color ≠ bottom_color ∧ top_color ≠ left_color ∧ 
                     right_color ≠ bottom_color ∧ right_color ≠ left_color ∧ 
                     bottom_color ≠ left_color

/-- Theorem stating that a colored rectangle can be formed if and only if its side lengths have the same parity -/
theorem colored_rectangle_iff_same_parity (r : ColoredRectangle) :
  (∃ (squares : List (List ColoredSquare)), 
    squares.length = r.height ∧ 
    (∀ row ∈ squares, row.length = r.width) ∧ 
    -- Additional conditions for correct arrangement of squares
    sorry
  ) ↔ 
  (r.width % 2 = r.height % 2) :=
sorry

end colored_rectangle_iff_same_parity_l3145_314593


namespace perpendicular_lines_l3145_314525

/-- Two lines are perpendicular if the sum of the products of their corresponding coefficients is zero -/
def perpendicular (a b c e f g : ℝ) : Prop := a * e + b * f = 0

/-- The line equation x + (m^2 - m)y = 4m - 1 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + (m^2 - m) * y = 4 * m - 1

/-- The line equation 2x - y - 5 = 0 -/
def line2 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem perpendicular_lines (m : ℝ) : 
  perpendicular 1 (m^2 - m) (1 - 4*m) 2 (-1) (-5) → m = -1 ∨ m = 2 :=
by sorry

end perpendicular_lines_l3145_314525


namespace max_red_surface_area_76_l3145_314529

/-- Represents a small cube with two red faces -/
inductive SmallCube
| Adjacent : SmallCube  -- Two adjacent faces are red
| Opposite : SmallCube  -- Two opposite faces are red

/-- Configuration of small cubes -/
structure CubeConfiguration where
  total : Nat
  adjacent : Nat
  opposite : Nat

/-- Represents the large cube assembled from small cubes -/
structure LargeCube where
  config : CubeConfiguration
  side_length : Nat

/-- Calculates the maximum red surface area of the large cube -/
def max_red_surface_area (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the maximum red surface area for the given configuration -/
theorem max_red_surface_area_76 :
  ∀ (lc : LargeCube),
    lc.config.total = 64 ∧
    lc.config.adjacent = 20 ∧
    lc.config.opposite = 44 ∧
    lc.side_length = 4 →
    max_red_surface_area lc = 76 :=
  sorry

end max_red_surface_area_76_l3145_314529


namespace remainder_after_adding_2023_l3145_314583

theorem remainder_after_adding_2023 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 := by
  sorry

end remainder_after_adding_2023_l3145_314583


namespace mile_equals_400_rods_l3145_314560

/-- Conversion rate from miles to furlongs -/
def mile_to_furlong : ℚ := 8

/-- Conversion rate from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

theorem mile_equals_400_rods : rods_in_mile = 400 := by
  sorry

end mile_equals_400_rods_l3145_314560


namespace line_equation_through_points_l3145_314563

/-- Prove that the equation x + 2y - 2 = 0 represents the line passing through points A(0,1) and B(2,0). -/
theorem line_equation_through_points :
  ∀ (x y : ℝ), x + 2*y - 2 = 0 ↔ ∃ (t : ℝ), (x, y) = (1 - t, t) * 2 + (0, 1) :=
by sorry

end line_equation_through_points_l3145_314563


namespace f_one_upper_bound_l3145_314553

/-- A quadratic function f(x) = 2x^2 - mx + 5 where m is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -2],
    then f(1) ≤ 15 -/
theorem f_one_upper_bound (m : ℝ) 
  (h : ∀ x y, x ≤ y → y ≤ -2 → f m x ≥ f m y) : 
  f m 1 ≤ 15 := by
  sorry

end f_one_upper_bound_l3145_314553


namespace article_cost_price_l3145_314516

/-- The cost price of an article that satisfies the given conditions -/
def cost_price : ℝ := 1600

/-- The selling price of the article with a 5% gain -/
def selling_price (c : ℝ) : ℝ := 1.05 * c

/-- The new cost price if bought at 5% less -/
def new_cost_price (c : ℝ) : ℝ := 0.95 * c

/-- The new selling price if sold for 8 less -/
def new_selling_price (c : ℝ) : ℝ := selling_price c - 8

theorem article_cost_price :
  selling_price cost_price = 1.05 * cost_price ∧
  new_cost_price cost_price = 0.95 * cost_price ∧
  new_selling_price cost_price = selling_price cost_price - 8 ∧
  new_selling_price cost_price = 1.1 * new_cost_price cost_price :=
by sorry

end article_cost_price_l3145_314516


namespace vectors_are_orthogonal_l3145_314589

def vector1 : Fin 4 → ℝ := ![2, -4, 3, 1]
def vector2 : Fin 4 → ℝ := ![-3, 1, 4, -2]

theorem vectors_are_orthogonal :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = 0 := by
  sorry

end vectors_are_orthogonal_l3145_314589


namespace avg_people_per_hour_rounded_l3145_314595

/-- The number of people moving to Texas in five days -/
def total_people : ℕ := 5000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving to Texas per hour -/
def avg_people_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_per_hour_rounded :
  round_to_nearest avg_people_per_hour = 42 := by
  sorry

end avg_people_per_hour_rounded_l3145_314595


namespace symmetric_latin_square_diagonal_property_l3145_314509

/-- A square matrix with odd size, filled with numbers 1 to n, where each row and column contains all numbers exactly once, and which is symmetric about the main diagonal. -/
structure SymmetricLatinSquare (n : ℕ) :=
  (matrix : Fin n → Fin n → Fin n)
  (odd : Odd n)
  (latin_square : ∀ (i j : Fin n), ∃! (k : Fin n), matrix i k = j ∧ ∃! (k : Fin n), matrix k j = i)
  (symmetric : ∀ (i j : Fin n), matrix i j = matrix j i)

/-- The main diagonal of a square matrix contains all numbers from 1 to n exactly once. -/
def diagonal_contains_all (n : ℕ) (matrix : Fin n → Fin n → Fin n) : Prop :=
  ∀ (k : Fin n), ∃! (i : Fin n), matrix i i = k

/-- If a SymmetricLatinSquare exists, then its main diagonal contains all numbers from 1 to n exactly once. -/
theorem symmetric_latin_square_diagonal_property {n : ℕ} (sls : SymmetricLatinSquare n) :
  diagonal_contains_all n sls.matrix :=
sorry

end symmetric_latin_square_diagonal_property_l3145_314509


namespace hash_of_hash_of_hash_4_l3145_314549

def hash (N : ℝ) : ℝ := 0.5 * N^2 + 1

theorem hash_of_hash_of_hash_4 : hash (hash (hash 4)) = 862.125 := by
  sorry

end hash_of_hash_of_hash_4_l3145_314549


namespace count_integers_in_range_l3145_314571

theorem count_integers_in_range : 
  (Finset.range (513 - 2)).card = 511 := by sorry

end count_integers_in_range_l3145_314571


namespace trey_kyle_turtle_difference_l3145_314511

/-- Proves that Trey has 60 more turtles than Kyle given the conditions in the problem -/
theorem trey_kyle_turtle_difference : 
  ∀ (kristen trey kris layla tim kyle : ℚ),
  kristen = 24.5 →
  kris = kristen / 3 →
  trey = 8.5 * kris →
  layla = 2 * trey →
  tim = 2 / 3 * kristen →
  kyle = tim / 2 →
  trey - kyle = 60 := by
  sorry

end trey_kyle_turtle_difference_l3145_314511


namespace taxi_ride_cost_l3145_314548

def taxi_cost (initial_charge : ℚ) (additional_charge : ℚ) (passenger_fee : ℚ) (luggage_fee : ℚ)
              (distance : ℚ) (passengers : ℕ) (luggage : ℕ) : ℚ :=
  let distance_quarters := (distance - 1/4).ceil * 4
  let distance_charge := initial_charge + additional_charge * (distance_quarters - 1)
  let passenger_charge := passenger_fee * (passengers - 1)
  let luggage_charge := luggage_fee * luggage
  distance_charge + passenger_charge + luggage_charge

theorem taxi_ride_cost :
  taxi_cost 5 0.6 1 2 12.4 3 2 = 39.8 := by
  sorry

end taxi_ride_cost_l3145_314548


namespace sum_of_roots_quadratic_l3145_314522

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y z : ℝ, x^2 - 2*x - 35 = 0 ∧ y + z = 2 ∧ (x = y ∨ x = z) := by
  sorry

end sum_of_roots_quadratic_l3145_314522


namespace equal_expressions_l3145_314517

theorem equal_expressions : 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) ∧ 
  (-2^3 = (-2)^3) := by
  sorry

end equal_expressions_l3145_314517


namespace garden_fencing_l3145_314518

theorem garden_fencing (garden_area : ℝ) (extension : ℝ) : 
  garden_area = 784 →
  extension = 10 →
  (4 * (Real.sqrt garden_area + extension)) = 152 := by
  sorry

end garden_fencing_l3145_314518


namespace modulus_of_two_over_one_plus_i_l3145_314544

open Complex

theorem modulus_of_two_over_one_plus_i :
  let z : ℂ := 2 / (1 + I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end modulus_of_two_over_one_plus_i_l3145_314544


namespace opponent_scissors_is_random_event_l3145_314586

/-- Represents the possible choices in the game of rock, paper, scissors -/
inductive Choice
  | Rock
  | Paper
  | Scissors

/-- Represents a game of rock, paper, scissors -/
structure RockPaperScissors where
  opponentChoice : Choice

/-- Defines what it means for an event to be random in this context -/
def isRandomEvent (game : RockPaperScissors → Prop) : Prop :=
  ∀ (c : Choice), ∃ (g : RockPaperScissors), g.opponentChoice = c ∧ game g

/-- The main theorem: opponent choosing scissors is a random event -/
theorem opponent_scissors_is_random_event :
  isRandomEvent (λ g => g.opponentChoice = Choice.Scissors) :=
sorry

end opponent_scissors_is_random_event_l3145_314586


namespace tommy_wheel_count_l3145_314558

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := (trucks_seen * truck_wheels) + (cars_seen * car_wheels)

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end tommy_wheel_count_l3145_314558


namespace factorization_problem1_l3145_314596

theorem factorization_problem1 (a b : ℝ) :
  4 * a^2 + 12 * a * b + 9 * b^2 = (2*a + 3*b)^2 := by sorry

end factorization_problem1_l3145_314596


namespace absolute_value_equation_l3145_314504

theorem absolute_value_equation (x : ℝ) : 
  |(-2 : ℝ)| * (|(-25 : ℝ)| - |x|) = 40 ↔ |x| = 5 := by sorry

end absolute_value_equation_l3145_314504
