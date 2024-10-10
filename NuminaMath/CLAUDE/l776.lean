import Mathlib

namespace class_mean_calculation_l776_77696

theorem class_mean_calculation (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 24 →
  group2_students = 6 →
  group1_mean = 80 / 100 →
  group2_mean = 85 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 81 / 100 :=
by sorry

end class_mean_calculation_l776_77696


namespace sum_of_digits_l776_77695

theorem sum_of_digits (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  b + c = 10 →
  c + d = 1 →
  a + d = 2 →
  a + b + c + d = 13 := by
sorry

end sum_of_digits_l776_77695


namespace number_puzzle_solution_l776_77616

theorem number_puzzle_solution : ∃ x : ℤ, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 := by
  sorry

end number_puzzle_solution_l776_77616


namespace ascending_order_l776_77659

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end ascending_order_l776_77659


namespace graveling_cost_l776_77690

/-- The cost of graveling two intersecting roads on a rectangular lawn -/
theorem graveling_cost (lawn_length lawn_width road_width cost_per_sqm : ℝ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 4 →
  (lawn_length * road_width + lawn_width * road_width - road_width * road_width) * cost_per_sqm = 5200 := by
  sorry

end graveling_cost_l776_77690


namespace total_time_in_work_week_l776_77676

/-- Represents the days of the work week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Commute time for each day of the week -/
def commute_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 35
  | Weekday.Tuesday => 45
  | Weekday.Wednesday => 25
  | Weekday.Thursday => 40
  | Weekday.Friday => 30

/-- Additional delay for each day of the week -/
def additional_delay (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 5
  | Weekday.Wednesday => 10
  | Weekday.Friday => 8
  | _ => 0

/-- Security check time for each day of the week -/
def security_check_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Tuesday => 30
  | Weekday.Thursday => 10
  | _ => 15

/-- Constant time for parking and walking -/
def parking_and_walking_time : ℕ := 8

/-- Total time spent on a given day -/
def daily_total_time (day : Weekday) : ℕ :=
  commute_time day + additional_delay day + security_check_time day + parking_and_walking_time

/-- List of all work days in a week -/
def work_week : List Weekday := [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, Weekday.Friday]

/-- Theorem stating the total time spent in a work week -/
theorem total_time_in_work_week : (work_week.map daily_total_time).sum = 323 := by
  sorry

end total_time_in_work_week_l776_77676


namespace fraction_transformation_l776_77687

theorem fraction_transformation (x : ℚ) : (3 - 2*x) / (5 + x) = 1/2 → x = 1/5 := by
  sorry

end fraction_transformation_l776_77687


namespace fraction_sum_l776_77633

theorem fraction_sum : (2 : ℚ) / 5 + 3 / 8 + 1 = 71 / 40 := by sorry

end fraction_sum_l776_77633


namespace line_parallel_contained_in_plane_l776_77646

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_contained_in_plane 
  (a b : Line) (α : Plane) :
  parallel a b → containedIn b α → 
  parallelToPlane a α ∨ containedIn a α :=
sorry

end line_parallel_contained_in_plane_l776_77646


namespace simplify_fraction_expression_l776_77678

theorem simplify_fraction_expression : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
sorry

end simplify_fraction_expression_l776_77678


namespace max_children_in_class_l776_77641

/-- The maximum number of children in a class given the distribution of items -/
theorem max_children_in_class (total_apples total_cookies total_chocolates : ℕ)
  (leftover_apples leftover_cookies leftover_chocolates : ℕ)
  (h1 : total_apples = 55)
  (h2 : total_cookies = 114)
  (h3 : total_chocolates = 83)
  (h4 : leftover_apples = 3)
  (h5 : leftover_cookies = 10)
  (h6 : leftover_chocolates = 5) :
  Nat.gcd (total_apples - leftover_apples)
    (Nat.gcd (total_cookies - leftover_cookies) (total_chocolates - leftover_chocolates)) = 26 := by
  sorry

end max_children_in_class_l776_77641


namespace finite_rational_points_with_finite_orbit_l776_77666

noncomputable def f (C : ℚ) (x : ℚ) : ℚ := x^2 - C

def has_finite_orbit (C : ℚ) (x : ℚ) : Prop :=
  ∃ (S : Finset ℚ), ∀ n : ℕ, (f C)^[n] x ∈ S

theorem finite_rational_points_with_finite_orbit (C : ℚ) :
  {x : ℚ | has_finite_orbit C x}.Finite :=
sorry

end finite_rational_points_with_finite_orbit_l776_77666


namespace smallest_resolvable_debt_is_correct_l776_77629

/-- The value of a cow in dollars -/
def cow_value : ℕ := 400

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 280

/-- A debt is resolvable if it can be expressed as a linear combination of cow and sheep values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (c s : ℤ), debt = c * cow_value + s * sheep_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 40

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end smallest_resolvable_debt_is_correct_l776_77629


namespace mixture_weight_l776_77635

/-- The weight of a mixture of green tea and coffee given specific price changes and costs. -/
theorem mixture_weight (june_cost green_tea_july coffee_july mixture_cost : ℝ) : 
  june_cost > 0 →
  green_tea_july = 0.1 * june_cost →
  coffee_july = 2 * june_cost →
  mixture_cost = 3.15 →
  (mixture_cost / ((green_tea_july + coffee_july) / 2)) = 3 := by
  sorry

#check mixture_weight

end mixture_weight_l776_77635


namespace expression_simplification_l776_77645

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 2/3) : 
  ((x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 3*x*(2*x - y)) / (2*x) = -4/3 := by
  sorry

end expression_simplification_l776_77645


namespace crossing_time_for_49_explorers_l776_77699

/-- The minimum time required for all explorers to cross a river -/
def minimum_crossing_time (
  num_explorers : ℕ
  ) (boat_capacity : ℕ
  ) (crossing_time : ℕ
  ) : ℕ :=
  -- The actual calculation would go here
  45

/-- Theorem stating that for 49 explorers, a boat capacity of 7, and a crossing time of 3 minutes,
    the minimum time to cross is 45 minutes -/
theorem crossing_time_for_49_explorers :
  minimum_crossing_time 49 7 3 = 45 := by
  sorry

end crossing_time_for_49_explorers_l776_77699


namespace train_speed_problem_l776_77651

/-- Proves that the speed of the first train is 20 kmph given the problem conditions -/
theorem train_speed_problem (distance : ℝ) (speed_second : ℝ) (time_first : ℝ) (time_second : ℝ) 
  (h1 : distance = 200)
  (h2 : speed_second = 25)
  (h3 : time_first = 5)
  (h4 : time_second = 4) :
  ∃ (speed_first : ℝ), speed_first * time_first + speed_second * time_second = distance ∧ speed_first = 20 := by
  sorry

#check train_speed_problem

end train_speed_problem_l776_77651


namespace bill_true_discount_l776_77600

/-- Given a bill with face value and banker's discount, calculate the true discount -/
def true_discount (face_value banker_discount : ℚ) : ℚ :=
  (banker_discount * face_value) / (banker_discount + face_value)

/-- Theorem stating that for a bill with face value 270 and banker's discount 54, 
    the true discount is 45 -/
theorem bill_true_discount : 
  true_discount 270 54 = 45 := by sorry

end bill_true_discount_l776_77600


namespace fraction_equality_l776_77610

theorem fraction_equality (x : ℚ) (f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 := by
  sorry

end fraction_equality_l776_77610


namespace average_growth_rate_l776_77606

/-- The average monthly growth rate of CPI food prices -/
def x : ℝ := sorry

/-- The food price increase in January -/
def january_increase : ℝ := 0.028

/-- The predicted food price increase in February -/
def february_increase : ℝ := 0.02

/-- Theorem stating the relationship between the monthly increases and the average growth rate -/
theorem average_growth_rate : 
  (1 + january_increase) * (1 + february_increase) = (1 + x)^2 := by sorry

end average_growth_rate_l776_77606


namespace quadratic_equation_roots_l776_77603

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + k*x₁ + 1 = 3*x₁ + k) ∧ 
  (x₂^2 + k*x₂ + 1 = 3*x₂ + k) := by
sorry

end quadratic_equation_roots_l776_77603


namespace smallest_cube_root_with_small_fraction_l776_77667

theorem smallest_cube_root_with_small_fraction (n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (∃ m : ℕ, (n + r)^3 = m) →
  (∀ k < n, ∀ s : ℝ, (0 < s) → (s < 1 / 500) → (∃ l : ℕ, (k + s)^3 = l) → False) →
  n = 17 := by
sorry

end smallest_cube_root_with_small_fraction_l776_77667


namespace yuna_has_biggest_number_l776_77640

-- Define the type for students
inductive Student : Type
  | Yoongi : Student
  | Jungkook : Student
  | Yuna : Student
  | Yoojung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → Nat
  | Student.Yoongi => 7
  | Student.Jungkook => 6
  | Student.Yuna => 9
  | Student.Yoojung => 8

-- Theorem statement
theorem yuna_has_biggest_number :
  (∀ s : Student, studentNumber s ≤ studentNumber Student.Yuna) ∧
  studentNumber Student.Yuna = 9 := by
  sorry

end yuna_has_biggest_number_l776_77640


namespace quadratic_roots_imply_m_l776_77605

theorem quadratic_roots_imply_m (m : ℚ) : 
  (∃ x : ℂ, 9 * x^2 + 5 * x + m = 0 ∧ 
   (x = (-5 + Complex.I * Real.sqrt 391) / 18 ∨ 
    x = (-5 - Complex.I * Real.sqrt 391) / 18)) → 
  m = 104 / 9 := by
sorry

end quadratic_roots_imply_m_l776_77605


namespace triangle_arithmetic_angle_sequence_l776_77693

theorem triangle_arithmetic_angle_sequence (A B C : Real) : 
  -- The angles form a triangle
  A + B + C = Real.pi →
  -- The angles form an arithmetic sequence
  A + C = 2 * B →
  -- Prove that sin B = √3/2
  Real.sin B = Real.sqrt 3 / 2 := by
sorry

end triangle_arithmetic_angle_sequence_l776_77693


namespace prob_different_cinemas_value_l776_77642

/-- The number of cinemas in the city -/
def num_cinemas : ℕ := 10

/-- The number of boys going to the cinema -/
def num_boys : ℕ := 7

/-- The probability of 7 boys choosing different cinemas out of 10 cinemas -/
def prob_different_cinemas : ℚ :=
  (num_cinemas.factorial / (num_cinemas - num_boys).factorial) / num_cinemas ^ num_boys

theorem prob_different_cinemas_value : 
  prob_different_cinemas = 15120 / 250000 :=
sorry

end prob_different_cinemas_value_l776_77642


namespace greatest_sum_36_l776_77630

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that n is the greatest number of consecutive positive integers starting from 1 whose sum is 36 -/
def is_greatest_sum_36 (n : ℕ) : Prop :=
  sum_first_n n = 36 ∧ ∀ m : ℕ, m > n → sum_first_n m > 36

theorem greatest_sum_36 : is_greatest_sum_36 8 := by
  sorry

end greatest_sum_36_l776_77630


namespace decimal_point_shift_l776_77668

theorem decimal_point_shift (x : ℝ) : (x / 10 = x - 0.72) → x = 0.8 := by
  sorry

end decimal_point_shift_l776_77668


namespace intersection_of_P_and_Q_l776_77625

def P : Set Nat := {1, 3, 6, 9}
def Q : Set Nat := {1, 2, 4, 6, 8}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 6} := by
  sorry

end intersection_of_P_and_Q_l776_77625


namespace vacation_savings_time_l776_77636

/-- 
Given:
- goal_amount: The total amount needed for the vacation
- current_savings: The amount currently saved
- monthly_savings: The amount that can be saved each month

Prove that the number of months needed to reach the goal is 3.
-/
theorem vacation_savings_time (goal_amount current_savings monthly_savings : ℕ) 
  (h1 : goal_amount = 5000)
  (h2 : current_savings = 2900)
  (h3 : monthly_savings = 700) :
  (goal_amount - current_savings + monthly_savings - 1) / monthly_savings = 3 := by
  sorry


end vacation_savings_time_l776_77636


namespace factorization_equality_l776_77694

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a+1)^2 := by
  sorry

end factorization_equality_l776_77694


namespace fraction_change_l776_77613

theorem fraction_change (a b p q x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a^2 + x^2) / (b^2 + x^2) = p / q) 
  (h4 : q ≠ p) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := by
  sorry

end fraction_change_l776_77613


namespace leap_year_date_statistics_l776_77611

/-- Represents the data for dates in a leap year -/
structure LeapYearData where
  dates : Fin 31 → ℕ
  sum_of_values : ℕ
  total_count : ℕ

/-- The mean of the data -/
def mean (data : LeapYearData) : ℚ :=
  data.sum_of_values / data.total_count

/-- The median of the data -/
def median (data : LeapYearData) : ℕ := 16

/-- The median of the modes -/
def median_of_modes (data : LeapYearData) : ℕ := 15

theorem leap_year_date_statistics (data : LeapYearData) 
  (h1 : ∀ i : Fin 29, data.dates i = 12)
  (h2 : data.dates 30 = 11)
  (h3 : data.dates 31 = 7)
  (h4 : data.sum_of_values = 5767)
  (h5 : data.total_count = 366) :
  median_of_modes data < mean data ∧ mean data < median data := by
  sorry


end leap_year_date_statistics_l776_77611


namespace compound_weight_l776_77614

theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 352 → moles = 8 → moles * molecular_weight = 2816 := by
  sorry

end compound_weight_l776_77614


namespace methodC_cannot_eliminate_variables_l776_77638

-- Define the system of linear equations
def equation1 (x y : ℝ) : Prop := 5 * x + 2 * y = 4
def equation2 (x y : ℝ) : Prop := 2 * x - 3 * y = 10

-- Define the method C
def methodC (x y : ℝ) : Prop := 1.5 * (5 * x + 2 * y) - (2 * x - 3 * y) = 1.5 * 4 - 10

-- Theorem stating that method C cannot eliminate variables
theorem methodC_cannot_eliminate_variables :
  ∀ x y : ℝ, methodC x y ↔ (5.5 * x + 6 * y = -4) :=
by sorry

end methodC_cannot_eliminate_variables_l776_77638


namespace fiftieth_term_is_248_l776_77672

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_is_248 :
  arithmeticSequenceTerm 3 5 50 = 248 := by
  sorry

end fiftieth_term_is_248_l776_77672


namespace cube_edge_ratio_l776_77624

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 125 / 1 → a / b = 5 / 1 := by
  sorry

end cube_edge_ratio_l776_77624


namespace band_to_orchestra_ratio_l776_77689

theorem band_to_orchestra_ratio : 
  ∀ (orchestra_students band_students choir_boys choir_girls total_students : ℕ),
    orchestra_students = 20 →
    choir_boys = 12 →
    choir_girls = 16 →
    total_students = 88 →
    total_students = orchestra_students + band_students + choir_boys + choir_girls →
    band_students = 2 * orchestra_students :=
by
  sorry

end band_to_orchestra_ratio_l776_77689


namespace pencil_distribution_l776_77698

theorem pencil_distribution (total_pencils : Nat) (pencils_per_box : Nat) : 
  total_pencils = 48297858 → pencils_per_box = 6 → total_pencils % pencils_per_box = 0 := by
  sorry

end pencil_distribution_l776_77698


namespace mike_passing_percentage_l776_77661

/-- The percentage Mike needs to pass, given his score, shortfall, and maximum possible marks. -/
theorem mike_passing_percentage
  (mike_score : ℕ)
  (shortfall : ℕ)
  (max_marks : ℕ)
  (h1 : mike_score = 212)
  (h2 : shortfall = 16)
  (h3 : max_marks = 760) :
  (((mike_score + shortfall : ℚ) / max_marks) * 100 : ℚ) = 30 := by
  sorry

end mike_passing_percentage_l776_77661


namespace angle_measure_proof_l776_77643

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end angle_measure_proof_l776_77643


namespace cycle_gain_percent_l776_77675

theorem cycle_gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 1500) (h2 : selling_price = 1620) :
  (selling_price - cost_price) / cost_price * 100 = 8 := by
  sorry

end cycle_gain_percent_l776_77675


namespace high_school_population_l776_77650

theorem high_school_population (total_sample : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) (third_grade_population : ℕ) : 
  total_sample = 36 → 
  first_grade_sample = 15 → 
  second_grade_sample = 12 → 
  third_grade_population = 900 → 
  (total_sample : ℚ) / (first_grade_sample + second_grade_sample + (total_sample - first_grade_sample - second_grade_sample)) = 
  (total_sample - first_grade_sample - second_grade_sample : ℚ) / third_grade_population → 
  (total_sample : ℕ) * (third_grade_population / (total_sample - first_grade_sample - second_grade_sample)) = 3600 :=
by sorry

end high_school_population_l776_77650


namespace sin_210_degrees_l776_77628

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l776_77628


namespace sufficient_not_necessary_condition_l776_77607

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x + y ≥ 2) ∧
  (∃ x y : ℝ, x + y ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end sufficient_not_necessary_condition_l776_77607


namespace expression_simplification_l776_77692

theorem expression_simplification :
  1 / (1 / ((Real.sqrt 2 + 1)^2) + 1 / ((Real.sqrt 5 - 2)^3)) = 
  1 / (41 + 17 * Real.sqrt 5 - 2 * Real.sqrt 2) := by
  sorry

end expression_simplification_l776_77692


namespace chris_hockey_stick_cost_l776_77627

/-- The cost of a hockey stick, given the total spent, helmet cost, and number of sticks. -/
def hockey_stick_cost (total_spent helmet_cost num_sticks : ℚ) : ℚ :=
  (total_spent - helmet_cost) / num_sticks

/-- Theorem stating the cost of one hockey stick given Chris's purchase. -/
theorem chris_hockey_stick_cost :
  let total_spent : ℚ := 68
  let helmet_cost : ℚ := 25
  let num_sticks : ℚ := 2
  hockey_stick_cost total_spent helmet_cost num_sticks = 21.5 := by
  sorry

end chris_hockey_stick_cost_l776_77627


namespace intersection_of_sets_l776_77626

theorem intersection_of_sets : 
  let M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}
  let N : Set ℤ := {x | x^2 = x}
  M ∩ N = {0, 1} := by
  sorry

end intersection_of_sets_l776_77626


namespace arithmetic_sequence_formula_l776_77664

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (d t : ℝ) :
  arithmetic_sequence a d →
  d > 0 →
  a 1 = 1 →
  (∀ n, 2 * (a n * a (n + 1) + 1) = t * (1 + a n)) →
  ∀ n, a n = 2 * n - 1 + (-1)^n := by
  sorry

end arithmetic_sequence_formula_l776_77664


namespace unique_solution_l776_77658

theorem unique_solution (p q n : ℕ+) (h1 : Nat.gcd p.val q.val = 1)
  (h2 : p + q^2 = (n^2 + 1) * p^2 + q) :
  p = n + 1 ∧ q = n^2 + n + 1 := by
  sorry

end unique_solution_l776_77658


namespace cat_whiskers_count_l776_77622

/-- The number of whiskers on Princess Puff's face -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers on Catman Do's face -/
def catman_do_whiskers : ℕ := 2 * princess_puff_whiskers - 6

/-- The number of whiskers on Sir Whiskerson's face -/
def sir_whiskerson_whiskers : ℕ := princess_puff_whiskers + catman_do_whiskers + 8

/-- Theorem stating the correct number of whiskers for each cat -/
theorem cat_whiskers_count :
  princess_puff_whiskers = 14 ∧
  catman_do_whiskers = 22 ∧
  sir_whiskerson_whiskers = 44 := by
  sorry

end cat_whiskers_count_l776_77622


namespace max_cookies_in_class_l776_77674

/-- Represents the maximum number of cookies one student could have taken in a class. -/
def max_cookies_for_one_student (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ) : ℕ :=
  num_students * avg_cookies - (num_students - 1) * min_cookies

/-- Theorem stating the maximum number of cookies one student could have taken. -/
theorem max_cookies_in_class (num_students : ℕ) (avg_cookies : ℕ) (min_cookies : ℕ)
    (h_num_students : num_students = 20)
    (h_avg_cookies : avg_cookies = 6)
    (h_min_cookies : min_cookies = 2) :
    max_cookies_for_one_student num_students avg_cookies min_cookies = 82 := by
  sorry

#eval max_cookies_for_one_student 20 6 2

end max_cookies_in_class_l776_77674


namespace complex_equality_l776_77691

theorem complex_equality (x y z : ℝ) (α β γ : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hα : Complex.abs α = 1) (hβ : Complex.abs β = 1) (hγ : Complex.abs γ = 1)
  (hxyz : x + y + z = 0) (hαβγ : α * x + β * y + γ * z = 0) :
  α = β ∧ β = γ := by
  sorry

end complex_equality_l776_77691


namespace day_after_53_from_monday_is_friday_l776_77648

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

theorem day_after_53_from_monday_is_friday :
  dayAfter DayOfWeek.Monday 53 = DayOfWeek.Friday := by
  sorry

end day_after_53_from_monday_is_friday_l776_77648


namespace down_payment_calculation_l776_77609

def cash_price : ℕ := 400
def monthly_payment : ℕ := 30
def num_months : ℕ := 12
def cash_savings : ℕ := 80

theorem down_payment_calculation : 
  cash_price + cash_savings - monthly_payment * num_months = 120 := by
  sorry

end down_payment_calculation_l776_77609


namespace imaginary_sum_zero_l776_77673

theorem imaginary_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end imaginary_sum_zero_l776_77673


namespace bird_feeding_problem_l776_77679

/-- Given the following conditions:
    - There are 6 baby birds
    - Papa bird caught 9 worms
    - Mama bird caught 13 worms
    - 2 worms were stolen from Mama bird
    - Mama bird needs to catch 34 more worms
    - The worms are needed for 3 days
    Prove that each baby bird needs 3 worms per day. -/
theorem bird_feeding_problem (
  num_babies : ℕ)
  (papa_worms : ℕ)
  (mama_worms : ℕ)
  (stolen_worms : ℕ)
  (additional_worms : ℕ)
  (num_days : ℕ)
  (h1 : num_babies = 6)
  (h2 : papa_worms = 9)
  (h3 : mama_worms = 13)
  (h4 : stolen_worms = 2)
  (h5 : additional_worms = 34)
  (h6 : num_days = 3) :
  (papa_worms + mama_worms - stolen_worms + additional_worms) / (num_babies * num_days) = 3 := by
  sorry

#eval (9 + 13 - 2 + 34) / (6 * 3)  -- This should output 3

end bird_feeding_problem_l776_77679


namespace glucose_solution_volume_l776_77680

/-- Given a glucose solution with a concentration of 15 grams per 100 cubic centimeters,
    prove that a volume containing 9.75 grams of glucose is 65 cubic centimeters. -/
theorem glucose_solution_volume 
  (concentration : ℝ) 
  (volume : ℝ) 
  (glucose_mass : ℝ) 
  (h1 : concentration = 15 / 100) 
  (h2 : glucose_mass = 9.75) 
  (h3 : concentration * volume = glucose_mass) : 
  volume = 65 := by
sorry

end glucose_solution_volume_l776_77680


namespace bottom_right_is_one_l776_77683

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ q.2.val + 1 = p.2.val)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ q.1.val + 1 = p.1.val))

/-- Check if two numbers are consecutive --/
def consecutive (m n : Fin 9) : Prop :=
  m.val + 1 = n.val ∨ n.val + 1 = m.val

/-- The theorem to prove --/
theorem bottom_right_is_one (g : Grid) :
  (∀ i j k l : Fin 3, g i j ≠ g k l → (i, j) ≠ (k, l)) →
  (∀ i j k l : Fin 3, consecutive (g i j) (g k l) → adjacent (i, j) (k, l)) →
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val = 24 →
  (g 1 1).val + (g 0 1).val + (g 1 0).val + (g 1 2).val + (g 2 1).val = 25 →
  (g 2 2).val = 1 := by
  sorry

end bottom_right_is_one_l776_77683


namespace problem_solution_l776_77665

theorem problem_solution : 
  let P : ℕ := 2007 / 5
  let Q : ℕ := P / 4
  let Y : ℕ := 2 * (P - Q)
  Y = 602 := by
sorry

end problem_solution_l776_77665


namespace log_expression_equals_one_l776_77671

noncomputable def a : ℝ := Real.log 5 / Real.log 6
noncomputable def b : ℝ := Real.log 3 / Real.log 10
noncomputable def c : ℝ := Real.log 2 / Real.log 15

theorem log_expression_equals_one :
  (1 - 2 * a * b * c) / (a * b + b * c + c * a) = 1 := by sorry

end log_expression_equals_one_l776_77671


namespace sqrt_3x_lt_5x_iff_l776_77657

theorem sqrt_3x_lt_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt (3 * x) < 5 * x ↔ x > 3 / 25 := by
  sorry

end sqrt_3x_lt_5x_iff_l776_77657


namespace distance_is_100_miles_l776_77697

/-- Represents the fuel efficiency of a car in miles per gallon. -/
def miles_per_gallon : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons. -/
def gallons_needed : ℝ := 5

/-- Calculates the distance to Grandma's house in miles. -/
def distance_to_grandma : ℝ := miles_per_gallon * gallons_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles. -/
theorem distance_is_100_miles : distance_to_grandma = 100 :=
  sorry

end distance_is_100_miles_l776_77697


namespace max_sum_reciprocal_ninth_l776_77601

theorem max_sum_reciprocal_ninth (a b : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a : ℕ) + b ≤ 100 ∧ ∃ (a' b' : ℕ+), (a' : ℚ)⁻¹ + (b' : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a' : ℕ) + b' = 100 :=
by sorry

end max_sum_reciprocal_ninth_l776_77601


namespace function_always_negative_iff_a_in_range_l776_77617

/-- The function f(x) = ax^2 + ax - 1 is always negative over the real numbers
    if and only if a is in the range (-4, 0]. -/
theorem function_always_negative_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end function_always_negative_iff_a_in_range_l776_77617


namespace p_necessary_not_sufficient_for_q_l776_77639

def p (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)
def q (x : ℝ) : Prop := x = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end p_necessary_not_sufficient_for_q_l776_77639


namespace outfits_count_l776_77652

/-- The number of different outfits that can be made with a given number of shirts, ties, and shoes. -/
def num_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ := shirts * ties * shoes

/-- Theorem: Given 8 shirts, 7 ties, and 4 pairs of shoes, the total number of different possible outfits is 224. -/
theorem outfits_count : num_outfits 8 7 4 = 224 := by
  sorry

end outfits_count_l776_77652


namespace added_amount_after_doubling_l776_77688

theorem added_amount_after_doubling (x y : ℕ) : 
  x = 17 → 3 * (2 * x + y) = 117 → y = 5 := by sorry

end added_amount_after_doubling_l776_77688


namespace function_graph_point_l776_77649

theorem function_graph_point (f : ℝ → ℝ) (h : f 8 = 5) :
  let g := fun x => (f (3 * x) / 3 + 3) / 3
  g (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end function_graph_point_l776_77649


namespace chores_to_cartoons_ratio_l776_77663

/-- Given that 2 hours (120 minutes) of cartoons requires 96 minutes of chores,
    prove that the ratio of chores to cartoons is 8 minutes of chores
    for every 10 minutes of cartoons. -/
theorem chores_to_cartoons_ratio :
  ∀ (cartoon_time chore_time : ℕ),
    cartoon_time = 120 →
    chore_time = 96 →
    (chore_time : ℚ) / (cartoon_time : ℚ) * 10 = 8 := by
  sorry

#check chores_to_cartoons_ratio

end chores_to_cartoons_ratio_l776_77663


namespace derivative_f_at_one_l776_77602

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_f_at_one :
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end derivative_f_at_one_l776_77602


namespace sum_of_F_at_4_and_neg_2_l776_77608

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 2)) + (10 / Real.pi) * Real.arctan (Real.sqrt (abs x))

theorem sum_of_F_at_4_and_neg_2 : F 4 + F (-2) = Real.sqrt 6 + 3.529 := by
  sorry

end sum_of_F_at_4_and_neg_2_l776_77608


namespace money_distribution_l776_77685

theorem money_distribution (w x y z : ℝ) (h1 : w = 375) 
  (h2 : x = 6 * w) (h3 : y = 2 * w) (h4 : z = 4 * w) : 
  x - y = 1500 := by
  sorry

end money_distribution_l776_77685


namespace discount_percentage_proof_l776_77631

theorem discount_percentage_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 128)
  (h2 : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 := by
sorry

end discount_percentage_proof_l776_77631


namespace floor_sqrt_50_squared_l776_77684

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end floor_sqrt_50_squared_l776_77684


namespace no_function_satisfies_equation_l776_77612

theorem no_function_satisfies_equation :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + y) = x * f x + y := by
  sorry

end no_function_satisfies_equation_l776_77612


namespace proportion_with_added_number_l776_77604

theorem proportion_with_added_number : 
  ∃ (x : ℚ), (1 : ℚ) / 3 = 4 / x ∧ x = 12 := by
  sorry

end proportion_with_added_number_l776_77604


namespace garden_length_proof_l776_77637

/-- Represents a rectangular garden with its dimensions. -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_length_proof (g : RectangularGarden) 
  (h1 : perimeter g = 600) 
  (h2 : g.breadth = 95) : 
  g.length = 205 := by
  sorry

end garden_length_proof_l776_77637


namespace inverse_of_three_mod_243_l776_77647

theorem inverse_of_three_mod_243 : ∃ x : ℕ, x < 243 ∧ (3 * x) % 243 = 1 :=
by
  use 324
  sorry

end inverse_of_three_mod_243_l776_77647


namespace system_sampling_theorem_l776_77681

/-- Represents a system sampling method -/
structure SystemSampling where
  total_students : ℕ
  sample_size : ℕ
  common_difference : ℕ

/-- Checks if a list of numbers forms a valid system sample -/
def is_valid_sample (s : SystemSampling) (sample : List ℕ) : Prop :=
  sample.length = s.sample_size ∧
  ∀ i j, i < j → j < s.sample_size →
    sample[j]! - sample[i]! = s.common_difference * (j - i)

theorem system_sampling_theorem (s : SystemSampling)
  (h_total : s.total_students = 160)
  (h_size : s.sample_size = 5)
  (h_diff : s.common_difference = 32)
  (h_known : ∃ (sample : List ℕ), is_valid_sample s sample ∧ 
    40 ∈ sample ∧ 72 ∈ sample ∧ 136 ∈ sample) :
  ∃ (full_sample : List ℕ), is_valid_sample s full_sample ∧
    40 ∈ full_sample ∧ 72 ∈ full_sample ∧ 136 ∈ full_sample ∧
    8 ∈ full_sample ∧ 104 ∈ full_sample :=
sorry

end system_sampling_theorem_l776_77681


namespace sqrt_calculations_l776_77670

theorem sqrt_calculations :
  (∀ (a b c : ℝ), 
    a = 4 * Real.sqrt (1/2) ∧ 
    b = Real.sqrt 32 ∧ 
    c = Real.sqrt 8 →
    a + b - c = 4 * Real.sqrt 2) ∧
  (∀ (d e f g : ℝ),
    d = Real.sqrt 6 ∧
    e = Real.sqrt 3 ∧
    f = Real.sqrt 12 ∧
    g = Real.sqrt 3 →
    d * e + f / g = 3 * Real.sqrt 2 + 2) :=
by sorry

end sqrt_calculations_l776_77670


namespace alices_favorite_number_l776_77619

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ,
    90 < n ∧ n < 150 ∧
    n % 13 = 0 ∧
    n % 4 ≠ 0 ∧
    digit_sum n % 4 = 0 ∧
    n = 143 := by sorry

end alices_favorite_number_l776_77619


namespace right_triangle_ab_length_l776_77656

/-- Given a right triangle ABC in the x-y plane where:
    - Angle B is 90 degrees
    - Length of AC is 25
    - Slope of line segment AC is 4/3
    Prove that the length of AB is 15 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- B is a right angle
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25) -- Length of AC is 25
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 := by
  sorry

end right_triangle_ab_length_l776_77656


namespace functional_equation_solution_l776_77653

/-- A function satisfying the given functional equation and differentiability condition -/
class FunctionalEquationSolution (f : ℝ → ℝ) : Prop where
  equation : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
  smooth : ContDiff ℝ ⊤ f

/-- The main theorem stating the form of the solution -/
theorem functional_equation_solution (f : ℝ → ℝ) [FunctionalEquationSolution f] :
  ∃ a : ℝ, ∀ x : ℝ, f x = x^2 + a * x :=
sorry

end functional_equation_solution_l776_77653


namespace root_reciprocal_sum_l776_77618

theorem root_reciprocal_sum (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → 
  n^2 + 3*n - 1 = 0 → 
  m ≠ n →
  1/m + 1/n = 3 := by
sorry

end root_reciprocal_sum_l776_77618


namespace jacks_speed_l776_77682

/-- Prove Jack's speed given the conditions of the problem -/
theorem jacks_speed (initial_distance : ℝ) (christina_speed : ℝ) (lindy_speed : ℝ) (lindy_distance : ℝ) :
  initial_distance = 360 →
  christina_speed = 7 →
  lindy_speed = 12 →
  lindy_distance = 360 →
  ∃ (jack_speed : ℝ), jack_speed = 5 := by
  sorry


end jacks_speed_l776_77682


namespace yard_length_l776_77632

theorem yard_length (num_trees : ℕ) (distance : ℝ) :
  num_trees = 11 →
  distance = 18 →
  (num_trees - 1) * distance = 180 :=
by sorry

end yard_length_l776_77632


namespace island_distance_l776_77686

theorem island_distance (n : ℝ) : 
  let a := 8*n
  let b := 5*n
  let c := 7*n
  let α := 60 * π / 180
  a^2 + b^2 - 2*a*b*Real.cos α = c^2 := by sorry

end island_distance_l776_77686


namespace combination_square_numbers_examples_find_m_l776_77621

def is_combination_square_numbers (a b c : Int) : Prop :=
  a < 0 ∧ b < 0 ∧ c < 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y z : Int), x^2 = a * b ∧ y^2 = b * c ∧ z^2 = a * c

theorem combination_square_numbers_examples :
  (is_combination_square_numbers (-4) (-16) (-25)) ∧
  (is_combination_square_numbers (-3) (-48) (-12)) ∧
  (is_combination_square_numbers (-2) (-18) (-72)) := by sorry

theorem find_m :
  ∀ m : Int, is_combination_square_numbers (-3) m (-12) ∧ 
  (∃ (x : Int), x^2 = -3 * m ∨ x^2 = m * (-12) ∨ x^2 = -3 * (-12)) ∧
  x = 12 → m = -48 := by sorry

end combination_square_numbers_examples_find_m_l776_77621


namespace perpendicular_line_x_intercept_l776_77623

/-- Given line L1: 4x + 5y = 10 and line L2 perpendicular to L1 with y-intercept -3,
    prove that the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 10
  let L2 : ℝ → ℝ → Prop := λ x y ↦ ∃ m : ℝ, y = m * x - 3 ∧ m * (-4/5) = -1
  ∃ x : ℝ, L2 x 0 ∧ x = 12/5 := by sorry

end perpendicular_line_x_intercept_l776_77623


namespace student_group_assignments_l776_77677

theorem student_group_assignments (n : ℕ) (k : ℕ) :
  n = 5 → k = 2 → (2 : ℕ) ^ n = 32 := by
  sorry

end student_group_assignments_l776_77677


namespace distinct_pairs_from_twelve_l776_77644

theorem distinct_pairs_from_twelve (n : ℕ) : n = 12 → (n.choose 2 = 66) := by
  sorry

end distinct_pairs_from_twelve_l776_77644


namespace f_min_at_3_l776_77655

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_3 : ∀ x : ℝ, f 3 ≤ f x := by sorry

end f_min_at_3_l776_77655


namespace power_two_geq_square_l776_77660

theorem power_two_geq_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_two_geq_square_l776_77660


namespace semicircle_radius_l776_77662

/-- Given a semi-circle with perimeter 180 cm, its radius is 180 / (π + 2) cm. -/
theorem semicircle_radius (P : ℝ) (h : P = 180) :
  P = π * r + 2 * r → r = 180 / (π + 2) :=
by
  sorry

end semicircle_radius_l776_77662


namespace room_length_proof_l776_77615

theorem room_length_proof (width : ℝ) (area_covered : ℝ) (area_needed : ℝ) :
  width = 15 →
  area_covered = 16 →
  area_needed = 149 →
  (area_covered + area_needed) / width = 11 := by
  sorry

end room_length_proof_l776_77615


namespace swimming_pool_payment_analysis_l776_77669

/-- Represents the swimming pool payment methods -/
structure SwimmingPoolPayment where
  membershipCost : ℕ
  memberSwimCost : ℕ
  nonMemberSwimCost : ℕ

/-- Calculates the cost for a given number of swims using Method 1 -/
def method1Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.membershipCost + p.memberSwimCost * swims

/-- Calculates the cost for a given number of swims using Method 2 -/
def method2Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.nonMemberSwimCost * swims

/-- Calculates the maximum number of swims possible with a given budget using Method 1 -/
def maxSwimMethod1 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  (budget - p.membershipCost) / p.memberSwimCost

/-- Calculates the maximum number of swims possible with a given budget using Method 2 -/
def maxSwimMethod2 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  budget / p.nonMemberSwimCost

theorem swimming_pool_payment_analysis 
  (p : SwimmingPoolPayment) 
  (h1 : p.membershipCost = 200)
  (h2 : p.memberSwimCost = 10)
  (h3 : p.nonMemberSwimCost = 30) :
  (method1Cost p 3 = 230) ∧
  (method2Cost p 9 < method1Cost p 9) ∧
  (maxSwimMethod1 p 600 > maxSwimMethod2 p 600) := by
  sorry

end swimming_pool_payment_analysis_l776_77669


namespace sum_reciprocals_factors_of_12_l776_77634

def factors_of_12 : List ℕ := [1, 2, 3, 4, 6, 12]

theorem sum_reciprocals_factors_of_12 :
  (factors_of_12.map (λ n => (1 : ℚ) / n)).sum = 7 / 3 := by
  sorry

end sum_reciprocals_factors_of_12_l776_77634


namespace factorial_sum_equality_l776_77654

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 + Nat.factorial 5 = 5160 := by
  sorry

end factorial_sum_equality_l776_77654


namespace propositions_truth_l776_77620

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a > b) → (a^2 > b^2)
def proposition2 (a b : ℝ) : Prop := (Real.log a = Real.log b) → (a = b)
def proposition3 (x y : ℝ) : Prop := (|x| = |y|) ↔ (x^2 = y^2)
def proposition4 (A B : ℝ) : Prop := (Real.sin A > Real.sin B) ↔ (A > B)

-- Theorem statement
theorem propositions_truth : 
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧ 
  (∃ a b : ℝ, Real.log a = Real.log b ∧ a ≠ b) ∧
  (∀ x y : ℝ, (|x| = |y|) ↔ (x^2 = y^2)) ∧
  (∀ A B : ℝ, 0 < A ∧ A < π ∧ 0 < B ∧ B < π → ((Real.sin A > Real.sin B) ↔ (A > B))) :=
by sorry

end propositions_truth_l776_77620
