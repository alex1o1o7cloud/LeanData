import Mathlib

namespace inequality_solution_l1170_117098

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 12 * x - 35) / ((x - 2) * (3 * x + 6)) < 2 ↔ 
  (x > -2 ∧ x < 11/18) ∨ x > 2 :=
sorry

end inequality_solution_l1170_117098


namespace messages_per_member_per_day_l1170_117007

theorem messages_per_member_per_day :
  let initial_members : ℕ := 150
  let removed_members : ℕ := 20
  let remaining_members : ℕ := initial_members - removed_members
  let total_weekly_messages : ℕ := 45500
  let messages_per_day : ℕ := total_weekly_messages / 7
  let messages_per_member_per_day : ℕ := messages_per_day / remaining_members
  messages_per_member_per_day = 50 :=
by sorry

end messages_per_member_per_day_l1170_117007


namespace scientific_notation_of_wetland_area_l1170_117000

/-- Proves that 29.47 thousand is equal to 2.947 × 10^4 in scientific notation -/
theorem scientific_notation_of_wetland_area :
  (29.47 * 1000 : ℝ) = 2.947 * (10 ^ 4) :=
by sorry

end scientific_notation_of_wetland_area_l1170_117000


namespace strawberries_left_l1170_117081

/-- Given 3.5 baskets of strawberries, with 50 strawberries per basket,
    distributed equally among 24 girls, prove that 7 strawberries are left. -/
theorem strawberries_left (baskets : ℚ) (strawberries_per_basket : ℕ) (girls : ℕ) :
  baskets = 3.5 ∧ strawberries_per_basket = 50 ∧ girls = 24 →
  (baskets * strawberries_per_basket : ℚ) - (↑girls * ↑⌊(baskets * strawberries_per_basket) / girls⌋) = 7 := by
  sorry

end strawberries_left_l1170_117081


namespace regular_polygon_properties_l1170_117004

/-- Properties of a regular polygon with 24-degree exterior angles -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  (180 * (n - 2) = 2340 ∧ (n * (n - 3)) / 2 = 90) :=
by sorry

end regular_polygon_properties_l1170_117004


namespace marble_distribution_l1170_117018

theorem marble_distribution (total_marbles : ℕ) (num_children : ℕ) 
  (h1 : total_marbles = 60) 
  (h2 : num_children = 7) : 
  (num_children - (total_marbles % num_children)) = 3 := by
  sorry

end marble_distribution_l1170_117018


namespace multiplication_table_even_fraction_l1170_117053

/-- The size of the multiplication table (16 in this case) -/
def table_size : ℕ := 16

/-- A number is even if it's divisible by 2 -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- The count of even numbers in the range [0, table_size - 1] -/
def even_count : ℕ := (table_size + 1) / 2

/-- The count of odd numbers in the range [0, table_size - 1] -/
def odd_count : ℕ := table_size - even_count

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of entries where both factors are odd -/
def odd_entries : ℕ := odd_count * odd_count

/-- The number of entries where at least one factor is even -/
def even_entries : ℕ := total_entries - odd_entries

/-- The fraction of even entries in the multiplication table -/
def even_fraction : ℚ := even_entries / total_entries

theorem multiplication_table_even_fraction :
  even_fraction = 3/4 := by sorry

end multiplication_table_even_fraction_l1170_117053


namespace range_of_x_minus_cosy_l1170_117075

theorem range_of_x_minus_cosy (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) :
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 := by
  sorry

end range_of_x_minus_cosy_l1170_117075


namespace least_positive_integer_congruence_l1170_117086

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ 
  (x : ℤ) ≡ 4 [ZMOD 7] ∧
  ∀ y : ℕ+, ((y : ℤ) + 7219 ≡ 5305 [ZMOD 17] ∧ (y : ℤ) ≡ 4 [ZMOD 7]) → x ≤ y :=
by sorry

end least_positive_integer_congruence_l1170_117086


namespace celyna_candy_purchase_l1170_117035

/-- Prove that given the conditions of Celyna's candy purchase, the amount of candy B is 500 grams -/
theorem celyna_candy_purchase (candy_a_weight : ℝ) (candy_a_cost : ℝ) (candy_b_cost : ℝ) (average_price : ℝ) :
  candy_a_weight = 300 →
  candy_a_cost = 5 →
  candy_b_cost = 7 →
  average_price = 1.5 →
  ∃ x : ℝ, x = 500 ∧ 
    (candy_a_cost + candy_b_cost) / ((candy_a_weight + x) / 100) = average_price :=
by sorry

end celyna_candy_purchase_l1170_117035


namespace negation_of_universal_proposition_l1170_117077

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) := by sorry

end negation_of_universal_proposition_l1170_117077


namespace staff_meeting_attendance_l1170_117021

theorem staff_meeting_attendance (total_doughnuts served_doughnuts left_doughnuts doughnuts_per_staff : ℕ) :
  served_doughnuts = 50 →
  doughnuts_per_staff = 2 →
  left_doughnuts = 12 →
  total_doughnuts = served_doughnuts - left_doughnuts →
  (total_doughnuts / doughnuts_per_staff : ℕ) = 19 :=
by sorry

end staff_meeting_attendance_l1170_117021


namespace M_intersect_N_l1170_117019

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem M_intersect_N : M ∩ N = {5, 7, 9} := by sorry

end M_intersect_N_l1170_117019


namespace ab_plus_cd_equals_twelve_l1170_117044

theorem ab_plus_cd_equals_twelve 
  (a b c d : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 5) :
  a * b + c * d = 12 := by
  sorry

end ab_plus_cd_equals_twelve_l1170_117044


namespace pharmacy_service_l1170_117068

/-- The number of customers served by three workers in a day -/
def customers_served (regular_hours work_rate reduced_hours : ℕ) : ℕ :=
  work_rate * (2 * regular_hours + reduced_hours)

/-- Theorem: Given the specific conditions, the total number of customers served is 154 -/
theorem pharmacy_service : customers_served 8 7 6 = 154 := by
  sorry

end pharmacy_service_l1170_117068


namespace contest_questions_l1170_117017

theorem contest_questions (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (a b c : ℕ), 10 < a ∧ a ≤ b ∧ b ≤ c ∧ c < 13) 
  (h3 : 4 * n = 10 + 13 + a + b + c) : n = 14 := by
  sorry

end contest_questions_l1170_117017


namespace area_swept_by_small_square_l1170_117090

/-- The area swept by a small square sliding along three sides of a larger square -/
theorem area_swept_by_small_square (large_side small_side : ℝ) :
  large_side > 0 ∧ small_side > 0 ∧ large_side > small_side →
  let swept_area := large_side^2 - (large_side - 2*small_side)^2
  swept_area = 36 ∧ large_side = 10 ∧ small_side = 1 := by
  sorry

#check area_swept_by_small_square

end area_swept_by_small_square_l1170_117090


namespace no_valid_triples_l1170_117069

theorem no_valid_triples : 
  ¬∃ (a b c : ℤ) (x : ℚ), 
    a < 0 ∧ 
    b^2 - 4*a*c = 5 ∧ 
    a * x^2 + b * x + c > 0 := by
  sorry

end no_valid_triples_l1170_117069


namespace johns_local_taxes_l1170_117006

/-- Proves that given John's hourly wage and local tax rate, the amount of local taxes paid in cents per hour is 60 cents. -/
theorem johns_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end johns_local_taxes_l1170_117006


namespace power_product_exponent_l1170_117023

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end power_product_exponent_l1170_117023


namespace range_of_f_l1170_117091

def f (x : ℝ) : ℝ := |x| + 1

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 :=
by sorry

end range_of_f_l1170_117091


namespace min_floor_sum_l1170_117055

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 :=
by sorry

end min_floor_sum_l1170_117055


namespace rectangle_with_border_area_l1170_117095

/-- Calculates the combined area of a rectangle and its border -/
def combinedArea (length width borderWidth : Real) : Real :=
  (length + 2 * borderWidth) * (width + 2 * borderWidth)

theorem rectangle_with_border_area :
  let length : Real := 0.6
  let width : Real := 0.35
  let borderWidth : Real := 0.05
  combinedArea length width borderWidth = 0.315 := by
  sorry

end rectangle_with_border_area_l1170_117095


namespace goldfish_remaining_l1170_117092

def initial_goldfish : ℕ := 15
def fewer_goldfish : ℕ := 11

theorem goldfish_remaining : initial_goldfish - fewer_goldfish = 4 := by
  sorry

end goldfish_remaining_l1170_117092


namespace sqrt_inequality_l1170_117073

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end sqrt_inequality_l1170_117073


namespace extreme_values_and_inequality_l1170_117036

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m*x + 1) / Real.exp x

theorem extreme_values_and_inequality (m : ℝ) (h₁ : m ≥ 0) :
  (m > 0 → (∃ (min_x max_x : ℝ), min_x = 1 - m ∧ max_x = 1 ∧
    ∀ x, f m x ≥ f m min_x ∧ f m x ≤ f m max_x)) ∧
  (m ∈ Set.Ioo 1 2 → ∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 1 m → x₂ ∈ Set.Icc 1 m →
    f m x₁ > -x₂ + 1 + 1 / Real.exp 1) :=
sorry

end extreme_values_and_inequality_l1170_117036


namespace new_supervisor_salary_l1170_117076

/-- Proves that the new supervisor's salary is $960 given the conditions of the problem -/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_avg_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_avg_salary : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_avg : initial_avg_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_avg : new_avg_salary = 440)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 960 ∧
    (num_workers : ℚ) * initial_avg_salary + old_supervisor_salary = (num_total : ℚ) * initial_avg_salary ∧
    (num_workers : ℚ) * initial_avg_salary + new_supervisor_salary = (num_total : ℚ) * new_avg_salary :=
by sorry

end new_supervisor_salary_l1170_117076


namespace arithmetic_sequence_solution_l1170_117061

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_solution :
  ∀ (a : ℕ → ℝ),
    is_arithmetic_sequence a →
    a 0 = 2^2 →
    a 2 = 5^2 →
    ∃ x : ℝ, x > 0 ∧ a 1 = x^2 ∧ x = Real.sqrt 14.5 := by
  sorry

end arithmetic_sequence_solution_l1170_117061


namespace fruit_seller_apples_l1170_117097

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * (1 - 0.4) = 420 → original = 700 := by
  sorry

end fruit_seller_apples_l1170_117097


namespace high_school_students_l1170_117083

theorem high_school_students (total_students : ℕ) 
  (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 40)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 450)
  (h5 : total_students = (music_students - both_students) + (art_students - both_students) + both_students + neither_students) :
  total_students = 500 := by
sorry

end high_school_students_l1170_117083


namespace grading_multiple_proof_l1170_117042

/-- Given a grading method that subtracts a multiple of incorrect responses
    from correct responses, prove that the multiple is 2 for a specific case. -/
theorem grading_multiple_proof (total_questions : ℕ) (correct_responses : ℕ) (score : ℕ) :
  total_questions = 100 →
  correct_responses = 87 →
  score = 61 →
  ∃ (m : ℚ), score = correct_responses - m * (total_questions - correct_responses) →
  m = 2 := by
  sorry

end grading_multiple_proof_l1170_117042


namespace total_cans_l1170_117015

def bag1 : ℕ := 5
def bag2 : ℕ := 7
def bag3 : ℕ := 12
def bag4 : ℕ := 4
def bag5 : ℕ := 8
def bag6 : ℕ := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end total_cans_l1170_117015


namespace line_l_properties_l1170_117059

/-- A line passing through (-2, 1) with y-intercept twice the x-intercept -/
def line_l (x y : ℝ) : Prop := 2*x + y + 3 = 0

theorem line_l_properties :
  (∃ x y : ℝ, line_l x y ∧ x = -2 ∧ y = 1) ∧
  (∃ a : ℝ, a ≠ 0 → line_l a 0 ∧ line_l 0 (2*a)) :=
sorry

end line_l_properties_l1170_117059


namespace hour_hand_angle_after_one_hour_l1170_117066

/-- Represents the angle turned by the hour hand of a watch. -/
def angle_turned (hours : ℝ) : ℝ :=
  -30 * hours

/-- The theorem states that the angle turned by the hour hand after 1 hour is -30°. -/
theorem hour_hand_angle_after_one_hour :
  angle_turned 1 = -30 := by sorry

end hour_hand_angle_after_one_hour_l1170_117066


namespace rectangle_thirteen_squares_l1170_117082

/-- A rectangle can be divided into 13 equal squares if and only if its side ratio is 13:1 or 1:13 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ (a = s ∧ b = 13 * s ∨ a = 13 * s ∧ b = s)) ↔
  (a * b = 13 * (a.min b) * (a.min b)) :=
sorry

end rectangle_thirteen_squares_l1170_117082


namespace expression_values_l1170_117039

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 := by
  sorry

end expression_values_l1170_117039


namespace sunday_max_available_l1170_117078

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the team members
inductive Member
  | Alice
  | Bob
  | Cara
  | Dave
  | Ella

-- Define a function to represent the availability of each member on each day
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cara, Day.Monday => false
  | Member.Cara, Day.Tuesday => false
  | Member.Cara, Day.Thursday => false
  | Member.Cara, Day.Saturday => false
  | Member.Cara, Day.Sunday => false
  | Member.Dave, Day.Wednesday => false
  | Member.Dave, Day.Saturday => false
  | Member.Ella, Day.Monday => false
  | Member.Ella, Day.Friday => false
  | Member.Ella, Day.Saturday => false
  | _, _ => true

-- Define a function to count the number of available members on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cara, Member.Dave, Member.Ella]).length

-- Theorem: Sunday has the maximum number of available team members
theorem sunday_max_available :
  ∀ d : Day, countAvailable Day.Sunday ≥ countAvailable d := by
  sorry


end sunday_max_available_l1170_117078


namespace tan_10pi_minus_theta_l1170_117071

theorem tan_10pi_minus_theta (θ : Real) (h1 : π < θ) (h2 : θ < 2*π) 
  (h3 : Real.cos (θ - 9*π) = -3/5) : Real.tan (10*π - θ) = 4/3 := by
  sorry

end tan_10pi_minus_theta_l1170_117071


namespace sum_of_even_integers_l1170_117024

theorem sum_of_even_integers (a b c d : ℤ) 
  (h1 : Even a) (h2 : Even b) (h3 : Even c) (h4 : Even d)
  (eq1 : a - b + c = 8)
  (eq2 : b - c + d = 10)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 6) :
  a + b + c + d = 28 := by
sorry

end sum_of_even_integers_l1170_117024


namespace larger_divided_by_smaller_l1170_117046

theorem larger_divided_by_smaller (L S Q : ℕ) : 
  L - S = 1365 →
  S = 270 →
  L = S * Q + 15 →
  Q = 6 := by sorry

end larger_divided_by_smaller_l1170_117046


namespace negation_of_universal_statement_l1170_117087

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔ 
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end negation_of_universal_statement_l1170_117087


namespace divisibility_implication_l1170_117010

/-- Represents a three-digit number with non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The product of digits of a three-digit number -/
def digit_product (n : ThreeDigitNumber) : Nat :=
  n.a * n.b * n.c

theorem divisibility_implication (n : ThreeDigitNumber) :
  (value n % digit_sum n = 0) ∧ (value n % digit_product n = 0) →
  90 * n.a % digit_sum n = 0 := by
  sorry

end divisibility_implication_l1170_117010


namespace least_product_of_primes_above_25_l1170_117084

theorem least_product_of_primes_above_25 (p q : ℕ) : 
  p.Prime → q.Prime → p > 25 → q > 25 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 899 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 25 → s > 25 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end least_product_of_primes_above_25_l1170_117084


namespace gordons_heavier_bag_weight_l1170_117074

theorem gordons_heavier_bag_weight (trace_bag_count : ℕ) (trace_bag_weight : ℝ)
  (gordon_bag_count : ℕ) (gordon_lighter_bag_weight : ℝ) :
  trace_bag_count = 5 →
  trace_bag_weight = 2 →
  gordon_bag_count = 2 →
  gordon_lighter_bag_weight = 3 →
  trace_bag_count * trace_bag_weight = gordon_lighter_bag_weight + gordon_heavier_bag_weight →
  gordon_heavier_bag_weight = 7 :=
by
  sorry

end gordons_heavier_bag_weight_l1170_117074


namespace equation_one_solution_equation_two_no_solution_l1170_117027

-- Equation 1
theorem equation_one_solution (x : ℚ) :
  x / (x - 1) = 3 / (2 * x - 2) - 2 ↔ x = 7 / 6 :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ (x : ℚ), (5 * x + 2) / (x^2 + x) = 3 / (x + 1) :=
sorry

end equation_one_solution_equation_two_no_solution_l1170_117027


namespace student_calculation_l1170_117056

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 124 → 
  (2 * chosen_number) - 138 = 110 := by
sorry

end student_calculation_l1170_117056


namespace athlete_exercise_time_l1170_117080

/-- Prove that given an athlete who burns 10 calories per minute while running,
    4 calories per minute while walking, burns 450 calories in total,
    and spends 35 minutes running, the total exercise time is 60 minutes. -/
theorem athlete_exercise_time
  (calories_per_minute_running : ℕ)
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (time_running : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : time_running = 35) :
  time_running + (total_calories_burned - calories_per_minute_running * time_running) / calories_per_minute_walking = 60 :=
by sorry

end athlete_exercise_time_l1170_117080


namespace superinverse_value_l1170_117033

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 27*x + 81

-- State that g is bijective
axiom g_bijective : Function.Bijective g

-- Define the superinverse property
def is_superinverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f ∘ g) x = Function.invFun g x

-- State that f is the superinverse of g
axiom f_is_superinverse : ∃ f : ℝ → ℝ, is_superinverse f g

-- The theorem to prove
theorem superinverse_value :
  ∃ f : ℝ → ℝ, is_superinverse f g ∧ |f (-289)| = 10 := by
  sorry

end superinverse_value_l1170_117033


namespace unique_integer_proof_l1170_117012

theorem unique_integer_proof : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (24.7 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25) :=
by
  use 612
  sorry

end unique_integer_proof_l1170_117012


namespace no_function_satisfies_conditions_l1170_117025

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end no_function_satisfies_conditions_l1170_117025


namespace product_of_primes_even_l1170_117037

theorem product_of_primes_even (P Q : ℕ+) : 
  Prime P.val → Prime Q.val → Prime (P.val - Q.val) → Prime (P.val + Q.val) → 
  Even (P.val * Q.val * (P.val - Q.val) * (P.val + Q.val)) := by
sorry

end product_of_primes_even_l1170_117037


namespace initial_crayon_packs_l1170_117088

theorem initial_crayon_packs : ℕ := by
  -- Define the cost of one pack of crayons
  let cost_per_pack : ℚ := 5/2

  -- Define the number of additional packs Michael buys
  let additional_packs : ℕ := 2

  -- Define the total value after purchase
  let total_value : ℚ := 15

  -- Define the initial number of packs (to be proven)
  let initial_packs : ℕ := 4

  -- Prove that the initial number of packs is 4
  have h : (cost_per_pack * (initial_packs + additional_packs : ℚ)) = total_value := by sorry

  -- Return the result
  exact initial_packs

end initial_crayon_packs_l1170_117088


namespace max_min_value_of_expression_l1170_117032

theorem max_min_value_of_expression (a b : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ Real.sqrt 3) 
  (hb : 1 ≤ b ∧ b ≤ Real.sqrt 3) :
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = 1) ∧
  (∃ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                y ∈ Set.Icc 1 (Real.sqrt 3) ∧ 
                (x^2 + y^2 - 1) / (x * y) = Real.sqrt 3) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 (Real.sqrt 3) → 
                y ∈ Set.Icc 1 (Real.sqrt 3) → 
                1 ≤ (x^2 + y^2 - 1) / (x * y) ∧ 
                (x^2 + y^2 - 1) / (x * y) ≤ Real.sqrt 3) :=
by sorry

end max_min_value_of_expression_l1170_117032


namespace expression_simplification_l1170_117093

theorem expression_simplification : 
  ((3 + 4 + 5 + 6)^2 / 4) + ((3 * 6 + 9)^2 / 3) = 324 := by sorry

end expression_simplification_l1170_117093


namespace bookstore_profit_percentage_l1170_117065

/-- Given three textbooks with their cost and selling prices, prove that the total profit percentage
    based on the combined selling prices is approximately 20.94%. -/
theorem bookstore_profit_percentage
  (cost1 : ℝ) (sell1 : ℝ) (cost2 : ℝ) (sell2 : ℝ) (cost3 : ℝ) (sell3 : ℝ)
  (h1 : cost1 = 44)
  (h2 : sell1 = 55)
  (h3 : cost2 = 58)
  (h4 : sell2 = 72)
  (h5 : cost3 = 83)
  (h6 : sell3 = 107) :
  let total_profit := (sell1 - cost1) + (sell2 - cost2) + (sell3 - cost3)
  let total_selling_price := sell1 + sell2 + sell3
  let profit_percentage := (total_profit / total_selling_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |profit_percentage - 20.94| < ε :=
by
  sorry


end bookstore_profit_percentage_l1170_117065


namespace amanda_walk_distance_l1170_117014

/-- Amanda's walk to Kimberly's house -/
theorem amanda_walk_distance :
  let initial_speed : ℝ := 2
  let time_before_break : ℝ := 1.5
  let break_duration : ℝ := 0.5
  let speed_after_break : ℝ := 3
  let total_time : ℝ := 3.5
  let distance_before_break := initial_speed * time_before_break
  let time_after_break := total_time - break_duration - time_before_break
  let distance_after_break := speed_after_break * time_after_break
  let total_distance := distance_before_break + distance_after_break
  total_distance = 7.5 := by sorry

end amanda_walk_distance_l1170_117014


namespace max_difference_reverse_digits_l1170_117041

theorem max_difference_reverse_digits (q r : ℕ) : 
  (10 ≤ q) ∧ (q < 100) ∧  -- q is a two-digit number
  (10 ≤ r) ∧ (r < 100) ∧  -- r is a two-digit number
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ q = 10*x + y ∧ r = 10*y + x) ∧  -- q and r have reversed digits
  (q - r < 30 ∨ r - q < 30) →  -- positive difference is less than 30
  (q - r ≤ 27 ∧ r - q ≤ 27) :=
by sorry

end max_difference_reverse_digits_l1170_117041


namespace square_plus_one_geq_two_abs_l1170_117072

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l1170_117072


namespace sufficient_not_necessary_condition_l1170_117020

theorem sufficient_not_necessary_condition :
  (∀ x > 0, x + (1/18) / (2*x) ≥ 1/3) ∧
  (∃ a ≠ 1/18, ∀ x > 0, x + a / (2*x) ≥ 1/3) := by
  sorry

end sufficient_not_necessary_condition_l1170_117020


namespace evaluate_g_l1170_117057

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 4 - 2 * g (-2) = 47 := by
  sorry

end evaluate_g_l1170_117057


namespace pie_division_l1170_117058

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 8/9 ∧ num_people = 4 → 
  total_pie / num_people = 2/9 := by sorry

end pie_division_l1170_117058


namespace trailing_zeros_factorial_product_mod_100_l1170_117003

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * Nat.factorial (i + 1)) 1

theorem trailing_zeros_factorial_product_mod_100 :
  trailingZeros (factorialProduct 50) % 100 = 12 := by
  sorry

end trailing_zeros_factorial_product_mod_100_l1170_117003


namespace tan_alpha_value_l1170_117013

theorem tan_alpha_value (h : Real.tan (π - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end tan_alpha_value_l1170_117013


namespace symmetry_about_center_three_zeros_existence_l1170_117070

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

-- Theorem for symmetry (Option B)
theorem symmetry_about_center (b : ℝ) :
  ∀ x : ℝ, f 0 b x + f 0 b (-x) = 2 :=
sorry

-- Theorem for three zeros (Option C)
theorem three_zeros_existence (a : ℝ) (h : a > -4) :
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  f a (a^2/4) x = 0 ∧ f a (a^2/4) y = 0 ∧ f a (a^2/4) z = 0 :=
sorry

end symmetry_about_center_three_zeros_existence_l1170_117070


namespace money_problem_l1170_117026

theorem money_problem (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 := by
  sorry

end money_problem_l1170_117026


namespace abs_m_minus_n_eq_two_sqrt_three_l1170_117060

theorem abs_m_minus_n_eq_two_sqrt_three (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 3 := by
sorry

end abs_m_minus_n_eq_two_sqrt_three_l1170_117060


namespace circle_and_line_theorem_l1170_117016

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line on which the center of C lies
def CenterLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 - 2 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)

-- Define point P
def P : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define a line given its equation ax + by + c = 0
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem circle_and_line_theorem :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center ∈ CenterLine ∧
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    (∃ (a b c : ℝ),
      (Line a b c = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 3 = 0} ∨
       Line a b c = {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 + 3 = 0}) ∧
      P ∈ Line a b c ∧
      (∃ (q : ℝ × ℝ), q ∈ XAxis ∧ q ∈ Line a b c) ∧
      (∃ (t : ℝ × ℝ), t ∈ Circle center radius ∧ t ∈ Line a b c)) ∧
    center = (2, 2) ∧
    radius = 1 :=
by sorry

end circle_and_line_theorem_l1170_117016


namespace equation_solution_l1170_117064

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end equation_solution_l1170_117064


namespace alissa_earring_ratio_l1170_117034

/-- The ratio of Alissa's total earrings to the number of earrings she was given -/
def earring_ratio (barbie_pairs : ℕ) (alissa_total : ℕ) : ℚ :=
  let barbie_total := 2 * barbie_pairs
  let alissa_given := barbie_total / 2
  alissa_total / alissa_given

/-- Theorem stating the ratio of Alissa's total earrings to the number of earrings she was given -/
theorem alissa_earring_ratio :
  let barbie_pairs := 12
  let alissa_total := 36
  earring_ratio barbie_pairs alissa_total = 3 := by
  sorry

end alissa_earring_ratio_l1170_117034


namespace sin_330_degrees_l1170_117050

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1170_117050


namespace circle_area_doubling_l1170_117043

theorem circle_area_doubling (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 2 * π * r^2) → (r = n * (Real.sqrt 2 + 1)) := by
  sorry

end circle_area_doubling_l1170_117043


namespace draining_cylinder_height_change_rate_l1170_117096

/-- The rate of change of liquid level height in a draining cylindrical container -/
theorem draining_cylinder_height_change_rate 
  (d : ℝ) -- diameter of the base
  (dV_dt : ℝ) -- rate of volume change (negative for draining)
  (h : ℝ → ℝ) -- height of liquid as a function of time
  (t : ℝ) -- time variable
  (h_diff : Differentiable ℝ h) -- h is differentiable
  (cylinder_volume : ∀ t, π * (d/2)^2 * h t = -dV_dt * t + C) -- volume equation
  (h_positive : ∀ t, h t > 0) -- height is always positive
  (dV_dt_negative : dV_dt < 0) -- volume is decreasing
  (d_positive : d > 0) -- diameter is positive
  (h_init : h 0 > 0) -- initial height is positive
  : d = 2 → dV_dt = -0.01 → deriv h t = -0.01 / π := by
  sorry

end draining_cylinder_height_change_rate_l1170_117096


namespace trains_meeting_point_l1170_117009

/-- Proves that two trains traveling towards each other on a 200 km track,
    with train A moving at 60 km/h and train B moving at 90 km/h,
    will meet at a distance of 80 km from train A's starting point. -/
theorem trains_meeting_point (distance : ℝ) (speed_A : ℝ) (speed_B : ℝ)
  (h1 : distance = 200)
  (h2 : speed_A = 60)
  (h3 : speed_B = 90) :
  speed_A * (distance / (speed_A + speed_B)) = 80 :=
by sorry

end trains_meeting_point_l1170_117009


namespace A_subset_B_A_equals_B_iff_l1170_117099

variable (a : ℝ)

def A : Set ℝ := {x | x^2 + a = x}
def B : Set ℝ := {x | (x^2 + a)^2 + a = x}

axiom A_nonempty : A a ≠ ∅

theorem A_subset_B : A a ⊆ B a := by sorry

theorem A_equals_B_iff : 
  A a = B a ↔ -3/4 ≤ a ∧ a ≤ 1/4 := by sorry

end A_subset_B_A_equals_B_iff_l1170_117099


namespace hexagon_diagonals_l1170_117002

/-- A hexagon is a polygon with 6 sides. -/
def Hexagon : Type := Unit

/-- The number of sides in a hexagon. -/
def num_sides (h : Hexagon) : ℕ := 6

/-- The number of diagonals in a polygon. -/
def num_diagonals (h : Hexagon) : ℕ := sorry

/-- Theorem: The number of diagonals in a hexagon is 9. -/
theorem hexagon_diagonals (h : Hexagon) : num_diagonals h = 9 := by sorry

end hexagon_diagonals_l1170_117002


namespace nadine_dog_cleaning_time_l1170_117063

/-- The time Nadine spends cleaning her dog -/
def dog_cleaning_time (hosing_time shampoo_time shampoo_count : ℕ) : ℕ :=
  hosing_time + shampoo_time * shampoo_count

/-- Theorem stating the total time Nadine spends cleaning her dog -/
theorem nadine_dog_cleaning_time :
  dog_cleaning_time 10 15 3 = 55 := by
  sorry

end nadine_dog_cleaning_time_l1170_117063


namespace smallest_constant_l1170_117062

-- Define the properties of the function f
def FunctionProperties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 0 = 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Theorem statement
theorem smallest_constant (f : ℝ → ℝ) (h : FunctionProperties f) :
  (∃ c > 0, ∀ x ∈ Set.Icc 0 1, f x ≤ c * x) ∧
  (∀ c < 2, ∃ x ∈ Set.Icc 0 1, f x > c * x) :=
sorry

end smallest_constant_l1170_117062


namespace root_product_l1170_117028

theorem root_product (d e : ℤ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 → x^7 - d*x^3 - e = 0) → 
  d * e = 70 := by
  sorry

end root_product_l1170_117028


namespace divisibility_relation_l1170_117052

theorem divisibility_relation (p a b n : ℕ) : 
  p ≥ 3 → 
  Nat.Prime p → 
  Nat.Coprime a b → 
  p ∣ (a^(2^n) + b^(2^n)) → 
  2^(n+1) ∣ (p-1) :=
by sorry

end divisibility_relation_l1170_117052


namespace cookie_jar_problem_l1170_117094

theorem cookie_jar_problem (initial_cookies : ℕ) (x : ℕ) 
  (h1 : initial_cookies = 7)
  (h2 : initial_cookies - 1 = (initial_cookies + x) / 2) : 
  x = 5 := by
  sorry

end cookie_jar_problem_l1170_117094


namespace price_reduction_achieves_target_profit_l1170_117001

/-- Represents the price reduction and resulting sales and profit changes for a toy product. -/
structure ToyPricing where
  initialSales : ℕ
  initialProfit : ℕ
  salesIncrease : ℕ
  priceReduction : ℕ
  targetProfit : ℕ

/-- Calculates the daily profit after price reduction. -/
def dailyProfitAfterReduction (t : ToyPricing) : ℕ :=
  (t.initialProfit - t.priceReduction) * (t.initialSales + t.salesIncrease * t.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in the target daily profit. -/
theorem price_reduction_achieves_target_profit (t : ToyPricing) 
  (h1 : t.initialSales = 20)
  (h2 : t.initialProfit = 40)
  (h3 : t.salesIncrease = 2)
  (h4 : t.targetProfit = 1200)
  (h5 : t.priceReduction = 20) :
  dailyProfitAfterReduction t = t.targetProfit :=
by
  sorry

#eval dailyProfitAfterReduction { 
  initialSales := 20, 
  initialProfit := 40, 
  salesIncrease := 2, 
  priceReduction := 20, 
  targetProfit := 1200 
}

end price_reduction_achieves_target_profit_l1170_117001


namespace travel_time_ratio_l1170_117011

theorem travel_time_ratio : 
  let distance : ℝ := 600
  let initial_time : ℝ := 5
  let new_speed : ℝ := 80
  let new_time : ℝ := distance / new_speed
  new_time / initial_time = 1.5 := by sorry

end travel_time_ratio_l1170_117011


namespace exists_intransitive_dice_l1170_117045

/-- Represents a die with 6 faces -/
def Die := Fin 6 → Nat

/-- The probability that one die shows a higher number than another -/
def winProbability (d1 d2 : Die) : ℚ :=
  (Finset.sum Finset.univ (λ i => 
    Finset.sum Finset.univ (λ j => 
      if d1 i > d2 j then 1 else 0
    )
  )) / 36

/-- Predicate for one die winning over another -/
def wins (d1 d2 : Die) : Prop := winProbability d1 d2 > 1/2

/-- Theorem stating the existence of three dice with the desired properties -/
theorem exists_intransitive_dice : ∃ (A B C : Die),
  wins B A ∧ wins C B ∧ wins A C := by sorry

end exists_intransitive_dice_l1170_117045


namespace total_pencils_count_l1170_117040

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def total_people : ℕ := 8

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_box * total_people

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end total_pencils_count_l1170_117040


namespace positive_expression_l1170_117054

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end positive_expression_l1170_117054


namespace sin_pi_fourth_plus_alpha_l1170_117049

theorem sin_pi_fourth_plus_alpha (α : ℝ) (h : Real.cos (π/4 - α) = 1/3) :
  Real.sin (π/4 + α) = 1/3 := by
  sorry

end sin_pi_fourth_plus_alpha_l1170_117049


namespace square_perimeter_from_area_l1170_117079

-- Define a square with a given area
def Square (area : ℝ) : Type :=
  { side : ℝ // side * side = area }

-- Define the perimeter of a square
def perimeter (s : Square 625) : ℝ :=
  4 * s.val

-- Theorem statement
theorem square_perimeter_from_area :
  ∀ s : Square 625, perimeter s = 100 := by
  sorry

end square_perimeter_from_area_l1170_117079


namespace canoe_rowing_probability_l1170_117047

def left_oar_prob : ℚ := 3/5
def right_oar_prob : ℚ := 3/5

theorem canoe_rowing_probability :
  let prob_at_least_one_oar := 
    left_oar_prob * right_oar_prob + 
    left_oar_prob * (1 - right_oar_prob) + 
    (1 - left_oar_prob) * right_oar_prob
  prob_at_least_one_oar = 21/25 := by
sorry

end canoe_rowing_probability_l1170_117047


namespace percentage_problem_l1170_117038

theorem percentage_problem (P : ℝ) : P = 20 → 0.25 * 1280 = (P / 100) * 650 + 190 := by
  sorry

end percentage_problem_l1170_117038


namespace grade_assignment_count_l1170_117067

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end grade_assignment_count_l1170_117067


namespace circle_center_reflection_l1170_117029

/-- Reflects a point (x, y) about the line y = x -/
def reflect_about_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem circle_center_reflection :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_equals_x original_center = (-3, 8) := by
  sorry

end circle_center_reflection_l1170_117029


namespace arithmetic_expression_equality_l1170_117030

theorem arithmetic_expression_equality : 72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 := by
  sorry

end arithmetic_expression_equality_l1170_117030


namespace combined_speed_difference_l1170_117008

-- Define the speed functions for each train
def zA (s : ℝ) : ℝ := s^2 + 2*s
def zB (s : ℝ) : ℝ := 2*s^2 + 3*s + 1
def zC (s : ℝ) : ℝ := s^3 - 4*s

-- Define the time constraints for each train
def trainA_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 7
def trainB_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 5
def trainC_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 4

-- Theorem statement
theorem combined_speed_difference :
  trainA_time_constraint 7 ∧
  trainA_time_constraint 2 ∧
  trainB_time_constraint 5 ∧
  trainB_time_constraint 2 ∧
  trainC_time_constraint 4 ∧
  trainC_time_constraint 2 →
  (zA 7 - zA 2) + (zB 5 - zB 2) + (zC 4 - zC 2) = 154 := by
  sorry

end combined_speed_difference_l1170_117008


namespace function_negative_on_interval_l1170_117005

theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  -Real.sqrt 2 / 2 < m ∧ m < 0 := by
sorry

end function_negative_on_interval_l1170_117005


namespace chord_length_in_circle_l1170_117022

theorem chord_length_in_circle (r d c : ℝ) (hr : r = 3) (hd : d = 2) :
  r^2 = d^2 + (c/2)^2 → c = 2 * Real.sqrt 5 := by
  sorry

end chord_length_in_circle_l1170_117022


namespace fraction_equality_l1170_117085

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + 2*y) / (x - 4*y) = -3) : 
  (2*x + 8*y) / (4*x - 2*y) = 38/13 := by
  sorry

end fraction_equality_l1170_117085


namespace number_puzzle_l1170_117051

theorem number_puzzle : ∃ N : ℚ, N = 90 ∧ 3 + (1/2) * (1/3) * (1/5) * N = (1/15) * N := by
  sorry

end number_puzzle_l1170_117051


namespace reciprocal_sum_relation_l1170_117031

theorem reciprocal_sum_relation (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (x + y) := by
  sorry

end reciprocal_sum_relation_l1170_117031


namespace family_movie_night_l1170_117048

/-- Proves the number of children in a family given ticket prices and payment information --/
theorem family_movie_night (regular_ticket_price : ℕ) 
                            (child_discount : ℕ)
                            (payment : ℕ)
                            (change : ℕ)
                            (num_adults : ℕ) :
  regular_ticket_price = 9 →
  child_discount = 2 →
  payment = 40 →
  change = 1 →
  num_adults = 2 →
  ∃ (num_children : ℕ),
    num_children = 3 ∧
    payment - change = 
      num_adults * regular_ticket_price + 
      num_children * (regular_ticket_price - child_discount) :=
by
  sorry


end family_movie_night_l1170_117048


namespace sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l1170_117089

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem sum_squares_two_odds_not_perfect_square (a b : ℤ) (ha : is_odd a) (hb : is_odd b) :
  ¬∃ n : ℤ, a^2 + b^2 = n^2 := by sorry

theorem sum_squares_three_odds_not_perfect_square (a b c : ℤ) (ha : is_odd a) (hb : is_odd b) (hc : is_odd c) :
  ¬∃ m : ℤ, a^2 + b^2 + c^2 = m^2 := by sorry

end sum_squares_two_odds_not_perfect_square_sum_squares_three_odds_not_perfect_square_l1170_117089
