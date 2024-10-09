import Mathlib

namespace ratio_20_to_10_exists_l588_58802

theorem ratio_20_to_10_exists (x : ℕ) (h : x = 20 * 10) : x = 200 :=
by sorry

end ratio_20_to_10_exists_l588_58802


namespace binomial_expansion_fifth_term_constant_l588_58805

open Classical -- Allows the use of classical logic

noncomputable def binomial_term (n r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (x ^ (n - r) / (x ^ r * (2 ^ r / x ^ r)))

theorem binomial_expansion_fifth_term_constant (n : ℕ) :
  (binomial_term n 4 x = (x ^ (n - 3 * 4) * (-2) ^ 4)) → n = 12 := by
  intro h
  sorry

end binomial_expansion_fifth_term_constant_l588_58805


namespace find_principal_sum_l588_58898

theorem find_principal_sum (P R : ℝ) (SI CI : ℝ) 
  (h1 : SI = 10200) 
  (h2 : CI = 11730) 
  (h3 : SI = P * R * 2 / 100)
  (h4 : CI = P * (1 + R / 100)^2 - P) :
  P = 17000 :=
by
  sorry

end find_principal_sum_l588_58898


namespace value_is_50_cents_l588_58853

-- Define Leah's total number of coins and the condition on the number of nickels and pennies.
variables (p n : ℕ)

-- Leah has a total of 18 coins
def total_coins : Prop := n + p = 18

-- Condition for nickels and pennies
def condition : Prop := p = n + 2

-- Calculate the total value of Leah's coins and check if it equals 50 cents
def total_value : ℕ := 5 * n + p

-- Proposition stating that under given conditions, total value is 50 cents
theorem value_is_50_cents (h1 : total_coins p n) (h2 : condition p n) :
  total_value p n = 50 := sorry

end value_is_50_cents_l588_58853


namespace find_y_l588_58861

theorem find_y (x y : ℤ) (q : ℤ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x = q * y + 6) (h4 : (x : ℚ) / y = 96.15) : y = 40 :=
sorry

end find_y_l588_58861


namespace math_problem_l588_58814

theorem math_problem 
  (a b c : ℝ) 
  (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : a^2 + b^2 = c^2 + ab) : 
  c^2 + ab < a*c + b*c := 
sorry

end math_problem_l588_58814


namespace max_value_on_interval_l588_58893

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / Real.exp x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := ((2 * a * x + b) * Real.exp x - (a * x^2 + b * x + c)) / Real.exp (2 * x)

variable (a b c : ℝ)

-- Given conditions
axiom pos_a : a > 0
axiom zero_point_neg3 : f' a b c (-3) = 0
axiom zero_point_0 : f' a b c 0 = 0
axiom min_value_neg3 : f a b c (-3) = -Real.exp 3

-- Goal: Maximum value of f(x) on the interval [-5, ∞) is 5e^5.
theorem max_value_on_interval : ∃ y ∈ Set.Ici (-5), f a b c y = 5 * Real.exp 5 := by
  sorry

end max_value_on_interval_l588_58893


namespace deck_length_is_30_l588_58897

theorem deck_length_is_30
  (x : ℕ)
  (h1 : ∀ a : ℕ, a = 40 * x)
  (h2 : ∀ b : ℕ, b = 3 * a + 1 * a ∧ b = 4800) :
  x = 30 := by
  sorry

end deck_length_is_30_l588_58897


namespace csc_square_value_l588_58839

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 ∨ x = 1 then 0 -- provision for the illegal inputs as defined in the question
else 1/(x / (x - 1))

theorem csc_square_value (t : ℝ) (ht : 0 ≤ t ∧ t ≤ π / 2) :
  f (1 / (Real.sin t)^2) = (Real.cos t)^2 :=
by
  sorry

end csc_square_value_l588_58839


namespace domain_of_function_l588_58851

theorem domain_of_function :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ↔ (1 - x ≥ 0 ∧ x ≠ 0) :=
by
  sorry

end domain_of_function_l588_58851


namespace gcd_m_n_l588_58856

def m : ℕ := 131^2 + 243^2 + 357^2
def n : ℕ := 130^2 + 242^2 + 358^2

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l588_58856


namespace ratio_of_functions_l588_58835

def f (x : ℕ) : ℕ := 3 * x + 4
def g (x : ℕ) : ℕ := 4 * x - 3

theorem ratio_of_functions :
  f (g (f 3)) * 121 = 151 * g (f (g 3)) :=
by
  sorry

end ratio_of_functions_l588_58835


namespace probability_statements_l588_58810

-- Assigning probabilities
def p_hit := 0.9
def p_miss := 1 - p_hit

-- Definitions based on the problem conditions
def shoot_4_times (shots : List Bool) : Bool :=
  shots.length = 4 ∧ ∀ (s : Bool), s ∈ shots → (s = true → s ≠ false) ∧ (s = false → s ≠ true ∧ s ≠ 0)

-- Statements derived from the conditions
def prob_shot_3 := p_hit

def prob_exact_3_out_of_4 := 
  let binom_4_3 := 4
  binom_4_3 * (p_hit^3) * (p_miss^1)

def prob_at_least_1_out_of_4 := 1 - (p_miss^4)

-- The equivalence proof
theorem probability_statements : 
  (prob_shot_3 = 0.9) ∧ 
  (prob_exact_3_out_of_4 = 0.2916) ∧ 
  (prob_at_least_1_out_of_4 = 0.9999) := 
by 
  sorry

end probability_statements_l588_58810


namespace visible_black_area_ratio_l588_58874

-- Definitions for circle areas as nonnegative real numbers
variables (A_b A_g A_w : ℝ) (hA_b : 0 ≤ A_b) (hA_g : 0 ≤ A_g) (hA_w : 0 ≤ A_w)
-- Condition: Initial visible black area is 7 times the white area
axiom initial_visible_black_area : 7 * A_w = A_b

-- Definition of new visible black area after movement
def new_visible_black_area := A_b - A_w

-- Prove the ratio of the visible black regions before and after moving the circles
theorem visible_black_area_ratio :
  (7 * A_w) / ((7 * A_w) - A_w) = 7 / 6 :=
by { sorry }

end visible_black_area_ratio_l588_58874


namespace find_a_l588_58854

theorem find_a (a : ℤ) (h : ∃ x1 x2 : ℤ, (x - x1) * (x - x2) = (x - a) * (x - 8) - 1) : a = 8 :=
sorry

end find_a_l588_58854


namespace possible_b_value_l588_58866

theorem possible_b_value (a b : ℤ) (h1 : a = 3^20) (h2 : a ≡ b [ZMOD 10]) : b = 2011 :=
by sorry

end possible_b_value_l588_58866


namespace division_of_cookies_l588_58882

theorem division_of_cookies (n p : Nat) (h1 : n = 24) (h2 : p = 6) : n / p = 4 :=
by sorry

end division_of_cookies_l588_58882


namespace arithmetic_seq_15th_term_is_53_l588_58841

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Original terms given
def a₁ : ℤ := -3
def d : ℤ := 4
def n : ℕ := 15

-- Prove that the 15th term is 53
theorem arithmetic_seq_15th_term_is_53 :
  arithmetic_seq a₁ d n = 53 :=
by
  sorry

end arithmetic_seq_15th_term_is_53_l588_58841


namespace kids_go_to_camp_l588_58808

theorem kids_go_to_camp (total_kids : ℕ) (kids_stay_home : ℕ) (h1 : total_kids = 898051) (h2 : kids_stay_home = 268627) : total_kids - kids_stay_home = 629424 :=
by
  sorry

end kids_go_to_camp_l588_58808


namespace solve_eqs_l588_58899

theorem solve_eqs (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) : x = -8 ∧ y = -1 := 
by
  sorry

end solve_eqs_l588_58899


namespace relationship_of_a_and_b_l588_58816

theorem relationship_of_a_and_b (a b : ℝ) (h_b_nonzero: b ≠ 0)
  (m n : ℤ) (h_intersection : ∃ (m n : ℤ), n = m^3 - a * m^2 - b * m ∧ n = a * m + b) :
  2 * a - b + 8 = 0 :=
  sorry

end relationship_of_a_and_b_l588_58816


namespace sin_30_eq_half_l588_58896

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l588_58896


namespace pima_initial_investment_l588_58850

/-- Pima's initial investment in Ethereum. The investment value gained 25% in the first week and 50% of its current value in the second week. The final investment value is $750. -/
theorem pima_initial_investment (I : ℝ) 
  (h1 : 1.25 * I * 1.5 = 750) : I = 400 :=
sorry

end pima_initial_investment_l588_58850


namespace arithmetic_expression_evaluation_l588_58819

theorem arithmetic_expression_evaluation :
  2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end arithmetic_expression_evaluation_l588_58819


namespace triangle_inequality_equality_condition_l588_58885

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end triangle_inequality_equality_condition_l588_58885


namespace area_of_circular_platform_l588_58844

theorem area_of_circular_platform (d : ℝ) (h : d = 2) : ∃ (A : ℝ), A = Real.pi ∧ A = π *(d / 2)^2 := by
  sorry

end area_of_circular_platform_l588_58844


namespace find_number_l588_58840

theorem find_number (n : ℝ) (h : 1 / 2 * n + 7 = 17) : n = 20 :=
by
  sorry

end find_number_l588_58840


namespace initial_average_is_100_l588_58804

-- Definitions based on the conditions from step a)
def students : ℕ := 10
def wrong_mark : ℕ := 90
def correct_mark : ℕ := 10
def correct_average : ℝ := 92

-- Initial average marks before correcting the error
def initial_average_marks (A : ℝ) : Prop :=
  10 * A = (students * correct_average) + (wrong_mark - correct_mark)

theorem initial_average_is_100 :
  ∃ A : ℝ, initial_average_marks A ∧ A = 100 :=
by {
  -- We are defining the placeholder for the actual proof.
  sorry
}

end initial_average_is_100_l588_58804


namespace red_light_at_A_prob_calc_l588_58831

-- Defining the conditions
def count_total_permutations : ℕ := Nat.factorial 4 / Nat.factorial 1
def count_favorable_permutations : ℕ := Nat.factorial 3 / Nat.factorial 1

-- Calculating the probability
def probability_red_at_A : ℚ := count_favorable_permutations / count_total_permutations

-- Statement to be proved
theorem red_light_at_A_prob_calc : probability_red_at_A = 1 / 4 :=
by
  sorry

end red_light_at_A_prob_calc_l588_58831


namespace probability_interval_l588_58860

theorem probability_interval (P_A P_B p : ℝ) (hP_A : P_A = 2 / 3) (hP_B : P_B = 3 / 5) :
  4 / 15 ≤ p ∧ p ≤ 3 / 5 := sorry

end probability_interval_l588_58860


namespace horror_movie_more_than_triple_romance_l588_58859

-- Definitions and Conditions
def tickets_sold_romance : ℕ := 25
def tickets_sold_horror : ℕ := 93
def triple_tickets_romance := 3 * tickets_sold_romance

-- Theorem Statement
theorem horror_movie_more_than_triple_romance :
  (tickets_sold_horror - triple_tickets_romance) = 18 :=
by
  sorry

end horror_movie_more_than_triple_romance_l588_58859


namespace union_intersection_l588_58801

variable (a : ℝ)

def setA (a : ℝ) : Set ℝ := { x | (x - 3) * (x - a) = 0 }
def setB : Set ℝ := {1, 4}

theorem union_intersection (a : ℝ) :
  (if a = 3 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = ∅ else 
   if a = 1 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {1} else
   if a = 4 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {4} else
   setA a ∪ setB = {1, 3, 4, a} ∧ setA a ∩ setB = ∅) := sorry

end union_intersection_l588_58801


namespace students_without_scholarships_l588_58864

theorem students_without_scholarships :
  let total_students := 300
  let full_merit_percent := 0.05
  let half_merit_percent := 0.10
  let sports_percent := 0.03
  let need_based_percent := 0.07
  let full_merit_and_sports_percent := 0.01
  let half_merit_and_need_based_percent := 0.02
  let full_merit := full_merit_percent * total_students
  let half_merit := half_merit_percent * total_students
  let sports := sports_percent * total_students
  let need_based := need_based_percent * total_students
  let full_merit_and_sports := full_merit_and_sports_percent * total_students
  let half_merit_and_need_based := half_merit_and_need_based_percent * total_students
  let total_with_scholarships := (full_merit + half_merit + sports + need_based) - (full_merit_and_sports + half_merit_and_need_based)
  let students_without_scholarships := total_students - total_with_scholarships
  students_without_scholarships = 234 := 
by
  sorry

end students_without_scholarships_l588_58864


namespace total_blue_marbles_l588_58891

def jason_blue_marbles : Nat := 44
def tom_blue_marbles : Nat := 24

theorem total_blue_marbles : jason_blue_marbles + tom_blue_marbles = 68 := by
  sorry

end total_blue_marbles_l588_58891


namespace smallest_sum_xyz_l588_58845

theorem smallest_sum_xyz (x y z : ℕ) (h : x * y * z = 40320) : x + y + z ≥ 103 :=
sorry

end smallest_sum_xyz_l588_58845


namespace minimum_teachers_to_cover_all_subjects_l588_58837

/- Define the problem conditions -/
def maths_teachers := 7
def physics_teachers := 6
def chemistry_teachers := 5
def max_subjects_per_teacher := 3

/- The proof statement -/
theorem minimum_teachers_to_cover_all_subjects : 
  (maths_teachers + physics_teachers + chemistry_teachers) / max_subjects_per_teacher = 7 :=
sorry

end minimum_teachers_to_cover_all_subjects_l588_58837


namespace units_digit_fraction_l588_58833

theorem units_digit_fraction : (2^3 * 31 * 33 * 17 * 7) % 10 = 6 := by
  sorry

end units_digit_fraction_l588_58833


namespace atleast_one_alarm_rings_on_time_l588_58863

def probability_alarm_A_rings := 0.80
def probability_alarm_B_rings := 0.90

def probability_atleast_one_rings := 1 - (1 - probability_alarm_A_rings) * (1 - probability_alarm_B_rings)

theorem atleast_one_alarm_rings_on_time :
  probability_atleast_one_rings = 0.98 :=
sorry

end atleast_one_alarm_rings_on_time_l588_58863


namespace pizza_promotion_savings_l588_58813

theorem pizza_promotion_savings :
  let regular_price : ℕ := 18
  let promo_price : ℕ := 5
  let num_pizzas : ℕ := 3
  let total_regular_price := num_pizzas * regular_price
  let total_promo_price := num_pizzas * promo_price
  let total_savings := total_regular_price - total_promo_price
  total_savings = 39 :=
by
  sorry

end pizza_promotion_savings_l588_58813


namespace find_d_l588_58847

noncomputable def d : ℝ := 3.44

theorem find_d :
  (∃ x : ℝ, (3 * x^2 + 19 * x - 84 = 0) ∧ x = ⌊d⌋) ∧
  (∃ y : ℝ, (5 * y^2 - 26 * y + 12 = 0) ∧ y = d - ⌊d⌋) →
  d = 3.44 :=
by
  sorry

end find_d_l588_58847


namespace inequality_solution_l588_58832

theorem inequality_solution :
  {x : ℝ | (x^2 + 5 * x) / ((x - 3) ^ 2) ≥ 0} = {x | x < -5} ∪ {x | 0 ≤ x ∧ x < 3} ∪ {x | x > 3} :=
by
  sorry

end inequality_solution_l588_58832


namespace inequality_implies_range_of_a_l588_58803

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2 * a) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end inequality_implies_range_of_a_l588_58803


namespace Benjie_is_older_by_5_l588_58852

def BenjieAge : ℕ := 6
def MargoFutureAge : ℕ := 4
def YearsToFuture : ℕ := 3

theorem Benjie_is_older_by_5 :
  BenjieAge - (MargoFutureAge - YearsToFuture) = 5 :=
by
  sorry

end Benjie_is_older_by_5_l588_58852


namespace restore_original_expression_l588_58858

-- Define the altered product and correct restored products
def original_expression_1 := 4 * 5 * 4 * 7 * 4
def original_expression_2 := 4 * 7 * 4 * 5 * 4
def altered_product := 2247
def corrected_product := 2240

-- Statement that proves the corrected restored product given the altered product
theorem restore_original_expression :
  (4 * 5 * 4 * 7 * 4 = corrected_product ∨ 4 * 7 * 4 * 5 * 4 = corrected_product) :=
sorry

end restore_original_expression_l588_58858


namespace tim_pays_300_l588_58800

def mri_cost : ℕ := 1200
def doctor_rate_per_hour : ℕ := 300
def examination_time_in_hours : ℕ := 1 / 2
def consultation_fee : ℕ := 150
def insurance_coverage : ℚ := 0.8

def examination_cost : ℕ := doctor_rate_per_hour * examination_time_in_hours
def total_cost_before_insurance : ℕ := mri_cost + examination_cost + consultation_fee
def insurance_coverage_amount : ℚ := total_cost_before_insurance * insurance_coverage
def amount_tim_pays : ℚ := total_cost_before_insurance - insurance_coverage_amount

theorem tim_pays_300 : amount_tim_pays = 300 := 
by
  -- proof goes here
  sorry

end tim_pays_300_l588_58800


namespace question1_question2_question3_l588_58809

open Set

-- Define sets A and B
def A := { x : ℝ | x^2 + 6 * x + 5 < 0 }
def B := { x : ℝ | -1 ≤ x ∧ x < 1 }

-- Universal set U is implicitly ℝ in Lean

-- Question 1: Prove A ∩ B = ∅
theorem question1 : A ∩ B = ∅ := 
sorry

-- Question 2: Prove complement of A ∪ B in ℝ is (-∞, -5] ∪ [1, ∞)
theorem question2 : compl (A ∪ B) = { x : ℝ | x ≤ -5 } ∪ { x : ℝ | x ≥ 1 } := 
sorry

-- Define set C which depends on parameter a
def C (a: ℝ) := { x : ℝ | x < a }

-- Question 3: Prove if B ∩ C = B, then a ≥ 1
theorem question3 (a : ℝ) (h : B ∩ C a = B) : a ≥ 1 := 
sorry

end question1_question2_question3_l588_58809


namespace cubed_difference_l588_58870

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference_l588_58870


namespace distance_covered_downstream_l588_58868

noncomputable def speed_in_still_water := 16 -- km/hr
noncomputable def speed_of_stream := 5 -- km/hr
noncomputable def time_taken := 5 -- hours
noncomputable def effective_speed_downstream := speed_in_still_water + speed_of_stream -- km/hr

theorem distance_covered_downstream :
  (effective_speed_downstream * time_taken = 105) :=
by
  sorry

end distance_covered_downstream_l588_58868


namespace perpendicular_vectors_x_value_l588_58828

-- Define the vectors a and b
def a : ℝ × ℝ := (3, -1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the dot product function for vectors in ℝ^2
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- The mathematical statement to prove
theorem perpendicular_vectors_x_value (x : ℝ) (h : dot_product a (b x) = 0) : x = 3 :=
by
  sorry

end perpendicular_vectors_x_value_l588_58828


namespace problem_1_problem_2_l588_58895

open Set

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_1 (a : ℝ) (ha : a = 1) : 
  {x : ℝ | x^2 - 4 * a * x + 3 * a ^ 2 < 0} ∩ {x : ℝ | (x - 3) / (x - 2) ≤ 0} = Ioo 2 3 :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0) → ¬((x - 3) / (x - 2) ≤ 0)) →
  (∃ x : ℝ, ¬((x - 3) / (x - 2) ≤ 0) → ¬(x^2 - 4 * a * x + 3 * a ^ 2 < 0)) →
  1 < a ∧ a ≤ 2 :=
sorry

end problem_1_problem_2_l588_58895


namespace max_area_of_rect_l588_58811

theorem max_area_of_rect (x y : ℝ) (h1 : x + y = 10) : 
  x * y ≤ 25 :=
by 
  sorry

end max_area_of_rect_l588_58811


namespace x_finishes_work_alone_in_18_days_l588_58865

theorem x_finishes_work_alone_in_18_days
  (y_days : ℕ) (y_worked : ℕ) (x_remaining_days : ℝ)
  (hy : y_days = 15) (hy_worked : y_worked = 10) 
  (hx_remaining : x_remaining_days = 6.000000000000001) :
  ∃ (x_days : ℝ), x_days = 18 :=
by 
  sorry

end x_finishes_work_alone_in_18_days_l588_58865


namespace find_purchase_price_l588_58823

noncomputable def purchase_price (a : ℝ) : ℝ := a
def retail_price : ℝ := 1100
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.1

theorem find_purchase_price (a : ℝ) (h : purchase_price a * (1 + profit_rate) = retail_price * discount_rate) : a = 800 := by
  sorry

end find_purchase_price_l588_58823


namespace celia_time_correct_lexie_time_correct_nik_time_correct_l588_58822

noncomputable def lexie_time_per_mile : ℝ := 20
noncomputable def celia_time_per_mile : ℝ := lexie_time_per_mile / 2
noncomputable def nik_time_per_mile : ℝ := lexie_time_per_mile / 1.5

noncomputable def total_distance : ℝ := 30

-- Calculate the baseline running time without obstacles
noncomputable def lexie_baseline_time : ℝ := lexie_time_per_mile * total_distance
noncomputable def celia_baseline_time : ℝ := celia_time_per_mile * total_distance
noncomputable def nik_baseline_time : ℝ := nik_time_per_mile * total_distance

-- Additional time due to obstacles
noncomputable def celia_muddy_extra_time : ℝ := 2 * (celia_time_per_mile * 1.25 - celia_time_per_mile)
noncomputable def lexie_bee_extra_time : ℝ := 2 * 10
noncomputable def nik_detour_extra_time : ℝ := 0.5 * nik_time_per_mile

-- Total time taken including obstacles
noncomputable def celia_total_time : ℝ := celia_baseline_time + celia_muddy_extra_time
noncomputable def lexie_total_time : ℝ := lexie_baseline_time + lexie_bee_extra_time
noncomputable def nik_total_time : ℝ := nik_baseline_time + nik_detour_extra_time

theorem celia_time_correct : celia_total_time = 305 := by sorry
theorem lexie_time_correct : lexie_total_time = 620 := by sorry
theorem nik_time_correct : nik_total_time = 406.565 := by sorry

end celia_time_correct_lexie_time_correct_nik_time_correct_l588_58822


namespace part1_part2_l588_58827

variable {α : Type*} [LinearOrderedField α]

-- Definitions based on given problem conditions.
def arithmetic_seq(a_n : ℕ → α) := ∃ a1 d, ∀ n, a_n n = a1 + ↑(n - 1) * d

noncomputable def a10_seq := (30 : α)
noncomputable def a20_seq := (50 : α)

-- Theorem statements to prove:
theorem part1 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq) :
  ∀ n, a_n n = 2 * ↑n + 10 := sorry

theorem part2 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq)
  (Sn : α) (hSn : Sn = 242) :
  ∃ n, Sn = (↑n / 2) * (2 * 12 + (↑n - 1) * 2) ∧ n = 11 := sorry

end part1_part2_l588_58827


namespace sum_values_of_cubes_eq_l588_58887

theorem sum_values_of_cubes_eq :
  ∀ (a b : ℝ), a^3 + b^3 + 3 * a * b = 1 → a + b = 1 ∨ a + b = -2 :=
by
  intros a b h
  sorry

end sum_values_of_cubes_eq_l588_58887


namespace arithmetic_series_first_term_l588_58829

theorem arithmetic_series_first_term :
  ∃ a d : ℚ, 
    (30 * (2 * a + 59 * d) = 240) ∧
    (30 * (2 * a + 179 * d) = 3240) ∧
    a = - (247 / 12) :=
by
  sorry

end arithmetic_series_first_term_l588_58829


namespace rebecca_bought_2_more_bottles_of_water_l588_58883

noncomputable def number_of_more_bottles_of_water_than_tent_stakes
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : Prop :=
  W - T = 2

theorem rebecca_bought_2_more_bottles_of_water
  (T D W : ℕ) 
  (hT : T = 4) 
  (hD : D = 3 * T) 
  (hTotal : T + D + W = 22) : 
  number_of_more_bottles_of_water_than_tent_stakes T D W hT hD hTotal :=
by 
  sorry

end rebecca_bought_2_more_bottles_of_water_l588_58883


namespace total_games_l588_58821

/-- Definition of the number of games Alyssa went to this year -/
def games_this_year : Nat := 11

/-- Definition of the number of games Alyssa went to last year -/
def games_last_year : Nat := 13

/-- Definition of the number of games Alyssa plans to go to next year -/
def games_next_year : Nat := 15

/-- Statement to prove the total number of games Alyssa will go to in all -/
theorem total_games : games_this_year + games_last_year + games_next_year = 39 := by
  -- A sorry placeholder to skip the proof
  sorry

end total_games_l588_58821


namespace quadratic_equation_single_solution_l588_58849

theorem quadratic_equation_single_solution (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 = 0) ∧ (∀ x1 x2 : ℝ, a * x1^2 + a * x1 + 1 = 0 → a * x2^2 + a * x2 + 1 = 0 → x1 = x2) → a = 4 :=
by sorry

end quadratic_equation_single_solution_l588_58849


namespace similar_triangle_perimeter_l588_58875

theorem similar_triangle_perimeter
  (a b c : ℕ)
  (h1 : a = 7)
  (h2 : b = 7)
  (h3 : c = 12)
  (similar_triangle_longest_side : ℕ)
  (h4 : similar_triangle_longest_side = 36)
  (h5 : c * similar_triangle_longest_side = 12 * 36) :
  ∃ P : ℕ, P = 78 := by
  sorry

end similar_triangle_perimeter_l588_58875


namespace allan_balloons_count_l588_58848

-- Definition of the conditions
def Total_balloons : ℕ := 3
def Jake_balloons : ℕ := 1

-- The theorem that corresponds to the problem statement
theorem allan_balloons_count (Allan_balloons : ℕ) (h : Allan_balloons + Jake_balloons = Total_balloons) : Allan_balloons = 2 := 
by
  sorry

end allan_balloons_count_l588_58848


namespace average_score_all_students_l588_58846

theorem average_score_all_students 
  (n1 n2 : Nat) 
  (avg1 avg2 : Nat) 
  (h1 : n1 = 20) 
  (h2 : avg1 = 80) 
  (h3 : n2 = 30) 
  (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 := 
by
  sorry

end average_score_all_students_l588_58846


namespace triangle_equilateral_l588_58889

-- Assume we are given side lengths a, b, and c of a triangle and angles A, B, and C in radians.
variables {a b c : ℝ} {A B C : ℝ}

-- We'll use the assumption that (a + b + c) * (b + c - a) = 3 * b * c and sin A = 2 * sin B * cos C.
axiom triangle_condition1 : (a + b + c) * (b + c - a) = 3 * b * c
axiom triangle_condition2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- We need to prove that the triangle is equilateral.
theorem triangle_equilateral : (a = b) ∧ (b = c) ∧ (c = a) := by
  sorry

end triangle_equilateral_l588_58889


namespace function_increment_l588_58815

theorem function_increment (x₁ x₂ : ℝ) (f : ℝ → ℝ) (h₁ : x₁ = 2) 
                           (h₂ : x₂ = 2.5) (h₃ : ∀ x, f x = x ^ 2) :
  f x₂ - f x₁ = 2.25 :=
by
  sorry

end function_increment_l588_58815


namespace over_limit_weight_l588_58872

variable (hc_books : ℕ) (hc_weight : ℕ → ℝ)
variable (tex_books : ℕ) (tex_weight : ℕ → ℝ)
variable (knick_knacks : ℕ) (knick_weight : ℕ → ℝ)
variable (weight_limit : ℝ)

axiom hc_books_value : hc_books = 70
axiom hc_weight_value : hc_weight hc_books = 0.5
axiom tex_books_value : tex_books = 30
axiom tex_weight_value : tex_weight tex_books = 2
axiom knick_knacks_value : knick_knacks = 3
axiom knick_weight_value : knick_weight knick_knacks = 6
axiom weight_limit_value : weight_limit = 80

theorem over_limit_weight : 
  (hc_books * hc_weight hc_books + tex_books * tex_weight tex_books + knick_knacks * knick_weight knick_knacks) - weight_limit = 33 := by
  sorry

end over_limit_weight_l588_58872


namespace probability_non_black_ball_l588_58884

/--
Given the odds of drawing a black ball as 5:3,
prove that the probability of drawing a non-black ball from the bag is 3/8.
-/
theorem probability_non_black_ball (n_black n_non_black : ℕ) (h : n_black = 5) (h' : n_non_black = 3) :
  (n_non_black : ℚ) / (n_black + n_non_black) = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_non_black_ball_l588_58884


namespace geometric_sequence_fourth_term_l588_58876

theorem geometric_sequence_fourth_term (x : ℚ) (r : ℚ)
  (h1 : x ≠ 0)
  (h2 : x ≠ -1)
  (h3 : 3 * x + 3 = r * x)
  (h4 : 5 * x + 5 = r * (3 * x + 3)) :
  r^3 * (5 * x + 5) = -125 / 12 :=
by
  sorry

end geometric_sequence_fourth_term_l588_58876


namespace necessary_and_sufficient_condition_l588_58880

open Classical

noncomputable def f (x a : ℝ) := x + a / x

theorem necessary_and_sufficient_condition
  (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ 2) ↔ (a ≥ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l588_58880


namespace inequality_proof_l588_58830

theorem inequality_proof (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end inequality_proof_l588_58830


namespace zoo_children_tuesday_l588_58890

theorem zoo_children_tuesday 
  (x : ℕ) 
  (child_ticket_cost adult_ticket_cost : ℕ) 
  (children_monday adults_monday adults_tuesday : ℕ)
  (total_revenue : ℕ) : 
  child_ticket_cost = 3 → 
  adult_ticket_cost = 4 → 
  children_monday = 7 → 
  adults_monday = 5 → 
  adults_tuesday = 2 → 
  total_revenue = 61 → 
  7 * 3 + 5 * 4 + x * 3 + 2 * 4 = total_revenue → 
  x = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zoo_children_tuesday_l588_58890


namespace manuscript_fee_l588_58824

noncomputable def tax (x : ℝ) : ℝ :=
  if x ≤ 800 then 0
  else if x <= 4000 then 0.14 * (x - 800)
  else 0.11 * x

theorem manuscript_fee (x : ℝ) (h₁ : tax x = 420)
  (h₂ : 800 < x ∧ x ≤ 4000 ∨ x > 4000) :
  x = 3800 :=
sorry

end manuscript_fee_l588_58824


namespace girl_walked_distance_l588_58826

-- Define the conditions
def speed : ℝ := 5 -- speed in kmph
def time : ℝ := 6 -- time in hours

-- Define the distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- The proof statement that we need to show
theorem girl_walked_distance :
  distance speed time = 30 := by
  sorry

end girl_walked_distance_l588_58826


namespace min_largest_value_in_set_l588_58894

theorem min_largest_value_in_set (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : (8:ℚ) / 19 * a * b ≤ (a - 1) * a / 2): a ≥ 13 :=
by
  sorry

end min_largest_value_in_set_l588_58894


namespace triangle_area_l588_58817

/-- Define the area of a triangle with one side of length 13, an opposite angle of 60 degrees, and side ratio 4:3. -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h_a : a = 13)
  (h_A : A = Real.pi / 3)
  (h_bc_ratio : b / c = 4 / 3)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h_area : S = 1 / 2 * b * c * Real.sin A) :
  S = 39 * Real.sqrt 3 :=
by
  sorry

end triangle_area_l588_58817


namespace square_no_remainder_5_mod_9_l588_58877

theorem square_no_remainder_5_mod_9 (n : ℤ) : (n^2 % 9 ≠ 5) :=
by sorry

end square_no_remainder_5_mod_9_l588_58877


namespace total_time_outside_class_l588_58834

-- Definitions based on given conditions
def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

-- Proof problem statement
theorem total_time_outside_class : first_recess + second_recess + lunch + third_recess = 80 := 
by sorry

end total_time_outside_class_l588_58834


namespace find_value_l588_58886

def equation := ∃ x : ℝ, x^2 - 2 * x - 3 = 0
def expression (x : ℝ) := 2 * x^2 - 4 * x + 12

theorem find_value :
  (∃ x : ℝ, (x^2 - 2 * x - 3 = 0) ∧ (expression x = 18)) :=
by
  sorry

end find_value_l588_58886


namespace cyclist_pedestrian_meeting_distance_l588_58873

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l588_58873


namespace find_k_l588_58869

theorem find_k (k : ℝ) (x₁ x₂ : ℝ)
  (h : x₁^2 + (2 * k - 1) * x₁ + k^2 - 1 = 0)
  (h' : x₂^2 + (2 * k - 1) * x₂ + k^2 - 1 = 0)
  (hx : x₁ ≠ x₂)
  (cond : x₁^2 + x₂^2 = 19) : k = -2 :=
sorry

end find_k_l588_58869


namespace volume_in_cubic_yards_l588_58878

-- Adding the conditions as definitions
def feet_to_yards : ℝ := 3 -- 3 feet in a yard
def cubic_feet_to_cubic_yards : ℝ := feet_to_yards^3 -- convert to cubic yards
def volume_in_cubic_feet : ℝ := 108 -- volume in cubic feet

-- The theorem to prove the equivalence
theorem volume_in_cubic_yards
  (h1 : feet_to_yards = 3)
  (h2 : volume_in_cubic_feet = 108)
  : (volume_in_cubic_feet / cubic_feet_to_cubic_yards) = 4 := 
sorry

end volume_in_cubic_yards_l588_58878


namespace determine_a2016_l588_58818

noncomputable def a_n (n : ℕ) : ℤ := sorry
noncomputable def S_n (n : ℕ) : ℤ := sorry

axiom S1 : S_n 1 = 6
axiom S2 : S_n 2 = 4
axiom S_pos (n : ℕ) : S_n n > 0
axiom geom_progression (n : ℕ) : (S_n (2 * n - 1))^2 = S_n (2 * n) * S_n (2 * n + 2)
axiom arith_progression (n : ℕ) : 2 * S_n (2 * n + 2) = S_n (2 * n - 1) + S_n (2 * n + 1)

theorem determine_a2016 : a_n 2016 = -1009 :=
by sorry

end determine_a2016_l588_58818


namespace correct_operation_l588_58862

theorem correct_operation (a m : ℝ) :
  ¬(a^5 / a^10 = a^2) ∧ 
  (-2 * a^3)^2 = 4 * a^6 ∧ 
  ¬((1 / (2 * m)) - (1 / m) = (1 / m)) ∧ 
  ¬(a^4 + a^3 = a^7) :=
by
  sorry

end correct_operation_l588_58862


namespace value_f2_f5_l588_58881

variable {α : Type} [AddGroup α]

noncomputable def f : α → ℤ := sorry

axiom func_eq : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4

axiom f_one : f 1 = 4

theorem value_f2_f5 :
  f 2 + f 5 = 125 :=
sorry

end value_f2_f5_l588_58881


namespace problem_statement_l588_58871

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 2)

def S : Set ℝ := {y | ∃ x ≥ 0, y = f x}

theorem problem_statement :
  (∀ y ∈ S, y ≤ 2) ∧ (¬ (2 ∈ S)) ∧ (∀ y ∈ S, y ≥ 3 / 2) ∧ (3 / 2 ∈ S) :=
by
  sorry

end problem_statement_l588_58871


namespace probability_sum_leq_12_l588_58892

theorem probability_sum_leq_12 (dice1 dice2 : ℕ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 8) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 8) :
  (∃ outcomes : ℕ, (outcomes = 64) ∧ 
   (∃ favorable : ℕ, (favorable = 54) ∧ 
   (favorable / outcomes = 27 / 32))) :=
sorry

end probability_sum_leq_12_l588_58892


namespace badminton_members_count_l588_58825

-- Definitions of the conditions
def total_members : ℕ := 40
def tennis_players : ℕ := 18
def neither_sport : ℕ := 5
def both_sports : ℕ := 3
def badminton_players : ℕ := 20 -- The answer we need to prove

-- The proof statement
theorem badminton_members_count :
  total_members = (badminton_players + tennis_players - both_sports) + neither_sport :=
by
  -- The proof is outlined here
  sorry

end badminton_members_count_l588_58825


namespace domain_of_f_l588_58857

noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_f_l588_58857


namespace number_of_triples_l588_58888

theorem number_of_triples : 
  ∃ n : ℕ, 
  n = 2 ∧
  ∀ (a b c : ℕ), 
    (2 ≤ a ∧ a ≤ b ∧ b ≤ c) →
    (a * b * c = 4 * (a * b + b * c + c * a)) →
    n = 2 :=
sorry

end number_of_triples_l588_58888


namespace find_cows_l588_58855

-- Define the number of ducks (D) and cows (C)
variables (D C : ℕ)

-- Define the main condition given in the problem
def legs_eq_condition (D C : ℕ) : Prop :=
  2 * D + 4 * C = 2 * (D + C) + 36

-- State the theorem we wish to prove
theorem find_cows (D C : ℕ) (h : legs_eq_condition D C) : C = 18 :=
sorry

end find_cows_l588_58855


namespace base_k_to_decimal_l588_58843

theorem base_k_to_decimal (k : ℕ) (h : 0 < k ∧ k < 10) : 
  1 * k^2 + 7 * k + 5 = 125 → k = 8 := 
by
  sorry

end base_k_to_decimal_l588_58843


namespace solution_interval_l588_58867

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x + x - 2

theorem solution_interval :
  ∃ x, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end solution_interval_l588_58867


namespace cos_diff_identity_l588_58820

variable {α : ℝ}

def sin_alpha := -3 / 5

def alpha_interval (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi)

theorem cos_diff_identity (h1 : Real.sin α = sin_alpha) (h2 : alpha_interval α) :
  Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 10 :=
  sorry

end cos_diff_identity_l588_58820


namespace certain_value_of_101n_squared_l588_58836

theorem certain_value_of_101n_squared 
  (n : ℤ) 
  (h : ∀ (n : ℤ), 101 * n^2 ≤ 4979 → n ≤ 7) : 
  4979 = 101 * 7^2 :=
by {
  /- proof goes here -/
  sorry
}

end certain_value_of_101n_squared_l588_58836


namespace nathaniel_tickets_l588_58807

theorem nathaniel_tickets :
  ∀ (B S : ℕ),
  (7 * B + 4 * S + 11 = 128) →
  (B + S = 20) :=
by
  intros B S h
  sorry

end nathaniel_tickets_l588_58807


namespace jose_is_12_years_older_l588_58838

theorem jose_is_12_years_older (J M : ℕ) (h1 : M = 14) (h2 : J + M = 40) : J - M = 12 :=
by
  sorry

end jose_is_12_years_older_l588_58838


namespace coplanar_vectors_m_value_l588_58812

variable (m : ℝ)
variable (α β : ℝ)
def a := (5, 9, m)
def b := (1, -1, 2)
def c := (2, 5, 1)

theorem coplanar_vectors_m_value :
  ∃ (α β : ℝ), (5 = α + 2 * β) ∧ (9 = -α + 5 * β) ∧ (m = 2 * α + β) → m = 4 :=
by
  sorry

end coplanar_vectors_m_value_l588_58812


namespace max_intersections_arith_geo_seq_l588_58842

def arithmetic_sequence (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

def geometric_sequence (n : ℕ) (q : ℝ) : ℝ := q ^ (n - 1)

theorem max_intersections_arith_geo_seq (d : ℝ) (q : ℝ) (h_d : d ≠ 0) (h_q_pos : q > 0) (h_q_neq1 : q ≠ 1) :
  (∃ n : ℕ, arithmetic_sequence n d = geometric_sequence n q) → ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ (arithmetic_sequence n₁ d = geometric_sequence n₁ q) ∧ (arithmetic_sequence n₂ d = geometric_sequence n₂ q) :=
sorry

end max_intersections_arith_geo_seq_l588_58842


namespace notebook_price_l588_58879

theorem notebook_price (x : ℝ) 
  (h1 : 3 * x + 1.50 + 1.70 = 6.80) : 
  x = 1.20 :=
by 
  sorry

end notebook_price_l588_58879


namespace swans_after_10_years_l588_58806

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l588_58806
