import Mathlib

namespace fraction_of_time_at_15_mph_l2050_205051

theorem fraction_of_time_at_15_mph
  (t1 t2 : ℝ)
  (h : (5 * t1 + 15 * t2) / (t1 + t2) = 10) :
  t2 / (t1 + t2) = 1 / 2 :=
by
  sorry

end fraction_of_time_at_15_mph_l2050_205051


namespace equation_of_line_l2050_205005

theorem equation_of_line (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (-1, 2)) (hA : A.2 = 0) (hB : B.1 = 0) (hMid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = 4) ∧ ∀ (x y : ℝ), y = a * x + b * y + c → 2 * x - y + 4 = 0 := 
  sorry

end equation_of_line_l2050_205005


namespace sales_worth_l2050_205086

variable (S : ℝ)
def old_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_remuneration S = old_remuneration S + 600 → S = 24000 :=
by
  intro h
  sorry

end sales_worth_l2050_205086


namespace no_consecutive_integers_square_difference_2000_l2050_205047

theorem no_consecutive_integers_square_difference_2000 :
  ¬ ∃ a : ℤ, (a + 1) ^ 2 - a ^ 2 = 2000 :=
by {
  -- some detailed steps might go here in a full proof
  sorry
}

end no_consecutive_integers_square_difference_2000_l2050_205047


namespace find_a_l2050_205098

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l2050_205098


namespace number_of_students_who_bought_2_pencils_l2050_205016

variable (a b c : ℕ)     -- a is the number of students buying 1 pencil, b is the number of students buying 2 pencils, c is the number of students buying 3 pencils.
variable (total_students total_pencils : ℕ) -- total_students is 36, total_pencils is 50
variable (students_condition1 students_condition2 : ℕ) -- conditions: students_condition1 for the sum of the students, students_condition2 for the sum of the pencils

theorem number_of_students_who_bought_2_pencils :
  total_students = 36 ∧
  total_pencils = 50 ∧
  total_students = a + b + c ∧
  total_pencils = a * 1 + b * 2 + c * 3 ∧
  a = 2 * (b + c) → 
  b = 10 :=
by sorry

end number_of_students_who_bought_2_pencils_l2050_205016


namespace correct_statements_l2050_205046

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem correct_statements :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (f (Real.log 3 / Real.log 2) ≠ 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (|x|) ≥ 0 ∧ f 0 = 0) :=
by
  sorry

end correct_statements_l2050_205046


namespace y_coordinate_of_C_l2050_205037

def Point : Type := (ℤ × ℤ)

def A : Point := (0, 0)
def B : Point := (0, 4)
def D : Point := (4, 4)
def E : Point := (4, 0)

def PentagonArea (C : Point) : ℚ :=
  let triangleArea : ℚ := (1/2 : ℚ) * 4 * ((C.2 : ℚ) - 4)
  let squareArea : ℚ := 4 * 4
  triangleArea + squareArea

theorem y_coordinate_of_C (h : ℤ) (C : Point := (2, h)) : PentagonArea C = 40 → C.2 = 16 :=
by
  sorry

end y_coordinate_of_C_l2050_205037


namespace circles_internally_tangent_l2050_205022

theorem circles_internally_tangent (R r : ℝ) (h1 : R + r = 5) (h2 : R * r = 6) (d : ℝ) (h3 : d = 1) : d = |R - r| :=
by
  -- This allows the logic of the solution to be captured as the theorem we need to prove
  sorry

end circles_internally_tangent_l2050_205022


namespace find_range_of_m_l2050_205028

-- Statements of the conditions given in the problem
axiom positive_real_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (1 / x + 4 / y = 1)

-- Main statement of the proof problem
theorem find_range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 / x + 4 / y = 1) :
  (∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (1 / x + 4 / y = 1) ∧ (x + y / 4 < m^2 - 3 * m)) ↔ (m < -1 ∨ m > 4) := 
sorry

end find_range_of_m_l2050_205028


namespace sum_of_cubes_l2050_205011

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 :=
sorry

end sum_of_cubes_l2050_205011


namespace cos_alpha_in_second_quadrant_l2050_205091

theorem cos_alpha_in_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_tan : Real.tan α = -1 / 2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l2050_205091


namespace pipe_A_filling_time_l2050_205035

theorem pipe_A_filling_time :
  ∃ (t : ℚ), 
  (∀ (t : ℚ), (t > 0) → (1 / t + 5 / t = 1 / 4.571428571428571) ↔ t = 27.42857142857143) := 
by
  -- definition of t and the corresponding conditions are directly derived from the problem
  sorry

end pipe_A_filling_time_l2050_205035


namespace inequality_min_m_l2050_205048

theorem inequality_min_m (m : ℝ) (x : ℝ) (hx : 1 < x) : 
  x + m * Real.log x + 1 / Real.exp x ≥ Real.exp (m * Real.log x) :=
sorry

end inequality_min_m_l2050_205048


namespace base6_addition_sum_l2050_205001

theorem base6_addition_sum 
  (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : Q ≠ R) 
  (h3 : P ≠ R) 
  (h4 : P < 6) 
  (h5 : Q < 6) 
  (h6 : R < 6) 
  (h7 : 2*R % 6 = P) 
  (h8 : 2*Q % 6 = R)
  : P + Q + R = 7 := 
  sorry

end base6_addition_sum_l2050_205001


namespace simplify_expression_l2050_205076

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : (3 * b - 3 - 5 * b) / 3 = - (2 / 3) * b - 1 :=
by
  sorry

end simplify_expression_l2050_205076


namespace bob_sheep_and_ratio_l2050_205010

-- Define the initial conditions
def mary_initial_sheep : ℕ := 300
def additional_sheep_bob_has : ℕ := 35
def sheep_mary_buys : ℕ := 266
def fewer_sheep_than_bob : ℕ := 69

-- Define the number of sheep Bob has
def bob_sheep (mary_initial_sheep : ℕ) (additional_sheep_bob_has : ℕ) : ℕ := 
  mary_initial_sheep + additional_sheep_bob_has

-- Define the number of sheep Mary has after buying more sheep
def mary_new_sheep (mary_initial_sheep : ℕ) (sheep_mary_buys : ℕ) : ℕ := 
  mary_initial_sheep + sheep_mary_buys

-- Define the relation between Mary's and Bob's sheep (after Mary buys sheep)
def mary_bob_relation (mary_new_sheep : ℕ) (fewer_sheep_than_bob : ℕ) : Prop :=
  mary_new_sheep + fewer_sheep_than_bob = bob_sheep mary_initial_sheep additional_sheep_bob_has

-- Define the proof problem
theorem bob_sheep_and_ratio : 
  bob_sheep mary_initial_sheep additional_sheep_bob_has = 635 ∧ 
  (bob_sheep mary_initial_sheep additional_sheep_bob_has) * 300 = 635 * mary_initial_sheep := 
by 
  sorry

end bob_sheep_and_ratio_l2050_205010


namespace avg_eq_pos_diff_l2050_205008

theorem avg_eq_pos_diff (y : ℝ) (h : (35 + y) / 2 = 42) : |35 - y| = 14 := 
sorry

end avg_eq_pos_diff_l2050_205008


namespace find_c_l2050_205036

-- Define the functions p and q as given in the conditions
def p (x : ℝ) : ℝ := 3 * x - 9
def q (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

-- State the main theorem with conditions and goal
theorem find_c (c : ℝ) (h : p (q 3 c) = 15) : c = 4 := by
  sorry -- Proof is not required

end find_c_l2050_205036


namespace apples_difference_l2050_205099

theorem apples_difference 
  (father_apples : ℕ := 8)
  (mother_apples : ℕ := 13)
  (jungkook_apples : ℕ := 7)
  (brother_apples : ℕ := 5) :
  max father_apples (max mother_apples (max jungkook_apples brother_apples)) - 
  min father_apples (min mother_apples (min jungkook_apples brother_apples)) = 8 :=
by
  sorry

end apples_difference_l2050_205099


namespace gcd_2750_9450_l2050_205024

theorem gcd_2750_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end gcd_2750_9450_l2050_205024


namespace triangle_area_l2050_205041

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end triangle_area_l2050_205041


namespace ABCD_eq_neg1_l2050_205085

noncomputable def A := (Real.sqrt 2013 + Real.sqrt 2012)
noncomputable def B := (- Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def C := (Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def D := (Real.sqrt 2012 - Real.sqrt 2013)

theorem ABCD_eq_neg1 : A * B * C * D = -1 :=
by sorry

end ABCD_eq_neg1_l2050_205085


namespace number_of_cookies_l2050_205065

def total_cake := 22
def total_chocolate := 16
def total_groceries := 42

theorem number_of_cookies :
  ∃ C : ℕ, total_groceries = total_cake + total_chocolate + C ∧ C = 4 := 
by
  sorry

end number_of_cookies_l2050_205065


namespace percentage_green_shirts_correct_l2050_205007

variable (total_students blue_percentage red_percentage other_students : ℕ)

noncomputable def percentage_green_shirts (total_students blue_percentage red_percentage other_students : ℕ) : ℕ :=
  let total_blue_shirts := blue_percentage * total_students / 100
  let total_red_shirts := red_percentage * total_students / 100
  let total_blue_red_other_shirts := total_blue_shirts + total_red_shirts + other_students
  let green_shirts := total_students - total_blue_red_other_shirts
  (green_shirts * 100) / total_students

theorem percentage_green_shirts_correct
  (h1 : total_students = 800) 
  (h2 : blue_percentage = 45)
  (h3 : red_percentage = 23)
  (h4 : other_students = 136) : 
  percentage_green_shirts total_students blue_percentage red_percentage other_students = 15 :=
by
  sorry

end percentage_green_shirts_correct_l2050_205007


namespace percentage_calculation_l2050_205031

theorem percentage_calculation : 
  (0.8 * 90) = ((P / 100) * 60.00000000000001 + 30) → P = 70 := by
  sorry

end percentage_calculation_l2050_205031


namespace problem_1_problem_2_l2050_205027

noncomputable def f (x : ℝ) : ℝ := (1 / (9 * (Real.sin x)^2)) + (4 / (9 * (Real.cos x)^2))

theorem problem_1 (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : f x ≥ 1 := 
sorry

theorem problem_2 (x : ℝ) : x^2 + |x-2| + 1 ≥ 3 ↔ (x ≤ 0 ∨ x ≥ 1) :=
sorry

end problem_1_problem_2_l2050_205027


namespace f_at_neg_one_l2050_205062

def f (x : ℝ) : ℝ := sorry

theorem f_at_neg_one (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 :=
by sorry

end f_at_neg_one_l2050_205062


namespace length_to_width_ratio_is_three_l2050_205002

def rectangle_ratio (x : ℝ) : Prop :=
  let side_length_large_square := 4 * x
  let length_rectangle := 4 * x
  let width_rectangle := x
  length_rectangle / width_rectangle = 3

-- We state the theorem to be proved
theorem length_to_width_ratio_is_three (x : ℝ) (h : 0 < x) :
  rectangle_ratio x :=
sorry

end length_to_width_ratio_is_three_l2050_205002


namespace minimum_value_of_sum_l2050_205095

variable (x y : ℝ)

theorem minimum_value_of_sum (hx : x > 0) (hy : y > 0) : ∃ x y, x > 0 ∧ y > 0 ∧ (x + 2 * y) = 9 :=
sorry

end minimum_value_of_sum_l2050_205095


namespace sin_sub_pi_over_3_eq_neg_one_third_l2050_205015

theorem sin_sub_pi_over_3_eq_neg_one_third {x : ℝ} (h : Real.cos (x + (π / 6)) = 1 / 3) :
  Real.sin (x - (π / 3)) = -1 / 3 := 
  sorry

end sin_sub_pi_over_3_eq_neg_one_third_l2050_205015


namespace min_max_f_l2050_205050

theorem min_max_f (a b x y z t : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hxz : x + z = 1) (hyt : y + t = 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hz : 0 ≤ z) (ht : 0 ≤ t) :
  1 ≤ ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ∧
  ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ≤ 2 :=
sorry

end min_max_f_l2050_205050


namespace choose_two_out_of_three_l2050_205082

-- Define the number of vegetables as n and the number to choose as k
def n : ℕ := 3
def k : ℕ := 2

-- The combination formula C(n, k) == n! / (k! * (n - k)!)
def combination (n k : ℕ) : ℕ := n.choose k

-- Problem statement: Prove that the number of ways to choose 2 out of 3 vegetables is 3
theorem choose_two_out_of_three : combination n k = 3 :=
by
  sorry

end choose_two_out_of_three_l2050_205082


namespace eric_age_l2050_205083

theorem eric_age (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 :=
by
  sorry

end eric_age_l2050_205083


namespace div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l2050_205006

-- Define the values provided in the problem
def div_1 := (8 : ℚ) / (8 / 17 : ℚ)
def div_2 := (6 / 11 : ℚ) / 3
def mul_1 := (5 / 4 : ℚ) * (1 / 5 : ℚ)

-- Prove the equivalences
theorem div_1_eq_17 : div_1 = 17 := by
  sorry

theorem div_2_eq_2_11 : div_2 = 2 / 11 := by
  sorry

theorem mul_1_eq_1_4 : mul_1 = 1 / 4 := by
  sorry

end div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l2050_205006


namespace train_cross_bridge_time_l2050_205038

noncomputable def time_to_cross_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (length_of_bridge : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_mps := speed_kmh * (1000 / 3600)
  total_distance / speed_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 110 72 112 = 11.1 :=
by
  sorry

end train_cross_bridge_time_l2050_205038


namespace mul_scientific_notation_l2050_205030

theorem mul_scientific_notation (a b : ℝ) (c d : ℝ) (h1 : a = 7 * 10⁻¹) (h2 : b = 8 * 10⁻¹) :
  (a * b = 0.56) :=
by
  sorry

end mul_scientific_notation_l2050_205030


namespace find_n_l2050_205059

theorem find_n (n a b : ℕ) 
  (h1 : a > 1)
  (h2 : a ∣ n)
  (h3 : b > a)
  (h4 : b ∣ n)
  (h5 : ∀ m, 1 < m ∧ m < a → ¬ m ∣ n)
  (h6 : ∀ m, a < m ∧ m < b → ¬ m ∣ n)
  (h7 : n = a^a + b^b)
  : n = 260 :=
by sorry

end find_n_l2050_205059


namespace jakes_present_weight_l2050_205069

theorem jakes_present_weight (J S : ℕ) (h1 : J - 32 = 2 * S) (h2 : J + S = 212) : J = 152 :=
by
  sorry

end jakes_present_weight_l2050_205069


namespace exponent_problem_l2050_205000

theorem exponent_problem 
  (a : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 0) 
  (h2 : a^x = 3) 
  (h3 : a^y = 5) : 
  a^(2*x + y/2) = 9 * Real.sqrt 5 :=
by
  sorry

end exponent_problem_l2050_205000


namespace intersection_result_l2050_205097

def A : Set ℝ := {x | |x - 2| ≤ 2}

def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

def intersection : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_result : (A ∩ B) = intersection :=
by
  sorry

end intersection_result_l2050_205097


namespace ratio_of_guests_l2050_205017

def bridgette_guests : Nat := 84
def alex_guests : Nat := sorry -- This will be inferred in the theorem
def extra_plates : Nat := 10
def total_asparagus_spears : Nat := 1200
def asparagus_per_plate : Nat := 8

theorem ratio_of_guests (A : Nat) (h1 : total_asparagus_spears / asparagus_per_plate = 150) (h2 : 150 - extra_plates = 140) (h3 : 140 - bridgette_guests = A) : A / bridgette_guests = 2 / 3 :=
by
  sorry

end ratio_of_guests_l2050_205017


namespace sheelas_total_net_monthly_income_l2050_205094

noncomputable def totalNetMonthlyIncome
    (PrimaryJobIncome : ℝ)
    (FreelanceIncome : ℝ)
    (FreelanceIncomeTaxRate : ℝ)
    (AnnualInterestIncome : ℝ)
    (InterestIncomeTaxRate : ℝ) : ℝ :=
    let PrimaryJobMonthlyIncome := 5000 / 0.20
    let FreelanceIncomeTax := FreelanceIncome * FreelanceIncomeTaxRate
    let NetFreelanceIncome := FreelanceIncome - FreelanceIncomeTax
    let InterestIncomeTax := AnnualInterestIncome * InterestIncomeTaxRate
    let NetAnnualInterestIncome := AnnualInterestIncome - InterestIncomeTax
    let NetMonthlyInterestIncome := NetAnnualInterestIncome / 12
    PrimaryJobMonthlyIncome + NetFreelanceIncome + NetMonthlyInterestIncome

theorem sheelas_total_net_monthly_income :
    totalNetMonthlyIncome 25000 3000 0.10 2400 0.05 = 27890 := 
by
    sorry

end sheelas_total_net_monthly_income_l2050_205094


namespace find_number_l2050_205054

-- Define the condition: a number exceeds by 40 from its 3/8 part.
def exceeds_by_40_from_its_fraction (x : ℝ) := x = (3/8) * x + 40

-- The theorem: prove that the number is 64 given the condition.
theorem find_number (x : ℝ) (h : exceeds_by_40_from_its_fraction x) : x = 64 := 
by
  sorry

end find_number_l2050_205054


namespace state_a_selection_percentage_l2050_205077

-- Definitions based on the conditions
variables {P : ℕ} -- percentage of candidates selected in State A

theorem state_a_selection_percentage 
  (candidates : ℕ) 
  (state_b_percentage : ℕ) 
  (extra_selected_in_b : ℕ) 
  (total_selected_in_b : ℕ) 
  (total_selected_in_a : ℕ)
  (appeared_in_each_state : ℕ) 
  (H1 : appeared_in_each_state = 8200)
  (H2 : state_b_percentage = 7)
  (H3 : extra_selected_in_b = 82)
  (H4 : total_selected_in_b = (state_b_percentage * appeared_in_each_state) / 100)
  (H5 : total_selected_in_a = total_selected_in_b - extra_selected_in_b)
  (H6 : total_selected_in_a = (P * appeared_in_each_state) / 100)
  : P = 6 :=
by {
  sorry
}

end state_a_selection_percentage_l2050_205077


namespace find_k_l2050_205003

def line_p (x y : ℝ) : Prop := y = -2 * x + 3
def line_q (x y k : ℝ) : Prop := y = k * x + 4
def intersection (x y k : ℝ) : Prop := line_p x y ∧ line_q x y k

theorem find_k (k : ℝ) (h_inter : intersection 1 1 k) : k = -3 :=
sorry

end find_k_l2050_205003


namespace Donny_spends_28_on_Thursday_l2050_205064

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l2050_205064


namespace selection_assignment_schemes_l2050_205045

noncomputable def number_of_selection_schemes (males females : ℕ) : ℕ :=
  if h : males + females < 3 then 0
  else
    let total3 := Nat.choose (males + females) 3
    let all_males := if hM : males < 3 then 0 else Nat.choose males 3
    let all_females := if hF : females < 3 then 0 else Nat.choose females 3
    total3 - all_males - all_females

theorem selection_assignment_schemes :
  number_of_selection_schemes 4 3 = 30 :=
by sorry

end selection_assignment_schemes_l2050_205045


namespace algorithm_can_contain_all_structures_l2050_205073

def sequential_structure : Prop := sorry
def conditional_structure : Prop := sorry
def loop_structure : Prop := sorry

def algorithm_contains_structure (str : Prop) : Prop := sorry

theorem algorithm_can_contain_all_structures :
  algorithm_contains_structure sequential_structure ∧
  algorithm_contains_structure conditional_structure ∧
  algorithm_contains_structure loop_structure := sorry

end algorithm_can_contain_all_structures_l2050_205073


namespace back_wheel_revolutions_l2050_205079

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (rev_front : ℝ) (r_front_eq : r_front = 3)
  (r_back_eq : r_back = 0.5) (rev_front_eq : rev_front = 50) :
  let C_front := 2 * Real.pi * r_front
  let D_front := C_front * rev_front
  let C_back := 2 * Real.pi * r_back
  let rev_back := D_front / C_back
  rev_back = 300 := by
  sorry

end back_wheel_revolutions_l2050_205079


namespace r_cube_plus_inv_r_cube_eq_zero_l2050_205004

theorem r_cube_plus_inv_r_cube_eq_zero {r : ℝ} (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := 
sorry

end r_cube_plus_inv_r_cube_eq_zero_l2050_205004


namespace compute_fraction_value_l2050_205089

theorem compute_fraction_value : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end compute_fraction_value_l2050_205089


namespace number_of_subsets_of_set_l2050_205053

theorem number_of_subsets_of_set (x y : ℝ) 
  (z : ℂ) (hz : z = (2 - (1 : ℂ) * Complex.I) / (1 + (2 : ℂ) * Complex.I))
  (hx : z.re = x) (hy : z.im = y) : 
  (Finset.powerset ({x, 2^x, y} : Finset ℝ)).card = 8 :=
by
  sorry

end number_of_subsets_of_set_l2050_205053


namespace specific_heat_capacity_l2050_205074

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end specific_heat_capacity_l2050_205074


namespace num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l2050_205093

def is_prime (n : ℕ) : Prop := Nat.Prime n
def ends_with_7 (n : ℕ) : Prop := n % 10 = 7

theorem num_prime_numbers_with_units_digit_7 (n : ℕ) (h1 : n < 100) (h2 : ends_with_7 n) : is_prime n :=
by sorry

theorem num_prime_numbers_less_than_100_with_units_digit_7 : 
  ∃ (l : List ℕ), (∀ x ∈ l, x < 100 ∧ ends_with_7 x ∧ is_prime x) ∧ l.length = 6 :=
by sorry

end num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l2050_205093


namespace remainder_div_2DD_l2050_205061

theorem remainder_div_2DD' (P D D' Q R Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') :
  P % (2 * D * D') = D * R' + R :=
sorry

end remainder_div_2DD_l2050_205061


namespace total_birds_and_storks_l2050_205070

theorem total_birds_and_storks (initial_birds initial_storks additional_storks : ℕ) 
  (h1 : initial_birds = 3) 
  (h2 : initial_storks = 4) 
  (h3 : additional_storks = 6) 
  : initial_birds + initial_storks + additional_storks = 13 := 
  by sorry

end total_birds_and_storks_l2050_205070


namespace det_of_matrix_l2050_205026

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem det_of_matrix :
  determinant_2x2 5 (-2) 3 1 = 11 := by
  sorry

end det_of_matrix_l2050_205026


namespace hamza_bucket_problem_l2050_205029

-- Definitions reflecting the problem conditions
def bucket_2_5_capacity : ℝ := 2.5
def bucket_3_0_capacity : ℝ := 3.0
def bucket_5_6_capacity : ℝ := 5.6
def bucket_6_5_capacity : ℝ := 6.5

def initial_fill_in_5_6 : ℝ := bucket_5_6_capacity
def pour_5_6_to_3_0_remaining : ℝ := 5.6 - 3.0
def remaining_in_5_6_after_second_fill : ℝ := bucket_5_6_capacity - 0.5

-- Main problem statement
theorem hamza_bucket_problem : (bucket_6_5_capacity - 2.6 = 3.9) :=
by sorry

end hamza_bucket_problem_l2050_205029


namespace smallest_number_is_C_l2050_205033

def A : ℕ := 36
def B : ℕ := 27 + 5
def C : ℕ := 3 * 10
def D : ℕ := 40 - 3

theorem smallest_number_is_C :
  min (min A B) (min C D) = C :=
by
  -- Proof steps go here
  sorry

end smallest_number_is_C_l2050_205033


namespace motorist_gas_problem_l2050_205020

noncomputable def original_price_per_gallon (P : ℝ) : Prop :=
  12 * P = 10 * (P + 0.30)

def fuel_efficiency := 25

def new_distance_travelled (P : ℝ) : ℝ :=
  10 * fuel_efficiency

theorem motorist_gas_problem :
  ∃ P : ℝ, original_price_per_gallon P ∧ P = 1.5 ∧ new_distance_travelled P = 250 :=
by
  use 1.5
  sorry

end motorist_gas_problem_l2050_205020


namespace definite_integral_sin_cos_l2050_205055

open Real

theorem definite_integral_sin_cos :
  ∫ x in - (π / 2)..(π / 2), (sin x + cos x) = 2 :=
sorry

end definite_integral_sin_cos_l2050_205055


namespace polygon_interior_angle_sum_360_l2050_205013

theorem polygon_interior_angle_sum_360 (n : ℕ) (h : (n-2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angle_sum_360_l2050_205013


namespace problem_l2050_205058

noncomputable def f (A B x : ℝ) : ℝ := A * x^2 + B
noncomputable def g (A B x : ℝ) : ℝ := B * x^2 + A

theorem problem (A B x : ℝ) (h : A ≠ B) 
  (h1 : f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := 
  sorry

end problem_l2050_205058


namespace determine_m_l2050_205067

theorem determine_m (a b : ℝ) (m : ℝ) :
  (a^2 + 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2) = (2 - m) * a * b - 3 * b^2 →
  (∀ a b : ℝ, (2 - m) * a * b = 0) →
  m = 2 :=
by
  sorry

end determine_m_l2050_205067


namespace remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l2050_205052

def f (x : ℝ) : ℝ := x^15 + 1

theorem remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0 : f (-1) = 0 := by
  sorry

end remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l2050_205052


namespace susan_bought_36_items_l2050_205072

noncomputable def cost_per_pencil : ℝ := 0.25
noncomputable def cost_per_pen : ℝ := 0.80
noncomputable def pencils_bought : ℕ := 16
noncomputable def total_spent : ℝ := 20.0

theorem susan_bought_36_items :
  ∃ (pens_bought : ℕ), pens_bought * cost_per_pen + pencils_bought * cost_per_pencil = total_spent ∧ pencils_bought + pens_bought = 36 := 
sorry

end susan_bought_36_items_l2050_205072


namespace xy_value_l2050_205068

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l2050_205068


namespace angle_at_7_20_is_100_degrees_l2050_205014

def angle_between_hands_at_7_20 : ℝ := 100

theorem angle_at_7_20_is_100_degrees
    (hour_hand_pos : ℝ := 210) -- 7 * 30 degrees
    (minute_hand_pos : ℝ := 120) -- 4 * 30 degrees
    (hour_hand_move_per_minute : ℝ := 0.5) -- 0.5 degrees per minute
    (time_past_7_clock : ℝ := 20) -- 20 minutes
    (adjacent_angle : ℝ := 30) -- angle between adjacent numbers
    : angle_between_hands_at_7_20 = 
      (hour_hand_pos - (minute_hand_pos - hour_hand_move_per_minute * time_past_7_clock)) :=
sorry

end angle_at_7_20_is_100_degrees_l2050_205014


namespace repeated_pair_exists_l2050_205081

theorem repeated_pair_exists (a : Fin 99 → Fin 10)
  (h1 : ∀ n : Fin 98, a n = 1 → a (n + 1) ≠ 2)
  (h2 : ∀ n : Fin 98, a n = 3 → a (n + 1) ≠ 4) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) :=
sorry

end repeated_pair_exists_l2050_205081


namespace problem_m_n_l2050_205039

theorem problem_m_n (m n : ℝ) (h1 : m * n = 1) (h2 : m^2 + n^2 = 3) (h3 : m^3 + n^3 = 44 + n^4) (h4 : m^5 + 5 = 11) : m^9 + n = -29 :=
sorry

end problem_m_n_l2050_205039


namespace findInitialVolume_l2050_205056

def initialVolume (V : ℝ) : Prop :=
  let newVolume := V + 18
  let initialSugar := 0.27 * V
  let addedSugar := 3.2
  let totalSugar := initialSugar + addedSugar
  let finalSugarPercentage := 0.26536312849162012
  finalSugarPercentage * newVolume = totalSugar 

theorem findInitialVolume : ∃ (V : ℝ), initialVolume V ∧ V = 340 := by
  use 340
  unfold initialVolume
  sorry

end findInitialVolume_l2050_205056


namespace first_customer_bought_5_l2050_205021

variables 
  (x : ℕ) -- Number of boxes the first customer bought
  (x2 : ℕ) -- Number of boxes the second customer bought
  (x3 : ℕ) -- Number of boxes the third customer bought
  (x4 : ℕ) -- Number of boxes the fourth customer bought
  (x5 : ℕ) -- Number of boxes the fifth customer bought

def goal : ℕ := 150
def remaining_boxes : ℕ := 75
def sold_boxes := x + x2 + x3 + x4 + x5

axiom second_customer (hx2 : x2 = 4 * x) : True
axiom third_customer (hx3 : x3 = (x2 / 2)) : True
axiom fourth_customer (hx4 : x4 = 3 * x3) : True
axiom fifth_customer (hx5 : x5 = 10) : True
axiom sales_goal (hgoal : sold_boxes = goal - remaining_boxes) : True

theorem first_customer_bought_5 (hx2 : x2 = 4 * x) 
                                (hx3 : x3 = (x2 / 2)) 
                                (hx4 : x4 = 3 * x3) 
                                (hx5 : x5 = 10) 
                                (hgoal : sold_boxes = goal - remaining_boxes) : 
                                x = 5 :=
by
  -- Here, we would perform the proof steps
  sorry

end first_customer_bought_5_l2050_205021


namespace product_sum_of_roots_l2050_205042

theorem product_sum_of_roots (p q r : ℂ)
  (h_eq : ∀ x : ℂ, (2 : ℂ) * x^3 + (1 : ℂ) * x^2 + (-7 : ℂ) * x + (2 : ℂ) = 0 → (x = p ∨ x = q ∨ x = r)) 
  : p * q + q * r + r * p = -7 / 2 := 
sorry

end product_sum_of_roots_l2050_205042


namespace percent_preferred_apples_l2050_205084

def frequencies : List ℕ := [75, 80, 45, 100, 50]
def frequency_apples : ℕ := 75
def total_frequency : ℕ := frequency_apples + frequencies[1] + frequencies[2] + frequencies[3] + frequencies[4]

theorem percent_preferred_apples :
  (frequency_apples * 100) / total_frequency = 21 := by
  -- Proof steps go here
  sorry

end percent_preferred_apples_l2050_205084


namespace proof_l2050_205096

variable {S : Type} 
variable (op : S → S → S)

-- Condition given in the problem
def condition (a b : S) : Prop :=
  op (op a b) a = b

-- Statement to be proven
theorem proof (h : ∀ a b : S, condition op a b) :
  ∀ a b : S, op a (op b a) = b :=
by
  intros a b
  sorry

end proof_l2050_205096


namespace lower_bound_of_expression_l2050_205044

theorem lower_bound_of_expression :
  ∃ L : ℤ, (∀ n : ℤ, ((-1 ≤ n ∧ n ≤ 8) → (L < 4 * n + 7 ∧ 4 * n + 7 < 40))) ∧ L = 1 :=
by {
  sorry
}

end lower_bound_of_expression_l2050_205044


namespace inequality_proof_l2050_205092

theorem inequality_proof 
(x1 x2 y1 y2 z1 z2 : ℝ) 
(hx1 : x1 > 0) 
(hx2 : x2 > 0) 
(hineq1 : x1 * y1 - z1^2 > 0) 
(hineq2 : x2 * y2 - z2^2 > 0)
: 
  8 / ((x1 + x2)*(y1 + y2) - (z1 + z2)^2) <= 
  1 / (x1 * y1 - z1^2) + 
  1 / (x2 * y2 - z2^2) := 
sorry

end inequality_proof_l2050_205092


namespace tangent_line_of_ellipse_l2050_205040

theorem tangent_line_of_ellipse 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (x₀ y₀ : ℝ) (hx₀ : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, x₀ * x / a^2 + y₀ * y / b^2 = 1 := 
sorry

end tangent_line_of_ellipse_l2050_205040


namespace picnic_men_count_l2050_205018

variables 
  (M W A C : ℕ)
  (h1 : M + W + C = 200) 
  (h2 : M = W + 20)
  (h3 : A = C + 20)
  (h4 : A = M + W)

theorem picnic_men_count : M = 65 :=
by
  sorry

end picnic_men_count_l2050_205018


namespace flag_arrangement_modulo_1000_l2050_205063

theorem flag_arrangement_modulo_1000 :
  let red_flags := 8
  let white_flags := 8
  let black_flags := 1
  let total_flags := red_flags + white_flags + black_flags
  let number_of_gaps := total_flags + 1
  let valid_arrangements := (Nat.choose number_of_gaps white_flags) * (number_of_gaps - 2)
  valid_arrangements % 1000 = 315 :=
by
  sorry

end flag_arrangement_modulo_1000_l2050_205063


namespace joe_lifting_problem_l2050_205025

theorem joe_lifting_problem (x y : ℝ) (h1 : x + y = 900) (h2 : 2 * x = y + 300) : x = 400 :=
sorry

end joe_lifting_problem_l2050_205025


namespace hexagons_formed_square_z_l2050_205066

theorem hexagons_formed_square_z (a b s z : ℕ) (hexagons_congruent : a = 9 ∧ b = 16 ∧ s = 12 ∧ z = 4): 
(z = 4) := by
  sorry

end hexagons_formed_square_z_l2050_205066


namespace range_of_a_l2050_205049

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ↔ (3 / 2 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l2050_205049


namespace complete_squares_l2050_205019

def valid_solutions (x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = -2 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = 6) ∨
  (x = 0 ∧ y = -2 ∧ z = 6) ∨
  (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 4 ∧ y = -2 ∧ z = 0) ∨
  (x = 4 ∧ y = 0 ∧ z = 6) ∨
  (x = 4 ∧ y = -2 ∧ z = 6)

theorem complete_squares (x y z : ℝ) : 
  (x - 2)^2 + (y + 1)^2 = 5 →
  (x - 2)^2 + (z - 3)^2 = 13 →
  (y + 1)^2 + (z - 3)^2 = 10 →
  valid_solutions x y z :=
by
  intros h1 h2 h3
  sorry

end complete_squares_l2050_205019


namespace regular_polygon_sides_l2050_205043

theorem regular_polygon_sides (n : ℕ) (h : 2 < n)
  (interior_angle : ∀ n, (n - 2) * 180 / n = 144) : n = 10 :=
sorry

end regular_polygon_sides_l2050_205043


namespace sandwiches_cost_l2050_205090

theorem sandwiches_cost (sandwiches sodas : ℝ) 
  (cost_sandwich : ℝ := 2.44)
  (cost_soda : ℝ := 0.87)
  (num_sodas : ℕ := 4)
  (total_cost : ℝ := 8.36)
  (total_soda_cost : ℝ := cost_soda * num_sodas)
  (total_sandwich_cost : ℝ := total_cost - total_soda_cost):
  sandwiches = (total_sandwich_cost / cost_sandwich) → sandwiches = 2 := by 
  sorry

end sandwiches_cost_l2050_205090


namespace part1_part2_l2050_205071

noncomputable def f (a x : ℝ) : ℝ := (a * Real.exp x - a - x) * Real.exp x

theorem part1 (a : ℝ) (h0 : a ≥ 0) (h1 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := 
sorry

theorem part2 (h1 : ∀ x : ℝ, f 1 x ≥ 0) :
  ∃! x0 : ℝ, (∀ x : ℝ, x0 = x → 
  (f 1 x0) = (f 1 x)) ∧ (0 < f 1 x0 ∧ f 1 x0 < 1/4) :=
sorry

end part1_part2_l2050_205071


namespace sufficient_but_not_necessary_l2050_205088

-- Definitions for lines a and b, and planes alpha and beta
variables {a b : Type} {α β : Type}

-- predicate for line a being in plane α
def line_in_plane (a : Type) (α : Type) : Prop := sorry

-- predicate for line b being perpendicular to plane β
def line_perpendicular_plane (b : Type) (β : Type) : Prop := sorry

-- predicate for plane α being parallel to plane β
def plane_parallel_plane (α : Type) (β : Type) : Prop := sorry

-- predicate for line a being perpendicular to line b
def line_perpendicular_line (a : Type) (b : Type) : Prop := sorry

-- Proof of the statement: The condition of line a being in plane α, line b being perpendicular to plane β,
-- and plane α being parallel to plane β is sufficient but not necessary for line a being perpendicular to line b.
theorem sufficient_but_not_necessary
  (a b : Type) (α β : Type)
  (h1 : line_in_plane a α)
  (h2 : line_perpendicular_plane b β)
  (h3 : plane_parallel_plane α β) :
  line_perpendicular_line a b :=
sorry

end sufficient_but_not_necessary_l2050_205088


namespace star_15_star_eq_neg_15_l2050_205075

-- Define the operations as given
def y_star (y : ℤ) := 9 - y
def star_y (y : ℤ) := y - 9

-- The theorem stating the required proof
theorem star_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by
  sorry

end star_15_star_eq_neg_15_l2050_205075


namespace sequence_to_one_l2050_205012

def nextStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n - 1

theorem sequence_to_one (n : ℕ) (h : n > 0) :
  ∃ seq : ℕ → ℕ, seq 0 = n ∧ (∀ i, seq (i + 1) = nextStep (seq i)) ∧ (∃ j, seq j = 1) := by
  sorry

end sequence_to_one_l2050_205012


namespace bogan_maggots_l2050_205034

theorem bogan_maggots (x : ℕ) (total_maggots : ℕ) (eaten_first : ℕ) (eaten_second : ℕ) (thrown_out : ℕ) 
  (h1 : eaten_first = 1) (h2 : eaten_second = 3) (h3 : total_maggots = 20) (h4 : thrown_out = total_maggots - eaten_first - eaten_second) 
  (h5 : x + eaten_first = thrown_out) : x = 15 :=
by
  -- Use the given conditions
  sorry

end bogan_maggots_l2050_205034


namespace man_l2050_205087

-- Define the conditions
def speed_downstream : ℕ := 8
def speed_upstream : ℕ := 4

-- Define the man's rate in still water
def rate_in_still_water : ℕ := (speed_downstream + speed_upstream) / 2

-- The target theorem
theorem man's_rate_in_still_water : rate_in_still_water = 6 := by
  -- The statement is set up. Proof to be added later.
  sorry

end man_l2050_205087


namespace virginia_initial_eggs_l2050_205032

theorem virginia_initial_eggs (final_eggs : ℕ) (taken_eggs : ℕ) (H : final_eggs = 93) (G : taken_eggs = 3) : final_eggs + taken_eggs = 96 := 
by
  -- proof part could go here
  sorry

end virginia_initial_eggs_l2050_205032


namespace bacon_suggestion_count_l2050_205078

theorem bacon_suggestion_count (B : ℕ) (h1 : 408 = B + 366) : B = 42 :=
by
  sorry

end bacon_suggestion_count_l2050_205078


namespace mona_unique_players_l2050_205023

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end mona_unique_players_l2050_205023


namespace M_intersection_N_l2050_205060

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the proof problem
theorem M_intersection_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_intersection_N_l2050_205060


namespace general_term_l2050_205009

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end general_term_l2050_205009


namespace arithmetic_expression_evaluation_l2050_205080

theorem arithmetic_expression_evaluation :
  (3 + 9) ^ 2 + (3 ^ 2) * (9 ^ 2) = 873 :=
by
  -- Proof is skipped, using sorry for now.
  sorry

end arithmetic_expression_evaluation_l2050_205080


namespace o_hara_triple_example_l2050_205057

-- definitions
def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a) + (Real.sqrt b) = x

-- conditions
def a : ℕ := 81
def b : ℕ := 49
def x : ℕ := 16

-- statement
theorem o_hara_triple_example : is_OHara_triple a b x :=
by
  sorry

end o_hara_triple_example_l2050_205057
