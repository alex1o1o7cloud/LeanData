import Mathlib

namespace fraction_product_is_simplified_form_l1439_143926

noncomputable def fraction_product : ℚ := (2 / 3) * (5 / 11) * (3 / 8)

theorem fraction_product_is_simplified_form :
  fraction_product = 5 / 44 :=
by
  sorry

end fraction_product_is_simplified_form_l1439_143926


namespace sum_of_two_digit_divisors_l1439_143918

theorem sum_of_two_digit_divisors (d : ℕ) (h_pos : d > 0) (h_mod : 145 % d = 4) : d = 47 := 
by sorry

end sum_of_two_digit_divisors_l1439_143918


namespace solve_system_of_inequalities_l1439_143964

theorem solve_system_of_inequalities (x : ℝ) :
  (x + 1 < 5) ∧ (2 * x - 1) / 3 ≥ 1 ↔ 2 ≤ x ∧ x < 4 :=
by
  sorry

end solve_system_of_inequalities_l1439_143964


namespace units_digit_difference_l1439_143983

-- Conditions based on the problem statement
def units_digit_of_power_of_5 (n : ℕ) : ℕ := 5

def units_digit_of_power_of_3 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0     => 1
  | 1     => 3
  | 2     => 9
  | 3     => 7
  | _     => 0  -- impossible due to mod 4

-- Problem statement in Lean as a theorem
theorem units_digit_difference : (5^2019 - 3^2019) % 10 = 8 :=
by
  have h1 : (5^2019 % 10) = units_digit_of_power_of_5 2019 := sorry
  have h2 : (3^2019 % 10) = units_digit_of_power_of_3 2019 := sorry
  -- The core proof step will go here
  sorry

end units_digit_difference_l1439_143983


namespace T_5_value_l1439_143928

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1 / y)^m

theorem T_5_value (y : ℝ) (h : y + 1 / y = 5) : T y 5 = 2525 := 
by {
  sorry
}

end T_5_value_l1439_143928


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l1439_143930

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l1439_143930


namespace different_prime_factors_mn_is_five_l1439_143991

theorem different_prime_factors_mn_is_five {m n : ℕ} 
  (m_prime_factors : ∃ (p_1 p_2 p_3 p_4 : ℕ), True)  -- m has 4 different prime factors
  (n_prime_factors : ∃ (q_1 q_2 q_3 : ℕ), True)  -- n has 3 different prime factors
  (gcd_m_n : Nat.gcd m n = 15) : 
  (∃ k : ℕ, k = 5 ∧ (∃ (x_1 x_2 x_3 x_4 x_5 : ℕ), True)) := sorry

end different_prime_factors_mn_is_five_l1439_143991


namespace identity_of_brothers_l1439_143962

theorem identity_of_brothers
  (first_brother_speaks : Prop)
  (second_brother_speaks : Prop)
  (one_tells_truth : first_brother_speaks → ¬ second_brother_speaks)
  (other_tells_truth : ¬first_brother_speaks → second_brother_speaks) :
  first_brother_speaks = false ∧ second_brother_speaks = true :=
by
  sorry

end identity_of_brothers_l1439_143962


namespace Bill_initial_money_l1439_143911

theorem Bill_initial_money (joint_money : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (final_bill_amount : ℕ) (initial_joint_money_eq : joint_money = 42) (pizza_cost_eq : pizza_cost = 11) (num_pizzas_eq : num_pizzas = 3) (final_bill_amount_eq : final_bill_amount = 39) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end Bill_initial_money_l1439_143911


namespace DeMorgansLaws_l1439_143924

variable (U : Type) (A B : Set U)

theorem DeMorgansLaws :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∧ (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ :=
by
  -- Statement of the theorems, proof is omitted
  sorry

end DeMorgansLaws_l1439_143924


namespace admission_charge_l1439_143980

variable (A : ℝ) -- Admission charge in dollars
variable (tour_charge : ℝ)
variable (group1_size : ℕ)
variable (group2_size : ℕ)
variable (total_earnings : ℝ)

-- Given conditions
axiom h1 : tour_charge = 6
axiom h2 : group1_size = 10
axiom h3 : group2_size = 5
axiom h4 : total_earnings = 240
axiom h5 : (group1_size * A + group1_size * tour_charge) + (group2_size * A) = total_earnings

theorem admission_charge : A = 12 :=
by
  sorry

end admission_charge_l1439_143980


namespace andy_demerits_l1439_143916

theorem andy_demerits (x : ℕ) :
  (∀ x, 6 * x + 15 = 27 → x = 2) :=
by
  intro
  sorry

end andy_demerits_l1439_143916


namespace solution_set_of_inequality_l1439_143989

theorem solution_set_of_inequality:
  {x : ℝ | x^2 - |x-1| - 1 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end solution_set_of_inequality_l1439_143989


namespace find_other_divisor_l1439_143995

theorem find_other_divisor (x : ℕ) (h : x ≠ 35) (h1 : 386 % 35 = 1) (h2 : 386 % x = 1) : x = 11 :=
sorry

end find_other_divisor_l1439_143995


namespace connie_marbles_l1439_143949

theorem connie_marbles (j c : ℕ) (h1 : j = 498) (h2 : j = c + 175) : c = 323 :=
by
  -- Placeholder for the proof
  sorry

end connie_marbles_l1439_143949


namespace twelfth_term_geometric_sequence_l1439_143973

theorem twelfth_term_geometric_sequence :
  let a1 := 5
  let r := (2 / 5 : ℝ)
  (a1 * r ^ 11) = (10240 / 48828125 : ℝ) :=
by
  sorry

end twelfth_term_geometric_sequence_l1439_143973


namespace set_equality_proof_l1439_143958

theorem set_equality_proof :
  {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} :=
by
  sorry

end set_equality_proof_l1439_143958


namespace line_through_circles_l1439_143929

theorem line_through_circles (D1 E1 D2 E2 : ℝ)
  (h1 : 2 * D1 - E1 + 2 = 0)
  (h2 : 2 * D2 - E2 + 2 = 0) :
  (2 * D1 - E1 + 2 = 0) ∧ (2 * D2 - E2 + 2 = 0) :=
by
  exact ⟨h1, h2⟩

end line_through_circles_l1439_143929


namespace variance_of_given_data_is_2_l1439_143968

-- Define the data set
def data_set : List ℕ := [198, 199, 200, 201, 202]

-- Define the mean function for a given data set
noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

-- Define the variance function for a given data set
noncomputable def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x : ℝ) - μ) |>.map (λ x => x^2)).sum / data.length

-- Proposition that the variance of the given data set is 2
theorem variance_of_given_data_is_2 : variance data_set = 2 := by
  sorry

end variance_of_given_data_is_2_l1439_143968


namespace cubic_inequality_l1439_143922

theorem cubic_inequality (x y z : ℝ) :
  x^3 + y^3 + z^3 + 3 * x * y * z ≥ x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) :=
sorry

end cubic_inequality_l1439_143922


namespace running_hours_per_week_l1439_143936

theorem running_hours_per_week 
  (initial_days : ℕ) (additional_days : ℕ) (morning_run_time : ℕ) (evening_run_time : ℕ)
  (total_days : ℕ) (total_run_time_per_day : ℕ) (total_run_time_per_week : ℕ)
  (H1 : initial_days = 3)
  (H2 : additional_days = 2)
  (H3 : morning_run_time = 1)
  (H4 : evening_run_time = 1)
  (H5 : total_days = initial_days + additional_days)
  (H6 : total_run_time_per_day = morning_run_time + evening_run_time)
  (H7 : total_run_time_per_week = total_days * total_run_time_per_day) :
  total_run_time_per_week = 10 := 
sorry

end running_hours_per_week_l1439_143936


namespace work_problem_l1439_143975

theorem work_problem (days_B : ℝ) (h : (1 / 20) + (1 / days_B) = 1 / 8.571428571428571) : days_B = 15 :=
sorry

end work_problem_l1439_143975


namespace Vasya_has_larger_amount_l1439_143988

-- Defining the conditions and given data
variables (V P : ℝ)

-- Vasya's profit calculation
def Vasya_profit (V : ℝ) : ℝ := 0.20 * V

-- Petya's profit calculation considering exchange rate increase
def Petya_profit (P : ℝ) : ℝ := 0.2045 * P

-- Proof statement
theorem Vasya_has_larger_amount (h : Vasya_profit V = Petya_profit P) : V > P :=
sorry

end Vasya_has_larger_amount_l1439_143988


namespace probability_two_boys_l1439_143941

-- Definitions for the conditions
def total_students : ℕ := 4
def boys : ℕ := 3
def girls : ℕ := 1
def select_students : ℕ := 2

-- Combination function definition
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_boys :
  (combination boys select_students) / (combination total_students select_students) = 1 / 2 := by
  sorry

end probability_two_boys_l1439_143941


namespace perimeter_correct_l1439_143904

-- Definitions based on the conditions
def large_rectangle_area : ℕ := 12 * 12
def shaded_rectangle_area : ℕ := 6 * 4
def non_shaded_area : ℕ := large_rectangle_area - shaded_rectangle_area
def perimeter_of_non_shaded_region : ℕ := 2 * ((12 - 6) + (12 - 4))

-- The theorem to prove
theorem perimeter_correct (large_rectangle_area_eq : large_rectangle_area = 144) :
  perimeter_of_non_shaded_region = 28 :=
by
  sorry

end perimeter_correct_l1439_143904


namespace total_pizzas_eaten_l1439_143959

-- Definitions for the conditions
def pizzasA : ℕ := 8
def pizzasB : ℕ := 7

-- Theorem stating the total number of pizzas eaten by both classes
theorem total_pizzas_eaten : pizzasA + pizzasB = 15 := 
by
  -- Proof is not required for the task, so we use sorry
  sorry

end total_pizzas_eaten_l1439_143959


namespace points_deducted_for_incorrect_answer_is_5_l1439_143931

-- Define the constants and variables used in the problem
def total_questions : ℕ := 30
def points_per_correct_answer : ℕ := 20
def correct_answers : ℕ := 19
def incorrect_answers : ℕ := total_questions - correct_answers
def final_score : ℕ := 325

-- Define a function that models the total score calculation
def calculate_final_score (points_deducted_per_incorrect : ℕ) : ℕ :=
  (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect)

-- The theorem that states the problem and expected solution
theorem points_deducted_for_incorrect_answer_is_5 :
  ∃ (x : ℕ), calculate_final_score x = final_score ∧ x = 5 :=
by
  sorry

end points_deducted_for_incorrect_answer_is_5_l1439_143931


namespace find_a4_l1439_143940

-- Define the arithmetic sequence and the sum of the first N terms
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first N terms in an arithmetic sequence
def sum_arithmetic_seq (a d N : ℕ) : ℕ := N * (2 * a + (N - 1) * d) / 2

-- Define the conditions
def condition1 (a d : ℕ) : Prop := a + (a + 2 * d) + (a + 4 * d) = 15
def condition2 (a d : ℕ) : Prop := sum_arithmetic_seq a d 4 = 16

-- Lean 4 statement to prove the value of a_4
theorem find_a4 (a d : ℕ) (h1 : condition1 a d) (h2 : condition2 a d) : arithmetic_seq a d 4 = 7 :=
sorry

end find_a4_l1439_143940


namespace ab_perpendicular_cd_l1439_143943

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assuming points are members of a metric space and distances are calculated using the distance function
variables (a b c d : A)

-- Given condition
def given_condition : Prop := 
  dist a c ^ 2 + dist b d ^ 2 = dist a d ^ 2 + dist b c ^ 2

-- Statement that needs to be proven
theorem ab_perpendicular_cd (h : given_condition a b c d) : dist a b * dist c d = 0 :=
sorry

end ab_perpendicular_cd_l1439_143943


namespace average_monthly_growth_rate_l1439_143967

-- Define the initial and final production quantities
def initial_production : ℝ := 100
def final_production : ℝ := 144

-- Define the average monthly growth rate
def avg_monthly_growth_rate (x : ℝ) : Prop :=
  initial_production * (1 + x)^2 = final_production

-- Statement of the problem to be verified
theorem average_monthly_growth_rate :
  ∃ x : ℝ, avg_monthly_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_growth_rate_l1439_143967


namespace average_next_3_numbers_l1439_143977

theorem average_next_3_numbers 
  (a1 a2 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_avg_total : (a1 + a2 + b1 + b2 + b3 + c1 + c2 + c3) / 8 = 25)
  (h_avg_first2: (a1 + a2) / 2 = 20)
  (h_c1_c2 : c1 + 4 = c2)
  (h_c1_c3 : c1 + 6 = c3)
  (h_c3_value : c3 = 30) :
  (b1 + b2 + b3) / 3 = 26 := 
sorry

end average_next_3_numbers_l1439_143977


namespace trains_meet_time_l1439_143947

theorem trains_meet_time :
  (∀ (D : ℝ) (s1 s2 t1 t2 : ℝ),
    D = 155 ∧ 
    s1 = 20 ∧ 
    s2 = 25 ∧ 
    t1 = 7 ∧ 
    t2 = 8 →
    (∃ t : ℝ, 20 * t + 25 * t = D - 20)) →
  8 + 3 = 11 :=
by {
  sorry
}

end trains_meet_time_l1439_143947


namespace quotient_base4_correct_l1439_143923

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 1302 => 1 * 4^3 + 3 * 4^2 + 0 * 4^1 + 2 * 4^0
  | 12 => 1 * 4^1 + 2 * 4^0
  | _ => 0

def base10_to_base4 (n : ℕ) : ℕ :=
  match n with
  | 19 => 1 * 4^2 + 0 * 4^1 + 3 * 4^0
  | _ => 0

theorem quotient_base4_correct : base10_to_base4 (114 / 6) = 103 := 
  by sorry

end quotient_base4_correct_l1439_143923


namespace area_of_EFGH_l1439_143982

variables (EF FG EH HG EG : ℝ)
variables (distEFGH : EF ≠ HG ∧ EG = 5 ∧ EF^2 + FG^2 = 25 ∧ EH^2 + HG^2 = 25)

theorem area_of_EFGH : 
  ∃ EF FG EH HG : ℕ, EF ≠ HG ∧ EG = 5 
  ∧ EF^2 + FG^2 = 25 
  ∧ EH^2 + HG^2 = 25 
  ∧ EF * FG / 2 + EH * HG / 2 = 12 :=
by { sorry }

end area_of_EFGH_l1439_143982


namespace base_conversion_l1439_143909

noncomputable def b_value : ℝ := Real.sqrt 21

theorem base_conversion (b : ℝ) (h : b = Real.sqrt 21) : 
  (1 * b^2 + 0 * b + 2) = 23 := 
by
  rw [h]
  sorry

end base_conversion_l1439_143909


namespace exponent_of_9_in_9_pow_7_l1439_143903

theorem exponent_of_9_in_9_pow_7 : ∀ x : ℕ, (3 ^ x ∣ 9 ^ 7) ↔ x ≤ 14 := by
  sorry

end exponent_of_9_in_9_pow_7_l1439_143903


namespace smaller_angle_at_3_20_correct_l1439_143913

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l1439_143913


namespace converse_l1439_143986

variables {x : ℝ}

def P (x : ℝ) : Prop := x < 0
def Q (x : ℝ) : Prop := x^2 > 0

theorem converse (h : Q x) : P x :=
sorry

end converse_l1439_143986


namespace simplify_expression_l1439_143979

-- Define the given expressions
def numerator : ℕ := 5^5 + 5^3 + 5
def denominator : ℕ := 5^4 - 2 * 5^2 + 5

-- Define the simplified fraction
def simplified_fraction : ℚ := numerator / denominator

-- Prove that the simplified fraction is equivalent to 651 / 116
theorem simplify_expression : simplified_fraction = 651 / 116 := by
  sorry

end simplify_expression_l1439_143979


namespace equation_of_perpendicular_line_l1439_143937

theorem equation_of_perpendicular_line (a b c : ℝ) (p q : ℝ) (hx : a ≠ 0) (hy : b ≠ 0)
  (h_perpendicular : a * 2 + b * 1 = 0) (h_point : (-1) * a + 2 * b + c = 0)
  : a = 1 ∧ b = -2 ∧ c = -5 → (x:ℝ) * 1 + (y:ℝ) * (-2) + (-5) = 0 :=
by sorry

end equation_of_perpendicular_line_l1439_143937


namespace total_order_cost_l1439_143921

theorem total_order_cost (n : ℕ) (cost_geo cost_eng : ℝ)
  (h1 : n = 35)
  (h2 : cost_geo = 10.50)
  (h3 : cost_eng = 7.50) :
  n * cost_geo + n * cost_eng = 630 := by
  -- proof steps should go here
  sorry

end total_order_cost_l1439_143921


namespace initial_distance_from_lens_l1439_143994

def focal_length := 150 -- focal length F in cm
def screen_shift := 40  -- screen moved by 40 cm

theorem initial_distance_from_lens (d : ℝ) (f : ℝ) (s : ℝ) 
  (h_focal_length : f = focal_length) 
  (h_screen_shift : s = screen_shift) 
  (h_parallel_beam : d = f / 2 ∨ d = 3 * f / 2) : 
  d = 130 ∨ d = 170 := 
by 
  sorry

end initial_distance_from_lens_l1439_143994


namespace minimization_problem_l1439_143953

theorem minimization_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) (h5 : x ≤ y) (h6 : y ≤ z) (h7 : z ≤ 3 * x) :
  x * y * z ≥ 1 / 18 := 
sorry

end minimization_problem_l1439_143953


namespace parabola_hyperbola_tangent_l1439_143902

open Real

theorem parabola_hyperbola_tangent (n : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 6 → y^2 - n * x^2 = 4 → y ≥ 6) ↔ (n = 12 + 4 * sqrt 7 ∨ n = 12 - 4 * sqrt 7) :=
by
  sorry

end parabola_hyperbola_tangent_l1439_143902


namespace biology_marks_l1439_143976

theorem biology_marks (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) (biology : ℕ) 
  (h1 : english = 36) 
  (h2 : math = 35) 
  (h3 : physics = 42) 
  (h4 : chemistry = 57) 
  (h5 : average = 45) 
  (h6 : (english + math + physics + chemistry + biology) / 5 = average) : 
  biology = 55 := 
by
  sorry

end biology_marks_l1439_143976


namespace number_of_divisors_of_3003_l1439_143984

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l1439_143984


namespace antipov_inequality_l1439_143933

theorem antipov_inequality (a b c : ℕ) 
  (h1 : ¬ (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) 
  (h2 : (ab + 1) ∣ (abc + 1)) : c ≥ b :=
sorry

end antipov_inequality_l1439_143933


namespace pentagon_area_eq_half_l1439_143981

variables {A B C D E : Type*} -- Assume A, B, C, D, E are some points in a plane

-- Assume the given conditions in the problem
variables (angle_A angle_C : ℝ)
variables (AB AE BC CD AC : ℝ)
variables (pentagon_area : ℝ)

-- Assume the constraints from the problem statement
axiom angle_A_eq_90 : angle_A = 90
axiom angle_C_eq_90 : angle_C = 90
axiom AB_eq_AE : AB = AE
axiom BC_eq_CD : BC = CD
axiom AC_eq_1 : AC = 1

theorem pentagon_area_eq_half : pentagon_area = 1 / 2 :=
sorry

end pentagon_area_eq_half_l1439_143981


namespace solve_for_x_l1439_143927

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l1439_143927


namespace rectangle_area_l1439_143996

theorem rectangle_area 
  (length_to_width_ratio : Real) 
  (width : Real) 
  (area : Real) 
  (h1 : length_to_width_ratio = 0.875) 
  (h2 : width = 24) 
  (h_area : area = 504) : 
  True := 
sorry

end rectangle_area_l1439_143996


namespace customers_left_tip_l1439_143942

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l1439_143942


namespace find_original_number_l1439_143957

theorem find_original_number (N : ℕ) (h : ∃ k : ℕ, N - 5 = 13 * k) : N = 18 :=
sorry

end find_original_number_l1439_143957


namespace mixed_nuts_price_l1439_143935

theorem mixed_nuts_price (total_weight : ℝ) (peanut_price : ℝ) (cashew_price : ℝ) (cashew_weight : ℝ) 
  (H1 : total_weight = 100) 
  (H2 : peanut_price = 3.50) 
  (H3 : cashew_price = 4.00) 
  (H4 : cashew_weight = 60) : 
  (cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price) / total_weight = 3.80 :=
by 
  sorry

end mixed_nuts_price_l1439_143935


namespace puppies_brought_in_l1439_143985

open Nat

theorem puppies_brought_in (orig_puppies adopt_rate days total_adopted brought_in_puppies : ℕ) 
  (h_orig : orig_puppies = 3)
  (h_adopt_rate : adopt_rate = 3)
  (h_days : days = 2)
  (h_total_adopted : total_adopted = adopt_rate * days)
  (h_equation : total_adopted = orig_puppies + brought_in_puppies) :
  brought_in_puppies = 3 :=
by
  sorry

end puppies_brought_in_l1439_143985


namespace smallest_class_size_l1439_143974

/--
In a science class, students are separated into five rows for an experiment. 
The class size must be greater than 50. 
Three rows have the same number of students, one row has two more students than the others, 
and another row has three more students than the others.
Prove that the smallest possible class size for this science class is 55.
-/
theorem smallest_class_size (class_size : ℕ) (n : ℕ) 
  (h1 : class_size = 3 * n + (n + 2) + (n + 3))
  (h2 : class_size > 50) :
  class_size = 55 :=
sorry

end smallest_class_size_l1439_143974


namespace conditions_for_k_b_l1439_143950

theorem conditions_for_k_b (k b : ℝ) :
  (∀ x : ℝ, (x - (kx + b) + 2) * (2) > 0) →
  (k = 1) ∧ (b < 2) :=
by
  intros h
  sorry

end conditions_for_k_b_l1439_143950


namespace find_triples_solution_l1439_143999

theorem find_triples_solution (x y z : ℕ) (h : x^5 + x^4 + 1 = 3^y * 7^z) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) :=
by
  sorry

end find_triples_solution_l1439_143999


namespace jack_marbles_l1439_143990

theorem jack_marbles (initial_marbles share_marbles : ℕ) (h_initial : initial_marbles = 62) (h_share : share_marbles = 33) : 
  initial_marbles - share_marbles = 29 :=
by 
  sorry

end jack_marbles_l1439_143990


namespace solution_set_of_inequality_l1439_143900

theorem solution_set_of_inequality (x : ℝ) : x > 1 ∨ (-1 < x ∧ x < 0) ↔ x > 1 ∨ (-1 < x ∧ x < 0) :=
by sorry

end solution_set_of_inequality_l1439_143900


namespace quadratic_single_root_a_l1439_143910

theorem quadratic_single_root_a (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end quadratic_single_root_a_l1439_143910


namespace benjamin_trip_odd_number_conditions_l1439_143917

theorem benjamin_trip_odd_number_conditions (a b c : ℕ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a + b + c ≤ 9) 
  (h5 : ∃ x : ℕ, 60 * x = 99 * (c - a)) :
  a^2 + b^2 + c^2 = 35 := 
sorry

end benjamin_trip_odd_number_conditions_l1439_143917


namespace calculation_result_l1439_143951

theorem calculation_result :
  (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 :=
by 
  sorry

end calculation_result_l1439_143951


namespace total_time_spent_l1439_143901

def timeDrivingToSchool := 20
def timeAtGroceryStore := 15
def timeFillingGas := 5
def timeAtParentTeacherNight := 70
def timeAtCoffeeShop := 30
def timeDrivingHome := timeDrivingToSchool

theorem total_time_spent : 
  timeDrivingToSchool + timeAtGroceryStore + timeFillingGas + timeAtParentTeacherNight + timeAtCoffeeShop + timeDrivingHome = 160 :=
by
  sorry

end total_time_spent_l1439_143901


namespace ratio_pen_pencil_l1439_143948

theorem ratio_pen_pencil (P : ℝ) (pencil_cost total_cost : ℝ) 
  (hc1 : pencil_cost = 8) 
  (hc2 : total_cost = 12)
  (hc3 : P + pencil_cost = total_cost) : 
  P / pencil_cost = 1 / 2 :=
by 
  sorry

end ratio_pen_pencil_l1439_143948


namespace number_of_girls_in_school_l1439_143972

theorem number_of_girls_in_school
  (total_students : ℕ)
  (avg_age_boys avg_age_girls avg_age_school : ℝ)
  (B G : ℕ)
  (h1 : total_students = 640)
  (h2 : avg_age_boys = 12)
  (h3 : avg_age_girls = 11)
  (h4 : avg_age_school = 11.75)
  (h5 : B + G = total_students)
  (h6 : (avg_age_boys * B + avg_age_girls * G = avg_age_school * total_students)) :
  G = 160 :=
by
  sorry

end number_of_girls_in_school_l1439_143972


namespace area_of_rectangular_field_l1439_143920

def length (L : ℝ) : Prop := L > 0
def breadth (L : ℝ) (B : ℝ) : Prop := B = 0.6 * L
def perimeter (L : ℝ) (B : ℝ) : Prop := 2 * L + 2 * B = 800
def area (L : ℝ) (B : ℝ) (A : ℝ) : Prop := A = L * B

theorem area_of_rectangular_field (L B A : ℝ) 
  (h1 : breadth L B) 
  (h2 : perimeter L B) : 
  area L B 37500 :=
sorry

end area_of_rectangular_field_l1439_143920


namespace age_ratio_l1439_143939

variable (R D : ℕ)

theorem age_ratio (h1 : D = 24) (h2 : R + 6 = 38) : R / D = 4 / 3 := by
  sorry

end age_ratio_l1439_143939


namespace find_judes_age_l1439_143963

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l1439_143963


namespace solution_set_f_ge_0_l1439_143925

noncomputable def f (x a : ℝ) : ℝ := 1 / Real.exp x - a / x

theorem solution_set_f_ge_0 (a m n : ℝ) (h : ∀ x, m ≤ x ∧ x ≤ n ↔ 1 / Real.exp x - a / x ≥ 0) : 
  0 < a ∧ a < 1 / Real.exp 1 :=
  sorry

end solution_set_f_ge_0_l1439_143925


namespace distance_between_stations_l1439_143998

theorem distance_between_stations
  (time_start_train1 time_meet time_start_train2 : ℕ) -- time in hours (7 a.m., 11 a.m., 8 a.m.)
  (speed_train1 speed_train2 : ℕ) -- speed in kmph (20 kmph, 25 kmph)
  (distance_covered_train1 distance_covered_train2 : ℕ)
  (total_distance : ℕ) :
  time_start_train1 = 7 ∧ time_meet = 11 ∧ time_start_train2 = 8 ∧ speed_train1 = 20 ∧ speed_train2 = 25 ∧
  distance_covered_train1 = (time_meet - time_start_train1) * speed_train1 ∧
  distance_covered_train2 = (time_meet - time_start_train2) * speed_train2 ∧
  total_distance = distance_covered_train1 + distance_covered_train2 →
  total_distance = 155 := by
{
  sorry
}

end distance_between_stations_l1439_143998


namespace cakes_left_l1439_143905

def cakes_yesterday : ℕ := 3
def baked_today : ℕ := 5
def sold_today : ℕ := 6

theorem cakes_left (cakes_yesterday baked_today sold_today : ℕ) : cakes_yesterday + baked_today - sold_today = 2 := by
  sorry

end cakes_left_l1439_143905


namespace meena_sold_to_stone_l1439_143969

def total_cookies_baked : ℕ := 5 * 12
def cookies_bought_brock : ℕ := 7
def cookies_bought_katy : ℕ := 2 * cookies_bought_brock
def cookies_left : ℕ := 15
def cookies_sold_total : ℕ := total_cookies_baked - cookies_left
def cookies_bought_friends : ℕ := cookies_bought_brock + cookies_bought_katy
def cookies_sold_stone : ℕ := cookies_sold_total - cookies_bought_friends
def dozens_sold_stone : ℕ := cookies_sold_stone / 12

theorem meena_sold_to_stone : dozens_sold_stone = 2 := by
  sorry

end meena_sold_to_stone_l1439_143969


namespace no_beverages_l1439_143993

noncomputable def businessmen := 30
def coffee := 15
def tea := 13
def water := 6
def coffee_tea := 7
def tea_water := 3
def coffee_water := 2
def all_three := 1

theorem no_beverages (businessmen coffee tea water coffee_tea tea_water coffee_water all_three):
  businessmen - (coffee + tea + water - coffee_tea - tea_water - coffee_water + all_three) = 7 :=
by sorry

end no_beverages_l1439_143993


namespace inverse_function_of_f_l1439_143997

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

theorem inverse_function_of_f:
  ∀ x : ℝ, x ≠ 3 → f (f_inv x) = x ∧ f_inv (f x) = x := by
sorry

end inverse_function_of_f_l1439_143997


namespace compare_fractions_l1439_143965

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l1439_143965


namespace series_divergence_l1439_143907

theorem series_divergence (a : ℕ → ℝ) (hdiv : ¬ ∃ l, ∑' n, a n = l) (hpos : ∀ n, a n > 0) (hnoninc : ∀ n m, n ≤ m → a m ≤ a n) : 
  ¬ ∃ l, ∑' n, (a n / (1 + n * a n)) = l :=
by
  sorry

end series_divergence_l1439_143907


namespace circle_common_chord_l1439_143914

theorem circle_common_chord (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧
  (x^2 + y^2 - 6 * x = 0) →
  (x + 3 * y = 0) :=
by
  sorry

end circle_common_chord_l1439_143914


namespace avg_weight_A_l1439_143906

-- Define the conditions
def num_students_A : ℕ := 40
def num_students_B : ℕ := 20
def avg_weight_B : ℝ := 40
def avg_weight_whole_class : ℝ := 46.67

-- State the theorem using these definitions
theorem avg_weight_A :
  ∃ W_A : ℝ,
    (num_students_A * W_A + num_students_B * avg_weight_B = (num_students_A + num_students_B) * avg_weight_whole_class) ∧
    W_A = 50.005 :=
by
  sorry

end avg_weight_A_l1439_143906


namespace james_spent_6_dollars_l1439_143956

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l1439_143956


namespace speed_of_first_train_l1439_143946

/-
Problem:
Two trains, with lengths 150 meters and 165 meters respectively, are running in opposite directions. One train is moving at 65 kmph, and they take 7.82006405004841 seconds to completely clear each other from the moment they meet. Prove that the speed of the first train is 79.99 kmph.
-/

theorem speed_of_first_train :
  ∀ (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) (speed1 : ℝ),
  length1 = 150 → length2 = 165 → speed2 = 65 → time = 7.82006405004841 →
  ( 3.6 * (length1 + length2) / time = speed1 + speed2 ) →
  speed1 = 79.99 :=
by
  intros length1 length2 speed2 time speed1 h_length1 h_length2 h_speed2 h_time h_formula
  rw [h_length1, h_length2, h_speed2, h_time] at h_formula
  sorry

end speed_of_first_train_l1439_143946


namespace not_all_divisible_by_6_have_prime_neighbors_l1439_143966

theorem not_all_divisible_by_6_have_prime_neighbors :
  ¬ ∀ n : ℕ, (6 ∣ n) → (Prime (n - 1) ∨ Prime (n + 1)) := by
  sorry

end not_all_divisible_by_6_have_prime_neighbors_l1439_143966


namespace jake_weight_l1439_143919

variable (J S : ℕ)

theorem jake_weight (h1 : J - 15 = 2 * S) (h2 : J + S = 132) : J = 93 := by
  sorry

end jake_weight_l1439_143919


namespace area_of_region_l1439_143944

noncomputable def T := 516

def region (x y : ℝ) : Prop :=
  |x| - |y| ≤ T - 500 ∧ |y| ≤ T - 500

theorem area_of_region :
  (4 * (T - 500)^2 = 1024) :=
  sorry

end area_of_region_l1439_143944


namespace regular_hours_l1439_143938

variable (R : ℕ)

theorem regular_hours (h1 : 5 * R + 6 * (44 - R) + 5 * R + 6 * (48 - R) = 472) : R = 40 :=
by
  sorry

end regular_hours_l1439_143938


namespace boat_distance_against_stream_l1439_143908

-- Definitions from Step a)
def speed_boat_still_water : ℝ := 15  -- speed of the boat in still water in km/hr
def distance_downstream : ℝ := 21  -- distance traveled downstream in one hour in km
def time_hours : ℝ := 1  -- time in hours

-- Translation of the described problem proof
theorem boat_distance_against_stream :
  ∃ (v_s : ℝ), (speed_boat_still_water + v_s = distance_downstream / time_hours) → 
               (15 - v_s = 9) :=
by
  sorry

end boat_distance_against_stream_l1439_143908


namespace girl_name_correct_l1439_143971

-- The Russian alphabet positions as a Lean list
def russianAlphabet : List (ℕ × Char) := [(1, 'А'), (2, 'Б'), (3, 'В'), (4, 'Г'), (5, 'Д'), (6, 'Е'), (7, 'Ё'), 
                                           (8, 'Ж'), (9, 'З'), (10, 'И'), (11, 'Й'), (12, 'К'), (13, 'Л'), 
                                           (14, 'М'), (15, 'Н'), (16, 'О'), (17, 'П'), (18, 'Р'), (19, 'С'), 
                                           (20, 'Т'), (21, 'У'), (22, 'Ф'), (23, 'Х'), (24, 'Ц'), (25, 'Ч'), 
                                           (26, 'Ш'), (27, 'Щ'), (28, 'Ъ'), (29, 'Ы'), (30, 'Ь'), (31, 'Э'), 
                                           (32, 'Ю'), (33, 'Я')]

-- The sequence of numbers representing the girl's name
def nameSequence : ℕ := 2011533

-- The corresponding name derived from the sequence
def derivedName : String := "ТАНЯ"

-- The equivalence proof statement
theorem girl_name_correct : 
  (nameSequence = 2011533 → derivedName = "ТАНЯ") :=
by
  intro h
  sorry

end girl_name_correct_l1439_143971


namespace proof_problem_l1439_143945

-- Define the conditions for the problem

def is_factor (a b : ℕ) : Prop :=
  ∃ n : ℕ, b = a * n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Statement that needs to be proven
theorem proof_problem :
  is_factor 5 65 ∧ ¬(is_divisor 19 361 ∧ ¬is_divisor 19 190) ∧ ¬(¬is_divisor 36 144 ∨ ¬is_divisor 36 73) ∧ ¬(is_divisor 14 28 ∧ ¬is_divisor 14 56) ∧ is_factor 9 144 :=
by sorry

end proof_problem_l1439_143945


namespace simplify_powers_of_ten_l1439_143970

theorem simplify_powers_of_ten :
  (10^0.4) * (10^0.5) * (10^0.2) * (10^(-0.6)) * (10^0.5) = 10 := 
by
  sorry

end simplify_powers_of_ten_l1439_143970


namespace work_done_together_in_one_day_l1439_143987

-- Defining the conditions
def time_to_finish_a : ℕ := 12
def time_to_finish_b : ℕ := time_to_finish_a / 2

-- Defining the work done in one day
def work_done_by_a_in_one_day : ℚ := 1 / time_to_finish_a
def work_done_by_b_in_one_day : ℚ := 1 / time_to_finish_b

-- The proof statement
theorem work_done_together_in_one_day : 
  work_done_by_a_in_one_day + work_done_by_b_in_one_day = 1 / 4 := by
  sorry

end work_done_together_in_one_day_l1439_143987


namespace max_value_of_a_l1439_143932

noncomputable def maximum_a : ℝ := 1/3

theorem max_value_of_a :
  ∀ x : ℝ, 1 + maximum_a * Real.cos x ≥ (2/3) * Real.sin ((Real.pi / 2) + 2 * x) :=
by 
  sorry

end max_value_of_a_l1439_143932


namespace apples_ratio_l1439_143952

theorem apples_ratio (initial_apples rickis_apples end_apples samsons_apples : ℕ)
(h_initial : initial_apples = 74)
(h_ricki : rickis_apples = 14)
(h_end : end_apples = 32)
(h_samson : initial_apples - rickis_apples - end_apples = samsons_apples) :
  samsons_apples / Nat.gcd samsons_apples rickis_apples = 2 ∧ rickis_apples / Nat.gcd samsons_apples rickis_apples = 1 :=
by
  sorry

end apples_ratio_l1439_143952


namespace infinite_fixpoints_l1439_143961

variable {f : ℕ+ → ℕ+}
variable (H : ∀ (m n : ℕ+), (∃ k : ℕ+ , k ≤ f n ∧ n ∣ f (m + k)) ∧ (∀ j : ℕ+ , j ≤ f n → j ≠ k → ¬ n ∣ f (m + j)))

theorem infinite_fixpoints : ∃ᶠ n in at_top, f n = n :=
sorry

end infinite_fixpoints_l1439_143961


namespace probability_of_yellow_face_l1439_143978

def total_faces : ℕ := 12
def red_faces : ℕ := 5
def yellow_faces : ℕ := 4
def blue_faces : ℕ := 2
def green_faces : ℕ := 1

theorem probability_of_yellow_face : (yellow_faces : ℚ) / (total_faces : ℚ) = 1 / 3 := by
  sorry

end probability_of_yellow_face_l1439_143978


namespace difference_of_two_numbers_l1439_143915

def nat_sum := 22305
def a := ∃ a: ℕ, 5 ∣ a
def is_b (a b: ℕ) := b = a / 10 + 3

theorem difference_of_two_numbers (a b : ℕ) (h : a + b = nat_sum) (h1 : 5 ∣ a) (h2 : is_b a b) : a - b = 14872 :=
by
  sorry

end difference_of_two_numbers_l1439_143915


namespace eighth_hexagonal_number_l1439_143992

theorem eighth_hexagonal_number : (8 * (2 * 8 - 1)) = 120 :=
  by
  sorry

end eighth_hexagonal_number_l1439_143992


namespace S_13_eq_3510_l1439_143955

def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

theorem S_13_eq_3510 : S 13 = 3510 :=
by
  sorry

end S_13_eq_3510_l1439_143955


namespace ranch_cows_variance_l1439_143934

variable (n : ℕ)
variable (p : ℝ)

-- Definition of the variance of a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem ranch_cows_variance : 
  binomial_variance 10 0.02 = 0.196 :=
by
  sorry

end ranch_cows_variance_l1439_143934


namespace arithmetic_progression_a_eq_1_l1439_143912

theorem arithmetic_progression_a_eq_1 
  (a : ℝ) 
  (h1 : 6 + 2 * a - 1 = 10 + 5 * a - (6 + 2 * a)) : 
  a = 1 :=
by
  sorry

end arithmetic_progression_a_eq_1_l1439_143912


namespace cab_speed_ratio_l1439_143954

variable (S_u S_c : ℝ)

theorem cab_speed_ratio (h1 : ∃ S_u S_c : ℝ, S_u * 25 = S_c * 30) : S_c / S_u = 5 / 6 :=
by
  sorry

end cab_speed_ratio_l1439_143954


namespace minimum_value_ineq_l1439_143960

theorem minimum_value_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 3 :=
by
  sorry

end minimum_value_ineq_l1439_143960
