import Mathlib

namespace prove_op_eq_l266_26612

-- Define the new operation ⊕
def op (x y : ℝ) := x^3 - 2*y + x

-- State that for any k, k ⊕ (k ⊕ k) = -k^3 + 3k
theorem prove_op_eq (k : ℝ) : op k (op k k) = -k^3 + 3*k :=
by 
  sorry

end prove_op_eq_l266_26612


namespace iron_weighs_more_l266_26672

-- Define the weights of the metal pieces
def weight_iron : ℝ := 11.17
def weight_aluminum : ℝ := 0.83

-- State the theorem to prove that the difference in weights is 10.34 pounds
theorem iron_weighs_more : weight_iron - weight_aluminum = 10.34 :=
by sorry

end iron_weighs_more_l266_26672


namespace average_weight_calculation_l266_26635

noncomputable def new_average_weight (initial_people : ℕ) (initial_avg_weight : ℝ) 
                                     (new_person_weight : ℝ) (total_people : ℕ) : ℝ :=
  (initial_people * initial_avg_weight + new_person_weight) / total_people

theorem average_weight_calculation :
  new_average_weight 6 160 97 7 = 151 := by
  sorry

end average_weight_calculation_l266_26635


namespace investment_ratio_l266_26664

theorem investment_ratio (total_investment Jim_investment : ℕ) (h₁ : total_investment = 80000) (h₂ : Jim_investment = 36000) :
  (total_investment - Jim_investment) / Nat.gcd (total_investment - Jim_investment) Jim_investment = 11 ∧ Jim_investment / Nat.gcd (total_investment - Jim_investment) Jim_investment = 9 :=
by
  sorry

end investment_ratio_l266_26664


namespace tulip_area_of_flower_bed_l266_26694

theorem tulip_area_of_flower_bed 
  (CD CF : ℝ) (DE : ℝ := 4) (EF : ℝ := 3) 
  (triangle : ∀ (A B C : ℝ), A = B + C) : 
  CD * CF = 12 :=
by sorry

end tulip_area_of_flower_bed_l266_26694


namespace find_hanyoung_weight_l266_26677

variable (H J : ℝ)

def hanyoung_is_lighter (H J : ℝ) : Prop := H = J - 4
def sum_of_weights (H J : ℝ) : Prop := H + J = 88

theorem find_hanyoung_weight (H J : ℝ) (h1 : hanyoung_is_lighter H J) (h2 : sum_of_weights H J) : H = 42 :=
by
  sorry

end find_hanyoung_weight_l266_26677


namespace coefficient_of_squared_term_l266_26631

theorem coefficient_of_squared_term (a b c : ℝ) (h_eq : 5 * a^2 + 14 * b + 5 = 0) :
  a = 5 :=
sorry

end coefficient_of_squared_term_l266_26631


namespace divide_friends_among_teams_l266_26611

theorem divide_friends_among_teams :
  let friends_num := 8
  let teams_num := 4
  (teams_num ^ friends_num) = 65536 := by
  sorry

end divide_friends_among_teams_l266_26611


namespace product_of_solutions_abs_eq_four_l266_26668

theorem product_of_solutions_abs_eq_four :
  (∀ x : ℝ, (|x - 5| - 4 = 0) → (x = 9 ∨ x = 1)) →
  (9 * 1 = 9) :=
by
  intros h
  sorry

end product_of_solutions_abs_eq_four_l266_26668


namespace pentagon_angle_E_l266_26621

theorem pentagon_angle_E 
    (A B C D E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
    (AB BC CD DE : ℝ)
    (angle_B angle_C angle_D : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = DE)
    (h4 : angle_B = 96)
    (h5 : angle_C = 108)
    (h6 : angle_D = 108) :
    ∃ angle_E : ℝ, angle_E = 102 := 
by
  sorry

end pentagon_angle_E_l266_26621


namespace sum_of_squares_l266_26662

theorem sum_of_squares (x : ℕ) (h : 2 * x = 14) : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862 := 
by 
  sorry

end sum_of_squares_l266_26662


namespace proof_problem_l266_26602

variable (a b c A B C : ℝ)
variable (h_a : a = Real.sqrt 3)
variable (h_b_ge_a : b ≥ a)
variable (h_cos : Real.cos (2 * C) - Real.cos (2 * A) =
  2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C))

theorem proof_problem :
  (A = Real.pi / 3) ∧ (2 * b - c ∈ Set.Ico (Real.sqrt 3) (2 * Real.sqrt 3)) :=
  sorry

end proof_problem_l266_26602


namespace smallest_solution_l266_26686

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l266_26686


namespace remainder_when_divided_by_seven_l266_26689

theorem remainder_when_divided_by_seven (n : ℕ) (h₁ : n^3 ≡ 3 [MOD 7]) (h₂ : n^4 ≡ 2 [MOD 7]) : 
  n ≡ 6 [MOD 7] :=
sorry

end remainder_when_divided_by_seven_l266_26689


namespace quadratic_completion_l266_26641

theorem quadratic_completion (x : ℝ) :
  (x^2 + 6 * x - 2) = ((x + 3)^2 - 11) := sorry

end quadratic_completion_l266_26641


namespace no_integers_exist_l266_26646

theorem no_integers_exist :
  ¬ ∃ a b : ℤ, ∃ x y : ℤ, a^5 * b + 3 = x^3 ∧ a * b^5 + 3 = y^3 :=
by
  sorry

end no_integers_exist_l266_26646


namespace plot_length_l266_26678

def breadth : ℝ := 40 -- Derived from conditions and cost equation solution
def length : ℝ := breadth + 20
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300

theorem plot_length :
  (2 * (breadth + (breadth + 20))) * cost_per_meter = total_cost → length = 60 :=
by {
  sorry
}

end plot_length_l266_26678


namespace one_third_of_product_l266_26643

theorem one_third_of_product (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 4) : (1 / 3 : ℚ) * (a * b * c : ℕ) = 84 := by
  sorry

end one_third_of_product_l266_26643


namespace problem_statement_l266_26676

def Omega (n : ℕ) : ℕ := 
  -- Number of prime factors of n, counting multiplicity
  sorry

def f1 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 1 (mod 4)
  sorry

def f3 (n : ℕ) : ℕ :=
  -- Sum of positive divisors d|n where Omega(d) ≡ 3 (mod 4)
  sorry

theorem problem_statement : 
  f3 (6 ^ 2020) - f1 (6 ^ 2020) = (1 / 10 : ℚ) * (6 ^ 2021 - 3 ^ 2021 - 2 ^ 2021 - 1) :=
sorry

end problem_statement_l266_26676


namespace sheep_to_cow_water_ratio_l266_26654

-- Set up the initial conditions
def number_of_cows := 40
def water_per_cow_per_day := 80
def number_of_sheep := 10 * number_of_cows
def total_water_per_week := 78400

-- Calculate total water consumption of cows per week
def water_cows_per_week := number_of_cows * water_per_cow_per_day * 7

-- Calculate total water consumption of sheep per week
def water_sheep_per_week := total_water_per_week - water_cows_per_week

-- Calculate daily water consumption per sheep
def water_sheep_per_day := water_sheep_per_week / 7
def daily_water_per_sheep := water_sheep_per_day / number_of_sheep

-- Define the target ratio
def target_ratio := 1 / 4

-- Statement to prove
theorem sheep_to_cow_water_ratio :
  (daily_water_per_sheep / water_per_cow_per_day) = target_ratio :=
sorry

end sheep_to_cow_water_ratio_l266_26654


namespace range_of_k_l266_26658

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 - 2 * x - 1 = 0) ↔ (k ≥ -3 ∧ k ≠ -2) :=
by
  sorry

end range_of_k_l266_26658


namespace vector_parallel_solution_l266_26618

theorem vector_parallel_solution (x : ℝ) :
  let a := (1, x)
  let b := (x - 1, 2)
  (a.1 * b.2 = a.2 * b.1) → (x = 2 ∨ x = -1) :=
by
  intros a b h
  let a := (1, x)
  let b := (x - 1, 2)
  sorry

end vector_parallel_solution_l266_26618


namespace find_number_l266_26628

theorem find_number (x : ℝ) (h : 4 * (3 * x / 5 - 220) = 320) : x = 500 :=
sorry

end find_number_l266_26628


namespace Kristyna_number_l266_26667

theorem Kristyna_number (k n : ℕ) (h1 : k = 6 * n + 3) (h2 : 3 * n + 1 + 2 * n = 1681) : k = 2019 := 
by
  -- Proof goes here
  sorry

end Kristyna_number_l266_26667


namespace winning_ticket_probability_l266_26659

theorem winning_ticket_probability (eligible_numbers : List ℕ) (length_eligible_numbers : eligible_numbers.length = 12)
(pick_6 : Π(t : List ℕ), List ℕ) (valid_ticket : List ℕ → Bool) (probability : ℚ) : 
(probability = (1 : ℚ) / (4 : ℚ)) :=
  sorry

end winning_ticket_probability_l266_26659


namespace max_xy_min_function_l266_26675

-- Problem 1: Prove that the maximum value of xy is 8 given the conditions
theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 8) : xy ≤ 8 :=
sorry

-- Problem 2: Prove that the minimum value of the function is 9 given the conditions
theorem min_function (x : ℝ) (hx : -1 < x) : (x + 4 / (x + 1) + 6) ≥ 9 :=
sorry

end max_xy_min_function_l266_26675


namespace max_value_expression_l266_26693

noncomputable def f : Real → Real := λ x => 3 * Real.sin x + 4 * Real.cos x

theorem max_value_expression (θ : Real) (h_max : ∀ x, f x ≤ 5) :
  (3 * Real.sin θ + 4 * Real.cos θ = 5) →
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end max_value_expression_l266_26693


namespace overall_profit_percentage_l266_26608

theorem overall_profit_percentage :
  let SP_A := 900
  let SP_B := 1200
  let SP_C := 1500
  let P_A := 300
  let P_B := 400
  let P_C := 500
  let CP_A := SP_A - P_A
  let CP_B := SP_B - P_B
  let CP_C := SP_C - P_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let ProfitPercentage := (TP / TCP) * 100
  ProfitPercentage = 50 := by
  sorry

end overall_profit_percentage_l266_26608


namespace jacob_total_bill_l266_26629

def base_cost : ℝ := 25
def included_hours : ℕ := 25
def cost_per_text : ℝ := 0.08
def cost_per_extra_minute : ℝ := 0.13
def jacob_texts : ℕ := 150
def jacob_hours : ℕ := 31

theorem jacob_total_bill : 
  let extra_minutes := (jacob_hours - included_hours) * 60
  let total_cost := base_cost + jacob_texts * cost_per_text + extra_minutes * cost_per_extra_minute
  total_cost = 83.80 := 
by 
  -- Placeholder for proof
  sorry

end jacob_total_bill_l266_26629


namespace number_of_positive_integers_l266_26651

theorem number_of_positive_integers (n : ℕ) : 
  (0 < n ∧ n < 36 ∧ (∃ k : ℕ, n = k * (36 - k))) → 
  n = 18 ∨ n = 24 ∨ n = 30 ∨ n = 32 ∨ n = 34 ∨ n = 35 :=
sorry

end number_of_positive_integers_l266_26651


namespace fraction_product_eq_l266_26697

theorem fraction_product_eq :
  (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) = 4 / 9 :=
by
  sorry

end fraction_product_eq_l266_26697


namespace boys_in_class_l266_26600

theorem boys_in_class 
  (avg_weight_incorrect : ℝ)
  (misread_weight_diff : ℝ)
  (avg_weight_correct : ℝ) 
  (n : ℕ) 
  (h1 : avg_weight_incorrect = 58.4) 
  (h2 : misread_weight_diff = 4) 
  (h3 : avg_weight_correct = 58.6) 
  (h4 : n * avg_weight_incorrect + misread_weight_diff = n * avg_weight_correct) :
  n = 20 := 
sorry

end boys_in_class_l266_26600


namespace power_function_evaluation_l266_26624

noncomputable def f (α : ℝ) (x : ℝ) := x ^ α

theorem power_function_evaluation (α : ℝ) (h : f α 8 = 2) : f α (-1/8) = -1/2 :=
by
  sorry

end power_function_evaluation_l266_26624


namespace simplify_expr1_simplify_expr2_l266_26692

theorem simplify_expr1 : 
  (1:ℝ) * (-3:ℝ) ^ 0 + (- (1/2:ℝ)) ^ (-2:ℝ) - (-3:ℝ) ^ (-1:ℝ) = 16 / 3 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 
  ((-2 * x^3) ^ 2 * (-x^2)) / ((-x)^2) ^ 3 = -4 * x^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l266_26692


namespace sequence_a8_value_l266_26613

theorem sequence_a8_value :
  ∃ a : ℕ → ℚ, a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) / a n = n / (n + 1)) ∧ a 8 = 1 / 8 :=
by
  -- To be proved
  sorry

end sequence_a8_value_l266_26613


namespace inequality_one_solution_inequality_two_solution_l266_26617

-- The statement for the first inequality
theorem inequality_one_solution (x : ℝ) :
  |1 - ((2 * x - 1) / 3)| ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5 := sorry

-- The statement for the second inequality
theorem inequality_two_solution (x : ℝ) :
  (2 - x) * (x + 3) < 2 - x ↔ x < -2 ∨ x > 2 := sorry

end inequality_one_solution_inequality_two_solution_l266_26617


namespace sean_total_cost_l266_26642

noncomputable def total_cost (soda_cost soup_cost sandwich_cost : ℕ) (num_soda num_soup num_sandwich : ℕ) : ℕ :=
  num_soda * soda_cost + num_soup * soup_cost + num_sandwich * sandwich_cost

theorem sean_total_cost :
  let soda_cost := 1
  let soup_cost := 3 * soda_cost
  let sandwich_cost := 3 * soup_cost
  let num_soda := 3
  let num_soup := 2
  let num_sandwich := 1
  total_cost soda_cost soup_cost sandwich_cost num_soda num_soup num_sandwich = 18 :=
by
  sorry

end sean_total_cost_l266_26642


namespace fraction_ratio_l266_26648

theorem fraction_ratio (x y : ℕ) (h : (x / y : ℚ) / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 27 ∧ y = 35 :=
by 
  sorry

end fraction_ratio_l266_26648


namespace geometric_sequence_first_term_l266_26695

theorem geometric_sequence_first_term (a r : ℚ) (third_term fourth_term : ℚ) 
  (h1 : third_term = a * r^2)
  (h2 : fourth_term = a * r^3)
  (h3 : third_term = 27)
  (h4 : fourth_term = 36) : 
  a = 243 / 16 :=
by
  sorry

end geometric_sequence_first_term_l266_26695


namespace smallest_n_for_inequality_l266_26687

theorem smallest_n_for_inequality :
  ∃ n : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧ 
    (∀ m : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ m * (w^6 + x^6 + y^6 + z^6)) → m ≥ 64) :=
by
  sorry

end smallest_n_for_inequality_l266_26687


namespace solve_equation_l266_26622

theorem solve_equation (x : ℕ) (h : x = 88320) : x + 1315 + 9211 - 1569 = 97277 :=
by sorry

end solve_equation_l266_26622


namespace infinite_solutions_of_system_l266_26670

theorem infinite_solutions_of_system :
  ∃x y : ℝ, (3 * x - 4 * y = 10 ∧ 6 * x - 8 * y = 20) :=
by
  sorry

end infinite_solutions_of_system_l266_26670


namespace slowest_time_l266_26609

open Real

def time_lola (stories : ℕ) (run_time : ℝ) : ℝ := stories * run_time

def time_sam (stories_run stories_elevator : ℕ) (run_time elevate_time stop_time : ℝ) (wait_time : ℝ) : ℝ :=
  let run_part  := stories_run * run_time
  let wait_part := wait_time
  let elevator_part := stories_elevator * elevate_time + (stories_elevator - 1) * stop_time
  run_part + wait_part + elevator_part

def time_tara (stories : ℕ) (elevate_time stop_time : ℝ) : ℝ :=
  stories * elevate_time + (stories - 1) * stop_time

theorem slowest_time 
  (build_stories : ℕ) (lola_run_time sam_run_time elevate_time stop_time wait_time : ℝ)
  (h_build : build_stories = 50)
  (h_lola_run : lola_run_time = 12) (h_sam_run : sam_run_time = 15)
  (h_elevate : elevate_time = 10) (h_stop : stop_time = 4) (h_wait : wait_time = 20) :
  max (time_lola build_stories lola_run_time) 
    (max (time_sam 25 25 sam_run_time elevate_time stop_time wait_time) 
         (time_tara build_stories elevate_time stop_time)) = 741 := by
  sorry

end slowest_time_l266_26609


namespace value_of_f_f_2_l266_26680

def f (x : ℤ) : ℤ := 4 * x^2 + 2 * x - 1

theorem value_of_f_f_2 : f (f 2) = 1481 := by
  sorry

end value_of_f_f_2_l266_26680


namespace find_slope_and_intercept_l266_26663

noncomputable def line_equation_to_slope_intercept_form 
  (x y : ℝ) : Prop :=
  (3 * (x - 2) - 4 * (y + 3) = 0) ↔ (y = (3 / 4) * x - 4.5)

theorem find_slope_and_intercept : 
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), (line_equation_to_slope_intercept_form x y) → m = 3/4 ∧ b = -4.5) :=
sorry

end find_slope_and_intercept_l266_26663


namespace house_orderings_l266_26637

/-- Ralph walks past five houses each painted in a different color: 
orange, red, blue, yellow, and green.
Conditions:
1. Ralph passed the orange house before the red house.
2. Ralph passed the blue house before the yellow house.
3. The blue house was not next to the yellow house.
4. Ralph passed the green house before the red house and after the blue house.
Given these conditions, prove that there are exactly 3 valid orderings of the houses.
-/
theorem house_orderings : 
  ∃ (orderings : Finset (List String)), 
  orderings.card = 3 ∧
  (∀ (o : List String), 
   o ∈ orderings ↔ 
    ∃ (idx_o idx_r idx_b idx_y idx_g : ℕ), 
    o = ["orange", "red", "blue", "yellow", "green"] ∧
    idx_o < idx_r ∧ 
    idx_b < idx_y ∧ 
    (idx_b + 1 < idx_y ∨ idx_y + 1 < idx_b) ∧ 
    idx_b < idx_g ∧ idx_g < idx_r) := sorry

end house_orderings_l266_26637


namespace area_RWP_l266_26660

-- Definitions
variables (X Y Z W P Q R : ℝ × ℝ)
variables (h₁ : (X.1 - Z.1) * (X.1 - Z.1) + (X.2 - Z.2) * (X.2 - Z.2) = 144)
variables (h₂ : P.1 = X.1 - 8 ∧ P.2 = X.2)
variables (h₃ : Q.1 = (Z.1 + P.1) / 2 ∧ Q.2 = (Z.2 + P.2) / 2)
variables (h₄ : R.1 = (Y.1 + P.1) / 2 ∧ R.2 = (Y.2 + P.2) / 2)
variables (h₅ : 1 / 2 * ((Z.1 - X.1) * (W.2 - X.2) - (Z.2 - X.2) * (W.1 - X.1)) = 72)
variables (h₆ : 1 / 2 * abs ((Q.1 - X.1) * (W.2 - X.2) - (Q.2 - X.2) * (W.1 - X.1)) = 20)

-- Theorem statement
theorem area_RWP : 
  1 / 2 * abs ((R.1 - W.1) * (P.2 - W.2) - (R.2 - W.2) * (P.1 - W.1)) = 12 :=
sorry

end area_RWP_l266_26660


namespace find_m_given_a3_eq_40_l266_26630

theorem find_m_given_a3_eq_40 (m : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 - m * x) ^ 5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_3 = 40 →
  m = -1 := 
by 
  sorry

end find_m_given_a3_eq_40_l266_26630


namespace log2_a_div_b_squared_l266_26671

variable (a b : ℝ)
variable (ha_ne_1 : a ≠ 1) (hb_ne_1 : b ≠ 1)
variable (ha_pos : 0 < a) (hb_pos : 0 < b)
variable (h1 : 2 ^ (Real.log 32 / Real.log b) = a)
variable (h2 : a * b = 128)

theorem log2_a_div_b_squared :
  (Real.log ((a / b) : ℝ) / Real.log 2) ^ 2 = 29 + (49 / 4) :=
sorry

end log2_a_div_b_squared_l266_26671


namespace number_of_pupils_l266_26639

theorem number_of_pupils (n : ℕ) (M : ℕ)
  (avg_all : 39 * n = M)
  (pupil_marks : 25 + 12 + 15 + 19 = 71)
  (new_avg : (M - 71) / (n - 4) = 44) :
  n = 21 := sorry

end number_of_pupils_l266_26639


namespace kris_fraction_l266_26633

-- Definitions based on problem conditions
def Trey (kris : ℕ) := 7 * kris
def Kristen := 12
def Trey_kristen_diff := 9
def Kris_fraction_to_Kristen (kris : ℕ) : ℚ := kris / Kristen

-- Theorem statement: Proving the required fraction
theorem kris_fraction (kris : ℕ) (h1 : Trey kris = Kristen + Trey_kristen_diff) : 
  Kris_fraction_to_Kristen kris = 1 / 4 :=
by
  sorry

end kris_fraction_l266_26633


namespace problem_solution_l266_26656

theorem problem_solution (x y z : ℝ) (h1 : 2 * x - y - 2 * z - 6 = 0) (h2 : x^2 + y^2 + z^2 ≤ 4) :
  2 * x + y + z = 2 / 3 := 
by 
  sorry

end problem_solution_l266_26656


namespace PetyaWinsAgainstSasha_l266_26690

def MatchesPlayed (name : String) : Nat :=
if name = "Petya" then 12 else if name = "Sasha" then 7 else if name = "Misha" then 11 else 0

def TotalGames : Nat := 15

def GamesMissed (name : String) : Nat :=
if name = "Petya" then TotalGames - MatchesPlayed name else 
if name = "Sasha" then TotalGames - MatchesPlayed name else
if name = "Misha" then TotalGames - MatchesPlayed name else 0

def CanNotMissConsecutiveGames : Prop := True

theorem PetyaWinsAgainstSasha : (GamesMissed "Misha" = 4) ∧ CanNotMissConsecutiveGames → 
  ∃ (winsByPetya : Nat), winsByPetya = 4 :=
by
  sorry

end PetyaWinsAgainstSasha_l266_26690


namespace solution_set_inequality_l266_26645

theorem solution_set_inequality {a b : ℝ} 
  (h₁ : {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ax^2 - bx + 2 < 0}) : a + b = -2 :=
by
  sorry

end solution_set_inequality_l266_26645


namespace inequality_proof_l266_26601

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l266_26601


namespace total_crayons_lost_or_given_away_l266_26616

def crayons_given_away : ℕ := 52
def crayons_lost : ℕ := 535

theorem total_crayons_lost_or_given_away :
  crayons_given_away + crayons_lost = 587 :=
by
  sorry

end total_crayons_lost_or_given_away_l266_26616


namespace tangency_point_l266_26673

theorem tangency_point (x y : ℝ) : 
  y = x ^ 2 + 20 * x + 70 ∧ x = y ^ 2 + 70 * y + 1225 →
  (x, y) = (-19 / 2, -69 / 2) :=
by {
  sorry
}

end tangency_point_l266_26673


namespace cherry_sodas_l266_26684

theorem cherry_sodas (C O : ℕ) (h1 : O = 2 * C) (h2 : C + O = 24) : C = 8 :=
by sorry

end cherry_sodas_l266_26684


namespace product_of_two_numbers_l266_26638

theorem product_of_two_numbers (x y : ℕ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 221) : x * y = 60 := sorry

end product_of_two_numbers_l266_26638


namespace min_value_function_l266_26649

theorem min_value_function (x : ℝ) (h : 1 < x) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y ≥ 3) :=
sorry

end min_value_function_l266_26649


namespace pages_removed_iff_original_pages_l266_26657

def booklet_sum (n r : ℕ) : ℕ :=
  (n * (2 * n + 1)) - (4 * r - 1)

theorem pages_removed_iff_original_pages (n r : ℕ) :
  booklet_sum n r = 963 ↔ (2 * n = 44 ∧ (2 * r - 1, 2 * r) = (13, 14)) :=
sorry

end pages_removed_iff_original_pages_l266_26657


namespace speed_of_second_cyclist_l266_26696

theorem speed_of_second_cyclist (v : ℝ) 
  (circumference : ℝ) 
  (time : ℝ) 
  (speed_first_cyclist : ℝ)
  (meet_time : ℝ)
  (circ_full: circumference = 300) 
  (time_full: time = 20)
  (speed_first: speed_first_cyclist = 7)
  (meet_full: meet_time = time):

  v = 8 := 
by
  sorry

end speed_of_second_cyclist_l266_26696


namespace small_slices_sold_l266_26620

theorem small_slices_sold (S L : ℕ) 
  (h1 : S + L = 5000) 
  (h2 : 150 * S + 250 * L = 1050000) : 
  S = 2000 :=
by
  sorry

end small_slices_sold_l266_26620


namespace ratio_of_roots_l266_26615

theorem ratio_of_roots (c : ℝ) :
  (∃ (x1 x2 : ℝ), 5 * x1^2 - 2 * x1 + c = 0 ∧ 5 * x2^2 - 2 * x2 + c = 0 ∧ x1 / x2 = -3 / 5) → c = -3 :=
by
  sorry

end ratio_of_roots_l266_26615


namespace range_of_a_for_solution_set_l266_26691

theorem range_of_a_for_solution_set (a : ℝ) :
  ((∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a ≤ 1)) :=
sorry

end range_of_a_for_solution_set_l266_26691


namespace range_of_a_l266_26669

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2 * a)^2 - 4 * 1 * 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end range_of_a_l266_26669


namespace smallest_number_of_packs_l266_26627

theorem smallest_number_of_packs (n b w : ℕ) (Hn : n = 13) (Hb : b = 8) (Hw : w = 17) :
  Nat.lcm (Nat.lcm n b) w = 1768 :=
by
  sorry

end smallest_number_of_packs_l266_26627


namespace last_digit_of_2_pow_2010_l266_26681

theorem last_digit_of_2_pow_2010 : (2 ^ 2010) % 10 = 4 :=
by
  sorry

end last_digit_of_2_pow_2010_l266_26681


namespace find_b_l266_26647

noncomputable def ellipse_foci (a b : ℝ) (hb : b > 0) (hab : a > b) : Prop :=
∃ (F1 F2 P : ℝ×ℝ), 
    (∃ (h : a > b), (2 * b^2 + 9 = a^2)) ∧ 
    (dist P F1 + dist P F2 = 2 * a) ∧ 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (2 * 4 * (a^2 - b^2) = 36)

theorem find_b (a b : ℝ) (hb : b > 0) (hab : a > b) : 
    ellipse_foci a b hb hab → b = 3 :=
by
  sorry

end find_b_l266_26647


namespace necessary_but_not_sufficient_for_parallel_lines_l266_26650

theorem necessary_but_not_sufficient_for_parallel_lines (m : ℝ) : 
  (m = -1/2 ∨ m = 0) ↔ (∀ x y : ℝ, (x + 2*m*y - 1 = 0 ∧ (3*m + 1)*x - m*y - 1 = 0) → false) :=
sorry

end necessary_but_not_sufficient_for_parallel_lines_l266_26650


namespace remainder_zero_when_divided_by_condition_l266_26698

noncomputable def remainder_problem (x : ℂ) : ℂ :=
  (2 * x^5 - x^4 + x^2 - 1) * (x^3 - 1)

theorem remainder_zero_when_divided_by_condition (x : ℂ) (h : x^2 - x + 1 = 0) :
  remainder_problem x % (x^2 - x + 1) = 0 := by
  sorry

end remainder_zero_when_divided_by_condition_l266_26698


namespace probability_is_seven_fifteenths_l266_26610

-- Define the problem conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def yellow_apples : ℕ := 2
def choose_3_from_10 : ℕ := Nat.choose 10 3
def choose_3_red : ℕ := Nat.choose 5 3
def choose_3_green : ℕ := Nat.choose 3 3
def choose_2_red_1_green : ℕ := Nat.choose 5 2 * Nat.choose 3 1
def choose_2_green_1_red : ℕ := Nat.choose 3 2 * Nat.choose 5 1

-- Calculate favorable outcomes
def favorable_outcomes : ℕ :=
  choose_3_red + choose_3_green + choose_2_red_1_green + choose_2_green_1_red

-- Calculate the required probability
def probability_all_red_or_green : ℚ := favorable_outcomes / choose_3_from_10

-- Prove that probability_all_red_or_green is 7/15
theorem probability_is_seven_fifteenths :
  probability_all_red_or_green = 7 / 15 :=
by 
  -- Leaving the proof as a sorry for now
  sorry

end probability_is_seven_fifteenths_l266_26610


namespace total_amount_l266_26682

theorem total_amount (P Q R : ℝ) (h1 : R = 2 / 3 * (P + Q)) (h2 : R = 3200) : P + Q + R = 8000 := 
by
  sorry

end total_amount_l266_26682


namespace min_n_for_constant_term_l266_26603

theorem min_n_for_constant_term (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), (2 * n = 5 * r) → n = 5 :=
by
  sorry

end min_n_for_constant_term_l266_26603


namespace pedestrians_speed_ratio_l266_26666

-- Definitions based on conditions
variable (v v1 v2 : ℝ)

-- Conditions
def first_meeting (v1 v : ℝ) := (1 / 3) * v1 = (1 / 4) * v
def second_meeting (v2 v : ℝ) := (5 / 12) * v2 = (1 / 6) * v

-- Theorem Statement
theorem pedestrians_speed_ratio (h1 : first_meeting v1 v) (h2 : second_meeting v2 v) : v1 / v2 = 15 / 8 :=
by
  -- Proof will go here
  sorry

end pedestrians_speed_ratio_l266_26666


namespace problem_statement_l266_26607

-- Define the function
def f (x : ℝ) := -2 * x^2

-- We need to show that f is monotonically decreasing and even on (0, +∞)
theorem problem_statement : (∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x) ∧ (∀ x : ℝ, f (-x) = f x) := 
by {
  sorry -- proof goes here
}

end problem_statement_l266_26607


namespace sequence_sum_problem_l266_26626

theorem sequence_sum_problem (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - n) :
  (2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) : ℚ) = 30 / 31 := 
sorry

end sequence_sum_problem_l266_26626


namespace find_sum_of_cubes_l266_26640

noncomputable def roots (a b c : ℝ) : Prop :=
  5 * a^3 + 2014 * a + 4027 = 0 ∧ 
  5 * b^3 + 2014 * b + 4027 = 0 ∧ 
  5 * c^3 + 2014 * c + 4027 = 0

theorem find_sum_of_cubes (a b c : ℝ) (h : roots a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 :=
sorry

end find_sum_of_cubes_l266_26640


namespace parabola_bisects_rectangle_l266_26625
open Real

theorem parabola_bisects_rectangle (a : ℝ) (h_pos : a > 0) : 
  ((a^3 + a) / 2 = (a^3 / 3 + a)) → a = sqrt 3 := by
  sorry

end parabola_bisects_rectangle_l266_26625


namespace determine_q_l266_26605

theorem determine_q (q : ℕ) (h : 81^10 = 3^q) : q = 40 :=
by
  sorry

end determine_q_l266_26605


namespace intersecting_lines_c_plus_d_l266_26665

theorem intersecting_lines_c_plus_d (c d : ℝ) 
  (h1 : ∀ y, ∃ x, x = (1/3) * y + c) 
  (h2 : ∀ x, ∃ y, y = (1/3) * x + d)
  (P : (3:ℝ) = (1 / 3) * (3:ℝ) + c) 
  (Q : (3:ℝ) = (1 / 3) * (3:ℝ) + d) : 
  c + d = 4 := 
by
  sorry

end intersecting_lines_c_plus_d_l266_26665


namespace difference_of_bases_l266_26634

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * (8^5) + 4 * (8^4) + 3 * (8^3) + 2 * (8^2) + 1 * (8^1) + 0 * (8^0)

def base5_to_base10 (n : ℕ) : ℕ :=
  4 * (5^4) + 3 * (5^3) + 2 * (5^2) + 1 * (5^1) + 0 * (5^0)

theorem difference_of_bases : 
  base8_to_base10 543210 - base5_to_base10 43210 = 177966 :=
by
  sorry

end difference_of_bases_l266_26634


namespace green_light_probability_l266_26679

-- Define the durations of the red, green, and yellow lights
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

-- Define the total cycle time
def total_cycle_time : ℕ := red_light_duration + green_light_duration + yellow_light_duration

-- Define the expected probability
def expected_probability : ℚ := 5 / 12

-- Prove the probability of seeing a green light equals the expected_probability
theorem green_light_probability :
  (green_light_duration : ℚ) / (total_cycle_time : ℚ) = expected_probability :=
by
  sorry

end green_light_probability_l266_26679


namespace area_of_given_sector_l266_26655

noncomputable def area_of_sector (alpha l : ℝ) : ℝ :=
  let r := l / alpha
  (1 / 2) * l * r

theorem area_of_given_sector :
  let alpha := Real.pi / 9
  let l := Real.pi / 3
  area_of_sector alpha l = Real.pi / 2 :=
by
  sorry

end area_of_given_sector_l266_26655


namespace jake_car_washes_l266_26632

theorem jake_car_washes :
  ∀ (washes_per_bottle cost_per_bottle total_spent weekly_washes : ℕ),
  washes_per_bottle = 4 →
  cost_per_bottle = 4 →
  total_spent = 20 →
  weekly_washes = 1 →
  (total_spent / cost_per_bottle) * washes_per_bottle / weekly_washes = 20 :=
by
  intros washes_per_bottle cost_per_bottle total_spent weekly_washes
  sorry

end jake_car_washes_l266_26632


namespace three_lines_intersect_single_point_l266_26614

theorem three_lines_intersect_single_point (a : ℝ) :
  (∀ x y : ℝ, (x + 2*y + a) * (x^2 - y^2) = 0) ↔ a = 0 := by
  sorry

end three_lines_intersect_single_point_l266_26614


namespace johns_chore_homework_time_l266_26623

-- Definitions based on problem conditions
def cartoons_time : ℕ := 150  -- John's cartoon watching time in minutes
def chores_homework_per_10 : ℕ := 13  -- 13 minutes combined chores and homework per 10 minutes of cartoons
def cartoon_period : ℕ := 10  -- Per 10 minutes period

-- Theorem statement
theorem johns_chore_homework_time :
  cartoons_time / cartoon_period * chores_homework_per_10 = 195 :=
by sorry

end johns_chore_homework_time_l266_26623


namespace complex_props_hold_l266_26606

theorem complex_props_hold (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ (a^2 = a*b → a = b) :=
by
  sorry

end complex_props_hold_l266_26606


namespace total_students_l266_26652

-- Definitions based on problem conditions
def H := 36
def S := 32
def union_H_S := 59
def history_not_statistics := 27

-- The proof statement
theorem total_students : H + S - (H - history_not_statistics) = union_H_S :=
by sorry

end total_students_l266_26652


namespace find_abs_ab_l266_26685

def ellipse_foci_distance := 5
def hyperbola_foci_distance := 7

def ellipse_condition (a b : ℝ) := b^2 - a^2 = ellipse_foci_distance^2
def hyperbola_condition (a b : ℝ) := a^2 + b^2 = hyperbola_foci_distance^2

theorem find_abs_ab (a b : ℝ) (h_ellipse : ellipse_condition a b) (h_hyperbola : hyperbola_condition a b) :
  |a * b| = 2 * Real.sqrt 111 :=
by
  sorry

end find_abs_ab_l266_26685


namespace averageSpeed_is_45_l266_26699

/-- Define the upstream and downstream speeds of the fish --/
def fishA_upstream_speed := 40
def fishA_downstream_speed := 60
def fishB_upstream_speed := 30
def fishB_downstream_speed := 50
def fishC_upstream_speed := 45
def fishC_downstream_speed := 65
def fishD_upstream_speed := 35
def fishD_downstream_speed := 55
def fishE_upstream_speed := 25
def fishE_downstream_speed := 45

/-- Define a function to calculate the speed in still water --/
def stillWaterSpeed (upstream_speed : ℕ) (downstream_speed : ℕ) : ℕ :=
  (upstream_speed + downstream_speed) / 2

/-- Calculate the still water speed for each fish --/
def fishA_speed := stillWaterSpeed fishA_upstream_speed fishA_downstream_speed
def fishB_speed := stillWaterSpeed fishB_upstream_speed fishB_downstream_speed
def fishC_speed := stillWaterSpeed fishC_upstream_speed fishC_downstream_speed
def fishD_speed := stillWaterSpeed fishD_upstream_speed fishD_downstream_speed
def fishE_speed := stillWaterSpeed fishE_upstream_speed fishE_downstream_speed

/-- Calculate the average speed of all fish in still water --/
def averageSpeedInStillWater :=
  (fishA_speed + fishB_speed + fishC_speed + fishD_speed + fishE_speed) / 5

/-- The statement to prove --/
theorem averageSpeed_is_45 : averageSpeedInStillWater = 45 :=
  sorry

end averageSpeed_is_45_l266_26699


namespace sum_of_n_plus_k_l266_26644

theorem sum_of_n_plus_k (n k : ℕ) (h1 : 2 * (n - k) = 3 * (k + 1)) (h2 : 3 * (n - k - 1) = 4 * (k + 2)) : n + k = 47 := by
  sorry

end sum_of_n_plus_k_l266_26644


namespace earrings_ratio_l266_26604

theorem earrings_ratio :
  ∃ (M R : ℕ), 10 = M / 4 ∧ 10 + M + R = 70 ∧ M / R = 2 := by
  sorry

end earrings_ratio_l266_26604


namespace minimum_value_proof_l266_26674

noncomputable def minimum_value : ℝ :=
  3 + 2 * Real.sqrt 2

theorem minimum_value_proof (a b : ℝ) (h_line_eq : ∀ x y : ℝ, a * x + b * y = 1)
  (h_ab_pos : a * b > 0)
  (h_center_bisect : ∃ x y : ℝ, (x - 1)^2 + (y - 2)^2 <= x^2 + y^2) :
  (1 / a + 1 / b) ≥ minimum_value :=
by
  -- Sorry placeholder for the proof
  sorry

end minimum_value_proof_l266_26674


namespace total_ways_is_13_l266_26661

-- Define the problem conditions
def num_bus_services : ℕ := 8
def num_train_services : ℕ := 3
def num_ferry_services : ℕ := 2

-- Define the total number of ways a person can travel from A to B
def total_ways : ℕ := num_bus_services + num_train_services + num_ferry_services

-- State the theorem that the total number of ways is 13
theorem total_ways_is_13 : total_ways = 13 :=
by
  -- Add a sorry placeholder for the proof
  sorry

end total_ways_is_13_l266_26661


namespace number_b_is_three_times_number_a_l266_26688

theorem number_b_is_three_times_number_a (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 :=
by
  -- This is where the proof would go
  sorry

end number_b_is_three_times_number_a_l266_26688


namespace probability_sum_of_three_dice_is_9_l266_26653

def sum_of_three_dice_is_9 : Prop :=
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 9)

theorem probability_sum_of_three_dice_is_9 : 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 → a + b + c = 9 → sum_of_three_dice_is_9) ∧ 
  (1 / 216 = 25 / 216) := 
by
  sorry

end probability_sum_of_three_dice_is_9_l266_26653


namespace range_of_a_l266_26619

theorem range_of_a (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f x) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, y = f (x - a) + f (x + a)) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l266_26619


namespace cost_of_50_roses_l266_26636

def cost_of_dozen_roses : ℝ := 24

def is_proportional (n : ℕ) (cost : ℝ) : Prop :=
  cost = (cost_of_dozen_roses / 12) * n

def has_discount (n : ℕ) : Prop :=
  n ≥ 45

theorem cost_of_50_roses :
  ∃ (cost : ℝ), is_proportional 50 cost ∧ has_discount 50 ∧ cost * 0.9 = 90 :=
by
  sorry

end cost_of_50_roses_l266_26636


namespace number_of_adults_in_sleeper_class_l266_26683

-- Number of passengers in the train
def total_passengers : ℕ := 320

-- Percentage of passengers who are adults
def percentage_adults : ℚ := 75 / 100

-- Percentage of adults who are in the sleeper class
def percentage_adults_sleeper_class : ℚ := 15 / 100

-- Mathematical statement to prove
theorem number_of_adults_in_sleeper_class :
  (total_passengers * percentage_adults * percentage_adults_sleeper_class) = 36 :=
by
  sorry

end number_of_adults_in_sleeper_class_l266_26683
