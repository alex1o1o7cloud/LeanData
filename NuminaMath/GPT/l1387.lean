import Mathlib

namespace ratio_of_A_to_B_l1387_138725

theorem ratio_of_A_to_B (v_A v_B : ℝ) (d_A d_B : ℝ) (h1 : d_A = 128) (h2 : d_B = 64) (h3 : d_A / v_A = d_B / v_B) : v_A / v_B = 2 := 
by
  sorry

end ratio_of_A_to_B_l1387_138725


namespace fraction_unchanged_when_increased_by_ten_l1387_138785

variable {x y : ℝ}

theorem fraction_unchanged_when_increased_by_ten (x y : ℝ) :
  (5 * (10 * x)) / (10 * x + 10 * y) = 5 * x / (x + y) :=
by
  sorry

end fraction_unchanged_when_increased_by_ten_l1387_138785


namespace condition_swap_l1387_138750

variable {p q : Prop}

theorem condition_swap (h : ¬ p → q) (nh : ¬ (¬ p ↔ q)) : (p → ¬ q) ∧ ¬ (¬ (p ↔ ¬ q)) :=
by
  sorry

end condition_swap_l1387_138750


namespace cube_painting_probability_l1387_138723

theorem cube_painting_probability :
  let total_configurations := 2^6 * 2^6
  let identical_configurations := 90
  (identical_configurations / total_configurations : ℚ) = 45 / 2048 :=
by
  sorry

end cube_painting_probability_l1387_138723


namespace price_increase_after_reduction_l1387_138794

theorem price_increase_after_reduction (P : ℝ) (h : P > 0) : 
  let reduced_price := P * 0.85
  let increase_factor := 1 / 0.85
  let percentage_increase := (increase_factor - 1) * 100
  percentage_increase = 17.65 := by
  sorry

end price_increase_after_reduction_l1387_138794


namespace find_p_over_q_at_0_l1387_138730

noncomputable def p (x : ℝ) := 3 * (x - 4) * (x - 1)
noncomputable def q (x : ℝ) := (x + 3) * (x - 1) * (x - 4)

theorem find_p_over_q_at_0 : (p 0) / (q 0) = 1 := 
by
  sorry

end find_p_over_q_at_0_l1387_138730


namespace probability_of_different_colors_is_correct_l1387_138708

noncomputable def probability_different_colors : ℚ :=
  let total_chips := 18
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let p_blue_then_not_blue := (blue_chips / total_chips) * ((red_chips + yellow_chips + green_chips) / total_chips)
  let p_red_then_not_red := (red_chips / total_chips) * ((blue_chips + yellow_chips + green_chips) / total_chips)
  let p_yellow_then_not_yellow := (yellow_chips / total_chips) * ((blue_chips + red_chips + green_chips) / total_chips)
  let p_green_then_not_green := (green_chips / total_chips) * ((blue_chips + red_chips + yellow_chips) / total_chips)
  p_blue_then_not_blue + p_red_then_not_red + p_yellow_then_not_yellow + p_green_then_not_green

theorem probability_of_different_colors_is_correct :
  probability_different_colors = 119 / 162 :=
by
  sorry

end probability_of_different_colors_is_correct_l1387_138708


namespace min_ab_l1387_138704

theorem min_ab (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_eq : a + b + 3 = a * b) : 9 ≤ a * b :=
sorry

end min_ab_l1387_138704


namespace solveCubicEquation_l1387_138738

-- Define the condition as a hypothesis
def equationCondition (x : ℝ) : Prop := (7 - x)^(1/3) = -5/3

-- State the theorem to be proved
theorem solveCubicEquation : ∃ x : ℝ, equationCondition x ∧ x = 314 / 27 :=
by 
  sorry

end solveCubicEquation_l1387_138738


namespace initial_butterfly_count_l1387_138715

theorem initial_butterfly_count (n : ℕ) (h : (2 / 3 : ℚ) * n = 6) : n = 9 :=
sorry

end initial_butterfly_count_l1387_138715


namespace total_tweets_correct_l1387_138764

-- Define the rates at which Polly tweets under different conditions
def happy_rate : ℕ := 18
def hungry_rate : ℕ := 4
def mirror_rate : ℕ := 45

-- Define the durations of each activity
def happy_duration : ℕ := 20
def hungry_duration : ℕ := 20
def mirror_duration : ℕ := 20

-- Compute the total number of tweets
def total_tweets : ℕ := happy_rate * happy_duration + hungry_rate * hungry_duration + mirror_rate * mirror_duration

-- Statement to prove
theorem total_tweets_correct : total_tweets = 1340 := by
  sorry

end total_tweets_correct_l1387_138764


namespace least_possible_number_of_straight_lines_l1387_138742

theorem least_possible_number_of_straight_lines :
  ∀ (segments : Fin 31 → (Fin 2 → ℝ)), 
  (∀ i j, i ≠ j → (segments i 0 = segments j 0) ∧ (segments i 1 = segments j 1) → false) →
  ∃ (lines_count : ℕ), lines_count = 16 :=
by
  sorry

end least_possible_number_of_straight_lines_l1387_138742


namespace smallest_positive_omega_l1387_138775

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * (x + Real.pi / 4) - Real.pi / 6)

theorem smallest_positive_omega (ω : ℝ) :
  (∀ x : ℝ, g (ω) x = g (ω) (-x)) → (ω = 4 / 3) := sorry

end smallest_positive_omega_l1387_138775


namespace shells_needed_l1387_138795

theorem shells_needed (current_shells : ℕ) (total_shells : ℕ) (difference : ℕ) :
  current_shells = 5 → total_shells = 17 → difference = total_shells - current_shells → difference = 12 :=
by
  intros h1 h2 h3
  sorry

end shells_needed_l1387_138795


namespace polygon_sides_l1387_138728

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l1387_138728


namespace new_salary_after_increase_l1387_138799

theorem new_salary_after_increase : 
  ∀ (previous_salary : ℝ) (percentage_increase : ℝ), 
    previous_salary = 2000 → percentage_increase = 0.05 → 
    previous_salary + (previous_salary * percentage_increase) = 2100 :=
by
  intros previous_salary percentage_increase h1 h2
  sorry

end new_salary_after_increase_l1387_138799


namespace negation_of_proposition_l1387_138793

-- Definitions and conditions from the problem
def original_proposition (x : ℝ) : Prop := x^3 - x^2 + 1 > 0

-- The proof problem: Prove the negation
theorem negation_of_proposition : (¬ ∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, ¬original_proposition x := 
by
  -- here we insert our proof later
  sorry

end negation_of_proposition_l1387_138793


namespace measure_of_third_angle_l1387_138772

-- Definitions based on given conditions
def angle_sum_of_triangle := 180
def angle1 := 30
def angle2 := 60

-- Problem Statement: Prove the third angle (angle3) in a triangle is 90 degrees
theorem measure_of_third_angle (angle_sum : ℕ := angle_sum_of_triangle) 
  (a1 : ℕ := angle1) (a2 : ℕ := angle2) : (angle_sum - (a1 + a2)) = 90 :=
by
  sorry

end measure_of_third_angle_l1387_138772


namespace solution_l1387_138749

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (4 + 2 * x) / (6 + 3 * x) = (3 + 2 * x) / (5 + 3 * x) ∧ x = -2

theorem solution : problem_statement :=
by
  sorry

end solution_l1387_138749


namespace complex_norm_solution_l1387_138733

noncomputable def complex_norm (z : Complex) : Real :=
  Complex.abs z

theorem complex_norm_solution (w z : Complex) 
  (wz_condition : w * z = 24 - 10 * Complex.I)
  (w_norm_condition : complex_norm w = Real.sqrt 29) :
  complex_norm z = (26 * Real.sqrt 29) / 29 :=
by
  sorry

end complex_norm_solution_l1387_138733


namespace onions_shelf_correct_l1387_138759

def onions_on_shelf (initial: ℕ) (sold: ℕ) (added: ℕ) (given_away: ℕ): ℕ :=
  initial - sold + added - given_away

theorem onions_shelf_correct :
  onions_on_shelf 98 65 20 10 = 43 :=
by
  sorry

end onions_shelf_correct_l1387_138759


namespace largest_quantity_l1387_138734

noncomputable def A := (2006 / 2005) + (2006 / 2007)
noncomputable def B := (2006 / 2007) + (2008 / 2007)
noncomputable def C := (2007 / 2006) + (2007 / 2008)

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l1387_138734


namespace MrKishoreSavings_l1387_138777

noncomputable def TotalExpenses : ℕ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

noncomputable def MonthlySalary : ℕ :=
  (TotalExpenses * 10) / 9

noncomputable def Savings : ℕ :=
  (MonthlySalary * 1) / 10

theorem MrKishoreSavings :
  Savings = 2300 :=
by
  sorry

end MrKishoreSavings_l1387_138777


namespace tangent_line_to_circle_l1387_138711

theorem tangent_line_to_circle (a : ℝ) :
  (∃ k : ℝ, k = a ∧ (∀ x y : ℝ, y = x + 4 → (x - k)^2 + (y - 3)^2 = 8)) ↔ (a = 3 ∨ a = -5) := by
  sorry

end tangent_line_to_circle_l1387_138711


namespace lollipops_per_day_l1387_138737

variable (Alison_lollipops : ℕ) (Henry_lollipops : ℕ) (Diane_lollipops : ℕ) (Total_lollipops : ℕ) (Days : ℕ)

-- Conditions given in the problem
axiom condition1 : Alison_lollipops = 60
axiom condition2 : Henry_lollipops = Alison_lollipops + 30
axiom condition3 : Alison_lollipops = Diane_lollipops / 2
axiom condition4 : Total_lollipops = Alison_lollipops + Henry_lollipops + Diane_lollipops
axiom condition5 : Days = 6

-- Question to prove
theorem lollipops_per_day : (Total_lollipops / Days) = 45 := sorry

end lollipops_per_day_l1387_138737


namespace evaluate_Y_l1387_138798

def Y (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2 + 3

theorem evaluate_Y : Y 2 5 = 2 :=
by
  sorry

end evaluate_Y_l1387_138798


namespace scientific_notation_of_3933_billion_l1387_138727

-- Definitions and conditions
def is_scientific_notation (a : ℝ) (n : ℤ) :=
  1 ≤ |a| ∧ |a| < 10 ∧ (39.33 * 10^9 = a * 10^n)

-- Theorem (statement only)
theorem scientific_notation_of_3933_billion : 
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ a = 3.933 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_3933_billion_l1387_138727


namespace jim_gold_per_hour_l1387_138784

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end jim_gold_per_hour_l1387_138784


namespace cheolsu_weight_l1387_138766

variable (C M : ℝ)

theorem cheolsu_weight:
  (C = (2/3) * M) →
  (C + 72 = 2 * M) →
  C = 36 :=
by
  intros h1 h2
  sorry

end cheolsu_weight_l1387_138766


namespace minimum_toothpicks_to_remove_l1387_138768

-- Definitions related to the problem statement
def total_toothpicks : Nat := 40
def initial_triangles : Nat := 36

-- Ensure that the minimal number of toothpicks to be removed to destroy all triangles is correct.
theorem minimum_toothpicks_to_remove : ∃ (n : Nat), n = 15 ∧ (∀ (t : Nat), t ≤ total_toothpicks - n → t = 0) :=
sorry

end minimum_toothpicks_to_remove_l1387_138768


namespace inequality_proof_l1387_138765

theorem inequality_proof (a b t : ℝ) (h₀ : 0 < t) (h₁ : t < 1) (h₂ : a * b > 0) : 
  (a^2 / t^3) + (b^2 / (1 - t^3)) ≥ (a + b)^2 :=
by
  sorry

end inequality_proof_l1387_138765


namespace solve_abs_equation_l1387_138702

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l1387_138702


namespace symmetric_line_equation_l1387_138792

theorem symmetric_line_equation (x y : ℝ) (h₁ : x + y + 1 = 0) : (2 - x) + (4 - y) - 7 = 0 :=
by
  sorry

end symmetric_line_equation_l1387_138792


namespace taxi_fare_distance_l1387_138776

theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (initial_distance : ℝ) (total_fare : ℝ) : 
  initial_fare = 2.0 →
  subsequent_fare = 0.60 →
  initial_distance = 1 / 5 →
  total_fare = 25.4 →
  ∃ d : ℝ, d = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end taxi_fare_distance_l1387_138776


namespace probability_of_circle_l1387_138756

theorem probability_of_circle :
  let numCircles := 4
  let numSquares := 3
  let numTriangles := 3
  let totalFigures := numCircles + numSquares + numTriangles
  let probability := numCircles / totalFigures
  probability = 2 / 5 :=
by
  sorry

end probability_of_circle_l1387_138756


namespace arithmetic_sequence_ratio_l1387_138722

theorem arithmetic_sequence_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (1/2) * n * (2 * a 1 + (n-1) * d))
  (h2 : ∀ n, T n = (1/2) * n * (2 * b 1 + (n-1) * d'))
  (h3 : ∀ n, S n / T n = 7*n / (n + 3)): a 5 / b 5 = 21 / 4 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l1387_138722


namespace jenny_run_distance_l1387_138788

theorem jenny_run_distance (walk_distance : ℝ) (ran_walk_diff : ℝ) (h_walk : walk_distance = 0.4) (h_diff : ran_walk_diff = 0.2) :
  (walk_distance + ran_walk_diff) = 0.6 :=
sorry

end jenny_run_distance_l1387_138788


namespace pirate_treasure_probability_l1387_138754

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_traps := 1 / 10
  let p_neither := 7 / 10
  let num_islands := 8
  let num_treasure := 4
  binomial num_islands num_treasure * p_treasure^num_treasure * p_neither^(num_islands - num_treasure) = 673 / 25000 :=
by
  sorry

end pirate_treasure_probability_l1387_138754


namespace liquid_level_ratio_l1387_138797

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l1387_138797


namespace smallest_n_divisible_by_2009_l1387_138779

theorem smallest_n_divisible_by_2009 : ∃ n : ℕ, n > 1 ∧ (n^2 * (n - 1)) % 2009 = 0 ∧ (∀ m : ℕ, m > 1 → (m^2 * (m - 1)) % 2009 = 0 → m ≥ n) :=
by
  sorry

end smallest_n_divisible_by_2009_l1387_138779


namespace machines_work_together_l1387_138717

theorem machines_work_together (x : ℝ) (h_pos : 0 < x) :
  (1 / (x + 2) + 1 / (x + 3) + 1 / (x + 1) = 1 / x) → x = 1 :=
by
  sorry

end machines_work_together_l1387_138717


namespace train_length_l1387_138752

theorem train_length (L : ℝ) (h1 : (L + 120) / 60 = L / 20) : L = 60 := 
sorry

end train_length_l1387_138752


namespace distance_blown_by_storm_l1387_138740

-- Definitions based on conditions
def speed : ℤ := 30
def time_travelled : ℤ := 20
def distance_travelled := speed * time_travelled
def total_distance := 2 * distance_travelled
def fractional_distance_left := total_distance / 3

-- Final statement to prove
theorem distance_blown_by_storm : distance_travelled - fractional_distance_left = 200 := by
  sorry

end distance_blown_by_storm_l1387_138740


namespace bridget_apples_l1387_138767

variable (x : ℕ)

-- Conditions as definitions
def apples_after_splitting : ℕ := x / 2
def apples_after_giving_to_cassie : ℕ := apples_after_splitting x - 5
def apples_after_finding_hidden : ℕ := apples_after_giving_to_cassie x + 2
def final_apples : ℕ := apples_after_finding_hidden x
def bridget_keeps : ℕ := 6

-- Proof statement
theorem bridget_apples : x / 2 - 5 + 2 = bridget_keeps → x = 18 := by
  intros h
  sorry

end bridget_apples_l1387_138767


namespace percentage_four_petals_l1387_138709

def total_clovers : ℝ := 200
def percentage_three_petals : ℝ := 0.75
def percentage_two_petals : ℝ := 0.24
def earnings : ℝ := 554 -- cents

theorem percentage_four_petals :
  (total_clovers - (percentage_three_petals * total_clovers + percentage_two_petals * total_clovers)) / total_clovers * 100 = 1 := 
by sorry

end percentage_four_petals_l1387_138709


namespace parabola_tangent_perp_l1387_138718

theorem parabola_tangent_perp (a b : ℝ) : 
  (∃ x y : ℝ, x^2 = 4 * y ∧ y = a ∧ b ≠ 0 ∧ x ≠ 0) ∧
  (∃ x' y' : ℝ, x'^2 = 4 * y' ∧ y' = b ∧ a ≠ 0 ∧ x' ≠ 0) ∧
  (a * b = -1) 
  → a^4 * b^4 = (a^2 + b^2)^3 :=
by
  sorry

end parabola_tangent_perp_l1387_138718


namespace norma_bananas_count_l1387_138739

-- Definitions for the conditions
def initial_bananas : ℕ := 47
def lost_bananas : ℕ := 45

-- The proof problem in Lean 4 statement
theorem norma_bananas_count : initial_bananas - lost_bananas = 2 := by
  -- Proof is omitted
  sorry

end norma_bananas_count_l1387_138739


namespace savings_l1387_138700

def distance_each_way : ℕ := 150
def round_trip_distance : ℕ := 2 * distance_each_way
def rental_cost_first_option : ℕ := 50
def rental_cost_second_option : ℕ := 90
def gasoline_efficiency : ℕ := 15
def gasoline_cost_per_liter : ℚ := 0.90
def gasoline_needed_for_trip : ℚ := round_trip_distance / gasoline_efficiency
def total_gasoline_cost : ℚ := gasoline_needed_for_trip * gasoline_cost_per_liter
def total_cost_first_option : ℚ := rental_cost_first_option + total_gasoline_cost
def total_cost_second_option : ℚ := rental_cost_second_option

theorem savings : total_cost_second_option - total_cost_first_option = 22 := by
  sorry

end savings_l1387_138700


namespace percent_decrease_internet_cost_l1387_138729

theorem percent_decrease_internet_cost :
  ∀ (initial_cost final_cost : ℝ), initial_cost = 120 → final_cost = 45 → 
  ((initial_cost - final_cost) / initial_cost) * 100 = 62.5 :=
by
  intros initial_cost final_cost h_initial h_final
  sorry

end percent_decrease_internet_cost_l1387_138729


namespace problem1_problem2_l1387_138753

-- Proof for Problem 1
theorem problem1 : (99^2 + 202*99 + 101^2) = 40000 := 
by {
  -- proof
  sorry
}

-- Proof for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : ((1 / (x - 1) - 2) / ((2 * x - 3) / (x^2 - 1))) = -x - 1 :=
by {
  -- proof
  sorry
}

end problem1_problem2_l1387_138753


namespace range_of_a_l1387_138716

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → -3 * x^2 + a ≤ 0) ↔ a ≤ 3 := by
  sorry

end range_of_a_l1387_138716


namespace math_problem_proof_l1387_138762

theorem math_problem_proof : 
  ((9 - 8 + 7) ^ 2 * 6 + 5 - 4 ^ 2 * 3 + 2 ^ 3 - 1) = 347 := 
by sorry

end math_problem_proof_l1387_138762


namespace contrapositive_proof_l1387_138781

theorem contrapositive_proof (x : ℝ) : (x^2 < 1 → -1 < x ∧ x < 1) → (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by
  sorry

end contrapositive_proof_l1387_138781


namespace probability_of_multiple_6_or_8_l1387_138707

def is_probability_of_multiple_6_or_8 (n : ℕ) : Prop := 
  let num_multiples (k : ℕ) := n / k
  let multiples_6 := num_multiples 6
  let multiples_8 := num_multiples 8
  let multiples_24 := num_multiples 24
  let total_multiples := multiples_6 + multiples_8 - multiples_24
  total_multiples / n = 1 / 4

theorem probability_of_multiple_6_or_8 : is_probability_of_multiple_6_or_8 72 :=
  by sorry

end probability_of_multiple_6_or_8_l1387_138707


namespace smallest_x_l1387_138755

theorem smallest_x (y : ℤ) (h1 : 0.9 = (y : ℚ) / (151 + x)) (h2 : 0 < x) (h3 : 0 < y) : x = 9 :=
sorry

end smallest_x_l1387_138755


namespace perfect_square_expression_l1387_138706

theorem perfect_square_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 :=
by
  sorry

end perfect_square_expression_l1387_138706


namespace gcd_xyz_times_xyz_is_square_l1387_138771

theorem gcd_xyz_times_xyz_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, k^2 = Nat.gcd x (Nat.gcd y z) * x * y * z :=
by
  sorry

end gcd_xyz_times_xyz_is_square_l1387_138771


namespace polynomial_factorization_l1387_138744

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end polynomial_factorization_l1387_138744


namespace bao_interest_l1387_138732

noncomputable def initial_amount : ℝ := 1000
noncomputable def interest_rate : ℝ := 0.05
noncomputable def periods : ℕ := 6
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ periods
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem bao_interest :
  interest_earned = 340.095 := by
  sorry

end bao_interest_l1387_138732


namespace fan_airflow_weekly_l1387_138791

def fan_airflow_per_second : ℕ := 10
def fan_work_minutes_per_day : ℕ := 10
def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def days_per_week : ℕ := 7

theorem fan_airflow_weekly : 
  (fan_airflow_per_second * (minutes_to_seconds fan_work_minutes_per_day) * days_per_week) = 42000 := 
by
  sorry

end fan_airflow_weekly_l1387_138791


namespace quadratic_function_range_l1387_138786

theorem quadratic_function_range (x : ℝ) (y : ℝ) (h1 : y = x^2 - 2*x - 3) (h2 : -2 ≤ x ∧ x ≤ 2) :
  -4 ≤ y ∧ y ≤ 5 :=
sorry

end quadratic_function_range_l1387_138786


namespace triangle_inequalities_l1387_138731

open Real

-- Define a structure for a triangle with its properties
structure Triangle :=
(a b c R ra rb rc : ℝ)

-- Main statement to be proved
theorem triangle_inequalities (Δ : Triangle) (h : 2 * Δ.R ≤ Δ.ra) :
  Δ.a > Δ.b ∧ Δ.a > Δ.c ∧ 2 * Δ.R > Δ.rb ∧ 2 * Δ.R > Δ.rc :=
sorry

end triangle_inequalities_l1387_138731


namespace find_sequence_formula_l1387_138757

variable (a : ℕ → ℝ)

noncomputable def sequence_formula := ∀ n : ℕ, a n = Real.sqrt n

lemma sequence_initial : a 1 = 1 :=
sorry

lemma sequence_recursive (n : ℕ) : a (n+1)^2 - a n^2 = 1 :=
sorry

theorem find_sequence_formula : sequence_formula a :=
sorry

end find_sequence_formula_l1387_138757


namespace spend_on_video_games_l1387_138789

/-- Given the total allowance and the fractions of spending on various categories,
prove the amount spent on video games. -/
theorem spend_on_video_games (total_allowance : ℝ)
  (fraction_books fraction_snacks fraction_crafts : ℝ)
  (h_total : total_allowance = 50)
  (h_fraction_books : fraction_books = 1 / 4)
  (h_fraction_snacks : fraction_snacks = 1 / 5)
  (h_fraction_crafts : fraction_crafts = 3 / 10) :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_crafts * total_allowance) = 12.5 :=
by
  sorry

end spend_on_video_games_l1387_138789


namespace cost_per_serving_of_pie_l1387_138735

theorem cost_per_serving_of_pie 
  (w_gs : ℝ) (p_gs : ℝ) (w_gala : ℝ) (p_gala : ℝ) (w_hc : ℝ) (p_hc : ℝ)
  (pie_crust_cost : ℝ) (lemon_cost : ℝ) (butter_cost : ℝ) (servings : ℕ)
  (total_weight_gs : w_gs = 0.5) (price_gs_per_pound : p_gs = 1.80)
  (total_weight_gala : w_gala = 0.8) (price_gala_per_pound : p_gala = 2.20)
  (total_weight_hc : w_hc = 0.7) (price_hc_per_pound : p_hc = 2.50)
  (cost_pie_crust : pie_crust_cost = 2.50) (cost_lemon : lemon_cost = 0.60)
  (cost_butter : butter_cost = 1.80) (total_servings : servings = 8) :
  (w_gs * p_gs + w_gala * p_gala + w_hc * p_hc + pie_crust_cost + lemon_cost + butter_cost) / servings = 1.16 :=
by 
  sorry

end cost_per_serving_of_pie_l1387_138735


namespace shell_placements_l1387_138787

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem shell_placements : factorial 14 / 7 = 10480142147302400 := by
  sorry

end shell_placements_l1387_138787


namespace part_one_solution_set_part_two_range_of_a_l1387_138763

def f (x : ℝ) (a : ℝ) : ℝ := |x - a| - 2

theorem part_one_solution_set (a : ℝ) (h : a = 1) : { x : ℝ | f x a + |2 * x - 3| > 0 } = { x : ℝ | x > 2 ∨ x < 2 / 3 } := 
sorry

theorem part_two_range_of_a : (∃ x : ℝ, f x (a) > |x - 3|) ↔ (a < 1 ∨ a > 5) :=
sorry

end part_one_solution_set_part_two_range_of_a_l1387_138763


namespace probability_merlin_dismissed_l1387_138758

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l1387_138758


namespace increasing_interval_m_range_l1387_138778

def y (x m : ℝ) : ℝ := x^2 + 2 * m * x + 10

theorem increasing_interval_m_range (m : ℝ) : (∀ x, 2 ≤ x → ∀ x', x' ≥ x → y x m ≤ y x' m) → (-2 : ℝ) ≤ m :=
sorry

end increasing_interval_m_range_l1387_138778


namespace int_coeffs_square_sum_l1387_138746

theorem int_coeffs_square_sum (a b c d e f : ℤ)
  (h : ∀ x, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := 
sorry

end int_coeffs_square_sum_l1387_138746


namespace school_basketballs_l1387_138770

theorem school_basketballs (n_classes n_basketballs_per_class total_basketballs : ℕ)
  (h1 : n_classes = 7)
  (h2 : n_basketballs_per_class = 7)
  (h3 : total_basketballs = n_classes * n_basketballs_per_class) :
  total_basketballs = 49 :=
sorry

end school_basketballs_l1387_138770


namespace fred_spending_correct_l1387_138713

noncomputable def fred_total_spending : ℝ :=
  let football_price_each := 2.73
  let football_quantity := 2
  let football_tax_rate := 0.05
  let pokemon_price := 4.01
  let pokemon_tax_rate := 0.08
  let baseball_original_price := 10
  let baseball_discount_rate := 0.10
  let baseball_tax_rate := 0.06
  let football_total_before_tax := football_price_each * football_quantity
  let football_total_tax := football_total_before_tax * football_tax_rate
  let football_total := football_total_before_tax + football_total_tax
  let pokemon_total_tax := pokemon_price * pokemon_tax_rate
  let pokemon_total := pokemon_price + pokemon_total_tax
  let baseball_discount := baseball_original_price * baseball_discount_rate
  let baseball_discounted_price := baseball_original_price - baseball_discount
  let baseball_total_tax := baseball_discounted_price * baseball_tax_rate
  let baseball_total := baseball_discounted_price + baseball_total_tax
  football_total + pokemon_total + baseball_total

theorem fred_spending_correct :
  fred_total_spending = 19.6038 := 
  by
    sorry

end fred_spending_correct_l1387_138713


namespace patients_per_doctor_l1387_138796

theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) (h_patients : total_patients = 400) (h_doctors : total_doctors = 16) : 
  (total_patients / total_doctors) = 25 :=
by
  sorry

end patients_per_doctor_l1387_138796


namespace find_b_l1387_138741

noncomputable def general_quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_b (a c : ℝ) (y1 y2 : ℝ) :
  y1 = general_quadratic a 3 c 2 →
  y2 = general_quadratic a 3 c (-2) →
  y1 - y2 = 12 →
  3 = 3 :=
by
  intros h1 h2 h3
  sorry

end find_b_l1387_138741


namespace max_quarters_in_wallet_l1387_138782

theorem max_quarters_in_wallet:
  ∃ (q n : ℕ), 
    (30 * n) + 50 = 31 * (n + 1) ∧ 
    q = 22 :=
by
  sorry

end max_quarters_in_wallet_l1387_138782


namespace edge_length_is_correct_l1387_138747

-- Define the given conditions
def volume_material : ℕ := 12 * 18 * 6
def edge_length : ℕ := 3
def number_cubes : ℕ := 48
def volume_cube (e : ℕ) : ℕ := e * e * e

-- Problem statement in Lean:
theorem edge_length_is_correct : volume_material = number_cubes * volume_cube edge_length → edge_length = 3 :=
by
  sorry

end edge_length_is_correct_l1387_138747


namespace cost_to_paint_cube_l1387_138780

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) 
  (h1 : cost_per_kg = 40) 
  (h2 : coverage_per_kg = 20) 
  (h3 : side_length = 10) 
  : (6 * side_length^2 / coverage_per_kg) * cost_per_kg = 1200 :=
by
  sorry

end cost_to_paint_cube_l1387_138780


namespace portraits_count_l1387_138719

theorem portraits_count (P S : ℕ) (h1 : S = 6 * P) (h2 : P + S = 200) : P = 28 := 
by
  -- The proof will be here.
  sorry

end portraits_count_l1387_138719


namespace equation_of_line_l1387_138774

theorem equation_of_line (l : ℝ → ℝ) :
  (∀ (P : ℝ × ℝ), P = (4, 2) → 
    ∃ (a b : ℝ), ((P = ( (4 - a), (2 - b)) ∨ P = ( (4 + a), (2 + b))) ∧ 
    ((4 - a)^2 / 36 + (2 - b)^2 / 9 = 1) ∧ ((4 + a)^2 / 36 + (2 + b)^2 / 9 = 1)) ∧
    (P.2 = l P.1)) →
  (∀ (x y : ℝ), y = l x ↔ 2 * x + 3 * y - 16 = 0) :=
by
  intros h P hp
  sorry -- Placeholder for the proof

end equation_of_line_l1387_138774


namespace amount_after_two_years_l1387_138760

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ)
  (hP : P = 64000) (hr : r = 1 / 6) (hn : n = 2) : 
  A = P * (1 + r) ^ n := by
  sorry

end amount_after_two_years_l1387_138760


namespace like_terms_sum_l1387_138721

theorem like_terms_sum (m n : ℕ) (h1 : m = 3) (h2 : 4 = n + 2) : m + n = 5 :=
by
  sorry

end like_terms_sum_l1387_138721


namespace fourth_quadrangle_area_l1387_138726

theorem fourth_quadrangle_area (S1 S2 S3 S4 : ℝ) (h : S1 + S4 = S2 + S3) : S4 = S2 + S3 - S1 :=
by
  sorry

end fourth_quadrangle_area_l1387_138726


namespace find_dads_dimes_l1387_138720

variable (original_dimes mother_dimes total_dimes dad_dimes : ℕ)

def proof_problem (original_dimes mother_dimes total_dimes dad_dimes : ℕ) : Prop :=
  original_dimes = 7 ∧
  mother_dimes = 4 ∧
  total_dimes = 19 ∧
  total_dimes = original_dimes + mother_dimes + dad_dimes

theorem find_dads_dimes (h : proof_problem 7 4 19 8) : dad_dimes = 8 :=
sorry

end find_dads_dimes_l1387_138720


namespace joan_has_10_books_l1387_138703

def toms_books := 38
def together_books := 48
def joans_books := together_books - toms_books

theorem joan_has_10_books : joans_books = 10 :=
by
  -- The proof goes here, but we'll add "sorry" to indicate it's a placeholder.
  sorry

end joan_has_10_books_l1387_138703


namespace line_equation_l1387_138712

theorem line_equation (t : ℝ) : 
  ∃ m b, (∀ x y : ℝ, (x, y) = (3 * t + 6, 5 * t - 7) → y = m * x + b) ∧
  m = 5 / 3 ∧ b = -17 :=
by
  use 5 / 3, -17
  sorry

end line_equation_l1387_138712


namespace min_omega_condition_l1387_138710

theorem min_omega_condition :
  ∃ (ω: ℝ) (k: ℤ), (ω > 0) ∧ (ω = 6 * k + 1 / 2) ∧ (∀ (ω' : ℝ), (ω' > 0) ∧ (∃ (k': ℤ), ω' = 6 * k' + 1 / 2) → ω ≤ ω') := 
sorry

end min_omega_condition_l1387_138710


namespace salary_reduction_l1387_138705

variable (S R : ℝ) (P : ℝ)
variable (h1 : R = S * (1 - P/100))
variable (h2 : S = R * (1 + 53.84615384615385 / 100))

theorem salary_reduction : P = 35 :=
by sorry

end salary_reduction_l1387_138705


namespace greatest_value_of_NPMK_l1387_138714

def is_digit (n : ℕ) : Prop := n < 10

theorem greatest_value_of_NPMK : 
  ∃ M K N P : ℕ, is_digit M ∧ is_digit K ∧ 
  M = K + 1 ∧ M = 9 ∧ K = 8 ∧ 
  1000 * N + 100 * P + 10 * M + K = 8010 ∧ 
  (100 * M + 10 * M + K) * M = 8010 := by
  sorry

end greatest_value_of_NPMK_l1387_138714


namespace travel_agency_choice_l1387_138790

noncomputable def cost_A (x : ℕ) : ℝ :=
  350 * x + 1000

noncomputable def cost_B (x : ℕ) : ℝ :=
  400 * x + 800

theorem travel_agency_choice (x : ℕ) :
  if x < 4 then cost_A x > cost_B x
  else if x = 4 then cost_A x = cost_B x
  else cost_A x < cost_B x :=
by sorry

end travel_agency_choice_l1387_138790


namespace spending_50_dollars_l1387_138761

def receiving_money (r : Int) : Prop := r > 0

def spending_money (s : Int) : Prop := s < 0

theorem spending_50_dollars :
  receiving_money 80 ∧ ∀ r, receiving_money r → spending_money (-r)
  → spending_money (-50) :=
by
  sorry

end spending_50_dollars_l1387_138761


namespace cos_double_angle_l1387_138773

open Real

theorem cos_double_angle (α : ℝ) (h0 : 0 < α ∧ α < π) (h1 : sin α + cos α = 1 / 2) : cos (2 * α) = -sqrt 7 / 4 :=
by
  sorry

end cos_double_angle_l1387_138773


namespace find_value_of_N_l1387_138701

theorem find_value_of_N 
  (N : ℝ) 
  (h : (20 / 100) * N = (30 / 100) * 2500) 
  : N = 3750 := 
sorry

end find_value_of_N_l1387_138701


namespace frank_remaining_money_l1387_138769

noncomputable def cheapest_lamp_cost : ℝ := 20
noncomputable def most_expensive_lamp_cost : ℝ := 3 * cheapest_lamp_cost
noncomputable def frank_initial_money : ℝ := 90

theorem frank_remaining_money : frank_initial_money - most_expensive_lamp_cost = 30 := by
  -- Proof will go here
  sorry

end frank_remaining_money_l1387_138769


namespace Freddy_is_18_l1387_138736

-- Definitions based on the conditions
def Job_age : Nat := 5
def Stephanie_age : Nat := 4 * Job_age
def Freddy_age : Nat := Stephanie_age - 2

-- Statement to prove
theorem Freddy_is_18 : Freddy_age = 18 := by
  sorry

end Freddy_is_18_l1387_138736


namespace perfect_square_is_289_l1387_138743

/-- The teacher tells a three-digit perfect square number by
revealing the hundreds digit to person A, the tens digit to person B,
and the units digit to person C, and tells them that all three digits
are different from each other. Each person only knows their own digit and
not the others. The three people have the following conversation:

Person A: I don't know what the perfect square number is.  
Person B: You don't need to say; I also know that you don't know.  
Person C: I already know what the number is.  
Person A: After hearing Person C, I also know what the number is.  
Person B: After hearing Person A also knows what the number is.

Given these conditions, the three-digit perfect square number is 289. -/
theorem perfect_square_is_289:
  ∃ n : ℕ, n^2 = 289 := by
  sorry

end perfect_square_is_289_l1387_138743


namespace average_of_rest_of_class_l1387_138751

theorem average_of_rest_of_class
  (n : ℕ)
  (h1 : n > 0)
  (avg_class : ℝ := 84)
  (avg_one_fourth : ℝ := 96)
  (total_sum : ℝ := avg_class * n)
  (sum_one_fourth : ℝ := avg_one_fourth * (n / 4))
  (sum_rest : ℝ := total_sum - sum_one_fourth)
  (num_rest : ℝ := (3 * n) / 4) :
  sum_rest / num_rest = 80 :=
sorry

end average_of_rest_of_class_l1387_138751


namespace possible_values_of_p1_l1387_138745

noncomputable def p (x : ℝ) (n : ℕ) : ℝ := sorry

axiom deg_p (n : ℕ) (h : n ≥ 2) (x : ℝ) : x^n = 1

axiom roots_le_one (r : ℝ) : r ≤ 1

axiom p_at_2 (n : ℕ) (h : n ≥ 2) : p 2 n = 3^n

theorem possible_values_of_p1 (n : ℕ) (h : n ≥ 2) : p 1 n = 0 ∨ p 1 n = (-1)^n * 2^n :=
by
  sorry

end possible_values_of_p1_l1387_138745


namespace simplify_expression_l1387_138783

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression_l1387_138783


namespace square_value_l1387_138724

theorem square_value {square : ℚ} (h : 8 / 12 = square / 3) : square = 2 :=
sorry

end square_value_l1387_138724


namespace min_value_of_a_plus_2b_l1387_138748

theorem min_value_of_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b - 3) :
  a + 2 * b = 4 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_a_plus_2b_l1387_138748
