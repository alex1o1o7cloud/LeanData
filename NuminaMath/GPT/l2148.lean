import Mathlib

namespace NUMINAMATH_GPT_valid_patents_growth_l2148_214861

variable (a b : ℝ)

def annual_growth_rate : ℝ := 0.23

theorem valid_patents_growth (h1 : b = (1 + annual_growth_rate)^2 * a) : b = (1 + 0.23)^2 * a :=
by
  sorry

end NUMINAMATH_GPT_valid_patents_growth_l2148_214861


namespace NUMINAMATH_GPT_parallel_lines_cond_l2148_214833

theorem parallel_lines_cond (a c : ℝ) :
    (∀ (x y : ℝ), (a * x - 2 * y - 1 = 0) ↔ (6 * x - 4 * y + c = 0)) → 
        (a = 3 ∧ ∃ (c : ℝ), c ≠ -2) ∨ (a = 3 ∧ c = -2) := 
sorry

end NUMINAMATH_GPT_parallel_lines_cond_l2148_214833


namespace NUMINAMATH_GPT_dimensions_of_triangle_from_square_l2148_214857

theorem dimensions_of_triangle_from_square :
  ∀ (a : ℝ) (triangle : ℝ × ℝ × ℝ), 
    a = 10 →
    triangle = (a, a, a * Real.sqrt 2) →
    triangle = (10, 10, 10 * Real.sqrt 2) :=
by
  intros a triangle a_eq triangle_eq
  -- Proof
  sorry

end NUMINAMATH_GPT_dimensions_of_triangle_from_square_l2148_214857


namespace NUMINAMATH_GPT_negation_exists_real_negation_of_quadratic_l2148_214891

theorem negation_exists_real (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

def quadratic (x : ℝ) : Prop := x^2 - 2*x + 3 ≤ 0

theorem negation_of_quadratic :
  (¬ ∀ x : ℝ, quadratic x) ↔ ∃ x : ℝ, ¬ quadratic x :=
by exact negation_exists_real quadratic

end NUMINAMATH_GPT_negation_exists_real_negation_of_quadratic_l2148_214891


namespace NUMINAMATH_GPT_arrangement_count_l2148_214841

def number_of_arrangements (n : ℕ) : ℕ :=
  if n = 6 then 5 * (Nat.factorial 5) else 0

theorem arrangement_count : number_of_arrangements 6 = 600 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_l2148_214841


namespace NUMINAMATH_GPT_find_b_l2148_214874

variable (a b c : ℝ)
variable (sin cos : ℝ → ℝ)

-- Assumptions or Conditions
variables (h1 : a^2 - c^2 = 2 * b) 
variables (h2 : sin (b) = 4 * cos (a) * sin (c))

theorem find_b (h1 : a^2 - c^2 = 2 * b) (h2 : sin (b) = 4 * cos (a) * sin (c)) : b = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l2148_214874


namespace NUMINAMATH_GPT_pints_in_two_liters_l2148_214814

theorem pints_in_two_liters (p : ℝ) (h : p = 1.575 / 0.75) : 2 * p = 4.2 := 
sorry

end NUMINAMATH_GPT_pints_in_two_liters_l2148_214814


namespace NUMINAMATH_GPT_blocks_tower_l2148_214894

theorem blocks_tower (T H Total : ℕ) (h1 : H = 53) (h2 : Total = 80) (h3 : T + H = Total) : T = 27 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_blocks_tower_l2148_214894


namespace NUMINAMATH_GPT_lisa_quiz_goal_l2148_214851

theorem lisa_quiz_goal (total_quizzes earned_A_on_first earned_A_goal remaining_quizzes additional_A_needed max_quizzes_below_A : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : earned_A_on_first = 30)
  (h3 : earned_A_goal = total_quizzes * 85 / 100)
  (h4 : remaining_quizzes = total_quizzes - 40)
  (h5 : additional_A_needed = earned_A_goal - earned_A_on_first)
  (h6 : max_quizzes_below_A = remaining_quizzes - additional_A_needed):
  max_quizzes_below_A = 0 :=
by sorry

end NUMINAMATH_GPT_lisa_quiz_goal_l2148_214851


namespace NUMINAMATH_GPT_employee_count_l2148_214853

theorem employee_count (avg_salary : ℕ) (manager_salary : ℕ) (new_avg_increase : ℕ) (E : ℕ) :
  (avg_salary = 1500) ∧ (manager_salary = 4650) ∧ (new_avg_increase = 150) →
  1500 * E + 4650 = 1650 * (E + 1) → E = 20 :=
by
  sorry

end NUMINAMATH_GPT_employee_count_l2148_214853


namespace NUMINAMATH_GPT_all_rationals_in_A_l2148_214896

noncomputable def f (n : ℕ) : ℚ := (n-1)/(n+2)

def A : Set ℚ := { q | ∃ (s : Finset ℕ), q = s.sum f }

theorem all_rationals_in_A : A = Set.univ :=
by
  sorry

end NUMINAMATH_GPT_all_rationals_in_A_l2148_214896


namespace NUMINAMATH_GPT_sum_of_squares_l2148_214867

theorem sum_of_squares :
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2148_214867


namespace NUMINAMATH_GPT_jam_cost_is_162_l2148_214864

theorem jam_cost_is_162 (N B J : ℕ) (h1 : N > 1) (h2 : 4 * B + 6 * J = 39) (h3 : N = 9) : 
  6 * N * J = 162 := 
by sorry

end NUMINAMATH_GPT_jam_cost_is_162_l2148_214864


namespace NUMINAMATH_GPT_conversion_problems_l2148_214830

-- Define the conversion factors
def square_meters_to_hectares (sqm : ℕ) : ℕ := sqm / 10000
def hectares_to_square_kilometers (ha : ℕ) : ℕ := ha / 100
def square_kilometers_to_hectares (sqkm : ℕ) : ℕ := sqkm * 100

-- Define the specific values from the problem
def value1_m2 : ℕ := 5000000
def value2_km2 : ℕ := 70000

-- The theorem to prove
theorem conversion_problems :
  (square_meters_to_hectares value1_m2 = 500) ∧
  (hectares_to_square_kilometers 500 = 5) ∧
  (square_kilometers_to_hectares value2_km2 = 7000000) :=
by
  sorry

end NUMINAMATH_GPT_conversion_problems_l2148_214830


namespace NUMINAMATH_GPT_white_balls_in_bag_l2148_214835

open BigOperators

theorem white_balls_in_bag (N : ℕ) (N_green : ℕ) (N_yellow : ℕ) (N_red : ℕ) (N_purple : ℕ)
  (prob_not_red_nor_purple : ℝ) (W : ℕ)
  (hN : N = 100)
  (hN_green : N_green = 30)
  (hN_yellow : N_yellow = 10)
  (hN_red : N_red = 47)
  (hN_purple : N_purple = 3)
  (h_prob_not_red_nor_purple : prob_not_red_nor_purple = 0.5) :
  W = 10 :=
sorry

end NUMINAMATH_GPT_white_balls_in_bag_l2148_214835


namespace NUMINAMATH_GPT_element_of_set_l2148_214849

theorem element_of_set : -1 ∈ { x : ℝ | x^2 - 1 = 0 } :=
sorry

end NUMINAMATH_GPT_element_of_set_l2148_214849


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_of_f_l2148_214818

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval_of_f :
  { x : ℝ | x > Real.exp 1 } = {y : ℝ | ∀ ε > 0, (x : ℝ) → (0 < x → (f (x + ε) < f x) ∧ (f x < f (x + ε)))}
:=
sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_of_f_l2148_214818


namespace NUMINAMATH_GPT_tim_pencils_l2148_214816

-- Problem statement: If x = 2 and z = 5, then y = z - x where y is the number of pencils Tim placed.
def pencils_problem (x y z : Nat) : Prop :=
  x = 2 ∧ z = 5 → y = z - x

theorem tim_pencils : pencils_problem 2 3 5 :=
by
  sorry

end NUMINAMATH_GPT_tim_pencils_l2148_214816


namespace NUMINAMATH_GPT_jason_earned_amount_l2148_214888

theorem jason_earned_amount (init_jason money_jason : ℤ)
    (h0 : init_jason = 3)
    (h1 : money_jason = 63) :
    money_jason - init_jason = 60 := 
by
  sorry

end NUMINAMATH_GPT_jason_earned_amount_l2148_214888


namespace NUMINAMATH_GPT_repeating_decimal_computation_l2148_214837

noncomputable def x := 864 / 999
noncomputable def y := 579 / 999
noncomputable def z := 135 / 999

theorem repeating_decimal_computation :
  x - y - z = 50 / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_computation_l2148_214837


namespace NUMINAMATH_GPT_steinburg_marching_band_l2148_214865

theorem steinburg_marching_band :
  ∃ n : ℤ, n > 0 ∧ 30 * n < 1200 ∧ 30 * n % 34 = 6 ∧ 30 * n = 720 := by
  sorry

end NUMINAMATH_GPT_steinburg_marching_band_l2148_214865


namespace NUMINAMATH_GPT_old_clock_slower_l2148_214828

-- Given conditions
def old_clock_coincidence_minutes : ℕ := 66

-- Standard clock coincidences in 24 hours
def standard_clock_coincidences_in_24_hours : ℕ := 22

-- Standard 24 hours in minutes
def standard_24_hours_in_minutes : ℕ := 24 * 60

-- Total time for old clock in minutes over what should be 24 hours
def total_time_for_old_clock : ℕ := standard_clock_coincidences_in_24_hours * old_clock_coincidence_minutes

-- Problem statement: prove that the old clock's 24 hours is 12 minutes slower 
theorem old_clock_slower : total_time_for_old_clock = standard_24_hours_in_minutes + 12 := by
  sorry

end NUMINAMATH_GPT_old_clock_slower_l2148_214828


namespace NUMINAMATH_GPT_students_selected_milk_l2148_214836

theorem students_selected_milk
    (total_students : ℕ)
    (students_soda students_milk students_juice : ℕ)
    (soda_percentage : ℚ)
    (milk_percentage : ℚ)
    (juice_percentage : ℚ)
    (h1 : soda_percentage = 0.7)
    (h2 : milk_percentage = 0.2)
    (h3 : juice_percentage = 0.1)
    (h4 : students_soda = 84)
    (h5 : total_students = students_soda / soda_percentage)
    : students_milk = total_students * milk_percentage :=
by
    sorry

end NUMINAMATH_GPT_students_selected_milk_l2148_214836


namespace NUMINAMATH_GPT_exists_polynomial_degree_n_l2148_214887

theorem exists_polynomial_degree_n (n : ℕ) (hn : 0 < n) : 
  ∃ (ω ψ : Polynomial ℤ), ω.degree = n ∧ (ω^2 = (X^2 - 1) * ψ^2 + 1) := 
sorry

end NUMINAMATH_GPT_exists_polynomial_degree_n_l2148_214887


namespace NUMINAMATH_GPT_triangle_right_triangle_of_consecutive_integers_sum_l2148_214881

theorem triangle_right_triangle_of_consecutive_integers_sum (
  m n : ℕ
) (h1 : 0 < m) (h2 : n^2 = 2*m + 1) : 
  n * n + m * m = (m + 1) * (m + 1) := 
sorry

end NUMINAMATH_GPT_triangle_right_triangle_of_consecutive_integers_sum_l2148_214881


namespace NUMINAMATH_GPT_ratio_proof_l2148_214878

theorem ratio_proof (a b c d e : ℕ) (h1 : a * 4 = 3 * b) (h2 : b * 9 = 7 * c)
  (h3 : c * 7 = 5 * d) (h4 : d * 13 = 11 * e) : a * 468 = 165 * e :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l2148_214878


namespace NUMINAMATH_GPT_grasshopper_visit_all_points_min_jumps_l2148_214821

noncomputable def grasshopper_min_jumps : ℕ := 18

theorem grasshopper_visit_all_points_min_jumps (n m : ℕ) (h₁ : n = 2014) (h₂ : m = 18) :
  ∃ k : ℕ, k ≤ m ∧ (∀ i : ℤ, 0 ≤ i → i < n → ∃ j : ℕ, j < k ∧ (j * 57 + i * 10) % n = i) :=
sorry

end NUMINAMATH_GPT_grasshopper_visit_all_points_min_jumps_l2148_214821


namespace NUMINAMATH_GPT_least_value_l2148_214827

-- Define the quadratic function and its conditions
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + q * x + r

-- Define the conditions for p, q, and r
def conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ (q^2 - 4 * p * r < 0)

-- State the theorem that given the conditions the least value is (4pr - q^2) / 4p
theorem least_value (p q r : ℝ) (h : conditions p q r) :
  ∃ x : ℝ, (∀ y : ℝ, quadratic_function p q r y ≥ quadratic_function p q r x) ∧
  quadratic_function p q r x = (4 * p * r - q^2) / (4 * p) :=
sorry

end NUMINAMATH_GPT_least_value_l2148_214827


namespace NUMINAMATH_GPT_prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l2148_214805

noncomputable def prob_A_wins_game := 2 / 3
noncomputable def prob_B_wins_game := 1 / 3

/-- The probability that the match ends after two games with player A's victory is 4/9. -/
theorem prob_match_ends_two_games_A_wins :
  prob_A_wins_game * prob_A_wins_game = 4 / 9 := by
  sorry

/-- The probability that the match ends exactly after four games is 20/81. -/
theorem prob_match_ends_four_games :
  2 * prob_A_wins_game * prob_B_wins_game * (prob_A_wins_game^2 + prob_B_wins_game^2) = 20 / 81 := by
  sorry

/-- The probability that player A wins the match overall is 74/81. -/
theorem prob_A_wins_overall :
  (prob_A_wins_game^2 + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game^2
  + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game * prob_B_wins_game) / (prob_A_wins_game + prob_B_wins_game) = 74 / 81 := by
  sorry

end NUMINAMATH_GPT_prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l2148_214805


namespace NUMINAMATH_GPT_percentage_saved_is_10_l2148_214826

-- Given conditions
def rent_expenses : ℕ := 5000
def milk_expenses : ℕ := 1500
def groceries_expenses : ℕ := 4500
def education_expenses : ℕ := 2500
def petrol_expenses : ℕ := 2000
def misc_expenses : ℕ := 3940
def savings : ℕ := 2160

-- Define the total expenses
def total_expenses : ℕ := rent_expenses + milk_expenses + groceries_expenses + education_expenses + petrol_expenses + misc_expenses

-- Define the total monthly salary
def total_monthly_salary : ℕ := total_expenses + savings

-- Define the percentage of savings
def percentage_saved : ℕ := (savings * 100) / total_monthly_salary

-- Prove that the percentage saved is 10%
theorem percentage_saved_is_10 :
  percentage_saved = 10 :=
sorry

end NUMINAMATH_GPT_percentage_saved_is_10_l2148_214826


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l2148_214802

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n+1) = a n + d

-- The given condition a_3 + a_5 = 12 - a_7
def condition : Prop :=
  a 3 + a 5 = 12 - a 7

-- The proof statement
theorem arithmetic_sequence_solution 
  (h_arith : is_arithmetic_sequence a d) 
  (h_cond : condition a): a 1 + a 9 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l2148_214802


namespace NUMINAMATH_GPT_travel_distance_bus_l2148_214886

theorem travel_distance_bus (D P T B : ℝ) 
    (hD : D = 1800)
    (hP : P = D / 3)
    (hT : T = (2 / 3) * B)
    (h_total : P + T + B = D) :
    B = 720 := 
by
    sorry

end NUMINAMATH_GPT_travel_distance_bus_l2148_214886


namespace NUMINAMATH_GPT_necessary_condition_lg_l2148_214899

theorem necessary_condition_lg (x : ℝ) : ¬(x > -1) → ¬(10^1 > x + 1) := by {
    sorry
}

end NUMINAMATH_GPT_necessary_condition_lg_l2148_214899


namespace NUMINAMATH_GPT_xiao_yang_correct_answers_l2148_214898

noncomputable def problems_group_a : ℕ := 5
noncomputable def points_per_problem_group_a : ℕ := 8
noncomputable def problems_group_b : ℕ := 12
noncomputable def points_per_problem_group_b_correct : ℕ := 5
noncomputable def points_per_problem_group_b_incorrect : ℤ := -2
noncomputable def total_score : ℕ := 71
noncomputable def correct_answers_group_a : ℕ := 2 -- minimum required
noncomputable def correct_answers_total : ℕ := 13 -- provided correct result by the problem

theorem xiao_yang_correct_answers : correct_answers_total = 13 := by
  sorry

end NUMINAMATH_GPT_xiao_yang_correct_answers_l2148_214898


namespace NUMINAMATH_GPT_interest_rate_annual_l2148_214824

theorem interest_rate_annual :
  ∃ R : ℝ, 
    (5000 * 2 * R / 100) + (3000 * 4 * R / 100) = 2640 ∧ 
    R = 12 :=
sorry

end NUMINAMATH_GPT_interest_rate_annual_l2148_214824


namespace NUMINAMATH_GPT_functional_equation_solution_l2148_214842

theorem functional_equation_solution (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x + y) * (f x - f y) = a * (x - y) * f (x + y)) :
  (a = 1 → ∃ α β : ℝ, ∀ x : ℝ, f x = α * x^2 + β * x) ∧
  (a ≠ 1 ∧ a ≠ 0 → ∀ x : ℝ, f x = 0) ∧
  (a = 0 → ∃ c : ℝ, ∀ x : ℝ, f x = c) :=
by sorry

end NUMINAMATH_GPT_functional_equation_solution_l2148_214842


namespace NUMINAMATH_GPT_geometric_sequence_product_l2148_214897

theorem geometric_sequence_product 
    (a : ℕ → ℝ)
    (h_geom : ∀ n m, a (n + m) = a n * a m)
    (h_roots : ∀ x, x^2 - 3*x + 2 = 0 → (x = a 7 ∨ x = a 13)) :
  a 2 * a 18 = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l2148_214897


namespace NUMINAMATH_GPT_remainder_of_sum_div_17_l2148_214840

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_div_17_l2148_214840


namespace NUMINAMATH_GPT_paula_shirts_count_l2148_214856

variable {P : Type}

-- Given conditions as variable definitions
def initial_money : ℕ := 109
def shirt_cost : ℕ := 11
def pants_cost : ℕ := 13
def money_left : ℕ := 74
def money_spent : ℕ := initial_money - money_left
def shirts_count : ℕ → ℕ := λ S => shirt_cost * S

-- Main proposition to prove
theorem paula_shirts_count (S : ℕ) (h : money_spent = shirts_count S + pants_cost) : 
  S = 2 := by
  /- 
    Following the steps of the proof:
    1. Calculate money spent is $35.
    2. Set up the equation $11S + 13 = 35.
    3. Solve for S.
  -/
  sorry

end NUMINAMATH_GPT_paula_shirts_count_l2148_214856


namespace NUMINAMATH_GPT_max_wickets_in_innings_l2148_214850

-- Define the max wickets a bowler can take per over
def max_wickets_per_over : ℕ := 3

-- Define the number of overs bowled by the bowler
def overs_bowled : ℕ := 6

-- Assume the total players in a cricket team
def total_players : ℕ := 11

-- Lean statement that proves the maximum number of wickets the bowler can take in an innings
theorem max_wickets_in_innings :
  3 * 6 ≥ total_players - 1 →
  max_wickets_per_over * overs_bowled ≥ total_players - 1 :=
by
  sorry

end NUMINAMATH_GPT_max_wickets_in_innings_l2148_214850


namespace NUMINAMATH_GPT_Vinnie_exceeded_word_limit_l2148_214823

theorem Vinnie_exceeded_word_limit :
  let words_limit := 1000
  let words_saturday := 450
  let words_sunday := 650
  let total_words := words_saturday + words_sunday
  total_words - words_limit = 100 :=
by
  sorry

end NUMINAMATH_GPT_Vinnie_exceeded_word_limit_l2148_214823


namespace NUMINAMATH_GPT_point_on_x_axis_right_of_origin_is_3_units_away_l2148_214845

theorem point_on_x_axis_right_of_origin_is_3_units_away :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ P.1 > 0 ∧ dist (P.1, P.2) (0, 0) = 3 ∧ P = (3, 0) := 
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_right_of_origin_is_3_units_away_l2148_214845


namespace NUMINAMATH_GPT_gcd_fx_x_l2148_214859

noncomputable def f (x : ℕ) : ℕ := (5 * x + 3) * (8 * x + 2) * (12 * x + 7) * (3 * x + 11)

theorem gcd_fx_x (x : ℕ) (h : ∃ k : ℕ, x = 18720 * k) : Nat.gcd (f x) x = 462 :=
sorry

end NUMINAMATH_GPT_gcd_fx_x_l2148_214859


namespace NUMINAMATH_GPT_number_divided_by_21_l2148_214834

theorem number_divided_by_21 (x : ℝ) (h : 6000 - (x / 21.0) = 5995) : x = 105 :=
by
  sorry

end NUMINAMATH_GPT_number_divided_by_21_l2148_214834


namespace NUMINAMATH_GPT_calculate_fraction_l2148_214889

theorem calculate_fraction : (2002 - 1999)^2 / 169 = 9 / 169 :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l2148_214889


namespace NUMINAMATH_GPT_probability_at_least_one_consonant_l2148_214877

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_consonant_l2148_214877


namespace NUMINAMATH_GPT_probability_all_same_color_l2148_214813

theorem probability_all_same_color :
  let total_marbles := 20
  let red_marbles := 5
  let white_marbles := 7
  let blue_marbles := 8
  let total_ways_to_draw_3 := (total_marbles * (total_marbles - 1) * (total_marbles - 2)) / 6
  let ways_to_draw_3_red := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) / 6
  let ways_to_draw_3_white := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) / 6
  let ways_to_draw_3_blue := (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) / 6
  let probability := (ways_to_draw_3_red + ways_to_draw_3_white + ways_to_draw_3_blue) / total_ways_to_draw_3
  probability = 101/1140 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_same_color_l2148_214813


namespace NUMINAMATH_GPT_tangent_parallel_to_line_at_point_l2148_214811

theorem tangent_parallel_to_line_at_point (P0 : ℝ × ℝ) 
  (curve : ℝ → ℝ) (line_slope : ℝ) : 
  curve = (fun x => x^3 + x - 2) ∧ line_slope = 4 ∧
  (∃ x0, P0 = (x0, curve x0) ∧ 3*x0^2 + 1 = line_slope) → 
  P0 = (1, 0) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_parallel_to_line_at_point_l2148_214811


namespace NUMINAMATH_GPT_sin_over_sin_l2148_214895

theorem sin_over_sin (a : Real) (h_cos : Real.cos (Real.pi / 4 - a) = 12 / 13)
  (h_quadrant : 0 < Real.pi / 4 - a ∧ Real.pi / 4 - a < Real.pi / 2) :
  Real.sin (Real.pi / 2 - 2 * a) / Real.sin (Real.pi / 4 + a) = 119 / 144 := by
sorry

end NUMINAMATH_GPT_sin_over_sin_l2148_214895


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2148_214872

theorem sufficient_not_necessary (x : ℝ) : (x < 1 → x < 2) ∧ (¬(x < 2 → x < 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2148_214872


namespace NUMINAMATH_GPT_inequality_not_always_true_l2148_214819

-- Declare the variables and conditions
variables {a b c : ℝ}

-- Given conditions
axiom h1 : a < b 
axiom h2 : b < c 
axiom h3 : a * c < 0

-- Statement of the problem
theorem inequality_not_always_true : ¬ (∀ a b c, (a < b ∧ b < c ∧ a * c < 0) → (c^2 / a < b^2 / a)) :=
by { sorry }

end NUMINAMATH_GPT_inequality_not_always_true_l2148_214819


namespace NUMINAMATH_GPT_f_2011_l2148_214893

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

theorem f_2011 : f 2011 = -2 :=
by sorry

end NUMINAMATH_GPT_f_2011_l2148_214893


namespace NUMINAMATH_GPT_discount_price_l2148_214884

theorem discount_price (P : ℝ) (h : P > 0) (discount : ℝ) (h_discount : discount = 0.80) : 
  (P - P * discount) = P * 0.20 :=
by
  sorry

end NUMINAMATH_GPT_discount_price_l2148_214884


namespace NUMINAMATH_GPT_calculate_fraction_l2148_214866

variables (n_bl: ℕ) (deg_warm: ℕ) (total_deg: ℕ) (total_bl: ℕ)

def blanket_fraction_added := total_deg / deg_warm

theorem calculate_fraction (h1: deg_warm = 3) (h2: total_deg = 21) (h3: total_bl = 14) :
  (blanket_fraction_added total_deg deg_warm) / total_bl = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_fraction_l2148_214866


namespace NUMINAMATH_GPT_tangent_line_circle_midpoint_locus_l2148_214832

/-- 
Let O be the circle x^2 + y^2 = 1,
M be the point (-1, -4), and
N be the point (2, 0).
-/
structure CircleTangentMidpointProblem where
  (x y : ℝ)
  (O_eq : x^2 + y^2 = 1)
  (M_eq : x = -1 ∧ y = -4)
  (N_eq : x = 2 ∧ y = 0)

/- Part (1) -/
theorem tangent_line_circle (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                            (Mx My : ℝ) : ((Mx = -1 ∧ My = -4) → 
                          
                            (x = -1 ∨ 15 * x - 8 * y - 17 = 0)) := by
  sorry

/- Part (2) -/
theorem midpoint_locus (x y : ℝ) (O_eq : x^2 + y^2 = 1) 
                       (Nx Ny : ℝ) : ((Nx = 2 ∧ Ny = 0) → 
                       
                       ((x-1)^2 + y^2 = 1 ∧ (0 ≤ x ∧ x < 1 / 2))) := by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_midpoint_locus_l2148_214832


namespace NUMINAMATH_GPT_opposite_of_neg_one_third_l2148_214873

noncomputable def a : ℚ := -1 / 3

theorem opposite_of_neg_one_third : -a = 1 / 3 := 
by 
sorry

end NUMINAMATH_GPT_opposite_of_neg_one_third_l2148_214873


namespace NUMINAMATH_GPT_first_discount_percentage_l2148_214883

/-- A theorem to determine the first discount percentage on sarees -/
theorem first_discount_percentage (x : ℝ) (h : 
((400 - (x / 100) * 400) - (8 / 100) * (400 - (x / 100) * 400) = 331.2)) : x = 10 := by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l2148_214883


namespace NUMINAMATH_GPT_x_gt_one_iff_x_cube_gt_one_l2148_214801

theorem x_gt_one_iff_x_cube_gt_one (x : ℝ) : x > 1 ↔ x^3 > 1 :=
by sorry

end NUMINAMATH_GPT_x_gt_one_iff_x_cube_gt_one_l2148_214801


namespace NUMINAMATH_GPT_xy_value_l2148_214804

theorem xy_value :
  ∃ a b c x y : ℝ,
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
    3 * a + 2 * b + c = 5 ∧
    2 * a + b - 3 * c = 1 ∧
    (∀ m, m = 3 * a + b - 7 * c → (m = x ∨ m = y)) ∧
    x = -5 / 7 ∧
    y = -1 / 11 ∧
    x * y = 5 / 77 :=
sorry

end NUMINAMATH_GPT_xy_value_l2148_214804


namespace NUMINAMATH_GPT_julia_money_remaining_l2148_214817

theorem julia_money_remaining 
  (initial_amount : ℝ)
  (tablet_percentage : ℝ)
  (phone_percentage : ℝ)
  (game_percentage : ℝ)
  (case_percentage : ℝ) 
  (final_money : ℝ) :
  initial_amount = 120 → 
  tablet_percentage = 0.45 → 
  phone_percentage = 1/3 → 
  game_percentage = 0.25 → 
  case_percentage = 0.10 → 
  final_money = initial_amount * (1 - tablet_percentage) * (1 - phone_percentage) * (1 - game_percentage) * (1 - case_percentage) →
  final_money = 29.70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_julia_money_remaining_l2148_214817


namespace NUMINAMATH_GPT_ratio_of_stock_values_l2148_214847

/-- Definitions and conditions -/
def value_expensive := 78
def shares_expensive := 14
def shares_other := 26
def total_assets := 2106

/-- The proof problem -/
theorem ratio_of_stock_values : 
  ∃ (V_other : ℝ), 26 * V_other = total_assets - (shares_expensive * value_expensive) ∧ 
  (value_expensive / V_other) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_stock_values_l2148_214847


namespace NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l2148_214848

theorem equation_1_solution (x : ℝ) (h : (2 * x - 3)^2 = 9 * x^2) : x = 3 / 5 ∨ x = -3 :=
sorry

theorem equation_2_solution (x : ℝ) (h : 2 * x * (x - 2) + x = 2) : x = 2 ∨ x = -1 / 2 :=
sorry

end NUMINAMATH_GPT_equation_1_solution_equation_2_solution_l2148_214848


namespace NUMINAMATH_GPT_subset_bound_l2148_214875

open Finset

variables {α : Type*}

theorem subset_bound (n : ℕ) (S : Finset (Finset (Fin (4 * n)))) (hS : ∀ {s t : Finset (Fin (4 * n))}, s ∈ S → t ∈ S → s ≠ t → (s ∩ t).card ≤ n) (h_card : ∀ s ∈ S, s.card = 2 * n) :
  S.card ≤ 6 ^ ((n + 1) / 2) :=
sorry

end NUMINAMATH_GPT_subset_bound_l2148_214875


namespace NUMINAMATH_GPT_frac_diff_zero_l2148_214844

theorem frac_diff_zero (a b : ℝ) (h : a + b = a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a) - (1 / b) = 0 := 
sorry

end NUMINAMATH_GPT_frac_diff_zero_l2148_214844


namespace NUMINAMATH_GPT_intersection_points_of_parametric_curve_l2148_214822

def parametric_curve_intersection_points (t : ℝ) : Prop :=
  let x := t - 1
  let y := t + 2
  (x = -3 ∧ y = 0) ∨ (x = 0 ∧ y = 3)

theorem intersection_points_of_parametric_curve :
  ∃ t1 t2 : ℝ, parametric_curve_intersection_points t1 ∧ parametric_curve_intersection_points t2 := 
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_parametric_curve_l2148_214822


namespace NUMINAMATH_GPT_sandy_initial_payment_l2148_214869

theorem sandy_initial_payment (P : ℝ) (H1 : P + 300 < P + 1320)
  (H2 : 1320 = 1.10 * (P + 300)) : P = 900 :=
sorry

end NUMINAMATH_GPT_sandy_initial_payment_l2148_214869


namespace NUMINAMATH_GPT_number_of_tea_bags_l2148_214880

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tea_bags_l2148_214880


namespace NUMINAMATH_GPT_ratio_of_areas_GHI_to_JKL_l2148_214862

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_areas_GHI_to_JKL_l2148_214862


namespace NUMINAMATH_GPT_zongzi_cost_per_bag_first_batch_l2148_214815

theorem zongzi_cost_per_bag_first_batch (x : ℝ)
  (h1 : 7500 / (x - 4) = 3 * (3000 / x))
  (h2 : 3000 > 0)
  (h3 : 7500 > 0)
  (h4 : x > 4) :
  x = 24 :=
by sorry

end NUMINAMATH_GPT_zongzi_cost_per_bag_first_batch_l2148_214815


namespace NUMINAMATH_GPT_compare_a_b_l2148_214863

theorem compare_a_b (a b : ℝ) (h₁ : a = 1.9 * 10^5) (h₂ : b = 9.1 * 10^4) : a > b := by
  sorry

end NUMINAMATH_GPT_compare_a_b_l2148_214863


namespace NUMINAMATH_GPT_value_of_T_l2148_214810

-- Define the main variables and conditions
variables {M T : ℝ}

-- State the conditions given in the problem
def condition1 (M T : ℝ) := 2 * M + T = 7000
def condition2 (M T : ℝ) := M + 2 * T = 9800

-- State the theorem to be proved
theorem value_of_T : 
  ∀ (M T : ℝ), condition1 M T ∧ condition2 M T → T = 4200 :=
by 
  -- Proof would go here; for now, we use "sorry" to skip it
  sorry

end NUMINAMATH_GPT_value_of_T_l2148_214810


namespace NUMINAMATH_GPT_radius_of_base_of_cone_correct_l2148_214885

noncomputable def radius_of_base_of_cone (n : ℕ) (r α : ℝ) : ℝ :=
  r * (1 / Real.sin (Real.pi / n) - 1 / Real.tan (Real.pi / 4 + α / 2))

theorem radius_of_base_of_cone_correct :
  radius_of_base_of_cone 11 3 (Real.pi / 6) = 3 / Real.sin (Real.pi / 11) - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_base_of_cone_correct_l2148_214885


namespace NUMINAMATH_GPT_find_primes_l2148_214838

theorem find_primes (A B C : ℕ) (hA : A < 20) (hB : B < 20) (hC : C < 20)
  (hA_prime : Prime A) (hB_prime : Prime B) (hC_prime : Prime C)
  (h_sum : A + B + C = 30) : 
  (A = 2 ∧ B = 11 ∧ C = 17) ∨ (A = 2 ∧ B = 17 ∧ C = 11) ∨ 
  (A = 11 ∧ B = 2 ∧ C = 17) ∨ (A = 11 ∧ B = 17 ∧ C = 2) ∨ 
  (A = 17 ∧ B = 2 ∧ C = 11) ∨ (A = 17 ∧ B = 11 ∧ C = 2) :=
sorry

end NUMINAMATH_GPT_find_primes_l2148_214838


namespace NUMINAMATH_GPT_find_k_for_linear_dependence_l2148_214855

structure vector2 :=
  (x : ℝ)
  (y : ℝ)

def linear_dependent (v1 v2 : vector2) :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
  c1 * v1.x + c2 * v2.x = 0 ∧
  c1 * v1.y + c2 * v2.y = 0

theorem find_k_for_linear_dependence :
  ∀ (k : ℝ), linear_dependent (vector2.mk 2 3) (vector2.mk 4 k) ↔ k = 6 :=
by sorry

end NUMINAMATH_GPT_find_k_for_linear_dependence_l2148_214855


namespace NUMINAMATH_GPT_charlie_and_elle_crayons_l2148_214858

theorem charlie_and_elle_crayons :
  (∃ (Lizzie Bobbie Billie Charlie Dave Elle : ℕ),
  Billie = 18 ∧
  Bobbie = 3 * Billie ∧
  Lizzie = Bobbie / 2 ∧
  Charlie = 2 * Lizzie ∧
  Dave = 4 * Billie ∧
  Elle = (Bobbie + Dave) / 2 ∧
  Charlie + Elle = 117) :=
sorry

end NUMINAMATH_GPT_charlie_and_elle_crayons_l2148_214858


namespace NUMINAMATH_GPT_evaluate_expression_l2148_214839

theorem evaluate_expression (a b x y c : ℝ) (h1 : a = -b) (h2 : x * y = 1) (h3 : |c| = 2) :
  (c = 2 → (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2) ∧
  (c = -2 → (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2148_214839


namespace NUMINAMATH_GPT_shaded_area_correct_l2148_214809

noncomputable def shaded_area (s r_small : ℝ) : ℝ :=
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2
  let semi_area := 6 * (1/2 * Real.pi * (s/2)^2)
  let small_circle_area := 6 * (Real.pi * (r_small)^2)
  hex_area - (semi_area + small_circle_area)

theorem shaded_area_correct : shaded_area 4 0.5 = 24 * Real.sqrt 3 - (27 * Real.pi / 2) := by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l2148_214809


namespace NUMINAMATH_GPT_find_angle_x_l2148_214868

theorem find_angle_x (x : ℝ) (h1 : 3 * x + 2 * x = 90) : x = 18 :=
  by
    sorry

end NUMINAMATH_GPT_find_angle_x_l2148_214868


namespace NUMINAMATH_GPT_find_f_l2148_214812

theorem find_f (d e f : ℝ) (h_g : 16 = g) 
  (h_mean_of_zeros : -d / 12 = 3 + d + e + f + 16) 
  (h_product_of_zeros_two_at_a_time : -d / 12 = e / 3) : 
  f = -39 :=
by
  sorry

end NUMINAMATH_GPT_find_f_l2148_214812


namespace NUMINAMATH_GPT_triangle_A_l2148_214879

variables {a b c : ℝ}
variables (A B C : ℝ) -- Represent vertices
variables (C1 C2 A1 A2 B1 B2 A' B' C' : ℝ)

-- Definition of equilateral triangle
def is_equilateral_trig (x y z : ℝ) : Prop :=
  dist x y = dist y z ∧ dist y z = dist z x

-- Given conditions
axiom ABC_equilateral : is_equilateral_trig A B C
axiom length_cond_1 : dist A1 A2 = a ∧ dist C B1 = a ∧ dist B C2 = a
axiom length_cond_2 : dist B1 B2 = b ∧ dist A C1 = b ∧ dist C A2 = b
axiom length_cond_3 : dist C1 C2 = c ∧ dist B A1 = c ∧ dist A B2 = c

-- Additional constructions
axiom A'_construction : is_equilateral_trig A' B2 C1
axiom B'_construction : is_equilateral_trig B' C2 A1
axiom C'_construction : is_equilateral_trig C' A2 B1

-- The final proof goal
theorem triangle_A'B'C'_equilateral : is_equilateral_trig A' B' C' :=
sorry

end NUMINAMATH_GPT_triangle_A_l2148_214879


namespace NUMINAMATH_GPT_dot_product_is_4_l2148_214871

-- Define the vectors a and b
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (2, 2)

-- Define collinearity condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define k based on the collinearity condition
def k_value : ℝ := 1 -- derived from solving the collinearity condition in the problem

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of a and b is 4 when k = 1
theorem dot_product_is_4 {k : ℝ} (h : k = k_value) : dot_product (a k) b = 4 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_dot_product_is_4_l2148_214871


namespace NUMINAMATH_GPT_minimum_value_condition_l2148_214892

-- Define the function y = x^3 - 2ax + a
noncomputable def f (a x : ℝ) : ℝ := x^3 - 2 * a * x + a

-- Define its derivative
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 - 2 * a

-- Define the lean theorem statement
theorem minimum_value_condition (a : ℝ) : 
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z ≥ y)) ∧
  ¬(∃ x y : ℝ, 0 < x ∧ x < 1 ∧ y = f a x ∧ (∀ z : ℝ, 0 < z ∧ z < 1 → f a z < y)) 
  ↔ 0 < a ∧ a < 3 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_condition_l2148_214892


namespace NUMINAMATH_GPT_sum_of_ais_l2148_214820

theorem sum_of_ais :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℕ), 
    (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (a4 > 0) ∧ (a5 > 0) ∧ (a6 > 0) ∧ (a7 > 0) ∧ (a8 > 0) ∧
    a1^2 + (2*a2)^2 + (3*a3)^2 + (4*a4)^2 + (5*a5)^2 + (6*a6)^2 + (7*a7)^2 + (8*a8)^2 = 204 ∧
    a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ais_l2148_214820


namespace NUMINAMATH_GPT_ζ_sum_8_l2148_214808

open Complex

def ζ1 : ℂ := sorry
def ζ2 : ℂ := sorry
def ζ3 : ℂ := sorry

def e1 := ζ1 + ζ2 + ζ3
def e2 := ζ1 * ζ2 + ζ2 * ζ3 + ζ3 * ζ1
def e3 := ζ1 * ζ2 * ζ3

axiom h1 : e1 = 2
axiom h2 : e1^2 - 2 * e2 = 8
axiom h3 : (e1^2 - 2 * e2)^2 - 2 * (e2^2 - 2 * e1 * e3) = 26

theorem ζ_sum_8 : ζ1^8 + ζ2^8 + ζ3^8 = 219 :=
by {
  -- The proof goes here, omitting solution steps as instructed.
  sorry
}

end NUMINAMATH_GPT_ζ_sum_8_l2148_214808


namespace NUMINAMATH_GPT_angle_measure_is_60_l2148_214825

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_angle_measure_is_60_l2148_214825


namespace NUMINAMATH_GPT_rate_of_grapes_calculation_l2148_214876

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_rate_of_grapes_calculation_l2148_214876


namespace NUMINAMATH_GPT_area_region_sum_l2148_214829

theorem area_region_sum (r1 r2 : ℝ) (angle : ℝ) (a b c : ℕ) : 
  r1 = 6 → r2 = 3 → angle = 30 → (54 * Real.sqrt 3 + (9 : ℝ) * Real.pi - (9 : ℝ) * Real.pi = a * Real.sqrt b + c * Real.pi) → a + b + c = 10 :=
by
  intros
  -- We fill this with the actual proof steps later
  sorry

end NUMINAMATH_GPT_area_region_sum_l2148_214829


namespace NUMINAMATH_GPT_residue_of_927_mod_37_l2148_214843

-- Define the condition of the problem, which is the modulus and the number
def modulus : ℤ := 37
def number : ℤ := -927

-- Define the statement we need to prove: that the residue of -927 mod 37 is 35
theorem residue_of_927_mod_37 : (number % modulus + modulus) % modulus = 35 := by
  sorry

end NUMINAMATH_GPT_residue_of_927_mod_37_l2148_214843


namespace NUMINAMATH_GPT_minimum_value_of_h_l2148_214890

noncomputable def h (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x))^2)

theorem minimum_value_of_h : (∀ x : ℝ, x > 0 → h x ≥ 2.25) ∧ (h 1 = 2.25) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_h_l2148_214890


namespace NUMINAMATH_GPT_part_a_part_b_l2148_214852

theorem part_a (N : ℕ) : ∃ (a : ℕ → ℕ), (∀ i : ℕ, 1 ≤ i → i ≤ N → a i > 0) ∧ (∀ i : ℕ, 2 ≤ i → i ≤ N → a i > a (i - 1)) ∧ 
(∀ i j : ℕ, 1 ≤ i → i < j → j ≤ N → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 1 - (1 : ℚ) / a 2) := sorry

theorem part_b : ¬ ∃ (a : ℕ → ℕ), (∀ i : ℕ, a i > 0) ∧ (∀ i : ℕ, a i < a (i + 1)) ∧ 
(∀ i j : ℕ, i < j → (1 : ℚ) / a i - (1 : ℚ) / a j = (1 : ℚ) / a 0 - (1 : ℚ) / a 1) := sorry

end NUMINAMATH_GPT_part_a_part_b_l2148_214852


namespace NUMINAMATH_GPT_jackson_has_1900_more_than_brandon_l2148_214807

-- Conditions
def initial_investment : ℝ := 500
def jackson_multiplier : ℝ := 4
def brandon_multiplier : ℝ := 0.20

-- Final values
def jackson_final_value := jackson_multiplier * initial_investment
def brandon_final_value := brandon_multiplier * initial_investment

-- Statement to prove the difference
theorem jackson_has_1900_more_than_brandon : jackson_final_value - brandon_final_value = 1900 := 
    by sorry

end NUMINAMATH_GPT_jackson_has_1900_more_than_brandon_l2148_214807


namespace NUMINAMATH_GPT_find_unknown_number_l2148_214870

theorem find_unknown_number (x : ℕ) (hx1 : 100 % x = 16) (hx2 : 200 % x = 4) : x = 28 :=
by 
  sorry

end NUMINAMATH_GPT_find_unknown_number_l2148_214870


namespace NUMINAMATH_GPT_largest_multiple_l2148_214846

theorem largest_multiple (n : ℤ) (h8 : 8 ∣ n) (h : -n > -80) : n = 72 :=
by 
  sorry

end NUMINAMATH_GPT_largest_multiple_l2148_214846


namespace NUMINAMATH_GPT_pages_remaining_l2148_214882

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end NUMINAMATH_GPT_pages_remaining_l2148_214882


namespace NUMINAMATH_GPT_height_difference_is_correct_l2148_214831

-- Define the heights of the trees as rational numbers.
def maple_tree_height : ℚ := 10 + 1 / 4
def spruce_tree_height : ℚ := 14 + 1 / 2

-- Prove that the spruce tree is 19 3/4 feet taller than the maple tree.
theorem height_difference_is_correct :
  spruce_tree_height - maple_tree_height = 19 + 3 / 4 := 
sorry

end NUMINAMATH_GPT_height_difference_is_correct_l2148_214831


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_minimum_l2148_214860

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_minimum (a1 : ℝ) (d : ℝ) :
  a1 = -20 ∧ (∀ n : ℕ, (S_n a1 d n) > (S_n a1 d 6)) → 
  (10 / 3 < d ∧ d < 4) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_minimum_l2148_214860


namespace NUMINAMATH_GPT_speed_of_stream_l2148_214800

def upstream_speed (v : ℝ) := 72 - v
def downstream_speed (v : ℝ) := 72 + v

theorem speed_of_stream (v : ℝ) (h : 1 / upstream_speed v = 2 * (1 / downstream_speed v)) : v = 24 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2148_214800


namespace NUMINAMATH_GPT_solve_for_x_y_l2148_214854

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def triangle_ABC (A B C E F : V) (x y : ℝ) : Prop :=
  (E - A) = (1 / 2) • (B - A) ∧
  (C - F) = (2 : ℝ) • (A - F) ∧
  (E - F) = x • (B - A) + y • (C - A)

theorem solve_for_x_y (A B C E F : V) (x y : ℝ) :
  triangle_ABC A B C E F x y →
  x + y = - (1 / 6 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_y_l2148_214854


namespace NUMINAMATH_GPT_find_third_integer_l2148_214803

noncomputable def third_odd_integer (x : ℤ) :=
  x + 4

theorem find_third_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : third_odd_integer x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_third_integer_l2148_214803


namespace NUMINAMATH_GPT_sum_of_y_coordinates_of_other_vertices_l2148_214806

theorem sum_of_y_coordinates_of_other_vertices
  (A B : ℝ × ℝ)
  (C D : ℝ × ℝ)
  (hA : A = (2, 15))
  (hB : B = (8, -2))
  (h_mid : midpoint ℝ A B = midpoint ℝ C D) :
  C.snd + D.snd = 13 := 
sorry

end NUMINAMATH_GPT_sum_of_y_coordinates_of_other_vertices_l2148_214806
