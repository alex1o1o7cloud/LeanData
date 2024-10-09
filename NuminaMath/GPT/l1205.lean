import Mathlib

namespace tilings_of_3_by_5_rectangle_l1205_120593

def num_tilings_of_3_by_5_rectangle : ℕ := 96

theorem tilings_of_3_by_5_rectangle (h : ℕ := 96) :
  (∃ (tiles : List (ℕ × ℕ)),
    tiles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] ∧
    -- Whether we are counting tiles in the context of a 3x5 rectangle
    -- with all distinct rotations and reflections allowed.
    True
  ) → num_tilings_of_3_by_5_rectangle = h :=
by {
  sorry -- Proof goes here
}

end tilings_of_3_by_5_rectangle_l1205_120593


namespace total_candies_darrel_took_l1205_120537

theorem total_candies_darrel_took (r b x : ℕ) (h1 : r = 3 * b)
  (h2 : r - x = 4 * (b - x))
  (h3 : r - x - 12 = 5 * (b - x - 12)) : 2 * x = 48 := sorry

end total_candies_darrel_took_l1205_120537


namespace roots_purely_imaginary_l1205_120558

open Complex

/-- 
  If m is a purely imaginary number, then the roots of the equation 
  8z^2 + 4i * z - m = 0 are purely imaginary.
-/
theorem roots_purely_imaginary (m : ℂ) (hm : m.im ≠ 0 ∧ m.re = 0) : 
  ∀ z : ℂ, 8 * z^2 + 4 * Complex.I * z - m = 0 → z.im ≠ 0 ∧ z.re = 0 :=
by
  sorry

end roots_purely_imaginary_l1205_120558


namespace largest_possible_s_l1205_120571

theorem largest_possible_s (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (h_angle : (101 : ℚ) / 97 * ((s - 2) * 180 / s : ℚ) = ((r - 2) * 180 / r : ℚ)) :
  s = 100 :=
by
  sorry

end largest_possible_s_l1205_120571


namespace tank_fills_in_56_minutes_l1205_120514

theorem tank_fills_in_56_minutes : 
  (∃ A B C : ℕ, (A = 40 ∧ B = 30 ∧ C = 20) ∧ 
                 ∃ capacity : ℕ, capacity = 950 ∧ 
                 ∃ time : ℕ, time = 56 ∧
                 ∀ cycle_time : ℕ, cycle_time = 3 ∧ 
                 ∀ net_water_per_cycle : ℕ, net_water_per_cycle = A + B - C ∧
                 ∀ total_cycles : ℕ, total_cycles = capacity / net_water_per_cycle ∧
                 ∀ total_time : ℕ, total_time = total_cycles * cycle_time - 1 ∧
                 total_time = time) :=
sorry

end tank_fills_in_56_minutes_l1205_120514


namespace soil_bags_needed_l1205_120502

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end soil_bags_needed_l1205_120502


namespace product_of_integers_l1205_120532

theorem product_of_integers (x y : ℕ) (h1 : x + y = 72) (h2 : x - y = 18) : x * y = 1215 := 
sorry

end product_of_integers_l1205_120532


namespace total_number_of_students_l1205_120527

theorem total_number_of_students (sample_size : ℕ) (first_year_selected : ℕ) (third_year_selected : ℕ) (second_year_students : ℕ) (second_year_selected : ℕ) (prob_selection : ℕ) :
  sample_size = 45 →
  first_year_selected = 20 →
  third_year_selected = 10 →
  second_year_students = 300 →
  second_year_selected = sample_size - first_year_selected - third_year_selected →
  prob_selection = second_year_selected / second_year_students →
  (sample_size / prob_selection) = 900 :=
by
  intros
  sorry

end total_number_of_students_l1205_120527


namespace min_valid_subset_card_eq_l1205_120585

open Finset

def pairs (n : ℕ) : Finset (ℕ × ℕ) := 
  (range n).product (range n)

def valid_subset (X : Finset (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (seq : ℕ → ℕ), ∃ k, (seq k, seq (k+1)) ∈ X

theorem min_valid_subset_card_eq (n : ℕ) (h : n = 10) : 
  ∃ X : Finset (ℕ × ℕ), valid_subset X n ∧ X.card = 55 := 
by 
  sorry

end min_valid_subset_card_eq_l1205_120585


namespace probability_not_win_l1205_120550

theorem probability_not_win (A B : Fin 16) : 
  (256 - 16) / 256 = 15 / 16 := 
by
  sorry

end probability_not_win_l1205_120550


namespace neither_sufficient_nor_necessary_l1205_120557

theorem neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a + b > 0 → ab > 0) ∧ (ab > 0 → a + b > 0)) :=
by {
  sorry
}

end neither_sufficient_nor_necessary_l1205_120557


namespace solve_quadratic_inequality_l1205_120554

theorem solve_quadratic_inequality (x : ℝ) :
  (x^2 - 2*x - 3 < 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end solve_quadratic_inequality_l1205_120554


namespace compound_interest_calculation_l1205_120508

-- Given conditions
def P : ℝ := 20000
def r : ℝ := 0.03
def t : ℕ := 5

-- The amount after t years with compound interest
def A := P * (1 + r) ^ t

-- Prove the total amount is as given in choice B
theorem compound_interest_calculation : 
  A = 20000 * (1 + 0.03) ^ 5 :=
by
  sorry

end compound_interest_calculation_l1205_120508


namespace cos_plus_sin_l1205_120512

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end cos_plus_sin_l1205_120512


namespace a_seq_correct_b_seq_max_m_l1205_120595

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 3 else (n + 1)^2 + 2

-- Verification that the sequence follows the provided conditions.
theorem a_seq_correct (n : ℕ) : 
  (a_seq 0 = 3) ∧
  (a_seq 1 = 6) ∧
  (a_seq 2 = 11) ∧
  (∀ m : ℕ, m ≥ 1 → a_seq (m + 1) - a_seq m = 2 * m + 1) := sorry

noncomputable def b_seq (n : ℕ) : ℝ := 
(a_seq n : ℝ) / (3 ^ (Real.sqrt (a_seq n - 2)))

theorem b_seq_max_m (m : ℝ) : 
  (∀ n : ℕ, b_seq n ≤ m) ↔ (1 ≤ m) := sorry

end a_seq_correct_b_seq_max_m_l1205_120595


namespace maximum_point_of_f_l1205_120506

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x - 2) * Real.exp x

theorem maximum_point_of_f : ∃ x : ℝ, x = -2 ∧
  ∀ y : ℝ, f y ≤ f x :=
sorry

end maximum_point_of_f_l1205_120506


namespace thirds_side_length_valid_l1205_120582

theorem thirds_side_length_valid (x : ℝ) (h1 : x > 5) (h2 : x < 13) : x = 12 :=
sorry

end thirds_side_length_valid_l1205_120582


namespace total_amount_divided_l1205_120596

theorem total_amount_divided (A B C : ℝ) (h1 : A = 2/3 * (B + C)) (h2 : B = 2/3 * (A + C)) (h3 : A = 80) : 
  A + B + C = 200 :=
by
  sorry

end total_amount_divided_l1205_120596


namespace initial_average_runs_l1205_120525

theorem initial_average_runs (A : ℕ) (h : 10 * A + 87 = 11 * (A + 5)) : A = 32 :=
by
  sorry

end initial_average_runs_l1205_120525


namespace intersection_is_3_l1205_120529

open Set -- Open the Set namespace to use set notation

theorem intersection_is_3 {A B : Set ℤ} (hA : A = {1, 3}) (hB : B = {-1, 2, 3}) :
  A ∩ B = {3} :=
by {
-- Proof goes here
  sorry
}

end intersection_is_3_l1205_120529


namespace pencils_in_stock_at_end_of_week_l1205_120528

def pencils_per_day : ℕ := 100
def days_per_week : ℕ := 5
def initial_pencils : ℕ := 80
def sold_pencils : ℕ := 350

theorem pencils_in_stock_at_end_of_week :
  (pencils_per_day * days_per_week + initial_pencils - sold_pencils) = 230 :=
by sorry  -- Proof will be filled in later

end pencils_in_stock_at_end_of_week_l1205_120528


namespace four_people_possible_l1205_120518

structure Person :=
(first_name : String)
(patronymic : String)
(surname : String)

def noThreePeopleShareSameAttribute (people : List Person) : Prop :=
  ∀ (attr : Person → String), ¬ ∃ (a b c : Person),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ (attr a = attr b) ∧ (attr b = attr c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def anyTwoPeopleShareAnAttribute (people : List Person) : Prop :=
  ∀ (a b : Person), a ∈ people ∧ b ∈ people ∧ a ≠ b →
    (a.first_name = b.first_name ∨ a.patronymic = b.patronymic ∨ a.surname = b.surname)

def validGroup (people : List Person) : Prop :=
  noThreePeopleShareSameAttribute people ∧ anyTwoPeopleShareAnAttribute people

theorem four_people_possible : ∃ (people : List Person), people.length = 4 ∧ validGroup people :=
sorry

end four_people_possible_l1205_120518


namespace jovana_added_shells_l1205_120586

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h_initial : initial_amount = 5) 
  (h_final : final_amount = 17) 
  (h_equation : final_amount = initial_amount + added_amount) : 
  added_amount = 12 := 
by 
  sorry

end jovana_added_shells_l1205_120586


namespace train_travel_time_change_l1205_120542

theorem train_travel_time_change 
  (t1 t2 : ℕ) (s1 s2 d : ℕ) 
  (h1 : t1 = 4) 
  (h2 : s1 = 50) 
  (h3 : s2 = 100) 
  (h4 : d = t1 * s1) :
  t2 = d / s2 → t2 = 2 :=
by
  intros
  sorry

end train_travel_time_change_l1205_120542


namespace jenny_kenny_see_each_other_l1205_120547

-- Definitions of conditions
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def paths_distance : ℝ := 300
def radius_building : ℝ := 75
def start_distance : ℝ := 300

-- Theorem statement
theorem jenny_kenny_see_each_other : ∃ t : ℝ, (t = 120) :=
by
  sorry

end jenny_kenny_see_each_other_l1205_120547


namespace sin_angle_GAC_correct_l1205_120515

noncomputable def sin_angle_GAC (AB AD AE : ℝ) := 
  let AC := Real.sqrt (AB^2 + AD^2)
  let AG := Real.sqrt (AB^2 + AD^2 + AE^2)
  (AC / AG)

theorem sin_angle_GAC_correct : sin_angle_GAC 2 3 4 = Real.sqrt 377 / 29 := by
  sorry

end sin_angle_GAC_correct_l1205_120515


namespace parallel_vectors_x_value_l1205_120522

theorem parallel_vectors_x_value :
  ∀ (x : ℝ), (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (2, x) → a.1 * b.2 = a.2 * b.1) → x = -4 :=
by
  intros x h
  have h_parallel := h (1, -2) (2, x) rfl rfl
  sorry

end parallel_vectors_x_value_l1205_120522


namespace unique_handshakes_l1205_120519

-- Define the circular arrangement and handshakes conditions
def num_people := 30
def handshakes_per_person := 2

theorem unique_handshakes : 
  (num_people * handshakes_per_person) / 2 = 30 :=
by
  -- Sorry is used here as a placeholder for the proof
  sorry

end unique_handshakes_l1205_120519


namespace arithmetic_sequence_a7_l1205_120590

/--
In an arithmetic sequence {a_n}, it is known that a_1 = 2 and a_3 + a_5 = 10.
Then, we need to prove that a_7 = 8.
-/
theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (h3 : ∀ n, a n = 2 + (n - 1) * d) : 
  a 7 = 8 := by
  sorry

end arithmetic_sequence_a7_l1205_120590


namespace gcd_490_910_l1205_120572

theorem gcd_490_910 : Nat.gcd 490 910 = 70 :=
by
  sorry

end gcd_490_910_l1205_120572


namespace sum_of_values_of_N_l1205_120591

theorem sum_of_values_of_N (N : ℂ) : (N * (N - 8) = 12) → (∃ x y : ℂ, N = x ∨ N = y ∧ x + y = 8) :=
by
  sorry

end sum_of_values_of_N_l1205_120591


namespace evaluate_expression_l1205_120511

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end evaluate_expression_l1205_120511


namespace jose_bottle_caps_proof_l1205_120505

def jose_bottle_caps_initial : Nat := 7
def rebecca_bottle_caps : Nat := 2
def jose_bottle_caps_final : Nat := 9

theorem jose_bottle_caps_proof : jose_bottle_caps_initial + rebecca_bottle_caps = jose_bottle_caps_final := by
  sorry

end jose_bottle_caps_proof_l1205_120505


namespace baker_total_cost_is_correct_l1205_120521

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l1205_120521


namespace exists_x0_l1205_120564

theorem exists_x0 : ∃ x0 : ℝ, x0^2 + 2*x0 + 1 ≤ 0 :=
sorry

end exists_x0_l1205_120564


namespace cell_population_l1205_120543

variable (n : ℕ)

def a (n : ℕ) : ℕ :=
  if n = 1 then 5
  else 1 -- Placeholder for general definition

theorem cell_population (n : ℕ) : a n = 2^(n-1) + 4 := by
  sorry

end cell_population_l1205_120543


namespace arithmetic_seq_necessary_not_sufficient_l1205_120576

noncomputable def arithmetic_sequence (a b c : ℝ) : Prop :=
  a + c = 2 * b

noncomputable def proposition_B (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ (a / b) + (c / b) = 2

theorem arithmetic_seq_necessary_not_sufficient (a b c : ℝ) :
  (arithmetic_sequence a b c → proposition_B a b c) ∧ 
  (∃ a' b' c', arithmetic_sequence a' b' c' ∧ ¬ proposition_B a' b' c') := by
  sorry

end arithmetic_seq_necessary_not_sufficient_l1205_120576


namespace sum_of_squares_of_consecutive_integers_l1205_120573

theorem sum_of_squares_of_consecutive_integers
  (a : ℤ) (h : (a - 1) * a * (a + 1) = 10 * ((a - 1) + a + (a + 1))) :
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 :=
sorry

end sum_of_squares_of_consecutive_integers_l1205_120573


namespace replace_square_l1205_120523

theorem replace_square (x : ℝ) (h : 10.0003 * x = 10000.3) : x = 1000 :=
sorry

end replace_square_l1205_120523


namespace rhombus_diagonal_sum_l1205_120575

theorem rhombus_diagonal_sum
  (d1 d2 : ℝ)
  (h1 : d1 ≤ 6)
  (h2 : 6 ≤ d2)
  (side_len : ℝ)
  (h_side : side_len = 5)
  (rhombus_relation : d1^2 + d2^2 = 4 * side_len^2) :
  d1 + d2 ≤ 14 :=
sorry

end rhombus_diagonal_sum_l1205_120575


namespace steve_speed_back_l1205_120560

theorem steve_speed_back :
  ∀ (v : ℝ), v > 0 → (20 / v + 20 / (2 * v) = 6) → 2 * v = 10 := 
by
  intros v v_pos h
  sorry

end steve_speed_back_l1205_120560


namespace solve_for_x_l1205_120533

theorem solve_for_x (x : ℝ) :
  5 * (x - 9) = 7 * (3 - 3 * x) + 10 → x = 38 / 13 :=
by
  intro h
  sorry

end solve_for_x_l1205_120533


namespace greatest_sum_l1205_120569

theorem greatest_sum (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 :=
sorry

end greatest_sum_l1205_120569


namespace largest_possible_s_l1205_120500

theorem largest_possible_s (r s: ℕ) (h1: r ≥ s) (h2: s ≥ 3)
  (h3: (59 : ℚ) / 58 * (180 * (s - 2) / s) = (180 * (r - 2) / r)) : s = 117 :=
sorry

end largest_possible_s_l1205_120500


namespace fraction_subtraction_l1205_120598

theorem fraction_subtraction : 
  (3 + 6 + 9) = 18 ∧ (2 + 5 + 8) = 15 ∧ (2 + 5 + 8) = 15 ∧ (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intro h
  sorry

end fraction_subtraction_l1205_120598


namespace find_missing_percentage_l1205_120555

theorem find_missing_percentage (P : ℝ) : (P * 50 = 2.125) → (P * 100 = 4.25) :=
by
  sorry

end find_missing_percentage_l1205_120555


namespace expected_red_hair_americans_l1205_120556

theorem expected_red_hair_americans (prob_red_hair : ℝ) (sample_size : ℕ) :
  prob_red_hair = 1 / 6 → sample_size = 300 → (prob_red_hair * sample_size = 50) := by
  intros
  sorry

end expected_red_hair_americans_l1205_120556


namespace Keith_picked_zero_apples_l1205_120574

variable (M J T K_A : ℕ)

theorem Keith_picked_zero_apples (hM : M = 14) (hJ : J = 41) (hT : T = 55) (hTotalOranges : M + J = T) : K_A = 0 :=
by
  sorry

end Keith_picked_zero_apples_l1205_120574


namespace original_price_of_dish_l1205_120530

-- Define the variables and conditions explicitly
variables (P : ℝ)

-- John's payment after discount and tip over original price
def john_payment : ℝ := 0.9 * P + 0.15 * P

-- Jane's payment after discount and tip over discounted price
def jane_payment : ℝ := 0.9 * P + 0.135 * P

-- Given condition that John's payment is $0.63 more than Jane's
def payment_difference : Prop := john_payment P - jane_payment P = 0.63

theorem original_price_of_dish (h : payment_difference P) : P = 42 :=
by sorry

end original_price_of_dish_l1205_120530


namespace numerical_expression_as_sum_of_squares_l1205_120551

theorem numerical_expression_as_sum_of_squares : 
  2 * (2009:ℕ)^2 + 2 * (2010:ℕ)^2 = (4019:ℕ)^2 + (1:ℕ)^2 := 
by
  sorry

end numerical_expression_as_sum_of_squares_l1205_120551


namespace infinite_divisible_269_l1205_120534

theorem infinite_divisible_269 (a : ℕ → ℤ) (h₀ : a 0 = 2) (h₁ : a 1 = 15) 
  (h_recur : ∀ n : ℕ, a (n + 2) = 15 * a (n + 1) + 16 * a n) :
  ∃ infinitely_many k: ℕ, 269 ∣ a k :=
by
  sorry

end infinite_divisible_269_l1205_120534


namespace number_property_l1205_120599

theorem number_property : ∀ n : ℕ, (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ n = 1 ∨ n = 4 :=
by sorry

end number_property_l1205_120599


namespace negation_of_exists_l1205_120531

theorem negation_of_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l1205_120531


namespace minimum_students_using_both_l1205_120584

theorem minimum_students_using_both (n L T x : ℕ) 
  (H1: 3 * n = 7 * L) 
  (H2: 5 * n = 6 * T) 
  (H3: n = 42) 
  (H4: n = L + T - x) : 
  x = 11 := 
by 
  sorry

end minimum_students_using_both_l1205_120584


namespace percentage_decrease_in_sale_l1205_120520

theorem percentage_decrease_in_sale (P Q : ℝ) (D : ℝ)
  (h1 : 1.80 * P * Q * (1 - D / 100) = 1.44 * P * Q) : 
  D = 20 :=
by
  -- Proof goes here
  sorry

end percentage_decrease_in_sale_l1205_120520


namespace find_g_9_l1205_120541

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l1205_120541


namespace no_five_consecutive_integers_with_fourth_powers_sum_l1205_120563

theorem no_five_consecutive_integers_with_fourth_powers_sum:
  ∀ n : ℤ, n^4 + (n + 1)^4 + (n + 2)^4 + (n + 3)^4 ≠ (n + 4)^4 :=
by
  intros
  sorry

end no_five_consecutive_integers_with_fourth_powers_sum_l1205_120563


namespace meal_cost_one_burger_one_shake_one_cola_l1205_120581

-- Define the costs of individual items
variables (B S C : ℝ)

-- Conditions based on given equations
def eq1 : Prop := 3 * B + 7 * S + C = 120
def eq2 : Prop := 4 * B + 10 * S + C = 160.50

-- Goal: Prove that the total cost of one burger, one shake, and one cola is $39
theorem meal_cost_one_burger_one_shake_one_cola :
  eq1 B S C → eq2 B S C → B + S + C = 39 :=
by 
  intros 
  sorry

end meal_cost_one_burger_one_shake_one_cola_l1205_120581


namespace minimum_value_expression_l1205_120526

-- Define the conditions in the problem
variable (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
variable (h3 : 2 * m + 2 * n = 2)

-- State the theorem proving the minimum value of the given expression
theorem minimum_value_expression : (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2 := by
  sorry

end minimum_value_expression_l1205_120526


namespace algebraic_identity_specific_case_l1205_120594

theorem algebraic_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2 * a * b :=
by sorry

theorem specific_case : 2021^2 - 2021 * 4034 + 2017^2 = 16 :=
by sorry

end algebraic_identity_specific_case_l1205_120594


namespace subtraction_example_l1205_120535

theorem subtraction_example : 3.57 - 1.45 = 2.12 :=
by 
  sorry

end subtraction_example_l1205_120535


namespace fraction_nonneg_if_x_ge_m8_l1205_120570

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8_l1205_120570


namespace parabola_fixed_point_thm_l1205_120592

-- Define the parabola condition
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus condition
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the slope product condition
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0 ∧ B.1 ≠ 0) → ((A.2 / A.1) * (B.2 / B.1) = -1 / 3)

-- Define the fixed point condition
def fixed_point (A B : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A ≠ B ∧ (x = 12) ∧ ((A.2 - B.2) / (A.1 - B.1)) * 12 = A.2

-- Problem statement in Lean
theorem parabola_fixed_point_thm (A B : ℝ × ℝ) (p : ℝ) :
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ C : ℝ → ℝ → ℝ → Prop, C = parabola) →
  (∃ F : ℝ × ℝ, focus F) →
  parabola A.2 A.1 p →
  parabola B.2 B.1 p →
  slope_product A B →
  fixed_point A B :=
by 
-- Sorry is used to skip the proof
sorry

end parabola_fixed_point_thm_l1205_120592


namespace fixed_point_on_line_AB_always_exists_l1205_120579

-- Define the line where P lies
def line (x y : ℝ) : Prop := x + 2 * y = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

-- Define the point P
def moving_point_P (x y : ℝ) : Prop := line x y

-- Define the function that checks if a point is a tangent to the ellipse
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  moving_point_P x0 y0 → (x * x0 + 4 * y * y0 = 4)

-- Statement: There exists a fixed point (1, 1/2) through which the line AB always passes
theorem fixed_point_on_line_AB_always_exists :
  ∀ (P A B : ℝ × ℝ),
    moving_point_P P.1 P.2 →
    is_tangent P.1 P.2 A.1 A.2 →
    is_tangent P.1 P.2 B.1 B.2 →
    ∃ (F : ℝ × ℝ), F = (1, 1/2) ∧ (F.1 - A.1) / (F.2 - A.2) = (F.1 - B.1) / (F.2 - B.2) :=
by
  sorry

end fixed_point_on_line_AB_always_exists_l1205_120579


namespace mark_saves_5_dollars_l1205_120577

def cost_per_pair : ℤ := 50

def promotionA_total_cost (cost : ℤ) : ℤ :=
  cost + (cost / 2)

def promotionB_total_cost (cost : ℤ) : ℤ :=
  cost + (cost - 20)

def savings (totalB totalA : ℤ) : ℤ :=
  totalB - totalA

theorem mark_saves_5_dollars :
  savings (promotionB_total_cost cost_per_pair) (promotionA_total_cost cost_per_pair) = 5 := by
  sorry

end mark_saves_5_dollars_l1205_120577


namespace problem_proof_l1205_120509

theorem problem_proof (a b c x y z : ℝ) (h₁ : 17 * x + b * y + c * z = 0) (h₂ : a * x + 29 * y + c * z = 0)
                      (h₃ : a * x + b * y + 53 * z = 0) (ha : a ≠ 17) (hx : x ≠ 0) :
                      (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
by
  -- proof goes here
  sorry

end problem_proof_l1205_120509


namespace cannot_form_polygon_l1205_120566

-- Define the stick lengths as a list
def stick_lengths : List ℕ := List.range 100 |>.map (λ n => 2^n)

-- Define the condition for forming a polygon
def can_form_polygon (lst : List ℕ) : Prop :=
  ∃ subset, subset ⊆ lst ∧ subset.length ≥ 3 ∧ (∀ s ∈ subset, s < (subset.sum - s))

-- The theorem to be proved
theorem cannot_form_polygon : ¬ can_form_polygon stick_lengths :=
by 
  sorry

end cannot_form_polygon_l1205_120566


namespace greatest_of_5_consecutive_integers_l1205_120517

theorem greatest_of_5_consecutive_integers (m n : ℤ) (h : 5 * n + 10 = m^3) : (n + 4) = 202 := by
sorry

end greatest_of_5_consecutive_integers_l1205_120517


namespace burn_rate_walking_l1205_120513

def burn_rate_running : ℕ := 10
def total_calories : ℕ := 450
def total_time : ℕ := 60
def running_time : ℕ := 35

theorem burn_rate_walking :
  ∃ (W : ℕ), ((running_time * burn_rate_running) + ((total_time - running_time) * W) = total_calories) ∧ (W = 4) :=
by
  sorry

end burn_rate_walking_l1205_120513


namespace wheat_pile_weight_l1205_120587

noncomputable def weight_of_conical_pile
  (circumference : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := circumference / (2 * 3.14)
  let volume := (1.0 / 3.0) * 3.14 * r^2 * height
  volume * density

theorem wheat_pile_weight :
  weight_of_conical_pile 12.56 1.2 30 = 150.72 :=
by
  sorry

end wheat_pile_weight_l1205_120587


namespace days_to_clear_land_l1205_120597

-- Definitions of all the conditions
def length_of_land := 200
def width_of_land := 900
def area_cleared_by_one_rabbit_per_day_square_yards := 10
def number_of_rabbits := 100
def conversion_square_yards_to_square_feet := 9
def total_area_of_land := length_of_land * width_of_land
def area_cleared_by_one_rabbit_per_day_square_feet := area_cleared_by_one_rabbit_per_day_square_yards * conversion_square_yards_to_square_feet
def area_cleared_by_all_rabbits_per_day := number_of_rabbits * area_cleared_by_one_rabbit_per_day_square_feet

-- Theorem to prove the number of days required to clear the land
theorem days_to_clear_land :
  total_area_of_land / area_cleared_by_all_rabbits_per_day = 20 := by
  sorry

end days_to_clear_land_l1205_120597


namespace sequence_inequality_l1205_120568

theorem sequence_inequality
  (n : ℕ) (h1 : 1 < n)
  (a : ℕ → ℕ)
  (h2 : ∀ i, i < n → a i < a (i + 1))
  (h3 : ∀ i, i < n - 1 → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) :
  a (n - 1) ≥ 2 * n ^ 2 - 1 :=
sorry

end sequence_inequality_l1205_120568


namespace sum_of_squares_divisible_by_three_l1205_120544

theorem sum_of_squares_divisible_by_three {a b : ℤ} 
  (h : 3 ∣ (a^2 + b^2)) : (3 ∣ a ∧ 3 ∣ b) :=
by 
  sorry

end sum_of_squares_divisible_by_three_l1205_120544


namespace kittens_percentage_rounded_l1205_120583

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end kittens_percentage_rounded_l1205_120583


namespace fifth_eqn_nth_eqn_l1205_120503

theorem fifth_eqn : 10 * 12 + 1 = 121 :=
by
  sorry

theorem nth_eqn (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end fifth_eqn_nth_eqn_l1205_120503


namespace mystery_number_addition_l1205_120516

theorem mystery_number_addition (mystery_number : ℕ) (h : mystery_number = 47) : mystery_number + 45 = 92 :=
by
  -- Proof goes here
  sorry

end mystery_number_addition_l1205_120516


namespace incorrect_expression_l1205_120504

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : x / (y - x) ≠ 5 / 2 := 
by
  sorry

end incorrect_expression_l1205_120504


namespace cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l1205_120536

-- Definitions for angles A, B, and C forming an arithmetic sequence and their sum being 180 degrees
variables {A B C : ℝ}

-- Definitions for side lengths a, b, and c forming a geometric sequence
variables {a b c : ℝ}

-- Question 1: Prove that cos B = 1/2 under the given conditions
theorem cos_B_equals_half 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) : 
  Real.cos B = 1 / 2 :=
sorry

-- Question 2: Prove that sin A * sin C = 3/4 under the given conditions
theorem sin_A_mul_sin_C_equals_three_fourths 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) 
  (h3 : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l1205_120536


namespace length_of_platform_is_180_l1205_120552

-- Define the train passing a platform and a man with given speeds and times
def train_pass_platform (speed : ℝ) (time_man time_platform : ℝ) (length_train length_platform : ℝ) :=
  time_man = length_train / speed ∧ 
  time_platform = (length_train + length_platform) / speed

-- Given conditions
noncomputable def train_length_platform :=
  ∃ length_platform,
    train_pass_platform 15 20 32 300 length_platform ∧
    length_platform = 180

-- The main theorem we want to prove
theorem length_of_platform_is_180 : train_length_platform :=
sorry

end length_of_platform_is_180_l1205_120552


namespace average_age_before_new_students_l1205_120588

theorem average_age_before_new_students
  (A : ℝ) (N : ℕ)
  (h1 : N = 15)
  (h2 : 15 * 32 + N * A = (N + 15) * (A - 4)) :
  A = 40 :=
by {
  sorry
}

end average_age_before_new_students_l1205_120588


namespace largest_distance_l1205_120578

noncomputable def max_distance_between_spheres 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) : ℝ :=
dist c1 c2 + r1 + r2

theorem largest_distance 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) 
  (h₁ : c1 = (-3, -15, 10))
  (h₂ : r1 = 24)
  (h₃ : c2 = (20, 18, -30))
  (h₄ : r2 = 95) : 
  max_distance_between_spheres c1 r1 c2 r2 = Real.sqrt 3218 + 119 := 
by
  sorry

end largest_distance_l1205_120578


namespace even_of_square_even_l1205_120589

theorem even_of_square_even (a : Int) (h1 : ∃ n : Int, a = 2 * n) (h2 : Even (a ^ 2)) : Even a := 
sorry

end even_of_square_even_l1205_120589


namespace no_valid_rook_placement_l1205_120538

theorem no_valid_rook_placement :
  ∀ (r b g : ℕ), r + b + g = 50 →
  (2 * r ≤ b) →
  (2 * b ≤ g) →
  (2 * g ≤ r) →
  False :=
by
  -- Proof goes here
  sorry

end no_valid_rook_placement_l1205_120538


namespace yellow_chip_count_l1205_120510

def point_values_equation (Y B G R : ℕ) : Prop :=
  2 ^ Y * 4 ^ B * 5 ^ G * 7 ^ R = 560000

theorem yellow_chip_count (Y B G R : ℕ) (h1 : B = 2 * G) (h2 : R = B / 2) (h3 : point_values_equation Y B G R) :
  Y = 2 :=
by
  sorry

end yellow_chip_count_l1205_120510


namespace algebraic_identity_l1205_120501

theorem algebraic_identity (x : ℝ) (h : x = Real.sqrt 3 + 2) : x^2 - 4 * x + 3 = 2 := 
by
  -- proof steps here
  sorry

end algebraic_identity_l1205_120501


namespace illiterate_employee_count_l1205_120507

variable (I : ℕ) -- Number of illiterate employees
variable (literate_count : ℕ) -- Number of literate employees
variable (initial_wage_illiterate : ℕ) -- Initial average wage of illiterate employees
variable (new_wage_illiterate : ℕ) -- New average wage of illiterate employees
variable (average_salary_decrease : ℕ) -- Decrease in the average salary of all employees

-- Given conditions:
def condition1 : initial_wage_illiterate = 25 := by sorry
def condition2 : new_wage_illiterate = 10 := by sorry
def condition3 : average_salary_decrease = 10 := by sorry
def condition4 : literate_count = 10 := by sorry

-- Main proof statement:
theorem illiterate_employee_count :
  initial_wage_illiterate - new_wage_illiterate = 15 →
  average_salary_decrease * (literate_count + I) = (initial_wage_illiterate - new_wage_illiterate) * I →
  I = 20 := by
  intros h1 h2
  -- provided conditions
  exact sorry

end illiterate_employee_count_l1205_120507


namespace total_number_of_numbers_l1205_120546

-- Definitions using the conditions from the problem
def sum_of_first_4_numbers : ℕ := 4 * 4
def sum_of_last_4_numbers : ℕ := 4 * 4
def average_of_all_numbers (n : ℕ) : ℕ := 3 * n
def fourth_number : ℕ := 11
def total_sum_of_numbers : ℕ := sum_of_first_4_numbers + sum_of_last_4_numbers - fourth_number

-- Theorem stating the problem
theorem total_number_of_numbers (n : ℕ) : total_sum_of_numbers = average_of_all_numbers n → n = 7 :=
by {
  sorry
}

end total_number_of_numbers_l1205_120546


namespace range_of_a_l1205_120559

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 3 → deriv (f a) x ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end range_of_a_l1205_120559


namespace discount_calculation_l1205_120548

noncomputable def cost_price : ℝ := 180
noncomputable def markup_percentage : ℝ := 0.4778
noncomputable def profit_percentage : ℝ := 0.20

noncomputable def marked_price (CP : ℝ) (MP_percent : ℝ) : ℝ := CP + (MP_percent * CP)
noncomputable def selling_price (CP : ℝ) (PP_percent : ℝ) : ℝ := CP + (PP_percent * CP)
noncomputable def discount (MP : ℝ) (SP : ℝ) : ℝ := MP - SP

theorem discount_calculation :
  discount (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 50.004 :=
by
  sorry

end discount_calculation_l1205_120548


namespace problem_solution_l1205_120561

def prop_p (a b c : ℝ) : Prop := a < b → a * c^2 < b * c^2

def prop_q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

theorem problem_solution : (p ∨ ¬q) := sorry

end problem_solution_l1205_120561


namespace distribute_papers_l1205_120565

theorem distribute_papers (n m : ℕ) (h_n : n = 5) (h_m : m = 10) : 
  (m ^ n) = 100000 :=
by 
  rw [h_n, h_m]
  rfl

end distribute_papers_l1205_120565


namespace students_in_class_l1205_120539

theorem students_in_class (n : ℕ) (S : ℕ) (h_avg_students : S / n = 14) (h_avg_including_teacher : (S + 45) / (n + 1) = 15) : n = 30 :=
by
  sorry

end students_in_class_l1205_120539


namespace sum_of_numbers_given_average_l1205_120553

variable (average : ℝ) (n : ℕ) (sum : ℝ)

theorem sum_of_numbers_given_average (h1 : average = 4.1) (h2 : n = 6) (h3 : average = sum / n) :
  sum = 24.6 :=
by
  sorry

end sum_of_numbers_given_average_l1205_120553


namespace length_of_edge_l1205_120549

-- Define all necessary conditions
def is_quadrangular_pyramid (e : ℝ) : Prop :=
  (8 * e = 14.8)

-- State the main theorem which is the equivalent proof problem
theorem length_of_edge (e : ℝ) (h : is_quadrangular_pyramid e) : e = 1.85 :=
by
  sorry

end length_of_edge_l1205_120549


namespace range_of_f_l1205_120524

open Set

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

theorem range_of_f :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → 2 ≤ f x ∧ f x ≤ 3 :=
by
  intro x hx
  sorry

end range_of_f_l1205_120524


namespace samuel_has_five_birds_l1205_120545

theorem samuel_has_five_birds
  (birds_berries_per_day : ℕ)
  (total_berries_in_4_days : ℕ)
  (n_birds : ℕ)
  (h1 : birds_berries_per_day = 7)
  (h2 : total_berries_in_4_days = 140)
  (h3 : n_birds * birds_berries_per_day * 4 = total_berries_in_4_days) :
  n_birds = 5 := by
  sorry

end samuel_has_five_birds_l1205_120545


namespace border_collie_catches_ball_in_32_seconds_l1205_120567

noncomputable def time_to_catch_ball (v_ball : ℕ) (t_ball : ℕ) (v_collie : ℕ) : ℕ := 
  (v_ball * t_ball) / v_collie

theorem border_collie_catches_ball_in_32_seconds :
  time_to_catch_ball 20 8 5 = 32 :=
by
  sorry

end border_collie_catches_ball_in_32_seconds_l1205_120567


namespace correct_answer_l1205_120562

theorem correct_answer (x : ℝ) (h1 : 2 * x = 60) : x / 2 = 15 :=
by
  sorry

end correct_answer_l1205_120562


namespace problem_I_problem_II_l1205_120580

-- Question I
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 :=
by
  sorry

-- Question II
theorem problem_II (a : ℝ) : (∀ x : ℝ, |x - a| + |2 * x - 1| ≥ 2) ↔ (a ≤ -3/2 ∨ a ≥ 5/2) :=
by
  sorry

end problem_I_problem_II_l1205_120580


namespace yellow_marbles_count_l1205_120540

-- Definitions based on given conditions
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def black_marbles : ℕ := 1
def probability_black : ℚ := 1 / 28
def total_marbles : ℕ := 28

-- Problem statement to prove
theorem yellow_marbles_count :
  (total_marbles = blue_marbles + green_marbles + black_marbles + n) →
  (probability_black = black_marbles / total_marbles) →
  n = 12 :=
by
  intros; sorry

end yellow_marbles_count_l1205_120540
