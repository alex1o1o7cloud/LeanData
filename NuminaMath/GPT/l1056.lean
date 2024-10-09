import Mathlib

namespace find_real_x_l1056_105633

theorem find_real_x (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end find_real_x_l1056_105633


namespace Taso_riddles_correct_l1056_105699

-- Definitions based on given conditions
def Josh_riddles : ℕ := 8
def Ivory_riddles : ℕ := Josh_riddles + 4
def Taso_riddles : ℕ := 2 * Ivory_riddles

-- The theorem to prove
theorem Taso_riddles_correct : Taso_riddles = 24 := by
  sorry

end Taso_riddles_correct_l1056_105699


namespace expression_zero_iff_x_eq_three_l1056_105618

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) → ((x^2 - 6 * x + 9 = 0) ↔ (x = 3)) :=
by
  sorry

end expression_zero_iff_x_eq_three_l1056_105618


namespace find_sequence_l1056_105607

noncomputable def seq (a : ℕ → ℝ) :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = (n / (n + 1)) * (a n + 1))

theorem find_sequence {a : ℕ → ℝ} (h : seq a) :
  ∀ n, a n = (n - 1) / 2 :=
sorry

end find_sequence_l1056_105607


namespace inequality_proof_l1056_105644

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 := 
by
  sorry

end inequality_proof_l1056_105644


namespace cube_root_simplification_l1056_105608

theorem cube_root_simplification (c d : ℕ) (h1 : c = 3) (h2 : d = 100) : c + d = 103 :=
by
  sorry

end cube_root_simplification_l1056_105608


namespace complete_square_form_l1056_105621

theorem complete_square_form {a h k : ℝ} :
  ∀ x, (x^2 - 5 * x) = a * (x - h)^2 + k → k = -25 / 4 :=
by
  intro x
  intro h_eq
  sorry

end complete_square_form_l1056_105621


namespace delores_money_left_l1056_105630

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l1056_105630


namespace equation_solutions_l1056_105622

theorem equation_solutions (m n x y : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  x^n + y^n = 3^m ↔ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2) ∨ (x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) :=
by
  sorry -- proof to be implemented

end equation_solutions_l1056_105622


namespace divisor_of_form_4k_minus_1_l1056_105631

theorem divisor_of_form_4k_minus_1
  (n : ℕ) (hn1 : Odd n) (hn_pos : 0 < n)
  (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (1 / (x : ℚ) + 1 / (y : ℚ) = 4 / n)) :
  ∃ k : ℕ, ∃ d, d ∣ n ∧ d = 4 * k - 1 ∧ k ∈ Set.Ici 1 :=
sorry

end divisor_of_form_4k_minus_1_l1056_105631


namespace max_area_trapezoid_l1056_105658

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid_l1056_105658


namespace find_A_B_l1056_105698

theorem find_A_B :
  ∀ (A B : ℝ), (∀ (x : ℝ), 1 < x → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  (A = 0) ∧ (B = 1) :=
by
  sorry

end find_A_B_l1056_105698


namespace area_of_square_field_l1056_105693

theorem area_of_square_field (x : ℝ) 
  (h₁ : 1.10 * (4 * x - 2) = 732.6) : 
  x = 167 → x ^ 2 = 27889 := by
  sorry

end area_of_square_field_l1056_105693


namespace train_crossing_time_l1056_105614

def speed := 60 -- in km/hr
def length := 300 -- in meters
def speed_in_m_per_s := (60 * 1000) / 3600 -- converting speed from km/hr to m/s
def expected_time := 18 -- in seconds

theorem train_crossing_time :
  (300 / (speed_in_m_per_s)) = expected_time :=
sorry

end train_crossing_time_l1056_105614


namespace find_a_l1056_105610

theorem find_a (a : ℝ) (h_pos : 0 < a) 
  (prob : (2 / a) = (1 / 3)) : a = 6 :=
by sorry

end find_a_l1056_105610


namespace chess_group_players_l1056_105683

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by {
  sorry
}

end chess_group_players_l1056_105683


namespace sequence_pattern_l1056_105615

theorem sequence_pattern (a b c : ℝ) (h1 : a = 19.8) (h2 : b = 18.6) (h3 : c = 17.4) 
  (h4 : ∀ n, n = a ∨ n = b ∨ n = c ∨ n = 16.2 ∨ n = 15) 
  (H : ∀ x y, (y = x - 1.2) → 
    (x = a ∨ x = b ∨ x = c ∨ y = 16.2 ∨ y = 15)) :
  (16.2 = c - 1.2) ∧ (15 = (c - 1.2) - 1.2) :=
by
  sorry

end sequence_pattern_l1056_105615


namespace haley_magazines_l1056_105672

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) (total_magazines : ℕ) :
  boxes = 7 →
  magazines_per_box = 9 →
  total_magazines = boxes * magazines_per_box →
  total_magazines = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end haley_magazines_l1056_105672


namespace calculate_z_l1056_105692

-- Given conditions
def equally_spaced : Prop := true -- assume equally spaced markings do exist
def total_distance : ℕ := 35
def number_of_steps : ℕ := 7
def step_length : ℕ := total_distance / number_of_steps
def starting_point : ℕ := 10
def steps_forward : ℕ := 4

-- Theorem to prove
theorem calculate_z (h1 : equally_spaced)
(h2 : step_length = 5)
: starting_point + (steps_forward * step_length) = 30 :=
by sorry

end calculate_z_l1056_105692


namespace matrix_product_is_zero_l1056_105624

-- Define the two matrices
def A (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -c], ![-d, 0, b], ![c, -b, 0]]

def B (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, b * d, c * d], ![b * d, b^2, b * c], ![c * d, b * c, c^2]]

-- Define the zero matrix
def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

-- The theorem to prove
theorem matrix_product_is_zero (b c d : ℝ) : A b c d * B b c d = zero_matrix :=
by sorry

end matrix_product_is_zero_l1056_105624


namespace four_people_fill_pool_together_in_12_minutes_l1056_105642

def combined_pool_time (j s t e : ℕ) : ℕ := 
  1 / ((1 / j) + (1 / s) + (1 / t) + (1 / e))

theorem four_people_fill_pool_together_in_12_minutes : 
  ∀ (j s t e : ℕ), j = 30 → s = 45 → t = 90 → e = 60 → combined_pool_time j s t e = 12 := 
by 
  intros j s t e h_j h_s h_t h_e
  unfold combined_pool_time
  rw [h_j, h_s, h_t, h_e]
  have r1 : 1 / 30 = 1 / 30 := rfl
  have r2 : 1 / 45 = 1 / 45 := rfl
  have r3 : 1 / 90 = 1 / 90 := rfl
  have r4 : 1 / 60 = 1 / 60 := rfl
  rw [r1, r2, r3, r4]
  norm_num
  sorry

end four_people_fill_pool_together_in_12_minutes_l1056_105642


namespace tank_water_after_rain_final_l1056_105682

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end tank_water_after_rain_final_l1056_105682


namespace solve_for_x_l1056_105653

theorem solve_for_x (x : ℕ) : (8^3 + 8^3 + 8^3 + 8^3 = 2^x) → x = 11 :=
by
  intro h
  sorry

end solve_for_x_l1056_105653


namespace minimum_production_volume_to_avoid_loss_l1056_105686

open Real

-- Define the cost function
def cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * (x ^ 2)

-- Define the revenue function
def revenue (x : ℕ) : ℝ := 25 * x

-- Condition: 0 < x < 240 and x ∈ ℕ (naturals greater than 0)
theorem minimum_production_volume_to_avoid_loss (x : ℕ) (hx1 : 0 < x) (hx2 : x < 240) (hx3 : x ∈ (Set.Ioi 0)) :
  revenue x ≥ cost x ↔ x ≥ 150 :=
by
  sorry

end minimum_production_volume_to_avoid_loss_l1056_105686


namespace area_of_paper_is_500_l1056_105665

-- Define the width and length of the rectangular drawing paper
def width := 25
def length := 20

-- Define the formula for the area of a rectangle
def area (w : Nat) (l : Nat) : Nat := w * l

-- Prove that the area of the paper is 500 square centimeters
theorem area_of_paper_is_500 : area width length = 500 := by
  -- placeholder for the proof
  sorry

end area_of_paper_is_500_l1056_105665


namespace find_g_l1056_105670

theorem find_g (g : ℕ) (h : g > 0) :
  (1 / 3) = ((4 + g * (g - 1)) / ((g + 4) * (g + 3))) → g = 5 :=
by
  intro h_eq
  sorry 

end find_g_l1056_105670


namespace isosceles_perimeter_l1056_105619

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_perimeter
  (k : ℝ)
  (a b : ℝ)
  (h1 : 4 = a)
  (h2 : k * b^2 - (k + 8) * b + 8 = 0)
  (h3 : k ≠ 0)
  (h4 : is_triangle 4 a a) : a + 4 + a = 9 :=
sorry

end isosceles_perimeter_l1056_105619


namespace combined_total_capacity_l1056_105625

theorem combined_total_capacity (A B C : ℝ) 
  (hA : 0.35 * A + 48 = 3 / 4 * A)
  (hB : 0.45 * B + 36 = 0.95 * B)
  (hC : 0.20 * C - 24 = 0.10 * C) :
  A + B + C = 432 := 
by 
  sorry

end combined_total_capacity_l1056_105625


namespace arrangement_count_SUCCESS_l1056_105634

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l1056_105634


namespace area_of_fourth_rectangle_l1056_105609

theorem area_of_fourth_rectangle (a b c d : ℕ) (h1 : a = 18) (h2 : b = 27) (h3 : c = 12) :
d = 93 :=
by
  -- Problem reduces to showing that d equals 93 using the given h1, h2, h3
  sorry

end area_of_fourth_rectangle_l1056_105609


namespace calculate_expression_l1056_105680

theorem calculate_expression :
  ( ( (1/6) - (1/8) + (1/9) ) / ( (1/3) - (1/4) + (1/5) ) ) * 3 = 55 / 34 :=
by
  sorry

end calculate_expression_l1056_105680


namespace ratio_of_AC_to_BD_l1056_105606

theorem ratio_of_AC_to_BD (A B C D : ℝ) (AB BC AD AC BD : ℝ) 
  (h1 : AB = 2) (h2 : BC = 5) (h3 : AD = 14) (h4 : AC = AB + BC) (h5 : BD = AD - AB) :
  AC / BD = 7 / 12 := by
  sorry

end ratio_of_AC_to_BD_l1056_105606


namespace platform_length_l1056_105661

noncomputable def train_length := 420 -- length of the train in meters
noncomputable def time_to_cross_platform := 60 -- time to cross the platform in seconds
noncomputable def time_to_cross_pole := 30 -- time to cross the signal pole in seconds

theorem platform_length :
  ∃ L, L = 420 ∧ train_length / time_to_cross_pole = train_length / time_to_cross_platform * (train_length + L) / time_to_cross_platform :=
by
  use 420
  sorry

end platform_length_l1056_105661


namespace weight_of_person_replaced_l1056_105629

theorem weight_of_person_replaced (W : ℝ) (old_avg_weight : ℝ) (new_avg_weight : ℝ)
  (h_avg_increase : new_avg_weight = old_avg_weight + 1.5) (new_person_weight : ℝ) :
  ∃ (person_replaced_weight : ℝ), new_person_weight = 77 ∧ old_avg_weight = W / 8 ∧
  new_avg_weight = (W - person_replaced_weight + 77) / 8 ∧ person_replaced_weight = 65 := by
    sorry

end weight_of_person_replaced_l1056_105629


namespace birds_problem_l1056_105678

-- Define the initial number of birds and the total number of birds as given conditions.
def initial_birds : ℕ := 2
def total_birds : ℕ := 6

-- Define the number of new birds that came to join.
def new_birds : ℕ := total_birds - initial_birds

-- State the theorem to be proved, asserting that the number of new birds is 4.
theorem birds_problem : new_birds = 4 := 
by
  -- required proof goes here
  sorry

end birds_problem_l1056_105678


namespace net_pay_is_correct_l1056_105641

-- Define the gross pay and taxes paid as constants
def gross_pay : ℕ := 450
def taxes_paid : ℕ := 135

-- Define net pay as a function of gross pay and taxes paid
def net_pay (gross : ℕ) (taxes : ℕ) : ℕ := gross - taxes

-- The proof statement
theorem net_pay_is_correct : net_pay gross_pay taxes_paid = 315 := by
  sorry -- The proof goes here

end net_pay_is_correct_l1056_105641


namespace train_speed_kmph_l1056_105639

def train_length : ℝ := 360
def bridge_length : ℝ := 140
def time_to_pass : ℝ := 40
def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

theorem train_speed_kmph : mps_to_kmph ((train_length + bridge_length) / time_to_pass) = 45 := 
by {
  sorry
}

end train_speed_kmph_l1056_105639


namespace number_of_correct_propositions_l1056_105646

def f (x b c : ℝ) := x * |x| + b * x + c

def proposition1 (b : ℝ) : Prop :=
  ∀ (x : ℝ), f x b 0 = -f (-x) b 0

def proposition2 (c : ℝ) : Prop :=
  c > 0 → ∃ (x : ℝ), ∀ (y : ℝ), f y 0 c = 0 → y = x

def proposition3 (b c : ℝ) : Prop :=
  ∀ (x : ℝ), f x b c = f (-x) b c + 2 * c

def proposition4 (b c : ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ : ℝ), f x₁ b c = 0 → f x₂ b c = 0 → f x₃ b c = 0 → x₁ = x₂ ∨ x₂ = x₃ ∨ x₁ = x₃

theorem number_of_correct_propositions (b c : ℝ) : 
  1 + (if c > 0 then 1 else 0) + 1 + 0 = 3 :=
  sorry

end number_of_correct_propositions_l1056_105646


namespace simple_interest_principal_l1056_105637

theorem simple_interest_principal (R : ℝ) (P : ℝ) (h : P * 7 * (R + 2) / 100 = P * 7 * R / 100 + 140) : P = 1000 :=
by
  sorry

end simple_interest_principal_l1056_105637


namespace Felix_distance_proof_l1056_105617

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end Felix_distance_proof_l1056_105617


namespace total_edge_length_of_parallelepiped_l1056_105645

/-- Kolya has 440 identical cubes with a side length of 1 cm.
Kolya constructs a rectangular parallelepiped from these cubes 
and all edges have lengths of at least 5 cm. Prove 
that the total length of all edges of the rectangular parallelepiped is 96 cm. -/
theorem total_edge_length_of_parallelepiped {a b c : ℕ} 
  (h1 : a * b * c = 440) 
  (h2 : a ≥ 5) 
  (h3 : b ≥ 5) 
  (h4 : c ≥ 5) : 
  4 * (a + b + c) = 96 :=
sorry

end total_edge_length_of_parallelepiped_l1056_105645


namespace difference_between_percent_and_fraction_l1056_105616

-- Define the number
def num : ℕ := 140

-- Define the percentage and fraction calculations
def percent_65 (n : ℕ) : ℕ := (65 * n) / 100
def fraction_4_5 (n : ℕ) : ℕ := (4 * n) / 5

-- Define the problem's conditions and the required proof
theorem difference_between_percent_and_fraction : 
  percent_65 num ≤ fraction_4_5 num ∧ (fraction_4_5 num - percent_65 num = 21) :=
by
  sorry

end difference_between_percent_and_fraction_l1056_105616


namespace solve_inequality_l1056_105603

theorem solve_inequality (x: ℝ) : (25 - 5 * Real.sqrt 3) ≤ x ∧ x ≤ (25 + 5 * Real.sqrt 3) ↔ x ^ 2 - 50 * x + 575 ≤ 25 :=
by
  sorry

end solve_inequality_l1056_105603


namespace iced_coffee_cost_correct_l1056_105612

-- Definitions based on the conditions 
def coffee_cost_per_day (iced_coffee_cost : ℝ) : ℝ := 3 + iced_coffee_cost
def total_spent (days : ℕ) (iced_coffee_cost : ℝ) : ℝ := days * coffee_cost_per_day iced_coffee_cost

-- Proof statement
theorem iced_coffee_cost_correct (iced_coffee_cost : ℝ) (h : total_spent 20 iced_coffee_cost = 110) : iced_coffee_cost = 2.5 :=
by
  sorry

end iced_coffee_cost_correct_l1056_105612


namespace binom_10_2_eq_45_l1056_105628

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l1056_105628


namespace operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l1056_105674

-- Define what an even integer is
def is_even (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * k

-- Define the operations
def add_four (a : ℤ) := a + 4
def subtract_six (a : ℤ) := a - 6
def multiply_by_eight (a : ℤ) := a * 8
def divide_by_two_add_two (a : ℤ) := a / 2 + 2
def average_with_ten (a : ℤ) := (a + 10) / 2

-- The proof statements
theorem operation_1_even_if_input_even (a : ℤ) (h : is_even a) : is_even (add_four a) := sorry
theorem operation_2_even_if_input_even (a : ℤ) (h : is_even a) : is_even (subtract_six a) := sorry
theorem operation_3_even_if_input_even (a : ℤ) (h : is_even a) : is_even (multiply_by_eight a) := sorry
theorem operation_4_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (divide_by_two_add_two a) := sorry
theorem operation_5_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (average_with_ten a) := sorry

end operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l1056_105674


namespace no_nat_x_y_square_l1056_105679

theorem no_nat_x_y_square (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := 
by 
  sorry

end no_nat_x_y_square_l1056_105679


namespace problem_statement_l1056_105635

theorem problem_statement (k x₁ x₂ : ℝ) (hx₁x₂ : x₁ < x₂)
  (h_eq : ∀ x : ℝ, x^2 - (k - 3) * x + (k + 4) = 0) 
  (P : ℝ) (hP : P ≠ 0) 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (hacute : ∀ A B : ℝ, A = x₁ ∧ B = x₂ ∧ A < 0 ∧ B > 0) :
  k < -4 ∧ α ≠ β ∧ α < β := 
sorry

end problem_statement_l1056_105635


namespace screws_weight_l1056_105660

theorem screws_weight (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 319) 
  (h2 : 2 * x + 3 * y = 351) : 
  x = 51 ∧ y = 83 :=
by 
  sorry

end screws_weight_l1056_105660


namespace wage_difference_l1056_105691

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.2

theorem wage_difference : manager_wage - chef_wage = 3.40 := 
by
  sorry

end wage_difference_l1056_105691


namespace seventh_term_of_arithmetic_sequence_l1056_105636

variable (a d : ℕ)

theorem seventh_term_of_arithmetic_sequence (h1 : 5 * a + 10 * d = 15) (h2 : a + 3 * d = 4) : a + 6 * d = 7 := 
by
  sorry

end seventh_term_of_arithmetic_sequence_l1056_105636


namespace value_of_fraction_l1056_105604

variables {a_1 q : ℝ}

-- Define the conditions and the mathematical equivalent of the problem.
def geometric_sequence (a_1 q : ℝ) (h_pos : a_1 > 0 ∧ q > 0) :=
  2 * a_1 + a_1 * q = a_1 * q^2

theorem value_of_fraction (h_pos : a_1 > 0 ∧ q > 0) (h_geom : geometric_sequence a_1 q h_pos) :
  (a_1 * q^3 + a_1 * q^4) / (a_1 * q^2 + a_1 * q^3) = 2 :=
sorry

end value_of_fraction_l1056_105604


namespace cement_total_l1056_105668

-- Defining variables for the weights of cement
def weight_self : ℕ := 215
def weight_son : ℕ := 137

-- Defining the function that calculates the total weight of the cement
def total_weight (a b : ℕ) : ℕ := a + b

-- Theorem statement: Proving the total cement weight is 352 lbs
theorem cement_total : total_weight weight_self weight_son = 352 :=
by
  sorry

end cement_total_l1056_105668


namespace bags_sold_on_Thursday_l1056_105675

theorem bags_sold_on_Thursday 
    (total_bags : ℕ) (sold_Monday : ℕ) (sold_Tuesday : ℕ) (sold_Wednesday : ℕ) (sold_Friday : ℕ) (percent_not_sold : ℕ) :
    total_bags = 600 →
    sold_Monday = 25 →
    sold_Tuesday = 70 →
    sold_Wednesday = 100 →
    sold_Friday = 145 →
    percent_not_sold = 25 →
    ∃ (sold_Thursday : ℕ), sold_Thursday = 110 :=
by
  sorry

end bags_sold_on_Thursday_l1056_105675


namespace triangle_inequality_squared_l1056_105652

theorem triangle_inequality_squared {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := sorry

end triangle_inequality_squared_l1056_105652


namespace beth_total_crayons_l1056_105681

theorem beth_total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) 
  (h1 : packs = 8) (h2 : crayons_per_pack = 20) (h3 : extra_crayons = 15) :
  packs * crayons_per_pack + extra_crayons = 175 :=
by
  sorry

end beth_total_crayons_l1056_105681


namespace wheel_distance_l1056_105654

noncomputable def diameter : ℝ := 9
noncomputable def revolutions : ℝ := 18.683651804670912
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def circumference (d : ℝ) : ℝ := pi_approx * d
noncomputable def distance (r : ℝ) (c : ℝ) : ℝ := r * c

theorem wheel_distance : distance revolutions (circumference diameter) = 528.219 :=
by
  unfold distance circumference diameter revolutions pi_approx
  -- Here we would perform the calculation and show that the result is approximately 528.219
  sorry

end wheel_distance_l1056_105654


namespace range_of_k_l1056_105690

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x^2 - 2 * x + k^2 - 3 > 0) -> (k > 2 ∨ k < -2) :=
by
  sorry

end range_of_k_l1056_105690


namespace min_value_of_quadratic_l1056_105620

theorem min_value_of_quadratic (x y s : ℝ) (h : x + y = s) : 
  ∃ x y, 3 * x^2 + 2 * y^2 = 6 * s^2 / 5 := sorry

end min_value_of_quadratic_l1056_105620


namespace youngest_child_is_five_l1056_105685

-- Define the set of prime numbers
def is_prime (n: ℕ) := n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the ages of the children
def youngest_child_age (x: ℕ) : Prop :=
  is_prime x ∧
  is_prime (x + 2) ∧
  is_prime (x + 6) ∧
  is_prime (x + 8) ∧
  is_prime (x + 12) ∧
  is_prime (x + 14)

-- The main theorem stating the age of the youngest child
theorem youngest_child_is_five : ∃ x: ℕ, youngest_child_age x ∧ x = 5 :=
  sorry

end youngest_child_is_five_l1056_105685


namespace tennis_ball_ratio_problem_solution_l1056_105659

def tennis_ball_ratio_problem (total_balls ordered_white ordered_yellow dispatched_yellow extra_yellow : ℕ) : Prop :=
  total_balls = 114 ∧ 
  ordered_white = total_balls / 2 ∧ 
  ordered_yellow = total_balls / 2 ∧ 
  dispatched_yellow = ordered_yellow + extra_yellow → 
  (ordered_white / dispatched_yellow = 57 / 107)

theorem tennis_ball_ratio_problem_solution :
  tennis_ball_ratio_problem 114 57 57 107 50 := by 
  sorry

end tennis_ball_ratio_problem_solution_l1056_105659


namespace polygon_sides_sum_l1056_105689

theorem polygon_sides_sum (triangle_hexagon_sum : ℕ) (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (h1 : triangle_hexagon_sum = 1260) 
  (h2 : triangle_sides = 3) 
  (h3 : hexagon_sides = 6) 
  (convex : ∀ n, 3 <= n) : 
  triangle_sides + hexagon_sides + 4 = 13 :=
by 
  sorry

end polygon_sides_sum_l1056_105689


namespace find_a_value_l1056_105648

theorem find_a_value :
  (∀ (x y : ℝ), (x = 1.5 → y = 8 → x * y = 12) ∧ 
               (x = 2 → y = 6 → x * y = 12) ∧ 
               (x = 3 → y = 4 → x * y = 12)) →
  ∃ (a : ℝ), (5 * a = 12 ∧ a = 2.4) :=
by
  sorry

end find_a_value_l1056_105648


namespace range_of_a_for_inequality_l1056_105662

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end range_of_a_for_inequality_l1056_105662


namespace correct_statements_l1056_105688

variable (a_1 a_2 b_1 b_2 : ℝ)

def ellipse1 := ∀ x y : ℝ, x^2 / a_1^2 + y^2 / b_1^2 = 1
def ellipse2 := ∀ x y : ℝ, x^2 / a_2^2 + y^2 / b_2^2 = 1

axiom a1_pos : a_1 > 0
axiom b1_pos : b_1 > 0
axiom a2_gt_b2_pos : a_2 > b_2 ∧ b_2 > 0
axiom same_foci : a_1^2 - b_1^2 = a_2^2 - b_2^2
axiom a1_gt_a2 : a_1 > a_2

theorem correct_statements : 
  (¬(∃ x y, (x^2 / a_1^2 + y^2 / b_1^2 = 1) ∧ (x^2 / a_2^2 + y^2 / b_2^2 = 1))) ∧ 
  (a_1^2 - a_2^2 = b_1^2 - b_2^2) :=
by 
  sorry

end correct_statements_l1056_105688


namespace Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l1056_105632

structure Person :=
  (name : String)
  (age : Nat)
  (location : String)
  (occupation : String)

def kate : Person := {name := "Kate Hashimoto", age := 30, location := "New York", occupation := "CPA"}

-- Conditions
def lives_on_15_dollars_a_month (p : Person) : Prop := p = kate → true
def dumpster_diving (p : Person) : Prop := p = kate → true
def upscale_stores_discard_good_items : Prop := true
def frugal_habits (p : Person) : Prop := p = kate → true

-- Proof
theorem Kate_relies_on_dumpster_diving : lives_on_15_dollars_a_month kate ∧ dumpster_diving kate → true := 
by sorry

theorem Upscale_stores_discard_items : upscale_stores_discard_good_items → true := 
by sorry

theorem Kate_frugal_habits : frugal_habits kate → true := 
by sorry

end Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l1056_105632


namespace max_value_of_expression_l1056_105696

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression_l1056_105696


namespace find_L_l1056_105687

theorem find_L (RI G SP T M N : ℝ) (h1 : RI + G + SP = 50) (h2 : RI + T + M = 63) (h3 : G + T + SP = 25) 
(h4 : SP + M = 13) (h5 : M + RI = 48) (h6 : N = 1) :
  ∃ L : ℝ, L * M * T + SP * RI * N * G = 2023 ∧ L = 341 / 40 := 
by
  sorry

end find_L_l1056_105687


namespace p_sufficient_not_necessary_for_q_l1056_105650

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- State the theorem that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l1056_105650


namespace case1_DC_correct_case2_DC_correct_l1056_105602

-- Case 1
theorem case1_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 10) (hAD : AD = 4)
  (hHM : HM = 6 / 5) (hBD : BD = 2 * Real.sqrt 21) (hDH : DH = 4 * Real.sqrt 21 / 5)
  (hMD : MD = 6 * (Real.sqrt 21 - 1) / 5):
  (BD - HM : ℝ) == (8 * Real.sqrt 21 - 12) / 5 :=
by {
  sorry
}

-- Case 2
theorem case2_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 8 * Real.sqrt 2) (hAD : AD = 4)
  (hHM : HM = Real.sqrt 2) (hBD : BD = 4 * Real.sqrt 7) (hDH : DH = Real.sqrt 14)
  (hMD : MD = Real.sqrt 14 - Real.sqrt 2):
  (BD - HM : ℝ) == 2 * Real.sqrt 14 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end case1_DC_correct_case2_DC_correct_l1056_105602


namespace math_problem_l1056_105656

theorem math_problem (n : ℕ) (h : n > 0) : 
  1957 ∣ (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n)) :=
sorry

end math_problem_l1056_105656


namespace find_present_age_of_eldest_l1056_105651

noncomputable def eldest_present_age (x : ℕ) : ℕ :=
  8 * x

theorem find_present_age_of_eldest :
  ∃ x : ℕ, 20 * x - 21 = 59 ∧ eldest_present_age x = 32 :=
by
  sorry

end find_present_age_of_eldest_l1056_105651


namespace square_division_l1056_105600

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end square_division_l1056_105600


namespace rectangle_x_value_l1056_105605

theorem rectangle_x_value (x : ℝ) (h : (4 * x) * (x + 7) = 2 * (4 * x) + 2 * (x + 7)) : x = 0.675 := 
sorry

end rectangle_x_value_l1056_105605


namespace unique_real_solution_l1056_105611

theorem unique_real_solution : ∃ x : ℝ, (∀ t : ℝ, x^2 - t * x + 36 = 0 ∧ x^2 - 8 * x + t = 0) ∧ x = 3 :=
by
  sorry

end unique_real_solution_l1056_105611


namespace adam_age_l1056_105640

variable (E A : ℕ)

namespace AgeProof

theorem adam_age (h1 : A = E - 5) (h2 : E + 1 = 3 * (A - 4)) : A = 9 :=
by
  sorry
end AgeProof

end adam_age_l1056_105640


namespace area_of_shaded_region_l1056_105669

noncomputable def shaded_area (length_in_feet : ℝ) (diameter : ℝ) : ℝ :=
  let length_in_inches := length_in_feet * 12
  let radius := diameter / 2
  let num_semicircles := length_in_inches / diameter
  let num_full_circles := num_semicircles / 2
  let area := num_full_circles * (radius ^ 2 * Real.pi)
  area

theorem area_of_shaded_region : shaded_area 1.5 3 = 13.5 * Real.pi :=
by
  sorry

end area_of_shaded_region_l1056_105669


namespace sum_of_perimeters_of_squares_l1056_105627

theorem sum_of_perimeters_of_squares
  (x y : ℝ)
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12*Real.sqrt 26 := by
  sorry

end sum_of_perimeters_of_squares_l1056_105627


namespace domain_f_2x_minus_1_l1056_105601

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f (2 * x - 1) = y) :=
by
  intro h
  sorry

end domain_f_2x_minus_1_l1056_105601


namespace reporters_local_politics_percentage_l1056_105623

theorem reporters_local_politics_percentage
  (T : ℕ) -- Total number of reporters
  (P : ℝ) -- Percentage of reporters covering politics
  (h1 : 30 / 100 * (P / 100) * T = (P / 100 - 0.7 * (P / 100)) * T)
  (h2 : 92.85714285714286 / 100 * T = (1 - P / 100) * T):
  (0.7 * (P / 100) * T) / T = 5 / 100 :=
by
  sorry

end reporters_local_politics_percentage_l1056_105623


namespace number_of_correct_propositions_l1056_105663

variable (Ω : Type) (R : Type) [Nonempty Ω] [Nonempty R]

-- Definitions of the conditions
def carsPassingIntersection (t : ℝ) : Ω → ℕ := sorry
def passengersInWaitingRoom (t : ℝ) : Ω → ℕ := sorry
def maximumFlowRiverEachYear : Ω → ℝ := sorry
def peopleExitingTheater (t : ℝ) : Ω → ℕ := sorry

-- Statement to prove the number of correct propositions
theorem number_of_correct_propositions : 4 = 4 := sorry

end number_of_correct_propositions_l1056_105663


namespace positive_difference_volumes_l1056_105673

open Real

noncomputable def charlies_height := 12
noncomputable def charlies_circumference := 10
noncomputable def danas_height := 8
noncomputable def danas_circumference := 10

theorem positive_difference_volumes (hC : ℝ := charlies_height) (CC : ℝ := charlies_circumference)
                                   (hD : ℝ := danas_height) (CD : ℝ := danas_circumference) :
    (π * (π * ((CD / (2 * π)) ^ 2) * hD - π * ((CC / (2 * π)) ^ 2) * hC)) = 100 :=
by
  have rC := CC / (2 * π)
  have VC := π * (rC ^ 2) * hC
  have rD := CD / (2 * π)
  have VD := π * (rD ^ 2) * hD
  sorry

end positive_difference_volumes_l1056_105673


namespace special_four_digit_numbers_l1056_105655

noncomputable def count_special_four_digit_numbers : Nat :=
  -- The task is to define the number of four-digit numbers formed using the digits {0, 1, 2, 3, 4}
  -- that contain the digit 0 and have exactly two digits repeating
  144

theorem special_four_digit_numbers : count_special_four_digit_numbers = 144 := by
  sorry

end special_four_digit_numbers_l1056_105655


namespace octagon_diagonals_l1056_105647

def num_sides := 8

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals : num_diagonals num_sides = 20 :=
by
  sorry

end octagon_diagonals_l1056_105647


namespace value_of_expression_l1056_105638

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by
  sorry

end value_of_expression_l1056_105638


namespace scale_of_diagram_l1056_105671

-- Definitions for the given conditions
def length_miniature_component_mm : ℕ := 4
def length_diagram_cm : ℕ := 8
def length_diagram_mm : ℕ := 80  -- Converted length from cm to mm

-- The problem statement
theorem scale_of_diagram :
  (length_diagram_mm : ℕ) / (length_miniature_component_mm : ℕ) = 20 :=
by
  have conversion : length_diagram_mm = length_diagram_cm * 10 := by sorry
  -- conversion states the formula for converting cm to mm
  have ratio : length_diagram_mm / length_miniature_component_mm = 80 / 4 := by sorry
  -- ratio states the initial computed ratio
  exact sorry

end scale_of_diagram_l1056_105671


namespace trig_problem_l1056_105684

theorem trig_problem (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

end trig_problem_l1056_105684


namespace fraction_to_decimal_l1056_105657

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l1056_105657


namespace ellipse_properties_l1056_105666

noncomputable def a_square : ℝ := 2
noncomputable def b_square : ℝ := 9 / 8
noncomputable def c_square : ℝ := a_square - b_square
noncomputable def c : ℝ := Real.sqrt c_square
noncomputable def distance_between_foci : ℝ := 2 * c
noncomputable def eccentricity : ℝ := c / Real.sqrt a_square

theorem ellipse_properties :
  (distance_between_foci = Real.sqrt 14) ∧ (eccentricity = Real.sqrt 7 / 4) := by
  sorry

end ellipse_properties_l1056_105666


namespace measure_of_angle_l1056_105649

theorem measure_of_angle (x : ℝ) 
  (h₁ : 180 - x = 3 * x - 10) : x = 47.5 :=
by 
  sorry

end measure_of_angle_l1056_105649


namespace machine_A_produces_40_percent_l1056_105664

theorem machine_A_produces_40_percent (p : ℝ) : 
  (0 < p ∧ p < 1 ∧
  (0.0156 = p * 0.009 + (1 - p) * 0.02)) → 
  p = 0.4 :=
by 
  intro h
  sorry

end machine_A_produces_40_percent_l1056_105664


namespace solve_system_l1056_105697

theorem solve_system :
  ∃ x y : ℝ, (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧ 
              (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0) ∧ 
              (x = -3 ∧ y = -1) :=
  sorry

end solve_system_l1056_105697


namespace distance_between_sasha_and_kolya_when_sasha_finished_l1056_105626

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l1056_105626


namespace integer_implies_perfect_square_l1056_105643

theorem integer_implies_perfect_square (n : ℕ) (h : ∃ m : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = m) :
  ∃ k : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = (k ^ 2) :=
by
  sorry

end integer_implies_perfect_square_l1056_105643


namespace nat_divisible_by_five_l1056_105613

theorem nat_divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  have h₀ : ¬ ((5 ∣ a) ∨ (5 ∣ b)) → ¬ (5 ∣ (a * b)) := sorry
  -- Proof by contradiction steps go here
  sorry

end nat_divisible_by_five_l1056_105613


namespace Mike_additional_money_needed_proof_l1056_105667

-- Definitions of conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_has_percentage : ℝ := 0.40

-- Definitions of intermediate calculations
def discounted_phone_cost : ℝ := phone_cost * (1 - phone_discount)
def discounted_smartwatch_cost : ℝ := smartwatch_cost * (1 - smartwatch_discount)
def total_cost_before_tax : ℝ := discounted_phone_cost + discounted_smartwatch_cost
def total_tax : ℝ := total_cost_before_tax * sales_tax
def total_cost_after_tax : ℝ := total_cost_before_tax + total_tax
def mike_has_amount : ℝ := total_cost_after_tax * mike_has_percentage
def additional_money_needed : ℝ := total_cost_after_tax - mike_has_amount

-- Theorem statement
theorem Mike_additional_money_needed_proof :
  additional_money_needed = 1023.99 :=
by sorry

end Mike_additional_money_needed_proof_l1056_105667


namespace replace_asterisk_l1056_105676

theorem replace_asterisk (x : ℕ) (h : (42 / 21) * (42 / x) = 1) : x = 84 := by
  sorry

end replace_asterisk_l1056_105676


namespace find_a2_l1056_105694

theorem find_a2 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * (a n - 1))
  (h2 : S 1 = a 1)
  (h3 : S 2 = a 1 + a 2) :
  a 2 = 4 :=
sorry

end find_a2_l1056_105694


namespace find_value_of_s_l1056_105677

theorem find_value_of_s
  (a b c w s p : ℕ)
  (h₁ : a + b = w)
  (h₂ : w + c = s)
  (h₃ : s + a = p)
  (h₄ : b + c + p = 16) :
  s = 8 :=
sorry

end find_value_of_s_l1056_105677


namespace line_tangent_to_ellipse_l1056_105695

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * (m * x + 2)^2 = 3) ∧ 
  (∀ x1 x2 : ℝ, (2 + 3 * m^2) * x1^2 + 12 * m * x1 + 9 = 0 ∧ 
                (2 + 3 * m^2) * x2^2 + 12 * m * x2 + 9 = 0 → x1 = x2) ↔ m^2 = 2 := 
sorry

end line_tangent_to_ellipse_l1056_105695
