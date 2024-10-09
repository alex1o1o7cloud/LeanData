import Mathlib

namespace eight_pow_n_over_three_eq_512_l91_9178

theorem eight_pow_n_over_three_eq_512 : 8^(9/3) = 512 :=
by
  -- sorry skips the proof
  sorry

end eight_pow_n_over_three_eq_512_l91_9178


namespace sum_a_t_l91_9162

theorem sum_a_t (a : ℝ) (t : ℝ) 
  (h₁ : a = 6)
  (h₂ : t = a^2 - 1) : a + t = 41 :=
by
  sorry

end sum_a_t_l91_9162


namespace necessary_but_not_sufficient_for_inequalities_l91_9146

theorem necessary_but_not_sufficient_for_inequalities (a b : ℝ) :
  (a + b > 4) ↔ (a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequalities_l91_9146


namespace find_temperature_on_December_25_l91_9131

theorem find_temperature_on_December_25 {f : ℕ → ℤ}
  (h_recurrence : ∀ n, f (n - 1) + f (n + 1) = f n)
  (h_initial1 : f 3 = 5)
  (h_initial2 : f 31 = 2) :
  f 25 = -3 :=
  sorry

end find_temperature_on_December_25_l91_9131


namespace cubes_sum_equiv_l91_9168

theorem cubes_sum_equiv (h : 2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) :
  3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675 := 
  sorry

end cubes_sum_equiv_l91_9168


namespace first_caller_to_win_all_prizes_is_900_l91_9136

-- Define the conditions: frequencies of win types
def every_25th_caller_wins_music_player (n : ℕ) : Prop := n % 25 = 0
def every_36th_caller_wins_concert_tickets (n : ℕ) : Prop := n % 36 = 0
def every_45th_caller_wins_backstage_passes (n : ℕ) : Prop := n % 45 = 0

-- Formalize the problem to prove
theorem first_caller_to_win_all_prizes_is_900 :
  ∃ n : ℕ, every_25th_caller_wins_music_player n ∧
           every_36th_caller_wins_concert_tickets n ∧
           every_45th_caller_wins_backstage_passes n ∧
           n = 900 :=
by {
  sorry
}

end first_caller_to_win_all_prizes_is_900_l91_9136


namespace emily_wrong_questions_l91_9177

variable (E F G H : ℕ)

theorem emily_wrong_questions (h1 : E + F + 4 = G + H) 
                             (h2 : E + H = F + G + 8) 
                             (h3 : G = 6) : 
                             E = 8 :=
sorry

end emily_wrong_questions_l91_9177


namespace price_relation_l91_9167

-- Defining the conditions
variable (TotalPrice : ℕ) (NumberOfPens : ℕ)
variable (total_price_val : TotalPrice = 24) (number_of_pens_val : NumberOfPens = 16)

-- Statement of the problem
theorem price_relation (y x : ℕ) (h_y : y = 3 / 2) : y = 3 / 2 * x := 
  sorry

end price_relation_l91_9167


namespace least_product_of_distinct_primes_greater_than_30_l91_9175

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

-- Define the problem
theorem least_product_of_distinct_primes_greater_than_30 :
  ∃ p q : ℕ, p ≠ q ∧ 30 < p ∧ 30 < q ∧ is_prime p ∧ is_prime q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_30_l91_9175


namespace g_eq_g_g_l91_9157

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem g_eq_g_g (x : ℝ) : 
  g (g x) = g x ↔ x = 2 + Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 + Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) := 
by
  sorry

end g_eq_g_g_l91_9157


namespace range_of_a_l91_9147

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → abs (2 * a - 1) ≤ abs (x + 1 / x)) →
  -1 / 2 ≤ a ∧ a ≤ 3 / 2 :=
by sorry

end range_of_a_l91_9147


namespace min_sum_length_perpendicular_chords_l91_9115

variables {p : ℝ} (h : p > 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * p * (x + p)

theorem min_sum_length_perpendicular_chords (h: p > 0) :
  ∃ (AB CD : ℝ), AB * CD = 1 → |AB| + |CD| = 16 * p := sorry

end min_sum_length_perpendicular_chords_l91_9115


namespace units_digit_product_l91_9171

theorem units_digit_product :
  ((734^99 + 347^83) % 10) * ((956^75 - 214^61) % 10) % 10 = 4 := by
  sorry

end units_digit_product_l91_9171


namespace largest_gcd_sum_1071_l91_9150

theorem largest_gcd_sum_1071 (x y: ℕ) (h1: x > 0) (h2: y > 0) (h3: x + y = 1071) : 
  ∃ d, d = Nat.gcd x y ∧ ∀ z, (z ∣ 1071 -> z ≤ d) := 
sorry

end largest_gcd_sum_1071_l91_9150


namespace children_neither_happy_nor_sad_l91_9134

-- conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10

-- proof problem
theorem children_neither_happy_nor_sad :
  total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l91_9134


namespace determine_c_for_inverse_l91_9143

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_c_for_inverse :
  (∀ x : ℝ, x ≠ 0 → f (f_inv x) c = x) ↔ c = 1 :=
sorry

end determine_c_for_inverse_l91_9143


namespace pqrs_product_l91_9160

noncomputable def product_of_area_and_perimeter :=
  let P := (1, 3)
  let Q := (4, 4)
  let R := (3, 1)
  let S := (0, 0)
  let side_length := Real.sqrt ((1 - 0)^2 * 4 + (3 - 0)^2 * 4)
  let area := side_length ^ 2
  let perimeter := 4 * side_length
  area * perimeter

theorem pqrs_product : product_of_area_and_perimeter = 208 * Real.sqrt 52 := 
  by 
    sorry

end pqrs_product_l91_9160


namespace product_of_integers_l91_9149

theorem product_of_integers (a b : ℤ) (h1 : Int.gcd a b = 12) (h2 : Int.lcm a b = 60) : a * b = 720 :=
sorry

end product_of_integers_l91_9149


namespace sum_of_common_ratios_l91_9108

theorem sum_of_common_ratios (k p r : ℝ) (h : k ≠ 0) (h1 : k * p ≠ k * r)
  (h2 : k * p ^ 2 - k * r ^ 2 = 3 * (k * p - k * r)) : p + r = 3 :=
by
  sorry

end sum_of_common_ratios_l91_9108


namespace angles_symmetric_about_y_axis_l91_9196

theorem angles_symmetric_about_y_axis (α β : ℝ) (k : ℤ) (h : β = (2 * ↑k + 1) * Real.pi - α) : 
  α + β = (2 * ↑k + 1) * Real.pi :=
sorry

end angles_symmetric_about_y_axis_l91_9196


namespace like_terms_exponents_l91_9154

theorem like_terms_exponents (m n : ℕ) (h₁ : m + 3 = 5) (h₂ : 6 = 2 * n) : m^n = 8 :=
by
  sorry

end like_terms_exponents_l91_9154


namespace joan_travel_time_correct_l91_9156

noncomputable def joan_travel_time (distance rate : ℕ) (lunch_break bathroom_breaks : ℕ) : ℕ := 
  let driving_time := distance / rate
  let break_time := lunch_break + 2 * bathroom_breaks
  driving_time + break_time / 60

theorem joan_travel_time_correct : joan_travel_time 480 60 30 15 = 9 := by
  sorry

end joan_travel_time_correct_l91_9156


namespace more_non_representable_ten_digit_numbers_l91_9184

-- Define the range of ten-digit numbers
def total_ten_digit_numbers : ℕ := 9 * 10^9

-- Define the range of five-digit numbers
def total_five_digit_numbers : ℕ := 90000

-- Calculate the number of pairs of five-digit numbers
def number_of_pairs_five_digit_numbers : ℕ :=
  total_five_digit_numbers * (total_five_digit_numbers + 1)

-- Problem statement
theorem more_non_representable_ten_digit_numbers:
  number_of_pairs_five_digit_numbers < total_ten_digit_numbers :=
by
  -- Proof is non-computable and should be added here
  sorry

end more_non_representable_ten_digit_numbers_l91_9184


namespace percent_paddyfield_warblers_l91_9191

variable (B : ℝ) -- The total number of birds.
variable (N_h : ℝ := 0.30 * B) -- Number of hawks.
variable (N_non_hawks : ℝ := 0.70 * B) -- Number of non-hawks.
variable (N_not_hpwk : ℝ := 0.35 * B) -- 35% are not hawks, paddyfield-warblers, or kingfishers.
variable (N_hpwk : ℝ := 0.65 * B) -- 65% are hawks, paddyfield-warblers, or kingfishers.
variable (P : ℝ) -- Percentage of non-hawks that are paddyfield-warblers, to be found.
variable (N_pw : ℝ := P * 0.70 * B) -- Number of paddyfield-warblers.
variable (N_k : ℝ := 0.25 * N_pw) -- Number of kingfishers.

theorem percent_paddyfield_warblers (h_eq : N_h + N_pw + N_k = 0.65 * B) : P = 0.5714 := by
  sorry

end percent_paddyfield_warblers_l91_9191


namespace population_net_change_l91_9126

theorem population_net_change
  (initial_population : ℝ)
  (year1_increase : initial_population * (6/5) = year1_population)
  (year2_increase : year1_population * (6/5) = year2_population)
  (year3_decrease : year2_population * (4/5) = year3_population)
  (year4_decrease : year3_population * (4/5) = final_population) :
  ((final_population - initial_population) / initial_population) * 100 = -8 :=
  sorry

end population_net_change_l91_9126


namespace find_a_l91_9105

theorem find_a (a : ℤ) : 
  (∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^3)) ↔ (a = 3^9) :=
by
  sorry

end find_a_l91_9105


namespace ammonia_moles_l91_9183

-- Definitions corresponding to the given conditions
def moles_KOH : ℚ := 3
def moles_NH4I : ℚ := 3

def balanced_equation (n_KOH n_NH4I : ℚ) : ℚ :=
  if n_KOH = n_NH4I then n_KOH else 0

-- Proof problem: Prove that the reaction produces 3 moles of NH3
theorem ammonia_moles (n_KOH n_NH4I : ℚ) (h1 : n_KOH = moles_KOH) (h2 : n_NH4I = moles_NH4I) :
  balanced_equation n_KOH n_NH4I = 3 :=
by 
  -- proof here 
  sorry

end ammonia_moles_l91_9183


namespace reciprocal_of_repeating_decimal_l91_9174

theorem reciprocal_of_repeating_decimal : 
  (1 : ℚ) / (34 / 99 : ℚ) = 99 / 34 :=
by sorry

end reciprocal_of_repeating_decimal_l91_9174


namespace non_science_majors_percentage_l91_9130

-- Definitions of conditions
def women_percentage (class_size : ℝ) : ℝ := 0.6 * class_size
def men_percentage (class_size : ℝ) : ℝ := 0.4 * class_size

def women_science_majors (class_size : ℝ) : ℝ := 0.2 * women_percentage class_size
def men_science_majors (class_size : ℝ) : ℝ := 0.7 * men_percentage class_size

def total_science_majors (class_size : ℝ) : ℝ := women_science_majors class_size + men_science_majors class_size

-- Theorem to prove the percentage of the class that are non-science majors is 60%
theorem non_science_majors_percentage (class_size : ℝ) : total_science_majors class_size / class_size = 0.4 → (class_size - total_science_majors class_size) / class_size = 0.6 := 
by
  sorry

end non_science_majors_percentage_l91_9130


namespace sum_S11_l91_9166

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {a1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom condition : a 3 + 4 = a 2 + a 7

theorem sum_S11 : S 11 = 44 := by
  sorry

end sum_S11_l91_9166


namespace passes_to_left_l91_9199

theorem passes_to_left
  (total_passes passes_left passes_right passes_center : ℕ)
  (h1 : total_passes = 50)
  (h2 : passes_right = 2 * passes_left)
  (h3 : passes_center = passes_left + 2)
  (h4 : total_passes = passes_left + passes_right + passes_center) :
  passes_left = 12 :=
by
  sorry

end passes_to_left_l91_9199


namespace prism_volume_l91_9140

noncomputable def volume_prism (x y z : ℝ) : ℝ := x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 12) (h2 : y * z = 8) (h3 : z * x = 6) :
  volume_prism x y z = 24 :=
by
  sorry

end prism_volume_l91_9140


namespace johns_donation_l91_9138

theorem johns_donation
    (A T D : ℝ)
    (n : ℕ)
    (hA1 : A * 1.75 = 100)
    (hA2 : A = 100 / 1.75)
    (hT : T = 10 * A)
    (hD : D = 11 * 100 - T)
    (hn : n = 10) :
    D = 3700 / 7 := 
sorry

end johns_donation_l91_9138


namespace largest_divisor_of_exp_and_linear_combination_l91_9187

theorem largest_divisor_of_exp_and_linear_combination :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ x = 18 :=
by
  sorry

end largest_divisor_of_exp_and_linear_combination_l91_9187


namespace train_length_l91_9122

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (length_m : ℚ) : 
  speed_kmh = 120 → 
  time_s = 25 → 
  length_m = 833.25 → 
  (speed_kmh * 1000 / 3600) * time_s = length_m :=
by
  intros
  sorry

end train_length_l91_9122


namespace no_five_distinct_natural_numbers_feasible_l91_9103

theorem no_five_distinct_natural_numbers_feasible :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
  sorry

end no_five_distinct_natural_numbers_feasible_l91_9103


namespace moles_HCl_combination_l91_9198

-- Define the conditions:
def moles_HCl (C5H12O: ℕ) (H2O: ℕ) : ℕ :=
  if H2O = 18 then 18 else 0

-- The main statement to prove:
theorem moles_HCl_combination :
  moles_HCl 1 18 = 18 :=
sorry

end moles_HCl_combination_l91_9198


namespace solutions_of_quadratic_eq_l91_9113

theorem solutions_of_quadratic_eq (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 :=
by {
  sorry
}

end solutions_of_quadratic_eq_l91_9113


namespace apples_total_l91_9148

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end apples_total_l91_9148


namespace solve_the_problem_l91_9111

noncomputable def solve_problem : Prop :=
  ∀ (θ t α : ℝ),
    (∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = 4 * Real.sin θ) → 
    (∃ x y : ℝ, x = 1 + t * Real.cos α ∧ y = 2 + t * Real.sin α) →
    (∃ m n : ℝ, m = 1 ∧ n = 2) →
    (-2 = Real.tan α)

theorem solve_the_problem : solve_problem := by
  sorry

end solve_the_problem_l91_9111


namespace rational_neither_positive_nor_fraction_l91_9158

def is_rational (q : ℚ) : Prop :=
  q.floor = q

def is_integer (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

def is_fraction (q : ℚ) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ q = p / q

def is_positive (q : ℚ) : Prop :=
  q > 0

theorem rational_neither_positive_nor_fraction (q : ℚ) :
  (is_rational q) ∧ ¬(is_positive q) ∧ ¬(is_fraction q) ↔
  (is_integer q ∧ q ≤ 0) :=
sorry

end rational_neither_positive_nor_fraction_l91_9158


namespace time_for_each_student_l91_9159

-- Define the conditions as variables
variables (num_students : ℕ) (period_length : ℕ) (num_periods : ℕ)
-- Assume the conditions from the problem
def conditions := num_students = 32 ∧ period_length = 40 ∧ num_periods = 4

-- Define the total time available
def total_time (num_periods period_length : ℕ) := num_periods * period_length

-- Define the time per student
def time_per_student (total_time num_students : ℕ) := total_time / num_students

-- State the theorem to be proven
theorem time_for_each_student : 
  conditions num_students period_length num_periods →
  time_per_student (total_time num_periods period_length) num_students = 5 := sorry

end time_for_each_student_l91_9159


namespace a_2_geometric_sequence_l91_9185

theorem a_2_geometric_sequence (a : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, S n = a * 3^n - 2) : S 2 = 12 :=
by 
  sorry

end a_2_geometric_sequence_l91_9185


namespace solve_for_phi_l91_9139

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * φ)

theorem solve_for_phi (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π / 2)
    (h_min_diff : |x1 - x2| = π / 6)
    (h_condition : |f x1 - g x2 φ| = 4) :
    φ = π / 3 := 
    sorry

end solve_for_phi_l91_9139


namespace complex_power_difference_l91_9102

theorem complex_power_difference (i : ℂ) (hi : i^2 = -1) : (1 + 2 * i)^8 - (1 - 2 * i)^8 = 672 * i := 
by
  sorry

end complex_power_difference_l91_9102


namespace find_speed_l91_9124

variable (d : ℝ) (t : ℝ)
variable (h1 : d = 50 * (t + 1/12))
variable (h2 : d = 70 * (t - 1/12))

theorem find_speed (d t : ℝ)
  (h1 : d = 50 * (t + 1/12))
  (h2 : d = 70 * (t - 1/12)) :
  58 = d / t := by
  sorry

end find_speed_l91_9124


namespace Aiden_sleep_fraction_l91_9169

theorem Aiden_sleep_fraction (minutes_slept : ℕ) (hour_minutes : ℕ) (h : minutes_slept = 15) (k : hour_minutes = 60) :
  (minutes_slept : ℚ) / hour_minutes = 1/4 :=
by
  sorry

end Aiden_sleep_fraction_l91_9169


namespace base_conversion_sum_l91_9180

noncomputable def A : ℕ := 10

noncomputable def base11_to_nat (x y z : ℕ) : ℕ :=
  x * 11^2 + y * 11^1 + z * 11^0

noncomputable def base12_to_nat (x y z : ℕ) : ℕ :=
  x * 12^2 + y * 12^1 + z * 12^0

theorem base_conversion_sum :
  base11_to_nat 3 7 9 + base12_to_nat 3 9 A = 999 :=
by
  sorry

end base_conversion_sum_l91_9180


namespace tree_height_at_2_years_l91_9106

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l91_9106


namespace find_loss_percentage_l91_9170

theorem find_loss_percentage (CP SP_new : ℝ) (h1 : CP = 875) (h2 : SP_new = CP * 1.04) (h3 : SP_new = SP + 140) : 
  ∃ L : ℝ, SP = CP - (L / 100 * CP) → L = 12 := 
by 
  sorry

end find_loss_percentage_l91_9170


namespace eq_a2b2_of_given_condition_l91_9120

theorem eq_a2b2_of_given_condition (a b : ℝ) (h : a^4 + b^4 = a^2 - 2 * a^2 * b^2 + b^2 + 6) : a^2 + b^2 = 3 :=
sorry

end eq_a2b2_of_given_condition_l91_9120


namespace largest_result_is_0_point_1_l91_9197

theorem largest_result_is_0_point_1 : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max (|1 - 1|) 0.1)) = 0.1 := 
by
  -- We will prove this by comparing each value to 0.1
  sorry

end largest_result_is_0_point_1_l91_9197


namespace compare_expressions_l91_9144

theorem compare_expressions (n : ℕ) (hn : 0 < n):
  (n ≤ 48 ∧ 99^n + 100^n > 101^n) ∨ (n > 48 ∧ 99^n + 100^n < 101^n) :=
sorry  -- Proof is omitted.

end compare_expressions_l91_9144


namespace median_of_data_set_l91_9195

def data_set := [2, 3, 3, 4, 6, 6, 8, 8]

def calculate_50th_percentile (l : List ℕ) : ℕ :=
  if H : l.length % 2 = 0 then
    (l.get ⟨l.length / 2 - 1, sorry⟩ + l.get ⟨l.length / 2, sorry⟩) / 2
  else
    l.get ⟨l.length / 2, sorry⟩

theorem median_of_data_set : calculate_50th_percentile data_set = 5 :=
by
  -- Insert the proof here
  sorry

end median_of_data_set_l91_9195


namespace wang_trip_duration_xiao_travel_times_l91_9161

variables (start_fee : ℝ) (time_fee_per_min : ℝ) (mileage_fee_per_km : ℝ) (long_distance_fee_per_km : ℝ)

-- Conditions
def billing_rules := 
  start_fee = 12 ∧ 
  time_fee_per_min = 0.5 ∧ 
  mileage_fee_per_km = 2.0 ∧ 
  long_distance_fee_per_km = 1.0

-- Proof for Mr. Wang's trip duration
theorem wang_trip_duration
  (x : ℝ) 
  (total_fare : ℝ)
  (distance : ℝ) 
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km) : 
  total_fare = 69.5 ∧ distance = 20 → 0.5 * x = 12.5 :=
by 
  sorry

-- Proof for Xiao Hong's and Xiao Lan's travel times
theorem xiao_travel_times 
  (x : ℝ) 
  (travel_time_multiplier : ℝ)
  (distance_hong : ℝ)
  (distance_lan : ℝ)
  (equal_fares : Prop)
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km)
  (p1 : distance_hong = 14 ∧ distance_lan = 16 ∧ travel_time_multiplier = 1.5) :
  equal_fares → 0.25 * x = 5 :=
by 
  sorry

end wang_trip_duration_xiao_travel_times_l91_9161


namespace comparison_of_y1_and_y2_l91_9125

variable {k y1 y2 : ℝ}

theorem comparison_of_y1_and_y2 (hk : 0 < k)
    (hy1 : y1 = k)
    (hy2 : y2 = k / 4) :
    y1 > y2 := by
  sorry

end comparison_of_y1_and_y2_l91_9125


namespace simplify_expression_l91_9188

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
    (Real.sqrt (4 + ( (x^3 - 2) / (3 * x) ) ^ 2)) = 
    (Real.sqrt (x^6 - 4 * x^3 + 36 * x^2 + 4) / (3 * x)) :=
by sorry

end simplify_expression_l91_9188


namespace age_of_B_l91_9107

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l91_9107


namespace tan_sin_cos_ratio_l91_9189

open Real

variable {α β : ℝ}

theorem tan_sin_cos_ratio (h1 : tan (α + β) = 2) (h2 : tan (α - β) = 3) :
  sin (2 * α) / cos (2 * β) = 5 / 7 := sorry

end tan_sin_cos_ratio_l91_9189


namespace game_ends_after_63_rounds_l91_9152

-- Define tokens for players A, B, C, and D at the start
def initial_tokens_A := 20
def initial_tokens_B := 18
def initial_tokens_C := 16
def initial_tokens_D := 14

-- Define the rules of the game
def game_rounds_to_end (A B C D : ℕ) : ℕ :=
  -- This function calculates the number of rounds after which any player runs out of tokens
  if (A, B, C, D) = (20, 18, 16, 14) then 63 else 0

-- Statement to prove
theorem game_ends_after_63_rounds :
  game_rounds_to_end initial_tokens_A initial_tokens_B initial_tokens_C initial_tokens_D = 63 :=
by sorry

end game_ends_after_63_rounds_l91_9152


namespace kyle_caught_14_fish_l91_9117

theorem kyle_caught_14_fish (K T C : ℕ) (h1 : K = T) (h2 : C = 8) (h3 : C + K + T = 36) : K = 14 :=
by
  -- Proof goes here
  sorry

end kyle_caught_14_fish_l91_9117


namespace find_f_sqrt2_l91_9129

theorem find_f_sqrt2 (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x * y) = f x + f y)
  (hf8 : f 8 = 3) :
  f (Real.sqrt 2) = 1 / 2 := by
  sorry

end find_f_sqrt2_l91_9129


namespace least_possible_value_of_s_l91_9133

theorem least_possible_value_of_s (a b : ℤ) 
(h : a^3 + b^3 - 60 * a * b * (a + b) ≥ 2012) : 
∃ a b, a^3 + b^3 - 60 * a * b * (a + b) = 2015 :=
by sorry

end least_possible_value_of_s_l91_9133


namespace jasmine_added_is_8_l91_9127

noncomputable def jasmine_problem (J : ℝ) : Prop :=
  let initial_volume := 80
  let initial_jasmine_concentration := 0.10
  let initial_jasmine_amount := initial_volume * initial_jasmine_concentration

  let added_water := 12
  let final_volume := initial_volume + J + added_water
  let final_jasmine_concentration := 0.16
  let final_jasmine_amount := final_volume * final_jasmine_concentration

  initial_jasmine_amount + J = final_jasmine_amount 

theorem jasmine_added_is_8 : jasmine_problem 8 :=
by
  sorry

end jasmine_added_is_8_l91_9127


namespace interest_rate_correct_l91_9165

namespace InterestProblem

variable (P : ℤ) (SI : ℤ) (T : ℤ)

def rate_of_interest (P : ℤ) (SI : ℤ) (T : ℤ) : ℚ :=
  (SI * 100) / (P * T)

theorem interest_rate_correct :
  rate_of_interest 400 140 2 = 17.5 := by
  sorry

end InterestProblem

end interest_rate_correct_l91_9165


namespace find_integer_x_l91_9109

theorem find_integer_x (x : ℕ) (pos_x : 0 < x) (ineq : x + 1000 > 1000 * x) : x = 1 :=
sorry

end find_integer_x_l91_9109


namespace solve_quadratic_l91_9164

theorem solve_quadratic : ∀ (x : ℝ), x * (x + 1) = 2014 * 2015 ↔ (x = 2014 ∨ x = -2015) := by
  sorry

end solve_quadratic_l91_9164


namespace inequality_holds_for_all_x_in_interval_l91_9163

theorem inequality_holds_for_all_x_in_interval (a b : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^2 + a * x + b| ≤ 1 / 8) ↔ (a = -1 ∧ b = 1 / 8) :=
sorry

end inequality_holds_for_all_x_in_interval_l91_9163


namespace forester_trees_planted_l91_9116

theorem forester_trees_planted (initial_trees : ℕ) (tripled_trees : ℕ) (trees_planted_monday : ℕ) (trees_planted_tuesday : ℕ) :
  initial_trees = 30 ∧ tripled_trees = 3 * initial_trees ∧ trees_planted_monday = tripled_trees - initial_trees ∧ trees_planted_tuesday = trees_planted_monday / 3 →
  trees_planted_monday + trees_planted_tuesday = 80 :=
by
  sorry

end forester_trees_planted_l91_9116


namespace train_crosses_post_in_approximately_18_seconds_l91_9186

noncomputable def train_length : ℕ := 300
noncomputable def platform_length : ℕ := 350
noncomputable def crossing_time_platform : ℕ := 39

noncomputable def combined_length : ℕ := train_length + platform_length
noncomputable def speed_train : ℝ := combined_length / crossing_time_platform

noncomputable def crossing_time_post : ℝ := train_length / speed_train

theorem train_crosses_post_in_approximately_18_seconds :
  abs (crossing_time_post - 18) < 1 :=
by
  admit

end train_crosses_post_in_approximately_18_seconds_l91_9186


namespace range_of_a_l91_9123

theorem range_of_a (a : ℝ) :
  (¬ ∃ x0 : ℝ, ∀ x : ℝ, x + a * x0 + 1 < 0) → (a ≥ -2 ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l91_9123


namespace range_of_k_l91_9182

theorem range_of_k (k : ℝ) : (∃ x : ℝ, 2 * x - 5 * k = x + 4 ∧ x > 0) → k > -4 / 5 :=
by
  sorry

end range_of_k_l91_9182


namespace max_fruits_is_15_l91_9194

def maxFruits (a m p : ℕ) : Prop :=
  3 * a + 4 * m + 5 * p = 50 ∧ a ≥ 1 ∧ m ≥ 1 ∧ p ≥ 1

theorem max_fruits_is_15 : ∃ a m p : ℕ, maxFruits a m p ∧ a + m + p = 15 := 
  sorry

end max_fruits_is_15_l91_9194


namespace heights_equal_l91_9114

-- Define base areas and volumes
variables {V : ℝ} {S : ℝ}

-- Assume equal volumes and base areas for the prism and cylinder
variables (h_prism h_cylinder : ℝ) (volume_eq : V = S * h_prism) (base_area_eq : S = S)

-- Define a proof goal
theorem heights_equal 
  (equal_volumes : V = S * h_prism) 
  (equal_base_areas : S = S) : 
  h_prism = h_cylinder :=
sorry

end heights_equal_l91_9114


namespace ratio_of_volumes_l91_9173

theorem ratio_of_volumes (A B : ℝ) (h : (3 / 4) * A = (2 / 3) * B) : A / B = 8 / 9 :=
by
  sorry

end ratio_of_volumes_l91_9173


namespace tangent_y_intercept_l91_9141

theorem tangent_y_intercept :
  let C1 := (2, 4)
  let r1 := 5
  let C2 := (14, 9)
  let r2 := 10
  let m := 120 / 119
  m > 0 → ∃ b, b = 912 / 119 := by
  sorry

end tangent_y_intercept_l91_9141


namespace amount_of_medication_B_l91_9118

def medicationAmounts (x y : ℝ) : Prop :=
  (x + y = 750) ∧ (0.40 * x + 0.20 * y = 215)

theorem amount_of_medication_B (x y : ℝ) (h : medicationAmounts x y) : y = 425 :=
  sorry

end amount_of_medication_B_l91_9118


namespace correct_survey_option_l91_9128

-- Definitions for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Predicate that checks if an option is suitable for a comprehensive survey method
def suitable_for_comprehensive_survey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => false
  | SurveyOption.B => false
  | SurveyOption.C => false
  | SurveyOption.D => true

-- Theorem statement
theorem correct_survey_option : suitable_for_comprehensive_survey SurveyOption.D := 
  by sorry

end correct_survey_option_l91_9128


namespace smallest_a_for_f_iter_3_l91_9172

def f (x : Int) : Int :=
  if x % 4 = 0 ∧ x % 9 = 0 then x / 36
  else if x % 9 = 0 then 4 * x
  else if x % 4 = 0 then 9 * x
  else x + 4

def f_iter (f : Int → Int) (a : Nat) (x : Int) : Int :=
  if a = 0 then x else f_iter f (a - 1) (f x)

theorem smallest_a_for_f_iter_3 (a : Nat) (h : a > 1) : 
  (∀b, b > 1 → b < a → f_iter f b 3 ≠ f 3) ∧ f_iter f a 3 = f 3 ↔ a = 9 := 
  by
  sorry

end smallest_a_for_f_iter_3_l91_9172


namespace coffee_students_l91_9151

variable (S : ℝ) -- Total number of students
variable (T : ℝ) -- Number of students who chose tea
variable (C : ℝ) -- Number of students who chose coffee

-- Given conditions
axiom h1 : 0.4 * S = 80   -- 40% of the students chose tea
axiom h2 : T = 80         -- Number of students who chose tea is 80
axiom h3 : 0.3 * S = C    -- 30% of the students chose coffee

-- Prove that the number of students who chose coffee is 60
theorem coffee_students : C = 60 := by
  sorry

end coffee_students_l91_9151


namespace range_of_m_l91_9112

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * x > m
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * m * x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ m ∈ Set.Ioo (-2:ℝ) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_m_l91_9112


namespace constant_in_price_equation_l91_9119

theorem constant_in_price_equation (x y: ℕ) (h: y = 70 * x) : ∃ c, ∀ (x: ℕ), y = c * x ∧ c = 70 :=
  sorry

end constant_in_price_equation_l91_9119


namespace square_area_with_circles_l91_9176

theorem square_area_with_circles 
  (r : ℝ)
  (nrows : ℕ)
  (ncols : ℕ)
  (circle_radius : r = 3)
  (rows : nrows = 2)
  (columns : ncols = 3)
  (num_circles : nrows * ncols = 6)
  : ∃ (side_length area : ℝ), side_length = ncols * 2 * r ∧ area = side_length ^ 2 ∧ area = 324 := 
by sorry

end square_area_with_circles_l91_9176


namespace minimum_value_fraction_l91_9179

-- Define the conditions in Lean
theorem minimum_value_fraction
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (line_through_center : ∀ x y, x = 1 ∧ y = -2 → a * x - b * y - 1 = 0) :
  (2 / a + 1 / b) = 8 := 
sorry

end minimum_value_fraction_l91_9179


namespace evaluate_expression_l91_9153

theorem evaluate_expression : 
  ∃ q : ℤ, ∀ (a : ℤ), a = 2022 → (2023 : ℚ) / 2022 - (2022 : ℚ) / 2023 = 4045 / q :=
by
  sorry

end evaluate_expression_l91_9153


namespace lena_muffins_l91_9135

theorem lena_muffins (x y z : Real) 
  (h1 : x + 2 * y + 3 * z = 3 * x + z)
  (h2 : 3 * x + z = 6 * y)
  (h3 : x + 2 * y + 3 * z = 6 * y)
  (lenas_spending : 2 * x + 2 * z = 6 * y) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end lena_muffins_l91_9135


namespace total_mass_grain_l91_9190

-- Given: the mass of the grain is 0.5 tons, and this constitutes 0.2 of the total mass
theorem total_mass_grain (m : ℝ) (h : 0.2 * m = 0.5) : m = 2.5 :=
by {
    -- Proof steps would go here
    sorry
}

end total_mass_grain_l91_9190


namespace remaining_two_by_two_square_exists_l91_9142

theorem remaining_two_by_two_square_exists (grid_size : ℕ) (cut_squares : ℕ) : grid_size = 29 → cut_squares = 99 → 
  ∃ remaining_square : ℕ, remaining_square = 1 :=
by
  intros
  sorry

end remaining_two_by_two_square_exists_l91_9142


namespace correct_operation_l91_9155

-- Defining the options as hypotheses
variable {a b : ℕ}

theorem correct_operation (hA : 4*a + 3*b ≠ 7*a*b)
    (hB : a^4 * a^3 = a^7)
    (hC : (3*a)^3 ≠ 9*a^3)
    (hD : a^6 / a^2 ≠ a^3) :
    a^4 * a^3 = a^7 := by
  sorry

end correct_operation_l91_9155


namespace find_angle_beta_l91_9110

theorem find_angle_beta (α β : ℝ)
  (h1 : (π / 2) < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19)
  (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := 
sorry

end find_angle_beta_l91_9110


namespace f_1986_l91_9101

noncomputable def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 3 * f (a * b)
axiom f_1 : f 1 = 2

theorem f_1986 : f 1986 = 2 :=
by
  sorry

end f_1986_l91_9101


namespace smallest_value_of_expression_l91_9100

theorem smallest_value_of_expression (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 - b^2 = 16) : 
  (∃ k : ℚ, k = (a + b) / (a - b) + (a - b) / (a + b) ∧ (∀ x : ℚ, x = (a + b) / (a - b) + (a - b) / (a + b) → x ≥ 9/4)) :=
sorry

end smallest_value_of_expression_l91_9100


namespace gas_pressure_inversely_proportional_l91_9145

theorem gas_pressure_inversely_proportional :
  ∀ (p v : ℝ), (p * v = 27.2) → (8 * 3.4 = 27.2) → (v = 6.8) → p = 4 :=
by
  intros p v h1 h2 h3
  have h4 : 27.2 = 8 * 3.4 := by sorry
  have h5 : p * 6.8 = 27.2 := by sorry
  exact sorry

end gas_pressure_inversely_proportional_l91_9145


namespace cruise_liner_travelers_l91_9192

theorem cruise_liner_travelers 
  (a : ℤ) 
  (h1 : 250 ≤ a) 
  (h2 : a ≤ 400) 
  (h3 : a % 15 = 7) 
  (h4 : a % 25 = -8) : 
  a = 292 ∨ a = 367 := sorry

end cruise_liner_travelers_l91_9192


namespace ratio_almonds_to_walnuts_l91_9137

theorem ratio_almonds_to_walnuts (almonds walnuts mixture : ℝ) 
  (h1 : almonds = 116.67)
  (h2 : mixture = 140)
  (h3 : walnuts = mixture - almonds) : 
  (almonds / walnuts) = 5 :=
by
  sorry

end ratio_almonds_to_walnuts_l91_9137


namespace price_after_9_years_l91_9193

-- Assume the initial conditions
def initial_price : ℝ := 640
def decrease_factor : ℝ := 0.75
def years : ℕ := 9
def period : ℕ := 3

-- Define the function to calculate the price after a certain number of years, given the period and decrease factor
def price_after_years (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_price * (decrease_factor ^ (years / period))

-- State the theorem that we intend to prove
theorem price_after_9_years : price_after_years initial_price decrease_factor 9 period = 270 := by
  sorry

end price_after_9_years_l91_9193


namespace tetrahedron_sum_of_faces_l91_9181

theorem tetrahedron_sum_of_faces (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum_vertices : b * c * d + a * c * d + a * b * d + a * b * c = 770) :
  a + b + c + d = 57 :=
sorry

end tetrahedron_sum_of_faces_l91_9181


namespace sphere_radius_l91_9121

theorem sphere_radius (tree_height sphere_shadow tree_shadow : ℝ) 
  (h_tree_shadow_pos : tree_shadow > 0) 
  (h_sphere_shadow_pos : sphere_shadow > 0) 
  (h_tree_height_pos : tree_height > 0)
  (h_tangent : (tree_height / tree_shadow) = (sphere_shadow / 15)) : 
  sphere_shadow = 11.25 :=
by
  sorry

end sphere_radius_l91_9121


namespace z_share_profit_correct_l91_9132

-- Define the investments as constants
def x_investment : ℕ := 20000
def y_investment : ℕ := 25000
def z_investment : ℕ := 30000

-- Define the number of months for each investment
def x_months : ℕ := 12
def y_months : ℕ := 12
def z_months : ℕ := 7

-- Define the annual profit
def annual_profit : ℕ := 50000

-- Calculate the active investment
def x_share : ℕ := x_investment * x_months
def y_share : ℕ := y_investment * y_months
def z_share : ℕ := z_investment * z_months

-- Calculate the total investment
def total_investment : ℕ := x_share + y_share + z_share

-- Define Z's ratio in terms of the total investment
def z_ratio : ℚ := z_share / total_investment

-- Calculate Z's share of the annual profit
def z_profit_share : ℚ := z_ratio * annual_profit

-- Theorem to prove Z's share in the annual profit
theorem z_share_profit_correct : z_profit_share = 14000 := by
  sorry

end z_share_profit_correct_l91_9132


namespace find_B_find_sin_A_find_sin_2A_minus_B_l91_9104

open Real

noncomputable def triangle_conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a * cos C + c * cos A = 2 * b * cos B) ∧ (7 * a = 5 * b)

theorem find_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  B = π / 3 :=
sorry

theorem find_sin_A (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) :
  sin A = 3 * sqrt 3 / 14 :=
sorry

theorem find_sin_2A_minus_B (a b c A B C : ℝ) (h : triangle_conditions a b c A B C)
  (hB : B = π / 3) (hA : sin A = 3 * sqrt 3 / 14) :
  sin (2 * A - B) = 8 * sqrt 3 / 49 :=
sorry

end find_B_find_sin_A_find_sin_2A_minus_B_l91_9104
