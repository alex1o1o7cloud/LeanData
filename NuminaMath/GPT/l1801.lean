import Mathlib

namespace NUMINAMATH_GPT_time_to_cross_man_l1801_180178

-- Definitions based on the conditions
def speed_faster_train_kmph := 72 -- km per hour
def speed_slower_train_kmph := 36 -- km per hour
def length_faster_train_m := 200 -- meters

-- Convert speeds from km/h to m/s
def speed_faster_train_mps := speed_faster_train_kmph * 1000 / 3600 -- meters per second
def speed_slower_train_mps := speed_slower_train_kmph * 1000 / 3600 -- meters per second

-- Relative speed calculation
def relative_speed_mps := speed_faster_train_mps - speed_slower_train_mps -- meters per second

-- Prove the time to cross the man in the slower train
theorem time_to_cross_man : length_faster_train_m / relative_speed_mps = 20 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_time_to_cross_man_l1801_180178


namespace NUMINAMATH_GPT_frames_sharing_point_with_line_e_l1801_180131

def frame_shares_common_point_with_line (n : ℕ) : Prop := 
  n = 0 ∨ n = 1 ∨ n = 9 ∨ n = 17 ∨ n = 25 ∨ n = 33 ∨ n = 41 ∨ n = 49 ∨
  n = 6 ∨ n = 14 ∨ n = 22 ∨ n = 30 ∨ n = 38 ∨ n = 46

theorem frames_sharing_point_with_line_e :
  ∀ (i : ℕ), i < 50 → frame_shares_common_point_with_line i = 
  (i = 0 ∨ i = 1 ∨ i = 9 ∨ i = 17 ∨ i = 25 ∨ i = 33 ∨ i = 41 ∨ i = 49 ∨
   i = 6 ∨ i = 14 ∨ i = 22 ∨ i = 30 ∨ i = 38 ∨ i = 46) := 
by 
  sorry

end NUMINAMATH_GPT_frames_sharing_point_with_line_e_l1801_180131


namespace NUMINAMATH_GPT_rowing_upstream_distance_l1801_180175

theorem rowing_upstream_distance 
  (b s t d1 d2 : ℝ)
  (h1 : s = 7)
  (h2 : d1 = 72)
  (h3 : t = 3)
  (h4 : d1 = (b + s) * t) :
  d2 = (b - s) * t → d2 = 30 :=
by 
  intros h5
  sorry

end NUMINAMATH_GPT_rowing_upstream_distance_l1801_180175


namespace NUMINAMATH_GPT_case_a_case_b_case_c_l1801_180154

-- Definitions of game manageable
inductive Player
| First
| Second

def sum_of_dimensions (m n : Nat) : Nat := m + n

def is_winning_position (m n : Nat) : Player :=
  if sum_of_dimensions m n % 2 = 1 then Player.First else Player.Second

-- Theorem statements for the given grid sizes
theorem case_a : is_winning_position 9 10 = Player.First := 
  sorry

theorem case_b : is_winning_position 10 12 = Player.Second := 
  sorry

theorem case_c : is_winning_position 9 11 = Player.Second := 
  sorry

end NUMINAMATH_GPT_case_a_case_b_case_c_l1801_180154


namespace NUMINAMATH_GPT_garbage_bill_problem_l1801_180183

theorem garbage_bill_problem
  (R : ℝ)
  (trash_bins : ℝ := 2)
  (recycling_bins : ℝ := 1)
  (weekly_trash_cost_per_bin : ℝ := 10)
  (weeks_per_month : ℝ := 4)
  (discount_rate : ℝ := 0.18)
  (fine : ℝ := 20)
  (final_bill : ℝ := 102) :
  (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  - discount_rate * (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  + fine = final_bill →
  R = 5 := 
by
  sorry

end NUMINAMATH_GPT_garbage_bill_problem_l1801_180183


namespace NUMINAMATH_GPT_grandmother_age_l1801_180103

theorem grandmother_age 
  (avg_age : ℝ)
  (age1 age2 age3 grandma_age : ℝ)
  (h_avg_age : avg_age = 20)
  (h_ages : age1 = 5)
  (h_ages2 : age2 = 10)
  (h_ages3 : age3 = 13)
  (h_eq : (age1 + age2 + age3 + grandma_age) / 4 = avg_age) : 
  grandma_age = 52 := 
by
  sorry

end NUMINAMATH_GPT_grandmother_age_l1801_180103


namespace NUMINAMATH_GPT_find_n_l1801_180106

theorem find_n (x : ℝ) (n : ℝ) (G : ℝ) (hG : G = (7*x^2 + 21*x + 5*n) / 7) :
  (∃ c d : ℝ, c^2 * x^2 + 2*c*d*x + d^2 = G) ↔ n = 63 / 20 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1801_180106


namespace NUMINAMATH_GPT_find_denominator_l1801_180114

theorem find_denominator (x : ℕ) (dec_form_of_frac_4128 : ℝ) (h1: 4128 / x = dec_form_of_frac_4128) 
    : x = 4387 :=
by
  have h: dec_form_of_frac_4128 = 0.9411764705882353 := sorry
  sorry

end NUMINAMATH_GPT_find_denominator_l1801_180114


namespace NUMINAMATH_GPT_total_distance_correct_l1801_180147

def d1 : ℕ := 350
def d2 : ℕ := 375
def d3 : ℕ := 275
def total_distance : ℕ := 1000

theorem total_distance_correct : d1 + d2 + d3 = total_distance := by
  sorry

end NUMINAMATH_GPT_total_distance_correct_l1801_180147


namespace NUMINAMATH_GPT_min_value_f_l1801_180100

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x + Real.sin (Real.pi / 2 + x)

theorem min_value_f : ∃ x : ℝ, f x = -2 := by
  sorry

end NUMINAMATH_GPT_min_value_f_l1801_180100


namespace NUMINAMATH_GPT_communication_scenarios_l1801_180169

theorem communication_scenarios
  (nA : ℕ) (nB : ℕ) (hA : nA = 10) (hB : nB = 20) : 
  (∃ scenarios : ℕ, scenarios = 2 ^ (nA * nB)) :=
by
  use 2 ^ (10 * 20)
  sorry

end NUMINAMATH_GPT_communication_scenarios_l1801_180169


namespace NUMINAMATH_GPT_radius_of_larger_circle_l1801_180163

theorem radius_of_larger_circle
  (r : ℝ) -- radius of the smaller circle
  (R : ℝ) -- radius of the larger circle
  (ratio : R = 4 * r) -- radii ratio 1:4
  (AC : ℝ) -- diameter of the larger circle
  (BC : ℝ) -- chord of the larger circle
  (AB : ℝ := 16) -- given condition AB = 16
  (diameter_AC : AC = 2 * R) -- AC is diameter of the larger circle
  (tangent : BC^2 = AB^2 + (2 * R)^2) -- Pythagorean theorem for the right triangle ABC
  :
  R = 32 := 
sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l1801_180163


namespace NUMINAMATH_GPT_cryptarithmetic_proof_l1801_180159

theorem cryptarithmetic_proof (A B C D : ℕ) 
  (h1 : A * B = 6) 
  (h2 : C = 2) 
  (h3 : A + B + D = 13) 
  (h4 : A + B + C = D) : 
  D = 6 :=
by
  sorry

end NUMINAMATH_GPT_cryptarithmetic_proof_l1801_180159


namespace NUMINAMATH_GPT_number_of_weeks_in_a_single_harvest_season_l1801_180162

-- Define constants based on conditions
def weeklyEarnings : ℕ := 1357
def totalHarvestSeasons : ℕ := 73
def totalEarnings : ℕ := 22090603

-- Prove the number of weeks in a single harvest season
theorem number_of_weeks_in_a_single_harvest_season :
  (totalEarnings / weeklyEarnings) / totalHarvestSeasons = 223 := 
  by
    sorry

end NUMINAMATH_GPT_number_of_weeks_in_a_single_harvest_season_l1801_180162


namespace NUMINAMATH_GPT_find_number_l1801_180137

theorem find_number (x : ℝ) (h : 0.60 * 50 = 0.45 * x + 16.5) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1801_180137


namespace NUMINAMATH_GPT_max_profit_achieved_when_x_is_1_l1801_180195

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2
noncomputable def fixed_costs : ℝ := 40
noncomputable def material_cost (x : ℕ) : ℝ := 5 * x
noncomputable def profit (x : ℕ) : ℝ := revenue x - (fixed_costs + material_cost x)
noncomputable def marginal_profit (x : ℕ) : ℝ := profit (x + 1) - profit x

theorem max_profit_achieved_when_x_is_1 :
  marginal_profit 1 = 24.40 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_max_profit_achieved_when_x_is_1_l1801_180195


namespace NUMINAMATH_GPT_purely_imaginary_satisfies_condition_l1801_180194

theorem purely_imaginary_satisfies_condition (m : ℝ) (h1 : m^2 + 3 * m - 4 = 0) (h2 : m + 4 ≠ 0) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_satisfies_condition_l1801_180194


namespace NUMINAMATH_GPT_intersection_A_B_complement_l1801_180139

def universal_set : Set ℝ := {x : ℝ | True}
def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def B_complement : Set ℝ := {x : ℝ | x ≤ 1}

theorem intersection_A_B_complement :
  (A ∩ B_complement) = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_complement_l1801_180139


namespace NUMINAMATH_GPT_quadratic_equation_general_form_l1801_180191

theorem quadratic_equation_general_form (x : ℝ) (h : 4 * x = x^2 - 8) : x^2 - 4 * x - 8 = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_general_form_l1801_180191


namespace NUMINAMATH_GPT_probability_P_plus_S_is_two_less_than_multiple_of_7_l1801_180107

def is_distinct (a b : ℕ) : Prop :=
  a ≠ b

def in_range (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100

def mod_condition (a b : ℕ) : Prop :=
  (a * b + a + b) % 7 = 5

noncomputable def probability_p_s (p q : ℕ) : ℚ :=
  p / q

theorem probability_P_plus_S_is_two_less_than_multiple_of_7 :
  probability_p_s (1295) (4950) = 259 / 990 := 
sorry

end NUMINAMATH_GPT_probability_P_plus_S_is_two_less_than_multiple_of_7_l1801_180107


namespace NUMINAMATH_GPT_geometric_progression_solution_l1801_180192

-- Definitions and conditions as per the problem
def geometric_progression_first_term (b q : ℝ) : Prop :=
  b * (1 + q + q^2) = 21

def geometric_progression_sum_of_squares (b q : ℝ) : Prop :=
  b^2 * (1 + q^2 + q^4) = 189

-- The main theorem to be proven
theorem geometric_progression_solution (b q : ℝ) :
  (geometric_progression_first_term b q ∧ geometric_progression_sum_of_squares b q) →
  (b = 3 ∧ q = 2) ∨ (b = 12 ∧ q = 1 / 2) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l1801_180192


namespace NUMINAMATH_GPT_ceil_of_neg_sqrt_frac_64_over_9_l1801_180117

theorem ceil_of_neg_sqrt_frac_64_over_9 :
  ⌈-Real.sqrt (64 / 9)⌉ = -2 :=
by
  sorry

end NUMINAMATH_GPT_ceil_of_neg_sqrt_frac_64_over_9_l1801_180117


namespace NUMINAMATH_GPT_polynomial_root_s_eq_pm1_l1801_180150

theorem polynomial_root_s_eq_pm1
  (b_3 b_2 b_1 : ℤ)
  (s : ℤ)
  (h1 : s^3 ∣ 50)
  (h2 : (s^4 + b_3 * s^3 + b_2 * s^2 + b_1 * s + 50) = 0) :
  s = 1 ∨ s = -1 :=
sorry

end NUMINAMATH_GPT_polynomial_root_s_eq_pm1_l1801_180150


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1801_180180

theorem value_of_x_plus_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1801_180180


namespace NUMINAMATH_GPT_palindromes_between_300_800_l1801_180172

def palindrome_count (l u : ℕ) : ℕ :=
  (u / 100 - l / 100 + 1) * 10

theorem palindromes_between_300_800 : palindrome_count 300 800 = 50 :=
by
  sorry

end NUMINAMATH_GPT_palindromes_between_300_800_l1801_180172


namespace NUMINAMATH_GPT_max_sum_of_factors_of_1764_l1801_180121

theorem max_sum_of_factors_of_1764 :
  ∃ (a b : ℕ), a * b = 1764 ∧ a + b = 884 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_factors_of_1764_l1801_180121


namespace NUMINAMATH_GPT_min_xy_positive_real_l1801_180188

theorem min_xy_positive_real (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 / (2 + x) + 3 / (2 + y) = 1) :
  ∃ m : ℝ, m = 16 ∧ ∀ xy : ℝ, (xy = x * y) → xy ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_xy_positive_real_l1801_180188


namespace NUMINAMATH_GPT_hundredth_odd_positive_integer_l1801_180167

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end NUMINAMATH_GPT_hundredth_odd_positive_integer_l1801_180167


namespace NUMINAMATH_GPT_no_prime_solution_in_2_to_7_l1801_180196

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_solution_in_2_to_7 : ∀ p : ℕ, is_prime p ∧ 2 ≤ p ∧ p ≤ 7 → (2 * p^3 - p^2 - 15 * p + 22) ≠ 0 :=
by
  intros p hp
  have h := hp.left
  sorry

end NUMINAMATH_GPT_no_prime_solution_in_2_to_7_l1801_180196


namespace NUMINAMATH_GPT_angle_alpha_not_2pi_over_9_l1801_180140

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x))

theorem angle_alpha_not_2pi_over_9 (α : ℝ) (h : f α = 1 / 8) : α ≠ 2 * π / 9 :=
sorry

end NUMINAMATH_GPT_angle_alpha_not_2pi_over_9_l1801_180140


namespace NUMINAMATH_GPT_harry_bought_l1801_180186

-- Definitions based on the conditions
def initial_bottles := 35
def jason_bought := 5
def final_bottles := 24

-- Theorem stating the number of bottles Harry bought
theorem harry_bought :
  (initial_bottles - jason_bought) - final_bottles = 6 :=
by
  sorry

end NUMINAMATH_GPT_harry_bought_l1801_180186


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1801_180185

theorem sum_of_x_and_y 
  (x y : ℤ)
  (h1 : x - y = 36) 
  (h2 : x = 28) : 
  x + y = 20 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1801_180185


namespace NUMINAMATH_GPT_intersecting_lines_l1801_180119

theorem intersecting_lines (a b c d : ℝ) (h₁ : a ≠ b) (h₂ : ∃ x y : ℝ, y = a*x + a ∧ y = b*x + b ∧ y = c*x + d) : c = d :=
sorry

end NUMINAMATH_GPT_intersecting_lines_l1801_180119


namespace NUMINAMATH_GPT_determine_identity_l1801_180165

-- Define the types for human and vampire
inductive Being
| human
| vampire

-- Define the responses for sanity questions
def claims_sanity (b : Being) : Prop :=
  match b with
  | Being.human   => true
  | Being.vampire => false

-- Proof statement: Given that a human always claims sanity and a vampire always claims insanity,
-- asking "Are you sane?" will determine their identity. 
theorem determine_identity (b : Being) (h : b = Being.human ↔ claims_sanity b = true) : 
  ((claims_sanity b = true) → b = Being.human) ∧ ((claims_sanity b = false) → b = Being.vampire) :=
sorry

end NUMINAMATH_GPT_determine_identity_l1801_180165


namespace NUMINAMATH_GPT_football_team_starting_lineup_count_l1801_180109

theorem football_team_starting_lineup_count :
  let total_members := 12
  let offensive_lineman_choices := 4
  let quarterback_choices := 2
  let remaining_after_ol := total_members - 1 -- after choosing one offensive lineman
  let remaining_after_qb := remaining_after_ol - 1 -- after choosing one quarterback
  let running_back_choices := remaining_after_ol
  let wide_receiver_choices := remaining_after_qb - 1
  let tight_end_choices := remaining_after_qb - 2
  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 5760 := 
by
  sorry

end NUMINAMATH_GPT_football_team_starting_lineup_count_l1801_180109


namespace NUMINAMATH_GPT_prove_pattern_example_l1801_180127

noncomputable def pattern_example : Prop :=
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  (123456 * 9 + 7 = 1111111)

theorem prove_pattern_example : pattern_example := by
  sorry

end NUMINAMATH_GPT_prove_pattern_example_l1801_180127


namespace NUMINAMATH_GPT_problem1_problem2_l1801_180132

-- Statement for Problem 1
theorem problem1 (x y : ℝ) : (x - y) ^ 2 + x * (x + 2 * y) = 2 * x ^ 2 + y ^ 2 :=
by sorry

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3 * x + 4) / (x - 1) + x) / ((x - 2) / (x ^ 2 - x)) = x ^ 2 - 2 * x :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1801_180132


namespace NUMINAMATH_GPT_max_tan_beta_l1801_180197

theorem max_tan_beta (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2) 
  (h : α + β ≠ π / 2) (h_sin_cos : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_max_tan_beta_l1801_180197


namespace NUMINAMATH_GPT_can_form_triangle_l1801_180158

theorem can_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  (a = 7 ∧ b = 12 ∧ c = 17) → True :=
by
  sorry

end NUMINAMATH_GPT_can_form_triangle_l1801_180158


namespace NUMINAMATH_GPT_avg_salary_difference_l1801_180168

theorem avg_salary_difference (factory_payroll : ℕ) (factory_workers : ℕ) (office_payroll : ℕ) (office_workers : ℕ)
  (h1 : factory_payroll = 30000) (h2 : factory_workers = 15)
  (h3 : office_payroll = 75000) (h4 : office_workers = 30) :
  (office_payroll / office_workers) - (factory_payroll / factory_workers) = 500 := by
  sorry

end NUMINAMATH_GPT_avg_salary_difference_l1801_180168


namespace NUMINAMATH_GPT_aspirin_mass_percentages_l1801_180141

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_aspirin : ℝ := (9 * atomic_mass_C) + (8 * atomic_mass_H) + (4 * atomic_mass_O)

theorem aspirin_mass_percentages :
  let mass_percent_H := ((8 * atomic_mass_H) / molar_mass_aspirin) * 100
  let mass_percent_C := ((9 * atomic_mass_C) / molar_mass_aspirin) * 100
  let mass_percent_O := ((4 * atomic_mass_O) / molar_mass_aspirin) * 100
  mass_percent_H = 4.48 ∧ mass_percent_C = 60.00 ∧ mass_percent_O = 35.52 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_aspirin_mass_percentages_l1801_180141


namespace NUMINAMATH_GPT_arun_gokul_age_subtract_l1801_180155

theorem arun_gokul_age_subtract:
  ∃ x : ℕ, (60 - x) / 18 = 3 → x = 6 :=
sorry

end NUMINAMATH_GPT_arun_gokul_age_subtract_l1801_180155


namespace NUMINAMATH_GPT_percentage_increase_second_movie_l1801_180102

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_second_movie_l1801_180102


namespace NUMINAMATH_GPT_initial_apples_l1801_180181

theorem initial_apples (X : ℕ) (h : X - 2 + 3 = 5) : X = 4 :=
sorry

end NUMINAMATH_GPT_initial_apples_l1801_180181


namespace NUMINAMATH_GPT_number_of_girls_in_club_l1801_180110

theorem number_of_girls_in_club (total : ℕ) (C1 : total = 36) 
    (C2 : ∀ (S : Finset ℕ), S.card = 33 → ∃ g b : ℕ, g + b = 33 ∧ g > b) 
    (C3 : ∃ (S : Finset ℕ), S.card = 31 ∧ ∃ g b : ℕ, g + b = 31 ∧ b > g) : 
    ∃ G : ℕ, G = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_club_l1801_180110


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1801_180164

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  2 * k ≠ 0 → (8 * k + 1)^2 - 64 * k^2 > 0 → k > -1 / 16 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1801_180164


namespace NUMINAMATH_GPT_minimum_sum_at_nine_l1801_180104

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem minimum_sum_at_nine {a1 d : ℤ} (h_a1_neg : a1 < 0) 
    (h_sum_equal : sum_of_arithmetic_sequence a1 d 12 = sum_of_arithmetic_sequence a1 d 6) :
  ∀ n : ℕ, (n = 9) → sum_of_arithmetic_sequence a1 d n ≤ sum_of_arithmetic_sequence a1 d m :=
sorry

end NUMINAMATH_GPT_minimum_sum_at_nine_l1801_180104


namespace NUMINAMATH_GPT_a_leq_neg4_l1801_180177

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def neg_p (a x : ℝ) : Prop := ¬(p a x)
def neg_q (x : ℝ) : Prop := ¬(q x)

theorem a_leq_neg4 (a : ℝ) (h_neg_p : ∀ x, neg_p a x → neg_q x) (h_a_neg : a < 0) :
  a ≤ -4 :=
sorry

end NUMINAMATH_GPT_a_leq_neg4_l1801_180177


namespace NUMINAMATH_GPT_find_number_l1801_180157

theorem find_number (x : ℝ) (h : 20 * (x / 5) = 40) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1801_180157


namespace NUMINAMATH_GPT_parabola_equation_l1801_180111

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end NUMINAMATH_GPT_parabola_equation_l1801_180111


namespace NUMINAMATH_GPT_intersection_of_sets_l1801_180133

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def set_B : Set ℝ := {x | (x + 1) * (x - 4) > 0}

theorem intersection_of_sets :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ {x | (x + 1) * (x - 4) > 0} = {x | -2 ≤ x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1801_180133


namespace NUMINAMATH_GPT_determinant_value_l1801_180156

-- Define the determinant calculation for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the initial conditions
variables {x : ℝ}
axiom h : x^2 - 3*x + 1 = 0

-- State the theorem to be proved
theorem determinant_value : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_determinant_value_l1801_180156


namespace NUMINAMATH_GPT_solution_statement_l1801_180189

-- Define the set of courses
inductive Course
| Physics | Chemistry | Literature | History | Philosophy | Psychology

open Course

-- Define the condition that a valid program must include Physics and at least one of Chemistry or Literature
def valid_program (program : Finset Course) : Prop :=
  Course.Physics ∈ program ∧
  (Course.Chemistry ∈ program ∨ Course.Literature ∈ program)

-- Define the problem statement
def problem_statement : Prop :=
  ∃ programs : Finset (Finset Course),
    programs.card = 9 ∧ ∀ program ∈ programs, program.card = 5 ∧ valid_program program

theorem solution_statement : problem_statement := sorry

end NUMINAMATH_GPT_solution_statement_l1801_180189


namespace NUMINAMATH_GPT_sequence_solution_l1801_180120

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → a (n + 1) = a n * ((n + 2) / n)

theorem sequence_solution (a : ℕ → ℝ) (h1 : seq a) (h2 : a 1 = 1) :
  ∀ n : ℕ, 0 < n → a n = (n * (n + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l1801_180120


namespace NUMINAMATH_GPT_range_of_2a_minus_b_l1801_180145

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2 * a^2 - a * b - b^2 - 4 = 0) :
  (2 * a - b) ∈ (Set.Ici (8 / 3)) :=
sorry

end NUMINAMATH_GPT_range_of_2a_minus_b_l1801_180145


namespace NUMINAMATH_GPT_integer_squared_equals_product_l1801_180134

theorem integer_squared_equals_product : 
  3^8 * 3^12 * 2^5 * 2^10 = 1889568^2 :=
by
  sorry

end NUMINAMATH_GPT_integer_squared_equals_product_l1801_180134


namespace NUMINAMATH_GPT_investment_in_business_l1801_180123

theorem investment_in_business (Q : ℕ) (P : ℕ) 
  (h1 : Q = 65000) 
  (h2 : 4 * Q = 5 * P) : 
  P = 52000 :=
by
  rw [h1] at h2
  linarith

end NUMINAMATH_GPT_investment_in_business_l1801_180123


namespace NUMINAMATH_GPT_maximize_q_l1801_180166

noncomputable def maximum_q (X Y Z : ℕ) : ℕ :=
X * Y * Z + X * Y + Y * Z + Z * X

theorem maximize_q : ∃ (X Y Z : ℕ), X + Y + Z = 15 ∧ (∀ (A B C : ℕ), A + B + C = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ A * B * C + A * B + B * C + C * A) ∧ maximum_q X Y Z = 200 :=
by
  sorry

end NUMINAMATH_GPT_maximize_q_l1801_180166


namespace NUMINAMATH_GPT_closest_point_on_line_y_eq_3x_plus_2_l1801_180143

theorem closest_point_on_line_y_eq_3x_plus_2 (x y : ℝ) :
  ∃ (p : ℝ × ℝ), p = (-1 / 2, 1 / 2) ∧ y = 3 * x + 2 ∧ p = (x, y) :=
by
-- We skip the proof steps and provide the statement only
sorry

end NUMINAMATH_GPT_closest_point_on_line_y_eq_3x_plus_2_l1801_180143


namespace NUMINAMATH_GPT_problem_statement_l1801_180118

def has_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def p : Prop := ∀ m : ℝ, has_solutions m

def q : Prop := ∃ x_0 : ℕ, x_0^2 - 2 * x_0 - 1 ≤ 0

theorem problem_statement : ¬ (p ∧ ¬ q) := 
sorry

end NUMINAMATH_GPT_problem_statement_l1801_180118


namespace NUMINAMATH_GPT_owen_initial_turtles_l1801_180198

variables (O J : ℕ)

-- Conditions
def johanna_turtles := J = O - 5
def owen_final_turtles := 2 * O + J / 2 = 50

-- Theorem statement
theorem owen_initial_turtles (h1 : johanna_turtles O J) (h2 : owen_final_turtles O J) : O = 21 :=
sorry

end NUMINAMATH_GPT_owen_initial_turtles_l1801_180198


namespace NUMINAMATH_GPT_min_value_1abc_l1801_180129

theorem min_value_1abc (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : c = 0) 
    (h₄ : (1000 + 100 * a + 10 * b + c) % 2 = 0) 
    (h₅ : (1000 + 100 * a + 10 * b + c) % 3 = 0) 
    (h₆ : (1000 + 100 * a + 10 * b + c) % 5 = 0)
  : 1000 + 100 * a + 10 * b + c = 1020 :=
by
  sorry

end NUMINAMATH_GPT_min_value_1abc_l1801_180129


namespace NUMINAMATH_GPT_pizza_slices_with_both_l1801_180182

theorem pizza_slices_with_both (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h_total : total_slices = 24) (h_pepperoni : pepperoni_slices = 15) (h_mushrooms : mushroom_slices = 14) :
  ∃ n, n = 5 ∧ total_slices = pepperoni_slices + mushroom_slices - n := 
by
  use 5
  sorry

end NUMINAMATH_GPT_pizza_slices_with_both_l1801_180182


namespace NUMINAMATH_GPT_inequality_solution_range_l1801_180144

theorem inequality_solution_range (x : ℝ) : (x^2 + 3*x - 10 < 0) ↔ (-5 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_range_l1801_180144


namespace NUMINAMATH_GPT_ellipse_has_correct_equation_l1801_180161

noncomputable def ellipse_Equation (a b : ℝ) (eccentricity : ℝ) (triangle_perimeter : ℝ) : Prop :=
  let c := a * eccentricity
  (a > b) ∧ (b > 0) ∧ (eccentricity = (Real.sqrt 3) / 3) ∧ (triangle_perimeter = 4 * (Real.sqrt 3)) ∧
  (a = Real.sqrt 3) ∧ (b^2 = a^2 - c^2) ∧
  (c = 1) ∧
  (b = Real.sqrt 2) ∧
  (∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ ((x^2 / 3) + (y^2 / 2) = 1))

theorem ellipse_has_correct_equation : ellipse_Equation (Real.sqrt 3) (Real.sqrt 2) ((Real.sqrt 3) / 3) (4 * (Real.sqrt 3)) := 
sorry

end NUMINAMATH_GPT_ellipse_has_correct_equation_l1801_180161


namespace NUMINAMATH_GPT_problem_statement_l1801_180153

theorem problem_statement (P : ℝ) (h : P = 1 / (Real.log 11 / Real.log 2) + 1 / (Real.log 11 / Real.log 3) + 1 / (Real.log 11 / Real.log 4) + 1 / (Real.log 11 / Real.log 5)) : 1 < P ∧ P < 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1801_180153


namespace NUMINAMATH_GPT_negation_of_proposition_l1801_180146

theorem negation_of_proposition (a : ℝ) :
  (¬ (∀ x : ℝ, (x - a) ^ 2 + 2 > 0)) ↔ (∃ x : ℝ, (x - a) ^ 2 + 2 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1801_180146


namespace NUMINAMATH_GPT_average_weight_bc_is_43_l1801_180125

variable (a b c : ℝ)

-- Definitions of the conditions
def average_weight_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_weight_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def weight_b (b : ℝ) : Prop := b = 31

-- The theorem to prove
theorem average_weight_bc_is_43 :
  ∀ (a b c : ℝ), average_weight_abc a b c → average_weight_ab a b → weight_b b → (b + c) / 2 = 43 :=
by
  intros a b c h_average_weight_abc h_average_weight_ab h_weight_b
  sorry

end NUMINAMATH_GPT_average_weight_bc_is_43_l1801_180125


namespace NUMINAMATH_GPT_pages_same_units_digit_l1801_180187

theorem pages_same_units_digit (n : ℕ) (H : n = 63) : 
  ∃ (count : ℕ), count = 13 ∧ ∀ x : ℕ, (1 ≤ x ∧ x ≤ n) → 
  (((x % 10) = ((n + 1 - x) % 10)) → (x = 2 ∨ x = 7 ∨ x = 12 ∨ x = 17 ∨ x = 22 ∨ x = 27 ∨ x = 32 ∨ x = 37 ∨ x = 42 ∨ x = 47 ∨ x = 52 ∨ x = 57 ∨ x = 62)) :=
by
  sorry

end NUMINAMATH_GPT_pages_same_units_digit_l1801_180187


namespace NUMINAMATH_GPT_quadratic_equal_roots_l1801_180113

theorem quadratic_equal_roots (k : ℝ) : (∃ r : ℝ, (r^2 - 2 * r + k = 0)) → k = 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_l1801_180113


namespace NUMINAMATH_GPT_product_of_repeating_decimals_l1801_180108

noncomputable def repeating_decimal_038 : ℚ := 38 / 999
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem product_of_repeating_decimals :
  repeating_decimal_038 * repeating_decimal_4 = 152 / 8991 :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimals_l1801_180108


namespace NUMINAMATH_GPT_y1_lt_y2_l1801_180122

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end NUMINAMATH_GPT_y1_lt_y2_l1801_180122


namespace NUMINAMATH_GPT_find_b_in_expression_l1801_180128

theorem find_b_in_expression
  (a b : ℚ)
  (h : (1 + Real.sqrt 3)^5 = a + b * Real.sqrt 3) :
  b = 44 :=
sorry

end NUMINAMATH_GPT_find_b_in_expression_l1801_180128


namespace NUMINAMATH_GPT_beth_friends_l1801_180152

theorem beth_friends (F : ℝ) (h1 : 4 / F + 6 = 6.4) : F = 10 :=
by
  sorry

end NUMINAMATH_GPT_beth_friends_l1801_180152


namespace NUMINAMATH_GPT_binomial_square_constant_l1801_180124

theorem binomial_square_constant :
  ∃ c : ℝ, (∀ x : ℝ, 9*x^2 - 21*x + c = (3*x + -3.5)^2) → c = 12.25 :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_constant_l1801_180124


namespace NUMINAMATH_GPT_problem_l1801_180101

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem problem : f (g 2) + g (f 2) = 38 / 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1801_180101


namespace NUMINAMATH_GPT_distance_MC_l1801_180112

theorem distance_MC (MA MB MC : ℝ) (hMA : MA = 2) (hMB : MB = 3) (hABC : ∀ x y z : ℝ, x + y > z ∧ y + z > x ∧ z + x > y) :
  1 ≤ MC ∧ MC ≤ 5 := 
by 
  sorry

end NUMINAMATH_GPT_distance_MC_l1801_180112


namespace NUMINAMATH_GPT_mean_age_euler_family_l1801_180115

theorem mean_age_euler_family :
  let ages := [6, 6, 9, 11, 13, 16]
  let total_children := 6
  let total_sum := 61
  (total_sum / total_children : ℝ) = (61 / 6 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_mean_age_euler_family_l1801_180115


namespace NUMINAMATH_GPT_smallest_of_three_l1801_180151

noncomputable def A : ℕ := 38 + 18
noncomputable def B : ℕ := A - 26
noncomputable def C : ℕ := B / 3

theorem smallest_of_three : C < A ∧ C < B := by
  sorry

end NUMINAMATH_GPT_smallest_of_three_l1801_180151


namespace NUMINAMATH_GPT_symmetry_condition_l1801_180176

-- Define grid and initial conditions
def grid : Type := ℕ × ℕ
def is_colored (pos : grid) : Prop := 
  pos = (1,4) ∨ pos = (2,1) ∨ pos = (4,2)

-- Conditions for symmetry: horizontal and vertical line symmetry and 180-degree rotational symmetry
def is_symmetric_line (grid_size : grid) (pos : grid) : Prop :=
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨ 
  (pos.1 > grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2) ∨
  (pos.1 > grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2)

def grid_size : grid := (4, 5)
def add_squares_needed (num : ℕ) : Prop :=
  ∀ (pos : grid), is_symmetric_line grid_size pos → is_colored pos

theorem symmetry_condition : 
  ∃ n, add_squares_needed n ∧ n = 9
  := sorry

end NUMINAMATH_GPT_symmetry_condition_l1801_180176


namespace NUMINAMATH_GPT_roots_range_of_a_l1801_180160

theorem roots_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + (a - 2)*|x - 3| + 9 - 2*a = 0) ↔ a > 0 ∨ a = -2 :=
sorry

end NUMINAMATH_GPT_roots_range_of_a_l1801_180160


namespace NUMINAMATH_GPT_constant_term_in_expansion_l1801_180193

-- Define the binomial expansion general term
def binomial_general_term (x : ℤ) (r : ℕ) : ℤ :=
  (-2)^r * 3^(5 - r) * (Nat.choose 5 r) * x^(10 - 5 * r)

-- Define the condition for the specific r that makes the exponent of x zero
def condition (r : ℕ) : Prop :=
  10 - 5 * r = 0

-- Define the constant term calculation
def const_term : ℤ :=
  4 * 27 * (Nat.choose 5 2)

-- Theorem statement
theorem constant_term_in_expansion : const_term = 1080 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_constant_term_in_expansion_l1801_180193


namespace NUMINAMATH_GPT_max_n_satisfying_inequality_l1801_180116

theorem max_n_satisfying_inequality : 
  ∃ (n : ℤ), 303 * n^3 ≤ 380000 ∧ ∀ m : ℤ, m > n → 303 * m^3 > 380000 := sorry

end NUMINAMATH_GPT_max_n_satisfying_inequality_l1801_180116


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l1801_180135

theorem parabola_hyperbola_focus {p : ℝ} :
  let focus_parabola := (p / 2, 0)
  let focus_hyperbola := (2, 0)
  focus_parabola = focus_hyperbola -> p = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l1801_180135


namespace NUMINAMATH_GPT_division_of_decimals_l1801_180199

theorem division_of_decimals : 0.25 / 0.005 = 50 := 
by
  sorry

end NUMINAMATH_GPT_division_of_decimals_l1801_180199


namespace NUMINAMATH_GPT_chocolate_flavored_cups_sold_l1801_180184

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end NUMINAMATH_GPT_chocolate_flavored_cups_sold_l1801_180184


namespace NUMINAMATH_GPT_wheels_in_garage_l1801_180126

theorem wheels_in_garage :
  let bicycles := 9
  let cars := 16
  let single_axle_trailers := 5
  let double_axle_trailers := 3
  let wheels_per_bicycle := 2
  let wheels_per_car := 4
  let wheels_per_single_axle_trailer := 2
  let wheels_per_double_axle_trailer := 4
  let total_wheels := bicycles * wheels_per_bicycle + cars * wheels_per_car + single_axle_trailers * wheels_per_single_axle_trailer + double_axle_trailers * wheels_per_double_axle_trailer
  total_wheels = 104 := by
  sorry

end NUMINAMATH_GPT_wheels_in_garage_l1801_180126


namespace NUMINAMATH_GPT_max_handshakes_25_people_l1801_180130

theorem max_handshakes_25_people : 
  (∃ n : ℕ, n = 25) → 
  (∀ p : ℕ, p ≤ 24) → 
  ∃ m : ℕ, m = 300 :=
by sorry

end NUMINAMATH_GPT_max_handshakes_25_people_l1801_180130


namespace NUMINAMATH_GPT_triangle_intersect_sum_l1801_180148

theorem triangle_intersect_sum (P Q R S T U : ℝ × ℝ) :
  P = (0, 8) →
  Q = (0, 0) →
  R = (10, 0) →
  S = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  T = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  ∃ U : ℝ × ℝ, 
    (U.1 = (P.1 + ((T.2 - P.2) / (T.1 - P.1)) * (U.1 - P.1)) ∧
     U.2 = (R.2 + ((S.2 - R.2) / (S.1 - R.1)) * (U.1 - R.1))) ∧
    (U.1 + U.2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_intersect_sum_l1801_180148


namespace NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l1801_180142

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l1801_180142


namespace NUMINAMATH_GPT_factorization_example_l1801_180179

theorem factorization_example : 
  ∀ (a : ℝ), a^2 - 6 * a + 9 = (a - 3)^2 :=
by
  intro a
  sorry

end NUMINAMATH_GPT_factorization_example_l1801_180179


namespace NUMINAMATH_GPT_cellphone_surveys_l1801_180171

theorem cellphone_surveys
  (regular_rate : ℕ)
  (total_surveys : ℕ)
  (higher_rate_multiplier : ℕ)
  (total_earnings : ℕ)
  (higher_rate_bonus : ℕ)
  (x : ℕ) :
  regular_rate = 10 → total_surveys = 100 →
  higher_rate_multiplier = 130 → total_earnings = 1180 →
  higher_rate_bonus = 3 → (10 * (100 - x) + 13 * x = 1180) →
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_cellphone_surveys_l1801_180171


namespace NUMINAMATH_GPT_min_S_n_at_24_l1801_180170

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n : ℤ) * (2 * n - 48)

theorem min_S_n_at_24 : (∀ n : ℕ, n > 0 → S_n n ≥ S_n 24) ∧ S_n 24 < S_n 25 :=
by 
  sorry

end NUMINAMATH_GPT_min_S_n_at_24_l1801_180170


namespace NUMINAMATH_GPT_cone_in_sphere_less_half_volume_l1801_180136

theorem cone_in_sphere_less_half_volume
  (R r m : ℝ)
  (h1 : m < 2 * R)
  (h2 : r <= R) :
  (1 / 3 * Real.pi * r^2 * m < 1 / 2 * 4 / 3 * Real.pi * R^3) :=
by
  sorry

end NUMINAMATH_GPT_cone_in_sphere_less_half_volume_l1801_180136


namespace NUMINAMATH_GPT_problem_statement_l1801_180149

theorem problem_statement : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1801_180149


namespace NUMINAMATH_GPT_option_C_correct_l1801_180105

theorem option_C_correct : (Real.sqrt 2) * (Real.sqrt 6) = 2 * (Real.sqrt 3) :=
by sorry

end NUMINAMATH_GPT_option_C_correct_l1801_180105


namespace NUMINAMATH_GPT_sum_q_p_evaluation_l1801_180173

def p (x : Int) : Int := x^2 - 3
def q (x : Int) : Int := x - 2

def T : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

noncomputable def f (x : Int) : Int := q (p x)

noncomputable def sum_f_T : Int := List.sum (List.map f T)

theorem sum_q_p_evaluation :
  sum_f_T = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_q_p_evaluation_l1801_180173


namespace NUMINAMATH_GPT_veranda_area_correct_l1801_180174

noncomputable def area_veranda (length_room : ℝ) (width_room : ℝ) (width_veranda : ℝ) (radius_obstacle : ℝ) : ℝ :=
  let total_length := length_room + 2 * width_veranda
  let total_width := width_room + 2 * width_veranda
  let area_total := total_length * total_width
  let area_room := length_room * width_room
  let area_circle := Real.pi * radius_obstacle^2
  area_total - area_room - area_circle

theorem veranda_area_correct :
  area_veranda 18 12 2 3 = 107.726 :=
by sorry

end NUMINAMATH_GPT_veranda_area_correct_l1801_180174


namespace NUMINAMATH_GPT_min_ab_l1801_180138

theorem min_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b - a * b + 3 = 0) : 
  9 ≤ a * b :=
sorry

end NUMINAMATH_GPT_min_ab_l1801_180138


namespace NUMINAMATH_GPT_Jessie_final_weight_l1801_180190

variable (initial_weight : ℝ) (loss_first_week : ℝ) (loss_rate_second_week : ℝ)
variable (loss_second_week : ℝ) (total_loss : ℝ) (final_weight : ℝ)

def Jessie_weight_loss_problem : Prop :=
  initial_weight = 92 ∧
  loss_first_week = 5 ∧
  loss_rate_second_week = 1.3 ∧
  loss_second_week = loss_rate_second_week * loss_first_week ∧
  total_loss = loss_first_week + loss_second_week ∧
  final_weight = initial_weight - total_loss ∧
  final_weight = 80.5

theorem Jessie_final_weight : Jessie_weight_loss_problem initial_weight loss_first_week loss_rate_second_week loss_second_week total_loss final_weight :=
by
  sorry

end NUMINAMATH_GPT_Jessie_final_weight_l1801_180190
