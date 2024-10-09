import Mathlib

namespace runners_meet_fractions_l1640_164039

theorem runners_meet_fractions (l V₁ V₂ : ℝ)
  (h1 : l / V₂ - l / V₁ = 10)
  (h2 : 720 * V₁ - 720 * V₂ = l) :
  (1 / V₁ = 1 / 80 ∧ 1 / V₂ = 1 / 90) ∨ (1 / V₁ = 1 / 90 ∧ 1 / V₂ = 1 / 80) :=
sorry

end runners_meet_fractions_l1640_164039


namespace percentage_of_men_is_55_l1640_164030

-- Define the percentage of men among all employees
def percent_of_men (M : ℝ) := M

-- Define the percentage of women among all employees
def percent_of_women (M : ℝ) := 1 - M

-- Define the contribution to picnic attendance by men
def attendance_by_men (M : ℝ) := 0.20 * M

-- Define the contribution to picnic attendance by women
def attendance_by_women (M : ℝ) := 0.40 * (percent_of_women M)

-- Define the total attendance
def total_attendance (M : ℝ) := attendance_by_men M + attendance_by_women M

theorem percentage_of_men_is_55 : ∀ M : ℝ, total_attendance M = 0.29 → M = 0.55 :=
by
  intro M
  intro h
  sorry

end percentage_of_men_is_55_l1640_164030


namespace root_in_interval_l1640_164014

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 / x

variable (h_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
variable (h_f_half : f (1 / 2) < 0)
variable (h_f_one : f 1 < 0)
variable (h_f_three_half : f (3 / 2) < 0)
variable (h_f_two : f 2 > 0)

theorem root_in_interval : ∃ c : ℝ, c ∈ Set.Ioo (3 / 2) 2 ∧ f c = 0 :=
sorry

end root_in_interval_l1640_164014


namespace Jaylen_total_vegetables_l1640_164055

def Jaylen_vegetables (J_bell_peppers J_green_beans J_carrots J_cucumbers : Nat) : Nat :=
  J_bell_peppers + J_green_beans + J_carrots + J_cucumbers

theorem Jaylen_total_vegetables :
  let Kristin_bell_peppers := 2
  let Kristin_green_beans := 20
  let Jaylen_bell_peppers := 2 * Kristin_bell_peppers
  let Jaylen_green_beans := (Kristin_green_beans / 2) - 3
  let Jaylen_carrots := 5
  let Jaylen_cucumbers := 2
  Jaylen_vegetables Jaylen_bell_peppers Jaylen_green_beans Jaylen_carrots Jaylen_cucumbers = 18 := 
by
  sorry

end Jaylen_total_vegetables_l1640_164055


namespace flat_odot_length_correct_l1640_164074

noncomputable def sides : ℤ × ℤ × ℤ := (4, 5, 6)

noncomputable def semiperimeter (a b c : ℤ) : ℚ :=
  (a + b + c) / 2

noncomputable def length_flat_odot (a b c : ℤ) : ℚ :=
  (semiperimeter a b c) - b

theorem flat_odot_length_correct : length_flat_odot 4 5 6 = 2.5 := by
  sorry

end flat_odot_length_correct_l1640_164074


namespace find_maximum_marks_l1640_164002

variable (percent_marks : ℝ := 0.92)
variable (obtained_marks : ℝ := 368)
variable (max_marks : ℝ := obtained_marks / percent_marks)

theorem find_maximum_marks : max_marks = 400 := by
  sorry

end find_maximum_marks_l1640_164002


namespace remainder_is_three_l1640_164046

def P (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem remainder_is_three : P 1 = 3 :=
by
  -- Proof goes here.
  sorry

end remainder_is_three_l1640_164046


namespace books_not_read_l1640_164078

theorem books_not_read (total_books read_books : ℕ) (h1 : total_books = 20) (h2 : read_books = 15) : total_books - read_books = 5 := by
  sorry

end books_not_read_l1640_164078


namespace max_z_val_l1640_164048

theorem max_z_val (x y : ℝ) (h1 : x + y ≤ 4) (h2 : y - 2 * x + 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ x y, z = x + 2 * y ∧ z = 6 :=
by
  sorry

end max_z_val_l1640_164048


namespace sum_of_cube_faces_l1640_164094

theorem sum_of_cube_faces (a b c d e f : ℕ) (h1 : a % 2 = 0) (h2 : b = a + 2) (h3 : c = b + 2) (h4 : d = c + 2) (h5 : e = d + 2) (h6 : f = e + 2)
(h_pairs : (a + f + 2) = (b + e + 2) ∧ (b + e + 2) = (c + d + 2)) :
  a + b + c + d + e + f = 90 :=
  sorry

end sum_of_cube_faces_l1640_164094


namespace exp_base_lt_imp_cube_l1640_164006

theorem exp_base_lt_imp_cube (a x y : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_exp : a^x > a^y) : x^3 < y^3 :=
by
  sorry

end exp_base_lt_imp_cube_l1640_164006


namespace domain_of_f_2x_plus_1_l1640_164073

theorem domain_of_f_2x_plus_1 {f : ℝ → ℝ} :
  (∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 3 → (-3 : ℝ) ≤ x - 1 ∧ x - 1 ≤ 2) →
  (∀ x, (-3 : ℝ) ≤ x ∧ x ≤ 2 → (-2 : ℝ) ≤ (x : ℝ) ∧ x ≤ 1/2) →
  ∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 1 / 2 → ∀ y, y = 2 * x + 1 → (-3 : ℝ) ≤ y ∧ y ≤ 2 :=
by
  sorry

end domain_of_f_2x_plus_1_l1640_164073


namespace production_today_l1640_164059

theorem production_today (n x: ℕ) (avg_past: ℕ) 
  (h1: avg_past = 50) 
  (h2: n = 1) 
  (h3: (avg_past * n + x) / (n + 1) = 55): 
  x = 60 := 
by 
  sorry

end production_today_l1640_164059


namespace production_days_l1640_164082

-- Definitions of the conditions
variables (n : ℕ) (P : ℕ)
variable (H1 : P = n * 50)
variable (H2 : (P + 60) / (n + 1) = 55)

-- Theorem to prove that n = 1 given the conditions
theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 60) / (n + 1) = 55) : n = 1 :=
by
  sorry

end production_days_l1640_164082


namespace john_candies_on_fourth_day_l1640_164029

theorem john_candies_on_fourth_day (c : ℕ) (h1 : 5 * c + 80 = 150) : c + 24 = 38 :=
by 
  -- Placeholder for proof
  sorry

end john_candies_on_fourth_day_l1640_164029


namespace min_students_l1640_164062

noncomputable def smallest_possible_number_of_students (b g : ℕ) : ℕ :=
if 3 * (3 * b) = 5 * (4 * g) then b + g else 0

theorem min_students (b g : ℕ) (h1 : 0 < b) (h2 : 0 < g) (h3 : 3 * (3 * b) = 5 * (4 * g)) :
  smallest_possible_number_of_students b g = 29 := sorry

end min_students_l1640_164062


namespace correct_omega_l1640_164076

theorem correct_omega (Ω : ℕ) (h : Ω * Ω = 2 * 2 * 2 * 2 * 3 * 3) : Ω = 2 * 2 * 3 :=
by
  sorry

end correct_omega_l1640_164076


namespace decreased_value_l1640_164081

noncomputable def original_expression (x y: ℝ) : ℝ :=
  x * y^2

noncomputable def decreased_expression (x y: ℝ) : ℝ :=
  (1 / 2) * x * (1 / 2 * y) ^ 2

theorem decreased_value (x y: ℝ) :
  decreased_expression x y = (1 / 8) * original_expression x y :=
by
  sorry

end decreased_value_l1640_164081


namespace positive_integer_solution_inequality_l1640_164077

theorem positive_integer_solution_inequality (x : ℕ) (h : 2 * (x + 1) ≥ 5 * x - 3) : x = 1 :=
by {
  sorry
}

end positive_integer_solution_inequality_l1640_164077


namespace simple_random_sampling_methods_proof_l1640_164095

-- Definitions based on conditions
def equal_probability (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
∀ s1 s2 : samples, p s1 = p s2

-- Define that Lottery Drawing Method and Random Number Table Method are part of simple random sampling
def is_lottery_drawing_method (samples : Type) : Prop := sorry
def is_random_number_table_method (samples : Type) : Prop := sorry

def simple_random_sampling_methods (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
  equal_probability samples p ∧ is_lottery_drawing_method samples ∧ is_random_number_table_method samples

-- Statement to be proven
theorem simple_random_sampling_methods_proof (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) :
  (∀ s1 s2 : samples, p s1 = p s2) → simple_random_sampling_methods samples p :=
by
  intro h
  unfold simple_random_sampling_methods
  constructor
  exact h
  constructor
  sorry -- Proof for is_lottery_drawing_method
  sorry -- Proof for is_random_number_table_method

end simple_random_sampling_methods_proof_l1640_164095


namespace min_value_point_on_line_l1640_164005

theorem min_value_point_on_line (m n : ℝ) (h : m + 2 * n = 1) : 
  2^m + 4^n ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_point_on_line_l1640_164005


namespace number_is_46000050_l1640_164068

-- Define the corresponding place values for the given digit placements
def ten_million (n : ℕ) : ℕ := n * 10000000
def hundred_thousand (n : ℕ) : ℕ := n * 100000
def hundred (n : ℕ) : ℕ := n * 100

-- Define the specific numbers given in the conditions.
def digit_4 : ℕ := ten_million 4
def digit_60 : ℕ := hundred_thousand 6
def digit_500 : ℕ := hundred 5

-- Combine these values to form the number
def combined_number : ℕ := digit_4 + digit_60 + digit_500

-- The theorem, stating the number equals 46000050
theorem number_is_46000050 : combined_number = 46000050 := by
  sorry

end number_is_46000050_l1640_164068


namespace convert_angle_degrees_to_radians_l1640_164007

theorem convert_angle_degrees_to_radians :
  ∃ (k : ℤ) (α : ℝ), -1125 * (Real.pi / 180) = 2 * k * Real.pi + α ∧ 0 ≤ α ∧ α < 2 * Real.pi ∧ (-8 * Real.pi + 7 * Real.pi / 4) = 2 * k * Real.pi + α :=
by {
  sorry
}

end convert_angle_degrees_to_radians_l1640_164007


namespace rice_and_flour_bags_l1640_164099

theorem rice_and_flour_bags (x : ℕ) (y : ℕ) 
  (h1 : x + y = 351)
  (h2 : x + 20 = 3 * (y - 50) + 1) : 
  x = 221 ∧ y = 130 :=
by
  sorry

end rice_and_flour_bags_l1640_164099


namespace no_hexagonal_pyramid_with_equal_edges_l1640_164087

theorem no_hexagonal_pyramid_with_equal_edges (edges : ℕ → ℝ)
  (regular_polygon : ℕ → ℝ → Prop)
  (equal_length_edges : ∀ (n : ℕ), regular_polygon n (edges n) → ∀ i j, edges i = edges j)
  (apex_above_centroid : ∀ (n : ℕ) (h : regular_polygon n (edges n)), True) :
  ¬ regular_polygon 6 (edges 6) :=
by
  sorry

end no_hexagonal_pyramid_with_equal_edges_l1640_164087


namespace set_membership_proof_l1640_164086

variable (A : Set ℕ) (B : Set (Set ℕ))

theorem set_membership_proof :
  A = {0, 1} → B = {x | x ⊆ A} → A ∈ B :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end set_membership_proof_l1640_164086


namespace grace_charges_for_pulling_weeds_l1640_164072

theorem grace_charges_for_pulling_weeds :
  (∃ (W : ℕ ), 63 * 6 + 9 * W + 10 * 9 = 567 → W = 11) :=
by
  use 11
  intro h
  sorry

end grace_charges_for_pulling_weeds_l1640_164072


namespace bella_age_is_five_l1640_164079

-- Definitions from the problem:
def is_age_relation (bella_age brother_age : ℕ) : Prop :=
  brother_age = bella_age + 9 ∧ bella_age + brother_age = 19

-- The main proof statement:
theorem bella_age_is_five (bella_age brother_age : ℕ) (h : is_age_relation bella_age brother_age) :
  bella_age = 5 :=
by {
  -- Placeholder for proof steps
  sorry
}

end bella_age_is_five_l1640_164079


namespace converse_proposition_l1640_164000

theorem converse_proposition (x : ℝ) (h : x = 1 → x^2 = 1) : x^2 = 1 → x = 1 :=
by
  sorry

end converse_proposition_l1640_164000


namespace repeating_decimal_to_fraction_l1640_164053

theorem repeating_decimal_to_fraction :
  7.4646464646 = (739 / 99) :=
  sorry

end repeating_decimal_to_fraction_l1640_164053


namespace packs_of_snacks_l1640_164058

theorem packs_of_snacks (kyle_bike_hours : ℝ) (pack_cost : ℝ) (ryan_budget : ℝ) :
  kyle_bike_hours = 2 →
  10 * (2 * kyle_bike_hours) = pack_cost →
  ryan_budget = 2000 →
  ryan_budget / pack_cost = 50 :=
by 
  sorry

end packs_of_snacks_l1640_164058


namespace triangle_cut_20_sided_polygon_l1640_164051

-- Definitions based on the conditions
def is_triangle (T : Type) : Prop := ∃ (a b c : ℝ), a + b + c = 180 

def can_form_20_sided_polygon (pieces : List (ℝ × ℝ)) : Prop := pieces.length = 20

-- Theorem statement
theorem triangle_cut_20_sided_polygon (T : Type) (P1 P2 : (ℝ × ℝ)) :
  is_triangle T → 
  (P1 ≠ P2) → 
  can_form_20_sided_polygon [P1, P2] :=
sorry

end triangle_cut_20_sided_polygon_l1640_164051


namespace kitty_cleaning_time_l1640_164042

def weekly_cleaning_time (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust

def total_cleaning_time (weeks: ℕ) (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  weeks * weekly_cleaning_time pick_up vacuum clean_windows dust

theorem kitty_cleaning_time :
  total_cleaning_time 4 5 20 15 10 = 200 := by
  sorry

end kitty_cleaning_time_l1640_164042


namespace sushi_eating_orders_l1640_164001

/-- Define a 2 x 3 grid with sushi pieces being distinguishable -/
inductive SushiPiece : Type
| A | B | C | D | E | F

open SushiPiece

/-- A function that counts the valid orders to eat sushi pieces satisfying the given conditions -/
noncomputable def countValidOrders : Nat :=
  sorry -- This is where the proof would go, stating the number of valid orders

theorem sushi_eating_orders :
  countValidOrders = 360 :=
sorry -- Skipping proof details

end sushi_eating_orders_l1640_164001


namespace christian_age_in_eight_years_l1640_164011

theorem christian_age_in_eight_years (b c : ℕ)
  (h1 : c = 2 * b)
  (h2 : b + 8 = 40) :
  c + 8 = 72 :=
sorry

end christian_age_in_eight_years_l1640_164011


namespace suitable_for_sampling_l1640_164084

-- Definitions based on conditions
def optionA_requires_comprehensive : Prop := true
def optionB_requires_comprehensive : Prop := true
def optionC_requires_comprehensive : Prop := true
def optionD_allows_sampling : Prop := true

-- Problem in Lean: Prove that option D is suitable for a sampling survey
theorem suitable_for_sampling : optionD_allows_sampling := by
  sorry

end suitable_for_sampling_l1640_164084


namespace find_cos_A_l1640_164009

variable {A : Real}

theorem find_cos_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.tan A = 2 / 3) : Real.cos A = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end find_cos_A_l1640_164009


namespace base_conversion_sum_l1640_164004

def A := 10
def B := 11

def convert_base11_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 11^2
  let d1 := (n % 11^2) / 11
  let d0 := n % 11
  d2 * 11^2 + d1 * 11 + d0

def convert_base12_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 12^2
  let d1 := (n % 12^2) / 12
  let d0 := n % 12
  d2 * 12^2 + d1 * 12 + d0

def n1 := 2 * 11^2 + 4 * 11 + 9    -- = 249_11 in base 10
def n2 := 3 * 12^2 + A * 12 + B   -- = 3AB_12 in base 10

theorem base_conversion_sum :
  (convert_base11_to_base10 294 + convert_base12_to_base10 563 = 858) := by
  sorry

end base_conversion_sum_l1640_164004


namespace sarith_laps_l1640_164022

theorem sarith_laps 
  (k_speed : ℝ) (s_speed : ℝ) (k_laps : ℝ) (s_laps : ℝ) (distance_ratio : ℝ) :
  k_speed = 3 * s_speed →
  distance_ratio = 1 / 2 →
  k_laps = 12 →
  s_laps = (k_laps * 2 / 3) →
  s_laps = 8 :=
by
  intros
  sorry

end sarith_laps_l1640_164022


namespace evaluate_fraction_l1640_164008

theorem evaluate_fraction : (3 : ℚ) / (2 - (3 / 4)) = (12 / 5) := 
by
  sorry

end evaluate_fraction_l1640_164008


namespace number_of_solutions_eq_six_l1640_164093

/-- 
The number of ordered pairs (m, n) of positive integers satisfying the equation
6/m + 3/n = 1 is 6.
-/
theorem number_of_solutions_eq_six : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, (1 < p.1 ∧ 1 < p.2) ∧ 6 / p.1 + 3 / p.2 = 1) ∧ s.card = 6 :=
sorry

end number_of_solutions_eq_six_l1640_164093


namespace election_votes_l1640_164063

noncomputable def third_candidate_votes (total_votes first_candidate_votes second_candidate_votes : ℕ) (winning_fraction : ℚ) : ℕ :=
  total_votes - (first_candidate_votes + second_candidate_votes)

theorem election_votes :
  ∃ total_votes : ℕ, 
  ∃ first_candidate_votes : ℕ,
  ∃ second_candidate_votes : ℕ,
  ∃ winning_fraction : ℚ,
  first_candidate_votes = 5000 ∧ 
  second_candidate_votes = 15000 ∧ 
  winning_fraction = 2/3 ∧ 
  total_votes = 60000 ∧ 
  third_candidate_votes total_votes first_candidate_votes second_candidate_votes winning_fraction = 40000 :=
    sorry

end election_votes_l1640_164063


namespace arithmetic_progression_condition_l1640_164067

theorem arithmetic_progression_condition
  (a b c : ℝ) (a1 d : ℝ) (p n k : ℕ) :
  a = a1 + (p - 1) * d →
  b = a1 + (n - 1) * d →
  c = a1 + (k - 1) * d →
  a * (n - k) + b * (k - p) + c * (p - n) = 0 :=
by
  intros h1 h2 h3
  sorry


end arithmetic_progression_condition_l1640_164067


namespace ball_hits_ground_time_l1640_164034

theorem ball_hits_ground_time (t : ℚ) :
  (-4.9 * (t : ℝ)^2 + 5 * (t : ℝ) + 10 = 0) → t = 10 / 7 :=
sorry

end ball_hits_ground_time_l1640_164034


namespace maximal_product_at_12_l1640_164024

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
a₁ * q^(n - 1)

noncomputable def product_first_n_terms (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
(a₁ ^ n) * (q ^ ((n - 1) * n / 2))

theorem maximal_product_at_12 :
  ∀ (a₁ : ℕ) (q : ℚ), 
  a₁ = 1536 → 
  q = -1/2 → 
  ∀ (n : ℕ), n ≠ 12 → 
  (product_first_n_terms a₁ q 12) > (product_first_n_terms a₁ q n) :=
by
  sorry

end maximal_product_at_12_l1640_164024


namespace problem_solution_l1640_164090

variable (a : ℝ)

theorem problem_solution (h : a ≠ 0) : a^2 + 1 > 1 :=
sorry

end problem_solution_l1640_164090


namespace find_a_of_complex_eq_l1640_164043

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end find_a_of_complex_eq_l1640_164043


namespace geom_seq_b_value_l1640_164045

variable (r : ℝ) (b : ℝ)

-- b is the second term of the geometric sequence with first term 180 and third term 36/25
-- condition 1
def geom_sequence_cond1 := 180 * r = b
-- condition 2
def geom_sequence_cond2 := b * r = 36 / 25

-- Prove b = 16.1 given the conditions
theorem geom_seq_b_value (hb_pos : b > 0) (h1 : geom_sequence_cond1 r b) (h2 : geom_sequence_cond2 r b) : b = 16.1 :=
by sorry

end geom_seq_b_value_l1640_164045


namespace graphs_intersect_once_l1640_164057

variable {a b c d : ℝ}

theorem graphs_intersect_once 
(h1: ∃ x, (2 * a + 1 / (x - b)) = (2 * c + 1 / (x - d)) ∧ 
∃ y₁ y₂: ℝ, ∀ x, (2 * a + 1 / (x - b)) ≠ 2 * c + 1 / (x - d)) : 
∃ x, ((2 * b + 1 / (x - a)) = (2 * d + 1 / (x - c))) ∧ 
∃ y₁ y₂: ℝ, ∀ x, 2 * b + 1 / (x - a) ≠ 2 * d + 1 / (x - c) := 
sorry

end graphs_intersect_once_l1640_164057


namespace inequality_solution_l1640_164085

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end inequality_solution_l1640_164085


namespace percentage_increase_is_50_l1640_164044

def initial : ℝ := 110
def final : ℝ := 165

theorem percentage_increase_is_50 :
  ((final - initial) / initial) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l1640_164044


namespace number_of_members_l1640_164025

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l1640_164025


namespace find_a_if_f_is_even_l1640_164083

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

theorem find_a_if_f_is_even
  (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 :=
sorry

end find_a_if_f_is_even_l1640_164083


namespace washer_total_cost_l1640_164056

variable (C : ℝ)
variable (h : 0.25 * C = 200)

theorem washer_total_cost : C = 800 :=
by
  sorry

end washer_total_cost_l1640_164056


namespace exists_a_lt_0_l1640_164037

noncomputable def f : ℝ → ℝ :=
sorry

theorem exists_a_lt_0 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (Real.sqrt (x * y)) = (f x + f y) / 2)
  (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :
  ∃ a : ℝ, 0 < a ∧ f a < 0 :=
sorry

end exists_a_lt_0_l1640_164037


namespace quadratic_discriminant_single_solution_l1640_164060

theorem quadratic_discriminant_single_solution :
  ∃ (n : ℝ), (∀ x : ℝ, 9 * x^2 + n * x + 36 = 0 → x = (-n) / (2 * 9)) → n = 36 :=
by
  sorry

end quadratic_discriminant_single_solution_l1640_164060


namespace remainder_of_m_l1640_164049

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l1640_164049


namespace total_amount_from_grandparents_l1640_164013

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l1640_164013


namespace blue_red_area_ratio_l1640_164017

theorem blue_red_area_ratio (d_small d_large : ℕ) (h1 : d_small = 2) (h2 : d_large = 6) :
    let r_small := d_small / 2
    let r_large := d_large / 2
    let A_red := Real.pi * (r_small : ℝ) ^ 2
    let A_large := Real.pi * (r_large : ℝ) ^ 2
    let A_blue := A_large - A_red
    A_blue / A_red = 8 :=
by
  sorry

end blue_red_area_ratio_l1640_164017


namespace smallest_number_of_students_l1640_164064

theorem smallest_number_of_students (n : ℕ) : 
  (6 * n + 2 > 40) → (∃ n, 4 * n + 2 * (n + 1) = 44) :=
 by
  intro h
  exact sorry

end smallest_number_of_students_l1640_164064


namespace largest_power_of_2_that_divides_n_l1640_164092

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_2_that_divides_n :
  ∃ k : ℕ, 2^k ∣ n ∧ ¬ (2^(k+1) ∣ n) ∧ k = 5 := sorry

end largest_power_of_2_that_divides_n_l1640_164092


namespace curve_is_parabola_l1640_164032

theorem curve_is_parabola (r θ : ℝ) : (r = 1 / (1 - Real.cos θ)) ↔ ∃ x y : ℝ, y^2 = 2 * x + 1 :=
by 
  sorry

end curve_is_parabola_l1640_164032


namespace find_e_l1640_164047

def P (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

-- Conditions
variables (d e f : ℝ)
-- Mean of zeros, twice product of zeros, and sum of coefficients are equal
variables (mean_of_zeros equals twice_product_of_zeros equals sum_of_coefficients equals: ℝ)
-- y-intercept is 9
axiom intercept_eq_nine : f = 9

-- Vieta's formulas for cubic polynomial
axiom product_of_zeros : twice_product_of_zeros = 2 * (- (f / 3))
axiom mean_of_zeros_sum : mean_of_zeros = -18/3  -- 3 times the mean of the zeros
axiom sum_of_coef : 3 + d + e + f = sum_of_coefficients

-- All these quantities are equal to the same value
axiom triple_equality : mean_of_zeros = twice_product_of_zeros
axiom triple_equality_coefs : mean_of_zeros = sum_of_coefficients

-- Lean statement we need to prove
theorem find_e : e = -72 :=
by
  sorry

end find_e_l1640_164047


namespace winning_post_distance_l1640_164097

theorem winning_post_distance (v_A v_B D : ℝ) (hvA : v_A = (5 / 3) * v_B) (head_start : 80 ≤ D) :
  (D / v_A = (D - 80) / v_B) → D = 200 :=
by
  sorry

end winning_post_distance_l1640_164097


namespace point_A_coordinates_l1640_164019

-- Given conditions
def point_A (a : ℝ) : ℝ × ℝ := (a + 1, a^2 - 4)
def negative_half_x_axis (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 = 0

-- Theorem statement
theorem point_A_coordinates (a : ℝ) (h : negative_half_x_axis (point_A a)) :
  point_A a = (-1, 0) :=
sorry

end point_A_coordinates_l1640_164019


namespace range_of_a_l1640_164071

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then (a - 5) * x + 8 else 2 * a / x

theorem range_of_a (a : ℝ) : 
  (∀ x y, x < y → f a x ≥ f a y) → (2 ≤ a ∧ a < 5) :=
sorry

end range_of_a_l1640_164071


namespace flagstaff_height_l1640_164015

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l1640_164015


namespace range_of_a_l1640_164040

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 8 ∧ a ≠ 4) ↔
  (a > 1 ∧ a < 8) ∧ (a > -4 ∧ a ≠ 4) :=
by sorry

end range_of_a_l1640_164040


namespace marcus_percentage_of_team_points_l1640_164065

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l1640_164065


namespace min_speed_x_l1640_164023

theorem min_speed_x (V_X : ℝ) : 
  let relative_speed_xy := V_X + 40;
  let relative_speed_xz := V_X - 30;
  (500 / relative_speed_xy) > (300 / relative_speed_xz) → 
  V_X ≥ 136 :=
by
  intros;
  sorry

end min_speed_x_l1640_164023


namespace pasha_mistake_l1640_164027

theorem pasha_mistake :
  ¬ (∃ (K R O S C T P : ℕ), K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ P < 10 ∧
    K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ P ∧
    R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ P ∧
    O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ P ∧
    S ≠ C ∧ S ≠ T ∧ S ≠ P ∧
    C ≠ T ∧ C ≠ P ∧ T ≠ P ∧
    10000 * K + 1000 * R + 100 * O + 10 * S + S + 2011 = 10000 * C + 1000 * T + 100 * A + 10 * P + T) :=
sorry

end pasha_mistake_l1640_164027


namespace num_pos_int_values_l1640_164010

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values_l1640_164010


namespace sum_of_reciprocals_six_l1640_164016

theorem sum_of_reciprocals_six {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x) + (1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_six_l1640_164016


namespace specific_five_card_order_probability_l1640_164020

open Classical

noncomputable def prob_five_cards_specified_order : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49) * (9 / 48)

theorem specific_five_card_order_probability :
  prob_five_cards_specified_order = 2304 / 31187500 :=
by
  sorry

end specific_five_card_order_probability_l1640_164020


namespace max_value_Tn_l1640_164096

noncomputable def geom_seq (a : ℕ → ℝ) : Prop := 
∀ n : ℕ, a (n+1) = 2 * a n

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (2 : ℝ)^n) / (1 - (2 : ℝ))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(9 * sum_first_n_terms a n - sum_first_n_terms a (2 * n)) / (a n * (2 : ℝ)^n)

theorem max_value_Tn (a : ℕ → ℝ) (h : geom_seq a) : 
  ∃ n, T_n a n ≤ 3 :=
sorry

end max_value_Tn_l1640_164096


namespace Rachel_picked_apples_l1640_164088

theorem Rachel_picked_apples :
  let apples_from_first_tree := 8
  let apples_from_second_tree := 10
  let apples_from_third_tree := 12
  let apples_from_fifth_tree := 6
  apples_from_first_tree + apples_from_second_tree + apples_from_third_tree + apples_from_fifth_tree = 36 :=
by
  sorry

end Rachel_picked_apples_l1640_164088


namespace square_field_side_length_l1640_164026

theorem square_field_side_length (t : ℕ) (v : ℕ) 
  (run_time : t = 56) 
  (run_speed : v = 9) : 
  ∃ l : ℝ, l = 35 := 
sorry

end square_field_side_length_l1640_164026


namespace large_square_area_l1640_164035

theorem large_square_area (a b c : ℕ) (h1 : 4 * a < b) (h2 : c^2 = a^2 + b^2 + 10) : c^2 = 36 :=
  sorry

end large_square_area_l1640_164035


namespace intersection_points_A_B_segment_length_MN_l1640_164061

section PolarCurves

-- Given conditions
def curve1 (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8
def curve2 (θ : ℝ) : Prop := θ = Real.pi / 6
def is_on_line (x y t : ℝ) : Prop := x = 2 + Real.sqrt 3 / 2 * t ∧ y = 1 / 2 * t

-- Polar coordinates of points A and B
theorem intersection_points_A_B :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : ℝ), curve1 ρ₁ θ₁ ∧ curve2 θ₁ ∧ curve1 ρ₂ θ₂ ∧ curve2 θ₂ ∧
    (ρ₁, θ₁) = (4, Real.pi / 6) ∧ (ρ₂, θ₂) = (4, -Real.pi / 6) :=
sorry

-- Length of the segment MN
theorem segment_length_MN :
  ∀ t : ℝ, curve1 (2 + Real.sqrt 3 / 2 * t) (1 / 2 * t) →
    ∃ t₁ t₂ : ℝ, (is_on_line (2 + Real.sqrt 3 / 2 * t₁) (1 / 2 * t₁) t₁) ∧
                (is_on_line (2 + Real.sqrt 3 / 2 * t₂) (1 / 2 * t₂) t₂) ∧
                Real.sqrt ((2 * -Real.sqrt 3 * 4)^2 - 4 * (-8)) = 4 * Real.sqrt 5 :=
sorry

end PolarCurves

end intersection_points_A_B_segment_length_MN_l1640_164061


namespace geometric_sequence_sum_correct_l1640_164066

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 2 then 2^(n + 1) - 2
else 64 * (1 - (1 / 2)^n)

theorem geometric_sequence_sum_correct (a1 q : ℝ) (n : ℕ) 
  (h1 : q > 0) 
  (h2 : a1 + a1 * q^4 = 34) 
  (h3 : a1^2 * q^4 = 64) :
  geometric_sequence_sum a1 q n = 
  if q = 2 then 2^(n + 1) - 2 else 64 * (1 - (1 / 2)^n) :=
sorry

end geometric_sequence_sum_correct_l1640_164066


namespace coeff_x4_expansion_l1640_164012

def binom_expansion (a : ℚ) : ℚ :=
  let term1 : ℚ := a * 28
  let term2 : ℚ := -56
  term1 + term2

theorem coeff_x4_expansion (a : ℚ) : (binom_expansion a = -42) → a = 1/2 := 
by 
  intro h
  -- continuation of proof will go here.
  sorry

end coeff_x4_expansion_l1640_164012


namespace konjok_gorbunok_should_act_l1640_164031

def magical_power_retention (eat : ℕ → Prop) (sleep : ℕ → Prop) (seven_days : ℕ) : Prop :=
  ∀ t : ℕ, (0 ≤ t ∧ t ≤ seven_days) → ¬(eat t ∨ sleep t)

def retains_power (need_action : Prop) : Prop :=
  need_action

theorem konjok_gorbunok_should_act
  (eat : ℕ → Prop) (sleep : ℕ → Prop)
  (seven_days : ℕ)
  (h : magical_power_retention eat sleep seven_days)
  (before_start : ℕ → Prop) :
  retains_power (before_start seven_days) :=
by
  sorry

end konjok_gorbunok_should_act_l1640_164031


namespace area_of_abs_inequality_l1640_164028

theorem area_of_abs_inequality :
  ∀ (x y : ℝ), |x + 2 * y| + |2 * x - y| ≤ 6 → 
  ∃ (area : ℝ), area = 12 := 
by
  -- This skips the proofs
  sorry

end area_of_abs_inequality_l1640_164028


namespace find_P_at_1_l1640_164069

noncomputable def P (x : ℝ) : ℝ := x ^ 2 + x + 1008

theorem find_P_at_1 :
  (∀ x : ℝ, P (P x) - (P x) ^ 2 = x ^ 2 + x + 2016) →
  P 1 = 1010 := by
  intros H
  sorry

end find_P_at_1_l1640_164069


namespace combined_difference_is_correct_l1640_164080

-- Define the number of cookies each person has
def alyssa_cookies : Nat := 129
def aiyanna_cookies : Nat := 140
def carl_cookies : Nat := 167

-- Define the differences between each pair of people's cookies
def diff_alyssa_aiyanna : Nat := aiyanna_cookies - alyssa_cookies
def diff_alyssa_carl : Nat := carl_cookies - alyssa_cookies
def diff_aiyanna_carl : Nat := carl_cookies - aiyanna_cookies

-- Define the combined difference
def combined_difference : Nat := diff_alyssa_aiyanna + diff_alyssa_carl + diff_aiyanna_carl

-- State the theorem to be proved
theorem combined_difference_is_correct : combined_difference = 76 := by
  sorry

end combined_difference_is_correct_l1640_164080


namespace moles_of_H2O_formed_l1640_164070

theorem moles_of_H2O_formed (moles_NH4NO3 moles_NaOH : ℕ) (percent_NaOH_reacts : ℝ)
  (h_decomposition : moles_NH4NO3 = 2) (h_NaOH : moles_NaOH = 2) 
  (h_percent : percent_NaOH_reacts = 0.85) : 
  (moles_NaOH * percent_NaOH_reacts = 1.7) :=
by
  sorry

end moles_of_H2O_formed_l1640_164070


namespace empty_vessel_mass_l1640_164052

theorem empty_vessel_mass
  (m1 : ℝ) (m2 : ℝ) (rho_K : ℝ) (rho_B : ℝ) (V : ℝ) (m_c : ℝ)
  (h1 : m1 = m_c + rho_K * V)
  (h2 : m2 = m_c + rho_B * V)
  (h_mass_kerosene : m1 = 31)
  (h_mass_water : m2 = 33)
  (h_rho_K : rho_K = 800)
  (h_rho_B : rho_B = 1000) :
  m_c = 23 :=
by
  -- Proof skipped
  sorry

end empty_vessel_mass_l1640_164052


namespace tangent_line_to_circle_l1640_164021

open Real

theorem tangent_line_to_circle (x y : ℝ) :
  ((x - 2) ^ 2 + (y + 1) ^ 2 = 9) ∧ ((x = -1) → (x = -1 ∧ y = 3) ∨ (y = (37 - 8*x) / 15)) :=
by {
  sorry
}

end tangent_line_to_circle_l1640_164021


namespace transform_negation_l1640_164036

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l1640_164036


namespace expression_value_l1640_164041

theorem expression_value (a b c : ℝ) (h : a + b + c = 0) : (a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b)) = 3 := 
by 
  sorry

end expression_value_l1640_164041


namespace total_surface_area_correct_l1640_164050

-- Definitions for side lengths of the cubes
def side_length_large := 5
def side_length_medium := 2
def side_length_small := 1

-- Surface area calculation for a single cube
def surface_area (side_length : ℕ) : ℕ := 6 * side_length^2

-- Surface areas for each size of the cube
def surface_area_large := surface_area side_length_large
def surface_area_medium := surface_area side_length_medium
def surface_area_small := surface_area side_length_small

-- Total surface areas for medium and small cubes
def surface_area_medium_total := 4 * surface_area_medium
def surface_area_small_total := 4 * surface_area_small

-- Total surface area of the structure
def total_surface_area := surface_area_large + surface_area_medium_total + surface_area_small_total

-- Expected result
def expected_surface_area := 270

-- Proof statement
theorem total_surface_area_correct : total_surface_area = expected_surface_area := by
  sorry

end total_surface_area_correct_l1640_164050


namespace circle_symmetric_equation_l1640_164003

noncomputable def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

noncomputable def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

noncomputable def symmetric_condition (x y : ℝ) (L : ℝ × ℝ → Prop) : Prop := 
  L (y + 1, x - 1)

theorem circle_symmetric_equation :
  ∀ (x y : ℝ),
  circle1 (y + 1) (x - 1) →
  (x-2)^2 + (y+2)^2 = 1 :=
by
  intros x y h
  sorry

end circle_symmetric_equation_l1640_164003


namespace odd_natural_of_form_l1640_164091

/-- 
  Prove that the only odd natural number n in the form (p + q) / (p - q)
  where p and q are prime numbers and p > q is 5.
-/
theorem odd_natural_of_form (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p > q) 
  (h2 : ∃ n : ℕ, n = (p + q) / (p - q) ∧ n % 2 = 1) : ∃ n : ℕ, n = 5 :=
sorry

end odd_natural_of_form_l1640_164091


namespace penny_initial_money_l1640_164075

theorem penny_initial_money
    (pairs_of_socks : ℕ)
    (cost_per_pair : ℝ)
    (number_of_pairs : ℕ)
    (cost_of_hat : ℝ)
    (money_left : ℝ)
    (initial_money : ℝ)
    (H1 : pairs_of_socks = 4)
    (H2 : cost_per_pair = 2)
    (H3 : number_of_pairs = pairs_of_socks)
    (H4 : cost_of_hat = 7)
    (H5 : money_left = 5)
    (H6 : initial_money = (number_of_pairs * cost_per_pair) + cost_of_hat + money_left) : initial_money = 20 :=
sorry

end penny_initial_money_l1640_164075


namespace find_point_A_l1640_164089

theorem find_point_A :
  (∃ A : ℤ, A + 2 = -2) ∨ (∃ A : ℤ, A - 2 = -2) → (∃ A : ℤ, A = 0 ∨ A = -4) :=
by
  sorry

end find_point_A_l1640_164089


namespace base_b_addition_correct_base_b_l1640_164054

theorem base_b_addition (b : ℕ) (hb : b > 5) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 :=
  by
    sorry

theorem correct_base_b : ∃ (b : ℕ), b > 5 ∧ 
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 ∧
  (4 + 5 = b + 1) ∧
  (2 + 1 + 1 = 4) :=
  ⟨8, 
   by decide,
   base_b_addition 8 (by decide),
   by decide,
   by decide⟩ 

end base_b_addition_correct_base_b_l1640_164054


namespace quadratic_range_l1640_164033

theorem quadratic_range (x y : ℝ) (h1 : y = -(x - 5) ^ 2 + 1) (h2 : 2 < x ∧ x < 6) :
  -8 < y ∧ y ≤ 1 := 
sorry

end quadratic_range_l1640_164033


namespace pranks_combinations_correct_l1640_164098

noncomputable def pranks_combinations : ℕ := by
  let monday_choice := 1
  let tuesday_choice := 2
  let wednesday_choice := 4
  let thursday_choice := 5
  let friday_choice := 1
  let total_combinations := monday_choice * tuesday_choice * wednesday_choice * thursday_choice * friday_choice
  exact 40

theorem pranks_combinations_correct : pranks_combinations = 40 := by
  unfold pranks_combinations
  sorry -- Proof omitted

end pranks_combinations_correct_l1640_164098


namespace polynomial_identity_l1640_164018

theorem polynomial_identity 
  (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  a^2 * ((x - b) * (x - c) / ((a - b) * (a - c))) +
  b^2 * ((x - c) * (x - a) / ((b - c) * (b - a))) +
  c^2 * ((x - a) * (x - b) / ((c - a) * (c - b))) = x^2 :=
by
  sorry

end polynomial_identity_l1640_164018


namespace eval_expression_eq_54_l1640_164038

theorem eval_expression_eq_54 : (3 * 4 * 6) * ((1/3 : ℚ) + 1/4 + 1/6) = 54 := 
by
  sorry

end eval_expression_eq_54_l1640_164038
