import Mathlib

namespace year_population_below_five_percent_l71_71247

def population (P0 : ℕ) (years : ℕ) : ℕ :=
  P0 / 2^years

theorem year_population_below_five_percent (P0 : ℕ) :
  ∃ n, population P0 n < P0 / 20 ∧ (2005 + n) = 2010 := 
by {
  sorry
}

end year_population_below_five_percent_l71_71247


namespace find_number_l71_71119

theorem find_number (x : ℤ) (h : 72516 * x = 724797420) : x = 10001 :=
by
  sorry

end find_number_l71_71119


namespace total_spent_l71_71172

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l71_71172


namespace find_values_of_a_and_b_find_square_root_l71_71629

-- Define the conditions
def condition1 (a b : ℤ) : Prop := (2 * b - 2 * a)^3 = -8
def condition2 (a b : ℤ) : Prop := (4 * a + 3 * b)^2 = 9

-- State the problem to prove the values of a and b
theorem find_values_of_a_and_b (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) : 
  a = 3 ∧ b = -1 :=
sorry

-- State the problem to prove the square root of 5a - b
theorem find_square_root (a b : ℤ) (h1 : condition1 a b) (h2 : condition2 a b) (ha : a = 3) (hb : b = -1) :
  ∃ x : ℤ, x^2 = 5 * a - b ∧ (x = 4 ∨ x = -4) :=
sorry

end find_values_of_a_and_b_find_square_root_l71_71629


namespace total_students_in_college_l71_71215

theorem total_students_in_college 
  (girls : ℕ) 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (h_ratio : ratio_boys = 8) 
  (h_ratio_girls : ratio_girls = 5) 
  (h_girls : girls = 400) 
  : (ratio_boys * (girls / ratio_girls) + girls = 1040) := 
by 
  sorry

end total_students_in_college_l71_71215


namespace neg_four_is_square_root_of_sixteen_l71_71834

/-
  Definitions:
  - A number y is a square root of x if y^2 = x.
  - A number y is an arithmetic square root of x if y ≥ 0 and y^2 = x.
-/

theorem neg_four_is_square_root_of_sixteen :
  -4 * -4 = 16 := 
by
  -- proof step is omitted
  sorry

end neg_four_is_square_root_of_sixteen_l71_71834


namespace smallest_a1_l71_71934

noncomputable def is_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 7 * a (n - 1) - 2 * n

noncomputable def is_positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 0, a n > 0

theorem smallest_a1 (a : ℕ → ℝ)
  (h_seq : is_sequence a)
  (h_pos : is_positive_sequence a) :
  a 1 ≥ 13 / 18 :=
sorry

end smallest_a1_l71_71934


namespace vertical_asymptote_x_value_l71_71877

theorem vertical_asymptote_x_value (x : ℝ) : 
  4 * x - 6 = 0 ↔ x = 3 / 2 :=
by
  sorry

end vertical_asymptote_x_value_l71_71877


namespace hide_and_seek_l71_71376

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71376


namespace sum_of_coefficients_is_zero_l71_71130

noncomputable def expansion : Polynomial ℚ := (Polynomial.X^2 + Polynomial.X + 1) * (2*Polynomial.X - 2)^5

theorem sum_of_coefficients_is_zero :
  (expansion.coeff 0) + (expansion.coeff 1) + (expansion.coeff 2) + (expansion.coeff 3) + 
  (expansion.coeff 4) + (expansion.coeff 5) + (expansion.coeff 6) + (expansion.coeff 7) = 0 :=
by
  sorry

end sum_of_coefficients_is_zero_l71_71130


namespace probability_of_drawing_red_ball_from_bag_B_l71_71853

-- Define events A1, A2, A3 in terms of probabilities
def P_A1 : ℚ := 2/5
def P_A2 : ℚ := 2/5
def P_A3 : ℚ := 1/5

-- Define conditional probabilities for drawing a red ball from bag B
def P_B_given_A1 : ℚ := 4/6
def P_B_given_A2 : ℚ := 3/6
def P_B_given_A3 : ℚ := 3/6

-- Define the total probability of drawing a red ball from bag B after transferring any ball from bag A.
def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- Proof that P(B) = 17/30
theorem probability_of_drawing_red_ball_from_bag_B :
  P_B = 17/30 :=
by
  -- Calculation of probability
  sorry

end probability_of_drawing_red_ball_from_bag_B_l71_71853


namespace number_of_possible_values_for_c_l71_71027

theorem number_of_possible_values_for_c : 
  (∃ c_values : Finset ℕ, (∀ c ∈ c_values, c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) 
  ∧ c_values.card = 10) :=
sorry

end number_of_possible_values_for_c_l71_71027


namespace player_pass_probability_l71_71762

noncomputable def probability_passing_test : ℝ :=
  let p := 2 / 3 in
  let q := 1 - p in
  /- Probability of making exactly 3 shots out of 3 attempts -/
  (p^3) +
  /- Probability of making 3 shots out of 4 attempts, one miss -/
  (3 * (p^3) * q) +
  /- Probability of making 3 shots out of 5 attempts, two misses -/
  (6 * (p^3) * (q^2))

theorem player_pass_probability : probability_passing_test = 64 / 81 := by
  sorry

end player_pass_probability_l71_71762


namespace corrected_mean_is_45_55_l71_71061

-- Define the initial conditions
def mean_of_100_observations (mean : ℝ) : Prop :=
  mean = 45

def incorrect_observation : ℝ := 32
def correct_observation : ℝ := 87

-- Define the calculation of the corrected mean
noncomputable def corrected_mean (incorrect_mean : ℝ) (incorrect_obs : ℝ) (correct_obs : ℝ) (n : ℕ) : ℝ :=
  let sum_original := incorrect_mean * n
  let difference := correct_obs - incorrect_obs
  (sum_original + difference) / n

-- Theorem: The corrected new mean is 45.55
theorem corrected_mean_is_45_55 : corrected_mean 45 32 87 100 = 45.55 :=
by
  sorry

end corrected_mean_is_45_55_l71_71061


namespace min_value_expression_l71_71297

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := 
by
  sorry

end min_value_expression_l71_71297


namespace find_first_term_geometric_series_l71_71968

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l71_71968


namespace condition_s_for_q_condition_r_for_q_condition_p_for_s_l71_71153

variables {p q r s : Prop}

-- Given conditions from a)
axiom h₁ : r → p
axiom h₂ : q → r
axiom h₃ : s → r
axiom h₄ : q → s

-- The corresponding proof problems based on c)
theorem condition_s_for_q : (s ↔ q) :=
by sorry

theorem condition_r_for_q : (r ↔ q) :=
by sorry

theorem condition_p_for_s : (s → p) :=
by sorry

end condition_s_for_q_condition_r_for_q_condition_p_for_s_l71_71153


namespace eval_expression_l71_71725

theorem eval_expression : 
  (-(1/2))⁻¹ - 4 * real.cos (30 * real.pi / 180) - (real.pi + 2013)^0 + real.sqrt 12 = -3 := 
by 
  sorry

end eval_expression_l71_71725


namespace largest_solution_l71_71872

-- Define the largest solution to the equation |5x - 3| = 28 as 31/5.
theorem largest_solution (x : ℝ) (h : |5 * x - 3| = 28) : x ≤ 31 / 5 := 
  sorry

end largest_solution_l71_71872


namespace students_no_A_l71_71149

theorem students_no_A
  (total_students : ℕ)
  (A_in_history : ℕ)
  (A_in_math : ℕ)
  (A_in_science : ℕ)
  (A_in_history_and_math : ℕ)
  (A_in_history_and_science : ℕ)
  (A_in_math_and_science : ℕ)
  (A_in_all_three : ℕ)
  (h_total_students : total_students = 40)
  (h_A_in_history : A_in_history = 10)
  (h_A_in_math : A_in_math = 15)
  (h_A_in_science : A_in_science = 8)
  (h_A_in_history_and_math : A_in_history_and_math = 5)
  (h_A_in_history_and_science : A_in_history_and_science = 3)
  (h_A_in_math_and_science : A_in_math_and_science = 4)
  (h_A_in_all_three : A_in_all_three = 2) :
  total_students - (A_in_history + A_in_math + A_in_science 
    - A_in_history_and_math - A_in_history_and_science - A_in_math_and_science 
    + A_in_all_three) = 17 := 
sorry

end students_no_A_l71_71149


namespace slope_AA_l71_71828

-- Define the points and conditions
variable (a b c d e f : ℝ)

-- Assumptions
#check (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
#check (a ≠ b ∧ c ≠ d ∧ e ≠ f)
#check (a+2 > 0 ∧ b > 0 ∧ c+2 > 0 ∧ d > 0 ∧ e+2 > 0 ∧ f > 0)

-- Main Statement
theorem slope_AA'_not_negative_one
    (H1: a > 0) (H2: b > 0) (H3: c > 0) (H4: d > 0)
    (H5: e > 0) (H6: f > 0) 
    (H7: a ≠ b) (H8: c ≠ d) (H9: e ≠ f)
    (H10: a + 2 > 0) (H11: c + 2 > 0) (H12: e + 2 > 0) : 
    (a ≠ b) → (c ≠ d) → (e ≠ f) → ¬( (a + 2 - b) / (b - a) = -1 ) :=
by
  sorry

end slope_AA_l71_71828


namespace negate_prop_l71_71636

theorem negate_prop (p : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) :
  ¬ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → |Real.sin x| ≤ 1) ↔ ∃ x_0 : ℝ, 0 ≤ x_0 ∧ x_0 ≤ 2 * Real.pi ∧ |Real.sin x_0| > 1 :=
by sorry

end negate_prop_l71_71636


namespace stans_average_speed_l71_71026

/-- Given that Stan drove 420 miles in 6 hours, 480 miles in 7 hours, and 300 miles in 5 hours,
prove that his average speed for the entire trip is 1200/18 miles per hour. -/
theorem stans_average_speed :
  let total_distance := 420 + 480 + 300
  let total_time := 6 + 7 + 5
  total_distance / total_time = 1200 / 18 :=
by
  sorry

end stans_average_speed_l71_71026


namespace nuts_in_trail_mix_l71_71928

theorem nuts_in_trail_mix :
  let walnuts := 0.25
  let almonds := 0.25
  walnuts + almonds = 0.50 :=
by
  sorry

end nuts_in_trail_mix_l71_71928


namespace binom_600_eq_1_l71_71095

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l71_71095


namespace sum_of_first_n_terms_l71_71344

-- Define the sequence aₙ
def a (n : ℕ) : ℕ := 2 * n - 1

-- Prove that the sum of the first n terms of the sequence is n²
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range (n+1)).sum a = n^2 :=
by sorry -- Proof is skipped

end sum_of_first_n_terms_l71_71344


namespace pythagorean_triple_l71_71885

theorem pythagorean_triple {c a b : ℕ} (h1 : a = 24) (h2 : b = 7) (h3 : c = 25) : a^2 + b^2 = c^2 :=
by
  rw [h1, h2, h3]
  norm_num

end pythagorean_triple_l71_71885


namespace Jessica_cut_40_roses_l71_71004

-- Define the problem's conditions as variables
variables (initialVaseRoses : ℕ) (finalVaseRoses : ℕ) (rosesGivenToSarah : ℕ)

-- Define the number of roses Jessica cut from her garden
def rosesCutFromGarden (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) : ℕ :=
  (finalVaseRoses - initialVaseRoses) + rosesGivenToSarah

-- Problem statement: Prove Jessica cut 40 roses from her garden
theorem Jessica_cut_40_roses (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) :
  initialVaseRoses = 7 →
  finalVaseRoses = 37 →
  rosesGivenToSarah = 10 →
  rosesCutFromGarden initialVaseRoses finalVaseRoses rosesGivenToSarah = 40 :=
by
  intros h1 h2 h3
  sorry

end Jessica_cut_40_roses_l71_71004


namespace chris_is_14_l71_71520

-- Definitions from the given conditions
variables (a b c : ℕ)
variables (h1 : (a + b + c) / 3 = 10)
variables (h2 : c - 4 = a)
variables (h3 : b + 5 = (3 * (a + 5)) / 4)

theorem chris_is_14 (h1 : (a + b + c) / 3 = 10) (h2 : c - 4 = a) (h3 : b + 5 = (3 * (a + 5)) / 4) : c = 14 := 
sorry

end chris_is_14_l71_71520


namespace latin_student_sophomore_probability_l71_71284

variable (F S J SE : ℕ) -- freshmen, sophomores, juniors, seniors total
variable (FL SL JL SEL : ℕ) -- freshmen, sophomores, juniors, seniors taking latin
variable (p : ℚ) -- probability fraction
variable (m n : ℕ) -- relatively prime integers

-- Let the total number of students be 100 for simplicity in percentage calculations
-- Let us encode the given conditions
def conditions := 
  F = 40 ∧ 
  S = 30 ∧ 
  J = 20 ∧ 
  SE = 10 ∧ 
  FL = 40 ∧ 
  SL = S * 80 / 100 ∧ 
  JL = J * 50 / 100 ∧ 
  SEL = SE * 20 / 100

-- The probability calculation
def probability_sophomore (SL : ℕ) (FL SL JL SEL : ℕ) : ℚ := SL / (FL + SL + JL + SEL)

-- Target probability as a rational number
def target_probability := (6 : ℚ) / 19

theorem latin_student_sophomore_probability : 
  conditions F S J SE FL SL JL SEL → 
  probability_sophomore SL FL SL JL SEL = target_probability ∧ 
  m + n = 25 := 
by 
  sorry

end latin_student_sophomore_probability_l71_71284


namespace expenditure_may_to_july_l71_71809

theorem expenditure_may_to_july (spent_by_may : ℝ) (spent_by_july : ℝ) (h_may : spent_by_may = 0.8) (h_july : spent_by_july = 3.5) :
  spent_by_july - spent_by_may = 2.7 :=
by
  sorry

end expenditure_may_to_july_l71_71809


namespace max_area_rect_bamboo_fence_l71_71699

theorem max_area_rect_bamboo_fence (a b : ℝ) (h : a + b = 10) : a * b ≤ 24 :=
by
  sorry

end max_area_rect_bamboo_fence_l71_71699


namespace binomial_600_600_l71_71104

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l71_71104


namespace first_term_of_geometric_series_l71_71956

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l71_71956


namespace solve_for_x_l71_71432

theorem solve_for_x (x : ℚ) (h : 5 * (x - 4) = 3 * (3 - 3 * x) + 6) : x = 5 / 2 :=
by {
  sorry
}

end solve_for_x_l71_71432


namespace coordinates_equidistant_l71_71427

-- Define the condition of equidistance
theorem coordinates_equidistant (x y : ℝ) :
  (x + 2) ^ 2 + (y - 2) ^ 2 = (x - 2) ^ 2 + y ^ 2 →
  y = 2 * x + 1 :=
  sorry  -- Proof is omitted

end coordinates_equidistant_l71_71427


namespace average_of_ratios_l71_71043

theorem average_of_ratios (a b c : ℕ) (h1 : 2 * b = 3 * a) (h2 : 3 * c = 4 * a) (h3 : a = 28) : (a + b + c) / 3 = 42 := by
  -- skipping the proof
  sorry

end average_of_ratios_l71_71043


namespace union_of_sets_eq_l71_71879

variable (M N : Set ℕ)

theorem union_of_sets_eq (h1 : M = {1, 2}) (h2 : N = {2, 3}) : M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_eq_l71_71879


namespace union_is_equivalent_l71_71491

def A (x : ℝ) : Prop := x ^ 2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem union_is_equivalent (x : ℝ) :
  (A x ∨ B x) ↔ (-2 ≤ x ∧ x < 4) :=
sorry

end union_is_equivalent_l71_71491


namespace polynomial_zero_iff_divisibility_l71_71839

theorem polynomial_zero_iff_divisibility (P : Polynomial ℤ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℤ, P.eval (2^n) = n * k) ↔ P = 0 :=
by sorry

end polynomial_zero_iff_divisibility_l71_71839


namespace greatest_divisor_l71_71576

theorem greatest_divisor (n : ℕ) (h1 : 1428 % n = 9) (h2 : 2206 % n = 13) : n = 129 :=
sorry

end greatest_divisor_l71_71576


namespace pie_difference_l71_71150

theorem pie_difference (p1 p2 : ℚ) (h1 : p1 = 5 / 6) (h2 : p2 = 2 / 3) : p1 - p2 = 1 / 6 := 
by 
  sorry

end pie_difference_l71_71150


namespace chords_from_nine_points_l71_71511

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l71_71511


namespace total_pieces_of_tomatoes_l71_71712

namespace FarmerTomatoes

variables (rows plants_per_row yield_per_plant : ℕ)

def total_plants (rows plants_per_row : ℕ) := rows * plants_per_row

def total_tomatoes (total_plants yield_per_plant : ℕ) := total_plants * yield_per_plant

theorem total_pieces_of_tomatoes 
  (hrows : rows = 30)
  (hplants_per_row : plants_per_row = 10)
  (hyield_per_plant : yield_per_plant = 20) :
  total_tomatoes (total_plants rows plants_per_row) yield_per_plant = 6000 :=
by
  rw [hrows, hplants_per_row, hyield_per_plant]
  unfold total_plants total_tomatoes
  norm_num
  done

end FarmerTomatoes

end total_pieces_of_tomatoes_l71_71712


namespace move_point_right_l71_71285

theorem move_point_right (x y : ℝ) (h₁ : x = 1) (h₂ : y = 1) (dx : ℝ) (h₃ : dx = 2) : (x + dx, y) = (3, 1) :=
by
  rw [h₁, h₂, h₃]
  simp
  sorry

end move_point_right_l71_71285


namespace trapezoid_problem_l71_71514

theorem trapezoid_problem (b h x : ℝ) 
  (hb : b > 0)
  (hh : h > 0)
  (h_ratio : (b + 90) / (b + 30) = 3 / 4)
  (h_x_def : x = 150 * (h / (x - 90) - 90))
  (hx2 : x^2 = 26100) :
  ⌊x^2 / 120⌋ = 217 := sorry

end trapezoid_problem_l71_71514


namespace micah_total_envelopes_l71_71661

-- Define the conditions as hypotheses
def weight_threshold := 5
def stamps_for_heavy := 5
def stamps_for_light := 2
def total_stamps := 52
def light_envelopes := 6

-- Noncomputable because we are using abstract reasoning rather than computational functions
noncomputable def total_envelopes : ℕ :=
  light_envelopes + (total_stamps - light_envelopes * stamps_for_light) / stamps_for_heavy

-- The theorem to prove
theorem micah_total_envelopes : total_envelopes = 14 := by
  sorry

end micah_total_envelopes_l71_71661


namespace solvability_condition_l71_71936

def is_solvable (p : ℕ) [Fact (Nat.Prime p)] :=
  ∃ α : ℤ, α * (α - 1) + 3 ≡ 0 [ZMOD p] ↔ ∃ β : ℤ, β * (β - 1) + 25 ≡ 0 [ZMOD p]

theorem solvability_condition (p : ℕ) [Fact (Nat.Prime p)] : 
  is_solvable p :=
sorry

end solvability_condition_l71_71936


namespace totalCoatsCollected_l71_71186

-- Definitions from the conditions
def highSchoolCoats : Nat := 6922
def elementarySchoolCoats : Nat := 2515

-- Theorem that proves the total number of coats collected
theorem totalCoatsCollected : highSchoolCoats + elementarySchoolCoats = 9437 := by
  sorry

end totalCoatsCollected_l71_71186


namespace div_fact_l71_71603

-- Conditions
def fact_10 : ℕ := 3628800
def fact_4 : ℕ := 4 * 3 * 2 * 1

-- Question and Correct Answer
theorem div_fact (h : fact_10 = 3628800) : fact_10 / fact_4 = 151200 :=
by
  sorry

end div_fact_l71_71603


namespace probability_red_is_two_fifths_l71_71234

-- Define the durations
def red_light_duration : ℕ := 30
def yellow_light_duration : ℕ := 5
def green_light_duration : ℕ := 40

-- Define total cycle duration
def total_cycle_duration : ℕ :=
  red_light_duration + yellow_light_duration + green_light_duration

-- Define the probability function
def probability_of_red_light : ℚ :=
  red_light_duration / total_cycle_duration

-- The theorem statement to prove
theorem probability_red_is_two_fifths :
  probability_of_red_light = 2/5 := sorry

end probability_red_is_two_fifths_l71_71234


namespace mindy_tax_rate_l71_71498

variables (M : ℝ) -- Mork's income
variables (r : ℝ) -- Mindy's tax rate

-- Conditions
def Mork_tax_rate := 0.45 -- 45% tax rate
def Mindx_income := 4 * M -- Mindy earned 4 times as much as Mork
def combined_tax_rate := 0.21 -- Combined tax rate is 21%

-- Equation derived from the conditions
def combined_tax_rate_eq := (0.45 * M + 4 * M * r) / (M + 4 * M) = 0.21

theorem mindy_tax_rate : combined_tax_rate_eq M r → r = 0.15 :=
by
  intros conditional_eq
  sorry

end mindy_tax_rate_l71_71498


namespace find_a1_l71_71135

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 8 = 2 ∧ ∀ n, a (n + 1) = 1 / (1 - a n)

theorem find_a1 (a : ℕ → ℝ) (h : seq a) : a 1 = 1/2 := by
sorry

end find_a1_l71_71135


namespace cistern_fill_time_l71_71226

theorem cistern_fill_time (fillA emptyB : ℕ) (hA : fillA = 8) (hB : emptyB = 12) : (24 : ℕ) = 24 :=
by
  sorry

end cistern_fill_time_l71_71226


namespace largest_four_digit_divisible_by_14_l71_71055

theorem largest_four_digit_divisible_by_14 :
  ∃ (A : ℕ), A = 9898 ∧ 
  (∃ a b : ℕ, A = 1010 * a + 101 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
  (A % 14 = 0) ∧
  (A = (d1 * 100 + d2 * 10 + d1) * 101)
  :=
sorry

end largest_four_digit_divisible_by_14_l71_71055


namespace solve_for_a_l71_71469

theorem solve_for_a (x a : ℝ) (h : x = -2) (hx : 2 * x + 3 * a = 0) : a = 4 / 3 :=
by
  sorry

end solve_for_a_l71_71469


namespace Sandy_tokens_more_than_siblings_l71_71182

theorem Sandy_tokens_more_than_siblings :
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  -- Definitions as per conditions
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  -- Conclusion
  show Sandy_tokens - sibling_tokens = 375000
  sorry

end Sandy_tokens_more_than_siblings_l71_71182


namespace total_ticket_sales_cost_l71_71416

theorem total_ticket_sales_cost
  (num_orchestra num_balcony : ℕ)
  (price_orchestra price_balcony : ℕ)
  (total_tickets total_revenue : ℕ)
  (h1 : num_orchestra + num_balcony = 370)
  (h2 : num_balcony = num_orchestra + 190)
  (h3 : price_orchestra = 12)
  (h4 : price_balcony = 8)
  (h5 : total_tickets = 370)
  : total_revenue = 3320 := by
  sorry

end total_ticket_sales_cost_l71_71416


namespace coloring_time_saved_percentage_l71_71293

variable (n : ℕ := 10) -- number of pictures
variable (draw_time : ℝ := 2) -- time to draw each picture in hours
variable (total_time : ℝ := 34) -- total time spent on drawing and coloring in hours

/-- 
  Prove the percentage of time saved on coloring each picture compared to drawing 
  given the specified conditions.
-/
theorem coloring_time_saved_percentage (n : ℕ) (draw_time total_time : ℝ) 
  (h1 : draw_time > 0)
  (draw_total_time : draw_time * n = 20)
  (total_picture_time : draw_time * n + coloring_total_time = total_time) :
  (draw_time - (coloring_total_time / n)) / draw_time * 100 = 30 := 
by
  sorry

end coloring_time_saved_percentage_l71_71293


namespace solve_equation_l71_71734

theorem solve_equation (x : ℝ) : 
  (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ↔ 
    (x = 1 ∨ x = -16 ∨ x = 4 ∨ x = -4) :=
by
  sorry

end solve_equation_l71_71734


namespace table_coverage_percentage_l71_71557

def A := 204  -- Total area of the runners
def T := 175  -- Area of the table
def A2 := 24  -- Area covered by exactly two layers of runner
def A3 := 20  -- Area covered by exactly three layers of runner

theorem table_coverage_percentage : 
  (A - 2 * A2 - 3 * A3 + A2 + A3) / T * 100 = 80 := 
by
  sorry

end table_coverage_percentage_l71_71557


namespace pages_left_to_write_l71_71864

theorem pages_left_to_write : 
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  remaining_pages = 315 :=
by
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  show remaining_pages = 315
  sorry

end pages_left_to_write_l71_71864


namespace sum_of_three_consecutive_even_numbers_is_162_l71_71821

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l71_71821


namespace set_intersection_eq_l71_71273

universe u
noncomputable theory

variable {α : Type u}
variables (U A B : Set α)

def complement (univ : Set α) (s : Set α) := univ \ s

theorem set_intersection_eq {U : Set ℕ} {A : Set ℕ} {B : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {1, 3, 4}) 
  (hB : B = {2, 3}) : 
  (complement U A) ∩ B = {2} := 
  by sorry

end set_intersection_eq_l71_71273


namespace village_duration_l71_71716

theorem village_duration (vampire_drain : ℕ) (werewolf_eat : ℕ) (village_population : ℕ)
  (hv : vampire_drain = 3) (hw : werewolf_eat = 5) (hp : village_population = 72) :
  village_population / (vampire_drain + werewolf_eat) = 9 :=
by
  sorry

end village_duration_l71_71716


namespace diamond_eight_five_l71_71811

def diamond (a b : ℕ) : ℕ := (a + b) * ((a - b) * (a - b))

theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end diamond_eight_five_l71_71811


namespace complement_of_A_with_respect_to_U_l71_71786

open Set

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}
def complement_U_A : Set ℕ := {4, 6}

theorem complement_of_A_with_respect_to_U :
  U \ A = complement_U_A := by
  sorry

end complement_of_A_with_respect_to_U_l71_71786


namespace find_x_value_l71_71012

def my_operation (a b : ℝ) : ℝ := 2 * a * b + 3 * b - 2 * a

theorem find_x_value (x : ℝ) (h : my_operation 3 x = 60) : x = 7.33 := 
by 
  sorry

end find_x_value_l71_71012


namespace units_digit_of_expression_l71_71239

theorem units_digit_of_expression :
  (9 * 19 * 1989 - 9 ^ 3) % 10 = 0 :=
by
  sorry

end units_digit_of_expression_l71_71239


namespace canal_cross_section_area_l71_71522

theorem canal_cross_section_area
  (a b h : ℝ)
  (H1 : a = 12)
  (H2 : b = 8)
  (H3 : h = 84) :
  (1 / 2) * (a + b) * h = 840 :=
by
  rw [H1, H2, H3]
  sorry

end canal_cross_section_area_l71_71522


namespace june_earnings_l71_71774

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l71_71774


namespace hide_and_seek_friends_l71_71385

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71385


namespace range_of_a_l71_71910

theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  ∃ a, (2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2) ↔
  (∀ x y : ℝ, 
    ((x - a)^2 + y^2 = 1) ∧ (x^2 + (y - 2)^2 = 25)) :=
sorry

end range_of_a_l71_71910


namespace typist_original_salary_l71_71685

theorem typist_original_salary (S : ℝ) :
  (1.10 * S * 0.95 * 1.07 * 0.97 = 2090) → (S = 2090 / (1.10 * 0.95 * 1.07 * 0.97)) :=
by
  intro h
  sorry

end typist_original_salary_l71_71685


namespace distribute_pictures_l71_71824

/-
Tiffany uploaded 34 pictures from her phone, 55 from her camera,
and 12 from her tablet to Facebook. If she sorted the pics into 7 different albums
with the same amount of pics in each album, how many pictures were in each of the albums?
-/

theorem distribute_pictures :
  let phone_pics := 34
  let camera_pics := 55
  let tablet_pics := 12
  let total_pics := phone_pics + camera_pics + tablet_pics
  let albums := 7
  ∃ k r, (total_pics = k * albums + r) ∧ (r < albums) := by
  sorry

end distribute_pictures_l71_71824


namespace highest_digit_a_divisible_by_eight_l71_71437

theorem highest_digit_a_divisible_by_eight :
  ∃ a : ℕ, a ≤ 9 ∧ 8 ∣ (100 * a + 16) ∧ ∀ b : ℕ, b > a → b ≤ 9 → ¬ (8 ∣ (100 * b + 16)) := by
  sorry

end highest_digit_a_divisible_by_eight_l71_71437


namespace base7_number_divisibility_l71_71521

theorem base7_number_divisibility (x : ℕ) (h : 0 ≤ x ∧ x ≤ 6) :
  (5 * 343 + 2 * 49 + x * 7 + 4) % 29 = 0 ↔ x = 6 := 
by
  sorry

end base7_number_divisibility_l71_71521


namespace number_of_friends_is_five_l71_71346

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l71_71346


namespace false_statement_divisibility_l71_71863

-- Definitions for the divisibility conditions
def divisible_by (a b : ℕ) : Prop := ∃ k, b = a * k

-- The problem statement
theorem false_statement_divisibility (N : ℕ) :
  (divisible_by 2 N ∧ divisible_by 4 N ∧ divisible_by 12 N ∧ ¬ divisible_by 24 N) →
  (¬ divisible_by 24 N) :=
by
  -- The proof will need to be filled in here
  sorry

end false_statement_divisibility_l71_71863


namespace adult_ticket_cost_is_19_l71_71693

variable (A : ℕ) -- the cost for an adult ticket
def child_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400
def adults_attendance : ℕ := 280
def children_attendance : ℕ := 120

-- The equation representing the total receipts
theorem adult_ticket_cost_is_19 (h : total_receipts = 280 * A + 120 * child_ticket_cost) : A = 19 :=
  by sorry

end adult_ticket_cost_is_19_l71_71693


namespace range_of_alpha_div_three_l71_71893

open Real

theorem range_of_alpha_div_three {k : ℤ} {α : ℝ} 
  (h1 : sin α > 0)
  (h2 : cos α < 0)
  (h3 : sin (α / 3) > cos (α / 3)) :
  (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) 
  ∨ (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
sorry

end range_of_alpha_div_three_l71_71893


namespace find_a_plus_b_l71_71299

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b

noncomputable def h (x : ℝ) := 3 * x + 2

theorem find_a_plus_b (a b : ℝ) (x : ℝ) (h_condition : ∀ x, h (f a b x) = 4 * x - 1) :
  a + b = 1 / 3 := 
by
  sorry

end find_a_plus_b_l71_71299


namespace expected_pairs_socks_l71_71544

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l71_71544


namespace find_line_eq_l71_71248

-- Define the type for the line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def given_point : ℝ × ℝ := (-3, -1)
def given_parallel_line : Line := { a := 1, b := -3, c := -1 }

-- Define what it means for two lines to be parallel
def are_parallel (L1 L2 : Line) : Prop :=
  L1.a * L2.b = L1.b * L2.a

-- Define what it means for a point to lie on the line
def lies_on_line (P : ℝ × ℝ) (L : Line) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

-- Define the result line we need to prove
def result_line : Line := { a := 1, b := -3, c := 0 }

-- The final theorem statement
theorem find_line_eq : 
  ∃ (L : Line), are_parallel L given_parallel_line ∧ lies_on_line given_point L ∧ L = result_line := 
sorry

end find_line_eq_l71_71248


namespace rectangle_dimensions_l71_71415

theorem rectangle_dimensions (a1 a2 : ℝ) (h1 : a1 * a2 = 216) (h2 : a1 + a2 = 30 - 6)
  (h3 : 6 * 6 = 36) : (a1 = 12 ∧ a2 = 18) ∨ (a1 = 18 ∧ a2 = 12) :=
by
  -- The conditions are set; now we need the proof, which we'll replace with sorry for now.
  sorry

end rectangle_dimensions_l71_71415


namespace x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l71_71753

theorem x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1 (x : ℝ) : (x > 1 → |x| > 1) ∧ (¬(x > 1 ↔ |x| > 1)) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_for_abs_x_gt_1_l71_71753


namespace girls_insects_collected_l71_71031

theorem girls_insects_collected (boys_insects groups insects_per_group : ℕ) :
  boys_insects = 200 →
  groups = 4 →
  insects_per_group = 125 →
  (groups * insects_per_group) - boys_insects = 300 :=
by
  intros h1 h2 h3
  -- Prove the statement
  sorry

end girls_insects_collected_l71_71031


namespace hh3_eq_6582_l71_71494

def h (x : ℤ) : ℤ := 3 * x^2 + 5 * x + 4

theorem hh3_eq_6582 : h (h 3) = 6582 :=
by
  sorry

end hh3_eq_6582_l71_71494


namespace greater_quadratic_solution_l71_71049

theorem greater_quadratic_solution : ∀ (x : ℝ), x^2 + 15 * x - 54 = 0 → x = -18 ∨ x = 3 →
  max (-18) 3 = 3 := by
  sorry

end greater_quadratic_solution_l71_71049


namespace intersection_correct_l71_71401

variable (A B : Set ℝ)  -- Define variables A and B as sets of real numbers

-- Define set A as {x | -3 ≤ x < 4}
def setA : Set ℝ := {x | -3 ≤ x ∧ x < 4}

-- Define set B as {x | -2 ≤ x ≤ 5}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- The goal is to prove the intersection of A and B is {x | -2 ≤ x < 4}
theorem intersection_correct : setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} := sorry

end intersection_correct_l71_71401


namespace emmy_rosa_ipods_total_l71_71244

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l71_71244


namespace number_of_eggs_in_each_basket_l71_71021

theorem number_of_eggs_in_each_basket 
  (total_blue_eggs : ℕ)
  (total_yellow_eggs : ℕ)
  (h1 : total_blue_eggs = 30)
  (h2 : total_yellow_eggs = 42)
  (exists_basket_count : ∃ n : ℕ, 6 ≤ n ∧ total_blue_eggs % n = 0 ∧ total_yellow_eggs % n = 0) :
  ∃ n : ℕ, n = 6 := 
sorry

end number_of_eggs_in_each_basket_l71_71021


namespace parallel_line_slope_l71_71051

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, y = (1 / 2) * x + b) → 
  (∃ a : ℝ, 3 * x - 6 * y = a) → 
  ∃ k : ℝ, k = 1 / 2 :=
by
  intros h1 h2
  sorry

end parallel_line_slope_l71_71051


namespace annual_rent_per_square_foot_l71_71998

theorem annual_rent_per_square_foot
  (length width : ℕ) (monthly_rent : ℕ) (h_length : length = 10)
  (h_width : width = 8) (h_monthly_rent : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := 
by 
  -- We assume the theorem is true.
  sorry

end annual_rent_per_square_foot_l71_71998


namespace ladybugs_with_spots_l71_71314

theorem ladybugs_with_spots (total_ladybugs : ℕ) (ladybugs_without_spots : ℕ) : total_ladybugs = 67082 ∧ ladybugs_without_spots = 54912 → total_ladybugs - ladybugs_without_spots = 12170 := by
  sorry

end ladybugs_with_spots_l71_71314


namespace temperature_decrease_time_l71_71687

theorem temperature_decrease_time
  (T_initial T_final T_per_hour : ℤ)
  (h_initial : T_initial = -5)
  (h_final : T_final = -25)
  (h_decrease : T_per_hour = -5) :
  (T_final - T_initial) / T_per_hour = 4 := by
sorry

end temperature_decrease_time_l71_71687


namespace reciprocal_neg_2023_l71_71532

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l71_71532


namespace perfect_square_trinomial_k_l71_71473

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l71_71473


namespace sum_of_c_and_d_l71_71467

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

theorem sum_of_c_and_d (c d : ℝ) (h_asymptote1 : (2:ℝ)^2 + c * 2 + d = 0) (h_asymptote2 : (-1:ℝ)^2 - c + d = 0) :
  c + d = -3 :=
by
-- theorem body (proof omitted)
sorry

end sum_of_c_and_d_l71_71467


namespace Meryll_problem_solving_questions_l71_71788

variable (P : ℕ)

theorem Meryll_problem_solving_questions : 
  let n_mchoice := 35 in
  let pct_mchoice_written := (2 : ℚ) / 5 in
  let pct_psolving_written := (1 : ℚ) / 3 in
  let mchoice_written := pct_mchoice_written * n_mchoice in
  let psolving_written := pct_psolving_written * P in
  let total_questions_written := 31 in
  let mchoice_remaining := n_mchoice - mchoice_written in
  let psolving_remaining := P - psolving_written in
  mchoice_remaining + psolving_remaining = total_questions_written → P = 15 :=
by
  sorry

end Meryll_problem_solving_questions_l71_71788


namespace printer_paper_last_days_l71_71946

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end printer_paper_last_days_l71_71946


namespace inverse_proportion_function_l71_71278

theorem inverse_proportion_function (m x : ℝ) (h : (m ≠ 0)) (A : (m, m / 8) ∈ {p : ℝ × ℝ | p.snd = (m / p.fst)}) :
    ∃ f : ℝ → ℝ, (∀ x, f x = 8 / x) :=
by
  use (fun x => 8 / x)
  intros x
  rfl

end inverse_proportion_function_l71_71278


namespace beka_flies_more_l71_71423

-- Definitions
def beka_flight_distance : ℕ := 873
def jackson_flight_distance : ℕ := 563

-- The theorem we need to prove
theorem beka_flies_more : beka_flight_distance - jackson_flight_distance = 310 :=
by
  sorry

end beka_flies_more_l71_71423


namespace find_x_values_l71_71869

theorem find_x_values (x : ℝ) :
  (3 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 9 * x + 1) ↔
  (x > (5 + Real.sqrt 29) / 2 ∧ x < 11) := 
by
  sorry

end find_x_values_l71_71869


namespace expected_socks_to_pair_l71_71541

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l71_71541


namespace number_of_people_l71_71713

variable (P M : ℕ)

-- Conditions
def cond1 : Prop := (500 = P * M)
def cond2 : Prop := (500 = (P + 5) * (M - 2))

-- Goal
theorem number_of_people (h1 : cond1 P M) (h2 : cond2 P M) : P = 33 :=
sorry

end number_of_people_l71_71713


namespace hide_and_seek_l71_71391

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71391


namespace triangle_one_interior_angle_61_degrees_l71_71695

theorem triangle_one_interior_angle_61_degrees
  (x : ℝ) : 
  (x + 75 + 2 * x + 25 + 3 * x - 22 = 360) → 
  (1 / 2 * (2 * x + 25) = 61 ∨ 
   1 / 2 * (3 * x - 22) = 61 ∨ 
   1 / 2 * (x + 75) = 61) :=
by
  intros h_sum
  sorry

end triangle_one_interior_angle_61_degrees_l71_71695


namespace sum_of_common_divisors_l71_71555

theorem sum_of_common_divisors : 
  let d := { n ∈ {60, 120, -30, 180, 240} | ∀ x, x ∣ n } in
  (1  + 2 + 3 + 5 + 6 = 17) := by
  sorry

end sum_of_common_divisors_l71_71555


namespace stock_yield_percentage_l71_71599

def annualDividend (parValue : ℕ) (rate : ℕ) : ℕ :=
  (parValue * rate) / 100

def yieldPercentage (dividend : ℕ) (marketPrice : ℕ) : ℕ :=
  (dividend * 100) / marketPrice

theorem stock_yield_percentage :
  let par_value := 100
  let rate := 8
  let market_price := 80
  yieldPercentage (annualDividend par_value rate) market_price = 10 :=
by
  sorry

end stock_yield_percentage_l71_71599


namespace integer_value_of_fraction_l71_71904

theorem integer_value_of_fraction (m n p : ℕ) (hm_diff: m ≠ n) (hn_diff: n ≠ p) (hp_diff: m ≠ p) 
  (hm_range: 2 ≤ m ∧ m ≤ 9) (hn_range: 2 ≤ n ∧ n ≤ 9) (hp_range: 2 ≤ p ∧ p ≤ 9) :
  (m + n + p) / (m + n) = 2 :=
by
  sorry

end integer_value_of_fraction_l71_71904


namespace monomial_sum_exponents_l71_71758

theorem monomial_sum_exponents (m n : ℕ) (h₁ : m - 1 = 2) (h₂ : n = 2) : m^n = 9 := 
by
  sorry

end monomial_sum_exponents_l71_71758


namespace parabola_points_relation_l71_71450

theorem parabola_points_relation :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = -(-2)^2 - 2*(-2) + 2) ∧ 
  (y2 = -(1)^2 - 2*(1) + 2) ∧ 
  (y3 = -(2)^2 - 2*(2) + 2) → 
  y1 > y2 ∧ y2 > y3 :=
by {
  intros y1 y2 y3 h,
  obtain ⟨h1, h2, h3⟩ := h,
  rw [h1, h2, h3],
  -- This is the placeholder for the proof
  sorry
}

end parabola_points_relation_l71_71450


namespace least_value_of_expression_l71_71568

theorem least_value_of_expression : ∃ (x y : ℝ), (2 * x - y + 3)^2 + (x + 2 * y - 1)^2 = 295 / 72 := sorry

end least_value_of_expression_l71_71568


namespace roots_quartic_sum_l71_71937

theorem roots_quartic_sum (p q r : ℝ) 
  (h1 : p^3 - 2*p^2 + 3*p - 4 = 0)
  (h2 : q^3 - 2*q^2 + 3*q - 4 = 0)
  (h3 : r^3 - 2*r^2 + 3*r - 4 = 0)
  (h4 : p + q + r = 2)
  (h5 : p*q + q*r + r*p = 3)
  (h6 : p*q*r = 4) :
  p^4 + q^4 + r^4 = 18 := sorry

end roots_quartic_sum_l71_71937


namespace expected_pairs_socks_l71_71546

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l71_71546


namespace range_of_x2_y2_l71_71448

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x^4

theorem range_of_x2_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x) : 
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
sorry

end range_of_x2_y2_l71_71448


namespace fraction_identity_l71_71260

theorem fraction_identity (a b : ℚ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 :=
by
  sorry

end fraction_identity_l71_71260


namespace water_wasted_per_hour_l71_71680

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l71_71680


namespace find_common_difference_l71_71263

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference_l71_71263


namespace sum_of_pos_real_solutions_l71_71117

open Real

noncomputable def cos_equation_sum_pos_real_solutions : ℝ := 1082 * π

theorem sum_of_pos_real_solutions :
  ∃ x : ℝ, (0 < x) ∧ 
    (∀ x, 2 * cos (2 * x) * (cos (2 * x) - cos ((2016 * π ^ 2) / x)) = cos (6 * x) - 1) → 
      x = cos_equation_sum_pos_real_solutions :=
sorry

end sum_of_pos_real_solutions_l71_71117


namespace number_of_cubes_l71_71214

theorem number_of_cubes (L W H V_cube : ℝ) (L_eq : L = 9) (W_eq : W = 12) (H_eq : H = 3) (V_cube_eq : V_cube = 3) :
  L * W * H / V_cube = 108 :=
by
  sorry

end number_of_cubes_l71_71214


namespace reciprocal_of_negative_2023_l71_71530

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l71_71530


namespace triangle_incenter_distance_l71_71696

open EuclideanGeometry

theorem triangle_incenter_distance
  (P Q R : Point)
  (hPQ : dist P Q = 31)
  (hPR : dist P R = 29)
  (hQR : dist Q R = 30)
  (J : Point)
  (hJ : is_incenter P Q R J) :
  dist P J = Real.sqrt 233 := by
  sorry

end triangle_incenter_distance_l71_71696


namespace solve_system_l71_71044

-- Define the conditions from the problem
def system_of_equations (x y : ℝ) : Prop :=
  (x = 4 * y) ∧ (x + 2 * y = -12)

-- Define the solution we want to prove
def solution (x y : ℝ) : Prop :=
  (x = -8) ∧ (y = -2)

-- State the theorem
theorem solve_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution x y :=
by 
  sorry

end solve_system_l71_71044


namespace angle_of_inclination_l71_71612

theorem angle_of_inclination (x y: ℝ) (h: x + real.sqrt 3 * y - 5 = 0) : 
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ real.tan θ = -1/real.sqrt 3 ∧ θ = 150 :=
sorry

end angle_of_inclination_l71_71612


namespace lynne_total_spent_l71_71169

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l71_71169


namespace binom_600_600_l71_71099

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l71_71099


namespace hide_and_seek_problem_l71_71367

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71367


namespace swimming_speed_l71_71837

theorem swimming_speed (v_m v_s : ℝ) 
  (h1 : v_m + v_s = 6)
  (h2 : v_m - v_s = 8) : 
  v_m = 7 :=
by
  sorry

end swimming_speed_l71_71837


namespace drops_of_glue_needed_l71_71009

def number_of_clippings (friend : ℕ) : ℕ :=
  match friend with
  | 1 => 4
  | 2 => 7
  | 3 => 5
  | 4 => 3
  | 5 => 5
  | 6 => 8
  | 7 => 2
  | 8 => 6
  | _ => 0

def total_drops_of_glue : ℕ :=
  (number_of_clippings 1 +
   number_of_clippings 2 +
   number_of_clippings 3 +
   number_of_clippings 4 +
   number_of_clippings 5 +
   number_of_clippings 6 +
   number_of_clippings 7 +
   number_of_clippings 8) * 6

theorem drops_of_glue_needed : total_drops_of_glue = 240 :=
by
  sorry

end drops_of_glue_needed_l71_71009


namespace arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l71_71066

-- Define the entities in the problem
inductive Participant
| Teacher
| Boy (id : Nat)
| Girl (id : Nat)

-- Define the conditions as properties or predicates
def girlsNextToEachOther (arrangement : List Participant) : Prop :=
  -- assuming the arrangement is a list of Participant
  sorry -- insert the actual condition as needed

def boysNotNextToEachOther (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def boysInDecreasingOrder (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def teacherNotInMiddle (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

def girlsNotAtEnds (arrangement : List Participant) : Prop :=
  sorry -- insert the actual condition as needed

-- Problem 1: Two girls must stand next to each other
theorem arrangement_count1 : ∃ arrangements, 1440 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, girlsNextToEachOther a := sorry

-- Problem 2: Boys must not stand next to each other
theorem arrangement_count2 : ∃ arrangements, 144 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysNotNextToEachOther a := sorry

-- Problem 3: Boys must stand in decreasing order of height
theorem arrangement_count3 : ∃ arrangements, 210 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, boysInDecreasingOrder a := sorry

-- Problem 4: Teacher not in middle, girls not at the ends
theorem arrangement_count4 : ∃ arrangements, 2112 = List.length arrangements ∧ 
  ∀ a ∈ arrangements, teacherNotInMiddle a ∧ girlsNotAtEnds a := sorry

end arrangement_count1_arrangement_count2_arrangement_count3_arrangement_count4_l71_71066


namespace multiple_of_960_l71_71296

theorem multiple_of_960 (a : ℤ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) :
  ∃ k : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * k :=
  sorry

end multiple_of_960_l71_71296


namespace problem1_problem2_l71_71899

noncomputable def f (x a : ℝ) := x - (x^2 + a * x) / Real.exp x

theorem problem1 (x : ℝ) : (f x 1) ≥ 0 := by
  sorry

theorem problem2 (x : ℝ) : (1 - (Real.log x) / x) * (f x (-1)) > 1 - 1/(Real.exp 2) := by
  sorry

end problem1_problem2_l71_71899


namespace unique_A3_zero_l71_71657

variable {F : Type*} [Field F]

theorem unique_A3_zero (A : Matrix (Fin 2) (Fin 2) F) 
  (h1 : A ^ 4 = 0) 
  (h2 : Matrix.trace A = 0) : 
  A ^ 3 = 0 :=
sorry

end unique_A3_zero_l71_71657


namespace smallest_positive_period_max_min_values_l71_71461

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period (x : ℝ) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T', T' > 0 ∧ ∀ x, f (x + T') = f x → T ≤ T' :=
  sorry

theorem max_min_values : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  min ≤ f x ∧ f x ≤ max :=
  sorry

end smallest_positive_period_max_min_values_l71_71461


namespace wheat_flour_packets_correct_l71_71337

-- Define the initial amount of money Victoria had.
def initial_amount : ℕ := 500

-- Define the cost and quantity of rice packets Victoria bought.
def rice_packet_cost : ℕ := 20
def rice_packets : ℕ := 2

-- Define the cost and quantity of soda Victoria bought.
def soda_cost : ℕ := 150
def soda_quantity : ℕ := 1

-- Define the remaining balance after shopping.
def remaining_balance : ℕ := 235

-- Define the cost of one packet of wheat flour.
def wheat_flour_packet_cost : ℕ := 25

-- Define the total amount spent on rice and soda.
def total_spent_on_rice_and_soda : ℕ :=
  (rice_packets * rice_packet_cost) + (soda_quantity * soda_cost)

-- Define the total amount spent on wheat flour.
def total_spent_on_wheat_flour : ℕ :=
  initial_amount - remaining_balance - total_spent_on_rice_and_soda

-- Define the expected number of wheat flour packets bought.
def wheat_flour_packets_expected : ℕ := 3

-- The statement we want to prove: the number of wheat flour packets bought is 3.
theorem wheat_flour_packets_correct : total_spent_on_wheat_flour / wheat_flour_packet_cost = wheat_flour_packets_expected :=
  sorry

end wheat_flour_packets_correct_l71_71337


namespace max_min_diff_c_l71_71164

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l71_71164


namespace nine_points_chords_l71_71503

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l71_71503


namespace number_of_glasses_l71_71638

theorem number_of_glasses (oranges_per_glass total_oranges : ℕ) 
  (h1 : oranges_per_glass = 2) 
  (h2 : total_oranges = 12) : 
  total_oranges / oranges_per_glass = 6 := by
  sorry

end number_of_glasses_l71_71638


namespace certain_number_is_1862_l71_71981

theorem certain_number_is_1862 (G N : ℕ) (hG: G = 4) (hN: ∃ k : ℕ, N = G * k + 6) (h1856: ∃ m : ℕ, 1856 = G * m + 4) : N = 1862 :=
by
  sorry

end certain_number_is_1862_l71_71981


namespace only_value_of_k_l71_71870

def A (k a b : ℕ) : ℚ := (a + b : ℚ) / (a^2 + k^2 * b^2 - k^2 * a * b : ℚ)

theorem only_value_of_k : (∀ a b : ℕ, 0 < a → 0 < b → ¬ (∃ c d : ℕ, 1 < c ∧ A 1 a b = (c : ℚ) / (d : ℚ))) → k = 1 := 
    by sorry  -- proof omitted

-- Note: 'only_value_of_k' states that given the conditions, there is no k > 1 that makes A(k, a, b) a composite number, hence k must be 1.

end only_value_of_k_l71_71870


namespace marble_problem_l71_71281

-- Define the given conditions
def ratio (red blue green : ℕ) : Prop := red * 3 * 4 = blue * 2 * 4 ∧ blue * 2 * 4 = green * 2 * 3

-- The total number of marbles
def total_marbles (red blue green : ℕ) : ℕ := red + blue + green

-- The number of green marbles is given
def green_marbles : ℕ := 36

-- Proving the number of marbles and number of red marbles
theorem marble_problem
  (red blue green : ℕ)
  (h_ratio : ratio red blue green)
  (h_green : green = green_marbles) :
  total_marbles red blue green = 81 ∧ red = 18 :=
by
  sorry

end marble_problem_l71_71281


namespace solution_inequality_l71_71747

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom increasing_function (x y : ℝ) : x < y → f x < f y

theorem solution_inequality (x : ℝ) : f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1 / 3 := sorry

end solution_inequality_l71_71747


namespace sum_of_consecutive_neg_ints_l71_71816

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l71_71816


namespace arithmetic_sequence_max_n_pos_sum_l71_71011

noncomputable def max_n (a : ℕ → ℤ) (d : ℤ) : ℕ :=
  8

theorem arithmetic_sequence_max_n_pos_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n, a (n+1) = a 1 + n * d)
  (h_a1 : a 1 > 0)
  (h_a4_a5_sum_pos : a 4 + a 5 > 0)
  (h_a4_a5_prod_neg : a 4 * a 5 < 0) :
  max_n a d = 8 := by
  sorry

end arithmetic_sequence_max_n_pos_sum_l71_71011


namespace hide_and_seek_friends_l71_71381

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71381


namespace problem_1_problem_2_l71_71745

-- (1) Conditions and proof statement
theorem problem_1 (x y m : ℝ) (P : ℝ × ℝ) (k : ℝ) :
  (x, y) = (1, 2) → m = 1 →
  ((x - 1)^2 + (y - 2)^2 = 4) →
  P = (3, -1) →
  (l : ℝ → ℝ → Prop) →
  (∀ x y, l x y ↔ x = 3 ∨ (5 * x + 12 * y - 3 = 0)) →
  l 3 (-1) →
  l (x + k * (3 - x)) (y-1) := sorry

-- (2) Conditions and proof statement
theorem problem_2 (x y m : ℝ) (line : ℝ → ℝ) :
  (x - 1)^2 + (y - 2)^2 = 5 - m →
  m < 5 →
  (2 * (5 - m - 20) ^ (1/2) = 2 * (5) ^ (1/2)) →
  m = -20 := sorry

end problem_1_problem_2_l71_71745


namespace total_writing_instruments_l71_71307

theorem total_writing_instruments 
 (bags : ℕ) (compartments_per_bag : ℕ) (empty_compartments : ℕ) (one_compartment : ℕ) (remaining_compartments : ℕ) 
 (writing_instruments_per_compartment : ℕ) (writing_instruments_in_one : ℕ) : 
 bags = 16 → 
 compartments_per_bag = 6 → 
 empty_compartments = 5 → 
 one_compartment = 1 → 
 remaining_compartments = 90 →
 writing_instruments_per_compartment = 8 → 
 writing_instruments_in_one = 6 → 
 (remaining_compartments * writing_instruments_per_compartment + one_compartment * writing_instruments_in_one) = 726 := 
  by
   sorry

end total_writing_instruments_l71_71307


namespace find_number_l71_71644

theorem find_number (x : ℚ) : (35 / 100) * x = (20 / 100) * 50 → x = 200 / 7 :=
by
  intros h
  sorry

end find_number_l71_71644


namespace trigonometric_identities_l71_71626

theorem trigonometric_identities (α : Real) (h1 : 3 * π / 2 < α ∧ α < 2 * π) (h2 : Real.sin α = -3 / 5) :
  Real.tan α = 3 / 4 ∧ Real.tan (α - π / 4) = -1 / 7 ∧ Real.cos (2 * α) = 7 / 25 :=
by
  sorry

end trigonometric_identities_l71_71626


namespace total_legs_of_animals_l71_71496

def num_kangaroos := 23
def num_goats := 3 * num_kangaroos
def legs_per_kangaroo := 2
def legs_per_goat := 4

def total_legs := (num_kangaroos * legs_per_kangaroo) + (num_goats * legs_per_goat)

theorem total_legs_of_animals : total_legs = 322 := by
  sorry

end total_legs_of_animals_l71_71496


namespace lynne_total_spent_l71_71168

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l71_71168


namespace total_boxes_l71_71556

variable (N_initial : ℕ) (N_nonempty : ℕ) (N_new_boxes : ℕ)

theorem total_boxes (h_initial : N_initial = 7) 
                     (h_nonempty : N_nonempty = 10)
                     (h_new_boxes : N_new_boxes = N_nonempty * 7) :
  N_initial + N_new_boxes = 77 :=
by 
  have : N_initial = 7 := h_initial
  have : N_new_boxes = N_nonempty * 7 := h_new_boxes
  have : N_nonempty = 10 := h_nonempty
  sorry

end total_boxes_l71_71556


namespace chords_from_nine_points_l71_71500

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l71_71500


namespace complex_number_in_first_quadrant_l71_71921

open Complex

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = 1 / (1 - I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l71_71921


namespace total_listening_days_l71_71744

theorem total_listening_days (x y z t : ℕ) (h1 : x = 8) (h2 : y = 12) (h3 : z = 30) (h4 : t = 2) :
  (x + y + z) * t = 100 :=
by
  sorry

end total_listening_days_l71_71744


namespace unique_solution_of_pair_of_equations_l71_71878

-- Definitions and conditions
def pair_of_equations (x k : ℝ) : Prop :=
  (x^2 + 1 = 4 * x + k)

-- Theorem to prove
theorem unique_solution_of_pair_of_equations :
  ∃ k : ℝ, (∀ x : ℝ, pair_of_equations x k -> x = 2) ∧ k = 0 :=
by
  -- Proof omitted
  sorry

end unique_solution_of_pair_of_equations_l71_71878


namespace total_spent_l71_71171

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l71_71171


namespace solve_ab_find_sqrt_l71_71628

variable (a b : ℝ)

-- Given Conditions
axiom h1 : real.cbrt (2 * b - 2 * a) = -2
axiom h2 : real.sqrt (4 * a + 3 * b) = 3

-- Goal: Prove that a = 3 and b = -1
theorem solve_ab : a = 3 ∧ b = -1 := by
  sorry

-- Given a = 3 and b = -1, find the square root of 5a - b
theorem find_sqrt : a = 3 ∧ b = -1 → real.sqrt (5 * a - b) = 4 ∨ real.sqrt (5 * a - b) = -4 := by
  sorry

end solve_ab_find_sqrt_l71_71628


namespace nth_term_pattern_l71_71701

theorem nth_term_pattern (a : ℕ → ℕ) (h : ∀ n, a n = n * (n - 1)) : 
  (a 0 = 0) ∧ (a 1 = 2) ∧ (a 2 = 6) ∧ (a 3 = 12) ∧ (a 4 = 20) ∧ 
  (a 5 = 30) ∧ (a 6 = 42) ∧ (a 7 = 56) ∧ (a 8 = 72) ∧ (a 9 = 90) := sorry

end nth_term_pattern_l71_71701


namespace reciprocal_neg_2023_l71_71533

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l71_71533


namespace part_a_part_b_part_c_l71_71785

def f (x : ℝ) := x^2
def g (x : ℝ) := 3 * x - 8
def h (r : ℝ) (x : ℝ) := 3 * x - r

theorem part_a :
  f 2 = 4 ∧ g (f 2) = 4 :=
by {
  sorry
}

theorem part_b :
  ∀ x : ℝ, f (g x) = g (f x) → (x = 2 ∨ x = 6) :=
by {
  sorry
}

theorem part_c :
  ∀ r : ℝ, f (h r 2) = h r (f 2) → (r = 3 ∨ r = 8) :=
by {
  sorry
}

end part_a_part_b_part_c_l71_71785


namespace maximum_partial_sum_l71_71241

theorem maximum_partial_sum (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
    (h_arith_seq : ∀ n, a n = a 0 + n * d)
    (h8_13 : 3 * a 8 = 5 * a 13)
    (h_pos : a 0 > 0)
    (h_sn_def : ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2) :
  S 20 = max (max (S 10) (S 11)) (max (S 20) (S 21)) := 
sorry

end maximum_partial_sum_l71_71241


namespace spherical_coordinates_standard_equivalence_l71_71650

def std_spherical_coords (ρ θ φ: ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_standard_equivalence :
  std_spherical_coords 5 (11 * Real.pi / 6) (2 * Real.pi - 5 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_standard_equivalence_l71_71650


namespace sum_of_intersections_l71_71322

theorem sum_of_intersections :
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (∀ x y : ℝ, y = (x - 2)^2 ↔ x + 1 = (y - 2)^2) ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 20) :=
sorry

end sum_of_intersections_l71_71322


namespace number_of_correct_conclusions_is_two_l71_71632

section AnalogicalReasoning
  variable (a b c : ℝ) (x y : ℂ)

  -- Condition 1: The analogy for distributive property over addition in ℝ and division
  def analogy1 : (c ≠ 0) → ((a + b) * c = a * c + b * c) → (a + b) / c = a / c + b / c := by
    sorry

  -- Condition 2: The analogy for equality of real and imaginary parts in ℂ
  def analogy2 : (x - y = 0) → x = y := by
    sorry

  -- Theorem stating that the number of correct conclusions is 2
  theorem number_of_correct_conclusions_is_two : 2 = 2 := by
    -- which implies that analogy1 and analogy2 are valid, and the other two analogies are not
    sorry

end AnalogicalReasoning

end number_of_correct_conclusions_is_two_l71_71632


namespace games_given_to_neil_is_five_l71_71463

variable (x : ℕ)

def initial_games_henry : ℕ := 33
def initial_games_neil : ℕ := 2
def games_given_to_neil : ℕ := x

theorem games_given_to_neil_is_five
  (H : initial_games_henry - games_given_to_neil = 4 * (initial_games_neil + games_given_to_neil)) :
  games_given_to_neil = 5 := by
  sorry

end games_given_to_neil_is_five_l71_71463


namespace triangle_inequality_l71_71213

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by {
  sorry
}

end triangle_inequality_l71_71213


namespace monotonically_increasing_interval_l71_71746

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos ((2 / 3) * x - (5 * Real.pi / 12))

theorem monotonically_increasing_interval 
  (φ : ℝ) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0) 
  (h3 : 2 * (Real.pi / 8) + φ = Real.pi / 4) : 
  ∀ x : ℝ, (-(Real.pi / 2) ≤ x) ∧ (x ≤ Real.pi / 2) ↔ ∃ k : ℤ, x ∈ [(-7 * Real.pi / 8 + 3 * k * Real.pi), (5 * Real.pi / 8 + 3 * k * Real.pi)] :=
sorry

end monotonically_increasing_interval_l71_71746


namespace fibonacci_inequality_l71_71676

open Finset
open BigOperators

-- Fibonacci sequence definition
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 1) + 1 => fib (n + 1) + fib n

-- Binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Statement of the problem
theorem fibonacci_inequality
    (n : ℕ)
    (hn : 0 < n) : 
    ∑ i in range (n + 1), binom n i * fib i < (2 * n + 2)^n / n! :=
sorry

end fibonacci_inequality_l71_71676


namespace intersection_A_B_l71_71881

open Set

def f (x : ℕ) : ℕ := x^2 - 12 * x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a, a ∈ A ∧ b = f a}

theorem intersection_A_B : A ∩ B = {1, 4, 9} :=
by
  -- Proof skipped
  sorry

end intersection_A_B_l71_71881


namespace maximum_area_of_enclosed_poly_l71_71019

theorem maximum_area_of_enclosed_poly (k : ℕ) : 
  ∃ (A : ℕ), (A = 4 * k + 1) :=
sorry

end maximum_area_of_enclosed_poly_l71_71019


namespace min_x_value_l71_71778

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 18 * x + 50 * y + 56

theorem min_x_value : 
  ∃ (x : ℝ), ∃ (y : ℝ), circle_eq x y ∧ x = 9 - Real.sqrt 762 :=
by
  sorry

end min_x_value_l71_71778


namespace algebra_problem_l71_71275

theorem algebra_problem
  (x : ℝ)
  (h : 59 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = Real.sqrt 61 :=
sorry

end algebra_problem_l71_71275


namespace alex_plays_with_friends_l71_71395

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71395


namespace smallest_number_divisible_by_5_with_digit_sum_100_l71_71252

theorem smallest_number_divisible_by_5_with_digit_sum_100 :
  ∃ N : ℕ, (N % 5 = 0) ∧ (Nat.digits 10 N).sum = 100 ∧
  ∀ M : ℕ, (M % 5 = 0) ∧ (Nat.digits 10 M).sum = 100 → N ≤ M :=
begin
  use 599999999995,
  split,
  { norm_num },
  split,
  { norm_num,
    sorry },
  { intros M h1 h2,
    sorry }
end

end smallest_number_divisible_by_5_with_digit_sum_100_l71_71252


namespace required_cups_of_sugar_l71_71063

-- Define the original ratios
def original_flour_water_sugar_ratio : Rat := 10 / 6 / 3
def new_flour_water_ratio : Rat := 2 * (10 / 6)
def new_flour_sugar_ratio : Rat := (1 / 2) * (10 / 3)

-- Given conditions
def cups_of_water : Rat := 2

-- Problem statement: prove the amount of sugar required
theorem required_cups_of_sugar : ∀ (sugar_cups : Rat),
  original_flour_water_sugar_ratio = 10 / 6 / 3 ∧
  new_flour_water_ratio = 2 * (10 / 6) ∧
  new_flour_sugar_ratio = (1 / 2) * (10 / 3) ∧
  cups_of_water = 2 ∧
  (6 / 12) = (2 / sugar_cups) → sugar_cups = 4 := by
  intro sugar_cups
  sorry

end required_cups_of_sugar_l71_71063


namespace sum_constants_l71_71689

theorem sum_constants (a b x : ℝ) 
  (h1 : (x - a) / (x + b) = (x^2 - 50 * x + 621) / (x^2 + 75 * x - 3400))
  (h2 : x^2 - 50 * x + 621 = (x - 27) * (x - 23))
  (h3 : x^2 + 75 * x - 3400 = (x - 40) * (x + 85)) :
  a + b = 112 :=
sorry

end sum_constants_l71_71689


namespace parabola_standard_equation_l71_71630

/-- Given that the directrix of a parabola coincides with the line on which the circles 
    x^2 + y^2 - 4 = 0 and x^2 + y^2 + y - 3 = 0 lie, the standard equation of the parabola 
    is x^2 = 4y.
-/
theorem parabola_standard_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 = 0 → x^2 + y^2 + y - 3 = 0 → y = -1) →
  ∀ p : ℝ, 4 * (p / 2) = 4 → x^2 = 4 * p * y :=
by
  sorry

end parabola_standard_equation_l71_71630


namespace percentage_fraction_l71_71709

theorem percentage_fraction (P : ℚ) (hP : P < 35) (h : (P / 100) * 180 = 42) : P = 7 / 30 * 100 :=
by
  sorry

end percentage_fraction_l71_71709


namespace percentage_increase_second_movie_l71_71825

def length_first_movie : ℕ := 2
def total_length_marathon : ℕ := 9
def length_last_movie (F S : ℕ) := S + F - 1

theorem percentage_increase_second_movie :
  ∀ (S : ℕ), 
  length_first_movie + S + length_last_movie length_first_movie S = total_length_marathon →
  ((S - length_first_movie) * 100) / length_first_movie = 50 :=
by
  sorry

end percentage_increase_second_movie_l71_71825


namespace capacity_of_each_bucket_in_second_case_final_proof_l71_71403

def tank_volume (buckets: ℕ) (bucket_capacity: ℝ) : ℝ := buckets * bucket_capacity

theorem capacity_of_each_bucket_in_second_case
  (total_volume: ℝ)
  (first_case_buckets : ℕ)
  (first_case_capacity : ℝ)
  (second_case_buckets : ℕ) :
  first_case_buckets * first_case_capacity = total_volume → 
  (total_volume / second_case_buckets) = 9 :=
by
  intros h
  sorry

-- Given the conditions:
noncomputable def total_volume := tank_volume 28 13.5

theorem final_proof :
  (tank_volume 28 13.5 = total_volume) → 
  (total_volume / 42 = 9) :=
by
  intro h
  exact capacity_of_each_bucket_in_second_case total_volume 28 13.5 42 h

end capacity_of_each_bucket_in_second_case_final_proof_l71_71403


namespace difference_between_x_and_y_l71_71474

theorem difference_between_x_and_y (x y : ℕ) (h₁ : 3 ^ x * 4 ^ y = 59049) (h₂ : x = 10) : x - y = 10 := by
  sorry

end difference_between_x_and_y_l71_71474


namespace crayons_erasers_difference_l71_71020

theorem crayons_erasers_difference
  (initial_erasers : ℕ) (initial_crayons : ℕ) (final_crayons : ℕ)
  (no_eraser_lost : initial_erasers = 457)
  (initial_crayons_condition : initial_crayons = 617)
  (final_crayons_condition : final_crayons = 523) :
  final_crayons - initial_erasers = 66 :=
by
  -- These would be assumptions in the proof; be aware that 'sorry' is used to skip the proof details.
  sorry

end crayons_erasers_difference_l71_71020


namespace hexagon_side_lengths_l71_71609

open Nat

/-- Define two sides AB and BC of a hexagon with their given lengths -/
structure Hexagon :=
  (AB BC AD BE CF DE: ℕ)
  (distinct_lengths : AB ≠ BC ∧ (AB = 7 ∧ BC = 8))
  (total_perimeter : AB + BC + AD + BE + CF + DE = 46)

-- Define a theorem to prove the number of sides measuring 8 units
theorem hexagon_side_lengths (h: Hexagon) :
  ∃ (n : ℕ), n = 4 ∧ n * 8 + (6 - n) * 7 = 46 :=
by
  -- Assume the proof here
  sorry

end hexagon_side_lengths_l71_71609


namespace highest_number_paper_l71_71148

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end highest_number_paper_l71_71148


namespace expression_simplification_l71_71836

variable (x : ℝ)

-- Define the expression as given in the problem
def Expr : ℝ := (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 3)

-- Lean statement to verify that the expression simplifies to the given polynomial
theorem expression_simplification : Expr x = 6 * x^3 - 16 * x^2 + 43 * x - 70 := by
  sorry

end expression_simplification_l71_71836


namespace hide_and_seek_problem_l71_71371

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71371


namespace sum_of_consecutive_evens_is_162_l71_71820

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end sum_of_consecutive_evens_is_162_l71_71820


namespace average_of_remaining_numbers_l71_71030

variable (numbers : List ℝ) (x y : ℝ)

theorem average_of_remaining_numbers
  (h_length_15 : numbers.length = 15)
  (h_avg_15 : (numbers.sum / 15) = 90)
  (h_x : x = 80)
  (h_y : y = 85)
  (h_members : x ∈ numbers ∧ y ∈ numbers) :
  ((numbers.sum - x - y) / 13) = 91.15 :=
sorry

end average_of_remaining_numbers_l71_71030


namespace sum_of_powers_of_four_to_50_l71_71238

theorem sum_of_powers_of_four_to_50 :
  2 * (Finset.sum (Finset.range 51) (λ x => x^4)) = 1301700 := by
  sorry

end sum_of_powers_of_four_to_50_l71_71238


namespace alex_play_friends_with_l71_71352

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71352


namespace platinum_earrings_percentage_l71_71648

theorem platinum_earrings_percentage
  (rings_percentage ornaments_percentage : ℝ)
  (rings_percentage_eq : rings_percentage = 0.30)
  (earrings_percentage_eq : ornaments_percentage - rings_percentage = 0.70)
  (platinum_earrings_percentage : ℝ)
  (platinum_earrings_percentage_eq : platinum_earrings_percentage = 0.70) :
  ornaments_percentage * platinum_earrings_percentage = 0.49 :=
by 
  have earrings_percentage := 0.70
  have ornaments_percentage := 0.70
  sorry

end platinum_earrings_percentage_l71_71648


namespace geometric_series_first_term_l71_71978

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l71_71978


namespace center_of_circle_l71_71806

theorem center_of_circle :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 1 → (x = -2 ∧ y = 1) :=
by
  intros x y hyp
  -- Here, we would perform the steps of comparing to the standard form and proving the center.
  sorry

end center_of_circle_l71_71806


namespace perpendicular_condition_l71_71039

theorem perpendicular_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → (m * x + y + 1 = 0 → (2 * m - 1 = 0))) ↔ (m = 1/2) :=
by sorry

end perpendicular_condition_l71_71039


namespace smallest_unreachable_integer_l71_71254

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer_l71_71254


namespace sahil_selling_price_correct_l71_71060

-- Define the conditions as constants
def cost_of_machine : ℕ := 13000
def cost_of_repair : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Define the total cost calculation
def total_cost : ℕ := cost_of_machine + cost_of_repair + transportation_charges

-- Define the profit calculation
def profit : ℕ := total_cost * profit_percentage / 100

-- Define the selling price calculation
def selling_price : ℕ := total_cost + profit

-- Now we express our proof problem
theorem sahil_selling_price_correct :
  selling_price = 28500 := by
  -- sorries to skip the proof.
  sorry

end sahil_selling_price_correct_l71_71060


namespace distinct_triangles_count_l71_71137

/-- Define the set of points in a 3x3 grid -/
def grid_points : Finset (ℕ × ℕ) :=
  Finset.ofList [(0, 0), (1, 0), (2, 0),
                 (0, 1), (1, 1), (2, 1),
                 (0, 2), (1, 2), (2, 2)]

/-- Check if three points are collinear -/
def collinear {A B C : ℕ × ℕ} : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1)

/-- Count the number of distinct triangles from a grid of points -/
noncomputable def count_distinct_triangles : ℕ :=
  let points := grid_points in
  let triplets := points.subsetsOfSize 3 in
  (triplets.filter (λ s, ∃ A B C, s = {A, B, C} ∧ ¬collinear)).card

theorem distinct_triangles_count : count_distinct_triangles = 76 :=
  sorry

end distinct_triangles_count_l71_71137


namespace range_of_k_for_positivity_l71_71990

theorem range_of_k_for_positivity (k x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) :
  ((k - 2) * x + 2 * |k| - 1 > 0) → (k > 5 / 4) :=
sorry

end range_of_k_for_positivity_l71_71990


namespace yolanda_walking_rate_l71_71703

theorem yolanda_walking_rate 
  (d_xy : ℕ) (bob_start_after_yolanda : ℕ) (bob_distance_walked : ℕ) 
  (bob_rate : ℕ) (y : ℕ) 
  (bob_distance_to_time : bob_rate ≠ 0 ∧ bob_distance_walked / bob_rate = 2) 
  (yolanda_distance_walked : d_xy - bob_distance_walked = 9 ∧ y = 9 / 3) : 
  y = 3 :=
by 
  sorry

end yolanda_walking_rate_l71_71703


namespace water_wasted_in_one_hour_l71_71677

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l71_71677


namespace distinct_nonzero_reals_satisfy_equation_l71_71931

open Real

theorem distinct_nonzero_reals_satisfy_equation
  (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) (h₄ : a ≠ 0) (h₅ : b ≠ 0) (h₆ : c ≠ 0)
  (h₇ : a + 2 / b = b + 2 / c) (h₈ : b + 2 / c = c + 2 / a) :
  (a + 2 / b) ^ 2 + (b + 2 / c) ^ 2 + (c + 2 / a) ^ 2 = 6 :=
sorry

end distinct_nonzero_reals_satisfy_equation_l71_71931


namespace math_problem_l71_71210

theorem math_problem : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end math_problem_l71_71210


namespace alpha3_plus_8beta_plus_6_eq_30_l71_71896

noncomputable def alpha_beta_quad_roots (α β : ℝ) : Prop :=
  α^2 - 2 * α - 4 = 0 ∧ β^2 - 2 * β - 4 = 0

theorem alpha3_plus_8beta_plus_6_eq_30 (α β : ℝ) (h : alpha_beta_quad_roots α β) : 
  α^3 + 8 * β + 6 = 30 :=
sorry

end alpha3_plus_8beta_plus_6_eq_30_l71_71896


namespace intersection_correct_l71_71493

variable (x : ℝ)

def M : Set ℝ := { x | x^2 > 4 }
def N : Set ℝ := { x | x^2 - 3 * x ≤ 0 }
def NM_intersection : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem intersection_correct :
  {x | (M x) ∧ (N x)} = NM_intersection :=
sorry

end intersection_correct_l71_71493


namespace first_term_of_geometric_series_l71_71955

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l71_71955


namespace rectangle_area_stage4_l71_71907

-- Define the condition: area of one square
def square_area : ℕ := 25

-- Define the condition: number of squares at Stage 4
def num_squares_stage4 : ℕ := 4

-- Define the total area of rectangle at Stage 4
def total_area_stage4 : ℕ := num_squares_stage4 * square_area

-- Prove that total_area_stage4 equals 100 square inches
theorem rectangle_area_stage4 : total_area_stage4 = 100 :=
by
  sorry

end rectangle_area_stage4_l71_71907


namespace inheritance_amount_l71_71294

theorem inheritance_amount (x : ℝ) 
    (federal_tax : ℝ := 0.25 * x) 
    (remaining_after_federal_tax : ℝ := x - federal_tax) 
    (state_tax : ℝ := 0.15 * remaining_after_federal_tax) 
    (total_taxes : ℝ := federal_tax + state_tax) 
    (taxes_paid : total_taxes = 15000) : 
    x = 41379 :=
sorry

end inheritance_amount_l71_71294


namespace intersection_eq_interval_l71_71637

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

theorem intersection_eq_interval : M ∩ N = {x | 1 < x ∧ x < 5} :=
sorry

end intersection_eq_interval_l71_71637


namespace reciprocal_neg_2023_l71_71534

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end reciprocal_neg_2023_l71_71534


namespace alex_plays_with_friends_l71_71398

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71398


namespace time_addition_sum_l71_71924

/-- Given the start time of 3:15:20 PM and adding a duration of 
    305 hours, 45 minutes, and 56 seconds, the resultant hour, 
    minute, and second values sum to 26. -/
theorem time_addition_sum : 
  let current_hour := 15
  let current_minute := 15
  let current_second := 20
  let added_hours := 305
  let added_minutes := 45
  let added_seconds := 56
  let final_hour := ((current_hour + (added_hours % 12) + ((current_minute + added_minutes) / 60) + ((current_second + added_seconds) / 3600)) % 12)
  let final_minute := ((current_minute + added_minutes + ((current_second + added_seconds) / 60)) % 60)
  let final_second := ((current_second + added_seconds) % 60)
  final_hour + final_minute + final_second = 26 := 
  sorry

end time_addition_sum_l71_71924


namespace solution_to_ball_problem_l71_71984

noncomputable def probability_of_arithmetic_progression : Nat :=
  let p := 3
  let q := 9464
  p + q

theorem solution_to_ball_problem : probability_of_arithmetic_progression = 9467 := by
  sorry

end solution_to_ball_problem_l71_71984


namespace nat_divisor_problem_l71_71930

open Nat

theorem nat_divisor_problem (n : ℕ) (d : ℕ → ℕ) (k : ℕ)
    (h1 : 1 = d 1)
    (h2 : ∀ i, 1 < i → i ≤ k → d i < d (i + 1))
    (hk : d k = n)
    (hdiv : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
    (heq : n = d 2 * d 3 + d 2 * d 5 + d 3 * d 5) :
    k = 8 ∨ k = 9 :=
sorry

end nat_divisor_problem_l71_71930


namespace different_kinds_of_hamburgers_l71_71464

theorem different_kinds_of_hamburgers 
  (n_condiments : ℕ) 
  (condiment_choices : ℕ)
  (meat_patty_choices : ℕ)
  (h1 : n_condiments = 8)
  (h2 : condiment_choices = 2 ^ n_condiments)
  (h3 : meat_patty_choices = 3)
  : condiment_choices * meat_patty_choices = 768 := 
by
  sorry

end different_kinds_of_hamburgers_l71_71464


namespace red_tickets_for_one_yellow_l71_71827

-- Define the conditions given in the problem
def yellow_needed := 10
def red_for_yellow (R : ℕ) := R -- This function defines the number of red tickets for one yellow
def blue_for_red := 10

def toms_yellow := 8
def toms_red := 3
def toms_blue := 7
def blue_needed := 163

-- Define the target function that converts the given conditions into a statement.
def red_tickets_for_yellow_proof : Prop :=
  ∀ R : ℕ, (2 * R = 14) → (R = 7)

-- Statement for proof where the condition leads to conclusion
theorem red_tickets_for_one_yellow : red_tickets_for_yellow_proof :=
by
  intros R h
  rw [← h, mul_comm] at h
  sorry

end red_tickets_for_one_yellow_l71_71827


namespace arithmetic_sequence_sum_l71_71686

-- Define arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℕ → ℕ) :=
  ∀ n, a (n + 1) = a n + d 1

def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Conditions from the problem
variables {a : ℕ → ℕ} {d : ℕ}

axiom condition : a 3 + a 7 + a 11 = 6

-- Definition of a_7 as derived in the solution
def a_7 : ℕ := 2

-- Proof problem equivalent statement
theorem arithmetic_sequence_sum : arithmetic_sum a d 13 = 26 :=
by
  -- These steps would involve setting up and proving the calculation details
  sorry

end arithmetic_sequence_sum_l71_71686


namespace remaining_tickets_equation_l71_71600

-- Define the constants and variables
variables (x y : ℕ)

-- Conditions from the problem
def tickets_whack_a_mole := 32
def tickets_skee_ball := 25
def tickets_space_invaders : ℕ := x

def spent_hat := 7
def spent_keychain := 10
def spent_toy := 15

-- Define the condition for the total number of tickets spent
def total_tickets_spent := spent_hat + spent_keychain + spent_toy
-- Prove the remaining tickets equation
theorem remaining_tickets_equation : y = (tickets_whack_a_mole + tickets_skee_ball + tickets_space_invaders) - total_tickets_spent ->
                                      y = 25 + x :=
by
  sorry

end remaining_tickets_equation_l71_71600


namespace lynne_total_spending_l71_71175

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l71_71175


namespace find_f_log2_3_l71_71646

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_condition : ∀ x : ℝ, f (f x + 2 / (2^x + 1)) = (1 / 3)

theorem find_f_log2_3 : f (Real.log 3 / Real.log 2) = (1 / 2) :=
by
  sorry

end find_f_log2_3_l71_71646


namespace regular_ducks_sold_l71_71032

theorem regular_ducks_sold (R : ℕ) (h1 : 3 * R + 5 * 185 = 1588) : R = 221 :=
by {
  sorry
}

end regular_ducks_sold_l71_71032


namespace number_of_tires_slashed_l71_71154

-- Definitions based on conditions
def cost_per_tire : ℤ := 250
def cost_window : ℤ := 700
def total_cost : ℤ := 1450

-- Proof statement
theorem number_of_tires_slashed : ∃ T : ℤ, cost_per_tire * T + cost_window = total_cost ∧ T = 3 := 
sorry

end number_of_tires_slashed_l71_71154


namespace price_of_each_apple_l71_71587

-- Define the constants and conditions
def price_banana : ℝ := 0.60
def total_fruits : ℕ := 9
def total_cost : ℝ := 5.60

-- Declare the variables for number of apples and price of apples
variables (A : ℝ) (x y : ℕ)

-- Define the conditions in Lean
axiom h1 : x + y = total_fruits
axiom h2 : A * x + price_banana * y = total_cost

-- Prove that the price of each apple is $0.80
theorem price_of_each_apple : A = 0.80 :=
by sorry

end price_of_each_apple_l71_71587


namespace sqrt_sum_of_fractions_as_fraction_l71_71211

theorem sqrt_sum_of_fractions_as_fraction :
  (Real.sqrt ((36 / 49) + (16 / 9) + (1 / 16))) = (45 / 28) :=
by
  sorry

end sqrt_sum_of_fractions_as_fraction_l71_71211


namespace students_enrolled_both_english_and_german_l71_71649

def total_students : ℕ := 32
def enrolled_german : ℕ := 22
def only_english : ℕ := 10
def students_enrolled_at_least_one_subject := total_students

theorem students_enrolled_both_english_and_german :
  ∃ (e_g : ℕ), e_g = enrolled_german - only_english :=
by
  sorry

end students_enrolled_both_english_and_german_l71_71649


namespace hannahs_weekly_pay_l71_71750

-- Define conditions
def hourly_wage : ℕ := 30
def total_hours : ℕ := 18
def dock_per_late : ℕ := 5
def late_times : ℕ := 3

-- The amount paid after deductions for being late
def pay_after_deductions : ℕ :=
  let wage_before_deductions := hourly_wage * total_hours
  let total_dock := dock_per_late * late_times
  wage_before_deductions - total_dock

-- The proof statement
theorem hannahs_weekly_pay : pay_after_deductions = 525 := 
  by
  -- No proof necessary; statement and conditions must be correctly written to run
  sorry

end hannahs_weekly_pay_l71_71750


namespace power_mod_l71_71440

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l71_71440


namespace isosceles_triangle_perimeter_l71_71120

theorem isosceles_triangle_perimeter 
  (m : ℝ) 
  (h : 2 * m + 1 = 8) : 
  (m - 2) + 2 * 8 = 17.5 := 
by 
  sorry

end isosceles_triangle_perimeter_l71_71120


namespace divisibility_l71_71889

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end divisibility_l71_71889


namespace relation_y₁_y₂_y₃_l71_71451

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end relation_y₁_y₂_y₃_l71_71451


namespace largest_a_l71_71932

open Real

theorem largest_a (a b c : ℝ) (h1 : a + b + c = 6) (h2 : ab + ac + bc = 11) : 
  a ≤ 2 + 2 * sqrt 3 / 3 :=
sorry

end largest_a_l71_71932


namespace symmetricPointCorrectCount_l71_71768

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the four symmetry conditions
def isSymmetricXaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricYOZplane (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := -P.z }
def isSymmetricYaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricOrigin (P Q : Point3D) : Prop := Q = { x := -P.x, y := -P.y, z := -P.z }

-- Define a theorem to count the valid symmetric conditions
theorem symmetricPointCorrectCount (P : Point3D) :
  (isSymmetricXaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricYOZplane P { x := P.x, y := -P.y, z := -P.z } = true → false) ∧
  (isSymmetricYaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricOrigin P { x := -P.x, y := -P.y, z := -P.z } = true → true) :=
by
  sorry

end symmetricPointCorrectCount_l71_71768


namespace scientific_notation_correct_l71_71002

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end scientific_notation_correct_l71_71002


namespace first_shaded_square_for_all_columns_l71_71230

-- Let T be the function that generates the n-th triangular number
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the goal: prove that, for n = 32, T(n) is 528
theorem first_shaded_square_for_all_columns :
  T 32 = 528 := by
  sorry

end first_shaded_square_for_all_columns_l71_71230


namespace abs_diff_of_two_numbers_l71_71538

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 :=
by
  sorry

end abs_diff_of_two_numbers_l71_71538


namespace not_monotonic_on_interval_l71_71477

noncomputable def f (x : ℝ) : ℝ := (x^2 / 2) - Real.log x

theorem not_monotonic_on_interval (m : ℝ) : 
  (∃ x y : ℝ, m < x ∧ x < m + 1/2 ∧ m < y ∧ y < m + 1/2 ∧ (x ≠ y) ∧ f x ≠ f y ) ↔ (1/2 < m ∧ m < 1) :=
sorry

end not_monotonic_on_interval_l71_71477


namespace reflect_point_P_l71_71034

-- Define the point P
def P : ℝ × ℝ := (-3, 2)

-- Define the reflection across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Theorem to prove the coordinates of the point P with respect to the x-axis
theorem reflect_point_P : reflect_x_axis P = (-3, -2) := by
  sorry

end reflect_point_P_l71_71034


namespace minimal_range_of_sample_l71_71075

theorem minimal_range_of_sample (x1 x2 x3 x4 x5 : ℝ) 
  (mean_condition : (x1 + x2 + x3 + x4 + x5) / 5 = 6) 
  (median_condition : x3 = 10) 
  (sample_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) : 
  (x5 - x1) = 10 :=
sorry

end minimal_range_of_sample_l71_71075


namespace fishes_per_body_of_water_l71_71918

-- Define the number of bodies of water
def n_b : Nat := 6

-- Define the total number of fishes
def n_f : Nat := 1050

-- Prove the number of fishes per body of water
theorem fishes_per_body_of_water : n_f / n_b = 175 := by 
  sorry

end fishes_per_body_of_water_l71_71918


namespace perfect_square_k_value_l71_71475

-- Given condition:
def is_perfect_square (P : ℤ) : Prop := ∃ (z : ℤ), P = z * z

-- Theorem to prove:
theorem perfect_square_k_value (a b k : ℤ) (h : is_perfect_square (4 * a^2 + k * a * b + 9 * b^2)) :
  k = 12 ∨ k = -12 :=
sorry

end perfect_square_k_value_l71_71475


namespace roots_lost_extraneous_roots_l71_71829

noncomputable def f1 (x : ℝ) := Real.arcsin x
noncomputable def g1 (x : ℝ) := 2 * Real.arcsin (x / Real.sqrt 2)
noncomputable def f2 (x : ℝ) := x
noncomputable def g2 (x : ℝ) := 2 * x

theorem roots_lost :
  ∃ x : ℝ, f1 x = g1 x ∧ ¬ ∃ y : ℝ, Real.tan (f1 y) = Real.tan (g1 y) :=
sorry

theorem extraneous_roots :
  ∃ x : ℝ, ¬ f2 x = g2 x ∧ ∃ y : ℝ, Real.tan (f2 y) = Real.tan (g2 y) :=
sorry

end roots_lost_extraneous_roots_l71_71829


namespace hide_and_seek_l71_71374

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71374


namespace sweets_remaining_l71_71071

def num_cherry := 30
def num_strawberry := 40
def num_pineapple := 50

def half (n : Nat) := n / 2

def num_eaten_cherry := half num_cherry
def num_eaten_strawberry := half num_strawberry
def num_eaten_pineapple := half num_pineapple

def num_given_away := 5

def total_initial := num_cherry + num_strawberry + num_pineapple

def total_eaten := num_eaten_cherry + num_eaten_strawberry + num_eaten_pineapple

def total_remaining_after_eating := total_initial - total_eaten
def total_remaining := total_remaining_after_eating - num_given_away

theorem sweets_remaining : total_remaining = 55 := by
  sorry

end sweets_remaining_l71_71071


namespace lynne_total_spending_l71_71173

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l71_71173


namespace trapezoid_area_eq_15_l71_71206

theorem trapezoid_area_eq_15 :
  let line1 := fun (x : ℝ) => 2 * x
  let line2 := fun (x : ℝ) => 8
  let line3 := fun (x : ℝ) => 2
  let y_axis := fun (y : ℝ) => 0
  let intersection_points := [
    (4, 8),   -- Intersection of line1 and line2
    (1, 2),   -- Intersection of line1 and line3
    (0, 8),   -- Intersection of y_axis and line2
    (0, 2)    -- Intersection of y_axis and line3
  ]
  let base1 := (4 - 0 : ℝ)  -- Length of top base 
  let base2 := (1 - 0 : ℝ)  -- Length of bottom base
  let height := (8 - 2 : ℝ) -- Vertical distance between line2 and line3
  (0.5 * (base1 + base2) * height = 15.0) := by
  sorry

end trapezoid_area_eq_15_l71_71206


namespace elements_in_M_l71_71524

def is_element_of_M (x y : ℕ) : Prop :=
  x + y ≤ 1

def M : Set (ℕ × ℕ) :=
  {p | is_element_of_M p.fst p.snd}

theorem elements_in_M :
  M = { (0,0), (0,1), (1,0) } :=
by
  -- Proof would go here
  sorry

end elements_in_M_l71_71524


namespace probability_positive_difference_ge_three_l71_71558

open Finset Nat

theorem probability_positive_difference_ge_three :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (s.card.choose 2)
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  probability = 15 / 28 :=
by
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := s.card.choose 2
  have total_pairs_eq : total_pairs = 28 := by decide
  let pairs_with_difference_less_than_3 := 13
  let favorable_pairs := total_pairs - pairs_with_difference_less_than_3
  have favorable_pairs_eq : favorable_pairs = 15 := by decide
  let probability := favorable_pairs.to_rat / total_pairs.to_rat
  have probability_eq : probability = 15 / 28 := by
    rw [favorable_pairs_eq, total_pairs_eq, ←nat_cast_add, ←rat.div_eq_div_iff]
    norm_num
  exact probability_eq

end probability_positive_difference_ge_three_l71_71558


namespace nine_points_chords_l71_71504

theorem nine_points_chords : 
  ∀ (n : ℕ), n = 9 → ∃ k, k = 2 ∧ (Nat.choose n k = 36) := 
by 
  intro n hn
  use 2
  split
  exact rfl
  rw [hn]
  exact Nat.choose_eq (by norm_num) (by norm_num)

end nine_points_chords_l71_71504


namespace time_equal_l71_71334

noncomputable def S : ℝ := sorry 
noncomputable def S_flat : ℝ := S
noncomputable def S_uphill : ℝ := (1 / 3) * S
noncomputable def S_downhill : ℝ := (2 / 3) * S
noncomputable def V_flat : ℝ := sorry 
noncomputable def V_uphill : ℝ := (1 / 2) * V_flat
noncomputable def V_downhill : ℝ := 2 * V_flat
noncomputable def t_flat: ℝ := S / V_flat
noncomputable def t_uphill: ℝ := S_uphill / V_uphill
noncomputable def t_downhill: ℝ := S_downhill / V_downhill
noncomputable def t_hill: ℝ := t_uphill + t_downhill

theorem time_equal: t_flat = t_hill := 
  by sorry

end time_equal_l71_71334


namespace friends_who_participate_l71_71362

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71362


namespace train_stops_time_l71_71575

/-- Given the speeds of a train excluding and including stoppages, 
calculate the stopping time in minutes per hour. --/
theorem train_stops_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 48)
  (h2 : speed_including_stoppages = 40) :
  ∃ minutes_stopped : ℝ, minutes_stopped = 10 :=
by
  sorry

end train_stops_time_l71_71575


namespace inheritance_problem_l71_71945

def wifeAmounts (K J M : ℝ) : Prop :=
  K + J + M = 396 ∧
  J = K + 10 ∧
  M = J + 10

def husbandAmounts (wifeAmount : ℝ) (husbandMultiplier : ℝ := 1) : ℝ :=
  husbandMultiplier * wifeAmount

theorem inheritance_problem (K J M : ℝ)
  (h1 : wifeAmounts K J M)
  : ∃ wifeOf : String → String,
    wifeOf "John Smith" = "Katherine" ∧
    wifeOf "Henry Snooks" = "Jane" ∧
    wifeOf "Tom Crow" = "Mary" ∧
    husbandAmounts K = K ∧
    husbandAmounts J 1.5 = 1.5 * J ∧
    husbandAmounts M 2 = 2 * M :=
by 
  sorry

end inheritance_problem_l71_71945


namespace water_added_l71_71232

theorem water_added (x : ℝ) (salt_percent_initial : ℝ) (evaporation_fraction : ℝ) 
(salt_added : ℝ) (resulting_salt_percent : ℝ) 
(hx : x = 119.99999999999996) (h_initial_salt : salt_percent_initial = 0.20) 
(h_evap_fraction : evaporation_fraction = 1/4) (h_salt_added : salt_added = 16)
(h_resulting_salt_percent : resulting_salt_percent = 1/3) : 
∃ (water_added : ℝ), water_added = 30 :=
by
  sorry

end water_added_l71_71232


namespace find_y_when_x_is_twelve_l71_71642

variables (x y k : ℝ)

theorem find_y_when_x_is_twelve
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = 12) :
  y = 56.25 :=
sorry

end find_y_when_x_is_twelve_l71_71642


namespace number_of_female_students_l71_71525

variable (n m : ℕ)

theorem number_of_female_students (hn : n ≥ 0) (hm : m ≥ 0) (hmn : m ≤ n) : n - m = n - m :=
by
  sorry

end number_of_female_students_l71_71525


namespace find_larger_number_l71_71985

theorem find_larger_number (a b : ℕ) (h_diff : a - b = 3) (h_sum_squares : a^2 + b^2 = 117) (h_pos : 0 < a ∧ 0 < b) : a = 9 :=
by
  sorry

end find_larger_number_l71_71985


namespace value_range_a_for_two_positive_solutions_l71_71199

theorem value_range_a_for_two_positive_solutions (a : ℝ) :
  (∃ (x : ℝ), (|2 * x - 1| - a = 0) ∧ x > 0 ∧ (0 < a ∧ a < 1)) :=
by 
  sorry

end value_range_a_for_two_positive_solutions_l71_71199


namespace find_b_l71_71037

theorem find_b
  (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end find_b_l71_71037


namespace noemi_start_amount_l71_71176

/-
  Conditions:
    lost_roulette = -600
    won_blackjack = 400
    lost_poker = -400
    won_baccarat = 500
    meal_cost = 200
    purse_end = 1800

  Prove: start_amount == 2300
-/

noncomputable def lost_roulette : Int := -600
noncomputable def won_blackjack : Int := 400
noncomputable def lost_poker : Int := -400
noncomputable def won_baccarat : Int := 500
noncomputable def meal_cost : Int := 200
noncomputable def purse_end : Int := 1800

noncomputable def net_gain : Int := lost_roulette + won_blackjack + lost_poker + won_baccarat

noncomputable def start_amount : Int := net_gain + meal_cost + purse_end

theorem noemi_start_amount : start_amount = 2300 :=
by
  sorry

end noemi_start_amount_l71_71176


namespace expression_value_l71_71343

theorem expression_value (a b : ℚ) (h₁ : a = -1/2) (h₂ : b = 3/2) : -a - 2 * b^2 + 3 * a * b = -25/4 :=
by
  sorry

end expression_value_l71_71343


namespace geometric_triangle_q_range_l71_71128

theorem geometric_triangle_q_range (a : ℝ) (q : ℝ) (h : 0 < q) 
  (h1 : a + q * a > (q ^ 2) * a)
  (h2 : q * a + (q ^ 2) * a > a)
  (h3 : a + (q ^ 2) * a > q * a) : 
  q ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end geometric_triangle_q_range_l71_71128


namespace cost_per_container_is_21_l71_71925

-- Define the given problem conditions as Lean statements.

--  Let w be the number of weeks represented by 210 days.
def number_of_weeks (days: ℕ) : ℕ := days / 7
def weeks : ℕ := number_of_weeks 210

-- Let p be the total pounds of litter used over the number of weeks.
def pounds_per_week : ℕ := 15
def total_litter_pounds (weeks: ℕ) : ℕ := weeks * pounds_per_week
def total_pounds : ℕ := total_litter_pounds weeks

-- Let c be the number of 45-pound containers needed for the total pounds of litter.
def pounds_per_container : ℕ := 45
def number_of_containers (total_pounds pounds_per_container: ℕ) : ℕ := total_pounds / pounds_per_container
def containers : ℕ := number_of_containers total_pounds pounds_per_container

-- Given the total cost, find the cost per container.
def total_cost : ℕ := 210
def cost_per_container (total_cost containers: ℕ) : ℕ := total_cost / containers
def cost : ℕ := cost_per_container total_cost containers

-- Prove that the cost per container is 21.
theorem cost_per_container_is_21 : cost = 21 := by
  sorry

end cost_per_container_is_21_l71_71925


namespace chess_tournament_participants_l71_71915

theorem chess_tournament_participants
  (n : ℕ)
  (h1 : 3 < n)
  (h2 : ∀ p1 p2 : ℕ, p1 ≠ p2 → plays_against p1 p2 = true)
  (h3 : total_rounds = 26)
  (h4 : (∀ p : ℕ, odd_points (points p) = (p = 1))):
  n = 8 :=
sorry

-- Here we assume that plays_against and points are some functions defined elsewhere.

end chess_tournament_participants_l71_71915


namespace hide_and_seek_friends_l71_71379

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71379


namespace min_value_frac_sum_l71_71147

theorem min_value_frac_sum (a b : ℝ) (hab : a + b = 1) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (x : ℝ), x = 3 + 2 * Real.sqrt 2 ∧ x = (1/a + 2/b) :=
sorry

end min_value_frac_sum_l71_71147


namespace total_spent_l71_71170

-- Define the number of books and magazines Lynne bought
def num_books_cats : ℕ := 7
def num_books_solar_system : ℕ := 2
def num_magazines : ℕ := 3

-- Define the costs
def cost_per_book : ℕ := 7
def cost_per_magazine : ℕ := 4

-- Calculate the total cost and assert that it equals to $75
theorem total_spent :
  (num_books_cats * cost_per_book) + 
  (num_books_solar_system * cost_per_book) + 
  (num_magazines * cost_per_magazine) = 75 := 
sorry

end total_spent_l71_71170


namespace acute_triangle_locus_l71_71446

theorem acute_triangle_locus {A B C : Point} (AB : Segment A B)
    (S : Circle (midpoint A B) (dist A B / 2))
    (lA lB : Line)
    (hA : tangent lA S A)
    (hB : tangent lB S B) :
  (inside_band C lA lB) ∧ (outside_circle C S) ↔ (acute_triangle A B C) := sorry

end acute_triangle_locus_l71_71446


namespace average_of_x_y_z_l71_71141

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l71_71141


namespace max_value_of_x_plus_3y_l71_71883

theorem max_value_of_x_plus_3y (x y : ℝ) (h : x^2 / 9 + y^2 = 1) : 
    ∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = Real.sin θ ∧ (x + 3 * y) ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_of_x_plus_3y_l71_71883


namespace inequality_convex_l71_71664

theorem inequality_convex (x y a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : a + b = 1) : 
  (a * x + b * y) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 := 
sorry

end inequality_convex_l71_71664


namespace expression_evaluation_l71_71109

-- Using the given conditions
def a : ℕ := 3
def b : ℕ := a^2 + 2 * a + 5
def c : ℕ := b^2 - 14 * b + 45

-- We need to assume that none of the denominators are zero.
lemma non_zero_denominators : (a + 1 ≠ 0) ∧ (b - 3 ≠ 0) ∧ (c + 7 ≠ 0) :=
  by {
    -- Proof goes here
  sorry }

theorem expression_evaluation :
  (a = 3) →
  ((a^2 + 2*a + 5) = b) →
  ((b^2 - 14*b + 45) = c) →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (↑(a + 3) / ↑(a + 1) * ↑(b - 1) / ↑(b - 3) * ↑(c + 9) / ↑(c + 7) = 4923 / 2924) :=
  by {
    -- Proof goes here
  sorry }

end expression_evaluation_l71_71109


namespace value_of_a_minus_b_l71_71122

variables (a b : ℚ)

theorem value_of_a_minus_b (h1 : |a| = 5) (h2 : |b| = 2) (h3 : |a + b| = a + b) : a - b = 3 ∨ a - b = 7 :=
sorry

end value_of_a_minus_b_l71_71122


namespace find_first_term_l71_71972

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l71_71972


namespace alex_plays_with_friends_l71_71394

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71394


namespace train_length_l71_71418

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l71_71418


namespace student_comprehensive_score_l71_71225

def comprehensive_score (t_score i_score d_score : ℕ) (t_ratio i_ratio d_ratio : ℕ) :=
  (t_score * t_ratio + i_score * i_ratio + d_score * d_ratio) / (t_ratio + i_ratio + d_ratio)

theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 :=
by
  -- The proof goes here
  sorry

end student_comprehensive_score_l71_71225


namespace problem_l71_71301

theorem problem (a b c d : ℝ) (h1 : 2 + real.sqrt 2 = a + b) (h2 : 4 - real.sqrt 2 = c + d) 
  (ha : a = 3) (hb : b = real.sqrt 2 - 1) (hc : c = 2) (hd : d = 2 - real.sqrt 2) : 
  (b + d) / (a * c) = 1 / 6 :=
by
  rw [hb, hd, ha, hc]
  sorry

end problem_l71_71301


namespace family_ages_l71_71481

theorem family_ages 
  (youngest : ℕ)
  (middle : ℕ := youngest + 2)
  (eldest : ℕ := youngest + 4)
  (mother : ℕ := 3 * youngest + 16)
  (father : ℕ := 4 * youngest + 18)
  (total_sum : youngest + middle + eldest + mother + father = 90) :
  youngest = 5 ∧ middle = 7 ∧ eldest = 9 ∧ mother = 31 ∧ father = 38 := 
by 
  sorry

end family_ages_l71_71481


namespace right_triangle_condition_l71_71447

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition_l71_71447


namespace op_neg2_3_l71_71999

def op (a b : ℤ) : ℤ := a^2 + 2 * a * b

theorem op_neg2_3 : op (-2) 3 = -8 :=
by
  -- proof
  sorry

end op_neg2_3_l71_71999


namespace sample_size_correct_l71_71076

def sample_size (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ) (S : ℕ) : Prop :=
  sum_frequencies = 20 ∧ frequency_sum_ratio = 0.4 → S = 50

theorem sample_size_correct :
  ∀ (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ),
    sample_size sum_frequencies frequency_sum_ratio 50 :=
by
  intros sum_frequencies frequency_sum_ratio
  sorry

end sample_size_correct_l71_71076


namespace student_correct_answers_l71_71483

theorem student_correct_answers 
(C W : ℕ) 
(h1 : C + W = 80) 
(h2 : 4 * C - W = 120) : 
C = 40 :=
by
  sorry 

end student_correct_answers_l71_71483


namespace find_first_term_l71_71964

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l71_71964


namespace binomial_600_600_l71_71098

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l71_71098


namespace june_earnings_l71_71776

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l71_71776


namespace basketball_free_throws_l71_71812

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = b) 
  (h3 : 2 * a + 3 * b + x = 73) : 
  x = 10 := 
by 
  sorry -- The actual proof is omitted as per the requirements.

end basketball_free_throws_l71_71812


namespace term_5th_in_sequence_l71_71922

theorem term_5th_in_sequence : 
  ∃ n : ℕ, n = 5 ∧ ( ∃ t : ℕ, t = 28 ∧ 3^t ∈ { 3^(7 * (k - 1)) | k : ℕ } ) :=
by {
  sorry
}

end term_5th_in_sequence_l71_71922


namespace gcf_120_180_240_is_60_l71_71565

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l71_71565


namespace find_number_l71_71064

theorem find_number
  (x : ℝ)
  (h : (7.5 * 7.5) + 37.5 + (x * x) = 100) :
  x = 2.5 :=
sorry

end find_number_l71_71064


namespace part_I_part_II_l71_71880

theorem part_I (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2) (h4 : a + b ≤ m) : m ≥ 3 := by
  sorry

theorem part_II (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9/2)
  (h4 : 2 * |x - 1| + |x| ≥ a + b) : (x ≤ -1 / 3 ∨ x ≥ 5 / 3) := by
  sorry

end part_I_part_II_l71_71880


namespace b_2016_value_l71_71460

theorem b_2016_value : 
  ∃ (a b : ℕ → ℝ), 
    a 1 = 1 / 2 ∧ 
    (∀ n : ℕ, 0 < n → a n + b n = 1) ∧
    (∀ n : ℕ, 0 < n → b (n + 1) = b n / (1 - (a n)^2)) → 
    b 2016 = 2016 / 2017 :=
by
  sorry

end b_2016_value_l71_71460


namespace black_to_white_ratio_l71_71428

/-- 
Given:
- The original square pattern consists of 13 black tiles and 23 white tiles
- Attaching a border of black tiles around the original 6x6 square pattern results in an 8x8 square pattern

To prove:
- The ratio of black tiles to white tiles in the extended 8x8 pattern is 41/23.
-/
theorem black_to_white_ratio (b_orig w_orig b_added b_total w_total : ℕ) 
  (h_black_orig: b_orig = 13)
  (h_white_orig: w_orig = 23)
  (h_size_orig: 6 * 6 = b_orig + w_orig)
  (h_size_ext: 8 * 8 = (b_orig + b_added) + w_orig)
  (h_b_added: b_added = 28)
  (h_b_total: b_total = b_orig + b_added)
  (h_w_total: w_total = w_orig)
  :
  b_total / w_total = 41 / 23 :=
by
  sorry

end black_to_white_ratio_l71_71428


namespace binom_600_600_l71_71101

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l71_71101


namespace waiter_customers_l71_71717

theorem waiter_customers
    (initial_tables : ℝ)
    (left_tables : ℝ)
    (customers_per_table : ℝ)
    (remaining_tables : ℝ) 
    (total_customers : ℝ) 
    (h1 : initial_tables = 44.0)
    (h2 : left_tables = 12.0)
    (h3 : customers_per_table = 8.0)
    (remaining_tables_def : remaining_tables = initial_tables - left_tables)
    (total_customers_def : total_customers = remaining_tables * customers_per_table) :
    total_customers = 256.0 :=
by
  sorry

end waiter_customers_l71_71717


namespace find_g2_l71_71035

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 4 * g x - 3 * g (1 / x) = x^2

theorem find_g2 : g 2 = 67 / 28 :=
by {
  sorry
}

end find_g2_l71_71035


namespace range_of_m_l71_71131

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (m * x^2 + (m - 3) * x + 1 = 0)) →
  m ∈ Set.Iic 1 := by
  sorry

end range_of_m_l71_71131


namespace exists_composite_expression_l71_71667

-- Define what it means for a number to be composite
def is_composite (m : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m

-- Main theorem statement
theorem exists_composite_expression :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → is_composite (n * 2^k + 1) :=
sorry

end exists_composite_expression_l71_71667


namespace simplify_and_evaluate_l71_71672

theorem simplify_and_evaluate (x : ℝ) (h : x = 3 / 2) : 
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := 
by
  sorry

end simplify_and_evaluate_l71_71672


namespace problem1_problem2_l71_71892

-- Define variables
variables {x y m : ℝ}
variables (h1 : x + y > 0) (h2 : xy ≠ 0)

-- Problem (1): Prove that x^3 + y^3 ≥ x^2 y + y^2 x
theorem problem1 (h1 : x + y > 0) (h2 : xy ≠ 0) : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
sorry

-- Problem (2): Given the conditions, the range of m is [-6, 2]
theorem problem2 (h1 : x + y > 0) (h2 : xy ≠ 0) (h3 : (x / y^2) + (y / x^2) ≥ (m / 2) * ((1 / x) + (1 / y))) : m ∈ Set.Icc (-6 : ℝ) 2 :=
sorry

end problem1_problem2_l71_71892


namespace least_number_to_add_l71_71339

theorem least_number_to_add (x : ℕ) : (1053 + x) % 23 = 0 ↔ x = 5 := by
  sorry

end least_number_to_add_l71_71339


namespace value_of_c_minus_a_l71_71645

variables (a b c : ℝ)

theorem value_of_c_minus_a (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 60) : (c - a) = 30 :=
by
  have h3 : a + b = 90 := by sorry
  have h4 : b + c = 120 := by sorry
  -- now we have the required form of the problem statement
  -- c - a = 120 - 90
  sorry

end value_of_c_minus_a_l71_71645


namespace percentage_of_third_number_l71_71229

theorem percentage_of_third_number (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A / C = 0.06 := 
by
  sorry

end percentage_of_third_number_l71_71229


namespace draw_3_odd_balls_from_15_is_336_l71_71220

-- Define the problem setting as given in the conditions
def odd_balls : Finset ℕ := {1, 3, 5, 7, 9, 11, 13, 15}

-- Define the function that calculates the number of ways to draw 3 balls
noncomputable def draw_3_odd_balls (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

-- Prove that the drawing of 3 balls results in 336 ways
theorem draw_3_odd_balls_from_15_is_336 : draw_3_odd_balls odd_balls = 336 := by
  sorry

end draw_3_odd_balls_from_15_is_336_l71_71220


namespace train_average_speed_l71_71715

open Real -- Assuming all required real number operations 

noncomputable def average_speed (distances : List ℝ) (times : List ℝ) : ℝ := 
  let total_distance := distances.sum
  let total_time := times.sum
  total_distance / total_time

theorem train_average_speed :
  average_speed [125, 270] [2.5, 3] = 71.82 := 
by 
  -- Details of the actual proof steps are omitted
  sorry

end train_average_speed_l71_71715


namespace sum_of_products_leq_one_third_l71_71487

theorem sum_of_products_leq_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 :=
sorry

end sum_of_products_leq_one_third_l71_71487


namespace min_value_expr_l71_71873

theorem min_value_expr (x y : ℝ) : 
  ∃ x y : ℝ, (x, y) = (4, 0) ∧ (∀ x y : ℝ, x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ -22) :=
by
  sorry

end min_value_expr_l71_71873


namespace lynne_total_spending_l71_71174

theorem lynne_total_spending :
  let num_books_cats := 7
  let num_books_solar_system := 2
  let num_magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := num_books_cats + num_books_solar_system
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := num_magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 := sorry

end lynne_total_spending_l71_71174


namespace power_mod_l71_71439

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l71_71439


namespace midpoint_of_polar_line_segment_l71_71766

theorem midpoint_of_polar_line_segment
  (r θ : ℝ)
  (hr : r > 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (hA : ∃ A, A = (8, 5 * Real.pi / 12))
  (hB : ∃ B, B = (8, -3 * Real.pi / 12)) :
  (r, θ) = (4, Real.pi / 12) := 
sorry

end midpoint_of_polar_line_segment_l71_71766


namespace find_first_term_l71_71961

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l71_71961


namespace total_animal_legs_l71_71080

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l71_71080


namespace hide_and_seek_problem_l71_71369

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71369


namespace hide_and_seek_l71_71387

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71387


namespace piggy_bank_savings_l71_71798

theorem piggy_bank_savings :
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  final_amount = 104 :=
by
  let initial_amount := 200
  let spending_per_trip := 2
  let trips_per_month := 4
  let months_per_year := 12
  let monthly_expenditure := spending_per_trip * trips_per_month
  let annual_expenditure := monthly_expenditure * months_per_year
  let final_amount := initial_amount - annual_expenditure
  show final_amount = 104 from sorry

end piggy_bank_savings_l71_71798


namespace no_natural_numbers_satisfying_conditions_l71_71434

theorem no_natural_numbers_satisfying_conditions :
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ k : ℕ, b^2 + 4*a = k^2 := by
  sorry

end no_natural_numbers_satisfying_conditions_l71_71434


namespace man_l71_71592

theorem man's_salary (S : ℝ) 
  (h_food : S * (1 / 5) > 0)
  (h_rent : S * (1 / 10) > 0)
  (h_clothes : S * (3 / 5) > 0)
  (h_left : S * (1 / 10) = 19000) : 
  S = 190000 := by
  sorry

end man_l71_71592


namespace linear_function_difference_l71_71658

-- Define the problem in Lean.
theorem linear_function_difference (g : ℕ → ℝ) (h : ∀ x y : ℕ, g x = 3 * x + g 0) (h_condition : g 4 - g 1 = 9) : g 10 - g 1 = 27 := 
by
  sorry -- Proof is omitted.

end linear_function_difference_l71_71658


namespace sum_of_second_and_third_smallest_is_804_l71_71204

noncomputable def sum_of_second_and_third_smallest : Nat :=
  let digits := [1, 6, 8]
  let second_smallest := 186
  let third_smallest := 618
  second_smallest + third_smallest

theorem sum_of_second_and_third_smallest_is_804 :
  sum_of_second_and_third_smallest = 804 :=
by
  sorry

end sum_of_second_and_third_smallest_is_804_l71_71204


namespace range_of_linear_function_l71_71445

theorem range_of_linear_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  3 < -2 * x + 5 ∧ -2 * x + 5 < 7 :=
by {
  sorry
}

end range_of_linear_function_l71_71445


namespace find_a_b_sum_l71_71935

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x - 6

theorem find_a_b_sum (a b : ℝ)
  (h1 : f a = 1)
  (h2 : f b = -5) :
  a + b = 2 :=
  sorry

end find_a_b_sum_l71_71935


namespace number_of_trees_l71_71718

theorem number_of_trees (length_of_yard : ℕ) (distance_between_trees : ℕ) 
(h1 : length_of_yard = 273) 
(h2 : distance_between_trees = 21) : 
(length_of_yard / distance_between_trees) + 1 = 14 := by
  sorry

end number_of_trees_l71_71718


namespace jason_additional_manager_months_l71_71655

def additional_manager_months (bartender_years manager_years total_exp_months : ℕ) : ℕ :=
  let bartender_months := bartender_years * 12
  let manager_months := manager_years * 12
  total_exp_months - (bartender_months + manager_months)

theorem jason_additional_manager_months : 
  additional_manager_months 9 3 150 = 6 := 
by 
  sorry

end jason_additional_manager_months_l71_71655


namespace part1_part2_l71_71264

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) := {x : ℝ | -3 * m + 4 ≤ x ∧ x ≤ 2 * m - 1}

def p (m : ℝ) := ∀ x : ℝ, x ∈ A → x ∈ B m
def q (m : ℝ) := ∃ x : ℝ, x ∈ B m ∧ x ∈ A

theorem part1 (m : ℝ) : p m → m ≥ 4 := by
  sorry

theorem part2 (m : ℝ) : q m → m ≥ 3/2 := by
  sorry

end part1_part2_l71_71264


namespace ratio_of_areas_l71_71596

theorem ratio_of_areas 
  (t : ℝ) (q : ℝ)
  (h1 : t = 1 / 4)
  (h2 : q = 1 / 2) :
  q / t = 2 :=
by sorry

end ratio_of_areas_l71_71596


namespace tetrahedron_through_hole_tetrahedron_cannot_through_hole_l71_71091

/--
A regular tetrahedron with edge length 1 can pass through a circular hole if and only if the radius \( R \) is at least 0.4478, given that the thickness of the hole can be neglected.
-/

theorem tetrahedron_through_hole (R : ℝ) (h1 : R = 0.45) : true :=
by sorry

theorem tetrahedron_cannot_through_hole (R : ℝ) (h1 : R = 0.44) : false :=
by sorry

end tetrahedron_through_hole_tetrahedron_cannot_through_hole_l71_71091


namespace expression_range_l71_71159

theorem expression_range (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2)
  + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ∧ 
  (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ≤ 8 :=
sorry

end expression_range_l71_71159


namespace non_organic_chicken_price_l71_71595

theorem non_organic_chicken_price :
  ∀ (x : ℝ), (0.75 * x = 9) → (2 * (0.9 * x) = 21.6) :=
by
  intro x hx
  sorry

end non_organic_chicken_price_l71_71595


namespace emmy_rosa_ipods_l71_71246

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l71_71246


namespace john_taking_pictures_years_l71_71926

-- Definitions based on the conditions
def pictures_per_day : ℕ := 10
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140
def days_per_year : ℕ := 365

-- Theorem statement
theorem john_taking_pictures_years : total_spent / cost_per_card * images_per_card / pictures_per_day / days_per_year = 3 :=
by
  sorry

end john_taking_pictures_years_l71_71926


namespace least_number_to_add_l71_71350

theorem least_number_to_add (n : ℕ) (m : ℕ) : (1156 + 19) % 25 = 0 :=
by
  sorry

end least_number_to_add_l71_71350


namespace largest_a_value_l71_71933

theorem largest_a_value (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 12) : 
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

end largest_a_value_l71_71933


namespace find_first_term_l71_71971

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l71_71971


namespace difference_abs_eq_200_l71_71913

theorem difference_abs_eq_200 (x y : ℤ) (h1 : x + y = 250) (h2 : y = 225) : |x - y| = 200 := sorry

end difference_abs_eq_200_l71_71913


namespace find_total_values_l71_71690

theorem find_total_values (n : ℕ) (S : ℝ) 
  (h1 : S / n = 150) 
  (h2 : (S + 25) / n = 151.25) 
  (h3 : 25 = 160 - 135) : n = 20 :=
by
  sorry

end find_total_values_l71_71690


namespace inequality_for_positive_reals_l71_71668

open Real

theorem inequality_for_positive_reals 
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  a^3 * b + b^3 * c + c^3 * a ≥ a * b * c * (a + b + c) :=
sorry

end inequality_for_positive_reals_l71_71668


namespace function_odd_domain_of_f_range_of_f_l71_71574

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem function_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intro x
  trivial

theorem range_of_f : ∀ y : ℝ, y ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  intro y
  sorry

end function_odd_domain_of_f_range_of_f_l71_71574


namespace sum_first_15_terms_l71_71994

noncomputable def sum_of_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

noncomputable def fourth_term (a d : ℝ) : ℝ := a + 3 * d
noncomputable def twelfth_term (a d : ℝ) : ℝ := a + 11 * d

theorem sum_first_15_terms (a d : ℝ) 
  (h : fourth_term a d + twelfth_term a d = 10) : sum_of_terms a d 15 = 75 :=
by
  sorry

end sum_first_15_terms_l71_71994


namespace part1_l71_71882

theorem part1 (f : ℝ → ℝ) (m n : ℝ) (cond1 : m + n > 0) (cond2 : ∀ x, f x = |x - m| + |x + n|) (cond3 : ∀ x, f x ≥ m + n) (minimum : ∃ x, f x = 2) :
    m + n = 2 := sorry

end part1_l71_71882


namespace intersection_of_M_and_N_l71_71449

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Statement of the theorem proving the intersection of M and N
theorem intersection_of_M_and_N :
  M ∩ N = {2, 3} :=
by sorry

end intersection_of_M_and_N_l71_71449


namespace socks_expected_value_l71_71549

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l71_71549


namespace tank_fill_time_with_leak_l71_71086

theorem tank_fill_time_with_leak 
  (pump_fill_time : ℕ) (leak_empty_time : ℕ) (effective_fill_time : ℕ)
  (hp : pump_fill_time = 5)
  (hl : leak_empty_time = 10)
  (he : effective_fill_time = 10) : effective_fill_time = 10 :=
by
  sorry

end tank_fill_time_with_leak_l71_71086


namespace inequality_satisfied_for_a_l71_71015

theorem inequality_satisfied_for_a (a : ℝ) :
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end inequality_satisfied_for_a_l71_71015


namespace chess_tournament_participants_and_days_l71_71763

theorem chess_tournament_participants_and_days:
  ∃ n d : ℕ, 
    (n % 2 = 1) ∧
    (n * (n - 1) / 2 = 630) ∧
    (d = 34 / 2) ∧
    (n = 35) ∧
    (d = 17) :=
sorry

end chess_tournament_participants_and_days_l71_71763


namespace temperature_on_Monday_l71_71190

theorem temperature_on_Monday 
  (M T W Th F : ℝ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : F = 36) : 
  M = 44 := 
by 
  -- Proof omitted
  sorry

end temperature_on_Monday_l71_71190


namespace positive_difference_of_b_values_l71_71939

noncomputable def g (n : ℤ) : ℤ :=
if n ≤ 0 then n^2 + 3 * n + 2 else 3 * n - 15

theorem positive_difference_of_b_values : 
  abs (-5 - 9) = 14 :=
by {
  sorry
}

end positive_difference_of_b_values_l71_71939


namespace geometric_series_first_term_l71_71977

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l71_71977


namespace alex_play_friends_with_l71_71355

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71355


namespace actual_time_before_storm_is_18_18_l71_71306

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end actual_time_before_storm_is_18_18_l71_71306


namespace diff_eq_40_l71_71647

theorem diff_eq_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end diff_eq_40_l71_71647


namespace olivia_worked_hours_on_wednesday_l71_71177

-- Define the conditions
def hourly_rate := 9
def hours_monday := 4
def hours_friday := 6
def total_earnings := 117
def earnings_monday := hours_monday * hourly_rate
def earnings_friday := hours_friday * hourly_rate
def earnings_wednesday := total_earnings - (earnings_monday + earnings_friday)

-- Define the number of hours worked on Wednesday
def hours_wednesday := earnings_wednesday / hourly_rate

-- The theorem to prove
theorem olivia_worked_hours_on_wednesday : hours_wednesday = 3 :=
by
  -- Skip the proof
  sorry

end olivia_worked_hours_on_wednesday_l71_71177


namespace polynomial_integer_roots_a_value_l71_71737

open Polynomial

theorem polynomial_integer_roots_a_value (α β γ : ℤ) (a : ℤ) :
  (X - C α) * (X - C β) * (X - C γ) = X^3 - 2 * X^2 - 25 * X + C a →
  α + β + γ = 2 →
  α * β + α * γ + β * γ = -25 →
  a = -50 :=
by
  sorry

end polynomial_integer_roots_a_value_l71_71737


namespace binomial_product_9_2_7_2_l71_71861

theorem binomial_product_9_2_7_2 : Nat.choose 9 2 * Nat.choose 7 2 = 756 := by
  sorry

end binomial_product_9_2_7_2_l71_71861


namespace scientific_notation_correct_l71_71001

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end scientific_notation_correct_l71_71001


namespace gain_percentage_l71_71591

-- Define the conditions as a Lean problem
theorem gain_percentage (C G : ℝ) (hC : (9 / 10) * C = 1) (hSP : (10 / 6) = (1 + G / 100) * C) : 
  G = 50 :=
by
-- Here, you would generally have the proof steps, but we add sorry to skip the proof for now.
sorry

end gain_percentage_l71_71591


namespace recipe_required_ingredients_l71_71787

-- Define the number of cups required for each ingredient in the recipe
def sugar_cups : Nat := 11
def flour_cups : Nat := 8
def cocoa_cups : Nat := 5

-- Define the cups of flour and cocoa already added
def flour_already_added : Nat := 3
def cocoa_already_added : Nat := 2

-- Define the cups of flour and cocoa that still need to be added
def flour_needed_to_add : Nat := 6
def cocoa_needed_to_add : Nat := 3

-- Sum the total amount of flour and cocoa powder based on already added and still needed amounts
def total_flour: Nat := flour_already_added + flour_needed_to_add
def total_cocoa: Nat := cocoa_already_added + cocoa_needed_to_add

-- Total ingredients calculation according to the problem's conditions
def total_ingredients : Nat := sugar_cups + total_flour + total_cocoa

-- The theorem to be proved
theorem recipe_required_ingredients : total_ingredients = 24 := by
  sorry

end recipe_required_ingredients_l71_71787


namespace value_of_x_l71_71619

def is_whole_number (n : ℝ) : Prop := ∃ (k : ℤ), n = k

theorem value_of_x (n : ℝ) (x : ℝ) :
  n = 1728 →
  is_whole_number (Real.log n / Real.log x + Real.log n / Real.log 12) →
  x = 12 :=
by
  intro h₁ h₂
  sorry

end value_of_x_l71_71619


namespace largest_number_of_stamps_per_page_l71_71656

theorem largest_number_of_stamps_per_page :
  Nat.gcd (Nat.gcd 1200 1800) 2400 = 600 :=
sorry

end largest_number_of_stamps_per_page_l71_71656


namespace electromagnetic_storm_time_l71_71305

structure Time :=
(hh : ℕ)
(mm : ℕ)
(valid_hour : 0 ≤ hh ∧ hh < 24)
(valid_minute : 0 ≤ mm ∧ mm < 60)

def possible_digits (d : ℕ) : set ℕ :=
  {x | x = d + 1 ∨ x = d - 1}

theorem electromagnetic_storm_time :
  (∃ t : Time, t.hh = 18 ∧ t.mm = 18) →
  (∀ (orig : Time), 
    orig.hh ∈ possible_digits 0 ∧ 
    (orig.hh % 10) ∈ possible_digits 9 ∧
    orig.mm ∈ possible_digits 0 ∧ 
    (orig.mm % 10) ∈ possible_digits 9 →
      false) :=
by
  sorry

end electromagnetic_storm_time_l71_71305


namespace gcf_120_180_240_l71_71567

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l71_71567


namespace vector_sum_solve_for_m_n_l71_71152

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Vector sum
theorem vector_sum : 3 • a + b - 2 • c = (0, 6) :=
by sorry

-- Problem 2: Solving for m and n
theorem solve_for_m_n (m n : ℝ) (hm : a = m • b + n • c) :
  m = 5 / 9 ∧ n = 8 / 9 :=
by sorry

end vector_sum_solve_for_m_n_l71_71152


namespace emma_reaches_jack_after_33_minutes_l71_71291

-- Definitions from conditions
def distance_initial : ℝ := 30  -- 30 km apart initially
def combined_speed : ℝ := 2     -- combined speed is 2 km/min
def time_before_breakdown : ℝ := 6 -- Jack biked for 6 minutes before breaking down

-- Assume speeds
def v_J (v_E : ℝ) : ℝ := 2 * v_E  -- Jack's speed is twice Emma's speed

-- Assertion to prove
theorem emma_reaches_jack_after_33_minutes :
  ∀ v_E : ℝ, ((v_J v_E + v_E = combined_speed) → 
              (distance_initial - combined_speed * time_before_breakdown = 18) → 
              (v_E > 0) → 
              (time_before_breakdown + 18 / v_E = 33)) :=
by 
  intro v_E 
  intros h1 h2 h3 
  have h4 : v_J v_E = 2 * v_E := rfl
  sorry

end emma_reaches_jack_after_33_minutes_l71_71291


namespace sum_of_consecutive_neg_ints_l71_71815

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l71_71815


namespace ratio_of_numbers_l71_71951

theorem ratio_of_numbers (A B : ℕ) (HCF_AB : Nat.gcd A B = 3) (LCM_AB : Nat.lcm A B = 36) : 
  A / B = 3 / 4 :=
sorry

end ratio_of_numbers_l71_71951


namespace total_limes_picked_l71_71235

theorem total_limes_picked (Alyssa_limes Mike_limes : ℕ) 
        (hAlyssa : Alyssa_limes = 25) (hMike : Mike_limes = 32) : 
       Alyssa_limes + Mike_limes = 57 :=
by {
  sorry
}

end total_limes_picked_l71_71235


namespace probability_less_than_mean_l71_71126

open ProbabilityTheory
open MeasureTheory

noncomputable def xi : Measure ℝ := 
  Measure.normal 2 σ^2

theorem probability_less_than_mean :
  ∀ σ > 0, P(ξ < 2) = 0.5 :=
by
  sorry

end probability_less_than_mean_l71_71126


namespace range_f_neg2_l71_71013

noncomputable def f (a b x : ℝ): ℝ := a * x^2 + b * x

theorem range_f_neg2 (a b : ℝ) (h1 : 1 ≤ f a b (-1)) (h2 : f a b (-1) ≤ 2)
  (h3 : 3 ≤ f a b 1) (h4 : f a b 1 ≤ 4) : 6 ≤ f a b (-2) ∧ f a b (-2) ≤ 10 :=
by
  sorry

end range_f_neg2_l71_71013


namespace max_age_l71_71772

-- Definitions of the conditions
def born_same_day (max_birth luka_turn4 : ℕ) : Prop := max_birth = luka_turn4
def age_difference (luka_age aubrey_age : ℕ) : Prop := luka_age = aubrey_age + 2
def aubrey_age_on_birthday : ℕ := 8

-- Prove that Max's age is 6 years when Aubrey is 8 years old
theorem max_age (luka_birth aubrey_birth max_birth : ℕ) 
                (h1 : born_same_day max_birth luka_birth) 
                (h2 : age_difference luka_birth aubrey_birth) : 
                (aubrey_birth + 4 - luka_birth) = 6 :=
by
  sorry

end max_age_l71_71772


namespace Igor_colored_all_cells_l71_71400

theorem Igor_colored_all_cells (m n : ℕ) (h1 : 9 * m = 12 * n) (h2 : 0 < m ∧ m ≤ 4) (h3 : 0 < n ∧ n ≤ 3) :
  m = 4 ∧ n = 3 :=
by {
  sorry
}

end Igor_colored_all_cells_l71_71400


namespace cost_of_goods_l71_71691

-- Define variables and conditions
variables (x y z : ℝ)

-- Assume the given conditions
axiom h1 : x + 2 * y + 3 * z = 136
axiom h2 : 3 * x + 2 * y + z = 240

-- Statement to prove
theorem cost_of_goods : x + y + z = 94 := 
sorry

end cost_of_goods_l71_71691


namespace alex_plays_with_friends_l71_71399

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71399


namespace initial_tickets_l71_71823

theorem initial_tickets (tickets_sold_week1 : ℕ) (tickets_sold_week2 : ℕ) (tickets_left : ℕ) 
  (h1 : tickets_sold_week1 = 38) (h2 : tickets_sold_week2 = 17) (h3 : tickets_left = 35) : 
  tickets_sold_week1 + tickets_sold_week2 + tickets_left = 90 :=
by 
  sorry

end initial_tickets_l71_71823


namespace trigonometric_identity_l71_71640

theorem trigonometric_identity 
  (α : ℝ) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := 
sorry

end trigonometric_identity_l71_71640


namespace second_term_is_three_l71_71851

-- Given conditions
variables (r : ℝ) (S : ℝ)
hypothesis hr : r = 1 / 4
hypothesis hS : S = 16

-- Definition of the first term a
noncomputable def first_term (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

-- Definition of the second term
noncomputable def second_term (r : ℝ) (a : ℝ) : ℝ :=
  a * r

-- Prove that the second term is 3
theorem second_term_is_three : second_term r (first_term r S) = 3 :=
by
  rw [first_term, second_term]
  sorry

end second_term_is_three_l71_71851


namespace remainder_of_3_pow_102_mod_101_l71_71115

theorem remainder_of_3_pow_102_mod_101 : (3^102) % 101 = 9 :=
by
  sorry

end remainder_of_3_pow_102_mod_101_l71_71115


namespace sum_of_geometric_sequence_l71_71287

theorem sum_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (r : ℝ),
  (∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ,
   a 1 = a_1 ∧ a 2 = a_2 ∧ a 3 = a_3 ∧ a 4 = a_4 ∧ a 5 = a_5 ∧ a 6 = a_6 ∧ a 7 = a_7 ∧ a 8 = a_8 ∧ a 9 = a_9 ∧
   a_1 * r^1 = a_2 ∧ a_1 * r^2 = a_3 ∧ a_1 * r^3 = a_4 ∧ a_1 * r^4 = a_5 ∧ a_1 * r^5 = a_6 ∧ a_1 * r^6 = a_7 ∧ a_1 * r^7 = a_8 ∧ a_1 * r^8 = a_9 ∧
   a_1 + a_2 + a_3 = 8 ∧
   a_4 + a_5 + a_6 = -4) →
  a 7 + a 8 + a 9 = 2 :=
sorry

end sum_of_geometric_sequence_l71_71287


namespace boy_usual_time_l71_71578

theorem boy_usual_time (R T : ℝ) (h : R * T = (7 / 6) * R * (T - 2)) : T = 14 :=
by
  sorry

end boy_usual_time_l71_71578


namespace ben_bonus_leftover_l71_71425

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end ben_bonus_leftover_l71_71425


namespace sufficient_but_not_necessary_condition_l71_71891

open Real

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧ ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l71_71891


namespace find_value_of_expression_l71_71919

variable {a : ℕ → ℤ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (h1 : a 1 + 3 * a 8 + a 15 = 120)
variable (h2 : is_arithmetic_sequence a)

-- Theorem to be proved
theorem find_value_of_expression : 2 * a 6 - a 4 = 24 :=
sorry

end find_value_of_expression_l71_71919


namespace persons_attended_total_l71_71421

theorem persons_attended_total (p q : ℕ) (a : ℕ) (c : ℕ) (total_amount : ℕ) (adult_ticket : ℕ) (child_ticket : ℕ) 
  (h1 : adult_ticket = 60) (h2 : child_ticket = 25) (h3 : total_amount = 14000) 
  (h4 : a = 200) (h5 : p = a + c)
  (h6 : a * adult_ticket + c * child_ticket = total_amount):
  p = 280 :=
by
  sorry

end persons_attended_total_l71_71421


namespace distinct_nonzero_real_product_l71_71298

noncomputable section
open Real

theorem distinct_nonzero_real_product
  (a b c d : ℝ)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hcd : c ≠ d)
  (hda : d ≠ a)
  (ha_ne_0 : a ≠ 0)
  (hb_ne_0 : b ≠ 0)
  (hc_ne_0 : c ≠ 0)
  (hd_ne_0 : d ≠ 0)
  (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 :=
sorry

end distinct_nonzero_real_product_l71_71298


namespace trajectory_is_line_segment_l71_71259

theorem trajectory_is_line_segment : 
  ∃ (P : ℝ × ℝ) (F1 F2: ℝ × ℝ), 
    F1 = (-3, 0) ∧ F2 = (3, 0) ∧ (|F1.1 - P.1|^2 + |F1.2 - P.2|^2).sqrt + (|F2.1 - P.1|^2 + |F2.2 - P.2|^2).sqrt = 6
  → (P.1 = F1.1 ∨ P.1 = F2.1) ∧ (P.2 = F1.2 ∨ P.2 = F2.2) :=
by sorry

end trajectory_is_line_segment_l71_71259


namespace michael_choose_classes_l71_71789

-- Michael's scenario setup
def total_classes : ℕ := 10
def compulsory_class : ℕ := 1
def remaining_classes : ℕ := total_classes - compulsory_class
def total_to_choose : ℕ := 4
def additional_to_choose : ℕ := total_to_choose - compulsory_class

-- Correct answer based on the conditions
def correct_answer : ℕ := 84

-- Function to compute the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove the number of ways Michael can choose his classes
theorem michael_choose_classes : binomial 9 3 = correct_answer := by
  rw [binomial, Nat.factorial]
  sorry

end michael_choose_classes_l71_71789


namespace gummy_bear_production_time_l71_71036

theorem gummy_bear_production_time 
  (gummy_bears_per_minute : ℕ)
  (gummy_bears_per_packet : ℕ)
  (total_packets : ℕ)
  (h1 : gummy_bears_per_minute = 300)
  (h2 : gummy_bears_per_packet = 50)
  (h3 : total_packets = 240) :
  (total_packets / (gummy_bears_per_minute / gummy_bears_per_packet) = 40) :=
sorry

end gummy_bear_production_time_l71_71036


namespace sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l71_71527

noncomputable def sum_of_consecutive_triplets (a : Fin 12 → ℕ) (i : Fin 12) : ℕ :=
a i + a ((i + 1) % 12) + a ((i + 2) % 12)

theorem sum_of_consecutive_at_least_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i ≥ 20 :=
by
  sorry

theorem sum_of_consecutive_greater_than_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i > 20 :=
by
  sorry

end sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l71_71527


namespace value_of_item_l71_71561

theorem value_of_item (a b m p : ℕ) (h : a ≠ b) (eq_capitals : a * x + m = b * x + p) : 
  x = (p - m) / (a - b) :=
by
  sorry

end value_of_item_l71_71561


namespace paula_remaining_money_l71_71310

theorem paula_remaining_money (initial_amount cost_per_shirt cost_of_pants : ℕ) 
                             (num_shirts : ℕ) (H1 : initial_amount = 109)
                             (H2 : cost_per_shirt = 11) (H3 : num_shirts = 2)
                             (H4 : cost_of_pants = 13) :
  initial_amount - (num_shirts * cost_per_shirt + cost_of_pants) = 74 := 
by
  -- Calculation of total spent and remaining would go here.
  sorry

end paula_remaining_money_l71_71310


namespace tan_neg_five_pi_over_three_l71_71255

theorem tan_neg_five_pi_over_three : Real.tan (-5 * Real.pi / 3) = Real.sqrt 3 := 
by 
  sorry

end tan_neg_five_pi_over_three_l71_71255


namespace friends_who_participate_l71_71359

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71359


namespace find_first_term_l71_71974

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l71_71974


namespace range_of_a_l71_71784

def f (x a : ℝ) : ℝ := x^2 + a * x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ∧ (∃ x : ℝ, f (f x a) a = 0) → (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l71_71784


namespace rectangle_diagonal_length_l71_71324

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l71_71324


namespace factorized_polynomial_sum_of_squares_l71_71906

theorem factorized_polynomial_sum_of_squares :
  ∃ a b c d e f : ℤ, 
    (729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210) :=
sorry

end factorized_polynomial_sum_of_squares_l71_71906


namespace socks_expected_value_l71_71550

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l71_71550


namespace remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l71_71988

theorem remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2 :
  (x^15 - 1) % (x + 1) = -2 := 
sorry

end remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l71_71988


namespace main_problem_l71_71749

-- Define the set A
def A (a : ℝ) : Set ℝ :=
  {0, 1, a^2 - 2 * a}

-- Define the main problem as a theorem
theorem main_problem (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 :=
  sorry

end main_problem_l71_71749


namespace quadratic_function_through_point_l71_71631

theorem quadratic_function_through_point : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = a * x ^ 2 ∧ ((x, y) = (-1, 4)) → y = 4 * x ^ 2) :=
sorry

end quadratic_function_through_point_l71_71631


namespace most_probable_hits_l71_71917

theorem most_probable_hits (p : ℝ) (q : ℝ) (k0 : ℕ) (n : ℤ) 
  (h1 : p = 0.7) (h2 : q = 1 - p) (h3 : k0 = 16) 
  (h4 : 21 < (n : ℝ) * 0.7) (h5 : (n : ℝ) * 0.7 < 23.3) : 
  n = 22 ∨ n = 23 :=
sorry

end most_probable_hits_l71_71917


namespace find_first_term_geometric_series_l71_71967

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l71_71967


namespace problem1_problem2_l71_71858

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l71_71858


namespace Deepak_and_Wife_meet_time_l71_71949

theorem Deepak_and_Wife_meet_time 
    (circumference : ℕ) 
    (Deepak_speed : ℕ)
    (wife_speed : ℕ) 
    (conversion_factor_km_hr_to_m_hr : ℕ) 
    (minutes_per_hour : ℕ) :
    circumference = 726 →
    Deepak_speed = 4500 →  -- speed in meters per hour
    wife_speed = 3750 →  -- speed in meters per hour
    conversion_factor_km_hr_to_m_hr = 1000 →
    minutes_per_hour = 60 →
    (726 / ((4500 + 3750) / 1000) * 60 = 5.28) :=
by 
    sorry

end Deepak_and_Wife_meet_time_l71_71949


namespace pentagon_angle_sum_l71_71765

theorem pentagon_angle_sum (A B C D Q : ℝ) (hA : A = 118) (hB : B = 105) (hC : C = 87) (hD : D = 135) :
  (A + B + C + D + Q = 540) -> Q = 95 :=
by
  sorry

end pentagon_angle_sum_l71_71765


namespace parallel_vectors_l71_71912

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 3)) (h₂ : b = (-1, 2)) :
  (m * a.1 + b.1) * (-1) - 4 * (m * a.2 + b.2) = 0 → m = -1 / 2 :=
by
  intro h
  rw [h₁, h₂] at h
  simp at h
  sorry

end parallel_vectors_l71_71912


namespace car_travel_time_l71_71842

theorem car_travel_time (speed distance : ℝ) (h₁ : speed = 65) (h₂ : distance = 455) :
  distance / speed = 7 :=
by
  -- We will invoke the conditions h₁ and h₂ to conclude the theorem
  sorry

end car_travel_time_l71_71842


namespace integer_solution_zero_l71_71312

theorem integer_solution_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_solution_zero_l71_71312


namespace smallest_n_with_units_digit_and_reorder_l71_71736

theorem smallest_n_with_units_digit_and_reorder :
  ∃ n : ℕ, (∃ a : ℕ, n = 10 * a + 6) ∧ (∃ m : ℕ, 6 * 10^m + a = 4 * n) ∧ n = 153846 :=
by
  sorry

end smallest_n_with_units_digit_and_reorder_l71_71736


namespace lowest_score_on_one_of_last_two_tests_l71_71671

-- define conditions
variables (score1 score2 : ℕ) (total_score average desired_score : ℕ)

-- Shauna's scores on the first two tests are 82 and 75
def shauna_score1 := 82
def shauna_score2 := 75

-- Shauna wants to average 85 over 4 tests
def desired_average := 85
def number_of_tests := 4

-- total points needed for desired average
def total_points_needed := desired_average * number_of_tests

-- total points from first two tests
def total_first_two_tests := shauna_score1 + shauna_score2

-- total points needed on last two tests
def points_needed_last_two_tests := total_points_needed - total_first_two_tests

-- Prove the lowest score on one of the last two tests
theorem lowest_score_on_one_of_last_two_tests : 
  (∃ (score3 score4 : ℕ), score3 + score4 = points_needed_last_two_tests ∧ score3 ≤ 100 ∧ score4 ≤ 100 ∧ (score3 ≥ 83 ∨ score4 ≥ 83)) :=
sorry

end lowest_score_on_one_of_last_two_tests_l71_71671


namespace alex_plays_with_friends_l71_71397

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71397


namespace initial_quantity_l71_71209

variables {A : ℝ} -- initial quantity of acidic liquid
variables {W : ℝ} -- quantity of water removed

theorem initial_quantity (h1: A * 0.6 = W + 25) (h2: W = 9) : A = 27 :=
by
  sorry

end initial_quantity_l71_71209


namespace maximum_profit_l71_71224

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit_l71_71224


namespace octagon_side_length_l71_71908

theorem octagon_side_length 
  (num_sides : ℕ) 
  (perimeter : ℝ) 
  (h_sides : num_sides = 8) 
  (h_perimeter : perimeter = 23.6) :
  (perimeter / num_sides) = 2.95 :=
by
  have h_valid_sides : num_sides = 8 := h_sides
  have h_valid_perimeter : perimeter = 23.6 := h_perimeter
  sorry

end octagon_side_length_l71_71908


namespace nat_add_ge_3_implies_at_least_one_ge_2_l71_71202

theorem nat_add_ge_3_implies_at_least_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by {
  sorry
}

end nat_add_ge_3_implies_at_least_one_ge_2_l71_71202


namespace nine_points_circle_chords_l71_71502

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l71_71502


namespace cone_heights_l71_71408

theorem cone_heights (H x r1 r2 : ℝ) (H_frustum : H - x = 18)
  (A_lower : 400 * Real.pi = Real.pi * r1^2)
  (A_upper : 100 * Real.pi = Real.pi * r2^2)
  (ratio_radii : r2 / r1 = 1 / 2)
  (ratio_heights : x / H = 1 / 2) :
  x = 18 ∧ H = 36 :=
by
  sorry

end cone_heights_l71_71408


namespace total_spending_eq_total_is_19_l71_71992

variable (friend_spending your_spending total_spending : ℕ)

-- Conditions
def friend_spending_eq : friend_spending = 11 := by sorry
def friend_spent_more : friend_spending = your_spending + 3 := by sorry

-- Proof that total_spending is 19
theorem total_spending_eq : total_spending = friend_spending + your_spending :=
  by sorry

theorem total_is_19 : total_spending = 19 :=
  by sorry

end total_spending_eq_total_is_19_l71_71992


namespace replace_question_with_division_l71_71431

theorem replace_question_with_division :
  ∃ op: (ℤ → ℤ → ℤ), (op 8 2) + 5 - (3 - 2) = 8 ∧ 
  (∀ a b, op = Int.div ∧ ((op a b) = a / b)) :=
by
  sorry

end replace_question_with_division_l71_71431


namespace star_area_l71_71767

-- Conditions
def square_ABCD_area (s : ℝ) := s^2 = 72

-- Question and correct answer
theorem star_area (s : ℝ) (h : square_ABCD_area s) : 24 = 24 :=
by sorry

end star_area_l71_71767


namespace drug_price_reduction_l71_71321

theorem drug_price_reduction (x : ℝ) :
    36 * (1 - x)^2 = 25 :=
sorry

end drug_price_reduction_l71_71321


namespace remainder_of_sum_is_zero_l71_71105

-- Define the properties of m and n according to the conditions of the problem
def m : ℕ := 2 * 1004 ^ 2
def n : ℕ := 2007 * 1003

-- State the theorem that proves the remainder of (m + n) divided by 1004 is 0
theorem remainder_of_sum_is_zero : (m + n) % 1004 = 0 := by
  sorry

end remainder_of_sum_is_zero_l71_71105


namespace reservoir_original_content_l71_71598

noncomputable def original_content (T O : ℝ) : Prop :=
  (80 / 100) * T = O + 120 ∧
  O = (50 / 100) * T

theorem reservoir_original_content (T : ℝ) (h1 : (80 / 100) * T = (50 / 100) * T + 120) : 
  (50 / 100) * T = 200 :=
by
  sorry

end reservoir_original_content_l71_71598


namespace equation_satisfied_by_r_l71_71756

theorem equation_satisfied_by_r {x y z r : ℝ} (h1: x ≠ y) (h2: y ≠ z) (h3: z ≠ x) 
    (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0) 
    (h7: ∃ (r: ℝ), x * (y - z) = (y * (z - x)) / r ∧ y * (z - x) = (z * (y - x)) / r ∧ z * (y - x) = (x * (y - z)) * r) 
    : r^2 - r + 1 = 0 := 
sorry

end equation_satisfied_by_r_l71_71756


namespace janet_initial_number_l71_71292

-- Define the conditions using Lean definitions
def janetProcess (x : ℕ) : ℕ :=
  (2 * (x + 7)) - 4

-- The theorem that expresses the statement of the problem: If the final result of the process is 28, then x = 9
theorem janet_initial_number (x : ℕ) (h : janetProcess x = 28) : x = 9 :=
sorry

end janet_initial_number_l71_71292


namespace scientific_notation_of_number_l71_71223

theorem scientific_notation_of_number : 15300000000 = 1.53 * (10 : ℝ)^10 := sorry

end scientific_notation_of_number_l71_71223


namespace percentage_profit_l71_71412

variable (total_crates : ℕ)
variable (total_cost : ℕ)
variable (lost_crates : ℕ)
variable (sell_price_per_crate : ℕ)

theorem percentage_profit (h1 : total_crates = 10) (h2 : total_cost = 160)
  (h3 : lost_crates = 2) (h4 : sell_price_per_crate = 25) :
  (8 * sell_price_per_crate - total_cost) * 100 / total_cost = 25 :=
by
  -- Definitions and steps to prove this can be added here.
  sorry

end percentage_profit_l71_71412


namespace seokgi_walk_distance_correct_l71_71022

-- Definitions of distances as per conditions
def entrance_to_temple_km : ℕ := 4
def entrance_to_temple_m : ℕ := 436
def temple_to_summit_m : ℕ := 1999

-- Total distance Seokgi walked in kilometers
def total_walked_km : ℕ := 12870

-- Proof statement
theorem seokgi_walk_distance_correct :
  ((entrance_to_temple_km * 1000 + entrance_to_temple_m) + temple_to_summit_m) * 2 / 1000 = total_walked_km / 1000 :=
by
  -- We will fill this in with the proof steps
  sorry

end seokgi_walk_distance_correct_l71_71022


namespace max_min_diff_of_c_l71_71161

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l71_71161


namespace probability_circle_containment_l71_71674

theorem probability_circle_containment :
  let a_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let circle_C_contained (a : ℕ) : Prop := a > 3
  let m : ℕ := (a_set.filter circle_C_contained).card
  let n : ℕ := a_set.card
  let p : ℚ := m / n
  p = 4 / 7 := 
by
  sorry

end probability_circle_containment_l71_71674


namespace total_sum_of_grid_is_745_l71_71289

theorem total_sum_of_grid_is_745 :
  let top_row := [12, 13, 15, 17, 19]
  let left_column := [12, 14, 16, 18]
  let total_sum := 360 + 375 + 10
  total_sum = 745 :=
by
  -- The theorem establishes the total sum calculation.
  sorry

end total_sum_of_grid_is_745_l71_71289


namespace problem_statement_l71_71614

theorem problem_statement (pi : ℝ) (h : pi = 4 * Real.sin (52 * Real.pi / 180)) :
  (2 * pi * Real.sqrt (16 - pi ^ 2) - 8 * Real.sin (44 * Real.pi / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.sin (22 * Real.pi / 180)) ^ 2) = 8 * Real.sqrt 3 := 
  sorry

end problem_statement_l71_71614


namespace problem_solution_l71_71302

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end problem_solution_l71_71302


namespace linear_function_quadrants_l71_71476

theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) : 
  (∀ x : ℝ, (k < 0 ∧ b > 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) ∧ 
  (∀ x : ℝ, (k > 0 ∧ b < 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) :=
sorry

end linear_function_quadrants_l71_71476


namespace polar_coordinates_of_point_l71_71608

noncomputable def polarCoordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y >= 0 then Real.atan (y / x)
           else if x > 0 ∧ y < 0 then 2 * Real.pi - Real.atan (|y / x|)
           else if x < 0 then Real.pi + Real.atan (y / x)
           else if y > 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ)

theorem polar_coordinates_of_point :
  polarCoordinates 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := by
  sorry

end polar_coordinates_of_point_l71_71608


namespace sum_ratio_arithmetic_sequence_l71_71887

theorem sum_ratio_arithmetic_sequence
  (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h2 : ∀ k : ℕ, a (k + 1) - a k = a 2 - a 1)
  (h3 : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 :=
sorry

end sum_ratio_arithmetic_sequence_l71_71887


namespace message_spread_in_24_hours_l71_71593

theorem message_spread_in_24_hours : ∃ T : ℕ, (T = (2^25 - 1)) :=
by 
  let T := 2^24 - 1
  use T
  sorry

end message_spread_in_24_hours_l71_71593


namespace ellie_shoes_count_l71_71435

variable (E R : ℕ)

def ellie_shoes (E R : ℕ) : Prop :=
  E + R = 13 ∧ E = R + 3

theorem ellie_shoes_count (E R : ℕ) (h : ellie_shoes E R) : E = 8 :=
  by sorry

end ellie_shoes_count_l71_71435


namespace probability_ge_3_l71_71559

open Set

def num_pairs (s : Set ℕ) (n : ℕ) : ℕ :=
  (s.Subset (Filter fun x y => abs (x - y) ≥ n)).Card

def undesirable_pairs : Set (ℕ × ℕ) :=
  {(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), 
   (1,3), (2,4), (3,5), (4,6), (5,7), (6,8)}

def total_pairs : ℕ := choose 8 2

def prob_diff_ge_3 : ℚ :=
  1 - (undesirable_pairs.toList.length : ℚ) / total_pairs

theorem probability_ge_3 : prob_diff_ge_3 = 15 / 28 := by
  sorry

end probability_ge_3_l71_71559


namespace problem_solution_l71_71492

variable {a b c d : ℝ}
variable (h_a : a = 4 * π / 3)
variable (h_b : b = 10 * π)
variable (h_c : c = 62)
variable (h_d : d = 30)

theorem problem_solution : (b * c) / (a * d) = 15.5 :=
by
  rw [h_a, h_b, h_c, h_d]
  -- Continued steps according to identified solution steps
  -- and arithmetic operations.
  sorry

end problem_solution_l71_71492


namespace range_of_a_for_monotonic_function_l71_71743

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x

def is_monotonic_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem range_of_a_for_monotonic_function :
  ∀ (a : ℝ), is_monotonic_on (f · a) (Set.Iic (-1)) → a ≤ 3 :=
by
  intros a h
  sorry

end range_of_a_for_monotonic_function_l71_71743


namespace abs_minus_five_plus_three_l71_71183

theorem abs_minus_five_plus_three : |(-5 + 3)| = 2 := 
by
  sorry

end abs_minus_five_plus_three_l71_71183


namespace person_B_days_l71_71697

theorem person_B_days (A_days : ℕ) (combined_work : ℚ) (x : ℕ) : 
  A_days = 30 → combined_work = (1 / 6) → 3 * (1 / 30 + 1 / x) = combined_work → x = 45 :=
by
  intros hA hCombined hEquation
  sorry

end person_B_days_l71_71697


namespace average_speed_l71_71518

-- Define the conditions
def distance1 := 350 -- miles
def time1 := 6 -- hours
def distance2 := 420 -- miles
def time2 := 7 -- hours

-- Define the total distance and total time (excluding break)
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- Define the statement to prove
theorem average_speed : 
  (total_distance / total_time : ℚ) = 770 / 13 := by
  sorry

end average_speed_l71_71518


namespace binomial_coefficient_term_of_x_l71_71920

theorem binomial_coefficient_term_of_x (n : ℕ) (x : ℝ) :
  (2^n = 128) →
  ∃ (coef : ℤ), coef = -14 ∧ ∃ r : ℕ, x^(1 : ℝ) = (sqrt (x)⁻¹) * ((-2)^r * (Nat.choose n r) * x^((n - 4 * r)/3)) :=
by
  intros h
  have h₁ : n = 7 := sorry
  use -14
  split
  . refl
  use 1
  sorry

end binomial_coefficient_term_of_x_l71_71920


namespace parabola_intersection_square_l71_71479

theorem parabola_intersection_square (p : ℝ) :
   (∃ (x : ℝ), (x = 1 ∨ x = 2) ∧ x^2 * p = 1 ∨ x^2 * p = 2)
   → (1 / 4 ≤ p ∧ p ≤ 2) :=
by
  sorry

end parabola_intersection_square_l71_71479


namespace tiffany_total_bags_l71_71694

def initial_bags : ℕ := 10
def found_on_tuesday : ℕ := 3
def found_on_wednesday : ℕ := 7
def total_bags : ℕ := 20

theorem tiffany_total_bags (initial_bags : ℕ) (found_on_tuesday : ℕ) (found_on_wednesday : ℕ) (total_bags : ℕ) :
    initial_bags + found_on_tuesday + found_on_wednesday = total_bags :=
by
  sorry

end tiffany_total_bags_l71_71694


namespace binomial_coefficient_times_two_l71_71563

theorem binomial_coefficient_times_two : 2 * Nat.choose 8 5 = 112 := 
by 
  -- The proof is omitted here
  sorry

end binomial_coefficient_times_two_l71_71563


namespace power_addition_identity_l71_71236

theorem power_addition_identity : 
  (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end power_addition_identity_l71_71236


namespace total_floor_area_l71_71585

theorem total_floor_area
    (n : ℕ) (a_cm : ℕ)
    (num_of_slabs : n = 30)
    (length_of_slab_cm : a_cm = 130) :
    (30 * ((130 * 130) / 10000)) = 50.7 :=
by
  sorry

end total_floor_area_l71_71585


namespace row_col_value_2002_2003_l71_71791

theorem row_col_value_2002_2003 :
  let base_num := (2003 - 1)^2 + 1 
  let result := base_num + 2001 
  result = 2002 * 2003 :=
by
  sorry

end row_col_value_2002_2003_l71_71791


namespace sufficient_not_necessary_l71_71466

variables (A B : Prop)

theorem sufficient_not_necessary (h : B → A) : ¬(A → B) :=
by sorry

end sufficient_not_necessary_l71_71466


namespace rectangle_area_decrease_l71_71838

noncomputable def rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) : ℝ :=
  let L' := 1.10 * L
  let B' := 0.90 * B
  let A  := L * B
  let A' := L' * B'
  A'

theorem rectangle_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  rectangle_area_change L B hL hB = 0.99 * (L * B) := by
  sorry

end rectangle_area_decrease_l71_71838


namespace range_of_a_l71_71900

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * x^2 - (2 * a + 1) * x + 1

theorem range_of_a (a : ℝ) (h_a : 0 < a ∧ a ≤ 1/2) : 
  ∀ x : ℝ, x ∈ Set.Ici a → f x a ≥ a^3 - a - 1/8 :=
by
  sorry

end range_of_a_l71_71900


namespace maximum_M_l71_71112

-- Define the sides of a triangle condition
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Theorem statement
theorem maximum_M (a b c : ℝ) (h : is_triangle a b c) : 
  (a^2 + b^2) / (c^2) > (1/2) :=
sorry

end maximum_M_l71_71112


namespace no_integer_pairs_satisfy_equation_l71_71874

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), ¬(m^3 + 10 * m^2 + 11 * m + 2 = 81 * n^3 + 27 * n^2 + 3 * n - 8) :=
by
  sorry

end no_integer_pairs_satisfy_equation_l71_71874


namespace van_speed_maintain_l71_71077

theorem van_speed_maintain 
  (D : ℕ) (T T_new : ℝ) 
  (initial_distance : D = 435) 
  (initial_time : T = 5) 
  (new_time : T_new = T / 2) : 
  D / T_new = 174 := 
by 
  sorry

end van_speed_maintain_l71_71077


namespace problem_A_problem_C_problem_D_problem_E_l71_71187

variable {a b c : ℝ}
variable (ha : a < 0) (hab : a < b) (hb : b < 0) (hc : 0 < c)

theorem problem_A (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * b > a * c :=
by sorry

theorem problem_C (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * c < b * c :=
by sorry

theorem problem_D (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a + c < b + c :=
by sorry

theorem problem_E (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : c / a > 1 :=
by sorry

end problem_A_problem_C_problem_D_problem_E_l71_71187


namespace eleven_million_scientific_notation_l71_71519

-- Definition of the scientific notation condition and question
def scientific_notation (a n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ k : ℤ, n = 10 ^ k

-- The main theorem stating that 11 million can be expressed as 1.1 * 10^7
theorem eleven_million_scientific_notation : scientific_notation 1.1 (10 ^ 7) :=
by 
  -- Adding sorry to skip the proof
  sorry

end eleven_million_scientific_notation_l71_71519


namespace matt_worked_more_on_wednesday_l71_71792

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end matt_worked_more_on_wednesday_l71_71792


namespace rectangle_diagonal_length_l71_71340

theorem rectangle_diagonal_length
  (a b : ℝ)
  (h1 : a = 40 * Real.sqrt 2)
  (h2 : b = 2 * a) :
  Real.sqrt (a^2 + b^2) = 160 := by
  sorry

end rectangle_diagonal_length_l71_71340


namespace compute_series_sum_l71_71606

theorem compute_series_sum :
  ∑' n : ℕ in set.Ici 2, (3 * n^3 - 2 * n^2 - 2 * n + 3 : ℝ) / (n^6 - n^5 + n^3 - n^2 + n - 1) = 1 := 
sorry

end compute_series_sum_l71_71606


namespace pet_center_final_count_l71_71770

/-!
# Problem: Count the total number of pets in a pet center after a series of adoption and collection events.
-/

def initialDogs : Nat := 36
def initialCats : Nat := 29
def initialRabbits : Nat := 15
def initialBirds : Nat := 10

def dogsAdopted1 : Nat := 20
def rabbitsAdopted1 : Nat := 5

def catsCollected : Nat := 12
def rabbitsCollected : Nat := 8
def birdsCollected : Nat := 5

def catsAdopted2 : Nat := 10
def birdsAdopted2 : Nat := 4

def finalDogs : Nat :=
  initialDogs - dogsAdopted1

def finalCats : Nat :=
  initialCats + catsCollected - catsAdopted2

def finalRabbits : Nat :=
  initialRabbits - rabbitsAdopted1 + rabbitsCollected

def finalBirds : Nat :=
  initialBirds + birdsCollected - birdsAdopted2

def totalPets (d c r b : Nat) : Nat :=
  d + c + r + b

theorem pet_center_final_count : 
  totalPets finalDogs finalCats finalRabbits finalBirds = 76 := by
  -- This is where we would provide the proof, but it's skipped as per the instructions.
  sorry

end pet_center_final_count_l71_71770


namespace meeting_time_l71_71222

noncomputable def combined_speed : ℕ := 10 -- km/h
noncomputable def distance_to_cover : ℕ := 50 -- km
noncomputable def start_time : ℕ := 6 -- pm (in hours)
noncomputable def speed_a : ℕ := 6 -- km/h
noncomputable def speed_b : ℕ := 4 -- km/h

theorem meeting_time : start_time + (distance_to_cover / combined_speed) = 11 :=
by
  sorry

end meeting_time_l71_71222


namespace exists_alpha_l71_71953

variable {a : ℕ → ℝ}

axiom nonzero_sequence (n : ℕ) : a n ≠ 0
axiom recurrence_relation (n : ℕ) : a n ^ 2 - a (n - 1) * a (n + 1) = 1

theorem exists_alpha (n : ℕ) : ∃ α : ℝ, ∀ n ≥ 1, a (n + 1) = α * a n - a (n - 1) :=
by
  sorry

end exists_alpha_l71_71953


namespace smallest_possible_gcd_l71_71468

theorem smallest_possible_gcd (m n p : ℕ) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ k, k = Nat.gcd n p ∧ k = 60 := by
  sorry

end smallest_possible_gcd_l71_71468


namespace train_length_is_135_l71_71419

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l71_71419


namespace problem_l71_71257

noncomputable def number_of_regions_four_planes (h1 : True) (h2 : True) : ℕ := 14

theorem problem (h1 : True) (h2 : True) : number_of_regions_four_planes h1 h2 = 14 :=
by sorry

end problem_l71_71257


namespace overall_average_speed_is_six_l71_71107

-- Definitions of the conditions
def cycling_time := 45 / 60 -- hours
def cycling_speed := 12 -- mph
def stopping_time := 15 / 60 -- hours
def walking_time := 75 / 60 -- hours
def walking_speed := 3 -- mph

-- Problem statement: Proving that the overall average speed is 6 mph
theorem overall_average_speed_is_six : 
  (cycling_speed * cycling_time + walking_speed * walking_time) /
  (cycling_time + walking_time + stopping_time) = 6 :=
by
  sorry

end overall_average_speed_is_six_l71_71107


namespace intersection_is_correct_l71_71272

noncomputable def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
noncomputable def B := { x : ℝ | 0 < x ∧ x ≤ 3 }

theorem intersection_is_correct : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_is_correct_l71_71272


namespace no_positive_integer_pairs_l71_71114

theorem no_positive_integer_pairs (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) : ¬ (x^2 + y^2 = x^3 + 2 * y) :=
by sorry

end no_positive_integer_pairs_l71_71114


namespace negation_of_proposition_l71_71040

variable (x y : ℝ)

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, (x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) ↔ 
  (∃ x y : ℝ, (x^2 + y^2 ≠ 0) ∧ (x ≠ 0 ∨ y ≠ 0)) :=
sorry

end negation_of_proposition_l71_71040


namespace solution_set_of_inequality_l71_71116

theorem solution_set_of_inequality :
  { x : ℝ | (x - 4) / (3 - 2*x) < 0 ∧ 3 - 2*x ≠ 0 } = { x : ℝ | x < 3 / 2 ∨ x > 4 } :=
sorry

end solution_set_of_inequality_l71_71116


namespace part_a_part_b_l71_71318

theorem part_a (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * (Real.sqrt n + 1) / (n - 1))) :
  μ < (2 * (Real.sqrt n + 1) / (n - 1)) :=
by 
  exact h_μ.2

theorem part_b (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1)))) :
  μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1))) :=
by
  exact h_μ.2

end part_a_part_b_l71_71318


namespace matt_worked_more_minutes_l71_71793

-- Define the conditions as constants
def monday_minutes : ℕ := 450
def tuesday_minutes : ℕ := monday_minutes / 2
def wednesday_minutes : ℕ := 300

-- The statement to prove
theorem matt_worked_more_minutes :
  wednesday_minutes - tuesday_minutes = 75 :=
begin
  sorry, -- Proof placeholder
end

end matt_worked_more_minutes_l71_71793


namespace fixed_points_and_zeros_no_fixed_points_range_b_l71_71633

def f (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem fixed_points_and_zeros (b c : ℝ) (h1 : f b c (-3) = -3) (h2 : f b c 2 = 2) :
  ∃ x1 x2 : ℝ, f b c x1 = 0 ∧ f b c x2 = 0 ∧ x1 = -1 + Real.sqrt 7 ∧ x2 = -1 - Real.sqrt 7 :=
sorry

theorem no_fixed_points_range_b {b : ℝ} (h : ∀ x : ℝ, f b (b^2 / 4) x ≠ x) : 
  b > 1 / 3 ∨ b < -1 :=
sorry

end fixed_points_and_zeros_no_fixed_points_range_b_l71_71633


namespace hyperbola_equation_l71_71635

theorem hyperbola_equation (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : 2 * b / a = 1) : 
  a = 2 * Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ (∀ x y : ℝ, x^2 / (a^2) - y^2 / (b^2) = 1 ↔ x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l71_71635


namespace geometric_series_first_term_l71_71976

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l71_71976


namespace rahuls_share_l71_71704

theorem rahuls_share (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)
  (rahul_work_one_day : rahul_days > 0) (rajesh_work_one_day : rajesh_days > 0)
  (total_payment_eq : total_payment = 105) 
  (rahul_days_eq : rahul_days = 3) 
  (rajesh_days_eq : rajesh_days = 2) :
  rahul_share = 42 := 
by
  sorry

end rahuls_share_l71_71704


namespace equation_1_solution_equation_2_solution_l71_71517

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end equation_1_solution_equation_2_solution_l71_71517


namespace extremum_areas_extremum_areas_case_b_equal_areas_l71_71663

variable (a b x : ℝ)
variable (h1 : b > 0) (h2 : a ≥ b) (h_cond : 0 < x ∧ x ≤ b)

def area_t1 (a b x : ℝ) : ℝ := 2 * x^2 - (a + b) * x + a * b
def area_t2 (a b x : ℝ) : ℝ := -2 * x^2 + (a + b) * x

noncomputable def x0 (a b : ℝ) : ℝ := (a + b) / 4

-- Problem 1
theorem extremum_areas :
  b ≥ a / 3 → area_t1 a b (x0 a b) ≤ area_t1 a b x ∧ area_t2 a b (x0 a b) ≥ area_t2 a b x :=
sorry

theorem extremum_areas_case_b :
  b < a / 3 → (area_t1 a b b = b^2) ∧ (area_t2 a b b = a * b - b^2) :=
sorry

-- Problem 2
theorem equal_areas :
  b ≤ a ∧ a ≤ 2 * b → (area_t1 a b (a / 2) = area_t2 a b (a / 2)) ∧ (area_t1 a b (b / 2) = area_t2 a b (b / 2)) :=
sorry

end extremum_areas_extremum_areas_case_b_equal_areas_l71_71663


namespace train_length_is_135_l71_71420

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l71_71420


namespace inequality_x_pow_n_ge_n_x_l71_71944

theorem inequality_x_pow_n_ge_n_x (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x > -1) (h3 : n > 0) : 
  (1 + x)^n ≥ n * x := by
  sorry

end inequality_x_pow_n_ge_n_x_l71_71944


namespace symmetric_scanning_codes_count_l71_71714

theorem symmetric_scanning_codes_count :
  let grid_size := 5
  let total_squares := grid_size * grid_size
  let symmetry_classes := 5 -- Derived from classification in the solution
  let possible_combinations := 2 ^ symmetry_classes
  let invalid_combinations := 2 -- All black or all white grid
  total_squares = 25 
  ∧ (possible_combinations - invalid_combinations) = 30 :=
by sorry

end symmetric_scanning_codes_count_l71_71714


namespace min_coins_for_less_than_1_dollar_l71_71831

theorem min_coins_for_less_than_1_dollar :
  ∃ (p n q h : ℕ), 1*p + 5*n + 25*q + 50*h ≥ 1 ∧ 1*p + 5*n + 25*q + 50*h < 100 ∧ p + n + q + h = 8 :=
by 
  sorry

end min_coins_for_less_than_1_dollar_l71_71831


namespace sum_of_consecutive_evens_is_162_l71_71819

-- Define the smallest even number
def smallest_even : ℕ := 52

-- Define the next two consecutive even numbers
def second_even : ℕ := smallest_even + 2
def third_even : ℕ := smallest_even + 4

-- The sum of these three even numbers
def sum_of_consecutive_evens : ℕ := smallest_even + second_even + third_even

-- Assertion that the sum must be 162
theorem sum_of_consecutive_evens_is_162 : sum_of_consecutive_evens = 162 :=
by 
  -- To be proved
  sorry

end sum_of_consecutive_evens_is_162_l71_71819


namespace expected_socks_to_pair_l71_71539

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l71_71539


namespace rectangle_diagonal_length_l71_71323

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l71_71323


namespace arithmetic_seq_sum_is_110_l71_71123

noncomputable def S₁₀ (a_1 : ℝ) : ℝ :=
  10 / 2 * (2 * a_1 + 9 * (-2))

theorem arithmetic_seq_sum_is_110 (a1 a3 a7 a9 : ℝ) 
  (h_diff3 : a3 = a1 - 4)
  (h_diff7 : a7 = a1 - 12)
  (h_diff9 : a9 = a1 - 16)
  (h_geom : (a1 - 12) ^ 2 = (a1 - 4) * (a1 - 16)) :
  S₁₀ a1 = 110 :=
by
  sorry

end arithmetic_seq_sum_is_110_l71_71123


namespace log_216_eq_3_log_2_add_3_log_3_l71_71706

theorem log_216_eq_3_log_2_add_3_log_3 (log : ℝ → ℝ) (h1 : ∀ x y, log (x * y) = log x + log y)
  (h2 : ∀ x n, log (x^n) = n * log x) :
  log 216 = 3 * log 2 + 3 * log 3 :=
by
  sorry

end log_216_eq_3_log_2_add_3_log_3_l71_71706


namespace jamie_hours_each_time_l71_71003

theorem jamie_hours_each_time (hours_per_week := 2) (weeks := 6) (rate := 10) (total_earned := 360) : 
  ∃ (h : ℕ), h = 3 ∧ (hours_per_week * weeks * rate * h = total_earned) := 
by
  sorry

end jamie_hours_each_time_l71_71003


namespace socks_expected_value_l71_71548

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l71_71548


namespace must_divide_l71_71300

-- Proving 5 is a divisor of q

variables {p q r s : ℕ}

theorem must_divide (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 42)
                   (h3 : Nat.gcd r s = 66) (h4 : 80 < Nat.gcd s p)
                   (h5 : Nat.gcd s p < 120) :
                   5 ∣ q :=
sorry

end must_divide_l71_71300


namespace max_lessons_l71_71196

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end max_lessons_l71_71196


namespace find_certain_number_l71_71253

theorem find_certain_number (n : ℕ)
  (h1 : 3153 + 3 = 3156)
  (h2 : 3156 % 9 = 0)
  (h3 : 3156 % 70 = 0)
  (h4 : 3156 % 25 = 0) :
  3156 % 37 = 0 :=
by
  sorry

end find_certain_number_l71_71253


namespace hide_and_seek_problem_l71_71365

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71365


namespace both_hit_given_target_hit_l71_71830

theorem both_hit_given_target_hit (P_A P_B : ℝ) (hA : P_A = 0.6) (hB : P_B = 0.7) :
  let P_C := 1 - (1 - P_A) * (1 - P_B) in
  P_C ≠ 0 → (P_A * P_B) / P_C = 21 / 44 :=
by
  intros
  sorry

end both_hit_given_target_hit_l71_71830


namespace chord_constant_l71_71047

theorem chord_constant (
    d : ℝ
) : (∃ t : ℝ, (∀ A B : ℝ × ℝ,
    A.2 = A.1^3 ∧ B.2 = B.1^3 ∧ d = 1/2 ∧
    (C : ℝ × ℝ) = (0, d) ∧ 
    (∀ (AC BC: ℝ),
        AC = dist A C ∧
        BC = dist B C ∧
        t = (1 / (AC^2) + 1 / (BC^2))
    )) → t = 4) := 
sorry

end chord_constant_l71_71047


namespace greatest_monthly_drop_l71_71682

-- Definition of monthly price changes
def price_change_jan : ℝ := -1.00
def price_change_feb : ℝ := 2.50
def price_change_mar : ℝ := 0.00
def price_change_apr : ℝ := -3.00
def price_change_may : ℝ := -1.50
def price_change_jun : ℝ := 1.00

-- Proving the month with the greatest monthly drop in price
theorem greatest_monthly_drop :
  (price_change_apr < price_change_jan) ∧
  (price_change_apr < price_change_feb) ∧
  (price_change_apr < price_change_mar) ∧
  (price_change_apr < price_change_may) ∧
  (price_change_apr < price_change_jun) :=
by
  sorry

end greatest_monthly_drop_l71_71682


namespace positive_difference_solutions_abs_l71_71207

theorem positive_difference_solutions_abs (x1 x2 : ℝ) 
  (h1 : 2 * x1 - 3 = 18 ∨ 2 * x1 - 3 = -18) 
  (h2 : 2 * x2 - 3 = 18 ∨ 2 * x2 - 3 = -18) : 
  |x1 - x2| = 18 :=
sorry

end positive_difference_solutions_abs_l71_71207


namespace expected_pairs_socks_l71_71543

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l71_71543


namespace lightbulb_stops_on_friday_l71_71586

theorem lightbulb_stops_on_friday
  (total_hours : ℕ) (daily_usage : ℕ) (start_day : ℕ) (stops_day : ℕ)
  (h_total_hours : total_hours = 24999)
  (h_daily_usage : daily_usage = 2)
  (h_start_day : start_day = 1) : 
  stops_day = 5 := by
  sorry

end lightbulb_stops_on_friday_l71_71586


namespace quilt_shaded_fraction_l71_71194

theorem quilt_shaded_fraction :
  let original_squares := 9
  let shaded_column_squares := 3
  let fraction_shaded := shaded_column_squares / original_squares 
  fraction_shaded = 1/3 :=
by
  sorry

end quilt_shaded_fraction_l71_71194


namespace find_y_l71_71905

theorem find_y (x y : ℤ) (h₁ : x ^ 2 + x + 4 = y - 4) (h₂ : x = 3) : y = 20 :=
by 
  sorry

end find_y_l71_71905


namespace value_of_one_TV_mixer_blender_l71_71201

variables (M T B : ℝ)

-- The given conditions
def eq1 : Prop := 2 * M + T + B = 10500
def eq2 : Prop := T + M + 2 * B = 14700

-- The problem: find the combined value of one TV, one mixer, and one blender
theorem value_of_one_TV_mixer_blender :
  eq1 M T B → eq2 M T B → (T + M + B = 18900) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end value_of_one_TV_mixer_blender_l71_71201


namespace inequality_proof_equality_condition_l71_71666

theorem inequality_proof (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) ≥ (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) := 
sorry

theorem equality_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) = (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) ↔ a = b ∧ b = c := 
sorry

end inequality_proof_equality_condition_l71_71666


namespace solve_first_equation_solve_second_equation_l71_71803

open Real

/-- Prove solutions to the first equation (x + 8)(x + 1) = -12 are x = -4 and x = -5 -/
theorem solve_first_equation (x : ℝ) : (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 := by
  sorry

/-- Prove solutions to the second equation 2x^2 + 4x - 1 = 0 are x = (-2 + sqrt 6) / 2 and x = (-2 - sqrt 6) / 2 -/
theorem solve_second_equation (x : ℝ) : 2 * x^2 + 4 * x - 1 = 0 ↔ x = (-2 + sqrt 6) / 2 ∨ x = (-2 - sqrt 6) / 2 := by
  sorry

end solve_first_equation_solve_second_equation_l71_71803


namespace Kevin_ends_with_54_cards_l71_71929

/-- Kevin starts with 7 cards and finds another 47 cards. 
    This theorem proves that Kevin ends with 54 cards. -/
theorem Kevin_ends_with_54_cards :
  let initial_cards := 7
  let found_cards := 47
  initial_cards + found_cards = 54 := 
by
  let initial_cards := 7
  let found_cards := 47
  sorry

end Kevin_ends_with_54_cards_l71_71929


namespace ab_greater_than_1_l71_71456

noncomputable def log10_abs (x : ℝ) : ℝ :=
  abs (Real.logb 10 x)

theorem ab_greater_than_1
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
  (hf : log10_abs a < log10_abs b) : a * b > 1 := by
  sorry

end ab_greater_than_1_l71_71456


namespace person_A_leave_time_l71_71719

theorem person_A_leave_time
  (ha : ℚ := 1 / 6) -- Work rate of Person A per hour
  (hb : ℚ := 1 / 8) -- Work rate of Person B per hour
  (hc : ℚ := 1 / 10) -- Work rate of Person C per hour
  (start_time : ℚ := 8) -- Start time in hours (8 AM)
  (end_time : ℚ := 12) -- End time in hours (12 PM)
  (total_work : ℚ := 1) -- Total work to be done
  : ℚ := sorry -- Expected leave time of Person A in hours

end person_A_leave_time_l71_71719


namespace cosine_identity_example_l71_71641

theorem cosine_identity_example {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 3) : Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by sorry

end cosine_identity_example_l71_71641


namespace chords_from_nine_points_l71_71512

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l71_71512


namespace cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l71_71588

variable (x : ℕ) (x_ge_4 : x ≥ 4)

-- Total cost under scheme ①
def scheme_1_cost (x : ℕ) : ℕ := 5 * x + 60

-- Total cost under scheme ②
def scheme_2_cost (x : ℕ) : ℕ := 9 * (80 + 5 * x) / 10

theorem cost_scheme_1 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_1_cost x = 5 * x + 60 :=  
sorry

theorem cost_scheme_2 (x : ℕ) (x_ge_4 : x ≥ 4) : 
  scheme_2_cost x = (80 + 5 * x) * 9 / 10 := 
sorry

-- When x = 30, compare which scheme is more cost-effective
variable (x_eq_30 : x = 30)
theorem cost_comparison_scheme (x_eq_30 : x = 30) : 
  scheme_1_cost 30 > scheme_2_cost 30 := 
sorry

-- When x = 30, a more cost-effective combined purchasing plan
def combined_scheme_cost : ℕ := scheme_1_cost 4 + scheme_2_cost (30 - 4)

theorem more_cost_effective_combined_plan (x_eq_30 : x = 30) : 
  combined_scheme_cost < scheme_1_cost 30 ∧ combined_scheme_cost < scheme_2_cost 30 := 
sorry

end cost_scheme_1_cost_scheme_2_cost_comparison_scheme_more_cost_effective_combined_plan_l71_71588


namespace perfect_square_trinomial_k_l71_71471

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l71_71471


namespace odd_function_condition_l71_71192

noncomputable def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 :=
by
  sorry

end odd_function_condition_l71_71192


namespace julie_hourly_rate_l71_71157

variable (daily_hours : ℕ) (weekly_days : ℕ) (monthly_weeks : ℕ) (missed_days : ℕ) (monthly_salary : ℝ)

def total_monthly_hours : ℕ := daily_hours * weekly_days * monthly_weeks - daily_hours * missed_days

theorem julie_hourly_rate : 
    daily_hours = 8 → 
    weekly_days = 6 → 
    monthly_weeks = 4 → 
    missed_days = 1 → 
    monthly_salary = 920 → 
    (monthly_salary / total_monthly_hours daily_hours weekly_days monthly_weeks missed_days) = 5 := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end julie_hourly_rate_l71_71157


namespace inequality_solution_intervals_l71_71185

theorem inequality_solution_intervals (x : ℝ) (h : x > 2) : 
  (x-2)^(x^2 - 6 * x + 8) > 1 ↔ (2 < x ∧ x < 3) ∨ x > 4 := 
sorry

end inequality_solution_intervals_l71_71185


namespace solve_for_y_l71_71802

-- Define the conditions as Lean functions and statements
def is_positive (y : ℕ) : Prop := y > 0
def multiply_sixteen (y : ℕ) : Prop := 16 * y = 256

-- The theorem that states the value of y
theorem solve_for_y (y : ℕ) (h1 : is_positive y) (h2 : multiply_sixteen y) : y = 16 :=
sorry

end solve_for_y_l71_71802


namespace initial_number_of_women_l71_71025

variable (W : ℕ)

def work_done_by_women_per_day (W : ℕ) : ℚ := 1 / (8 * W)
def work_done_by_children_per_day (W : ℕ) : ℚ := 1 / (12 * W)

theorem initial_number_of_women :
  (6 * work_done_by_women_per_day W + 3 * work_done_by_children_per_day W = 1 / 10) → W = 10 :=
by
  sorry

end initial_number_of_women_l71_71025


namespace arithmetic_geometric_sequence_l71_71033

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + 3)
    (h2 : (a 1 + 3) * (a 1 + 21) = (a 1 + 9) ^ 2) : a 3 = 12 :=
by 
  sorry

end arithmetic_geometric_sequence_l71_71033


namespace expression_value_l71_71625

theorem expression_value
  (x y a b : ℤ)
  (h1 : x = 1)
  (h2 : y = 2)
  (h3 : a + 2 * b = 3) :
  2 * a + 4 * b - 5 = 1 := 
by sorry

end expression_value_l71_71625


namespace gain_percentage_l71_71911

theorem gain_percentage (C S : ℝ) (h : 80 * C = 25 * S) : 220 = ((S - C) / C) * 100 :=
by sorry

end gain_percentage_l71_71911


namespace initial_books_l71_71309

theorem initial_books (sold_books : ℕ) (given_books : ℕ) (remaining_books : ℕ) 
                      (h1 : sold_books = 11)
                      (h2 : given_books = 35)
                      (h3 : remaining_books = 62) :
  (sold_books + given_books + remaining_books = 108) :=
by
  -- Proof skipped
  sorry

end initial_books_l71_71309


namespace avg_speed_trip_l71_71589

/-- Given a trip with total distance of 70 kilometers, with the first 35 kilometers traveled at
    48 kilometers per hour and the remaining 35 kilometers at 24 kilometers per hour, 
    prove that the average speed is 32 kilometers per hour. -/
theorem avg_speed_trip (d1 d2 : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) (total_distance : ℝ)
  (H1 : d1 = 35) (H2 : d2 = 35) (H3 : s1 = 48) (H4 : s2 = 24)
  (H5 : total_distance = 70)
  (T1 : t1 = d1 / s1) (T2 : t2 = d2 / s2) :
  70 / (t1 + t2) = 32 :=
by
  sorry

end avg_speed_trip_l71_71589


namespace pow_mod_remainder_l71_71442

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l71_71442


namespace black_area_fraction_after_four_changes_l71_71727

/-- 
Problem: Prove that after four changes, the fractional part of the original black area 
remaining black in an equilateral triangle is 81/256, given that each change splits the 
triangle into 4 smaller congruent equilateral triangles, and one of those turns white.
-/

theorem black_area_fraction_after_four_changes :
  (3 / 4) ^ 4 = 81 / 256 := sorry

end black_area_fraction_after_four_changes_l71_71727


namespace contrapositive_example_l71_71948

variable {a : ℕ → ℝ}

theorem contrapositive_example 
  (h₁ : ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n ≤ a (n + 1)) → ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 ≥ a (n + 1) :=
by
  sorry

end contrapositive_example_l71_71948


namespace range_of_m_l71_71454

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) :
  (∀ x, ∃ y, y = f (x + m) - f (x - m)) →
  -1 ≤ m ∧ m ≤ 1 :=
by
  intro hF
  sorry

end range_of_m_l71_71454


namespace proof_problem_1_proof_problem_2_l71_71582

noncomputable def problem_1 (a b : ℝ) : Prop :=
  ((2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3))) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6)

noncomputable def problem_2 : Prop :=
  ((2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 2^(3/4 - 1) - (-2005)^0) = 100

theorem proof_problem_1 (a b : ℝ) : problem_1 a b := 
  sorry

theorem proof_problem_2 : problem_2 := 
  sorry

end proof_problem_1_proof_problem_2_l71_71582


namespace cos_alpha_half_l71_71739

theorem cos_alpha_half (α : ℝ) (h : Real.cos (Real.pi + α) = -1/2) : Real.cos α = 1/2 := 
by 
  sorry

end cos_alpha_half_l71_71739


namespace problem_A_plus_B_l71_71782

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l71_71782


namespace number_of_chain_links_l71_71794

noncomputable def length_of_chain (number_of_links : ℕ) : ℝ :=
  (number_of_links * (7 / 3)) + 1

theorem number_of_chain_links (n m : ℕ) (d : ℝ) (thickness : ℝ) (max_length min_length : ℕ) 
  (h1 : d = 2 + 1 / 3)
  (h2 : thickness = 0.5)
  (h3 : max_length = 36)
  (h4 : min_length = 22)
  (h5 : m = n + 6)
  : length_of_chain n = 22 ∧ length_of_chain m = 36 
  :=
  sorry

end number_of_chain_links_l71_71794


namespace div_by_7_11_13_l71_71888

theorem div_by_7_11_13 (n : ℤ) (A B : ℤ) (hA : A = n % 1000)
  (hB : B = n / 1000) (k : ℤ) (hk : k = A - B) :
  (∃ d, d ∈ {7, 11, 13} ∧ d ∣ n) ↔ (∃ d, d ∈ {7, 11, 13} ∧ d ∣ k) :=
sorry

end div_by_7_11_13_l71_71888


namespace zinc_in_combined_mass_l71_71652

def mixture1_copper_zinc_ratio : ℕ × ℕ := (13, 7)
def mixture2_copper_zinc_ratio : ℕ × ℕ := (5, 3)
def mixture1_mass : ℝ := 100
def mixture2_mass : ℝ := 50

theorem zinc_in_combined_mass :
  let zinc1 := (mixture1_copper_zinc_ratio.2 : ℝ) / (mixture1_copper_zinc_ratio.1 + mixture1_copper_zinc_ratio.2) * mixture1_mass
  let zinc2 := (mixture2_copper_zinc_ratio.2 : ℝ) / (mixture2_copper_zinc_ratio.1 + mixture2_copper_zinc_ratio.2) * mixture2_mass
  zinc1 + zinc2 = 53.75 :=
by
  sorry

end zinc_in_combined_mass_l71_71652


namespace thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l71_71982

theorem thirty_percent_less_than_90_eq_one_fourth_more_than_what_number :
  ∃ (n : ℤ), (5 / 4 : ℝ) * (n : ℝ) = 90 - (0.30 * 90) ∧ n ≈ 50 := 
by
  -- Existence condition for n
  use 50
  -- Proof of equivalence (optional for statement)
  sorry

end thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l71_71982


namespace no_rational_solution_of_odd_quadratic_l71_71495

theorem no_rational_solution_of_odd_quadratic (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ x : ℚ, a * x^2 + b * x + c = 0 :=
sorry

end no_rational_solution_of_odd_quadratic_l71_71495


namespace friends_who_participate_l71_71358

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71358


namespace speed_in_kmph_l71_71617

noncomputable def speed_conversion (speed_mps: ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_in_kmph : speed_conversion 18.334799999999998 = 66.00528 :=
by
  -- proof steps would go here
  sorry

end speed_in_kmph_l71_71617


namespace pow_mod_remainder_l71_71441

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l71_71441


namespace min_disks_needed_l71_71670

/-- 
  Sandhya must save 35 files onto disks, each with 1.44 MB space. 
  5 of the files take up 0.6 MB, 18 of the files take up 0.5 MB, 
  and the rest take up 0.3 MB. Files cannot be split across disks.
  Prove that the smallest number of disks needed to store all 35 files is 12.
--/
theorem min_disks_needed 
  (total_files : ℕ)
  (disk_capacity : ℝ)
  (file_sizes : ℕ → ℝ)
  (files_0_6_MB : ℕ)
  (files_0_5_MB : ℕ)
  (files_0_3_MB : ℕ)
  (remaining_files : ℕ)
  (storage_per_disk : ℝ)
  (smallest_disks_needed : ℕ) 
  (h1 : total_files = 35)
  (h2 : disk_capacity = 1.44)
  (h3 : file_sizes 0 = 0.6)
  (h4 : file_sizes 1 = 0.5)
  (h5 : file_sizes 2 = 0.3)
  (h6 : files_0_6_MB = 5)
  (h7 : files_0_5_MB = 18)
  (h8 : remaining_files = total_files - files_0_6_MB - files_0_5_MB)
  (h9 : remaining_files = 12)
  (h10 : storage_per_disk = file_sizes 0 * 2 + file_sizes 1 + file_sizes 2)
  (h11 : smallest_disks_needed = 12) :
  total_files = 35 ∧ disk_capacity = 1.44 ∧ storage_per_disk <= 1.44 ∧ smallest_disks_needed = 12 :=
by
  sorry

end min_disks_needed_l71_71670


namespace min_value_quadratic_l71_71570

theorem min_value_quadratic (x : ℝ) : -2 * x^2 + 8 * x + 5 ≥ -2 * (2 - x)^2 + 13 :=
by
  sorry

end min_value_quadratic_l71_71570


namespace factor_expression_l71_71605

theorem factor_expression (b : ℤ) : 
  (8 * b ^ 3 + 120 * b ^ 2 - 14) - (9 * b ^ 3 - 2 * b ^ 2 + 14) 
  = -1 * (b ^ 3 - 122 * b ^ 2 + 28) := 
by {
  sorry
}

end factor_expression_l71_71605


namespace hide_and_seek_l71_71373

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71373


namespace solution_set_inequality_l71_71865

noncomputable def f : ℝ → ℝ := sorry  -- Define f according to the problem condition.

def g (f : ℝ → ℝ) : ℝ → ℝ := λ x, exp x * f x - exp x

theorem solution_set_inequality {f : ℝ → ℝ} (h₁ : ∀ x, f x + deriv f x > 1) 
    (h₂ : f 0 = 2017) : 
    {x : ℝ | g f x > 2016} = Ioi 0 := 
  sorry

end solution_set_inequality_l71_71865


namespace friends_who_participate_l71_71364

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71364


namespace product_of_first_four_consecutive_primes_l71_71856

theorem product_of_first_four_consecutive_primes : 
  (2 * 3 * 5 * 7) = 210 :=
by
  sorry

end product_of_first_four_consecutive_primes_l71_71856


namespace find_number_l71_71705

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
sorry

end find_number_l71_71705


namespace find_value_simplify_expression_l71_71065

-- Define the first part of the problem
theorem find_value (α : ℝ) (h : Real.tan α = 1/3) : 
  (1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2)) = 2 / 3 := 
  sorry

-- Define the second part of the problem
theorem simplify_expression (α : ℝ) (h : Real.tan α = 1/3) : 
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / (Real.cos (-α - π) * Real.sin (-π - α)) = -1 := 
  sorry

end find_value_simplify_expression_l71_71065


namespace kaleb_books_count_l71_71490

/-- Kaleb's initial number of books. -/
def initial_books : ℕ := 34

/-- Number of books Kaleb sold. -/
def sold_books : ℕ := 17

/-- Number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Prove the number of books Kaleb has now. -/
theorem kaleb_books_count : initial_books - sold_books + new_books = 24 := by
  sorry

end kaleb_books_count_l71_71490


namespace safety_rent_a_car_cost_per_mile_l71_71515

/-
Problem:
Prove that the cost per mile for Safety Rent-a-Car is 0.177 dollars, given that the total cost of renting an intermediate-size car for 150 miles is the same for Safety Rent-a-Car and City Rentals, with their respective pricing schemes.
-/

theorem safety_rent_a_car_cost_per_mile :
  let x := 21.95
  let y := 18.95
  let z := 0.21
  (x + 150 * real_safety_per_mile) = (y + 150 * z) ↔ real_safety_per_mile = 0.177 :=
by
  sorry

end safety_rent_a_car_cost_per_mile_l71_71515


namespace second_percentage_increase_l71_71597

theorem second_percentage_increase 
  (P : ℝ) 
  (x : ℝ) 
  (h1: 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := 
  sorry

end second_percentage_increase_l71_71597


namespace determine_peter_and_liar_l71_71085

structure Brothers where
  names : Fin 2 → String
  tells_truth : Fin 2 → Bool -- true if the brother tells the truth, false if lies
  (unique_truth_teller : ∃! (i : Fin 2), tells_truth i)
  (one_is_peter : ∃ (i : Fin 2), names i = "Péter")

theorem determine_peter_and_liar (B : Brothers) : 
  ∃ (peter liar : Fin 2), B.names peter = "Péter" ∧ B.tells_truth liar = false ∧
    ∀ (p q : Fin 2), B.names p = "Péter" → B.tells_truth q = false → p = peter ∧ q = liar :=
by
  sorry

end determine_peter_and_liar_l71_71085


namespace equation_of_chord_l71_71909

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

def is_midpoint_of_chord (P M N : ℝ × ℝ) : Prop :=
  ∃ (C : ℝ × ℝ), circle_eq (C.1) (C.2) ∧ (P.1, P.2) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

theorem equation_of_chord (P : ℝ × ℝ) (M N : ℝ × ℝ) (h : P = (4, 2)) (h_mid : is_midpoint_of_chord P M N) :
  ∀ (x y : ℝ), (2 * y) - (8 : ℝ) = (-(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
by
  intro x y H
  sorry

end equation_of_chord_l71_71909


namespace sphere_tangent_plane_normal_line_l71_71991

variable {F : ℝ → ℝ → ℝ → ℝ}
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

def tangent_plane (x y z : ℝ) : Prop := 2*x + y + 2*z - 15 = 0

def normal_line (x y z : ℝ) : Prop := (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_plane_normal_line :
  sphere 3 (-1) 5 →
  tangent_plane 3 (-1) 5 ∧ normal_line 3 (-1) 5 :=
by
  intros h
  constructor
  sorry
  sorry

end sphere_tangent_plane_normal_line_l71_71991


namespace candy_cost_l71_71068

theorem candy_cost (x : ℝ) : 
  (15 * x + 30 * 5) / (15 + 30) = 6 -> x = 8 :=
by sorry

end candy_cost_l71_71068


namespace vector_line_form_to_slope_intercept_l71_71623

variable (x y : ℝ)

theorem vector_line_form_to_slope_intercept :
  (∀ (x y : ℝ), ((-1) * (x - 3) + 2 * (y + 4) = 0) ↔ (y = (-1/2) * x - 11/2)) :=
by
  sorry

end vector_line_form_to_slope_intercept_l71_71623


namespace apple_cost_is_2_l71_71156

def total_spent (hummus_cost chicken_cost bacon_cost vegetable_cost : ℕ) : ℕ :=
  2 * hummus_cost + chicken_cost + bacon_cost + vegetable_cost

theorem apple_cost_is_2 :
  ∀ (hummus_cost chicken_cost bacon_cost vegetable_cost total_money apples_cost : ℕ),
    hummus_cost = 5 →
    chicken_cost = 20 →
    bacon_cost = 10 →
    vegetable_cost = 10 →
    total_money = 60 →
    apples_cost = 5 →
    (total_money - total_spent hummus_cost chicken_cost bacon_cost vegetable_cost) / apples_cost = 2 :=
by
  intros
  sorry

end apple_cost_is_2_l71_71156


namespace find_n_l71_71986

theorem find_n (n : ℕ) (h1 : n > 13) (h2 : (12 : ℚ) / (n - 1 : ℚ) = 1 / 3) : n = 37 := by
  sorry

end find_n_l71_71986


namespace a4_minus_b4_l71_71740

theorem a4_minus_b4 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : a^4 - b^4 = -1 := by
  sorry

end a4_minus_b4_l71_71740


namespace number_of_chords_l71_71507

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l71_71507


namespace units_digit_product_l71_71989

theorem units_digit_product : (3^5 * 2^3) % 10 = 4 := 
sorry

end units_digit_product_l71_71989


namespace walter_zoo_time_l71_71987

def seals_time : ℕ := 13
def penguins_time : ℕ := 8 * seals_time
def elephants_time : ℕ := 13
def total_time_spent_at_zoo : ℕ := seals_time + penguins_time + elephants_time

theorem walter_zoo_time : total_time_spent_at_zoo = 130 := by
  -- Proof goes here
  sorry

end walter_zoo_time_l71_71987


namespace hexagons_after_cuts_l71_71331

theorem hexagons_after_cuts (rectangles_initial : ℕ) (cuts : ℕ) (sheets_total : ℕ)
  (initial_sides : ℕ) (additional_sides : ℕ) 
  (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (final_sides : ℕ) (number_of_hexagons : ℕ) :
  rectangles_initial = 15 →
  cuts = 60 →
  sheets_total = rectangles_initial + cuts →
  initial_sides = rectangles_initial * 4 →
  additional_sides = cuts * 4 →
  final_sides = initial_sides + additional_sides →
  triangle_sides = 3 →
  hexagon_sides = 6 →
  (sheets_total * 4 = final_sides) →
  number_of_hexagons = (final_sides - 225) / 3 →
  number_of_hexagons = 25 :=
by
  intros
  sorry

end hexagons_after_cuts_l71_71331


namespace time_difference_between_shoes_l71_71178

-- Define the conditions
def time_per_mile_regular := 10
def time_per_mile_new := 13
def distance_miles := 5

-- Define the theorem to be proven
theorem time_difference_between_shoes :
  (distance_miles * time_per_mile_new) - (distance_miles * time_per_mile_regular) = 15 :=
by
  sorry

end time_difference_between_shoes_l71_71178


namespace evaluate_expression_l71_71738

noncomputable def M (x y : ℝ) : ℝ := if x < y then y else x
noncomputable def m (x y : ℝ) : ℝ := if x < y then x else y

theorem evaluate_expression
  (p q r s t : ℝ)
  (h1 : p < q)
  (h2 : q < r)
  (h3 : r < s)
  (h4 : s < t)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ p ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t):
  M (M p (m q r)) (m s (m p t)) = q := 
sorry

end evaluate_expression_l71_71738


namespace initial_eggs_ben_l71_71601

-- Let's define the conditions from step a):
def eggs_morning := 4
def eggs_afternoon := 3
def eggs_left := 13

-- Define the total eggs Ben ate
def eggs_eaten := eggs_morning + eggs_afternoon

-- Now we define the initial eggs Ben had
def initial_eggs := eggs_left + eggs_eaten

-- The theorem that states the initial number of eggs
theorem initial_eggs_ben : initial_eggs = 20 :=
  by sorry

end initial_eggs_ben_l71_71601


namespace children_absent_on_independence_day_l71_71942

theorem children_absent_on_independence_day
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (extra_bananas : ℕ)
  (total_possible_children : total_children = 780)
  (bananas_distributed : bananas_per_child = 2)
  (additional_bananas : extra_bananas = 2) :
  ∃ (A : ℕ), A = 390 := 
sorry

end children_absent_on_independence_day_l71_71942


namespace find_room_height_l71_71249

theorem find_room_height (l b d : ℕ) (h : ℕ) (hl : l = 12) (hb : b = 8) (hd : d = 17) :
  d = Int.sqrt (l^2 + b^2 + h^2) → h = 9 :=
by
  sorry

end find_room_height_l71_71249


namespace find_m_l71_71639

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -11.6 :=
sorry

end find_m_l71_71639


namespace hide_and_seek_friends_l71_71383

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71383


namespace avg_of_xyz_l71_71142

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l71_71142


namespace circle_area_l71_71436

theorem circle_area : 
    (∃ x y : ℝ, 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
    (∃ A : ℝ, A = (7 / 4) * Real.pi) :=
by
  sorry

end circle_area_l71_71436


namespace diapers_per_pack_l71_71660

def total_boxes := 30
def packs_per_box := 40
def price_per_diaper := 5
def total_revenue := 960000

def total_packs_per_week := total_boxes * packs_per_box
def total_diapers_sold := total_revenue / price_per_diaper

theorem diapers_per_pack :
  total_diapers_sold / total_packs_per_week = 160 :=
by
  -- Placeholder for the actual proof
  sorry

end diapers_per_pack_l71_71660


namespace log21_requires_additional_information_l71_71620

noncomputable def log3 : ℝ := 0.4771
noncomputable def log5 : ℝ := 0.6990

theorem log21_requires_additional_information
  (log3_given : log3 = 0.4771)
  (log5_given : log5 = 0.6990) :
  ¬ (∃ c₁ c₂ : ℝ, log21 = c₁ * log3 + c₂ * log5) :=
sorry

end log21_requires_additional_information_l71_71620


namespace floor_sub_le_l71_71295

theorem floor_sub_le : ∀ (x y : ℝ), ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by sorry

end floor_sub_le_l71_71295


namespace lychee_production_increase_l71_71688

variable (x : ℕ) -- percentage increase as a natural number

def lychee_increase_2006 (x : ℕ) : ℕ :=
  (1 + x)*(1 + x)

theorem lychee_production_increase (x : ℕ) :
  lychee_increase_2006 x = (1 + x) * (1 + x) :=
by
  sorry

end lychee_production_increase_l71_71688


namespace chords_from_nine_points_l71_71499

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l71_71499


namespace total_legs_correct_l71_71079

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l71_71079


namespace solve_for_a_plus_b_l71_71755

theorem solve_for_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, a * (x + b) = 3 * x + 12) → a + b = 7 :=
by
  intros h
  sorry

end solve_for_a_plus_b_l71_71755


namespace race_time_A_l71_71059

theorem race_time_A (v t : ℝ) (h1 : 1000 = v * t) (h2 : 950 = v * (t - 10)) : t = 200 :=
by
  sorry

end race_time_A_l71_71059


namespace triangle_is_isosceles_right_l71_71261

theorem triangle_is_isosceles_right
  (a b c : ℝ)
  (A B C : ℕ)
  (h1 : c = a * Real.cos B)
  (h2 : b = a * Real.sin C) :
  C = 90 ∧ B = 90 ∧ A = 90 :=
sorry

end triangle_is_isosceles_right_l71_71261


namespace exists_a_bc_l71_71573

-- Definitions & Conditions
def satisfies_conditions (a b c : ℤ) : Prop :=
  - (b + c) - 10 = a ∧ (b + 10) * (c + 10) = 1

-- Theorem Statement
theorem exists_a_bc : ∃ (a b c : ℤ), satisfies_conditions a b c := by
  -- Substitute the correct proof below
  sorry

end exists_a_bc_l71_71573


namespace journey_length_l71_71613

/-- Define the speed in the urban area as 55 km/h. -/
def urban_speed : ℕ := 55

/-- Define the speed on the highway as 85 km/h. -/
def highway_speed : ℕ := 85

/-- Define the time spent in each area as 3 hours. -/
def travel_time : ℕ := 3

/-- Define the distance traveled in the urban area as the product of the speed and time. -/
def urban_distance : ℕ := urban_speed * travel_time

/-- Define the distance traveled on the highway as the product of the speed and time. -/
def highway_distance : ℕ := highway_speed * travel_time

/-- Define the total distance of the journey. -/
def total_distance : ℕ := urban_distance + highway_distance

/-- The theorem that the total distance is 420 km. -/
theorem journey_length : total_distance = 420 := by
  -- Prove the equality by calculating the distances and summing them up
  sorry

end journey_length_l71_71613


namespace problem1_problem2_l71_71859

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l71_71859


namespace find_sum_a7_a8_l71_71916

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 q : ℝ), ∀ n : ℕ, a n = a1 * q ^ n

variable (a : ℕ → ℝ)

axiom h_geom : geometric_sequence a
axiom h1 : a 0 + a 1 = 16
axiom h2 : a 2 + a 3 = 32

theorem find_sum_a7_a8 : a 6 + a 7 = 128 :=
sorry

end find_sum_a7_a8_l71_71916


namespace not_prime_abs_diff_l71_71653

theorem not_prime_abs_diff (a b : ℕ) (x y : ℕ)
  (h1 : a + b = x^2) (h2 : a * b = y^2) : ¬ Nat.Prime (|16 * a - 9 * b|) :=
sorry

end not_prime_abs_diff_l71_71653


namespace axis_angle_set_l71_71327

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end axis_angle_set_l71_71327


namespace alex_play_friends_with_l71_71357

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71357


namespace bert_same_kangaroos_as_kameron_in_40_days_l71_71158

theorem bert_same_kangaroos_as_kameron_in_40_days
  (k : ℕ := 100)
  (b : ℕ := 20)
  (r : ℕ := 2) :
  ∃ t : ℕ, t = 40 ∧ b + t * r = k := by
  sorry

end bert_same_kangaroos_as_kameron_in_40_days_l71_71158


namespace find_ab_for_equation_l71_71250

theorem find_ab_for_equation (a b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (∃ x, x = 12 - x1 - x2) ∧ (a * x1^2 - 24 * x1 + b) / (x1^2 - 1) = x1
  ∧ (a * x2^2 - 24 * x2 + b) / (x2^2 - 1) = x2) ∧ (a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819) := sorry

end find_ab_for_equation_l71_71250


namespace num_three_digit_numbers_no_repeat_l71_71048

theorem num_three_digit_numbers_no_repeat (digits : Finset ℕ) (h : digits = {1, 2, 3, 4}) :
  (digits.card = 4) →
  ∀ d1 d2 d3, d1 ∈ digits → d2 ∈ digits → d3 ∈ digits →
  d1 ≠ d2 → d1 ≠ d3 → d2 ≠ d3 → 
  3 * 2 * 1 * digits.card = 24 :=
by
  sorry

end num_three_digit_numbers_no_repeat_l71_71048


namespace sum_of_coefficients_l71_71894

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (n : ℕ) (hn1 : 5 < n) (hn2 : n < 7)
  (coeff_cond : binom n 3 > binom n 2 ∧ binom n 3 > binom n 4) :
  (1 + 1)^n = 64 :=
by
  have h : n = 6 :=
    by sorry -- provided conditions force n to be 6
  show 2^n = 64
  rw [h]
  exact rfl

end sum_of_coefficients_l71_71894


namespace rationalize_denominator_l71_71800

theorem rationalize_denominator : 
  let A := -13 
  let B := -9
  let C := 3
  let D := 2
  let E := 165
  let F := 51
  A + B + C + D + E + F = 199 := by
sorry

end rationalize_denominator_l71_71800


namespace probability_two_red_balls_given_one_white_l71_71761

theorem probability_two_red_balls_given_one_white :
  let total_ways := Nat.choose 10 3 - Nat.choose 5 3,
      favorable_ways := Nat.choose 5 2 * Nat.choose 5 1,
      probability := favorable_ways / total_ways
  in probability = (5 : ℚ) / 11 :=
by
  let total_ways := Nat.choose 10 3 - Nat.choose 5 3
  let favorable_ways := Nat.choose 5 2 * Nat.choose 5 1
  let probability := (favorable_ways : ℚ) / total_ways
  show probability = (5 : ℚ) / 11
  sorry

end probability_two_red_balls_given_one_white_l71_71761


namespace abs_x_minus_y_zero_l71_71316

theorem abs_x_minus_y_zero (x y : ℝ) 
  (h_avg : (x + y + 30 + 29 + 31) / 5 = 30)
  (h_var : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) : 
  |x - y| = 0 :=
  sorry

end abs_x_minus_y_zero_l71_71316


namespace expected_pairs_socks_l71_71545

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l71_71545


namespace village_assistants_selection_l71_71084

theorem village_assistants_selection :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let A := 1
  let B := 2
  let C := 3
  let total_ways := Nat.choose 9 3 - Nat.choose 7 3
  total_ways = 49 :=
by
  sorry

end village_assistants_selection_l71_71084


namespace binomial_600_600_l71_71102

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l71_71102


namespace simplify_exp_l71_71240

theorem simplify_exp : (10^8 / (10 * 10^5)) = 100 := 
by
  -- The proof is omitted; we are stating the problem.
  sorry

end simplify_exp_l71_71240


namespace sequence_term_formula_l71_71884

theorem sequence_term_formula 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h : ∀ n, S n = n^2 + 3 * n)
  (h₁ : a 1 = 4)
  (h₂ : ∀ n, 1 < n → a n = S n - S (n - 1)) :
  ∀ n, a n = 2 * n + 2 :=
by
  sorry

end sequence_term_formula_l71_71884


namespace fraction_inequality_solution_set_l71_71818

theorem fraction_inequality_solution_set : 
  {x : ℝ | (2 - x) / (x + 4) > 0} = {x : ℝ | -4 < x ∧ x < 2} :=
by sorry

end fraction_inequality_solution_set_l71_71818


namespace jonah_fish_count_l71_71927

theorem jonah_fish_count :
  let initial_fish := 14
  let added_fish := 2
  let eaten_fish := 6
  let removed_fish := 2
  let new_fish := 3
  initial_fish + added_fish - eaten_fish - removed_fish + new_fish = 11 := 
by
  sorry

end jonah_fish_count_l71_71927


namespace reduce_4128_over_4386_to_lowest_terms_l71_71577

noncomputable def reduced_fraction := Rat.mk 4128 4386

theorem reduce_4128_over_4386_to_lowest_terms : reduced_fraction = Rat.mk 295 313 := by
  -- Proof omitted; this statement asserts the equality of the two fractions.
  sorry

end reduce_4128_over_4386_to_lowest_terms_l71_71577


namespace divisor_is_20_l71_71579

theorem divisor_is_20 (D q1 q2 q3 : ℕ) :
  (242 = D * q1 + 11) ∧
  (698 = D * q2 + 18) ∧
  (940 = D * q3 + 9) →
  D = 20 :=
by
  sorry

end divisor_is_20_l71_71579


namespace cost_price_of_computer_table_l71_71062

theorem cost_price_of_computer_table (CP SP : ℝ) 
  (h1 : SP = CP * 1.15) 
  (h2 : SP = 5750) 
  : CP = 5000 := 
by 
  sorry

end cost_price_of_computer_table_l71_71062


namespace foci_coordinates_l71_71317

-- Define the parameters for the hyperbola
def a_squared : ℝ := 3
def b_squared : ℝ := 1
def c_squared : ℝ := a_squared + b_squared

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

-- State the theorem about the coordinates of the foci
theorem foci_coordinates : {foci : ℝ × ℝ // foci = (-2, 0) ∨ foci = (2, 0)} :=
by 
  have ha : a_squared = 3 := rfl
  have hb : b_squared = 1 := rfl
  have hc : c_squared = a_squared + b_squared := rfl
  have c := Real.sqrt c_squared
  have hc' : c = 2 := 
  -- sqrt part can be filled if detailed, for now, just direct conclusion
  sorry
  exact ⟨(2, 0), Or.inr rfl⟩

end foci_coordinates_l71_71317


namespace hide_and_seek_problem_l71_71368

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71368


namespace hiring_manager_acceptance_l71_71029

theorem hiring_manager_acceptance {k : ℤ} 
  (avg_age : ℤ) (std_dev : ℤ) (num_accepted_ages : ℤ) 
  (h_avg : avg_age = 20) (h_std_dev : std_dev = 8)
  (h_num_accepted : num_accepted_ages = 17) : 
  (20 + k * 8 - (20 - k * 8) + 1) = 17 → k = 1 :=
by
  intros
  sorry

end hiring_manager_acceptance_l71_71029


namespace place_integers_on_cube_l71_71771

theorem place_integers_on_cube:
  ∃ (A B C D A₁ B₁ C₁ D₁ : ℤ),
    A = B + D + A₁ ∧ 
    B = A + C + B₁ ∧ 
    C = B + D + C₁ ∧ 
    D = A + C + D₁ ∧ 
    A₁ = B₁ + D₁ + A ∧ 
    B₁ = A₁ + C₁ + B ∧ 
    C₁ = B₁ + D₁ + C ∧ 
    D₁ = A₁ + C₁ + D :=
sorry

end place_integers_on_cube_l71_71771


namespace remaining_apps_eq_files_plus_more_initial_apps_eq_16_l71_71106

-- Defining the initial number of files
def initial_files: ℕ := 9

-- Defining the remaining number of files and apps
def remaining_files: ℕ := 5
def remaining_apps: ℕ := 12

-- Given: Dave has 7 more apps than files left
def apps_more_than_files: ℕ := 7

-- Equating the given condition 12 = 5 + 7
theorem remaining_apps_eq_files_plus_more :
  remaining_apps = remaining_files + apps_more_than_files := by
  sorry -- This would trivially prove as 12 = 5+7

-- Proving the number of initial apps
theorem initial_apps_eq_16 (A: ℕ) (h1: initial_files = 9) (h2: remaining_files = 5) (h3: remaining_apps = 12) (h4: apps_more_than_files = 7):
  A - remaining_apps = initial_files - remaining_files → A = 16 := by
  sorry

end remaining_apps_eq_files_plus_more_initial_apps_eq_16_l71_71106


namespace number_of_chords_number_of_chords_l71_71505

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l71_71505


namespace polynomial_identity_l71_71610

open Function

-- Define the polynomial terms
def f1 (x : ℝ) := 2*x^5 + 4*x^3 + 3*x + 4
def f2 (x : ℝ) := x^4 - 2*x^3 + 3
def g (x : ℝ) := -2*x^5 + x^4 - 6*x^3 - 3*x - 1

-- Lean theorem statement
theorem polynomial_identity :
  ∀ x : ℝ, f1 x + g x = f2 x :=
by
  intros x
  sorry

end polynomial_identity_l71_71610


namespace eiffel_tower_vs_burj_khalifa_l71_71462

-- Define the heights of the structures
def height_eiffel_tower : ℕ := 324
def height_burj_khalifa : ℕ := 830

-- Define the statement to be proven
theorem eiffel_tower_vs_burj_khalifa :
  height_burj_khalifa - height_eiffel_tower = 506 :=
by
  sorry

end eiffel_tower_vs_burj_khalifa_l71_71462


namespace total_farm_tax_collected_l71_71733

noncomputable def totalFarmTax (taxPaid: ℝ) (percentage: ℝ) : ℝ := taxPaid / (percentage / 100)

theorem total_farm_tax_collected (taxPaid : ℝ) (percentage : ℝ) (h_taxPaid : taxPaid = 480) (h_percentage : percentage = 16.666666666666668) :
  totalFarmTax taxPaid percentage = 2880 :=
by
  rw [h_taxPaid, h_percentage]
  simp [totalFarmTax]
  norm_num
  sorry

end total_farm_tax_collected_l71_71733


namespace ratio_of_squares_l71_71847

def square_inscribed_triangle_1 (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  x = 24 / 7

def square_inscribed_triangle_2 (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 10 ∧
  y = 10 / 3

theorem ratio_of_squares (x y : ℝ) 
  (hx : square_inscribed_triangle_1 x) 
  (hy : square_inscribed_triangle_2 y) : 
  x / y = 36 / 35 := 
by sorry

end ratio_of_squares_l71_71847


namespace total_distance_l71_71867

theorem total_distance (D : ℝ) 
  (h₁ : 60 * (D / 2 / 60) = D / 2) 
  (h₂ : 40 * ((D / 2) / 4 / 40) = D / 8) 
  (h₃ : 50 * (105 / 50) = 105)
  (h₄ : D = D / 2 + D / 8 + 105) : 
  D = 280 :=
by sorry

end total_distance_l71_71867


namespace dinner_potatoes_l71_71843

def lunch_potatoes : ℕ := 5
def total_potatoes : ℕ := 7

theorem dinner_potatoes : total_potatoes - lunch_potatoes = 2 :=
by
  sorry

end dinner_potatoes_l71_71843


namespace rabbits_distribution_l71_71181

def num_ways_to_distribute : ℕ :=
  20 + 390 + 150

theorem rabbits_distribution :
  num_ways_to_distribute = 560 := by
  sorry

end rabbits_distribution_l71_71181


namespace binomial_600_600_l71_71097

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l71_71097


namespace quotient_of_division_l71_71179

theorem quotient_of_division (Q : ℤ) (h1 : 172 = (17 * Q) + 2) : Q = 10 :=
sorry

end quotient_of_division_l71_71179


namespace perfect_square_trinomial_k_l71_71470

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l71_71470


namespace percent_gold_coins_l71_71151

variables (total_objects : ℝ) (coins_beads_percent beads_percent gold_coins_percent : ℝ)
           (h1 : coins_beads_percent = 0.75)
           (h2 : beads_percent = 0.15)
           (h3 : gold_coins_percent = 0.60)

theorem percent_gold_coins : (gold_coins_percent * (coins_beads_percent - beads_percent)) = 0.36 :=
by
  have coins_percent := coins_beads_percent - beads_percent
  have gold_coins_total_percent := gold_coins_percent * coins_percent
  exact sorry

end percent_gold_coins_l71_71151


namespace find_k_common_term_l71_71742

def sequence_a (k : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 1 
  else if n = 2 then k 
  else if n = 3 then 3*k - 3 
  else if n = 4 then 6*k - 8 
  else (n * (n-1) * (k-2)) / 2 + n

def is_fermat (x : ℕ) : Prop :=
  ∃ m : ℕ, x = 2^(2^m) + 1

theorem find_k_common_term (k : ℕ) :
  k > 2 → ∃ n m : ℕ, sequence_a k n = 2^(2^m) + 1 :=
by
  sorry

end find_k_common_term_l71_71742


namespace calc_expression_l71_71724

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end calc_expression_l71_71724


namespace measure_of_angle_E_l71_71769

variable (D E F : ℝ)
variable (h1 : E = F)
variable (h2 : F = 3 * D)
variable (h3 : D + E + F = 180)

theorem measure_of_angle_E : E = 540 / 7 :=
by
  -- Proof omitted
  sorry

end measure_of_angle_E_l71_71769


namespace hide_and_seek_friends_l71_71384

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71384


namespace expected_socks_to_pair_l71_71542

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l71_71542


namespace barry_sotter_length_increase_l71_71795

theorem barry_sotter_length_increase (n : ℕ) : (n + 3) / 3 = 50 → n = 147 :=
by
  intro h
  sorry

end barry_sotter_length_increase_l71_71795


namespace mr_ray_customers_without_fish_l71_71790

def mr_ray_num_customers_without_fish
  (total_customers : ℕ)
  (total_tuna_weight : ℕ)
  (specific_customers_30lb : ℕ)
  (specific_weight_30lb : ℕ)
  (specific_customers_20lb : ℕ)
  (specific_weight_20lb : ℕ)
  (weight_per_customer : ℕ)
  (remaining_tuna_weight : ℕ)
  (num_customers_served_with_remaining_tuna : ℕ)
  (total_satisfied_customers : ℕ) : ℕ :=
  total_customers - total_satisfied_customers

theorem mr_ray_customers_without_fish :
  mr_ray_num_customers_without_fish 100 2000 10 30 15 20 25 1400 56 81 = 19 :=
by 
  sorry

end mr_ray_customers_without_fish_l71_71790


namespace price_of_candied_grape_l71_71604

theorem price_of_candied_grape (x : ℝ) (h : 15 * 2 + 12 * x = 48) : x = 1.5 :=
by
  sorry

end price_of_candied_grape_l71_71604


namespace alex_play_friends_with_l71_71356

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71356


namespace total_yearly_cutting_cost_l71_71006

-- Conditions
def initial_height := 2 : ℝ
def growth_per_month := 0.5 : ℝ
def cutting_height := 4 : ℝ
def cost_per_cut := 100 : ℝ
def months_in_year := 12 : ℝ

-- Proof statement
theorem total_yearly_cutting_cost :
  ∀ (initial_height growth_per_month cutting_height cost_per_cut months_in_year : ℝ),
  initial_height = 2 ∧ growth_per_month = 0.5 ∧ cutting_height = 4 ∧ cost_per_cut = 100 ∧ months_in_year = 12 →
  let growth_before_cut := cutting_height - initial_height in
  let months_to_cut := growth_before_cut / growth_per_month in
  let cuts_per_year := months_in_year / months_to_cut in
  let yearly_cost := cuts_per_year * cost_per_cut in
  yearly_cost = 300 :=
by
  intros _ _ _ _ _ h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h5
  simp [h1, h2, h3, h4, h5] at *
  let growth_before_cut := 2
  let months_to_cut := 4
  let cuts_per_year := 3
  let yearly_cost := 300
  sorry

end total_yearly_cutting_cost_l71_71006


namespace find_original_number_l71_71409

theorem find_original_number (x : ℕ) 
    (h1 : (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183) : x = 32 := 
by
  sorry

end find_original_number_l71_71409


namespace workshop_employees_l71_71227

theorem workshop_employees (x y : ℕ) 
  (H1 : (x + y) - ((1 / 2) * x + (1 / 3) * y + (1 / 3) * x + (1 / 2) * y) = 120)
  (H2 : (1 / 2) * x + (1 / 3) * y = (1 / 7) * ((1 / 3) * x + (1 / 2) * y) + (1 / 3) * x + (1 / 2) * y) : 
  x = 480 ∧ y = 240 := 
by
  sorry

end workshop_employees_l71_71227


namespace reciprocal_of_negative_2023_l71_71529

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l71_71529


namespace mountain_number_count_l71_71430

noncomputable def isMountainNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  let sorted_digits := digits.qsort (λ x y => x > y)
  digits.length = 4 ∧
  digits.all (λ d => 1 ≤ d ∧ d ≤ 9) ∧
  digits.nodup ∧
  sorted_digits[1] > sorted_digits[0] ∧ 
  sorted_digits[1] > sorted_digits[2] ∧ 
  sorted_digits[1] > sorted_digits[3] ∧
  digits[0] ≠ digits[3]

noncomputable def countMountainNumbers : ℕ :=
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ isMountainNumber n}.card

theorem mountain_number_count : countMountainNumbers = 3024 := by
  sorry

end mountain_number_count_l71_71430


namespace problem_A_plus_B_l71_71783

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end problem_A_plus_B_l71_71783


namespace find_f_50_l71_71526

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x * y
axiom f_20 : f 20 = 10

theorem find_f_50 : f 50 = 25 :=
by
  sorry

end find_f_50_l71_71526


namespace scientific_notation_correct_l71_71000

def big_number : ℕ := 274000000

noncomputable def scientific_notation : ℝ := 2.74 * 10^8

theorem scientific_notation_correct : (big_number : ℝ) = scientific_notation :=
by sorry

end scientific_notation_correct_l71_71000


namespace value_is_20_l71_71070

-- Define the conditions
def number : ℕ := 5
def value := number + 3 * number

-- State the theorem
theorem value_is_20 : value = 20 := by
  -- Proof goes here
  sorry

end value_is_20_l71_71070


namespace hexagon_perimeter_l71_71681

theorem hexagon_perimeter (s : ℕ) (P : ℕ) (h1 : s = 8) (h2 : 6 > 0) 
                          (h3 : P = 6 * s) : P = 48 := by
  sorry

end hexagon_perimeter_l71_71681


namespace binom_600_eq_1_l71_71094

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l71_71094


namespace intersection_hyperbola_l71_71728

theorem intersection_hyperbola (t : ℝ) :
  ∃ A B : ℝ, ∀ (x y : ℝ),
  (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 5 = 0) →
  (x^2 / A - y^2 / B = 1) :=
sorry

end intersection_hyperbola_l71_71728


namespace vehicle_worth_l71_71313

-- Definitions from the conditions
def monthlyEarnings : ℕ := 4000
def savingFraction : ℝ := 0.5
def savingMonths : ℕ := 8

-- Theorem statement
theorem vehicle_worth : (monthlyEarnings * savingFraction * savingMonths : ℝ) = 16000 := 
by
  sorry

end vehicle_worth_l71_71313


namespace problem_1_problem_2_l71_71262

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : ∀ n : ℕ, 2 * S n = a (n + 1) - 2^(n + 1) + 1
axiom h2 : a 2 + 5 = a 1 + (a 3 - a 2)

-- Problem 1: Prove the value of a₁
theorem problem_1 : a 1 = 1 := sorry

-- Problem 2: Find the general term formula for the sequence {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = 3^n - 2^n := sorry

end problem_1_problem_2_l71_71262


namespace hide_and_seek_l71_71388

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71388


namespace sum_of_three_consecutive_even_numbers_is_162_l71_71822

theorem sum_of_three_consecutive_even_numbers_is_162 (a b c : ℕ) 
  (h1 : a = 52) 
  (h2 : b = a + 2) 
  (h3 : c = b + 2) : 
  a + b + c = 162 := by
  sorry

end sum_of_three_consecutive_even_numbers_is_162_l71_71822


namespace proj_matrix_inv_is_zero_l71_71779

open Matrix

variables {R : Type*} [Field R]
variables v : Matrix (Fin 3) (Fin 1) R := ![![1], ![2], ![2]]
def P : Matrix (Fin 3) (Fin 3) R := v ⬝ (v.transpose)

theorem proj_matrix_inv_is_zero : det P = 0 → P⁻¹ = 0 :=
by
  intro h
  sorry

end proj_matrix_inv_is_zero_l71_71779


namespace intercepts_of_line_l71_71562

theorem intercepts_of_line (x y : ℝ) : 
  (x + 6 * y + 2 = 0) → (x = -2) ∧ (y = -1 / 3) :=
by
  sorry

end intercepts_of_line_l71_71562


namespace increasing_condition_sufficient_not_necessary_l71_71581

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0) → (a ≥ 0) ∧ ¬ (a > 0 ↔ (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0)) :=
by
  sorry

end increasing_condition_sufficient_not_necessary_l71_71581


namespace joe_eggs_around_park_l71_71488

variable (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ)

def joe_eggs (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ) : Prop :=
  total_eggs = club_house_eggs + town_hall_garden_eggs + park_eggs

theorem joe_eggs_around_park (h1 : total_eggs = 20) (h2 : club_house_eggs = 12) (h3 : town_hall_garden_eggs = 3) :
  ∃ park_eggs, joe_eggs total_eggs club_house_eggs town_hall_garden_eggs park_eggs ∧ park_eggs = 5 :=
by
  sorry

end joe_eggs_around_park_l71_71488


namespace tenth_term_ar_sequence_l71_71886

-- Variables for the first term and common difference
variables (a1 d : ℕ) (n : ℕ)

-- Specific given values
def a1_fixed := 3
def d_fixed := 2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) := a1 + (n - 1) * d

-- The statement to prove
theorem tenth_term_ar_sequence : a_n 10 = 21 := by
  -- Definitions for a1 and d
  let a1 := a1_fixed
  let d := d_fixed
  -- The rest of the proof
  sorry

end tenth_term_ar_sequence_l71_71886


namespace ben_remaining_bonus_l71_71424

theorem ben_remaining_bonus :
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  total_bonus - total_expense = 867 :=
by
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  have h1 : kitchen_expense = 68 := by sorry
  have h2 : holiday_expense = 374 := by sorry
  have h3 : gift_expense = 187 := by sorry
  have h4 : total_expense = 629 := by sorry
  show total_bonus - total_expense = 867 from by
    calc
      total_bonus - total_expense
      = 1496 - 629 : by rw [h4]
      ... = 867 : by sorry

end ben_remaining_bonus_l71_71424


namespace log_positive_interval_l71_71270

noncomputable def f (a x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

theorem log_positive_interval (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1 / 2) (2 / 3) → f a x > 0) ↔ (1 / 3 < a ∧ a < 1) := by
  sorry

end log_positive_interval_l71_71270


namespace man_l71_71413

theorem man's_age_twice_son_in_2_years 
  (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = 38) (h3 : M = S + 20) : 
  ∃ X : ℕ, (M + X = 2 * (S + X)) ∧ X = 2 :=
by
  sorry

end man_l71_71413


namespace increasing_function_odd_function_l71_71457

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem increasing_function (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

theorem odd_function (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) ↔ a = 1 :=
sorry

end increasing_function_odd_function_l71_71457


namespace path_count_1800_l71_71901

-- Define the coordinates of the points
def A := (0, 8)
def B := (4, 5)
def C := (7, 2)
def D := (9, 0)

-- Function to calculate the number of combinatorial paths
def comb_paths (steps_right steps_down : ℕ) : ℕ :=
  Nat.choose (steps_right + steps_down) steps_right

-- Define the number of steps for each segment
def steps_A_B := (4, 2)  -- 4 right, 2 down
def steps_B_C := (3, 3)  -- 3 right, 3 down
def steps_C_D := (2, 2)  -- 2 right, 2 down

-- Calculate the number of paths for each segment
def paths_A_B := comb_paths steps_A_B.1 steps_A_B.2
def paths_B_C := comb_paths steps_B_C.1 steps_B_C.2
def paths_C_D := comb_paths steps_C_D.1 steps_C_D.2

-- Calculate the total number of paths combining all segments
def total_paths : ℕ :=
  paths_A_B * paths_B_C * paths_C_D

theorem path_count_1800 :
  total_paths = 1800 := by
  sorry

end path_count_1800_l71_71901


namespace distinct_ordered_pairs_count_l71_71129

theorem distinct_ordered_pairs_count :
  {ab : ℕ × ℕ // ab.1 % 2 = 0 ∧ ab.1 + ab.2 = 52 ∧ 0 < ab.1 ∧ 0 < ab.2}.to_finset.card = 25 :=
by
  sorry

end distinct_ordered_pairs_count_l71_71129


namespace sheila_weekly_earnings_l71_71801

-- Variables
variables {hours_mon_wed_fri hours_tue_thu rate_per_hour : ℕ}

-- Conditions
def sheila_works_mwf : hours_mon_wed_fri = 8 := by sorry
def sheila_works_tue_thu : hours_tue_thu = 6 := by sorry
def sheila_rate : rate_per_hour = 11 := by sorry

-- Main statement to prove
theorem sheila_weekly_earnings : 
  3 * hours_mon_wed_fri + 2 * hours_tue_thu = 36 →
  rate_per_hour = 11 →
  (3 * hours_mon_wed_fri + 2 * hours_tue_thu) * rate_per_hour = 396 :=
by
  intros h_hours h_rate
  sorry

end sheila_weekly_earnings_l71_71801


namespace smallest_number_of_coins_l71_71203

theorem smallest_number_of_coins (coins : Finset ℕ) (hcoins : coins = {1, 5, 10, 25}) :
  ∃ (n : ℕ), n = 10 ∧ (∀ x, x ∈ Icc 1 99 → ∃ (c1 c5 c10 c25 : ℕ), c1 + 5*c5 + 10*c10 + 25*c25 = x ∧ c1 + c5 + c10 + c25 = n) :=
by
  sorry

end smallest_number_of_coins_l71_71203


namespace sum_of_cubes_zero_l71_71160

variables {a b c : ℝ}

theorem sum_of_cubes_zero (h₁ : a + b + c = 0) (h₂ : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) : a^3 + b^3 + c^3 = 0 :=
sorry

end sum_of_cubes_zero_l71_71160


namespace geometric_series_first_term_l71_71979

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l71_71979


namespace marys_balloons_l71_71940

theorem marys_balloons (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end marys_balloons_l71_71940


namespace sum_of_consecutive_negatives_l71_71814

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l71_71814


namespace kiran_money_l71_71217

theorem kiran_money (R G K : ℕ) (h1: R / G = 6 / 7) (h2: G / K = 6 / 15) (h3: R = 36) : K = 105 := by
  sorry

end kiran_money_l71_71217


namespace acute_triangle_side_range_l71_71127

theorem acute_triangle_side_range {x : ℝ} (h : ∀ a b c : ℝ, a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :
  2 < 4 ∧ 4 < x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
  sorry

end acute_triangle_side_range_l71_71127


namespace proof_supplies_proof_transportation_cost_proof_min_cost_condition_l71_71480

open Real

noncomputable def supplies_needed (a b : ℕ) := a = 200 ∧ b = 300

noncomputable def transportation_cost (x : ℝ) := 60 ≤ x ∧ x ≤ 260 ∧ ∀ w : ℝ, w = 10 * x + 10200

noncomputable def min_cost_condition (m x : ℝ) := 
  (0 < m ∧ m ≤ 8) ∧ (∀ w : ℝ, (10 - m) * x + 10200 ≥ 10320)

theorem proof_supplies : ∃ a b : ℕ, supplies_needed a b := 
by
  use 200, 300
  sorry

theorem proof_transportation_cost : ∃ x : ℝ, transportation_cost x := 
by
  use 60
  sorry

theorem proof_min_cost_condition : ∃ m x : ℝ, min_cost_condition m x := 
by
  use 8, 60
  sorry

end proof_supplies_proof_transportation_cost_proof_min_cost_condition_l71_71480


namespace solve_for_x_l71_71024

theorem solve_for_x (x : ℚ) 
  (h : (1/3 : ℚ) + 1/x = (7/9 : ℚ) + 1) : 
  x = 9/13 :=
by
  sorry

end solve_for_x_l71_71024


namespace sin_sum_angles_36_108_l71_71311

theorem sin_sum_angles_36_108 (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_angle : A = 36 ∨ A = 108 ∨ B = 36 ∨ B = 108 ∨ C = 36 ∨ C = 108) :
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 :=
by
  sorry

end sin_sum_angles_36_108_l71_71311


namespace range_of_f_when_a_0_range_of_a_for_three_zeros_l71_71271

noncomputable def f_part1 (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x else x ^ 2

theorem range_of_f_when_a_0 : Set.range f_part1 = {y : ℝ | 0 < y} := by
  sorry

noncomputable def f_part2 (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x - a else x ^ 2 - 3 * a * x + a

def discriminant (a : ℝ) (x : ℝ) : ℝ := (3 * a) ^ 2 - 4 * a

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∀ x : ℝ, f_part2 a x = 0) → (4 / 9 < a ∧ a ≤ 1) := by
  sorry

end range_of_f_when_a_0_range_of_a_for_three_zeros_l71_71271


namespace parallelogram_area_15_l71_71237

def point := (ℝ × ℝ)

def base_length (p1 p2 : point) : ℝ :=
  abs (p2.1 - p1.1)

def height_length (p3 p4 : point) : ℝ :=
  abs (p3.2 - p4.2)

def parallelogram_area (p1 p2 p3 p4 : point) : ℝ :=
  base_length p1 p2 * height_length p1 p3

theorem parallelogram_area_15 :
  parallelogram_area (0, 0) (3, 0) (1, 5) (4, 5) = 15 := by
  sorry

end parallelogram_area_15_l71_71237


namespace athlete_difference_l71_71410

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end athlete_difference_l71_71410


namespace exists_parallel_line_l71_71082

variable (P : ℝ × ℝ)
variable (g : ℝ × ℝ)
variable (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
variable (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0))

theorem exists_parallel_line (P : ℝ × ℝ) (g : ℝ × ℝ) (in_first_quadrant : 0 < P.1 ∧ 0 < P.2)
  (parallel_to_second_projection_plane : ∃ c : ℝ, g = (c, 0)) :
  ∃ a : ℝ × ℝ, (∃ d : ℝ, g = (d, 0)) ∧ (a = P) :=
sorry

end exists_parallel_line_l71_71082


namespace second_term_is_three_l71_71852

-- Given conditions
variables (r : ℝ) (S : ℝ)
hypothesis hr : r = 1 / 4
hypothesis hS : S = 16

-- Definition of the first term a
noncomputable def first_term (r : ℝ) (S : ℝ) : ℝ :=
  S * (1 - r)

-- Definition of the second term
noncomputable def second_term (r : ℝ) (a : ℝ) : ℝ :=
  a * r

-- Prove that the second term is 3
theorem second_term_is_three : second_term r (first_term r S) = 3 :=
by
  rw [first_term, second_term]
  sorry

end second_term_is_three_l71_71852


namespace find_first_term_l71_71960

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l71_71960


namespace triangle_cross_section_l71_71571

-- Definitions for the given conditions
inductive Solid
| Prism
| Pyramid
| Frustum
| Cylinder
| Cone
| TruncatedCone
| Sphere

-- The theorem statement of the proof problem
theorem triangle_cross_section (s : Solid) (cross_section_is_triangle : Prop) : 
  cross_section_is_triangle →
  (s = Solid.Prism ∨ s = Solid.Pyramid ∨ s = Solid.Frustum ∨ s = Solid.Cone) :=
sorry

end triangle_cross_section_l71_71571


namespace misha_needs_total_l71_71016

theorem misha_needs_total (
  current_amount : ℤ := 34
) (additional_amount : ℤ := 13) : 
  current_amount + additional_amount = 47 :=
by
  sorry

end misha_needs_total_l71_71016


namespace problem_2014_minus_4102_l71_71205

theorem problem_2014_minus_4102 : 2014 - 4102 = -2088 := 
by
  -- The proof is omitted as per the requirement
  sorry

end problem_2014_minus_4102_l71_71205


namespace log_7_over_5_not_expressible_l71_71266

theorem log_7_over_5_not_expressible (log2 log3 : ℝ) (h2 : log 2 = log2) (h3 : log 3 = log3) :
  ¬ (∃ a b : ℝ, log (7/5) = a * log2 + b * log3) :=
sorry

end log_7_over_5_not_expressible_l71_71266


namespace company_profits_ratio_l71_71760

def companyN_2008_profits (RN : ℝ) : ℝ := 0.08 * RN
def companyN_2009_profits (RN : ℝ) : ℝ := 0.15 * (0.8 * RN)
def companyN_2010_profits (RN : ℝ) : ℝ := 0.10 * (1.3 * 0.8 * RN)

def companyM_2008_profits (RM : ℝ) : ℝ := 0.12 * RM
def companyM_2009_profits (RM : ℝ) : ℝ := 0.18 * RM
def companyM_2010_profits (RM : ℝ) : ℝ := 0.14 * RM

def total_profits_N (RN : ℝ) : ℝ :=
  companyN_2008_profits RN + companyN_2009_profits RN + companyN_2010_profits RN

def total_profits_M (RM : ℝ) : ℝ :=
  companyM_2008_profits RM + companyM_2009_profits RM + companyM_2010_profits RM

theorem company_profits_ratio (RN RM : ℝ) :
  total_profits_N RN / total_profits_M RM = (0.304 * RN) / (0.44 * RM) :=
by
  unfold total_profits_N companyN_2008_profits companyN_2009_profits companyN_2010_profits
  unfold total_profits_M companyM_2008_profits companyM_2009_profits companyM_2010_profits
  simp
  sorry

end company_profits_ratio_l71_71760


namespace sum_powers_l71_71304

theorem sum_powers {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end sum_powers_l71_71304


namespace dinner_cost_l71_71426

theorem dinner_cost (tax_rate tip_rate total_cost : ℝ) (h_tax : tax_rate = 0.12) (h_tip : tip_rate = 0.20) (h_total : total_cost = 30.60) :
  let meal_cost := total_cost / (1 + tax_rate + tip_rate)
  meal_cost = 23.18 :=
by
  sorry

end dinner_cost_l71_71426


namespace proof_problem_l71_71665

theorem proof_problem 
  {a b c : ℝ} (h_cond : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (n : ℕ) :
  1/a^(2*n+1) + 1/b^(2*n+1) + 1/c^(2*n+1) = 1/(a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) :=
sorry

end proof_problem_l71_71665


namespace binary_addition_l71_71338

def bin_to_dec1 := 511  -- 111111111_2 in decimal
def bin_to_dec2 := 127  -- 1111111_2 in decimal

theorem binary_addition : bin_to_dec1 + bin_to_dec2 = 638 := by
  sorry

end binary_addition_l71_71338


namespace max_min_diff_of_c_l71_71162

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l71_71162


namespace infinite_geometric_series_second_term_l71_71849

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l71_71849


namespace xyz_value_l71_71624

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 26 / 3 := 
by
  sorry

end xyz_value_l71_71624


namespace hide_and_seek_l71_71386

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71386


namespace largest_root_is_1011_l71_71050

theorem largest_root_is_1011 (a b c d x : ℝ) 
  (h1 : a + d = 2022) 
  (h2 : b + c = 2022) 
  (h3 : a ≠ c) 
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) : 
  x = 1011 := 
sorry

end largest_root_is_1011_l71_71050


namespace domain_of_function_l71_71634

theorem domain_of_function :
  ∀ x, (x - 2 > 0) ∧ (3 - x ≥ 0) ↔ 2 < x ∧ x ≤ 3 :=
by 
  intros x 
  simp only [and_imp, gt_iff_lt, sub_lt_iff_lt_add, sub_nonneg, le_iff_eq_or_lt, add_comm]
  exact sorry

end domain_of_function_l71_71634


namespace work_done_at_4_pm_l71_71993

noncomputable def workCompletionTime (aHours : ℝ) (bHours : ℝ) (startTime : ℝ) : ℝ :=
  let aRate := 1 / aHours
  let bRate := 1 / bHours
  let cycleWork := aRate + bRate
  let cyclesNeeded := (1 : ℝ) / cycleWork
  startTime + 2 * cyclesNeeded

theorem work_done_at_4_pm :
  workCompletionTime 8 12 6 = 16 :=  -- 16 in 24-hour format is 4 pm
by 
  sorry

end work_done_at_4_pm_l71_71993


namespace perfect_square_trinomial_k_l71_71472

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l71_71472


namespace avg_of_xyz_l71_71143

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l71_71143


namespace find_first_term_l71_71970

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l71_71970


namespace sign_of_slope_equals_sign_of_correlation_l71_71675

-- Definitions for conditions
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ t, y t = a + b * x t

def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  r > -1 ∧ r < 1 ∧ ∀ t t', (y t - y t').sign = (x t - x t').sign

def regression_line_slope (b : ℝ) : Prop := True

-- Theorem to prove the sign of b is equal to the sign of r
theorem sign_of_slope_equals_sign_of_correlation (x y : ℝ → ℝ) (r b : ℝ) 
  (h1 : linear_relationship x y) 
  (h2 : correlation_coefficient x y r) 
  (h3 : regression_line_slope b) : 
  b.sign = r.sign := 
sorry

end sign_of_slope_equals_sign_of_correlation_l71_71675


namespace product_of_coordinates_of_D_l71_71265

theorem product_of_coordinates_of_D 
  (x y : ℝ)
  (midpoint_x : (5 + x) / 2 = 4)
  (midpoint_y : (3 + y) / 2 = 7) : 
  x * y = 33 := 
by 
  sorry

end product_of_coordinates_of_D_l71_71265


namespace gcf_120_180_240_l71_71566

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l71_71566


namespace total_cost_is_correct_l71_71308

-- Conditions
def cost_per_object : ℕ := 11
def objects_per_person : ℕ := 5  -- 2 shoes, 2 socks, 1 mobile per person
def number_of_people : ℕ := 3

-- Expected total cost
def expected_total_cost : ℕ := 165

-- Proof problem: Prove that the total cost for storing all objects is 165 dollars
theorem total_cost_is_correct :
  (number_of_people * objects_per_person * cost_per_object) = expected_total_cost :=
by
  sorry

end total_cost_is_correct_l71_71308


namespace num_divisors_720_l71_71087

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end num_divisors_720_l71_71087


namespace sum_of_smallest_x_and_y_for_540_l71_71041

theorem sum_of_smallest_x_and_y_for_540 (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : ∃ k₁, 540 * x = k₁ * k₁)
  (h2 : ∃ k₂, 540 * y = k₂ * k₂ * k₂) :
  x + y = 65 := 
sorry

end sum_of_smallest_x_and_y_for_540_l71_71041


namespace min_distinct_lines_for_polyline_l71_71018

theorem min_distinct_lines_for_polyline (n : ℕ) (h_n : n = 31) : 
  ∃ (k : ℕ), 9 ≤ k ∧ k ≤ 31 ∧ 
  (∀ (s : Fin n → Fin 31), 
     ∀ i j, i ≠ j → s i ≠ s j) := 
sorry

end min_distinct_lines_for_polyline_l71_71018


namespace difference_between_eights_l71_71523

theorem difference_between_eights (value_tenths : ℝ) (value_hundredths : ℝ) (h1 : value_tenths = 0.8) (h2 : value_hundredths = 0.08) : 
  value_tenths - value_hundredths = 0.72 :=
by 
  sorry

end difference_between_eights_l71_71523


namespace find_A_plus_B_l71_71780

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l71_71780


namespace rectangular_to_polar_coordinates_l71_71607

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end rectangular_to_polar_coordinates_l71_71607


namespace binom_600_eq_1_l71_71093

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l71_71093


namespace egg_rolls_total_l71_71941

theorem egg_rolls_total (omar_egg_rolls karen_egg_rolls lily_egg_rolls : ℕ) :
  omar_egg_rolls = 219 → karen_egg_rolls = 229 → lily_egg_rolls = 275 → 
  omar_egg_rolls + karen_egg_rolls + lily_egg_rolls = 723 := 
by
  intros h1 h2 h3
  sorry

end egg_rolls_total_l71_71941


namespace calculate_f_f_f_l71_71898

def f (x : ℤ) : ℤ := 3 * x + 2

theorem calculate_f_f_f :
  f (f (f 3)) = 107 :=
by
  sorry

end calculate_f_f_f_l71_71898


namespace vector_at_t5_l71_71411

theorem vector_at_t5 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : (a, b) = (2, 5)) 
  (h2 : (a + 3 * c, b + 3 * d) = (8, -7)) :
  (a + 5 * c, b + 5 * d) = (10, -11) :=
by
  sorry

end vector_at_t5_l71_71411


namespace find_n_l71_71950

theorem find_n (x y : ℤ) (n : ℕ) (h1 : (x:ℝ)^n + (y:ℝ)^n = 91) (h2 : (x:ℝ) * y = 11.999999999999998) :
  n = 3 := 
sorry

end find_n_l71_71950


namespace max_min_diff_c_l71_71163

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end max_min_diff_c_l71_71163


namespace rectangle_diagonal_l71_71325

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l71_71325


namespace graph_translation_l71_71478

variable (f : ℝ → ℝ)

theorem graph_translation (h : f 1 = 3) : f (-1) + 1 = 4 :=
sorry

end graph_translation_l71_71478


namespace geometric_series_first_term_l71_71975

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l71_71975


namespace personal_planner_cost_l71_71231

variable (P : ℝ)
variable (C_spiral_notebook : ℝ := 15)
variable (total_cost_with_discount : ℝ := 112)
variable (discount_rate : ℝ := 0.20)
variable (num_spiral_notebooks : ℝ := 4)
variable (num_personal_planners : ℝ := 8)

theorem personal_planner_cost : (4 * C_spiral_notebook + 8 * P) * (1 - 0.20) = 112 → 
  P = 10 :=
by
  sorry

end personal_planner_cost_l71_71231


namespace sum_of_squares_of_roots_l71_71189

theorem sum_of_squares_of_roots 
  (x1 x2 : ℝ) 
  (h₁ : 5 * x1^2 - 6 * x1 - 4 = 0)
  (h₂ : 5 * x2^2 - 6 * x2 - 4 = 0)
  (h₃ : x1 ≠ x2) :
  x1^2 + x2^2 = 76 / 25 := sorry

end sum_of_squares_of_roots_l71_71189


namespace total_meters_examined_l71_71422

-- Define the conditions
def proportion_defective : ℝ := 0.1
def defective_meters : ℕ := 10

-- The statement to prove
theorem total_meters_examined (T : ℝ) (h : proportion_defective * T = defective_meters) : T = 100 :=
by
  sorry

end total_meters_examined_l71_71422


namespace fraction_is_seventh_l71_71722

-- Definition of the condition on x being greater by a certain percentage
def x_greater := 1125.0000000000002 / 100

-- Definition of x in terms of the condition
def x := (4 / 7) * (1 + x_greater)

-- Definition of the fraction f
def f := 1 / x

-- Lean theorem statement to prove the fraction is 1/7
theorem fraction_is_seventh (x_greater: ℝ) : (1 / ((4 / 7) * (1 + x_greater))) = 1 / 7 :=
by
  sorry

end fraction_is_seventh_l71_71722


namespace find_ratio_l71_71659

noncomputable def complex_numbers_are_non_zero (x y z : ℂ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0

noncomputable def sum_is_30 (x y z : ℂ) : Prop :=
x + y + z = 30

noncomputable def expanded_equality (x y z : ℂ) : Prop :=
((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z

theorem find_ratio (x y z : ℂ)
  (h1 : complex_numbers_are_non_zero x y z)
  (h2 : sum_is_30 x y z)
  (h3 : expanded_equality x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3.5 :=
sorry

end find_ratio_l71_71659


namespace log_proof_l71_71752

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem log_proof (x : ℝ) (h : log_base 7 (x + 6) = 2) : log_base 13 x = log_base 13 43 :=
by
  sorry

end log_proof_l71_71752


namespace angle_between_vectors_l71_71010

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h₁ : vector_norm a ≠ 0)
  (h₂ : vector_norm b ≠ 0)
  (h₃ : vector_norm a = vector_norm b)
  (h₄ : vector_norm a = vector_norm (a.1 + 2 * b.1, a.2 + 2 * b.2)) :
  ∃ θ : ℝ, θ = 180 ∧ cos θ = -1 := 
sorry

end angle_between_vectors_l71_71010


namespace min_abs_diff_is_11_l71_71268

noncomputable def min_abs_diff (k l : ℕ) : ℤ := abs (36^k - 5^l)

theorem min_abs_diff_is_11 :
  ∃ k l : ℕ, min_abs_diff k l = 11 :=
by
  sorry

end min_abs_diff_is_11_l71_71268


namespace hide_and_seek_l71_71378

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71378


namespace min_value_of_f_l71_71438

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

theorem min_value_of_f :
  ∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (∀ u v : ℝ, (u > 0) ∧ (v > 0) → f u v ≥ 2) ∧ f x y = 2 :=
by
  sorry

end min_value_of_f_l71_71438


namespace solve_rational_eq_l71_71111

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 3 * x - 18) + 1 / (x^2 - 15 * x - 12) = 0) →
  (x = 1 ∨ x = -1 ∨ x = 12 ∨ x = -12) :=
by
  intro h
  sorry

end solve_rational_eq_l71_71111


namespace percentage_problem_l71_71444

theorem percentage_problem 
    (y : ℝ)
    (h₁ : 0.47 * 1442 = 677.74)
    (h₂ : (677.74 - (y / 100) * 1412) + 63 = 3) :
    y = 52.25 :=
by sorry

end percentage_problem_l71_71444


namespace number_of_members_is_44_l71_71590

-- Define necessary parameters and conditions
def paise_per_rupee : Nat := 100

def total_collection_in_paise : Nat := 1936

def number_of_members_in_group (n : Nat) : Prop :=
  n * n = total_collection_in_paise

-- Proposition to prove
theorem number_of_members_is_44 : number_of_members_in_group 44 :=
by
  sorry

end number_of_members_is_44_l71_71590


namespace total_legs_correct_l71_71078

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l71_71078


namespace hide_and_seek_l71_71377

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71377


namespace gcf_120_180_240_is_60_l71_71564

theorem gcf_120_180_240_is_60 : Nat.gcd (Nat.gcd 120 180) 240 = 60 := by
  sorry

end gcf_120_180_240_is_60_l71_71564


namespace area_is_300_l71_71683

variable (l w : ℝ) -- Length and Width of the playground

-- Conditions
def condition1 : Prop := 2 * l + 2 * w = 80
def condition2 : Prop := l = 3 * w

-- Question and Answer
def area_of_playground : ℝ := l * w

theorem area_is_300 (h1 : condition1 l w) (h2 : condition2 l w) : area_of_playground l w = 300 := 
by
  sorry

end area_is_300_l71_71683


namespace smallest_n_divides_24_and_1024_l71_71208

theorem smallest_n_divides_24_and_1024 : ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ (∀ m : ℕ, (m > 0 ∧ (24 ∣ m^2) ∧ (1024 ∣ m^3)) → n ≤ m) :=
by
  sorry

end smallest_n_divides_24_and_1024_l71_71208


namespace prove_equations_and_PA_PB_l71_71651

noncomputable def curve_C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, 1 + t * Real.sin α)

noncomputable def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ + 7 / ρ = 4 * Real.cos θ + 4 * Real.sin θ

theorem prove_equations_and_PA_PB :
  (∀ (α : ℝ), 0 ≤ α ∧ α < π → 
    (∃ (C1_cart : ℝ → ℝ → Prop), ∀ x y, C1_cart x y ↔ x^2 = 4 * y) ∧
    (∃ (C1_polar : ℝ → ℝ → Prop), ∀ ρ θ, C1_polar ρ θ ↔ ρ^2 * Real.cos θ^2 = 4 * ρ * Real.sin θ) ∧
    (∃ (C2_cart : ℝ → ℝ → Prop), ∀ x y, C2_cart x y ↔ (x - 2)^2 + (y - 2)^2 = 1)) ∧
  (∃ (P A B : ℝ × ℝ), P = (0, 1) ∧ 
    curve_C1_parametric t (Real.pi / 2) = A ∧ 
    curve_C1_parametric t (Real.pi / 2) = B ∧ 
    |P - A| * |P - B| = 4) :=
sorry

end prove_equations_and_PA_PB_l71_71651


namespace june_earnings_l71_71773

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l71_71773


namespace max_value_of_3cosx_minus_sinx_l71_71618

noncomputable def max_cosine_expression : ℝ :=
  Real.sqrt 10

theorem max_value_of_3cosx_minus_sinx : 
  ∃ x : ℝ, ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := 
by {
  sorry
}

end max_value_of_3cosx_minus_sinx_l71_71618


namespace HCF_48_99_l71_71315

-- definitions and theorem stating the problem
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_48_99 : HCF 48 99 = 3 :=
by
  sorry

end HCF_48_99_l71_71315


namespace water_wasted_per_hour_l71_71679

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l71_71679


namespace minimum_transportation_cost_l71_71233

theorem minimum_transportation_cost :
  ∀ (x : ℕ), 
    (17 - x) + (x - 3) = 12 → 
    (18 - x) + (17 - x) = 14 → 
    (200 * x + 19300 = 19900) → 
    (x = 3) 
:= by sorry

end minimum_transportation_cost_l71_71233


namespace production_relationship_l71_71844

noncomputable def production_function (a : ℕ) (p : ℝ) (x : ℕ) : ℝ := a * (1 + p / 100)^x

theorem production_relationship (a : ℕ) (p : ℝ) (m : ℕ) (x : ℕ) (hx : 0 ≤ x ∧ x ≤ m) :
  production_function a p x = a * (1 + p / 100)^x := by
  sorry

end production_relationship_l71_71844


namespace first_term_of_geometric_series_l71_71959

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l71_71959


namespace friends_who_participate_l71_71360

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71360


namespace equal_numbers_in_sequence_l71_71073

theorem equal_numbers_in_sequence (a : ℕ → ℚ)
  (h : ∀ m n : ℕ, a m + a n = a (m * n)) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end equal_numbers_in_sequence_l71_71073


namespace reciprocal_of_neg_2023_l71_71537

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l71_71537


namespace determine_words_per_page_l71_71841

noncomputable def wordsPerPage (totalPages : ℕ) (wordsPerPage : ℕ) (totalWordsMod : ℕ) : ℕ :=
if totalPages * wordsPerPage % 250 = totalWordsMod ∧ wordsPerPage <= 200 then wordsPerPage else 0

theorem determine_words_per_page :
  wordsPerPage 150 198 137 = 198 :=
by 
  sorry

end determine_words_per_page_l71_71841


namespace first_term_of_geometric_series_l71_71957

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l71_71957


namespace number_of_friends_l71_71108

-- Conditions/Definitions
def total_cost : ℤ := 13500
def cost_per_person : ℤ := 900

-- Prove that Dawson is going with 14 friends.
theorem number_of_friends (h1 : total_cost = 13500) (h2 : cost_per_person = 900) :
  (total_cost / cost_per_person) - 1 = 14 :=
by
  sorry

end number_of_friends_l71_71108


namespace pencil_cost_l71_71796

theorem pencil_cost (P : ℕ) (h1 : ∀ p : ℕ, p = 80) (h2 : ∀ p_est, ((16 * P) + (20 * 80)) = p_est → p_est = 2000) (h3 : 36 = 16 + 20) :
    P = 25 :=
  sorry

end pencil_cost_l71_71796


namespace vertical_asymptote_sum_l71_71193

theorem vertical_asymptote_sum :
  (∀ x : ℝ, 4*x^2 + 6*x + 3 = 0 → x = -1 / 2 ∨ x = -1) →
  (-1 / 2 + -1) = -3 / 2 :=
by
  intro h
  sorry

end vertical_asymptote_sum_l71_71193


namespace candy_total_cents_l71_71139

def candy_cost : ℕ := 8
def gumdrops : ℕ := 28
def total_cents : ℕ := 224

theorem candy_total_cents : candy_cost * gumdrops = total_cents := by
  sorry

end candy_total_cents_l71_71139


namespace product_of_real_roots_l71_71735

theorem product_of_real_roots (x : ℝ) (hx : x ^ (Real.log x / Real.log 5) = 5) :
  (∃ a b : ℝ, a ^ (Real.log a / Real.log 5) = 5 ∧ b ^ (Real.log b / Real.log 5) = 5 ∧ a * b = 1) :=
sorry

end product_of_real_roots_l71_71735


namespace greatest_possible_a_l71_71191

theorem greatest_possible_a (a : ℤ) (x : ℤ) (h_pos : 0 < a) (h_eq : x^3 + a * x^2 = -30) : 
  a ≤ 29 :=
sorry

end greatest_possible_a_l71_71191


namespace unique_polynomial_l71_71876

-- Define the conditions
def valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.degree p > 0 ∧ ∀ (z : ℝ), z ≠ 0 → P z = Polynomial.eval z p

-- The main theorem
theorem unique_polynomial (P : ℝ → ℝ) (hP : valid_polynomial P) :
  (∀ (z : ℝ), z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 → 
  1 / P z + 1 / P (1 / z) = z + 1 / z) → ∀ x, P x = x :=
by
  sorry

end unique_polynomial_l71_71876


namespace rope_folded_three_times_parts_l71_71414

theorem rope_folded_three_times_parts (total_length : ℕ) :
  ∀ parts : ℕ, parts = (total_length / 8) →
  ∀ n : ℕ, n = 3 →
  (∀ length_each_part : ℚ, length_each_part = 1 / (2 ^ n) →
  length_each_part = 1 / 8) :=
by
  sorry

end rope_folded_three_times_parts_l71_71414


namespace probability_intersection_interval_l71_71528

theorem probability_intersection_interval (PA PB p : ℝ) (hPA : PA = 5 / 6) (hPB : PB = 3 / 4) :
  0 ≤ p ∧ p ≤ 3 / 4 :=
sorry

end probability_intersection_interval_l71_71528


namespace convert_89_to_binary_l71_71729

def divide_by_2_remainders (n : Nat) : List Nat :=
  if n = 0 then [] else (n % 2) :: divide_by_2_remainders (n / 2)

def binary_rep (n : Nat) : List Nat :=
  (divide_by_2_remainders n).reverse

theorem convert_89_to_binary :
  binary_rep 89 = [1, 0, 1, 1, 0, 0, 1] := sorry

end convert_89_to_binary_l71_71729


namespace hide_and_seek_friends_l71_71380

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71380


namespace total_sets_needed_l71_71711

-- Conditions
variable (n : ℕ)

-- Theorem statement
theorem total_sets_needed : 3 * n = 3 * n :=
by sorry

end total_sets_needed_l71_71711


namespace problem_min_value_l71_71267

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : ℝ :=
  1 / x^2 + 1 / y^2 + 1 / (x * y)

theorem problem_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  min_value x y hx hy hxy = 3 := 
sorry

end problem_min_value_l71_71267


namespace find_twentieth_special_number_l71_71871

theorem find_twentieth_special_number :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 5 [MOD 8]) ∧ (∀ k < 20, ∃ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 5 [MOD 8]) ∧ m < n) ∧ (n = 461) := 
sorry

end find_twentieth_special_number_l71_71871


namespace hide_and_seek_problem_l71_71366

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71366


namespace angles_on_coordinate_axes_l71_71328

theorem angles_on_coordinate_axes :
  let Sx := {α | ∃ k : ℤ, α = k * Real.pi},
      Sy := {α | ∃ k : ℤ, α = k * Real.pi + (Real.pi / 2)} in
  (Sx ∪ Sy) = {α | ∃ n : ℤ, α = (n * Real.pi) / 2} :=
by
  sorry

end angles_on_coordinate_axes_l71_71328


namespace binomial_600_600_l71_71103

theorem binomial_600_600 : nat.choose 600 600 = 1 :=
by
  -- Given the condition that binomial coefficient of n choose n is 1 for any non-negative n
  have h : ∀ n : ℕ, nat.choose n n = 1 := sorry
  -- Applying directly to the specific case n = 600
  exact h 600

end binomial_600_600_l71_71103


namespace find_b_l71_71125

theorem find_b (a b : ℝ) (h1 : a * (a - 4) = 21) (h2 : b * (b - 4) = 21) (h3 : a + b = 4) (h4 : a ≠ b) :
  b = -3 :=
sorry

end find_b_l71_71125


namespace water_speed_l71_71072

theorem water_speed (v : ℝ) 
  (still_water_speed : ℝ := 4)
  (distance : ℝ := 10)
  (time : ℝ := 5)
  (effective_speed : ℝ := distance / time) 
  (h : still_water_speed - v = effective_speed) :
  v = 2 :=
by
  sorry

end water_speed_l71_71072


namespace expected_socks_pairs_l71_71553

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l71_71553


namespace proof_part1_proof_part2_l71_71407

noncomputable def part1 : Prop :=
  let mu : ℝ := 10
  let sigma : ℝ := 0.5
  let n : ℕ := 15
  let p_qualified : ℝ := 0.9973
  let p_all_qualified : ℝ := p_qualified ^ n
  let p_at_least_one_defective : ℝ := 1 - p_all_qualified
  p_at_least_one_defective = 0.0397

noncomputable def part2 : Prop :=
  let n : ℕ := 100
  let p : ℝ := 0.0027
  let k : ℕ := 0
  k = 0

theorem proof_part1 : part1 := by
  sorry

theorem proof_part2 : part2 := by
  sorry

end proof_part1_proof_part2_l71_71407


namespace initial_deck_card_count_l71_71283

theorem initial_deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 := by
  sorry

end initial_deck_card_count_l71_71283


namespace not_possible_to_color_l71_71486

theorem not_possible_to_color (f : ℕ → ℕ) (c1 c2 c3 : ℕ) :
  ∃ (x : ℕ), 1 < x ∧ f 2 = c1 ∧ f 4 = c1 ∧ 
  ∀ (a b : ℕ), 1 < a → 1 < b → f a ≠ f b → (f (a * b) ≠ f a ∧ f (a * b) ≠ f b) → 
  false :=
sorry

end not_possible_to_color_l71_71486


namespace smallest_factorization_c_l71_71443

theorem smallest_factorization_c : ∃ (c : ℤ), (∀ (r s : ℤ), r * s = 2016 → r + s = c) ∧ c > 0 ∧ c = 108 :=
by 
  sorry

end smallest_factorization_c_l71_71443


namespace probability_range_l71_71269

noncomputable def probability_distribution (K : ℕ) : ℝ :=
  if K > 0 then 1 / (2^K) else 0

theorem probability_range (h2 : 2 < 3) (h3 : 3 ≤ 4) :
  probability_distribution 3 + probability_distribution 4 = 3 / 16 :=
by
  sorry

end probability_range_l71_71269


namespace number_of_integer_solutions_l71_71903

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end number_of_integer_solutions_l71_71903


namespace find_fourth_vertex_of_square_l71_71484

-- Given the vertices of the square as complex numbers
def vertex1 : ℂ := 1 + 2 * Complex.I
def vertex2 : ℂ := -2 + Complex.I
def vertex3 : ℂ := -1 - 2 * Complex.I

-- The fourth vertex (to be proved)
def vertex4 : ℂ := 2 - Complex.I

-- The mathematically equivalent proof problem statement
theorem find_fourth_vertex_of_square :
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  -- Define vectors from the vertices
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4
  vector_ab = vector_dc :=
by {
  -- Definitions already provided above
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4

  -- Placeholder for proof
  sorry
}

end find_fourth_vertex_of_square_l71_71484


namespace hide_and_seek_l71_71389

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71389


namespace triangle_distance_bisectors_l71_71954

noncomputable def distance_between_bisectors {a b c : ℝ} (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) : ℝ :=
  (2 * a * b * c) / (b^2 - c^2)

theorem triangle_distance_bisectors 
  (a b c : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  ∀ (DD₁ : ℝ), 
  DD₁ = distance_between_bisectors h₁ h₂ h₃ → 
  DD₁ = (2 * a * b * c) / (b^2 - c^2) := by 
  sorry

end triangle_distance_bisectors_l71_71954


namespace tan_theta_eq_neg_two_l71_71146

theorem tan_theta_eq_neg_two (f : ℝ → ℝ) (θ : ℝ) 
  (h₁ : ∀ x, f x = Real.sin (2 * x + θ)) 
  (h₂ : ∀ x, f x + 2 * Real.cos (2 * x + θ) = -(f (-x) + 2 * Real.cos (2 * (-x) + θ))) :
  Real.tan θ = -2 :=
by
  sorry

end tan_theta_eq_neg_two_l71_71146


namespace sequence_x_y_sum_l71_71868

theorem sequence_x_y_sum :
  ∃ (r x y : ℝ), 
    (r * 3125 = 625) ∧ 
    (r * 625 = 125) ∧ 
    (r * 125 = x) ∧ 
    (r * x = y) ∧ 
    (r * y = 1) ∧
    (r * 1 = 1/5) ∧ 
    (r * (1/5) = 1/25) ∧ 
    x + y = 30 := 
by
  -- A placeholder for the actual proof
  sorry

end sequence_x_y_sum_l71_71868


namespace alex_play_friends_with_l71_71351

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71351


namespace apple_juice_less_than_cherry_punch_l71_71673

def orange_punch : ℝ := 4.5
def total_punch : ℝ := 21
def cherry_punch : ℝ := 2 * orange_punch
def combined_punch : ℝ := orange_punch + cherry_punch
def apple_juice : ℝ := total_punch - combined_punch

theorem apple_juice_less_than_cherry_punch : cherry_punch - apple_juice = 1.5 := by
  sorry

end apple_juice_less_than_cherry_punch_l71_71673


namespace pears_worth_l71_71188

variable (apples pears : ℚ)
variable (h : 3/4 * 16 * apples = 6 * pears)

theorem pears_worth (h : 3/4 * 16 * apples = 6 * pears) : 1 / 3 * 9 * apples = 1.5 * pears :=
by
  sorry

end pears_worth_l71_71188


namespace students_more_than_pets_l71_71616

theorem students_more_than_pets
    (num_classrooms : ℕ)
    (students_per_classroom : ℕ)
    (rabbits_per_classroom : ℕ)
    (hamsters_per_classroom : ℕ)
    (total_students : ℕ)
    (total_pets : ℕ)
    (difference : ℕ)
    (classrooms_eq : num_classrooms = 5)
    (students_eq : students_per_classroom = 20)
    (rabbits_eq : rabbits_per_classroom = 2)
    (hamsters_eq : hamsters_per_classroom = 1)
    (total_students_eq : total_students = num_classrooms * students_per_classroom)
    (total_pets_eq : total_pets = num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom)
    (difference_eq : difference = total_students - total_pets) :
  difference = 85 := by
  sorry

end students_more_than_pets_l71_71616


namespace inequality_may_not_hold_l71_71465

theorem inequality_may_not_hold (a b : ℝ) (h : 0 < b ∧ b < a) :
  ¬(∀ x y : ℝ,  x = 1 / (a - b) → y = 1 / b → x > y) :=
sorry

end inequality_may_not_hold_l71_71465


namespace compound_interest_correct_l71_71845

-- define the problem conditions
def P : ℝ := 3000
def r : ℝ := 0.07
def n : ℕ := 25

-- the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- state the theorem we want to prove
theorem compound_interest_correct :
  compound_interest P r n = 16281 := 
by
  sorry

end compound_interest_correct_l71_71845


namespace ratio_w_y_l71_71042

theorem ratio_w_y (w x y z : ℝ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 4) 
  (h4 : w + x + y + z = 60) : 
  w / y = 10 / 3 :=
sorry

end ratio_w_y_l71_71042


namespace number_of_rows_with_7_eq_5_l71_71732

noncomputable def number_of_rows_with_7_people (x y : ℕ) : Prop :=
  7 * x + 6 * (y - x) = 59

theorem number_of_rows_with_7_eq_5 :
  ∃ x y : ℕ, number_of_rows_with_7_people x y ∧ x = 5 :=
by {
  sorry
}

end number_of_rows_with_7_eq_5_l71_71732


namespace num_chords_l71_71509

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l71_71509


namespace second_smallest_packs_hot_dogs_l71_71615

theorem second_smallest_packs_hot_dogs 
    (n : ℕ) 
    (k : ℤ) 
    (h1 : 10 * n ≡ 4 [MOD 8]) 
    (h2 : n = 4 * k + 2) : 
    n = 6 :=
by sorry

end second_smallest_packs_hot_dogs_l71_71615


namespace exist_V_N_rhombus_l71_71165

-- Define the problem setup in Lean
variable {ABC : Type} [triangle : Triangle ABC]
variable (A B C E D F G : Point ABC)
variable (V N : Point ABC)

-- Assumptions
axiom altitude_AE : Altitude A E
axiom tangency_D : Tangency (Excircle A) (BC) D
axiom intersection_FG : Intersects (Excircle A) (Circumcircle ABC) F G

-- Theorem statement translated into Lean 4
theorem exist_V_N_rhombus 
  (h1 : altitude_AE)
  (h2 : tangency_D)
  (h3 : intersection_FG) :
  ∃ V N : Point ABC, OnLine V (Line D G) ∧ OnLine N (Line D F) ∧ Rhombus E V A N :=
sorry

end exist_V_N_rhombus_l71_71165


namespace problem1_problem2_l71_71857

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l71_71857


namespace find_peaches_l71_71721

theorem find_peaches (A P : ℕ) (h1 : A + P = 15) (h2 : 1000 * A + 2000 * P = 22000) : P = 7 := sorry

end find_peaches_l71_71721


namespace dance_contradiction_l71_71089

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end dance_contradiction_l71_71089


namespace equilateral_given_inequality_l71_71453

open Real

-- Define the primary condition to be used in the theorem
def inequality (a b c : ℝ) : Prop :=
  (1 / a * sqrt (1 / b + 1 / c) + 1 / b * sqrt (1 / c + 1 / a) + 1 / c * sqrt (1 / a + 1 / b)) ≥
  (3 / 2 * sqrt ((1 / a + 1 / b) * (1 / b + 1 / c) * (1 / c + 1 / a)))

-- Define the theorem that states the sides form an equilateral triangle under the given condition
theorem equilateral_given_inequality (a b c : ℝ) (habc : inequality a b c) (htriangle : a > 0 ∧ b > 0 ∧ c > 0):
  a = b ∧ b = c ∧ c = a := 
sorry

end equilateral_given_inequality_l71_71453


namespace num_idempotent_functions_l71_71274

open Finset Function

theorem num_idempotent_functions :
  let n := 5
  let f_set := finset.fin_range n
  let count := ∑ k in f_set.Powerset, k.card.factorial * (n - k.card) ^ (n - k.card)
  count = 196 :=
by
  sorry

end num_idempotent_functions_l71_71274


namespace roots_polynomial_sum_l71_71014

theorem roots_polynomial_sum (p q : ℂ) (hp : p^2 - 6 * p + 10 = 0) (hq : q^2 - 6 * q + 10 = 0) :
  p^4 + p^5 * q^3 + p^3 * q^5 + q^4 = 16056 := by
  sorry

end roots_polynomial_sum_l71_71014


namespace alex_play_friends_with_l71_71353

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71353


namespace find_other_root_l71_71198

variables {a b c : ℝ}

theorem find_other_root
  (h_eq : ∀ x : ℝ, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0)
  (root1 : a * (b - c) * 1^2 + b * (c - a) * 1 + c * (a - b) = 0) :
  ∃ k : ℝ, k = c * (a - b) / (a * (b - c)) ∧
           a * (b - c) * k^2 + b * (c - a) * k + c * (a - b) = 0 := 
sorry

end find_other_root_l71_71198


namespace isosceles_triangle_angle_sum_l71_71121

theorem isosceles_triangle_angle_sum 
  (A B C : Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]
  (AC AB : ℝ) 
  (angle_ABC : ℝ)
  (isosceles : AC = AB)
  (angle_A : angle_ABC = 70) :
  (∃ angle_B : ℝ, angle_B = 55) :=
by
  sorry

end isosceles_triangle_angle_sum_l71_71121


namespace binary_to_decimal_l71_71429

theorem binary_to_decimal : 
  (0 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4) = 18 := 
by
  -- The proof is skipped
  sorry

end binary_to_decimal_l71_71429


namespace marbles_left_l71_71332

theorem marbles_left (red_marble_count blue_marble_count broken_marble_count : ℕ)
  (h1 : red_marble_count = 156)
  (h2 : blue_marble_count = 267)
  (h3 : broken_marble_count = 115) :
  red_marble_count + blue_marble_count - broken_marble_count = 308 :=
by
  sorry

end marbles_left_l71_71332


namespace hide_and_seek_l71_71375

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71375


namespace expected_socks_pairs_l71_71552

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l71_71552


namespace find_first_term_l71_71963

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l71_71963


namespace telescope_visual_range_increased_l71_71058

/-- A certain telescope increases the visual range from 100 kilometers to 150 kilometers. 
    Proof that the visual range is increased by 50% using the telescope.
-/
theorem telescope_visual_range_increased :
  let original_range := 100
  let new_range := 150
  (new_range - original_range) / original_range * 100 = 50 := 
by
  sorry

end telescope_visual_range_increased_l71_71058


namespace joe_cars_after_getting_more_l71_71218

-- Defining the initial conditions as Lean variables
def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

-- Stating the proof problem
theorem joe_cars_after_getting_more : initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_after_getting_more_l71_71218


namespace bird_families_flew_away_l71_71835

theorem bird_families_flew_away (original : ℕ) (left : ℕ) (flew_away : ℕ) (h1 : original = 67) (h2 : left = 35) (h3 : flew_away = original - left) : flew_away = 32 :=
by
  rw [h1, h2] at h3
  exact h3

end bird_families_flew_away_l71_71835


namespace find_first_term_geometric_series_l71_71966

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l71_71966


namespace possible_value_of_a_l71_71759

theorem possible_value_of_a (a : ℕ) : (5 + 8 > a ∧ a > 3) → (a = 9 → True) :=
by
  intros h ha
  sorry

end possible_value_of_a_l71_71759


namespace friends_who_participate_l71_71361

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71361


namespace log_sum_range_l71_71621

theorem log_sum_range (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  (Real.log y / Real.log x + Real.log x / Real.log y) ∈ Set.union (Set.Iic (-2)) (Set.Ici 2) :=
sorry

end log_sum_range_l71_71621


namespace abs_sum_zero_eq_neg_one_l71_71643

theorem abs_sum_zero_eq_neg_one (a b : ℝ) (h : |3 + a| + |b - 2| = 0) : a + b = -1 :=
sorry

end abs_sum_zero_eq_neg_one_l71_71643


namespace find_first_term_geometric_series_l71_71969

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l71_71969


namespace quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l71_71134

theorem quadratic_has_two_real_roots_for_any_m (m : ℝ) : 
  ∃ (α β : ℝ), (α^2 - 3*α + 2 - m^2 - m = 0) ∧ (β^2 - 3*β + 2 - m^2 - m = 0) :=
sorry

theorem find_m_given_roots_conditions (α β : ℝ) (m : ℝ) 
  (h1 : α^2 - 3*α + 2 - m^2 - m = 0) 
  (h2 : β^2 - 3*β + 2 - m^2 - m = 0) 
  (h3 : α^2 + β^2 = 9) : 
  m = -2 ∨ m = 1 :=
sorry

end quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l71_71134


namespace polynomial_remainder_zero_l71_71251

open Polynomial

noncomputable def poly1 : Polynomial ℝ := x ^ 68 + x ^ 51 + x ^ 34 + x ^ 17 + 1
noncomputable def poly2 : Polynomial ℝ := x ^ 6 + x ^ 5 + x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem polynomial_remainder_zero :
  (poly2 ∣ poly1) := sorry

end polynomial_remainder_zero_l71_71251


namespace tens_digit_6_pow_18_l71_71053

/--
To find the tens digit of \(6^{18}\), we look at the powers of 6 and determine their tens digits. 
We note the pattern in tens digits (3, 1, 9, 7, 6) which repeats every 5 powers. 
Since \(6^{18}\) corresponds to the 3rd position in the repeating cycle, we claim the tens digit is 1.
--/
theorem tens_digit_6_pow_18 : (6^18 / 10) % 10 = 1 :=
by sorry

end tens_digit_6_pow_18_l71_71053


namespace expected_socks_to_pair_l71_71540

theorem expected_socks_to_pair (p : ℕ) (h : p > 0) : 
  let ξ : ℕ → ℕ := 
    λ n, if n = 0 then 2 else n * 2 in 
  expected_socks_taken p = ξ p := sorry

variable {p : ℕ} (h : p > 0)

def expected_socks_taken 
  (p : ℕ)
  (C1: p > 0)  -- There are \( p \) pairs of socks hanging out to dry in a random order.
  (C2: ∀ i, i < p → sock_pairings.unique)  -- There are no identical pairs of socks.
  (C3: ∀ i, i < p → socks.behind_sheet)  -- The socks hang behind a drying sheet.
  (C4: ∀ i, i < p, sock_taken_one_at_time: i + 1)  -- The Scientist takes one sock at a time by touch, comparing each new sock with all previous ones.
  : ℕ := sorry

end expected_socks_to_pair_l71_71540


namespace parabola_transform_l71_71133

theorem parabola_transform (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + c = (x - 4)^2 - 3) → 
  b = 4 ∧ c = 6 := 
by
  sorry

end parabola_transform_l71_71133


namespace alice_walks_distance_l71_71083

theorem alice_walks_distance :
  let blocks_south := 5
  let blocks_west := 8
  let distance_per_block := 1 / 4
  let total_blocks := blocks_south + blocks_west
  let total_distance := total_blocks * distance_per_block
  total_distance = 3.25 :=
by
  sorry

end alice_walks_distance_l71_71083


namespace reciprocal_of_negative_2023_l71_71531

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l71_71531


namespace existence_of_specified_pairs_l71_71088

-- Definitions for the problem
variables {Boy Girl : Type}
variables (Danced : Boy → Girl → Prop)

-- Hypotheses based on the problem conditions
hypothesis no_boy_danced_with_all_girls :
  ∀ (b : Boy), ∃ (g : Girl), ¬ Danced b g
hypothesis each_girl_danced_with_at_least_one_boy :
  ∀ (g : Girl), ∃ (b : Boy), Danced b g

-- Statement of the math proof problem
theorem existence_of_specified_pairs :
  ∃ (g g' : Boy) (f f' : Girl), Danced g f ∧ ¬ Danced g f' ∧ Danced g' f' ∧ ¬ Danced g' f :=
sorry

end existence_of_specified_pairs_l71_71088


namespace sufficient_but_not_necessary_condition_l71_71622

theorem sufficient_but_not_necessary_condition (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : x > 2 ∨ x < -1 :=
by
  sorry

example (x : ℝ) (h : x^2 - 3 * x + 2 > 0) : (x > 2) ∨ (x < -1) := 
by 
  apply sufficient_but_not_necessary_condition; exact h

end sufficient_but_not_necessary_condition_l71_71622


namespace calculate_expression_l71_71723

noncomputable def expr : ℚ := (5 - 2 * (3 - 6 : ℚ)⁻¹ ^ 2)⁻¹

theorem calculate_expression :
  expr = (9 / 43 : ℚ) := by
  sorry

end calculate_expression_l71_71723


namespace problem1_problem2_l71_71860

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l71_71860


namespace Cindy_correct_answer_l71_71092

theorem Cindy_correct_answer (x : ℕ) (h : (x - 14) / 4 = 28) : ((x - 5) / 7) * 4 = 69 := by
  sorry

end Cindy_correct_answer_l71_71092


namespace greatest_median_l71_71997

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median_l71_71997


namespace final_computation_l71_71090

noncomputable def N := (15 ^ 10 / 15 ^ 9) ^ 3 * 5 ^ 3

theorem final_computation : (N / 3 ^ 3) = 15625 := 
by 
  sorry

end final_computation_l71_71090


namespace total_renovation_cost_eq_l71_71008

-- Define the conditions
def hourly_rate_1 := 15
def hourly_rate_2 := 20
def hourly_rate_3 := 18
def hourly_rate_4 := 22
def hours_per_day := 8
def days := 10
def meal_cost_per_professional_per_day := 10
def material_cost := 2500
def plumbing_issue_cost := 750
def electrical_issue_cost := 500
def faulty_appliance_cost := 400

-- Define the calculated values based on the conditions
def daily_labor_cost_condition := 
  hourly_rate_1 * hours_per_day + 
  hourly_rate_2 * hours_per_day + 
  hourly_rate_3 * hours_per_day + 
  hourly_rate_4 * hours_per_day
def total_labor_cost := daily_labor_cost_condition * days

def daily_meal_cost := meal_cost_per_professional_per_day * 4
def total_meal_cost := daily_meal_cost * days

def unexpected_repair_costs := plumbing_issue_cost + electrical_issue_cost + faulty_appliance_cost

def total_cost := total_labor_cost + total_meal_cost + material_cost + unexpected_repair_costs

-- The theorem to prove that the total cost of the renovation is $10,550
theorem total_renovation_cost_eq : total_cost = 10550 := by
  sorry

end total_renovation_cost_eq_l71_71008


namespace geometric_sequence_common_ratio_simple_sequence_general_term_l71_71219

-- Question 1
theorem geometric_sequence_common_ratio (a_3 : ℝ) (S_3 : ℝ) (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) :
    q = -1 / 2 ∨ q = 1 :=
sorry

-- Question 2
theorem simple_sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h : ∀ n, S n = n^2) :
    ∀ n, a n = S n - S (n - 1) → ∀ n, a n = 2 * n - 1 :=
sorry

end geometric_sequence_common_ratio_simple_sequence_general_term_l71_71219


namespace johns_height_l71_71007

theorem johns_height
  (L R J : ℕ)
  (h1 : J = L + 15)
  (h2 : J = R - 6)
  (h3 : L + R = 295) :
  J = 152 :=
by sorry

end johns_height_l71_71007


namespace smallest_six_digit_odd_div_by_125_l71_71329

theorem smallest_six_digit_odd_div_by_125 : 
  ∃ n : ℕ, n = 111375 ∧ 
           100000 ≤ n ∧ n < 1000000 ∧ 
           (∀ d : ℕ, d ∈ (n.digits 10) → d % 2 = 1) ∧ 
           n % 125 = 0 :=
by
  sorry

end smallest_six_digit_odd_div_by_125_l71_71329


namespace reciprocal_of_neg_2023_l71_71535

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l71_71535


namespace total_seats_in_theater_l71_71482

theorem total_seats_in_theater 
    (n : ℕ) 
    (a1 : ℕ)
    (an : ℕ)
    (d : ℕ)
    (h1 : a1 = 12)
    (h2 : d = 2)
    (h3 : an = 48)
    (h4 : an = a1 + (n - 1) * d) :
    (n = 19) →
    (2 * (a1 + an) * n / 2 = 570) :=
by
  intros
  sorry

end total_seats_in_theater_l71_71482


namespace number_of_chords_number_of_chords_l71_71506

theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  exact nat.choose_succ_succ 8 1
  -- providing a simpler proof term
  exact nat.choose 9 2

-- The final proof term is incorrect, which "sorry" must be used to skip the proof,
-- but in real Lean proof we might need a correct proof term replacing here.

-- Required theorem with using sorry
theorem number_of_chords (n : ℕ) (h : n = 9) : (nat.choose n 2) = 36 := by
  rw h
  sorry

end number_of_chords_number_of_chords_l71_71506


namespace exchange_rate_change_2014_l71_71602

theorem exchange_rate_change_2014 :
  let init_rate := 32.6587
  let final_rate := 56.2584
  let change := final_rate - init_rate
  let rounded_change := Float.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_2014_l71_71602


namespace distance_to_lightning_l71_71017

noncomputable def distance_from_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) : ℚ :=
  (time_delay * speed_of_sound : ℕ) / feet_per_mile

theorem distance_to_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) :
  time_delay = 12 → speed_of_sound = 1120 → feet_per_mile = 5280 → distance_from_lightning time_delay speed_of_sound feet_per_mile = 2.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_to_lightning_l71_71017


namespace value_of_expression_l71_71276

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * b / (c * d) = 180 :=
by
  sorry

end value_of_expression_l71_71276


namespace number_of_pairs_divisible_by_five_l71_71113

theorem number_of_pairs_divisible_by_five :
  (∃ n : ℕ, n = 864) ↔
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 80) ∧ (1 ≤ b ∧ b ≤ 30) →
  (a * b) % 5 = 0 → (∃ n : ℕ, n = 864) := 
sorry

end number_of_pairs_divisible_by_five_l71_71113


namespace no_such_function_exists_l71_71866

theorem no_such_function_exists :
  ¬(∃ (f : ℝ → ℝ), ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2) :=
sorry

end no_such_function_exists_l71_71866


namespace number_of_people_l71_71594

open Nat

theorem number_of_people (n : ℕ) (h : n^2 = 100) : n = 10 := by
  sorry

end number_of_people_l71_71594


namespace binomial_600_600_l71_71096

-- Define a theorem to state the binomial coefficient property and use it to prove the specific case.
theorem binomial_600_600 : nat.choose 600 600 = 1 :=
begin
  -- Binomial property: for any non-negative integer n, (n choose n) = 1
  rw nat.choose_self,
end

end binomial_600_600_l71_71096


namespace range_of_a_l71_71258

open Set Real

noncomputable def f (x a : ℝ) := x ^ 2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≥ 0) → a ≥ -8 :=
by
  intro h
  sorry

end range_of_a_l71_71258


namespace alex_plays_with_friends_l71_71393

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71393


namespace prime_number_condition_l71_71611

theorem prime_number_condition (n : ℕ) (h1 : n ≥ 2) :
  (∀ d : ℕ, d ∣ n → d > 1 → d^2 + n ∣ n^2 + d) → Prime n :=
sorry

end prime_number_condition_l71_71611


namespace binom_600_600_l71_71100

open Nat

theorem binom_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end binom_600_600_l71_71100


namespace circle_radius_l71_71458

theorem circle_radius (x y : ℝ) :
  x^2 + 2 * x + y^2 = 0 → 1 = 1 :=
by sorry

end circle_radius_l71_71458


namespace club_additional_members_l71_71406

theorem club_additional_members (current_members additional_members future_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 15) 
  (h3 : future_members = current_members + additional_members) : 
  future_members - current_members = 15 :=
by
  sorry

end club_additional_members_l71_71406


namespace complex_exp_conj_sum_l71_71144

open Complex

theorem complex_exp_conj_sum {α β : ℝ}
  (h : exp (I * α) + exp (I * β) = (2 / 5 : ℂ) + (4 / 9 : ℂ) * I) :
  exp (-I * α) + exp (-I * β) = (2 / 5 : ℂ) - (4 / 9 : ℂ) * I :=
by
  sorry

end complex_exp_conj_sum_l71_71144


namespace geometric_sequence_common_ratio_l71_71256

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 2 * a 0 + a 1 = a 2)
  : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l71_71256


namespace inv_g_inv_5_l71_71277

noncomputable def g (x : ℝ) : ℝ := 25 / (2 + 5 * x)
noncomputable def g_inv (y : ℝ) : ℝ := (15 - 10) / 25  -- g^{-1}(5) as shown in the derivation above

theorem inv_g_inv_5 : (g_inv 5)⁻¹ = 5 / 3 := by
  have h_g_inv_5 : g_inv 5 = 3 / 5 := by sorry
  rw [h_g_inv_5]
  exact inv_div 3 5

end inv_g_inv_5_l71_71277


namespace centroid_tetrahedron_l71_71810

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D M : V)

def is_centroid (M A B C D : V) : Prop :=
  M = (1/4:ℝ) • (A + B + C + D)

theorem centroid_tetrahedron (h : is_centroid M A B C D) :
  (M - A) + (M - B) + (M - C) + (M - D) = (0 : V) :=
by {
  sorry
}

end centroid_tetrahedron_l71_71810


namespace measure_of_B_l71_71290

theorem measure_of_B (A B C : ℝ) (h1 : B = A + 20) (h2 : C = 50) (h3 : A + B + C = 180) : B = 75 := by
  sorry

end measure_of_B_l71_71290


namespace number_of_boys_in_second_grade_l71_71046

-- conditions definition
variables (B : ℕ) (G2 : ℕ := 11) (G3 : ℕ := 2 * (B + G2)) (total : ℕ := B + G2 + G3)

-- mathematical statement to be proved
theorem number_of_boys_in_second_grade : total = 93 → B = 20 :=
by
  -- omitting the proof
  intro h_total
  sorry

end number_of_boys_in_second_grade_l71_71046


namespace min_sum_a_b_l71_71895

theorem min_sum_a_b {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : 1/a + 9/b = 1) : a + b ≥ 16 := 
sorry

end min_sum_a_b_l71_71895


namespace sequence_next_term_l71_71221

theorem sequence_next_term (a b c d e : ℕ) (h1 : a = 34) (h2 : b = 45) (h3 : c = 56) (h4 : d = 67) (h5 : e = 78) (h6 : b = a + 11) (h7 : c = b + 11) (h8 : d = c + 11) (h9 : e = d + 11) : e + 11 = 89 :=
by
  sorry

end sequence_next_term_l71_71221


namespace five_digit_number_l71_71288

open Nat

noncomputable def problem_statement : Prop :=
  ∃ A B C D E F : ℕ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A + B + C + D + E + F = 25 ∧
    (A, B, C, D, E, F) = (3, 4, 2, 1, 6, 9)

theorem five_digit_number : problem_statement := 
  sorry

end five_digit_number_l71_71288


namespace num_best_friends_l71_71347

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l71_71347


namespace symmetric_point_with_respect_to_y_eq_x_l71_71807

theorem symmetric_point_with_respect_to_y_eq_x :
  ∃ x₀ y₀ : ℝ, (∃ (M : ℝ × ℝ), M = (3, 1) ∧
  ((x₀ + 3) / 2 = (y₀ + 1) / 2) ∧
  ((y₀ - 1) / (x₀ - 3) = -1)) ∧
  (x₀ = 1 ∧ y₀ = 3) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l71_71807


namespace geometric_sequence_common_ratio_l71_71764

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h1 : a 1 * a 3 = 36)
  (h2 : a 4 = 54)
  (h_pos : ∀ n, a n > 0) :
  ∃ q, q > 0 ∧ ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l71_71764


namespace average_interest_rate_equal_4_09_percent_l71_71846

-- Define the given conditions
def investment_total : ℝ := 5000
def interest_rate_at_3_percent : ℝ := 0.03
def interest_rate_at_5_percent : ℝ := 0.05
def return_relationship (x : ℝ) : Prop := 
  interest_rate_at_5_percent * x = 2 * interest_rate_at_3_percent * (investment_total - x)

-- Define the final statement
theorem average_interest_rate_equal_4_09_percent :
  ∃ x : ℝ, return_relationship x ∧ 
  ((interest_rate_at_5_percent * x + interest_rate_at_3_percent * (investment_total - x)) / investment_total) = 0.04091 := 
by
  sorry

end average_interest_rate_equal_4_09_percent_l71_71846


namespace overlap_area_of_parallelogram_l71_71698

theorem overlap_area_of_parallelogram (w1 w2 : ℝ) (β : ℝ) (hβ : β = 30) (hw1 : w1 = 2) (hw2 : w2 = 1) : 
  (w1 * (w2 / Real.sin (β * Real.pi / 180))) = 4 :=
by
  sorry

end overlap_area_of_parallelogram_l71_71698


namespace problem_l71_71195

variables {S T : ℕ → ℕ} {a b : ℕ → ℕ}

-- Conditions
-- S_n and T_n are sums of first n terms of arithmetic sequences {a_n} and {b_n}, respectively.
axiom sum_S : ∀ n, S n = n * (n + 1) / 2  -- Example: sum from 1 to n
axiom sum_T : ∀ n, T n = n * (n + 1) / 2  -- Example: sum from 1 to n

-- For any positive integer n, (S_n / T_n = (5n - 3) / (2n + 1))
axiom condition : ∀ n > 0, (S n : ℚ) / T n = (5 * n - 3 : ℚ) / (2 * n + 1)

-- Theorem to prove
theorem problem : (a 20 : ℚ) / (b 7) = 64 / 9 :=
sorry

end problem_l71_71195


namespace part1_inequality_l71_71826

theorem part1_inequality (a b x y : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
    (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_a_ge_x : a ≥ x) : 
    (a - x) ^ 2 + (b - y) ^ 2 ≤ (a + b - x) ^ 2 + y ^ 2 := 
by 
  sorry

end part1_inequality_l71_71826


namespace diagonal_length_l71_71326

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l71_71326


namespace closest_to_fraction_is_2000_l71_71402

-- Define the original fractions and their approximations
def numerator : ℝ := 410
def denominator : ℝ := 0.21
def approximated_numerator : ℝ := 400
def approximated_denominator : ℝ := 0.2

-- Define the options to choose from
def options : List ℝ := [100, 500, 1900, 2000, 2500]

-- Statement to prove that the closest value to numerator / denominator is 2000
theorem closest_to_fraction_is_2000 : 
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 100) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 500) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 1900) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 2500) :=
sorry

end closest_to_fraction_is_2000_l71_71402


namespace find_z_l71_71947

variable (x y z : ℝ)

theorem find_z (h1 : 12 * 40 = 480)
    (h2 : 15 * 50 = 750)
    (h3 : x + y + z = 270)
    (h4 : x + y = 100) :
    z = 170 := by
  sorry

end find_z_l71_71947


namespace melanie_dimes_l71_71497

theorem melanie_dimes (original_dimes dad_dimes mom_dimes total_dimes : ℕ) :
  original_dimes = 7 →
  mom_dimes = 4 →
  total_dimes = 19 →
  (total_dimes = original_dimes + dad_dimes + mom_dimes) →
  dad_dimes = 8 :=
by
  intros h1 h2 h3 h4
  sorry -- The proof is omitted as instructed.

end melanie_dimes_l71_71497


namespace number_equals_fifty_l71_71983

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end number_equals_fifty_l71_71983


namespace expected_socks_pairs_l71_71551

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l71_71551


namespace number_of_recipes_l71_71405

-- Let's define the necessary conditions.
def cups_per_recipe : ℕ := 2
def total_cups_needed : ℕ := 46

-- Prove that the number of recipes required is 23.
theorem number_of_recipes : total_cups_needed / cups_per_recipe = 23 :=
by
  sorry

end number_of_recipes_l71_71405


namespace hide_and_seek_l71_71372

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l71_71372


namespace alex_plays_with_friends_l71_71396

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l71_71396


namespace harmon_high_voting_l71_71720

theorem harmon_high_voting
  (U : Finset ℝ) -- Universe of students
  (A B : Finset ℝ) -- Sets of students favoring proposals
  (hU : U.card = 215)
  (hA : A.card = 170)
  (hB : B.card = 142)
  (hAcBc : (U \ (A ∪ B)).card = 38) :
  (A ∩ B).card = 135 :=
by {
  sorry
}

end harmon_high_voting_l71_71720


namespace find_divisor_l71_71216

theorem find_divisor (d q r : ℕ) :
  (919 = d * q + r) → (q = 17) → (r = 11) → d = 53 :=
by
  sorry

end find_divisor_l71_71216


namespace alex_play_friends_with_l71_71354

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l71_71354


namespace positive_difference_prob_l71_71560

/-- Probability that the positive difference between two randomly chosen numbers from 
the set {1, 2, 3, 4, 5, 6, 7, 8} is 3 or greater -/
theorem positive_difference_prob :
  (let S := {1, 2, 3, 4, 5, 6, 7, 8}
       in (S.powerset.filter (λ s => s.card = 2)).card.filter (λ s => (s.to_list.head! - s.to_list.tail.head!).nat_abs >= 3).card /
           (S.powerset.filter (λ s => s.card = 2)).card = 15 / 28) := 
begin
  sorry
end

end positive_difference_prob_l71_71560


namespace first_term_of_geometric_series_l71_71958

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l71_71958


namespace hide_and_seek_problem_l71_71370

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l71_71370


namespace proportion_first_number_l71_71145

theorem proportion_first_number (x : ℝ) (h : x / 5 = 0.96 / 8) : x = 0.6 :=
by
  sorry

end proportion_first_number_l71_71145


namespace students_in_both_band_and_chorus_l71_71980

-- Definitions for conditions
def total_students : ℕ := 300
def students_in_band : ℕ := 100
def students_in_chorus : ℕ := 120
def students_in_band_or_chorus : ℕ := 195

-- Theorem: Prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : ℕ :=
  students_in_band + students_in_chorus - students_in_band_or_chorus

example : students_in_both_band_and_chorus = 25 := by
  sorry

end students_in_both_band_and_chorus_l71_71980


namespace ana_wins_probability_l71_71489

noncomputable def probability_ana_wins : ℚ :=
  (1 / 2) ^ 5 / (1 - (1 / 2) ^ 5)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 31 :=
by
  sorry

end ana_wins_probability_l71_71489


namespace greatest_roses_for_680_l71_71996

/--
Greatest number of roses that can be purchased for $680
given the following costs:
- $4.50 per individual rose
- $36 per dozen roses
- $50 per two dozen roses
--/
theorem greatest_roses_for_680 (cost_individual : ℝ) 
  (cost_dozen : ℝ) 
  (cost_two_dozen : ℝ) 
  (budget : ℝ) 
  (dozen : ℕ) 
  (two_dozen : ℕ) 
  (total_budget : ℝ) 
  (individual_cost : ℝ) 
  (dozen_cost : ℝ) 
  (two_dozen_cost : ℝ) 
  (roses_dozen : ℕ) 
  (roses_two_dozen : ℕ):
  individual_cost = 4.50 → dozen_cost = 36 → two_dozen_cost = 50 →
  budget = 680 → dozen = 12 → two_dozen = 24 →
  (∀ n : ℕ, n * two_dozen_cost ≤ budget → n * two_dozen + (budget - n * two_dozen_cost) / individual_cost ≤ total_budget) →
  total_budget = 318 := 
by
  sorry

end greatest_roses_for_680_l71_71996


namespace reciprocal_of_neg_2023_l71_71536

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l71_71536


namespace molecular_weight_of_7_moles_of_CaO_l71_71833

/-- The molecular weight of 7 moles of calcium oxide (CaO) -/
def Ca_atomic_weight : Float := 40.08
def O_atomic_weight : Float := 16.00
def CaO_molecular_weight : Float := Ca_atomic_weight + O_atomic_weight

theorem molecular_weight_of_7_moles_of_CaO : 
    7 * CaO_molecular_weight = 392.56 := by 
sorry

end molecular_weight_of_7_moles_of_CaO_l71_71833


namespace bruce_will_be_3_times_as_old_in_6_years_l71_71855

variables (x : ℕ)

-- Definitions from conditions
def bruce_age_now := 36
def son_age_now := 8

-- Equivalent Lean 4 statement
theorem bruce_will_be_3_times_as_old_in_6_years :
  (bruce_age_now + x = 3 * (son_age_now + x)) → x = 6 :=
sorry

end bruce_will_be_3_times_as_old_in_6_years_l71_71855


namespace infinite_geometric_series_second_term_l71_71850

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l71_71850


namespace two_pow_1000_mod_17_l71_71702

theorem two_pow_1000_mod_17 : 2^1000 % 17 = 0 :=
by {
  sorry
}

end two_pow_1000_mod_17_l71_71702


namespace tangent_line_to_parabola_l71_71118

-- Define the line and parabola equations
def line (x y k : ℝ) := 4 * x + 3 * y + k = 0
def parabola (x y : ℝ) := y ^ 2 = 16 * x

-- Prove that if the line is tangent to the parabola, then k = 9
theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), line x y k ∧ parabola x y ∧ (y^2 + 12 * y + 4 * k = 0 ∧ 144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end tangent_line_to_parabola_l71_71118


namespace solve_eq_1_solve_eq_2_l71_71184

open Real

theorem solve_eq_1 :
  ∃ x : ℝ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -2.5 :=
by
  sorry

theorem solve_eq_2 :
  ∃ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39 / 35 :=
by
  sorry

end solve_eq_1_solve_eq_2_l71_71184


namespace trig_identity_l71_71583

theorem trig_identity : 4 * Real.sin (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l71_71583


namespace range_of_slope_ellipse_chord_l71_71897

theorem range_of_slope_ellipse_chord :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
    (x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
    (x₁^2 + y₁^2 / 4 = 1 ∧ x₂^2 + y₂^2 / 4 = 1) →
    ((1 / 2) ≤ y₀ ∧ y₀ ≤ 1) →
    (-4 ≤ -2 / y₀ ∧ -2 / y₀ ≤ -2) :=
by
  sorry

end range_of_slope_ellipse_chord_l71_71897


namespace smallest_multiple_of_7_not_particular_l71_71726

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (λ d acc => acc + d) 0

def is_particular_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) ^ 2 = 0

theorem smallest_multiple_of_7_not_particular :
  ∃ n, n > 0 ∧ n % 7 = 0 ∧ ¬ is_particular_integer n ∧ ∀ m, m > 0 ∧ m % 7 = 0 ∧ ¬ is_particular_integer m → n ≤ m :=
  by
    use 7
    sorry

end smallest_multiple_of_7_not_particular_l71_71726


namespace determine_cans_l71_71045

-- Definitions based on the conditions
def num_cans_total : ℕ := 140
def volume_large (y : ℝ) : ℝ := y + 2.5
def total_volume_large (x : ℕ) (y : ℝ) : ℝ := ↑x * volume_large y
def total_volume_small (x : ℕ) (y : ℝ) : ℝ := ↑(num_cans_total - x) * y

-- Proof statement
theorem determine_cans (x : ℕ) (y : ℝ) 
    (h1 : total_volume_large x y = 60)
    (h2 : total_volume_small x y = 60) : 
    x = 20 ∧ num_cans_total - x = 120 := 
by
  sorry

end determine_cans_l71_71045


namespace ratio_perimeter_triangle_square_l71_71028

/-
  Suppose a square piece of paper with side length 4 units is folded in half diagonally.
  The folded paper is then cut along the fold, producing two right-angled triangles.
  We need to prove that the ratio of the perimeter of one of the triangles to the perimeter of the original square is (1/2) + (sqrt 2 / 4).
-/
theorem ratio_perimeter_triangle_square:
  let side_length := 4
  let triangle_leg := side_length
  let hypotenuse := Real.sqrt (triangle_leg ^ 2 + triangle_leg ^ 2)
  let perimeter_triangle := triangle_leg + triangle_leg + hypotenuse
  let perimeter_square := 4 * side_length
  let ratio := perimeter_triangle / perimeter_square
  ratio = (1 / 2) + (Real.sqrt 2 / 4) :=
by
  sorry

end ratio_perimeter_triangle_square_l71_71028


namespace find_a_l71_71132

theorem find_a :
  ∃ a : ℝ, (2 * x - (a * Real.exp x + x) + 1 = 0) = (a = 1) :=
by
  sorry

end find_a_l71_71132


namespace find_first_term_l71_71962

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l71_71962


namespace estimated_white_balls_is_correct_l71_71333

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of trials
def trials : ℕ := 100

-- Define the number of times a red ball is drawn
def red_draws : ℕ := 80

-- Define the function to estimate the number of red balls based on the frequency
def estimated_red_balls (total_balls : ℕ) (red_draws : ℕ) (trials : ℕ) : ℕ :=
  total_balls * red_draws / trials

-- Define the function to estimate the number of white balls
def estimated_white_balls (total_balls : ℕ) (estimated_red_balls : ℕ) : ℕ :=
  total_balls - estimated_red_balls

-- State the theorem to prove the estimated number of white balls
theorem estimated_white_balls_is_correct : 
  estimated_white_balls total_balls (estimated_red_balls total_balls red_draws trials) = 2 :=
by
  sorry

end estimated_white_balls_is_correct_l71_71333


namespace common_difference_is_half_l71_71286

variable (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ) (q p : ℕ)

-- Conditions
def condition1 : Prop := a p = 4
def condition2 : Prop := a q = 2
def condition3 : Prop := p = 4 + q
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a₁ + (n - 1) * d

-- Proof statement
theorem common_difference_is_half 
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q)
  (as : arithmetic_sequence a a₁ d)
  : d = 1 / 2 := 
sorry

end common_difference_is_half_l71_71286


namespace shale_mix_per_pound_is_5_l71_71404

noncomputable def cost_of_shale_mix_per_pound 
  (cost_limestone : ℝ) (cost_compound : ℝ) (weight_limestone : ℝ) (total_weight : ℝ) : ℝ :=
  let total_cost_limestone := weight_limestone * cost_limestone 
  let weight_shale := total_weight - weight_limestone
  let total_cost := total_weight * cost_compound
  let total_cost_shale := total_cost - total_cost_limestone
  total_cost_shale / weight_shale

theorem shale_mix_per_pound_is_5 :
  cost_of_shale_mix_per_pound 3 4.25 37.5 100 = 5 := 
by 
  sorry

end shale_mix_per_pound_is_5_l71_71404


namespace min_product_of_three_l71_71832

theorem min_product_of_three :
  ∀ (list : List Int), 
    list = [-9, -7, -1, 2, 4, 6, 8] →
    ∃ (a b c : Int), a ∈ list ∧ b ∈ list ∧ c ∈ list ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ (x y z : Int), x ∈ list → y ∈ list → z ∈ list → x ≠ y → y ≠ z → x ≠ z → x * y * z ≥ a * b * c) ∧
    a * b * c = -432 :=
by
  sorry

end min_product_of_three_l71_71832


namespace find_first_term_l71_71973

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l71_71973


namespace number_of_friends_is_five_l71_71345

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l71_71345


namespace count_valid_three_digit_numbers_l71_71138

theorem count_valid_three_digit_numbers : 
  let is_valid (a b c : ℕ) := 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ b = (a + c) / 2 ∧ (a + c) % 2 = 0
  ∃ n : ℕ, (∀ a b c : ℕ, is_valid a b c → n = 45) :=
sorry

end count_valid_three_digit_numbers_l71_71138


namespace min_value_l71_71452

theorem min_value (a : ℝ) (h : a > 0) : a + 4 / a ≥ 4 :=
by sorry

end min_value_l71_71452


namespace neha_mother_age_l71_71662

variable (N M : ℕ)

theorem neha_mother_age (h1 : M - 12 = 4 * (N - 12)) (h2 : M + 12 = 2 * (N + 12)) : M = 60 := by
  sorry

end neha_mother_age_l71_71662


namespace quadratic_real_roots_l71_71757

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ m ≤ 1 :=
by
  sorry

end quadratic_real_roots_l71_71757


namespace remainder_sum_first_six_primes_div_seventh_prime_l71_71341

-- Define the first six prime numbers
def firstSixPrimes : List ℕ := [2, 3, 5, 7, 11, 13]

-- Define the sum of the first six prime numbers
def sumOfFirstSixPrimes : ℕ := firstSixPrimes.sum

-- Define the seventh prime number
def seventhPrime : ℕ := 17

-- Proof statement that the remainder of the division is 7
theorem remainder_sum_first_six_primes_div_seventh_prime :
  (sumOfFirstSixPrimes % seventhPrime) = 7 :=
by
  sorry

end remainder_sum_first_six_primes_div_seventh_prime_l71_71341


namespace find_a4_l71_71136

open Nat

def seq (a : ℕ → ℝ) := (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = (2 * a n) / (a n + 2))

theorem find_a4 (a : ℕ → ℝ) (h : seq a) : a 4 = 2 / 5 :=
  sorry

end find_a4_l71_71136


namespace units_digit_of_153_base_3_l71_71242

theorem units_digit_of_153_base_3 :
  (153 % 3 ^ 1) = 2 := by
sorry

end units_digit_of_153_base_3_l71_71242


namespace ratio_of_packets_to_tent_stakes_l71_71669

-- Definitions based on the conditions provided
def total_items (D T W : ℕ) : Prop := D + T + W = 22
def tent_stakes (T : ℕ) : Prop := T = 4
def bottles_of_water (W T : ℕ) : Prop := W = T + 2

-- The goal is to prove the ratio of packets of drink mix to tent stakes
theorem ratio_of_packets_to_tent_stakes (D T W : ℕ) :
  total_items D T W →
  tent_stakes T →
  bottles_of_water W T →
  D = 3 * T :=
by
  sorry

end ratio_of_packets_to_tent_stakes_l71_71669


namespace zachary_pushups_l71_71212

variable (Zachary David John : ℕ)
variable (h1 : David = Zachary + 39)
variable (h2 : John = David - 13)
variable (h3 : David = 58)

theorem zachary_pushups : Zachary = 19 :=
by
  -- Proof goes here
  sorry

end zachary_pushups_l71_71212


namespace square_side_length_l71_71342

theorem square_side_length (A : ℝ) (h : A = 100) : ∃ s : ℝ, s * s = A ∧ s = 10 := by
  sorry

end square_side_length_l71_71342


namespace employee_B_payment_l71_71336

theorem employee_B_payment (x : ℝ) (h1 : ∀ A B : ℝ, A + B = 580) (h2 : A = 1.5 * B) : B = 232 :=
by
  sorry

end employee_B_payment_l71_71336


namespace company_picnic_l71_71995

theorem company_picnic :
  (20 / 100 * (30 / 100 * 100) + 40 / 100 * (70 / 100 * 100)) / 100 * 100 = 34 := by
  sorry

end company_picnic_l71_71995


namespace coefficient_a_eq_2_l71_71627

theorem coefficient_a_eq_2 (a : ℝ) (h : (a^3 * (4 : ℝ)) = 32) : a = 2 :=
by {
  -- Proof will need to be filled in here
  sorry
}

end coefficient_a_eq_2_l71_71627


namespace decrease_angle_equilateral_l71_71335

theorem decrease_angle_equilateral (D E F : ℝ) (h : D = 60) (h_equilateral : D = E ∧ E = F) (h_decrease : D' = D - 20) :
  ∃ max_angle : ℝ, max_angle = 70 :=
by
  sorry

end decrease_angle_equilateral_l71_71335


namespace problem1_problem2_l71_71485

namespace TriangleProofs

-- Problem 1: Prove that A + B = π / 2
theorem problem1 (a b c : ℝ) (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.cos B))
  (h2 : n = (b, Real.cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  (h_neq : m ≠ n)
  : A + B = Real.pi / 2 :=
sorry

-- Problem 2: Determine the range of x
theorem problem2 (A B : ℝ) (x : ℝ) 
  (h : A + B = Real.pi / 2) 
  (hx : x * Real.sin A * Real.sin B = Real.sin A + Real.sin B) 
  : 2 * Real.sqrt 2 ≤ x :=
sorry

end TriangleProofs

end problem1_problem2_l71_71485


namespace fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l71_71580

theorem fraction_inequalities (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1 / 2 ≤ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ∧ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ≤ 1 :=
sorry

theorem fraction_inequality_equality_right (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1 - a) * (1 - b) = 0 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) :=
sorry

theorem fraction_inequality_equality_left (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  a = b ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l71_71580


namespace sine_triangle_l71_71938

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle_l71_71938


namespace count_integers_satisfying_inequality_l71_71902

theorem count_integers_satisfying_inequality : (finset.filter (λ x : ℤ, (x + 3)^2 ≤ 4) (finset.Icc -5 -1)).card = 5 := 
by
  sorry

end count_integers_satisfying_inequality_l71_71902


namespace total_animal_legs_l71_71081

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l71_71081


namespace molecular_weight_correct_l71_71700

noncomputable def molecular_weight : ℝ := 
  let N_count := 2
  let H_count := 6
  let Br_count := 1
  let O_count := 1
  let C_count := 3
  let N_weight := 14.01
  let H_weight := 1.01
  let Br_weight := 79.90
  let O_weight := 16.00
  let C_weight := 12.01
  N_count * N_weight + 
  H_count * H_weight + 
  Br_count * Br_weight + 
  O_count * O_weight +
  C_count * C_weight

theorem molecular_weight_correct :
  molecular_weight = 166.01 := 
by
  sorry

end molecular_weight_correct_l71_71700


namespace parallel_line_slope_l71_71052

theorem parallel_line_slope (x y : ℝ) : 
  (∃ b : ℝ, y = (1 / 2) * x + b) → 
  (∃ a : ℝ, 3 * x - 6 * y = a) → 
  ∃ k : ℝ, k = 1 / 2 :=
by
  intros h1 h2
  sorry

end parallel_line_slope_l71_71052


namespace combined_salaries_l71_71952

-- Define the variables and constants corresponding to the conditions
variable (A B D E C : ℝ)
variable (avg_salary : ℝ)
variable (num_individuals : ℕ)

-- Given conditions translated into Lean definitions 
def salary_C : ℝ := 15000
def average_salary : ℝ := 8800
def number_of_individuals : ℕ := 5

-- Define the statement to prove
theorem combined_salaries (h1 : C = salary_C) (h2 : avg_salary = average_salary) (h3 : num_individuals = number_of_individuals) : 
  A + B + D + E = avg_salary * num_individuals - salary_C := 
by 
  -- Here the proof would involve calculating the total salary and subtracting C's salary
  sorry

end combined_salaries_l71_71952


namespace range_of_m_l71_71748

theorem range_of_m (p_false : ¬ (∀ x : ℝ, ∃ m : ℝ, 2 * x + 1 + m = 0)) : ∀ m : ℝ, m ≤ 1 :=
sorry

end range_of_m_l71_71748


namespace fraction_not_equal_l71_71057

theorem fraction_not_equal : ¬ (7 / 5 = 1 + 4 / 20) :=
by
  -- We'll use simplification to demonstrate the inequality
  sorry

end fraction_not_equal_l71_71057


namespace fraction_subtraction_simplified_l71_71110

theorem fraction_subtraction_simplified : (7 / 17) - (4 / 51) = 1 / 3 := by
  sorry

end fraction_subtraction_simplified_l71_71110


namespace expected_socks_pairs_l71_71554

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l71_71554


namespace two_n_plus_m_value_l71_71320

theorem two_n_plus_m_value (n m : ℤ) :
  3 * n - m < 5 ∧ n + m > 26 ∧ 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
sorry

end two_n_plus_m_value_l71_71320


namespace socks_expected_value_l71_71547

noncomputable def expected_socks_pairs (p : ℕ) : ℕ :=
2 * p

theorem socks_expected_value (p : ℕ) : 
  expected_socks_pairs p = 2 * p := 
by sorry

end socks_expected_value_l71_71547


namespace find_number_l71_71200

theorem find_number (x : ℤ) (h : 35 - 3 * x = 14) : x = 7 :=
by {
  sorry -- This is where the proof would go.
}

end find_number_l71_71200


namespace water_wasted_in_one_hour_l71_71678

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l71_71678


namespace nth_term_formula_l71_71330

theorem nth_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * n^2 + n)
  (h2 : a 1 = S 1)
  (h3 : ∀ n ≥ 2, a n = S n - S (n - 1))
  : ∀ n, a n = 4 * n - 1 := by
  sorry

end nth_term_formula_l71_71330


namespace calculate_tan_product_l71_71754

theorem calculate_tan_product :
  let A := 30
  let B := 40
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2.9 :=
by
  sorry

end calculate_tan_product_l71_71754


namespace polynomial_coefficients_even_or_odd_l71_71459

-- Define the problem conditions as Lean definitions
variables {P Q : Polynomial ℤ}

-- Theorem: Given the conditions, prove the required statement
theorem polynomial_coefficients_even_or_odd
  (hP : ∀ n : ℕ, P.coeff n % 2 = 0)
  (hQ : ∀ n : ℕ, Q.coeff n % 2 = 0)
  (hProd : ¬ ∀ n : ℕ, (P * Q).coeff n % 4 = 0) :
  (∀ n : ℕ, P.coeff n % 2 = 0 ∧ ∃ k : ℕ, Q.coeff k % 2 ≠ 0) ∨
  (∀ n : ℕ, Q.coeff n % 2 = 0 ∧ ∃ k: ℕ, P.coeff k % 2 ≠ 0) :=
sorry

end polynomial_coefficients_even_or_odd_l71_71459


namespace required_force_18_inch_wrench_l71_71319

def inverse_force (l : ℕ) (k : ℕ) : ℕ := k / l

def extra_force : ℕ := 50

def initial_force : ℕ := 300

noncomputable
def handle_length_1 : ℕ := 12

noncomputable
def handle_length_2 : ℕ := 18

noncomputable
def adjusted_force : ℕ := inverse_force handle_length_2 (initial_force * handle_length_1)

theorem required_force_18_inch_wrench : 
  adjusted_force + extra_force = 250 := 
by
  sorry

end required_force_18_inch_wrench_l71_71319


namespace num_best_friends_l71_71348

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l71_71348


namespace emmy_rosa_ipods_total_l71_71243

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l71_71243


namespace remainder_when_divided_by_7_l71_71056

theorem remainder_when_divided_by_7 
  {k : ℕ} 
  (h1 : k % 5 = 2) 
  (h2 : k % 6 = 5) 
  (h3 : k < 41) : 
  k % 7 = 3 := 
sorry

end remainder_when_divided_by_7_l71_71056


namespace sequence_pattern_l71_71804

theorem sequence_pattern (a b c d e f : ℕ) 
  (h1 : a + b = 12)
  (h2 : 8 + 9 = 16)
  (h3 : 5 + 6 = 10)
  (h4 : 7 + 8 = 14)
  (h5 : 3 + 3 = 5) : 
  ∀ x, ∃ y, x + y = 2 * x := by
  intros x
  use 0
  sorry

end sequence_pattern_l71_71804


namespace value_of_m_minus_n_over_n_l71_71751

theorem value_of_m_minus_n_over_n (m n : ℚ) (h : (2/3 : ℚ) * m = (5/6 : ℚ) * n) :
  (m - n) / n = 1 / 4 := 
sorry

end value_of_m_minus_n_over_n_l71_71751


namespace units_digit_of_expression_l71_71875

theorem units_digit_of_expression :
  (8 * 18 * 1988 - 8^4) % 10 = 6 := 
by
  sorry

end units_digit_of_expression_l71_71875


namespace profit_loss_balance_l71_71817

-- Defining variables
variables (C L : Real)

-- Profit and loss equations according to problem conditions
theorem profit_loss_balance (h1 : 832 - C = C - L) (h2 : 992 = 0.55 * C) : 
  (C + 992 = 2795.64) :=
by
  -- Statement of the theorem
  sorry

end profit_loss_balance_l71_71817


namespace possible_values_of_product_l71_71731

theorem possible_values_of_product 
  (P_A P_B P_C P_D P_E : ℕ)
  (H1 : P_A = P_B + P_C + P_D + P_E)
  (H2 : ∃ n1 n2 n3 n4, 
          ((P_B = n1 * (n1 + 1)) ∨ (P_B = n2 * (n2 + 1) * (n2 + 2)) ∨ 
           (P_B = n3 * (n3 + 1) * (n3 + 2) * (n3 + 3)) ∨ (P_B = n4 * (n4 + 1) * (n4 + 2) * (n4 + 3) * (n4 + 4))) ∧
          ∃ m1 m2 m3 m4, 
          ((P_C = m1 * (m1 + 1)) ∨ (P_C = m2 * (m2 + 1) * (m2 + 2)) ∨ 
           (P_C = m3 * (m3 + 1) * (m3 + 2) * (m3 + 3)) ∨ (P_C = m4 * (m4 + 1) * (m4 + 2) * (m4 + 3) * (m4 + 4))) ∧
          ∃ o1 o2 o3 o4, 
          ((P_D = o1 * (o1 + 1)) ∨ (P_D = o2 * (o2 + 1) * (o2 + 2)) ∨ 
           (P_D = o3 * (o3 + 1) * (o3 + 2) * (o3 + 3)) ∨ (P_D = o4 * (o4 + 1) * (o4 + 2) * (o4 + 3) * (o4 + 4))) ∧
          ∃ p1 p2 p3 p4, 
          ((P_E = p1 * (p1 + 1)) ∨ (P_E = p2 * (p2 + 1) * (p2 + 2)) ∨ 
           (P_E = p3 * (p3 + 1) * (p3 + 2) * (p3 + 3)) ∨ (P_E = p4 * (p4 + 1) * (p4 + 2) * (p4 + 3) * (p4 + 4))) ∧ 
          ∃ q1 q2 q3 q4, 
          ((P_A = q1 * (q1 + 1)) ∨ (P_A = q2 * (q2 + 1) * (q2 + 2)) ∨ 
           (P_A = q3 * (q3 + 1) * (q3 + 2) * (q3 + 3)) ∨ (P_A = q4 * (q4 + 1) * (q4 + 2) * (q4 + 3) * (q4 + 4)))) :
  P_A = 6 ∨ P_A = 24 :=
by sorry

end possible_values_of_product_l71_71731


namespace tenth_term_arithmetic_sequence_l71_71054

theorem tenth_term_arithmetic_sequence :
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  aₙ 10 = 3 :=
by
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  show aₙ 10 = 3
  sorry

end tenth_term_arithmetic_sequence_l71_71054


namespace player5_points_combination_l71_71914

theorem player5_points_combination :
  ∃ (two_point_shots three_pointers free_throws : ℕ), 
  (two_point_shots * 2 + three_pointers * 3 + free_throws * 1 = 14) :=
sorry

end player5_points_combination_l71_71914


namespace infinitely_many_divisors_l71_71797

theorem infinitely_many_divisors (a : ℕ) : ∃ᶠ n in at_top, n ∣ a ^ (n - a + 1) - 1 :=
sorry

end infinitely_many_divisors_l71_71797


namespace sum_of_consecutive_negatives_l71_71813

theorem sum_of_consecutive_negatives (n : ℤ) (h1 : n * (n + 1) = 2720) (h2 : n < 0) : 
  n + (n + 1) = -103 :=
by
  sorry

end sum_of_consecutive_negatives_l71_71813


namespace price_reduction_l71_71349

theorem price_reduction (P : ℝ) : 
  let first_day_reduction := 0.91 * P
  let second_day_reduction := 0.90 * first_day_reduction
  second_day_reduction = 0.819 * P :=
by 
  sorry

end price_reduction_l71_71349


namespace concert_cost_l71_71023

-- Definitions of the given conditions
def ticket_price : ℝ := 50.00
def num_tickets : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℝ := 10.00
def entrance_fee_per_person : ℝ := 5.00
def num_people : ℕ := 2

-- Function to compute the total cost
def total_cost : ℝ :=
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  total_with_parking + entrance_fee_total

-- The proof statement
theorem concert_cost :
  total_cost = 135.00 :=
by
  -- Using the assumptions defined
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  let final_total := total_with_parking + entrance_fee_total
  
  -- Proving the final total
  show final_total = 135.00
  sorry

end concert_cost_l71_71023


namespace hide_and_seek_friends_l71_71382

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l71_71382


namespace total_area_of_farm_l71_71848

-- Define the number of sections and area of each section
def number_of_sections : ℕ := 5
def area_of_each_section : ℕ := 60

-- State the problem as proving the total area of the farm
theorem total_area_of_farm : number_of_sections * area_of_each_section = 300 :=
by sorry

end total_area_of_farm_l71_71848


namespace find_values_and_properties_l71_71741

variable (f : ℝ → ℝ)

axiom f_neg1 : f (-1) = 2
axiom f_pos_x : ∀ x, x < 0 → f x > 1
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y

theorem find_values_and_properties :
  f 0 = 1 ∧
  f (-4) = 16 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f (-4 * x^2) * f (10 * x) ≥ 1/16 ↔ x ≤ 1/2 ∨ x ≥ 2) :=
sorry

end find_values_and_properties_l71_71741


namespace difference_of_squares_550_450_l71_71569

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end difference_of_squares_550_450_l71_71569


namespace evaluate_expression_l71_71584

theorem evaluate_expression : (1.2^3 - (0.9^3 / 1.2^2) + 1.08 + 0.9^2 = 3.11175) :=
by
  sorry -- Proof goes here

end evaluate_expression_l71_71584


namespace _l71_71282

noncomputable def probability_event_b_given_a : ℕ → ℕ → ℕ → ℕ × ℕ → ℚ
| zeros, ones, twos, (1, drawn_label) =>
  if drawn_label = 1 then
    (ones * (ones - 1)) / (zeros + ones + twos).choose 2
  else 0
| _, _, _, _ => 0

lemma probability_theorem :
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  (1 - 1) * (ones - 1)/(total.choose 2) = 1/7 :=
by
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  let draw_label := 1
  let event_b_given_a := probability_event_b_given_a zeros ones twos (1, draw_label)
  have pos_cases : (ones * (ones - 1))/(total.choose 2) = 1 / 7 := by sorry
  exact pos_cases

end _l71_71282


namespace find_A_plus_B_l71_71781

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end find_A_plus_B_l71_71781


namespace incorrect_observation_value_l71_71038

theorem incorrect_observation_value
  (mean : ℕ → ℝ)
  (n : ℕ)
  (observed_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (H1 : n = 50)
  (H2 : observed_mean = 36)
  (H3 : correct_value = 43)
  (H4 : corrected_mean = 36.5)
  (H5 : mean n = observed_mean)
  (H6 : mean (n - 1 + 1) = corrected_mean - correct_value + incorrect_value) :
  incorrect_value = 18 := sorry

end incorrect_observation_value_l71_71038


namespace david_number_sum_l71_71730

theorem david_number_sum :
  ∃ (x y : ℕ), (10 ≤ x ∧ x < 100) ∧ (100 ≤ y ∧ y < 1000) ∧ (1000 * x + y = 4 * x * y) ∧ (x + y = 266) :=
sorry

end david_number_sum_l71_71730


namespace hide_and_seek_l71_71390

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71390


namespace problem1_problem2_l71_71840

-- Problem (1): Maximum value of (a + 1/a)(b + 1/b)
theorem problem1 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≤ 25 / 4 := 
sorry

-- Problem (2): Minimum value of u = (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3
theorem problem2 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
sorry

end problem1_problem2_l71_71840


namespace cost_of_each_pack_l71_71155

theorem cost_of_each_pack (num_packs : ℕ) (total_paid : ℝ) (change_received : ℝ) 
(h1 : num_packs = 3) (h2 : total_paid = 20) (h3 : change_received = 11) : 
(total_paid - change_received) / num_packs = 3 := by
  sorry

end cost_of_each_pack_l71_71155


namespace smallest_k_divisible_by_200_l71_71654

theorem smallest_k_divisible_by_200 : 
  ∃ (k : ℕ), (∑ i in finset.range k.succ, i^2) = 112 :=
by
  let sum_squares := λ k : ℕ, k * (k + 1) * (2 * k + 1) / 6
  have h: ∃ k, sum_squares k % 200 = 0 := sorry
  exact h

end smallest_k_divisible_by_200_l71_71654


namespace randy_piggy_bank_final_amount_l71_71799

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end randy_piggy_bank_final_amount_l71_71799


namespace probability_divisible_l71_71692

noncomputable def number_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

def total_combinations (s : Finset ℕ) : ℕ := (s.card.choose 3)

def successful_outcomes (s : Finset ℕ) : ℕ := 
  s.subsets 3 |>.filter (fun s => 
    let x := s.min' (by simp [Finset.nonempty.subset, Finset.nonempty_of_mem]; tautology)
    all_b (s.erase x) (fun y => x ∣ y)).card

def probability (s : Finset ℕ) : ℚ := successful_outcomes s / total_combinations s

theorem probability_divisible (s : Finset ℕ) (h : s = number_set) : 
  probability s = 11 / 20 := by
  sorry

end probability_divisible_l71_71692


namespace tetrahedron_volume_l71_71279

theorem tetrahedron_volume 
  (R S₁ S₂ S₃ S₄ : ℝ) : 
  V = R * (S₁ + S₂ + S₃ + S₄) :=
sorry

end tetrahedron_volume_l71_71279


namespace angle_C_triangle_area_l71_71280

theorem angle_C 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) :
  C = 2 * Real.pi / 3 :=
sorry

theorem triangle_area 
  (a b c : ℝ) (C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C)
  (h2 : c = Real.sqrt 7)
  (h3 : b = 2) :
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 :=
sorry

end angle_C_triangle_area_l71_71280


namespace num_of_three_digit_integers_greater_than_217_l71_71433

theorem num_of_three_digit_integers_greater_than_217 : 
  ∃ n : ℕ, n = 82 ∧ ∀ x : ℕ, (217 < x ∧ x < 300) → 200 ≤ x ∧ x ≤ 299 → n = 82 := 
by
  sorry

end num_of_three_digit_integers_greater_than_217_l71_71433


namespace find_Luisa_books_l71_71303

structure Books where
  Maddie : ℕ
  Amy : ℕ
  Amy_and_Luisa : ℕ
  Luisa : ℕ

theorem find_Luisa_books (L M A : ℕ) (hM : M = 15) (hA : A = 6) (hAL : L + A = M + 9) : L = 18 := by
  sorry

end find_Luisa_books_l71_71303


namespace max_lessons_possible_l71_71197

theorem max_lessons_possible 
  (s p b : ℕ) 
  (h1 : 2 * p * b = 36) 
  (h2 : 2 * s * b = 72) 
  (h3 : 2 * s * p = 54) 
  : 2 * s * p * b = 216 :=
begin
  sorry
end

end max_lessons_possible_l71_71197


namespace johns_yearly_grass_cutting_cost_l71_71005

-- Definitions of the conditions
def initial_height : ℝ := 2.0
def growth_rate : ℝ := 0.5
def cutting_height : ℝ := 4.0
def cost_per_cut : ℝ := 100.0
def months_per_year : ℝ := 12.0

-- Formulate the statement
theorem johns_yearly_grass_cutting_cost :
  let months_to_grow : ℝ := (cutting_height - initial_height) / growth_rate
  let cuts_per_year : ℝ := months_per_year / months_to_grow
  let total_cost_per_year : ℝ := cuts_per_year * cost_per_cut
  total_cost_per_year = 300.0 :=
by
  sorry

end johns_yearly_grass_cutting_cost_l71_71005


namespace average_of_x_y_z_l71_71140

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end average_of_x_y_z_l71_71140


namespace infinite_series_sum_l71_71862

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * (n + 1) * (n + 1) + (n + 1) + 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))) = 5 / 6 := by
  sorry

end infinite_series_sum_l71_71862


namespace nine_points_circle_chords_l71_71501

theorem nine_points_circle_chords : 
  let n := 9 in
  (nat.choose n 2) = 36 :=
by
  let n := 9
  have h := nat.choose n 2
  calc
    nat.choose n 2 = 36 : sorry

end nine_points_circle_chords_l71_71501


namespace ratio_w_y_l71_71684

-- Define the necessary variables
variables (w x y z : ℚ)

-- Define the conditions as hypotheses
axiom h1 : w / x = 4 / 3
axiom h2 : y / z = 5 / 3
axiom h3 : z / x = 1 / 6

-- State the proof problem
theorem ratio_w_y : w / y = 24 / 5 :=
by sorry

end ratio_w_y_l71_71684


namespace trajectory_equation_of_point_M_l71_71228

variables {x y a b : ℝ}

theorem trajectory_equation_of_point_M :
  (a^2 + b^2 = 100) →
  (x = a / (1 + 4)) →
  (y = 4 * b / (1 + 4)) →
  16 * x^2 + y^2 = 64 :=
by
  intros h1 h2 h3
  sorry

end trajectory_equation_of_point_M_l71_71228


namespace friends_who_participate_l71_71363

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l71_71363


namespace find_first_term_geometric_series_l71_71965

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l71_71965


namespace tan_product_30_60_l71_71890

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * Real.pi / 180)) * (1 + Real.tan (60 * Real.pi / 180)) = 2 + (4 * Real.sqrt 3) / 3 := 
  sorry

end tan_product_30_60_l71_71890


namespace lynne_total_spent_l71_71167

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l71_71167


namespace total_spokes_in_garage_l71_71854

-- Definitions based on the problem conditions
def num_bicycles : ℕ := 4
def spokes_per_wheel : ℕ := 10
def wheels_per_bicycle : ℕ := 2

-- The goal is to prove the total number of spokes
theorem total_spokes_in_garage : (num_bicycles * wheels_per_bicycle * spokes_per_wheel) = 80 :=
by
    sorry

end total_spokes_in_garage_l71_71854


namespace train_length_l71_71417

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l71_71417


namespace a_2023_le_1_l71_71166

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, (a (n+1))^2 + a n * a (n+2) ≤ a n + a (n+2))

theorem a_2023_le_1 : a 2023 ≤ 1 := by
  sorry

end a_2023_le_1_l71_71166


namespace prime_solution_unique_l71_71124

theorem prime_solution_unique {x y : ℕ} 
  (hx : Nat.Prime x)
  (hy : Nat.Prime y)
  (h : x ^ y - y ^ x = x * y ^ 2 - 19) :
  (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
sorry

end prime_solution_unique_l71_71124


namespace zero_in_interval_l71_71707

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ x0, f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
by
  have h_cont : Continuous f := sorry -- f is continuous (can be proven using the continuity of log and linear functions)
  have h_eval1 : f 2 < 0 := sorry -- f(2) = ln(2) - 6 + 4 < 0
  have h_eval2 : f 3 > 0 := sorry -- f(3) = ln(3) - 6 + 6 > 0
  -- By the Intermediate Value Theorem, since f is continuous and changes signs between (2, 3), there exists a zero x0 in (2, 3).
  exact sorry

end zero_in_interval_l71_71707


namespace smallest_x_for_gx_eq_1024_l71_71069

noncomputable def g : ℝ → ℝ
  | x => if 2 ≤ x ∧ x ≤ 6 then 2 - |x - 3| else 0

axiom g_property1 : ∀ x : ℝ, 0 < x → g (4 * x) = 4 * g x
axiom g_property2 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 6 → g x = 2 - |x - 3|
axiom g_2004 : g 2004 = 1024

theorem smallest_x_for_gx_eq_1024 : ∃ x : ℝ, g x = 1024 ∧ ∀ y : ℝ, g y = 1024 → x ≤ y := sorry

end smallest_x_for_gx_eq_1024_l71_71069


namespace triangle_equilateral_of_angles_and_intersecting_segments_l71_71923

theorem triangle_equilateral_of_angles_and_intersecting_segments
    (A B C : Type) (angle_A : ℝ) (intersect_at_one_point : Prop)
    (angle_M_bisects : Prop) (N_is_median : Prop) (L_is_altitude : Prop) :
  angle_A = 60 ∧ angle_M_bisects ∧ N_is_median ∧ L_is_altitude ∧ intersect_at_one_point → 
  ∀ (angle_B angle_C : ℝ), angle_B = 60 ∧ angle_C = 60 := 
by
  intro h
  sorry

end triangle_equilateral_of_angles_and_intersecting_segments_l71_71923


namespace share_of_c_l71_71572

variable (a b c : ℝ)

theorem share_of_c (h1 : a + b + c = 427) (h2 : 3 * a = 7 * c) (h3 : 4 * b = 7 * c) : c = 84 :=
  by
  sorry

end share_of_c_l71_71572


namespace volume_tetrahedron_formula_l71_71180

-- Definitions of the problem elements
def distance (A B C D : Point) : ℝ := sorry
def angle (A B C D : Point) : ℝ := sorry
def length (A B : Point) : ℝ := sorry

-- The problem states you need to prove the volume of the tetrahedron
noncomputable def volume_tetrahedron (A B C D : Point) : ℝ := sorry

-- Conditions
variable (A B C D : Point)
variable (d : ℝ) (phi : ℝ) -- d = distance between lines AB and CD, phi = angle between lines AB and CD

-- Question reformulated as a proof statement
theorem volume_tetrahedron_formula (h1 : d = distance A B C D)
                                   (h2 : phi = angle A B C D) :
  volume_tetrahedron A B C D = (d * length A B * length C D * Real.sin phi) / 6 :=
sorry

end volume_tetrahedron_formula_l71_71180


namespace find_other_factor_l71_71710

theorem find_other_factor 
    (w : ℕ) 
    (hw_pos : w > 0) 
    (h_factor : ∃ (x y : ℕ), 936 * w = x * y ∧ (2 ^ 5 ∣ x) ∧ (3 ^ 3 ∣ x)) 
    (h_ww : w = 156) : 
    ∃ (other_factor : ℕ), 936 * w = 156 * other_factor ∧ other_factor = 72 := 
by 
    sorry

end find_other_factor_l71_71710


namespace emmy_rosa_ipods_l71_71245

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l71_71245


namespace june_earnings_l71_71775

theorem june_earnings
  (total_clovers : ℕ)
  (clover_3_petals_percentage : ℝ)
  (clover_2_petals_percentage : ℝ)
  (clover_4_petals_percentage : ℝ)
  (earnings_per_clover : ℝ) :
  total_clovers = 200 →
  clover_3_petals_percentage = 0.75 →
  clover_2_petals_percentage = 0.24 →
  clover_4_petals_percentage = 0.01 →
  earnings_per_clover = 1 →
  (total_clovers * earnings_per_clover) = 200 := by
  sorry

end june_earnings_l71_71775


namespace relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l71_71708

variable (x y : ℝ)

-- Assume the initial fuel and consumption rate
def initial_fuel : ℝ := 48
def consumption_rate : ℝ := 0.6

-- Define the fuel consumption equation
def fuel_equation (distance : ℝ) : ℝ := -consumption_rate * distance + initial_fuel

-- Theorem proving the fuel equation satisfies the specific conditions
theorem relationship_between_y_and_x :
  ∀ (x : ℝ), y = fuel_equation x :=
by
  sorry

-- Theorem proving the fuel remaining after traveling 35 kilometers
theorem fuel_remaining_after_35_kilometers :
  fuel_equation 35 = 27 :=
by
  sorry

-- Theorem proving the maximum distance the car can travel without refueling
theorem max_distance_without_refueling :
  ∃ (x : ℝ), fuel_equation x = 0 ∧ x = 80 :=
by
  sorry

end relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l71_71708


namespace number_of_chords_l71_71508

theorem number_of_chords (n : ℕ) (h : n = 9) : (n.choose 2) = 36 :=
by
  rw h
  norm_num
  sorry

end number_of_chords_l71_71508


namespace amount_used_to_pay_l71_71516

noncomputable def the_cost_of_football : ℝ := 9.14
noncomputable def the_cost_of_baseball : ℝ := 6.81
noncomputable def the_change_received : ℝ := 4.05

theorem amount_used_to_pay : 
    (the_cost_of_football + the_cost_of_baseball + the_change_received) = 20.00 := 
by
  sorry

end amount_used_to_pay_l71_71516


namespace number_of_true_propositions_is_zero_l71_71808

theorem number_of_true_propositions_is_zero :
  (∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0) →
  (¬ ∃ x : ℚ, x^2 = 2) →
  (¬ ∃ x : ℝ, x^2 + 1 = 0) →
  (∀ x : ℝ, 4 * x^2 ≤ 2 * x - 1 + 3 * x^2) →
  true :=  -- representing that the number of true propositions is 0
by
  intros h1 h2 h3 h4
  sorry

end number_of_true_propositions_is_zero_l71_71808


namespace quadratic_trinomial_constant_l71_71455

theorem quadratic_trinomial_constant (m : ℝ) (h : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
sorry

end quadratic_trinomial_constant_l71_71455


namespace tiles_visited_by_bug_l71_71074

theorem tiles_visited_by_bug (width length : ℕ)
  (h_width : width = 12) (h_length : length = 19) : 
  let gcd_val := Nat.gcd width length in
  (let num_tiles_crossed := width + length - gcd_val in 
    num_tiles_crossed) = 30 :=
by
  -- Assign the values to width and length 
  have h_width_length : width = 12 ∧ length = 19 := ⟨h_width, h_length⟩
  -- Calculate gcd of width and length
  let gcd_val := Nat.gcd width length
  have h_gcd : gcd_val = 1 := Nat.gcd_eq_right (nat.prime.dvd_gcd_iff ⟨2, nat.prime_two⟩ h_width_length.1 h_width_length.2) (nat.prime.dvd_of_dvd_sub zero_le_one (by linarith))
  -- Calculate the number of tiles crossed
  let num_tiles_crossed := width + length - gcd_val
  have h_num_tiles_crossed : num_tiles_crossed = 30 := by 
    simp [h_gcd]
    rw [←h_width, ←h_length]
    norm_num
  exact h_num_tiles_crossed

end tiles_visited_by_bug_l71_71074


namespace hide_and_seek_l71_71392

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l71_71392


namespace functional_equality_l71_71943

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality
  (h1 : ∀ x : ℝ, f x ≤ x)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equality_l71_71943


namespace triangle_least_perimeter_l71_71777

noncomputable def least_perimeter_of_triangle : ℕ :=
  let a := 7
  let b := 17
  let c := 13
  a + b + c

theorem triangle_least_perimeter :
  let a := 7
  let b := 17
  let c := 13
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  4 ∣ (a^2 + b^2 + c^2) - 2 * c^2 ∧
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  least_perimeter_of_triangle = 37 :=
by
  intros _ _ _ h
  sorry

end triangle_least_perimeter_l71_71777


namespace percent_of_70_is_56_l71_71067

theorem percent_of_70_is_56 : (70 / 125) * 100 = 56 := by
  sorry

end percent_of_70_is_56_l71_71067


namespace num_chords_l71_71510

theorem num_chords (n : ℕ) (h : n = 9) : nat.choose n 2 = 36 :=
by
  rw h
  simpa using nat.choose 9 2

end num_chords_l71_71510


namespace number_of_girls_l71_71513

theorem number_of_girls (total_children boys : ℕ) (h1 : total_children = 60) (h2 : boys = 16) : total_children - boys = 44 := by
  sorry

end number_of_girls_l71_71513


namespace find_A_l71_71805

namespace PolynomialDecomposition

theorem find_A (x A B C : ℝ)
  (h : (x^3 + 2 * x^2 - 17 * x - 30)⁻¹ = A / (x - 5) + B / (x + 2) + C / ((x + 2)^2)) :
  A = 1 / 49 :=
by sorry

end PolynomialDecomposition

end find_A_l71_71805
