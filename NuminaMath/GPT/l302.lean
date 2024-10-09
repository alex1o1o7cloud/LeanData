import Mathlib

namespace mass_percentage_O_mixture_l302_30244

noncomputable def molar_mass_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)
noncomputable def molar_mass_Cr2O3 : ℝ := (2 * 51.99) + (3 * 16.00)
noncomputable def mass_of_O_in_Al2O3 : ℝ := 3 * 16.00
noncomputable def mass_of_O_in_Cr2O3 : ℝ := 3 * 16.00
noncomputable def mass_percentage_O_in_Al2O3 : ℝ := (mass_of_O_in_Al2O3 / molar_mass_Al2O3) * 100
noncomputable def mass_percentage_O_in_Cr2O3 : ℝ := (mass_of_O_in_Cr2O3 / molar_mass_Cr2O3) * 100
noncomputable def mass_percentage_O_in_mixture : ℝ := (0.50 * mass_percentage_O_in_Al2O3) + (0.50 * mass_percentage_O_in_Cr2O3)

theorem mass_percentage_O_mixture : mass_percentage_O_in_mixture = 39.325 := by
  sorry

end mass_percentage_O_mixture_l302_30244


namespace deepak_present_age_l302_30219

def rahul_age (x : ℕ) : ℕ := 4 * x
def deepak_age (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age (x : ℕ) (h1 : rahul_age x + 10 = 26) : deepak_age x = 12 :=
by sorry

end deepak_present_age_l302_30219


namespace pipe_length_l302_30265

theorem pipe_length (S L : ℕ) (h1: S = 28) (h2: L = S + 12) : S + L = 68 := 
by
  sorry

end pipe_length_l302_30265


namespace clock_hand_overlaps_in_24_hours_l302_30285

-- Define the number of revolutions of the hour hand in 24 hours.
def hour_hand_revolutions_24_hours : ℕ := 2

-- Define the number of revolutions of the minute hand in 24 hours.
def minute_hand_revolutions_24_hours : ℕ := 24

-- Define the number of overlaps as a constant.
def number_of_overlaps (hour_rev : ℕ) (minute_rev : ℕ) : ℕ :=
  minute_rev - hour_rev

-- The theorem we want to prove:
theorem clock_hand_overlaps_in_24_hours :
  number_of_overlaps hour_hand_revolutions_24_hours minute_hand_revolutions_24_hours = 22 :=
sorry

end clock_hand_overlaps_in_24_hours_l302_30285


namespace move_line_up_l302_30223

theorem move_line_up (x : ℝ) :
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  y_moved = 4 * x + 1 :=
by
  let y_initial := 4 * x - 1
  let y_moved := y_initial + 2
  show y_moved = 4 * x + 1
  sorry

end move_line_up_l302_30223


namespace find_other_x_intercept_l302_30207

theorem find_other_x_intercept (a b c : ℝ) (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 4)^2 + 9)
  (h2 : a * 0^2 + b * 0 + c = 0) : ∃ x, x ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=
by
  sorry

end find_other_x_intercept_l302_30207


namespace angle_sum_proof_l302_30204

theorem angle_sum_proof (x α β : ℝ) (h1 : 3 * x + 4 * x + α = 180)
 (h2 : α + 5 * x + β = 180)
 (h3 : 2 * x + 2 * x + 6 * x = 180) :
  x = 18 := by
  sorry

end angle_sum_proof_l302_30204


namespace solve_abs_equation_l302_30210

-- Define the condition for the equation
def condition (x : ℝ) : Prop := 3 * x + 5 ≥ 0

-- The main theorem to prove that x = 1/5 is the only solution
theorem solve_abs_equation (x : ℝ) (h : condition x) : |2 * x - 6| = 3 * x + 5 ↔ x = 1 / 5 := by
  sorry

end solve_abs_equation_l302_30210


namespace min_5a2_plus_6a3_l302_30233

theorem min_5a2_plus_6a3 (a_1 a_2 a_3 : ℝ) (r : ℝ)
  (h1 : a_1 = 2)
  (h2 : a_2 = a_1 * r)
  (h3 : a_3 = a_1 * r^2) :
  5 * a_2 + 6 * a_3 ≥ -25 / 12 :=
by
  sorry

end min_5a2_plus_6a3_l302_30233


namespace fraction_second_year_not_third_year_l302_30271

theorem fraction_second_year_not_third_year (N T S : ℕ) (hN : N = 100) (hT : T = N / 2) (hS : S = N * 3 / 10) :
  (S / (N - T) : ℚ) = 3 / 5 :=
by
  rw [hN, hT, hS]
  norm_num
  sorry

end fraction_second_year_not_third_year_l302_30271


namespace rectangular_prism_diagonal_inequality_l302_30267

theorem rectangular_prism_diagonal_inequality 
  (a b c l : ℝ) 
  (h : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
by sorry

end rectangular_prism_diagonal_inequality_l302_30267


namespace part1_part2_l302_30299

def P (a : ℝ) := ∀ x : ℝ, x^2 - a * x + a + 5 / 4 > 0
def Q (a : ℝ) := 4 * a + 7 ≠ 0 ∧ a - 3 ≠ 0 ∧ (4 * a + 7) * (a - 3) < 0

theorem part1 (h : Q a) : -7 / 4 < a ∧ a < 3 := sorry

theorem part2 (h : (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) :
  (-7 / 4 < a ∧ a ≤ -1) ∨ (3 ≤ a ∧ a < 5) := sorry

end part1_part2_l302_30299


namespace Paul_work_time_l302_30225

def work_completed (rate: ℚ) (time: ℚ) : ℚ := rate * time

noncomputable def George_work_rate : ℚ := 3 / 5 / 9

noncomputable def combined_work_rate : ℚ := 2 / 5 / 4

noncomputable def Paul_work_rate : ℚ := combined_work_rate - George_work_rate

theorem Paul_work_time :
  (work_completed Paul_work_rate 30) = 1 :=
by
  have h_george_rate : George_work_rate = 1 / 15 :=
    by norm_num [George_work_rate]
  have h_combined_rate : combined_work_rate = 1 / 10 :=
    by norm_num [combined_work_rate]
  have h_paul_rate : Paul_work_rate = 1 / 30 :=
    by norm_num [Paul_work_rate, h_combined_rate, h_george_rate]
  sorry -- Complete proof statement here

end Paul_work_time_l302_30225


namespace innings_count_l302_30230

-- Definitions of the problem conditions
def total_runs (n : ℕ) : ℕ := 63 * n
def highest_score : ℕ := 248
def lowest_score : ℕ := 98

theorem innings_count (n : ℕ) (h : total_runs n - highest_score - lowest_score = 58 * (n - 2)) : n = 46 :=
  sorry

end innings_count_l302_30230


namespace geometric_progressions_sum_eq_l302_30208

variable {a q b : ℝ}
variable {n : ℕ}
variable (h1 : q ≠ 1)

/-- The given statement in Lean 4 -/
theorem geometric_progressions_sum_eq (h : a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1)) : 
  b = a * (1 + q + q^2) := 
by
  sorry

end geometric_progressions_sum_eq_l302_30208


namespace chord_length_of_intersection_l302_30205

def ellipse (x y : ℝ) := x^2 + 4 * y^2 = 16
def line (x y : ℝ) := y = (1/2) * x + 1

theorem chord_length_of_intersection :
  ∃ A B : ℝ × ℝ, ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧ line A.fst A.snd ∧ line B.fst B.snd ∧
  dist A B = Real.sqrt 35 :=
sorry

end chord_length_of_intersection_l302_30205


namespace find_a_equals_two_l302_30282

noncomputable def a := ((7 + 4 * Real.sqrt 3) ^ (1 / 2) - (7 - 4 * Real.sqrt 3) ^ (1 / 2)) / Real.sqrt 3

theorem find_a_equals_two : a = 2 := 
sorry

end find_a_equals_two_l302_30282


namespace sum_a_b_l302_30284

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem sum_a_b (a b : ℝ) 
  (H : ∀ x, 2 < x ∧ x < 3 → otimes (x - a) (x - b) > 0) : a + b = 4 :=
by
  sorry

end sum_a_b_l302_30284


namespace cos4_x_minus_sin4_x_l302_30294

theorem cos4_x_minus_sin4_x (x : ℝ) (h : x = π / 12) : (Real.cos x) ^ 4 - (Real.sin x) ^ 4 = (Real.sqrt 3) / 2 := by
  sorry

end cos4_x_minus_sin4_x_l302_30294


namespace even_function_phi_l302_30272

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def f' (x φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f x φ + f' x φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem even_function_phi :
  (∀ x : ℝ, y x φ = y (-x) φ) → ∃ k : ℤ, φ = -Real.pi / 3 + k * Real.pi :=
by
  sorry

end even_function_phi_l302_30272


namespace green_team_final_score_l302_30235

theorem green_team_final_score (G : ℕ) :
  (∀ G : ℕ, 68 = G + 29 → G = 39) :=
by
  sorry

end green_team_final_score_l302_30235


namespace probability_jammed_l302_30237

theorem probability_jammed (T τ : ℝ) (h : τ < T) : 
    (2 * τ / T - (τ / T) ^ 2) = (T^2 - (T - τ)^2) / T^2 := 
by
  sorry

end probability_jammed_l302_30237


namespace percentage_salt_l302_30246

-- Variables
variables {S1 S2 R : ℝ}

-- Conditions
def first_solution := S1
def second_solution := (25 / 100) * 19.000000000000007
def resulting_solution := 16

theorem percentage_salt (S1 S2 : ℝ) (H1: S2 = 19.000000000000007) 
(H2: (75 / 100) * S1 + (25 / 100) * S2 = 16) : 
S1 = 15 :=
by
    rw [H1] at H2
    sorry

end percentage_salt_l302_30246


namespace composite_function_increasing_l302_30260

variable {F : ℝ → ℝ}

/-- An odd function is a function that satisfies f(-x) = -f(x) for all x. -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is strictly increasing on negative values if it satisfies the given conditions. -/
def strictly_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 < x2 → x2 < 0 → f x1 < f x2

/-- Combining properties of an odd function and strictly increasing for negative inputs:
  We need to prove that the composite function is strictly increasing for positive inputs. -/
theorem composite_function_increasing (hf_odd : odd_function F)
    (hf_strict_inc_neg : strictly_increasing_on_neg F)
    : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → F (F x1) < F (F x2) :=
  sorry

end composite_function_increasing_l302_30260


namespace parabola_directrix_l302_30253

theorem parabola_directrix :
  ∀ (p : ℝ), (y^2 = 6 * x) → (x = -3/2) :=
by
  sorry

end parabola_directrix_l302_30253


namespace bags_with_chocolate_hearts_l302_30261

-- Definitions for given conditions
def total_candies : ℕ := 63
def total_bags : ℕ := 9
def candies_per_bag : ℕ := total_candies / total_bags
def chocolate_kiss_bags : ℕ := 3
def not_chocolate_candies : ℕ := 28
def bags_not_chocolate : ℕ := not_chocolate_candies / candies_per_bag
def remaining_bags : ℕ := total_bags - chocolate_kiss_bags - bags_not_chocolate

-- Statement to be proved
theorem bags_with_chocolate_hearts :
  remaining_bags = 2 := by 
  sorry

end bags_with_chocolate_hearts_l302_30261


namespace cost_of_paper_l302_30241

noncomputable def cost_of_paper_per_kg (edge_length : ℕ) (coverage_per_kg : ℕ) (expenditure : ℕ) : ℕ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_per_kg
  expenditure / paper_needed

theorem cost_of_paper (h1 : edge_length = 10) (h2 : coverage_per_kg = 20) (h3 : expenditure = 1800) : 
  cost_of_paper_per_kg 10 20 1800 = 60 :=
by
  -- Using the hypothesis to directly derive the result.
  unfold cost_of_paper_per_kg
  sorry

end cost_of_paper_l302_30241


namespace min_people_liking_both_l302_30226

theorem min_people_liking_both (A B C V : ℕ) (hA : A = 200) (hB : B = 150) (hC : C = 120) (hV : V = 80) :
  ∃ D, D = 80 ∧ D ≤ min B (A - C + V) :=
by {
  sorry
}

end min_people_liking_both_l302_30226


namespace chemical_x_percentage_l302_30258

-- Define the initial volume of the mixture
def initial_volume : ℕ := 80

-- Define the percentage of chemical x in the initial mixture
def percentage_x_initial : ℚ := 0.30

-- Define the volume of chemical x added to the mixture
def added_volume_x : ℕ := 20

-- Define the calculation of the amount of chemical x in the initial mixture
def initial_amount_x : ℚ := percentage_x_initial * initial_volume

-- Define the calculation of the total amount of chemical x after adding more
def total_amount_x : ℚ := initial_amount_x + added_volume_x

-- Define the calculation of the total volume after adding 20 liters of chemical x
def total_volume : ℚ := initial_volume + added_volume_x

-- Define the percentage of chemical x in the final mixture
def percentage_x_final : ℚ := (total_amount_x / total_volume) * 100

-- The proof goal
theorem chemical_x_percentage : percentage_x_final = 44 := 
by
  sorry

end chemical_x_percentage_l302_30258


namespace alex_jamie_casey_probability_l302_30214

-- Probability definitions and conditions
def alex_win_prob := 1/3
def casey_win_prob := 1/6
def jamie_win_prob := 1/2

def total_rounds := 8
def alex_wins := 4
def jamie_wins := 3
def casey_wins := 1

-- The probability computation
theorem alex_jamie_casey_probability : 
  alex_win_prob ^ alex_wins * jamie_win_prob ^ jamie_wins * casey_win_prob ^ casey_wins * (Nat.choose total_rounds (alex_wins + jamie_wins + casey_wins)) = 35 / 486 := 
sorry

end alex_jamie_casey_probability_l302_30214


namespace circle_m_condition_l302_30229

theorem circle_m_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + m = 0) → m < 5 :=
by
  sorry

end circle_m_condition_l302_30229


namespace sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l302_30221

theorem sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_β : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π / 4 := sorry

end sin_alpha_sqrt5_div5_and_sin_beta_sqrt10_div10_acute_sum_pi_div4_l302_30221


namespace curve_intersects_self_at_6_6_l302_30239

-- Definitions for the given conditions
def x (t : ℝ) : ℝ := t^2 - 3
def y (t : ℝ) : ℝ := t^4 - t^2 - 9 * t + 6

-- Lean statement stating that the curve intersects itself at the coordinate (6, 6)
theorem curve_intersects_self_at_6_6 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2 ∧ x t1 = 6 ∧ y t1 = 6 :=
by
  sorry

end curve_intersects_self_at_6_6_l302_30239


namespace find_value_of_triangle_l302_30211

theorem find_value_of_triangle (p : ℕ) (triangle : ℕ) 
  (h1 : triangle + p = 47) 
  (h2 : 3 * (triangle + p) - p = 133) :
  triangle = 39 :=
by 
  sorry

end find_value_of_triangle_l302_30211


namespace determine_n_l302_30297

theorem determine_n (n : ℕ) (h : n ≥ 2)
    (condition : ∀ i j : ℕ, i ≤ n → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) :
    ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 := 
sorry

end determine_n_l302_30297


namespace equivalent_lemons_l302_30220

theorem equivalent_lemons 
  (lemons_per_apple_approx : ∀ apples : ℝ, 3/4 * 14 = 9 → 1 = 9 / (3/4 * 14))
  (apples_to_lemons : ℝ) :
  5 / 7 * 7 = 30 / 7 :=
by
  sorry

end equivalent_lemons_l302_30220


namespace min_n_plus_d_l302_30255

theorem min_n_plus_d (a : ℕ → ℕ) (n d : ℕ) (h1 : a 1 = 1) (h2 : a n = 51)
  (h3 : ∀ i, a i = a 1 + (i-1) * d) : n + d = 16 :=
by
  sorry

end min_n_plus_d_l302_30255


namespace sequence_an_sequence_Tn_l302_30268

theorem sequence_an (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, 2 * S n = a n ^ 2 + a n):
  ∀ n, a n = n :=
sorry

theorem sequence_Tn (b : ℕ → ℕ) (T : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = a n ^ 2 + a n) (h2 : ∀ n, a n = n) (h3 : ∀ n, b n = 2^n * a n):
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_sequence_Tn_l302_30268


namespace sodium_chloride_moles_produced_l302_30259

theorem sodium_chloride_moles_produced (NaOH HCl NaCl : ℕ) : 
    (NaOH = 3) → (HCl = 3) → NaCl = 3 :=
by
  intro hNaOH hHCl
  -- Placeholder for actual proof
  sorry

end sodium_chloride_moles_produced_l302_30259


namespace find_n_l302_30295

theorem find_n (n : ℝ) (h1 : (n ≠ 0)) (h2 : ∃ (n' : ℝ), n = n' ∧ -n' = -9 / n') (h3 : ∀ x : ℝ, x > 0 → -n * x < 0) : n = 3 :=
sorry

end find_n_l302_30295


namespace avg_age_9_proof_l302_30263

-- Definitions of the given conditions
def total_persons := 16
def avg_age_all := 15
def total_age_all := total_persons * avg_age_all -- 240
def persons_5 := 5
def avg_age_5 := 14
def total_age_5 := persons_5 * avg_age_5 -- 70
def age_15th_person := 26
def persons_9 := 9

-- The theorem to prove the average age of the remaining 9 persons
theorem avg_age_9_proof : 
  total_age_all - total_age_5 - age_15th_person = persons_9 * 16 :=
by
  sorry

end avg_age_9_proof_l302_30263


namespace polygons_ratio_four_three_l302_30222

theorem polygons_ratio_four_three : 
  ∃ (r k : ℕ), 3 ≤ r ∧ 3 ≤ k ∧ 
  (180 - (360 / r : ℝ)) / (180 - (360 / k : ℝ)) = 4 / 3 
  ∧ ((r, k) = (42,7) ∨ (r, k) = (18,6) ∨ (r, k) = (10,5) ∨ (r, k) = (6,4)) :=
sorry

end polygons_ratio_four_three_l302_30222


namespace quadratic_expression_and_intersections_l302_30238

noncomputable def quadratic_eq_expression (a b c : ℝ) : Prop :=
  ∃ a b c : ℝ, (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = -3) ∧ (4 * a + 2 * b + c = - 5 / 2) ∧ (b = -2 * a) ∧ (c = -5 / 2) ∧ (a = 1 / 2)

noncomputable def find_m (a b c : ℝ) : Prop :=
  ∀ x m : ℝ, (a * (-2:ℝ)^2 + b * (-2:ℝ) + c = m) → (a * (4:ℝ) + b * (4:ℝ) + c = m) → (6:ℝ) = abs (x - (-2:ℝ)) → m = 3 / 2

noncomputable def y_range (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
  (x^2 * a + x * b + c >= -3) ∧ 
  (x^2 * a + x * b + c < 5) ↔ (-3 < x ∧ x < 3)

theorem quadratic_expression_and_intersections 
  (a b c : ℝ) (h1 : quadratic_eq_expression a b c) (h2 : find_m a b c) : y_range a b c :=
  sorry

end quadratic_expression_and_intersections_l302_30238


namespace find_x_values_l302_30206

theorem find_x_values (x : ℝ) :
  (3 * x + 2 < (x - 1) ^ 2 ∧ (x - 1) ^ 2 < 9 * x + 1) ↔
  (x > (5 + Real.sqrt 29) / 2 ∧ x < 11) := 
by
  sorry

end find_x_values_l302_30206


namespace divisibility_by_120_l302_30273

theorem divisibility_by_120 (n : ℕ) : 120 ∣ (n^7 - n^3) :=
sorry

end divisibility_by_120_l302_30273


namespace abc_le_one_eighth_l302_30290

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l302_30290


namespace total_savings_l302_30209

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l302_30209


namespace f_g_of_2_eq_4_l302_30202

def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (x : ℝ) : ℝ := 2*x - 5

theorem f_g_of_2_eq_4 : f (g 2) = 4 := by
  sorry

end f_g_of_2_eq_4_l302_30202


namespace solve_abs_inequality_l302_30234

theorem solve_abs_inequality (x : ℝ) :
  |x + 2| + |x - 2| < x + 7 ↔ -7 / 3 < x ∧ x < 7 :=
sorry

end solve_abs_inequality_l302_30234


namespace sequence_inequality_l302_30236

open Real

def seq (F : ℕ → ℝ) : Prop :=
  F 1 = 1 ∧ F 2 = 2 ∧ ∀ n ≥ 1, F (n + 2) = F (n + 1) + F n

theorem sequence_inequality (F : ℕ → ℝ) (h : seq F) (n : ℕ) : 
  sqrt (F (n+1))^(1/(n:ℝ)) ≥ 1 + 1 / sqrt (F n)^(1/(n:ℝ)) :=
sorry

end sequence_inequality_l302_30236


namespace average_of_primes_less_than_twenty_l302_30245

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sum_primes : ℕ := 77
def count_primes : ℕ := 8
def average_primes : ℚ := 77 / 8

theorem average_of_primes_less_than_twenty : (primes_less_than_twenty.sum / count_primes : ℚ) = 9.625 := by
  sorry

end average_of_primes_less_than_twenty_l302_30245


namespace cube_surface_area_is_24_l302_30251

def edge_length : ℝ := 2

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a * a

theorem cube_surface_area_is_24 : surface_area_of_cube edge_length = 24 := 
by 
  -- Compute the surface area of the cube with given edge length
  -- surface_area_of_cube 2 = 6 * 2 * 2 = 24
  sorry

end cube_surface_area_is_24_l302_30251


namespace unit_prices_possible_combinations_l302_30224

-- Part 1: Unit Prices
theorem unit_prices (x y : ℕ) (h1 : x = y - 20) (h2 : 3 * x + 2 * y = 340) : x = 60 ∧ y = 80 := 
by 
  sorry

-- Part 2: Possible Combinations
theorem possible_combinations (a : ℕ) (h3 : 60 * a + 80 * (150 - a) ≤ 10840) (h4 : 150 - a ≥ 3 * a / 2) : 
  a = 58 ∨ a = 59 ∨ a = 60 := 
by 
  sorry

end unit_prices_possible_combinations_l302_30224


namespace determine_range_of_m_l302_30216

variable {m : ℝ}

-- Condition (p) for all x in ℝ, x^2 - mx + 3/2 > 0
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + (3 / 2) > 0

-- Condition (q) the foci of the ellipse lie on the x-axis, implying 2 < m < 3
def condition_q (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ ((3 - m) > 0) ∧ ((m - 1) > (3 - m))

theorem determine_range_of_m (h1 : condition_p m) (h2 : condition_q m) : 2 < m ∧ m < Real.sqrt 6 :=
  sorry

end determine_range_of_m_l302_30216


namespace gigi_initial_batches_l302_30249

-- Define the conditions
def flour_per_batch := 2 
def initial_flour := 20 
def remaining_flour := 14 
def future_batches := 7

-- Prove the number of batches initially baked is 3
theorem gigi_initial_batches :
  (initial_flour - remaining_flour) / flour_per_batch = 3 :=
by
  sorry

end gigi_initial_batches_l302_30249


namespace sum_of_circumferences_eq_28pi_l302_30292

theorem sum_of_circumferences_eq_28pi (R r : ℝ) (h1 : r = (1:ℝ)/3 * R) (h2 : R - r = 7) : 
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end sum_of_circumferences_eq_28pi_l302_30292


namespace amount_paid_to_Y_l302_30212

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end amount_paid_to_Y_l302_30212


namespace percentage_running_wickets_l302_30248

-- Conditions provided as definitions and assumptions in Lean
def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def boundary_runs (b : ℕ) := b * 4
def six_runs (s : ℕ) := s * 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries := boundary_runs boundaries
def runs_from_sixes := six_runs sixes
def runs_not_from_boundaries_and_sixes := total_runs - (runs_from_boundaries + runs_from_sixes)

-- Proof that the percentage of the total score by running between the wickets is 50%
theorem percentage_running_wickets :
  (runs_not_from_boundaries_and_sixes : ℝ) / (total_runs : ℝ) * 100 = 50 :=
by
  sorry

end percentage_running_wickets_l302_30248


namespace R_depends_on_d_and_n_l302_30298

-- Define the given properties of the arithmetic progression sums
def s1 (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2
def s3 (a d n : ℕ) : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
def s5 (a d n : ℕ) : ℕ := (5 * n * (2 * a + (5 * n - 1) * d)) / 2

-- Define R in terms of s1, s3, and s5
def R (a d n : ℕ) : ℕ := s5 a d n - s3 a d n - s1 a d n

-- The main theorem to prove the statement about R's dependency
theorem R_depends_on_d_and_n (a d n : ℕ) : R a d n = 7 * d * n^2 := by 
  sorry

end R_depends_on_d_and_n_l302_30298


namespace hyperbola_properties_l302_30288

open Real

def is_asymptote (y x : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_properties :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  ∀ (a b c : ℝ), 
  (a = 2) →
  (b = 1) →
  (c = sqrt (a^2 + b^2)) →
  (∀ y x : ℝ, (is_asymptote y x)) ∧ (eccentricity a (sqrt (a^2 + b^2)) = sqrt 5 / 2) :=
by
  intros x y h a b c ha hb hc
  sorry

end hyperbola_properties_l302_30288


namespace population_decreases_l302_30269

theorem population_decreases (P_0 : ℝ) (k : ℝ) (n : ℕ) (hP0 : P_0 > 0) (hk : -1 < k ∧ k < 0) : 
  P_0 * (1 + k)^n * k < 0 → P_0 * (1 + k)^(n + 1) < P_0 * (1 + k)^n := by
  sorry

end population_decreases_l302_30269


namespace parabola_directrix_l302_30201

theorem parabola_directrix (y : ℝ) : 
  x = -((1:ℝ)/4)*y^2 → x = 1 :=
by 
  sorry

end parabola_directrix_l302_30201


namespace paint_can_distribution_l302_30240

-- Definitions based on conditions provided in the problem.
def ratio_red := 3
def ratio_white := 2
def ratio_blue := 1
def total_paint := 60
def ratio_sum := ratio_red + ratio_white + ratio_blue

-- Definition of the problem to be proved.
theorem paint_can_distribution :
  (ratio_red * total_paint) / ratio_sum = 30 ∧
  (ratio_white * total_paint) / ratio_sum = 20 ∧
  (ratio_blue * total_paint) / ratio_sum = 10 := 
by
  sorry

end paint_can_distribution_l302_30240


namespace exists_k_simplifies_expression_to_5x_squared_l302_30274

theorem exists_k_simplifies_expression_to_5x_squared :
  ∃ k : ℝ, (∀ x : ℝ, (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :=
by
  sorry

end exists_k_simplifies_expression_to_5x_squared_l302_30274


namespace max_discount_l302_30254

theorem max_discount (cost_price selling_price : ℝ) (min_profit_margin : ℝ) (x : ℝ) : 
  cost_price = 400 → selling_price = 500 → min_profit_margin = 0.0625 → 
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin * cost_price) → x ≤ 15 :=
by
  intros h1 h2 h3 h4
  sorry

end max_discount_l302_30254


namespace perpendicularity_proof_l302_30218

-- Definitions of geometric entities and properties
variable (Plane Line : Type)
variable (α β : Plane) -- α and β are planes
variable (m n : Line) -- m and n are lines

-- Geometric properties and relations
variable (subset : Line → Plane → Prop) -- Line is subset of plane
variable (perpendicular : Line → Plane → Prop) -- Line is perpendicular to plane
variable (line_perpendicular : Line → Line → Prop) -- Line is perpendicular to another line

-- Conditions
axiom planes_different : α ≠ β
axiom lines_different : m ≠ n
axiom m_in_beta : subset m β
axiom n_in_beta : subset n β

-- Proof problem statement
theorem perpendicularity_proof :
  (subset m α) → (perpendicular n α) → (line_perpendicular n m) :=
by
  sorry

end perpendicularity_proof_l302_30218


namespace min_b_for_factorization_l302_30291

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l302_30291


namespace student_history_mark_l302_30252

theorem student_history_mark
  (math_score : ℕ)
  (desired_average : ℕ)
  (third_subject_score : ℕ)
  (history_score : ℕ) :
  math_score = 74 →
  desired_average = 75 →
  third_subject_score = 70 →
  (math_score + history_score + third_subject_score) / 3 = desired_average →
  history_score = 81 :=
by
  intros h_math h_avg h_third h_equiv
  sorry

end student_history_mark_l302_30252


namespace johns_pieces_of_gum_l302_30250

theorem johns_pieces_of_gum : 
  (∃ (john cole aubrey : ℕ), 
    cole = 45 ∧ 
    aubrey = 0 ∧ 
    (john + cole + aubrey) = 3 * 33) → 
  ∃ john : ℕ, john = 54 :=
by 
  sorry

end johns_pieces_of_gum_l302_30250


namespace find_d1_l302_30247

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  ∃ (d4 d3 d2 d0 : ℤ), 
  (∀ (n : ℕ), n ≥ 4 ∧ n % 2 = 0 → 
     E n = d4 * n^4 + d3 * n^3 + d2 * n^2 + (12 : ℤ) * n + d0) :=
sorry

end find_d1_l302_30247


namespace jane_average_speed_correct_l302_30262

noncomputable def jane_average_speed : ℝ :=
  let total_distance : ℝ := 250
  let total_time : ℝ := 6
  total_distance / total_time

theorem jane_average_speed_correct : jane_average_speed = 41.67 := by
  sorry

end jane_average_speed_correct_l302_30262


namespace problem_statement_l302_30213

variable (x y : ℝ)

theorem problem_statement
  (h1 : 4 * x + y = 9)
  (h2 : x + 4 * y = 16) :
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 :=
sorry

end problem_statement_l302_30213


namespace tens_digit_of_8_pow_2023_l302_30287

theorem tens_digit_of_8_pow_2023 :
    ∃ d, 0 ≤ d ∧ d < 10 ∧ (8^2023 % 100) / 10 = d ∧ d = 1 :=
by
  sorry

end tens_digit_of_8_pow_2023_l302_30287


namespace jamies_father_days_to_lose_weight_l302_30243

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def calories_burned_per_day : ℕ := 2500
def calories_consumed_per_day : ℕ := 2000
def net_calories_burned_per_day : ℕ := calories_burned_per_day - calories_consumed_per_day
def total_calories_to_burn : ℕ := pounds_to_lose * calories_per_pound
def days_to_burn_calories := total_calories_to_burn / net_calories_burned_per_day

theorem jamies_father_days_to_lose_weight : days_to_burn_calories = 35 := by
  sorry

end jamies_father_days_to_lose_weight_l302_30243


namespace multiple_6_9_statements_false_l302_30264

theorem multiple_6_9_statements_false
    (a b : ℤ)
    (h₁ : ∃ m : ℤ, a = 6 * m)
    (h₂ : ∃ n : ℤ, b = 9 * n) :
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → ((a + b) % 2 = 0)) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 6 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 ≠ 0) :=
by
  sorry

end multiple_6_9_statements_false_l302_30264


namespace pipe_filling_time_l302_30278

-- Definitions for the conditions
variables (A : ℝ) (h : 1 / A - 1 / 24 = 1 / 12)

-- The statement of the problem
theorem pipe_filling_time : A = 8 :=
by
  sorry

end pipe_filling_time_l302_30278


namespace book_price_distribution_l302_30289

theorem book_price_distribution :
  ∃ (x y z: ℤ), 
  x + y + z = 109 ∧
  (34 * x + 27.5 * y + 17.5 * z : ℝ) = 2845 ∧
  (x - y : ℤ).natAbs ≤ 2 ∧ (y - z).natAbs ≤ 2 := 
sorry

end book_price_distribution_l302_30289


namespace unique_shell_arrangements_l302_30200

theorem unique_shell_arrangements : 
  let shells := 12
  let symmetry_ops := 12
  let total_arrangements := Nat.factorial shells
  let distinct_arrangements := total_arrangements / symmetry_ops
  distinct_arrangements = 39916800 := by
  sorry

end unique_shell_arrangements_l302_30200


namespace number_of_algebra_textbooks_l302_30232

theorem number_of_algebra_textbooks
  (x y n : ℕ)
  (h₁ : x * n + y = 2015)
  (h₂ : y * n + x = 1580) :
  y = 287 := 
sorry

end number_of_algebra_textbooks_l302_30232


namespace reflect_over_y_axis_matrix_l302_30283

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end reflect_over_y_axis_matrix_l302_30283


namespace arrange_balls_l302_30281

/-- Given 4 yellow balls and 3 red balls, we want to prove that there are 35 different ways to arrange these balls in a row. -/
theorem arrange_balls : (Nat.choose 7 4) = 35 := by
  sorry

end arrange_balls_l302_30281


namespace regular_soda_count_l302_30256

theorem regular_soda_count 
  (diet_soda : ℕ) 
  (additional_soda : ℕ) 
  (h1 : diet_soda = 19) 
  (h2 : additional_soda = 41) 
  : diet_soda + additional_soda = 60 :=
by
  sorry

end regular_soda_count_l302_30256


namespace mixture_contains_pecans_l302_30279

theorem mixture_contains_pecans 
  (price_per_cashew_per_pound : ℝ)
  (cashews_weight : ℝ)
  (price_per_mixture_per_pound : ℝ)
  (price_of_cashews : ℝ)
  (mixture_weight : ℝ)
  (pecans_weight : ℝ)
  (price_per_pecan_per_pound : ℝ)
  (pecans_price : ℝ)
  (total_cost_of_mixture : ℝ)
  
  (h1 : price_per_cashew_per_pound = 3.50) 
  (h2 : cashews_weight = 2)
  (h3 : price_per_mixture_per_pound = 4.34) 
  (h4 : pecans_weight = 1.33333333333)
  (h5 : price_per_pecan_per_pound = 5.60)
  
  (h6 : price_of_cashews = cashews_weight * price_per_cashew_per_pound)
  (h7 : mixture_weight = cashews_weight + pecans_weight)
  (h8 : pecans_price = pecans_weight * price_per_pecan_per_pound)
  (h9 : total_cost_of_mixture = price_of_cashews + pecans_price)

  (h10 : price_per_mixture_per_pound = total_cost_of_mixture / mixture_weight)
  
  : pecans_weight = 1.33333333333 :=
sorry

end mixture_contains_pecans_l302_30279


namespace radius_of_third_circle_l302_30257

theorem radius_of_third_circle (r₁ r₂ : ℝ) (r₁_val : r₁ = 23) (r₂_val : r₂ = 37) : 
  ∃ r : ℝ, r = 2 * Real.sqrt 210 :=
by
  sorry

end radius_of_third_circle_l302_30257


namespace largest_number_is_B_l302_30280

noncomputable def numA : ℝ := 7.196533
noncomputable def numB : ℝ := 7.19655555555555555555555555555555555555 -- 7.196\overline{5}
noncomputable def numC : ℝ := 7.1965656565656565656565656565656565 -- 7.19\overline{65}
noncomputable def numD : ℝ := 7.196596596596596596596596596596596 -- 7.1\overline{965}
noncomputable def numE : ℝ := 7.196519651965196519651965196519651 -- 7.\overline{1965}

theorem largest_number_is_B : 
  numB > numA ∧ numB > numC ∧ numB > numD ∧ numB > numE :=
by
  sorry

end largest_number_is_B_l302_30280


namespace abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l302_30227

/-- Part 1: Prove that the number \overline{abba} is divisible by 11 -/
theorem abba_divisible_by_11 (a b : ℕ) : 11 ∣ (1000 * a + 100 * b + 10 * b + a) :=
sorry

/-- Part 2: Prove that the number \overline{aaabbb} is divisible by 37 -/
theorem aaabbb_divisible_by_37 (a b : ℕ) : 37 ∣ (1000 * 111 * a + 111 * b) :=
sorry

/-- Part 3: Prove that the number \overline{ababab} is divisible by 7 -/
theorem ababab_divisible_by_7 (a b : ℕ) : 7 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) :=
sorry

/-- Part 4: Prove that the number \overline{abab} - \overline{baba} is divisible by 9 and 101 -/
theorem abab_baba_divisible_by_9_and_101 (a b : ℕ) :
  9 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) ∧
  101 ∣ (1000 * a + 100 * b + 10 * a + b - (1000 * b + 100 * a + 10 * b + a)) :=
sorry

end abba_divisible_by_11_aaabbb_divisible_by_37_ababab_divisible_by_7_abab_baba_divisible_by_9_and_101_l302_30227


namespace mr_williams_land_percentage_l302_30242

-- Given conditions
def farm_tax_percent : ℝ := 60
def total_tax_collected : ℝ := 5000
def mr_williams_tax_paid : ℝ := 480

-- Theorem statement
theorem mr_williams_land_percentage :
  (mr_williams_tax_paid / total_tax_collected) * 100 = 9.6 := by
  sorry

end mr_williams_land_percentage_l302_30242


namespace integer_solutions_are_zero_l302_30276

-- Definitions for integers and the given equation
def satisfies_equation (a b : ℤ) : Prop :=
  a^2 * b^2 = a^2 + b^2

-- The main statement to prove
theorem integer_solutions_are_zero :
  ∀ (a b : ℤ), satisfies_equation a b → (a = 0 ∧ b = 0) :=
sorry

end integer_solutions_are_zero_l302_30276


namespace half_vectorAB_is_2_1_l302_30275

def point := ℝ × ℝ -- Define a point as a pair of real numbers
def vector := ℝ × ℝ -- Define a vector as a pair of real numbers

def A : point := (-1, 0) -- Define point A
def B : point := (3, 2) -- Define point B

noncomputable def vectorAB : vector := (B.1 - A.1, B.2 - A.2) -- Define vector AB as B - A

noncomputable def half_vectorAB : vector := (1 / 2 * vectorAB.1, 1 / 2 * vectorAB.2) -- Define half of vector AB

theorem half_vectorAB_is_2_1 : half_vectorAB = (2, 1) := by
  -- Sorry is a placeholder for the proof
  sorry

end half_vectorAB_is_2_1_l302_30275


namespace length_O_D1_l302_30270

-- Definitions for the setup of the cube and its faces, the center of the sphere, and the intersecting circles
def O : Point := sorry -- Center of the sphere and cube
def radius : ℝ := 10 -- Radius of the sphere

-- Intersection circles with given radii on specific faces of the cube
def r_ADA1D1 : ℝ := 1 -- Radius of the intersection circle on face ADA1D1
def r_A1B1C1D1 : ℝ := 1 -- Radius of the intersection circle on face A1B1C1D1
def r_CDD1C1 : ℝ := 3 -- Radius of the intersection circle on face CDD1C1

-- Distances derived from the problem
def OX1_sq : ℝ := radius^2 - r_ADA1D1^2
def OX2_sq : ℝ := radius^2 - r_A1B1C1D1^2
def OX_sq : ℝ := radius^2 - r_CDD1C1^2

-- To simplify, replace OX1, OX2, and OX with their squared values directly
def OX1_sq_calc : ℝ := 99
def OX2_sq_calc : ℝ := 99
def OX_sq_calc : ℝ := 91

theorem length_O_D1 : (OX1_sq_calc + OX2_sq_calc + OX_sq_calc) = 289 ↔ OD1 = 17 := by
  sorry

end length_O_D1_l302_30270


namespace compare_fractions_l302_30286

theorem compare_fractions : - (1 + 3 / 5) < -1.5 := 
by
  sorry

end compare_fractions_l302_30286


namespace distinct_solutions_eq_l302_30266

theorem distinct_solutions_eq : ∃! x : ℝ, abs (x - 5) = abs (x + 3) :=
by
  sorry

end distinct_solutions_eq_l302_30266


namespace smallest_n_with_square_ending_in_2016_l302_30215

theorem smallest_n_with_square_ending_in_2016 : 
  ∃ n : ℕ, (n^2 % 10000 = 2016) ∧ (n = 996) :=
by
  sorry

end smallest_n_with_square_ending_in_2016_l302_30215


namespace simplify_fraction_l302_30203

theorem simplify_fraction :
  ( (5^2010)^2 - (5^2008)^2 ) / ( (5^2009)^2 - (5^2007)^2 ) = 25 := by
  sorry

end simplify_fraction_l302_30203


namespace range_of_a_l302_30293

-- Define the inequality condition
def inequality (x a : ℝ) : Prop :=
  2 * x^2 + a * x - a^2 > 0

-- State the main problem
theorem range_of_a (a: ℝ) : 
  inequality 2 a -> (-2 < a) ∧ (a < 4) :=
by
  sorry

end range_of_a_l302_30293


namespace slip_2_5_in_A_or_C_l302_30228

-- Define the slips and their values
def slips : List ℚ := [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 4.5, 5, 5.5, 6]

-- Define the cups
inductive Cup
| A | B | C | D | E | F

open Cup

-- Define the given cups constraints
def sum_constraints : Cup → ℚ
| A => 6
| B => 7
| C => 8
| D => 9
| E => 10
| F => 10

-- Initial conditions for slips placement
def slips_in_cups (c : Cup) : List ℚ :=
match c with
| F => [1.5]
| B => [4]
| _ => []

-- We'd like to prove that:
def slip_2_5_can_go_into : Prop :=
  (slips_in_cups A = [2.5] ∧ slips_in_cups C = [2.5])

theorem slip_2_5_in_A_or_C : slip_2_5_can_go_into :=
sorry

end slip_2_5_in_A_or_C_l302_30228


namespace max_area_of_rectangular_fence_l302_30217

theorem max_area_of_rectangular_fence (x y : ℕ) (h : x + y = 75) : 
  (x * (75 - x) ≤ 1406) ∧ (∀ x' y', x' + y' = 75 → x' * y' ≤ 1406) :=
by
  sorry

end max_area_of_rectangular_fence_l302_30217


namespace geom_seq_m_equals_11_l302_30296

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ (n : ℕ), a n = a1 * q ^ n

theorem geom_seq_m_equals_11 {a : ℕ → ℝ} {q : ℝ} (hq : q ≠ 1) 
  (h : geometric_sequence a 1 q) : 
  a 11 = a 1 * a 2 * a 3 * a 4 * a 5 := 
by sorry

end geom_seq_m_equals_11_l302_30296


namespace overlap_32_l302_30231

section
variables (t : ℝ)
def position_A : ℝ := 120 - 50 * t
def position_B : ℝ := 220 - 50 * t
def position_N : ℝ := 30 * t - 30
def position_M : ℝ := 30 * t + 10

theorem overlap_32 :
  (∃ t : ℝ, (30 * t + 10 - (120 - 50 * t) = 32) ∨ 
            (-50 * t + 220 - (30 * t - 30) = 32)) ↔
  (t = 71 / 40 ∨ t = 109 / 40) :=
sorry
end

end overlap_32_l302_30231


namespace solve_for_nabla_l302_30277

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l302_30277
