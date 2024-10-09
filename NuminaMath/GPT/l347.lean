import Mathlib

namespace jill_has_6_more_dolls_than_jane_l347_34743

theorem jill_has_6_more_dolls_than_jane
  (total_dolls : ℕ) 
  (jane_dolls : ℕ) 
  (more_dolls_than : ℕ → ℕ → Prop)
  (h1 : total_dolls = 32) 
  (h2 : jane_dolls = 13) 
  (jill_dolls : ℕ)
  (h3 : more_dolls_than jill_dolls jane_dolls) :
  (jill_dolls - jane_dolls) = 6 :=
by
  -- the proof goes here
  sorry

end jill_has_6_more_dolls_than_jane_l347_34743


namespace kylie_coins_count_l347_34739

theorem kylie_coins_count 
  (P : ℕ) 
  (from_brother : ℕ) 
  (from_father : ℕ) 
  (given_to_Laura : ℕ) 
  (coins_left : ℕ) 
  (h1 : from_brother = 13) 
  (h2 : from_father = 8) 
  (h3 : given_to_Laura = 21) 
  (h4 : coins_left = 15) : (P + from_brother + from_father) - given_to_Laura = coins_left → P = 15 :=
by
  sorry

end kylie_coins_count_l347_34739


namespace minimum_value_of_quadratic_l347_34727

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = - (p + q) / 2 ∧ ∀ y : ℝ, (y^2 + p*y + q*y) ≥ ((- (p + q) / 2)^2 + p*(- (p + q) / 2) + q*(- (p + q) / 2)) := by
  sorry

end minimum_value_of_quadratic_l347_34727


namespace post_tax_income_correct_l347_34715

noncomputable def worker_a_pre_tax_income : ℝ :=
  80 * 30 + 50 * 30 * 1.20 + 35 * 30 * 1.50 + (35 * 30 * 1.50) * 0.05

noncomputable def worker_b_pre_tax_income : ℝ :=
  90 * 25 + 45 * 25 * 1.25 + 40 * 25 * 1.45 + (40 * 25 * 1.45) * 0.05

noncomputable def worker_c_pre_tax_income : ℝ :=
  70 * 35 + 40 * 35 * 1.15 + 60 * 35 * 1.60 + (60 * 35 * 1.60) * 0.05

noncomputable def worker_a_post_tax_income : ℝ := 
  worker_a_pre_tax_income * 0.85 - 200

noncomputable def worker_b_post_tax_income : ℝ := 
  worker_b_pre_tax_income * 0.82 - 250

noncomputable def worker_c_post_tax_income : ℝ := 
  worker_c_pre_tax_income * 0.80 - 300

theorem post_tax_income_correct :
  worker_a_post_tax_income = 4775.69 ∧ 
  worker_b_post_tax_income = 3996.57 ∧ 
  worker_c_post_tax_income = 5770.40 :=
by {
  sorry
}

end post_tax_income_correct_l347_34715


namespace factorize_m_minimize_ab_find_abc_l347_34724

-- Problem 1: Factorization
theorem factorize_m (m : ℝ) : m^2 - 6 * m + 5 = (m - 1) * (m - 5) :=
sorry

-- Problem 2: Minimization
theorem minimize_ab (a b : ℝ) (h1 : (a - 2)^2 ≥ 0) (h2 : (b + 5)^2 ≥ 0) :
  ∃ (a b : ℝ), (a - 2)^2 + (b + 5)^2 + 4 = 4 ∧ a = 2 ∧ b = -5 :=
sorry

-- Problem 3: Value of a + b + c
theorem find_abc (a b c : ℝ) (h1 : a - b = 8) (h2 : a * b + c^2 - 4 * c + 20 = 0) :
  a + b + c = 2 :=
sorry

end factorize_m_minimize_ab_find_abc_l347_34724


namespace trigonometric_identity_l347_34734

theorem trigonometric_identity (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 1) : 
  (2 * Real.sin α + Real.cos α) / (3 * Real.cos α - Real.sin α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l347_34734


namespace election_votes_l347_34770

theorem election_votes (V : ℕ) (h1 : ∃ Vb, Vb = 2509 ∧ (0.8 * V : ℝ) = (Vb + 0.15 * (V : ℝ)) + Vb) : V = 7720 :=
sorry

end election_votes_l347_34770


namespace measure_of_C_l347_34769

-- Define angles and their magnitudes
variables (A B C X : Type) [LinearOrder C]
def angle_measure (angle : Type) : ℕ := sorry
def parallel (l1 l2 : Type) : Prop := sorry
def transversal (l1 l2 l3 : Type) : Prop := sorry
def alternate_interior (angle1 angle2 : Type) : Prop := sorry
def adjacent (angle1 angle2 : Type) : Prop := sorry
def complementary (angle1 angle2 : Type) : Prop := sorry

-- The given conditions
axiom h1 : parallel A X
axiom h2 : transversal A B X
axiom h3 : angle_measure A = 85
axiom h4 : angle_measure B = 35
axiom h5 : alternate_interior C A
axiom h6 : complementary B X
axiom h7 : adjacent C X

-- Define the proof problem
theorem measure_of_C : angle_measure C = 85 :=
by {
  -- The proof goes here, skipping with sorry
  sorry
}

end measure_of_C_l347_34769


namespace Sn_eq_S9_l347_34722

-- Definition of the arithmetic sequence sum formula.
def Sn (n a1 d : ℕ) : ℕ := (n * a1) + (n * (n - 1) / 2 * d)

theorem Sn_eq_S9 (a1 d : ℕ) (h1 : Sn 3 a1 d = 9) (h2 : Sn 6 a1 d = 36) : Sn 9 a1 d = 81 := by
  sorry

end Sn_eq_S9_l347_34722


namespace find_number_l347_34788

theorem find_number (x : ℝ) : 0.5 * 56 = 0.3 * x + 13 ↔ x = 50 :=
by
  -- Proof would go here
  sorry

end find_number_l347_34788


namespace union_inter_distrib_inter_union_distrib_l347_34796

section
variables {α : Type*} (A B C : Set α)

-- Problem (a)
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) :=
sorry

-- Problem (b)
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) :=
sorry
end

end union_inter_distrib_inter_union_distrib_l347_34796


namespace polygon_with_15_diagonals_has_7_sides_l347_34758

-- Define the number of diagonals formula for a regular polygon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement
theorem polygon_with_15_diagonals_has_7_sides :
  ∃ n : ℕ, number_of_diagonals n = 15 ∧ n = 7 :=
by
  sorry

end polygon_with_15_diagonals_has_7_sides_l347_34758


namespace find_sixth_number_l347_34701

theorem find_sixth_number (A : ℕ → ℤ) 
  (h1 : (1 / 11 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 60)
  (h2 : (1 / 6 : ℚ) * (A 1 + A 2 + A 3 + A 4 + A 5 + A 6) = 88)
  (h3 : (1 / 6 : ℚ) * (A 6 + A 7 + A 8 + A 9 + A 10 + A 11) = 65) :
  A 6 = 258 :=
sorry

end find_sixth_number_l347_34701


namespace explicit_formula_inequality_solution_l347_34745

noncomputable def f (x : ℝ) : ℝ := (x : ℝ) / (x^2 + 1)

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x → y < b → x < y → f x < f y
def f_half_eq_two_fifths : Prop := f (1/2) = 2/5

-- Questions rewritten as goals
theorem explicit_formula :
  odd_function f ∧ increasing_on_interval f (-1) 1 ∧ f_half_eq_two_fifths →
  ∀ x, f x = x / (x^2 + 1) := by 
sorry

theorem inequality_solution :
  odd_function f ∧ increasing_on_interval f (-1) 1 →
  ∀ t, (f (t - 1) + f t < 0) ↔ (0 < t ∧ t < 1/2) := by 
sorry

end explicit_formula_inequality_solution_l347_34745


namespace solve_abs_eq_l347_34732

theorem solve_abs_eq (x : ℝ) : 
  (|x - 4| + 3 * x = 12) ↔ (x = 4) :=
by
  sorry

end solve_abs_eq_l347_34732


namespace valid_triples_l347_34718

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ∣ (y + 1)) (hyz : y ∣ (z + 1)) (hzx : z ∣ (x + 1)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
  (x = 1 ∧ y = 2 ∧ z = 3) :=
sorry

end valid_triples_l347_34718


namespace term_containing_x3_l347_34756

-- Define the problem statement in Lean 4
theorem term_containing_x3 (a : ℝ) (x : ℝ) (hx : x ≠ 0) 
(h_sum_coeff : (2 + a) ^ 5 = 0) :
  (2 * x + a / x) ^ 5 = -160 * x ^ 3 :=
sorry

end term_containing_x3_l347_34756


namespace solve_for_x_l347_34774

theorem solve_for_x (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) : 
  (x + 5) / (x - 3) = (x - 4) / (x + 2) → x = 1 / 7 :=
by
  sorry

end solve_for_x_l347_34774


namespace unique_solution_l347_34707

variables {x y z : ℝ}

def equation1 (x y z : ℝ) : Prop :=
  (x^2 + x*y + y^2) * (y^2 + y*z + z^2) * (z^2 + z*x + x^2) = x*y*z

def equation2 (x y z : ℝ) : Prop :=
  (x^4 + x^2*y^2 + y^4) * (y^4 + y^2*z^2 + z^4) * (z^4 + z^2*x^2 + x^4) = x^3*y^3*z^3

theorem unique_solution :
  equation1 x y z ∧ equation2 x y z → x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
by
  sorry

end unique_solution_l347_34707


namespace total_size_of_game_is_880_l347_34719

-- Define the initial amount already downloaded
def initialAmountDownloaded : ℕ := 310

-- Define the download speed after the connection slows (in MB per minute)
def downloadSpeed : ℕ := 3

-- Define the remaining download time (in minutes)
def remainingDownloadTime : ℕ := 190

-- Define the total additional data to be downloaded in the remaining time (speed * time)
def additionalDataDownloaded : ℕ := downloadSpeed * remainingDownloadTime

-- Define the total size of the game as the sum of initial and additional data downloaded
def totalSizeOfGame : ℕ := initialAmountDownloaded + additionalDataDownloaded

-- State the theorem to prove
theorem total_size_of_game_is_880 : totalSizeOfGame = 880 :=
by 
  -- We provide no proof here; 'sorry' indicates an unfinished proof.
  sorry

end total_size_of_game_is_880_l347_34719


namespace subtract_base3_sum_eq_result_l347_34717

theorem subtract_base3_sum_eq_result :
  let a := 10 -- interpreted as 10_3
  let b := 1101 -- interpreted as 1101_3
  let c := 2102 -- interpreted as 2102_3
  let d := 212 -- interpreted as 212_3
  let sum := 1210 -- interpreted as the base 3 sum of a + b + c
  let result := 1101 -- interpreted as the final base 3 result
  sum - d = result :=
by sorry

end subtract_base3_sum_eq_result_l347_34717


namespace total_shopping_cost_l347_34766

theorem total_shopping_cost 
  (sandwiches : ℕ := 3)
  (sandwich_cost : ℕ := 3)
  (water_bottle : ℕ := 1)
  (water_cost : ℕ := 2)
  : sandwiches * sandwich_cost + water_bottle * water_cost = 11 :=
by
  sorry

end total_shopping_cost_l347_34766


namespace ratio_youngest_sister_to_yvonne_l347_34742

def laps_yvonne := 10
def laps_joel := 15
def joel_ratio := 3

theorem ratio_youngest_sister_to_yvonne
  (laps_yvonne : ℕ)
  (laps_joel : ℕ)
  (joel_ratio : ℕ)
  (H_joel : laps_joel = 3 * (laps_yvonne / joel_ratio))
  : (laps_joel / joel_ratio) = laps_yvonne / 2 :=
by
  sorry

end ratio_youngest_sister_to_yvonne_l347_34742


namespace scientific_notation_470M_l347_34726

theorem scientific_notation_470M :
  (470000000 : ℝ) = 4.7 * 10^8 :=
sorry

end scientific_notation_470M_l347_34726


namespace earliest_year_for_mismatched_pairs_l347_34720

def num_pairs (year : ℕ) : ℕ := 2 ^ (year - 2013)

def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

theorem earliest_year_for_mismatched_pairs (year : ℕ) (h : year ≥ 2013) :
  (∃ pairs, (num_pairs year = pairs) ∧ (mismatched_pairs pairs ≥ 500)) → year = 2018 :=
by
  sorry

end earliest_year_for_mismatched_pairs_l347_34720


namespace binom_20_10_eq_184756_l347_34759

theorem binom_20_10_eq_184756 
  (h1 : Nat.choose 19 9 = 92378)
  (h2 : Nat.choose 19 10 = Nat.choose 19 9) : 
  Nat.choose 20 10 = 184756 := 
by
  sorry

end binom_20_10_eq_184756_l347_34759


namespace find_x_proportionally_l347_34782

theorem find_x_proportionally (k m x z : ℝ) (h1 : ∀ y, x = k * y^2) (h2 : ∀ z, y = m / (Real.sqrt z)) (h3 : x = 7 ∧ z = 16) :
  ∃ x, x = 7 / 9 := by
  sorry

end find_x_proportionally_l347_34782


namespace club_truncator_more_wins_than_losses_l347_34765

noncomputable def clubTruncatorWinsProbability : ℚ :=
  let total_matches := 8
  let prob := 1/3
  -- The combinatorial calculations for the balanced outcomes
  let balanced_outcomes := 70 + 560 + 420 + 28 + 1
  let total_outcomes := 3^total_matches
  let prob_balanced := balanced_outcomes / total_outcomes
  let prob_more_wins_or_more_losses := 1 - prob_balanced
  (prob_more_wins_or_more_losses / 2)

theorem club_truncator_more_wins_than_losses : 
  clubTruncatorWinsProbability = 2741 / 6561 := 
by 
  sorry

#check club_truncator_more_wins_than_losses

end club_truncator_more_wins_than_losses_l347_34765


namespace number_of_teachers_l347_34760

theorem number_of_teachers (total_population sample_size teachers_within_sample students_within_sample : ℕ) 
    (h_total_population : total_population = 3000) 
    (h_sample_size : sample_size = 150) 
    (h_students_within_sample : students_within_sample = 140) 
    (h_teachers_within_sample : teachers_within_sample = sample_size - students_within_sample) 
    (h_ratio : (total_population - students_within_sample) * sample_size = total_population * teachers_within_sample) : 
    total_population - students_within_sample = 200 :=
by {
  sorry
}

end number_of_teachers_l347_34760


namespace minimize_quadratic_l347_34797

def f (x : ℝ) := 3 * x^2 - 18 * x + 7

theorem minimize_quadratic : ∃ x : ℝ, f x = -20 ∧ ∀ y : ℝ, f y ≥ -20 := by
  sorry

end minimize_quadratic_l347_34797


namespace largest_n_for_factored_polynomial_l347_34716

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l347_34716


namespace ratio_of_a_to_b_l347_34773

theorem ratio_of_a_to_b 
  (b c a : ℝ)
  (h1 : b / c = 1 / 5) 
  (h2 : a / c = 1 / 7.5) : 
  a / b = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_l347_34773


namespace recruits_count_l347_34780

def x := 50
def y := 100
def z := 170

theorem recruits_count :
  ∃ n : ℕ, n = 211 ∧ (∀ a b c : ℕ, (b = 4 * a ∨ a = 4 * c ∨ c = 4 * b) → (b + 100 = a + 150) ∨ (a + 50 = c + 150) ∨ (c + 170 = b + 100)) :=
sorry

end recruits_count_l347_34780


namespace arrangements_of_6_books_l347_34706

theorem arrangements_of_6_books : ∃ (n : ℕ), n = 720 ∧ n = Nat.factorial 6 :=
by
  use 720
  constructor
  · rfl
  · sorry

end arrangements_of_6_books_l347_34706


namespace distance_city_A_to_C_l347_34755

variable (V_E V_F : ℝ) -- Define the average speeds of Eddy and Freddy
variable (time : ℝ) -- Define the time variable

-- Given conditions
def eddy_time : time = 3 := sorry
def freddy_time : time = 3 := sorry
def eddy_distance : ℝ := 600
def speed_ratio : V_E = 2 * V_F := sorry

-- Derived condition for Eddy's speed
def eddy_speed : V_E = eddy_distance / time := sorry

-- Derived conclusion for Freddy's distance
theorem distance_city_A_to_C (time : ℝ) (V_F : ℝ) : V_F * time = 300 := 
by 
  sorry

end distance_city_A_to_C_l347_34755


namespace unit_cost_calculation_l347_34791

theorem unit_cost_calculation : 
  ∀ (total_cost : ℕ) (ounces : ℕ), total_cost = 84 → ounces = 12 → (total_cost / ounces = 7) :=
by
  intros total_cost ounces h1 h2
  sorry

end unit_cost_calculation_l347_34791


namespace cindy_added_pens_l347_34737

-- Definitions based on conditions:
def initial_pens : ℕ := 20
def mike_pens : ℕ := 22
def sharon_pens : ℕ := 19
def final_pens : ℕ := 65

-- Intermediate calculations:
def pens_after_mike : ℕ := initial_pens + mike_pens
def pens_after_sharon : ℕ := pens_after_mike - sharon_pens

-- Proof statement:
theorem cindy_added_pens : pens_after_sharon + 42 = final_pens :=
by
  sorry

end cindy_added_pens_l347_34737


namespace base_12_addition_l347_34746

theorem base_12_addition (A B: ℕ) (hA: A = 10) (hB: B = 11) : 
  8 * 12^2 + A * 12 + 2 + (3 * 12^2 + B * 12 + 7) = 1 * 12^3 + 0 * 12^2 + 9 * 12 + 9 := 
by
  sorry

end base_12_addition_l347_34746


namespace sequence_general_term_l347_34799

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n / (a n + 3)) :
  ∀ n : ℕ, n > 0 → a n = 3 / (n + 2) := 
by 
  sorry

end sequence_general_term_l347_34799


namespace find_x_l347_34778

-- Define the conditions as variables and the target equation
variable (x : ℝ)

theorem find_x : 67 * x - 59 * x = 4828 → x = 603.5 := by
  intro h
  sorry

end find_x_l347_34778


namespace circle_radius_l347_34761

theorem circle_radius (x y : ℝ) : x^2 + y^2 - 2*y = 0 → ∃ r : ℝ, r = 1 :=
by
  sorry

end circle_radius_l347_34761


namespace time_ratio_A_to_B_l347_34784

theorem time_ratio_A_to_B (T_A T_B : ℝ) (hB : T_B = 36) (hTogether : 1 / T_A + 1 / T_B = 1 / 6) : T_A / T_B = 1 / 5 :=
by
  sorry

end time_ratio_A_to_B_l347_34784


namespace candle_burning_problem_l347_34713

theorem candle_burning_problem (burn_time_per_night_1h : ∀ n : ℕ, n = 8) 
                                (nightly_burn_rate : ∀ h : ℕ, h / 2 = 4) 
                                (total_nights : ℕ) 
                                (two_hour_nightly_burn : ∀ t : ℕ, t = 24) 
                                : ∃ candles : ℕ, candles = 6 := 
by {
  sorry
}

end candle_burning_problem_l347_34713


namespace no_positive_rational_solution_l347_34783

theorem no_positive_rational_solution :
  ¬ ∃ q : ℚ, 0 < q ∧ q^3 - 10 * q^2 + q - 2021 = 0 :=
by sorry

end no_positive_rational_solution_l347_34783


namespace quadratic_complete_square_l347_34754

theorem quadratic_complete_square :
  ∀ x : ℝ, (x^2 - 2 * x + 3 = (x - 1)^2 + 2) := 
by
  intro x
  sorry

end quadratic_complete_square_l347_34754


namespace proj_vector_correct_l347_34703

open Real

noncomputable def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let mag_sq := v.1 * v.1 + v.2 * v.2
  (dot / mag_sq) • v

theorem proj_vector_correct :
  vector_proj ⟨3, -1⟩ ⟨4, -6⟩ = ⟨18 / 13, -27 / 13⟩ :=
  sorry

end proj_vector_correct_l347_34703


namespace sector_area_max_angle_l347_34781

theorem sector_area_max_angle (r : ℝ) (θ : ℝ) (h : 0 < r ∧ r < 10) 
  (H : 2 * r + r * θ = 20) : θ = 2 :=
by
  sorry

end sector_area_max_angle_l347_34781


namespace max_value_negative_one_l347_34790

theorem max_value_negative_one (f : ℝ → ℝ) (hx : ∀ x, x < 1 → f x ≤ -1) :
  ∀ x, x < 1 → ∃ M, (∀ y, y < 1 → f y ≤ M) ∧ f x = M :=
sorry

end max_value_negative_one_l347_34790


namespace smallest_k_remainder_1_l347_34748

theorem smallest_k_remainder_1
  (k : ℤ) : 
  (k > 1) ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 4 = 1)
  ↔ k = 105 :=
by
  sorry

end smallest_k_remainder_1_l347_34748


namespace correct_fraction_l347_34749

theorem correct_fraction (x y : ℤ) (h : (5 / 6 : ℚ) * 384 = (x / y : ℚ) * 384 + 200) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l347_34749


namespace ellipse_foci_l347_34767

noncomputable def focal_coordinates (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

-- Given the equation of the ellipse: x^2 / a^2 + y^2 / b^2 = 1
def ellipse_equation (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

-- Proposition stating that if the ellipse equation holds for a=√5 and b=2, then the foci are at (± c, 0)
theorem ellipse_foci (x y : ℝ) (h : ellipse_equation x y (Real.sqrt 5) 2) :
  y = 0 ∧ (x = 1 ∨ x = -1) :=
sorry

end ellipse_foci_l347_34767


namespace Jen_distance_from_start_l347_34786

-- Define the rate of Jen's walking (in miles per hour)
def walking_rate : ℝ := 4

-- Define the time Jen walks forward (in hours)
def forward_time : ℝ := 2

-- Define the time Jen walks back (in hours)
def back_time : ℝ := 1

-- Define the distance walked forward
def distance_forward : ℝ := walking_rate * forward_time

-- Define the distance walked back
def distance_back : ℝ := walking_rate * back_time

-- Define the net distance from the starting point
def net_distance : ℝ := distance_forward - distance_back

-- Theorem stating the net distance from the starting point is 4.0 miles
theorem Jen_distance_from_start : net_distance = 4.0 := by
  sorry

end Jen_distance_from_start_l347_34786


namespace triangle_XYZ_ratio_l347_34776

theorem triangle_XYZ_ratio (XZ YZ : ℝ)
  (hXZ : XZ = 9) (hYZ : YZ = 40)
  (XY : ℝ) (hXY : XY = Real.sqrt (XZ ^ 2 + YZ ^ 2))
  (ZD : ℝ) (hZD : ZD = Real.sqrt (XZ * YZ))
  (XJ YJ : ℝ) (hXJ : XJ = Real.sqrt (XZ * (XZ + 2 * ZD)))
  (hYJ : YJ = Real.sqrt (YZ * (YZ + 2 * ZD)))
  (ratio : ℝ) (h_ratio : ratio = (XJ + YJ + XY) / XY) :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ ratio = p / q ∧ p + q = 203 := sorry

end triangle_XYZ_ratio_l347_34776


namespace least_alpha_prime_l347_34792

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_distinct_prime (α β : ℕ) : Prop :=
  α ≠ β ∧ is_prime α ∧ is_prime β

theorem least_alpha_prime (α : ℕ) :
  is_distinct_prime α (180 - 2 * α) → α ≥ 41 :=
sorry

end least_alpha_prime_l347_34792


namespace monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l347_34736

noncomputable def f (a x : ℝ) : ℝ := (a + 1) * Real.log x + a * x^2 + 1

theorem monotonicity_of_f {a : ℝ} (x : ℝ) (hx : 0 < x) :
  (f a x) = (f a x) := sorry

theorem abs_f_diff_ge_four_abs_diff {a x1 x2: ℝ} (ha : a ≤ -2) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  |f a x1 - f a x2| ≥ 4 * |x1 - x2| := sorry

end monotonicity_of_f_abs_f_diff_ge_four_abs_diff_l347_34736


namespace estate_area_correct_l347_34705

-- Define the basic parameters given in the problem
def scale : ℝ := 500  -- 500 miles per inch
def width_on_map : ℝ := 5  -- 5 inches
def height_on_map : ℝ := 3  -- 3 inches

-- Define actual dimensions based on the scale
def actual_width : ℝ := width_on_map * scale  -- actual width in miles
def actual_height : ℝ := height_on_map * scale  -- actual height in miles

-- Define the expected actual area of the estate
def actual_area : ℝ := 3750000  -- actual area in square miles

-- The main theorem to prove
theorem estate_area_correct :
  (actual_width * actual_height) = actual_area := by
  sorry

end estate_area_correct_l347_34705


namespace problem_statement_l347_34751

def f (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem problem_statement : f (f (-1)) = 10 := by
  sorry

end problem_statement_l347_34751


namespace money_bounds_l347_34714

   theorem money_bounds (a b : ℝ) (h₁ : 4 * a + 2 * b > 110) (h₂ : 2 * a + 3 * b = 105) : a > 15 ∧ b < 25 :=
   by
     sorry
   
end money_bounds_l347_34714


namespace fourth_number_in_pascals_triangle_row_15_l347_34763

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end fourth_number_in_pascals_triangle_row_15_l347_34763


namespace cassandra_makes_four_pies_l347_34744

-- Define the number of dozens and respective apples per dozen
def dozens : ℕ := 4
def apples_per_dozen : ℕ := 12

-- Define the total number of apples
def total_apples : ℕ := dozens * apples_per_dozen

-- Define apples per slice and slices per pie
def apples_per_slice : ℕ := 2
def slices_per_pie : ℕ := 6

-- Calculate the number of slices and number of pies based on conditions
def total_slices : ℕ := total_apples / apples_per_slice
def total_pies : ℕ := total_slices / slices_per_pie

-- Prove that the number of pies is 4
theorem cassandra_makes_four_pies : total_pies = 4 := by
  sorry

end cassandra_makes_four_pies_l347_34744


namespace minimum_a_condition_l347_34747

-- Define the quadratic function
def f (a x : ℝ) := x^2 + a * x + 1

-- Define the condition that the function remains non-negative in the open interval (0, 1/2)
def f_non_negative_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1 / 2 → f a x ≥ 0

-- State the theorem that the minimum value for a with the given condition is -5/2
theorem minimum_a_condition : ∀ (a : ℝ), f_non_negative_in_interval a → a ≥ -5 / 2 :=
by sorry

end minimum_a_condition_l347_34747


namespace exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l347_34704

theorem exceeding_speed_limit_percentages
  (percentage_A : ℕ) (percentage_B : ℕ) (percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  percentage_A = 30 ∧ percentage_B = 20 ∧ percentage_C = 25 := by
  sorry

theorem overall_exceeding_speed_limit_percentage
  (percentage_A percentage_B percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  (percentage_A + percentage_B + percentage_C) / 3 = 25 := by
  sorry

end exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l347_34704


namespace average_score_10_students_l347_34750

theorem average_score_10_students (x : ℝ)
  (h1 : 15 * 70 = 1050)
  (h2 : 25 * 78 = 1950)
  (h3 : 15 * 70 + 10 * x = 25 * 78) :
  x = 90 :=
sorry

end average_score_10_students_l347_34750


namespace lightest_height_is_135_l347_34710

-- Definitions based on the problem conditions
def heights_in_ratio (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x ∧ d = 6 * x

def height_condition (a c d : ℕ) : Prop :=
  d + a = c + 180

-- Lean statement describing the proof problem
theorem lightest_height_is_135 :
  ∀ (a b c d : ℕ),
  heights_in_ratio a b c d →
  height_condition a c d →
  a = 135 :=
by
  intro a b c d
  intro h_in_ratio h_condition
  sorry

end lightest_height_is_135_l347_34710


namespace angle_ADE_l347_34798

-- Definitions and conditions
variable (x : ℝ)

def angle_ABC := 60
def angle_CAD := x
def angle_BAD := x
def angle_BCA := 120 - 2 * x
def angle_DCE := 180 - (120 - 2 * x)

-- Theorem statement
theorem angle_ADE (x : ℝ) : angle_CAD x = x → angle_BAD x = x → angle_ABC = 60 → 
                            angle_DCE x = 180 - angle_BCA x → 
                            120 - 3 * x = 120 - 3 * x := 
by
  intro h1 h2 h3 h4
  sorry

end angle_ADE_l347_34798


namespace simplify_and_evaluate_l347_34709

theorem simplify_and_evaluate (x : ℤ) (h : x = 2) :
  (2 * x + 1) ^ 2 - (x + 3) * (x - 3) = 30 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l347_34709


namespace transformed_equation_sum_l347_34795

theorem transformed_equation_sum (a b : ℝ) (h_eqn : ∀ x : ℝ, x^2 - 6 * x - 5 = 0 ↔ (x + a)^2 = b) :
  a + b = 11 :=
sorry

end transformed_equation_sum_l347_34795


namespace carla_bought_marbles_l347_34741

def starting_marbles : ℕ := 2289
def total_marbles : ℝ := 2778.0

theorem carla_bought_marbles : (total_marbles - starting_marbles) = 489 := 
by
  sorry

end carla_bought_marbles_l347_34741


namespace cheryl_initial_mms_l347_34733

theorem cheryl_initial_mms (lunch_mms : ℕ) (dinner_mms : ℕ) (sister_mms : ℕ) (total_mms : ℕ) 
  (h1 : lunch_mms = 7) (h2 : dinner_mms = 5) (h3 : sister_mms = 13) (h4 : total_mms = lunch_mms + dinner_mms + sister_mms) : 
  total_mms = 25 := 
by 
  rw [h1, h2, h3] at h4
  exact h4

end cheryl_initial_mms_l347_34733


namespace total_marks_l347_34768

variable (A M SS Mu : ℝ)

-- Conditions
def cond1 : Prop := M = A - 20
def cond2 : Prop := SS = Mu + 10
def cond3 : Prop := Mu = 70
def cond4 : Prop := M = (9 / 10) * A

-- Theorem statement
theorem total_marks (A M SS Mu : ℝ) (h1 : cond1 A M)
                                      (h2 : cond2 SS Mu)
                                      (h3 : cond3 Mu)
                                      (h4 : cond4 A M) :
    A + M + SS + Mu = 530 :=
by 
  sorry

end total_marks_l347_34768


namespace sam_total_spent_l347_34771

-- Define the values of a penny and a dime in dollars
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.10

-- Define what Sam spent
def friday_spent : ℝ := 2 * penny_value
def saturday_spent : ℝ := 12 * dime_value

-- Define total spent
def total_spent : ℝ := friday_spent + saturday_spent

theorem sam_total_spent : total_spent = 1.22 := 
by
  -- The following is a placeholder for the actual proof
  sorry

end sam_total_spent_l347_34771


namespace betty_needs_more_flies_l347_34752

def flies_per_day := 2
def days_per_week := 7
def flies_needed_per_week := flies_per_day * days_per_week

def flies_caught_morning := 5
def flies_caught_afternoon := 6
def fly_escaped := 1

def flies_caught_total := flies_caught_morning + flies_caught_afternoon - fly_escaped

theorem betty_needs_more_flies : 
  flies_needed_per_week - flies_caught_total = 4 := by
  sorry

end betty_needs_more_flies_l347_34752


namespace range_of_a_l347_34702

open Set

variable {a x : ℝ}

def A (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem range_of_a (h : A a ∩ B = ∅) : a ≤ 0 ∨ a ≥ 6 := 
by 
  sorry

end range_of_a_l347_34702


namespace trig_identity_A_trig_identity_D_l347_34779

theorem trig_identity_A : 
  (Real.tan (25 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) + Real.tan (25 * Real.pi / 180) * Real.tan (20 * Real.pi / 180) = 1) :=
by sorry

theorem trig_identity_D : 
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180) = 4) :=
by sorry

end trig_identity_A_trig_identity_D_l347_34779


namespace goods_train_crossing_time_l347_34729

def speed_kmh : ℕ := 72
def train_length_m : ℕ := 230
def platform_length_m : ℕ := 290

noncomputable def crossing_time_seconds (speed_kmh train_length_m platform_length_m : ℕ) : ℕ :=
  let distance_m := train_length_m + platform_length_m
  let speed_ms := speed_kmh * 1000 / 3600
  distance_m / speed_ms

theorem goods_train_crossing_time :
  crossing_time_seconds speed_kmh train_length_m platform_length_m = 26 :=
by
  -- The proof should be filled in here
  sorry

end goods_train_crossing_time_l347_34729


namespace perpendicular_line_theorem_l347_34711

-- Mathematical definitions used in the condition.
def Line := Type
def Plane := Type

variables {l m : Line} {π : Plane}

-- Given the predicate that a line is perpendicular to another line on the plane
def is_perpendicular (l m : Line) (π : Plane) : Prop :=
sorry -- Definition of perpendicularity in Lean (abstracted here)

-- Given condition: l is perpendicular to the projection of m on plane π
axiom projection_of_oblique (m : Line) (π : Plane) : Line

-- The Perpendicular Line Theorem
theorem perpendicular_line_theorem (h : is_perpendicular l (projection_of_oblique m π) π) : is_perpendicular l m π :=
sorry

end perpendicular_line_theorem_l347_34711


namespace floor_e_equals_two_l347_34787

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l347_34787


namespace rhombus_perimeter_l347_34762

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
    let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
    (4 * s) = 52 :=
by
  sorry

end rhombus_perimeter_l347_34762


namespace part1_part2_l347_34764

theorem part1 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 2) : a^2 + b^2 = 21 :=
  sorry

theorem part2 (a b : ℝ) (h1 : a + b = 10) (h2 : a^2 + b^2 = 50^2) : a * b = -1200 :=
  sorry

end part1_part2_l347_34764


namespace brad_must_make_5_trips_l347_34772

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r ^ 2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r ^ 2 * h

theorem brad_must_make_5_trips (r_barrel h_barrel r_bucket h_bucket : ℝ)
    (h1 : r_barrel = 10) (h2 : h_barrel = 15) (h3 : r_bucket = 10) (h4 : h_bucket = 10) :
    let trips := volume_of_cylinder r_barrel h_barrel / volume_of_cone r_bucket h_bucket
    let trips_needed := Int.ceil trips
    trips_needed = 5 := 
by
  sorry

end brad_must_make_5_trips_l347_34772


namespace boys_tried_out_l347_34712

theorem boys_tried_out (B : ℕ) (girls : ℕ) (called_back : ℕ) (not_cut : ℕ) (total_tryouts : ℕ) 
  (h1 : girls = 39)
  (h2 : called_back = 26)
  (h3 : not_cut = 17)
  (h4 : total_tryouts = girls + B)
  (h5 : total_tryouts = called_back + not_cut) : 
  B = 4 := 
by
  sorry

end boys_tried_out_l347_34712


namespace number_of_teachers_under_40_in_sample_l347_34738

def proportion_teachers_under_40 (total_teachers teachers_under_40 : ℕ) : ℚ :=
  teachers_under_40 / total_teachers

def sample_teachers_under_40 (sample_size : ℕ) (proportion : ℚ) : ℚ :=
  sample_size * proportion

theorem number_of_teachers_under_40_in_sample
(total_teachers teachers_under_40 teachers_40_and_above sample_size : ℕ)
(h_total : total_teachers = 400)
(h_under_40 : teachers_under_40 = 250)
(h_40_and_above : teachers_40_and_above = 150)
(h_sample_size : sample_size = 80)
: sample_teachers_under_40 sample_size 
  (proportion_teachers_under_40 total_teachers teachers_under_40) = 50 := by
sorry

end number_of_teachers_under_40_in_sample_l347_34738


namespace consecutive_integers_sum_l347_34728

theorem consecutive_integers_sum (x : ℤ) (h : x * (x + 1) = 440) : x + (x + 1) = 43 :=
by sorry

end consecutive_integers_sum_l347_34728


namespace range_of_a_l347_34775

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ 0 ≤ a ∧ a < 4 := sorry

end range_of_a_l347_34775


namespace value_of_expression_is_correct_l347_34794

-- Defining the sub-expressions as Lean terms
def three_squared : ℕ := 3^2
def intermediate_result : ℕ := three_squared - 3
def final_result : ℕ := intermediate_result^2

-- The statement we need to prove
theorem value_of_expression_is_correct : final_result = 36 := by
  sorry

end value_of_expression_is_correct_l347_34794


namespace number_of_real_values_p_l347_34789

theorem number_of_real_values_p (p : ℝ) :
  (∀ p: ℝ, x^2 - (p + 1) * x + (p + 1)^2 = 0 -> (p + 1) ^ 2 = 0) ↔ p = -1 := by
  sorry

end number_of_real_values_p_l347_34789


namespace find_natural_number_l347_34785

-- Definitions reflecting the conditions and result
def is_sum_of_two_squares (n : ℕ) := ∃ a b : ℕ, a * a + b * b = n

def has_exactly_one_not_sum_of_two_squares (n : ℕ) :=
  ∃! x : ℤ, ¬is_sum_of_two_squares (x.natAbs % n)

theorem find_natural_number (n : ℕ) (h : n ≥ 2) : 
  has_exactly_one_not_sum_of_two_squares n ↔ n = 4 :=
sorry

end find_natural_number_l347_34785


namespace athletes_meeting_time_and_overtakes_l347_34735

-- Define the constants for the problem
noncomputable def track_length : ℕ := 400
noncomputable def speed1 : ℕ := 155
noncomputable def speed2 : ℕ := 200
noncomputable def speed3 : ℕ := 275

-- The main theorem for the problem statement
theorem athletes_meeting_time_and_overtakes :
  ∃ (t : ℚ) (n_overtakes : ℕ), 
  (t = 80 / 3) ∧
  (n_overtakes = 13) ∧
  (∀ n : ℕ, n * (track_length / 45) = t) ∧
  (∀ k : ℕ, k * (track_length / 120) = t) ∧
  (∀ m : ℕ, m * (track_length / 75) = t) := 
sorry

end athletes_meeting_time_and_overtakes_l347_34735


namespace claire_photos_l347_34700

theorem claire_photos (C : ℕ) (h1 : 3 * C = C + 20) : C = 10 :=
sorry

end claire_photos_l347_34700


namespace cost_for_paving_is_486_l347_34777

-- Definitions and conditions
def ratio_longer_side : ℝ := 4
def ratio_shorter_side : ℝ := 3
def diagonal : ℝ := 45
def cost_per_sqm : ℝ := 0.5 -- converting pence to pounds

-- Mathematical formulation
def longer_side (x : ℝ) : ℝ := ratio_longer_side * x
def shorter_side (x : ℝ) : ℝ := ratio_shorter_side * x
def area_of_rectangle (l w : ℝ) : ℝ := l * w
def cost_paving (area : ℝ) (cost_per_sqm : ℝ) : ℝ := area * cost_per_sqm

-- Main problem: given the conditions, prove that the cost is £486.
theorem cost_for_paving_is_486 (x : ℝ) 
  (h1 : (ratio_longer_side^2 + ratio_shorter_side^2) * x^2 = diagonal^2) :
  cost_paving (area_of_rectangle (longer_side x) (shorter_side x)) cost_per_sqm = 486 :=
by
  sorry

end cost_for_paving_is_486_l347_34777


namespace prime_product_correct_l347_34753

theorem prime_product_correct 
    (p1 : Nat := 1021031) (pr1 : Prime p1)
    (p2 : Nat := 237019) (pr2 : Prime p2) :
    p1 * p2 = 241940557349 :=
by
  sorry

end prime_product_correct_l347_34753


namespace solve_r_minus_s_l347_34731

noncomputable def r := 20
noncomputable def s := 4

theorem solve_r_minus_s
  (h1 : r^2 - 24 * r + 80 = 0)
  (h2 : s^2 - 24 * s + 80 = 0)
  (h3 : r > s) : r - s = 16 :=
by
  sorry

end solve_r_minus_s_l347_34731


namespace quadratic_has_two_real_roots_l347_34730

theorem quadratic_has_two_real_roots (k : ℝ) (h1 : k ≠ 0) (h2 : 4 - 12 * k ≥ 0) : 0 < k ∧ k ≤ 1 / 3 :=
sorry

end quadratic_has_two_real_roots_l347_34730


namespace smaller_solution_of_quadratic_l347_34721

theorem smaller_solution_of_quadratic :
  ∀ x : ℝ, x^2 + 17 * x - 72 = 0 → x = -24 ∨ x = 3 :=
by sorry

end smaller_solution_of_quadratic_l347_34721


namespace upper_limit_l347_34723

noncomputable def upper_limit_Arun (w : ℝ) (X : ℝ) : Prop :=
  (w > 66 ∧ w < X) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 69) ∧ ((66 + X) / 2 = 68)

theorem upper_limit (w : ℝ) (X : ℝ) (h : upper_limit_Arun w X) : X = 69 :=
by sorry

end upper_limit_l347_34723


namespace integer_x_cubed_prime_l347_34725

theorem integer_x_cubed_prime (x : ℕ) : 
  (∃ p : ℕ, Prime p ∧ (2^x + x^2 + 25 = p^3)) → x = 6 :=
by
  sorry

end integer_x_cubed_prime_l347_34725


namespace sum_of_integers_equals_75_l347_34793

theorem sum_of_integers_equals_75 
  (n m : ℤ) 
  (h1 : n * (n + 1) * (n + 2) = 924) 
  (h2 : m * (m + 1) * (m + 2) * (m + 3) = 924) 
  (sum_seven_integers : ℤ := n + (n + 1) + (n + 2) + m + (m + 1) + (m + 2) + (m + 3)) :
  sum_seven_integers = 75 := 
  sorry

end sum_of_integers_equals_75_l347_34793


namespace greatest_N_exists_l347_34708

def is_condition_satisfied (N : ℕ) (xs : Fin N → ℤ) : Prop :=
  ∀ i j : Fin N, i ≠ j → ¬ (1111 ∣ ((xs i) * (xs i) - (xs i) * (xs j)))

theorem greatest_N_exists : ∃ N : ℕ, (∀ M : ℕ, (∀ xs : Fin M → ℤ, is_condition_satisfied M xs → M ≤ N)) ∧ N = 1000 :=
by
  sorry

end greatest_N_exists_l347_34708


namespace anne_already_made_8_drawings_l347_34740

-- Define the conditions as Lean definitions
def num_markers : ℕ := 12
def drawings_per_marker : ℚ := 3 / 2 -- Equivalent to 1.5
def remaining_drawings : ℕ := 10

-- Calculate the total number of drawings Anne can make with her markers
def total_drawings : ℚ := num_markers * drawings_per_marker

-- Calculate the already made drawings
def already_made_drawings : ℚ := total_drawings - remaining_drawings

-- The theorem to prove
theorem anne_already_made_8_drawings : already_made_drawings = 8 := 
by 
  have h1 : total_drawings = 18 := by sorry -- Calculating total drawings as 18
  have h2 : already_made_drawings = 8 := by sorry -- Calculating already made drawings as total drawings minus remaining drawings
  exact h2

end anne_already_made_8_drawings_l347_34740


namespace negation_correct_l347_34757

namespace NegationProof

-- Define the original proposition 
def orig_prop : Prop := ∃ x : ℝ, x ≤ 0

-- Define the negation of the original proposition
def neg_prop : Prop := ∀ x : ℝ, x > 0

-- The theorem we need to prove
theorem negation_correct : ¬ orig_prop = neg_prop := by
  sorry

end NegationProof

end negation_correct_l347_34757
