import Mathlib

namespace NUMINAMATH_GPT_tan_alpha_eq_neg_sqrt_15_l1117_111728

/-- Given α in the interval (0, π) and the equation tan(2α) = sin(α) / (2 + cos(α)), prove that tan(α) = -√15. -/
theorem tan_alpha_eq_neg_sqrt_15 (α : ℝ) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_sqrt_15_l1117_111728


namespace NUMINAMATH_GPT_find_first_term_l1117_111713

noncomputable def firstTermOfGeometricSeries (a r : ℝ) : Prop :=
  (a / (1 - r) = 30) ∧ (a^2 / (1 - r^2) = 120)

theorem find_first_term :
  ∃ a r : ℝ, firstTermOfGeometricSeries a r ∧ a = 120 / 17 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_l1117_111713


namespace NUMINAMATH_GPT_digits_difference_l1117_111736

theorem digits_difference (d A B : ℕ) (hd : d > 7) (h : d^2 * A + d * B + d^2 * A + d * A = 1 * d^2 + 7 * d + 2) : 
  A - B = 4 := 
sorry

end NUMINAMATH_GPT_digits_difference_l1117_111736


namespace NUMINAMATH_GPT_winning_percentage_l1117_111709

theorem winning_percentage (total_votes majority : ℕ) (h1 : total_votes = 455) (h2 : majority = 182) :
  ∃ P : ℕ, P = 70 ∧ (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority := 
sorry

end NUMINAMATH_GPT_winning_percentage_l1117_111709


namespace NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1117_111726

theorem sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_120_eq_sqrt3_div_2_l1117_111726


namespace NUMINAMATH_GPT_total_people_bought_tickets_l1117_111750

-- Definitions based on the conditions from step a)
def num_adults := 375
def num_children := 3 * num_adults
def total_revenue := 7 * num_adults + 3 * num_children

-- Statement of the theorem based on the question in step a)
theorem total_people_bought_tickets : (num_adults + num_children) = 1500 :=
by
  -- The proof is omitted, but we're ensuring the correctness of the theorem statement.
  sorry

end NUMINAMATH_GPT_total_people_bought_tickets_l1117_111750


namespace NUMINAMATH_GPT_problem_trigonometric_identity_l1117_111771

-- Define the problem conditions
theorem problem_trigonometric_identity
  (α : ℝ)
  (h : 3 * Real.sin (33 * Real.pi / 14 + α) = -5 * Real.cos (5 * Real.pi / 14 + α)) :
  Real.tan (5 * Real.pi / 14 + α) = -5 / 3 :=
sorry

end NUMINAMATH_GPT_problem_trigonometric_identity_l1117_111771


namespace NUMINAMATH_GPT_triangle_sine_cosine_l1117_111706

theorem triangle_sine_cosine (a b A : ℝ) (B C : ℝ) (c : ℝ) 
  (ha : a = Real.sqrt 7) 
  (hb : b = 2) 
  (hA : A = 60 * Real.pi / 180) 
  (hsinB : Real.sin B = Real.sin B := by sorry)
  (hc : c = 3 := by sorry) :
  (Real.sin B = Real.sqrt 21 / 7) ∧ (c = 3) := 
sorry

end NUMINAMATH_GPT_triangle_sine_cosine_l1117_111706


namespace NUMINAMATH_GPT_solution_l1117_111758

-- Conditions
def x : ℚ := 3/5
def y : ℚ := 5/3

-- Proof problem
theorem solution : (1/3) * x^8 * y^9 = 5/9 := sorry

end NUMINAMATH_GPT_solution_l1117_111758


namespace NUMINAMATH_GPT_range_of_a_l1117_111711

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {x : ℝ | (a * x - 1) / (x - a) < 0}

theorem range_of_a (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : (1 / 3 : ℝ) ≤ a ∧ a < 1 / 2 ∨ 2 < a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1117_111711


namespace NUMINAMATH_GPT_maximize_rectangle_area_l1117_111764

theorem maximize_rectangle_area (l w : ℝ) (h : l + w ≥ 40) : l * w ≤ 400 :=
by sorry

end NUMINAMATH_GPT_maximize_rectangle_area_l1117_111764


namespace NUMINAMATH_GPT_ratio_of_probabilities_l1117_111745

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_probabilities_l1117_111745


namespace NUMINAMATH_GPT_fred_earned_correctly_l1117_111700

-- Assuming Fred's earnings from different sources
def fred_earned_newspapers := 16 -- dollars
def fred_earned_cars := 74 -- dollars

-- Total earnings over the weekend
def fred_earnings := fred_earned_newspapers + fred_earned_cars

-- Given condition that Fred earned 90 dollars over the weekend
def fred_earnings_given := 90 -- dollars

-- The theorem stating that Fred's total earnings match the given earnings
theorem fred_earned_correctly : fred_earnings = fred_earnings_given := by
  sorry

end NUMINAMATH_GPT_fred_earned_correctly_l1117_111700


namespace NUMINAMATH_GPT_domain_f_l1117_111759

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.sqrt (x + 5)) + Real.log (2^x + 1)

theorem domain_f :
  {x : ℝ | (-5 ≤ x)} = {x : ℝ | f x ∈ Set.univ} := sorry

end NUMINAMATH_GPT_domain_f_l1117_111759


namespace NUMINAMATH_GPT_negation_statement_l1117_111720

variables {a b c : ℝ}

theorem negation_statement (h : a * b * c = 0) : ¬(a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end NUMINAMATH_GPT_negation_statement_l1117_111720


namespace NUMINAMATH_GPT_potatoes_cost_l1117_111717

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end NUMINAMATH_GPT_potatoes_cost_l1117_111717


namespace NUMINAMATH_GPT_third_team_pieces_l1117_111797

theorem third_team_pieces (total_pieces : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) : 
  total_pieces = 500 → first_team = 189 → second_team = 131 → third_team = total_pieces - first_team - second_team → third_team = 180 :=
by
  intros h_total h_first h_second h_third
  rw [h_total, h_first, h_second] at h_third
  exact h_third

end NUMINAMATH_GPT_third_team_pieces_l1117_111797


namespace NUMINAMATH_GPT_integer_pairs_summing_to_six_l1117_111753

theorem integer_pairs_summing_to_six :
  ∃ m n : ℤ, m + n + m * n = 6 ∧ ((m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0)) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_summing_to_six_l1117_111753


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l1117_111789

open Nat

theorem ratio_of_boys_to_girls
    (B G : ℕ) 
    (boys_avg : ℕ) 
    (girls_avg : ℕ) 
    (class_avg : ℕ)
    (h1 : boys_avg = 90)
    (h2 : girls_avg = 96)
    (h3 : class_avg = 94)
    (h4 : 94 * (B + G) = 90 * B + 96 * G) :
    2 * B = G :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l1117_111789


namespace NUMINAMATH_GPT_chord_length_l1117_111751

theorem chord_length (x y : ℝ) :
  (x^2 + y^2 - 2 * x - 4 * y = 0) →
  (x + 2 * y - 5 + Real.sqrt 5 = 0) →
  ∃ l, l = 4 :=
by
  intros h_circle h_line
  sorry

end NUMINAMATH_GPT_chord_length_l1117_111751


namespace NUMINAMATH_GPT_negation_of_existential_statement_l1117_111792

variable (A : Set ℝ)

theorem negation_of_existential_statement :
  ¬(∃ x ∈ A, x^2 - 2 * x - 3 > 0) ↔ ∀ x ∈ A, x^2 - 2 * x - 3 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_existential_statement_l1117_111792


namespace NUMINAMATH_GPT_min_b_minus_a_l1117_111796

noncomputable def f (x : ℝ) : ℝ := 1 + x - (x^2) / 2 + (x^3) / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + (x^2) / 2 - (x^3) / 3
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_b_minus_a (a b : ℤ) (h : ∀ x, F x = 0 → a ≤ x ∧ x ≤ b) (h_a_lt_b : a < b) : b - a = 3 :=
sorry

end NUMINAMATH_GPT_min_b_minus_a_l1117_111796


namespace NUMINAMATH_GPT_PQ_PR_QR_div_l1117_111707

theorem PQ_PR_QR_div (p q r : ℝ)
    (midQR : p = 0) (midPR : q = 0) (midPQ : r = 0) :
    (4 * (q ^ 2 + r ^ 2) + 4 * (p ^ 2 + r ^ 2) + 4 * (p ^ 2 + q ^ 2)) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by {
    sorry
}

end NUMINAMATH_GPT_PQ_PR_QR_div_l1117_111707


namespace NUMINAMATH_GPT_a_pow_b_iff_a_minus_1_b_positive_l1117_111748

theorem a_pow_b_iff_a_minus_1_b_positive (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : 
  (a^b > 1) ↔ ((a - 1) * b > 0) := 
sorry

end NUMINAMATH_GPT_a_pow_b_iff_a_minus_1_b_positive_l1117_111748


namespace NUMINAMATH_GPT_diameter_of_circle_is_60_l1117_111779

noncomputable def diameter_of_circle (M N : ℝ) : ℝ :=
  if h : N ≠ 0 then 2 * (M / N * (1 / (2 * Real.pi))) else 0

theorem diameter_of_circle_is_60 (M N : ℝ) (h : M / N = 15) :
  diameter_of_circle M N = 60 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_circle_is_60_l1117_111779


namespace NUMINAMATH_GPT_simplify_rationalize_expr_l1117_111761

theorem simplify_rationalize_expr : 
  (1 / (2 + 1 / (Real.sqrt 5 - 2))) = (4 - Real.sqrt 5) / 11 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_rationalize_expr_l1117_111761


namespace NUMINAMATH_GPT_problem1_problem2_l1117_111774

-- Definitions based on the conditions
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 5

-- Problem 1: Both doctor A and B must join the team
theorem problem1 : ∃ (ways : ℕ), ways = 816 :=
  by
    let remaining_doctors := total_doctors - 2
    let choose := remaining_doctors.choose (team_size - 2)
    have h1 : choose = 816 := sorry
    exact ⟨choose, h1⟩

-- Problem 2: At least one of doctors A or B must join the team
theorem problem2 : ∃ (ways : ℕ), ways = 5661 :=
  by
    let remaining_doctors := total_doctors - 1
    let scenario1 := 2 * remaining_doctors.choose (team_size - 1)
    let scenario2 := (total_doctors - 2).choose (team_size - 2)
    let total_ways := scenario1 + scenario2
    have h2 : total_ways = 5661 := sorry
    exact ⟨total_ways, h2⟩

end NUMINAMATH_GPT_problem1_problem2_l1117_111774


namespace NUMINAMATH_GPT_total_payment_l1117_111754

def cement_bags := 500
def cost_per_bag := 10
def lorries := 20
def tons_per_lorry := 10
def cost_per_ton := 40

theorem total_payment : cement_bags * cost_per_bag + lorries * tons_per_lorry * cost_per_ton = 13000 := by
  sorry

end NUMINAMATH_GPT_total_payment_l1117_111754


namespace NUMINAMATH_GPT_swimming_pool_length_correct_l1117_111740

noncomputable def swimming_pool_length (V_removed: ℝ) (W: ℝ) (H: ℝ) (gal_to_cuft: ℝ): ℝ :=
  V_removed / (W * H / gal_to_cuft)

theorem swimming_pool_length_correct:
  swimming_pool_length 3750 25 0.5 7.48052 = 40.11 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_length_correct_l1117_111740


namespace NUMINAMATH_GPT_allocation_methods_count_l1117_111725

theorem allocation_methods_count :
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  ∃ (allocation_methods : ℕ), allocation_methods = 12 := 
by
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  use doctors * Nat.choose nurses 2
  sorry

end NUMINAMATH_GPT_allocation_methods_count_l1117_111725


namespace NUMINAMATH_GPT_solve_for_x_l1117_111702

theorem solve_for_x (x : ℝ) (h : 3 * (x - 5) = 3 * (18 - 5)) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1117_111702


namespace NUMINAMATH_GPT_exists_gcd_one_l1117_111747

theorem exists_gcd_one (p q r : ℤ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : Int.gcd p (Int.gcd q r) = 1) : ∃ a : ℤ, Int.gcd p (q + a * r) = 1 :=
sorry

end NUMINAMATH_GPT_exists_gcd_one_l1117_111747


namespace NUMINAMATH_GPT_a_can_be_any_real_l1117_111765

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : e ≠ 0) :
  ∃ a : ℝ, true :=
by sorry

end NUMINAMATH_GPT_a_can_be_any_real_l1117_111765


namespace NUMINAMATH_GPT_tan_435_eq_2_add_sqrt_3_l1117_111743

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_435_eq_2_add_sqrt_3_l1117_111743


namespace NUMINAMATH_GPT_largest_root_in_interval_l1117_111712

theorem largest_root_in_interval :
  ∃ (r : ℝ), (2 < r ∧ r < 3) ∧ (∃ (a_2 a_1 a_0 : ℝ), 
    |a_2| ≤ 3 ∧ |a_1| ≤ 3 ∧ |a_0| ≤ 3 ∧ a_2 + a_1 + a_0 = -6 ∧ r^3 + a_2 * r^2 + a_1 * r + a_0 = 0) :=
sorry

end NUMINAMATH_GPT_largest_root_in_interval_l1117_111712


namespace NUMINAMATH_GPT_problem1_problem2_l1117_111769

theorem problem1 (a b : ℝ) :
  5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b := 
by sorry

theorem problem2 (m n : ℝ) :
  -5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1117_111769


namespace NUMINAMATH_GPT_solve_inequality1_solve_inequality2_l1117_111744

-- Proof problem 1
theorem solve_inequality1 (x : ℝ) : 
  2 < |2 * x - 5| → |2 * x - 5| ≤ 7 → -1 ≤ x ∧ x < (3 / 2) ∨ (7 / 2) < x ∧ x ≤ 6 :=
sorry

-- Proof problem 2
theorem solve_inequality2 (x : ℝ) : 
  (1 / (x - 1)) > (x + 1) → x < - Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_solve_inequality1_solve_inequality2_l1117_111744


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l1117_111749

theorem radius_of_inscribed_circle (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 10) (hc : c = 20)
  (h : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))) :
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by
  -- Statements and conditions are setup, but the proof is omitted.
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l1117_111749


namespace NUMINAMATH_GPT_total_valid_votes_l1117_111798

theorem total_valid_votes (V : ℝ) (H_majority : 0.70 * V - 0.30 * V = 188) : V = 470 :=
by
  sorry

end NUMINAMATH_GPT_total_valid_votes_l1117_111798


namespace NUMINAMATH_GPT_inverse_proposition_l1117_111788

theorem inverse_proposition (a : ℝ) :
  (a > 1 → a > 0) → (a > 0 → a > 1) :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_inverse_proposition_l1117_111788


namespace NUMINAMATH_GPT_number_of_factors_l1117_111735

theorem number_of_factors : 
  ∃ (count : ℕ), count = 45 ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 500) → 
      ∃ a b : ℤ, (x - a) * (x - b) = x^2 + 2 * x - n) :=
by
  sorry

end NUMINAMATH_GPT_number_of_factors_l1117_111735


namespace NUMINAMATH_GPT_probability_neither_nearsighted_l1117_111783

-- Definitions based on problem conditions
def P_A : ℝ := 0.4
def P_not_A : ℝ := 1 - P_A
def event_B₁_not_nearsighted : Prop := true
def event_B₂_not_nearsighted : Prop := true

-- Independence assumption
variables (indep_B₁_B₂ : event_B₁_not_nearsighted) (event_B₂_not_nearsighted)

-- Theorem statement
theorem probability_neither_nearsighted (H1 : P_A = 0.4) (H2 : P_not_A = 0.6)
  (indep_B₁_B₂ : event_B₁_not_nearsighted ∧ event_B₂_not_nearsighted) :
  P_not_A * P_not_A = 0.36 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_neither_nearsighted_l1117_111783


namespace NUMINAMATH_GPT_proof_problem_l1117_111727

theorem proof_problem (a b : ℝ) (h1 : (5 * a + 2)^(1/3) = 3) (h2 : (3 * a + b - 1)^(1/2) = 4) :
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3)^(1/2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1117_111727


namespace NUMINAMATH_GPT_Sara_sister_notebooks_l1117_111793

theorem Sara_sister_notebooks :
  let initial_notebooks := 4 
  let ordered_notebooks := (3 / 2) * initial_notebooks -- 150% more notebooks
  let notebooks_after_order := initial_notebooks + ordered_notebooks
  let notebooks_after_loss := notebooks_after_order - 2 -- lost 2 notebooks
  let sold_notebooks := (1 / 4) * notebooks_after_loss -- sold 25% of remaining notebooks
  let notebooks_after_sales := notebooks_after_loss - sold_notebooks
  let notebooks_after_giveaway := notebooks_after_sales - 3 -- gave away 3 notebooks
  notebooks_after_giveaway = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_Sara_sister_notebooks_l1117_111793


namespace NUMINAMATH_GPT_min_value_expr_l1117_111746

theorem min_value_expr (x y : ℝ) : 
  ∃ x y : ℝ, (x, y) = (4, 0) ∧ (∀ x y : ℝ, x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ -22) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1117_111746


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1117_111715

theorem solution_set_of_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ (x > 2 ∨ x < -1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1117_111715


namespace NUMINAMATH_GPT_sine_angle_greater_implies_angle_greater_l1117_111767

noncomputable def triangle := {ABC : Type* // Π A B C : ℕ, 
  A + B + C = 180 ∧ 0 < A ∧ A < 180 ∧ 0 < B ∧ B < 180 ∧ 0 < C ∧ C < 180}

variables {A B C : ℕ} (T : triangle)

theorem sine_angle_greater_implies_angle_greater (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180)
  (h3 : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) (h_sine : Real.sin A > Real.sin B) :
  A > B := 
sorry

end NUMINAMATH_GPT_sine_angle_greater_implies_angle_greater_l1117_111767


namespace NUMINAMATH_GPT_find_T_l1117_111768

theorem find_T : 
  ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (1 / 5) * (1 / 4) * 120 ∧ T = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_T_l1117_111768


namespace NUMINAMATH_GPT_joe_money_fraction_l1117_111791

theorem joe_money_fraction :
  ∃ f : ℝ,
    (200 : ℝ) = 160 + (200 - 160) ∧
    160 - 160 * f - 20 = 40 + 160 * f + 20 ∧
    f = 1 / 4 :=
by
  -- The proof should go here.
  sorry

end NUMINAMATH_GPT_joe_money_fraction_l1117_111791


namespace NUMINAMATH_GPT_find_x_5pi_over_4_l1117_111773

open Real

theorem find_x_5pi_over_4 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = -sqrt 2) : x = 5 * π / 4 := 
sorry

end NUMINAMATH_GPT_find_x_5pi_over_4_l1117_111773


namespace NUMINAMATH_GPT_cakes_remain_l1117_111703

def initial_cakes := 110
def sold_cakes := 75
def new_cakes := 76

theorem cakes_remain : (initial_cakes - sold_cakes) + new_cakes = 111 :=
by
  sorry

end NUMINAMATH_GPT_cakes_remain_l1117_111703


namespace NUMINAMATH_GPT_divisor_is_50_l1117_111772

theorem divisor_is_50 (D : ℕ) (h1 : ∃ n, n = 44 * 432 ∧ n % 44 = 0)
                      (h2 : ∃ n, n = 44 * 432 ∧ n % D = 8) : D = 50 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_50_l1117_111772


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l1117_111777

theorem smallest_sum_of_squares :
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 ≥ 36 ∧ y^2 ≥ 36 ∧ x^2 + y^2 = 625 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l1117_111777


namespace NUMINAMATH_GPT_calc_tan_fraction_l1117_111701

theorem calc_tan_fraction :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h_tan_30 : Real.tan (30 * Real.pi / 180) = Real.sqrt 3 / 3 := by sorry
  sorry

end NUMINAMATH_GPT_calc_tan_fraction_l1117_111701


namespace NUMINAMATH_GPT_sum_of_legs_of_right_triangle_l1117_111730

theorem sum_of_legs_of_right_triangle
  (a b : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b = a + 2)
  (h3 : a^2 + b^2 = 50^2) :
  a + b = 70 := by
  sorry

end NUMINAMATH_GPT_sum_of_legs_of_right_triangle_l1117_111730


namespace NUMINAMATH_GPT_count_random_events_l1117_111781

-- Definitions based on conditions in the problem
def total_products : ℕ := 100
def genuine_products : ℕ := 95
def defective_products : ℕ := 5
def drawn_products : ℕ := 6

-- Events definitions
def event_1 := drawn_products > defective_products  -- at least 1 genuine product
def event_2 := drawn_products ≥ 3  -- at least 3 defective products
def event_3 := drawn_products = defective_products  -- all 6 are defective
def event_4 := drawn_products - 2 = 4  -- 2 defective and 4 genuine products

-- Dummy definition for random event counter state in the problem context
def random_events : ℕ := 2

-- Main theorem statement
theorem count_random_events :
  (event_1 → true) ∧ 
  (event_2 ∧ ¬ event_3 ∧ event_4) →
  random_events = 2 :=
by
  sorry

end NUMINAMATH_GPT_count_random_events_l1117_111781


namespace NUMINAMATH_GPT_smallest_x_l1117_111757

theorem smallest_x (x : ℚ) (h : 7 * (4 * x^2 + 4 * x + 5) = x * (4 * x - 35)) : 
  x = -5/3 ∨ x = -7/8 := by
  sorry

end NUMINAMATH_GPT_smallest_x_l1117_111757


namespace NUMINAMATH_GPT_heather_lighter_than_combined_weights_l1117_111722

noncomputable def heather_weight : ℝ := 87.5
noncomputable def emily_weight : ℝ := 45.3
noncomputable def elizabeth_weight : ℝ := 38.7
noncomputable def george_weight : ℝ := 56.9

theorem heather_lighter_than_combined_weights :
  heather_weight - (emily_weight + elizabeth_weight + george_weight) = -53.4 :=
by 
  sorry

end NUMINAMATH_GPT_heather_lighter_than_combined_weights_l1117_111722


namespace NUMINAMATH_GPT_unique_solution_x_y_z_l1117_111770

theorem unique_solution_x_y_z (x y z : ℕ) (h1 : Prime y) (h2 : ¬ z % 3 = 0) (h3 : ¬ z % y = 0) :
    x^3 - y^3 = z^2 ↔ (x, y, z) = (8, 7, 13) := by
  sorry

end NUMINAMATH_GPT_unique_solution_x_y_z_l1117_111770


namespace NUMINAMATH_GPT_Frank_time_correct_l1117_111786

def Dave_time := 10
def Chuck_time := 5 * Dave_time
def Erica_time := 13 * Chuck_time / 10
def Frank_time := 12 * Erica_time / 10

theorem Frank_time_correct : Frank_time = 78 :=
by
  sorry

end NUMINAMATH_GPT_Frank_time_correct_l1117_111786


namespace NUMINAMATH_GPT_geometric_sequence_x_l1117_111714

theorem geometric_sequence_x (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_x_l1117_111714


namespace NUMINAMATH_GPT_range_of_sum_coords_on_ellipse_l1117_111724

theorem range_of_sum_coords_on_ellipse (x y : ℝ) 
  (h : x^2 / 144 + y^2 / 25 = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := 
sorry

end NUMINAMATH_GPT_range_of_sum_coords_on_ellipse_l1117_111724


namespace NUMINAMATH_GPT_overlapping_area_of_congruent_isosceles_triangles_l1117_111710

noncomputable def isosceles_right_triangle (hypotenuse : ℝ) := 
  {l : ℝ // l = hypotenuse / Real.sqrt 2}

theorem overlapping_area_of_congruent_isosceles_triangles (hypotenuse : ℝ) 
  (A₁ A₂ : isosceles_right_triangle hypotenuse) (h_congruent : A₁ = A₂) :
  hypotenuse = 10 → 
  let leg := hypotenuse / Real.sqrt 2 
  let area := (leg * leg) / 2 
  let shared_area := area / 2 
  shared_area = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_overlapping_area_of_congruent_isosceles_triangles_l1117_111710


namespace NUMINAMATH_GPT_jacob_total_distance_l1117_111775

/- Jacob jogs at a constant rate of 4 miles per hour.
   He jogs for 2 hours, then stops to take a rest for 30 minutes.
   After the break, he continues jogging for another 1 hour.
   Prove that the total distance jogged by Jacob is 12.0 miles.
-/
theorem jacob_total_distance :
  let joggingSpeed := 4 -- in miles per hour
  let jogBeforeBreak := 2 -- in hours
  let restDuration := 0.5 -- in hours (though it does not affect the distance)
  let jogAfterBreak := 1 -- in hours
  let totalDistance := joggingSpeed * jogBeforeBreak + joggingSpeed * jogAfterBreak
  totalDistance = 12.0 := 
by
  sorry

end NUMINAMATH_GPT_jacob_total_distance_l1117_111775


namespace NUMINAMATH_GPT_kids_french_fries_cost_l1117_111739

noncomputable def cost_burger : ℝ := 5
noncomputable def cost_fries : ℝ := 3
noncomputable def cost_soft_drink : ℝ := 3
noncomputable def cost_special_burger_meal : ℝ := 9.50
noncomputable def cost_kids_burger : ℝ := 3
noncomputable def cost_kids_juice_box : ℝ := 2
noncomputable def cost_kids_meal : ℝ := 5
noncomputable def savings : ℝ := 10

noncomputable def total_adult_meal_individual : ℝ := 2 * cost_burger + 2 * cost_fries + 2 * cost_soft_drink
noncomputable def total_adult_meal_deal : ℝ := 2 * cost_special_burger_meal

noncomputable def total_kids_meal_individual (F : ℝ) : ℝ := 2 * cost_kids_burger + 2 * F + 2 * cost_kids_juice_box
noncomputable def total_kids_meal_deal : ℝ := 2 * cost_kids_meal

noncomputable def total_cost_individual (F : ℝ) : ℝ := total_adult_meal_individual + total_kids_meal_individual F
noncomputable def total_cost_deal : ℝ := total_adult_meal_deal + total_kids_meal_deal

theorem kids_french_fries_cost : ∃ F : ℝ, total_cost_individual F - total_cost_deal = savings ∧ F = 3.50 := 
by
  use 3.50
  sorry

end NUMINAMATH_GPT_kids_french_fries_cost_l1117_111739


namespace NUMINAMATH_GPT_sum_of_squares_divisibility_l1117_111799

theorem sum_of_squares_divisibility (n : ℤ) : 
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  (S % 4 = 0 ∧ S % 3 ≠ 0) :=
by
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  sorry

end NUMINAMATH_GPT_sum_of_squares_divisibility_l1117_111799


namespace NUMINAMATH_GPT_third_side_length_l1117_111738

theorem third_side_length (x : ℝ) (h1 : 2 + 4 > x) (h2 : 4 + x > 2) (h3 : x + 2 > 4) : x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_third_side_length_l1117_111738


namespace NUMINAMATH_GPT_pyramid_volume_and_base_edge_l1117_111705

theorem pyramid_volume_and_base_edge:
  ∀ (r: ℝ) (h: ℝ) (_: r = 5) (_: h = 10), 
  ∃ s V: ℝ,
    s = (10 * Real.sqrt 6) / 3 ∧ 
    V = (2000 / 9) :=
by
    sorry

end NUMINAMATH_GPT_pyramid_volume_and_base_edge_l1117_111705


namespace NUMINAMATH_GPT_car_speed_l1117_111780

theorem car_speed (v : ℝ) : 
  (4 + (1 / (80 / 3600))) = (1 / (v / 3600)) → v = 3600 / 49 :=
sorry

end NUMINAMATH_GPT_car_speed_l1117_111780


namespace NUMINAMATH_GPT_equation_of_parallel_line_l1117_111760

theorem equation_of_parallel_line : 
  ∃ l : ℝ, (∀ x y : ℝ, 2 * x - 3 * y + 8 = 0 ↔ l = 2 * x - 3 * y + 8) :=
sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l1117_111760


namespace NUMINAMATH_GPT_car_tank_capacity_is_12_gallons_l1117_111762

noncomputable def truck_tank_capacity : ℕ := 20
noncomputable def truck_tank_half_full : ℕ := truck_tank_capacity / 2
noncomputable def car_tank_third_full (car_tank_capacity : ℕ) : ℕ := car_tank_capacity / 3
noncomputable def total_gallons_added : ℕ := 18

theorem car_tank_capacity_is_12_gallons (car_tank_capacity : ℕ) 
    (h1 : truck_tank_half_full + (car_tank_third_full car_tank_capacity) + 18 = truck_tank_capacity + car_tank_capacity) 
    (h2 : total_gallons_added = 18) : car_tank_capacity = 12 := 
by
  sorry

end NUMINAMATH_GPT_car_tank_capacity_is_12_gallons_l1117_111762


namespace NUMINAMATH_GPT_lower_denomination_cost_l1117_111756

-- Conditions
def total_stamps : ℕ := 20
def total_cost_cents : ℕ := 706
def high_denomination_stamps : ℕ := 18
def high_denomination_cost : ℕ := 37
def low_denomination_stamps : ℕ := total_stamps - high_denomination_stamps

-- Theorem proving the cost of the lower denomination stamp.
theorem lower_denomination_cost :
  ∃ (x : ℕ), (high_denomination_stamps * high_denomination_cost) + (low_denomination_stamps * x) = total_cost_cents
  ∧ x = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_lower_denomination_cost_l1117_111756


namespace NUMINAMATH_GPT_current_population_l1117_111718

def initial_population : ℕ := 4200
def percentage_died : ℕ := 10
def percentage_left : ℕ := 15

theorem current_population (pop : ℕ) (died left : ℕ) 
  (h1 : pop = initial_population) 
  (h2 : died = pop * percentage_died / 100) 
  (h3 : left = (pop - died) * percentage_left / 100) 
  (h4 : ∀ remaining, remaining = pop - died - left) 
  : (pop - died - left) = 3213 := 
by sorry

end NUMINAMATH_GPT_current_population_l1117_111718


namespace NUMINAMATH_GPT_discriminant_eq_13_l1117_111778

theorem discriminant_eq_13 (m : ℝ) (h : (3)^2 - 4*1*(-m) = 13) : m = 1 :=
sorry

end NUMINAMATH_GPT_discriminant_eq_13_l1117_111778


namespace NUMINAMATH_GPT_find_X_l1117_111719

def tax_problem (X I T : ℝ) (income : ℝ) (total_tax : ℝ) :=
  (income = 56000) ∧ (total_tax = 8000) ∧ (T = 0.12 * X + 0.20 * (I - X))

theorem find_X :
  ∃ X : ℝ, ∀ I T : ℝ, tax_problem X I T 56000 8000 → X = 40000 := 
  by
    sorry

end NUMINAMATH_GPT_find_X_l1117_111719


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1117_111755

-- Define the universal set U
def U : Set ℕ := {2, 3, 4}

-- Define set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Prove the complement of A in U is {4}
theorem complement_of_A_in_U : C_U_A = {4} := 
  by 
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1117_111755


namespace NUMINAMATH_GPT_sets_satisfy_union_l1117_111729

theorem sets_satisfy_union (A : Set Int) : (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (X : Finset (Set Int)), X.card = 4 ∧ ∀ B ∈ X, A = B) :=
  sorry

end NUMINAMATH_GPT_sets_satisfy_union_l1117_111729


namespace NUMINAMATH_GPT_percentage_invalid_votes_l1117_111766

theorem percentage_invalid_votes
  (total_votes : ℕ)
  (votes_for_A : ℕ)
  (candidate_A_percentage : ℝ)
  (total_votes_count : total_votes = 560000)
  (votes_for_A_count : votes_for_A = 404600)
  (candidate_A_percentage_count : candidate_A_percentage = 0.85) :
  ∃ (x : ℝ), (x / 100) * total_votes = total_votes - votes_for_A / candidate_A_percentage ∧ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_invalid_votes_l1117_111766


namespace NUMINAMATH_GPT_ratio_of_heights_eq_three_twentieths_l1117_111708

noncomputable def base_circumference : ℝ := 32 * Real.pi
noncomputable def original_height : ℝ := 60
noncomputable def shorter_volume : ℝ := 768 * Real.pi

theorem ratio_of_heights_eq_three_twentieths
  (base_circumference : ℝ)
  (original_height : ℝ)
  (shorter_volume : ℝ)
  (h' : ℝ)
  (ratio : ℝ) :
  base_circumference = 32 * Real.pi →
  original_height = 60 →
  shorter_volume = 768 * Real.pi →
  (1 / 3 * Real.pi * (base_circumference / (2 * Real.pi))^2 * h') = shorter_volume →
  ratio = h' / original_height →
  ratio = 3 / 20 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_ratio_of_heights_eq_three_twentieths_l1117_111708


namespace NUMINAMATH_GPT_solve_xyz_l1117_111731

variable {x y z : ℝ}

theorem solve_xyz (h1 : (x + y + z) * (xy + xz + yz) = 35) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : x * y * z = 8 := 
by
  sorry

end NUMINAMATH_GPT_solve_xyz_l1117_111731


namespace NUMINAMATH_GPT_geese_percentage_l1117_111737

noncomputable def percentage_of_geese_among_non_swans (geese swans herons ducks : ℝ) : ℝ :=
  (geese / (100 - swans)) * 100

theorem geese_percentage (geese swans herons ducks : ℝ)
  (h1 : geese = 40)
  (h2 : swans = 20)
  (h3 : herons = 15)
  (h4 : ducks = 25) :
  percentage_of_geese_among_non_swans geese swans herons ducks = 50 :=
by
  simp [percentage_of_geese_among_non_swans, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_geese_percentage_l1117_111737


namespace NUMINAMATH_GPT_function_inverse_l1117_111790

theorem function_inverse (x : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 - 7 * x) 
  (k_def : ∀ x, k x = (6 - x) / 7) : 
  h (k x) = x ∧ k (h x) = x := 
  sorry

end NUMINAMATH_GPT_function_inverse_l1117_111790


namespace NUMINAMATH_GPT_polygon_sides_l1117_111742

theorem polygon_sides (h : ∀ (θ : ℕ), θ = 108) : ∃ n : ℕ, n = 5 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1117_111742


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1117_111741

theorem ellipse_standard_equation :
  ∀ (a b c : ℝ), a = 9 → c = 6 → b = Real.sqrt (a^2 - c^2) →
  (b ≠ 0 ∧ a ≠ 0 → (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1117_111741


namespace NUMINAMATH_GPT_number_of_ways_to_label_decagon_equal_sums_l1117_111763

open Nat

-- Formal definition of the problem
def sum_of_digits : Nat := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- The problem statement: Prove there are 3840 ways to label digits ensuring the given condition
theorem number_of_ways_to_label_decagon_equal_sums :
  ∃ (n : Nat), n = 3840 ∧ ∀ (A B C D E F G H I J K L : Nat), 
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧ (A ≠ H) ∧ (A ≠ I) ∧ (A ≠ J) ∧ (A ≠ K) ∧ (A ≠ L) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧ (B ≠ H) ∧ (B ≠ I) ∧ (B ≠ J) ∧ (B ≠ K) ∧ (B ≠ L) ∧
    (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧ (C ≠ H) ∧ (C ≠ I) ∧ (C ≠ J) ∧ (C ≠ K) ∧ (C ≠ L) ∧
    (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧ (D ≠ H) ∧ (D ≠ I) ∧ (D ≠ J) ∧ (D ≠ K) ∧ (D ≠ L) ∧
    (E ≠ F) ∧ (E ≠ G) ∧ (E ≠ H) ∧ (E ≠ I) ∧ (E ≠ J) ∧ (E ≠ K) ∧ (E ≠ L) ∧
    (F ≠ G) ∧ (F ≠ H) ∧ (F ≠ I) ∧ (F ≠ J) ∧ (F ≠ K) ∧ (F ≠ L) ∧
    (G ≠ H) ∧ (G ≠ I) ∧ (G ≠ J) ∧ (G ≠ K) ∧ (G ≠ L) ∧
    (H ≠ I) ∧ (H ≠ J) ∧ (H ≠ K) ∧ (H ≠ L) ∧
    (I ≠ J) ∧ (I ≠ K) ∧ (I ≠ L) ∧
    (J ≠ K) ∧ (J ≠ L) ∧
    (K ≠ L) ∧
    (A + L + F = B + L + G) ∧ (B + L + G = C + L + H) ∧ 
    (C + L + H = D + L + I) ∧ (D + L + I = E + L + J) ∧ 
    (E + L + J = F + L + K) ∧ (F + L + K = A + L + F) :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_label_decagon_equal_sums_l1117_111763


namespace NUMINAMATH_GPT_probability_two_green_in_four_l1117_111785

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def bag_marbles := 12
def green_marbles := 5
def blue_marbles := 3
def yellow_marbles := 4
def total_picked := 4
def green_picked := 2
def remaining_marbles := bag_marbles - green_marbles
def non_green_picked := total_picked - green_picked

theorem probability_two_green_in_four : 
  (choose green_marbles green_picked * choose remaining_marbles non_green_picked : ℚ) / (choose bag_marbles total_picked) = 14 / 33 := by
  sorry

end NUMINAMATH_GPT_probability_two_green_in_four_l1117_111785


namespace NUMINAMATH_GPT_arrangement_exists_l1117_111704

-- Definitions of pairwise coprimeness and gcd
def pairwise_coprime (a b c d : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1

def common_divisor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

-- Main theorem statement
theorem arrangement_exists :
  ∃ a b c d ab cd ad bc abcd : ℕ,
    pairwise_coprime a b c d ∧
    ab = a * b ∧ cd = c * d ∧ ad = a * d ∧ bc = b * c ∧ abcd = a * b * c * d ∧
    (common_divisor ab abcd ∧ common_divisor cd abcd ∧ common_divisor ad abcd ∧ common_divisor bc abcd) ∧
    (common_divisor ab ad ∧ common_divisor ab bc ∧ common_divisor cd ad ∧ common_divisor cd bc) ∧
    (relatively_prime ab cd ∧ relatively_prime ad bc) :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_arrangement_exists_l1117_111704


namespace NUMINAMATH_GPT_lindy_total_distance_l1117_111787

theorem lindy_total_distance (distance_jc : ℝ) (speed_j : ℝ) (speed_c : ℝ) (speed_l : ℝ)
  (h1 : distance_jc = 270) (h2 : speed_j = 4) (h3 : speed_c = 5) (h4 : speed_l = 8) : 
  ∃ time : ℝ, time = distance_jc / (speed_j + speed_c) ∧ speed_l * time = 240 :=
by
  sorry

end NUMINAMATH_GPT_lindy_total_distance_l1117_111787


namespace NUMINAMATH_GPT_unattainable_value_l1117_111723

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) : 
  ¬ ∃ y : ℝ, y = (1 - x) / (3 * x + 4) ∧ y = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_unattainable_value_l1117_111723


namespace NUMINAMATH_GPT_set_intersection_complement_l1117_111776

def U := {x : ℝ | x > -3}
def A := {x : ℝ | x < -2 ∨ x > 3}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

theorem set_intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x < -2 ∨ x > 4} :=
by sorry

end NUMINAMATH_GPT_set_intersection_complement_l1117_111776


namespace NUMINAMATH_GPT_find_pairs_l1117_111782

theorem find_pairs (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : (mn - 1) ∣ (n^3 - 1)) :
  ∃ k : ℕ, 1 < k ∧ ((m = k ∧ n = k^2) ∨ (m = k^2 ∧ n = k)) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1117_111782


namespace NUMINAMATH_GPT_Erik_money_left_l1117_111732

theorem Erik_money_left 
  (init_money : ℝ)
  (loaf_of_bread : ℝ) (n_loaves_of_bread : ℝ)
  (carton_of_orange_juice : ℝ) (n_cartons_of_orange_juice : ℝ)
  (dozen_eggs : ℝ) (n_dozens_of_eggs : ℝ)
  (chocolate_bar : ℝ) (n_chocolate_bars : ℝ)
  (pound_apples : ℝ) (n_pounds_apples : ℝ)
  (pound_grapes : ℝ) (n_pounds_grapes : ℝ)
  (discount_bread_and_eggs : ℝ) (discount_other_items : ℝ)
  (sales_tax : ℝ) :
  n_loaves_of_bread = 3 →
  loaf_of_bread = 3 →
  n_cartons_of_orange_juice = 3 →
  carton_of_orange_juice = 6 →
  n_dozens_of_eggs = 2 →
  dozen_eggs = 4 →
  n_chocolate_bars = 5 →
  chocolate_bar = 2 →
  n_pounds_apples = 4 →
  pound_apples = 1.25 →
  n_pounds_grapes = 1.5 →
  pound_grapes = 2.5 →
  discount_bread_and_eggs = 0.1 →
  discount_other_items = 0.05 →
  sales_tax = 0.06 →
  init_money = 86 →
  (init_money - 
     (n_loaves_of_bread * loaf_of_bread * (1 - discount_bread_and_eggs) + 
      n_cartons_of_orange_juice * carton_of_orange_juice * (1 - discount_other_items) + 
      n_dozens_of_eggs * dozen_eggs * (1 - discount_bread_and_eggs) + 
      n_chocolate_bars * chocolate_bar * (1 - discount_other_items) + 
      n_pounds_apples * pound_apples * (1 - discount_other_items) + 
      n_pounds_grapes * pound_grapes * (1 - discount_other_items)) * (1 + sales_tax)) = 32.78 :=
by
  sorry

end NUMINAMATH_GPT_Erik_money_left_l1117_111732


namespace NUMINAMATH_GPT_age_of_b_l1117_111752

-- Define the conditions as per the problem statement
variables (A B C D E : ℚ)

axiom cond1 : A = B + 2
axiom cond2 : B = 2 * C
axiom cond3 : D = A - 3
axiom cond4 : E = D / 2 + 3
axiom cond5 : A + B + C + D + E = 70

theorem age_of_b : B = 16.625 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_age_of_b_l1117_111752


namespace NUMINAMATH_GPT_aunt_money_calculation_l1117_111721

variable (total_money_received aunt_money : ℕ)
variable (bank_amount grandfather_money : ℕ := 150)

theorem aunt_money_calculation (h1 : bank_amount = 45) (h2 : bank_amount = total_money_received / 5) (h3 : total_money_received = aunt_money + grandfather_money) :
  aunt_money = 75 :=
by
  -- The proof is captured in these statements:
  sorry

end NUMINAMATH_GPT_aunt_money_calculation_l1117_111721


namespace NUMINAMATH_GPT_max_groups_l1117_111716

theorem max_groups (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) : Nat.gcd boys girls = 20 := 
  by
  rw [h1, h2]
  -- Proof steps would be here
  sorry

end NUMINAMATH_GPT_max_groups_l1117_111716


namespace NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l1117_111734

theorem problem_A : 2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5 := by
  sorry

theorem problem_B : 3 * Real.sqrt 3 * (3 * Real.sqrt 2) ≠ 3 * Real.sqrt 6 := by
  sorry

theorem problem_C : (Real.sqrt 27 / Real.sqrt 3) = 3 := by
  sorry

theorem problem_D : 2 * Real.sqrt 2 - Real.sqrt 2 ≠ 2 := by
  sorry

end NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l1117_111734


namespace NUMINAMATH_GPT_solve_equation_l1117_111795

theorem solve_equation : ∀ (x : ℝ), (x / 2 - 1 = 3) → x = 8 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_equation_l1117_111795


namespace NUMINAMATH_GPT_find_point_on_y_axis_l1117_111733

/-- 
Given points A (1, 2, 3) and B (2, -1, 4), and a point P on the y-axis 
such that the distances |PA| and |PB| are equal, 
prove that the coordinates of point P are (0, -7/6, 0).
 -/
theorem find_point_on_y_axis
  (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, 3))
  (hB : B = (2, -1, 4))
  (P : ℝ × ℝ × ℝ)
  (hP : ∃ y : ℝ, P = (0, y, 0)) :
  dist A P = dist B P → P = (0, -7/6, 0) :=
by
  sorry

end NUMINAMATH_GPT_find_point_on_y_axis_l1117_111733


namespace NUMINAMATH_GPT_plane_equation_intercept_l1117_111794

theorem plane_equation_intercept (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x y z : ℝ, ∃ k : ℝ, k = 1 → (x / a + y / b + z / c) = k :=
by sorry

end NUMINAMATH_GPT_plane_equation_intercept_l1117_111794


namespace NUMINAMATH_GPT_jeremie_friends_l1117_111784

-- Define the costs as constants.
def ticket_cost : ℕ := 18
def snack_cost : ℕ := 5
def total_cost : ℕ := 92
def per_person_cost : ℕ := ticket_cost + snack_cost

-- Define the number of friends Jeremie is going with (to be solved/proven).
def number_of_friends (total_cost : ℕ) (per_person_cost : ℕ) : ℕ :=
  let total_people := total_cost / per_person_cost
  total_people - 1

-- The statement that we want to prove.
theorem jeremie_friends : number_of_friends total_cost per_person_cost = 3 := by
  sorry

end NUMINAMATH_GPT_jeremie_friends_l1117_111784
