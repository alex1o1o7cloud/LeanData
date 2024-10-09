import Mathlib

namespace absolute_value_equation_solution_l921_92144

theorem absolute_value_equation_solution (x : ℝ) : |x - 30| + |x - 24| = |3 * x - 72| ↔ x = 26 :=
by sorry

end absolute_value_equation_solution_l921_92144


namespace find_a_l921_92163

theorem find_a (a : ℝ) (h : (1 - 2016 * a) = 2017) : a = -1 := by
  -- proof omitted
  sorry

end find_a_l921_92163


namespace vacation_costs_l921_92197

variable (Anne_paid Beth_paid Carlos_paid : ℕ) (a b : ℕ)

theorem vacation_costs (hAnne : Anne_paid = 120) (hBeth : Beth_paid = 180) (hCarlos : Carlos_paid = 150)
  (h_a : a = 30) (h_b : b = 30) :
  a - b = 0 := sorry

end vacation_costs_l921_92197


namespace combined_hits_and_misses_total_l921_92176

/-
  Prove that given the conditions for each day regarding the number of misses and
  the ratio of misses to hits, the combined total of hits and misses for the 
  three days is 322.
-/

theorem combined_hits_and_misses_total :
  (∀ (H1 : ℕ) (H2 : ℕ) (H3 : ℕ), 
    (2 * H1 = 60) ∧ (3 * H2 = 84) ∧ (5 * H3 = 100) →
    60 + 84 + 100 + H1 + H2 + H3 = 322) :=
by
  sorry

end combined_hits_and_misses_total_l921_92176


namespace even_perfect_squares_between_50_and_200_l921_92143

theorem even_perfect_squares_between_50_and_200 : ∃ s : Finset ℕ, 
  (∀ n ∈ s, (n^2 ≥ 50) ∧ (n^2 ≤ 200) ∧ n^2 % 2 = 0) ∧ s.card = 4 := by
  sorry

end even_perfect_squares_between_50_and_200_l921_92143


namespace solve_for_x_l921_92125

theorem solve_for_x :
  ∀ x : ℤ, (35 - (23 - (15 - x)) = (12 * 2) / 1 / 2) → x = -21 :=
by
  intro x
  sorry

end solve_for_x_l921_92125


namespace arithmetic_sequence_sum_l921_92157

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a n = a 0 + n * d)
  (h1 : a 0 + a 3 + a 6 = 45)
  (h2 : a 1 + a 4 + a 7 = 39) :
  a 2 + a 5 + a 8 = 33 := 
by
  sorry

end arithmetic_sequence_sum_l921_92157


namespace right_triangle_m_c_l921_92182

theorem right_triangle_m_c (a b c : ℝ) (m_c : ℝ) 
  (h : (1 / a) + (1 / b) = 3 / c) : 
  m_c = (c * (1 + Real.sqrt 10)) / 9 :=
sorry

end right_triangle_m_c_l921_92182


namespace cows_in_group_l921_92191

variable (c h : ℕ)

/--
In a group of cows and chickens, the number of legs was 20 more than twice the number of heads.
Cows have 4 legs each and chickens have 2 legs each.
Each animal has one head.
-/
theorem cows_in_group (h : ℕ) (hc : 4 * c + 2 * h = 2 * (c + h) + 20) : c = 10 :=
by
  sorry

end cows_in_group_l921_92191


namespace variation_relationship_l921_92103

theorem variation_relationship (k j : ℝ) (y z x : ℝ) (h1 : x = k * y^3) (h2 : y = j * z^(1/5)) :
  ∃ m : ℝ, x = m * z^(3/5) :=
by
  sorry

end variation_relationship_l921_92103


namespace possible_values_of_K_l921_92194

theorem possible_values_of_K (K M : ℕ) (h : K * (K + 1) = M^2) (hM : M < 100) : K = 8 ∨ K = 35 :=
by sorry

end possible_values_of_K_l921_92194


namespace expenditure_should_increase_by_21_percent_l921_92175

noncomputable def old_income := 100.0
noncomputable def ratio_exp_sav := (3 : ℝ) / (2 : ℝ)
noncomputable def income_increase_percent := 15.0 / 100.0
noncomputable def savings_increase_percent := 6.0 / 100.0
noncomputable def old_expenditure := old_income * (3 / (3 + 2))
noncomputable def old_savings := old_income * (2 / (3 + 2))
noncomputable def new_income := old_income * (1 + income_increase_percent)
noncomputable def new_savings := old_savings * (1 + savings_increase_percent)
noncomputable def new_expenditure := new_income - new_savings
noncomputable def expenditure_increase_percent := ((new_expenditure - old_expenditure) / old_expenditure) * 100

theorem expenditure_should_increase_by_21_percent :
  expenditure_increase_percent = 21 :=
sorry

end expenditure_should_increase_by_21_percent_l921_92175


namespace quadratic_complete_square_l921_92164

/-- Given quadratic expression, complete the square to find the equivalent form
    and calculate the sum of the coefficients a, h, k. -/
theorem quadratic_complete_square (a h k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 :=
by
  intro h₁
  sorry

end quadratic_complete_square_l921_92164


namespace total_words_in_poem_l921_92185

theorem total_words_in_poem (s l w : ℕ) (h1 : s = 35) (h2 : l = 15) (h3 : w = 12) : 
  s * l * w = 6300 := 
by 
  -- the proof will be inserted here
  sorry

end total_words_in_poem_l921_92185


namespace initial_profit_percentage_l921_92126

theorem initial_profit_percentage
  (CP : ℝ)
  (h1 : CP = 2400)
  (h2 : ∀ SP : ℝ, 15 / 100 * CP = 120 + SP) :
  ∃ P : ℝ, (P / 100) * CP = 10 :=
by
  sorry

end initial_profit_percentage_l921_92126


namespace triangle_side_lengths_l921_92178

theorem triangle_side_lengths (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) :
  (a = 15 ∧ b = 20 ∧ c = 25) :=
sorry

end triangle_side_lengths_l921_92178


namespace fred_paid_amount_l921_92137

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def borrowed_movie_price : ℝ := 6.79
def change_received : ℝ := 1.37

def total_cost : ℝ := (number_of_tickets : ℝ) * ticket_price + borrowed_movie_price
def amount_paid : ℝ := total_cost + change_received

theorem fred_paid_amount : amount_paid = 20.00 := sorry

end fred_paid_amount_l921_92137


namespace complement_A_in_B_l921_92151

-- Define the sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

-- Define the complement of A in B
def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Statement to prove
theorem complement_A_in_B :
  complement B A = {0, 1, 4} := by
  sorry

end complement_A_in_B_l921_92151


namespace max_value_part1_l921_92113

theorem max_value_part1 (a : ℝ) (h : a < 3 / 2) : 2 * a + 4 / (2 * a - 3) + 3 ≤ 2 :=
sorry

end max_value_part1_l921_92113


namespace correct_triangle_l921_92199

-- Define the conditions for the sides of each option
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (3, 1, 1)
def sides_D := (3, 4, 7)

-- Conditions for forming a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Prove the problem statement
theorem correct_triangle : is_triangle 3 4 5 :=
by
  sorry

end correct_triangle_l921_92199


namespace students_left_is_31_l921_92121

-- Define the conditions based on the problem statement
def total_students : ℕ := 124
def checked_out_early : ℕ := 93

-- Define the theorem that states the problem we want to prove
theorem students_left_is_31 :
  total_students - checked_out_early = 31 :=
by
  -- Proof would go here
  sorry

end students_left_is_31_l921_92121


namespace salary_of_thomas_l921_92161

variable (R Ro T : ℕ)

theorem salary_of_thomas 
  (h1 : R + Ro = 8000) 
  (h2 : R + Ro + T = 15000) : 
  T = 7000 := by
  sorry

end salary_of_thomas_l921_92161


namespace line_always_passes_fixed_point_l921_92190

theorem line_always_passes_fixed_point (m : ℝ) :
  m * 1 + (1 - m) * 2 + m - 2 = 0 :=
by
  sorry

end line_always_passes_fixed_point_l921_92190


namespace bill_experience_now_l921_92148

theorem bill_experience_now (B J : ℕ) 
  (h1 : J = 3 * B) 
  (h2 : J + 5 = 2 * (B + 5)) : B + 5 = 10 :=
by
  sorry

end bill_experience_now_l921_92148


namespace problem_projection_eq_l921_92100

variable (m n : ℝ × ℝ)
variable (m_val : m = (1, 2))
variable (n_val : n = (2, 3))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

theorem problem_projection_eq : projection m n = (8 * Real.sqrt 13) / 13 :=
by
  rw [m_val, n_val]
  sorry

end problem_projection_eq_l921_92100


namespace find_a_and_b_minimum_value_of_polynomial_l921_92198

noncomputable def polynomial_has_maximum (x y a b : ℝ) : Prop :=
  y = a * x ^ 3 + b * x ^ 2 ∧ x = 1 ∧ y = 3

noncomputable def polynomial_minimum_value (y : ℝ) : Prop :=
  y = 0

theorem find_a_and_b (a b x y : ℝ) (h : polynomial_has_maximum x y a b) :
  a = -6 ∧ b = 9 :=
by sorry

theorem minimum_value_of_polynomial (a b y : ℝ) (h : a = -6 ∧ b = 9) :
  polynomial_minimum_value y :=
by sorry

end find_a_and_b_minimum_value_of_polynomial_l921_92198


namespace identify_urea_decomposing_bacteria_l921_92160

-- Definitions of different methods
def methodA (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (phenol_red : culture_medium), phenol_red = urea_only

def methodB (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (EMB_reagent : culture_medium), EMB_reagent = urea_only

def methodC (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Sudan_III : culture_medium), Sudan_III = urea_only

def methodD (culture_medium : Type) := 
  ∃ (urea_only : culture_medium) (Biuret_reagent : culture_medium), Biuret_reagent = urea_only

-- The proof problem statement
theorem identify_urea_decomposing_bacteria (culture_medium : Type) :
  methodA culture_medium :=
sorry

end identify_urea_decomposing_bacteria_l921_92160


namespace num_factors_2012_l921_92145

theorem num_factors_2012 : (Nat.factors 2012).length = 6 := by
  sorry

end num_factors_2012_l921_92145


namespace min_expression_value_2023_l921_92146

noncomputable def min_expr_val := ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023

noncomputable def least_value : ℝ := 2023

theorem min_expression_value_2023 : min_expr_val ∧ (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = least_value) := 
by sorry

end min_expression_value_2023_l921_92146


namespace computation_l921_92167

theorem computation :
  (13 + 12)^2 - (13 - 12)^2 = 624 :=
by
  sorry

end computation_l921_92167


namespace average_weight_of_remaining_boys_l921_92123

theorem average_weight_of_remaining_boys (avg_weight_16: ℝ) (avg_weight_total: ℝ) (weight_16: ℝ) (total_boys: ℝ) (avg_weight_8: ℝ) : 
  (avg_weight_16 = 50.25) → (avg_weight_total = 48.55) → (weight_16 = 16 * avg_weight_16) → (total_boys = 24) → 
  (total_weight = total_boys * avg_weight_total) → (weight_16 + 8 * avg_weight_8 = total_weight) → avg_weight_8 = 45.15 :=
by
  intros h_avg_weight_16 h_avg_weight_total h_weight_16 h_total_boys h_total_weight h_equation
  sorry

end average_weight_of_remaining_boys_l921_92123


namespace tony_rope_length_l921_92177

-- Definitions based on the conditions in the problem
def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss_per_knot : ℝ := 1.2
def number_of_knots : ℕ := 5

-- The final length of the rope after tying all pieces together and losing length per knot
def final_rope_length (lengths : List ℝ) (loss_per_knot : ℝ) (number_of_knots : ℕ) : ℝ :=
  List.sum lengths - (loss_per_knot * number_of_knots)

theorem tony_rope_length :
  final_rope_length rope_lengths knot_loss_per_knot number_of_knots = 35 := by
  sorry

end tony_rope_length_l921_92177


namespace stones_required_to_pave_hall_l921_92153

theorem stones_required_to_pave_hall :
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    (area_hall_dm2 / area_stone_dm2) = 3600 :=
by
    -- Definitions
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5

    -- Convert to decimeters
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    
    -- Calculate areas
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    
    -- Calculate number of stones 
    let number_of_stones := area_hall_dm2 / area_stone_dm2

    -- Prove the required number of stones
    have h : number_of_stones = 3600 := sorry
    exact h

end stones_required_to_pave_hall_l921_92153


namespace number_of_a_l921_92105

theorem number_of_a (h : ∃ a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2) : 
  ∃! a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2 :=
sorry

end number_of_a_l921_92105


namespace kurt_savings_l921_92169

def daily_cost_old : ℝ := 0.85
def daily_cost_new : ℝ := 0.45
def days : ℕ := 30

theorem kurt_savings : (daily_cost_old * days) - (daily_cost_new * days) = 12.00 := by
  sorry

end kurt_savings_l921_92169


namespace cooper_saved_days_l921_92141

variable (daily_saving : ℕ) (total_saving : ℕ) (n : ℕ)

-- Conditions
def cooper_saved (daily_saving total_saving n : ℕ) : Prop :=
  total_saving = daily_saving * n

-- Theorem stating the question equals the correct answer
theorem cooper_saved_days :
  cooper_saved 34 12410 365 :=
by
  sorry

end cooper_saved_days_l921_92141


namespace geometric_sequence_k_value_l921_92181

theorem geometric_sequence_k_value :
  ∀ {S : ℕ → ℤ} (a : ℕ → ℤ) (k : ℤ),
    (∀ n, S n = 3 * 2^n + k) → 
    (∀ n ≥ 2, a n = S n - S (n - 1)) → 
    (∀ n ≥ 2, a n ^ 2 = a 1 * a 3) → 
    k = -3 :=
by
  sorry

end geometric_sequence_k_value_l921_92181


namespace comprehensive_score_l921_92156

variable (regularAssessmentScore : ℕ)
variable (finalExamScore : ℕ)
variable (regularAssessmentWeighting : ℝ)
variable (finalExamWeighting : ℝ)

theorem comprehensive_score 
  (h1 : regularAssessmentScore = 95)
  (h2 : finalExamScore = 90)
  (h3 : regularAssessmentWeighting = 0.20)
  (h4 : finalExamWeighting = 0.80) :
  (regularAssessmentScore * regularAssessmentWeighting + finalExamScore * finalExamWeighting) = 91 :=
sorry

end comprehensive_score_l921_92156


namespace FGH_supermarkets_US_l921_92101

/-- There are 60 supermarkets in the FGH chain,
all of them are either in the US or Canada,
there are 14 more FGH supermarkets in the US than in Canada.
Prove that there are 37 FGH supermarkets in the US. -/
theorem FGH_supermarkets_US (C U : ℕ) (h1 : C + U = 60) (h2 : U = C + 14) : U = 37 := by
  sorry

end FGH_supermarkets_US_l921_92101


namespace smallest_integer_with_divisors_l921_92142

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l921_92142


namespace probability_of_a_plus_b_gt_5_l921_92120

noncomputable def all_events : Finset (ℕ × ℕ) := 
  { (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4) }

noncomputable def successful_events : Finset (ℕ × ℕ) :=
  { (2, 4), (3, 3), (3, 4) }

theorem probability_of_a_plus_b_gt_5 : 
  (successful_events.card : ℚ) / (all_events.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_a_plus_b_gt_5_l921_92120


namespace average_cups_of_tea_sold_l921_92196

theorem average_cups_of_tea_sold (x_avg : ℝ) (y_regression : ℝ → ℝ) 
  (h1 : x_avg = 12) (h2 : ∀ x, y_regression x = -2*x + 58) : 
  y_regression x_avg = 34 := by
  sorry

end average_cups_of_tea_sold_l921_92196


namespace each_child_receive_amount_l921_92122

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end each_child_receive_amount_l921_92122


namespace repeating_decimal_to_fraction_l921_92116

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end repeating_decimal_to_fraction_l921_92116


namespace abe_age_sum_l921_92110

theorem abe_age_sum (x : ℕ) : 25 + (25 - x) = 29 ↔ x = 21 :=
by sorry

end abe_age_sum_l921_92110


namespace eval_expression_l921_92192

theorem eval_expression : 3 - (-1) + 4 - 5 + (-6) - (-7) + 8 - 9 = 3 := 
  sorry

end eval_expression_l921_92192


namespace find_m_l921_92179

theorem find_m (m : ℝ) :
  (m - 2013 = 0) → (m = 2013) ∧ (m - 1 ≠ 0) :=
by {
  sorry
}

end find_m_l921_92179


namespace gcd_7654321_6789012_l921_92173

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end gcd_7654321_6789012_l921_92173


namespace current_speed_l921_92172

-- The main statement of our problem
theorem current_speed (v_with_current v_against_current c man_speed : ℝ) 
  (h1 : v_with_current = man_speed + c) 
  (h2 : v_against_current = man_speed - c) 
  (h_with : v_with_current = 15) 
  (h_against : v_against_current = 9.4) : 
  c = 2.8 :=
by
  sorry

end current_speed_l921_92172


namespace halfway_between_fractions_l921_92129

-- Definitions used in the conditions
def one_eighth := (1 : ℚ) / 8
def three_tenths := (3 : ℚ) / 10

-- The mathematical assertion to prove
theorem halfway_between_fractions : (one_eighth + three_tenths) / 2 = 17 / 80 := by
  sorry

end halfway_between_fractions_l921_92129


namespace valid_two_digit_numbers_l921_92155

def is_valid_two_digit_number_pair (a b : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a > b ∧ (Nat.gcd (10 * a + b) (10 * b + a) = a^2 - b^2)

theorem valid_two_digit_numbers :
  (is_valid_two_digit_number_pair 2 1 ∨ is_valid_two_digit_number_pair 5 4) ∧
  ∀ a b, is_valid_two_digit_number_pair a b → (a = 2 ∧ b = 1 ∨ a = 5 ∧ b = 4) :=
by
  sorry

end valid_two_digit_numbers_l921_92155


namespace parabola_equation_with_left_focus_l921_92106

theorem parabola_equation_with_left_focus (x y : ℝ) :
  (∀ x y : ℝ, (x^2)/25 + (y^2)/9 = 1 → (y^2 = -16 * x)) :=
by
  sorry

end parabola_equation_with_left_focus_l921_92106


namespace sum_of_fractions_is_514_l921_92162

theorem sum_of_fractions_is_514 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 5 / 14 := 
by
  sorry

end sum_of_fractions_is_514_l921_92162


namespace fraction_value_l921_92140

theorem fraction_value (a b : ℚ) (h₁ : b / (a - 2) = 3 / 4) (h₂ : b / (a + 9) = 5 / 7) : b / a = 165 / 222 := 
by sorry

end fraction_value_l921_92140


namespace cannot_determine_exact_insect_l921_92152

-- Defining the conditions as premises
def insect_legs : ℕ := 6

def total_legs_two_insects (legs_per_insect : ℕ) (num_insects : ℕ) : ℕ :=
  legs_per_insect * num_insects

-- Statement: Proving that given just the number of legs, we cannot determine the exact type of insect
theorem cannot_determine_exact_insect (legs : ℕ) (num_insects : ℕ) (h1 : legs = 6) (h2 : num_insects = 2) (h3 : total_legs_two_insects legs num_insects = 12) :
  ∃ insect_type, insect_type :=
by
  sorry

end cannot_determine_exact_insect_l921_92152


namespace min_value_xy_inv_xy_l921_92159

theorem min_value_xy_inv_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy_sum : x + y = 2) :
  ∃ m : ℝ, m = xy + 4 / xy ∧ m ≥ 5 :=
by
  sorry

end min_value_xy_inv_xy_l921_92159


namespace smallest_number_of_white_marbles_l921_92133

theorem smallest_number_of_white_marbles
  (n : ℕ)
  (hn1 : n > 0)
  (orange_marbles : ℕ := n / 5)
  (hn_orange : n % 5 = 0)
  (purple_marbles : ℕ := n / 6)
  (hn_purple : n % 6 = 0)
  (green_marbles : ℕ := 9)
  : (n - (orange_marbles + purple_marbles + green_marbles)) = 10 → n = 30 :=
by
  sorry

end smallest_number_of_white_marbles_l921_92133


namespace cookies_difference_l921_92112

theorem cookies_difference :
  let bags := 9
  let boxes := 8
  let cookies_per_bag := 7
  let cookies_per_box := 12
  8 * 12 - 9 * 7 = 33 := 
by
  sorry

end cookies_difference_l921_92112


namespace deepak_present_age_l921_92131

theorem deepak_present_age (R D : ℕ) (h1 : R =  4 * D / 3) (h2 : R + 10 = 26) : D = 12 :=
by
  sorry

end deepak_present_age_l921_92131


namespace num_two_digit_palindromes_l921_92170

theorem num_two_digit_palindromes : 
  let is_palindrome (n : ℕ) : Prop := (n / 10) = (n % 10)
  ∃ n : ℕ, 10 ≤ n ∧ n < 90 ∧ is_palindrome n →
  ∃ count : ℕ, count = 9 := 
sorry

end num_two_digit_palindromes_l921_92170


namespace particle_paths_l921_92135

open Nat

-- Define the conditions of the problem
def move_right (a b : ℕ) : ℕ × ℕ := (a + 1, b)
def move_up (a b : ℕ) : ℕ × ℕ := (a, b + 1)
def move_diagonal (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

-- Define a function to count paths without right-angle turns
noncomputable def count_paths (n : ℕ) : ℕ :=
  if n = 6 then 247 else 0

-- The theorem to be proven
theorem particle_paths :
  count_paths 6 = 247 :=
  sorry

end particle_paths_l921_92135


namespace automobile_travel_distance_l921_92139

theorem automobile_travel_distance (a r : ℝ) :
  (2 * a / 5) / (2 * r) * 5 * 60 / 3 = 20 * a / r :=
by 
  -- skipping proof details
  sorry

end automobile_travel_distance_l921_92139


namespace correct_calculation_l921_92165

def original_number (x : ℕ) : Prop := x + 12 = 48

theorem correct_calculation (x : ℕ) (h : original_number x) : x + 22 = 58 := by
  sorry

end correct_calculation_l921_92165


namespace cab_to_bus_ratio_l921_92180

noncomputable def train_distance : ℤ := 300
noncomputable def bus_distance : ℤ := train_distance / 2
noncomputable def total_distance : ℤ := 500
noncomputable def cab_distance : ℤ := total_distance - (train_distance + bus_distance)
noncomputable def ratio : ℚ := cab_distance / bus_distance

theorem cab_to_bus_ratio :
  ratio = 1 / 3 := by
  sorry

end cab_to_bus_ratio_l921_92180


namespace money_left_after_deductions_l921_92117

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end money_left_after_deductions_l921_92117


namespace problem_part1_problem_part2_l921_92149

def ellipse_condition (m : ℝ) : Prop :=
  m + 1 > 4 - m ∧ 4 - m > 0

def circle_condition (m : ℝ) : Prop :=
  m^2 - 4 > 0

theorem problem_part1 (m : ℝ) :
  ellipse_condition m → (3 / 2 < m ∧ m < 4) :=
sorry

theorem problem_part2 (m : ℝ) :
  ellipse_condition m ∧ circle_condition m → (2 < m ∧ m < 4) :=
sorry

end problem_part1_problem_part2_l921_92149


namespace two_bedroom_units_l921_92189

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l921_92189


namespace largest_square_not_divisible_by_100_l921_92132

theorem largest_square_not_divisible_by_100
  (n : ℕ) (h1 : ∃ a : ℕ, a^2 = n) 
  (h2 : n % 100 ≠ 0)
  (h3 : ∃ m : ℕ, m * 100 + n % 100 = n ∧ ∃ b : ℕ, b^2 = m) :
  n = 1681 := sorry

end largest_square_not_divisible_by_100_l921_92132


namespace total_seats_in_theater_l921_92134

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

end total_seats_in_theater_l921_92134


namespace compute_fg_l921_92171

def g (x : ℕ) : ℕ := 2 * x + 6
def f (x : ℕ) : ℕ := 4 * x - 8
def x : ℕ := 10

theorem compute_fg : f (g x) = 96 := by
  sorry

end compute_fg_l921_92171


namespace integral_abs_x_minus_two_l921_92136

theorem integral_abs_x_minus_two : ∫ x in (0:ℝ)..4, |x - 2| = 4 := 
by
  sorry

end integral_abs_x_minus_two_l921_92136


namespace second_player_wins_optimal_play_l921_92124

def players_take_turns : Prop := sorry
def win_condition (box_count : ℕ) : Prop := box_count = 21

theorem second_player_wins_optimal_play (boxes : Fin 11 → ℕ)
    (h_turns : players_take_turns)
    (h_win : ∀ i : Fin 11, win_condition (boxes i)) : 
    ∃ P : ℕ, P = 2 :=
sorry

end second_player_wins_optimal_play_l921_92124


namespace find_r_l921_92128

theorem find_r (r : ℝ) (h₁ : 0 < r) (h₂ : ∀ x y : ℝ, (x - y = r → x^2 + y^2 = r → False)) : r = 2 :=
sorry

end find_r_l921_92128


namespace evaluate_at_3_l921_92186

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 + x + 6

theorem evaluate_at_3 : g 3 = 135 := 
  by
  sorry

end evaluate_at_3_l921_92186


namespace cos_four_alpha_sub_9pi_over_2_l921_92109

open Real

theorem cos_four_alpha_sub_9pi_over_2 (α : ℝ) 
  (cond : 4.53 * (1 + cos (2 * α - 2 * π) + cos (4 * α + 2 * π) - cos (6 * α - π)) /
                  (cos (2 * π - 2 * α) + 2 * cos (2 * α + π) ^ 2 - 1) = 2 * cos (2 * α)) :
  cos (4 * α - 9 * π / 2) = cos (4 * α - π / 2) :=
by sorry

end cos_four_alpha_sub_9pi_over_2_l921_92109


namespace calculate_interest_rate_l921_92187

theorem calculate_interest_rate
  (total_investment : ℝ)
  (invested_at_eleven_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_first_type : ℝ) :
  total_investment = 100000 ∧ 
  invested_at_eleven_percent = 30000 ∧ 
  total_interest = 9.6 → 
  interest_rate_first_type = 9 :=
by
  intros
  sorry

end calculate_interest_rate_l921_92187


namespace exists_infinite_bisecting_circles_l921_92127

-- Define circle and bisecting condition
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def bisects (B C : Circle) : Prop :=
  let chord_len := (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2
  (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2 = B.radius^2

-- Define the theorem statement
theorem exists_infinite_bisecting_circles (C1 C2 : Circle) (h : C1.center ≠ C2.center) :
  ∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧
  ∀ (b_center : ℝ × ℝ), (∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧ B.center = b_center) ↔
  2 * (C2.center.1 - C1.center.1) * b_center.1 + 2 * (C2.center.2 - C1.center.2) * b_center.2 =
  (C2.center.1^2 - C1.center.1^2) + (C2.center.2^2 - C1.center.2^2) + (C2.radius^2 - C1.radius^2) := 
sorry

end exists_infinite_bisecting_circles_l921_92127


namespace new_area_rhombus_l921_92154

theorem new_area_rhombus (d1 d2 : ℝ) (h : (d1 * d2) / 2 = 3) : 
  ((5 * d1) * (5 * d2)) / 2 = 75 := 
by
  sorry

end new_area_rhombus_l921_92154


namespace total_wholesale_cost_is_correct_l921_92111

-- Given values
def retail_price_pants : ℝ := 36
def markup_pants : ℝ := 0.8

def retail_price_shirt : ℝ := 45
def markup_shirt : ℝ := 0.6

def retail_price_jacket : ℝ := 120
def markup_jacket : ℝ := 0.5

noncomputable def wholesale_cost_pants : ℝ := retail_price_pants / (1 + markup_pants)
noncomputable def wholesale_cost_shirt : ℝ := retail_price_shirt / (1 + markup_shirt)
noncomputable def wholesale_cost_jacket : ℝ := retail_price_jacket / (1 + markup_jacket)

noncomputable def total_wholesale_cost : ℝ :=
  wholesale_cost_pants + wholesale_cost_shirt + wholesale_cost_jacket

theorem total_wholesale_cost_is_correct :
  total_wholesale_cost = 128.125 := by
  sorry

end total_wholesale_cost_is_correct_l921_92111


namespace miyoung_largest_square_side_l921_92188

theorem miyoung_largest_square_side :
  ∃ (G : ℕ), G > 0 ∧ ∀ (a b : ℕ), (a = 32) → (b = 74) → (gcd a b = G) → (G = 2) :=
by {
  sorry
}

end miyoung_largest_square_side_l921_92188


namespace exists_bounding_constant_M_l921_92193

variable (α : ℝ) (a : ℕ → ℝ)
variable (hα : α > 1)
variable (h_seq : ∀ n : ℕ, n > 0 →
  a n.succ = a n + (a n / n) ^ α)

theorem exists_bounding_constant_M (h_a1 : 0 < a 1 ∧ a 1 < 1) : 
  ∃ M, ∀ n > 0, a n ≤ M := 
sorry

end exists_bounding_constant_M_l921_92193


namespace cost_of_running_tv_for_week_l921_92168

def powerUsage : ℕ := 125
def hoursPerDay : ℕ := 4
def costPerkWh : ℕ := 14

theorem cost_of_running_tv_for_week :
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  weeklyCost = 49 := by
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  sorry

end cost_of_running_tv_for_week_l921_92168


namespace chandler_total_rolls_l921_92158

-- Definitions based on given conditions
def rolls_sold_grandmother : ℕ := 3
def rolls_sold_uncle : ℕ := 4
def rolls_sold_neighbor : ℕ := 3
def rolls_needed_more : ℕ := 2

-- Total rolls sold so far and needed
def total_rolls_to_sell : ℕ :=
  rolls_sold_grandmother + rolls_sold_uncle + rolls_sold_neighbor + rolls_needed_more

theorem chandler_total_rolls : total_rolls_to_sell = 12 :=
by
  sorry

end chandler_total_rolls_l921_92158


namespace paolo_coconuts_l921_92114

theorem paolo_coconuts
  (P : ℕ)
  (dante_coconuts : ℕ := 3 * P)
  (dante_sold : ℕ := 10)
  (dante_left : ℕ := 32)
  (h : dante_left + dante_sold = dante_coconuts) : P = 14 :=
by {
  sorry
}

end paolo_coconuts_l921_92114


namespace correct_operation_is_a_l921_92195

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l921_92195


namespace find_savings_l921_92147

-- Definitions of given conditions
def income : ℕ := 10000
def ratio_income_expenditure : ℕ × ℕ := (10, 8)

-- Proving the savings based on given conditions
theorem find_savings (income : ℕ) (ratio_income_expenditure : ℕ × ℕ) :
  let expenditure := (ratio_income_expenditure.2 * income) / ratio_income_expenditure.1
  let savings := income - expenditure
  savings = 2000 :=
by
  sorry

end find_savings_l921_92147


namespace range_of_a_l921_92118

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (x + 2) - abs (x - 1) ≥ a^3 - 4 * a^2 - 3) → a ≤ 4 :=
sorry

end range_of_a_l921_92118


namespace bottles_left_after_purchase_l921_92150

def initial_bottles : ℕ := 35
def jason_bottles : ℕ := 5
def harry_bottles : ℕ := 6
def jason_effective_bottles (n : ℕ) : ℕ := n  -- Jason buys 5 bottles
def harry_effective_bottles (n : ℕ) : ℕ := n + 1 -- Harry gets one additional free bottle

theorem bottles_left_after_purchase (j_b h_b i_b : ℕ) (j_effective h_effective : ℕ → ℕ) :
  j_b = 5 → h_b = 6 → i_b = 35 → j_effective j_b = 5 → h_effective h_b = 7 →
  i_b - (j_effective j_b + h_effective h_b) = 23 :=
by
  intros
  sorry

end bottles_left_after_purchase_l921_92150


namespace cube_inscribed_sphere_volume_l921_92174

noncomputable def cubeSurfaceArea (a : ℝ) : ℝ := 6 * a^2
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def inscribedSphereRadius (a : ℝ) : ℝ := a / 2

theorem cube_inscribed_sphere_volume :
  ∀ (a : ℝ), cubeSurfaceArea a = 24 → sphereVolume (inscribedSphereRadius a) = (4 / 3) * Real.pi := 
by 
  intros a h₁
  sorry

end cube_inscribed_sphere_volume_l921_92174


namespace entrance_exam_proof_l921_92108

-- Define the conditions
variables (x y : ℕ)
variables (h1 : x + y = 70)
variables (h2 : 3 * x - y = 38)

-- The proof goal
theorem entrance_exam_proof : x = 27 :=
by
  -- The actual proof steps are omitted here
  sorry

end entrance_exam_proof_l921_92108


namespace smallest_next_divisor_l921_92138

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor (m : ℕ) (h_even : is_even m)
  (h_four_digit : is_four_digit m)
  (h_div_437 : is_divisor 437 m) :
  ∃ next_div : ℕ, next_div > 437 ∧ is_divisor next_div m ∧ 
  ∀ d, d > 437 ∧ is_divisor d m → next_div ≤ d :=
sorry

end smallest_next_divisor_l921_92138


namespace rug_area_calculation_l921_92130

theorem rug_area_calculation (length_floor width_floor strip_width : ℕ)
  (h_length : length_floor = 10)
  (h_width : width_floor = 8)
  (h_strip : strip_width = 2) :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := by
  sorry

end rug_area_calculation_l921_92130


namespace alice_distance_from_start_l921_92104

theorem alice_distance_from_start :
  let hexagon_side := 3
  let distance_walked := 10
  let final_distance := 3 * Real.sqrt 3 / 2
  final_distance =
    let a := (0, 0)
    let b := (3, 0)
    let c := (4.5, 3 * Real.sqrt 3 / 2)
    let d := (1.5, 3 * Real.sqrt 3 / 2)
    let e := (0, 3 * Real.sqrt 3 / 2)
    dist a e := sorry

end alice_distance_from_start_l921_92104


namespace least_positive_divisible_by_primes_l921_92119

theorem least_positive_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7
  ∃ n : ℕ, n > 0 ∧ (n % p1 = 0) ∧ (n % p2 = 0) ∧ (n % p3 = 0) ∧ (n % p4 = 0) ∧ 
  (∀ m : ℕ, m > 0 → (m % p1 = 0) ∧ (m % p2 = 0) ∧ (m % p3 = 0) ∧ (m % p4 = 0) → m ≥ n) ∧ n = 210 := 
by {
  sorry
}

end least_positive_divisible_by_primes_l921_92119


namespace domain_of_function_l921_92166

theorem domain_of_function :
  {x : ℝ | 2 - x ≥ 0} = {x : ℝ | x ≤ 2} :=
by
  sorry

end domain_of_function_l921_92166


namespace trains_cross_each_other_in_5_76_seconds_l921_92102

noncomputable def trains_crossing_time (l1 l2 v1_kmh v2_kmh : ℕ) : ℚ :=
  let v1 := (v1_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let v2 := (v2_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let total_distance := (l1 : ℚ) + (l2 : ℚ)
  let relative_velocity := v1 + v2
  total_distance / relative_velocity

theorem trains_cross_each_other_in_5_76_seconds :
  trains_crossing_time 100 60 60 40 = 160 / 27.78 := by
  sorry

end trains_cross_each_other_in_5_76_seconds_l921_92102


namespace second_student_catches_up_l921_92107

open Nat

-- Definitions for the problems
def distance_first_student (n : ℕ) : ℕ := 7 * n
def distance_second_student (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement indicating the second student catches up with the first at n = 13
theorem second_student_catches_up : ∃ n, (distance_first_student n = distance_second_student n) ∧ n = 13 := 
by 
  sorry

end second_student_catches_up_l921_92107


namespace find_number_eq_l921_92115

theorem find_number_eq : ∃ x : ℚ, (35 / 100) * x = (25 / 100) * 40 ∧ x = 200 / 7 :=
by
  sorry

end find_number_eq_l921_92115


namespace polynomial_bound_l921_92184

theorem polynomial_bound (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l921_92184


namespace terminating_decimal_multiples_l921_92183

theorem terminating_decimal_multiples :
  (∃ n : ℕ, 20 = n ∧ ∀ m, 1 ≤ m ∧ m ≤ 180 → 
  (∃ k : ℕ, m = 9 * k)) :=
by
  sorry

end terminating_decimal_multiples_l921_92183
