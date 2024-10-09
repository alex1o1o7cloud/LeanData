import Mathlib

namespace sum_of_powers_of_i_l511_51185

-- Let i be the imaginary unit
def i : ℂ := Complex.I

theorem sum_of_powers_of_i : (1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 + i^10) = i := by
  sorry

end sum_of_powers_of_i_l511_51185


namespace alberto_vs_bjorn_distance_difference_l511_51174

noncomputable def alberto_distance (t : ℝ) : ℝ := (3.75 / 5) * t
noncomputable def bjorn_distance (t : ℝ) : ℝ := (3.4375 / 5) * t

theorem alberto_vs_bjorn_distance_difference :
  alberto_distance 5 - bjorn_distance 5 = 0.3125 :=
by
  -- proof goes here
  sorry

end alberto_vs_bjorn_distance_difference_l511_51174


namespace cauchy_schwarz_inequality_l511_51176

theorem cauchy_schwarz_inequality 
  (a b a1 b1 : ℝ) : ((a * a1 + b * b1) ^ 2 ≤ (a^2 + b^2) * (a1^2 + b1^2)) :=
 by sorry

end cauchy_schwarz_inequality_l511_51176


namespace problem_l511_51163

theorem problem (a b a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℕ)
  (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ : ℕ) 
  (h1 : a₁ < a₂) (h2 : a₂ < a₃) (h3 : a₃ < a₄) 
  (h4 : a₄ < a₅) (h5 : a₅ < a₆) (h6 : a₆ < a₇)
  (h7 : a₇ < a₈) (h8 : a₈ < a₉) (h9 : a₉ < a₁₀)
  (h10 : a₁₀ < a₁₁) (h11 : b₁ < b₂) (h12 : b₂ < b₃)
  (h13 : b₃ < b₄) (h14 : b₄ < b₅) (h15 : b₅ < b₆)
  (h16 : b₆ < b₇) (h17 : b₇ < b₈) (h18 : b₈ < b₉)
  (h19 : b₉ < b₁₀) (h20 : b₁₀ < b₁₁) 
  (h21 : a₁₀ + b₁₀ = a) (h22 : a₁₁ + b₁₁ = b) : 
  a = 1024 ∧ b = 2048 :=
sorry

end problem_l511_51163


namespace find_f_of_2_l511_51105

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ (x : ℝ), x > 0 → f (Real.log x / Real.log 2) = 2 ^ x) : f 2 = 16 :=
by
  sorry

end find_f_of_2_l511_51105


namespace jane_change_l511_51128

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l511_51128


namespace find_k_unique_solution_l511_51194

theorem find_k_unique_solution (k : ℝ) (h: k ≠ 0) : (∀ x : ℝ, (x + 3) / (k * x - 2) = x → k = -3/4) :=
sorry

end find_k_unique_solution_l511_51194


namespace least_3_digit_number_l511_51150

variables (k S h t u : ℕ)

def is_3_digit_number (k : ℕ) : Prop := k ≥ 100 ∧ k < 1000

def digits_sum_eq (k h t u S : ℕ) : Prop :=
  k = 100 * h + 10 * t + u ∧ S = h + t + u

def difference_condition (h t : ℕ) : Prop :=
  t - h = 8

theorem least_3_digit_number (k S h t u : ℕ) :
  is_3_digit_number k →
  digits_sum_eq k h t u S →
  difference_condition h t →
  k * 3 < 200 →
  k = 19 * S :=
sorry

end least_3_digit_number_l511_51150


namespace volume_is_correct_l511_51155

noncomputable def volume_of_target_cube (V₁ : ℝ) (A₂ : ℝ) : ℝ :=
  if h₁ : V₁ = 8 then
    let s₁ := (8 : ℝ)^(1/3)
    let A₁ := 6 * s₁^2
    if h₂ : A₂ = 2 * A₁ then
      let s₂ := (A₂ / 6)^(1/2)
      let V₂ := s₂^3
      V₂
    else 0
  else 0

theorem volume_is_correct : volume_of_target_cube 8 48 = 16 * Real.sqrt 2 :=
by
  sorry

end volume_is_correct_l511_51155


namespace tan_ratio_proof_l511_51129

theorem tan_ratio_proof (α : ℝ) (h : 5 * Real.sin (2 * α) = Real.sin 2) : 
  Real.tan (α + 1 * Real.pi / 180) / Real.tan (α - 1 * Real.pi / 180) = - 3 / 2 := 
sorry

end tan_ratio_proof_l511_51129


namespace find_sides_of_triangle_l511_51188

theorem find_sides_of_triangle (c : ℝ) (θ : ℝ) (h_ratio : ℝ) 
  (h_c : c = 2 * Real.sqrt 7)
  (h_theta : θ = Real.pi / 6) -- 30 degrees in radians
  (h_ratio_eq : ∃ k : ℝ, ∀ a b : ℝ, a = k ∧ b = h_ratio * k) :
  ∃ (a b : ℝ), a = 2 ∧ b = 4 * Real.sqrt 3 := by
  sorry

end find_sides_of_triangle_l511_51188


namespace original_denominator_l511_51187

theorem original_denominator (d : ℤ) : 
  (∀ n : ℤ, n = 3 → (n + 8) / (d + 8) = 1 / 3) → d = 25 :=
by
  intro h
  specialize h 3 rfl
  sorry

end original_denominator_l511_51187


namespace solve_for_z_l511_51157

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
  sorry

end solve_for_z_l511_51157


namespace two_pow_start_digits_l511_51167

theorem two_pow_start_digits (A : ℕ) : 
  ∃ (m n : ℕ), 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1) :=
  sorry

end two_pow_start_digits_l511_51167


namespace trig_problem_l511_51134

variables (θ : ℝ)

theorem trig_problem (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.tan θ + 1 / Real.tan θ = 4 :=
sorry

end trig_problem_l511_51134


namespace minimum_value_of_a_l511_51118

theorem minimum_value_of_a (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2 - b * c) 
  (h2 : (1/2) * b * c * (Real.sin A) = (3 * Real.sqrt 3) / 4)
  (h3 : A = Real.arccos (1/2)) :
  a ≥ Real.sqrt 3 := sorry

end minimum_value_of_a_l511_51118


namespace pebbles_sum_at_12_days_l511_51133

def pebbles_collected (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + pebbles_collected (n - 1)

theorem pebbles_sum_at_12_days : pebbles_collected 12 = 78 := by
  -- This would be the place for the proof, but adding sorry as instructed.
  sorry

end pebbles_sum_at_12_days_l511_51133


namespace num_terminating_decimals_l511_51154

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 518) :
  (∃ k, (1 ≤ k ∧ k ≤ 518) ∧ n = k * 21) ↔ n = 24 :=
sorry

end num_terminating_decimals_l511_51154


namespace didi_total_fund_l511_51111

-- Define the conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℕ := 1
def first_business_owner_donation_per_slice : ℚ := 0.5
def second_business_owner_donation_per_slice : ℚ := 0.25

-- Define the proof problem statement
theorem didi_total_fund (h1 : cakes * slices_per_cake = 80)
    (h2 : (80 : ℕ) * price_per_slice = 80)
    (h3 : (80 : ℕ) * first_business_owner_donation_per_slice = 40)
    (h4 : (80 : ℕ) * second_business_owner_donation_per_slice = 20) : 
    (80 : ℕ) + 40 + 20 = 140 := by
  -- The proof itself will be constructed here
  sorry

end didi_total_fund_l511_51111


namespace range_subset_pos_iff_l511_51195

theorem range_subset_pos_iff (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_subset_pos_iff_l511_51195


namespace sum_of_ages_is_29_l511_51123

theorem sum_of_ages_is_29 (age1 age2 age3 : ℕ) (h1 : age1 = 9) (h2 : age2 = 9) (h3 : age3 = 11) :
  age1 + age2 + age3 = 29 := by
  -- skipping the proof
  sorry

end sum_of_ages_is_29_l511_51123


namespace function_range_is_correct_l511_51132

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 - 6 * x + 17) }

theorem function_range_is_correct : function_range = {x : ℝ | x ≤ Real.log 8} :=
by
  sorry

end function_range_is_correct_l511_51132


namespace sqrt_of_225_eq_15_l511_51199

theorem sqrt_of_225_eq_15 : Real.sqrt 225 = 15 :=
by
  sorry

end sqrt_of_225_eq_15_l511_51199


namespace minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l511_51138

theorem minimum_additional_games_to_reach_90_percent (N : ℕ) : 
  (2 + N) * 10 ≥ (5 + N) * 9 ↔ N ≥ 25 := 
sorry

-- An alternative approach to assert directly as exactly 25 by using the condition’s natural number ℕ could be as follows:
theorem hawks_minimum_games_needed_to_win (N : ℕ) : 
  ∀ N, (2 + N) * 10 / (5 + N) ≥ 9 / 10 → N ≥ 25 := 
sorry

end minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l511_51138


namespace monkeys_bananas_l511_51171

theorem monkeys_bananas (c₁ c₂ c₃ : ℕ) (h1 : ∀ (k₁ k₂ k₃ : ℕ), k₁ = c₁ → k₂ = c₂ → k₃ = c₃ → 4 * (k₁ / 3 + k₂ / 6 + k₃ / 18) = 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) ∧ 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) = k₁ / 6 + k₂ / 6 + k₃ / 6)
  (h2 : c₃ % 6 = 0) (h3 : 4 * (c₁ / 3 + c₂ / 6 + c₃ / 18) < 2 * (c₁ / 6 + c₂ / 3 + c₃ / 18 + 1)) :
  c₁ + c₂ + c₃ = 2352 :=
sorry

end monkeys_bananas_l511_51171


namespace find_xsq_plus_inv_xsq_l511_51126

theorem find_xsq_plus_inv_xsq (x : ℝ) (h : 35 = x^6 + 1/(x^6)) : x^2 + 1/(x^2) = 37 :=
sorry

end find_xsq_plus_inv_xsq_l511_51126


namespace arcsin_cos_arcsin_arccos_sin_arccos_l511_51158

-- Define the statement
theorem arcsin_cos_arcsin_arccos_sin_arccos (x : ℝ) 
  (h1 : -1 ≤ x) 
  (h2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) := 
sorry

end arcsin_cos_arcsin_arccos_sin_arccos_l511_51158


namespace difference_in_perimeters_of_rectangles_l511_51104

theorem difference_in_perimeters_of_rectangles 
  (l h : ℝ) (hl : l ≥ 0) (hh : h ≥ 0) :
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  difference = 24 :=
by
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  sorry

end difference_in_perimeters_of_rectangles_l511_51104


namespace angle_BAC_in_isosceles_triangle_l511_51106

theorem angle_BAC_in_isosceles_triangle
  (A B C D : Type)
  (AB AC : ℝ)
  (BD DC : ℝ)
  (angle_BDA : ℝ)
  (isosceles_triangle : AB = AC)
  (midpoint_D : BD = DC)
  (external_angle_D : angle_BDA = 120) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 :=
by
  sorry

end angle_BAC_in_isosceles_triangle_l511_51106


namespace compute_n_l511_51165

theorem compute_n (avg1 avg2 avg3 avg4 avg5 : ℚ) (h1 : avg1 = 1234 ∨ avg2 = 1234 ∨ avg3 = 1234 ∨ avg4 = 1234 ∨ avg5 = 1234)
  (h2 : avg1 = 345 ∨ avg2 = 345 ∨ avg3 = 345 ∨ avg4 = 345 ∨ avg5 = 345)
  (h3 : avg1 = 128 ∨ avg2 = 128 ∨ avg3 = 128 ∨ avg4 = 128 ∨ avg5 = 128)
  (h4 : avg1 = 19 ∨ avg2 = 19 ∨ avg3 = 19 ∨ avg4 = 19 ∨ avg5 = 19)
  (h5 : avg1 = 9.5 ∨ avg2 = 9.5 ∨ avg3 = 9.5 ∨ avg4 = 9.5 ∨ avg5 = 9.5) :
  ∃ n : ℕ, n = 2014 :=
by
  sorry

end compute_n_l511_51165


namespace paul_taxes_and_fees_l511_51103

theorem paul_taxes_and_fees 
  (hourly_wage: ℝ) 
  (hours_worked : ℕ)
  (spent_on_gummy_bears_percentage : ℝ)
  (final_amount : ℝ)
  (gross_earnings := hourly_wage * hours_worked)
  (taxes_and_fees := gross_earnings - final_amount / (1 - spent_on_gummy_bears_percentage)):
  hourly_wage = 12.50 →
  hours_worked = 40 →
  spent_on_gummy_bears_percentage = 0.15 →
  final_amount = 340 →
  taxes_and_fees / gross_earnings = 0.20 :=
by
  intros
  sorry

end paul_taxes_and_fees_l511_51103


namespace hoseok_more_paper_than_minyoung_l511_51180

theorem hoseok_more_paper_than_minyoung : 
  ∀ (initial : ℕ) (minyoung_bought : ℕ) (hoseok_bought : ℕ), 
  initial = 150 →
  minyoung_bought = 32 →
  hoseok_bought = 49 →
  (initial + hoseok_bought) - (initial + minyoung_bought) = 17 :=
by
  intros initial minyoung_bought hoseok_bought h_initial h_min h_hos
  sorry

end hoseok_more_paper_than_minyoung_l511_51180


namespace find_last_number_l511_51172

theorem find_last_number (A B C D : ℕ) 
  (h1 : A + B + C = 18) 
  (h2 : B + C + D = 9) 
  (h3 : A + D = 13) 
  : D = 2 := by 
  sorry

end find_last_number_l511_51172


namespace arithmetic_sequence_fifth_term_l511_51110

theorem arithmetic_sequence_fifth_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 6) (h3 : a 3 = 2) (h_arith_seq : ∀ n, a (n + 1) = a n + d) : a 5 = -2 :=
sorry

end arithmetic_sequence_fifth_term_l511_51110


namespace quadratic_polynomial_discriminant_l511_51119

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end quadratic_polynomial_discriminant_l511_51119


namespace nested_series_sum_l511_51127

theorem nested_series_sum : 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))) = 126 :=
by
  sorry

end nested_series_sum_l511_51127


namespace cos_300_eq_half_l511_51148

theorem cos_300_eq_half : Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l511_51148


namespace problem_statement_l511_51161

theorem problem_statement (x y : ℚ) (h1 : x + y = 500) (h2 : x / y = 4 / 5) : y - x = 500 / 9 := 
by
  sorry

end problem_statement_l511_51161


namespace solve_for_q_l511_51183

theorem solve_for_q (t h q : ℝ) (h_eq : h = -14 * (t - 3)^2 + q) (h_5_eq : h = 94) (t_5_eq : t = 3 + 2) : q = 150 :=
by
  sorry

end solve_for_q_l511_51183


namespace bugs_ate_each_l511_51169

theorem bugs_ate_each : 
  ∀ (total_bugs total_flowers each_bug_flowers : ℕ), 
    total_bugs = 3 ∧ total_flowers = 6 ∧ each_bug_flowers = total_flowers / total_bugs -> each_bug_flowers = 2 := by
  sorry

end bugs_ate_each_l511_51169


namespace number_of_squares_l511_51192

def draws_88_lines (lines: ℕ) : Prop := lines = 88
def draws_triangles (triangles: ℕ) : Prop := triangles = 12
def draws_pentagons (pentagons: ℕ) : Prop := pentagons = 4

theorem number_of_squares (triangles pentagons sq_sides: ℕ) (h1: draws_88_lines (triangles * 3 + pentagons * 5 + sq_sides * 4))
    (h2: draws_triangles triangles) (h3: draws_pentagons pentagons) : sq_sides = 8 := by
  sorry

end number_of_squares_l511_51192


namespace max_value_of_expression_l511_51190

theorem max_value_of_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + 2 * b + 3 * c = 1) :
    (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) ≤ 7) :=
sorry

end max_value_of_expression_l511_51190


namespace arithmetic_proof_l511_51143

def arithmetic_expression := 3889 + 12.952 - 47.95000000000027
def expected_result := 3854.002

theorem arithmetic_proof : arithmetic_expression = expected_result := by
  -- The proof goes here
  sorry

end arithmetic_proof_l511_51143


namespace grocery_store_spending_l511_51182

/-- Lenny has $84 initially. He spent $24 on video games and has $39 left.
We need to prove that he spent $21 at the grocery store. --/
theorem grocery_store_spending (initial_amount spent_on_video_games amount_left after_games_left : ℕ) 
    (h1 : initial_amount = 84)
    (h2 : spent_on_video_games = 24)
    (h3 : amount_left = 39)
    (h4 : after_games_left = initial_amount - spent_on_video_games) 
    : after_games_left - amount_left = 21 := 
sorry

end grocery_store_spending_l511_51182


namespace bucket_weight_l511_51156

theorem bucket_weight (c d : ℝ) (x y : ℝ) 
  (h1 : x + 3/4 * y = c) 
  (h2 : x + 1/3 * y = d) :
  x + 1/4 * y = (6 * d - c) / 5 := 
sorry

end bucket_weight_l511_51156


namespace james_total_cost_l511_51166

def suit1 := 300
def suit2_pretail := 3 * suit1
def suit2 := suit2_pretail + 200
def total_cost := suit1 + suit2

theorem james_total_cost : total_cost = 1400 := by
  sorry

end james_total_cost_l511_51166


namespace natural_number_squares_l511_51149

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares_l511_51149


namespace f_log2_9_eq_neg_16_div_9_l511_51131

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x - 2) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2 ^ x

theorem f_log2_9_eq_neg_16_div_9 : f (Real.log 9 / Real.log 2) = -16 / 9 := 
by 
  sorry

end f_log2_9_eq_neg_16_div_9_l511_51131


namespace min_value_expr_l511_51175

theorem min_value_expr (a b : ℝ) (h : a - 2 * b + 8 = 0) : ∃ x : ℝ, x = 2^a + 1 / 4^b ∧ x = 1 / 8 :=
by
  sorry

end min_value_expr_l511_51175


namespace sqrt_90000_eq_300_l511_51114

theorem sqrt_90000_eq_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_eq_300_l511_51114


namespace Maria_bought_7_roses_l511_51186

theorem Maria_bought_7_roses
  (R : ℕ)
  (h1 : ∀ f : ℕ, 6 * f = 6 * f)
  (h2 : ∀ r : ℕ, ∃ d : ℕ, r = R ∧ d = 3)
  (h3 : 6 * R + 18 = 60) : R = 7 := by
  sorry

end Maria_bought_7_roses_l511_51186


namespace correct_option_l511_51100

theorem correct_option : (-1 - 3 = -4) ∧ ¬(-2 + 8 = 10) ∧ ¬(-2 * 2 = 4) ∧ ¬(-8 / -1 = -1 / 8) :=
by
  sorry

end correct_option_l511_51100


namespace total_legs_l511_51113

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l511_51113


namespace gcd_consecutive_term_max_l511_51101

def b (n : ℕ) : ℕ := n.factorial + 2^n + n 

theorem gcd_consecutive_term_max (n : ℕ) (hn : n ≥ 0) :
  ∃ m ≤ (n : ℕ), (m = 2) := sorry

end gcd_consecutive_term_max_l511_51101


namespace find_initial_time_l511_51198

-- The initial distance d
def distance : ℕ := 288

-- Conditions
def initial_condition (v t : ℕ) : Prop :=
  distance = v * t

def new_condition (t : ℕ) : Prop :=
  distance = 32 * (3 * t / 2)

-- Proof Problem Statement
theorem find_initial_time (v t : ℕ) (h1 : initial_condition v t)
  (h2 : new_condition t) : t = 6 := by
  sorry

end find_initial_time_l511_51198


namespace geometric_sum_5_l511_51121

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = a n * r ∧ a (m + 1) = a m * r

theorem geometric_sum_5 (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) (h3 : ∀ n, 0 < a n) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sum_5_l511_51121


namespace problem_inequality_l511_51130

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem problem_inequality (a : ℝ) (m n : ℝ) 
  (h1 : m ∈ Set.Icc 0 2) (h2 : n ∈ Set.Icc 0 2) 
  (h3 : |m - n| ≥ 1) 
  (h4 : f m a / f n a = 1) : 
  1 ≤ a / (Real.exp 1 - 1) ∧ a / (Real.exp 1 - 1) ≤ Real.exp 1 :=
by sorry

end problem_inequality_l511_51130


namespace find_unit_vector_l511_51164

theorem find_unit_vector (a b : ℝ) : 
  a^2 + b^2 = 1 ∧ 3 * a + 4 * b = 0 →
  (a = 4 / 5 ∧ b = -3 / 5) ∨ (a = -4 / 5 ∧ b = 3 / 5) :=
by sorry

end find_unit_vector_l511_51164


namespace sally_pens_initial_count_l511_51168

theorem sally_pens_initial_count :
  ∃ P : ℕ, (P - (7 * 44)) / 2 = 17 ∧ P = 342 :=
by 
  sorry

end sally_pens_initial_count_l511_51168


namespace binary_to_decimal_101101_l511_51102

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end binary_to_decimal_101101_l511_51102


namespace parabola_line_dot_product_l511_51162

theorem parabola_line_dot_product (k x1 x2 y1 y2 : ℝ) 
  (h_line: ∀ x, y = k * x + 2)
  (h_parabola: ∀ x, y = (1 / 4) * x ^ 2) 
  (h_A: y1 = k * x1 + 2 ∧ y1 = (1 / 4) * x1 ^ 2)
  (h_B: y2 = k * x2 + 2 ∧ y2 = (1 / 4) * x2 ^ 2) :
  x1 * x2 + y1 * y2 = -4 := 
sorry

end parabola_line_dot_product_l511_51162


namespace discount_on_pickles_l511_51140

theorem discount_on_pickles :
  ∀ (meat_weight : ℝ) (meat_price_per_pound : ℝ) (bun_price : ℝ) (lettuce_price : ℝ)
    (tomato_weight : ℝ) (tomato_price_per_pound : ℝ) (pickles_price : ℝ) (total_paid : ℝ) (change : ℝ),
  meat_weight = 2 ∧
  meat_price_per_pound = 3.50 ∧
  bun_price = 1.50 ∧
  lettuce_price = 1.00 ∧
  tomato_weight = 1.5 ∧
  tomato_price_per_pound = 2.00 ∧
  pickles_price = 2.50 ∧
  total_paid = 20.00 ∧
  change = 6 →
  pickles_price - (total_paid - change - (meat_weight * meat_price_per_pound + tomato_weight * tomato_price_per_pound + bun_price + lettuce_price)) = 1 := 
by
  -- Begin the proof here (not required for this task)
  sorry

end discount_on_pickles_l511_51140


namespace sin_330_eq_neg_half_l511_51196

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  -- Proof would go here
  sorry

end sin_330_eq_neg_half_l511_51196


namespace grace_hours_pulling_weeds_l511_51184

variable (Charge_mowing : ℕ) (Charge_weeding : ℕ) (Charge_mulching : ℕ)
variable (H_m : ℕ) (H_u : ℕ) (E_s : ℕ)

theorem grace_hours_pulling_weeds 
  (Charge_mowing_eq : Charge_mowing = 6)
  (Charge_weeding_eq : Charge_weeding = 11)
  (Charge_mulching_eq : Charge_mulching = 9)
  (H_m_eq : H_m = 63)
  (H_u_eq : H_u = 10)
  (E_s_eq : E_s = 567) :
  ∃ W : ℕ, 6 * 63 + 11 * W + 9 * 10 = 567 ∧ W = 9 := by
  sorry

end grace_hours_pulling_weeds_l511_51184


namespace smallest_sum_l511_51125

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l511_51125


namespace matrix_solution_correct_l511_51141

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -7/3], ![4, -1/3]]

def v1 : Fin 2 → ℚ := ![4, 0]
def v2 : Fin 2 → ℚ := ![2, 3]

def result1 : Fin 2 → ℚ := ![12, 16]
def result2 : Fin 2 → ℚ := ![-1, 7]

theorem matrix_solution_correct :
  (mulVec N v1 = result1) ∧ 
  (mulVec N v2 = result2) := by
  sorry

end matrix_solution_correct_l511_51141


namespace complement_of_M_in_U_l511_51153

namespace ProofProblem

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end ProofProblem

end complement_of_M_in_U_l511_51153


namespace a_eq_3x_or_neg2x_l511_51109

theorem a_eq_3x_or_neg2x (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 19 * x^3) (h3 : a - b = x) :
    a = 3 * x ∨ a = -2 * x :=
by
  -- The proof will go here
  sorry

end a_eq_3x_or_neg2x_l511_51109


namespace fill_in_the_blank_l511_51139

theorem fill_in_the_blank (x : ℕ) (h : (x - x) + x * x + x / x = 50) : x = 7 :=
sorry

end fill_in_the_blank_l511_51139


namespace probability_of_specific_sequence_l511_51197

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem probability_of_specific_sequence :
  (1 - p)^7 * p^3 = sorry :=
by sorry

end probability_of_specific_sequence_l511_51197


namespace speed_of_first_train_l511_51178

-- Definitions of the conditions
def ratio_speed (speed1 speed2 : ℝ) := speed1 / speed2 = 7 / 8
def speed_of_second_train := 400 / 4

-- The theorem we want to prove
theorem speed_of_first_train (speed2 := speed_of_second_train) (h : ratio_speed S1 speed2) :
  S1 = 87.5 :=
by 
  sorry

end speed_of_first_train_l511_51178


namespace smallest_angle_of_cyclic_quadrilateral_l511_51181

theorem smallest_angle_of_cyclic_quadrilateral (angles : ℝ → ℝ) (a d : ℝ) :
  -- Conditions
  (∀ n : ℕ, angles n = a + n * d) ∧ 
  (angles 3 = 140) ∧
  (a + d + (a + 3 * d) = 180) →
  -- Conclusion
  (a = 40) :=
by sorry

end smallest_angle_of_cyclic_quadrilateral_l511_51181


namespace initial_students_count_l511_51159

theorem initial_students_count (n : ℕ) (W : ℝ) :
  (W = n * 28) →
  (W + 4 = (n + 1) * 27.2) →
  n = 29 :=
by
  intros hW hw_avg
  -- Proof goes here
  sorry

end initial_students_count_l511_51159


namespace periodic_odd_function_value_l511_51145

theorem periodic_odd_function_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_periodic : ∀ x : ℝ, f (x + 2) = f x) (h_value : f 0.5 = -1) : f 7.5 = 1 :=
by
  -- Proof would go here.
  sorry

end periodic_odd_function_value_l511_51145


namespace range_a_condition_l511_51189

theorem range_a_condition (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ a → x^2 ≤ 2 * x + 3) ↔ (1 / 2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_a_condition_l511_51189


namespace number_of_C_atoms_in_compound_is_4_l511_51108

def atomic_weight_C : ℕ := 12
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16

def molecular_weight : ℕ := 65

def weight_contributed_by_H_O : ℕ := atomic_weight_H + atomic_weight_O -- 17 amu

def weight_contributed_by_C : ℕ := molecular_weight - weight_contributed_by_H_O -- 48 amu

def number_of_C_atoms := weight_contributed_by_C / atomic_weight_C -- The quotient of 48 amu divided by 12 amu per C atom

theorem number_of_C_atoms_in_compound_is_4 : number_of_C_atoms = 4 :=
by
  sorry -- This is where the proof would go, but it's omitted as per instructions.

end number_of_C_atoms_in_compound_is_4_l511_51108


namespace expectation_fair_coin_5_tosses_l511_51193

noncomputable def fairCoinExpectation (n : ℕ) : ℚ :=
  n * (1/2)

theorem expectation_fair_coin_5_tosses :
  fairCoinExpectation 5 = 5 / 2 :=
by
  sorry

end expectation_fair_coin_5_tosses_l511_51193


namespace value_of_a0_plus_a8_l511_51122

/-- Theorem stating the value of a0 + a8 from the given polynomial equation -/
theorem value_of_a0_plus_a8 (a_0 a_8 : ℤ) :
  (∀ x : ℤ, (1 + x) ^ 10 = a_0 + a_1 * (1 - x) + a_2 * (1 - x) ^ 2 + 
              a_3 * (1 - x) ^ 3 + a_4 * (1 - x) ^ 4 + a_5 * (1 - x) ^ 5 +
              a_6 * (1 - x) ^ 6 + a_7 * (1 - x) ^ 7 + a_8 * (1 - x) ^ 8 + 
              a_9 * (1 - x) ^ 9 + a_10 * (1 - x) ^ 10) →
  a_0 + a_8 = 1204 :=
by
  sorry

end value_of_a0_plus_a8_l511_51122


namespace ones_digit_of_prime_in_sequence_l511_51170

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def valid_arithmetic_sequence (p1 p2 p3 p4: Nat) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4) ∧ (p3 = p2 + 4) ∧ (p4 = p3 + 4)

theorem ones_digit_of_prime_in_sequence (p1 p2 p3 p4 : Nat) (hp_seq : valid_arithmetic_sequence p1 p2 p3 p4) (hp1_gt_3 : p1 > 3) : 
  (p1 % 10) = 9 :=
sorry

end ones_digit_of_prime_in_sequence_l511_51170


namespace total_red_yellow_black_l511_51146

/-- Calculate the total number of red, yellow, and black shirts Gavin has,
given that he has 420 shirts in total, 85 of them are blue, and 157 are
green. -/
theorem total_red_yellow_black (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) :
  total_shirts = 420 → blue_shirts = 85 → green_shirts = 157 → 
  (total_shirts - (blue_shirts + green_shirts) = 178) :=
by
  intros h1 h2 h3
  sorry

end total_red_yellow_black_l511_51146


namespace distance_between_towns_l511_51112

theorem distance_between_towns 
  (rate1 rate2 : ℕ) (time : ℕ) (distance : ℕ)
  (h_rate1 : rate1 = 48)
  (h_rate2 : rate2 = 42)
  (h_time : time = 5)
  (h_distance : distance = rate1 * time + rate2 * time) : 
  distance = 450 :=
by
  sorry

end distance_between_towns_l511_51112


namespace angle_sum_acutes_l511_51173

theorem angle_sum_acutes (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_condition : |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0) : 
  α + β = π * 5/12 :=
by sorry

end angle_sum_acutes_l511_51173


namespace complete_the_square_l511_51147

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) :=
by
  intro h
  sorry

end complete_the_square_l511_51147


namespace range_of_k_for_real_roots_l511_51179

theorem range_of_k_for_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - x + 3 = 0) ↔ (k <= 1 / 12 ∧ k ≠ 0) :=
sorry

end range_of_k_for_real_roots_l511_51179


namespace james_browsers_l511_51144

def num_tabs_per_window := 10
def num_windows_per_browser := 3
def total_tabs := 60

theorem james_browsers : ∃ B : ℕ, (B * (num_windows_per_browser * num_tabs_per_window) = total_tabs) ∧ (B = 2) := sorry

end james_browsers_l511_51144


namespace sum_sequence_six_l511_51191

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℤ := sorry

theorem sum_sequence_six :
  (∀ n, S n = 2 * a n + 1) → S 6 = 63 :=
by
  sorry

end sum_sequence_six_l511_51191


namespace digit_solve_l511_51124

theorem digit_solve : ∀ (D : ℕ), D < 10 → (D * 9 + 6 = D * 10 + 3) → D = 3 :=
by
  intros D hD h
  sorry

end digit_solve_l511_51124


namespace isabel_homework_problems_l511_51142

theorem isabel_homework_problems (initial_problems finished_problems remaining_pages problems_per_page : ℕ) 
  (h1 : initial_problems = 72)
  (h2 : finished_problems = 32)
  (h3 : remaining_pages = 5)
  (h4 : initial_problems - finished_problems = 40)
  (h5 : 40 = remaining_pages * problems_per_page) : 
  problems_per_page = 8 := 
by sorry

end isabel_homework_problems_l511_51142


namespace inequality_ge_zero_l511_51120

theorem inequality_ge_zero (x y z : ℝ) : 
  4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := 
sorry

end inequality_ge_zero_l511_51120


namespace first_player_wins_l511_51151

def initial_piles (p1 p2 : Nat) : Prop :=
  p1 = 33 ∧ p2 = 35

def winning_strategy (p1 p2 : Nat) : Prop :=
  ∃ moves : List (Nat × Nat), 
  (initial_piles p1 p2) →
  (∀ (p1' p2' : Nat), 
    (p1', p2') ∈ moves →
    p1' = 1 ∧ p2' = 1 ∨ p1' = 2 ∧ p2' = 1)

theorem first_player_wins : winning_strategy 33 35 :=
sorry

end first_player_wins_l511_51151


namespace root_monotonicity_l511_51107

noncomputable def f (x : ℝ) := 3^x + 2 / (1 - x)

theorem root_monotonicity
  (x0 : ℝ) (H_root : f x0 = 0)
  (x1 x2 : ℝ) (H1 : x1 > 1) (H2 : x1 < x0) (H3 : x2 > x0) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end root_monotonicity_l511_51107


namespace sin_300_eq_neg_sqrt3_div_2_l511_51160

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l511_51160


namespace order_of_numbers_l511_51116

theorem order_of_numbers (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : 
  -m > n ∧ n > -n ∧ -n > m := 
by
  sorry

end order_of_numbers_l511_51116


namespace C_share_l511_51136

theorem C_share (a b c : ℕ) (h1 : a + b + c = 1010)
                (h2 : ∃ k : ℕ, a = 3 * k + 25 ∧ b = 2 * k + 10 ∧ c = 5 * k + 15) : c = 495 :=
by
  -- Sorry is used to skip the proof
  sorry

end C_share_l511_51136


namespace ratio_mercedes_jonathan_l511_51135

theorem ratio_mercedes_jonathan (M : ℝ) (J : ℝ) (D : ℝ) 
  (h1 : J = 7.5) 
  (h2 : D = M + 2) 
  (h3 : M + D = 32) : M / J = 2 :=
by
  sorry

end ratio_mercedes_jonathan_l511_51135


namespace tim_will_attend_game_probability_l511_51152

theorem tim_will_attend_game_probability :
  let P_rain := 0.60
  let P_sunny := 1 - P_rain
  let P_attends_given_rain := 0.25
  let P_attends_given_sunny := 0.70
  let P_rain_and_attends := P_rain * P_attends_given_rain
  let P_sunny_and_attends := P_sunny * P_attends_given_sunny
  (P_rain_and_attends + P_sunny_and_attends) = 0.43 :=
by
  sorry

end tim_will_attend_game_probability_l511_51152


namespace value_of_a_ab_b_l511_51177

-- Define conditions
variables {a b : ℝ} (h1 : a * b = 1) (h2 : b = a + 2)

-- The proof problem
theorem value_of_a_ab_b : a - a * b - b = -3 :=
by
  sorry

end value_of_a_ab_b_l511_51177


namespace find_z_given_x4_l511_51117

theorem find_z_given_x4 (k : ℝ) (z : ℝ) (x : ℝ) :
  (7 * 4 = k / 2^3) → (7 * z = k / x^3) → (x = 4) → (z = 0.5) :=
by
  intro h1 h2 h3
  sorry

end find_z_given_x4_l511_51117


namespace first_issue_pages_l511_51115

-- Define the conditions
def total_pages := 220
def pages_third_issue (x : ℕ) := x + 4

-- Statement of the problem
theorem first_issue_pages (x : ℕ) (hx : 3 * x + 4 = total_pages) : x = 72 :=
sorry

end first_issue_pages_l511_51115


namespace freshmen_count_l511_51137

theorem freshmen_count (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ↔ n = 574 := 
by sorry

end freshmen_count_l511_51137
