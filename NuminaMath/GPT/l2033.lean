import Mathlib

namespace NUMINAMATH_GPT_xyz_value_l2033_203335

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l2033_203335


namespace NUMINAMATH_GPT_evaluate_expression_l2033_203361

theorem evaluate_expression : ((3^4)^3 + 5) - ((4^3)^4 + 5) = -16245775 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2033_203361


namespace NUMINAMATH_GPT_ratio_q_p_l2033_203346

variable (p q : ℝ)
variable (hpq_pos : 0 < p ∧ 0 < q)
variable (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18)

theorem ratio_q_p (p q : ℝ) (hpq_pos : 0 < p ∧ 0 < q) 
    (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18) :
    q / p = (Real.sqrt 5 - 1) / 2 :=
  sorry

end NUMINAMATH_GPT_ratio_q_p_l2033_203346


namespace NUMINAMATH_GPT_weighted_average_fish_caught_l2033_203336

-- Define the daily catches for each person
def AangCatches := [5, 7, 9]
def SokkaCatches := [8, 5, 6]
def TophCatches := [10, 12, 8]
def ZukoCatches := [6, 7, 10]

-- Define the group catches
def GroupCatches := AangCatches ++ SokkaCatches ++ TophCatches ++ ZukoCatches

-- Calculate the total number of fish caught by the group
def TotalFishCaught := List.sum GroupCatches

-- Calculate the total number of days fished by the group
def TotalDaysFished := 4 * 3

-- Calculate the weighted average
def WeightedAverage := TotalFishCaught.toFloat / TotalDaysFished.toFloat

-- Proof statement
theorem weighted_average_fish_caught :
  WeightedAverage = 7.75 := by
  sorry

end NUMINAMATH_GPT_weighted_average_fish_caught_l2033_203336


namespace NUMINAMATH_GPT_frankie_candies_l2033_203350

theorem frankie_candies (M D F : ℕ) (h1 : M = 92) (h2 : D = 18) (h3 : F = M - D) : F = 74 :=
by
  sorry

end NUMINAMATH_GPT_frankie_candies_l2033_203350


namespace NUMINAMATH_GPT_not_possible_to_create_3_piles_l2033_203303

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end NUMINAMATH_GPT_not_possible_to_create_3_piles_l2033_203303


namespace NUMINAMATH_GPT_scientific_notation_of_concentration_l2033_203355

theorem scientific_notation_of_concentration :
  0.000042 = 4.2 * 10^(-5) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_concentration_l2033_203355


namespace NUMINAMATH_GPT_midpoint_product_l2033_203364

theorem midpoint_product (x' y' : ℤ) 
  (h1 : (0 + x') / 2 = 2) 
  (h2 : (9 + y') / 2 = 4) : 
  (x' * y') = -4 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_product_l2033_203364


namespace NUMINAMATH_GPT_mrs_jackson_boxes_l2033_203338

theorem mrs_jackson_boxes (decorations_per_box used_decorations given_decorations : ℤ) 
(h1 : decorations_per_box = 15)
(h2 : used_decorations = 35)
(h3 : given_decorations = 25) :
  (used_decorations + given_decorations) / decorations_per_box = 4 := 
by sorry

end NUMINAMATH_GPT_mrs_jackson_boxes_l2033_203338


namespace NUMINAMATH_GPT_max_2x_plus_y_value_l2033_203382

open Real

def on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def max_value_2x_plus_y (P : ℝ × ℝ) (h : on_ellipse P) : ℝ := 
  2 * P.1 + P.2

theorem max_2x_plus_y_value (P : ℝ × ℝ) (h : on_ellipse P):
  ∃ (m : ℝ), max_value_2x_plus_y P h = m ∧ m = sqrt 17 :=
sorry

end NUMINAMATH_GPT_max_2x_plus_y_value_l2033_203382


namespace NUMINAMATH_GPT_jason_cost_l2033_203353

variable (full_page_cost_per_square_inch : ℝ := 6.50)
variable (half_page_cost_per_square_inch : ℝ := 8)
variable (quarter_page_cost_per_square_inch : ℝ := 10)

variable (full_page_area : ℝ := 9 * 12)
variable (half_page_area : ℝ := full_page_area / 2)
variable (quarter_page_area : ℝ := full_page_area / 4)

variable (half_page_ads : ℝ := 1)
variable (quarter_page_ads : ℝ := 4)

variable (total_ads : ℝ := half_page_ads + quarter_page_ads)
variable (bulk_discount : ℝ := if total_ads >= 4 then 0.10 else 0.0)

variable (half_page_cost : ℝ := half_page_area * half_page_cost_per_square_inch)
variable (quarter_page_cost : ℝ := quarter_page_ads * (quarter_page_area * quarter_page_cost_per_square_inch))

variable (total_cost_before_discount : ℝ := half_page_cost + quarter_page_cost)
variable (discount_amount : ℝ := total_cost_before_discount * bulk_discount)
variable (final_cost : ℝ := total_cost_before_discount - discount_amount)

theorem jason_cost :
  final_cost = 1360.80 := by
  sorry

end NUMINAMATH_GPT_jason_cost_l2033_203353


namespace NUMINAMATH_GPT_find_d_l2033_203314

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2 * c^2 + 4 = 2 * d + Real.sqrt (a^2 + b^2 + c - d)) :
  d = 1/2 :=
sorry

end NUMINAMATH_GPT_find_d_l2033_203314


namespace NUMINAMATH_GPT_school_club_profit_l2033_203383

-- Definition of the problem conditions
def candy_bars_bought : ℕ := 800
def cost_per_four_bars : ℚ := 3
def bars_per_four_bars : ℕ := 4
def sell_price_per_three_bars : ℚ := 2
def bars_per_three_bars : ℕ := 3
def sales_fee_per_bar : ℚ := 0.05

-- Definition for cost calculations
def cost_per_bar : ℚ := cost_per_four_bars / bars_per_four_bars
def total_cost : ℚ := candy_bars_bought * cost_per_bar

-- Definition for revenue calculations
def sell_price_per_bar : ℚ := sell_price_per_three_bars / bars_per_three_bars
def total_revenue : ℚ := candy_bars_bought * sell_price_per_bar

-- Definition for total sales fee
def total_sales_fee : ℚ := candy_bars_bought * sales_fee_per_bar

-- Definition of profit
def profit : ℚ := total_revenue - total_cost - total_sales_fee

-- The statement to be proved
theorem school_club_profit : profit = -106.64 := by sorry

end NUMINAMATH_GPT_school_club_profit_l2033_203383


namespace NUMINAMATH_GPT_proposition_and_implication_l2033_203318

theorem proposition_and_implication
  (m : ℝ)
  (h1 : 5/4 * (m^2 + m) > 0)
  (h2 : 1 + 9 - 4 * (5/4 * (m^2 + m)) > 0)
  (h3 : m + 3/2 ≥ 0)
  (h4 : m - 1/2 ≤ 0) :
  (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_GPT_proposition_and_implication_l2033_203318


namespace NUMINAMATH_GPT_problem_l2033_203326

theorem problem (a b c : ℝ) (h1 : ∀ (x : ℝ), x^2 + 3 * x - 1 = 0 → x^4 + a * x^2 + b * x + c = 0) :
  a + b + 4 * c + 100 = 93 := 
sorry

end NUMINAMATH_GPT_problem_l2033_203326


namespace NUMINAMATH_GPT_find_d_l2033_203332

-- Definitions of the functions f and g and condition on f(g(x))
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d x : ℝ) (h : f (g x c) c = 15 * x + d) : d = 18 :=
sorry

end NUMINAMATH_GPT_find_d_l2033_203332


namespace NUMINAMATH_GPT_triangle_problem_l2033_203360

noncomputable def triangle_sin_B (a b : ℝ) (A : ℝ) : ℝ :=
  b * Real.sin A / a

noncomputable def triangle_side_c (a b A : ℝ) : ℝ :=
  let discr := b^2 + a^2 - 2 * b * a * Real.cos A
  Real.sqrt discr

noncomputable def sin_diff_angle (sinB cosB sinC cosC : ℝ) : ℝ :=
  sinB * cosC - cosB * sinC

theorem triangle_problem
  (a b : ℝ)
  (A : ℝ)
  (ha : a = Real.sqrt 39)
  (hb : b = 2)
  (hA : A = Real.pi * (2 / 3)) :
  (triangle_sin_B a b A = Real.sqrt 13 / 13) ∧
  (triangle_side_c a b A = 5) ∧
  (sin_diff_angle (Real.sqrt 13 / 13) (2 * Real.sqrt 39 / 13) (5 * Real.sqrt 13 / 26) (3 * Real.sqrt 39 / 26) = -7 * Real.sqrt 3 / 26) :=
by sorry

end NUMINAMATH_GPT_triangle_problem_l2033_203360


namespace NUMINAMATH_GPT_min_value_fraction_subtraction_l2033_203333

theorem min_value_fraction_subtraction
  (a b : ℝ)
  (ha : 0 < a ∧ a ≤ 3 / 4)
  (hb : 0 < b ∧ b ≤ 3 - a)
  (hineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) :
  ∃ a b, (0 < a ∧ a ≤ 3 / 4) ∧ (0 < b ∧ b ≤ 3 - a) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) ∧ (1 / a - b = 1) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_fraction_subtraction_l2033_203333


namespace NUMINAMATH_GPT_equal_number_of_frogs_after_6_months_l2033_203321

theorem equal_number_of_frogs_after_6_months :
  ∃ n : ℕ, 
    n = 6 ∧ 
    (∀ Dn Qn : ℕ, 
      (Dn = 5^(n + 1) ∧ Qn = 3^(n + 5)) → 
      Dn = Qn) :=
by
  sorry

end NUMINAMATH_GPT_equal_number_of_frogs_after_6_months_l2033_203321


namespace NUMINAMATH_GPT_find_simple_interest_rate_l2033_203379

theorem find_simple_interest_rate (P A T SI R : ℝ)
  (hP : P = 750)
  (hA : A = 1125)
  (hT : T = 5)
  (hSI : SI = A - P)
  (hSI_def : SI = (P * R * T) / 100) : R = 10 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_simple_interest_rate_l2033_203379


namespace NUMINAMATH_GPT_divisible_by_900_l2033_203381

theorem divisible_by_900 (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) := 
by 
  sorry

end NUMINAMATH_GPT_divisible_by_900_l2033_203381


namespace NUMINAMATH_GPT_number_of_dogs_in_shelter_l2033_203388

variables (D C R P : ℕ)

-- Conditions
axiom h1 : 15 * C = 7 * D
axiom h2 : 9 * P = 5 * R
axiom h3 : 15 * (C + 8) = 11 * D
axiom h4 : 7 * P = 5 * (R + 6)

theorem number_of_dogs_in_shelter : D = 30 :=
by sorry

end NUMINAMATH_GPT_number_of_dogs_in_shelter_l2033_203388


namespace NUMINAMATH_GPT_simplify_expression_l2033_203357

theorem simplify_expression (x : ℝ) : 5 * x + 2 * x + 7 * x = 14 * x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2033_203357


namespace NUMINAMATH_GPT_cary_needs_six_weekends_l2033_203356

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end NUMINAMATH_GPT_cary_needs_six_weekends_l2033_203356


namespace NUMINAMATH_GPT_overall_percentage_change_in_membership_l2033_203393

theorem overall_percentage_change_in_membership :
  let M := 1
  let fall_inc := 1.08
  let winter_inc := 1.15
  let spring_dec := 0.81
  (M * fall_inc * winter_inc * spring_dec - M) / M * 100 = 24.2 := by
  sorry

end NUMINAMATH_GPT_overall_percentage_change_in_membership_l2033_203393


namespace NUMINAMATH_GPT_decryption_ease_comparison_l2033_203305

def unique_letters_of_thermometer : Finset Char := {'т', 'е', 'р', 'м', 'о'}
def unique_letters_of_remont : Finset Char := {'р', 'е', 'м', 'о', 'н', 'т'}
def easier_to_decrypt : Prop :=
  unique_letters_of_remont.card > unique_letters_of_thermometer.card

theorem decryption_ease_comparison : easier_to_decrypt :=
by
  -- We need to prove that |unique_letters_of_remont| > |unique_letters_of_thermometer|
  sorry

end NUMINAMATH_GPT_decryption_ease_comparison_l2033_203305


namespace NUMINAMATH_GPT_longest_chord_of_circle_l2033_203369

theorem longest_chord_of_circle (r : ℝ) (h : r = 3) : ∃ l, l = 6 := by
  sorry

end NUMINAMATH_GPT_longest_chord_of_circle_l2033_203369


namespace NUMINAMATH_GPT_simply_connected_polyhedron_faces_l2033_203392

def polyhedron_faces_condition (σ3 σ4 σ5 : Nat) (V E F : Nat) : Prop :=
  V - E + F = 2

theorem simply_connected_polyhedron_faces : 
  ∀ (σ3 σ4 σ5 : Nat) (V E F : Nat),
  polyhedron_faces_condition σ3 σ4 σ5 V E F →
  (σ4 = 0 ∧ σ5 = 0 → σ3 ≥ 4) ∧
  (σ3 = 0 ∧ σ5 = 0 → σ4 ≥ 6) ∧
  (σ3 = 0 ∧ σ4 = 0 → σ5 ≥ 12) := 
by
  intros
  sorry

end NUMINAMATH_GPT_simply_connected_polyhedron_faces_l2033_203392


namespace NUMINAMATH_GPT_CandyGivenToJanetEmily_l2033_203339

noncomputable def initial_candy : ℝ := 78.5
noncomputable def candy_left_after_janet : ℝ := 68.75
noncomputable def candy_given_to_emily : ℝ := 2.25

theorem CandyGivenToJanetEmily :
  initial_candy - candy_left_after_janet + candy_given_to_emily = 12 := 
by
  sorry

end NUMINAMATH_GPT_CandyGivenToJanetEmily_l2033_203339


namespace NUMINAMATH_GPT_people_in_gym_l2033_203330

-- Define the initial number of people in the gym
def initial_people : ℕ := 16

-- Define the number of additional people entering the gym
def additional_people : ℕ := 5

-- Define the number of people leaving the gym
def people_leaving : ℕ := 2

-- Define the final number of people in the gym as per the conditions
def final_people (initial : ℕ) (additional : ℕ) (leaving : ℕ) : ℕ :=
  initial + additional - leaving

-- The theorem to prove
theorem people_in_gym : final_people initial_people additional_people people_leaving = 19 :=
  by
    sorry

end NUMINAMATH_GPT_people_in_gym_l2033_203330


namespace NUMINAMATH_GPT_option_d_always_holds_l2033_203302

theorem option_d_always_holds (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b := by
  sorry

end NUMINAMATH_GPT_option_d_always_holds_l2033_203302


namespace NUMINAMATH_GPT_initial_card_count_l2033_203300

theorem initial_card_count (r b : ℕ) (h₁ : (r : ℝ)/(r + b) = 1/4)
    (h₂ : (r : ℝ)/(r + (b + 6)) = 1/6) : r + b = 12 :=
by
  sorry

end NUMINAMATH_GPT_initial_card_count_l2033_203300


namespace NUMINAMATH_GPT_sum_of_first_six_terms_of_geometric_series_l2033_203341

-- Definitions for the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 6

-- Define the formula for the sum of the first n terms of a geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The equivalent Lean 4 statement
theorem sum_of_first_six_terms_of_geometric_series :
  geometric_series_sum a r n = 4095 / 12288 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_six_terms_of_geometric_series_l2033_203341


namespace NUMINAMATH_GPT_bridge_length_correct_l2033_203312

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 72
noncomputable def crossing_time : ℝ := 12.399008079353651

-- converting train speed from km/hr to m/s
noncomputable def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- total length the train covers to cross the bridge
noncomputable def total_length : ℝ := train_speed_m_per_s * crossing_time

-- length of the bridge
noncomputable def bridge_length : ℝ := total_length - train_length

theorem bridge_length_correct :
  bridge_length = 137.98 :=
by 
  sorry

end NUMINAMATH_GPT_bridge_length_correct_l2033_203312


namespace NUMINAMATH_GPT_find_t_l2033_203376

noncomputable def a_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 2 → a (n + 1) = 3 * a n + 3 ^ n

noncomputable def b_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) : Prop :=
  ∀ n : ℕ, b n = (a (n + 1) + t) / 3^(n + 1)

theorem find_t (a : ℕ → ℤ) (b : ℕ → ℤ) (t : ℤ) :
  a_sequence a →
  b_sequence a b t →
  (∀ n : ℕ, (b (n + 1) - b n) = (b 1 - b 0)) →
  t = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l2033_203376


namespace NUMINAMATH_GPT_number_students_first_class_l2033_203396

theorem number_students_first_class
  (average_first_class : ℝ)
  (average_second_class : ℝ)
  (students_second_class : ℕ)
  (combined_average : ℝ)
  (total_students : ℕ)
  (total_marks_first_class : ℝ)
  (total_marks_second_class : ℝ)
  (total_combined_marks : ℝ)
  (x : ℕ)
  (h1 : average_first_class = 50)
  (h2 : average_second_class = 65)
  (h3 : students_second_class = 40)
  (h4 : combined_average = 59.23076923076923)
  (h5 : total_students = x + 40)
  (h6 : total_marks_first_class = 50 * x)
  (h7 : total_marks_second_class = 65 * 40)
  (h8 : total_combined_marks = 59.23076923076923 * (x + 40))
  (h9 : total_marks_first_class + total_marks_second_class = total_combined_marks) :
  x = 25 :=
sorry

end NUMINAMATH_GPT_number_students_first_class_l2033_203396


namespace NUMINAMATH_GPT_part1_part2_l2033_203307

section
variables (x a m n : ℝ)
-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- a) Prove the solution of the inequality f(x) >= 4 + |x-3| - |x-1| given a=3.
theorem part1 (h_a : a = 3) :
  {x | f x a ≥ 4 + abs (x - 3) - abs (x - 1)} = {x | x ≤ 0} ∪ {x | x ≥ 4} :=
sorry

-- b) Prove that m + 2n >= 2 given f(x) <= 1 + |x-3| with solution set [1, 3] and 1/m + 1/(2n) = a
theorem part2 (h_sol : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + abs (x - 3)) 
  (h_a : 1 / m + 1 / (2 * n) = 2) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  m + 2 * n ≥ 2 :=
sorry
end

end NUMINAMATH_GPT_part1_part2_l2033_203307


namespace NUMINAMATH_GPT_length_of_DF_l2033_203372

theorem length_of_DF
  (D E F P Q: Type)
  (DP: ℝ)
  (EQ: ℝ)
  (h1: DP = 27)
  (h2: EQ = 36)
  (perp: ∀ (u v: Type), u ≠ v):
  ∃ (DF: ℝ), DF = 4 * Real.sqrt 117 :=
by
  sorry

end NUMINAMATH_GPT_length_of_DF_l2033_203372


namespace NUMINAMATH_GPT_minimum_value_of_f_l2033_203329

noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem minimum_value_of_f (x : ℝ) (hx : x > 1) : (∃ y : ℝ, f x = 5 ∧ ∀ y > 1, f y ≥ 5) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2033_203329


namespace NUMINAMATH_GPT_side_length_of_square_l2033_203310

theorem side_length_of_square (length_rect width_rect : ℝ) (h_length : length_rect = 7) (h_width : width_rect = 5) :
  (∃ side_length : ℝ, 4 * side_length = 2 * (length_rect + width_rect) ∧ side_length = 6) :=
by
  use 6
  simp [h_length, h_width]
  sorry

end NUMINAMATH_GPT_side_length_of_square_l2033_203310


namespace NUMINAMATH_GPT_Nicole_cards_l2033_203308

variables (N : ℕ)

-- Conditions from step A
def Cindy_collected (N : ℕ) : ℕ := 2 * N
def Nicole_and_Cindy_combined (N : ℕ) : ℕ := N + Cindy_collected N
def Rex_collected (N : ℕ) : ℕ := (Nicole_and_Cindy_combined N) / 2
def Rex_cards_each (N : ℕ) : ℕ := Rex_collected N / 4

-- Question: How many cards did Nicole collect? Answer: N = 400
theorem Nicole_cards (N : ℕ) (h : Rex_cards_each N = 150) : N = 400 :=
sorry

end NUMINAMATH_GPT_Nicole_cards_l2033_203308


namespace NUMINAMATH_GPT_average_salary_company_l2033_203342

-- Define the conditions
def num_managers : Nat := 15
def num_associates : Nat := 75
def avg_salary_managers : ℤ := 90000
def avg_salary_associates : ℤ := 30000

-- Define the goal to prove
theorem average_salary_company : 
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
  sorry

end NUMINAMATH_GPT_average_salary_company_l2033_203342


namespace NUMINAMATH_GPT_prob_both_selected_l2033_203322

-- Define the probabilities of selection
def prob_selection_x : ℚ := 1 / 5
def prob_selection_y : ℚ := 2 / 3

-- Prove that the probability that both x and y are selected is 2 / 15
theorem prob_both_selected : prob_selection_x * prob_selection_y = 2 / 15 := 
by
  sorry

end NUMINAMATH_GPT_prob_both_selected_l2033_203322


namespace NUMINAMATH_GPT_van_distance_l2033_203365

noncomputable def distance_covered (initial_time new_time speed : ℝ) : ℝ :=
  speed * new_time

theorem van_distance :
  distance_covered 5 (5 * (3 / 2)) 60 = 450 := 
by
  sorry

end NUMINAMATH_GPT_van_distance_l2033_203365


namespace NUMINAMATH_GPT_find_some_number_l2033_203394

def op (x w : ℕ) := (2^x) / (2^w)

theorem find_some_number (n : ℕ) (hn : 0 < n) : (op (op 4 n) n) = 4 → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l2033_203394


namespace NUMINAMATH_GPT_find_n_l2033_203375

theorem find_n (n : ℕ) (h : ∀ x : ℝ, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → 3 * x - 5 = 0) :
  n = 1 :=
sorry

end NUMINAMATH_GPT_find_n_l2033_203375


namespace NUMINAMATH_GPT_roots_of_quadratic_l2033_203345

theorem roots_of_quadratic (a b : ℝ) (h : a ≠ 0) (h1 : a + b = 0) :
  ∀ x, (a * x^2 + b * x = 0) → (x = 0 ∨ x = 1) := 
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l2033_203345


namespace NUMINAMATH_GPT_f_2023_pi_over_3_eq_4_l2033_203313

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.cos x
| (n + 1), x => 4 / (2 - f n x)

theorem f_2023_pi_over_3_eq_4 : f 2023 (Real.pi / 3) = 4 := 
  sorry

end NUMINAMATH_GPT_f_2023_pi_over_3_eq_4_l2033_203313


namespace NUMINAMATH_GPT_find_k_l2033_203374

variables {x k : ℝ}

theorem find_k (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (h2 : k ≠ 0) : k = 8 :=
sorry

end NUMINAMATH_GPT_find_k_l2033_203374


namespace NUMINAMATH_GPT_jessica_final_balance_l2033_203349

variable {original_balance current_balance final_balance withdrawal1 withdrawal2 deposit1 deposit2 : ℝ}

theorem jessica_final_balance:
  (2 / 5) * original_balance = 200 → 
  current_balance = original_balance - 200 → 
  withdrawal1 = (1 / 3) * current_balance → 
  current_balance - withdrawal1 = current_balance - (1 / 3 * current_balance) → 
  deposit1 = (1 / 5) * (current_balance - (1 / 3 * current_balance)) → 
  final_balance = (current_balance - (1 / 3 * current_balance)) + deposit1 → 
  deposit2 / 7 * 3 = final_balance - (current_balance - (1 / 3 * current_balance) + deposit1) → 
  (final_balance + deposit2) = 420 :=
sorry

end NUMINAMATH_GPT_jessica_final_balance_l2033_203349


namespace NUMINAMATH_GPT_remainder_division_123456789012_by_112_l2033_203351

-- Define the conditions
def M : ℕ := 123456789012
def m7 : ℕ := M % 7
def m16 : ℕ := M % 16

-- State the proof problem
theorem remainder_division_123456789012_by_112 : M % 112 = 76 :=
by
  -- Conditions
  have h1 : m7 = 3 := by sorry
  have h2 : m16 = 12 := by sorry
  -- Conclusion
  sorry

end NUMINAMATH_GPT_remainder_division_123456789012_by_112_l2033_203351


namespace NUMINAMATH_GPT_ratio_of_red_to_black_l2033_203377

theorem ratio_of_red_to_black (r b : ℕ) (h_r : r = 26) (h_b : b = 70) :
  r / Nat.gcd r b = 13 ∧ b / Nat.gcd r b = 35 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_red_to_black_l2033_203377


namespace NUMINAMATH_GPT_Craig_walk_distance_l2033_203363

/-- Craig walked some distance from school to David's house and 0.7 miles from David's house to his own house. 
In total, Craig walked 0.9 miles. Prove that the distance Craig walked from school to David's house is 0.2 miles. 
--/
theorem Craig_walk_distance (d_school_David d_David_Craig d_total : ℝ) 
  (h1 : d_David_Craig = 0.7) 
  (h2 : d_total = 0.9) : 
  d_school_David = 0.2 :=
by 
  sorry

end NUMINAMATH_GPT_Craig_walk_distance_l2033_203363


namespace NUMINAMATH_GPT_speed_of_point_C_l2033_203389

theorem speed_of_point_C 
    (a T R L x : ℝ) 
    (h1 : x = L * (a * T) / R - L) 
    (h_eq: (a * T) / (a * T - R) = (L + x) / x) :
    (a * L) / R = x / T :=
by
  sorry

end NUMINAMATH_GPT_speed_of_point_C_l2033_203389


namespace NUMINAMATH_GPT_equivalent_expression_l2033_203367

theorem equivalent_expression (x : ℝ) (hx : x > 0) : (x^2 * x^(1/4))^(1/3) = x^(3/4) := 
  sorry

end NUMINAMATH_GPT_equivalent_expression_l2033_203367


namespace NUMINAMATH_GPT_inequality_am_gm_holds_l2033_203366

theorem inequality_am_gm_holds 
    (a b c : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (hc : c > 0) 
    (h : a^3 + b^3 = c^3) : 
  a^2 + b^2 - c^2 > 6 * (c - a) * (c - b) := 
sorry

end NUMINAMATH_GPT_inequality_am_gm_holds_l2033_203366


namespace NUMINAMATH_GPT_cylinder_curved_surface_area_l2033_203331

theorem cylinder_curved_surface_area {r h : ℝ} (hr: r = 2) (hh: h = 5) :  2 * Real.pi * r * h = 20 * Real.pi :=
by
  rw [hr, hh]
  sorry

end NUMINAMATH_GPT_cylinder_curved_surface_area_l2033_203331


namespace NUMINAMATH_GPT_lunch_cost_total_l2033_203316

theorem lunch_cost_total (x y : ℝ) (h1 : y = 45) (h2 : x = (2 / 3) * y) : 
  x + y + y = 120 := by
  sorry

end NUMINAMATH_GPT_lunch_cost_total_l2033_203316


namespace NUMINAMATH_GPT_inequality_holds_l2033_203340

theorem inequality_holds (x : ℝ) (m : ℝ) :
  (∀ x : ℝ, (x^2 - m * x - 2) / (x^2 - 3 * x + 4) > -1) ↔ (-7 < m ∧ m < 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2033_203340


namespace NUMINAMATH_GPT_age_problem_l2033_203337

-- Definitions from conditions
variables (p q : ℕ) -- ages of p and q as natural numbers
variables (Y : ℕ) -- number of years ago p was half the age of q

-- Main statement
theorem age_problem :
  (p + q = 28) ∧ (p / q = 3 / 4) ∧ (p - Y = (q - Y) / 2) → Y = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l2033_203337


namespace NUMINAMATH_GPT_alchemy_value_l2033_203359

def letter_values : List Int :=
  [3, 2, 1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1,
  0, 1, 2, 3]

def char_value (c : Char) : Int :=
  letter_values.getD ((c.toNat - 'A'.toNat) % 13) 0

def word_value (s : String) : Int :=
  s.toList.map char_value |>.sum

theorem alchemy_value :
  word_value "ALCHEMY" = 8 :=
by
  sorry

end NUMINAMATH_GPT_alchemy_value_l2033_203359


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2033_203327

-- Definitions of conditions
def asymptotes_of_hyperbola (a b x y : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (b * x + a * y = 0) ∨ (b * x - a * y = 0)

def circle_tangent_to_asymptotes (x y a b : ℝ) : Prop :=
  ∀ x1 y1 : ℝ, 
  (x1, y1) = (0, 4) → 
  (Real.sqrt (b^2 + a^2) = 2 * a)

-- Main statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_asymptotes : ∀ (x y : ℝ), asymptotes_of_hyperbola a b x y h_a h_b) 
  (h_tangent : circle_tangent_to_asymptotes 0 4 a b) : 
  ∃ e : ℝ, e = 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2033_203327


namespace NUMINAMATH_GPT_find_a_b_l2033_203348

theorem find_a_b (a b x y : ℝ) (h1 : x = 2) (h2 : y = 4) (h3 : a * x + b * y = 16) (h4 : b * x - a * y = -12) : a = 4 ∧ b = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_b_l2033_203348


namespace NUMINAMATH_GPT_rectangular_prism_sum_l2033_203324

theorem rectangular_prism_sum : 
  let edges := 12
  let vertices := 8
  let faces := 6
  edges + vertices + faces = 26 := by
sorry

end NUMINAMATH_GPT_rectangular_prism_sum_l2033_203324


namespace NUMINAMATH_GPT_congruence_equiv_l2033_203378

theorem congruence_equiv (x : ℤ) (h : 5 * x + 9 ≡ 3 [ZMOD 18]) : 3 * x + 14 ≡ 14 [ZMOD 18] :=
sorry

end NUMINAMATH_GPT_congruence_equiv_l2033_203378


namespace NUMINAMATH_GPT_max_value_of_f_l2033_203371

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.cos x - 3

theorem max_value_of_f : ∀ x : ℝ, f x ≤ -1/2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2033_203371


namespace NUMINAMATH_GPT_infinite_series_sum_l2033_203328

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 1998)^n) = (3992004 / 3988009) :=
by sorry

end NUMINAMATH_GPT_infinite_series_sum_l2033_203328


namespace NUMINAMATH_GPT_value_of_y_l2033_203306

theorem value_of_y 
  (x y : ℤ) 
  (h1 : x - y = 10) 
  (h2 : x + y = 8) 
  : y = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l2033_203306


namespace NUMINAMATH_GPT_two_digit_numbers_non_repeating_l2033_203395

-- The set of available digits is given as 0, 1, 2, 3, 4
def digits : List ℕ := [0, 1, 2, 3, 4]

-- Ensure the tens place digits are subset of 1, 2, 3, 4 (exclude 0)
def valid_tens : List ℕ := [1, 2, 3, 4]

theorem two_digit_numbers_non_repeating :
  let num_tens := valid_tens.length
  let num_units := (digits.length - 1)
  num_tens * num_units = 16 :=
by
  -- Observe num_tens = 4, since valid_tens = [1, 2, 3, 4]
  -- Observe num_units = 4, since digits.length = 5 and we exclude the tens place digit
  sorry

end NUMINAMATH_GPT_two_digit_numbers_non_repeating_l2033_203395


namespace NUMINAMATH_GPT_arithmetic_progression_K_l2033_203343

theorem arithmetic_progression_K (K : ℕ) : 
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_K_l2033_203343


namespace NUMINAMATH_GPT_sum_remainder_zero_l2033_203347

theorem sum_remainder_zero
  (a b c : ℕ)
  (h₁ : a % 53 = 31)
  (h₂ : b % 53 = 15)
  (h₃ : c % 53 = 7) :
  (a + b + c) % 53 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_zero_l2033_203347


namespace NUMINAMATH_GPT_volume_of_sphere_eq_4_sqrt3_pi_l2033_203311

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

theorem volume_of_sphere_eq_4_sqrt3_pi
  (r : ℝ) (h : 4 * Real.pi * r ^ 2 = 2 * Real.sqrt 3 * Real.pi * (2 * r)) :
  volume_of_sphere r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_eq_4_sqrt3_pi_l2033_203311


namespace NUMINAMATH_GPT_typist_speeds_l2033_203325

noncomputable def num_pages : ℕ := 72
noncomputable def ratio : ℚ := 6 / 5
noncomputable def time_difference : ℚ := 1.5

theorem typist_speeds :
  ∃ (x y : ℚ), (x = 9.6 ∧ y = 8) ∧ 
                (num_pages / x - num_pages / y = time_difference) ∧
                (x / y = ratio) :=
by
  -- Let's skip the proof for now
  sorry

end NUMINAMATH_GPT_typist_speeds_l2033_203325


namespace NUMINAMATH_GPT_stefan_more_vail_l2033_203368

/-- Aiguo had 20 seashells --/
def a : ℕ := 20

/-- Vail had 5 less seashells than Aiguo --/
def v : ℕ := a - 5

/-- The total number of seashells of Stefan, Vail, and Aiguo is 66 --/
def total_seashells (s v a : ℕ) : Prop := s + v + a = 66

theorem stefan_more_vail (s v a : ℕ)
  (h_a : a = 20)
  (h_v : v = a - 5)
  (h_total : total_seashells s v a) :
  s - v = 16 :=
by {
  -- proofs would go here
  sorry
}

end NUMINAMATH_GPT_stefan_more_vail_l2033_203368


namespace NUMINAMATH_GPT_garbage_collection_l2033_203315

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end NUMINAMATH_GPT_garbage_collection_l2033_203315


namespace NUMINAMATH_GPT_marked_price_correct_l2033_203386

noncomputable def marked_price (cost_price : ℝ) (profit_margin : ℝ) (selling_percentage : ℝ) : ℝ :=
  (cost_price * (1 + profit_margin)) / selling_percentage

theorem marked_price_correct :
  marked_price 1360 0.15 0.8 = 1955 :=
by
  sorry

end NUMINAMATH_GPT_marked_price_correct_l2033_203386


namespace NUMINAMATH_GPT_intersection_of_sets_l2033_203352

def setA : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x : ℝ | 2 < x }

theorem intersection_of_sets : setA ∩ setB = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2033_203352


namespace NUMINAMATH_GPT_perpendicular_distance_H_to_plane_EFG_l2033_203334

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def E : Point3D := ⟨5, 0, 0⟩
def F : Point3D := ⟨0, 3, 0⟩
def G : Point3D := ⟨0, 0, 4⟩
def H : Point3D := ⟨0, 0, 0⟩

def distancePointToPlane (H E F G : Point3D) : ℝ := sorry

theorem perpendicular_distance_H_to_plane_EFG :
  distancePointToPlane H E F G = 1.8 := sorry

end NUMINAMATH_GPT_perpendicular_distance_H_to_plane_EFG_l2033_203334


namespace NUMINAMATH_GPT_eval_floor_abs_neg_45_7_l2033_203373

theorem eval_floor_abs_neg_45_7 : ∀ x : ℝ, x = -45.7 → (⌊|x|⌋ = 45) := by
  intros x hx
  sorry

end NUMINAMATH_GPT_eval_floor_abs_neg_45_7_l2033_203373


namespace NUMINAMATH_GPT_sum_in_base4_l2033_203309

def dec_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert (n : ℕ) (acc : ℕ) (power : ℕ) :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * power) (power * 10)
  convert n 0 1

theorem sum_in_base4 : dec_to_base4 (234 + 78) = 13020 :=
  sorry

end NUMINAMATH_GPT_sum_in_base4_l2033_203309


namespace NUMINAMATH_GPT_total_bill_l2033_203398

theorem total_bill (total_people : ℕ) (children : ℕ) (adult_cost : ℕ) (child_cost : ℕ)
  (h : total_people = 201) (hc : children = 161) (ha : adult_cost = 8) (hc_cost : child_cost = 4) :
  (201 - 161) * 8 + 161 * 4 = 964 :=
by
  rw [←h, ←hc, ←ha, ←hc_cost]
  sorry

end NUMINAMATH_GPT_total_bill_l2033_203398


namespace NUMINAMATH_GPT_profit_with_discount_l2033_203370

theorem profit_with_discount (CP SP_with_discount SP_no_discount : ℝ) (discount profit_no_discount : ℝ) (H1 : discount = 0.1) (H2 : profit_no_discount = 0.3889) (H3 : SP_no_discount = CP * (1 + profit_no_discount)) (H4 : SP_with_discount = SP_no_discount * (1 - discount)) : (SP_with_discount - CP) / CP * 100 = 25 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_profit_with_discount_l2033_203370


namespace NUMINAMATH_GPT_zoe_total_songs_l2033_203317

def initial_songs : ℕ := 15
def deleted_songs : ℕ := 8
def added_songs : ℕ := 50

theorem zoe_total_songs : initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end NUMINAMATH_GPT_zoe_total_songs_l2033_203317


namespace NUMINAMATH_GPT_janets_total_pockets_l2033_203319

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end NUMINAMATH_GPT_janets_total_pockets_l2033_203319


namespace NUMINAMATH_GPT_r_s_t_u_bounds_l2033_203354

theorem r_s_t_u_bounds (r s t u : ℝ) 
  (H1: 5 * r + 4 * s + 3 * t + 6 * u = 100)
  (H2: r ≥ s)
  (H3: s ≥ t)
  (H4: t ≥ u)
  (H5: u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := 
sorry

end NUMINAMATH_GPT_r_s_t_u_bounds_l2033_203354


namespace NUMINAMATH_GPT_gcd_n_cube_plus_m_square_l2033_203391

theorem gcd_n_cube_plus_m_square (n m : ℤ) (h : n > 2^3) : Int.gcd (n^3 + m^2) (n + 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_n_cube_plus_m_square_l2033_203391


namespace NUMINAMATH_GPT_problem1_problem2_l2033_203387

variable {α : ℝ}

-- Given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 3

-- Proof statements to be shown
theorem problem1 (h : tan_alpha α) : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by sorry

theorem problem2 (h : tan_alpha α) : Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2033_203387


namespace NUMINAMATH_GPT_sin_cos_15_degree_l2033_203385

theorem sin_cos_15_degree :
  (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_15_degree_l2033_203385


namespace NUMINAMATH_GPT_sum_of_200_terms_l2033_203390

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (a1 a200 : ℝ)

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def collinearity_condition (a1 a200 : ℝ) : Prop :=
a1 + a200 = 1

-- Proof statement
theorem sum_of_200_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 a200 : ℝ) 
  (h_seq : arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms S a)
  (h_collinear : collinearity_condition a1 a200) : 
  S 200 = 100 := 
sorry

end NUMINAMATH_GPT_sum_of_200_terms_l2033_203390


namespace NUMINAMATH_GPT_find_y_payment_l2033_203323

-- Defining the conditions
def total_payment : ℝ := 700
def x_payment (y_payment : ℝ) : ℝ := 1.2 * y_payment

-- The theorem we want to prove
theorem find_y_payment (y_payment : ℝ) (h1 : y_payment + x_payment y_payment = total_payment) :
  y_payment = 318.18 := 
sorry

end NUMINAMATH_GPT_find_y_payment_l2033_203323


namespace NUMINAMATH_GPT_exists_integer_roots_l2033_203358

theorem exists_integer_roots : 
  ∃ (a b c d e f : ℤ), ∃ r1 r2 r3 r4 r5 r6 : ℤ,
  (r1 + a) * (r2 ^ 2 + b * r2 + c) * (r3 ^ 3 + d * r3 ^ 2 + e * r3 + f) = 0 ∧
  (r4 + a) * (r5 ^ 2 + b * r5 + c) * (r6 ^ 3 + d * r6 ^ 2 + e * r6 + f) = 0 :=
  sorry

end NUMINAMATH_GPT_exists_integer_roots_l2033_203358


namespace NUMINAMATH_GPT_oranges_weigh_4_ounces_each_l2033_203301

def apple_weight : ℕ := 4
def max_bag_capacity : ℕ := 49
def num_bags : ℕ := 3
def total_weight : ℕ := num_bags * max_bag_capacity
def total_apple_weight : ℕ := 84
def num_apples : ℕ := total_apple_weight / apple_weight
def num_oranges : ℕ := num_apples
def total_orange_weight : ℕ := total_apple_weight
def weight_per_orange : ℕ := total_orange_weight / num_oranges

theorem oranges_weigh_4_ounces_each :
  weight_per_orange = 4 := by
  sorry

end NUMINAMATH_GPT_oranges_weigh_4_ounces_each_l2033_203301


namespace NUMINAMATH_GPT_triangle_area_on_ellipse_l2033_203397

def onEllipse (p : ℝ × ℝ) : Prop := (p.1)^2 + 4 * (p.2)^2 = 4

def isCentroid (C : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  C = ((A.1 + B.1) / 3, (A.2 + B.2) / 3)

theorem triangle_area_on_ellipse
  (A B C : ℝ × ℝ)
  (h₁ : A ≠ B)
  (h₂ : B ≠ C)
  (h₃ : C ≠ A)
  (h₄ : onEllipse A)
  (h₅ : onEllipse B)
  (h₆ : onEllipse C)
  (h₇ : isCentroid C A B)
  (h₈ : C = (0, 0))  : 
  1 / 2 * (A.1 - B.1) * (B.2 - A.2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_on_ellipse_l2033_203397


namespace NUMINAMATH_GPT_intersection_of_sets_l2033_203304

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3}) (hB : B = { x | x < 3 ∧ x ∈ Set.univ }) :
  A ∩ B = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2033_203304


namespace NUMINAMATH_GPT_students_can_do_both_l2033_203399

variable (total_students swimmers gymnasts neither : ℕ)

theorem students_can_do_both (h1 : total_students = 60)
                             (h2 : swimmers = 27)
                             (h3 : gymnasts = 28)
                             (h4 : neither = 15) : 
                             total_students - (total_students - swimmers + total_students - gymnasts - neither) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_students_can_do_both_l2033_203399


namespace NUMINAMATH_GPT_max_value_l2033_203384

-- Definition of the ellipse and the goal function
def ellipse (x y : ℝ) := 2 * x^2 + 3 * y^2 = 12

-- Definition of the function we want to maximize
def func (x y : ℝ) := x + 2 * y

-- The theorem to prove that the maximum value of x + 2y on the ellipse is √22
theorem max_value (x y : ℝ) (h : ellipse x y) : ∃ θ : ℝ, func x y ≤ Real.sqrt 22 :=
by
  sorry

end NUMINAMATH_GPT_max_value_l2033_203384


namespace NUMINAMATH_GPT_samantha_original_cans_l2033_203320

theorem samantha_original_cans : 
  ∀ (cans_per_classroom : ℚ),
  (cans_per_classroom = (50 - 38) / 5) →
  (50 / cans_per_classroom) = 21 := 
by
  sorry

end NUMINAMATH_GPT_samantha_original_cans_l2033_203320


namespace NUMINAMATH_GPT_storks_more_than_birds_l2033_203362

theorem storks_more_than_birds 
  (initial_birds : ℕ) 
  (joined_storks : ℕ) 
  (joined_birds : ℕ) 
  (h_init_birds : initial_birds = 3) 
  (h_joined_storks : joined_storks = 6) 
  (h_joined_birds : joined_birds = 2) : 
  (joined_storks - (initial_birds + joined_birds)) = 1 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_storks_more_than_birds_l2033_203362


namespace NUMINAMATH_GPT_neg_of_exists_l2033_203380

theorem neg_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x: ℝ, x ≥ 3 ∧ x^2 - 2 * x + 3 < 0) ↔ (∀ x: ℝ, x ≥ 3 → x^2 - 2 * x + 3 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_of_exists_l2033_203380


namespace NUMINAMATH_GPT_firetruck_reachable_area_l2033_203344

theorem firetruck_reachable_area :
  let speed_highway := 50
  let speed_prairie := 14
  let travel_time := 0.1
  let area := 16800 / 961
  ∀ (x r : ℝ),
    (x / speed_highway + r / speed_prairie = travel_time) →
    (0 ≤ x ∧ 0 ≤ r) →
    ∃ m n : ℕ, gcd m n = 1 ∧
    m = 16800 ∧ n = 961 ∧
    m + n = 16800 + 961 := by
  sorry

end NUMINAMATH_GPT_firetruck_reachable_area_l2033_203344
