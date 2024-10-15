import Mathlib

namespace NUMINAMATH_GPT_ratio_of_books_sold_l49_4952

theorem ratio_of_books_sold
  (T W R : ℕ)
  (hT : T = 7)
  (hW : W = 3 * T)
  (hTotal : T + W + R = 91) :
  R / W = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_books_sold_l49_4952


namespace NUMINAMATH_GPT_total_amount_l49_4980

theorem total_amount (A B C : ℤ) (S : ℤ) (h_ratio : 100 * B = 45 * A ∧ 100 * C = 30 * A) (h_B : B = 6300) : S = 24500 := by
  sorry

end NUMINAMATH_GPT_total_amount_l49_4980


namespace NUMINAMATH_GPT_eval_36_pow_five_over_two_l49_4970

theorem eval_36_pow_five_over_two : (36 : ℝ)^(5/2) = 7776 := by
  sorry

end NUMINAMATH_GPT_eval_36_pow_five_over_two_l49_4970


namespace NUMINAMATH_GPT_michelle_scored_30_l49_4982

-- Define the total team points
def team_points : ℕ := 72

-- Define the number of other players
def num_other_players : ℕ := 7

-- Define the average points scored by the other players
def avg_points_other_players : ℕ := 6

-- Calculate the total points scored by the other players
def total_points_other_players : ℕ := num_other_players * avg_points_other_players

-- Define the points scored by Michelle
def michelle_points : ℕ := team_points - total_points_other_players

-- Prove that the points scored by Michelle is 30
theorem michelle_scored_30 : michelle_points = 30 :=
by
  -- Here would be the proof, but we skip it with sorry.
  sorry

end NUMINAMATH_GPT_michelle_scored_30_l49_4982


namespace NUMINAMATH_GPT_cost_of_each_sale_puppy_l49_4974

-- Conditions
def total_cost (total: ℚ) : Prop := total = 800
def non_sale_puppy_cost (cost: ℚ) : Prop := cost = 175
def num_puppies (num: ℕ) : Prop := num = 5

-- Question to Prove
theorem cost_of_each_sale_puppy (total cost : ℚ) (num: ℕ):
  total_cost total →
  non_sale_puppy_cost cost →
  num_puppies num →
  (total - 2 * cost) / (num - 2) = 150 := 
sorry

end NUMINAMATH_GPT_cost_of_each_sale_puppy_l49_4974


namespace NUMINAMATH_GPT_square_area_l49_4918

theorem square_area (x : ℝ) (side1 side2 : ℝ) 
  (h_side1 : side1 = 6 * x - 27) 
  (h_side2 : side2 = 30 - 2 * x) 
  (h_equiv : side1 = side2) : 
  (side1 * side1 = 248.0625) := 
by
  sorry

end NUMINAMATH_GPT_square_area_l49_4918


namespace NUMINAMATH_GPT_determine_point_T_l49_4971

noncomputable def point : Type := ℝ × ℝ

def is_square (O P Q R : point) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧
  Q.1 = 3 ∧ Q.2 = 3 ∧
  P.1 = 3 ∧ P.2 = 0 ∧
  R.1 = 0 ∧ R.2 = 3

def twice_area_square_eq_area_triangle (O P Q T : point) : Prop :=
  2 * (3 * 3) = abs ((P.1 * Q.2 + Q.1 * T.2 + T.1 * P.2 - P.2 * Q.1 - Q.2 * T.1 - T.2 * P.1) / 2)

theorem determine_point_T (O P Q R T : point) (h1 : is_square O P Q R) : 
  twice_area_square_eq_area_triangle O P Q T ↔ T = (3, 12) :=
sorry

end NUMINAMATH_GPT_determine_point_T_l49_4971


namespace NUMINAMATH_GPT_shortest_distance_midpoint_parabola_chord_l49_4961

theorem shortest_distance_midpoint_parabola_chord
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2)
  (hB : B.1 ^ 2 = 4 * B.2)
  (cord_length : dist A B = 6)
  : dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (0, 0) = 2 :=
sorry

end NUMINAMATH_GPT_shortest_distance_midpoint_parabola_chord_l49_4961


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l49_4911

noncomputable def a := Real.sqrt 0.5
noncomputable def b := Real.sqrt 0.3
noncomputable def c := Real.log 0.2 / Real.log 0.3

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l49_4911


namespace NUMINAMATH_GPT_lily_pad_cover_entire_lake_l49_4969

-- Definitions per the conditions
def doublesInSizeEveryDay (P : ℕ → ℝ) : Prop :=
  ∀ n, P (n + 1) = 2 * P n

-- The initial state that it takes 36 days to cover the lake
def coversEntireLakeIn36Days (P : ℕ → ℝ) (L : ℝ) : Prop :=
  P 36 = L

-- The main theorem to prove
theorem lily_pad_cover_entire_lake (P : ℕ → ℝ) (L : ℝ) (h1 : doublesInSizeEveryDay P) (h2 : coversEntireLakeIn36Days P L) :
  ∃ n, n = 36 := 
by
  sorry

end NUMINAMATH_GPT_lily_pad_cover_entire_lake_l49_4969


namespace NUMINAMATH_GPT_trig_product_identity_l49_4963

theorem trig_product_identity :
  (1 + Real.sin (Real.pi / 12)) * (1 + Real.sin (5 * Real.pi / 12)) *
  (1 + Real.sin (7 * Real.pi / 12)) * (1 + Real.sin (11 * Real.pi / 12)) =
  (1 + Real.sin (Real.pi / 12))^2 * (1 + Real.sin (5 * Real.pi / 12))^2 :=
by
  sorry

end NUMINAMATH_GPT_trig_product_identity_l49_4963


namespace NUMINAMATH_GPT_erica_pie_percentage_l49_4962

theorem erica_pie_percentage (a c : ℚ) (ha : a = 1/5) (hc : c = 3/4) : 
  (a + c) * 100 = 95 := 
sorry

end NUMINAMATH_GPT_erica_pie_percentage_l49_4962


namespace NUMINAMATH_GPT_bucket_volume_l49_4934

theorem bucket_volume :
  ∃ (V : ℝ), -- The total volume of the bucket
    (∀ (rate_A rate_B rate_combined : ℝ),
      rate_A = 3 ∧ 
      rate_B = V / 60 ∧ 
      rate_combined = V / 10 ∧ 
      rate_A + rate_B = rate_combined) →
    V = 36 :=
by
  sorry

end NUMINAMATH_GPT_bucket_volume_l49_4934


namespace NUMINAMATH_GPT_a_100_correct_l49_4917

variable (a_n : ℕ → ℕ) (S₉ : ℕ) (a₁₀ : ℕ)

def is_arth_seq (a_n : ℕ → ℕ) := ∃ a d, ∀ n, a_n n = a + n * d

noncomputable def a_100 (a₅ d : ℕ) : ℕ := a₅ + 95 * d

theorem a_100_correct
  (h1 : ∃ S₉, 9 * a_n 4 = S₉)
  (h2 : a_n 9 = 8)
  (h3 : is_arth_seq a_n) :
  a_100 (a_n 4) 1 = 98 :=
by
  sorry

end NUMINAMATH_GPT_a_100_correct_l49_4917


namespace NUMINAMATH_GPT_savings_calculation_l49_4983

noncomputable def weekly_rate_peak : ℕ := 10
noncomputable def weekly_rate_non_peak : ℕ := 8
noncomputable def monthly_rate_peak : ℕ := 40
noncomputable def monthly_rate_non_peak : ℕ := 35
noncomputable def non_peak_duration_weeks : ℝ := 17.33
noncomputable def peak_duration_weeks : ℝ := 52 - non_peak_duration_weeks
noncomputable def non_peak_duration_months : ℕ := 4
noncomputable def peak_duration_months : ℕ := 12 - non_peak_duration_months

noncomputable def total_weekly_cost := (non_peak_duration_weeks * weekly_rate_non_peak) 
                                     + (peak_duration_weeks * weekly_rate_peak)

noncomputable def total_monthly_cost := (non_peak_duration_months * monthly_rate_non_peak) 
                                      + (peak_duration_months * monthly_rate_peak)

noncomputable def savings := total_weekly_cost - total_monthly_cost

theorem savings_calculation 
  : savings = 25.34 := by
  sorry

end NUMINAMATH_GPT_savings_calculation_l49_4983


namespace NUMINAMATH_GPT_perpendicular_vectors_m_solution_l49_4975

theorem perpendicular_vectors_m_solution (m : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = -2 := by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_solution_l49_4975


namespace NUMINAMATH_GPT_douglas_won_percentage_l49_4992

theorem douglas_won_percentage (p_X p_Y : ℝ) (r : ℝ) (V : ℝ) (h1 : p_X = 0.76) (h2 : p_Y = 0.4000000000000002) (h3 : r = 2) :
  (1.52 * V + 0.4000000000000002 * V) / (2 * V + V) * 100 = 64 := by
  sorry

end NUMINAMATH_GPT_douglas_won_percentage_l49_4992


namespace NUMINAMATH_GPT_ratio_is_two_l49_4990

noncomputable def ratio_of_altitude_to_base (area base : ℕ) : ℕ :=
  have h : ℕ := area / base
  h / base

theorem ratio_is_two (area base : ℕ) (h : ℕ)  (h_area : area = 288) (h_base : base = 12) (h_altitude : h = area / base) : ratio_of_altitude_to_base area base = 2 :=
  by
    sorry 

end NUMINAMATH_GPT_ratio_is_two_l49_4990


namespace NUMINAMATH_GPT_max_ratio_xy_l49_4957

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem max_ratio_xy (x y : ℕ) (hx : two_digit x) (hy : two_digit y) (hmean : (x + y) / 2 = 60) : x / y ≤ 33 / 7 :=
by
  sorry

end NUMINAMATH_GPT_max_ratio_xy_l49_4957


namespace NUMINAMATH_GPT_three_legged_extraterrestrials_l49_4931

-- Define the conditions
variables (x y : ℕ)

-- Total number of heads
def heads_equation := x + y = 300

-- Total number of legs
def legs_equation := 3 * x + 4 * y = 846

theorem three_legged_extraterrestrials : heads_equation x y ∧ legs_equation x y → x = 246 :=
by
  sorry

end NUMINAMATH_GPT_three_legged_extraterrestrials_l49_4931


namespace NUMINAMATH_GPT_tyler_bird_pairs_l49_4914

theorem tyler_bird_pairs (n_species : ℕ) (pairs_per_species : ℕ) (total_pairs : ℕ)
  (h1 : n_species = 29)
  (h2 : pairs_per_species = 7)
  (h3 : total_pairs = n_species * pairs_per_species) : total_pairs = 203 :=
by
  sorry

end NUMINAMATH_GPT_tyler_bird_pairs_l49_4914


namespace NUMINAMATH_GPT_even_function_derivative_at_zero_l49_4954

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_diff : Differentiable ℝ f)

theorem even_function_derivative_at_zero : deriv f 0 = 0 :=
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_even_function_derivative_at_zero_l49_4954


namespace NUMINAMATH_GPT_find_quadrant_372_degrees_l49_4933

theorem find_quadrant_372_degrees : 
  ∃ q : ℕ, q = 1 ↔ (372 % 360 = 12 ∧ (0 ≤ 12 ∧ 12 < 90)) :=
by
  sorry

end NUMINAMATH_GPT_find_quadrant_372_degrees_l49_4933


namespace NUMINAMATH_GPT_distance_between_points_l49_4901

theorem distance_between_points (x y : ℝ) (h : x + y = 10 / 3) : 
  4 * (x + y) = 40 / 3 :=
sorry

end NUMINAMATH_GPT_distance_between_points_l49_4901


namespace NUMINAMATH_GPT_A_can_complete_work_in_28_days_l49_4998
noncomputable def work_days_for_A (x : ℕ) (h : 4 / x = 1 / 21) : ℕ :=
  x / 3

theorem A_can_complete_work_in_28_days (x : ℕ) (h : 4 / x = 1 / 21) :
  work_days_for_A x h = 28 :=
  sorry

end NUMINAMATH_GPT_A_can_complete_work_in_28_days_l49_4998


namespace NUMINAMATH_GPT_B_is_345_complement_U_A_inter_B_is_3_l49_4965

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set ℕ := {2, 4, 5}

-- Define set B as given in the conditions
def B : Set ℕ := {x ∈ U | 2 < x ∧ x < 6}

-- Prove that B is {3, 4, 5}
theorem B_is_345 : B = {3, 4, 5} := by
  sorry

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := U \ A

-- Prove the intersection of the complement of A and B is {3}
theorem complement_U_A_inter_B_is_3 : (complement_U_A ∩ B) = {3} := by
  sorry

end NUMINAMATH_GPT_B_is_345_complement_U_A_inter_B_is_3_l49_4965


namespace NUMINAMATH_GPT_intersection_M_N_l49_4981

def M (x : ℝ) : Prop := -2 < x ∧ x < 2
def N (x : ℝ) : Prop := |x - 1| ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l49_4981


namespace NUMINAMATH_GPT_area_of_triangle_l49_4940

theorem area_of_triangle (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) 
  : (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_l49_4940


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequality_l49_4928

variables (a b : ℝ)

theorem necessary_but_not_sufficient_for_inequality (h : a ≠ b) (hab_pos : a * b > 0) :
  (b/a + a/b > 2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_inequality_l49_4928


namespace NUMINAMATH_GPT_no_equilateral_triangle_OAB_exists_l49_4909

theorem no_equilateral_triangle_OAB_exists :
  ∀ (A B : ℝ × ℝ), 
  ((∃ a : ℝ, A = (a, (3 / 2) ^ a)) ∧ B.1 > 0 ∧ B.2 = 0) → 
  ¬ (∃ k : ℝ, k = (A.2 / A.1) ∧ k > (3 ^ (1 / 2)) / 3) := 
by 
  intro A B h
  sorry

end NUMINAMATH_GPT_no_equilateral_triangle_OAB_exists_l49_4909


namespace NUMINAMATH_GPT_simplify_expression_l49_4947

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l49_4947


namespace NUMINAMATH_GPT_space_mission_contribution_l49_4968

theorem space_mission_contribution 
  (mission_cost_million : ℕ := 30000) 
  (combined_population_million : ℕ := 350) : 
  mission_cost_million / combined_population_million = 86 := by
  sorry

end NUMINAMATH_GPT_space_mission_contribution_l49_4968


namespace NUMINAMATH_GPT_min_value_l49_4916

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 1/(a-1) + 4/(b-1) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l49_4916


namespace NUMINAMATH_GPT_percentage_of_students_in_grade_8_combined_l49_4915

theorem percentage_of_students_in_grade_8_combined (parkwood_students maplewood_students : ℕ)
  (parkwood_percentages maplewood_percentages : ℕ → ℕ) 
  (H_parkwood : parkwood_students = 150)
  (H_maplewood : maplewood_students = 120)
  (H_parkwood_percent : parkwood_percentages 8 = 18)
  (H_maplewood_percent : maplewood_percentages 8 = 25):
  (57 / 270) * 100 = 21.11 := 
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_percentage_of_students_in_grade_8_combined_l49_4915


namespace NUMINAMATH_GPT_arctan_addition_formula_l49_4921

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end NUMINAMATH_GPT_arctan_addition_formula_l49_4921


namespace NUMINAMATH_GPT_negative_number_zero_exponent_l49_4924

theorem negative_number_zero_exponent (a : ℤ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end NUMINAMATH_GPT_negative_number_zero_exponent_l49_4924


namespace NUMINAMATH_GPT_transformed_curve_eq_l49_4936

-- Define the original ellipse curve
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Prove the transformed curve satisfies x'^2 + y'^2 = 4
theorem transformed_curve_eq :
  ∀ (x y x' y' : ℝ), ellipse x y → transform x y x' y' → (x'^2 + y'^2 = 4) :=
by
  intros x y x' y' h_ellipse h_transform
  simp [ellipse, transform] at *
  sorry

end NUMINAMATH_GPT_transformed_curve_eq_l49_4936


namespace NUMINAMATH_GPT_emily_orange_count_l49_4904

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end NUMINAMATH_GPT_emily_orange_count_l49_4904


namespace NUMINAMATH_GPT_number_of_dimes_l49_4942

theorem number_of_dimes (x : ℕ) (h1 : 10 * x + 25 * x + 50 * x = 2040) : x = 24 :=
by {
  -- The proof will go here if you need to fill it out.
  sorry
}

end NUMINAMATH_GPT_number_of_dimes_l49_4942


namespace NUMINAMATH_GPT_man_speed_is_correct_l49_4907

noncomputable def speed_of_man (train_length : ℝ) (train_speed : ℝ) (cross_time : ℝ) : ℝ :=
  let train_speed_m_s := train_speed * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let man_speed_m_s := relative_speed - train_speed_m_s
  man_speed_m_s * (3600 / 1000)

theorem man_speed_is_correct :
  speed_of_man 210 25 28 = 2 := by
  sorry

end NUMINAMATH_GPT_man_speed_is_correct_l49_4907


namespace NUMINAMATH_GPT_number_of_male_students_drawn_l49_4906

theorem number_of_male_students_drawn (total_students : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) (sample_size : ℕ)
    (H1 : total_students = 350)
    (H2 : total_male_students = 70)
    (H3 : total_female_students = 280)
    (H4 : sample_size = 50) :
    total_male_students * sample_size / total_students = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_male_students_drawn_l49_4906


namespace NUMINAMATH_GPT_cost_of_fencing_l49_4927

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (C : ℝ) (cost : ℝ) : 
  d = 22 → rate = 3 → C = Real.pi * d → cost = C * rate → cost = 207 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l49_4927


namespace NUMINAMATH_GPT_sin_cos_identity_l49_4999

theorem sin_cos_identity (α : ℝ) (hα_cos : Real.cos α = 3/5) (hα_sin : Real.sin α = 4/5) : Real.sin α + 2 * Real.cos α = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l49_4999


namespace NUMINAMATH_GPT_problem_statement_l49_4935

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem problem_statement : ((U \ A) ∪ (U \ B)) = {0, 1, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_problem_statement_l49_4935


namespace NUMINAMATH_GPT_cost_of_child_ticket_is_4_l49_4950

def cost_of_child_ticket (cost_adult cost_total tickets_sold tickets_child receipts_total : ℕ) : ℕ :=
  let tickets_adult := tickets_sold - tickets_child
  let receipts_adult := tickets_adult * cost_adult
  let receipts_child := receipts_total - receipts_adult
  receipts_child / tickets_child

theorem cost_of_child_ticket_is_4 (cost_adult : ℕ) (cost_total : ℕ)
  (tickets_sold : ℕ) (tickets_child : ℕ) (receipts_total : ℕ) :
  cost_of_child_ticket 12 4 130 90 840 = 4 := by
  sorry

end NUMINAMATH_GPT_cost_of_child_ticket_is_4_l49_4950


namespace NUMINAMATH_GPT_equipment_value_decrease_l49_4912

theorem equipment_value_decrease (a : ℝ) (b : ℝ) (n : ℕ) :
  (a * (1 - b / 100)^n) = a * (1 - b/100)^n :=
sorry

end NUMINAMATH_GPT_equipment_value_decrease_l49_4912


namespace NUMINAMATH_GPT_muffins_sugar_l49_4973

theorem muffins_sugar (cups_muffins_ratio : 24 * 3 = 72 * s / 9) : s = 9 := by
  sorry

end NUMINAMATH_GPT_muffins_sugar_l49_4973


namespace NUMINAMATH_GPT_sequence_remainder_4_l49_4978

def sequence_of_numbers (n : ℕ) : ℕ :=
  7 * n + 4

theorem sequence_remainder_4 (n : ℕ) : (sequence_of_numbers n) % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_sequence_remainder_4_l49_4978


namespace NUMINAMATH_GPT_cost_price_of_apple_l49_4903

variable (CP SP: ℝ)
variable (loss: ℝ)
variable (h1: SP = 18)
variable (h2: loss = CP / 6)
variable (h3: SP = CP - loss)

theorem cost_price_of_apple : CP = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_apple_l49_4903


namespace NUMINAMATH_GPT_domain_of_function_l49_4995

-- Define the setting and the constants involved
variables {f : ℝ → ℝ}
variable {c : ℝ}

-- The statement about the function's domain
theorem domain_of_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≤ 0 ∧ x ≠ -c) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l49_4995


namespace NUMINAMATH_GPT_intersection_of_A_B_C_l49_4941

-- Define the sets A, B, and C as given conditions:
def A : Set ℕ := { x | ∃ n : ℕ, x = 2 * n }
def B : Set ℕ := { x | ∃ n : ℕ, x = 3 * n }
def C : Set ℕ := { x | ∃ n : ℕ, x = n ^ 2 }

-- Prove that A ∩ B ∩ C = { x | ∃ n : ℕ, x = 36 * n ^ 2 }
theorem intersection_of_A_B_C :
  (A ∩ B ∩ C) = { x | ∃ n : ℕ, x = 36 * n ^ 2 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_B_C_l49_4941


namespace NUMINAMATH_GPT_union_of_sets_l49_4966

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the proof problem
theorem union_of_sets : A ∪ B = {x | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l49_4966


namespace NUMINAMATH_GPT_two_digit_number_eq_27_l49_4987

theorem two_digit_number_eq_27 (A : ℕ) (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
    (h : A = 10 * x + y) (hcond : A = 3 * (x + y)) : A = 27 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_eq_27_l49_4987


namespace NUMINAMATH_GPT_train_length_correct_l49_4923

noncomputable def train_length (time : ℝ) (platform_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time
  total_distance - platform_length

theorem train_length_correct :
  train_length 17.998560115190784 200 90 = 249.9640028797696 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l49_4923


namespace NUMINAMATH_GPT_fg_value_l49_4960

def g (x : ℕ) : ℕ := 4 * x + 10
def f (x : ℕ) : ℕ := 6 * x - 12

theorem fg_value : f (g 10) = 288 := by
  sorry

end NUMINAMATH_GPT_fg_value_l49_4960


namespace NUMINAMATH_GPT_sum_youngest_oldest_l49_4993

-- Define the ages of the cousins
variables (a1 a2 a3 a4 : ℕ)

-- Conditions given in the problem
def mean_age (a1 a2 a3 a4 : ℕ) : Prop := (a1 + a2 + a3 + a4) / 4 = 8
def median_age (a2 a3 : ℕ) : Prop := (a2 + a3) / 2 = 5

-- Main theorem statement to be proved
theorem sum_youngest_oldest (h_mean : mean_age a1 a2 a3 a4) (h_median : median_age a2 a3) :
  a1 + a4 = 22 :=
sorry

end NUMINAMATH_GPT_sum_youngest_oldest_l49_4993


namespace NUMINAMATH_GPT_transistor_length_scientific_notation_l49_4937

theorem transistor_length_scientific_notation :
  0.000000006 = 6 * 10^(-9) := 
sorry

end NUMINAMATH_GPT_transistor_length_scientific_notation_l49_4937


namespace NUMINAMATH_GPT_simon_age_is_10_l49_4951

def alvin_age : ℕ := 30

def simon_age (alvin_age : ℕ) : ℕ :=
  (alvin_age / 2) - 5

theorem simon_age_is_10 : simon_age alvin_age = 10 := by
  sorry

end NUMINAMATH_GPT_simon_age_is_10_l49_4951


namespace NUMINAMATH_GPT_find_a10_l49_4994

theorem find_a10 (a : ℕ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ, 0 < n → (1 / (a (n + 1) - 1)) = (1 / (a n - 1)) - 1) : 
  a 10 = 10/11 := 
sorry

end NUMINAMATH_GPT_find_a10_l49_4994


namespace NUMINAMATH_GPT_magnitude_of_z_l49_4920

theorem magnitude_of_z (z : ℂ) (h : z * (2 - Complex.I) = Complex.I) : Complex.abs z = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l49_4920


namespace NUMINAMATH_GPT_mechanic_hours_l49_4984

theorem mechanic_hours (h : ℕ) (labor_cost_per_hour parts_cost total_bill : ℕ) 
  (H1 : labor_cost_per_hour = 45) 
  (H2 : parts_cost = 225) 
  (H3 : total_bill = 450) 
  (H4 : labor_cost_per_hour * h + parts_cost = total_bill) : 
  h = 5 := 
by
  sorry

end NUMINAMATH_GPT_mechanic_hours_l49_4984


namespace NUMINAMATH_GPT_company_l49_4930

-- Define conditions
def initial_outlay : ℝ := 10000

def material_cost_per_set_first_300 : ℝ := 20
def material_cost_per_set_beyond_300 : ℝ := 15

def exchange_rate : ℝ := 1.1

def import_tax_rate : ℝ := 0.10

def sales_price_per_set_first_400 : ℝ := 50
def sales_price_per_set_beyond_400 : ℝ := 45

def export_tax_threshold : ℕ := 500
def export_tax_rate : ℝ := 0.05

def production_and_sales : ℕ := 800

-- Helper functions for the problem
def material_cost_first_300_sets : ℝ :=
  300 * material_cost_per_set_first_300 * exchange_rate

def material_cost_next_500_sets : ℝ :=
  (production_and_sales - 300) * material_cost_per_set_beyond_300 * exchange_rate

def total_material_cost : ℝ :=
  material_cost_first_300_sets + material_cost_next_500_sets

def import_tax : ℝ := total_material_cost * import_tax_rate

def total_manufacturing_cost : ℝ :=
  initial_outlay + total_material_cost + import_tax

def sales_revenue_first_400_sets : ℝ :=
  400 * sales_price_per_set_first_400

def sales_revenue_next_400_sets : ℝ :=
  (production_and_sales - 400) * sales_price_per_set_beyond_400

def total_sales_revenue_before_export_tax : ℝ :=
  sales_revenue_first_400_sets + sales_revenue_next_400_sets

def sales_revenue_beyond_threshold : ℝ :=
  (production_and_sales - export_tax_threshold) * sales_price_per_set_beyond_400

def export_tax : ℝ := sales_revenue_beyond_threshold * export_tax_rate

def total_sales_revenue_after_export_tax : ℝ :=
  total_sales_revenue_before_export_tax - export_tax

def profit : ℝ :=
  total_sales_revenue_after_export_tax - total_manufacturing_cost

-- Lean 4 statement for the proof problem
theorem company's_profit_is_10990 :
  profit = 10990 := by
  sorry

end NUMINAMATH_GPT_company_l49_4930


namespace NUMINAMATH_GPT_silver_dollars_l49_4985

variable (C : ℕ)
variable (H : ℕ)
variable (P : ℕ)

theorem silver_dollars (h1 : H = P + 5) (h2 : P = C + 16) (h3 : C + P + H = 205) : C = 56 :=
by
  sorry

end NUMINAMATH_GPT_silver_dollars_l49_4985


namespace NUMINAMATH_GPT_simplify_and_evaluate_equals_l49_4986

noncomputable def simplify_and_evaluate (a : ℝ) : ℝ :=
  (a^2 - 4) / (a^2 - 4 * a + 4) - a / (a - 2) / (a^2 + 2 * a) / (a - 2)

theorem simplify_and_evaluate_equals (a : ℝ) (h : a^2 + 2 * a - 8 = 0) : 
  simplify_and_evaluate a = 1 / 4 :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_equals_l49_4986


namespace NUMINAMATH_GPT_range_of_m_l49_4991

-- Defining the conditions
variable (x m : ℝ)

-- The theorem statement
theorem range_of_m (h : ∀ x : ℝ, x < m → 2*x + 1 < 5) : m ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l49_4991


namespace NUMINAMATH_GPT_original_price_l49_4902

theorem original_price (selling_price profit_percent : ℝ) (h_sell : selling_price = 63) (h_profit : profit_percent = 5) : 
  selling_price / (1 + profit_percent / 100) = 60 :=
by sorry

end NUMINAMATH_GPT_original_price_l49_4902


namespace NUMINAMATH_GPT_initial_gasoline_percentage_calculation_l49_4997

variable (initial_volume : ℝ)
variable (initial_ethanol_percentage : ℝ)
variable (additional_ethanol : ℝ)
variable (final_ethanol_percentage : ℝ)

theorem initial_gasoline_percentage_calculation
  (h1: initial_ethanol_percentage = 5)
  (h2: initial_volume = 45)
  (h3: additional_ethanol = 2.5)
  (h4: final_ethanol_percentage = 10) :
  100 - initial_ethanol_percentage = 95 :=
by
  sorry

end NUMINAMATH_GPT_initial_gasoline_percentage_calculation_l49_4997


namespace NUMINAMATH_GPT_sum_GCF_LCM_l49_4908

-- Definitions of GCD and LCM for the numbers 18, 27, and 36
def GCF : ℕ := Nat.gcd (Nat.gcd 18 27) 36
def LCM : ℕ := Nat.lcm (Nat.lcm 18 27) 36

-- Theorem statement proof
theorem sum_GCF_LCM : GCF + LCM = 117 := by
  sorry

end NUMINAMATH_GPT_sum_GCF_LCM_l49_4908


namespace NUMINAMATH_GPT_square_same_area_as_rectangle_l49_4972

theorem square_same_area_as_rectangle (l w : ℝ) (rect_area sq_side : ℝ) :
  l = 25 → w = 9 → rect_area = l * w → sq_side^2 = rect_area → sq_side = 15 :=
by
  intros h_l h_w h_rect_area h_sq_area
  rw [h_l, h_w] at h_rect_area
  sorry

end NUMINAMATH_GPT_square_same_area_as_rectangle_l49_4972


namespace NUMINAMATH_GPT_star_four_three_l49_4948

def star (x y : ℕ) : ℕ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_GPT_star_four_three_l49_4948


namespace NUMINAMATH_GPT_hari_joins_l49_4988

theorem hari_joins {x : ℕ} :
  let praveen_start := 3500
  let hari_start := 9000
  let total_months := 12
  (praveen_start * total_months) * 3 = (hari_start * (total_months - x)) * 2
  → x = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hari_joins_l49_4988


namespace NUMINAMATH_GPT_at_least_one_nonnegative_l49_4905

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 :=
sorry

end NUMINAMATH_GPT_at_least_one_nonnegative_l49_4905


namespace NUMINAMATH_GPT_two_packs_remainder_l49_4938

theorem two_packs_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_packs_remainder_l49_4938


namespace NUMINAMATH_GPT_math_problem_proof_l49_4939

theorem math_problem_proof (a b x y : ℝ) 
  (h1: x = a) 
  (h2: y = b)
  (h3: a + a = b * a)
  (h4: y = a)
  (h5: a * a = a + a)
  (h6: b = 3) : 
  x * y = 4 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_proof_l49_4939


namespace NUMINAMATH_GPT_simplify_expression_l49_4964

-- Define the variables x and y
variables (x y : ℝ)

-- State the theorem
theorem simplify_expression (x y : ℝ) (hy : y ≠ 0) :
  ((x + 3 * y)^2 - (x + y) * (x - y)) / (2 * y) = 3 * x + 5 * y := 
by 
  -- skip the proof
  sorry

end NUMINAMATH_GPT_simplify_expression_l49_4964


namespace NUMINAMATH_GPT_min_area_OBX_l49_4932

structure Point : Type :=
  (x : ℤ)
  (y : ℤ)

def O : Point := ⟨0, 0⟩
def B : Point := ⟨11, 8⟩

def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def in_rectangle (X : Point) : Prop :=
  0 ≤ X.x ∧ X.x ≤ 11 ∧ 0 ≤ X.y ∧ X.y ≤ 8

theorem min_area_OBX : ∃ (X : Point), in_rectangle X ∧ area_triangle O B X = 1 / 2 :=
sorry

end NUMINAMATH_GPT_min_area_OBX_l49_4932


namespace NUMINAMATH_GPT_range_of_a_l49_4955

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ∈ {x : ℝ | x ≥ 3 ∨ x ≤ -1} ∩ {x : ℝ | x ≤ a} ↔ x ∈ {x : ℝ | x ≤ a})) ↔ a ≤ -1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l49_4955


namespace NUMINAMATH_GPT_intersection_eq_set_l49_4929

-- Define set A based on the inequality
def A : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set B based on the inequality
def B : Set ℝ := {x | 0 ≤ Real.log (x + 1) / Real.log 2 ∧ Real.log (x + 1) / Real.log 2 < 2}

-- Translate the question to a lean theorem
theorem intersection_eq_set : (A ∩ B) = {x | 0 ≤ x ∧ x < 1} := 
sorry

end NUMINAMATH_GPT_intersection_eq_set_l49_4929


namespace NUMINAMATH_GPT_broken_seashells_count_l49_4926

def total_seashells : ℕ := 7
def unbroken_seashells : ℕ := 3

theorem broken_seashells_count : (total_seashells - unbroken_seashells) = 4 := by
  sorry

end NUMINAMATH_GPT_broken_seashells_count_l49_4926


namespace NUMINAMATH_GPT_pie_split_l49_4989

theorem pie_split (initial_pie : ℚ) (number_of_people : ℕ) (amount_taken_by_each : ℚ) 
  (h1 : initial_pie = 5/6) (h2 : number_of_people = 4) : amount_taken_by_each = 5/24 :=
by
  sorry

end NUMINAMATH_GPT_pie_split_l49_4989


namespace NUMINAMATH_GPT_pinning_7_nails_l49_4996

theorem pinning_7_nails {n : ℕ} (circles : Fin n → Set (ℝ × ℝ)) :
  (∀ i j : Fin n, i ≠ j → ∃ p : ℝ × ℝ, p ∈ circles i ∧ p ∈ circles j) →
  ∃ s : Finset (ℝ × ℝ), s.card ≤ 7 ∧ ∀ i : Fin n, ∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ circles i :=
by sorry

end NUMINAMATH_GPT_pinning_7_nails_l49_4996


namespace NUMINAMATH_GPT_no_pairs_satisfy_equation_l49_4910

theorem no_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ¬ (2 / a + 2 / b = 1 / (a + b)) :=
by
  intros a b ha hb h
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_no_pairs_satisfy_equation_l49_4910


namespace NUMINAMATH_GPT_fixed_costs_16699_50_l49_4967

noncomputable def fixed_monthly_costs (production_cost shipping_cost units_sold price_per_unit : ℝ) : ℝ :=
  let total_variable_cost := (production_cost + shipping_cost) * units_sold
  let total_revenue := price_per_unit * units_sold
  total_revenue - total_variable_cost

theorem fixed_costs_16699_50 :
  fixed_monthly_costs 80 7 150 198.33 = 16699.5 :=
by
  sorry

end NUMINAMATH_GPT_fixed_costs_16699_50_l49_4967


namespace NUMINAMATH_GPT_sticker_price_of_laptop_l49_4953

variable (x : ℝ)

-- Conditions
noncomputable def price_store_A : ℝ := 0.90 * x - 100
noncomputable def price_store_B : ℝ := 0.80 * x
noncomputable def savings : ℝ := price_store_B x - price_store_A x

-- Theorem statement
theorem sticker_price_of_laptop (x : ℝ) (h : savings x = 20) : x = 800 :=
by
  sorry

end NUMINAMATH_GPT_sticker_price_of_laptop_l49_4953


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l49_4943

theorem solution_set_of_inequalities :
  {x : ℝ | 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9} = {x : ℝ | x > 45 / 26} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l49_4943


namespace NUMINAMATH_GPT_min_value_sum_reciprocal_squares_l49_4900

open Real

theorem min_value_sum_reciprocal_squares 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :  
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 27 := 
sorry

end NUMINAMATH_GPT_min_value_sum_reciprocal_squares_l49_4900


namespace NUMINAMATH_GPT_cylinder_volume_triple_radius_quadruple_height_l49_4979

open Real

theorem cylinder_volume_triple_radius_quadruple_height (r h : ℝ) (V : ℝ) (hV : V = π * r^2 * h) :
  (3 * r) ^ 2 * 4 * h * π = 360 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_triple_radius_quadruple_height_l49_4979


namespace NUMINAMATH_GPT_friends_gcd_l49_4958

theorem friends_gcd {a b : ℤ} (h : ∃ n : ℤ, a * b = n * n) : 
  ∃ m : ℤ, a * Int.gcd a b = m * m :=
sorry

end NUMINAMATH_GPT_friends_gcd_l49_4958


namespace NUMINAMATH_GPT_lines_intersect_sum_c_d_l49_4919

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end NUMINAMATH_GPT_lines_intersect_sum_c_d_l49_4919


namespace NUMINAMATH_GPT_min_value_of_sum_eq_l49_4945

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_min_value_of_sum_eq_l49_4945


namespace NUMINAMATH_GPT_cos_75_degree_identity_l49_4949

theorem cos_75_degree_identity :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_75_degree_identity_l49_4949


namespace NUMINAMATH_GPT_proportion_of_capacity_filled_l49_4946

noncomputable def milk_proportion_8cup_bottle : ℚ := 16 / 3
noncomputable def total_milk := 8

theorem proportion_of_capacity_filled :
  ∃ p : ℚ, (8 * p = milk_proportion_8cup_bottle) ∧ (4 * p = total_milk - milk_proportion_8cup_bottle) ∧ (p = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_capacity_filled_l49_4946


namespace NUMINAMATH_GPT_solve_for_x_l49_4976

theorem solve_for_x (x : ℝ) (h : 12 - 2 * x = 6) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l49_4976


namespace NUMINAMATH_GPT_sum_of_two_integers_eq_sqrt_466_l49_4959

theorem sum_of_two_integers_eq_sqrt_466
  (x y : ℝ)
  (hx : x^2 + y^2 = 250)
  (hy : x * y = 108) :
  x + y = Real.sqrt 466 :=
sorry

end NUMINAMATH_GPT_sum_of_two_integers_eq_sqrt_466_l49_4959


namespace NUMINAMATH_GPT_seq_product_l49_4925

theorem seq_product (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 2^n - 1)
  (ha : ∀ n, a n = if n = 1 then 1 else 2^(n-1)) :
  a 2 * a 6 = 64 :=
by 
  sorry

end NUMINAMATH_GPT_seq_product_l49_4925


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l49_4977

theorem arithmetic_expression_evaluation :
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := 
by
  -- Skipping the proof.
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l49_4977


namespace NUMINAMATH_GPT_tangent_line_circle_l49_4956

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x - 2*y + m = 0 ↔ (x^2 + y^2 - 4*x + 6*y + 8 = 0)) →
  m = -3 ∨ m = -13 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l49_4956


namespace NUMINAMATH_GPT_committeeFormation_l49_4913

-- Establish the given problem conditions in Lean

open Classical

-- Noncomputable because we are working with combinations and products
noncomputable def numberOfWaysToFormCommittee (numSchools : ℕ) (membersPerSchool : ℕ) (hostSchools : ℕ) (hostReps : ℕ) (nonHostReps : ℕ) : ℕ :=
  let totalSchools := numSchools
  let chooseHostSchools := Nat.choose totalSchools hostSchools
  let chooseHostRepsPerSchool := Nat.choose membersPerSchool hostReps
  let allHostRepsChosen := chooseHostRepsPerSchool ^ hostSchools
  let chooseNonHostRepsPerSchool := Nat.choose membersPerSchool nonHostReps
  let allNonHostRepsChosen := chooseNonHostRepsPerSchool ^ (totalSchools - hostSchools)
  chooseHostSchools * allHostRepsChosen * allNonHostRepsChosen

-- We now state our theorem
theorem committeeFormation : numberOfWaysToFormCommittee 4 6 2 3 1 = 86400 :=
by
  -- This is the lemma we need to prove
  sorry

end NUMINAMATH_GPT_committeeFormation_l49_4913


namespace NUMINAMATH_GPT_subtract_from_sum_base8_l49_4922

def add_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) + (b % 8)) % 8
  + (((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) % 8) * 8
  + (((((a / 8) % 8 + (b / 8) % 8 + ((a % 8) + (b % 8)) / 8) / 8) + ((a / 64) % 8 + (b / 64) % 8)) % 8) * 64

def subtract_in_base_8 (a b : ℕ) : ℕ :=
  ((a % 8) - (b % 8) + 8) % 8
  + (((a / 8) % 8 - (b / 8) % 8 - if (a % 8) < (b % 8) then 1 else 0 + 8) % 8) * 8
  + (((a / 64) - (b / 64) - if (a / 8) % 8 < (b / 8) % 8 then 1 else 0) % 8) * 64

theorem subtract_from_sum_base8 :
  subtract_in_base_8 (add_in_base_8 652 147) 53 = 50 := by
  sorry

end NUMINAMATH_GPT_subtract_from_sum_base8_l49_4922


namespace NUMINAMATH_GPT_last_passenger_probability_l49_4944

noncomputable def probability_last_passenger_gets_seat {n : ℕ} (h : n > 0) : ℚ :=
  if n = 1 then 1 else 1/2

theorem last_passenger_probability
  (n : ℕ) (h : n > 0) :
  probability_last_passenger_gets_seat h = 1/2 :=
  sorry

end NUMINAMATH_GPT_last_passenger_probability_l49_4944
