import Mathlib

namespace NUMINAMATH_GPT_original_cost_of_meal_l1309_130952

-- Definitions for conditions
def meal_cost (initial_cost : ℝ) : ℝ :=
  initial_cost + 0.085 * initial_cost + 0.18 * initial_cost

-- The theorem we aim to prove
theorem original_cost_of_meal (total_cost : ℝ) (h : total_cost = 35.70) :
  ∃ initial_cost : ℝ, initial_cost = 28.23 ∧ meal_cost initial_cost = total_cost :=
by
  use 28.23
  rw [meal_cost, h]
  sorry

end NUMINAMATH_GPT_original_cost_of_meal_l1309_130952


namespace NUMINAMATH_GPT_largest_even_number_in_series_l1309_130957

/-- 
  If the sum of 25 consecutive even numbers is 10,000,
  what is the largest number among these 25 consecutive even numbers? 
-/
theorem largest_even_number_in_series (n : ℤ) (S : ℤ) (h : S = 25 * (n - 24)) (h_sum : S = 10000) :
  n = 424 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_largest_even_number_in_series_l1309_130957


namespace NUMINAMATH_GPT_anja_equal_integers_l1309_130938

theorem anja_equal_integers (S : Finset ℤ) (h_card : S.card = 2014)
  (h_mean : ∀ (x y z : ℤ), x ∈ S → y ∈ S → z ∈ S → (x + y + z) / 3 ∈ S) :
  ∃ k, ∀ x ∈ S, x = k :=
sorry

end NUMINAMATH_GPT_anja_equal_integers_l1309_130938


namespace NUMINAMATH_GPT_sum_of_two_integers_is_22_l1309_130972

noncomputable def a_and_b_sum_to_S : Prop :=
  ∃ (a b S : ℕ), 
    a + b = S ∧ 
    a^2 - b^2 = 44 ∧ 
    a * b = 120 ∧ 
    S = 22

theorem sum_of_two_integers_is_22 : a_and_b_sum_to_S :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_two_integers_is_22_l1309_130972


namespace NUMINAMATH_GPT_vertex_of_parabola_l1309_130951

theorem vertex_of_parabola (a b c : ℝ) (h k : ℝ) (x y : ℝ) :
  (∀ x, y = (1/2) * (x - 1)^2 + 2) → (h, k) = (1, 2) :=
by
  intro hy
  exact sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1309_130951


namespace NUMINAMATH_GPT_arithmetic_expression_equals_fraction_l1309_130910

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equals_fraction_l1309_130910


namespace NUMINAMATH_GPT_area_of_annulus_l1309_130931

variables (R r x : ℝ) (hRr : R > r) (h : R^2 - r^2 = x^2)

theorem area_of_annulus : π * R^2 - π * r^2 = π * x^2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_annulus_l1309_130931


namespace NUMINAMATH_GPT_least_integer_value_l1309_130962

theorem least_integer_value (x : ℤ) :
  (|3 * x + 4| ≤ 25) → ∃ y : ℤ, x = y ∧ y = -9 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_l1309_130962


namespace NUMINAMATH_GPT_angles_arithmetic_progression_l1309_130982

theorem angles_arithmetic_progression (A B C : ℝ) (h_sum : A + B + C = 180) :
  (B = 60) ↔ (A + C = 2 * B) :=
by
  sorry

end NUMINAMATH_GPT_angles_arithmetic_progression_l1309_130982


namespace NUMINAMATH_GPT_cole_drive_time_to_work_l1309_130989

theorem cole_drive_time_to_work :
  ∀ (D : ℝ),
    (D / 80 + D / 120 = 3) → (D / 80 * 60 = 108) :=
by
  intro D h
  sorry

end NUMINAMATH_GPT_cole_drive_time_to_work_l1309_130989


namespace NUMINAMATH_GPT_find_t_l1309_130980

-- Define the elements and the conditions
def vector_a : ℝ × ℝ := (1, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 1)

def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def sub_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Lean statement of the problem
theorem find_t (t : ℝ) : 
  parallel (add_vectors vector_a (vector_b t)) (sub_vectors vector_a (vector_b t)) → t = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1309_130980


namespace NUMINAMATH_GPT_train_length_l1309_130956

theorem train_length (V L : ℝ) (h1 : L = V * 18) (h2 : L + 550 = V * 51) : L = 300 := sorry

end NUMINAMATH_GPT_train_length_l1309_130956


namespace NUMINAMATH_GPT_closest_integer_to_10_minus_sqrt_12_l1309_130966

theorem closest_integer_to_10_minus_sqrt_12 (a b c d : ℤ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  d = 7 :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_10_minus_sqrt_12_l1309_130966


namespace NUMINAMATH_GPT_base8_subtraction_l1309_130974

theorem base8_subtraction : (53 - 26 : ℕ) = 25 :=
by sorry

end NUMINAMATH_GPT_base8_subtraction_l1309_130974


namespace NUMINAMATH_GPT_goose_eggs_l1309_130928

theorem goose_eggs (E : ℕ) 
  (H1 : (2/3 : ℚ) * E = h) 
  (H2 : (3/4 : ℚ) * h = m)
  (H3 : (2/5 : ℚ) * m = 180) : 
  E = 2700 := 
sorry

end NUMINAMATH_GPT_goose_eggs_l1309_130928


namespace NUMINAMATH_GPT_number_of_four_digit_numbers_with_two_identical_digits_l1309_130975

-- Define the conditions
def starts_with_nine (n : ℕ) : Prop := n / 1000 = 9
def has_exactly_two_identical_digits (n : ℕ) : Prop := 
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d2) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d2 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d1) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d2 ∧ n % 10 = d1)

-- Define the proof problem
theorem number_of_four_digit_numbers_with_two_identical_digits : 
  ∃ n, starts_with_nine n ∧ has_exactly_two_identical_digits n ∧ n = 432 := 
sorry

end NUMINAMATH_GPT_number_of_four_digit_numbers_with_two_identical_digits_l1309_130975


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1309_130907

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 8 + (5 : ℚ) / 12 / 2 = 19 / 48 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1309_130907


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1309_130912

theorem opposite_of_neg_2023 :
  ∃ y : ℝ, (-2023 + y = 0) ∧ y = 2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1309_130912


namespace NUMINAMATH_GPT_spadesuit_eval_l1309_130935

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end NUMINAMATH_GPT_spadesuit_eval_l1309_130935


namespace NUMINAMATH_GPT_supermarket_profit_and_discount_l1309_130961

theorem supermarket_profit_and_discount :
  ∃ (x : ℕ) (nB1 nB2 : ℕ) (discount_rate : ℝ),
    22*x + 30*(nB1) = 6000 ∧
    nB1 = (1 / 2 : ℝ) * x + 15 ∧
    150 * (29 - 22) + 90 * (40 - 30) = 1950 ∧
    nB2 = 3 * nB1 ∧
    150 * (29 - 22) + 270 * (40 * (1 - discount_rate / 100) - 30) = 2130 ∧
    discount_rate = 8.5 := sorry

end NUMINAMATH_GPT_supermarket_profit_and_discount_l1309_130961


namespace NUMINAMATH_GPT_fewest_posts_required_l1309_130909

def dimensions_garden : ℕ × ℕ := (32, 72)
def post_spacing : ℕ := 8

theorem fewest_posts_required
  (d : ℕ × ℕ := dimensions_garden)
  (s : ℕ := post_spacing) :
  d = (32, 72) ∧ s = 8 → 
  ∃ N, N = 26 := 
by 
  sorry

end NUMINAMATH_GPT_fewest_posts_required_l1309_130909


namespace NUMINAMATH_GPT_molecular_weight_of_benzene_l1309_130964

def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def number_of_C_atoms : ℕ := 6
def number_of_H_atoms : ℕ := 6

theorem molecular_weight_of_benzene : 
  (number_of_C_atoms * molecular_weight_C + number_of_H_atoms * molecular_weight_H) = 78.108 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_benzene_l1309_130964


namespace NUMINAMATH_GPT_book_count_l1309_130943

theorem book_count (P C B : ℕ) (h1 : P = 3 * C / 2) (h2 : B = 3 * C / 4) (h3 : P + C + B > 3000) : 
  P + C + B = 3003 := by
  sorry

end NUMINAMATH_GPT_book_count_l1309_130943


namespace NUMINAMATH_GPT_initial_fliers_l1309_130911

variable (F : ℕ) -- Initial number of fliers

-- Conditions
axiom morning_send : F - (1 / 5) * F = (4 / 5) * F
axiom afternoon_send : (4 / 5) * F - (1 / 4) * ((4 / 5) * F) = (3 / 5) * F
axiom final_count : (3 / 5) * F = 600

theorem initial_fliers : F = 1000 := by
  sorry

end NUMINAMATH_GPT_initial_fliers_l1309_130911


namespace NUMINAMATH_GPT_total_passengers_landed_l1309_130998

theorem total_passengers_landed (on_time late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) : 
    on_time + late = 14720 :=
by
  sorry

end NUMINAMATH_GPT_total_passengers_landed_l1309_130998


namespace NUMINAMATH_GPT_pie_chart_shows_percentage_l1309_130950

-- Define the different types of graphs
inductive GraphType
| PieChart
| BarGraph
| LineGraph
| Histogram

-- Define conditions of the problem
def shows_percentage_of_whole (g : GraphType) : Prop :=
  g = GraphType.PieChart

def displays_with_rectangular_bars (g : GraphType) : Prop :=
  g = GraphType.BarGraph

def shows_trends (g : GraphType) : Prop :=
  g = GraphType.LineGraph

def shows_frequency_distribution (g : GraphType) : Prop :=
  g = GraphType.Histogram

-- We need to prove that a pie chart satisfies the condition of showing percentages of parts in a whole
theorem pie_chart_shows_percentage : shows_percentage_of_whole GraphType.PieChart :=
  by
    -- Proof is skipped
    sorry

end NUMINAMATH_GPT_pie_chart_shows_percentage_l1309_130950


namespace NUMINAMATH_GPT_sum21_exists_l1309_130981

theorem sum21_exists (S : Finset ℕ) (h_size : S.card = 11) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 20) :
  ∃ a b, a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ a + b = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum21_exists_l1309_130981


namespace NUMINAMATH_GPT_max_consecutive_sum_l1309_130932

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_consecutive_sum_l1309_130932


namespace NUMINAMATH_GPT_candy_distribution_l1309_130970

theorem candy_distribution (n : Nat) : ∃ k : Nat, n = 2 ^ k :=
sorry

end NUMINAMATH_GPT_candy_distribution_l1309_130970


namespace NUMINAMATH_GPT_x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l1309_130933

theorem x_eq_1_sufficient_not_necessary_for_x_sq_eq_1 (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ((x^2 = 1) → (x = 1 ∨ x = -1)) :=
by 
  sorry

end NUMINAMATH_GPT_x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l1309_130933


namespace NUMINAMATH_GPT_average_age_of_cricket_team_l1309_130993

theorem average_age_of_cricket_team
  (A : ℝ)
  (captain_age : ℝ) (wicket_keeper_age : ℝ)
  (team_size : ℕ) (remaining_players : ℕ)
  (captain_age_eq : captain_age = 24)
  (wicket_keeper_age_eq : wicket_keeper_age = 27)
  (remaining_players_eq : remaining_players = team_size - 2)
  (average_age_condition : (team_size * A - (captain_age + wicket_keeper_age)) = remaining_players * (A - 1)) : 
  A = 21 := by
  sorry

end NUMINAMATH_GPT_average_age_of_cricket_team_l1309_130993


namespace NUMINAMATH_GPT_evaluate_expression_l1309_130976

theorem evaluate_expression : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1309_130976


namespace NUMINAMATH_GPT_shorter_piece_length_l1309_130939

-- Definitions for the conditions
def total_length : ℕ := 70
def ratio (short long : ℕ) : Prop := long = (5 * short) / 2

-- The proof problem statement
theorem shorter_piece_length (x : ℕ) (h1 : total_length = x + (5 * x) / 2) : x = 20 :=
sorry

end NUMINAMATH_GPT_shorter_piece_length_l1309_130939


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l1309_130984

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l1309_130984


namespace NUMINAMATH_GPT_cos_double_angle_of_tan_l1309_130949

theorem cos_double_angle_of_tan (θ : ℝ) (h : Real.tan θ = -1 / 3) : Real.cos (2 * θ) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_of_tan_l1309_130949


namespace NUMINAMATH_GPT_tommys_profit_l1309_130924

-- Definitions of the conditions
def crateA_cost : ℕ := 220
def crateB_cost : ℕ := 375
def crateC_cost : ℕ := 180

def crateA_count : ℕ := 2
def crateB_count : ℕ := 3
def crateC_count : ℕ := 1

def crateA_capacity : ℕ := 20
def crateB_capacity : ℕ := 25
def crateC_capacity : ℕ := 30

def crateA_rotten : ℕ := 4
def crateB_rotten : ℕ := 5
def crateC_rotten : ℕ := 3

def crateA_price_per_kg : ℕ := 5
def crateB_price_per_kg : ℕ := 6
def crateC_price_per_kg : ℕ := 7

-- Calculations based on the conditions
def total_cost : ℕ := crateA_cost + crateB_cost + crateC_cost

def sellable_weightA : ℕ := crateA_count * crateA_capacity - crateA_rotten
def sellable_weightB : ℕ := crateB_count * crateB_capacity - crateB_rotten
def sellable_weightC : ℕ := crateC_count * crateC_capacity - crateC_rotten

def revenueA : ℕ := sellable_weightA * crateA_price_per_kg
def revenueB : ℕ := sellable_weightB * crateB_price_per_kg
def revenueC : ℕ := sellable_weightC * crateC_price_per_kg

def total_revenue : ℕ := revenueA + revenueB + revenueC

def profit : ℕ := total_revenue - total_cost

-- The theorem we want to verify
theorem tommys_profit : profit = 14 := by
  sorry

end NUMINAMATH_GPT_tommys_profit_l1309_130924


namespace NUMINAMATH_GPT_xy_relationship_l1309_130906

theorem xy_relationship (x y : ℝ) (h : y = 2 * x - 1 - Real.sqrt (y^2 - 2 * x * y + 3 * x - 2)) :
  (x ≠ 1 → y = 2 * x - 1.5) ∧ (x = 1 → y ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_xy_relationship_l1309_130906


namespace NUMINAMATH_GPT_find_x_value_l1309_130997

def my_operation (a b : ℝ) : ℝ := 2 * a * b + 3 * b - 2 * a

theorem find_x_value (x : ℝ) (h : my_operation 3 x = 60) : x = 7.33 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l1309_130997


namespace NUMINAMATH_GPT_al_original_amount_l1309_130922

theorem al_original_amount : 
  ∃ (a b c : ℝ), 
    a + b + c = 1200 ∧ 
    (a - 200 + 3 * b + 4 * c) = 1800 ∧ 
    b = 2800 - 3 * a ∧ 
    c = 1200 - a - b ∧ 
    a = 860 := by
  sorry

end NUMINAMATH_GPT_al_original_amount_l1309_130922


namespace NUMINAMATH_GPT_base_133_not_perfect_square_l1309_130948

theorem base_133_not_perfect_square (b : ℤ) : ¬ ∃ k : ℤ, b^2 + 3 * b + 3 = k^2 := by
  sorry

end NUMINAMATH_GPT_base_133_not_perfect_square_l1309_130948


namespace NUMINAMATH_GPT_nancy_spelling_problems_l1309_130947

structure NancyProblems where
  math_problems : ℝ
  rate : ℝ
  hours : ℝ
  total_problems : ℝ

noncomputable def calculate_spelling_problems (n : NancyProblems) : ℝ :=
  n.total_problems - n.math_problems

theorem nancy_spelling_problems :
  ∀ (n : NancyProblems), n.math_problems = 17.0 ∧ n.rate = 8.0 ∧ n.hours = 4.0 ∧ n.total_problems = 32.0 →
  calculate_spelling_problems n = 15.0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nancy_spelling_problems_l1309_130947


namespace NUMINAMATH_GPT_total_years_l1309_130958

variable (T D : ℕ)
variable (Tom_years : T = 50)
variable (Devin_years : D = 25 - 5)

theorem total_years (hT : T = 50) (hD : D = 25 - 5) : T + D = 70 := by
  sorry

end NUMINAMATH_GPT_total_years_l1309_130958


namespace NUMINAMATH_GPT_no_int_solutions_for_equation_l1309_130988

theorem no_int_solutions_for_equation :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * y^2 = x^4 + x := 
sorry

end NUMINAMATH_GPT_no_int_solutions_for_equation_l1309_130988


namespace NUMINAMATH_GPT_polynomial_coeff_diff_l1309_130990

theorem polynomial_coeff_diff (a b c d e f : ℝ) :
  ((3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a - b + c - d + e - f = 32) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeff_diff_l1309_130990


namespace NUMINAMATH_GPT_Tom_money_made_l1309_130969

theorem Tom_money_made (money_last_week money_now : ℕ) (h1 : money_last_week = 74) (h2 : money_now = 160) : 
  (money_now - money_last_week = 86) :=
by 
  sorry

end NUMINAMATH_GPT_Tom_money_made_l1309_130969


namespace NUMINAMATH_GPT_triangle_centroid_property_l1309_130919

def distance_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem triangle_centroid_property
  (A B C P : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hG : G = ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )) :
  distance_sq A P + distance_sq B P + distance_sq C P = 
  distance_sq A G + distance_sq B G + distance_sq C G + 3 * distance_sq G P :=
by
  sorry

end NUMINAMATH_GPT_triangle_centroid_property_l1309_130919


namespace NUMINAMATH_GPT_final_alcohol_percentage_l1309_130942

noncomputable def initial_volume : ℝ := 6
noncomputable def initial_percentage : ℝ := 0.25
noncomputable def added_alcohol : ℝ := 3
noncomputable def final_volume : ℝ := initial_volume + added_alcohol
noncomputable def final_percentage : ℝ := (initial_volume * initial_percentage + added_alcohol) / final_volume * 100

theorem final_alcohol_percentage :
  final_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_final_alcohol_percentage_l1309_130942


namespace NUMINAMATH_GPT_meaning_of_a2_add_b2_ne_zero_l1309_130918

theorem meaning_of_a2_add_b2_ne_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_meaning_of_a2_add_b2_ne_zero_l1309_130918


namespace NUMINAMATH_GPT_dice_probability_l1309_130936

theorem dice_probability :
  let prob_roll_less_than_four := 3 / 6
  let prob_roll_even := 3 / 6
  let prob_roll_greater_than_four := 2 / 6
  prob_roll_less_than_four * prob_roll_even * prob_roll_greater_than_four = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l1309_130936


namespace NUMINAMATH_GPT_find_a_l1309_130954

theorem find_a (a : ℝ) : 3 * a + 150 = 360 → a = 70 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1309_130954


namespace NUMINAMATH_GPT_sum_of_cousins_ages_l1309_130937

theorem sum_of_cousins_ages :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧
    a * b = 36 ∧ c * d = 40 ∧ a + b + c + d + e = 33 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cousins_ages_l1309_130937


namespace NUMINAMATH_GPT_total_amount_paid_l1309_130999

theorem total_amount_paid (monthly_payment_1 monthly_payment_2 : ℕ) (years_1 years_2 : ℕ)
  (monthly_payment_1_eq : monthly_payment_1 = 300)
  (monthly_payment_2_eq : monthly_payment_2 = 350)
  (years_1_eq : years_1 = 3)
  (years_2_eq : years_2 = 2) :
  let annual_payment_1 := monthly_payment_1 * 12
  let annual_payment_2 := monthly_payment_2 * 12
  let total_1 := annual_payment_1 * years_1
  let total_2 := annual_payment_2 * years_2
  total_1 + total_2 = 19200 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1309_130999


namespace NUMINAMATH_GPT_sum_of_all_digits_divisible_by_nine_l1309_130901

theorem sum_of_all_digits_divisible_by_nine :
  ∀ (A B C D : ℕ),
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
  sorry

end NUMINAMATH_GPT_sum_of_all_digits_divisible_by_nine_l1309_130901


namespace NUMINAMATH_GPT_evaluate_f_at_1_l1309_130927

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem evaluate_f_at_1 : f 1 = 6 := 
  sorry

end NUMINAMATH_GPT_evaluate_f_at_1_l1309_130927


namespace NUMINAMATH_GPT_range_of_m_l1309_130971

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Dot product function for two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition for acute angle
def is_acute (m : ℝ) : Prop := dot_product a (b m) > 0

-- Definition of the range of m
def m_range : Set ℝ := {m | m > -12 ∧ m ≠ 4/3}

-- The theorem to prove
theorem range_of_m (m : ℝ) : is_acute m → m ∈ m_range :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1309_130971


namespace NUMINAMATH_GPT_find_K_l1309_130929

theorem find_K (Z K : ℕ)
  (hZ1 : 700 < Z)
  (hZ2 : Z < 1500)
  (hK : K > 1)
  (hZ_eq : Z = K^4)
  (hZ_perfect : ∃ n : ℕ, Z = n^6) :
  K = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_K_l1309_130929


namespace NUMINAMATH_GPT_fraction_historical_fiction_new_releases_l1309_130953

-- Define constants for book categories and new releases
def historical_fiction_percentage : ℝ := 0.40
def science_fiction_percentage : ℝ := 0.25
def biographies_percentage : ℝ := 0.15
def mystery_novels_percentage : ℝ := 0.20

def historical_fiction_new_releases : ℝ := 0.45
def science_fiction_new_releases : ℝ := 0.30
def biographies_new_releases : ℝ := 0.50
def mystery_novels_new_releases : ℝ := 0.35

-- Statement of the problem to prove
theorem fraction_historical_fiction_new_releases :
  (historical_fiction_percentage * historical_fiction_new_releases) /
    (historical_fiction_percentage * historical_fiction_new_releases +
     science_fiction_percentage * science_fiction_new_releases +
     biographies_percentage * biographies_new_releases +
     mystery_novels_percentage * mystery_novels_new_releases) = 9 / 20 :=
by
  sorry

end NUMINAMATH_GPT_fraction_historical_fiction_new_releases_l1309_130953


namespace NUMINAMATH_GPT_triangle_angle_sum_property_l1309_130926

theorem triangle_angle_sum_property (A B C : ℝ) (h1: C = 3 * B) (h2: B = 15) : A = 120 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_property_l1309_130926


namespace NUMINAMATH_GPT_total_cost_for_photos_l1309_130900

def total_cost (n : ℕ) (f : ℝ) (c : ℝ) : ℝ :=
  f + (n - 4) * c

theorem total_cost_for_photos :
  total_cost 54 24.5 2.3 = 139.5 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_for_photos_l1309_130900


namespace NUMINAMATH_GPT_increased_expenses_percent_l1309_130973

theorem increased_expenses_percent (S : ℝ) (hS : S = 6250) (initial_save_percent : ℝ) (final_savings : ℝ) 
  (initial_save_percent_def : initial_save_percent = 20) 
  (final_savings_def : final_savings = 250) : 
  (initial_save_percent / 100 * S - final_savings) / (S - initial_save_percent / 100 * S) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_increased_expenses_percent_l1309_130973


namespace NUMINAMATH_GPT_number_of_students_l1309_130913

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N) (h2 : (T - 250) / (N - 5) = 90) : N = 20 :=
sorry

end NUMINAMATH_GPT_number_of_students_l1309_130913


namespace NUMINAMATH_GPT_max_z_value_l1309_130960

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z = 13 / 3 := 
sorry

end NUMINAMATH_GPT_max_z_value_l1309_130960


namespace NUMINAMATH_GPT_solution_correctness_l1309_130985

def is_prime (n : ℕ) : Prop := Nat.Prime n

def problem_statement (a b c : ℕ) : Prop :=
  (a * b * c = 56) ∧
  (a * b + b * c + a * c = 311) ∧
  is_prime a ∧ is_prime b ∧ is_prime c

theorem solution_correctness (a b c : ℕ) (h : problem_statement a b c) :
  a = 2 ∨ a = 13 ∨ a = 19 ∧
  b = 2 ∨ b = 13 ∨ b = 19 ∧
  c = 2 ∨ c = 13 ∨ c = 19 :=
by
  sorry

end NUMINAMATH_GPT_solution_correctness_l1309_130985


namespace NUMINAMATH_GPT_jordan_novels_read_l1309_130996

variable (J A : ℕ)

theorem jordan_novels_read (h1 : A = (1 / 10) * J)
                          (h2 : J = A + 108) :
                          J = 120 := 
by
  sorry

end NUMINAMATH_GPT_jordan_novels_read_l1309_130996


namespace NUMINAMATH_GPT_divide_coal_l1309_130930

noncomputable def part_of_pile (whole: ℚ) (parts: ℕ) := whole / parts
noncomputable def part_tons (total_tons: ℚ) (fraction: ℚ) := total_tons * fraction

theorem divide_coal (total_tons: ℚ) (parts: ℕ) (h: total_tons = 3 ∧ parts = 5):
  (part_of_pile 1 parts = 1/parts) ∧ (part_tons total_tons (1/parts) = total_tons / parts) :=
by
  sorry

end NUMINAMATH_GPT_divide_coal_l1309_130930


namespace NUMINAMATH_GPT_yellow_tiled_area_is_correct_l1309_130925

noncomputable def length : ℝ := 3.6
noncomputable def width : ℝ := 2.5 * length
noncomputable def total_area : ℝ := length * width
noncomputable def yellow_tiled_area : ℝ := total_area / 2

theorem yellow_tiled_area_is_correct (length_eq : length = 3.6)
    (width_eq : width = 2.5 * length)
    (total_area_eq : total_area = length * width)
    (yellow_area_eq : yellow_tiled_area = total_area / 2) :
    yellow_tiled_area = 16.2 := 
by sorry

end NUMINAMATH_GPT_yellow_tiled_area_is_correct_l1309_130925


namespace NUMINAMATH_GPT_triangle_inequality_l1309_130967

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (hS : S = (1/4) * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1309_130967


namespace NUMINAMATH_GPT_license_plate_count_l1309_130902

theorem license_plate_count : 
  let vowels := 5
  let consonants := 21
  let digits := 10
  21 * 21 * 5 * 5 * 10 = 110250 := 
by 
  sorry

end NUMINAMATH_GPT_license_plate_count_l1309_130902


namespace NUMINAMATH_GPT_martha_bedroom_size_l1309_130945

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end NUMINAMATH_GPT_martha_bedroom_size_l1309_130945


namespace NUMINAMATH_GPT_band_section_student_count_l1309_130921

theorem band_section_student_count :
  (0.5 * 500) + (0.12 * 500) + (0.23 * 500) + (0.08 * 500) = 465 :=
by 
  sorry

end NUMINAMATH_GPT_band_section_student_count_l1309_130921


namespace NUMINAMATH_GPT_part1_part2_part3_l1309_130914

-- Definitions of conditions
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_even_between (a b : ℕ) : ℕ := sum_even b - sum_even a

-- Problem 1: Prove that for n = 8, S = 72
theorem part1 (n : ℕ) (h : n = 8) : sum_even n = 72 := by
  rw [h]
  exact rfl

-- Problem 2: Prove the general formula for the sum of the first n consecutive even numbers
theorem part2 (n : ℕ) : sum_even n = n * (n + 1) := by
  exact rfl

-- Problem 3: Prove the sum of 102 to 212 is 8792 using the formula
theorem part3 : sum_even_between 50 106 = 8792 := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1309_130914


namespace NUMINAMATH_GPT_fg_of_2_eq_15_l1309_130916

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2_eq_15 : f (g 2) = 15 :=
by
  -- The detailed proof would go here
  sorry

end NUMINAMATH_GPT_fg_of_2_eq_15_l1309_130916


namespace NUMINAMATH_GPT_rectangle_area_l1309_130944

variable (L B : ℕ)

theorem rectangle_area :
  (L - B = 23) ∧ (2 * L + 2 * B = 166) → (L * B = 1590) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1309_130944


namespace NUMINAMATH_GPT_vector_addition_simplification_l1309_130917

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_simplification
  (AB BC AC DC CD : V)
  (h1 : AB + BC = AC)
  (h2 : - DC = CD) :
  AB + BC - AC - DC = CD :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_vector_addition_simplification_l1309_130917


namespace NUMINAMATH_GPT_post_spacing_change_l1309_130979

theorem post_spacing_change :
  ∀ (posts : ℕ → ℝ) (constant_spacing : ℝ), 
  (∀ n, 1 ≤ n ∧ n < 16 → posts (n + 1) - posts n = constant_spacing) →
  posts 16 - posts 1 = 48 → 
  posts 28 - posts 16 = 36 →
  ∃ (k : ℕ), 16 < k ∧ k ≤ 28 ∧ posts (k + 1) - posts k ≠ constant_spacing ∧ posts (k + 1) - posts k = 2.9 ∧ k = 20 := 
  sorry

end NUMINAMATH_GPT_post_spacing_change_l1309_130979


namespace NUMINAMATH_GPT_xiaoming_accuracy_l1309_130941

theorem xiaoming_accuracy :
  ∀ (correct already_wrong extra_needed : ℕ),
  correct = 30 →
  already_wrong = 6 →
  (correct + extra_needed).toFloat / (correct + already_wrong + extra_needed).toFloat = 0.85 →
  extra_needed = 4 := by
  intros correct already_wrong extra_needed h_correct h_wrong h_accuracy
  sorry

end NUMINAMATH_GPT_xiaoming_accuracy_l1309_130941


namespace NUMINAMATH_GPT_tens_digit_of_23_pow_2057_l1309_130959

theorem tens_digit_of_23_pow_2057 : (23^2057 % 100) / 10 % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_23_pow_2057_l1309_130959


namespace NUMINAMATH_GPT_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l1309_130968

open Nat

section LockCombination

-- Number of buttons
def num_buttons : ℕ := 10

-- Number of buttons that need to be pressed simultaneously
def combo_buttons : ℕ := 3

-- Total number of combinations
def total_combinations : ℕ := Nat.choose num_buttons combo_buttons

-- Time for each attempt
def time_per_attempt : ℕ := 2

-- Part (a): Total time to definitely get inside
theorem time_to_get_inside : Nat.succ (total_combinations * time_per_attempt) = 240 := by
  sorry

-- Part (b): Average time to get inside
theorem average_time_to_get_inside : (1 + total_combinations) * time_per_attempt = 242 := by
  sorry

-- Part (c): Probability to get inside in less than a minute
theorem probability_to_get_inside_in_less_than_a_minute : 29 / total_combinations = 29 / 120 := by
  sorry

end LockCombination

end NUMINAMATH_GPT_time_to_get_inside_average_time_to_get_inside_probability_to_get_inside_in_less_than_a_minute_l1309_130968


namespace NUMINAMATH_GPT_boy_usual_time_l1309_130995

noncomputable def usual_rate (R : ℝ) := R
noncomputable def usual_time (T : ℝ) := T
noncomputable def faster_rate (R : ℝ) := (7 / 6) * R
noncomputable def faster_time (T : ℝ) := T - 5

theorem boy_usual_time
  (R : ℝ) (T : ℝ) 
  (h1 : usual_rate R * usual_time T = faster_rate R * faster_time T) :
  T = 35 :=
by 
  unfold usual_rate usual_time faster_rate faster_time at h1
  sorry

end NUMINAMATH_GPT_boy_usual_time_l1309_130995


namespace NUMINAMATH_GPT_part1_part2_l1309_130986

open Set

/-- Define sets A and B as per given conditions --/
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

/-- Part 1: Prove the intersection and union with complements --/
theorem part1 :
  A ∩ B = {x | 3 ≤ x ∧ x < 6} ∧ (compl B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by {
  sorry
}

/-- Part 2: Given C ⊆ B, prove the constraints on a --/
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem part2 (a : ℝ) (h : C a ⊆ B) : 2 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l1309_130986


namespace NUMINAMATH_GPT_residue_neg_1234_mod_32_l1309_130940

theorem residue_neg_1234_mod_32 : -1234 % 32 = 14 := 
by sorry

end NUMINAMATH_GPT_residue_neg_1234_mod_32_l1309_130940


namespace NUMINAMATH_GPT_father_age_l1309_130908

variable (F S x : ℕ)

-- Conditions
axiom h1 : F + S = 75
axiom h2 : F = 8 * (S - x)
axiom h3 : F - x = S

-- Theorem to prove
theorem father_age : F = 48 :=
sorry

end NUMINAMATH_GPT_father_age_l1309_130908


namespace NUMINAMATH_GPT_cost_price_equals_selling_price_l1309_130963

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (hp : C > 0) (profit : ℝ := 0.25) (h : 30 * C = (1 + profit) * C * x) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_equals_selling_price_l1309_130963


namespace NUMINAMATH_GPT_line_contains_point_l1309_130946

theorem line_contains_point {
    k : ℝ
} :
  (2 - k * 3 = -4 * 1) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_line_contains_point_l1309_130946


namespace NUMINAMATH_GPT_intersection_points_l1309_130994

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def parabola2 (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem intersection_points :
  {p : ℝ × ℝ |
    (∃ x : ℝ, p = (x, parabola1 x) ∧ parabola1 x = parabola2 x)} =
  { 
    ( (3 + Real.sqrt 13) / 4, (74 + 14 * Real.sqrt 13) / 16 ),
    ( (3 - Real.sqrt 13) / 4, (74 - 14 * Real.sqrt 13) / 16 )
  } := sorry

end NUMINAMATH_GPT_intersection_points_l1309_130994


namespace NUMINAMATH_GPT_math_problem_solution_l1309_130977

noncomputable def a_range : Set ℝ := {a : ℝ | (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)}

theorem math_problem_solution (a : ℝ) :
  (1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∨ ((a - 3)^2 - 4 < 0)
  ∧ ¬((1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∧ ((a - 3)^2 - 4 < 0)) →
  a ∈ a_range :=
sorry

end NUMINAMATH_GPT_math_problem_solution_l1309_130977


namespace NUMINAMATH_GPT_proof_problem_l1309_130983

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 * n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  3 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ :=
  sequence_b (sequence_a n)

theorem proof_problem :
  sequence_c 2017 = 27 ^ 2017 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1309_130983


namespace NUMINAMATH_GPT_enemy_defeat_points_l1309_130923

theorem enemy_defeat_points 
    (points_per_enemy : ℕ) (total_enemies : ℕ) (undefeated_enemies : ℕ) (defeated : ℕ) (points_earned : ℕ) :
    points_per_enemy = 8 →
    total_enemies = 7 →
    undefeated_enemies = 2 →
    defeated = total_enemies - undefeated_enemies →
    points_earned = defeated * points_per_enemy →
    points_earned = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_enemy_defeat_points_l1309_130923


namespace NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l1309_130987

-- Definitions of conditions
variables (k b x y : ℝ)
variable  (h₁ : k > 0) -- condition k > 0
variable  (h₂ : b < 0) -- condition b < 0


theorem line_does_not_pass_second_quadrant : 
  ¬∃ (x y : ℝ), (x < 0 ∧ y > 0) ∧ (y = k * x + b) :=
sorry

end NUMINAMATH_GPT_line_does_not_pass_second_quadrant_l1309_130987


namespace NUMINAMATH_GPT_units_digit_G1000_l1309_130903

def Gn (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G1000 : (Gn 1000) % 10 = 2 :=
by sorry

end NUMINAMATH_GPT_units_digit_G1000_l1309_130903


namespace NUMINAMATH_GPT_total_elephants_l1309_130904

-- Define the conditions in Lean
def G (W : ℕ) : ℕ := 3 * W
def N (G : ℕ) : ℕ := 5 * G
def W : ℕ := 70

-- Define the statement to prove
theorem total_elephants :
  G W + W + N (G W) = 1330 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_total_elephants_l1309_130904


namespace NUMINAMATH_GPT_fourth_rectangle_area_is_112_l1309_130978

def area_of_fourth_rectangle (length : ℕ) (width : ℕ) (area1 : ℕ) (area2 : ℕ) (area3 : ℕ) : ℕ :=
  length * width - area1 - area2 - area3

theorem fourth_rectangle_area_is_112 :
  area_of_fourth_rectangle 20 12 24 48 36 = 112 :=
by
  sorry

end NUMINAMATH_GPT_fourth_rectangle_area_is_112_l1309_130978


namespace NUMINAMATH_GPT_ben_time_to_school_l1309_130920

/-- Amy's steps per minute -/
def amy_steps_per_minute : ℕ := 80

/-- Length of each of Amy's steps in cm -/
def amy_step_length : ℕ := 70

/-- Time taken by Amy to reach school in minutes -/
def amy_time_to_school : ℕ := 20

/-- Ben's steps per minute -/
def ben_steps_per_minute : ℕ := 120

/-- Length of each of Ben's steps in cm -/
def ben_step_length : ℕ := 50

/-- Given the above conditions, we aim to prove that Ben takes 18 2/3 minutes to reach school. -/
theorem ben_time_to_school : (112000 / 6000 : ℚ) = 18 + 2 / 3 := 
by sorry

end NUMINAMATH_GPT_ben_time_to_school_l1309_130920


namespace NUMINAMATH_GPT_max_value_of_8a_5b_15c_l1309_130991

theorem max_value_of_8a_5b_15c (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  8*a + 5*b + 15*c ≤ (Real.sqrt 115) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_8a_5b_15c_l1309_130991


namespace NUMINAMATH_GPT_amount_due_years_l1309_130992

noncomputable def years_due (PV FV : ℝ) (r : ℝ) : ℝ :=
  (Real.log (FV / PV)) / (Real.log (1 + r))

theorem amount_due_years : 
  years_due 200 242 0.10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_amount_due_years_l1309_130992


namespace NUMINAMATH_GPT_max_M_l1309_130955

noncomputable def conditions (x y z u : ℝ) : Prop :=
  (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) ∧ (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < u) ∧ (z ≥ y)

theorem max_M (x y z u : ℝ) : conditions x y z u → ∃ M : ℝ, M = 6 + 4 * Real.sqrt 2 ∧ M ≤ z / y :=
by {
  sorry
}

end NUMINAMATH_GPT_max_M_l1309_130955


namespace NUMINAMATH_GPT_julie_can_print_100_newspapers_l1309_130915

def num_boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

theorem julie_can_print_100_newspapers :
  (num_boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end NUMINAMATH_GPT_julie_can_print_100_newspapers_l1309_130915


namespace NUMINAMATH_GPT_num_proper_subsets_of_A_l1309_130965

open Set

def A : Finset ℕ := {2, 3}

theorem num_proper_subsets_of_A : (A.powerset \ {A, ∅}).card = 3 := by
  sorry

end NUMINAMATH_GPT_num_proper_subsets_of_A_l1309_130965


namespace NUMINAMATH_GPT_regular_price_of_polo_shirt_l1309_130934

/--
Zane purchases 2 polo shirts from the 40% off rack at the men's store. 
The polo shirts are priced at a certain amount at the regular price. 
He paid $60 for the shirts. 
Prove that the regular price of each polo shirt is $50.
-/
theorem regular_price_of_polo_shirt (P : ℝ) 
  (h1 : ∀ (x : ℝ), x = 0.6 * P → 2 * x = 60) : 
  P = 50 :=
sorry

end NUMINAMATH_GPT_regular_price_of_polo_shirt_l1309_130934


namespace NUMINAMATH_GPT_find_theta_l1309_130905

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (x θ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + θ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_theta (θ : ℝ) : 
  (∀ x, g x θ = g (-x) θ) → θ = Real.pi / 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_theta_l1309_130905
