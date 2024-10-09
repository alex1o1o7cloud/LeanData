import Mathlib

namespace more_likely_condition_l1453_145310

-- Definitions for the problem
def total_placements (n : ℕ) := n * n * (n * n - 1)

def not_same_intersection_placements (n : ℕ) := n * n * (n * n - 1)

def same_row_or_column_exclusions (n : ℕ) := 2 * n * (n - 1) * n

def not_same_street_placements (n : ℕ) := total_placements n - same_row_or_column_exclusions n

def probability_not_same_intersection (n : ℕ) := not_same_intersection_placements n / total_placements n

def probability_not_same_street (n : ℕ) := not_same_street_placements n / total_placements n

-- Main proposition
theorem more_likely_condition (n : ℕ) (h : n = 7) :
  probability_not_same_intersection n > probability_not_same_street n := 
by 
  sorry

end more_likely_condition_l1453_145310


namespace scientific_notation_of_12000_l1453_145329

theorem scientific_notation_of_12000 : 12000 = 1.2 * 10^4 := 
by sorry

end scientific_notation_of_12000_l1453_145329


namespace no_int_x_divisible_by_169_l1453_145389

theorem no_int_x_divisible_by_169 (x : ℤ) : ¬ (169 ∣ (x^2 + 5 * x + 16)) := by
  sorry

end no_int_x_divisible_by_169_l1453_145389


namespace sixtieth_term_of_arithmetic_sequence_l1453_145353

theorem sixtieth_term_of_arithmetic_sequence (a1 a15 : ℚ) (d : ℚ) (h1 : a1 = 7) (h2 : a15 = 37)
  (h3 : a15 = a1 + 14 * d) : a1 + 59 * d = 134.5 := by
  sorry

end sixtieth_term_of_arithmetic_sequence_l1453_145353


namespace apples_per_pie_l1453_145340

theorem apples_per_pie (total_apples : ℕ) (apples_given : ℕ) (pies : ℕ) : 
  total_apples = 47 ∧ apples_given = 27 ∧ pies = 5 →
  (total_apples - apples_given) / pies = 4 :=
by
  intros h
  sorry

end apples_per_pie_l1453_145340


namespace marathon_speed_ratio_l1453_145357

theorem marathon_speed_ratio (M D : ℝ) (J : ℝ) (H1 : D = 9) (H2 : J = 4/3 * M) (H3 : M + J + D = 23) :
  D / M = 3 / 2 :=
by
  sorry

end marathon_speed_ratio_l1453_145357


namespace simplify_expression_l1453_145361

variable (p q r : ℝ)
variable (hp : p ≠ 2)
variable (hq : q ≠ 3)
variable (hr : r ≠ 4)

theorem simplify_expression : 
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 :=
by
  -- Skipping the proof using sorry
  sorry

end simplify_expression_l1453_145361


namespace max_full_marks_probability_l1453_145347

-- Define the total number of mock exams
def total_mock_exams : ℕ := 20
-- Define the number of full marks scored in mock exams
def full_marks_in_mocks : ℕ := 8

-- Define the probability of event A (scoring full marks in the first test)
def P_A : ℚ := full_marks_in_mocks / total_mock_exams

-- Define the probability of not scoring full marks in the first test
def P_neg_A : ℚ := 1 - P_A

-- Define the probability of event B (scoring full marks in the second test)
def P_B : ℚ := 1 / 2

-- Define the maximum probability of scoring full marks in either the first or the second test
def max_probability : ℚ := P_A + P_neg_A * P_B

-- The main theorem conjecture
theorem max_full_marks_probability :
  max_probability = 7 / 10 :=
by
  -- Inserting placeholder to skip the proof for now
  sorry

end max_full_marks_probability_l1453_145347


namespace resulting_figure_perimeter_l1453_145355

def original_square_side : ℕ := 100

def original_square_area : ℕ := original_square_side * original_square_side

def rect1_side1 : ℕ := original_square_side
def rect1_side2 : ℕ := original_square_side / 2

def rect2_side1 : ℕ := original_square_side
def rect2_side2 : ℕ := original_square_side / 2

def new_figure_perimeter : ℕ :=
  3 * original_square_side + 4 * (original_square_side / 2)

theorem resulting_figure_perimeter :
  new_figure_perimeter = 500 :=
by {
    sorry
}

end resulting_figure_perimeter_l1453_145355


namespace kim_points_correct_l1453_145371

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end kim_points_correct_l1453_145371


namespace find_m_n_l1453_145319

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem find_m_n (m n : ℕ) (h1 : binom (n+1) (m+1) / binom (n+1) m = 5 / 3) 
  (h2 : binom (n+1) m / binom (n+1) (m-1) = 5 / 3) : m = 3 ∧ n = 6 :=
  sorry

end find_m_n_l1453_145319


namespace pasta_needed_for_family_reunion_l1453_145375

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end pasta_needed_for_family_reunion_l1453_145375


namespace mul_97_97_eq_9409_l1453_145316

theorem mul_97_97_eq_9409 : 97 * 97 = 9409 := 
  sorry

end mul_97_97_eq_9409_l1453_145316


namespace minimum_value_of_f_range_of_a_l1453_145373

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3

theorem minimum_value_of_f :
  ∃ x_min : ℝ, ∀ x : ℝ, 0 < x → f x ≥ -1/Real.exp 1 := sorry -- This statement asserts that the minimum value of f(x) is -1/e.

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 := sorry -- This statement asserts that if 2f(x) ≥ g(x) for all x > 0, then a is at most 4.

end minimum_value_of_f_range_of_a_l1453_145373


namespace sum_of_transformed_numbers_l1453_145332

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
    3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l1453_145332


namespace james_muffins_baked_l1453_145391

-- Define the number of muffins Arthur baked
def muffinsArthur : ℕ := 115

-- Define the multiplication factor
def multiplicationFactor : ℕ := 12

-- Define the number of muffins James baked
def muffinsJames : ℕ := muffinsArthur * multiplicationFactor

-- The theorem that needs to be proved
theorem james_muffins_baked : muffinsJames = 1380 :=
by
  sorry

end james_muffins_baked_l1453_145391


namespace sum_and_product_of_roots_cube_l1453_145397

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end sum_and_product_of_roots_cube_l1453_145397


namespace factorize_expression_l1453_145328

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l1453_145328


namespace option_B_equals_six_l1453_145304

theorem option_B_equals_six :
  (3 - (-3)) = 6 :=
by
  sorry

end option_B_equals_six_l1453_145304


namespace TV_height_l1453_145313

theorem TV_height (Area Width Height : ℝ) (h_area : Area = 21) (h_width : Width = 3) (h_area_def : Area = Width * Height) : Height = 7 := 
by
  sorry

end TV_height_l1453_145313


namespace ball_bounce_height_l1453_145330

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (3 / 4 : ℝ)^k < 2) ∧ ∀ n < k, ¬ (20 * (3 / 4 : ℝ)^n < 2) :=
sorry

end ball_bounce_height_l1453_145330


namespace volleyball_team_girls_l1453_145383

theorem volleyball_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : G = 15 :=
sorry

end volleyball_team_girls_l1453_145383


namespace jill_third_month_days_l1453_145372

theorem jill_third_month_days :
  ∀ (days : ℕ),
    (earnings_first_month : ℕ) = 10 * 30 →
    (earnings_second_month : ℕ) = 20 * 30 →
    (total_earnings : ℕ) = 1200 →
    (total_earnings_two_months : ℕ) = earnings_first_month + earnings_second_month →
    (earnings_third_month : ℕ) = total_earnings - total_earnings_two_months →
    earnings_third_month = 300 →
    days = earnings_third_month / 20 →
    days = 15 := 
sorry

end jill_third_month_days_l1453_145372


namespace product_of_number_and_sum_of_digits_l1453_145352

-- Definitions according to the conditions
def units_digit (a b : ℕ) : Prop := b = a + 2
def number_equals_24 (a b : ℕ) : Prop := 10 * a + b = 24

-- The main statement to prove the product of the number and the sum of its digits
theorem product_of_number_and_sum_of_digits :
  ∃ (a b : ℕ), units_digit a b ∧ number_equals_24 a b ∧ (24 * (a + b) = 144) :=
sorry

end product_of_number_and_sum_of_digits_l1453_145352


namespace bucket_full_weight_l1453_145363

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
    (h1 : x + (3/4) * y = p) 
    (h2 : x + (1/3) * y = q) : 
    x + y = (8 * p - 3 * q) / 5 :=
sorry

end bucket_full_weight_l1453_145363


namespace problem_a_problem_b_problem_c_problem_d_l1453_145387

-- Problem a
theorem problem_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 :=
by sorry

-- Problem b
theorem problem_b (a : ℝ) : (2 * a + 3) * (2 * a - 3) = 4 * a^2 - 9 :=
by sorry

-- Problem c
theorem problem_c (m n : ℝ) : (m^3 - n^5) * (n^5 + m^3) = m^6 - n^10 :=
by sorry

-- Problem d
theorem problem_d (m n : ℝ) : (3 * m^2 - 5 * n^2) * (3 * m^2 + 5 * n^2) = 9 * m^4 - 25 * n^4 :=
by sorry

end problem_a_problem_b_problem_c_problem_d_l1453_145387


namespace book_cost_proof_l1453_145331

variable (C1 C2 : ℝ)

theorem book_cost_proof (h1 : C1 + C2 = 460)
                        (h2 : C1 * 0.85 = C2 * 1.19) :
    C1 = 268.53 := by
  sorry

end book_cost_proof_l1453_145331


namespace tom_saves_promotion_l1453_145346

open Nat

theorem tom_saves_promotion (price : ℕ) (disc_percent : ℕ) (discount_amount : ℕ) 
    (promotion_x_cost second_pair_cost_promo_x promotion_y_cost promotion_savings : ℕ) 
    (h1 : price = 50)
    (h2 : disc_percent = 40)
    (h3 : discount_amount = 15)
    (h4 : second_pair_cost_promo_x = price - (price * disc_percent / 100))
    (h5 : promotion_x_cost = price + second_pair_cost_promo_x)
    (h6 : promotion_y_cost = price + (price - discount_amount))
    (h7 : promotion_savings = promotion_y_cost - promotion_x_cost) :
  promotion_savings = 5 :=
by
  sorry

end tom_saves_promotion_l1453_145346


namespace abigail_initial_money_l1453_145393

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end abigail_initial_money_l1453_145393


namespace alice_travel_time_l1453_145326

theorem alice_travel_time (distance_AB : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) (max_time_diff_hr : ℝ) (time_conversion : ℝ) :
  distance_AB = 60 →
  bob_speed = 40 →
  alice_speed = 60 →
  max_time_diff_hr = 0.5 →
  time_conversion = 60 →
  max_time_diff_hr * time_conversion = 30 :=
by
  intros
  sorry

end alice_travel_time_l1453_145326


namespace cylindrical_coords_of_point_l1453_145374

theorem cylindrical_coords_of_point :
  ∃ (r θ z : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
                 r = Real.sqrt (3^2 + 3^2) ∧
                 θ = Real.arctan (3 / 3) ∧
                 z = 4 ∧
                 (3, 3, 4) = (r * Real.cos θ, r * Real.sin θ, z) :=
by
  sorry

end cylindrical_coords_of_point_l1453_145374


namespace range_of_m_l1453_145318

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l1453_145318


namespace area_of_gray_region_l1453_145312

theorem area_of_gray_region :
  (radius_smaller = (2 : ℝ) / 2) →
  (radius_larger = 4 * radius_smaller) →
  (gray_area = π * radius_larger ^ 2 - π * radius_smaller ^ 2) →
  gray_area = 15 * π :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  sorry

end area_of_gray_region_l1453_145312


namespace sum_gcd_lcm_l1453_145368

theorem sum_gcd_lcm (a b : ℕ) (h_a : a = 75) (h_b : b = 4500) :
  Nat.gcd a b + Nat.lcm a b = 4575 := by
  sorry

end sum_gcd_lcm_l1453_145368


namespace eval_fraction_expression_l1453_145341
noncomputable def inner_expr := 2 + 2
noncomputable def middle_expr := 2 + (1 / inner_expr)
noncomputable def outer_expr := 2 + (1 / middle_expr)

theorem eval_fraction_expression : outer_expr = 22 / 9 := by
  sorry

end eval_fraction_expression_l1453_145341


namespace cube_volume_equality_l1453_145305

open BigOperators Real

-- Definitions
def initial_volume : ℝ := 1

def removed_volume (x : ℝ) : ℝ := x^2

def removed_volume_with_overlap (x y : ℝ) : ℝ := x^2 - (x^2 * y)

def remaining_volume (a b c : ℝ) : ℝ := 
  initial_volume - removed_volume c - removed_volume_with_overlap b c - removed_volume_with_overlap a c - removed_volume_with_overlap a b + (c^2 * b)

-- Main theorem to prove
theorem cube_volume_equality (c b a : ℝ) (hcb : c < b) (hba : b < a) (ha1 : a < 1):
  (c = 1 / 2) ∧ 
  (b = (1 + Real.sqrt 17) / 8) ∧ 
  (a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64) :=
sorry

end cube_volume_equality_l1453_145305


namespace polygon_diagonals_l1453_145354

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 4) : n = 7 :=
sorry

end polygon_diagonals_l1453_145354


namespace speed_in_still_water_l1453_145322

-- Define the given conditions
def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

-- State the theorem to be proven
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 40 := by
  -- Proof omitted
  sorry

end speed_in_still_water_l1453_145322


namespace profit_percent_calc_l1453_145344

theorem profit_percent_calc (SP CP : ℝ) (h : CP = 0.25 * SP) : (SP - CP) / CP * 100 = 300 :=
by
  sorry

end profit_percent_calc_l1453_145344


namespace moles_of_H2O_formed_l1453_145338

theorem moles_of_H2O_formed
  (moles_H2SO4 : ℕ)
  (moles_H2O : ℕ)
  (H : moles_H2SO4 = 3)
  (H' : moles_H2O = 3) :
  moles_H2O = 3 :=
by
  sorry

end moles_of_H2O_formed_l1453_145338


namespace line_intersects_parabola_exactly_once_at_m_l1453_145311

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l1453_145311


namespace ratio_of_e_to_l_l1453_145308

-- Define the conditions
def e (S : ℕ) : ℕ := 4 * S
def l (S : ℕ) : ℕ := 8 * S

-- Prove the main statement
theorem ratio_of_e_to_l (S : ℕ) (h_e : e S = 4 * S) (h_l : l S = 8 * S) : e S / gcd (e S) (l S) / l S / gcd (e S) (l S) = 1 / 2 := by
  sorry

end ratio_of_e_to_l_l1453_145308


namespace no_such_function_exists_l1453_145342

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := 
by
  sorry

end no_such_function_exists_l1453_145342


namespace sin_alpha_of_terminal_side_l1453_145381

theorem sin_alpha_of_terminal_side (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (5, 12)) :
  Real.sin α = 12 / 13 := sorry

end sin_alpha_of_terminal_side_l1453_145381


namespace soccer_points_l1453_145394

def total_points (wins draws losses : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : ℕ :=
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

theorem soccer_points : total_points 14 4 2 3 1 0 = 46 :=
by
  sorry

end soccer_points_l1453_145394


namespace combined_earnings_l1453_145359

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l1453_145359


namespace dot_product_a_b_l1453_145343

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (4, -3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Statement of the theorem to prove
theorem dot_product_a_b : dot_product vector_a vector_b = -1 := 
by sorry

end dot_product_a_b_l1453_145343


namespace jellybeans_in_jar_l1453_145356

theorem jellybeans_in_jar (num_kids_normal : ℕ) (num_absent : ℕ) (num_jellybeans_each : ℕ) (num_leftover : ℕ) 
  (h1 : num_kids_normal = 24) (h2 : num_absent = 2) (h3 : num_jellybeans_each = 3) (h4 : num_leftover = 34) : 
  (num_kids_normal - num_absent) * num_jellybeans_each + num_leftover = 100 :=
by sorry

end jellybeans_in_jar_l1453_145356


namespace sin_double_angle_identity_l1453_145348

theorem sin_double_angle_identity: 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_double_angle_identity_l1453_145348


namespace angle_ratio_l1453_145395

theorem angle_ratio (BP BQ BM: ℝ) (ABC: ℝ) (quadrisect : BP = ABC/4 ∧ BQ = ABC)
  (bisect : BM = (3/4) * ABC / 2):
  (BM / (ABC / 4 + ABC / 4)) = 1 / 6 := by
    sorry

end angle_ratio_l1453_145395


namespace max_min_product_of_three_l1453_145379

open List

theorem max_min_product_of_three (s : List Int) (h : s = [-1, -2, 3, 4]) : 
  ∃ (max min : Int), 
    max = 8 ∧ min = -24 ∧ 
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≤ max) ∧
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≥ min) := 
by
  sorry

end max_min_product_of_three_l1453_145379


namespace large_planks_need_15_nails_l1453_145350

-- Definitions based on given conditions
def total_nails : ℕ := 20
def small_planks_nails : ℕ := 5

-- Question: How many nails do the large planks need together?
-- Prove that the large planks need 15 nails together given the conditions.
theorem large_planks_need_15_nails : total_nails - small_planks_nails = 15 :=
by
  sorry

end large_planks_need_15_nails_l1453_145350


namespace correlation_coefficient_l1453_145360

theorem correlation_coefficient (variation_explained_by_height : ℝ)
    (variation_explained_by_errors : ℝ)
    (total_variation : variation_explained_by_height + variation_explained_by_errors = 1)
    (percentage_explained_by_height : variation_explained_by_height = 0.71) :
  variation_explained_by_height = 0.71 := 
by
  sorry

end correlation_coefficient_l1453_145360


namespace cyclist_avg_speed_l1453_145398

theorem cyclist_avg_speed (d : ℝ) (h1 : d > 0) :
  let t_1 := d / 17
  let t_2 := d / 23
  let total_time := t_1 + t_2
  let total_distance := 2 * d
  (total_distance / total_time) = 19.55 :=
by
  -- Proof steps here
  sorry

end cyclist_avg_speed_l1453_145398


namespace find_x_y_l1453_145302

theorem find_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : (x * y / 7) ^ (3 / 2) = x) 
  (h2 : (x * y / 7) = y) : 
  x = 7 ∧ y = 7 ^ (2 / 3) :=
by
  sorry

end find_x_y_l1453_145302


namespace find_growth_rate_calculate_fourth_day_donation_l1453_145384

-- Define the conditions
def first_day_donation : ℝ := 3000
def third_day_donation : ℝ := 4320
def growth_rate (x : ℝ) : Prop := (1 + x)^2 = third_day_donation / first_day_donation

-- Since the problem states growth rate for second and third day is the same,
-- we need to find that rate which is equivalent to solving the above proposition for x.

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.2 := by
  sorry

-- Calculate the fourth day's donation based on the growth rate found.
def fourth_day_donation (third_day : ℝ) (growth_rate : ℝ) : ℝ :=
  third_day * (1 + growth_rate)

theorem calculate_fourth_day_donation : 
  ∀ x : ℝ, growth_rate x → x = 0.2 → fourth_day_donation third_day_donation x = 5184 := by 
  sorry

end find_growth_rate_calculate_fourth_day_donation_l1453_145384


namespace Chris_age_l1453_145315

theorem Chris_age (a b c : ℚ) 
  (h1 : a + b + c = 30)
  (h2 : c - 5 = 2 * a)
  (h3 : b = (3/4) * a - 1) :
  c = 263/11 := by
  sorry

end Chris_age_l1453_145315


namespace inequality_am_gm_l1453_145376

theorem inequality_am_gm (a b : ℝ) (p q : ℝ) (h1: a > 0) (h2: b > 0) (h3: p > 1) (h4: q > 1) (h5 : 1/p + 1/q = 1) : 
  a^(1/p) * b^(1/q) ≤ a/p + b/q :=
by
  sorry

end inequality_am_gm_l1453_145376


namespace tapA_fill_time_l1453_145388

-- Define the conditions
def fillTapA (t : ℕ) := 1 / t
def fillTapB := 1 / 40
def fillCombined (t : ℕ) := 9 * (fillTapA t + fillTapB)
def fillRemaining := 23 * fillTapB

-- Main theorem statement
theorem tapA_fill_time : ∀ (t : ℕ), fillCombined t + fillRemaining = 1 → t = 45 := by
  sorry

end tapA_fill_time_l1453_145388


namespace only_pairs_satisfying_conditions_l1453_145306

theorem only_pairs_satisfying_conditions (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (b^2 + b + 1) % a = 0 ∧ (a^2 + a + 1) % b = 0 → a = 1 ∧ b = 1 :=
by
  sorry

end only_pairs_satisfying_conditions_l1453_145306


namespace sum_of_x_l1453_145349

-- define the function f as an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- define the function f as strictly monotonic on the interval (0, +∞)
def is_strictly_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- define the main problem statement
theorem sum_of_x (f : ℝ → ℝ) (x : ℝ) (h1 : is_even_function f) (h2 : is_strictly_monotonic_on_positive f) (h3 : x ≠ 0)
  (hx : f (x^2 - 2*x - 1) = f (x + 1)) : 
  ∃ (x1 x2 x3 x4 : ℝ), (x1 + x2 + x3 + x4 = 4) ∧
                        (x1^2 - 3*x1 - 2 = 0) ∧
                        (x2^2 - 3*x2 - 2 = 0) ∧
                        (x3^2 - x3 = 0) ∧
                        (x4^2 - x4 = 0) :=
sorry

end sum_of_x_l1453_145349


namespace sophie_saves_money_by_using_wool_balls_l1453_145385

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l1453_145385


namespace probability_not_red_l1453_145380

theorem probability_not_red (h : odds_red = 1 / 3) : probability_not_red_card = 3 / 4 :=
by
  sorry

end probability_not_red_l1453_145380


namespace garden_area_l1453_145300

def radius : ℝ := 0.6
def pi_approx : ℝ := 3
def circle_area (r : ℝ) (π : ℝ) := π * r^2

theorem garden_area : circle_area radius pi_approx = 1.08 :=
by
  sorry

end garden_area_l1453_145300


namespace max_area_rectangle_l1453_145309

theorem max_area_rectangle (P : ℝ) (hP : P = 60) (a b : ℝ) (h1 : b = 3 * a) (h2 : 2 * a + 2 * b = P) : a * b = 168.75 :=
by
  sorry

end max_area_rectangle_l1453_145309


namespace find_function_l1453_145367

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1) :
  ∀ x : ℝ, f x = if x ≠ 0.5 then 1 / (0.5 - x) else 0.5 :=
by
  sorry

end find_function_l1453_145367


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l1453_145377

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l1453_145377


namespace minimum_value_of_fraction_l1453_145334

theorem minimum_value_of_fraction (x : ℝ) (hx : x > 10) : ∃ m, m = 30 ∧ ∀ y > 10, (y * y) / (y - 10) ≥ m :=
by 
  sorry

end minimum_value_of_fraction_l1453_145334


namespace chords_from_nine_points_l1453_145364

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l1453_145364


namespace jacqueline_guavas_l1453_145327

theorem jacqueline_guavas 
  (G : ℕ) 
  (plums : ℕ := 16) 
  (apples : ℕ := 21) 
  (given : ℕ := 40) 
  (remaining : ℕ := 15) 
  (initial_fruits : ℕ := plums + G + apples)
  (total_fruits_after_given : ℕ := remaining + given) : 
  initial_fruits = total_fruits_after_given → G = 18 := 
by
  intro h
  sorry

end jacqueline_guavas_l1453_145327


namespace problem_statement_l1453_145362

theorem problem_statement (a b : ℝ) (h : (1 / a + 1 / b) / (1 / a - 1 / b) = 2023) : (a + b) / (a - b) = 2023 :=
by
  sorry

end problem_statement_l1453_145362


namespace solve_equation_l1453_145390

theorem solve_equation : ∀ x : ℝ, (10 - x) ^ 2 = 4 * x ^ 2 ↔ x = 10 / 3 ∨ x = -10 :=
by
  intros x
  sorry

end solve_equation_l1453_145390


namespace servant_received_amount_l1453_145321

def annual_salary := 900
def uniform_price := 100
def fraction_of_year_served := 3 / 4

theorem servant_received_amount :
  annual_salary * fraction_of_year_served + uniform_price = 775 := by
  sorry

end servant_received_amount_l1453_145321


namespace sequence_periodic_l1453_145320

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_periodic :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3 :=
by
  sorry

end sequence_periodic_l1453_145320


namespace farmer_goats_sheep_unique_solution_l1453_145317

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution_l1453_145317


namespace complex_number_pure_imaginary_l1453_145386

theorem complex_number_pure_imaginary (a : ℝ) 
  (h1 : ∃ a : ℝ, (a^2 - 2*a - 3 = 0) ∧ (a + 1 ≠ 0)) 
  : a = 3 := sorry

end complex_number_pure_imaginary_l1453_145386


namespace spring_bud_cup_eq_289_l1453_145365

theorem spring_bud_cup_eq_289 (x : ℕ) (h : x + x = 578) : x = 289 :=
sorry

end spring_bud_cup_eq_289_l1453_145365


namespace vertex_in_first_quadrant_l1453_145396

theorem vertex_in_first_quadrant (a : ℝ) (h : a > 1) : 
  let x_vertex := (a + 1) / 2
  let y_vertex := (a + 3)^2 / 4
  x_vertex > 0 ∧ y_vertex > 0 := 
by
  sorry

end vertex_in_first_quadrant_l1453_145396


namespace mod_inverse_3_40_l1453_145303

theorem mod_inverse_3_40 : 3 * 27 % 40 = 1 := by
  sorry

end mod_inverse_3_40_l1453_145303


namespace evaluate_sum_of_squares_l1453_145336

theorem evaluate_sum_of_squares 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + y = 25) : (x + y)^2 = 49 :=
  sorry

end evaluate_sum_of_squares_l1453_145336


namespace knights_statements_l1453_145358

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end knights_statements_l1453_145358


namespace sequence_result_l1453_145323

theorem sequence_result :
  (1 + 2)^2 + 1 = 10 ∧
  (2 + 3)^2 + 1 = 26 ∧
  (4 + 5)^2 + 1 = 82 →
  (3 + 4)^2 + 1 = 50 :=
by sorry

end sequence_result_l1453_145323


namespace alloy_problem_l1453_145324

theorem alloy_problem (x : ℝ) (h1 : 0.12 * x + 0.08 * 30 = 0.09333333333333334 * (x + 30)) : x = 15 :=
by
  sorry

end alloy_problem_l1453_145324


namespace x_squared_minus_y_squared_l1453_145369

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l1453_145369


namespace find_k_from_polynomial_l1453_145392

theorem find_k_from_polynomial :
  ∃ (k : ℝ),
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₂ * x₃ * x₄ = -1984 ∧
    x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄ = k ∧
    x₁ + x₂ + x₃ + x₄ = 18 ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32))
  → k = 86 :=
by
  sorry

end find_k_from_polynomial_l1453_145392


namespace part1_part2_l1453_145307

theorem part1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (1 - 4 / (2 * a^0 + a)) = 0) : a = 2 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x : ℝ, (2^x + 1) * (1 - 2 / (2^x + 1)) + k = 0) : k < 1 :=
sorry

end part1_part2_l1453_145307


namespace factor_correct_l1453_145335

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l1453_145335


namespace parallel_lines_solution_l1453_145314

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (x + a * y + 6 = 0) → (a - 2) * x + 3 * y + 2 * a = 0) → (a = -1) :=
by
  intro h
  -- Add more formal argument insights if needed
  sorry

end parallel_lines_solution_l1453_145314


namespace expenditure_increase_l1453_145325

theorem expenditure_increase (x : ℝ) (h₁ : 3 * x / (3 * x + 2 * x) = 3 / 5)
  (h₂ : 2 * x / (3 * x + 2 * x) = 2 / 5)
  (h₃ : ((5 * x) + 0.15 * (5 * x)) = 5.75 * x) 
  (h₄ : (2 * x + 0.06 * 2 * x) = 2.12 * x) 
  : ((3.63 * x - 3 * x) / (3 * x) * 100) = 21 := 
  by
  sorry

end expenditure_increase_l1453_145325


namespace greatest_b_no_minus_six_in_range_l1453_145378

open Real

theorem greatest_b_no_minus_six_in_range :
  ∃ (b : ℤ), (b = 8) → (¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 15 = -6) :=
by {
  -- We need to find the largest integer b such that -6 is not in the range of f(x) = x^2 + bx + 15
  sorry
}

end greatest_b_no_minus_six_in_range_l1453_145378


namespace intersection_sets_l1453_145382

theorem intersection_sets (x : ℝ) :
  let M := {x | 2 * x - x^2 ≥ 0 }
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_sets_l1453_145382


namespace length_of_de_l1453_145345

theorem length_of_de
  {a b c d e : ℝ} 
  (h1 : b - a = 5) 
  (h2 : c - a = 11) 
  (h3 : e - a = 22) 
  (h4 : c - b = 2 * (d - c)) :
  e - d = 8 :=
by 
  sorry

end length_of_de_l1453_145345


namespace find_years_lent_to_B_l1453_145370

def principal_B := 5000
def principal_C := 3000
def rate := 8
def time_C := 4
def total_interest := 1760

-- Interest calculation for B
def interest_B (n : ℕ) := (principal_B * rate * n) / 100

-- Interest calculation for C (constant time of 4 years)
def interest_C := (principal_C * rate * time_C) / 100

-- Total interest received
def total_interest_received (n : ℕ) := interest_B n + interest_C

theorem find_years_lent_to_B (n : ℕ) (h : total_interest_received n = total_interest) : n = 2 :=
by
  sorry

end find_years_lent_to_B_l1453_145370


namespace domain_of_log_function_l1453_145351

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_log_function : {
  x : ℝ // ∃ y : ℝ, f y = x
} = { x : ℝ | x > 1 / 2 } := by
sorry

end domain_of_log_function_l1453_145351


namespace cross_section_quadrilateral_is_cylinder_l1453_145399

-- Definition of the solids
inductive Solid
| cone
| cylinder
| sphere

-- Predicate for the cross-section being a quadrilateral
def is_quadrilateral_cross_section (solid : Solid) : Prop :=
  match solid with
  | Solid.cylinder => true
  | Solid.cone     => false
  | Solid.sphere   => false

-- Main theorem statement
theorem cross_section_quadrilateral_is_cylinder (s : Solid) :
  is_quadrilateral_cross_section s → s = Solid.cylinder :=
by
  cases s
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]

end cross_section_quadrilateral_is_cylinder_l1453_145399


namespace find_x_l1453_145337

theorem find_x (x : ℝ) (h : x ^ 2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
sorry

end find_x_l1453_145337


namespace find_x_l1453_145301

theorem find_x (x : ℕ) : (x % 9 = 0) ∧ (x^2 > 144) ∧ (x < 30) → (x = 18 ∨ x = 27) :=
by 
  sorry

end find_x_l1453_145301


namespace sum_of_a2_and_a3_l1453_145333

theorem sum_of_a2_and_a3 (S : ℕ → ℕ) (hS : ∀ n, S n = 3^n + 1) :
  S 3 - S 1 = 24 :=
by
  sorry

end sum_of_a2_and_a3_l1453_145333


namespace pie_piece_cost_l1453_145366

theorem pie_piece_cost (pieces_per_pie : ℕ) (pies_per_hour : ℕ) (total_earnings : ℝ) :
  pieces_per_pie = 3 → pies_per_hour = 12 → total_earnings = 138 →
  (total_earnings / (pieces_per_pie * pies_per_hour)) = 3.83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pie_piece_cost_l1453_145366


namespace charles_pictures_after_work_l1453_145339

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l1453_145339
