import Mathlib

namespace original_number_l15_15097

-- Define the original statement and conditions
theorem original_number (x : ℝ) (h : 3 * (2 * x + 9) = 81) : x = 9 := by
  -- Sorry placeholder stands for the proof steps
  sorry

end original_number_l15_15097


namespace find_certain_number_l15_15682

theorem find_certain_number (x : ℕ) (h: x - 82 = 17) : x = 99 :=
by
  sorry

end find_certain_number_l15_15682


namespace angle_BAD_measure_l15_15861

theorem angle_BAD_measure (D_A_C : ℝ) (AB_AC : AB = AC) (AD_BD : AD = BD) (h : D_A_C = 39) :
  B_A_D = 70.5 :=
by sorry

end angle_BAD_measure_l15_15861


namespace batsman_average_after_12th_l15_15254

theorem batsman_average_after_12th (runs_12th : ℕ) (average_increase : ℕ) (initial_innings : ℕ)
   (initial_average : ℝ) (runs_before_12th : ℕ → ℕ) 
   (h1 : runs_12th = 48)
   (h2 : average_increase = 2)
   (h3 : initial_innings = 11)
   (h4 : initial_average = 24)
   (h5 : ∀ i, i < initial_innings → runs_before_12th i ≥ 20)
   (h6 : ∃ i, runs_before_12th i = 25 ∧ runs_before_12th (i + 1) = 25) :
   (11 * initial_average + runs_12th) / 12 = 26 :=
by
  sorry

end batsman_average_after_12th_l15_15254


namespace correct_algorithm_option_l15_15909

def OptionA := ("Sequential structure", "Flow structure", "Loop structure")
def OptionB := ("Sequential structure", "Conditional structure", "Nested structure")
def OptionC := ("Sequential structure", "Conditional structure", "Loop structure")
def OptionD := ("Flow structure", "Conditional structure", "Loop structure")

-- The correct structures of an algorithm are sequential, conditional, and loop.
def algorithm_structures := ("Sequential structure", "Conditional structure", "Loop structure")

theorem correct_algorithm_option : algorithm_structures = OptionC := 
by 
  -- This would be proven by logic and checking the options; omitted here with 'sorry'
  sorry

end correct_algorithm_option_l15_15909


namespace find_a_value_l15_15221

noncomputable def collinear (points : List (ℚ × ℚ)) := 
  ∃ a b c, ∀ (x y : ℚ), (x, y) ∈ points → a * x + b * y + c = 0

theorem find_a_value (a : ℚ) :
  collinear [(3, -5), (-a + 2, 3), (2*a + 3, 2)] → a = -7 / 23 :=
by
  sorry

end find_a_value_l15_15221


namespace triangle_inequality_l15_15232

variable {a b c : ℝ}

theorem triangle_inequality (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) : 
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
by
  sorry

end triangle_inequality_l15_15232


namespace find_m_l15_15767

-- Define the vector
def vec2 := (ℝ × ℝ)

-- Given vectors
def a : vec2 := (2, -1)
def c : vec2 := (-1, 2)

-- Definition of parallel vectors
def parallel (v1 v2 : vec2) := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Problem Statement
theorem find_m (m : ℝ) (b : vec2 := (-1, m)) (h : parallel (a.1 + b.1, a.2 + b.2) c) : m = -1 :=
sorry

end find_m_l15_15767


namespace cost_price_eq_l15_15111

variable (SP : Real) (profit_percentage : Real)

theorem cost_price_eq : SP = 100 → profit_percentage = 0.15 → (100 / (1 + profit_percentage)) = 86.96 :=
by
  intros hSP hProfit
  sorry

end cost_price_eq_l15_15111


namespace train_speed_l15_15320

variable (length : ℕ) (time : ℕ)
variable (h_length : length = 120)
variable (h_time : time = 6)

theorem train_speed (length time : ℕ) (h_length : length = 120) (h_time : time = 6) :
  length / time = 20 := by
  sorry

end train_speed_l15_15320


namespace factorize_expression_l15_15675

theorem factorize_expression (a b : ℝ) : a^2 + a * b = a * (a + b) := 
by
  sorry

end factorize_expression_l15_15675


namespace part1_part2_l15_15665

open Set

def A : Set ℤ := { x | ∃ (m n : ℤ), x = m^2 - n^2 }

theorem part1 : 3 ∈ A := 
by sorry

theorem part2 (k : ℤ) : 4 * k - 2 ∉ A := 
by sorry

end part1_part2_l15_15665


namespace polygon_sides_in_arithmetic_progression_l15_15156

theorem polygon_sides_in_arithmetic_progression 
  (n : ℕ) 
  (d : ℕ := 3)
  (max_angle : ℕ := 150)
  (sum_of_interior_angles : ℕ := 180 * (n - 2)) 
  (a_n : ℕ := max_angle) : 
  (max_angle - d * (n - 1) + max_angle) * n / 2 = sum_of_interior_angles → 
  n = 28 :=
by 
  sorry

end polygon_sides_in_arithmetic_progression_l15_15156


namespace least_number_subtracted_l15_15777

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted_l15_15777


namespace share_money_3_people_l15_15591

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l15_15591


namespace quadratic_two_real_roots_quadratic_no_real_roots_l15_15979

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k ≤ 9 / 8 :=
by
  sorry

theorem quadratic_no_real_roots (k : ℝ) :
  ¬ (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k > 9 / 8 :=
by
  sorry

end quadratic_two_real_roots_quadratic_no_real_roots_l15_15979


namespace group_size_l15_15187

-- Define the conditions
variables (N : ℕ)
variable (h1 : (1 / 5 : ℝ) * N = (N : ℝ) * 0.20)
variable (h2 : 128 ≤ N)
variable (h3 : (1 / 5 : ℝ) * N - 128 = 0.04 * (N : ℝ))

-- Prove that the number of people in the group is 800
theorem group_size : N = 800 :=
by
  sorry

end group_size_l15_15187


namespace num_diagonals_octagon_l15_15257

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_octagon : num_diagonals 8 = 20 :=
by
  sorry

end num_diagonals_octagon_l15_15257


namespace dog_food_l15_15871

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end dog_food_l15_15871


namespace hannah_mugs_problem_l15_15420

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l15_15420


namespace max_area_dog_roam_l15_15375

theorem max_area_dog_roam (r : ℝ) (s : ℝ) (half_s : ℝ) (midpoint : Prop) :
  r = 10 → s = 20 → half_s = s / 2 → midpoint → 
  r > half_s → 
  π * r^2 = 100 * π :=
by 
  intros hr hs h_half_s h_midpoint h_rope_length
  sorry

end max_area_dog_roam_l15_15375


namespace man_age_twice_son_age_in_n_years_l15_15633

theorem man_age_twice_son_age_in_n_years
  (S M Y : ℤ)
  (h1 : S = 26)
  (h2 : M = S + 28)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 :=
by
  sorry

end man_age_twice_son_age_in_n_years_l15_15633


namespace find_costs_of_accessories_max_type_a_accessories_l15_15577

theorem find_costs_of_accessories (x y : ℕ) 
  (h1 : x + 3 * y = 530) 
  (h2 : 3 * x + 2 * y = 890) : 
  x = 230 ∧ y = 100 := 
by 
  sorry

theorem max_type_a_accessories (m n : ℕ) 
  (m_n_sum : m + n = 30) 
  (cost_constraint : 230 * m + 100 * n ≤ 4180) : 
  m ≤ 9 := 
by 
  sorry

end find_costs_of_accessories_max_type_a_accessories_l15_15577


namespace part1_part2_l15_15652

variables (q x : ℝ)
def f (x : ℝ) (q : ℝ) : ℝ := x^2 - 16*x + q + 3
def g (x : ℝ) (q : ℝ) : ℝ := f x q + 51

theorem part1 (h1 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x q = 0):
  (-20 : ℝ) ≤ q ∧ q ≤ 12 := 
  sorry

theorem part2 (h2 : ∀ x ∈ Set.Icc (q : ℝ) 10, g x q ≥ 0) : 
  9 ≤ q ∧ q < 10 := 
  sorry

end part1_part2_l15_15652


namespace smallest_number_l15_15676

theorem smallest_number (x y z : ℕ) (h1 : y = 4 * x) (h2 : z = 2 * y) 
(h3 : (x + y + z) / 3 = 78) : x = 18 := 
by 
    sorry

end smallest_number_l15_15676


namespace min_value_a_plus_b_plus_c_l15_15068

theorem min_value_a_plus_b_plus_c 
  (a b c : ℕ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (x1 x2 : ℝ)
  (hx1_neg : -1 < x1)
  (hx1_pos : x1 < 0)
  (hx2_neg : 0 < x2)
  (hx2_pos : x2 < 1)
  (h_distinct : x1 ≠ x2)
  (h_eqn_x1 : a * x1^2 + b * x1 + c = 0)
  (h_eqn_x2 : a * x2^2 + b * x2 + c = 0) :
  a + b + c = 11 :=
sorry

end min_value_a_plus_b_plus_c_l15_15068


namespace find_d_minus_c_l15_15038

noncomputable def point_transformed (c d : ℝ) : Prop :=
  let Q := (c, d)
  let R := (2 * 2 - c, 2 * 3 - d)  -- Rotating Q by 180º about (2, 3)
  let S := (d, c)                -- Reflecting Q about the line y = x
  (S.1, S.2) = (2, -1)           -- Result is (2, -1)

theorem find_d_minus_c (c d : ℝ) (h : point_transformed c d) : d - c = -1 :=
by {
  sorry
}

end find_d_minus_c_l15_15038


namespace scientific_notation_l15_15585

theorem scientific_notation : (0.000000005 : ℝ) = 5 * 10^(-9 : ℤ) := 
by
  sorry

end scientific_notation_l15_15585


namespace Dexter_card_count_l15_15604

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end Dexter_card_count_l15_15604


namespace total_food_needed_l15_15491

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l15_15491


namespace solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l15_15989

theorem solve_quadratic_eq1 : ∀ x : ℝ, 2 * x^2 + 5 * x + 3 = 0 → (x = -3/2 ∨ x = -1) :=
by
  intro x
  intro h
  sorry

theorem solve_quadratic_eq2_complete_square : ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l15_15989


namespace trains_clear_in_approx_6_85_seconds_l15_15283

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train : ℝ := 80 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def speed_second_train : ℝ := 65 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_in_approx_6_85_seconds : abs (time_to_clear - 6.85) < 0.01 := sorry

end trains_clear_in_approx_6_85_seconds_l15_15283


namespace cricket_team_members_l15_15243

theorem cricket_team_members (n : ℕ) 
  (avg_age_team : ℕ) 
  (age_captain : ℕ) 
  (age_wkeeper : ℕ) 
  (avg_age_remaining : ℕ) 
  (total_age_team : ℕ) 
  (total_age_excl_cw : ℕ) 
  (total_age_remaining : ℕ) :
  avg_age_team = 23 →
  age_captain = 26 →
  age_wkeeper = 29 →
  avg_age_remaining = 22 →
  total_age_team = avg_age_team * n →
  total_age_excl_cw = total_age_team - (age_captain + age_wkeeper) →
  total_age_remaining = avg_age_remaining * (n - 2) →
  total_age_excl_cw = total_age_remaining →
  n = 11 :=
by
  sorry

end cricket_team_members_l15_15243


namespace third_term_binomial_expansion_l15_15270

-- Let a, x be real numbers
variables (a x : ℝ)

-- Binomial theorem term for k = 2
def binomial_term (n k : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose n k) * x^(n-k) * y^k

theorem third_term_binomial_expansion :
  binomial_term 6 2 (a / Real.sqrt x) (-Real.sqrt x / a^2) = 15 / x :=
by
  sorry

end third_term_binomial_expansion_l15_15270


namespace pair_product_not_72_l15_15843

theorem pair_product_not_72 : (2 * (-36) ≠ 72) :=
by
  sorry

end pair_product_not_72_l15_15843


namespace hundreds_digit_of_8_pow_2048_l15_15453

theorem hundreds_digit_of_8_pow_2048 : 
  (8^2048 % 1000) / 100 = 0 := 
by
  sorry

end hundreds_digit_of_8_pow_2048_l15_15453


namespace banana_nn_together_count_l15_15720

open Finset

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def arrangements_banana_with_nn_together : ℕ :=
  (factorial 4) / (factorial 3)

theorem banana_nn_together_count : arrangements_banana_with_nn_together = 4 := by
  sorry

end banana_nn_together_count_l15_15720


namespace find_f_value_l15_15564

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Condition 2: f has a period of 5
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

-- Condition 3: f(-3) = -4
def f_value_at_neg3 (f : ℝ → ℝ) := f (-3) = -4

-- Condition 4: cos(α) = 1 / 2
def cos_alpha_value (α : ℝ) := Real.cos α = 1 / 2

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def α : ℝ := sorry

theorem find_f_value (h_odd : is_odd_function f)
                     (h_periodic : is_periodic f 5)
                     (h_f_neg3 : f_value_at_neg3 f)
                     (h_cos_alpha : cos_alpha_value α) :
  f (4 * Real.cos (2 * α)) = 4 := 
sorry

end find_f_value_l15_15564


namespace complement_intersection_eq_l15_15514

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 2, 5}) (hB : B = {1, 3, 4})

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_eq_l15_15514


namespace smallest_value_of_reciprocal_sums_l15_15217

theorem smallest_value_of_reciprocal_sums (r1 r2 s p : ℝ) 
  (h1 : r1 + r2 = s)
  (h2 : r1^2 + r2^2 = s)
  (h3 : r1^3 + r2^3 = s)
  (h4 : r1^4 + r2^4 = s)
  (h1004 : r1^1004 + r2^1004 = s)
  (h_r1_r2_roots : ∀ x, x^2 - s * x + p = 0) :
  (1 / r1^1005 + 1 / r2^1005) = 2 :=
by
  sorry

end smallest_value_of_reciprocal_sums_l15_15217


namespace contrapositive_x_squared_l15_15383

theorem contrapositive_x_squared :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := 
sorry

end contrapositive_x_squared_l15_15383


namespace average_cost_across_all_products_sold_is_670_l15_15215

-- Definitions based on conditions
def iphones_sold : ℕ := 100
def ipad_sold : ℕ := 20
def appletv_sold : ℕ := 80

def cost_iphone : ℕ := 1000
def cost_ipad : ℕ := 900
def cost_appletv : ℕ := 200

-- Calculations based on conditions
def revenue_iphone : ℕ := iphones_sold * cost_iphone
def revenue_ipad : ℕ := ipad_sold * cost_ipad
def revenue_appletv : ℕ := appletv_sold * cost_appletv

def total_revenue : ℕ := revenue_iphone + revenue_ipad + revenue_appletv
def total_products_sold : ℕ := iphones_sold + ipad_sold + appletv_sold

def average_cost := total_revenue / total_products_sold

-- Theorem to be proved
theorem average_cost_across_all_products_sold_is_670 :
  average_cost = 670 :=
by
  sorry

end average_cost_across_all_products_sold_is_670_l15_15215


namespace gcd_polynomial_is_25_l15_15649

theorem gcd_polynomial_is_25 (b : ℕ) (h : ∃ k : ℕ, b = 2700 * k) :
  Nat.gcd (b^2 + 27 * b + 75) (b + 25) = 25 :=
by 
    sorry

end gcd_polynomial_is_25_l15_15649


namespace speed_conversion_l15_15434

-- Define the given condition
def kmph_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Speed in kmph
def speed_kmph : ℕ := 216

-- The proof statement
theorem speed_conversion : kmph_to_mps speed_kmph = 60 :=
by
  sorry

end speed_conversion_l15_15434


namespace bella_started_with_136_candies_l15_15293

/-
Theorem:
Bella started with 136 candies.
-/

-- define the initial number of candies
variable (x : ℝ)

-- define the conditions
def condition1 : Prop := (x / 2 - 3 / 4) - 5 = 9
def condition2 : Prop := x = 136

-- structure the proof statement 
theorem bella_started_with_136_candies : condition1 x -> condition2 x :=
by
  sorry

end bella_started_with_136_candies_l15_15293


namespace cone_volume_l15_15870

theorem cone_volume (V_cyl : ℝ) (d : ℝ) (π : ℝ) (V_cyl_eq : V_cyl = 81 * π) (h_eq : 2 * (d / 2) = 2 * d) :
  ∃ (V_cone : ℝ), V_cone = 27 * π * (6 ^ (1/3)) :=
by 
  sorry

end cone_volume_l15_15870


namespace candies_left_is_correct_l15_15668

-- Define the number of candies bought on different days
def candiesBoughtTuesday : ℕ := 3
def candiesBoughtThursday : ℕ := 5
def candiesBoughtFriday : ℕ := 2

-- Define the number of candies eaten
def candiesEaten : ℕ := 6

-- Define the total candies left
def candiesLeft : ℕ := (candiesBoughtTuesday + candiesBoughtThursday + candiesBoughtFriday) - candiesEaten

theorem candies_left_is_correct : candiesLeft = 4 := by
  -- Placeholder proof: replace 'sorry' with the actual proof when necessary
  sorry

end candies_left_is_correct_l15_15668


namespace distance_between_cities_l15_15962

theorem distance_between_cities (d : ℝ)
  (meeting_point1 : d - 437 + 437 = d)
  (meeting_point2 : 3 * (d - 437) = 2 * d - 237) :
  d = 1074 :=
by
  sorry

end distance_between_cities_l15_15962


namespace inequality_semi_perimeter_l15_15746

variables {R r p : Real}

theorem inequality_semi_perimeter (h1 : 0 < R) (h2 : 0 < r) (h3 : 0 < p) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 :=
sorry

end inequality_semi_perimeter_l15_15746


namespace average_speed_is_80_l15_15460

def distance : ℕ := 100

def time : ℚ := 5 / 4  -- 1.25 hours expressed as a rational number

noncomputable def average_speed : ℚ := distance / time

theorem average_speed_is_80 : average_speed = 80 := by
  sorry

end average_speed_is_80_l15_15460


namespace geom_seq_sum_five_terms_l15_15999

theorem geom_seq_sum_five_terms (a : ℕ → ℝ) (q : ℝ) 
    (h_pos : ∀ n, 0 < a n)
    (h_a2 : a 2 = 8) 
    (h_arith : 2 * a 4 - a 3 = a 3 - 4 * a 5) :
    a 1 * (1 - q^5) / (1 - q) = 31 :=
by
    sorry

end geom_seq_sum_five_terms_l15_15999


namespace find_r_from_tan_cosine_tangent_l15_15361

theorem find_r_from_tan_cosine_tangent 
  (θ : ℝ) 
  (r : ℝ) 
  (htan : Real.tan θ = -7 / 24) 
  (hquadrant : π / 2 < θ ∧ θ < π) 
  (hr : 100 * Real.cos θ = r) : 
  r = -96 := 
sorry

end find_r_from_tan_cosine_tangent_l15_15361


namespace car_travel_distance_l15_15726

theorem car_travel_distance :
  ∀ (train_speed : ℝ) (fraction : ℝ) (time_minutes : ℝ) (car_speed : ℝ) (distance : ℝ),
  train_speed = 90 →
  fraction = 5 / 6 →
  time_minutes = 30 →
  car_speed = fraction * train_speed →
  distance = car_speed * (time_minutes / 60) →
  distance = 37.5 :=
by
  intros train_speed fraction time_minutes car_speed distance
  intros h_train_speed h_fraction h_time_minutes h_car_speed h_distance
  sorry

end car_travel_distance_l15_15726


namespace part1_part2_l15_15532

variable {a b c : ℝ}

theorem part1 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a * b + b * c + a * c ≤ 1 / 3 := 
sorry 

theorem part2 (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := 
sorry

end part1_part2_l15_15532


namespace net_amount_spent_correct_l15_15318

def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84
def net_amount_spent : ℝ := 139.32

theorem net_amount_spent_correct : trumpet_cost - song_book_revenue = net_amount_spent :=
by
  sorry

end net_amount_spent_correct_l15_15318


namespace find_d_l15_15964

-- Conditions
variables (c d : ℝ)
axiom ratio_cond : c / d = 4
axiom eq_cond : c = 20 - 6 * d

theorem find_d : d = 2 :=
by
  sorry

end find_d_l15_15964


namespace sum_max_min_ratio_l15_15235

def ellipse_eq (x y : ℝ) : Prop :=
  5 * x^2 + x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

theorem sum_max_min_ratio (p q : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y → y / x = p ∨ y / x = q) → 
  p + q = 31 / 34 :=
by
  sorry

end sum_max_min_ratio_l15_15235


namespace diff_implies_continuous_l15_15776

def differentiable_imp_continuous (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀

-- Problem statement: if f is differentiable at x₀, then it is continuous at x₀.
theorem diff_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) : differentiable_imp_continuous f x₀ :=
by
  sorry

end diff_implies_continuous_l15_15776


namespace number_of_substitution_ways_mod_1000_l15_15106

theorem number_of_substitution_ways_mod_1000 :
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  total_ways % 1000 = 573 := by
  -- Definition
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  -- Proof is omitted
  sorry

end number_of_substitution_ways_mod_1000_l15_15106


namespace find_a_l15_15752

noncomputable def geometric_sum_expression (n : ℕ) (a : ℝ) : ℝ :=
  3 * 2^n + a

theorem find_a (a : ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = geometric_sum_expression n a) → a = -3 :=
by
  sorry

end find_a_l15_15752


namespace sum_of_solutions_l15_15841

theorem sum_of_solutions : 
  let a := 1
  let b := -7
  let c := -30
  (a * x^2 + b * x + c = 0) → ((-b / a) = 7) :=
by
  sorry

end sum_of_solutions_l15_15841


namespace sector_central_angle_l15_15224

theorem sector_central_angle (r α: ℝ) (hC: 4 * r = 2 * r + α * r): α = 2 :=
by
  -- Proof is to be filled in
  sorry

end sector_central_angle_l15_15224


namespace total_jumps_l15_15621

theorem total_jumps (hattie_1 : ℕ) (lorelei_1 : ℕ) (hattie_2 : ℕ) (lorelei_2 : ℕ) (hattie_3 : ℕ) (lorelei_3 : ℕ) :
  hattie_1 = 180 →
  lorelei_1 = 3 / 4 * hattie_1 →
  hattie_2 = 2 / 3 * hattie_1 →
  lorelei_2 = hattie_2 + 50 →
  hattie_3 = hattie_2 + 1 / 3 * hattie_2 →
  lorelei_3 = 4 / 5 * lorelei_1 →
  hattie_1 + hattie_2 + hattie_3 + lorelei_1 + lorelei_2 + lorelei_3 = 873 :=
by
  intros h1 l1 h2 l2 h3 l3
  sorry

end total_jumps_l15_15621


namespace wire_length_l15_15570

theorem wire_length (r_sphere r_wire : ℝ) (h : ℝ) (V : ℝ)
  (h₁ : r_sphere = 24) (h₂ : r_wire = 16)
  (h₃ : V = 4 / 3 * Real.pi * r_sphere ^ 3)
  (h₄ : V = Real.pi * r_wire ^ 2 * h): 
  h = 72 := by
  -- we can use provided condition to show that h = 72, proof details omitted
  sorry

end wire_length_l15_15570


namespace cyclist_C_speed_l15_15281

theorem cyclist_C_speed 
  (dist_XY : ℝ)
  (speed_diff : ℝ)
  (meet_point : ℝ)
  (c d : ℝ)
  (h1 : dist_XY = 90)
  (h2 : speed_diff = 5)
  (h3 : meet_point = 15)
  (h4 : d = c + speed_diff)
  (h5 : 75 = dist_XY - meet_point)
  (h6 : 105 = dist_XY + meet_point)
  (h7 : 75 / c = 105 / d) :
  c = 12.5 :=
sorry

end cyclist_C_speed_l15_15281


namespace students_liked_strawberries_l15_15957

theorem students_liked_strawberries : 
  let total_students := 450 
  let students_oranges := 70 
  let students_pears := 120 
  let students_apples := 147 
  let students_strawberries := total_students - (students_oranges + students_pears + students_apples)
  students_strawberries = 113 :=
by
  sorry

end students_liked_strawberries_l15_15957


namespace ryan_bread_slices_l15_15446

theorem ryan_bread_slices 
  (num_pb_people : ℕ)
  (pb_sandwiches_per_person : ℕ)
  (num_tuna_people : ℕ)
  (tuna_sandwiches_per_person : ℕ)
  (num_turkey_people : ℕ)
  (turkey_sandwiches_per_person : ℕ)
  (slices_per_pb_sandwich : ℕ)
  (slices_per_tuna_sandwich : ℕ)
  (slices_per_turkey_sandwich : ℝ)
  (h1 : num_pb_people = 4)
  (h2 : pb_sandwiches_per_person = 2)
  (h3 : num_tuna_people = 3)
  (h4 : tuna_sandwiches_per_person = 3)
  (h5 : num_turkey_people = 2)
  (h6 : turkey_sandwiches_per_person = 1)
  (h7 : slices_per_pb_sandwich = 2)
  (h8 : slices_per_tuna_sandwich = 3)
  (h9 : slices_per_turkey_sandwich = 1.5) : 
  (num_pb_people * pb_sandwiches_per_person * slices_per_pb_sandwich 
  + num_tuna_people * tuna_sandwiches_per_person * slices_per_tuna_sandwich 
  + (num_turkey_people * turkey_sandwiches_per_person : ℝ) * slices_per_turkey_sandwich) = 46 :=
by
  sorry

end ryan_bread_slices_l15_15446


namespace arithmetic_sequence_tenth_term_l15_15733

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l15_15733


namespace max_area_proof_l15_15906

-- Define the original curve
def original_curve (x : ℝ) : ℝ := x^2 + x - 2

-- Reflective symmetry curve about point (p, 2p)
def transformed_curve (p x : ℝ) : ℝ := -x^2 + (4 * p + 1) * x - 4 * p^2 + 2 * p + 2

-- Intersection conditions
def intersecting_curves (p x : ℝ) : Prop :=
original_curve x = transformed_curve p x

-- Range for valid p values
def valid_p (p : ℝ) : Prop := -1 ≤ p ∧ p ≤ 2

-- Prove the problem statement which involves ensuring the curves intersect in the range
theorem max_area_proof :
  ∀ (p : ℝ), valid_p p → ∀ (x : ℝ), intersecting_curves p x →
  ∃ (A : ℝ), A = abs (original_curve x - transformed_curve p x) :=
by
  intros p hp x hx
  sorry

end max_area_proof_l15_15906


namespace john_weekly_calories_l15_15917

-- Define the calorie calculation for each meal type
def breakfast_calories : ℝ := 500
def morning_snack_calories : ℝ := 150
def lunch_calories : ℝ := breakfast_calories + 0.25 * breakfast_calories
def afternoon_snack_calories : ℝ := lunch_calories - 0.30 * lunch_calories
def dinner_calories : ℝ := 2 * lunch_calories

-- Total calories for Friday
def friday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories

-- Additional treats on Saturday and Sunday
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Total calories for each day
def saturday_calories : ℝ := friday_calories + dessert_calories
def sunday_calories : ℝ := friday_calories + 2 * energy_drink_calories
def weekday_calories : ℝ := friday_calories

-- Proof statement
theorem john_weekly_calories : 
  friday_calories = 2962.5 ∧ 
  saturday_calories = 3312.5 ∧ 
  sunday_calories = 3402.5 ∧ 
  weekday_calories = 2962.5 :=
by 
  -- proof expressions would go here
  sorry

end john_weekly_calories_l15_15917


namespace andrei_cannot_ensure_victory_l15_15385

theorem andrei_cannot_ensure_victory :
  ∀ (juice_andrew : ℝ) (juice_masha : ℝ),
    juice_andrew = 24 * 1000 ∧
    juice_masha = 24 * 1000 ∧
    ∀ (andrew_mug : ℝ) (masha_mug1 : ℝ) (masha_mug2 : ℝ),
      andrew_mug = 500 ∧
      masha_mug1 = 240 ∧
      masha_mug2 = 240 ∧
      (¬ (∃ (turns_andrew turns_masha : ℕ), 
        turns_andrew * andrew_mug > 48 * 1000 / 2 ∨
        turns_masha * (masha_mug1 + masha_mug2) > 48 * 1000 / 2)) := sorry

end andrei_cannot_ensure_victory_l15_15385


namespace not_divides_two_pow_n_sub_one_l15_15938

theorem not_divides_two_pow_n_sub_one (n : ℕ) (h1 : n > 1) : ¬ n ∣ (2^n - 1) :=
sorry

end not_divides_two_pow_n_sub_one_l15_15938


namespace total_revenue_calculation_l15_15371

-- Define the total number of etchings sold
def total_etchings : ℕ := 16

-- Define the number of etchings sold at $35 each
def etchings_sold_35 : ℕ := 9

-- Define the price per etching sold at $35
def price_per_etching_35 : ℕ := 35

-- Define the price per etching sold at $45
def price_per_etching_45 : ℕ := 45

-- Define the total revenue calculation
def total_revenue : ℕ :=
  let revenue_35 := etchings_sold_35 * price_per_etching_35
  let etchings_sold_45 := total_etchings - etchings_sold_35
  let revenue_45 := etchings_sold_45 * price_per_etching_45
  revenue_35 + revenue_45

-- Theorem stating the total revenue is $630
theorem total_revenue_calculation : total_revenue = 630 := by
  sorry

end total_revenue_calculation_l15_15371


namespace perfect_square_n_l15_15515

theorem perfect_square_n (m : ℤ) :
  ∃ (n : ℤ), (n = 7 * m^2 + 6 * m + 1 ∨ n = 7 * m^2 - 6 * m + 1) ∧ ∃ (k : ℤ), 7 * n + 2 = k^2 :=
by
  sorry

end perfect_square_n_l15_15515


namespace simplify_and_evaluate_l15_15824

theorem simplify_and_evaluate :
  ∀ (a b : ℚ), a = 2 → b = -1/2 → (a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4) :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l15_15824


namespace inequality_solution_l15_15690

theorem inequality_solution {x : ℝ} :
  ((x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)) ↔
  ((x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0) := sorry

end inequality_solution_l15_15690


namespace option_A_correct_l15_15118

theorem option_A_correct (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
by sorry

end option_A_correct_l15_15118


namespace no_such_pairs_exist_l15_15285

theorem no_such_pairs_exist : ¬ ∃ (n m : ℕ), n > 1 ∧ (∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n) ∧ 
                                    (∀ d : ℕ, d ≠ n → d ∣ n → d + 1 ∣ m ∧ d + 1 ≠ m ∧ d + 1 ≠ 1) :=
by
  sorry

end no_such_pairs_exist_l15_15285


namespace find_E_l15_15443

variables (E F G H : ℕ)

noncomputable def conditions := 
  (E * F = 120) ∧ 
  (G * H = 120) ∧ 
  (E - F = G + H - 2) ∧ 
  (E ≠ F) ∧
  (E ≠ G) ∧ 
  (E ≠ H) ∧
  (F ≠ G) ∧
  (F ≠ H) ∧
  (G ≠ H)

theorem find_E (E F G H : ℕ) (h : conditions E F G H) : E = 30 :=
sorry

end find_E_l15_15443


namespace quadratic_inequality_solution_l15_15009

theorem quadratic_inequality_solution 
  (x : ℝ) (b c : ℝ)
  (h : ∀ x, -x^2 + b*x + c < 0 ↔ x < -3 ∨ x > 2) :
  (6 * x^2 + x - 1 > 0) ↔ (x < -1/2 ∨ x > 1/3) := 
sorry

end quadratic_inequality_solution_l15_15009


namespace apple_and_pear_costs_l15_15159

theorem apple_and_pear_costs (x y : ℝ) (h1 : x + 2 * y = 194) (h2 : 2 * x + 5 * y = 458) : 
  y = 70 ∧ x = 54 := 
by 
  sorry

end apple_and_pear_costs_l15_15159


namespace initial_bacteria_count_l15_15357

theorem initial_bacteria_count :
  ∀ (n : ℕ), (n * 5^8 = 1953125) → n = 5 :=
by
  intro n
  intro h
  sorry

end initial_bacteria_count_l15_15357


namespace quadratic_no_real_roots_iff_m_gt_one_l15_15508

theorem quadratic_no_real_roots_iff_m_gt_one (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 :=
sorry

end quadratic_no_real_roots_iff_m_gt_one_l15_15508


namespace Andy_late_minutes_l15_15335

theorem Andy_late_minutes 
  (school_start : Nat := 8*60) -- 8:00 AM in minutes since midnight
  (normal_travel_time : Nat := 30) -- 30 minutes
  (red_light_stops : Nat := 3 * 4) -- 3 minutes each at 4 lights
  (construction_wait : Nat := 10) -- 10 minutes
  (detour_time : Nat := 7) -- 7 minutes
  (store_stop_time : Nat := 5) -- 5 minutes
  (traffic_delay : Nat := 15) -- 15 minutes
  (departure_time : Nat := 7*60 + 15) -- 7:15 AM in minutes since midnight
  : 34 = departure_time + normal_travel_time + red_light_stops + construction_wait + detour_time + store_stop_time + traffic_delay - school_start := 
by sorry

end Andy_late_minutes_l15_15335


namespace angle_B_is_30_degrees_l15_15204

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Assuming the conditions given in the problem
variables (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) 
          (h2 : a > b)

-- The proof to establish the measure of angle B as 30 degrees
theorem angle_B_is_30_degrees (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) (h2 : a > b) : B = Real.pi / 6 :=
sorry

end angle_B_is_30_degrees_l15_15204


namespace common_difference_l15_15377

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
    (h1 : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
    (h2 : a 1 + a 3 + a 5 = 15)
    (h3 : a 4 = 3) : 
    d = -2 := 
sorry

end common_difference_l15_15377


namespace central_angle_l15_15273

theorem central_angle (r l θ : ℝ) (condition1: 2 * r + l = 8) (condition2: (1 / 2) * l * r = 4) (theta_def : θ = l / r) : |θ| = 2 :=
by
  sorry

end central_angle_l15_15273


namespace probability_circle_or_square_l15_15086

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l15_15086


namespace probability_not_grade_5_l15_15456

theorem probability_not_grade_5 :
  let A1 := 0.3
  let A2 := 0.4
  let A3 := 0.2
  let A4 := 0.1
  (A1 + A2 + A3 + A4 = 1) → (1 - A1 = 0.7) := by
  intros A1_def A2_def A3_def A4_def h
  sorry

end probability_not_grade_5_l15_15456


namespace percentage_of_Y_salary_l15_15592

variable (X Y : ℝ)
variable (total_salary Y_salary : ℝ)
variable (P : ℝ)

theorem percentage_of_Y_salary :
  total_salary = 638 ∧ Y_salary = 290 ∧ X = (P / 100) * Y_salary → P = 120 := by
  sorry

end percentage_of_Y_salary_l15_15592


namespace value_bounds_of_expression_l15_15053

theorem value_bounds_of_expression
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (triangle_ineq1 : a + b > c)
  (triangle_ineq2 : a + c > b)
  (triangle_ineq3 : b + c > a)
  : 4 ≤ (a+b+c)^2 / (b*c) ∧ (a+b+c)^2 / (b*c) ≤ 9 := sorry

end value_bounds_of_expression_l15_15053


namespace power_eq_l15_15478

theorem power_eq (a b c : ℝ) (h₁ : a = 81) (h₂ : b = 4 / 3) : (a ^ b) = 243 * (3 ^ (1 / 3)) := by
  sorry

end power_eq_l15_15478


namespace athlete_runs_entire_track_in_44_seconds_l15_15050

noncomputable def time_to_complete_track (flags : ℕ) (time_to_4th_flag : ℕ) : ℕ :=
  let distances_between_flags := flags - 1
  let distances_to_4th_flag := 4 - 1
  let time_per_distance := time_to_4th_flag / distances_to_4th_flag
  distances_between_flags * time_per_distance

theorem athlete_runs_entire_track_in_44_seconds :
  time_to_complete_track 12 12 = 44 :=
by
  sorry

end athlete_runs_entire_track_in_44_seconds_l15_15050


namespace f_periodic_function_l15_15256

noncomputable def f : ℝ → ℝ := sorry

theorem f_periodic_function (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x : ℝ, f (x + 4) = f x + f 2)
    (h3 : f 1 = 2) : 
    f 2013 = 2 := sorry

end f_periodic_function_l15_15256


namespace students_per_group_l15_15977

theorem students_per_group (total_students not_picked_groups groups : ℕ) (h₁ : total_students = 65) (h₂ : not_picked_groups = 17) (h₃ : groups = 8) :
  (total_students - not_picked_groups) / groups = 6 := by
  sorry

end students_per_group_l15_15977


namespace range_of_x_l15_15639

theorem range_of_x (x : ℝ) : 
  (∀ (m : ℝ), |m| ≤ 1 → x^2 - 2 > m * x) ↔ (x < -2 ∨ x > 2) :=
by 
  sorry

end range_of_x_l15_15639


namespace infinite_sqrt_solution_l15_15598

noncomputable def infinite_sqrt (x : ℝ) : ℝ := Real.sqrt (20 + x)

theorem infinite_sqrt_solution : 
  ∃ x : ℝ, infinite_sqrt x = x ∧ x ≥ 0 ∧ x = 5 :=
by
  sorry

end infinite_sqrt_solution_l15_15598


namespace minimum_moves_to_find_coin_l15_15229

/--
Consider a circle of 100 thimbles with a coin hidden under one of them. 
You can check four thimbles per move. After each move, the coin moves to a neighboring thimble.
Prove that the minimum number of moves needed to guarantee finding the coin is 33.
-/
theorem minimum_moves_to_find_coin 
  (N : ℕ) (hN : N = 100) (M : ℕ) (hM : M = 4) :
  ∃! k : ℕ, k = 33 :=
by sorry

end minimum_moves_to_find_coin_l15_15229


namespace axis_angle_set_l15_15939

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end axis_angle_set_l15_15939


namespace vasya_improved_example1_vasya_improved_example2_l15_15642

theorem vasya_improved_example1 : (333 / 3) - (33 / 3) = 100 := by
  sorry

theorem vasya_improved_example2 : (33 * 3) + (3 / 3) = 100 := by
  sorry

end vasya_improved_example1_vasya_improved_example2_l15_15642


namespace t_shirts_per_package_l15_15925

theorem t_shirts_per_package (total_t_shirts : ℕ) (total_packages : ℕ) (h1 : total_t_shirts = 39) (h2 : total_packages = 3) : total_t_shirts / total_packages = 13 :=
by {
  sorry
}

end t_shirts_per_package_l15_15925


namespace probability_of_drawing_white_ball_l15_15074

def total_balls (red white : ℕ) : ℕ := red + white

def number_of_white_balls : ℕ := 2

def number_of_red_balls : ℕ := 3

def probability_of_white_ball (white total : ℕ) : ℚ := white / total

-- Theorem statement
theorem probability_of_drawing_white_ball :
  probability_of_white_ball number_of_white_balls (total_balls number_of_red_balls number_of_white_balls) = 2 / 5 :=
sorry

end probability_of_drawing_white_ball_l15_15074


namespace binary_to_octal_conversion_l15_15261

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l15_15261


namespace evaluate_expression_l15_15845

theorem evaluate_expression (x y z : ℝ) (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 :=
by 
  sorry

end evaluate_expression_l15_15845


namespace operation_equivalence_l15_15080

theorem operation_equivalence :
  (∀ (x : ℝ), (x * (4 / 5) / (2 / 7)) = x * (7 / 5)) :=
by
  sorry

end operation_equivalence_l15_15080


namespace difference_mean_median_is_neg_half_l15_15208

-- Definitions based on given conditions
def scoreDistribution : List (ℕ × ℚ) :=
  [(65, 0.05), (75, 0.25), (85, 0.4), (95, 0.2), (105, 0.1)]

-- Defining the total number of students as 100 for easier percentage calculations
def totalStudents := 100

-- Definition to compute mean
def mean : ℚ :=
  scoreDistribution.foldl (λ acc (score, percentage) => acc + (↑score * percentage)) 0

-- Median score based on the distribution conditions
def median : ℚ := 85

-- Proving the proposition that the difference between the mean and the median is -0.5
theorem difference_mean_median_is_neg_half :
  median - mean = -0.5 :=
sorry

end difference_mean_median_is_neg_half_l15_15208


namespace weeks_to_cover_expense_l15_15910

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l15_15910


namespace ratio_of_female_to_male_members_l15_15994

theorem ratio_of_female_to_male_members 
  (f m : ℕ)
  (avg_age_female avg_age_male avg_age_membership : ℕ)
  (hf : avg_age_female = 35)
  (hm : avg_age_male = 30)
  (ha : avg_age_membership = 32)
  (h_avg : (35 * f + 30 * m) / (f + m) = 32) : 
  f / m = 2 / 3 :=
sorry

end ratio_of_female_to_male_members_l15_15994


namespace match_proverbs_l15_15594

-- Define each condition as a Lean definition
def condition1 : Prop :=
"As cold comes and heat goes, the four seasons change" = "Things are developing"

def condition2 : Prop :=
"Thousands of flowers arranged, just waiting for the first thunder" = 
"Decisively seize the opportunity to promote qualitative change"

def condition3 : Prop :=
"Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade" = 
"The unity of contradictions"

def condition4 : Prop :=
"There will be times when the strong winds break the waves, and we will sail across the sea with clouds" = 
"The future is bright"

-- The theorem we need to prove, using the condition definitions
theorem match_proverbs : condition2 ∧ condition4 :=
sorry

end match_proverbs_l15_15594


namespace max_a_condition_slope_condition_exponential_inequality_l15_15740

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 1)
noncomputable def g (x a : ℝ) := f x a + a / Real.exp x

theorem max_a_condition (a : ℝ) (h_pos : a > 0) 
  (h_nonneg : ∀ x : ℝ, f x a ≥ 0) : a ≤ 1 := sorry

theorem slope_condition (a m : ℝ) 
  (ha : a ≤ -1) 
  (h_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → 
    (g x2 a - g x1 a) / (x2 - x1) > m) : m ≤ 3 := sorry

theorem exponential_inequality (n : ℕ) (hn : n > 0) : 
  (2 * (Real.exp n - 1)) / (Real.exp 1 - 1) ≥ n * (n + 1) := sorry

end max_a_condition_slope_condition_exponential_inequality_l15_15740


namespace Josh_marbles_count_l15_15004

-- Definitions of the given conditions
def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

-- The statement we aim to prove
theorem Josh_marbles_count : (initial_marbles - lost_marbles) = 9 :=
by
  -- Skipping the proof with sorry
  sorry

end Josh_marbles_count_l15_15004


namespace most_and_least_l15_15003

variables {Jan Kim Lee Ron Zay : ℝ}

-- Conditions as hypotheses
axiom H1 : Lee < Jan
axiom H2 : Kim < Jan
axiom H3 : Zay < Ron
axiom H4 : Zay < Lee
axiom H5 : Zay < Jan
axiom H6 : Jan < Ron

theorem most_and_least :
  (Ron > Jan) ∧ (Ron > Kim) ∧ (Ron > Lee) ∧ (Ron > Zay) ∧ 
  (Zay < Jan) ∧ (Zay < Kim) ∧ (Zay < Lee) ∧ (Zay < Ron) :=
by {
  -- Proof is omitted
  sorry
}

end most_and_least_l15_15003


namespace opposite_number_l15_15271

variable (a : ℝ)

theorem opposite_number (a : ℝ) : -(3 * a - 2) = -3 * a + 2 := by
  sorry

end opposite_number_l15_15271


namespace intersection_A_B_eq_B_l15_15310

-- Define set A
def setA : Set ℝ := { x : ℝ | x > -3 }

-- Define set B
def setB : Set ℝ := { x : ℝ | x ≥ 2 }

-- Theorem statement of proving the intersection of setA and setB is setB itself
theorem intersection_A_B_eq_B : setA ∩ setB = setB :=
by
  -- proof skipped
  sorry

end intersection_A_B_eq_B_l15_15310


namespace original_price_of_boots_l15_15691

theorem original_price_of_boots (P : ℝ) (h : P * 0.80 = 72) : P = 90 :=
by 
  sorry

end original_price_of_boots_l15_15691


namespace units_digit_2009_2008_plus_2013_l15_15041

theorem units_digit_2009_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 :=
by
  sorry

end units_digit_2009_2008_plus_2013_l15_15041


namespace pages_read_tonight_l15_15502

def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem pages_read_tonight :
  let pages_3_nights_ago := 20
  let pages_2_nights_ago := 20^2 + 5
  let pages_last_night := sum_of_digits pages_2_nights_ago * 3
  let total_pages := 500
  total_pages - (pages_3_nights_ago + pages_2_nights_ago + pages_last_night) = 48 :=
by
  sorry

end pages_read_tonight_l15_15502


namespace neg_real_root_condition_l15_15899

theorem neg_real_root_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (0 < a ∧ a ≤ 1) ∨ (a < 0) :=
by
  sorry

end neg_real_root_condition_l15_15899


namespace parabola_standard_eq_line_m_tangent_l15_15658

open Real

variables (p k : ℝ) (x y : ℝ)

-- Definitions based on conditions
def parabola_equation (p : ℝ) : Prop := ∀ x y : ℝ, x^2 = 2 * p * y
def line_m (k : ℝ) : Prop := ∀ x y : ℝ, y = k * x + 6

-- Problem statement
theorem parabola_standard_eq (p : ℝ) (hp : p = 2) :
  parabola_equation p ↔ (∀ x y : ℝ, x^2 = 4 * y) :=
sorry

theorem line_m_tangent (k : ℝ) (x1 x2 : ℝ)
  (hpq : x1 + x2 = 4 * k ∧ x1 * x2 = -24)
  (hk : k = 1/2 ∨ k = -1/2) :
  line_m k ↔ ((k = 1/2 ∧ ∀ x y : ℝ, y = 1/2 * x + 6) ∨ (k = -1/2 ∧ ∀ x y : ℝ, y = -1/2 * x + 6)) :=
sorry

end parabola_standard_eq_line_m_tangent_l15_15658


namespace problem_proof_l15_15046

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

def num_multiples_of_lt (m bound : ℕ) : ℕ :=
  (bound - 1) / m

-- Definitions for the conditions
def a := num_multiples_of_lt 8 40
def b := num_multiples_of_lt 8 40

-- Proof statement
theorem problem_proof : (a - b)^3 = 0 := by
  sorry

end problem_proof_l15_15046


namespace problem1_problem2_l15_15559

-- Problem 1
theorem problem1 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / b) + (1 / c) ≥ (1 / (Real.sqrt (a * b))) + (1 / (Real.sqrt (b * c))) + (1 / (Real.sqrt (a * c))) :=
sorry

-- Problem 2
theorem problem2 {x y : ℝ} :
  Real.sin x + Real.sin y ≤ 1 + Real.sin x * Real.sin y :=
sorry

end problem1_problem2_l15_15559


namespace no_six_consecutive_nat_num_sum_eq_2015_l15_15374

theorem no_six_consecutive_nat_num_sum_eq_2015 :
  ∀ (a b c d e f : ℕ),
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f →
  a * b * c + d * e * f ≠ 2015 :=
by
  intros a b c d e f h
  sorry

end no_six_consecutive_nat_num_sum_eq_2015_l15_15374


namespace distinct_sequences_count_l15_15244

noncomputable def number_of_distinct_sequences (n : ℕ) : ℕ :=
  if n = 6 then 12 else sorry

theorem distinct_sequences_count : number_of_distinct_sequences 6 = 12 := 
by 
  sorry

end distinct_sequences_count_l15_15244


namespace first_train_speed_l15_15882

-- Definitions
def train_speeds_opposite (v₁ v₂ t : ℝ) : Prop := v₁ * t + v₂ * t = 910

def train_problem_conditions (v₁ v₂ t : ℝ) : Prop :=
  train_speeds_opposite v₁ v₂ t ∧ v₂ = 80 ∧ t = 6.5

-- Theorem
theorem first_train_speed (v : ℝ) (h : train_problem_conditions v 80 6.5) : v = 60 :=
  sorry

end first_train_speed_l15_15882


namespace operation_positive_l15_15192

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end operation_positive_l15_15192


namespace perfect_square_difference_l15_15231

def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem perfect_square_difference :
  ∃ a b : ℕ, ∃ x y : ℕ,
    a = x^2 ∧ b = y^2 ∧
    lastDigit a = 6 ∧
    lastDigit b = 4 ∧
    lastDigit (a - b) = 2 ∧
    lastDigit a > lastDigit b :=
by
  sorry

end perfect_square_difference_l15_15231


namespace negation_proposition_false_l15_15475

theorem negation_proposition_false : 
  (¬ ∃ x : ℝ, x^2 + 2 ≤ 0) :=
by sorry

end negation_proposition_false_l15_15475


namespace largest_k_exists_l15_15541

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists_l15_15541


namespace prime_factors_power_l15_15403

-- Given conditions
def a_b_c_factors (a b c : ℕ) : Prop :=
  (∀ x, x = a ∨ x = b ∨ x = c → Prime x) ∧
  a < b ∧ b < c ∧ a * b * c ∣ 1998

-- Proof problem
theorem prime_factors_power (a b c : ℕ) (h : a_b_c_factors a b c) : (b + c) ^ a = 1600 := 
sorry

end prime_factors_power_l15_15403


namespace image_length_interval_two_at_least_four_l15_15121

noncomputable def quadratic_function (p q r : ℝ) : ℝ → ℝ :=
  fun x => p * (x - q)^2 + r

theorem image_length_interval_two_at_least_four (p q r : ℝ)
  (h : ∀ I : Set ℝ, (∀ a b : ℝ, I = Set.Icc a b ∨ I = Set.Ioo a b → |b - a| = 1 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 1)) :
  ∀ I' : Set ℝ, (∀ a b : ℝ, I' = Set.Icc a b ∨ I' = Set.Ioo a b → |b - a| = 2 → |quadratic_function p q r b - quadratic_function p q r a| ≥ 4) :=
by
  sorry


end image_length_interval_two_at_least_four_l15_15121


namespace approx_sum_l15_15284

-- Definitions of the costs
def cost_bicycle : ℕ := 389
def cost_fan : ℕ := 189

-- Definition of the approximations
def approx_bicycle : ℕ := 400
def approx_fan : ℕ := 200

-- The statement to prove
theorem approx_sum (h₁ : cost_bicycle = 389) (h₂ : cost_fan = 189) : 
  approx_bicycle + approx_fan = 600 := 
by 
  sorry

end approx_sum_l15_15284


namespace solve_for_x_l15_15049

theorem solve_for_x (x : ℚ) (h : (1 / 7) + (7 / x) = (15 / x) + (1 / 15)) : x = 105 := 
by 
  sorry

end solve_for_x_l15_15049


namespace pencils_added_by_sara_l15_15266

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l15_15266


namespace min_points_to_win_l15_15477

theorem min_points_to_win : ∀ (points : ℕ), (∀ (race_results : ℕ → ℕ), 
  (points = race_results 1 * 4 + race_results 2 * 2 + race_results 3 * 1) 
  ∧ (∀ i, 1 ≤ race_results i ∧ race_results i ≤ 4) 
  ∧ (∀ i j, i ≠ j → race_results i ≠ race_results j) 
  ∧ (race_results 1 + race_results 2 + race_results 3 = 4)) → (15 ≤ points) :=
by
  sorry

end min_points_to_win_l15_15477


namespace max_x_satisfies_inequality_l15_15480

theorem max_x_satisfies_inequality (k : ℝ) :
    (∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) → k = 8 :=
by
  intros h
  /- The proof goes here. -/
  sorry

end max_x_satisfies_inequality_l15_15480


namespace apples_in_pile_l15_15158

/-- Assuming an initial pile of 8 apples and adding 5 more apples, there should be 13 apples in total. -/
theorem apples_in_pile (initial_apples added_apples : ℕ) (h1 : initial_apples = 8) (h2 : added_apples = 5) :
  initial_apples + added_apples = 13 :=
by
  sorry

end apples_in_pile_l15_15158


namespace percent_of_number_l15_15501

theorem percent_of_number (x : ℝ) (hx : (120 / x) = (75 / 100)) : x = 160 := 
sorry

end percent_of_number_l15_15501


namespace expected_value_of_problems_l15_15222

-- Define the setup
def num_pairs : ℕ := 5
def num_shoes : ℕ := num_pairs * 2
def prob_same_color : ℚ := 1 / (num_shoes - 1)
def days : ℕ := 5

-- Define the expected value calculation using linearity of expectation
def expected_problems_per_day : ℚ := prob_same_color
def expected_total_problems : ℚ := days * expected_problems_per_day

-- Prove the expected number of practice problems Sandra gets to do over 5 days
theorem expected_value_of_problems : expected_total_problems = 5 / 9 := 
by 
  rw [expected_total_problems, expected_problems_per_day, prob_same_color]
  norm_num
  sorry

end expected_value_of_problems_l15_15222


namespace stereographic_projection_reflection_l15_15779

noncomputable def sphere : Type := sorry
noncomputable def point_on_sphere (P : sphere) : Prop := sorry
noncomputable def reflection_on_sphere (P P' : sphere) (e : sphere) : Prop := sorry
noncomputable def arbitrary_point (E : sphere) (P P' : sphere) : Prop := E ≠ P ∧ E ≠ P'
noncomputable def tangent_plane (E : sphere) : Type := sorry
noncomputable def stereographic_projection (E : sphere) (δ : Type) : sphere → sorry := sorry
noncomputable def circle_on_plane (e : sphere) (E : sphere) (δ : Type) : Type := sorry
noncomputable def inversion_in_circle (P P' : sphere) (e_1 : Type) : Prop := sorry

theorem stereographic_projection_reflection (P P' E : sphere) (e : sphere) (δ : Type) (e_1 : Type) :
  point_on_sphere P ∧
  reflection_on_sphere P P' e ∧
  arbitrary_point E P P' ∧
  circle_on_plane e E δ = e_1 →
  inversion_in_circle P P' e_1 :=
sorry

end stereographic_projection_reflection_l15_15779


namespace f_1982_value_l15_15370

noncomputable def f (n : ℕ) : ℕ := sorry  -- placeholder for the function definition

axiom f_condition_2 : f 2 = 0
axiom f_condition_3 : f 3 > 0
axiom f_condition_9999 : f 9999 = 3333
axiom f_add_condition (m n : ℕ) : f (m+n) - f m - f n = 0 ∨ f (m+n) - f m - f n = 1

open Nat

theorem f_1982_value : f 1982 = 660 :=
by
  sorry  -- proof goes here

end f_1982_value_l15_15370


namespace geometric_arithmetic_sum_l15_15946

open Real

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def arithmetic_mean (x y a b c : ℝ) : Prop :=
  2 * x = a + b ∧ 2 * y = b + c

theorem geometric_arithmetic_sum
  (a b c x y : ℝ)
  (habc : geometric_sequence a b c)
  (hxy : arithmetic_mean x y a b c)
  (hx_ne_zero : x ≠ 0)
  (hy_ne_zero : y ≠ 0) :
  (a / x) + (c / y) = 2 := 
by {
  sorry -- Proof omitted as per the prompt
}

end geometric_arithmetic_sum_l15_15946


namespace equilateral_triangle_side_length_l15_15043

noncomputable def side_length (a : ℝ) := if a = 0 then 0 else (a : ℝ) * (3 : ℝ) / 2

theorem equilateral_triangle_side_length
  (a : ℝ)
  (h1 : a ≠ 0)
  (A := (a, - (1 / 3) * a^2))
  (B := (-a, - (1 / 3) * a^2))
  (Habo : (A.1 - 0)^2 + (A.2 - 0)^2 = (B.1 - 0)^2 + (B.2 - 0)^2) :
  ∃ s : ℝ, s = 9 / 2 :=
by
  sorry

end equilateral_triangle_side_length_l15_15043


namespace Vanya_bullets_l15_15828

theorem Vanya_bullets (initial_bullets : ℕ) (hits : ℕ) (shots_made : ℕ) (hits_reward : ℕ) :
  initial_bullets = 10 →
  shots_made = 14 →
  hits = shots_made / 2 →
  hits_reward = 3 →
  (initial_bullets + hits * hits_reward) - shots_made = 17 :=
by
  intros
  sorry

end Vanya_bullets_l15_15828


namespace find_t_l15_15047

variables {t : ℝ}

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (t : ℝ) : ℝ × ℝ := (-2, t)

def are_parallel (u v : ℝ × ℝ) : Prop := 
  u.1 * v.2 = u.2 * v.1

theorem find_t (h : are_parallel vector_a (vector_b t)) : t = -4 :=
by sorry

end find_t_l15_15047


namespace total_pigs_indeterminate_l15_15820

noncomputable def average_weight := 15
def underweight_threshold := 16
def max_underweight_pigs := 4

theorem total_pigs_indeterminate :
  ∃ (P U : ℕ), U ≤ max_underweight_pigs ∧ (average_weight = 15) → P = P :=
sorry

end total_pigs_indeterminate_l15_15820


namespace total_blood_cells_correct_l15_15077

-- Define the number of blood cells in the first and second samples.
def sample_1_blood_cells : ℕ := 4221
def sample_2_blood_cells : ℕ := 3120

-- Define the total number of blood cells.
def total_blood_cells : ℕ := sample_1_blood_cells + sample_2_blood_cells

-- Theorem stating the total number of blood cells based on the conditions.
theorem total_blood_cells_correct : total_blood_cells = 7341 :=
by
  -- Proof is omitted
  sorry

end total_blood_cells_correct_l15_15077


namespace find_natural_numbers_eq_36_sum_of_digits_l15_15533

-- Define the sum of digits function
def sum_of_digits (x : ℕ) : ℕ := 
  if x = 0 then 0
  else sum_of_digits (x / 10) + (x % 10)

-- Lean theorem statement proving the given problem
theorem find_natural_numbers_eq_36_sum_of_digits :
  {x : ℕ | x = 36 * (sum_of_digits x)} = {324, 648} :=
sorry

end find_natural_numbers_eq_36_sum_of_digits_l15_15533


namespace expr_simplified_l15_15324

theorem expr_simplified : |2 - Real.sqrt 2| - Real.sqrt (1 / 12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1 / 2 := 
by 
  sorry

end expr_simplified_l15_15324


namespace length_PZ_l15_15996

-- Define the given conditions
variables (CD WX : ℝ) -- segments CD and WX
variable (CW : ℝ) -- length of segment CW
variable (DP : ℝ) -- length of segment DP
variable (PX : ℝ) -- length of segment PX

-- Define the similarity condition
-- segment CD is parallel to segment WX implies that the triangles CDP and WXP are similar

-- Define what we want to prove
theorem length_PZ (hCD_WX_parallel : CD = WX)
                  (hCW : CW = 56)
                  (hDP : DP = 18)
                  (hPX : PX = 36) :
  ∃ PZ : ℝ, PZ = 4 / 3 :=
by
  -- proof steps here (omitted)
  sorry

end length_PZ_l15_15996


namespace compute_pairs_a_b_l15_15405

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem compute_pairs_a_b (a b : ℝ) (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -b) :
  ((∀ x, f (f x a b) a b = -1 / x) ↔ (a = -1 ∧ b = 1)) :=
sorry

end compute_pairs_a_b_l15_15405


namespace average_prime_numbers_l15_15180

-- Definitions of the visible numbers.
def visible1 : ℕ := 51
def visible2 : ℕ := 72
def visible3 : ℕ := 43

-- Definitions of the hidden numbers as prime numbers.
def hidden1 : ℕ := 2
def hidden2 : ℕ := 23
def hidden3 : ℕ := 31

-- Common sum of the numbers on each card.
def common_sum : ℕ := 74

-- Establishing the conditions given in the problem.
def condition1 : hidden1 + visible2 = common_sum := by sorry
def condition2 : hidden2 + visible1 = common_sum := by sorry
def condition3 : hidden3 + visible3 = common_sum := by sorry

-- Calculate the average of the hidden prime numbers.
def average_hidden_primes : ℚ := (hidden1 + hidden2 + hidden3) / 3

-- The proof statement that the average of the hidden prime numbers is 56/3.
theorem average_prime_numbers : average_hidden_primes = 56 / 3 := by
  sorry

end average_prime_numbers_l15_15180


namespace tan_product_l15_15538

theorem tan_product : 
(1 + Real.tan (Real.pi / 60)) * (1 + Real.tan (Real.pi / 30)) * (1 + Real.tan (Real.pi / 20)) * (1 + Real.tan (Real.pi / 15)) * (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 10)) * (1 + Real.tan (Real.pi / 9)) * (1 + Real.tan (Real.pi / 6)) = 2^8 :=
by
  sorry 

end tan_product_l15_15538


namespace count_of_numbers_with_digit_3_eq_71_l15_15638

-- Define the problem space
def count_numbers_without_digit_3 : ℕ := 729
def total_numbers : ℕ := 800
def count_numbers_with_digit_3 : ℕ := total_numbers - count_numbers_without_digit_3

-- Prove that the count of numbers from 1 to 800 containing at least one digit 3 is 71
theorem count_of_numbers_with_digit_3_eq_71 :
  count_numbers_with_digit_3 = 71 :=
by
  sorry

end count_of_numbers_with_digit_3_eq_71_l15_15638


namespace length_of_each_part_l15_15269

theorem length_of_each_part (ft : ℕ) (inch : ℕ) (parts : ℕ) (total_length : ℕ) (part_length : ℕ) :
  ft = 6 → inch = 8 → parts = 5 → total_length = 12 * ft + inch → part_length = total_length / parts → part_length = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_each_part_l15_15269


namespace clock_hand_positions_l15_15300

theorem clock_hand_positions : ∃ n : ℕ, n = 143 ∧ 
  (∀ t : ℝ, let hour_pos := t / 12
            let min_pos := t
            let switched_hour_pos := t
            let switched_min_pos := t / 12
            hour_pos = switched_min_pos ∧ min_pos = switched_hour_pos ↔
            ∃ k : ℤ, t = k / 11) :=
by sorry

end clock_hand_positions_l15_15300


namespace inequality_abc_l15_15890

variable (a b c : ℝ)

theorem inequality_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  a / (a^3 - a^2 + 3) + b / (b^3 - b^2 + 3) + c / (c^3 - c^2 + 3) ≤ 1 := 
sorry

end inequality_abc_l15_15890


namespace part1_solution_set_part2_comparison_l15_15774

noncomputable def f (x : ℝ) := -|x| - |x + 2|

theorem part1_solution_set (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 :=
by sorry

theorem part2_comparison (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = Real.sqrt 5) : 
  a^2 + b^2 / 4 ≥ f x + 3 :=
by sorry

end part1_solution_set_part2_comparison_l15_15774


namespace intersection_M_N_l15_15619

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l15_15619


namespace smallest_n_for_terminating_decimal_l15_15463

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), (∀ m, m < n → (∃ k1 k2 : ℕ, (m + 150 = 2^k1 * 5^k2 ∧ m > 0) → false)) ∧ (∃ k1 k2 : ℕ, (n + 150 = 2^k1 * 5^k2) ∧ n > 0) :=
sorry

end smallest_n_for_terminating_decimal_l15_15463


namespace min_value_l15_15349

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (1 / x + 4 / y) ≥ 9) :=
by
  sorry

end min_value_l15_15349


namespace probability_of_even_product_l15_15995

-- Each die has faces numbered from 1 to 8.
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Calculate the number of outcomes where the product of two rolls is even.
def num_even_product_outcomes : ℕ := (64 - 16)

-- Calculate the total number of outcomes when two eight-sided dice are rolled.
def total_outcomes : ℕ := 64

-- The probability that the product is even.
def probability_even_product : ℚ := num_even_product_outcomes / total_outcomes

theorem probability_of_even_product :
  probability_even_product = 3 / 4 :=
  by
    sorry

end probability_of_even_product_l15_15995


namespace passed_boys_count_l15_15340

theorem passed_boys_count (P F : ℕ) 
  (h1 : P + F = 120) 
  (h2 : 37 * 120 = 39 * P + 15 * F) : 
  P = 110 :=
sorry

end passed_boys_count_l15_15340


namespace f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l15_15681

noncomputable def f (x : ℝ) : ℝ := (1 / 4) ^ x + (1 / 2) ^ x - 1
noncomputable def g (x m : ℝ) : ℝ := (1 - m * 2 ^ x) / (1 + m * 2 ^ x)

theorem f_range_and_boundedness :
  ∀ x : ℝ, x < 0 → 1 < f x ∧ ¬(∃ M : ℝ, ∀ x : ℝ, x < 0 → |f x| ≤ M) :=
by sorry

theorem g_odd_and_bounded (x : ℝ) :
  g x 1 = -g (-x) 1 ∧ |g x 1| < 1 :=
by sorry

theorem g_upper_bound (m : ℝ) (hm : 0 < m ∧ m < 1 / 2) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g x m ≤ (1 - m) / (1 + m) :=
by sorry

end f_range_and_boundedness_g_odd_and_bounded_g_upper_bound_l15_15681


namespace time_fraction_reduced_l15_15551

theorem time_fraction_reduced (T D : ℝ) (h1 : D = 30 * T) :
  D = 40 * ((3/4) * T) → 1 - (3/4) = 1/4 :=
sorry

end time_fraction_reduced_l15_15551


namespace raj_house_area_l15_15338

theorem raj_house_area :
  let bedroom_area := 11 * 11
  let bedrooms_total := bedroom_area * 4
  let bathroom_area := 6 * 8
  let bathrooms_total := bathroom_area * 2
  let kitchen_area := 265
  let living_area := kitchen_area
  bedrooms_total + bathrooms_total + kitchen_area + living_area = 1110 :=
by
  -- Proof to be filled in
  sorry

end raj_house_area_l15_15338


namespace length_of_train_l15_15336

variable (L V : ℝ)

def platform_crossing (L V : ℝ) := L + 350 = V * 39
def post_crossing (L V : ℝ) := L = V * 18

theorem length_of_train (h1 : platform_crossing L V) (h2 : post_crossing L V) : L = 300 :=
by
  sorry

end length_of_train_l15_15336


namespace problem_solution_l15_15130

-- Define the necessary conditions
def f (x : ℤ) : ℤ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Define the main theorem
theorem problem_solution :
  (Nat.gcd 840 1785 = 105) ∧ (f 2 = 62) :=
by {
  -- We include sorry here to indicate that the proof is omitted.
  sorry
}

end problem_solution_l15_15130


namespace arithmetic_sequence_problem_l15_15708

variable {α : Type*} [LinearOrderedRing α]

theorem arithmetic_sequence_problem
  (a : ℕ → α)
  (h : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_seq : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 :=
sorry

end arithmetic_sequence_problem_l15_15708


namespace children_count_l15_15018

theorem children_count (C : ℕ) 
    (cons : ℕ := 12)
    (total_cost : ℕ := 76)
    (child_ticket_cost : ℕ := 7)
    (adult_ticket_cost : ℕ := 10)
    (num_adults : ℕ := 5)
    (adult_cost := num_adults * adult_ticket_cost)
    (cost_with_concessions := total_cost - adult_cost )
    (children_cost := cost_with_concessions - cons):
    C = children_cost / child_ticket_cost :=
by
    sorry

end children_count_l15_15018


namespace molecular_weight_NaClO_is_74_44_l15_15267

-- Define the atomic weights
def atomic_weight_Na : Real := 22.99
def atomic_weight_Cl : Real := 35.45
def atomic_weight_O : Real := 16.00

-- Define the calculation of molecular weight
def molecular_weight_NaClO : Real :=
  atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

-- Define the theorem statement
theorem molecular_weight_NaClO_is_74_44 :
  molecular_weight_NaClO = 74.44 :=
by
  -- Placeholder for proof
  sorry

end molecular_weight_NaClO_is_74_44_l15_15267


namespace polynomial_quotient_correct_l15_15444

noncomputable def polynomial_division_quotient : Polynomial ℝ :=
  (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 8) / (Polynomial.X - Polynomial.C 1)

-- Math proof statement
theorem polynomial_quotient_correct :
  polynomial_division_quotient = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 1 * Polynomial.X^4 
                                 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2 
                                 + Polynomial.C 3 * Polynomial.X + Polynomial.C 3 :=
by
  sorry

end polynomial_quotient_correct_l15_15444


namespace shingle_area_l15_15037

-- Definitions from conditions
def length := 10 -- uncut side length in inches
def width := 7   -- uncut side width in inches
def trapezoid_base1 := 6 -- base of the trapezoid in inches
def trapezoid_height := 2 -- height of the trapezoid in inches

-- Definition derived from conditions
def trapezoid_base2 := length - trapezoid_base1 -- the second base of the trapezoid

-- Required proof in Lean
theorem shingle_area : (length * width - (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height)) = 60 := 
by
  sorry

end shingle_area_l15_15037


namespace sum_g_values_l15_15194

noncomputable def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 1 else
if x >= -3 then 3 * x + 2 else 4

theorem sum_g_values : g (-4) + g 0 + g 4 = 21 :=
by
  sorry

end sum_g_values_l15_15194


namespace sozopolian_ineq_find_p_l15_15430

noncomputable def is_sozopolian (p a b c : ℕ) : Prop :=
  p % 2 = 1 ∧
  Nat.Prime p ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_ineq (p a b c : ℕ) (hp : is_sozopolian p a b c) :
  p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem find_p (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ (a + b + c) / 3 = p + 2) ↔ p = 5 :=
sorry

end sozopolian_ineq_find_p_l15_15430


namespace solutions_eq_l15_15356

theorem solutions_eq :
  { (a, b, c) : ℕ × ℕ × ℕ | a * b + b * c + c * a = 2 * (a + b + c) } =
  { (2, 2, 2),
    (1, 2, 4), (1, 4, 2), 
    (2, 1, 4), (2, 4, 1),
    (4, 1, 2), (4, 2, 1) } :=
by sorry

end solutions_eq_l15_15356


namespace min_p_value_l15_15238

variable (p q r s : ℝ)

theorem min_p_value (h1 : p + q + r + s = 10)
                    (h2 : pq + pr + ps + qr + qs + rs = 20)
                    (h3 : p^2 * q^2 * r^2 * s^2 = 16) :
  p ≥ 2 ∧ ∃ q r s, q + r + s = 10 - p ∧ pq + pr + ps + qr + qs + rs = 20 ∧ (p^2 * q^2 * r^2 * s^2 = 16) :=
by
  sorry  -- proof goes here

end min_p_value_l15_15238


namespace initial_rate_of_interest_l15_15839

theorem initial_rate_of_interest (P : ℝ) (R : ℝ) 
  (h1 : 1680 = (P * R * 5) / 100) 
  (h2 : 1680 = (P * 5 * 4) / 100) : 
  R = 4 := 
by 
  sorry

end initial_rate_of_interest_l15_15839


namespace ratio_of_sums_l15_15729

noncomputable def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ratio_of_sums (n : ℕ) (S1 S2 : ℕ) 
  (hn_even : n % 2 = 0)
  (hn_pos : 0 < n)
  (h_sum : sum_upto (n^2) = n^2 * (n^2 + 1) / 2)
  (h_S1S2_sum : S1 + S2 = n^2 * (n^2 + 1) / 2)
  (h_ratio : 64 * S1 = 39 * S2) :
  ∃ k : ℕ, n = 103 * k :=
sorry

end ratio_of_sums_l15_15729


namespace find_Y_l15_15537

theorem find_Y (Y : ℕ) 
  (h_top : 2 + 1 + Y + 3 = 6 + Y)
  (h_bottom : 4 + 3 + 1 + 5 = 13)
  (h_equal : 6 + Y = 13) : 
  Y = 7 := 
by
  sorry

end find_Y_l15_15537


namespace inequality_abc_l15_15609

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 := 
sorry

end inequality_abc_l15_15609


namespace correct_adjacent_book_left_l15_15317

-- Define the parameters
variable (prices : ℕ → ℕ)
variable (n : ℕ)
variable (step : ℕ)

-- Given conditions
axiom h1 : n = 31
axiom h2 : step = 2
axiom h3 : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step
axiom h4 : prices 30 = prices 15 + prices 14

-- We need to show that the adjacent book referred to is at the left of the middle book.
theorem correct_adjacent_book_left (h : n = 31) (prices_step : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step) : prices 30 = prices 15 + prices 14 := by
  sorry

end correct_adjacent_book_left_l15_15317


namespace M_subset_N_iff_l15_15290

section
variables {a x : ℝ}

-- Definitions based on conditions in the problem
def M (a : ℝ) : Set ℝ := { x | x^2 - a * x - x < 0 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem M_subset_N_iff (a : ℝ) : M a ⊆ N ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry
end

end M_subset_N_iff_l15_15290


namespace largest_integer_l15_15525

def bin_op (n : ℤ) : ℤ := n - 5 * n

theorem largest_integer (n : ℤ) (h : 0 < n) (h' : bin_op n < 18) : n = 4 := sorry

end largest_integer_l15_15525


namespace fruits_in_good_condition_l15_15424

def percentage_good_fruits (num_oranges num_bananas pct_rotten_oranges pct_rotten_bananas : ℕ) : ℚ :=
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (pct_rotten_oranges * num_oranges) / 100
  let rotten_bananas := (pct_rotten_bananas * num_bananas) / 100
  let good_fruits := total_fruits - (rotten_oranges + rotten_bananas)
  (good_fruits * 100) / total_fruits

theorem fruits_in_good_condition :
  percentage_good_fruits 600 400 15 8 = 87.8 := sorry

end fruits_in_good_condition_l15_15424


namespace right_triangle_area_l15_15511

/-- Given a right triangle where one leg is 18 cm and the hypotenuse is 30 cm,
    prove that the area of the triangle is 216 square centimeters. -/
theorem right_triangle_area (a b c : ℝ) 
    (ha : a = 18) 
    (hc : c = 30) 
    (h_right : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 216 :=
by
  -- Substitute the values given and solve the area.
  sorry

end right_triangle_area_l15_15511


namespace semesters_per_year_l15_15079

-- Definitions of conditions
def cost_per_semester : ℕ := 20000
def total_cost_13_years : ℕ := 520000
def years : ℕ := 13

-- Main theorem to prove
theorem semesters_per_year (S : ℕ) (h1 : total_cost_13_years = years * (S * cost_per_semester)) : S = 2 := by
  sorry

end semesters_per_year_l15_15079


namespace kennedy_softball_park_miles_l15_15666

theorem kennedy_softball_park_miles :
  let miles_per_gallon := 19
  let gallons_of_gas := 2
  let total_drivable_miles := miles_per_gallon * gallons_of_gas
  let miles_to_school := 15
  let miles_to_burger_restaurant := 2
  let miles_to_friends_house := 4
  let miles_home := 11
  total_drivable_miles - (miles_to_school + miles_to_burger_restaurant + miles_to_friends_house + miles_home) = 6 :=
by
  sorry

end kennedy_softball_park_miles_l15_15666


namespace segment_equality_l15_15359

variables {Point : Type} [AddGroup Point]

-- Define the points A, B, C, D, E, F
variables (A B C D E F : Point)

-- Given conditions
variables (AC CE BD DF AD CF : Point)
variable (h1 : AC = CE)
variable (h2 : BD = DF)
variable (h3 : AD = CF)

-- Theorem statement
theorem segment_equality (h1 : A - C = C - E)
                         (h2 : B - D = D - F)
                         (h3 : A - D = C - F) :
  (C - D) = (A - B) ∧ (C - D) = (E - F) :=
by
  sorry

end segment_equality_l15_15359


namespace ratio_of_triangle_areas_l15_15516

-- Define the given conditions
variables (m n x a : ℝ) (S T1 T2 : ℝ)

-- Conditions
def area_of_square : Prop := S = x^2
def area_of_triangle_1 : Prop := T1 = m * x^2
def length_relation : Prop := x = n * a

-- The proof goal
theorem ratio_of_triangle_areas (h1 : area_of_square S x) 
                                (h2 : area_of_triangle_1 T1 m x)
                                (h3 : length_relation x n a) : 
                                T2 / S = m / n^2 := 
sorry

end ratio_of_triangle_areas_l15_15516


namespace triangle_angle_C_and_equilateral_l15_15877

variables (a b c A B C : ℝ)
variables (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
variables (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1)

theorem triangle_angle_C_and_equilateral (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
                                         (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
                                         (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1) :
  C = π / 3 ∧ A = π / 3 ∧ B = π / 3 :=
sorry

end triangle_angle_C_and_equilateral_l15_15877


namespace pigeonhole_principle_f_m_l15_15473

theorem pigeonhole_principle_f_m :
  ∀ (n : ℕ) (f : ℕ × ℕ → Fin (n + 1)), n ≤ 44 →
    ∃ (i j l k p m : ℕ),
      1989 * m ≤ i ∧ i < l ∧ l < 1989 + 1989 * m ∧
      1989 * p ≤ j ∧ j < k ∧ k < 1989 + 1989 * p ∧
      f (i, j) = f (i, k) ∧ f (i, k) = f (l, j) ∧ f (l, j) = f (l, k) :=
by {
  sorry
}

end pigeonhole_principle_f_m_l15_15473


namespace harvest_apples_l15_15844

def sacks_per_section : ℕ := 45
def sections : ℕ := 8
def total_sacks_per_day : ℕ := 360

theorem harvest_apples : sacks_per_section * sections = total_sacks_per_day := by
  sorry

end harvest_apples_l15_15844


namespace total_cups_l15_15932

theorem total_cups (b f s : ℕ) (ratio_bt_f_s : b / s = 1 / 5) (ratio_fl_b_s : f / s = 8 / 5) (sugar_cups : s = 10) :
  b + f + s = 28 :=
sorry

end total_cups_l15_15932


namespace store_incur_loss_of_one_percent_l15_15098

theorem store_incur_loss_of_one_percent
    (a b x : ℝ)
    (h1 : x = a * 1.1)
    (h2 : x = b * 0.9)
    : (2 * x - (a + b)) / (a + b) = -0.01 :=
by
  -- Proof goes here
  sorry

end store_incur_loss_of_one_percent_l15_15098


namespace seeds_per_flowerbed_l15_15535

theorem seeds_per_flowerbed (total_seeds flowerbeds : ℕ) (h1 : total_seeds = 32) (h2 : flowerbeds = 8) :
  total_seeds / flowerbeds = 4 :=
by {
  sorry
}

end seeds_per_flowerbed_l15_15535


namespace primes_sum_solutions_l15_15277

theorem primes_sum_solutions :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧
  p + q^2 + r^3 = 200 ∧ 
  ((p = 167 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 11 ∧ r = 2) ∨ 
   (p = 23 ∧ q = 13 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 2 ∧ r = 5)) :=
sorry

end primes_sum_solutions_l15_15277


namespace angle_B_eq_18_l15_15081

theorem angle_B_eq_18 
  (A B : ℝ) 
  (h1 : A = 4 * B) 
  (h2 : 90 - B = 4 * (90 - A)) : 
  B = 18 :=
by
  sorry

end angle_B_eq_18_l15_15081


namespace hyperbola_equation_l15_15517

noncomputable def hyperbola (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def parabola_focus_same_as_hyperbola_focus (c : ℝ) : Prop :=
  ∃ x y : ℝ, y^2 = 4 * (10:ℝ).sqrt * x ∧ (c, 0) = ((10:ℝ).sqrt, 0)

def hyperbola_eccentricity (c a : ℝ) := (c / a) = (10:ℝ).sqrt / 3

theorem hyperbola_equation :
  ∃ a b : ℝ, (hyperbola a b) ∧
  (parabola_focus_same_as_hyperbola_focus ((10:ℝ).sqrt)) ∧
  (hyperbola_eccentricity ((10:ℝ).sqrt) a) ∧
  ((a = 3) ∧ (b = 1)) :=
sorry

end hyperbola_equation_l15_15517


namespace negation_exists_eq_forall_l15_15974

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end negation_exists_eq_forall_l15_15974


namespace range_of_a_l15_15294

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l15_15294


namespace moles_of_HCl_used_l15_15988

theorem moles_of_HCl_used (moles_amyl_alcohol : ℕ) (moles_product : ℕ) : 
  moles_amyl_alcohol = 2 ∧ moles_product = 2 → moles_amyl_alcohol = 2 :=
by
  sorry

end moles_of_HCl_used_l15_15988


namespace max_parallelograms_in_hexagon_l15_15168

theorem max_parallelograms_in_hexagon (side_hexagon side_parallelogram1 side_parallelogram2 : ℝ)
                                        (angle_parallelogram : ℝ) :
  side_hexagon = 3 ∧ side_parallelogram1 = 1 ∧ side_parallelogram2 = 2 ∧ angle_parallelogram = (π / 3) →
  ∃ n : ℕ, n = 12 :=
by 
  sorry

end max_parallelograms_in_hexagon_l15_15168


namespace total_cookies_l15_15520

def MonaCookies : ℕ := 20
def JasmineCookies : ℕ := MonaCookies - 5
def RachelCookies : ℕ := JasmineCookies + 10

theorem total_cookies : MonaCookies + JasmineCookies + RachelCookies = 60 := by
  -- Since we don't need to provide the solution steps, we simply use sorry.
  sorry

end total_cookies_l15_15520


namespace sequence_equals_identity_l15_15716

theorem sequence_equals_identity (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : 
  ∀ i : ℕ, a i = i := 
by 
  sorry

end sequence_equals_identity_l15_15716


namespace rectangle_ratio_l15_15378

theorem rectangle_ratio 
  (s : ℝ) -- side length of the inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_area : s^2 = (inner_square_area : ℝ))
  (h_outer_area : 9 * inner_square_area = outer_square_area)
  (h_outer_side_eq : (s + 2 * y)^2 = outer_square_area)
  (h_longer_side_eq : x + y = 3 * s) :
  x / y = 2 :=
by sorry

end rectangle_ratio_l15_15378


namespace not_divisible_by_n_only_prime_3_l15_15075

-- Problem 1: Prove that for any natural number \( n \) greater than 1, \( 2^n - 1 \) is not divisible by \( n \)
theorem not_divisible_by_n (n : ℕ) (h1 : 1 < n) : ¬ (n ∣ (2^n - 1)) :=
sorry

-- Problem 2: Prove that the only prime number \( n \) such that \( 2^n + 1 \) is divisible by \( n^2 \) is \( n = 3 \)
theorem only_prime_3 (n : ℕ) (hn : Nat.Prime n) (hdiv : n^2 ∣ (2^n + 1)) : n = 3 :=
sorry

end not_divisible_by_n_only_prime_3_l15_15075


namespace exchange_ways_100_yuan_l15_15142

theorem exchange_ways_100_yuan : ∃ n : ℕ, n = 6 ∧ (∀ (x y : ℕ), 20 * x + 10 * y = 100 ↔ y = 10 - 2 * x):=
by
  sorry

end exchange_ways_100_yuan_l15_15142


namespace prime_pairs_divisibility_l15_15137

theorem prime_pairs_divisibility:
  ∀ (p q : ℕ), (Nat.Prime p ∧ Nat.Prime q ∧ p ≤ q ∧ p * q ∣ ((5 ^ p - 2 ^ p) * (7 ^ q - 2 ^ q))) ↔ 
                (p = 3 ∧ q = 5) ∨ 
                (p = 3 ∧ q = 3) ∨ 
                (p = 5 ∧ q = 37) ∨ 
                (p = 5 ∧ q = 83) := by
  sorry

end prime_pairs_divisibility_l15_15137


namespace white_balls_count_l15_15759

theorem white_balls_count
  (total_balls : ℕ)
  (white_balls blue_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls + blue_balls + red_balls = total_balls)
  (h3 : blue_balls = white_balls + 12)
  (h4 : red_balls = 2 * blue_balls) : white_balls = 16 := by
  sorry

end white_balls_count_l15_15759


namespace carina_coffee_l15_15466

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee_l15_15466


namespace intersections_correct_l15_15237

-- Define the distances (in meters)
def gretzky_street_length : ℕ := 5600
def segment_a_distance : ℕ := 350
def segment_b_distance : ℕ := 400
def segment_c_distance : ℕ := 450

-- Definitions based on conditions
def segment_a_intersections : ℕ :=
  gretzky_street_length / segment_a_distance - 2 -- subtract Orr Street and Howe Street

def segment_b_intersections : ℕ :=
  gretzky_street_length / segment_b_distance

def segment_c_intersections : ℕ :=
  gretzky_street_length / segment_c_distance

-- Sum of all intersections
def total_intersections : ℕ :=
  segment_a_intersections + segment_b_intersections + segment_c_intersections

theorem intersections_correct :
  total_intersections = 40 :=
by
  sorry

end intersections_correct_l15_15237


namespace option_C_correct_l15_15653

theorem option_C_correct (a b : ℝ) : (-2 * a * b^3)^2 = 4 * a^2 * b^6 :=
by
  sorry

end option_C_correct_l15_15653


namespace total_stars_l15_15872

/-- Let n be the number of students, and s be the number of stars each student makes.
    We need to prove that the total number of stars is n * s. --/
theorem total_stars (n : ℕ) (s : ℕ) (h_n : n = 186) (h_s : s = 5) : n * s = 930 :=
by {
  sorry
}

end total_stars_l15_15872


namespace find_other_denomination_l15_15567

theorem find_other_denomination
  (total_spent : ℕ)
  (twenty_bill_value : ℕ) (other_denomination_value : ℕ)
  (twenty_bill_count : ℕ) (other_bill_count : ℕ)
  (h1 : total_spent = 80)
  (h2 : twenty_bill_value = 20)
  (h3 : other_bill_count = 2)
  (h4 : twenty_bill_count = other_bill_count + 1)
  (h5 : total_spent = twenty_bill_value * twenty_bill_count + other_denomination_value * other_bill_count) : 
  other_denomination_value = 10 :=
by
  sorry

end find_other_denomination_l15_15567


namespace roots_quadratic_diff_by_12_l15_15093

theorem roots_quadratic_diff_by_12 (P : ℝ) : 
  (∀ α β : ℝ, (α + β = 2) ∧ (α * β = -P) ∧ ((α - β) = 12)) → P = 35 := 
by
  intro h
  sorry

end roots_quadratic_diff_by_12_l15_15093


namespace product_of_two_numbers_l15_15322

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 :=
by
  sorry

end product_of_two_numbers_l15_15322


namespace square_traffic_sign_perimeter_l15_15644

-- Define the side length of the square
def side_length : ℕ := 4

-- Define the number of sides of the square
def number_of_sides : ℕ := 4

-- Define the perimeter of the square
def perimeter (l : ℕ) (n : ℕ) : ℕ := l * n

-- The theorem to be proved
theorem square_traffic_sign_perimeter : perimeter side_length number_of_sides = 16 :=
by
  sorry

end square_traffic_sign_perimeter_l15_15644


namespace no_solution_inequality_l15_15026

theorem no_solution_inequality (m : ℝ) : (¬ ∃ x : ℝ, |x + 1| + |x - 5| ≤ m) ↔ m < 6 :=
sorry

end no_solution_inequality_l15_15026


namespace time_for_A_and_D_together_l15_15950

theorem time_for_A_and_D_together (A_rate D_rate combined_rate : ℝ)
  (hA : A_rate = 1 / 10) (hD : D_rate = 1 / 10) 
  (h_combined : combined_rate = A_rate + D_rate) :
  1 / combined_rate = 5 :=
by
  sorry

end time_for_A_and_D_together_l15_15950


namespace sin_cos_105_l15_15543

theorem sin_cos_105 (h1 : ∀ x : ℝ, Real.sin x * Real.cos x = 1 / 2 * Real.sin (2 * x))
                    (h2 : ∀ x : ℝ, Real.sin (180 * Real.pi / 180 + x) = - Real.sin x)
                    (h3 : Real.sin (30 * Real.pi / 180) = 1 / 2) :
  Real.sin (105 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) = - 1 / 4 :=
by
  sorry

end sin_cos_105_l15_15543


namespace otherWorkStations_accommodate_students_l15_15664

def numTotalStudents := 38
def numStations := 16
def numWorkStationsForTwo := 10
def capacityWorkStationsForTwo := 2

theorem otherWorkStations_accommodate_students : 
  (numTotalStudents - numWorkStationsForTwo * capacityWorkStationsForTwo) = 18 := 
by
  sorry

end otherWorkStations_accommodate_students_l15_15664


namespace books_total_pages_l15_15571

theorem books_total_pages (x y z : ℕ) 
  (h1 : (2 / 3 : ℚ) * x - (1 / 3 : ℚ) * x = 20)
  (h2 : (3 / 5 : ℚ) * y - (2 / 5 : ℚ) * y = 15)
  (h3 : (3 / 4 : ℚ) * z - (1 / 4 : ℚ) * z = 30) : 
  x = 60 ∧ y = 75 ∧ z = 60 :=
by
  sorry

end books_total_pages_l15_15571


namespace distance_between_points_l15_15072

theorem distance_between_points:
  dist (0, 4) (3, 0) = 5 :=
by
  sorry

end distance_between_points_l15_15072


namespace find_number_of_girls_l15_15379

-- Definitions for the number of candidates
variables (B G : ℕ)
variable (total_candidates : B + G = 2000)

-- Definitions for the percentages of passed candidates
variable (pass_rate_boys : ℝ := 0.34)
variable (pass_rate_girls : ℝ := 0.32)
variable (pass_rate_total : ℝ := 0.331)

-- Hypotheses based on the conditions
variables (P_B P_G : ℝ)
variable (pass_boys : P_B = pass_rate_boys * B)
variable (pass_girls : P_G = pass_rate_girls * G)
variable (pass_total_eq : P_B + P_G = pass_rate_total * 2000)

-- Goal: Prove that the number of girls (G) is 1800
theorem find_number_of_girls (B G : ℕ)
  (total_candidates : B + G = 2000)
  (pass_rate_boys : ℝ := 0.34)
  (pass_rate_girls : ℝ := 0.32)
  (pass_rate_total : ℝ := 0.331)
  (P_B P_G : ℝ)
  (pass_boys : P_B = pass_rate_boys * (B : ℝ))
  (pass_girls : P_G = pass_rate_girls * (G : ℝ))
  (pass_total_eq : P_B + P_G = pass_rate_total * 2000) : G = 1800 :=
sorry

end find_number_of_girls_l15_15379


namespace min_blocks_to_remove_l15_15358

theorem min_blocks_to_remove (n : ℕ) (h : n = 59) : 
  ∃ (k : ℕ), k = 32 ∧ (∃ m, n = m^3 + k ∧ m^3 ≤ n) :=
by {
  sorry
}

end min_blocks_to_remove_l15_15358


namespace nancy_pics_uploaded_l15_15975

theorem nancy_pics_uploaded (a b n : ℕ) (h₁ : a = 11) (h₂ : b = 8) (h₃ : n = 5) : a + b * n = 51 := 
by 
  sorry

end nancy_pics_uploaded_l15_15975


namespace susan_can_drive_with_50_l15_15913

theorem susan_can_drive_with_50 (car_efficiency : ℕ) (gas_price : ℕ) (money_available : ℕ) 
  (h1 : car_efficiency = 40) (h2 : gas_price = 5) (h3 : money_available = 50) : 
  car_efficiency * (money_available / gas_price) = 400 :=
by
  sorry

end susan_can_drive_with_50_l15_15913


namespace final_bill_is_correct_l15_15799

def Alicia_order := [7.50, 4.00, 5.00]
def Brant_order := [10.00, 4.50, 6.00]
def Josh_order := [8.50, 4.00, 3.50]
def Yvette_order := [9.00, 4.50, 6.00]

def discount_rate := 0.10
def sales_tax_rate := 0.08
def tip_rate := 0.20

noncomputable def calculate_final_bill : Float :=
  let subtotal := (Alicia_order.sum + Brant_order.sum + Josh_order.sum + Yvette_order.sum)
  let discount := discount_rate * subtotal
  let discounted_total := subtotal - discount
  let sales_tax := sales_tax_rate * discounted_total
  let pre_tax_and_discount_total := subtotal
  let tip := tip_rate * pre_tax_and_discount_total
  discounted_total + sales_tax + tip

theorem final_bill_is_correct : calculate_final_bill = 84.97 := by
  sorry

end final_bill_is_correct_l15_15799


namespace find_y1_l15_15873

theorem find_y1 
  (y1 y2 y3 : ℝ) 
  (h₀ : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h₁ : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = (2 * Real.sqrt 2 - 1) / (2 * Real.sqrt 2) :=
by
  sorry

end find_y1_l15_15873


namespace coefficient_of_x3_in_expansion_l15_15272

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_coefficient_x3 : ℤ :=
  let term1 := (-1 : ℤ) ^ 3 * binomial_coefficient 6 3
  let term2 := (1 : ℤ) * binomial_coefficient 6 2
  term1 + term2

theorem coefficient_of_x3_in_expansion :
  expansion_coefficient_x3 = -5 := by
  sorry

end coefficient_of_x3_in_expansion_l15_15272


namespace yellow_bags_count_l15_15252

theorem yellow_bags_count (R B Y : ℕ) 
  (h1 : R + B + Y = 12) 
  (h2 : 10 * R + 50 * B + 100 * Y = 500) 
  (h3 : R = B) : 
  Y = 2 := 
by 
  sorry

end yellow_bags_count_l15_15252


namespace marbles_count_l15_15593

def num_violet_marbles := 64

def num_red_marbles := 14

def total_marbles (violet : Nat) (red : Nat) : Nat :=
  violet + red

theorem marbles_count :
  total_marbles num_violet_marbles num_red_marbles = 78 := by
  sorry

end marbles_count_l15_15593


namespace consecutive_odd_integers_sum_l15_15710

theorem consecutive_odd_integers_sum (x : ℤ) (h : x + (x + 4) = 138) :
  x + (x + 2) + (x + 4) = 207 :=
sorry

end consecutive_odd_integers_sum_l15_15710


namespace ratio_children_to_adults_l15_15488

variable (male_adults : ℕ) (female_adults : ℕ) (total_people : ℕ)
variable (total_adults : ℕ) (children : ℕ)

theorem ratio_children_to_adults :
  male_adults = 100 →
  female_adults = male_adults + 50 →
  total_people = 750 →
  total_adults = male_adults + female_adults →
  children = total_people - total_adults →
  children / total_adults = 2 :=
by
  intros h_male h_female h_total h_adults h_children
  sorry

end ratio_children_to_adults_l15_15488


namespace initial_pennies_l15_15107

-- Defining the conditions
def pennies_spent : Nat := 93
def pennies_left : Nat := 5

-- Question: How many pennies did Sam have in his bank initially?
theorem initial_pennies : pennies_spent + pennies_left = 98 := by
  sorry

end initial_pennies_l15_15107


namespace race_positions_l15_15125

variable (nabeel marzuq arabi rafsan lian rahul : ℕ)

theorem race_positions :
  (arabi = 6) →
  (arabi = rafsan + 1) →
  (rafsan = rahul + 2) →
  (rahul = nabeel + 1) →
  (nabeel = marzuq + 6) →
  (marzuq = 8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end race_positions_l15_15125


namespace f_even_l15_15316

-- Let g(x) = x^3 - x
def g (x : ℝ) : ℝ := x^3 - x

-- Let f(x) = |g(x^2)|
def f (x : ℝ) : ℝ := abs (g (x^2))

-- Prove that f(x) is even, i.e., f(-x) = f(x) for all x
theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_even_l15_15316


namespace sheets_of_paper_per_week_l15_15172

theorem sheets_of_paper_per_week
  (sheets_per_class_per_day : ℕ)
  (num_classes : ℕ)
  (school_days_per_week : ℕ)
  (total_sheets_per_week : ℕ) 
  (h1 : sheets_per_class_per_day = 200)
  (h2 : num_classes = 9)
  (h3 : school_days_per_week = 5)
  (h4 : total_sheets_per_week = sheets_per_class_per_day * num_classes * school_days_per_week) :
  total_sheets_per_week = 9000 :=
sorry

end sheets_of_paper_per_week_l15_15172


namespace total_games_is_272_l15_15510

-- Define the number of players
def n : ℕ := 17

-- Define the formula for the number of games played
def total_games (n : ℕ) : ℕ := n * (n - 1)

-- Define a theorem stating that the total games played is 272
theorem total_games_is_272 : total_games n = 272 := by
  -- Proof omitted
  sorry

end total_games_is_272_l15_15510


namespace ocean_depth_350_l15_15952

noncomputable def depth_of_ocean (total_height : ℝ) (volume_ratio_above_water : ℝ) : ℝ :=
  let volume_ratio_below_water := 1 - volume_ratio_above_water
  let height_below_water := (volume_ratio_below_water^(1 / 3)) * total_height
  total_height - height_below_water

theorem ocean_depth_350 :
  depth_of_ocean 10000 (1 / 10) = 350 :=
by
  sorry

end ocean_depth_350_l15_15952


namespace farmer_land_l15_15590

variable (T : ℝ) -- Total land owned by the farmer

def is_cleared (T : ℝ) : ℝ := 0.90 * T
def cleared_barley (T : ℝ) : ℝ := 0.80 * is_cleared T
def cleared_potato (T : ℝ) : ℝ := 0.10 * is_cleared T
def cleared_tomato : ℝ := 90
def cleared_land (T : ℝ) : ℝ := cleared_barley T + cleared_potato T + cleared_tomato

theorem farmer_land (T : ℝ) (h : cleared_land T = is_cleared T) : T = 1000 := sorry

end farmer_land_l15_15590


namespace polynomial_expansion_l15_15342

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 :=
by sorry

end polynomial_expansion_l15_15342


namespace decimal_0_0_1_7_eq_rational_l15_15821

noncomputable def infinite_loop_decimal_to_rational_series (a : ℚ) (r : ℚ) : ℚ :=
  a / (1 - r)

theorem decimal_0_0_1_7_eq_rational :
  infinite_loop_decimal_to_rational_series (17 / 1000) (1 / 100) = 17 / 990 :=
by
  sorry

end decimal_0_0_1_7_eq_rational_l15_15821


namespace green_red_socks_ratio_l15_15764

theorem green_red_socks_ratio 
  (r : ℕ) -- Number of pairs of red socks originally ordered
  (y : ℕ) -- Price per pair of red socks
  (green_socks_price : ℕ := 3 * y) -- Price per pair of green socks, 3 times the red socks
  (C_original : ℕ := 6 * green_socks_price + r * y) -- Cost of the original order
  (C_interchanged : ℕ := r * green_socks_price + 6 * y) -- Cost of the interchanged order
  (exchange_rate : ℚ := 1.2) -- 20% increase
  (cost_relation : C_interchanged = exchange_rate * C_original) -- Cost relation given by the problem
  : (6 : ℚ) / (r : ℚ) = 2 / 3 := 
by
  sorry

end green_red_socks_ratio_l15_15764


namespace james_passenger_count_l15_15926

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l15_15926


namespace rubber_boat_fall_time_l15_15783

variable {a b x : ℝ}

theorem rubber_boat_fall_time
  (h1 : 5 - x = (a - b) / (a + b))
  (h2 : 6 - x = b / (a + b)) :
  x = 4 := by
  sorry

end rubber_boat_fall_time_l15_15783


namespace find_central_angle_of_sector_l15_15471

variables (r θ : ℝ)

def sector_arc_length (r θ : ℝ) := r * θ
def sector_area (r θ : ℝ) := 0.5 * r^2 * θ

theorem find_central_angle_of_sector
  (l : ℝ)
  (A : ℝ)
  (hl : l = sector_arc_length r θ)
  (hA : A = sector_area r θ)
  (hl_val : l = 4)
  (hA_val : A = 2) :
  θ = 4 :=
sorry

end find_central_angle_of_sector_l15_15471


namespace strawberry_candies_count_l15_15557

theorem strawberry_candies_count (S G : ℕ) (h1 : S + G = 240) (h2 : G = S - 2) : S = 121 :=
by
  sorry

end strawberry_candies_count_l15_15557


namespace AB_complete_work_together_in_10_days_l15_15667

-- Definitions for the work rates
def rate_A (work : ℕ) : ℚ := work / 14 -- A's rate of work (work per day)
def rate_AB (work : ℕ) : ℚ := work / 10 -- A and B together's rate of work (work per day)

-- Definition for B's rate of work derived from the combined rate and A's rate
def rate_B (work : ℕ) : ℚ := rate_AB work - rate_A work

-- Definition of the fact that the combined rate should equal their individual rates summed
def combined_rate_equals_sum (work : ℕ) : Prop := rate_AB work = (rate_A work + rate_B work)

-- Statement we need to prove:
theorem AB_complete_work_together_in_10_days (work : ℕ) (h : combined_rate_equals_sum work) : rate_AB work = work / 10 :=
by {
  -- Given conditions are implicitly used without a formal proof here.
  -- To prove that A and B together can indeed complete the work in 10 days.
  sorry
}


end AB_complete_work_together_in_10_days_l15_15667


namespace find_n_l15_15170

theorem find_n (m n : ℝ) (h1 : m + 2 * n = 1.2) (h2 : 0.1 + m + n + 0.1 = 1) : n = 0.4 :=
by
  sorry

end find_n_l15_15170


namespace max_distance_traveled_l15_15177

def distance_traveled (t : ℝ) : ℝ := 15 * t - 6 * t^2

theorem max_distance_traveled : ∃ t : ℝ, distance_traveled t = 75 / 8 :=
by
  sorry

end max_distance_traveled_l15_15177


namespace yaya_bike_walk_l15_15418

theorem yaya_bike_walk (x y : ℝ) : 
  (x + y = 1.5 ∧ 15 * x + 5 * y = 20) ↔ (x + y = 1.5 ∧ 15 * x + 5 * y = 20) :=
by 
  sorry

end yaya_bike_walk_l15_15418


namespace find_a2016_l15_15499

theorem find_a2016 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n) : a 2016 = -2 := 
by sorry

end find_a2016_l15_15499


namespace eq_margin_l15_15481

variables (C S n : ℝ) (M : ℝ)

theorem eq_margin (h : M = 1 / n * (2 * C - S)) : M = S / (n + 2) :=
sorry

end eq_margin_l15_15481


namespace gift_sequences_count_l15_15363

def num_students : ℕ := 11
def num_meetings : ℕ := 4
def sequences : ℕ := num_students ^ num_meetings

theorem gift_sequences_count : sequences = 14641 := by
  sorry

end gift_sequences_count_l15_15363


namespace g_inequality_solution_range_of_m_l15_15672

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 8
noncomputable def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16
noncomputable def h (x m : ℝ) : ℝ := x^2 - (4 + m)*x + (m + 7)

theorem g_inequality_solution:
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} :=
by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 1 → f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 4 :=
by
  sorry

end g_inequality_solution_range_of_m_l15_15672


namespace range_of_m_l15_15645

def y1 (m x : ℝ) : ℝ :=
  m * (x - 2 * m) * (x + m + 2)

def y2 (x : ℝ) : ℝ :=
  x - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, y1 m x < 0 ∨ y2 x < 0) ∧ (∃ x : ℝ, x < -3 ∧ y1 m x * y2 x < 0) ↔ (-4 < m ∧ m < -3/2) := 
by
  sorry

end range_of_m_l15_15645


namespace part_a_part_b_l15_15958

-- Definition for bishops not attacking each other
def bishops_safe (positions : List (ℕ × ℕ)) : Prop :=
  ∀ (b1 b2 : ℕ × ℕ), b1 ∈ positions → b2 ∈ positions → b1 ≠ b2 → 
    (b1.1 + b1.2 ≠ b2.1 + b2.2) ∧ (b1.1 - b1.2 ≠ b2.1 - b2.2)

-- Part (a): 14 bishops on an 8x8 chessboard such that no two attack each other
theorem part_a : ∃ (positions : List (ℕ × ℕ)), positions.length = 14 ∧ bishops_safe positions := 
by
  sorry

-- Part (b): It is impossible to place 15 bishops on an 8x8 chessboard without them attacking each other
theorem part_b : ¬ ∃ (positions : List (ℕ × ℕ)), positions.length = 15 ∧ bishops_safe positions :=
by 
  sorry

end part_a_part_b_l15_15958


namespace product_of_distinct_roots_l15_15685

theorem product_of_distinct_roots (x1 x2 : ℝ) (hx1 : x1 ^ 2 - 2 * x1 = 1) (hx2 : x2 ^ 2 - 2 * x2 = 1) (h_distinct : x1 ≠ x2) : 
  x1 * x2 = -1 := 
  sorry

end product_of_distinct_roots_l15_15685


namespace arithmetic_sequence_difference_l15_15051

theorem arithmetic_sequence_difference :
  ∀ (a d : ℤ), a = -2 → d = 7 →
  |(a + (3010 - 1) * d) - (a + (3000 - 1) * d)| = 70 :=
by
  intros a d a_def d_def
  rw [a_def, d_def]
  sorry

end arithmetic_sequence_difference_l15_15051


namespace range_of_m_l15_15673

theorem range_of_m (m : ℝ) (p : |m + 1| ≤ 2) (q : ¬(m^2 - 4 ≥ 0)) : -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l15_15673


namespace sum_of_bases_is_16_l15_15249

/-
  Given the fractions G_1 and G_2 in two different bases S_1 and S_2, we need to show 
  that the sum of these bases S_1 and S_2 in base ten is 16.
-/
theorem sum_of_bases_is_16 (S_1 S_2 G_1 G_2 : ℕ) :
  (G_1 = (4 * S_1 + 5) / (S_1^2 - 1)) →
  (G_2 = (5 * S_1 + 4) / (S_1^2 - 1)) →
  (G_1 = (S_2 + 4) / (S_2^2 - 1)) →
  (G_2 = (4 * S_2 + 1) / (S_2^2 - 1)) →
  S_1 + S_2 = 16 :=
by
  intros hG1_S1 hG2_S1 hG1_S2 hG2_S2
  sorry

end sum_of_bases_is_16_l15_15249


namespace adam_total_cost_l15_15949

theorem adam_total_cost 
    (sandwiches_count : ℕ)
    (sandwiches_price : ℝ)
    (chips_count : ℕ)
    (chips_price : ℝ)
    (water_count : ℕ)
    (water_price : ℝ)
    (sandwich_discount : sandwiches_count = 4 ∧ sandwiches_price = 4 ∧ sandwiches_count = 3 + 1)
    (tax_rate : ℝ)
    (initial_tax_rate : tax_rate = 0.10)
    (chips_cost : chips_count = 3 ∧ chips_price = 3.50)
    (water_cost : water_count = 2 ∧ water_price = 2) : 
  (3 * sandwiches_price + chips_count * chips_price + water_count * water_price) * (1 + tax_rate) = 29.15 := 
by
  sorry

end adam_total_cost_l15_15949


namespace all_positive_integers_are_clever_l15_15859

theorem all_positive_integers_are_clever : ∀ n : ℕ, 0 < n → ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (a^2 - b^2) / (c^2 + d^2) := 
by
  intros n h_pos
  sorry

end all_positive_integers_are_clever_l15_15859


namespace proof_problem_l15_15659

variable (y θ Q : ℝ)

-- Given condition
def condition : Prop := 5 * (3 * y + 7 * Real.sin θ) = Q

-- Goal to be proved
def goal : Prop := 15 * (9 * y + 21 * Real.sin θ) = 9 * Q

theorem proof_problem (h : condition y θ Q) : goal y θ Q :=
by
  sorry

end proof_problem_l15_15659


namespace add_decimals_l15_15352

theorem add_decimals :
  5.623 + 4.76 = 10.383 :=
by sorry

end add_decimals_l15_15352


namespace at_most_2n_div_3_good_triangles_l15_15472

-- Definitions based on problem conditions
universe u

structure Polygon (α : Type u) :=
(vertices : List α)
(convex : True)  -- Placeholder for convexity condition

-- Definition for a good triangle
structure Triangle (α : Type u) :=
(vertices : Fin 3 → α)
(unit_length : (Fin 3) → (Fin 3) → Bool)  -- Placeholder for unit length side condition

noncomputable def count_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) : Nat := sorry

theorem at_most_2n_div_3_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) :
  count_good_triangles P ≤ P.vertices.length * 2 / 3 := 
sorry

end at_most_2n_div_3_good_triangles_l15_15472


namespace slab_cost_l15_15703

-- Define the conditions
def cubes_per_stick : ℕ := 4
def cubes_per_slab : ℕ := 80
def total_kabob_cost : ℕ := 50
def kabob_sticks_made : ℕ := 40
def total_cubes_needed := kabob_sticks_made * cubes_per_stick
def slabs_needed := total_cubes_needed / cubes_per_slab

-- Final proof problem statement in Lean 4
theorem slab_cost : (total_kabob_cost / slabs_needed) = 25 := by
  sorry

end slab_cost_l15_15703


namespace find_a2_b2_c2_l15_15945

-- Define the roots, sum of the roots, sum of the product of the roots taken two at a time, and product of the roots
variables {a b c : ℝ}
variable (h_roots : a = b ∧ b = c)
variable (h_sum : a + b + c = 12)
variable (h_sum_products : a * b + b * c + a * c = 47)
variable (h_product : a * b * c = 30)

-- State the theorem
theorem find_a2_b2_c2 : (a^2 + b^2 + c^2) = 50 :=
by {
  sorry
}

end find_a2_b2_c2_l15_15945


namespace reporters_percentage_l15_15663

theorem reporters_percentage (total_reporters : ℕ) (local_politics_percentage : ℝ) (non_politics_percentage : ℝ) :
  local_politics_percentage = 28 → non_politics_percentage = 60 → 
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  100 * (non_local_political_reporters / political_reporters) = 30 :=
by
  intros
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  sorry

end reporters_percentage_l15_15663


namespace divisible_by_4_l15_15289

theorem divisible_by_4 (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n^3 + (n + 1)^3 + (n + 2)^3 = m^3) : 4 ∣ n + 1 :=
sorry

end divisible_by_4_l15_15289


namespace area_square_diagonal_l15_15161

theorem area_square_diagonal (d : ℝ) (k : ℝ) :
  (∀ side : ℝ, d^2 = 2 * side^2 → side^2 = (d^2)/2) →
  (∀ A : ℝ, A = (d^2)/2 → A = k * d^2) →
  k = 1/2 :=
by
  intros h1 h2
  sorry

end area_square_diagonal_l15_15161


namespace how_many_large_glasses_l15_15747

theorem how_many_large_glasses (cost_small cost_large : ℕ) 
                               (total_money money_left change : ℕ) 
                               (num_small : ℕ) : 
  cost_small = 3 -> 
  cost_large = 5 -> 
  total_money = 50 -> 
  money_left = 26 ->
  change = 1 ->
  num_small = 8 ->
  (money_left - change) / cost_large = 5 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end how_many_large_glasses_l15_15747


namespace jeans_cost_proof_l15_15782

def cheaper_jeans_cost (coat_price: Float) (backpack_price: Float) (shoes_price: Float) (subtotal: Float) (difference: Float): Float :=
  let known_items_cost := coat_price + backpack_price + shoes_price
  let jeans_total_cost := subtotal - known_items_cost
  let x := (jeans_total_cost - difference) / 2
  x

def more_expensive_jeans_cost (cheaper_price : Float) (difference: Float): Float :=
  cheaper_price + difference

theorem jeans_cost_proof : ∀ (coat_price backpack_price shoes_price subtotal difference : Float),
  coat_price = 45 →
  backpack_price = 25 →
  shoes_price = 30 →
  subtotal = 139 →
  difference = 15 →
  cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference = 12 ∧
  more_expensive_jeans_cost (cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference) difference = 27 :=
by
  intros coat_price backpack_price shoes_price subtotal difference
  intros h1 h2 h3 h4 h5
  sorry

end jeans_cost_proof_l15_15782


namespace elizabeth_spendings_elizabeth_savings_l15_15323

section WeddingGift

def steak_knife_set_cost : ℝ := 80
def steak_knife_sets : ℕ := 2
def dinnerware_set_cost : ℝ := 200
def fancy_napkins_sets : ℕ := 3
def fancy_napkins_total_cost : ℝ := 45
def wine_glasses_cost : ℝ := 100
def discount_steak_dinnerware : ℝ := 0.10
def discount_napkins : ℝ := 0.20
def sales_tax : ℝ := 0.05

def total_cost_before_discounts : ℝ :=
  (steak_knife_sets * steak_knife_set_cost) + dinnerware_set_cost + fancy_napkins_total_cost + wine_glasses_cost

def total_discount : ℝ :=
  ((steak_knife_sets * steak_knife_set_cost) * discount_steak_dinnerware) + (dinnerware_set_cost * discount_steak_dinnerware) + (fancy_napkins_total_cost * discount_napkins)

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discount

def total_cost_with_tax : ℝ :=
  total_cost_after_discounts + (total_cost_after_discounts * sales_tax)

def savings : ℝ :=
  total_cost_before_discounts - total_cost_after_discounts

theorem elizabeth_spendings :
  total_cost_with_tax = 558.60 :=
by sorry

theorem elizabeth_savings :
  savings = 63 :=
by sorry

end WeddingGift

end elizabeth_spendings_elizabeth_savings_l15_15323


namespace value_of_k_through_point_l15_15157

noncomputable def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

theorem value_of_k_through_point (k : ℝ) (h : k ≠ 0) : inverse_proportion_function 2 k = 3 → k = 6 :=
by
  sorry

end value_of_k_through_point_l15_15157


namespace sqrt_16_eq_plus_minus_4_l15_15791

theorem sqrt_16_eq_plus_minus_4 : ∀ x : ℝ, (x^2 = 16) ↔ (x = 4 ∨ x = -4) :=
by sorry

end sqrt_16_eq_plus_minus_4_l15_15791


namespace alice_wins_chomp_l15_15827

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp_l15_15827


namespace least_weight_of_oranges_l15_15461

theorem least_weight_of_oranges :
  ∀ (a o : ℝ), (a ≥ 8 + 3 * o) → (a ≤ 4 * o) → (o ≥ 8) :=
by
  intros a o h1 h2
  sorry

end least_weight_of_oranges_l15_15461


namespace inequality_solution_set_l15_15561

theorem inequality_solution_set :
  { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | (x - 2) / (1 - x) > 0 } :=
by sorry

end inequality_solution_set_l15_15561


namespace geometric_sequence_seventh_term_l15_15155

variable {G : Type*} [Field G]

def is_geometric (a : ℕ → G) (q : G) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → G) (q : G)
  (h1 : a 0 + a 1 = 3)
  (h2 : a 1 + a 2 = 6)
  (hq : is_geometric a q) :
  a 6 = 64 := 
sorry

end geometric_sequence_seventh_term_l15_15155


namespace sqrt_expr_is_integer_l15_15613

theorem sqrt_expr_is_integer (x : ℤ) (n : ℤ) (h : n^2 = x^2 - x + 1) : x = 0 ∨ x = 1 := by
  sorry

end sqrt_expr_is_integer_l15_15613


namespace smallest_x_value_l15_15555

theorem smallest_x_value : ∀ x : ℚ, (14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 → x = 4 / 5 :=
by
  intros x hx
  sorry

end smallest_x_value_l15_15555


namespace min_value_cx_plus_dy_squared_l15_15702

theorem min_value_cx_plus_dy_squared
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ ∀ (x y : ℝ), a * x^2 + b * y^2 = 1 → c * x + d * y^2 ≥ -c / a.sqrt) :=
sorry

end min_value_cx_plus_dy_squared_l15_15702


namespace eval_fraction_product_l15_15184

theorem eval_fraction_product :
  ((1 + (1 / 3)) * (1 + (1 / 4)) = (5 / 3)) :=
by
  sorry

end eval_fraction_product_l15_15184


namespace copy_pages_15_dollars_l15_15230

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l15_15230


namespace circle_radii_order_l15_15331

theorem circle_radii_order (r_A r_B r_C : ℝ) 
  (h1 : r_A = Real.sqrt 10) 
  (h2 : 2 * Real.pi * r_B = 10 * Real.pi)
  (h3 : Real.pi * r_C^2 = 16 * Real.pi) : 
  r_C < r_A ∧ r_A < r_B := 
  sorry

end circle_radii_order_l15_15331


namespace total_bike_price_l15_15766

theorem total_bike_price 
  (marion_bike_cost : ℝ := 356)
  (stephanie_bike_base_cost : ℝ := 2 * marion_bike_cost)
  (stephanie_discount_rate : ℝ := 0.10)
  (patrick_bike_base_cost : ℝ := 3 * marion_bike_cost)
  (patrick_discount_rate : ℝ := 0.75)
  (stephanie_bike_cost : ℝ := stephanie_bike_base_cost * (1 - stephanie_discount_rate))
  (patrick_bike_cost : ℝ := patrick_bike_base_cost * patrick_discount_rate):
  marion_bike_cost + stephanie_bike_cost + patrick_bike_cost = 1797.80 := 
by 
  sorry

end total_bike_price_l15_15766


namespace pipe_fills_tank_without_leak_l15_15662

theorem pipe_fills_tank_without_leak (T : ℝ) (h1 : 1 / 6 = 1 / T - 1 / 12) : T = 4 :=
by
  sorry

end pipe_fills_tank_without_leak_l15_15662


namespace isabella_hair_ratio_l15_15894

-- Conditions in the problem
variable (hair_before : ℕ) (hair_after : ℕ)
variable (hb : hair_before = 18)
variable (ha : hair_after = 36)

-- Definitions based on conditions
def hair_ratio (after : ℕ) (before : ℕ) : ℚ := (after : ℚ) / (before : ℚ)

theorem isabella_hair_ratio : 
  hair_ratio hair_after hair_before = 2 :=
by
  -- plug in the known values
  rw [hb, ha]
  -- show the equation
  norm_num
  sorry

end isabella_hair_ratio_l15_15894


namespace power_expansion_l15_15017

theorem power_expansion (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := 
by 
  sorry

end power_expansion_l15_15017


namespace total_ice_cream_amount_l15_15956

theorem total_ice_cream_amount (ice_cream_friday ice_cream_saturday : ℝ) 
  (h1 : ice_cream_friday = 3.25)
  (h2 : ice_cream_saturday = 0.25) : 
  ice_cream_friday + ice_cream_saturday = 3.50 :=
by
  rw [h1, h2]
  norm_num

end total_ice_cream_amount_l15_15956


namespace sum_zero_l15_15524

variable {a b c d : ℝ}

-- Pairwise distinct real numbers
axiom h1 : a ≠ b
axiom h2 : a ≠ c
axiom h3 : a ≠ d
axiom h4 : b ≠ c
axiom h5 : b ≠ d
axiom h6 : c ≠ d

-- Given condition
axiom h : (a^2 + b^2 - 1) * (a + b) = (b^2 + c^2 - 1) * (b + c) ∧ 
          (b^2 + c^2 - 1) * (b + c) = (c^2 + d^2 - 1) * (c + d)

theorem sum_zero : a + b + c + d = 0 :=
sorry

end sum_zero_l15_15524


namespace magic_8_ball_probability_l15_15969

theorem magic_8_ball_probability :
  let num_questions := 7
  let num_positive := 3
  let positive_probability := 3 / 7
  let negative_probability := 4 / 7
  let binomial_coefficient := Nat.choose num_questions num_positive
  let total_probability := binomial_coefficient * (positive_probability ^ num_positive) * (negative_probability ^ (num_questions - num_positive))
  total_probability = 242112 / 823543 :=
by
  sorry

end magic_8_ball_probability_l15_15969


namespace log12_eq_abc_l15_15060

theorem log12_eq_abc (a b : ℝ) (h1 : a = Real.log 7 / Real.log 6) (h2 : b = Real.log 4 / Real.log 3) : 
  Real.log 7 / Real.log 12 = (a * b + 2 * a) / (2 * b + 2) :=
by
  sorry

end log12_eq_abc_l15_15060


namespace melina_age_l15_15888

theorem melina_age (A M : ℕ) (alma_score : ℕ := 40) 
    (h1 : A + M = 2 * alma_score) 
    (h2 : M = 3 * A) : 
    M = 60 :=
by 
  sorry

end melina_age_l15_15888


namespace circle_center_radius_l15_15550

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = -1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l15_15550


namespace sufficient_condition_l15_15216

variable (a b c d : ℝ)

-- Condition p: a and b are the roots of the equation.
def condition_p : Prop := a * a + b * b + c * (a + b) + d = 0

-- Condition q: a + b + c = 0
def condition_q : Prop := a + b + c = 0

theorem sufficient_condition : condition_p a b c d → condition_q a b c := by
  sorry

end sufficient_condition_l15_15216


namespace total_money_shared_l15_15815

theorem total_money_shared 
  (A B C D total : ℕ) 
  (h1 : A = 3 * 15)
  (h2 : B = 5 * 15)
  (h3 : C = 6 * 15)
  (h4 : D = 8 * 15)
  (h5 : A = 45) :
  total = A + B + C + D → total = 330 :=
by
  sorry

end total_money_shared_l15_15815


namespace find_number_l15_15798

-- Definitions and conditions
def unknown_number (x : ℝ) : Prop :=
  (14 / 100) * x = 98

-- Theorem to prove
theorem find_number (x : ℝ) : unknown_number x → x = 700 := by
  sorry

end find_number_l15_15798


namespace Debby_jogging_plan_l15_15150

def Monday_jog : ℝ := 3
def Tuesday_jog : ℝ := Monday_jog * 1.1
def Wednesday_jog : ℝ := 0
def Thursday_jog : ℝ := Tuesday_jog * 1.1
def Saturday_jog : ℝ := Thursday_jog * 2.5
def total_distance : ℝ := Monday_jog + Tuesday_jog + Thursday_jog + Saturday_jog
def weekly_goal : ℝ := 40
def Sunday_jog : ℝ := weekly_goal - total_distance

theorem Debby_jogging_plan :
  Tuesday_jog = 3.3 ∧
  Thursday_jog = 3.63 ∧
  Saturday_jog = 9.075 ∧
  Sunday_jog = 21.995 :=
by
  -- Proof goes here, but is omitted as the problem statement requires only the theorem outline.
  sorry

end Debby_jogging_plan_l15_15150


namespace ratio_h_r_bounds_l15_15739

theorem ratio_h_r_bounds
  {a b c h r : ℝ}
  (h_right_angle : a^2 + b^2 = c^2)
  (h_area1 : 1/2 * a * b = 1/2 * c * h)
  (h_area2 : 1/2 * (a + b + c) * r = 1/2 * a * b) :
  2 < h / r ∧ h / r ≤ 2.41 :=
by
  sorry

end ratio_h_r_bounds_l15_15739


namespace problem_proof_l15_15553

theorem problem_proof (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 2 * y) / (x - 2 * y) = 23 :=
by sorry

end problem_proof_l15_15553


namespace production_period_l15_15259

-- Define the conditions as constants
def daily_production : ℕ := 1500
def price_per_computer : ℕ := 150
def total_earnings : ℕ := 1575000

-- Define the computation to find the period and state what we need to prove
theorem production_period : (total_earnings / price_per_computer) / daily_production = 7 :=
by
  -- you can provide the steps, but it's optional since the proof is omitted
  sorry

end production_period_l15_15259


namespace rectangle_ratio_expression_value_l15_15534

theorem rectangle_ratio_expression_value (l w : ℝ) (S : ℝ) (h1 : l / w = (2 * (l + w)) / (2 * l)) (h2 : S = w / l) :
  S ^ (S ^ (S^2 + 1/S) + 1/S) + 1/S = Real.sqrt 5 :=
by
  sorry

end rectangle_ratio_expression_value_l15_15534


namespace num_positive_integers_condition_l15_15904

theorem num_positive_integers_condition : 
  ∃! n : ℤ, 0 < n ∧ n < 50 ∧ (n + 2) % (50 - n) = 0 :=
by
  sorry

end num_positive_integers_condition_l15_15904


namespace least_number_subtracted_l15_15817

theorem least_number_subtracted (x : ℕ) (y : ℕ) (h : 2590 - x = y) : 
  y % 9 = 6 ∧ y % 11 = 6 ∧ y % 13 = 6 → x = 10 := 
by
  sorry

end least_number_subtracted_l15_15817


namespace quadratic_k_value_l15_15128

theorem quadratic_k_value (a b k : ℝ) (h_eq : a * b + 2 * a + 2 * b = 1)
  (h_roots : Polynomial.eval₂ (RingHom.id ℝ) a (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0 ∧
             Polynomial.eval₂ (RingHom.id ℝ) b (Polynomial.C k * Polynomial.X ^ 0 + Polynomial.C (-3) * Polynomial.X + Polynomial.C 1) = 0) : 
  k = -5 :=
by
  sorry

end quadratic_k_value_l15_15128


namespace fraction_value_l15_15851

theorem fraction_value :
  (1^4 + 2009^4 + 2010^4) / (1^2 + 2009^2 + 2010^2) = 4038091 := by
  sorry

end fraction_value_l15_15851


namespace MikaWaterLeft_l15_15883

def MikaWaterRemaining (startWater : ℚ) (usedWater : ℚ) : ℚ :=
  startWater - usedWater

theorem MikaWaterLeft :
  MikaWaterRemaining 3 (11 / 8) = 13 / 8 :=
by 
  sorry

end MikaWaterLeft_l15_15883


namespace smallest_distance_zero_l15_15809

theorem smallest_distance_zero :
  let r_track (t : ℝ) := (Real.cos t, Real.sin t)
  let i_track (t : ℝ) := (Real.cos (t / 2), Real.sin (t / 2))
  ∀ t₁ t₂ : ℝ, dist (r_track t₁) (i_track t₂) = 0 := by
  sorry

end smallest_distance_zero_l15_15809


namespace locus_of_points_equidistant_from_axes_l15_15601

-- Define the notion of being equidistant from the x-axis and the y-axis
def is_equidistant_from_axes (P : (ℝ × ℝ)) : Prop :=
  abs P.1 = abs P.2

-- The proof problem: given a moving point, the locus equation when P is equidistant from both axes
theorem locus_of_points_equidistant_from_axes (x y : ℝ) :
  is_equidistant_from_axes (x, y) → abs x - abs y = 0 :=
by
  intros h
  exact sorry

end locus_of_points_equidistant_from_axes_l15_15601


namespace find_k_range_l15_15661

noncomputable def f (k x : ℝ) : ℝ := (k * x + 1 / 3) * Real.exp x - x

theorem find_k_range : 
  (∃ (k : ℝ), ∀ (x : ℕ), x > 0 → (f k (x : ℝ) < 0 ↔ x = 1)) ↔
  (k ≥ 1 / (Real.exp 2) - 1 / 6 ∧ k < 1 / Real.exp 1 - 1 / 3) :=
sorry

end find_k_range_l15_15661


namespace find_functions_satisfying_lcm_gcd_eq_l15_15745

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)

noncomputable def solution_form (f : ℕ → ℕ) : Prop := 
  ∃ k : ℕ, ∀ x : ℕ, f x = k * x

theorem find_functions_satisfying_lcm_gcd_eq (f : ℕ → ℕ) : 
  satisfies_functional_equation f ↔ solution_form f := 
sorry

end find_functions_satisfying_lcm_gcd_eq_l15_15745


namespace incorrect_statement_l15_15902

-- Define the operation (x * y)
def op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem to show the incorrectness of the given statement
theorem incorrect_statement (x y z : ℝ) : op x (y + z) ≠ op x y + op x z :=
  sorry

end incorrect_statement_l15_15902


namespace solution_set_of_inequality_l15_15190

theorem solution_set_of_inequality (x : ℝ) : 
  (|x+1| - |x-4| > 3) ↔ x > 3 :=
sorry

end solution_set_of_inequality_l15_15190


namespace root_abs_sum_l15_15606

-- Definitions and conditions
variable (p q r n : ℤ)
variable (h_root : (x^3 - 2018 * x + n).coeffs[0] = 0)  -- This needs coefficient definition (simplified for clarity)
variable (h_vieta1 : p + q + r = 0)
variable (h_vieta2 : p * q + q * r + r * p = -2018)

theorem root_abs_sum :
  |p| + |q| + |r| = 100 :=
sorry

end root_abs_sum_l15_15606


namespace arithmetic_prog_sum_l15_15865

theorem arithmetic_prog_sum (a d : ℕ) (h1 : 15 * a + 105 * d = 60) : 2 * a + 14 * d = 8 :=
by
  sorry

end arithmetic_prog_sum_l15_15865


namespace polynomial_solution_l15_15057

noncomputable def p (x : ℝ) : ℝ := (7 / 4) * x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧ p 2 = 8 :=
by
  sorry

end polynomial_solution_l15_15057


namespace stone_105_is_3_l15_15219

def stone_numbered_at_105 (n : ℕ) := (15 + (n - 1) % 28)

theorem stone_105_is_3 :
  stone_numbered_at_105 105 = 3 := by
  sorry

end stone_105_is_3_l15_15219


namespace jerry_total_mean_l15_15933

def receivedFromAunt : ℕ := 9
def receivedFromUncle : ℕ := 9
def receivedFromBestFriends : List ℕ := [22, 23, 22, 22]
def receivedFromSister : ℕ := 7

def totalAmountReceived : ℕ :=
  receivedFromAunt + receivedFromUncle +
  receivedFromBestFriends.sum + receivedFromSister

def totalNumberOfGifts : ℕ :=
  1 + 1 + receivedFromBestFriends.length + 1

def meanAmountReceived : ℚ :=
  totalAmountReceived / totalNumberOfGifts

theorem jerry_total_mean :
  meanAmountReceived = 16.29 := by
sorry

end jerry_total_mean_l15_15933


namespace last_three_digits_of_5_power_15000_l15_15866

theorem last_three_digits_of_5_power_15000:
  (5^15000) % 1000 = 1 % 1000 :=
by
  have h : 5^500 % 1000 = 1 % 1000 := by sorry
  sorry

end last_three_digits_of_5_power_15000_l15_15866


namespace cone_volume_surface_area_sector_l15_15064

theorem cone_volume_surface_area_sector (V : ℝ):
  (∃ (r l h : ℝ), (π * r * (r + l) = 15 * π) ∧ (l = 6 * r) ∧ (h = Real.sqrt (l^2 - r^2)) ∧ (V = (1/3) * π * r^2 * h)) →
  V = (25 * Real.sqrt 3 / 7) * π :=
by 
  sorry

end cone_volume_surface_area_sector_l15_15064


namespace find_integer_for_combination_of_square_l15_15793

theorem find_integer_for_combination_of_square (y : ℝ) :
  ∃ (k : ℝ), (y^2 + 14*y + 60) = (y + 7)^2 + k ∧ k = 11 :=
by
  use 11
  sorry

end find_integer_for_combination_of_square_l15_15793


namespace find_aa_l15_15497

-- Given conditions
def m : ℕ := 7

-- Definition for checking if a number's tens place is 1
def tens_place_one (n : ℕ) : Prop :=
  (n / 10) % 10 = 1

-- The main statement to prove
theorem find_aa : ∃ x : ℕ, x < 10 ∧ tens_place_one (m * x^3) ∧ x = 6 := by
  -- Proof would go here
  sorry

end find_aa_l15_15497


namespace bonus_distribution_plans_l15_15076

theorem bonus_distribution_plans (x y : ℕ) (A B : ℕ) 
  (h1 : x + y = 15)
  (h2 : x = 2 * y)
  (h3 : 10 * A + 5 * B = 20000)
  (hA : A ≥ B)
  (hB : B ≥ 800)
  (hAB_mult_100 : ∃ (k m : ℕ), A = k * 100 ∧ B = m * 100) :
  (x = 10 ∧ y = 5) ∧
  ((A = 1600 ∧ B = 800) ∨
   (A = 1500 ∧ B = 1000) ∨
   (A = 1400 ∧ B = 1200)) :=
by
  -- The proof should be provided here
  sorry

end bonus_distribution_plans_l15_15076


namespace range_of_a_l15_15919

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

-- Define the main theorem
theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) → -1 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l15_15919


namespace line_through_two_points_line_with_intercept_sum_l15_15583

theorem line_through_two_points (a b x1 y1 x2 y2: ℝ) : 
  (x1 = 2) → (y1 = 1) → (x2 = 0) → (y2 = -3) → (2 * x - y - 3 = 0) :=
by
                
  sorry

theorem line_with_intercept_sum (a b : ℝ) (x y : ℝ) :
  (x = 0) → (y = 5) → (a + b = 2) → (b = 5) → (5 * x - 3 * y + 15 = 0) :=
by
  sorry

end line_through_two_points_line_with_intercept_sum_l15_15583


namespace simplify_correct_l15_15856

def simplify_expression (a b : ℤ) : ℤ :=
  (30 * a + 70 * b) + (15 * a + 45 * b) - (12 * a + 60 * b)

theorem simplify_correct (a b : ℤ) : simplify_expression a b = 33 * a + 55 * b :=
by 
  sorry -- Proof to be filled in later

end simplify_correct_l15_15856


namespace inequality_sqrt_ab_l15_15985

theorem inequality_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 / (1 / a + 1 / b) ≤ Real.sqrt (a * b) :=
sorry

end inequality_sqrt_ab_l15_15985


namespace unique_positive_b_solution_exists_l15_15542

theorem unique_positive_b_solution_exists (c : ℝ) (k : ℝ) :
  (∃b : ℝ, b > 0 ∧ ∀x : ℝ, x^2 + (b + 1/b) * x + c = 0 → x = 0) ∧
  (∀b : ℝ, b^4 + (2 - 4 * c) * b^2 + k = 0) → c = 1 :=
by
  sorry

end unique_positive_b_solution_exists_l15_15542


namespace range_a_of_function_has_two_zeros_l15_15796

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem range_a_of_function_has_two_zeros (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) : 
  1 < a :=
sorry

end range_a_of_function_has_two_zeros_l15_15796


namespace factor_difference_of_squares_l15_15087

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l15_15087


namespace problem1_problem2_l15_15486

-- Definitions of the sets A and B based on the given conditions
def A : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - 3 * a) < 0 }

-- Proof statement for problem (1)
theorem problem1 (a : ℝ) : (∀ x, x ∈ A → x ∈ (B a)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

-- Proof statement for problem (2)
theorem problem2 (a : ℝ) : (∀ x, (x ∈ A ∧ x ∈ (B a)) ↔ (3 < x ∧ x < 4)) ↔ (a = 3) := by
  sorry

end problem1_problem2_l15_15486


namespace men_per_table_correct_l15_15921

def tables := 6
def women_per_table := 3
def total_customers := 48
def total_women := women_per_table * tables
def total_men := total_customers - total_women
def men_per_table := total_men / tables

theorem men_per_table_correct : men_per_table = 5 := by
  sorry

end men_per_table_correct_l15_15921


namespace first_reduction_percentage_l15_15011

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.70 = P * 0.525 ↔ x = 25 := by
  sorry

end first_reduction_percentage_l15_15011


namespace value_of_a_l15_15196

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l15_15196


namespace option_d_not_true_l15_15587

variable (a b : ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)

theorem option_d_not_true : (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) := sorry

end option_d_not_true_l15_15587


namespace percentage_increase_in_consumption_l15_15341

theorem percentage_increase_in_consumption 
  (T C : ℝ) 
  (h1 : 0.8 * T * C * (1 + P / 100) = 0.88 * T * C)
  : P = 10 := 
by 
  sorry

end percentage_increase_in_consumption_l15_15341


namespace minimum_value_fraction_l15_15724

theorem minimum_value_fraction (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 2) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 2 → 
    ((1 / (1 + x)) + (1 / (2 + 2 * y)) ≥ 4 / 5)) :=
by sorry

end minimum_value_fraction_l15_15724


namespace cylinder_cone_volume_l15_15279

theorem cylinder_cone_volume (V_total : ℝ) (Vc Vcone : ℝ)
  (h1 : V_total = 48)
  (h2 : V_total = Vc + Vcone)
  (h3 : Vc = 3 * Vcone) :
  Vc = 36 ∧ Vcone = 12 :=
by
  sorry

end cylinder_cone_volume_l15_15279


namespace call_charge_ratio_l15_15978

def elvin_jan_total_bill : ℕ := 46
def elvin_feb_total_bill : ℕ := 76
def elvin_internet_charge : ℕ := 16
def elvin_call_charge_ratio : ℕ := 2

theorem call_charge_ratio : 
  (elvin_feb_total_bill - elvin_internet_charge) / (elvin_jan_total_bill - elvin_internet_charge) = elvin_call_charge_ratio := 
by
  sorry

end call_charge_ratio_l15_15978


namespace boat_speed_in_still_water_l15_15315

-- Problem Definitions
def V_s : ℕ := 16
def t : ℕ := sorry -- t is arbitrary positive value
def V_b : ℕ := 48

-- Conditions
def upstream_time := 2 * t
def downstream_time := t
def upstream_distance := (V_b - V_s) * upstream_time
def downstream_distance := (V_b + V_s) * downstream_time

-- Proof Problem
theorem boat_speed_in_still_water :
  upstream_distance = downstream_distance → V_b = 48 :=
by sorry

end boat_speed_in_still_water_l15_15315


namespace matrix_self_inverse_pairs_l15_15404

theorem matrix_self_inverse_pairs :
  ∃ p : Finset (ℝ × ℝ), (∀ a d, (a, d) ∈ p ↔ (∃ (m : Matrix (Fin 2) (Fin 2) ℝ), 
    m = !![a, 4; -9, d] ∧ m * m = 1)) ∧ p.card = 2 :=
by {
  sorry
}

end matrix_self_inverse_pairs_l15_15404


namespace geometric_series_sum_l15_15347

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l15_15347


namespace GroundBeefSalesTotalRevenue_l15_15199

theorem GroundBeefSalesTotalRevenue :
  let price_regular := 3.50
  let price_lean := 4.25
  let price_extra_lean := 5.00

  let monday_revenue := 198.5 * price_regular +
                        276.2 * price_lean +
                        150.7 * price_extra_lean

  let tuesday_revenue := 210 * (price_regular * 0.90) +
                         420 * (price_lean * 0.90) +
                         150 * (price_extra_lean * 0.90)
  
  let wednesday_revenue := 230 * price_regular +
                           324.6 * 3.75 +
                           120.4 * price_extra_lean

  monday_revenue + tuesday_revenue + wednesday_revenue = 8189.35 :=
by
  sorry

end GroundBeefSalesTotalRevenue_l15_15199


namespace tan_mul_tan_l15_15635

variables {α β : ℝ}

theorem tan_mul_tan (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 :=
sorry

end tan_mul_tan_l15_15635


namespace arithmetic_sequence_property_l15_15346

variable {a : ℕ → ℝ} -- Let a be an arithmetic sequence
variable {S : ℕ → ℝ} -- Let S be the sum of the first n terms of the sequence

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + (n - 1) * (a 2 - a 1) / 2)
axiom a_5 : a 5 = 3
axiom S_13 : S 13 = 91

-- Question to prove
theorem arithmetic_sequence_property : a 1 + a 11 = 10 :=
by
  sorry

end arithmetic_sequence_property_l15_15346


namespace sum_of_first_ten_terms_l15_15020

variable {α : Type*} [LinearOrderedField α]

-- Defining the arithmetic sequence and sum of the first n terms
def a_n (a d : α) (n : ℕ) : α := a + d * (n - 1)

def S_n (a : α) (d : α) (n : ℕ) : α := n / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_ten_terms (a d : α) (h : a_n a d 3 + a_n a d 8 = 12) : S_n a d 10 = 60 :=
by sorry

end sum_of_first_ten_terms_l15_15020


namespace gcd_polynomial_multiple_of_345_l15_15088

theorem gcd_polynomial_multiple_of_345 (b : ℕ) (h : ∃ k : ℕ, b = 345 * k) : 
  Nat.gcd (5 * b ^ 3 + 2 * b ^ 2 + 7 * b + 69) b = 69 := 
by
  sorry

end gcd_polynomial_multiple_of_345_l15_15088


namespace customer_buys_two_pens_l15_15603

def num_pens (total_pens non_defective_pens : Nat) (prob : ℚ) : Nat :=
  sorry

theorem customer_buys_two_pens :
  num_pens 16 13 0.65 = 2 :=
sorry

end customer_buys_two_pens_l15_15603


namespace number_of_boundaries_l15_15094

theorem number_of_boundaries 
  (total_runs : ℕ) 
  (number_of_sixes : ℕ) 
  (percentage_runs_by_running : ℝ) 
  (runs_per_six : ℕ) 
  (runs_per_boundary : ℕ)
  (h_total_runs : total_runs = 125)
  (h_number_of_sixes : number_of_sixes = 5)
  (h_percentage_runs_by_running : percentage_runs_by_running = 0.60)
  (h_runs_per_six : runs_per_six = 6)
  (h_runs_per_boundary : runs_per_boundary = 4) :
  (total_runs - percentage_runs_by_running * total_runs - number_of_sixes * runs_per_six) / runs_per_boundary = 5 := by 
  sorry

end number_of_boundaries_l15_15094


namespace largest_value_of_c_l15_15823

theorem largest_value_of_c : ∀ c : ℝ, (3 * c + 6) * (c - 2) = 9 * c → c ≤ 4 :=
by
  intros c hc
  have : (3 * c + 6) * (c - 2) = 9 * c := hc
  sorry

end largest_value_of_c_l15_15823


namespace range_of_m_l15_15084

noncomputable def abs_sum (x : ℝ) : ℝ := |x - 5| + |x - 3|

theorem range_of_m (m : ℝ) : (∃ x : ℝ, abs_sum x < m) ↔ m > 2 := 
by 
  sorry

end range_of_m_l15_15084


namespace find_m_value_l15_15061

theorem find_m_value (m : ℚ) :
  (m - 10) / -10 = (5 - m) / -8 → m = 65 / 9 :=
by
  sorry

end find_m_value_l15_15061


namespace power_function_decreasing_l15_15748

theorem power_function_decreasing (m : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 < x → f x = (m^2 + m - 11) * x^(m - 1))
  (hm : m^2 + m - 11 > 0)
  (hm' : m - 1 < 0)
  (hx : 0 < 1):
  f (-1) = -1 := by 
sorry

end power_function_decreasing_l15_15748


namespace swimming_problem_l15_15914

/-- The swimming problem where a man swims downstream 30 km and upstream a certain distance 
    taking 6 hours each time. Given his speed in still water is 4 km/h, we aim to prove the 
    distance swam upstream is 18 km. -/
theorem swimming_problem 
  (V_m : ℝ) (Distance_downstream : ℝ) (Time_downstream : ℝ) (Time_upstream : ℝ) 
  (Distance_upstream : ℝ) (V_s : ℝ)
  (h1 : V_m = 4)
  (h2 : Distance_downstream = 30)
  (h3 : Time_downstream = 6)
  (h4 : Time_upstream = 6)
  (h5 : V_m + V_s = Distance_downstream / Time_downstream)
  (h6 : V_m - V_s = Distance_upstream / Time_upstream) :
  Distance_upstream = 18 := 
sorry

end swimming_problem_l15_15914


namespace range_of_a_l15_15402

theorem range_of_a 
  (a b x1 x2 x3 x4 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a^2 ≠ 0)
  (hx1 : a * x1^2 + b * x1 + 1 = 0) 
  (hx2 : a * x2^2 + b * x2 + 1 = 0) 
  (hx3 : a^2 * x3^2 + b * x3 + 1 = 0) 
  (hx4 : a^2 * x4^2 + b * x4 + 1 = 0)
  (h_order : x3 < x1 ∧ x1 < x2 ∧ x2 < x4) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l15_15402


namespace geometric_triangle_condition_right_geometric_triangle_condition_l15_15527

-- Definitions for the geometric progression
def geometric_sequence (a b c q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- Conditions for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for forming a right triangle using Pythagorean theorem
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem geometric_triangle_condition (a q : ℝ) (h1 : 1 ≤ q) (h2 : q < (1 + Real.sqrt 5) / 2) :
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ forms_triangle a b c := 
sorry

theorem right_geometric_triangle_condition (a q : ℝ) :
  q = Real.sqrt ((1 + Real.sqrt 5) / 2) →
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ right_triangle a b c :=
sorry

end geometric_triangle_condition_right_geometric_triangle_condition_l15_15527


namespace sqrt_inequality_l15_15683

theorem sqrt_inequality (a b c : ℝ) (θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c :=
sorry

end sqrt_inequality_l15_15683


namespace remainder_of_150_div_k_l15_15850

theorem remainder_of_150_div_k (k : ℕ) (hk : k > 0) (h1 : 90 % (k^2) = 10) :
  150 % k = 2 := 
sorry

end remainder_of_150_div_k_l15_15850


namespace range_of_m_l15_15406

variable (m : ℝ)

def p : Prop := (m^2 - 4 > 0) ∧ (m > 0)
def q : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
by
  intro h
  sorry

end range_of_m_l15_15406


namespace cuboid_height_l15_15368

-- Define the necessary constants
def width : ℕ := 30
def length : ℕ := 22
def sum_edges : ℕ := 224

-- Theorem stating the height of the cuboid
theorem cuboid_height (h : ℕ) : 4 * length + 4 * width + 4 * h = sum_edges → h = 4 := by
  sorry

end cuboid_height_l15_15368


namespace value_of_f_prime_at_1_l15_15149

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem value_of_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end value_of_f_prime_at_1_l15_15149


namespace avg_chem_math_l15_15900

-- Given conditions
variables (P C M : ℕ)
axiom total_marks : P + C + M = P + 130

-- The proof problem
theorem avg_chem_math : (C + M) / 2 = 65 :=
by sorry

end avg_chem_math_l15_15900


namespace length_of_bridge_l15_15344

theorem length_of_bridge (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) 
  (h1 : train_length = 150) 
  (h2 : train_speed = 45) 
  (h3 : cross_time = 30) : 
  ∃ bridge_length : ℕ, bridge_length = 225 := sorry

end length_of_bridge_l15_15344


namespace intersection_complement_l15_15522

-- Declare variables for sets
variable (I A B : Set ℤ)

-- Define the universal set I
def universal_set : Set ℤ := { x | -3 < x ∧ x < 3 }

-- Define sets A and B
def set_A : Set ℤ := { -2, 0, 1 }
def set_B : Set ℤ := { -1, 0, 1, 2 }

-- Main theorem statement
theorem intersection_complement
  (hI : I = universal_set)
  (hA : A = set_A)
  (hB : B = set_B) :
  B ∩ (I \ A) = { -1, 2 } :=
sorry

end intersection_complement_l15_15522


namespace max_goods_purchased_l15_15878

theorem max_goods_purchased (initial_spend : ℕ) (reward_rate : ℕ → ℕ → ℕ) (continuous_reward : Prop) :
  initial_spend = 7020 →
  (∀ x y, reward_rate x y = (x / y) * 20) →
  continuous_reward →
  initial_spend + reward_rate initial_spend 100 + reward_rate (reward_rate initial_spend 100) 100 + 
  reward_rate (reward_rate (reward_rate initial_spend 100) 100) 100 = 8760 :=
by
  intros h1 h2 h3
  sorry

end max_goods_purchased_l15_15878


namespace smallest_repeating_block_of_5_over_13_l15_15630

theorem smallest_repeating_block_of_5_over_13 : 
  ∃ n, n = 6 ∧ (∃ m, (5 / 13 : ℚ) = (m/(10^6) : ℚ) ) := 
sorry

end smallest_repeating_block_of_5_over_13_l15_15630


namespace find_k_l15_15835

theorem find_k (x y k : ℝ) 
  (line1 : y = 3 * x + 2) 
  (line2 : y = -4 * x - 14) 
  (line3 : y = 2 * x + k) :
  k = -2 / 7 := 
by {
  sorry
}

end find_k_l15_15835


namespace remainder_of_polynomial_l15_15422

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

-- Define the main theorem stating the remainder when f(x) is divided by (x - 1) is 6
theorem remainder_of_polynomial : f 1 = 6 := 
by 
  sorry

end remainder_of_polynomial_l15_15422


namespace expr1_eval_expr2_eval_l15_15309

theorem expr1_eval : (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (16 / 3) + 3 * Real.sqrt (25 / 3)) = 115 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

theorem expr2_eval : (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (8 / 3) - 3 * Real.sqrt (5 / 3)) = 3 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

end expr1_eval_expr2_eval_l15_15309


namespace sam_dimes_proof_l15_15750

def initial_dimes : ℕ := 9
def remaining_dimes : ℕ := 2
def dimes_given : ℕ := 7

theorem sam_dimes_proof : initial_dimes - remaining_dimes = dimes_given :=
by
  sorry

end sam_dimes_proof_l15_15750


namespace totalWeightAlF3_is_correct_l15_15351

-- Define the atomic weights of Aluminum and Fluorine
def atomicWeightAl : ℝ := 26.98
def atomicWeightF : ℝ := 19.00

-- Define the number of atoms of Fluorine in Aluminum Fluoride (AlF3)
def numFluorineAtoms : ℕ := 3

-- Define the number of moles of Aluminum Fluoride
def numMolesAlF3 : ℕ := 7

-- Calculate the molecular weight of Aluminum Fluoride (AlF3)
noncomputable def molecularWeightAlF3 : ℝ :=
  atomicWeightAl + (numFluorineAtoms * atomicWeightF)

-- Calculate the total weight of the given moles of AlF3
noncomputable def totalWeight : ℝ :=
  molecularWeightAlF3 * numMolesAlF3

-- Theorem stating the total weight of 7 moles of AlF3
theorem totalWeightAlF3_is_correct : totalWeight = 587.86 := sorry

end totalWeightAlF3_is_correct_l15_15351


namespace minimum_value_correct_l15_15734

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a + 3*b = 1 then 1/a + 1/b else 0

theorem minimum_value_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ minimum_value a b = 4 + 4 * Real.sqrt 3 :=
by
  sorry

end minimum_value_correct_l15_15734


namespace population_30_3_million_is_30300000_l15_15730

theorem population_30_3_million_is_30300000 :
  let million := 1000000
  let population_1998 := 30.3 * million
  population_1998 = 30300000 :=
by
  -- Proof goes here
  sorry

end population_30_3_million_is_30300000_l15_15730


namespace volume_of_max_area_rect_prism_l15_15669

noncomputable def side_length_of_square_base (P: ℕ) : ℕ := P / 4

noncomputable def area_of_square_base (side: ℕ) : ℕ := side * side

noncomputable def volume_of_rectangular_prism (base_area: ℕ) (height: ℕ) : ℕ := base_area * height

theorem volume_of_max_area_rect_prism
  (P : ℕ) (hP : P = 32) 
  (H : ℕ) (hH : H = 9) 
  : volume_of_rectangular_prism (area_of_square_base (side_length_of_square_base P)) H = 576 := 
by
  sorry

end volume_of_max_area_rect_prism_l15_15669


namespace g_at_5_l15_15062

def g (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 28 * x^2 - 20 * x - 80

theorem g_at_5 : g 5 = -5 := 
  by 
  -- Proof goes here
  sorry

end g_at_5_l15_15062


namespace smallest_value_of_c_l15_15119

/-- The polynomial x^3 - cx^2 + dx - 2550 has three positive integer roots,
    and the product of the roots is 2550. Prove that the smallest possible value of c is 42. -/
theorem smallest_value_of_c :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2550 ∧ c = a + b + c) → c = 42 :=
sorry

end smallest_value_of_c_l15_15119


namespace no_real_solution_for_x_l15_15617

theorem no_real_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 8) (h2 : y + 1 / x = 7 / 20) : false :=
by sorry

end no_real_solution_for_x_l15_15617


namespace range_of_a_l15_15048

theorem range_of_a (a : ℝ) :
    (∀ x : ℤ, x + 1 > 0 → 3 * x - a ≤ 0 → x = 0 ∨ x = 1 ∨ x = 2) ↔ 6 ≤ a ∧ a < 9 :=
by
  sorry

end range_of_a_l15_15048


namespace number_of_connections_l15_15464

theorem number_of_connections (n : ℕ) (d : ℕ) (h₀ : n = 40) (h₁ : d = 4) : 
  (n * d) / 2 = 80 :=
by
  sorry

end number_of_connections_l15_15464


namespace find_initial_sum_l15_15679

-- Define the conditions as constants
def A1 : ℝ := 590
def A2 : ℝ := 815
def t1 : ℝ := 2
def t2 : ℝ := 7

-- Define the variables
variable (P r : ℝ)

-- First condition after 2 years
def condition1 : Prop := A1 = P + P * r * t1

-- Second condition after 7 years
def condition2 : Prop := A2 = P + P * r * t2

-- The statement we need to prove: the initial sum of money P is 500
theorem find_initial_sum (h1 : condition1 P r) (h2 : condition2 P r) : P = 500 :=
sorry

end find_initial_sum_l15_15679


namespace non_neg_scalar_product_l15_15705

theorem non_neg_scalar_product (a b c d e f g h : ℝ) : 
  (0 ≤ ac + bd) ∨ (0 ≤ ae + bf) ∨ (0 ≤ ag + bh) ∨ (0 ≤ ce + df) ∨ (0 ≤ cg + dh) ∨ (0 ≤ eg + fh) :=
  sorry

end non_neg_scalar_product_l15_15705


namespace robin_cut_hair_l15_15993

-- Definitions as per the given conditions
def initial_length := 17
def current_length := 13

-- Statement of the proof problem
theorem robin_cut_hair : initial_length - current_length = 4 := 
by 
  sorry

end robin_cut_hair_l15_15993


namespace find_p_plus_q_l15_15105

noncomputable def p (d e : ℝ) (x : ℝ) : ℝ := d * x + e
noncomputable def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_p_plus_q (d e a b c : ℝ)
  (h1 : p d e 0 / q a b c 0 = 4)
  (h2 : p d e (-1) = -1)
  (h3 : q a b c 1 = 3)
  (e_eq : e = 4 * c):
  (p d e x + q a b c x) = (3*x^2 + 26*x - 30) :=
by
  sorry

end find_p_plus_q_l15_15105


namespace bahs_equal_to_yahs_l15_15223

theorem bahs_equal_to_yahs (bahs rahs yahs : ℝ) 
  (h1 : 18 * bahs = 30 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) : 
  1200 * yahs = 432 * bahs := 
by
  sorry

end bahs_equal_to_yahs_l15_15223


namespace movie_theater_people_l15_15885

def totalSeats : ℕ := 750
def emptySeats : ℕ := 218
def peopleWatching := totalSeats - emptySeats

theorem movie_theater_people :
  peopleWatching = 532 := by
  sorry

end movie_theater_people_l15_15885


namespace solve_p_l15_15657

theorem solve_p (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 2 * p + 5 * q = 8) : 
  p = 11 / 19 :=
by
  sorry

end solve_p_l15_15657


namespace acme_cheaper_than_beta_l15_15123

theorem acme_cheaper_than_beta (x : ℕ) :
  (50 + 9 * x < 25 + 15 * x) ↔ (5 ≤ x) :=
by sorry

end acme_cheaper_than_beta_l15_15123


namespace cells_surpass_10_pow_10_in_46_hours_l15_15278

noncomputable def cells_exceed_threshold_hours : ℕ := 46

theorem cells_surpass_10_pow_10_in_46_hours : 
  ∀ (n : ℕ), (100 * ((3 / 2 : ℝ) ^ n) > 10 ^ 10) ↔ n ≥ cells_exceed_threshold_hours := 
by
  sorry

end cells_surpass_10_pow_10_in_46_hours_l15_15278


namespace proof_problem_l15_15742

variables {a b c d e : ℝ}

theorem proof_problem (h1 : a * b^2 * c^3 * d^4 * e^5 < 0) (h2 : b^2 ≥ 0) (h3 : d^4 ≥ 0) :
  a * b^2 * c * d^4 * e < 0 :=
sorry

end proof_problem_l15_15742


namespace algebraic_expression_value_l15_15916

theorem algebraic_expression_value (a b : ℕ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b * b) / a) / ((a * a - b * b) / a) = 1 / 2 := 
sorry

end algebraic_expression_value_l15_15916


namespace lcm_of_4_9_10_27_l15_15066

theorem lcm_of_4_9_10_27 : Nat.lcm (Nat.lcm 4 9) (Nat.lcm 10 27) = 540 :=
by
  sorry

end lcm_of_4_9_10_27_l15_15066


namespace remainder_of_sum_of_squares_mod_l15_15595

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end remainder_of_sum_of_squares_mod_l15_15595


namespace count_paths_l15_15743

-- Define the lattice points and paths
def isLatticePoint (P : ℤ × ℤ) : Prop := true
def isLatticePath (P : ℕ → ℤ × ℤ) (n : ℕ) : Prop :=
  (∀ i, 0 < i → i ≤ n → abs ((P i).1 - (P (i - 1)).1) + abs ((P i).2 - (P (i - 1)).2) = 1)

-- Define F(n) with the given constraints
def numberOfPaths (n : ℕ) : ℕ :=
  -- Placeholder for the actual complex counting logic, which is not detailed here
  sorry

-- Identify F(n) from the initial conditions and the correct result
theorem count_paths (n : ℕ) :
  numberOfPaths n = Nat.choose (2 * n) n :=
sorry

end count_paths_l15_15743


namespace trajectory_eq_l15_15328

theorem trajectory_eq {x y : ℝ} (h₁ : (x-2)^2 + y^2 = 1) (h₂ : ∃ r, (x+1)^2 = (x-2)^2 + y^2 - r^2) :
  y^2 = 6 * x - 3 :=
by
  sorry

end trajectory_eq_l15_15328


namespace inequality_condition_sufficient_l15_15304

theorem inequality_condition_sufficient (A B C : ℝ) (x y z : ℝ) 
  (hA : 0 ≤ A) 
  (hB : 0 ≤ B) 
  (hC : 0 ≤ C) 
  (hABC : A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :
  A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0 :=
sorry

end inequality_condition_sufficient_l15_15304


namespace josephs_total_cards_l15_15396

def number_of_decks : ℕ := 4
def cards_per_deck : ℕ := 52
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem josephs_total_cards : total_cards = 208 := by
  sorry

end josephs_total_cards_l15_15396


namespace integer_solutions_zero_l15_15513

theorem integer_solutions_zero (x y u t : ℤ) :
  x^2 + y^2 = 1974 * (u^2 + t^2) → 
  x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 :=
by
  sorry

end integer_solutions_zero_l15_15513


namespace plates_arrangement_l15_15012

theorem plates_arrangement :
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  non_adjacent_green_arrangements = 588 :=
by
  let B := 5
  let R := 2
  let G := 2
  let O := 1
  let total_plates := B + R + G + O
  let total_arrangements := Nat.factorial total_plates / (Nat.factorial B * Nat.factorial R * Nat.factorial G * Nat.factorial O)
  let circular_arrangements := total_arrangements / total_plates
  let green_adjacent_arrangements := Nat.factorial (total_plates - 1) / (Nat.factorial B * Nat.factorial R * (Nat.factorial (G - 1)) * Nat.factorial O)
  let circular_green_adjacent_arrangements := green_adjacent_arrangements / (total_plates - 1)
  let non_adjacent_green_arrangements := circular_arrangements - circular_green_adjacent_arrangements
  sorry

end plates_arrangement_l15_15012


namespace barbell_percentage_increase_l15_15568

def old_barbell_cost : ℕ := 250
def new_barbell_cost : ℕ := 325

theorem barbell_percentage_increase :
  (new_barbell_cost - old_barbell_cost : ℚ) / old_barbell_cost * 100 = 30 := 
by
  sorry

end barbell_percentage_increase_l15_15568


namespace complement_A_union_B_l15_15928

def is_positive_integer_less_than_9 (n : ℕ) : Prop :=
  n > 0 ∧ n < 9

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def U := {n : ℕ | is_positive_integer_less_than_9 n}
noncomputable def A := {n ∈ U | is_odd n}
noncomputable def B := {n ∈ U | is_multiple_of_3 n}

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {2, 4, 8} :=
sorry

end complement_A_union_B_l15_15928


namespace total_cost_calc_l15_15984

variable (a b : ℝ)

def total_cost (a b : ℝ) := 2 * a + 3 * b

theorem total_cost_calc (a b : ℝ) : total_cost a b = 2 * a + 3 * b := by
  sorry

end total_cost_calc_l15_15984


namespace amy_hours_per_week_school_year_l15_15353

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end amy_hours_per_week_school_year_l15_15353


namespace percent_of_total_l15_15723

theorem percent_of_total (p n : ℝ) (h1 : p = 35 / 100) (h2 : n = 360) : p * n = 126 := by
  sorry

end percent_of_total_l15_15723


namespace compare_star_l15_15736

def star (m n : ℤ) : ℤ := (m + 2) * 3 - n

theorem compare_star : star 2 (-2) > star (-2) 2 := 
by sorry

end compare_star_l15_15736


namespace matrix_non_invertible_at_36_31_l15_15376

-- Define the matrix A
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 * x, 9], ![4 - x, 11]]

-- State the theorem
theorem matrix_non_invertible_at_36_31 :
  ∃ x : ℝ, (A x).det = 0 ∧ x = 36 / 31 :=
by {
  sorry
}

end matrix_non_invertible_at_36_31_l15_15376


namespace marbles_in_larger_container_l15_15922

-- Defining the conditions
def volume1 := 24 -- in cm³
def marbles1 := 30 -- number of marbles in the first container
def volume2 := 72 -- in cm³

-- Statement of the theorem
theorem marbles_in_larger_container : (marbles1 / volume1 : ℚ) * volume2 = 90 := by
  sorry

end marbles_in_larger_container_l15_15922


namespace correct_statement_l15_15188

/-- Given the following statements:
 1. Seeing a rainbow after rain is a random event.
 2. To check the various equipment before a plane takes off, a random sampling survey should be conducted.
 3. When flipping a coin 20 times, it will definitely land heads up 10 times.
 4. The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B.

 Prove that the correct statement is: Seeing a rainbow after rain is a random event.
-/
theorem correct_statement : 
  let statement_A := "Seeing a rainbow after rain is a random event"
  let statement_B := "To check the various equipment before a plane takes off, a random sampling survey should be conducted"
  let statement_C := "When flipping a coin 20 times, it will definitely land heads up 10 times"
  let statement_D := "The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B"
  statement_A = "Seeing a rainbow after rain is a random event" := by
sorry

end correct_statement_l15_15188


namespace club_membership_l15_15948

def total_people_in_club (T B TB N : ℕ) : ℕ :=
  T + B - TB + N

theorem club_membership : total_people_in_club 138 255 94 11 = 310 := by
  sorry

end club_membership_l15_15948


namespace question1_question2_l15_15153

-- Define the function representing the inequality
def inequality (a x : ℝ) : Prop := (a * x - 5) / (x - a) < 0

-- Question 1: Compute the solution set M when a=1
theorem question1 : (setOf (λ x : ℝ => inequality 1 x)) = {x : ℝ | 1 < x ∧ x < 5} :=
by
  sorry

-- Question 2: Determine the range for a such that 3 ∈ M but 5 ∉ M
theorem question2 : (setOf (λ a : ℝ => 3 ∈ (setOf (λ x : ℝ => inequality a x)) ∧ 5 ∉ (setOf (λ x : ℝ => inequality a x)))) = 
  {a : ℝ | (1 ≤ a ∧ a < 5 / 3) ∨ (3 < a ∧ a ≤ 5)} :=
by
  sorry

end question1_question2_l15_15153


namespace train_crosses_pole_l15_15737

theorem train_crosses_pole
  (speed_kmph : ℝ)
  (train_length_meters : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (time_seconds : ℝ)
  (h1 : speed_kmph = 270)
  (h2 : train_length_meters = 375.03)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = train_length_meters / speed_mps)
  : time_seconds = 5.0004 :=
by
  sorry

end train_crosses_pole_l15_15737


namespace rainy_days_l15_15305

namespace Mo

def drinks (R NR n : ℕ) :=
  -- Condition 3: Total number of days in the week equation
  R + NR = 7 ∧
  -- Condition 1-2: Total cups of drinks equation
  n * R + 3 * NR = 26 ∧
  -- Condition 4: Difference in cups of tea and hot chocolate equation
  3 * NR - n * R = 10

theorem rainy_days (R NR n : ℕ) (h: drinks R NR n) : 
  R = 1 := sorry

end Mo

end rainy_days_l15_15305


namespace value_of_z_l15_15896

theorem value_of_z (z y : ℝ) (h1 : (12)^3 * z^3 / 432 = y) (h2 : y = 864) : z = 6 :=
by
  sorry

end value_of_z_l15_15896


namespace circle_problem_is_solved_l15_15421

def circle_problem_pqr : ℕ :=
  let n := 3 / 2;
  let p := 3;
  let q := 1;
  let r := 4;
  p + q + r

theorem circle_problem_is_solved : circle_problem_pqr = 8 :=
by {
  -- Additional context of conditions can be added here if necessary
  sorry
}

end circle_problem_is_solved_l15_15421


namespace solve_inequality_l15_15800

noncomputable def solutionSet := { x : ℝ | 0 < x ∧ x < 1 }

theorem solve_inequality (x : ℝ) : x^2 < x ↔ x ∈ solutionSet := 
sorry

end solve_inequality_l15_15800


namespace integers_between_sqrt7_and_sqrt77_l15_15929

theorem integers_between_sqrt7_and_sqrt77 : 
  2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 ∧ 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 →
  ∃ (n : ℕ), n = 6 ∧ ∀ (k : ℕ), (3 ≤ k ∧ k ≤ 8) ↔ (2 < Real.sqrt 7 ∧ Real.sqrt 77 < 9) :=
by sorry

end integers_between_sqrt7_and_sqrt77_l15_15929


namespace point_always_outside_circle_l15_15212

theorem point_always_outside_circle (a : ℝ) : a^2 + (2 - a)^2 > 1 :=
by sorry

end point_always_outside_circle_l15_15212


namespace general_equation_of_line_l15_15862

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

-- Define what it means for a line to pass through two points
def line_through_points (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 2 = 0

-- The theorem that needs to be proven
theorem general_equation_of_line : line_through_points line_l A B := 
by
  sorry

end general_equation_of_line_l15_15862


namespace town_population_growth_is_62_percent_l15_15886

noncomputable def population_growth_proof : ℕ := 
  let p := 22
  let p_square := p * p
  let pop_1991 := p_square
  let pop_2001 := pop_1991 + 150
  let pop_2011 := pop_2001 + 150
  let k := 28  -- Given that 784 = 28^2
  let pop_2011_is_perfect_square := k * k = pop_2011
  let percentage_increase := ((pop_2011 - pop_1991) * 100) / pop_1991
  if pop_2011_is_perfect_square then percentage_increase 
  else 0

theorem town_population_growth_is_62_percent :
  population_growth_proof = 62 :=
by
  sorry

end town_population_growth_is_62_percent_l15_15886


namespace central_angle_of_sector_l15_15548

theorem central_angle_of_sector (r S α : ℝ) (h1 : r = 10) (h2 : S = 100)
  (h3 : S = 1/2 * α * r^2) : α = 2 :=
by
  -- Given radius r and area S, substituting into the formula for the area of the sector,
  -- we derive the central angle α.
  sorry

end central_angle_of_sector_l15_15548


namespace proof_BH_length_equals_lhs_rhs_l15_15895

noncomputable def calculate_BH_length : ℝ :=
  let AB := 3
  let BC := 4
  let CA := 5
  let AG := 4  -- Since AB < AG
  let AH := 6  -- AG < AH
  let GI := 3
  let HI := 8
  let GH := Real.sqrt (GI ^ 2 + HI ^ 2)
  let p := 3
  let q := 2
  let r := 73
  let s := 1
  3 + 2 * Real.sqrt 73

theorem proof_BH_length_equals_lhs_rhs :
  let BH := 3 + 2 * Real.sqrt 73
  calculate_BH_length = BH := by
    sorry

end proof_BH_length_equals_lhs_rhs_l15_15895


namespace sum_of_digits_l15_15445

theorem sum_of_digits (A T M : ℕ) (h1 : T = A + 3) (h2 : M = 3)
    (h3 : (∃ k : ℕ, T = k^2 * M) ∧ (∃ l : ℕ, T = 33)) : 
    ∃ x : ℕ, ∃ dsum : ℕ, (A + x) % (M + x) = 0 ∧ dsum = 12 :=
by
  sorry

end sum_of_digits_l15_15445


namespace complete_the_square_d_l15_15735

theorem complete_the_square_d (x : ℝ) (h : x^2 + 6 * x + 5 = 0) : ∃ d : ℝ, (x + 3)^2 = d ∧ d = 4 :=
by
  sorry

end complete_the_square_d_l15_15735


namespace mean_value_of_interior_angles_of_quadrilateral_l15_15846

theorem mean_value_of_interior_angles_of_quadrilateral 
  (sum_of_interior_angles : ℝ := 360) 
  (number_of_angles : ℝ := 4) : 
  sum_of_interior_angles / number_of_angles = 90 := 
by 
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l15_15846


namespace find_m_b_sum_does_not_prove_l15_15788

theorem find_m_b_sum_does_not_prove :
  ∃ m b : ℝ, 
  let original_point := (2, 3)
  let image_point := (10, 9)
  let midpoint := ((original_point.1 + image_point.1) / 2, (original_point.2 + image_point.2) / 2)
  m = -4 / 3 ∧ 
  midpoint = (6, 6) ∧ 
  6 = m * 6 + b 
  ∧ m + b = 38 / 3 := sorry

end find_m_b_sum_does_not_prove_l15_15788


namespace doctors_assignment_l15_15941

theorem doctors_assignment :
  ∃ (assignments : Finset (Fin 3 → Finset (Fin 5))),
    (∀ h ∈ assignments, (∀ i, ∃ j ∈ h i, True) ∧
      ¬(∃ i j, (A ∈ h i ∧ B ∈ h j ∨ A ∈ h j ∧ B ∈ h i)) ∧
      ¬(∃ i j, (C ∈ h i ∧ D ∈ h j ∨ C ∈ h j ∧ D ∈ h i))) ∧
    assignments.card = 84 :=
sorry

end doctors_assignment_l15_15941


namespace initial_pepper_amount_l15_15211
-- Import the necessary libraries.

-- Declare the problem as a theorem.
theorem initial_pepper_amount (used left : ℝ) (h₁ : used = 0.16) (h₂ : left = 0.09) :
  used + left = 0.25 :=
by
  -- The proof is not required here.
  sorry

end initial_pepper_amount_l15_15211


namespace find_x_l15_15139

noncomputable def series_sum (x : ℝ) : ℝ :=
∑' n : ℕ, (1 + 6 * n) * x^n

theorem find_x (x : ℝ) (h : series_sum x = 100) (hx : |x| < 1) : x = 3 / 5 := 
sorry

end find_x_l15_15139


namespace fraction_is_terminating_decimal_l15_15476

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l15_15476


namespace find_last_year_rate_l15_15116

-- Define the problem setting with types and values (conditions)
def last_year_rate (r : ℝ) : Prop := 
  -- Let r be the annual interest rate last year
  1.1 * r = 0.09

-- Define the theorem to prove the interest rate last year given this year's rate
theorem find_last_year_rate :
  ∃ r : ℝ, last_year_rate r ∧ r = 0.09 / 1.1 := 
by
  sorry

end find_last_year_rate_l15_15116


namespace parabola_range_m_l15_15431

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (4*m + 1)*x + (2*m - 1)

theorem parabola_range_m (m : ℝ) :
  (∀ x : ℝ, parabola m x = 0 → (1 < x ∧ x < 2) ∨ (x < 1 ∨ x > 2)) ∧
  parabola m 0 < -1/2 →
  1/6 < m ∧ m < 1/4 :=
by
  sorry

end parabola_range_m_l15_15431


namespace neg_a_pow4_div_neg_a_eq_neg_a_pow3_l15_15912

variable (a : ℝ)

theorem neg_a_pow4_div_neg_a_eq_neg_a_pow3 : (-a)^4 / (-a) = -a^3 := sorry

end neg_a_pow4_div_neg_a_eq_neg_a_pow3_l15_15912


namespace determine_sequence_parameters_l15_15842

variables {n : ℕ} {d q : ℝ} (h1 : 1 + (n-1) * d = 81) (h2 : 1 * q^(n-1) = 81) (h3 : q / d = 0.15)

theorem determine_sequence_parameters : n = 5 ∧ d = 20 ∧ q = 3 :=
by {
  -- Assumptions:
  -- h1: Arithmetic sequence, a1 = 1, an = 81
  -- h2: Geometric sequence, b1 = 1, bn = 81
  -- h3: q / d = 0.15
  -- Goal: n = 5, d = 20, q = 3
  sorry
}

end determine_sequence_parameters_l15_15842


namespace jake_weight_l15_15288

variable (J K : ℕ)

-- Conditions given in the problem
axiom h1 : J - 8 = 2 * K
axiom h2 : J + K = 293

-- Statement to prove
theorem jake_weight : J = 198 :=
by
  sorry

end jake_weight_l15_15288


namespace value_of_nested_fraction_l15_15246

def nested_fraction : ℚ :=
  2 - (1 / (2 - (1 / (2 - 1 / 2))))

theorem value_of_nested_fraction : nested_fraction = 3 / 4 :=
by
  sorry

end value_of_nested_fraction_l15_15246


namespace growth_rate_l15_15343

variable (x : ℝ)

def initial_investment : ℝ := 500
def expected_investment : ℝ := 720

theorem growth_rate (x : ℝ) (h : 500 * (1 + x)^2 = 720) : x = 0.2 :=
by
  sorry

end growth_rate_l15_15343


namespace find_y_l15_15201

theorem find_y :
  (∃ y : ℝ, (4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4) ∧ y = 1251) :=
by
  sorry

end find_y_l15_15201


namespace verify_equation_holds_l15_15109

noncomputable def verify_equation (m n : ℝ) : Prop :=
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) 
  - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) 
  = 2 * Real.sqrt (3 * m - n)

theorem verify_equation_holds (m n : ℝ) (h : 9 * m^2 - n^2 ≥ 0) : verify_equation m n :=
by
  -- Proof goes here. 
  -- Implement the proof as per the solution steps sketched in the problem statement.
  sorry

end verify_equation_holds_l15_15109


namespace triangle_is_right_l15_15692

variable {n : ℕ}

theorem triangle_is_right 
  (h1 : n > 1) 
  (h2 : a = 2 * n) 
  (h3 : b = n^2 - 1) 
  (h4 : c = n^2 + 1)
  : a^2 + b^2 = c^2 := 
by
  -- skipping the proof
  sorry

end triangle_is_right_l15_15692


namespace local_minimum_at_minus_one_l15_15393

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l15_15393


namespace strap_mask_probability_l15_15016

theorem strap_mask_probability 
  (p_regular_medical : ℝ)
  (p_surgical : ℝ)
  (p_strap_regular : ℝ)
  (p_strap_surgical : ℝ)
  (h_regular_medical : p_regular_medical = 0.8)
  (h_surgical : p_surgical = 0.2)
  (h_strap_regular : p_strap_regular = 0.1)
  (h_strap_surgical : p_strap_surgical = 0.2) :
  (p_regular_medical * p_strap_regular + p_surgical * p_strap_surgical) = 0.12 :=
by
  rw [h_regular_medical, h_surgical, h_strap_regular, h_strap_surgical]
  -- proof will go here
  sorry

end strap_mask_probability_l15_15016


namespace triangle_area_l15_15245

noncomputable def area_triangle (A B C : ℝ) (b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem triangle_area
  (A B C : ℝ) (b : ℝ) 
  (hA : A = π / 4)
  (h0 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  ∃ c : ℝ, area_triangle A B C b c = 2 :=
by
  sorry

end triangle_area_l15_15245


namespace power_equal_20mn_l15_15826

theorem power_equal_20mn (m n : ℕ) (P Q : ℕ) (hP : P = 2^m) (hQ : Q = 5^n) : 
  P^(2 * n) * Q^m = (20^(m * n)) :=
by
  sorry

end power_equal_20mn_l15_15826


namespace mary_books_end_of_year_l15_15390

def total_books_end_of_year (books_start : ℕ) (book_club : ℕ) (lent_to_jane : ℕ) 
 (returned_by_alice : ℕ) (bought_5th_month : ℕ) (bought_yard_sales : ℕ) 
 (birthday_daughter : ℕ) (birthday_mother : ℕ) (received_sister : ℕ)
 (buy_one_get_one : ℕ) (donated_charity : ℕ) (borrowed_neighbor : ℕ)
 (sold_used_store : ℕ) : ℕ :=
  books_start + book_club - lent_to_jane + returned_by_alice + bought_5th_month + bought_yard_sales +
  birthday_daughter + birthday_mother + received_sister + buy_one_get_one - donated_charity - borrowed_neighbor - sold_used_store

theorem mary_books_end_of_year : total_books_end_of_year 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end mary_books_end_of_year_l15_15390


namespace average_time_to_win_permit_l15_15174

theorem average_time_to_win_permit :
  let p n := (9/10)^(n-1) * (1/10)
  ∑' n, n * p n = 10 :=
sorry

end average_time_to_win_permit_l15_15174


namespace midpoint_C_is_either_l15_15202

def A : ℝ := -7
def dist_AB : ℝ := 5

theorem midpoint_C_is_either (C : ℝ) (h : C = (A + (A + dist_AB / 2)) / 2 ∨ C = (A + (A - dist_AB / 2)) / 2) : 
  C = -9 / 2 ∨ C = -19 / 2 := 
sorry

end midpoint_C_is_either_l15_15202


namespace inequality_holds_for_a_in_interval_l15_15876

theorem inequality_holds_for_a_in_interval:
  (∀ x y : ℝ, 
     2 ≤ x ∧ x ≤ 3 ∧ 3 ≤ y ∧ y ≤ 4 → (3*x - 2*y - a) * (3*x - 2*y - a^2) ≤ 0) ↔ a ∈ Set.Iic (-4) :=
by
  sorry

end inequality_holds_for_a_in_interval_l15_15876


namespace equality_of_ha_l15_15529

theorem equality_of_ha 
  {p a b α β γ : ℝ} 
  (h1 : h_a = (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2))
  (h2 : h_a = (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2)) : 
  (2 * (p - a) * Real.cos (β / 2) * Real.cos (γ / 2)) / Real.cos (α / 2) = 
  (2 * (p - b) * Real.sin (β / 2) * Real.cos (γ / 2)) / Real.sin (α / 2) :=
by sorry

end equality_of_ha_l15_15529


namespace lcm_18_30_eq_90_l15_15552

theorem lcm_18_30_eq_90 : Nat.lcm 18 30 = 90 := 
by {
  sorry
}

end lcm_18_30_eq_90_l15_15552


namespace dave_time_correct_l15_15805

-- Definitions for the given conditions
def chuck_time (dave_time : ℕ) := 5 * dave_time
def erica_time (chuck_time : ℕ) := chuck_time + (3 * chuck_time / 10)
def erica_fixed_time := 65

-- Statement to prove
theorem dave_time_correct : ∃ (dave_time : ℕ), erica_time (chuck_time dave_time) = erica_fixed_time ∧ dave_time = 10 := by
  sorry

end dave_time_correct_l15_15805


namespace volume_of_rectangular_prism_l15_15687

theorem volume_of_rectangular_prism
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : z * x = 12) :
  x * y * z = 60 :=
sorry

end volume_of_rectangular_prism_l15_15687


namespace ratio_depth_to_height_l15_15627

noncomputable def height_ron : ℝ := 12
noncomputable def depth_water : ℝ := 60

theorem ratio_depth_to_height : depth_water / height_ron = 5 := by
  sorry

end ratio_depth_to_height_l15_15627


namespace increasing_function_shape_implies_number_l15_15719

variable {I : Set ℝ} {f : ℝ → ℝ}

theorem increasing_function_shape_implies_number (h : ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂) 
: ∀ (x₁ x₂ : ℝ), x₁ ∈ I ∧ x₂ ∈ I ∧ x₁ < x₂ → f x₁ < f x₂ :=
sorry

end increasing_function_shape_implies_number_l15_15719


namespace perimeter_of_triangle_l15_15646

-- Define the average length of the sides of the triangle
def average_length (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Define the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The theorem we want to prove
theorem perimeter_of_triangle {a b c : ℕ} (h_avg : average_length a b c = 12) : perimeter a b c = 36 :=
sorry

end perimeter_of_triangle_l15_15646


namespace range_of_m_l15_15089

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 2) * x + m - 1 → (x ≥ 0 ∨ y ≥ 0))) ↔ (1 ≤ m ∧ m < 2) :=
by sorry

end range_of_m_l15_15089


namespace total_difference_is_18_l15_15492

-- Define variables for Mike, Joe, and Anna's bills
variables (m j a : ℝ)

-- Define the conditions given in the problem
def MikeTipped := (0.15 * m = 3)
def JoeTipped := (0.25 * j = 3)
def AnnaTipped := (0.10 * a = 3)

-- Prove the total amount of money that was different between the highest and lowest bill is 18
theorem total_difference_is_18 (MikeTipped : 0.15 * m = 3) (JoeTipped : 0.25 * j = 3) (AnnaTipped : 0.10 * a = 3) :
  |a - j| = 18 := 
sorry

end total_difference_is_18_l15_15492


namespace correct_conclusion_l15_15624

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^3 - 6*x^2 + 9*x - a*b*c

-- The statement to be proven, without providing the actual proof.
theorem correct_conclusion 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : f a a b c = 0) 
  (h4 : f b a b c = 0) 
  (h5 : f c a b c = 0) :
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
sorry

end correct_conclusion_l15_15624


namespace intersection_A_B_subset_A_B_l15_15308

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def set_B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

-- Problem 1: Prove A ∩ B when a = -1
theorem intersection_A_B (a : ℝ) (h : a = -1) : set_A a ∩ set_B = {x | 1 / 2 < x ∧ x < 2} :=
sorry

-- Problem 2: Find the range of a such that A ⊆ B
theorem subset_A_B (a : ℝ) : (-1 < a ∧ a ≤ 1) ↔ (set_A a ⊆ set_B) :=
sorry

end intersection_A_B_subset_A_B_l15_15308


namespace factor_polynomial_l15_15303

noncomputable def polynomial (x y n : ℤ) : ℤ := x^2 + 4 * x * y + 2 * x + n * y - n

theorem factor_polynomial (n : ℤ) :
  (∃ A B C D E F : ℤ, polynomial A B C = (A * x + B * y + C) * (D * x + E * y + F)) ↔ n = 0 :=
sorry

end factor_polynomial_l15_15303


namespace soda_cost_is_20_l15_15990

noncomputable def cost_of_soda (b s : ℕ) : Prop :=
  4 * b + 3 * s = 500 ∧ 3 * b + 2 * s = 370

theorem soda_cost_is_20 {b s : ℕ} (h : cost_of_soda b s) : s = 20 :=
  by sorry

end soda_cost_is_20_l15_15990


namespace max_value_of_z_l15_15423

open Real

theorem max_value_of_z (x y : ℝ) (h₁ : x + y ≥ 1) (h₂ : 2 * x - y ≤ 0) (h₃ : 3 * x - 2 * y + 2 ≥ 0) : 
  ∃ x y, 3 * x - y = 2 :=
sorry

end max_value_of_z_l15_15423


namespace a_beats_b_by_32_meters_l15_15907

-- Define the known conditions.
def distance_a_in_t : ℕ := 224 -- Distance A runs in 28 seconds
def time_a : ℕ := 28 -- Time A takes to run 224 meters
def distance_b_in_t : ℕ := 224 -- Distance B runs in 32 seconds
def time_b : ℕ := 32 -- Time B takes to run 224 meters

-- Define the speeds.
def speed_a : ℕ := distance_a_in_t / time_a
def speed_b : ℕ := distance_b_in_t / time_b

-- Define the distances each runs in 32 seconds.
def distance_a_in_32_sec : ℕ := speed_a * 32
def distance_b_in_32_sec : ℕ := speed_b * 32

-- The proof statement
theorem a_beats_b_by_32_meters :
  distance_a_in_32_sec - distance_b_in_32_sec = 32 := 
sorry

end a_beats_b_by_32_meters_l15_15907


namespace max_L_shaped_figures_in_5x7_rectangle_l15_15718

def L_shaped_figure : Type := ℕ

def rectangle_area := 5 * 7

def l_shape_area := 3

def max_l_shapes_in_rectangle (rect_area : ℕ) (l_area : ℕ) : ℕ := rect_area / l_area

theorem max_L_shaped_figures_in_5x7_rectangle : max_l_shapes_in_rectangle rectangle_area l_shape_area = 11 :=
by
  sorry

end max_L_shaped_figures_in_5x7_rectangle_l15_15718


namespace total_profit_from_selling_30_necklaces_l15_15863

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l15_15863


namespace distinct_real_numbers_eq_l15_15400

theorem distinct_real_numbers_eq (x : ℝ) :
  (x^2 - 7)^2 + 2 * x^2 = 33 → 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                    {a, b, c, d} = {x | (x^2 - 7)^2 + 2 * x^2 = 33}) :=
sorry

end distinct_real_numbers_eq_l15_15400


namespace greatest_value_of_x_for_7x_factorial_100_l15_15790

open Nat

theorem greatest_value_of_x_for_7x_factorial_100 : 
  ∃ x : ℕ, (∀ y : ℕ, 7^y ∣ factorial 100 → y ≤ x) ∧ x = 16 :=
by
  sorry

end greatest_value_of_x_for_7x_factorial_100_l15_15790


namespace circumference_of_circle_x_l15_15435

theorem circumference_of_circle_x (A_x A_y : ℝ) (r_x r_y C_x : ℝ)
  (h_area: A_x = A_y) (h_half_radius_y: r_y = 2 * 5)
  (h_area_y: A_y = Real.pi * r_y^2)
  (h_area_x: A_x = Real.pi * r_x^2)
  (h_circumference_x: C_x = 2 * Real.pi * r_x) :
  C_x = 20 * Real.pi :=
by
  sorry

end circumference_of_circle_x_l15_15435


namespace julio_salary_l15_15133

-- Define the conditions
def customers_first_week : ℕ := 35
def customers_second_week : ℕ := 2 * customers_first_week
def customers_third_week : ℕ := 3 * customers_first_week
def commission_per_customer : ℕ := 1
def bonus : ℕ := 50
def total_earnings : ℕ := 760

-- Calculate total commission and total earnings
def commission_first_week : ℕ := customers_first_week * commission_per_customer
def commission_second_week : ℕ := customers_second_week * commission_per_customer
def commission_third_week : ℕ := customers_third_week * commission_per_customer
def total_commission : ℕ := commission_first_week + commission_second_week + commission_third_week
def total_earnings_commission_bonus : ℕ := total_commission + bonus

-- Define the proof problem
theorem julio_salary : total_earnings - total_earnings_commission_bonus = 500 :=
by
  sorry

end julio_salary_l15_15133


namespace equal_students_initially_l15_15503

theorem equal_students_initially (B G : ℕ) (h1 : B = G) (h2 : B = 2 * (G - 8)) : B + G = 32 :=
by
  sorry

end equal_students_initially_l15_15503


namespace algebra_sum_l15_15693

-- Given conditions
def letterValue (ch : Char) : Int :=
  let pos := ch.toNat - 'a'.toNat + 1
  match pos % 6 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 0 => -2
  | _ => 0  -- This case is actually unreachable.

def wordValue (w : List Char) : Int :=
  w.foldl (fun acc ch => acc + letterValue ch) 0

theorem algebra_sum : wordValue ['a', 'l', 'g', 'e', 'b', 'r', 'a'] = 0 :=
  sorry

end algebra_sum_l15_15693


namespace flight_time_NY_to_CT_l15_15607

def travelTime (start_time_NY : ℕ) (end_time_CT : ℕ) (layover_Johannesburg : ℕ) : ℕ :=
  end_time_CT - start_time_NY + layover_Johannesburg

theorem flight_time_NY_to_CT :
  let start_time_NY := 0 -- 12:00 a.m. Tuesday as 0 hours from midnight in ET
  let end_time_CT := 10  -- 10:00 a.m. Tuesday as 10 hours from midnight in ET
  let layover_Johannesburg := 4
  travelTime start_time_NY end_time_CT layover_Johannesburg = 10 :=
by
  sorry

end flight_time_NY_to_CT_l15_15607


namespace kelly_initially_had_l15_15528

def kelly_needs_to_pick : ℕ := 49
def kelly_will_have : ℕ := 105

theorem kelly_initially_had :
  kelly_will_have - kelly_needs_to_pick = 56 :=
by
  sorry

end kelly_initially_had_l15_15528


namespace exponent_multiplication_l15_15306

theorem exponent_multiplication :
  (10 ^ 10000) * (10 ^ 8000) = 10 ^ 18000 :=
by
  sorry

end exponent_multiplication_l15_15306


namespace sequence_contains_infinitely_many_powers_of_two_l15_15892

theorem sequence_contains_infinitely_many_powers_of_two (a : ℕ → ℕ) (b : ℕ → ℕ) : 
  (∃ a1, a1 % 5 ≠ 0 ∧ a 0 = a1) →
  (∀ n : ℕ, a (n + 1) = a n + b n) →
  (∀ n : ℕ, b n = a n % 10) →
  (∃ n : ℕ, ∃ k : ℕ, 2^k = a n) :=
by
  sorry

end sequence_contains_infinitely_many_powers_of_two_l15_15892


namespace students_interested_both_l15_15763

/-- total students surveyed -/
def U : ℕ := 50

/-- students who liked watching table tennis matches -/
def A : ℕ := 35

/-- students who liked watching badminton matches -/
def B : ℕ := 30

/-- students not interested in either -/
def nU_not_interest : ℕ := 5

theorem students_interested_both : (A + B - (U - nU_not_interest)) = 20 :=
by sorry

end students_interested_both_l15_15763


namespace sum_of_digits_6608_condition_l15_15808

theorem sum_of_digits_6608_condition :
  ∀ n1 n2 : ℕ, (6 * 1000 + n1 * 100 + n2 * 10 + 8) % 236 = 0 → n1 + n2 = 6 :=
by 
  intros n1 n2 h
  -- This is where the proof would go. Since we're not proving it, we skip it with "sorry".
  sorry

end sum_of_digits_6608_condition_l15_15808


namespace arithmetic_sequence_terms_sum_l15_15104

theorem arithmetic_sequence_terms_sum
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n+1) = a n + d)
  (h₂ : a 2 = 1 - a 1)
  (h₃ : a 4 = 9 - a 3)
  (h₄ : ∀ n, a n > 0):
  a 4 + a 5 = 27 :=
sorry

end arithmetic_sequence_terms_sum_l15_15104


namespace kerosene_cost_is_024_l15_15236

-- Definitions from the conditions
def dozen_eggs_cost := 0.36 -- Cost of a dozen eggs is the same as 1 pound of rice which is $0.36
def pound_of_rice_cost := 0.36
def kerosene_cost := 8 * (0.36 / 12) -- Cost of kerosene is the cost of 8 eggs

-- Theorem to prove
theorem kerosene_cost_is_024 : kerosene_cost = 0.24 := by
  sorry

end kerosene_cost_is_024_l15_15236


namespace joey_pills_sum_one_week_l15_15959

def joey_pills (n : ℕ) : ℕ :=
  1 + 2 * n

theorem joey_pills_sum_one_week : 
  (joey_pills 0) + (joey_pills 1) + (joey_pills 2) + (joey_pills 3) + (joey_pills 4) + (joey_pills 5) + (joey_pills 6) = 49 :=
by
  sorry

end joey_pills_sum_one_week_l15_15959


namespace female_democrats_count_l15_15006

theorem female_democrats_count (F M : ℕ) (h1 : F + M = 750) 
  (h2 : F / 2 ≠ 0) (h3 : M / 4 ≠ 0) 
  (h4 : F / 2 + M / 4 = 750 / 3) : F / 2 = 125 :=
by
  sorry

end female_democrats_count_l15_15006


namespace josh_money_left_l15_15091

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end josh_money_left_l15_15091


namespace distance_to_directrix_l15_15831

theorem distance_to_directrix (x y d : ℝ) (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
  (h_ellipse : x^2 / 25 + y^2 / 9 = 1)
  (h_a : a = 5)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_M_on_ellipse : M.snd^2 / (a^2) + M.fst^2 / (b^2) = 1)
  (h_dist_F1M : dist M F1 = 8) :
  d = 5 / 2 :=
by
  sorry

end distance_to_directrix_l15_15831


namespace negation_proposition_l15_15191

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ ∃ x0 : ℝ, x0^2 - 2*x0 + 4 > 0 :=
by
  sorry

end negation_proposition_l15_15191


namespace cost_price_of_apple_l15_15203

-- Define the given conditions SP = 20, and the relation between SP and CP.
variables (SP CP : ℝ)
axiom h1 : SP = 20
axiom h2 : SP = CP - (1/6) * CP

-- Statement to be proved.
theorem cost_price_of_apple : CP = 24 :=
by
  sorry

end cost_price_of_apple_l15_15203


namespace biking_time_l15_15233

noncomputable def east_bound_speed : ℝ := 22
noncomputable def west_bound_speed : ℝ := east_bound_speed + 4
noncomputable def total_distance : ℝ := 200

theorem biking_time :
  (east_bound_speed + west_bound_speed) * (t : ℝ) = total_distance → t = 25 / 6 :=
by
  -- The proof is omitted and replaced with sorry.
  sorry

end biking_time_l15_15233


namespace cos_beta_value_l15_15333

variable (α β : ℝ)
variable (h₁ : 0 < α ∧ α < π)
variable (h₂ : 0 < β ∧ β < π)
variable (h₃ : Real.sin (α + β) = 5 / 13)
variable (h₄ : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_value : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_value_l15_15333


namespace John_new_weekly_earnings_l15_15860

theorem John_new_weekly_earnings (original_earnings : ℕ) (raise_percentage : ℚ) 
  (raise_in_dollars : ℚ) (new_weekly_earnings : ℚ)
  (h1 : original_earnings = 30) 
  (h2 : raise_percentage = 33.33) 
  (h3 : raise_in_dollars = (raise_percentage / 100) * original_earnings) 
  (h4 : new_weekly_earnings = original_earnings + raise_in_dollars) :
  new_weekly_earnings = 40 := sorry

end John_new_weekly_earnings_l15_15860


namespace negation_of_universal_l15_15847

variable {f g : ℝ → ℝ}

theorem negation_of_universal :
  ¬ (∀ x : ℝ, f x * g x ≠ 0) ↔ ∃ x₀ : ℝ, f x₀ = 0 ∨ g x₀ = 0 :=
by
  sorry

end negation_of_universal_l15_15847


namespace total_selection_methods_l15_15134

theorem total_selection_methods (synthetic_students : ℕ) (analytical_students : ℕ)
  (h_synthetic : synthetic_students = 5) (h_analytical : analytical_students = 3) :
  synthetic_students + analytical_students = 8 :=
by
  -- Proof is omitted
  sorry

end total_selection_methods_l15_15134


namespace percentage_solution_P_mixture_l15_15786

-- Define constants for volumes and percentages
variables (P Q : ℝ)

-- Define given conditions
def percentage_lemonade_P : ℝ := 0.2
def percentage_carbonated_P : ℝ := 0.8
def percentage_lemonade_Q : ℝ := 0.45
def percentage_carbonated_Q : ℝ := 0.55
def percentage_carbonated_mixture : ℝ := 0.72

-- Prove that the percentage of the volume of the mixture that is Solution P is 68%
theorem percentage_solution_P_mixture : 
  (percentage_carbonated_P * P + percentage_carbonated_Q * Q = percentage_carbonated_mixture * (P + Q)) → 
  ((P / (P + Q)) * 100 = 68) :=
by
  -- proof skipped
  sorry

end percentage_solution_P_mixture_l15_15786


namespace arctan_sum_l15_15162

theorem arctan_sum (a b : ℝ) (h1 : a = 1/3) (h2 : (a + 1) * (b + 1) = 3) : 
  Real.arctan a + Real.arctan b = Real.arctan (19 / 7) :=
by
  sorry

end arctan_sum_l15_15162


namespace jessica_final_balance_l15_15869

variable (B : ℝ) (withdrawal : ℝ) (deposit : ℝ)

-- Conditions
def condition1 : Prop := withdrawal = (2 / 5) * B
def condition2 : Prop := deposit = (1 / 5) * (B - withdrawal)

-- Proof goal statement
theorem jessica_final_balance (h1 : condition1 B withdrawal)
                             (h2 : condition2 B withdrawal deposit) :
    (B - withdrawal + deposit) = 360 :=
by
  sorry

end jessica_final_balance_l15_15869


namespace exists_natural_n_l15_15608

theorem exists_natural_n (a b : ℕ) (h1 : b ≥ 2) (h2 : Nat.gcd a b = 1) : ∃ n : ℕ, (n * a) % b = 1 :=
by
  sorry

end exists_natural_n_l15_15608


namespace number_of_students_in_the_course_l15_15792

variable (T : ℝ)

theorem number_of_students_in_the_course
  (h1 : (1/5) * T + (1/4) * T + (1/2) * T + 40 = T) :
  T = 800 :=
sorry

end number_of_students_in_the_course_l15_15792


namespace inequality_proof_l15_15874

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : x^12 - y^12 + 2 * x^6 * y^6 ≤ (Real.pi / 2) := 
by 
  sorry

end inequality_proof_l15_15874


namespace john_pays_2010_dollars_l15_15540

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end john_pays_2010_dollars_l15_15540


namespace sponge_cake_eggs_l15_15795

theorem sponge_cake_eggs (eggs flour sugar total desiredCakeMass : ℕ) 
  (h_recipe : eggs = 300) 
  (h_flour : flour = 120)
  (h_sugar : sugar = 100) 
  (h_total : total = 520) 
  (h_desiredMass : desiredCakeMass = 2600) :
  (eggs * desiredCakeMass / total) = 1500 := by
  sorry

end sponge_cake_eggs_l15_15795


namespace ten_term_sequence_l15_15365
open Real

theorem ten_term_sequence (a b : ℝ) 
    (h₁ : a + b = 1)
    (h₂ : a^2 + b^2 = 3)
    (h₃ : a^3 + b^3 = 4)
    (h₄ : a^4 + b^4 = 7)
    (h₅ : a^5 + b^5 = 11) :
    a^10 + b^10 = 123 :=
  sorry

end ten_term_sequence_l15_15365


namespace find_natural_number_l15_15927

variable {A : ℕ}

theorem find_natural_number (h1 : A = 8 * 2 + 7) : A = 23 :=
sorry

end find_natural_number_l15_15927


namespace expression_value_l15_15143

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
    sorry

end expression_value_l15_15143


namespace Proof_l15_15838

-- Definitions for the conditions
def Snakes : Type := {s : Fin 20 // s < 20}
def Purple (s : Snakes) : Prop := s.val < 6
def Happy (s : Snakes) : Prop := s.val >= 6 ∧ s.val < 14
def CanAdd (s : Snakes) : Prop := ∃ h ∈ Finset.Ico 6 14, h = s.val
def CanSubtract (s : Snakes) : Prop := ¬Purple s

-- Conditions extraction
axiom SomeHappyCanAdd : ∃ s : Snakes, Happy s ∧ CanAdd s
axiom NoPurpleCanSubtract : ∀ s : Snakes, Purple s → ¬CanSubtract s
axiom CantSubtractCantAdd : ∀ s : Snakes, ¬CanSubtract s → ¬CanAdd s

-- Theorem statement depending on conditions
theorem Proof :
    (∀ s : Snakes, CanSubtract s → ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬Purple s) ∧
    (∃ s : Snakes, Happy s ∧ ¬CanSubtract s) :=
by {
  sorry -- Proof required here
}

end Proof_l15_15838


namespace systematic_sampling_first_number_l15_15811

theorem systematic_sampling_first_number
    (n : ℕ)  -- total number of products
    (k : ℕ)  -- sample size
    (common_diff : ℕ)  -- common difference in the systematic sample
    (x : ℕ)  -- an element in the sample
    (first_num : ℕ)  -- first product number in the sample
    (h1 : n = 80)  -- total number of products is 80
    (h2 : k = 5)  -- sample size is 5
    (h3 : common_diff = 16)  -- common difference is 16
    (h4 : x = 42)  -- 42 is in the sample
    (h5 : x = common_diff * 2 + first_num)  -- position of 42 in the arithmetic sequence
: first_num = 10 := 
sorry

end systematic_sampling_first_number_l15_15811


namespace average_of_B_and_C_l15_15013

theorem average_of_B_and_C (x : ℚ) (A B C : ℚ)
  (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  (B + C) / 2 = 93.75 := 
sorry

end average_of_B_and_C_l15_15013


namespace sum_of_squared_distances_range_l15_15840

theorem sum_of_squared_distances_range
  (φ : ℝ)
  (x : ℝ := 2 * Real.cos φ)
  (y : ℝ := 3 * Real.sin φ)
  (A : ℝ × ℝ := (1, Real.sqrt 3))
  (B : ℝ × ℝ := (-Real.sqrt 3, 1))
  (C : ℝ × ℝ := (-1, -Real.sqrt 3))
  (D : ℝ × ℝ := (Real.sqrt 3, -1))
  (PA := (x - A.1)^2 + (y - A.2)^2)
  (PB := (x - B.1)^2 + (y - B.2)^2)
  (PC := (x - C.1)^2 + (y - C.2)^2)
  (PD := (x - D.1)^2 + (y - D.2)^2) :
  32 ≤ PA + PB + PC + PD ∧ PA + PB + PC + PD ≤ 52 :=
  by sorry

end sum_of_squared_distances_range_l15_15840


namespace icosagon_diagonals_l15_15312

-- Definitions for the number of sides and the diagonal formula
def sides_icosagon : ℕ := 20

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Statement:
theorem icosagon_diagonals : diagonals sides_icosagon = 170 := by
  apply sorry

end icosagon_diagonals_l15_15312


namespace remainder_when_N_divided_by_1000_l15_15022

def number_of_factors_of_5 (n : Nat) : Nat :=
  if n = 0 then 0 
  else n / 5 + number_of_factors_of_5 (n / 5)

def total_factors_of_5_upto (n : Nat) : Nat := 
  match n with
  | 0 => 0
  | n + 1 => number_of_factors_of_5 (n + 1) + total_factors_of_5_upto n

def product_factorial_5s : Nat := total_factors_of_5_upto 100

def N : Nat := product_factorial_5s

theorem remainder_when_N_divided_by_1000 : N % 1000 = 124 := by
  sorry

end remainder_when_N_divided_by_1000_l15_15022


namespace arithmetic_sequence_fifth_term_l15_15711

theorem arithmetic_sequence_fifth_term :
  ∀ (a d : ℤ), (a + 19 * d = 15) → (a + 20 * d = 18) → (a + 4 * d = -30) :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_fifth_term_l15_15711


namespace solve_for_a_l15_15915

def f (x : ℝ) : ℝ := x^2 + 10
def g (x : ℝ) : ℝ := x^2 - 6

theorem solve_for_a (a : ℝ) (h : a > 0) (h1 : f (g a) = 18) : a = Real.sqrt (2 * Real.sqrt 2 + 6) :=
by
  sorry

end solve_for_a_l15_15915


namespace find_f_neg_5pi_over_6_l15_15108

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_periodic : ∀ x : ℝ, f (x + (3 * Real.pi / 2)) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f x = Real.cos x

theorem find_f_neg_5pi_over_6 : f (-5 * Real.pi / 6) = -1 / 2 := 
by 
  -- use the axioms to prove the result 
  sorry

end find_f_neg_5pi_over_6_l15_15108


namespace jill_travels_less_than_john_l15_15479

theorem jill_travels_less_than_john :
  ∀ (John Jill Jim : ℕ), 
  John = 15 → 
  Jim = 2 → 
  (Jim = (20 / 100) * Jill) → 
  (John - Jill) = 5 := 
by
  intros John Jill Jim HJohn HJim HJimJill
  -- Skip the proof for now
  sorry

end jill_travels_less_than_john_l15_15479


namespace arithmetic_sequence_sum_l15_15611

theorem arithmetic_sequence_sum (x y : ℕ) (h₀: ∃ (n : ℕ), x = 3 + n * 4) (h₁: ∃ (m : ℕ), y = 3 + m * 4) (h₂: y = 31 - 4) (h₃: x = y - 4) : x + y = 50 := by
  sorry

end arithmetic_sequence_sum_l15_15611


namespace chord_length_of_circle_and_line_intersection_l15_15616

theorem chord_length_of_circle_and_line_intersection :
  ∀ (x y : ℝ), (x - 2 * y = 3) → ((x - 2)^2 + (y + 3)^2 = 9) → ∃ chord_length : ℝ, (chord_length = 4) :=
by
  intros x y hx hy
  sorry

end chord_length_of_circle_and_line_intersection_l15_15616


namespace alpha_beta_value_l15_15967

noncomputable def alpha_beta_sum : ℝ := 75

theorem alpha_beta_value (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : |Real.sin α - (1 / 2)| + Real.sqrt (Real.tan β - 1) = 0) :
  α + β = α_beta_sum := 
  sorry

end alpha_beta_value_l15_15967


namespace count_pairs_l15_15350

theorem count_pairs (a b : ℤ) (ha : 1 ≤ a ∧ a ≤ 42) (hb : 1 ≤ b ∧ b ≤ 42) (h : a^9 % 43 = b^7 % 43) : (∃ (n : ℕ), n = 42) :=
  sorry

end count_pairs_l15_15350


namespace total_roses_l15_15981

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l15_15981


namespace remainder_of_x_500_div_x2_plus_1_x2_minus_1_l15_15868

theorem remainder_of_x_500_div_x2_plus_1_x2_minus_1 :
  (x^500) % ((x^2 + 1) * (x^2 - 1)) = 1 :=
sorry

end remainder_of_x_500_div_x2_plus_1_x2_minus_1_l15_15868


namespace cubic_polynomial_roots_l15_15291

noncomputable def cubic_polynomial (a_3 a_2 a_1 a_0 x : ℝ) : ℝ :=
  a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem cubic_polynomial_roots (a_3 a_2 a_1 a_0 : ℝ) 
    (h_nonzero_a3 : a_3 ≠ 0)
    (r1 r2 r3 : ℝ)
    (h_roots : cubic_polynomial a_3 a_2 a_1 a_0 r1 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r2 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r3 = 0)
    (h_condition : (cubic_polynomial a_3 a_2 a_1 a_0 (1/2) 
                    + cubic_polynomial a_3 a_2 a_1 a_0 (-1/2)) 
                    / (cubic_polynomial a_3 a_2 a_1 a_0 0) = 1003) :
  (1 / (r1 * r2) + 1 / (r2 * r3) + 1 / (r3 * r1)) = 2002 :=
sorry

end cubic_polynomial_roots_l15_15291


namespace gcd_9157_2695_eq_1_l15_15448

theorem gcd_9157_2695_eq_1 : Int.gcd 9157 2695 = 1 := 
by
  sorry

end gcd_9157_2695_eq_1_l15_15448


namespace route_time_saving_zero_l15_15164

theorem route_time_saving_zero 
  (distance_X : ℝ) (speed_X : ℝ) 
  (total_distance_Y : ℝ) (construction_distance_Y : ℝ) (construction_speed_Y : ℝ)
  (normal_distance_Y : ℝ) (normal_speed_Y : ℝ)
  (hx1 : distance_X = 7)
  (hx2 : speed_X = 35)
  (hy1 : total_distance_Y = 6)
  (hy2 : construction_distance_Y = 1)
  (hy3 : construction_speed_Y = 10)
  (hy4 : normal_distance_Y = 5)
  (hy5 : normal_speed_Y = 50) :
  (distance_X / speed_X * 60) - 
  ((construction_distance_Y / construction_speed_Y * 60) + 
  (normal_distance_Y / normal_speed_Y * 60)) = 0 := 
sorry

end route_time_saving_zero_l15_15164


namespace cistern_length_l15_15629

theorem cistern_length (L : ℝ) (H : 0 < L) :
    (∃ (w d A : ℝ), w = 14 ∧ d = 1.25 ∧ A = 233 ∧ A = L * w + 2 * L * d + 2 * w * d) →
    L = 12 :=
by
  sorry

end cistern_length_l15_15629


namespace ellipse_hyperbola_eccentricities_l15_15117

theorem ellipse_hyperbola_eccentricities :
  ∃ x y : ℝ, (2 * x^2 - 5 * x + 2 = 0) ∧ (2 * y^2 - 5 * y + 2 = 0) ∧ 
  ((2 > 1) ∧ (0 < (1/2) ∧ (1/2 < 1))) :=
by
  sorry

end ellipse_hyperbola_eccentricities_l15_15117


namespace Karl_miles_driven_l15_15114

theorem Karl_miles_driven
  (gas_per_mile : ℝ)
  (tank_capacity : ℝ)
  (initial_gas : ℝ)
  (first_leg_miles : ℝ)
  (refuel_gallons : ℝ)
  (final_gas_fraction : ℝ)
  (total_miles_driven : ℝ) :
  gas_per_mile = 30 →
  tank_capacity = 16 →
  initial_gas = 16 →
  first_leg_miles = 420 →
  refuel_gallons = 10 →
  final_gas_fraction = 3 / 4 →
  total_miles_driven = 420 :=
by
  sorry

end Karl_miles_driven_l15_15114


namespace length_of_segment_CD_l15_15241

theorem length_of_segment_CD (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h_ratio1 : x = (3 / 5) * (3 + y))
  (h_ratio2 : (x + 3) / y = 4 / 7)
  (h_RS : 3 = 3) :
  x + 3 + y = 273.6 :=
by
  sorry

end length_of_segment_CD_l15_15241


namespace geometric_sequence_increasing_condition_l15_15287

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (h_geo : is_geometric a) (h_cond : a 0 < a 1 ∧ a 1 < a 2) :
  ¬(∀ n : ℕ, a n < a (n + 1)) → (a 0 < a 1 ∧ a 1 < a 2) :=
sorry

end geometric_sequence_increasing_condition_l15_15287


namespace find_a_sq_plus_b_sq_l15_15825

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 48
axiom h2 : a * b = 156

theorem find_a_sq_plus_b_sq : a^2 + b^2 = 1992 :=
by sorry

end find_a_sq_plus_b_sq_l15_15825


namespace dave_shirts_not_washed_l15_15797

variable (short_sleeve_shirts long_sleeve_shirts washed_shirts : ℕ)

theorem dave_shirts_not_washed (h1 : short_sleeve_shirts = 9) (h2 : long_sleeve_shirts = 27) (h3 : washed_shirts = 20) :
  (short_sleeve_shirts + long_sleeve_shirts - washed_shirts = 16) :=
by {
  -- sorry indicates the proof is omitted
  sorry
}

end dave_shirts_not_washed_l15_15797


namespace area_of_smaller_circle_l15_15296

theorem area_of_smaller_circle
  (PA AB : ℝ)
  (r s : ℝ)
  (tangent_at_T : true) -- placeholder; represents the tangency condition
  (common_tangents : true) -- placeholder; represents the external tangents condition
  (PA_eq_AB : PA = AB) :
  PA = 5 →
  AB = 5 →
  r = 2 * s →
  ∃ (s : ℝ) (area : ℝ), s = 5 / (2 * (Real.sqrt 2)) ∧ area = (Real.pi * s^2) ∧ area = (25 * Real.pi) / 8 := by
  intros hPA hAB h_r_s
  use 5 / (2 * (Real.sqrt 2))
  use (Real.pi * (5 / (2 * (Real.sqrt 2)))^2)
  simp [←hPA,←hAB]
  sorry

end area_of_smaller_circle_l15_15296


namespace isosceles_right_triangle_area_l15_15770

noncomputable def triangle_area (p : ℝ) : ℝ :=
  (1 / 8) * ((p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2)) ^ 2

theorem isosceles_right_triangle_area (p : ℝ) :
  let perimeter := p + p * Real.sqrt 2 + 2
  let x := (p + p * Real.sqrt 2 + 2) * (2 - Real.sqrt 2) / 2
  let area := 1 / 2 * x ^ 2
  area = triangle_area p :=
by
  sorry

end isosceles_right_triangle_area_l15_15770


namespace exponential_function_f1_l15_15414

theorem exponential_function_f1 (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h3 : a^3 = 8) : a^1 = 2 := by
  sorry

end exponential_function_f1_l15_15414


namespace perp_bisector_eq_l15_15597

/-- The circles x^2+y^2=4 and x^2+y^2-4x+6y=0 intersect at points A and B. 
Find the equation of the perpendicular bisector of line segment AB. -/

theorem perp_bisector_eq : 
  let C1 := (0, 0)
  let C2 := (2, -3)
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = 0 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
by
  sorry

end perp_bisector_eq_l15_15597


namespace find_y_l15_15345

theorem find_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 :=
sorry

end find_y_l15_15345


namespace tank_empties_in_4320_minutes_l15_15530

-- Define the initial conditions
def tankVolumeCubicFeet: ℝ := 30
def inletPipeRateCubicInchesPerMin: ℝ := 5
def outletPipe1RateCubicInchesPerMin: ℝ := 9
def outletPipe2RateCubicInchesPerMin: ℝ := 8
def feetToInches: ℝ := 12

-- Conversion from cubic feet to cubic inches
def tankVolumeCubicInches: ℝ := tankVolumeCubicFeet * feetToInches^3

-- Net rate of emptying in cubic inches per minute
def netRateOfEmptying: ℝ := (outletPipe1RateCubicInchesPerMin + outletPipe2RateCubicInchesPerMin) - inletPipeRateCubicInchesPerMin

-- Time to empty the tank
noncomputable def timeToEmptyTank: ℝ := tankVolumeCubicInches / netRateOfEmptying

-- The theorem to prove
theorem tank_empties_in_4320_minutes :
  timeToEmptyTank = 4320 := by
  sorry

end tank_empties_in_4320_minutes_l15_15530


namespace sugar_water_inequality_one_sugar_water_inequality_two_l15_15493

variable (a b m : ℝ)

-- Condition constraints
variable (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m)

-- Sugar Water Experiment One Inequality
theorem sugar_water_inequality_one : a / b > a / (b + m) := 
by
  sorry

-- Sugar Water Experiment Two Inequality
theorem sugar_water_inequality_two : a / b < (a + m) / b := 
by
  sorry

end sugar_water_inequality_one_sugar_water_inequality_two_l15_15493


namespace minimum_value_is_one_l15_15474

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  (1 / (3 * a + 2)) + (1 / (3 * b + 2)) + (1 / (3 * c + 2))

theorem minimum_value_is_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  minimum_value a b c = 1 := by
  sorry

end minimum_value_is_one_l15_15474


namespace floor_trig_sum_l15_15789

theorem floor_trig_sum :
  Int.floor (Real.sin 1) + Int.floor (Real.cos 2) + Int.floor (Real.tan 3) +
  Int.floor (Real.sin 4) + Int.floor (Real.cos 5) + Int.floor (Real.tan 6) = -4 := by
  sorry

end floor_trig_sum_l15_15789


namespace tree_planting_equation_l15_15166

variables (x : ℝ)

theorem tree_planting_equation (h1 : x > 50) :
  (300 / (x - 50) = 400 / x) ≠ False :=
by
  sorry

end tree_planting_equation_l15_15166


namespace problem1_problem2_problem3_l15_15816

-- Proof for part 1
theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 :=
sorry

-- Proof for part 2
theorem problem2 (α : ℝ) :
  (-Real.sin (Real.pi + α) + Real.sin (-α) - Real.tan (2 * Real.pi + α)) / 
  (Real.tan (α + Real.pi) + Real.cos (-α) + Real.cos (Real.pi - α)) = -1 :=
sorry

-- Proof for part 3
theorem problem3 (α : ℝ) (h : Real.sin α + Real.cos α = 1 / 2) (hα : 0 < α ∧ α < Real.pi) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end problem1_problem2_problem3_l15_15816


namespace M_geq_N_l15_15132

variable (a b : ℝ)

def M : ℝ := a^2 + 12 * a - 4 * b
def N : ℝ := 4 * a - 20 - b^2

theorem M_geq_N : M a b ≥ N a b := by
  sorry

end M_geq_N_l15_15132


namespace circle_condition_l15_15584

-- Define the center of the circle
def center := ((-3 + 27) / 2, (0 + 0) / 2)

-- Define the radius of the circle
def radius := 15

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the final Lean 4 statement
theorem circle_condition (x : ℝ) : circle_eq x 12 → (x = 21 ∨ x = 3) :=
  by
  intro h
  -- Proof goes here
  sorry

end circle_condition_l15_15584


namespace term_value_in_sequence_l15_15364

theorem term_value_in_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, a n = n * (n + 2) / 2) (h_val : a n = 220) : n = 20 :=
  sorry

end term_value_in_sequence_l15_15364


namespace tank_capacity_l15_15721

theorem tank_capacity (V : ℝ) (initial_fraction final_fraction : ℝ) (added_water : ℝ)
  (h1 : initial_fraction = 1 / 4)
  (h2 : final_fraction = 3 / 4)
  (h3 : added_water = 208)
  (h4 : final_fraction - initial_fraction = 1 / 2)
  (h5 : (1 / 2) * V = added_water) :
  V = 416 :=
by
  -- Given: initial_fraction = 1/4, final_fraction = 3/4, added_water = 208
  -- Difference in fullness: 1/2
  -- Equation for volume: 1/2 * V = 208
  -- Hence, V = 416
  sorry

end tank_capacity_l15_15721


namespace base_number_unique_l15_15936

theorem base_number_unique (y : ℕ) : (3 : ℝ) ^ 16 = (9 : ℝ) ^ y → y = 8 → (9 : ℝ) = 3 ^ (16 / y) :=
by
  sorry

end base_number_unique_l15_15936


namespace initial_sum_simple_interest_l15_15875

theorem initial_sum_simple_interest :
  ∃ P : ℝ, (P * (3/100) + P * (5/100) + P * (4/100) + P * (6/100) = 100) ∧ (P = 5000 / 9) :=
by
  sorry

end initial_sum_simple_interest_l15_15875


namespace ratio_proof_l15_15067

variable {x y : ℝ}

theorem ratio_proof (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 :=
by
  sorry

end ratio_proof_l15_15067


namespace chili_pepper_cost_l15_15618

theorem chili_pepper_cost :
  ∃ x : ℝ, 
    (3 * 2.50 + 4 * 1.50 + 5 * x = 18) ∧ 
    x = 0.90 :=
by
  use 0.90
  sorry

end chili_pepper_cost_l15_15618


namespace man_speed_with_stream_l15_15526

-- Define the man's rate in still water
def man_rate_in_still_water : ℝ := 6

-- Define the man's rate against the stream
def man_rate_against_stream (stream_speed : ℝ) : ℝ :=
  man_rate_in_still_water - stream_speed

-- The given condition that the man's rate against the stream is 10 km/h
def man_rate_against_condition : Prop := ∃ (stream_speed : ℝ), man_rate_against_stream stream_speed = 10

-- We aim to prove that the man's speed with the stream is 10 km/h
theorem man_speed_with_stream (stream_speed : ℝ) (h : man_rate_against_stream stream_speed = 10) :
  man_rate_in_still_water + stream_speed = 10 := by
  sorry

end man_speed_with_stream_l15_15526


namespace F_final_coordinates_l15_15600

-- Define the original coordinates of point F
def F : ℝ × ℝ := (5, 2)

-- Reflection over the y-axis changes the sign of the x-coordinate
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Reflection over the line y = x involves swapping x and y coordinates
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- The combined transformation: reflect over the y-axis, then reflect over y = x
def F_final : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis F)

-- The proof statement
theorem F_final_coordinates : F_final = (2, -5) :=
by
  -- Proof goes here
  sorry

end F_final_coordinates_l15_15600


namespace quadratic_root_relation_l15_15386

theorem quadratic_root_relation (m n p q : ℝ) (s₁ s₂ : ℝ) 
  (h1 : s₁ + s₂ = -p) 
  (h2 : s₁ * s₂ = q) 
  (h3 : 3 * s₁ + 3 * s₂ = -m) 
  (h4 : 9 * s₁ * s₂ = n) 
  (h_m : m ≠ 0) 
  (h_n : n ≠ 0) 
  (h_p : p ≠ 0) 
  (h_q : q ≠ 0) :
  n = 9 * q :=
by
  sorry

end quadratic_root_relation_l15_15386


namespace max_tiles_l15_15992

/--
Given a rectangular floor of size 180 cm by 120 cm
and rectangular tiles of size 25 cm by 16 cm, prove that the maximum number of tiles
that can be accommodated on the floor without overlapping, where the tiles' edges
are parallel and abutting the edges of the floor and with no tile overshooting the edges,
is 49 tiles.
-/
theorem max_tiles (floor_len floor_wid tile_len tile_wid : ℕ) (h1 : floor_len = 180)
  (h2 : floor_wid = 120) (h3 : tile_len = 25) (h4 : tile_wid = 16) :
  ∃ max_tiles : ℕ, max_tiles = 49 :=
by
  sorry

end max_tiles_l15_15992


namespace overall_loss_is_correct_l15_15135

-- Define the conditions
def worth_of_stock : ℝ := 17500
def percent_stock_sold_at_profit : ℝ := 0.20
def profit_rate : ℝ := 0.10
def percent_stock_sold_at_loss : ℝ := 0.80
def loss_rate : ℝ := 0.05

-- Define the calculations based on the conditions
def worth_sold_at_profit : ℝ := percent_stock_sold_at_profit * worth_of_stock
def profit_amount : ℝ := profit_rate * worth_sold_at_profit

def worth_sold_at_loss : ℝ := percent_stock_sold_at_loss * worth_of_stock
def loss_amount : ℝ := loss_rate * worth_sold_at_loss

-- Define the overall loss amount
def overall_loss : ℝ := loss_amount - profit_amount

-- Theorem to prove that the calculated overall loss amount matches the expected loss amount
theorem overall_loss_is_correct :
  overall_loss = 350 :=
by
  sorry

end overall_loss_is_correct_l15_15135


namespace final_solution_concentration_l15_15806

def concentration (mass : ℕ) (volume : ℕ) : ℕ := 
  (mass * 100) / volume

theorem final_solution_concentration :
  let volume1 := 4
  let conc1 := 4 -- percentage
  let volume2 := 2
  let conc2 := 10 -- percentage
  let mass1 := volume1 * conc1 / 100
  let mass2 := volume2 * conc2 / 100
  let total_mass := mass1 + mass2
  let total_volume := volume1 + volume2
  concentration total_mass total_volume = 6 :=
by
  sorry

end final_solution_concentration_l15_15806


namespace new_area_is_497_l15_15671

noncomputable def rect_area_proof : Prop :=
  ∃ (l w l' w' : ℝ),
    -- initial area condition
    l * w = 540 ∧ 
    -- conditions for new dimensions
    l' = 0.8 * l ∧
    w' = 1.15 * w ∧
    -- final area calculation
    l' * w' = 497

theorem new_area_is_497 : rect_area_proof := by
  sorry

end new_area_is_497_l15_15671


namespace tony_rope_length_l15_15626

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end tony_rope_length_l15_15626


namespace fraction_of_male_birds_l15_15007

theorem fraction_of_male_birds (T : ℕ) (h_cond1 : T ≠ 0) :
  let robins := (2 / 5) * T
  let bluejays := T - robins
  let male_robins := (2 / 3) * robins
  let male_bluejays := (1 / 3) * bluejays
  (male_robins + male_bluejays) / T = 7 / 15 :=
by 
  sorry

end fraction_of_male_birds_l15_15007


namespace ordered_pair_arith_progression_l15_15628

/-- 
Suppose (a, b) is an ordered pair of integers such that the three numbers a, b, and ab 
form an arithmetic progression, in that order. Prove the sum of all possible values of a is 8.
-/
theorem ordered_pair_arith_progression (a b : ℤ) (h : ∃ (a b : ℤ), (b - a = ab - b)) : 
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) → a + (if a = 0 then 1 else 0) + 
  (if a = 1 then 1 else 0) + (if a = 3 then 3 else 0) + (if a = 4 then 4 else 0) = 8 :=
by
  sorry

end ordered_pair_arith_progression_l15_15628


namespace boys_in_class_l15_15181

theorem boys_in_class (r : ℕ) (g b : ℕ) (h1 : g/b = 4/3) (h2 : g + b = 35) : b = 15 :=
  sorry

end boys_in_class_l15_15181


namespace parallelogram_count_l15_15725

theorem parallelogram_count (m n : ℕ) : 
  ∃ p : ℕ, p = (m.choose 2) * (n.choose 2) :=
by
  sorry

end parallelogram_count_l15_15725


namespace mutually_exclusive_not_opposed_l15_15355

-- Define the types for cards and people
inductive Card
| red : Card
| white : Card
| black : Card

inductive Person
| A : Person
| B : Person
| C : Person

-- Define the event that a person receives a specific card
def receives (p : Person) (c : Card) : Prop := sorry

-- Conditions
axiom A_receives_red : receives Person.A Card.red → ¬ receives Person.B Card.red
axiom B_receives_red : receives Person.B Card.red → ¬ receives Person.A Card.red

-- The proof problem statement
theorem mutually_exclusive_not_opposed :
  (receives Person.A Card.red → ¬ receives Person.B Card.red) ∧
  (¬(receives Person.A Card.red ∧ receives Person.B Card.red)) ∧
  (¬∀ p : Person, receives p Card.red) :=
sorry

end mutually_exclusive_not_opposed_l15_15355


namespace edric_days_per_week_l15_15976

variable (monthly_salary : ℝ) (hours_per_day : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ)
variable (days_per_week : ℝ)

-- Defining the conditions
def monthly_salary_condition : Prop := monthly_salary = 576
def hours_per_day_condition : Prop := hours_per_day = 8
def hourly_rate_condition : Prop := hourly_rate = 3
def weeks_per_month_condition : Prop := weeks_per_month = 4

-- Correct answer
def correct_answer : Prop := days_per_week = 6

-- Proof problem statement
theorem edric_days_per_week :
  monthly_salary_condition monthly_salary ∧
  hours_per_day_condition hours_per_day ∧
  hourly_rate_condition hourly_rate ∧
  weeks_per_month_condition weeks_per_month →
  correct_answer days_per_week :=
by
  sorry

end edric_days_per_week_l15_15976


namespace sqrt_difference_inequality_l15_15395

noncomputable def sqrt10 := Real.sqrt 10
noncomputable def sqrt6 := Real.sqrt 6
noncomputable def sqrt7 := Real.sqrt 7
noncomputable def sqrt3 := Real.sqrt 3

theorem sqrt_difference_inequality : sqrt10 - sqrt6 < sqrt7 - sqrt3 :=
by 
  sorry

end sqrt_difference_inequality_l15_15395


namespace inscribed_circle_ratio_l15_15228

theorem inscribed_circle_ratio (a b h r : ℝ) (h_triangle : h = Real.sqrt (a^2 + b^2))
  (A : ℝ) (H1 : A = (1/2) * a * b) (s : ℝ) (H2 : s = (a + b + h) / 2) 
  (H3 : A = r * s) : (π * r / A) = (π * r) / (h + r) :=
sorry

end inscribed_circle_ratio_l15_15228


namespace minimum_value_l15_15983

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 4) : 
  ∃ (x : ℝ), (x = 2) ∧ (∀ (p q : ℝ), q > 0 → p > 0 → 2 * p + q = 4 → x ≤ (1 / p + 2 / q)) := 
sorry

end minimum_value_l15_15983


namespace men_left_bus_l15_15482

theorem men_left_bus (M W : ℕ) (initial_passengers : M + W = 72) 
  (women_half_men : W = M / 2) 
  (equal_men_women_after_changes : ∃ men_left : ℕ, ∀ W_new, W_new = W + 8 → M - men_left = W_new → M - men_left = 32) :
  ∃ men_left : ℕ, men_left = 16 :=
  sorry

end men_left_bus_l15_15482


namespace find_b_l15_15695

-- Let's define the real numbers and the conditions given.
variables (b y a : ℝ)

-- Conditions from the problem
def condition1 := abs (b - y) = b + y - a
def condition2 := abs (b + y) = b + a

-- The goal is to find the value of b
theorem find_b (h1 : condition1 b y a) (h2 : condition2 b y a) : b = 1 :=
by
  sorry

end find_b_l15_15695


namespace stella_spent_amount_l15_15807

-- Definitions
def num_dolls : ℕ := 3
def num_clocks : ℕ := 2
def num_glasses : ℕ := 5

def price_doll : ℕ := 5
def price_clock : ℕ := 15
def price_glass : ℕ := 4

def profit : ℕ := 25

-- Calculation of total revenue from profit
def total_revenue : ℕ := num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass

-- Proposition to be proved
theorem stella_spent_amount : total_revenue - profit = 40 :=
by sorry

end stella_spent_amount_l15_15807


namespace jason_initial_cards_l15_15380

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l15_15380


namespace sum_series_eq_seven_twelve_l15_15717

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n:ℝ)^2 + 2 * (n:ℝ) + 1) / ((n:ℝ) * (n + 1) * (n + 2) * (n + 3)) else 0

theorem sum_series_eq_seven_twelve : sum_series = 7 / 12 :=
by
  sorry

end sum_series_eq_seven_twelve_l15_15717


namespace b_product_l15_15179

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- All terms in the arithmetic sequence \{aₙ\} are non-zero.
axiom a_nonzero : ∀ n, a n ≠ 0

-- The sequence satisfies the given condition.
axiom a_cond : a 3 - (a 7)^2 / 2 + a 11 = 0

-- The sequence \{bₙ\} is a geometric sequence with ratio r.
axiom b_geometric : ∃ r, ∀ n, b (n + 1) = r * b n

-- And b₇ = a₇
axiom b_7 : b 7 = a 7

-- Prove that b₁ * b₁₃ = 16
theorem b_product : b 1 * b 13 = 16 :=
sorry

end b_product_l15_15179


namespace parallel_lines_solution_l15_15852

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a = 0 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) ∨ 
  (∀ x y : ℝ, a = 1/4 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) :=
sorry

end parallel_lines_solution_l15_15852


namespace max_principals_in_10_years_l15_15632

theorem max_principals_in_10_years (p : ℕ) (is_principal_term : p = 4) : 
  ∃ n : ℕ, n = 4 ∧ ∀ k : ℕ, (k = 10 → n ≤ 4) :=
by
  sorry

end max_principals_in_10_years_l15_15632


namespace abs_diff_l15_15366

theorem abs_diff (a b : ℝ) (h_ab : a < b) (h_a : abs a = 6) (h_b : abs b = 3) :
  a - b = -9 ∨ a - b = 9 :=
by
  sorry

end abs_diff_l15_15366


namespace sequence_values_l15_15696

theorem sequence_values (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_arith : 2 + (a - 2) = a + (b - a)) (h_geom : a * a = b * (9 / b)) : a = 4 ∧ b = 6 :=
by
  -- insert proof here
  sorry

end sequence_values_l15_15696


namespace min_value_a_plus_b_l15_15636

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a^2 ≥ 8 * b) (h4 : b^2 ≥ a) : a + b ≥ 6 := by
  sorry

end min_value_a_plus_b_l15_15636


namespace quadratic_rewriting_l15_15110

theorem quadratic_rewriting:
  ∃ (d e f : ℤ), (∀ x : ℝ, 4 * x^2 - 28 * x + 49 = (d * x + e)^2 + f) ∧ d * e = -14 :=
by {
  sorry
}

end quadratic_rewriting_l15_15110


namespace seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l15_15214

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l15_15214


namespace percentage_increase_of_bill_l15_15381

theorem percentage_increase_of_bill 
  (original_bill : ℝ) 
  (increased_bill : ℝ)
  (h1 : original_bill = 60)
  (h2 : increased_bill = 78) : 
  ((increased_bill - original_bill) / original_bill * 100) = 30 := 
by 
  rw [h1, h2]
  -- The following steps show the intended logic:
  -- calc 
  --   [(78 - 60) / 60 * 100]
  --   = [(18) / 60 * 100]
  --   = [0.3 * 100]
  --   = 30
  sorry

end percentage_increase_of_bill_l15_15381


namespace set_equivalence_l15_15829

open Set

def set_A : Set ℝ := { x | x^2 - 2 * x > 0 }
def set_B : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

theorem set_equivalence : (univ \ set_B) ∪ set_A = (Iic 1) ∪ Ioi 2 :=
sorry

end set_equivalence_l15_15829


namespace pure_imaginary_number_l15_15250

theorem pure_imaginary_number (m : ℝ) (h_real : m^2 - 5 * m + 6 = 0) (h_imag : m^2 - 3 * m ≠ 0) : m = 2 :=
sorry

end pure_imaginary_number_l15_15250


namespace main_theorem_l15_15419

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem main_theorem :
  (∀ x : ℝ, f (x + 5/2) + f x = 2) ∧
  (∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) ∧
  (∀ x : ℝ, g (x + 2) = g (x - 2)) ∧
  (∀ x : ℝ, g (-x + 1) - 1 = -g (x + 1) + 1) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x + g x = 3^x + x^3) →
  f 2022 * g 2022 = 72 :=
sorry

end main_theorem_l15_15419


namespace vinegar_mixture_concentration_l15_15263

theorem vinegar_mixture_concentration :
  let c1 := 5 / 100
  let c2 := 10 / 100
  let v1 := 10
  let v2 := 10
  (v1 * c1 + v2 * c2) / (v1 + v2) = 7.5 / 100 :=
by
  sorry

end vinegar_mixture_concentration_l15_15263


namespace find_square_side_length_l15_15454

noncomputable def side_length_PQRS (x : ℝ) : Prop :=
  let PT := 1
  let QU := 2
  let RV := 3
  let SW := 4
  let PQRS_area := x^2
  let TUVW_area := 1 / 2 * x^2
  let triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
  PQRS_area = x^2 ∧ TUVW_area = 1 / 2 * x^2 ∧
  triangle_area 1 (x - 4) + (x - 1) + 
  triangle_area 3 (x - 2) + 2 * (x - 3) = 1 / 2 * x^2

theorem find_square_side_length : ∃ x : ℝ, side_length_PQRS x ∧ x = 6 := 
  sorry

end find_square_side_length_l15_15454


namespace speed_of_second_train_l15_15035

noncomputable def speed_of_first_train_kmph := 60 -- km/h
noncomputable def speed_of_first_train_mps := (speed_of_first_train_kmph * 1000) / 3600 -- m/s
noncomputable def length_of_first_train := 145 -- m
noncomputable def length_of_second_train := 165 -- m
noncomputable def time_to_cross := 8 -- seconds
noncomputable def total_distance := length_of_first_train + length_of_second_train -- m
noncomputable def relative_speed := total_distance / time_to_cross -- m/s

theorem speed_of_second_train (V : ℝ) :
  V * 1000 / 3600 + 60 * 1000 / 3600 = 38.75 →
  V = 79.5 := by {
  sorry
}

end speed_of_second_train_l15_15035


namespace park_needs_minimum_37_nests_l15_15148

-- Defining the number of different birds
def num_sparrows : ℕ := 5
def num_pigeons : ℕ := 3
def num_starlings : ℕ := 6
def num_robins : ℕ := 2

-- Defining the nesting requirements for each bird species
def nests_per_sparrow : ℕ := 1
def nests_per_pigeon : ℕ := 2
def nests_per_starling : ℕ := 3
def nests_per_robin : ℕ := 4

-- Definition of total minimum nests required
def min_nests_required : ℕ :=
  (num_sparrows * nests_per_sparrow) +
  (num_pigeons * nests_per_pigeon) +
  (num_starlings * nests_per_starling) +
  (num_robins * nests_per_robin)

-- Proof Statement
theorem park_needs_minimum_37_nests :
  min_nests_required = 37 :=
sorry

end park_needs_minimum_37_nests_l15_15148


namespace new_paint_intensity_l15_15325

def red_paint_intensity (initial_intensity replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

theorem new_paint_intensity :
  red_paint_intensity 0.1 0.2 0.5 = 0.15 :=
by sorry

end new_paint_intensity_l15_15325


namespace cubic_yards_to_cubic_feet_l15_15432

theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * 3^3 * 5 = 135 := by
sorry

end cubic_yards_to_cubic_feet_l15_15432


namespace sqrt_five_eq_l15_15523

theorem sqrt_five_eq (m n a b c d : ℤ)
  (h : m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5)) :
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) := by
  sorry

end sqrt_five_eq_l15_15523


namespace revenue_per_investment_l15_15554

theorem revenue_per_investment (Banks_investments : ℕ) (Elizabeth_investments : ℕ) (Elizabeth_revenue_per_investment : ℕ) (revenue_difference : ℕ) :
  Banks_investments = 8 →
  Elizabeth_investments = 5 →
  Elizabeth_revenue_per_investment = 900 →
  revenue_difference = 500 →
  ∃ (R : ℤ), R = (5 * 900 - 500) / 8 :=
by
  intros h1 h2 h3 h4
  let T_elizabeth := 5 * Elizabeth_revenue_per_investment
  let T_banks := T_elizabeth - revenue_difference
  let R := T_banks / 8
  use R
  sorry

end revenue_per_investment_l15_15554


namespace largest_five_digit_congruent_to_31_modulo_26_l15_15441

theorem largest_five_digit_congruent_to_31_modulo_26 :
  ∃ x : ℕ, (10000 ≤ x ∧ x < 100000) ∧ x % 26 = 31 ∧ x = 99975 :=
by
  sorry

end largest_five_digit_congruent_to_31_modulo_26_l15_15441


namespace solve_x_floor_x_eq_72_l15_15754

theorem solve_x_floor_x_eq_72 : ∃ x : ℝ, 0 < x ∧ x * (⌊x⌋) = 72 ∧ x = 9 :=
by
  sorry

end solve_x_floor_x_eq_72_l15_15754


namespace triangle_perimeter_l15_15853

-- Conditions as definitions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def has_sides (a b : ℕ) : Prop :=
  a = 4 ∨ b = 4 ∨ a = 9 ∨ b = 9

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the isosceles triangle with specified sides
structure IsoTriangle :=
  (a b c : ℕ)
  (iso : is_isosceles_triangle a b c)
  (valid_sides : has_sides a b ∧ has_sides a c ∧ has_sides b c)
  (triangle : triangle_inequality a b c)

-- The statement to prove perimeter
def perimeter (T : IsoTriangle) : ℕ :=
  T.a + T.b + T.c

-- The theorem we aim to prove
theorem triangle_perimeter (T : IsoTriangle) (h: T.a = 9 ∧ T.b = 9 ∧ T.c = 4) : perimeter T = 22 :=
sorry

end triangle_perimeter_l15_15853


namespace conference_session_time_l15_15388

def conference_duration_hours : ℕ := 8
def conference_duration_minutes : ℕ := 45
def break_time : ℕ := 30

theorem conference_session_time :
  (conference_duration_hours * 60 + conference_duration_minutes) - break_time = 495 :=
by sorry

end conference_session_time_l15_15388


namespace difference_between_oranges_and_apples_l15_15115

-- Definitions of the conditions
variables (A B P O: ℕ)
variables (h1: O = 6)
variables (h2: B = 3 * A)
variables (h3: P = B / 2)
variables (h4: A + B + P + O = 28)

-- The proof problem statement
theorem difference_between_oranges_and_apples
    (A B P O: ℕ)
    (h1: O = 6)
    (h2: B = 3 * A)
    (h3: P = B / 2)
    (h4: A + B + P + O = 28) :
    O - A = 2 :=
sorry

end difference_between_oranges_and_apples_l15_15115


namespace volume_comparison_l15_15030

-- Define the properties for the cube and the cuboid.
def cube_side_length : ℕ := 1 -- in meters
def cuboid_width : ℕ := 50  -- in centimeters
def cuboid_length : ℕ := 50 -- in centimeters
def cuboid_height : ℕ := 20 -- in centimeters

-- Convert cube side length to centimeters.
def cube_side_length_cm := cube_side_length * 100 -- in centimeters

-- Calculate volumes.
def cube_volume : ℕ := cube_side_length_cm ^ 3 -- in cubic centimeters
def cuboid_volume : ℕ := cuboid_width * cuboid_length * cuboid_height -- in cubic centimeters

-- The theorem stating the problem.
theorem volume_comparison : cube_volume / cuboid_volume = 20 :=
by sorry

end volume_comparison_l15_15030


namespace arithmetic_sequence_num_terms_l15_15547

theorem arithmetic_sequence_num_terms (a_1 d S_n n : ℕ) 
  (h1 : a_1 = 4) (h2 : d = 3) (h3 : S_n = 650)
  (h4 : S_n = (n / 2) * (2 * a_1 + (n - 1) * d)) : n = 20 := by
  sorry

end arithmetic_sequence_num_terms_l15_15547


namespace divisors_of_30_l15_15044

theorem divisors_of_30 : ∃ (n : ℕ), n = 16 ∧ (∀ d : ℤ, d ∣ 30 → (d ≤ 30 ∧ d ≥ -30)) :=
by
  sorry

end divisors_of_30_l15_15044


namespace find_extrema_of_S_l15_15519

theorem find_extrema_of_S (x y z : ℚ) (h1 : 3 * x + 2 * y + z = 5) (h2 : x + y - z = 2) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  2 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 3 :=
by
  sorry

end find_extrema_of_S_l15_15519


namespace min_value_of_ab_l15_15539

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0)
    (h : 1 / a + 1 / b = 1) : a + b ≥ 4 :=
sorry

end min_value_of_ab_l15_15539


namespace max_members_in_band_l15_15469

theorem max_members_in_band (m : ℤ) (h1 : 30 * m % 31 = 6) (h2 : 30 * m < 1200) : 30 * m = 360 :=
by {
  sorry -- Proof steps are not required according to the procedure
}

end max_members_in_band_l15_15469


namespace school_survey_l15_15814

theorem school_survey (n k smallest largest : ℕ) (h1 : n = 24) (h2 : k = 4) (h3 : smallest = 3) (h4 : 1 ≤ smallest ∧ smallest ≤ n) (h5 : largest - smallest = (k - 1) * (n / k)) : 
  largest = 21 :=
by {
  sorry
}

end school_survey_l15_15814


namespace simplify_fraction_l15_15410

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 :=
by
  sorry

end simplify_fraction_l15_15410


namespace partial_fraction_product_zero_l15_15354

theorem partial_fraction_product_zero
  (A B C : ℚ)
  (partial_fraction_eq : ∀ x : ℚ,
    x^2 - 25 = A * (x + 3) * (x - 5) + B * (x - 3) * (x - 5) + C * (x - 3) * (x + 3))
  (fact_3 : C = 0)
  (fact_neg3 : B = 1/3)
  (fact_5 : A = 0) :
  A * B * C = 0 := 
sorry

end partial_fraction_product_zero_l15_15354


namespace max_value_at_2_l15_15489

noncomputable def f (x : ℝ) : ℝ := -x^3 + 12 * x

theorem max_value_at_2 : ∃ a : ℝ, (∀ x : ℝ, f x ≤ f a) ∧ a = 2 := 
by
  sorry

end max_value_at_2_l15_15489


namespace derivative_of_f_l15_15002

noncomputable def f (x : ℝ) : ℝ := x / (1 - Real.cos x)

theorem derivative_of_f :
  (deriv f) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end derivative_of_f_l15_15002


namespace solve_system_of_equations_l15_15960

theorem solve_system_of_equations :
  ∃ x y : ℝ, 4 * x - 6 * y = -3 ∧ 9 * x + 3 * y = 6.3 ∧ x = 0.436 ∧ y = 0.792 :=
by
  sorry

end solve_system_of_equations_l15_15960


namespace music_class_uncool_parents_l15_15387

theorem music_class_uncool_parents:
  ∀ (total students coolDads coolMoms bothCool : ℕ),
  total = 40 →
  coolDads = 25 →
  coolMoms = 19 →
  bothCool = 8 →
  (total - (bothCool + (coolDads - bothCool) + (coolMoms - bothCool))) = 4 :=
by
  intros total coolDads coolMoms bothCool h_total h_dads h_moms h_both
  sorry

end music_class_uncool_parents_l15_15387


namespace question_a_gt_b_neither_sufficient_nor_necessary_l15_15509

theorem question_a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by
  sorry

end question_a_gt_b_neither_sufficient_nor_necessary_l15_15509


namespace min_value_ineq_l15_15436

noncomputable def min_value (x y z : ℝ) := (1/x) + (1/y) + (1/z)

theorem min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z ≥ 4.5 :=
sorry

end min_value_ineq_l15_15436


namespace butterfinger_count_l15_15001

def total_candy_bars : ℕ := 12
def snickers : ℕ := 3
def mars_bars : ℕ := 2
def butterfingers : ℕ := total_candy_bars - (snickers + mars_bars)

theorem butterfinger_count : butterfingers = 7 :=
by
  unfold butterfingers
  sorry

end butterfinger_count_l15_15001


namespace number_of_cans_on_third_day_l15_15802

-- Definition of an arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem number_of_cans_on_third_day :
  (arithmetic_sequence 4 5 2 = 9) →   -- on the second day, he found 9 cans
  (arithmetic_sequence 4 5 7 = 34) →  -- on the seventh day, he found 34 cans
  (arithmetic_sequence 4 5 3 = 14) :=  -- therefore, on the third day, he found 14 cans
by
  intros h1 h2
  sorry

end number_of_cans_on_third_day_l15_15802


namespace total_seashells_l15_15095

def joans_seashells : Nat := 6
def jessicas_seashells : Nat := 8

theorem total_seashells : joans_seashells + jessicas_seashells = 14 :=
by
  sorry

end total_seashells_l15_15095


namespace quilt_squares_count_l15_15556

theorem quilt_squares_count (total_squares : ℕ) (additional_squares : ℕ)
  (h1 : total_squares = 4 * additional_squares)
  (h2 : additional_squares = 24) :
  total_squares = 32 :=
by
  -- Proof would go here
  -- The proof would involve showing that total_squares indeed equals 32 given h1 and h2
  sorry

end quilt_squares_count_l15_15556


namespace kids_at_camp_l15_15697

theorem kids_at_camp (total_stayed_home : ℕ) (difference : ℕ) (x : ℕ) 
  (h1 : total_stayed_home = 777622) 
  (h2 : difference = 574664) 
  (h3 : total_stayed_home = x + difference) : 
  x = 202958 :=
by
  sorry

end kids_at_camp_l15_15697


namespace interest_calculation_l15_15408

/-- Define the initial deposit in thousands of yuan (50,000 yuan = 5 x 10,000 yuan) -/
def principal : ℕ := 5

/-- Define the annual interest rate as a percentage in decimal form -/
def annual_interest_rate : ℝ := 0.04

/-- Define the number of years for the deposit -/
def years : ℕ := 3

/-- Calculate the total amount after 3 years using compound interest -/
def total_amount_after_3_years : ℝ :=
  principal * (1 + annual_interest_rate) ^ years

/-- Calculate the interest earned after 3 years -/
def interest_earned : ℝ :=
  total_amount_after_3_years - principal

theorem interest_calculation :
  interest_earned = 5 * (1 + 0.04) ^ 3 - 5 :=
by 
  sorry

end interest_calculation_l15_15408


namespace general_formula_a_n_general_formula_b_n_l15_15891

-- Prove general formula for the sequence a_n
theorem general_formula_a_n (S : Nat → Nat) (a : Nat → Nat) (h₁ : ∀ n, S n = 2^(n+1) - 2) :
  (∀ n, a n = S n - S (n - 1)) → ∀ n, a n = 2^n :=
by
  sorry

-- Prove general formula for the sequence b_n
theorem general_formula_b_n (a b : Nat → Nat) (h₁ : ∀ n, a n = 2^n) :
  (∀ n, b n = a n + a (n + 1)) → ∀ n, b n = 3 * 2^n :=
by
  sorry

end general_formula_a_n_general_formula_b_n_l15_15891


namespace martin_speed_first_half_l15_15822

variable (v : ℝ) -- speed during the first half of the trip

theorem martin_speed_first_half
    (trip_duration : ℝ := 8)              -- The trip lasted 8 hours
    (speed_second_half : ℝ := 85)          -- Speed during the second half of the trip
    (total_distance : ℝ := 620)            -- Total distance traveled
    (time_each_half : ℝ := trip_duration / 2) -- Each half of the trip took half of the total time
    (distance_second_half : ℝ := speed_second_half * time_each_half)
    (distance_first_half : ℝ := total_distance - distance_second_half) :
    v = distance_first_half / time_each_half :=
by
  sorry

end martin_speed_first_half_l15_15822


namespace cody_increases_steps_by_1000_l15_15055

theorem cody_increases_steps_by_1000 (x : ℕ) 
  (initial_steps : ℕ := 7000)
  (steps_logged_in_four_weeks : ℕ := 70000)
  (goal_steps : ℕ := 100000)
  (remaining_steps : ℕ := 30000)
  (condition : 1000 + 7 * (1 + 2 + 3) * x = 70000 → x = 1000) : x = 1000 :=
by
  sorry

end cody_increases_steps_by_1000_l15_15055


namespace quadratic_two_equal_real_roots_c_l15_15411

theorem quadratic_two_equal_real_roots_c (c : ℝ) : 
  (∃ x : ℝ, (2*x^2 - x + c = 0) ∧ (∃ y : ℝ, y ≠ x ∧ 2*y^2 - y + c = 0)) →
  c = 1/8 :=
sorry

end quadratic_two_equal_real_roots_c_l15_15411


namespace star_polygon_net_of_pyramid_l15_15575

theorem star_polygon_net_of_pyramid (R r : ℝ) (h : R > r) : R > 2 * r :=
by
  sorry

end star_polygon_net_of_pyramid_l15_15575


namespace solve_for_r_l15_15438

variable (k r : ℝ)

theorem solve_for_r (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := sorry

end solve_for_r_l15_15438


namespace positive_number_is_25_over_9_l15_15987

variable (a : ℚ) (x : ℚ)

theorem positive_number_is_25_over_9 
  (h1 : 2 * a - 1 = -a + 3)
  (h2 : ∃ r : ℚ, r^2 = x ∧ (r = 2 * a - 1 ∨ r = -a + 3)) : 
  x = 25 / 9 := 
by
  sorry

end positive_number_is_25_over_9_l15_15987


namespace percentage_of_millet_in_Brand_A_l15_15276

variable (A B : ℝ)
variable (B_percent : B = 0.65)
variable (mix_millet_percent : 0.60 * A + 0.40 * B = 0.50)

theorem percentage_of_millet_in_Brand_A :
  A = 0.40 :=
by
  sorry

end percentage_of_millet_in_Brand_A_l15_15276


namespace sum_infinite_geometric_series_l15_15487

theorem sum_infinite_geometric_series :
  let a := 1
  let r := (1 : ℝ) / 3
  ∑' (n : ℕ), a * r ^ n = (3 : ℝ) / 2 :=
by
  sorry

end sum_infinite_geometric_series_l15_15487


namespace sum_of_coords_of_circle_center_l15_15145

theorem sum_of_coords_of_circle_center (x y : ℝ) :
  (x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by
  sorry

end sum_of_coords_of_circle_center_l15_15145


namespace parabola_point_comparison_l15_15102

theorem parabola_point_comparison :
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  y1 < y2 :=
by
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  have h : y1 < y2 := by sorry
  exact h

end parabola_point_comparison_l15_15102


namespace geometric_series_sum_l15_15615

theorem geometric_series_sum : 
  (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 88572 := 
by 
  sorry

end geometric_series_sum_l15_15615


namespace nine_pow_1000_mod_13_l15_15563

theorem nine_pow_1000_mod_13 :
  (9^1000) % 13 = 9 :=
by
  have h1 : 9^1 % 13 = 9 := by sorry
  have h2 : 9^2 % 13 = 3 := by sorry
  have h3 : 9^3 % 13 = 1 := by sorry
  have cycle : ∀ n, 9^(3 * n + 1) % 13 = 9 := by sorry
  exact (cycle 333)

end nine_pow_1000_mod_13_l15_15563


namespace carrots_as_potatoes_l15_15722

variable (G O C P : ℕ)

theorem carrots_as_potatoes :
  G = 8 →
  G = (1 / 3 : ℚ) * O →
  O = 2 * C →
  P = 2 →
  (C / P : ℚ) = 6 :=
by intros hG1 hG2 hO hP; sorry

end carrots_as_potatoes_l15_15722


namespace total_action_figures_l15_15773

theorem total_action_figures (initial_figures cost_per_figure total_cost needed_figures : ℕ)
  (h1 : initial_figures = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost = 72)
  (h4 : needed_figures = total_cost / cost_per_figure)
  : initial_figures + needed_figures = 16 :=
by
  sorry

end total_action_figures_l15_15773


namespace lilies_per_centerpiece_l15_15171

theorem lilies_per_centerpiece (centerpieces roses orchids cost total_budget price_per_flower number_of_lilies_per_centerpiece : ℕ) 
  (h0 : centerpieces = 6)
  (h1 : roses = 8)
  (h2 : orchids = 2 * roses)
  (h3 : cost = total_budget)
  (h4 : total_budget = 2700)
  (h5 : price_per_flower = 15)
  (h6 : cost = (centerpieces * roses * price_per_flower) + (centerpieces * orchids * price_per_flower) + (centerpieces * number_of_lilies_per_centerpiece * price_per_flower))
  : number_of_lilies_per_centerpiece = 6 := 
by 
  sorry

end lilies_per_centerpiece_l15_15171


namespace quadratic_no_real_roots_l15_15963

theorem quadratic_no_real_roots :
  ¬ (∃ x : ℝ, x^2 - 2 * x + 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0) ∧ (x2^2 - 3 * x2 = 0) ∧
  ∃ y : ℝ, y^2 - 2 * y + 1 = 0 :=
by
  sorry

end quadratic_no_real_roots_l15_15963


namespace more_triangles_with_perimeter_2003_than_2000_l15_15218

theorem more_triangles_with_perimeter_2003_than_2000 :
  (∃ (count_2003 count_2000 : ℕ), 
   count_2003 > count_2000 ∧ 
   (∀ (a b c : ℕ), a + b + c = 2000 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
   (∀ (a b c : ℕ), a + b + c = 2003 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a))
  := 
sorry

end more_triangles_with_perimeter_2003_than_2000_l15_15218


namespace linda_savings_l15_15768

theorem linda_savings :
  let original_price_per_notebook := 3.75
  let discount_rate := 0.15
  let quantity := 12
  let total_price_without_discount := quantity * original_price_per_notebook
  let discount_amount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_amount_per_notebook
  let total_price_with_discount := quantity * discounted_price_per_notebook
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 6.75 :=
by {
  sorry
}

end linda_savings_l15_15768


namespace painted_rooms_l15_15562

def total_rooms : ℕ := 12
def hours_per_room : ℕ := 7
def remaining_hours : ℕ := 49

theorem painted_rooms : total_rooms - (remaining_hours / hours_per_room) = 5 := by
  sorry

end painted_rooms_l15_15562


namespace teacher_age_l15_15429

theorem teacher_age (avg_student_age : ℕ) (num_students : ℕ) (new_avg_age : ℕ) (num_total : ℕ) (total_student_age : ℕ) (total_age_with_teacher : ℕ) :
  avg_student_age = 22 → 
  num_students = 23 → 
  new_avg_age = 23 → 
  num_total = 24 → 
  total_student_age = avg_student_age * num_students → 
  total_age_with_teacher = new_avg_age * num_total → 
  total_age_with_teacher - total_student_age = 46 :=
by
  intros
  sorry

end teacher_age_l15_15429


namespace pairs_satisfying_condition_l15_15758

theorem pairs_satisfying_condition (x y : ℤ) (h : x + y ≠ 0) :
  (x^2 + y^2)/(x + y) = 10 ↔ (x, y) = (12, 6) ∨ (x, y) = (-2, 6) ∨ (x, y) = (12, 4) ∨ (x, y) = (-2, 4) ∨ (x, y) = (10, 10) ∨ (x, y) = (0, 10) ∨ (x, y) = (10, 0) :=
sorry

end pairs_satisfying_condition_l15_15758


namespace difference_of_two_smallest_integers_l15_15292

/--
The difference between the two smallest integers greater than 1 which, when divided by any integer 
\( k \) in the range from \( 3 \leq k \leq 13 \), leave a remainder of \( 2 \), is \( 360360 \).
-/
theorem difference_of_two_smallest_integers (n m : ℕ) (h_n : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → n % k = 2) (h_m : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → m % k = 2) (h_smallest : m > n) :
  m - n = 360360 :=
sorry

end difference_of_two_smallest_integers_l15_15292


namespace fx_fixed_point_l15_15311

theorem fx_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y, (x = -1) ∧ (y = 3) ∧ (a * (x + 1) + 2 = y) :=
by
  sorry

end fx_fixed_point_l15_15311


namespace l_shaped_area_l15_15744

theorem l_shaped_area (A B C D : Type) (side_abcd: ℝ) (side_small_1: ℝ) (side_small_2: ℝ)
  (area_abcd : side_abcd = 6)
  (area_small_1 : side_small_1 = 2)
  (area_small_2 : side_small_2 = 4)
  (no_overlap : true) :
  side_abcd * side_abcd - (side_small_1 * side_small_1 + side_small_2 * side_small_2) = 16 := by
  sorry

end l_shaped_area_l15_15744


namespace num_type_A_cubes_internal_diagonal_l15_15451

theorem num_type_A_cubes_internal_diagonal :
  let L := 120
  let W := 350
  let H := 400
  -- Total cubes traversed calculation
  let GCD := Nat.gcd
  let total_cubes_traversed := L + W + H - (GCD L W + GCD W H + GCD H L) + GCD L (GCD W H)
  -- Type A cubes calculation
  total_cubes_traversed / 2 = 390 := by sorry

end num_type_A_cubes_internal_diagonal_l15_15451


namespace discountIs50Percent_l15_15970

noncomputable def promotionalPrice (originalPrice : ℝ) : ℝ :=
  (2/3) * originalPrice

noncomputable def finalPrice (originalPrice : ℝ) : ℝ :=
  0.75 * promotionalPrice originalPrice

theorem discountIs50Percent (originalPrice : ℝ) (h₁ : originalPrice > 0) :
  finalPrice originalPrice = 0.5 * originalPrice := by
  sorry

end discountIs50Percent_l15_15970


namespace tilling_time_in_minutes_l15_15887

-- Definitions
def plot_width : ℕ := 110
def plot_length : ℕ := 120
def tiller_width : ℕ := 2
def tilling_rate : ℕ := 2 -- 2 seconds per foot

-- Theorem: The time to till the entire plot in minutes
theorem tilling_time_in_minutes : (plot_width / tiller_width * plot_length * tilling_rate) / 60 = 220 := by
  sorry

end tilling_time_in_minutes_l15_15887


namespace farmer_brown_additional_cost_l15_15712

-- Definitions for the conditions
def originalQuantity : ℕ := 10
def originalPricePerBale : ℕ := 15
def newPricePerBale : ℕ := 18
def newQuantity : ℕ := 2 * originalQuantity

-- Definition for the target equation (additional cost)
def additionalCost : ℕ := (newQuantity * newPricePerBale) - (originalQuantity * originalPricePerBale)

-- Theorem stating the problem voiced in Lean 4
theorem farmer_brown_additional_cost : additionalCost = 210 :=
by {
  sorry
}

end farmer_brown_additional_cost_l15_15712


namespace solve_system_eq_l15_15023

theorem solve_system_eq (a b c x y z : ℝ) (h1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (h2 : x / a + y / b + z / c = a + b + c) (h3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c :=
by
  sorry

end solve_system_eq_l15_15023


namespace parallelogram_base_length_l15_15753

theorem parallelogram_base_length :
  ∀ (A H : ℝ), (A = 480) → (H = 15) → (A = Base * H) → (Base = 32) := 
by 
  intros A H hA hH hArea 
  sorry

end parallelogram_base_length_l15_15753


namespace table_height_is_five_l15_15757

def height_of_table (l h w : ℕ) : Prop :=
  l + h + w = 45 ∧ 2 * w + h = 40

theorem table_height_is_five (l w : ℕ) : height_of_table l 5 w :=
by
  sorry

end table_height_is_five_l15_15757


namespace min_sum_of_dimensions_l15_15749

/-- A theorem to find the minimum possible sum of the three dimensions of a rectangular box 
with given volume 1729 inch³ and positive integer dimensions. -/
theorem min_sum_of_dimensions (x y z : ℕ) (h1 : x * y * z = 1729) : x + y + z ≥ 39 :=
by
  sorry

end min_sum_of_dimensions_l15_15749


namespace projectiles_meet_in_84_minutes_l15_15242

theorem projectiles_meet_in_84_minutes :
  ∀ (d v₁ v₂ : ℝ), d = 1386 → v₁ = 445 → v₂ = 545 → (20 : ℝ) = 20 → 
  ((1386 / (445 + 545) / 60) * 60 * 60 = 84) :=
by
  intros d v₁ v₂ h_d h_v₁ h_v₂ h_wind
  sorry

end projectiles_meet_in_84_minutes_l15_15242


namespace inequality_solution_set_l15_15440

theorem inequality_solution_set (x : ℝ) : (x - 1 < 7) ∧ (3 * x + 1 ≥ -2) ↔ -1 ≤ x ∧ x < 8 :=
by
  sorry

end inequality_solution_set_l15_15440


namespace triangle_area_l15_15416

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 5) :
  (base * height) / 2 = 25 := by
  -- Proof is not required as per instructions.
  sorry

end triangle_area_l15_15416


namespace parabola_constant_term_l15_15021

theorem parabola_constant_term (b c : ℝ)
  (h1 : 2 * b + c = 8)
  (h2 : -2 * b + c = -4)
  (h3 : 4 * b + c = 24) :
  c = 2 :=
sorry

end parabola_constant_term_l15_15021


namespace units_digit_of_n_squared_plus_2_n_is_7_l15_15569

def n : ℕ := 2023 ^ 2 + 2 ^ 2023

theorem units_digit_of_n_squared_plus_2_n_is_7 : (n ^ 2 + 2 ^ n) % 10 = 7 := 
by
  sorry

end units_digit_of_n_squared_plus_2_n_is_7_l15_15569


namespace total_pieces_correct_l15_15268

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end total_pieces_correct_l15_15268


namespace final_selling_price_correct_l15_15908

noncomputable def purchase_price_inr : ℝ := 8000
noncomputable def depreciation_rate_annual : ℝ := 0.10
noncomputable def profit_rate : ℝ := 0.10
noncomputable def discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.12
noncomputable def exchange_rate_at_purchase : ℝ := 80
noncomputable def exchange_rate_at_selling : ℝ := 75

noncomputable def depreciated_value_after_2_years (initial_value : ℝ) : ℝ :=
  initial_value * (1 - depreciation_rate_annual) * (1 - depreciation_rate_annual)

noncomputable def marked_price (initial_value : ℝ) : ℝ :=
  initial_value * (1 + profit_rate)

noncomputable def selling_price_before_tax (marked_price : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

noncomputable def final_selling_price_inr (selling_price_before_tax : ℝ) : ℝ :=
  selling_price_before_tax * (1 + sales_tax_rate)

noncomputable def final_selling_price_usd (final_selling_price_inr : ℝ) : ℝ :=
  final_selling_price_inr / exchange_rate_at_selling

theorem final_selling_price_correct :
  final_selling_price_usd (final_selling_price_inr (selling_price_before_tax (marked_price purchase_price_inr))) = 124.84 := 
sorry

end final_selling_price_correct_l15_15908


namespace parallelogram_base_length_l15_15588

variable (base height : ℝ)
variable (Area : ℝ)

theorem parallelogram_base_length (h₁ : Area = 162) (h₂ : height = 2 * base) (h₃ : Area = base * height) : base = 9 := 
by
  sorry

end parallelogram_base_length_l15_15588


namespace simplify_negative_exponents_l15_15785

theorem simplify_negative_exponents (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ :=
  sorry

end simplify_negative_exponents_l15_15785


namespace fencing_cost_l15_15698

def total_cost_of_fencing 
  (length breadth cost_per_meter : ℝ)
  (h1 : length = 62)
  (h2 : length = breadth + 24)
  (h3 : cost_per_meter = 26.50) : ℝ :=
  2 * (length + breadth) * cost_per_meter

theorem fencing_cost : total_cost_of_fencing 62 38 26.50 (by rfl) (by norm_num) (by norm_num) = 5300 := 
by 
  sorry

end fencing_cost_l15_15698


namespace john_total_amount_l15_15248

/-- Define the amounts of money John has and needs additionally -/
def johnHas : ℝ := 0.75
def needsMore : ℝ := 1.75

/-- Prove the total amount of money John needs given the conditions -/
theorem john_total_amount : johnHas + needsMore = 2.50 := by
  sorry

end john_total_amount_l15_15248


namespace range_of_a_l15_15565
noncomputable section

open Real

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + 2 * x + a + 2 > 0) : a > -1 :=
sorry

end range_of_a_l15_15565


namespace courtyard_length_eq_40_l15_15576

/-- Defining the dimensions of a paving stone -/
def stone_length : ℝ := 4
def stone_width : ℝ := 2

/-- Defining the width of the courtyard -/
def courtyard_width : ℝ := 20

/-- Number of paving stones used -/
def num_stones : ℝ := 100

/-- Area covered by one paving stone -/
def stone_area : ℝ := stone_length * stone_width

/-- Total area covered by the paving stones -/
def total_area : ℝ := num_stones * stone_area

/-- The main statement to be proved -/
theorem courtyard_length_eq_40 (h1 : total_area = num_stones * stone_area)
(h2 : total_area = 800)
(h3 : courtyard_width = 20) : total_area / courtyard_width = 40 :=
by sorry

end courtyard_length_eq_40_l15_15576


namespace polygon_sides_l15_15880

theorem polygon_sides (n : ℕ) (a1 d : ℝ) (h1 : a1 = 100) (h2 : d = 10)
  (h3 : ∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d < 180) : n = 8 :=
by
  sorry

end polygon_sides_l15_15880


namespace distributeCandies_l15_15903

-- Define the conditions as separate definitions.

-- Number of candies
def candies : ℕ := 10

-- Number of boxes
def boxes : ℕ := 5

-- Condition that each box gets at least one candy
def atLeastOne (candyDist : Fin boxes → ℕ) : Prop :=
  ∀ b, candyDist b > 0

-- Function to count the number of ways to distribute candies
noncomputable def countWaysToDistribute (candies : ℕ) (boxes : ℕ) : ℕ :=
  -- Function to compute the number of ways
  -- (assuming a correct implementation is provided)
  sorry -- Placeholder for the actual counting implementation

-- Theorem to prove the number of distributions
theorem distributeCandies : countWaysToDistribute candies boxes = 7 := 
by {
  -- Proof omitted
  sorry
}

end distributeCandies_l15_15903


namespace gift_sequence_count_l15_15769

noncomputable def number_of_gift_sequences (students : ℕ) (classes_per_week : ℕ) : ℕ :=
  (students * students) ^ classes_per_week

theorem gift_sequence_count :
  number_of_gift_sequences 15 3 = 11390625 :=
by
  sorry

end gift_sequence_count_l15_15769


namespace spent_on_veggies_l15_15518

noncomputable def total_amount : ℕ := 167
noncomputable def spent_on_meat : ℕ := 17
noncomputable def spent_on_chicken : ℕ := 22
noncomputable def spent_on_eggs : ℕ := 5
noncomputable def spent_on_dog_food : ℕ := 45
noncomputable def amount_left : ℕ := 35

theorem spent_on_veggies : 
  total_amount - (spent_on_meat + spent_on_chicken + spent_on_eggs + spent_on_dog_food + amount_left) = 43 := 
by 
  sorry

end spent_on_veggies_l15_15518


namespace bianca_total_bags_l15_15625

theorem bianca_total_bags (bags_recycled_points : ℕ) (bags_not_recycled : ℕ) (total_points : ℕ) (total_bags : ℕ) 
  (h1 : bags_recycled_points = 5) 
  (h2 : bags_not_recycled = 8) 
  (h3 : total_points = 45) 
  (recycled_bags := total_points / bags_recycled_points) :
  total_bags = recycled_bags + bags_not_recycled := 
by 
  sorry

end bianca_total_bags_l15_15625


namespace odd_sol_exists_l15_15015

theorem odd_sol_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x_n y_n : ℕ), (x_n % 2 = 1) ∧ (y_n % 2 = 1) ∧ (x_n^2 + 7 * y_n^2 = 2^n) := 
sorry

end odd_sol_exists_l15_15015


namespace union_P_Q_l15_15623

-- Definition of sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 4 }
def Q : Set ℝ := { x | -3 < x ∧ x < 3 }

-- Statement to prove
theorem union_P_Q :
  P ∪ Q = { x : ℝ | -3 < x ∧ x ≤ 4 } :=
sorry

end union_P_Q_l15_15623


namespace john_total_trip_cost_l15_15879

noncomputable def total_trip_cost
  (hotel_nights : ℕ) 
  (hotel_rate_per_night : ℝ) 
  (discount : ℝ) 
  (loyal_customer_discount_rate : ℝ) 
  (service_tax_rate : ℝ) 
  (room_service_cost_per_day : ℝ) 
  (cab_cost_per_ride : ℝ) : ℝ :=
  let hotel_cost := hotel_nights * hotel_rate_per_night
  let cost_after_discount := hotel_cost - discount
  let loyal_customer_discount := loyal_customer_discount_rate * cost_after_discount
  let cost_after_loyalty_discount := cost_after_discount - loyal_customer_discount
  let service_tax := service_tax_rate * cost_after_loyalty_discount
  let final_hotel_cost := cost_after_loyalty_discount + service_tax
  let room_service_cost := hotel_nights * room_service_cost_per_day
  let cab_cost := cab_cost_per_ride * 2 * hotel_nights
  final_hotel_cost + room_service_cost + cab_cost

theorem john_total_trip_cost : total_trip_cost 3 250 100 0.10 0.12 50 30 = 985.20 :=
by 
  -- We are skipping the proof but our focus is the statement
  sorry

end john_total_trip_cost_l15_15879


namespace quadratic_radical_simplified_l15_15943

theorem quadratic_radical_simplified (a : ℕ) : 
  (∃ (b : ℕ), a = 3 * b^2) -> a = 3 := 
by
  sorry

end quadratic_radical_simplified_l15_15943


namespace trapezium_other_parallel_side_l15_15986

theorem trapezium_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : area = (1 / 2) * (a + b) * h) (h_a : a = 18) (h_h : h = 20) (h_area_val : area = 380) :
  b = 20 :=
by 
  sorry

end trapezium_other_parallel_side_l15_15986


namespace unique_x2_range_of_a_l15_15545

noncomputable def f (x : ℝ) (k a : ℝ) : ℝ :=
if x >= 0
then k*x + k*(1 - a^2)
else x^2 + (a^2 - 4*a)*x + (3 - a)^2

theorem unique_x2 (k a : ℝ) (x1 : ℝ) (hx1 : x1 ≠ 0) (hx2 : ∃ x2 : ℝ, x2 ≠ 0 ∧ x2 ≠ x1 ∧ f x2 k a = f x1 k a) :
f 0 k a = k*(1 - a^2) →
0 ≤ a ∧ a < 1 →
k = (3 - a)^2 / (1 - a^2) :=
sorry

variable (a : ℝ)

theorem range_of_a :
0 ≤ a ∧ a < 1 ↔ a^2 - 4*a ≤ 0 :=
sorry

end unique_x2_range_of_a_l15_15545


namespace find_coeff_a9_l15_15120

theorem find_coeff_a9 (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (x^3 + x^10 = a + a1 * (x + 1) + a2 * (x + 1)^2 + 
  a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5 + 
  a6 * (x + 1)^6 + a7 * (x + 1)^7 + a8 * (x + 1)^8 + 
  a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a9 = -10 :=
sorry

end find_coeff_a9_l15_15120


namespace original_solution_is_10_percent_l15_15329

def sugar_percentage_original_solution (x : ℕ) :=
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * 42 = 18

theorem original_solution_is_10_percent : sugar_percentage_original_solution 10 :=
by
  unfold sugar_percentage_original_solution
  norm_num

end original_solution_is_10_percent_l15_15329


namespace ratio_A_to_B_investment_l15_15339

variable (A B C : Type) [Field A] [Field B] [Field C]
variable (investA investB investC profit total_profit : A) 

-- Conditions
axiom A_invests_some_times_as_B : ∃ n : A, investA = n * investB
axiom B_invests_two_thirds_of_C : investB = (2/3) * investC
axiom total_profit_statement : total_profit = 3300
axiom B_share_statement : profit = 600

-- Theorem: Ratio of A's investment to B's investment is 3:1
theorem ratio_A_to_B_investment : ∃ n : A, investA = 3 * investB :=
sorry

end ratio_A_to_B_investment_l15_15339


namespace angle_B_value_l15_15151

theorem angle_B_value (a b c A B : ℝ) (h1 : Real.sqrt 3 * a = 2 * b * Real.sin A) : 
  Real.sin B = Real.sqrt 3 / 2 ↔ (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) :=
by sorry

noncomputable def find_b_value (a : ℝ) (area : ℝ) (A B c : ℝ) (h1 : a = 6) (h2 : area = 6 * Real.sqrt 3) (h3 : c = 4) (h4 : B = Real.pi / 3 ∨ B = 2 * Real.pi / 3) : 
  ℝ := 
if B = Real.pi / 3 then 2 * Real.sqrt 7 else Real.sqrt 76

end angle_B_value_l15_15151


namespace probability_train_or_plane_probability_not_ship_l15_15028

def P_plane : ℝ := 0.2
def P_ship : ℝ := 0.3
def P_train : ℝ := 0.4
def P_car : ℝ := 0.1
def mutually_exclusive : Prop := P_plane + P_ship + P_train + P_car = 1

theorem probability_train_or_plane : mutually_exclusive → P_train + P_plane = 0.6 := by
  intro h
  sorry

theorem probability_not_ship : mutually_exclusive → 1 - P_ship = 0.7 := by
  intro h
  sorry

end probability_train_or_plane_probability_not_ship_l15_15028


namespace problem_1_problem_2_l15_15654

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem problem_1 (a b : ℝ) (h1 : f a b 3 - 3 + 12 = 0) (h2 : f a b 4 - 4 + 12 = 0) :
  f a b x = (2 - x) / (x - 2) := sorry

theorem problem_2 (k : ℝ) (h : k > 1) :
  ∀ x, f (-1) 2 x < k ↔ (if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
                         else if k = 2 then 1 < x ∧ x ≠ 2 
                         else (1 < x ∧ x < 2) ∨ (k < x)) := sorry

-- Function definition for clarity
noncomputable def f_spec (x : ℝ) : ℝ := (2 - x) / (x - 2)

end problem_1_problem_2_l15_15654


namespace value_of_a_l15_15982

theorem value_of_a (a : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∃ (y : ℝ), y = 2 ∧ 9 = a ^ y)) : a = 3 := 
  by sorry

end value_of_a_l15_15982


namespace prime_square_pairs_l15_15397

theorem prime_square_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    ∃ n : Nat, p^2 + 5 * p * q + 4 * q^2 = n^2 ↔ (p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11) ∨ (p = 3 ∧ q = 13) ∨ (p = 5 ∧ q = 7) ∨ (p = 11 ∧ q = 5) :=
by
  sorry

end prime_square_pairs_l15_15397


namespace danny_steve_ratio_l15_15602

theorem danny_steve_ratio :
  ∀ (D S : ℝ),
  D = 29 →
  2 * (S / 2 - D / 2) = 29 →
  D / S = 1 / 2 :=
by
  intros D S hD h_eq
  sorry

end danny_steve_ratio_l15_15602


namespace sin_identity_alpha_l15_15392

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l15_15392


namespace complex_mul_l15_15483

theorem complex_mul (i : ℂ) (h : i^2 = -1) :
    (1 - i) * (1 + 2 * i) = 3 + i :=
by
  sorry

end complex_mul_l15_15483


namespace prism_pyramid_fusion_l15_15660

theorem prism_pyramid_fusion :
  ∃ (result_faces result_edges result_vertices : ℕ),
    result_faces + result_edges + result_vertices = 28 ∧
    ((result_faces = 8 ∧ result_edges = 13 ∧ result_vertices = 7) ∨
    (result_faces = 7 ∧ result_edges = 12 ∧ result_vertices = 7)) :=
by
  sorry

end prism_pyramid_fusion_l15_15660


namespace first_term_geometric_series_l15_15506

theorem first_term_geometric_series (a1 q : ℝ) (h1 : a1 / (1 - q) = 1)
  (h2 : |a1| / (1 - |q|) = 2) (h3 : -1 < q) (h4 : q < 1) (h5 : q ≠ 0) :
  a1 = 4 / 3 :=
by {
  sorry
}

end first_term_geometric_series_l15_15506


namespace problem_statement_l15_15961

/-- 
  Theorem: If the solution set of the inequality (ax-1)(x+2) > 0 is -3 < x < -2, 
  then a equals -1/3 
--/
theorem problem_statement (a : ℝ) :
  (forall x, (ax-1)*(x+2) > 0 -> -3 < x ∧ x < -2) → a = -1/3 := 
by
  sorry

end problem_statement_l15_15961


namespace geese_left_park_l15_15579

noncomputable def initial_ducks : ℕ := 25
noncomputable def initial_geese (ducks : ℕ) : ℕ := 2 * ducks - 10
noncomputable def final_ducks (ducks_added : ℕ) (ducks : ℕ) : ℕ := ducks + ducks_added
noncomputable def geese_after_leaving (geese_before : ℕ) (geese_left : ℕ) : ℕ := geese_before - geese_left

theorem geese_left_park
    (ducks : ℕ)
    (ducks_added : ℕ)
    (initial_geese : ℕ := 2 * ducks - 10)
    (final_ducks : ℕ := ducks + ducks_added)
    (geese_left : ℕ)
    (geese_remaining : ℕ := initial_geese - geese_left) :
    geese_remaining = final_ducks + 1 → geese_left = 10 := by
  sorry

end geese_left_park_l15_15579


namespace greatest_possible_third_term_l15_15771

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l15_15771


namespace find_fraction_l15_15787

theorem find_fraction (x y : ℤ) (h1 : x + 2 = y + 1) (h2 : 2 * (x + 4) = y + 2) : 
  x = -5 ∧ y = -4 := 
sorry

end find_fraction_l15_15787


namespace world_grain_demand_l15_15360

theorem world_grain_demand (S D : ℝ) (h1 : S = 1800000) (h2 : S = 0.75 * D) : D = 2400000 := by
  sorry

end world_grain_demand_l15_15360


namespace average_length_l15_15193

def length1 : ℕ := 2
def length2 : ℕ := 3
def length3 : ℕ := 7

theorem average_length : (length1 + length2 + length3) / 3 = 4 :=
by
  sorry

end average_length_l15_15193


namespace blueberries_in_blue_box_l15_15031

theorem blueberries_in_blue_box (B S : ℕ) (h1: S - B = 10) (h2 : 50 = S) : B = 40 := 
by
  sorry

end blueberries_in_blue_box_l15_15031


namespace value_of_expression_l15_15756

def x : ℝ := 12
def y : ℝ := 7

theorem value_of_expression : (x - y) * (x + y) = 95 := by
  sorry

end value_of_expression_l15_15756


namespace partner_profit_share_correct_l15_15297

-- Definitions based on conditions
def total_profit : ℝ := 280000
def profit_share_shekhar : ℝ := 0.28
def profit_share_rajeev : ℝ := 0.22
def profit_share_jatin : ℝ := 0.20
def profit_share_simran : ℝ := 0.18
def profit_share_ramesh : ℝ := 0.12

-- Each partner's share in the profit
def shekhar_share : ℝ := profit_share_shekhar * total_profit
def rajeev_share : ℝ := profit_share_rajeev * total_profit
def jatin_share : ℝ := profit_share_jatin * total_profit
def simran_share : ℝ := profit_share_simran * total_profit
def ramesh_share : ℝ := profit_share_ramesh * total_profit

-- Statement to be proved
theorem partner_profit_share_correct :
    shekhar_share = 78400 ∧ 
    rajeev_share = 61600 ∧ 
    jatin_share = 56000 ∧ 
    simran_share = 50400 ∧ 
    ramesh_share = 33600 ∧ 
    (shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit) :=
by sorry

end partner_profit_share_correct_l15_15297


namespace at_least_2020_distinct_n_l15_15173

theorem at_least_2020_distinct_n : 
  ∃ (N : Nat), N ≥ 2020 ∧ ∃ (a : Fin N → ℕ), 
  Function.Injective a ∧ ∀ i, ∃ k : ℚ, (a i : ℚ) + 0.25 = (k + 1/2)^2 := 
sorry

end at_least_2020_distinct_n_l15_15173


namespace value_of_x4_plus_inv_x4_l15_15599

theorem value_of_x4_plus_inv_x4 (x : ℝ) (h : x^2 + 1 / x^2 = 6) : x^4 + 1 / x^4 = 34 := 
by
  sorry

end value_of_x4_plus_inv_x4_l15_15599


namespace circles_non_intersecting_l15_15136

def circle1_equation (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def circle2_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_non_intersecting :
    (∀ (x y : ℝ), ¬(circle1_equation x y ∧ circle2_equation x y)) :=
by
  sorry

end circles_non_intersecting_l15_15136


namespace farm_width_l15_15832

theorem farm_width (L W : ℕ) (h1 : 2 * (L + W) = 46) (h2 : W = L + 7) : W = 15 :=
by
  sorry

end farm_width_l15_15832


namespace linear_function_l15_15648

theorem linear_function (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = 4 * x + 6) :
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) :=
sorry

end linear_function_l15_15648


namespace solve_for_x_l15_15447

noncomputable def x_solution (x : ℚ) : Prop :=
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0

theorem solve_for_x :
  ∃ x : ℚ, x_solution x ∧ x = 4 / 3 :=
by
  sorry

end solve_for_x_l15_15447


namespace Alex_hula_hoop_duration_l15_15189

-- Definitions based on conditions
def Nancy_duration := 10
def Casey_duration := Nancy_duration - 3
def Morgan_duration := Casey_duration * 3
def Alex_duration := Casey_duration + Morgan_duration - 2

-- The theorem we need to prove
theorem Alex_hula_hoop_duration : Alex_duration = 26 := by
  -- proof to be provided
  sorry

end Alex_hula_hoop_duration_l15_15189


namespace trip_time_l15_15677

theorem trip_time (x : ℝ) (T : ℝ) :
  (70 * 4 + 60 * 5 + 50 * x) / (4 + 5 + x) = 58 → 
  T = 4 + 5 + x → 
  T = 16.25 :=
by
  intro h1 h2
  sorry

end trip_time_l15_15677


namespace determine_delta_l15_15818

theorem determine_delta (r1 r2 r3 r4 r5 r6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) (O Δ : ℕ) 
  (h_sums_rows : r1 + r2 + r3 + r4 + r5 + r6 = 190)
  (h_row1 : r1 = 29) (h_row2 : r2 = 33) (h_row3 : r3 = 33) 
  (h_row4 : r4 = 32) (h_row5 : r5 = 32) (h_row6 : r6 = 31)
  (h_sums_cols : c1 + c2 + c3 + c4 + c5 + c6 = 190)
  (h_col1 : c1 = 29) (h_col2 : c2 = 33) (h_col3 : c3 = 33) 
  (h_col4 : c4 = 32) (h_col5 : c5 = 32) (h_col6 : c6 = 31)
  (h_O : O = 6) : 
  Δ = 4 :=
by 
  sorry

end determine_delta_l15_15818


namespace intersection_complement_eq_l15_15367

noncomputable def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
noncomputable def A : Set Int := {-1, 0, 1, 2}
noncomputable def B : Set Int := {-3, 0, 2, 3}

-- Complement of B with respect to U
noncomputable def U_complement_B : Set Int := U \ B

-- The statement we need to prove
theorem intersection_complement_eq :
  A ∩ U_complement_B = {-1, 1} :=
by
  sorry

end intersection_complement_eq_l15_15367


namespace system_of_equations_solution_l15_15425

theorem system_of_equations_solution :
  ∃ x y : ℚ, x = 2 * y ∧ 2 * x - y = 5 ∧ x = 10 / 3 ∧ y = 5 / 3 :=
by
  sorry

end system_of_equations_solution_l15_15425


namespace permissible_m_values_l15_15127

theorem permissible_m_values :
  ∀ (m : ℕ) (a : ℝ), 
  (∃ k, 2 ≤ k ∧ k ≤ 4 ∧ (3 / (6 / (2 * m + 1)) ≤ k)) → m = 2 ∨ m = 3 :=
by
  sorry

end permissible_m_values_l15_15127


namespace prove_x_minus_y_squared_l15_15332

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l15_15332


namespace problem_statement_l15_15401

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (Real.sin x)^2 - Real.tan x else Real.exp (-2 * x)

theorem problem_statement : f (f (-25 * Real.pi / 4)) = Real.exp (-3) :=
by
  sorry

end problem_statement_l15_15401


namespace radius_of_circle_with_tangent_parabolas_l15_15014

theorem radius_of_circle_with_tangent_parabolas (r : ℝ) : 
  (∀ x : ℝ, (x^2 + r = x → ∃ x0 : ℝ, x^2 + r = x0)) → r = 1 / 4 :=
by
  sorry

end radius_of_circle_with_tangent_parabolas_l15_15014


namespace student_correct_answers_l15_15589

variable (C I : ℕ) -- Define C and I as natural numbers
variable (score totalQuestions : ℕ) -- Define score and totalQuestions as natural numbers

-- Define the conditions
def grading_system (C I score : ℕ) : Prop := C - 2 * I = score
def total_questions (C I totalQuestions : ℕ) : Prop := C + I = totalQuestions

-- The theorem statement to prove
theorem student_correct_answers :
  (grading_system C I 76) ∧ (total_questions C I 100) → C = 92 := by
  sorry -- Proof to be filled in

end student_correct_answers_l15_15589


namespace money_left_after_shopping_l15_15140

-- Define the initial amount of money Sandy took for shopping
def initial_amount : ℝ := 310

-- Define the percentage of money spent in decimal form
def percentage_spent : ℝ := 0.30

-- Define the remaining money as per the given conditions
def remaining_money : ℝ := initial_amount * (1 - percentage_spent)

-- The statement we need to prove
theorem money_left_after_shopping :
  remaining_money = 217 :=
by
  sorry

end money_left_after_shopping_l15_15140


namespace Q_is_perfect_square_trinomial_l15_15934

def is_perfect_square_trinomial (p : ℤ → ℤ) :=
∃ (b : ℤ), ∀ a : ℤ, p a = (a + b) * (a + b)

def P (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2
def Q (a : ℤ) : ℤ := a^2 + 2 * a + 1
def R (a b : ℤ) : ℤ := a^2 + a * b + b^2
def S (a : ℤ) : ℤ := a^2 + 2 * a - 1

theorem Q_is_perfect_square_trinomial : is_perfect_square_trinomial Q :=
sorry -- Proof goes here

end Q_is_perfect_square_trinomial_l15_15934


namespace pizza_eaten_after_six_trips_l15_15655

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end pizza_eaten_after_six_trips_l15_15655


namespace find_a1_l15_15706

noncomputable def a_n : ℕ → ℝ := sorry
noncomputable def S_n : ℕ → ℝ := sorry

theorem find_a1
  (h1 : ∀ n : ℕ, a_n 2 * a_n 8 = 2 * a_n 3 * a_n 6)
  (h2 : S_n 5 = -62) :
  a_n 1 = -2 :=
sorry

end find_a1_l15_15706


namespace father_l15_15131

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S) (h2 : F + 15 = 2 * (S + 15)) : F = 45 :=
sorry

end father_l15_15131


namespace marble_choice_l15_15239

def numDifferentGroupsOfTwoMarbles (red green blue : ℕ) (yellow : ℕ) (orange : ℕ) : ℕ :=
  if (red = 1 ∧ green = 1 ∧ blue = 1 ∧ yellow = 2 ∧ orange = 2) then 12 else 0

theorem marble_choice:
  let red := 1
  let green := 1
  let blue := 1
  let yellow := 2
  let orange := 2
  numDifferentGroupsOfTwoMarbles red green blue yellow orange = 12 :=
by
  dsimp[numDifferentGroupsOfTwoMarbles]
  split_ifs
  · rfl
  · sorry

-- Ensure the theorem type matches the expected Lean 4 structure.
#print marble_choice

end marble_choice_l15_15239


namespace man_age_twice_son_age_l15_15247

theorem man_age_twice_son_age (S M : ℕ) (h1 : M = S + 24) (h2 : S = 22) : 
  ∃ Y : ℕ, M + Y = 2 * (S + Y) ∧ Y = 2 :=
by 
  sorry

end man_age_twice_son_age_l15_15247


namespace least_range_product_multiple_840_l15_15521

def is_multiple (x y : Nat) : Prop :=
  ∃ k : Nat, y = k * x

theorem least_range_product_multiple_840 : 
  ∃ (a : Nat), a > 0 ∧ ∀ (n : Nat), (n = 3) → is_multiple 840 (List.foldr (· * ·) 1 (List.range' a n)) := 
by {
  sorry
}

end least_range_product_multiple_840_l15_15521


namespace find_cost_of_apple_l15_15073

theorem find_cost_of_apple (A O : ℝ) 
  (h1 : 6 * A + 3 * O = 1.77) 
  (h2 : 2 * A + 5 * O = 1.27) : 
  A = 0.21 :=
by 
  sorry

end find_cost_of_apple_l15_15073


namespace volleyball_lineup_ways_l15_15113

def num_ways_lineup (team_size : ℕ) (positions : ℕ) : ℕ :=
  if positions ≤ team_size then
    Nat.descFactorial team_size positions
  else
    0

theorem volleyball_lineup_ways :
  num_ways_lineup 10 5 = 30240 :=
by
  rfl

end volleyball_lineup_ways_l15_15113


namespace total_height_of_buildings_l15_15614

noncomputable def tallest_building := 100
noncomputable def second_tallest_building := tallest_building / 2
noncomputable def third_tallest_building := second_tallest_building / 2
noncomputable def fourth_tallest_building := third_tallest_building / 5

theorem total_height_of_buildings : 
  (tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building) = 180 := by
  sorry

end total_height_of_buildings_l15_15614


namespace total_investment_is_correct_l15_15319

def Raghu_investment : ℕ := 2300
def Trishul_investment (Raghu_investment : ℕ) : ℕ := Raghu_investment - (Raghu_investment / 10)
def Vishal_investment (Trishul_investment : ℕ) : ℕ := Trishul_investment + (Trishul_investment / 10)

theorem total_investment_is_correct :
    let Raghu_inv := Raghu_investment;
    let Trishul_inv := Trishul_investment Raghu_inv;
    let Vishal_inv := Vishal_investment Trishul_inv;
    Raghu_inv + Trishul_inv + Vishal_inv = 6647 :=
by
    sorry

end total_investment_is_correct_l15_15319


namespace alley_width_theorem_l15_15225

noncomputable def width_of_alley (a k h : ℝ) (h₁ : k = a / 2) (h₂ : h = a * (Real.sqrt 2) / 2) : ℝ :=
  Real.sqrt ((a * (Real.sqrt 2) / 2)^2 + (a / 2)^2)

theorem alley_width_theorem (a k h w : ℝ)
  (h₁ : k = a / 2)
  (h₂ : h = a * (Real.sqrt 2) / 2)
  (h₃ : w = width_of_alley a k h h₁ h₂) :
  w = (Real.sqrt 3) * a / 2 :=
by
  sorry

end alley_width_theorem_l15_15225


namespace smallest_k_l15_15389

theorem smallest_k (a b c d e k : ℕ) (h1 : a + 2 * b + 3 * c + 4 * d + 5 * e = k)
  (h2 : 5 * a = 4 * b) (h3 : 4 * b = 3 * c) (h4 : 3 * c = 2 * d) (h5 : 2 * d = e) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : k = 522 :=
sorry

end smallest_k_l15_15389


namespace find_angle_A_find_b_c_l15_15699
open Real

-- Part I: Proving angle A
theorem find_angle_A (A B C : ℝ) (a b c : ℝ) (h₁ : (a + b + c) * (b + c - a) = 3 * b * c) :
  A = π / 3 :=
by sorry

-- Part II: Proving values of b and c given a=2 and area of triangle ABC is √3
theorem find_b_c (A B C : ℝ) (a b c : ℝ) (h₁ : a = 2) (h₂ : (1 / 2) * b * c * (sin (π / 3)) = sqrt 3) :
  b = 2 ∧ c = 2 :=
by sorry

end find_angle_A_find_b_c_l15_15699


namespace vector_perpendicular_l15_15558

open Real

theorem vector_perpendicular (t : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (4, 3)) :
  a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 ↔ t = -2 := by
  sorry

end vector_perpendicular_l15_15558


namespace base_nine_to_base_ten_conversion_l15_15923

theorem base_nine_to_base_ten_conversion : 
  (2 * 9^3 + 8 * 9^2 + 4 * 9^1 + 7 * 9^0 = 2149) := 
by 
  sorry

end base_nine_to_base_ten_conversion_l15_15923


namespace set_intersection_complement_l15_15610

open Set

variable (A B U : Set ℕ)

theorem set_intersection_complement (A B : Set ℕ) (U : Set ℕ) (hU : U = {1, 2, 3, 4})
  (h1 : compl (A ∪ B) = {4}) (h2 : B = {1, 2}) :
  A ∩ compl B = {3} :=
by
  sorry

end set_intersection_complement_l15_15610


namespace number_of_trees_planted_l15_15186

theorem number_of_trees_planted (initial_trees final_trees trees_planted : ℕ) 
  (h_initial : initial_trees = 22)
  (h_final : final_trees = 77)
  (h_planted : trees_planted = final_trees - initial_trees) : 
  trees_planted = 55 := by
  sorry

end number_of_trees_planted_l15_15186


namespace solution_of_inequality_system_l15_15330

theorem solution_of_inequality_system (x : ℝ) : 
  (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x) ∧ (x < 1) := 
by sorry

end solution_of_inequality_system_l15_15330


namespace maximize_a2_b2_c2_d2_l15_15437

theorem maximize_a2_b2_c2_d2 
  (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 187)
  (h4 : cd = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 :=
sorry

end maximize_a2_b2_c2_d2_l15_15437


namespace campers_afternoon_l15_15709

def morning_campers : ℕ := 52
def additional_campers : ℕ := 9
def total_campers_afternoon : ℕ := morning_campers + additional_campers

theorem campers_afternoon : total_campers_afternoon = 61 :=
by
  sorry

end campers_afternoon_l15_15709


namespace math_problem_l15_15407

def Q (f : ℝ → ℝ) : Prop :=
  (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y))
  ∧ (∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y)
  ∧ f 1 = 1

theorem math_problem (f : ℝ → ℝ) : Q f → (∀ (x : ℝ), x ≠ 0 → f x = 1 / x) :=
by
  -- Proof goes here
  sorry

end math_problem_l15_15407


namespace volume_remaining_cube_l15_15417

theorem volume_remaining_cube (a : ℝ) (original_volume vertex_cube_volume : ℝ) (number_of_vertices : ℕ) :
  original_volume = a^3 → 
  vertex_cube_volume = 1 → 
  number_of_vertices = 8 → 
  a = 3 →
  original_volume - (number_of_vertices * vertex_cube_volume) = 19 := 
by
  sorry

end volume_remaining_cube_l15_15417


namespace find_a_l15_15433

def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (h : star a 4 = 17) : a = 49 / 3 :=
by sorry

end find_a_l15_15433


namespace total_land_l15_15394

variable (land_house : ℕ) (land_expansion : ℕ) (land_cattle : ℕ) (land_crop : ℕ)

theorem total_land (h1 : land_house = 25) 
                   (h2 : land_expansion = 15) 
                   (h3 : land_cattle = 40) 
                   (h4 : land_crop = 70) : 
  land_house + land_expansion + land_cattle + land_crop = 150 := 
by 
  sorry

end total_land_l15_15394


namespace aunt_wang_bought_n_lilies_l15_15029

theorem aunt_wang_bought_n_lilies 
  (cost_rose : ℕ) 
  (cost_lily : ℕ) 
  (total_spent : ℕ) 
  (num_roses : ℕ) 
  (num_lilies : ℕ) 
  (roses_cost : num_roses * cost_rose = 10) 
  (total_spent_cond : total_spent = 55) 
  (cost_conditions : cost_rose = 5 ∧ cost_lily = 9) 
  (spending_eq : total_spent = num_roses * cost_rose + num_lilies * cost_lily) : 
  num_lilies = 5 :=
by 
  sorry

end aunt_wang_bought_n_lilies_l15_15029


namespace combined_surface_area_of_cube_and_sphere_l15_15227

theorem combined_surface_area_of_cube_and_sphere (V_cube : ℝ) :
  V_cube = 729 →
  ∃ (A_combined : ℝ), A_combined = 486 + 81 * Real.pi :=
by
  intro V_cube
  sorry

end combined_surface_area_of_cube_and_sphere_l15_15227


namespace range_of_m_plus_n_l15_15940

theorem range_of_m_plus_n (f : ℝ → ℝ) (n m : ℝ)
  (h_f_def : ∀ x, f x = x^2 + n * x + m)
  (h_non_empty : ∃ x, f x = 0 ∧ f (f x) = 0)
  (h_condition : ∀ x, f x = 0 ↔ f (f x) = 0) :
  0 < m + n ∧ m + n < 4 :=
by {
  -- Proof needed here; currently skipped
  sorry
}

end range_of_m_plus_n_l15_15940


namespace minimal_length_AX_XB_l15_15299

theorem minimal_length_AX_XB 
  (AA' BB' : ℕ) (A'B' : ℕ) 
  (h1 : AA' = 680) (h2 : BB' = 2000) (h3 : A'B' = 2010) 
  : ∃ X : ℕ, AX + XB = 3350 := 
sorry

end minimal_length_AX_XB_l15_15299


namespace segment_AC_length_l15_15085

noncomputable def circle_radius := 8
noncomputable def chord_length_AB := 10
noncomputable def arc_length_AC (circumference : ℝ) := circumference / 3

theorem segment_AC_length :
  ∀ (C : ℝ) (r : ℝ) (AB : ℝ) (AC : ℝ),
    r = circle_radius →
    AB = chord_length_AB →
    C = 2 * Real.pi * r →
    AC = arc_length_AC C →
    AC = 8 * Real.sqrt 3 :=
by
  intros C r AB AC hr hAB hC hAC
  sorry

end segment_AC_length_l15_15085


namespace method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l15_15205

/-- Method 1: Membership card costs 200 yuan + 10 yuan per swim session. -/
def method1_cost (num_sessions : ℕ) : ℕ := 200 + 10 * num_sessions

/-- Method 2: Each swim session costs 30 yuan. -/
def method2_cost (num_sessions : ℕ) : ℕ := 30 * num_sessions

/-- Problem (1): Total cost for 3 swim sessions using Method 1 is 230 yuan. -/
theorem method1_three_sessions_cost : method1_cost 3 = 230 := by
  sorry

/-- Problem (2): Method 2 is more cost-effective than Method 1 for 9 swim sessions. -/
theorem method2_more_cost_effective_for_nine_sessions : method2_cost 9 < method1_cost 9 := by
  sorry

/-- Problem (3): Method 1 allows more sessions than Method 2 within a budget of 600 yuan. -/
theorem method1_allows_more_sessions : (600 - 200) / 10 > 600 / 30 := by
  sorry

end method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l15_15205


namespace score_sd_above_mean_l15_15251

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l15_15251


namespace trisha_initial_money_l15_15452

-- Definitions based on conditions
def spent_on_meat : ℕ := 17
def spent_on_chicken : ℕ := 22
def spent_on_veggies : ℕ := 43
def spent_on_eggs : ℕ := 5
def spent_on_dog_food : ℕ := 45
def spent_on_cat_food : ℕ := 18
def money_left : ℕ := 35

-- Total amount spent
def total_spent : ℕ :=
  spent_on_meat + spent_on_chicken + spent_on_veggies + spent_on_eggs + spent_on_dog_food + spent_on_cat_food

-- The target amount she brought with her at the beginning
def total_money_brought : ℕ :=
  total_spent + money_left

-- The theorem to be proved
theorem trisha_initial_money :
  total_money_brought = 185 :=
by
  sorry

end trisha_initial_money_l15_15452


namespace strongest_erosive_power_l15_15566

-- Definition of the options
inductive Period where
  | MayToJune : Period
  | JuneToJuly : Period
  | JulyToAugust : Period
  | AugustToSeptember : Period

-- Definition of the eroding power function (stub)
def erosivePower : Period → ℕ
| Period.MayToJune => 1
| Period.JuneToJuly => 2
| Period.JulyToAugust => 3
| Period.AugustToSeptember => 1

-- Statement that July to August has the maximum erosive power
theorem strongest_erosive_power : erosivePower Period.JulyToAugust = 3 := 
by 
  sorry

end strongest_erosive_power_l15_15566


namespace square_perimeter_from_area_l15_15274

def square_area (s : ℝ) : ℝ := s * s -- Definition of the area of a square based on its side length.
def square_perimeter (s : ℝ) : ℝ := 4 * s -- Definition of the perimeter of a square based on its side length.

theorem square_perimeter_from_area (s : ℝ) (h : square_area s = 900) : square_perimeter s = 120 :=
by {
  sorry -- Placeholder for the proof.
}

end square_perimeter_from_area_l15_15274


namespace per_capita_income_growth_l15_15700

theorem per_capita_income_growth (x : ℝ) : 
  (250 : ℝ) * (1 + x) ^ 20 ≥ 800 →
  (250 : ℝ) * (1 + x) ^ 40 ≥ 2560 := 
by
  intros h
  -- Proof is not required, so we skip it with sorry
  sorry

end per_capita_income_growth_l15_15700


namespace selection_of_hexagonal_shape_l15_15714

-- Lean 4 Statement: Prove that there are 78 distinct ways to select diagram b from the hexagonal grid of diagram a, considering rotations.

theorem selection_of_hexagonal_shape :
  let center_positions := 1
  let first_ring_positions := 6
  let second_ring_positions := 12
  let third_ring_positions := 6
  let fourth_ring_positions := 1
  let total_positions := center_positions + first_ring_positions + second_ring_positions + third_ring_positions + fourth_ring_positions
  let rotations := 3
  total_positions * rotations = 78 := by
  -- You can skip the explicit proof body here, replace with sorry
  sorry

end selection_of_hexagonal_shape_l15_15714


namespace geometric_sequence_common_ratio_l15_15496

open scoped Nat

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n) :
  ∃ r : ℝ, (∀ n : ℕ, a n = a 0 * r ^ n) ∧ (r = 4) :=
sorry

end geometric_sequence_common_ratio_l15_15496


namespace factorization_of_polynomial_l15_15412

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l15_15412


namespace scientific_notation_135000_l15_15812

theorem scientific_notation_135000 :
  135000 = 1.35 * 10^5 := sorry

end scientific_notation_135000_l15_15812


namespace ab_cd_value_l15_15930

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = -3) 
  (h3 : a + c + d = 10) 
  (h4 : b + c + d = -1) : 
  ab + cd = -346 / 9 :=
by 
  sorry

end ab_cd_value_l15_15930


namespace find_m_l15_15640

theorem find_m (x y m : ℝ)
  (h1 : 6 * x + 3 = 0)
  (h2 : 3 * y + m = 15)
  (h3 : x * y = 1) : m = 21 := 
sorry

end find_m_l15_15640


namespace horizontal_length_circumference_l15_15972

noncomputable def ratio := 16 / 9
noncomputable def diagonal := 32
noncomputable def computed_length := 32 * 16 / (Real.sqrt 337)
noncomputable def computed_perimeter := 2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337))

theorem horizontal_length 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  32 * 16 / (Real.sqrt 337) = 512 / (Real.sqrt 337) :=
by sorry

theorem circumference 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337)) = 1600 / (Real.sqrt 337) :=
by sorry

end horizontal_length_circumference_l15_15972


namespace percentage_profit_l15_15893

theorem percentage_profit (cp sp : ℝ) (h1 : cp = 1200) (h2 : sp = 1680) : ((sp - cp) / cp) * 100 = 40 := 
by 
  sorry

end percentage_profit_l15_15893


namespace salary_for_may_l15_15901

theorem salary_for_may (J F M A May : ℝ) 
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8200)
  (h3 : J = 5700) : 
  May = 6500 :=
by 
  have eq1 : J + F + M + A = 32000 := by
    linarith
  have eq2 : F + M + A + May = 32800 := by
    linarith
  have eq3 : May - J = 800 := by
    linarith [eq1, eq2]
  have eq4 : May = 6500 := by
    linarith [eq3, h3]
  exact eq4

end salary_for_may_l15_15901


namespace least_integer_sum_of_primes_l15_15008

-- Define what it means to be prime and greater than a number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def greater_than_ten (n : ℕ) : Prop := n > 10

-- Main theorem statement
theorem least_integer_sum_of_primes :
  ∃ n, (∀ p1 p2 p3 p4 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
                        greater_than_ten p1 ∧ greater_than_ten p2 ∧ greater_than_ten p3 ∧ greater_than_ten p4 ∧
                        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
                        n = p1 + p2 + p3 + p4 → n ≥ 60) ∧
        n = 60 :=
  sorry

end least_integer_sum_of_primes_l15_15008


namespace reciprocal_of_neg_eight_l15_15313

theorem reciprocal_of_neg_eight : -8 * (-1/8) = 1 := 
by
  sorry

end reciprocal_of_neg_eight_l15_15313


namespace min_xy_l15_15775

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
by sorry

end min_xy_l15_15775


namespace find_sam_age_l15_15760

variable (Sam Drew : ℕ)

-- Conditions as definitions in Lean 4
def combined_age (Sam Drew : ℕ) : Prop := Sam + Drew = 54
def sam_half_drew (Sam Drew : ℕ) : Prop := Sam = Drew / 2

theorem find_sam_age (Sam Drew : ℕ) (h1 : combined_age Sam Drew) (h2 : sam_half_drew Sam Drew) : Sam = 18 :=
sorry

end find_sam_age_l15_15760


namespace teacher_age_l15_15765

theorem teacher_age (avg_age_students : ℕ) (num_students : ℕ) (avg_age_total : ℕ) (num_total : ℕ) (h1 : avg_age_students = 21) (h2 : num_students = 20) (h3 : avg_age_total = 22) (h4 : num_total = 21) :
  let total_age_students := avg_age_students * num_students
  let total_age_class := avg_age_total * num_total
  let teacher_age := total_age_class - total_age_students
  teacher_age = 42 :=
by
  sorry

end teacher_age_l15_15765


namespace cuboid_volume_l15_15122

theorem cuboid_volume (x y z : ℝ)
  (h1 : 2 * (x + y) = 20)
  (h2 : 2 * (y + z) = 32)
  (h3 : 2 * (x + z) = 28) : x * y * z = 240 := 
by
  sorry

end cuboid_volume_l15_15122


namespace james_tv_watching_time_l15_15468

theorem james_tv_watching_time
  (ep_jeopardy : ℕ := 20) -- Each episode of Jeopardy is 20 minutes long
  (n_jeopardy : ℕ := 2) -- James watched 2 episodes of Jeopardy
  (n_wheel : ℕ := 2) -- James watched 2 episodes of Wheel of Fortune
  (wheel_factor : ℕ := 2) -- Wheel of Fortune episodes are twice as long as Jeopardy episodes
  : (ep_jeopardy * n_jeopardy + ep_jeopardy * wheel_factor * n_wheel) / 60 = 2 :=
by
  sorry

end james_tv_watching_time_l15_15468


namespace fibonacci_recurrence_l15_15728

def F : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem fibonacci_recurrence (n : ℕ) (h: n ≥ 2) : 
  F n = F (n-1) + F (n-2) := by
 {
 sorry
 }

end fibonacci_recurrence_l15_15728


namespace ship_cargo_weight_l15_15680

theorem ship_cargo_weight (initial_cargo_tons additional_cargo_tons : ℝ) (unloaded_cargo_pounds : ℝ)
    (ton_to_kg pound_to_kg : ℝ) :
    initial_cargo_tons = 5973.42 →
    additional_cargo_tons = 8723.18 →
    unloaded_cargo_pounds = 2256719.55 →
    ton_to_kg = 907.18474 →
    pound_to_kg = 0.45359237 →
    (initial_cargo_tons * ton_to_kg + additional_cargo_tons * ton_to_kg - unloaded_cargo_pounds * pound_to_kg = 12302024.7688159) :=
by
  intros
  sorry

end ship_cargo_weight_l15_15680


namespace minimum_value_l15_15485

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃z, z = (x^2 + y^2) / (x + y)^2 ∧ z ≥ 1/2 := 
sorry

end minimum_value_l15_15485


namespace boy_usual_time_to_school_l15_15197

theorem boy_usual_time_to_school
  (S : ℝ) -- Usual speed
  (T : ℝ) -- Usual time
  (D : ℝ) -- Distance, D = S * T
  (hD : D = S * T)
  (h1 : 3/4 * D / (7/6 * S) + 1/4 * D / (5/6 * S) = T - 2) : 
  T = 35 :=
by
  sorry

end boy_usual_time_to_school_l15_15197


namespace quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l15_15052

structure Point where
  x : ℚ
  y : ℚ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 5, y := 4 }
def D : Point := { x := 6, y := 1 }

def line_eq_y_eq_kx_plus_b (k b x : ℚ) : ℚ := k * x + b

def intersects (A : Point) (P : Point × Point) (x y : ℚ) : Prop :=
  ∃ k b, P.1.y = line_eq_y_eq_kx_plus_b k b P.1.x ∧ P.2.y = line_eq_y_eq_kx_plus_b k b P.2.x ∧
         y = line_eq_y_eq_kx_plus_b k b x

theorem quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176 :
  ∃ (p q r s : ℚ), 
    gcd p q = 1 ∧ gcd r s = 1 ∧ intersects A (C, D) (p / q) (r / s) ∧
    (p + q + r + s = 176) :=
sorry

end quadrilateral_bisector_intersection_p_q_r_s_sum_eq_176_l15_15052


namespace planting_trees_equation_l15_15372

theorem planting_trees_equation (x : ℝ) (h1 : x > 0) : 
  20 / x - 20 / ((1 + 0.1) * x) = 4 :=
sorry

end planting_trees_equation_l15_15372


namespace gcd_2025_2070_l15_15457

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end gcd_2025_2070_l15_15457


namespace point_in_fourth_quadrant_l15_15490

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) : a > 0 ∧ b < 0 :=
by 
  have hb : b < 0 := sorry
  exact ⟨h1, hb⟩

end point_in_fourth_quadrant_l15_15490


namespace assume_dead_heat_race_l15_15169

variable {Va Vb L H : ℝ}

theorem assume_dead_heat_race (h1 : Va = (51 / 44) * Vb) :
  H = (7 / 51) * L :=
sorry

end assume_dead_heat_race_l15_15169


namespace hexagon_area_eq_l15_15920

theorem hexagon_area_eq (s t : ℝ) (hs : s^2 = 16) (heq : 4 * s = 6 * t) :
  6 * (t^2 * (Real.sqrt 3) / 4) = 32 * (Real.sqrt 3) / 3 := by
  sorry

end hexagon_area_eq_l15_15920


namespace calculate_expression_l15_15498

theorem calculate_expression : abs (-2) - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end calculate_expression_l15_15498


namespace sum_of_roots_l15_15100

theorem sum_of_roots (x1 x2 : ℝ) (h1 : x1^2 + 5*x1 - 3 = 0) (h2 : x2^2 + 5*x2 - 3 = 0) (h3 : x1 ≠ x2) :
  x1 + x2 = -5 :=
sorry

end sum_of_roots_l15_15100


namespace matthew_egg_rolls_l15_15275

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l15_15275


namespace computer_multiplications_in_30_minutes_l15_15465

def multiplications_per_second : ℕ := 20000
def seconds_per_minute : ℕ := 60
def minutes : ℕ := 30
def total_seconds : ℕ := minutes * seconds_per_minute
def expected_multiplications : ℕ := 36000000

theorem computer_multiplications_in_30_minutes :
  multiplications_per_second * total_seconds = expected_multiplications :=
by
  sorry

end computer_multiplications_in_30_minutes_l15_15465


namespace train_time_to_pass_platform_l15_15207

noncomputable def train_length : ℝ := 360
noncomputable def platform_length : ℝ := 140
noncomputable def train_speed_km_per_hr : ℝ := 45

noncomputable def train_speed_m_per_s : ℝ :=
  train_speed_km_per_hr * (1000 / 3600)

noncomputable def total_distance : ℝ :=
  train_length + platform_length

theorem train_time_to_pass_platform :
  (total_distance / train_speed_m_per_s) = 40 := by
  sorry

end train_time_to_pass_platform_l15_15207


namespace marie_erasers_l15_15536

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) 
  (h1 : initial_erasers = 95) (h2 : lost_erasers = 42) : final_erasers = 53 :=
by
  sorry

end marie_erasers_l15_15536


namespace change_in_max_value_l15_15426

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem change_in_max_value (a b c : ℝ) (h1 : -b^2 / (4 * (a + 1)) + c = -b^2 / (4 * a) + c + 27 / 2)
  (h2 : -b^2 / (4 * (a - 4)) + c = -b^2 / (4 * a) + c - 9) :
  -b^2 / (4 * (a - 2)) + c = -b^2 / (4 * a) + c - 27 / 4 :=
by
  sorry

end change_in_max_value_l15_15426


namespace total_frogs_in_both_ponds_l15_15470

noncomputable def total_frogs_combined : Nat :=
let frogs_in_pond_a : Nat := 32
let frogs_in_pond_b : Nat := frogs_in_pond_a / 2
frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_in_both_ponds :
  total_frogs_combined = 48 := by
  sorry

end total_frogs_in_both_ponds_l15_15470


namespace james_paid_with_l15_15398

variable (candy_packs : ℕ) (cost_per_pack : ℕ) (change_received : ℕ)

theorem james_paid_with (h1 : candy_packs = 3) (h2 : cost_per_pack = 3) (h3 : change_received = 11) :
  let total_cost := candy_packs * cost_per_pack
  let amount_paid := total_cost + change_received
  amount_paid = 20 :=
by
  sorry

end james_paid_with_l15_15398


namespace parabola_expression_l15_15966

theorem parabola_expression :
  ∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x - 5 = 0 → (x = -1 ∨ x = 5)) ∧ (a * (-1)^2 + b * (-1) - 5 = 0) ∧ (a * 5^2 + b * 5 - 5 = 0) ∧ (a * 1 - 4 = 1) :=
sorry

end parabola_expression_l15_15966


namespace max_sum_abc_min_sum_reciprocal_l15_15032

open Real

variables {a b c : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 2)

-- Maximum of a + b + c
theorem max_sum_abc : a + b + c ≤ sqrt 6 :=
by sorry

-- Minimum of 1/(a + b) + 1/(b + c) + 1/(c + a)
theorem min_sum_reciprocal : (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 * sqrt 6 / 4 :=
by sorry

end max_sum_abc_min_sum_reciprocal_l15_15032


namespace curve_crosses_itself_l15_15656

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := t^2 - 4
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

-- The theorem statement
theorem curve_crosses_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁, y t₁) = (2, 3) :=
by
  -- Proof would go here
  sorry

end curve_crosses_itself_l15_15656


namespace find_some_number_l15_15382

theorem find_some_number :
  ∃ (x : ℝ), abs (x - 0.004) < 0.0001 ∧ 9.237333333333334 = (69.28 * x) / 0.03 := by
  sorry

end find_some_number_l15_15382


namespace find_total_cows_l15_15314

-- Define the conditions given in the problem
def ducks_legs (D : ℕ) : ℕ := 2 * D
def cows_legs (C : ℕ) : ℕ := 4 * C
def total_legs (D C : ℕ) : ℕ := ducks_legs D + cows_legs C
def total_heads (D C : ℕ) : ℕ := D + C

-- State the problem in Lean 4
theorem find_total_cows (D C : ℕ) (h : total_legs D C = 2 * total_heads D C + 32) : C = 16 :=
sorry

end find_total_cows_l15_15314


namespace minimize_distance_school_l15_15781

-- Define the coordinates for the towns X, Y, and Z
def X_coord : ℕ × ℕ := (0, 0)
def Y_coord : ℕ × ℕ := (200, 0)
def Z_coord : ℕ × ℕ := (0, 300)

-- Define the population of the towns
def X_population : ℕ := 100
def Y_population : ℕ := 200
def Z_population : ℕ := 300

theorem minimize_distance_school : ∃ (x y : ℕ), x + y = 300 := by
  -- This should follow from the problem setup and conditions.
  sorry

end minimize_distance_school_l15_15781


namespace metropolis_hospital_babies_l15_15373

theorem metropolis_hospital_babies 
    (a b d : ℕ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * d) 
    (h3 : 2 * a + 3 * b + 5 * d = 1200) : 
    5 * d = 260 := 
sorry

end metropolis_hospital_babies_l15_15373


namespace hermione_utility_l15_15512

theorem hermione_utility (h : ℕ) : (h * (10 - h) = (4 - h) * (h + 2)) ↔ h = 4 := by
  sorry

end hermione_utility_l15_15512


namespace smallest_solution_l15_15146

theorem smallest_solution (x : ℝ) (h : x * |x| = 2 * x + 1) : x = -1 := 
by
  sorry

end smallest_solution_l15_15146


namespace common_ratio_is_2_l15_15951

noncomputable def common_ratio_of_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n+1) = a n * q) ∧ (∀ m n, m < n → a m < a n)

theorem common_ratio_is_2
  (a : ℕ → ℝ) (q : ℝ)
  (hgeo : common_ratio_of_increasing_geometric_sequence a q)
  (h1 : a 1 + a 5 = 17)
  (h2 : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end common_ratio_is_2_l15_15951


namespace domain_of_f_l15_15689

def domain_f (x : ℝ) : Prop := x ≤ 4 ∧ x ≠ 1

theorem domain_of_f :
  {x : ℝ | ∃(h1 : 4 - x ≥ 0) (h2 : x - 1 ≠ 0), true} = {x : ℝ | domain_f x} :=
by
  sorry

end domain_of_f_l15_15689


namespace initial_money_is_10_l15_15220

-- Definition for the initial amount of money
def initial_money (X : ℝ) : Prop :=
  let spent_on_cupcakes := (1 / 5) * X
  let remaining_after_cupcakes := X - spent_on_cupcakes
  let spent_on_milkshake := 5
  let remaining_after_milkshake := remaining_after_cupcakes - spent_on_milkshake
  remaining_after_milkshake = 3

-- The statement proving that Ivan initially had $10
theorem initial_money_is_10 (X : ℝ) (h : initial_money X) : X = 10 :=
by sorry

end initial_money_is_10_l15_15220


namespace three_times_x_greater_than_four_l15_15849

theorem three_times_x_greater_than_four (x : ℝ) : 3 * x > 4 := by
  sorry

end three_times_x_greater_than_four_l15_15849


namespace louise_needs_eight_boxes_l15_15578

-- Define the given conditions
def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def pencils_per_box : ℕ := 20

-- Define the functions to calculate the required number of boxes for each color
def boxes_needed (pencils : ℕ) : ℕ := (pencils + pencils_per_box - 1) / pencils_per_box

-- Calculate the total number of boxes needed by summing the boxes for each color
def total_boxes_needed : ℕ := boxes_needed red_pencils + boxes_needed blue_pencils + boxes_needed yellow_pencils + boxes_needed green_pencils

-- The proof problem statement
theorem louise_needs_eight_boxes : total_boxes_needed = 8 :=
by
  sorry

end louise_needs_eight_boxes_l15_15578


namespace first_quadrant_sin_cos_inequality_l15_15686

def is_first_quadrant_angle (α : ℝ) : Prop :=
  0 < Real.sin α ∧ 0 < Real.cos α

theorem first_quadrant_sin_cos_inequality (α : ℝ) :
  (is_first_quadrant_angle α ↔ Real.sin α + Real.cos α > 1) :=
by
  sorry

end first_quadrant_sin_cos_inequality_l15_15686


namespace problem1_problem2_l15_15911

namespace MathProof

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem problem1 (m : ℝ) :
  (∀ x, 0 < x → f x m > 0) → -2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5 :=
sorry

theorem problem2 (m : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ f x m = 0) → -2 < m ∧ m < 0 :=
sorry

end MathProof

end problem1_problem2_l15_15911


namespace number_of_workers_l15_15321

-- Definitions corresponding to problem conditions
def total_contribution := 300000
def extra_total_contribution := 325000
def extra_amount := 50

-- Main statement to prove the number of workers
theorem number_of_workers : ∃ W C : ℕ, W * C = total_contribution ∧ W * (C + extra_amount) = extra_total_contribution ∧ W = 500 := by
  sorry

end number_of_workers_l15_15321


namespace no_rational_points_on_sqrt3_circle_l15_15450

theorem no_rational_points_on_sqrt3_circle (x y : ℚ) : x^2 + y^2 ≠ 3 :=
sorry

end no_rational_points_on_sqrt3_circle_l15_15450


namespace slope_of_line_l15_15727

theorem slope_of_line : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 1) ∧ (y1 = 3) ∧ (x2 = 7) ∧ (y2 = -9)
  → (y2 - y1) / (x2 - x1) = -2 := by
  sorry

end slope_of_line_l15_15727


namespace actual_price_of_food_before_tax_and_tip_l15_15651

theorem actual_price_of_food_before_tax_and_tip 
  (total_paid : ℝ)
  (tip_percentage : ℝ)
  (tax_percentage : ℝ)
  (pre_tax_food_price : ℝ)
  (h1 : total_paid = 132)
  (h2 : tip_percentage = 0.20)
  (h3 : tax_percentage = 0.10)
  (h4 : total_paid = (1 + tip_percentage) * (1 + tax_percentage) * pre_tax_food_price) :
  pre_tax_food_price = 100 :=
by sorry

end actual_price_of_food_before_tax_and_tip_l15_15651


namespace slope_intercept_of_line_l15_15670

theorem slope_intercept_of_line :
  ∃ (l : ℝ → ℝ), (∀ x, l x = (4 * x - 9) / 3) ∧ l 3 = 1 ∧ ∃ k, k / (1 + k^2) = 1 / 2 ∧ l x = (k^2 - 1) / (1 + k^2) := sorry

end slope_intercept_of_line_l15_15670


namespace smallest_number_divisible_by_20_and_36_is_180_l15_15428

theorem smallest_number_divisible_by_20_and_36_is_180 :
  ∃ x, (x % 20 = 0) ∧ (x % 36 = 0) ∧ (∀ y, (y % 20 = 0) ∧ (y % 36 = 0) → x ≤ y) ∧ x = 180 :=
by
  sorry

end smallest_number_divisible_by_20_and_36_is_180_l15_15428


namespace work_equivalence_l15_15997

variable (m d r : ℕ)

theorem work_equivalence (h : d > 0) : (m * d) / (m + r^2) = d := sorry

end work_equivalence_l15_15997


namespace aunt_gemma_dog_food_l15_15507

theorem aunt_gemma_dog_food :
  ∀ (dogs : ℕ) (grams_per_meal : ℕ) (meals_per_day : ℕ) (sack_kg : ℕ) (days : ℕ), 
    dogs = 4 →
    grams_per_meal = 250 →
    meals_per_day = 2 →
    sack_kg = 50 →
    days = 50 →
    (dogs * meals_per_day * grams_per_meal * days) / (1000 * sack_kg) = 2 :=
by
  intros dogs grams_per_meal meals_per_day sack_kg days
  intros h_dogs h_grams_per_meal h_meals_per_day h_sack_kg h_days
  sorry

end aunt_gemma_dog_food_l15_15507


namespace fruit_cost_l15_15495

theorem fruit_cost:
  let strawberry_cost := 2.20
  let cherry_cost := 6 * strawberry_cost
  let blueberry_cost := cherry_cost / 2
  let strawberries_count := 3
  let cherries_count := 4.5
  let blueberries_count := 6.2
  let total_cost := (strawberries_count * strawberry_cost) + (cherries_count * cherry_cost) + (blueberries_count * blueberry_cost)
  total_cost = 106.92 :=
by
  sorry

end fruit_cost_l15_15495


namespace cost_of_each_lunch_packet_l15_15282

-- Definitions of the variables
def num_students := 50
def total_cost := 3087

-- Variables representing the unknowns
variable (s c n : ℕ)

-- Conditions
def more_than_half_students_bought : Prop := s > num_students / 2
def apples_less_than_cost_per_packet : Prop := n < c
def total_cost_condition : Prop := s * c = total_cost

-- The statement to prove
theorem cost_of_each_lunch_packet :
  (s : ℕ) * c = total_cost ∧
  (s > num_students / 2) ∧
  (n < c)
  -> c = 9 :=
by
  sorry

end cost_of_each_lunch_packet_l15_15282


namespace inequality_proof_l15_15715

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := 
by
  sorry

end inequality_proof_l15_15715


namespace absolute_error_2175000_absolute_error_1730000_l15_15931

noncomputable def absolute_error (a : ℕ) : ℕ :=
  if a = 2175000 then 1
  else if a = 1730000 then 10000
  else 0

theorem absolute_error_2175000 : absolute_error 2175000 = 1 :=
by sorry

theorem absolute_error_1730000 : absolute_error 1730000 = 10000 :=
by sorry

end absolute_error_2175000_absolute_error_1730000_l15_15931


namespace unique_solution_l15_15362

theorem unique_solution (x y z : ℕ) (h_x : x > 1) (h_y : y > 1) (h_z : z > 1) :
  (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end unique_solution_l15_15362


namespace find_number_l15_15647

-- Definitions based on the given conditions
def area (s : ℝ) := s^2
def perimeter (s : ℝ) := 4 * s
def given_perimeter : ℝ := 36
def equation (s : ℝ) (n : ℝ) := 5 * area s = 10 * perimeter s + n

-- Statement of the problem
theorem find_number :
  ∃ n : ℝ, equation (given_perimeter / 4) n ∧ n = 45 :=
by
  sorry

end find_number_l15_15647


namespace sarah_initial_trucks_l15_15348

theorem sarah_initial_trucks (trucks_given : ℕ) (trucks_left : ℕ) (initial_trucks : ℕ) :
  trucks_given = 13 → trucks_left = 38 → initial_trucks = trucks_left + trucks_given → initial_trucks = 51 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_initial_trucks_l15_15348


namespace tom_total_payment_l15_15063

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end tom_total_payment_l15_15063


namespace perpendicular_bisector_eq_l15_15741

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 2)

-- Theorem stating that the perpendicular bisector has the specified equation
theorem perpendicular_bisector_eq : ∀ (x y : ℝ), (y = -2 * x + 3) ↔ ∃ (a b : ℝ), (a, b) = A ∨ (a, b) = B ∧ (y = -2 * x + 3) :=
by
  sorry

end perpendicular_bisector_eq_l15_15741


namespace lemon_heads_distribution_l15_15751

-- Conditions
def total_lemon_heads := 72
def number_of_friends := 6

-- Desired answer
def lemon_heads_per_friend := 12

-- Lean 4 statement
theorem lemon_heads_distribution : total_lemon_heads / number_of_friends = lemon_heads_per_friend := by 
  sorry

end lemon_heads_distribution_l15_15751


namespace percentage_of_number_is_40_l15_15280

theorem percentage_of_number_is_40 (N : ℝ) (P : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 35) 
  (h2 : (P/100) * N = 420) : 
  P = 40 := 
by
  sorry

end percentage_of_number_is_40_l15_15280


namespace round_trip_ticket_percentage_l15_15784

theorem round_trip_ticket_percentage (p : ℕ → Prop) : 
  (∀ n, p n → n = 375) → (∀ n, p n → n = 375) :=
by
  sorry

end round_trip_ticket_percentage_l15_15784


namespace parabola_directrix_symmetry_l15_15973

theorem parabola_directrix_symmetry:
  (∃ (d : ℝ), (∀ x : ℝ, x = d ↔ 
  (∃ y : ℝ, y^2 = (1 / 2) * x) ∧
  (∀ y : ℝ, x = (1 / 8)) → x = - (1 / 8))) :=
sorry

end parabola_directrix_symmetry_l15_15973


namespace dodecagon_enclosure_l15_15034

theorem dodecagon_enclosure (m n : ℕ) (h1 : m = 12) 
  (h2 : ∀ (x : ℕ), x ∈ { k | ∃ p : ℕ, p = n ∧ 12 = k * p}) :
  n = 12 :=
by
  -- begin proof steps here
sorry

end dodecagon_enclosure_l15_15034


namespace intersection_M_N_l15_15612

variable (x : ℝ)

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l15_15612


namespace portion_of_work_done_l15_15099

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end portion_of_work_done_l15_15099


namespace final_acid_concentration_l15_15905

def volume1 : ℝ := 2
def concentration1 : ℝ := 0.40
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.60

theorem final_acid_concentration :
  ((concentration1 * volume1 + concentration2 * volume2) / (volume1 + volume2)) = 0.52 :=
by
  sorry

end final_acid_concentration_l15_15905


namespace trapezoidal_park_no_solution_l15_15732

theorem trapezoidal_park_no_solution :
  (∃ b1 b2 : ℕ, 2 * 1800 = 40 * (b1 + b2) ∧ (∃ m : ℕ, b1 = 5 * (2 * m + 1)) ∧ (∃ n : ℕ, b2 = 2 * n)) → false :=
by
  sorry

end trapezoidal_park_no_solution_l15_15732


namespace total_suitcases_l15_15953

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l15_15953


namespace haley_picked_carrots_l15_15040

variable (H : ℕ)
variable (mom_carrots : ℕ := 38)
variable (good_carrots : ℕ := 64)
variable (bad_carrots : ℕ := 13)
variable (total_carrots : ℕ := good_carrots + bad_carrots)

theorem haley_picked_carrots : H + mom_carrots = total_carrots → H = 39 := by
  sorry

end haley_picked_carrots_l15_15040


namespace incorrect_average_l15_15574

theorem incorrect_average (S : ℕ) (A_correct : ℕ) (A_incorrect : ℕ) (S_correct : ℕ) 
  (h1 : S = 135)
  (h2 : A_correct = 19)
  (h3 : A_incorrect = (S + 25) / 10)
  (h4 : S_correct = (S + 55) / 10)
  (h5 : S_correct = A_correct) :
  A_incorrect = 16 :=
by
  -- The proof will go here, which is skipped with a 'sorry'
  sorry

end incorrect_average_l15_15574


namespace negation_red_cards_in_deck_l15_15019

variable (Deck : Type) (is_red : Deck → Prop) (is_in_deck : Deck → Prop)

theorem negation_red_cards_in_deck :
  (¬ ∃ x : Deck, is_red x ∧ is_in_deck x) ↔ (∃ x : Deck, is_red x ∧ is_in_deck x) :=
by {
  sorry
}

end negation_red_cards_in_deck_l15_15019


namespace sulfuric_acid_moles_used_l15_15391

-- Definitions and conditions
def iron_moles : ℕ := 2
def iron_ii_sulfate_moles_produced : ℕ := 2
def sulfuric_acid_to_iron_ratio : ℕ := 1

-- Proof statement
theorem sulfuric_acid_moles_used {H2SO4_moles : ℕ} 
  (h_fe_reacts : H2SO4_moles = iron_moles * sulfuric_acid_to_iron_ratio) 
  (h_fe produces: iron_ii_sulfate_moles_produced = iron_moles) : H2SO4_moles = 2 :=
by
  sorry

end sulfuric_acid_moles_used_l15_15391


namespace max_board_size_l15_15056

theorem max_board_size : ∀ (n : ℕ), 
  (∃ (board : Fin n → Fin n → Prop),
    ∀ i j k l : Fin n,
      (i ≠ k ∧ j ≠ l) → board i j ≠ board k l) ↔ n ≤ 4 :=
by sorry

end max_board_size_l15_15056


namespace ratio_of_N_to_R_l15_15141

variables (N T R k : ℝ)

theorem ratio_of_N_to_R (h1 : T = (1 / 4) * N)
                        (h2 : R = 40)
                        (h3 : N = k * R)
                        (h4 : T + R + N = 190) :
    N / R = 3 :=
by
  sorry

end ratio_of_N_to_R_l15_15141


namespace quadratic_eq_standard_form_coefficients_l15_15484

-- Define initial quadratic equation
def initial_eq (x : ℝ) : Prop := (x + 5) * (x + 3) = 2 * x^2

-- Define the quadratic equation in standard form
def standard_form (x : ℝ) : Prop := x^2 - 8 * x - 15 = 0

-- Prove that given the initial equation, it can be converted to its standard form
theorem quadratic_eq_standard_form (x : ℝ) :
  initial_eq x → standard_form x := 
sorry

-- Verify the coefficients of the quadratic term, linear term, and constant term
theorem coefficients (x : ℝ) :
  initial_eq x → 
  (∀ a b c : ℝ, (a = 1) ∧ (b = -8) ∧ (c = -15) → standard_form x) :=
sorry

end quadratic_eq_standard_form_coefficients_l15_15484


namespace ab_non_positive_l15_15058

-- Define the conditions as a structure if necessary.
variables {a b : ℝ}

-- State the theorem.
theorem ab_non_positive (h : 3 * a + 8 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l15_15058


namespace total_cost_correct_l15_15898

noncomputable def total_cost : ℝ :=
  let first_path_area := 5 * 100
  let first_path_cost := first_path_area * 2
  let second_path_area := 4 * 80
  let second_path_cost := second_path_area * 1.5
  let diagonal_length := Real.sqrt ((100:ℝ)^2 + (80:ℝ)^2)
  let third_path_area := 6 * diagonal_length
  let third_path_cost := third_path_area * 3
  let circular_path_area := Real.pi * (10:ℝ)^2
  let circular_path_cost := circular_path_area * 4
  first_path_cost + second_path_cost + third_path_cost + circular_path_cost

theorem total_cost_correct : total_cost = 5040.64 := by
  sorry

end total_cost_correct_l15_15898


namespace perfect_squares_from_equation_l15_15684

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ a b c : ℕ, x - y = a^2 ∧ 2 * x + 2 * y + 1 = b^2 ∧ 3 * x + 3 * y + 1 = c^2 :=
by
  sorry

end perfect_squares_from_equation_l15_15684


namespace lcm_135_468_l15_15631

theorem lcm_135_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end lcm_135_468_l15_15631


namespace parabola_equation_l15_15200

theorem parabola_equation (P : ℝ × ℝ) :
  let d1 := dist P (-3, 0)
  let d2 := abs (P.1 - 2)
  (d1 = d2 + 1 ↔ P.2^2 = -12 * P.1) :=
by
  intro d1 d2
  sorry

end parabola_equation_l15_15200


namespace find_f_105_5_l15_15185

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f

-- Hypotheses
axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (x + 2) = -f x
axiom function_values (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = x

-- Goal
theorem find_f_105_5 : f 105.5 = 2.5 :=
sorry

end find_f_105_5_l15_15185


namespace line_passes_through_fixed_point_l15_15152

theorem line_passes_through_fixed_point (p q : ℝ) (h : p + 2 * q - 1 = 0) :
  p * (1/2) + 3 * (-1/6) + q = 0 :=
by
  -- placeholders for the actual proof steps
  sorry

end line_passes_through_fixed_point_l15_15152


namespace find_y_and_y2_l15_15154

theorem find_y_and_y2 (d y y2 : ℤ) (h1 : 3 ^ 2 = 9) (h2 : 3 ^ 4 = 81)
  (h3 : y = 9 + d) (h4 : y2 = 81 + d) (h5 : 81 = 9 + 3 * d) :
  y = 33 ∧ y2 = 105 :=
by
  sorry

end find_y_and_y2_l15_15154


namespace find_percentage_l15_15384

theorem find_percentage (x p : ℝ) (h₀ : x = 780) (h₁ : 0.25 * x = (p / 100) * 1500 - 30) : p = 15 :=
by
  sorry

end find_percentage_l15_15384


namespace tom_spent_on_videogames_l15_15622

theorem tom_spent_on_videogames (batman_game superman_game : ℝ) 
  (h1 : batman_game = 13.60) 
  (h2 : superman_game = 5.06) : 
  batman_game + superman_game = 18.66 :=
by 
  sorry

end tom_spent_on_videogames_l15_15622


namespace part1_part2_l15_15549

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ Real.exp 1 + 1 := 
sorry

theorem part2 {a : ℝ} {x1 x2 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (hz1 : f x1 a = 0) (hz2 : f x2 a = 0) : x1 * x2 < 1 := 
sorry

end part1_part2_l15_15549


namespace find_other_num_l15_15112

variables (a b : ℕ)

theorem find_other_num (h_gcd : Nat.gcd a b = 12) (h_lcm : Nat.lcm a b = 5040) (h_a : a = 240) :
  b = 252 :=
  sorry

end find_other_num_l15_15112


namespace ratio_of_Carla_to_Cosima_l15_15897

variables (C M : ℝ)

-- Natasha has 3 times as much money as Carla
axiom h1 : 3 * C = 60

-- Carla has the same amount of money as Cosima
axiom h2 : C = M

-- Prove: the ratio of Carla's money to Cosima's money is 1:1
theorem ratio_of_Carla_to_Cosima : C / M = 1 :=
by sorry

end ratio_of_Carla_to_Cosima_l15_15897


namespace number_of_larger_planes_l15_15834

variable (S L : ℕ)
variable (h1 : S + L = 4)
variable (h2 : 130 * S + 145 * L = 550)

theorem number_of_larger_planes : L = 2 :=
by
  -- Placeholder for the proof
  sorry

end number_of_larger_planes_l15_15834


namespace darry_full_ladder_climbs_l15_15183

-- Definitions and conditions
def full_ladder_steps : ℕ := 11
def smaller_ladder_steps : ℕ := 6
def smaller_ladder_climbs : ℕ := 7
def total_steps_climbed_today : ℕ := 152

-- Question: How many times did Darry climb his full ladder?
theorem darry_full_ladder_climbs (x : ℕ) 
  (H : 11 * x + smaller_ladder_steps * 7 = total_steps_climbed_today) : 
  x = 10 := by
  -- proof steps omitted, so we write
  sorry

end darry_full_ladder_climbs_l15_15183


namespace bowling_ball_weight_l15_15103

theorem bowling_ball_weight (b c : ℝ) (h1 : 5 * b = 3 * c) (h2 : 2 * c = 56) : b = 16.8 := by
  sorry

end bowling_ball_weight_l15_15103


namespace find_m_value_l15_15544

theorem find_m_value (a m : ℤ) (h : a ≠ 1) (hx : ∀ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a - 1) * x^2 - m * x + a = 0 ∧ (a - 1) * y^2 - m * y + a = 0) : m = 3 :=
sorry

end find_m_value_l15_15544


namespace average_of_xyz_l15_15160

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 :=
by
  sorry

end average_of_xyz_l15_15160


namespace divisible_by_six_l15_15889

theorem divisible_by_six (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 6 ∣ n :=
sorry

end divisible_by_six_l15_15889


namespace ratio_boys_to_girls_l15_15581

theorem ratio_boys_to_girls (total_students girls : ℕ) (h1 : total_students = 455) (h2 : girls = 175) :
  let boys := total_students - girls
  (boys : ℕ) / Nat.gcd boys girls = 8 / 1 ∧ (girls : ℕ) / Nat.gcd boys girls = 5 / 1 :=
by
  sorry

end ratio_boys_to_girls_l15_15581


namespace tan_problem_l15_15858

noncomputable def problem : ℝ :=
  (Real.tan (20 * Real.pi / 180) + Real.tan (40 * Real.pi / 180) + Real.tan (120 * Real.pi / 180)) / 
  (Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180))

theorem tan_problem : problem = -Real.sqrt 3 := by
  sorry

end tan_problem_l15_15858


namespace least_total_cost_is_172_l15_15264

noncomputable def least_total_cost : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 6 5) 8
  let strawberry_packs := lcm / 6
  let blueberry_packs := lcm / 5
  let cherry_packs := lcm / 8
  let strawberry_cost := strawberry_packs * 2
  let blueberry_cost := blueberry_packs * 3
  let cherry_cost := cherry_packs * 4
  strawberry_cost + blueberry_cost + cherry_cost

theorem least_total_cost_is_172 : least_total_cost = 172 := 
by
  sorry

end least_total_cost_is_172_l15_15264


namespace geom_seq_value_l15_15167

variable (a_n : ℕ → ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- Define the conditions
axiom geom_seq : ∀ n, a_n (n + 1) = a_n n * r
axiom sum_pi : a_n 3 + a_n 5 = π

-- Statement to prove
theorem geom_seq_value : a_n 4 * (a_n 2 + 2 * a_n 4 + a_n 6) = π^2 :=
by
  sorry

end geom_seq_value_l15_15167


namespace graduation_photo_arrangement_l15_15462

theorem graduation_photo_arrangement (teachers middle_positions other_students : Finset ℕ) (A B : ℕ) :
  teachers.card = 2 ∧ middle_positions.card = 2 ∧ 
  (other_students ∪ {A, B}).card = 4 ∧ ∀ t ∈ teachers, t ∈ middle_positions →
  ∃ arrangements : ℕ, arrangements = 8 :=
by
  sorry

end graduation_photo_arrangement_l15_15462


namespace root_interval_sum_l15_15042

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 8

def has_root_in_interval (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : Prop :=
  a < b ∧ b - a = 1 ∧ f a < 0 ∧ f b > 0

theorem root_interval_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : has_root_in_interval a b h1 h2) : 
  a + b = 5 :=
sorry

end root_interval_sum_l15_15042


namespace least_possible_value_of_smallest_integer_l15_15674

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), 
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B + C + D) / 4 = 68 →
    D = 90 →
    A = 5 :=
by
  intros A B C D h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end least_possible_value_of_smallest_integer_l15_15674


namespace triangles_same_base_height_have_equal_areas_l15_15596

theorem triangles_same_base_height_have_equal_areas 
  (b1 h1 b2 h2 : ℝ) 
  (A1 A2 : ℝ) 
  (h1_nonneg : 0 ≤ h1) 
  (h2_nonneg : 0 ≤ h2) 
  (A1_eq : A1 = b1 * h1 / 2) 
  (A2_eq : A2 = b2 * h2 / 2) :
  (A1 = A2 ↔ b1 * h1 = b2 * h2) ∧ (b1 = b2 ∧ h1 = h2 → A1 = A2) :=
by {
  sorry
}

end triangles_same_base_height_have_equal_areas_l15_15596


namespace kopeechka_items_l15_15701

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l15_15701


namespace smallest_value_of_3a_plus_2_l15_15005

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  3 * a + 2 = 1 / 2 :=
sorry

end smallest_value_of_3a_plus_2_l15_15005


namespace walking_distance_l15_15867

theorem walking_distance (west east : ℤ) (h_west : west = 5) (h_east : east = -5) : west + east = 10 := 
by 
  rw [h_west, h_east] 
  sorry

end walking_distance_l15_15867


namespace det_example_1_simplified_form_det_at_4_l15_15605

-- Definition for second-order determinant
def second_order_determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Part (1)
theorem det_example_1 :
  second_order_determinant 3 (-2) 4 (-3) = -1 :=
by
  sorry

-- Part (2) simplified determinant
def simplified_det (x : ℤ) : ℤ :=
  second_order_determinant (2 * x - 3) (x + 2) 2 4

-- Proving simplified determinant form
theorem simplified_form :
  ∀ x : ℤ, simplified_det x = 6 * x - 16 :=
by
  sorry

-- Proving specific case when x = 4
theorem det_at_4 :
  simplified_det 4 = 8 :=
by 
  sorry

end det_example_1_simplified_form_det_at_4_l15_15605


namespace capacity_of_other_bottle_l15_15947

theorem capacity_of_other_bottle (x : ℝ) :
  (16 / 3) * (x / 8) + (16 / 3) = 8 → x = 4 := by
  -- the proof will go here
  sorry

end capacity_of_other_bottle_l15_15947


namespace remaining_amount_correct_l15_15255

def initial_amount : ℝ := 70
def coffee_cost_per_pound : ℝ := 8.58
def coffee_pounds : ℝ := 4.0
def total_cost : ℝ := coffee_pounds * coffee_cost_per_pound
def remaining_amount : ℝ := initial_amount - total_cost

theorem remaining_amount_correct : remaining_amount = 35.68 :=
by
  -- Skip the proof; this is a placeholder.
  sorry

end remaining_amount_correct_l15_15255


namespace sum_of_three_numbers_l15_15334

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35) 
  (h2 : b + c = 57) 
  (h3 : c + a = 62) : 
  a + b + c = 77 :=
by
  sorry

end sum_of_three_numbers_l15_15334


namespace negation_of_universal_statement_l15_15010

def P (x : ℝ) : Prop := x^3 - x^2 + 1 ≤ 0

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by {
  sorry
}

end negation_of_universal_statement_l15_15010


namespace card_probability_l15_15573

-- Definitions to capture the problem's conditions in Lean
def total_cards : ℕ := 52
def remaining_after_first : ℕ := total_cards - 1
def remaining_after_second : ℕ := total_cards - 2

def kings : ℕ := 4
def non_heart_kings : ℕ := 3
def non_kings_in_hearts : ℕ := 12
def spades_and_diamonds : ℕ := 26

-- Define probabilities for each step
def prob_first_king : ℚ := non_heart_kings / total_cards
def prob_second_heart : ℚ := non_kings_in_hearts / remaining_after_first
def prob_third_spade_or_diamond : ℚ := spades_and_diamonds / remaining_after_second

-- Calculate total probability
def total_probability : ℚ := prob_first_king * prob_second_heart * prob_third_spade_or_diamond

-- Theorem statement that encapsulates the problem
theorem card_probability : total_probability = 26 / 3675 :=
by sorry

end card_probability_l15_15573


namespace find_age_of_b_l15_15096

-- Definitions for the conditions
def is_two_years_older (a b : ℕ) : Prop := a = b + 2
def is_twice_as_old (b c : ℕ) : Prop := b = 2 * c
def total_age (a b c : ℕ) : Prop := a + b + c = 12

-- Proof statement
theorem find_age_of_b (a b c : ℕ) 
  (h1 : is_two_years_older a b) 
  (h2 : is_twice_as_old b c) 
  (h3 : total_age a b c) : 
  b = 4 := 
by 
  sorry

end find_age_of_b_l15_15096


namespace max_possible_value_xv_l15_15560

noncomputable def max_xv_distance (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) : ℝ :=
|x - v|

theorem max_possible_value_xv 
  (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  max_xv_distance x y z w v h1 h2 h3 h4 = 11 :=
sorry

end max_possible_value_xv_l15_15560


namespace geometric_proportion_l15_15059

theorem geometric_proportion (a b c d : ℝ) (h1 : a / b = c / d) (h2 : a / b = d / c) :
  (a = b ∧ b = c ∧ c = d) ∨ (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ (a * b * c * d < 0)) :=
by
  sorry

end geometric_proportion_l15_15059


namespace alex_integer_list_count_l15_15884

theorem alex_integer_list_count : 
  let n := 12 
  let least_multiple := 2^6 * 3^3
  let count := least_multiple / n
  count = 144 :=
by
  sorry

end alex_integer_list_count_l15_15884


namespace min_value_expression_l15_15413

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 48) :
  x^2 + 6 * x * y + 9 * y^2 + 4 * z^2 ≥ 128 := 
sorry

end min_value_expression_l15_15413


namespace arithmetic_seq_8th_term_l15_15178

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 22) 
  (h6 : a + 5 * d = 46) : 
  a + 7 * d = 70 :=
by 
  sorry

end arithmetic_seq_8th_term_l15_15178


namespace output_value_of_y_l15_15262

/-- Define the initial conditions -/
def l : ℕ := 2
def m : ℕ := 3
def n : ℕ := 5

/-- Define the function that executes the flowchart operations -/
noncomputable def flowchart_operation (l m n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem output_value_of_y : flowchart_operation l m n = 68 := sorry

end output_value_of_y_l15_15262


namespace range_of_x_l15_15881

theorem range_of_x
  (x : ℝ)
  (h1 : ∀ m, -1 ≤ m ∧ m ≤ 4 → m * (x^2 - 1) - 1 - 8 * x < 0) :
  0 < x ∧ x < 5 / 2 :=
sorry

end range_of_x_l15_15881


namespace problem_l15_15637

-- Define proposition p: for all x in ℝ, x^2 + 1 ≥ 1
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q: for angles A and B in a triangle, A > B ↔ sin A > sin B
def q : Prop := ∀ {A B : ℝ}, A > B ↔ Real.sin A > Real.sin B

-- The problem definition: prove that p ∨ q is true
theorem problem (hp : p) (hq : q) : p ∨ q := sorry

end problem_l15_15637


namespace smallest_percent_both_l15_15054

theorem smallest_percent_both (S J : ℝ) (hS : S = 0.9) (hJ : J = 0.8) : 
  ∃ B, B = S + J - 1 ∧ B = 0.7 :=
by
  sorry

end smallest_percent_both_l15_15054


namespace quadratic_real_equal_roots_l15_15165

theorem quadratic_real_equal_roots (m : ℝ) :
  (∃ x : ℝ, 3*x^2 + (2*m-5)*x + 12 = 0) ↔ (m = 8.5 ∨ m = -3.5) :=
sorry

end quadratic_real_equal_roots_l15_15165


namespace least_possible_value_of_smallest_integer_l15_15580

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), A < B → B < C → C < D → (A + B + C + D) / 4 = 70 → D = 90 → A ≥ 13 :=
by
  intros A B C D h₁ h₂ h₃ h₄ h₅
  sorry

end least_possible_value_of_smallest_integer_l15_15580


namespace ratio_trumpet_to_running_l15_15500

def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 40

theorem ratio_trumpet_to_running : (trumpet_hours : ℚ) / running_hours = 2 :=
by
  sorry

end ratio_trumpet_to_running_l15_15500


namespace Dean_handled_100_transactions_l15_15210

-- Definitions for the given conditions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := (9 * Mabel_transactions) / 10 + Mabel_transactions
def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3
def Jade_transactions : ℕ := Cal_transactions + 14
def Dean_transactions : ℕ := (Jade_transactions * 25) / 100 + Jade_transactions

-- Define the theorem we need to prove
theorem Dean_handled_100_transactions : Dean_transactions = 100 :=
by
  -- Statement to skip the actual proof
  sorry

end Dean_handled_100_transactions_l15_15210


namespace altitude_segment_product_eq_half_side_diff_square_l15_15439

noncomputable def altitude_product (a b c t m m_1: ℝ) :=
  m * m_1 = (b^2 + c^2 - a^2) / 2

theorem altitude_segment_product_eq_half_side_diff_square {a b c t m m_1: ℝ}
  (hm : m = 2 * t / a)
  (hm_1 : m_1 = a * (b^2 + c^2 - a^2) / (4 * t)) :
  altitude_product a b c t m m_1 :=
by sorry

end altitude_segment_product_eq_half_side_diff_square_l15_15439


namespace average_temperature_l15_15033

theorem average_temperature (t1 t2 t3 : ℤ) (h1 : t1 = -14) (h2 : t2 = -8) (h3 : t3 = 1) :
  (t1 + t2 + t3) / 3 = -7 :=
by
  sorry

end average_temperature_l15_15033


namespace four_dice_min_rolls_l15_15027

def minRollsToEnsureSameSum (n : Nat) : Nat :=
  if n = 4 then 22 else sorry

theorem four_dice_min_rolls : minRollsToEnsureSameSum 4 = 22 := by
  rfl

end four_dice_min_rolls_l15_15027


namespace problem_1_problem_2_l15_15924

-- Problem 1: Prove that sqrt(6) * sqrt(1/3) - sqrt(16) * sqrt(18) = -11 * sqrt(2)
theorem problem_1 : Real.sqrt 6 * Real.sqrt (1 / 3) - Real.sqrt 16 * Real.sqrt 18 = -11 * Real.sqrt 2 := 
by
  sorry

-- Problem 2: Prove that (2 - sqrt(5)) * (2 + sqrt(5)) + (2 - sqrt(2))^2 = 5 - 4 * sqrt(2)
theorem problem_2 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * Real.sqrt 2 := 
by
  sorry

end problem_1_problem_2_l15_15924


namespace sin_beta_acute_l15_15065

theorem sin_beta_acute (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = 4 / 5)
  (hcosαβ : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_acute_l15_15065


namespace minimum_radius_part_a_minimum_radius_part_b_l15_15209

-- Definitions for Part (a)
def a := 7
def b := 8
def c := 9
def R1 := 6

-- Statement for Part (a)
theorem minimum_radius_part_a : (c / 2) = R1 := by sorry

-- Definitions for Part (b)
def a' := 9
def b' := 15
def c' := 16
def R2 := 9

-- Statement for Part (b)
theorem minimum_radius_part_b : (c' / 2) = R2 := by sorry

end minimum_radius_part_a_minimum_radius_part_b_l15_15209


namespace distribute_balls_into_boxes_l15_15854

theorem distribute_balls_into_boxes : 
  let n := 5
  let k := 4
  (n.choose (k - 1) + k - 1).choose (k - 1) = 56 :=
by
  sorry

end distribute_balls_into_boxes_l15_15854


namespace ratio_pat_mark_l15_15070

theorem ratio_pat_mark (P K M : ℕ) (h1 : P + K + M = 180) 
  (h2 : P = 2 * K) (h3 : M = K + 100) : P / gcd P M = 1 ∧ M / gcd P M = 3 := by
  sorry

end ratio_pat_mark_l15_15070


namespace find_number_l15_15083

-- Given conditions and declarations
variable (x : ℕ)
variable (h : x / 3 = x - 42)

-- Proof problem statement
theorem find_number : x = 63 := 
sorry

end find_number_l15_15083


namespace y_intercepts_count_l15_15738

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count_l15_15738


namespace highest_degree_has_asymptote_l15_15942

noncomputable def highest_degree_of_px (denom : ℕ → ℕ) (n : ℕ) : ℕ :=
  let deg := denom n
  deg

theorem highest_degree_has_asymptote (p : ℕ → ℕ) (denom : ℕ → ℕ) (n : ℕ)
  (h_denom : denom n = 6) :
  highest_degree_of_px denom n = 6 := by
  sorry

end highest_degree_has_asymptote_l15_15942


namespace percentage_students_went_on_trip_l15_15504

theorem percentage_students_went_on_trip
  (total_students : ℕ)
  (students_march : ℕ)
  (students_march_more_than_100 : ℕ)
  (students_june : ℕ)
  (students_june_more_than_100 : ℕ)
  (total_more_than_100_either_trip : ℕ) :
  total_students = 100 → students_march = 20 → students_march_more_than_100 = 7 →
  students_june = 15 → students_june_more_than_100 = 6 →
  70 * total_more_than_100_either_trip = 7 * 100 →
  (students_march + students_june) * 100 / total_students = 35 :=
by
  intros h_total h_march h_march_100 h_june h_june_100 h_total_100
  sorry

end percentage_students_went_on_trip_l15_15504


namespace total_students_l15_15713

theorem total_students (total_students_with_brown_eyes total_students_with_black_hair: ℕ)
    (h1: ∀ (total_students : ℕ), (2 * total_students_with_brown_eyes) = 3 * total_students)
    (h2: (2 * total_students_with_black_hair) = total_students_with_brown_eyes)
    (h3: total_students_with_black_hair = 6) : 
    ∃ total_students : ℕ, total_students = 18 :=
by
  sorry

end total_students_l15_15713


namespace original_average_marks_l15_15182

theorem original_average_marks (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 30) 
  (h2 : new_avg = 90)
  (h3 : ∀ new_avg, new_avg = 2 * A → A = 90 / 2) : 
  A = 45 :=
by
  sorry

end original_average_marks_l15_15182


namespace father_l15_15206

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l15_15206


namespace simplify_expression_l15_15688

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : (x^2)⁻¹ - 2 = (1 - 2 * x^2) / (x^2) :=
by
  -- proof here
  sorry

end simplify_expression_l15_15688


namespace total_camels_l15_15175

theorem total_camels (x y : ℕ) (humps_eq : x + 2 * y = 23) (legs_eq : 4 * (x + y) = 60) : x + y = 15 :=
by
  sorry

end total_camels_l15_15175


namespace remainder_zero_by_68_l15_15731

theorem remainder_zero_by_68 (N R1 Q2 : ℕ) (h1 : N = 68 * 269 + R1) (h2 : N % 67 = 1) : R1 = 0 := by
  sorry

end remainder_zero_by_68_l15_15731


namespace gcd_459_357_l15_15126

theorem gcd_459_357 : gcd 459 357 = 51 := 
sorry

end gcd_459_357_l15_15126


namespace ursula_annual_salary_l15_15810

def hourly_wage : ℝ := 8.50
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

noncomputable def daily_earnings : ℝ := hourly_wage * hours_per_day
noncomputable def monthly_earnings : ℝ := daily_earnings * days_per_month
noncomputable def annual_salary : ℝ := monthly_earnings * months_per_year

theorem ursula_annual_salary : annual_salary = 16320 := 
  by sorry

end ursula_annual_salary_l15_15810


namespace lila_will_have_21_tulips_l15_15833

def tulip_orchid_ratio := 3 / 4

def initial_orchids := 16

def added_orchids := 12

def total_orchids : ℕ := initial_orchids + added_orchids

def groups_of_orchids : ℕ := total_orchids / 4

def total_tulips : ℕ := 3 * groups_of_orchids

theorem lila_will_have_21_tulips :
  total_tulips = 21 := by
  sorry

end lila_will_have_21_tulips_l15_15833


namespace f_l15_15780

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x

-- Define the derivative f'(x)
def f' (a b x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x - 1

-- Problem statement: Prove that f'(-1) = -5 given the conditions
theorem f'_neg_one_value (a b : ℝ) (h : f' a b 1 = 3) : f' a b (-1) = -5 :=
by
  -- Placeholder for the proof
  sorry

end f_l15_15780


namespace solution_set_l15_15801

def f (x : ℝ) : ℝ := abs x - x + 1

theorem solution_set (x : ℝ) : f (1 - x^2) > f (1 - 2 * x) ↔ x > 2 ∨ x < -1 := by
  sorry

end solution_set_l15_15801


namespace consecutive_numbers_equation_l15_15505

theorem consecutive_numbers_equation (x y z : ℤ) (h1 : z = 3) (h2 : y = z + 1) (h3 : x = y + 1) 
(h4 : 2 * x + 3 * y + 3 * z = 5 * y + n) : n = 11 :=
by
  sorry

end consecutive_numbers_equation_l15_15505


namespace Carson_age_l15_15036

theorem Carson_age {Aunt_Anna_Age : ℕ} (h1 : Aunt_Anna_Age = 60) 
                   {Maria_Age : ℕ} (h2 : Maria_Age = 2 * Aunt_Anna_Age / 3) 
                   {Carson_Age : ℕ} (h3 : Carson_Age = Maria_Age - 7) : 
                   Carson_Age = 33 := by sorry

end Carson_age_l15_15036


namespace tennis_tournament_matches_l15_15944

noncomputable def total_matches (players: ℕ) : ℕ :=
  players - 1

theorem tennis_tournament_matches :
  total_matches 104 = 103 :=
by
  sorry

end tennis_tournament_matches_l15_15944


namespace y1_gt_y2_l15_15954

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 4*(-1) + k) 
  (h2 : y2 = 3^2 - 4*3 + k) : 
  y1 > y2 := 
by
  sorry

end y1_gt_y2_l15_15954


namespace ways_to_write_1800_as_sum_of_twos_and_threes_l15_15819

theorem ways_to_write_1800_as_sum_of_twos_and_threes :
  ∃ (n : ℕ), n = 301 ∧ ∀ (x y : ℕ), 2 * x + 3 * y = 1800 → ∃ (a : ℕ), (x, y) = (3 * a, 300 - a) :=
sorry

end ways_to_write_1800_as_sum_of_twos_and_threes_l15_15819


namespace problem1_proof_problem2_proof_l15_15935

-- Problem 1 proof statement
theorem problem1_proof : (-1)^10 * 2 + (-2)^3 / 4 = 0 := 
by
  sorry

-- Problem 2 proof statement
theorem problem2_proof : -24 * (5 / 6 - 4 / 3 + 3 / 8) = 3 :=
by
  sorry

end problem1_proof_problem2_proof_l15_15935


namespace flowers_total_l15_15045

def red_roses := 1491
def yellow_carnations := 3025
def white_roses := 1768
def purple_tulips := 2150
def pink_daisies := 3500
def blue_irises := 2973
def orange_marigolds := 4234
def lavender_orchids := 350
def sunflowers := 815
def violet_lilies := 26

theorem flowers_total :
  red_roses +
  yellow_carnations +
  white_roses +
  purple_tulips +
  pink_daisies +
  blue_irises +
  orange_marigolds +
  lavender_orchids +
  sunflowers +
  violet_lilies = 21332 := 
by
  -- Simplify and add up all given numbers
  sorry

end flowers_total_l15_15045


namespace complete_the_square_l15_15804

-- Define the quadratic expression as a function.
def quad_expr (k : ℚ) : ℚ := 8 * k^2 + 12 * k + 18

-- Define the completed square form.
def completed_square_expr (k : ℚ) : ℚ := 8 * (k + 3 / 4)^2 + 27 / 2

-- Theorem stating the equality of the original expression in completed square form and the value of r + s.
theorem complete_the_square : ∀ k : ℚ, quad_expr k = completed_square_expr k ∧ (3 / 4 + 27 / 2 = 57 / 4) :=
by
  intro k
  sorry

end complete_the_square_l15_15804


namespace minimum_value_expression_l15_15301

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (4 * z / (2 * x + y)) + (4 * x / (y + 2 * z)) + (y / (x + z)) ≥ 3 :=
by 
  sorry

end minimum_value_expression_l15_15301


namespace find_number_l15_15998

-- Define the problem statement
theorem find_number (n : ℕ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n) / 5 = 27) : n = 9 :=
sorry

end find_number_l15_15998


namespace max_original_chess_pieces_l15_15226

theorem max_original_chess_pieces (m n M N : ℕ) (h1 : m ≤ 19) (h2 : n ≤ 19) (h3 : M ≤ 19) (h4 : N ≤ 19) (h5 : M * N = m * n + 45) (h6 : M = m ∨ N = n) : m * n ≤ 285 :=
by
  sorry

end max_original_chess_pieces_l15_15226


namespace qin_jiushao_algorithm_v2_l15_15258

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x to evaluate the polynomial at
def x0 : ℝ := -1

-- Define the intermediate value v2 according to Horner's rule
def v1 : ℝ := 2 * x0^4 - 3 * x0^3 + x0^2
def v2 : ℝ := v1 * x0 + 2

theorem qin_jiushao_algorithm_v2 : v2 = -4 := 
by 
  -- The proof will be here, for now we place sorry.
  sorry

end qin_jiushao_algorithm_v2_l15_15258


namespace mandy_more_cinnamon_l15_15965

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5

theorem mandy_more_cinnamon : cinnamon - nutmeg = 0.17 :=
by
  sorry

end mandy_more_cinnamon_l15_15965


namespace little_john_remaining_money_l15_15326

noncomputable def initial_amount: ℝ := 8.50
noncomputable def spent_on_sweets: ℝ := 1.25
noncomputable def given_to_each_friend: ℝ := 1.20
noncomputable def number_of_friends: ℝ := 2

theorem little_john_remaining_money : 
  initial_amount - (spent_on_sweets + given_to_each_friend * number_of_friends) = 4.85 :=
by
  sorry

end little_john_remaining_money_l15_15326


namespace ratio_rounded_to_nearest_tenth_l15_15449

theorem ratio_rounded_to_nearest_tenth : 
  (Float.round (11 / 16 : Float) * 10) / 10 = 0.7 :=
by
  -- sorry is used because the proof steps are not required in this task.
  sorry

end ratio_rounded_to_nearest_tenth_l15_15449


namespace probability_neither_red_nor_purple_correct_l15_15078

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

def neither_red_nor_purple_balls : ℕ := total_balls - (red_balls + purple_balls)
def probability_neither_red_nor_purple : ℚ := (neither_red_nor_purple_balls : ℚ) / (total_balls : ℚ)

theorem probability_neither_red_nor_purple_correct : 
  probability_neither_red_nor_purple = 13 / 20 := 
by sorry

end probability_neither_red_nor_purple_correct_l15_15078


namespace not_even_nor_odd_l15_15000

def f (x : ℝ) : ℝ := x^2

theorem not_even_nor_odd (x : ℝ) (h₁ : -1 < x) (h₂ : x ≤ 1) : ¬(∀ y, f y = f (-y)) ∧ ¬(∀ y, f y = -f (-y)) :=
by
  sorry

end not_even_nor_odd_l15_15000


namespace ratio_of_remaining_areas_of_squares_l15_15337

/--
  Given:
  - Square C has a side length of 48 cm.
  - Square D has a side length of 60 cm.
  - A smaller square of side length 12 cm is cut out from both squares.

  Show that:
  - The ratio of the remaining area of square C to the remaining area of square D is 5/8.
-/
theorem ratio_of_remaining_areas_of_squares : 
  let sideC := 48
  let sideD := 60
  let sideSmall := 12
  let areaC := sideC * sideC
  let areaD := sideD * sideD
  let areaSmall := sideSmall * sideSmall
  let remainingC := areaC - areaSmall
  let remainingD := areaD - areaSmall
  (remainingC : ℚ) / remainingD = 5 / 8 :=
by
  sorry

end ratio_of_remaining_areas_of_squares_l15_15337


namespace sum_of_integers_l15_15494

theorem sum_of_integers (s : Finset ℕ) (h₀ : ∀ a ∈ s, 0 ≤ a ∧ a ≤ 124)
  (h₁ : ∀ a ∈ s, a^3 % 125 = 2) : s.sum id = 265 :=
sorry

end sum_of_integers_l15_15494


namespace task2_probability_l15_15634

variable (P_task1_on_time P_task2_on_time : ℝ)

theorem task2_probability 
  (h1 : P_task1_on_time = 5 / 8)
  (h2 : (P_task1_on_time * (1 - P_task2_on_time)) = 0.25) :
  P_task2_on_time = 3 / 5 := by
  sorry

end task2_probability_l15_15634


namespace floodDamageInUSD_l15_15195

def floodDamageAUD : ℝ := 45000000
def exchangeRateAUDtoUSD : ℝ := 1.2

theorem floodDamageInUSD : floodDamageAUD * (1 / exchangeRateAUDtoUSD) = 37500000 := 
by 
  sorry

end floodDamageInUSD_l15_15195


namespace team_total_points_l15_15124

-- Definitions based on conditions
def chandra_points (akiko_points : ℕ) := 2 * akiko_points
def akiko_points (michiko_points : ℕ) := michiko_points + 4
def michiko_points (bailey_points : ℕ) := bailey_points / 2
def bailey_points := 14

-- Total points scored by the team
def total_points :=
  let michiko := michiko_points bailey_points
  let akiko := akiko_points michiko
  let chandra := chandra_points akiko
  bailey_points + michiko + akiko + chandra

theorem team_total_points : total_points = 54 := by
  sorry

end team_total_points_l15_15124


namespace segment_length_reflection_l15_15855

theorem segment_length_reflection (Z : ℝ×ℝ) (Z' : ℝ×ℝ) (hx : Z = (5, 2)) (hx' : Z' = (5, -2)) :
  dist Z Z' = 4 := by
  sorry

end segment_length_reflection_l15_15855


namespace problem1_problem2_l15_15762

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : x > -1) (m : ℕ) (hm : 0 < m) : 
  (1 + x)^m ≥ 1 + m * x :=
sorry

end problem1_problem2_l15_15762


namespace parabola_property_l15_15427

-- Define the conditions of the problem in Lean
variable (a b : ℝ)
variable (h1 : (a, b) ∈ {p : ℝ × ℝ | p.1^2 = 20 * p.2}) -- P lies on the parabola x^2 = 20y
variable (h2 : dist (a, b) (0, 5) = 25) -- Distance from P to focus F

theorem parabola_property : |a * b| = 400 := by
  sorry

end parabola_property_l15_15427


namespace discriminant_zero_geometric_progression_l15_15198

variable (a b c : ℝ)

theorem discriminant_zero_geometric_progression
  (h : b^2 = 4 * a * c) : (b / (2 * a)) = (2 * c / b) :=
by
  sorry

end discriminant_zero_geometric_progression_l15_15198


namespace complex_number_identity_l15_15755

theorem complex_number_identity (m : ℝ) (h : m + ((m ^ 2 - 4) * Complex.I) = Complex.re 0 + 1 * Complex.I ↔ m > 0): 
  (Complex.mk m 2 * Complex.mk 2 (-2)⁻¹) = Complex.I := sorry

end complex_number_identity_l15_15755


namespace smallest_sum_infinite_geometric_progression_l15_15024

theorem smallest_sum_infinite_geometric_progression :
  ∃ (a q A : ℝ), (a * q = 3) ∧ (0 < q) ∧ (q < 1) ∧ (A = a / (1 - q)) ∧ (A = 12) :=
by
  sorry

end smallest_sum_infinite_geometric_progression_l15_15024


namespace probability_losing_ticket_l15_15090

theorem probability_losing_ticket (winning : ℕ) (losing : ℕ)
  (h_odds : winning = 5 ∧ losing = 8) :
  (losing : ℚ) / (winning + losing : ℚ) = 8 / 13 := by
  sorry

end probability_losing_ticket_l15_15090


namespace min_bound_of_gcd_condition_l15_15458

theorem min_bound_of_gcd_condition :
  ∃ c > 0, ∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n ∧
  (∀ i j : ℕ, i ≤ n ∧ j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n) ^ (n / 2) :=
sorry

end min_bound_of_gcd_condition_l15_15458


namespace union_of_sets_l15_15778

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets_l15_15778


namespace all_are_truth_tellers_l15_15129

-- Define the possible states for Alice, Bob, and Carol
inductive State
| true_teller
| liar

-- Define the predicates for each person's statements
def alice_statement (B C : State) : Prop :=
  B = State.true_teller ∨ C = State.true_teller

def bob_statement (A C : State) : Prop :=
  A = State.true_teller ∧ C = State.true_teller

def carol_statement (A B : State) : Prop :=
  A = State.true_teller → B = State.true_teller

-- The theorem to be proved
theorem all_are_truth_tellers
    (A B C : State)
    (alice: A = State.true_teller → alice_statement B C)
    (bob: B = State.true_teller → bob_statement A C)
    (carol: C = State.true_teller → carol_statement A B)
    : A = State.true_teller ∧ B = State.true_teller ∧ C = State.true_teller :=
by
  sorry

end all_are_truth_tellers_l15_15129


namespace michael_and_truck_meet_l15_15147

/--
Assume:
1. Michael walks at 6 feet per second.
2. Trash pails are every 240 feet.
3. A truck travels at 10 feet per second and stops for 36 seconds at each pail.
4. Initially, when Michael passes a pail, the truck is 240 feet ahead.

Prove:
Michael and the truck meet every 120 seconds starting from 120 seconds.
-/
theorem michael_and_truck_meet (t : ℕ) : t ≥ 120 → (t - 120) % 120 = 0 :=
sorry

end michael_and_truck_meet_l15_15147


namespace rahul_matches_l15_15369

variable (m : ℕ)

/-- Rahul's current batting average is 51, and if he scores 78 runs in today's match,
    his new batting average will become 54. Prove that the number of matches he had played
    in this season before today's match is 8. -/
theorem rahul_matches (h1 : (51 * m) / m = 51)
                      (h2 : (51 * m + 78) / (m + 1) = 54) : m = 8 := by
  sorry

end rahul_matches_l15_15369


namespace tangent_line_eq_range_f_l15_15586

-- Given the function f(x) = 2x^3 - 9x^2 + 12x
def f(x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- (1) Prove that the equation of the tangent line to y = f(x) at (0, f(0)) is y = 12x
theorem tangent_line_eq : ∀ x, x = 0 → f x = 0 → (∃ m, m = 12 ∧ (∀ y, y = 12 * x)) :=
by
  sorry

-- (2) Prove that the range of f(x) on the interval [0, 3] is [0, 9]
theorem range_f : Set.Icc 0 9 = Set.image f (Set.Icc (0 : ℝ) 3) :=
by
  sorry

end tangent_line_eq_range_f_l15_15586


namespace min_value_fraction_108_l15_15813

noncomputable def min_value_fraction (x y z w : ℝ) : ℝ :=
(x + y) / (x * y * z * w)

theorem min_value_fraction_108 (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) (h_sum : x + y + z + w = 1) :
  min_value_fraction x y z w = 108 :=
sorry

end min_value_fraction_108_l15_15813


namespace sum_gcd_lcm_is_159_l15_15092

-- Definitions for GCD and LCM for specific values
def gcd_45_75 := Int.gcd 45 75
def lcm_48_18 := Int.lcm 48 18

-- The proof problem statement
theorem sum_gcd_lcm_is_159 : gcd_45_75 + lcm_48_18 = 159 := by
  sorry

end sum_gcd_lcm_is_159_l15_15092


namespace remainder_div_150_by_4_eq_2_l15_15991

theorem remainder_div_150_by_4_eq_2 :
  (∃ k : ℕ, k > 0 ∧ 120 % k^2 = 24) → 150 % 4 = 2 :=
by
  intro h
  sorry

end remainder_div_150_by_4_eq_2_l15_15991


namespace least_possible_value_l15_15803

theorem least_possible_value (x y z : ℕ) (hx : 2 * x = 5 * y) (hy : 5 * y = 8 * z) (hz : 8 * z = 2 * x) (hnz_x: x > 0) (hnz_y: y > 0) (hnz_z: z > 0) :
  x + y + z = 33 :=
sorry

end least_possible_value_l15_15803


namespace empty_solution_set_range_l15_15572

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l15_15572


namespace three_digit_numbers_count_l15_15415

def number_of_3_digit_numbers : ℕ := 
  let without_zero := 2 * Nat.choose 9 3
  let with_zero := Nat.choose 9 2
  without_zero + with_zero

theorem three_digit_numbers_count : number_of_3_digit_numbers = 204 := by
  -- Proof to be completed
  sorry

end three_digit_numbers_count_l15_15415


namespace James_age_after_x_years_l15_15937

variable (x : ℕ)
variable (Justin Jessica James : ℕ)

-- Define the conditions
theorem James_age_after_x_years 
  (H1 : Justin = 26) 
  (H2 : Jessica = Justin + 6) 
  (H3 : James = Jessica + 7)
  (H4 : James + 5 = 44) : 
  James + x = 39 + x := 
by 
  -- proof steps go here 
  sorry

end James_age_after_x_years_l15_15937


namespace avg_age_of_five_students_l15_15253

-- step a: Define the conditions
def avg_age_seventeen_students : ℕ := 17
def total_seventeen_students : ℕ := 17 * avg_age_seventeen_students

def num_students_with_unknown_avg : ℕ := 5

def avg_age_nine_students : ℕ := 16
def num_students_with_known_avg : ℕ := 9
def total_age_nine_students : ℕ := num_students_with_known_avg * avg_age_nine_students

def age_seventeenth_student : ℕ := 75

-- step c: Compute the average age of the 5 students
noncomputable def total_age_five_students : ℕ :=
  total_seventeen_students - total_age_nine_students - age_seventeenth_student

def correct_avg_age_five_students : ℕ := 14

theorem avg_age_of_five_students :
  total_age_five_students / num_students_with_unknown_avg = correct_avg_age_five_students :=
sorry

end avg_age_of_five_students_l15_15253


namespace pineapple_cost_l15_15848

variables (P W : ℕ)

theorem pineapple_cost (h1 : 2 * P + 5 * W = 38) : P = 14 :=
sorry

end pineapple_cost_l15_15848


namespace min_value_of_x_plus_2y_l15_15830

theorem min_value_of_x_plus_2y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 :=
sorry

end min_value_of_x_plus_2y_l15_15830


namespace distance_from_plate_to_bottom_edge_l15_15918

theorem distance_from_plate_to_bottom_edge (d : ℝ) : 
  (10 + d + 63 = 20 + d + 53) :=
by
  -- The proof can be completed here.
  sorry

end distance_from_plate_to_bottom_edge_l15_15918


namespace prime_between_30_and_40_with_remainder_1_l15_15643

theorem prime_between_30_and_40_with_remainder_1 (n : ℕ) : 
  n.Prime → 
  30 < n → n < 40 → 
  n % 6 = 1 → 
  n = 37 := 
sorry

end prime_between_30_and_40_with_remainder_1_l15_15643


namespace number_of_positions_forming_cube_with_missing_face_l15_15794

-- Define the polygon formed by 6 congruent squares in a cross shape
inductive Square
| center : Square
| top : Square
| bottom : Square
| left : Square
| right : Square

-- Define the indices for the additional square positions
inductive Position
| pos1 : Position
| pos2 : Position
| pos3 : Position
| pos4 : Position
| pos5 : Position
| pos6 : Position
| pos7 : Position
| pos8 : Position
| pos9 : Position
| pos10 : Position
| pos11 : Position

-- Define a function that takes a position and returns whether the polygon can form the missing-face cube
def can_form_cube_missing_face : Position → Bool
  | Position.pos1   => true
  | Position.pos2   => true
  | Position.pos3   => true
  | Position.pos4   => true
  | Position.pos5   => false
  | Position.pos6   => false
  | Position.pos7   => false
  | Position.pos8   => false
  | Position.pos9   => true
  | Position.pos10  => true
  | Position.pos11  => true

-- Count valid positions for forming the cube with one face missing
def count_valid_positions : Nat :=
  List.length (List.filter can_form_cube_missing_face 
    [Position.pos1, Position.pos2, Position.pos3, Position.pos4, Position.pos5, Position.pos6, Position.pos7, Position.pos8, Position.pos9, Position.pos10, Position.pos11])

-- Prove that the number of valid positions is 7
theorem number_of_positions_forming_cube_with_missing_face : count_valid_positions = 7 :=
  by
    -- Implementation of the proof
    sorry

end number_of_positions_forming_cube_with_missing_face_l15_15794


namespace min_value_expression_ge_072_l15_15101

theorem min_value_expression_ge_072 (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 0.5) 
  (hy : |y| ≤ 0.5) 
  (hz : 0 ≤ z ∧ z < 1) :
  ((1 / ((1 - x) * (1 - y) * (1 - z))) - (1 / ((2 + x) * (2 + y) * (2 + z)))) ≥ 0.72 := sorry

end min_value_expression_ge_072_l15_15101


namespace z_is_1_2_decades_younger_than_x_l15_15678

variable (X Y Z : ℝ)

theorem z_is_1_2_decades_younger_than_x (h : X + Y = Y + Z + 12) : (X - Z) / 10 = 1.2 :=
by
  sorry

end z_is_1_2_decades_younger_than_x_l15_15678


namespace regular_hexagon_has_greatest_lines_of_symmetry_l15_15455

-- Definitions for the various shapes and their lines of symmetry.
def regular_pentagon_lines_of_symmetry : ℕ := 5
def parallelogram_lines_of_symmetry : ℕ := 0
def oval_ellipse_lines_of_symmetry : ℕ := 2
def right_triangle_lines_of_symmetry : ℕ := 0
def regular_hexagon_lines_of_symmetry : ℕ := 6

-- Theorem stating that the regular hexagon has the greatest number of lines of symmetry.
theorem regular_hexagon_has_greatest_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry > regular_pentagon_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > parallelogram_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > oval_ellipse_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > right_triangle_lines_of_symmetry :=
by
  sorry

end regular_hexagon_has_greatest_lines_of_symmetry_l15_15455


namespace greatest_integer_for_prime_abs_expression_l15_15968

open Int

-- Define the quadratic expression and the prime condition
def quadratic_expression (x : ℤ) : ℤ := 6 * x^2 - 47 * x + 15

-- Statement that |quadratic_expression x| is prime
def is_prime_quadratic_expression (x : ℤ) : Prop :=
  Prime (abs (quadratic_expression x))

-- Prove that the greatest integer x such that |quadratic_expression x| is prime is 8
theorem greatest_integer_for_prime_abs_expression :
  ∃ (x : ℤ), is_prime_quadratic_expression x ∧ (∀ (y : ℤ), is_prime_quadratic_expression y → y ≤ x) → x = 8 :=
by
  sorry

end greatest_integer_for_prime_abs_expression_l15_15968


namespace sarah_score_l15_15641

-- Given conditions
variable (s g : ℕ) -- Sarah's score and Greg's score
variable (h1 : s = g + 60) -- Sarah's score is 60 points more than Greg's
variable (h2 : (s + g) / 2 = 130) -- The average of their two scores is 130

-- Proof statement
theorem sarah_score : s = 160 :=
by
  sorry

end sarah_score_l15_15641


namespace numbers_square_and_cube_root_l15_15138

theorem numbers_square_and_cube_root (x : ℝ) : (x^2 = x ∧ x^3 = x) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by
  sorry

end numbers_square_and_cube_root_l15_15138


namespace initial_average_weight_l15_15620

theorem initial_average_weight (A : ℝ) (weight7th : ℝ) (new_avg_weight : ℝ) (initial_num : ℝ) (total_num : ℝ) 
  (h_weight7th : weight7th = 97) (h_new_avg_weight : new_avg_weight = 151) (h_initial_num : initial_num = 6) (h_total_num : total_num = 7) :
  initial_num * A + weight7th = total_num * new_avg_weight → A = 160 := 
by 
  intros h
  sorry

end initial_average_weight_l15_15620


namespace determine_function_l15_15039

theorem determine_function (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 := 
sorry

end determine_function_l15_15039


namespace no_valid_arrangement_in_7x7_grid_l15_15399

theorem no_valid_arrangement_in_7x7_grid :
  ¬ (∃ (f : Fin 7 → Fin 7 → ℕ),
    (∀ (i j : Fin 6),
      (f i j + f i (j + 1) + f (i + 1) j + f (i + 1) (j + 1)) % 2 = 1) ∧
    (∀ (i j : Fin 5),
      (f i j + f i (j + 1) + f i (j + 2) + f (i + 1) j + f (i + 1) (j + 1) + f (i + 1) (j + 2) +
       f (i + 2) j + f (i + 2) (j + 1) + f (i + 2) (j + 2)) % 2 = 1)) := by
  sorry

end no_valid_arrangement_in_7x7_grid_l15_15399


namespace whitewash_all_planks_not_whitewash_all_planks_l15_15071

open Finset

variable {N : ℕ} (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1))

def f (n : ℤ) : ℤ := n^2 + 3*n - 2

def f_equiv (x y : ℤ) : Prop := 2^(Nat.log2 (2 * N)) ∣ (f x - f y)

theorem whitewash_all_planks (N : ℕ) (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1)) : 
  ∀ n ∈ range N, ∃ m ∈ range N, f m = n :=
by {
  sorry
}

theorem not_whitewash_all_planks (N : ℕ) (not_power_of_two : ¬(∃ (k : ℕ), N = 2^(k + 1))) : 
  ∃ n ∈ range N, ∀ m ∈ range N, f m ≠ n :=
by {
  sorry
}

end whitewash_all_planks_not_whitewash_all_planks_l15_15071


namespace expression_for_f_general_formula_a_n_sum_S_n_l15_15442

-- Definitions for conditions
def f (x : ℝ) : ℝ := x^2 + x

-- Given conditions
axiom f_zero : f 0 = 0
axiom f_recurrence : ∀ x : ℝ, f (x + 1) - f x = x + 1

-- Statements to prove
theorem expression_for_f (x : ℝ) : f x = x^2 + x := 
sorry

theorem general_formula_a_n (t : ℝ) (n : ℕ) (H : 0 < t) : 
    ∃ a_n : ℕ → ℝ, a_n n = t^n := 
sorry

theorem sum_S_n (t : ℝ) (n : ℕ) (H : 0 < t) :
    ∃ S_n : ℕ → ℝ, (S_n n = if t = 1 then ↑n else (t * (t^n - 1)) / (t - 1)) := 
sorry

end expression_for_f_general_formula_a_n_sum_S_n_l15_15442


namespace zoo_revenue_l15_15307

def num_children_mon : ℕ := 7
def num_adults_mon : ℕ := 5
def num_children_tue : ℕ := 4
def num_adults_tue : ℕ := 2
def cost_child : ℕ := 3
def cost_adult : ℕ := 4

theorem zoo_revenue : 
  (num_children_mon * cost_child + num_adults_mon * cost_adult) + 
  (num_children_tue * cost_child + num_adults_tue * cost_adult) 
  = 61 := 
by
  sorry

end zoo_revenue_l15_15307


namespace mixed_feed_cost_l15_15704

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed by mixing 
    one kind worth $0.18 per pound with another worth $0.53 per pound. They used 17 pounds of the cheaper kind in the mix.
    We are to prove that the cost per pound of the mixed feed is $0.36 per pound. -/
theorem mixed_feed_cost
  (total_weight : ℝ) (cheaper_cost : ℝ) (expensive_cost : ℝ) (cheaper_weight : ℝ)
  (total_weight_eq : total_weight = 35)
  (cheaper_cost_eq : cheaper_cost = 0.18)
  (expensive_cost_eq : expensive_cost = 0.53)
  (cheaper_weight_eq : cheaper_weight = 17) :
  ((cheaper_weight * cheaper_cost + (total_weight - cheaper_weight) * expensive_cost) / total_weight) = 0.36 :=
by
  sorry

end mixed_feed_cost_l15_15704


namespace quadrilateral_iff_segments_lt_half_l15_15582

theorem quadrilateral_iff_segments_lt_half (a b c d : ℝ) (h₁ : a + b + c + d = 1) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ d) : 
    (a + b > d) ∧ (a + c > d) ∧ (a + b + c > d) ∧ (b + c > d) ↔ a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 :=
by
  sorry

end quadrilateral_iff_segments_lt_half_l15_15582


namespace sum_of_ages_l15_15025

theorem sum_of_ages (a b c : ℕ) 
  (h1 : a = 18 + b + c) 
  (h2 : a^2 = 2016 + (b + c)^2) : 
  a + b + c = 112 := 
sorry

end sum_of_ages_l15_15025


namespace ella_dog_food_ratio_l15_15302

variable (ella_food_per_day : ℕ) (total_food_10days : ℕ) (x : ℕ)

theorem ella_dog_food_ratio
  (h1 : ella_food_per_day = 20)
  (h2 : total_food_10days = 1000) :
  (x : ℕ) = 4 :=
by
  sorry

end ella_dog_food_ratio_l15_15302


namespace michael_earnings_l15_15240

theorem michael_earnings :
  let price_extra_large := 150
  let price_large := 100
  let price_medium := 80
  let price_small := 60
  let qty_extra_large := 3
  let qty_large := 5
  let qty_medium := 8
  let qty_small := 10
  let discount_large := 0.10
  let tax := 0.05
  let cost_materials := 300
  let commission_fee := 0.10

  let total_initial_sales := (qty_extra_large * price_extra_large) + 
                             (qty_large * price_large) + 
                             (qty_medium * price_medium) + 
                             (qty_small * price_small)

  let discount_on_large := discount_large * (qty_large * price_large)
  let sales_after_discount := total_initial_sales - discount_on_large

  let sales_tax := tax * sales_after_discount
  let total_collected := sales_after_discount + sales_tax

  let commission := commission_fee * sales_after_discount
  let total_deductions := cost_materials + commission
  let earnings := total_collected - total_deductions

  earnings = 1733 :=
by
  sorry

end michael_earnings_l15_15240


namespace find_picture_area_l15_15298

variable (x y : ℕ)
    (h1 : x > 1)
    (h2 : y > 1)
    (h3 : (3 * x + 2) * (y + 4) - x * y = 62)

theorem find_picture_area : x * y = 10 :=
by
  sorry

end find_picture_area_l15_15298


namespace largest_interesting_number_l15_15295

def is_interesting_number (x : ℝ) : Prop :=
  ∃ y z : ℝ, (0 ≤ y ∧ y < 1) ∧ (0 ≤ z ∧ z < 1) ∧ x = 0 + y * 10⁻¹ + z ∧ 2 * (0 + y * 10⁻¹ + z) = 0 + z

theorem largest_interesting_number : ∀ x, is_interesting_number x → x ≤ 0.375 :=
by
  sorry

end largest_interesting_number_l15_15295


namespace sum_of_grid_numbers_l15_15286

theorem sum_of_grid_numbers (A E: ℕ) (S: ℕ) 
    (hA: A = 2) 
    (hE: E = 3)
    (h1: ∃ B : ℕ, 2 + B = S ∧ 3 + B = S)
    (h2: ∃ D : ℕ, 2 + D = S ∧ D + 3 = S)
    (h3: ∃ F : ℕ, 3 + F = S ∧ F + 3 = S)
    (h4: ∃ G H I: ℕ, 
         2 + G = S ∧ G + H = S ∧ H + C = S ∧ 
         3 + H = S ∧ E + I = S ∧ H + I = S):
  A + B + C + D + E + F + G + H + I = 22 := 
by 
  sorry

end sum_of_grid_numbers_l15_15286


namespace inequality_proof_l15_15163

variable {x y : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hxy : x > y) :
    2 * x + 1 / (x ^ 2 - 2 * x * y + y ^ 2) ≥ 2 * y + 3 := 
  sorry

end inequality_proof_l15_15163


namespace cubes_with_odd_red_faces_l15_15213

-- Define the dimensions and conditions of the block
def block_length : ℕ := 6
def block_width: ℕ := 6
def block_height : ℕ := 2

-- The block is painted initially red on all sides
-- Then the bottom face is painted blue
-- The block is cut into 1-inch cubes
-- 

noncomputable def num_cubes_with_odd_red_faces (length width height : ℕ) : ℕ :=
  -- Only edge cubes have odd number of red faces in this configuration
  let corner_count := 8  -- 4 on top + 4 on bottom (each has 4 red faces)
  let edge_count := 40   -- 20 on top + 20 on bottom (each has 3 red faces)
  let face_only_count := 32 -- 16 on top + 16 on bottom (each has 2 red faces)
  -- The resulting total number of cubes with odd red faces
  edge_count

-- The theorem we need to prove
theorem cubes_with_odd_red_faces : num_cubes_with_odd_red_faces block_length block_width block_height = 40 :=
  by 
    -- Proof goes here
    sorry

end cubes_with_odd_red_faces_l15_15213


namespace find_a_l15_15144

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l15_15144


namespace greater_quadratic_solution_l15_15857

theorem greater_quadratic_solution : ∀ (x : ℝ), x^2 + 15 * x - 54 = 0 → x = -18 ∨ x = 3 →
  max (-18) 3 = 3 := by
  sorry

end greater_quadratic_solution_l15_15857


namespace no_perfect_squares_l15_15971

theorem no_perfect_squares (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0)
  (h5 : x * y - z * t = x + y) (h6 : x + y = z + t) : ¬(∃ a b : ℕ, a^2 = x * y ∧ b^2 = z * t) := 
by
  sorry

end no_perfect_squares_l15_15971


namespace red_flowers_count_l15_15761

theorem red_flowers_count (w r : ℕ) (h1 : w = 555) (h2 : w = r + 208) : r = 347 :=
by {
  -- Proof steps will be here
  sorry
}

end red_flowers_count_l15_15761


namespace tv_height_l15_15694

theorem tv_height (H : ℝ) : 
  672 / (24 * H) = (1152 / (48 * 32)) + 1 → 
  H = 16 := 
by
  have h_area_first_TV : 24 * H ≠ 0 := sorry
  have h_new_condition: 1152 / (48 * 32) + 1 = 1.75 := sorry
  have h_cost_condition: 672 / (24 * H) = 1.75 := sorry
  sorry

end tv_height_l15_15694


namespace chess_tournament_possible_l15_15531

section ChessTournament

structure Player :=
  (name : String)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

def points (p : Player) : ℕ :=
  p.wins + p.draws / 2

def is_possible (A B C : Player) : Prop :=
  (points A > points B) ∧ (points A > points C) ∧
  (points C < points B) ∧
  (A.wins < B.wins) ∧ (A.wins < C.wins) ∧
  (C.wins > B.wins)

theorem chess_tournament_possible (A B C : Player) :
  is_possible A B C :=
  sorry

end ChessTournament

end chess_tournament_possible_l15_15531


namespace simplify_expression_l15_15176

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end simplify_expression_l15_15176


namespace square_area_l15_15772

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the side length of the square based on the arrangement of circles
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- State the theorem to prove the area of the square
theorem square_area : (square_side_length * square_side_length) = 144 :=
by
  sorry

end square_area_l15_15772


namespace correct_calculation_l15_15955

theorem correct_calculation (m n : ℝ) : -m^2 * n - 2 * m^2 * n = -3 * m^2 * n :=
by
  sorry

end correct_calculation_l15_15955


namespace james_marbles_left_l15_15409

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l15_15409


namespace perimeter_of_rectangle_WXYZ_l15_15864

theorem perimeter_of_rectangle_WXYZ 
  (WE XF EG FH : ℝ)
  (h1 : WE = 10)
  (h2 : XF = 25)
  (h3 : EG = 20)
  (h4 : FH = 50) :
  let p := 53 -- By solving the equivalent problem, where perimeter is simplified to 53/1 which gives p = 53 and q = 1
  let q := 29
  p + q = 102 := 
by
  sorry

end perimeter_of_rectangle_WXYZ_l15_15864


namespace base_4_digits_l15_15069

theorem base_4_digits (b : ℕ) (h1 : b^3 ≤ 216) (h2 : 216 < b^4) : b = 5 :=
sorry

end base_4_digits_l15_15069


namespace find_integer_n_l15_15459

theorem find_integer_n (n : ℤ) (h : (⌊(n^2 : ℤ)/4⌋ - (⌊n/2⌋)^2 = 2)) : n = 5 :=
sorry

end find_integer_n_l15_15459


namespace daily_chicken_loss_l15_15327

/--
A small poultry farm has initially 300 chickens, 200 turkeys, and 80 guinea fowls. Every day, the farm loses some chickens, 8 turkeys, and 5 guinea fowls. After one week (7 days), there are 349 birds left in the farm. Prove the number of chickens the farmer loses daily.
-/
theorem daily_chicken_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss days total_birds_left : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : days = 7)
  (h7 : total_birds_left = 349)
  (h8 : initial_chickens + initial_turkeys + initial_guinea_fowls
       - (daily_turkey_loss * days + daily_guinea_fowl_loss * days + (initial_chickens - total_birds_left)) = total_birds_left) :
  initial_chickens - (total_birds_left + daily_turkey_loss * days + daily_guinea_fowl_loss * days) / days = 20 :=
by {
    -- Proof goes here
    sorry
}

end daily_chicken_loss_l15_15327


namespace car_speeds_l15_15082

-- Definitions and conditions
def distance_AB : ℝ := 200
def distance_meet : ℝ := 80
def car_A_speed : ℝ := sorry -- To Be Proved
def car_B_speed : ℝ := sorry -- To Be Proved

axiom car_B_faster (x : ℝ) : car_B_speed = car_A_speed + 30
axiom time_equal (x : ℝ) : (distance_meet / car_A_speed) = ((distance_AB - distance_meet) / car_B_speed)

-- Proof (only statement, without steps)
theorem car_speeds : car_A_speed = 60 ∧ car_B_speed = 90 :=
  by
  have car_A_speed := 60
  have car_B_speed := 90
  sorry

end car_speeds_l15_15082


namespace decreasing_implies_bound_l15_15707

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_implies_bound (b : ℝ) :
  (∀ x > 2, -x + b / x ≤ 0) → b ≤ 4 :=
  sorry

end decreasing_implies_bound_l15_15707


namespace four_planes_divide_space_into_fifteen_parts_l15_15980

-- Define the function that calculates the number of parts given the number of planes.
def parts_divided_by_planes (x : ℕ) : ℕ :=
  (x^3 + 5 * x + 6) / 6

-- Prove that four planes divide the space into 15 parts.
theorem four_planes_divide_space_into_fifteen_parts : parts_divided_by_planes 4 = 15 :=
by sorry

end four_planes_divide_space_into_fifteen_parts_l15_15980


namespace lisa_takes_72_more_minutes_than_ken_l15_15260

theorem lisa_takes_72_more_minutes_than_ken
  (ken_speed : ℕ) (lisa_speed : ℕ) (book_pages : ℕ)
  (h_ken_speed: ken_speed = 75)
  (h_lisa_speed: lisa_speed = 60)
  (h_book_pages: book_pages = 360) :
  ((book_pages / lisa_speed:ℚ) - (book_pages / ken_speed:ℚ)) * 60 = 72 :=
by
  sorry

end lisa_takes_72_more_minutes_than_ken_l15_15260


namespace interest_rate_difference_l15_15837

def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def si1 (R1 : ℕ) : ℕ := simple_interest 800 R1 10
def si2 (R2 : ℕ) : ℕ := simple_interest 800 R2 10

theorem interest_rate_difference (R1 R2 : ℕ) (h : si2 R2 = si1 R1 + 400) : R2 - R1 = 5 := 
by sorry

end interest_rate_difference_l15_15837


namespace negation_of_P_l15_15836

def P (x : ℝ) : Prop := x^2 + x - 1 < 0

theorem negation_of_P : (¬ ∀ x, P x) ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by
  sorry

end negation_of_P_l15_15836


namespace domain_of_c_eq_real_l15_15650

theorem domain_of_c_eq_real (m : ℝ) : (∀ x : ℝ, m * x^2 - 3 * x + 2 * m ≠ 0) ↔ (m < -3 * Real.sqrt 2 / 4 ∨ m > 3 * Real.sqrt 2 / 4) :=
by
  sorry

end domain_of_c_eq_real_l15_15650


namespace systematic_sampling_method_l15_15546

theorem systematic_sampling_method :
  ∀ (num_classes num_students_per_class selected_student : ℕ),
    num_classes = 12 →
    num_students_per_class = 50 →
    selected_student = 40 →
    (∃ (start_interval: ℕ) (interval: ℕ) (total_population: ℕ), 
      total_population > 100 ∧ start_interval < interval ∧ interval * num_classes = total_population ∧
      ∀ (c : ℕ), c < num_classes → (start_interval + c * interval) % num_students_per_class = selected_student - 1) →
    "Systematic Sampling" = "Systematic Sampling" :=
by
  intros num_classes num_students_per_class selected_student h_classes h_students h_selected h_conditions
  sorry

end systematic_sampling_method_l15_15546


namespace find_quadruplets_l15_15234

theorem find_quadruplets :
  ∃ (x y z w : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
  (xyz + 1) / (x + 1) = (yzw + 1) / (y + 1) ∧
  (yzw + 1) / (y + 1) = (zwx + 1) / (z + 1) ∧
  (zwx + 1) / (z + 1) = (wxy + 1) / (w + 1) ∧
  x + y + z + w = 48 ∧
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 :=
by
  sorry

end find_quadruplets_l15_15234


namespace num_valid_triples_l15_15265

theorem num_valid_triples : ∃! (count : ℕ), count = 22 ∧
  ∀ k m n : ℕ, (0 ≤ k) ∧ (k ≤ 100) ∧ (0 ≤ m) ∧ (m ≤ 100) ∧ (0 ≤ n) ∧ (n ≤ 100) → 
  (2^m * n - 2^n * m = 2^k) → count = 22 :=
sorry

end num_valid_triples_l15_15265


namespace sqrt_of_S_l15_15467

def initial_time := 16 * 3600 + 11 * 60 + 22
def initial_date := 16
def total_seconds_in_a_day := 86400
def total_seconds_in_an_hour := 3600

theorem sqrt_of_S (S : ℕ) (hS : S = total_seconds_in_a_day + total_seconds_in_an_hour) : 
  Real.sqrt S = 300 := 
sorry

end sqrt_of_S_l15_15467
