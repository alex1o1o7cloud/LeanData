import Mathlib

namespace smallest_positive_integer_modulo_l377_37766

theorem smallest_positive_integer_modulo {n : ℕ} (h : 19 * n ≡ 546 [MOD 13]) : n = 11 := by
  sorry

end smallest_positive_integer_modulo_l377_37766


namespace kids_joined_in_l377_37773

-- Define the given conditions
def original : ℕ := 14
def current : ℕ := 36

-- State the goal
theorem kids_joined_in : (current - original = 22) :=
by
  sorry

end kids_joined_in_l377_37773


namespace aleena_vs_bob_distance_l377_37727

theorem aleena_vs_bob_distance :
  let AleenaDistance := 75
  let BobDistance := 60
  AleenaDistance - BobDistance = 15 :=
by
  let AleenaDistance := 75
  let BobDistance := 60
  show AleenaDistance - BobDistance = 15
  sorry

end aleena_vs_bob_distance_l377_37727


namespace convert_quadratic_l377_37716

theorem convert_quadratic :
  ∀ x : ℝ, (x^2 + 2*x + 4) = ((x + 1)^2 + 3) :=
by
  sorry

end convert_quadratic_l377_37716


namespace find_value_l377_37752

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity_condition : ∀ x : ℝ, f (2 + x) = f (-x)
axiom value_at_half : f (1/2) = 1/2

theorem find_value : f (2023 / 2) = 1/2 := by
  sorry

end find_value_l377_37752


namespace peg_stickers_total_l377_37746

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end peg_stickers_total_l377_37746


namespace abs_neg_six_l377_37719

theorem abs_neg_six : |(-6)| = 6 := by
  sorry

end abs_neg_six_l377_37719


namespace jaylen_dog_food_consumption_l377_37748

theorem jaylen_dog_food_consumption :
  ∀ (morning evening daily_consumption total_food : ℕ)
  (days : ℕ),
  (morning = evening) →
  (total_food = 32) →
  (days = 16) →
  (daily_consumption = total_food / days) →
  (morning + evening = daily_consumption) →
  morning = 1 := by
  intros morning evening daily_consumption total_food days h_eq h_total h_days h_daily h_sum
  sorry

end jaylen_dog_food_consumption_l377_37748


namespace smallest_value_of_x_l377_37703

theorem smallest_value_of_x (x : ℝ) (h : 4 * x^2 - 20 * x + 24 = 0) : x = 2 :=
    sorry

end smallest_value_of_x_l377_37703


namespace cherries_cost_l377_37701

def cost_per_kg (total_cost kilograms : ℕ) : ℕ :=
  total_cost / kilograms

theorem cherries_cost 
  (genevieve_amount : ℕ) 
  (short_amount : ℕ)
  (total_kilograms : ℕ) 
  (total_cost : ℕ := genevieve_amount + short_amount) 
  (cost : ℕ := cost_per_kg total_cost total_kilograms) : 
  cost = 8 :=
by
  have h1 : genevieve_amount = 1600 := by sorry
  have h2 : short_amount = 400 := by sorry
  have h3 : total_kilograms = 250 := by sorry
  sorry

end cherries_cost_l377_37701


namespace monotonic_intervals_range_of_k_l377_37738

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

-- Conditions: a > 0
variables (a : ℝ) (h_a : 0 < a)

-- Part (1): Monotonic Intervals
theorem monotonic_intervals :
  (∀ x, f x a < f (x + 1) a ↔ x < 0 ∨ a < x) ∧
  (∀ x, f (x + 1) a < f x a ↔ 0 < x ∧ x < a) :=
  sorry

-- Part (2): Range of k
theorem range_of_k (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  (f x1 a - f x2 a < k * a^3) ↔ k ≥ -1/6 :=
  sorry

end monotonic_intervals_range_of_k_l377_37738


namespace total_sequins_is_162_l377_37795

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l377_37795


namespace heaviest_weight_is_aq3_l377_37799

variable (a q : ℝ) (h : 0 < a) (hq : 1 < q)

theorem heaviest_weight_is_aq3 :
  let w1 := a
  let w2 := a * q
  let w3 := a * q^2
  let w4 := a * q^3
  w4 > w3 ∧ w4 > w2 ∧ w4 > w1 ∧ w1 + w4 > w2 + w3 :=
by
  sorry

end heaviest_weight_is_aq3_l377_37799


namespace relationship_between_a_and_b_l377_37768

-- Define the given linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) * x + 1

-- Formalize the relationship between a and b given the points and the linear function
theorem relationship_between_a_and_b (a b k : ℝ) 
  (hP : a = linear_function k (-4))
  (hQ : b = linear_function k 2) :
  a < b := 
by
  sorry  -- Proof to be filled in by the theorem prover

end relationship_between_a_and_b_l377_37768


namespace find_x_and_verify_l377_37737

theorem find_x_and_verify (x : ℤ) (h : (x - 14) / 10 = 4) : (x - 5) / 7 = 7 := 
by 
  sorry

end find_x_and_verify_l377_37737


namespace min_value_of_expression_l377_37797

theorem min_value_of_expression : ∃ x : ℝ, (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 :=
by
  sorry

end min_value_of_expression_l377_37797


namespace tom_average_speed_l377_37753

theorem tom_average_speed 
  (d1 d2 : ℝ) (s1 s2 t1 t2 : ℝ)
  (h_d1 : d1 = 30) 
  (h_d2 : d2 = 50) 
  (h_s1 : s1 = 30) 
  (h_s2 : s2 = 50) 
  (h_t1 : t1 = d1 / s1) 
  (h_t2 : t2 = d2 / s2)
  (h_total_distance : d1 + d2 = 80) 
  (h_total_time : t1 + t2 = 2) :
  (d1 + d2) / (t1 + t2) = 40 := 
by {
  sorry
}

end tom_average_speed_l377_37753


namespace geometric_sum_4500_l377_37712

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l377_37712


namespace white_marbles_in_C_equals_15_l377_37763

variables (A_red A_yellow B_green B_yellow C_yellow : ℕ) (w : ℕ)

-- Conditions from the problem
def conditions : Prop :=
  A_red = 4 ∧ A_yellow = 2 ∧
  B_green = 6 ∧ B_yellow = 1 ∧
  C_yellow = 9 ∧
  (A_red - A_yellow = 2) ∧
  (B_green - B_yellow = 5) ∧
  (w - C_yellow = 6)

-- Proving w = 15 given the conditions
theorem white_marbles_in_C_equals_15 (h : conditions A_red A_yellow B_green B_yellow C_yellow w) : w = 15 :=
  sorry

end white_marbles_in_C_equals_15_l377_37763


namespace percentage_of_non_honda_red_cars_l377_37725

/-- 
Total car population in Chennai is 9000.
Honda cars in Chennai is 5000.
Out of every 100 Honda cars, 90 are red.
60% of the total car population is red.
Prove that the percentage of non-Honda cars that are red is 22.5%.
--/
theorem percentage_of_non_honda_red_cars 
  (total_cars : ℕ) (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) (total_red_ratio : ℚ) 
  (h : total_cars = 9000) 
  (h1 : honda_cars = 5000) 
  (h2 : red_honda_ratio = 90 / 100) 
  (h3 : total_red_ratio = 60 / 100) : 
  (900 / (9000 - 5000) * 100 = 22.5) := 
sorry

end percentage_of_non_honda_red_cars_l377_37725


namespace b_share_of_earnings_l377_37764

-- Definitions derived from conditions
def work_rate_a := 1 / 6
def work_rate_b := 1 / 8
def work_rate_c := 1 / 12
def total_earnings := 1170

-- Mathematically equivalent Lean statement
theorem b_share_of_earnings : 
  (work_rate_b / (work_rate_a + work_rate_b + work_rate_c)) * total_earnings = 390 := 
by
  sorry

end b_share_of_earnings_l377_37764


namespace books_sold_on_tuesday_l377_37714

theorem books_sold_on_tuesday (total_stock : ℕ) (monday_sold : ℕ) (wednesday_sold : ℕ)
  (thursday_sold : ℕ) (friday_sold : ℕ) (percent_unsold : ℚ) (tuesday_sold : ℕ) :
  total_stock = 1100 →
  monday_sold = 75 →
  wednesday_sold = 64 →
  thursday_sold = 78 →
  friday_sold = 135 →
  percent_unsold = 63.45 →
  tuesday_sold = total_stock - (monday_sold + wednesday_sold + thursday_sold + friday_sold + (total_stock * percent_unsold / 100)) :=
by sorry

end books_sold_on_tuesday_l377_37714


namespace tv_price_with_tax_l377_37742

-- Define the original price of the TV
def originalPrice : ℝ := 1700

-- Define the value-added tax rate
def taxRate : ℝ := 0.15

-- Calculate the total price including tax
theorem tv_price_with_tax : originalPrice * (1 + taxRate) = 1955 :=
by
  sorry

end tv_price_with_tax_l377_37742


namespace shortest_chord_intercept_l377_37757

theorem shortest_chord_intercept (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 3 → x + m * y - m - 1 = 0 → m = 1) :=
sorry

end shortest_chord_intercept_l377_37757


namespace badger_hid_35_l377_37724

-- Define the variables
variables (h_b h_f x : ℕ)

-- Define the conditions based on the problem
def badger_hides : Prop := 5 * h_b = x
def fox_hides : Prop := 7 * h_f = x
def fewer_holes : Prop := h_b = h_f + 2

-- The main theorem to prove the badger hid 35 walnuts
theorem badger_hid_35 (h_b h_f x : ℕ) :
  badger_hides h_b x ∧ fox_hides h_f x ∧ fewer_holes h_b h_f → x = 35 :=
by sorry

end badger_hid_35_l377_37724


namespace carla_sheep_l377_37721

theorem carla_sheep (T : ℝ) (pen_sheep wilderness_sheep : ℝ) 
(h1: 0.90 * T = 81) (h2: pen_sheep = 81) 
(h3: wilderness_sheep = 0.10 * T) : wilderness_sheep = 9 :=
sorry

end carla_sheep_l377_37721


namespace determinant_scaled_l377_37731

-- Define the initial determinant condition
def init_det (x y z w : ℝ) : Prop :=
  x * w - y * z = -3

-- Define the scaled determinant
def scaled_det (x y z w : ℝ) : ℝ :=
  3 * x * (3 * w) - 3 * y * (3 * z)

-- State the theorem we want to prove
theorem determinant_scaled (x y z w : ℝ) (h : init_det x y z w) :
  scaled_det x y z w = -27 :=
by
  sorry

end determinant_scaled_l377_37731


namespace value_of_f_csc_squared_l377_37720

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 ∧ x ≠ 1 then 1 / x else 0

lemma csc_sq_identity (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  (f (x / (x - 1)) = 1 / x) := 
  by sorry

theorem value_of_f_csc_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f ((1 / (Real.sin t) ^ 2)) = - (Real.cos t) ^ 2 :=
  by sorry

end value_of_f_csc_squared_l377_37720


namespace laser_total_distance_l377_37700

noncomputable def laser_path_distance : ℝ :=
  let A := (2, 4)
  let B := (2, -4)
  let C := (-2, -4)
  let D := (8, 4)
  let distance (p q : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B + distance B C + distance C D

theorem laser_total_distance :
  laser_path_distance = 12 + 2 * Real.sqrt 41 :=
by sorry

end laser_total_distance_l377_37700


namespace total_cupcakes_l377_37711

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (h1 : children = 8) (h2 : cupcakes_per_child = 12) : children * cupcakes_per_child = 96 :=
by
  sorry

end total_cupcakes_l377_37711


namespace ball_bounces_to_less_than_two_feet_l377_37783

noncomputable def bounce_height (n : ℕ) : ℝ := 20 * (3 / 4) ^ n

theorem ball_bounces_to_less_than_two_feet : ∃ k : ℕ, bounce_height k < 2 ∧ k = 7 :=
by
  -- We need to show that bounce_height k < 2 when k = 7
  sorry

end ball_bounces_to_less_than_two_feet_l377_37783


namespace min_value_of_expression_l377_37791

theorem min_value_of_expression (a b : ℝ) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  42 + b^2 + 1 / (a * b) ≥ 17 / 2 := 
sorry

end min_value_of_expression_l377_37791


namespace hawks_loss_percentage_is_30_l377_37774

-- Define the variables and the conditions
def matches_won (x : ℕ) : ℕ := 7 * x
def matches_lost (x : ℕ) : ℕ := 3 * x
def total_matches (x : ℕ) : ℕ := matches_won x + matches_lost x
def percent_lost (x : ℕ) : ℕ := (matches_lost x * 100) / total_matches x

-- The goal statement in Lean 4
theorem hawks_loss_percentage_is_30 (x : ℕ) (h : x > 0) : percent_lost x = 30 :=
by sorry

end hawks_loss_percentage_is_30_l377_37774


namespace proof_problem_l377_37781

noncomputable def problem (x y : ℝ) : ℝ :=
  let A := 2 * x + y
  let B := 2 * x - y
  (A ^ 2 - B ^ 2) * (x - 2 * y)

theorem proof_problem : problem (-1) 2 = 80 := by
  sorry

end proof_problem_l377_37781


namespace efficiency_difference_l377_37756

variables (Rp Rq : ℚ)

-- Given conditions
def p_rate := Rp = 1 / 21
def combined_rate := Rp + Rq = 1 / 11

-- Define the percentage efficiency difference
def percentage_difference := (Rp - Rq) / Rq * 100

-- Main statement to prove
theorem efficiency_difference : 
  p_rate Rp ∧ 
  combined_rate Rp Rq → 
  percentage_difference Rp Rq = 10 :=
sorry

end efficiency_difference_l377_37756


namespace find_sum_l377_37741

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l377_37741


namespace log_expression_eval_find_m_from_conditions_l377_37728

-- (1) Prove that lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3.
theorem log_expression_eval : 
  Real.logb 10 (5^2) + (2 / 3) * Real.logb 10 8 + Real.logb 10 5 * Real.logb 10 20 + (Real.logb 10 2)^2 = 3 := 
sorry

-- (2) Given 2^a = 5^b = m and 1/a + 1/b = 2, prove that m = sqrt(10).
theorem find_m_from_conditions (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 :=
sorry

end log_expression_eval_find_m_from_conditions_l377_37728


namespace number_of_young_teachers_selected_l377_37785

theorem number_of_young_teachers_selected 
  (total_teachers elderly_teachers middle_aged_teachers young_teachers sample_size : ℕ)
  (h_total: total_teachers = 200)
  (h_elderly: elderly_teachers = 25)
  (h_middle_aged: middle_aged_teachers = 75)
  (h_young: young_teachers = 100)
  (h_sample_size: sample_size = 40)
  : young_teachers * sample_size / total_teachers = 20 := 
sorry

end number_of_young_teachers_selected_l377_37785


namespace quadratic_equation_factored_form_l377_37771

theorem quadratic_equation_factored_form : 
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x - 3)^2 = 15 := 
by 
  sorry

end quadratic_equation_factored_form_l377_37771


namespace smallest_successive_number_l377_37760

theorem smallest_successive_number :
  ∃ n : ℕ, n * (n + 1) * (n + 2) = 1059460 ∧ ∀ m : ℕ, m * (m + 1) * (m + 2) = 1059460 → n ≤ m :=
sorry

end smallest_successive_number_l377_37760


namespace sum_of_first_8_terms_l377_37754

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Given conditions
def c1 (a : ℕ → ℝ) : Prop := geometric_sequence a 2
def c2 (a : ℕ → ℝ) : Prop := sum_of_first_n_terms a 4 = 1

-- The statement to prove
theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : c1 a) (h2 : c2 a) : sum_of_first_n_terms a 8 = 17 :=
by
  sorry

end sum_of_first_8_terms_l377_37754


namespace area_of_new_triangle_geq_twice_sum_of_areas_l377_37707

noncomputable def area_of_triangle (a b c : ℝ) (alpha : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin alpha)

theorem area_of_new_triangle_geq_twice_sum_of_areas
  (a1 b1 c a2 b2 alpha : ℝ)
  (h1 : a1 <= b1) (h2 : b1 <= c) (h3 : a2 <= b2) (h4 : b2 <= c) :
  let α_1 := Real.arcsin ((a1 + a2) / (2 * c))
  let area1 := area_of_triangle a1 b1 c alpha
  let area2 := area_of_triangle a2 b2 c alpha
  let area_new := area_of_triangle (a1 + a2) (b1 + b2) (2 * c) α_1
  area_new >= 2 * (area1 + area2) :=
sorry

end area_of_new_triangle_geq_twice_sum_of_areas_l377_37707


namespace vivian_mail_june_l377_37702

theorem vivian_mail_june :
  ∀ (m_apr m_may m_jul m_aug : ℕ),
  m_apr = 5 →
  m_may = 10 →
  m_jul = 40 →
  ∃ m_jun : ℕ,
  ∃ pattern : ℕ → ℕ,
  (pattern m_apr = m_may) →
  (pattern m_may = m_jun) →
  (pattern m_jun = m_jul) →
  (pattern m_jul = m_aug) →
  (m_aug = 80) →
  pattern m_may = m_may * 2 →
  pattern m_jun = m_jun * 2 →
  pattern m_jun = 20 :=
by
  sorry

end vivian_mail_june_l377_37702


namespace not_product_of_consecutives_l377_37704

theorem not_product_of_consecutives (n k : ℕ) : 
  ¬ (∃ a b: ℕ, a + 1 = b ∧ (2 * n^(3 * k) + 4 * n^k + 10 = a * b)) :=
by sorry

end not_product_of_consecutives_l377_37704


namespace right_triangle_area_l377_37733

theorem right_triangle_area (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) :
  (1/2) * (a : ℝ) * b = 30 :=
by
  sorry

end right_triangle_area_l377_37733


namespace number_of_camels_l377_37758

theorem number_of_camels (hens goats keepers camel_feet heads total_feet : ℕ)
  (h_hens : hens = 50) (h_goats : goats = 45) (h_keepers : keepers = 15)
  (h_feet_diff : total_feet = heads + 224)
  (h_heads : heads = hens + goats + keepers)
  (h_hens_feet : hens * 2 = 100)
  (h_goats_feet : goats * 4 = 180)
  (h_keepers_feet : keepers * 2 = 30)
  (h_camels_feet : camel_feet = 24)
  (h_total_feet : total_feet = 334)
  (h_feet_without_camels : 100 + 180 + 30 = 310) :
  camel_feet / 4 = 6 := sorry

end number_of_camels_l377_37758


namespace solve_for_x_l377_37793

theorem solve_for_x (x : ℝ) (h : (2 + x) / (4 + x) = (3 + x) / (7 + x)) : x = -1 :=
by {
  sorry
}

end solve_for_x_l377_37793


namespace chess_tournament_games_l377_37734

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games :
  number_of_games 20 = 190 :=
by
  sorry

end chess_tournament_games_l377_37734


namespace smallest_c_for_3_in_range_l377_37747

theorem smallest_c_for_3_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, (x^2 - 6 * x + c) = 3) ↔ (c ≥ 12) :=
by {
  sorry
}

end smallest_c_for_3_in_range_l377_37747


namespace susan_took_longer_l377_37784
variables (M S J T x : ℕ)
theorem susan_took_longer (h1 : M = 2 * S)
                         (h2 : S = J + x)
                         (h3 : J = 30)
                         (h4 : T = M - 7)
                         (h5 : M + S + J + T = 223) : x = 10 :=
sorry

end susan_took_longer_l377_37784


namespace calculation_result_l377_37710

theorem calculation_result : 
  2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := 
by 
  sorry

end calculation_result_l377_37710


namespace evaluate_expression_l377_37713

variable (x : ℝ)
variable (hx : x^3 - 3 * x = 6)

theorem evaluate_expression : x^7 - 27 * x^2 = 9 * (x + 1) * (x + 6) :=
by
  sorry

end evaluate_expression_l377_37713


namespace turtles_in_lake_l377_37772

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l377_37772


namespace billy_sleep_total_hours_l377_37739

theorem billy_sleep_total_hours : 
    let first_night := 6
    let second_night := 2 * first_night
    let third_night := second_night - 3
    let fourth_night := 3 * third_night
    first_night + second_night + third_night + fourth_night = 54
  := by
    sorry

end billy_sleep_total_hours_l377_37739


namespace transformed_interval_l377_37770

noncomputable def transformation (x : ℝ) : ℝ := 8 * x - 2

theorem transformed_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2 ≤ transformation x ∧ transformation x ≤ 6 := by
  intro x h
  unfold transformation
  sorry

end transformed_interval_l377_37770


namespace problem_solution_l377_37732

theorem problem_solution (a b c d : ℝ) 
  (h1 : 3 * a + 2 * b + 4 * c + 8 * d = 40)
  (h2 : 4 * (d + c) = b)
  (h3 : 2 * b + 2 * c = a)
  (h4 : c + 1 = d) :
  a * b * c * d = 0 :=
sorry

end problem_solution_l377_37732


namespace max_g_8_l377_37745

noncomputable def g (x : ℝ) : ℝ := sorry -- To be filled with the specific polynomial

theorem max_g_8 (g : ℝ → ℝ)
  (h_nonneg : ∀ x, 0 ≤ g x)
  (h4 : g 4 = 16)
  (h16 : g 16 = 1024) : g 8 ≤ 128 :=
sorry

end max_g_8_l377_37745


namespace alice_walk_time_l377_37794

theorem alice_walk_time (bob_time : ℝ) 
  (bob_distance : ℝ) 
  (alice_distance1 : ℝ) 
  (alice_distance2 : ℝ) 
  (time_ratio : ℝ) 
  (expected_alice_time : ℝ) :
  bob_time = 36 →
  bob_distance = 6 →
  alice_distance1 = 4 →
  alice_distance2 = 7 →
  time_ratio = 1 / 3 →
  expected_alice_time = 21 →
  (expected_alice_time = alice_distance2 / (alice_distance1 / (bob_time * time_ratio))) := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h3, h5]
  have h_speed : ℝ := alice_distance1 / (bob_time * time_ratio)
  rw [h4, h6]
  linarith [h_speed]

end alice_walk_time_l377_37794


namespace hh3_value_l377_37729

noncomputable def h (x : ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x - 1

theorem hh3_value : h (h 3) = 3406935 := by
  sorry

end hh3_value_l377_37729


namespace ratio_of_small_square_to_shaded_area_l377_37792

theorem ratio_of_small_square_to_shaded_area :
  let small_square_area := 2 * 2
  let large_square_area := 5 * 5
  let shaded_area := (large_square_area / 2) - (small_square_area / 2)
  (small_square_area : ℚ) / shaded_area = 8 / 21 :=
by
  sorry

end ratio_of_small_square_to_shaded_area_l377_37792


namespace chord_length_l377_37769

noncomputable def circle_center (c: ℝ × ℝ) (r: ℝ): Prop := 
  ∃ x y: ℝ, 
    (x - c.1)^2 + (y - c.2)^2 = r^2

noncomputable def line_equation (a b c: ℝ): Prop := 
  ∀ x y: ℝ, 
    a*x + b*y + c = 0

theorem chord_length (a: ℝ): 
  circle_center (2, 1) 2 ∧ line_equation a 1 (-5) ∧
  ∃(chord_len: ℝ), chord_len = 4 → 
  a = 2 :=
by
  sorry

end chord_length_l377_37769


namespace cassidy_posters_l377_37776

theorem cassidy_posters (p_two_years_ago : ℕ) (p_double : ℕ) (p_current : ℕ) (p_added : ℕ) 
    (h1 : p_two_years_ago = 14) 
    (h2 : p_double = 2 * p_two_years_ago)
    (h3 : p_current = 22)
    (h4 : p_added = p_double - p_current) : 
    p_added = 6 := 
by
  sorry

end cassidy_posters_l377_37776


namespace tangent_line_circle_l377_37726

theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*y = 0 → y = a) → (a = 0 ∨ a = 2) :=
by
  sorry

end tangent_line_circle_l377_37726


namespace general_formula_compare_Tn_l377_37730

open scoped BigOperators

-- Define the sequence {a_n} and its sum S_n
noncomputable def aSeq (n : ℕ) : ℕ := n + 1
noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range n, aSeq (k + 1)

-- Given condition
axiom given_condition (n : ℕ) : 2 * S n = (aSeq n - 1) * (aSeq n + 2)

-- Prove the general formula of the sequence
theorem general_formula (n : ℕ) : aSeq n = n + 1 :=
by
  sorry  -- proof

-- Define T_n sequence
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (k - 1) * 2^k / (k * aSeq k)

-- Compare T_n with the given expression
theorem compare_Tn (n : ℕ) : 
  if n < 17 then T n < (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else if n = 17 then T n = (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else T n > (2^(n+1)*(18-n)-2*n-2)/(n+1) :=
by
  sorry  -- proof

end general_formula_compare_Tn_l377_37730


namespace first_even_number_of_8_sum_424_l377_37744

theorem first_even_number_of_8_sum_424 (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + 
                   (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) : x = 46 :=
by sorry

end first_even_number_of_8_sum_424_l377_37744


namespace count_whole_numbers_in_interval_l377_37775

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end count_whole_numbers_in_interval_l377_37775


namespace tangent_line_equation_at_point_l377_37715

-- Defining the function and the point
def f (x : ℝ) : ℝ := x^2 + 2 * x
def point : ℝ × ℝ := (1, 3)

-- Main theorem stating the tangent line equation at the given point
theorem tangent_line_equation_at_point : 
  ∃ m b, (m = (2 * 1 + 2)) ∧ 
         (b = (3 - m * 1)) ∧ 
         (∀ x y, y = f x → y = m * x + b → 4 * x - y - 1 = 0) :=
by
  -- Proof is omitted and can be filled in later
  sorry

end tangent_line_equation_at_point_l377_37715


namespace percentage_of_childrens_books_l377_37777

/-- Conditions: 
- There are 160 books in total.
- 104 of them are for adults.
Prove that the percentage of books intended for children is 35%. --/
theorem percentage_of_childrens_books (total_books : ℕ) (adult_books : ℕ) 
  (h_total : total_books = 160) (h_adult : adult_books = 104) :
  (160 - 104) / 160 * 100 = 35 := 
by {
  sorry -- Proof skipped
}

end percentage_of_childrens_books_l377_37777


namespace range_of_a_l377_37722

variable {a b c d : ℝ}

theorem range_of_a (h1 : a + b + c + d = 3) (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l377_37722


namespace mary_cut_roses_l377_37762

theorem mary_cut_roses (initial_roses add_roses total_roses : ℕ) (h1 : initial_roses = 6) (h2 : total_roses = 16) (h3 : total_roses = initial_roses + add_roses) : add_roses = 10 :=
by
  sorry

end mary_cut_roses_l377_37762


namespace isosceles_right_triangle_ratio_l377_37790

theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_ratio_l377_37790


namespace min_value_inverse_sum_l377_37755

theorem min_value_inverse_sum (a m n : ℝ) (a_pos : 0 < a) (a_ne_one : a ≠ 1) (mn_pos : 0 < m * n) :
  (a^(1-1) = 1) ∧ (m + n = 1) → (1/m + 1/n) = 4 :=
by
  sorry

end min_value_inverse_sum_l377_37755


namespace angles_with_same_terminal_side_l377_37759

theorem angles_with_same_terminal_side (k : ℤ) :
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} = 
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + (-460 % 360)} :=
by sorry

end angles_with_same_terminal_side_l377_37759


namespace cos_60_eq_sqrt3_div_2_l377_37779

theorem cos_60_eq_sqrt3_div_2 : Real.cos (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry

end cos_60_eq_sqrt3_div_2_l377_37779


namespace willie_stickers_l377_37705

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 124 → given_stickers = 23 → remaining_stickers = initial_stickers - given_stickers → remaining_stickers = 101 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining.trans rfl

end willie_stickers_l377_37705


namespace price_of_75_cans_l377_37717

/-- The price of 75 cans of a certain brand of soda purchased in 24-can cases,
    given the regular price per can is $0.15 and a 10% discount is applied when
    purchased in 24-can cases, is $10.125.
-/
theorem price_of_75_cans (regular_price : ℝ) (discount : ℝ) (cases_needed : ℕ) (remaining_cans : ℕ) 
  (discounted_price : ℝ) (total_price : ℝ) :
  regular_price = 0.15 →
  discount = 0.10 →
  discounted_price = regular_price - (discount * regular_price) →
  cases_needed = 75 / 24 ∧ remaining_cans = 75 % 24 →
  total_price = (cases_needed * 24 + remaining_cans) * discounted_price →
  total_price = 10.125 :=
by
  sorry

end price_of_75_cans_l377_37717


namespace Sabrina_pencils_l377_37750

variable (S : ℕ) (J : ℕ)

theorem Sabrina_pencils (h1 : S + J = 50) (h2 : J = 2 * S + 8) :
  S = 14 :=
by
  sorry

end Sabrina_pencils_l377_37750


namespace maximum_sum_is_42_l377_37708

-- Definitions according to the conditions in the problem

def initial_faces : ℕ := 7 -- 2 pentagonal + 5 rectangular
def initial_vertices : ℕ := 10 -- 5 at the top and 5 at the bottom
def initial_edges : ℕ := 15 -- 5 for each pentagon and 5 linking them

def added_faces : ℕ := 5 -- 5 new triangular faces
def added_vertices : ℕ := 1 -- 1 new vertex at the apex of the pyramid
def added_edges : ℕ := 5 -- 5 new edges connecting the new vertex to the pentagon's vertices

-- New quantities after adding the pyramid
def new_faces : ℕ := initial_faces - 1 + added_faces
def new_vertices : ℕ := initial_vertices + added_vertices
def new_edges : ℕ := initial_edges + added_edges

-- Sum of the new shape's characteristics
def sum_faces_vertices_edges : ℕ := new_faces + new_vertices + new_edges

-- Statement to be proved
theorem maximum_sum_is_42 : sum_faces_vertices_edges = 42 := by
  sorry

end maximum_sum_is_42_l377_37708


namespace number_multiplied_value_l377_37778

theorem number_multiplied_value (x : ℝ) :
  (4 / 6) * x = 8 → x = 12 :=
by
  sorry

end number_multiplied_value_l377_37778


namespace total_revenue_correct_l377_37723

noncomputable def total_revenue : ℚ := 
  let revenue_v1 := 23 * 5 * 0.50
  let revenue_v2 := 28 * 6 * 0.60
  let revenue_v3 := 35 * 7 * 0.50
  let revenue_v4 := 43 * 8 * 0.60
  let revenue_v5 := 50 * 9 * 0.50
  let revenue_v6 := 64 * 10 * 0.60
  revenue_v1 + revenue_v2 + revenue_v3 + revenue_v4 + revenue_v5 + revenue_v6

theorem total_revenue_correct : total_revenue = 1096.20 := 
by
  sorry

end total_revenue_correct_l377_37723


namespace volume_of_sphere_l377_37765

noncomputable def cuboid_volume (a b c : ℝ) := a * b * c

noncomputable def sphere_volume (r : ℝ) := (4/3) * Real.pi * r^3

theorem volume_of_sphere
  (a b c : ℝ) 
  (sphere_radius : ℝ)
  (h1 : a = 1)
  (h2 : b = Real.sqrt 3)
  (h3 : c = 2)
  (h4 : sphere_radius = Real.sqrt (a^2 + b^2 + c^2) / 2)
  : sphere_volume sphere_radius = (8 * Real.sqrt 2 / 3) * Real.pi := 
by
  sorry

end volume_of_sphere_l377_37765


namespace simplify_expression_l377_37798

theorem simplify_expression :
  (3 + 4 + 5 + 7) / 3 + (3 * 6 + 9) / 4 = 157 / 12 :=
by
  sorry

end simplify_expression_l377_37798


namespace sin_double_angle_l377_37740

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end sin_double_angle_l377_37740


namespace parallel_lines_suff_cond_not_necess_l377_37718

theorem parallel_lines_suff_cond_not_necess (a : ℝ) :
  a = -2 → 
  (∀ x y : ℝ, (2 * x + y - 3 = 0) ∧ (2 * x + y + 4 = 0) → 
    (∃ a : ℝ, a = -2 ∨ a = 1)) ∧
    (a = -2 → ∃ a : ℝ, a = -2 ∨ a = 1) :=
by {
  sorry
}

end parallel_lines_suff_cond_not_necess_l377_37718


namespace sqrt_factorial_squared_l377_37786

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l377_37786


namespace problem1_problem2_l377_37736

open Real

theorem problem1 : sin (420 * π / 180) * cos (330 * π / 180) + sin (-690 * π / 180) * cos (-660 * π / 180) = 1 := by
  sorry

theorem problem2 (α : ℝ) : 
  (sin (π / 2 + α) * cos (π / 2 - α) / cos (π + α)) + 
  (sin (π - α) * cos (π / 2 + α) / sin (π + α)) = 0 := by
  sorry

end problem1_problem2_l377_37736


namespace age_of_teacher_l377_37761

variables (age_students : ℕ) (age_all : ℕ) (teacher_age : ℕ)

def avg_age_students := 15
def num_students := 10
def num_people := 11
def avg_age_people := 16

theorem age_of_teacher
  (h1 : age_students = num_students * avg_age_students)
  (h2 : age_all = num_people * avg_age_people)
  (h3 : age_all = age_students + teacher_age) : teacher_age = 26 :=
by
  sorry

end age_of_teacher_l377_37761


namespace equal_sum_seq_example_l377_37787

def EqualSumSeq (a : ℕ → ℕ) (c : ℕ) : Prop := ∀ n, a n + a (n + 1) = c

theorem equal_sum_seq_example (a : ℕ → ℕ) 
  (h1 : EqualSumSeq a 5) 
  (h2 : a 1 = 2) : a 6 = 3 :=
by 
  sorry

end equal_sum_seq_example_l377_37787


namespace no_intersect_x_axis_intersection_points_m_minus3_l377_37709

-- Define the quadratic function y = x^2 - 6x + 2m - 1
def quadratic_function (x m : ℝ) : ℝ := x^2 - 6 * x + 2 * m - 1

-- Theorem for Question 1: The function does not intersect the x-axis if and only if m > 5
theorem no_intersect_x_axis (m : ℝ) : (∀ x : ℝ, quadratic_function x m ≠ 0) ↔ m > 5 := sorry

-- Specific case when m = -3
def quadratic_function_m_minus3 (x : ℝ) : ℝ := x^2 - 6 * x - 7

-- Theorem for Question 2: Intersection points with coordinate axes for m = -3
theorem intersection_points_m_minus3 :
  ((∃ x : ℝ, quadratic_function_m_minus3 x = 0 ∧ (x = -1 ∨ x = 7)) ∧
   quadratic_function_m_minus3 0 = -7) := sorry

end no_intersect_x_axis_intersection_points_m_minus3_l377_37709


namespace ordered_pair_solution_l377_37751

theorem ordered_pair_solution :
  ∃ (x y : ℤ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ (x, y) = (2, 4) :=
by
  sorry

end ordered_pair_solution_l377_37751


namespace painting_area_l377_37788

theorem painting_area (wall_height wall_length bookshelf_height bookshelf_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_bookshelf_height : bookshelf_height = 3)
  (h_bookshelf_length : bookshelf_length = 5) :
  wall_height * wall_length - bookshelf_height * bookshelf_length = 135 := 
by
  sorry

end painting_area_l377_37788


namespace only_positive_odd_integer_dividing_3n_plus_1_l377_37706

theorem only_positive_odd_integer_dividing_3n_plus_1 : 
  ∀ (n : ℕ), (0 < n) → (n % 2 = 1) → (n ∣ (3 ^ n + 1)) → n = 1 := by
  sorry

end only_positive_odd_integer_dividing_3n_plus_1_l377_37706


namespace total_operation_time_correct_l377_37767

def accessories_per_doll := 2 + 3 + 1 + 5
def number_of_dolls := 12000
def time_per_doll := 45
def time_per_accessory := 10
def total_accessories := number_of_dolls * accessories_per_doll
def time_for_dolls := number_of_dolls * time_per_doll
def time_for_accessories := total_accessories * time_per_accessory
def total_combined_time := time_for_dolls + time_for_accessories

theorem total_operation_time_correct :
  total_combined_time = 1860000 :=
by
  sorry

end total_operation_time_correct_l377_37767


namespace smallest_six_consecutive_number_exists_max_value_N_perfect_square_l377_37780

-- Definition of 'six-consecutive numbers'
def is_six_consecutive (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧
  b ≠ d ∧ c ≠ d ∧ (a + b) * (c + d) = 60

-- Definition of the function F
def F (a b c d : ℕ) : ℤ :=
  let p := (10 * a + c) - (10 * b + d)
  let q := (10 * a + d) - (10 * b + c)
  q - p

-- Exists statement for the smallest six-consecutive number
theorem smallest_six_consecutive_number_exists :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ (1000 * a + 100 * b + 10 * c + d) = 1369 := 
sorry

-- Exists statement for the maximum N such that F(N) is perfect square
theorem max_value_N_perfect_square :
  ∃ (a b c d : ℕ), is_six_consecutive a b c d ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 9613 ∧
  ∃ (k : ℤ), F a b c d = k ^ 2 := 
sorry

end smallest_six_consecutive_number_exists_max_value_N_perfect_square_l377_37780


namespace mod_50_remainder_of_b86_l377_37749

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem mod_50_remainder_of_b86 : (b 86) % 50 = 40 := 
by 
-- Given definition of b and the problem is to prove the remainder of b_86 when divided by 50 is 40
sorry

end mod_50_remainder_of_b86_l377_37749


namespace number_of_women_more_than_men_l377_37789

variables (M W : ℕ)

def ratio_condition : Prop := M * 3 = 2 * W
def total_condition : Prop := M + W = 20
def correct_answer : Prop := W - M = 4

theorem number_of_women_more_than_men 
  (h1 : ratio_condition M W) 
  (h2 : total_condition M W) : 
  correct_answer M W := 
by 
  sorry

end number_of_women_more_than_men_l377_37789


namespace annie_spent_on_candies_l377_37735

theorem annie_spent_on_candies : 
  ∀ (num_classmates : ℕ) (candies_per_classmate : ℕ) (candies_left : ℕ) (cost_per_candy : ℚ),
  num_classmates = 35 →
  candies_per_classmate = 2 →
  candies_left = 12 →
  cost_per_candy = 0.1 →
  (num_classmates * candies_per_classmate + candies_left) * cost_per_candy = 8.2 :=
by
  intros num_classmates candies_per_classmate candies_left cost_per_candy
         h_classmates h_candies_per_classmate h_candies_left h_cost_per_candy
  simp [h_classmates, h_candies_per_classmate, h_candies_left, h_cost_per_candy]
  sorry

end annie_spent_on_candies_l377_37735


namespace rick_iron_hours_l377_37743

def can_iron_dress_shirts (h : ℕ) : ℕ := 4 * h

def can_iron_dress_pants (hours : ℕ) : ℕ := 3 * hours

def total_clothes_ironed (h : ℕ) : ℕ := can_iron_dress_shirts h + can_iron_dress_pants 5

theorem rick_iron_hours (h : ℕ) (H : total_clothes_ironed h = 27) : h = 3 :=
by sorry

end rick_iron_hours_l377_37743


namespace total_germs_l377_37796

-- Define variables and constants
namespace BiologyLab

def petri_dishes : ℕ := 75
def germs_per_dish : ℕ := 48

-- The goal is to prove that the total number of germs is as expected.
theorem total_germs : (petri_dishes * germs_per_dish) = 3600 :=
by
  -- Proof is omitted for this example
  sorry

end BiologyLab

end total_germs_l377_37796


namespace prove_ab_l377_37782

theorem prove_ab 
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 :=
by
  sorry

end prove_ab_l377_37782
