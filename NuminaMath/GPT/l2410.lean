import Mathlib

namespace unique_triple_solution_l2410_241023

theorem unique_triple_solution (x y z : ℝ) :
  (1 + x^4 ≤ 2 * (y - z)^2) →
  (1 + y^4 ≤ 2 * (z - x)^2) →
  (1 + z^4 ≤ 2 * (x - y)^2) →
  (x = 1 ∧ y = 0 ∧ z = -1) :=
sorry

end unique_triple_solution_l2410_241023


namespace tetrahedron_volume_ratio_l2410_241099

theorem tetrahedron_volume_ratio
  (a b : ℝ)
  (larger_tetrahedron : a = 6)
  (smaller_tetrahedron : b = a / 2) :
  (b^3 / a^3) = 1 / 8 := 
by 
  sorry

end tetrahedron_volume_ratio_l2410_241099


namespace ball_hits_ground_time_l2410_241003

def ball_height (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, ball_height t = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
sorry

end ball_hits_ground_time_l2410_241003


namespace probability_all_vertical_faces_green_l2410_241006

theorem probability_all_vertical_faces_green :
  let color_prob := (1 / 2 : ℚ)
  let total_arrangements := 2^6
  let valid_arrangements := 2 + 12 + 6
  ((valid_arrangements : ℚ) / total_arrangements) = 5 / 16 := by
  sorry

end probability_all_vertical_faces_green_l2410_241006


namespace max_value_g_l2410_241074

def g (n : ℕ) : ℕ :=
  if n < 12 then n + 12 else g (n - 7)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ M = 23 :=
  sorry

end max_value_g_l2410_241074


namespace value_of_a_l2410_241011

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 24 - 4 * a) : a = 3 :=
by
  sorry

end value_of_a_l2410_241011


namespace percentage_increase_consumption_l2410_241047

theorem percentage_increase_consumption
  (T C : ℝ) 
  (h_tax : ∀ t, t = 0.60 * T)
  (h_revenue : ∀ r, r = 0.75 * T * C) :
  1.25 * C = (0.75 * T * C) / (0.60 * T) := by
sorry

end percentage_increase_consumption_l2410_241047


namespace gcd_in_base3_l2410_241037

def gcd_2134_1455_is_97 : ℕ :=
  gcd 2134 1455

def base3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else aux (n / 3) ++ [n % 3]
    aux n

theorem gcd_in_base3 :
  gcd_2134_1455_is_97 = 97 ∧ base3 97 = [1, 0, 1, 2, 1] :=
by
  sorry

end gcd_in_base3_l2410_241037


namespace percentage_increase_l2410_241095

theorem percentage_increase 
    (P : ℝ)
    (buying_price : ℝ) (h1 : buying_price = 0.80 * P)
    (selling_price : ℝ) (h2 : selling_price = 1.24 * P) :
    ((selling_price - buying_price) / buying_price) * 100 = 55 := by 
  sorry

end percentage_increase_l2410_241095


namespace find_number_l2410_241090

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_number :
  (∃ x : ℕ, hash 3 x = 63 ∧ x = 7) :=
sorry

end find_number_l2410_241090


namespace roof_collapse_days_l2410_241010

def leaves_per_pound : ℕ := 1000
def pounds_limit_of_roof : ℕ := 500
def leaves_per_day : ℕ := 100

theorem roof_collapse_days : (pounds_limit_of_roof * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

end roof_collapse_days_l2410_241010


namespace maria_towels_l2410_241053

theorem maria_towels (green_towels white_towels given_towels : ℕ) (h1 : green_towels = 35) (h2 : white_towels = 21) (h3 : given_towels = 34) :
  green_towels + white_towels - given_towels = 22 :=
by
  sorry

end maria_towels_l2410_241053


namespace sophomores_stratified_sampling_l2410_241036

theorem sophomores_stratified_sampling 
  (total_students freshmen sophomores seniors selected_total : ℕ) 
  (H1 : total_students = 2800) 
  (H2 : freshmen = 970) 
  (H3 : sophomores = 930) 
  (H4 : seniors = 900) 
  (H_selected_total : selected_total = 280) : 
  (sophomores / total_students) * selected_total = 93 :=
by sorry

end sophomores_stratified_sampling_l2410_241036


namespace min_queries_to_determine_parity_l2410_241057

def num_bags := 100
def num_queries := 3
def bags := Finset (Fin num_bags)

def can_query_parity (bags : Finset (Fin num_bags)) : Prop :=
  bags.card = 15

theorem min_queries_to_determine_parity :
  ∀ (query : Fin num_queries → Finset (Fin num_bags)),
  (∀ i, can_query_parity (query i)) →
  (∀ i j k, query i ∪ query j ∪ query k = {a : Fin num_bags | a.val = 1}) →
  num_queries ≥ 3 :=
  sorry

end min_queries_to_determine_parity_l2410_241057


namespace find_equation_line_l2410_241092

noncomputable def line_through_point_area (A : Real × Real) (S : Real) : Prop :=
  ∃ (k : Real), (k < 0) ∧ (2 * A.1 + A.2 - 4 = 0) ∧
    (1 / 2 * (2 - k) * (1 - 2 / k) = S)

theorem find_equation_line (A : ℝ × ℝ) (S : ℝ) (hA : A = (1, 2)) (hS : S = 4) :
  line_through_point_area A S →
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ 2 * x + y - 4 = 0 :=
by
  sorry

end find_equation_line_l2410_241092


namespace work_rate_c_l2410_241038

theorem work_rate_c (A B C : ℝ) 
  (h1 : A + B = 1 / 15) 
  (h2 : A + B + C = 1 / 5) :
  (1 / C) = 7.5 :=
by 
  sorry

end work_rate_c_l2410_241038


namespace abs_non_positive_eq_zero_l2410_241013

theorem abs_non_positive_eq_zero (y : ℚ) (h : |4 * y - 7| ≤ 0) : y = 7 / 4 :=
by
  sorry

end abs_non_positive_eq_zero_l2410_241013


namespace perfect_square_of_d_l2410_241088

theorem perfect_square_of_d (a b c d : ℤ) (h : d = (a + (2:ℝ)^(1/3) * b + (4:ℝ)^(1/3) * c)^2) : ∃ k : ℤ, d = k^2 :=
by
  sorry

end perfect_square_of_d_l2410_241088


namespace simplify_sqrt_90000_l2410_241065

theorem simplify_sqrt_90000 : Real.sqrt 90000 = 300 :=
by
  /- Proof goes here -/
  sorry

end simplify_sqrt_90000_l2410_241065


namespace find_value_of_fraction_l2410_241063

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : (x / y) + (y / x) = 8)

theorem find_value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l2410_241063


namespace apron_more_than_recipe_book_l2410_241054

-- Define the prices and the total spent
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def total_ingredient_cost : ℕ := 5 * ingredient_cost
def total_spent : ℕ := 40

-- Define the condition that the total cost including the apron is $40
def total_without_apron : ℕ := recipe_book_cost + baking_dish_cost + total_ingredient_cost
def apron_cost : ℕ := total_spent - total_without_apron

-- Prove that the apron cost $1 more than the recipe book
theorem apron_more_than_recipe_book : apron_cost - recipe_book_cost = 1 := by
  -- The proof goes here
  sorry

end apron_more_than_recipe_book_l2410_241054


namespace prime_a_b_l2410_241004

theorem prime_a_b (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^11 + b = 2089) : 49 * b - a = 2007 :=
sorry

end prime_a_b_l2410_241004


namespace total_value_of_coins_is_correct_l2410_241044

def rolls_dollars : ℕ := 6
def rolls_half_dollars : ℕ := 5
def rolls_quarters : ℕ := 7
def rolls_dimes : ℕ := 4
def rolls_nickels : ℕ := 3
def rolls_pennies : ℕ := 2

def coins_per_dollar_roll : ℕ := 20
def coins_per_half_dollar_roll : ℕ := 25
def coins_per_quarter_roll : ℕ := 40
def coins_per_dime_roll : ℕ := 50
def coins_per_nickel_roll : ℕ := 40
def coins_per_penny_roll : ℕ := 50

def value_per_dollar : ℚ := 1
def value_per_half_dollar : ℚ := 0.5
def value_per_quarter : ℚ := 0.25
def value_per_dime : ℚ := 0.10
def value_per_nickel : ℚ := 0.05
def value_per_penny : ℚ := 0.01

theorem total_value_of_coins_is_correct : 
  rolls_dollars * coins_per_dollar_roll * value_per_dollar +
  rolls_half_dollars * coins_per_half_dollar_roll * value_per_half_dollar +
  rolls_quarters * coins_per_quarter_roll * value_per_quarter +
  rolls_dimes * coins_per_dime_roll * value_per_dime +
  rolls_nickels * coins_per_nickel_roll * value_per_nickel +
  rolls_pennies * coins_per_penny_roll * value_per_penny = 279.50 := 
sorry

end total_value_of_coins_is_correct_l2410_241044


namespace cost_of_traveling_all_roads_l2410_241046

noncomputable def total_cost_of_roads (length width road_width : ℝ) (cost_per_sq_m : ℝ) : ℝ :=
  let area_road_parallel_length := length * road_width
  let area_road_parallel_breadth := width * road_width
  let diagonal_length := Real.sqrt (length^2 + width^2)
  let area_road_diagonal := diagonal_length * road_width
  let total_area := area_road_parallel_length + area_road_parallel_breadth + area_road_diagonal
  total_area * cost_per_sq_m

theorem cost_of_traveling_all_roads :
  total_cost_of_roads 80 50 10 3 = 6730.2 :=
by
  sorry

end cost_of_traveling_all_roads_l2410_241046


namespace geometric_sequence_a3_eq_2_l2410_241034

theorem geometric_sequence_a3_eq_2 
  (a_1 a_3 a_5 : ℝ) 
  (h1 : a_1 * a_3 * a_5 = 8) 
  (h2 : a_3^2 = a_1 * a_5) : 
  a_3 = 2 :=
by 
  sorry

end geometric_sequence_a3_eq_2_l2410_241034


namespace price_per_sq_ft_l2410_241018

def house_sq_ft : ℕ := 2400
def barn_sq_ft : ℕ := 1000
def total_property_value : ℝ := 333200

theorem price_per_sq_ft : 
  (total_property_value / (house_sq_ft + barn_sq_ft)) = 98 := 
by 
  sorry

end price_per_sq_ft_l2410_241018


namespace max_surface_area_of_rectangular_solid_on_sphere_l2410_241021

noncomputable def max_surface_area_rectangular_solid (a b c : ℝ) :=
  2 * a * b + 2 * a * c + 2 * b * c

theorem max_surface_area_of_rectangular_solid_on_sphere :
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 36 → max_surface_area_rectangular_solid a b c ≤ 72) :=
by
  intros a b c h
  sorry

end max_surface_area_of_rectangular_solid_on_sphere_l2410_241021


namespace linear_function_quadrant_l2410_241040

theorem linear_function_quadrant (x y : ℝ) (h : y = 2 * x - 3) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = 2 * x - 3) :=
sorry

end linear_function_quadrant_l2410_241040


namespace find_a_l2410_241056

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 3) (h3 : a * x - 2 * y = 4) : a = 10 :=
by {
  sorry
}

end find_a_l2410_241056


namespace cat_litter_cost_l2410_241049

theorem cat_litter_cost 
    (container_weight : ℕ) (container_cost : ℕ)
    (litter_box_capacity : ℕ) (change_interval : ℕ) 
    (days_needed : ℕ) (cost : ℕ) :
  container_weight = 45 → 
  container_cost = 21 → 
  litter_box_capacity = 15 → 
  change_interval = 7 →
  days_needed = 210 → 
  cost = 210 :=
by
  intros h1 h2 h3 h4 h5
  /- Here we would add the proof steps, but this is not required. -/
  sorry

end cat_litter_cost_l2410_241049


namespace parabola_focus_l2410_241073

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * a) - b)

theorem parabola_focus : focus_of_parabola 4 3 = (0, -47 / 16) :=
by
  -- Function definition: focus_of_parabola a b gives the focus of y = ax^2 - b
  -- Given: a = 4, b = 3
  -- Focus: (0, 1 / (4 * 4) - 3)
  -- Proof: Skipping detailed algebraic manipulation, assume function correctness
  sorry

end parabola_focus_l2410_241073


namespace coronavirus_transmission_l2410_241059

theorem coronavirus_transmission:
  (∃ x: ℝ, (1 + x)^2 = 225) :=
by
  sorry

end coronavirus_transmission_l2410_241059


namespace novels_per_month_l2410_241002

theorem novels_per_month (pages_per_novel : ℕ) (total_pages_per_year : ℕ) (months_in_year : ℕ) 
  (h1 : pages_per_novel = 200) (h2 : total_pages_per_year = 9600) (h3 : months_in_year = 12) : 
  (total_pages_per_year / pages_per_novel) / months_in_year = 4 :=
by
  have novels_per_year := total_pages_per_year / pages_per_novel
  have novels_per_month := novels_per_year / months_in_year
  sorry

end novels_per_month_l2410_241002


namespace solve_in_primes_l2410_241094

theorem solve_in_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p^2 - 6 * p * q + q^2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) := 
sorry

end solve_in_primes_l2410_241094


namespace amount_paid_to_shopkeeper_l2410_241068

theorem amount_paid_to_shopkeeper :
  let price_of_grapes := 8 * 70
  let price_of_mangoes := 9 * 55
  price_of_grapes + price_of_mangoes = 1055 :=
by
  sorry

end amount_paid_to_shopkeeper_l2410_241068


namespace count_8_digit_even_ending_l2410_241042

theorem count_8_digit_even_ending : 
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  (choices_first_digit * choices_middle_digits * choices_last_digit) = 45000000 :=
by
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  sorry

end count_8_digit_even_ending_l2410_241042


namespace not_B_l2410_241019

def op (x y : ℝ) := (x - y) ^ 2

theorem not_B (x y : ℝ) : 2 * (op x y) ≠ op (2 * x) (2 * y) :=
by
  sorry

end not_B_l2410_241019


namespace find_x_when_y_equals_two_l2410_241022

theorem find_x_when_y_equals_two (x : ℝ) (y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end find_x_when_y_equals_two_l2410_241022


namespace fraction_sequence_calc_l2410_241015

theorem fraction_sequence_calc : 
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) * (8 / 9) - 1 = -(7 / 9) := 
by 
  sorry

end fraction_sequence_calc_l2410_241015


namespace percent_problem_l2410_241089

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l2410_241089


namespace total_songs_correct_l2410_241091

-- Define the conditions of the problem
def num_country_albums := 2
def songs_per_country_album := 12
def num_pop_albums := 8
def songs_per_pop_album := 7
def num_rock_albums := 5
def songs_per_rock_album := 10
def num_jazz_albums := 2
def songs_per_jazz_album := 15

-- Define the total number of songs
def total_songs :=
  num_country_albums * songs_per_country_album +
  num_pop_albums * songs_per_pop_album +
  num_rock_albums * songs_per_rock_album +
  num_jazz_albums * songs_per_jazz_album

-- Proposition stating the correct total number of songs
theorem total_songs_correct : total_songs = 160 :=
by {
  sorry -- Proof not required
}

end total_songs_correct_l2410_241091


namespace Jaco_total_gift_budget_l2410_241020

theorem Jaco_total_gift_budget :
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  friends_gifts + parents_gifts = 100 :=
by
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  show friends_gifts + parents_gifts = 100
  sorry

end Jaco_total_gift_budget_l2410_241020


namespace x_to_the_12_eq_14449_l2410_241024

/-
Given the condition x + 1/x = 2*sqrt(2), prove that x^12 = 14449.
-/

theorem x_to_the_12_eq_14449 (x : ℂ) (hx : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := 
sorry

end x_to_the_12_eq_14449_l2410_241024


namespace inequality_proof_l2410_241026

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 := 
by 
  sorry

end inequality_proof_l2410_241026


namespace fx_properties_l2410_241052

-- Definition of the function
def f (x : ℝ) : ℝ := x * |x|

-- Lean statement for the proof problem
theorem fx_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) :=
by
  -- Definition used directly from the conditions
  sorry

end fx_properties_l2410_241052


namespace max_f_geq_l2410_241085

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq (x : ℝ) : ∃ x, f x ≥ (3 + Real.sqrt 3) / 2 := sorry

end max_f_geq_l2410_241085


namespace adults_eat_one_third_l2410_241028

theorem adults_eat_one_third (n c k : ℕ) (hn : n = 120) (hc : c = 4) (hk : k = 20) :
  ((n - c * k) / n : ℚ) = 1 / 3 :=
by
  sorry

end adults_eat_one_third_l2410_241028


namespace calculate_expression_l2410_241050

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l2410_241050


namespace math_problem_statements_are_correct_l2410_241043

theorem math_problem_statements_are_correct (a b : ℝ) (h : a > b ∧ b > 0) :
  (¬ (b / a > (b + 3) / (a + 3))) ∧ ((3 * a + 2 * b) / (2 * a + 3 * b) < a / b) ∧
  (¬ (2 * Real.sqrt a < Real.sqrt (a - b) + Real.sqrt b)) ∧ 
  (Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2) :=
by
  sorry

end math_problem_statements_are_correct_l2410_241043


namespace ratio_proof_l2410_241000

theorem ratio_proof (a b c d : ℝ) (h1 : a / b = 20) (h2 : c / b = 5) (h3 : c / d = 1 / 8) : 
  a / d = 1 / 2 :=
by
  sorry

end ratio_proof_l2410_241000


namespace salary_increase_l2410_241062

theorem salary_increase (S0 S3 : ℕ) (r : ℕ) : 
  S0 = 3000 ∧ S3 = 8232 ∧ (S0 * (1 + r / 100)^3 = S3) → r = 40 :=
by
  sorry

end salary_increase_l2410_241062


namespace quadratic_inequality_condition_l2410_241061

theorem quadratic_inequality_condition
  (a b c : ℝ)
  (h1 : b^2 - 4 * a * c < 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) :
  False :=
sorry

end quadratic_inequality_condition_l2410_241061


namespace maximum_value_a_over_b_plus_c_l2410_241051

open Real

noncomputable def max_frac_a_over_b_plus_c (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * (a + b + c) = b * c) : ℝ :=
  if (b = c) then (Real.sqrt 2 - 1) / 2 else -1 -- placeholder for irrelevant case

theorem maximum_value_a_over_b_plus_c 
  (a b c : ℝ) 
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq: a * (a + b + c) = b * c) :
  max_frac_a_over_b_plus_c a b c h_pos h_eq = (Real.sqrt 2 - 1) / 2 :=
sorry

end maximum_value_a_over_b_plus_c_l2410_241051


namespace washing_machine_capacity_l2410_241096

-- Definitions of the conditions
def total_pounds_per_day : ℕ := 200
def number_of_machines : ℕ := 8

-- Main theorem to prove the question == answer given the conditions
theorem washing_machine_capacity :
  total_pounds_per_day / number_of_machines = 25 :=
by
  sorry

end washing_machine_capacity_l2410_241096


namespace geometric_sum_sequence_l2410_241033

theorem geometric_sum_sequence (n : ℕ) (a : ℕ → ℕ) (a1 : a 1 = 2) (a4 : a 4 = 16) :
    (∃ q : ℕ, a 2 = a 1 * q) → (∃ S_n : ℕ, S_n = 2 * (2 ^ n - 1)) :=
by
  sorry

end geometric_sum_sequence_l2410_241033


namespace F_equiv_A_l2410_241009

-- Define the function F
def F : ℝ → ℝ := sorry

-- Given condition
axiom F_property (x : ℝ) : F ((1 - x) / (1 + x)) = x

-- The theorem that needs to be proved
theorem F_equiv_A (x : ℝ) : F (-2 - x) = -2 - F x := sorry

end F_equiv_A_l2410_241009


namespace length_of_room_l2410_241077

theorem length_of_room (b : ℕ) (t : ℕ) (L : ℕ) (blue_tiles : ℕ) (tile_area : ℕ) (total_area : ℕ) (effective_area : ℕ) (blue_area : ℕ) :
  b = 10 →
  t = 2 →
  blue_tiles = 16 →
  tile_area = t * t →
  total_area = (L - 4) * (b - 4) →
  blue_area = blue_tiles * tile_area →
  2 * blue_area = 3 * total_area →
  L = 20 :=
by
  intros h_b h_t h_blue_tiles h_tile_area h_total_area h_blue_area h_proportion
  sorry

end length_of_room_l2410_241077


namespace derivative_given_limit_l2410_241048

open Real

variable {f : ℝ → ℝ} {x₀ : ℝ}

theorem derivative_given_limit (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ - 2 * Δx) - f x₀) / Δx + 2) < ε) :
  deriv f x₀ = -1 := by
  sorry

end derivative_given_limit_l2410_241048


namespace work_completion_time_l2410_241087

theorem work_completion_time 
(w : ℝ)  -- total amount of work
(A B : ℝ)  -- work rate of a and b per day
(h1 : A + B = w / 30)  -- combined work rate
(h2 : 20 * (A + B) + 20 * A = w) : 
  (1 / A = 60) :=
sorry

end work_completion_time_l2410_241087


namespace not_divisible_by_121_l2410_241082

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 3 * n + 5)) :=
by
  sorry

end not_divisible_by_121_l2410_241082


namespace cells_after_10_days_l2410_241083

theorem cells_after_10_days :
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  a_n = 64 :=
by
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  show a_n = 64
  sorry

end cells_after_10_days_l2410_241083


namespace arithmetic_to_geometric_progression_l2410_241064

theorem arithmetic_to_geometric_progression (x y z : ℝ) 
  (hAP : 2 * y^2 - y * x = z^2) : 
  z^2 = y * (2 * y - x) := 
  by 
  sorry

end arithmetic_to_geometric_progression_l2410_241064


namespace jason_text_messages_per_day_l2410_241078

theorem jason_text_messages_per_day
  (monday_messages : ℕ)
  (tuesday_messages : ℕ)
  (total_messages : ℕ)
  (average_per_day : ℕ)
  (messages_wednesday_friday_per_day : ℕ) :
  monday_messages = 220 →
  tuesday_messages = monday_messages / 2 →
  average_per_day = 96 →
  total_messages = 5 * average_per_day →
  total_messages - (monday_messages + tuesday_messages) = 3 * messages_wednesday_friday_per_day →
  messages_wednesday_friday_per_day = 50 :=
by
  intros
  sorry

end jason_text_messages_per_day_l2410_241078


namespace line_interparabola_length_l2410_241086

theorem line_interparabola_length :
  (∀ (x y : ℝ), y = x - 2 → y^2 = 4 * x) →
  ∃ (A B : ℝ × ℝ), (∃ (x1 y1 x2 y2 : ℝ), A = (x1, y1) ∧ B = (x2, y2)) →
  (dist A B = 4 * Real.sqrt 6) :=
by
  intros
  sorry

end line_interparabola_length_l2410_241086


namespace rectangle_area_l2410_241008

theorem rectangle_area (P L W : ℝ) (hP : P = 2 * (L + W)) (hRatio : L / W = 5 / 2) (hP_val : P = 280) : 
  L * W = 4000 :=
by 
  sorry

end rectangle_area_l2410_241008


namespace min_value_fraction_l2410_241079

theorem min_value_fraction (x : ℝ) (hx : x < 2) : ∃ y : ℝ, y = (5 - 4 * x + x^2) / (2 - x) ∧ y = 2 :=
by sorry

end min_value_fraction_l2410_241079


namespace common_roots_product_l2410_241030

theorem common_roots_product
  (p q r s : ℝ)
  (hpqrs1 : p + q + r = 0)
  (hpqrs2 : pqr = -20)
  (hpqrs3 : p + q + s = -4)
  (hpqrs4 : pqs = -80)
  : p * q = 20 :=
sorry

end common_roots_product_l2410_241030


namespace area_of_black_region_l2410_241075

def side_length_square : ℝ := 10
def length_rectangle : ℝ := 5
def width_rectangle : ℝ := 2

theorem area_of_black_region :
  (side_length_square * side_length_square) - (length_rectangle * width_rectangle) = 90 := by
sorry

end area_of_black_region_l2410_241075


namespace only_one_positive_integer_n_l2410_241084

theorem only_one_positive_integer_n (k : ℕ) (hk : 0 < k) (m : ℕ) (hm : k + 2 ≤ m) :
  ∃! (n : ℕ), 0 < n ∧ n^m ∣ 5^(n^k) + 1 :=
sorry

end only_one_positive_integer_n_l2410_241084


namespace minimum_value_of_f_l2410_241067

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 1/x + 1/(x^2 + 1/x)

theorem minimum_value_of_f : 
  ∃ x > 0, f x = 2.5 :=
by 
  sorry

end minimum_value_of_f_l2410_241067


namespace quadrilateral_inequality_l2410_241093

theorem quadrilateral_inequality 
  (A B C D : Type)
  (AB AC AD BC BD CD : ℝ)
  (hAB_pos : 0 < AB)
  (hBC_pos : 0 < BC)
  (hCD_pos : 0 < CD)
  (hDA_pos : 0 < DA)
  (hAC_pos : 0 < AC)
  (hBD_pos : 0 < BD): 
  AC * BD ≤ AB * CD + BC * AD := 
sorry

end quadrilateral_inequality_l2410_241093


namespace load_transportable_l2410_241045

theorem load_transportable :
  ∃ (n : ℕ), n ≤ 11 ∧ (∀ (box_weight : ℕ) (total_weight : ℕ),
    total_weight = 13500 ∧ 
    box_weight ≤ 350 ∧ 
    (n * 1500) ≥ total_weight) :=
by
  sorry

end load_transportable_l2410_241045


namespace corn_height_after_three_weeks_l2410_241007

theorem corn_height_after_three_weeks 
  (week1_growth : ℕ) (week2_growth : ℕ) (week3_growth : ℕ) 
  (h1 : week1_growth = 2)
  (h2 : week2_growth = 2 * week1_growth)
  (h3 : week3_growth = 4 * week2_growth) :
  week1_growth + week2_growth + week3_growth = 22 :=
by {
  sorry
}

end corn_height_after_three_weeks_l2410_241007


namespace lying_dwarf_possible_numbers_l2410_241076

theorem lying_dwarf_possible_numbers (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
  (h2 : a_2 = a_1) 
  (h3 : a_3 = a_1 + a_2) 
  (h4 : a_4 = a_1 + a_2 + a_3) 
  (h5 : a_5 = a_1 + a_2 + a_3 + a_4) 
  (h6 : a_6 = a_1 + a_2 + a_3 + a_4 + a_5) 
  (h7 : a_7 = a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 58)
  (h_lied : ∃ (d : ℕ), d ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] ∧ (d ≠ a_1 ∧ d ≠ a_2 ∧ d ≠ a_3 ∧ d ≠ a_4 ∧ d ≠ a_5 ∧ d ≠ a_6 ∧ d ≠ a_7)) : 
  ∃ x : ℕ, x = 13 ∨ x = 26 :=
by
  sorry

end lying_dwarf_possible_numbers_l2410_241076


namespace train_speed_l2410_241080

def train_length : ℝ := 400  -- Length of the train in meters
def crossing_time : ℝ := 40  -- Time to cross the electric pole in seconds

theorem train_speed : train_length / crossing_time = 10 := by
  sorry  -- Proof to be completed

end train_speed_l2410_241080


namespace shaded_area_fraction_l2410_241035

-- Define the problem conditions
def total_squares : ℕ := 18
def half_squares : ℕ := 10
def whole_squares : ℕ := 3

-- Define the total shaded area given the conditions
def shaded_area := (half_squares * (1/2) + whole_squares)

-- Define the total area of the rectangle
def total_area := total_squares

-- Lean 4 theorem statement
theorem shaded_area_fraction :
  shaded_area / total_area = (4 : ℚ) / 9 :=
by sorry

end shaded_area_fraction_l2410_241035


namespace find_g_l2410_241039

noncomputable def g : ℝ → ℝ
| x => 2 * (4^x - 3^x)

theorem find_g :
  (g 1 = 2) ∧
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) →
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end find_g_l2410_241039


namespace find_fraction_abs_l2410_241058

-- Define the conditions and the main proof problem
theorem find_fraction_abs (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5 * x * y) :
  abs ((x + y) / (x - y)) = Real.sqrt ((7 : ℝ) / 3) :=
by
  sorry

end find_fraction_abs_l2410_241058


namespace inequality_abc_l2410_241055

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1) * (b + 1) * (a + c) * (b + c) ≥ 16 * a * b * c :=
by
  sorry

end inequality_abc_l2410_241055


namespace solve_real_roots_in_intervals_l2410_241066

noncomputable def real_roots_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ x₁ x₂ : ℝ,
    (3 * x₁^2 - 2 * (a - b) * x₁ - a * b = 0) ∧
    (3 * x₂^2 - 2 * (a - b) * x₂ - a * b = 0) ∧
    (a / 3 < x₁ ∧ x₁ < 2 * a / 3) ∧
    (-2 * b / 3 < x₂ ∧ x₂ < -b / 3)

-- Statement of the problem:
theorem solve_real_roots_in_intervals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  real_roots_intervals a b ha hb :=
sorry

end solve_real_roots_in_intervals_l2410_241066


namespace ellipse_eccentricity_l2410_241070

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (B F A C : ℝ × ℝ) 
    (h3 : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
    (h4 : (C.1 ^ 2 / a ^ 2 + C.2 ^ 2 / b ^ 2 = 1))
    (h5 : B.1 > 0 ∧ B.2 > 0)
    (h6 : C.1 > 0 ∧ C.2 > 0)
    (h7 : ∃ M : ℝ × ℝ, M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ (F = M)) :
    ∃ e : ℝ, e = (1 / 3) := 
  sorry

end ellipse_eccentricity_l2410_241070


namespace crescent_moon_area_l2410_241060

theorem crescent_moon_area :
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  crescent_area = 2 * Real.pi :=
by
  let big_quarter_circle := (4 * 4 * Real.pi) / 4
  let small_semi_circle := (2 * 2 * Real.pi) / 2
  let crescent_area := big_quarter_circle - small_semi_circle
  have h_bqc : big_quarter_circle = 4 * Real.pi := by
    sorry
  have h_ssc : small_semi_circle = 2 * Real.pi := by
    sorry
  have h_ca : crescent_area = 2 * Real.pi := by
    sorry
  exact h_ca

end crescent_moon_area_l2410_241060


namespace exists_integers_gcd_eq_one_addition_l2410_241081

theorem exists_integers_gcd_eq_one_addition 
  (n k : ℕ) 
  (hnk_pos : n > 0 ∧ k > 0) 
  (hn_even_or_nk_even : (¬ n % 2 = 0) ∨ (n % 2 = 0 ∧ k % 2 = 0)) :
  ∃ a b : ℤ, Int.gcd a ↑n = 1 ∧ Int.gcd b ↑n = 1 ∧ k = a + b :=
by
  sorry

end exists_integers_gcd_eq_one_addition_l2410_241081


namespace solve_cubic_eq_l2410_241005

theorem solve_cubic_eq (x : ℝ) : (8 - x)^3 = x^3 → x = 8 :=
by
  sorry

end solve_cubic_eq_l2410_241005


namespace jesse_total_carpet_l2410_241025

theorem jesse_total_carpet : 
  let length_rect := 12
  let width_rect := 8
  let base_tri := 10
  let height_tri := 6
  let area_rect := length_rect * width_rect
  let area_tri := (base_tri * height_tri) / 2
  area_rect + area_tri = 126 :=
by
  sorry

end jesse_total_carpet_l2410_241025


namespace circle_center_radius_l2410_241071

theorem circle_center_radius (x y : ℝ) :
  (x ^ 2 + y ^ 2 + 2 * x - 4 * y - 6 = 0) →
  ((x + 1) ^ 2 + (y - 2) ^ 2 = 11) :=
by sorry

end circle_center_radius_l2410_241071


namespace problem_a2_b_c_in_M_l2410_241027

def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem problem_a2_b_c_in_M (a b c : ℤ) (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
sorry

end problem_a2_b_c_in_M_l2410_241027


namespace Talia_father_age_l2410_241069

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end Talia_father_age_l2410_241069


namespace quadratic_equation_completing_square_l2410_241029

theorem quadratic_equation_completing_square :
  ∃ a b c : ℤ, a > 0 ∧ (25 * x^2 + 30 * x - 75 = 0 → (a * x + b)^2 = c) ∧ a + b + c = -58 :=
  sorry

end quadratic_equation_completing_square_l2410_241029


namespace milton_apple_pie_slices_l2410_241012

theorem milton_apple_pie_slices :
  ∀ (A : ℕ),
  (∀ (peach_pie_slices_per : ℕ), peach_pie_slices_per = 6) →
  (∀ (apple_pie_slices_sold : ℕ), apple_pie_slices_sold = 56) →
  (∀ (peach_pie_slices_sold : ℕ), peach_pie_slices_sold = 48) →
  (∀ (total_pies_sold : ℕ), total_pies_sold = 15) →
  (∃ (apple_pie_slices : ℕ), apple_pie_slices = 56 / (total_pies_sold - (peach_pie_slices_sold / peach_pie_slices_per))) → 
  A = 8 :=
by sorry

end milton_apple_pie_slices_l2410_241012


namespace solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l2410_241072

variable (a b : ℝ)

theorem solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0 :
  (∀ x : ℝ, (|x - 2| > 1 ↔ x^2 + a * x + b > 0)) → a + b = -1 :=
by
  sorry

end solution_set_equality_of_abs_x_minus_2_gt_1_and_quadratic_gt_0_l2410_241072


namespace single_point_graph_value_of_d_l2410_241041

theorem single_point_graph_value_of_d (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + d = 0 → x = -2 ∧ y = 3) ↔ d = 21 := 
by 
  sorry

end single_point_graph_value_of_d_l2410_241041


namespace shift_quadratic_function_left_l2410_241098

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the shifted quadratic function
def shifted_function (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem shift_quadratic_function_left :
  ∀ x : ℝ, shifted_function x = original_function (x + 1) := by
  sorry

end shift_quadratic_function_left_l2410_241098


namespace ratio_of_first_term_to_common_difference_l2410_241031

theorem ratio_of_first_term_to_common_difference
  (a d : ℝ)
  (h : (8 / 2 * (2 * a + 7 * d)) = 3 * (5 / 2 * (2 * a + 4 * d))) :
  a / d = 2 / 7 :=
by
  sorry

end ratio_of_first_term_to_common_difference_l2410_241031


namespace expression_evaluation_l2410_241016

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x^2 - 4 * y + 5 = 24 :=
by
  sorry

end expression_evaluation_l2410_241016


namespace pizzas_served_today_l2410_241014

theorem pizzas_served_today (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (h1 : lunch_pizzas = 9) (h2 : dinner_pizzas = 6) : lunch_pizzas + dinner_pizzas = 15 :=
by sorry

end pizzas_served_today_l2410_241014


namespace integer_solutions_exist_l2410_241017

theorem integer_solutions_exist (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 := 
sorry

end integer_solutions_exist_l2410_241017


namespace sum_of_first_100_positive_odd_integers_is_correct_l2410_241001

def sum_first_100_positive_odd_integers : ℕ :=
  10000

theorem sum_of_first_100_positive_odd_integers_is_correct :
  sum_first_100_positive_odd_integers = 10000 :=
by
  sorry

end sum_of_first_100_positive_odd_integers_is_correct_l2410_241001


namespace inequality_for_positive_integer_l2410_241032

theorem inequality_for_positive_integer (n : ℕ) (h : n > 0) :
  n^n ≤ (n!)^2 ∧ (n!)^2 ≤ ((n + 1) * (n + 2) / 6)^n := by
  sorry

end inequality_for_positive_integer_l2410_241032


namespace arccos_sin_2_equals_l2410_241097

theorem arccos_sin_2_equals : Real.arccos (Real.sin 2) = 2 - Real.pi / 2 := by
  sorry

end arccos_sin_2_equals_l2410_241097
