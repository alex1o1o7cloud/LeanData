import Mathlib

namespace division_multiplication_identity_l1359_135946

theorem division_multiplication_identity :
  24 / (-6) * (3 / 2) / (- (4 / 3)) = 9 / 2 := 
by 
  sorry

end division_multiplication_identity_l1359_135946


namespace range_of_a_l1359_135906

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * x^2 - (2 * a + 1) * x + 1

theorem range_of_a (a : ℝ) (h_a : 0 < a ∧ a ≤ 1/2) : 
  ∀ x : ℝ, x ∈ Set.Ici a → f x a ≥ a^3 - a - 1/8 :=
by
  sorry

end range_of_a_l1359_135906


namespace largest_band_members_l1359_135902

def band_formation (m r x : ℕ) : Prop :=
  m < 100 ∧ m = r * x + 2 ∧ (r - 2) * (x + 1) = m ∧ r - 2 * x = 4

theorem largest_band_members : ∃ (r x m : ℕ), band_formation m r x ∧ m = 98 := 
  sorry

end largest_band_members_l1359_135902


namespace dan_has_remaining_cards_l1359_135962

-- Define the initial conditions
def initial_cards : ℕ := 97
def cards_sold_to_sam : ℕ := 15

-- Define the expected result
def remaining_cards (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

-- State the theorem to prove
theorem dan_has_remaining_cards : remaining_cards initial_cards cards_sold_to_sam = 82 :=
by
  -- This insertion is a placeholder for the proof
  sorry

end dan_has_remaining_cards_l1359_135962


namespace total_students_count_l1359_135917

theorem total_students_count (n1 n2 n: ℕ) (avg1 avg2 avg_tot: ℝ)
  (h1: n1 = 15) (h2: avg1 = 70) (h3: n2 = 10) (h4: avg2 = 90) (h5: avg_tot = 78)
  (h6: (n1 * avg1 + n2 * avg2) / (n1 + n2) = avg_tot) :
  n = 25 :=
by
  sorry

end total_students_count_l1359_135917


namespace absolute_value_zero_l1359_135961

theorem absolute_value_zero (x : ℝ) (h : |4 * x + 6| = 0) : x = -3 / 2 :=
sorry

end absolute_value_zero_l1359_135961


namespace exists_indices_divisible_2019_l1359_135997

theorem exists_indices_divisible_2019 (x : Fin 2020 → ℤ) : 
  ∃ (i j : Fin 2020), i ≠ j ∧ (x j - x i) % 2019 = 0 := 
  sorry

end exists_indices_divisible_2019_l1359_135997


namespace last_three_digits_2005_pow_2005_l1359_135925

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_digits_2005_pow_2005 :
  last_three_digits (2005 ^ 2005) = 125 :=
sorry

end last_three_digits_2005_pow_2005_l1359_135925


namespace years_in_future_l1359_135915

theorem years_in_future (Shekhar Shobha : ℕ) (h1 : Shekhar / Shobha = 4 / 3) (h2 : Shobha = 15) (h3 : Shekhar + t = 26)
  : t = 6 :=
by
  sorry

end years_in_future_l1359_135915


namespace incorrect_membership_l1359_135954

-- Let's define the sets involved.
def a : Set ℕ := {1}             -- singleton set {a}
def ab : Set (Set ℕ) := {{1}, {2}}  -- set {a, b}

-- Now, the proof statement.
theorem incorrect_membership : ¬ (a ∈ ab) := 
by { sorry }

end incorrect_membership_l1359_135954


namespace remainder_sequences_mod_1000_l1359_135928

theorem remainder_sequences_mod_1000 :
  ∃ m, (m = 752) ∧ (m % 1000 = 752) ∧ 
  (∃ (a : ℕ → ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ 6 → (a i) - i % 2 = 1), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 6 → a i ≤ a j) ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 6 → 1 ≤ a i ∧ a i ≤ 1500)
  ) := by
    -- proof would go here
    sorry

end remainder_sequences_mod_1000_l1359_135928


namespace problem_l1359_135998

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem problem {a α b β : ℝ} (h : f 2001 a α b β = 3) : f 2012 a α b β = -3 := by
  sorry

end problem_l1359_135998


namespace tangerines_more_than_oranges_l1359_135923

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l1359_135923


namespace meal_combinations_l1359_135992

theorem meal_combinations (MenuA_items : ℕ) (MenuB_items : ℕ) : MenuA_items = 15 ∧ MenuB_items = 12 → MenuA_items * MenuB_items = 180 :=
by
  sorry

end meal_combinations_l1359_135992


namespace find_x_l1359_135937

theorem find_x (x : ℚ) (h : (3 - x) / (2 - x) - 1 / (x - 2) = 3) : x = 1 := 
  sorry

end find_x_l1359_135937


namespace Jessica_has_3_dozens_l1359_135973

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l1359_135973


namespace waiter_earnings_l1359_135959

theorem waiter_earnings
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (customers_tipped : total_customers - no_tip_customers = 3)
  (tips_per_customer : tip_amount = 9) :
  (total_customers - no_tip_customers) * tip_amount = 27 := by
  sorry

end waiter_earnings_l1359_135959


namespace find_c_l1359_135920

theorem find_c (b c : ℤ) (H : (b - 4) / (2 * b + 42) = c / 6) : c = 2 := 
sorry

end find_c_l1359_135920


namespace flowers_lost_l1359_135903

theorem flowers_lost 
  (time_per_flower : ℕ)
  (gathered_time : ℕ) 
  (additional_time : ℕ) 
  (classmates : ℕ) 
  (collected_flowers : ℕ) 
  (total_needed : ℕ)
  (lost_flowers : ℕ) 
  (H1 : time_per_flower = 10)
  (H2 : gathered_time = 120)
  (H3 : additional_time = 210)
  (H4 : classmates = 30)
  (H5 : collected_flowers = gathered_time / time_per_flower)
  (H6 : total_needed = classmates + (additional_time / time_per_flower))
  (H7 : lost_flowers = total_needed - classmates) :
lost_flowers = 3 := 
sorry

end flowers_lost_l1359_135903


namespace common_difference_arithmetic_sequence_l1359_135908

theorem common_difference_arithmetic_sequence (d : ℝ) :
  (∀ (n : ℝ) (a_1 : ℝ), a_1 = 9 ∧
  (∃ a₄ a₈ : ℝ, a₄ = a_1 + 3 * d ∧ a₈ = a_1 + 7 * d ∧ a₄ = (a_1 * a₈)^(1/2)) →
  d = 1) :=
sorry

end common_difference_arithmetic_sequence_l1359_135908


namespace problem1_problem2_l1359_135900

variable (x a : ℝ)

def P := x^2 - 5*a*x + 4*a^2 < 0
def Q := (x^2 - 2*x - 8 <= 0) ∧ (x^2 + 3*x - 10 > 0)

theorem problem1 (h : 1 = a) (hP : P x a) (hQ : Q x) : 2 < x ∧ x ≤ 4 :=
sorry

theorem problem2 (h1 : ∀ x, ¬P x a → ¬Q x) (h2 : ∃ x, P x a ∧ ¬Q x) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l1359_135900


namespace problem_sign_of_trig_product_l1359_135945

open Real

theorem problem_sign_of_trig_product (θ : ℝ) (hθ : π / 2 < θ ∧ θ < π) :
  sin (cos θ) * cos (sin (2 * θ)) < 0 :=
sorry

end problem_sign_of_trig_product_l1359_135945


namespace value_of_sum_ratio_l1359_135984

theorem value_of_sum_ratio (w x y: ℝ) (hx: w / x = 1 / 3) (hy: w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end value_of_sum_ratio_l1359_135984


namespace fractional_inequality_solution_l1359_135940

theorem fractional_inequality_solution :
  ∃ (m n : ℕ), n = m^2 - 1 ∧ 
               (m + 2) / (n + 2 : ℝ) > 1 / 3 ∧ 
               (m - 3) / (n - 3 : ℝ) < 1 / 10 ∧ 
               1 ≤ m ∧ m ≤ 9 ∧ 1 ≤ n ∧ n ≤ 9 ∧ 
               (m = 3) ∧ (n = 8) := 
by
  sorry

end fractional_inequality_solution_l1359_135940


namespace angle_y_in_triangle_l1359_135968

theorem angle_y_in_triangle (y : ℝ) (h1 : ∀ a b c : ℝ, a + b + c = 180) (h2 : 3 * y + y + 40 = 180) : y = 35 :=
sorry

end angle_y_in_triangle_l1359_135968


namespace ratio_of_sold_phones_to_production_l1359_135999

def last_years_production : ℕ := 5000
def this_years_production : ℕ := 2 * last_years_production
def phones_left_in_factory : ℕ := 7500
def sold_phones : ℕ := this_years_production - phones_left_in_factory

theorem ratio_of_sold_phones_to_production : 
  (sold_phones : ℚ) / this_years_production = 1 / 4 := 
by
  sorry

end ratio_of_sold_phones_to_production_l1359_135999


namespace simple_interest_l1359_135924

theorem simple_interest (P R T : ℝ) (hP : P = 8965) (hR : R = 9) (hT : T = 5) : 
    (P * R * T) / 100 = 806.85 := 
by 
  sorry

end simple_interest_l1359_135924


namespace calculate_total_tulips_l1359_135953

def number_of_red_tulips_for_eyes := 8 * 2
def number_of_purple_tulips_for_eyebrows := 5 * 2
def number_of_red_tulips_for_nose := 12
def number_of_red_tulips_for_smile := 18
def number_of_yellow_tulips_for_background := 9 * number_of_red_tulips_for_smile

def total_number_of_tulips : ℕ :=
  number_of_red_tulips_for_eyes + 
  number_of_red_tulips_for_nose + 
  number_of_red_tulips_for_smile + 
  number_of_purple_tulips_for_eyebrows + 
  number_of_yellow_tulips_for_background

theorem calculate_total_tulips : total_number_of_tulips = 218 := by
  sorry

end calculate_total_tulips_l1359_135953


namespace arthur_walking_distance_l1359_135963

/-- Arthur walks 8 blocks west and 10 blocks south, 
    each block being 1/4 mile -/
theorem arthur_walking_distance 
  (blocks_west : ℕ) (blocks_south : ℕ) (block_distance : ℚ)
  (h1 : blocks_west = 8) (h2 : blocks_south = 10) (h3 : block_distance = 1/4) :
  (blocks_west + blocks_south) * block_distance = 4.5 := 
by
  sorry

end arthur_walking_distance_l1359_135963


namespace tan_alpha_plus_pi_over_4_rational_expression_of_trig_l1359_135936

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + Real.pi / 4) = -1 / 7 := 
by 
  sorry

theorem rational_expression_of_trig (α : ℝ) (h : Real.tan (α / 2) = 2) : 
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_rational_expression_of_trig_l1359_135936


namespace actual_road_length_l1359_135975

theorem actual_road_length
  (scale_factor : ℕ → ℕ → Prop)
  (map_length_cm : ℕ)
  (actual_length_km : ℝ) : 
  (scale_factor 1 50000) →
  (map_length_cm = 15) →
  (actual_length_km = 7.5) :=
by
  sorry

end actual_road_length_l1359_135975


namespace FourConsecIntsSum34Unique_l1359_135948

theorem FourConsecIntsSum34Unique :
  ∃! (a b c d : ℕ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (a + b + c + d = 34) ∧ (d = a + 3) :=
by
  -- The proof will be placed here
  sorry

end FourConsecIntsSum34Unique_l1359_135948


namespace leak_empty_time_l1359_135943

theorem leak_empty_time
  (pump_fill_time : ℝ)
  (leak_fill_time : ℝ)
  (pump_fill_rate : pump_fill_time = 5)
  (leak_fill_rate : leak_fill_time = 10)
  : (1 / 5 - 1 / leak_fill_time)⁻¹ = 10 :=
by
  -- you can fill in the proof here
  sorry

end leak_empty_time_l1359_135943


namespace scientific_notation_of_tourists_l1359_135930

theorem scientific_notation_of_tourists : 
  (23766400 : ℝ) = 2.37664 * 10^7 :=
by 
  sorry

end scientific_notation_of_tourists_l1359_135930


namespace percentage_of_number_l1359_135969

theorem percentage_of_number (N P : ℕ) (h₁ : N = 50) (h₂ : N = (P * N / 100) + 42) : P = 16 :=
by
  sorry

end percentage_of_number_l1359_135969


namespace compare_neg_fractions_l1359_135942

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (5 / 7 : ℝ) := 
by 
  sorry

end compare_neg_fractions_l1359_135942


namespace fourth_friend_payment_l1359_135988

theorem fourth_friend_payment (a b c d : ℕ) 
  (h1 : a = (1 / 3) * (b + c + d)) 
  (h2 : b = (1 / 4) * (a + c + d)) 
  (h3 : c = (1 / 5) * (a + b + d))
  (h4 : a + b + c + d = 84) : 
  d = 40 := by
sorry

end fourth_friend_payment_l1359_135988


namespace number_of_tests_in_series_l1359_135952

theorem number_of_tests_in_series (S : ℝ) (n : ℝ) :
  (S + 97) / n = 90 →
  (S + 73) / n = 87 →
  n = 8 :=
by 
  sorry

end number_of_tests_in_series_l1359_135952


namespace infinite_solutions_l1359_135991

theorem infinite_solutions (x y : ℝ) : ∃ x y : ℝ, x^3 + y^2 * x - 6 * x + 5 * y + 1 = 0 :=
sorry

end infinite_solutions_l1359_135991


namespace probability_full_house_after_rerolling_l1359_135996

theorem probability_full_house_after_rerolling
  (a b c : ℕ)
  (h0 : a ≠ b)
  (h1 : c ≠ a)
  (h2 : c ≠ b) :
  (2 / 6 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end probability_full_house_after_rerolling_l1359_135996


namespace number_of_x_intercepts_l1359_135951

def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

theorem number_of_x_intercepts : ∃! (x : ℝ), ∃ (y : ℝ), parabola y = x ∧ y = 0 :=
by
  sorry

end number_of_x_intercepts_l1359_135951


namespace eight_diamond_three_l1359_135990

def diamond (x y : ℤ) : ℤ := sorry

axiom diamond_zero (x : ℤ) : diamond x 0 = x
axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x
axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

theorem eight_diamond_three : diamond 8 3 = 39 :=
sorry

end eight_diamond_three_l1359_135990


namespace average_speed_l1359_135947

theorem average_speed (uphill_speed downhill_speed : ℚ) (t : ℚ) (v : ℚ) :
  uphill_speed = 4 →
  downhill_speed = 6 →
  (1 / uphill_speed + 1 / downhill_speed = t) →
  (v * t = 2) →
  v = 4.8 :=
by
  intros
  sorry

end average_speed_l1359_135947


namespace problem1_problem2_l1359_135960

open Set

variable {x y z a b : ℝ}

-- Problem 1: Prove the inequality
theorem problem1 (x y z : ℝ) : 
  5 * x^2 + y^2 + z^2 ≥ 2 * x * y + 4 * x + 2 * z - 2 :=
by
  sorry

-- Problem 2: Prove the range of 10a - 5b is [−1, 20]
theorem problem2 (a b : ℝ) 
  (h1 : 1 ≤ 2 * a + b ∧ 2 * a + b ≤ 4)
  (h2 : -1 ≤ a - 2 * b ∧ a - 2 * b ≤ 2) : 
  -1 ≤ 10 * a - 5 * b ∧ 10 * a - 5 * b ≤ 20 :=
by
  sorry

end problem1_problem2_l1359_135960


namespace math_problem_l1359_135966

theorem math_problem {x y : ℕ} (h1 : 1059 % x = y) (h2 : 1417 % x = y) (h3 : 2312 % x = y) : x - y = 15 := by
  sorry

end math_problem_l1359_135966


namespace problem_statement_l1359_135967

def system_eq1 (x y : ℝ) := x^3 - 5 * x * y^2 = 21
def system_eq2 (y x : ℝ) := y^3 - 5 * x^2 * y = 28

theorem problem_statement
(x1 y1 x2 y2 x3 y3 : ℝ)
(h1 : system_eq1 x1 y1)
(h2 : system_eq2 y1 x1)
(h3 : system_eq1 x2 y2)
(h4 : system_eq2 y2 x2)
(h5 : system_eq1 x3 y3)
(h6 : system_eq2 y3 x3)
(h_distinct : (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :
  (11 - x1 / y1) * (11 - x2 / y2) * (11 - x3 / y3) = 1729 :=
sorry

end problem_statement_l1359_135967


namespace book_length_ratio_is_4_l1359_135982

-- Define the initial conditions
def pages_when_6 : ℕ := 8
def age_when_start := 6
def multiple_at_twice_age := 5
def multiple_eight_years_after := 3
def current_pages : ℕ := 480

def pages_when_12 := pages_when_6 * multiple_at_twice_age
def pages_when_20 := pages_when_12 * multiple_eight_years_after

theorem book_length_ratio_is_4 :
  (current_pages : ℚ) / pages_when_20 = 4 := by
  -- We need to show the proof for the equality
  sorry

end book_length_ratio_is_4_l1359_135982


namespace product_of_two_numbers_is_320_l1359_135933

theorem product_of_two_numbers_is_320 (x y : ℕ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x = 5 * (y / 4)) : x * y = 320 :=
by {
  sorry
}

end product_of_two_numbers_is_320_l1359_135933


namespace equation_1_equation_2_l1359_135970

theorem equation_1 (x : ℝ) : x^2 - 1 = 8 ↔ x = 3 ∨ x = -3 :=
by sorry

theorem equation_2 (x : ℝ) : (x + 4)^3 = -64 ↔ x = -8 :=
by sorry

end equation_1_equation_2_l1359_135970


namespace goose_eggs_count_l1359_135935

theorem goose_eggs_count (E : ℕ)
  (h1 : (2/3 : ℚ) * E ≥ 0)
  (h2 : (3/4 : ℚ) * (2/3 : ℚ) * E ≥ 0)
  (h3 : 100 = (2/5 : ℚ) * (3/4 : ℚ) * (2/3 : ℚ) * E) :
  E = 500 := by
  sorry

end goose_eggs_count_l1359_135935


namespace x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l1359_135980

theorem x_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 6 * x (k - 1) - x (k - 2) := 
by sorry

theorem x_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 34 * x (k - 2) - x (k - 4) := 
by sorry

theorem x_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 198 * x (k - 3) - x (k - 6) := 
by sorry

theorem y_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 6 * y (k - 1) - y (k - 2) := 
by sorry

theorem y_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 34 * y (k - 2) - y (k - 4) := 
by sorry

theorem y_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 198 * y (k - 3) - y (k - 6) := 
by sorry

end x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l1359_135980


namespace expression_equals_minus_0p125_l1359_135927

-- Define the expression
def compute_expression : ℝ := 0.125^8 * (-8)^7

-- State the theorem to prove
theorem expression_equals_minus_0p125 : compute_expression = -0.125 :=
by {
  sorry
}

end expression_equals_minus_0p125_l1359_135927


namespace units_digit_m_squared_plus_3_pow_m_l1359_135901

def m := 2023^2 + 3^2023

theorem units_digit_m_squared_plus_3_pow_m : 
  (m^2 + 3^m) % 10 = 5 := sorry

end units_digit_m_squared_plus_3_pow_m_l1359_135901


namespace sample_size_is_10_l1359_135971

def product := Type

noncomputable def number_of_products : ℕ := 80
noncomputable def selected_products_for_quality_inspection : ℕ := 10

theorem sample_size_is_10 
  (N : ℕ) (sample_size : ℕ) 
  (hN : N = 80) 
  (h_sample_size : sample_size = 10) : 
  sample_size = 10 :=
by 
  sorry

end sample_size_is_10_l1359_135971


namespace intersection_point_interval_l1359_135957

theorem intersection_point_interval (x₀ : ℝ) (h : x₀^3 = 2^x₀ + 1) : 
  1 < x₀ ∧ x₀ < 2 :=
by
  sorry

end intersection_point_interval_l1359_135957


namespace Hillary_reading_time_on_sunday_l1359_135949

-- Define the assigned reading times for both books
def assigned_time_book_a : ℕ := 60 -- minutes
def assigned_time_book_b : ℕ := 45 -- minutes

-- Define the reading times already spent on each book
def time_spent_friday_book_a : ℕ := 16 -- minutes
def time_spent_saturday_book_a : ℕ := 28 -- minutes
def time_spent_saturday_book_b : ℕ := 15 -- minutes

-- Calculate the total time already read for each book
def total_time_read_book_a : ℕ := time_spent_friday_book_a + time_spent_saturday_book_a
def total_time_read_book_b : ℕ := time_spent_saturday_book_b

-- Calculate the remaining time needed for each book
def remaining_time_book_a : ℕ := assigned_time_book_a - total_time_read_book_a
def remaining_time_book_b : ℕ := assigned_time_book_b - total_time_read_book_b

-- Calculate the total remaining time and the equal time division
def total_remaining_time : ℕ := remaining_time_book_a + remaining_time_book_b
def equal_time_division : ℕ := total_remaining_time / 2

-- Theorem statement to prove Hillary's reading time for each book on Sunday
theorem Hillary_reading_time_on_sunday : equal_time_division = 23 := by
  sorry

end Hillary_reading_time_on_sunday_l1359_135949


namespace value_of_a_minus_b_l1359_135919

theorem value_of_a_minus_b (a b : ℝ) (h₁ : |a| = 2) (h₂ : |b| = 5) (h₃ : a < b) :
  a - b = -3 ∨ a - b = -7 := 
sorry

end value_of_a_minus_b_l1359_135919


namespace large_block_volume_l1359_135958

theorem large_block_volume (W D L : ℝ) (h : W * D * L = 4) :
    (2 * W) * (2 * D) * (2 * L) = 32 :=
by
  sorry

end large_block_volume_l1359_135958


namespace triangle_heights_inequality_l1359_135939

variable {R : Type} [OrderedRing R]

theorem triangle_heights_inequality (m_a m_b m_c s : R) 
  (h_m_a_nonneg : 0 ≤ m_a) (h_m_b_nonneg : 0 ≤ m_b) (h_m_c_nonneg : 0 ≤ m_c)
  (h_s_nonneg : 0 ≤ s) : 
  m_a^2 + m_b^2 + m_c^2 ≤ s^2 := 
by
  sorry

end triangle_heights_inequality_l1359_135939


namespace least_number_to_subtract_l1359_135914

theorem least_number_to_subtract (x : ℕ) (h : x = 7538 % 14) : (7538 - x) % 14 = 0 :=
by
  -- Proof goes here
  sorry

end least_number_to_subtract_l1359_135914


namespace shop_owner_cheat_percentage_l1359_135965

def CP : ℝ := 100
def cheating_buying : ℝ := 0.15  -- 15% cheating
def actual_cost_price : ℝ := CP * (1 + cheating_buying)  -- $115
def profit_percentage : ℝ := 43.75

theorem shop_owner_cheat_percentage :
  ∃ x : ℝ, profit_percentage = ((CP - x * CP / 100 - actual_cost_price) / actual_cost_price * 100) ∧ x = 65.26 :=
by
  sorry

end shop_owner_cheat_percentage_l1359_135965


namespace contrapositive_geometric_sequence_l1359_135905

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (b^2 ≠ a * c) → ¬geometric_sequence a b c :=
by
  intros h
  unfold geometric_sequence
  assumption

end contrapositive_geometric_sequence_l1359_135905


namespace remainder_when_divided_by_13_l1359_135904

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) (hk : N = 39 * k + 15) : N % 13 = 2 :=
sorry

end remainder_when_divided_by_13_l1359_135904


namespace no_integer_roots_l1359_135938

theorem no_integer_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) : ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_integer_roots_l1359_135938


namespace abs_neg_frac_l1359_135986

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l1359_135986


namespace probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l1359_135987

noncomputable def normalCDF (z : ℝ) : ℝ :=
  sorry -- Assuming some CDF function for the sake of the example.

variable (X : ℝ → ℝ)
variable (μ : ℝ := 3)
variable (σ : ℝ := sqrt 4)

-- 1. Proof that P(-1 < X < 5) = 0.8185
theorem probability_X_between_neg1_and_5 : 
  ((-1 < X) ∧ (X < 5) → (normalCDF 1 - normalCDF (-2)) = 0.8185) :=
  sorry

-- 2. Proof that P(X ≤ 8) = 0.9938
theorem probability_X_le_8 : 
  (X ≤ 8 → normalCDF 2.5 = 0.9938) :=
  sorry

-- 3. Proof that P(X ≥ 5) = 0.1587
theorem probability_X_ge_5 : 
  (X ≥ 5 → (1 - normalCDF 1) = 0.1587) :=
  sorry

-- 4. Proof that P(-3 < X < 9) = 0.9972
theorem probability_X_between_neg3_and_9 : 
  ((-3 < X) ∧ (X < 9) → (2 * normalCDF 3 - 1) = 0.9972) :=
  sorry

end probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l1359_135987


namespace value_of_sum_l1359_135950

theorem value_of_sum (a b c d : ℤ) 
  (h1 : a - b + c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 12 := 
  sorry

end value_of_sum_l1359_135950


namespace right_angled_triangle_area_l1359_135955

theorem right_angled_triangle_area 
  (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 18) (h3 : a^2 + b^2 + c^2 = 128) : 
  (1/2) * a * b = 9 :=
by
  -- Proof will be added here
  sorry

end right_angled_triangle_area_l1359_135955


namespace zachary_crunches_more_than_pushups_l1359_135944

def push_ups_zachary : ℕ := 46
def crunches_zachary : ℕ := 58

theorem zachary_crunches_more_than_pushups : crunches_zachary - push_ups_zachary = 12 := by
  sorry

end zachary_crunches_more_than_pushups_l1359_135944


namespace winning_percentage_l1359_135907

theorem winning_percentage (total_games first_games remaining_games : ℕ) 
                           (first_win_percent remaining_win_percent : ℝ)
                           (total_games_eq : total_games = 60)
                           (first_games_eq : first_games = 30)
                           (remaining_games_eq : remaining_games = 30)
                           (first_win_percent_eq : first_win_percent = 0.40)
                           (remaining_win_percent_eq : remaining_win_percent = 0.80) :
                           (first_win_percent * (first_games : ℝ) +
                            remaining_win_percent * (remaining_games : ℝ)) /
                           (total_games : ℝ) * 100 = 60 := sorry

end winning_percentage_l1359_135907


namespace billy_age_l1359_135921

variable (B J : ℕ)

theorem billy_age (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l1359_135921


namespace ellipse_foci_y_axis_l1359_135956

-- Given the equation of the ellipse x^2 + k * y^2 = 2 with foci on the y-axis,
-- prove that the range of k such that the ellipse is oriented with foci on the y-axis is (0, 1).
theorem ellipse_foci_y_axis (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  ∃ (a b : ℝ), a^2 + b^2 = 2 ∧ a > 0 ∧ b > 0 ∧ b / a = k ∧ x^2 + k * y^2 = 2 :=
sorry

end ellipse_foci_y_axis_l1359_135956


namespace W_3_7_eq_13_l1359_135993

-- Define the operation W
def W (x y : ℤ) : ℤ := y + 5 * x - x^2

-- State the theorem
theorem W_3_7_eq_13 : W 3 7 = 13 := by
  sorry

end W_3_7_eq_13_l1359_135993


namespace fruits_eaten_total_l1359_135989

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end fruits_eaten_total_l1359_135989


namespace value_of_expression_l1359_135926

theorem value_of_expression {p q : ℝ} (hp : 3 * p^2 + 9 * p - 21 = 0) (hq : 3 * q^2 + 9 * q - 21 = 0) : 
  (3 * p - 4) * (6 * q - 8) = 122 :=
by
  sorry

end value_of_expression_l1359_135926


namespace smallest_n_l1359_135981

-- Define the conditions as properties of integers
def connected (a b : ℕ): Prop := sorry -- Assume we have a definition for connectivity

def condition1 (a b n : ℕ) : Prop :=
  ¬connected a b → Nat.gcd (a^2 + b^2) n = 1

def condition2 (a b n : ℕ) : Prop :=
  connected a b → Nat.gcd (a^2 + b^2) n > 1

theorem smallest_n : ∃ n, n = 65 ∧ ∀ (a b : ℕ), condition1 a b n ∧ condition2 a b n := by
  sorry

end smallest_n_l1359_135981


namespace least_number_of_marbles_l1359_135929

theorem least_number_of_marbles :
  ∃ n, (∀ d ∈ ({3, 4, 5, 7, 8} : Set ℕ), d ∣ n) ∧ n = 840 :=
by
  sorry

end least_number_of_marbles_l1359_135929


namespace car_value_decrease_per_year_l1359_135978

theorem car_value_decrease_per_year 
  (initial_value : ℝ) (final_value : ℝ) (years : ℝ) (decrease_per_year : ℝ)
  (h1 : initial_value = 20000)
  (h2 : final_value = 14000)
  (h3 : years = 6)
  (h4 : initial_value - final_value = 6 * decrease_per_year) : 
  decrease_per_year = 1000 :=
sorry

end car_value_decrease_per_year_l1359_135978


namespace shark_feed_l1359_135985

theorem shark_feed (S : ℝ) (h1 : S + S/2 + 5 * S = 26) : S = 4 := 
by sorry

end shark_feed_l1359_135985


namespace intersecting_diagonals_probability_l1359_135976

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end intersecting_diagonals_probability_l1359_135976


namespace sabrina_basil_leaves_l1359_135912

-- Definitions of variables
variables (S B V : ℕ)

-- Conditions as definitions in Lean
def condition1 : Prop := B = 2 * S
def condition2 : Prop := S = V - 5
def condition3 : Prop := B + S + V = 29

-- Problem statement
theorem sabrina_basil_leaves (h1 : condition1 S B) (h2 : condition2 S V) (h3 : condition3 S B V) : B = 12 :=
by {
  sorry
}

end sabrina_basil_leaves_l1359_135912


namespace division_quotient_l1359_135979

theorem division_quotient (dividend divisor remainder quotient : ℕ) 
  (h₁ : dividend = 95) (h₂ : divisor = 15) (h₃ : remainder = 5)
  (h₄ : dividend = divisor * quotient + remainder) : quotient = 6 :=
by
  sorry

end division_quotient_l1359_135979


namespace select_two_subsets_union_six_elements_l1359_135909

def f (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * f (n - 1) - 1

theorem select_two_subsets_union_six_elements :
  f 6 = 365 :=
by
  sorry

end select_two_subsets_union_six_elements_l1359_135909


namespace min_value_of_reciprocals_l1359_135910

theorem min_value_of_reciprocals {x y a b : ℝ} 
  (h1 : 8 * x - y - 4 ≤ 0)
  (h2 : x + y + 1 ≥ 0)
  (h3 : y - 4 * x ≤ 0)
  (h4 : 2 = a * (1 / 2) + b * 1)
  (ha : a > 0)
  (hb : b > 0) :
  (1 / a) + (1 / b) = 9 / 2 :=
sorry

end min_value_of_reciprocals_l1359_135910


namespace inequality_proof_l1359_135932

theorem inequality_proof (a b c : ℝ) (h : a * b * c = 1) : 
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 := 
by
  sorry

end inequality_proof_l1359_135932


namespace proof_a_square_plus_a_plus_one_l1359_135995

theorem proof_a_square_plus_a_plus_one (a : ℝ) (h : 2 * (5 - a) * (6 + a) = 100) : a^2 + a + 1 = -19 := 
by 
  sorry

end proof_a_square_plus_a_plus_one_l1359_135995


namespace smallest_N_l1359_135916

theorem smallest_N (N : ℕ) : 
  (N = 484) ∧ 
  (∃ k : ℕ, 484 = 4 * k) ∧
  (∃ k : ℕ, 485 = 25 * k) ∧
  (∃ k : ℕ, 486 = 9 * k) ∧
  (∃ k : ℕ, 487 = 121 * k) :=
by
  -- Proof omitted (replaced by sorry)
  sorry

end smallest_N_l1359_135916


namespace total_apples_l1359_135918

-- Definitions and Conditions
variable (a : ℕ) -- original number of apples in the first pile (scaled integer type)
variable (n m : ℕ) -- arbitrary positions in the sequence

-- Arithmetic sequence of initial piles
def initial_piles := [a, 2*a, 3*a, 4*a, 5*a, 6*a]

-- Given condition transformations
def after_removal_distribution (initial_piles : List ℕ) (k : ℕ) : List ℕ :=
  match k with
  | 0 => [0, 2*a + 10, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 1 => [a + 10, 0, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 2 => [a + 10, 2*a + 20, 0, 4*a + 30, 5*a + 40, 6*a + 50]
  | 3 => [a + 10, 2*a + 20, 3*a + 30, 0, 5*a + 40, 6*a + 50]
  | 4 => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 0, 6*a + 50]
  | _ => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 5*a + 50, 0]

-- Prove the total number of apples
theorem total_apples : (a = 35) → (a + 2 * a + 3 * a + 4 * a + 5 * a + 6 * a = 735) :=
by
  intros h1
  sorry

end total_apples_l1359_135918


namespace find_f_2022_l1359_135911

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end find_f_2022_l1359_135911


namespace find_x_l1359_135931

noncomputable def geometric_series_sum (x: ℝ) : ℝ := 
  1 + x + 2 * x^2 + 3 * x^3 + 4 * x^4 + ∑' n: ℕ, (n + 1) * x^(n + 1)

theorem find_x (x: ℝ) (hx : geometric_series_sum x = 16) : x = 15 / 16 := 
by
  sorry

end find_x_l1359_135931


namespace population_in_terms_of_t_l1359_135972

noncomputable def boys_girls_teachers_total (b g t : ℕ) : ℕ :=
  b + g + t

theorem population_in_terms_of_t (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) :
  boys_girls_teachers_total b g t = 26 * t :=
by
  sorry

end population_in_terms_of_t_l1359_135972


namespace gcd_180_308_l1359_135964

theorem gcd_180_308 : Nat.gcd 180 308 = 4 :=
by
  sorry

end gcd_180_308_l1359_135964


namespace Kimiko_age_proof_l1359_135941

variables (Kimiko_age Kayla_age : ℕ)
variables (min_driving_age wait_years : ℕ)

def is_half_age (a b : ℕ) : Prop := a = b / 2
def minimum_driving_age (a b : ℕ) : Prop := a + b = 18

theorem Kimiko_age_proof
  (h1 : is_half_age Kayla_age Kimiko_age)
  (h2 : wait_years = 5)
  (h3 : minimum_driving_age Kayla_age wait_years) :
  Kimiko_age = 26 :=
sorry

end Kimiko_age_proof_l1359_135941


namespace part_one_part_two_l1359_135983

theorem part_one (a b : ℝ) (h : a ≠ 0) : |a + b| + |a - b| ≥ 2 * |a| :=
by sorry

theorem part_two (x : ℝ) : |x - 1| + |x - 2| ≤ 2 ↔ (1 / 2 : ℝ) ≤ x ∧ x ≤ (5 / 2 : ℝ) :=
by sorry

end part_one_part_two_l1359_135983


namespace sufficient_conditions_for_sum_positive_l1359_135974

variable {a b : ℝ}

theorem sufficient_conditions_for_sum_positive (h₃ : a + b > 2) (h₄ : a > 0 ∧ b > 0) : a + b > 0 :=
by {
  sorry
}

end sufficient_conditions_for_sum_positive_l1359_135974


namespace rebecca_haircut_charge_l1359_135922

-- Define the conditions
variable (H : ℕ) -- Charge for a haircut
def perm_charge : ℕ := 40
def dye_charge : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_today : ℕ := 4
def perms_today : ℕ := 1
def dye_jobs_today : ℕ := 2
def tips_today : ℕ := 50
def total_amount_end_day : ℕ := 310

-- State the proof problem
theorem rebecca_haircut_charge :
  4 * H + perms_today * perm_charge + dye_jobs_today * dye_charge + tips_today - dye_jobs_today * dye_cost = total_amount_end_day →
  H = 30 :=
by
  sorry

end rebecca_haircut_charge_l1359_135922


namespace total_pages_in_book_l1359_135913

variable (p1 p2 p_total : ℕ)
variable (read_first_four_days : p1 = 4 * 45)
variable (read_next_three_days : p2 = 3 * 52)
variable (total_until_last_day : p_total = p1 + p2 + 15)

theorem total_pages_in_book : p_total = 351 :=
by
  -- Introduce the conditions
  rw [read_first_four_days, read_next_three_days] at total_until_last_day
  sorry

end total_pages_in_book_l1359_135913


namespace sky_color_change_l1359_135934

theorem sky_color_change (hours: ℕ) (colors: ℕ) (minutes_per_hour: ℕ) 
                          (H1: hours = 2) 
                          (H2: colors = 12) 
                          (H3: minutes_per_hour = 60) : 
                          (hours * minutes_per_hour) / colors = 10 := 
by
  sorry

end sky_color_change_l1359_135934


namespace solution_set_inequality_l1359_135994

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

def f_increasing_on_pos : Prop := is_increasing_on f (Set.Ioi 0)

def f_at_one_zero : Prop := f 1 = 0

theorem solution_set_inequality : 
    is_odd f →
    f_increasing_on_pos →
    f_at_one_zero →
    {x : ℝ | x * (f x - f (-x)) < 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
sorry

end solution_set_inequality_l1359_135994


namespace green_faction_lies_more_l1359_135977

theorem green_faction_lies_more (r1 r2 r3 l1 l2 l3 : ℕ) 
  (h1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016) 
  (h2 : r1 + l2 + l3 = 1208) 
  (h3 : r2 + l1 + l3 = 908) 
  (h4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end green_faction_lies_more_l1359_135977
