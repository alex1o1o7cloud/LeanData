import Mathlib

namespace license_plate_count_l1891_189120

-- Formalize the conditions
def is_letter (c : Char) : Prop := 'a' ≤ c ∧ c ≤ 'z'
def is_digit (c : Char) : Prop := '0' ≤ c ∧ c ≤ '9'

-- Define the main proof problem
theorem license_plate_count :
  (26 * (25 + 9) * 26 * 10 = 236600) :=
by sorry

end license_plate_count_l1891_189120


namespace max_sn_at_16_l1891_189175

variable {a : ℕ → ℝ} -- the sequence a_n is represented by a

-- Conditions given in the problem
def isArithmetic (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := a n * a (n + 1) * a (n + 2)

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (bn a)

-- Condition: a_{12} = 3/8 * a_5 and a_12 > 0
def specificCondition (a : ℕ → ℝ) : Prop := a 12 = (3 / 8) * a 5 ∧ a 12 > 0

-- The theorem to prove that for S n, the maximum value is reached at n = 16
theorem max_sn_at_16 (a : ℕ → ℝ) (h_arithmetic : isArithmetic a) (h_condition : specificCondition a) :
  ∀ n : ℕ, Sn a n ≤ Sn a 16 := sorry

end max_sn_at_16_l1891_189175


namespace find_coefficients_l1891_189127

noncomputable def polynomial_h (x : ℚ) : ℚ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def polynomial_j (b c d x : ℚ) : ℚ := x^3 + b * x^2 + c * x + d

theorem find_coefficients :
  (∃ b c d : ℚ,
     (∀ s : ℚ, polynomial_h s = 0 → polynomial_j b c d (s^3) = 0) ∧
     (b, c, d) = (6, 12, 8)) :=
sorry

end find_coefficients_l1891_189127


namespace fraction_of_suitable_dishes_l1891_189158

theorem fraction_of_suitable_dishes {T : Type} (total_menu: ℕ) (vegan_dishes: ℕ) (vegan_fraction: ℚ) (gluten_inclusive_vegan_dishes: ℕ) (low_sugar_gluten_free_vegan_dishes: ℕ) 
(h1: vegan_dishes = 6)
(h2: vegan_fraction = 1/4)
(h3: gluten_inclusive_vegan_dishes = 4)
(h4: low_sugar_gluten_free_vegan_dishes = 1)
(h5: total_menu = vegan_dishes / vegan_fraction) :
(1 : ℚ) / (total_menu : ℚ) = (1 : ℚ) / 24 := 
by
  sorry

end fraction_of_suitable_dishes_l1891_189158


namespace strawberries_per_jar_l1891_189168

-- Let's define the conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
def jars_of_jam : ℕ := 40 / 4

-- Now we need to prove that the number of strawberries used in one jar of jam is 7.
theorem strawberries_per_jar : total_strawberries / jars_of_jam = 7 := by
  sorry

end strawberries_per_jar_l1891_189168


namespace exists_seq_two_reals_l1891_189136

theorem exists_seq_two_reals (x y : ℝ) (a : ℕ → ℝ) (h_recur : ∀ n, a (n + 2) = x * a (n + 1) + y * a n) :
  (∀ r > 0, ∃ i j : ℕ, 0 < |a i| ∧ |a i| < r ∧ r < |a j|) → ∃ x y : ℝ, ∃ a : ℕ → ℝ, (∀ n, a (n + 2) = x * a (n + 1) + y * a n) :=
by
  sorry

end exists_seq_two_reals_l1891_189136


namespace product_arithmetic_sequence_mod_100_l1891_189152

def is_arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ → Prop) : Prop :=
  ∀ k, n k → k = a + d * (k / d)

theorem product_arithmetic_sequence_mod_100 :
  ∀ P : ℕ,
    (∀ k, 7 ≤ k ∧ k ≤ 1999 ∧ ((k - 7) % 12 = 0) → P = k) →
    (P % 100 = 75) :=
by {
  sorry
}

end product_arithmetic_sequence_mod_100_l1891_189152


namespace measure_angle_C_l1891_189102

theorem measure_angle_C (A B C : ℝ) (h1 : A = 60) (h2 : B = 60) (h3 : C = 60 - 10) (sum_angles : A + B + C = 180) : C = 53.33 :=
by
  sorry

end measure_angle_C_l1891_189102


namespace max_value_frac_x1_x2_et_l1891_189180

theorem max_value_frac_x1_x2_et (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x * Real.exp x)
  (hg : ∀ x, g x = - (Real.log x) / x)
  (x1 x2 t : ℝ)
  (hx1 : f x1 = t)
  (hx2 : g x2 = t)
  (ht_pos : t > 0) :
  ∃ x1 x2, (f x1 = t ∧ g x2 = t) ∧ (∀ u v, (f u = t ∧ g v = t → u / (v * Real.exp t) ≤ 1 / Real.exp 1)) :=
by
  sorry

end max_value_frac_x1_x2_et_l1891_189180


namespace classroom_count_l1891_189182

-- Definitions for conditions
def average_age_all (sum_ages : ℕ) (num_people : ℕ) : ℕ := sum_ages / num_people
def average_age_excluding_teacher (sum_ages : ℕ) (num_people : ℕ) (teacher_age : ℕ) : ℕ :=
  (sum_ages - teacher_age) / (num_people - 1)

-- Theorem statement using the provided conditions
theorem classroom_count (x : ℕ) (h1 : average_age_all (11 * x) x = 11)
  (h2 : average_age_excluding_teacher (11 * x) x 30 = 10) : x = 20 :=
  sorry

end classroom_count_l1891_189182


namespace initial_pokemon_cards_l1891_189100

variables (x : ℕ)

theorem initial_pokemon_cards (h : x - 2 = 1) : x = 3 := 
sorry

end initial_pokemon_cards_l1891_189100


namespace SoccerBallPrices_SoccerBallPurchasingPlans_l1891_189179

theorem SoccerBallPrices :
  ∃ (priceA priceB : ℕ), priceA = 100 ∧ priceB = 80 ∧ (900 / priceA) = (720 / (priceB - 20)) :=
sorry

theorem SoccerBallPurchasingPlans :
  ∃ (m n : ℕ), (m + n = 90) ∧ (m ≥ 2 * n) ∧ (100 * m + 80 * n ≤ 8500) ∧
  (m ∈ Finset.range 66 \ Finset.range 60) ∧ 
  (∀ k ∈ Finset.range 66 \ Finset.range 60, 100 * k + 80 * (90 - k) ≥ 8400) :=
sorry

end SoccerBallPrices_SoccerBallPurchasingPlans_l1891_189179


namespace max_handshakes_l1891_189197

-- Definitions based on the given conditions
def num_people := 30
def handshake_formula (n : ℕ) := n * (n - 1) / 2

-- Formal statement of the problem
theorem max_handshakes : handshake_formula num_people = 435 :=
by
  -- Calculation here would be carried out in the proof, but not included in the statement itself.
  sorry

end max_handshakes_l1891_189197


namespace value_of_a_l1891_189105

theorem value_of_a (a b k : ℝ) (h1 : a = k / b^2) (h2 : a = 40) (h3 : b = 12) (h4 : b = 24) : a = 10 := 
by
  sorry

end value_of_a_l1891_189105


namespace iso_triangle_perimeter_l1891_189141

theorem iso_triangle_perimeter :
  ∃ p : ℕ, (p = 11 ∨ p = 13) ∧ ∃ a b : ℕ, a ≠ b ∧ a^2 - 8 * a + 15 = 0 ∧ b^2 - 8 * b + 15 = 0 :=
by
  sorry

end iso_triangle_perimeter_l1891_189141


namespace francie_remaining_money_l1891_189138

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end francie_remaining_money_l1891_189138


namespace christine_min_bottles_l1891_189156

theorem christine_min_bottles
  (fluid_ounces_needed : ℕ)
  (bottle_volume_ml : ℕ)
  (fluid_ounces_per_liter : ℝ)
  (liters_in_milliliter : ℕ)
  (required_bottles : ℕ)
  (h1 : fluid_ounces_needed = 45)
  (h2 : bottle_volume_ml = 200)
  (h3 : fluid_ounces_per_liter = 33.8)
  (h4 : liters_in_milliliter = 1000)
  (h5 : required_bottles = 7) :
  required_bottles = ⌈(fluid_ounces_needed * liters_in_milliliter) / (bottle_volume_ml * fluid_ounces_per_liter)⌉ := by
  sorry

end christine_min_bottles_l1891_189156


namespace inequality_proof_l1891_189193

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
  sorry

end inequality_proof_l1891_189193


namespace total_people_in_bus_l1891_189133

-- Definitions based on the conditions
def left_seats : Nat := 15
def right_seats := left_seats - 3
def people_per_seat := 3
def back_seat_people := 9

-- Theorem statement
theorem total_people_in_bus : 
  (left_seats * people_per_seat) +
  (right_seats * people_per_seat) + 
  back_seat_people = 90 := 
by sorry

end total_people_in_bus_l1891_189133


namespace p_sufficient_but_not_necessary_for_q_l1891_189164

def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x = 1 ∨ x = -2

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) := 
by {
  sorry
}

end p_sufficient_but_not_necessary_for_q_l1891_189164


namespace base_of_isosceles_triangle_l1891_189199

theorem base_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : 3 * a = 45) 
  (h₂ : 2 * b + c = 40) 
  (h₃ : b = a ∨ b = a) : c = 10 := 
sorry

end base_of_isosceles_triangle_l1891_189199


namespace contrapositive_proposition_contrapositive_version_l1891_189169

variable {a b : ℝ}

theorem contrapositive_proposition (h : a + b = 1) : a^2 + b^2 ≥ 1/2 :=
sorry

theorem contrapositive_version : a^2 + b^2 < 1/2 → a + b ≠ 1 :=
by
  intros h
  intro hab
  apply not_le.mpr h
  exact contrapositive_proposition hab

end contrapositive_proposition_contrapositive_version_l1891_189169


namespace initial_pinecones_l1891_189123

theorem initial_pinecones (P : ℝ) :
  (0.20 * P + 2 * 0.20 * P + 0.25 * (0.40 * P) = 0.70 * P - 0.10 * P) ∧ (0.30 * P = 600) → P = 2000 :=
by
  intro h
  sorry

end initial_pinecones_l1891_189123


namespace saree_final_price_l1891_189112

noncomputable def saree_original_price : ℝ := 5000
noncomputable def first_discount_rate : ℝ := 0.20
noncomputable def second_discount_rate : ℝ := 0.15
noncomputable def third_discount_rate : ℝ := 0.10
noncomputable def fourth_discount_rate : ℝ := 0.05
noncomputable def tax_rate : ℝ := 0.12
noncomputable def luxury_tax_rate : ℝ := 0.05
noncomputable def custom_fee : ℝ := 200
noncomputable def exchange_rate_to_usd : ℝ := 0.013

theorem saree_final_price :
  let price_after_first_discount := saree_original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  let price_after_third_discount := price_after_second_discount * (1 - third_discount_rate)
  let price_after_fourth_discount := price_after_third_discount * (1 - fourth_discount_rate)
  let tax := price_after_fourth_discount * tax_rate
  let luxury_tax := price_after_fourth_discount * luxury_tax_rate
  let total_charges := tax + luxury_tax + custom_fee
  let total_price_rs := price_after_fourth_discount + total_charges
  let final_price_usd := total_price_rs * exchange_rate_to_usd
  abs (final_price_usd - 46.82) < 0.01 :=
by sorry

end saree_final_price_l1891_189112


namespace travel_time_difference_l1891_189155

variable (x : ℝ)

theorem travel_time_difference 
  (distance : ℝ) 
  (speed_diff : ℝ)
  (time_diff_minutes : ℝ)
  (personB_speed : ℝ) 
  (personA_speed := personB_speed - speed_diff) 
  (time_diff_hours := time_diff_minutes / 60) :
  distance = 30 ∧ speed_diff = 3 ∧ time_diff_minutes = 40 ∧ personB_speed = x → 
    (30 / (x - 3)) - (30 / x) = 40 / 60 := 
by 
  sorry

end travel_time_difference_l1891_189155


namespace f_g_of_4_l1891_189129

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

theorem f_g_of_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  sorry

end f_g_of_4_l1891_189129


namespace max_movies_watched_l1891_189191

-- Conditions given in the problem
def movie_duration : Nat := 90
def tuesday_minutes : Nat := 4 * 60 + 30
def tuesday_movies : Nat := tuesday_minutes / movie_duration
def wednesday_movies : Nat := 2 * tuesday_movies

-- Problem statement: Total movies watched in two days
theorem max_movies_watched : 
  tuesday_movies + wednesday_movies = 9 := 
by
  -- We add the placeholder for the proof here
  sorry

end max_movies_watched_l1891_189191


namespace triangle_inequality_l1891_189174

-- Define the nondegenerate condition for the triangle's side lengths.
def nondegenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter condition for the triangle.
def triangle_perimeter (a b c : ℝ) (p : ℝ) : Prop :=
  a + b + c = p

-- The main theorem to prove the given inequality.
theorem triangle_inequality (a b c : ℝ) (h_non_deg : nondegenerate_triangle a b c) (h_perim : triangle_perimeter a b c 1) :
  abs ((a - b) / (c + a * b)) + abs ((b - c) / (a + b * c)) + abs ((c - a) / (b + a * c)) < 2 :=
by
  sorry

end triangle_inequality_l1891_189174


namespace community_theater_ticket_sales_l1891_189157

theorem community_theater_ticket_sales (A C : ℕ) 
  (h1 : 12 * A + 4 * C = 840) 
  (h2 : A + C = 130) :
  A = 40 :=
sorry

end community_theater_ticket_sales_l1891_189157


namespace constant_temperature_l1891_189171

def stable_system (T : ℤ × ℤ × ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), T (a, b, c) = (1 / 6) * (T (a + 1, b, c) + T (a - 1, b, c) + T (a, b + 1, c) + T (a, b - 1, c) + T (a, b, c + 1) + T (a, b, c - 1))

theorem constant_temperature (T : ℤ × ℤ × ℤ → ℝ) 
    (h1 : ∀ (x : ℤ × ℤ × ℤ), 0 ≤ T x ∧ T x ≤ 1)
    (h2 : stable_system T) : 
  ∃ c : ℝ, ∀ x : ℤ × ℤ × ℤ, T x = c := 
sorry

end constant_temperature_l1891_189171


namespace icosahedron_inscribed_in_cube_l1891_189165

theorem icosahedron_inscribed_in_cube (a m : ℝ) (points_on_faces : Fin 6 → Fin 2 → ℝ × ℝ × ℝ) :
  (∃ points : Fin 12 → ℝ × ℝ × ℝ, 
   (∀ i : Fin 12, ∃ j : Fin 6, (points i).fst = (points_on_faces j 0).fst ∨ (points i).fst = (points_on_faces j 1).fst) ∧
   ∃ segments : Fin 12 → Fin 12 → ℝ, 
   (∀ i j : Fin 12, (segments i j) = m ∨ (segments i j) = a)) →
  a^2 - a*m - m^2 = 0 := sorry

end icosahedron_inscribed_in_cube_l1891_189165


namespace set_equality_l1891_189177

def M : Set ℝ := {x | x^2 - x > 0}

def N : Set ℝ := {x | 1 / x < 1}

theorem set_equality : M = N := 
by
  sorry

end set_equality_l1891_189177


namespace three_is_square_root_of_nine_l1891_189159

theorem three_is_square_root_of_nine :
  ∃ x : ℝ, x * x = 9 ∧ x = 3 :=
sorry

end three_is_square_root_of_nine_l1891_189159


namespace polygon_sides_twice_diagonals_l1891_189196

theorem polygon_sides_twice_diagonals (n : ℕ) (h1 : n ≥ 3) (h2 : n * (n - 3) / 2 = 2 * n) : n = 7 :=
sorry

end polygon_sides_twice_diagonals_l1891_189196


namespace avg_distance_is_600_l1891_189122

-- Assuming Mickey runs half as many times as Johnny
def num_laps_johnny := 4
def num_laps_mickey := num_laps_johnny / 2
def lap_distance := 200

-- Calculating distances
def distance_johnny := num_laps_johnny * lap_distance
def distance_mickey := num_laps_mickey * lap_distance

-- Total distance run by both Mickey and Johnny
def total_distance := distance_johnny + distance_mickey

-- Average distance run by Johnny and Mickey
def avg_distance := total_distance / 2

-- Prove that the average distance run by Johnny and Mickey is 600 meters
theorem avg_distance_is_600 : avg_distance = 600 :=
by
  sorry

end avg_distance_is_600_l1891_189122


namespace common_ratio_of_geometric_series_l1891_189149

theorem common_ratio_of_geometric_series 
  (a1 q : ℝ) 
  (h1 : a1 + a1 * q^2 = 5) 
  (h2 : a1 * q + a1 * q^3 = 10) : 
  q = 2 := 
by 
  sorry

end common_ratio_of_geometric_series_l1891_189149


namespace remainder_71_3_73_5_mod_8_l1891_189163

theorem remainder_71_3_73_5_mod_8 :
  (71^3) * (73^5) % 8 = 7 :=
by {
  -- hint, use the conditions given: 71 ≡ -1 (mod 8) and 73 ≡ 1 (mod 8)
  sorry
}

end remainder_71_3_73_5_mod_8_l1891_189163


namespace sets_are_equal_l1891_189143

-- Define sets according to the given options
def option_a_M : Set (ℕ × ℕ) := {(3, 2)}
def option_a_N : Set (ℕ × ℕ) := {(2, 3)}

def option_b_M : Set ℕ := {3, 2}
def option_b_N : Set (ℕ × ℕ) := {(3, 2)}

def option_c_M : Set (ℕ × ℕ) := {(x, y) | x + y = 1}
def option_c_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_d_M : Set ℕ := {3, 2}
def option_d_N : Set ℕ := {2, 3}

-- Proof goal
theorem sets_are_equal : option_d_M = option_d_N :=
sorry

end sets_are_equal_l1891_189143


namespace hexagon_circle_radius_l1891_189137

theorem hexagon_circle_radius (r : ℝ) :
  let side_length := 3
  let probability := (1 : ℝ) / 3
  (probability = 1 / 3) →
  r = 12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2) :=
by
  -- Begin proof here
  sorry

end hexagon_circle_radius_l1891_189137


namespace Sarah_ate_one_apple_l1891_189101

theorem Sarah_ate_one_apple:
  ∀ (total_apples apples_given_to_teachers apples_given_to_friends apples_left: ℕ), 
  total_apples = 25 →
  apples_given_to_teachers = 16 →
  apples_given_to_friends = 5 →
  apples_left = 3 →
  total_apples - (apples_given_to_teachers + apples_given_to_friends + apples_left) = 1 :=
by
  intros total_apples apples_given_to_teachers apples_given_to_friends apples_left
  intro ht ht gt hf
  sorry

end Sarah_ate_one_apple_l1891_189101


namespace dartboard_area_ratio_l1891_189140

theorem dartboard_area_ratio
    (larger_square_side_length : ℝ)
    (inner_square_side_length : ℝ)
    (angle_division : ℝ)
    (s : ℝ)
    (p : ℝ)
    (h1 : larger_square_side_length = 4)
    (h2 : inner_square_side_length = 2)
    (h3 : angle_division = 45)
    (h4 : s = 1/4)
    (h5 : p = 3) :
    p / s = 12 :=
by
    sorry

end dartboard_area_ratio_l1891_189140


namespace part1_part2_l1891_189126

-- Let m be the cost price this year
-- Let x be the selling price per bottle
-- Assuming:
-- 1. The cost price per bottle increased by 4 yuan this year compared to last year.
-- 2. The quantity of detergent purchased for 1440 yuan this year equals to the quantity purchased for 1200 yuan last year.
-- 3. The selling price per bottle is 36 yuan with 600 bottles sold per week.
-- 4. Weekly sales increase by 100 bottles for every 1 yuan reduction in price.
-- 5. The selling price cannot be lower than the cost price.

-- Definition for improved readability:
def costPriceLastYear (m : ℕ) : ℕ := m - 4

-- Quantity equations
def quantityPurchasedThisYear (m : ℕ) : ℕ := 1440 / m
def quantityPurchasedLastYear (m : ℕ) : ℕ := 1200 / (costPriceLastYear m)

-- Profit Function
def profitFunction (m x : ℝ) : ℝ :=
  (x - m) * (600 + 100 * (36 - x))

-- Maximum Profit and Best Selling Price
def maxProfit : ℝ := 8100
def bestSellingPrice : ℝ := 33

theorem part1 (m : ℕ) (h₁ : 1440 / m = 1200 / costPriceLastYear m) : m = 24 := by
  sorry  -- Will be proved later

theorem part2 (m : ℝ) (x : ℝ)
    (h₀ : m = 24)
    (hx : 600 + 100 * (36 - x) > 0)
    (hx₁ : x ≥ m)
    : profitFunction m x ≤ maxProfit ∧ (∃! (y : ℝ), y = bestSellingPrice ∧ profitFunction m y = maxProfit) := by
  sorry  -- Will be proved later

end part1_part2_l1891_189126


namespace total_raisins_l1891_189184

theorem total_raisins (yellow raisins black raisins : ℝ) (h_yellow : yellow = 0.3) (h_black : black = 0.4) : yellow + black = 0.7 := 
by
  sorry

end total_raisins_l1891_189184


namespace expression_divisible_by_84_l1891_189110

theorem expression_divisible_by_84 (p : ℕ) (hp : p > 0) : (4 ^ (2 * p) - 3 ^ (2 * p) - 7) % 84 = 0 :=
by
  sorry

end expression_divisible_by_84_l1891_189110


namespace solve_eq1_solve_eq2_l1891_189162

-- Proof for the first equation
theorem solve_eq1 (y : ℝ) : 8 * y - 4 * (3 * y + 2) = 6 ↔ y = -7 / 2 := 
by 
  sorry

-- Proof for the second equation
theorem solve_eq2 (x : ℝ) : 2 - (x + 2) / 3 = x - (x - 1) / 6 ↔ x = 1 := 
by 
  sorry

end solve_eq1_solve_eq2_l1891_189162


namespace toads_l1891_189147

theorem toads (Tim Jim Sarah : ℕ) 
  (h1 : Jim = Tim + 20) 
  (h2 : Sarah = 2 * Jim) 
  (h3 : Sarah = 100) : Tim = 30 := 
by 
  -- Proof will be provided later
  sorry

end toads_l1891_189147


namespace find_prob_p_l1891_189185

variable (p : ℚ)

theorem find_prob_p (h : 15 * p^4 * (1 - p)^2 = 500 / 2187) : p = 3 / 7 := 
  sorry

end find_prob_p_l1891_189185


namespace draw_at_least_two_first_grade_products_l1891_189117

theorem draw_at_least_two_first_grade_products :
  let total_products := 9
  let first_grade := 4
  let second_grade := 3
  let third_grade := 2
  let total_draws := 4
  let ways_to_draw := Nat.choose total_products total_draws
  let ways_no_first_grade := Nat.choose (second_grade + third_grade) total_draws
  let ways_one_first_grade := Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (total_draws - 1)
  ways_to_draw - ways_no_first_grade - ways_one_first_grade = 81 := sorry

end draw_at_least_two_first_grade_products_l1891_189117


namespace cookies_ratio_l1891_189170

theorem cookies_ratio (T : ℝ) (h1 : 0 ≤ T) (h_total : 5 + T + 1.4 * T = 29) : T / 5 = 2 :=
by sorry

end cookies_ratio_l1891_189170


namespace complex_division_example_l1891_189186

theorem complex_division_example (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + i) = (1/2 : ℂ) - (3/2 : ℂ) * i :=
by
  -- proof would go here
  sorry

end complex_division_example_l1891_189186


namespace same_face_probability_l1891_189167

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l1891_189167


namespace quadratic_roots_p_l1891_189145

noncomputable def equation : Type* := sorry

theorem quadratic_roots_p
  (α β : ℝ)
  (K : ℝ)
  (h1 : 3 * α ^ 2 + 7 * α + K = 0)
  (h2 : 3 * β ^ 2 + 7 * β + K = 0)
  (sum_roots : α + β = -7 / 3)
  (prod_roots : α * β = K / 3)
  : ∃ p : ℝ, p = -70 / 9 + 2 * K / 3 := 
sorry

end quadratic_roots_p_l1891_189145


namespace find_cost_l1891_189161

def cost_of_article (C : ℝ) (G : ℝ) : Prop :=
  (580 = C + G) ∧ (600 = C + G + 0.05 * G)

theorem find_cost (C : ℝ) (G : ℝ) (h : cost_of_article C G) : C = 180 :=
by
  sorry

end find_cost_l1891_189161


namespace necessary_but_not_sufficient_condition_l1891_189119

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1891_189119


namespace stadium_height_l1891_189130

theorem stadium_height
  (l w d : ℕ) (h : ℕ) 
  (hl : l = 24) 
  (hw : w = 18) 
  (hd : d = 34) 
  (h_eq : d^2 = l^2 + w^2 + h^2) : 
  h = 16 := by 
  sorry

end stadium_height_l1891_189130


namespace ab_divisible_by_six_l1891_189176

def last_digit (n : ℕ) : ℕ :=
  (2 ^ n) % 10

def b_value (n : ℕ) (a : ℕ) : ℕ :=
  2 ^ n - a

theorem ab_divisible_by_six (n : ℕ) (h : n > 3) :
  let a := last_digit n
  let b := b_value n a
  ∃ k : ℕ, ab = 6 * k :=
by
  sorry

end ab_divisible_by_six_l1891_189176


namespace combined_mpg_l1891_189189

theorem combined_mpg
  (R_eff : ℝ) (T_eff : ℝ)
  (R_dist : ℝ) (T_dist : ℝ)
  (H_R_eff : R_eff = 35)
  (H_T_eff : T_eff = 15)
  (H_R_dist : R_dist = 420)
  (H_T_dist : T_dist = 300)
  : (R_dist + T_dist) / (R_dist / R_eff + T_dist / T_eff) = 22.5 := 
by
  rw [H_R_eff, H_T_eff, H_R_dist, H_T_dist]
  -- Proof steps would go here, but we'll use sorry to skip it.
  sorry

end combined_mpg_l1891_189189


namespace triangle_angles_inequality_l1891_189104

theorem triangle_angles_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := 
sorry

end triangle_angles_inequality_l1891_189104


namespace find_y_l1891_189125

theorem find_y (DEG EFG y : ℝ) 
  (h1 : DEG = 150)
  (h2 : EFG = 40)
  (h3 : DEG = EFG + y) :
  y = 110 :=
by
  sorry

end find_y_l1891_189125


namespace ball_hits_ground_l1891_189121

theorem ball_hits_ground :
  ∃ (t : ℝ), (t = 2) ∧ (-4.9 * t^2 + 5.7 * t + 7 = 0) :=
sorry

end ball_hits_ground_l1891_189121


namespace bridge_length_is_correct_l1891_189113

def speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def crossing_time_s : ℝ := 30

noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_m_s * crossing_time_s
noncomputable def bridge_length_m : ℝ := total_distance_m - train_length_m

theorem bridge_length_is_correct : bridge_length_m = 255 := by
  sorry

end bridge_length_is_correct_l1891_189113


namespace minimum_value_is_4_l1891_189144

noncomputable def minimum_value (m n : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ m + n = 1 then (1 / m) + (1 / n) else 0

theorem minimum_value_is_4 :
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m + n = 1) →
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m + n = 1 ∧ minimum_value m n = 4 :=
by
  sorry

end minimum_value_is_4_l1891_189144


namespace junior_score_is_90_l1891_189190

theorem junior_score_is_90 {n : ℕ} (hn : n > 0)
    (j : ℕ := n / 5) (s : ℕ := 4 * n / 5)
    (overall_avg : ℝ := 86)
    (senior_avg : ℝ := 85)
    (junior_score : ℝ)
    (h1 : 20 * j = n)
    (h2 : 80 * s = n * 4)
    (h3 : overall_avg * n = 86 * n)
    (h4 : senior_avg * s = 85 * s)
    (h5 : j * junior_score = overall_avg * n - senior_avg * s) :
    junior_score = 90 :=
by
  sorry

end junior_score_is_90_l1891_189190


namespace find_additional_discount_l1891_189142

noncomputable def calculate_additional_discount (msrp : ℝ) (regular_discount_percent : ℝ) (final_price : ℝ) : ℝ :=
  let regular_discounted_price := msrp * (1 - regular_discount_percent / 100)
  let additional_discount_percent := ((regular_discounted_price - final_price) / regular_discounted_price) * 100
  additional_discount_percent

theorem find_additional_discount :
  calculate_additional_discount 35 30 19.6 = 20 :=
by
  sorry

end find_additional_discount_l1891_189142


namespace cube_side_length_and_combined_volume_l1891_189139

theorem cube_side_length_and_combined_volume
  (surface_area_large_cube : ℕ)
  (h_surface_area : surface_area_large_cube = 864)
  (side_length_large_cube : ℕ)
  (combined_volume : ℕ) :
  side_length_large_cube = 12 ∧ combined_volume = 1728 :=
by
  -- Since we only need the statement, the proof steps are not included.
  sorry

end cube_side_length_and_combined_volume_l1891_189139


namespace train_speed_l1891_189166

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end train_speed_l1891_189166


namespace product_of_three_consecutive_not_div_by_5_adjacency_l1891_189148

theorem product_of_three_consecutive_not_div_by_5_adjacency (a b c : ℕ) (h₁ : a + 1 = b) (h₂ : b + 1 = c) (h₃ : a % 5 ≠ 0) (h₄ : b % 5 ≠ 0) (h₅ : c % 5 ≠ 0) :
  ((a * b * c) % 5 = 1) ∨ ((a * b * c) % 5 = 4) := 
sorry

end product_of_three_consecutive_not_div_by_5_adjacency_l1891_189148


namespace mary_remaining_money_l1891_189131

variable (p : ℝ) -- p is the price per drink in dollars

def drinks_cost : ℝ := 3 * p
def medium_pizzas_cost : ℝ := 2 * (2 * p)
def large_pizza_cost : ℝ := 3 * p

def total_cost : ℝ := drinks_cost p + medium_pizzas_cost p + large_pizza_cost p

theorem mary_remaining_money : 
  30 - total_cost p = 30 - 10 * p := 
by
  sorry

end mary_remaining_money_l1891_189131


namespace square_area_l1891_189194

theorem square_area (side_length : ℕ) (h : side_length = 17) : side_length * side_length = 289 :=
by sorry

end square_area_l1891_189194


namespace sqrt_expression_simplification_l1891_189198

theorem sqrt_expression_simplification :
  (Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt (3) - 1| + Real.sqrt 3) = -13 / 4 + 2 * Real.sqrt 3 :=
by
  have h1 : Real.sqrt (1 / 16) = 1 / 4 := sorry
  have h2 : Real.sqrt (25 / 4) = 5 / 2 := sorry
  have h3 : |Real.sqrt 3 - 1| = Real.sqrt 3 - 1 := sorry
  linarith [h1, h2, h3]

end sqrt_expression_simplification_l1891_189198


namespace gateway_academy_problem_l1891_189150

theorem gateway_academy_problem :
  let total_students := 100
  let students_like_skating := 0.4 * total_students
  let students_dislike_skating := total_students - students_like_skating
  let like_and_say_like := 0.7 * students_like_skating
  let like_and_say_dislike := students_like_skating - like_and_say_like
  let dislike_and_say_dislike := 0.8 * students_dislike_skating
  let dislike_and_say_like := students_dislike_skating - dislike_and_say_dislike
  let says_dislike := like_and_say_dislike + dislike_and_say_dislike
  (like_and_say_dislike / says_dislike) = 0.2 :=
by
  sorry

end gateway_academy_problem_l1891_189150


namespace distance_to_place_equals_2_point_25_l1891_189146

-- Definitions based on conditions
def rowing_speed : ℝ := 4
def river_speed : ℝ := 2
def total_time_hours : ℝ := 1.5

-- Downstream speed = rowing_speed + river_speed
def downstream_speed : ℝ := rowing_speed + river_speed
-- Upstream speed = rowing_speed - river_speed
def upstream_speed : ℝ := rowing_speed - river_speed

-- Define the distance d
def distance (d : ℝ) : Prop :=
  (d / downstream_speed + d / upstream_speed = total_time_hours)

-- The theorem statement
theorem distance_to_place_equals_2_point_25 :
  ∃ d : ℝ, distance d ∧ d = 2.25 :=
by
  sorry

end distance_to_place_equals_2_point_25_l1891_189146


namespace diophantine_solution_exists_l1891_189183

theorem diophantine_solution_exists (D : ℤ) : 
  ∃ (x y z : ℕ), x^2 - D * y^2 = z^2 ∧ ∃ m n : ℕ, m^2 > D * n^2 :=
sorry

end diophantine_solution_exists_l1891_189183


namespace f_transform_l1891_189178

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 4 * x - 5

theorem f_transform (x h : ℝ) : 
  f (x + h) - f x = 6 * x ^ 2 - 6 * x + 6 * x * h + 2 * h ^ 2 - 3 * h + 4 := 
by
  sorry

end f_transform_l1891_189178


namespace calc_expression_l1891_189115

theorem calc_expression : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end calc_expression_l1891_189115


namespace all_edges_same_color_l1891_189188

-- Define the vertices in the two pentagons and the set of all vertices
inductive vertex
| A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | B5
open vertex

-- Predicate to identify edges between vertices
def edge (v1 v2 : vertex) : Prop :=
  match (v1, v2) with
  | (A1, A2) | (A2, A3) | (A3, A4) | (A4, A5) | (A5, A1) => true
  | (B1, B2) | (B2, B3) | (B3, B4) | (B4, B5) | (B5, B1) => true
  | (A1, B1) | (A1, B2) | (A1, B3) | (A1, B4) | (A1, B5) => true
  | (A2, B1) | (A2, B2) | (A2, B3) | (A2, B4) | (A2, B5) => true
  | (A3, B1) | (A3, B2) | (A3, B3) | (A3, B4) | (A3, B5) => true
  | (A4, B1) | (A4, B2) | (A4, B3) | (A4, B4) | (A4, B5) => true
  | (A5, B1) | (A5, B2) | (A5, B3) | (A5, B4) | (A5, B5) => true
  | _ => false

-- Edge coloring predicate 'black' or 'white'
inductive color
| black | white
open color

def edge_color (v1 v2 : vertex) : color → Prop :=
  sorry -- Coloring function needs to be defined accordingly

-- Predicate to check for monochrome triangles
def no_monochrome_triangle : Prop :=
  ∀ v1 v2 v3 : vertex,
    (edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v1) →
    ¬ (∃ c : color, edge_color v1 v2 c ∧ edge_color v2 v3 c ∧ edge_color v3 v1 c)

-- Main theorem statement
theorem all_edges_same_color (no_mt : no_monochrome_triangle) :
  ∃ c : color, ∀ v1 v2 : vertex,
    (edge v1 v2 ∧ (v1 = A1 ∨ v1 = A2 ∨ v1 = A3 ∨ v1 = A4 ∨ v1 = A5) ∧
                 (v2 = A1 ∨ v2 = A2 ∨ v2 = A3 ∨ v2 = A4 ∨ v2 = A5) ) →
    edge_color v1 v2 c ∧
    (edge v1 v2 ∧ (v1 = B1 ∨ v1 = B2 ∨ v1 = B3 ∨ v1 = B4 ∨ v1 = B5) ∧
                 (v2 = B1 ∨ v2 = B2 ∨ v2 = B3 ∨ v2 = B4 ∨ v2 = B5) ) →
    edge_color v1 v2 c := sorry

end all_edges_same_color_l1891_189188


namespace ratio_of_marbles_l1891_189114

noncomputable def marble_ratio : ℕ :=
  let initial_marbles := 40
  let marbles_after_breakfast := initial_marbles - 3
  let marbles_after_lunch := marbles_after_breakfast - 5
  let marbles_after_moms_gift := marbles_after_lunch + 12
  let final_marbles := 54
  let marbles_given_back_by_Susie := final_marbles - marbles_after_moms_gift
  marbles_given_back_by_Susie / 5

theorem ratio_of_marbles : marble_ratio = 2 := by
  -- proof steps would go here
  sorry

end ratio_of_marbles_l1891_189114


namespace num_integers_achievable_le_2014_l1891_189173

def floor_div (x : ℤ) : ℤ := x / 2

def button1 (x : ℤ) : ℤ := floor_div x

def button2 (x : ℤ) : ℤ := 4 * x + 1

def num_valid_sequences (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 2
  else num_valid_sequences (n - 1) + num_valid_sequences (n - 2)

theorem num_integers_achievable_le_2014 :
  num_valid_sequences 11 = 233 :=
  by
    -- Proof starts here
    sorry

end num_integers_achievable_le_2014_l1891_189173


namespace valuable_files_count_l1891_189153

theorem valuable_files_count 
    (initial_files : ℕ) 
    (deleted_fraction_initial : ℚ) 
    (additional_files : ℕ) 
    (irrelevant_fraction_additional : ℚ) 
    (h1 : initial_files = 800) 
    (h2 : deleted_fraction_initial = (70:ℚ) / 100)
    (h3 : additional_files = 400)
    (h4 : irrelevant_fraction_additional = (3:ℚ) / 5) : 
    (initial_files - ⌊deleted_fraction_initial * initial_files⌋ + additional_files - ⌊irrelevant_fraction_additional * additional_files⌋) = 400 :=
by sorry

end valuable_files_count_l1891_189153


namespace inequality_proof_l1891_189172

variable (k : ℕ) (a b c : ℝ)
variables (hk : 0 < k) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_proof (hk : k > 0) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * (1 - a^k) + b * (1 - (a + b)^k) + c * (1 - (a + b + c)^k) < k / (k + 1) :=
sorry

end inequality_proof_l1891_189172


namespace vector_parallel_l1891_189106

theorem vector_parallel {x : ℝ} (h : (4 / x) = (-2 / 5)) : x = -10 :=
  by
  sorry

end vector_parallel_l1891_189106


namespace length_of_train_l1891_189107

-- We define the conditions
def crosses_platform_1 (L : ℝ) : Prop := 
  let v := (L + 100) / 15
  v = (L + 100) / 15

def crosses_platform_2 (L : ℝ) : Prop := 
  let v := (L + 250) / 20
  v = (L + 250) / 20

-- We state the main theorem we need to prove
theorem length_of_train :
  ∃ L : ℝ, crosses_platform_1 L ∧ crosses_platform_2 L ∧ (L = 350) :=
sorry

end length_of_train_l1891_189107


namespace tangent_line_eq_l1891_189108

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_eq
  (a b : ℝ)
  (h1 : 3 + 2*a + b = 2*a)
  (h2 : 12 + 4*a + b = -b)
  : ∀ x y : ℝ , (f a b 1 = -5/2 ∧
  y - (f a b 1) = -3 * (x - 1))
  → (6*x + 2*y - 1 = 0) :=
by
  sorry

end tangent_line_eq_l1891_189108


namespace sum_of_roots_eq_k_div_4_l1891_189132

variables {k d y_1 y_2 : ℝ}

theorem sum_of_roots_eq_k_div_4 (h1 : y_1 ≠ y_2)
                                  (h2 : 4 * y_1^2 - k * y_1 = d)
                                  (h3 : 4 * y_2^2 - k * y_2 = d) :
  y_1 + y_2 = k / 4 :=
sorry

end sum_of_roots_eq_k_div_4_l1891_189132


namespace ratio_movies_allowance_l1891_189109

variable (M A : ℕ)
variable (weeklyAllowance moneyEarned endMoney : ℕ)
variable (H1 : weeklyAllowance = 8)
variable (H2 : moneyEarned = 8)
variable (H3 : endMoney = 12)
variable (H4 : weeklyAllowance + moneyEarned - M = endMoney)
variable (H5 : A = 8)
variable (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1)

theorem ratio_movies_allowance (M A : ℕ) 
  (weeklyAllowance moneyEarned endMoney : ℕ)
  (H1 : weeklyAllowance = 8)
  (H2 : moneyEarned = 8)
  (H3 : endMoney = 12)
  (H4 : weeklyAllowance + moneyEarned - M = endMoney)
  (H5 : A = 8)
  (H6 : M = weeklyAllowance + moneyEarned - endMoney / 1) :
  M / A = 1 / 2 :=
sorry

end ratio_movies_allowance_l1891_189109


namespace cookies_left_l1891_189118

theorem cookies_left (days_baking : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) (frank_eats_per_day : ℕ) (ted_eats_on_sixth_day : ℕ) :
  trays_per_day * cookies_per_tray * days_baking - frank_eats_per_day * days_baking - ted_eats_on_sixth_day = 134 :=
by
  have days_baking := 6
  have trays_per_day := 2
  have cookies_per_tray := 12
  have frank_eats_per_day := 1
  have ted_eats_on_sixth_day := 4
  sorry

end cookies_left_l1891_189118


namespace center_cell_value_l1891_189192

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l1891_189192


namespace find_n_in_geometric_series_l1891_189128

theorem find_n_in_geometric_series :
  let a1 : ℕ := 15
  let a2 : ℕ := 5
  let r1 := a2 / a1
  let S1 := a1 / (1 - r1: ℝ)
  let S2 := 3 * S1
  let r2 := (5 + n) / a1
  S2 = 15 / (1 - r2) →
  n = 20 / 3 :=
by
  sorry

end find_n_in_geometric_series_l1891_189128


namespace problem_solution_set_l1891_189111

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l1891_189111


namespace avg_salary_rest_of_workers_l1891_189103

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_tech : ℝ) (total_workers : ℕ)
  (total_avg_salary : avg_all = 8000) (tech_avg_salary : avg_tech = 12000) (workers_count : total_workers = 30) :
  (20 * (total_workers * avg_all - 10 * avg_tech) / 20) = 6000 :=
by
  sorry

end avg_salary_rest_of_workers_l1891_189103


namespace exists_seq_nat_lcm_decreasing_l1891_189181

-- Natural number sequence and conditions
def seq_nat_lcm_decreasing : Prop :=
  ∃ (a : Fin 100 → ℕ), 
  ((∀ i j : Fin 100, i < j → a i < a j) ∧
  (∀ (i : Fin 99), Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))))

theorem exists_seq_nat_lcm_decreasing : seq_nat_lcm_decreasing :=
  sorry

end exists_seq_nat_lcm_decreasing_l1891_189181


namespace polar_to_rect_l1891_189187

theorem polar_to_rect (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (2.5, 5 * Real.sqrt 3 / 2) :=
by
  rw [hr, hθ]
  sorry

end polar_to_rect_l1891_189187


namespace segment_length_greater_than_inradius_sqrt_two_l1891_189151

variables {a b c : ℝ} -- sides of the triangle
variables {P Q : ℝ} -- points on sides of the triangle
variables {S_ABC S_PCQ : ℝ} -- areas of the triangles
variables {s : ℝ} -- semi-perimeter of the triangle
variables {r : ℝ} -- radius of the inscribed circle
variables {ℓ : ℝ} -- length of segment dividing the triangle's area

-- Given conditions in the form of assumptions
variables (h1 : S_PCQ = S_ABC / 2)
variables (h2 : PQ = ℓ)
variables (h3 : r = S_ABC / s)

-- The statement of the theorem
theorem segment_length_greater_than_inradius_sqrt_two
  (h1 : S_PCQ = S_ABC / 2) 
  (h2 : PQ = ℓ) 
  (h3 : r = S_ABC / s)
  (h4 : s = (a + b + c) / 2) 
  (h5 : S_ABC = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h6 : ℓ^2 = a^2 + b^2 - (a^2 + b^2 - c^2) / 2) :
  ℓ > r * Real.sqrt 2 :=
sorry

end segment_length_greater_than_inradius_sqrt_two_l1891_189151


namespace cos_225_degrees_l1891_189154

theorem cos_225_degrees : Real.cos (225 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_225_degrees_l1891_189154


namespace arithmetic_sequence_probability_correct_l1891_189116

noncomputable def arithmetic_sequence_probability : ℚ := 
  let total_ways := Nat.choose 5 3
  let arithmetic_sequences := 4
  (arithmetic_sequences : ℚ) / (total_ways : ℚ)

theorem arithmetic_sequence_probability_correct :
  arithmetic_sequence_probability = 0.4 := by
  unfold arithmetic_sequence_probability
  sorry

end arithmetic_sequence_probability_correct_l1891_189116


namespace negation_of_universal_quantification_l1891_189160

theorem negation_of_universal_quantification (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ ∃ x ∈ S, |x| ≤ 1 :=
by
  sorry

end negation_of_universal_quantification_l1891_189160


namespace arithmetic_series_sum_l1891_189195

theorem arithmetic_series_sum (n P q S₃n : ℕ) (h₁ : 2 * S₃n = 3 * P - q) : S₃n = 3 * P - q :=
by
  sorry

end arithmetic_series_sum_l1891_189195


namespace not_perfect_square_of_sum_300_l1891_189124

theorem not_perfect_square_of_sum_300 : ¬(∃ n : ℕ, n = 10^300 - 1 ∧ (∃ m : ℕ, n = m^2)) :=
by
  sorry

end not_perfect_square_of_sum_300_l1891_189124


namespace binary_division_remainder_l1891_189134

theorem binary_division_remainder : 
  let b := 0b101101011010
  let n := 8
  b % n = 2 
:= by 
  sorry

end binary_division_remainder_l1891_189134


namespace parallelepiped_diagonal_l1891_189135

theorem parallelepiped_diagonal 
  (x y z m n p d : ℝ)
  (h1 : x^2 + y^2 = m^2)
  (h2 : x^2 + z^2 = n^2)
  (h3 : y^2 + z^2 = p^2)
  : d = Real.sqrt ((m^2 + n^2 + p^2) / 2) := 
sorry

end parallelepiped_diagonal_l1891_189135
