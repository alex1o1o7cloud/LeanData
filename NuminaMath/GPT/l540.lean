import Mathlib

namespace no_partition_square_isosceles_10deg_l540_54021

theorem no_partition_square_isosceles_10deg :
  ¬ ∃ (P : ℝ → ℝ → Prop), 
    (∀ x y, P x y → ((x = y) ∨ ((10 * x + 10 * y + 160 * (180 - x - y)) = 9 * 10))) ∧
    (∀ x y, P x 90 → P x y) ∧
    (P 90 90 → False) :=
by
  sorry

end no_partition_square_isosceles_10deg_l540_54021


namespace necessary_but_not_sufficient_l540_54024

theorem necessary_but_not_sufficient (a b x y : ℤ) (ha : 0 < a) (hb : 0 < b) (h1 : x - y > a + b) (h2 : x * y > a * b) : 
  (x > a ∧ y > b) := sorry

end necessary_but_not_sufficient_l540_54024


namespace rest_days_in_1200_days_l540_54062

noncomputable def rest_days_coinciding (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 6) else 0

theorem rest_days_in_1200_days :
  rest_days_coinciding 1200 = 200 :=
by
  sorry

end rest_days_in_1200_days_l540_54062


namespace smallest_percent_increase_is_100_l540_54046

-- The values for each question
def prize_values : List ℕ := [150, 300, 450, 900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 460800, 921600, 1843200]

-- Definition of percent increase calculation
def percent_increase (old new : ℕ) : ℕ :=
  ((new - old : ℕ) * 100) / old

-- Lean theorem statement
theorem smallest_percent_increase_is_100 :
  percent_increase (prize_values.get! 5) (prize_values.get! 6) = 100 ∧
  percent_increase (prize_values.get! 7) (prize_values.get! 8) = 100 ∧
  percent_increase (prize_values.get! 9) (prize_values.get! 10) = 100 ∧
  percent_increase (prize_values.get! 10) (prize_values.get! 11) = 100 ∧
  percent_increase (prize_values.get! 13) (prize_values.get! 14) = 100 :=
by
  sorry

end smallest_percent_increase_is_100_l540_54046


namespace ratio_first_term_l540_54065

theorem ratio_first_term (x : ℝ) (h1 : 60 / 100 = x / 25) : x = 15 := 
sorry

end ratio_first_term_l540_54065


namespace ricky_roses_l540_54078

theorem ricky_roses (initial_roses : ℕ) (stolen_roses : ℕ) (people : ℕ) (remaining_roses : ℕ)
  (h1 : initial_roses = 40)
  (h2 : stolen_roses = 4)
  (h3 : people = 9)
  (h4 : remaining_roses = initial_roses - stolen_roses) :
  remaining_roses / people = 4 :=
by sorry

end ricky_roses_l540_54078


namespace ratio_proof_l540_54015

theorem ratio_proof (a b x : ℝ) (h : a > b) (h_b_pos : b > 0)
  (h_x : x = 0.5 * Real.sqrt (a / b) + 0.5 * Real.sqrt (b / a)) :
  2 * b * Real.sqrt (x^2 - 1) / (x - Real.sqrt (x^2 - 1)) = a - b := 
sorry

end ratio_proof_l540_54015


namespace find_x_l540_54089

theorem find_x (x : ℕ) : 8000 * 6000 = x * 10^5 → x = 480 := by
  sorry

end find_x_l540_54089


namespace total_onions_grown_l540_54011

-- Given conditions
def onions_grown_by_Nancy : ℕ := 2
def onions_grown_by_Dan : ℕ := 9
def onions_grown_by_Mike : ℕ := 4
def days_worked : ℕ := 6

-- Statement we need to prove
theorem total_onions_grown : onions_grown_by_Nancy + onions_grown_by_Dan + onions_grown_by_Mike = 15 :=
by sorry

end total_onions_grown_l540_54011


namespace rectangle_x_is_18_l540_54053

-- Definitions for the conditions
def rectangle (a b x : ℕ) : Prop := 
  (a = 2 * b) ∧
  (x = 2 * (a + b)) ∧
  (x = a * b)

-- Theorem to prove the equivalence of the conditions and the answer \( x = 18 \)
theorem rectangle_x_is_18 : ∀ a b x : ℕ, rectangle a b x → x = 18 :=
by
  sorry

end rectangle_x_is_18_l540_54053


namespace randy_blocks_l540_54016

theorem randy_blocks (total_blocks house_blocks diff_blocks tower_blocks : ℕ) 
  (h_total : total_blocks = 90)
  (h_house : house_blocks = 89)
  (h_diff : house_blocks = tower_blocks + diff_blocks)
  (h_diff_value : diff_blocks = 26) :
  tower_blocks = 63 :=
by
  -- sorry is placed here to skip the proof.
  sorry

end randy_blocks_l540_54016


namespace no_nat_nums_satisfying_l540_54055

theorem no_nat_nums_satisfying (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k :=
by
  sorry

end no_nat_nums_satisfying_l540_54055


namespace basketball_tournament_l540_54094

theorem basketball_tournament (teams : Finset ℕ) (games_played : ℕ → ℕ → ℕ) (win_chance : ℕ → ℕ → Prop) 
(points : ℕ → ℕ) (X Y : ℕ) :
  teams.card = 6 → 
  (∀ t₁ t₂, t₁ ≠ t₂ → games_played t₁ t₂ = 1) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ ∨ win_chance t₂ t₁) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ → points t₁ = points t₁ + 1 ∧ points t₂ = points t₂) → 
  win_chance X Y →
  0.5 = 0.5 →
  0.5 * (1 - ((252 : ℚ) / 1024)) = (193 : ℚ) / 512 →
  ((63 : ℚ) / 256) + ((193 : ℚ) / 512) = (319 : ℚ) / 512 :=
by 
  sorry 

end basketball_tournament_l540_54094


namespace inequality_example_l540_54069

variable (a b : ℝ)

theorem inequality_example (h1 : a > 1/2) (h2 : b > 1/2) : a + 2 * b - 5 * a * b < 1/4 :=
by
  sorry

end inequality_example_l540_54069


namespace find_smallest_c_plus_d_l540_54077

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l540_54077


namespace students_prefer_mac_l540_54081

-- Define number of students in survey, and let M be the number who prefer Mac to Windows
variables (M E no_pref windows_pref : ℕ)
-- Total number of students surveyed
variable (total_students : ℕ)
-- Define that the total number of students is 210
axiom H_total : total_students = 210
-- Define that one third as many of the students who prefer Mac equally prefer both brands
axiom H_equal_preference : E = M / 3
-- Define that 90 students had no preference
axiom H_no_pref : no_pref = 90
-- Define that 40 students preferred Windows to Mac
axiom H_windows_pref : windows_pref = 40
-- Define that the total number of students is the sum of all groups
axiom H_students_sum : M + E + no_pref + windows_pref = total_students

-- The statement we need to prove
theorem students_prefer_mac :
  M = 60 :=
by sorry

end students_prefer_mac_l540_54081


namespace f_2014_value_l540_54095

def f : ℝ → ℝ :=
sorry

lemma f_periodic (x : ℝ) : f (x + 2) = f (x - 2) :=
sorry

lemma f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 4) : f x = x^2 :=
sorry

theorem f_2014_value : f 2014 = 4 :=
by
  -- Insert proof here
  sorry

end f_2014_value_l540_54095


namespace probability_left_red_off_second_blue_on_right_blue_on_l540_54036

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def total_lamps : ℕ := num_red_lamps + num_blue_lamps
def num_on : ℕ := 4
def position := Fin total_lamps
def lamp_state := {state // state < (total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)}

def valid_configuration (leftmost : position) (second_left : position) (rightmost : position) (s : lamp_state) : Prop :=
(leftmost.1 = 1 ∧ second_left.1 = 2 ∧ rightmost.1 = 8) ∧ (s.1 =  (((total_lamps - 3).choose 3) * ((total_lamps - 3).choose 2)))

theorem probability_left_red_off_second_blue_on_right_blue_on :
  ∀ (leftmost second_left rightmost : position) (s : lamp_state),
  valid_configuration leftmost second_left rightmost s ->
  ((total_lamps.choose num_red_lamps) * (total_lamps.choose num_on)) = 49 :=
sorry

end probability_left_red_off_second_blue_on_right_blue_on_l540_54036


namespace apples_in_bowl_l540_54085

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l540_54085


namespace range_of_k_l540_54035

theorem range_of_k (x y k : ℝ) 
  (h1 : 2 * x + y = k + 1) 
  (h2 : x + 2 * y = 2) 
  (h3 : x + y < 0) : 
  k < -3 :=
sorry

end range_of_k_l540_54035


namespace find_a_b_l540_54050

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 + a * x + b

theorem find_a_b 
  (a b : ℝ) 
  (h_tangent : ∀ x y, y = 2 * x - 5 → y = f 1 a b - 3) 
  : a = -1 ∧ b = -3 :=
by 
{
  sorry
}

end find_a_b_l540_54050


namespace lana_total_spending_l540_54067

theorem lana_total_spending (ticket_price : ℕ) (tickets_friends : ℕ) (tickets_extra : ℕ)
  (H1 : ticket_price = 6)
  (H2 : tickets_friends = 8)
  (H3 : tickets_extra = 2) :
  ticket_price * (tickets_friends + tickets_extra) = 60 :=
by
  sorry

end lana_total_spending_l540_54067


namespace ratio_of_adults_to_children_closest_to_one_l540_54023

theorem ratio_of_adults_to_children_closest_to_one (a c : ℕ) 
  (h₁ : 25 * a + 12 * c = 1950) 
  (h₂ : a ≥ 1) 
  (h₃ : c ≥ 1) : (a : ℚ) / (c : ℚ) = 27 / 25 := 
by 
  sorry

end ratio_of_adults_to_children_closest_to_one_l540_54023


namespace similar_triangles_l540_54068

-- Define the similarity condition between the triangles
theorem similar_triangles (x : ℝ) (h₁ : 12 / x = 9 / 6) : x = 8 := 
by sorry

end similar_triangles_l540_54068


namespace survivor_probability_l540_54037

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem survivor_probability :
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  probability = 20 / 95 :=
by
  let total_people := 20
  let tribe_size := 10
  let droppers := 3
  let total_ways := choose total_people droppers
  let tribe_ways := choose tribe_size droppers
  let same_tribe_ways := 2 * tribe_ways
  let probability := same_tribe_ways / total_ways
  have : probability = 20 / 95 := sorry
  exact this

end survivor_probability_l540_54037


namespace find_a_l540_54017

theorem find_a (a : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x y : ℝ, x^2 + a*y^2 + a^2 = 0) (h₃ : 4 = 4) :
  a = (1 - Real.sqrt 17) / 2 := sorry

end find_a_l540_54017


namespace rowing_speed_still_water_l540_54007

theorem rowing_speed_still_water (v r : ℕ) (h1 : r = 18) (h2 : 1 / (v - r) = 3 * (1 / (v + r))) : v = 36 :=
by sorry

end rowing_speed_still_water_l540_54007


namespace total_buttons_l540_54064

theorem total_buttons (green buttons: ℕ) (yellow buttons: ℕ) (blue buttons: ℕ) (total buttons: ℕ) 
(h1: green = 90) (h2: yellow = green + 10) (h3: blue = green - 5) : total = green + yellow + blue → total = 275 :=
by
  sorry

end total_buttons_l540_54064


namespace amelia_drove_distance_on_Monday_l540_54008

theorem amelia_drove_distance_on_Monday 
  (total_distance : ℕ) (tuesday_distance : ℕ) (remaining_distance : ℕ)
  (total_distance_eq : total_distance = 8205) 
  (tuesday_distance_eq : tuesday_distance = 582) 
  (remaining_distance_eq : remaining_distance = 6716) :
  ∃ x : ℕ, x + tuesday_distance + remaining_distance = total_distance ∧ x = 907 :=
by
  sorry

end amelia_drove_distance_on_Monday_l540_54008


namespace tickets_left_l540_54083

theorem tickets_left (initial_tickets used_tickets tickets_left : ℕ) 
  (h1 : initial_tickets = 127) 
  (h2 : used_tickets = 84) : 
  tickets_left = initial_tickets - used_tickets := 
by
  sorry

end tickets_left_l540_54083


namespace max_value_of_expression_l540_54072

theorem max_value_of_expression (x y z : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 :=
sorry

end max_value_of_expression_l540_54072


namespace probability_one_each_l540_54058

-- Define the counts of letters
def total_letters : ℕ := 11
def cybil_count : ℕ := 5
def ronda_count : ℕ := 5
def andy_initial_count : ℕ := 1

-- Define the probability calculation
def probability_one_from_cybil_and_one_from_ronda : ℚ :=
  (cybil_count / total_letters) * (ronda_count / (total_letters - 1)) +
  (ronda_count / total_letters) * (cybil_count / (total_letters - 1))

theorem probability_one_each (total_letters cybil_count ronda_count andy_initial_count : ℕ) :
  probability_one_from_cybil_and_one_from_ronda = 5 / 11 := sorry

end probability_one_each_l540_54058


namespace factor_quadratic_l540_54076

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l540_54076


namespace greatest_common_divisor_of_98_and_n_l540_54070

theorem greatest_common_divisor_of_98_and_n (n : ℕ) (h1 : ∃ (d : Finset ℕ),  d = {1, 7, 49} ∧ ∀ x ∈ d, x ∣ 98 ∧ x ∣ n) :
  ∃ (g : ℕ), g = 49 :=
by
  sorry

end greatest_common_divisor_of_98_and_n_l540_54070


namespace train_speed_l540_54059

theorem train_speed
  (length_of_train : ℝ) 
  (time_to_cross : ℝ) 
  (train_length_is_140 : length_of_train = 140)
  (time_is_6 : time_to_cross = 6) :
  (length_of_train / time_to_cross) = 23.33 :=
sorry

end train_speed_l540_54059


namespace basketball_free_throws_l540_54084

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = b) 
  (h3 : 2 * a + 3 * b + x = 73) : 
  x = 10 := 
by 
  sorry -- The actual proof is omitted as per the requirements.

end basketball_free_throws_l540_54084


namespace xyz_value_l540_54082

-- Define the basic conditions
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (h1 : x * y = 40 * (4:ℝ)^(1/3))
variables (h2 : x * z = 56 * (4:ℝ)^(1/3))
variables (h3 : y * z = 32 * (4:ℝ)^(1/3))
variables (h4 : x + y = 18)

-- The target theorem
theorem xyz_value : x * y * z = 16 * (895:ℝ)^(1/2) :=
by
  -- Here goes the proof, but we add 'sorry' to end the theorem placeholder
  sorry

end xyz_value_l540_54082


namespace negation_of_forall_x_squared_nonnegative_l540_54027

theorem negation_of_forall_x_squared_nonnegative :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_of_forall_x_squared_nonnegative_l540_54027


namespace hyperbola_asymptotes_l540_54012

theorem hyperbola_asymptotes (p : ℝ) (h : (p / 2, 0) ∈ {q : ℝ × ℝ | q.1 ^ 2 / 8 - q.2 ^ 2 / p = 1}) :
  (y = x) ∨ (y = -x) :=
by
  sorry

end hyperbola_asymptotes_l540_54012


namespace ratio_Mary_to_Seth_in_a_year_l540_54099

-- Given conditions
def Seth_current_age : ℝ := 3.5
def age_difference : ℝ := 9

-- Definitions derived from conditions
def Mary_current_age : ℝ := Seth_current_age + age_difference
def Seth_age_in_a_year : ℝ := Seth_current_age + 1
def Mary_age_in_a_year : ℝ := Mary_current_age + 1

-- The statement to prove
theorem ratio_Mary_to_Seth_in_a_year : (Mary_age_in_a_year / Seth_age_in_a_year) = 3 := sorry

end ratio_Mary_to_Seth_in_a_year_l540_54099


namespace expression_pos_intervals_l540_54014

theorem expression_pos_intervals :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ∨ (x > 3) ↔ (x + 1) * (x - 1) * (x - 3) > 0 := by
  sorry

end expression_pos_intervals_l540_54014


namespace equation_relationship_linear_l540_54051

theorem equation_relationship_linear 
  (x y : ℕ)
  (h1 : (x, y) = (0, 200) ∨ (x, y) = (1, 160) ∨ (x, y) = (2, 120) ∨ (x, y) = (3, 80) ∨ (x, y) = (4, 40)) :
  y = 200 - 40 * x :=
  sorry

end equation_relationship_linear_l540_54051


namespace ratio_of_lengths_l540_54060

theorem ratio_of_lengths (l1 l2 l3 : ℝ)
    (h1 : l2 = (1/2) * (l1 + l3))
    (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
    l1 / l3 = 7 / 5 := by
  sorry

end ratio_of_lengths_l540_54060


namespace range_of_k_l540_54092

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- State the theorem
theorem range_of_k (k : ℝ) : (M ∩ N k).Nonempty ↔ k ∈ Set.Ici (-1) :=
by
  sorry

end range_of_k_l540_54092


namespace trivia_game_points_l540_54090

theorem trivia_game_points (first_round_points second_round_points points_lost last_round_points : ℤ) 
    (h1 : first_round_points = 16)
    (h2 : second_round_points = 33)
    (h3 : points_lost = 48) : 
    first_round_points + second_round_points - points_lost = 1 :=
by
    rw [h1, h2, h3]
    rfl

end trivia_game_points_l540_54090


namespace sum_of_squares_l540_54044

/-- 
Given two real numbers x and y, if their product is 120 and their sum is 23, 
then the sum of their squares is 289.
-/
theorem sum_of_squares (x y : ℝ) (h₁ : x * y = 120) (h₂ : x + y = 23) :
  x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l540_54044


namespace horner_v1_value_l540_54079

-- Definitions based on the problem conditions
def f (x : ℝ) : ℝ := 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

def horner (x : ℝ) (coeffs : List ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_v1_value :
  let x := 5
  let coeffs := [4, -12, 3.5, -2.6, 1.7, -0.8]
  let v0 := coeffs.head!
  let v1 := v0 * x + coeffs.getD 1 0
  v1 = 8 := by
  -- skip the actual proof steps
  sorry

end horner_v1_value_l540_54079


namespace largest_tangential_quadrilaterals_l540_54009

-- Definitions and conditions
def convex_ngon {n : ℕ} (h : n ≥ 5) : Type := sorry -- Placeholder for defining a convex n-gon with ≥ 5 sides
def tangential_quadrilateral {n : ℕ} (h : n ≥ 5) (k : ℕ) : Prop := 
  -- Placeholder for the property that exactly k quadrilaterals out of all possible ones 
  -- in a convex n-gon have an inscribed circle
  sorry

theorem largest_tangential_quadrilaterals {n : ℕ} (h : n ≥ 5) : 
  ∃ k : ℕ, tangential_quadrilateral h k ∧ k = n / 2 :=
sorry

end largest_tangential_quadrilaterals_l540_54009


namespace taxi_ride_cost_l540_54047

noncomputable def fixed_cost : ℝ := 2.00
noncomputable def cost_per_mile : ℝ := 0.30
noncomputable def distance_traveled : ℝ := 8

theorem taxi_ride_cost :
  fixed_cost + (cost_per_mile * distance_traveled) = 4.40 := by
  sorry

end taxi_ride_cost_l540_54047


namespace other_number_l540_54061

theorem other_number (a b : ℝ) (h : a = 0.650) (h2 : a = b + 0.525) : b = 0.125 :=
sorry

end other_number_l540_54061


namespace chord_length_l540_54086

variable (x y : ℝ)

/--
The chord length cut by the line y = 2x - 2 on the circle (x-2)^2 + (y-2)^2 = 25 is 10.
-/
theorem chord_length (h₁ : y = 2 * x - 2) (h₂ : (x - 2)^2 + (y - 2)^2 = 25) : 
  ∃ length : ℝ, length = 10 :=
sorry

end chord_length_l540_54086


namespace gcd_of_three_numbers_l540_54043

theorem gcd_of_three_numbers : Nat.gcd 16434 (Nat.gcd 24651 43002) = 3 := by
  sorry

end gcd_of_three_numbers_l540_54043


namespace mixed_candy_price_l540_54004

noncomputable def price_per_pound (a b c : ℕ) (pa pb pc : ℝ) : ℝ :=
  (a * pa + b * pb + c * pc) / (a + b + c)

theorem mixed_candy_price :
  let a := 30
  let b := 15
  let c := 20
  let pa := 10.0
  let pb := 12.0
  let pc := 15.0
  price_per_pound a b c pa pb pc * 0.9 = 10.8 := by
  sorry

end mixed_candy_price_l540_54004


namespace no_positive_int_squares_l540_54031

theorem no_positive_int_squares (n : ℕ) (h_pos : 0 < n) :
  ¬ (∃ a b c : ℕ, a ^ 2 = 2 * n ^ 2 + 1 ∧ b ^ 2 = 3 * n ^ 2 + 1 ∧ c ^ 2 = 6 * n ^ 2 + 1) := by
  sorry

end no_positive_int_squares_l540_54031


namespace second_smallest_packs_hot_dogs_l540_54045

theorem second_smallest_packs_hot_dogs (n : ℕ) :
  (∃ k : ℕ, n = 5 * k + 3) →
  n > 0 →
  ∃ m : ℕ, m < n ∧ (∃ k2 : ℕ, m = 5 * k2 + 3) →
  n = 8 :=
by
  sorry

end second_smallest_packs_hot_dogs_l540_54045


namespace books_sold_on_monday_75_l540_54088

namespace Bookstore

variables (total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold : ℕ)
variable (percent_not_sold : ℝ)

def given_conditions : Prop :=
  total_books = 1200 ∧
  percent_not_sold = 0.665 ∧
  sold_Tuesday = 50 ∧
  sold_Wednesday = 64 ∧
  sold_Thursday = 78 ∧
  sold_Friday = 135 ∧
  books_not_sold = (percent_not_sold * total_books) ∧
  (total_books - books_not_sold) = (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Thursday + sold_Friday)

theorem books_sold_on_monday_75 (h : given_conditions total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold percent_not_sold) :
  sold_Monday = 75 :=
sorry

end Bookstore

end books_sold_on_monday_75_l540_54088


namespace min_sum_log_geq_four_l540_54005

theorem min_sum_log_geq_four (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (hlog : Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4) : 
  m + n ≥ 18 :=
sorry

end min_sum_log_geq_four_l540_54005


namespace reciprocal_2023_l540_54040

def reciprocal (x : ℕ) := 1 / x

theorem reciprocal_2023 : reciprocal 2023 = 1 / 2023 :=
by
  sorry

end reciprocal_2023_l540_54040


namespace cookies_with_flour_l540_54071

theorem cookies_with_flour (x: ℕ) (c1: ℕ) (c2: ℕ) (h: c1 = 18 ∧ c2 = 2 ∧ x = 9 * 5):
  x = 45 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end cookies_with_flour_l540_54071


namespace greatest_integer_gcd_four_l540_54087

theorem greatest_integer_gcd_four {n : ℕ} (h1 : n < 150) (h2 : Nat.gcd n 12 = 4) : n <= 148 :=
by {
  sorry
}

end greatest_integer_gcd_four_l540_54087


namespace Marla_colors_green_squares_l540_54049

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l540_54049


namespace polygon_interior_angles_sum_l540_54048

theorem polygon_interior_angles_sum (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := 
by sorry

end polygon_interior_angles_sum_l540_54048


namespace last_three_digits_of_8_pow_108_l540_54018

theorem last_three_digits_of_8_pow_108 :
  (8^108 % 1000) = 38 := 
sorry

end last_three_digits_of_8_pow_108_l540_54018


namespace smallest_prime_sum_l540_54038

theorem smallest_prime_sum (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (H1 : Prime (a + b + c + d))
  (H2 : Prime (a + b)) (H3 : Prime (a + c)) (H4 : Prime (a + d)) (H5 : Prime (b + c)) (H6 : Prime (b + d)) (H7 : Prime (c + d))
  (H8 : Prime (a + b + c)) (H9 : Prime (a + b + d)) (H10 : Prime (a + c + d)) (H11 : Prime (b + c + d))
  : a + b + c + d = 31 :=
sorry

end smallest_prime_sum_l540_54038


namespace sheets_in_stack_l540_54006

theorem sheets_in_stack (thickness_per_500_sheets : ℝ) (stack_height : ℝ) (total_sheets : ℕ) :
  thickness_per_500_sheets = 4 → stack_height = 10 → total_sheets = 1250 :=
by
  intros h1 h2
  -- We will provide the mathematical proof steps here.
  sorry

end sheets_in_stack_l540_54006


namespace total_houses_l540_54098

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l540_54098


namespace total_pages_in_book_l540_54063

theorem total_pages_in_book 
    (pages_read : ℕ) (pages_left : ℕ) 
    (h₁ : pages_read = 11) 
    (h₂ : pages_left = 6) : 
    pages_read + pages_left = 17 := 
by 
    sorry

end total_pages_in_book_l540_54063


namespace vector_magnitude_problem_l540_54056

open Real

noncomputable def magnitude (x : ℝ × ℝ) : ℝ := sqrt (x.1 ^ 2 + x.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h_a : a = (1, 3))
  (h_perp : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) :
  magnitude b = sqrt 10 := 
sorry

end vector_magnitude_problem_l540_54056


namespace other_number_is_286_l540_54057

theorem other_number_is_286 (a b hcf lcm : ℕ) (h_hcf : hcf = 26) (h_lcm : lcm = 2310) (h_one_num : a = 210) 
  (rel : lcm * hcf = a * b) : b = 286 :=
by
  sorry

end other_number_is_286_l540_54057


namespace base_n_representation_l540_54080

theorem base_n_representation 
  (n : ℕ) 
  (hn : n > 0)
  (a b c : ℕ) 
  (ha : 0 ≤ a ∧ a < n)
  (hb : 0 ≤ b ∧ b < n) 
  (hc : 0 ≤ c ∧ c < n) 
  (h_digits_sum : a + b + c = 24)
  (h_base_repr : 1998 = a * n^2 + b * n + c) 
  : n = 15 ∨ n = 22 ∨ n = 43 :=
sorry

end base_n_representation_l540_54080


namespace no_solution_inequalities_l540_54042

theorem no_solution_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ (x > 3 ∧ x < a)) ↔ (a ≤ 3) :=
by
  sorry

end no_solution_inequalities_l540_54042


namespace kimberly_initial_skittles_l540_54010

theorem kimberly_initial_skittles (total new initial : ℕ) (h1 : total = 12) (h2 : new = 7) (h3 : total = initial + new) : initial = 5 :=
by {
  -- Using the given conditions to form the proof
  sorry
}

end kimberly_initial_skittles_l540_54010


namespace factorial_last_nonzero_digit_non_periodic_l540_54025

def last_nonzero_digit (n : ℕ) : ℕ :=
  -- function to compute last nonzero digit of n!
  sorry

def sequence_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a n = a (n + T)

theorem factorial_last_nonzero_digit_non_periodic : ¬ ∃ T, sequence_periodic last_nonzero_digit T :=
  sorry

end factorial_last_nonzero_digit_non_periodic_l540_54025


namespace parabola_constant_c_l540_54028

theorem parabola_constant_c (b c : ℝ): 
  (∀ x : ℝ, y = x^2 + b * x + c) ∧ 
  (10 = 2^2 + b * 2 + c) ∧ 
  (31 = 4^2 + b * 4 + c) → 
  c = -3 :=
by
  sorry

end parabola_constant_c_l540_54028


namespace range_of_a_l540_54032

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

end range_of_a_l540_54032


namespace number_of_tacos_you_ordered_l540_54002

variable {E : ℝ} -- E represents the cost of one enchilada in dollars

-- Conditions
axiom h1 : ∃ t : ℕ, 0.9 * (t : ℝ) + 3 * E = 7.80
axiom h2 : 0.9 * 3 + 5 * E = 12.70

theorem number_of_tacos_you_ordered (E : ℝ) : ∃ t : ℕ, t = 2 := by
  sorry

end number_of_tacos_you_ordered_l540_54002


namespace greatest_multiple_of_3_lt_1000_l540_54020

theorem greatest_multiple_of_3_lt_1000 :
  ∃ (x : ℕ), (x % 3 = 0) ∧ (x > 0) ∧ (x^3 < 1000) ∧ ∀ (y : ℕ), (y % 3 = 0) ∧ (y > 0) ∧ (y^3 < 1000) → y ≤ x := 
sorry

end greatest_multiple_of_3_lt_1000_l540_54020


namespace total_fruits_is_174_l540_54093

def basket1_apples : ℕ := 9
def basket1_oranges : ℕ := 15
def basket1_bananas : ℕ := 14
def basket1_grapes : ℕ := 12

def basket4_apples : ℕ := basket1_apples - 2
def basket4_oranges : ℕ := basket1_oranges - 2
def basket4_bananas : ℕ := basket1_bananas - 2
def basket4_grapes : ℕ := basket1_grapes - 2

def basket5_apples : ℕ := basket1_apples + 3
def basket5_oranges : ℕ := basket1_oranges - 5
def basket5_bananas : ℕ := basket1_bananas
def basket5_grapes : ℕ := basket1_grapes

def basket6_bananas : ℕ := basket1_bananas * 2
def basket6_grapes : ℕ := basket1_grapes / 2

def total_fruits_b1_3 : ℕ := basket1_apples + basket1_oranges + basket1_bananas + basket1_grapes
def total_fruits_b4 : ℕ := basket4_apples + basket4_oranges + basket4_bananas + basket4_grapes
def total_fruits_b5 : ℕ := basket5_apples + basket5_oranges + basket5_bananas + basket5_grapes
def total_fruits_b6 : ℕ := basket6_bananas + basket6_grapes

def total_fruits_all : ℕ := total_fruits_b1_3 + total_fruits_b4 + total_fruits_b5 + total_fruits_b6

theorem total_fruits_is_174 : total_fruits_all = 174 := by
  -- proof will go here
  sorry

end total_fruits_is_174_l540_54093


namespace trajectory_of_moving_circle_l540_54066

noncomputable def ellipse_trajectory_eq (x y : ℝ) : Prop :=
  (x^2)/25 + (y^2)/9 = 1

theorem trajectory_of_moving_circle
  (x y : ℝ)
  (A : ℝ × ℝ)
  (C : ℝ × ℝ)
  (radius_C : ℝ)
  (hC : (x + 4)^2 + y^2 = 100)
  (hA : A = (4, 0))
  (radius_C_eq : radius_C = 10) :
  ellipse_trajectory_eq x y :=
sorry

end trajectory_of_moving_circle_l540_54066


namespace hotel_bill_amount_l540_54001

-- Definition of the variables used in the conditions
def each_paid : ℝ := 124.11
def friends : ℕ := 9

-- The Lean 4 theorem statement
theorem hotel_bill_amount :
  friends * each_paid = 1116.99 := sorry

end hotel_bill_amount_l540_54001


namespace minimum_x_y_sum_l540_54026

theorem minimum_x_y_sum (x y : ℕ) (hx : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 1 / 15) : x + y = 64 :=
  sorry

end minimum_x_y_sum_l540_54026


namespace circle_radius_l540_54033

theorem circle_radius 
  {XA XB XC r : ℝ}
  (h1 : XA = 3)
  (h2 : XB = 5)
  (h3 : XC = 1)
  (hx : XA * XB = XC * r)
  (hh : 2 * r = CD) :
  r = 8 :=
by
  sorry

end circle_radius_l540_54033


namespace equation_line_through_intersections_l540_54000

theorem equation_line_through_intersections (A1 B1 A2 B2 : ℝ)
  (h1 : 2 * A1 + 3 * B1 = 1)
  (h2 : 2 * A2 + 3 * B2 = 1) :
  ∃ (a b c : ℝ), a = 2 ∧ b = 3 ∧ c = -1 ∧ (a * x + b * y + c = 0) := 
sorry

end equation_line_through_intersections_l540_54000


namespace line_passes_through_fixed_point_l540_54034

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * (1) + 3 * k = 0 :=
by
  intro k
  sorry

end line_passes_through_fixed_point_l540_54034


namespace negation_of_exists_leq_zero_l540_54096

theorem negation_of_exists_leq_zero (x : ℝ) : (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) :=
by
  sorry

end negation_of_exists_leq_zero_l540_54096


namespace train_speed_on_time_l540_54013

theorem train_speed_on_time :
  ∃ (v : ℝ), 
  (∀ (d : ℝ) (t : ℝ),
    d = 133.33 ∧ 
    80 * (t + 1/3) = d ∧ 
    v * t = d) → 
  v = 100 :=
by
  sorry

end train_speed_on_time_l540_54013


namespace triangle_expression_negative_l540_54039

theorem triangle_expression_negative {a b c : ℝ} (habc : a > 0 ∧ b > 0 ∧ c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a^2 + b^2 - c^2 - 2 * a * b < 0 :=
sorry

end triangle_expression_negative_l540_54039


namespace toll_for_18_wheel_truck_l540_54029

-- Define the number of wheels and axles conditions
def total_wheels : ℕ := 18
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def number_of_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the toll calculation formula
def toll (x : ℕ) : ℝ := 1.50 + 1.50 * (x - 2)

-- Lean theorem statement asserting that the toll for the given truck is 6 dollars
theorem toll_for_18_wheel_truck : toll number_of_axles = 6 := by
  -- Skipping the actual proof using sorry
  sorry

end toll_for_18_wheel_truck_l540_54029


namespace simplify_and_evaluate_l540_54019

-- Definitions and conditions 
def x := ℝ
def given_condition (x: ℝ) : Prop := x + 2 = Real.sqrt 2

-- The problem statement translated into Lean 4
theorem simplify_and_evaluate (x: ℝ) (h: given_condition x) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3 * x)) = Real.sqrt 2 - 1 :=
sorry

end simplify_and_evaluate_l540_54019


namespace zero_points_of_f_l540_54091

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f : (f (-1/2) = 0) ∧ (f (-1) = 0) :=
by
  sorry

end zero_points_of_f_l540_54091


namespace henry_total_cost_l540_54003

def henry_initial_figures : ℕ := 3
def henry_total_needed_figures : ℕ := 15
def cost_per_figure : ℕ := 12

theorem henry_total_cost :
  (henry_total_needed_figures - henry_initial_figures) * cost_per_figure = 144 :=
by
  sorry

end henry_total_cost_l540_54003


namespace pasture_feeding_l540_54022

-- The definitions corresponding to the given conditions
def portion_per_cow_per_day := 1

def food_needed (cows : ℕ) (days : ℕ) : ℕ := cows * days

def growth_rate (food10for20 : ℕ) (food15for10 : ℕ) (days10_20 : ℕ) : ℕ :=
  (food10for20 - food15for10) / days10_20

def food_growth_rate := growth_rate (food_needed 10 20) (food_needed 15 10) 10

def new_grass_feed_cows_per_day := food_growth_rate / portion_per_cow_per_day

def original_grass := (food_needed 10 20) - (food_growth_rate * 20)

def days_to_feed_30_cows := original_grass / (30 - new_grass_feed_cows_per_day)

-- The statement we want to prove
theorem pasture_feeding :
  new_grass_feed_cows_per_day = 5 ∧ days_to_feed_30_cows = 4 := by
  sorry

end pasture_feeding_l540_54022


namespace find_incorrect_expression_l540_54041

variable {x y : ℚ}

theorem find_incorrect_expression
  (h : x / y = 5 / 6) :
  ¬ (
    (x + 3 * y) / x = 23 / 5
  ) := by
  sorry

end find_incorrect_expression_l540_54041


namespace ratio_senior_junior_l540_54074

theorem ratio_senior_junior
  (J S : ℕ)
  (h1 : ∃ k : ℕ, S = k * J)
  (h2 : (3 / 8) * S + (1 / 4) * J = (1 / 3) * (S + J)) :
  S = 2 * J :=
by
  -- The proof is to be provided
  sorry

end ratio_senior_junior_l540_54074


namespace july_birth_percentage_l540_54030

theorem july_birth_percentage (total : ℕ) (july : ℕ) (h1 : total = 150) (h2 : july = 18) : (july : ℚ) / total * 100 = 12 := sorry

end july_birth_percentage_l540_54030


namespace new_boarders_joined_l540_54097

theorem new_boarders_joined (boarders_initial day_students_initial boarders_final x : ℕ)
  (h1 : boarders_initial = 220)
  (h2 : (5:ℕ) * day_students_initial = (12:ℕ) * boarders_initial)
  (h3 : day_students_initial = 528)
  (h4 : (1:ℕ) * day_students_initial = (2:ℕ) * (boarders_initial + x)) :
  x = 44 := by
  sorry

end new_boarders_joined_l540_54097


namespace sum_of_perimeters_l540_54075

theorem sum_of_perimeters (a : ℝ) : 
    ∑' n : ℕ, (3 * a) * (1/3)^n = 9 * a / 2 :=
by sorry

end sum_of_perimeters_l540_54075


namespace sequence_sum_S15_S22_S31_l540_54073

def sequence_sum (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | m + 1 => sequence_sum m + (-1)^m * (3 * (m + 1) - 1)

theorem sequence_sum_S15_S22_S31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -57 := 
sorry

end sequence_sum_S15_S22_S31_l540_54073


namespace infinitely_many_MTRP_numbers_l540_54052

def sum_of_digits (n : ℕ) : ℕ := 
n.digits 10 |>.sum

def is_MTRP_number (m n : ℕ) : Prop :=
  n % m = 1 ∧ sum_of_digits (n^2) ≥ sum_of_digits n

theorem infinitely_many_MTRP_numbers (m : ℕ) : 
  ∀ N : ℕ, ∃ n > N, is_MTRP_number m n :=
by sorry

end infinitely_many_MTRP_numbers_l540_54052


namespace find_a_for_even_function_l540_54054

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l540_54054
