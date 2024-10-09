import Mathlib

namespace product_of_sums_of_conjugates_l1650_165097

theorem product_of_sums_of_conjugates :
  let a := 8 - Real.sqrt 500
  let b := 8 + Real.sqrt 500
  let c := 12 - Real.sqrt 72
  let d := 12 + Real.sqrt 72
  (a + b) * (c + d) = 384 :=
by
  sorry

end product_of_sums_of_conjugates_l1650_165097


namespace pipe_c_empty_time_l1650_165042

theorem pipe_c_empty_time :
  (1 / 45 + 1 / 60 - x = 1 / 40) → (1 / x = 72) :=
by
  sorry

end pipe_c_empty_time_l1650_165042


namespace binary_111_eq_7_l1650_165053

theorem binary_111_eq_7 : (1 * 2^0 + 1 * 2^1 + 1 * 2^2) = 7 :=
by
  sorry

end binary_111_eq_7_l1650_165053


namespace auditorium_rows_l1650_165091

theorem auditorium_rows (x : ℕ) (hx : (320 / x + 4) * (x + 1) = 420) : x = 20 :=
by
  sorry

end auditorium_rows_l1650_165091


namespace b_investment_correct_l1650_165050

-- Constants for shares and investments
def a_investment : ℕ := 11000
def a_share : ℕ := 2431
def b_share : ℕ := 3315
def c_investment : ℕ := 23000

-- Goal: Prove b's investment given the conditions
theorem b_investment_correct (b_investment : ℕ) (h : 2431 * b_investment = 11000 * 3315) :
  b_investment = 15000 := by
  sorry

end b_investment_correct_l1650_165050


namespace betty_oranges_l1650_165067

-- Define the givens and result as Lean definitions and theorems
theorem betty_oranges (kg_apples : ℕ) (cost_apples_per_kg cost_oranges_per_kg total_cost_oranges num_oranges : ℕ) 
    (h1 : kg_apples = 3)
    (h2 : cost_apples_per_kg = 2)
    (h3 : cost_apples_per_kg * 2 = cost_oranges_per_kg)
    (h4 : 12 = total_cost_oranges)
    (h5 : total_cost_oranges / cost_oranges_per_kg = num_oranges) :
    num_oranges = 3 :=
sorry

end betty_oranges_l1650_165067


namespace squares_in_ap_l1650_165011

theorem squares_in_ap (a b c : ℝ) (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 :=
by
  sorry

end squares_in_ap_l1650_165011


namespace determine_dimensions_l1650_165083

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end determine_dimensions_l1650_165083


namespace trigonometric_value_l1650_165033

theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - Real.pi / 4)) = 13 / 4 := 
sorry

end trigonometric_value_l1650_165033


namespace no_real_a_l1650_165020

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}

theorem no_real_a (a : ℝ) : ¬ ((A a ≠ B) ∧ (A a ∪ B = B) ∧ (∅ ⊂ (A a ∩ B))) :=
by
  intro h
  sorry

end no_real_a_l1650_165020


namespace excircle_diameter_l1650_165040

noncomputable def diameter_of_excircle (a b c S : ℝ) (s : ℝ) : ℝ :=
  2 * S / (s - a)

theorem excircle_diameter (a b c S h_A : ℝ) (s : ℝ) (h_v : 2 * ((a + b + c) / 2) = a + b + c) :
    diameter_of_excircle a b c S s = 2 * S / (s - a) :=
by
  sorry

end excircle_diameter_l1650_165040


namespace product_of_squares_l1650_165072

theorem product_of_squares (a_1 a_2 a_3 b_1 b_2 b_3 : ℕ) (N : ℕ) (h1 : (a_1 * b_1)^2 = N) (h2 : (a_2 * b_2)^2 = N) (h3 : (a_3 * b_3)^2 = N) 
: (a_1^2 * b_1^2) = 36 ∨  (a_2^2 * b_2^2) = 36 ∨ (a_3^2 * b_3^2) = 36:= 
sorry

end product_of_squares_l1650_165072


namespace income_of_A_l1650_165004

theorem income_of_A (x y : ℝ) 
    (ratio_income : 5 * x = y * 4)
    (ratio_expenditure : 3 * x = y * 2)
    (savings_A : 5 * x - 3 * y = 1600)
    (savings_B : 4 * x - 2 * y = 1600) : 
    5 * x = 4000 := 
by
  sorry

end income_of_A_l1650_165004


namespace probability_of_same_team_is_one_third_l1650_165014

noncomputable def probability_same_team : ℚ :=
  let teams := 3
  let total_combinations := teams * teams
  let successful_outcomes := teams
  successful_outcomes / total_combinations

theorem probability_of_same_team_is_one_third :
  probability_same_team = 1 / 3 := by
  sorry

end probability_of_same_team_is_one_third_l1650_165014


namespace emergency_vehicle_reachable_area_l1650_165095

theorem emergency_vehicle_reachable_area :
  let speed_roads := 60 -- velocity on roads in miles per hour
    let speed_sand := 10 -- velocity on sand in miles per hour
    let time_limit := 5 / 60 -- time limit in hours
    let max_distance_on_roads := speed_roads * time_limit -- max distance on roads
    let radius_sand_circle := (10 / 12) -- radius on the sand
    -- calculate area covered
  (5 * 5 + 4 * (1 / 4 * Real.pi * (radius_sand_circle)^2)) = (25 + (25 * Real.pi) / 36) :=
by
  sorry

end emergency_vehicle_reachable_area_l1650_165095


namespace cos_double_angle_value_l1650_165094

theorem cos_double_angle_value (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_value_l1650_165094


namespace extended_fishing_rod_length_l1650_165064

def original_length : ℝ := 48
def increase_factor : ℝ := 1.33
def extended_length (orig_len : ℝ) (factor : ℝ) : ℝ := orig_len * factor

theorem extended_fishing_rod_length : extended_length original_length increase_factor = 63.84 :=
  by
    -- proof goes here
    sorry

end extended_fishing_rod_length_l1650_165064


namespace movie_sale_price_l1650_165012

/-- 
Given the conditions:
- cost of actors: $1200
- number of people: 50
- cost of food per person: $3
- equipment rental costs twice as much as food and actors combined
- profit made: $5950

Prove that the selling price of the movie was $10,000.
-/
theorem movie_sale_price :
  let cost_of_actors := 1200
  let num_people := 50
  let food_cost_per_person := 3
  let total_food_cost := num_people * food_cost_per_person
  let combined_cost := total_food_cost + cost_of_actors
  let equipment_rental_cost := 2 * combined_cost
  let total_cost := cost_of_actors + total_food_cost + equipment_rental_cost
  let profit := 5950
  let sale_price := total_cost + profit
  sale_price = 10000 := 
by
  sorry

end movie_sale_price_l1650_165012


namespace number_of_men_first_group_l1650_165002

theorem number_of_men_first_group :
  (∃ M : ℕ, 30 * 3 * (M : ℚ) * (84 / 30) / 3 = 112 / 6) → ∃ M : ℕ, M = 20 := 
by
  sorry

end number_of_men_first_group_l1650_165002


namespace cow_manure_growth_percentage_l1650_165092

variable (control_height bone_meal_growth_percentage cow_manure_height : ℝ)
variable (bone_meal_height : ℝ := bone_meal_growth_percentage * control_height)
variable (percentage_growth : ℝ := (cow_manure_height / bone_meal_height) * 100)

theorem cow_manure_growth_percentage 
  (h₁ : control_height = 36)
  (h₂ : bone_meal_growth_percentage = 1.25)
  (h₃ : cow_manure_height = 90) :
  percentage_growth = 200 :=
by {
  sorry
}

end cow_manure_growth_percentage_l1650_165092


namespace attended_college_percentage_l1650_165062

variable (total_boys : ℕ) (total_girls : ℕ) (percent_not_attend_boys : ℕ) (percent_not_attend_girls : ℕ)

def total_boys_attended_college (total_boys percent_not_attend_boys : ℕ) : ℕ :=
  total_boys - percent_not_attend_boys * total_boys / 100

def total_girls_attended_college (total_girls percent_not_attend_girls : ℕ) : ℕ :=
  total_girls - percent_not_attend_girls * total_girls / 100

noncomputable def total_student_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_boys_attended_college total_boys percent_not_attend_boys +
  total_girls_attended_college total_girls percent_not_attend_girls

noncomputable def percent_class_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_student_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls * 100 /
  (total_boys + total_girls)

theorem attended_college_percentage :
  total_boys = 300 → total_girls = 240 → percent_not_attend_boys = 30 → percent_not_attend_girls = 30 →
  percent_class_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls = 70 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end attended_college_percentage_l1650_165062


namespace radar_coverage_proof_l1650_165017

theorem radar_coverage_proof (n : ℕ) (r : ℝ) (w : ℝ) (d : ℝ) (A : ℝ) : 
  n = 9 ∧ r = 37 ∧ w = 24 ∧ d = 35 / Real.sin (Real.pi / 9) ∧
  A = 1680 * Real.pi / Real.tan (Real.pi / 9) → 
  ∃ OB S_ring, OB = d ∧ S_ring = A 
:= by sorry

end radar_coverage_proof_l1650_165017


namespace positive_value_of_X_l1650_165096

-- Definition for the problem's conditions
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Statement of the proof problem
theorem positive_value_of_X (X : ℝ) (h : hash X 7 = 170) : X = 11 :=
by
  sorry

end positive_value_of_X_l1650_165096


namespace rearrange_possible_l1650_165085

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l1650_165085


namespace range_of_real_number_m_l1650_165036

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m_l1650_165036


namespace students_in_photo_l1650_165025

theorem students_in_photo (m n : ℕ) (h1 : n = m + 5) (h2 : n = m + 5 ∧ m = 3) : 
  m * n = 24 :=
by
  -- h1: n = m + 5    (new row is 4 students fewer)
  -- h2: m = 3        (all rows have the same number of students after rearrangement)
  -- Prove m * n = 24
  sorry

end students_in_photo_l1650_165025


namespace factorable_polynomial_l1650_165038

theorem factorable_polynomial (d f e g b : ℤ) (h1 : d * f = 28) (h2 : e * g = 14)
  (h3 : d * g + e * f = b) : b = 42 :=
by sorry

end factorable_polynomial_l1650_165038


namespace remainder_div_197_l1650_165061

theorem remainder_div_197 (x q : ℕ) (h_pos : 0 < x) (h_div : 100 = q * x + 3) : 197 % x = 3 :=
sorry

end remainder_div_197_l1650_165061


namespace range_of_a_if_f_has_three_zeros_l1650_165084

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l1650_165084


namespace josh_money_remaining_l1650_165027

theorem josh_money_remaining :
  let initial := 50.00
  let shirt := 7.85
  let meal := 15.49
  let magazine := 6.13
  let friends_debt := 3.27
  let cd := 11.75
  initial - shirt - meal - magazine - friends_debt - cd = 5.51 :=
by
  sorry

end josh_money_remaining_l1650_165027


namespace least_small_barrels_l1650_165065

theorem least_small_barrels (total_oil : ℕ) (large_barrel : ℕ) (small_barrel : ℕ) (L S : ℕ)
  (h1 : total_oil = 745) (h2 : large_barrel = 11) (h3 : small_barrel = 7)
  (h4 : 11 * L + 7 * S = 745) (h5 : total_oil - 11 * L = 7 * S) : S = 1 :=
by
  sorry

end least_small_barrels_l1650_165065


namespace distinct_distances_l1650_165046

theorem distinct_distances (points : Finset (ℝ × ℝ)) (h : points.card = 2016) :
  ∃ s : Finset ℝ, s.card ≥ 45 ∧ ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ 
    (s = (points.image (λ r => dist p r)).filter (λ x => x ≠ 0)) :=
by
  sorry

end distinct_distances_l1650_165046


namespace coefficient_sum_eq_512_l1650_165081

theorem coefficient_sum_eq_512 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) :
  (1 - x) ^ 9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 512 :=
sorry

end coefficient_sum_eq_512_l1650_165081


namespace verify_min_n_for_coprime_subset_l1650_165080

def is_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s), a ≠ b → Nat.gcd a b = 1

def contains_4_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∃ t : Finset ℕ, t ⊆ s ∧ t.card = 4 ∧ is_pairwise_coprime t

def min_n_for_coprime_subset : ℕ :=
  111

theorem verify_min_n_for_coprime_subset (S : Finset ℕ) (hS : S = Finset.range 151) :
  ∀ (n : ℕ), (∀ s : Finset ℕ, s ⊆ S ∧ s.card = n → contains_4_pairwise_coprime s) ↔ (n ≥ min_n_for_coprime_subset) :=
sorry

end verify_min_n_for_coprime_subset_l1650_165080


namespace jelly_bean_probabilities_l1650_165063

theorem jelly_bean_probabilities :
  let p_red := 0.15
  let p_orange := 0.35
  let p_yellow := 0.2
  let p_green := 0.3
  p_red + p_orange + p_yellow + p_green = 1 :=
by
  sorry

end jelly_bean_probabilities_l1650_165063


namespace white_pairs_coincide_l1650_165003

theorem white_pairs_coincide 
    (red_triangles : ℕ)
    (blue_triangles : ℕ)
    (white_triangles : ℕ)
    (red_pairs : ℕ)
    (blue_pairs : ℕ)
    (red_white_pairs : ℕ)
    (coinciding_white_pairs : ℕ) :
    red_triangles = 4 → 
    blue_triangles = 6 →
    white_triangles = 10 →
    red_pairs = 3 →
    blue_pairs = 4 →
    red_white_pairs = 3 →
    coinciding_white_pairs = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end white_pairs_coincide_l1650_165003


namespace length_of_train_l1650_165013

theorem length_of_train (v : ℝ) (t : ℝ) (L : ℝ) 
  (h₁ : v = 36) 
  (h₂ : t = 1) 
  (h_eq_lengths : true) -- assuming the equality of lengths tacitly without naming
  : L = 300 := 
by 
  -- proof steps would go here
  sorry

end length_of_train_l1650_165013


namespace tail_count_likelihood_draw_and_rainy_l1650_165024

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draws_when_heads : ℕ := 7
def rainy_when_tails : ℕ := 4

theorem tail_count :
  coin_tosses - heads_count = 14 :=
sorry

theorem likelihood_draw_and_rainy :
  0 = 0 :=
sorry

end tail_count_likelihood_draw_and_rainy_l1650_165024


namespace train_speed_84_kmph_l1650_165026

theorem train_speed_84_kmph (length : ℕ) (time : ℕ) (conversion_factor : ℚ)
  (h_length : length = 140) (h_time : time = 6) (h_conversion_factor : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 84 :=
  sorry

end train_speed_84_kmph_l1650_165026


namespace a_9_equals_18_l1650_165059

def is_sequence_of_positive_integers (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 0 < a n

def satisfies_recursive_relation (a : ℕ → ℕ) : Prop :=
∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem a_9_equals_18 (a : ℕ → ℕ)
  (H1 : is_sequence_of_positive_integers a)
  (H2 : satisfies_recursive_relation a)
  (H3 : a 2 = 4) : a 9 = 18 :=
sorry

end a_9_equals_18_l1650_165059


namespace salary_for_May_l1650_165032

variable (J F M A May : ℕ)

axiom condition1 : (J + F + M + A) / 4 = 8000
axiom condition2 : (F + M + A + May) / 4 = 8800
axiom condition3 : J = 3300

theorem salary_for_May : May = 6500 :=
by sorry

end salary_for_May_l1650_165032


namespace problem1_problem2_problem3_l1650_165023

variables (x y a b c : ℚ)

-- Definition of the operation *
def op_star (x y : ℚ) : ℚ := x * y + 1

-- Prove that 2 * 3 = 7 using the operation *
theorem problem1 : op_star 2 3 = 7 :=
by
  sorry

-- Prove that (1 * 4) * (-1/2) = -3/2 using the operation *
theorem problem2 : op_star (op_star 1 4) (-1/2) = -3/2 :=
by
  sorry

-- Prove the relationship a * (b + c) + 1 = a * b + a * c using the operation *
theorem problem3 : op_star a (b + c) + 1 = op_star a b + op_star a c :=
by
  sorry

end problem1_problem2_problem3_l1650_165023


namespace pen_cost_l1650_165052

theorem pen_cost (x : ℝ) (h1 : 5 * x + x = 24) : x = 4 :=
by
  sorry

end pen_cost_l1650_165052


namespace find_m_n_l1650_165075

theorem find_m_n : ∃ (m n : ℕ), m^m + (m * n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end find_m_n_l1650_165075


namespace production_average_l1650_165071

-- Define the conditions and question
theorem production_average (n : ℕ) (P : ℕ) (P_new : ℕ) (h1 : P = n * 70) (h2 : P_new = P + 90) (h3 : P_new = (n + 1) * 75) : n = 3 := 
by sorry

end production_average_l1650_165071


namespace arithmetic_sequence_50th_term_l1650_165001

-- Define the arithmetic sequence parameters
def first_term : Int := 2
def common_difference : Int := 5

-- Define the formula to calculate the n-th term of the sequence
def nth_term (n : Nat) : Int :=
  first_term + (n - 1) * common_difference

-- Prove that the 50th term of the sequence is 247
theorem arithmetic_sequence_50th_term : nth_term 50 = 247 :=
  by
  -- Proof goes here
  sorry

end arithmetic_sequence_50th_term_l1650_165001


namespace B_starts_6_hours_after_A_l1650_165043

theorem B_starts_6_hours_after_A 
    (A_walk_speed : ℝ) (B_cycle_speed : ℝ) (catch_up_distance : ℝ)
    (hA : A_walk_speed = 10) (hB : B_cycle_speed = 20) (hD : catch_up_distance = 120) :
    ∃ t : ℝ, t = 6 :=
by
  sorry

end B_starts_6_hours_after_A_l1650_165043


namespace seq_eventually_reaches_one_l1650_165008

theorem seq_eventually_reaches_one (a : ℕ → ℤ) (h₁ : a 1 > 0) :
  (∀ n, n % 4 = 0 → a (n + 1) = a n / 2) →
  (∀ n, n % 4 = 1 → a (n + 1) = 3 * a n + 1) →
  (∀ n, n % 4 = 2 → a (n + 1) = 2 * a n - 1) →
  (∀ n, n % 4 = 3 → a (n + 1) = (a n + 1) / 4) →
  ∃ m, a m = 1 :=
by
  sorry

end seq_eventually_reaches_one_l1650_165008


namespace C_alone_work_days_l1650_165093

theorem C_alone_work_days (A_work_days B_work_days combined_work_days : ℝ) 
  (A_work_rate B_work_rate C_work_rate combined_work_rate : ℝ)
  (hA : A_work_days = 6)
  (hB : B_work_days = 5)
  (hCombined : combined_work_days = 2)
  (hA_work_rate : A_work_rate = 1 / A_work_days)
  (hB_work_rate : B_work_rate = 1 / B_work_days)
  (hCombined_work_rate : combined_work_rate = 1 / combined_work_days)
  (work_rate_eq : A_work_rate + B_work_rate + C_work_rate = combined_work_rate):
  (1 / C_work_rate) = 7.5 :=
by
  sorry

end C_alone_work_days_l1650_165093


namespace partition_2004_ways_l1650_165058

theorem partition_2004_ways : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2004 → 
  ∃! (q r : ℕ), 2004 = q * n + r ∧ 0 ≤ r ∧ r < n :=
by
  sorry

end partition_2004_ways_l1650_165058


namespace problem_1_problem_2_problem_3_l1650_165044

theorem problem_1 (x y : ℝ) : x^2 + y^2 + x * y + x + y ≥ -1 / 3 := 
by sorry

theorem problem_2 (x y z : ℝ) : x^2 + y^2 + z^2 + x * y + y * z + z * x + x + y + z ≥ -3 / 8 := 
by sorry

theorem problem_3 (x y z r : ℝ) : x^2 + y^2 + z^2 + r^2 + x * y + x * z + x * r + y * z + y * r + z * r + x + y + z + r ≥ -2 / 5 := 
by sorry

end problem_1_problem_2_problem_3_l1650_165044


namespace total_filled_water_balloons_l1650_165082

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end total_filled_water_balloons_l1650_165082


namespace sum_of_reciprocals_of_factors_of_12_l1650_165039

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l1650_165039


namespace minimize_expr_l1650_165006

theorem minimize_expr : ∃ c : ℝ, (∀ d : ℝ, (3/4 * c^2 - 9 * c + 5) ≤ (3/4 * d^2 - 9 * d + 5)) ∧ c = 6 :=
by
  use 6
  sorry

end minimize_expr_l1650_165006


namespace seats_required_l1650_165030

def children := 58
def per_seat := 2
def seats_needed (children : ℕ) (per_seat : ℕ) := children / per_seat

theorem seats_required : seats_needed children per_seat = 29 := 
by
  sorry

end seats_required_l1650_165030


namespace meal_cost_l1650_165089

theorem meal_cost:
  ∀ (s c p k : ℝ), 
  (2 * s + 5 * c + 2 * p + 3 * k = 6.30) →
  (3 * s + 8 * c + 2 * p + 4 * k = 8.40) →
  (s + c + p + k = 3.15) :=
by
  intros s c p k h1 h2
  sorry

end meal_cost_l1650_165089


namespace complement_of_A_l1650_165048

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (C_UA : Set ℕ) :
  U = {2, 3, 4} →
  A = {x | (x - 1) * (x - 4) < 0 ∧ x ∈ Set.univ} →
  C_UA = {x ∈ U | x ∉ A} →
  C_UA = {4} :=
by
  intros hU hA hCUA
  -- proof omitted, sorry placeholder
  sorry

end complement_of_A_l1650_165048


namespace quadratic_eq_real_roots_roots_diff_l1650_165068

theorem quadratic_eq_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + (m-2)*x - m = 0) ∧
  (y^2 + (m-2)*y - m = 0) := sorry

theorem roots_diff (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0)
  (h_roots : (m^2 + (m-2)*m - m = 0) ∧ (n^2 + (m-2)*n - m = 0)) :
  m - n = 5/2 := sorry

end quadratic_eq_real_roots_roots_diff_l1650_165068


namespace probability_of_chosen_primes_l1650_165009

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def total_ways : ℕ := Nat.choose 30 2
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def primes_not_divisible_by_5 : List ℕ := [2, 3, 7, 11, 13, 17, 19, 23, 29]

def chosen_primes (s : Finset ℕ) : Prop :=
  s.card = 2 ∧
  (∀ n ∈ s, n ∈ primes_not_divisible_by_5)  ∧
  (∀ n ∈ s, n ≠ 5) -- (5 is already excluded in the prime list, but for completeness)

def favorable_ways : ℕ := Nat.choose 9 2  -- 9 primes not divisible by 5

def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_chosen_primes:
  probability = (12 / 145 : ℚ) :=
by
  sorry

end probability_of_chosen_primes_l1650_165009


namespace boys_other_communities_l1650_165007

/-- 
In a school of 850 boys, 44% are Muslims, 28% are Hindus, 
10% are Sikhs, and the remaining belong to other communities.
Prove that the number of boys belonging to other communities is 153.
-/
theorem boys_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℚ)
  (h_total_boys : total_boys = 850)
  (h_percentage_muslims : percentage_muslims = 44)
  (h_percentage_hindus : percentage_hindus = 28)
  (h_percentage_sikhs : percentage_sikhs = 10) :
  let percentage_others := 100 - (percentage_muslims + percentage_hindus + percentage_sikhs)
  let number_others := (percentage_others / 100) * total_boys
  number_others = 153 := 
by
  sorry

end boys_other_communities_l1650_165007


namespace isosceles_triangle_perimeter_eq_70_l1650_165005

-- Define the conditions
def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 60
def isosceles_triangle_base : ℕ := 30

-- Calculate the side of equilateral triangle
def equilateral_triangle_side : ℕ := equilateral_triangle_perimeter / 3

-- Lean 4 statement
theorem isosceles_triangle_perimeter_eq_70 :
  ∃ (a b c : ℕ), is_equilateral_triangle a b c ∧ 
  a + b + c = equilateral_triangle_perimeter →
  (is_isosceles_triangle a a isosceles_triangle_base) →
  a + a + isosceles_triangle_base = 70 :=
by
  sorry -- proof is omitted

end isosceles_triangle_perimeter_eq_70_l1650_165005


namespace isosceles_triangle_perimeter_l1650_165015

theorem isosceles_triangle_perimeter 
  (m : ℝ) 
  (h : 2 * m + 1 = 8) : 
  (m - 2) + 2 * 8 = 17.5 := 
by 
  sorry

end isosceles_triangle_perimeter_l1650_165015


namespace bells_toll_together_l1650_165099

noncomputable def LCM (a b : Nat) : Nat := (a * b) / (Nat.gcd a b)

theorem bells_toll_together :
  let intervals := [2, 4, 6, 8, 10, 12]
  let lcm := intervals.foldl LCM 1
  lcm = 120 →
  let duration := 30 * 60 -- 1800 seconds
  let tolls := duration / lcm
  tolls + 1 = 16 :=
by
  sorry

end bells_toll_together_l1650_165099


namespace highest_number_on_dice_l1650_165018

theorem highest_number_on_dice (n : ℕ) (h1 : 0 < n)
  (h2 : ∃ p : ℝ, p = 0.1111111111111111) 
  (h3 : 1 / 9 = 4 / (n * n)) 
  : n = 6 :=
sorry

end highest_number_on_dice_l1650_165018


namespace avg_rate_of_change_l1650_165049

def f (x : ℝ) := 2 * x + 1

theorem avg_rate_of_change : (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end avg_rate_of_change_l1650_165049


namespace certain_event_at_least_one_genuine_l1650_165054

def products : Finset (Fin 12) := sorry
def genuine : Finset (Fin 12) := sorry
def defective : Finset (Fin 12) := sorry
noncomputable def draw3 : Finset (Finset (Fin 12)) := sorry

-- Condition: 12 identical products, 10 genuine, 2 defective
axiom products_condition_1 : products.card = 12
axiom products_condition_2 : genuine.card = 10
axiom products_condition_3 : defective.card = 2
axiom products_condition_4 : ∀ x ∈ genuine, x ∈ products
axiom products_condition_5 : ∀ x ∈ defective, x ∈ products
axiom products_condition_6 : genuine ∩ defective = ∅

-- The statement to be proved: when drawing 3 products randomly, it is certain that at least 1 is genuine.
theorem certain_event_at_least_one_genuine :
  ∀ s ∈ draw3, ∃ x ∈ s, x ∈ genuine :=
sorry

end certain_event_at_least_one_genuine_l1650_165054


namespace train_crosses_pole_in_1_5_seconds_l1650_165028

noncomputable def time_to_cross_pole (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  length / speed_m_s

theorem train_crosses_pole_in_1_5_seconds :
  time_to_cross_pole 60 144 = 1.5 :=
by
  unfold time_to_cross_pole
  -- simplified proof would be here
  sorry

end train_crosses_pole_in_1_5_seconds_l1650_165028


namespace length_of_train_l1650_165045

theorem length_of_train (speed_kmph : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) 
  (h1 : speed_kmph = 45) (h2 : bridge_length_m = 220) (h3 : crossing_time_s = 30) :
  ∃ train_length_m : ℕ, train_length_m = 155 :=
by
  sorry

end length_of_train_l1650_165045


namespace problem1_problem2_l1650_165035

def count_good_subsets (n : ℕ) : ℕ := 
if n % 2 = 1 then 2^(n - 1) 
else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2)

def sum_f_good_subsets (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2)

theorem problem1 (n : ℕ)  :
  (count_good_subsets n = (if n % 2 = 1 then 2^(n - 1) else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2))) :=
sorry

theorem problem2 (n : ℕ) :
  (sum_f_good_subsets n = (if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
  else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2))) := 
sorry

end problem1_problem2_l1650_165035


namespace geom_seq_sum_relation_l1650_165019

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_sum_relation (h_geom : is_geometric_sequence a q)
  (h_pos : ∀ n, a n > 0) (h_q_ne_one : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 :=
by
  sorry

end geom_seq_sum_relation_l1650_165019


namespace polyhedron_euler_formula_l1650_165087

variable (A F S : ℕ)
variable (closed_polyhedron : Prop)

theorem polyhedron_euler_formula (h : closed_polyhedron) : A + 2 = F + S := sorry

end polyhedron_euler_formula_l1650_165087


namespace greatest_common_factor_of_two_digit_palindromes_is_11_l1650_165021

-- Define a two-digit palindrome
def is_two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

-- Define the GCD of the set of all such numbers
def GCF_two_digit_palindromes : ℕ :=
  gcd (11 * 1) (gcd (11 * 2) (gcd (11 * 3) (gcd (11 * 4)
  (gcd (11 * 5) (gcd (11 * 6) (gcd (11 * 7) (gcd (11 * 8) (11 * 9))))))))

-- The statement to prove
theorem greatest_common_factor_of_two_digit_palindromes_is_11 :
  GCF_two_digit_palindromes = 11 :=
by
  sorry

end greatest_common_factor_of_two_digit_palindromes_is_11_l1650_165021


namespace find_rabbits_l1650_165055

theorem find_rabbits (heads rabbits chickens : ℕ) (h1 : rabbits + chickens = 40) (h2 : 4 * rabbits = 10 * 2 * chickens - 8) : rabbits = 33 :=
by
  -- We skip the proof here
  sorry

end find_rabbits_l1650_165055


namespace total_bill_is_correct_l1650_165034

-- Define conditions as constant values
def cost_per_scoop : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

-- Define the total bill calculation
def total_bill := (pierre_scoops * cost_per_scoop) + (mom_scoops * cost_per_scoop)

-- State the theorem that the total bill equals 14
theorem total_bill_is_correct : total_bill = 14 := by
  sorry

end total_bill_is_correct_l1650_165034


namespace calculate_expression_l1650_165086

theorem calculate_expression (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) =
    6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) :=
by {
  sorry
}

end calculate_expression_l1650_165086


namespace differences_impossible_l1650_165041

def sum_of_digits (n : ℕ) : ℕ :=
  -- A simple definition for the sum of digits function
  n.digits 10 |>.sum

theorem differences_impossible (a : Fin 100 → ℕ) :
    ¬∃ (perm : Fin 100 → Fin 100), 
      (∀ i, a i - sum_of_digits (a (perm (i : ℕ) % 100)) = i + 1) :=
by
  sorry

end differences_impossible_l1650_165041


namespace find_original_height_l1650_165056

noncomputable def original_height : ℝ := by
  let H := 102.19
  sorry

lemma ball_rebound (H : ℝ) : 
  (H + 2 * 0.8 * H + 2 * 0.56 * H + 2 * 0.336 * H + 2 * 0.168 * H + 2 * 0.0672 * H + 2 * 0.02016 * H = 500) :=
by
  sorry

theorem find_original_height : original_height = 102.19 :=
by
  have h := ball_rebound original_height
  sorry

end find_original_height_l1650_165056


namespace parabola_vertex_y_coord_l1650_165000

theorem parabola_vertex_y_coord (a b c x y : ℝ) (h : a = 2 ∧ b = 16 ∧ c = 35 ∧ y = a*x^2 + b*x + c ∧ x = -b / (2 * a)) : y = 3 :=
by
  sorry

end parabola_vertex_y_coord_l1650_165000


namespace bobby_weekly_salary_l1650_165047

variable (S : ℝ)
variables (federal_tax : ℝ) (state_tax : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) (city_fee : ℝ) (net_paycheck : ℝ)

def bobby_salary_equation := 
  S - (federal_tax * S) - (state_tax * S) - health_insurance - life_insurance - city_fee = net_paycheck

theorem bobby_weekly_salary 
  (S : ℝ) 
  (federal_tax : ℝ := 1/3) 
  (state_tax : ℝ := 0.08) 
  (health_insurance : ℝ := 50) 
  (life_insurance : ℝ := 20) 
  (city_fee : ℝ := 10) 
  (net_paycheck : ℝ := 184) 
  (valid_solution : bobby_salary_equation S (1/3) 0.08 50 20 10 184) : 
  S = 450.03 := 
  sorry

end bobby_weekly_salary_l1650_165047


namespace Katie_homework_problems_l1650_165070

theorem Katie_homework_problems :
  let finished_problems := 5
  let remaining_problems := 4
  let total_problems := finished_problems + remaining_problems
  total_problems = 9 :=
by
  sorry

end Katie_homework_problems_l1650_165070


namespace max_rectangles_1x2_l1650_165051

-- Define the problem conditions
def single_cell_squares : Type := sorry
def rectangles_1x2 (figure : single_cell_squares) : Prop := sorry

-- State the maximum number theorem
theorem max_rectangles_1x2 (figure : single_cell_squares) (h : rectangles_1x2 figure) :
  ∃ (n : ℕ), n ≤ 5 ∧ ∀ m : ℕ, rectangles_1x2 figure ∧ m ≤ 5 → m = 5 :=
sorry

end max_rectangles_1x2_l1650_165051


namespace percentage_of_500_l1650_165066

theorem percentage_of_500 : (110 * 500) / 100 = 550 :=
by
  sorry

end percentage_of_500_l1650_165066


namespace no_perfect_squares_exist_l1650_165010

theorem no_perfect_squares_exist (x y : ℕ) :
  ¬(∃ k1 k2 : ℕ, x^2 + y = k1^2 ∧ y^2 + x = k2^2) :=
sorry

end no_perfect_squares_exist_l1650_165010


namespace solve_system_l1650_165031

theorem solve_system :
  {p : ℝ × ℝ | p.1^3 + p.2^3 = 19 ∧ p.1^2 + p.2^2 + 5 * p.1 + 5 * p.2 + p.1 * p.2 = 12} = {(3, -2), (-2, 3)} :=
sorry

end solve_system_l1650_165031


namespace union_A_B_l1650_165088

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem union_A_B :
  A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

end union_A_B_l1650_165088


namespace employed_population_is_60_percent_l1650_165098

def percent_employed (P : ℝ) (E : ℝ) : Prop :=
  ∃ (P_0 : ℝ) (E_male : ℝ) (E_female : ℝ),
    P_0 = P * 0.45 ∧    -- 45 percent of the population are employed males
    E_female = (E * 0.25) * P ∧   -- 25 percent of the employed people are females
    (0.75 * E = 0.45) ∧    -- 75 percent of the employed people are males which equals to 45% of the total population
    E = 0.6            -- 60% of the population are employed

theorem employed_population_is_60_percent (P : ℝ) (E : ℝ):
  percent_employed P E :=
by
  sorry

end employed_population_is_60_percent_l1650_165098


namespace number_of_green_balls_l1650_165037

theorem number_of_green_balls (b g : ℕ) (h1 : b = 9) (h2 : (b : ℚ) / (b + g) = 3 / 10) : g = 21 :=
sorry

end number_of_green_balls_l1650_165037


namespace range_of_a_l1650_165090

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) ↔ 0 ≤ a :=
sorry

end range_of_a_l1650_165090


namespace average_growth_rate_bing_dwen_dwen_l1650_165078

noncomputable def sales_growth_rate (v0 v2 : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ 2 = (v2 : ℝ) / (v0 : ℝ)

theorem average_growth_rate_bing_dwen_dwen :
  ∀ (v0 v2 : ℕ) (x : ℝ),
    v0 = 10000 →
    v2 = 12100 →
    sales_growth_rate v0 v2 x →
    x = 0.1 :=
by
  intros v0 v2 x h₀ h₂ h_growth
  sorry

end average_growth_rate_bing_dwen_dwen_l1650_165078


namespace birthday_cars_equal_12_l1650_165060

namespace ToyCars

def initial_cars : Nat := 14
def bought_cars : Nat := 28
def sister_gave : Nat := 8
def friend_gave : Nat := 3
def remaining_cars : Nat := 43

def total_initial_cars := initial_cars + bought_cars
def total_given_away := sister_gave + friend_gave

theorem birthday_cars_equal_12 (B : Nat) (h : total_initial_cars + B - total_given_away = remaining_cars) : B = 12 :=
sorry

end ToyCars

end birthday_cars_equal_12_l1650_165060


namespace new_students_count_l1650_165076

theorem new_students_count (x : ℕ) (avg_age_group new_avg_age avg_new_students : ℕ)
  (h1 : avg_age_group = 14) (h2 : new_avg_age = 15) (h3 : avg_new_students = 17)
  (initial_students : ℕ) (initial_avg_age : ℕ)
  (h4 : initial_students = 10) (h5 : initial_avg_age = initial_students * avg_age_group)
  (h6 : new_avg_age * (initial_students + x) = initial_avg_age + (x * avg_new_students)) :
  x = 5 := 
by
  sorry

end new_students_count_l1650_165076


namespace min_value_of_box_l1650_165073

theorem min_value_of_box 
  (a b : ℤ) 
  (h_distinct : a ≠ b) 
  (h_eq : (a * x + b) * (b * x + a) = 34 * x^2 + Box * x + 34) 
  (h_prod : a * b = 34) :
  ∃ (Box : ℤ), Box = 293 :=
by
  sorry

end min_value_of_box_l1650_165073


namespace log_expression_equals_eight_l1650_165074

theorem log_expression_equals_eight :
  (Real.log 4 / Real.log 10) + 
  2 * (Real.log 5 / Real.log 10) + 
  3 * (Real.log 2 / Real.log 10) + 
  6 * (Real.log 5 / Real.log 10) + 
  (Real.log 8 / Real.log 10) = 8 := 
by 
  sorry

end log_expression_equals_eight_l1650_165074


namespace midpoint_product_l1650_165057

theorem midpoint_product (x y : ℝ) (h1 : (4 : ℝ) = (x + 10) / 2) (h2 : (-2 : ℝ) = (-6 + y) / 2) : x * y = -4 := by
  sorry

end midpoint_product_l1650_165057


namespace prime_divisor_form_l1650_165029

theorem prime_divisor_form {p q : ℕ} (hp : Nat.Prime p) (hpgt2 : p > 2) (hq : Nat.Prime q) (hq_dvd : q ∣ 2^p - 1) : 
  ∃ k : ℕ, q = 2 * k * p + 1 := 
sorry

end prime_divisor_form_l1650_165029


namespace sum_a_b_l1650_165022

theorem sum_a_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 2) (h_bound : a^b < 500)
  (h_max : ∀ a' b', a' > 0 → b' > 2 → a'^b' < 500 → a'^b' ≤ a^b) :
  a + b = 8 :=
by sorry

end sum_a_b_l1650_165022


namespace complex_number_powers_l1650_165069

theorem complex_number_powers (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 :=
sorry

end complex_number_powers_l1650_165069


namespace sin_pi_six_minus_two_alpha_l1650_165016

theorem sin_pi_six_minus_two_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = - 7 / 9 :=
by
  sorry

end sin_pi_six_minus_two_alpha_l1650_165016


namespace ceilings_left_correct_l1650_165079

def total_ceilings : ℕ := 28
def ceilings_painted_this_week : ℕ := 12
def ceilings_painted_next_week : ℕ := ceilings_painted_this_week / 4
def ceilings_left_to_paint : ℕ := total_ceilings - (ceilings_painted_this_week + ceilings_painted_next_week)

theorem ceilings_left_correct : ceilings_left_to_paint = 13 := by
  sorry

end ceilings_left_correct_l1650_165079


namespace five_circles_intersect_l1650_165077

-- Assume we have five circles
variables (circle1 circle2 circle3 circle4 circle5 : Set Point)

-- Assume every four of them intersect at a single point
axiom four_intersect (c1 c2 c3 c4 : Set Point) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 ∧ p ∈ c3 ∧ p ∈ c4

-- The goal is to prove that there exists a point through which all five circles pass.
theorem five_circles_intersect :
  (∃ p : Point, p ∈ circle1 ∧ p ∈ circle2 ∧ p ∈ circle3 ∧ p ∈ circle4 ∧ p ∈ circle5) :=
sorry

end five_circles_intersect_l1650_165077
