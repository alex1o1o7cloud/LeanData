import Mathlib

namespace NUMINAMATH_GPT_probability_green_l1880_188059

def total_marbles : ℕ := 100

def P_white : ℚ := 1 / 4

def P_red_or_blue : ℚ := 0.55

def P_sum : ℚ := 1

theorem probability_green :
  P_sum = P_white + P_red_or_blue + P_green →
  P_green = 0.2 :=
sorry

end NUMINAMATH_GPT_probability_green_l1880_188059


namespace NUMINAMATH_GPT_impossible_to_achieve_25_percent_grape_juice_l1880_188041

theorem impossible_to_achieve_25_percent_grape_juice (x y : ℝ) 
  (h1 : ∀ a b : ℝ, (8 / (8 + 32) = 2 / 10) → (6 / (6 + 24) = 2 / 10))
  (h2 : (8 * x + 6 * y) / (40 * x + 30 * y) = 1 / 4) : false :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_achieve_25_percent_grape_juice_l1880_188041


namespace NUMINAMATH_GPT_lineup_count_l1880_188043

def total_players : ℕ := 15
def out_players : ℕ := 3  -- Alice, Max, and John
def lineup_size : ℕ := 6

-- Define the binomial coefficient in Lean
def binom (n k : ℕ) : ℕ :=
  if h : n ≥ k then
    Nat.choose n k
  else
    0

theorem lineup_count (total_players out_players lineup_size : ℕ) :
  let remaining_with_alice := total_players - out_players + 1 
  let remaining_without_alice := total_players - out_players + 1 
  let remaining_without_both := total_players - out_players 
  binom remaining_with_alice (lineup_size-1) + binom remaining_without_alice (lineup_size-1) + binom remaining_without_both lineup_size = 3498 :=
by
  sorry

end NUMINAMATH_GPT_lineup_count_l1880_188043


namespace NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1880_188016

noncomputable def area_ratio_triangle_GHI_JKL
  (a_GHI b_GHI c_GHI : ℕ) (a_JKL b_JKL c_JKL : ℕ) 
  (alt_ratio_GHI : ℕ × ℕ) (alt_ratio_JKL : ℕ × ℕ) : ℚ :=
  let area_GHI := (a_GHI * b_GHI) / 2
  let area_JKL := (a_JKL * b_JKL) / 2
  area_GHI / area_JKL

theorem ratio_of_areas_of_triangles :
  let GHI_sides := (7, 24, 25)
  let JKL_sides := (9, 40, 41)
  area_ratio_triangle_GHI_JKL 7 24 25 9 40 41 (2, 3) (4, 5) = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_triangles_l1880_188016


namespace NUMINAMATH_GPT_pyramid_base_length_of_tangent_hemisphere_l1880_188085

noncomputable def pyramid_base_side_length (radius height : ℝ) (tangent : ℝ → ℝ → Prop) : ℝ := sorry

theorem pyramid_base_length_of_tangent_hemisphere 
(r h : ℝ) (tangent : ℝ → ℝ → Prop) (tangent_property : ∀ x y, tangent x y → y = 0) 
(h_radius : r = 3) (h_height : h = 9) 
(tangent_conditions : tangent r h → tangent r h) : 
  pyramid_base_side_length r h tangent = 9 :=
sorry

end NUMINAMATH_GPT_pyramid_base_length_of_tangent_hemisphere_l1880_188085


namespace NUMINAMATH_GPT_evaluate_expression_l1880_188011

-- Define the operation * given by the table
def op (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1,1) => 1 | (1,2) => 2 | (1,3) => 3 | (1,4) => 4
  | (2,1) => 2 | (2,2) => 4 | (2,3) => 1 | (2,4) => 3
  | (3,1) => 3 | (3,2) => 1 | (3,3) => 4 | (3,4) => 2
  | (4,1) => 4 | (4,2) => 3 | (4,3) => 2 | (4,4) => 1
  | _ => 0  -- default to handle cases outside the defined table

-- Define the theorem to prove $(2*4)*(1*3) = 4$
theorem evaluate_expression : op (op 2 4) (op 1 3) = 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1880_188011


namespace NUMINAMATH_GPT_rectangle_side_ratio_square_l1880_188053

noncomputable def ratio_square (a b : ℝ) : ℝ :=
(a / b) ^ 2

theorem rectangle_side_ratio_square (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : 
  ratio_square a b = 4 := by
  sorry

end NUMINAMATH_GPT_rectangle_side_ratio_square_l1880_188053


namespace NUMINAMATH_GPT_ratio_proof_l1880_188062

theorem ratio_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = 3) :
    (x + 4 * y) / (4 * x - y) = 9 / 53 :=
  sorry

end NUMINAMATH_GPT_ratio_proof_l1880_188062


namespace NUMINAMATH_GPT_solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l1880_188067

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Proof Problem 1 Statement:
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} :=
sorry

-- Proof Problem 2 Statement:
theorem range_of_a_for_f_geq_abs_a_minus_4 (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ -1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_GPT_solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l1880_188067


namespace NUMINAMATH_GPT_binomial_expansion_example_l1880_188069

theorem binomial_expansion_example :
  57^3 + 3 * (57^2) * 4 + 3 * 57 * (4^2) + 4^3 = 226981 :=
by
  -- The proof would go here, using the steps outlined.
  sorry

end NUMINAMATH_GPT_binomial_expansion_example_l1880_188069


namespace NUMINAMATH_GPT_find_M_l1880_188035

theorem find_M 
  (M : ℕ)
  (h : 997 + 999 + 1001 + 1003 + 1005 = 5100 - M) :
  M = 95 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l1880_188035


namespace NUMINAMATH_GPT_product_of_a_l1880_188078

theorem product_of_a : 
  (∃ a b : ℝ, (3 * a - 5)^2 + (a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2 ∧ 
    (a * b = -8.32)) :=
by 
  sorry

end NUMINAMATH_GPT_product_of_a_l1880_188078


namespace NUMINAMATH_GPT_find_length_of_rod_l1880_188049

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end NUMINAMATH_GPT_find_length_of_rod_l1880_188049


namespace NUMINAMATH_GPT_inequality_not_holds_l1880_188003

variable (x y : ℝ)

theorem inequality_not_holds (h1 : x > 1) (h2 : 1 > y) : x - 1 ≤ 1 - y :=
sorry

end NUMINAMATH_GPT_inequality_not_holds_l1880_188003


namespace NUMINAMATH_GPT_Annie_cookies_sum_l1880_188028

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end NUMINAMATH_GPT_Annie_cookies_sum_l1880_188028


namespace NUMINAMATH_GPT_vasya_numbers_l1880_188006

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_vasya_numbers_l1880_188006


namespace NUMINAMATH_GPT_intervals_of_positivity_l1880_188081

theorem intervals_of_positivity :
  {x : ℝ | (x + 1) * (x - 1) * (x - 2) > 0} = {x : ℝ | (-1 < x ∧ x < 1) ∨ (2 < x)} :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_positivity_l1880_188081


namespace NUMINAMATH_GPT_root_in_interval_l1880_188045

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
sorry

end NUMINAMATH_GPT_root_in_interval_l1880_188045


namespace NUMINAMATH_GPT_distance_to_place_l1880_188080

theorem distance_to_place (rowing_speed still_water : ℝ) (downstream_speed : ℝ)
                         (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  rowing_speed = 10 → downstream_speed = 2 → upstream_speed = 3 →
  total_time = 10 → distance = 44.21 → 
  (distance / (rowing_speed + downstream_speed) + distance / (rowing_speed - upstream_speed)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  field_simp
  sorry

end NUMINAMATH_GPT_distance_to_place_l1880_188080


namespace NUMINAMATH_GPT_value_of_expression_l1880_188090

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1880_188090


namespace NUMINAMATH_GPT_country_X_tax_l1880_188023

theorem country_X_tax (I T x : ℝ) (hI : I = 51999.99) (hT : T = 8000) (h : T = 0.14 * x + 0.20 * (I - x)) : 
  x = 39999.97 := sorry

end NUMINAMATH_GPT_country_X_tax_l1880_188023


namespace NUMINAMATH_GPT_sugar_cheaper_than_apples_l1880_188009

/-- Given conditions about the prices and quantities of items that Fabian wants to buy,
    prove the price difference between one pack of sugar and one kilogram of apples. --/
theorem sugar_cheaper_than_apples
  (price_kg_apples : ℝ)
  (price_kg_walnuts : ℝ)
  (total_cost : ℝ)
  (cost_diff : ℝ)
  (num_kg_apples : ℕ := 5)
  (num_packs_sugar : ℕ := 3)
  (num_kg_walnuts : ℝ := 0.5)
  (price_kg_apples_val : price_kg_apples = 2)
  (price_kg_walnuts_val : price_kg_walnuts = 6)
  (total_cost_val : total_cost = 16) :
  cost_diff = price_kg_apples - (total_cost - (num_kg_apples * price_kg_apples + num_kg_walnuts * price_kg_walnuts))/num_packs_sugar → 
  cost_diff = 1 :=
by
  sorry

end NUMINAMATH_GPT_sugar_cheaper_than_apples_l1880_188009


namespace NUMINAMATH_GPT_eight_girls_circle_least_distance_l1880_188048

theorem eight_girls_circle_least_distance :
  let r := 50
  let num_girls := 8
  let total_distance := (8 * (3 * (r * Real.sqrt 2) + 2 * (2 * r)))
  total_distance = 1200 * Real.sqrt 2 + 1600 :=
by
  sorry

end NUMINAMATH_GPT_eight_girls_circle_least_distance_l1880_188048


namespace NUMINAMATH_GPT_greatest_award_correct_l1880_188052

-- Definitions and constants
def total_prize : ℕ := 600
def num_winners : ℕ := 15
def min_award : ℕ := 15
def prize_fraction_num : ℕ := 2
def prize_fraction_den : ℕ := 5
def winners_fraction_num : ℕ := 3
def winners_fraction_den : ℕ := 5

-- Conditions (translated and simplified)
def num_specific_winners : ℕ := (winners_fraction_num * num_winners) / winners_fraction_den
def specific_prize : ℕ := (prize_fraction_num * total_prize) / prize_fraction_den
def remaining_winners : ℕ := num_winners - num_specific_winners
def min_total_award_remaining : ℕ := remaining_winners * min_award
def remaining_prize : ℕ := total_prize - min_total_award_remaining
def min_award_specific : ℕ := num_specific_winners - 1
def sum_min_awards_specific : ℕ := min_award_specific * min_award

-- Correct answer
def greatest_award : ℕ := remaining_prize - sum_min_awards_specific

-- Theorem statement (Proof skipped with sorry)
theorem greatest_award_correct :
  greatest_award = 390 := sorry

end NUMINAMATH_GPT_greatest_award_correct_l1880_188052


namespace NUMINAMATH_GPT_alpha_eq_two_thirds_l1880_188007

theorem alpha_eq_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : α = 2 / 3 :=
sorry

end NUMINAMATH_GPT_alpha_eq_two_thirds_l1880_188007


namespace NUMINAMATH_GPT_probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l1880_188079

noncomputable def total_outcomes : ℕ := Nat.choose 6 2

noncomputable def prob_both_boys : ℚ := (Nat.choose 4 2 : ℚ) / total_outcomes

noncomputable def prob_exactly_one_girl : ℚ := ((Nat.choose 4 1) * (Nat.choose 2 1) : ℚ) / total_outcomes

noncomputable def prob_at_least_one_girl : ℚ := 1 - prob_both_boys

theorem probability_both_boys : prob_both_boys = 2 / 5 := by sorry
theorem probability_exactly_one_girl : prob_exactly_one_girl = 8 / 15 := by sorry
theorem probability_at_least_one_girl : prob_at_least_one_girl = 3 / 5 := by sorry

end NUMINAMATH_GPT_probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l1880_188079


namespace NUMINAMATH_GPT_symmetric_circle_with_respect_to_origin_l1880_188017

theorem symmetric_circle_with_respect_to_origin :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 1) ^ 2 = 1 → (x - 2) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  intros x y h
  -- Symmetric transformation and verification will be implemented here
  sorry

end NUMINAMATH_GPT_symmetric_circle_with_respect_to_origin_l1880_188017


namespace NUMINAMATH_GPT_quadratic_vertex_ordinate_l1880_188084

theorem quadratic_vertex_ordinate :
  let a := 2
  let b := -4
  let c := -1
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x ^ 2 + b * vertex_x + c
  vertex_y = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_vertex_ordinate_l1880_188084


namespace NUMINAMATH_GPT_probability_blackboard_empty_k_l1880_188064

-- Define the conditions for the problem
def Ben_blackboard_empty_probability (n : ℕ) : ℚ :=
  if h : n = 2013 then (2 * (2013 / 3) + 1) / 2^(2013 / 3 * 2) else 0 / 1

-- Define the theorem that Ben's blackboard is empty after 2013 flips, and determine k
theorem probability_blackboard_empty_k :
  ∃ (u v k : ℕ), Ben_blackboard_empty_probability 2013 = (2 * u + 1) / (2^k * (2 * v + 1)) ∧ k = 1336 :=
by sorry

end NUMINAMATH_GPT_probability_blackboard_empty_k_l1880_188064


namespace NUMINAMATH_GPT_Ferris_wheel_ticket_cost_l1880_188044

theorem Ferris_wheel_ticket_cost
  (cost_rc : ℕ) (rides_rc : ℕ) (cost_c : ℕ) (rides_c : ℕ) (total_tickets : ℕ) (rides_fw : ℕ)
  (H1 : cost_rc = 4) (H2 : rides_rc = 3) (H3 : cost_c = 4) (H4 : rides_c = 2) (H5 : total_tickets = 21) (H6 : rides_fw = 1) :
  21 - (3 * 4 + 2 * 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_Ferris_wheel_ticket_cost_l1880_188044


namespace NUMINAMATH_GPT_solve_for_x_l1880_188092

theorem solve_for_x : ∀ (x : ℝ), (2 * x + 3) / 5 = 11 → x = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1880_188092


namespace NUMINAMATH_GPT_intersection_A_B_l1880_188057

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l1880_188057


namespace NUMINAMATH_GPT_solve_for_N_l1880_188000

theorem solve_for_N : ∃ N : ℕ, 32^4 * 4^5 = 2^N ∧ N = 30 := by
  sorry

end NUMINAMATH_GPT_solve_for_N_l1880_188000


namespace NUMINAMATH_GPT_combined_weight_is_correct_l1880_188099

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_is_correct_l1880_188099


namespace NUMINAMATH_GPT_petri_dish_count_l1880_188005

theorem petri_dish_count (total_germs : ℝ) (germs_per_dish : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : germs_per_dish = 199.99999999999997) :
  total_germs / germs_per_dish = 18 :=
by
  sorry

end NUMINAMATH_GPT_petri_dish_count_l1880_188005


namespace NUMINAMATH_GPT_AdultsNotWearingBlue_l1880_188091

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end NUMINAMATH_GPT_AdultsNotWearingBlue_l1880_188091


namespace NUMINAMATH_GPT_geometric_sequence_tenth_term_l1880_188020

theorem geometric_sequence_tenth_term :
  let a : ℚ := 4
  let r : ℚ := 5/3
  let n : ℕ := 10
  a * r^(n-1) = 7812500 / 19683 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_tenth_term_l1880_188020


namespace NUMINAMATH_GPT_cost_effective_bus_choice_l1880_188008

theorem cost_effective_bus_choice (x y : ℕ) (h1 : y = x - 1) (h2 : 32 < 48 * x - 64 * y ∧ 48 * x - 64 * y < 64) : 
  64 * 300 < x * 2600 → True :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_effective_bus_choice_l1880_188008


namespace NUMINAMATH_GPT_odd_function_sum_l1880_188058

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_sum :
  (∀ x, f x = -f (-x)) ∧ 
  (∀ x y (hx : 3 ≤ x) (hy : y ≤ 7), x < y → f x < f y) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = 8) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = -1) →
  (2 * f (-6) + f (-3) = -15) :=
by
  intros
  sorry

end NUMINAMATH_GPT_odd_function_sum_l1880_188058


namespace NUMINAMATH_GPT_set_D_is_empty_l1880_188082

theorem set_D_is_empty :
  {x : ℝ | x^2 + 2 = 0} = ∅ :=
by {
  sorry
}

end NUMINAMATH_GPT_set_D_is_empty_l1880_188082


namespace NUMINAMATH_GPT_percentage_increase_in_average_visibility_l1880_188031

theorem percentage_increase_in_average_visibility :
  let avg_visibility_without_telescope := (100 + 110) / 2
  let avg_visibility_with_telescope := (150 + 165) / 2
  let increase_in_avg_visibility := avg_visibility_with_telescope - avg_visibility_without_telescope
  let percentage_increase := (increase_in_avg_visibility / avg_visibility_without_telescope) * 100
  percentage_increase = 50 := by
  -- calculations are omitted; proof goes here
  sorry

end NUMINAMATH_GPT_percentage_increase_in_average_visibility_l1880_188031


namespace NUMINAMATH_GPT_product_of_solutions_l1880_188001

theorem product_of_solutions (x : ℚ) (h : abs (12 / x + 3) = 2) :
  x = -12 ∨ x = -12 / 5 → x₁ * x₂ = 144 / 5 := by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l1880_188001


namespace NUMINAMATH_GPT_three_digit_number_problem_l1880_188093

theorem three_digit_number_problem (c d : ℕ) (h1 : 400 + c*10 + 1 = 786 - (300 + d*10 + 5)) (h2 : (300 + d*10 + 5) % 7 = 0) : c + d = 8 := 
sorry

end NUMINAMATH_GPT_three_digit_number_problem_l1880_188093


namespace NUMINAMATH_GPT_fundraiser_price_per_item_l1880_188086

theorem fundraiser_price_per_item
  (students_brownies : ℕ)
  (brownies_per_student : ℕ)
  (students_cookies : ℕ)
  (cookies_per_student : ℕ)
  (students_donuts : ℕ)
  (donuts_per_student : ℕ)
  (total_amount_raised : ℕ)
  (total_brownies : ℕ := students_brownies * brownies_per_student)
  (total_cookies : ℕ := students_cookies * cookies_per_student)
  (total_donuts : ℕ := students_donuts * donuts_per_student)
  (total_items : ℕ := total_brownies + total_cookies + total_donuts)
  (price_per_item : ℕ := total_amount_raised / total_items) :
  students_brownies = 30 →
  brownies_per_student = 12 →
  students_cookies = 20 →
  cookies_per_student = 24 →
  students_donuts = 15 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  price_per_item = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end NUMINAMATH_GPT_fundraiser_price_per_item_l1880_188086


namespace NUMINAMATH_GPT_total_value_of_item_l1880_188076

variable {V : ℝ}

theorem total_value_of_item (h : 0.07 * (V - 1000) = 109.20) : V = 2560 := 
by
  sorry

end NUMINAMATH_GPT_total_value_of_item_l1880_188076


namespace NUMINAMATH_GPT_rhombus_locus_l1880_188098

-- Define the coordinates of the vertices of the rhombus
structure Point :=
(x : ℝ)
(y : ℝ)

def A (e : ℝ) : Point := ⟨e, 0⟩
def B (f : ℝ) : Point := ⟨0, f⟩
def C (e : ℝ) : Point := ⟨-e, 0⟩
def D (f : ℝ) : Point := ⟨0, -f⟩

-- Define the distance squared from a point P to a point Q
def dist_sq (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the geometric locus problem
theorem rhombus_locus (P : Point) (e f : ℝ) :
  dist_sq P (A e) = dist_sq P (B f) + dist_sq P (C e) + dist_sq P (D f) ↔
  (if e > f then
    (dist_sq P (A e) = (e^2 - f^2) ∨ dist_sq P (C e) = (e^2 - f^2))
   else if e = f then
    (P = A e ∨ P = B f ∨ P = C e ∨ P = D f)
   else
    false) :=
sorry

end NUMINAMATH_GPT_rhombus_locus_l1880_188098


namespace NUMINAMATH_GPT_parallel_segments_k_value_l1880_188018

open Real

theorem parallel_segments_k_value :
  let A' := (-6, 0)
  let B' := (0, -6)
  let X' := (0, 12)
  ∃ k : ℝ,
  let Y' := (18, k)
  let m_ab := (B'.2 - A'.2) / (B'.1 - A'.1)
  let m_xy := (Y'.2 - X'.2) / (Y'.1 - X'.1)
  m_ab = m_xy → k = -6 :=
by
  sorry

end NUMINAMATH_GPT_parallel_segments_k_value_l1880_188018


namespace NUMINAMATH_GPT_picnic_problem_l1880_188083

theorem picnic_problem
  (M W C A : ℕ)
  (h1 : M + W + C = 240)
  (h2 : M = W + 80)
  (h3 : A = C + 80)
  (h4 : A = M + W) :
  M = 120 :=
by
  sorry

end NUMINAMATH_GPT_picnic_problem_l1880_188083


namespace NUMINAMATH_GPT_trapezium_distance_parallel_sides_l1880_188096

theorem trapezium_distance_parallel_sides (a b A : ℝ) (h : ℝ) (h1 : a = 20) (h2 : b = 18) (h3 : A = 380) :
  A = (1 / 2) * (a + b) * h → h = 20 :=
by
  intro h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_trapezium_distance_parallel_sides_l1880_188096


namespace NUMINAMATH_GPT_softball_players_l1880_188068

theorem softball_players (cricket hockey football total : ℕ) (h1 : cricket = 12) (h2 : hockey = 17) (h3 : football = 11) (h4 : total = 50) : 
  total - (cricket + hockey + football) = 10 :=
by
  sorry

end NUMINAMATH_GPT_softball_players_l1880_188068


namespace NUMINAMATH_GPT_sqrt_1708249_eq_1307_l1880_188072

theorem sqrt_1708249_eq_1307 :
  ∃ (n : ℕ), n * n = 1708249 ∧ n = 1307 :=
sorry

end NUMINAMATH_GPT_sqrt_1708249_eq_1307_l1880_188072


namespace NUMINAMATH_GPT_dogwood_trees_current_l1880_188037

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_trees_current_l1880_188037


namespace NUMINAMATH_GPT_fewer_trombone_than_trumpet_l1880_188088

theorem fewer_trombone_than_trumpet 
  (flute_players : ℕ)
  (trumpet_players : ℕ)
  (trombone_players : ℕ)
  (drummers : ℕ)
  (clarinet_players : ℕ)
  (french_horn_players : ℕ)
  (total_members : ℕ) :
  flute_players = 5 →
  trumpet_players = 3 * flute_players →
  clarinet_players = 2 * flute_players →
  drummers = trombone_players + 11 →
  french_horn_players = trombone_players + 3 →
  total_members = flute_players + clarinet_players + trumpet_players + trombone_players + drummers + french_horn_players →
  total_members = 65 →
  trombone_players = 7 ∧ trumpet_players - trombone_players = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3] at h6
  sorry

end NUMINAMATH_GPT_fewer_trombone_than_trumpet_l1880_188088


namespace NUMINAMATH_GPT_students_answered_both_questions_correctly_l1880_188010

theorem students_answered_both_questions_correctly (P_A P_B P_A'_B' : ℝ) (h_P_A : P_A = 0.75) (h_P_B : P_B = 0.7) (h_P_A'_B' : P_A'_B' = 0.2) :
  ∃ P_A_B : ℝ, P_A_B = 0.65 := 
by
  sorry

end NUMINAMATH_GPT_students_answered_both_questions_correctly_l1880_188010


namespace NUMINAMATH_GPT_mod_remainder_l1880_188024

theorem mod_remainder (a b c : ℕ) : 
  (7 * 10 ^ 20 + 1 ^ 20) % 11 = 8 := by
  -- Lean proof will be written here
  sorry

end NUMINAMATH_GPT_mod_remainder_l1880_188024


namespace NUMINAMATH_GPT_total_number_of_bees_is_fifteen_l1880_188034

noncomputable def totalBees (B : ℝ) : Prop :=
  (1/5) * B + (1/3) * B + (2/5) * B + 1 = B

theorem total_number_of_bees_is_fifteen : ∃ B : ℝ, totalBees B ∧ B = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_bees_is_fifteen_l1880_188034


namespace NUMINAMATH_GPT_next_wednesday_l1880_188040
open Nat

/-- Prove that the next year after 2010 when April 16 falls on a Wednesday is 2014,
    given the conditions:
    1. 2010 is a non-leap year.
    2. The day advances by 1 day for a non-leap year and 2 days for a leap year.
    3. April 16, 2010 was a Friday. -/
theorem next_wednesday (initial_year : ℕ) (initial_day : String) (target_day : String) : 
  (initial_year = 2010) ∧
  (initial_day = "Friday") ∧ 
  (target_day = "Wednesday") →
  2014 = 2010 + 4 :=
by
  sorry

end NUMINAMATH_GPT_next_wednesday_l1880_188040


namespace NUMINAMATH_GPT_sin_90_deg_l1880_188071

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end NUMINAMATH_GPT_sin_90_deg_l1880_188071


namespace NUMINAMATH_GPT_find_a_plus_b_l1880_188042

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_plus_b (a b : ℝ) (h_cond : ∀ x : ℝ, h (f a b x) = 4 * x + 3) : a + b = 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1880_188042


namespace NUMINAMATH_GPT_max_value_of_n_l1880_188073

theorem max_value_of_n (A B : ℤ) (h1 : A * B = 48) : 
  ∃ n, (∀ n', (∃ A' B', (A' * B' = 48) ∧ (n' = 2 * B' + 3 * A')) → n' ≤ n) ∧ n = 99 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_n_l1880_188073


namespace NUMINAMATH_GPT_number_of_sheep_l1880_188033

theorem number_of_sheep (legs animals : ℕ) (h1 : legs = 60) (h2 : animals = 20)
  (chickens sheep : ℕ) (hc : chickens + sheep = animals) (hl : 2 * chickens + 4 * sheep = legs) :
  sheep = 10 :=
sorry

end NUMINAMATH_GPT_number_of_sheep_l1880_188033


namespace NUMINAMATH_GPT_curves_intersect_on_x_axis_l1880_188097

theorem curves_intersect_on_x_axis (t θ a : ℝ) (h : a > 0) :
  (∃ t, (t + 1, 1 - 2 * t).snd = 0) →
  (∃ θ, (a * Real.cos θ, 3 * Real.cos θ).snd = 0) →
  (t + 1 = a * Real.cos θ) →
  a = 3 / 2 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_curves_intersect_on_x_axis_l1880_188097


namespace NUMINAMATH_GPT_distance_from_origin_l1880_188055

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end NUMINAMATH_GPT_distance_from_origin_l1880_188055


namespace NUMINAMATH_GPT_crayons_count_l1880_188089

def crayons_per_box : ℕ := 8
def number_of_boxes : ℕ := 10
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem crayons_count : total_crayons = 80 := by
  sorry

end NUMINAMATH_GPT_crayons_count_l1880_188089


namespace NUMINAMATH_GPT_find_normal_price_l1880_188026

open Real

theorem find_normal_price (P : ℝ) (h1 : 0.612 * P = 108) : P = 176.47 := by
  sorry

end NUMINAMATH_GPT_find_normal_price_l1880_188026


namespace NUMINAMATH_GPT_pond_field_ratio_l1880_188002

theorem pond_field_ratio (L W : ℕ) (pond_side : ℕ) (hL : L = 24) (hLW : L = 2 * W) (hPond : pond_side = 6) :
  pond_side * pond_side / (L * W) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_pond_field_ratio_l1880_188002


namespace NUMINAMATH_GPT_min_val_f_l1880_188046

noncomputable def f (x : ℝ) : ℝ :=
  4 / (x - 2) + x

theorem min_val_f (x : ℝ) (h : x > 2) : ∃ y, y = f x ∧ y ≥ 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_val_f_l1880_188046


namespace NUMINAMATH_GPT_triangle_cos_area_l1880_188074

/-- In triangle ABC, with angles A, B, and C, opposite sides a, b, and c respectively, given the condition 
    a * cos C = (2 * b - c) * cos A, prove: 
    1. cos A = 1/2
    2. If a = 6 and b + c = 8, then the area of triangle ABC is 7 * sqrt 3 / 3 --/
theorem triangle_cos_area (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (h2 : a = 6) (h3 : b + c = 8) :
  Real.cos A = 1 / 2 ∧ ∃ area : ℝ, area = 7 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_cos_area_l1880_188074


namespace NUMINAMATH_GPT_sum_of_integer_solutions_l1880_188056

theorem sum_of_integer_solutions (n_values : List ℤ) : 
  (∀ n ∈ n_values, ∃ (k : ℤ), 2 * n - 3 = k ∧ k ∣ 18) → (n_values.sum = 11) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_integer_solutions_l1880_188056


namespace NUMINAMATH_GPT_degrees_to_radians_18_l1880_188036

theorem degrees_to_radians_18 (degrees : ℝ) (h : degrees = 18) : 
  (degrees * (Real.pi / 180) = Real.pi / 10) :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_18_l1880_188036


namespace NUMINAMATH_GPT_polynomial_real_root_l1880_188050

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 + x + 1 = 0) ↔
  (a ∈ (Set.Iic (-1/2)) ∨ a ∈ (Set.Ici (1/2))) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_real_root_l1880_188050


namespace NUMINAMATH_GPT_soccer_tournament_games_l1880_188094

-- Define the single-elimination tournament problem
def single_elimination_games (teams : ℕ) : ℕ :=
  teams - 1

-- Define the specific problem instance
def teams := 20

-- State the theorem
theorem soccer_tournament_games : single_elimination_games teams = 19 :=
  sorry

end NUMINAMATH_GPT_soccer_tournament_games_l1880_188094


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1880_188038

theorem simplify_and_evaluate (x : Real) (h : x = Real.sqrt 2 - 1) :
  ( (1 / (x - 1) - 1 / (x + 1)) / (2 / (x - 1) ^ 2) ) = 1 - Real.sqrt 2 :=
by
  subst h
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1880_188038


namespace NUMINAMATH_GPT_shortest_side_length_l1880_188013

theorem shortest_side_length (A B C : ℝ) (a b c : ℝ)
  (h_sinA : Real.sin A = 5 / 13)
  (h_cosB : Real.cos B = 3 / 5)
  (h_longest : c = 63)
  (h_angles : A < B ∧ C = π - (A + B)) :
  a = 25 := by
sorry

end NUMINAMATH_GPT_shortest_side_length_l1880_188013


namespace NUMINAMATH_GPT_value_of_f3_f10_l1880_188065

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) : f (x + 4) = f x + f 2
axiom f_at_one : f 1 = 4

theorem value_of_f3_f10 : f 3 + f 10 = 4 := sorry

end NUMINAMATH_GPT_value_of_f3_f10_l1880_188065


namespace NUMINAMATH_GPT_election_result_l1880_188087

theorem election_result (total_votes : ℕ) (invalid_vote_percentage valid_vote_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (hv: valid_vote_percentage = 1 - invalid_vote_percentage) 
  (ht: total_votes = 560000) 
  (hi: invalid_vote_percentage = 0.15) 
  (hc: candidate_A_percentage = 0.80) : 
  (candidate_A_percentage * valid_vote_percentage * total_votes = 380800) :=
by 
  sorry

end NUMINAMATH_GPT_election_result_l1880_188087


namespace NUMINAMATH_GPT_fraction_zero_implies_x_zero_l1880_188061

theorem fraction_zero_implies_x_zero (x : ℝ) (h : x / (2 * x - 1) = 0) : x = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_zero_implies_x_zero_l1880_188061


namespace NUMINAMATH_GPT_can_use_bisection_method_l1880_188060

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := abs x
noncomputable def f4 (x : ℝ) : ℝ := x^3

theorem can_use_bisection_method : ∃ (a b : ℝ), a < b ∧ (f4 a) * (f4 b) < 0 := 
sorry

end NUMINAMATH_GPT_can_use_bisection_method_l1880_188060


namespace NUMINAMATH_GPT_inequality_solution_l1880_188066

theorem inequality_solution (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1880_188066


namespace NUMINAMATH_GPT_Norbs_age_l1880_188047

def guesses : List ℕ := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def two_off_by_one (n : ℕ) (guesses : List ℕ) : Prop := 
  (n - 1 ∈ guesses) ∧ (n + 1 ∈ guesses)

def at_least_half_too_low (n : ℕ) (guesses : List ℕ) : Prop := 
  (guesses.filter (· < n)).length ≥ guesses.length / 2

theorem Norbs_age : 
  ∃ x, is_prime x ∧ two_off_by_one x guesses ∧ at_least_half_too_low x guesses ∧ x = 37 := 
by 
  sorry

end NUMINAMATH_GPT_Norbs_age_l1880_188047


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l1880_188025

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : sin_cos_identity α) :
  Real.tan (α + Real.pi / 4) = -3 :=
  by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l1880_188025


namespace NUMINAMATH_GPT_eval_expression_l1880_188015

theorem eval_expression : (2^5 - 5^2) = 7 :=
by {
  -- Proof steps will be here
  sorry
}

end NUMINAMATH_GPT_eval_expression_l1880_188015


namespace NUMINAMATH_GPT_units_digit_p_plus_5_l1880_188039

theorem units_digit_p_plus_5 (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 = 6) (h3 : (p^3 % 10) - (p^2 % 10) = 0) : (p + 5) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_p_plus_5_l1880_188039


namespace NUMINAMATH_GPT_literature_more_than_science_science_less_than_literature_percent_l1880_188021

theorem literature_more_than_science (l s : ℕ) (h : 8 * s = 5 * l) : (l - s) / s = 3 / 5 :=
by {
  -- definition and given condition will be provided
  sorry
}

theorem science_less_than_literature_percent (l s : ℕ) (h : 8 * s = 5 * l) : ((l - s : ℚ) / l) * 100 = 37.5 :=
by {
  -- definition and given condition will be provided
  sorry
}

end NUMINAMATH_GPT_literature_more_than_science_science_less_than_literature_percent_l1880_188021


namespace NUMINAMATH_GPT_Jazmin_strips_width_l1880_188004

theorem Jazmin_strips_width (w1 w2 g : ℕ) (h1 : w1 = 44) (h2 : w2 = 33) (hg : g = Nat.gcd w1 w2) : g = 11 := by
  -- Markdown above outlines:
  -- w1, w2 are widths of the construction paper
  -- h1: w1 = 44
  -- h2: w2 = 33
  -- hg: g = gcd(w1, w2)
  -- Prove g == 11
  sorry

end NUMINAMATH_GPT_Jazmin_strips_width_l1880_188004


namespace NUMINAMATH_GPT_solve_quadratics_and_sum_l1880_188032

theorem solve_quadratics_and_sum (d e f : ℤ) 
  (h1 : ∃ d e : ℤ, d + e = 19 ∧ d * e = 88) 
  (h2 : ∃ e f : ℤ, e + f = 23 ∧ e * f = 120) : 
  d + e + f = 31 := by
  sorry

end NUMINAMATH_GPT_solve_quadratics_and_sum_l1880_188032


namespace NUMINAMATH_GPT_values_of_a_and_b_l1880_188051

theorem values_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2 * x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) → (a = 2 ∧ b = 1) :=
by 
  sorry

end NUMINAMATH_GPT_values_of_a_and_b_l1880_188051


namespace NUMINAMATH_GPT_miles_run_on_tuesday_l1880_188077

-- Defining the distances run on specific days
def distance_monday : ℝ := 4.2
def distance_wednesday : ℝ := 3.6
def distance_thursday : ℝ := 4.4

-- Average distance run on each of the days Terese runs
def average_distance : ℝ := 4
-- Number of days Terese runs
def running_days : ℕ := 4

-- Defining the total distance calculated using the average distance and number of days
def total_distance : ℝ := average_distance * running_days

-- Defining the total distance run on Monday, Wednesday, and Thursday
def total_other_days : ℝ := distance_monday + distance_wednesday + distance_thursday

-- The distance run on Tuesday can be defined as the difference between the total distance and the total distance on other days
theorem miles_run_on_tuesday : 
  total_distance - total_other_days = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_miles_run_on_tuesday_l1880_188077


namespace NUMINAMATH_GPT_julia_played_tag_with_4_kids_on_tuesday_l1880_188014

variable (k_monday : ℕ) (k_diff : ℕ)

theorem julia_played_tag_with_4_kids_on_tuesday
  (h_monday : k_monday = 16)
  (h_diff : k_monday = k_tuesday + 12) :
  k_tuesday = 4 :=
by
  sorry

end NUMINAMATH_GPT_julia_played_tag_with_4_kids_on_tuesday_l1880_188014


namespace NUMINAMATH_GPT_length_of_LN_l1880_188095

theorem length_of_LN 
  (sinN : ℝ)
  (LM LN : ℝ)
  (h1 : sinN = 3 / 5)
  (h2 : LM = 20)
  (h3 : sinN = LM / LN) :
  LN = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_LN_l1880_188095


namespace NUMINAMATH_GPT_percentage_of_female_students_l1880_188070

theorem percentage_of_female_students {F : ℝ} (h1 : 200 > 0): ((200 * (F / 100)) * 0.5 * 0.5 = 30) → (F = 60) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_female_students_l1880_188070


namespace NUMINAMATH_GPT_greatest_divisor_four_consecutive_l1880_188075

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_four_consecutive_l1880_188075


namespace NUMINAMATH_GPT_range_of_m_l1880_188063

theorem range_of_m (m : ℝ) : 
  ((m - 1) * x^2 - 4 * x + 1 = 0) → 
  ((20 - 4 * m ≥ 0) ∧ (m ≠ 1)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1880_188063


namespace NUMINAMATH_GPT_arithmetic_progression_power_of_two_l1880_188012

theorem arithmetic_progression_power_of_two 
  (a d : ℤ) (n : ℕ) (k : ℕ) 
  (Sn : ℤ)
  (h_sum : Sn = 2^k)
  (h_ap : Sn = n * (2 * a + (n - 1) * d) / 2)  :
  ∃ m : ℕ, n = 2^m := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_power_of_two_l1880_188012


namespace NUMINAMATH_GPT_age_ratio_l1880_188030

-- Definitions as per the conditions
variable (j e x : ℕ)

-- Conditions from the problem
def condition1 : Prop := j - 4 = 2 * (e - 4)
def condition2 : Prop := j - 10 = 3 * (e - 10)

-- The statement we need to prove
theorem age_ratio (j e x : ℕ) (h1 : condition1 j e)
(h2 : condition2 j e) :
(j + x) * 2 = (e + x) * 3 ↔ x = 8 :=
sorry

end NUMINAMATH_GPT_age_ratio_l1880_188030


namespace NUMINAMATH_GPT_mary_stickers_left_l1880_188027

def initial_stickers : ℕ := 50
def stickers_per_friend : ℕ := 4
def number_of_friends : ℕ := 5
def total_students_including_mary : ℕ := 17
def stickers_per_other_student : ℕ := 2

theorem mary_stickers_left :
  let friends_stickers := stickers_per_friend * number_of_friends
  let other_students := total_students_including_mary - 1 - number_of_friends
  let other_students_stickers := stickers_per_other_student * other_students
  let total_given_away := friends_stickers + other_students_stickers
  initial_stickers - total_given_away = 8 :=
by
  sorry

end NUMINAMATH_GPT_mary_stickers_left_l1880_188027


namespace NUMINAMATH_GPT_basketball_game_count_l1880_188054

noncomputable def total_games_played (teams games_each_opp : ℕ) : ℕ :=
  (teams * (teams - 1) / 2) * games_each_opp

theorem basketball_game_count (n : ℕ) (g : ℕ) (h_n : n = 10) (h_g : g = 4) : total_games_played n g = 180 :=
by
  -- Use 'h_n' and 'h_g' as hypotheses
  rw [h_n, h_g]
  show (10 * 9 / 2) * 4 = 180
  sorry

end NUMINAMATH_GPT_basketball_game_count_l1880_188054


namespace NUMINAMATH_GPT_geometric_sum_of_ratios_l1880_188029

theorem geometric_sum_of_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ) 
  (ha2 : a2 = k * p) (ha3 : a3 = k * p^2) 
  (hb2 : b2 = k * r) (hb3 : b3 = k * r^2) 
  (h : a3 - b3 = 5 * (a2 - b2)) :
  p + r = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sum_of_ratios_l1880_188029


namespace NUMINAMATH_GPT_billboards_and_road_length_l1880_188019

theorem billboards_and_road_length :
  ∃ (x y : ℕ), 5 * (x + 21 - 1) = y ∧ (55 * (x - 1)) / 10 = y ∧ x = 200 ∧ y = 1100 :=
sorry

end NUMINAMATH_GPT_billboards_and_road_length_l1880_188019


namespace NUMINAMATH_GPT_perimeter_of_playground_l1880_188022

theorem perimeter_of_playground 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 900) 
  (h2 : x * y = 216) : 
  2 * (x + y) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_playground_l1880_188022
