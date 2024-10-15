import Mathlib

namespace NUMINAMATH_GPT_point_A_coordinates_l2178_217852

variable {a : ℝ}
variable {f : ℝ → ℝ}

theorem point_A_coordinates (h1 : a > 0) (h2 : a ≠ 1) (hf : ∀ x, f x = a^(x - 1)) :
  f 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_point_A_coordinates_l2178_217852


namespace NUMINAMATH_GPT_determine_b_l2178_217806

-- Define the problem conditions
variable (n b : ℝ)
variable (h_pos_b : b > 0)
variable (h_eq : ∀ x : ℝ, (x + n) ^ 2 + 16 = x^2 + b * x + 88)

-- State that we want to prove that b equals 12 * sqrt(2)
theorem determine_b : b = 12 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l2178_217806


namespace NUMINAMATH_GPT_max_rectangle_perimeter_l2178_217895

theorem max_rectangle_perimeter (n : ℕ) (a b : ℕ) (ha : a * b = 180) (hb: ∀ (a b : ℕ),  6 ∣ (a * b) → a * b = 180): 
  2 * (a + b) ≤ 184 :=
sorry

end NUMINAMATH_GPT_max_rectangle_perimeter_l2178_217895


namespace NUMINAMATH_GPT_expression_value_l2178_217893

theorem expression_value (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x ^ 4 - 6 * x ^ 3 - 2 * x ^ 2 + 18 * x + 23) / (x ^ 2 - 8 * x + 15) = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l2178_217893


namespace NUMINAMATH_GPT_grape_juice_amount_l2178_217831

-- Definitions for the conditions
def total_weight : ℝ := 150
def orange_percentage : ℝ := 0.35
def watermelon_percentage : ℝ := 0.35

-- Theorem statement to prove the amount of grape juice
theorem grape_juice_amount : 
  (total_weight * (1 - orange_percentage - watermelon_percentage)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_grape_juice_amount_l2178_217831


namespace NUMINAMATH_GPT_magnitude_of_resultant_vector_is_sqrt_5_l2178_217820

-- We denote the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b (y : ℝ) : ℝ × ℝ := (-2, y)

-- We encode the condition that vectors are parallel
def parallel_vectors (y : ℝ) : Prop := 1 * y = (-2) * (-2)

-- We calculate the resultant vector and its magnitude
def resultant_vector (y : ℝ) : ℝ × ℝ :=
  ((3 * 1 + 2 * -2), (3 * -2 + 2 * y))

def magnitude_square (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- The target statement
theorem magnitude_of_resultant_vector_is_sqrt_5 (y : ℝ) (hy : parallel_vectors y) :
  magnitude_square (resultant_vector y) = 5 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_resultant_vector_is_sqrt_5_l2178_217820


namespace NUMINAMATH_GPT_A_subset_B_l2178_217824

def A (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 ≤ 5 / 4

def B (x y : ℝ) (a : ℝ) : Prop :=
  abs (x - 1) + 2 * abs (y - 2) ≤ a

theorem A_subset_B (a : ℝ) (h : a ≥ 5 / 2) : 
  ∀ x y : ℝ, A x y → B x y a := 
sorry

end NUMINAMATH_GPT_A_subset_B_l2178_217824


namespace NUMINAMATH_GPT_megan_earnings_l2178_217819

-- Define the given conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- Define the total number of necklaces
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

-- Define the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings are 90 dollars
theorem megan_earnings : total_earnings = 90 := by
  sorry

end NUMINAMATH_GPT_megan_earnings_l2178_217819


namespace NUMINAMATH_GPT_max_matching_pairs_l2178_217850

theorem max_matching_pairs 
  (total_pairs : ℕ := 23) 
  (total_colors : ℕ := 6) 
  (total_sizes : ℕ := 3) 
  (lost_shoes : ℕ := 9)
  (shoes_per_pair : ℕ := 2) 
  (total_shoes := total_pairs * shoes_per_pair) 
  (remaining_shoes := total_shoes - lost_shoes) :
  ∃ max_pairs : ℕ, max_pairs = total_pairs - lost_shoes / shoes_per_pair :=
sorry

end NUMINAMATH_GPT_max_matching_pairs_l2178_217850


namespace NUMINAMATH_GPT_inequality_not_less_than_l2178_217859

theorem inequality_not_less_than (y : ℝ) : 2 * y + 8 ≥ -3 := 
sorry

end NUMINAMATH_GPT_inequality_not_less_than_l2178_217859


namespace NUMINAMATH_GPT_a_plus_b_minus_c_in_S_l2178_217834

-- Define the sets P, Q, and S
def P := {x : ℤ | ∃ k : ℤ, x = 3 * k}
def Q := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def S := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

-- Define the elements a, b, and c as members of sets P, Q, and S respectively
variables (a b c : ℤ)
variable (ha : a ∈ P) -- a ∈ P
variable (hb : b ∈ Q) -- b ∈ Q
variable (hc : c ∈ S) -- c ∈ S

-- Theorem statement proving the question
theorem a_plus_b_minus_c_in_S : a + b - c ∈ S := sorry

end NUMINAMATH_GPT_a_plus_b_minus_c_in_S_l2178_217834


namespace NUMINAMATH_GPT_min_value_of_expression_l2178_217856

open Real

theorem min_value_of_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2178_217856


namespace NUMINAMATH_GPT_find_n_l2178_217871

theorem find_n (n : ℕ) (h : (1 + n) / (2 ^ n) = 3 / 16) : n = 5 :=
by sorry

end NUMINAMATH_GPT_find_n_l2178_217871


namespace NUMINAMATH_GPT_permutations_BANANA_l2178_217886

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end NUMINAMATH_GPT_permutations_BANANA_l2178_217886


namespace NUMINAMATH_GPT_equal_sharing_l2178_217889

theorem equal_sharing (total_cards friends : ℕ) (h1 : total_cards = 455) (h2 : friends = 5) : total_cards / friends = 91 := by
  sorry

end NUMINAMATH_GPT_equal_sharing_l2178_217889


namespace NUMINAMATH_GPT_wendy_pictures_in_one_album_l2178_217802

theorem wendy_pictures_in_one_album 
  (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ)
  (h_total : total_pictures = 45) (h_pictures_per_album : pictures_per_album = 2) 
  (h_num_other_albums : num_other_albums = 9) : 
  ∃ (pictures_in_one_album : ℕ), pictures_in_one_album = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_wendy_pictures_in_one_album_l2178_217802


namespace NUMINAMATH_GPT_collective_land_area_l2178_217868

theorem collective_land_area 
  (C W : ℕ) 
  (h1 : 42 * C + 35 * W = 165200)
  (h2 : W = 3400)
  : C + W = 4500 :=
sorry

end NUMINAMATH_GPT_collective_land_area_l2178_217868


namespace NUMINAMATH_GPT_geom_sequence_sum_l2178_217875

theorem geom_sequence_sum (n : ℕ) (a : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 4 ^ n + a) : 
  a = -1 := 
by
  sorry

end NUMINAMATH_GPT_geom_sequence_sum_l2178_217875


namespace NUMINAMATH_GPT_curve_is_circle_l2178_217862

noncomputable def curve_eqn_polar (r θ : ℝ) : Prop :=
  r = 1 / (Real.sin θ + Real.cos θ)

theorem curve_is_circle : ∀ r θ, curve_eqn_polar r θ →
  ∃ x y : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ 
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_curve_is_circle_l2178_217862


namespace NUMINAMATH_GPT_number_of_cubes_with_icing_on_two_sides_l2178_217836

def cake_cube : ℕ := 3
def smaller_cubes : ℕ := 27
def covered_faces : ℕ := 3
def layers_with_icing : ℕ := 2
def edge_cubes_per_layer_per_face : ℕ := 2

theorem number_of_cubes_with_icing_on_two_sides :
  (covered_faces * edge_cubes_per_layer_per_face * layers_with_icing) = 12 := by
  sorry

end NUMINAMATH_GPT_number_of_cubes_with_icing_on_two_sides_l2178_217836


namespace NUMINAMATH_GPT_fraction_of_male_first_class_l2178_217804

theorem fraction_of_male_first_class (total_passengers : ℕ) (percent_female : ℚ) (percent_first_class : ℚ)
    (females_in_coach : ℕ) (h1 : total_passengers = 120) (h2 : percent_female = 0.45) (h3 : percent_first_class = 0.10)
    (h4 : females_in_coach = 46) :
    (((percent_first_class * total_passengers - (percent_female * total_passengers - females_in_coach)))
    / (percent_first_class * total_passengers))  = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_male_first_class_l2178_217804


namespace NUMINAMATH_GPT_find_a_l2178_217833

open Complex

theorem find_a (a : ℝ) (i : ℂ := Complex.I) (h : (a - i) ^ 2 = 2 * i) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l2178_217833


namespace NUMINAMATH_GPT_yi_catches_jia_on_DA_l2178_217896

def square_side_length : ℝ := 90
def jia_speed : ℝ := 65
def yi_speed : ℝ := 72
def jia_start : ℝ := 0
def yi_start : ℝ := 90

theorem yi_catches_jia_on_DA :
  let square_perimeter := 4 * square_side_length
  let initial_gap := 3 * square_side_length
  let relative_speed := yi_speed - jia_speed
  let time_to_catch := initial_gap / relative_speed
  let distance_travelled_by_yi := yi_speed * time_to_catch
  let number_of_laps := distance_travelled_by_yi / square_perimeter
  let additional_distance := distance_travelled_by_yi % square_perimeter
  additional_distance = 0 →
  square_side_length * (number_of_laps % 4) = 0 ∨ number_of_laps % 4 = 3 :=
by
  -- We only provide the statement, the proof is omitted.
  sorry

end NUMINAMATH_GPT_yi_catches_jia_on_DA_l2178_217896


namespace NUMINAMATH_GPT_pair_exists_l2178_217867

theorem pair_exists (x : Fin 670 → ℝ) (h_distinct : Function.Injective x) (h_bounds : ∀ i, 0 < x i ∧ x i < 1) :
  ∃ (i j : Fin 670), 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := 
by
  sorry

end NUMINAMATH_GPT_pair_exists_l2178_217867


namespace NUMINAMATH_GPT_transmitter_finding_probability_l2178_217825

/-- 
  A license plate in the country Kerrania consists of 4 digits followed by two letters.
  The letters A, B, and C are used only by government vehicles while the letters D through Z are used by non-government vehicles.
  Kerrania's intelligence agency has recently captured a message from the country Gonzalia indicating that an electronic transmitter 
  has been installed in a Kerrania government vehicle with a license plate starting with 79. 
  In addition, the message reveals that the last three digits of the license plate form a palindromic sequence (meaning that they are 
  the same forward and backward), and the second digit is either a 3 or a 5. 
  If it takes the police 10 minutes to inspect each vehicle, what is the probability that the police will find the transmitter 
  within 3 hours, considering the additional restrictions on the possible license plate combinations?
-/
theorem transmitter_finding_probability :
  0.1 = 18 / 180 :=
by
  sorry

end NUMINAMATH_GPT_transmitter_finding_probability_l2178_217825


namespace NUMINAMATH_GPT_angle_bisector_ratio_l2178_217897

theorem angle_bisector_ratio (A B C Q : Type) (AC CB AQ QB : ℝ) (k : ℝ) 
  (hAC : AC = 4 * k) (hCB : CB = 5 * k) (angle_bisector_theorem : AQ / QB = AC / CB) :
  AQ / QB = 4 / 5 := 
by sorry

end NUMINAMATH_GPT_angle_bisector_ratio_l2178_217897


namespace NUMINAMATH_GPT_differentiable_function_inequality_l2178_217899

theorem differentiable_function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * (f 1) :=
sorry

end NUMINAMATH_GPT_differentiable_function_inequality_l2178_217899


namespace NUMINAMATH_GPT_staples_left_in_stapler_l2178_217800

def initial_staples : ℕ := 50
def reports_stapled : ℕ := 3 * 12
def staples_per_report : ℕ := 1
def remaining_staples : ℕ := initial_staples - (reports_stapled * staples_per_report)

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  sorry

end NUMINAMATH_GPT_staples_left_in_stapler_l2178_217800


namespace NUMINAMATH_GPT_abs_eq_condition_l2178_217828

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_condition_l2178_217828


namespace NUMINAMATH_GPT_volume_of_cone_l2178_217848

theorem volume_of_cone
  (r h l : ℝ) -- declaring variables
  (base_area : ℝ) (lateral_surface_is_semicircle : ℝ) 
  (h_eq : h = Real.sqrt (l^2 - r^2))
  (base_area_eq : π * r^2 = π)
  (lateral_surface_eq : π * l = 2 * π * r) : 
  (∀ (V : ℝ), V = (1 / 3) * π * r^2 * h → V = (Real.sqrt 3) / 3 * π) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cone_l2178_217848


namespace NUMINAMATH_GPT_total_legs_of_passengers_l2178_217890

theorem total_legs_of_passengers :
  ∀ (total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs : ℕ),
  total_heads = 15 →
  cats = 7 →
  cat_legs = 4 →
  human_heads = (total_heads - cats) →
  normal_human_legs = 2 →
  one_legged_captain_legs = 1 →
  ((cats * cat_legs) + ((human_heads - 1) * normal_human_legs) + one_legged_captain_legs) = 43 :=
by
  intros total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_legs_of_passengers_l2178_217890


namespace NUMINAMATH_GPT_flat_tyre_problem_l2178_217860

theorem flat_tyre_problem
    (x : ℝ)
    (h1 : 0 < x)
    (h2 : 1 / x + 1 / 6 = 1 / 5.6) :
  x = 84 :=
sorry

end NUMINAMATH_GPT_flat_tyre_problem_l2178_217860


namespace NUMINAMATH_GPT_Adam_current_money_is_8_l2178_217857

variable (Adam_initial : ℕ) (spent_on_game : ℕ) (allowance : ℕ)

def money_left_after_spending (initial : ℕ) (spent : ℕ) := initial - spent
def current_money (money_left : ℕ) (allowance : ℕ) := money_left + allowance

theorem Adam_current_money_is_8 
    (h1 : Adam_initial = 5)
    (h2 : spent_on_game = 2)
    (h3 : allowance = 5) :
    current_money (money_left_after_spending Adam_initial spent_on_game) allowance = 8 := 
by sorry

end NUMINAMATH_GPT_Adam_current_money_is_8_l2178_217857


namespace NUMINAMATH_GPT_remaining_payment_l2178_217878

theorem remaining_payment (part_payment total_cost : ℝ) (percent_payment : ℝ) 
  (h1 : part_payment = 650) 
  (h2 : percent_payment = 15 / 100) 
  (h3 : part_payment = percent_payment * total_cost) : 
  total_cost - part_payment = 3683.33 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_payment_l2178_217878


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2178_217813

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 → ¬ (x - 1)^2 < 9) →
  a < -4 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2178_217813


namespace NUMINAMATH_GPT_rectangle_side_greater_than_12_l2178_217812

theorem rectangle_side_greater_than_12 
  (a b : ℝ) (h₁ : a ≠ b) (h₂ : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_side_greater_than_12_l2178_217812


namespace NUMINAMATH_GPT_additional_license_plates_l2178_217839

def original_license_plates : ℕ := 5 * 3 * 5
def new_license_plates : ℕ := 6 * 4 * 5

theorem additional_license_plates : new_license_plates - original_license_plates = 45 := by
  sorry

end NUMINAMATH_GPT_additional_license_plates_l2178_217839


namespace NUMINAMATH_GPT_nitin_ranks_from_last_l2178_217866

def total_students : ℕ := 75

def math_rank_start : ℕ := 24
def english_rank_start : ℕ := 18

def rank_from_last (total : ℕ) (rank_start : ℕ) : ℕ :=
  total - rank_start + 1

theorem nitin_ranks_from_last :
  rank_from_last total_students math_rank_start = 52 ∧
  rank_from_last total_students english_rank_start = 58 :=
by
  sorry

end NUMINAMATH_GPT_nitin_ranks_from_last_l2178_217866


namespace NUMINAMATH_GPT_range_of_a_l2178_217874

variable (a : ℝ)
def p : Prop := a > 1/4
def q : Prop := a ≤ -1 ∨ a ≥ 1

theorem range_of_a :
  ((p a ∧ ¬ (q a)) ∨ (q a ∧ ¬ (p a))) ↔ (a > 1/4 ∧ a < 1) ∨ (a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2178_217874


namespace NUMINAMATH_GPT_four_digit_composite_l2178_217846

theorem four_digit_composite (abcd : ℕ) (h : 1000 ≤ abcd ∧ abcd < 10000) :
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≥ 2 ∧ m * n = (abcd * 10001) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_composite_l2178_217846


namespace NUMINAMATH_GPT_parabola_directrix_eq_l2178_217892

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end NUMINAMATH_GPT_parabola_directrix_eq_l2178_217892


namespace NUMINAMATH_GPT_sole_mart_meals_l2178_217894

theorem sole_mart_meals (c_c_meals : ℕ) (meals_given_away : ℕ) (meals_left : ℕ)
  (h1 : c_c_meals = 113) (h2 : meals_givenAway = 85) (h3 : meals_left = 78)  :
  ∃ m : ℕ, m + c_c_meals = meals_givenAway + meals_left ∧ m = 50 := 
by
  sorry

end NUMINAMATH_GPT_sole_mart_meals_l2178_217894


namespace NUMINAMATH_GPT_drunk_drivers_traffic_class_l2178_217849

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end NUMINAMATH_GPT_drunk_drivers_traffic_class_l2178_217849


namespace NUMINAMATH_GPT_card_sequence_probability_l2178_217832

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end NUMINAMATH_GPT_card_sequence_probability_l2178_217832


namespace NUMINAMATH_GPT_average_of_25_results_is_24_l2178_217816

theorem average_of_25_results_is_24 
  (first12_sum : ℕ)
  (last12_sum : ℕ)
  (result13 : ℕ)
  (n1 n2 n3 : ℕ)
  (h1 : n1 = 12)
  (h2 : n2 = 12)
  (h3 : n3 = 25)
  (avg_first12 : first12_sum = 14 * n1)
  (avg_last12 : last12_sum = 17 * n2)
  (res_13 : result13 = 228) :
  (first12_sum + last12_sum + result13) / n3 = 24 :=
by
  sorry

end NUMINAMATH_GPT_average_of_25_results_is_24_l2178_217816


namespace NUMINAMATH_GPT_mask_usage_duration_l2178_217887

-- Define given conditions
def TotalMasks : ℕ := 75
def FamilyMembers : ℕ := 7
def MaskChangeInterval : ℕ := 2

-- Define the goal statement, which is to prove that the family will take 21 days to use all masks
theorem mask_usage_duration 
  (M : ℕ := 75)  -- total masks
  (N : ℕ := 7)   -- family members
  (d : ℕ := 2)   -- mask change interval
  : (M / N) * d + 1 = 21 :=
sorry

end NUMINAMATH_GPT_mask_usage_duration_l2178_217887


namespace NUMINAMATH_GPT_total_initial_amounts_l2178_217821

theorem total_initial_amounts :
  ∃ (a j t : ℝ), a = 50 ∧ t = 50 ∧ (50 + j + 50 = 187.5) :=
sorry

end NUMINAMATH_GPT_total_initial_amounts_l2178_217821


namespace NUMINAMATH_GPT_systematic_sampling_student_selection_l2178_217827

theorem systematic_sampling_student_selection
    (total_students : ℕ)
    (num_groups : ℕ)
    (students_per_group : ℕ)
    (third_group_selected : ℕ)
    (third_group_num : ℕ)
    (eighth_group_num : ℕ)
    (h1 : total_students = 50)
    (h2 : num_groups = 10)
    (h3 : students_per_group = total_students / num_groups)
    (h4 : students_per_group = 5)
    (h5 : 11 ≤ third_group_selected ∧ third_group_selected ≤ 15)
    (h6 : third_group_selected = 12)
    (h7 : third_group_num = 3)
    (h8 : eighth_group_num = 8) :
  eighth_group_selected = 37 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_student_selection_l2178_217827


namespace NUMINAMATH_GPT_sum_of_4n_pos_integers_l2178_217863

theorem sum_of_4n_pos_integers (n : ℕ) (Sn : ℕ → ℕ)
  (hSn : ∀ k, Sn k = k * (k + 1) / 2)
  (h_condition : Sn (3 * n) - Sn n = 150) :
  Sn (4 * n) = 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_4n_pos_integers_l2178_217863


namespace NUMINAMATH_GPT_willy_crayons_difference_l2178_217858

def willy : Int := 5092
def lucy : Int := 3971
def jake : Int := 2435

theorem willy_crayons_difference : willy - (lucy + jake) = -1314 := by
  sorry

end NUMINAMATH_GPT_willy_crayons_difference_l2178_217858


namespace NUMINAMATH_GPT_twentieth_triangular_number_l2178_217869

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentieth_triangular_number : triangular_number 20 = 210 :=
by
  sorry

end NUMINAMATH_GPT_twentieth_triangular_number_l2178_217869


namespace NUMINAMATH_GPT_smallest_white_marbles_l2178_217881

/-
Let n be the total number of Peter's marbles.
Half of the marbles are orange.
One fifth of the marbles are purple.
Peter has 8 silver marbles.
-/
def total_marbles (n : ℕ) : ℕ :=
  n

def orange_marbles (n : ℕ) : ℕ :=
  n / 2

def purple_marbles (n : ℕ) : ℕ :=
  n / 5

def silver_marbles : ℕ :=
  8

def white_marbles (n : ℕ) : ℕ :=
  n - (orange_marbles n + purple_marbles n + silver_marbles)

-- Prove that the smallest number of white marbles Peter could have is 1.
theorem smallest_white_marbles : ∃ n : ℕ, n % 10 = 0 ∧ white_marbles n = 1 :=
sorry

end NUMINAMATH_GPT_smallest_white_marbles_l2178_217881


namespace NUMINAMATH_GPT_y_is_never_perfect_square_l2178_217826

theorem y_is_never_perfect_square (x : ℕ) : ¬ ∃ k : ℕ, k^2 = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 :=
sorry

end NUMINAMATH_GPT_y_is_never_perfect_square_l2178_217826


namespace NUMINAMATH_GPT_change_in_expression_l2178_217883

theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let original_expr := x^2 - 5 * x + 2
  let new_x := x + b
  let new_expr := (new_x)^2 - 5 * (new_x) + 2
  new_expr - original_expr = 2 * b * x + b^2 - 5 * b :=
by
  sorry

end NUMINAMATH_GPT_change_in_expression_l2178_217883


namespace NUMINAMATH_GPT_salary_increase_gt_90_percent_l2178_217844

theorem salary_increase_gt_90_percent (S : ℝ) : 
  (S * (1.12^6) - S) / S > 0.90 :=
by
  -- Here we skip the proof with sorry
  sorry

end NUMINAMATH_GPT_salary_increase_gt_90_percent_l2178_217844


namespace NUMINAMATH_GPT_real_solution_of_equation_l2178_217885

theorem real_solution_of_equation :
  ∀ x : ℝ, (x ≠ 5) → (x ≠ 3) →
  ((x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3)) 
  / ((x - 5) * (x - 3) * (x - 5)) = 1 ↔ x = 1 :=
by sorry

end NUMINAMATH_GPT_real_solution_of_equation_l2178_217885


namespace NUMINAMATH_GPT_symmetric_line_equation_l2178_217830

theorem symmetric_line_equation (x y : ℝ) :
  let line_original := x - 2 * y + 1 = 0
  let line_symmetry := x = 1
  let line_symmetric := x + 2 * y - 3 = 0
  ∀ (x y : ℝ), (2 - x - 2 * y + 1 = 0) ↔ (x + 2 * y - 3 = 0) := by
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l2178_217830


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2178_217877

theorem arithmetic_sequence_problem (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 6 = 36)
  (h2 : S n = 324)
  (h3 : S (n - 6) = 144) :
  n = 18 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2178_217877


namespace NUMINAMATH_GPT_quadrant_of_complex_number_l2178_217843

theorem quadrant_of_complex_number
  (h : ∀ x : ℝ, 0 < x → (a^2 + a + 2)/x < 1/x^2 + 1) :
  ∃ a : ℝ, -1 < a ∧ a < 0 ∧ i^27 = -i :=
sorry

end NUMINAMATH_GPT_quadrant_of_complex_number_l2178_217843


namespace NUMINAMATH_GPT_simple_sampling_methods_l2178_217808

theorem simple_sampling_methods :
  methods_of_implementing_simple_sampling = ["lottery method", "random number table method"] :=
sorry

end NUMINAMATH_GPT_simple_sampling_methods_l2178_217808


namespace NUMINAMATH_GPT_distribution_ways_l2178_217807

theorem distribution_ways :
  ∃ (n : ℕ) (erasers pencils notebooks pens : ℕ),
  pencils = 4 ∧ notebooks = 2 ∧ pens = 3 ∧ 
  n = 6 := sorry

end NUMINAMATH_GPT_distribution_ways_l2178_217807


namespace NUMINAMATH_GPT_find_p_tilde_one_l2178_217803

noncomputable def p (x : ℝ) : ℝ :=
  let r : ℝ := -1 / 9
  let s : ℝ := 1
  x^2 - (r + s) * x + (r * s)

theorem find_p_tilde_one : p 1 = 0 := by
  sorry

end NUMINAMATH_GPT_find_p_tilde_one_l2178_217803


namespace NUMINAMATH_GPT_integer_square_mod_4_l2178_217851

theorem integer_square_mod_4 (N : ℤ) : (N^2 % 4 = 0) ∨ (N^2 % 4 = 1) :=
by sorry

end NUMINAMATH_GPT_integer_square_mod_4_l2178_217851


namespace NUMINAMATH_GPT_decrease_neg_of_odd_and_decrease_nonneg_l2178_217840

-- Define the properties of the function f
variable (f : ℝ → ℝ)

-- f is odd
def odd_function : Prop := ∀ x : ℝ, f (-x) = - f x

-- f is decreasing on [0, +∞)
def decreasing_on_nonneg : Prop := ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2 → f x1 > f x2)

-- Goal: f is decreasing on (-∞, 0)
def decreasing_on_neg : Prop := ∀ x1 x2 : ℝ, (x1 < 0) → (x2 < 0) → (x1 < x2) → f x1 > f x2

-- The theorem to be proved
theorem decrease_neg_of_odd_and_decrease_nonneg 
  (h_odd : odd_function f) (h_decreasing_nonneg : decreasing_on_nonneg f) :
  decreasing_on_neg f :=
sorry

end NUMINAMATH_GPT_decrease_neg_of_odd_and_decrease_nonneg_l2178_217840


namespace NUMINAMATH_GPT_guinea_pigs_food_difference_l2178_217811

theorem guinea_pigs_food_difference :
  ∀ (first second third total : ℕ),
  first = 2 →
  second = first * 2 →
  total = 13 →
  first + second + third = total →
  third - second = 3 :=
by 
  intros first second third total h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_guinea_pigs_food_difference_l2178_217811


namespace NUMINAMATH_GPT_convex_quad_no_triangle_l2178_217888

/-- Given four angles of a convex quadrilateral, it is not always possible to choose any 
three of these angles so that they represent the lengths of the sides of some triangle. -/
theorem convex_quad_no_triangle (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) :
  ¬(∀ a b c : ℝ, a + b + c = 360 → (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end NUMINAMATH_GPT_convex_quad_no_triangle_l2178_217888


namespace NUMINAMATH_GPT_hotel_towels_l2178_217876

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end NUMINAMATH_GPT_hotel_towels_l2178_217876


namespace NUMINAMATH_GPT_base7_divisibility_rules_2_base7_divisibility_rules_3_l2178_217838

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end NUMINAMATH_GPT_base7_divisibility_rules_2_base7_divisibility_rules_3_l2178_217838


namespace NUMINAMATH_GPT_diagonal_length_of_quadrilateral_l2178_217841

theorem diagonal_length_of_quadrilateral 
  (area : ℝ) (m n : ℝ) (d : ℝ) 
  (h_area : area = 210) 
  (h_m : m = 9) 
  (h_n : n = 6) 
  (h_formula : area = 0.5 * d * (m + n)) : 
  d = 28 :=
by 
  sorry

end NUMINAMATH_GPT_diagonal_length_of_quadrilateral_l2178_217841


namespace NUMINAMATH_GPT_max_enclosed_area_perimeter_160_length_twice_width_l2178_217823

theorem max_enclosed_area_perimeter_160_length_twice_width 
  (W L : ℕ) 
  (h1 : 2 * (L + W) = 160) 
  (h2 : L = 2 * W) : 
  L * W = 1352 := 
sorry

end NUMINAMATH_GPT_max_enclosed_area_perimeter_160_length_twice_width_l2178_217823


namespace NUMINAMATH_GPT_problem1_problem2_l2178_217891

-- Definition for Problem 1
def cube_root_8 : ℝ := 2
def abs_neg5 : ℝ := 5
def pow_neg1_2023 : ℝ := -1

theorem problem1 : cube_root_8 + abs_neg5 + pow_neg1_2023 = 6 := by
  sorry

-- Definitions for Problem 2
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := 0, y := 1 }
def point2 : Point := { x := 2, y := 5 }

def linear_function (k b x : ℝ) : ℝ := k * x + b

def is_linear_function (p1 p2 : Point) (k b : ℝ) : Prop :=
  p1.y = linear_function k b p1.x ∧ p2.y = linear_function k b p2.x

theorem problem2 : ∃ k b : ℝ, is_linear_function point1 point2 k b ∧ (linear_function k b x = 2 * x + 1) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2178_217891


namespace NUMINAMATH_GPT_path_count_in_grid_l2178_217855

theorem path_count_in_grid :
  let grid_width := 6
  let grid_height := 5
  let total_steps := 8
  let right_steps := 5
  let up_steps := 3
  ∃ (C : Nat), C = Nat.choose total_steps up_steps ∧ C = 56 :=
by
  sorry

end NUMINAMATH_GPT_path_count_in_grid_l2178_217855


namespace NUMINAMATH_GPT_max_rocket_height_l2178_217870

-- Define the quadratic function representing the rocket's height
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

-- State the maximum height problem
theorem max_rocket_height : ∃ t : ℝ, rocket_height t = 175 ∧ ∀ t' : ℝ, rocket_height t' ≤ 175 :=
by
  use 2.5
  sorry -- The proof will show that the maximum height is 175 meters at time t = 2.5 seconds

end NUMINAMATH_GPT_max_rocket_height_l2178_217870


namespace NUMINAMATH_GPT_alternating_sum_of_coefficients_l2178_217829

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * x + 1)^5

theorem alternating_sum_of_coefficients :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), polynomial_expansion x = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h
  sorry

end NUMINAMATH_GPT_alternating_sum_of_coefficients_l2178_217829


namespace NUMINAMATH_GPT_sufficient_condition_for_lg_m_lt_1_l2178_217853

theorem sufficient_condition_for_lg_m_lt_1 (m : ℝ) (h1 : m ∈ ({1, 2} : Set ℝ)) : Real.log m < 1 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_lg_m_lt_1_l2178_217853


namespace NUMINAMATH_GPT_chocolates_bought_at_cost_price_l2178_217880

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end NUMINAMATH_GPT_chocolates_bought_at_cost_price_l2178_217880


namespace NUMINAMATH_GPT_value_of_a_minus_b_l2178_217835

theorem value_of_a_minus_b (a b : ℝ)
  (h1 : ∃ (x : ℝ), x = 3 ∧ (ax / (x - 1)) = 1)
  (h2 : ∀ (x : ℝ), (ax / (x - 1)) < 1 ↔ (x < b ∨ x > 3)) :
  a - b = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l2178_217835


namespace NUMINAMATH_GPT_original_total_thumbtacks_l2178_217847

-- Conditions
def num_cans : ℕ := 3
def num_boards_tested : ℕ := 120
def thumbtacks_per_board : ℕ := 3
def thumbtacks_remaining_per_can : ℕ := 30

-- Question
theorem original_total_thumbtacks :
  (num_cans * num_boards_tested * thumbtacks_per_board) + (num_cans * thumbtacks_remaining_per_can) = 450 :=
sorry

end NUMINAMATH_GPT_original_total_thumbtacks_l2178_217847


namespace NUMINAMATH_GPT_evaluate_x2_plus_y2_plus_z2_l2178_217815

theorem evaluate_x2_plus_y2_plus_z2 (x y z : ℤ) 
  (h1 : x^2 * y + y^2 * z + z^2 * x = 2186)
  (h2 : x * y^2 + y * z^2 + z * x^2 = 2188) 
  : x^2 + y^2 + z^2 = 245 := 
sorry

end NUMINAMATH_GPT_evaluate_x2_plus_y2_plus_z2_l2178_217815


namespace NUMINAMATH_GPT_train_speed_kmph_l2178_217864

noncomputable def speed_of_train
  (train_length : ℝ) (bridge_cross_time : ℝ) (total_length : ℝ) : ℝ :=
  (total_length / bridge_cross_time) * 3.6

theorem train_speed_kmph
  (train_length : ℝ := 130) 
  (bridge_cross_time : ℝ := 30) 
  (total_length : ℝ := 245) : 
  speed_of_train train_length bridge_cross_time total_length = 29.4 := by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l2178_217864


namespace NUMINAMATH_GPT_fraction_identity_l2178_217882

theorem fraction_identity (m n r t : ℚ) (h1 : m / n = 5 / 3) (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_identity_l2178_217882


namespace NUMINAMATH_GPT_long_side_length_l2178_217861

variable {a b d : ℝ}

theorem long_side_length (h1 : a / b = 2 * (b / d)) (h2 : a = 4) (hd : d = Real.sqrt (a^2 + b^2)) :
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
sorry

end NUMINAMATH_GPT_long_side_length_l2178_217861


namespace NUMINAMATH_GPT_find_d_l2178_217809

theorem find_d (d : ℝ) (h : ∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x = -d / 3 ∧ y = -d / 5 ∧ -d / 3 + (-d / 5) = 15) : d = -225 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_d_l2178_217809


namespace NUMINAMATH_GPT_slices_remaining_l2178_217801

theorem slices_remaining (large_pizza_slices : ℕ) (xl_pizza_slices : ℕ) (large_pizza_ordered : ℕ) (xl_pizza_ordered : ℕ) (mary_eats_large : ℕ) (mary_eats_xl : ℕ) :
  large_pizza_slices = 8 →
  xl_pizza_slices = 12 →
  large_pizza_ordered = 1 →
  xl_pizza_ordered = 1 →
  mary_eats_large = 7 →
  mary_eats_xl = 3 →
  (large_pizza_slices * large_pizza_ordered - mary_eats_large + xl_pizza_slices * xl_pizza_ordered - mary_eats_xl) = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_slices_remaining_l2178_217801


namespace NUMINAMATH_GPT_axis_of_symmetry_range_l2178_217818

theorem axis_of_symmetry_range (a : ℝ) : (-(a + 2) / (3 - 4 * a) > 0) ↔ (a < -2 ∨ a > 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_range_l2178_217818


namespace NUMINAMATH_GPT_baron_not_boasting_l2178_217845

-- Define a function to verify if a given list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- Define a list that represents the sequence given in the solution
def sequence_19 : List ℕ :=
  [9, 18, 7, 16, 5, 14, 3, 12, 1, 10, 11, 2, 13, 4, 15, 6, 17, 8, 19]

-- Prove that the sequence forms a palindrome
theorem baron_not_boasting : is_palindrome sequence_19 :=
by {
  -- Insert actual proof steps here
  sorry
}

end NUMINAMATH_GPT_baron_not_boasting_l2178_217845


namespace NUMINAMATH_GPT_oil_remaining_in_tank_l2178_217898

/- Definitions for the problem conditions -/
def tankCapacity : Nat := 32
def totalOilPurchased : Nat := 728

/- Theorem statement -/
theorem oil_remaining_in_tank : totalOilPurchased % tankCapacity = 24 := by
  sorry

end NUMINAMATH_GPT_oil_remaining_in_tank_l2178_217898


namespace NUMINAMATH_GPT_three_digit_number_possibilities_l2178_217805

theorem three_digit_number_possibilities (A B C : ℕ) (hA : A ≠ 0) (hC : C ≠ 0) (h_diff : A - C = 5) :
  ∃ (x : ℕ), x = 100 * A + 10 * B + C ∧ (x - (100 * C + 10 * B + A) = 495) ∧ ∃ n, n = 40 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_possibilities_l2178_217805


namespace NUMINAMATH_GPT_product_of_rational_solutions_eq_twelve_l2178_217884

theorem product_of_rational_solutions_eq_twelve :
  ∃ c1 c2 : ℕ, (c1 > 0) ∧ (c2 > 0) ∧ 
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c1 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c1 = d^2) ∧
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c2 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c2 = d^2) ∧
               c1 * c2 = 12 := sorry

end NUMINAMATH_GPT_product_of_rational_solutions_eq_twelve_l2178_217884


namespace NUMINAMATH_GPT_kiyiv_first_problem_kiyiv_second_problem_l2178_217873

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that x^3 + y^3 + 4xy ≥ x^2 + y^2 + x + y + 2. -/
theorem kiyiv_first_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  x^3 + y^3 + 4 * x * y ≥ x^2 + y^2 + x + y + 2 :=
sorry

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that 2(x^3 + y^3 + xy + x + y) ≥ 5(x^2 + y^2). -/
theorem kiyiv_second_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  2 * (x^3 + y^3 + x * y + x + y) ≥ 5 * (x^2 + y^2) :=
sorry

end NUMINAMATH_GPT_kiyiv_first_problem_kiyiv_second_problem_l2178_217873


namespace NUMINAMATH_GPT_closest_perfect_square_l2178_217810

theorem closest_perfect_square (n : ℕ) (h1 : n = 325) : 
    ∃ m : ℕ, m^2 = 324 ∧ 
    (∀ k : ℕ, (k^2 ≤ n ∨ k^2 ≥ n) → (k = 18 ∨ k^2 > 361 ∨ k^2 < 289)) := 
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_l2178_217810


namespace NUMINAMATH_GPT_impossible_to_all_minus_l2178_217854

def initial_grid : List (List Int) :=
  [[1, 1, -1, 1], 
   [-1, -1, 1, 1], 
   [1, 1, 1, 1], 
   [1, -1, 1, -1]]

-- Define the operation of flipping a row
def flip_row (grid : List (List Int)) (r : Nat) : List (List Int) :=
  grid.mapIdx (fun i row => if i == r then row.map (fun x => -x) else row)

-- Define the operation of flipping a column
def flip_col (grid : List (List Int)) (c : Nat) : List (List Int) :=
  grid.map (fun row => row.mapIdx (fun j x => if j == c then -x else x))

-- Predicate to check if all elements in the grid are -1
def all_minus (grid : List (List Int)) : Prop :=
  grid.all (fun row => row.all (fun x => x = -1))

-- The main theorem
theorem impossible_to_all_minus (init : List (List Int)) (hf1 : init = initial_grid) :
  ∀ grid, (grid = init ∨ ∃ r, grid = flip_row grid r ∨ ∃ c, grid = flip_col grid c) →
  ¬ all_minus grid := by
    sorry

end NUMINAMATH_GPT_impossible_to_all_minus_l2178_217854


namespace NUMINAMATH_GPT_probability_fx_lt_0_l2178_217837

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem probability_fx_lt_0 :
  (∫ x in -Real.pi..Real.pi, if f x < 0 then 1 else 0) / (2 * Real.pi) = 2 / Real.pi :=
by sorry

end NUMINAMATH_GPT_probability_fx_lt_0_l2178_217837


namespace NUMINAMATH_GPT_line_slope_l2178_217842

theorem line_slope (x1 y1 x2 y2 : ℝ) (h1 : x1 = 0) (h2 : y1 = 100) (h3 : x2 = 50) (h4 : y2 = 300) :
  (y2 - y1) / (x2 - x1) = 4 :=
by sorry

end NUMINAMATH_GPT_line_slope_l2178_217842


namespace NUMINAMATH_GPT_additional_houses_built_by_october_l2178_217817

def total_houses : ℕ := 2000
def fraction_built_first_half : ℚ := 3 / 5
def houses_needed_by_october : ℕ := 500

def houses_built_first_half : ℚ := fraction_built_first_half * total_houses
def houses_built_by_october : ℕ := total_houses - houses_needed_by_october

theorem additional_houses_built_by_october :
  (houses_built_by_october - houses_built_first_half) = 300 := by
  sorry

end NUMINAMATH_GPT_additional_houses_built_by_october_l2178_217817


namespace NUMINAMATH_GPT_binomial_term_is_constant_range_of_a_over_b_l2178_217879

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end NUMINAMATH_GPT_binomial_term_is_constant_range_of_a_over_b_l2178_217879


namespace NUMINAMATH_GPT_not_traversable_n_62_l2178_217865

theorem not_traversable_n_62 :
  ¬ (∃ (path : ℕ → ℕ), ∀ i < 62, path (i + 1) = (path i + 8) % 62 ∨ path (i + 1) = (path i + 9) % 62 ∨ path (i + 1) = (path i + 10) % 62) :=
by sorry

end NUMINAMATH_GPT_not_traversable_n_62_l2178_217865


namespace NUMINAMATH_GPT_jennifer_boxes_l2178_217822

theorem jennifer_boxes (kim_sold : ℕ) (h₁ : kim_sold = 54) (h₂ : ∃ jennifer_sold, jennifer_sold = kim_sold + 17) : ∃ jennifer_sold, jennifer_sold = 71 := by
  sorry

end NUMINAMATH_GPT_jennifer_boxes_l2178_217822


namespace NUMINAMATH_GPT_minimum_containers_needed_l2178_217814

-- Definition of the problem conditions
def container_sizes := [5, 10, 20]
def target_units := 85

-- Proposition stating the minimum number of containers required
theorem minimum_containers_needed : 
  ∃ (x y z : ℕ), 
    5 * x + 10 * y + 20 * z = target_units ∧ 
    x + y + z = 5 :=
sorry

end NUMINAMATH_GPT_minimum_containers_needed_l2178_217814


namespace NUMINAMATH_GPT_solve_firm_problem_l2178_217872

def firm_problem : Prop :=
  ∃ (P A : ℕ), 
    (P / A = 2 / 63) ∧ 
    (P / (A + 50) = 1 / 34) ∧ 
    (P = 20)

theorem solve_firm_problem : firm_problem :=
  sorry

end NUMINAMATH_GPT_solve_firm_problem_l2178_217872
