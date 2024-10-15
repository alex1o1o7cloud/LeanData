import Mathlib

namespace NUMINAMATH_GPT_correct_proposition_D_l30_3073

theorem correct_proposition_D (a b c : ℝ) (h : a > b) : a - c > b - c :=
by
  sorry

end NUMINAMATH_GPT_correct_proposition_D_l30_3073


namespace NUMINAMATH_GPT_least_positive_value_tan_inv_k_l30_3083

theorem least_positive_value_tan_inv_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) 
  : x = Real.arctan 1 := 
sorry

end NUMINAMATH_GPT_least_positive_value_tan_inv_k_l30_3083


namespace NUMINAMATH_GPT_sum_terms_a1_a17_l30_3012

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 := by
  sorry

end NUMINAMATH_GPT_sum_terms_a1_a17_l30_3012


namespace NUMINAMATH_GPT_supplements_delivered_l30_3093

-- Define the conditions as given in the problem
def total_medicine_boxes : ℕ := 760
def vitamin_boxes : ℕ := 472

-- Define the number of supplement boxes
def supplement_boxes : ℕ := total_medicine_boxes - vitamin_boxes

-- State the theorem to be proved
theorem supplements_delivered : supplement_boxes = 288 :=
by
  -- The actual proof is not required, so we use "sorry"
  sorry

end NUMINAMATH_GPT_supplements_delivered_l30_3093


namespace NUMINAMATH_GPT_number_of_people_going_on_trip_l30_3042

theorem number_of_people_going_on_trip
  (bags_per_person : ℕ)
  (weight_per_bag : ℕ)
  (total_luggage_capacity : ℕ)
  (additional_capacity : ℕ)
  (bags_per_additional_capacity : ℕ)
  (h1 : bags_per_person = 5)
  (h2 : weight_per_bag = 50)
  (h3 : total_luggage_capacity = 6000)
  (h4 : additional_capacity = 90) :
  (total_luggage_capacity + (bags_per_additional_capacity * weight_per_bag)) / (weight_per_bag * bags_per_person) = 42 := 
by
  simp [h1, h2, h3, h4]
  repeat { sorry }

end NUMINAMATH_GPT_number_of_people_going_on_trip_l30_3042


namespace NUMINAMATH_GPT_seashells_unbroken_l30_3018

theorem seashells_unbroken (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) 
  (h3 : unbroken_seashells = total_seashells - broken_seashells) :
  unbroken_seashells = 2 :=
by
  sorry

end NUMINAMATH_GPT_seashells_unbroken_l30_3018


namespace NUMINAMATH_GPT_angles_on_x_axis_l30_3002

theorem angles_on_x_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi) ∨ (∃ k : ℤ, α = (2 * k + 1) * Real.pi) ↔ 
  ∃ k : ℤ, α = k * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_angles_on_x_axis_l30_3002


namespace NUMINAMATH_GPT_math_problem_l30_3032

theorem math_problem 
  (X : ℝ)
  (num1 : ℝ := 1 + 28/63)
  (num2 : ℝ := 8 + 7/16)
  (frac_sub1 : ℝ := 19/24 - 21/40)
  (frac_sub2 : ℝ := 1 + 28/63 - 17/21)
  (denom_calc : ℝ := 0.675 * 2.4 - 0.02) :
  0.125 * X / (frac_sub1 * num2) = (frac_sub2 * 0.7) / denom_calc → X = 5 := 
sorry

end NUMINAMATH_GPT_math_problem_l30_3032


namespace NUMINAMATH_GPT_expression_eq_l30_3033

theorem expression_eq (x : ℝ) : 
    (x + 1)^4 + 4 * (x + 1)^3 + 6 * (x + 1)^2 + 4 * (x + 1) + 1 = (x + 2)^4 := 
  sorry

end NUMINAMATH_GPT_expression_eq_l30_3033


namespace NUMINAMATH_GPT_max_distance_AB_l30_3066

-- Define curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the problem to prove the maximum value of distance AB is 8
theorem max_distance_AB :
  ∀ (Ax Ay Bx By : ℝ),
    C1 Ax Ay →
    C2 Bx By →
    dist (Ax, Ay) (Bx, By) ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_distance_AB_l30_3066


namespace NUMINAMATH_GPT_distance_from_P_to_focus_l30_3057

-- Define the parabola equation and the definition of the point P
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the given condition that P's distance to the x-axis is 12
def point_P (x y : ℝ) : Prop := parabola x y ∧ |y| = 12

-- The Lean proof problem statement
theorem distance_from_P_to_focus :
  ∃ (x y : ℝ), point_P x y → dist (x, y) (4, 0) = 13 :=
by {
  sorry   -- proof to be completed
}

end NUMINAMATH_GPT_distance_from_P_to_focus_l30_3057


namespace NUMINAMATH_GPT_danny_bottle_cap_count_l30_3068

theorem danny_bottle_cap_count 
  (initial_caps : Int) 
  (found_caps : Int) 
  (final_caps : Int) 
  (h1 : initial_caps = 6) 
  (h2 : found_caps = 22) 
  (h3 : final_caps = initial_caps + found_caps) : 
  final_caps = 28 :=
by
  sorry

end NUMINAMATH_GPT_danny_bottle_cap_count_l30_3068


namespace NUMINAMATH_GPT_average_age_l30_3037

theorem average_age (women men : ℕ) (avg_age_women avg_age_men : ℝ) 
  (h_women : women = 12) 
  (h_men : men = 18) 
  (h_avg_women : avg_age_women = 28) 
  (h_avg_men : avg_age_men = 40) : 
  (12 * 28 + 18 * 40) / (12 + 18) = 35.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_age_l30_3037


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l30_3007

def lies_in_fourth_quadrant (P : ℤ × ℤ) : Prop :=
  P.fst > 0 ∧ P.snd < 0

theorem point_in_fourth_quadrant : lies_in_fourth_quadrant (2023, -2024) :=
by
  -- Here is where the proof steps would go
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l30_3007


namespace NUMINAMATH_GPT_cargo_total_ship_l30_3017

-- Define the initial cargo and the additional cargo loaded
def initial_cargo := 5973
def additional_cargo := 8723

-- Define the total cargo the ship holds after loading additional cargo
def total_cargo := initial_cargo + additional_cargo

-- Statement of the problem
theorem cargo_total_ship (h1 : initial_cargo = 5973) (h2 : additional_cargo = 8723) : 
  total_cargo = 14696 := 
by
  sorry

end NUMINAMATH_GPT_cargo_total_ship_l30_3017


namespace NUMINAMATH_GPT_cakes_count_l30_3091

theorem cakes_count (x y : ℕ) 
  (price_fruit price_chocolate total_cost : ℝ) 
  (avg_price : ℝ) 
  (H1 : price_fruit = 4.8)
  (H2 : price_chocolate = 6.6)
  (H3 : total_cost = 167.4)
  (H4 : avg_price = 6.2)
  (H5 : price_fruit * x + price_chocolate * y = total_cost)
  (H6 : total_cost / (x + y) = avg_price) : 
  x = 6 ∧ y = 21 := 
by
  sorry

end NUMINAMATH_GPT_cakes_count_l30_3091


namespace NUMINAMATH_GPT_number_to_multiply_l30_3069

theorem number_to_multiply (a b x : ℝ) (h1 : x * a = 4 * b) (h2 : a * b ≠ 0) (h3 : a / 4 = b / 3) : x = 3 :=
sorry

end NUMINAMATH_GPT_number_to_multiply_l30_3069


namespace NUMINAMATH_GPT_marks_of_A_l30_3097

variable (a b c d e : ℕ)

theorem marks_of_A:
  (a + b + c = 144) →
  (a + b + c + d = 188) →
  (e = d + 3) →
  (b + c + d + e = 192) →
  a = 43 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_marks_of_A_l30_3097


namespace NUMINAMATH_GPT_third_smallest_number_l30_3047

/-- 
  The third smallest two-decimal-digit number that can be made
  using the digits 3, 8, 2, and 7 each exactly once is 27.38.
-/
theorem third_smallest_number (digits : List ℕ) (h : digits = [3, 8, 2, 7]) : 
  ∃ x y, 
  x < y ∧
  x = 23.78 ∧
  y = 23.87 ∧
  ∀ z, z > x ∧ z < y → z = 27.38 :=
by 
  sorry

end NUMINAMATH_GPT_third_smallest_number_l30_3047


namespace NUMINAMATH_GPT_opposite_of_2023_l30_3020

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l30_3020


namespace NUMINAMATH_GPT_Question_D_condition_l30_3022

theorem Question_D_condition (P Q : Prop) (h : P → Q) : ¬ Q → ¬ P :=
by sorry

end NUMINAMATH_GPT_Question_D_condition_l30_3022


namespace NUMINAMATH_GPT_b6_b8_equals_16_l30_3054

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_arithmetic : ∃ d, ∀ n, a_seq (n + 1) = a_seq n + d
axiom b_geometric : ∃ r, ∀ n, b_seq (n + 1) = b_seq n * r
axiom a_nonzero : ∀ n, a_seq n ≠ 0
axiom a_eq : 2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0
axiom b7_eq_a7 : b_seq 7 = a_seq 7

theorem b6_b8_equals_16 : b_seq 6 * b_seq 8 = 16 := by
  sorry

end NUMINAMATH_GPT_b6_b8_equals_16_l30_3054


namespace NUMINAMATH_GPT_cone_dimensions_l30_3075

noncomputable def cone_height (r_sector : ℝ) (r_cone_base : ℝ) : ℝ :=
  Real.sqrt (r_sector^2 - r_cone_base^2)

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * radius^2 * height

theorem cone_dimensions 
  (r_circle : ℝ) (num_sectors : ℕ) (r_cone_base : ℝ) :
  r_circle = 12 → num_sectors = 4 → r_cone_base = 3 → 
  cone_height r_circle r_cone_base = 3 * Real.sqrt 15 ∧ 
  cone_volume r_cone_base (cone_height r_circle r_cone_base) = 9 * Real.pi * Real.sqrt 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cone_dimensions_l30_3075


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l30_3024

theorem quadratic_inequality_solution
  (x : ℝ) :
  -2 * x^2 + x < -3 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l30_3024


namespace NUMINAMATH_GPT_new_train_travel_distance_l30_3001

-- Definitions of the trains' travel distances
def older_train_distance : ℝ := 180
def new_train_additional_distance_ratio : ℝ := 0.50

-- Proof that the new train can travel 270 miles
theorem new_train_travel_distance
: new_train_additional_distance_ratio * older_train_distance + older_train_distance = 270 := 
by
  sorry

end NUMINAMATH_GPT_new_train_travel_distance_l30_3001


namespace NUMINAMATH_GPT_prime_gt_p_l30_3078

theorem prime_gt_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hgt : q > 5) (hdiv : q ∣ 2^p + 3^p) : q > p := 
sorry

end NUMINAMATH_GPT_prime_gt_p_l30_3078


namespace NUMINAMATH_GPT_cube_faces_sum_eq_neg_3_l30_3095

theorem cube_faces_sum_eq_neg_3 
    (a b c d e f : ℤ)
    (h1 : a = -3)
    (h2 : b = a + 1)
    (h3 : c = b + 1)
    (h4 : d = c + 1)
    (h5 : e = d + 1)
    (h6 : f = e + 1)
    (h7 : a + f = b + e)
    (h8 : b + e = c + d) :
  a + b + c + d + e + f = -3 := sorry

end NUMINAMATH_GPT_cube_faces_sum_eq_neg_3_l30_3095


namespace NUMINAMATH_GPT_fred_carrots_l30_3045

-- Define the conditions
def sally_carrots : Nat := 6
def total_carrots : Nat := 10

-- Define the problem question and the proof statement
theorem fred_carrots : ∃ fred_carrots : Nat, fred_carrots = total_carrots - sally_carrots := 
by
  sorry

end NUMINAMATH_GPT_fred_carrots_l30_3045


namespace NUMINAMATH_GPT_apple_slices_count_l30_3081

theorem apple_slices_count :
  let boxes := 7
  let apples_per_box := 7
  let slices_per_apple := 8
  let total_apples := boxes * apples_per_box
  let total_slices := total_apples * slices_per_apple
  total_slices = 392 :=
by
  sorry

end NUMINAMATH_GPT_apple_slices_count_l30_3081


namespace NUMINAMATH_GPT_largest_divisor_of_product_l30_3049

theorem largest_divisor_of_product (n : ℕ) (h : n % 3 = 0) : ∃ d, d = 288 ∧ ∀ n (h : n % 3 = 0), d ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_product_l30_3049


namespace NUMINAMATH_GPT_number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l30_3051

-- Define the conditions
def total_matches := 16
def played_matches := 9
def lost_matches := 2
def current_points := 19
def max_points_per_win := 3
def draw_points := 1
def remaining_matches := total_matches - played_matches
def required_points := 34

-- Statements to prove
theorem number_of_wins_in_first_9_matches :
  ∃ wins_in_first_9, 3 * wins_in_first_9 + draw_points * (played_matches - lost_matches - wins_in_first_9) = current_points :=
sorry

theorem highest_possible_points :
  current_points + remaining_matches * max_points_per_win = 40 :=
sorry

theorem minimum_wins_in_remaining_matches :
  ∃ min_wins_in_remaining_7, (min_wins_in_remaining_7 = 4 ∧ 3 * min_wins_in_remaining_7 + current_points + (remaining_matches - min_wins_in_remaining_7) * draw_points ≥ required_points) :=
sorry

end NUMINAMATH_GPT_number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l30_3051


namespace NUMINAMATH_GPT_correct_answer_l30_3010

theorem correct_answer (a b c : ℤ) 
  (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by 
  sorry

end NUMINAMATH_GPT_correct_answer_l30_3010


namespace NUMINAMATH_GPT_find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l30_3005

theorem find_counterfeit_80_coins_in_4_weighings :
  ∃ f : Fin 80 → Bool, (∃ i, f i = true) ∧ (∃ i j, f i ≠ f j) := sorry

theorem min_weighings_for_n_coins (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, 3^(k-1) < n ∧ n ≤ 3^k := sorry

end NUMINAMATH_GPT_find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l30_3005


namespace NUMINAMATH_GPT_more_roses_than_orchids_l30_3023

theorem more_roses_than_orchids (roses orchids : ℕ) (h1 : roses = 12) (h2 : orchids = 2) : roses - orchids = 10 := by
  sorry

end NUMINAMATH_GPT_more_roses_than_orchids_l30_3023


namespace NUMINAMATH_GPT_height_relationship_l30_3015

theorem height_relationship 
  (r₁ h₁ r₂ h₂ : ℝ)
  (h_volume : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius : r₂ = (6/5) * r₁) :
  h₁ = 1.44 * h₂ :=
by
  sorry

end NUMINAMATH_GPT_height_relationship_l30_3015


namespace NUMINAMATH_GPT_eggs_in_each_group_l30_3088

theorem eggs_in_each_group (eggs marbles groups : ℕ) 
  (h_eggs : eggs = 15)
  (h_groups : groups = 3) 
  (h_marbles : marbles = 4) :
  eggs / groups = 5 :=
by sorry

end NUMINAMATH_GPT_eggs_in_each_group_l30_3088


namespace NUMINAMATH_GPT_bob_ears_left_l30_3028

namespace CornProblem

-- Definitions of the given conditions
def initial_bob_bushels : ℕ := 120
def ears_per_bushel : ℕ := 15

def given_away_bushels_terry : ℕ := 15
def given_away_bushels_jerry : ℕ := 8
def given_away_bushels_linda : ℕ := 25
def given_away_ears_stacy : ℕ := 42
def given_away_bushels_susan : ℕ := 9
def given_away_bushels_tim : ℕ := 4
def given_away_ears_tim : ℕ := 18

-- Calculate initial ears of corn
noncomputable def initial_ears_of_corn : ℕ := initial_bob_bushels * ears_per_bushel

-- Calculate total ears given away in bushels
def total_ears_given_away_bushels : ℕ :=
  (given_away_bushels_terry + given_away_bushels_jerry + given_away_bushels_linda +
   given_away_bushels_susan + given_away_bushels_tim) * ears_per_bushel

-- Calculate total ears directly given away
def total_ears_given_away_direct : ℕ :=
  given_away_ears_stacy + given_away_ears_tim

-- Calculate total ears given away
def total_ears_given_away : ℕ :=
  total_ears_given_away_bushels + total_ears_given_away_direct

-- Calculate ears of corn Bob has left
noncomputable def ears_left : ℕ :=
  initial_ears_of_corn - total_ears_given_away

-- The proof statement
theorem bob_ears_left : ears_left = 825 := by
  sorry

end CornProblem

end NUMINAMATH_GPT_bob_ears_left_l30_3028


namespace NUMINAMATH_GPT_sum_of_twos_and_threes_3024_l30_3058

theorem sum_of_twos_and_threes_3024 : ∃ n : ℕ, n = 337 ∧ (∃ (a b : ℕ), 3024 = 2 * a + 3 * b) :=
sorry

end NUMINAMATH_GPT_sum_of_twos_and_threes_3024_l30_3058


namespace NUMINAMATH_GPT_number_of_teams_l30_3060

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l30_3060


namespace NUMINAMATH_GPT_solve_math_problem_l30_3070

-- Math problem definition
def math_problem (A : ℝ) : Prop :=
  (0 < A ∧ A < (Real.pi / 2)) ∧ (Real.cos A = 3 / 5) →
  Real.sin (2 * A) = 24 / 25

-- Example theorem statement in Lean
theorem solve_math_problem (A : ℝ) : math_problem A :=
sorry

end NUMINAMATH_GPT_solve_math_problem_l30_3070


namespace NUMINAMATH_GPT_right_triangle_condition_l30_3043

theorem right_triangle_condition (a d : ℝ) (h : d > 0) : 
  (a = d * (1 + Real.sqrt 7)) ↔ (a^2 + (a + 2 * d)^2 = (a + 4 * d)^2) := 
sorry

end NUMINAMATH_GPT_right_triangle_condition_l30_3043


namespace NUMINAMATH_GPT_predict_height_at_age_10_l30_3026

def regression_line := fun (x : ℝ) => 7.19 * x + 73.93

theorem predict_height_at_age_10 :
  regression_line 10 = 145.83 :=
by
  sorry

end NUMINAMATH_GPT_predict_height_at_age_10_l30_3026


namespace NUMINAMATH_GPT_rental_cost_equation_l30_3086

theorem rental_cost_equation (x : ℕ) (h : x > 0) :
  180 / x - 180 / (x + 2) = 3 :=
sorry

end NUMINAMATH_GPT_rental_cost_equation_l30_3086


namespace NUMINAMATH_GPT_evaluate_expression_l30_3059

theorem evaluate_expression :
  ( ( ( 5 / 2 : ℚ ) / ( 7 / 12 : ℚ ) ) - ( 4 / 9 : ℚ ) ) = ( 242 / 63 : ℚ ) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l30_3059


namespace NUMINAMATH_GPT_three_g_two_plus_two_g_neg_four_l30_3096

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x + 11

theorem three_g_two_plus_two_g_neg_four : 3 * g 2 + 2 * g (-4) = 147 := by
  sorry

end NUMINAMATH_GPT_three_g_two_plus_two_g_neg_four_l30_3096


namespace NUMINAMATH_GPT_golden_section_AC_correct_l30_3025

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def segment_length := 20
noncomputable def golden_section_point (AB AC BC : ℝ) (h1 : AB = AC + BC) (h2 : AC > BC) (h3 : AB = segment_length) : Prop :=
  AC = (Real.sqrt 5 - 1) / 2 * AB

theorem golden_section_AC_correct :
  ∃ (AC BC : ℝ), (AC + BC = segment_length) ∧ (AC > BC) ∧ (AC = 10 * (Real.sqrt 5 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_golden_section_AC_correct_l30_3025


namespace NUMINAMATH_GPT_star_test_one_star_test_two_l30_3016

def star (x y : ℤ) : ℤ :=
  if x = 0 then Int.natAbs y
  else if y = 0 then Int.natAbs x
  else if (x < 0) = (y < 0) then Int.natAbs x + Int.natAbs y
  else -(Int.natAbs x + Int.natAbs y)

theorem star_test_one :
  star 11 (star 0 (-12)) = 23 :=
by
  sorry

theorem star_test_two (a : ℤ) :
  2 * (2 * star 1 a) - 1 = 3 * a ↔ a = 3 ∨ a = -5 :=
by
  sorry

end NUMINAMATH_GPT_star_test_one_star_test_two_l30_3016


namespace NUMINAMATH_GPT_park_is_square_l30_3000

-- Defining the concept of a square field
def square_field : ℕ := 4

-- Given condition: The sum of the right angles from the park and the square field
axiom angles_sum (park_angles : ℕ) : park_angles + square_field = 8

-- The theorem to be proven
theorem park_is_square (park_angles : ℕ) (h : park_angles + square_field = 8) : park_angles = 4 :=
by sorry

end NUMINAMATH_GPT_park_is_square_l30_3000


namespace NUMINAMATH_GPT_probability_of_intersecting_diagonals_l30_3092

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end NUMINAMATH_GPT_probability_of_intersecting_diagonals_l30_3092


namespace NUMINAMATH_GPT_multiplication_value_l30_3035

theorem multiplication_value (x : ℝ) (h : (2.25 / 3) * x = 9) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_value_l30_3035


namespace NUMINAMATH_GPT_simplified_value_of_sum_l30_3014

theorem simplified_value_of_sum :
  (-1)^(2004) + (-1)^(2005) + 1^(2006) - 1^(2007) = -2 := by
  sorry

end NUMINAMATH_GPT_simplified_value_of_sum_l30_3014


namespace NUMINAMATH_GPT_initial_speed_solution_l30_3036

def initial_speed_problem : Prop :=
  ∃ V : ℝ, 
    (∀ t t_new : ℝ, 
      t = 300 / V ∧ 
      t_new = t - 4 / 5 ∧ 
      (∀ d d_remaining : ℝ, 
        d = V * (5 / 4) ∧ 
        d_remaining = 300 - d ∧ 
        t_new = (5 / 4) + d_remaining / (V + 16)) 
    ) → 
    V = 60

theorem initial_speed_solution : initial_speed_problem :=
by
  unfold initial_speed_problem
  sorry

end NUMINAMATH_GPT_initial_speed_solution_l30_3036


namespace NUMINAMATH_GPT_HorseKeepsPower_l30_3027

/-- If the Little Humpbacked Horse does not eat for seven days or does not sleep for seven days,
    he will lose his magic power. Suppose he did not eat or sleep for a whole week. 
    Prove that by the end of the seventh day, he must do the activity he did not do right before 
    the start of the first period of seven days in order to keep his power. -/
theorem HorseKeepsPower (eat sleep : ℕ → Prop) :
  (∀ (n : ℕ), (n ≥ 7 → ¬eat n) ∨ (n ≥ 7 → ¬sleep n)) →
  (∀ (n : ℕ), n < 7 → (¬eat n ∧ ¬sleep n)) →
  ∃ (t : ℕ), t > 7 → (eat t ∨ sleep t) :=
sorry

end NUMINAMATH_GPT_HorseKeepsPower_l30_3027


namespace NUMINAMATH_GPT_krishan_money_l30_3080

theorem krishan_money (R G K : ℕ) 
  (h_ratio1 : R * 17 = G * 7) 
  (h_ratio2 : G * 17 = K * 7) 
  (h_R : R = 735) : 
  K = 4335 := 
sorry

end NUMINAMATH_GPT_krishan_money_l30_3080


namespace NUMINAMATH_GPT_simplify_expression_l30_3029

theorem simplify_expression :
  18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l30_3029


namespace NUMINAMATH_GPT_geometric_sequence_a9_l30_3006

theorem geometric_sequence_a9 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 2) 
  (h2 : a 4 = 8 * a 7) 
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (hq : q > 0) 
  : a 9 = 1 / 32 := 
by sorry

end NUMINAMATH_GPT_geometric_sequence_a9_l30_3006


namespace NUMINAMATH_GPT_sri_lanka_population_problem_l30_3064

theorem sri_lanka_population_problem
  (P : ℝ)
  (h1 : 0.85 * (0.9 * P) = 3213) :
  P = 4200 :=
sorry

end NUMINAMATH_GPT_sri_lanka_population_problem_l30_3064


namespace NUMINAMATH_GPT_walking_rate_on_escalator_l30_3040

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 160)
  (time_taken : ℝ := 8)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) :
  v = 8 :=
by
  sorry

end NUMINAMATH_GPT_walking_rate_on_escalator_l30_3040


namespace NUMINAMATH_GPT_ralph_squares_count_l30_3055

def total_matchsticks := 50
def elvis_square_sticks := 4
def ralph_square_sticks := 8
def elvis_squares := 5
def leftover_sticks := 6

theorem ralph_squares_count : 
  ∃ R : ℕ, 
  (elvis_squares * elvis_square_sticks) + (R * ralph_square_sticks) + leftover_sticks = total_matchsticks ∧ R = 3 :=
by 
  sorry

end NUMINAMATH_GPT_ralph_squares_count_l30_3055


namespace NUMINAMATH_GPT_k_greater_than_inv_e_l30_3077

theorem k_greater_than_inv_e (k : ℝ) (x : ℝ) (hx_pos : 0 < x) (hcond : k * (Real.exp (k * x) + 1) - (1 + (1 / x)) * Real.log x > 0) : 
  k > 1 / Real.exp 1 :=
sorry

end NUMINAMATH_GPT_k_greater_than_inv_e_l30_3077


namespace NUMINAMATH_GPT_find_f1_find_f8_inequality_l30_3038

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f x
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_multiplicative : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y
axiom f_of_2 : f 2 = 4

-- Statements to prove
theorem find_f1 : f 1 = 1 := sorry
theorem find_f8 : f 8 = 64 := sorry
theorem inequality : ∀ x : ℝ, 3 < x → x ≤ 7 / 2 → 16 * f (1 / (x - 3)) ≥ f (2 * x + 1) := sorry

end NUMINAMATH_GPT_find_f1_find_f8_inequality_l30_3038


namespace NUMINAMATH_GPT_find_rate_of_interest_l30_3087

noncomputable def rate_of_interest (P : ℝ) (r : ℝ) : Prop :=
  let CI2 := P * (1 + r)^2 - P
  let CI3 := P * (1 + r)^3 - P
  CI2 = 1200 ∧ CI3 = 1272 → r = 0.06

theorem find_rate_of_interest (P : ℝ) (r : ℝ) : rate_of_interest P r :=
by sorry

end NUMINAMATH_GPT_find_rate_of_interest_l30_3087


namespace NUMINAMATH_GPT_problem1_problem2_l30_3013

-- Definition for the first problem: determine the number of arrangements when no box is empty and ball 3 is in box B
def arrangements_with_ball3_in_B_and_no_empty_box : ℕ :=
  12

theorem problem1 : arrangements_with_ball3_in_B_and_no_empty_box = 12 :=
  by
    sorry

-- Definition for the second problem: determine the number of arrangements when ball 1 is not in box A and ball 2 is not in box B
def arrangements_with_ball1_not_in_A_and_ball2_not_in_B : ℕ :=
  36

theorem problem2 : arrangements_with_ball1_not_in_A_and_ball2_not_in_B = 36 :=
  by
    sorry

end NUMINAMATH_GPT_problem1_problem2_l30_3013


namespace NUMINAMATH_GPT_solve_missing_figure_l30_3098

theorem solve_missing_figure (x : ℝ) (h : 0.25/100 * x = 0.04) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_missing_figure_l30_3098


namespace NUMINAMATH_GPT_ceil_floor_arith_l30_3090

theorem ceil_floor_arith :
  (Int.ceil (((15: ℚ) / 8)^2 * (-34 / 4)) - Int.floor ((15 / 8) * Int.floor (-34 / 4))) = -12 :=
by sorry

end NUMINAMATH_GPT_ceil_floor_arith_l30_3090


namespace NUMINAMATH_GPT_tripodasaurus_flock_l30_3099

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end NUMINAMATH_GPT_tripodasaurus_flock_l30_3099


namespace NUMINAMATH_GPT_sequence_formula_general_formula_l30_3019

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def S_n (n : ℕ) : ℕ :=
  n^2 + 3 * n + 1

theorem sequence_formula :
  ∀ n, a_n n =
    if n = 1 then 5 else 2 * n + 2 := by
  sorry

theorem general_formula (n : ℕ) :
  a_n n =
    if n = 1 then S_n 1 else S_n n - S_n (n - 1) := by
  sorry

end NUMINAMATH_GPT_sequence_formula_general_formula_l30_3019


namespace NUMINAMATH_GPT_real_solutions_in_interval_l30_3062

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ 
  (x ≠ -1) ∧
  (x^2 / (x + 1 - Real.sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2)
  
theorem real_solutions_in_interval (x : ℝ) (h : problem_statement x) : -1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_GPT_real_solutions_in_interval_l30_3062


namespace NUMINAMATH_GPT_distance_covered_l30_3046

noncomputable def boat_speed_still_water : ℝ := 6.5
noncomputable def current_speed : ℝ := 2.5
noncomputable def time_taken : ℝ := 35.99712023038157

noncomputable def effective_speed_downstream (boat_speed_still_water current_speed : ℝ) : ℝ :=
  boat_speed_still_water + current_speed

noncomputable def convert_kmph_to_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def calculate_distance (speed_in_mps time_in_seconds : ℝ) : ℝ :=
  speed_in_mps * time_in_seconds

theorem distance_covered :
  calculate_distance (convert_kmph_to_mps (effective_speed_downstream boat_speed_still_water current_speed)) time_taken = 89.99280057595392 :=
by
  sorry

end NUMINAMATH_GPT_distance_covered_l30_3046


namespace NUMINAMATH_GPT_cost_of_notebooks_and_markers_l30_3084

theorem cost_of_notebooks_and_markers 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30) 
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_notebooks_and_markers_l30_3084


namespace NUMINAMATH_GPT_misha_needs_total_l30_3048

theorem misha_needs_total (
  current_amount : ℤ := 34
) (additional_amount : ℤ := 13) : 
  current_amount + additional_amount = 47 :=
by
  sorry

end NUMINAMATH_GPT_misha_needs_total_l30_3048


namespace NUMINAMATH_GPT_equal_roots_of_quadratic_eq_l30_3011

theorem equal_roots_of_quadratic_eq (n : ℝ) : (∃ x : ℝ, (x^2 - x + n = 0) ∧ (Δ = 0)) ↔ n = 1 / 4 :=
by
  have h₁ : Δ = 0 := by sorry  -- The discriminant condition
  sorry  -- Placeholder for completing the theorem proof

end NUMINAMATH_GPT_equal_roots_of_quadratic_eq_l30_3011


namespace NUMINAMATH_GPT_total_marbles_l30_3053

variables (y : ℝ) 

def first_friend_marbles : ℝ := 2 * y + 2
def second_friend_marbles : ℝ := y
def third_friend_marbles : ℝ := 3 * y - 1

theorem total_marbles :
  (first_friend_marbles y) + (second_friend_marbles y) + (third_friend_marbles y) = 6 * y + 1 :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l30_3053


namespace NUMINAMATH_GPT_number_of_kiwis_l30_3008

/-
There are 500 pieces of fruit in a crate. One fourth of the fruits are apples,
20% are oranges, one fifth are strawberries, and the rest are kiwis.
Prove that the number of kiwis is 175.
-/

theorem number_of_kiwis (total_fruits apples oranges strawberries kiwis : ℕ)
  (h1 : total_fruits = 500)
  (h2 : apples = total_fruits / 4)
  (h3 : oranges = 20 * total_fruits / 100)
  (h4 : strawberries = total_fruits / 5)
  (h5 : kiwis = total_fruits - (apples + oranges + strawberries)) :
  kiwis = 175 :=
sorry

end NUMINAMATH_GPT_number_of_kiwis_l30_3008


namespace NUMINAMATH_GPT_quadratic_root_solution_l30_3009

theorem quadratic_root_solution (k : ℤ) (a : ℤ) :
  (∀ x, x^2 + k * x - 10 = 0 → x = 2 ∨ x = a) →
  2 + a = -k →
  2 * a = -10 →
  k = 3 ∧ a = -5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_solution_l30_3009


namespace NUMINAMATH_GPT_find_vector_l30_3031

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 3 + 2 * t)

noncomputable def line_m (s : ℝ) : ℝ × ℝ :=
  (-4 + 3 * s, 5 + 2 * s)

def vector_condition (v1 v2 : ℝ) : Prop :=
  v1 - v2 = 1

theorem find_vector :
  ∃ (v1 v2 : ℝ), vector_condition v1 v2 ∧ (v1, v2) = (3, 2) :=
sorry

end NUMINAMATH_GPT_find_vector_l30_3031


namespace NUMINAMATH_GPT_total_earnings_l30_3079

def num_members : ℕ := 20
def candy_bars_per_member : ℕ := 8
def cost_per_candy_bar : ℝ := 0.5

theorem total_earnings :
  (num_members * candy_bars_per_member * cost_per_candy_bar) = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_l30_3079


namespace NUMINAMATH_GPT_probability_not_greater_than_two_l30_3074

theorem probability_not_greater_than_two : 
  let cards := [1, 2, 3, 4]
  let favorable_cards := [1, 2]
  let total_scenarios := cards.length
  let favorable_scenarios := favorable_cards.length
  let prob := favorable_scenarios / total_scenarios
  prob = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_greater_than_two_l30_3074


namespace NUMINAMATH_GPT_second_increase_is_40_l30_3003

variable (P : ℝ) (x : ℝ)

def second_increase (P : ℝ) (x : ℝ) : Prop :=
  1.30 * P * (1 + x / 100) = 1.82 * P

theorem second_increase_is_40 (P : ℝ) : ∃ x, second_increase P x ∧ x = 40 := by
  use 40
  sorry

end NUMINAMATH_GPT_second_increase_is_40_l30_3003


namespace NUMINAMATH_GPT_election_total_votes_l30_3039

theorem election_total_votes (V_A V_B V : ℕ) (H1 : V_A = V_B + 15/100 * V) (H2 : V_A + V_B = 80/100 * V) (H3 : V_B = 2184) : V = 6720 :=
sorry

end NUMINAMATH_GPT_election_total_votes_l30_3039


namespace NUMINAMATH_GPT_walkway_time_l30_3082

theorem walkway_time {v_p v_w : ℝ} 
  (cond1 : 60 = (v_p + v_w) * 30) 
  (cond2 : 60 = (v_p - v_w) * 120) 
  : 60 / v_p = 48 := 
by
  sorry

end NUMINAMATH_GPT_walkway_time_l30_3082


namespace NUMINAMATH_GPT_point_P_coordinates_l30_3004

noncomputable def P_coordinates (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 4 * Real.sin θ)

theorem point_P_coordinates : 
  ∀ θ, (0 ≤ θ ∧ θ ≤ Real.pi ∧ 1 = (4 / 3) * Real.tan θ) →
  P_coordinates θ = (12 / 5, 12 / 5) :=
by
  intro θ h
  sorry

end NUMINAMATH_GPT_point_P_coordinates_l30_3004


namespace NUMINAMATH_GPT_set_intersection_l30_3094

noncomputable def SetA : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
noncomputable def SetB : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem set_intersection :
  SetA ∩ SetB = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_set_intersection_l30_3094


namespace NUMINAMATH_GPT_sum_digits_probability_l30_3065

noncomputable def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def numInRange : ℕ := 1000000

noncomputable def coefficient : ℕ :=
  Nat.choose 24 5 - 6 * Nat.choose 14 5

noncomputable def probability : ℚ :=
  coefficient / numInRange

theorem sum_digits_probability :
  probability = 7623 / 250000 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_probability_l30_3065


namespace NUMINAMATH_GPT_infinite_bad_numbers_l30_3061

-- Define types for natural numbers
variables {a b : ℕ}

-- The theorem statement
theorem infinite_bad_numbers (a b : ℕ) : ∃ᶠ (n : ℕ) in at_top, n > 0 ∧ ¬ (n^b + 1 ∣ a^n + 1) :=
sorry

end NUMINAMATH_GPT_infinite_bad_numbers_l30_3061


namespace NUMINAMATH_GPT_min_value_ineq_least_3_l30_3072

noncomputable def min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : ℝ :=
  1 / (x + y) + (x + y) / z

theorem min_value_ineq_least_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  min_value_ineq x y z h1 h2 h3 h4 ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_ineq_least_3_l30_3072


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l30_3071

theorem arithmetic_sequence_n_value (a_1 d a_nm1 n : ℤ) (h1 : a_1 = -1) (h2 : d = 2) (h3 : a_nm1 = 15) :
    a_nm1 = a_1 + (n - 2) * d → n = 10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l30_3071


namespace NUMINAMATH_GPT_min_balls_to_draw_l30_3044

theorem min_balls_to_draw (black white red : ℕ) (h_black : black = 10) (h_white : white = 9) (h_red : red = 8) :
  ∃ n, n = 20 ∧
  ∀ k, (k < 20) → ¬ (∃ b w r, b + w + r = k ∧ b ≤ black ∧ w ≤ white ∧ r ≤ red ∧ r > 0 ∧ w > 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_balls_to_draw_l30_3044


namespace NUMINAMATH_GPT_closest_perfect_square_to_325_is_324_l30_3085

theorem closest_perfect_square_to_325_is_324 :
  ∃ n : ℕ, n^2 = 324 ∧ (∀ m : ℕ, m * m ≠ 325) ∧
    (n = 18 ∧ (∀ k : ℕ, (k*k < 325 ∧ (325 - k*k) > 325 - 324) ∨ 
               (k*k > 325 ∧ (k*k - 325) > 361 - 325))) :=
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_325_is_324_l30_3085


namespace NUMINAMATH_GPT_sum_of_first_8_terms_l30_3076

noncomputable def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms 
  (a r : ℝ)
  (h₁ : sum_of_geometric_sequence a r 4 = 5)
  (h₂ : sum_of_geometric_sequence a r 12 = 35) :
  sum_of_geometric_sequence a r 8 = 15 := 
sorry

end NUMINAMATH_GPT_sum_of_first_8_terms_l30_3076


namespace NUMINAMATH_GPT_num_factors_of_2_pow_20_minus_1_l30_3089

/-- 
Prove that the number of positive two-digit integers 
that are factors of \(2^{20} - 1\) is 5.
-/
theorem num_factors_of_2_pow_20_minus_1 :
  ∃ (n : ℕ), n = 5 ∧ (∀ (k : ℕ), k ∣ (2^20 - 1) → 10 ≤ k ∧ k < 100 → k = 33 ∨ k = 15 ∨ k = 27 ∨ k = 41 ∨ k = 45) 
  :=
sorry

end NUMINAMATH_GPT_num_factors_of_2_pow_20_minus_1_l30_3089


namespace NUMINAMATH_GPT_angle_in_third_quadrant_half_l30_3067

theorem angle_in_third_quadrant_half {
  k : ℤ 
} (h1: (k * 360 + 180) < α) (h2 : α < k * 360 + 270) :
  (k * 180 + 90) < (α / 2) ∧ (α / 2) < (k * 180 + 135) :=
sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_half_l30_3067


namespace NUMINAMATH_GPT_collinear_probability_l30_3063

-- Define the rectangular array
def rows : ℕ := 4
def cols : ℕ := 5
def total_dots : ℕ := rows * cols
def chosen_dots : ℕ := 4

-- Define the collinear sets
def horizontal_lines : ℕ := rows
def vertical_lines : ℕ := cols
def collinear_sets : ℕ := horizontal_lines + vertical_lines

-- Define the total combinations of choosing 4 dots out of 20
def total_combinations : ℕ := Nat.choose total_dots chosen_dots

-- Define the probability
def probability : ℚ := collinear_sets / total_combinations

theorem collinear_probability : probability = 9 / 4845 := by
  sorry

end NUMINAMATH_GPT_collinear_probability_l30_3063


namespace NUMINAMATH_GPT_vasya_drives_fraction_l30_3021

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_vasya_drives_fraction_l30_3021


namespace NUMINAMATH_GPT_range_of_root_difference_l30_3041

variable (a b c d : ℝ)
variable (x1 x2 : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_of_root_difference
  (h1 : a ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hroot1 : f a b c x1 = 0)
  (hroot2 : f a b c x2 = 0)
  : |x1 - x2| ∈ Set.Ico (Real.sqrt 3 / 3) (2 / 3) := sorry

end NUMINAMATH_GPT_range_of_root_difference_l30_3041


namespace NUMINAMATH_GPT_circle_center_l30_3034

theorem circle_center :
    ∃ (h k : ℝ), (x^2 - 10 * x + y^2 - 4 * y = -4) →
                 (x - h)^2 + (y - k)^2 = 25 ∧ h = 5 ∧ k = 2 :=
sorry

end NUMINAMATH_GPT_circle_center_l30_3034


namespace NUMINAMATH_GPT_range_of_a_l30_3056

theorem range_of_a (m a : ℝ) (h1 : m < a) (h2 : m ≤ -1) : a > -1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l30_3056


namespace NUMINAMATH_GPT_regular_polygon_interior_angle_l30_3052

theorem regular_polygon_interior_angle (S : ℝ) (n : ℕ) (h1 : S = 720) (h2 : (n - 2) * 180 = S) : 
  (S / n) = 120 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_interior_angle_l30_3052


namespace NUMINAMATH_GPT_inverse_prop_relation_l30_3030

theorem inverse_prop_relation (y₁ y₂ y₃ : ℝ) :
  (y₁ = (1 : ℝ) / (-1)) →
  (y₂ = (1 : ℝ) / (-2)) →
  (y₃ = (1 : ℝ) / (3)) →
  y₃ > y₂ ∧ y₂ > y₁ :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_GPT_inverse_prop_relation_l30_3030


namespace NUMINAMATH_GPT_find_a1_l30_3050

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

theorem find_a1 (h_arith : is_arithmetic_sequence a 3) (ha2 : a 2 = -5) : a 1 = -8 :=
sorry

end NUMINAMATH_GPT_find_a1_l30_3050
