import Mathlib

namespace NUMINAMATH_GPT_solid_could_be_rectangular_prism_or_cylinder_l2396_239636

-- Definitions for the conditions
def is_rectangular_prism (solid : Type) : Prop := sorry
def is_cylinder (solid : Type) : Prop := sorry
def front_view_is_rectangle (solid : Type) : Prop := sorry
def side_view_is_rectangle (solid : Type) : Prop := sorry

-- Main statement
theorem solid_could_be_rectangular_prism_or_cylinder
  {solid : Type}
  (h1 : front_view_is_rectangle solid)
  (h2 : side_view_is_rectangle solid) :
  is_rectangular_prism solid ∨ is_cylinder solid :=
sorry

end NUMINAMATH_GPT_solid_could_be_rectangular_prism_or_cylinder_l2396_239636


namespace NUMINAMATH_GPT_original_cube_edge_length_l2396_239645

theorem original_cube_edge_length (a : ℕ) (h1 : 6 * (a ^ 3) = 7 * (6 * (a ^ 2))) : a = 7 := 
by 
  sorry

end NUMINAMATH_GPT_original_cube_edge_length_l2396_239645


namespace NUMINAMATH_GPT_linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l2396_239639

-- Defining the first part of the problem
theorem linear_function_increasing_and_composition_eq_implies_values
  (a b : ℝ)
  (H1 : ∀ x y : ℝ, x < y → a * x + b < a * y + b)
  (H2 : ∀ x : ℝ, a * (a * x + b) + b = 16 * x + 5) :
  a = 4 ∧ b = 1 :=
by
  sorry

-- Defining the second part of the problem
theorem monotonic_gx_implies_m_range (m : ℝ)
  (H3 : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → (x2 + m) * (4 * x2 + 1) > (x1 + m) * (4 * x1 + 1)) :
  -9 / 4 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l2396_239639


namespace NUMINAMATH_GPT_Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l2396_239682

-- Problem 1: Factorization
theorem Problem1_factorize (a : ℝ) : a^2 - 8 * a + 15 = (a - 3) * (a - 5) :=
  sorry

-- Problem 2: Minimum Perimeter of triangle ABC
theorem Problem2_min_perimeter_triangle (a b c : ℝ) 
  (h : a^2 + b^2 - 14 * a - 8 * b + 65 = 0) (hc : ∃ k : ℤ, 2 * k + 1 = c) : 
  a + b + c ≥ 16 :=
  sorry

-- Problem 3: Maximum Value of the Polynomial
theorem Problem3_max_value_polynomial : 
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, y ≠ -1 → -2 * x^2 - 4 * x + 3 ≥ -2 * y^2 - 4 * y + 3 :=
  sorry

end NUMINAMATH_GPT_Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l2396_239682


namespace NUMINAMATH_GPT_martin_walk_distance_l2396_239610

-- Define the conditions
def time : ℝ := 6 -- Martin's walking time in hours
def speed : ℝ := 2 -- Martin's walking speed in miles per hour

-- Define the target distance
noncomputable def distance : ℝ := 12 -- Distance from Martin's house to Lawrence's house

-- The theorem to prove the target distance given the conditions
theorem martin_walk_distance : (speed * time = distance) :=
by
  sorry

end NUMINAMATH_GPT_martin_walk_distance_l2396_239610


namespace NUMINAMATH_GPT_bowling_ball_volume_l2396_239611

open Real

noncomputable def remaining_volume (d_bowling_ball d1 d2 d3 d4 h1 h2 h3 h4 : ℝ) : ℝ :=
  let r_bowling_ball := d_bowling_ball / 2
  let v_bowling_ball := (4/3) * π * (r_bowling_ball ^ 3)
  let v_hole1 := π * ((d1 / 2) ^ 2) * h1
  let v_hole2 := π * ((d2 / 2) ^ 2) * h2
  let v_hole3 := π * ((d3 / 2) ^ 2) * h3
  let v_hole4 := π * ((d4 / 2) ^ 2) * h4
  v_bowling_ball - (v_hole1 + v_hole2 + v_hole3 + v_hole4)

theorem bowling_ball_volume :
  remaining_volume 40 3 3 4 5 10 10 12 8 = 10523.67 * π :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_volume_l2396_239611


namespace NUMINAMATH_GPT_range_of_z_l2396_239634

variable (x y z : ℝ)

theorem range_of_z (hx : x ≥ 0) (hy : y ≥ x) (hxy : 4*x + 3*y ≤ 12) 
(hz : z = (x + 2 * y + 3) / (x + 1)) : 
2 ≤ z ∧ z ≤ 6 :=
sorry

end NUMINAMATH_GPT_range_of_z_l2396_239634


namespace NUMINAMATH_GPT_geometric_arithmetic_sum_l2396_239627

theorem geometric_arithmetic_sum {a : Nat → ℝ} {b : Nat → ℝ} 
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h_condition : a 3 * a 11 = 4 * a 7)
  (h_equal : a 7 = b 7) :
  b 5 + b 9 = 8 :=
sorry

end NUMINAMATH_GPT_geometric_arithmetic_sum_l2396_239627


namespace NUMINAMATH_GPT_ratio_jake_to_clementine_l2396_239638

-- Definitions based on conditions
def ClementineCookies : Nat := 72
def ToryCookies (J : Nat) : Nat := (J + ClementineCookies) / 2
def TotalCookies (J : Nat) : Nat := ClementineCookies + J + ToryCookies J
def TotalRevenue : Nat := 648
def CookiePrice : Nat := 2
def TotalCookiesSold : Nat := TotalRevenue / CookiePrice

-- The main proof statement
theorem ratio_jake_to_clementine : 
  ∃ J : Nat, TotalCookies J = TotalCookiesSold ∧ J / ClementineCookies = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_jake_to_clementine_l2396_239638


namespace NUMINAMATH_GPT_fraction_of_orange_juice_in_mixture_l2396_239621

theorem fraction_of_orange_juice_in_mixture
  (capacity_pitcher : ℕ)
  (fraction_first_pitcher : ℚ)
  (fraction_second_pitcher : ℚ)
  (condition1 : capacity_pitcher = 500)
  (condition2 : fraction_first_pitcher = 1/4)
  (condition3 : fraction_second_pitcher = 3/7) :
  (125 + 500 * (3/7)) / (2 * 500) = 95 / 280 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_orange_juice_in_mixture_l2396_239621


namespace NUMINAMATH_GPT_unit_A_saplings_l2396_239619

theorem unit_A_saplings 
  (Y B D J : ℕ)
  (h1 : J = 2 * Y + 20)
  (h2 : J = 3 * B + 24)
  (h3 : J = 5 * D - 45)
  (h4 : J + Y + B + D = 2126) :
  J = 1050 :=
by sorry

end NUMINAMATH_GPT_unit_A_saplings_l2396_239619


namespace NUMINAMATH_GPT_birthday_gift_l2396_239679

-- Define the conditions
def friends : Nat := 8
def dollars_per_friend : Nat := 15

-- Formulate the statement to prove
theorem birthday_gift : friends * dollars_per_friend = 120 := by
  -- Proof is skipped using 'sorry'
  sorry

end NUMINAMATH_GPT_birthday_gift_l2396_239679


namespace NUMINAMATH_GPT_area_not_covered_by_smaller_squares_l2396_239674

-- Define the conditions given in the problem
def side_length_larger_square : ℕ := 10
def side_length_smaller_square : ℕ := 4
def area_of_larger_square : ℕ := side_length_larger_square * side_length_larger_square
def area_of_each_smaller_square : ℕ := side_length_smaller_square * side_length_smaller_square

-- Define the total area of the two smaller squares
def total_area_smaller_squares : ℕ := area_of_each_smaller_square * 2

-- Define the uncovered area
def uncovered_area : ℕ := area_of_larger_square - total_area_smaller_squares

-- State the theorem to prove
theorem area_not_covered_by_smaller_squares :
  uncovered_area = 68 := by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_area_not_covered_by_smaller_squares_l2396_239674


namespace NUMINAMATH_GPT_number_of_cannoneers_l2396_239615

-- Define the variables for cannoneers, women, and men respectively
variables (C W M : ℕ)

-- Define the conditions as assumptions
def conditions : Prop :=
  W = 2 * C ∧
  M = 2 * W ∧
  M + W = 378

-- Prove that the number of cannoneers is 63
theorem number_of_cannoneers (h : conditions C W M) : C = 63 :=
by sorry

end NUMINAMATH_GPT_number_of_cannoneers_l2396_239615


namespace NUMINAMATH_GPT_total_rent_payment_l2396_239685

def weekly_rent : ℕ := 388
def number_of_weeks : ℕ := 1359

theorem total_rent_payment : weekly_rent * number_of_weeks = 526692 := 
  by 
  sorry

end NUMINAMATH_GPT_total_rent_payment_l2396_239685


namespace NUMINAMATH_GPT_not_enough_space_in_cube_l2396_239629

-- Define the edge length of the cube in kilometers.
def cube_edge_length_km : ℝ := 3

-- Define the global population exceeding threshold.
def global_population : ℝ := 7 * 10^9

-- Define the function to calculate the volume of a cube given its edge length in kilometers.
def cube_volume_km (edge_length: ℝ) : ℝ := edge_length^3

-- Define the conversion from kilometers to meters.
def km_to_m (distance_km: ℝ) : ℝ := distance_km * 1000

-- Define the function to calculate the volume of the cube in cubic meters.
def cube_volume_m (edge_length_km: ℝ) : ℝ := (km_to_m edge_length_km)^3

-- Statement: The entire population and all buildings and structures will not fit inside the cube.
theorem not_enough_space_in_cube :
  cube_volume_m cube_edge_length_km < global_population * (some_constant_value_to_account_for_buildings_and_structures) :=
sorry

end NUMINAMATH_GPT_not_enough_space_in_cube_l2396_239629


namespace NUMINAMATH_GPT_sum_of_possible_values_l2396_239657

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2396_239657


namespace NUMINAMATH_GPT_volume_of_parallelepiped_l2396_239616

theorem volume_of_parallelepiped (x y z : ℝ)
  (h1 : (x^2 + y^2) * z^2 = 13)
  (h2 : (y^2 + z^2) * x^2 = 40)
  (h3 : (x^2 + z^2) * y^2 = 45) :
  x * y * z = 6 :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_parallelepiped_l2396_239616


namespace NUMINAMATH_GPT_trace_bag_weight_is_two_l2396_239618

-- Define the weights of Gordon's shopping bags
def weight_gordon1 : ℕ := 3
def weight_gordon2 : ℕ := 7

-- Summarize Gordon's total weight
def total_weight_gordon : ℕ := weight_gordon1 + weight_gordon2

-- Provide necessary conditions from problem statement
def trace_bags_count : ℕ := 5
def trace_total_weight : ℕ := total_weight_gordon
def trace_one_bag_weight : ℕ := trace_total_weight / trace_bags_count

theorem trace_bag_weight_is_two : trace_one_bag_weight = 2 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_trace_bag_weight_is_two_l2396_239618


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2396_239692

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
    (h1 : a 1 = -1)
    (h2 : a 2 + a 3 = -2) :
    ∃ q : ℝ, (a 2 = a 1 * q) ∧ (a 3 = a 1 * q^2) ∧ (q = -2 ∨ q = 1) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2396_239692


namespace NUMINAMATH_GPT_fabric_length_l2396_239666

-- Define the width and area as given in the problem
def width : ℝ := 3
def area : ℝ := 24

-- Prove that the length is 8 cm
theorem fabric_length : (area / width) = 8 :=
by
  sorry

end NUMINAMATH_GPT_fabric_length_l2396_239666


namespace NUMINAMATH_GPT_coefficient_x3y5_l2396_239688

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition for the binomial expansion term of (x-y)^7
def expansion_term (r : ℕ) : ℤ := 
  (binom 7 r) * (-1) ^ r

-- The target coefficient for the term x^3 y^5 in (x+y)(x-y)^7
theorem coefficient_x3y5 :
  (expansion_term 5) * 1 + (expansion_term 4) * 1 = 14 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_coefficient_x3y5_l2396_239688


namespace NUMINAMATH_GPT_determine_pairs_l2396_239633

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem determine_pairs (n p : ℕ) (hn_pos : 0 < n) (hp_prime : is_prime p) (hn_le_2p : n ≤ 2 * p) (divisibility : n^p - 1 ∣ (p - 1)^n + 1):
  (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
by
  sorry

end NUMINAMATH_GPT_determine_pairs_l2396_239633


namespace NUMINAMATH_GPT_each_person_paid_45_l2396_239644

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end NUMINAMATH_GPT_each_person_paid_45_l2396_239644


namespace NUMINAMATH_GPT_total_pawns_left_is_10_l2396_239676

noncomputable def total_pawns_left_in_game 
    (initial_pawns : ℕ)
    (sophia_lost : ℕ)
    (chloe_lost : ℕ) : ℕ :=
  initial_pawns - sophia_lost + (initial_pawns - chloe_lost)

theorem total_pawns_left_is_10 :
  total_pawns_left_in_game 8 5 1 = 10 := by
  sorry

end NUMINAMATH_GPT_total_pawns_left_is_10_l2396_239676


namespace NUMINAMATH_GPT_domain_of_tan_arcsin_xsq_l2396_239684

noncomputable def domain_f (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -1 ∧ -1 ≤ x ∧ x ≤ 1

theorem domain_of_tan_arcsin_xsq :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ domain_f x := 
sorry

end NUMINAMATH_GPT_domain_of_tan_arcsin_xsq_l2396_239684


namespace NUMINAMATH_GPT_probability_area_less_than_circumference_l2396_239663

theorem probability_area_less_than_circumference :
  let probability (d : ℕ) := if d = 2 then (1 / 100 : ℚ)
                             else if d = 3 then (1 / 50 : ℚ)
                             else 0
  let sum_prob (d_s : List ℚ) := d_s.foldl (· + ·) 0
  let outcomes : List ℕ := List.range' 2 19 -- dice sum range from 2 to 20
  let valid_outcomes : List ℕ := outcomes.filter (· < 4)
  sum_prob (valid_outcomes.map probability) = (3 / 100 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_area_less_than_circumference_l2396_239663


namespace NUMINAMATH_GPT_harold_shared_with_five_friends_l2396_239694

theorem harold_shared_with_five_friends 
  (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) (shared : ℕ) (friends : ℕ)
  (H1 : total_marbles = 100)
  (H2 : kept_marbles = 20)
  (H3 : marbles_per_friend = 16)
  (H4 : shared = total_marbles - kept_marbles)
  (H5 : friends = shared / marbles_per_friend) :
  friends = 5 :=
by
  sorry

end NUMINAMATH_GPT_harold_shared_with_five_friends_l2396_239694


namespace NUMINAMATH_GPT_possible_rectangle_areas_l2396_239623

def is_valid_pair (a b : ℕ) := 
  a + b = 12 ∧ a > 0 ∧ b > 0

def rectangle_area (a b : ℕ) := a * b

theorem possible_rectangle_areas :
  {area | ∃ (a b : ℕ), is_valid_pair a b ∧ area = rectangle_area a b} 
  = {11, 20, 27, 32, 35, 36} := 
by 
  sorry

end NUMINAMATH_GPT_possible_rectangle_areas_l2396_239623


namespace NUMINAMATH_GPT_downward_parabola_with_symmetry_l2396_239606

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end NUMINAMATH_GPT_downward_parabola_with_symmetry_l2396_239606


namespace NUMINAMATH_GPT_cannot_fit_all_pictures_l2396_239602

theorem cannot_fit_all_pictures 
  (typeA_capacity : Nat) (typeB_capacity : Nat) (typeC_capacity : Nat)
  (typeA_count : Nat) (typeB_count : Nat) (typeC_count : Nat)
  (total_pictures : Nat)
  (h1 : typeA_capacity = 12)
  (h2 : typeB_capacity = 18)
  (h3 : typeC_capacity = 24)
  (h4 : typeA_count = 6)
  (h5 : typeB_count = 4)
  (h6 : typeC_count = 3)
  (h7 : total_pictures = 480) :
  (typeA_capacity * typeA_count + typeB_capacity * typeB_count + typeC_capacity * typeC_count < total_pictures) :=
  by sorry

end NUMINAMATH_GPT_cannot_fit_all_pictures_l2396_239602


namespace NUMINAMATH_GPT_price_per_glass_on_first_day_eq_half_l2396_239683

structure OrangeadeProblem where
  O : ℝ
  W : ℝ
  P1 : ℝ
  P2 : ℝ
  W_eq_O : W = O
  P2_value : P2 = 0.3333333333333333
  revenue_eq : 2 * O * P1 = 3 * O * P2

theorem price_per_glass_on_first_day_eq_half (prob : OrangeadeProblem) : prob.P1 = 0.50 := 
by
  sorry

end NUMINAMATH_GPT_price_per_glass_on_first_day_eq_half_l2396_239683


namespace NUMINAMATH_GPT_distance_to_place_is_24_l2396_239661

-- Definitions of the problem's conditions
def rowing_speed_still_water := 10    -- kmph
def current_velocity := 2             -- kmph
def round_trip_time := 5              -- hours

-- Effective speeds
def effective_speed_with_current := rowing_speed_still_water + current_velocity
def effective_speed_against_current := rowing_speed_still_water - current_velocity

-- Define the unknown distance D
variable (D : ℕ)

-- Define the times for each leg of the trip
def time_with_current := D / effective_speed_with_current
def time_against_current := D / effective_speed_against_current

-- The final theorem stating the round trip distance
theorem distance_to_place_is_24 :
  time_with_current + time_against_current = round_trip_time → D = 24 :=
by sorry

end NUMINAMATH_GPT_distance_to_place_is_24_l2396_239661


namespace NUMINAMATH_GPT_average_age_of_women_l2396_239624

variable {A W : ℝ}

theorem average_age_of_women (A : ℝ) (h : 12 * (A + 3) = 12 * A - 90 + W) : 
  W / 3 = 42 := by
  sorry

end NUMINAMATH_GPT_average_age_of_women_l2396_239624


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l2396_239628

theorem common_ratio_of_geometric_series (a1 a2 a3 : ℚ) (h1 : a1 = -4 / 7)
                                         (h2 : a2 = 14 / 3) (h3 : a3 = -98 / 9) :
  ∃ r : ℚ, r = a2 / a1 ∧ r = a3 / a2 ∧ r = -49 / 6 :=
by
  use -49 / 6
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l2396_239628


namespace NUMINAMATH_GPT_value_of_Y_l2396_239625

theorem value_of_Y :
  let part1 := 15 * 180 / 100  -- 15% of 180
  let part2 := part1 - part1 / 3  -- one-third less than 15% of 180
  let part3 := 24.5 * (2 * 270 / 3) / 100  -- 24.5% of (2/3 * 270)
  let part4 := (5.4 * 2) / (0.25 * 0.25)  -- (5.4 * 2) / (0.25)^2
  let Y := part2 + part3 - part4
  Y = -110.7 := by
    -- proof skipped
    sorry

end NUMINAMATH_GPT_value_of_Y_l2396_239625


namespace NUMINAMATH_GPT_find_overlap_length_l2396_239612

-- Define the given conditions
def plank_length : ℝ := 30 -- length of each plank in cm
def number_of_planks : ℕ := 25 -- number of planks
def total_fence_length : ℝ := 690 -- total length of the fence in cm

-- Definition for the overlap length
def overlap_length (y : ℝ) : Prop :=
  total_fence_length = (13 * plank_length) + (12 * (plank_length - 2 * y))

-- Theorem statement to prove the required overlap length
theorem find_overlap_length : ∃ y : ℝ, overlap_length y ∧ y = 2.5 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_overlap_length_l2396_239612


namespace NUMINAMATH_GPT_Ivy_cupcakes_l2396_239670

theorem Ivy_cupcakes (M : ℕ) (h1 : M + (M + 15) = 55) : M = 20 :=
by
  sorry

end NUMINAMATH_GPT_Ivy_cupcakes_l2396_239670


namespace NUMINAMATH_GPT_ratio_population_A_to_F_l2396_239620

variable (F : ℕ)

def population_E := 6 * F
def population_D := 2 * population_E
def population_C := 8 * population_D
def population_B := 3 * population_C
def population_A := 5 * population_B

theorem ratio_population_A_to_F (F_pos : F > 0) :
  population_A F / F = 1440 := by
sorry

end NUMINAMATH_GPT_ratio_population_A_to_F_l2396_239620


namespace NUMINAMATH_GPT_coins_remainder_l2396_239675

theorem coins_remainder (N : ℕ) (h1 : N % 8 = 5) (h2 : N % 7 = 2) (hN_min : ∀ M : ℕ, (M % 8 = 5 ∧ M % 7 = 2) → N ≤ M) : N % 9 = 1 :=
sorry

end NUMINAMATH_GPT_coins_remainder_l2396_239675


namespace NUMINAMATH_GPT_sum_of_five_consecutive_odd_numbers_l2396_239609

theorem sum_of_five_consecutive_odd_numbers (x : ℤ) : 
  (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 5 * x :=
by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_odd_numbers_l2396_239609


namespace NUMINAMATH_GPT_solve_for_p_l2396_239696

theorem solve_for_p (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end NUMINAMATH_GPT_solve_for_p_l2396_239696


namespace NUMINAMATH_GPT_cricket_team_throwers_l2396_239604

def cricket_equation (T N : ℕ) := 
  (2 * N / 3 = 51 - T) ∧ (T + N = 58)

theorem cricket_team_throwers : 
  ∃ T : ℕ, ∃ N : ℕ, cricket_equation T N ∧ T = 37 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_throwers_l2396_239604


namespace NUMINAMATH_GPT_find_cd_l2396_239662

noncomputable def g (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) :
  g 2 c d = -9 ∧ g (-1) c d = -19 ↔
  (c = 19/3 ∧ d = -7/3) :=
by
  sorry

end NUMINAMATH_GPT_find_cd_l2396_239662


namespace NUMINAMATH_GPT_Annika_hike_time_l2396_239690

-- Define the conditions
def hike_rate : ℝ := 12 -- in minutes per kilometer
def initial_distance_east : ℝ := 2.75 -- in kilometers
def total_distance_east : ℝ := 3.041666666666667 -- in kilometers
def total_time_needed : ℝ := 40 -- in minutes

-- The theorem to prove
theorem Annika_hike_time : 
  (initial_distance_east + (total_distance_east - initial_distance_east)) * hike_rate + total_distance_east * hike_rate = total_time_needed := 
by
  sorry

end NUMINAMATH_GPT_Annika_hike_time_l2396_239690


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l2396_239698

theorem radius_of_tangent_circle 
    (side_length : ℝ) 
    (tangent_angle : ℝ) 
    (sin_15 : ℝ)
    (circle_radius : ℝ) :
    side_length = 2 * Real.sqrt 3 →
    tangent_angle = 30 →
    sin_15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2) →
    circle_radius = 2 :=
by sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l2396_239698


namespace NUMINAMATH_GPT_school_club_members_l2396_239697

theorem school_club_members :
  ∃ n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 3 ∧
  n % 8 = 5 ∧
  n % 9 = 7 ∧
  n = 269 :=
by
  existsi 269
  sorry

end NUMINAMATH_GPT_school_club_members_l2396_239697


namespace NUMINAMATH_GPT_shark_sightings_in_Daytona_Beach_l2396_239626

def CM : ℕ := 7

def DB : ℕ := 3 * CM + 5

theorem shark_sightings_in_Daytona_Beach : DB = 26 := by
  sorry

end NUMINAMATH_GPT_shark_sightings_in_Daytona_Beach_l2396_239626


namespace NUMINAMATH_GPT_f_has_two_zeros_iff_l2396_239630

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * exp x + a * (x - 1)^2

theorem f_has_two_zeros_iff (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a :=
sorry

end NUMINAMATH_GPT_f_has_two_zeros_iff_l2396_239630


namespace NUMINAMATH_GPT_frequency_distribution_table_understanding_l2396_239695

theorem frequency_distribution_table_understanding (size_sample_group : Prop) :
  (∃ (size_proportion : Prop) (corresponding_situation : Prop),
    size_sample_group → size_proportion ∧ corresponding_situation) :=
sorry

end NUMINAMATH_GPT_frequency_distribution_table_understanding_l2396_239695


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2396_239643

theorem arithmetic_sequence_sum :
  let a1 := 1
  let d := 2
  let n := 10
  let an := 19
  let sum := 100
  let general_term := fun (n : ℕ) => a1 + (n - 1) * d
  (general_term n = an) → (n = 10) → (sum = (n * (a1 + an)) / 2) →
  sum = 100 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2396_239643


namespace NUMINAMATH_GPT_chris_raisins_nuts_l2396_239614

theorem chris_raisins_nuts (R N x : ℝ) 
  (hN : N = 4 * R) 
  (hxR : x * R = 0.15789473684210525 * (x * R + 4 * N)) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_chris_raisins_nuts_l2396_239614


namespace NUMINAMATH_GPT_quadratic_root_property_l2396_239686

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_property_l2396_239686


namespace NUMINAMATH_GPT_clock_angle_8_30_l2396_239613

theorem clock_angle_8_30 
  (angle_per_hour_mark : ℝ := 30)
  (angle_per_minute_mark : ℝ := 6)
  (hour_hand_angle_8 : ℝ := 8 * angle_per_hour_mark)
  (half_hour_movement : ℝ := 0.5 * angle_per_hour_mark)
  (hour_hand_angle_8_30 : ℝ := hour_hand_angle_8 + half_hour_movement)
  (minute_hand_angle_30 : ℝ := 30 * angle_per_minute_mark) :
  abs (hour_hand_angle_8_30 - minute_hand_angle_30) = 75 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_8_30_l2396_239613


namespace NUMINAMATH_GPT_unique_number_not_in_range_of_g_l2396_239617

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range_of_g 
  (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g 5 a b c d = 5) (h6 : g 25 a b c d = 25) 
  (h7 : ∀ x, x ≠ -d/c → g (g x a b c d) a b c d = x) :
  ∃ r, r = 15 ∧ ∀ y, g y a b c d ≠ r := 
by
  sorry

end NUMINAMATH_GPT_unique_number_not_in_range_of_g_l2396_239617


namespace NUMINAMATH_GPT_gaokun_population_scientific_notation_l2396_239637

theorem gaokun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (425000 = a * 10^n) ∧ (a = 4.25) ∧ (n = 5) :=
by
  sorry

end NUMINAMATH_GPT_gaokun_population_scientific_notation_l2396_239637


namespace NUMINAMATH_GPT_alice_speed_proof_l2396_239669

-- Problem definitions
def distance : ℕ := 1000
def abel_speed : ℕ := 50
def abel_arrival_time := distance / abel_speed
def alice_delay : ℕ := 1  -- Alice starts 1 hour later
def earlier_arrival_abel : ℕ := 6  -- Abel arrives 6 hours earlier than Alice

noncomputable def alice_speed : ℕ := (distance / (abel_arrival_time + earlier_arrival_abel))

theorem alice_speed_proof : alice_speed = 200 / 3 := by
  sorry -- proof not required as per instructions

end NUMINAMATH_GPT_alice_speed_proof_l2396_239669


namespace NUMINAMATH_GPT_all_numbers_equal_l2396_239648

theorem all_numbers_equal (x : Fin 101 → ℝ) 
  (h : ∀ i : Fin 100, x i.val^3 + x ⟨(i.val + 1) % 101, sorry⟩ = (x ⟨(i.val + 1) % 101, sorry⟩)^3 + x ⟨(i.val + 2) % 101, sorry⟩) :
  ∀ i j : Fin 101, x i = x j := 
by 
  sorry

end NUMINAMATH_GPT_all_numbers_equal_l2396_239648


namespace NUMINAMATH_GPT_abs_k_eq_sqrt_19_div_4_l2396_239646

theorem abs_k_eq_sqrt_19_div_4
  (k : ℝ)
  (h : ∀ x : ℝ, x^2 - 4 * k * x + 1 = 0 → (x = r ∨ x = s))
  (h₁ : r + s = 4 * k)
  (h₂ : r * s = 1)
  (h₃ : r^2 + s^2 = 17) :
  |k| = (Real.sqrt 19) / 4 := by
sorry

end NUMINAMATH_GPT_abs_k_eq_sqrt_19_div_4_l2396_239646


namespace NUMINAMATH_GPT_boys_ages_l2396_239673

theorem boys_ages (a b : ℕ) (h1 : a = b) (h2 : a + b + 11 = 29) : a = 9 :=
by
  sorry

end NUMINAMATH_GPT_boys_ages_l2396_239673


namespace NUMINAMATH_GPT_determine_integer_n_l2396_239642

theorem determine_integer_n (n : ℤ) :
  (n + 15 ≥ 16) ∧ (-5 * n < -10) → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_integer_n_l2396_239642


namespace NUMINAMATH_GPT_solve_for_m_l2396_239650

theorem solve_for_m (m x : ℤ) (h : 4 * x + 2 * m - 14 = 0) (hx : x = 2) : m = 3 :=
by
  -- Proof steps will go here.
  sorry

end NUMINAMATH_GPT_solve_for_m_l2396_239650


namespace NUMINAMATH_GPT_price_reduction_daily_profit_l2396_239647

theorem price_reduction_daily_profit
    (profit_per_item : ℕ)
    (avg_daily_sales : ℕ)
    (item_increase_per_unit_price_reduction : ℕ)
    (target_daily_profit : ℕ)
    (x : ℕ) :
    profit_per_item = 40 →
    avg_daily_sales = 20 →
    item_increase_per_unit_price_reduction = 2 →
    target_daily_profit = 1200 →

    ((profit_per_item - x) * (avg_daily_sales + item_increase_per_unit_price_reduction * x) = target_daily_profit) →
    x = 20 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_price_reduction_daily_profit_l2396_239647


namespace NUMINAMATH_GPT_randy_fifth_quiz_score_l2396_239689

def scores : List ℕ := [90, 98, 92, 94]

def goal_average : ℕ := 94

def total_points (n : ℕ) (avg : ℕ) : ℕ := n * avg

def current_points (l : List ℕ) : ℕ := l.sum

def needed_score (total current : ℕ) : ℕ := total - current

theorem randy_fifth_quiz_score :
  needed_score (total_points 5 goal_average) (current_points scores) = 96 :=
by 
  sorry

end NUMINAMATH_GPT_randy_fifth_quiz_score_l2396_239689


namespace NUMINAMATH_GPT_parallel_lines_have_equal_slopes_l2396_239699

theorem parallel_lines_have_equal_slopes (a : ℝ) :
  (∃ a : ℝ, (∀ y : ℝ, 2 * a * y - 1 = 0) ∧ (∃ x y : ℝ, (3 * a - 1) * x + y - 1 = 0) 
  → (∃ a : ℝ, (1 / (2 * a)) = - (3 * a - 1))) 
→ a = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_have_equal_slopes_l2396_239699


namespace NUMINAMATH_GPT_problem1_problem2_l2396_239603

-- Problem 1
theorem problem1 :
  2 * Real.cos (Real.pi / 4) + (Real.pi - Real.sqrt 3)^0 - Real.sqrt 8 = 1 - Real.sqrt 2 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) :
  (2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2396_239603


namespace NUMINAMATH_GPT_matrix_inverse_l2396_239605

-- Define the given matrix
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, 4], ![-2, 8]]

-- Define the expected inverse matrix
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/6, -1/12], ![1/24, 5/48]]

-- The main statement: Prove that the inverse of A is equal to the expected inverse
theorem matrix_inverse :
  A⁻¹ = A_inv_expected := sorry

end NUMINAMATH_GPT_matrix_inverse_l2396_239605


namespace NUMINAMATH_GPT_mac_runs_faster_by_120_minutes_l2396_239665

theorem mac_runs_faster_by_120_minutes :
  ∀ (D : ℝ), (D / 3 - D / 4 = 2) → 2 * 60 = 120 := by
  -- Definitions matching the conditions
  intro D
  intro h

  -- The proof is not required, hence using sorry
  sorry

end NUMINAMATH_GPT_mac_runs_faster_by_120_minutes_l2396_239665


namespace NUMINAMATH_GPT_tree_height_end_of_third_year_l2396_239653

theorem tree_height_end_of_third_year (h : ℝ) : 
    (∃ h0 h3 h6 : ℝ, 
      h3 = h0 * 3^3 ∧ 
      h6 = h3 * 2^3 ∧ 
      h6 = 1458) → h3 = 182.25 :=
by sorry

end NUMINAMATH_GPT_tree_height_end_of_third_year_l2396_239653


namespace NUMINAMATH_GPT_sales_in_fifth_month_l2396_239680

-- Define the sales figures and average target
def s1 : ℕ := 6435
def s2 : ℕ := 6927
def s3 : ℕ := 6855
def s4 : ℕ := 7230
def s6 : ℕ := 6191
def s_target : ℕ := 6700
def n_months : ℕ := 6

-- Define the total sales and the required fifth month sale
def total_sales : ℕ := s_target * n_months
def s5 : ℕ := total_sales - (s1 + s2 + s3 + s4 + s6)

-- The main theorem statement we need to prove
theorem sales_in_fifth_month :
  s5 = 6562 :=
sorry

end NUMINAMATH_GPT_sales_in_fifth_month_l2396_239680


namespace NUMINAMATH_GPT_value_of_business_l2396_239658

variable (business_value : ℝ) -- We are looking for the value of the business
variable (man_ownership_fraction : ℝ := 2/3) -- The fraction of the business the man owns
variable (sale_fraction : ℝ := 3/4) -- The fraction of the man's shares that were sold
variable (sale_amount : ℝ := 6500) -- The amount for which the fraction of the shares were sold

-- The main theorem we are trying to prove
theorem value_of_business (h1 : man_ownership_fraction = 2/3) (h2 : sale_fraction = 3/4) (h3 : sale_amount = 6500) :
    business_value = 39000 := 
sorry

end NUMINAMATH_GPT_value_of_business_l2396_239658


namespace NUMINAMATH_GPT_red_marbles_count_l2396_239607

variable (n : ℕ)

-- Conditions
def ratio_green_yellow_red := (3 * n, 4 * n, 2 * n)
def not_red_marbles := 3 * n + 4 * n = 63

-- Goal
theorem red_marbles_count (hn : not_red_marbles n) : 2 * n = 18 :=
by
  sorry

end NUMINAMATH_GPT_red_marbles_count_l2396_239607


namespace NUMINAMATH_GPT_gcd_correct_l2396_239640

def gcd_87654321_12345678 : ℕ :=
  gcd 87654321 12345678

theorem gcd_correct : gcd_87654321_12345678 = 75 := by 
  sorry

end NUMINAMATH_GPT_gcd_correct_l2396_239640


namespace NUMINAMATH_GPT_time_2556_hours_from_now_main_l2396_239656

theorem time_2556_hours_from_now (h : ℕ) (mod_res : h % 12 = 0) :
  (3 + h) % 12 = 3 :=
by {
  sorry
}

-- Constants
def current_time : ℕ := 3
def hours_passed : ℕ := 2556
-- Proof input
def modular_result : hours_passed % 12 = 0 := by {
 sorry -- In the real proof, we should show that 2556 is divisible by 12
}

-- Main theorem instance
theorem main : (current_time + hours_passed) % 12 = 3 := 
  time_2556_hours_from_now hours_passed modular_result

end NUMINAMATH_GPT_time_2556_hours_from_now_main_l2396_239656


namespace NUMINAMATH_GPT_assign_grades_l2396_239667

def num_students : ℕ := 15
def options_per_student : ℕ := 4

theorem assign_grades:
  options_per_student ^ num_students = 1073741824 := by
  sorry

end NUMINAMATH_GPT_assign_grades_l2396_239667


namespace NUMINAMATH_GPT_find_k_l2396_239668

theorem find_k 
  (e1 : ℝ × ℝ) (h_e1 : e1 = (1, 0))
  (e2 : ℝ × ℝ) (h_e2 : e2 = (0, 1))
  (a : ℝ × ℝ) (h_a : a = (1, -2))
  (b : ℝ × ℝ) (h_b : b = (k, 1))
  (parallel : ∃ m : ℝ, a = (m * b.1, m * b.2)) : 
  k = -1/2 :=
sorry

end NUMINAMATH_GPT_find_k_l2396_239668


namespace NUMINAMATH_GPT_square_side_length_equals_4_l2396_239678

theorem square_side_length_equals_4 (s : ℝ) (h : s^2 = 4 * s) : s = 4 :=
sorry

end NUMINAMATH_GPT_square_side_length_equals_4_l2396_239678


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l2396_239677

theorem geometric_sequence_general_term (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) 
  (h1 : a 5 = a1 * q^4)
  (h2 : a 10 = a1 * q^9)
  (h3 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h4 : ∀ n, a n = a1 * q^(n - 1))
  (h_inc : q > 1) :
  ∀ n, a n = 2^n :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l2396_239677


namespace NUMINAMATH_GPT_larger_number_l2396_239652

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_l2396_239652


namespace NUMINAMATH_GPT_find_a_from_binomial_l2396_239601

variable (x : ℝ) (a : ℝ)

def binomial_term (r : ℕ) : ℝ :=
  (Nat.choose 5 r) * ((-a)^r) * x^(5 - 2 * r)

theorem find_a_from_binomial :
  (∃ x : ℝ, ∃ a : ℝ, (binomial_term x a 1 = 10)) → a = -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_from_binomial_l2396_239601


namespace NUMINAMATH_GPT_find_length_of_chord_AB_l2396_239631

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of points A and B
variables (x1 x2 y1 y2 : ℝ)

-- Define the conditions
def conditions : Prop := 
  parabola x1 y1 ∧ parabola x2 y2 ∧ (x1 + x2 = 4 / 3)

-- Define the length of chord AB
def length_of_chord_AB : ℝ := 
  (x1 + 1) + (x2 + 1)

-- Prove the length of chord AB
theorem find_length_of_chord_AB (x1 x2 y1 y2 : ℝ) (h : conditions x1 x2 y1 y2) :
  length_of_chord_AB x1 x2 = 10 / 3 :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_find_length_of_chord_AB_l2396_239631


namespace NUMINAMATH_GPT_product_sum_even_l2396_239632

theorem product_sum_even (m n : ℤ) : Even (m * n * (m + n)) := 
sorry

end NUMINAMATH_GPT_product_sum_even_l2396_239632


namespace NUMINAMATH_GPT_find_length_of_GH_l2396_239622

variable {A B C F G H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace F] [MetricSpace G] [MetricSpace H]

variables (AB BC Res : ℝ)
variables (ratio1 ratio2 : ℝ)
variable (similar : SimilarTriangles A B C F G H)

def length_of_GH (GH : ℝ) : Prop :=
  GH = 15

theorem find_length_of_GH (h1 : AB = 15) (h2 : BC = 25) (h3 : ratio1 = 5) (h4 : ratio2 = 3)
  (h5 : similar) : ∃ GH, length_of_GH GH :=
by
  have ratio : ratio2 / ratio1 = 3 / 5 := by assumption
  sorry

end NUMINAMATH_GPT_find_length_of_GH_l2396_239622


namespace NUMINAMATH_GPT_parabola_distance_l2396_239681

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end NUMINAMATH_GPT_parabola_distance_l2396_239681


namespace NUMINAMATH_GPT_tank_capacity_l2396_239600

variable (x : ℝ) -- Total capacity of the tank

theorem tank_capacity (h1 : x / 8 = 120 / (1 / 2 - 1 / 8)) :
  x = 320 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l2396_239600


namespace NUMINAMATH_GPT_expand_expression_l2396_239635

theorem expand_expression (x y : ℤ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l2396_239635


namespace NUMINAMATH_GPT_correlation_non_deterministic_relationship_l2396_239671

theorem correlation_non_deterministic_relationship
  (independent_var_fixed : Prop)
  (dependent_var_random : Prop)
  (correlation_def : Prop)
  (correlation_randomness : Prop) :
  (correlation_def → non_deterministic) :=
by
  sorry

end NUMINAMATH_GPT_correlation_non_deterministic_relationship_l2396_239671


namespace NUMINAMATH_GPT_max_min_diff_value_l2396_239660

noncomputable def max_min_diff_c (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : ℝ :=
  (10 / 3) - (-2)

theorem max_min_diff_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : 
  max_min_diff_c a b c h1 h2 = 16 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_max_min_diff_value_l2396_239660


namespace NUMINAMATH_GPT_park_width_l2396_239672

/-- The rectangular park theorem -/
theorem park_width 
  (length : ℕ)
  (lawn_area : ℤ)
  (road_width : ℕ)
  (crossroads : ℕ)
  (W : ℝ) :
  length = 60 →
  lawn_area = 2109 →
  road_width = 3 →
  crossroads = 2 →
  W = (2109 + (2 * 3 * 60) : ℝ) / 60 :=
sorry

end NUMINAMATH_GPT_park_width_l2396_239672


namespace NUMINAMATH_GPT_pears_thrown_away_on_first_day_l2396_239664

theorem pears_thrown_away_on_first_day (x : ℝ) (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.8 * P = P * 0.8)
  (total_thrown_percentage : (x / 100) * 0.2 * P + 0.2 * (1 - x / 100) * 0.2 * P = 0.12 * P ) : 
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_pears_thrown_away_on_first_day_l2396_239664


namespace NUMINAMATH_GPT_check_triangle_345_l2396_239654

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem check_triangle_345 : satisfies_triangle_inequality 3 4 5 := by
  sorry

end NUMINAMATH_GPT_check_triangle_345_l2396_239654


namespace NUMINAMATH_GPT_max_r_value_l2396_239651

theorem max_r_value (r : ℕ) (hr : r ≥ 2)
  (m n : Fin r → ℤ)
  (h : ∀ i j : Fin r, i < j → |m i * n j - m j * n i| = 1) :
  r ≤ 3 := 
sorry

end NUMINAMATH_GPT_max_r_value_l2396_239651


namespace NUMINAMATH_GPT_shaded_fraction_in_fourth_square_l2396_239608

theorem shaded_fraction_in_fourth_square : 
  ∀ (f : ℕ → ℕ), (f 1 = 1)
  ∧ (f 2 = 3)
  ∧ (f 3 = 5)
  ∧ (f 4 = f 3 + (3 - 1) + (5 - 3))
  ∧ (f 4 * 2 = 14)
  → (f 4 = 7)
  → (f 4 / 16 = 7 / 16) :=
sorry

end NUMINAMATH_GPT_shaded_fraction_in_fourth_square_l2396_239608


namespace NUMINAMATH_GPT_min_value_of_quadratic_l2396_239691

theorem min_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y^2 - 6 * y + 5 ≥ (x - 3)^2 - 4) ∧ (y^2 - 6 * y + 5 = -4) :=
by sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l2396_239691


namespace NUMINAMATH_GPT_selling_price_correct_l2396_239659

-- Define the conditions
def boxes := 3
def face_masks_per_box := 20
def cost_price := 15  -- in dollars
def profit := 15      -- in dollars

-- Define the total number of face masks
def total_face_masks := boxes * face_masks_per_box

-- Define the total amount he wants after selling all face masks
def total_amount := cost_price + profit

-- Prove that the selling price per face mask is $0.50
noncomputable def selling_price_per_face_mask : ℚ :=
  total_amount / total_face_masks

theorem selling_price_correct : selling_price_per_face_mask = 0.50 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l2396_239659


namespace NUMINAMATH_GPT_original_wire_length_l2396_239687

theorem original_wire_length (side_len total_area : ℕ) (h1 : side_len = 2) (h2 : total_area = 92) :
  (total_area / (side_len * side_len)) * (4 * side_len) = 184 := 
by
  sorry

end NUMINAMATH_GPT_original_wire_length_l2396_239687


namespace NUMINAMATH_GPT_probability_at_least_one_male_l2396_239655

-- Definitions according to the problem conditions
def total_finalists : ℕ := 8
def female_finalists : ℕ := 5
def male_finalists : ℕ := 3
def num_selected : ℕ := 3

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probabilistic statement
theorem probability_at_least_one_male :
  let total_ways := binom total_finalists num_selected
  let ways_all_females := binom female_finalists num_selected
  let ways_at_least_one_male := total_ways - ways_all_females
  (ways_at_least_one_male : ℚ) / total_ways = 23 / 28 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_male_l2396_239655


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l2396_239693

theorem equilateral_triangle_side_length (perimeter : ℝ) (h : perimeter = 2) : abs (perimeter / 3 - 0.67) < 0.01 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l2396_239693


namespace NUMINAMATH_GPT_length_of_equal_sides_l2396_239641

-- Definitions based on conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ a = c)

def is_triangle (a b c : ℝ) : Prop :=
(a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def has_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
a + b + c = P

def one_side_length (a : ℝ) : Prop :=
a = 3

-- The proof statement
theorem length_of_equal_sides (a b c : ℝ) :
isosceles_triangle a b c →
is_triangle a b c →
has_perimeter a b c 7 →
one_side_length a ∨ one_side_length b ∨ one_side_length c →
(b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) :=
by
  intros iso tri per side_length
  sorry

end NUMINAMATH_GPT_length_of_equal_sides_l2396_239641


namespace NUMINAMATH_GPT_matrix_corner_sum_eq_l2396_239649

theorem matrix_corner_sum_eq (M : Matrix (Fin 2000) (Fin 2000) ℤ)
  (h : ∀ i j : Fin 1999, M i j + M (i+1) (j+1) = M i (j+1) + M (i+1) j) :
  M 0 0 + M 1999 1999 = M 0 1999 + M 1999 0 :=
sorry

end NUMINAMATH_GPT_matrix_corner_sum_eq_l2396_239649
