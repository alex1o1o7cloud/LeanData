import Mathlib

namespace NUMINAMATH_GPT_prob_two_packs_tablets_at_10am_dec31_l1890_189009
noncomputable def prob_two_packs_tablets (n : ℕ) : ℝ :=
  let numer := (2^n - 1)
  let denom := 2^(n-1) * n
  numer / denom

theorem prob_two_packs_tablets_at_10am_dec31 :
  prob_two_packs_tablets 10 = 1023 / 5120 := by
  sorry

end NUMINAMATH_GPT_prob_two_packs_tablets_at_10am_dec31_l1890_189009


namespace NUMINAMATH_GPT_range_of_values_for_a_l1890_189036

theorem range_of_values_for_a 
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x = x - 1/x - a * Real.log x)
  (h2 : ∀ x > 0, (x^2 - a * x + 1) ≥ 0) : 
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_values_for_a_l1890_189036


namespace NUMINAMATH_GPT_area_of_shaded_region_l1890_189091

theorem area_of_shaded_region :
  let inner_square_side_length := 3
  let triangle_base := 2
  let triangle_height := 1
  let number_of_triangles := 8
  let area_inner_square := inner_square_side_length * inner_square_side_length
  let area_one_triangle := (1/2) * triangle_base * triangle_height
  let total_area_triangles := number_of_triangles * area_one_triangle
  let total_area_shaded := area_inner_square + total_area_triangles
  total_area_shaded = 17 :=
sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1890_189091


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1890_189043

variables {R : Type*} [Field R] (a b c : R)

def condition1 : Prop := (a / b) = (b / c)
def condition2 : Prop := b^2 = a * c

theorem necessary_but_not_sufficient :
  (condition1 a b c → condition2 a b c) ∧ ¬ (condition2 a b c → condition1 a b c) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1890_189043


namespace NUMINAMATH_GPT_central_cell_value_l1890_189050

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end NUMINAMATH_GPT_central_cell_value_l1890_189050


namespace NUMINAMATH_GPT_fifth_number_21st_row_is_809_l1890_189054

-- Define the sequence of positive odd numbers
def nth_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the last odd number in the nth row
def last_odd_number_in_row (n : ℕ) : ℕ :=
  nth_odd_number (n * n)

-- Define the position of the 5th number in the 21st row
def pos_5th_in_21st_row : ℕ :=
  let sum_first_20_rows := 400
  sum_first_20_rows + 5

-- The 5th number from the left in the 21st row
def fifth_number_in_21st_row : ℕ :=
  nth_odd_number pos_5th_in_21st_row

-- The proof statement
theorem fifth_number_21st_row_is_809 : fifth_number_in_21st_row = 809 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_fifth_number_21st_row_is_809_l1890_189054


namespace NUMINAMATH_GPT_count_4_digit_multiples_of_5_is_9_l1890_189012

noncomputable def count_4_digit_multiples_of_5 : Nat :=
  let digits := [2, 7, 4, 5]
  let last_digit := 5
  let remaining_digits := [2, 7, 4]
  let case_1 := 3
  let case_2 := 3 * 2
  case_1 + case_2

theorem count_4_digit_multiples_of_5_is_9 : count_4_digit_multiples_of_5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_count_4_digit_multiples_of_5_is_9_l1890_189012


namespace NUMINAMATH_GPT_factorize_expression_1_factorize_expression_2_l1890_189030

theorem factorize_expression_1 (m : ℤ) : 
  m^3 - 2 * m^2 - 4 * m + 8 = (m - 2)^2 * (m + 2) := 
sorry

theorem factorize_expression_2 (x y : ℤ) : 
  x^2 - 2 * x * y + y^2 - 9 = (x - y + 3) * (x - y - 3) :=
sorry

end NUMINAMATH_GPT_factorize_expression_1_factorize_expression_2_l1890_189030


namespace NUMINAMATH_GPT_rabbit_toy_cost_l1890_189056

theorem rabbit_toy_cost 
  (cost_pet_food : ℝ) 
  (cost_cage : ℝ) 
  (found_dollar : ℝ)
  (total_cost : ℝ) 
  (h1 : cost_pet_food = 5.79) 
  (h2 : cost_cage = 12.51)
  (h3 : found_dollar = 1.00)
  (h4 : total_cost = 24.81):
  ∃ (cost_rabbit_toy : ℝ), cost_rabbit_toy = 7.51 := by
  let cost_rabbit_toy := total_cost - (cost_pet_food + cost_cage) + found_dollar
  use cost_rabbit_toy
  sorry

end NUMINAMATH_GPT_rabbit_toy_cost_l1890_189056


namespace NUMINAMATH_GPT_arithmetic_mean_difference_l1890_189033

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := 
by sorry

end NUMINAMATH_GPT_arithmetic_mean_difference_l1890_189033


namespace NUMINAMATH_GPT_weight_of_b_l1890_189072

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 135)
  (h2 : A + B = 80)
  (h3 : B + C = 94) : 
  B = 39 := 
by 
  sorry

end NUMINAMATH_GPT_weight_of_b_l1890_189072


namespace NUMINAMATH_GPT_cos_alpha_add_beta_div2_l1890_189077

open Real 

theorem cos_alpha_add_beta_div2 (α β : ℝ) 
  (h_range : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h_cos1 : cos (π/4 + α) = 1/3)
  (h_cos2 : cos (π/4 - β/2) = sqrt 3 / 3) :
  cos (α + β/2) = 5 * sqrt 3 / 9 :=
sorry

end NUMINAMATH_GPT_cos_alpha_add_beta_div2_l1890_189077


namespace NUMINAMATH_GPT_triangle_angles_are_30_60_90_l1890_189002

theorem triangle_angles_are_30_60_90
  (a b c OH R r : ℝ)
  (h1 : OH = c / 2)
  (h2 : OH = a)
  (h3 : a < b)
  (h4 : b < c)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  ∃ (A B C : ℝ), (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_angles_are_30_60_90_l1890_189002


namespace NUMINAMATH_GPT_points_on_line_l1890_189044

-- Define the two points the line connects
def P1 : (ℝ × ℝ) := (8, 10)
def P2 : (ℝ × ℝ) := (2, -2)

-- Define the candidate points
def A : (ℝ × ℝ) := (5, 4)
def E : (ℝ × ℝ) := (1, -4)

-- Define the line equation, given the slope and y-intercept
def line (x : ℝ) : ℝ := 2 * x - 6

theorem points_on_line :
  (A.snd = line A.fst) ∧ (E.snd = line E.fst) :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l1890_189044


namespace NUMINAMATH_GPT_earning_80_yuan_represents_l1890_189098

-- Defining the context of the problem
def spending (n : Int) : Int := -n
def earning (n : Int) : Int := n

-- The problem statement as a Lean theorem
theorem earning_80_yuan_represents (x : Int) (hx : earning x = 80) : x = 80 := 
by
  sorry

end NUMINAMATH_GPT_earning_80_yuan_represents_l1890_189098


namespace NUMINAMATH_GPT_bus_fare_one_way_cost_l1890_189011

-- Define the conditions
def zoo_entry (dollars : ℕ) : ℕ := dollars -- Zoo entry cost is $5 per person
def initial_money : ℕ := 40 -- They bring $40 with them
def money_left : ℕ := 24 -- They have $24 left after spending on zoo entry and bus fare

-- Given values
def noah_ava : ℕ := 2 -- Number of persons, Noah and Ava
def zoo_entry_cost : ℕ := 5 -- $5 per person for zoo entry
def total_money_spent := initial_money - money_left -- Money spent on zoo entry and bus fare

-- Function to calculate the total cost based on bus fare x
def total_cost (x : ℕ) : ℕ := noah_ava * zoo_entry_cost + 2 * noah_ava * x

-- Assertion to be proved
theorem bus_fare_one_way_cost : 
  ∃ (x : ℕ), total_cost x = total_money_spent ∧ x = 150 / 100 := sorry

end NUMINAMATH_GPT_bus_fare_one_way_cost_l1890_189011


namespace NUMINAMATH_GPT_element_in_set_l1890_189027

def M : Set (ℤ × ℤ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M :=
by
  sorry

end NUMINAMATH_GPT_element_in_set_l1890_189027


namespace NUMINAMATH_GPT_determine_h_l1890_189066

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_determine_h_l1890_189066


namespace NUMINAMATH_GPT_eight_xyz_le_one_equality_conditions_l1890_189019

theorem eight_xyz_le_one (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z ≤ 1 :=
sorry

theorem equality_conditions (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2 * x * y * z = 1) :
  8 * x * y * z = 1 ↔ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = -1/2 ∧ z = 1/2) ∨
                   (x = -1/2 ∧ y = 1/2 ∧ z = -1/2) ∨
                   (x = 1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end NUMINAMATH_GPT_eight_xyz_le_one_equality_conditions_l1890_189019


namespace NUMINAMATH_GPT_andrew_permit_rate_l1890_189068

def permits_per_hour (a h_a H T : ℕ) : ℕ :=
  T / (H - (a * h_a))

theorem andrew_permit_rate :
  permits_per_hour 2 3 8 100 = 50 := by
  sorry

end NUMINAMATH_GPT_andrew_permit_rate_l1890_189068


namespace NUMINAMATH_GPT_product_closest_to_106_l1890_189062

theorem product_closest_to_106 :
  let product := (2.1 : ℝ) * (50.8 - 0.45)
  abs (product - 106) < abs (product - 105) ∧
  abs (product - 106) < abs (product - 107) ∧
  abs (product - 106) < abs (product - 108) ∧
  abs (product - 106) < abs (product - 110) :=
by
  sorry

end NUMINAMATH_GPT_product_closest_to_106_l1890_189062


namespace NUMINAMATH_GPT_shara_shells_l1890_189039

def initial_shells : ℕ := 20
def first_vacation_day1_3 : ℕ := 5 * 3
def first_vacation_day4 : ℕ := 6
def second_vacation_day1_2 : ℕ := 4 * 2
def second_vacation_day3 : ℕ := 7
def third_vacation_day1 : ℕ := 8
def third_vacation_day2 : ℕ := 4
def third_vacation_day3_4 : ℕ := 3 * 2

def total_shells : ℕ :=
  initial_shells + 
  (first_vacation_day1_3 + first_vacation_day4) +
  (second_vacation_day1_2 + second_vacation_day3) + 
  (third_vacation_day1 + third_vacation_day2 + third_vacation_day3_4)

theorem shara_shells : total_shells = 74 :=
by
  sorry

end NUMINAMATH_GPT_shara_shells_l1890_189039


namespace NUMINAMATH_GPT_main_theorem_l1890_189082

def d_digits (d : ℕ) : Prop :=
  ∃ (d_1 d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 : ℕ),
    d = d_1 * 10^8 + d_2 * 10^7 + d_3 * 10^6 + d_4 * 10^5 + d_5 * 10^4 + d_6 * 10^3 + d_7 * 10^2 + d_8 * 10 + d_9

noncomputable def condition1 (d e : ℕ) (i : ℕ) : Prop :=
  (e - (d / 10^(8 - i) % 10)) * 10^(8 - i) + d ≡ 0 [MOD 7]

noncomputable def condition2 (e f : ℕ) (i : ℕ) : Prop :=
  (f - (e / 10^(8 - i) % 10)) * 10^(8 - i) + e ≡ 0 [MOD 7]

theorem main_theorem
  (d e f : ℕ)
  (h1 : d_digits d)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition1 d e i)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition2 e f i) :
  ∀ i, 1 ≤ i ∧ i ≤ 9 → (d / 10^(8 - i) % 10) ≡ (f / 10^(8 - i) % 10) [MOD 7] := sorry

end NUMINAMATH_GPT_main_theorem_l1890_189082


namespace NUMINAMATH_GPT_geo_seq_sum_neg_six_l1890_189080

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (a₁ q : ℝ), q ≠ 0 ∧ ∀ n, a n = a₁ * q^n

theorem geo_seq_sum_neg_six
  (a : ℕ → ℝ)
  (hgeom : geometric_sequence a)
  (ha_neg : a 1 < 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = -6 :=
  sorry

end NUMINAMATH_GPT_geo_seq_sum_neg_six_l1890_189080


namespace NUMINAMATH_GPT_primes_sum_product_composite_l1890_189085

theorem primes_sum_product_composite {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hdistinct_pq : p ≠ q) (hdistinct_pr : p ≠ r) (hdistinct_qr : q ≠ r) :
  ¬ Nat.Prime (p + q + r + p * q * r) :=
by
  sorry

end NUMINAMATH_GPT_primes_sum_product_composite_l1890_189085


namespace NUMINAMATH_GPT_jasmine_max_cards_l1890_189071

-- Define constants and conditions
def initial_card_price : ℝ := 0.95
def discount_card_price : ℝ := 0.85
def budget : ℝ := 9.00
def threshold : ℕ := 6

-- Define the condition for the total cost if more than 6 cards are bought
def total_cost (n : ℕ) : ℝ :=
  if n ≤ threshold then initial_card_price * n
  else initial_card_price * threshold + discount_card_price * (n - threshold)

-- Define the condition for the maximum number of cards Jasmine can buy 
def max_cards (n : ℕ) : Prop :=
  total_cost n ≤ budget ∧ ∀ m : ℕ, total_cost m ≤ budget → m ≤ n

-- Theore statement stating Jasmine can buy a maximum of 9 cards
theorem jasmine_max_cards : max_cards 9 :=
sorry

end NUMINAMATH_GPT_jasmine_max_cards_l1890_189071


namespace NUMINAMATH_GPT_smaller_square_area_l1890_189035

theorem smaller_square_area (A_L : ℝ) (h : A_L = 100) : ∃ A_S : ℝ, A_S = 50 := 
by
  sorry

end NUMINAMATH_GPT_smaller_square_area_l1890_189035


namespace NUMINAMATH_GPT_calculate_sum_l1890_189047

theorem calculate_sum (P r : ℝ) (h1 : 2 * P * r = 10200) (h2 : P * ((1 + r) ^ 2 - 1) = 11730) : P = 17000 :=
sorry

end NUMINAMATH_GPT_calculate_sum_l1890_189047


namespace NUMINAMATH_GPT_shorter_piece_length_l1890_189059

theorem shorter_piece_length (x : ℝ) (h : 3 * x = 60) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l1890_189059


namespace NUMINAMATH_GPT_highest_y_coordinate_l1890_189004

theorem highest_y_coordinate : 
  (∀ x y : ℝ, ((x - 4)^2 / 25 + y^2 / 49 = 0) → y = 0) := 
by
  sorry

end NUMINAMATH_GPT_highest_y_coordinate_l1890_189004


namespace NUMINAMATH_GPT_find_a_l1890_189046

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (x + 1) = 3 * x + 2) (h2 : f a = 5) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1890_189046


namespace NUMINAMATH_GPT_Lenny_pens_left_l1890_189006

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_Lenny_pens_left_l1890_189006


namespace NUMINAMATH_GPT_triangle_angles_ratios_l1890_189057

def angles_of_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

theorem triangle_angles_ratios (α β γ : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : β = 2 * α)
  (h3 : γ = 3 * α) : 
  angles_of_triangle 60 45 75 ∨ angles_of_triangle 45 22.5 112.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_ratios_l1890_189057


namespace NUMINAMATH_GPT_sum_of_square_areas_l1890_189093

theorem sum_of_square_areas (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_square_areas_l1890_189093


namespace NUMINAMATH_GPT_greater_number_is_18_l1890_189074

theorem greater_number_is_18 (x y : ℕ) (h₁ : x + y = 30) (h₂ : x - y = 6) : x = 18 :=
by
  sorry

end NUMINAMATH_GPT_greater_number_is_18_l1890_189074


namespace NUMINAMATH_GPT_Beth_bought_10_cans_of_corn_l1890_189055

theorem Beth_bought_10_cans_of_corn (a b : ℕ) (h1 : b = 15 + 2 * a) (h2 : b = 35) : a = 10 := by
  sorry

end NUMINAMATH_GPT_Beth_bought_10_cans_of_corn_l1890_189055


namespace NUMINAMATH_GPT_correct_conclusions_l1890_189026

-- Definitions based on conditions
def condition_1 (x : ℝ) : Prop := x ≠ 0 → x + |x| > 0
def condition_3 (a b c : ℝ) (Δ : ℝ) : Prop := a > 0 ∧ Δ ≤ 0 ∧ Δ = b^2 - 4*a*c → 
  ∀ x, a*x^2 + b*x + c ≥ 0

-- Stating the proof problem
theorem correct_conclusions (x a b c Δ : ℝ) :
  (condition_1 x) ∧ (condition_3 a b c Δ) :=
sorry

end NUMINAMATH_GPT_correct_conclusions_l1890_189026


namespace NUMINAMATH_GPT_mike_travel_time_l1890_189079

-- Definitions of conditions
def dave_steps_per_min : ℕ := 85
def dave_step_length_cm : ℕ := 70
def dave_time_min : ℕ := 20
def mike_steps_per_min : ℕ := 95
def mike_step_length_cm : ℕ := 65

-- Calculate Dave's speed in cm/min
def dave_speed_cm_per_min := dave_steps_per_min * dave_step_length_cm

-- Calculate the distance to school in cm
def school_distance_cm := dave_speed_cm_per_min * dave_time_min

-- Calculate Mike's speed in cm/min
def mike_speed_cm_per_min := mike_steps_per_min * mike_step_length_cm

-- Calculate the time for Mike to get to school in minutes as a rational number
def mike_time_min := (school_distance_cm : ℚ) / mike_speed_cm_per_min

-- The proof problem statement
theorem mike_travel_time :
  mike_time_min = 19 + 2 / 7 :=
sorry

end NUMINAMATH_GPT_mike_travel_time_l1890_189079


namespace NUMINAMATH_GPT_find_common_difference_l1890_189015

-- Definitions based on conditions in a)
def common_difference_4_10 (a₁ d : ℝ) : Prop :=
  (a₁ + 3 * d) + (a₁ + 9 * d) = 0

def sum_relation (a₁ d : ℝ) : Prop :=
  2 * (12 * a₁ + 66 * d) = (2 * a₁ + d + 10)

-- Math proof problem statement
theorem find_common_difference (a₁ d : ℝ) 
  (h₁ : common_difference_4_10 a₁ d) 
  (h₂ : sum_relation a₁ d) : 
  d = -10 :=
sorry

end NUMINAMATH_GPT_find_common_difference_l1890_189015


namespace NUMINAMATH_GPT_password_count_correct_l1890_189087

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_password_count_correct_l1890_189087


namespace NUMINAMATH_GPT_percentage_multiplication_l1890_189088

theorem percentage_multiplication :
  (0.15 * 0.20 * 0.25) * 100 = 0.75 := 
by
  sorry

end NUMINAMATH_GPT_percentage_multiplication_l1890_189088


namespace NUMINAMATH_GPT_calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l1890_189083

noncomputable def cost_plan1_fixed (num_suits num_ties : ℕ) : ℕ :=
  if num_ties > num_suits then 200 * num_suits + 40 * (num_ties - num_suits)
  else 200 * num_suits

noncomputable def cost_plan2_fixed (num_suits num_ties : ℕ) : ℕ :=
  (200 * num_suits + 40 * num_ties) * 9 / 10

noncomputable def cost_plan1_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  200 * num_suits + 40 * (x - num_suits)

noncomputable def cost_plan2_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  (200 * num_suits + 40 * x) * 9 / 10

theorem calculate_fixed_payment :
  cost_plan1_fixed 20 22 = 4080 ∧ cost_plan2_fixed 20 22 = 4392 :=
by sorry

theorem calculate_variable_payment (x : ℕ) (hx : x > 20) :
  cost_plan1_variable 20 x = 40 * x + 3200 ∧ cost_plan2_variable 20 x = 36 * x + 3600 :=
by sorry

theorem compare_plans_for_x_eq_30 :
  cost_plan1_variable 20 30 < cost_plan2_variable 20 30 :=
by sorry


end NUMINAMATH_GPT_calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l1890_189083


namespace NUMINAMATH_GPT_eiffel_tower_height_l1890_189041

-- Define the constants for heights and difference
def BurjKhalifa : ℝ := 830
def height_difference : ℝ := 506

-- The goal: Prove that the height of the Eiffel Tower is 324 m.
theorem eiffel_tower_height : BurjKhalifa - height_difference = 324 := 
by 
sorry

end NUMINAMATH_GPT_eiffel_tower_height_l1890_189041


namespace NUMINAMATH_GPT_area_excluding_hole_l1890_189078

theorem area_excluding_hole (x : ℝ) : 
  (2 * x + 8) * (x + 6) - (2 * x - 2) * (x - 1) = 24 * x + 46 :=
by
  sorry

end NUMINAMATH_GPT_area_excluding_hole_l1890_189078


namespace NUMINAMATH_GPT_coordinates_of_A_after_move_l1890_189086

noncomputable def moved_coordinates (a : ℝ) : ℝ × ℝ :=
  let x := 2 * a - 9 + 5
  let y := 1 - 2 * a
  (x, y)

theorem coordinates_of_A_after_move (a : ℝ) (h : moved_coordinates a = (0, 1 - 2 * a)) :
  moved_coordinates 2 = (-5, -3) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_coordinates_of_A_after_move_l1890_189086


namespace NUMINAMATH_GPT_dalton_needs_more_money_l1890_189058

-- Definitions based on the conditions
def jumpRopeCost : ℕ := 7
def boardGameCost : ℕ := 12
def ballCost : ℕ := 4
def savedAllowance : ℕ := 6
def moneyFromUncle : ℕ := 13

-- Computation of how much more money is needed
theorem dalton_needs_more_money : 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  totalCost - totalMoney = 4 := 
by 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  have h1 : totalCost = 23 := by rfl
  have h2 : totalMoney = 19 := by rfl
  calc
    totalCost - totalMoney = 23 - 19 := by rw [h1, h2]
    _ = 4 := by rfl

end NUMINAMATH_GPT_dalton_needs_more_money_l1890_189058


namespace NUMINAMATH_GPT_part1_part2_l1890_189094
open Real

noncomputable def f (x : ℝ) (m : ℝ) := x^2 - m * log x
noncomputable def h (x : ℝ) (a : ℝ) := x^2 - x + a
noncomputable def k (x : ℝ) (a : ℝ) := x - 2 * log x - a

theorem part1 (x : ℝ) (m : ℝ) (h_pos_x : 1 < x) : 
  (f x m) - (h x 0) ≥ 0 → m ≤ exp 1 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x < 2 → k x a < 0) ∧ 
  (k 2 a < 0) ∧ 
  (∀ x, 2 < x ∧ x ≤ 3 → k x a > 0) →
  2 - 2 * log 2 < a ∧ a ≤ 3 - 2 * log 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1890_189094


namespace NUMINAMATH_GPT_cookies_initial_count_l1890_189034

theorem cookies_initial_count (C : ℕ) (h1 : C / 8 = 8) : C = 64 :=
by
  sorry

end NUMINAMATH_GPT_cookies_initial_count_l1890_189034


namespace NUMINAMATH_GPT_train_length_calculation_l1890_189014

noncomputable def length_of_train 
  (time : ℝ) (speed_train : ℝ) (speed_man : ℝ) : ℝ :=
  let speed_relative := speed_train - speed_man
  let speed_relative_mps := speed_relative * (5 / 18)
  speed_relative_mps * time

theorem train_length_calculation :
  length_of_train 29.997600191984642 63 3 = 1666.67 := 
by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l1890_189014


namespace NUMINAMATH_GPT_find_sales_tax_percentage_l1890_189025

noncomputable def salesTaxPercentage (price_with_tax : ℝ) (price_difference : ℝ) : ℝ :=
  (price_difference * 100) / (price_with_tax - price_difference)

theorem find_sales_tax_percentage :
  salesTaxPercentage 2468 161.46 = 7 := by
  sorry

end NUMINAMATH_GPT_find_sales_tax_percentage_l1890_189025


namespace NUMINAMATH_GPT_recurring_fraction_division_l1890_189095

-- Define the values
def x : ℚ := 8 / 11
def y : ℚ := 20 / 11

-- The theorem statement function to prove x / y = 2 / 5
theorem recurring_fraction_division :
  (x / y = (2 : ℚ) / 5) :=
by 
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_recurring_fraction_division_l1890_189095


namespace NUMINAMATH_GPT_completing_square_l1890_189084

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_completing_square_l1890_189084


namespace NUMINAMATH_GPT_eggs_left_for_sunny_side_up_l1890_189018

-- Given conditions:
def ordered_dozen_eggs : ℕ := 3 * 12
def eggs_used_for_crepes (total_eggs : ℕ) : ℕ := total_eggs * 1 / 4
def eggs_after_crepes (total_eggs : ℕ) (used_for_crepes : ℕ) : ℕ := total_eggs - used_for_crepes
def eggs_used_for_cupcakes (remaining_eggs : ℕ) : ℕ := remaining_eggs * 2 / 3
def eggs_left (remaining_eggs : ℕ) (used_for_cupcakes : ℕ) : ℕ := remaining_eggs - used_for_cupcakes

-- Proposition:
theorem eggs_left_for_sunny_side_up : 
  eggs_left (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs)) 
            (eggs_used_for_cupcakes (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs))) = 9 :=
sorry

end NUMINAMATH_GPT_eggs_left_for_sunny_side_up_l1890_189018


namespace NUMINAMATH_GPT_multiplication_identity_l1890_189081

theorem multiplication_identity (x y : ℝ) : 
  (2*x^3 - 5*y^2) * (4*x^6 + 10*x^3*y^2 + 25*y^4) = 8*x^9 - 125*y^6 := 
by
  sorry

end NUMINAMATH_GPT_multiplication_identity_l1890_189081


namespace NUMINAMATH_GPT_f_nonneg_f_positive_f_zero_condition_l1890_189003

noncomputable def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) +
  B * (a^2 * b + b^2 * c + c^2 * a + a * b^2 + b * c^2 + c * a^2) +
  C * a * b * c

theorem f_nonneg (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 ≥ 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

theorem f_positive (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 > 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c > 0 :=
by sorry

theorem f_zero_condition (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 = 0) 
  (h2 : f A B C 1 1 0 > 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

end NUMINAMATH_GPT_f_nonneg_f_positive_f_zero_condition_l1890_189003


namespace NUMINAMATH_GPT_Ryan_stickers_l1890_189007

def Ryan_has_30_stickers (R S T : ℕ) : Prop :=
  S = 3 * R ∧ T = S + 20 ∧ R + S + T = 230 → R = 30

theorem Ryan_stickers : ∃ R S T : ℕ, Ryan_has_30_stickers R S T :=
sorry

end NUMINAMATH_GPT_Ryan_stickers_l1890_189007


namespace NUMINAMATH_GPT_negative_integer_solution_l1890_189031

theorem negative_integer_solution (N : ℤ) (hN : N^2 + N = -12) : N = -3 ∨ N = -4 :=
sorry

end NUMINAMATH_GPT_negative_integer_solution_l1890_189031


namespace NUMINAMATH_GPT_baseball_games_per_month_l1890_189005

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end NUMINAMATH_GPT_baseball_games_per_month_l1890_189005


namespace NUMINAMATH_GPT_curtain_length_correct_l1890_189008

-- Define the problem conditions in Lean
def room_height_feet : ℝ := 8
def feet_to_inches : ℝ := 12
def additional_material_inches : ℝ := 5

-- Define the target length of the curtains
def curtain_length_inches : ℝ :=
  (room_height_feet * feet_to_inches) + additional_material_inches

-- Statement to prove the length of the curtains is 101 inches.
theorem curtain_length_correct :
  curtain_length_inches = 101 := by
  sorry

end NUMINAMATH_GPT_curtain_length_correct_l1890_189008


namespace NUMINAMATH_GPT_boat_distance_downstream_l1890_189045

theorem boat_distance_downstream
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (time_downstream : ℕ)
  (h1 : speed_boat = 22)
  (h2 : speed_stream = 5)
  (h3 : time_downstream = 8) :
  speed_boat + speed_stream * time_downstream = 216 :=
by
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_l1890_189045


namespace NUMINAMATH_GPT_initially_marked_points_l1890_189038

theorem initially_marked_points (k : ℕ) (h : 4 * k - 3 = 101) : k = 26 :=
by
  sorry

end NUMINAMATH_GPT_initially_marked_points_l1890_189038


namespace NUMINAMATH_GPT_sin_double_angle_half_pi_l1890_189024

theorem sin_double_angle_half_pi (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1 / 3) : 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_half_pi_l1890_189024


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1890_189064

theorem arithmetic_sequence_fifth_term (a1 d : ℕ) (a_n : ℕ → ℕ) 
  (h_a1 : a1 = 2) (h_d : d = 1) (h_a_n : ∀ n : ℕ, a_n n = a1 + (n-1) * d) : 
  a_n 5 = 6 := 
    by
    -- Given the conditions, we need to prove a_n evaluated at 5 is equal to 6.
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1890_189064


namespace NUMINAMATH_GPT_minute_hand_coincides_hour_hand_11_times_l1890_189052

noncomputable def number_of_coincidences : ℕ := 11

theorem minute_hand_coincides_hour_hand_11_times :
  ∀ (t : ℝ), (0 < t ∧ t < 12) → ∃(n : ℕ), (1 ≤ n ∧ n ≤ 11) ∧ t = (n * 1 + n * (5 / 11)) :=
sorry

end NUMINAMATH_GPT_minute_hand_coincides_hour_hand_11_times_l1890_189052


namespace NUMINAMATH_GPT_arithmetic_sequence_eleven_term_l1890_189090

theorem arithmetic_sequence_eleven_term (a1 d a11 : ℕ) (h_sum7 : 7 * (2 * a1 + 6 * d) = 154) (h_a1 : a1 = 5) :
  a11 = a1 + 10 * d → a11 = 25 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_eleven_term_l1890_189090


namespace NUMINAMATH_GPT_tangent_line_eq_l1890_189048

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

def derivative_curve (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the problem as a theorem statement
theorem tangent_line_eq (L : ℝ → ℝ) (hL : ∀ x, L x = 2 * x ∨ L x = - x/4) :
  (∀ x, x = 0 → L x = 0) →
  (∀ x x0, L x = curve x → derivative_curve x0 = derivative_curve 0 → x0 = 0 ∨ x0 = 3/2) →
  (L x = 2 * x - curve x ∨ L x = 4 * x + curve x) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1890_189048


namespace NUMINAMATH_GPT_paperclips_in_64_volume_box_l1890_189060

def volume_16 : ℝ := 16
def volume_32 : ℝ := 32
def volume_64 : ℝ := 64
def paperclips_50 : ℝ := 50
def paperclips_100 : ℝ := 100

theorem paperclips_in_64_volume_box :
  ∃ (k p : ℝ), 
  (paperclips_50 = k * volume_16^p) ∧ 
  (paperclips_100 = k * volume_32^p) ∧ 
  (200 = k * volume_64^p) :=
by
  sorry

end NUMINAMATH_GPT_paperclips_in_64_volume_box_l1890_189060


namespace NUMINAMATH_GPT_number_of_girls_l1890_189070

theorem number_of_girls
  (total_pupils : ℕ)
  (boys : ℕ)
  (teachers : ℕ)
  (girls : ℕ)
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36)
  (h4 : girls = total_pupils - boys - teachers) :
  girls = 272 :=
by
  rw [h1, h2, h3] at h4
  exact h4

-- Proof is not required, hence 'sorry' can be used for practical purposes
-- exact sorry

end NUMINAMATH_GPT_number_of_girls_l1890_189070


namespace NUMINAMATH_GPT_find_second_number_l1890_189000

theorem find_second_number (x : ℝ) (h : (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8) : x = 40 :=
sorry

end NUMINAMATH_GPT_find_second_number_l1890_189000


namespace NUMINAMATH_GPT_area_of_park_l1890_189022

theorem area_of_park (x : ℕ) (rate_per_meter : ℝ) (total_cost : ℝ)
  (ratio_len_wid : ℕ × ℕ)
  (h_ratio : ratio_len_wid = (3, 2))
  (h_cost : total_cost = 140)
  (unit_rate : rate_per_meter = 0.50)
  (h_perimeter : 10 * x * rate_per_meter = total_cost) :
  6 * x^2 = 4704 :=
by
  sorry

end NUMINAMATH_GPT_area_of_park_l1890_189022


namespace NUMINAMATH_GPT_number_of_dimes_l1890_189099

-- Definitions based on conditions
def total_coins : Nat := 28
def nickels : Nat := 4

-- Definition of the number of dimes.
def dimes : Nat := total_coins - nickels

-- Theorem statement with the expected answer
theorem number_of_dimes : dimes = 24 := by
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_number_of_dimes_l1890_189099


namespace NUMINAMATH_GPT_S_div_T_is_one_half_l1890_189049

def T (x y z : ℝ) := x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1

def supports (a b c x y z : ℝ) := 
  (x >= a ∧ y >= b ∧ z < c) ∨ 
  (x >= a ∧ z >= c ∧ y < b) ∨ 
  (y >= b ∧ z >= c ∧ x < a)

def S (x y z : ℝ) := T x y z ∧ supports (1/4) (1/4) (1/2) x y z

theorem S_div_T_is_one_half :
  let area_T := 1 -- Normalizing since area of T is in fact √3 / 2 but we care about ratios
  let area_S := 1/2 * area_T -- Given by the problem solution
  area_S / area_T = 1/2 := 
sorry

end NUMINAMATH_GPT_S_div_T_is_one_half_l1890_189049


namespace NUMINAMATH_GPT_son_distance_from_father_is_correct_l1890_189092

noncomputable def distance_between_son_and_father 
  (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1) 
  (incident_point_condition : F / d = L / (d + x) ∧ S / x = F / (d + x)) : ℝ :=
  4.9

theorem son_distance_from_father_is_correct (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1)
  (incident_point_condition : F / d = L / (d + 4.9) ∧ S / 4.9 = F / (d + 4.9)) : 
  distance_between_son_and_father L F S d h_L h_F h_S h_d incident_point_condition = 4.9 :=
sorry

end NUMINAMATH_GPT_son_distance_from_father_is_correct_l1890_189092


namespace NUMINAMATH_GPT_eq_sqrt_pattern_l1890_189042

theorem eq_sqrt_pattern (a t : ℝ) (ha : a = 6) (ht : t = a^2 - 1) (h_pos : 0 < a ∧ 0 < t) :
  a + t = 41 := by
  sorry

end NUMINAMATH_GPT_eq_sqrt_pattern_l1890_189042


namespace NUMINAMATH_GPT_ratio_of_novels_read_l1890_189010

theorem ratio_of_novels_read (jordan_read : ℕ) (alexandre_read : ℕ)
  (h_jordan_read : jordan_read = 120) 
  (h_diff : jordan_read = alexandre_read + 108) :
  alexandre_read / jordan_read = 1 / 10 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_ratio_of_novels_read_l1890_189010


namespace NUMINAMATH_GPT_cab_speed_fraction_l1890_189061

theorem cab_speed_fraction :
  ∀ (S R : ℝ),
    (75 * S = 90 * R) →
    (R / S = 5 / 6) :=
by
  intros S R h
  sorry

end NUMINAMATH_GPT_cab_speed_fraction_l1890_189061


namespace NUMINAMATH_GPT_athlete_speed_l1890_189075

theorem athlete_speed (distance time : ℝ) (h1 : distance = 200) (h2 : time = 25) :
  (distance / time) = 8 := by
  sorry

end NUMINAMATH_GPT_athlete_speed_l1890_189075


namespace NUMINAMATH_GPT_polynomial_roots_identity_l1890_189021

variables {c d : ℂ}

theorem polynomial_roots_identity (hc : c + d = 5) (hd : c * d = 6) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_roots_identity_l1890_189021


namespace NUMINAMATH_GPT_greatest_multiple_less_150_l1890_189051

/-- Define the LCM of two natural numbers -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_multiple_less_150 (x y : ℕ) (h1 : x = 15) (h2 : y = 20) : 
  (∃ m : ℕ, LCM x y * m < 150 ∧ ∀ n : ℕ, LCM x y * n < 150 → LCM x y * n ≤ LCM x y * m) ∧ 
  (∃ m : ℕ, LCM x y * m = 120) :=
by
  sorry

end NUMINAMATH_GPT_greatest_multiple_less_150_l1890_189051


namespace NUMINAMATH_GPT_goats_more_than_pigs_l1890_189096

-- Defining the number of goats
def number_of_goats : ℕ := 66

-- Condition: there are twice as many chickens as goats
def number_of_chickens : ℕ := 2 * number_of_goats

-- Calculating the total number of goats and chickens
def total_goats_and_chickens : ℕ := number_of_goats + number_of_chickens

-- Condition: the number of ducks is half of the total number of goats and chickens
def number_of_ducks : ℕ := total_goats_and_chickens / 2

-- Condition: the number of pigs is a third of the number of ducks
def number_of_pigs : ℕ := number_of_ducks / 3

-- The statement we need to prove
theorem goats_more_than_pigs : number_of_goats - number_of_pigs = 33 := by
  -- The proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_goats_more_than_pigs_l1890_189096


namespace NUMINAMATH_GPT_research_development_success_l1890_189076

theorem research_development_success 
  (P_A : ℝ)  -- probability of Team A successfully developing a product
  (P_B : ℝ)  -- probability of Team B successfully developing a product
  (independent : Bool)  -- independence condition (dummy for clarity)
  (h1 : P_A = 2/3)
  (h2 : P_B = 3/5) 
  (h3 : independent = true) :
  (1 - (1 - P_A) * (1 - P_B) = 13/15) :=
by
  sorry

end NUMINAMATH_GPT_research_development_success_l1890_189076


namespace NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1890_189020

theorem cubic_roots_sum_of_cubes :
  ∀ (a b c : ℝ), 
  (∀ x : ℝ, 9 * x^3 + 14 * x^2 + 2047 * x + 3024 = 0 → (x = a ∨ x = b ∨ x = c)) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = -58198 / 729 :=
by
  intros a b c roota_eqn
  sorry

end NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1890_189020


namespace NUMINAMATH_GPT_minimum_n_value_l1890_189013

def satisfies_terms_condition (n : ℕ) : Prop :=
  (n + 1) * (n + 1) ≥ 2021

theorem minimum_n_value :
  ∃ n : ℕ, n > 0 ∧ satisfies_terms_condition n ∧ ∀ m : ℕ, m > 0 ∧ satisfies_terms_condition m → n ≤ m := by
  sorry

end NUMINAMATH_GPT_minimum_n_value_l1890_189013


namespace NUMINAMATH_GPT_cost_price_article_l1890_189032
-- Importing the required library

-- Definition of the problem
theorem cost_price_article
  (C S C_new S_new : ℝ)
  (h1 : S = 1.05 * C)
  (h2 : C_new = 0.95 * C)
  (h3 : S_new = S - 1)
  (h4 : S_new = 1.045 * C) :
  C = 200 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cost_price_article_l1890_189032


namespace NUMINAMATH_GPT_difference_of_squares_l1890_189089

theorem difference_of_squares (a b : ℕ) (h₁ : a + b = 60) (h₂ : a - b = 14) : a^2 - b^2 = 840 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1890_189089


namespace NUMINAMATH_GPT_triple_hash_72_eq_7_25_l1890_189016

def hash (N : ℝ) : ℝ := 0.5 * N - 1

theorem triple_hash_72_eq_7_25 : hash (hash (hash 72)) = 7.25 :=
by
  sorry

end NUMINAMATH_GPT_triple_hash_72_eq_7_25_l1890_189016


namespace NUMINAMATH_GPT_at_least_one_closed_l1890_189053

theorem at_least_one_closed {T V : Set ℤ} (hT : T.Nonempty) (hV : V.Nonempty) (h_disjoint : ∀ x, x ∈ T → x ∉ V)
  (h_union : ∀ x, x ∈ T ∨ x ∈ V)
  (hT_closed : ∀ a b c, a ∈ T → b ∈ T → c ∈ T → a * b * c ∈ T)
  (hV_closed : ∀ x y z, x ∈ V → y ∈ V → z ∈ V → x * y * z ∈ V) :
  (∀ a b, a ∈ T → b ∈ T → a * b ∈ T) ∨ (∀ x y, x ∈ V → y ∈ V → x * y ∈ V) := sorry

end NUMINAMATH_GPT_at_least_one_closed_l1890_189053


namespace NUMINAMATH_GPT_time_to_reach_julia_via_lee_l1890_189023

theorem time_to_reach_julia_via_lee (d1 d2 d3 : ℕ) (t1 t2 : ℕ) :
  d1 = 2 → 
  t1 = 6 → 
  d3 = 3 → 
  (∀ v, v = d1 / t1) → 
  t2 = d3 / v → 
  t2 = 9 :=
by
  intros h1 h2 h3 hv ht2
  sorry

end NUMINAMATH_GPT_time_to_reach_julia_via_lee_l1890_189023


namespace NUMINAMATH_GPT_height_of_new_TV_l1890_189001

theorem height_of_new_TV 
  (width1 height1 cost1 : ℝ) 
  (width2 cost2 : ℝ) 
  (cost_diff_per_sq_inch : ℝ) 
  (h1 : width1 = 24) 
  (h2 : height1 = 16) 
  (h3 : cost1 = 672) 
  (h4 : width2 = 48) 
  (h5 : cost2 = 1152) 
  (h6 : cost_diff_per_sq_inch = 1) : 
  ∃ height2 : ℝ, height2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_height_of_new_TV_l1890_189001


namespace NUMINAMATH_GPT_divisors_count_30_l1890_189037

theorem divisors_count_30 : 
  (∃ n : ℤ, n > 1 ∧ 30 % n = 0) 
  → 
  (∃ k : ℕ, k = 14) :=
by
  sorry

end NUMINAMATH_GPT_divisors_count_30_l1890_189037


namespace NUMINAMATH_GPT_intersection_eq_l1890_189097

def A : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}
def B : Set ℝ := {y | y ≤ 1}

theorem intersection_eq : A ∩ {y | y ∈ Set.Icc (-2 : ℤ) 1} = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1890_189097


namespace NUMINAMATH_GPT_strictly_decreasing_exponential_l1890_189017

theorem strictly_decreasing_exponential (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → (1/2 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_strictly_decreasing_exponential_l1890_189017


namespace NUMINAMATH_GPT_dot_product_of_a_and_b_l1890_189067

noncomputable def vector_a (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
a

noncomputable def vector_b (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
b

theorem dot_product_of_a_and_b {a b : ℝ × ℝ} 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  (a.1 * b.1 + a.2 * b.2) = -12 := 
sorry

end NUMINAMATH_GPT_dot_product_of_a_and_b_l1890_189067


namespace NUMINAMATH_GPT_factorize_polynomial_l1890_189028

variable (x : ℝ)

theorem factorize_polynomial : 4 * x^3 - 8 * x^2 + 4 * x = 4 * x * (x - 1)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l1890_189028


namespace NUMINAMATH_GPT_plot_length_l1890_189040

-- Define the conditions
def rent_per_acre_per_month : ℝ := 30
def total_rent_per_month : ℝ := 300
def width_feet : ℝ := 1210
def area_acres : ℝ := 10
def square_feet_per_acre : ℝ := 43560

-- Prove that the length of the plot is 360 feet
theorem plot_length (h1 : rent_per_acre_per_month = 30)
                    (h2 : total_rent_per_month = 300)
                    (h3 : width_feet = 1210)
                    (h4 : area_acres = 10)
                    (h5 : square_feet_per_acre = 43560) :
  (area_acres * square_feet_per_acre) / width_feet = 360 := 
by {
  sorry
}

end NUMINAMATH_GPT_plot_length_l1890_189040


namespace NUMINAMATH_GPT_prob_two_more_heads_than_tails_eq_210_1024_l1890_189065

-- Let P be the probability of getting exactly two more heads than tails when flipping 10 coins.
def P (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2^n : ℚ)

theorem prob_two_more_heads_than_tails_eq_210_1024 :
  P 10 6 = 210 / 1024 :=
by
  -- The steps leading to the proof are omitted and hence skipped
  sorry

end NUMINAMATH_GPT_prob_two_more_heads_than_tails_eq_210_1024_l1890_189065


namespace NUMINAMATH_GPT_find_radius_l1890_189029

theorem find_radius 
  (r : ℝ)
  (h1 : ∀ (x y : ℝ), ((x - r) ^ 2 + y ^ 2 = r ^ 2) → (4 * x ^ 2 + 9 * y ^ 2 = 36)) 
  (h2 : (4 * r ^ 2 + 9 * 0 ^ 2 = 36)) 
  (h3 : ∃ r : ℝ, r > 0) : 
  r = (2 * Real.sqrt 5) / 3 :=
sorry

end NUMINAMATH_GPT_find_radius_l1890_189029


namespace NUMINAMATH_GPT_kanul_total_amount_l1890_189073

variable (T : ℝ)
variable (H1 : 3000 + 2000 + 0.10 * T = T)

theorem kanul_total_amount : T = 5555.56 := 
by 
  /- with the conditions given, 
     we can proceed to prove T = 5555.56 -/
  sorry

end NUMINAMATH_GPT_kanul_total_amount_l1890_189073


namespace NUMINAMATH_GPT_min_expression_value_l1890_189069

theorem min_expression_value (m n : ℝ) (h : m - n^2 = 1) : ∃ min_val : ℝ, min_val = 4 ∧ (∀ x y, x - y^2 = 1 → m^2 + 2 * y^2 + 4 * x - 1 ≥ min_val) :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l1890_189069


namespace NUMINAMATH_GPT_div_by_27_l1890_189063

theorem div_by_27 (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
sorry

end NUMINAMATH_GPT_div_by_27_l1890_189063
