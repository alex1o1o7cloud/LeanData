import Mathlib

namespace NUMINAMATH_GPT_number_of_set_B_l1279_127965

theorem number_of_set_B (U A B : Finset ℕ) (hU : U.card = 193) (hA_inter_B : (A ∩ B).card = 25) (hA : A.card = 110) (h_not_in_A_or_B : 193 - (A ∪ B).card = 59) : B.card = 49 := 
by
  sorry

end NUMINAMATH_GPT_number_of_set_B_l1279_127965


namespace NUMINAMATH_GPT_minimum_value_MP_MF_l1279_127901

noncomputable def min_value (M P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := |dist M P + dist M F|

theorem minimum_value_MP_MF :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) →
  ∀ (F : ℝ × ℝ), (F = (1, 0)) →
  ∀ (P : ℝ × ℝ), (P = (3, 1)) →
  min_value M P F = 4 :=
by
  intros M h_para F h_focus P h_fixed
  rw [min_value]
  sorry

end NUMINAMATH_GPT_minimum_value_MP_MF_l1279_127901


namespace NUMINAMATH_GPT_ratio_of_c_to_b_l1279_127977

    theorem ratio_of_c_to_b (a b c : ℤ) (h0 : a = 0) (h1 : a < b) (h2 : b < c)
      (h3 : (a + b + c) / 3 = b / 2) : c / b = 1 / 2 :=
    by
      -- proof steps go here
      sorry
    
end NUMINAMATH_GPT_ratio_of_c_to_b_l1279_127977


namespace NUMINAMATH_GPT_martha_correct_guess_probability_l1279_127967

namespace MarthaGuess

-- Definitions for the conditions
def height_guess_child_accurate : ℚ := 4 / 5
def height_guess_adult_accurate : ℚ := 5 / 6
def weight_guess_tight_clothing_accurate : ℚ := 3 / 4
def weight_guess_loose_clothing_accurate : ℚ := 7 / 10

-- Probabilities of incorrect guesses
def height_guess_child_inaccurate : ℚ := 1 - height_guess_child_accurate
def height_guess_adult_inaccurate : ℚ := 1 - height_guess_adult_accurate
def weight_guess_tight_clothing_inaccurate : ℚ := 1 - weight_guess_tight_clothing_accurate
def weight_guess_loose_clothing_inaccurate : ℚ := 1 - weight_guess_loose_clothing_accurate

-- Combined probability of guessing incorrectly for each case
def incorrect_prob_child_loose : ℚ := height_guess_child_inaccurate * weight_guess_loose_clothing_inaccurate
def incorrect_prob_adult_tight : ℚ := height_guess_adult_inaccurate * weight_guess_tight_clothing_inaccurate
def incorrect_prob_adult_loose : ℚ := height_guess_adult_inaccurate * weight_guess_loose_clothing_inaccurate

-- Total probability of incorrect guesses for all three cases
def total_incorrect_prob : ℚ := incorrect_prob_child_loose * incorrect_prob_adult_tight * incorrect_prob_adult_loose

-- Probability of at least one correct guess
def correct_prob_at_least_once : ℚ := 1 - total_incorrect_prob

-- Main theorem stating the final result
theorem martha_correct_guess_probability : correct_prob_at_least_once = 7999 / 8000 := by
  sorry

end MarthaGuess

end NUMINAMATH_GPT_martha_correct_guess_probability_l1279_127967


namespace NUMINAMATH_GPT_ammonium_chloride_potassium_hydroxide_ammonia_l1279_127907

theorem ammonium_chloride_potassium_hydroxide_ammonia
  (moles_KOH : ℕ) (moles_NH3 : ℕ) (moles_NH4Cl : ℕ) 
  (reaction : moles_KOH = 3 ∧ moles_NH3 = moles_KOH ∧ moles_NH4Cl >= moles_KOH) : 
  moles_NH3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ammonium_chloride_potassium_hydroxide_ammonia_l1279_127907


namespace NUMINAMATH_GPT_first_digit_l1279_127984

-- Definitions and conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

def number (x y : ℕ) : ℕ := 653 * 100 + x * 10 + y

-- Main theorem
theorem first_digit (x y : ℕ) (h₁ : isDivisibleBy (number x y) 80) (h₂ : x + y = 2) : x = 2 :=
sorry

end NUMINAMATH_GPT_first_digit_l1279_127984


namespace NUMINAMATH_GPT_frustum_lateral_surface_area_l1279_127958

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) (hh : h = 5) :
  let d := r1 - r2
  let s := Real.sqrt (h^2 + d^2)
  let A := Real.pi * s * (r1 + r2)
  A = 12 * Real.pi * Real.sqrt 41 :=
by
  -- hr1 and hr2 imply that r1 and r2 are constants, therefore d = 8 - 4 = 4
  -- h = 5 and d = 4 imply s = sqrt (5^2 + 4^2) = sqrt 41
  -- The area A is then pi * sqrt 41 * (8 + 4) = 12 * pi * sqrt 41
  sorry

end NUMINAMATH_GPT_frustum_lateral_surface_area_l1279_127958


namespace NUMINAMATH_GPT_total_scissors_l1279_127917

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_scissors_l1279_127917


namespace NUMINAMATH_GPT_people_left_gym_l1279_127946

theorem people_left_gym (initial : ℕ) (additional : ℕ) (current : ℕ) (H1 : initial = 16) (H2 : additional = 5) (H3 : current = 19) : (initial + additional - current) = 2 :=
by
  sorry

end NUMINAMATH_GPT_people_left_gym_l1279_127946


namespace NUMINAMATH_GPT_correlation_graph_is_scatter_plot_l1279_127952

/-- The definition of a scatter plot graph -/
def scatter_plot_graph (x y : ℝ → ℝ) : Prop := 
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)

/-- Prove that the graph representing a set of data for two variables with a correlation is called a "scatter plot" -/
theorem correlation_graph_is_scatter_plot (x y : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)) → 
  (scatter_plot_graph x y) :=
by
  sorry

end NUMINAMATH_GPT_correlation_graph_is_scatter_plot_l1279_127952


namespace NUMINAMATH_GPT_probability_of_rolling_8_l1279_127992

theorem probability_of_rolling_8 :
  let num_favorable := 5
  let num_total := 36
  let probability := (5 : ℚ) / 36
  probability =
    (num_favorable : ℚ) / num_total :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_8_l1279_127992


namespace NUMINAMATH_GPT_find_range_of_m_l1279_127914

variable (x m : ℝ)

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + (4 * m - 3) > 0

def proposition_q (m : ℝ) : Prop := (∀ m > 2, m + 1 / (m - 2) ≥ 4) ∧ (∃ m, m + 1 / (m - 2) = 4)

def range_m : Set ℝ := {m | 1 < m ∧ m ≤ 2} ∪ {m | m ≥ 3}

theorem find_range_of_m
  (h_p : proposition_p m ∨ ¬proposition_p m)
  (h_q : proposition_q m ∨ ¬proposition_q m)
  (h_exclusive : (proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m))
  : m ∈ range_m := sorry

end NUMINAMATH_GPT_find_range_of_m_l1279_127914


namespace NUMINAMATH_GPT_expand_expression_l1279_127919

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l1279_127919


namespace NUMINAMATH_GPT_find_n_from_binomial_condition_l1279_127910

theorem find_n_from_binomial_condition (n : ℕ) (h : Nat.choose n 3 = 7 * Nat.choose n 1) : n = 43 :=
by
  -- The proof steps would be filled in here
  sorry

end NUMINAMATH_GPT_find_n_from_binomial_condition_l1279_127910


namespace NUMINAMATH_GPT_Walter_receives_49_bananas_l1279_127983

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end NUMINAMATH_GPT_Walter_receives_49_bananas_l1279_127983


namespace NUMINAMATH_GPT_bird_mammal_difference_africa_asia_l1279_127900

noncomputable def bird_families_to_africa := 42
noncomputable def bird_families_to_asia := 31
noncomputable def bird_families_to_south_america := 7

noncomputable def mammal_families_to_africa := 24
noncomputable def mammal_families_to_asia := 18
noncomputable def mammal_families_to_south_america := 15

noncomputable def reptile_families_to_africa := 15
noncomputable def reptile_families_to_asia := 9
noncomputable def reptile_families_to_south_america := 5

-- Calculate the total number of families migrating to Africa, Asia, and South America
noncomputable def total_families_to_africa := bird_families_to_africa + mammal_families_to_africa + reptile_families_to_africa
noncomputable def total_families_to_asia := bird_families_to_asia + mammal_families_to_asia + reptile_families_to_asia
noncomputable def total_families_to_south_america := bird_families_to_south_america + mammal_families_to_south_america + reptile_families_to_south_america

-- Calculate the combined total of bird and mammal families going to Africa
noncomputable def bird_and_mammal_families_to_africa := bird_families_to_africa + mammal_families_to_africa

-- Difference between bird and mammal families to Africa and total animal families to Asia
noncomputable def difference := bird_and_mammal_families_to_africa - total_families_to_asia

theorem bird_mammal_difference_africa_asia : difference = 8 := 
by
  sorry

end NUMINAMATH_GPT_bird_mammal_difference_africa_asia_l1279_127900


namespace NUMINAMATH_GPT_g_value_at_50_l1279_127939

noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

theorem g_value_at_50 :
  (∀ x y : ℝ, 0 < x → 0 < y → 
  (x * g y - y * g x = g (x / y) + x - y)) →
  g 50 = -24.5 :=
by
  intro h
  have h_g : ∀ x : ℝ, 0 < x → g x = (1 - x) / 2 := 
    fun x x_pos => sorry -- g(x) derivation proof goes here
  exact sorry -- Final answer proof goes here

end NUMINAMATH_GPT_g_value_at_50_l1279_127939


namespace NUMINAMATH_GPT_cubic_identity_l1279_127949

variable (a b c : ℝ)
variable (h1 : a + b + c = 13)
variable (h2 : ab + ac + bc = 30)

theorem cubic_identity : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 :=
by 
  sorry

end NUMINAMATH_GPT_cubic_identity_l1279_127949


namespace NUMINAMATH_GPT_pyramid_volume_l1279_127969

noncomputable def volume_pyramid (a b : ℝ) : ℝ :=
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2))

theorem pyramid_volume (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 < 4 * b^2) :
  volume_pyramid a b =
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2)) :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l1279_127969


namespace NUMINAMATH_GPT_water_speed_l1279_127954

theorem water_speed (v : ℝ) (h1 : 4 - v > 0) (h2 : 6 * (4 - v) = 12) : v = 2 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_water_speed_l1279_127954


namespace NUMINAMATH_GPT_bicycle_cost_price_l1279_127938

theorem bicycle_cost_price 
  (CP_A : ℝ) 
  (H : CP_A * (1.20 * 0.85 * 1.30 * 0.90) = 285) : 
  CP_A = 285 / (1.20 * 0.85 * 1.30 * 0.90) :=
sorry

end NUMINAMATH_GPT_bicycle_cost_price_l1279_127938


namespace NUMINAMATH_GPT_midpoint_of_line_segment_on_hyperbola_l1279_127922

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_of_line_segment_on_hyperbola_l1279_127922


namespace NUMINAMATH_GPT_figure_side_length_l1279_127902

theorem figure_side_length (number_of_sides : ℕ) (perimeter : ℝ) (length_of_one_side : ℝ) 
  (h1 : number_of_sides = 8) (h2 : perimeter = 23.6) : length_of_one_side = 2.95 :=
by
  sorry

end NUMINAMATH_GPT_figure_side_length_l1279_127902


namespace NUMINAMATH_GPT_number_divisible_by_23_and_29_l1279_127988

theorem number_divisible_by_23_and_29 (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  23 ∣ (200100 * a + 20010 * b + 2001 * c) ∧ 29 ∣ (200100 * a + 20010 * b + 2001 * c) :=
by
  sorry

end NUMINAMATH_GPT_number_divisible_by_23_and_29_l1279_127988


namespace NUMINAMATH_GPT_actual_discount_is_expected_discount_l1279_127995

-- Define the conditions
def promotional_discount := 20 / 100  -- 20% discount
def vip_card_discount := 10 / 100  -- 10% additional discount

-- Define the combined discount calculation
def combined_discount := (1 - promotional_discount) * (1 - vip_card_discount)

-- Define the expected discount off the original price
def expected_discount := 28 / 100  -- 28% discount

-- Theorem statement proving the combined discount is equivalent to the expected discount
theorem actual_discount_is_expected_discount :
  combined_discount = 1 - expected_discount :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_actual_discount_is_expected_discount_l1279_127995


namespace NUMINAMATH_GPT_compare_probabilities_l1279_127933

noncomputable def box_bad_coin_prob_method_one : ℝ := 1 - (0.99 ^ 10)
noncomputable def box_bad_coin_prob_method_two : ℝ := 1 - ((49 / 50) ^ 5)

theorem compare_probabilities : box_bad_coin_prob_method_one < box_bad_coin_prob_method_two := by
  sorry

end NUMINAMATH_GPT_compare_probabilities_l1279_127933


namespace NUMINAMATH_GPT_expected_rainfall_week_l1279_127973

theorem expected_rainfall_week :
  let P_sun := 0.35
  let P_2 := 0.40
  let P_8 := 0.25
  let rainfall_2 := 2
  let rainfall_8 := 8
  let daily_expected := P_sun * 0 + P_2 * rainfall_2 + P_8 * rainfall_8
  let total_expected := 7 * daily_expected
  total_expected = 19.6 :=
by
  sorry

end NUMINAMATH_GPT_expected_rainfall_week_l1279_127973


namespace NUMINAMATH_GPT_james_take_home_pay_l1279_127962

theorem james_take_home_pay :
  let main_hourly_rate := 20
  let second_hourly_rate := main_hourly_rate - (main_hourly_rate * 0.20)
  let main_hours := 30
  let second_hours := main_hours / 2
  let side_gig_earnings := 100 * 2
  let overtime_hours := 5
  let overtime_rate := main_hourly_rate * 1.5
  let irs_tax_rate := 0.18
  let state_tax_rate := 0.05
  
  -- Main job earnings
  let main_regular_earnings := main_hours * main_hourly_rate
  let main_overtime_earnings := overtime_hours * overtime_rate
  let main_total_earnings := main_regular_earnings + main_overtime_earnings
  
  -- Second job earnings
  let second_total_earnings := second_hours * second_hourly_rate
  
  -- Total earnings before taxes
  let total_earnings := main_total_earnings + second_total_earnings + side_gig_earnings
  
  -- Tax calculations
  let federal_tax := total_earnings * irs_tax_rate
  let state_tax := total_earnings * state_tax_rate
  let total_taxes := federal_tax + state_tax

  -- Total take home pay after taxes
  let take_home_pay := total_earnings - total_taxes

  take_home_pay = 916.30 := 
sorry

end NUMINAMATH_GPT_james_take_home_pay_l1279_127962


namespace NUMINAMATH_GPT_remainder_of_101_pow_37_mod_100_l1279_127913

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_101_pow_37_mod_100_l1279_127913


namespace NUMINAMATH_GPT_find_a33_in_arithmetic_sequence_grid_l1279_127945

theorem find_a33_in_arithmetic_sequence_grid 
  (matrix : ℕ → ℕ → ℕ)
  (rows_are_arithmetic : ∀ i, ∃ a b, ∀ j, matrix i j = a + b * (j - 1))
  (columns_are_arithmetic : ∀ j, ∃ c d, ∀ i, matrix i j = c + d * (i - 1))
  : matrix 3 3 = 31 :=
sorry

end NUMINAMATH_GPT_find_a33_in_arithmetic_sequence_grid_l1279_127945


namespace NUMINAMATH_GPT_chi_square_relationship_l1279_127909

noncomputable def chi_square_statistic {X Y : Type*} (data : X → Y → ℝ) : ℝ := 
  sorry -- Actual definition is omitted for simplicity.

theorem chi_square_relationship (X Y : Type*) (data : X → Y → ℝ) :
  ( ∀ Χ2 : ℝ, Χ2 = chi_square_statistic data →
  (Χ2 = 0 → ∃ (credible : Prop), ¬credible)) → 
  (Χ2 > 0 → ∃ (credible : Prop), credible) :=
sorry

end NUMINAMATH_GPT_chi_square_relationship_l1279_127909


namespace NUMINAMATH_GPT_arithmetic_common_difference_l1279_127921

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  a + (n - 1) * d

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_common_difference (a10 : α) (s10 : α) (d : α) (a1 : α) :
  arithmetic_seq a1 d 10 = a10 →
  sum_arithmetic_seq a1 d 10 = s10 →
  d = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_common_difference_l1279_127921


namespace NUMINAMATH_GPT_find_smallest_angle_l1279_127997

theorem find_smallest_angle 
  (x y : ℝ)
  (hx : x + y = 45)
  (hy : y = x - 5)
  (hz : x > 0 ∧ y > 0 ∧ x + y < 180) :
  min x y = 20 := 
sorry

end NUMINAMATH_GPT_find_smallest_angle_l1279_127997


namespace NUMINAMATH_GPT_polynomial_value_at_five_l1279_127993

def f (x : ℤ) : ℤ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem polynomial_value_at_five : f 5 = 2677 := by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_polynomial_value_at_five_l1279_127993


namespace NUMINAMATH_GPT_color_opposite_orange_is_indigo_l1279_127911

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end NUMINAMATH_GPT_color_opposite_orange_is_indigo_l1279_127911


namespace NUMINAMATH_GPT_failed_by_35_l1279_127923

variables (M S P : ℝ)
variables (hM : M = 153.84615384615384)
variables (hS : S = 45)
variables (hP : P = 0.52 * M)

theorem failed_by_35 (hM : M = 153.84615384615384) (hS : S = 45) (hP : P = 0.52 * M) : P - S = 35 :=
by
  sorry

end NUMINAMATH_GPT_failed_by_35_l1279_127923


namespace NUMINAMATH_GPT_kate_collected_money_l1279_127944

-- Define the conditions
def wand_cost : ℕ := 60
def num_wands_bought : ℕ := 3
def extra_charge : ℕ := 5
def num_wands_sold : ℕ := 2

-- Define the selling price per wand
def selling_price_per_wand : ℕ := wand_cost + extra_charge

-- Define the total amount collected from the sale
def total_collected : ℕ := num_wands_sold * selling_price_per_wand

-- Prove that the total collected is $130
theorem kate_collected_money :
  total_collected = 130 :=
sorry

end NUMINAMATH_GPT_kate_collected_money_l1279_127944


namespace NUMINAMATH_GPT_no_solution_exists_l1279_127929

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end NUMINAMATH_GPT_no_solution_exists_l1279_127929


namespace NUMINAMATH_GPT_max_rectangle_area_l1279_127987

theorem max_rectangle_area (l w : ℕ) (h : 3 * l + 5 * w ≤ 50) : (l * w ≤ 35) :=
by sorry

end NUMINAMATH_GPT_max_rectangle_area_l1279_127987


namespace NUMINAMATH_GPT_vegetable_difference_is_30_l1279_127937

def initial_tomatoes : Int := 17
def initial_carrots : Int := 13
def initial_cucumbers : Int := 8
def initial_bell_peppers : Int := 15
def initial_radishes : Int := 0

def picked_tomatoes : Int := 5
def picked_carrots : Int := 6
def picked_cucumbers : Int := 3
def picked_bell_peppers : Int := 8

def given_neighbor1_tomatoes : Int := 3
def given_neighbor1_carrots : Int := 2

def exchanged_neighbor2_tomatoes : Int := 2
def exchanged_neighbor2_cucumbers : Int := 3
def exchanged_neighbor2_radishes : Int := 5

def given_neighbor3_bell_peppers : Int := 3

noncomputable def initial_total := 
  initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers + initial_radishes

noncomputable def remaining_after_picking :=
  (initial_tomatoes - picked_tomatoes) +
  (initial_carrots - picked_carrots) +
  (initial_cucumbers - picked_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers)

noncomputable def remaining_after_exchanges :=
  ((initial_tomatoes - picked_tomatoes - given_neighbor1_tomatoes - exchanged_neighbor2_tomatoes) +
  (initial_carrots - picked_carrots - given_neighbor1_carrots) +
  (initial_cucumbers - picked_cucumbers - exchanged_neighbor2_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers - given_neighbor3_bell_peppers) +
  exchanged_neighbor2_radishes)

noncomputable def remaining_total := remaining_after_exchanges

noncomputable def total_difference := initial_total - remaining_total

theorem vegetable_difference_is_30 : total_difference = 30 := by
  sorry

end NUMINAMATH_GPT_vegetable_difference_is_30_l1279_127937


namespace NUMINAMATH_GPT_pqrs_sum_l1279_127912

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end NUMINAMATH_GPT_pqrs_sum_l1279_127912


namespace NUMINAMATH_GPT_triangle_angle_sum_l1279_127998

theorem triangle_angle_sum (α β γ : ℝ) (h : α + β + γ = 180) (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
sorry

end NUMINAMATH_GPT_triangle_angle_sum_l1279_127998


namespace NUMINAMATH_GPT_lines_intersect_at_same_point_l1279_127903

theorem lines_intersect_at_same_point : 
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = -3 * x + 4 ∧ y = 4 * x + m) → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_same_point_l1279_127903


namespace NUMINAMATH_GPT_simplify_polynomial_expression_l1279_127953

variable {R : Type*} [CommRing R]

theorem simplify_polynomial_expression (x : R) :
  (2 * x^6 + 3 * x^5 + 4 * x^4 + x^3 + x^2 + x + 20) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 2 * x^2 + 5) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - x^2 + 15 := 
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_expression_l1279_127953


namespace NUMINAMATH_GPT_maximum_of_fraction_l1279_127925

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end NUMINAMATH_GPT_maximum_of_fraction_l1279_127925


namespace NUMINAMATH_GPT_no_such_function_exists_l1279_127920

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ (∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n) :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l1279_127920


namespace NUMINAMATH_GPT_jellybean_probability_l1279_127916

theorem jellybean_probability :
  let total_ways := Nat.choose 15 4
  let red_ways := Nat.choose 5 2
  let blue_ways := Nat.choose 3 2
  let favorable_ways := red_ways * blue_ways
  let probability := favorable_ways / total_ways
  probability = (2 : ℚ) / 91 := by
  sorry

end NUMINAMATH_GPT_jellybean_probability_l1279_127916


namespace NUMINAMATH_GPT_find_perpendicular_slope_value_l1279_127978

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end NUMINAMATH_GPT_find_perpendicular_slope_value_l1279_127978


namespace NUMINAMATH_GPT_solve_for_x_l1279_127951

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  5 * y ^ 2 + 3 * y + 2 = 3 * (8 * x ^ 2 + y + 1) ↔ x = 1 / Real.sqrt 21 ∨ x = -1 / Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1279_127951


namespace NUMINAMATH_GPT_simplify_expression_l1279_127926

theorem simplify_expression (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2) ^ 2) + Real.sqrt ((a - 8) ^ 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1279_127926


namespace NUMINAMATH_GPT_tables_needed_l1279_127948

open Nat

def base7_to_base10 (n : Nat) : Nat := 
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem tables_needed (attendees_base7 : Nat) (attendees_base10 : Nat) (tables : Nat) :
  attendees_base7 = 312 ∧ attendees_base10 = base7_to_base10 attendees_base7 ∧ attendees_base10 = 156 ∧ tables = attendees_base10 / 3 → tables = 52 := 
by
  intros
  sorry

end NUMINAMATH_GPT_tables_needed_l1279_127948


namespace NUMINAMATH_GPT_valid_S2_example_l1279_127964

def satisfies_transformation (S1 S2 : List ℕ) : Prop :=
  S2 = S1.map (λ n => (S1.count n : ℕ))

theorem valid_S2_example : 
  ∃ S1 : List ℕ, satisfies_transformation S1 [1, 2, 1, 1, 2] :=
by
  sorry

end NUMINAMATH_GPT_valid_S2_example_l1279_127964


namespace NUMINAMATH_GPT_isosceles_triangle_sum_x_l1279_127928

noncomputable def sum_possible_values_of_x : ℝ :=
  let x1 : ℝ := 20
  let x2 : ℝ := 50
  let x3 : ℝ := 80
  x1 + x2 + x3

theorem isosceles_triangle_sum_x (x : ℝ) (h1 : x = 20 ∨ x = 50 ∨ x = 80) : sum_possible_values_of_x = 150 :=
  by
    sorry

end NUMINAMATH_GPT_isosceles_triangle_sum_x_l1279_127928


namespace NUMINAMATH_GPT_range_of_m_l1279_127930

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ (∀ x, x = x₁ ∨ x = x₂ ∨ f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)

theorem range_of_m :
  ∀ m : ℝ, has_two_extreme_points (m) ↔ 0 < m ∧ m < 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1279_127930


namespace NUMINAMATH_GPT_sum_of_squares_l1279_127979

theorem sum_of_squares :
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1279_127979


namespace NUMINAMATH_GPT_solve_printer_problem_l1279_127980

noncomputable def printer_problem : Prop :=
  let rate_A := 10
  let rate_B := rate_A + 8
  let rate_C := rate_B - 4
  let combined_rate := rate_A + rate_B + rate_C
  let total_minutes := 20
  let total_pages := combined_rate * total_minutes
  total_pages = 840

theorem solve_printer_problem : printer_problem :=
by
  sorry

end NUMINAMATH_GPT_solve_printer_problem_l1279_127980


namespace NUMINAMATH_GPT_find_geometric_sequence_element_l1279_127918

theorem find_geometric_sequence_element (a b c d e : ℕ) (r : ℚ)
  (h1 : 2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100)
  (h2 : Nat.gcd a e = 1)
  (h3 : r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4)
  : c = 36 :=
  sorry

end NUMINAMATH_GPT_find_geometric_sequence_element_l1279_127918


namespace NUMINAMATH_GPT_rectangle_perimeter_l1279_127957

variable (L W : ℝ)

-- Conditions
def width := 70
def length := (7 / 5) * width

-- Perimeter calculation and proof goal
def perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_perimeter : perimeter (length) (width) = 336 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1279_127957


namespace NUMINAMATH_GPT_distance_travelled_downstream_l1279_127996

def speed_boat_still_water : ℕ := 24
def speed_stream : ℕ := 4
def time_downstream : ℕ := 6

def effective_speed_downstream : ℕ := speed_boat_still_water + speed_stream
def distance_downstream : ℕ := effective_speed_downstream * time_downstream

theorem distance_travelled_downstream : distance_downstream = 168 := by
  sorry

end NUMINAMATH_GPT_distance_travelled_downstream_l1279_127996


namespace NUMINAMATH_GPT_distance_between_poles_l1279_127976

theorem distance_between_poles (length width : ℝ) (num_poles : ℕ) (h_length : length = 90)
  (h_width : width = 40) (h_num_poles : num_poles = 52) : 
  (2 * (length + width)) / (num_poles - 1) = 5.098 := 
by 
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_distance_between_poles_l1279_127976


namespace NUMINAMATH_GPT_manager_salary_l1279_127947

theorem manager_salary 
  (a : ℝ) (n : ℕ) (m_total : ℝ) (new_avg : ℝ) (m_avg_inc : ℝ)
  (h1 : n = 20) 
  (h2 : a = 1600) 
  (h3 : m_avg_inc = 100) 
  (h4 : new_avg = a + m_avg_inc)
  (h5 : m_total = n * a)
  (h6 : new_avg = (m_total + M) / (n + 1)) : 
  M = 3700 :=
by
  sorry

end NUMINAMATH_GPT_manager_salary_l1279_127947


namespace NUMINAMATH_GPT_intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l1279_127975

-- Define the solution sets A and B given conditions
def solution_set_A (a : ℝ) : Set ℝ :=
  { x | |x - 1| ≤ a }

def solution_set_B : Set ℝ :=
  { x | (x - 2) * (x + 2) > 0 }

theorem intersection_A_B_when_a_eq_2 :
  solution_set_A 2 ∩ solution_set_B = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ (a : ℝ), solution_set_A a ∩ solution_set_B = ∅ → 0 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l1279_127975


namespace NUMINAMATH_GPT_find_a_l1279_127974

theorem find_a (a : ℝ) : (∃ k : ℝ, (x - 2) * (x + k) = x^2 + a * x - 5) ↔ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1279_127974


namespace NUMINAMATH_GPT_valid_lineup_count_l1279_127936

noncomputable def num_valid_lineups : ℕ :=
  let total_lineups := Nat.choose 18 8
  let unwanted_lineups := Nat.choose 14 4
  total_lineups - unwanted_lineups

theorem valid_lineup_count : num_valid_lineups = 42757 := by
  sorry

end NUMINAMATH_GPT_valid_lineup_count_l1279_127936


namespace NUMINAMATH_GPT_exam_failure_l1279_127935

structure ExamData where
  max_marks : ℕ
  passing_percentage : ℚ
  secured_marks : ℕ

def passing_marks (data : ExamData) : ℚ :=
  data.passing_percentage * data.max_marks

theorem exam_failure (data : ExamData)
  (h1 : data.max_marks = 150)
  (h2 : data.passing_percentage = 40 / 100)
  (h3 : data.secured_marks = 40) :
  (passing_marks data - data.secured_marks : ℚ) = 20 := by
    sorry

end NUMINAMATH_GPT_exam_failure_l1279_127935


namespace NUMINAMATH_GPT_f_f_is_even_l1279_127982

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end NUMINAMATH_GPT_f_f_is_even_l1279_127982


namespace NUMINAMATH_GPT_value_of_x_l1279_127972

theorem value_of_x (x : ℝ) (h : (0.7 * x) - ((1 / 3) * x) = 110) : x = 300 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1279_127972


namespace NUMINAMATH_GPT_find_common_difference_l1279_127924

def common_difference (S_odd S_even n : ℕ) (d : ℤ) : Prop :=
  S_even - S_odd = n / 2 * d

theorem find_common_difference :
  ∃ d : ℤ, common_difference 132 112 20 d ∧ d = -2 :=
  sorry

end NUMINAMATH_GPT_find_common_difference_l1279_127924


namespace NUMINAMATH_GPT_ozverin_concentration_after_5_times_l1279_127966

noncomputable def ozverin_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

theorem ozverin_concentration_after_5_times :
  ∀ (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ), V = 0.5 → C₀ = 0.4 → v = 50 → n = 5 →
  ozverin_concentration V C₀ v n = 0.236196 :=
by
  intros V C₀ v n hV hC₀ hv hn
  rw [hV, hC₀, hv, hn]
  simp only [ozverin_concentration]
  norm_num
  sorry

end NUMINAMATH_GPT_ozverin_concentration_after_5_times_l1279_127966


namespace NUMINAMATH_GPT_clubs_equal_students_l1279_127985

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end NUMINAMATH_GPT_clubs_equal_students_l1279_127985


namespace NUMINAMATH_GPT_cost_per_meter_l1279_127956

def length_of_plot : ℝ := 75
def cost_of_fencing : ℝ := 5300

-- Define breadth as a variable b
def breadth_of_plot (b : ℝ) : Prop := length_of_plot = b + 50

-- Calculate the perimeter given the known breadth
def perimeter (b : ℝ) : ℝ := 2 * length_of_plot + 2 * b

-- Define the proof problem
theorem cost_per_meter (b : ℝ) (hb : breadth_of_plot b) : 5300 / (perimeter b) = 26.5 := by
  -- Given hb: length_of_plot = b + 50, perimeter calculation follows
  sorry

end NUMINAMATH_GPT_cost_per_meter_l1279_127956


namespace NUMINAMATH_GPT_diameter_of_circle_l1279_127908

theorem diameter_of_circle {a b c d e f D : ℕ} 
  (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (h4 : d = 33) (h5 : e = 56) (h6 : f = 65)
  (h_right_triangle1 : a^2 + b^2 = c^2)
  (h_right_triangle2 : d^2 + e^2 = f^2)
  (h_inscribed_triangles : true) -- This represents that both triangles are inscribed in the circle.
: D = 65 :=
sorry

end NUMINAMATH_GPT_diameter_of_circle_l1279_127908


namespace NUMINAMATH_GPT_five_digit_sine_rule_count_l1279_127986

theorem five_digit_sine_rule_count :
    ∃ (count : ℕ), 
        (∀ (a b c d e : ℕ), 
          (a <  b) ∧
          (b >  c) ∧
          (c >  d) ∧
          (d <  e) ∧
          (a >  d) ∧
          (b >  e) ∧
          (∃ (num : ℕ), num = 10000 * a + 1000 * b + 100 * c + 10 * d + e))
        →
        count = 2892 :=
sorry

end NUMINAMATH_GPT_five_digit_sine_rule_count_l1279_127986


namespace NUMINAMATH_GPT_total_cows_l1279_127931

theorem total_cows (cows : ℕ) (h1 : cows / 3 + cows / 5 + cows / 6 + 12 = cows) : cows = 40 :=
sorry

end NUMINAMATH_GPT_total_cows_l1279_127931


namespace NUMINAMATH_GPT_rotated_number_divisibility_l1279_127906

theorem rotated_number_divisibility 
  (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h : 7 ∣ (10^5 * a1 + 10^4 * a2 + 10^3 * a3 + 10^2 * a4 + 10 * a5 + a6)) :
  7 ∣ (10^5 * a6 + 10^4 * a1 + 10^3 * a2 + 10^2 * a3 + 10 * a4 + a5) := 
sorry

end NUMINAMATH_GPT_rotated_number_divisibility_l1279_127906


namespace NUMINAMATH_GPT_sum_of_lengths_of_legs_of_larger_triangle_l1279_127940

theorem sum_of_lengths_of_legs_of_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypo_small : ℝ)
  (h_area_small : area_small = 18) (h_area_large : area_large = 288) (h_hypo_small : hypo_small = 10) :
  ∃ (sum_legs_large : ℝ), sum_legs_large = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_lengths_of_legs_of_larger_triangle_l1279_127940


namespace NUMINAMATH_GPT_find_vector_coordinates_l1279_127999

structure Point3D :=
  (x y z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
  Point3D.mk (b.x - a.x) (b.y - a.y) (b.z - a.z)

theorem find_vector_coordinates (A B : Point3D)
  (hA : A = { x := 1, y := -3, z := 4 })
  (hB : B = { x := -3, y := 2, z := 1 }) :
  vector_sub A B = { x := -4, y := 5, z := -3 } :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_vector_coordinates_l1279_127999


namespace NUMINAMATH_GPT_speed_of_stream_l1279_127970

theorem speed_of_stream (v : ℝ) : (13 + v) * 4 = 68 → v = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1279_127970


namespace NUMINAMATH_GPT_curve_cartesian_equation_chord_length_l1279_127991
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * θ.cos, ρ * θ.sin)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + 1/2 * t, (Real.sqrt 3) / 2 * t)

theorem curve_cartesian_equation :
  ∀ (ρ θ : ℝ), 
    ρ * θ.sin * θ.sin = 8 * θ.cos →
    (ρ * θ.cos) ^ 2 + (ρ * θ.sin) ^ 2 = 
    8 * (ρ * θ.cos) :=
by sorry

theorem chord_length :
  ∀ (t₁ t₂ : ℝ),
    (3 * t₁^2 - 16 * t₁ - 64 = 0) →
    (3 * t₂^2 - 16 * t₂ - 64 = 0) →
    |t₁ - t₂| = (32 / 3) :=
by sorry

end NUMINAMATH_GPT_curve_cartesian_equation_chord_length_l1279_127991


namespace NUMINAMATH_GPT_x_intersection_difference_l1279_127943

-- Define the conditions
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem x_intersection_difference :
  let x₁ := (1 + Real.sqrt 6) / 5
  let x₂ := (1 - Real.sqrt 6) / 5
  (parabola1 x₁ = parabola2 x₁) → (parabola1 x₂ = parabola2 x₂) →
  (x₁ - x₂) = (2 * Real.sqrt 6) / 5 := 
by
  sorry

end NUMINAMATH_GPT_x_intersection_difference_l1279_127943


namespace NUMINAMATH_GPT_inequality_and_equality_l1279_127959

theorem inequality_and_equality (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c ∧ (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_and_equality_l1279_127959


namespace NUMINAMATH_GPT_candies_markus_l1279_127990

theorem candies_markus (m k s : ℕ) (h_initial_m : m = 9) (h_initial_k : k = 5) (h_total_s : s = 10) :
  (m + s) / 2 = 12 := by
  sorry

end NUMINAMATH_GPT_candies_markus_l1279_127990


namespace NUMINAMATH_GPT_symmetric_about_y_axis_l1279_127905

-- Condition: f is an odd function defined on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given that f is odd and F is defined as specified
theorem symmetric_about_y_axis (f : ℝ → ℝ)
  (hf : odd_function f) :
  ∀ x : ℝ, |f x| + f (|x|) = |f (-x)| + f (|x|) := 
by
  sorry

end NUMINAMATH_GPT_symmetric_about_y_axis_l1279_127905


namespace NUMINAMATH_GPT_count_valid_pairs_l1279_127981

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 3 ∧ ∀ (m n : ℕ), m > n → n ≥ 4 → (m + n) ≤ 40 → (m - n)^2 = m + n → (m, n) ∈ [(10, 6), (15, 10), (21, 15)] := 
by {
  sorry 
}

end NUMINAMATH_GPT_count_valid_pairs_l1279_127981


namespace NUMINAMATH_GPT_translation_coordinates_l1279_127927

variable (A B A1 B1 : ℝ × ℝ)

theorem translation_coordinates
  (hA : A = (-1, 0))
  (hB : B = (1, 2))
  (hA1 : A1 = (2, -1))
  (translation_A : A1 = (A.1 + 3, A.2 - 1))
  (translation_B : B1 = (B.1 + 3, B.2 - 1)) :
  B1 = (4, 1) :=
sorry

end NUMINAMATH_GPT_translation_coordinates_l1279_127927


namespace NUMINAMATH_GPT_ratio_of_pentagon_to_rectangle_l1279_127968

theorem ratio_of_pentagon_to_rectangle (p l : ℕ) 
  (h1 : 5 * p = 30) (h2 : 2 * l + 2 * 5 = 30) : 
  p / l = 3 / 5 :=
by {
  sorry 
}

end NUMINAMATH_GPT_ratio_of_pentagon_to_rectangle_l1279_127968


namespace NUMINAMATH_GPT_girls_in_class_l1279_127963

theorem girls_in_class (k : ℕ) (n_girls n_boys total_students : ℕ)
  (h1 : n_girls = 3 * k) (h2 : n_boys = 4 * k) (h3 : total_students = 35) 
  (h4 : n_girls + n_boys = total_students) : 
  n_girls = 15 :=
by
  -- The proof would normally go here, but is omitted per instructions.
  sorry

end NUMINAMATH_GPT_girls_in_class_l1279_127963


namespace NUMINAMATH_GPT_fraction_to_decimal_l1279_127904

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1279_127904


namespace NUMINAMATH_GPT_correct_statements_l1279_127941

-- A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test.
def statement1 := false -- This statement is incorrect because this is systematic sampling, not stratified sampling.

-- In the frequency distribution histogram, the sum of the areas of all small rectangles is 1.
def statement2 := true -- This is correct.

-- In the regression line equation \(\hat{y} = 0.2x + 12\), when the variable \(x\) increases by one unit, the variable \(y\) definitely increases by 0.2 units.
def statement3 := false -- This is incorrect because y increases on average by 0.2 units, not definitely.

-- For two categorical variables \(X\) and \(Y\), calculating the statistic \(K^2\) and its observed value \(k\), the larger the observed value \(k\), the more confident we are that “X and Y are related”.
def statement4 := true -- This is correct.

-- We need to prove that the correct statements are only statement2 and statement4.
theorem correct_statements : (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) → (statement2 ∧ statement4) :=
by sorry

end NUMINAMATH_GPT_correct_statements_l1279_127941


namespace NUMINAMATH_GPT_kenya_peanuts_l1279_127971

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end NUMINAMATH_GPT_kenya_peanuts_l1279_127971


namespace NUMINAMATH_GPT_typing_speed_ratio_l1279_127915

theorem typing_speed_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.25 * M = 14) : M / T = 2 :=
by
  sorry

end NUMINAMATH_GPT_typing_speed_ratio_l1279_127915


namespace NUMINAMATH_GPT_range_of_a_l1279_127960

theorem range_of_a
  (a : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ (x1 * x2 = 2 * a + 6)) :
  a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1279_127960


namespace NUMINAMATH_GPT_arithmetic_mean_q_r_l1279_127994

theorem arithmetic_mean_q_r (p q r : ℝ) (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) (h3 : r - p = 34) : (q + r) / 2 = 27 :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_q_r_l1279_127994


namespace NUMINAMATH_GPT_solve_for_s_l1279_127955

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * 3^s) 
  (h2 : 45 = m * 9^s) : 
  s = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_s_l1279_127955


namespace NUMINAMATH_GPT_compute_expression_l1279_127989

theorem compute_expression : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : Int) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := 
by 
  sorry

end NUMINAMATH_GPT_compute_expression_l1279_127989


namespace NUMINAMATH_GPT_shuttle_speeds_l1279_127932

def speed_at_altitude (speed_per_sec : ℕ) : ℕ :=
  speed_per_sec * 3600

theorem shuttle_speeds (speed_300 speed_800 avg_speed : ℕ) :
  speed_at_altitude 7 = 25200 ∧ 
  speed_at_altitude 6 = 21600 ∧ 
  avg_speed = (25200 + 21600) / 2 ∧ 
  avg_speed = 23400 := 
by
  sorry

end NUMINAMATH_GPT_shuttle_speeds_l1279_127932


namespace NUMINAMATH_GPT_find_m_l1279_127961

theorem find_m (m : ℕ) (h1 : List ℕ := [27, 32, 39, m, 46, 47])
            (h2 : List ℕ := [30, 31, 34, 41, 42, 45])
            (h3 : (39 + m) / 2 = 42) :
            m = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l1279_127961


namespace NUMINAMATH_GPT_solve_for_a_l1279_127934

theorem solve_for_a (x a : ℤ) (h1 : x = 3) (h2 : x + 2 * a = -1) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1279_127934


namespace NUMINAMATH_GPT_inequality_solution_l1279_127942

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := 3/4 ≤ x ∧ x ≤ 2

-- Theorem statement to prove the equivalence
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ solution_set x := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1279_127942


namespace NUMINAMATH_GPT_min_A_max_B_l1279_127950

-- Part (a): prove A = 15 is the smallest value satisfying the condition
theorem min_A (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : A = 15 := 
sorry

-- Part (b): prove B = 76 is the largest value satisfying the condition
theorem max_B (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : B = 76 := 
sorry

end NUMINAMATH_GPT_min_A_max_B_l1279_127950
