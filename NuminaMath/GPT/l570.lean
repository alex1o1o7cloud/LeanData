import Mathlib

namespace NUMINAMATH_GPT_max_non_colored_cubes_l570_57012

open Nat

-- Define the conditions
def isRectangularPrism (length width height volume : ℕ) := length * width * height = volume

-- The theorem stating the equivalent math proof problem
theorem max_non_colored_cubes (length width height : ℕ) (h₁ : isRectangularPrism length width height 1024) :
(length > 2 ∧ width > 2 ∧ height > 2) → (length - 2) * (width - 2) * (height - 2) = 504 := by
  sorry

end NUMINAMATH_GPT_max_non_colored_cubes_l570_57012


namespace NUMINAMATH_GPT_union_of_A_and_B_l570_57068

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_union_of_A_and_B_l570_57068


namespace NUMINAMATH_GPT_approximate_number_of_fish_l570_57035

/-
  In a pond, 50 fish were tagged and returned. 
  Later, in another catch of 50 fish, 2 were tagged. 
  Assuming the proportion of tagged fish in the second catch approximates that of the pond,
  prove that the total number of fish in the pond is approximately 1250.
-/

theorem approximate_number_of_fish (N : ℕ) 
  (tagged_in_pond : ℕ := 50) 
  (total_in_second_catch : ℕ := 50) 
  (tagged_in_second_catch : ℕ := 2) 
  (proportion_approx : tagged_in_second_catch / total_in_second_catch = tagged_in_pond / N) :
  N = 1250 :=
by
  sorry

end NUMINAMATH_GPT_approximate_number_of_fish_l570_57035


namespace NUMINAMATH_GPT_radii_of_cylinder_and_cone_are_equal_l570_57009

theorem radii_of_cylinder_and_cone_are_equal
  (h : ℝ)
  (r : ℝ)
  (V_cylinder : ℝ := π * r^2 * h)
  (V_cone : ℝ := (1/3) * π * r^2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
  r = r :=
by
  sorry

end NUMINAMATH_GPT_radii_of_cylinder_and_cone_are_equal_l570_57009


namespace NUMINAMATH_GPT_yanna_gave_100_l570_57041

/--
Yanna buys 10 shirts at $5 each and 3 pairs of sandals at $3 each, 
and she receives $41 in change. Prove that she gave $100.
-/
theorem yanna_gave_100 :
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  total_cost + change = 100 :=
by
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  show total_cost + change = 100
  sorry

end NUMINAMATH_GPT_yanna_gave_100_l570_57041


namespace NUMINAMATH_GPT_An_integer_and_parity_l570_57038

theorem An_integer_and_parity (k : Nat) (h : k > 0) : 
  ∀ n ≥ 1, ∃ A : Nat, 
   (A = 1 ∨ (∀ A' : Nat, A' = ( (A * n + 2 * (n+1) ^ (2 * k)) / (n+2)))) 
  ∧ (A % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) := 
by 
  sorry

end NUMINAMATH_GPT_An_integer_and_parity_l570_57038


namespace NUMINAMATH_GPT_cross_section_area_ratio_correct_l570_57028

variable (α : ℝ)
noncomputable def cross_section_area_ratio : ℝ := 2 * (Real.cos α)

theorem cross_section_area_ratio_correct (α : ℝ) : 
  cross_section_area_ratio α = 2 * Real.cos α :=
by
  unfold cross_section_area_ratio
  sorry

end NUMINAMATH_GPT_cross_section_area_ratio_correct_l570_57028


namespace NUMINAMATH_GPT_abs_difference_21st_term_l570_57070

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 - 14 * (n - 1)

theorem abs_difference_21st_term :
  |sequence_C 21 - sequence_D 21| = 520 := by
  sorry

end NUMINAMATH_GPT_abs_difference_21st_term_l570_57070


namespace NUMINAMATH_GPT_goods_train_length_is_280_meters_l570_57031

def speed_of_man_train_kmph : ℝ := 80
def speed_of_goods_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 9

theorem goods_train_length_is_280_meters :
  let relative_speed_kmph := speed_of_man_train_kmph + speed_of_goods_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let length_of_goods_train := relative_speed_mps * time_to_pass_seconds
  abs (length_of_goods_train - 280) < 1 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_goods_train_length_is_280_meters_l570_57031


namespace NUMINAMATH_GPT_unique_solution_for_a_l570_57064

theorem unique_solution_for_a (a : ℝ) :
  (∃! x : ℝ, 2 ^ |2 * x - 2| - a * Real.cos (1 - x) = 0) ↔ a = 1 :=
sorry

end NUMINAMATH_GPT_unique_solution_for_a_l570_57064


namespace NUMINAMATH_GPT_min_value_abc_l570_57030

theorem min_value_abc : 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^b % 10 = 4) ∧ (b^c % 10 = 2) ∧ (c^a % 10 = 9) ∧ 
    (a + b + c = 17) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_min_value_abc_l570_57030


namespace NUMINAMATH_GPT_roger_steps_to_minutes_l570_57044

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end NUMINAMATH_GPT_roger_steps_to_minutes_l570_57044


namespace NUMINAMATH_GPT_initial_earning_members_l570_57091

theorem initial_earning_members (n : ℕ) (h1 : (n * 735) - ((n - 1) * 650) = 905) : n = 3 := by
  sorry

end NUMINAMATH_GPT_initial_earning_members_l570_57091


namespace NUMINAMATH_GPT_residue_neg_437_mod_13_l570_57076

theorem residue_neg_437_mod_13 : (-437) % 13 = 5 :=
by
  sorry

end NUMINAMATH_GPT_residue_neg_437_mod_13_l570_57076


namespace NUMINAMATH_GPT_frank_more_miles_than_jim_in_an_hour_l570_57027

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end NUMINAMATH_GPT_frank_more_miles_than_jim_in_an_hour_l570_57027


namespace NUMINAMATH_GPT_raine_change_l570_57001

noncomputable def price_bracelet : ℝ := 15
noncomputable def price_necklace : ℝ := 10
noncomputable def price_mug : ℝ := 20
noncomputable def price_keychain : ℝ := 5

noncomputable def quantity_bracelet : ℕ := 3
noncomputable def quantity_necklace : ℕ := 2
noncomputable def quantity_mug : ℕ := 1
noncomputable def quantity_keychain : ℕ := 4

noncomputable def discount_rate : ℝ := 0.12

noncomputable def amount_given : ℝ := 100

-- The total cost before discount
noncomputable def total_before_discount : ℝ := 
  quantity_bracelet * price_bracelet + 
  quantity_necklace * price_necklace + 
  quantity_mug * price_mug + 
  quantity_keychain * price_keychain

-- The discount amount
noncomputable def discount_amount : ℝ := total_before_discount * discount_rate

-- The final amount Raine has to pay after discount
noncomputable def final_amount : ℝ := total_before_discount - discount_amount

-- The change Raine gets back
noncomputable def change : ℝ := amount_given - final_amount

theorem raine_change : change = 7.60 := 
by sorry

end NUMINAMATH_GPT_raine_change_l570_57001


namespace NUMINAMATH_GPT_kim_total_ounces_l570_57096

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_kim_total_ounces_l570_57096


namespace NUMINAMATH_GPT_solution_set_inequality_l570_57019

theorem solution_set_inequality (x : ℝ) : (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l570_57019


namespace NUMINAMATH_GPT_rate_of_interest_l570_57002

-- Define the conditions
def P : ℝ := 1200
def SI : ℝ := 432
def T (R : ℝ) : ℝ := R

-- Define the statement to be proven
theorem rate_of_interest (R : ℝ) (h : SI = (P * R * T R) / 100) : R = 6 :=
by sorry

end NUMINAMATH_GPT_rate_of_interest_l570_57002


namespace NUMINAMATH_GPT_weight_of_11_25m_rod_l570_57018

noncomputable def weight_per_meter (total_weight : ℝ) (length : ℝ) : ℝ :=
  total_weight / length

def weight_of_rod (weight_per_length : ℝ) (length : ℝ) : ℝ :=
  weight_per_length * length

theorem weight_of_11_25m_rod :
  let total_weight_8m := 30.4
  let length_8m := 8.0
  let length_11_25m := 11.25
  let weight_per_length := weight_per_meter total_weight_8m length_8m
  weight_of_rod weight_per_length length_11_25m = 42.75 :=
by sorry

end NUMINAMATH_GPT_weight_of_11_25m_rod_l570_57018


namespace NUMINAMATH_GPT_find_percentage_l570_57040

theorem find_percentage (P : ℕ) (h1 : P * 64 = 320 * 10) : P = 5 := 
  by
  sorry

end NUMINAMATH_GPT_find_percentage_l570_57040


namespace NUMINAMATH_GPT_foci_equality_ellipse_hyperbola_l570_57067

theorem foci_equality_ellipse_hyperbola (m : ℝ) (h : m > 0) 
  (hl: ∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (4 - m^2)) 
  (hh: ∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (m^2 + 2)) : 
  m = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_foci_equality_ellipse_hyperbola_l570_57067


namespace NUMINAMATH_GPT_correct_equation_l570_57036

-- Define the initial deposit
def initial_deposit : ℝ := 2500

-- Define the total amount after one year with interest tax deducted
def total_amount : ℝ := 2650

-- Define the annual interest rate
variable (x : ℝ)

-- Define the interest tax rate
def interest_tax_rate : ℝ := 0.20

-- Define the equation for the total amount after one year considering the tax
theorem correct_equation :
  initial_deposit * (1 + (1 - interest_tax_rate) * x) = total_amount :=
sorry

end NUMINAMATH_GPT_correct_equation_l570_57036


namespace NUMINAMATH_GPT_cake_stand_cost_calculation_l570_57017

-- Define the constants given in the problem
def flour_cost : ℕ := 5
def money_given : ℕ := 43
def change_received : ℕ := 10

-- Define the cost of the cake stand based on the problem's conditions
def cake_stand_cost : ℕ := (money_given - change_received) - flour_cost

-- The theorem we want to prove
theorem cake_stand_cost_calculation : cake_stand_cost = 28 :=
by
  sorry

end NUMINAMATH_GPT_cake_stand_cost_calculation_l570_57017


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_g7_l570_57049

def f (x : ℝ) : ℝ := x ^ 2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g7 :
  let x1 := 3 + Real.sqrt 2;
  let x2 := 3 - Real.sqrt 2;
  let g1 := g x1;
  let g2 := g x2;
  g (f 7) = g1 + g2 := by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_g7_l570_57049


namespace NUMINAMATH_GPT_find_f_of_2_l570_57088

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x) = (1 + x) / x) : f 2 = 3 :=
sorry

end NUMINAMATH_GPT_find_f_of_2_l570_57088


namespace NUMINAMATH_GPT_complex_number_a_eq_1_l570_57077

theorem complex_number_a_eq_1 
  (a : ℝ) 
  (h : ∃ b : ℝ, (a - b * I) / (1 + I) = 0 + b * I) : 
  a = 1 := 
sorry

end NUMINAMATH_GPT_complex_number_a_eq_1_l570_57077


namespace NUMINAMATH_GPT_cost_of_acai_berry_juice_l570_57053

theorem cost_of_acai_berry_juice (cost_per_litre_cocktail : ℝ)
                                 (cost_per_litre_fruit_juice : ℝ)
                                 (litres_fruit_juice : ℝ)
                                 (litres_acai_juice : ℝ)
                                 (total_cost_cocktail : ℝ)
                                 (cost_per_litre_acai : ℝ) :
  cost_per_litre_cocktail = 1399.45 →
  cost_per_litre_fruit_juice = 262.85 →
  litres_fruit_juice = 34 →
  litres_acai_juice = 22.666666666666668 →
  total_cost_cocktail = (34 + 22.666666666666668) * 1399.45 →
  (litres_fruit_juice * cost_per_litre_fruit_juice + litres_acai_juice * cost_per_litre_acai) = total_cost_cocktail →
  cost_per_litre_acai = 3106.66666666666666 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_acai_berry_juice_l570_57053


namespace NUMINAMATH_GPT_no_couples_next_to_each_other_l570_57003

def factorial (n: Nat): Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (m n p q: Nat): Nat :=
  factorial m - n * factorial (m - 1) + p * factorial (m - 2) - q * factorial (m - 3)

theorem no_couples_next_to_each_other :
  arrangements 7 8 24 32 + 16 * factorial 3 = 1488 :=
by
  -- Here we state that the calculation of special arrangements equals 1488.
  sorry

end NUMINAMATH_GPT_no_couples_next_to_each_other_l570_57003


namespace NUMINAMATH_GPT_boat_distance_against_stream_l570_57011

-- Define the conditions
variable (v_s : ℝ)
variable (speed_still_water : ℝ := 9)
variable (distance_downstream : ℝ := 13)

-- Assert the given condition
axiom condition : speed_still_water + v_s = distance_downstream

-- Prove the required distance against the stream
theorem boat_distance_against_stream : (speed_still_water - (distance_downstream - speed_still_water)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l570_57011


namespace NUMINAMATH_GPT_range_of_x_l570_57026

variable (f : ℝ → ℝ)

def even_function :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

def f_value_at_2 := f 2 = 0

theorem range_of_x (h1 : even_function f) (h2 : monotonically_decreasing f) (h3 : f_value_at_2 f) :
  { x : ℝ | f (x - 1) > 0 } = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_range_of_x_l570_57026


namespace NUMINAMATH_GPT_joker_probability_l570_57095

-- Definition of the problem parameters according to the conditions
def total_cards := 54
def jokers := 2

-- Calculate the probability
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

-- State the theorem that we want to prove
theorem joker_probability : probability jokers total_cards = 1 / 27 := by
  sorry

end NUMINAMATH_GPT_joker_probability_l570_57095


namespace NUMINAMATH_GPT_cube_root_simplification_l570_57050

theorem cube_root_simplification {a b : ℕ} (h : (a * b^(1/3) : ℝ) = (2450 : ℝ)^(1/3)) 
  (a_pos : 0 < a) (b_pos : 0 < b) (h_smallest : ∀ b', 0 < b' → (∃ a', (a' * b'^(1/3) : ℝ) = (2450 : ℝ)^(1/3) → b ≤ b')) :
  a + b = 37 := 
sorry

end NUMINAMATH_GPT_cube_root_simplification_l570_57050


namespace NUMINAMATH_GPT_systematic_sampling_correct_l570_57086

-- Define the conditions for the problem
def num_employees : ℕ := 840
def num_selected : ℕ := 42
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Define systematic sampling interval
def sampling_interval := num_employees / num_selected

-- Define the length of the given interval
def interval_length := interval_end - interval_start + 1

-- The theorem to prove
theorem systematic_sampling_correct :
  (interval_length / sampling_interval) = 12 := sorry

end NUMINAMATH_GPT_systematic_sampling_correct_l570_57086


namespace NUMINAMATH_GPT_odd_n_divides_3n_plus_1_is_1_l570_57087

theorem odd_n_divides_3n_plus_1_is_1 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) (h3 : n ∣ 3^n + 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_odd_n_divides_3n_plus_1_is_1_l570_57087


namespace NUMINAMATH_GPT_arithmetic_sequence_a9_l570_57042

noncomputable def S (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_a9 (a₁ a₁₇ : ℝ) (h1 : S 17 a₁ a₁₇ = 102) : (a₁ + a₁₇) / 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a9_l570_57042


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l570_57065

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := 
  sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l570_57065


namespace NUMINAMATH_GPT_solution_l570_57057

theorem solution (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) 
  : x * y * z = 8 := 
by sorry

end NUMINAMATH_GPT_solution_l570_57057


namespace NUMINAMATH_GPT_randy_total_trees_l570_57062

def mango_trees : ℕ := 60
def coconut_trees : ℕ := mango_trees / 2 - 5
def total_trees (mangos coconuts : ℕ) : ℕ := mangos + coconuts

theorem randy_total_trees : total_trees mango_trees coconut_trees = 85 :=
by
  sorry

end NUMINAMATH_GPT_randy_total_trees_l570_57062


namespace NUMINAMATH_GPT_total_weight_of_onions_l570_57043

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_onions_l570_57043


namespace NUMINAMATH_GPT_total_votes_l570_57000

theorem total_votes (V : ℝ) (C R : ℝ) 
  (hC : C = 0.10 * V)
  (hR1 : R = 0.10 * V + 16000)
  (hR2 : R = 0.90 * V) :
  V = 20000 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l570_57000


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_l570_57082

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_l570_57082


namespace NUMINAMATH_GPT_product_possible_values_l570_57092

theorem product_possible_values (N L M M_5: ℤ) :
  M = L + N → 
  M_5 = M - 8 → 
  ∃ L_5, L_5 = L + 5 ∧ |M_5 - L_5| = 6 →
  N = 19 ∨ N = 7 → 19 * 7 = 133 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_possible_values_l570_57092


namespace NUMINAMATH_GPT_ratio_of_votes_l570_57063

theorem ratio_of_votes (up_votes down_votes : ℕ) (h_up : up_votes = 18) (h_down : down_votes = 4) : (up_votes / Nat.gcd up_votes down_votes) = 9 ∧ (down_votes / Nat.gcd up_votes down_votes) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_votes_l570_57063


namespace NUMINAMATH_GPT_units_digit_13_pow_2003_l570_57074

theorem units_digit_13_pow_2003 : (13 ^ 2003) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_13_pow_2003_l570_57074


namespace NUMINAMATH_GPT_value_of_m_l570_57051

theorem value_of_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 8) (h2 : x1 = 3 * x2) : m = 12 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_value_of_m_l570_57051


namespace NUMINAMATH_GPT_nth_number_eq_l570_57073

noncomputable def nth_number (n : Nat) : ℚ := n / (n^2 + 1)

theorem nth_number_eq (n : Nat) : nth_number n = n / (n^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_nth_number_eq_l570_57073


namespace NUMINAMATH_GPT_correct_operation_l570_57093

theorem correct_operation (x y : ℝ) : (-x - y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 :=
sorry

end NUMINAMATH_GPT_correct_operation_l570_57093


namespace NUMINAMATH_GPT_range_of_alpha_l570_57008

variable {x : ℝ}

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_alpha (x : ℝ) (α : ℝ) (h : α = Real.arctan (3*x^2 - 1)) :
  α ∈ Set.Ico 0 (Real.pi / 2) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
sorry

end NUMINAMATH_GPT_range_of_alpha_l570_57008


namespace NUMINAMATH_GPT_point_divides_segment_in_ratio_l570_57098

theorem point_divides_segment_in_ratio (A B C C1 A1 P : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup P]
  (h1 : AP / PA1 = 3 / 2)
  (h2 : CP / PC1 = 2 / 1) :
  AC1 / C1B = 2 / 3 :=
sorry

end NUMINAMATH_GPT_point_divides_segment_in_ratio_l570_57098


namespace NUMINAMATH_GPT_complex_magnitude_sixth_power_l570_57075

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_sixth_power_l570_57075


namespace NUMINAMATH_GPT_equation_of_tangent_circle_l570_57024

-- Define the point and conditional tangency
def center : ℝ × ℝ := (5, 4)
def tangent_to_x_axis : Prop := true -- Placeholder for the tangency condition, which is encoded in our reasoning

-- Define the proof statement
theorem equation_of_tangent_circle :
  (∀ (x y : ℝ), tangent_to_x_axis → 
  (center = (5, 4)) → 
  ((x - 5) ^ 2 + (y - 4) ^ 2 = 16)) := 
sorry

end NUMINAMATH_GPT_equation_of_tangent_circle_l570_57024


namespace NUMINAMATH_GPT_mat_weaves_problem_l570_57056

theorem mat_weaves_problem (S1 S2: ℕ) (days1 days2: ℕ) (mats1 mats2: ℕ) (H1: S1 = 1)
    (H2: S2 = 8) (H3: days1 = 4) (H4: days2 = 8) (H5: mats1 = 4) (H6: mats2 = 16) 
    (rate_consistency: (mats1 / days1) = (mats2 / days2 / S2)): S1 = 4 := 
by
  sorry

end NUMINAMATH_GPT_mat_weaves_problem_l570_57056


namespace NUMINAMATH_GPT_amount_of_brown_paint_l570_57010

-- Definition of the conditions
def white_paint : ℕ := 20
def green_paint : ℕ := 15
def total_paint : ℕ := 69

-- Theorem statement for the amount of brown paint
theorem amount_of_brown_paint : (total_paint - (white_paint + green_paint)) = 34 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_brown_paint_l570_57010


namespace NUMINAMATH_GPT_find_r_fourth_l570_57005

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_r_fourth_l570_57005


namespace NUMINAMATH_GPT_circle_radius_eq_five_l570_57045

theorem circle_radius_eq_five : 
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 8 * y = 0) → (∃ r : ℝ, ((x - 3)^2 + (y + 4)^2 = r^2) ∧ r = 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_eq_five_l570_57045


namespace NUMINAMATH_GPT_sum_of_possible_x_l570_57094

theorem sum_of_possible_x 
  (x : ℝ)
  (squareSide : ℝ) 
  (rectangleLength : ℝ) 
  (rectangleWidth : ℝ) 
  (areaCondition : (rectangleLength * rectangleWidth) = 3 * (squareSide ^ 2)) : 
  6 + 6.5 = 12.5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_possible_x_l570_57094


namespace NUMINAMATH_GPT_prime_divides_30_l570_57046

theorem prime_divides_30 (p : ℕ) (h_prime : Prime p) (h_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
  sorry

end NUMINAMATH_GPT_prime_divides_30_l570_57046


namespace NUMINAMATH_GPT_number_of_correct_answers_l570_57015

theorem number_of_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = 58 :=
sorry

end NUMINAMATH_GPT_number_of_correct_answers_l570_57015


namespace NUMINAMATH_GPT_miles_driven_before_gas_stop_l570_57054

def total_distance : ℕ := 78
def distance_left : ℕ := 46

theorem miles_driven_before_gas_stop : total_distance - distance_left = 32 := by
  sorry

end NUMINAMATH_GPT_miles_driven_before_gas_stop_l570_57054


namespace NUMINAMATH_GPT_largest_among_four_numbers_l570_57021

theorem largest_among_four_numbers
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : a + b = 1) :
  b > max (max (1/2) (2 * a * b)) (a^2 + b^2) := 
sorry

end NUMINAMATH_GPT_largest_among_four_numbers_l570_57021


namespace NUMINAMATH_GPT_evaluate_expression_at_values_l570_57080

theorem evaluate_expression_at_values :
  let x := 2
  let y := -1
  let z := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
    sorry

end NUMINAMATH_GPT_evaluate_expression_at_values_l570_57080


namespace NUMINAMATH_GPT_odds_against_y_winning_l570_57047

/- 
   Define the conditions: 
   odds_w: odds against W winning is 4:1
   odds_x: odds against X winning is 5:3
-/
def odds_w : ℚ := 4 / 1
def odds_x : ℚ := 5 / 3

/- 
   Calculate the odds against Y winning 
-/
theorem odds_against_y_winning : 
  (4 / (4 + 1)) + (5 / (5 + 3)) < 1 ∧
  (1 - ((4 / (4 + 1)) + (5 / (5 + 3)))) = 17 / 40 ∧
  ((1 - (17 / 40)) / (17 / 40)) = 23 / 17 := by
  sorry

end NUMINAMATH_GPT_odds_against_y_winning_l570_57047


namespace NUMINAMATH_GPT_sandy_total_spent_on_clothes_l570_57025

theorem sandy_total_spent_on_clothes :
  let shorts := 13.99
  let shirt := 12.14 
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := 
by
  sorry

end NUMINAMATH_GPT_sandy_total_spent_on_clothes_l570_57025


namespace NUMINAMATH_GPT_common_points_line_circle_l570_57007

theorem common_points_line_circle (a b : ℝ) :
    (∃ x y : ℝ, x / a + y / b = 1 ∧ x^2 + y^2 = 1) →
    (1 / (a * a) + 1 / (b * b) ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_common_points_line_circle_l570_57007


namespace NUMINAMATH_GPT_same_side_of_line_l570_57072

theorem same_side_of_line (a : ℝ) :
    let point1 := (3, -1)
    let point2 := (-4, -3)
    let line_eq (x y : ℝ) := 3 * x - 2 * y + a
    (line_eq point1.1 point1.2) * (line_eq point2.1 point2.2) > 0 ↔
        (a < -11 ∨ a > 6) := sorry

end NUMINAMATH_GPT_same_side_of_line_l570_57072


namespace NUMINAMATH_GPT_universal_proposition_is_B_l570_57029

theorem universal_proposition_is_B :
  (∀ n : ℤ, (2 * n % 2 = 0)) = True :=
sorry

end NUMINAMATH_GPT_universal_proposition_is_B_l570_57029


namespace NUMINAMATH_GPT_geometric_sequence_20_sum_is_2_pow_20_sub_1_l570_57013

def geometric_sequence_sum_condition (a : ℕ → ℕ) (q : ℕ) : Prop :=
  (a 1 * q + 2 * a 1 = 4) ∧ (a 1 ^ 2 * q ^ 4 = a 1 * q ^ 4)

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) : ℕ :=
  (a 1 * (1 - q ^ 20)) / (1 - q)

theorem geometric_sequence_20_sum_is_2_pow_20_sub_1 (a : ℕ → ℕ) (q : ℕ) 
  (h : geometric_sequence_sum_condition a q) : 
  geometric_sequence_sum a q =  2 ^ 20 - 1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_20_sum_is_2_pow_20_sub_1_l570_57013


namespace NUMINAMATH_GPT_real_solution_unique_l570_57023

theorem real_solution_unique (x : ℝ) (h : x^4 + (2 - x)^4 + 2 * x = 34) : x = 0 :=
sorry

end NUMINAMATH_GPT_real_solution_unique_l570_57023


namespace NUMINAMATH_GPT_probability_roll_2_four_times_in_five_rolls_l570_57048

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end NUMINAMATH_GPT_probability_roll_2_four_times_in_five_rolls_l570_57048


namespace NUMINAMATH_GPT_sum_of_20th_and_30th_triangular_numbers_l570_57016

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_20th_and_30th_triangular_numbers :
  triangular_number 20 + triangular_number 30 = 675 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_20th_and_30th_triangular_numbers_l570_57016


namespace NUMINAMATH_GPT_solve_for_y_l570_57081

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l570_57081


namespace NUMINAMATH_GPT_diamond_value_l570_57078

variable {a b : ℤ}

-- Define the operation diamond following the given condition.
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Define the conditions given in the problem.
axiom h1 : a + b = 10
axiom h2 : a * b = 24

-- State the target theorem.
theorem diamond_value : diamond a b = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_diamond_value_l570_57078


namespace NUMINAMATH_GPT_sophomores_in_seminar_l570_57032

theorem sophomores_in_seminar (P Q x y : ℕ)
  (h1 : P + Q = 50)
  (h2 : x = y)
  (h3 : x = (1 / 5 : ℚ) * P)
  (h4 : y = (1 / 4 : ℚ) * Q) :
  P = 22 :=
by
  sorry

end NUMINAMATH_GPT_sophomores_in_seminar_l570_57032


namespace NUMINAMATH_GPT_polynomials_exist_l570_57033

theorem polynomials_exist (p : ℕ) (hp : Nat.Prime p) :
  ∃ (P Q : Polynomial ℤ),
  ¬(Polynomial.degree P = 0) ∧ ¬(Polynomial.degree Q = 0) ∧
  (∀ n, (Polynomial.coeff (P * Q) n).natAbs % p =
    if n = 0 then 1
    else if n = 4 then 1
    else if n = 2 then p - 2
    else 0) :=
sorry

end NUMINAMATH_GPT_polynomials_exist_l570_57033


namespace NUMINAMATH_GPT_max_10a_3b_15c_l570_57061

theorem max_10a_3b_15c (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) : 
  10 * a + 3 * b + 15 * c ≤ (Real.sqrt 337) / 6 := 
sorry

end NUMINAMATH_GPT_max_10a_3b_15c_l570_57061


namespace NUMINAMATH_GPT_bicycle_profit_theorem_l570_57060

def bicycle_profit_problem : Prop :=
  let CP_A : ℝ := 120
  let SP_C : ℝ := 225
  let profit_percentage_B : ℝ := 0.25
  -- intermediate calculations
  let CP_B : ℝ := SP_C / (1 + profit_percentage_B)
  let SP_A : ℝ := CP_B
  let Profit_A : ℝ := SP_A - CP_A
  let Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100
  -- final statement to prove
  Profit_Percentage_A = 50

theorem bicycle_profit_theorem : bicycle_profit_problem := 
by
  sorry

end NUMINAMATH_GPT_bicycle_profit_theorem_l570_57060


namespace NUMINAMATH_GPT_Q_subset_P_l570_57066

def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | x^2 < 4 }

theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_GPT_Q_subset_P_l570_57066


namespace NUMINAMATH_GPT_clock_correction_l570_57004

def gain_per_day : ℚ := 13 / 4
def hours_per_day : ℕ := 24
def days_passed : ℕ := 9
def extra_hours : ℕ := 8
def total_hours : ℕ := days_passed * hours_per_day + extra_hours
def gain_per_hour : ℚ := gain_per_day / hours_per_day
def total_gain : ℚ := total_hours * gain_per_hour
def required_correction : ℚ := 30.33

theorem clock_correction :
  total_gain = required_correction :=
  by sorry

end NUMINAMATH_GPT_clock_correction_l570_57004


namespace NUMINAMATH_GPT_probability_of_two_hearts_and_three_diff_suits_l570_57020

def prob_two_hearts_and_three_diff_suits (n : ℕ) : ℚ :=
  if n = 5 then 135 / 1024 else 0

theorem probability_of_two_hearts_and_three_diff_suits :
  prob_two_hearts_and_three_diff_suits 5 = 135 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_hearts_and_three_diff_suits_l570_57020


namespace NUMINAMATH_GPT_range_of_x_l570_57085

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l570_57085


namespace NUMINAMATH_GPT_inverse_proportion_decreases_l570_57034

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_decreases_l570_57034


namespace NUMINAMATH_GPT_trajectory_of_point_l570_57079

/-- 
  Given points A and B on the coordinate plane, with |AB|=2, 
  and a moving point P such that the sum of the distances from P
  to points A and B is constantly 2, the trajectory of point P 
  is the line segment AB. 
-/
theorem trajectory_of_point (A B P : ℝ × ℝ) 
  (h_AB : dist A B = 2) 
  (h_sum : dist P A + dist P B = 2) :
  P ∈ segment ℝ A B :=
sorry

end NUMINAMATH_GPT_trajectory_of_point_l570_57079


namespace NUMINAMATH_GPT_find_missing_number_l570_57097

theorem find_missing_number (x : ℝ) (h : 0.00375 * x = 153.75) : x = 41000 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l570_57097


namespace NUMINAMATH_GPT_minimum_value_is_1297_l570_57089

noncomputable def find_minimum_value (a b c n : ℕ) : ℕ :=
  if (a + b ≠ b + c) ∧ (b + c ≠ c + a) ∧ (a + b ≠ c + a) ∧
     ((a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
      (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
      (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) then
    a^2 + b^2 + c^2
  else
    0

theorem minimum_value_is_1297 (a b c n : ℕ) :
  a ≠ b → b ≠ c → c ≠ a → (∃ a b c n, (a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
                                  (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
                                  (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) →
  (∃ a b c, a^2 + b^2 + c^2 = 1297) :=
by sorry

end NUMINAMATH_GPT_minimum_value_is_1297_l570_57089


namespace NUMINAMATH_GPT_total_stones_is_odd_l570_57037

variable (d : ℕ) (total_distance : ℕ)

theorem total_stones_is_odd (h1 : d = 10) (h2 : total_distance = 4800) :
  ∃ (N : ℕ), N % 2 = 1 ∧ total_distance = ((N - 1) * 2 * d) :=
by
  -- Let's denote the number of stones as N
  -- Given dx = 10 and total distance as 4800, we want to show that N is odd and 
  -- satisfies the equation: total_distance = ((N - 1) * 2 * d)
  sorry

end NUMINAMATH_GPT_total_stones_is_odd_l570_57037


namespace NUMINAMATH_GPT_cricket_team_matches_played_in_august_l570_57083

theorem cricket_team_matches_played_in_august
    (M : ℕ)
    (h1 : ∃ W : ℕ, W = 24 * M / 100)
    (h2 : ∃ W : ℕ, W + 70 = 52 * (M + 70) / 100) :
    M = 120 :=
sorry

end NUMINAMATH_GPT_cricket_team_matches_played_in_august_l570_57083


namespace NUMINAMATH_GPT_count_multiples_4_6_10_less_300_l570_57022

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end NUMINAMATH_GPT_count_multiples_4_6_10_less_300_l570_57022


namespace NUMINAMATH_GPT_intersection_M_P_l570_57071

def M : Set ℝ := {0, 1, 2, 3}
def P : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_P : M ∩ P = {0, 1} := 
by
  -- You can fill in the proof here
  sorry

end NUMINAMATH_GPT_intersection_M_P_l570_57071


namespace NUMINAMATH_GPT_justin_and_tim_play_same_game_210_times_l570_57090

def number_of_games_with_justin_and_tim : ℕ :=
  have num_players : ℕ := 12
  have game_size : ℕ := 6
  have justin_and_tim_fixed : ℕ := 2
  have remaining_players : ℕ := num_players - justin_and_tim_fixed
  have players_to_choose : ℕ := game_size - justin_and_tim_fixed
  Nat.choose remaining_players players_to_choose

theorem justin_and_tim_play_same_game_210_times :
  number_of_games_with_justin_and_tim = 210 :=
by sorry

end NUMINAMATH_GPT_justin_and_tim_play_same_game_210_times_l570_57090


namespace NUMINAMATH_GPT_find_FC_l570_57069

theorem find_FC (DC : ℝ) (CB : ℝ) (AB AD ED FC : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (h3 : AB = (1/3) * AD) 
  (h4 : ED = (3/4) * AD) 
  (h5 : FC = 14.625) : FC = 14.625 :=
by sorry

end NUMINAMATH_GPT_find_FC_l570_57069


namespace NUMINAMATH_GPT_find_integer_n_l570_57006

def s : List ℤ := [8, 11, 12, 14, 15]

theorem find_integer_n (n : ℤ) (h : (s.sum + n) / (s.length + 1) = (25 / 100) * (s.sum / s.length) + (s.sum / s.length)) : n = 30 := by
  sorry

end NUMINAMATH_GPT_find_integer_n_l570_57006


namespace NUMINAMATH_GPT_fewer_noodles_than_pirates_l570_57039

theorem fewer_noodles_than_pirates 
  (P : ℕ) (N : ℕ) (h1 : P = 45) (h2 : N + P = 83) : P - N = 7 := by 
  sorry

end NUMINAMATH_GPT_fewer_noodles_than_pirates_l570_57039


namespace NUMINAMATH_GPT_quadratic_minimum_value_proof_l570_57084

-- Define the quadratic function and its properties
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the condition that the coefficient of the squared term is positive
def coefficient_positive : Prop := (2 : ℝ) > 0

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : Prop := h = 3

-- Define the minimum value of the quadratic function
def minimum_value (y_min : ℝ) : Prop := ∀ x : ℝ, y_min ≤ quadratic_function x 

-- Define the correct answer choice
def correct_answer : Prop := minimum_value 2

-- The theorem stating the proof problem
theorem quadratic_minimum_value_proof :
  coefficient_positive ∧ axis_of_symmetry 3 → correct_answer :=
sorry

end NUMINAMATH_GPT_quadratic_minimum_value_proof_l570_57084


namespace NUMINAMATH_GPT_range_of_m_l570_57052

noncomputable def f (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x

def tangents_condition (m : ℝ) : Prop := ∃ x : ℝ, (-3 * x^2 + 12 * x - 9) * (x + 1) + m = -x^3 + 6 * x^2 - 9 * x

theorem range_of_m (m : ℝ) : tangents_condition m → -11 < m ∧ m < 16 :=
sorry

end NUMINAMATH_GPT_range_of_m_l570_57052


namespace NUMINAMATH_GPT_monomials_exponents_l570_57058

theorem monomials_exponents (m n : ℕ) 
  (h₁ : 3 * x ^ 5 * y ^ m + -2 * x ^ n * y ^ 7 = 0) : m - n = 2 := 
by
  sorry

end NUMINAMATH_GPT_monomials_exponents_l570_57058


namespace NUMINAMATH_GPT_sum_of_six_consecutive_odd_numbers_l570_57014

theorem sum_of_six_consecutive_odd_numbers (a b c d e f : ℕ) 
  (ha : 135135 = a * b * c * d * e * f)
  (hb : a < b) (hc : b < c) (hd : c < d) (he : d < e) (hf : e < f)
  (hzero : a % 2 = 1) (hone : b % 2 = 1) (htwo : c % 2 = 1) 
  (hthree : d % 2 = 1) (hfour : e % 2 = 1) (hfive : f % 2 = 1) :
  a + b + c + d + e + f = 48 := by
  sorry

end NUMINAMATH_GPT_sum_of_six_consecutive_odd_numbers_l570_57014


namespace NUMINAMATH_GPT_find_wind_speed_l570_57099

-- Definitions from conditions
def speed_with_wind (j w : ℝ) := (j + w) * 6 = 3000
def speed_against_wind (j w : ℝ) := (j - w) * 9 = 3000

-- Theorem to prove the wind speed is 83.335 mph
theorem find_wind_speed (j w : ℝ) (h1 : speed_with_wind j w) (h2 : speed_against_wind j w) : w = 83.335 :=
by 
  -- Here we would prove the theorem using the given conditions
  sorry

end NUMINAMATH_GPT_find_wind_speed_l570_57099


namespace NUMINAMATH_GPT_fourth_person_height_l570_57059

theorem fourth_person_height 
  (h : ℝ)
  (height_average : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79)
  : h + 10 = 85 := 
by
  sorry

end NUMINAMATH_GPT_fourth_person_height_l570_57059


namespace NUMINAMATH_GPT_expand_and_count_nonzero_terms_l570_57055

theorem expand_and_count_nonzero_terms (x : ℝ) : 
  (x-3)*(3*x^2-2*x+6) + 2*(x^3 + x^2 - 4*x) = 5*x^3 - 9*x^2 + 4*x - 18 ∧ 
  (5 ≠ 0 ∧ -9 ≠ 0 ∧ 4 ≠ 0 ∧ -18 ≠ 0) :=
sorry

end NUMINAMATH_GPT_expand_and_count_nonzero_terms_l570_57055
