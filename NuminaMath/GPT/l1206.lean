import Mathlib

namespace NUMINAMATH_GPT_product_of_possible_b_values_l1206_120680

theorem product_of_possible_b_values : 
  ∀ b : ℝ, 
    (abs (b - 2) = 2 * (4 - 1)) → 
    (b = 8 ∨ b = -4) → 
    (8 * (-4) = -32) := by
  sorry

end NUMINAMATH_GPT_product_of_possible_b_values_l1206_120680


namespace NUMINAMATH_GPT_principal_amount_borrowed_l1206_120657

theorem principal_amount_borrowed (SI R T : ℝ) (h_SI : SI = 2000) (h_R : R = 4) (h_T : T = 10) : 
    ∃ P, SI = (P * R * T) / 100 ∧ P = 5000 :=
by
    sorry

end NUMINAMATH_GPT_principal_amount_borrowed_l1206_120657


namespace NUMINAMATH_GPT_gcd_840_1764_l1206_120639

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1206_120639


namespace NUMINAMATH_GPT_bens_old_car_cost_l1206_120634

theorem bens_old_car_cost :
  ∃ (O N : ℕ), N = 2 * O ∧ O = 1800 ∧ N = 1800 + 2000 ∧ O = 1900 :=
by 
  sorry

end NUMINAMATH_GPT_bens_old_car_cost_l1206_120634


namespace NUMINAMATH_GPT_jackies_lotion_bottles_l1206_120631

theorem jackies_lotion_bottles (L: ℕ) : 
  (10 + 10) + 6 * L + 12 = 50 → L = 3 :=
by
  sorry

end NUMINAMATH_GPT_jackies_lotion_bottles_l1206_120631


namespace NUMINAMATH_GPT_num_sequences_to_initial_position_8_l1206_120628

def validSequenceCount : ℕ := 4900

noncomputable def numberOfSequencesToInitialPosition (n : ℕ) : ℕ :=
if h : n = 8 then validSequenceCount else 0

theorem num_sequences_to_initial_position_8 :
  numberOfSequencesToInitialPosition 8 = 4900 :=
by
  sorry

end NUMINAMATH_GPT_num_sequences_to_initial_position_8_l1206_120628


namespace NUMINAMATH_GPT_magnification_proof_l1206_120649

-- Define the conditions: actual diameter of the tissue and diameter of the magnified image
def actual_diameter := 0.0002
def magnified_diameter := 0.2

-- Define the magnification factor
def magnification_factor := magnified_diameter / actual_diameter

-- Prove that the magnification factor is 1000
theorem magnification_proof : magnification_factor = 1000 := by
  unfold magnification_factor
  unfold magnified_diameter
  unfold actual_diameter
  norm_num
  sorry

end NUMINAMATH_GPT_magnification_proof_l1206_120649


namespace NUMINAMATH_GPT_calculate_dollar_value_l1206_120622

def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y - 5

theorem calculate_dollar_value : dollar 3 (-1) = -5 := by
  sorry

end NUMINAMATH_GPT_calculate_dollar_value_l1206_120622


namespace NUMINAMATH_GPT_y_value_l1206_120615

-- Given conditions
variables (x y : ℝ)
axiom h1 : x - y = 20
axiom h2 : x + y = 14

-- Prove that y = -3
theorem y_value : y = -3 :=
by { sorry }

end NUMINAMATH_GPT_y_value_l1206_120615


namespace NUMINAMATH_GPT_parabola_opens_downwards_l1206_120611

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end NUMINAMATH_GPT_parabola_opens_downwards_l1206_120611


namespace NUMINAMATH_GPT_power_of_54_l1206_120692

theorem power_of_54 (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
(h_eq : 54^a = a^b) : ∃ k : ℕ, a = 54^k := by
  sorry

end NUMINAMATH_GPT_power_of_54_l1206_120692


namespace NUMINAMATH_GPT_midpoint_pentagon_inequality_l1206_120648

noncomputable def pentagon_area_midpoints (T : ℝ) : ℝ := sorry

theorem midpoint_pentagon_inequality {T t : ℝ} 
  (h1 : t = pentagon_area_midpoints T)
  (h2 : 0 < T) : 
  (3/4) * T > t ∧ t > (1/2) * T :=
  sorry

end NUMINAMATH_GPT_midpoint_pentagon_inequality_l1206_120648


namespace NUMINAMATH_GPT_two_digit_sequence_partition_property_l1206_120682

theorem two_digit_sequence_partition_property :
  ∀ (A B : Set ℕ), (A ∪ B = {x | x < 100 ∧ x % 10 < 10}) →
  ∃ (C : Set ℕ), (C = A ∨ C = B) ∧ 
  ∃ (lst : List ℕ), (∀ (x : ℕ), x ∈ lst → x ∈ C) ∧ 
  (∀ (x y : ℕ), (x, y) ∈ lst.zip lst.tail → (y = x + 1 ∨ y = x + 10 ∨ y = x + 11)) :=
by
  intros A B partition_condition
  sorry

end NUMINAMATH_GPT_two_digit_sequence_partition_property_l1206_120682


namespace NUMINAMATH_GPT_tan_add_pi_over_3_l1206_120640

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + (Real.pi / 3)) = -(6 * Real.sqrt 3 + 2) / 13 := 
by
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_3_l1206_120640


namespace NUMINAMATH_GPT_syllogistic_reasoning_problem_l1206_120681

theorem syllogistic_reasoning_problem
  (H1 : ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I)
  (H2 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.re z = 2)
  (H3 : ∀ (z : ℂ), z = 2 + 3 * Complex.I → Complex.im z = 3) :
  (¬ ∀ (z : ℂ), ∃ (a b : ℝ), z = a + b * Complex.I) → "The conclusion is wrong due to the incorrect major premise" = "A" :=
sorry

end NUMINAMATH_GPT_syllogistic_reasoning_problem_l1206_120681


namespace NUMINAMATH_GPT_fraction_subtraction_l1206_120647

theorem fraction_subtraction : (18 : ℚ) / 45 - (3 : ℚ) / 8 = (1 : ℚ) / 40 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1206_120647


namespace NUMINAMATH_GPT_find_four_digit_numbers_l1206_120662

noncomputable def four_digit_number_permutations_sum (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) : Prop :=
  6 * (x + y + z + t) * (1000 + 100 + 10 + 1) = 10 * (1111 * x)

theorem find_four_digit_numbers (x y z t : ℕ) (distinct : x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t) (nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ t ≠ 0) :
  four_digit_number_permutations_sum x y z t distinct nonzero :=
  sorry

end NUMINAMATH_GPT_find_four_digit_numbers_l1206_120662


namespace NUMINAMATH_GPT_nonneg_int_values_of_fraction_condition_l1206_120675

theorem nonneg_int_values_of_fraction_condition (n : ℕ) : (∃ k : ℤ, 30 * n + 2 = k * (12 * n + 1)) → n = 0 := by
  sorry

end NUMINAMATH_GPT_nonneg_int_values_of_fraction_condition_l1206_120675


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_book_l1206_120661

theorem number_of_ways_to_choose_one_book:
  let chinese_books := 10
  let english_books := 7
  let mathematics_books := 5
  chinese_books + english_books + mathematics_books = 22 := by
    -- The actual proof should go here.
    sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_one_book_l1206_120661


namespace NUMINAMATH_GPT_distance_interval_l1206_120623

def distance_to_town (d : ℝ) : Prop :=
  ¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 9)

theorem distance_interval (d : ℝ) : distance_to_town d → d ∈ Set.Ioo 7 8 :=
by
  intro h
  have h1 : d < 8 := by sorry
  have h2 : d > 7 := by sorry
  rw [Set.mem_Ioo]
  exact ⟨h2, h1⟩

end NUMINAMATH_GPT_distance_interval_l1206_120623


namespace NUMINAMATH_GPT_toy_cost_price_and_profit_l1206_120673

-- Define the cost price of type A toy
def cost_A (x : ℝ) : ℝ := x

-- Define the cost price of type B toy
def cost_B (x : ℝ) : ℝ := 1.5 * x

-- Spending conditions
def spending_A (x : ℝ) (num_A : ℝ) : Prop := num_A = 1200 / x
def spending_B (x : ℝ) (num_B : ℝ) : Prop := num_B = 1500 / (1.5 * x)

-- Quantity difference condition
def quantity_difference (num_A num_B : ℝ) : Prop := num_A - num_B = 20

-- Selling prices
def selling_price_A : ℝ := 12
def selling_price_B : ℝ := 20

-- Total toys purchased condition
def total_toys (num_A num_B : ℝ) : Prop := num_A + num_B = 75

-- Profit condition
def profit_condition (num_A num_B cost_A cost_B : ℝ) : Prop :=
  (selling_price_A - cost_A) * num_A + (selling_price_B - cost_B) * num_B ≥ 300

theorem toy_cost_price_and_profit :
  ∃ (x : ℝ), 
  cost_A x = 10 ∧
  cost_B x = 15 ∧
  ∀ (num_A num_B : ℝ),
  spending_A x num_A →
  spending_B x num_B →
  quantity_difference num_A num_B →
  total_toys num_A num_B →
  profit_condition num_A num_B (cost_A x) (cost_B x) →
  num_A ≤ 25 :=
by
  sorry

end NUMINAMATH_GPT_toy_cost_price_and_profit_l1206_120673


namespace NUMINAMATH_GPT_find_c_and_general_formula_l1206_120653

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) := ∀ n : ℕ, a (n + 1) = a n + c * 2^n

theorem find_c_and_general_formula : 
  ∀ (c : ℕ) (a : ℕ → ℕ),
    (a 1 = 2) →
    (seq a c) →
    ((a 3) = (a 1) * ((a 2) / (a 1))^2) →
    ((a 2) = (a 1) * (a 2) / (a 1)) →
    c = 1 ∧ (∀ n, a n = 2^n) := 
by
  sorry

end NUMINAMATH_GPT_find_c_and_general_formula_l1206_120653


namespace NUMINAMATH_GPT_Robin_total_distance_walked_l1206_120650

-- Define the conditions
def distance_house_to_city_center := 500
def distance_walked_initially := 200

-- Define the proof problem
theorem Robin_total_distance_walked :
  distance_walked_initially * 2 + distance_house_to_city_center = 900 := by
  sorry

end NUMINAMATH_GPT_Robin_total_distance_walked_l1206_120650


namespace NUMINAMATH_GPT_num_solutions_abcd_eq_2020_l1206_120676

theorem num_solutions_abcd_eq_2020 :
  ∃ S : Finset (ℕ × ℕ × ℕ × ℕ), 
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔ (a^2 + b^2) * (c^2 - d^2) = 2020 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧
    S.card = 6 :=
sorry

end NUMINAMATH_GPT_num_solutions_abcd_eq_2020_l1206_120676


namespace NUMINAMATH_GPT_jenny_spent_fraction_l1206_120644

theorem jenny_spent_fraction
  (x : ℝ) -- The original amount of money Jenny had
  (h_half_x : 1/2 * x = 21) -- Half of the original amount is $21
  (h_left_money : x - 24 = 24) -- Jenny had $24 left after spending
  : (x - 24) / x = 3 / 7 := sorry

end NUMINAMATH_GPT_jenny_spent_fraction_l1206_120644


namespace NUMINAMATH_GPT_boys_girls_ratio_l1206_120613

theorem boys_girls_ratio (T G : ℕ) (h : (1/2 : ℚ) * G = (1/6 : ℚ) * T) :
  ((T - G) : ℚ) / G = 2 :=
by 
  sorry

end NUMINAMATH_GPT_boys_girls_ratio_l1206_120613


namespace NUMINAMATH_GPT_quadratic_no_solution_l1206_120629

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_no_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  0 < a ∧ discriminant a b c ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_solution_l1206_120629


namespace NUMINAMATH_GPT_volume_inhaled_per_breath_is_correct_l1206_120659

def breaths_per_minute : ℤ := 17
def volume_inhaled_24_hours : ℤ := 13600
def minutes_per_hour : ℤ := 60
def hours_per_day : ℤ := 24

def total_minutes_24_hours : ℤ := hours_per_day * minutes_per_hour
def total_breaths_24_hours : ℤ := total_minutes_24_hours * breaths_per_minute
def volume_per_breath := (volume_inhaled_24_hours : ℚ) / (total_breaths_24_hours : ℚ)

theorem volume_inhaled_per_breath_is_correct :
  volume_per_breath = 0.5556 := by
  sorry

end NUMINAMATH_GPT_volume_inhaled_per_breath_is_correct_l1206_120659


namespace NUMINAMATH_GPT_approx_values_relationship_l1206_120633

theorem approx_values_relationship : 
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a = b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a > b) ∧
  (∃ a b : ℝ, 2.35 ≤ a ∧ a ≤ 2.44 ∧ 2.395 ≤ b ∧ b ≤ 2.404 ∧ a < b) :=
by sorry

end NUMINAMATH_GPT_approx_values_relationship_l1206_120633


namespace NUMINAMATH_GPT_calculate_g_inv_l1206_120643

noncomputable def g : ℤ → ℤ := sorry
noncomputable def g_inv : ℤ → ℤ := sorry

axiom g_inv_eq : ∀ x, g (g_inv x) = x

axiom cond1 : g (-1) = 2
axiom cond2 : g (0) = 3
axiom cond3 : g (1) = 6

theorem calculate_g_inv : 
  g_inv (g_inv 6 - g_inv 2) = -1 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_calculate_g_inv_l1206_120643


namespace NUMINAMATH_GPT_roger_has_more_candy_l1206_120642

-- Defining the conditions
def sandra_bag1 : Nat := 6
def sandra_bag2 : Nat := 6
def roger_bag1 : Nat := 11
def roger_bag2 : Nat := 3

-- Calculating the total pieces of candy for Sandra and Roger
def total_sandra : Nat := sandra_bag1 + sandra_bag2
def total_roger : Nat := roger_bag1 + roger_bag2

-- Statement of the proof problem
theorem roger_has_more_candy : total_roger - total_sandra = 2 := by
  sorry

end NUMINAMATH_GPT_roger_has_more_candy_l1206_120642


namespace NUMINAMATH_GPT_geometric_series_sum_l1206_120664

theorem geometric_series_sum : 
    ∑' n : ℕ, (1 : ℝ) * (-1 / 2) ^ n = 2 / 3 :=
by
    sorry

end NUMINAMATH_GPT_geometric_series_sum_l1206_120664


namespace NUMINAMATH_GPT_fraction_habitable_l1206_120630

theorem fraction_habitable : (1 / 3) * (1 / 3) = 1 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_habitable_l1206_120630


namespace NUMINAMATH_GPT_sqrt_inequality_l1206_120687

open Real

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z) 
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  sqrt (x + y + z) ≥ sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) :=
sorry

end NUMINAMATH_GPT_sqrt_inequality_l1206_120687


namespace NUMINAMATH_GPT_sin_2x_equals_neg_61_div_72_l1206_120671

variable (x y : Real)
variable (h1 : Real.sin y = (3 / 2) * Real.sin x + (2 / 3) * Real.cos x)
variable (h2 : Real.cos y = (2 / 3) * Real.sin x + (3 / 2) * Real.cos x)

theorem sin_2x_equals_neg_61_div_72 : Real.sin (2 * x) = -61 / 72 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_2x_equals_neg_61_div_72_l1206_120671


namespace NUMINAMATH_GPT_amount_of_water_in_first_tank_l1206_120641

theorem amount_of_water_in_first_tank 
  (C : ℝ)
  (H1 : 0 < C)
  (H2 : 0.45 * C = 450)
  (water_in_first_tank : ℝ)
  (water_in_second_tank : ℝ := 450)
  (additional_water_needed : ℝ := 1250)
  (total_capacity : ℝ := 2 * C)
  (total_water_needed : ℝ := 2000) : 
  water_in_first_tank = 300 :=
by 
  sorry

end NUMINAMATH_GPT_amount_of_water_in_first_tank_l1206_120641


namespace NUMINAMATH_GPT_barneys_grocery_store_items_left_l1206_120656

theorem barneys_grocery_store_items_left 
    (ordered_items : ℕ) 
    (sold_items : ℕ) 
    (storeroom_items : ℕ) 
    (damaged_percentage : ℝ)
    (h1 : ordered_items = 4458) 
    (h2 : sold_items = 1561) 
    (h3 : storeroom_items = 575) 
    (h4 : damaged_percentage = 5/100) : 
    ordered_items - (sold_items + ⌊damaged_percentage * ordered_items⌋) + storeroom_items = 3250 :=
by
    sorry

end NUMINAMATH_GPT_barneys_grocery_store_items_left_l1206_120656


namespace NUMINAMATH_GPT_closest_vector_l1206_120668

open Real

def u (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, -4 + 7 * s, 2 + 4 * s)
def b : ℝ × ℝ × ℝ := (5, 1, -3)
def direction : ℝ × ℝ × ℝ := (3, 7, 4)

theorem closest_vector (s : ℝ) :
  (u s - b) • direction = 0 ↔ s = 27 / 74 :=
sorry

end NUMINAMATH_GPT_closest_vector_l1206_120668


namespace NUMINAMATH_GPT_magic_8_ball_probability_l1206_120670

theorem magic_8_ball_probability :
  let p_pos := 1 / 3
  let p_neg := 2 / 3
  let n := 6
  let k := 3
  (Nat.choose n k * (p_pos ^ k) * (p_neg ^ (n - k)) = 160 / 729) :=
by
  sorry

end NUMINAMATH_GPT_magic_8_ball_probability_l1206_120670


namespace NUMINAMATH_GPT_variance_of_data_set_l1206_120691

open Real

def dataSet := [11, 12, 15, 18, 13, 15]

theorem variance_of_data_set :
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  variance = 16 / 3 :=
by
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  have h : mean = 14 := sorry
  have h_variance : variance = 16 / 3 := sorry
  exact h_variance

end NUMINAMATH_GPT_variance_of_data_set_l1206_120691


namespace NUMINAMATH_GPT_sugar_needed_in_two_minutes_l1206_120678

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end NUMINAMATH_GPT_sugar_needed_in_two_minutes_l1206_120678


namespace NUMINAMATH_GPT_order_of_magnitudes_l1206_120602

theorem order_of_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : x < x^(x^x) ∧ x^(x^x) < x^x :=
by
  -- Definitions for y and z.
  let y := x^x
  let z := x^(x^x)
  have h1 : x < y := sorry
  have h2 : z < y := sorry
  have h3 : x < z := sorry
  exact ⟨h3, h2⟩

end NUMINAMATH_GPT_order_of_magnitudes_l1206_120602


namespace NUMINAMATH_GPT_dogs_with_pointy_ears_l1206_120654

theorem dogs_with_pointy_ears (total_dogs with_spots with_pointy_ears: ℕ) 
  (h1: with_spots = total_dogs / 2)
  (h2: total_dogs = 30) :
  with_pointy_ears = total_dogs / 5 :=
by
  sorry

end NUMINAMATH_GPT_dogs_with_pointy_ears_l1206_120654


namespace NUMINAMATH_GPT_artist_used_17_ounces_of_paint_l1206_120618

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end NUMINAMATH_GPT_artist_used_17_ounces_of_paint_l1206_120618


namespace NUMINAMATH_GPT_perpendicular_lines_k_value_l1206_120620

theorem perpendicular_lines_k_value (k : ℚ) : (∀ x y : ℚ, y = 3 * x + 7) ∧ (∀ x y : ℚ, 4 * y + k * x = 4) → k = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_k_value_l1206_120620


namespace NUMINAMATH_GPT_cube_root_inequality_l1206_120660

theorem cube_root_inequality {a b : ℝ} (h : a > b) : (a^(1/3)) > (b^(1/3)) :=
sorry

end NUMINAMATH_GPT_cube_root_inequality_l1206_120660


namespace NUMINAMATH_GPT_find_m_l1206_120624

-- Define points O, A, B, C
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (1, 5)
def C (m : ℝ) : (ℝ × ℝ) := (m, 3)

-- Define vectors AB and OC
def vector_AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)  -- (B - A)
def vector_OC (m : ℝ) : (ℝ × ℝ) := (m, 3)  -- (C - O)

-- Define the dot product
def dot_product (v₁ v₂ : (ℝ × ℝ)) : ℝ := (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem: vector_AB ⊥ vector_OC implies m = 6
theorem find_m (m : ℝ) (h : dot_product vector_AB (vector_OC m) = 0) : m = 6 :=
by
  -- Proof part not required
  sorry

end NUMINAMATH_GPT_find_m_l1206_120624


namespace NUMINAMATH_GPT_sin_2x_value_l1206_120616

theorem sin_2x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : Real.sin (2 * x) = 7 / 9 := by
  sorry

end NUMINAMATH_GPT_sin_2x_value_l1206_120616


namespace NUMINAMATH_GPT_minimum_value_exists_l1206_120651

theorem minimum_value_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_condition : x + 4 * y = 2) : 
  ∃ z : ℝ, z = (x + 40 * y + 4) / (3 * x * y) ∧ z ≥ 18 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_exists_l1206_120651


namespace NUMINAMATH_GPT_initial_weight_of_mixture_eq_20_l1206_120604

theorem initial_weight_of_mixture_eq_20
  (W : ℝ) (h1 : 0.1 * W + 4 = 0.25 * (W + 4)) :
  W = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_weight_of_mixture_eq_20_l1206_120604


namespace NUMINAMATH_GPT_geom_seq_result_l1206_120674

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ)

-- Conditions
axiom h1 : a 1 + a 3 = 5 / 2
axiom h2 : a 2 + a 4 = 5 / 4

-- General properties
axiom geom_seq_common_ratio : ∃ q : ℚ, ∀ n, a (n + 1) = a n * q

-- Sum of the first n terms of the geometric sequence
axiom S_def : S n = (2 * (1 - (1 / 2)^n)) / (1 - 1 / 2)

-- General term of the geometric sequence
axiom a_n_def : a n = 2 * (1 / 2)^(n - 1)

-- Result to be proved
theorem geom_seq_result : S n / a n = 2^n - 1 := 
  by sorry

end NUMINAMATH_GPT_geom_seq_result_l1206_120674


namespace NUMINAMATH_GPT_xiao_ming_runs_distance_l1206_120697

theorem xiao_ming_runs_distance 
  (num_trees : ℕ) 
  (first_tree : ℕ) 
  (last_tree : ℕ) 
  (distance_between_trees : ℕ) 
  (gap_count : ℕ) 
  (total_distance : ℕ)
  (h1 : num_trees = 200) 
  (h2 : first_tree = 1) 
  (h3 : last_tree = 200) 
  (h4 : distance_between_trees = 6) 
  (h5 : gap_count = last_tree - first_tree)
  (h6 : total_distance = gap_count * distance_between_trees) :
  total_distance = 1194 :=
sorry

end NUMINAMATH_GPT_xiao_ming_runs_distance_l1206_120697


namespace NUMINAMATH_GPT_olivers_friend_gave_l1206_120672

variable (initial_amount saved_amount spent_frisbee spent_puzzle final_amount : ℕ) 

theorem olivers_friend_gave (h1 : initial_amount = 9) 
                           (h2 : saved_amount = 5) 
                           (h3 : spent_frisbee = 4) 
                           (h4 : spent_puzzle = 3) 
                           (h5 : final_amount = 15) : 
                           final_amount - (initial_amount + saved_amount - (spent_frisbee + spent_puzzle)) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_olivers_friend_gave_l1206_120672


namespace NUMINAMATH_GPT_faye_earned_total_money_l1206_120626

def bead_necklaces : ℕ := 3
def gem_necklaces : ℕ := 7
def price_per_necklace : ℕ := 7

theorem faye_earned_total_money :
  (bead_necklaces + gem_necklaces) * price_per_necklace = 70 :=
by
  sorry

end NUMINAMATH_GPT_faye_earned_total_money_l1206_120626


namespace NUMINAMATH_GPT_time_to_pick_up_dog_l1206_120658

def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90
def dinner_time_in_minutes : ℕ := 180  -- 7:00 pm - 4:00 pm in minutes

def total_known_time : ℕ := commute_time + grocery_time + dry_cleaning_time + cooking_time

theorem time_to_pick_up_dog : (dinner_time_in_minutes - total_known_time) = 20 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_time_to_pick_up_dog_l1206_120658


namespace NUMINAMATH_GPT_sqrt_of_product_eq_540_l1206_120677

theorem sqrt_of_product_eq_540 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := 
by 
  sorry 

end NUMINAMATH_GPT_sqrt_of_product_eq_540_l1206_120677


namespace NUMINAMATH_GPT_average_weight_of_rock_l1206_120610

-- Define all the conditions
def price_per_pound : ℝ := 4
def total_amount : ℝ := 60
def number_of_rocks : ℕ := 10

-- The statement we need to prove
theorem average_weight_of_rock :
  (total_amount / price_per_pound) / number_of_rocks = 1.5 :=
sorry

end NUMINAMATH_GPT_average_weight_of_rock_l1206_120610


namespace NUMINAMATH_GPT_maximum_value_expr_l1206_120663

theorem maximum_value_expr :
  ∀ (a b c d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1) →
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_maximum_value_expr_l1206_120663


namespace NUMINAMATH_GPT_chord_length_circle_l1206_120614

theorem chord_length_circle {x y : ℝ} :
  (x - 1)^2 + (y - 1)^2 = 2 →
  (exists (p q : ℝ), (p-1)^2 = 1 ∧ (q-1)^2 = 1 ∧ p ≠ q ∧ abs (p - q) = 2) :=
by
  intro h
  use (2 : ℝ)
  use (0 : ℝ)
  -- Formal proof omitted
  sorry

end NUMINAMATH_GPT_chord_length_circle_l1206_120614


namespace NUMINAMATH_GPT_sqrt_six_greater_two_l1206_120632

theorem sqrt_six_greater_two : Real.sqrt 6 > 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_six_greater_two_l1206_120632


namespace NUMINAMATH_GPT_length_of_BD_l1206_120605

noncomputable def points_on_circle (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ) : Prop :=
  BC = 4 ∧ CD = 4 ∧ AE = 6 ∧ (0 < y) ∧ (0 < z) ∧ (AE * 2 = y * z) ∧ (8 > y + z)

theorem length_of_BD (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ)
  (h : points_on_circle A B C D E BD AE BC CD y z) : 
  BD = 7 :=
by
  sorry

end NUMINAMATH_GPT_length_of_BD_l1206_120605


namespace NUMINAMATH_GPT_problem1_problem2_l1206_120655

variables (a b c d e f : ℝ)

-- Define the probabilities and the sum condition
def total_probability (a b c d e f : ℝ) : Prop := a + b + c + d + e + f = 1

-- Define P and Q
def P (a b c d e f : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + e^2 + f^2
def Q (a b c d e f : ℝ) : ℝ := (a + c + e) * (b + d + f)

-- Problem 1
theorem problem1 (h : total_probability a b c d e f) : P a b c d e f ≥ 1/6 := sorry

-- Problem 2
theorem problem2 (h : total_probability a b c d e f) : 
  1/4 ≥ Q a b c d e f ∧ Q a b c d e f ≥ 1/2 - 3/2 * P a b c d e f := sorry

end NUMINAMATH_GPT_problem1_problem2_l1206_120655


namespace NUMINAMATH_GPT_probability_none_solve_l1206_120621

theorem probability_none_solve (a b c : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_prob : ((1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15)) : 
  (1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15 := 
by 
  sorry

end NUMINAMATH_GPT_probability_none_solve_l1206_120621


namespace NUMINAMATH_GPT_f_2a_eq_3_l1206_120669

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / 2^x

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2 * a) = 3 := by
  sorry

end NUMINAMATH_GPT_f_2a_eq_3_l1206_120669


namespace NUMINAMATH_GPT_find_x_range_l1206_120627

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem find_x_range (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end NUMINAMATH_GPT_find_x_range_l1206_120627


namespace NUMINAMATH_GPT_bird_population_in_1997_l1206_120694

theorem bird_population_in_1997 
  (k : ℝ)
  (pop_1995 pop_1996 pop_1998 : ℝ)
  (h1 : pop_1995 = 45)
  (h2 : pop_1996 = 70)
  (h3 : pop_1998 = 145)
  (h4 : pop_1997 - pop_1995 = k * pop_1996)
  (h5 : pop_1998 - pop_1996 = k * pop_1997) : 
  pop_1997 = 105 :=
by
  sorry

end NUMINAMATH_GPT_bird_population_in_1997_l1206_120694


namespace NUMINAMATH_GPT_ellipse_through_points_parabola_equation_l1206_120625

-- Ellipse Problem: Prove the standard equation
theorem ellipse_through_points (m n : ℝ) (m_pos : m > 0) (n_pos : n > 0) (m_ne_n : m ≠ n) :
  (m * 0^2 + n * (5/3)^2 = 1) ∧ (m * 1^2 + n * 1^2 = 1) →
  (m = 16 / 25 ∧ n = 9 / 25) → (m * x^2 + n * y^2 = 1) ↔ (16 * x^2 + 9 * y^2 = 225) :=
sorry

-- Parabola Problem: Prove the equation
theorem parabola_equation (p x y : ℝ) (p_pos : p > 0)
  (dist_focus : abs (x + p / 2) = 10) (dist_axis : y^2 = 36) :
  (p = 2 ∨ p = 18) →
  (y^2 = 2 * p * x) ↔ (y^2 = 4 * x ∨ y^2 = 36 * x) :=
sorry

end NUMINAMATH_GPT_ellipse_through_points_parabola_equation_l1206_120625


namespace NUMINAMATH_GPT_solution_l1206_120685

-- Define the conditions
variable (f : ℝ → ℝ)
variable (f_odd : ∀ x, f (-x) = -f x)
variable (f_periodic : ∀ x, f (x + 1) = f (1 - x))
variable (f_cubed : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x ^ 3)

-- Define the goal
theorem solution : f 2019 = -1 :=
by sorry

end NUMINAMATH_GPT_solution_l1206_120685


namespace NUMINAMATH_GPT_amanda_quizzes_l1206_120666

theorem amanda_quizzes (n : ℕ) (h1 : n > 0) (h2 : 92 * n + 97 = 93 * 5) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_amanda_quizzes_l1206_120666


namespace NUMINAMATH_GPT_louis_age_currently_31_l1206_120606

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end NUMINAMATH_GPT_louis_age_currently_31_l1206_120606


namespace NUMINAMATH_GPT_incorrect_statement_D_l1206_120698

theorem incorrect_statement_D
  (passes_through_center : ∀ (x_vals y_vals : List ℝ), ∃ (regression_line : ℝ → ℝ), 
    regression_line (x_vals.sum / x_vals.length) = (y_vals.sum / y_vals.length))
  (higher_r2_better_fit : ∀ (r2 : ℝ), r2 > 0 → ∃ (residual_sum_squares : ℝ), residual_sum_squares < (1 - r2))
  (slope_interpretation : ∀ (x : ℝ), (0.2 * x + 0.8) - (0.2 * (x - 1) + 0.8) = 0.2)
  (chi_squared_k2 : ∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), (k > 0) → 
    ∃ (confidence : ℝ), confidence > 0) :
  ¬(∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), k > 0 → 
    ∃ (confidence : ℝ), confidence < 0) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l1206_120698


namespace NUMINAMATH_GPT_root_in_interval_imp_range_m_l1206_120652

theorem root_in_interval_imp_range_m (m : ℝ) (f : ℝ → ℝ) (h : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0) : 2 < m ∧ m < 4 :=
by
  have exists_x : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 := h
  sorry

end NUMINAMATH_GPT_root_in_interval_imp_range_m_l1206_120652


namespace NUMINAMATH_GPT_bottle_caps_total_l1206_120683

def initial_bottle_caps := 51.0
def given_bottle_caps := 36.0

theorem bottle_caps_total : initial_bottle_caps + given_bottle_caps = 87.0 := by
  sorry

end NUMINAMATH_GPT_bottle_caps_total_l1206_120683


namespace NUMINAMATH_GPT_number_solution_l1206_120645

-- Statement based on identified conditions and answer
theorem number_solution (x : ℝ) (h : 0.10 * 0.30 * 0.50 * x = 90) : x = 6000 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_number_solution_l1206_120645


namespace NUMINAMATH_GPT_equation_of_midpoint_trajectory_l1206_120690

theorem equation_of_midpoint_trajectory
  (M : ℝ × ℝ)
  (hM : M.1 ^ 2 + M.2 ^ 2 = 1)
  (N : ℝ × ℝ := (2, 0))
  (P : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) :
  (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 / 4 := 
sorry

end NUMINAMATH_GPT_equation_of_midpoint_trajectory_l1206_120690


namespace NUMINAMATH_GPT_check_correct_options_l1206_120667

noncomputable def f (x a b: ℝ) := x^3 - a*x^2 + b*x + 1

theorem check_correct_options :
  (∀ (b: ℝ), b = 0 → ¬(∃ x: ℝ, 3 * x^2 - 2 * a * x = 0)) ∧
  (∀ (a: ℝ), a = 0 → (∀ x: ℝ, f x a b + f (-x) a b = 2)) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), b = a^2 / 4 ∧ a > -4 → ∃ x1 x2 x3: ℝ, f x1 a b = 0 ∧ f x2 a b = 0 ∧ f x3 a b = 0) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), (∀ x: ℝ, 3 * x^2 - 2 * a * x + b ≥ 0) → (a^2 ≤ 3*b)) := sorry

end NUMINAMATH_GPT_check_correct_options_l1206_120667


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1206_120688

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x + y = 1 → xy ≤ 1 / 4) ∧ (∃ x y : ℝ, xy ≤ 1 / 4 ∧ x + y ≠ 1) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1206_120688


namespace NUMINAMATH_GPT_oranges_weight_is_10_l1206_120679

def applesWeight (A : ℕ) : ℕ := A
def orangesWeight (A : ℕ) : ℕ := 5 * A
def totalWeight (A : ℕ) (O : ℕ) : ℕ := A + O
def totalCost (A : ℕ) (x : ℕ) (O : ℕ) (y : ℕ) : ℕ := A * x + O * y

theorem oranges_weight_is_10 (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := by
  sorry

end NUMINAMATH_GPT_oranges_weight_is_10_l1206_120679


namespace NUMINAMATH_GPT_speed_in_still_water_l1206_120601

def upstream_speed : ℝ := 35
def downstream_speed : ℝ := 45

theorem speed_in_still_water:
  (upstream_speed + downstream_speed) / 2 = 40 := 
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1206_120601


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1206_120696

theorem ball_hits_ground_time :
  ∀ t : ℝ, y = -20 * t^2 + 30 * t + 60 → y = 0 → t = (3 + Real.sqrt 57) / 4 := by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1206_120696


namespace NUMINAMATH_GPT_point_of_tangency_l1206_120637

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem point_of_tangency (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) 
  (h_slope : ∃ x : ℝ, Real.exp x - 1 / Real.exp x = 3 / 2) :
  ∃ x : ℝ, x = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_point_of_tangency_l1206_120637


namespace NUMINAMATH_GPT_professors_seat_choice_count_l1206_120612

theorem professors_seat_choice_count : 
    let chairs := 11 -- number of chairs
    let students := 7 -- number of students
    let professors := 4 -- number of professors
    ∀ (P: Fin professors -> Fin chairs), 
    (∀ (p : Fin professors), 1 ≤ P p ∧ P p ≤ 9) -- Each professor is between seats 2-10
    ∧ (P 0 < P 1) ∧ (P 1 < P 2) ∧ (P 2 < P 3) -- Professors must be placed with at least one seat gap
    ∧ (P 0 ≠ 1 ∧ P 3 ≠ 11) -- First and last seats are excluded
    → ∃ (ways : ℕ), ways = 840 := sorry

end NUMINAMATH_GPT_professors_seat_choice_count_l1206_120612


namespace NUMINAMATH_GPT_difference_between_balls_l1206_120619

theorem difference_between_balls (B R : ℕ) (h1 : R - 152 = B + 152 + 346) : R - B = 650 := 
sorry

end NUMINAMATH_GPT_difference_between_balls_l1206_120619


namespace NUMINAMATH_GPT_ratio_w_y_l1206_120699

open Real

theorem ratio_w_y (w x y z : ℝ) (h1 : w / x = 5 / 2) (h2 : y / z = 3 / 2) (h3 : z / x = 1 / 4) : w / y = 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_w_y_l1206_120699


namespace NUMINAMATH_GPT_find_a_l1206_120635

def setA (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def setB : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) : setA a ⊆ setB ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1206_120635


namespace NUMINAMATH_GPT_jordan_rect_width_is_10_l1206_120609

def carol_rect_length : ℕ := 5
def carol_rect_width : ℕ := 24
def jordan_rect_length : ℕ := 12

def carol_rect_area : ℕ := carol_rect_length * carol_rect_width
def jordan_rect_width := carol_rect_area / jordan_rect_length

theorem jordan_rect_width_is_10 : jordan_rect_width = 10 :=
by
  sorry

end NUMINAMATH_GPT_jordan_rect_width_is_10_l1206_120609


namespace NUMINAMATH_GPT_partnership_total_annual_gain_l1206_120646

theorem partnership_total_annual_gain 
  (x : ℝ) 
  (G : ℝ)
  (hA_investment : x * 12 = A_investment)
  (hB_investment : 2 * x * 6 = B_investment)
  (hC_investment : 3 * x * 4 = C_investment)
  (A_share : (A_investment / (A_investment + B_investment + C_investment)) * G = 6000) :
  G = 18000 := 
sorry

end NUMINAMATH_GPT_partnership_total_annual_gain_l1206_120646


namespace NUMINAMATH_GPT_length_of_LO_l1206_120684

theorem length_of_LO (MN LO : ℝ) (alt_O_MN alt_N_LO : ℝ) (h_MN : MN = 15) 
  (h_alt_O_MN : alt_O_MN = 9) (h_alt_N_LO : alt_N_LO = 7) : 
  LO = 19 + 2 / 7 :=
by
  -- Sorry means to skip the proof.
  sorry

end NUMINAMATH_GPT_length_of_LO_l1206_120684


namespace NUMINAMATH_GPT_value_of_coupon_l1206_120607

theorem value_of_coupon (price_per_bag : ℝ) (oz_per_bag : ℕ) (cost_per_serving_with_coupon : ℝ) (total_servings : ℕ) :
  price_per_bag = 25 → oz_per_bag = 40 → cost_per_serving_with_coupon = 0.50 → total_servings = 40 →
  (price_per_bag - (cost_per_serving_with_coupon * total_servings)) = 5 :=
by 
  intros hpb hob hcpwcs hts
  sorry

end NUMINAMATH_GPT_value_of_coupon_l1206_120607


namespace NUMINAMATH_GPT_employed_population_percentage_l1206_120686

noncomputable def percent_population_employed (total_population employed_males employed_females : ℝ) : ℝ :=
  employed_males + employed_females

theorem employed_population_percentage (population employed_males_percentage employed_females_percentage : ℝ) 
  (h1 : employed_males_percentage = 0.36 * population)
  (h2 : employed_females_percentage = 0.36 * population)
  (h3 : employed_females_percentage + employed_males_percentage = 0.50 * total_population)
  : total_population = 0.72 * population :=
by 
  sorry

end NUMINAMATH_GPT_employed_population_percentage_l1206_120686


namespace NUMINAMATH_GPT_fraction_equivalence_1_algebraic_identity_l1206_120638

/-- First Problem: Prove the equivalence of the fractions 171717/252525 and 17/25. -/
theorem fraction_equivalence_1 : 
  (171717 : ℚ) / 252525 = 17 / 25 := 
sorry

/-- Second Problem: Prove the equivalence of the algebraic expressions on both sides. -/
theorem algebraic_identity (a b : ℚ) : 
  2 * b^5 + (a^4 + a^3 * b + a^2 * b^2 + a * b^3 + b^4) * (a - b) = 
  (a^4 - a^3 * b + a^2 * b^2 - a * b^3 + b^4) * (a + b) := 
sorry

end NUMINAMATH_GPT_fraction_equivalence_1_algebraic_identity_l1206_120638


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1206_120600

variable {x y : ℝ} -- Declare x and y as real numbers

theorem simplify_expression1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
sorry

theorem simplify_expression2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - (3/2) * x^2 * y) + x^2 * y^2) = - x^2 * y^2 :=
sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1206_120600


namespace NUMINAMATH_GPT_same_terminal_side_l1206_120617

theorem same_terminal_side (k : ℤ) : 
  ∃ (α : ℤ), α = k * 360 + 330 ∧ (α = 510 ∨ α = 150 ∨ α = -150 ∨ α = -390) :=
by
  sorry

end NUMINAMATH_GPT_same_terminal_side_l1206_120617


namespace NUMINAMATH_GPT_minimum_value_expression_l1206_120693

theorem minimum_value_expression (a b : ℝ) (h : a * b > 0) : 
  ∃ m : ℝ, (∀ x y : ℝ, x * y > 0 → (4 * y / x + (x - 2 * y) / y) ≥ m) ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1206_120693


namespace NUMINAMATH_GPT_no_nat_triplet_exists_l1206_120695

theorem no_nat_triplet_exists (x y z : ℕ) : ¬ (x ^ 2 + y ^ 2 = 7 * z ^ 2) := 
sorry

end NUMINAMATH_GPT_no_nat_triplet_exists_l1206_120695


namespace NUMINAMATH_GPT_interest_rate_l1206_120665

theorem interest_rate (P1 P2 I T1 T2 total_amount : ℝ) (r : ℝ) :
  P1 = 10000 →
  P2 = 22000 →
  T1 = 2 →
  T2 = 3 →
  total_amount = 27160 →
  (I = P1 * r * T1 / 100 + P2 * r * T2 / 100) →
  P1 + P2 = 22000 →
  (P1 + I = total_amount) →
  r = 6 :=
by
  intros hP1 hP2 hT1 hT2 htotal_amount hI hP_total hP1_I_total
  -- Actual proof would go here
  sorry

end NUMINAMATH_GPT_interest_rate_l1206_120665


namespace NUMINAMATH_GPT_marion_score_correct_l1206_120608

-- Definitions based on conditions
def total_items : ℕ := 40
def ella_incorrect : ℕ := 4
def ella_correct : ℕ := total_items - ella_incorrect
def marion_score : ℕ := (ella_correct / 2) + 6

-- Statement of the theorem
theorem marion_score_correct : marion_score = 24 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_marion_score_correct_l1206_120608


namespace NUMINAMATH_GPT_complement_inter_proof_l1206_120636

open Set

variable (U : Set ℕ) (A B : Set ℕ)

def complement_inter (U A B : Set ℕ) : Set ℕ :=
  compl (A ∩ B)

theorem complement_inter_proof (hU : U = {1, 2, 3, 4, 5, 6, 7, 8} )
  (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) :
  complement_inter U A B = {1, 4, 5, 6, 7, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_inter_proof_l1206_120636


namespace NUMINAMATH_GPT_problem_k_value_l1206_120603

theorem problem_k_value (a b c : ℕ) (h1 : a + b / c = 101) (h2 : a / c + b = 68) :
  (a + b) / c = 13 :=
sorry

end NUMINAMATH_GPT_problem_k_value_l1206_120603


namespace NUMINAMATH_GPT_gumballs_initial_count_l1206_120689

theorem gumballs_initial_count (x : ℝ) (h : (0.75 ^ 3) * x = 27) : x = 64 :=
by
  sorry

end NUMINAMATH_GPT_gumballs_initial_count_l1206_120689
