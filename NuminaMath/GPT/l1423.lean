import Mathlib

namespace NUMINAMATH_GPT_next_equalities_from_conditions_l1423_142358

-- Definitions of the equality conditions
def eq1 : Prop := 3^2 + 4^2 = 5^2
def eq2 : Prop := 10^2 + 11^2 + 12^2 = 13^2 + 14^2
def eq3 : Prop := 21^2 + 22^2 + 23^2 + 24^2 = 25^2 + 26^2 + 27^2
def eq4 : Prop := 36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2

-- The next equalities we want to prove
def eq5 : Prop := 55^2 + 56^2 + 57^2 + 58^2 + 59^2 + 60^2 = 61^2 + 62^2 + 63^2 + 64^2 + 65^2
def eq6 : Prop := 78^2 + 79^2 + 80^2 + 81^2 + 82^2 + 83^2 + 84^2 = 85^2 + 86^2 + 87^2 + 88^2 + 89^2 + 90^2

theorem next_equalities_from_conditions : eq1 → eq2 → eq3 → eq4 → (eq5 ∧ eq6) :=
by
  sorry

end NUMINAMATH_GPT_next_equalities_from_conditions_l1423_142358


namespace NUMINAMATH_GPT_fifth_power_ends_with_same_digit_l1423_142350

theorem fifth_power_ends_with_same_digit (a : ℕ) : a^5 % 10 = a % 10 :=
by sorry

end NUMINAMATH_GPT_fifth_power_ends_with_same_digit_l1423_142350


namespace NUMINAMATH_GPT_jacob_ate_five_pies_l1423_142361

theorem jacob_ate_five_pies (weight_hot_dog weight_burger weight_pie noah_burgers mason_hotdogs_total_weight : ℕ)
    (H1 : weight_hot_dog = 2)
    (H2 : weight_burger = 5)
    (H3 : weight_pie = 10)
    (H4 : noah_burgers = 8)
    (H5 : mason_hotdogs_total_weight = 30)
    (H6 : ∀ x, 3 * x = (mason_hotdogs_total_weight / weight_hot_dog)) :
    (∃ y, y = (mason_hotdogs_total_weight / weight_hot_dog / 3) ∧ y = 5) :=
by
  sorry

end NUMINAMATH_GPT_jacob_ate_five_pies_l1423_142361


namespace NUMINAMATH_GPT_polygon_perimeter_greater_than_2_l1423_142376

-- Definition of the conditions
variable (polygon : Set (ℝ × ℝ))
variable (A B : ℝ × ℝ)
variable (P : ℝ)

axiom point_in_polygon (p : ℝ × ℝ) : p ∈ polygon
axiom A_in_polygon : A ∈ polygon
axiom B_in_polygon : B ∈ polygon
axiom path_length_condition (γ : ℝ → ℝ × ℝ) (γ_in_polygon : ∀ t, γ t ∈ polygon) (hA : γ 0 = A) (hB : γ 1 = B) : ∀ t₁ t₂, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ 1 → dist (γ t₁) (γ t₂) > 1

-- Statement to prove
theorem polygon_perimeter_greater_than_2 : P > 2 :=
sorry

end NUMINAMATH_GPT_polygon_perimeter_greater_than_2_l1423_142376


namespace NUMINAMATH_GPT_polygon_interior_exterior_relation_l1423_142305

theorem polygon_interior_exterior_relation (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n ≥ 3) :
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_exterior_relation_l1423_142305


namespace NUMINAMATH_GPT_range_of_a_l1423_142396

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1423_142396


namespace NUMINAMATH_GPT_simplification_of_fractional_equation_l1423_142365

theorem simplification_of_fractional_equation (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) -> (x - 4 * (3 - x) = -6) :=
by
  sorry

end NUMINAMATH_GPT_simplification_of_fractional_equation_l1423_142365


namespace NUMINAMATH_GPT_remaining_shape_perimeter_l1423_142319

def rectangle_perimeter (L W : ℕ) : ℕ := 2 * (L + W)

theorem remaining_shape_perimeter (L W S : ℕ) (hL : L = 12) (hW : W = 5) (hS : S = 2) :
  rectangle_perimeter L W = 34 :=
by
  rw [hL, hW]
  rfl

end NUMINAMATH_GPT_remaining_shape_perimeter_l1423_142319


namespace NUMINAMATH_GPT_cyclist_average_speed_l1423_142342

noncomputable def total_distance : ℝ := 10 + 5 + 15 + 20 + 30
noncomputable def time_first_segment : ℝ := 10 / 12
noncomputable def time_second_segment : ℝ := 5 / 6
noncomputable def time_third_segment : ℝ := 15 / 16
noncomputable def time_fourth_segment : ℝ := 20 / 14
noncomputable def time_fifth_segment : ℝ := 30 / 20

noncomputable def total_time : ℝ := time_first_segment + time_second_segment + time_third_segment + time_fourth_segment + time_fifth_segment

noncomputable def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed = 12.93 := by
  sorry

end NUMINAMATH_GPT_cyclist_average_speed_l1423_142342


namespace NUMINAMATH_GPT_find_a_value_l1423_142304

theorem find_a_value 
  (a : ℝ)
  (h : abs (1 - (-1 / (4 * a))) = 2) :
  a = 1 / 4 ∨ a = -1 / 12 :=
sorry

end NUMINAMATH_GPT_find_a_value_l1423_142304


namespace NUMINAMATH_GPT_additional_distance_sam_runs_more_than_sarah_l1423_142372

theorem additional_distance_sam_runs_more_than_sarah
  (street_width : ℝ) (block_side_length : ℝ)
  (h1 : street_width = 30) (h2 : block_side_length = 500) :
  let P_Sarah := 4 * block_side_length
  let P_Sam := 4 * (block_side_length + 2 * street_width)
  P_Sam - P_Sarah = 240 :=
by
  sorry

end NUMINAMATH_GPT_additional_distance_sam_runs_more_than_sarah_l1423_142372


namespace NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l1423_142389

-- Define the sequence and its cyclic property
def cyclicSequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 4) = 1000 * (seq n % 10) + 100 * (seq (n + 1) % 10) + 10 * (seq (n + 2) % 10) + (seq (n + 3) % 10)

-- Define the property of T being the sum of the sequence
def sumOfSequence (seq : ℕ → ℕ) (T : ℕ) : Prop :=
  T = seq 0 + seq 1 + seq 2 + seq 3

-- Define the statement that T is always divisible by 101
theorem largest_prime_divisor_of_sum_of_cyclic_sequence_is_101
  (seq : ℕ → ℕ) (T : ℕ)
  (h1 : cyclicSequence seq)
  (h2 : sumOfSequence seq T) :
  (101 ∣ T) := 
sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l1423_142389


namespace NUMINAMATH_GPT_min_calls_correct_l1423_142384

-- Define a function that calculates the minimum number of calls given n people
def min_calls (n : ℕ) : ℕ :=
  2 * n - 2

-- Theorem to prove that min_calls(n) given the conditions is equal to 2n - 2
theorem min_calls_correct (n : ℕ) (h : n ≥ 2) : min_calls n = 2 * n - 2 :=
by
  sorry

end NUMINAMATH_GPT_min_calls_correct_l1423_142384


namespace NUMINAMATH_GPT_tan_sum_example_l1423_142380

theorem tan_sum_example :
  let t1 := Real.tan (17 * Real.pi / 180)
  let t2 := Real.tan (43 * Real.pi / 180)
  t1 + t2 + Real.sqrt 3 * t1 * t2 = Real.sqrt 3 := sorry

end NUMINAMATH_GPT_tan_sum_example_l1423_142380


namespace NUMINAMATH_GPT_appropriate_mass_units_l1423_142351

def unit_of_mass_basket_of_eggs : String :=
  if 5 = 5 then "kilograms" else "unknown"

def unit_of_mass_honeybee : String :=
  if 5 = 5 then "grams" else "unknown"

def unit_of_mass_tank : String :=
  if 6 = 6 then "tons" else "unknown"

theorem appropriate_mass_units :
  unit_of_mass_basket_of_eggs = "kilograms" ∧
  unit_of_mass_honeybee = "grams" ∧
  unit_of_mass_tank = "tons" :=
by {
  -- skip the proof
  sorry
}

end NUMINAMATH_GPT_appropriate_mass_units_l1423_142351


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1423_142378

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (h1 : 0 < a * b ∧ a * b < 1) : ¬ (b < 1 / a) ∨ ¬ (1 / a < b) := by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1423_142378


namespace NUMINAMATH_GPT_units_digit_sum_of_factorials_l1423_142302

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  ones_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
              factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_sum_of_factorials_l1423_142302


namespace NUMINAMATH_GPT_price_of_72_cans_is_18_36_l1423_142388

def regular_price_per_can : ℝ := 0.30
def discount_percent : ℝ := 0.15
def number_of_cans : ℝ := 72

def discounted_price_per_can : ℝ := regular_price_per_can - (discount_percent * regular_price_per_can)
def total_price (num_cans : ℝ) : ℝ := num_cans * discounted_price_per_can

theorem price_of_72_cans_is_18_36 :
  total_price number_of_cans = 18.36 :=
by
  /- Proof details omitted -/
  sorry

end NUMINAMATH_GPT_price_of_72_cans_is_18_36_l1423_142388


namespace NUMINAMATH_GPT_total_cost_ice_cream_l1423_142349

noncomputable def price_Chocolate : ℝ := 2.50
noncomputable def price_Vanilla : ℝ := 2.00
noncomputable def price_Strawberry : ℝ := 2.25
noncomputable def price_Mint : ℝ := 2.20
noncomputable def price_WaffleCone : ℝ := 1.50
noncomputable def price_ChocolateChips : ℝ := 1.00
noncomputable def price_Fudge : ℝ := 1.25
noncomputable def price_WhippedCream : ℝ := 0.75

def scoops_Pierre : ℕ := 3  -- 2 scoops Chocolate + 1 scoop Mint
def scoops_Mother : ℕ := 4  -- 2 scoops Vanilla + 1 scoop Strawberry + 1 scoop Mint

noncomputable def price_Pierre_BeforeOffer : ℝ :=
  2 * price_Chocolate + price_Mint + price_WaffleCone + price_ChocolateChips

noncomputable def free_Pierre : ℝ := price_Mint -- Mint is the cheapest among Pierre's choices

noncomputable def price_Pierre_AfterOffer : ℝ := price_Pierre_BeforeOffer - free_Pierre

noncomputable def price_Mother_BeforeOffer : ℝ :=
  2 * price_Vanilla + price_Strawberry + price_Mint + price_WaffleCone + price_Fudge + price_WhippedCream

noncomputable def free_Mother : ℝ := price_Vanilla -- Vanilla is the cheapest among Mother's choices

noncomputable def price_Mother_AfterOffer : ℝ := price_Mother_BeforeOffer - free_Mother

noncomputable def total_BeforeDiscount : ℝ := price_Pierre_AfterOffer + price_Mother_AfterOffer

noncomputable def discount_Amount : ℝ := total_BeforeDiscount * 0.15

noncomputable def total_AfterDiscount : ℝ := total_BeforeDiscount - discount_Amount

theorem total_cost_ice_cream : total_AfterDiscount = 14.83 := by
  sorry


end NUMINAMATH_GPT_total_cost_ice_cream_l1423_142349


namespace NUMINAMATH_GPT_corrected_mean_l1423_142316

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_observation correct_observation : ℝ)
  (h_n : n = 50)
  (h_incorrect_mean : incorrect_mean = 30)
  (h_incorrect_observation : incorrect_observation = 23)
  (h_correct_observation : correct_observation = 48) :
  (incorrect_mean * n - incorrect_observation + correct_observation) / n = 30.5 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_l1423_142316


namespace NUMINAMATH_GPT_Brenda_bakes_cakes_l1423_142311

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_Brenda_bakes_cakes_l1423_142311


namespace NUMINAMATH_GPT_probability_not_below_x_axis_half_l1423_142398

-- Define the vertices of the parallelogram
def P : (ℝ × ℝ) := (4, 4)
def Q : (ℝ × ℝ) := (-2, -2)
def R : (ℝ × ℝ) := (-8, -2)
def S : (ℝ × ℝ) := (-2, 4)

-- Define a predicate for points within the parallelogram
def in_parallelogram (A B C D : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area_of_parallelogram (A B C D : ℝ × ℝ) : ℝ := sorry

noncomputable def probability_not_below_x_axis (A B C D : ℝ × ℝ) : ℝ :=
  let total_area := area_of_parallelogram A B C D
  let area_above_x_axis := area_of_parallelogram (0, 0) D A (0, 0) / 2
  area_above_x_axis / total_area

theorem probability_not_below_x_axis_half :
  probability_not_below_x_axis P Q R S = 1 / 2 :=
sorry

end NUMINAMATH_GPT_probability_not_below_x_axis_half_l1423_142398


namespace NUMINAMATH_GPT_smallest_h_l1423_142309

theorem smallest_h (h : ℕ) : 
  (∀ k, h = k → (k + 5) % 8 = 0 ∧ 
        (k + 5) % 11 = 0 ∧ 
        (k + 5) % 24 = 0) ↔ h = 259 :=
by
  sorry

end NUMINAMATH_GPT_smallest_h_l1423_142309


namespace NUMINAMATH_GPT_sqrt_expression_evaluation_l1423_142390

theorem sqrt_expression_evaluation (sqrt48 : Real) (sqrt1div3 : Real) 
  (h1 : sqrt48 = 4 * Real.sqrt 3) (h2 : sqrt1div3 = Real.sqrt (1 / 3)) :
  (-1 / 2) * sqrt48 * sqrt1div3 = -2 :=
by 
  rw [h1, h2]
  -- Continue with the simplification steps, however
  sorry

end NUMINAMATH_GPT_sqrt_expression_evaluation_l1423_142390


namespace NUMINAMATH_GPT_smallest_n_with_314_in_decimal_l1423_142354

theorem smallest_n_with_314_in_decimal {m n : ℕ} (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n) 
  (h_contains_314 : ∃ k : ℕ, (10^k * m) % n == 314) : n = 315 :=
sorry

end NUMINAMATH_GPT_smallest_n_with_314_in_decimal_l1423_142354


namespace NUMINAMATH_GPT_find_other_integer_l1423_142356

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 140) (h2 : x = 20 ∨ y = 20) : x = 20 ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_other_integer_l1423_142356


namespace NUMINAMATH_GPT_students_voted_for_meat_l1423_142397

theorem students_voted_for_meat (total_votes veggies_votes : ℕ) (h_total: total_votes = 672) (h_veggies: veggies_votes = 337) :
  total_votes - veggies_votes = 335 := 
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_students_voted_for_meat_l1423_142397


namespace NUMINAMATH_GPT_final_weight_is_sixteen_l1423_142395

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end NUMINAMATH_GPT_final_weight_is_sixteen_l1423_142395


namespace NUMINAMATH_GPT_smallest_x_for_square_l1423_142324

theorem smallest_x_for_square (N : ℕ) (h1 : ∃ x : ℕ, x > 0 ∧ 1260 * x = N^2) : ∃ x : ℕ, x = 35 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_square_l1423_142324


namespace NUMINAMATH_GPT_G_at_8_l1423_142345

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_8 :
  (G 4 = 8) →
  (∀ x : ℝ, (x^2 + 3 * x + 2 ≠ 0) →
    G (2 * x) / G (x + 2) = 4 - (16 * x + 8) / (x^2 + 3 * x + 2)) →
  G 8 = 112 / 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_G_at_8_l1423_142345


namespace NUMINAMATH_GPT_children_on_bus_l1423_142377

theorem children_on_bus (initial_children additional_children total_children : ℕ)
  (h1 : initial_children = 64)
  (h2 : additional_children = 14)
  (h3 : total_children = initial_children + additional_children) :
  total_children = 78 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_children_on_bus_l1423_142377


namespace NUMINAMATH_GPT_decode_CLUE_is_8671_l1423_142362

def BEST_OF_LUCK_code : List (Char × Nat) :=
  [('B', 0), ('E', 1), ('S', 2), ('T', 3), ('O', 4), ('F', 5),
   ('L', 6), ('U', 7), ('C', 8), ('K', 9)]

def decode (code : List (Char × Nat)) (word : String) : Option Nat :=
  word.toList.mapM (λ c => List.lookup c code) >>= (λ digits => 
  Option.some (Nat.ofDigits 10 digits))

theorem decode_CLUE_is_8671 :
  decode BEST_OF_LUCK_code "CLUE" = some 8671 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_decode_CLUE_is_8671_l1423_142362


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1423_142334

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1423_142334


namespace NUMINAMATH_GPT_range_of_a_l1423_142332

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) (h4 : ∀ x, g a x ≤ 0 → ¬(x < 0 ∧ g a x > 0)) :
  2 < a ∧ a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1423_142332


namespace NUMINAMATH_GPT_problem_solution_l1423_142328

variable (x y z : ℝ)

theorem problem_solution
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + x * y = 8)
  (h2 : y + z + y * z = 15)
  (h3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1423_142328


namespace NUMINAMATH_GPT_inverse_proportion_first_third_quadrant_l1423_142317

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_first_third_quadrant_l1423_142317


namespace NUMINAMATH_GPT_sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l1423_142392

-- Definitions based on conditions
def standard_flavor_to_water_ratio := 1 / 30
def standard_flavor_to_corn_syrup_ratio := 1 / 12
def sport_water_amount := 60
def sport_corn_syrup_amount := 4
def sport_flavor_to_water_ratio := 1 / 60
def sport_flavor_amount := 1 -- derived from sport_water_amount * sport_flavor_to_water_ratio

-- The main theorem to prove
theorem sport_flavoring_to_corn_syrup_ratio_is_three_times_standard :
  1 / 4 = 3 * (1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_sport_flavoring_to_corn_syrup_ratio_is_three_times_standard_l1423_142392


namespace NUMINAMATH_GPT_inequality_transfers_l1423_142368

variables (a b c d : ℝ)

theorem inequality_transfers (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_GPT_inequality_transfers_l1423_142368


namespace NUMINAMATH_GPT_num_math_not_science_l1423_142322

-- Definitions as conditions
def students_total : ℕ := 30
def both_clubs : ℕ := 2
def math_to_science_ratio : ℕ := 3

-- The proof we need to show
theorem num_math_not_science :
  ∃ x y : ℕ, (x + y + both_clubs = students_total) ∧ (y = math_to_science_ratio * (x + both_clubs) - 2 * (math_to_science_ratio - 1)) ∧ (y - both_clubs = 20) :=
by
  sorry

end NUMINAMATH_GPT_num_math_not_science_l1423_142322


namespace NUMINAMATH_GPT_bank_exceeds_1600cents_in_9_days_after_Sunday_l1423_142314

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end NUMINAMATH_GPT_bank_exceeds_1600cents_in_9_days_after_Sunday_l1423_142314


namespace NUMINAMATH_GPT_problem_statement_l1423_142363

theorem problem_statement (a : ℝ) (h : a^2 - 2 * a + 1 = 0) : 4 * a - 2 * a^2 + 2 = 4 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1423_142363


namespace NUMINAMATH_GPT_infinitely_many_gt_sqrt_l1423_142301

open Real

noncomputable def sequences := ℕ → ℕ × ℕ

def strictly_increasing_ratios (seq : sequences) : Prop :=
  ∀ n : ℕ, 0 < n → (seq (n + 1)).2 / (seq (n + 1)).1 > (seq n).2 / (seq n).1

theorem infinitely_many_gt_sqrt (seq : sequences) 
  (positive_integers : ∀ n : ℕ, (seq n).1 > 0 ∧ (seq n).2 > 0) 
  (inc_ratios : strictly_increasing_ratios seq) :
  ∃ᶠ n in at_top, (seq n).2 > sqrt n :=
sorry

end NUMINAMATH_GPT_infinitely_many_gt_sqrt_l1423_142301


namespace NUMINAMATH_GPT_ratio_spaghetti_to_manicotti_l1423_142343

-- Definitions of the given conditions
def total_students : ℕ := 800
def spaghetti_preferred : ℕ := 320
def manicotti_preferred : ℕ := 160

-- The theorem statement
theorem ratio_spaghetti_to_manicotti : spaghetti_preferred / manicotti_preferred = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_spaghetti_to_manicotti_l1423_142343


namespace NUMINAMATH_GPT_total_number_of_coins_is_15_l1423_142364

theorem total_number_of_coins_is_15 (x : ℕ) (h : 1*x + 5*x + 10*x + 25*x + 50*x = 273) : 5 * x = 15 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_total_number_of_coins_is_15_l1423_142364


namespace NUMINAMATH_GPT_quadratic_has_equal_roots_l1423_142336

theorem quadratic_has_equal_roots (b : ℝ) (h : ∃ x : ℝ, b*x^2 + 2*b*x + 4 = 0 ∧ b*x^2 + 2*b*x + 4 = 0) :
  b = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_has_equal_roots_l1423_142336


namespace NUMINAMATH_GPT_pages_to_read_tomorrow_l1423_142394

theorem pages_to_read_tomorrow (total_pages : ℕ) 
                              (days : ℕ)
                              (pages_read_yesterday : ℕ)
                              (pages_read_today : ℕ)
                              (yesterday_diff : pages_read_today = pages_read_yesterday - 5)
                              (total_pages_eq : total_pages = 100)
                              (days_eq : days = 3)
                              (yesterday_eq : pages_read_yesterday = 35) : 
                              ∃ pages_read_tomorrow,  pages_read_tomorrow = total_pages - (pages_read_yesterday + pages_read_today) := 
                              by
  use 35
  sorry

end NUMINAMATH_GPT_pages_to_read_tomorrow_l1423_142394


namespace NUMINAMATH_GPT_gross_profit_without_discount_l1423_142385

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_gross_profit_without_discount_l1423_142385


namespace NUMINAMATH_GPT_slope_of_perpendicular_line_l1423_142382

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_slope_of_perpendicular_line_l1423_142382


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1423_142308

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → (x ≤ a))
  → (∃ x : ℝ, (x ≤ a ∧ ¬((-2 ≤ x ∧ x ≤ 2))))
  → (a ≥ 2) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1423_142308


namespace NUMINAMATH_GPT_abs_inequality_l1423_142331

theorem abs_inequality (x y : ℝ) (h1 : |x| < 2) (h2 : |y| < 2) : |4 - x * y| > 2 * |x - y| :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_l1423_142331


namespace NUMINAMATH_GPT_num_choices_l1423_142300

theorem num_choices (classes scenic_spots : ℕ) (h_classes : classes = 4) (h_scenic_spots : scenic_spots = 3) :
  (scenic_spots ^ classes) = 81 :=
by
  -- The detailed proof goes here
  sorry

end NUMINAMATH_GPT_num_choices_l1423_142300


namespace NUMINAMATH_GPT_pies_from_apples_l1423_142341

theorem pies_from_apples (total_apples : ℕ) (percent_handout : ℝ) (apples_per_pie : ℕ) 
  (h_total : total_apples = 800) (h_percent : percent_handout = 0.65) (h_per_pie : apples_per_pie = 15) : 
  (total_apples * (1 - percent_handout)) / apples_per_pie = 18 := 
by 
  sorry

end NUMINAMATH_GPT_pies_from_apples_l1423_142341


namespace NUMINAMATH_GPT_no_2018_zero_on_curve_l1423_142370

theorem no_2018_zero_on_curve (a c d : ℝ) (hac : a * c > 0) : ¬∃(d : ℝ), (2018 : ℝ) ^ 2 * a + 2018 * c + d = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_no_2018_zero_on_curve_l1423_142370


namespace NUMINAMATH_GPT_sophie_one_dollar_bills_l1423_142321

theorem sophie_one_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 55) 
  (h2 : x + 2 * y + 5 * z = 126) 
  : x = 18 := by
  sorry

end NUMINAMATH_GPT_sophie_one_dollar_bills_l1423_142321


namespace NUMINAMATH_GPT_imaginary_part_of_z_l1423_142338

namespace ComplexNumberProof

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number
def z : ℂ := i^2 * (1 + i)

-- Prove the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
    -- Proof goes here
    sorry

end ComplexNumberProof

end NUMINAMATH_GPT_imaginary_part_of_z_l1423_142338


namespace NUMINAMATH_GPT_polynomial_roots_l1423_142387

noncomputable def f (x : ℝ) : ℝ := 8 * x^4 + 28 * x^3 - 74 * x^2 - 8 * x + 48

theorem polynomial_roots:
  ∃ (a b c d : ℝ), a = -3 ∧ b = -1 ∧ c = -1 ∧ d = 2 ∧ 
  (f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) :=
sorry

end NUMINAMATH_GPT_polynomial_roots_l1423_142387


namespace NUMINAMATH_GPT_height_radius_ratio_l1423_142353

variables (R H V : ℝ) (π : ℝ) (A : ℝ)

-- Given conditions
def volume_condition : Prop := π * R^2 * H = V / 2
def surface_area : ℝ := 2 * π * R^2 + 2 * π * R * H

-- Statement to prove
theorem height_radius_ratio (h_volume : volume_condition R H V π) :
  H / R = 2 := 
sorry

end NUMINAMATH_GPT_height_radius_ratio_l1423_142353


namespace NUMINAMATH_GPT_find_f_prime_zero_l1423_142320

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition given in the problem.
def f_def : ∀ x : ℝ, f x = x^2 + 2 * x * f' 1 := 
sorry

-- Statement we want to prove.
theorem find_f_prime_zero : f' 0 = -4 := 
sorry

end NUMINAMATH_GPT_find_f_prime_zero_l1423_142320


namespace NUMINAMATH_GPT_no_counterexample_exists_l1423_142329

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_exists : ∀ n : ℕ, sum_of_digits n % 9 = 0 → n % 9 = 0 :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_no_counterexample_exists_l1423_142329


namespace NUMINAMATH_GPT_highest_number_paper_l1423_142313

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end NUMINAMATH_GPT_highest_number_paper_l1423_142313


namespace NUMINAMATH_GPT_ratio_of_red_to_blue_marbles_l1423_142357

theorem ratio_of_red_to_blue_marbles:
  ∀ (R B : ℕ), 
    R + B = 30 →
    2 * (20 - B) = 10 →
    B = 15 → 
    R = 15 →
    R / B = 1 :=
by intros R B h₁ h₂ h₃ h₄
   sorry

end NUMINAMATH_GPT_ratio_of_red_to_blue_marbles_l1423_142357


namespace NUMINAMATH_GPT_reading_homework_is_4_l1423_142381

-- Defining the conditions.
variables (R : ℕ)  -- Number of pages of reading homework
variables (M : ℕ)  -- Number of pages of math homework

-- Rachel has 7 pages of math homework.
def math_homework_equals_7 : Prop := M = 7

-- Rachel has 3 more pages of math homework than reading homework.
def math_minus_reads_is_3 : Prop := M = R + 3

-- Prove the number of pages of reading homework is 4.
theorem reading_homework_is_4 (M R : ℕ) 
  (h1 : math_homework_equals_7 M) -- M = 7
  (h2 : math_minus_reads_is_3 M R) -- M = R + 3
  : R = 4 :=
sorry

end NUMINAMATH_GPT_reading_homework_is_4_l1423_142381


namespace NUMINAMATH_GPT_construct_all_naturals_starting_from_4_l1423_142391

-- Define the operations f, g, h
def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n  -- h is only meaningful if n is even

-- Main theorem: prove that starting from 4, every natural number can be constructed
theorem construct_all_naturals_starting_from_4 :
  ∀ (n : ℕ), ∃ (k : ℕ), (f^[k] 4 = n ∨ g^[k] 4 = n ∨ h^[k] 4 = n) :=
by sorry


end NUMINAMATH_GPT_construct_all_naturals_starting_from_4_l1423_142391


namespace NUMINAMATH_GPT_rectangle_area_l1423_142340

def length : ℝ := 15
def width : ℝ := 0.9 * length
def area : ℝ := length * width

theorem rectangle_area : area = 202.5 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1423_142340


namespace NUMINAMATH_GPT_find_first_day_speed_l1423_142330

theorem find_first_day_speed (t : ℝ) (d : ℝ) (v : ℝ) (h1 : d = 2.5) 
  (h2 : v * (t - 7/60) = d) (h3 : 10 * (t - 8/60) = d) : v = 9.375 :=
by {
  -- Proof omitted for brevity
  sorry
}

end NUMINAMATH_GPT_find_first_day_speed_l1423_142330


namespace NUMINAMATH_GPT_sum_of_numbers_l1423_142383

theorem sum_of_numbers : 3 + 33 + 333 + 33.3 = 402.3 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_numbers_l1423_142383


namespace NUMINAMATH_GPT_linear_function_quadrants_l1423_142393

theorem linear_function_quadrants (k b : ℝ) :
  (∀ x, (0 < x → 0 < k * x + b) ∧ (x < 0 → 0 < k * x + b) ∧ (x < 0 → k * x + b < 0)) →
  k > 0 ∧ b > 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_l1423_142393


namespace NUMINAMATH_GPT_cost_of_eight_books_l1423_142379

theorem cost_of_eight_books (x : ℝ) (h : 2 * x = 34) : 8 * x = 136 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_eight_books_l1423_142379


namespace NUMINAMATH_GPT_solution1_solution2_l1423_142371

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end NUMINAMATH_GPT_solution1_solution2_l1423_142371


namespace NUMINAMATH_GPT_find_value_l1423_142367

theorem find_value (x y : ℝ) (h : x - 2 * y = 1) : 3 - 4 * y + 2 * x = 5 := sorry

end NUMINAMATH_GPT_find_value_l1423_142367


namespace NUMINAMATH_GPT_simplify_fraction_l1423_142347

theorem simplify_fraction (x y z : ℕ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1423_142347


namespace NUMINAMATH_GPT_right_triangle_48_55_l1423_142315

def right_triangle_properties (a b : ℕ) (ha : a = 48) (hb : b = 55) : Prop :=
  let area := 1 / 2 * a * b
  let hypotenuse := Real.sqrt (a ^ 2 + b ^ 2)
  area = 1320 ∧ hypotenuse = 73

theorem right_triangle_48_55 : right_triangle_properties 48 55 (by rfl) (by rfl) :=
  sorry

end NUMINAMATH_GPT_right_triangle_48_55_l1423_142315


namespace NUMINAMATH_GPT_range_of_a_l1423_142366

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) : -4 ≤ a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1423_142366


namespace NUMINAMATH_GPT_sets_equal_l1423_142326

-- Defining the sets and proving their equality
theorem sets_equal : { x : ℝ | x^2 + 1 = 0 } = (∅ : Set ℝ) :=
  sorry

end NUMINAMATH_GPT_sets_equal_l1423_142326


namespace NUMINAMATH_GPT_probability_of_earning_exactly_2300_in_3_spins_l1423_142333

-- Definitions of the conditions
def spinner_sections : List ℕ := [0, 1000, 200, 7000, 300]
def equal_area_sections : Prop := true  -- Each section has the same area, simple condition

-- Proving the probability of earning exactly $2300 in three spins
theorem probability_of_earning_exactly_2300_in_3_spins :
  ∃ p : ℚ, p = 3 / 125 := sorry

end NUMINAMATH_GPT_probability_of_earning_exactly_2300_in_3_spins_l1423_142333


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1423_142374

open Nat

theorem part_a (y z : ℕ) (hy : 0 < y) (hz : 0 < z) : 
  (1 = 1 / y + 1 / z) ↔ (y = 2 ∧ z = 1) := 
by 
  sorry

theorem part_b (y z : ℕ) (hy : y ≥ 2) (hz : 0 < z) : 
  (1 / 2 + 1 / y = 1 / 2 + 1 / z) ↔ (y = z ∧ y ≥ 2) ∨ (y = 1 ∧ z = 1) := 
by 
  sorry 

theorem part_c (y z : ℕ) (hy : y ≥ 3) (hz : 0 < z) : 
  (1 / 3 + 1 / y = 1 / 2 + 1 / z) ↔ 
    (y = 3 ∧ z = 6) ∨ 
    (y = 4 ∧ z = 12) ∨ 
    (y = 5 ∧ z = 30) ∨ 
    (y = 2 ∧ z = 3) := 
by 
  sorry 

theorem part_d (x y : ℕ) (hx : x ≥ 4) (hy : y ≥ 4) : 
  ¬(1 / x + 1 / y = 1 / 2 + 1 / z) := 
by 
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l1423_142374


namespace NUMINAMATH_GPT_sin_cos_difference_theorem_tan_theorem_l1423_142303

open Real

noncomputable def sin_cos_difference (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5)

theorem sin_cos_difference_theorem (x : ℝ) (h : sin_cos_difference x) : 
  sin x - cos x = - 7 / 5 := by
  sorry

noncomputable def sin_cos_ratio (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5) ∧ (tan x = -3 / 4)

theorem tan_theorem (x : ℝ) (h : sin_cos_ratio x) :
  tan x = -3 / 4 := by
  sorry

end NUMINAMATH_GPT_sin_cos_difference_theorem_tan_theorem_l1423_142303


namespace NUMINAMATH_GPT_a_equals_2t_squared_l1423_142360

theorem a_equals_2t_squared {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4 * a = b^2) :
  ∃ t : ℕ, 0 < t ∧ a = 2 * t^2 :=
sorry

end NUMINAMATH_GPT_a_equals_2t_squared_l1423_142360


namespace NUMINAMATH_GPT_regular_pentagon_diagonal_square_l1423_142310

variable (a d : ℝ)
def is_regular_pentagon (a d : ℝ) : Prop :=
d ^ 2 = a ^ 2 + a * d

theorem regular_pentagon_diagonal_square :
  is_regular_pentagon a d :=
sorry

end NUMINAMATH_GPT_regular_pentagon_diagonal_square_l1423_142310


namespace NUMINAMATH_GPT_parabola_equation_l1423_142375

-- Define the given conditions
def vertex : ℝ × ℝ := (3, 5)
def point_on_parabola : ℝ × ℝ := (4, 2)

-- Prove that the equation is as specified
theorem parabola_equation :
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x y : ℝ, (y = a * x^2 + b * x + c) ↔
     (y = -3 * x^2 + 18 * x - 22) ∧ (vertex.snd = -3 * (vertex.fst - 3)^2 + 5) ∧
     (point_on_parabola.snd = a * point_on_parabola.fst^2 + b * point_on_parabola.fst + c)) := 
sorry

end NUMINAMATH_GPT_parabola_equation_l1423_142375


namespace NUMINAMATH_GPT_sum_first_five_terms_eq_ninety_three_l1423_142369

variable (a : ℕ → ℕ)

-- Definitions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

variables (a1 : ℕ) (a2 : ℕ) (a4 : ℕ)
variables (S : ℕ → ℕ)

-- Conditions
axiom a1_value : a1 = 3
axiom a2a4_value : a2 * a4 = 144

-- Question: Prove S_5 = 93
theorem sum_first_five_terms_eq_ninety_three
    (h1 : geometric_sequence a)
    (h2 : a 1 = a1)
    (h3 : a 2 = a2)
    (h4 : a 4 = a4)
    (Sn_def : S 5 = (a1 * (1 - (2:ℕ)^5)) / (1 - 2)) :
  S 5 = 93 :=
sorry

end NUMINAMATH_GPT_sum_first_five_terms_eq_ninety_three_l1423_142369


namespace NUMINAMATH_GPT_geometric_prog_105_l1423_142344

theorem geometric_prog_105 {a q : ℝ} 
  (h_sum : a + a * q + a * q^2 = 105) 
  (h_arith : a * q - a = (a * q^2 - 15) - a * q) :
  (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 0.5) :=
by
  sorry

end NUMINAMATH_GPT_geometric_prog_105_l1423_142344


namespace NUMINAMATH_GPT_not_prime_sum_l1423_142323

theorem not_prime_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_square : ∃ k : ℕ, a^2 - b * c = k^2) : ¬ Nat.Prime (2 * a + b + c) := 
sorry

end NUMINAMATH_GPT_not_prime_sum_l1423_142323


namespace NUMINAMATH_GPT_max_min_product_xy_theorem_l1423_142352

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end NUMINAMATH_GPT_max_min_product_xy_theorem_l1423_142352


namespace NUMINAMATH_GPT_f_at_2_l1423_142327

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- State the theorem that we need to prove
theorem f_at_2 : f 2 = 2 := by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_f_at_2_l1423_142327


namespace NUMINAMATH_GPT_min_distance_sum_well_l1423_142318

theorem min_distance_sum_well (A B C : ℝ) (h1 : B = A + 50) (h2 : C = B + 50) :
  ∃ X : ℝ, X = B ∧ (∀ Y : ℝ, (dist Y A + dist Y B + dist Y C) ≥ (dist B A + dist B B + dist B C)) :=
sorry

end NUMINAMATH_GPT_min_distance_sum_well_l1423_142318


namespace NUMINAMATH_GPT_value_of_a_sub_b_l1423_142307

theorem value_of_a_sub_b (a b : ℝ) (h1 : abs a = 8) (h2 : abs b = 5) (h3 : a > 0) (h4 : b < 0) : a - b = 13 := 
  sorry

end NUMINAMATH_GPT_value_of_a_sub_b_l1423_142307


namespace NUMINAMATH_GPT_monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l1423_142373

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Define the problem stating that when a <= 0, f(x) is increasing on (0, +∞)
theorem monotonicity_increasing_when_a_nonpos (a : ℝ) (h : a ≤ 0) :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Define the problem stating that when a > 0, f(x) is increasing on (0, 1/a) and decreasing on (1/a, +∞)
theorem monotonicity_increasing_decreasing_when_a_pos (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x < (1 / a) → y < (1 / a) → f a x < f a y) ∧
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (1 / a) < x → (1 / a) < y → f a y < f a x) :=
sorry

-- Define the problem for the range of a such that f(x) < 0 for all x in (0, +∞)
theorem range_of_a_for_f_less_than_zero (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x < 0) ↔ a ∈ Set.Ioi (1 / Real.exp 1) :=
sorry

end NUMINAMATH_GPT_monotonicity_increasing_when_a_nonpos_monotonicity_increasing_decreasing_when_a_pos_range_of_a_for_f_less_than_zero_l1423_142373


namespace NUMINAMATH_GPT_find_a_range_find_value_x1_x2_l1423_142312

noncomputable def quadratic_equation_roots_and_discriminant (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
      (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
      (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧
      (x1 ≠ x2) ∧ 
      (∀ Δ > 0, Δ = 9 - 8 * a - 4)

theorem find_a_range (a : ℝ) : 
  (quadratic_equation_roots_and_discriminant a) → a < 5 / 8 :=
sorry

theorem find_value_x1_x2 (a : ℤ) (h : a = 0) (x1 x2 : ℝ) :
  (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
  (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧ 
  (x1 + x2 = 3) ∧ 
  (x1 * x2 = 1) → 
  (x1^2 * x2 + x1 * x2^2 = 3) :=
sorry

end NUMINAMATH_GPT_find_a_range_find_value_x1_x2_l1423_142312


namespace NUMINAMATH_GPT_katie_earnings_l1423_142325

def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

theorem katie_earnings : bead_necklaces + gemstone_necklaces * cost_per_necklace = 21 := 
by
  sorry

end NUMINAMATH_GPT_katie_earnings_l1423_142325


namespace NUMINAMATH_GPT_algebra_expression_never_zero_l1423_142346

theorem algebra_expression_never_zero (x : ℝ) : (1 : ℝ) / (x - 1) ≠ 0 :=
sorry

end NUMINAMATH_GPT_algebra_expression_never_zero_l1423_142346


namespace NUMINAMATH_GPT_maximize_area_of_sector_l1423_142359

noncomputable def area_of_sector (x y : ℝ) : ℝ := (1 / 2) * x * y

theorem maximize_area_of_sector : 
  ∃ x y : ℝ, 2 * x + y = 20 ∧ (∀ (x : ℝ), x > 0 → 
  (∀ (y : ℝ), y > 0 → 2 * x + y = 20 → area_of_sector x y ≤ area_of_sector 5 (20 - 2 * 5))) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_of_sector_l1423_142359


namespace NUMINAMATH_GPT_total_highlighters_is_49_l1423_142348

-- Define the number of highlighters of each color
def pink_highlighters : Nat := 15
def yellow_highlighters : Nat := 12
def blue_highlighters : Nat := 9
def green_highlighters : Nat := 7
def purple_highlighters : Nat := 6

-- Define the total number of highlighters
def total_highlighters : Nat := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters

-- Statement that the total number of highlighters should be 49
theorem total_highlighters_is_49 : total_highlighters = 49 := by
  sorry

end NUMINAMATH_GPT_total_highlighters_is_49_l1423_142348


namespace NUMINAMATH_GPT_cubic_polynomial_range_l1423_142339

-- Define the conditions and the goal in Lean
theorem cubic_polynomial_range :
  ∀ x : ℝ, (x^2 - 5 * x + 6 < 0) → (41 < x^3 + 5 * x^2 + 6 * x + 1) ∧ (x^3 + 5 * x^2 + 6 * x + 1 < 91) :=
by
  intros x hx
  have h1 : 2 < x := sorry
  have h2 : x < 3 := sorry
  have h3 : (x^3 + 5 * x^2 + 6 * x + 1) > 41 := sorry
  have h4 : (x^3 + 5 * x^2 + 6 * x + 1) < 91 := sorry
  exact ⟨h3, h4⟩ 

end NUMINAMATH_GPT_cubic_polynomial_range_l1423_142339


namespace NUMINAMATH_GPT_digit_is_4_l1423_142399

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem digit_is_4 (d : ℕ) (hd0 : is_even d) (hd1 : is_divisible_by_3 (14 + d)) : d = 4 :=
  sorry

end NUMINAMATH_GPT_digit_is_4_l1423_142399


namespace NUMINAMATH_GPT_reduced_price_per_kg_l1423_142337

theorem reduced_price_per_kg {P R : ℝ} (H1 : R = 0.75 * P) (H2 : 1100 = 1100 / P * P) (H3 : 1100 = (1100 / P + 5) * R) : R = 55 :=
by sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l1423_142337


namespace NUMINAMATH_GPT_stratified_sampling_third_grade_l1423_142306

theorem stratified_sampling_third_grade (total_students : ℕ) (first_grade_students : ℕ)
  (second_grade_students : ℕ) (third_grade_students : ℕ) (sample_size : ℕ)
  (h_total : total_students = 270000) (h_first : first_grade_students = 99000)
  (h_second : second_grade_students = 90000) (h_third : third_grade_students = 81000)
  (h_sample : sample_size = 3000) :
  third_grade_students * (sample_size / total_students) = 900 := 
by {
  sorry
}

end NUMINAMATH_GPT_stratified_sampling_third_grade_l1423_142306


namespace NUMINAMATH_GPT_verify_extrema_l1423_142386

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end NUMINAMATH_GPT_verify_extrema_l1423_142386


namespace NUMINAMATH_GPT_coat_price_reduction_l1423_142355

theorem coat_price_reduction :
  let orig_price := 500
  let first_discount := 0.15 * orig_price
  let price_after_first := orig_price - first_discount
  let second_discount := 0.10 * price_after_first
  let price_after_second := price_after_first - second_discount
  let tax := 0.07 * price_after_second
  let price_with_tax := price_after_second + tax
  let final_price := price_with_tax - 200
  let reduction_amount := orig_price - final_price
  let percent_reduction := (reduction_amount / orig_price) * 100
  percent_reduction = 58.145 :=
by
  sorry

end NUMINAMATH_GPT_coat_price_reduction_l1423_142355


namespace NUMINAMATH_GPT_isabel_earnings_l1423_142335

theorem isabel_earnings :
  ∀ (bead_necklaces gem_necklaces cost_per_necklace : ℕ),
    bead_necklaces = 3 →
    gem_necklaces = 3 →
    cost_per_necklace = 6 →
    (bead_necklaces + gem_necklaces) * cost_per_necklace = 36 := by
sorry

end NUMINAMATH_GPT_isabel_earnings_l1423_142335
