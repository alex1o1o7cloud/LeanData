import Mathlib

namespace cos_alpha_plus_5pi_over_4_eq_16_over_65_l93_9391

theorem cos_alpha_plus_5pi_over_4_eq_16_over_65
  (α β : ℝ)
  (hα : -π / 4 < α ∧ α < 0)
  (hβ : π / 2 < β ∧ β < π)
  (hcos_sum : Real.cos (α + β) = -4/5)
  (hcos_diff : Real.cos (β - π / 4) = 5/13) :
  Real.cos (α + 5 * π / 4) = 16/65 :=
by
  sorry

end cos_alpha_plus_5pi_over_4_eq_16_over_65_l93_9391


namespace motorcycles_meet_after_54_minutes_l93_9372

noncomputable def motorcycles_meet_time : ℕ := sorry

theorem motorcycles_meet_after_54_minutes :
  motorcycles_meet_time = 54 := sorry

end motorcycles_meet_after_54_minutes_l93_9372


namespace true_propositions_l93_9338

theorem true_propositions : 
  (∀ x : ℝ, x^3 < 1 → x^2 + 1 > 0) ∧ (∀ x : ℚ, x^2 = 2 → false) ∧ 
  (∀ x : ℕ, x^3 > x^2 → false) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by 
  -- proof goes here
  sorry

end true_propositions_l93_9338


namespace average_minutes_per_day_l93_9375

theorem average_minutes_per_day (e : ℕ) (h_e_pos : 0 < e) : 
  let sixth_grade_minutes := 20
  let seventh_grade_minutes := 18
  let eighth_grade_minutes := 12
  
  let sixth_graders := 3 * e
  let seventh_graders := 4 * e
  let eighth_graders := e
  
  let total_minutes := sixth_grade_minutes * sixth_graders + seventh_grade_minutes * seventh_graders + eighth_grade_minutes * eighth_graders
  let total_students := sixth_graders + seventh_graders + eighth_graders
  
  (total_minutes / total_students) = 18 := by
sorry

end average_minutes_per_day_l93_9375


namespace angle_A_in_triangle_l93_9369

theorem angle_A_in_triangle (a b c : ℝ) (h : a^2 = b^2 + b * c + c^2) : A = 120 :=
sorry

end angle_A_in_triangle_l93_9369


namespace map_distance_l93_9394

variable (map_distance_km : ℚ) (map_distance_inches : ℚ) (actual_distance_km: ℚ)

theorem map_distance (h1 : actual_distance_km = 136)
                     (h2 : map_distance_inches = 42)
                     (h3 : map_distance_km = 18.307692307692307) :
  (actual_distance_km * map_distance_inches / map_distance_km = 312) :=
by sorry

end map_distance_l93_9394


namespace total_students_in_classes_l93_9396

theorem total_students_in_classes (t1 t2 x y: ℕ) (h1 : t1 = 273) (h2 : t2 = 273) (h3 : (x - 1) * 7 = t1) (h4 : (y - 1) * 13 = t2) : x + y = 62 :=
by
  sorry

end total_students_in_classes_l93_9396


namespace product_of_two_numbers_l93_9303

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end product_of_two_numbers_l93_9303


namespace thabo_books_l93_9306

theorem thabo_books :
  ∃ (H P F : ℕ), 
    P = H + 20 ∧ 
    F = 2 * P ∧ 
    H + P + F = 200 ∧ 
    H = 35 :=
by
  sorry

end thabo_books_l93_9306


namespace Shawn_scored_6_points_l93_9300

theorem Shawn_scored_6_points
  (points_per_basket : ℤ)
  (matthew_points : ℤ)
  (total_baskets : ℤ)
  (h1 : points_per_basket = 3)
  (h2 : matthew_points = 9)
  (h3 : total_baskets = 5)
  : (∃ shawn_points : ℤ, shawn_points = 6) :=
by
  sorry

end Shawn_scored_6_points_l93_9300


namespace train_speed_approx_l93_9361

noncomputable def man_speed_kmh : ℝ := 3
noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600
noncomputable def train_length : ℝ := 900
noncomputable def time_to_cross : ℝ := 53.99568034557235
noncomputable def train_speed_ms := (train_length / time_to_cross) + man_speed_ms
noncomputable def train_speed_kmh := (train_speed_ms * 3600) / 1000

theorem train_speed_approx :
  abs (train_speed_kmh - 63.009972) < 1e-5 := sorry

end train_speed_approx_l93_9361


namespace triangle_construction_possible_l93_9376

theorem triangle_construction_possible (r l_alpha k_alpha : ℝ) (h1 : r > 0) (h2 : l_alpha > 0) (h3 : k_alpha > 0) :
  l_alpha^2 < (4 * k_alpha^2 * r^2) / (k_alpha^2 + r^2) :=
sorry

end triangle_construction_possible_l93_9376


namespace compute_expression_l93_9392

theorem compute_expression :
  120 * 2400 - 20 * 2400 - 100 * 2400 = 0 :=
sorry

end compute_expression_l93_9392


namespace event_A_probability_l93_9359

theorem event_A_probability (n : ℕ) (m₀ : ℕ) (H_n : n = 120) (H_m₀ : m₀ = 32) (p : ℝ) :
  (n * p - (1 - p) ≤ m₀) ∧ (n * p + p ≥ m₀) → 
  (32 / 121 : ℝ) ≤ p ∧ p ≤ (33 / 121 : ℝ) :=
sorry

end event_A_probability_l93_9359


namespace face_value_of_stock_l93_9327

-- Define variables and constants
def quoted_price : ℝ := 200
def yield_quoted : ℝ := 0.10
def percentage_yield : ℝ := 0.20

-- Define the annual income from the quoted price and percentage yield
def annual_income_from_quoted_price : ℝ := yield_quoted * quoted_price
def annual_income_from_face_value (FV : ℝ) : ℝ := percentage_yield * FV

-- Problem statement to prove
theorem face_value_of_stock (FV : ℝ) :
  annual_income_from_face_value FV = annual_income_from_quoted_price →
  FV = 100 := 
by
  sorry

end face_value_of_stock_l93_9327


namespace variance_of_heights_l93_9360
-- Importing all necessary libraries

-- Define a list of heights
def heights : List ℕ := [160, 162, 159, 160, 159]

-- Define the function to calculate the mean of a list of natural numbers
def mean (list : List ℕ) : ℚ :=
  list.sum / list.length

-- Define the function to calculate the variance of a list of natural numbers
def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (λ x => (x - μ) ^ 2)).sum / list.length

-- The theorem statement that proves the variance is 6/5
theorem variance_of_heights : variance heights = 6 / 5 :=
  sorry

end variance_of_heights_l93_9360


namespace spending_on_gifts_l93_9339

-- Defining the conditions as Lean statements
def num_sons_teachers : ℕ := 3
def num_daughters_teachers : ℕ := 4
def cost_per_gift : ℕ := 10

-- The total number of teachers
def total_teachers : ℕ := num_sons_teachers + num_daughters_teachers

-- Proving that the total spending on gifts is $70
theorem spending_on_gifts : total_teachers * cost_per_gift = 70 :=
by
  -- proof goes here
  sorry

end spending_on_gifts_l93_9339


namespace greatest_third_side_l93_9368

theorem greatest_third_side (a b : ℕ) (h1 : a = 5) (h2 : b = 10) : 
  ∃ c : ℕ, c < a + b ∧ c > (b - a) ∧ c = 14 := 
by
  sorry

end greatest_third_side_l93_9368


namespace floor_equation_solution_l93_9326

theorem floor_equation_solution (x : ℝ) :
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := 
sorry

end floor_equation_solution_l93_9326


namespace game_is_unfair_l93_9385

def pencil_game_unfair : Prop :=
∀ (take1 take2 : ℕ → ℕ),
  take1 1 = 1 ∨ take1 1 = 2 →
  take2 2 = 1 ∨ take2 2 = 2 →
  ∀ n : ℕ,
    n = 5 → (∃ first_move : ℕ, (take1 first_move = 2) ∧ (take2 (take1 first_move) = 1 ∨ take2 (take1 first_move) = 2) ∧ (take1 (take2 (n - take1 first_move)) = 1 ∨ take1 (take2 (n - take1 first_move)) = 2) ∧
    ∀ second_move : ℕ, (second_move = n - first_move - take2 (n - take1 first_move)) → 
    n - first_move - take2 (n - take1 first_move) = 1 ∨ n - first_move - take2 (n - take1 first_move) = 2)

theorem game_is_unfair : pencil_game_unfair := 
sorry

end game_is_unfair_l93_9385


namespace range_of_a_l93_9311

open Real

theorem range_of_a (a : ℝ) :
  ((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∨ (a^2 - 2 * a - 3 < 0)) ∧
  ¬((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∧ (a^2 - 2 * a - 3 < 0)) ↔
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l93_9311


namespace compute_65_sq_minus_55_sq_l93_9341

theorem compute_65_sq_minus_55_sq : 65^2 - 55^2 = 1200 :=
by
  -- We'll skip the proof here for simplicity
  sorry

end compute_65_sq_minus_55_sq_l93_9341


namespace largest_divisor_n4_n2_l93_9380

theorem largest_divisor_n4_n2 (n : ℤ) : (6 : ℤ) ∣ (n^4 - n^2) :=
sorry

end largest_divisor_n4_n2_l93_9380


namespace minimum_value_l93_9318

theorem minimum_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x - 2 * y + 3 = 0) : 
  ∃ z : ℝ, z = 3 ∧ (∀ z' : ℝ, (z' = y^2 / x) → z ≤ z') :=
sorry

end minimum_value_l93_9318


namespace totalWatermelons_l93_9365

def initialWatermelons : ℕ := 4
def additionalWatermelons : ℕ := 3

theorem totalWatermelons : initialWatermelons + additionalWatermelons = 7 := by
  sorry

end totalWatermelons_l93_9365


namespace valid_three_digit_numbers_count_l93_9331

noncomputable def count_valid_three_digit_numbers : ℕ :=
  let total_three_digit_numbers := 900
  let excluded_numbers := 81 + 72
  total_three_digit_numbers - excluded_numbers

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 747 :=
by
  sorry

end valid_three_digit_numbers_count_l93_9331


namespace polynomial_value_l93_9328

theorem polynomial_value 
  (x : ℝ) 
  (h1 : x = (1 + (1994 : ℝ).sqrt) / 2) : 
  (4 * x ^ 3 - 1997 * x - 1994) ^ 20001 = -1 := 
  sorry

end polynomial_value_l93_9328


namespace combined_age_of_sam_and_drew_l93_9315

theorem combined_age_of_sam_and_drew
  (sam_age : ℕ)
  (drew_age : ℕ)
  (h1 : sam_age = 18)
  (h2 : sam_age = drew_age / 2):
  sam_age + drew_age = 54 := sorry

end combined_age_of_sam_and_drew_l93_9315


namespace find_r_l93_9352

noncomputable def g (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

theorem find_r 
  (p q r : ℝ) 
  (h1 : ∀ x : ℝ, g x p q r = (x + 100) * (x + 0) * (x + 0))
  (h2 : p + q + r = 100) : 
  r = 0 := 
by
  sorry

end find_r_l93_9352


namespace train_crossing_tree_time_l93_9378

noncomputable def time_to_cross_platform (train_length : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) : ℕ :=
  (train_length + platform_length) / time_to_cross_platform

noncomputable def time_to_cross_tree (train_length : ℕ) (speed : ℕ) : ℕ :=
  train_length / speed

theorem train_crossing_tree_time :
  ∀ (train_length platform_length time platform_time speed : ℕ),
  train_length = 1200 →
  platform_length = 900 →
  platform_time = 210 →
  speed = (train_length + platform_length) / platform_time →
  time = train_length / speed →
  time = 120 :=
by
  intros train_length platform_length time platform_time speed h_train_length h_platform_length h_platform_time h_speed h_time
  sorry

end train_crossing_tree_time_l93_9378


namespace linda_total_miles_l93_9377

def calculate_total_miles (x : ℕ) : ℕ :=
  (60 / x) + (60 / (x + 4)) + (60 / (x + 8)) + (60 / (x + 12)) + (60 / (x + 16))

theorem linda_total_miles (x : ℕ) (hx1 : x > 0)
(hdx2 : 60 % x = 0)
(hdx3 : 60 % (x + 4) = 0) 
(hdx4 : 60 % (x + 8) = 0) 
(hdx5 : 60 % (x + 12) = 0) 
(hdx6 : 60 % (x + 16) = 0) :
  calculate_total_miles x = 33 := by
  sorry

end linda_total_miles_l93_9377


namespace tv_cost_l93_9362

theorem tv_cost (savings : ℕ) (fraction_spent_on_furniture : ℚ) (amount_spent_on_furniture : ℚ) (remaining_savings : ℚ) :
  savings = 1000 →
  fraction_spent_on_furniture = 3/5 →
  amount_spent_on_furniture = fraction_spent_on_furniture * savings →
  remaining_savings = savings - amount_spent_on_furniture →
  remaining_savings = 400 :=
by
  sorry

end tv_cost_l93_9362


namespace Cade_remaining_marbles_l93_9313

def initial_marbles := 87
def given_marbles := 8
def remaining_marbles := initial_marbles - given_marbles

theorem Cade_remaining_marbles : remaining_marbles = 79 := by
  sorry

end Cade_remaining_marbles_l93_9313


namespace sum_of_integers_is_18_l93_9343

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18_l93_9343


namespace sin_150_eq_half_l93_9358

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l93_9358


namespace binomial_expansion_coefficient_l93_9321

theorem binomial_expansion_coefficient (a : ℝ)
  (h : ∃ r, 9 - 3 * r = 6 ∧ (-a)^r * (Nat.choose 9 r) = 36) :
  a = -4 :=
  sorry

end binomial_expansion_coefficient_l93_9321


namespace weight_of_NH4I_H2O_l93_9366

noncomputable def total_weight (moles_NH4I : ℕ) (molar_mass_NH4I : ℝ) 
                             (moles_H2O : ℕ) (molar_mass_H2O : ℝ) : ℝ :=
  (moles_NH4I * molar_mass_NH4I) + (moles_H2O * molar_mass_H2O)

theorem weight_of_NH4I_H2O :
  total_weight 15 144.95 7 18.02 = 2300.39 :=
by
  sorry

end weight_of_NH4I_H2O_l93_9366


namespace bc_guilty_l93_9370

-- Definition of guilty status of defendants
variables (A B C : Prop)

-- Conditions
axiom condition1 : A ∨ B ∨ C
axiom condition2 : A → ¬B → ¬C

-- Theorem stating that one of B or C is guilty
theorem bc_guilty : B ∨ C :=
by {
  -- Proof goes here
  sorry
}

end bc_guilty_l93_9370


namespace quadrilateral_angle_l93_9363

theorem quadrilateral_angle (x y : ℝ) (h1 : 3 * x ^ 2 - x + 4 = 5) (h2 : x ^ 2 + y ^ 2 = 9) :
  x = (1 + Real.sqrt 13) / 6 :=
by
  sorry

end quadrilateral_angle_l93_9363


namespace sam_possible_lunches_without_violation_l93_9342

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end sam_possible_lunches_without_violation_l93_9342


namespace sum_of_digits_0_to_999_l93_9350

-- Sum of digits from 0 to 9
def sum_of_digits : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Sum of digits from 1 to 9
def sum_of_digits_without_zero : ℕ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Units place sum
def units_sum : ℕ := sum_of_digits * 100

-- Tens place sum
def tens_sum : ℕ := sum_of_digits * 100

-- Hundreds place sum
def hundreds_sum : ℕ := sum_of_digits_without_zero * 100

-- Total sum
def total_sum : ℕ := units_sum + tens_sum + hundreds_sum

theorem sum_of_digits_0_to_999 : total_sum = 13500 := by
  sorry

end sum_of_digits_0_to_999_l93_9350


namespace smallest_nat_number_l93_9393

theorem smallest_nat_number : ∃ a : ℕ, (a % 3 = 2) ∧ (a % 5 = 4) ∧ (a % 7 = 4) ∧ (∀ b : ℕ, (b % 3 = 2) ∧ (b % 5 = 4) ∧ (b % 7 = 4) → a ≤ b) ∧ a = 74 := 
sorry

end smallest_nat_number_l93_9393


namespace solve_for_x_l93_9304

theorem solve_for_x (x : ℝ) (h : (1/3) + (1/x) = 2/3) : x = 3 :=
by
  sorry

end solve_for_x_l93_9304


namespace hyperbola_focal_length_l93_9317

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 4 = 1)) ∧ (∀ f : ℝ, f = 6) → m = 5 := 
  by 
    -- Using the condition that the focal length is 6
    sorry

end hyperbola_focal_length_l93_9317


namespace one_third_sugar_l93_9348

theorem one_third_sugar (sugar : ℚ) (h : sugar = 3 + 3 / 4) : sugar / 3 = 1 + 1 / 4 :=
by sorry

end one_third_sugar_l93_9348


namespace penguins_seals_ratio_l93_9346

theorem penguins_seals_ratio (t_total t_seals t_elephants t_penguins : ℕ) 
    (h1 : t_total = 130) 
    (h2 : t_seals = 13) 
    (h3 : t_elephants = 13) 
    (h4 : t_penguins = t_total - t_seals - t_elephants) : 
    (t_penguins / t_seals = 8) := by
  sorry

end penguins_seals_ratio_l93_9346


namespace packages_eq_nine_l93_9381

-- Definitions of the given conditions
def x : ℕ := 50
def y : ℕ := 5
def z : ℕ := 5

-- Statement: Prove that the number of packages Amy could make equals 9
theorem packages_eq_nine : (x - y) / z = 9 :=
by
  sorry

end packages_eq_nine_l93_9381


namespace prob_A_exactly_once_l93_9354

theorem prob_A_exactly_once (P : ℚ) (h : 1 - (1 - P)^3 = 63 / 64) : 
  (3 * P * (1 - P)^2 = 9 / 64) :=
by
  sorry

end prob_A_exactly_once_l93_9354


namespace number_of_cars_l93_9312

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end number_of_cars_l93_9312


namespace loss_per_meter_is_5_l93_9337

-- Define the conditions
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 50
def quantity : ℕ := 400

-- Define the statement to prove (question == answer given conditions)
theorem loss_per_meter_is_5 : 
  ((cost_price_per_meter * quantity - selling_price) / quantity) = 5 := 
by
  sorry

end loss_per_meter_is_5_l93_9337


namespace triangle_problem_l93_9390

noncomputable def length_of_side_c (a : ℝ) (cosB : ℝ) (C : ℝ) : ℝ :=
  a * (Real.sqrt 2 / 2) / (Real.sqrt (1 - cosB^2))

noncomputable def cos_A_minus_pi_over_6 (cosB : ℝ) (cosA : ℝ) (sinA : ℝ) : ℝ :=
  cosA * (Real.sqrt 3 / 2) + sinA * (1 / 2)

theorem triangle_problem (a : ℝ) (cosB : ℝ) (C : ℝ) 
  (ha : a = 6) (hcosB : cosB = 4/5) (hC : C = Real.pi / 4) : 
  (length_of_side_c a cosB C = 5 * Real.sqrt 2) ∧ 
  (cos_A_minus_pi_over_6 cosB (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2)))) (Real.sqrt (1 - (- (cosB * (Real.sqrt 2 / 2) - (Real.sqrt (1 - cosB^2) * (Real.sqrt 2 / 2))))^2)) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20) :=
by 
  sorry

end triangle_problem_l93_9390


namespace min_value_l93_9383

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 10 * x - 6 * y + 3

theorem min_value : ∃ x y : ℝ, (x + y = 2) ∧ (f x y = -(1/7)) :=
by
  sorry

end min_value_l93_9383


namespace fraction_power_multiplication_l93_9371

theorem fraction_power_multiplication :
  ( (8 / 9)^3 * (5 / 3)^3 ) = (64000 / 19683) :=
by
  sorry

end fraction_power_multiplication_l93_9371


namespace natasha_avg_speed_climbing_l93_9336

-- Natasha climbs up a hill in 4 hours and descends in 2 hours.
-- Her average speed along the whole journey is 1.5 km/h.
-- Prove that her average speed while climbing to the top is 1.125 km/h.

theorem natasha_avg_speed_climbing (v_up v_down : ℝ) :
  (4 * v_up = 2 * v_down) ∧ (1.5 = (2 * (4 * v_up) / 6)) → v_up = 1.125 :=
by
  -- We provide no proof here; this is just the statement.
  sorry

end natasha_avg_speed_climbing_l93_9336


namespace find_n_in_arithmetic_sequence_l93_9373

theorem find_n_in_arithmetic_sequence 
  (a : ℕ → ℕ)
  (a_1 : ℕ)
  (d : ℕ) 
  (a_n : ℕ) 
  (n : ℕ)
  (h₀ : a_1 = 11)
  (h₁ : d = 2)
  (h₂ : a n = a_1 + (n - 1) * d)
  (h₃ : a n = 2009) :
  n = 1000 := 
by
  -- The proof steps would go here
  sorry

end find_n_in_arithmetic_sequence_l93_9373


namespace number_of_diet_soda_bottles_l93_9356

theorem number_of_diet_soda_bottles (apples regular_soda total_bottles diet_soda : ℕ)
    (h_apples : apples = 36)
    (h_regular_soda : regular_soda = 80)
    (h_total_bottles : total_bottles = apples + 98)
    (h_diet_soda_eq : total_bottles = regular_soda + diet_soda) :
    diet_soda = 54 := by
  sorry

end number_of_diet_soda_bottles_l93_9356


namespace transform_equation_to_polynomial_l93_9384

variable (x y : ℝ)

theorem transform_equation_to_polynomial (h : (x^2 + 2) / (x + 1) = y) :
    (x^2 + 2) / (x + 1) + (5 * (x + 1)) / (x^2 + 2) = 6 → y^2 - 6 * y + 5 = 0 :=
by
  intro h_eq
  sorry

end transform_equation_to_polynomial_l93_9384


namespace factor_polynomial_l93_9309

theorem factor_polynomial (x : ℤ) :
  36 * x ^ 6 - 189 * x ^ 12 + 81 * x ^ 9 = 9 * x ^ 6 * (4 + 9 * x ^ 3 - 21 * x ^ 6) := 
sorry

end factor_polynomial_l93_9309


namespace vitya_catchup_time_l93_9319

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end vitya_catchup_time_l93_9319


namespace interest_rate_proof_l93_9388

variable (P : ℝ) (n : ℕ) (CI SI : ℝ → ℝ → ℕ → ℝ) (diff : ℝ → ℝ → ℝ)

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n
def simple_interest (P r : ℝ) (n : ℕ) : ℝ := P * r * n

theorem interest_rate_proof (r : ℝ) :
  diff (compound_interest 5400 r 2) (simple_interest 5400 r 2) = 216 → r = 0.2 :=
by sorry

end interest_rate_proof_l93_9388


namespace find_c_plus_d_l93_9329

theorem find_c_plus_d (c d : ℝ) (h1 : 2 * c = 6) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end find_c_plus_d_l93_9329


namespace completing_the_square_x_squared_minus_4x_plus_1_eq_0_l93_9399

theorem completing_the_square_x_squared_minus_4x_plus_1_eq_0 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x
  intros h
  sorry

end completing_the_square_x_squared_minus_4x_plus_1_eq_0_l93_9399


namespace fiona_initial_seat_l93_9323

theorem fiona_initial_seat (greg hannah ian jane kayla lou : Fin 7)
  (greg_final : Fin 7 := greg + 3)
  (hannah_final : Fin 7 := hannah - 2)
  (ian_final : Fin 7 := jane)
  (jane_final : Fin 7 := ian)
  (kayla_final : Fin 7 := kayla + 1)
  (lou_final : Fin 7 := lou - 2)
  (fiona_final : Fin 7) :
  (fiona_final = 0 ∨ fiona_final = 6) →
  ∀ (fiona_initial : Fin 7), 
  (greg_final ≠ fiona_initial ∧ hannah_final ≠ fiona_initial ∧ ian_final ≠ fiona_initial ∧ 
   jane_final ≠ fiona_initial ∧ kayla_final ≠ fiona_initial ∧ lou_final ≠ fiona_initial) →
  fiona_initial = 0 :=
by
  sorry

end fiona_initial_seat_l93_9323


namespace regular_polygon_sides_l93_9325

theorem regular_polygon_sides (exterior_angle : ℝ) (total_exterior_angle_sum : ℝ) (h1 : exterior_angle = 18) (h2 : total_exterior_angle_sum = 360) :
  let n := total_exterior_angle_sum / exterior_angle
  n = 20 :=
by
  sorry

end regular_polygon_sides_l93_9325


namespace triangle_is_right_l93_9333

-- Definitions based on the conditions given in the problem
variables {a b c A B C : ℝ}

-- Introduction of the conditions in Lean
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

def given_condition (A b c : ℝ) : Prop :=
  (Real.cos (A / 2))^2 = (b + c) / (2 * c)

-- Theorem statement to prove the conclusion based on given conditions
theorem triangle_is_right (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_given : given_condition A b c) :
  A = 90 := sorry

end triangle_is_right_l93_9333


namespace total_distance_biked_l93_9307

theorem total_distance_biked :
  let monday_distance := 12
  let tuesday_distance := 2 * monday_distance - 3
  let wednesday_distance := 2 * 11
  let thursday_distance := wednesday_distance + 2
  let friday_distance := thursday_distance + 2
  let saturday_distance := friday_distance + 2
  let sunday_distance := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance + saturday_distance + sunday_distance = 151 := 
by
  sorry

end total_distance_biked_l93_9307


namespace copper_needed_l93_9367

theorem copper_needed (T : ℝ) (lead_percentage : ℝ) (lead_weight : ℝ) (copper_percentage : ℝ) 
  (h_lead_percentage : lead_percentage = 0.25)
  (h_lead_weight : lead_weight = 5)
  (h_copper_percentage : copper_percentage = 0.60)
  (h_total_weight : T = lead_weight / lead_percentage) :
  copper_percentage * T = 12 := 
by
  sorry

end copper_needed_l93_9367


namespace angle_half_in_first_quadrant_l93_9301

theorem angle_half_in_first_quadrant (α : ℝ) (hα : 90 < α ∧ α < 180) : 0 < α / 2 ∧ α / 2 < 90 := 
sorry

end angle_half_in_first_quadrant_l93_9301


namespace arithmetic_sequence_value_y_l93_9382

theorem arithmetic_sequence_value_y :
  ∀ (a₁ a₃ y : ℤ), 
  a₁ = 3 ^ 3 →
  a₃ = 5 ^ 3 →
  y = (a₁ + a₃) / 2 →
  y = 76 :=
by 
  intros a₁ a₃ y h₁ h₃ hy 
  sorry

end arithmetic_sequence_value_y_l93_9382


namespace simultaneous_equations_solution_exists_l93_9314

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end simultaneous_equations_solution_exists_l93_9314


namespace part_I_intersection_part_I_union_complements_part_II_range_l93_9335

namespace MathProof

-- Definitions of the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Prove that the intersection of A and B is {x | 3 < x ∧ x < 6}
theorem part_I_intersection : A ∩ B = {x | 3 < x ∧ x < 6} := sorry

-- Prove that the union of the complements of A and B is {x | x ≤ 3 ∨ x ≥ 6}
theorem part_I_union_complements : (Aᶜ ∪ Bᶜ) = {x | x ≤ 3 ∨ x ≥ 6} := sorry

-- Prove the range of a such that C is a subset of B and B union C equals B
theorem part_II_range (a : ℝ) : B ∪ C a = B → (a ≤ 1 ∨ 2 ≤ a ∧ a ≤ 5) := sorry

end MathProof

end part_I_intersection_part_I_union_complements_part_II_range_l93_9335


namespace smallest_n_l93_9355

theorem smallest_n (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3)
  (h4 : x * y * z ∣ (x + y + z)^n) : n = 13 :=
sorry

end smallest_n_l93_9355


namespace jerome_family_members_l93_9353

-- Define the conditions of the problem
variables (C F M T : ℕ)
variables (hC : C = 20) (hF : F = C / 2) (hT : T = 33)

-- Formulate the theorem to prove
theorem jerome_family_members :
  M = T - (C + F) :=
sorry

end jerome_family_members_l93_9353


namespace range_of_x_l93_9397

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_x (x m : ℝ) (hx : x > -2 ∧ x < 2/3) (hm : m ≥ -2 ∧ m ≤ 2) :
    f (m * x - 2) + f x < 0 := sorry

end range_of_x_l93_9397


namespace smallest_factor_of_32_not_8_l93_9322

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ (n ∣ 32) ∧ ¬(n ∣ 8) ∧ ∀ m : ℕ, (m ∣ 32) ∧ ¬(m ∣ 8) → n ≤ m :=
by
  sorry

end smallest_factor_of_32_not_8_l93_9322


namespace exists_four_digit_number_sum_digits_14_divisible_by_14_l93_9316

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100 % 10) % 10 + (n / 10 % 10) % 10 + (n % 10)

theorem exists_four_digit_number_sum_digits_14_divisible_by_14 :
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ sum_of_digits n = 14 ∧ n % 14 = 0 :=
sorry

end exists_four_digit_number_sum_digits_14_divisible_by_14_l93_9316


namespace find_number_with_divisors_condition_l93_9305

theorem find_number_with_divisors_condition :
  ∃ n : ℕ, (∃ d1 d2 d3 d4 : ℕ, 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 * d4 ∣ n ∧
    d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 = n) ∧ n = 130 :=
by
  sorry

end find_number_with_divisors_condition_l93_9305


namespace matchstick_triangles_l93_9387

/-- Using 12 equal-length matchsticks, it is possible to form an isosceles triangle, an equilateral triangle, and a right-angled triangle without breaking or overlapping the matchsticks. --/
theorem matchstick_triangles :
  ∃ a b c : ℕ, a + b + c = 12 ∧ (a = b ∨ b = c ∨ a = c) ∧ (a * a + b * b = c * c ∨ a = b ∧ b = c) :=
by
  sorry

end matchstick_triangles_l93_9387


namespace proposition_B_correct_l93_9330

theorem proposition_B_correct (a b c : ℝ) (hc : c ≠ 0) : ac^2 > b * c^2 → a > b := sorry

end proposition_B_correct_l93_9330


namespace triangle_angle_and_perimeter_l93_9357

/-
In a triangle ABC, given c * sin B = sqrt 3 * cos C,
prove that angle C equals pi / 3,
and given a + b = 6, find the minimum perimeter of triangle ABC.
-/
theorem triangle_angle_and_perimeter (A B C : ℝ) (a b c : ℝ) 
  (h1 : c * Real.sin B = Real.sqrt 3 * Real.cos C)
  (h2 : a + b = 6) :
  C = Real.pi / 3 ∧ a + b + (Real.sqrt (36 - a * b)) = 9 :=
by
  sorry

end triangle_angle_and_perimeter_l93_9357


namespace sum_of_even_sequence_is_194_l93_9351

theorem sum_of_even_sequence_is_194
  (a b c d : ℕ) 
  (even_a : a % 2 = 0) 
  (even_b : b % 2 = 0) 
  (even_c : c % 2 = 0) 
  (even_d : d % 2 = 0)
  (a_lt_b : a < b) 
  (b_lt_c : b < c) 
  (c_lt_d : c < d)
  (diff_da : d - a = 90)
  (arith_ab_c : 2 * b = a + c)
  (geo_bc_d : c^2 = b * d)
  : a + b + c + d = 194 := 
sorry

end sum_of_even_sequence_is_194_l93_9351


namespace rectangle_inscribed_circle_circumference_l93_9334

/-- A 9 cm by 12 cm rectangle is inscribed in a circle. The circumference of the circle is 15π cm. -/
theorem rectangle_inscribed_circle_circumference :
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  circumference = 15 * Real.pi :=
by
  let width := 9
  let height := 12
  let diameter := Real.sqrt ((width)^2 + (height)^2)
  let circumference := Real.pi * diameter
  have h_diameter : diameter = 15 := by
    sorry
  have h_circumference : circumference = 15 * Real.pi := by
    sorry
  exact h_circumference

end rectangle_inscribed_circle_circumference_l93_9334


namespace gcd_possible_values_count_l93_9374

theorem gcd_possible_values_count (a b : ℕ) (h_ab : a * b = 360) : 
  (∃ d, d = Nat.gcd a b ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 12)) ∧ 
  (∃ n, n = 6) := 
by
  sorry

end gcd_possible_values_count_l93_9374


namespace solve_equation_l93_9345

theorem solve_equation : ∃ x : ℝ, (x^3 - ⌊x⌋ = 3) := 
sorry

end solve_equation_l93_9345


namespace non_officers_count_l93_9320

theorem non_officers_count (avg_salary_all : ℕ) (avg_salary_officers : ℕ) (avg_salary_non_officers : ℕ) (num_officers : ℕ) 
  (N : ℕ) 
  (h_avg_salary_all : avg_salary_all = 120) 
  (h_avg_salary_officers : avg_salary_officers = 430) 
  (h_avg_salary_non_officers : avg_salary_non_officers = 110) 
  (h_num_officers : num_officers = 15) 
  (h_eq : avg_salary_all * (num_officers + N) = avg_salary_officers * num_officers + avg_salary_non_officers * N) 
  : N = 465 :=
by
  -- Proof would be here
  sorry

end non_officers_count_l93_9320


namespace lcm_of_ratio_and_hcf_l93_9395

theorem lcm_of_ratio_and_hcf (a b : ℕ) (x : ℕ) (h_ratio : a = 3 * x ∧ b = 4 * x) (h_hcf : Nat.gcd a b = 4) : Nat.lcm a b = 48 :=
by
  sorry

end lcm_of_ratio_and_hcf_l93_9395


namespace sqrt_of_expression_l93_9389

theorem sqrt_of_expression :
  Real.sqrt (4^4 * 9^2) = 144 :=
sorry

end sqrt_of_expression_l93_9389


namespace length_width_ratio_l93_9310

theorem length_width_ratio 
  (W : ℕ) (P : ℕ) (L : ℕ)
  (hW : W = 90) 
  (hP : P = 432) 
  (hP_eq : P = 2 * L + 2 * W) : 
  (L / W = 7 / 5) := 
  sorry

end length_width_ratio_l93_9310


namespace string_cheese_packages_l93_9308

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end string_cheese_packages_l93_9308


namespace parallel_lines_l93_9332

theorem parallel_lines (a : ℝ) :
  ((3 * a + 2) * x + a * y + 6 = 0) ↔
  (a * x - y + 3 = 0) →
  a = -1 :=
by sorry

end parallel_lines_l93_9332


namespace jar_ratios_l93_9386

theorem jar_ratios (C_X C_Y : ℝ) 
  (h1 : 0 < C_X) 
  (h2 : 0 < C_Y)
  (h3 : (1/2) * C_X + (1/2) * C_Y = (3/4) * C_X) : 
  C_Y = (1/2) * C_X := 
sorry

end jar_ratios_l93_9386


namespace exists_k_consecutive_squareful_numbers_l93_9340

-- Define what it means for a number to be squareful
def is_squareful (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

-- State the theorem
theorem exists_k_consecutive_squareful_numbers (k : ℕ) : 
  ∃ (a : ℕ), ∀ i, i < k → is_squareful (a + i) :=
sorry

end exists_k_consecutive_squareful_numbers_l93_9340


namespace ben_david_bagel_cost_l93_9344

theorem ben_david_bagel_cost (B D : ℝ)
  (h1 : D = 0.5 * B)
  (h2 : B = D + 16) :
  B + D = 48 := 
sorry

end ben_david_bagel_cost_l93_9344


namespace mod_remainder_l93_9398

theorem mod_remainder (a b c d : ℕ) (h1 : a = 11) (h2 : b = 9) (h3 : c = 7) (h4 : d = 7) :
  (a^d + b^(d + 1) + c^(d + 2)) % d = 1 := 
by 
  sorry

end mod_remainder_l93_9398


namespace area_of_30_60_90_triangle_l93_9349

theorem area_of_30_60_90_triangle (altitude : ℝ) (h : altitude = 3) : 
  ∃ (area : ℝ), area = 6 * Real.sqrt 3 := 
sorry

end area_of_30_60_90_triangle_l93_9349


namespace boat_distance_against_stream_l93_9324

-- Define the speed of the boat in still water
def speed_boat_still : ℝ := 8

-- Define the distance covered by the boat along the stream in one hour
def distance_along_stream : ℝ := 11

-- Define the time duration for the journey
def time_duration : ℝ := 1

-- Define the speed of the stream
def speed_stream : ℝ := distance_along_stream - speed_boat_still

-- Define the speed of the boat against the stream
def speed_against_stream : ℝ := speed_boat_still - speed_stream

-- Define the distance covered by the boat against the stream in one hour
def distance_against_stream (t : ℝ) : ℝ := speed_against_stream * t

-- The main theorem: The boat travels 5 km against the stream in one hour
theorem boat_distance_against_stream : distance_against_stream time_duration = 5 := by
  sorry

end boat_distance_against_stream_l93_9324


namespace least_positive_integer_to_add_l93_9364

theorem least_positive_integer_to_add (n : ℕ) (h1 : n > 0) (h2 : (624 + n) % 5 = 0) : n = 1 := 
by
  sorry

end least_positive_integer_to_add_l93_9364


namespace vector_sum_l93_9302

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (-1, 0, 1)

-- Define the target vector c
def c : ℝ × ℝ × ℝ := (-1, 2, 5)

-- State the theorem to be proven
theorem vector_sum : a + (2:ℝ) • b = c :=
by 
  -- Not providing the proof, just adding a sorry
  sorry

end vector_sum_l93_9302


namespace number_of_animals_per_aquarium_l93_9347

variable (aq : ℕ) (ani : ℕ) (a : ℕ)

axiom condition1 : aq = 26
axiom condition2 : ani = 52
axiom condition3 : ani = aq * a

theorem number_of_animals_per_aquarium : a = 2 :=
by
  sorry

end number_of_animals_per_aquarium_l93_9347


namespace actual_distance_traveled_l93_9379

theorem actual_distance_traveled
  (D : ℝ) 
  (H : ∃ T : ℝ, D = 5 * T ∧ D + 20 = 15 * T) : 
  D = 10 :=
by
  sorry

end actual_distance_traveled_l93_9379
