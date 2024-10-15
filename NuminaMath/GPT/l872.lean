import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l872_87229

variables (x y : ℝ)

theorem problem_statement
  (h1 : abs x = 4)
  (h2 : abs y = 2)
  (h3 : abs (x + y) = x + y) : 
  x - y = 2 ∨ x - y = 6 :=
sorry

end NUMINAMATH_GPT_problem_statement_l872_87229


namespace NUMINAMATH_GPT_JaneReadingSpeed_l872_87287

theorem JaneReadingSpeed (total_pages read_second_half_speed total_days pages_first_half days_first_half_speed : ℕ)
  (h1 : total_pages = 500)
  (h2 : read_second_half_speed = 5)
  (h3 : total_days = 75)
  (h4 : pages_first_half = 250)
  (h5 : days_first_half_speed = pages_first_half / (total_days - (pages_first_half / read_second_half_speed))) :
  days_first_half_speed = 10 := by
  sorry

end NUMINAMATH_GPT_JaneReadingSpeed_l872_87287


namespace NUMINAMATH_GPT_total_weight_of_lifts_l872_87246

theorem total_weight_of_lifts
  (F S : ℕ)
  (h1 : F = 600)
  (h2 : 2 * F = S + 300) :
  F + S = 1500 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_lifts_l872_87246


namespace NUMINAMATH_GPT_find_m_l872_87265

-- Define the function with given conditions
def f (m : ℕ) (n : ℕ) : ℕ := 
if n > m^2 then n - m + 14 else sorry

-- Define the main problem
theorem find_m (m : ℕ) (hyp : m ≥ 14) : f m 1995 = 1995 ↔ m = 14 ∨ m = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l872_87265


namespace NUMINAMATH_GPT_identify_letter_X_l872_87294

-- Define the conditions
def date_behind_D (z : ℕ) : ℕ := z
def date_behind_E (z : ℕ) : ℕ := z + 1
def date_behind_F (z : ℕ) : ℕ := z + 14

-- Define the sum condition
def sum_date_E_F (z : ℕ) : ℕ := date_behind_E z + date_behind_F z

-- Define the target date behind another letter
def target_date_behind_another_letter (z : ℕ) : ℕ := z + 15

-- Theorem statement
theorem identify_letter_X (z : ℕ) :
  ∃ (x : Char), sum_date_E_F z = date_behind_D z + target_date_behind_another_letter z → x = 'X' :=
by
  -- The actual proof would go here; we'll defer it for now
  sorry

end NUMINAMATH_GPT_identify_letter_X_l872_87294


namespace NUMINAMATH_GPT_integer_pairs_satisfying_equation_l872_87283

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 →
    (x = 1 ∧ y = 12) ∨ (x = 1 ∧ y = -12) ∨ 
    (x = -9 ∧ y = 12) ∨ (x = -9 ∧ y = -12) ∨ 
    (x = -4 ∧ y = 12) ∨ (x = -4 ∧ y = -12) ∨ 
    (x = 0 ∧ y = 0) ∨ (x = -8 ∧ y = 0) ∨ 
    (x = -1 ∧ y = 0) ∨ (x = -7 ∧ y = 0) :=
by sorry

end NUMINAMATH_GPT_integer_pairs_satisfying_equation_l872_87283


namespace NUMINAMATH_GPT_domain_correct_l872_87269

def domain_of_function (x : ℝ) : Prop :=
  (x > 2) ∧ (x ≠ 5)

theorem domain_correct : {x : ℝ | domain_of_function x} = {x : ℝ | x > 2 ∧ x ≠ 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_correct_l872_87269


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l872_87221

-- Statement for the first problem
theorem simplify_expression_1 (a : ℝ) : 2 * a * (a - 3) - a^2 = a^2 - 6 * a := 
by sorry

-- Statement for the second problem
theorem simplify_expression_2 (x : ℝ) : (x - 1) * (x + 2) - x * (x + 1) = -2 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l872_87221


namespace NUMINAMATH_GPT_joan_change_received_l872_87257

/-- Definition of the cat toy cost -/
def cat_toy_cost : ℝ := 8.77

/-- Definition of the cage cost -/
def cage_cost : ℝ := 10.97

/-- Definition of the total cost -/
def total_cost : ℝ := cat_toy_cost + cage_cost

/-- Definition of the payment amount -/
def payment : ℝ := 20.00

/-- Definition of the change received -/
def change_received : ℝ := payment - total_cost

/-- Statement proving that Joan received $0.26 in change -/
theorem joan_change_received : change_received = 0.26 := by
  sorry

end NUMINAMATH_GPT_joan_change_received_l872_87257


namespace NUMINAMATH_GPT_crumble_topping_correct_amount_l872_87247

noncomputable def crumble_topping_total_mass (flour butter sugar : ℕ) (factor : ℚ) : ℚ :=
  factor * (flour + butter + sugar) / 1000  -- convert grams to kilograms

theorem crumble_topping_correct_amount {flour butter sugar : ℕ} (factor : ℚ) (h_flour : flour = 100) (h_butter : butter = 50) (h_sugar : sugar = 50) (h_factor : factor = 2.5) :
  crumble_topping_total_mass flour butter sugar factor = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_crumble_topping_correct_amount_l872_87247


namespace NUMINAMATH_GPT_max_squares_covered_by_card_l872_87278

theorem max_squares_covered_by_card (side_len : ℕ) (card_side : ℕ) : 
  side_len = 1 → card_side = 2 → n ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_max_squares_covered_by_card_l872_87278


namespace NUMINAMATH_GPT_sector_angle_measure_l872_87253

theorem sector_angle_measure (r α : ℝ) 
  (h1 : 2 * r + α * r = 6)
  (h2 : (1 / 2) * α * r^2 = 2) :
  α = 1 ∨ α = 4 := 
sorry

end NUMINAMATH_GPT_sector_angle_measure_l872_87253


namespace NUMINAMATH_GPT_braden_total_money_after_winning_bet_l872_87266

def initial_amount : ℕ := 400
def factor : ℕ := 2
def winnings (initial_amount : ℕ) (factor : ℕ) : ℕ := factor * initial_amount

theorem braden_total_money_after_winning_bet : 
  winnings initial_amount factor + initial_amount = 1200 := 
by
  sorry

end NUMINAMATH_GPT_braden_total_money_after_winning_bet_l872_87266


namespace NUMINAMATH_GPT_bigger_part_of_dividing_56_l872_87252

theorem bigger_part_of_dividing_56 (x y : ℕ) (h₁ : x + y = 56) (h₂ : 10 * x + 22 * y = 780) : max x y = 38 :=
by
  sorry

end NUMINAMATH_GPT_bigger_part_of_dividing_56_l872_87252


namespace NUMINAMATH_GPT_euler_criterion_l872_87276

theorem euler_criterion (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) (ha : 1 ≤ a ∧ a ≤ p - 1) : 
  (∃ b : ℕ, b^2 % p = a % p) ↔ a^((p - 1) / 2) % p = 1 :=
sorry

end NUMINAMATH_GPT_euler_criterion_l872_87276


namespace NUMINAMATH_GPT_A_less_B_C_A_relationship_l872_87227

variable (a : ℝ)
def A := a + 2
def B := 2 * a^2 - 3 * a + 10
def C := a^2 + 5 * a - 3

theorem A_less_B : A a - B a < 0 := by
  sorry

theorem C_A_relationship :
  if a < -5 then C a > A a
  else if a = -5 then C a = A a
  else if a < 1 then C a < A a
  else if a = 1 then C a = A a
  else C a > A a := by
  sorry

end NUMINAMATH_GPT_A_less_B_C_A_relationship_l872_87227


namespace NUMINAMATH_GPT_donna_pays_total_l872_87297

def original_price_vase : ℝ := 250
def discount_vase : ℝ := original_price_vase * 0.25

def original_price_teacups : ℝ := 350
def discount_teacups : ℝ := original_price_teacups * 0.30

def original_price_plate : ℝ := 450
def discount_plate : ℝ := 0

def original_price_ornament : ℝ := 150
def discount_ornament : ℝ := original_price_ornament * 0.20

def membership_discount_vase : ℝ := (original_price_vase - discount_vase) * 0.05
def membership_discount_plate : ℝ := original_price_plate * 0.05

def tax_vase : ℝ := ((original_price_vase - discount_vase - membership_discount_vase) * 0.12)
def tax_teacups : ℝ := ((original_price_teacups - discount_teacups) * 0.08)
def tax_plate : ℝ := ((original_price_plate - membership_discount_plate) * 0.10)
def tax_ornament : ℝ := ((original_price_ornament - discount_ornament) * 0.06)

def final_price_vase : ℝ := (original_price_vase - discount_vase - membership_discount_vase) + tax_vase
def final_price_teacups : ℝ := (original_price_teacups - discount_teacups) + tax_teacups
def final_price_plate : ℝ := (original_price_plate - membership_discount_plate) + tax_plate
def final_price_ornament : ℝ := (original_price_ornament - discount_ornament) + tax_ornament

def total_price : ℝ := final_price_vase + final_price_teacups + final_price_plate + final_price_ornament

theorem donna_pays_total :
  total_price = 1061.55 :=
by
  sorry

end NUMINAMATH_GPT_donna_pays_total_l872_87297


namespace NUMINAMATH_GPT_age_problem_l872_87242

theorem age_problem (my_age mother_age : ℕ) 
  (h1 : mother_age = 3 * my_age) 
  (h2 : my_age + mother_age = 40)
  : my_age = 10 :=
by 
  sorry

end NUMINAMATH_GPT_age_problem_l872_87242


namespace NUMINAMATH_GPT_intersection_A_B_l872_87254

-- Defining the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Stating the theorem that A ∩ B equals (1, 2)
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l872_87254


namespace NUMINAMATH_GPT_man_speed_is_correct_l872_87281

noncomputable def train_length : ℝ := 275
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def time_seconds : ℝ := 14.998800095992323

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_length / time_seconds
noncomputable def man_speed_ms : ℝ := relative_speed_ms - train_speed_ms
noncomputable def man_speed_kmh : ℝ := man_speed_ms * (3600 / 1000)
noncomputable def expected_man_speed_kmh : ℝ := 6.006

theorem man_speed_is_correct : abs (man_speed_kmh - expected_man_speed_kmh) < 0.001 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_man_speed_is_correct_l872_87281


namespace NUMINAMATH_GPT_problem_statement_l872_87230

variable (a : ℝ)

theorem problem_statement (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := 
by sorry

end NUMINAMATH_GPT_problem_statement_l872_87230


namespace NUMINAMATH_GPT_min_x9_minus_x1_l872_87285

theorem min_x9_minus_x1
  (x : Fin 9 → ℕ)
  (h_pos : ∀ i, x i > 0)
  (h_sorted : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.univ.sum x) = 220) :
    ∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ,
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6 ∧ x6 < x7 ∧ x7 < x8 ∧ x8 < x9 ∧
    (x1 + x2 + x3 + x4 + x5 = 110) ∧
    x1 = x 0 ∧ x2 = x 1 ∧ x3 = x 2 ∧ x4 = x 3 ∧ x5 = x 4 ∧ x6 = x 5 ∧ x7 = x 6 ∧ x8 = x 7 ∧ x9 = x 8
    ∧ (x9 - x1 = 9) :=
sorry

end NUMINAMATH_GPT_min_x9_minus_x1_l872_87285


namespace NUMINAMATH_GPT_total_puppies_count_l872_87209

def first_week_puppies : Nat := 20
def second_week_puppies : Nat := 2 * first_week_puppies / 5
def third_week_puppies : Nat := 3 * second_week_puppies / 8
def fourth_week_puppies : Nat := 2 * second_week_puppies
def fifth_week_puppies : Nat := first_week_puppies + 10
def sixth_week_puppies : Nat := 2 * third_week_puppies - 5
def seventh_week_puppies : Nat := 2 * sixth_week_puppies
def eighth_week_puppies : Nat := 5 * seventh_week_puppies / 6 / 1 -- Assuming rounding down to nearest whole number

def total_puppies : Nat :=
  first_week_puppies + second_week_puppies + third_week_puppies +
  fourth_week_puppies + fifth_week_puppies + sixth_week_puppies +
  seventh_week_puppies + eighth_week_puppies

theorem total_puppies_count : total_puppies = 81 := by
  sorry

end NUMINAMATH_GPT_total_puppies_count_l872_87209


namespace NUMINAMATH_GPT_relationship_abc_l872_87261

open Real

variable {x : ℝ}
variable (a b c : ℝ)
variable (h1 : 0 < x ∧ x ≤ 1)
variable (h2 : a = (sin x / x) ^ 2)
variable (h3 : b = sin x / x)
variable (h4 : c = sin (x^2) / x^2)

theorem relationship_abc (h1 : 0 < x ∧ x ≤ 1) (h2 : a = (sin x / x) ^ 2) (h3 : b = sin x / x) (h4 : c = sin (x^2) / x^2) :
  a < b ∧ b ≤ c :=
sorry

end NUMINAMATH_GPT_relationship_abc_l872_87261


namespace NUMINAMATH_GPT_number_of_desired_numbers_l872_87202

-- Define a predicate for a four-digit number with the thousands digit 3
def isDesiredNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (n / 1000) % 10 = 3

-- Statement of the theorem
theorem number_of_desired_numbers : 
  ∃ k, k = 1000 ∧ (∀ n, isDesiredNumber n ↔ 3000 ≤ n ∧ n < 4000) := 
by
  -- Proof omitted, using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_number_of_desired_numbers_l872_87202


namespace NUMINAMATH_GPT_unique_exponential_function_l872_87289

theorem unique_exponential_function (g : ℝ → ℝ) :
  (∀ x1 x2 : ℝ, g (x1 + x2) = g x1 * g x2) →
  g 1 = 3 →
  (∀ x1 x2 : ℝ, x1 < x2 → g x1 < g x2) →
  ∀ x : ℝ, g x = 3^x :=
by
  sorry

end NUMINAMATH_GPT_unique_exponential_function_l872_87289


namespace NUMINAMATH_GPT_find_cost_of_crackers_l872_87203

-- Definitions based on the given conditions
def cost_hamburger_meat : ℝ := 5.00
def cost_per_bag_vegetables : ℝ := 2.00
def number_of_bags_vegetables : ℕ := 4
def cost_cheese : ℝ := 3.50
def discount_rate : ℝ := 0.10
def total_after_discount : ℝ := 18

-- Definition of the box of crackers, which we aim to prove
def cost_crackers : ℝ := 3.50

-- The Lean statement for the proof
theorem find_cost_of_crackers
  (C : ℝ)
  (h : C = cost_crackers)
  (H : 0.9 * (cost_hamburger_meat + cost_per_bag_vegetables * number_of_bags_vegetables + cost_cheese + C) = total_after_discount) :
  C = 3.50 :=
  sorry

end NUMINAMATH_GPT_find_cost_of_crackers_l872_87203


namespace NUMINAMATH_GPT_circle_area_difference_l872_87225

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (h1 : r1 = 30) (h2 : r2 = 7.5) : 
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_circle_area_difference_l872_87225


namespace NUMINAMATH_GPT_cesar_watched_fraction_l872_87296

theorem cesar_watched_fraction
  (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ)
  (h1 : total_seasons = 12)
  (h2 : episodes_per_season = 20)
  (h3 : remaining_episodes = 160) :
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_cesar_watched_fraction_l872_87296


namespace NUMINAMATH_GPT_net_effect_on_sale_value_l872_87290

theorem net_effect_on_sale_value
(P Q : ℝ)
(h_new_price : ∃ P', P' = P - 0.22 * P)
(h_new_qty : ∃ Q', Q' = Q + 0.86 * Q) :
  let original_sale_value := P * Q
  let new_sale_value := (0.78 * P) * (1.86 * Q)
  let net_effect := ((new_sale_value / original_sale_value - 1) * 100 : ℝ)
  net_effect = 45.08 :=
by {
  sorry
}

end NUMINAMATH_GPT_net_effect_on_sale_value_l872_87290


namespace NUMINAMATH_GPT_problem1_problem2_l872_87292

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x 2 ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 :=
  sorry -- the proof goes here

theorem problem2 (a : ℝ) (h₁ : 1 < a) : 
  (∀ x : ℝ, f x a + |x - 1| ≥ 1) ∧ (2 ≤ a) :=
  sorry -- the proof goes here

end NUMINAMATH_GPT_problem1_problem2_l872_87292


namespace NUMINAMATH_GPT_arithmetic_sequence_l872_87295

theorem arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) (h : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l872_87295


namespace NUMINAMATH_GPT_sqrt_abc_sum_eq_54_sqrt_5_l872_87282

theorem sqrt_abc_sum_eq_54_sqrt_5 
  (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 18) 
  (h3 : a + b = 19) : 
  Real.sqrt (a * b * c * (a + b + c)) = 54 * Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_abc_sum_eq_54_sqrt_5_l872_87282


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l872_87299

theorem number_of_correct_conclusions : 
    (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
    (∀ x : ℝ, (x ≠ 0 → x - Real.sin x ≠ 0)) ∧
    (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
    (¬ (∀ x : ℝ, x - Real.log x > 0))
    → 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l872_87299


namespace NUMINAMATH_GPT_vertical_angles_congruent_l872_87258

-- Define what it means for two angles to be vertical angles
def areVerticalAngles (a b : ℝ) : Prop := -- placeholder definition
  sorry

-- Define what it means for two angles to be congruent
def areCongruentAngles (a b : ℝ) : Prop := a = b

-- State the problem in the form of a theorem
theorem vertical_angles_congruent (a b : ℝ) :
  areVerticalAngles a b → areCongruentAngles a b := by
  sorry

end NUMINAMATH_GPT_vertical_angles_congruent_l872_87258


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l872_87204

theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ e : ℝ, e = 5 / 4 ∧ (∀ x y : ℝ, (x^2 / 16) - (y^2 / m) = 1)) → m = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l872_87204


namespace NUMINAMATH_GPT_angle_B_in_triangle_l872_87273

theorem angle_B_in_triangle (A B C : ℝ) (hA : A = 60) (hB : B = 2 * C) (hSum : A + B + C = 180) : B = 80 :=
by sorry

end NUMINAMATH_GPT_angle_B_in_triangle_l872_87273


namespace NUMINAMATH_GPT_circle_center_l872_87220

theorem circle_center (x y : ℝ) : ∀ (h k : ℝ), (x^2 - 6*x + y^2 + 2*y = 9) → (x - h)^2 + (y - k)^2 = 19 → h = 3 ∧ k = -1 :=
by
  intros h k h_eq c_eq
  sorry

end NUMINAMATH_GPT_circle_center_l872_87220


namespace NUMINAMATH_GPT_total_bears_l872_87277

-- Definitions based on given conditions
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27

-- Theorem to prove the total number of bears
theorem total_bears : brown_bears + white_bears + black_bears = 66 := by
  sorry

end NUMINAMATH_GPT_total_bears_l872_87277


namespace NUMINAMATH_GPT_similar_triangles_area_ratio_l872_87248

theorem similar_triangles_area_ratio (r : ℚ) (h : r = 1/3) : (r^2) = 1/9 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_area_ratio_l872_87248


namespace NUMINAMATH_GPT_shaded_area_l872_87241

theorem shaded_area 
  (R r : ℝ) 
  (h_area_larger_circle : π * R ^ 2 = 100 * π) 
  (h_shaded_larger_fraction : 2 / 3 = (area_shaded_larger / (π * R ^ 2))) 
  (h_relationship_radius : r = R / 2) 
  (h_area_smaller_circle : π * r ^ 2 = 25 * π)
  (h_shaded_smaller_fraction : 1 / 3 = (area_shaded_smaller / (π * r ^ 2))) : 
  (area_shaded_larger + area_shaded_smaller = 75 * π) := 
sorry

end NUMINAMATH_GPT_shaded_area_l872_87241


namespace NUMINAMATH_GPT_flight_distance_each_way_l872_87207

variables (D : ℝ) (T_out T_return total_time : ℝ)

-- Defining conditions
def condition1 : Prop := T_out = D / 300
def condition2 : Prop := T_return = D / 500
def condition3 : Prop := total_time = 8

-- Given conditions
axiom h1 : condition1 D T_out
axiom h2 : condition2 D T_return
axiom h3 : condition3 total_time

-- The proof problem statement
theorem flight_distance_each_way : T_out + T_return = total_time → D = 1500 :=
by
  sorry

end NUMINAMATH_GPT_flight_distance_each_way_l872_87207


namespace NUMINAMATH_GPT_lemonade_in_pitcher_l872_87217

theorem lemonade_in_pitcher (iced_tea lemonade total_pitcher total_in_drink lemonade_ratio : ℚ)
  (h1 : iced_tea = 1/4)
  (h2 : lemonade = 5/4)
  (h3 : total_in_drink = iced_tea + lemonade)
  (h4 : lemonade_ratio = lemonade / total_in_drink)
  (h5 : total_pitcher = 18) :
  (total_pitcher * lemonade_ratio) = 15 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_in_pitcher_l872_87217


namespace NUMINAMATH_GPT_proof_problem_l872_87288

noncomputable def f (x a : ℝ) : ℝ := (1 + x^2) * Real.exp x - a
noncomputable def f' (x a : ℝ) : ℝ := (1 + 2 * x + x^2) * Real.exp x
noncomputable def k_OP (a : ℝ) : ℝ := a - 2 / Real.exp 1
noncomputable def g (m : ℝ) : ℝ := Real.exp m - (m + 1)

theorem proof_problem (a m : ℝ) (h₁ : a > 0) (h₂ : f' (-1) a = 0) (h₃ : f' m a = k_OP a) 
  : m + 1 ≤ 3 * a - 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l872_87288


namespace NUMINAMATH_GPT_unique_identity_element_l872_87255

variable {G : Type*} [Group G]

theorem unique_identity_element (e e' : G) (h1 : ∀ g : G, e * g = g ∧ g * e = g) (h2 : ∀ g : G, e' * g = g ∧ g * e' = g) : e = e' :=
by 
sorry

end NUMINAMATH_GPT_unique_identity_element_l872_87255


namespace NUMINAMATH_GPT_initial_quantity_of_milk_in_container_A_l872_87267

variables {CA MB MC : ℝ}

theorem initial_quantity_of_milk_in_container_A (h1 : MB = 0.375 * CA)
    (h2 : MC = 0.625 * CA)
    (h_eq : MB + 156 = MC - 156) :
    CA = 1248 :=
by
  sorry

end NUMINAMATH_GPT_initial_quantity_of_milk_in_container_A_l872_87267


namespace NUMINAMATH_GPT_part_a_part_b_l872_87239

-- Step d: Lean statements for the proof problems
theorem part_a (p : ℕ) (hp : Nat.Prime p) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 :=
by {
  sorry
}

theorem part_b (p : ℕ) (hp : Nat.Prime p) : (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ (a^2 + b^2 + 2018) % p = 0 ∧ a % p ≠ 0 ∧ b % p ≠ 0) ↔ p ≠ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_part_a_part_b_l872_87239


namespace NUMINAMATH_GPT_product_a2_a3_a4_l872_87260

open Classical

noncomputable def geometric_sequence (a : ℕ → ℚ) (a1 : ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n - 1)

theorem product_a2_a3_a4 (a : ℕ → ℚ) (q : ℚ) 
  (h_seq : geometric_sequence a 1 q)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 1 / 9) :
  a 2 * a 3 * a 4 = 1 / 27 :=
sorry

end NUMINAMATH_GPT_product_a2_a3_a4_l872_87260


namespace NUMINAMATH_GPT_smallest_period_sin_cos_l872_87279

theorem smallest_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_smallest_period_sin_cos_l872_87279


namespace NUMINAMATH_GPT_problem_divisible_by_64_l872_87231

theorem problem_divisible_by_64 (n : ℕ) (hn : n > 0) : (3 ^ (2 * n + 2) - 8 * n - 9) % 64 = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_divisible_by_64_l872_87231


namespace NUMINAMATH_GPT_average_speed_last_segment_l872_87259

theorem average_speed_last_segment (D : ℝ) (T_mins : ℝ) (S1 S2 : ℝ) (t : ℝ) (S_last : ℝ) :
  D = 150 ∧ T_mins = 135 ∧ S1 = 50 ∧ S2 = 60 ∧ t = 45 →
  S_last = 90 :=
by
    sorry

end NUMINAMATH_GPT_average_speed_last_segment_l872_87259


namespace NUMINAMATH_GPT_average_of_four_given_conditions_l872_87263

noncomputable def average_of_four_integers : ℕ × ℕ × ℕ × ℕ → ℚ :=
  λ ⟨a, b, c, d⟩ => (a + b + c + d : ℚ) / 4

theorem average_of_four_given_conditions :
  ∀ (A B C D : ℕ), 
    (A + B) / 2 = 35 → 
    C = 130 → 
    D = 1 → 
    average_of_four_integers (A, B, C, D) = 50.25 := 
by
  intros A B C D hAB hC hD
  unfold average_of_four_integers
  sorry

end NUMINAMATH_GPT_average_of_four_given_conditions_l872_87263


namespace NUMINAMATH_GPT_globe_surface_area_l872_87233

theorem globe_surface_area (d : ℚ) (h : d = 9) : 
  4 * Real.pi * (d / 2) ^ 2 = 81 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_globe_surface_area_l872_87233


namespace NUMINAMATH_GPT_rowing_distance_upstream_l872_87201

theorem rowing_distance_upstream 
  (v : ℝ) (d : ℝ)
  (h1 : 75 = (v + 3) * 5)
  (h2 : d = (v - 3) * 5) :
  d = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_rowing_distance_upstream_l872_87201


namespace NUMINAMATH_GPT_triangular_number_30_l872_87251

theorem triangular_number_30 : (30 * (30 + 1)) / 2 = 465 :=
by
  sorry

end NUMINAMATH_GPT_triangular_number_30_l872_87251


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l872_87212

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → m * x + y + 1 = 0 → False) ↔ m = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l872_87212


namespace NUMINAMATH_GPT_mod_remainder_w_l872_87219

theorem mod_remainder_w (w : ℕ) (h : w = 3^39) : w % 13 = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_remainder_w_l872_87219


namespace NUMINAMATH_GPT_blocks_differ_in_two_ways_exactly_l872_87284

theorem blocks_differ_in_two_ways_exactly 
  (materials : Finset String := {"plastic", "wood", "metal"})
  (sizes : Finset String := {"small", "medium", "large"})
  (colors : Finset String := {"blue", "green", "red", "yellow"})
  (shapes : Finset String := {"circle", "hexagon", "square", "triangle"})
  (target : String := "plastic medium red circle") :
  ∃ (n : ℕ), n = 37 := by
  sorry

end NUMINAMATH_GPT_blocks_differ_in_two_ways_exactly_l872_87284


namespace NUMINAMATH_GPT_burrs_count_l872_87286

variable (B T : ℕ)

theorem burrs_count 
  (h1 : T = 6 * B) 
  (h2 : B + T = 84) : 
  B = 12 := 
by
  sorry

end NUMINAMATH_GPT_burrs_count_l872_87286


namespace NUMINAMATH_GPT_harbor_distance_l872_87271

-- Definitions from conditions
variable (d : ℝ)

-- Define the assumptions
def condition_dave := d < 10
def condition_elena := d > 9

-- The proof statement that the interval for d is (9, 10)
theorem harbor_distance (hd : condition_dave d) (he : condition_elena d) : d ∈ Set.Ioo 9 10 :=
sorry

end NUMINAMATH_GPT_harbor_distance_l872_87271


namespace NUMINAMATH_GPT_sum_of_three_squares_l872_87268

theorem sum_of_three_squares (a b c : ℤ) (h1 : 2 * a + 2 * b + c = 27) (h2 : a + 3 * b + c = 25) : 3 * c = 33 :=
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_l872_87268


namespace NUMINAMATH_GPT_Jeanine_gave_fraction_of_pencils_l872_87222

theorem Jeanine_gave_fraction_of_pencils
  (Jeanine_initial_pencils Clare_initial_pencils Jeanine_pencils_after Clare_pencils_after : ℕ)
  (h1 : Jeanine_initial_pencils = 18)
  (h2 : Clare_initial_pencils = Jeanine_initial_pencils / 2)
  (h3 : Jeanine_pencils_after = Clare_pencils_after + 3)
  (h4 : Clare_pencils_after = Clare_initial_pencils)
  (h5 : Jeanine_pencils_after + (Jeanine_initial_pencils - Jeanine_pencils_after) = Jeanine_initial_pencils) :
  (Jeanine_initial_pencils - Jeanine_pencils_after) / Jeanine_initial_pencils = 1 / 3 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_Jeanine_gave_fraction_of_pencils_l872_87222


namespace NUMINAMATH_GPT_john_horizontal_distance_l872_87270

theorem john_horizontal_distance
  (vertical_distance_ratio horizontal_distance_ratio : ℕ)
  (initial_elevation final_elevation : ℕ)
  (h_ratio : vertical_distance_ratio = 1)
  (h_dist_ratio : horizontal_distance_ratio = 3)
  (h_initial : initial_elevation = 500)
  (h_final : final_elevation = 3450) :
  (final_elevation - initial_elevation) * horizontal_distance_ratio = 8850 := 
by {
  sorry
}

end NUMINAMATH_GPT_john_horizontal_distance_l872_87270


namespace NUMINAMATH_GPT_total_prayers_in_a_week_l872_87213

def prayers_per_week (pastor_prayers : ℕ → ℕ) : ℕ :=
  (pastor_prayers 0) + (pastor_prayers 1) + (pastor_prayers 2) +
  (pastor_prayers 3) + (pastor_prayers 4) + (pastor_prayers 5) + (pastor_prayers 6)

def pastor_paul (day : ℕ) : ℕ :=
  if day = 6 then 40 else 20

def pastor_bruce (day : ℕ) : ℕ :=
  if day = 6 then 80 else 10

def pastor_caroline (day : ℕ) : ℕ :=
  if day = 6 then 30 else 10

theorem total_prayers_in_a_week :
  prayers_per_week pastor_paul + prayers_per_week pastor_bruce + prayers_per_week pastor_caroline = 390 :=
sorry

end NUMINAMATH_GPT_total_prayers_in_a_week_l872_87213


namespace NUMINAMATH_GPT_simplify_radicals_l872_87200

theorem simplify_radicals :
  (Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3)) = 2 * Real.sqrt 6 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_radicals_l872_87200


namespace NUMINAMATH_GPT_problem1_l872_87274

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end NUMINAMATH_GPT_problem1_l872_87274


namespace NUMINAMATH_GPT_probability_of_X_eq_2_l872_87240

-- Define the random variable distribution condition
def random_variable_distribution (a : ℝ) (P : ℝ → ℝ) : Prop :=
  P 1 = 1 / (2 * a) ∧ P 2 = 2 / (2 * a) ∧ P 3 = 3 / (2 * a) ∧
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) = 1)

-- State the theorem given the conditions and the result
theorem probability_of_X_eq_2 (a : ℝ) (P : ℝ → ℝ) (h : random_variable_distribution a P) : 
  P 2 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_of_X_eq_2_l872_87240


namespace NUMINAMATH_GPT_find_minimal_N_l872_87238

theorem find_minimal_N (N : ℕ) (l m n : ℕ) (h1 : (l - 1) * (m - 1) * (n - 1) = 252)
  (h2 : l ≥ 5 ∨ m ≥ 5 ∨ n ≥ 5) : N = l * m * n → N = 280 :=
by
  sorry

end NUMINAMATH_GPT_find_minimal_N_l872_87238


namespace NUMINAMATH_GPT_correct_statements_l872_87245

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

def satisfies_condition (f : ℝ → ℝ) : Prop := 
  ∀ x, f (1 - x) + f (1 + x) = 0

def is_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (a + x)

theorem correct_statements (f : ℝ → ℝ) :
  is_even f →
  is_monotonically_increasing f (-1) 0 →
  satisfies_condition f →
  (f (-3) = 0 ∧
   is_monotonically_increasing f 1 2 ∧
   is_symmetric_about_line f 1) :=
by
  intros h_even h_mono h_cond
  sorry

end NUMINAMATH_GPT_correct_statements_l872_87245


namespace NUMINAMATH_GPT_no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l872_87224

-- Conditions: Expressing the sum of three reciprocals
def sum_of_reciprocals (a b c : ℕ) : ℚ := (1 / a) + (1 / b) + (1 / c)

-- Proof Problem 1: Prove that the sum of the reciprocals of any three positive integers cannot equal 9/11
theorem no_three_reciprocals_sum_to_nine_eleven :
  ∀ (a b c : ℕ), sum_of_reciprocals a b c ≠ 9 / 11 := sorry

-- Proof Problem 2: Prove that there exists no rational number between 41/42 and 1 that can be expressed as the sum of the reciprocals of three positive integers other than 41/42
theorem no_rational_between_fortyone_fortytwo_and_one :
  ∀ (K : ℚ), 41 / 42 < K ∧ K < 1 → ¬ (∃ (a b c : ℕ), sum_of_reciprocals a b c = K) := sorry

end NUMINAMATH_GPT_no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l872_87224


namespace NUMINAMATH_GPT_find_larger_number_of_two_l872_87216

theorem find_larger_number_of_two (A B : ℕ) (hcf lcm : ℕ) (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 13)
  (h_factor2 : factor2 = 16)
  (h_lcm : lcm = hcf * factor1 * factor2)
  (h_A : A = hcf * m ∧ m = factor1)
  (h_B : B = hcf * n ∧ n = factor2):
  max A B = 368 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_of_two_l872_87216


namespace NUMINAMATH_GPT_line_does_not_pass_through_point_l872_87215

theorem line_does_not_pass_through_point 
  (m : ℝ) (h : (2*m + 1)^2 - 4*(m^2 + 4) > 0) : 
  ¬((2*m - 3)*(-2) - 4*m + 7 = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_point_l872_87215


namespace NUMINAMATH_GPT_num_of_poly_sci_majors_l872_87262

-- Define the total number of applicants
def total_applicants : ℕ := 40

-- Define the number of applicants with GPA > 3.0
def gpa_higher_than_3_point_0 : ℕ := 20

-- Define the number of applicants who did not major in political science and had GPA ≤ 3.0
def non_poly_sci_and_low_gpa : ℕ := 10

-- Define the number of political science majors with GPA > 3.0
def poly_sci_with_high_gpa : ℕ := 5

-- Prove the number of political science majors
theorem num_of_poly_sci_majors : ∀ (P : ℕ),
  P = poly_sci_with_high_gpa + 
      (total_applicants - non_poly_sci_and_low_gpa - 
       (gpa_higher_than_3_point_0 - poly_sci_with_high_gpa)) → 
  P = 20 :=
by
  intros P h
  sorry

end NUMINAMATH_GPT_num_of_poly_sci_majors_l872_87262


namespace NUMINAMATH_GPT_find_functions_l872_87223

theorem find_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_find_functions_l872_87223


namespace NUMINAMATH_GPT_tan_75_eq_2_plus_sqrt_3_l872_87208

theorem tan_75_eq_2_plus_sqrt_3 :
  Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_75_eq_2_plus_sqrt_3_l872_87208


namespace NUMINAMATH_GPT_total_number_of_workers_l872_87235

theorem total_number_of_workers (W : ℕ) (R : ℕ) 
  (h1 : (7 + R) * 8000 = 7 * 18000 + R * 6000) 
  (h2 : W = 7 + R) : W = 42 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_total_number_of_workers_l872_87235


namespace NUMINAMATH_GPT_borya_number_l872_87228

theorem borya_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) 
  (h3 : (n * 2 + 5) * 5 = 715) : n = 69 :=
sorry

end NUMINAMATH_GPT_borya_number_l872_87228


namespace NUMINAMATH_GPT_peaches_sold_to_friends_l872_87244

theorem peaches_sold_to_friends (x : ℕ) (total_peaches : ℕ) (peaches_to_relatives : ℕ) (peach_price_friend : ℕ) (peach_price_relative : ℝ) (total_earnings : ℝ) (peaches_left : ℕ) (total_peaches_sold : ℕ) 
  (h1 : total_peaches = 15) 
  (h2 : peaches_to_relatives = 4) 
  (h3 : peach_price_relative = 1.25) 
  (h4 : total_earnings = 25) 
  (h5 : peaches_left = 1)
  (h6 : total_peaches_sold = 14)
  (h7 : total_earnings = peach_price_friend * x + peach_price_relative * peaches_to_relatives)
  (h8 : total_peaches_sold = total_peaches - peaches_left) :
  x = 10 := 
sorry

end NUMINAMATH_GPT_peaches_sold_to_friends_l872_87244


namespace NUMINAMATH_GPT_fat_caterpillars_left_l872_87256

-- Define the initial and the newly hatched caterpillars
def initial_caterpillars : ℕ := 14
def hatched_caterpillars : ℕ := 4

-- Define the caterpillars left on the tree now
def current_caterpillars : ℕ := 10

-- Define the total caterpillars before any left
def total_caterpillars : ℕ := initial_caterpillars + hatched_caterpillars
-- Define the caterpillars leaving the tree
def caterpillars_left : ℕ := total_caterpillars - current_caterpillars

-- The theorem to be proven
theorem fat_caterpillars_left : caterpillars_left = 8 :=
by
  sorry

end NUMINAMATH_GPT_fat_caterpillars_left_l872_87256


namespace NUMINAMATH_GPT_rectangle_within_l872_87232

theorem rectangle_within (a b c d : ℝ) (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_within_l872_87232


namespace NUMINAMATH_GPT_total_social_media_hours_in_a_week_l872_87237

variable (daily_social_media_hours : ℕ) (days_in_week : ℕ)

theorem total_social_media_hours_in_a_week
(h1 : daily_social_media_hours = 3)
(h2 : days_in_week = 7) :
daily_social_media_hours * days_in_week = 21 := by
  sorry

end NUMINAMATH_GPT_total_social_media_hours_in_a_week_l872_87237


namespace NUMINAMATH_GPT_quadratic_roots_form_l872_87280

theorem quadratic_roots_form {a b c : ℤ} (h : a = 3 ∧ b = -7 ∧ c = 1) :
  ∃ (m n p : ℤ), (∀ x, 3*x^2 - 7*x + 1 = 0 ↔ x = (m + Real.sqrt n)/p ∨ x = (m - Real.sqrt n)/p)
  ∧ Int.gcd m (Int.gcd n p) = 1 ∧ n = 37 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_form_l872_87280


namespace NUMINAMATH_GPT_distinct_lines_through_point_and_parabola_l872_87211

noncomputable def num_distinct_lines : ℕ :=
  let num_divisors (n : ℕ) : ℕ :=
    have factors := [2^5, 3^2, 7]
    factors.foldl (fun acc f => acc * (f + 1)) 1
  (num_divisors 2016) / 2 -- as each pair (x_1, x_2) corresponds twice

theorem distinct_lines_through_point_and_parabola :
  num_distinct_lines = 36 :=
by
  sorry

end NUMINAMATH_GPT_distinct_lines_through_point_and_parabola_l872_87211


namespace NUMINAMATH_GPT_disqualified_team_participants_l872_87210

theorem disqualified_team_participants
  (initial_teams : ℕ) (initial_avg : ℕ) (final_teams : ℕ) (final_avg : ℕ)
  (total_initial : ℕ) (total_final : ℕ) :
  initial_teams = 9 →
  initial_avg = 7 →
  final_teams = 8 →
  final_avg = 6 →
  total_initial = initial_teams * initial_avg →
  total_final = final_teams * final_avg →
  total_initial - total_final = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_disqualified_team_participants_l872_87210


namespace NUMINAMATH_GPT_marbles_per_customer_l872_87214

theorem marbles_per_customer
  (initial_marbles remaining_marbles customers marbles_per_customer : ℕ)
  (h1 : initial_marbles = 400)
  (h2 : remaining_marbles = 100)
  (h3 : customers = 20)
  (h4 : initial_marbles - remaining_marbles = customers * marbles_per_customer) :
  marbles_per_customer = 15 :=
by
  sorry

end NUMINAMATH_GPT_marbles_per_customer_l872_87214


namespace NUMINAMATH_GPT_sara_lunch_total_l872_87298

theorem sara_lunch_total :
  let hotdog := 5.36
  let salad := 5.10
  hotdog + salad = 10.46 :=
by
  let hotdog := 5.36
  let salad := 5.10
  sorry

end NUMINAMATH_GPT_sara_lunch_total_l872_87298


namespace NUMINAMATH_GPT_nine_odot_three_l872_87291

-- Defining the operation based on the given conditions
axiom odot_def (a b : ℕ) : ℕ

axiom odot_eq_1 : odot_def 2 4 = 8
axiom odot_eq_2 : odot_def 4 6 = 14
axiom odot_eq_3 : odot_def 5 3 = 13
axiom odot_eq_4 : odot_def 8 7 = 23

-- Proving that 9 ⊙ 3 = 21
theorem nine_odot_three : odot_def 9 3 = 21 := 
by
  sorry

end NUMINAMATH_GPT_nine_odot_three_l872_87291


namespace NUMINAMATH_GPT_min_triangle_perimeter_l872_87293

/-- Given a point (a, b) with 0 < b < a,
    determine the minimum perimeter of a triangle with one vertex at (a, b),
    one on the x-axis, and one on the line y = x. 
    The minimum perimeter is √(2(a^2 + b^2)).
-/
theorem min_triangle_perimeter (a b : ℝ) (h : 0 < b ∧ b < a) 
  : ∃ c d : ℝ, c^2 + d^2 = 2 * (a^2 + b^2) := sorry

end NUMINAMATH_GPT_min_triangle_perimeter_l872_87293


namespace NUMINAMATH_GPT_sandy_money_l872_87205

theorem sandy_money (X : ℝ) (h1 : 0.70 * X = 224) : X = 320 := 
by {
  sorry
}

end NUMINAMATH_GPT_sandy_money_l872_87205


namespace NUMINAMATH_GPT_sheep_transaction_gain_l872_87264

noncomputable def percent_gain (cost_per_sheep total_sheep sold_sheep remaining_sheep : ℕ) : ℚ :=
let total_cost := (cost_per_sheep : ℚ) * total_sheep
let initial_revenue := total_cost
let price_per_sheep := initial_revenue / sold_sheep
let remaining_revenue := remaining_sheep * price_per_sheep
let total_revenue := initial_revenue + remaining_revenue
let profit := total_revenue - total_cost
(profit / total_cost) * 100

theorem sheep_transaction_gain :
  percent_gain 1 1000 950 50 = -47.37 := sorry

end NUMINAMATH_GPT_sheep_transaction_gain_l872_87264


namespace NUMINAMATH_GPT_initial_amount_calc_l872_87236

theorem initial_amount_calc 
  (M : ℝ)
  (H1 : M * 0.3675 = 350) :
  M = 952.38 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_calc_l872_87236


namespace NUMINAMATH_GPT_juice_oranges_l872_87243

theorem juice_oranges (oranges_per_glass : ℕ) (glasses : ℕ) (total_oranges : ℕ)
  (h1 : oranges_per_glass = 3)
  (h2 : glasses = 10)
  (h3 : total_oranges = oranges_per_glass * glasses) :
  total_oranges = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_juice_oranges_l872_87243


namespace NUMINAMATH_GPT_find_n_l872_87226

-- Definitions based on conditions
variable (n : ℕ)  -- number of persons
variable (A : Fin n → Finset (Fin n))  -- acquaintance relation, specified as a set of neighbors for each person
-- Condition 1: Each person is acquainted with exactly 8 others
def acquaintances := ∀ i : Fin n, (A i).card = 8
-- Condition 2: Any two acquainted persons have exactly 4 common acquaintances
def common_acquaintances_adj := ∀ i j : Fin n, i ≠ j → j ∈ (A i) → (A i ∩ A j).card = 4
-- Condition 3: Any two non-acquainted persons have exactly 2 common acquaintances
def common_acquaintances_non_adj := ∀ i j : Fin n, i ≠ j → j ∉ (A i) → (A i ∩ A j).card = 2

-- Statement to prove
theorem find_n (h1 : acquaintances n A) (h2 : common_acquaintances_adj n A) (h3 : common_acquaintances_non_adj n A) :
  n = 21 := 
sorry

end NUMINAMATH_GPT_find_n_l872_87226


namespace NUMINAMATH_GPT_rachel_total_clothing_l872_87234

def box_1_scarves : ℕ := 2
def box_1_mittens : ℕ := 3
def box_1_hats : ℕ := 1
def box_2_scarves : ℕ := 4
def box_2_mittens : ℕ := 2
def box_2_hats : ℕ := 2
def box_3_scarves : ℕ := 1
def box_3_mittens : ℕ := 5
def box_3_hats : ℕ := 3
def box_4_scarves : ℕ := 3
def box_4_mittens : ℕ := 4
def box_4_hats : ℕ := 1
def box_5_scarves : ℕ := 5
def box_5_mittens : ℕ := 3
def box_5_hats : ℕ := 2
def box_6_scarves : ℕ := 2
def box_6_mittens : ℕ := 6
def box_6_hats : ℕ := 0
def box_7_scarves : ℕ := 4
def box_7_mittens : ℕ := 1
def box_7_hats : ℕ := 3
def box_8_scarves : ℕ := 3
def box_8_mittens : ℕ := 2
def box_8_hats : ℕ := 4
def box_9_scarves : ℕ := 1
def box_9_mittens : ℕ := 4
def box_9_hats : ℕ := 5

def total_clothing : ℕ := 
  box_1_scarves + box_1_mittens + box_1_hats +
  box_2_scarves + box_2_mittens + box_2_hats +
  box_3_scarves + box_3_mittens + box_3_hats +
  box_4_scarves + box_4_mittens + box_4_hats +
  box_5_scarves + box_5_mittens + box_5_hats +
  box_6_scarves + box_6_mittens + box_6_hats +
  box_7_scarves + box_7_mittens + box_7_hats +
  box_8_scarves + box_8_mittens + box_8_hats +
  box_9_scarves + box_9_mittens + box_9_hats

theorem rachel_total_clothing : total_clothing = 76 :=
by
  sorry

end NUMINAMATH_GPT_rachel_total_clothing_l872_87234


namespace NUMINAMATH_GPT_linear_relationship_increase_in_y_l872_87275

theorem linear_relationship_increase_in_y (x y : ℝ) (hx : x = 12) (hy : y = 10 / 4 * x) : y = 30 := by
  sorry

end NUMINAMATH_GPT_linear_relationship_increase_in_y_l872_87275


namespace NUMINAMATH_GPT_car_rent_per_day_leq_30_l872_87206

variable (D : ℝ) -- daily rental rate
variable (cost_per_mile : ℝ := 0.23) -- cost per mile
variable (daily_budget : ℝ := 76) -- daily budget
variable (distance : ℝ := 200) -- distance driven

theorem car_rent_per_day_leq_30 :
  D + cost_per_mile * distance ≤ daily_budget → D ≤ 30 :=
sorry

end NUMINAMATH_GPT_car_rent_per_day_leq_30_l872_87206


namespace NUMINAMATH_GPT_simplify_expression_l872_87249
open Real

theorem simplify_expression (x y : ℝ) : -x + y - 2 * x - 3 * y = -3 * x - 2 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l872_87249


namespace NUMINAMATH_GPT_part1_part2_l872_87218

noncomputable def setA : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }
noncomputable def setB (m : ℝ) : Set ℝ := { x | m - 1 < x ∧ x < 2*m + 1 }

theorem part1 (x : ℝ) : 
  setA ∪ setB 3 = { x | -1 ≤ x ∧ x < 7 } :=
sorry

theorem part2 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∃ x, x ∈ setB m ∧ x ∉ setA) ↔ 
  m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l872_87218


namespace NUMINAMATH_GPT_gcd_8Tn_nplus1_eq_4_l872_87272

noncomputable def T_n (n : ℕ) : ℕ :=
(n * (n + 1)) / 2

theorem gcd_8Tn_nplus1_eq_4 (n : ℕ) (hn: 0 < n) : gcd (8 * T_n n) (n + 1) = 4 :=
sorry

end NUMINAMATH_GPT_gcd_8Tn_nplus1_eq_4_l872_87272


namespace NUMINAMATH_GPT_domain_sqrt_frac_l872_87250

theorem domain_sqrt_frac (x : ℝ) :
  (x^2 + 4*x + 3 ≠ 0) ∧ (x + 3 ≥ 0) ↔ ((x ∈ Set.Ioc (-3) (-1)) ∨ (x ∈ Set.Ioi (-1))) :=
by
  sorry

end NUMINAMATH_GPT_domain_sqrt_frac_l872_87250
