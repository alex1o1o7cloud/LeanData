import Mathlib

namespace NUMINAMATH_GPT_perfect_square_n_l82_8210

theorem perfect_square_n (n : ℕ) (hn_pos : n > 0) :
  (∃ (m : ℕ), m * m = (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2 :=
by sorry

end NUMINAMATH_GPT_perfect_square_n_l82_8210


namespace NUMINAMATH_GPT_total_children_with_cats_l82_8272

variable (D C B : ℕ)
variable (h1 : D = 18)
variable (h2 : B = 6)
variable (h3 : D + C + B = 30)

theorem total_children_with_cats : C + B = 12 := by
  sorry

end NUMINAMATH_GPT_total_children_with_cats_l82_8272


namespace NUMINAMATH_GPT_cos_double_angle_zero_l82_8264

theorem cos_double_angle_zero
  (θ : ℝ)
  (a : ℝ×ℝ := (1, -Real.cos θ))
  (b : ℝ×ℝ := (1, 2 * Real.cos θ))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.cos (2 * θ) = 0 :=
by sorry

end NUMINAMATH_GPT_cos_double_angle_zero_l82_8264


namespace NUMINAMATH_GPT_star_addition_l82_8236

-- Definition of the binary operation "star"
def star (x y : ℤ) := 5 * x - 2 * y

-- Statement of the problem
theorem star_addition : star 3 4 + star 2 2 = 13 :=
by
  -- By calculation, we have:
  -- star 3 4 = 7 and star 2 2 = 6
  -- Thus, star 3 4 + star 2 2 = 7 + 6 = 13
  sorry

end NUMINAMATH_GPT_star_addition_l82_8236


namespace NUMINAMATH_GPT_five_power_l82_8256

theorem five_power (a : ℕ) (h : 5^a = 3125) : 5^(a - 3) = 25 := 
  sorry

end NUMINAMATH_GPT_five_power_l82_8256


namespace NUMINAMATH_GPT_smallest_number_gt_sum_digits_1755_l82_8252

theorem smallest_number_gt_sum_digits_1755 :
  ∃ (n : ℕ) (a b c d : ℕ), a ≠ 0 ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ n = (a + b + c + d) + 1755 ∧ n = 1770 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_gt_sum_digits_1755_l82_8252


namespace NUMINAMATH_GPT_min_games_needed_l82_8225

theorem min_games_needed (N : ℕ) : 
  (2 + N) * 10 ≥ 9 * (5 + N) ↔ N ≥ 25 := 
by {
  sorry
}

end NUMINAMATH_GPT_min_games_needed_l82_8225


namespace NUMINAMATH_GPT_subtract_value_is_34_l82_8211

theorem subtract_value_is_34 
    (x y : ℤ) 
    (h1 : (x - 5) / 7 = 7) 
    (h2 : (x - y) / 10 = 2) : 
    y = 34 := 
sorry

end NUMINAMATH_GPT_subtract_value_is_34_l82_8211


namespace NUMINAMATH_GPT_BillCookingTime_l82_8204

-- Definitions corresponding to the conditions
def chopTimePepper : Nat := 3  -- minutes to chop one pepper
def chopTimeOnion : Nat := 4   -- minutes to chop one onion
def grateTimeCheese : Nat := 1 -- minutes to grate cheese for one omelet
def cookTimeOmelet : Nat := 5  -- minutes to assemble and cook one omelet

def numberOfPeppers : Nat := 4  -- number of peppers Bill needs to chop
def numberOfOnions : Nat := 2   -- number of onions Bill needs to chop
def numberOfOmelets : Nat := 5  -- number of omelets Bill prepares

-- Calculations based on conditions
def totalChopTimePepper : Nat := numberOfPeppers * chopTimePepper
def totalChopTimeOnion : Nat := numberOfOnions * chopTimeOnion
def totalGrateTimeCheese : Nat := numberOfOmelets * grateTimeCheese
def totalCookTimeOmelet : Nat := numberOfOmelets * cookTimeOmelet

-- Total preparation and cooking time
def totalTime : Nat := totalChopTimePepper + totalChopTimeOnion + totalGrateTimeCheese + totalCookTimeOmelet

-- Theorem statement
theorem BillCookingTime :
  totalTime = 50 := by
  sorry

end NUMINAMATH_GPT_BillCookingTime_l82_8204


namespace NUMINAMATH_GPT_sum_of_squares_bounds_l82_8275

-- Given quadrilateral vertices' distances from the nearest vertices of the square
variable (w x y z : ℝ)
-- The side length of the square
def side_length_square : ℝ := 1

-- Expression for the square of each side of the quadrilateral
def square_AB : ℝ := w^2 + x^2
def square_BC : ℝ := (side_length_square - x)^2 + y^2
def square_CD : ℝ := (side_length_square - y)^2 + z^2
def square_DA : ℝ := (side_length_square - z)^2 + (side_length_square - w)^2

-- Sum of the squares of the sides
def sum_of_squares := square_AB w x + square_BC x y + square_CD y z + square_DA z w

-- Proof that the sum of the squares is within the bounds [2, 4]
theorem sum_of_squares_bounds (hw : 0 ≤ w ∧ w ≤ side_length_square)
                              (hx : 0 ≤ x ∧ x ≤ side_length_square)
                              (hy : 0 ≤ y ∧ y ≤ side_length_square)
                              (hz : 0 ≤ z ∧ z ≤ side_length_square)
                              : 2 ≤ sum_of_squares w x y z ∧ sum_of_squares w x y z ≤ 4 := sorry

end NUMINAMATH_GPT_sum_of_squares_bounds_l82_8275


namespace NUMINAMATH_GPT_domain_of_function_l82_8232

theorem domain_of_function :
  ∀ x : ℝ, (x - 1 ≥ 0) ↔ (x ≥ 1) ∧ (x + 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l82_8232


namespace NUMINAMATH_GPT_savings_equal_in_820_weeks_l82_8261

-- Definitions for the conditions
def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

-- The statement we want to prove
theorem savings_equal_in_820_weeks : 
  ∃ (w : ℕ), (sara_initial_savings + w * sara_weekly_savings) = (w * jim_weekly_savings) ∧ w = 820 :=
by
  sorry

end NUMINAMATH_GPT_savings_equal_in_820_weeks_l82_8261


namespace NUMINAMATH_GPT_kevin_leap_day_2024_is_monday_l82_8224

def days_between_leap_birthdays (years: ℕ) (leap_year_count: ℕ) : ℕ :=
  (years - leap_year_count) * 365 + leap_year_count * 366

def day_of_week_after_days (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

noncomputable def kevin_leap_day_weekday_2024 : ℕ :=
  let days := days_between_leap_birthdays 24 6
  let start_day := 2 -- Tuesday as 2 (assuming 0 = Sunday, 1 = Monday,..., 6 = Saturday)
  day_of_week_after_days start_day days

theorem kevin_leap_day_2024_is_monday :
  kevin_leap_day_weekday_2024 = 1 -- 1 represents Monday
  :=
by
  sorry

end NUMINAMATH_GPT_kevin_leap_day_2024_is_monday_l82_8224


namespace NUMINAMATH_GPT_ratio_length_to_width_l82_8299

def garden_length := 80
def garden_perimeter := 240

theorem ratio_length_to_width : ∃ W, 2 * garden_length + 2 * W = garden_perimeter ∧ garden_length / W = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_length_to_width_l82_8299


namespace NUMINAMATH_GPT_find_other_package_size_l82_8283

variable (total_coffee : ℕ)
variable (total_5_ounce_packages : ℕ)
variable (num_other_packages : ℕ)
variable (other_package_size : ℕ)

theorem find_other_package_size
  (h1 : total_coffee = 85)
  (h2 : total_5_ounce_packages = num_other_packages + 2)
  (h3 : num_other_packages = 5)
  (h4 : 5 * total_5_ounce_packages + other_package_size * num_other_packages = total_coffee) :
  other_package_size = 10 :=
sorry

end NUMINAMATH_GPT_find_other_package_size_l82_8283


namespace NUMINAMATH_GPT_find_p_l82_8278

theorem find_p (n : ℝ) (p : ℝ) (h1 : p = 4 * n * (1 / (2 ^ 2009)) ^ Real.log 1) (h2 : n = 9 / 4) : p = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l82_8278


namespace NUMINAMATH_GPT_fraction_expression_l82_8244

theorem fraction_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_expression_l82_8244


namespace NUMINAMATH_GPT_rhombus_area_correct_l82_8243

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 30 12 = 180 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_correct_l82_8243


namespace NUMINAMATH_GPT_perfect_square_and_solutions_exist_l82_8294

theorem perfect_square_and_solutions_exist (m n t : ℕ)
  (h1 : t > 0) (h2 : m > 0) (h3 : n > 0)
  (h4 : t * (m^2 - n^2) + m - n^2 - n = 0) :
  ∃ (k : ℕ), m - n = k * k ∧ (∀ t > 0, ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (t * (m^2 - n^2) + m - n^2 - n = 0)) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_and_solutions_exist_l82_8294


namespace NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l82_8226

-- Conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0
def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Proof target
theorem p_is_necessary_but_not_sufficient_for_q : 
  (∀ a : ℝ, p a → q a) ∧ ¬(∀ a : ℝ, q a → p a) :=
sorry

end NUMINAMATH_GPT_p_is_necessary_but_not_sufficient_for_q_l82_8226


namespace NUMINAMATH_GPT_distinct_solutions_subtraction_l82_8277

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_solutions_subtraction_l82_8277


namespace NUMINAMATH_GPT_cubic_has_real_root_l82_8265

open Real

-- Define the conditions
variables (a0 a1 a2 a3 : ℝ) (h : a0 ≠ 0)

-- Define the cubic polynomial function
def cubic (x : ℝ) : ℝ :=
  a0 * x^3 + a1 * x^2 + a2 * x + a3

-- State the theorem
theorem cubic_has_real_root : ∃ x : ℝ, cubic a0 a1 a2 a3 x = 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_has_real_root_l82_8265


namespace NUMINAMATH_GPT_items_purchased_total_profit_l82_8296

-- Definitions based on conditions given in part (a)
def total_cost := 6000
def cost_A := 22
def cost_B := 30
def sell_A := 29
def sell_B := 40

-- Proven answers from the solution (part (b))
def items_A := 150
def items_B := 90
def profit := 1950

-- Lean theorem statements (problems to be proved)
theorem items_purchased : (22 * items_A + 30 * (items_A / 2 + 15) = total_cost) → 
                          (items_A = 150) ∧ (items_B = 90) := sorry

theorem total_profit : (items_A = 150) → (items_B = 90) → 
                       ((items_A * (sell_A - cost_A) + items_B * (sell_B - cost_B)) = profit) := sorry

end NUMINAMATH_GPT_items_purchased_total_profit_l82_8296


namespace NUMINAMATH_GPT_triangle_equilateral_from_midpoint_circles_l82_8284

theorem triangle_equilateral_from_midpoint_circles (a b c : ℝ)
  (h1 : ∃ E F G : ℝ → ℝ, ∀ x, (|E x| = a/4 ∨ |F x| = b/4 ∨ |G x| = c/4))
  (h2 : (|a/2| ≤ a/4 + b/4) ∧ (|b/2| ≤ b/4 + c/4) ∧ (|c/2| ≤ c/4 + a/4)) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_from_midpoint_circles_l82_8284


namespace NUMINAMATH_GPT_clara_current_age_l82_8242

theorem clara_current_age (a c : ℕ) (h1 : a = 54) (h2 : (c - 41) = 3 * (a - 41)) : c = 80 :=
by
  -- This is where the proof would be constructed.
  sorry

end NUMINAMATH_GPT_clara_current_age_l82_8242


namespace NUMINAMATH_GPT_rectangle_k_value_l82_8251

theorem rectangle_k_value (x d : ℝ)
  (h_ratio : ∃ x, ∀ l w, l = 5 * x ∧ w = 4 * x)
  (h_diagonal : ∀ l w, l = 5 * x ∧ w = 4 * x → d^2 = (5 * x)^2 + (4 * x)^2)
  (h_area_written : ∃ k, ∀ A, A = (5 * x) * (4 * x) → A = k * d^2) :
  ∃ k, k = 20 / 41 := sorry

end NUMINAMATH_GPT_rectangle_k_value_l82_8251


namespace NUMINAMATH_GPT_thelma_tomato_count_l82_8246

-- Definitions and conditions
def slices_per_tomato : ℕ := 8
def slices_per_meal_per_person : ℕ := 20
def family_members : ℕ := 8
def total_slices_needed : ℕ := slices_per_meal_per_person * family_members
def tomatoes_needed : ℕ := total_slices_needed / slices_per_tomato

-- Statement of the theorem to be proved
theorem thelma_tomato_count :
  tomatoes_needed = 20 := by
  sorry

end NUMINAMATH_GPT_thelma_tomato_count_l82_8246


namespace NUMINAMATH_GPT_find_sum_l82_8202

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end NUMINAMATH_GPT_find_sum_l82_8202


namespace NUMINAMATH_GPT_total_red_stripes_l82_8292

theorem total_red_stripes 
  (flagA_stripes : ℕ := 30) 
  (flagB_stripes : ℕ := 45) 
  (flagC_stripes : ℕ := 60)
  (flagA_count : ℕ := 20) 
  (flagB_count : ℕ := 30) 
  (flagC_count : ℕ := 40)
  (flagA_red : ℕ := 15)
  (flagB_red : ℕ := 15)
  (flagC_red : ℕ := 14) : 
  300 + 450 + 560 = 1310 := 
by
  have flagA_red_stripes : 15 = 15 := by rfl
  have flagB_red_stripes : 15 = 15 := by rfl
  have flagC_red_stripes : 14 = 14 := by rfl
  have total_A_red_stripes : 15 * 20 = 300 := by norm_num
  have total_B_red_stripes : 15 * 30 = 450 := by norm_num
  have total_C_red_stripes : 14 * 40 = 560 := by norm_num
  exact add_assoc 300 450 560 ▸ rfl

end NUMINAMATH_GPT_total_red_stripes_l82_8292


namespace NUMINAMATH_GPT_jay_change_l82_8240

def cost_book : ℝ := 25
def cost_pen : ℝ := 4
def cost_ruler : ℝ := 1
def payment : ℝ := 50

theorem jay_change : (payment - (cost_book + cost_pen + cost_ruler) = 20) := sorry

end NUMINAMATH_GPT_jay_change_l82_8240


namespace NUMINAMATH_GPT_points_on_line_l82_8219

theorem points_on_line : 
    ∀ (P : ℝ × ℝ),
      (P = (1, 2) ∨ P = (0, 0) ∨ P = (2, 4) ∨ P = (5, 10) ∨ P = (-1, -2))
      → (∃ m b, m = 2 ∧ b = 0 ∧ P.2 = m * P.1 + b) :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l82_8219


namespace NUMINAMATH_GPT_time_for_10_strikes_l82_8238

-- Assume a clock takes 7 seconds to strike 7 times
def clock_time_for_N_strikes (N : ℕ) : ℕ :=
  if N = 7 then 7 else sorry  -- This would usually be a function, simplified here for the specific condition

-- Assume there are 6 intervals for 7 strikes
def intervals_between_strikes (N : ℕ) : ℕ :=
  if N = 7 then 6 else N - 1

-- Function to calculate total time for any number of strikes based on intervals and time per strike
def total_time_for_strikes (N : ℕ) : ℚ :=
  (intervals_between_strikes N) * (clock_time_for_N_strikes 7 / intervals_between_strikes 7 : ℚ)

theorem time_for_10_strikes : total_time_for_strikes 10 = 10.5 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_time_for_10_strikes_l82_8238


namespace NUMINAMATH_GPT_find_polynomial_q_l82_8231

theorem find_polynomial_q (q : ℝ → ℝ) :
  (∀ x : ℝ, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x : ℝ, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by 
  sorry

end NUMINAMATH_GPT_find_polynomial_q_l82_8231


namespace NUMINAMATH_GPT_real_estate_profit_l82_8237

def purchase_price_first : ℝ := 350000
def purchase_price_second : ℝ := 450000
def purchase_price_third : ℝ := 600000

def gain_first : ℝ := 0.12
def loss_second : ℝ := 0.08
def gain_third : ℝ := 0.18

def selling_price_first : ℝ :=
  purchase_price_first + (purchase_price_first * gain_first)
def selling_price_second : ℝ :=
  purchase_price_second - (purchase_price_second * loss_second)
def selling_price_third : ℝ :=
  purchase_price_third + (purchase_price_third * gain_third)

def total_purchase_price : ℝ :=
  purchase_price_first + purchase_price_second + purchase_price_third
def total_selling_price : ℝ :=
  selling_price_first + selling_price_second + selling_price_third

def overall_gain : ℝ :=
  total_selling_price - total_purchase_price

theorem real_estate_profit :
  overall_gain = 114000 := by
  sorry

end NUMINAMATH_GPT_real_estate_profit_l82_8237


namespace NUMINAMATH_GPT_rectangle_perimeter_l82_8270

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  2 * (a + b)

theorem rectangle_perimeter (p q: ℕ) (rel_prime: Nat.gcd p q = 1) :
  ∃ (a b c: ℕ), p = 2 * (a + b) ∧ p + q = 52 ∧ a = 5 ∧ b = 12 ∧ c = 7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l82_8270


namespace NUMINAMATH_GPT_inequality_of_transformed_division_l82_8268

theorem inequality_of_transformed_division (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (h : A * 5 = B * 4) : A ≤ B := by
  sorry

end NUMINAMATH_GPT_inequality_of_transformed_division_l82_8268


namespace NUMINAMATH_GPT_largest_m_for_game_with_2022_grids_l82_8227

variables (n : ℕ) (f : ℕ → ℕ)

/- Definitions using conditions given -/

/-- Definition of the game and the marking process -/
def game (n : ℕ) : ℕ := 
  if n % 4 = 0 then n / 2 + 1
  else if n % 4 = 2 then n / 2 + 1
  else 0

/-- Main theorem statement -/
theorem largest_m_for_game_with_2022_grids : game 2022 = 1011 :=
by sorry

end NUMINAMATH_GPT_largest_m_for_game_with_2022_grids_l82_8227


namespace NUMINAMATH_GPT_ordered_pairs_count_l82_8215

theorem ordered_pairs_count : 
  (∀ (b c : ℕ), b > 0 ∧ b ≤ 6 ∧ c > 0 ∧ c ≤ 6 ∧ b^2 - 4 * c < 0 ∧ c^2 - 4 * b < 0 → 
  ((b = 1 ∧ (c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 2 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 3 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 4 ∧ (c = 5 ∨ c = 6)))) ∧
  (∃ (n : ℕ), n = 15) := sorry

end NUMINAMATH_GPT_ordered_pairs_count_l82_8215


namespace NUMINAMATH_GPT_sum_of_squares_of_biking_jogging_swimming_rates_l82_8285

theorem sum_of_squares_of_biking_jogging_swimming_rates (b j s : ℕ) 
  (h1 : 2 * b + 3 * j + 4 * s = 74) 
  (h2 : 4 * b + 2 * j + 3 * s = 91) : 
  (b^2 + j^2 + s^2 = 314) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_biking_jogging_swimming_rates_l82_8285


namespace NUMINAMATH_GPT_initially_tagged_fish_l82_8276

theorem initially_tagged_fish (second_catch_total : ℕ) (second_catch_tagged : ℕ)
  (total_fish_pond : ℕ) (approx_ratio : ℚ) 
  (h1 : second_catch_total = 50)
  (h2 : second_catch_tagged = 2)
  (h3 : total_fish_pond = 1750)
  (h4 : approx_ratio = (second_catch_tagged : ℚ) / second_catch_total) :
  ∃ T : ℕ, T = 70 :=
by
  sorry

end NUMINAMATH_GPT_initially_tagged_fish_l82_8276


namespace NUMINAMATH_GPT_fraction_product_l82_8286

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_l82_8286


namespace NUMINAMATH_GPT_initial_scissors_l82_8293

-- Define conditions as per the problem
def Keith_placed (added : ℕ) : Prop := added = 22
def total_now (total : ℕ) : Prop := total = 76

-- Define the problem statement as a theorem
theorem initial_scissors (added total initial : ℕ) (h1 : Keith_placed added) (h2 : total_now total) 
  (h3 : total = initial + added) : initial = 54 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_initial_scissors_l82_8293


namespace NUMINAMATH_GPT_ratio_AH_HD_triangle_l82_8279

theorem ratio_AH_HD_triangle (BC AC : ℝ) (angleC : ℝ) (H AD HD : ℝ) 
  (hBC : BC = 4) (hAC : AC = 3 * Real.sqrt 2) (hAngleC : angleC = 45) 
  (hAD : AD = 3) (hHD : HD = 1) : 
  (AH / HD) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_AH_HD_triangle_l82_8279


namespace NUMINAMATH_GPT_chess_tournament_l82_8280

theorem chess_tournament (m p k n : ℕ) 
  (h1 : m * 9 = p * 6) 
  (h2 : m * n = k * 8) 
  (h3 : p * 2 = k * 6) : 
  n = 4 := 
by 
  sorry

end NUMINAMATH_GPT_chess_tournament_l82_8280


namespace NUMINAMATH_GPT_non_neg_sequence_l82_8221

theorem non_neg_sequence (a : ℝ) (x : ℕ → ℝ) (h0 : x 0 = 0)
  (h1 : ∀ n, x (n + 1) = 1 - a * Real.exp (x n)) (ha : a ≤ 1) :
  ∀ n, x n ≥ 0 := 
  sorry

end NUMINAMATH_GPT_non_neg_sequence_l82_8221


namespace NUMINAMATH_GPT_invitees_count_l82_8259

theorem invitees_count 
  (packages : ℕ) 
  (weight_per_package : ℕ) 
  (weight_per_burger : ℕ) 
  (total_people : ℕ)
  (H1 : packages = 4)
  (H2 : weight_per_package = 5)
  (H3 : weight_per_burger = 2)
  (H4 : total_people + 1 = (packages * weight_per_package) / weight_per_burger) :
  total_people = 9 := 
by
  sorry

end NUMINAMATH_GPT_invitees_count_l82_8259


namespace NUMINAMATH_GPT_transform_to_zero_set_l82_8269

def S (p : ℕ) : Finset ℕ := Finset.range p

def P (p : ℕ) (x : ℕ) : ℕ := 3 * x ^ ((2 * p - 1) / 3) + x ^ ((p + 1) / 3) + x + 1

def remainder (n p : ℕ) : ℕ := n % p

theorem transform_to_zero_set (p k : ℕ) (hp : Nat.Prime p) (h_cong : p % 3 = 2) (hk : 0 < k) :
  (∃ n : ℕ, ∀ i ∈ S p, remainder (P p i) p = n) ∨ (∃ n : ℕ, ∀ i ∈ S p, remainder (i ^ k) p = n) ↔
  Nat.gcd k (p - 1) > 1 :=
sorry

end NUMINAMATH_GPT_transform_to_zero_set_l82_8269


namespace NUMINAMATH_GPT_machines_finish_job_in_24_over_11_hours_l82_8288

theorem machines_finish_job_in_24_over_11_hours :
    let work_rate_A := 1 / 4
    let work_rate_B := 1 / 12
    let work_rate_C := 1 / 8
    let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
    (1 : ℝ) / combined_work_rate = 24 / 11 :=
by
  sorry

end NUMINAMATH_GPT_machines_finish_job_in_24_over_11_hours_l82_8288


namespace NUMINAMATH_GPT_cannot_finish_third_l82_8262

-- Definitions for the orders of runners
def order (a b : String) : Prop := a < b

-- The problem statement and conditions
def conditions (P Q R S T U : String) : Prop :=
  order P Q ∧ order P R ∧ order Q S ∧ order P U ∧ order U T ∧ order T Q

theorem cannot_finish_third (P Q R S T U : String) (h : conditions P Q R S T U) :
  (P = "third" → False) ∧ (S = "third" → False) :=
by
  sorry

end NUMINAMATH_GPT_cannot_finish_third_l82_8262


namespace NUMINAMATH_GPT_speedster_convertibles_approx_l82_8245

-- Definitions corresponding to conditions
def total_inventory : ℕ := 120
def num_non_speedsters : ℕ := 40
def num_speedsters : ℕ := 2 * total_inventory / 3
def num_speedster_convertibles : ℕ := 64

-- Theorem statement
theorem speedster_convertibles_approx :
  2 * total_inventory / 3 - num_non_speedsters + num_speedster_convertibles = total_inventory :=
sorry

end NUMINAMATH_GPT_speedster_convertibles_approx_l82_8245


namespace NUMINAMATH_GPT_required_fencing_l82_8222

-- Given definitions and conditions
def area (L W : ℕ) : ℕ := L * W

def fencing (W L : ℕ) : ℕ := 2 * W + L

theorem required_fencing
  (L W : ℕ)
  (hL : L = 10)
  (hA : area L W = 600) :
  fencing W L = 130 := by
  sorry

end NUMINAMATH_GPT_required_fencing_l82_8222


namespace NUMINAMATH_GPT_sqrt_400_div_2_l82_8273

theorem sqrt_400_div_2 : (Nat.sqrt 400) / 2 = 10 := by
  sorry

end NUMINAMATH_GPT_sqrt_400_div_2_l82_8273


namespace NUMINAMATH_GPT_two_digit_numbers_condition_l82_8214

theorem two_digit_numbers_condition (a b : ℕ) (h1 : a ≠ 0) (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : 0 ≤ b ∧ b ≤ 9) :
  (a + 1) * (b + 1) = 10 * a + b + 1 ↔ b = 9 := 
sorry

end NUMINAMATH_GPT_two_digit_numbers_condition_l82_8214


namespace NUMINAMATH_GPT_trailing_zeros_30_factorial_l82_8274

-- Definitions directly from conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def trailing_zeros (n : ℕ) : ℕ :=
  let count_five_factors (k : ℕ) : ℕ :=
    k / 5 + k / 25 + k / 125 -- This generalizes for higher powers of 5 which is sufficient here.
  count_five_factors n

-- Mathematical proof problem statement
theorem trailing_zeros_30_factorial : trailing_zeros 30 = 7 := by
  sorry

end NUMINAMATH_GPT_trailing_zeros_30_factorial_l82_8274


namespace NUMINAMATH_GPT_jovana_bucket_shells_l82_8208

theorem jovana_bucket_shells :
  let a0 := 5.2
  let a1 := a0 + 15.7
  let a2 := a1 + 17.5
  let a3 := a2 - 4.3
  let a4 := 3 * a3
  a4 = 102.3 := 
by
  sorry

end NUMINAMATH_GPT_jovana_bucket_shells_l82_8208


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l82_8205

theorem sufficient_condition_for_inequality (m : ℝ) : (m ≥ 2) → (∀ x : ℝ, x^2 - 2 * x + m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l82_8205


namespace NUMINAMATH_GPT_smaller_of_two_digit_numbers_l82_8230

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end NUMINAMATH_GPT_smaller_of_two_digit_numbers_l82_8230


namespace NUMINAMATH_GPT_dealer_overall_gain_l82_8223

noncomputable def dealer_gain_percentage (weight1 weight2 : ℕ) (cost_price : ℕ) : ℚ :=
  let actual_weight_sold := weight1 + weight2
  let supposed_weight_sold := 1000 + 1000
  let gain_item1 := cost_price - (weight1 / 1000) * cost_price
  let gain_item2 := cost_price - (weight2 / 1000) * cost_price
  let total_gain := gain_item1 + gain_item2
  let total_actual_cost := (actual_weight_sold / 1000) * cost_price
  (total_gain / total_actual_cost) * 100

theorem dealer_overall_gain :
  dealer_gain_percentage 900 850 100 = 14.29 := 
sorry

end NUMINAMATH_GPT_dealer_overall_gain_l82_8223


namespace NUMINAMATH_GPT_final_value_after_determinant_and_addition_l82_8254

theorem final_value_after_determinant_and_addition :
  let a := 5
  let b := 7
  let c := 3
  let d := 4
  let det := a * d - b * c
  det + 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_final_value_after_determinant_and_addition_l82_8254


namespace NUMINAMATH_GPT_total_drink_volume_l82_8249

theorem total_drink_volume (oj wj gj : ℕ) (hoj : oj = 25) (hwj : wj = 40) (hgj : gj = 70) : (gj * 100) / (100 - oj - wj) = 200 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_total_drink_volume_l82_8249


namespace NUMINAMATH_GPT_actual_number_of_children_l82_8209

-- Define the conditions of the problem
def condition1 (C B : ℕ) : Prop := B = 2 * C
def condition2 : ℕ := 320
def condition3 (C B : ℕ) : Prop := B = 4 * (C - condition2)

-- Define the statement to be proved
theorem actual_number_of_children (C B : ℕ) 
  (h1 : condition1 C B) (h2 : condition3 C B) : C = 640 :=
by 
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_actual_number_of_children_l82_8209


namespace NUMINAMATH_GPT_horse_revolutions_l82_8271

theorem horse_revolutions (r1 r2 : ℝ) (rev1 rev2 : ℕ) (h1 : r1 = 30) (h2 : rev1 = 25) (h3 : r2 = 10) : 
  rev2 = 75 :=
by 
  sorry

end NUMINAMATH_GPT_horse_revolutions_l82_8271


namespace NUMINAMATH_GPT_number_of_permutations_l82_8287

def total_letters : ℕ := 10
def freq_s : ℕ := 3
def freq_t : ℕ := 2
def freq_i : ℕ := 2
def freq_a : ℕ := 1
def freq_c : ℕ := 1

theorem number_of_permutations : 
  (total_letters.factorial / (freq_s.factorial * freq_t.factorial * freq_i.factorial * freq_a.factorial * freq_c.factorial)) = 75600 :=
by
  sorry

end NUMINAMATH_GPT_number_of_permutations_l82_8287


namespace NUMINAMATH_GPT_books_per_week_l82_8216

-- Define the conditions
def total_books_read : ℕ := 20
def weeks : ℕ := 5

-- Define the statement to be proved
theorem books_per_week : (total_books_read / weeks) = 4 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_books_per_week_l82_8216


namespace NUMINAMATH_GPT_C3PO_Optimal_Play_Wins_l82_8201

def initial_number : List ℕ := [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]

-- Conditions for the game
structure GameConditions where
  number : List ℕ
  robots : List String
  cannot_swap : List (ℕ × ℕ) -- Pair of digits that cannot be swapped again
  cannot_start_with_zero : Bool
  c3po_starts : Bool

-- Define the initial conditions
def initial_conditions : GameConditions :=
{
  number := initial_number,
  robots := ["C3PO", "R2D2"],
  cannot_swap := [],
  cannot_start_with_zero := true,
  c3po_starts := true
}

-- Define the winning condition for C3PO
def C3PO_wins : Prop :=
  ∀ game : GameConditions, game = initial_conditions → ∃ is_c3po_winner : Bool, is_c3po_winner = true

-- The theorem statement
theorem C3PO_Optimal_Play_Wins : C3PO_wins :=
by
  sorry

end NUMINAMATH_GPT_C3PO_Optimal_Play_Wins_l82_8201


namespace NUMINAMATH_GPT_largest_difference_l82_8253

noncomputable def A := 3 * (1003 ^ 1004)
noncomputable def B := 1003 ^ 1004
noncomputable def C := 1002 * (1003 ^ 1003)
noncomputable def D := 3 * (1003 ^ 1003)
noncomputable def E := 1003 ^ 1003
noncomputable def F := 1003 ^ 1002

theorem largest_difference : 
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end NUMINAMATH_GPT_largest_difference_l82_8253


namespace NUMINAMATH_GPT_find_value_of_N_l82_8234

theorem find_value_of_N (N : ℝ) : 
  2 * ((3.6 * N * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 → 
  N = 0.4800000000000001 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_N_l82_8234


namespace NUMINAMATH_GPT_problem_solution_l82_8218

theorem problem_solution (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 972) : (x + 2) * (x - 2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l82_8218


namespace NUMINAMATH_GPT_no_such_p_l82_8247

theorem no_such_p : ¬ ∃ p : ℕ, p > 0 ∧ (∃ k : ℤ, 4 * p + 35 = k * (3 * p - 7)) :=
by
  sorry

end NUMINAMATH_GPT_no_such_p_l82_8247


namespace NUMINAMATH_GPT_find_f_at_7_l82_8258

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_at_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_7_l82_8258


namespace NUMINAMATH_GPT_abs_eq_self_nonneg_l82_8267

theorem abs_eq_self_nonneg (x : ℝ) : abs x = x ↔ x ≥ 0 :=
sorry

end NUMINAMATH_GPT_abs_eq_self_nonneg_l82_8267


namespace NUMINAMATH_GPT_counterexample_exists_l82_8250

-- Define prime predicate
def is_prime (n : ℕ) : Prop :=
∀ m, m ∣ n → m = 1 ∨ m = n

def counterexample_to_statement (n : ℕ) : Prop :=
  is_prime n ∧ ¬ is_prime (n + 2)

theorem counterexample_exists : ∃ n ∈ [3, 5, 11, 17, 23], is_prime n ∧ ¬ is_prime (n + 2) :=
by
  sorry

end NUMINAMATH_GPT_counterexample_exists_l82_8250


namespace NUMINAMATH_GPT_S_11_l82_8295

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
-- Define that {a_n} is an arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = n * (a 1 + a n) / 2

-- Given condition: a_5 + a_7 = 14
def sum_condition (a : ℕ → ℕ) := a 5 + a 7 = 14

-- Prove S_{11} = 77
theorem S_11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : sum_condition a) :
  S 11 = 77 := by
  -- The proof steps would follow here.
  sorry

end NUMINAMATH_GPT_S_11_l82_8295


namespace NUMINAMATH_GPT_find_a_l82_8241

def A : Set ℝ := {-1, 0, 1}
noncomputable def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem find_a (a : ℝ) : (A ∩ B a = {0}) → a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l82_8241


namespace NUMINAMATH_GPT_groupDivisionWays_l82_8220

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end NUMINAMATH_GPT_groupDivisionWays_l82_8220


namespace NUMINAMATH_GPT_range_of_a_l82_8282

theorem range_of_a (a : ℝ) :
  (∀ x, (x - 2)/5 + 2 ≤ x - 4/5 ∨ x ≤ a) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l82_8282


namespace NUMINAMATH_GPT_calculate_expression_l82_8228

theorem calculate_expression : (36 / (9 + 2 - 6)) * 4 = 28.8 := 
by
    sorry

end NUMINAMATH_GPT_calculate_expression_l82_8228


namespace NUMINAMATH_GPT_solve_inequality_l82_8289

theorem solve_inequality (x : ℝ) : 6 - x - 2 * x^2 < 0 ↔ x < -2 ∨ x > 3 / 2 := sorry

end NUMINAMATH_GPT_solve_inequality_l82_8289


namespace NUMINAMATH_GPT_number_of_knights_l82_8203

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end NUMINAMATH_GPT_number_of_knights_l82_8203


namespace NUMINAMATH_GPT_find_a3_l82_8217

theorem find_a3 (a : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) 
    (h1 : (1 + x) * (a - x)^6 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7)
    (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = 0) :
  a = 1 → a3 = -5 := 
by 
  sorry

end NUMINAMATH_GPT_find_a3_l82_8217


namespace NUMINAMATH_GPT_ratio_of_ages_l82_8263

-- Necessary conditions as definitions in Lean
def combined_age (S D : ℕ) : Prop := S + D = 54
def sam_is_18 (S : ℕ) : Prop := S = 18

-- The statement that we need to prove
theorem ratio_of_ages (S D : ℕ) (h1 : combined_age S D) (h2 : sam_is_18 S) : S / D = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l82_8263


namespace NUMINAMATH_GPT_bracelet_arrangements_l82_8297

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem bracelet_arrangements : 
  (factorial 8) / (8 * 2) = 2520 := by
    sorry

end NUMINAMATH_GPT_bracelet_arrangements_l82_8297


namespace NUMINAMATH_GPT_no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l82_8290

theorem no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5 :
  ¬ ∃ n : ℕ, (∀ d ∈ (Nat.digits 10 n), 5 < d) ∧ (∀ d ∈ (Nat.digits 10 (n^2)), d < 5) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l82_8290


namespace NUMINAMATH_GPT_dance_team_members_l82_8229

theorem dance_team_members (a b c : ℕ)
  (h1 : a + b + c = 100)
  (h2 : b = 2 * a)
  (h3 : c = 2 * a + 10) :
  c = 46 := by
  sorry

end NUMINAMATH_GPT_dance_team_members_l82_8229


namespace NUMINAMATH_GPT_min_L_pieces_correct_l82_8200

noncomputable def min_L_pieces : ℕ :=
  have pieces : Nat := 11
  pieces

theorem min_L_pieces_correct :
  min_L_pieces = 11 := 
by
  sorry

end NUMINAMATH_GPT_min_L_pieces_correct_l82_8200


namespace NUMINAMATH_GPT_intersection_is_correct_l82_8233

namespace IntervalProofs

def setA := {x : ℝ | 3 * x^2 - 14 * x + 16 ≤ 0}
def setB := {x : ℝ | (3 * x - 7) / x > 0}

theorem intersection_is_correct :
  {x | 7 / 3 < x ∧ x ≤ 8 / 3} = setA ∩ setB :=
by
  sorry

end IntervalProofs

end NUMINAMATH_GPT_intersection_is_correct_l82_8233


namespace NUMINAMATH_GPT_probability_correct_l82_8281

structure Bag :=
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

def marbles_drawn_sequence (bag : Bag) : ℚ :=
  let total_marbles := bag.blue + bag.green + bag.yellow
  let prob_blue_first := ↑bag.blue / total_marbles
  let prob_green_second := ↑bag.green / (total_marbles - 1)
  let prob_yellow_third := ↑bag.yellow / (total_marbles - 2)
  prob_blue_first * prob_green_second * prob_yellow_third

theorem probability_correct (bag : Bag) (h : bag = ⟨4, 6, 5⟩) : 
  marbles_drawn_sequence bag = 20 / 455 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_l82_8281


namespace NUMINAMATH_GPT_find_k_l82_8266

-- Auxiliary function to calculate the product of the digits of a number
def productOfDigits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (λ acc d => acc * d) 1

theorem find_k (k : ℕ) (h1 : 0 < k) (h2 : productOfDigits k = (25 * k) / 8 - 211) : 
  k = 72 ∨ k = 88 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l82_8266


namespace NUMINAMATH_GPT_arccos_cos_of_11_l82_8255

-- Define the initial conditions
def angle_in_radians (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 * Real.pi

def arccos_principal_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ Real.pi

-- Define the main theorem to be proved
theorem arccos_cos_of_11 :
  angle_in_radians 11 →
  arccos_principal_range (Real.arccos (Real.cos 11)) →
  Real.arccos (Real.cos 11) = 4.71682 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_arccos_cos_of_11_l82_8255


namespace NUMINAMATH_GPT_find_other_person_weight_l82_8207

theorem find_other_person_weight
    (initial_avg_weight : ℕ)
    (final_avg_weight : ℕ)
    (initial_group_size : ℕ)
    (new_person_weight : ℕ)
    (final_group_size : ℕ)
    (initial_total_weight : ℕ)
    (final_total_weight : ℕ)
    (new_total_weight : ℕ)
    (other_person_weight : ℕ) :
  initial_avg_weight = 48 →
  final_avg_weight = 51 →
  initial_group_size = 23 →
  final_group_size = 25 →
  new_person_weight = 93 →
  initial_total_weight = initial_group_size * initial_avg_weight →
  final_total_weight = final_group_size * final_avg_weight →
  new_total_weight = initial_total_weight + new_person_weight + other_person_weight →
  final_total_weight = new_total_weight →
  other_person_weight = 78 :=
by
  sorry

end NUMINAMATH_GPT_find_other_person_weight_l82_8207


namespace NUMINAMATH_GPT_fraction_of_reciprocal_l82_8239

theorem fraction_of_reciprocal (x : ℝ) (hx : 0 < x) (h : (2/3) * x = y / x) (hx1 : x = 1) : y = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_reciprocal_l82_8239


namespace NUMINAMATH_GPT_student_chose_121_l82_8260

theorem student_chose_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := by
  sorry

end NUMINAMATH_GPT_student_chose_121_l82_8260


namespace NUMINAMATH_GPT_total_acres_cleaned_l82_8213

theorem total_acres_cleaned (A D : ℕ) (h1 : (D - 1) * 90 + 30 = A) (h2 : D * 80 = A) : A = 480 :=
sorry

end NUMINAMATH_GPT_total_acres_cleaned_l82_8213


namespace NUMINAMATH_GPT_find_x_plus_y_l82_8257

theorem find_x_plus_y (x y : ℚ) (h1 : 5 * x - 3 * y = 27) (h2 : 3 * x + 5 * y = 1) : x + y = 31 / 17 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l82_8257


namespace NUMINAMATH_GPT_race_head_start_l82_8206

theorem race_head_start
  (v_A v_B L x : ℝ)
  (h1 : v_A = (4 / 3) * v_B)
  (h2 : L / v_A = (L - x * L) / v_B) :
  x = 1 / 4 :=
sorry

end NUMINAMATH_GPT_race_head_start_l82_8206


namespace NUMINAMATH_GPT_shaded_area_calculation_l82_8235

noncomputable section

-- Definition of the total area of the grid
def total_area (rows columns : ℕ) : ℝ :=
  rows * columns

-- Definition of the area of a right triangle
def triangle_area (base height : ℕ) : ℝ :=
  1 / 2 * base * height

-- Definition of the shaded area in the grid
def shaded_area (total_area triangle_area : ℝ) : ℝ :=
  total_area - triangle_area

-- Theorem stating the shaded area
theorem shaded_area_calculation :
  let rows := 4
  let columns := 13
  let height := 3
  shaded_area (total_area rows columns) (triangle_area columns height) = 32.5 :=
  sorry

end NUMINAMATH_GPT_shaded_area_calculation_l82_8235


namespace NUMINAMATH_GPT_cylindrical_surface_area_increase_l82_8248

theorem cylindrical_surface_area_increase (x : ℝ) :
  (2 * Real.pi * (10 + x)^2 + 2 * Real.pi * (10 + x) * (5 + x) = 
   2 * Real.pi * 10^2 + 2 * Real.pi * 10 * (5 + x)) →
   (x = -10 + 5 * Real.sqrt 6 ∨ x = -10 - 5 * Real.sqrt 6) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cylindrical_surface_area_increase_l82_8248


namespace NUMINAMATH_GPT_petya_square_larger_l82_8291

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end NUMINAMATH_GPT_petya_square_larger_l82_8291


namespace NUMINAMATH_GPT_remainder_of_p_div_x_minus_3_l82_8212

def p (x : ℝ) : ℝ := x^4 - x^3 - 4 * x + 7

theorem remainder_of_p_div_x_minus_3 : 
  let remainder := p 3 
  remainder = 49 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_p_div_x_minus_3_l82_8212


namespace NUMINAMATH_GPT_cube_sum_div_by_9_implies_prod_div_by_3_l82_8298

theorem cube_sum_div_by_9_implies_prod_div_by_3 
  {a1 a2 a3 a4 a5 : ℤ} 
  (h : 9 ∣ a1^3 + a2^3 + a3^3 + a4^3 + a5^3) : 
  3 ∣ a1 * a2 * a3 * a4 * a5 := by
  sorry

end NUMINAMATH_GPT_cube_sum_div_by_9_implies_prod_div_by_3_l82_8298
