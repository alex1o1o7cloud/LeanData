import Mathlib

namespace NUMINAMATH_GPT_GregsAgeIs16_l1812_181277

def CindyAge := 5
def JanAge := CindyAge + 2
def MarciaAge := 2 * JanAge
def GregAge := MarciaAge + 2

theorem GregsAgeIs16 : GregAge = 16 := by
  sorry

end NUMINAMATH_GPT_GregsAgeIs16_l1812_181277


namespace NUMINAMATH_GPT_profit_when_sold_at_double_price_l1812_181210

-- Define the problem parameters

-- Assume cost price (CP)
def CP : ℕ := 100

-- Define initial selling price (SP) with 50% profit
def SP : ℕ := CP + (CP / 2)

-- Define new selling price when sold at double the initial selling price
def SP2 : ℕ := 2 * SP

-- Define profit when sold at SP2
def profit : ℕ := SP2 - CP

-- Define the percentage profit
def profit_percentage : ℕ := (profit * 100) / CP

-- The proof goal: if selling at double the price, percentage profit is 200%
theorem profit_when_sold_at_double_price : profit_percentage = 200 :=
by {sorry}

end NUMINAMATH_GPT_profit_when_sold_at_double_price_l1812_181210


namespace NUMINAMATH_GPT_boy_running_time_l1812_181207

theorem boy_running_time :
  let side_length := 60
  let speed1 := 9 * 1000 / 3600       -- 9 km/h to m/s
  let speed2 := 6 * 1000 / 3600       -- 6 km/h to m/s
  let speed3 := 8 * 1000 / 3600       -- 8 km/h to m/s
  let speed4 := 7 * 1000 / 3600       -- 7 km/h to m/s
  let hurdle_time := 5 * 3 * 4        -- 3 hurdles per side, 4 sides
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3
  let time4 := side_length / speed4
  let total_time := time1 + time2 + time3 + time4 + hurdle_time
  total_time = 177.86 := by
{
  -- actual proof would be provided here
  sorry
}

end NUMINAMATH_GPT_boy_running_time_l1812_181207


namespace NUMINAMATH_GPT_find_value_of_a_l1812_181278

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 = 0) → (x - y + 3 = 0) → (-a) * 1 = -1) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1812_181278


namespace NUMINAMATH_GPT_letters_symmetry_l1812_181279

theorem letters_symmetry (people : Fin 20) (sends : Fin 20 → Finset (Fin 20)) (h : ∀ p, (sends p).card = 10) :
  ∃ i j : Fin 20, i ≠ j ∧ j ∈ sends i ∧ i ∈ sends j :=
by
  sorry

end NUMINAMATH_GPT_letters_symmetry_l1812_181279


namespace NUMINAMATH_GPT_product_213_16_l1812_181248

theorem product_213_16 :
  (213 * 16 = 3408) :=
by
  have h1 : (0.16 * 2.13 = 0.3408) := by sorry
  sorry

end NUMINAMATH_GPT_product_213_16_l1812_181248


namespace NUMINAMATH_GPT_additional_savings_is_300_l1812_181238

-- Define constants
def price_per_window : ℕ := 120
def discount_threshold : ℕ := 10
def discount_per_window : ℕ := 10
def free_window_threshold : ℕ := 5

-- Define the number of windows Alice needs
def alice_windows : ℕ := 9

-- Define the number of windows Bob needs
def bob_windows : ℕ := 12

-- Define the function to calculate total cost without discount
def cost_without_discount (n : ℕ) : ℕ := n * price_per_window

-- Define the function to calculate cost with discount
def cost_with_discount (n : ℕ) : ℕ :=
  let full_windows := n - n / free_window_threshold
  let discounted_price := if n > discount_threshold then price_per_window - discount_per_window else price_per_window
  full_windows * discounted_price

-- Define the function to calculate savings when windows are bought separately
def savings_separately : ℕ :=
  (cost_without_discount alice_windows + cost_without_discount bob_windows) 
  - (cost_with_discount alice_windows + cost_with_discount bob_windows)

-- Define the function to calculate savings when windows are bought together
def savings_together : ℕ :=
  let combined_windows := alice_windows + bob_windows
  cost_without_discount combined_windows - cost_with_discount combined_windows

-- Prove that the additional savings when buying together is $300
theorem additional_savings_is_300 : savings_together - savings_separately = 300 := by
  -- missing proof
  sorry

end NUMINAMATH_GPT_additional_savings_is_300_l1812_181238


namespace NUMINAMATH_GPT_mark_second_part_playtime_l1812_181235

theorem mark_second_part_playtime (total_time initial_time sideline_time : ℕ) 
  (h1 : total_time = 90) (h2 : initial_time = 20) (h3 : sideline_time = 35) :
  total_time - initial_time - sideline_time = 35 :=
sorry

end NUMINAMATH_GPT_mark_second_part_playtime_l1812_181235


namespace NUMINAMATH_GPT_parabola_tangent_circle_l1812_181206

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

theorem parabola_tangent_circle (p : ℝ) (hp : p > 0)
  (x0 : ℝ) (hx0 : x0 = p)
  (M : ℝ × ℝ) (hM : M = (x0, 2 * (Real.sqrt 2)))
  (MA AF : ℝ) (h_ratio : MA / AF = 2) :
  p = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangent_circle_l1812_181206


namespace NUMINAMATH_GPT_chocolate_syrup_amount_l1812_181223

theorem chocolate_syrup_amount (x : ℝ) (H1 : 2 * x + 6 = 14) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_syrup_amount_l1812_181223


namespace NUMINAMATH_GPT_actual_distance_traveled_l1812_181284

-- Definitions based on conditions
def original_speed : ℕ := 12
def increased_speed : ℕ := 20
def distance_difference : ℕ := 24

-- We need to prove the actual distance traveled by the person.
theorem actual_distance_traveled : 
  ∃ t : ℕ, increased_speed * t = original_speed * t + distance_difference → original_speed * t = 36 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1812_181284


namespace NUMINAMATH_GPT_pam_total_apples_l1812_181270

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end NUMINAMATH_GPT_pam_total_apples_l1812_181270


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l1812_181282

theorem smallest_prime_divisor_of_3_pow_19_add_11_pow_23 :
  ∀ (n : ℕ), Prime n → n ∣ 3^19 + 11^23 → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l1812_181282


namespace NUMINAMATH_GPT_candy_box_price_increase_l1812_181258

theorem candy_box_price_increase
  (C : ℝ) -- Original price of the candy box
  (S : ℝ := 12) -- Original price of a can of soda
  (combined_price : C + S = 16) -- Combined price before increase
  (candy_box_increase : C + 0.25 * C = 1.25 * C) -- Price increase definition
  (soda_increase : S + 0.50 * S = 18) -- New price of soda after increase
  : 1.25 * C = 5 := sorry

end NUMINAMATH_GPT_candy_box_price_increase_l1812_181258


namespace NUMINAMATH_GPT_lisa_caffeine_l1812_181265

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end NUMINAMATH_GPT_lisa_caffeine_l1812_181265


namespace NUMINAMATH_GPT_area_sum_eq_l1812_181280

-- Define the conditions given in the problem
variables {A B C P Q R M N : Type*}

-- Define the properties of the points
variables (triangle_ABC : Triangle A B C)
          (point_P : OnSegment P A B)
          (point_Q : OnSegment Q B C)
          (point_R : OnSegment R A C)
          (parallelogram_PQCR : Parallelogram P Q C R)
          (intersection_M : Intersection M (LineSegment AQ) (LineSegment PR))
          (intersection_N : Intersection N (LineSegment BR) (LineSegment PQ))

-- Define the areas of the triangles involved
variables (area_AMP area_BNP area_CQR : ℝ)

-- Define the conditions for the areas of the triangles
variables (h_area_AMP : area_AMP = Area (Triangle A M P))
          (h_area_BNP : area_BNP = Area (Triangle B N P))
          (h_area_CQR : area_CQR = Area (Triangle C Q R))

-- The theorem to be proved
theorem area_sum_eq :
  area_AMP + area_BNP = area_CQR :=
sorry

end NUMINAMATH_GPT_area_sum_eq_l1812_181280


namespace NUMINAMATH_GPT_units_digit_42_pow_5_add_27_pow_5_l1812_181261

theorem units_digit_42_pow_5_add_27_pow_5 :
  (42 ^ 5 + 27 ^ 5) % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_42_pow_5_add_27_pow_5_l1812_181261


namespace NUMINAMATH_GPT_perpendicular_vector_x_value_l1812_181228

-- Definitions based on the given problem conditions
def dot_product_perpendicular (a1 a2 b1 b2 x : ℝ) : Prop :=
  (a1 * b1 + a2 * b2 = 0)

-- Statement to be proved
theorem perpendicular_vector_x_value (x : ℝ) :
  dot_product_perpendicular 4 x 2 4 x → x = -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_perpendicular_vector_x_value_l1812_181228


namespace NUMINAMATH_GPT_math_problem_l1812_181229

theorem math_problem :
  (- (1 / 8)) ^ 2007 * (- 8) ^ 2008 = -8 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1812_181229


namespace NUMINAMATH_GPT_find_a_l1812_181299

theorem find_a (a : ℝ) : (dist (⟨-2, -1⟩ : ℝ × ℝ) (⟨a, 3⟩ : ℝ × ℝ) = 5) ↔ (a = 1 ∨ a = -5) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1812_181299


namespace NUMINAMATH_GPT_apples_sold_l1812_181294

theorem apples_sold (a1 a2 a3 : ℕ) (h1 : a3 = a2 / 4 + 8) (h2 : a2 = a1 / 4 + 8) (h3 : a3 = 18) : a1 = 128 :=
by
  sorry

end NUMINAMATH_GPT_apples_sold_l1812_181294


namespace NUMINAMATH_GPT_lives_after_bonus_l1812_181267

variable (X Y Z : ℕ)

theorem lives_after_bonus (X Y Z : ℕ) : (X - Y + 3 * Z) = (X - Y + 3 * Z) :=
sorry

end NUMINAMATH_GPT_lives_after_bonus_l1812_181267


namespace NUMINAMATH_GPT_five_million_squared_l1812_181217

theorem five_million_squared : (5 * 10^6)^2 = 25 * 10^12 := by
  sorry

end NUMINAMATH_GPT_five_million_squared_l1812_181217


namespace NUMINAMATH_GPT_general_formula_sequence_less_than_zero_maximum_sum_value_l1812_181260

variable (n : ℕ)

-- Helper definition
def arithmetic_seq (d : ℤ) (a₁ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Conditions given in the problem
def a₁ : ℤ := 31
def a₄ : ℤ := 7
def d : ℤ := (a₄ - a₁) / 3

-- Definitions extracted from problem conditions
def an (n : ℕ) : ℤ := arithmetic_seq d a₁ n
def Sn (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Proving the general formula aₙ = -8n + 39
theorem general_formula :
  ∀ (n : ℕ), an n = -8 * n + 39 :=
by
  sorry

-- Proving when the sequence starts to be less than 0
theorem sequence_less_than_zero :
  ∀ (n : ℕ), n ≥ 5 → an n < 0 :=
by
  sorry

-- Proving that the sum Sn has a maximum value
theorem maximum_sum_value :
  Sn 4 = 76 ∧ ∀ (n : ℕ), Sn n ≤ 76 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_sequence_less_than_zero_maximum_sum_value_l1812_181260


namespace NUMINAMATH_GPT_probability_of_qualification_l1812_181272

-- Define the probability of hitting a target and the number of shots
def probability_hit : ℝ := 0.4
def number_of_shots : ℕ := 3

-- Define the probability of hitting a specific number of targets
noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the event of qualifying by hitting at least 2 targets
noncomputable def probability_qualify (n : ℕ) (p : ℝ) : ℝ :=
  binomial n 2 p + binomial n 3 p

-- The theorem we want to prove
theorem probability_of_qualification : probability_qualify number_of_shots probability_hit = 0.352 :=
  by sorry

end NUMINAMATH_GPT_probability_of_qualification_l1812_181272


namespace NUMINAMATH_GPT_smallest_x_value_l1812_181236

-- Definitions based on given problem conditions
def is_solution (x y : ℕ) : Prop :=
  0 < x ∧ 0 < y ∧ (3 : ℝ) / 4 = y / (252 + x)

theorem smallest_x_value : ∃ x : ℕ, ∀ y : ℕ, is_solution x y → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_value_l1812_181236


namespace NUMINAMATH_GPT_tangent_line_eq_l1812_181214

theorem tangent_line_eq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x y : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f' x = Real.exp x) →
  f 0 = 1 →
  f' 0 = 1 →
  x = 0 →
  y = 1 →
  x - y + 1 = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1812_181214


namespace NUMINAMATH_GPT_terminating_decimal_expansion_l1812_181276

theorem terminating_decimal_expansion (a b : ℝ) :
  (13 / 200 = a / 10^b) → a = 52 ∧ b = 3 ∧ a / 10^b = 0.052 :=
by sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_l1812_181276


namespace NUMINAMATH_GPT_yellow_jelly_bean_probability_l1812_181293

theorem yellow_jelly_bean_probability :
  let p_red := 0.15
  let p_orange := 0.35
  let p_green := 0.25
  let p_yellow := 1 - (p_red + p_orange + p_green)
  p_yellow = 0.25 := by
    let p_red := 0.15
    let p_orange := 0.35
    let p_green := 0.25
    let p_yellow := 1 - (p_red + p_orange + p_green)
    show p_yellow = 0.25
    sorry

end NUMINAMATH_GPT_yellow_jelly_bean_probability_l1812_181293


namespace NUMINAMATH_GPT_rectangle_area_l1812_181273

theorem rectangle_area
  (x : ℝ)
  (perimeter_eq_160 : 10 * x = 160) :
  4 * (4 * x * x) = 1024 :=
by
  -- We would solve the problem and show the steps here
  sorry

end NUMINAMATH_GPT_rectangle_area_l1812_181273


namespace NUMINAMATH_GPT_amc12a_2006_p24_l1812_181232

theorem amc12a_2006_p24 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
by
  sorry

end NUMINAMATH_GPT_amc12a_2006_p24_l1812_181232


namespace NUMINAMATH_GPT_original_number_is_19_l1812_181283

theorem original_number_is_19 (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 := 
by 
  sorry

end NUMINAMATH_GPT_original_number_is_19_l1812_181283


namespace NUMINAMATH_GPT_complex_number_is_real_implies_m_eq_3_l1812_181218

open Complex

theorem complex_number_is_real_implies_m_eq_3 (m : ℝ) :
  (∃ (z : ℂ), z = (1 / (m + 5) : ℝ) + (m^2 + 2 * m - 15) * I ∧ z.im = 0) →
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_is_real_implies_m_eq_3_l1812_181218


namespace NUMINAMATH_GPT_original_number_is_16_l1812_181288

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_16_l1812_181288


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1812_181295

theorem sale_in_fifth_month 
  (sale_month_1 : ℕ) (sale_month_2 : ℕ) (sale_month_3 : ℕ) (sale_month_4 : ℕ) 
  (sale_month_6 : ℕ) (average_sale : ℕ) 
  (h1 : sale_month_1 = 5266) (h2 : sale_month_2 = 5744) (h3 : sale_month_3 = 5864) 
  (h4 : sale_month_4 = 6122) (h6 : sale_month_6 = 4916) (h_avg : average_sale = 5750) :
  ∃ sale_month_5, sale_month_5 = 6588 :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1812_181295


namespace NUMINAMATH_GPT_find_a_l1812_181289

def mul_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : mul_op a 3 = 7) : a = 8 :=
sorry

end NUMINAMATH_GPT_find_a_l1812_181289


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_a_range_l1812_181225

-- Define the function f
def f (x a : ℝ) := |x - a| + 5 * x

-- Problem Part 1: Prove for a = -1, the solution set for f(x) ≤ 5x + 3 is [-4, 2]
theorem problem1_solution_set :
  ∀ (x : ℝ), f x (-1) ≤ 5 * x + 3 ↔ -4 ≤ x ∧ x ≤ 2 := by
  sorry

-- Problem Part 2: Prove that if f(x) ≥ 0 for all x ≥ -1, then a ≥ 4 or a ≤ -6
theorem problem2_a_range (a : ℝ) :
  (∀ (x : ℝ), x ≥ -1 → f x a ≥ 0) ↔ a ≥ 4 ∨ a ≤ -6 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_a_range_l1812_181225


namespace NUMINAMATH_GPT_new_person_weight_l1812_181227

-- The conditions from part (a)
variables (average_increase: ℝ) (num_people: ℕ) (weight_lost_person: ℝ)
variables (total_increase: ℝ) (new_weight: ℝ)

-- Assigning the given conditions
axiom h1 : average_increase = 2.5
axiom h2 : num_people = 8
axiom h3 : weight_lost_person = 45
axiom h4 : total_increase = num_people * average_increase
axiom h5 : new_weight = weight_lost_person + total_increase

-- The proof goal: proving that the new person's weight is 65 kg
theorem new_person_weight : new_weight = 65 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_new_person_weight_l1812_181227


namespace NUMINAMATH_GPT_binomial_7_2_l1812_181251

open Nat

theorem binomial_7_2 : (Nat.choose 7 2) = 21 :=
by
  sorry

end NUMINAMATH_GPT_binomial_7_2_l1812_181251


namespace NUMINAMATH_GPT_range_a_l1812_181212

theorem range_a (a : ℝ) (x : ℝ) : 
    (∀ x, (x = 1 → x - a ≥ 1) ∧ (x = -1 → ¬(x - a ≥ 1))) ↔ (-2 < a ∧ a ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1812_181212


namespace NUMINAMATH_GPT_smallest_C_l1812_181274

-- Defining the problem and the conditions
theorem smallest_C (k : ℕ) (C : ℕ) :
  (∀ n : ℕ, n ≥ k → (C * Nat.choose (2 * n) (n + k)) % (n + k + 1) = 0) ↔
  C = 2 * k + 1 :=
by sorry

end NUMINAMATH_GPT_smallest_C_l1812_181274


namespace NUMINAMATH_GPT_treasures_found_second_level_l1812_181275

theorem treasures_found_second_level:
  ∀ (P T1 S T2 : ℕ), 
    P = 4 → 
    T1 = 6 → 
    S = 32 → 
    S = P * T1 + P * T2 → 
    T2 = 2 := 
by
  intros P T1 S T2 hP hT1 hS hTotal
  sorry

end NUMINAMATH_GPT_treasures_found_second_level_l1812_181275


namespace NUMINAMATH_GPT_erasers_per_friend_l1812_181255

variable (erasers friends : ℕ)

theorem erasers_per_friend (h1 : erasers = 3840) (h2 : friends = 48) :
  erasers / friends = 80 :=
by sorry

end NUMINAMATH_GPT_erasers_per_friend_l1812_181255


namespace NUMINAMATH_GPT_eval_expression_l1812_181259

theorem eval_expression : (256 : ℝ) ^ ((-2 : ℝ) ^ (-3 : ℝ)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1812_181259


namespace NUMINAMATH_GPT_recreation_percentage_l1812_181209

def wages_last_week (W : ℝ) : ℝ := W
def spent_on_recreation_last_week (W : ℝ) : ℝ := 0.15 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def spent_on_recreation_this_week (W : ℝ) : ℝ := 0.30 * (wages_this_week W)

theorem recreation_percentage (W : ℝ) (hW: W > 0) :
  (spent_on_recreation_this_week W) / (spent_on_recreation_last_week W) * 100 = 180 := by
  sorry

end NUMINAMATH_GPT_recreation_percentage_l1812_181209


namespace NUMINAMATH_GPT_watermelon_ratio_l1812_181287

theorem watermelon_ratio (michael_weight : ℕ) (john_weight : ℕ) (clay_weight : ℕ)
  (h₁ : michael_weight = 8) 
  (h₂ : john_weight = 12) 
  (h₃ : john_weight * 2 = clay_weight) :
  clay_weight / michael_weight = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_watermelon_ratio_l1812_181287


namespace NUMINAMATH_GPT_john_got_80_percent_of_value_l1812_181204

noncomputable def percentage_of_value (P : ℝ) : Prop :=
  let old_system_cost := 250
  let new_system_cost := 600
  let discount_percentage := 0.25
  let pocket_spent := 250
  let discount_amount := discount_percentage * new_system_cost
  let price_after_discount := new_system_cost - discount_amount
  let value_for_old_system := (P / 100) * old_system_cost
  value_for_old_system + pocket_spent = price_after_discount

theorem john_got_80_percent_of_value : percentage_of_value 80 :=
by
  sorry

end NUMINAMATH_GPT_john_got_80_percent_of_value_l1812_181204


namespace NUMINAMATH_GPT_probability_calculation_l1812_181202

open Classical

def probability_odd_sum_given_even_product :=
  let num_even := 4  -- even numbers: 2, 4, 6, 8
  let num_odd := 4   -- odd numbers: 1, 3, 5, 7
  let total_outcomes := 8^5
  let prob_all_odd := (num_odd / 8)^5
  let prob_even_product := 1 - prob_all_odd

  let ways_one_odd := 5 * num_odd * num_even^4
  let ways_three_odd := Nat.choose 5 3 * num_odd^3 * num_even^2
  let ways_five_odd := num_odd^5

  let favorable_outcomes := ways_one_odd + ways_three_odd + ways_five_odd
  let total_even_product_outcomes := total_outcomes * prob_even_product

  favorable_outcomes / total_even_product_outcomes

theorem probability_calculation :
  probability_odd_sum_given_even_product = rational_result := sorry

end NUMINAMATH_GPT_probability_calculation_l1812_181202


namespace NUMINAMATH_GPT_floss_leftover_l1812_181231

noncomputable def leftover_floss
    (students : ℕ)
    (floss_per_student : ℚ)
    (floss_per_packet : ℚ) :
    ℚ :=
  let total_needed := students * floss_per_student
  let packets_needed := (total_needed / floss_per_packet).ceil
  let total_floss := packets_needed * floss_per_packet
  total_floss - total_needed

theorem floss_leftover {students : ℕ} {floss_per_student floss_per_packet : ℚ}
    (h_students : students = 20)
    (h_floss_per_student : floss_per_student = 3 / 2)
    (h_floss_per_packet : floss_per_packet = 35) :
    leftover_floss students floss_per_student floss_per_packet = 5 :=
by
  rw [h_students, h_floss_per_student, h_floss_per_packet]
  simp only [leftover_floss]
  norm_num
  sorry

end NUMINAMATH_GPT_floss_leftover_l1812_181231


namespace NUMINAMATH_GPT_exists_consecutive_nat_with_integer_quotient_l1812_181266

theorem exists_consecutive_nat_with_integer_quotient :
  ∃ n : ℕ, (n + 1) / n = 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_consecutive_nat_with_integer_quotient_l1812_181266


namespace NUMINAMATH_GPT_Carla_more_miles_than_Daniel_after_5_hours_l1812_181286

theorem Carla_more_miles_than_Daniel_after_5_hours (Carla_distance : ℝ) (Daniel_distance : ℝ) (h_Carla : Carla_distance = 100) (h_Daniel : Daniel_distance = 75) : 
  Carla_distance - Daniel_distance = 25 := 
by
  sorry

end NUMINAMATH_GPT_Carla_more_miles_than_Daniel_after_5_hours_l1812_181286


namespace NUMINAMATH_GPT_custom_operation_example_l1812_181208

-- Define the custom operation
def custom_operation (a b : ℕ) : ℕ := a * b + (a - b)

-- State the theorem
theorem custom_operation_example : custom_operation (custom_operation 3 2) 4 = 31 :=
by
  -- the proof will go here, but we skip it for now
  sorry

end NUMINAMATH_GPT_custom_operation_example_l1812_181208


namespace NUMINAMATH_GPT_select_pencils_l1812_181242

theorem select_pencils (boxes : Fin 10 → ℕ) (colors : ∀ (i : Fin 10), Fin (boxes i) → Fin 10) :
  (∀ i : Fin 10, 1 ≤ boxes i) → -- Each box is non-empty
  (∀ i j : Fin 10, i ≠ j → boxes i ≠ boxes j) → -- Different number of pencils in each box
  ∃ (selection : Fin 10 → Fin 10), -- Function to select a pencil color from each box
  Function.Injective selection := -- All selected pencils have different colors
sorry

end NUMINAMATH_GPT_select_pencils_l1812_181242


namespace NUMINAMATH_GPT_train_length_l1812_181246

-- Definitions based on conditions
def train_speed_kmh := 54 -- speed of the train in km/h
def time_to_cross_sec := 16 -- time to cross the telegraph post in seconds
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 5 / 18 -- conversion factor from km/h to m/s

-- Prove that the length of the train is 240 meters
theorem train_length (h1 : train_speed_kmh = 54) (h2 : time_to_cross_sec = 16) : 
  (kmh_to_ms train_speed_kmh * time_to_cross_sec) = 240 := by
  sorry

end NUMINAMATH_GPT_train_length_l1812_181246


namespace NUMINAMATH_GPT_expression_evaluation_l1812_181244

theorem expression_evaluation :
  5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1812_181244


namespace NUMINAMATH_GPT_first_four_cards_all_red_l1812_181264

noncomputable def probability_first_four_red_cards : ℚ :=
  (26 / 52) * (25 / 51) * (24 / 50) * (23 / 49)

theorem first_four_cards_all_red :
  probability_first_four_red_cards = 276 / 9801 :=
by
  -- The proof itself is not required; we are only stating it.
  sorry

end NUMINAMATH_GPT_first_four_cards_all_red_l1812_181264


namespace NUMINAMATH_GPT_perfect_game_points_l1812_181269

theorem perfect_game_points (points_per_game games_played total_points : ℕ) 
  (h1 : points_per_game = 21) 
  (h2 : games_played = 11) 
  (h3 : total_points = points_per_game * games_played) : 
  total_points = 231 := 
by 
  sorry

end NUMINAMATH_GPT_perfect_game_points_l1812_181269


namespace NUMINAMATH_GPT_digit_D_value_l1812_181241

/- The main conditions are:
1. A, B, C, D are digits (0 through 9)
2. Addition equation: AB + CA = D0
3. Subtraction equation: AB - CA = 00
-/

theorem digit_D_value (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (add_eq : 10 * A + B + 10 * C + A = 10 * D + 0)
  (sub_eq : 10 * A + B - (10 * C + A) = 0) :
  D = 1 :=
sorry

end NUMINAMATH_GPT_digit_D_value_l1812_181241


namespace NUMINAMATH_GPT_find_z_l1812_181237

theorem find_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z * i = 2 - i) : z = -1 - 2 * i := 
by
  sorry

end NUMINAMATH_GPT_find_z_l1812_181237


namespace NUMINAMATH_GPT_ratio_pentagon_rectangle_l1812_181222

theorem ratio_pentagon_rectangle (P: ℝ) (a w: ℝ) (h1: 5 * a = P) (h2: 6 * w = P) (h3: P = 75) : a / w = 6 / 5 := 
by 
  -- Proof steps will be provided to conclude this result 
  sorry

end NUMINAMATH_GPT_ratio_pentagon_rectangle_l1812_181222


namespace NUMINAMATH_GPT_businesses_can_apply_l1812_181254

-- Define conditions
def total_businesses : ℕ := 72
def businesses_fired : ℕ := 36 -- Half of total businesses (72 / 2)
def businesses_quit : ℕ := 24 -- One third of total businesses (72 / 3)

-- Theorem: Number of businesses Brandon can still apply to
theorem businesses_can_apply : (total_businesses - (businesses_fired + businesses_quit)) = 12 := 
by
  sorry

end NUMINAMATH_GPT_businesses_can_apply_l1812_181254


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l1812_181203

noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ + 1/2 + Real.sin x

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) := sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l1812_181203


namespace NUMINAMATH_GPT_find_x_value_l1812_181245

theorem find_x_value (x : ℝ) (h : 3 * x + 6 * x + x + 2 * x = 360) : x = 30 :=
by sorry

end NUMINAMATH_GPT_find_x_value_l1812_181245


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l1812_181215

theorem domain_of_sqrt_function (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, (1 / (Real.log x) - 2) ≥ 0) 
  (h2 : ∀ x, Real.log x ≠ 0) : 
  (1 < x ∧ x ≤ Real.sqrt 10) ↔ (∀ x, 0 < Real.log x ∧ Real.log x ≤ 1 / 2) := 
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l1812_181215


namespace NUMINAMATH_GPT_gasoline_price_increase_l1812_181297

theorem gasoline_price_increase
  (P Q : ℝ)
  (h1 : (P * Q) * 1.10 = P * (1 + X / 100) * Q * 0.88) :
  X = 25 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l1812_181297


namespace NUMINAMATH_GPT_number_of_ways_to_fulfill_order_l1812_181292

open Finset Nat

/-- Bill must buy exactly eight donuts from a shop offering five types, 
with at least two of the first type and one of each of the other four types. 
Prove that there are exactly 15 different ways to fulfill this order. -/
theorem number_of_ways_to_fulfill_order : 
  let total_donuts := 8
  let types_of_donuts := 5
  let mandatory_first_type := 2
  let mandatory_each_other_type := 1
  let remaining_donuts := total_donuts - (mandatory_first_type + 4 * mandatory_each_other_type)
  let combinations := (remaining_donuts + types_of_donuts - 1).choose (types_of_donuts - 1)
  combinations = 15 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_fulfill_order_l1812_181292


namespace NUMINAMATH_GPT_solve_equation_l1812_181230

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1812_181230


namespace NUMINAMATH_GPT_number_of_shirts_l1812_181224

theorem number_of_shirts (ratio_pants_shirts: ℕ) (num_pants: ℕ) (S: ℕ) : 
  ratio_pants_shirts = 7 ∧ num_pants = 14 → S = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_shirts_l1812_181224


namespace NUMINAMATH_GPT_first_term_of_arithmetic_series_l1812_181234

theorem first_term_of_arithmetic_series 
  (a d : ℝ)
  (h1 : 20 * (2 * a + 39 * d) = 600)
  (h2 : 20 * (2 * a + 119 * d) = 1800) :
  a = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_series_l1812_181234


namespace NUMINAMATH_GPT_lesser_fraction_l1812_181247

theorem lesser_fraction 
  (x y : ℚ)
  (h_sum : x + y = 13 / 14)
  (h_prod : x * y = 1 / 5) :
  min x y = 87 / 700 := sorry

end NUMINAMATH_GPT_lesser_fraction_l1812_181247


namespace NUMINAMATH_GPT_find_RS_length_l1812_181213

-- Define the given conditions
def tetrahedron_edges (a b c d e f : ℕ) : Prop :=
  (a = 8 ∨ a = 14 ∨ a = 19 ∨ a = 28 ∨ a = 37 ∨ a = 42) ∧
  (b = 8 ∨ b = 14 ∨ b = 19 ∨ b = 28 ∨ b = 37 ∨ b = 42) ∧
  (c = 8 ∨ c = 14 ∨ c = 19 ∨ c = 28 ∨ c = 37 ∨ c = 42) ∧
  (d = 8 ∨ d = 14 ∨ d = 19 ∨ d = 28 ∨ d = 37 ∨ d = 42) ∧
  (e = 8 ∨ e = 14 ∨ e = 19 ∨ e = 28 ∨ e = 37 ∨ e = 42) ∧
  (f = 8 ∨ f = 14 ∨ f = 19 ∨ f = 28 ∨ f = 37 ∨ f = 42) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def length_of_PQ (pq : ℕ) : Prop := pq = 42

def length_of_RS (rs : ℕ) (a b c d e f pq : ℕ) : Prop :=
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  (rs = 14)

-- The theorem statement
theorem find_RS_length (a b c d e f pq rs : ℕ) :
  tetrahedron_edges a b c d e f ∧ length_of_PQ pq →
  length_of_RS rs a b c d e f pq :=
by sorry

end NUMINAMATH_GPT_find_RS_length_l1812_181213


namespace NUMINAMATH_GPT_distinct_cubes_meet_condition_l1812_181221

theorem distinct_cubes_meet_condition :
  ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + b + c + d + e + f = 60) ∧
    ∃ (k : ℕ), 
        ((a = k) ∧ (b = k) ∧ (c = k) ∧ (d = k) ∧ (e = k) ∧ (f = k)) ∧
        -- Number of distinct ways
        (∃ (num_ways : ℕ), num_ways = 84) :=
sorry

end NUMINAMATH_GPT_distinct_cubes_meet_condition_l1812_181221


namespace NUMINAMATH_GPT_pirate_schooner_problem_l1812_181290

theorem pirate_schooner_problem (p : ℕ) (h1 : 10 < p) 
  (h2 : 0.54 * (p - 10) = (54 : ℝ) / 100 * (p - 10)) 
  (h3 : 0.34 * (p - 10) = (34 : ℝ) / 100 * (p - 10)) 
  (h4 : 2 / 3 * p = (2 : ℝ) / 3 * p) : 
  p = 60 := 
sorry

end NUMINAMATH_GPT_pirate_schooner_problem_l1812_181290


namespace NUMINAMATH_GPT_avg_height_and_weight_of_class_l1812_181253

-- Defining the given conditions
def num_students : ℕ := 70
def num_girls : ℕ := 40
def num_boys : ℕ := 30

def avg_height_30_girls : ℕ := 160
def avg_height_10_girls : ℕ := 156
def avg_height_15_boys_high : ℕ := 170
def avg_height_15_boys_low : ℕ := 160
def avg_weight_girls : ℕ := 55
def avg_weight_boys : ℕ := 60

-- Theorem stating the given question
theorem avg_height_and_weight_of_class :
  ∃ (avg_height avg_weight : ℚ),
    avg_height = (30 * 160 + 10 * 156 + 15 * 170 + 15 * 160) / num_students ∧
    avg_weight = (40 * 55 + 30 * 60) / num_students ∧
    avg_height = 161.57 ∧
    avg_weight = 57.14 :=
by
  -- include the solution steps here if required
  -- examples using appropriate constructs like ring, norm_num, etc.
  sorry

end NUMINAMATH_GPT_avg_height_and_weight_of_class_l1812_181253


namespace NUMINAMATH_GPT_bead_count_l1812_181268

variable (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)

theorem bead_count : total_beads = 40 ∧ blue_beads = 5 ∧ red_beads = 2 * blue_beads ∧ white_beads = blue_beads + red_beads ∧ silver_beads = total_beads - (blue_beads + red_beads + white_beads) → silver_beads = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bead_count_l1812_181268


namespace NUMINAMATH_GPT_regular_polygon_sides_l1812_181271

-- Define the measure of each exterior angle
def exterior_angle (n : ℕ) (angle : ℝ) : Prop :=
  angle = 40.0

-- Define the sum of exterior angles of any polygon
def sum_exterior_angles (n : ℕ) (total_angle : ℝ) : Prop :=
  total_angle = 360.0

-- Theorem to prove
theorem regular_polygon_sides (n : ℕ) :
  (exterior_angle n 40.0) ∧ (sum_exterior_angles n 360.0) → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1812_181271


namespace NUMINAMATH_GPT_quadrilateral_area_l1812_181250

theorem quadrilateral_area {ABCQ : ℝ} 
  (side_length : ℝ) 
  (D P E N : ℝ → Prop) 
  (midpoints : ℝ) 
  (W X Y Z : ℝ → Prop) :
  side_length = 4 → 
  (∀ a b : ℝ, D a ∧ P b → a = 1 ∧ b = 1) → 
  (∀ c d : ℝ, E c ∧ N d → c = 1 ∧ d = 1) →
  (∀ w x y z : ℝ, W w ∧ X x ∧ Y y ∧ Z z → w = 0.5 ∧ x = 0.5 ∧ y = 0.5 ∧ z = 0.5) →
  ∃ (area : ℝ), area = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1812_181250


namespace NUMINAMATH_GPT_find_x_for_vectors_l1812_181200

theorem find_x_for_vectors
  (x : ℝ)
  (h1 : x ∈ Set.Icc 0 Real.pi)
  (a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2)))
  (b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2)))
  (h2 : (a.1 + b.1)^2 + (a.2 + b.2)^2 = 1) :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_vectors_l1812_181200


namespace NUMINAMATH_GPT_thm1_thm2_thm3_thm4_l1812_181256

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions relating lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p q : Plane) : Prop := sorry
def perpendicular_planes (p q : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: This statement is false, so we negate its for proof.
theorem thm1 (h1 : parallel_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  ¬ parallel_lines m n :=
sorry

-- Theorem 2: This statement is true, we need to prove it.
theorem thm2 (h1 : perpendicular_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 3: This statement is true, we need to prove it.
theorem thm3 (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 4: This statement is false, so we negate its for proof.
theorem thm4 (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  ¬ parallel_lines m n :=
sorry

end NUMINAMATH_GPT_thm1_thm2_thm3_thm4_l1812_181256


namespace NUMINAMATH_GPT_triangle_area_l1812_181263

def right_triangle_area (hypotenuse leg1 : ℕ) : ℕ :=
  if (hypotenuse ^ 2 - leg1 ^ 2) > 0 then (1 / 2) * leg1 * (hypotenuse ^ 2 - leg1 ^ 2).sqrt else 0

theorem triangle_area (hypotenuse leg1 : ℕ) (h_hypotenuse : hypotenuse = 13) (h_leg1 : leg1 = 5) :
  right_triangle_area hypotenuse leg1 = 30 :=
by
  rw [h_hypotenuse, h_leg1]
  sorry

end NUMINAMATH_GPT_triangle_area_l1812_181263


namespace NUMINAMATH_GPT_not_equivalent_expression_l1812_181211

/--
Let A, B, C, D be expressions defined as follows:
A := 3 * (x + 2)
B := (-9 * x - 18) / -3
C := (1/3) * (3 * x) + (2/3) * 9
D := (1/3) * (9 * x + 18)

Prove that only C is not equivalent to 3 * x + 6.
-/
theorem not_equivalent_expression (x : ℝ) :
  let A := 3 * (x + 2)
  let B := (-9 * x - 18) / -3
  let C := (1/3) * (3 * x) + (2/3) * 9
  let D := (1/3) * (9 * x + 18)
  C ≠ 3 * x + 6 :=
by
  intros A B C D
  sorry

end NUMINAMATH_GPT_not_equivalent_expression_l1812_181211


namespace NUMINAMATH_GPT_range_of_m_l1812_181281

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, e ≤ x ∧ x ≤ e^2 ∧ f x - m * x - 1/2 + m ≤ 0) →
  1/2 ≤ m := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1812_181281


namespace NUMINAMATH_GPT_ratio_of_rectangles_l1812_181262

noncomputable def rect_ratio (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : ℝ :=
  let A_A := a * b
  let A_B := (a * 5 / 3) * (b * 5 / 3)
  let A_C := (a * 4 / 7) * (b * 4 / 7)
  let A_BC := A_B + A_C
  A_A / A_BC

theorem ratio_of_rectangles (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : 
  rect_ratio a b c d e f h1 h2 h3 h4 = 441 / 1369 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_rectangles_l1812_181262


namespace NUMINAMATH_GPT_matches_length_l1812_181201

-- Definitions and conditions
def area_shaded_figure : ℝ := 300 -- given in cm^2
def num_small_squares : ℕ := 8
def large_square_area_coefficient : ℕ := 4
def area_small_square (a : ℝ) : ℝ := num_small_squares * a + large_square_area_coefficient * a

-- Question and answer to be proven
theorem matches_length (a : ℝ) (side_length: ℝ) :
  area_shaded_figure = 300 → 
  area_small_square a = area_shaded_figure →
  (a = 25) →
  (side_length = 5) →
  4 * 7 * side_length = 140 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_matches_length_l1812_181201


namespace NUMINAMATH_GPT_prime_condition_l1812_181226

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_condition (p : ℕ) (h1 : is_prime p) (h2 : is_prime (8 * p^2 + 1)) : 
  p = 3 ∧ is_prime (8 * p^2 - p + 2) :=
by
  sorry

end NUMINAMATH_GPT_prime_condition_l1812_181226


namespace NUMINAMATH_GPT_solution_to_diff_eq_l1812_181233

def y (x C : ℝ) : ℝ := x^2 + x + C

theorem solution_to_diff_eq (C : ℝ) : ∀ x : ℝ, 
  (dy = (2 * x + 1) * dx) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_diff_eq_l1812_181233


namespace NUMINAMATH_GPT_average_weight_l1812_181296

def weights (A B C : ℝ) : Prop :=
  (A + B + C = 135) ∧
  (B + C = 86) ∧
  (B = 31)

theorem average_weight (A B C : ℝ) (h : weights A B C) :
  (A + B) / 2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_l1812_181296


namespace NUMINAMATH_GPT_greatest_integer_value_l1812_181243

theorem greatest_integer_value (x : ℤ) : 7 - 3 * x > 20 → x ≤ -5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_greatest_integer_value_l1812_181243


namespace NUMINAMATH_GPT_percentage_of_september_authors_l1812_181252

def total_authors : ℕ := 120
def september_authors : ℕ := 15

theorem percentage_of_september_authors : 
  (september_authors / total_authors : ℚ) * 100 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_september_authors_l1812_181252


namespace NUMINAMATH_GPT_normal_line_equation_at_x0_l1812_181240

def curve (x : ℝ) : ℝ := x - x^3
noncomputable def x0 : ℝ := -1
noncomputable def y0 : ℝ := curve x0

theorem normal_line_equation_at_x0 :
  ∀ (y : ℝ), y = (1/2 : ℝ) * x + 1/2 ↔ (∃ (x : ℝ), y = curve x ∧ x = x0) :=
by
  sorry

end NUMINAMATH_GPT_normal_line_equation_at_x0_l1812_181240


namespace NUMINAMATH_GPT_range_of_a_l1812_181291

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 4 → |x - 1| < a) ↔ 3 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1812_181291


namespace NUMINAMATH_GPT_product_p_yi_eq_neg26_l1812_181205

-- Definitions of the polynomials h and p.
def h (y : ℂ) : ℂ := y^3 - 3 * y + 1
def p (y : ℂ) : ℂ := y^3 + 2

-- Given that y1, y2, y3 are roots of h(y)
variables (y1 y2 y3 : ℂ) (H1 : h y1 = 0) (H2 : h y2 = 0) (H3 : h y3 = 0)

-- State the theorem to show p(y1) * p(y2) * p(y3) = -26
theorem product_p_yi_eq_neg26 : p y1 * p y2 * p y3 = -26 :=
sorry

end NUMINAMATH_GPT_product_p_yi_eq_neg26_l1812_181205


namespace NUMINAMATH_GPT_expression_equality_l1812_181257

theorem expression_equality :
  - (2^3) = (-2)^3 :=
by sorry

end NUMINAMATH_GPT_expression_equality_l1812_181257


namespace NUMINAMATH_GPT_linear_dependent_vectors_l1812_181216

variable (m : ℝ) (a b : ℝ) 

theorem linear_dependent_vectors :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨5, m⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) ↔ m = 15 / 2 :=
sorry

end NUMINAMATH_GPT_linear_dependent_vectors_l1812_181216


namespace NUMINAMATH_GPT_total_students_count_l1812_181220

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) : Prop := g * 4 = b * 3
def boys_count : ℕ := 28

-- Theorem to prove the total number of students
theorem total_students_count {g : ℕ} (h : ratio_girls_to_boys g boys_count) : g + boys_count = 49 :=
sorry

end NUMINAMATH_GPT_total_students_count_l1812_181220


namespace NUMINAMATH_GPT_infinitely_many_composite_numbers_l1812_181285

-- We define n in a specialized form.
def n (m : ℕ) : ℕ := (3 * m) ^ 3

-- We state that m is an odd positive integer.
def odd_positive_integer (m : ℕ) : Prop := m > 0 ∧ (m % 2 = 1)

-- The main statement: for infinitely many odd values of n, 2^n + n - 1 is composite.
theorem infinitely_many_composite_numbers : 
  ∃ (m : ℕ), odd_positive_integer m ∧ Nat.Prime (n m) ∧ ∃ d : ℕ, d > 1 ∧ d < n m ∧ (2^(n m) + n m - 1) % d = 0 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_composite_numbers_l1812_181285


namespace NUMINAMATH_GPT_mirror_full_body_view_l1812_181249

theorem mirror_full_body_view (AB MN : ℝ) (h : AB > 0): 
  (MN = 1/2 * AB) ↔
  ∀ (P : ℝ), (0 < P) → (P < AB) → 
    (P < MN + (AB - P)) ∧ (P > AB - MN + P) := 
by
  sorry

end NUMINAMATH_GPT_mirror_full_body_view_l1812_181249


namespace NUMINAMATH_GPT_count_even_three_digit_numbers_sum_tens_units_eq_12_l1812_181219

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end NUMINAMATH_GPT_count_even_three_digit_numbers_sum_tens_units_eq_12_l1812_181219


namespace NUMINAMATH_GPT_constant_two_l1812_181239

theorem constant_two (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) (c : ℕ) (n : ℕ) (h_n : n = c * p) (h_even_divisors : ∀ d : ℕ, d ∣ n → (d % 2 = 0) → d = 2) : c = 2 := by
  sorry

end NUMINAMATH_GPT_constant_two_l1812_181239


namespace NUMINAMATH_GPT_sqrt_x_minus_1_meaningful_l1812_181298

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_1_meaningful_l1812_181298
