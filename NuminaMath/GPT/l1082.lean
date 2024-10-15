import Mathlib

namespace NUMINAMATH_GPT_Charlie_age_when_Jenny_twice_as_Bobby_l1082_108238

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end NUMINAMATH_GPT_Charlie_age_when_Jenny_twice_as_Bobby_l1082_108238


namespace NUMINAMATH_GPT_least_k_inequality_l1082_108251

theorem least_k_inequality :
  ∃ k : ℝ, (∀ a b c : ℝ, 
    ((2 * a / (a - b)) ^ 2 + (2 * b / (b - c)) ^ 2 + (2 * c / (c - a)) ^ 2 + k 
    ≥ 4 * (2 * a / (a - b) + 2 * b / (b - c) + 2 * c / (c - a)))) ∧ k = 8 :=
by
  sorry  -- proof is omitted

end NUMINAMATH_GPT_least_k_inequality_l1082_108251


namespace NUMINAMATH_GPT_quadratic_root_l1082_108200

/-- If one root of the quadratic equation x^2 - 2x + n = 0 is 3, then n is -3. -/
theorem quadratic_root (n : ℝ) (h : (3 : ℝ)^2 - 2 * 3 + n = 0) : n = -3 :=
sorry

end NUMINAMATH_GPT_quadratic_root_l1082_108200


namespace NUMINAMATH_GPT_gcd_of_three_digit_palindromes_l1082_108272

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 101 * a + 10 * b

theorem gcd_of_three_digit_palindromes :
  ∀ n, is_palindrome n → Nat.gcd n 1 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_of_three_digit_palindromes_l1082_108272


namespace NUMINAMATH_GPT_fraction_inequality_l1082_108205

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end NUMINAMATH_GPT_fraction_inequality_l1082_108205


namespace NUMINAMATH_GPT_remaining_pie_l1082_108208

theorem remaining_pie (carlos_take: ℝ) (sophia_share : ℝ) (final_remaining : ℝ) :
  carlos_take = 0.6 ∧ sophia_share = (1 - carlos_take) / 4 ∧ final_remaining = (1 - carlos_take) - sophia_share →
  final_remaining = 0.3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_remaining_pie_l1082_108208


namespace NUMINAMATH_GPT_largest_divisor_of_even_square_difference_l1082_108240

theorem largest_divisor_of_even_square_difference (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) :
  ∃ (k : ℕ), k = 8 ∧ ∀ m n : ℕ, m % 2 = 0 → n % 2 = 0 → n < m → k ∣ (m^2 - n^2) := by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_even_square_difference_l1082_108240


namespace NUMINAMATH_GPT_clock_shows_l1082_108244

-- Definitions for the hands and their positions
variables {A B C : ℕ} -- Representing hands A, B, and C as natural numbers for simplicity

-- Conditions based on the problem description:
-- 1. Hands A and B point exactly at the hour markers.
-- 2. Hand C is slightly off from an hour marker.
axiom hand_A_hour_marker : A % 12 = A
axiom hand_B_hour_marker : B % 12 = B
axiom hand_C_slightly_off : C % 12 ≠ C

-- Theorem stating that given these conditions, the clock shows the time 4:50
theorem clock_shows (h1: A % 12 = A) (h2: B % 12 = B) (h3: C % 12 ≠ C) : A = 50 ∧ B = 12 ∧ C = 4 :=
sorry

end NUMINAMATH_GPT_clock_shows_l1082_108244


namespace NUMINAMATH_GPT_run_time_is_48_minutes_l1082_108264

noncomputable def cycling_speed : ℚ := 5 / 2
noncomputable def running_speed : ℚ := cycling_speed * 0.5
noncomputable def walking_speed : ℚ := running_speed * 0.5

theorem run_time_is_48_minutes (d : ℚ) (h : (d / cycling_speed) + (d / walking_speed) = 2) : 
  (60 * d / running_speed) = 48 :=
by
  sorry

end NUMINAMATH_GPT_run_time_is_48_minutes_l1082_108264


namespace NUMINAMATH_GPT_remainder_division_l1082_108213

theorem remainder_division (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_division_l1082_108213


namespace NUMINAMATH_GPT_caterer_min_people_l1082_108260

theorem caterer_min_people (x : ℕ) : 150 + 18 * x > 250 + 15 * x → x ≥ 34 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_caterer_min_people_l1082_108260


namespace NUMINAMATH_GPT_maximum_k_l1082_108250

theorem maximum_k (m k : ℝ) (h0 : 0 < m) (h1 : m < 1/2) (h2 : (1/m + 2/(1-2*m)) ≥ k): k ≤ 8 :=
sorry

end NUMINAMATH_GPT_maximum_k_l1082_108250


namespace NUMINAMATH_GPT_y_coords_diff_of_ellipse_incircle_area_l1082_108255

theorem y_coords_diff_of_ellipse_incircle_area
  (x1 y1 x2 y2 : ℝ)
  (F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : F1 = (-4, 0))
  (h4 : F2 = (4, 0))
  (h5 : 4 * (|y1 - y2|) = 20)
  (h6 : ∃ (x : ℝ), (x / 25)^2 + (y1 / 9)^2 = 1 ∧ (x / 25)^2 + (y2 / 9)^2 = 1) :
  |y1 - y2| = 5 :=
sorry

end NUMINAMATH_GPT_y_coords_diff_of_ellipse_incircle_area_l1082_108255


namespace NUMINAMATH_GPT_quadratic_min_value_l1082_108224

theorem quadratic_min_value (p q : ℝ) (h : ∀ x : ℝ, 3 * x^2 + p * x + q ≥ 4) : q = p^2 / 12 + 4 :=
sorry

end NUMINAMATH_GPT_quadratic_min_value_l1082_108224


namespace NUMINAMATH_GPT_third_term_of_arithmetic_sequence_is_negative_22_l1082_108232

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem third_term_of_arithmetic_sequence_is_negative_22
  (a d : ℤ)
  (H1 : arithmetic_sequence a d 14 = 14)
  (H2 : arithmetic_sequence a d 15 = 17) :
  arithmetic_sequence a d 2 = -22 :=
sorry

end NUMINAMATH_GPT_third_term_of_arithmetic_sequence_is_negative_22_l1082_108232


namespace NUMINAMATH_GPT_apples_difference_l1082_108269

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_apples_difference_l1082_108269


namespace NUMINAMATH_GPT_find_geometric_sequence_values_l1082_108273

theorem find_geometric_sequence_values :
  ∃ (a b c : ℤ), (∃ q : ℤ, q ≠ 0 ∧ 2 * q ^ 4 = 32 ∧ a = 2 * q ∧ b = 2 * q ^ 2 ∧ c = 2 * q ^ 3)
                 ↔ ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end NUMINAMATH_GPT_find_geometric_sequence_values_l1082_108273


namespace NUMINAMATH_GPT_unique_b_positive_solution_l1082_108297

theorem unique_b_positive_solution (c : ℝ) (h : c ≠ 0) : 
  (∃ b : ℝ, b > 0 ∧ ∀ b : ℝ, b ≠ 0 → 
    ∀ x : ℝ, x^2 + (b + 1 / b) * x + c = 0 → x = - (b + 1 / b) / 2) 
  ↔ c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_b_positive_solution_l1082_108297


namespace NUMINAMATH_GPT_selling_price_is_correct_l1082_108220

-- Definitions of the given conditions

def cost_of_string_per_bracelet := 1
def cost_of_beads_per_bracelet := 3
def number_of_bracelets_sold := 25
def total_profit := 50

def cost_of_bracelet := cost_of_string_per_bracelet + cost_of_beads_per_bracelet
def total_cost := cost_of_bracelet * number_of_bracelets_sold
def total_revenue := total_profit + total_cost
def selling_price_per_bracelet := total_revenue / number_of_bracelets_sold

-- Target theorem
theorem selling_price_is_correct : selling_price_per_bracelet = 6 :=
  by
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l1082_108220


namespace NUMINAMATH_GPT_theater_seats_l1082_108263

theorem theater_seats (x y t : ℕ) (h1 : x = 532) (h2 : y = 218) (h3 : t = x + y) : t = 750 := 
by 
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_theater_seats_l1082_108263


namespace NUMINAMATH_GPT_car_highway_miles_per_tankful_l1082_108243

-- Defining conditions as per given problem
def city_miles_per_tank : ℕ := 336
def city_miles_per_gallon : ℕ := 8
def difference_miles_per_gallon : ℕ := 3
def highway_miles_per_gallon := city_miles_per_gallon + difference_miles_per_gallon
def tank_size := city_miles_per_tank / city_miles_per_gallon
def highway_miles_per_tank := highway_miles_per_gallon * tank_size

-- Theorem statement to prove
theorem car_highway_miles_per_tankful :
  highway_miles_per_tank = 462 :=
sorry

end NUMINAMATH_GPT_car_highway_miles_per_tankful_l1082_108243


namespace NUMINAMATH_GPT_choose_three_cards_of_different_suits_l1082_108241

/-- The number of ways to choose 3 cards from a standard deck of 52 cards,
if all three cards must be of different suits -/
theorem choose_three_cards_of_different_suits :
  let n := 4
  let r := 3
  let suits_combinations := Nat.choose n r
  let cards_per_suit := 13
  let total_ways := suits_combinations * (cards_per_suit ^ r)
  total_ways = 8788 :=
by
  sorry

end NUMINAMATH_GPT_choose_three_cards_of_different_suits_l1082_108241


namespace NUMINAMATH_GPT_todd_money_left_l1082_108217

def candy_bar_cost : ℝ := 2.50
def chewing_gum_cost : ℝ := 1.50
def soda_cost : ℝ := 3
def discount : ℝ := 0.20
def initial_money : ℝ := 50
def number_of_candy_bars : ℕ := 7
def number_of_chewing_gum : ℕ := 5
def number_of_soda : ℕ := 3

noncomputable def total_candy_bar_cost : ℝ := number_of_candy_bars * candy_bar_cost
noncomputable def total_chewing_gum_cost : ℝ := number_of_chewing_gum * chewing_gum_cost
noncomputable def total_soda_cost : ℝ := number_of_soda * soda_cost
noncomputable def discount_amount : ℝ := total_soda_cost * discount
noncomputable def discounted_soda_cost : ℝ := total_soda_cost - discount_amount
noncomputable def total_cost : ℝ := total_candy_bar_cost + total_chewing_gum_cost + discounted_soda_cost
noncomputable def money_left : ℝ := initial_money - total_cost

theorem todd_money_left : money_left = 17.80 :=
by sorry

end NUMINAMATH_GPT_todd_money_left_l1082_108217


namespace NUMINAMATH_GPT_range_of_m_for_decreasing_interval_l1082_108282

def function_monotonically_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def f (x : ℝ) : ℝ := x ^ 3 - 12 * x

theorem range_of_m_for_decreasing_interval :
  ∀ m : ℝ, function_monotonically_decreasing_in_interval f (2 * m) (m + 1) → -1 ≤ m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_decreasing_interval_l1082_108282


namespace NUMINAMATH_GPT_largest_k_divides_2_pow_3_pow_m_add_1_l1082_108212

theorem largest_k_divides_2_pow_3_pow_m_add_1 (m : ℕ) : 9 ∣ 2^(3^m) + 1 := sorry

end NUMINAMATH_GPT_largest_k_divides_2_pow_3_pow_m_add_1_l1082_108212


namespace NUMINAMATH_GPT_scientific_notation_000073_l1082_108286

theorem scientific_notation_000073 : 0.000073 = 7.3 * 10^(-5) := by
  sorry

end NUMINAMATH_GPT_scientific_notation_000073_l1082_108286


namespace NUMINAMATH_GPT_find_angle_AOD_l1082_108215

noncomputable def angleAOD (x : ℝ) : ℝ :=
4 * x

theorem find_angle_AOD (x : ℝ) (h1 : 4 * x = 180) : angleAOD x = 135 :=
by
  -- x = 45
  have h2 : x = 45 := by linarith

  -- angleAOD 45 = 4 * 45 = 135
  rw [angleAOD, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_find_angle_AOD_l1082_108215


namespace NUMINAMATH_GPT_no_equal_numbers_from_19_and_98_l1082_108267

theorem no_equal_numbers_from_19_and_98 :
  ¬ (∃ s : ℕ, ∃ (a b : ℕ → ℕ), 
       (a 0 = 19) ∧ (b 0 = 98) ∧
       (∀ k, a (k + 1) = a k * a k ∨ a (k + 1) = a k + 1) ∧
       (∀ k, b (k + 1) = b k * b k ∨ b (k + 1) = b k + 1) ∧
       a s = b s) :=
sorry

end NUMINAMATH_GPT_no_equal_numbers_from_19_and_98_l1082_108267


namespace NUMINAMATH_GPT_range_tan_squared_plus_tan_plus_one_l1082_108222

theorem range_tan_squared_plus_tan_plus_one :
  (∀ y, ∃ x : ℝ, x ≠ (k : ℤ) * Real.pi + Real.pi / 2 → y = Real.tan x ^ 2 + Real.tan x + 1) ↔ 
  ∀ y, y ∈ Set.Ici (3 / 4) :=
sorry

end NUMINAMATH_GPT_range_tan_squared_plus_tan_plus_one_l1082_108222


namespace NUMINAMATH_GPT_annual_growth_rate_l1082_108206

theorem annual_growth_rate (x : ℝ) (h : 2000 * (1 + x) ^ 2 = 2880) : x = 0.2 :=
by sorry

end NUMINAMATH_GPT_annual_growth_rate_l1082_108206


namespace NUMINAMATH_GPT_polar_to_rectangular_coordinates_l1082_108252

theorem polar_to_rectangular_coordinates (r θ : ℝ) (hr : r = 5) (hθ : θ = (3 * Real.pi) / 2) :
    (r * Real.cos θ, r * Real.sin θ) = (0, -5) :=
by
  rw [hr, hθ]
  simp [Real.cos, Real.sin]
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_coordinates_l1082_108252


namespace NUMINAMATH_GPT_intersection_complement_l1082_108248

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | x^2 - 2 * x > 0 }

theorem intersection_complement (A B : Set ℝ) : 
  (A ∩ (U \ B)) = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1082_108248


namespace NUMINAMATH_GPT_no_eight_roots_for_nested_quadratics_l1082_108249

theorem no_eight_roots_for_nested_quadratics
  (f g h : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e k : ℝ, ∀ x, g x = d * x^2 + e * x + k)
  (hh : ∃ p q r : ℝ, ∀ x, h x = p * x^2 + q * x + r)
  (hroots : ∀ x, f (g (h x)) = 0 → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8)) :
  false :=
by
  sorry

end NUMINAMATH_GPT_no_eight_roots_for_nested_quadratics_l1082_108249


namespace NUMINAMATH_GPT_ferris_wheel_cost_per_child_l1082_108293

namespace AmusementPark

def num_children := 5
def daring_children := 3
def merry_go_round_cost_per_child := 3
def ice_cream_cones_per_child := 2
def ice_cream_cost_per_cone := 8
def total_spent := 110

theorem ferris_wheel_cost_per_child (F : ℝ) :
  (daring_children * F + num_children * merry_go_round_cost_per_child +
   num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone = total_spent) →
  F = 5 :=
by
  -- Here we would proceed with the proof steps, but adding sorry to skip it.
  sorry

end AmusementPark

end NUMINAMATH_GPT_ferris_wheel_cost_per_child_l1082_108293


namespace NUMINAMATH_GPT_cubic_expression_l1082_108204

theorem cubic_expression (x : ℝ) (hx : x + 1/x = -7) : x^3 + 1/x^3 = -322 :=
by sorry

end NUMINAMATH_GPT_cubic_expression_l1082_108204


namespace NUMINAMATH_GPT_seating_arrangements_count_is_134_l1082_108201

theorem seating_arrangements_count_is_134 (front_row_seats : ℕ) (back_row_seats : ℕ) (valid_arrangements_with_no_next_to_each_other : ℕ) : 
  front_row_seats = 6 → back_row_seats = 7 → valid_arrangements_with_no_next_to_each_other = 134 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_seating_arrangements_count_is_134_l1082_108201


namespace NUMINAMATH_GPT_four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l1082_108223

noncomputable def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

noncomputable def sum_of_digits_is_11 (n : ℕ) : Prop := 
  let d1 := n / 1000
  let r1 := n % 1000
  let d2 := r1 / 100
  let r2 := r1 % 100
  let d3 := r2 / 10
  let d4 := r2 % 10
  d1 + d2 + d3 + d4 = 11

theorem four_digit_numbers_divisible_by_11_with_sum_of_digits_11
  (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : is_divisible_by_11 n)
  (h3 : sum_of_digits_is_11 n) : 
  n = 2090 ∨ n = 3080 ∨ n = 4070 ∨ n = 5060 ∨ n = 6050 ∨ n = 7040 ∨ n = 8030 ∨ n = 9020 :=
sorry

end NUMINAMATH_GPT_four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l1082_108223


namespace NUMINAMATH_GPT_angle_BMC_not_obtuse_angle_BAC_is_120_l1082_108221

theorem angle_BMC_not_obtuse (α β γ : ℝ) (h : α + β + γ = 180) :
  0 < 90 - α / 2 ∧ 90 - α / 2 < 90 :=
sorry

theorem angle_BAC_is_120 (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : 90 - α / 2 = α / 2) : α = 120 :=
sorry

end NUMINAMATH_GPT_angle_BMC_not_obtuse_angle_BAC_is_120_l1082_108221


namespace NUMINAMATH_GPT_winnie_servings_l1082_108276

theorem winnie_servings:
  ∀ (x : ℝ), 
  (2 / 5) * x + (21 / 25) * x = 82 →
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_winnie_servings_l1082_108276


namespace NUMINAMATH_GPT_mixture_replacement_l1082_108229

theorem mixture_replacement (A B T x : ℝ)
  (h1 : A / (A + B) = 7 / 12)
  (h2 : A = 21)
  (h3 : (A / (B + x)) = 7 / 9) :
  x = 12 :=
by
  sorry

end NUMINAMATH_GPT_mixture_replacement_l1082_108229


namespace NUMINAMATH_GPT_Tim_paid_amount_l1082_108245

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end NUMINAMATH_GPT_Tim_paid_amount_l1082_108245


namespace NUMINAMATH_GPT_square_remainder_is_square_l1082_108214

theorem square_remainder_is_square (a : ℤ) : ∃ b : ℕ, (a^2 % 16 = b) ∧ (∃ c : ℕ, b = c^2) :=
by
  sorry

end NUMINAMATH_GPT_square_remainder_is_square_l1082_108214


namespace NUMINAMATH_GPT_squares_triangles_product_l1082_108247

theorem squares_triangles_product :
  let S := 7
  let T := 10
  S * T = 70 :=
by
  let S := 7
  let T := 10
  show (S * T = 70)
  sorry

end NUMINAMATH_GPT_squares_triangles_product_l1082_108247


namespace NUMINAMATH_GPT_find_real_number_l1082_108281

theorem find_real_number :
    (∃ y : ℝ, y = 3 + (5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + sorry)))))))))) ∧ 
    y = (3 + Real.sqrt 29) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_real_number_l1082_108281


namespace NUMINAMATH_GPT_younger_person_age_l1082_108275

theorem younger_person_age 
  (y e : ℕ)
  (h1 : e = y + 20)
  (h2 : e - 4 = 5 * (y - 4)) : 
  y = 9 := 
sorry

end NUMINAMATH_GPT_younger_person_age_l1082_108275


namespace NUMINAMATH_GPT_cylinder_surface_area_l1082_108210

/-- A right cylinder with radius 3 inches and height twice the radius has a total surface area of 54π square inches. -/
theorem cylinder_surface_area (r : ℝ) (h : ℝ) (A_total : ℝ) (π : ℝ) : r = 3 → h = 2 * r → π = Real.pi → A_total = 54 * π :=
by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1082_108210


namespace NUMINAMATH_GPT_find_m_l1082_108253

def g (n : Int) : Int :=
  if n % 2 ≠ 0 then n + 5 else 
  if n % 3 = 0 then n / 3 else n

theorem find_m (m : Int) 
  (h_odd : m % 2 ≠ 0) 
  (h_ggg : g (g (g m)) = 35) : 
  m = 85 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1082_108253


namespace NUMINAMATH_GPT_instantaneous_velocity_at_2_l1082_108225

def displacement (t : ℝ) : ℝ := 14 * t - t^2 

def velocity (t : ℝ) : ℝ :=
  sorry -- The velocity function which is the derivative of displacement

theorem instantaneous_velocity_at_2 :
  velocity 2 = 10 := 
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_2_l1082_108225


namespace NUMINAMATH_GPT_find_x_l1082_108258

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
sorry

end NUMINAMATH_GPT_find_x_l1082_108258


namespace NUMINAMATH_GPT_find_c_l1082_108298

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Iio (-2) ∪ Set.Ioi 3 → x^2 - c * x + 6 > 0) → c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1082_108298


namespace NUMINAMATH_GPT_gemstones_needed_l1082_108211

noncomputable def magnets_per_earring : ℕ := 2

noncomputable def buttons_per_earring : ℕ := magnets_per_earring / 2

noncomputable def gemstones_per_earring : ℕ := 3 * buttons_per_earring

noncomputable def sets_of_earrings : ℕ := 4

noncomputable def earrings_per_set : ℕ := 2

noncomputable def total_gemstones : ℕ := sets_of_earrings * earrings_per_set * gemstones_per_earring

theorem gemstones_needed :
  total_gemstones = 24 :=
  by
    sorry

end NUMINAMATH_GPT_gemstones_needed_l1082_108211


namespace NUMINAMATH_GPT_smallest_five_digit_divisible_by_15_32_54_l1082_108290

theorem smallest_five_digit_divisible_by_15_32_54 : 
  ∃ n : ℤ, n >= 10000 ∧ n < 100000 ∧ (15 ∣ n) ∧ (32 ∣ n) ∧ (54 ∣ n) ∧ n = 17280 :=
  sorry

end NUMINAMATH_GPT_smallest_five_digit_divisible_by_15_32_54_l1082_108290


namespace NUMINAMATH_GPT_simplify_expression_l1082_108209

theorem simplify_expression :
  (∃ (a b c d e f : ℝ), 
    a = (7)^(1/4) ∧ 
    b = (3)^(1/3) ∧ 
    c = (7)^(1/2) ∧ 
    d = (3)^(1/6) ∧ 
    e = (a / b) / (c / d) ∧ 
    f = ((1 / 7)^(1/4)) * ((1 / 3)^(1/6))
    → e = f) :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l1082_108209


namespace NUMINAMATH_GPT_roots_squared_sum_l1082_108287

theorem roots_squared_sum (x1 x2 : ℝ) (h₁ : x1^2 - 5 * x1 + 3 = 0) (h₂ : x2^2 - 5 * x2 + 3 = 0) :
  x1^2 + x2^2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_roots_squared_sum_l1082_108287


namespace NUMINAMATH_GPT_five_letter_sequences_l1082_108207

-- Define the quantities of each vowel.
def quantity_vowel_A : Nat := 3
def quantity_vowel_E : Nat := 4
def quantity_vowel_I : Nat := 5
def quantity_vowel_O : Nat := 6
def quantity_vowel_U : Nat := 7

-- Define the number of choices for each letter in a five-letter sequence.
def choices_per_letter : Nat := 5

-- Define the total number of five-letter sequences.
noncomputable def total_sequences : Nat := choices_per_letter ^ 5

-- Prove that the number of five-letter sequences is 3125.
theorem five_letter_sequences : total_sequences = 3125 :=
by sorry

end NUMINAMATH_GPT_five_letter_sequences_l1082_108207


namespace NUMINAMATH_GPT_shirt_cost_l1082_108219

-- Definitions and conditions
def num_ten_bills : ℕ := 2
def num_twenty_bills : ℕ := num_ten_bills + 1

def ten_bill_value : ℕ := 10
def twenty_bill_value : ℕ := 20

-- Statement to prove
theorem shirt_cost :
  (num_ten_bills * ten_bill_value) + (num_twenty_bills * twenty_bill_value) = 80 :=
by
  sorry

end NUMINAMATH_GPT_shirt_cost_l1082_108219


namespace NUMINAMATH_GPT_scientific_notation_of_425000_l1082_108288

def scientific_notation (x : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_425000 :
  scientific_notation 425000 = (4.25, 5) := sorry

end NUMINAMATH_GPT_scientific_notation_of_425000_l1082_108288


namespace NUMINAMATH_GPT_leak_empty_time_l1082_108218

theorem leak_empty_time
  (R : ℝ) (L : ℝ)
  (hR : R = 1 / 8)
  (hRL : R - L = 1 / 10) :
  1 / L = 40 :=
by
  sorry

end NUMINAMATH_GPT_leak_empty_time_l1082_108218


namespace NUMINAMATH_GPT_rectangle_length_l1082_108262

variable (w l : ℝ)

def perimeter (w l : ℝ) : ℝ := 2 * w + 2 * l

theorem rectangle_length (h1 : l = w + 2) (h2 : perimeter w l = 20) : l = 6 :=
by sorry

end NUMINAMATH_GPT_rectangle_length_l1082_108262


namespace NUMINAMATH_GPT_evelyn_total_marbles_l1082_108228

def initial_marbles := 95
def marbles_from_henry := 9
def marbles_from_grace := 12
def number_of_cards := 6
def marbles_per_card := 4

theorem evelyn_total_marbles :
  initial_marbles + marbles_from_henry + marbles_from_grace + number_of_cards * marbles_per_card = 140 := 
by 
  sorry

end NUMINAMATH_GPT_evelyn_total_marbles_l1082_108228


namespace NUMINAMATH_GPT_complex_simplify_l1082_108268

theorem complex_simplify :
  10.25 * Real.sqrt 6 * Complex.exp (Complex.I * 160 * Real.pi / 180)
  / (Real.sqrt 3 * Complex.exp (Complex.I * 40 * Real.pi / 180))
  = (-Real.sqrt 2 / 2) + Complex.I * (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_GPT_complex_simplify_l1082_108268


namespace NUMINAMATH_GPT_johns_minutes_billed_l1082_108279

theorem johns_minutes_billed 
  (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 5) (h2 : cost_per_minute = 0.25) (h3 : total_bill = 12.02) :
  ⌊(total_bill - monthly_fee) / cost_per_minute⌋ = 28 :=
by
  sorry

end NUMINAMATH_GPT_johns_minutes_billed_l1082_108279


namespace NUMINAMATH_GPT_foci_and_directrices_of_ellipse_l1082_108265

noncomputable def parametricEllipse
    (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ + 1, 4 * Real.sin θ)

theorem foci_and_directrices_of_ellipse :
  (∀ θ : ℝ, parametricEllipse θ = (x, y)) →
  (∃ (f1 f2 : ℝ × ℝ) (d1 d2 : ℝ → Prop),
    f1 = (1, Real.sqrt 7) ∧
    f2 = (1, -Real.sqrt 7) ∧
    d1 = fun x => x = 1 + 9 / Real.sqrt 7 ∧
    d2 = fun x => x = 1 - 9 / Real.sqrt 7) := sorry

end NUMINAMATH_GPT_foci_and_directrices_of_ellipse_l1082_108265


namespace NUMINAMATH_GPT_part1_part2_l1082_108259

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (1 + x)

theorem part1 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : f x ≥ 1 - x + x^2 := sorry

theorem part2 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : 3 / 4 < f x ∧ f x ≤ 3 / 2 := sorry

end NUMINAMATH_GPT_part1_part2_l1082_108259


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1082_108283

def f (x : ℝ) := x^2
def g (x : ℝ) := 3 * x - 8
def h (r : ℝ) (x : ℝ) := 3 * x - r

theorem part_a :
  f 2 = 4 ∧ g (f 2) = 4 :=
by {
  sorry
}

theorem part_b :
  ∀ x : ℝ, f (g x) = g (f x) → (x = 2 ∨ x = 6) :=
by {
  sorry
}

theorem part_c :
  ∀ r : ℝ, f (h r 2) = h r (f 2) → (r = 3 ∨ r = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_part_a_part_b_part_c_l1082_108283


namespace NUMINAMATH_GPT_range_f_neg2_l1082_108277

noncomputable def f (a b x : ℝ): ℝ := a * x^2 + b * x

theorem range_f_neg2 (a b : ℝ) (h1 : 1 ≤ f a b (-1)) (h2 : f a b (-1) ≤ 2)
  (h3 : 3 ≤ f a b 1) (h4 : f a b 1 ≤ 4) : 6 ≤ f a b (-2) ∧ f a b (-2) ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_range_f_neg2_l1082_108277


namespace NUMINAMATH_GPT_divisibility_of_f_by_cubic_factor_l1082_108291

noncomputable def f (x : ℂ) (m n : ℕ) : ℂ := x^(3 * m + 2) + (-x^2 - 1)^(3 * n + 1) + 1

theorem divisibility_of_f_by_cubic_factor (m n : ℕ) : ∀ x : ℂ, x^2 + x + 1 = 0 → f x m n = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisibility_of_f_by_cubic_factor_l1082_108291


namespace NUMINAMATH_GPT_fraction_equals_seven_twentyfive_l1082_108233

theorem fraction_equals_seven_twentyfive :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = (7 / 25) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equals_seven_twentyfive_l1082_108233


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1082_108254

open Real

theorem min_value_of_x_plus_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0)
  (a : ℝ × ℝ := (1 - x, 4)) (b : ℝ × ℝ := (x, -y))
  (h₃ : ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)) :
  x + y = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1082_108254


namespace NUMINAMATH_GPT_combined_work_rate_l1082_108270

def work_done_in_one_day (A B : ℕ) (work_to_days : ℕ -> ℕ) : ℚ :=
  (work_to_days A + work_to_days B)

theorem combined_work_rate (A : ℕ) (B : ℕ) (work_to_days : ℕ -> ℕ) :
  work_to_days A = 1/18 ∧ work_to_days B = 1/9 → work_done_in_one_day A B (work_to_days) = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_combined_work_rate_l1082_108270


namespace NUMINAMATH_GPT_factor_between_l1082_108296

theorem factor_between (n a b : ℕ) (h1 : 10 < n) 
(h2 : n = a * a + b) 
(h3 : a ∣ n) 
(h4 : b ∣ n) 
(h5 : a ≠ b) 
(h6 : 1 < a) 
(h7 : 1 < b) : 
    ∃ m : ℕ, b = m * a ∧ 1 < m ∧ a < a + m ∧ a + m < b  :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_factor_between_l1082_108296


namespace NUMINAMATH_GPT_circle_equation_bisected_and_tangent_l1082_108202

theorem circle_equation_bisected_and_tangent :
  (∃ x0 y0 r : ℝ, x0 = y0 ∧ (x0 + y0 - 2 * r) = 0 ∧ (∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2)) := sorry

end NUMINAMATH_GPT_circle_equation_bisected_and_tangent_l1082_108202


namespace NUMINAMATH_GPT_number_of_children_l1082_108285

theorem number_of_children (total_oranges : ℕ) (oranges_per_child : ℕ) (h1 : oranges_per_child = 3) (h2 : total_oranges = 12) : total_oranges / oranges_per_child = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1082_108285


namespace NUMINAMATH_GPT_parabola_hyperbola_coincide_directrix_l1082_108284

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2
noncomputable def hyperbola_directrix : ℝ := -3 / 2

theorem parabola_hyperbola_coincide_directrix (p : ℝ) (hp : 0 < p) 
  (h_eq : parabola_directrix p = hyperbola_directrix) : p = 3 :=
by
  have hp_directrix : parabola_directrix p = -p / 2 := rfl
  have h_directrix : hyperbola_directrix = -3 / 2 := rfl
  rw [hp_directrix, h_directrix] at h_eq
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_coincide_directrix_l1082_108284


namespace NUMINAMATH_GPT_find_first_discount_l1082_108257

-- Definitions for the given conditions
def list_price : ℝ := 150
def final_price : ℝ := 105
def second_discount : ℝ := 12.5

-- Statement representing the mathematical proof problem
theorem find_first_discount (x : ℝ) : 
  list_price * ((100 - x) / 100) * ((100 - second_discount) / 100) = final_price → x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_first_discount_l1082_108257


namespace NUMINAMATH_GPT_counties_no_rain_l1082_108231

theorem counties_no_rain 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) :
  P_A = 0.7 → P_B = 0.5 → P_A_and_B = 0.4 →
  (1 - (P_A + P_B - P_A_and_B) = 0.2) :=
by intros h1 h2 h3; sorry

end NUMINAMATH_GPT_counties_no_rain_l1082_108231


namespace NUMINAMATH_GPT_find_b_l1082_108216

-- Definitions for the conditions
variables (a b c d : ℝ)
def four_segments_proportional := a / b = c / d

theorem find_b (h1: a = 3) (h2: d = 4) (h3: c = 6) (h4: four_segments_proportional a b c d) : b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1082_108216


namespace NUMINAMATH_GPT_number_of_slices_with_both_l1082_108235

def total_slices : ℕ := 20
def slices_with_pepperoni : ℕ := 12
def slices_with_mushrooms : ℕ := 14
def slices_with_both_toppings (n : ℕ) : Prop :=
  n + (slices_with_pepperoni - n) + (slices_with_mushrooms - n) = total_slices

theorem number_of_slices_with_both (n : ℕ) (h : slices_with_both_toppings n) : n = 6 :=
sorry

end NUMINAMATH_GPT_number_of_slices_with_both_l1082_108235


namespace NUMINAMATH_GPT_tom_calories_l1082_108236

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end NUMINAMATH_GPT_tom_calories_l1082_108236


namespace NUMINAMATH_GPT_ludvik_favorite_number_l1082_108227

variable (a b : ℕ)
variable (ℓ : ℝ)

theorem ludvik_favorite_number (h1 : 2 * a = (b + 12) * ℓ)
(h2 : a - 42 = (b / 2) * ℓ) : ℓ = 7 :=
sorry

end NUMINAMATH_GPT_ludvik_favorite_number_l1082_108227


namespace NUMINAMATH_GPT_find_number_l1082_108237

theorem find_number (x : ℝ) (h : 0.30 * x = 108.0) : x = 360 := 
sorry

end NUMINAMATH_GPT_find_number_l1082_108237


namespace NUMINAMATH_GPT_part1_part2_l1082_108239

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1082_108239


namespace NUMINAMATH_GPT_equation_of_ellipse_equation_of_line_AB_l1082_108278

-- Step 1: Given conditions for the ellipse and related hyperbola.
def condition_eccentricity (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ c / a = Real.sqrt 2 / 2

def condition_distance_focus_asymptote (c : ℝ) : Prop :=
  abs c / Real.sqrt (1 + 2) = Real.sqrt 3 / 3

-- Step 2: Given conditions for the line AB.
def condition_line_A_B (k m : ℝ) : Prop :=
  k < 0 ∧ m^2 = 4 / 5 * (1 + k^2) ∧
  ∃ (x1 x2 y1 y2 : ℝ), 
  (1 + 2 * k^2) * x1^2 + 4 * k * m * x1 + 2 * m^2 - 2 = 0 ∧ 
  (1 + 2 * k^2) * x2^2 + 4 * k * m * x2 + 2 * m^2 - 2 = 0 ∧
  x1 + x2 = -4 * k * m / (1 + 2*k^2) ∧ 
  x1 * x2 = (2 * m^2 - 2) / (1 + 2*k^2)

def condition_circle_passes_F2 (x1 x2 k m : ℝ) : Prop :=
  (1 + k^2) * x1 * x2 + (k * m - 1) * (x1 + x2) + m^2 + 1 = 0

noncomputable def problem_data : Prop :=
  ∃ (a b c k m x1 x2 : ℝ),
    condition_eccentricity a b c ∧
    condition_distance_focus_asymptote c ∧
    condition_line_A_B k m ∧
    condition_circle_passes_F2 x1 x2 k m

-- Step 3: Statements to be proven.
theorem equation_of_ellipse : problem_data → 
  ∃ (a b : ℝ), a = Real.sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) :=
by sorry

theorem equation_of_line_AB : problem_data → 
  ∃ (k m : ℝ), m = 1 ∧ k = -1/2 ∧ ∀ x y : ℝ, (y = k * x + m) ↔ (y = -0.5 * x + 1) :=
by sorry

end NUMINAMATH_GPT_equation_of_ellipse_equation_of_line_AB_l1082_108278


namespace NUMINAMATH_GPT_central_projection_intersect_l1082_108230

def central_projection (lines : Set (Set Point)) : Prop :=
  ∃ point : Point, ∀ line ∈ lines, line (point)

theorem central_projection_intersect :
  ∀ lines : Set (Set Point), central_projection lines → ∃ point : Point, ∀ line ∈ lines, line (point) :=
by
  sorry

end NUMINAMATH_GPT_central_projection_intersect_l1082_108230


namespace NUMINAMATH_GPT_slices_served_today_l1082_108234

-- Definitions based on conditions from part a)
def slices_lunch_today : ℕ := 7
def slices_dinner_today : ℕ := 5

-- Proof statement based on part c)
theorem slices_served_today : slices_lunch_today + slices_dinner_today = 12 := 
by
  sorry

end NUMINAMATH_GPT_slices_served_today_l1082_108234


namespace NUMINAMATH_GPT_distance_to_second_museum_l1082_108280

theorem distance_to_second_museum (d x : ℕ) (h1 : d = 5) (h2 : 2 * d + 2 * x = 40) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_second_museum_l1082_108280


namespace NUMINAMATH_GPT_ratio_of_average_speeds_l1082_108274

theorem ratio_of_average_speeds
    (time_eddy : ℝ) (distance_eddy : ℝ)
    (time_freddy : ℝ) (distance_freddy : ℝ) :
  time_eddy = 3 ∧ distance_eddy = 600 ∧ time_freddy = 4 ∧ distance_freddy = 460 →
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_average_speeds_l1082_108274


namespace NUMINAMATH_GPT_at_most_one_perfect_square_l1082_108256

theorem at_most_one_perfect_square (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n ^ 3 + 103) →
  (∃ n1, ∃ n2, a n1 = k1^2 ∧ a n2 = k2^2) → n1 = n2 
    ∨ (∀ n, a n ≠ k1^2) 
    ∨ (∀ n, a n ≠ k2^2) :=
sorry

end NUMINAMATH_GPT_at_most_one_perfect_square_l1082_108256


namespace NUMINAMATH_GPT_ratio_fraction_4A3B_5C2A_l1082_108294

def ratio (a b c : ℝ) := a / b = 3 / 2 ∧ b / c = 2 / 6 ∧ a / c = 3 / 6

theorem ratio_fraction_4A3B_5C2A (A B C : ℝ) (h : ratio A B C) : (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := 
  sorry

end NUMINAMATH_GPT_ratio_fraction_4A3B_5C2A_l1082_108294


namespace NUMINAMATH_GPT_fourth_vertex_of_tetrahedron_exists_l1082_108292

theorem fourth_vertex_of_tetrahedron_exists (x y z : ℤ) :
  (∃ (x y z : ℤ), 
     ((x - 1) ^ 2 + y ^ 2 + (z - 3) ^ 2 = 26) ∧ 
     ((x - 5) ^ 2 + (y - 3) ^ 2 + (z - 2) ^ 2 = 26) ∧ 
     ((x - 4) ^ 2 + y ^ 2 + (z - 6) ^ 2 = 26)) :=
sorry

end NUMINAMATH_GPT_fourth_vertex_of_tetrahedron_exists_l1082_108292


namespace NUMINAMATH_GPT_selling_price_750_max_daily_profit_l1082_108261

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 10) * (-10 * x + 300)

theorem selling_price_750 (x : ℝ) : profit x = 750 ↔ (x = 15 ∨ x = 25) :=
by sorry

theorem max_daily_profit : (∀ x : ℝ, profit x ≤ 1000) ∧ (profit 20 = 1000) :=
by sorry

end NUMINAMATH_GPT_selling_price_750_max_daily_profit_l1082_108261


namespace NUMINAMATH_GPT_rational_inequality_solution_l1082_108266

variable (x : ℝ)

def inequality_conditions : Prop := (2 * x - 1) / (x + 1) > 1

def inequality_solution : Prop := x < -1 ∨ x > 2

theorem rational_inequality_solution : inequality_conditions x → inequality_solution x :=
by
  sorry

end NUMINAMATH_GPT_rational_inequality_solution_l1082_108266


namespace NUMINAMATH_GPT_interior_angle_of_regular_hexagon_l1082_108242

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end NUMINAMATH_GPT_interior_angle_of_regular_hexagon_l1082_108242


namespace NUMINAMATH_GPT_fractions_of_120_equals_2_halves_l1082_108203

theorem fractions_of_120_equals_2_halves :
  (1 / 6) * (1 / 4) * (1 / 5) * 120 = 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_fractions_of_120_equals_2_halves_l1082_108203


namespace NUMINAMATH_GPT_ratio_of_weights_l1082_108289

def initial_weight : ℝ := 2
def weight_after_brownies (w : ℝ) : ℝ := w * 3
def weight_after_more_jelly_beans (w : ℝ) : ℝ := w + 2
def final_weight : ℝ := 16
def weight_before_adding_gummy_worms : ℝ := weight_after_more_jelly_beans (weight_after_brownies initial_weight)

theorem ratio_of_weights :
  final_weight / weight_before_adding_gummy_worms = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_weights_l1082_108289


namespace NUMINAMATH_GPT_find_positive_A_l1082_108299

theorem find_positive_A (A : ℕ) : (A^2 + 7^2 = 130) → A = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_positive_A_l1082_108299


namespace NUMINAMATH_GPT_simplify_expression_l1082_108226

theorem simplify_expression (x y : ℤ) (h₁ : x = 2) (h₂ : y = -3) :
  ((2 * x - y) ^ 2 - (x - y) * (x + y) - 2 * y ^ 2) / x = 18 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1082_108226


namespace NUMINAMATH_GPT_square_area_l1082_108295

theorem square_area (x : ℝ) (h1 : BG = GH) (h2 : GH = HD) (h3 : BG = 20 * Real.sqrt 2) : x = 40 * Real.sqrt 2 → x^2 = 3200 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1082_108295


namespace NUMINAMATH_GPT_closest_point_on_line_l1082_108246

structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def line (s : ℚ) : Point ℚ :=
⟨3 + s, 2 - 3 * s, 4 * s⟩

def distance (p1 p2 : Point ℚ) : ℚ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def closestPoint : Point ℚ := ⟨37/17, 74/17, -56/17⟩

def givenPoint : Point ℚ := ⟨1, 4, -2⟩

theorem closest_point_on_line :
  ∃ s : ℚ, line s = closestPoint ∧ 
           ∀ t : ℚ, distance closestPoint givenPoint ≤ distance (line t) givenPoint :=
by
  sorry

end NUMINAMATH_GPT_closest_point_on_line_l1082_108246


namespace NUMINAMATH_GPT_find_a_l1082_108271

theorem find_a (a : ℚ) (h : a + a / 3 + a / 4 = 4) : a = 48 / 19 := by
  sorry

end NUMINAMATH_GPT_find_a_l1082_108271
