import Mathlib

namespace calculate_a5_l1040_104070

variable {a1 : ℝ} -- geometric sequence first term
variable {a : ℕ → ℝ} -- geometric sequence
variable {n : ℕ} -- sequence index
variable {r : ℝ} -- common ratio

-- Definitions based on the given conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a1 * r ^ n

-- Given conditions
axiom common_ratio_is_two : r = 2
axiom product_condition : a 2 * a 10 = 16 -- indices offset by 1, so a3 = a 2 and a11 = a 10
axiom positive_terms : ∀ n, a n > 0

-- Goal: calculate a 4
theorem calculate_a5 : a 4 = 1 :=
sorry

end calculate_a5_l1040_104070


namespace range_of_a_l1040_104035

theorem range_of_a (a : ℝ) (x : ℝ) : (∃ x, x^2 - a*x - a ≤ -3) → (a ≤ -6 ∨ a ≥ 2) :=
sorry

end range_of_a_l1040_104035


namespace common_ratio_of_geometric_series_l1040_104097

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l1040_104097


namespace exists_multiple_digits_0_1_l1040_104079

theorem exists_multiple_digits_0_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k ≤ n) ∧ (∃ m : ℕ, m * n = k) ∧ (∀ d : ℕ, ∃ i : ℕ, i ≤ n ∧ d = 0 ∨ d = 1) :=
sorry

end exists_multiple_digits_0_1_l1040_104079


namespace calculate_smaller_sphere_radius_l1040_104005

noncomputable def smaller_sphere_radius (r1 r2 r3 r4 : ℝ) : ℝ := 
  if h : r1 = 2 ∧ r2 = 2 ∧ r3 = 3 ∧ r4 = 3 then 
    6 / 11 
  else 
    0

theorem calculate_smaller_sphere_radius :
  smaller_sphere_radius 2 2 3 3 = 6 / 11 :=
by
  sorry

end calculate_smaller_sphere_radius_l1040_104005


namespace find_m_l1040_104077

variable (m : ℝ)

-- Definitions of the vectors
def AB : ℝ × ℝ := (m + 3, 2 * m + 1)
def CD : ℝ × ℝ := (m + 3, -5)

-- Definition of perpendicular vectors, dot product is zero
def perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_m (h : perp (AB m) (CD m)) : m = 2 := by
  sorry

end find_m_l1040_104077


namespace find_xy_such_that_product_is_fifth_power_of_prime_l1040_104041

theorem find_xy_such_that_product_is_fifth_power_of_prime
  (x y : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (x^2 + y) * (y^2 + x) = p^5) :
  (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
sorry

end find_xy_such_that_product_is_fifth_power_of_prime_l1040_104041


namespace max_value_of_fraction_l1040_104004

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l1040_104004


namespace mixed_groups_count_l1040_104012

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ)
  (group_size : ℕ) (total_boy_boy_photos : ℕ)
  (total_girl_girl_photos : ℕ)
  (total_photos : ℕ)
  (each_group_photos : ℕ)
  (total_mixed_photos : ℕ)
  (mixed_group_count : ℕ):
  total_children = 300 ∧
  total_groups = 100 ∧
  group_size = 3 ∧
  total_boy_boy_photos = 100 ∧
  total_girl_girl_photos = 56 ∧
  each_group_photos = 3 ∧
  total_photos = 100 * each_group_photos ∧
  total_mixed_photos = total_photos - total_boy_boy_photos - total_girl_girl_photos ∧
  mixed_group_count = total_mixed_photos / 2 →
  mixed_group_count = 72 :=
by
  sorry

end mixed_groups_count_l1040_104012


namespace shirt_selling_price_l1040_104009

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l1040_104009


namespace original_difference_in_books_l1040_104044

theorem original_difference_in_books 
  (x y : ℕ) 
  (h1 : x + y = 5000) 
  (h2 : (1 / 2 : ℚ) * (x - 400) - (y + 400) = 400) : 
  x - y = 3000 := 
by 
  -- Placeholder for the proof
  sorry

end original_difference_in_books_l1040_104044


namespace exponent_property_l1040_104020

theorem exponent_property :
  4^4 * 9^4 * 4^9 * 9^9 = 36^13 :=
by
  -- Add the proof here
  sorry

end exponent_property_l1040_104020


namespace bride_groom_couples_sum_l1040_104034

def wedding_reception (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) : Prop :=
  total_guests - friends = couples_guests

theorem bride_groom_couples_sum (B G : ℕ) (total_guests : ℕ) (friends : ℕ) (couples_guests : ℕ) 
  (h1 : total_guests = 180) (h2 : friends = 100) (h3 : wedding_reception total_guests friends couples_guests) 
  (h4 : couples_guests = 80) : B + G = 40 := 
  by
  sorry

end bride_groom_couples_sum_l1040_104034


namespace find_A_l1040_104055

def A : ℕ := 7 * 5 + 3

theorem find_A : A = 38 :=
by
  sorry

end find_A_l1040_104055


namespace probability_A_correct_l1040_104024

-- Definitions of probabilities
variable (P_A P_B : Prop)
variable (P_AB : Prop := P_A ∧ P_B)
variable (prob_AB : ℝ := 2 / 3)
variable (prob_B_given_A : ℝ := 8 / 9)

-- Lean statement of the mathematical problem
theorem probability_A_correct :
  (P_AB → P_A ∧ P_B) →
  (prob_AB = (2 / 3)) →
  (prob_B_given_A = (2 / 3) / prob_A) →
  (∃ prob_A : ℝ, prob_A = 3 / 4) :=
by
  sorry

end probability_A_correct_l1040_104024


namespace largest_angle_is_90_degrees_l1040_104056

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem largest_angle_is_90_degrees (u : ℝ) (a b c : ℝ) (v : ℝ) (h_v : v = 1)
  (h_a : a = Real.sqrt (2 * u - 1))
  (h_b : b = Real.sqrt (2 * u + 3))
  (h_c : c = 2 * Real.sqrt (u + v)) :
  is_right_triangle a b c :=
by
  sorry

end largest_angle_is_90_degrees_l1040_104056


namespace decorative_object_height_l1040_104059

def diameter_fountain := 20 -- meters
def radius_fountain := diameter_fountain / 2 -- meters

def max_height := 8 -- meters
def distance_to_max_height := 2 -- meters

-- The initial height of the water jets at the decorative object
def initial_height := 7.5 -- meters

theorem decorative_object_height :
  initial_height = 7.5 :=
  sorry

end decorative_object_height_l1040_104059


namespace smallest_integer_larger_than_expr_is_248_l1040_104080

noncomputable def small_int_larger_than_expr : ℕ :=
  let expr := (Real.sqrt 5 + Real.sqrt 3)^4
  248

theorem smallest_integer_larger_than_expr_is_248 :
    ∃ (n : ℕ), n > (Real.sqrt 5 + Real.sqrt 3)^4 ∧ n = small_int_larger_than_expr := 
by
  -- We introduce the target integer 248
  use (248 : ℕ)
  -- The given conditions should lead us to 248 being greater than the expression.
  sorry

end smallest_integer_larger_than_expr_is_248_l1040_104080


namespace ryan_more_hours_english_than_spanish_l1040_104094

-- Define the time spent on various languages as constants
def hoursEnglish : ℕ := 7
def hoursSpanish : ℕ := 4

-- State the problem as a theorem
theorem ryan_more_hours_english_than_spanish : hoursEnglish - hoursSpanish = 3 :=
by sorry

end ryan_more_hours_english_than_spanish_l1040_104094


namespace quotient_of_division_l1040_104045

theorem quotient_of_division 
  (dividend divisor remainder : ℕ) 
  (h_dividend : dividend = 265) 
  (h_divisor : divisor = 22) 
  (h_remainder : remainder = 1) 
  (h_div : dividend = divisor * (dividend / divisor) + remainder) : 
  (dividend / divisor) = 12 := 
by
  sorry

end quotient_of_division_l1040_104045


namespace arithmetic_sequence_sum_l1040_104019

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end arithmetic_sequence_sum_l1040_104019


namespace similar_triangles_same_heights_ratio_l1040_104090

theorem similar_triangles_same_heights_ratio (h1 h2 : ℝ) 
  (sim_ratio : h1 / h2 = 1 / 4) : h1 / h2 = 1 / 4 :=
by
  sorry

end similar_triangles_same_heights_ratio_l1040_104090


namespace find_second_bank_account_balance_l1040_104071

theorem find_second_bank_account_balance : 
  (exists (X : ℝ),  
    let raw_material_cost := 100
    let machinery_cost := 125
    let raw_material_tax := 0.05 * raw_material_cost
    let discounted_machinery_cost := machinery_cost - (0.1 * machinery_cost)
    let machinery_tax := 0.08 * discounted_machinery_cost
    let total_raw_material_cost := raw_material_cost + raw_material_tax
    let total_machinery_cost := discounted_machinery_cost + machinery_tax
    let total_spent := total_raw_material_cost + total_machinery_cost
    let total_cash := 900 + X
    let spent_proportion := 0.2 * total_cash
    total_spent = spent_proportion → X = 232.50) :=
by {
  sorry
}

end find_second_bank_account_balance_l1040_104071


namespace boat_distance_downstream_l1040_104085

theorem boat_distance_downstream (v_s : ℝ) (h : 8 - v_s = 5) :
  8 + v_s = 11 :=
by
  sorry

end boat_distance_downstream_l1040_104085


namespace total_race_distance_l1040_104060

theorem total_race_distance :
  let sadie_time := 2
  let sadie_speed := 3
  let ariana_time := 0.5
  let ariana_speed := 6
  let total_time := 4.5
  let sarah_speed := 4
  let sarah_time := total_time - sadie_time - ariana_time
  let sadie_distance := sadie_speed * sadie_time
  let ariana_distance := ariana_speed * ariana_time
  let sarah_distance := sarah_speed * sarah_time
  let total_distance := sadie_distance + ariana_distance + sarah_distance
  total_distance = 17 :=
by
  sorry

end total_race_distance_l1040_104060


namespace find_a_l1040_104022

noncomputable def f (x : Real) (a : Real) : Real :=
if h : 0 < x ∧ x < 2 then (Real.log x - a * x) 
else 
if h' : -2 < x ∧ x < 0 then sorry
else 
   sorry

theorem find_a (a : Real) : (∀ x : Real, f x a = - f (-x) a) → (∀ x: Real, (0 < x ∧ x < 2) → f x a = Real.log x - a * x) → a > (1 / 2) → (∀ x: Real, (-2 < x ∧ x < 0) → f x a ≥ 1) → a = 1 := 
sorry

end find_a_l1040_104022


namespace no_real_solutions_l1040_104030

theorem no_real_solutions :
  ¬ ∃ (a b c d : ℝ), 
  (a^3 + c^3 = 2) ∧ 
  (a^2 * b + c^2 * d = 0) ∧ 
  (b^3 + d^3 = 1) ∧ 
  (a * b^2 + c * d^2 = -6) := 
by
  sorry

end no_real_solutions_l1040_104030


namespace monotonic_exponential_decreasing_l1040_104091

variable (a : ℝ) (f : ℝ → ℝ)

theorem monotonic_exponential_decreasing {m n : ℝ}
  (h0 : a = (Real.sqrt 5 - 1) / 2)
  (h1 : ∀ x, f x = a^x)
  (h2 : 0 < a ∧ a < 1)
  (h3 : f m > f n) :
  m < n :=
sorry

end monotonic_exponential_decreasing_l1040_104091


namespace green_ball_probability_l1040_104078

def containerA := (8, 2) -- 8 green, 2 red
def containerB := (6, 4) -- 6 green, 4 red
def containerC := (5, 5) -- 5 green, 5 red
def containerD := (8, 2) -- 8 green, 2 red

def probability_of_green : ℚ :=
  (1 / 4) * (8 / 10) + (1 / 4) * (6 / 10) + (1 / 4) * (5 / 10) + (1 / 4) * (8 / 10)
  
theorem green_ball_probability :
  probability_of_green = 43 / 160 :=
sorry

end green_ball_probability_l1040_104078


namespace correct_sum_of_integers_l1040_104053

theorem correct_sum_of_integers (a b : ℕ) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := 
  sorry

end correct_sum_of_integers_l1040_104053


namespace time_in_future_is_4_l1040_104037

def current_time := 5
def future_hours := 1007
def modulo := 12
def future_time := (current_time + future_hours) % modulo

theorem time_in_future_is_4 : future_time = 4 := by
  sorry

end time_in_future_is_4_l1040_104037


namespace revenue_equation_l1040_104075

theorem revenue_equation (x : ℝ) (r_j r_t : ℝ) (h1 : r_j = 90) (h2 : r_t = 144) :
  r_j + r_j * (1 + x) + r_j * (1 + x)^2 = r_t :=
by
  rw [h1, h2]
  sorry

end revenue_equation_l1040_104075


namespace horse_food_per_day_l1040_104002

theorem horse_food_per_day
  (total_horse_food_per_day : ℕ)
  (sheep_count : ℕ)
  (sheep_to_horse_ratio : ℕ)
  (horse_to_sheep_ratio : ℕ)
  (horse_food_per_horse_per_day : ℕ) :
  sheep_to_horse_ratio * horse_food_per_horse_per_day = total_horse_food_per_day / (sheep_count / sheep_to_horse_ratio * horse_to_sheep_ratio) :=
by
  -- Given
  let total_horse_food_per_day := 12880
  let sheep_count := 24
  let sheep_to_horse_ratio := 3
  let horse_to_sheep_ratio := 7

  -- We need to show that horse_food_per_horse_per_day = 230
  have horse_count : ℕ := (sheep_count / sheep_to_horse_ratio) * horse_to_sheep_ratio
  have horse_food_per_horse_per_day : ℕ := total_horse_food_per_day / horse_count

  -- Desired proof statement
  sorry

end horse_food_per_day_l1040_104002


namespace minimum_m_n_sum_l1040_104040

theorem minimum_m_n_sum:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 90 * m = n ^ 3 ∧ m + n = 330 :=
sorry

end minimum_m_n_sum_l1040_104040


namespace f_at_2008_l1040_104025

noncomputable def f : ℝ → ℝ := sorry
noncomputable def finv : ℝ → ℝ := sorry

axiom f_inverse : ∀ x, f (finv x) = x ∧ finv (f x) = x
axiom f_at_9 : f 9 = 18

theorem f_at_2008 : f 2008 = -1981 :=
by
  sorry

end f_at_2008_l1040_104025


namespace initial_amount_l1040_104087

theorem initial_amount (x : ℕ) (h1 : x - 3 + 14 = 22) : x = 11 :=
sorry

end initial_amount_l1040_104087


namespace range_of_alpha_l1040_104007

open Real

theorem range_of_alpha 
  (α : ℝ) (k : ℤ) :
  (sin α > 0) ∧ (cos α < 0) ∧ (sin α > cos α) →
  (∃ k : ℤ, (2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) ∨ 
  (2 * k * π + (3 * π / 2) < α ∧ α < 2 * k * π + 2 * π)) := 
by 
  sorry

end range_of_alpha_l1040_104007


namespace initial_points_l1040_104058

theorem initial_points (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end initial_points_l1040_104058


namespace find_X_l1040_104032

theorem find_X :
  let N := 90
  let X := (1 / 15) * N - (1 / 2 * 1 / 3 * 1 / 5 * N)
  X = 3 := by
  sorry

end find_X_l1040_104032


namespace part1_part2_l1040_104031

noncomputable def cost_prices (x y : ℕ) : Prop := 
  8800 / (y + 4) = 2 * (4000 / x) ∧ 
  x = 40 ∧ 
  y = 44

theorem part1 : ∃ x y : ℕ, cost_prices x y := sorry

noncomputable def minimum_lucky_rabbits (m : ℕ) : Prop := 
  26 * m + 20 * (200 - m) ≥ 4120 ∧ 
  m = 20

theorem part2 : ∃ m : ℕ, minimum_lucky_rabbits m := sorry

end part1_part2_l1040_104031


namespace sophomore_spaghetti_tortellini_ratio_l1040_104017

theorem sophomore_spaghetti_tortellini_ratio
    (total_students : ℕ)
    (spaghetti_lovers : ℕ)
    (tortellini_lovers : ℕ)
    (grade_levels : ℕ)
    (spaghetti_sophomores : ℕ)
    (tortellini_sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : spaghetti_lovers = 300)
    (h3 : tortellini_lovers = 120)
    (h4 : grade_levels = 4)
    (h5 : spaghetti_sophomores = spaghetti_lovers / grade_levels)
    (h6 : tortellini_sophomores = tortellini_lovers / grade_levels) :
    (spaghetti_sophomores : ℚ) / (tortellini_sophomores : ℚ) = 5 / 2 := by
  sorry

end sophomore_spaghetti_tortellini_ratio_l1040_104017


namespace polygon_sides_l1040_104027

theorem polygon_sides (perimeter side_length : ℕ) (h₁ : perimeter = 150) (h₂ : side_length = 15): 
  (perimeter / side_length) = 10 := 
by
  -- Here goes the proof part
  sorry

end polygon_sides_l1040_104027


namespace tan_ratio_of_triangle_sides_l1040_104062

theorem tan_ratio_of_triangle_sides (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : α + β + γ = π)
  (h3 : c ≠ 0):
  ( (Real.tan γ) / (Real.tan α + Real.tan β) ) = (a * b) / (1011 * c^2) := 
sorry

end tan_ratio_of_triangle_sides_l1040_104062


namespace line_equation_of_projection_l1040_104001

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_v2 := v.1 * v.1 + v.2 * v.2
  (dot_uv / norm_v2 * v.1, dot_uv / norm_v2 * v.2)

theorem line_equation_of_projection (x y : ℝ) :
  proj (x, y) (3, -4) = (9 / 5, -12 / 5) ↔ y = (3 / 4) * x - 15 / 4 :=
sorry

end line_equation_of_projection_l1040_104001


namespace Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l1040_104033

def Q (x : ℂ) (n : ℕ) : ℂ := (x + 1)^n + x^n + 1
def P (x : ℂ) : ℂ := x^2 + x + 1

-- Part a) Q(x) is divisible by P(x) if and only if n ≡ 2 (mod 6) or n ≡ 4 (mod 6)
theorem Q_divisible_by_P (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 2 ∨ n % 6 = 4) := sorry

-- Part b) Q(x) is divisible by P(x)^2 if and only if n ≡ 4 (mod 6)
theorem Q_divisible_by_P_squared (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 4 := sorry

-- Part c) Q(x) is never divisible by P(x)^3
theorem Q_not_divisible_by_P_cubed (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^3 ≠ 0 := sorry

end Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l1040_104033


namespace george_speed_to_school_l1040_104023

theorem george_speed_to_school :
  ∀ (d1 d2 v1 v2 v_arrive : ℝ), 
  d1 = 1.0 → d2 = 0.5 → v1 = 3.0 → v2 * (d1 / v1 + d2 / v2) = (d1 + d2) / 4.0 → v_arrive = 12.0 :=
by sorry

end george_speed_to_school_l1040_104023


namespace integer_value_l1040_104084

theorem integer_value (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (h3 : x > 0) (h4 : y > 0) (h5 : z > 0) :
  ∃ a : ℕ, a + y + z = 26 ∧ a = 15 := by
  sorry

end integer_value_l1040_104084


namespace value_of_y_l1040_104011

theorem value_of_y (y : ℝ) (h : (3 * y - 9) / 3 = 18) : y = 21 :=
sorry

end value_of_y_l1040_104011


namespace total_candies_l1040_104082

-- Define variables and conditions
variables (x y z : ℕ)
axiom h1 : x = y / 2
axiom h2 : x + z = 24
axiom h3 : y + z = 34

-- The statement to be proved
theorem total_candies : x + y + z = 44 :=
by
  sorry

end total_candies_l1040_104082


namespace sufficient_not_necessary_condition_l1040_104069

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < 1 → x < 2) ∧ ¬ (x < 2 → x < 1) :=
by
  sorry

end sufficient_not_necessary_condition_l1040_104069


namespace no_four_distinct_numbers_l1040_104014

theorem no_four_distinct_numbers (x y : ℝ) (h : x ≠ y ∧ 
    (x^(10:ℕ) + (x^(9:ℕ)) * y + (x^(8:ℕ)) * (y^(2:ℕ)) + 
    (x^(7:ℕ)) * (y^(3:ℕ)) + (x^(6:ℕ)) * (y^(4:ℕ)) + 
    (x^(5:ℕ)) * (y^(5:ℕ)) + (x^(4:ℕ)) * (y^(6:ℕ)) + 
    (x^(3:ℕ)) * (y^(7:ℕ)) + (x^(2:ℕ)) * (y^(8:ℕ)) + 
    (x^(1:ℕ)) * (y^(9:ℕ)) + (y^(10:ℕ)) = 1)) : False :=
by
  sorry

end no_four_distinct_numbers_l1040_104014


namespace calc_perimeter_l1040_104061

noncomputable def width (w: ℝ) (h: ℝ) : Prop :=
  h = w + 10

noncomputable def cost (P: ℝ) (rate: ℝ) (total_cost: ℝ) : Prop :=
  total_cost = P * rate

noncomputable def perimeter (w: ℝ) (P: ℝ) : Prop :=
  P = 2 * (w + (w + 10))

theorem calc_perimeter {w P : ℝ} (h_rate : ℝ) (h_total_cost : ℝ)
  (h1 : width w (w + 10))
  (h2 : cost (2 * (w + (w + 10))) h_rate h_total_cost) :
  P = 2 * (w + (w + 10)) →
  h_total_cost = 910 →
  h_rate = 6.5 →
  w = 30 →
  P = 140 :=
sorry

end calc_perimeter_l1040_104061


namespace positive_integer_sixk_l1040_104048

theorem positive_integer_sixk (n : ℕ) :
  (∃ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ d1 + d2 + d3 = n ∧ d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n) ↔ (∃ k : ℕ, n = 6 * k) :=
by
  sorry

end positive_integer_sixk_l1040_104048


namespace problem_T8_l1040_104013

noncomputable def a : Nat → ℚ
| 0     => 1/2
| (n+1) => a n / (1 + 3 * a n)

noncomputable def T (n : Nat) : ℚ :=
  (Finset.range n).sum (λ i => 1 / a (i + 1))

theorem problem_T8 : T 8 = 100 :=
sorry

end problem_T8_l1040_104013


namespace max_tiles_on_floor_l1040_104043

   -- Definitions corresponding to conditions
   def tile_length_1 : ℕ := 35
   def tile_width_1 : ℕ := 30
   def tile_length_2 : ℕ := 30
   def tile_width_2 : ℕ := 35
   def floor_length : ℕ := 1000
   def floor_width : ℕ := 210

   -- Conditions:
   -- 1. Tiles do not overlap.
   -- 2. Tiles are placed with edges jutting against each other on all edges.
   -- 3. A tile can be placed in any orientation so long as its edges are parallel to the edges of the floor.
   -- 4. No tile should overshoot any edge of the floor.

   theorem max_tiles_on_floor :
     let tiles_orientation_1 := (floor_length / tile_length_1) * (floor_width / tile_width_1)
     let tiles_orientation_2 := (floor_length / tile_length_2) * (floor_width / tile_width_2)
     max tiles_orientation_1 tiles_orientation_2 = 198 :=
   by {
     -- The actual proof handling is skipped, as per instructions.
     sorry
   }
   
end max_tiles_on_floor_l1040_104043


namespace problem_l1040_104066

theorem problem (y : ℝ) (hy : 5 = y^2 + 4 / y^2) : y + 2 / y = 3 ∨ y + 2 / y = -3 :=
by
  sorry

end problem_l1040_104066


namespace movie_hours_sum_l1040_104081

noncomputable def total_movie_hours 
  (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : ℕ :=
  Joyce + Michael + Nikki + Ryn + Sam

theorem movie_hours_sum (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : 
  total_movie_hours Michael Joyce Nikki Ryn Sam h1 h2 h3 h4 h5 = 94 :=
by 
  -- The actual proof will go here, to demonstrate the calculations resulting in 94 hours
  sorry

end movie_hours_sum_l1040_104081


namespace eggs_in_each_basket_l1040_104072

theorem eggs_in_each_basket
  (total_red_eggs : ℕ)
  (total_orange_eggs : ℕ)
  (h_red : total_red_eggs = 30)
  (h_orange : total_orange_eggs = 45)
  (eggs_in_each_basket : ℕ)
  (h_at_least : eggs_in_each_basket ≥ 5) :
  (total_red_eggs % eggs_in_each_basket = 0) ∧ 
  (total_orange_eggs % eggs_in_each_basket = 0) ∧
  eggs_in_each_basket = 15 := sorry

end eggs_in_each_basket_l1040_104072


namespace least_number_to_subtract_l1040_104065

theorem least_number_to_subtract (n : ℕ) (h1 : n = 157632)
  (h2 : ∃ k : ℕ, k = 12 * 18 * 24 / (gcd 12 (gcd 18 24)) ∧ k ∣ n - 24) :
  n - 24 = 24 := 
sorry

end least_number_to_subtract_l1040_104065


namespace percentage_problem_l1040_104096

theorem percentage_problem (P : ℝ) :
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 :=
by
  intro h
  sorry

end percentage_problem_l1040_104096


namespace find_a_l1040_104042

theorem find_a (a x : ℝ)
    (h1 : 6 * (x + 8) = 18 * x)
    (h2 : 6 * x - 2 * (a - x) = 2 * a + x) :
    a = 7 :=
  sorry

end find_a_l1040_104042


namespace time_descend_hill_l1040_104036

-- Definitions
def time_to_top : ℝ := 4
def avg_speed_whole_journey : ℝ := 3
def avg_speed_uphill : ℝ := 2.25

-- Theorem statement
theorem time_descend_hill (t : ℝ) 
  (h1 : time_to_top = 4) 
  (h2 : avg_speed_whole_journey = 3) 
  (h3 : avg_speed_uphill = 2.25) : 
  t = 2 := 
sorry

end time_descend_hill_l1040_104036


namespace player_A_winning_strategy_l1040_104015

-- Define the game state and the player's move
inductive Move
| single (index : Nat) : Move
| double (index : Nat) : Move

-- Winning strategy prop
def winning_strategy (n : Nat) (first_player : Bool) : Prop :=
  ∀ moves : List Move, moves.length ≤ n → (first_player → false) → true

-- Main theorem stating that player A always has a winning strategy
theorem player_A_winning_strategy (n : Nat) (h : n ≥ 1) : winning_strategy n true := 
by 
  -- directly prove the statement
  sorry

end player_A_winning_strategy_l1040_104015


namespace bisection_method_third_interval_l1040_104092

theorem bisection_method_third_interval 
  (f : ℝ → ℝ) (a b : ℝ) (H1 : a = -2) (H2 : b = 4) 
  (H3 : f a * f b ≤ 0) : 
  ∃ c d : ℝ, c = -1/2 ∧ d = 1 ∧ f c * f d ≤ 0 :=
by 
  sorry

end bisection_method_third_interval_l1040_104092


namespace jade_and_julia_total_money_l1040_104038

theorem jade_and_julia_total_money (x : ℕ) : 
  let jade_initial := 38 
  let julia_initial := jade_initial / 2 
  let jade_after := jade_initial + x 
  let julia_after := julia_initial + x 
  jade_after + julia_after = 57 + 2 * x := by
  sorry

end jade_and_julia_total_money_l1040_104038


namespace frequency_of_middle_rectangle_l1040_104098

theorem frequency_of_middle_rectangle
    (n : ℕ)
    (A : ℕ)
    (h1 : A + (n - 1) * A = 160) :
    A = 32 :=
by
  sorry

end frequency_of_middle_rectangle_l1040_104098


namespace combined_ratio_l1040_104029

theorem combined_ratio (cayley_students fermat_students : ℕ) 
                       (cayley_ratio_boys cayley_ratio_girls fermat_ratio_boys fermat_ratio_girls : ℕ) 
                       (h_cayley : cayley_students = 400) 
                       (h_cayley_ratio : (cayley_ratio_boys, cayley_ratio_girls) = (3, 2)) 
                       (h_fermat : fermat_students = 600) 
                       (h_fermat_ratio : (fermat_ratio_boys, fermat_ratio_girls) = (2, 3)) :
  (480 : ℚ) / 520 = 12 / 13 := 
by 
  sorry

end combined_ratio_l1040_104029


namespace original_ratio_of_flour_to_baking_soda_l1040_104028

-- Define the conditions
def sugar_to_flour_ratio_5_to_5 (sugar flour : ℕ) : Prop :=
  sugar = 2400 ∧ sugar = flour

def baking_soda_mass_condition (flour : ℕ) (baking_soda : ℕ) : Prop :=
  flour = 2400 ∧ (∃ b : ℕ, baking_soda = b ∧ flour / (b + 60) = 8)

-- The theorem statement we need to prove
theorem original_ratio_of_flour_to_baking_soda :
  ∃ flour baking_soda : ℕ,
  sugar_to_flour_ratio_5_to_5 2400 flour ∧
  baking_soda_mass_condition flour baking_soda →
  flour / baking_soda = 10 :=
by
  sorry

end original_ratio_of_flour_to_baking_soda_l1040_104028


namespace final_result_is_106_l1040_104099

def chosen_number : ℕ := 122
def multiplied_by_2 (x : ℕ) : ℕ := 2 * x
def subtract_138 (y : ℕ) : ℕ := y - 138

theorem final_result_is_106 : subtract_138 (multiplied_by_2 chosen_number) = 106 :=
by
  -- proof is omitted
  sorry

end final_result_is_106_l1040_104099


namespace original_cost_price_40_l1040_104076

theorem original_cost_price_40
  (selling_price : ℝ)
  (decrease_rate : ℝ)
  (profit_increase_rate : ℝ)
  (new_selling_price := selling_price)
  (original_cost_price : ℝ)
  (new_cost_price := (1 - decrease_rate) * original_cost_price)
  (original_profit_margin := (selling_price - original_cost_price) / original_cost_price)
  (new_profit_margin := (new_selling_price - new_cost_price) / new_cost_price)
  (profit_margin_increase := profit_increase_rate)
  (h1 : selling_price = 48)
  (h2 : decrease_rate = 0.04)
  (h3 : profit_increase_rate = 0.05)
  (h4 : new_profit_margin = original_profit_margin + profit_margin_increase) :
  original_cost_price = 40 := 
by 
  sorry

end original_cost_price_40_l1040_104076


namespace jackson_grade_increase_per_hour_l1040_104052

-- Define the necessary variables
variables (v s p G : ℕ)

-- The conditions from the problem
def study_condition1 : v = 9 := sorry
def study_condition2 : s = v / 3 := sorry
def grade_starts_at_zero : G = s * p := sorry
def final_grade : G = 45 := sorry

-- The final problem statement to prove
theorem jackson_grade_increase_per_hour :
  p = 15 :=
by
  -- Add our sorry to indicate the partial proof
  sorry

end jackson_grade_increase_per_hour_l1040_104052


namespace calc_expr_correct_l1040_104003

noncomputable def eval_expr : ℚ :=
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5

theorem calc_expr_correct : eval_expr = 12.5 :=
by
  sorry

end calc_expr_correct_l1040_104003


namespace intersection_M_N_l1040_104026

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l1040_104026


namespace percentage_of_girls_l1040_104086

def total_students : ℕ := 100
def boys : ℕ := 50
def girls : ℕ := total_students - boys

theorem percentage_of_girls :
  (girls / total_students) * 100 = 50 := sorry

end percentage_of_girls_l1040_104086


namespace walnut_trees_currently_in_park_l1040_104018

-- Definitions from the conditions
def total_trees : ℕ := 77
def trees_to_be_planted : ℕ := 44

-- Statement to prove: number of current trees = 33
theorem walnut_trees_currently_in_park : total_trees - trees_to_be_planted = 33 :=
by
  sorry

end walnut_trees_currently_in_park_l1040_104018


namespace classes_Mr_Gates_has_l1040_104067

theorem classes_Mr_Gates_has (buns_per_package packages_bought students_per_class buns_per_student : ℕ) :
  buns_per_package = 8 → 
  packages_bought = 30 → 
  students_per_class = 30 → 
  buns_per_student = 2 → 
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 := 
by
  sorry

end classes_Mr_Gates_has_l1040_104067


namespace part1_part2_l1040_104068

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0 ∧ x < 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 + m * y + 1 = 0 ∧ y < 0)
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0

-- Lean statement for part 1
theorem part1 (m : ℝ) :
  ¬ ¬ p m → m > 2 :=
sorry

-- Lean statement for part 2
theorem part2 (m : ℝ) :
  (p m ∨ q m) ∧ (¬(p m ∧ q m)) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end part1_part2_l1040_104068


namespace complex_square_l1040_104047

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end complex_square_l1040_104047


namespace avg_speed_3x_km_l1040_104073

-- Definitions based on the conditions
def distance1 (x : ℕ) : ℕ := x
def speed1 : ℕ := 90
def distance2 (x : ℕ) : ℕ := 2 * x
def speed2 : ℕ := 20

-- The total distance covered
def total_distance (x : ℕ) : ℕ := distance1 x + distance2 x

-- The time taken for each part of the journey
def time1 (x : ℕ) : ℚ := distance1 x / speed1
def time2 (x : ℕ) : ℚ := distance2 x / speed2

-- The total time taken
def total_time (x : ℕ) : ℚ := time1 x + time2 x

-- The average speed
def average_speed (x : ℕ) : ℚ := total_distance x / total_time x

-- The theorem we want to prove
theorem avg_speed_3x_km (x : ℕ) : average_speed x = 27 := by
  sorry

end avg_speed_3x_km_l1040_104073


namespace number_solution_exists_l1040_104083

theorem number_solution_exists (x : ℝ) (h : 0.80 * x = (4 / 5 * 15) + 20) : x = 40 :=
sorry

end number_solution_exists_l1040_104083


namespace child_ticket_cost_l1040_104088

variable (A C : ℕ) -- A stands for the number of adults, C stands for the cost of one child's ticket

theorem child_ticket_cost 
  (number_of_adults : ℕ) 
  (number_of_children : ℕ) 
  (cost_concessions : ℕ) 
  (total_cost_trip : ℕ)
  (cost_adult_ticket : ℕ) 
  (ticket_costs : ℕ) 
  (total_adult_cost : ℕ) 
  (remaining_ticket_cost : ℕ) 
  (child_ticket : ℕ) :
  number_of_adults = 5 →
  number_of_children = 2 →
  cost_concessions = 12 →
  total_cost_trip = 76 →
  cost_adult_ticket = 10 →
  ticket_costs = total_cost_trip - cost_concessions →
  total_adult_cost = number_of_adults * cost_adult_ticket →
  remaining_ticket_cost = ticket_costs - total_adult_cost →
  child_ticket = remaining_ticket_cost / number_of_children →
  C = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Adding sorry since the proof is not required
  sorry

end child_ticket_cost_l1040_104088


namespace isosceles_triangle_perimeter_l1040_104008

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_eq_triangle : a + b + c = 60) (h_eq_sides : a = b) 
  (isosceles_base : c = 15) (isosceles_side1_eq : a = 20) : a + b + c = 55 :=
by
  sorry

end isosceles_triangle_perimeter_l1040_104008


namespace min_socks_to_guarantee_10_pairs_l1040_104000

/--
Given a drawer containing 100 red socks, 80 green socks, 60 blue socks, and 40 black socks, 
and socks are selected one at a time without seeing their color. 
The minimum number of socks that must be selected to guarantee at least 10 pairs is 23.
-/
theorem min_socks_to_guarantee_10_pairs 
  (red_socks green_socks blue_socks black_socks : ℕ) 
  (total_pairs : ℕ)
  (h_red : red_socks = 100)
  (h_green : green_socks = 80)
  (h_blue : blue_socks = 60)
  (h_black : black_socks = 40)
  (h_total_pairs : total_pairs = 10) :
  ∃ (n : ℕ), n = 23 := 
sorry

end min_socks_to_guarantee_10_pairs_l1040_104000


namespace m_leq_nine_l1040_104006

theorem m_leq_nine (m : ℝ) : (∀ x : ℝ, (x^2 - 4*x + 3 < 0) → (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + m < 0)) → m ≤ 9 :=
by
sorry

end m_leq_nine_l1040_104006


namespace geometric_sequence_common_ratio_l1040_104016

theorem geometric_sequence_common_ratio {a : ℕ+ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_a3 : a 3 = 1) (h_a5 : a 5 = 4) : q = 2 ∨ q = -2 :=
by
  sorry

end geometric_sequence_common_ratio_l1040_104016


namespace john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l1040_104074

theorem john_needs_to_sell_1200_pencils_to_make_120_dollars_profit :
  ∀ (buy_rate_pencils : ℕ) (buy_rate_dollars : ℕ) (sell_rate_pencils : ℕ) (sell_rate_dollars : ℕ),
    buy_rate_pencils = 5 →
    buy_rate_dollars = 7 →
    sell_rate_pencils = 4 →
    sell_rate_dollars = 6 →
    ∃ (n_pencils : ℕ), n_pencils = 1200 ∧ 
                        (sell_rate_dollars / sell_rate_pencils - buy_rate_dollars / buy_rate_pencils) * n_pencils = 120 :=
by
  sorry

end john_needs_to_sell_1200_pencils_to_make_120_dollars_profit_l1040_104074


namespace solve_for_a_when_diamond_eq_6_l1040_104010

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem solve_for_a_when_diamond_eq_6 (a : ℝ) : diamond a 3 = 6 → a = 8 :=
by
  intros h
  simp [diamond] at h
  sorry

end solve_for_a_when_diamond_eq_6_l1040_104010


namespace remainder_of_7_pow_308_mod_11_l1040_104064

theorem remainder_of_7_pow_308_mod_11 :
  (7 ^ 308) % 11 = 9 :=
by
  sorry

end remainder_of_7_pow_308_mod_11_l1040_104064


namespace find_f1_l1040_104051

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 1 / 2 → f x + f ((x + 2) / (1 - 2 * x)) = x) :
  f 1 = 7 / 6 :=
sorry

end find_f1_l1040_104051


namespace logic_problem_l1040_104095

theorem logic_problem (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬ (p ∨ q) :=
sorry

end logic_problem_l1040_104095


namespace ratio_cube_sphere_surface_area_l1040_104049

theorem ratio_cube_sphere_surface_area (R : ℝ) (h1 : R > 0) :
  let Scube := 24 * R^2
  let Ssphere := 4 * Real.pi * R^2
  (Scube / Ssphere) = (6 / Real.pi) :=
by
  sorry

end ratio_cube_sphere_surface_area_l1040_104049


namespace bus_children_problem_l1040_104046

theorem bus_children_problem :
  ∃ X, 5 - 63 + X = 14 ∧ X - 63 = 9 :=
by 
  sorry

end bus_children_problem_l1040_104046


namespace min_value_expression_l1040_104093

open Real

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_abc : a * b * c = 1 / 2) : 
    a^2 + 8 * a * b + 32 * b^2 + 16 * b * c + 8 * c^2 ≥ 18 :=
sorry

end min_value_expression_l1040_104093


namespace cars_with_neither_features_l1040_104057

-- Define the given conditions
def total_cars : ℕ := 65
def cars_with_power_steering : ℕ := 45
def cars_with_power_windows : ℕ := 25
def cars_with_both_features : ℕ := 17

-- Define the statement to be proved
theorem cars_with_neither_features : total_cars - (cars_with_power_steering + cars_with_power_windows - cars_with_both_features) = 12 :=
by
  sorry

end cars_with_neither_features_l1040_104057


namespace min_distance_ants_l1040_104089

open Real

theorem min_distance_ants (points : Fin 1390 → ℝ × ℝ) :
  (∀ i j : Fin 1390, i ≠ j → dist (points i) (points j) > 0.02) → 
  (∀ i : Fin 1390, |(points i).snd| < 0.01) → 
  ∃ i j : Fin 1390, i ≠ j ∧ dist (points i) (points j) > 10 :=
by
  sorry

end min_distance_ants_l1040_104089


namespace a_10_eq_505_l1040_104063

-- The sequence definition
def a (n : ℕ) : ℕ :=
  let start := (n * (n - 1)) / 2 + 1
  List.sum (List.range' start n)

-- Theorem that the 10th term of the sequence is 505
theorem a_10_eq_505 : a 10 = 505 := 
by
  sorry

end a_10_eq_505_l1040_104063


namespace cage_cost_correct_l1040_104050

noncomputable def total_amount_paid : ℝ := 20
noncomputable def change_received : ℝ := 0.26
noncomputable def cat_toy_cost : ℝ := 8.77
noncomputable def cage_cost := total_amount_paid - change_received

theorem cage_cost_correct : cage_cost = 19.74 := by
  sorry

end cage_cost_correct_l1040_104050


namespace no_real_pairs_for_same_lines_l1040_104054

theorem no_real_pairs_for_same_lines : ¬ ∃ (a b : ℝ), (∀ x y : ℝ, 2 * x + a * y + b = 0 ↔ b * x - 3 * y + 15 = 0) :=
by {
  sorry
}

end no_real_pairs_for_same_lines_l1040_104054


namespace geometric_sequence_a4_l1040_104021

theorem geometric_sequence_a4 (x a_4 : ℝ) (h1 : 2*x + 2 = (3*x + 3) * (2*x + 2) / x)
  (h2 : x = -4 ∨ x = -1) (h3 : x = -4) : a_4 = -27 / 2 :=
by
  sorry

end geometric_sequence_a4_l1040_104021


namespace geometric_sequence_a6_l1040_104039

theorem geometric_sequence_a6 :
  ∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n) ∧ (a 4 * a 10 = 16) → (a 6 = 2) :=
by
  sorry

end geometric_sequence_a6_l1040_104039
