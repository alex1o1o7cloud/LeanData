import Mathlib

namespace least_possible_sum_l961_96146

theorem least_possible_sum
  (a b x y z : ℕ)
  (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hpos_x : 0 < x) (hpos_y : 0 < y)
  (hpos_z : 0 < z)
  (h : 3 * a = 7 * b ∧ 7 * b = 5 * x ∧ 5 * x = 4 * y ∧ 4 * y = 6 * z) :
  a + b + x + y + z = 459 :=
by
  sorry

end least_possible_sum_l961_96146


namespace partial_fraction_decomposition_l961_96115

noncomputable def polynomial := λ x: ℝ => x^3 - 24 * x^2 + 88 * x - 75

theorem partial_fraction_decomposition
  (p q r A B C : ℝ)
  (hpq : p ≠ q)
  (hpr : p ≠ r)
  (hqr : q ≠ r)
  (hroots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (hdecomposition: ∀ s: ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
                      1 / polynomial s = A / (s - p) + B / (s - q) + C / (s - r)) :
  (1 / A + 1 / B + 1 / C = 256) := sorry

end partial_fraction_decomposition_l961_96115


namespace min_value_expression_l961_96108

open Real

/-- 
  Given that the function y = log_a(2x+3) - 4 passes through a fixed point P and the fixed point P lies on the line l: ax + by + 7 = 0,
  prove the minimum value of 1/(a+2) + 1/(4b) is 4/9, where a > 0, a ≠ 1, and b > 0.
-/
theorem min_value_expression (a b : ℝ) (h_a : 0 < a) (h_a_ne_1 : a ≠ 1) (h_b : 0 < b)
  (h_eqn : (a * -1 + b * -4 + 7 = 0) → (a + 2 + 4 * b = 9)):
  (1 / (a + 2) + 1 / (4 * b)) = 4 / 9 :=
by
  sorry

end min_value_expression_l961_96108


namespace smallest_positive_expr_l961_96190

theorem smallest_positive_expr (m n : ℤ) : ∃ (m n : ℤ), 216 * m + 493 * n = 1 := 
sorry

end smallest_positive_expr_l961_96190


namespace minute_hand_gains_per_hour_l961_96171

theorem minute_hand_gains_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_by_6pm : total_gain = 63) (hours_from_9_to_6 : total_hours = 9) : (total_gain / total_hours) = 7 :=
by
  -- The proof is not required as per instruction.
  sorry

end minute_hand_gains_per_hour_l961_96171


namespace simplify_expression_l961_96180

theorem simplify_expression (y : ℝ) :
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l961_96180


namespace present_age_of_son_l961_96119

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 32) (h2 : M + 2 = 2 * (S + 2)) : S = 30 :=
by
  sorry

end present_age_of_son_l961_96119


namespace sequence_v_20_l961_96142

noncomputable def sequence_v : ℕ → ℝ → ℝ
| 0, b => b
| (n + 1), b => - (2 / (sequence_v n b + 2))

theorem sequence_v_20 (b : ℝ) (hb : 0 < b) : sequence_v 20 b = -(2 / (b + 2)) :=
by
  sorry

end sequence_v_20_l961_96142


namespace percent_defective_units_shipped_l961_96151

theorem percent_defective_units_shipped :
  let total_units_defective := 6 / 100
  let defective_units_shipped := 4 / 100
  let percent_defective_units_shipped := (total_units_defective * defective_units_shipped) * 100
  percent_defective_units_shipped = 0.24 := by
  sorry

end percent_defective_units_shipped_l961_96151


namespace problem1_l961_96130

variable {a b : ℝ}

theorem problem1 (ha : a > 0) (hb : b > 0) : 
  (1 / (a + b) ≤ 1 / 4 * (1 / a + 1 / b)) :=
sorry

end problem1_l961_96130


namespace olivia_hourly_rate_l961_96113

theorem olivia_hourly_rate (h_worked_monday : ℕ) (h_worked_wednesday : ℕ) (h_worked_friday : ℕ) (h_total_payment : ℕ) (h_total_hours : h_worked_monday + h_worked_wednesday + h_worked_friday = 13) (h_total_amount : h_total_payment = 117) :
  h_total_payment / (h_worked_monday + h_worked_wednesday + h_worked_friday) = 9 :=
by
  sorry

end olivia_hourly_rate_l961_96113


namespace smallest_integer_l961_96199

/-- The smallest integer m such that m > 1 and m has a remainder of 1 when divided by any of 5, 7, and 3 is 106. -/
theorem smallest_integer (m : ℕ) : m > 1 ∧ m % 5 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 ↔ m = 106 :=
by
    sorry

end smallest_integer_l961_96199


namespace right_triangle_sides_l961_96126

/-- Given a right triangle with area 2 * r^2 / 3 where r is the radius of a circle touching one leg,
the extension of the other leg, and the hypotenuse, the sides of the triangle are given by r, 4/3 * r, and 5/3 * r. -/
theorem right_triangle_sides (r : ℝ) (x y : ℝ)
  (h_area : (x * y) / 2 = 2 * r^2 / 3)
  (h_hypotenuse : (x^2 + y^2) = (2 * r + x - y)^2) :
  x = r ∧ y = 4 * r / 3 :=
sorry

end right_triangle_sides_l961_96126


namespace norm_2u_equals_10_l961_96127

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l961_96127


namespace inequality_holds_infinitely_many_times_l961_96161

variable {a : ℕ → ℝ}

theorem inequality_holds_infinitely_many_times
    (h_pos : ∀ n, 0 < a n) :
    ∃ᶠ n in at_top, 1 + a n > a (n - 1) * 2^(1 / n) :=
sorry

end inequality_holds_infinitely_many_times_l961_96161


namespace parabolas_intersect_at_point_l961_96162

theorem parabolas_intersect_at_point :
  ∀ (p q : ℝ), p + q = 2019 → (1 : ℝ)^2 + (p : ℝ) * 1 + q = 2020 :=
by
  intros p q h
  sorry

end parabolas_intersect_at_point_l961_96162


namespace root_sum_value_l961_96179

theorem root_sum_value (r s t : ℝ) (h1: r + s + t = 24) (h2: r * s + s * t + t * r = 50) (h3: r * s * t = 24) :
  r / (1/r + s * t) + s / (1/s + t * r) + t / (1/t + r * s) = 19.04 :=
sorry

end root_sum_value_l961_96179


namespace storks_more_than_birds_l961_96169

-- Definitions based on given conditions
def initial_birds : ℕ := 3
def added_birds : ℕ := 2
def total_birds : ℕ := initial_birds + added_birds
def storks : ℕ := 6

-- Statement to prove the correct answer
theorem storks_more_than_birds : (storks - total_birds = 1) :=
by
  sorry

end storks_more_than_birds_l961_96169


namespace number_is_multiple_of_15_l961_96143

theorem number_is_multiple_of_15
  (W X Y Z D : ℤ)
  (h1 : X - W = 1)
  (h2 : Y - W = 9)
  (h3 : Y - X = 8)
  (h4 : Z - W = 11)
  (h5 : Z - X = 10)
  (h6 : Z - Y = 2)
  (hD : D - X = 5) :
  15 ∣ D :=
by
  sorry -- Proof goes here

end number_is_multiple_of_15_l961_96143


namespace goods_train_speed_l961_96159

theorem goods_train_speed :
  ∀ (length_train length_platform time : ℝ),
    length_train = 250.0416 →
    length_platform = 270 →
    time = 26 →
    (length_train + length_platform) / time = 20 :=
by
  intros length_train length_platform time H_train H_platform H_time
  rw [H_train, H_platform, H_time]
  norm_num
  sorry

end goods_train_speed_l961_96159


namespace jackson_maximum_usd_l961_96170

-- Define the rates for chores in various currencies
def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400
def eur_per_hour : ℝ := 4

-- Define the hours Jackson worked for each task
def usd_hours_vacuuming : ℝ := 2 * 2
def gbp_hours_washing_dishes : ℝ := 0.5
def jpy_hours_cleaning_bathroom : ℝ := 1.5
def eur_hours_sweeping_yard : ℝ := 1

-- Define the exchange rates over three days
def exchange_rates_day1 := (1.35, 0.009, 1.18)  -- (GBP to USD, JPY to USD, EUR to USD)
def exchange_rates_day2 := (1.38, 0.0085, 1.20)
def exchange_rates_day3 := (1.33, 0.0095, 1.21)

-- Define a function to convert currency to USD based on best exchange rates
noncomputable def max_usd (gbp_to_usd jpy_to_usd eur_to_usd : ℝ) : ℝ :=
  (usd_hours_vacuuming * usd_per_hour) +
  (gbp_hours_washing_dishes * gbp_per_hour * gbp_to_usd) +
  (jpy_hours_cleaning_bathroom * jpy_per_hour * jpy_to_usd) +
  (eur_hours_sweeping_yard * eur_per_hour * eur_to_usd)

-- Prove the maximum USD Jackson can have by choosing optimal rates is $32.61
theorem jackson_maximum_usd : max_usd 1.38 0.0095 1.21 = 32.61 :=
by
  sorry

end jackson_maximum_usd_l961_96170


namespace women_ratio_l961_96155

theorem women_ratio (pop : ℕ) (w_retail : ℕ) (w_fraction : ℚ) (h_pop : pop = 6000000) (h_w_retail : w_retail = 1000000) (h_w_fraction : w_fraction = 1 / 3) : 
  (3000000 : ℚ) / (6000000 : ℚ) = 1 / 2 :=
by sorry

end women_ratio_l961_96155


namespace probability_of_different_topics_l961_96140

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end probability_of_different_topics_l961_96140


namespace fractional_equation_solution_l961_96105

noncomputable def problem_statement (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 = 2 / (x^2 - 1))

theorem fractional_equation_solution :
  ∀ x : ℝ, problem_statement x → x = -2 :=
by
  intro x hx
  sorry

end fractional_equation_solution_l961_96105


namespace exponent_arithmetic_proof_l961_96181

theorem exponent_arithmetic_proof :
  ( (6 ^ 6 / 6 ^ 5) ^ 3 * 8 ^ 3 / 4 ^ 3) = 1728 := by
  sorry

end exponent_arithmetic_proof_l961_96181


namespace order_of_f_values_l961_96122

noncomputable def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else 0 -- define f such that it handles the missing part

theorem order_of_f_values :
  (∀ x: ℝ, f (2 - x) = f (1 + x)) ∧ (∀ x: ℝ, x >= 1 → f x = 3^x - 1) →
  f 0 < f 3 ∧ f 3 < f (-2) :=
by
  sorry

end order_of_f_values_l961_96122


namespace bags_weight_after_removal_l961_96110

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l961_96110


namespace rhombus_angles_l961_96198

-- Define the conditions for the proof
variables (a e f : ℝ) (α β : ℝ)

-- Using the geometric mean condition
def geometric_mean_condition := a^2 = e * f

-- Using the condition that diagonals of a rhombus intersect at right angles and bisect each other
def diagonals_intersect_perpendicularly := α + β = 180 ∧ α = 30 ∧ β = 150

-- Prove the question assuming the given conditions
theorem rhombus_angles (h1 : geometric_mean_condition a e f) (h2 : diagonals_intersect_perpendicularly α β) : 
  (α = 30) ∧ (β = 150) :=
sorry

end rhombus_angles_l961_96198


namespace simplify_expression_l961_96145

theorem simplify_expression (a b : ℤ) : 
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b :=
by
  sorry

end simplify_expression_l961_96145


namespace cookie_ratio_l961_96160

theorem cookie_ratio (f : ℚ) (h_monday : 32 = 32) (h_tuesday : (f : ℚ) * 32 = 32 * (f : ℚ)) 
    (h_wednesday : 3 * (f : ℚ) * 32 - 4 + 32 + (f : ℚ) * 32 = 92) :
    f = 1/2 :=
by
  sorry

end cookie_ratio_l961_96160


namespace area_of_quadrilateral_ABCD_l961_96196

theorem area_of_quadrilateral_ABCD
  (BD : ℝ) (hA : ℝ) (hC : ℝ) (angle_ABD : ℝ) :
  BD = 28 ∧ hA = 8 ∧ hC = 2 ∧ angle_ABD = 60 →
  ∃ (area_ABCD : ℝ), area_ABCD = 140 :=
by
  sorry

end area_of_quadrilateral_ABCD_l961_96196


namespace f_2019_value_l961_96136

noncomputable def f : ℕ → ℕ := sorry

theorem f_2019_value
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  f 2019 = 2019 :=
sorry

end f_2019_value_l961_96136


namespace f_of_integral_ratio_l961_96103

variable {f : ℝ → ℝ} (h_cont : ∀ x > 0, continuous_at f x)
variable (h_int : ∀ a b : ℝ, a > 0 → b > 0 → ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a))

theorem f_of_integral_ratio :
  (∃ c : ℝ, ∀ x > 0, f x = c / x) :=
sorry

end f_of_integral_ratio_l961_96103


namespace city_roads_different_colors_l961_96114

-- Definitions and conditions
def Intersection (α : Type) := α × α × α

def City (α : Type) :=
  { intersections : α → Intersection α // 
    ∀ i : α, ∃ c₁ c₂ c₃ : α, intersections i = (c₁, c₂, c₃) 
    ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ 
  }

variables {α : Type}

-- Statement to prove that the three roads leading out of the city have different colors
theorem city_roads_different_colors (c : City α) 
  (roads_outside : α → Prop)
  (h : ∃ r₁ r₂ r₃, roads_outside r₁ ∧ roads_outside r₂ ∧ roads_outside r₃ ∧ 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁) : 
  true := 
sorry

end city_roads_different_colors_l961_96114


namespace lemonade_water_requirement_l961_96197

variables (W S L H : ℕ)

-- Definitions based on the conditions
def water_equation (W S : ℕ) := W = 5 * S
def sugar_equation (S L : ℕ) := S = 3 * L
def honey_equation (H L : ℕ) := H = L
def lemon_juice_amount (L : ℕ) := L = 2

-- Theorem statement for the proof problem
theorem lemonade_water_requirement :
  ∀ (W S L H : ℕ), 
  (water_equation W S) →
  (sugar_equation S L) →
  (honey_equation H L) →
  (lemon_juice_amount L) →
  W = 30 :=
by
  intros W S L H hW hS hH hL
  sorry

end lemonade_water_requirement_l961_96197


namespace common_rational_root_is_neg_one_third_l961_96176

theorem common_rational_root_is_neg_one_third (a b c d e f g : ℚ) :
  ∃ k : ℚ, (75 * k^4 + a * k^3 + b * k^2 + c * k + 12 = 0) ∧
           (12 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 75 = 0) ∧
           (¬ k.isInt) ∧ (k < 0) ∧ (k = -1/3) :=
sorry

end common_rational_root_is_neg_one_third_l961_96176


namespace sequence_solution_l961_96100

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 5 ∧ a 8 = 8 ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 20) ∧
  (a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 5 ∧ a 5 = 8 ∧ a 6 = 7 ∧ a 7 = 5 ∧ a 8 = 8) :=
by {
  sorry
}

end sequence_solution_l961_96100


namespace number_of_girls_l961_96133

theorem number_of_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 2 * B / 3 + G = 18) : G = 18 :=
by
  sorry

end number_of_girls_l961_96133


namespace num_perfect_square_factors_of_180_l961_96132

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l961_96132


namespace exclude_domain_and_sum_l961_96109

noncomputable def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem exclude_domain_and_sum :
  { x : ℝ | x = 0 ∨ x = -1/2 ∨ x = -1/4 } = { x : ℝ | ¬(x ≠ 0 ∧ (2 + 1 / x ≠ 0) ∧ (2 + 1 / (2 + 1 / x) ≠ 0)) } ∧
  (0 + (-1 / 2) + (-1 / 4) = -3 / 4) :=
by
  sorry

end exclude_domain_and_sum_l961_96109


namespace solution_set_f1_geq_4_min_value_pq_l961_96106

-- Define the function f(x) for the first question
def f1 (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for part (I)
theorem solution_set_f1_geq_4 (x : ℝ) : f1 x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 :=
by
  sorry

-- Define the function f(x) for the second question
def f2 (m x : ℝ) : ℝ := |x - m| + |x - 3|

-- Theorem for part (II)
theorem min_value_pq (p q m : ℝ) (h_pos_p : p > 0) (h_pos_q : q > 0)
    (h_eq : 1 / p + 1 / (2 * q) = m)
    (h_min_f : ∀ x : ℝ, f2 m x ≥ 3) :
    pq = 1 / 18 :=
by
  sorry

end solution_set_f1_geq_4_min_value_pq_l961_96106


namespace number_decomposition_l961_96184

theorem number_decomposition : 10101 = 10000 + 100 + 1 :=
by
  sorry

end number_decomposition_l961_96184


namespace joe_total_toy_cars_l961_96116

def initial_toy_cars : ℕ := 50
def uncle_additional_factor : ℝ := 1.5

theorem joe_total_toy_cars :
  (initial_toy_cars : ℝ) + uncle_additional_factor * initial_toy_cars = 125 := 
by
  sorry

end joe_total_toy_cars_l961_96116


namespace tan_of_alpha_l961_96189

theorem tan_of_alpha 
  (α : ℝ)
  (h1 : Real.sin α = (3 / 5))
  (h2 : α ∈ Set.Ioo (π / 2) π) : Real.tan α = -3 / 4 :=
sorry

end tan_of_alpha_l961_96189


namespace find_dividend_l961_96120

-- Conditions
def quotient : ℕ := 4
def divisor : ℕ := 4

-- Dividend computation
def dividend (q d : ℕ) : ℕ := q * d

-- Theorem to prove
theorem find_dividend : dividend quotient divisor = 16 := 
by
  -- Placeholder for the proof, not needed as per instructions
  sorry

end find_dividend_l961_96120


namespace factorize_expression_l961_96173

theorem factorize_expression (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := 
by sorry

end factorize_expression_l961_96173


namespace tan_to_sin_cos_l961_96104

theorem tan_to_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := 
sorry

end tan_to_sin_cos_l961_96104


namespace correct_average_l961_96112

theorem correct_average (incorrect_avg : ℝ) (num_values : ℕ) (misread_value actual_value : ℝ) 
  (h1 : incorrect_avg = 16) 
  (h2 : num_values = 10)
  (h3 : misread_value = 26)
  (h4 : actual_value = 46) : 
  (incorrect_avg * num_values + (actual_value - misread_value)) / num_values = 18 := 
by
  sorry

end correct_average_l961_96112


namespace alice_bob_numbers_sum_l961_96121

-- Fifty slips of paper numbered 1 to 50 are placed in a hat.
-- Alice and Bob each draw one number from the hat without replacement, keeping their numbers hidden from each other.
-- Alice cannot tell who has the larger number.
-- Bob knows who has the larger number.
-- Bob's number is composite.
-- If Bob's number is multiplied by 50 and Alice's number is added, the result is a perfect square.
-- Prove that the sum of Alice's and Bob's numbers is 29.

theorem alice_bob_numbers_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 50) (hB : 1 ≤ B ∧ B ≤ 50) 
  (hAB_distinct : A ≠ B) (hA_unknown : ¬(A = 1 ∨ A = 50))
  (hB_composite : ∃ d > 1, d < B ∧ B % d = 0) (h_perfect_square : ∃ k, 50 * B + A = k ^ 2) :
  A + B = 29 := by
  sorry

end alice_bob_numbers_sum_l961_96121


namespace calculate_expression_l961_96154

theorem calculate_expression (m n : ℝ) : 9 * m^2 - (m - 2 * n)^2 = 4 * (2 * m - n) * (m + n) :=
by
  sorry

end calculate_expression_l961_96154


namespace average_age_when_youngest_born_l961_96157

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_y : ℕ) (total_yr : ℕ) (reduction_yr yr_older : ℕ) (avg_age_older : ℕ) 
  (h1 : n = 7)
  (h2 : avg_age = 30)
  (h3 : current_y = 7)
  (h4 : total_yr = n * avg_age)
  (h5 : reduction_yr = (n - 1) * current_y)
  (h6 : yr_older = total_yr - reduction_yr)
  (h7 : avg_age_older = yr_older / (n - 1)) :
  avg_age_older = 28 :=
by 
  sorry

end average_age_when_youngest_born_l961_96157


namespace juniors_in_club_l961_96124

theorem juniors_in_club
  (j s x y : ℝ)
  (h1 : x = 0.4 * j)
  (h2 : y = 0.25 * s)
  (h3 : j + s = 36)
  (h4 : x = 2 * y) :
  j = 20 :=
by
  sorry

end juniors_in_club_l961_96124


namespace equation_of_perpendicular_line_l961_96165

theorem equation_of_perpendicular_line (x y c : ℝ) (h₁ : x = -1) (h₂ : y = 2)
  (h₃ : 2 * x - 3 * y = -c) (h₄ : 3 * x + 2 * y - 7 = 0) :
  2 * x - 3 * y + 8 = 0 :=
sorry

end equation_of_perpendicular_line_l961_96165


namespace cost_of_soda_l961_96123

-- Define the system of equations
theorem cost_of_soda (b s f : ℕ): 
  3 * b + s = 390 ∧ 
  2 * b + 3 * s = 440 ∧ 
  b + 2 * f = 230 ∧ 
  s + 3 * f = 270 → 
  s = 234 := 
by 
  sorry

end cost_of_soda_l961_96123


namespace amusement_park_admission_l961_96153

def number_of_children (children_fee : ℤ) (adults_fee : ℤ) (total_people : ℤ) (total_fees : ℤ) : ℤ :=
  let y := (total_fees - total_people * children_fee) / (adults_fee - children_fee)
  total_people - y

theorem amusement_park_admission :
  number_of_children 15 40 315 8100 = 180 :=
by
  -- Fees in cents to avoid decimals
  sorry  -- Placeholder for the proof

end amusement_park_admission_l961_96153


namespace slope_range_l961_96178

theorem slope_range (a b : ℝ) (h₁ : a ≠ -2) (h₂ : a ≠ 2) 
  (h₃ : a^2 / 4 + b^2 / 3 = 1) (h₄ : -2 ≤ b / (a - 2) ∧ b / (a - 2) ≤ -1) :
  (3 / 8 ≤ b / (a + 2) ∧ b / (a + 2) ≤ 3 / 4) :=
sorry

end slope_range_l961_96178


namespace number_of_bricks_in_wall_l961_96118

noncomputable def rate_one_bricklayer (x : ℕ) : ℚ := x / 8
noncomputable def rate_other_bricklayer (x : ℕ) : ℚ := x / 12
noncomputable def combined_rate_with_efficiency (x : ℕ) : ℚ := (rate_one_bricklayer x + rate_other_bricklayer x - 15)
noncomputable def total_time (x : ℕ) : ℚ := 6 * combined_rate_with_efficiency x

theorem number_of_bricks_in_wall (x : ℕ) : total_time x = x → x = 360 :=
by sorry

end number_of_bricks_in_wall_l961_96118


namespace tom_pays_l961_96117

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end tom_pays_l961_96117


namespace sin_add_arcsin_arctan_l961_96195

theorem sin_add_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  Real.sin (a + b) = (2 + 3 * Real.sqrt 3) / 10 :=
by
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (Real.sqrt 3)
  sorry

end sin_add_arcsin_arctan_l961_96195


namespace number_of_paths_l961_96186

/-
We need to define the conditions and the main theorem
-/

def grid_width : ℕ := 5
def grid_height : ℕ := 4
def total_steps : ℕ := 8
def steps_right : ℕ := 5
def steps_up : ℕ := 3

theorem number_of_paths : (Nat.choose total_steps steps_up) = 56 := by
  sorry

end number_of_paths_l961_96186


namespace sheepdog_catches_sheep_in_20_seconds_l961_96137

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end sheepdog_catches_sheep_in_20_seconds_l961_96137


namespace necessary_but_not_sufficient_condition_l961_96139

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a < 1) → ((a + 1) * (a - 2) < 0) ∧ ((∃ b : ℝ, (b + 1) * (b - 2) < 0 ∧ ¬(0 < b ∧ b < 1))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l961_96139


namespace symmetric_point_correct_line_passes_second_quadrant_l961_96111

theorem symmetric_point_correct (x y: ℝ) (h_line : y = x + 1) :
  (x, y) = (-1, 2) :=
sorry

theorem line_passes_second_quadrant (m x y: ℝ) (h_line: m * x + y + m - 1 = 0) :
  (x, y) = (-1, 1) :=
sorry

end symmetric_point_correct_line_passes_second_quadrant_l961_96111


namespace Warriors_won_25_games_l961_96164

def CricketResults (Sharks Falcons Warriors Foxes Knights : ℕ) :=
  Sharks > Falcons ∧
  (Warriors > Foxes ∧ Warriors < Knights) ∧
  Foxes > 15 ∧
  (Foxes = 20 ∨ Foxes = 25 ∨ Foxes = 30) ∧
  (Warriors = 20 ∨ Warriors = 25 ∨ Warriors = 30) ∧
  (Knights = 20 ∨ Knights = 25 ∨ Knights = 30)

theorem Warriors_won_25_games (Sharks Falcons Warriors Foxes Knights : ℕ) 
  (h : CricketResults Sharks Falcons Warriors Foxes Knights) :
  Warriors = 25 :=
by
  sorry

end Warriors_won_25_games_l961_96164


namespace common_remainder_is_zero_l961_96156

noncomputable def least_number := 100040

theorem common_remainder_is_zero 
  (n : ℕ) 
  (h1 : n = least_number) 
  (condition1 : 4 ∣ n)
  (condition2 : 610 ∣ n)
  (condition3 : 15 ∣ n)
  (h2 : (n.digits 10).sum = 5)
  : ∃ r : ℕ, ∀ (a : ℕ), (a ∈ [4, 610, 15] → n % a = r) ∧ r = 0 :=
by {
  sorry
}

end common_remainder_is_zero_l961_96156


namespace min_balls_in_circle_l961_96144

theorem min_balls_in_circle (b w n k : ℕ) 
  (h1 : b = 2 * w)
  (h2 : n = b + w) 
  (h3 : n - 2 * k = 6 * k) :
  n >= 24 :=
sorry

end min_balls_in_circle_l961_96144


namespace value_of_six_prime_prime_l961_96177

-- Define the function q' 
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Stating the main theorem we want to prove
theorem value_of_six_prime_prime : prime (prime 6) = 42 :=
by
  sorry

end value_of_six_prime_prime_l961_96177


namespace fixed_point_exists_l961_96152

theorem fixed_point_exists : ∀ (m : ℝ), (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  intro m
  have h : (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
    sorry
  exact h

end fixed_point_exists_l961_96152


namespace minimize_sum_of_f_seq_l961_96135

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end minimize_sum_of_f_seq_l961_96135


namespace train_length_proof_l961_96125

def train_length_crosses_bridge (train_speed_kmh : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  distance - bridge_length_m

theorem train_length_proof : 
  train_length_crosses_bridge 72 150 20 = 250 :=
by
  let train_speed_kmh := 72
  let bridge_length_m := 150
  let crossing_time_s := 20
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  have h : distance = 400 := by sorry
  have h_eq : distance - bridge_length_m = 250 := by sorry
  exact h_eq

end train_length_proof_l961_96125


namespace polygon_interior_exterior_equal_l961_96158

theorem polygon_interior_exterior_equal (n : ℕ) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end polygon_interior_exterior_equal_l961_96158


namespace three_kids_savings_l961_96148

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end three_kids_savings_l961_96148


namespace abigail_time_to_finish_l961_96167

noncomputable def words_total : ℕ := 1000
noncomputable def words_per_30_min : ℕ := 300
noncomputable def words_already_written : ℕ := 200
noncomputable def time_per_word : ℝ := 30 / words_per_30_min

theorem abigail_time_to_finish :
  (words_total - words_already_written) * time_per_word = 80 :=
by
  sorry

end abigail_time_to_finish_l961_96167


namespace line_intersects_x_axis_at_neg3_l961_96138

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (5 * y - 7 * x = 21 ∧ y = 0) ↔ (x = -3 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_neg3_l961_96138


namespace sum_of_consecutive_integers_bound_sqrt_40_l961_96102

theorem sum_of_consecutive_integers_bound_sqrt_40 (a b : ℤ) (h₁ : a < Real.sqrt 40) (h₂ : Real.sqrt 40 < b) (h₃ : b = a + 1) : a + b = 13 :=
by
  sorry

end sum_of_consecutive_integers_bound_sqrt_40_l961_96102


namespace largest_digit_divisible_by_6_l961_96182

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ (56780 + N) % 6 = 0 ∧ (∀ M : ℕ, M ≤ 9 → (M % 2 = 0 ∧ (56780 + M) % 3 = 0) → M ≤ N) :=
by
  sorry

end largest_digit_divisible_by_6_l961_96182


namespace simplify_and_evaluate_l961_96141

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end simplify_and_evaluate_l961_96141


namespace perpendicular_vectors_parallel_vectors_l961_96147

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors_l961_96147


namespace sin_cos_identity_tan_identity_l961_96191

open Real

namespace Trigonometry

variable (α : ℝ)

-- Given conditions
def given_conditions := (sin α + cos α = (1/5)) ∧ (0 < α) ∧ (α < π)

-- Prove that sin(α) * cos(α) = -12/25
theorem sin_cos_identity (h : given_conditions α) : sin α * cos α = -12/25 := 
sorry

-- Prove that tan(α) = -4/3
theorem tan_identity (h : given_conditions α) : tan α = -4/3 :=
sorry

end Trigonometry

end sin_cos_identity_tan_identity_l961_96191


namespace train_time_to_pass_bridge_l961_96172

theorem train_time_to_pass_bridge
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmph : ℝ)
  (h1 : length_train = 500) (h2 : length_bridge = 200) (h3 : speed_kmph = 72) :
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_mps
  time = 35 :=
by
  sorry

end train_time_to_pass_bridge_l961_96172


namespace total_profit_l961_96188

-- Definitions based on the conditions
variables (A B C : ℝ) (P : ℝ)
variables (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400)

-- The theorem we are going to prove
theorem total_profit (A B C P : ℝ) (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400) : 
  P = 7700 :=
by
  sorry

end total_profit_l961_96188


namespace units_digit_of_product_composites_l961_96131

def is_composite (n : ℕ) : Prop := 
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem units_digit_of_product_composites (h1 : is_composite 9) (h2 : is_composite 10) (h3 : is_composite 12) :
  (9 * 10 * 12) % 10 = 0 :=
by
  sorry

end units_digit_of_product_composites_l961_96131


namespace time_to_cover_escalator_l961_96185

-- Definitions of the rates and length
def escalator_speed : ℝ := 12
def person_speed : ℝ := 2
def escalator_length : ℝ := 210

-- Theorem statement that we need to prove
theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed) = 15) :=
by
  sorry

end time_to_cover_escalator_l961_96185


namespace eval_expr_l961_96192

theorem eval_expr : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end eval_expr_l961_96192


namespace apples_difference_l961_96175

-- Definitions based on conditions
def JackiesApples : Nat := 10
def AdamsApples : Nat := 8

-- Statement
theorem apples_difference : JackiesApples - AdamsApples = 2 := by
  sorry

end apples_difference_l961_96175


namespace three_y_squared_value_l961_96101

theorem three_y_squared_value : ∃ x y : ℤ, 3 * x + y = 40 ∧ 2 * x - y = 20 ∧ 3 * y ^ 2 = 48 :=
by
  sorry

end three_y_squared_value_l961_96101


namespace curve_is_circle_l961_96163

-- Definition of the curve in polar coordinates
def curve (r θ : ℝ) : Prop :=
  r = 3 * Real.sin θ

-- The theorem to prove
theorem curve_is_circle : ∀ θ : ℝ, ∃ r : ℝ, curve r θ → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ p : ℝ × ℝ, (Real.sqrt ((p.1 - c.1) ^ 2 + (p.2 - c.2) ^ 2) = R)) :=
by
  sorry

end curve_is_circle_l961_96163


namespace sticks_form_equilateral_triangle_l961_96129

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l961_96129


namespace number_of_true_propositions_l961_96128

-- Definitions based on the problem
def proposition1 (α β : ℝ) : Prop := (α + β = 180) → (α + β = 90)
def proposition2 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)
def proposition3 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)

-- Proof problem statement
theorem number_of_true_propositions : ∃ n : ℕ, n = 2 :=
by
  let p1 := false
  let p2 := false
  let p3 := true
  existsi (if p3 then 1 else 0 + if p2 then 1 else 0 + if p1 then 1 else 0)
  simp
  sorry

end number_of_true_propositions_l961_96128


namespace evaluate_fraction_l961_96183

theorem evaluate_fraction:
  (125 : ℝ)^(1/3) / (64 : ℝ)^(1/2) * (81 : ℝ)^(1/4) = 15 / 8 := 
by
  sorry

end evaluate_fraction_l961_96183


namespace remainder_y150_div_yminus2_4_l961_96174

theorem remainder_y150_div_yminus2_4 (y : ℝ) :
  (y ^ 150) % ((y - 2) ^ 4) = 554350 * (y - 2) ^ 3 + 22350 * (y - 2) ^ 2 + 600 * (y - 2) + 8 * 2 ^ 147 :=
by
  sorry

end remainder_y150_div_yminus2_4_l961_96174


namespace noelle_speed_l961_96194

theorem noelle_speed (v d : ℝ) (h1 : d > 0) (h2 : v > 0) 
  (h3 : (2 * d) / ((d / v) + (d / 15)) = 5) : v = 3 := 
sorry

end noelle_speed_l961_96194


namespace sum_fourth_power_l961_96149

  theorem sum_fourth_power (x y z : ℝ) 
    (h1 : x + y + z = 2) 
    (h2 : x^2 + y^2 + z^2 = 6) 
    (h3 : x^3 + y^3 + z^3 = 8) : 
    x^4 + y^4 + z^4 = 26 := 
  by 
    sorry
  
end sum_fourth_power_l961_96149


namespace sector_area_proof_l961_96134

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_proof
  (r : ℝ) (l : ℝ) (perimeter : ℝ) (theta : ℝ) (h1 : perimeter = 2 * r + l)
  (h2 : l = r * theta) (h3 : perimeter = 16) (h4 : theta = 2) :
  sector_area r theta = 16 := by
  sorry

end sector_area_proof_l961_96134


namespace geometric_sequence_properties_l961_96150

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a q →
    a 2 = 6 →
    a 5 - 2 * a 4 - a 3 + 12 = 0 →
    ∀ n, a n = 6 ∨ a n = 6 * (-1)^(n-2) ∨ a n = 6 * 2^(n-2) :=
by
  sorry

end geometric_sequence_properties_l961_96150


namespace solve_for_Theta_l961_96166

-- Define the two-digit number representation condition
def fourTheta (Θ : ℕ) : ℕ := 40 + Θ

-- Main theorem statement
theorem solve_for_Theta (Θ : ℕ) (h1 : 198 / Θ = fourTheta Θ + Θ) (h2 : 0 < Θ ∧ Θ < 10) : Θ = 4 :=
by
  sorry

end solve_for_Theta_l961_96166


namespace mary_should_drink_six_glasses_per_day_l961_96107

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l961_96107


namespace history_book_cost_is_correct_l961_96187

-- Define the conditions
def total_books : ℕ := 80
def math_book_cost : ℕ := 4
def total_price : ℕ := 390
def math_books_purchased : ℕ := 10

-- The number of history books
def history_books_purchased : ℕ := total_books - math_books_purchased

-- The total cost of math books
def total_cost_math_books : ℕ := math_books_purchased * math_book_cost

-- The total cost of history books
def total_cost_history_books : ℕ := total_price - total_cost_math_books

-- Define the cost of each history book
def history_book_cost : ℕ := total_cost_history_books / history_books_purchased

-- The theorem to be proven
theorem history_book_cost_is_correct : history_book_cost = 5 := 
by
  sorry

end history_book_cost_is_correct_l961_96187


namespace required_brick_volume_l961_96193

theorem required_brick_volume :
  let height := 4 / 12 -- in feet
  let length := 6 -- in feet
  let thickness := 4 / 12 -- in feet
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  rounded_volume = 1 := 
by
  let height := 1 / 3
  let length := 6
  let thickness := 1 / 3
  let volume := height * length * thickness
  let rounded_volume := Nat.ceil volume
  show rounded_volume = 1
  sorry

end required_brick_volume_l961_96193


namespace caleb_ice_cream_vs_frozen_yoghurt_l961_96168

theorem caleb_ice_cream_vs_frozen_yoghurt :
  let cost_chocolate_ice_cream := 6 * 5
  let discount_chocolate := 0.10 * cost_chocolate_ice_cream
  let total_chocolate_ice_cream := cost_chocolate_ice_cream - discount_chocolate

  let cost_vanilla_ice_cream := 4 * 4
  let discount_vanilla := 0.07 * cost_vanilla_ice_cream
  let total_vanilla_ice_cream := cost_vanilla_ice_cream - discount_vanilla

  let total_ice_cream := total_chocolate_ice_cream + total_vanilla_ice_cream

  let cost_strawberry_yoghurt := 3 * 3
  let tax_strawberry := 0.05 * cost_strawberry_yoghurt
  let total_strawberry_yoghurt := cost_strawberry_yoghurt + tax_strawberry

  let cost_mango_yoghurt := 2 * 2
  let tax_mango := 0.03 * cost_mango_yoghurt
  let total_mango_yoghurt := cost_mango_yoghurt + tax_mango

  let total_yoghurt := total_strawberry_yoghurt + total_mango_yoghurt

  (total_ice_cream - total_yoghurt = 28.31) := by
  sorry

end caleb_ice_cream_vs_frozen_yoghurt_l961_96168
