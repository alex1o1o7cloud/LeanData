import Mathlib

namespace NUMINAMATH_GPT_escalator_ride_time_l411_41124

theorem escalator_ride_time (x y k t : ℝ)
  (h1 : 75 * x = y)
  (h2 : 30 * (x + k) = y)
  (h3 : t = y / k) :
  t = 50 := by
  sorry

end NUMINAMATH_GPT_escalator_ride_time_l411_41124


namespace NUMINAMATH_GPT_tangent_line_condition_l411_41191

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end NUMINAMATH_GPT_tangent_line_condition_l411_41191


namespace NUMINAMATH_GPT_f_is_odd_max_min_values_l411_41162

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
variable (f_one : f 1 = -2)
variable (f_neg : ∀ x > 0, f x < 0)

-- Define the statement in Lean for Part 1: proving the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by sorry

-- Define the statement in Lean for Part 2: proving the max and min values on [-3, 3]
theorem max_min_values : 
  ∃ max_value min_value : ℝ, 
  (max_value = f (-3) ∧ max_value = 6) ∧ 
  (min_value = f (3) ∧ min_value = -6) := by sorry

end NUMINAMATH_GPT_f_is_odd_max_min_values_l411_41162


namespace NUMINAMATH_GPT_problem_proof_l411_41190

theorem problem_proof (a b c x y z : ℝ) (h₁ : 17 * x + b * y + c * z = 0) (h₂ : a * x + 29 * y + c * z = 0)
                      (h₃ : a * x + b * y + 53 * z = 0) (ha : a ≠ 17) (hx : x ≠ 0) :
                      (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_proof_l411_41190


namespace NUMINAMATH_GPT_jose_bottle_caps_proof_l411_41177

def jose_bottle_caps_initial : Nat := 7
def rebecca_bottle_caps : Nat := 2
def jose_bottle_caps_final : Nat := 9

theorem jose_bottle_caps_proof : jose_bottle_caps_initial + rebecca_bottle_caps = jose_bottle_caps_final := by
  sorry

end NUMINAMATH_GPT_jose_bottle_caps_proof_l411_41177


namespace NUMINAMATH_GPT_puppies_left_l411_41133

theorem puppies_left (initial_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : initial_puppies = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_puppies = initial_puppies - given_away) : 
  remaining_puppies = 5 :=
  by
  sorry

end NUMINAMATH_GPT_puppies_left_l411_41133


namespace NUMINAMATH_GPT_lettuce_types_l411_41103

/-- Let L be the number of types of lettuce. 
    Given that Terry has 3 types of tomatoes, 4 types of olives, 
    and 2 types of soup. The total number of options for his lunch combo is 48. 
    Prove that L = 2. --/

theorem lettuce_types (L : ℕ) (H : 3 * 4 * 2 * L = 48) : L = 2 :=
by {
  -- beginning of the proof
  sorry
}

end NUMINAMATH_GPT_lettuce_types_l411_41103


namespace NUMINAMATH_GPT_evaluate_expression_l411_41171

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l411_41171


namespace NUMINAMATH_GPT_halved_r_value_of_n_l411_41147

theorem halved_r_value_of_n (r a : ℝ) (n : ℕ) (h₁ : a = (2 * r)^n)
  (h₂ : 0.125 * a = r^n) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_halved_r_value_of_n_l411_41147


namespace NUMINAMATH_GPT_max_number_of_children_l411_41137

theorem max_number_of_children (apples cookies chocolates : ℕ) (remaining_apples remaining_cookies remaining_chocolates : ℕ) 
  (h₁ : apples = 55) 
  (h₂ : cookies = 114) 
  (h₃ : chocolates = 83) 
  (h₄ : remaining_apples = 3) 
  (h₅ : remaining_cookies = 10) 
  (h₆ : remaining_chocolates = 5) : 
  gcd (apples - remaining_apples) (gcd (cookies - remaining_cookies) (chocolates - remaining_chocolates)) = 26 :=
by
  sorry

end NUMINAMATH_GPT_max_number_of_children_l411_41137


namespace NUMINAMATH_GPT_age_of_B_l411_41120

/--
A is two years older than B.
B is twice as old as C.
The total of the ages of A, B, and C is 32.
How old is B?
-/
theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 32) : B = 12 :=
by
  sorry

end NUMINAMATH_GPT_age_of_B_l411_41120


namespace NUMINAMATH_GPT_total_area_covered_by_strips_l411_41118

theorem total_area_covered_by_strips (L W : ℝ) (n : ℕ) (overlap_area : ℝ) (end_to_end_area : ℝ) :
  L = 15 → W = 1 → n = 4 → overlap_area = 15 → end_to_end_area = 30 → 
  (L * W * n - overlap_area + end_to_end_area) = 45 :=
by
  intros hL hW hn hoverlap hend_to_end
  sorry

end NUMINAMATH_GPT_total_area_covered_by_strips_l411_41118


namespace NUMINAMATH_GPT_incorrect_expression_l411_41168

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : x / (y - x) ≠ 5 / 2 := 
by
  sorry

end NUMINAMATH_GPT_incorrect_expression_l411_41168


namespace NUMINAMATH_GPT_fraction_division_l411_41179

theorem fraction_division : 
  ((8 / 4) * (9 / 3) * (20 / 5)) / ((10 / 5) * (12 / 4) * (15 / 3)) = (4 / 5) := 
by
  sorry

end NUMINAMATH_GPT_fraction_division_l411_41179


namespace NUMINAMATH_GPT_probability_of_both_red_is_one_sixth_l411_41110

noncomputable def probability_both_red (red blue green : ℕ) (balls_picked : ℕ) : ℚ :=
  if balls_picked = 2 ∧ red = 4 ∧ blue = 3 ∧ green = 2 then (4 / 9) * (3 / 8) else 0

theorem probability_of_both_red_is_one_sixth :
  probability_both_red 4 3 2 2 = 1 / 6 :=
by
  unfold probability_both_red
  split_ifs
  · sorry
  · contradiction

end NUMINAMATH_GPT_probability_of_both_red_is_one_sixth_l411_41110


namespace NUMINAMATH_GPT_evaluate_expression_l411_41145

theorem evaluate_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l411_41145


namespace NUMINAMATH_GPT_Lindas_savings_l411_41165

theorem Lindas_savings (S : ℝ) (h1 : (1/3) * S = 250) : S = 750 := 
by
  sorry

end NUMINAMATH_GPT_Lindas_savings_l411_41165


namespace NUMINAMATH_GPT_makarala_meetings_percentage_l411_41141

def work_day_to_minutes (hours: ℕ) : ℕ :=
  60 * hours

def total_meeting_time (first: ℕ) (second: ℕ) : ℕ :=
  let third := first + second
  first + second + third

def percentage_of_day_spent (meeting_time: ℕ) (work_day_time: ℕ) : ℚ :=
  (meeting_time : ℚ) / (work_day_time : ℚ) * 100

theorem makarala_meetings_percentage
  (work_hours: ℕ)
  (first_meeting: ℕ)
  (second_meeting: ℕ)
  : percentage_of_day_spent (total_meeting_time first_meeting second_meeting) (work_day_to_minutes work_hours) = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_makarala_meetings_percentage_l411_41141


namespace NUMINAMATH_GPT_base8_subtraction_correct_l411_41150

-- Define what it means to subtract in base 8
def base8_sub (a b : ℕ) : ℕ :=
  let a_base10 := 8 * (a / 10) + (a % 10)
  let b_base10 := 8 * (b / 10) + (b % 10)
  let result_base10 := a_base10 - b_base10
  8 * (result_base10 / 8) + (result_base10 % 8)

-- The given numbers in base 8
def num1 : ℕ := 52
def num2 : ℕ := 31
def expected_result : ℕ := 21

-- The proof problem statement
theorem base8_subtraction_correct : base8_sub num1 num2 = expected_result := by
  sorry

end NUMINAMATH_GPT_base8_subtraction_correct_l411_41150


namespace NUMINAMATH_GPT_curves_intersect_at_4_points_l411_41193

theorem curves_intersect_at_4_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = a^2 ∧ y = x^2 - a → ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x3, y3) ≠ (x4, y4) ∧
  (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x4, y4) ∧
  (x4, y4) ≠ (x3, y3) ∧ x1^2 + (y1 - 1)^2 = a^2 ∧ y1 = x1^2 - a ∧
  x2^2 + (y2 - 1)^2 = a^2 ∧ y2 = x2^2 - a ∧
  x3^2 + (y3 - 1)^2 = a^2 ∧ y3 = x3^2 - a ∧
  x4^2 + (y4 - 1)^2 = a^2 ∧ y4 = x4^2 - a) ↔ a > 0 :=
sorry

end NUMINAMATH_GPT_curves_intersect_at_4_points_l411_41193


namespace NUMINAMATH_GPT_part1_daily_sales_profit_at_60_part2_selling_price_1350_l411_41129

-- Definitions from conditions
def cost_per_piece : ℕ := 40
def selling_price_50_sales_volume : ℕ := 100
def sales_decrease_per_dollar : ℕ := 2
def max_selling_price : ℕ := 65

-- Problem Part (1)
def profit_at_60_yuan := 
  let selling_price := 60
  let profit_per_piece := selling_price - cost_per_piece
  let sales_decrease := (selling_price - 50) * sales_decrease_per_dollar
  let sales_volume := selling_price_50_sales_volume - sales_decrease
  let daily_profit := profit_per_piece * sales_volume
  daily_profit

theorem part1_daily_sales_profit_at_60 : profit_at_60_yuan = 1600 := by
  sorry

-- Problem Part (2)
def selling_price_for_1350_profit :=
  let desired_profit := 1350
  let sales_volume (x : ℕ) := selling_price_50_sales_volume - sales_decrease_per_dollar * (x - 50)
  let profit_per_x_piece (x : ℕ) := x - cost_per_piece
  let daily_sales_profit (x : ℕ) := (profit_per_x_piece x) * (sales_volume x)
  daily_sales_profit

theorem part2_selling_price_1350 : 
  ∃ x, x ≤ max_selling_price ∧ selling_price_for_1350_profit x = 1350 ∧ x = 55 := by
  sorry

end NUMINAMATH_GPT_part1_daily_sales_profit_at_60_part2_selling_price_1350_l411_41129


namespace NUMINAMATH_GPT_compute_a_b_difference_square_l411_41104

noncomputable def count_multiples (m n : ℕ) : ℕ :=
  (n - 1) / m

theorem compute_a_b_difference_square :
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  (a - b) ^ 2 = 0 :=
by
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  show (a - b) ^ 2 = 0
  sorry

end NUMINAMATH_GPT_compute_a_b_difference_square_l411_41104


namespace NUMINAMATH_GPT_simple_interest_correct_l411_41132

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end NUMINAMATH_GPT_simple_interest_correct_l411_41132


namespace NUMINAMATH_GPT_restaurant_discount_l411_41161

theorem restaurant_discount :
  let coffee_price := 6
  let cheesecake_price := 10
  let discount_rate := 0.25
  let total_price := coffee_price + cheesecake_price
  let discount := discount_rate * total_price
  let final_price := total_price - discount
  final_price = 12 := by
  sorry

end NUMINAMATH_GPT_restaurant_discount_l411_41161


namespace NUMINAMATH_GPT_stan_water_intake_l411_41100

-- Define the constants and parameters given in the conditions
def words_per_minute : ℕ := 50
def pages : ℕ := 5
def words_per_page : ℕ := 400
def water_per_hour : ℚ := 15  -- use rational numbers for precise division

-- Define the derived quantities from the conditions
def total_words : ℕ := pages * words_per_page
def total_minutes : ℕ := total_words / words_per_minute
def water_per_minute : ℚ := water_per_hour / 60

-- State the theorem
theorem stan_water_intake : 10 = total_minutes * water_per_minute := by
  sorry

end NUMINAMATH_GPT_stan_water_intake_l411_41100


namespace NUMINAMATH_GPT_ball_colors_l411_41122

theorem ball_colors (R G B : ℕ) (h1 : R + G + B = 15) (h2 : B = R + 1) (h3 : R = G) (h4 : B = G + 5) : false :=
by
  sorry

end NUMINAMATH_GPT_ball_colors_l411_41122


namespace NUMINAMATH_GPT_maximize_x3y4_l411_41112

noncomputable def maximize_expr (x y : ℝ) : ℝ :=
x^3 * y^4

theorem maximize_x3y4 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 50 ∧ maximize_expr x y = maximize_expr 30 20 :=
by
  sorry

end NUMINAMATH_GPT_maximize_x3y4_l411_41112


namespace NUMINAMATH_GPT_verify_differential_eq_l411_41125

noncomputable def y (x : ℝ) : ℝ := (2 + 3 * x - 3 * x^2)^(1 / 3 : ℝ)
noncomputable def y_prime (x : ℝ) : ℝ := 
  1 / 3 * (2 + 3 * x - 3 * x^2)^(-2 / 3 : ℝ) * (3 - 6 * x)

theorem verify_differential_eq (x : ℝ) :
  y x * y_prime x = (1 - 2 * x) / y x :=
by
  sorry

end NUMINAMATH_GPT_verify_differential_eq_l411_41125


namespace NUMINAMATH_GPT_square_side_length_l411_41164

theorem square_side_length (a b : ℕ) (h : a = 9) (h' : b = 16) (A : ℕ) (h1: A = a * b) :
  ∃ (s : ℕ), s * s = A ∧ s = 12 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l411_41164


namespace NUMINAMATH_GPT_find_y_l411_41131

theorem find_y (y : ℕ) (h : 2^10 = 32^y) : y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l411_41131


namespace NUMINAMATH_GPT_base_radius_of_cone_l411_41159

-- Definitions of the conditions
def R1 : ℕ := 5
def R2 : ℕ := 4
def R3 : ℕ := 4
def height_radius_ratio := 4 / 3

-- Main theorem statement
theorem base_radius_of_cone : 
  (R1 = 5) → (R2 = 4) → (R3 = 4) → (height_radius_ratio = 4 / 3) → 
  ∃ r : ℚ, r = 169 / 60 :=
by 
  intros hR1 hR2 hR3 hRatio
  sorry

end NUMINAMATH_GPT_base_radius_of_cone_l411_41159


namespace NUMINAMATH_GPT_product_of_real_values_r_l411_41175

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end NUMINAMATH_GPT_product_of_real_values_r_l411_41175


namespace NUMINAMATH_GPT_marsupial_protein_l411_41128

theorem marsupial_protein (absorbed : ℝ) (percent_absorbed : ℝ) (consumed : ℝ) :
  absorbed = 16 ∧ percent_absorbed = 0.4 → consumed = 40 :=
by
  sorry

end NUMINAMATH_GPT_marsupial_protein_l411_41128


namespace NUMINAMATH_GPT_value_of_expression_l411_41105

theorem value_of_expression : (2207 - 2024)^2 * 4 / 144 = 930.25 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l411_41105


namespace NUMINAMATH_GPT_length_of_jordans_rectangle_l411_41134

theorem length_of_jordans_rectangle 
  (h1 : ∃ (length width : ℕ), length = 5 ∧ width = 24) 
  (h2 : ∃ (width_area : ℕ), width_area = 30 ∧ ∃ (area : ℕ), area = 5 * 24 ∧ ∃ (L : ℕ), area = L * width_area) :
  ∃ L, L = 4 := by 
  sorry

end NUMINAMATH_GPT_length_of_jordans_rectangle_l411_41134


namespace NUMINAMATH_GPT_infinite_set_k_l411_41142

theorem infinite_set_k (C : ℝ) : ∃ᶠ k : ℤ in at_top, (k : ℝ) * Real.sin k > C :=
sorry

end NUMINAMATH_GPT_infinite_set_k_l411_41142


namespace NUMINAMATH_GPT_sum_of_squares_nonzero_l411_41136

theorem sum_of_squares_nonzero {a b : ℝ} (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_nonzero_l411_41136


namespace NUMINAMATH_GPT_perimeter_of_plot_is_340_l411_41109

def width : ℝ := 80 -- Derived width from the given conditions
def length (w : ℝ) : ℝ := w + 10 -- Length is 10 meters more than width
def perimeter (w : ℝ) : ℝ := 2 * (w + length w) -- Perimeter of the rectangle
def cost_per_meter : ℝ := 6.5 -- Cost rate per meter
def total_cost : ℝ := 2210 -- Total cost given

theorem perimeter_of_plot_is_340 :
  cost_per_meter * perimeter width = total_cost → perimeter width = 340 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_plot_is_340_l411_41109


namespace NUMINAMATH_GPT_problem_inequality_l411_41127

variable {x y : ℝ}

theorem problem_inequality (hx : 2 < x) (hy : 2 < y) : 
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2 / 3 := 
  sorry

end NUMINAMATH_GPT_problem_inequality_l411_41127


namespace NUMINAMATH_GPT_percentage_difference_highest_lowest_salary_l411_41116

variables (R : ℝ)
def Ram_salary := 1.25 * R
def Simran_salary := 0.85 * R
def Rahul_salary := 0.85 * R * 1.10

theorem percentage_difference_highest_lowest_salary :
  let highest_salary := Ram_salary R
  let lowest_salary := Simran_salary R
  (highest_salary ≠ 0) → ((highest_salary - lowest_salary) / highest_salary) * 100 = 32 :=
by
  intros
  -- Sorry in place of proof
  sorry

end NUMINAMATH_GPT_percentage_difference_highest_lowest_salary_l411_41116


namespace NUMINAMATH_GPT_find_rate_percent_l411_41197

theorem find_rate_percent (P : ℝ) (r : ℝ) (A1 A2 : ℝ) (t1 t2 : ℕ)
  (h1 : A1 = P * (1 + r)^t1) (h2 : A2 = P * (1 + r)^t2) (hA1 : A1 = 2420) (hA2 : A2 = 3146) (ht1 : t1 = 2) (ht2 : t2 = 3) :
  r = 0.2992 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l411_41197


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l411_41108

def parabola (y : ℝ) : ℝ := y^2
def line (m : ℝ) (y : ℝ) : ℝ := m * y + 1

theorem sufficient_but_not_necessary_condition {m : ℝ} :
  (m ≠ 0) → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ parabola y1 = line m y1 ∧ parabola y2 = line m y2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l411_41108


namespace NUMINAMATH_GPT_maple_tree_total_l411_41189

-- Conditions
def initial_maple_trees : ℕ := 53
def trees_planted_today : ℕ := 11

-- Theorem to prove the result
theorem maple_tree_total : initial_maple_trees + trees_planted_today = 64 := by
  sorry

end NUMINAMATH_GPT_maple_tree_total_l411_41189


namespace NUMINAMATH_GPT_tangent_parallel_l411_41176

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

theorem tangent_parallel (a : ℝ) (H : ∀ x1 : ℝ, ∃ x2 : ℝ, (a - 2 * Real.sin x1) = (-Real.exp x2 - 1)) :
  a < -3 := by
  sorry

end NUMINAMATH_GPT_tangent_parallel_l411_41176


namespace NUMINAMATH_GPT_q_investment_time_l411_41143

theorem q_investment_time (x t : ℝ)
  (h1 : (7 * 20 * x) / (5 * t * x) = 7 / 10) : t = 40 :=
by
  sorry

end NUMINAMATH_GPT_q_investment_time_l411_41143


namespace NUMINAMATH_GPT_sin_angle_GAC_correct_l411_41153

noncomputable def sin_angle_GAC (AB AD AE : ℝ) := 
  let AC := Real.sqrt (AB^2 + AD^2)
  let AG := Real.sqrt (AB^2 + AD^2 + AE^2)
  (AC / AG)

theorem sin_angle_GAC_correct : sin_angle_GAC 2 3 4 = Real.sqrt 377 / 29 := by
  sorry

end NUMINAMATH_GPT_sin_angle_GAC_correct_l411_41153


namespace NUMINAMATH_GPT_max_m_plus_n_l411_41194

theorem max_m_plus_n (m n : ℝ) (h : n = -m^2 - 3*m + 3) : m + n ≤ 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_m_plus_n_l411_41194


namespace NUMINAMATH_GPT_factor_expression_l411_41115

theorem factor_expression (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) /
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) =
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l411_41115


namespace NUMINAMATH_GPT_slopes_of_line_intersecting_ellipse_l411_41180

noncomputable def possible_slopes : Set ℝ := {m : ℝ | m ≤ -1/Real.sqrt 20 ∨ m ≥ 1/Real.sqrt 20}

theorem slopes_of_line_intersecting_ellipse (m : ℝ) (h : ∃ x y, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) : 
  m ∈ possible_slopes :=
sorry

end NUMINAMATH_GPT_slopes_of_line_intersecting_ellipse_l411_41180


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_43_l411_41186

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_43_l411_41186


namespace NUMINAMATH_GPT_pq_r_zero_l411_41160

theorem pq_r_zero (p q r : ℝ) : 
  (∀ x : ℝ, x^4 + 6 * x^3 + 4 * p * x^2 + 2 * q * x + r = (x^3 + 4 * x^2 + 2 * x + 1) * (x - 2)) → 
  (p + q) * r = 0 :=
by
  sorry

end NUMINAMATH_GPT_pq_r_zero_l411_41160


namespace NUMINAMATH_GPT_range_a_of_tangents_coincide_l411_41151

theorem range_a_of_tangents_coincide (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (a : ℝ)
  (h3 : -1 / (x2 ^ 2) = 2 * x1 + 1) (h4 : x1 ^ 2 = -a) :
  1/4 < a ∧ a < 1 :=
by
  sorry 

end NUMINAMATH_GPT_range_a_of_tangents_coincide_l411_41151


namespace NUMINAMATH_GPT_max_lateral_surface_area_cylinder_optimizes_l411_41123

noncomputable def max_lateral_surface_area_cylinder (r m : ℝ) : ℝ × ℝ :=
  let r_c := r / 2
  let h_c := m / 2
  (r_c, h_c)

theorem max_lateral_surface_area_cylinder_optimizes {r m : ℝ} (hr : 0 < r) (hm : 0 < m) :
  let (r_c, h_c) := max_lateral_surface_area_cylinder r m
  r_c = r / 2 ∧ h_c = m / 2 :=
sorry

end NUMINAMATH_GPT_max_lateral_surface_area_cylinder_optimizes_l411_41123


namespace NUMINAMATH_GPT_cups_filled_with_tea_l411_41149

theorem cups_filled_with_tea (total_tea ml_each_cup : ℕ)
  (h1 : total_tea = 1050)
  (h2 : ml_each_cup = 65) :
  total_tea / ml_each_cup = 16 := sorry

end NUMINAMATH_GPT_cups_filled_with_tea_l411_41149


namespace NUMINAMATH_GPT_greatest_of_5_consecutive_integers_l411_41173

theorem greatest_of_5_consecutive_integers (m n : ℤ) (h : 5 * n + 10 = m^3) : (n + 4) = 202 := by
sorry

end NUMINAMATH_GPT_greatest_of_5_consecutive_integers_l411_41173


namespace NUMINAMATH_GPT_area_of_tangency_triangle_l411_41185

theorem area_of_tangency_triangle (c a b T varrho : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_area : T = (1/2) * a * b) (h_inradius : varrho = (a + b - c) / 2) :
  (area_tangency : ℝ) = (varrho / c) * T :=
sorry

end NUMINAMATH_GPT_area_of_tangency_triangle_l411_41185


namespace NUMINAMATH_GPT_solve_for_k_l411_41187

theorem solve_for_k (x y k : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : 5 * x - k * y - 7 = 0) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l411_41187


namespace NUMINAMATH_GPT_cos_plus_sin_l411_41155

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end NUMINAMATH_GPT_cos_plus_sin_l411_41155


namespace NUMINAMATH_GPT_smallest_omega_l411_41158

theorem smallest_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω = 6 * k) ∧ (∀ k : ℤ, k > 0 → ω = 6 * k → ω = 6) :=
by sorry

end NUMINAMATH_GPT_smallest_omega_l411_41158


namespace NUMINAMATH_GPT_algebraic_identity_l411_41167

theorem algebraic_identity (x : ℝ) (h : x = Real.sqrt 3 + 2) : x^2 - 4 * x + 3 = 2 := 
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_algebraic_identity_l411_41167


namespace NUMINAMATH_GPT_yellow_chip_count_l411_41170

def point_values_equation (Y B G R : ℕ) : Prop :=
  2 ^ Y * 4 ^ B * 5 ^ G * 7 ^ R = 560000

theorem yellow_chip_count (Y B G R : ℕ) (h1 : B = 2 * G) (h2 : R = B / 2) (h3 : point_values_equation Y B G R) :
  Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_yellow_chip_count_l411_41170


namespace NUMINAMATH_GPT_sum_four_variables_l411_41140

theorem sum_four_variables 
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + 2 = x)
  (h2 : b + 3 = x)
  (h3 : c + 4 = x)
  (h4 : d + 5 = x)
  (h5 : a + b + c + d + 8 = x) :
  a + b + c + d = -6 :=
by
  sorry

end NUMINAMATH_GPT_sum_four_variables_l411_41140


namespace NUMINAMATH_GPT_not_a_function_l411_41113

theorem not_a_function (angle_sine : ℝ → ℝ) 
                       (side_length_area : ℝ → ℝ) 
                       (sides_sum_int_angles : ℕ → ℝ)
                       (person_age_height : ℕ → Set ℝ) :
  (∃ y₁ y₂, y₁ ∈ person_age_height 20 ∧ y₂ ∈ person_age_height 20 ∧ y₁ ≠ y₂) :=
by {
  sorry
}

end NUMINAMATH_GPT_not_a_function_l411_41113


namespace NUMINAMATH_GPT_solve_for_x_l411_41119

variable (a b x : ℝ)

def operation (a b : ℝ) : ℝ := (a + 5) * b

theorem solve_for_x (h : operation x 1.3 = 11.05) : x = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l411_41119


namespace NUMINAMATH_GPT_income_is_108000_l411_41196

theorem income_is_108000 (S I : ℝ) (h1 : S / I = 5 / 9) (h2 : 48000 = I - S) : I = 108000 :=
by
  sorry

end NUMINAMATH_GPT_income_is_108000_l411_41196


namespace NUMINAMATH_GPT_geometry_progressions_not_exhaust_nat_l411_41166

theorem geometry_progressions_not_exhaust_nat :
  ∃ (g : Fin 1975 → ℕ → ℕ), 
  (∀ i : Fin 1975, ∃ (a r : ℤ), ∀ n : ℕ, g i n = (a * r^n)) ∧
  (∃ m : ℕ, ∀ i : Fin 1975, ∀ n : ℕ, m ≠ g i n) :=
sorry

end NUMINAMATH_GPT_geometry_progressions_not_exhaust_nat_l411_41166


namespace NUMINAMATH_GPT_sasha_mistake_l411_41163

/-- If Sasha obtained three numbers by raising 4 to various powers, such that all three units digits are different, 
     then Sasha's numbers cannot have three distinct last digits. -/
theorem sasha_mistake (h : ∀ n1 n2 n3 : ℕ, ∃ k1 k2 k3, n1 = 4^k1 ∧ n2 = 4^k2 ∧ n3 = 4^k3 ∧ (n1 % 10 ≠ n2 % 10) ∧ (n2 % 10 ≠ n3 % 10) ∧ (n1 % 10 ≠ n3 % 10)) :
False :=
sorry

end NUMINAMATH_GPT_sasha_mistake_l411_41163


namespace NUMINAMATH_GPT_hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l411_41130

-- Definitions for shooting events for clarity
def hits_9_rings (s : String) := s = "9 rings"
def hits_8_rings (s : String) := s = "8 rings"

def hits_10_rings (s : String) := s = "10 rings"

def hits_target (s: String) := s = "hits target"
def does_not_hit_target (s: String) := s = "does not hit target"

-- Mutual exclusivity:
def mutually_exclusive (E1 E2 : Prop) := ¬ (E1 ∧ E2)

-- Problem 1:
theorem hits_9_and_8_mutually_exclusive :
  mutually_exclusive (hits_9_rings "9 rings") (hits_8_rings "8 rings") :=
sorry

-- Problem 2:
theorem hits_10_and_8_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_10_rings "10 rings" ) (hits_8_rings "8 rings") :=
sorry

-- Problem 3:
theorem both_hit_target_and_neither_hit_target_mutually_exclusive :
  mutually_exclusive (hits_target "both hit target") (does_not_hit_target "neither hit target") :=
sorry

-- Problem 4:
theorem at_least_one_hits_and_A_not_B_does_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_target "at least one hits target") (does_not_hit_target "A not but B does hit target") :=
sorry

end NUMINAMATH_GPT_hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l411_41130


namespace NUMINAMATH_GPT_max_divisor_of_expression_l411_41183

theorem max_divisor_of_expression 
  (n : ℕ) (hn : n > 0) : ∃ k, k = 8 ∧ 8 ∣ (5^n + 2 * 3^(n-1) + 1) :=
by
  sorry

end NUMINAMATH_GPT_max_divisor_of_expression_l411_41183


namespace NUMINAMATH_GPT_burn_rate_walking_l411_41188

def burn_rate_running : ℕ := 10
def total_calories : ℕ := 450
def total_time : ℕ := 60
def running_time : ℕ := 35

theorem burn_rate_walking :
  ∃ (W : ℕ), ((running_time * burn_rate_running) + ((total_time - running_time) * W) = total_calories) ∧ (W = 4) :=
by
  sorry

end NUMINAMATH_GPT_burn_rate_walking_l411_41188


namespace NUMINAMATH_GPT_equation_of_line_perpendicular_and_passing_point_l411_41144

theorem equation_of_line_perpendicular_and_passing_point :
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = -1 ∧
  (∀ (x y : ℝ), (2 * x - 3 * y + 4 = 0 → y = (2 / 3) * x + 4 / 3) →
  (∀ (x1 y1 : ℝ), x1 = -1 ∧ y1 = 2 →
  (a * x1 + b * y1 + c = 0) ∧
  (∀ (x y : ℝ), (-3 / 2) * (x + 1) + 2 = y) →
  (a * x + b * y + c = 0))) :=
sorry

end NUMINAMATH_GPT_equation_of_line_perpendicular_and_passing_point_l411_41144


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l411_41174

theorem sufficient_but_not_necessary_condition 
    (a : ℝ) (h_pos : a > 0)
    (h_line : ∀ x y, 2 * a * x - y + 2 * a^2 = 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / 4 = 1) :
    (a ≥ 2) → 
    (∀ x y, ¬ (2 * a * x - y + 2 * a^2 = 0 ∧ x^2 / a^2 - y^2 / 4 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l411_41174


namespace NUMINAMATH_GPT_polygon_sides_count_l411_41121

-- Definitions for each polygon and their sides
def pentagon_sides := 5
def square_sides := 4
def hexagon_sides := 6
def heptagon_sides := 7
def nonagon_sides := 9

-- Compute the total number of sides
def total_exposed_sides :=
  (pentagon_sides + nonagon_sides - 2) + (square_sides + hexagon_sides + heptagon_sides - 6)

theorem polygon_sides_count : total_exposed_sides = 23 :=
by
  -- Mathematical proof steps can be detailed here
  -- For now, let's assume it is correctly given as a single number
  sorry

end NUMINAMATH_GPT_polygon_sides_count_l411_41121


namespace NUMINAMATH_GPT_root_of_inverse_f_plus_x_eq_k_l411_41107

variable {α : Type*} [Nonempty α] [Field α]
variable (f : α → α)
variable (f_inv : α → α)
variable (k : α)

def root_of_f_plus_x_eq_k (x : α) : Prop :=
  f x + x = k

def inverse_function (f : α → α) (f_inv : α → α) : Prop :=
  ∀ y : α, f (f_inv y) = y ∧ f_inv (f y) = y

theorem root_of_inverse_f_plus_x_eq_k
  (h1 : root_of_f_plus_x_eq_k f 5 k)
  (h2 : inverse_function f f_inv) :
  f_inv (k - 5) + (k - 5) = k :=
by
  sorry

end NUMINAMATH_GPT_root_of_inverse_f_plus_x_eq_k_l411_41107


namespace NUMINAMATH_GPT_soil_bags_needed_l411_41154

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end NUMINAMATH_GPT_soil_bags_needed_l411_41154


namespace NUMINAMATH_GPT_total_students_l411_41139

theorem total_students (S : ℕ) (R : ℕ) :
  (2 * 0 + 12 * 1 + 13 * 2 + R * 3) / S = 2 →
  2 + 12 + 13 + R = S →
  S = 43 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l411_41139


namespace NUMINAMATH_GPT_fraction_of_3_5_eq_2_15_l411_41102

theorem fraction_of_3_5_eq_2_15 : (2 / 15) / (3 / 5) = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_of_3_5_eq_2_15_l411_41102


namespace NUMINAMATH_GPT_no_preimage_range_l411_41169

open Set

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem no_preimage_range :
  { k : ℝ | ∀ x : ℝ, f x ≠ k } = Iio 2 := by
  sorry

end NUMINAMATH_GPT_no_preimage_range_l411_41169


namespace NUMINAMATH_GPT_drawn_from_grade12_correct_l411_41199

-- Variables for the conditions
variable (total_students : ℕ) (sample_size : ℕ) (grade10_students : ℕ) 
          (grade11_students : ℕ) (grade12_students : ℕ) (drawn_from_grade12 : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 2400 ∧
  sample_size = 120 ∧
  grade10_students = 820 ∧
  grade11_students = 780 ∧
  grade12_students = total_students - grade10_students - grade11_students ∧
  drawn_from_grade12 = (grade12_students * sample_size) / total_students

-- Theorem to prove
theorem drawn_from_grade12_correct : conditions total_students sample_size grade10_students grade11_students grade12_students drawn_from_grade12 → drawn_from_grade12 = 40 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end NUMINAMATH_GPT_drawn_from_grade12_correct_l411_41199


namespace NUMINAMATH_GPT_price_of_each_book_l411_41192

theorem price_of_each_book (B P : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = 36) -- Number of unsold books is 1/3 of the total books and it equals 36
  (h2 : (2 / 3 : ℚ) * B * P = 144) -- Total amount received for the books sold is $144
  : P = 2 := 
by
  sorry

end NUMINAMATH_GPT_price_of_each_book_l411_41192


namespace NUMINAMATH_GPT_radical_axis_theorem_l411_41101

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def power_of_point (p : Point) (c : Circle) : ℝ :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2 - c.radius^2)

theorem radical_axis_theorem (O1 O2 : Circle) :
  ∃ L : ℝ → Point, 
  (∀ p : Point, (power_of_point p O1 = power_of_point p O2) → (L p.x = p)) ∧ 
  (O1.center.y = O2.center.y) ∧ 
  (∃ k : ℝ, ∀ x, L x = Point.mk x k) :=
sorry

end NUMINAMATH_GPT_radical_axis_theorem_l411_41101


namespace NUMINAMATH_GPT_sqrt_inequality_l411_41117

theorem sqrt_inequality (x : ℝ) : abs ((x^2 - 9) / 3) < 3 ↔ -Real.sqrt 18 < x ∧ x < Real.sqrt 18 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_inequality_l411_41117


namespace NUMINAMATH_GPT_illiterate_employee_count_l411_41198

variable (I : ℕ) -- Number of illiterate employees
variable (literate_count : ℕ) -- Number of literate employees
variable (initial_wage_illiterate : ℕ) -- Initial average wage of illiterate employees
variable (new_wage_illiterate : ℕ) -- New average wage of illiterate employees
variable (average_salary_decrease : ℕ) -- Decrease in the average salary of all employees

-- Given conditions:
def condition1 : initial_wage_illiterate = 25 := by sorry
def condition2 : new_wage_illiterate = 10 := by sorry
def condition3 : average_salary_decrease = 10 := by sorry
def condition4 : literate_count = 10 := by sorry

-- Main proof statement:
theorem illiterate_employee_count :
  initial_wage_illiterate - new_wage_illiterate = 15 →
  average_salary_decrease * (literate_count + I) = (initial_wage_illiterate - new_wage_illiterate) * I →
  I = 20 := by
  intros h1 h2
  -- provided conditions
  exact sorry

end NUMINAMATH_GPT_illiterate_employee_count_l411_41198


namespace NUMINAMATH_GPT_maximum_point_of_f_l411_41178

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x - 2) * Real.exp x

theorem maximum_point_of_f : ∃ x : ℝ, x = -2 ∧
  ∀ y : ℝ, f y ≤ f x :=
sorry

end NUMINAMATH_GPT_maximum_point_of_f_l411_41178


namespace NUMINAMATH_GPT_area_range_of_triangle_l411_41106

-- Defining the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 2

-- Function to compute the area of triangle ABP
noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2))

-- The proof goal statement
theorem area_range_of_triangle (P : ℝ × ℝ) (hp : on_circle P) :
  2 ≤ area_of_triangle P ∧ area_of_triangle P ≤ 6 :=
sorry

end NUMINAMATH_GPT_area_range_of_triangle_l411_41106


namespace NUMINAMATH_GPT_compound_interest_calculation_l411_41181

-- Given conditions
def P : ℝ := 20000
def r : ℝ := 0.03
def t : ℕ := 5

-- The amount after t years with compound interest
def A := P * (1 + r) ^ t

-- Prove the total amount is as given in choice B
theorem compound_interest_calculation : 
  A = 20000 * (1 + 0.03) ^ 5 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l411_41181


namespace NUMINAMATH_GPT_beacon_population_l411_41111

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_beacon_population_l411_41111


namespace NUMINAMATH_GPT_number_of_distinct_real_roots_l411_41148

theorem number_of_distinct_real_roots (k : ℕ) :
  (∃ k : ℕ, ∀ x : ℝ, |x| - 4 = (3 * |x|) / 2 → 0 = k) :=
  sorry

end NUMINAMATH_GPT_number_of_distinct_real_roots_l411_41148


namespace NUMINAMATH_GPT_solution_set_of_inequality_l411_41138

theorem solution_set_of_inequality : {x : ℝ | 8 * x^2 + 6 * x ≤ 2} = { x : ℝ | -1 ≤ x ∧ x ≤ (1/4) } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l411_41138


namespace NUMINAMATH_GPT_fifth_eqn_nth_eqn_l411_41182

theorem fifth_eqn : 10 * 12 + 1 = 121 :=
by
  sorry

theorem nth_eqn (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_fifth_eqn_nth_eqn_l411_41182


namespace NUMINAMATH_GPT_tank_fills_in_56_minutes_l411_41152

theorem tank_fills_in_56_minutes : 
  (∃ A B C : ℕ, (A = 40 ∧ B = 30 ∧ C = 20) ∧ 
                 ∃ capacity : ℕ, capacity = 950 ∧ 
                 ∃ time : ℕ, time = 56 ∧
                 ∀ cycle_time : ℕ, cycle_time = 3 ∧ 
                 ∀ net_water_per_cycle : ℕ, net_water_per_cycle = A + B - C ∧
                 ∀ total_cycles : ℕ, total_cycles = capacity / net_water_per_cycle ∧
                 ∀ total_time : ℕ, total_time = total_cycles * cycle_time - 1 ∧
                 total_time = time) :=
sorry

end NUMINAMATH_GPT_tank_fills_in_56_minutes_l411_41152


namespace NUMINAMATH_GPT_train_length_correct_l411_41126

-- Define the conditions
def train_speed : ℝ := 63
def time_crossing : ℝ := 40
def expected_length : ℝ := 2520

-- The statement to prove
theorem train_length_correct : train_speed * time_crossing = expected_length :=
by
  exact sorry

end NUMINAMATH_GPT_train_length_correct_l411_41126


namespace NUMINAMATH_GPT_cone_ratio_l411_41184

noncomputable def cone_height_ratio : ℚ :=
  let original_height := 40
  let circumference := 24 * Real.pi
  let original_radius := 12
  let new_volume := 432 * Real.pi
  let new_height := 9
  new_height / original_height

theorem cone_ratio (h : cone_height_ratio = 9 / 40) : (9 : ℚ) / 40 = 9 / 40 := by
  sorry

end NUMINAMATH_GPT_cone_ratio_l411_41184


namespace NUMINAMATH_GPT_aurelia_percentage_l411_41157

variables (P : ℝ)

theorem aurelia_percentage (h1 : 2000 + (P / 100) * 2000 = 3400) : 
  P = 70 :=
by
  sorry

end NUMINAMATH_GPT_aurelia_percentage_l411_41157


namespace NUMINAMATH_GPT_mean_of_three_digit_multiples_of_8_l411_41135

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end NUMINAMATH_GPT_mean_of_three_digit_multiples_of_8_l411_41135


namespace NUMINAMATH_GPT_largest_possible_s_l411_41195

theorem largest_possible_s (r s: ℕ) (h1: r ≥ s) (h2: s ≥ 3)
  (h3: (59 : ℚ) / 58 * (180 * (s - 2) / s) = (180 * (r - 2) / r)) : s = 117 :=
sorry

end NUMINAMATH_GPT_largest_possible_s_l411_41195


namespace NUMINAMATH_GPT_functional_equation_solution_l411_41156

def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intros h x
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l411_41156


namespace NUMINAMATH_GPT_mystery_number_addition_l411_41172

theorem mystery_number_addition (mystery_number : ℕ) (h : mystery_number = 47) : mystery_number + 45 = 92 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mystery_number_addition_l411_41172


namespace NUMINAMATH_GPT_middle_letter_value_l411_41114

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end NUMINAMATH_GPT_middle_letter_value_l411_41114


namespace NUMINAMATH_GPT_cos_4pi_over_3_l411_41146

theorem cos_4pi_over_3 : Real.cos (4 * Real.pi / 3) = -1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_cos_4pi_over_3_l411_41146
