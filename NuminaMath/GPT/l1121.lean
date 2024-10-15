import Mathlib

namespace NUMINAMATH_GPT_find_sum_of_m1_m2_l1121_112199

-- Define the quadratic equation and the conditions
def quadratic (m : ℂ) (x : ℂ) : ℂ := m * x^2 - (3 * m - 2) * x + 7

-- Define the roots a and b
def are_roots (m a b : ℂ) : Prop := quadratic m a = 0 ∧ quadratic m b = 0

-- The condition given in the problem
def root_condition (a b : ℂ) : Prop := a / b + b / a = 3 / 2

-- Main theorem to be proved
theorem find_sum_of_m1_m2 (m1 m2 a1 a2 b1 b2 : ℂ) 
  (h1 : are_roots m1 a1 b1) 
  (h2 : are_roots m2 a2 b2) 
  (hc1 : root_condition a1 b1) 
  (hc2 : root_condition a2 b2) : 
  m1 + m2 = 73 / 18 :=
by sorry

end NUMINAMATH_GPT_find_sum_of_m1_m2_l1121_112199


namespace NUMINAMATH_GPT_mean_of_five_numbers_l1121_112137

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_l1121_112137


namespace NUMINAMATH_GPT_quadratic_sum_r_s_l1121_112162

/-- Solve the quadratic equation and identify the sum of r and s 
from the equivalent completed square form (x + r)^2 = s. -/
theorem quadratic_sum_r_s (r s : ℤ) :
  (∃ r s : ℤ, (x - r)^2 = s → r + s = 11) :=
sorry

end NUMINAMATH_GPT_quadratic_sum_r_s_l1121_112162


namespace NUMINAMATH_GPT_maximum_xy_value_l1121_112108

theorem maximum_xy_value :
  ∃ (x y : ℕ), 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by
  sorry

end NUMINAMATH_GPT_maximum_xy_value_l1121_112108


namespace NUMINAMATH_GPT_find_coefficients_l1121_112110

theorem find_coefficients (a1 a2 : ℚ) :
  (4 * a1 + 5 * a2 = 9) ∧ (-a1 + 3 * a2 = 4) ↔ (a1 = 181 / 136) ∧ (a2 = 25 / 68) := 
sorry

end NUMINAMATH_GPT_find_coefficients_l1121_112110


namespace NUMINAMATH_GPT_find_alpha_l1121_112113

theorem find_alpha (α : ℝ) (h1 : Real.tan α = -1) (h2 : 0 < α ∧ α ≤ Real.pi) : α = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_alpha_l1121_112113


namespace NUMINAMATH_GPT_book_area_correct_l1121_112183

/-- Converts inches to centimeters -/
def inch_to_cm (inches : ℚ) : ℚ :=
  inches * 2.54

/-- The length of the book given a parameter x -/
def book_length (x : ℚ) : ℚ :=
  3 * x - 4

/-- The width of the book in inches -/
def book_width_in_inches : ℚ :=
  5 / 2

/-- The width of the book in centimeters -/
def book_width : ℚ :=
  inch_to_cm book_width_in_inches

/-- The area of the book given a parameter x -/
def book_area (x : ℚ) : ℚ :=
  book_length x * book_width

/-- Proof that the area of the book with x = 5 is 69.85 cm² -/
theorem book_area_correct : book_area 5 = 69.85 := by
  sorry

end NUMINAMATH_GPT_book_area_correct_l1121_112183


namespace NUMINAMATH_GPT_sally_picked_peaches_l1121_112100

variable (p_initial p_current p_picked : ℕ)

theorem sally_picked_peaches (h1 : p_initial = 13) (h2 : p_current = 55) :
  p_picked = p_current - p_initial → p_picked = 42 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sally_picked_peaches_l1121_112100


namespace NUMINAMATH_GPT_final_value_l1121_112151

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end NUMINAMATH_GPT_final_value_l1121_112151


namespace NUMINAMATH_GPT_find_f_one_l1121_112135

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f_defined_for_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 1

-- Statement that needs to be proven
theorem find_f_one (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : f_defined_for_neg f) :
  f 1 = -1 :=
  sorry

end NUMINAMATH_GPT_find_f_one_l1121_112135


namespace NUMINAMATH_GPT_min_value_inequality_l1121_112160

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l1121_112160


namespace NUMINAMATH_GPT_transform_to_A_plus_one_l1121_112143

theorem transform_to_A_plus_one (A : ℕ) (hA : A > 0) : 
  ∃ n : ℕ, (∀ i : ℕ, (i ≤ n) → ((A + 9 * i) = A + 1 ∨ ∃ j : ℕ, (A + 9 * i) = (A + 1 + 10 * j))) :=
sorry

end NUMINAMATH_GPT_transform_to_A_plus_one_l1121_112143


namespace NUMINAMATH_GPT_pablo_distributed_fraction_l1121_112173

-- Definitions based on the problem statement
def mia_coins (m : ℕ) := m
def sofia_coins (m : ℕ) := 3 * m
def pablo_coins (m : ℕ) := 12 * m

-- Condition for equal distribution
def target_coins (m : ℕ) := (mia_coins m + sofia_coins m + pablo_coins m) / 3

-- Needs for redistribution
def sofia_needs (m : ℕ) := target_coins m - sofia_coins m
def mia_needs (m : ℕ) := target_coins m - mia_coins m

-- Total distributed coins by Pablo
def total_distributed_by_pablo (m : ℕ) := sofia_needs m + mia_needs m

-- Fraction of coins Pablo distributes
noncomputable def fraction_distributed_by_pablo (m : ℕ) := (total_distributed_by_pablo m) / (pablo_coins m)

-- Theorem to prove
theorem pablo_distributed_fraction (m : ℕ) : fraction_distributed_by_pablo m = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_pablo_distributed_fraction_l1121_112173


namespace NUMINAMATH_GPT_find_S3_l1121_112178

-- Define the known scores
def S1 : ℕ := 55
def S2 : ℕ := 67
def S4 : ℕ := 55
def Avg : ℕ := 67

-- Statement to prove
theorem find_S3 : ∃ S3 : ℕ, (S1 + S2 + S3 + S4) / 4 = Avg ∧ S3 = 91 :=
by
  sorry

end NUMINAMATH_GPT_find_S3_l1121_112178


namespace NUMINAMATH_GPT_calculate_womans_haircut_cost_l1121_112187

-- Define the necessary constants and conditions
def W : ℝ := sorry
def child_haircut_cost : ℝ := 36
def tip_percentage : ℝ := 0.20
def total_tip : ℝ := 24
def number_of_children : ℕ := 2

-- Helper function to calculate total cost before the tip
def total_cost_before_tip (W : ℝ) (number_of_children : ℕ) (child_haircut_cost : ℝ) : ℝ :=
  W + number_of_children * child_haircut_cost

-- Lean statement for the main theorem
theorem calculate_womans_haircut_cost (W : ℝ) (child_haircut_cost : ℝ) (tip_percentage : ℝ)
  (total_tip : ℝ) (number_of_children : ℕ) :
  (tip_percentage * total_cost_before_tip W number_of_children child_haircut_cost) = total_tip →
  W = 48 :=
by
  sorry

end NUMINAMATH_GPT_calculate_womans_haircut_cost_l1121_112187


namespace NUMINAMATH_GPT_probability_AB_selected_l1121_112175

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_AB_selected_l1121_112175


namespace NUMINAMATH_GPT_lcm_of_4_5_6_9_is_180_l1121_112158

theorem lcm_of_4_5_6_9_is_180 : Nat.lcm (Nat.lcm 4 5) (Nat.lcm 6 9) = 180 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_4_5_6_9_is_180_l1121_112158


namespace NUMINAMATH_GPT_julia_total_balls_l1121_112171

theorem julia_total_balls :
  (3 * 19) + (10 * 19) + (8 * 19) = 399 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_julia_total_balls_l1121_112171


namespace NUMINAMATH_GPT_math_problem_l1121_112197

theorem math_problem (x y : Int)
  (hx : x = 2 - 4 + 6)
  (hy : y = 1 - 3 + 5) :
  x - y = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1121_112197


namespace NUMINAMATH_GPT_cos_585_eq_neg_sqrt2_div_2_l1121_112142

theorem cos_585_eq_neg_sqrt2_div_2 : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by sorry

end NUMINAMATH_GPT_cos_585_eq_neg_sqrt2_div_2_l1121_112142


namespace NUMINAMATH_GPT_graph_does_not_pass_through_quadrant_II_l1121_112119

noncomputable def linear_function (x : ℝ) : ℝ := 3 * x - 4

def passes_through_quadrant_I (x : ℝ) : Prop := x > 0 ∧ linear_function x > 0
def passes_through_quadrant_II (x : ℝ) : Prop := x < 0 ∧ linear_function x > 0
def passes_through_quadrant_III (x : ℝ) : Prop := x < 0 ∧ linear_function x < 0
def passes_through_quadrant_IV (x : ℝ) : Prop := x > 0 ∧ linear_function x < 0

theorem graph_does_not_pass_through_quadrant_II :
  ¬(∃ x : ℝ, passes_through_quadrant_II x) :=
sorry

end NUMINAMATH_GPT_graph_does_not_pass_through_quadrant_II_l1121_112119


namespace NUMINAMATH_GPT_ratio_of_sides_l1121_112141

theorem ratio_of_sides (s x y : ℝ) 
    (h1 : 0.1 * s^2 = 0.25 * x * y)
    (h2 : x = s / 10)
    (h3 : y = 4 * s) : x / y = 1 / 40 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l1121_112141


namespace NUMINAMATH_GPT_range_f_x_negative_l1121_112138

-- We define the conditions: f is an even function, increasing on (-∞, 0), and f(2) = 0.
variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 0 ∧ y < 0 → f x ≤ f y

def f_at_2_is_zero (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The theorem to be proven.
theorem range_f_x_negative (hf_even : even_function f)
  (hf_incr : increasing_on_neg_infinity_to_zero f)
  (hf_at2 : f_at_2_is_zero f) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 :=
by
  sorry

end NUMINAMATH_GPT_range_f_x_negative_l1121_112138


namespace NUMINAMATH_GPT_mistaken_quotient_is_35_l1121_112133

theorem mistaken_quotient_is_35 (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ) 
    (h1 : D = correct_divisor * correct_quotient)
    (h2 : correct_divisor = 21)
    (h3 : mistaken_divisor = 12)
    (h4 : correct_quotient = 20)
    : D / mistaken_divisor = 35 := by
  sorry

end NUMINAMATH_GPT_mistaken_quotient_is_35_l1121_112133


namespace NUMINAMATH_GPT_fraction_twins_l1121_112111

variables (P₀ I E P_f f : ℕ) (x : ℚ)

def initial_population := P₀ = 300000
def immigrants := I = 50000
def emigrants := E = 30000
def pregnant_fraction := f = 1 / 8
def final_population := P_f = 370000

theorem fraction_twins :
  initial_population P₀ ∧ immigrants I ∧ emigrants E ∧ pregnant_fraction f ∧ final_population P_f →
  x = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_twins_l1121_112111


namespace NUMINAMATH_GPT_unique_solution_for_quadratic_l1121_112130

theorem unique_solution_for_quadratic (a : ℝ) : 
  ∃! (x : ℝ), x^2 - 2 * a * x + a^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_for_quadratic_l1121_112130


namespace NUMINAMATH_GPT_jake_snakes_l1121_112176

theorem jake_snakes (S : ℕ) 
  (h1 : 2 * S + 1 = 6) 
  (h2 : 2250 = 5 * 250 + 1000) :
  S = 3 := 
by
  sorry

end NUMINAMATH_GPT_jake_snakes_l1121_112176


namespace NUMINAMATH_GPT_work_rate_l1121_112164

/-- 
A alone can finish a work in some days which B alone can finish in 15 days. 
If they work together and finish it, then out of a total wages of Rs. 3400, 
A will get Rs. 2040. Prove that A alone can finish the work in 22.5 days. 
-/
theorem work_rate (A : ℚ) (B_rate : ℚ) 
  (total_wages : ℚ) (A_wages : ℚ) 
  (total_rate : ℚ) 
  (hB : B_rate = 1 / 15) 
  (hWages : total_wages = 3400 ∧ A_wages = 2040) 
  (hTotal : total_rate = 1 / A + B_rate)
  (hWorkTogether : 
    (A_wages / (total_wages - A_wages) = 51 / 34) ↔ 
    (A / (A + 15) = 51 / 85)) : 
  A = 22.5 := 
sorry

end NUMINAMATH_GPT_work_rate_l1121_112164


namespace NUMINAMATH_GPT_negation_of_proposition_p_is_false_l1121_112109

variable (p : Prop)

theorem negation_of_proposition_p_is_false
  (h : ¬p) : ¬(¬p) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_p_is_false_l1121_112109


namespace NUMINAMATH_GPT_joan_seashells_left_l1121_112145

theorem joan_seashells_left (original_seashells : ℕ) (given_seashells : ℕ) (seashells_left : ℕ)
  (h1 : original_seashells = 70) (h2 : given_seashells = 43) : seashells_left = 27 :=
by
  sorry

end NUMINAMATH_GPT_joan_seashells_left_l1121_112145


namespace NUMINAMATH_GPT_find_min_max_l1121_112115

noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

theorem find_min_max :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 2 * Real.pi → 
    (0 ≤ f x y ∧ f x y ≤ 3 * Real.sqrt 3 / 2)) :=
sorry

end NUMINAMATH_GPT_find_min_max_l1121_112115


namespace NUMINAMATH_GPT_find_b_vector_l1121_112136

-- Define input vectors a, b, and their sum.
def vec_a : ℝ × ℝ × ℝ := (1, -2, 1)
def vec_b : ℝ × ℝ × ℝ := (-2, 4, -2)
def vec_sum : ℝ × ℝ × ℝ := (-1, 2, -1)

-- The theorem statement to prove that b is calculated correctly.
theorem find_b_vector :
  vec_a + vec_b = vec_sum →
  vec_b = (-2, 4, -2) :=
by
  sorry

end NUMINAMATH_GPT_find_b_vector_l1121_112136


namespace NUMINAMATH_GPT_travel_period_l1121_112170

-- Nina's travel pattern
def travels_in_one_month : ℕ := 400
def travels_in_two_months : ℕ := travels_in_one_month + 2 * travels_in_one_month

-- The total distance Nina wants to travel
def total_distance : ℕ := 14400

-- The period in months during which Nina travels the given total distance 
def required_period_in_months (d_per_2_months : ℕ) (total_d : ℕ) : ℕ := (total_d / d_per_2_months) * 2

-- Statement we need to prove
theorem travel_period : required_period_in_months travels_in_two_months total_distance = 24 := by
  sorry

end NUMINAMATH_GPT_travel_period_l1121_112170


namespace NUMINAMATH_GPT_calories_in_250g_of_lemonade_l1121_112196

structure Lemonade :=
(lemon_juice_grams : ℕ)
(sugar_grams : ℕ)
(water_grams : ℕ)
(lemon_juice_calories_per_100g : ℕ)
(sugar_calories_per_100g : ℕ)
(water_calories_per_100g : ℕ)

def calorie_count (l : Lemonade) : ℕ :=
(l.lemon_juice_grams * l.lemon_juice_calories_per_100g / 100) +
(l.sugar_grams * l.sugar_calories_per_100g / 100) +
(l.water_grams * l.water_calories_per_100g / 100)

def total_weight (l : Lemonade) : ℕ :=
l.lemon_juice_grams + l.sugar_grams + l.water_grams

def caloric_density (l : Lemonade) : ℚ :=
calorie_count l / total_weight l

theorem calories_in_250g_of_lemonade :
  ∀ (l : Lemonade), 
  l = { lemon_juice_grams := 200, sugar_grams := 300, water_grams := 500,
        lemon_juice_calories_per_100g := 40,
        sugar_calories_per_100g := 390,
        water_calories_per_100g := 0 } →
  (caloric_density l * 250 = 312.5) :=
sorry

end NUMINAMATH_GPT_calories_in_250g_of_lemonade_l1121_112196


namespace NUMINAMATH_GPT_find_positive_real_number_l1121_112144

theorem find_positive_real_number (x : ℝ) (hx : x = 25 + 2 * Real.sqrt 159) :
  1 / 2 * (3 * x ^ 2 - 1) = (x ^ 2 - 50 * x - 10) * (x ^ 2 + 25 * x + 5) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_real_number_l1121_112144


namespace NUMINAMATH_GPT_hour_hand_rotations_l1121_112169

theorem hour_hand_rotations (degrees_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) (rotations_per_day : ℕ) :
  degrees_per_hour = 30 →
  hours_per_day = 24 →
  rotations_per_day = (degrees_per_hour * hours_per_day) / 360 →
  days = 6 →
  rotations_per_day * days = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hour_hand_rotations_l1121_112169


namespace NUMINAMATH_GPT_tangent_line_at_origin_l1121_112122

noncomputable def f (x : ℝ) := Real.log (1 + x) + x * Real.exp (-x)

theorem tangent_line_at_origin : 
  ∀ (x : ℝ), (1 : ℝ) * x + (0 : ℝ) = 2 * x := 
sorry

end NUMINAMATH_GPT_tangent_line_at_origin_l1121_112122


namespace NUMINAMATH_GPT_value_of_k_l1121_112159

theorem value_of_k (k : ℝ) (x : ℝ) (h : (k - 3) * x^2 + 6 * x + k^2 - k = 0) (r : x = -1) : 
  k = -3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1121_112159


namespace NUMINAMATH_GPT_find_g5_l1121_112132

def g : ℤ → ℤ := sorry

axiom g_cond1 : g 1 > 1
axiom g_cond2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom g_cond3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 :=
by
  sorry

end NUMINAMATH_GPT_find_g5_l1121_112132


namespace NUMINAMATH_GPT_new_customers_needed_l1121_112156

theorem new_customers_needed 
  (initial_customers : ℕ)
  (customers_after_some_left : ℕ)
  (first_group_left : ℕ)
  (second_group_left : ℕ)
  (new_customers : ℕ)
  (h1 : initial_customers = 13)
  (h2 : customers_after_some_left = 9)
  (h3 : first_group_left = initial_customers - customers_after_some_left)
  (h4 : second_group_left = 8)
  (h5 : new_customers = first_group_left + second_group_left) :
  new_customers = 12 :=
by
  sorry

end NUMINAMATH_GPT_new_customers_needed_l1121_112156


namespace NUMINAMATH_GPT_cookie_cost_l1121_112177

variables (m o c : ℝ)
variables (H1 : m = 2 * o)
variables (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c)

theorem cookie_cost (H1 : m = 2 * o) (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c) : c = (13 / 4) * o :=
by sorry

end NUMINAMATH_GPT_cookie_cost_l1121_112177


namespace NUMINAMATH_GPT_necklace_price_l1121_112153

variable (N : ℝ)

def price_of_bracelet : ℝ := 15.00
def price_of_earring : ℝ := 10.00
def num_necklaces_sold : ℝ := 5
def num_bracelets_sold : ℝ := 10
def num_earrings_sold : ℝ := 20
def num_complete_ensembles_sold : ℝ := 2
def price_of_complete_ensemble : ℝ := 45.00
def total_amount_made : ℝ := 565.0

theorem necklace_price :
  5 * N + 10 * price_of_bracelet + 20 * price_of_earring
  + 2 * price_of_complete_ensemble = total_amount_made → N = 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_necklace_price_l1121_112153


namespace NUMINAMATH_GPT_value_of_b_l1121_112127

theorem value_of_b 
  (a b : ℝ) 
  (h : ∃ c : ℝ, (ax^3 + bx^2 + 1) = (x^2 - x - 1) * (x + c)) : 
  b = -2 :=
  sorry

end NUMINAMATH_GPT_value_of_b_l1121_112127


namespace NUMINAMATH_GPT_initial_price_of_TV_l1121_112118

theorem initial_price_of_TV (T : ℤ) (phone_price_increase : ℤ) (total_amount : ℤ) 
    (h1 : phone_price_increase = (400: ℤ) + (40 * 400 / 100)) 
    (h2 : total_amount = T + (2 * T / 5) + phone_price_increase) 
    (h3 : total_amount = 1260) : 
    T = 500 := by
  sorry

end NUMINAMATH_GPT_initial_price_of_TV_l1121_112118


namespace NUMINAMATH_GPT_option_one_better_than_option_two_l1121_112106

/-- Define the probability of winning in the first lottery option (drawing two red balls from a box
containing 4 red balls and 2 white balls). -/
def probability_option_one : ℚ := 2 / 5

/-- Define the probability of winning in the second lottery option (rolling two dice and having at least one die show a four). -/
def probability_option_two : ℚ := 11 / 36

/-- Prove that the probability of winning in the first lottery option is greater than the probability of winning in the second lottery option. -/
theorem option_one_better_than_option_two : probability_option_one > probability_option_two :=
by sorry

end NUMINAMATH_GPT_option_one_better_than_option_two_l1121_112106


namespace NUMINAMATH_GPT_find_f_of_functions_l1121_112155

theorem find_f_of_functions
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = - f x)
  (h_even : ∀ x, g (-x) = g x)
  (h_eq : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := 
sorry

end NUMINAMATH_GPT_find_f_of_functions_l1121_112155


namespace NUMINAMATH_GPT_range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l1121_112186

-- Define the function
def f (x a : ℝ) : ℝ := x^2 - a * x + 4 - a^2

-- Problem (1): Range of the function when a = 2
theorem range_of_f_when_a_eq_2 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x 2 = (x - 1)^2 - 1) →
  Set.image (f 2) (Set.Icc (-2 : ℝ) 3) = Set.Icc (-1 : ℝ) 8 := sorry

-- Problem (2): Sufficient but not necessary condition
theorem sufficient_but_not_necessary_condition_for_q :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x 4 ≤ 0) →
  (Set.Icc (-2 : ℝ) 2 → (∃ (M : Set ℝ), Set.singleton 4 ⊆ M ∧ 
    (∀ a ∈ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 0) ∧
    (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∃ a' ∉ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a' ≤ 0))) := sorry

end NUMINAMATH_GPT_range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l1121_112186


namespace NUMINAMATH_GPT_number_of_numbers_l1121_112148

theorem number_of_numbers 
  (avg : ℚ) (avg1 : ℚ) (avg2 : ℚ) (avg3 : ℚ)
  (h_avg : avg = 4.60) 
  (h_avg1 : avg1 = 3.4) 
  (h_avg2 : avg2 = 3.8) 
  (h_avg3 : avg3 = 6.6) 
  (h_sum_eq : 2 * avg1 + 2 * avg2 + 2 * avg3 = 27.6) : 
  (27.6 / avg = 6) := 
  by sorry

end NUMINAMATH_GPT_number_of_numbers_l1121_112148


namespace NUMINAMATH_GPT_tea_drinking_problem_l1121_112193

theorem tea_drinking_problem 
  (k b c t s : ℕ) 
  (hk : k = 1) 
  (hb : b = 15) 
  (hc : c = 3) 
  (ht : t = 2) 
  (hs : s = 1) : 
  17 = 17 := 
by {
  sorry
}

end NUMINAMATH_GPT_tea_drinking_problem_l1121_112193


namespace NUMINAMATH_GPT_max_kings_l1121_112128

theorem max_kings (initial_kings : ℕ) (kings_attacking_each_other : initial_kings = 21) 
  (no_two_kings_attack : ∀ kings_remaining, kings_remaining ≤ 16) : 
  ∃ kings_remaining, kings_remaining = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_kings_l1121_112128


namespace NUMINAMATH_GPT_czakler_inequality_l1121_112161

variable {a b : ℕ} (ha : a > 0) (hb : b > 0)
variable {c : ℝ} (hc : c > 0)

theorem czakler_inequality (h : (a + 1 : ℝ) / (b + c) = b / a) : c ≥ 1 := by
  sorry

end NUMINAMATH_GPT_czakler_inequality_l1121_112161


namespace NUMINAMATH_GPT_fraction_of_passengers_from_Africa_l1121_112105

theorem fraction_of_passengers_from_Africa :
  (1/4 + 1/8 + 1/6 + A + 36/96 = 1) → (96 - 36) = (11/24 * 96) → 
  A = 1/12 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_passengers_from_Africa_l1121_112105


namespace NUMINAMATH_GPT_solve_for_N_l1121_112180

theorem solve_for_N (a b c N : ℝ) 
  (h1 : a + b + c = 72) 
  (h2 : a - 7 = N) 
  (h3 : b + 7 = N) 
  (h4 : 2 * c = N) : 
  N = 28.8 := 
sorry

end NUMINAMATH_GPT_solve_for_N_l1121_112180


namespace NUMINAMATH_GPT_triangle_area_l1121_112189

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 ∧ 0.5 * a * b = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_area_l1121_112189


namespace NUMINAMATH_GPT_intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l1121_112146

def l1 (x y : ℝ) : Prop := x + y = 2
def l2 (x y : ℝ) : Prop := x - 3 * y = -10
def l3 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

def M : (ℝ × ℝ) := (-1, 3)

-- Part (Ⅰ): Prove that M is the intersection point of l1 and l2
theorem intersection_l1_l2 : l1 M.1 M.2 ∧ l2 M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅱ): Prove the equation of the line passing through M and parallel to l3 is 3x - 4y + 15 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y + 15 = 0

theorem line_parallel_to_l3 : parallel_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅲ): Prove the equation of the line passing through M and perpendicular to l3 is 4x + 3y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

theorem line_perpendicular_to_l3 : perpendicular_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l1121_112146


namespace NUMINAMATH_GPT_other_acute_angle_of_right_triangle_l1121_112102

theorem other_acute_angle_of_right_triangle (a : ℝ) (h₀ : 0 < a ∧ a < 90) (h₁ : a = 20) :
  ∃ b, b = 90 - a ∧ b = 70 := by
    sorry

end NUMINAMATH_GPT_other_acute_angle_of_right_triangle_l1121_112102


namespace NUMINAMATH_GPT_perpendicular_line_through_circle_center_l1121_112157

theorem perpendicular_line_through_circle_center :
  ∀ (x y : ℝ), (x^2 + (y-1)^2 = 4) → (3*x + 2*y + 1 = 0) → (2*x - 3*y + 3 = 0) :=
by
  intros x y h_circle h_line
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_circle_center_l1121_112157


namespace NUMINAMATH_GPT_no_real_roots_iff_k_gt_2_l1121_112150

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end NUMINAMATH_GPT_no_real_roots_iff_k_gt_2_l1121_112150


namespace NUMINAMATH_GPT_mixture_problem_l1121_112182

theorem mixture_problem
  (x : ℝ)
  (c1 c2 c_final : ℝ)
  (v1 v2 v_final : ℝ)
  (h1 : c1 = 0.60)
  (h2 : c2 = 0.75)
  (h3 : c_final = 0.72)
  (h4 : v1 = 4)
  (h5 : x = 16)
  (h6 : v2 = x)
  (h7 : v_final = v1 + v2) :
  v_final = 20 ∧ c_final * v_final = c1 * v1 + c2 * v2 :=
by
  sorry

end NUMINAMATH_GPT_mixture_problem_l1121_112182


namespace NUMINAMATH_GPT_expression_value_l1121_112168

theorem expression_value :
  3 * 12^2 - 3 * 13 + 2 * 16 * 11^2 = 4265 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1121_112168


namespace NUMINAMATH_GPT_josanna_minimum_test_score_l1121_112191

theorem josanna_minimum_test_score 
  (scores : List ℕ) (target_increase : ℕ) (new_score : ℕ)
  (h_scores : scores = [92, 78, 84, 76, 88]) 
  (h_target_increase : target_increase = 5):
  (List.sum scores + new_score) / (List.length scores + 1) ≥ (List.sum scores / List.length scores + target_increase) →
  new_score = 114 :=
by
  sorry

end NUMINAMATH_GPT_josanna_minimum_test_score_l1121_112191


namespace NUMINAMATH_GPT_time_to_cross_trains_l1121_112152

/-- Length of the first train in meters -/
def length_train1 : ℕ := 50

/-- Length of the second train in meters -/
def length_train2 : ℕ := 120

/-- Speed of the first train in km/hr -/
def speed_train1_kmh : ℕ := 60

/-- Speed of the second train in km/hr -/
def speed_train2_kmh : ℕ := 40

/-- Relative speed in km/hr as trains are moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1_kmh + speed_train2_kmh

/-- Convert speed from km/hr to m/s -/
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

/-- Relative speed in m/s -/
def relative_speed_ms : ℚ := kmh_to_ms relative_speed_kmh

/-- Total distance to be covered in meters -/
def total_distance : ℕ := length_train1 + length_train2

/-- Time taken in seconds to cross each other -/
def time_to_cross : ℚ := total_distance / relative_speed_ms

theorem time_to_cross_trains :
  time_to_cross = 6.12 := 
sorry

end NUMINAMATH_GPT_time_to_cross_trains_l1121_112152


namespace NUMINAMATH_GPT_total_cupcakes_l1121_112126

noncomputable def cupcakesForBonnie : ℕ := 24
noncomputable def cupcakesPerDay : ℕ := 60
noncomputable def days : ℕ := 2

theorem total_cupcakes : (cupcakesPerDay * days + cupcakesForBonnie) = 144 := 
by
  sorry

end NUMINAMATH_GPT_total_cupcakes_l1121_112126


namespace NUMINAMATH_GPT_carriages_per_train_l1121_112123

variable (c : ℕ)

theorem carriages_per_train :
  (∃ c : ℕ, (25 + 10) * c * 3 = 420) → c = 4 :=
by
  sorry

end NUMINAMATH_GPT_carriages_per_train_l1121_112123


namespace NUMINAMATH_GPT_probability_black_ball_l1121_112114

theorem probability_black_ball :
  let P_red := 0.41
  let P_white := 0.27
  let P_black := 1 - P_red - P_white
  P_black = 0.32 :=
by
  sorry

end NUMINAMATH_GPT_probability_black_ball_l1121_112114


namespace NUMINAMATH_GPT_min_f_over_f_prime_at_1_l1121_112147

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def quadratic_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem min_f_over_f_prime_at_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b > 0) (h₂ : ∀ x, quadratic_function a b c x ≥ 0) :
  (∃ k, (∀ x, quadratic_function a b c x ≥ 0 → quadratic_function a b c ((-b)/(2*a)) ≤ x) ∧ k = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_f_over_f_prime_at_1_l1121_112147


namespace NUMINAMATH_GPT_exponentiation_evaluation_l1121_112195

theorem exponentiation_evaluation :
  (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_GPT_exponentiation_evaluation_l1121_112195


namespace NUMINAMATH_GPT_y_in_terms_of_w_l1121_112104

theorem y_in_terms_of_w (y w : ℝ) (h1 : y = 3^2 - 1) (h2 : w = 2) : y = 4 * w :=
by
  sorry

end NUMINAMATH_GPT_y_in_terms_of_w_l1121_112104


namespace NUMINAMATH_GPT_no_such_x_exists_l1121_112166

theorem no_such_x_exists : ¬ ∃ x : ℝ, 
  (∃ x1 : ℤ, x - 1/x = x1) ∧ 
  (∃ x2 : ℤ, 1/x - 1/(x^2 + 1) = x2) ∧ 
  (∃ x3 : ℤ, 1/(x^2 + 1) - 2*x = x3) :=
by
  sorry

end NUMINAMATH_GPT_no_such_x_exists_l1121_112166


namespace NUMINAMATH_GPT_friends_gift_l1121_112131

-- Define the original number of balloons and the final number of balloons
def original_balloons := 8
def final_balloons := 10

-- The main theorem: Joan's friend gave her 2 orange balloons.
theorem friends_gift : (final_balloons - original_balloons) = 2 := by
  sorry

end NUMINAMATH_GPT_friends_gift_l1121_112131


namespace NUMINAMATH_GPT_no_right_angle_sequence_l1121_112165

theorem no_right_angle_sequence 
  (A B C : Type)
  (angle_A angle_B angle_C : ℝ)
  (angle_A_eq : angle_A = 59)
  (angle_B_eq : angle_B = 61)
  (angle_C_eq : angle_C = 60)
  (midpoint : A → A → A)
  (A0 B0 C0 : A) :
  ¬ ∃ n : ℕ, ∃ An Bn Cn : A, 
    (An = midpoint Bn Cn) ∧ 
    (Bn = midpoint An Cn) ∧ 
    (Cn = midpoint An Bn) ∧ 
    (angle_A = 90 ∨ angle_B = 90 ∨ angle_C = 90) :=
sorry

end NUMINAMATH_GPT_no_right_angle_sequence_l1121_112165


namespace NUMINAMATH_GPT_inequality_min_value_l1121_112129

theorem inequality_min_value (a : ℝ) : 
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ a) → (a ≤ 3) := 
by
  sorry

end NUMINAMATH_GPT_inequality_min_value_l1121_112129


namespace NUMINAMATH_GPT_carpet_area_l1121_112181

def room_length_ft := 16
def room_width_ft := 12
def column_side_ft := 2
def ft_to_inches := 12

def room_length_in := room_length_ft * ft_to_inches
def room_width_in := room_width_ft * ft_to_inches
def column_side_in := column_side_ft * ft_to_inches

def room_area_in_sq := room_length_in * room_width_in
def column_area_in_sq := column_side_in * column_side_in

def remaining_area_in_sq := room_area_in_sq - column_area_in_sq

theorem carpet_area : remaining_area_in_sq = 27072 := by
  sorry

end NUMINAMATH_GPT_carpet_area_l1121_112181


namespace NUMINAMATH_GPT_inequality_proof_l1121_112154

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1121_112154


namespace NUMINAMATH_GPT_pages_copied_l1121_112174

-- Define the assumptions
def cost_per_pages (cent_per_pages: ℕ) : Prop := 
  5 * cent_per_pages = 7 * 1

def total_cents (dollars: ℕ) (cents: ℕ) : Prop :=
  cents = dollars * 100

-- The problem to prove
theorem pages_copied (dollars: ℕ) (cents: ℕ) (cent_per_pages: ℕ) : 
  cost_per_pages cent_per_pages → total_cents dollars cents → dollars = 35 → cents = 3500 → 
  3500 * (5/7 : ℚ) = 2500 :=
by
  sorry

end NUMINAMATH_GPT_pages_copied_l1121_112174


namespace NUMINAMATH_GPT_total_tickets_l1121_112112

-- Definitions based on given conditions
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def additional_tickets : ℕ := 6

-- Proof statement (only statement, proof is not required)
theorem total_tickets : (initial_tickets - spent_tickets + additional_tickets = 30) :=
  sorry

end NUMINAMATH_GPT_total_tickets_l1121_112112


namespace NUMINAMATH_GPT_total_savings_l1121_112179

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end NUMINAMATH_GPT_total_savings_l1121_112179


namespace NUMINAMATH_GPT_prove_AP_BP_CP_product_l1121_112124

open Classical

-- Defines that the point P is inside the acute-angled triangle ABC
variables {A B C P: Type} [MetricSpace P] 
variables (PA1 PB1 PC1 AP BP CP : ℝ)

-- Conditions
def conditions (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) : Prop :=
  PA1 = 3 ∧ PB1 = 3 ∧ PC1 = 3 ∧ AP + BP + CP = 43

-- Proof goal
theorem prove_AP_BP_CP_product (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) :
  AP * BP * CP = 441 :=
by {
  -- Proof steps will be filled here
  sorry
}

end NUMINAMATH_GPT_prove_AP_BP_CP_product_l1121_112124


namespace NUMINAMATH_GPT_Ginger_sold_10_lilacs_l1121_112107

variable (R L G : ℕ)

def condition1 := R = 3 * L
def condition2 := G = L / 2
def condition3 := L + R + G = 45

theorem Ginger_sold_10_lilacs
    (h1 : condition1 R L)
    (h2 : condition2 G L)
    (h3 : condition3 L R G) :
  L = 10 := 
  sorry

end NUMINAMATH_GPT_Ginger_sold_10_lilacs_l1121_112107


namespace NUMINAMATH_GPT_rectangle_area_l1121_112117

theorem rectangle_area (y : ℝ) (h_rect : (5 - (-3)) * (y - (-1)) = 48) (h_pos : 0 < y) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1121_112117


namespace NUMINAMATH_GPT_find_new_person_age_l1121_112116

variables (A X : ℕ) -- A is the original average age, X is the age of the new person

def original_total_age (A : ℕ) := 10 * A
def new_total_age (A X : ℕ) := 10 * (A - 3)

theorem find_new_person_age (A : ℕ) (h : new_total_age A X = original_total_age A - 45 + X) : X = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_new_person_age_l1121_112116


namespace NUMINAMATH_GPT_average_matches_rounded_l1121_112125

def total_matches : ℕ := 6 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_players : ℕ := 6 + 3 + 3 + 2 + 6

noncomputable def average_matches : ℚ := total_matches / total_players

theorem average_matches_rounded : Int.floor (average_matches + 0.5) = 3 :=
by
  unfold average_matches total_matches total_players
  norm_num
  sorry

end NUMINAMATH_GPT_average_matches_rounded_l1121_112125


namespace NUMINAMATH_GPT_root_equation_alpha_beta_property_l1121_112139

theorem root_equation_alpha_beta_property {α β : ℝ} (h1 : α^2 + α - 1 = 0) (h2 : β^2 + β - 1 = 0) :
    α^2 + 2 * β^2 + β = 4 :=
by
  sorry

end NUMINAMATH_GPT_root_equation_alpha_beta_property_l1121_112139


namespace NUMINAMATH_GPT_best_value_l1121_112140

variables {cS qS cM qL cL : ℝ}
variables (medium_cost : cM = 1.4 * cS) (medium_quantity : qM = 0.7 * qL)
variables (large_quantity : qL = 1.5 * qS) (large_cost : cL = 1.2 * cM)

theorem best_value :
  let small_value := cS / qS
  let medium_value := cM / (0.7 * qL)
  let large_value := cL / qL
  small_value < large_value ∧ large_value < medium_value :=
sorry

end NUMINAMATH_GPT_best_value_l1121_112140


namespace NUMINAMATH_GPT_avg_age_women_is_52_l1121_112192

-- Definitions
def avg_age_men (A : ℚ) := 9 * A
def total_increase := 36
def combined_age_replaced := 36 + 32
def combined_age_women := combined_age_replaced + total_increase
def avg_age_women (W : ℚ) := W / 2

-- Theorem statement
theorem avg_age_women_is_52 (A : ℚ) : avg_age_women combined_age_women = 52 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_women_is_52_l1121_112192


namespace NUMINAMATH_GPT_max_value_of_z_l1121_112198

theorem max_value_of_z
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + y ≤ 1)
  (h3 : y ≥ -1) :
  ∃ x y, (y ≥ x) ∧ (x + y ≤ 1) ∧ (y ≥ -1) ∧ (2 * x - y = 1 / 2) := by 
  sorry

end NUMINAMATH_GPT_max_value_of_z_l1121_112198


namespace NUMINAMATH_GPT_union_complement_A_eq_l1121_112101

open Set

universe u

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ (x : ℝ), y = x^2 + 1 }

theorem union_complement_A_eq :
  A ∪ ((U \ B : Set ℝ) : Set ℝ) = { x | x < 2 } := by
  sorry

end NUMINAMATH_GPT_union_complement_A_eq_l1121_112101


namespace NUMINAMATH_GPT_even_suff_not_nec_l1121_112103

theorem even_suff_not_nec (f g : ℝ → ℝ) 
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hg_even : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x + g x) = ((f + g) x) ∧ (∀ h : ℝ → ℝ, ∃ f g : ℝ → ℝ, h = f + g ∧ ∀ x : ℝ, (h (-x) = h x) ↔ (f (-x) = f x ∧ g (-x) = g x)) :=
by 
  sorry

end NUMINAMATH_GPT_even_suff_not_nec_l1121_112103


namespace NUMINAMATH_GPT_bret_spends_77_dollars_l1121_112194

def num_people : ℕ := 4
def main_meal_cost : ℝ := 12.0
def num_appetizers : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0

def total_cost (num_people : ℕ) (main_meal_cost : ℝ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_order_fee : ℝ) : ℝ :=
  let main_meal_total := num_people * main_meal_cost
  let appetizer_total := num_appetizers * appetizer_cost
  let subtotal := main_meal_total + appetizer_total
  let tip := tip_rate * subtotal
  subtotal + tip + rush_order_fee

theorem bret_spends_77_dollars :
  total_cost num_people main_meal_cost num_appetizers appetizer_cost tip_rate rush_order_fee = 77.0 :=
by
  sorry

end NUMINAMATH_GPT_bret_spends_77_dollars_l1121_112194


namespace NUMINAMATH_GPT_simplify_exponent_l1121_112134

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponent_l1121_112134


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1121_112121

theorem repeating_decimal_to_fraction : (6 + 81 / 99) = 75 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1121_112121


namespace NUMINAMATH_GPT_ratio_of_areas_of_shaded_and_white_region_l1121_112172

theorem ratio_of_areas_of_shaded_and_white_region
  (all_squares_have_vertices_in_middle: ∀ (n : ℕ), n ≠ 0 → (square_vertices_positioned_mid : Prop)) :
  ∃ (ratio : ℚ), ratio = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_shaded_and_white_region_l1121_112172


namespace NUMINAMATH_GPT_total_cars_produced_l1121_112188

theorem total_cars_produced (cars_NA cars_EU : ℕ) (h1 : cars_NA = 3884) (h2 : cars_EU = 2871) : cars_NA + cars_EU = 6755 := by
  sorry

end NUMINAMATH_GPT_total_cars_produced_l1121_112188


namespace NUMINAMATH_GPT_Nunzio_eats_pizza_every_day_l1121_112184

theorem Nunzio_eats_pizza_every_day
  (one_piece_fraction : ℚ := 1/8)
  (total_pizzas : ℕ := 27)
  (total_days : ℕ := 72)
  (pieces_per_pizza : ℕ := 8)
  (total_pieces : ℕ := total_pizzas * pieces_per_pizza)
  : (total_pieces / total_days = 3) :=
by
  -- We assume 1/8 as a fraction for the pieces of pizza is stated in the conditions, therefore no condition here.
  -- We need to show that Nunzio eats 3 pieces of pizza every day given the total pieces and days.
  sorry

end NUMINAMATH_GPT_Nunzio_eats_pizza_every_day_l1121_112184


namespace NUMINAMATH_GPT_minimum_k_l1121_112167

variable {a b k : ℝ}

theorem minimum_k (h_a : a > 0) (h_b : b > 0) (h : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a) + (1 / b) + (k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end NUMINAMATH_GPT_minimum_k_l1121_112167


namespace NUMINAMATH_GPT_q_join_after_days_l1121_112185

noncomputable def workRate (totalWork : ℕ) (days : ℕ) : ℚ :=
  totalWork / days

theorem q_join_after_days (W : ℕ) (days_p : ℕ) (days_q : ℕ) (total_days : ℕ) (x : ℕ) :
  days_p = 80 ∧ days_q = 48 ∧ total_days = 35 ∧ 
  ((workRate W days_p) * x + (workRate W days_p + workRate W days_q) * (total_days - x) = W) 
  → x = 8 := sorry

end NUMINAMATH_GPT_q_join_after_days_l1121_112185


namespace NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1121_112190

-- Definitions of the lines
def l1 (a x y : ℝ) := x + a * y - 2 * a - 2 = 0
def l2 (a x y : ℝ) := a * x + y - 1 - a = 0

-- Statement for parallel lines
theorem parallel_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = 0 ∨ x = 1) → a = 1 :=
by 
  -- proof outline
  sorry

-- Statement for perpendicular lines
theorem perpendicular_lines (a : ℝ) : (∀ x y, l1 a x y → l2 a x y → x = y) → a = 0 :=
by 
  -- proof outline
  sorry

end NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1121_112190


namespace NUMINAMATH_GPT_smallest_angle_in_scalene_triangle_l1121_112149

theorem smallest_angle_in_scalene_triangle :
  ∃ (triangle : Type) (a b c : ℝ),
    ∀ (A B C : triangle),
      a = 162 ∧
      b / c = 3 / 4 ∧
      a + b + c = 180 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c ->
        min b c = 7.7 :=
sorry

end NUMINAMATH_GPT_smallest_angle_in_scalene_triangle_l1121_112149


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1121_112120

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + (1 / (a * b)) + (1 / (a * (a - b)))

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_value a b >= 4 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1121_112120


namespace NUMINAMATH_GPT_average_after_12th_innings_l1121_112163

variable (runs_11 score_12 increase_avg : ℕ)
variable (A : ℕ)

theorem average_after_12th_innings
  (h1 : score_12 = 60)
  (h2 : increase_avg = 2)
  (h3 : 11 * A = runs_11)
  (h4 : (runs_11 + score_12) / 12 = A + increase_avg) :
  (A + 2 = 38) :=
by
  sorry

end NUMINAMATH_GPT_average_after_12th_innings_l1121_112163
