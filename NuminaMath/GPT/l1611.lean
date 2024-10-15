import Mathlib

namespace NUMINAMATH_GPT_most_economical_speed_and_cost_l1611_161118

open Real

theorem most_economical_speed_and_cost :
  ∀ (x : ℝ),
  (120:ℝ) / x * 36 + (120:ℝ) / x * 6 * (4 + x^2 / 360) = ((7200:ℝ) / x) + 2 * x → 
  50 ≤ x ∧ x ≤ 100 → 
  (∀ v : ℝ, (50 ≤ v ∧ v ≤ 100) → 
  (120 / v * 36 + 120 / v * 6 * (4 + v^2 / 360) ≤ 120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360)) ) → 
  x = 60 → 
  (120 / x * 36 + 120 / x * 6 * (4 + x^2 / 360) = 240) :=
by
  intros x hx bounds min_cost opt_speed
  sorry

end NUMINAMATH_GPT_most_economical_speed_and_cost_l1611_161118


namespace NUMINAMATH_GPT_equal_roots_quadratic_l1611_161186

theorem equal_roots_quadratic {k : ℝ} 
  (h : (∃ x : ℝ, x^2 - 6 * x + k = 0 ∧ x^2 - 6 * x + k = 0)) : 
  k = 9 :=
sorry

end NUMINAMATH_GPT_equal_roots_quadratic_l1611_161186


namespace NUMINAMATH_GPT_evaluate_at_2_l1611_161144

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem evaluate_at_2 : f 2 = 62 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_at_2_l1611_161144


namespace NUMINAMATH_GPT_perpendicular_lines_k_value_l1611_161180

theorem perpendicular_lines_k_value :
  ∀ (k : ℝ), (∀ (x y : ℝ), x + 4 * y - 1 = 0) →
             (∀ (x y : ℝ), k * x + y + 2 = 0) →
             (-1 / 4 * -k = -1) →
             k = -4 :=
by
  intros k h1 h2 h3
  sorry

end NUMINAMATH_GPT_perpendicular_lines_k_value_l1611_161180


namespace NUMINAMATH_GPT_quadratic_roots_l1611_161163

theorem quadratic_roots (m x1 x2 : ℝ) 
  (h1 : 2*x1^2 + 4*m*x1 + m = 0)
  (h2 : 2*x2^2 + 4*m*x2 + m = 0)
  (h3 : x1 ≠ x2)
  (h4 : x1^2 + x2^2 = 3/16) : 
  m = -1/8 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_l1611_161163


namespace NUMINAMATH_GPT_present_condition_l1611_161146

variable {α : Type} [Finite α]

-- We will represent children as members of a type α and assume there are precisely 3n children.
variable (n : ℕ) (h_odd : odd n) [h : Fintype α] (card_3n : Fintype.card α = 3 * n)

noncomputable def makes_present_to (A B : α) : α := sorry -- Create a function that maps pairs of children to exactly one child.

theorem present_condition : ∀ (A B C : α), makes_present_to A B = C → makes_present_to A C = B :=
sorry

end NUMINAMATH_GPT_present_condition_l1611_161146


namespace NUMINAMATH_GPT_abs_diff_of_prod_and_sum_l1611_161104

theorem abs_diff_of_prod_and_sum (m n : ℝ) (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 :=
by
  -- The proof is not required as per the instructions.
  sorry

end NUMINAMATH_GPT_abs_diff_of_prod_and_sum_l1611_161104


namespace NUMINAMATH_GPT_power_difference_divisible_by_10000_l1611_161148

theorem power_difference_divisible_by_10000 (a b : ℤ) (m : ℤ) (h : a - b = 100 * m) : ∃ k : ℤ, a^100 - b^100 = 10000 * k := by
  sorry

end NUMINAMATH_GPT_power_difference_divisible_by_10000_l1611_161148


namespace NUMINAMATH_GPT_total_votes_l1611_161183

variable (V : ℝ)

theorem total_votes (h1 : 0.34 * V + 640 = 0.66 * V) : V = 2000 :=
by 
  sorry

end NUMINAMATH_GPT_total_votes_l1611_161183


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1611_161109
-- Importing the entire Mathlib library for mathematical functions

-- Define the two numbers and the conditions given in the problem
variables (x y : ℕ)

-- State the conditions as hypotheses
def conditions (h1 : x + y = 45) (h2 : Nat.lcm x y = 120) (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Prop :=
  True

-- State the theorem we want to prove
theorem hcf_of_two_numbers (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Nat.gcd x y = 1 :=
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1611_161109


namespace NUMINAMATH_GPT_total_pears_l1611_161173

noncomputable def Jason_pears : ℝ := 46
noncomputable def Keith_pears : ℝ := 47
noncomputable def Mike_pears : ℝ := 12
noncomputable def Sarah_pears : ℝ := 32.5
noncomputable def Emma_pears : ℝ := (2 / 3) * Mike_pears
noncomputable def James_pears : ℝ := (2 * Sarah_pears) - 3

theorem total_pears :
  Jason_pears + Keith_pears + Mike_pears + Sarah_pears + Emma_pears + James_pears = 207.5 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_l1611_161173


namespace NUMINAMATH_GPT_solution_set_correct_l1611_161127

theorem solution_set_correct (a b : ℝ) :
  (∀ x : ℝ, - 1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0) →
  (a - b = -10) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_correct_l1611_161127


namespace NUMINAMATH_GPT_tan_trig_identity_l1611_161187

noncomputable def given_condition (α : ℝ) : Prop :=
  Real.tan (α + Real.pi / 3) = 2

theorem tan_trig_identity (α : ℝ) (h : given_condition α) :
  (Real.sin (α + (4 * Real.pi / 3)) + Real.cos ((2 * Real.pi / 3) - α)) /
  (Real.cos ((Real.pi / 6) - α) - Real.sin (α + (5 * Real.pi / 6))) = -3 :=
sorry

end NUMINAMATH_GPT_tan_trig_identity_l1611_161187


namespace NUMINAMATH_GPT_monomial_2024_l1611_161156

def monomial (n : ℕ) : ℤ × ℕ := ((-1)^(n + 1) * (2 * n - 1), n)

theorem monomial_2024 :
  monomial 2024 = (-4047, 2024) :=
sorry

end NUMINAMATH_GPT_monomial_2024_l1611_161156


namespace NUMINAMATH_GPT_regular_polygon_sides_l1611_161192

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1611_161192


namespace NUMINAMATH_GPT_nth_equation_l1611_161126

theorem nth_equation (n : ℕ) : (2 * n + 2) ^ 2 - (2 * n) ^ 2 = 4 * (2 * n + 1) :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l1611_161126


namespace NUMINAMATH_GPT_incorrect_variance_l1611_161137

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (- (x - 1)^2 / 2)

theorem incorrect_variance :
  (∫ x, normal_pdf x * x^2) - (∫ x, normal_pdf x * x)^2 ≠ 2 := 
sorry

end NUMINAMATH_GPT_incorrect_variance_l1611_161137


namespace NUMINAMATH_GPT_negation_of_exists_leq_l1611_161110

theorem negation_of_exists_leq (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_leq_l1611_161110


namespace NUMINAMATH_GPT_find_difference_condition_l1611_161194

variable (a b c : ℝ)

theorem find_difference_condition (h1 : (a + b) / 2 = 40) (h2 : (b + c) / 2 = 60) : c - a = 40 := by
  sorry

end NUMINAMATH_GPT_find_difference_condition_l1611_161194


namespace NUMINAMATH_GPT_quadratic_inequality_empty_solution_set_l1611_161119

theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ := 
by sorry

end NUMINAMATH_GPT_quadratic_inequality_empty_solution_set_l1611_161119


namespace NUMINAMATH_GPT_contrapositive_prop_l1611_161195

theorem contrapositive_prop {α : Type} [Mul α] [Zero α] (a b : α) : 
  (a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0) :=
by sorry

end NUMINAMATH_GPT_contrapositive_prop_l1611_161195


namespace NUMINAMATH_GPT_valid_sequences_l1611_161159

-- Define the transformation function for a ten-digit number
noncomputable def transform (n : ℕ) : ℕ := sorry

-- Given sequences
def seq1 := 1101111111
def seq2 := 1201201020
def seq3 := 1021021020
def seq4 := 0112102011

-- The proof problem statement
theorem valid_sequences :
  (transform 1101111111 = seq1) ∧
  (transform 1021021020 = seq3) ∧
  (transform 0112102011 = seq4) :=
sorry

end NUMINAMATH_GPT_valid_sequences_l1611_161159


namespace NUMINAMATH_GPT_domain_of_function_l1611_161152

def valid_domain (x : ℝ) : Prop :=
  (2 - x ≥ 0) ∧ (x > 0) ∧ (x ≠ 2)

theorem domain_of_function :
  {x : ℝ | ∃ (y : ℝ), y = x ∧ valid_domain x} = {x | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1611_161152


namespace NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l1611_161129

noncomputable def f (x a : ℝ) := x^3 - a * x^2 + 1

theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < 2) ∧ (0 < y ∧ y < 2) → x < y → f x a ≥ f y a) → (a ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_monotonically_decreasing_l1611_161129


namespace NUMINAMATH_GPT_six_digit_number_property_l1611_161101

theorem six_digit_number_property {a b c d e f : ℕ} 
  (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 0 ≤ c ∧ c < 10) (h4 : 0 ≤ d ∧ d < 10)
  (h5 : 0 ≤ e ∧ e < 10) (h6 : 0 ≤ f ∧ f < 10) 
  (h7 : 100000 ≤ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f ∧
        a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f < 1000000) :
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 3 * (f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e)) ↔ 
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 428571 ∨ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 857142) :=
sorry

end NUMINAMATH_GPT_six_digit_number_property_l1611_161101


namespace NUMINAMATH_GPT_tan_sum_l1611_161131

theorem tan_sum (α : ℝ) (h : Real.cos (π / 2 + α) = 2 * Real.cos α) : 
  Real.tan α + Real.tan (2 * α) = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_l1611_161131


namespace NUMINAMATH_GPT_find_two_digit_number_l1611_161151

def product_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the product of the digits of n
sorry

def sum_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the sum of the digits of n
sorry

theorem find_two_digit_number (M : ℕ) (h1 : 10 ≤ M ∧ M < 100) (h2 : M = product_of_digits M + sum_of_digits M + 1) : M = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1611_161151


namespace NUMINAMATH_GPT_simplify_expr_l1611_161190

theorem simplify_expr : (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1611_161190


namespace NUMINAMATH_GPT_car_speed_is_104_mph_l1611_161191

noncomputable def speed_of_car_in_mph
  (fuel_efficiency_km_per_liter : ℝ) -- car travels 64 km per liter
  (fuel_consumption_gallons : ℝ) -- fuel tank decreases by 3.9 gallons
  (time_hours : ℝ) -- period of 5.7 hours
  (gallon_to_liter : ℝ) -- 1 gallon is 3.8 liters
  (km_to_mile : ℝ) -- 1 mile is 1.6 km
  : ℝ :=
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter
  let distance_km := fuel_efficiency_km_per_liter * fuel_consumption_liters
  let distance_miles := distance_km / km_to_mile
  let speed_mph := distance_miles / time_hours
  speed_mph

theorem car_speed_is_104_mph 
  (fuel_efficiency_km_per_liter : ℝ := 64)
  (fuel_consumption_gallons : ℝ := 3.9)
  (time_hours : ℝ := 5.7)
  (gallon_to_liter : ℝ := 3.8)
  (km_to_mile : ℝ := 1.6)
  : speed_of_car_in_mph fuel_efficiency_km_per_liter fuel_consumption_gallons time_hours gallon_to_liter km_to_mile = 104 :=
  by
    sorry

end NUMINAMATH_GPT_car_speed_is_104_mph_l1611_161191


namespace NUMINAMATH_GPT_triangle_height_and_segments_l1611_161169

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_height_and_segments_l1611_161169


namespace NUMINAMATH_GPT_find_p_if_parabola_axis_tangent_to_circle_l1611_161103

theorem find_p_if_parabola_axis_tangent_to_circle :
  ∀ (p : ℝ), 0 < p →
    (∃ (C : ℝ × ℝ) (r : ℝ), 
      (C = (2, 0)) ∧ (r = 3) ∧ (dist (C.1 + p / 2, C.2) (C.1, C.2) = r) 
    ) → p = 2 :=
by
  intro p hp h
  rcases h with ⟨C, r, hC, hr, h_dist⟩ 
  have h_eq : C = (2, 0) := hC
  have hr_eq : r = 3 := hr
  rw [h_eq, hr_eq] at h_dist
  sorry

end NUMINAMATH_GPT_find_p_if_parabola_axis_tangent_to_circle_l1611_161103


namespace NUMINAMATH_GPT_doubling_profit_condition_l1611_161134

-- Definitions
def purchase_price : ℝ := 210
def initial_selling_price : ℝ := 270
def initial_items_sold : ℝ := 30
def profit_per_item (selling_price : ℝ) : ℝ := selling_price - purchase_price
def daily_profit (selling_price : ℝ) (items_sold : ℝ) : ℝ := profit_per_item selling_price * items_sold
def increase_in_items_sold_per_yuan (reduction : ℝ) : ℝ := 3 * reduction

-- Condition: Initial daily profit
def initial_daily_profit : ℝ := daily_profit initial_selling_price initial_items_sold

-- Proof problem
theorem doubling_profit_condition (reduction : ℝ) :
  daily_profit (initial_selling_price - reduction) (initial_items_sold + increase_in_items_sold_per_yuan reduction) = 2 * initial_daily_profit :=
sorry

end NUMINAMATH_GPT_doubling_profit_condition_l1611_161134


namespace NUMINAMATH_GPT_download_time_l1611_161142

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end NUMINAMATH_GPT_download_time_l1611_161142


namespace NUMINAMATH_GPT_g_of_10_l1611_161181

noncomputable def g : ℕ → ℝ := sorry

axiom g_initial : g 1 = 2

axiom g_condition : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = 2 * g m + 3 * g n

theorem g_of_10 : g 10 = 496 :=
by
  sorry

end NUMINAMATH_GPT_g_of_10_l1611_161181


namespace NUMINAMATH_GPT_solve_for_x_l1611_161164

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1611_161164


namespace NUMINAMATH_GPT_cistern_water_depth_l1611_161193

theorem cistern_water_depth:
  ∀ h: ℝ,
  (4 * 4 + 4 * h * 4 + 4 * h * 4 = 36) → h = 1.25 := by
    sorry

end NUMINAMATH_GPT_cistern_water_depth_l1611_161193


namespace NUMINAMATH_GPT_smallest_n_produces_terminating_decimal_l1611_161174

noncomputable def smallest_n := 12

theorem smallest_n_produces_terminating_decimal (n : ℕ) (h_pos: 0 < n) : 
    (∀ m : ℕ, m > 113 → (n = m - 113 → (∃ k : ℕ, 1 ≤ k ∧ (m = 2^k ∨ m = 5^k)))) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_produces_terminating_decimal_l1611_161174


namespace NUMINAMATH_GPT_sue_charge_per_dog_l1611_161182

def amount_saved_christian : ℝ := 5
def amount_saved_sue : ℝ := 7
def charge_per_yard : ℝ := 5
def yards_mowed_christian : ℝ := 4
def total_cost_perfume : ℝ := 50
def additional_amount_needed : ℝ := 6
def dogs_walked_sue : ℝ := 6

theorem sue_charge_per_dog :
  (amount_saved_christian + (charge_per_yard * yards_mowed_christian) + amount_saved_sue + (dogs_walked_sue * x) + additional_amount_needed = total_cost_perfume) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_sue_charge_per_dog_l1611_161182


namespace NUMINAMATH_GPT_solve_inequalities_l1611_161135

-- Define the interval [-1, 1]
def interval := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

-- State the problem
theorem solve_inequalities :
  {x : ℝ | 3 * x^2 + 2 * x - 9 ≤ 0 ∧ x ≥ -1} = interval := 
sorry

end NUMINAMATH_GPT_solve_inequalities_l1611_161135


namespace NUMINAMATH_GPT_t_n_minus_n_even_l1611_161111

noncomputable def number_of_nonempty_subsets_with_integer_average (n : ℕ) : ℕ := 
  sorry

theorem t_n_minus_n_even (N : ℕ) (hN : N > 1) :
  ∃ T_n, T_n = number_of_nonempty_subsets_with_integer_average N ∧ (T_n - N) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_t_n_minus_n_even_l1611_161111


namespace NUMINAMATH_GPT_a_power_2018_plus_b_power_2018_eq_2_l1611_161143

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem a_power_2018_plus_b_power_2018_eq_2 (a b : ℝ) :
  (∀ x : ℝ, f x a b + f (1 / x) a b = 0) → a^2018 + b^2018 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_a_power_2018_plus_b_power_2018_eq_2_l1611_161143


namespace NUMINAMATH_GPT_original_selling_price_l1611_161154

theorem original_selling_price (P SP1 SP2 : ℝ) (h1 : SP1 = 1.10 * P)
    (h2 : SP2 = 1.17 * P) (h3 : SP2 - SP1 = 35) : SP1 = 550 :=
by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1611_161154


namespace NUMINAMATH_GPT_loom_weaving_rate_l1611_161175

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (rate : ℝ) 
  (h1 : total_cloth = 26) (h2 : total_time = 203.125) : rate = total_cloth / total_time := by
  sorry

#check loom_weaving_rate

end NUMINAMATH_GPT_loom_weaving_rate_l1611_161175


namespace NUMINAMATH_GPT_no_integer_coeff_trinomials_with_integer_roots_l1611_161116

theorem no_integer_coeff_trinomials_with_integer_roots :
  ¬ ∃ (a b c : ℤ),
    (∀ x : ℤ, a * x^2 + b * x + c = 0 → (∃ x1 x2 : ℤ, a = 0 ∧ x = x1 ∨ a ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) ∧
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0 → (∃ x1 x2 : ℤ, (a + 1) = 0 ∧ x = x1 ∨ (a + 1) ≠ 0 ∧ x = x1 ∨ x = x2 ∨ x = x1 ∧ x = x2)) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_coeff_trinomials_with_integer_roots_l1611_161116


namespace NUMINAMATH_GPT_geometric_common_ratio_of_arithmetic_seq_l1611_161177

theorem geometric_common_ratio_of_arithmetic_seq 
  (a : ℕ → ℝ) (d q : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 2)
  (h_nonzero_diff : d ≠ 0)
  (h_geo_seq : a 1 = 2 ∧ a 3 = 2 * q ∧ a 11 = 2 * q^2) : 
  q = 4 := 
by
  sorry

end NUMINAMATH_GPT_geometric_common_ratio_of_arithmetic_seq_l1611_161177


namespace NUMINAMATH_GPT_find_d_l1611_161179

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)  
  (h1 : α = c) 
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : α + d + β + γ = 180) :
  d = 42 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1611_161179


namespace NUMINAMATH_GPT_number_of_roots_in_right_half_plane_is_one_l1611_161188

def Q5 (z : ℂ) : ℂ := z^5 + z^4 + 2*z^3 - 8*z - 1

theorem number_of_roots_in_right_half_plane_is_one :
  (∃ n, ∀ z, Q5 z = 0 ∧ z.re > 0 ↔ n = 1) := 
sorry

end NUMINAMATH_GPT_number_of_roots_in_right_half_plane_is_one_l1611_161188


namespace NUMINAMATH_GPT_simplify_expression_l1611_161133

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : 1 - x ≠ 0) :
  (1 - x) / x / ((1 - x) / x^2) = x := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1611_161133


namespace NUMINAMATH_GPT_average_tickets_sold_by_male_members_l1611_161178

theorem average_tickets_sold_by_male_members 
  (M F : ℕ)
  (total_average : ℕ)
  (female_average : ℕ)
  (ratio : ℕ × ℕ)
  (h1 : total_average = 66)
  (h2 : female_average = 70)
  (h3 : ratio = (1, 2))
  (h4 : F = 2 * M)
  (h5 : (M + F) * total_average = M * r + F * female_average) :
  r = 58 :=
sorry

end NUMINAMATH_GPT_average_tickets_sold_by_male_members_l1611_161178


namespace NUMINAMATH_GPT_range_of_a_l1611_161158

theorem range_of_a (a : ℝ) (x : ℝ) : (x > a ∧ x > 1) → (x > 1) → (a ≤ 1) :=
by 
  intros hsol hx
  sorry

end NUMINAMATH_GPT_range_of_a_l1611_161158


namespace NUMINAMATH_GPT_find_c_l1611_161149

open Function

noncomputable def g (x : ℝ) : ℝ :=
  (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 255 - 5

theorem find_c (c : ℤ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = c ∧ g x₂ = c ∧ g x₃ = c ∧ g x₄ = c ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  ∀ k : ℤ, k < c → ¬ ∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ = k ∧ g x₂ = k ∧ g x₃ = k ∧ g x₄ = k ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ :=
sorry

end NUMINAMATH_GPT_find_c_l1611_161149


namespace NUMINAMATH_GPT_percent_of_dollar_in_pocket_l1611_161124

def value_of_penny : ℕ := 1  -- value of one penny in cents
def value_of_nickel : ℕ := 5  -- value of one nickel in cents
def value_of_half_dollar : ℕ := 50 -- value of one half-dollar in cents

def pennies : ℕ := 3  -- number of pennies
def nickels : ℕ := 2  -- number of nickels
def half_dollars : ℕ := 1  -- number of half-dollars

def total_value_in_cents : ℕ :=
  (pennies * value_of_penny) + (nickels * value_of_nickel) + (half_dollars * value_of_half_dollar)

def value_of_dollar_in_cents : ℕ := 100

def percent_of_dollar (value : ℕ) (total : ℕ) : ℚ := (value / total) * 100

theorem percent_of_dollar_in_pocket : percent_of_dollar total_value_in_cents value_of_dollar_in_cents = 63 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_dollar_in_pocket_l1611_161124


namespace NUMINAMATH_GPT_expression_divisible_by_24_l1611_161107

theorem expression_divisible_by_24 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end NUMINAMATH_GPT_expression_divisible_by_24_l1611_161107


namespace NUMINAMATH_GPT_consecutive_integers_average_and_product_l1611_161123

theorem consecutive_integers_average_and_product (n m : ℤ) (hnm : n ≤ m) 
  (h1 : (n + m) / 2 = 20) 
  (h2 : n * m = 391) :  m - n + 1 = 7 :=
  sorry

end NUMINAMATH_GPT_consecutive_integers_average_and_product_l1611_161123


namespace NUMINAMATH_GPT_original_number_l1611_161189

theorem original_number (x : ℤ) (h : (x - 5) / 4 = (x - 4) / 5) : x = 9 :=
sorry

end NUMINAMATH_GPT_original_number_l1611_161189


namespace NUMINAMATH_GPT_power_equiv_l1611_161147

theorem power_equiv (x_0 : ℝ) (h : x_0 ^ 11 + x_0 ^ 7 + x_0 ^ 3 = 1) : x_0 ^ 4 + x_0 ^ 3 - 1 = x_0 ^ 15 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_power_equiv_l1611_161147


namespace NUMINAMATH_GPT_complete_the_square_l1611_161172

theorem complete_the_square :
  ∀ (x : ℝ), (x^2 + 14 * x + 24 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 25) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_complete_the_square_l1611_161172


namespace NUMINAMATH_GPT_percentage_increase_l1611_161139

variable (presentIncome : ℝ) (newIncome : ℝ)

theorem percentage_increase (h1 : presentIncome = 12000) (h2 : newIncome = 12240) :
  ((newIncome - presentIncome) / presentIncome) * 100 = 2 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1611_161139


namespace NUMINAMATH_GPT_greatest_common_multiple_of_9_and_15_less_than_120_l1611_161115

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_greatest_common_multiple_of_9_and_15_less_than_120_l1611_161115


namespace NUMINAMATH_GPT_largest_cube_edge_length_l1611_161132

theorem largest_cube_edge_length (a : ℕ) : 
  (6 * a ^ 2 ≤ 1500) ∧
  (a * 15 ≤ 60) ∧
  (a * 15 ≤ 25) →
  a ≤ 15 :=
by
  sorry

end NUMINAMATH_GPT_largest_cube_edge_length_l1611_161132


namespace NUMINAMATH_GPT_solve_for_a_l1611_161196

theorem solve_for_a (a : ℚ) (h : a + a/3 + a/4 = 11/4) : a = 33/19 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1611_161196


namespace NUMINAMATH_GPT_store_sales_correct_l1611_161150

def price_eraser_pencil : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

def quantity_eraser_pencil : ℕ := 200
def quantity_regular_pencil : ℕ := 40
def quantity_short_pencil : ℕ := 35
def quantity_mechanical_pencil : ℕ := 25
def quantity_novelty_pencil : ℕ := 15

def total_sales : ℝ :=
  (quantity_eraser_pencil * price_eraser_pencil) +
  (quantity_regular_pencil * price_regular_pencil) +
  (quantity_short_pencil * price_short_pencil) +
  (quantity_mechanical_pencil * price_mechanical_pencil) +
  (quantity_novelty_pencil * price_novelty_pencil)

theorem store_sales_correct : total_sales = 246.5 :=
by sorry

end NUMINAMATH_GPT_store_sales_correct_l1611_161150


namespace NUMINAMATH_GPT_trapezoid_PQRS_perimeter_l1611_161153

noncomputable def trapezoid_perimeter (PQ RS : ℝ) (height : ℝ) (PS QR : ℝ) : ℝ :=
  PQ + RS + PS + QR

theorem trapezoid_PQRS_perimeter :
  ∀ (PQ RS : ℝ) (height : ℝ)
  (PS QR : ℝ),
  PQ = 6 →
  RS = 10 →
  height = 5 →
  PS = Real.sqrt (5^2 + 4^2) →
  QR = Real.sqrt (5^2 + 4^2) →
  trapezoid_perimeter PQ RS height PS QR = 16 + 2 * Real.sqrt 41 :=
by
  intros
  sorry

end NUMINAMATH_GPT_trapezoid_PQRS_perimeter_l1611_161153


namespace NUMINAMATH_GPT_department_store_earnings_l1611_161176

theorem department_store_earnings :
  let original_price : ℝ := 1000000
  let discount_rate : ℝ := 0.1
  let prizes := [ (5, 1000), (10, 500), (20, 200), (40, 100), (5000, 10) ]
  let A_earnings := original_price * (1 - discount_rate)
  let total_prizes := prizes.foldl (fun sum (count, amount) => sum + count * amount) 0
  let B_earnings := original_price - total_prizes
  (B_earnings - A_earnings) >= 32000 := by
  sorry

end NUMINAMATH_GPT_department_store_earnings_l1611_161176


namespace NUMINAMATH_GPT_score_order_l1611_161171

theorem score_order (a b c d : ℕ) 
  (h1 : b + d = a + c)
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_score_order_l1611_161171


namespace NUMINAMATH_GPT_infinitely_many_perfect_squares_of_form_l1611_161166

theorem infinitely_many_perfect_squares_of_form (k : ℕ) (h : k > 0) : 
  ∃ (n : ℕ), ∃ m : ℕ, n * 2^k - 7 = m^2 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_perfect_squares_of_form_l1611_161166


namespace NUMINAMATH_GPT_john_needs_60_bags_l1611_161184

theorem john_needs_60_bags
  (horses : ℕ)
  (feeding_per_day : ℕ)
  (food_per_feeding : ℕ)
  (bag_weight : ℕ)
  (days : ℕ)
  (tons_in_pounds : ℕ)
  (half : ℕ)
  (h1 : horses = 25)
  (h2 : feeding_per_day = 2)
  (h3 : food_per_feeding = 20)
  (h4 : bag_weight = 1000)
  (h5 : days = 60)
  (h6 : tons_in_pounds = 2000)
  (h7 : half = 1 / 2) :
  ((horses * feeding_per_day * food_per_feeding * days) / (tons_in_pounds * half)) = 60 := by
  sorry

end NUMINAMATH_GPT_john_needs_60_bags_l1611_161184


namespace NUMINAMATH_GPT_youseff_blocks_l1611_161130

-- Definition of the conditions
def time_to_walk (x : ℕ) : ℕ := x
def time_to_ride (x : ℕ) : ℕ := (20 * x) / 60
def extra_time (x : ℕ) : ℕ := time_to_walk x - time_to_ride x

-- Statement of the problem in Lean
theorem youseff_blocks : ∃ x : ℕ, extra_time x = 6 ∧ x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_youseff_blocks_l1611_161130


namespace NUMINAMATH_GPT_solution_set_inequality_l1611_161106

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1611_161106


namespace NUMINAMATH_GPT_find_g_of_2_l1611_161157

theorem find_g_of_2 {g : ℝ → ℝ} (h : ∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) : g 2 = 14 :=
sorry

end NUMINAMATH_GPT_find_g_of_2_l1611_161157


namespace NUMINAMATH_GPT_parking_space_area_l1611_161140

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : 2 * W + L = 37) : L * W = 126 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_parking_space_area_l1611_161140


namespace NUMINAMATH_GPT_find_six_quotients_l1611_161125

def is_5twos_3ones (n: ℕ) : Prop :=
  n.digits 10 = [2, 2, 2, 2, 2, 1, 1, 1]

def divides_by_7 (n: ℕ) : Prop :=
  n % 7 = 0

theorem find_six_quotients:
  ∃ n₁ n₂ n₃ n₄ n₅: ℕ, 
    n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄ ∧ n₁ ≠ n₅ ∧ n₂ ≠ n₅ ∧ n₃ ≠ n₅ ∧ n₄ ≠ n₅ ∧
    is_5twos_3ones n₁ ∧ is_5twos_3ones n₂ ∧ is_5twos_3ones n₃ ∧ is_5twos_3ones n₄ ∧ is_5twos_3ones n₅ ∧
    divides_by_7 n₁ ∧ divides_by_7 n₂ ∧ divides_by_7 n₃ ∧ divides_by_7 n₄ ∧ divides_by_7 n₅ ∧
    n₁ / 7 = 1744603 ∧ n₂ / 7 = 3031603 ∧ n₃ / 7 = 3160303 ∧ n₄ / 7 = 3017446 ∧ n₅ / 7 = 3030316 :=
sorry

end NUMINAMATH_GPT_find_six_quotients_l1611_161125


namespace NUMINAMATH_GPT_total_nails_needed_l1611_161121

-- Given conditions
def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

-- Prove the total number of nails required
theorem total_nails_needed : nails_per_plank * number_of_planks = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_nails_needed_l1611_161121


namespace NUMINAMATH_GPT_max_girls_with_five_boys_l1611_161161

theorem max_girls_with_five_boys : 
  ∃ n : ℕ, n = 20 ∧ ∀ (boys : Fin 5 → ℝ × ℝ), 
  (∃ (girls : Fin n → ℝ × ℝ),
  (∀ i : Fin n, ∃ j k : Fin 5, j ≠ k ∧ dist (girls i) (boys j) = 5 ∧ dist (girls i) (boys k) = 5)) :=
sorry

end NUMINAMATH_GPT_max_girls_with_five_boys_l1611_161161


namespace NUMINAMATH_GPT_vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l1611_161155

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 15
def g (x a : ℝ) : ℝ := (2 - 2 * a) * x - f x

theorem vertex_and_segment_condition : 
  (f 1 = 16) ∧ ∃ x1 x2 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (x2 - x1 = 8) := 
sorry

theorem g_monotonically_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x1 a ≤ g x2 a) ↔ a ≤ 0 :=
sorry

theorem g_minimum_value (a : ℝ) :
  (0 < a ∧ g 2 a = -4 * a - 11) ∨ (a < 0 ∧ g 0 a = -15) ∨ (0 ≤ a ∧ a ≤ 2 ∧ g a a = -a^2 - 15) :=
sorry

end NUMINAMATH_GPT_vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l1611_161155


namespace NUMINAMATH_GPT_factor_expression_l1611_161198

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1611_161198


namespace NUMINAMATH_GPT_person_birth_year_and_age_l1611_161170

theorem person_birth_year_and_age (x y: ℕ) (h1: x ≤ 9) (h2: y ≤ 9) (hy: y = (88 - 10 * x) / (x + 1)):
  1988 - (1900 + 10 * x + y) = x * y → 1900 + 10 * x + y = 1964 ∧ 1988 - (1900 + 10 * x + y) = 24 :=
by
  sorry

end NUMINAMATH_GPT_person_birth_year_and_age_l1611_161170


namespace NUMINAMATH_GPT_wax_he_has_l1611_161138

def total_wax : ℕ := 353
def additional_wax : ℕ := 22

theorem wax_he_has : total_wax - additional_wax = 331 := by
  sorry

end NUMINAMATH_GPT_wax_he_has_l1611_161138


namespace NUMINAMATH_GPT_number_of_boys_in_second_grade_l1611_161160

-- conditions definition
variables (B : ℕ) (G2 : ℕ := 11) (G3 : ℕ := 2 * (B + G2)) (total : ℕ := B + G2 + G3)

-- mathematical statement to be proved
theorem number_of_boys_in_second_grade : total = 93 → B = 20 :=
by
  -- omitting the proof
  intro h_total
  sorry

end NUMINAMATH_GPT_number_of_boys_in_second_grade_l1611_161160


namespace NUMINAMATH_GPT_largest_of_three_consecutive_integers_sum_90_is_31_l1611_161168

theorem largest_of_three_consecutive_integers_sum_90_is_31 :
  ∃ (a b c : ℤ), (a + b + c = 90) ∧ (b = a + 1) ∧ (c = b + 1) ∧ (c = 31) :=
by
  sorry

end NUMINAMATH_GPT_largest_of_three_consecutive_integers_sum_90_is_31_l1611_161168


namespace NUMINAMATH_GPT_initial_beavers_l1611_161122

theorem initial_beavers (B C : ℕ) (h1 : C = 40) (h2 : B + C + 2 * B + (C - 10) = 130) : B = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_beavers_l1611_161122


namespace NUMINAMATH_GPT_cloth_cost_price_per_metre_l1611_161113

theorem cloth_cost_price_per_metre (total_metres : ℕ) (total_price : ℕ) (loss_per_metre : ℕ) :
  total_metres = 300 → total_price = 18000 → loss_per_metre = 5 → (total_price / total_metres + loss_per_metre) = 65 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cloth_cost_price_per_metre_l1611_161113


namespace NUMINAMATH_GPT_yanna_sandals_l1611_161100

theorem yanna_sandals (shirts_cost: ℕ) (sandal_cost: ℕ) (total_money: ℕ) (change: ℕ) (num_shirts: ℕ)
  (h1: shirts_cost = 5)
  (h2: sandal_cost = 3)
  (h3: total_money = 100)
  (h4: change = 41)
  (h5: num_shirts = 10) : 
  ∃ num_sandals: ℕ, num_sandals = 3 :=
sorry

end NUMINAMATH_GPT_yanna_sandals_l1611_161100


namespace NUMINAMATH_GPT_total_supervisors_l1611_161105

def buses : ℕ := 7
def supervisors_per_bus : ℕ := 3

theorem total_supervisors : buses * supervisors_per_bus = 21 := 
by
  have h : buses * supervisors_per_bus = 21 := by sorry
  exact h

end NUMINAMATH_GPT_total_supervisors_l1611_161105


namespace NUMINAMATH_GPT_seating_arrangement_l1611_161162

theorem seating_arrangement (x y : ℕ) (h : x + y ≤ 8) (h1 : 9 * x + 6 * y = 57) : x = 5 := 
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1611_161162


namespace NUMINAMATH_GPT_measure_8_liters_possible_l1611_161108

-- Define the types for buckets
structure Bucket :=
  (capacity : ℕ)
  (water : ℕ := 0)

-- Initial state with a 10-liter bucket and a 6-liter bucket, both empty
def B10_init := Bucket.mk 10 0
def B6_init := Bucket.mk 6 0

-- Define a function to check if we can measure 8 liters in B10
def can_measure_8_liters (B10 B6 : Bucket) : Prop :=
  (B10.water = 8 ∧ B10.capacity = 10 ∧ B6.capacity = 6)

-- The statement to prove there exists a sequence of operations to measure 8 liters in B10
theorem measure_8_liters_possible : ∃ (B10 B6 : Bucket), can_measure_8_liters B10 B6 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_measure_8_liters_possible_l1611_161108


namespace NUMINAMATH_GPT_muffin_expense_l1611_161167

theorem muffin_expense (B D : ℝ) 
    (h1 : D = 0.90 * B) 
    (h2 : B = D + 15) : 
    B + D = 285 := 
    sorry

end NUMINAMATH_GPT_muffin_expense_l1611_161167


namespace NUMINAMATH_GPT_smallest_b_l1611_161112

theorem smallest_b (a b : ℕ) (hp : a > 0) (hq : b > 0) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 8) : b = 4 :=
sorry

end NUMINAMATH_GPT_smallest_b_l1611_161112


namespace NUMINAMATH_GPT_prob1_prob2_l1611_161120

-- Define the polynomial function
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Problem 1: Prove |b| ≤ 1, given conditions
theorem prob1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : |b| ≤ 1 :=
sorry

-- Problem 2: Find a = 2, given conditions
theorem prob2 (a b c : ℝ) 
  (h1 : polynomial a b c 0 = -1) 
  (h2 : polynomial a b c 1 = 1) 
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |polynomial a b c x| ≤ 1) : 
  a = 2 :=
sorry

end NUMINAMATH_GPT_prob1_prob2_l1611_161120


namespace NUMINAMATH_GPT_waiters_dropped_out_l1611_161114

theorem waiters_dropped_out (initial_chefs initial_waiters chefs_dropped remaining_staff : ℕ)
  (h1 : initial_chefs = 16) 
  (h2 : initial_waiters = 16) 
  (h3 : chefs_dropped = 6) 
  (h4 : remaining_staff = 23) : 
  initial_waiters - (remaining_staff - (initial_chefs - chefs_dropped)) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_waiters_dropped_out_l1611_161114


namespace NUMINAMATH_GPT_number_of_boys_in_school_l1611_161141

theorem number_of_boys_in_school (x g : ℕ) (h1 : x + g = 400) (h2 : g = (x * 400) / 100) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l1611_161141


namespace NUMINAMATH_GPT_general_term_a_sum_of_bn_l1611_161165

-- Define sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b (n : ℕ) : ℚ := 1 / ((2 * n + 1) * (2 * n + 3))

-- Conditions
lemma condition_1 (n : ℕ) : a n > 0 := by sorry
lemma condition_2 (n : ℕ) : (a n)^2 + 2 * (a n) = 4 * (n * (n + 1)) + 3 := 
  by sorry

-- Theorem for question 1
theorem general_term_a (n : ℕ) : a n = 2 * n + 1 := by sorry

-- Theorem for question 2
theorem sum_of_bn (n : ℕ) : 
  (Finset.range n).sum b = (n : ℚ) / (6 * n + 9) := by sorry

end NUMINAMATH_GPT_general_term_a_sum_of_bn_l1611_161165


namespace NUMINAMATH_GPT_functional_expression_value_at_x_equals_zero_l1611_161117

-- Define the basic properties
def y_inversely_proportional_to_x_plus_2 (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x + 2)

-- Given condition: y = 3 when x = -1
def condition (y x : ℝ) : Prop :=
  y = 3 ∧ x = -1

-- Theorems to prove
theorem functional_expression (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → y = 3 / (x + 2) :=
by
  sorry

theorem value_at_x_equals_zero (y x : ℝ) :
  y_inversely_proportional_to_x_plus_2 y x ∧ condition y x → (y = 3 / (x + 2) ∧ x = 0 → y = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_functional_expression_value_at_x_equals_zero_l1611_161117


namespace NUMINAMATH_GPT_square_tiles_count_l1611_161199

theorem square_tiles_count (t s p : ℕ) (h1 : t + s + p = 30) (h2 : 3 * t + 4 * s + 5 * p = 108) : s = 6 := by
  sorry

end NUMINAMATH_GPT_square_tiles_count_l1611_161199


namespace NUMINAMATH_GPT_problem_1_l1611_161102

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |2 * x + 1| + |x - 2| ≥ a ^ 2 - a + (1 / 2)) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_l1611_161102


namespace NUMINAMATH_GPT_bananas_per_chimp_per_day_l1611_161136

theorem bananas_per_chimp_per_day (total_chimps total_bananas : ℝ) (h_chimps : total_chimps = 45) (h_bananas : total_bananas = 72) :
  total_bananas / total_chimps = 1.6 :=
by
  rw [h_chimps, h_bananas]
  norm_num

end NUMINAMATH_GPT_bananas_per_chimp_per_day_l1611_161136


namespace NUMINAMATH_GPT_parabola_equation_l1611_161185

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l1611_161185


namespace NUMINAMATH_GPT_find_f_of_7_l1611_161145

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem find_f_of_7 (h1 : is_odd_function f)
                    (h2 : is_periodic_function f 4)
                    (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_f_of_7_l1611_161145


namespace NUMINAMATH_GPT_jessica_withdrew_200_l1611_161197

noncomputable def initial_balance (final_balance : ℝ) : ℝ :=
  (final_balance * 25 / 18)

noncomputable def withdrawn_amount (initial_balance : ℝ) : ℝ :=
  (initial_balance * 2 / 5)

theorem jessica_withdrew_200 :
  ∀ (final_balance : ℝ), final_balance = 360 → withdrawn_amount (initial_balance final_balance) = 200 :=
by
  intros final_balance h
  rw [h]
  unfold initial_balance withdrawn_amount
  sorry

end NUMINAMATH_GPT_jessica_withdrew_200_l1611_161197


namespace NUMINAMATH_GPT_mary_income_percentage_more_than_tim_l1611_161128

variables (J T M : ℝ)
-- Define the conditions
def condition1 := T = 0.5 * J -- Tim's income is 50% less than Juan's
def condition2 := M = 0.8 * J -- Mary's income is 80% of Juan's

-- Define the theorem stating the question and the correct answer
theorem mary_income_percentage_more_than_tim (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 0.8 * J) : 
  (M - T) / T * 100 = 60 := 
  by sorry

end NUMINAMATH_GPT_mary_income_percentage_more_than_tim_l1611_161128
