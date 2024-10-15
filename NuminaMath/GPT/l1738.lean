import Mathlib

namespace NUMINAMATH_GPT_number_of_students_like_basketball_but_not_table_tennis_l1738_173809

-- Given definitions
def total_students : Nat := 40
def students_like_basketball : Nat := 24
def students_like_table_tennis : Nat := 16
def students_dislike_both : Nat := 6

-- Proposition to prove
theorem number_of_students_like_basketball_but_not_table_tennis : 
  students_like_basketball - (students_like_basketball + students_like_table_tennis - (total_students - students_dislike_both)) = 18 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_like_basketball_but_not_table_tennis_l1738_173809


namespace NUMINAMATH_GPT_point_above_line_range_l1738_173855

theorem point_above_line_range (a : ℝ) :
  (2 * a - (-1) + 1 < 0) ↔ a < -1 :=
by
  sorry

end NUMINAMATH_GPT_point_above_line_range_l1738_173855


namespace NUMINAMATH_GPT_domain_width_of_g_l1738_173888

theorem domain_width_of_g (h : ℝ → ℝ) (domain_h : ∀ x, -8 ≤ x ∧ x ≤ 8 → h x = h x) :
  let g (x : ℝ) := h (x / 2)
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → ∃ y, g x = y) ∧ (b - a = 32) := 
sorry

end NUMINAMATH_GPT_domain_width_of_g_l1738_173888


namespace NUMINAMATH_GPT_digit_D_eq_9_l1738_173849

-- Define digits and the basic operations on 2-digit numbers
def is_digit (n : ℕ) : Prop := n < 10
def tens (n : ℕ) : ℕ := n / 10
def units (n : ℕ) : ℕ := n % 10
def two_digit (a b : ℕ) : ℕ := 10 * a + b

theorem digit_D_eq_9 (A B C D : ℕ):
  is_digit A → is_digit B → is_digit C → is_digit D →
  (two_digit A B) + (two_digit C B) = two_digit D A →
  (two_digit A B) - (two_digit C B) = A →
  D = 9 :=
by sorry

end NUMINAMATH_GPT_digit_D_eq_9_l1738_173849


namespace NUMINAMATH_GPT_convert_base10_to_base7_l1738_173874

-- Definitions for powers and conditions
def n1 : ℕ := 7
def n2 : ℕ := n1 * n1
def n3 : ℕ := n2 * n1
def n4 : ℕ := n3 * n1

theorem convert_base10_to_base7 (n : ℕ) (h₁ : n = 395) : 
  ∃ a b c d : ℕ, 
    a * n3 + b * n2 + c * n1 + d = 395 ∧
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 3 :=
by { sorry }

end NUMINAMATH_GPT_convert_base10_to_base7_l1738_173874


namespace NUMINAMATH_GPT_f_x_when_x_negative_l1738_173803

-- Define the properties of the function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → f x = x * (1 + x)

-- The theorem we want to prove
theorem f_x_when_x_negative (f : ℝ → ℝ) 
  (h1: odd_function f)
  (h2: f_definition f) : 
  ∀ x, x < 0 → f x = -x * (1 - x) :=
by
  sorry

end NUMINAMATH_GPT_f_x_when_x_negative_l1738_173803


namespace NUMINAMATH_GPT_largest_divisor_of_n4_minus_n_l1738_173872

theorem largest_divisor_of_n4_minus_n (n : ℤ) (h : ∃ k : ℤ, n = 4 * k) : 4 ∣ (n^4 - n) :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_n4_minus_n_l1738_173872


namespace NUMINAMATH_GPT_personBCatchesPersonAAtB_l1738_173879

-- Definitions based on the given problem's conditions
def personADepartsTime : ℕ := 8 * 60  -- Person A departs at 8:00 AM, given in minutes
def personBDepartsTime : ℕ := 9 * 60  -- Person B departs at 9:00 AM, given in minutes
def catchUpTime : ℕ := 11 * 60        -- Persons meet at 11:00 AM, given in minutes
def returnMultiplier : ℕ := 2         -- Person B returns at double the speed
def chaseMultiplier : ℕ := 2          -- After returning, Person B doubles their speed again

-- Exact question we want to prove
def meetAtBTime : ℕ := 12 * 60 + 48   -- Time when Person B catches up with Person A at point B

-- Statement to be proven
theorem personBCatchesPersonAAtB :
  ∀ (VA VB : ℕ) (x : ℕ),
    VA = 2 * x ∧ VB = 3 * x →
    ∃ t : ℕ, t = meetAtBTime := by
  sorry

end NUMINAMATH_GPT_personBCatchesPersonAAtB_l1738_173879


namespace NUMINAMATH_GPT_tangent_line_eq_l1738_173894

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x^2 + 1) : 
  (x = -1 ∧ y = 3) → (4 * x + y + 1 = 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1738_173894


namespace NUMINAMATH_GPT_jerry_removed_old_figures_l1738_173853

-- Let's declare the conditions
variables (initial_count added_count current_count removed_count : ℕ)
variables (h1 : initial_count = 7)
variables (h2 : added_count = 11)
variables (h3 : current_count = 8)

-- The statement to prove
theorem jerry_removed_old_figures : removed_count = initial_count + added_count - current_count :=
by
  -- The proof will go here, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_jerry_removed_old_figures_l1738_173853


namespace NUMINAMATH_GPT_child_wants_to_buy_3_toys_l1738_173861

/- 
  Problem Statement:
  There are 10 toys, and the number of ways to select a certain number 
  of those toys in any order is 120. We need to find out how many toys 
  were selected.
-/

def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem child_wants_to_buy_3_toys :
  ∃ r : ℕ, r ≤ 10 ∧ comb 10 r = 120 :=
by
  use 3
  -- Here you would write the proof
  sorry

end NUMINAMATH_GPT_child_wants_to_buy_3_toys_l1738_173861


namespace NUMINAMATH_GPT_hyperbola_equations_l1738_173825

def eq1 (x y : ℝ) : Prop := x^2 - 4 * y^2 = (5 + Real.sqrt 6)^2
def eq2 (x y : ℝ) : Prop := 4 * y^2 - x^2 = 4

theorem hyperbola_equations 
  (x y : ℝ)
  (hx1 : x - 2 * y = 0)
  (hx2 : x + 2 * y = 0)
  (dist : Real.sqrt ((x - 5)^2 + y^2) = Real.sqrt 6) :
  eq1 x y ∧ eq2 x y := 
sorry

end NUMINAMATH_GPT_hyperbola_equations_l1738_173825


namespace NUMINAMATH_GPT_smallest_right_triangle_area_l1738_173805

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end NUMINAMATH_GPT_smallest_right_triangle_area_l1738_173805


namespace NUMINAMATH_GPT_factor_3x2_minus_3y2_l1738_173829

theorem factor_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_GPT_factor_3x2_minus_3y2_l1738_173829


namespace NUMINAMATH_GPT_walk_to_bus_stop_usual_time_l1738_173817

variable (S : ℝ) -- assuming S is the usual speed, a positive real number
variable (T : ℝ) -- assuming T is the usual time, which we need to determine
variable (new_speed : ℝ := (4 / 5) * S) -- the new speed is 4/5 of usual speed
noncomputable def time_to_bus_at_usual_speed : ℝ := T -- time to bus stop at usual speed

theorem walk_to_bus_stop_usual_time :
  (time_to_bus_at_usual_speed S = 30) ↔ (S * (T + 6) = (4 / 5) * S * T) :=
by
  sorry

end NUMINAMATH_GPT_walk_to_bus_stop_usual_time_l1738_173817


namespace NUMINAMATH_GPT_Berry_temperature_on_Sunday_l1738_173828

theorem Berry_temperature_on_Sunday :
  let avg_temp := 99.0
  let days_in_week := 7
  let temp_day1 := 98.2
  let temp_day2 := 98.7
  let temp_day3 := 99.3
  let temp_day4 := 99.8
  let temp_day5 := 99.0
  let temp_day6 := 98.9
  let total_temp_week := avg_temp * days_in_week
  let total_temp_six_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5 + temp_day6
  let temp_on_sunday := total_temp_week - total_temp_six_days
  temp_on_sunday = 98.1 :=
by
  -- Proof of the statement goes here
  sorry

end NUMINAMATH_GPT_Berry_temperature_on_Sunday_l1738_173828


namespace NUMINAMATH_GPT_max_ab_plus_2bc_l1738_173876

theorem max_ab_plus_2bc (A B C : ℝ) (AB AC BC : ℝ) (hB : B = 60) (hAC : AC = Real.sqrt 3) :
  (AB + 2 * BC) ≤ 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_max_ab_plus_2bc_l1738_173876


namespace NUMINAMATH_GPT_rational_numbers_sum_reciprocal_integer_l1738_173891

theorem rational_numbers_sum_reciprocal_integer (p1 q1 p2 q2 : ℤ) (k m : ℤ)
  (h1 : Int.gcd p1 q1 = 1)
  (h2 : Int.gcd p2 q2 = 1)
  (h3 : p1 * q2 + p2 * q1 = k * q1 * q2)
  (h4 : q1 * p2 + q2 * p1 = m * p1 * p2) :
  (p1, q1, p2, q2) = (x, y, -x, y) ∨
  (p1, q1, p2, q2) = (2, 1, 2, 1) ∨
  (p1, q1, p2, q2) = (-2, 1, -2, 1) ∨
  (p1, q1, p2, q2) = (1, 1, 1, 1) ∨
  (p1, q1, p2, q2) = (-1, 1, -1, 1) ∨
  (p1, q1, p2, q2) = (1, 2, 1, 2) ∨
  (p1, q1, p2, q2) = (-1, 2, -1, 2) :=
sorry

end NUMINAMATH_GPT_rational_numbers_sum_reciprocal_integer_l1738_173891


namespace NUMINAMATH_GPT_loss_per_meter_is_five_l1738_173869

def cost_price_per_meter : ℝ := 50
def total_meters_sold : ℝ := 400
def selling_price : ℝ := 18000

noncomputable def total_cost_price : ℝ := cost_price_per_meter * total_meters_sold
noncomputable def total_loss : ℝ := total_cost_price - selling_price
noncomputable def loss_per_meter : ℝ := total_loss / total_meters_sold

theorem loss_per_meter_is_five : loss_per_meter = 5 :=
by sorry

end NUMINAMATH_GPT_loss_per_meter_is_five_l1738_173869


namespace NUMINAMATH_GPT_binomial_expansion_l1738_173821

theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (1 + 2 * 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 ∧
  (1 + 2 * -1)^5 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 → 
  a_0 + a_2 + a_4 = 121 :=
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end NUMINAMATH_GPT_binomial_expansion_l1738_173821


namespace NUMINAMATH_GPT_part1_part2_part3_l1738_173885

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem part1 : ∃ x : ℝ, (h x ≤ 2) := sorry

theorem part2 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) := sorry

theorem part3 (k : ℤ) : (∀ x : ℝ, x > 1 → k * (x - 1) < x * f x + 3 * g' x + 4) ↔ k ≤ 5 := sorry

end NUMINAMATH_GPT_part1_part2_part3_l1738_173885


namespace NUMINAMATH_GPT_hiking_supplies_l1738_173866

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end NUMINAMATH_GPT_hiking_supplies_l1738_173866


namespace NUMINAMATH_GPT_solution_alcohol_content_l1738_173810

noncomputable def volume_of_solution_y_and_z (V: ℝ) : Prop :=
  let vol_X := 300.0
  let conc_X := 0.10
  let conc_Y := 0.30
  let conc_Z := 0.40
  let vol_Y := 2 * V
  let vol_new := vol_X + vol_Y + V
  let alcohol_new := conc_X * vol_X + conc_Y * vol_Y + conc_Z * V
  (alcohol_new / vol_new) = 0.22

theorem solution_alcohol_content : volume_of_solution_y_and_z 300.0 :=
by
  sorry

end NUMINAMATH_GPT_solution_alcohol_content_l1738_173810


namespace NUMINAMATH_GPT_range_of_m_l1738_173886

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x - 3| ≤ 2 → 1 ≤ x ∧ x ≤ 5) → 
  (∀ x : ℝ, (x - m + 1) * (x - m - 1) ≤ 0 → m - 1 ≤ x ∧ x ≤ m + 1) → 
  (∀ x : ℝ, x < 1 ∨ x > 5 → x < m - 1 ∨ x > m + 1) → 
  2 ≤ m ∧ m ≤ 4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1738_173886


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1738_173893

theorem arithmetic_sequence_product (a d : ℕ) :
  (a + 7 * d = 20) → (d = 2) → ((a + d) * (a + 2 * d) = 80) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1738_173893


namespace NUMINAMATH_GPT_lychee_ratio_l1738_173859

theorem lychee_ratio (total_lychees : ℕ) (sold_lychees : ℕ) (remaining_home : ℕ) (remaining_after_eat : ℕ) 
    (h1: total_lychees = 500) 
    (h2: sold_lychees = total_lychees / 2) 
    (h3: remaining_home = total_lychees - sold_lychees) 
    (h4: remaining_after_eat = 100)
    (h5: remaining_after_eat + (remaining_home - remaining_after_eat) = remaining_home) : 
    (remaining_home - remaining_after_eat) / remaining_home = 3 / 5 :=
by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_lychee_ratio_l1738_173859


namespace NUMINAMATH_GPT_find_a_and_b_l1738_173897

theorem find_a_and_b (a b : ℝ) (h1 : b ≠ 0) 
  (h2 : (ab = a + b ∨ ab = a - b ∨ ab = a / b) 
  ∧ (a + b = a - b ∨ a + b = a / b) 
  ∧ (a - b = a / b)) : 
  (a = 1 / 2 ∨ a = -1 / 2) ∧ b = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1738_173897


namespace NUMINAMATH_GPT_polynomial_expansion_sum_is_21_l1738_173880

theorem polynomial_expansion_sum_is_21 :
  ∃ (A B C D : ℤ), (∀ (x : ℤ), (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) ∧
  A + B + C + D = 21 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_is_21_l1738_173880


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1738_173850

noncomputable def total_shaded_area : ℝ :=
  let s := 10
  let R := s / (2 * Real.sin (Real.pi / 8))
  let A := (1 / 2) * R^2 * Real.sin (2 * Real.pi / 8)
  4 * A

theorem shaded_area_is_correct :
  total_shaded_area = 200 * Real.sqrt 2 / Real.sin (Real.pi / 8)^2 := 
sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1738_173850


namespace NUMINAMATH_GPT_prob_heads_even_correct_l1738_173856

noncomputable def prob_heads_even (n : Nat) : ℝ :=
  if n = 0 then 1
  else (2 / 3) - (1 / 3) * prob_heads_even (n - 1)

theorem prob_heads_even_correct : 
  prob_heads_even 50 = (1 / 2) * (1 + (1 / 3 ^ 50)) :=
sorry

end NUMINAMATH_GPT_prob_heads_even_correct_l1738_173856


namespace NUMINAMATH_GPT_pencils_bought_l1738_173813

theorem pencils_bought (total_spent notebook_cost ruler_cost pencil_cost : ℕ)
  (h_total : total_spent = 74)
  (h_notebook : notebook_cost = 35)
  (h_ruler : ruler_cost = 18)
  (h_pencil : pencil_cost = 7) :
  (total_spent - (notebook_cost + ruler_cost)) / pencil_cost = 3 :=
by
  sorry

end NUMINAMATH_GPT_pencils_bought_l1738_173813


namespace NUMINAMATH_GPT_elizabeth_net_profit_l1738_173831

noncomputable section

def net_profit : ℝ :=
  let cost_bag_1 := 2.5
  let cost_bag_2 := 3.5
  let total_cost := 10 * cost_bag_1 + 10 * cost_bag_2
  let selling_price := 6.0
  let sold_bags_1_no_discount := 7 * selling_price
  let sold_bags_2_no_discount := 8 * selling_price
  let discount_1 := 0.2
  let discount_2 := 0.3
  let discounted_price_1 := selling_price * (1 - discount_1)
  let discounted_price_2 := selling_price * (1 - discount_2)
  let sold_bags_1_with_discount := 3 * discounted_price_1
  let sold_bags_2_with_discount := 2 * discounted_price_2
  let total_revenue := sold_bags_1_no_discount + sold_bags_2_no_discount + sold_bags_1_with_discount + sold_bags_2_with_discount
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.8 := by
  sorry

end NUMINAMATH_GPT_elizabeth_net_profit_l1738_173831


namespace NUMINAMATH_GPT_original_manufacturing_cost_l1738_173845

variable (SP OC : ℝ)
variable (ManuCost : ℝ) -- Declaring manufacturing cost

-- Current conditions
axiom profit_percentage_constant : ∀ SP, 0.5 * SP = SP - 50

-- Problem Statement
theorem original_manufacturing_cost : (∃ OC, 0.5 * SP - OC = 0.5 * SP) ∧ ManuCost = 50 → OC = 50 := by
  sorry

end NUMINAMATH_GPT_original_manufacturing_cost_l1738_173845


namespace NUMINAMATH_GPT_sequence_an_eq_n_l1738_173895

theorem sequence_an_eq_n (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, n ≥ 1 → a n > 0) 
  (h₁ : ∀ n, n ≥ 1 → a n + 1 / 2 = Real.sqrt (2 * S n + 1 / 4)) : 
  ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end NUMINAMATH_GPT_sequence_an_eq_n_l1738_173895


namespace NUMINAMATH_GPT_axis_of_symmetry_of_parabola_l1738_173840

-- Definitions (from conditions):
def quadratic_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root_of_quadratic (a b c x : ℝ) : Prop := quadratic_equation a b c x = 0

-- Given conditions
variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)
variable (h_root1 : is_root_of_quadratic a b c 1)
variable (h_root2 : is_root_of_quadratic a b c 5)

-- Problem statement
theorem axis_of_symmetry_of_parabola : (3 : ℝ) = (1 + 5) / 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_of_parabola_l1738_173840


namespace NUMINAMATH_GPT_parity_of_f_monotonicity_of_f_9_l1738_173830

-- Condition: f(x) = x + k / x with k ≠ 0
variable (k : ℝ) (hkn0 : k ≠ 0)
noncomputable def f (x : ℝ) : ℝ := x + k / x

-- 1. Prove the parity of the function is odd
theorem parity_of_f : ∀ x : ℝ, f k (-x) = -f k x := by
  sorry

-- Given condition: f(3) = 6, we derive k = 9
def k_9 : ℝ := 9
noncomputable def f_9 (x : ℝ) : ℝ := x + k_9 / x

-- 2. Prove the monotonicity of the function y = f(x) in the interval (-∞, -3]
theorem monotonicity_of_f_9 : ∀ (x1 x2 : ℝ), x1 < x2 → x1 ≤ -3 → x2 ≤ -3 → f_9 x1 < f_9 x2 := by
  sorry

end NUMINAMATH_GPT_parity_of_f_monotonicity_of_f_9_l1738_173830


namespace NUMINAMATH_GPT_average_runs_in_30_matches_l1738_173815

theorem average_runs_in_30_matches (avg_runs_15: ℕ) (avg_runs_20: ℕ) 
    (matches_15: ℕ) (matches_20: ℕ)
    (h1: avg_runs_15 = 30) (h2: avg_runs_20 = 15)
    (h3: matches_15 = 15) (h4: matches_20 = 20) : 
    (matches_15 * avg_runs_15 + matches_20 * avg_runs_20) / (matches_15 + matches_20) = 25 := 
by 
  sorry

end NUMINAMATH_GPT_average_runs_in_30_matches_l1738_173815


namespace NUMINAMATH_GPT_transistors_in_2002_transistors_in_2010_l1738_173875

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end NUMINAMATH_GPT_transistors_in_2002_transistors_in_2010_l1738_173875


namespace NUMINAMATH_GPT_min_time_adult_worms_l1738_173863

noncomputable def f : ℕ → ℝ
| 1 => 0
| n => (1 - 1 / (2 ^ (n - 1)))

theorem min_time_adult_worms (n : ℕ) (h : n ≥ 1) : 
  ∃ min_time : ℝ, 
  (min_time = 1 - 1 / (2 ^ (n - 1))) ∧ 
  (∀ t : ℝ, (t = 1 - 1 / (2 ^ (n - 1)))) := 
sorry

end NUMINAMATH_GPT_min_time_adult_worms_l1738_173863


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_l1738_173873

theorem hyperbola_asymptote_slope (m : ℝ) :
  (∀ x y : ℝ, mx^2 + y^2 = 1) →
  (∀ x y : ℝ, y = 2 * x) →
  m = -4 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slope_l1738_173873


namespace NUMINAMATH_GPT_percentage_problem_l1738_173814

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_problem_l1738_173814


namespace NUMINAMATH_GPT_coin_collection_l1738_173804

def initial_ratio (G S : ℕ) : Prop := G = S / 3
def new_ratio (G S : ℕ) (addedG : ℕ) : Prop := G + addedG = S / 2
def total_coins_after (G S addedG : ℕ) : ℕ := G + addedG + S

theorem coin_collection (G S : ℕ) (addedG : ℕ) 
  (h1 : initial_ratio G S) 
  (h2 : addedG = 15) 
  (h3 : new_ratio G S addedG) : 
  total_coins_after G S addedG = 135 := 
by {
  sorry
}

end NUMINAMATH_GPT_coin_collection_l1738_173804


namespace NUMINAMATH_GPT_min_candidates_for_same_score_l1738_173898

theorem min_candidates_for_same_score :
  (∃ S : ℕ, S ≥ 25 ∧ (∀ elect : Fin S → Fin 12, ∃ s : Fin 12, ∃ a b c : Fin S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ elect a = s ∧ elect b = s ∧ elect c = s)) := 
sorry

end NUMINAMATH_GPT_min_candidates_for_same_score_l1738_173898


namespace NUMINAMATH_GPT_gasoline_expense_l1738_173801

-- Definitions for the conditions
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10
def initial_amount : ℝ := 50
def amount_left_for_return_trip : ℝ := 36.35

-- Definition for the total gift cost
def total_gift_cost : ℝ := number_of_people * gift_cost_per_person

-- Definition for the total amount received from grandma
def total_grandma_gift : ℝ := number_of_people * grandma_gift_per_person

-- Definition for the total initial amount including the gift from grandma
def total_initial_amount_with_gift : ℝ := initial_amount + total_grandma_gift

-- Definition for remaining amount after spending on lunch and gifts
def remaining_after_known_expenses : ℝ := total_initial_amount_with_gift - lunch_cost - total_gift_cost

-- The Lean theorem to prove the gasoline expense
theorem gasoline_expense : remaining_after_known_expenses - amount_left_for_return_trip = 8 := by
  sorry

end NUMINAMATH_GPT_gasoline_expense_l1738_173801


namespace NUMINAMATH_GPT_remainder_when_divided_by_five_l1738_173841

theorem remainder_when_divided_by_five :
  let E := 1250 * 1625 * 1830 * 2075 + 245
  E % 5 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_five_l1738_173841


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l1738_173842

variable (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : g + b = 36)
                               (h₂ : g = b + 6) : g / b = 7 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l1738_173842


namespace NUMINAMATH_GPT_find_number_l1738_173899

theorem find_number (x : ℕ) (h : x + 5 * 8 = 340) : x = 300 :=
sorry

end NUMINAMATH_GPT_find_number_l1738_173899


namespace NUMINAMATH_GPT_initial_alarm_time_was_l1738_173882

def faster_watch_gain (rate : ℝ) (hours : ℝ) : ℝ := hours * rate

def absolute_time_difference (faster_time : ℝ) (correct_time : ℝ) : ℝ := faster_time - correct_time

theorem initial_alarm_time_was :
  ∀ (rate minutes time_difference : ℝ),
  rate = 2 →
  minutes = 12 →
  time_difference = minutes / rate →
  abs (4 - (4 - time_difference)) = 6 →
  (24 - 6) = 22 :=
by
  intros rate minutes time_difference hrate hminutes htime_diff htime
  sorry

end NUMINAMATH_GPT_initial_alarm_time_was_l1738_173882


namespace NUMINAMATH_GPT_molecular_weight_calculation_l1738_173837

/-- Define the molecular weight of the compound as 972 grams per mole. -/
def molecular_weight : ℕ := 972

/-- Define the number of moles as 9 moles. -/
def number_of_moles : ℕ := 9

/-- Define the total weight of the compound for the given number of moles. -/
def total_weight : ℕ := number_of_moles * molecular_weight

/-- Prove the total weight is 8748 grams. -/
theorem molecular_weight_calculation : total_weight = 8748 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_calculation_l1738_173837


namespace NUMINAMATH_GPT_expression_eval_l1738_173867

noncomputable def a : ℕ := 2001
noncomputable def b : ℕ := 2003

theorem expression_eval : 
  b^3 - a * b^2 - a^2 * b + a^3 = 8 :=
by sorry

end NUMINAMATH_GPT_expression_eval_l1738_173867


namespace NUMINAMATH_GPT_midpoint_trajectory_l1738_173824

-- Define the data for the problem
variable (P : (ℝ × ℝ)) (Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable (hQ : Q = (2*x - 2, 2*y)) -- Definition of point Q based on midpoint M
variable (hC : (Q.1)^2 + (Q.2)^2 = 1) -- Q moves on the circle x^2 + y^2 = 1

-- Define the proof problem
theorem midpoint_trajectory (P : (ℝ × ℝ)) (hP : P = (2, 0)) (M : ℝ × ℝ) (hQ : Q = (2*M.1 - 2, 2*M.2))
  (hC : (Q.1)^2 + (Q.2)^2 = 1) : 4*(M.1 - 1)^2 + 4*(M.2)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_midpoint_trajectory_l1738_173824


namespace NUMINAMATH_GPT_range_of_a_l1738_173887

-- Define the propositions p and q
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the main theorem which combines both propositions and infers the range of a
theorem range_of_a (a : ℝ) : prop_p a ∧ prop_q a → a ≤ -2 := sorry

end NUMINAMATH_GPT_range_of_a_l1738_173887


namespace NUMINAMATH_GPT_worker_weekly_pay_l1738_173835

variable (regular_rate : ℕ) -- Regular rate of Rs. 10 per survey
variable (total_surveys : ℕ) -- Worker completes 100 surveys per week
variable (cellphone_surveys : ℕ) -- 60 surveys involve the use of cellphone
variable (increased_rate : ℕ) -- Increased rate 30% higher than regular rate

-- Defining given values
def reg_rate : ℕ := 10
def total_survey_count : ℕ := 100
def cellphone_survey_count : ℕ := 60
def inc_rate : ℕ := reg_rate + 3

-- Calculating payments
def regular_survey_count : ℕ := total_survey_count - cellphone_survey_count
def regular_pay : ℕ := regular_survey_count * reg_rate
def cellphone_pay : ℕ := cellphone_survey_count * inc_rate

-- Total pay calculation
def total_pay : ℕ := regular_pay + cellphone_pay

-- Theorem to be proved
theorem worker_weekly_pay : total_pay = 1180 := 
by
  -- instantiate variables
  let regular_rate := reg_rate
  let total_surveys := total_survey_count
  let cellphone_surveys := cellphone_survey_count
  let increased_rate := inc_rate
  
  -- skip proof
  sorry

end NUMINAMATH_GPT_worker_weekly_pay_l1738_173835


namespace NUMINAMATH_GPT_div30k_929260_l1738_173858

theorem div30k_929260 (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 1 := by
  sorry

end NUMINAMATH_GPT_div30k_929260_l1738_173858


namespace NUMINAMATH_GPT_root_conditions_l1738_173857

-- Given conditions and definitions:
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m - 3) * x + m

-- The proof problem statement
theorem root_conditions (m : ℝ) (h1 : ∃ x y : ℝ, quadratic_eq m x = 0 ∧ quadratic_eq m y = 0 ∧ x > 1 ∧ y < 1) : m < 1 :=
sorry

end NUMINAMATH_GPT_root_conditions_l1738_173857


namespace NUMINAMATH_GPT_maximize_quadratic_function_l1738_173838

theorem maximize_quadratic_function (x : ℝ) :
  (∀ x, -2 * x ^ 2 - 8 * x + 18 ≤ 26) ∧ (-2 * (-2) ^ 2 - 8 * (-2) + 18 = 26) :=
by (
  sorry
)

end NUMINAMATH_GPT_maximize_quadratic_function_l1738_173838


namespace NUMINAMATH_GPT_quadrilateral_diagonals_l1738_173892

-- Define the points of the quadrilateral
variables {A B C D P Q R S : ℝ × ℝ}

-- Define the midpoints condition
def is_midpoint (M : ℝ × ℝ) (X Y : ℝ × ℝ) := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Define the lengths squared condition
def dist_sq (X Y : ℝ × ℝ) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Main theorem to prove
theorem quadrilateral_diagonals (hP : is_midpoint P A B) (hQ : is_midpoint Q B C)
  (hR : is_midpoint R C D) (hS : is_midpoint S D A) :
  dist_sq A C + dist_sq B D = 2 * (dist_sq P R + dist_sq Q S) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_diagonals_l1738_173892


namespace NUMINAMATH_GPT_problem_statement_l1738_173862

theorem problem_statement (a b c : ℝ)
  (h : a * b * c = ( Real.sqrt ( (a + 2) * (b + 3) ) ) / (c + 1)) :
  6 * 15 * 7 = 1.5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1738_173862


namespace NUMINAMATH_GPT_intersection_l1738_173819

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

noncomputable def g (x : ℝ) (a b c d k : ℝ) : ℝ := -2 * x - 4 + k / (x - d)

theorem intersection (a b c k : ℝ) (h_d : d = 3) (h_k : k = 36) : 
  ∃ (x y : ℝ), x ≠ -3 ∧ (f x = g x 0 0 0 d k) ∧ (x, y) = (6.8, -32 / 19) :=
by
  sorry

end NUMINAMATH_GPT_intersection_l1738_173819


namespace NUMINAMATH_GPT_gross_profit_percentage_l1738_173834

theorem gross_profit_percentage (sales_price gross_profit : ℝ) (h_sales_price : sales_price = 91) (h_gross_profit : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 :=
by
  sorry

end NUMINAMATH_GPT_gross_profit_percentage_l1738_173834


namespace NUMINAMATH_GPT_length_minus_width_l1738_173852

theorem length_minus_width 
  (area length diff width : ℝ)
  (h_area : area = 171)
  (h_length : length = 19.13)
  (h_diff : diff = length - width)
  (h_area_eq : area = length * width) :
  diff = 10.19 := 
by {
  sorry
}

end NUMINAMATH_GPT_length_minus_width_l1738_173852


namespace NUMINAMATH_GPT_calculate_f_f_f_one_l1738_173870

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem calculate_f_f_f_one : f (f (f 1)) = 9184 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_f_f_one_l1738_173870


namespace NUMINAMATH_GPT_complete_square_solution_l1738_173871

theorem complete_square_solution (x : ℝ) :
  (x^2 + 6 * x - 4 = 0) → ((x + 3)^2 = 13) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l1738_173871


namespace NUMINAMATH_GPT_charity_event_assignment_l1738_173826

theorem charity_event_assignment (students : Finset ℕ) (h_students : students.card = 5) :
  ∃ (num_ways : ℕ), num_ways = 60 :=
by
  let select_two_for_friday := Nat.choose 5 2
  let remaining_students_after_friday := 5 - 2
  let select_one_for_saturday := Nat.choose remaining_students_after_friday 1
  let remaining_students_after_saturday := remaining_students_after_friday - 1
  let select_one_for_sunday := Nat.choose remaining_students_after_saturday 1
  let total_ways := select_two_for_friday * select_one_for_saturday * select_one_for_sunday
  use total_ways
  sorry

end NUMINAMATH_GPT_charity_event_assignment_l1738_173826


namespace NUMINAMATH_GPT_correct_operation_l1738_173808

variable (a b : ℝ)

theorem correct_operation : (a^2 * a^3 = a^5) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1738_173808


namespace NUMINAMATH_GPT_new_pressure_eq_l1738_173812

-- Defining the initial conditions and values
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3.5
def new_volume : ℝ := 10.5
def k : ℝ := initial_pressure * initial_volume

-- The statement to prove
theorem new_pressure_eq :
  ∃ p_new : ℝ, new_volume * p_new = k ∧ p_new = 8 / 3 :=
by
  use (8 / 3)
  sorry

end NUMINAMATH_GPT_new_pressure_eq_l1738_173812


namespace NUMINAMATH_GPT_farmer_initial_productivity_l1738_173802

theorem farmer_initial_productivity (x : ℝ) (d : ℝ)
  (hx1 : d = 1440 / x)
  (hx2 : 2 * x + (d - 4) * 1.25 * x = 1440) :
  x = 120 :=
by
  sorry

end NUMINAMATH_GPT_farmer_initial_productivity_l1738_173802


namespace NUMINAMATH_GPT_find_k_for_circle_of_radius_8_l1738_173807

theorem find_k_for_circle_of_radius_8 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ∧ (∀ r : ℝ, r = 8) → k = -1 :=
sorry

end NUMINAMATH_GPT_find_k_for_circle_of_radius_8_l1738_173807


namespace NUMINAMATH_GPT_cos_sin_sequence_rational_l1738_173884

variable (α : ℝ) (h₁ : ∃ r : ℚ, r = (Real.sin α + Real.cos α))

theorem cos_sin_sequence_rational :
    (∀ n : ℕ, n > 0 → ∃ r : ℚ, r = (Real.cos α)^n + (Real.sin α)^n) :=
by
  sorry

end NUMINAMATH_GPT_cos_sin_sequence_rational_l1738_173884


namespace NUMINAMATH_GPT_inequality_proof_l1738_173883

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)^2) + (1 / (1 + b)^2) + (1 / (1 + c)^2) + (1 / (1 + d)^2) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1738_173883


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1738_173822

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (2 * x^2 + x - 1 ≥ 0) → (x ≥ 1/2) ∨ (x ≤ -1) :=
by
  -- The given inequality and condition imply this result.
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1738_173822


namespace NUMINAMATH_GPT_find_nat_numbers_l1738_173889

theorem find_nat_numbers (a b : ℕ) (h : 1 / (a - b) = 3 * (1 / (a * b))) : a = 6 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_find_nat_numbers_l1738_173889


namespace NUMINAMATH_GPT_room_width_correct_l1738_173848

noncomputable def length_of_room : ℝ := 5
noncomputable def total_cost_of_paving : ℝ := 21375
noncomputable def cost_per_square_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem room_width_correct :
  (total_cost_of_paving / cost_per_square_meter) = (length_of_room * width_of_room) :=
by
  sorry

end NUMINAMATH_GPT_room_width_correct_l1738_173848


namespace NUMINAMATH_GPT_calculate_angles_and_side_l1738_173854

theorem calculate_angles_and_side (a b B : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 2) (h_B : B = 45) :
  ∃ A C c, (A = 60 ∧ C = 75 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨ (A = 120 ∧ C = 15 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry

end NUMINAMATH_GPT_calculate_angles_and_side_l1738_173854


namespace NUMINAMATH_GPT_satisfy_inequality_l1738_173860

theorem satisfy_inequality (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_GPT_satisfy_inequality_l1738_173860


namespace NUMINAMATH_GPT_increasing_function_range_l1738_173868

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) : 
  1.5 ≤ a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_increasing_function_range_l1738_173868


namespace NUMINAMATH_GPT_B_days_to_complete_work_l1738_173839

theorem B_days_to_complete_work 
  (W : ℝ) -- Define the amount of work
  (A_rate : ℝ := W / 15) -- A can complete the work in 15 days
  (B_days : ℝ) -- B can complete the work in B_days days
  (B_rate : ℝ := W / B_days) -- B's rate of work
  (total_days : ℝ := 12) -- Total days to complete the work
  (A_days_after_B_leaves : ℝ := 10) -- Days A works alone after B leaves
  (work_done_together : ℝ := 2 * (A_rate + B_rate)) -- Work done together in 2 days
  (work_done_by_A : ℝ := 10 * A_rate) -- Work done by A alone in 10 days
  (total_work_done : ℝ := work_done_together + work_done_by_A) -- Total work done
  (h_total_work_done : total_work_done = W) -- Total work equals W
  : B_days = 10 :=
sorry

end NUMINAMATH_GPT_B_days_to_complete_work_l1738_173839


namespace NUMINAMATH_GPT_value_of_x_squared_plus_one_over_x_squared_l1738_173836

noncomputable def x: ℝ := sorry

theorem value_of_x_squared_plus_one_over_x_squared (h : 20 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 23 :=
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_one_over_x_squared_l1738_173836


namespace NUMINAMATH_GPT_sequence_value_238_l1738_173811

theorem sequence_value_238 (a : ℕ → ℚ) :
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → (n % 2 = 0 → a n = a (n - 1) / 2 + 1) ∧ (n % 2 = 1 → a n = 1 / a (n - 1))) ∧
  (∃ n, a n = 30 / 19) → ∃ n, a n = 30 / 19 ∧ n = 238 :=
by
  sorry

end NUMINAMATH_GPT_sequence_value_238_l1738_173811


namespace NUMINAMATH_GPT_trig_identity_l1738_173864

theorem trig_identity :
  (Real.cos (80 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) + 
   Real.sin (80 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1738_173864


namespace NUMINAMATH_GPT_minimum_value_expression_l1738_173881

theorem minimum_value_expression : ∃ x : ℝ, (3 * x^2 - 18 * x + 2023) = 1996 := sorry

end NUMINAMATH_GPT_minimum_value_expression_l1738_173881


namespace NUMINAMATH_GPT_dylan_ice_cubes_l1738_173877

-- Definitions based on conditions
def trays := 2
def spaces_per_tray := 12
def total_tray_ice := trays * spaces_per_tray
def pitcher_multiplier := 2

-- The statement to be proven
theorem dylan_ice_cubes (x : ℕ) : x + pitcher_multiplier * x = total_tray_ice → x = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_dylan_ice_cubes_l1738_173877


namespace NUMINAMATH_GPT_sequence_general_term_l1738_173878

theorem sequence_general_term (n : ℕ) (hn : 0 < n) : 
  ∃ (a_n : ℕ), a_n = 2 * Int.floor (Real.sqrt (n - 1)) + 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1738_173878


namespace NUMINAMATH_GPT_sum_of_integers_l1738_173816

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 300) (h2 : m * (m + 1) * (m + 2) = 300) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 49 := 
by sorry

end NUMINAMATH_GPT_sum_of_integers_l1738_173816


namespace NUMINAMATH_GPT_simplify_trig_expression_l1738_173827

open Real

/-- 
Given that θ is in the interval (π/2, π), simplify the expression 
( sin θ / sqrt (1 - sin^2 θ) ) + ( sqrt (1 - cos^2 θ) / cos θ ) to 0.
-/
theorem simplify_trig_expression (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  (sin θ / sqrt (1 - sin θ ^ 2)) + (sqrt (1 - cos θ ^ 2) / cos θ) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l1738_173827


namespace NUMINAMATH_GPT_solve_arcsin_sin_l1738_173820

theorem solve_arcsin_sin (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.arcsin (Real.sin (2 * x)) = x ↔ x = 0 ∨ x = Real.pi / 3 ∨ x = -Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_arcsin_sin_l1738_173820


namespace NUMINAMATH_GPT_amc_inequality_l1738_173890

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end NUMINAMATH_GPT_amc_inequality_l1738_173890


namespace NUMINAMATH_GPT_smallest_base_to_represent_124_with_three_digits_l1738_173896

theorem smallest_base_to_represent_124_with_three_digits : 
  ∃ (b : ℕ), b^2 ≤ 124 ∧ 124 < b^3 ∧ ∀ c, (c^2 ≤ 124 ∧ 124 < c^3) → (5 ≤ c) :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_to_represent_124_with_three_digits_l1738_173896


namespace NUMINAMATH_GPT_drone_altitude_l1738_173865

theorem drone_altitude (h c d : ℝ) (HC HD CD : ℝ)
  (HCO_eq : h^2 + c^2 = HC^2)
  (HDO_eq : h^2 + d^2 = HD^2)
  (CD_eq : c^2 + d^2 = CD^2) 
  (HC_val : HC = 170)
  (HD_val : HD = 160)
  (CD_val : CD = 200) :
  h = 50 * Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_GPT_drone_altitude_l1738_173865


namespace NUMINAMATH_GPT_number_of_tables_l1738_173847

theorem number_of_tables (last_year_distance : ℕ) (factor : ℕ) 
  (distance_between_table_1_and_3 : ℕ) (number_of_tables : ℕ) :
  (last_year_distance = 300) ∧ 
  (factor = 4) ∧ 
  (distance_between_table_1_and_3 = 400) ∧
  (number_of_tables = ((factor * last_year_distance) / (distance_between_table_1_and_3 / 2)) + 1) 
  → number_of_tables = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_tables_l1738_173847


namespace NUMINAMATH_GPT_jerry_age_l1738_173832

theorem jerry_age (M J : ℕ) (h1 : M = 4 * J + 10) (h2 : M = 30) : J = 5 := by
  sorry

end NUMINAMATH_GPT_jerry_age_l1738_173832


namespace NUMINAMATH_GPT_harry_worked_41_hours_l1738_173800

def james_earnings (x : ℝ) : ℝ :=
  (40 * x) + (7 * 2 * x)

def harry_earnings (x : ℝ) (h : ℝ) : ℝ :=
  (24 * x) + (11 * 1.5 * x) + (2 * h * x)

def harry_hours_worked (h : ℝ) : ℝ :=
  24 + 11 + h

theorem harry_worked_41_hours (x : ℝ) (h : ℝ) 
  (james_worked : james_earnings x = 54 * x)
  (harry_paid_same : harry_earnings x h = james_earnings x) :
  harry_hours_worked h = 41 :=
by
  -- sorry is used to skip the proof steps
  sorry

end NUMINAMATH_GPT_harry_worked_41_hours_l1738_173800


namespace NUMINAMATH_GPT_part1_3kg_part2_5kg_part2_function_part3_compare_l1738_173806

noncomputable def supermarket_A_cost (x : ℝ) : ℝ :=
if x <= 4 then 10 * x
else 6 * x + 16

noncomputable def supermarket_B_cost (x : ℝ) : ℝ :=
8 * x

-- Proof that supermarket_A_cost 3 = 30
theorem part1_3kg : supermarket_A_cost 3 = 30 :=
by sorry

-- Proof that supermarket_A_cost 5 = 46
theorem part2_5kg : supermarket_A_cost 5 = 46 :=
by sorry

-- Proof that the cost function is correct
theorem part2_function (x : ℝ) : 
(0 < x ∧ x <= 4 → supermarket_A_cost x = 10 * x) ∧ 
(x > 4 → supermarket_A_cost x = 6 * x + 16) :=
by sorry

-- Proof that supermarket A is cheaper for 10 kg apples
theorem part3_compare : supermarket_A_cost 10 < supermarket_B_cost 10 :=
by sorry

end NUMINAMATH_GPT_part1_3kg_part2_5kg_part2_function_part3_compare_l1738_173806


namespace NUMINAMATH_GPT_unique_zero_of_quadratic_l1738_173843

theorem unique_zero_of_quadratic {m : ℝ} (h : ∃ x : ℝ, x^2 + 2*x + m = 0 ∧ (∀ y : ℝ, y^2 + 2*y + m = 0 → y = x)) : m = 1 :=
sorry

end NUMINAMATH_GPT_unique_zero_of_quadratic_l1738_173843


namespace NUMINAMATH_GPT_power_quotient_example_l1738_173846

theorem power_quotient_example (a : ℕ) (m n : ℕ) (h : 23^11 / 23^8 = 23^(11 - 8)) : 23^3 = 12167 := by
  sorry

end NUMINAMATH_GPT_power_quotient_example_l1738_173846


namespace NUMINAMATH_GPT_nina_total_money_l1738_173823

def original_cost_widget (C : ℝ) : ℝ := C
def num_widgets_nina_can_buy_original (C : ℝ) : ℝ := 6
def num_widgets_nina_can_buy_reduced (C : ℝ) : ℝ := 8
def cost_reduction : ℝ := 1.5

theorem nina_total_money (C : ℝ) (hc : 6 * C = 8 * (C - cost_reduction)) : 
  6 * C = 36 :=
by
  sorry

end NUMINAMATH_GPT_nina_total_money_l1738_173823


namespace NUMINAMATH_GPT_sum_of_adjacent_cells_multiple_of_4_l1738_173844

theorem sum_of_adjacent_cells_multiple_of_4 :
  ∃ (i j : ℕ) (a b : ℕ) (H₁ : i < 22) (H₂ : j < 22),
    let grid (i j : ℕ) : ℕ := -- define the function for grid indexing
      ((i * 22) + j + 1 : ℕ)
    ∃ (i1 j1 : ℕ) (H₁₁ : i1 = i ∨ i1 = i + 1 ∨ i1 = i - 1)
                   (H₁₂ : j1 = j ∨ j1 = j + 1 ∨ j1 = j - 1),
      a = grid i j ∧ b = grid i1 j1 ∧ (a + b) % 4 = 0 := sorry

end NUMINAMATH_GPT_sum_of_adjacent_cells_multiple_of_4_l1738_173844


namespace NUMINAMATH_GPT_percentage_of_rotten_bananas_l1738_173818

theorem percentage_of_rotten_bananas :
  ∀ (total_oranges total_bananas : ℕ) 
    (percent_rotten_oranges : ℝ) 
    (percent_good_fruits : ℝ), 
  total_oranges = 600 → total_bananas = 400 → 
  percent_rotten_oranges = 0.15 → percent_good_fruits = 0.89 → 
  (100 - (((percent_good_fruits * (total_oranges + total_bananas)) - 
  ((1 - percent_rotten_oranges) * total_oranges)) / total_bananas) * 100) = 5 := 
by
  intros total_oranges total_bananas percent_rotten_oranges percent_good_fruits 
  intro ho hb hro hpf 
  sorry

end NUMINAMATH_GPT_percentage_of_rotten_bananas_l1738_173818


namespace NUMINAMATH_GPT_m_gt_p_l1738_173851

theorem m_gt_p (p m n : ℕ) (prime_p : Nat.Prime p) (pos_m : 0 < m) (pos_n : 0 < n) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end NUMINAMATH_GPT_m_gt_p_l1738_173851


namespace NUMINAMATH_GPT_terminal_side_quadrant_l1738_173833

theorem terminal_side_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) :
  ∃ k : ℤ, (k % 2 = 0 ∧ (k * Real.pi + Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + Real.pi / 2)) ∨
           (k % 2 = 1 ∧ (k * Real.pi + 3 * Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + 5 * Real.pi / 4)) := sorry

end NUMINAMATH_GPT_terminal_side_quadrant_l1738_173833
