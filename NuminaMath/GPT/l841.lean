import Mathlib

namespace NUMINAMATH_GPT_walking_speed_is_correct_l841_84109

-- Define the conditions
def time_in_minutes : ℝ := 10
def distance_in_meters : ℝ := 1666.6666666666665
def speed_in_km_per_hr : ℝ := 2.777777777777775

-- Define the theorem to prove
theorem walking_speed_is_correct :
  (distance_in_meters / time_in_minutes) * 60 / 1000 = speed_in_km_per_hr :=
sorry

end NUMINAMATH_GPT_walking_speed_is_correct_l841_84109


namespace NUMINAMATH_GPT_interval_contains_integer_l841_84103

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, (3 * a < n) ∧ (n < 5 * a - 2)) ↔ (1.2 < a ∧ a < 4 / 3) ∨ (7 / 5 < a) :=
by sorry

end NUMINAMATH_GPT_interval_contains_integer_l841_84103


namespace NUMINAMATH_GPT_sum_gcd_lcm_l841_84184

theorem sum_gcd_lcm (a₁ a₂ : ℕ) (h₁ : a₁ = 36) (h₂ : a₂ = 495) :
  Nat.gcd a₁ a₂ + Nat.lcm a₁ a₂ = 1989 :=
by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_l841_84184


namespace NUMINAMATH_GPT_inequality_true_l841_84175

theorem inequality_true (a b : ℝ) (h : a > b) (x : ℝ) : 
  (a > b) → (x ≥ 0) → (a / ((2^x) + 1) > b / ((2^x) + 1)) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_true_l841_84175


namespace NUMINAMATH_GPT_mul_powers_same_base_l841_84166

theorem mul_powers_same_base (a : ℝ) : a^3 * a^4 = a^7 := 
by 
  sorry

end NUMINAMATH_GPT_mul_powers_same_base_l841_84166


namespace NUMINAMATH_GPT_domain_of_f_2x_minus_1_l841_84131

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  ∀ x, (1 / 2) ≤ x ∧ x ≤ (3 / 2) → ∃ y, f y = (2 * x - 1) :=
by
  intros h x hx
  sorry

end NUMINAMATH_GPT_domain_of_f_2x_minus_1_l841_84131


namespace NUMINAMATH_GPT_fraction_simplify_l841_84138

theorem fraction_simplify :
  (3 + 9 - 27 + 81 - 243 + 729) / (9 + 27 - 81 + 243 - 729 + 2187) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplify_l841_84138


namespace NUMINAMATH_GPT_wider_can_radius_l841_84151

theorem wider_can_radius (h : ℝ) : 
  (∃ r : ℝ, ∀ V : ℝ, V = π * 8^2 * 2 * h → V = π * r^2 * h → r = 8 * Real.sqrt 2) :=
by 
  sorry

end NUMINAMATH_GPT_wider_can_radius_l841_84151


namespace NUMINAMATH_GPT_laptop_price_reduction_l841_84162

-- Conditions definitions
def initial_price (P : ℝ) : ℝ := P
def seasonal_sale (P : ℝ) : ℝ := 0.7 * P
def special_promotion (seasonal_price : ℝ) : ℝ := 0.8 * seasonal_price
def clearance_event (promotion_price : ℝ) : ℝ := 0.9 * promotion_price

-- Proof statement
theorem laptop_price_reduction (P : ℝ) (h1 : seasonal_sale P = 0.7 * P) 
    (h2 : special_promotion (seasonal_sale P) = 0.8 * (seasonal_sale P)) 
    (h3 : clearance_event (special_promotion (seasonal_sale P)) = 0.9 * (special_promotion (seasonal_sale P))) : 
    (initial_price P - clearance_event (special_promotion (seasonal_sale P))) / (initial_price P) = 0.496 := 
by 
  sorry

end NUMINAMATH_GPT_laptop_price_reduction_l841_84162


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l841_84150

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l841_84150


namespace NUMINAMATH_GPT_find_amplitude_l841_84191

noncomputable def amplitude (a b c d x : ℝ) := a * Real.sin (b * x + c) + d

theorem find_amplitude (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_range : ∀ x, -1 ≤ amplitude a b c d x ∧ amplitude a b c d x ≤ 7) :
  a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_amplitude_l841_84191


namespace NUMINAMATH_GPT_greg_rolls_more_ones_than_fives_l841_84130

def probability_more_ones_than_fives (n : ℕ) : ℚ :=
  if n = 6 then 695 / 1944 else 0

theorem greg_rolls_more_ones_than_fives :
  probability_more_ones_than_fives 6 = 695 / 1944 :=
by sorry

end NUMINAMATH_GPT_greg_rolls_more_ones_than_fives_l841_84130


namespace NUMINAMATH_GPT_roots_of_cubic_8th_power_sum_l841_84172

theorem roots_of_cubic_8th_power_sum :
  ∀ a b c : ℂ, 
  (a + b + c = 0) → 
  (a * b + b * c + c * a = -1) → 
  (a * b * c = -1) → 
  (a^8 + b^8 + c^8 = 10) := 
by
  sorry

end NUMINAMATH_GPT_roots_of_cubic_8th_power_sum_l841_84172


namespace NUMINAMATH_GPT_some_number_is_l841_84126

theorem some_number_is (x some_number : ℤ) (h1 : x = 4) (h2 : 5 * x + 3 = 10 * x - some_number) : some_number = 17 := by
  sorry

end NUMINAMATH_GPT_some_number_is_l841_84126


namespace NUMINAMATH_GPT_minimum_value_expression_l841_84178

theorem minimum_value_expression {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x / y + y) * (y / x + x) ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l841_84178


namespace NUMINAMATH_GPT_problem_solution_l841_84147

variables (p q : Prop)

theorem problem_solution (h1 : ¬ (p ∧ q)) (h2 : p ∨ q) : ¬ p ∨ ¬ q := by
  sorry

end NUMINAMATH_GPT_problem_solution_l841_84147


namespace NUMINAMATH_GPT_perimeter_of_figure_l841_84188

theorem perimeter_of_figure (a b c d : ℕ) (p : ℕ) (h1 : a = 6) (h2 : b = 3) (h3 : c = 2) (h4 : d = 4) (h5 : p = a * b + c * d) : p = 26 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_l841_84188


namespace NUMINAMATH_GPT_difference_of_numbers_l841_84196

theorem difference_of_numbers (a : ℕ) (h : a + (10 * a + 5) = 30000) : (10 * a + 5) - a = 24548 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l841_84196


namespace NUMINAMATH_GPT_reflection_sum_coordinates_l841_84192

theorem reflection_sum_coordinates :
  ∀ (C D : ℝ × ℝ), 
  C = (5, -3) →
  D = (5, -C.2) →
  (C.1 + C.2 + D.1 + D.2 = 10) :=
by
  intros C D hC hD
  rw [hC, hD]
  simp
  sorry

end NUMINAMATH_GPT_reflection_sum_coordinates_l841_84192


namespace NUMINAMATH_GPT_trig_identity_proof_l841_84136

theorem trig_identity_proof
  (h1: Float.sin 50 = Float.cos 40)
  (h2: Float.tan 45 = 1)
  (h3: Float.tan 10 = Float.sin 10 / Float.cos 10)
  (h4: Float.sin 80 = Float.cos 10) :
  Float.sin 50 * (Float.tan 45 + Float.sqrt 3 * Float.tan 10) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l841_84136


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l841_84144

theorem cost_price_of_computer_table (S : ℝ) (C : ℝ) (h1 : 1.80 * C = S) (h2 : S = 3500) : C = 1944.44 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l841_84144


namespace NUMINAMATH_GPT_no_real_solution_range_of_a_l841_84190

theorem no_real_solution_range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ≤ 3 :=
by
  sorry  -- Proof skipped

end NUMINAMATH_GPT_no_real_solution_range_of_a_l841_84190


namespace NUMINAMATH_GPT_find_side_b_l841_84180

variables {A B C a b c x : ℝ}

theorem find_side_b 
  (cos_A : ℝ) (cos_C : ℝ) (a : ℝ) (hcosA : cos_A = 4/5) 
  (hcosC : cos_C = 5/13) (ha : a = 1) : 
  b = 21/13 :=
by
  sorry

end NUMINAMATH_GPT_find_side_b_l841_84180


namespace NUMINAMATH_GPT_doughnut_cost_l841_84133

theorem doughnut_cost:
  ∃ (D C : ℝ), 
    3 * D + 4 * C = 4.91 ∧ 
    5 * D + 6 * C = 7.59 ∧ 
    D = 0.45 :=
by
  sorry

end NUMINAMATH_GPT_doughnut_cost_l841_84133


namespace NUMINAMATH_GPT_simplified_expression_l841_84183

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 1

theorem simplified_expression :
  (f (g (f 3))) / (g (f (g 3))) = 79 / 37 :=
by  sorry

end NUMINAMATH_GPT_simplified_expression_l841_84183


namespace NUMINAMATH_GPT_ellipse_eq_l841_84140

theorem ellipse_eq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 - b^2 = 4)
  (h4 : ∃ (line_eq : ℝ → ℝ), ∀ (x : ℝ), line_eq x = 3 * x + 7)
  (h5 : ∃ (mid_y : ℝ), mid_y = 1 ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    ((y1 = 3 * x1 + 7) ∧ (y2 = 3 * x2 + 7)) ∧ 
    (y1 + y2) / 2 = mid_y): 
  (∀ x y : ℝ, (y^2 / (a^2 - 4) + x^2 / b^2 = 1) ↔ 
  (x^2 / 8 + y^2 / 12 = 1)) :=
by { sorry }

end NUMINAMATH_GPT_ellipse_eq_l841_84140


namespace NUMINAMATH_GPT_youngest_sibling_age_l841_84114

theorem youngest_sibling_age
  (Y : ℕ)
  (h1 : Y + (Y + 3) + (Y + 6) + (Y + 7) = 120) :
  Y = 26 :=
by
  -- proof steps would be here 
  sorry

end NUMINAMATH_GPT_youngest_sibling_age_l841_84114


namespace NUMINAMATH_GPT_total_turtles_l841_84170

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end NUMINAMATH_GPT_total_turtles_l841_84170


namespace NUMINAMATH_GPT_sum_even_integers_eq_930_l841_84157

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end NUMINAMATH_GPT_sum_even_integers_eq_930_l841_84157


namespace NUMINAMATH_GPT_total_area_of_room_l841_84102

theorem total_area_of_room : 
  let length_rect := 8 
  let width_rect := 6 
  let base_triangle := 6 
  let height_triangle := 3 
  let area_rect := length_rect * width_rect 
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle 
  let total_area := area_rect + area_triangle 
  total_area = 57 := 
by 
  sorry

end NUMINAMATH_GPT_total_area_of_room_l841_84102


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l841_84117

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l841_84117


namespace NUMINAMATH_GPT_rowing_speed_downstream_l841_84198

theorem rowing_speed_downstream (V_u V_s V_d : ℝ) (h1 : V_u = 10) (h2 : V_s = 15)
  (h3 : V_s = (V_u + V_d) / 2) : V_d = 20 := by
  sorry

end NUMINAMATH_GPT_rowing_speed_downstream_l841_84198


namespace NUMINAMATH_GPT_solve_for_y_l841_84176

theorem solve_for_y :
  ∀ (y : ℝ), (9 * y^2 + 49 * y^2 + 21/2 * y^2 = 1300) → y = 4.34 := 
by sorry

end NUMINAMATH_GPT_solve_for_y_l841_84176


namespace NUMINAMATH_GPT_integer_solutions_l841_84153

theorem integer_solutions :
  ∀ (m n : ℤ), (m^3 - n^3 = 2 * m * n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2)) :=
by
  intros m n
  sorry

end NUMINAMATH_GPT_integer_solutions_l841_84153


namespace NUMINAMATH_GPT_f_at_47_l841_84129

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_equation : ∀ x : ℝ, f (x - 1) + f (x + 1) = 0
axiom f_interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem f_at_47 : f 47 = -1 := by
  sorry

end NUMINAMATH_GPT_f_at_47_l841_84129


namespace NUMINAMATH_GPT_smallest_b_theorem_l841_84121

open Real

noncomputable def smallest_b (a b c: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) : Prop :=
  b = 5

theorem smallest_b_theorem (a b c: ℝ) (r: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) :
  smallest_b a b c h1 h2 h3 h4 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_b_theorem_l841_84121


namespace NUMINAMATH_GPT_radius_of_circle_l841_84108

theorem radius_of_circle (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → (∃ r, r = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_circle_l841_84108


namespace NUMINAMATH_GPT_hyperbola_focal_distance_solution_l841_84164

-- Definitions corresponding to the problem conditions
def hyperbola_equation (x y m : ℝ) :=
  x^2 / m - y^2 / 6 = 1

def focal_distance (c : ℝ) := 2 * c

-- Theorem statement to prove m = 3 based on given conditions
theorem hyperbola_focal_distance_solution (m : ℝ) (h_eq : ∀ x y : ℝ, hyperbola_equation x y m) (h_focal : focal_distance 3 = 6) :
  m = 3 :=
by {
  -- sorry is used here as a placeholder for the actual proof steps
  sorry
}

end NUMINAMATH_GPT_hyperbola_focal_distance_solution_l841_84164


namespace NUMINAMATH_GPT_grandparents_gift_l841_84194

theorem grandparents_gift (june_stickers bonnie_stickers total_stickers : ℕ) (x : ℕ)
  (h₁ : june_stickers = 76)
  (h₂ : bonnie_stickers = 63)
  (h₃ : total_stickers = 189) :
  june_stickers + bonnie_stickers + 2 * x = total_stickers → x = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_grandparents_gift_l841_84194


namespace NUMINAMATH_GPT_find_a22_l841_84139

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end NUMINAMATH_GPT_find_a22_l841_84139


namespace NUMINAMATH_GPT_Eli_saves_more_with_discount_A_l841_84156

-- Define the prices and discounts
def price_book : ℝ := 25
def discount_A (price : ℝ) : ℝ := price * 0.4
def discount_B : ℝ := 5

-- Define the cost calculations:
def cost_with_discount_A (price : ℝ) : ℝ := price + (price - discount_A price)
def cost_with_discount_B (price : ℝ) : ℝ := price + (price - discount_B)

-- Define the savings calculation:
def savings (cost_B : ℝ) (cost_A : ℝ) : ℝ := cost_B - cost_A

-- The main statement to prove:
theorem Eli_saves_more_with_discount_A :
  savings (cost_with_discount_B price_book) (cost_with_discount_A price_book) = 5 :=
by
  sorry

end NUMINAMATH_GPT_Eli_saves_more_with_discount_A_l841_84156


namespace NUMINAMATH_GPT_tank_capacity_l841_84122

theorem tank_capacity (T : ℕ) (h1 : T > 0) 
    (h2 : (2 * T) / 5 + 15 + 20 = T - 25) : 
    T = 100 := 
  by 
    sorry

end NUMINAMATH_GPT_tank_capacity_l841_84122


namespace NUMINAMATH_GPT_each_charity_gets_45_dollars_l841_84195

def dozens : ℤ := 6
def cookies_per_dozen : ℤ := 12
def total_cookies : ℤ := dozens * cookies_per_dozen
def selling_price_per_cookie : ℚ := 1.5
def cost_per_cookie : ℚ := 0.25
def profit_per_cookie : ℚ := selling_price_per_cookie - cost_per_cookie
def total_profit : ℚ := profit_per_cookie * total_cookies
def charities : ℤ := 2
def amount_per_charity : ℚ := total_profit / charities

theorem each_charity_gets_45_dollars : amount_per_charity = 45 := 
by
  sorry

end NUMINAMATH_GPT_each_charity_gets_45_dollars_l841_84195


namespace NUMINAMATH_GPT_uncolored_area_of_rectangle_l841_84128

theorem uncolored_area_of_rectangle :
  let width := 30
  let length := 50
  let radius := width / 4
  let rectangle_area := width * length
  let circle_area := π * (radius ^ 2)
  let total_circles_area := 4 * circle_area
  rectangle_area - total_circles_area = 1500 - 225 * π := by
  sorry

end NUMINAMATH_GPT_uncolored_area_of_rectangle_l841_84128


namespace NUMINAMATH_GPT_max_xy_l841_84163

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  xy ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_l841_84163


namespace NUMINAMATH_GPT_inequality_solution_set_l841_84199

theorem inequality_solution_set (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) ≤ 1) ↔ (x < 2 ∨ 3 ≤ x) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l841_84199


namespace NUMINAMATH_GPT_pos_rel_lines_l841_84179

-- Definition of the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := 2 * x - y + k = 0
def line2 (x y : ℝ) : Prop := 4 * x - 2 * y + 1 = 0

-- Theorem stating the positional relationship between the two lines
theorem pos_rel_lines (k : ℝ) : 
  (∀ x y : ℝ, line1 k x y → line2 x y → 2 * k - 1 = 0) → 
  (∀ x y : ℝ, line1 k x y → ¬ line2 x y → 2 * k - 1 ≠ 0) → 
  (k = 1/2 ∨ k ≠ 1/2) :=
by sorry

end NUMINAMATH_GPT_pos_rel_lines_l841_84179


namespace NUMINAMATH_GPT_two_abs_inequality_l841_84107

theorem two_abs_inequality (x y : ℝ) :
  2 * abs (x + y) ≤ abs x + abs y ↔ 
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -x / 3) ∨ 
  (x < 0 ∧ -x / 3 ≤ y ∧ y ≤ -3 * x) :=
by
  sorry

end NUMINAMATH_GPT_two_abs_inequality_l841_84107


namespace NUMINAMATH_GPT_Mark_charged_more_l841_84168

theorem Mark_charged_more (K P M : ℕ) 
  (h1 : P = 2 * K) 
  (h2 : P = M / 3)
  (h3 : K + P + M = 153) : M - K = 85 :=
by
  -- proof to be filled in later
  sorry

end NUMINAMATH_GPT_Mark_charged_more_l841_84168


namespace NUMINAMATH_GPT_bowler_overs_l841_84149

theorem bowler_overs (x : ℕ) (h1 : ∀ y, y ≤ 3 * x) 
                     (h2 : y = 10) : x = 4 := by
  sorry

end NUMINAMATH_GPT_bowler_overs_l841_84149


namespace NUMINAMATH_GPT_raft_min_capacity_l841_84158

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end NUMINAMATH_GPT_raft_min_capacity_l841_84158


namespace NUMINAMATH_GPT_product_mod_7_l841_84174

theorem product_mod_7 (a b c : ℕ) (ha : a % 7 = 3) (hb : b % 7 = 4) (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 4 :=
sorry

end NUMINAMATH_GPT_product_mod_7_l841_84174


namespace NUMINAMATH_GPT_park_area_calculation_l841_84125

noncomputable def width_of_park := Real.sqrt (9000000 / 65)
noncomputable def length_of_park := 8 * width_of_park

def actual_area_of_park (w l : ℝ) : ℝ := w * l

theorem park_area_calculation :
  let w := width_of_park
  let l := length_of_park
  actual_area_of_park w l = 1107746.48 :=
by
  -- Calculations from solution are provided here directly as conditions and definitions
  sorry

end NUMINAMATH_GPT_park_area_calculation_l841_84125


namespace NUMINAMATH_GPT_opposite_numbers_abs_l841_84113

theorem opposite_numbers_abs (a b : ℤ) (h : a + b = 0) : |a - 2014 + b| = 2014 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_opposite_numbers_abs_l841_84113


namespace NUMINAMATH_GPT_felix_brother_lifting_capacity_is_600_l841_84115

-- Define the conditions
def felix_lifting_capacity (felix_weight : ℝ) : ℝ := 1.5 * felix_weight
def felix_brother_weight (felix_weight : ℝ) : ℝ := 2 * felix_weight
def felix_brother_lifting_capacity (brother_weight : ℝ) : ℝ := 3 * brother_weight
def felix_actual_lifting_capacity : ℝ := 150

-- Define the proof problem
theorem felix_brother_lifting_capacity_is_600 :
  ∃ felix_weight : ℝ,
    felix_lifting_capacity felix_weight = felix_actual_lifting_capacity ∧
    felix_brother_lifting_capacity (felix_brother_weight felix_weight) = 600 :=
by
  sorry

end NUMINAMATH_GPT_felix_brother_lifting_capacity_is_600_l841_84115


namespace NUMINAMATH_GPT_dot_product_of_vectors_l841_84193

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1) - vector_a

theorem dot_product_of_vectors :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = -4 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_of_vectors_l841_84193


namespace NUMINAMATH_GPT_area_of_T_l841_84123

open Complex Real

noncomputable def omega := -1 / 2 + (1 / 2) * Complex.I * Real.sqrt 3
noncomputable def omega2 := -1 / 2 - (1 / 2) * Complex.I * Real.sqrt 3

def inT (z : ℂ) (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 ∧
  0 ≤ b ∧ b ≤ 1 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  z = a + b * omega + c * omega2

theorem area_of_T : ∃ A : ℝ, A = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_area_of_T_l841_84123


namespace NUMINAMATH_GPT_jack_and_jill_meet_distance_l841_84154

theorem jack_and_jill_meet_distance :
  ∃ t : ℝ, t = 15 / 60 ∧ 14 * t ≤ 4 ∧ 15 * (t - 15 / 60) ≤ 4 ∧
  ( 14 * t - 4 + 18 * (t - 2 / 7) = 15 * (t - 15 / 60) ∨ 15 * (t - 15 / 60) = 4 - 18 * (t - 2 / 7) ) ∧
  4 - 15 * (t - 15 / 60) = 851 / 154 :=
sorry

end NUMINAMATH_GPT_jack_and_jill_meet_distance_l841_84154


namespace NUMINAMATH_GPT_greatest_possible_x_lcm_l841_84159

theorem greatest_possible_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105): x = 105 := 
sorry

end NUMINAMATH_GPT_greatest_possible_x_lcm_l841_84159


namespace NUMINAMATH_GPT_parallel_line_through_intersection_perpendicular_line_through_intersection_l841_84189

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and parallel to the line 2x - y - 1 = 0 
is 2x - y + 1 = 0 --/
theorem parallel_line_through_intersection :
  ∃ (c : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (2 * x - y + c = 0) ∧ c = 1 :=
by
  sorry

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and perpendicular to the line 2x - y - 1 = 0
is x + 2y - 7 = 0 --/
theorem perpendicular_line_through_intersection :
  ∃ (d : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (x + 2 * y + d = 0) ∧ d = -7 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_through_intersection_perpendicular_line_through_intersection_l841_84189


namespace NUMINAMATH_GPT_A_investment_amount_l841_84161

theorem A_investment_amount
  (B_investment : ℝ) (C_investment : ℝ) 
  (total_profit : ℝ) (A_profit_share : ℝ)
  (h1 : B_investment = 4200)
  (h2 : C_investment = 10500)
  (h3 : total_profit = 14200)
  (h4 : A_profit_share = 4260) :
  ∃ (A_investment : ℝ), 
    A_profit_share / total_profit = A_investment / (A_investment + B_investment + C_investment) ∧ 
    A_investment = 6600 :=
by {
  sorry  -- Proof not required per instructions
}

end NUMINAMATH_GPT_A_investment_amount_l841_84161


namespace NUMINAMATH_GPT_no_real_roots_of_ffx_or_ggx_l841_84181

noncomputable def is_unitary_quadratic_trinomial (p : ℝ → ℝ) : Prop :=
∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c

theorem no_real_roots_of_ffx_or_ggx 
    (f g : ℝ → ℝ) 
    (hf : is_unitary_quadratic_trinomial f) 
    (hg : is_unitary_quadratic_trinomial g)
    (hf_ng : ∀ x : ℝ, f (g x) ≠ 0)
    (hg_nf : ∀ x : ℝ, g (f x) ≠ 0) :
    (∀ x : ℝ, f (f x) ≠ 0) ∨ (∀ x : ℝ, g (g x) ≠ 0) :=
sorry

end NUMINAMATH_GPT_no_real_roots_of_ffx_or_ggx_l841_84181


namespace NUMINAMATH_GPT_cost_of_steel_ingot_l841_84187

theorem cost_of_steel_ingot :
  ∃ P : ℝ, 
    (∃ initial_weight : ℝ, initial_weight = 60) ∧
    (∃ weight_increase_percentage : ℝ, weight_increase_percentage = 0.6) ∧
    (∃ ingot_weight : ℝ, ingot_weight = 2) ∧
    (weight_needed = initial_weight * weight_increase_percentage) ∧
    (number_of_ingots = weight_needed / ingot_weight) ∧
    (number_of_ingots > 10) ∧
    (discount_percentage = 0.2) ∧
    (total_cost = 72) ∧
    (discounted_price_per_ingot = P * (1 - discount_percentage)) ∧
    (total_cost = discounted_price_per_ingot * number_of_ingots) ∧
    P = 5 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_steel_ingot_l841_84187


namespace NUMINAMATH_GPT_additional_grassy_area_l841_84119

theorem additional_grassy_area (r1 r2 : ℝ) (r1_pos : r1 = 10) (r2_pos : r2 = 35) : 
  let A1 := π * r1^2
  let A2 := π * r2^2
  (A2 - A1) = 1125 * π :=
by 
  sorry

end NUMINAMATH_GPT_additional_grassy_area_l841_84119


namespace NUMINAMATH_GPT_volume_tetrahedron_375sqrt2_l841_84143

noncomputable def tetrahedronVolume (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  let h_BCD := (2 * area_BCD) / BC
  let h_D_ABD := h_BCD * Real.sin angle_ABC_BCD
  (1 / 3) * area_ABC * h_D_ABD

theorem volume_tetrahedron_375sqrt2 :
  tetrahedronVolume 150 90 12 (Real.pi / 4) = 375 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_volume_tetrahedron_375sqrt2_l841_84143


namespace NUMINAMATH_GPT_factorize_expression_l841_84152

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_factorize_expression_l841_84152


namespace NUMINAMATH_GPT_value_of_k_l841_84105

noncomputable def find_k (x1 x2 : ℝ) (k : ℝ) : Prop :=
  (2 * x1^2 + k * x1 - 2 = 0) ∧ (2 * x2^2 + k * x2 - 2 = 0) ∧ ((x1 - 2) * (x2 - 2) = 10)

theorem value_of_k (x1 x2 : ℝ) (k : ℝ) (h : find_k x1 x2 k) : k = 7 :=
sorry

end NUMINAMATH_GPT_value_of_k_l841_84105


namespace NUMINAMATH_GPT_henri_total_miles_l841_84141

noncomputable def g_total : ℕ := 315 * 3
noncomputable def h_total : ℕ := g_total + 305

theorem henri_total_miles : h_total = 1250 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_henri_total_miles_l841_84141


namespace NUMINAMATH_GPT_exists_cube_number_divisible_by_six_in_range_l841_84118

theorem exists_cube_number_divisible_by_six_in_range :
  ∃ (y : ℕ), y > 50 ∧ y < 350 ∧ (∃ (n : ℕ), y = n^3) ∧ y % 6 = 0 :=
by 
  use 216
  sorry

end NUMINAMATH_GPT_exists_cube_number_divisible_by_six_in_range_l841_84118


namespace NUMINAMATH_GPT_determine_a_l841_84111

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 - 2 * a * x + 1 

theorem determine_a (a : ℝ) (h : ¬ (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0)) : a > 1 :=
sorry

end NUMINAMATH_GPT_determine_a_l841_84111


namespace NUMINAMATH_GPT_count_invitations_l841_84104

theorem count_invitations (teachers : Finset ℕ) (A B : ℕ) (hA : A ∈ teachers) (hB : B ∈ teachers) (h_size : teachers.card = 10):
  ∃ (ways : ℕ), ways = 140 ∧ ∀ (S : Finset ℕ), S.card = 6 → ((A ∈ S ∧ B ∉ S) ∨ (A ∉ S ∧ B ∈ S) ∨ (A ∉ S ∧ B ∉ S)) ↔ ways = 140 := 
sorry

end NUMINAMATH_GPT_count_invitations_l841_84104


namespace NUMINAMATH_GPT_merchant_profit_after_discount_l841_84173

/-- A merchant marks his goods up by 40% and then offers a discount of 20% 
on the marked price. Prove that the merchant makes a profit of 12%. -/
theorem merchant_profit_after_discount :
  ∀ (CP MP SP : ℝ),
    CP > 0 →
    MP = CP * 1.4 →
    SP = MP * 0.8 →
    ((SP - CP) / CP) * 100 = 12 :=
by
  intros CP MP SP hCP hMP hSP
  sorry

end NUMINAMATH_GPT_merchant_profit_after_discount_l841_84173


namespace NUMINAMATH_GPT_acute_triangle_inequality_l841_84135

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C) ≤
    Real.pi * (1 / A + 1 / B + 1 / C) :=
sorry

end NUMINAMATH_GPT_acute_triangle_inequality_l841_84135


namespace NUMINAMATH_GPT_sara_has_green_marbles_l841_84169

-- Definition of the total number of green marbles and Tom's green marbles
def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4

-- Definition of Sara's green marbles
def sara_green_marbles : ℕ := total_green_marbles - tom_green_marbles

-- The proof statement
theorem sara_has_green_marbles : sara_green_marbles = 3 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_sara_has_green_marbles_l841_84169


namespace NUMINAMATH_GPT_john_money_left_l841_84134

-- Given definitions
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := q
def large_pizza_cost (q : ℝ) := 4 * q
def initial_amount := 50

-- Problem statement
theorem john_money_left (q : ℝ) : initial_amount - (4 * drink_cost q + 2 * small_pizza_cost q + large_pizza_cost q) = 50 - 10 * q :=
by
  sorry

end NUMINAMATH_GPT_john_money_left_l841_84134


namespace NUMINAMATH_GPT_negation_of_proposition_l841_84160

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x ≥ 0) ↔ (∃ x > 0, x^2 + x < 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l841_84160


namespace NUMINAMATH_GPT_symmetric_points_l841_84142

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end NUMINAMATH_GPT_symmetric_points_l841_84142


namespace NUMINAMATH_GPT_Lizzie_group_number_l841_84124

theorem Lizzie_group_number (x : ℕ) (h1 : x + (x + 17) = 91) : x + 17 = 54 :=
by
  sorry

end NUMINAMATH_GPT_Lizzie_group_number_l841_84124


namespace NUMINAMATH_GPT_planned_daily_catch_l841_84112

theorem planned_daily_catch (x y : ℝ) 
  (h1 : x * y = 1800)
  (h2 : (x / 3) * (y - 20) + ((2 * x / 3) - 1) * (y + 20) = 1800) :
  y = 100 :=
by
  sorry

end NUMINAMATH_GPT_planned_daily_catch_l841_84112


namespace NUMINAMATH_GPT_triangle_area_l841_84148

theorem triangle_area (base height : ℝ) (h_base : base = 4.5) (h_height : height = 6) :
  (base * height) / 2 = 13.5 := 
by
  rw [h_base, h_height]
  norm_num

-- sorry
-- The later use of sorry statement is commented out because the proof itself has been provided in by block.

end NUMINAMATH_GPT_triangle_area_l841_84148


namespace NUMINAMATH_GPT_problem_solution_l841_84185

open Set

theorem problem_solution (x : ℝ) :
  (x ∈ {y : ℝ | (2 / (y + 2) + 4 / (y + 8) ≥ 1)} ↔ x ∈ Ioo (-8 : ℝ) (-2 : ℝ)) :=
sorry

end NUMINAMATH_GPT_problem_solution_l841_84185


namespace NUMINAMATH_GPT_sides_of_regular_polygon_l841_84182

theorem sides_of_regular_polygon {n : ℕ} (h₁ : n ≥ 3)
  (h₂ : (n * (n - 3)) / 2 + 6 = 2 * n) : n = 4 :=
sorry

end NUMINAMATH_GPT_sides_of_regular_polygon_l841_84182


namespace NUMINAMATH_GPT_buoy_min_force_l841_84116

-- Define the problem in Lean
variables (M : ℝ) (ax : ℝ) (T_star : ℝ) (a : ℝ) (F_current : ℝ)
-- Conditions
variables (h_horizontal_component : T_star * Real.sin a = F_current)
          (h_zero_net_force : M * ax = 0)

theorem buoy_min_force (h_horizontal_component : T_star * Real.sin a = F_current) : 
  F_current = 400 := 
sorry

end NUMINAMATH_GPT_buoy_min_force_l841_84116


namespace NUMINAMATH_GPT_ratio_of_cookies_l841_84186

-- Definitions based on the conditions
def initial_cookies : ℕ := 19
def cookies_to_friend : ℕ := 5
def cookies_left : ℕ := 5
def cookies_eaten : ℕ := 2

-- Calculating the number of cookies left after giving cookies to the friend
def cookies_after_giving_to_friend := initial_cookies - cookies_to_friend

-- Maria gave to her family the remaining cookies minus the cookies she has left and she has eaten.
def cookies_given_to_family := cookies_after_giving_to_friend - cookies_eaten - cookies_left

-- The ratio to be proven 1:2, which is mathematically 1/2
theorem ratio_of_cookies : (cookies_given_to_family : ℚ) / (cookies_after_giving_to_friend : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_cookies_l841_84186


namespace NUMINAMATH_GPT_quadratic_discriminant_correct_l841_84145

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant_correct :
  discriminant 5 (5 + 1/2) (-1/2) = 161 / 4 :=
by
  -- let's prove the equality directly
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_correct_l841_84145


namespace NUMINAMATH_GPT_bananas_proof_l841_84120

noncomputable def number_of_bananas (total_oranges : ℕ) (total_fruits_percent_good : ℝ) 
  (percent_rotten_oranges : ℝ) (percent_rotten_bananas : ℝ) : ℕ := 448

theorem bananas_proof :
  let total_oranges := 600
  let percent_rotten_oranges := 0.15
  let percent_rotten_bananas := 0.08
  let total_fruits_percent_good := 0.878
  
  number_of_bananas total_oranges total_fruits_percent_good percent_rotten_oranges percent_rotten_bananas = 448 :=
by
  sorry

end NUMINAMATH_GPT_bananas_proof_l841_84120


namespace NUMINAMATH_GPT_radius_of_circumscribed_sphere_l841_84197

noncomputable def circumscribedSphereRadius (a : ℝ) (α := 60 * Real.pi / 180) : ℝ :=
  5 * a / (4 * Real.sqrt 3)

theorem radius_of_circumscribed_sphere (a : ℝ) :
  circumscribedSphereRadius a = 5 * a / (4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_radius_of_circumscribed_sphere_l841_84197


namespace NUMINAMATH_GPT_divisible_by_pow3_l841_84106

-- Define the digit sequence function
def num_with_digits (a n : Nat) : Nat :=
  a * ((10 ^ (3 ^ n) - 1) / 9)

-- Main theorem statement
theorem divisible_by_pow3 (a n : Nat) (h_pos : 0 < n) : (num_with_digits a n) % (3 ^ n) = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisible_by_pow3_l841_84106


namespace NUMINAMATH_GPT_average_mark_of_first_class_is_40_l841_84101

open Classical

noncomputable def average_mark_first_class (n1 n2 : ℕ) (m2 : ℕ) (a : ℚ) : ℚ :=
  let x := (a * (n1 + n2) - n2 * m2) / n1
  x

theorem average_mark_of_first_class_is_40 : average_mark_first_class 30 50 90 71.25 = 40 := by
  sorry

end NUMINAMATH_GPT_average_mark_of_first_class_is_40_l841_84101


namespace NUMINAMATH_GPT_team_incorrect_answers_l841_84146

theorem team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) :
  total_questions = 35 → riley_mistakes = 3 → 
  ofelia_correct = ((total_questions - riley_mistakes) / 2 + 5) → 
  riley_mistakes + (total_questions - ofelia_correct) = 17 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_team_incorrect_answers_l841_84146


namespace NUMINAMATH_GPT_graph_is_hyperbola_l841_84177

theorem graph_is_hyperbola : 
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 4 ↔ x * y = 2 := 
by
  sorry

end NUMINAMATH_GPT_graph_is_hyperbola_l841_84177


namespace NUMINAMATH_GPT_base9_num_digits_2500_l841_84137

theorem base9_num_digits_2500 : 
  ∀ (n : ℕ), (9^1 = 9) → (9^2 = 81) → (9^3 = 729) → (9^4 = 6561) → n = 4 := by
  sorry

end NUMINAMATH_GPT_base9_num_digits_2500_l841_84137


namespace NUMINAMATH_GPT_sum_a_b_eq_34_over_3_l841_84127

theorem sum_a_b_eq_34_over_3 (a b: ℚ)
  (h1 : 2 * a + 5 * b = 43)
  (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 :=
sorry

end NUMINAMATH_GPT_sum_a_b_eq_34_over_3_l841_84127


namespace NUMINAMATH_GPT_highest_price_more_than_lowest_l841_84155

-- Define the highest price and lowest price.
def highest_price : ℕ := 350
def lowest_price : ℕ := 250

-- Define the calculation for the percentage increase.
def percentage_increase (hp lp : ℕ) : ℕ :=
  ((hp - lp) * 100) / lp

-- The theorem to prove the required percentage increase.
theorem highest_price_more_than_lowest : percentage_increase highest_price lowest_price = 40 := 
  by sorry

end NUMINAMATH_GPT_highest_price_more_than_lowest_l841_84155


namespace NUMINAMATH_GPT_min_capacity_for_raft_l841_84171

-- Define the weights of the animals
def weight_mouse : ℕ := 70
def weight_mole : ℕ := 90
def weight_hamster : ℕ := 120

-- Define the number of each type of animal
def number_mice : ℕ := 5
def number_moles : ℕ := 3
def number_hamsters : ℕ := 4

-- Define the minimum weight capacity for the raft
def min_weight_capacity : ℕ := 140

-- Prove that the minimum weight capacity the raft must have to transport all animals is 140 grams.
theorem min_capacity_for_raft :
  (weight_mouse * 2 ≤ min_weight_capacity) ∧ 
  (∀ trip_weight, trip_weight ≥ min_weight_capacity → 
    (trip_weight = weight_mouse * 2 ∨ trip_weight = weight_mole * 2 ∨ trip_weight = weight_hamster * 2)) :=
by 
  sorry

end NUMINAMATH_GPT_min_capacity_for_raft_l841_84171


namespace NUMINAMATH_GPT_fishing_tomorrow_l841_84165

theorem fishing_tomorrow (yesterday_fishers today_fishers : ℕ)
  (every_day_fishers every_other_day_fishers every_three_days_fishers : ℕ)
  (total_population : ℕ):
  yesterday_fishers = 12 → 
  today_fishers = 10 → 
  every_day_fishers = 7 → 
  every_other_day_fishers = 8 → 
  every_three_days_fishers = 3 → 
  total_population = yesterday_fishers + today_fishers + (total_population - (every_day_fishers + every_other_day_fishers + every_three_days_fishers)) →
  ∃ tomorrow_fishers : ℕ, tomorrow_fishers = 15 :=
by {
  -- This is a statement definition, the proof is not required and thus marked as "sorry:"
  sorry
}

end NUMINAMATH_GPT_fishing_tomorrow_l841_84165


namespace NUMINAMATH_GPT_mean_height_basketball_team_l841_84110

def heights : List ℕ :=
  [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

def mean_height (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_height_basketball_team :
  mean_height heights = 70 := by
  sorry

end NUMINAMATH_GPT_mean_height_basketball_team_l841_84110


namespace NUMINAMATH_GPT_integer_root_of_P_l841_84167

def P (x : ℤ) : ℤ := x^3 - 4 * x^2 - 8 * x + 24 

theorem integer_root_of_P :
  (∃ x : ℤ, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x = 2) :=
sorry

end NUMINAMATH_GPT_integer_root_of_P_l841_84167


namespace NUMINAMATH_GPT_smallest_number_groups_l841_84100

theorem smallest_number_groups (x : ℕ) (h₁ : x % 18 = 0) (h₂ : x % 45 = 0) : x = 90 :=
sorry

end NUMINAMATH_GPT_smallest_number_groups_l841_84100


namespace NUMINAMATH_GPT_bins_of_vegetables_l841_84132

-- Define the conditions
def total_bins : ℝ := 0.75
def bins_of_soup : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

-- Define the statement to be proved
theorem bins_of_vegetables :
  total_bins = bins_of_soup + (0.13) + bins_of_pasta := 
sorry

end NUMINAMATH_GPT_bins_of_vegetables_l841_84132
