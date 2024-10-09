import Mathlib

namespace part_a_part_b_part_c_l828_82812

def op (a b : ℕ) : ℕ := a ^ b + b ^ a

theorem part_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : op a b = op b a :=
by
  dsimp [op]
  rw [add_comm]

theorem part_b (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op a (op b c) = op (op a b) c) :=
by
  -- example counter: a = 2, b = 2, c = 2 
  -- 2 ^ (2^2 + 2^2) + (2^2 + 2^2) ^ 2 ≠ (2^2 + 2 ^ 2) ^ 2 + 8 ^ 2
  sorry

theorem part_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op (op a b) (op b c) = op (op b a) (op c b)) :=
by
  -- example counter: a = 2, b = 3, c = 2 
  -- This will involve specific calculations showing the inequality.
  sorry

end part_a_part_b_part_c_l828_82812


namespace cylinder_height_in_hemisphere_l828_82810

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l828_82810


namespace four_P_plus_five_square_of_nat_l828_82840

theorem four_P_plus_five_square_of_nat 
  (a b : ℕ)
  (P : ℕ)
  (hP : P = (Nat.lcm a b) / (a + 1) + (Nat.lcm a b) / (b + 1))
  (h_prime : Nat.Prime P) : 
  ∃ n : ℕ, 4 * P + 5 = (2 * n + 1) ^ 2 :=
by
  sorry

end four_P_plus_five_square_of_nat_l828_82840


namespace area_of_rectangle_is_432_l828_82814

/-- Define the width of the rectangle --/
def width : ℕ := 12

/-- Define the length of the rectangle, which is three times the width --/
def length : ℕ := 3 * width

/-- The area of the rectangle is length multiplied by width --/
def area : ℕ := length * width

/-- Proof problem: the area of the rectangle is 432 square meters --/
theorem area_of_rectangle_is_432 :
  area = 432 :=
sorry

end area_of_rectangle_is_432_l828_82814


namespace cos_double_angle_sub_pi_six_l828_82804

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π / 3)
variable (h2 : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5)

theorem cos_double_angle_sub_pi_six :
  Real.cos (2 * α - π / 6) = 4 / 5 :=
by
  sorry

end cos_double_angle_sub_pi_six_l828_82804


namespace factorial_square_ge_power_l828_82846

theorem factorial_square_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := 
by sorry

end factorial_square_ge_power_l828_82846


namespace apple_tree_yield_l828_82891

theorem apple_tree_yield (A : ℝ) 
    (h1 : Magdalena_picks_day1 = A / 5)
    (h2 : Magdalena_picks_day2 = 2 * (A / 5))
    (h3 : Magdalena_picks_day3 = (A / 5) + 20)
    (h4 : remaining_apples = 20)
    (total_picked : Magdalena_picks_day1 + Magdalena_picks_day2 + Magdalena_picks_day3 + remaining_apples = A)
    : A = 200 :=
by
    sorry

end apple_tree_yield_l828_82891


namespace sequence_sum_l828_82830

theorem sequence_sum :
  1 - 4 + 7 - 10 + 13 - 16 + 19 - 22 + 25 - 28 + 31 - 34 + 37 - 40 + 43 - 46 + 49 - 52 + 55 = 28 :=
by
  sorry

end sequence_sum_l828_82830


namespace distance_from_dormitory_to_city_l828_82873

theorem distance_from_dormitory_to_city (D : ℝ) (h : (1/2) * D + (1/4) * D + 6 = D) : D = 24 :=
by
  sorry

end distance_from_dormitory_to_city_l828_82873


namespace income_increase_is_17_percent_l828_82875

def sales_percent_increase (original_items : ℕ) 
                           (original_price : ℝ) 
                           (discount_percent : ℝ) 
                           (sales_increase_percent : ℝ) 
                           (new_items_sold : ℕ) 
                           (new_income : ℝ)
                           (percent_increase : ℝ) : Prop :=
  let original_income := original_items * original_price
  let discounted_price := original_price * (1 - discount_percent / 100)
  let increased_sales := original_items + (original_items * sales_increase_percent / 100)
  original_income = original_items * original_price ∧
  new_income = discounted_price * increased_sales ∧
  new_items_sold = original_items * (1 + sales_increase_percent / 100) ∧
  percent_increase = ((new_income - original_income) / original_income) * 100 ∧
  original_items = 100 ∧ original_price = 1 ∧ discount_percent = 10 ∧ sales_increase_percent = 30 ∧ 
  new_items_sold = 130 ∧ new_income = 117 ∧ percent_increase = 17

theorem income_increase_is_17_percent :
  sales_percent_increase 100 1 10 30 130 117 17 :=
sorry

end income_increase_is_17_percent_l828_82875


namespace sum_is_integer_l828_82870

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 :=
  sorry

end sum_is_integer_l828_82870


namespace smallest_K_for_triangle_l828_82880

theorem smallest_K_for_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) 
  : ∃ K : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → a + c > b → (a^2 + c^2) / b^2 > K) ∧ K = 1 / 2 :=
by
  sorry

end smallest_K_for_triangle_l828_82880


namespace length_of_BC_l828_82858

-- Define the given conditions and the theorem using Lean
theorem length_of_BC 
  (A B C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hB : ∃ b : ℝ, B = (-b, -b^2)) 
  (hC : ∃ b : ℝ, C = (b, -b^2)) 
  (hBC_parallel_x_axis : ∀ b : ℝ, C.2 = B.2)
  (hArea : ∀ b : ℝ, b^3 = 72) 
  : ∀ b : ℝ, (BC : ℝ) = 2 * b := 
by
  sorry

end length_of_BC_l828_82858


namespace minimize_f_minimize_f_exact_l828_82871

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 14 * x - 20

-- State the theorem that x = -7 minimizes the function f(x)
theorem minimize_f : ∀ x : ℝ, f x ≥ f (-7) :=
by
  intro x
  unfold f
  sorry

-- An alternative statement could include the exact condition for the minimum value
theorem minimize_f_exact : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ x = -7 :=
by
  use -7
  intro y
  unfold f
  sorry

end minimize_f_minimize_f_exact_l828_82871


namespace lamp_count_and_profit_l828_82827

-- Define the parameters given in the problem
def total_lamps : ℕ := 50
def total_cost : ℕ := 2500
def cost_A : ℕ := 40
def cost_B : ℕ := 65
def marked_A : ℕ := 60
def marked_B : ℕ := 100
def discount_A : ℕ := 10 -- percent
def discount_B : ℕ := 30 -- percent

-- Derived definitions from the solution
def lamps_A : ℕ := 30
def lamps_B : ℕ := 20
def selling_price_A : ℕ := marked_A * (100 - discount_A) / 100
def selling_price_B : ℕ := marked_B * (100 - discount_B) / 100
def profit_A : ℕ := selling_price_A - cost_A
def profit_B : ℕ := selling_price_B - cost_B
def total_profit : ℕ := (profit_A * lamps_A) + (profit_B * lamps_B)

-- Lean statement
theorem lamp_count_and_profit :
  lamps_A + lamps_B = total_lamps ∧
  (cost_A * lamps_A + cost_B * lamps_B) = total_cost ∧
  total_profit = 520 := by
  -- proofs will go here
  sorry

end lamp_count_and_profit_l828_82827


namespace Danielle_has_6_rooms_l828_82816

axiom Danielle_rooms : ℕ
axiom Heidi_rooms : ℕ
axiom Grant_rooms : ℕ

axiom Heidi_has_3_times_Danielle : Heidi_rooms = 3 * Danielle_rooms
axiom Grant_has_1_9_Heidi : Grant_rooms = Heidi_rooms / 9
axiom Grant_has_2_rooms : Grant_rooms = 2

theorem Danielle_has_6_rooms : Danielle_rooms = 6 :=
by {
  -- proof steps would go here
  sorry
}

end Danielle_has_6_rooms_l828_82816


namespace weeks_of_exercise_l828_82868

def hours_per_day : ℕ := 1
def days_per_week : ℕ := 5
def total_hours : ℕ := 40

def weekly_hours : ℕ := hours_per_day * days_per_week

theorem weeks_of_exercise (W : ℕ) (h : total_hours = weekly_hours * W) : W = 8 :=
by
  sorry

end weeks_of_exercise_l828_82868


namespace find_g_l828_82800

-- Define given functions and terms
def f1 (x : ℝ) := 7 * x^4 - 4 * x^3 + 2 * x - 5
def f2 (x : ℝ) := 5 * x^3 - 3 * x^2 + 4 * x - 1
def g (x : ℝ) := -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4

-- Theorem to prove that g(x) satisfies the given condition
theorem find_g : ∀ x : ℝ, f1 x + g x = f2 x :=
by 
  -- Alternatively: Proof is required here
  sorry

end find_g_l828_82800


namespace min_value_of_x_prime_factors_l828_82884

theorem min_value_of_x_prime_factors (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
    (h : 5 * x^7 = 13 * y^11)
    (hx_factorization : x = a^c * b^d) : a + b + c + d = 32 := sorry

end min_value_of_x_prime_factors_l828_82884


namespace roots_of_f_non_roots_of_g_l828_82874

-- Part (a)

def f (x : ℚ) := x^20 - 123 * x^10 + 1

theorem roots_of_f (a : ℚ) (h : f a = 0) : 
  f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0 :=
by
  sorry

-- Part (b)

def g (x : ℚ) := x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1

theorem non_roots_of_g (β : ℚ) (h : g β = 0) : 
  g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0 :=
by
  sorry

end roots_of_f_non_roots_of_g_l828_82874


namespace text_messages_relationship_l828_82801

theorem text_messages_relationship (l x : ℕ) (h_l : l = 111) (h_combined : l + x = 283) : x = l + 61 :=
by sorry

end text_messages_relationship_l828_82801


namespace find_middle_side_length_l828_82815

theorem find_middle_side_length (a b c : ℕ) (h1 : a + b + c = 2022) (h2 : c - b = 1) (h3 : b - a = 2) :
  b = 674 := 
by
  -- The proof goes here, but we skip it using sorry.
  sorry

end find_middle_side_length_l828_82815


namespace sqrt_of_sqrt_81_l828_82866

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l828_82866


namespace suresh_borrowed_amount_l828_82828

theorem suresh_borrowed_amount 
  (P: ℝ)
  (i1 i2 i3: ℝ)
  (t1 t2 t3: ℝ)
  (total_interest: ℝ)
  (h1 : i1 = 0.12) 
  (h2 : t1 = 3)
  (h3 : i2 = 0.09)
  (h4 : t2 = 5)
  (h5 : i3 = 0.13)
  (h6 : t3 = 3)
  (h_total : total_interest = 8160) 
  (h_interest_eq : total_interest = P * i1 * t1 + P * i2 * t2 + P * i3 * t3)
  : P = 6800 :=
by
  sorry

end suresh_borrowed_amount_l828_82828


namespace incorrect_transformation_l828_82822

-- Definitions based on conditions
variable (a b c : ℝ)

-- Conditions
axiom eq_add_six (h : a = b) : a + 6 = b + 6
axiom eq_div_nine (h : a = b) : a / 9 = b / 9
axiom eq_mul_c (h : a / c = b / c) (hc : c ≠ 0) : a = b
axiom eq_div_neg_two (h : -2 * a = -2 * b) : a = b

-- Proving the incorrect transformation statement
theorem incorrect_transformation : ¬ (a = -b) ∧ (-2 * a = -2 * b → a = b) := by
  sorry

end incorrect_transformation_l828_82822


namespace max_value_of_k_l828_82894

theorem max_value_of_k (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  (∀ k : ℝ, (1 / m + 2 / (1 - 2 * m)) ≥ k) ↔ k ≤ 8 := 
sorry

end max_value_of_k_l828_82894


namespace area_of_equilateral_triangle_example_l828_82897

noncomputable def area_of_equilateral_triangle_with_internal_point (a b c : ℝ) (d_pa : ℝ) (d_pb : ℝ) (d_pc : ℝ) : ℝ :=
  if h : ((d_pa = 3) ∧ (d_pb = 4) ∧ (d_pc = 5)) then
    (9 + (25 * Real.sqrt 3)/4)
  else
    0

theorem area_of_equilateral_triangle_example :
  area_of_equilateral_triangle_with_internal_point 3 4 5 3 4 5 = 9 + (25 * Real.sqrt 3)/4 :=
  by sorry

end area_of_equilateral_triangle_example_l828_82897


namespace min_x_plus_y_l828_82849

theorem min_x_plus_y (x y : ℝ) (h1 : x * y = 2 * x + y + 2) (h2 : x > 1) :
  x + y ≥ 7 :=
sorry

end min_x_plus_y_l828_82849


namespace sum_of_g1_values_l828_82864

noncomputable def g : Polynomial ℝ := sorry

theorem sum_of_g1_values :
  (∀ x : ℝ, x ≠ 0 → g.eval (x-1) + g.eval x + g.eval (x+1) = (g.eval x)^2 / (4036 * x)) →
  g.degree ≠ 0 →
  g.eval 1 = 12108 :=
by
  sorry

end sum_of_g1_values_l828_82864


namespace triangle_AC_length_l828_82854

open Real

theorem triangle_AC_length (A : ℝ) (AB AC S : ℝ) (h1 : A = π / 3) (h2 : AB = 2) (h3 : S = sqrt 3 / 2) : AC = 1 :=
by
  sorry

end triangle_AC_length_l828_82854


namespace length_of_AE_l828_82845

theorem length_of_AE (AF CE ED : ℝ) (ABCD_area : ℝ) (hAF : AF = 30) (hCE : CE = 40) (hED : ED = 50) (hABCD_area : ABCD_area = 7200) : ∃ AE : ℝ, AE = 322.5 := sorry

end length_of_AE_l828_82845


namespace no_hexagon_cross_section_l828_82850

-- Define the shape of the cross-section resulting from cutting a triangular prism with a plane
inductive Shape
| triangle
| quadrilateral
| pentagon
| hexagon

-- Define the condition of cutting a triangular prism
structure TriangularPrism where
  cut : Shape

-- The theorem stating that cutting a triangular prism with a plane cannot result in a hexagon
theorem no_hexagon_cross_section (P : TriangularPrism) : P.cut ≠ Shape.hexagon :=
by
  sorry

end no_hexagon_cross_section_l828_82850


namespace range_of_m_l828_82809

theorem range_of_m (x m : ℝ) (h1 : (x ≥ 0) ∧ (x ≠ 1) ∧ (x = (6 - m) / 4)) :
    m ≤ 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l828_82809


namespace ellipse_semi_focal_range_l828_82877

-- Definitions and conditions from the problem
variables (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : a^2 = b^2 + c^2)

-- Statement of the theorem
theorem ellipse_semi_focal_range : 1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 :=
by 
  sorry

end ellipse_semi_focal_range_l828_82877


namespace f_has_exactly_one_zero_point_a_range_condition_l828_82818

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * Real.log x + 2 / (x + 1)

theorem f_has_exactly_one_zero_point :
  ∃! x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ f x = 0 := sorry

theorem a_range_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) 1 → ∀ t : ℝ, t ∈ Set.Icc (1 / 2) 2 → f x ≥ t^3 - t^2 - 2 * a * t + 2) → a ≥ 5 / 4 := sorry

end f_has_exactly_one_zero_point_a_range_condition_l828_82818


namespace original_hourly_wage_l828_82811

theorem original_hourly_wage 
  (daily_wage_increase : ∀ W : ℝ, 1.60 * W + 10 = 45)
  (work_hours : ℝ := 8) : 
  ∃ W_hourly : ℝ, W_hourly = 2.73 :=
by 
  have W : ℝ := (45 - 10) / 1.60 
  have W_hourly : ℝ := W / work_hours
  use W_hourly 
  sorry

end original_hourly_wage_l828_82811


namespace solve_for_x_l828_82823

-- declare an existential quantifier to encapsulate the condition and the answer.
theorem solve_for_x : ∃ x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := 
by 
  -- begin sorry to skip the proof part
  sorry

end solve_for_x_l828_82823


namespace ratio_of_buttons_to_magnets_per_earring_l828_82831

-- Definitions related to the problem statement
def gemstones_per_button : ℕ := 3
def magnets_per_earring : ℕ := 2
def sets_of_earrings : ℕ := 4
def required_gemstones : ℕ := 24

-- Problem statement translation into Lean 4
theorem ratio_of_buttons_to_magnets_per_earring :
  (required_gemstones / gemstones_per_button / (sets_of_earrings * 2)) = 1 / 2 := by
  sorry

end ratio_of_buttons_to_magnets_per_earring_l828_82831


namespace derivative_at_x₀_l828_82808

-- Define the function y = (x - 2)^2
def f (x : ℝ) : ℝ := (x - 2) ^ 2

-- Define the point of interest
def x₀ : ℝ := 1

-- State the problem and the correct answer
theorem derivative_at_x₀ : (deriv f x₀) = -2 := by
  sorry

end derivative_at_x₀_l828_82808


namespace slope_of_parallel_line_l828_82844

theorem slope_of_parallel_line (a b c : ℝ) (h: 3*a + 6*b = -24) :
  ∃ m : ℝ, (a * 3 + b * 6 = c) → m = -1/2 :=
by
  sorry

end slope_of_parallel_line_l828_82844


namespace find_number_l828_82826

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l828_82826


namespace bowling_ball_weight_l828_82867

theorem bowling_ball_weight (b c : ℝ) (h1 : 9 * b = 6 * c) (h2 : 4 * c = 120) : b = 20 :=
sorry

end bowling_ball_weight_l828_82867


namespace function_is_odd_and_increasing_l828_82819

-- Define the function y = x^(3/5)
def f (x : ℝ) : ℝ := x ^ (3 / 5)

-- Define what it means for the function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for the function to be increasing in its domain
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- The proposition to prove
theorem function_is_odd_and_increasing :
  is_odd f ∧ is_increasing f :=
by
  sorry

end function_is_odd_and_increasing_l828_82819


namespace original_price_calc_l828_82834

theorem original_price_calc (h : 1.08 * x = 2) : x = 100 / 54 := by
  sorry

end original_price_calc_l828_82834


namespace distance_triangle_four_points_l828_82878

variable {X : Type*} [MetricSpace X]

theorem distance_triangle_four_points (A B C D : X) :
  dist A D ≤ dist A B + dist B C + dist C D :=
by
  sorry

end distance_triangle_four_points_l828_82878


namespace solution_set_of_inequality_l828_82885

theorem solution_set_of_inequality (x : ℝ) : (x^2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ 1) := 
by 
  sorry

end solution_set_of_inequality_l828_82885


namespace derivative_at_x_equals_1_l828_82806

variable (x : ℝ)
def y : ℝ := (x + 1) * (x - 1)

theorem derivative_at_x_equals_1 : deriv y 1 = 2 :=
by
  sorry

end derivative_at_x_equals_1_l828_82806


namespace evaluate_expression_l828_82838

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) : 2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end evaluate_expression_l828_82838


namespace domain_of_sqrt_l828_82872

theorem domain_of_sqrt (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by sorry

end domain_of_sqrt_l828_82872


namespace rose_spent_on_food_l828_82821

theorem rose_spent_on_food (T : ℝ) 
  (h_clothing : 0.5 * T = 0.5 * T)
  (h_other_items : 0.3 * T = 0.3 * T)
  (h_total_tax : 0.044 * T = 0.044 * T)
  (h_tax_clothing : 0.04 * 0.5 * T = 0.02 * T)
  (h_tax_other_items : 0.08 * 0.3 * T = 0.024 * T) :
  (0.2 * T = T - (0.5 * T + 0.3 * T)) :=
by sorry

end rose_spent_on_food_l828_82821


namespace sum_faces_of_cube_l828_82876

theorem sum_faces_of_cube (p u q v r w : ℕ) (hp : 0 < p) (hu : 0 < u) (hq : 0 < q) (hv : 0 < v)
    (hr : 0 < r) (hw : 0 < w)
    (h_sum_vertices : p * q * r + p * v * r + p * q * w + p * v * w 
        + u * q * r + u * v * r + u * q * w + u * v * w = 2310) : 
    p + u + q + v + r + w = 40 := 
sorry

end sum_faces_of_cube_l828_82876


namespace Lacy_correct_percentage_l828_82802

def problems_exam (y : ℕ) := 10 * y
def problems_section1 (y : ℕ) := 6 * y
def problems_section2 (y : ℕ) := 4 * y
def missed_section1 (y : ℕ) := 2 * y
def missed_section2 (y : ℕ) := y
def solved_section1 (y : ℕ) := problems_section1 y - missed_section1 y
def solved_section2 (y : ℕ) := problems_section2 y - missed_section2 y
def total_solved (y : ℕ) := solved_section1 y + solved_section2 y
def percent_correct (y : ℕ) := (total_solved y : ℚ) / (problems_exam y) * 100

theorem Lacy_correct_percentage (y : ℕ) : percent_correct y = 70 := by
  -- Proof would go here
  sorry

end Lacy_correct_percentage_l828_82802


namespace top_card_is_red_l828_82842

noncomputable def standard_deck (ranks : ℕ) (suits : ℕ) : ℕ := ranks * suits

def red_cards_in_deck (hearts : ℕ) (diamonds : ℕ) : ℕ := hearts + diamonds

noncomputable def probability_red_card (red_cards : ℕ) (total_cards : ℕ) : ℚ := red_cards / total_cards

theorem top_card_is_red (hearts diamonds spades clubs : ℕ) (deck_size : ℕ)
  (H1 : hearts = 13) (H2 : diamonds = 13) (H3 : spades = 13) (H4 : clubs = 13) (H5 : deck_size = 52):
  probability_red_card (red_cards_in_deck hearts diamonds) deck_size = 1/2 :=
by 
  sorry

end top_card_is_red_l828_82842


namespace gel_pen_price_relation_b_l828_82860

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l828_82860


namespace monotone_decreasing_intervals_l828_82887

theorem monotone_decreasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, deriv f x = (x - 2) * (x^2 - 1)) :
  ((∀ x : ℝ, x < -1 → deriv f x < 0) ∧ (∀ x : ℝ, 1 < x → x < 2 → deriv f x < 0)) :=
by
  sorry

end monotone_decreasing_intervals_l828_82887


namespace greatest_integer_b_for_no_real_roots_l828_82863

theorem greatest_integer_b_for_no_real_roots (b : ℤ) :
  (∀ x : ℝ, x^2 + (b:ℝ)*x + 10 ≠ 0) ↔ b ≤ 6 :=
sorry

end greatest_integer_b_for_no_real_roots_l828_82863


namespace quadratic_has_two_real_roots_l828_82892

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x^2 - (m + 1) * x + (3 * m - 6) = 0 :=
by
  sorry

end quadratic_has_two_real_roots_l828_82892


namespace ounces_per_gallon_l828_82847

-- conditions
def gallons_of_milk (james : Type) : ℕ := 3
def ounces_drank (james : Type) : ℕ := 13
def ounces_left (james : Type) : ℕ := 371

-- question
def ounces_in_gallon (james : Type) : ℕ := 128

-- proof statement
theorem ounces_per_gallon (james : Type) :
  (gallons_of_milk james) * (ounces_in_gallon james) = (ounces_left james + ounces_drank james) :=
sorry

end ounces_per_gallon_l828_82847


namespace smallest_n_for_divisibility_l828_82829

noncomputable def geometric_sequence (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem smallest_n_for_divisibility (h₁ : ∀ n : ℕ, geometric_sequence (1/2 : ℚ) 60 n = (1/2 : ℚ) * 60^(n-1))
    (h₂ : (60 : ℚ) * (1 / 2) = 30)
    (n : ℕ) :
  (∃ n : ℕ, n ≥ 1 ∧ (geometric_sequence (1/2 : ℚ) 60 n) ≥ 10^6) ↔ n = 7 :=
by
  sorry

end smallest_n_for_divisibility_l828_82829


namespace possible_values_of_a_l828_82895

noncomputable def f (x a : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 2 * a * x + 2 else x + 9 / x - 3 * a

theorem possible_values_of_a (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ 1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end possible_values_of_a_l828_82895


namespace mean_total_sample_variance_total_sample_expected_final_score_l828_82861

section SeagrassStatistics

variables (m n : ℕ) (mean_x mean_y: ℝ) (var_x var_y: ℝ) (A_win_A B_win_A : ℝ)

-- Assumptions from the conditions
variable (hp1 : m = 12)
variable (hp2 : mean_x = 18)
variable (hp3 : var_x = 19)
variable (hp4 : n = 18)
variable (hp5 : mean_y = 36)
variable (hp6 : var_y = 70)
variable (hp7 : A_win_A = 3 / 5)
variable (hp8 : B_win_A = 1 / 2)

-- Statements to prove
theorem mean_total_sample (m n : ℕ) (mean_x mean_y : ℝ) : 
  m * mean_x + n * mean_y = (m + n) * 28.8 := sorry

theorem variance_total_sample (m n : ℕ) (mean_x mean_y var_x var_y : ℝ) :
  m * (var_x + (mean_x - 28.8)^2) + n * (var_y + (mean_y - 28.8)^2) = (m + n) * 127.36 := sorry

theorem expected_final_score (A_win_A B_win_A : ℝ) :
  2 * ((6/25) * 1 + (15/25) * 2 + (4/25) * 0) = 36 / 25 := sorry

end SeagrassStatistics

end mean_total_sample_variance_total_sample_expected_final_score_l828_82861


namespace find_first_number_l828_82825

/-- The Least Common Multiple (LCM) of two numbers A and B is 2310,
    and their Highest Common Factor (HCF) is 30.
    Given one of the numbers B is 180, find the other number A. -/
theorem find_first_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 30) (h3 : B = 180) (h4 : A * B = LCM * HCF) :
  A = 385 :=
by sorry

end find_first_number_l828_82825


namespace total_fare_for_100_miles_l828_82853

theorem total_fare_for_100_miles (b c : ℝ) (h₁ : 200 = b + 80 * c) : 240 = b + 100 * c :=
sorry

end total_fare_for_100_miles_l828_82853


namespace green_ish_count_l828_82835

theorem green_ish_count (total : ℕ) (blue_ish : ℕ) (both : ℕ) (neither : ℕ) (green_ish : ℕ) :
  total = 150 ∧ blue_ish = 90 ∧ both = 40 ∧ neither = 30 → green_ish = 70 :=
by
  sorry

end green_ish_count_l828_82835


namespace max_students_divide_equal_pen_pencil_l828_82807

theorem max_students_divide_equal_pen_pencil : Nat.gcd 2500 1575 = 25 := 
by
  sorry

end max_students_divide_equal_pen_pencil_l828_82807


namespace broken_shells_count_l828_82813

-- Definitions from conditions
def total_perfect_shells := 17
def non_spiral_perfect_shells := 12
def extra_broken_spiral_shells := 21

-- Derived definitions
def perfect_spiral_shells : ℕ := total_perfect_shells - non_spiral_perfect_shells
def broken_spiral_shells : ℕ := perfect_spiral_shells + extra_broken_spiral_shells
def broken_shells : ℕ := 2 * broken_spiral_shells

-- The theorem to be proved
theorem broken_shells_count : broken_shells = 52 := by
  sorry

end broken_shells_count_l828_82813


namespace friends_carrying_bananas_l828_82841

theorem friends_carrying_bananas :
  let total_friends := 35
  let friends_with_pears := 14
  let friends_with_oranges := 8
  let friends_with_apples := 5
  total_friends - (friends_with_pears + friends_with_oranges + friends_with_apples) = 8 := 
by
  sorry

end friends_carrying_bananas_l828_82841


namespace jellybeans_final_count_l828_82898

-- Defining the initial number of jellybeans and operations
def initial_jellybeans : ℕ := 37
def removed_first : ℕ := 15
def added_back : ℕ := 5
def removed_second : ℕ := 4

-- Defining the final number of jellybeans to prove it equals 23
def final_jellybeans : ℕ := (initial_jellybeans - removed_first) + added_back - removed_second

-- The theorem that states the final number of jellybeans is 23
theorem jellybeans_final_count : final_jellybeans = 23 :=
by
  -- The proof will be provided here if needed
  sorry

end jellybeans_final_count_l828_82898


namespace hyperbola_real_axis_length_l828_82869

theorem hyperbola_real_axis_length
    (a b : ℝ) 
    (h_pos_a : a > 0) 
    (h_pos_b : b > 0) 
    (h_eccentricity : a * Real.sqrt 5 = Real.sqrt (a^2 + b^2))
    (h_distance : b * a * Real.sqrt 5 / Real.sqrt (a^2 + b^2) = 8) :
    2 * a = 8 :=
sorry

end hyperbola_real_axis_length_l828_82869


namespace solution_set_f_neg_x_l828_82833

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_f_neg_x (a b : ℝ) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x, f a b (-x) < 0 ↔ (x < -3 ∨ x > 1) :=
by
  intro x
  specialize h (-x)
  sorry

end solution_set_f_neg_x_l828_82833


namespace peter_total_dogs_l828_82852

def num_german_shepherds_sam : ℕ := 3
def num_french_bulldogs_sam : ℕ := 4
def num_german_shepherds_peter := 3 * num_german_shepherds_sam
def num_french_bulldogs_peter := 2 * num_french_bulldogs_sam

theorem peter_total_dogs : num_german_shepherds_peter + num_french_bulldogs_peter = 17 :=
by {
  -- adding proofs later
  sorry
}

end peter_total_dogs_l828_82852


namespace work_done_isothermal_l828_82805

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end work_done_isothermal_l828_82805


namespace min_top_block_sum_l828_82855

theorem min_top_block_sum : 
  ∀ (assign_numbers : ℕ → ℕ) 
  (layer_1 : Fin 16 → ℕ) (layer_2 : Fin 9 → ℕ) (layer_3 : Fin 4 → ℕ) (top_block : ℕ),
  (∀ i, layer_3 i = layer_2 (i / 2) + layer_2 ((i / 2) + 1) + layer_2 ((i / 2) + 3) + layer_2 ((i / 2) + 4)) →
  (∀ i, layer_2 i = layer_1 (i / 2) + layer_1 ((i / 2) + 1) + layer_1 ((i / 2) + 3) + layer_1 ((i / 2) + 4)) →
  (top_block = layer_3 0 + layer_3 1 + layer_3 2 + layer_3 3) →
  top_block = 40 :=
sorry

end min_top_block_sum_l828_82855


namespace problem1_solution_set_problem2_proof_l828_82832

-- Define the function f(x) with a given value of a.
def f (x : ℝ) (a : ℝ) : ℝ := |x + a|

-- Problem 1: Solve the inequality f(x) ≥ 5 - |x - 2| when a = 1.
theorem problem1_solution_set (x : ℝ) :
  f x 1 ≥ 5 - |x - 2| ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

-- Problem 2: Given the solution set of f(x) ≤ 5 is [-9, 1] and the equation 1/m + 1/(2n) = a, prove m + 2n ≥ 1
theorem problem2_proof (a m n : ℝ) (hma : a = 4) (hmpos : m > 0) (hnpos : n > 0) :
  (1 / m + 1 / (2 * n) = a) → m + 2 * n ≥ 1 :=
sorry

end problem1_solution_set_problem2_proof_l828_82832


namespace beth_coins_sold_l828_82851

def initial_coins : ℕ := 250
def additional_coins : ℕ := 75
def percentage_sold : ℚ := 60 / 100
def total_coins : ℕ := initial_coins + additional_coins
def coins_sold : ℚ := percentage_sold * total_coins

theorem beth_coins_sold : coins_sold = 195 :=
by
  -- Sorry is used to skip the proof as requested
  sorry

end beth_coins_sold_l828_82851


namespace scientific_notation_of_8200000_l828_82881

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l828_82881


namespace compare_neg_third_and_neg_point_three_l828_82859

/-- Compare two numbers -1/3 and -0.3 -/
theorem compare_neg_third_and_neg_point_three : (-1 / 3 : ℝ) < -0.3 :=
sorry

end compare_neg_third_and_neg_point_three_l828_82859


namespace exists_set_no_three_ap_l828_82889

theorem exists_set_no_three_ap (n : ℕ) (k : ℕ) :
  (n ≥ 1983) →
  (k ≤ 100000) →
  ∃ S : Finset ℕ,
    S.card = n ∧
    (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → b ≠ (a + c) / 2) :=
sorry

end exists_set_no_three_ap_l828_82889


namespace confidence_level_for_relationship_l828_82888

-- Define the problem conditions and the target question.
def chi_squared_value : ℝ := 8.654
def critical_value : ℝ := 6.635
def confidence_level : ℝ := 99

theorem confidence_level_for_relationship (h : chi_squared_value > critical_value) : confidence_level = 99 :=
sorry

end confidence_level_for_relationship_l828_82888


namespace frank_completes_book_in_three_days_l828_82803

-- Define the total number of pages in a book
def total_pages : ℕ := 249

-- Define the number of pages Frank reads per day
def pages_per_day : ℕ := 83

-- Define the number of days Frank needs to finish a book
def days_to_finish_book (total_pages pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

-- Theorem statement to prove that Frank finishes a book in 3 days
theorem frank_completes_book_in_three_days : days_to_finish_book total_pages pages_per_day = 3 := 
by {
  -- Proof goes here
  sorry
}

end frank_completes_book_in_three_days_l828_82803


namespace john_hourly_wage_l828_82839

theorem john_hourly_wage (days_off: ℕ) (hours_per_day: ℕ) (weekly_wage: ℕ) 
  (days_off_eq: days_off = 3) (hours_per_day_eq: hours_per_day = 4) (weekly_wage_eq: weekly_wage = 160):
  (weekly_wage / ((7 - days_off) * hours_per_day) = 10) :=
by
  /-
  Given:
  days_off = 3
  hours_per_day = 4
  weekly_wage = 160

  To prove:
  weekly_wage / ((7 - days_off) * hours_per_day) = 10
  -/
  sorry

end john_hourly_wage_l828_82839


namespace part1_part2_l828_82856

def A (x : ℤ) := ∃ m n : ℤ, x = m^2 - n^2
def B (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

theorem part1 (h1: A 8) (h2: A 9) (h3: ¬ A 10) : 
  (A 8) ∧ (A 9) ∧ (¬ A 10) :=
by {
  sorry
}

theorem part2 (x : ℤ) (h : A x) : B x :=
by {
  sorry
}

end part1_part2_l828_82856


namespace cos_angles_difference_cos_angles_sum_l828_82865

-- Part (a)
theorem cos_angles_difference: 
  (Real.cos (36 * Real.pi / 180) - Real.cos (72 * Real.pi / 180) = 1 / 2) :=
sorry

-- Part (b)
theorem cos_angles_sum: 
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2) :=
sorry

end cos_angles_difference_cos_angles_sum_l828_82865


namespace angle_greater_than_150_l828_82890

theorem angle_greater_than_150 (a b c R : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c < 2 * R) : 
  ∃ (A : ℝ), A > 150 ∧ ( ∃ (B C : ℝ), A + B + C = 180 ) :=
sorry

end angle_greater_than_150_l828_82890


namespace pressure_increases_when_block_submerged_l828_82857

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l828_82857


namespace smallest_positive_even_integer_l828_82862

noncomputable def smallest_even_integer (n : ℕ) : ℕ := 
  if 2 * n > 0 ∧ (3^(n * (n + 1) / 8)) > 500 then n else 0

theorem smallest_positive_even_integer :
  smallest_even_integer 6 = 6 :=
by
  -- Skipping the proofs
  sorry

end smallest_positive_even_integer_l828_82862


namespace temperature_range_l828_82820

-- Define the highest and lowest temperature conditions
variable (t : ℝ)
def highest_temp := t ≤ 30
def lowest_temp := 20 ≤ t

-- The theorem to prove the range of temperature change
theorem temperature_range (t : ℝ) (h_high : highest_temp t) (h_low : lowest_temp t) : 20 ≤ t ∧ t ≤ 30 :=
by 
  -- Insert the proof or leave as sorry for now
  sorry

end temperature_range_l828_82820


namespace ordered_pairs_divide_square_sum_l828_82882

theorem ordered_pairs_divide_square_sum :
  { (m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (m^2 + n^2) } = { (1, 2), (1, 3), (2, 1), (3, 1) } := 
sorry

end ordered_pairs_divide_square_sum_l828_82882


namespace sum_and_product_of_roots_l828_82836

theorem sum_and_product_of_roots (a b : ℝ) (h1 : a * a * a - 4 * a * a - a + 4 = 0)
  (h2 : b * b * b - 4 * b * b - b + 4 = 0) :
  a + b + a * b = -1 :=
sorry

end sum_and_product_of_roots_l828_82836


namespace total_amount_proof_l828_82899

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem total_amount_proof (x_ratio y_ratio z_ratio : ℝ) (y_share : ℝ) 
  (h1 : y_ratio = 0.45) (h2 : z_ratio = 0.50) (h3 : y_share = 54) 
  : total_amount (y_share / y_ratio) y_share (z_ratio * (y_share / y_ratio)) = 234 :=
by
  sorry

end total_amount_proof_l828_82899


namespace largest_integer_x_l828_82843

theorem largest_integer_x (x : ℤ) :
  (x ^ 2 - 11 * x + 28 < 0) → x ≤ 6 := sorry

end largest_integer_x_l828_82843


namespace B_pow_150_l828_82883

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_150 : B ^ 150 = 1 :=
by
  sorry

end B_pow_150_l828_82883


namespace total_tickets_sold_l828_82886

theorem total_tickets_sold
    (n₄₅ : ℕ) (n₆₀ : ℕ) (total_sales : ℝ) 
    (price₄₅ price₆₀ : ℝ)
    (h₁ : n₄₅ = 205)
    (h₂ : price₄₅ = 4.5)
    (h₃ : total_sales = 1972.5)
    (h₄ : price₆₀ = 6.0)
    (h₅ : total_sales = n₄₅ * price₄₅ + n₆₀ * price₆₀) :
    n₄₅ + n₆₀ = 380 := 
by
  sorry

end total_tickets_sold_l828_82886


namespace hugo_probability_l828_82817

noncomputable def P_hugo_first_roll_seven_given_win (P_Hugo_wins : ℚ) (P_first_roll_seven : ℚ)
  (P_all_others_roll_less_than_seven : ℚ) : ℚ :=
(P_first_roll_seven * P_all_others_roll_less_than_seven) / P_Hugo_wins

theorem hugo_probability :
  let P_Hugo_wins := (1 : ℚ) / 4
  let P_first_roll_seven := (1 : ℚ) / 8
  let P_all_others_roll_less_than_seven := (27 : ℚ) / 64
  P_hugo_first_roll_seven_given_win P_Hugo_wins P_first_roll_seven P_all_others_roll_less_than_seven = (27 : ℚ) / 128 :=
by
  sorry

end hugo_probability_l828_82817


namespace shaded_fraction_l828_82879

theorem shaded_fraction (side_length : ℝ) (base : ℝ) (height : ℝ) (H1: side_length = 4) (H2: base = 3) (H3: height = 2):
  ((side_length ^ 2) - 2 * (1 / 2 * base * height)) / (side_length ^ 2) = 5 / 8 := by
  sorry

end shaded_fraction_l828_82879


namespace solution_comparison_l828_82893

open Real

theorem solution_comparison (c d e f : ℝ) (hc : c ≠ 0) (he : e ≠ 0) :
  (-(d / c) > -(f / e)) ↔ ((f / e) > (d / c)) :=
by
  sorry

end solution_comparison_l828_82893


namespace number_of_sides_is_15_l828_82837

variable {n : ℕ} -- n is the number of sides

-- Define the conditions
def sum_of_all_but_one_angle (n : ℕ) : Prop :=
  180 * (n - 2) - 2190 > 0 ∧ 180 * (n - 2) - 2190 < 180

-- State the theorem to be proven
theorem number_of_sides_is_15 (n : ℕ) (h : sum_of_all_but_one_angle n) : n = 15 :=
sorry

end number_of_sides_is_15_l828_82837


namespace nails_remaining_proof_l828_82848

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l828_82848


namespace problem1_l828_82824

theorem problem1 (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end problem1_l828_82824


namespace fourth_term_eq_156_l828_82896

-- Definition of the sequence term
def seq_term (n : ℕ) : ℕ :=
  (List.range n).map (λ k => 5^k) |>.sum

-- Theorem to prove the fourth term equals 156
theorem fourth_term_eq_156 : seq_term 4 = 156 :=
sorry

end fourth_term_eq_156_l828_82896
