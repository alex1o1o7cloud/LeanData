import Mathlib

namespace max_xy_l710_71059

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  xy ≤ 2 :=
by
  sorry

end max_xy_l710_71059


namespace susan_age_in_5_years_l710_71066

variable (J N S X : ℕ)

-- Conditions
axiom h1 : J - 8 = 2 * (N - 8)
axiom h2 : J + X = 37
axiom h3 : S = N - 3

-- Theorem statement
theorem susan_age_in_5_years : S + 5 = N + 2 :=
by sorry

end susan_age_in_5_years_l710_71066


namespace tangent_lines_to_curve_l710_71036

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the general form of a tangent line
def tangent_line (x : ℝ) (y : ℝ) (m : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the conditions
def condition1 : Prop :=
  tangent_line 1 1 3 1 1

def condition2 : Prop :=
  tangent_line 1 1 (3/4) (-1/2) ((-1/2)^3)

-- Define the equations of the tangent lines
def line1 : Prop :=
  ∀ x y : ℝ, 3 * x - y - 2 = 0

def line2 : Prop :=
  ∀ x y : ℝ, 3 * x - 4 * y + 1 = 0

-- The final theorem statement
theorem tangent_lines_to_curve :
  (condition1 → line1) ∧ (condition2 → line2) :=
  by
    sorry -- Placeholder for proof

end tangent_lines_to_curve_l710_71036


namespace reflection_sum_coordinates_l710_71012

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

end reflection_sum_coordinates_l710_71012


namespace mul_powers_same_base_l710_71038

theorem mul_powers_same_base (a : ℝ) : a^3 * a^4 = a^7 := 
by 
  sorry

end mul_powers_same_base_l710_71038


namespace cylinder_volume_transformation_l710_71076

-- Define the original volume of the cylinder
def original_volume (V: ℝ) := V = 5

-- Define the transformation of quadrupling the dimensions of the cylinder
def new_volume (V V': ℝ) := V' = 64 * V

-- The goal is to show that under these conditions, the new volume is 320 gallons
theorem cylinder_volume_transformation (V V': ℝ) (h: original_volume V) (h': new_volume V V'):
  V' = 320 :=
by
  -- Proof is left as an exercise
  sorry

end cylinder_volume_transformation_l710_71076


namespace product_mod_7_l710_71054

theorem product_mod_7 (a b c : ℕ) (ha : a % 7 = 3) (hb : b % 7 = 4) (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 4 :=
sorry

end product_mod_7_l710_71054


namespace sum_of_roots_of_quadratic_eq_l710_71028

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots_of_quadratic_eq : 
  ∀ x y : ℝ, quadratic_eq 1 (-6) 8 x → quadratic_eq 1 (-6) 8 y → (x + y) = 6 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l710_71028


namespace Ricciana_run_distance_l710_71033

def Ricciana_jump : ℕ := 4

def Margarita_run : ℕ := 18

def Margarita_jump (Ricciana_jump : ℕ) : ℕ := 2 * Ricciana_jump - 1

def Margarita_total_distance (Margarita_run Margarita_jump : ℕ) : ℕ := Margarita_run + Margarita_jump

def Ricciana_total_distance (Ricciana_run Ricciana_jump : ℕ) : ℕ := Ricciana_run + Ricciana_jump

theorem Ricciana_run_distance (R : ℕ) 
  (Ricciana_total : ℕ := R + Ricciana_jump) 
  (Margarita_total : ℕ := Margarita_run + Margarita_jump Ricciana_jump) 
  (h : Margarita_total = Ricciana_total + 1) : 
  R = 20 :=
by
  sorry

end Ricciana_run_distance_l710_71033


namespace negation_of_proposition_l710_71083

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
sorry

end negation_of_proposition_l710_71083


namespace multiply_expression_l710_71072

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end multiply_expression_l710_71072


namespace log_equality_l710_71087

theorem log_equality (x : ℝ) : (8 : ℝ)^x = 16 ↔ x = 4 / 3 :=
by
  sorry

end log_equality_l710_71087


namespace max_remaining_area_l710_71063

theorem max_remaining_area (original_area : ℕ) (rec1 : ℕ × ℕ) (rec2 : ℕ × ℕ) (rec3 : ℕ × ℕ)
  (rec4 : ℕ × ℕ) (total_area_cutout : ℕ):
  original_area = 132 →
  rec1 = (1, 4) →
  rec2 = (2, 2) →
  rec3 = (2, 3) →
  rec4 = (2, 3) →
  total_area_cutout = 20 →
  original_area - total_area_cutout = 112 :=
by
  intros
  sorry

end max_remaining_area_l710_71063


namespace binomial_expansion_coefficient_x_l710_71090

theorem binomial_expansion_coefficient_x :
  (∃ (c : ℕ), (x : ℝ) → (x + 1/x^(1/2))^7 = c * x + (rest)) ∧ c = 35 := by
  sorry

end binomial_expansion_coefficient_x_l710_71090


namespace radius_of_circumscribed_sphere_l710_71057

noncomputable def circumscribedSphereRadius (a : ℝ) (α := 60 * Real.pi / 180) : ℝ :=
  5 * a / (4 * Real.sqrt 3)

theorem radius_of_circumscribed_sphere (a : ℝ) :
  circumscribedSphereRadius a = 5 * a / (4 * Real.sqrt 3) := by
  sorry

end radius_of_circumscribed_sphere_l710_71057


namespace measure_of_angle_B_l710_71018

-- Define the conditions and the goal as a theorem
theorem measure_of_angle_B (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : A = 3 * B)
  (triangle_angle_sum : A + B + C = 180) : B = 30 :=
by
  -- Substitute the conditions into Lean to express and prove the statement
  sorry

end measure_of_angle_B_l710_71018


namespace bowler_overs_l710_71031

theorem bowler_overs (x : ℕ) (h1 : ∀ y, y ≤ 3 * x) 
                     (h2 : y = 10) : x = 4 := by
  sorry

end bowler_overs_l710_71031


namespace merchant_profit_after_discount_l710_71049

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

end merchant_profit_after_discount_l710_71049


namespace line_circle_intersection_l710_71068

theorem line_circle_intersection (x y : ℝ) (h1 : 7 * x + 5 * y = 14) (h2 : x^2 + y^2 = 4) :
  ∃ (p q : ℝ), (7 * p + 5 * q = 14) ∧ (p^2 + q^2 = 4) ∧ (7 * p + 5 * q = 14) ∧ (p ≠ q) :=
sorry

end line_circle_intersection_l710_71068


namespace sum_coefficients_equals_l710_71003

theorem sum_coefficients_equals :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ), 
  (∀ x : ℤ, (2 * x + 1) ^ 5 = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_0 = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 3^5 - 1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h h0
  sorry

end sum_coefficients_equals_l710_71003


namespace laptop_price_reduction_l710_71045

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

end laptop_price_reduction_l710_71045


namespace lambda_value_l710_71094

-- Definitions provided in the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (A B C D : V)
-- Non-collinear vectors e1 and e2
variables (h_non_collinear : ∃ a b : ℝ, a ≠ b ∧ a • e1 + b • e2 ≠ 0)
-- Given vectors AB, BC, CD
variables (AB BC CD : V)
variables (lambda : ℝ)
-- Vector definitions based on given conditions
variables (h1 : AB = 2 • e1 + e2)
variables (h2 : BC = -e1 + 3 • e2)
variables (h3 : CD = lambda • e1 - e2)
-- Collinearity condition of points A, B, D
variables (collinear : ∃ β : ℝ, AB = β • (BC + CD))

-- The proof goal
theorem lambda_value (h1 : AB = 2 • e1 + e2) (h2 : BC = -e1 + 3 • e2) (h3 : CD = lambda • e1 - e2) (collinear : ∃ β : ℝ, AB = β • (BC + CD)) : lambda = 5 := 
sorry

end lambda_value_l710_71094


namespace solve_for_y_l710_71024

theorem solve_for_y :
  ∀ (y : ℝ), (9 * y^2 + 49 * y^2 + 21/2 * y^2 = 1300) → y = 4.34 := 
by sorry

end solve_for_y_l710_71024


namespace fourth_equation_l710_71004

theorem fourth_equation :
  (5 * 6 * 7 * 8) = (2^4) * 1 * 3 * 5 * 7 :=
by
  sorry

end fourth_equation_l710_71004


namespace regular_pay_per_hour_l710_71016

theorem regular_pay_per_hour (R : ℝ) (h : 40 * R + 11 * (2 * R) = 186) : R = 3 :=
by
  sorry

end regular_pay_per_hour_l710_71016


namespace dot_product_of_vectors_l710_71013

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1) - vector_a

theorem dot_product_of_vectors :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = -4 :=
by
  sorry

end dot_product_of_vectors_l710_71013


namespace ratio_of_cookies_l710_71021

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

end ratio_of_cookies_l710_71021


namespace fishing_tomorrow_l710_71037

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

end fishing_tomorrow_l710_71037


namespace wider_can_radius_l710_71060

theorem wider_can_radius (h : ℝ) : 
  (∃ r : ℝ, ∀ V : ℝ, V = π * 8^2 * 2 * h → V = π * r^2 * h → r = 8 * Real.sqrt 2) :=
by 
  sorry

end wider_can_radius_l710_71060


namespace hyperbola_focal_distance_solution_l710_71029

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

end hyperbola_focal_distance_solution_l710_71029


namespace A_investment_amount_l710_71010

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

end A_investment_amount_l710_71010


namespace negation_of_proposition_l710_71034

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x ≥ 0) ↔ (∃ x > 0, x^2 + x < 0) :=
by 
  sorry

end negation_of_proposition_l710_71034


namespace crows_eat_worms_l710_71086

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end crows_eat_worms_l710_71086


namespace B_completes_remaining_work_in_12_days_l710_71084

-- Definitions for conditions.
def work_rate_a := 1/15
def work_rate_b := 1/18
def days_worked_by_a := 5

-- Calculation of work done by A and the remaining work for B
def work_done_by_a := days_worked_by_a * work_rate_a
def remaining_work := 1 - work_done_by_a

-- Proof statement
theorem B_completes_remaining_work_in_12_days : 
  ∀ (work_rate_a work_rate_b : ℚ), 
    work_rate_a = 1/15 → 
    work_rate_b = 1/18 → 
    days_worked_by_a = 5 → 
    work_done_by_a = days_worked_by_a * work_rate_a → 
    remaining_work = 1 - work_done_by_a → 
    (remaining_work / work_rate_b) = 12 :=
by 
  intros 
  sorry

end B_completes_remaining_work_in_12_days_l710_71084


namespace range_of_a_circle_C_intersects_circle_D_l710_71005

/-- Definitions of circles C and D --/
def circle_C_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def circle_D_eq (x y m : ℝ) := x^2 + y^2 - 2 * m * x = 0

/-- Condition for the line intersecting Circle C --/
def line_intersects_circle_C (a : ℝ) := (∃ x y : ℝ, circle_C_eq x y ∧ (x + y = a))

/-- Proof of range for a --/
theorem range_of_a (a : ℝ) : line_intersects_circle_C a → (2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2) :=
sorry

/-- Proposition for point A lying on circle C and satisfying the inequality --/
def point_A_on_circle_C_and_inequality (m : ℝ) (x y : ℝ) :=
  circle_C_eq x y ∧ x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

/-- Proof that Circle C intersects Circle D --/
theorem circle_C_intersects_circle_D (m : ℝ) (a : ℝ) : 
  (∀ (x y : ℝ), point_A_on_circle_C_and_inequality m x y) →
  (1 ≤ m ∧
   ∃ (x y : ℝ), (circle_D_eq x y m ∧ (Real.sqrt ((m - 1)^2 + 1) < m + 1 ∧ Real.sqrt ((m - 1)^2 + 1) > m - 1))) :=
sorry

end range_of_a_circle_C_intersects_circle_D_l710_71005


namespace factorize_expression_l710_71041

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l710_71041


namespace annie_start_crayons_l710_71000

def start_crayons (end_crayons : ℕ) (added_crayons : ℕ) : ℕ := end_crayons - added_crayons

theorem annie_start_crayons (added_crayons end_crayons : ℕ) (h1 : added_crayons = 36) (h2 : end_crayons = 40) :
  start_crayons end_crayons added_crayons = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add sorry  -- skips the detailed proof

end annie_start_crayons_l710_71000


namespace cost_of_steel_ingot_l710_71014

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

end cost_of_steel_ingot_l710_71014


namespace Eli_saves_more_with_discount_A_l710_71042

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

end Eli_saves_more_with_discount_A_l710_71042


namespace each_charity_gets_45_dollars_l710_71062

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

end each_charity_gets_45_dollars_l710_71062


namespace raft_min_capacity_l710_71017

theorem raft_min_capacity
  (num_mice : ℕ) (weight_mouse : ℕ)
  (num_moles : ℕ) (weight_mole : ℕ)
  (num_hamsters : ℕ) (weight_hamster : ℕ)
  (raft_condition : ∀ (x y : ℕ), x + y ≥ 2 ∧ (x = weight_mouse ∨ x = weight_mole ∨ x = weight_hamster) ∧ (y = weight_mouse ∨ y = weight_mole ∨ y = weight_hamster) → x + y ≥ 140)
  : 140 ≤ ((num_mice*weight_mouse + num_moles*weight_mole + num_hamsters*weight_hamster) / 2) := sorry

end raft_min_capacity_l710_71017


namespace perimeter_of_figure_l710_71015

theorem perimeter_of_figure (a b c d : ℕ) (p : ℕ) (h1 : a = 6) (h2 : b = 3) (h3 : c = 2) (h4 : d = 4) (h5 : p = a * b + c * d) : p = 26 :=
by
  sorry

end perimeter_of_figure_l710_71015


namespace min_m_plus_inv_m_min_frac_expr_l710_71088

-- Sub-problem (1): Minimum value of m + 1/m for m > 0.
theorem min_m_plus_inv_m (m : ℝ) (h : m > 0) : m + 1/m = 2 :=
sorry

-- Sub-problem (2): Minimum value of (x^2 + x - 5)/(x - 2) for x > 2.
theorem min_frac_expr (x : ℝ) (h : x > 2) : (x^2 + x - 5)/(x - 2) = 7 :=
sorry

end min_m_plus_inv_m_min_frac_expr_l710_71088


namespace value_of_b_pos_sum_for_all_x_l710_71093

noncomputable def f (b : ℝ) (x : ℝ) := 3 * x^2 - 2 * x + b
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + b * x - 1
noncomputable def sum_f_g (b : ℝ) (x : ℝ) := f b x + g b x

theorem value_of_b (b : ℝ) (h : ∀ x : ℝ, (sum_f_g b x = 4 * x^2 + (b - 2) * x + (b - 1))) :
  b = 2 := 
sorry

theorem pos_sum_for_all_x :
  ∀ x : ℝ, 4 * x^2 + 1 > 0 := 
sorry

end value_of_b_pos_sum_for_all_x_l710_71093


namespace integer_solutions_l710_71035

theorem integer_solutions :
  ∀ (m n : ℤ), (m^3 - n^3 = 2 * m * n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2)) :=
by
  intros m n
  sorry

end integer_solutions_l710_71035


namespace minimum_red_chips_l710_71096

variable (w b r : ℕ)

-- Define the conditions
def condition1 : Prop := b ≥ 3 * w / 4
def condition2 : Prop := b ≤ r / 4
def condition3 : Prop := 60 ≤ w + b ∧ w + b ≤ 80

-- Prove the minimum number of red chips r is 108
theorem minimum_red_chips (H1 : condition1 w b) (H2 : condition2 b r) (H3 : condition3 w b) : r ≥ 108 := 
sorry

end minimum_red_chips_l710_71096


namespace sum_even_integers_eq_930_l710_71043

theorem sum_even_integers_eq_930 :
  let sum_first_30_even := 2 * (30 * (30 + 1) / 2)
  let sum_consecutive_even (n : ℤ) := (n - 8) + (n - 6) + (n - 4) + (n - 2) + n
  ∀ n : ℤ, sum_first_30_even = 930 → sum_consecutive_even n = 930 → n = 190 :=
by
  intros sum_first_30_even sum_consecutive_even n h1 h2
  sorry

end sum_even_integers_eq_930_l710_71043


namespace inequality_solution_set_l710_71046

theorem inequality_solution_set (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) ≤ 1) ↔ (x < 2 ∨ 3 ≤ x) :=
sorry

end inequality_solution_set_l710_71046


namespace difference_of_numbers_l710_71056

theorem difference_of_numbers (a : ℕ) (h : a + (10 * a + 5) = 30000) : (10 * a + 5) - a = 24548 :=
by
  sorry

end difference_of_numbers_l710_71056


namespace roots_of_cubic_8th_power_sum_l710_71048

theorem roots_of_cubic_8th_power_sum :
  ∀ a b c : ℂ, 
  (a + b + c = 0) → 
  (a * b + b * c + c * a = -1) → 
  (a * b * c = -1) → 
  (a^8 + b^8 + c^8 = 10) := 
by
  sorry

end roots_of_cubic_8th_power_sum_l710_71048


namespace correct_calculation_l710_71097

theorem correct_calculation (x : ℕ) (h : 954 - x = 468) : 954 + x = 1440 := by
  sorry

end correct_calculation_l710_71097


namespace rowing_speed_downstream_l710_71058

theorem rowing_speed_downstream (V_u V_s V_d : ℝ) (h1 : V_u = 10) (h2 : V_s = 15)
  (h3 : V_s = (V_u + V_d) / 2) : V_d = 20 := by
  sorry

end rowing_speed_downstream_l710_71058


namespace find_side_b_l710_71039

variables {A B C a b c x : ℝ}

theorem find_side_b 
  (cos_A : ℝ) (cos_C : ℝ) (a : ℝ) (hcosA : cos_A = 4/5) 
  (hcosC : cos_C = 5/13) (ha : a = 1) : 
  b = 21/13 :=
by
  sorry

end find_side_b_l710_71039


namespace max_rectangle_area_l710_71095

theorem max_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 120) : l * w ≤ 900 :=
by 
  sorry

end max_rectangle_area_l710_71095


namespace grandparents_gift_l710_71052

theorem grandparents_gift (june_stickers bonnie_stickers total_stickers : ℕ) (x : ℕ)
  (h₁ : june_stickers = 76)
  (h₂ : bonnie_stickers = 63)
  (h₃ : total_stickers = 189) :
  june_stickers + bonnie_stickers + 2 * x = total_stickers → x = 25 :=
by
  intros
  sorry

end grandparents_gift_l710_71052


namespace sides_of_regular_polygon_l710_71006

theorem sides_of_regular_polygon {n : ℕ} (h₁ : n ≥ 3)
  (h₂ : (n * (n - 3)) / 2 + 6 = 2 * n) : n = 4 :=
sorry

end sides_of_regular_polygon_l710_71006


namespace pos_rel_lines_l710_71026

-- Definition of the lines
def line1 (k : ℝ) (x y : ℝ) : Prop := 2 * x - y + k = 0
def line2 (x y : ℝ) : Prop := 4 * x - 2 * y + 1 = 0

-- Theorem stating the positional relationship between the two lines
theorem pos_rel_lines (k : ℝ) : 
  (∀ x y : ℝ, line1 k x y → line2 x y → 2 * k - 1 = 0) → 
  (∀ x y : ℝ, line1 k x y → ¬ line2 x y → 2 * k - 1 ≠ 0) → 
  (k = 1/2 ∨ k ≠ 1/2) :=
by sorry

end pos_rel_lines_l710_71026


namespace contrapositive_example_l710_71075

theorem contrapositive_example (x : ℝ) : (x = 1 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 1) :=
by
  sorry

end contrapositive_example_l710_71075


namespace no_real_roots_of_ffx_or_ggx_l710_71027

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

end no_real_roots_of_ffx_or_ggx_l710_71027


namespace inequality_true_l710_71023

theorem inequality_true (a b : ℝ) (h : a > b) (x : ℝ) : 
  (a > b) → (x ≥ 0) → (a / ((2^x) + 1) > b / ((2^x) + 1)) :=
by 
  sorry

end inequality_true_l710_71023


namespace full_batches_needed_l710_71001

def students : Nat := 150
def cookies_per_student : Nat := 3
def cookies_per_batch : Nat := 20
def attendance_rate : Rat := 0.70

theorem full_batches_needed : 
  let attendees := (students : Rat) * attendance_rate
  let total_cookies_needed := attendees * (cookies_per_student : Rat)
  let batches_needed := total_cookies_needed / (cookies_per_batch : Rat)
  batches_needed.ceil = 16 :=
by
  sorry

end full_batches_needed_l710_71001


namespace division_proof_l710_71071

-- Defining the given conditions
def total_books := 1200
def first_div := 3
def second_div := 4
def final_books_per_category := 15

-- Calculating the number of books per each category after each division
def books_per_first_category := total_books / first_div
def books_per_second_group := books_per_first_category / second_div

-- Correcting the third division to ensure each part has 15 books
def third_div := books_per_second_group / final_books_per_category
def rounded_parts := (books_per_second_group : ℕ) / final_books_per_category -- Rounded to the nearest integer

-- The number of final parts must be correct to ensure the total final categories
def final_division := first_div * second_div * rounded_parts

-- Required proof statement
theorem division_proof : final_division = 84 ∧ books_per_second_group = final_books_per_category :=
by 
  sorry

end division_proof_l710_71071


namespace Mark_charged_more_l710_71008

theorem Mark_charged_more (K P M : ℕ) 
  (h1 : P = 2 * K) 
  (h2 : P = M / 3)
  (h3 : K + P + M = 153) : M - K = 85 :=
by
  -- proof to be filled in later
  sorry

end Mark_charged_more_l710_71008


namespace increasing_function_range_of_a_l710_71085

variable {f : ℝ → ℝ}

theorem increasing_function_range_of_a (a : ℝ) (h : ∀ x : ℝ, 3 * a * x^2 ≥ 0) : a > 0 :=
sorry

end increasing_function_range_of_a_l710_71085


namespace planes_contain_at_least_three_midpoints_l710_71070

-- Define the cube structure and edge midpoints
structure Cube where
  edges : Fin 12

def midpoints (c : Cube) : Set (Fin 12) := { e | true }

-- Define the total planes considering the constraints
noncomputable def planes : ℕ := 4 + 18 + 56

-- The proof goal
theorem planes_contain_at_least_three_midpoints :
  planes = 81 := by
  sorry

end planes_contain_at_least_three_midpoints_l710_71070


namespace integer_root_of_P_l710_71047

def P (x : ℤ) : ℤ := x^3 - 4 * x^2 - 8 * x + 24 

theorem integer_root_of_P :
  (∃ x : ℤ, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x = 2) :=
sorry

end integer_root_of_P_l710_71047


namespace quadratic_complete_square_l710_71081

theorem quadratic_complete_square (x d e: ℝ) (h : x^2 - 26 * x + 129 = (x + d)^2 + e) : 
d + e = -53 := sorry

end quadratic_complete_square_l710_71081


namespace jack_and_jill_meet_distance_l710_71064

theorem jack_and_jill_meet_distance :
  ∃ t : ℝ, t = 15 / 60 ∧ 14 * t ≤ 4 ∧ 15 * (t - 15 / 60) ≤ 4 ∧
  ( 14 * t - 4 + 18 * (t - 2 / 7) = 15 * (t - 15 / 60) ∨ 15 * (t - 15 / 60) = 4 - 18 * (t - 2 / 7) ) ∧
  4 - 15 * (t - 15 / 60) = 851 / 154 :=
sorry

end jack_and_jill_meet_distance_l710_71064


namespace remainder_492381_div_6_l710_71002

theorem remainder_492381_div_6 : 492381 % 6 = 3 := 
by
  sorry

end remainder_492381_div_6_l710_71002


namespace min_capacity_for_raft_l710_71030

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

end min_capacity_for_raft_l710_71030


namespace necessary_but_not_sufficient_condition_l710_71067
open Locale

variables {l m : Line} {α β : Plane}

def perp (l : Line) (p : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

theorem necessary_but_not_sufficient_condition (h1 : perp l α) (h2 : subset m β) (h3 : perp l m) :
  ∃ (α : Plane) (β : Plane), parallel α β ∧ (perp l α → perp l β) ∧ (parallel α β → perp l β)  :=
sorry

end necessary_but_not_sufficient_condition_l710_71067


namespace find_amplitude_l710_71051

noncomputable def amplitude (a b c d x : ℝ) := a * Real.sin (b * x + c) + d

theorem find_amplitude (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_range : ∀ x, -1 ≤ amplitude a b c d x ∧ amplitude a b c d x ≤ 7) :
  a = 4 :=
by
  sorry

end find_amplitude_l710_71051


namespace athlete_more_stable_l710_71078

theorem athlete_more_stable (var_A var_B : ℝ) 
                                (h1 : var_A = 0.024) 
                                (h2 : var_B = 0.008) 
                                (h3 : var_A > var_B) : 
  var_B < var_A :=
by
  exact h3

end athlete_more_stable_l710_71078


namespace find_inverse_modulo_l710_71080

theorem find_inverse_modulo :
  113 * 113 ≡ 1 [MOD 114] :=
by
  sorry

end find_inverse_modulo_l710_71080


namespace parallel_line_through_intersection_perpendicular_line_through_intersection_l710_71053

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

end parallel_line_through_intersection_perpendicular_line_through_intersection_l710_71053


namespace minimum_value_expression_l710_71025

theorem minimum_value_expression {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x / y + y) * (y / x + x) ≥ 4 :=
sorry

end minimum_value_expression_l710_71025


namespace graph_is_hyperbola_l710_71040

theorem graph_is_hyperbola : 
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 4 ↔ x * y = 2 := 
by
  sorry

end graph_is_hyperbola_l710_71040


namespace donations_received_l710_71074

def profit : Nat := 960
def half_profit: Nat := profit / 2
def goal: Nat := 610
def extra: Nat := 180
def total_needed: Nat := goal + extra
def donations: Nat := total_needed - half_profit

theorem donations_received :
  donations = 310 := by
  -- Proof omitted
  sorry

end donations_received_l710_71074


namespace no_factors_of_p_l710_71091

open Polynomial

noncomputable def p : Polynomial ℝ := X^4 - 4 * X^2 + 16
noncomputable def optionA : Polynomial ℝ := X^2 + 4
noncomputable def optionB : Polynomial ℝ := X + 2
noncomputable def optionC : Polynomial ℝ := X^2 - 4*X + 4
noncomputable def optionD : Polynomial ℝ := X^2 - 4

theorem no_factors_of_p (h : Polynomial ℝ) : h ≠ p / optionA ∧ h ≠ p / optionB ∧ h ≠ p / optionC ∧ h ≠ p / optionD := by
  sorry

end no_factors_of_p_l710_71091


namespace min_value_expression_l710_71079

noncomputable def f (x y : ℝ) : ℝ := 
  (x + 1 / y) * (x + 1 / y - 2023) + (y + 1 / x) * (y + 1 / x - 2023)

theorem min_value_expression : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ f x y = -2048113 :=
sorry

end min_value_expression_l710_71079


namespace john_buys_packs_l710_71069

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l710_71069


namespace sum_gcd_lcm_l710_71022

theorem sum_gcd_lcm (a₁ a₂ : ℕ) (h₁ : a₁ = 36) (h₂ : a₂ = 495) :
  Nat.gcd a₁ a₂ + Nat.lcm a₁ a₂ = 1989 :=
by
  -- Proof can be added here
  sorry

end sum_gcd_lcm_l710_71022


namespace Iris_shorts_l710_71077

theorem Iris_shorts :
  ∃ s, (3 * 10) + s * 6 + (4 * 12) = 90 ∧ s = 2 := 
by
  existsi 2
  sorry

end Iris_shorts_l710_71077


namespace combined_tax_rate_l710_71055

theorem combined_tax_rate
  (Mork_income : ℝ)
  (Mindy_income : ℝ)
  (h1 : Mindy_income = 3 * Mork_income)
  (Mork_tax_rate : ℝ := 0.30)
  (Mindy_tax_rate : ℝ := 0.20) :
  (Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income) * 100 = 22.5 :=
by
  sorry

end combined_tax_rate_l710_71055


namespace highest_price_more_than_lowest_l710_71065

-- Define the highest price and lowest price.
def highest_price : ℕ := 350
def lowest_price : ℕ := 250

-- Define the calculation for the percentage increase.
def percentage_increase (hp lp : ℕ) : ℕ :=
  ((hp - lp) * 100) / lp

-- The theorem to prove the required percentage increase.
theorem highest_price_more_than_lowest : percentage_increase highest_price lowest_price = 40 := 
  by sorry

end highest_price_more_than_lowest_l710_71065


namespace simplified_expression_l710_71007

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 1

theorem simplified_expression :
  (f (g (f 3))) / (g (f (g 3))) = 79 / 37 :=
by  sorry

end simplified_expression_l710_71007


namespace diameter_of_lake_l710_71099

-- Given conditions: the radius of the circular lake
def radius : ℝ := 7

-- The proof problem: proving the diameter of the lake is 14 meters
theorem diameter_of_lake : 2 * radius = 14 :=
by
  sorry

end diameter_of_lake_l710_71099


namespace greatest_possible_x_lcm_l710_71011

theorem greatest_possible_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105): x = 105 := 
sorry

end greatest_possible_x_lcm_l710_71011


namespace solution_set_of_inverse_inequality_l710_71032

open Function

variable {f : ℝ → ℝ}

theorem solution_set_of_inverse_inequality 
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_A : f (-2) = 2)
  (h_B : f 2 = -2)
  : { x : ℝ | |(invFun f (x + 1))| ≤ 2 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
sorry

end solution_set_of_inverse_inequality_l710_71032


namespace sara_has_green_marbles_l710_71044

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

end sara_has_green_marbles_l710_71044


namespace triangle_area_l710_71061

theorem triangle_area (base height : ℝ) (h_base : base = 4.5) (h_height : height = 6) :
  (base * height) / 2 = 13.5 := 
by
  rw [h_base, h_height]
  norm_num

-- sorry
-- The later use of sorry statement is commented out because the proof itself has been provided in by block.

end triangle_area_l710_71061


namespace unique_perpendicular_line_through_point_l710_71082

-- Definitions of the geometric entities and their relationships
structure Point := (x : ℝ) (y : ℝ)

structure Line := (m : ℝ) (b : ℝ)

-- A function to check if a point lies on a given line
def point_on_line (P : Point) (l : Line) : Prop := P.y = l.m * P.x + l.b

-- A function to represent that a line is perpendicular to another line at a given point
def perpendicular_lines_at_point (P : Point) (l1 l2 : Line) : Prop :=
  l1.m = -(1 / l2.m) ∧ point_on_line P l1 ∧ point_on_line P l2

-- The statement to be proved
theorem unique_perpendicular_line_through_point (P : Point) (l : Line) (h : point_on_line P l) :
  ∃! l' : Line, perpendicular_lines_at_point P l' l :=
by
  sorry

end unique_perpendicular_line_through_point_l710_71082


namespace find_principal_l710_71073

theorem find_principal (R P : ℝ) (h₁ : (P * R * 10) / 100 = P * R * 0.1)
  (h₂ : (P * (R + 3) * 10) / 100 = P * (R + 3) * 0.1)
  (h₃ : P * 0.1 * (R + 3) - P * 0.1 * R = 300) : 
  P = 1000 := 
sorry

end find_principal_l710_71073


namespace floor_ceil_inequality_l710_71019

theorem floor_ceil_inequality 
  (a b c : ℝ)
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := 
by
  sorry 

end floor_ceil_inequality_l710_71019


namespace no_real_solution_range_of_a_l710_71050

theorem no_real_solution_range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ≤ 3 :=
by
  sorry  -- Proof skipped

end no_real_solution_range_of_a_l710_71050


namespace find_complex_z_modulus_of_z_l710_71089

open Complex

theorem find_complex_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    z = -1 + 3 * I := by 
  sorry

theorem modulus_of_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    Complex.abs (z / (1 - I)) = Real.sqrt 5 := by 
  sorry

end find_complex_z_modulus_of_z_l710_71089


namespace total_turtles_l710_71009

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l710_71009


namespace problem_solution_l710_71020

open Set

theorem problem_solution (x : ℝ) :
  (x ∈ {y : ℝ | (2 / (y + 2) + 4 / (y + 8) ≥ 1)} ↔ x ∈ Ioo (-8 : ℝ) (-2 : ℝ)) :=
sorry

end problem_solution_l710_71020


namespace convert_scientific_notation_l710_71098

theorem convert_scientific_notation (a : ℝ) (b : ℤ) (h : a = 6.03 ∧ b = 5) : a * 10^b = 603000 := by
  cases h with
  | intro ha hb =>
    rw [ha, hb]
    sorry

end convert_scientific_notation_l710_71098


namespace simplify_expression_l710_71092

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by
  sorry

end simplify_expression_l710_71092
