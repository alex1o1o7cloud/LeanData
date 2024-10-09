import Mathlib

namespace voucher_placement_l1752_175270

/-- A company wants to popularize the sweets they market by hiding prize vouchers in some of the boxes.
The management believes the promotion is effective and the cost is bearable if a customer who buys 10 boxes has approximately a 50% chance of finding at least one voucher.
We aim to determine how often vouchers should be placed in the boxes to meet this requirement. -/
theorem voucher_placement (n : ℕ) (h_positive : n > 0) :
  (1 - (1 - 1/n)^10) ≥ 1/2 → n ≤ 15 :=
sorry

end voucher_placement_l1752_175270


namespace pentagon_stack_l1752_175216

/-- Given a stack of identical regular pentagons with vertices numbered from 1 to 5, rotated and flipped
such that the sums of numbers at each vertex are the same, the number of pentagons in the stacks can be
any natural number except 1 and 3. -/
theorem pentagon_stack (n : ℕ) (h0 : identical_pentagons_with_vertices_1_to_5)
  (h1 : pentagons_can_be_rotated_and_flipped)
  (h2 : stacked_vertex_to_vertex)
  (h3 : sums_at_each_vertex_are_equal) :
  ∃ k : ℕ, k = n ∧ n ≠ 1 ∧ n ≠ 3 :=
sorry

end pentagon_stack_l1752_175216


namespace egg_problem_l1752_175244

theorem egg_problem :
  ∃ (N F E : ℕ), N + F + E = 100 ∧ 5 * N + F + E / 2 = 100 ∧ (N = F ∨ N = E ∨ F = E) ∧ N = 10 ∧ F = 10 ∧ E = 80 :=
by
  sorry

end egg_problem_l1752_175244


namespace find_a_l1752_175287

noncomputable def slope1 (a : ℝ) : ℝ := -3 / (3^a - 3)
noncomputable def slope2 : ℝ := 2

theorem find_a (a : ℝ) (h : slope1 a * slope2 = -1) : a = 2 :=
sorry

end find_a_l1752_175287


namespace simplify_expression_l1752_175240

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 5 + 2) + 2 / (Real.sqrt 7 - 2))) = 
  (6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35) :=
  sorry

end simplify_expression_l1752_175240


namespace breadth_decrease_percentage_l1752_175291

theorem breadth_decrease_percentage
  (L B : ℝ)
  (hLpos : L > 0)
  (hBpos : B > 0)
  (harea_change : (1.15 * L) * (B - p/100 * B) = 1.035 * (L * B)) :
  p = 10 := 
sorry

end breadth_decrease_percentage_l1752_175291


namespace sum_of_three_consecutive_natural_numbers_not_prime_l1752_175248

theorem sum_of_three_consecutive_natural_numbers_not_prime (n : ℕ) : 
  ¬ Prime (n + (n+1) + (n+2)) := by
  sorry

end sum_of_three_consecutive_natural_numbers_not_prime_l1752_175248


namespace isosceles_triangle_perimeter_l1752_175258

theorem isosceles_triangle_perimeter
  (a b c : ℝ )
  (ha : a = 20)
  (hb : b = 20)
  (hc : c = (2/5) * 20)
  (triangle_ineq1 : a ≤ b + c)
  (triangle_ineq2 : b ≤ a + c)
  (triangle_ineq3 : c ≤ a + b) :
  a + b + c = 48 := by
  sorry

end isosceles_triangle_perimeter_l1752_175258


namespace investment_ratio_l1752_175260

theorem investment_ratio (total_profit b_profit : ℝ) (a c b : ℝ) :
  total_profit = 150000 ∧ b_profit = 75000 ∧ a / c = 2 ∧ a + b + c = total_profit →
  a / b = 2 / 3 :=
by
  sorry

end investment_ratio_l1752_175260


namespace chip_sheets_per_pack_l1752_175245

noncomputable def sheets_per_pack (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) 
                                  (weeks : ℕ) (packs : ℕ) : ℕ :=
(pages_per_day * days_per_week * classes * weeks) / packs

theorem chip_sheets_per_pack :
  sheets_per_pack 2 5 5 6 3 = 100 :=
sorry

end chip_sheets_per_pack_l1752_175245


namespace factorize_expression_l1752_175275

theorem factorize_expression (x y : ℝ) : x^3 * y - 4 * x * y = x * y * (x - 2) * (x + 2) :=
sorry

end factorize_expression_l1752_175275


namespace probability_both_hit_l1752_175218

-- Define the probabilities of hitting the target for shooters A and B.
def prob_A_hits : ℝ := 0.7
def prob_B_hits : ℝ := 0.8

-- Define the independence condition (not needed as a direct definition but implicitly acknowledges independence).
axiom A_and_B_independent : true

-- The statement we want to prove: the probability that both shooters hit the target.
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.56 :=
by
  -- Placeholder for proof
  sorry

end probability_both_hit_l1752_175218


namespace worker_b_days_l1752_175230

variables (W_A W_B W : ℝ)
variables (h1 : W_A = 2 * W_B)
variables (h2 : (W_A + W_B) * 10 = W)
variables (h3 : W = 30 * W_B)

theorem worker_b_days : ∃ days : ℝ, days = 30 :=
by
  sorry

end worker_b_days_l1752_175230


namespace syllogism_correct_l1752_175221

-- Define that natural numbers are integers
axiom nat_is_int : ∀ (n : ℕ), ∃ (m : ℤ), m = n

-- Define that 4 is a natural number
axiom four_is_nat : ∃ (n : ℕ), n = 4

-- The syllogism's conclusion: 4 is an integer
theorem syllogism_correct : ∃ (m : ℤ), m = 4 :=
by
  have h1 := nat_is_int 4
  have h2 := four_is_nat
  exact h1

end syllogism_correct_l1752_175221


namespace coordinates_of_S_l1752_175295

variable (P Q R S : (ℝ × ℝ))
variable (hp : P = (3, -2))
variable (hq : Q = (3, 1))
variable (hr : R = (7, 1))
variable (h : Rectangle P Q R S)

def Rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  let (xS, yS) := S
  (xP = xQ ∧ yR = yS) ∧ (xS = xR ∧ yP = yQ) 

theorem coordinates_of_S : S = (7, -2) := by
  sorry

end coordinates_of_S_l1752_175295


namespace original_expenditure_mess_l1752_175264

theorem original_expenditure_mess : 
  ∀ (x : ℝ), 
  35 * x + 42 = 42 * (x - 1) + 35 * x → 
  35 * 12 = 420 :=
by
  intro x
  intro h
  sorry

end original_expenditure_mess_l1752_175264


namespace circle_has_greatest_symmetry_l1752_175210

-- Definitions based on the conditions
def lines_of_symmetry (figure : String) : ℕ∞ := 
  match figure with
  | "regular pentagon" => 5
  | "isosceles triangle" => 1
  | "circle" => ⊤  -- Using the symbol ⊤ to represent infinity in Lean.
  | "rectangle" => 2
  | "parallelogram" => 0
  | _ => 0          -- default case

theorem circle_has_greatest_symmetry :
  ∃ fig, fig = "circle" ∧ ∀ other_fig, lines_of_symmetry fig ≥ lines_of_symmetry other_fig := 
by
  sorry

end circle_has_greatest_symmetry_l1752_175210


namespace initial_amount_solution_l1752_175294

noncomputable def initialAmount (P : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then P else (1 + 1/8) * initialAmount P (n - 1)

theorem initial_amount_solution (P : ℝ) (h₁ : initialAmount P 2 = 2025) : P = 1600 :=
  sorry

end initial_amount_solution_l1752_175294


namespace sum_sin_double_angles_eq_l1752_175257

theorem sum_sin_double_angles_eq (
  α β γ : ℝ
) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

end sum_sin_double_angles_eq_l1752_175257


namespace unique_nat_number_sum_preceding_eq_self_l1752_175271

theorem unique_nat_number_sum_preceding_eq_self :
  ∃! (n : ℕ), (n * (n - 1)) / 2 = n :=
sorry

end unique_nat_number_sum_preceding_eq_self_l1752_175271


namespace remaining_money_l1752_175278

-- Definitions
def cost_per_app : ℕ := 4
def num_apps : ℕ := 15
def total_money : ℕ := 66

-- Theorem
theorem remaining_money : total_money - (num_apps * cost_per_app) = 6 := by
  sorry

end remaining_money_l1752_175278


namespace solve_divisor_problem_l1752_175236

def divisor_problem : Prop :=
  ∃ D : ℕ, 12401 = (D * 76) + 13 ∧ D = 163

theorem solve_divisor_problem : divisor_problem :=
sorry

end solve_divisor_problem_l1752_175236


namespace parallel_lines_slope_l1752_175256

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope_l1752_175256


namespace construct_triangle_num_of_solutions_l1752_175276

theorem construct_triangle_num_of_solutions
  (r : ℝ) -- Circumradius
  (beta_gamma_diff : ℝ) -- Angle difference \beta - \gamma
  (KA1 : ℝ) -- Segment K A_1
  (KA1_lt_r : KA1 < r) -- Segment K A1 should be less than the circumradius
  (delta : ℝ := beta_gamma_diff) : 1 ≤ num_solutions ∧ num_solutions ≤ 2 :=
sorry

end construct_triangle_num_of_solutions_l1752_175276


namespace parabola_focus_l1752_175220

theorem parabola_focus (p : ℝ) (hp : 0 < p) (h : ∀ y x : ℝ, y^2 = 2 * p * x → (x = 2 ∧ y = 0)) : p = 4 :=
sorry

end parabola_focus_l1752_175220


namespace curve_transformation_l1752_175253

variable (x y x0 y0 : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def C (x0 y0 : ℝ) : Prop := (x0 - y0)^2 + y0^2 = 1

def transform (x0 y0 : ℝ) : ℝ × ℝ :=
  let x := 2 * x0 - 2 * y0
  let y := y0
  (x, y)

def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem curve_transformation :
  ∀ x0 y0, C x0 y0 → C' (2 * x0 - 2 * y0) y0 := sorry

end curve_transformation_l1752_175253


namespace revenue_difference_l1752_175223

def original_revenue : ℕ := 10000

def vasya_revenue (X : ℕ) : ℕ :=
  2 * (original_revenue / X) * (4 * X / 5)

def kolya_revenue (X : ℕ) : ℕ :=
  (original_revenue / X) * (8 * X / 3)

theorem revenue_difference (X : ℕ) (hX : X > 0) : vasya_revenue X = 16000 ∧ kolya_revenue X = 13333 ∧ vasya_revenue X - original_revenue = 6000 := 
by
  sorry

end revenue_difference_l1752_175223


namespace brooke_added_balloons_l1752_175241

-- Definitions stemming from the conditions
def initial_balloons_brooke : Nat := 12
def added_balloons_brooke (x : Nat) : Nat := x
def initial_balloons_tracy : Nat := 6
def added_balloons_tracy : Nat := 24
def total_balloons_tracy : Nat := initial_balloons_tracy + added_balloons_tracy
def final_balloons_tracy : Nat := total_balloons_tracy / 2
def total_balloons (x : Nat) : Nat := initial_balloons_brooke + added_balloons_brooke x + final_balloons_tracy

-- Mathematical proof problem
theorem brooke_added_balloons (x : Nat) :
  total_balloons x = 35 → x = 8 := by
  sorry

end brooke_added_balloons_l1752_175241


namespace round_trip_time_l1752_175217

variable (boat_speed standing_water_speed stream_speed distance : ℕ)

theorem round_trip_time (boat_speed := 9) (stream_speed := 6) (distance := 170) : 
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed)) = 68 := by 
  sorry

end round_trip_time_l1752_175217


namespace distinguishable_large_equilateral_triangles_l1752_175226

-- Definitions based on conditions.
def num_colors : ℕ := 8

def same_color_corners : ℕ := num_colors
def two_same_one_diff_colors : ℕ := num_colors * (num_colors - 1)
def all_diff_colors : ℕ := (num_colors * (num_colors - 1) * (num_colors - 2)) / 6

def corner_configurations : ℕ := same_color_corners + two_same_one_diff_colors + all_diff_colors
def triangle_between_center_and_corner : ℕ := num_colors
def center_triangle : ℕ := num_colors

def total_distinguishable_triangles : ℕ := corner_configurations * triangle_between_center_and_corner * center_triangle

theorem distinguishable_large_equilateral_triangles : total_distinguishable_triangles = 7680 :=
by
  sorry

end distinguishable_large_equilateral_triangles_l1752_175226


namespace smaller_interior_angle_of_parallelogram_l1752_175242

theorem smaller_interior_angle_of_parallelogram (x : ℝ) 
  (h1 : ∃ l, l = x + 90 ∧ x + l = 180) :
  x = 45 :=
by
  obtain ⟨l, hl1, hl2⟩ := h1
  simp only [hl1] at hl2
  linarith

end smaller_interior_angle_of_parallelogram_l1752_175242


namespace new_avg_weight_l1752_175288

-- Define the weights of individuals
variables (A B C D E : ℕ)
-- Conditions
axiom avg_ABC : (A + B + C) / 3 = 84
axiom avg_ABCD : (A + B + C + D) / 4 = 80
axiom E_def : E = D + 8
axiom A_80 : A = 80

theorem new_avg_weight (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 8) 
  (h4 : A = 80) 
  : (B + C + D + E) / 4 = 79 := 
by
  sorry

end new_avg_weight_l1752_175288


namespace negation_equiv_l1752_175235
variable (x : ℝ)

theorem negation_equiv :
  (¬ ∃ x : ℝ, x^2 + 1 > 3 * x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3 * x) :=
by 
  sorry

end negation_equiv_l1752_175235


namespace wheel_revolutions_l1752_175279

theorem wheel_revolutions (r_course r_wheel : ℝ) (laps : ℕ) (C_course C_wheel : ℝ) (d_total : ℝ) :
  r_course = 7 →
  r_wheel = 5 →
  laps = 15 →
  C_course = 2 * Real.pi * r_course →
  d_total = laps * C_course →
  C_wheel = 2 * Real.pi * r_wheel →
  ((d_total) / (C_wheel)) = 21 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end wheel_revolutions_l1752_175279


namespace binomial_coefficient_12_10_l1752_175268

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end binomial_coefficient_12_10_l1752_175268


namespace jane_earnings_two_weeks_l1752_175255

def num_chickens : ℕ := 10
def num_eggs_per_chicken_per_week : ℕ := 6
def dollars_per_dozen : ℕ := 2
def dozens_in_12_eggs : ℕ := 12

theorem jane_earnings_two_weeks :
  (num_chickens * num_eggs_per_chicken_per_week * 2 / dozens_in_12_eggs * dollars_per_dozen) = 20 := by
  sorry

end jane_earnings_two_weeks_l1752_175255


namespace triangle_side_length_l1752_175259

theorem triangle_side_length (a : ℝ) (B : ℝ) (C : ℝ) (c : ℝ) 
  (h₀ : a = 10) (h₁ : B = 60) (h₂ : C = 45) : 
  c = 10 * (Real.sqrt 3 - 1) :=
sorry

end triangle_side_length_l1752_175259


namespace units_digit_a_2017_l1752_175290

noncomputable def a_n (n : ℕ) : ℝ :=
  (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

theorem units_digit_a_2017 : (Nat.floor (a_n 2017)) % 10 = 2 :=
  sorry

end units_digit_a_2017_l1752_175290


namespace time_to_fill_tank_l1752_175289

theorem time_to_fill_tank (T : ℝ) :
  (1 / 2 * T) + ((1 / 2 * T) / 4) = 10 → T = 16 :=
by { sorry }

end time_to_fill_tank_l1752_175289


namespace solve_for_x_l1752_175263

theorem solve_for_x (x : ℝ) (h : 2 * (1/x + 3/x / 6/x) - 1/x = 1.5) : x = 2 := 
by 
  sorry

end solve_for_x_l1752_175263


namespace age_difference_l1752_175204

theorem age_difference (J P : ℕ) 
  (h1 : P = 16 - 10) 
  (h2 : P = (1 / 3) * J) : 
  (J + 10) - 16 = 12 := 
by 
  sorry

end age_difference_l1752_175204


namespace total_hamburgers_sold_is_63_l1752_175262

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end total_hamburgers_sold_is_63_l1752_175262


namespace probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l1752_175252

-- Problem 1
theorem probability_meeting_twin (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (p + 1) = (2 * p) / (p + 1) :=
by
  sorry

-- Problem 2
theorem probability_twin_in_family (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (2 * p + (1 - p) ^ 2) = (2 * p) / (2 * p + (1 - p) ^ 2) :=
by
  sorry

-- Problem 3
theorem expected_twin_pairs (N : ℕ) (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  N * p / (p + 1) = N * p / (p + 1) :=
by
  sorry

end probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l1752_175252


namespace amy_soups_total_l1752_175247

def total_soups (chicken_soups tomato_soups : ℕ) : ℕ :=
  chicken_soups + tomato_soups

theorem amy_soups_total : total_soups 6 3 = 9 :=
by
  -- insert the proof here
  sorry

end amy_soups_total_l1752_175247


namespace bear_hunting_l1752_175234

theorem bear_hunting
    (mother_meat_req : ℕ) (cub_meat_req : ℕ) (num_cubs : ℕ) (num_animals_daily : ℕ)
    (weekly_meat_req : mother_meat_req = 210)
    (weekly_meat_per_cub : cub_meat_req = 35)
    (number_of_cubs : num_cubs = 4)
    (animals_hunted_daily : num_animals_daily = 10)
    (total_weekly_meat : mother_meat_req + num_cubs * cub_meat_req = 350) :
    ∃ w : ℕ, (w * num_animals_daily * 7 = 350) ∧ w = 5 :=
by
  sorry

end bear_hunting_l1752_175234


namespace q_value_l1752_175233

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l1752_175233


namespace cost_of_pears_l1752_175232

theorem cost_of_pears (P : ℕ)
  (apples_cost : ℕ := 40)
  (dozens : ℕ := 14)
  (total_cost : ℕ := 1260)
  (h_p : dozens * P + dozens * apples_cost = total_cost) :
  P = 50 :=
by
  sorry

end cost_of_pears_l1752_175232


namespace option_D_correct_l1752_175238

theorem option_D_correct (a b : ℝ) : -a * b + 3 * b * a = 2 * a * b :=
by sorry

end option_D_correct_l1752_175238


namespace min_value_m_plus_n_l1752_175205

theorem min_value_m_plus_n (m n : ℕ) (h : 108 * m = n^3) (hm : 0 < m) (hn : 0 < n) : m + n = 8 :=
sorry

end min_value_m_plus_n_l1752_175205


namespace symmetric_point_correct_l1752_175280

-- Define the coordinates of point A
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D :=
  { x := -3
    y := -4
    z := 5 }

-- Define the symmetry function with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { p with y := -p.y }

-- The expected coordinates of the point symmetric to A with respect to the plane xOz
def D_expected : Point3D :=
  { x := -3
    y := 4
    z := 5 }

-- Theorem stating that the symmetric point of A with respect to the plane xOz is D_expected
theorem symmetric_point_correct :
  symmetric_xOz A = D_expected := 
by 
  sorry

end symmetric_point_correct_l1752_175280


namespace ship_passengers_percentage_l1752_175214

variables (P R : ℝ)

-- Conditions
def condition1 : Prop := (0.20 * P) = (0.60 * R)

-- Target
def target : Prop := R / P = 1 / 3

theorem ship_passengers_percentage
  (h1 : condition1 P R) :
  target P R :=
by
  sorry

end ship_passengers_percentage_l1752_175214


namespace first_digit_base_8_of_725_is_1_l1752_175203

-- Define conditions
def decimal_val := 725

-- Helper function to get the largest power of 8 less than the decimal value
def largest_power_base_eight (n : ℕ) : ℕ :=
  if 8^3 <= n then 8^3 else if 8^2 <= n then 8^2 else if 8^1 <= n then 8^1 else if 8^0 <= n then 8^0 else 0

-- The target theorem
theorem first_digit_base_8_of_725_is_1 : 
  (725 / largest_power_base_eight 725) = 1 :=
by 
  -- Proof goes here
  sorry

end first_digit_base_8_of_725_is_1_l1752_175203


namespace units_digit_7_pow_5_l1752_175283

theorem units_digit_7_pow_5 : (7 ^ 5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l1752_175283


namespace Sara_lunch_bill_l1752_175250

theorem Sara_lunch_bill :
  let hotdog := 5.36
  let salad := 5.10
  let drink := 2.50
  let side_item := 3.75
  hotdog + salad + drink + side_item = 16.71 :=
by
  sorry

end Sara_lunch_bill_l1752_175250


namespace three_pizzas_needed_l1752_175261

noncomputable def masha_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "sausage" ∉ p

noncomputable def vanya_pizza (p : Set String) : Prop :=
  "mushrooms" ∈ p

noncomputable def dasha_pizza (p : Set String) : Prop :=
  "tomatoes" ∉ p

noncomputable def nikita_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "mushrooms" ∉ p

noncomputable def igor_pizza (p : Set String) : Prop :=
  "mushrooms" ∉ p ∧ "sausage" ∈ p

theorem three_pizzas_needed (p1 p2 p3 : Set String) :
  (∃ p1, masha_pizza p1 ∧ vanya_pizza p1 ∧ dasha_pizza p1 ∧ nikita_pizza p1 ∧ igor_pizza p1) →
  (∃ p2, masha_pizza p2 ∧ vanya_pizza p2 ∧ dasha_pizza p2 ∧ nikita_pizza p2 ∧ igor_pizza p2) →
  (∃ p3, masha_pizza p3 ∧ vanya_pizza p3 ∧ dasha_pizza p3 ∧ nikita_pizza p3 ∧ igor_pizza p3) →
  ∀ p, ¬ ((masha_pizza p ∨ dasha_pizza p) ∧ vanya_pizza p ∧ (nikita_pizza p ∨ igor_pizza p)) :=
sorry

end three_pizzas_needed_l1752_175261


namespace solve_system_eqs_l1752_175222

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs_l1752_175222


namespace boys_to_girls_ratio_l1752_175207

theorem boys_to_girls_ratio (boys girls : ℕ) (h_boys : boys = 1500) (h_girls : girls = 1200) : 
  (boys / Nat.gcd boys girls) = 5 ∧ (girls / Nat.gcd boys girls) = 4 := 
by 
  sorry

end boys_to_girls_ratio_l1752_175207


namespace at_least_half_sectors_occupied_l1752_175274

theorem at_least_half_sectors_occupied (n : ℕ) (chips : Finset (Fin n.succ)) 
(h_chips_count: chips.card = n + 1) :
  ∃ (steps : ℕ), ∀ (t : ℕ), t ≥ steps → (∃ sector_occupied : Finset (Fin n), sector_occupied.card ≥ n / 2) :=
sorry

end at_least_half_sectors_occupied_l1752_175274


namespace acute_angle_implies_x_range_l1752_175267

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end acute_angle_implies_x_range_l1752_175267


namespace jean_jail_time_l1752_175228

def num_arson := 3
def num_burglary := 2
def ratio_larceny_to_burglary := 6
def sentence_arson := 36
def sentence_burglary := 18
def sentence_larceny := sentence_burglary / 3

def total_arson_time := num_arson * sentence_arson
def total_burglary_time := num_burglary * sentence_burglary
def num_larceny := num_burglary * ratio_larceny_to_burglary
def total_larceny_time := num_larceny * sentence_larceny

def total_jail_time := total_arson_time + total_burglary_time + total_larceny_time

theorem jean_jail_time : total_jail_time = 216 := by
  sorry

end jean_jail_time_l1752_175228


namespace vector_parallel_l1752_175231

theorem vector_parallel (x : ℝ) : ∃ x, (1, x) = k * (-2, 3) → x = -3 / 2 :=
by 
  sorry

end vector_parallel_l1752_175231


namespace problem1_problem2_l1752_175208

-- Definitions and conditions:
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := (x^2 - 2 * x + 1 - m^2 ≤ 0) ∧ (m > 0)

-- Question (1) statement: Prove that if p is a sufficient condition for q, then m ≥ 4
theorem problem1 (p_implies_q : ∀ x : ℝ, p x → q x m) : m ≥ 4 := sorry

-- Question (2) statement: Prove that if m = 5 and p ∨ q is true but p ∧ q is false,
-- then the range of x is [-4, -1) ∪ (5, 6]
theorem problem2 (m_eq : m = 5) (p_or_q : ∃ x : ℝ, p x ∨ q x m) (p_and_not_q : ¬ (∃ x : ℝ, p x ∧ q x m)) :
  ∃ x : ℝ, (x < -1 ∧ -4 ≤ x) ∨ (5 < x ∧ x ≤ 6) := sorry

end problem1_problem2_l1752_175208


namespace action_figures_per_shelf_l1752_175281

theorem action_figures_per_shelf (total_figures shelves : ℕ) (h1 : total_figures = 27) (h2 : shelves = 3) :
  (total_figures / shelves = 9) :=
by
  sorry

end action_figures_per_shelf_l1752_175281


namespace find_xyz_area_proof_l1752_175211

-- Conditions given in the problem
variable (x y z : ℝ)
-- Side lengths derived from condition of inscribed circle
def conditions :=
  (x + y = 5) ∧
  (x + z = 6) ∧
  (y + z = 8)

-- The proof problem: Show the relationships between x, y, and z given the side lengths
theorem find_xyz_area_proof (h : conditions x y z) :
  (z - y = 1) ∧ (z - x = 3) ∧ (z = 4.5) ∧ (x = 1.5) ∧ (y = 3.5) :=
by
  sorry

end find_xyz_area_proof_l1752_175211


namespace calc_dz_calc_d2z_calc_d3z_l1752_175224

variables (x y dx dy : ℝ)

def z : ℝ := x^5 * y^3

-- Define the first differential dz
def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

-- Define the second differential d2z
def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

-- Define the third differential d3z
def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem calc_dz : (dz x y dx dy) = (5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) := 
by sorry

theorem calc_d2z : (d2z x y dx dy) = (20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) :=
by sorry

theorem calc_d3z : (d3z x y dx dy) = (60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end calc_dz_calc_d2z_calc_d3z_l1752_175224


namespace age_difference_l1752_175219

theorem age_difference (Rona Rachel Collete : ℕ) (h1 : Rachel = 2 * Rona) (h2 : Collete = Rona / 2) (h3 : Rona = 8) : Rachel - Collete = 12 :=
by
  sorry

end age_difference_l1752_175219


namespace population_after_4_years_l1752_175202

theorem population_after_4_years 
  (initial_population : ℕ) 
  (new_people : ℕ) 
  (people_moved_out : ℕ) 
  (years : ℕ) 
  (final_population : ℕ) :
  initial_population = 780 →
  new_people = 100 →
  people_moved_out = 400 →
  years = 4 →
  final_population = initial_population + new_people - people_moved_out →
  final_population / 2 / 2 / 2 / 2 = 30 :=
by
  sorry

end population_after_4_years_l1752_175202


namespace f_eq_32x5_l1752_175212

def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

theorem f_eq_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  -- the proof proceeds here
  sorry

end f_eq_32x5_l1752_175212


namespace value_of_a_minus_b_l1752_175298

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 :=
by
  sorry

end value_of_a_minus_b_l1752_175298


namespace line_intersects_circle_two_points_find_value_of_m_l1752_175286

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem line_intersects_circle_two_points (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ),
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  x1 ≠ x2 ∨ y1 ≠ y2 := sorry

theorem find_value_of_m (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ), 
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  dist (x1, y1) (x2, y2) = sqrt 17 → 
  m = sqrt 3 ∨ m = -sqrt 3 := sorry

end line_intersects_circle_two_points_find_value_of_m_l1752_175286


namespace point_not_in_third_quadrant_l1752_175249

theorem point_not_in_third_quadrant (A : ℝ × ℝ) (h : A.snd = -A.fst + 8) : ¬ (A.fst < 0 ∧ A.snd < 0) :=
sorry

end point_not_in_third_quadrant_l1752_175249


namespace find_a2_plus_b2_l1752_175200

theorem find_a2_plus_b2
  (a b : ℝ)
  (h1 : a^3 - 3 * a * b^2 = 39)
  (h2 : b^3 - 3 * a^2 * b = 26) :
  a^2 + b^2 = 13 :=
sorry

end find_a2_plus_b2_l1752_175200


namespace range_of_a_l1752_175215

theorem range_of_a (a : ℝ) :
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → (a ≤ -1) :=
by 
  sorry

end range_of_a_l1752_175215


namespace baseball_fans_count_l1752_175229

theorem baseball_fans_count
  (Y M R : ℕ) 
  (h1 : Y = (3 * M) / 2)
  (h2 : R = (5 * M) / 4)
  (hM : M = 104) :
  Y + M + R = 390 :=
by
  sorry 

end baseball_fans_count_l1752_175229


namespace trigonometric_identity_l1752_175277

open Real

theorem trigonometric_identity (α β : ℝ) :
  sin (2 * α) ^ 2 + sin β ^ 2 + cos (2 * α + β) * cos (2 * α - β) = 1 :=
sorry

end trigonometric_identity_l1752_175277


namespace chocolates_problem_l1752_175246

-- Let denote the quantities as follows:
-- C: number of caramels
-- N: number of nougats
-- T: number of truffles
-- P: number of peanut clusters

def C_nougats_truffles_peanutclusters (C N T P : ℕ) :=
  N = 2 * C ∧
  T = C + 6 ∧
  C + N + T + P = 50 ∧
  P = 32

theorem chocolates_problem (C N T P : ℕ) :
  C_nougats_truffles_peanutclusters C N T P → C = 3 :=
by
  intros h
  have hN := h.1
  have hT := h.2.1
  have hSum := h.2.2.1
  have hP := h.2.2.2
  sorry

end chocolates_problem_l1752_175246


namespace monic_poly_7_r_8_l1752_175293

theorem monic_poly_7_r_8 :
  ∃ (r : ℕ → ℕ), (r 1 = 1) ∧ (r 2 = 2) ∧ (r 3 = 3) ∧ (r 4 = 4) ∧ (r 5 = 5) ∧ (r 6 = 6) ∧ (r 7 = 7) ∧ (∀ (n : ℕ), 8 < n → r n = n) ∧ r 8 = 5048 :=
sorry

end monic_poly_7_r_8_l1752_175293


namespace average_pregnancies_per_kettle_l1752_175239

-- Define the given conditions
def num_kettles : ℕ := 6
def babies_per_pregnancy : ℕ := 4
def survival_rate : ℝ := 0.75
def total_expected_babies : ℕ := 270

-- Calculate surviving babies per pregnancy
def surviving_babies_per_pregnancy : ℝ := babies_per_pregnancy * survival_rate

-- Prove that the average number of pregnancies per kettle is 15
theorem average_pregnancies_per_kettle : ∃ P : ℝ, num_kettles * P * surviving_babies_per_pregnancy = total_expected_babies ∧ P = 15 :=
by
  sorry

end average_pregnancies_per_kettle_l1752_175239


namespace roots_of_unity_real_root_l1752_175251

theorem roots_of_unity_real_root (n : ℕ) (h_even : n % 2 = 0) : ∃ z : ℝ, z ≠ 1 ∧ z^n = 1 :=
by
  sorry

end roots_of_unity_real_root_l1752_175251


namespace num_integers_satisfying_abs_leq_bound_l1752_175254

theorem num_integers_satisfying_abs_leq_bound : ∃ n : ℕ, n = 19 ∧ ∀ x : ℤ, |x| ≤ 3 * Real.sqrt 10 → (x ≥ -9 ∧ x ≤ 9) := by
  sorry

end num_integers_satisfying_abs_leq_bound_l1752_175254


namespace candy_amount_in_peanut_butter_jar_l1752_175206

-- Definitions of the candy amounts in each jar
def banana_jar := 43
def grape_jar := banana_jar + 5
def peanut_butter_jar := 4 * grape_jar
def coconut_jar := (3 / 2) * banana_jar

-- The statement we need to prove
theorem candy_amount_in_peanut_butter_jar : peanut_butter_jar = 192 := by
  sorry

end candy_amount_in_peanut_butter_jar_l1752_175206


namespace complex_division_example_l1752_175265

theorem complex_division_example : (2 - (1 : ℂ) * Complex.I) / (1 - (1 : ℂ) * Complex.I) = (3 / 2) + (1 / 2) * Complex.I :=
by
  sorry

end complex_division_example_l1752_175265


namespace janessa_initial_cards_l1752_175237

theorem janessa_initial_cards (X : ℕ)  :
  (X + 45 = 49) →
  X = 4 :=
by
  intro h
  sorry

end janessa_initial_cards_l1752_175237


namespace ratio_equiv_solve_x_l1752_175297

theorem ratio_equiv_solve_x (x : ℕ) (h : 3 / 12 = 3 / x) : x = 12 :=
sorry

end ratio_equiv_solve_x_l1752_175297


namespace max_underwear_pairs_l1752_175243

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end max_underwear_pairs_l1752_175243


namespace difference_of_numbers_l1752_175273

theorem difference_of_numbers (x y : ℝ) (h₁ : x + y = 25) (h₂ : x * y = 144) : |x - y| = 7 :=
sorry

end difference_of_numbers_l1752_175273


namespace superhero_speed_conversion_l1752_175209

theorem superhero_speed_conversion
    (speed_km_per_min : ℕ)
    (conversion_factor : ℝ)
    (minutes_in_hour : ℕ)
    (H1 : speed_km_per_min = 1000)
    (H2 : conversion_factor = 0.6)
    (H3 : minutes_in_hour = 60) :
    (speed_km_per_min * conversion_factor * minutes_in_hour = 36000) :=
by
    sorry

end superhero_speed_conversion_l1752_175209


namespace Elise_savings_l1752_175269

theorem Elise_savings :
  let initial_dollars := 8
  let saved_euros := 11
  let euro_to_dollar := 1.18
  let comic_cost := 2
  let puzzle_pounds := 13
  let pound_to_dollar := 1.38
  let euros_to_dollars := saved_euros * euro_to_dollar
  let total_after_saving := initial_dollars + euros_to_dollars
  let after_comic := total_after_saving - comic_cost
  let pounds_to_dollars := puzzle_pounds * pound_to_dollar
  let final_amount := after_comic - pounds_to_dollars
  final_amount = 1.04 :=
by
  sorry

end Elise_savings_l1752_175269


namespace train_speed_l1752_175284

-- Defining the lengths and time
def length_train : ℕ := 100
def length_bridge : ℕ := 300
def time_crossing : ℕ := 15

-- Defining the total distance
def total_distance : ℕ := length_train + length_bridge

-- Proving the speed of the train
theorem train_speed : (total_distance / time_crossing : ℚ) = 26.67 := by
  sorry

end train_speed_l1752_175284


namespace sum_two_integers_l1752_175213

theorem sum_two_integers (a b : ℤ) (h1 : a = 17) (h2 : b = 19) : a + b = 36 := by
  sorry

end sum_two_integers_l1752_175213


namespace Cathy_total_money_l1752_175272

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end Cathy_total_money_l1752_175272


namespace fraction_addition_l1752_175282

theorem fraction_addition (x y : ℚ) (h : x / y = 2 / 3) : (x + y) / y = 5 / 3 := 
by 
  sorry

end fraction_addition_l1752_175282


namespace reduction_for_same_profit_cannot_reach_460_profit_l1752_175227

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l1752_175227


namespace Morse_code_distinct_symbols_l1752_175201

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l1752_175201


namespace population_is_24000_l1752_175285

theorem population_is_24000 (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 := sorry

end population_is_24000_l1752_175285


namespace parent_combinations_for_O_l1752_175299

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB

open BloodType

-- Define the conditions given in the problem
def parent_not_AB (p : BloodType) : Prop :=
  p ≠ AB

def possible_parent_types : List BloodType :=
  [A, B, O]

-- The math proof problem
theorem parent_combinations_for_O :
  ∀ (mother father : BloodType),
    parent_not_AB mother →
    parent_not_AB father →
    mother ∈ possible_parent_types →
    father ∈ possible_parent_types →
    (possible_parent_types.length * possible_parent_types.length) = 9 := 
by
  intro mother father h1 h2 h3 h4
  sorry

end parent_combinations_for_O_l1752_175299


namespace proof_problem_l1752_175225

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x| > 1}

def B : Set ℝ := {x | (0 : ℝ) < x ∧ x ≤ 2}

def complement_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def intersection (s1 s2 : Set ℝ) : Set ℝ := s1 ∩ s2

theorem proof_problem : (complement_A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end proof_problem_l1752_175225


namespace percentage_reduction_of_faculty_l1752_175296

noncomputable def percentage_reduction (original reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem percentage_reduction_of_faculty :
  percentage_reduction 226.74 195 = 13.99 :=
by sorry

end percentage_reduction_of_faculty_l1752_175296


namespace largest_of_seven_consecutive_numbers_l1752_175292

theorem largest_of_seven_consecutive_numbers (avg : ℕ) (h : avg = 20) :
  ∃ n : ℕ, n + 6 = 23 := 
by
  sorry

end largest_of_seven_consecutive_numbers_l1752_175292


namespace general_solution_of_differential_equation_l1752_175266

theorem general_solution_of_differential_equation (a₀ : ℝ) (x : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x = (y x)^2) ∧ y x = a₀ / (1 - a₀ * x) :=
sorry

end general_solution_of_differential_equation_l1752_175266
