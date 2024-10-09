import Mathlib

namespace percentage_of_water_in_mixture_is_17_14_l1466_146653

def Liquid_A_water_percentage : ℝ := 0.10
def Liquid_B_water_percentage : ℝ := 0.15
def Liquid_C_water_percentage : ℝ := 0.25
def Liquid_D_water_percentage : ℝ := 0.35

def parts_A : ℝ := 3
def parts_B : ℝ := 2
def parts_C : ℝ := 1
def parts_D : ℝ := 1

def part_unit : ℝ := 100

noncomputable def total_units : ℝ := 
  parts_A * part_unit + parts_B * part_unit + parts_C * part_unit + parts_D * part_unit

noncomputable def total_water_units : ℝ :=
  parts_A * part_unit * Liquid_A_water_percentage +
  parts_B * part_unit * Liquid_B_water_percentage +
  parts_C * part_unit * Liquid_C_water_percentage +
  parts_D * part_unit * Liquid_D_water_percentage

noncomputable def percentage_water : ℝ := (total_water_units / total_units) * 100

theorem percentage_of_water_in_mixture_is_17_14 :
  percentage_water = 17.14 := sorry

end percentage_of_water_in_mixture_is_17_14_l1466_146653


namespace sqrt_nested_eq_x_pow_eleven_eighths_l1466_146640

theorem sqrt_nested_eq_x_pow_eleven_eighths (x : ℝ) (hx : 0 ≤ x) : 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (11 / 8) :=
  sorry

end sqrt_nested_eq_x_pow_eleven_eighths_l1466_146640


namespace total_employees_l1466_146634

variable (E : ℕ)
variable (employees_prefer_X employees_prefer_Y number_of_prefers : ℕ)
variable (X_percentage Y_percentage : ℝ)

-- Conditions based on the problem
axiom prefer_X : X_percentage = 0.60
axiom prefer_Y : Y_percentage = 0.40
axiom max_preference_relocation : number_of_prefers = 140

-- Defining the total number of employees who prefer city X or Y and get relocated accordingly:
axiom equation : X_percentage * E + Y_percentage * E = number_of_prefers

-- The theorem we are proving
theorem total_employees : E = 140 :=
by
  -- Proof placeholder
  sorry

end total_employees_l1466_146634


namespace greatest_difference_l1466_146676

theorem greatest_difference (n m : ℕ) (hn : 1023 = 17 * n + m) (hn_pos : 0 < n) (hm_pos : 0 < m) : n - m = 57 :=
sorry

end greatest_difference_l1466_146676


namespace remainder_of_sum_mod_eight_l1466_146614

theorem remainder_of_sum_mod_eight (m : ℤ) : 
  ((10 - 3 * m) + (5 * m + 6)) % 8 = (2 * m) % 8 :=
by
  sorry

end remainder_of_sum_mod_eight_l1466_146614


namespace non_adjacent_divisibility_l1466_146675

theorem non_adjacent_divisibility (a : Fin 7 → ℕ) (h : ∀ i, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) :
  ∃ i j : Fin 7, i ≠ j ∧ (¬(i + 1)%7 = j) ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end non_adjacent_divisibility_l1466_146675


namespace rate_of_fuel_consumption_l1466_146692

-- Define the necessary conditions
def total_fuel : ℝ := 100
def total_hours : ℝ := 175

-- Prove the rate of fuel consumption per hour
theorem rate_of_fuel_consumption : (total_fuel / total_hours) = 100 / 175 := 
by 
  sorry

end rate_of_fuel_consumption_l1466_146692


namespace problem_f_g_comp_sum_l1466_146688

-- Define the functions
def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

-- Define the statement we want to prove
theorem problem_f_g_comp_sum (x : ℚ) (h : x = 2) : f (g x) + g (f x) = 36 / 5 := by
  sorry

end problem_f_g_comp_sum_l1466_146688


namespace prove_f_x1_minus_f_x2_lt_zero_l1466_146616

variable {f : ℝ → ℝ}

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Specify that f is decreasing for x < 0
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < 0 → y < 0 → x < y → f x > f y

theorem prove_f_x1_minus_f_x2_lt_zero (hx1x2 : |x1| < |x2|)
  (h_even : even_function f)
  (h_decreasing : decreasing_on_negative f) :
  f x1 - f x2 < 0 :=
sorry

end prove_f_x1_minus_f_x2_lt_zero_l1466_146616


namespace ratio_of_segments_l1466_146697

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end ratio_of_segments_l1466_146697


namespace find_x_l1466_146698

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l1466_146698


namespace find_x_for_equation_l1466_146656

def f (x : ℝ) : ℝ := 2 * x - 3

theorem find_x_for_equation : (2 * f x - 21 = f (x - 4)) ↔ (x = 8) :=
by
  sorry

end find_x_for_equation_l1466_146656


namespace total_ticket_cost_is_14_l1466_146673

-- Definitions of the ticket costs
def ticket_cost_hat : ℕ := 2
def ticket_cost_stuffed_animal : ℕ := 10
def ticket_cost_yoyo : ℕ := 2

-- Definition of the total ticket cost
def total_ticket_cost : ℕ := ticket_cost_hat + ticket_cost_stuffed_animal + ticket_cost_yoyo

-- Theorem stating the total ticket cost is 14
theorem total_ticket_cost_is_14 : total_ticket_cost = 14 := by
  -- Proof would go here, but sorry is used to skip it
  sorry

end total_ticket_cost_is_14_l1466_146673


namespace combination_recurrence_l1466_146659

variable {n r : ℕ}
variable (C : ℕ → ℕ → ℕ)

theorem combination_recurrence (hn : n > 0) (hr : r > 0) (h : n > r)
  (h2 : ∀ (k : ℕ), k = 1 → C 2 1 = C 1 1 + C 1) 
  (h3 : ∀ (k : ℕ), k = 1 → C 3 1 = C 2 1 + C 2) 
  (h4 : ∀ (k : ℕ), k = 2 → C 3 2 = C 2 2 + C 2 1)
  (h5 : ∀ (k : ℕ), k = 1 → C 4 1 = C 3 1 + C 3) 
  (h6 : ∀ (k : ℕ), k = 2 → C 4 2 = C 3 2 + C 3 1)
  (h7 : ∀ (k : ℕ), k = 3 → C 4 3 = C 3 3 + C 3 2)
  (h8 : ∀ n r : ℕ, (n > r) → C n r = C (n-1) r + C (n-1) (r-1)) :
  C n r = C (n-1) r + C (n-1) (r-1) :=
sorry

end combination_recurrence_l1466_146659


namespace correct_equations_l1466_146624

variable (x y : ℝ)

theorem correct_equations :
  (18 * x = y + 3) ∧ (17 * x = y - 4) ↔ (18 * x = y + 3) ∧ (17 * x = y - 4) :=
by
  sorry

end correct_equations_l1466_146624


namespace find_roots_l1466_146680

theorem find_roots (x : ℝ) :
  5 * x^4 - 28 * x^3 + 46 * x^2 - 28 * x + 5 = 0 → x = 3.2 ∨ x = 0.8 ∨ x = 1 :=
by
  intro h
  sorry

end find_roots_l1466_146680


namespace annual_return_percentage_l1466_146628

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end annual_return_percentage_l1466_146628


namespace initial_men_employed_l1466_146681

theorem initial_men_employed (M : ℕ) 
  (h1 : ∀ m d, m * d = 2 * 10)
  (h2 : ∀ m t, (m + 30) * t = 10 * 30) : 
  M = 75 :=
by
  sorry

end initial_men_employed_l1466_146681


namespace area_of_quadrilateral_is_correct_l1466_146623

noncomputable def area_of_quadrilateral_BGFAC : ℝ :=
  let a := 3 -- side of the equilateral triangle
  let triangle_area := (a^2 * Real.sqrt 3) / 4 -- area of ABC
  let ratio_AG_GC := 2 -- ratio AG:GC = 2:1
  let area_AGC := triangle_area / 3 -- area of triangle AGC
  let area_BGC := triangle_area / 3 -- area of triangle BGC
  let area_BFC := (2 : ℝ) * triangle_area / 3 -- area of triangle BFC
  let area_BGFC := area_BGC + area_BFC -- area of quadrilateral BGFC
  area_BGFC

theorem area_of_quadrilateral_is_correct :
  area_of_quadrilateral_BGFAC = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof will be provided here
  sorry

end area_of_quadrilateral_is_correct_l1466_146623


namespace terminating_decimal_l1466_146631

-- Define the given fraction
def frac : ℚ := 21 / 160

-- Define the decimal representation
def dec : ℚ := 13125 / 100000

-- State the theorem to be proved
theorem terminating_decimal : frac = dec := by
  sorry

end terminating_decimal_l1466_146631


namespace cos_theta_sub_pi_div_3_value_l1466_146658

open Real

noncomputable def problem_statement (θ : ℝ) : Prop :=
  sin (3 * π - θ) = (sqrt 5 / 2) * sin (π / 2 + θ)

theorem cos_theta_sub_pi_div_3_value (θ : ℝ) (hθ : problem_statement θ) :
  cos (θ - π / 3) = 1 / 3 + sqrt 15 / 6 ∨ cos (θ - π / 3) = - (1 / 3 + sqrt 15 / 6) :=
sorry

end cos_theta_sub_pi_div_3_value_l1466_146658


namespace integer_solution_unique_l1466_146677

theorem integer_solution_unique (x y z : ℤ) : x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end integer_solution_unique_l1466_146677


namespace earnings_difference_l1466_146693

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end earnings_difference_l1466_146693


namespace ratio_of_numbers_l1466_146633

theorem ratio_of_numbers (a b : ℕ) (h1 : a.gcd b = 5) (h2 : a.lcm b = 60) (h3 : a = 3 * 5) (h4 : b = 4 * 5) : (a / a.gcd b) / (b / a.gcd b) = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l1466_146633


namespace sales_in_fourth_month_l1466_146645

theorem sales_in_fourth_month (sale_m1 sale_m2 sale_m3 sale_m5 sale_m6 avg_sales total_months : ℕ)
    (H1 : sale_m1 = 7435) (H2 : sale_m2 = 7927) (H3 : sale_m3 = 7855) 
    (H4 : sale_m5 = 7562) (H5 : sale_m6 = 5991) (H6 : avg_sales = 7500) (H7 : total_months = 6) :
    ∃ sale_m4 : ℕ, sale_m4 = 8230 := by
  sorry

end sales_in_fourth_month_l1466_146645


namespace evaluate_expression_l1466_146649

theorem evaluate_expression : (2^2010 * 3^2012 * 25) / 6^2011 = 37.5 := by
  sorry

end evaluate_expression_l1466_146649


namespace rationalize_and_subtract_l1466_146602

theorem rationalize_and_subtract :
  (7 / (3 + Real.sqrt 15)) * (3 - Real.sqrt 15) / (3^2 - (Real.sqrt 15)^2) 
  - (1 / 2) = -4 + (7 * Real.sqrt 15) / 6 :=
by
  sorry

end rationalize_and_subtract_l1466_146602


namespace triangle_acute_l1466_146647

theorem triangle_acute
  (A B C : ℝ)
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  -- proof goes here
  sorry

end triangle_acute_l1466_146647


namespace money_left_is_correct_l1466_146636

-- Define initial amount of money Dan has
def initial_amount : ℕ := 3

-- Define the cost of the candy bar
def candy_cost : ℕ := 1

-- Define the money left after the purchase
def money_left : ℕ := initial_amount - candy_cost

-- The theorem stating that the money left is 2
theorem money_left_is_correct : money_left = 2 := by
  sorry

end money_left_is_correct_l1466_146636


namespace seven_in_M_l1466_146654

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the set M complement with respect to U
def compl_U_M : Set ℕ := {1, 3, 5}

-- Define the set M
def M : Set ℕ := U \ compl_U_M

-- Prove that 7 is an element of M
theorem seven_in_M : 7 ∈ M :=
by {
  sorry
}

end seven_in_M_l1466_146654


namespace minimal_surface_area_l1466_146663

-- Definitions based on the conditions in the problem.
def unit_cube (a b c : ℕ) : Prop := a * b * c = 25
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

-- The proof problem statement.
theorem minimal_surface_area : ∃ (a b c : ℕ), unit_cube a b c ∧ surface_area a b c = 54 := 
sorry

end minimal_surface_area_l1466_146663


namespace value_large_cube_l1466_146641

-- Definitions based on conditions
def volume_small := 1 -- volume of one-inch cube in cubic inches
def volume_large := 64 -- volume of four-inch cube in cubic inches
def value_small : ℝ := 1000 -- value of one-inch cube of gold in dollars
def proportion (x y : ℝ) : Prop := y = 64 * x -- proportionality condition

-- Prove that the value of the four-inch cube of gold is $64000
theorem value_large_cube : proportion value_small 64000 := by
  -- Proof skipped
  sorry

end value_large_cube_l1466_146641


namespace completing_the_square_l1466_146687

theorem completing_the_square (x : ℝ) (h : x^2 - 6 * x + 7 = 0) : (x - 3)^2 - 2 = 0 := 
by sorry

end completing_the_square_l1466_146687


namespace min_value_expr_l1466_146684

theorem min_value_expr (x y z w : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) (hw : -2 < w ∧ w < 2) :
  2 ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w / 2)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w / 2))) :=
sorry

end min_value_expr_l1466_146684


namespace quadratic_range_and_value_l1466_146678

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l1466_146678


namespace arithmetic_square_root_of_9_is_3_l1466_146632

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l1466_146632


namespace cup_of_coffee_price_l1466_146637

def price_cheesecake : ℝ := 10
def price_set : ℝ := 12
def discount : ℝ := 0.75

theorem cup_of_coffee_price (C : ℝ) (h : price_set = discount * (C + price_cheesecake)) : C = 6 :=
by
  sorry

end cup_of_coffee_price_l1466_146637


namespace train_passing_through_tunnel_l1466_146668

theorem train_passing_through_tunnel :
  let train_length : ℝ := 300
  let tunnel_length : ℝ := 1200
  let speed_in_kmh : ℝ := 54
  let speed_in_mps : ℝ := speed_in_kmh * (1000 / 3600)
  let total_distance : ℝ := train_length + tunnel_length
  let time : ℝ := total_distance / speed_in_mps
  time = 100 :=
by
  sorry

end train_passing_through_tunnel_l1466_146668


namespace saved_percent_l1466_146618

-- Definitions for conditions:
def last_year_saved (S : ℝ) : ℝ := 0.10 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_saved (S : ℝ) : ℝ := 0.06 * (1.10 * S)

-- Given conditions and proof goal:
theorem saved_percent (S : ℝ) (hl_last_year_saved : last_year_saved S = 0.10 * S)
  (hl_this_year_salary : this_year_salary S = 1.10 * S)
  (hl_this_year_saved : this_year_saved S = 0.066 * S) :
  (this_year_saved S / last_year_saved S) * 100 = 66 :=
by
  sorry

end saved_percent_l1466_146618


namespace triangle_has_angle_45_l1466_146696

theorem triangle_has_angle_45
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : B + C = 3 * A) :
  A = 45 :=
by
  sorry

end triangle_has_angle_45_l1466_146696


namespace crayons_at_the_end_of_thursday_l1466_146674

-- Definitions for each day's changes
def monday_crayons : ℕ := 7
def tuesday_crayons (initial : ℕ) := initial + 3
def wednesday_crayons (initial : ℕ) := initial - 5 + 4
def thursday_crayons (initial : ℕ) := initial + 6 - 2

-- Proof statement to show the number of crayons at the end of Thursday
theorem crayons_at_the_end_of_thursday : thursday_crayons (wednesday_crayons (tuesday_crayons monday_crayons)) = 13 :=
by
  sorry

end crayons_at_the_end_of_thursday_l1466_146674


namespace sector_arc_length_120_degrees_radius_3_l1466_146665

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem sector_arc_length_120_degrees_radius_3 :
  arc_length 120 3 = 2 * Real.pi :=
by
  sorry

end sector_arc_length_120_degrees_radius_3_l1466_146665


namespace find_triples_l1466_146660

theorem find_triples (a b c : ℝ) 
  (h1 : a = (b + c) ^ 2) 
  (h2 : b = (a + c) ^ 2) 
  (h3 : c = (a + b) ^ 2) : 
  (a = 0 ∧ b = 0 ∧ c = 0) 
  ∨ 
  (a = 1/4 ∧ b = 1/4 ∧ c = 1/4) :=
  sorry

end find_triples_l1466_146660


namespace riding_time_fraction_l1466_146652

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end riding_time_fraction_l1466_146652


namespace sum_of_two_numbers_l1466_146661

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 9) (h2 : (1 / x) = 4 * (1 / y)) : x + y = 15 / 2 :=
  sorry

end sum_of_two_numbers_l1466_146661


namespace min_value_x_y_l1466_146621

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 :=
by 
  sorry

end min_value_x_y_l1466_146621


namespace evaluate_composite_function_l1466_146627

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_composite_function :
  f (g (-2)) = 26 := by
  sorry

end evaluate_composite_function_l1466_146627


namespace number_added_is_10_l1466_146642

-- Define the conditions.
def number_thought_of : ℕ := 55
def result : ℕ := 21

-- Define the statement of the problem.
theorem number_added_is_10 : ∃ (y : ℕ), (number_thought_of / 5 + y = result) ∧ (y = 10) := by
  sorry

end number_added_is_10_l1466_146642


namespace how_many_peaches_l1466_146609

-- Define the variables
variables (Jake Steven : ℕ)

-- Conditions
def has_fewer_peaches : Prop := Jake = Steven - 7
def jake_has_9_peaches : Prop := Jake = 9

-- The theorem that proves Steven's number of peaches
theorem how_many_peaches (Jake Steven : ℕ) (h1 : has_fewer_peaches Jake Steven) (h2 : jake_has_9_peaches Jake) : Steven = 16 :=
by
  -- Proof goes here
  sorry

end how_many_peaches_l1466_146609


namespace M_positive_l1466_146610

theorem M_positive (x y : ℝ) : (3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13) > 0 :=
by
  sorry

end M_positive_l1466_146610


namespace problem1_problem2_l1466_146689

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - a*x - b

-- Problem 1: Prove that a + b = 3
theorem problem1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : ∀ x, f x a b ≤ 3) : a + b = 3 := 
sorry

-- Problem 2: Prove that 1/2 < a < 3
theorem problem2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 3) 
  (h₃ : ∀ x, x ≥ a → g x a b < f x a b) : 1/2 < a ∧ a < 3 := 
sorry

end problem1_problem2_l1466_146689


namespace four_digit_numbers_gt_3000_l1466_146695

theorem four_digit_numbers_gt_3000 (d1 d2 d3 d4 : ℕ) (h_digits : (d1, d2, d3, d4) = (2, 0, 5, 5)) (h_distinct_4digit : (d1 * 1000 + d2 * 100 + d3 * 10 + d4) > 3000) :
  ∃ count, count = 3 := sorry

end four_digit_numbers_gt_3000_l1466_146695


namespace tangent_parallel_l1466_146608

noncomputable def f (x : ℝ) : ℝ := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P.1 = 1) (hP_cond : P.2 = f P.1) 
  (tangent_parallel : ∀ x, deriv f x = 3) : P = (1, 0) := 
by 
  have h_deriv : deriv f 1 = 4 * 1^3 - 1 := by sorry
  have slope_eq : deriv f (P.1) = 3 := by sorry
  have solve_a : P.1 = 1 := by sorry
  have solve_b : f 1 = 0 := by sorry
  exact sorry

end tangent_parallel_l1466_146608


namespace initial_percentage_of_milk_l1466_146613

theorem initial_percentage_of_milk 
  (initial_solution_volume : ℝ)
  (extra_water_volume : ℝ)
  (desired_percentage : ℝ)
  (new_total_volume : ℝ)
  (initial_percentage : ℝ) :
  initial_solution_volume = 60 →
  extra_water_volume = 33.33333333333333 →
  desired_percentage = 54 →
  new_total_volume = initial_solution_volume + extra_water_volume →
  (initial_percentage / 100 * initial_solution_volume = desired_percentage / 100 * new_total_volume) →
  initial_percentage = 84 := 
by 
  intros initial_volume_eq extra_water_eq desired_perc_eq new_volume_eq equation
  -- proof steps here
  sorry

end initial_percentage_of_milk_l1466_146613


namespace arithmetic_sequence_l1466_146671

theorem arithmetic_sequence (S : ℕ → ℕ) (h : ∀ n, S n = 3 * n * n) :
  (∃ a d : ℕ, ∀ n : ℕ, S n - S (n - 1) = a + (n - 1) * d) ∧
  (∀ n, S n - S (n - 1) = 6 * n - 3) :=
by
  sorry

end arithmetic_sequence_l1466_146671


namespace find_number_l1466_146619

theorem find_number 
  (x : ℝ)
  (h : (1 / 10) * x - (1 / 1000) * x = 700) :
  x = 700000 / 99 :=
by 
  sorry

end find_number_l1466_146619


namespace parity_of_solutions_l1466_146630

theorem parity_of_solutions
  (n m x y : ℤ)
  (hn : Odd n) 
  (hm : Odd m) 
  (h1 : x + 2 * y = n) 
  (h2 : 3 * x - y = m) :
  Odd x ∧ Even y :=
by
  sorry

end parity_of_solutions_l1466_146630


namespace transformed_eq_l1466_146629

theorem transformed_eq (a b c : ℤ) (h : a > 0) :
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 → (a * x + b)^2 = c) →
  a + b + c = 64 :=
by
  sorry

end transformed_eq_l1466_146629


namespace find_floors_l1466_146644

theorem find_floors
  (a b : ℕ)
  (alexie_bathrooms_per_floor : ℕ := 3)
  (alexie_bedrooms_per_floor : ℕ := 2)
  (baptiste_bathrooms_per_floor : ℕ := 4)
  (baptiste_bedrooms_per_floor : ℕ := 3)
  (total_bathrooms : ℕ := 25)
  (total_bedrooms : ℕ := 18)
  (h1 : alexie_bathrooms_per_floor * a + baptiste_bathrooms_per_floor * b = total_bathrooms)
  (h2 : alexie_bedrooms_per_floor * a + baptiste_bedrooms_per_floor * b = total_bedrooms) :
  a = 3 ∧ b = 4 :=
by
  sorry

end find_floors_l1466_146644


namespace partiallyFilledBoxes_l1466_146662

/-- Define the number of cards Joe collected -/
def numPokemonCards : Nat := 65
def numMagicCards : Nat := 55
def numYuGiOhCards : Nat := 40

/-- Define the number of cards each full box can hold -/
def pokemonBoxCapacity : Nat := 8
def magicBoxCapacity : Nat := 10
def yuGiOhBoxCapacity : Nat := 12

/-- Define the partially filled boxes for each type -/
def pokemonPartialBox : Nat := numPokemonCards % pokemonBoxCapacity
def magicPartialBox : Nat := numMagicCards % magicBoxCapacity
def yuGiOhPartialBox : Nat := numYuGiOhCards % yuGiOhBoxCapacity

/-- Theorem to prove number of cards in each partially filled box -/
theorem partiallyFilledBoxes :
  pokemonPartialBox = 1 ∧
  magicPartialBox = 5 ∧
  yuGiOhPartialBox = 4 :=
by
  -- proof goes here
  sorry

end partiallyFilledBoxes_l1466_146662


namespace water_tank_capacity_l1466_146667

theorem water_tank_capacity (C : ℝ) :
  (0.40 * C - 0.25 * C = 36) → C = 240 :=
  sorry

end water_tank_capacity_l1466_146667


namespace total_pictures_on_wall_l1466_146611

theorem total_pictures_on_wall (oil_paintings watercolor_paintings : ℕ) (h1 : oil_paintings = 9) (h2 : watercolor_paintings = 7) :
  oil_paintings + watercolor_paintings = 16 := 
by
  sorry

end total_pictures_on_wall_l1466_146611


namespace intersection_property_l1466_146669

theorem intersection_property (x_0 : ℝ) (h1 : x_0 > 0) (h2 : -x_0 = Real.tan x_0) :
  (x_0^2 + 1) * (Real.cos (2 * x_0) + 1) = 2 :=
sorry

end intersection_property_l1466_146669


namespace valid_numbers_count_l1466_146625

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end valid_numbers_count_l1466_146625


namespace compute_ab_l1466_146694

namespace MathProof

variable {a b : ℝ}

theorem compute_ab (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := 
by
  sorry

end MathProof

end compute_ab_l1466_146694


namespace multiples_of_3_ending_number_l1466_146670

theorem multiples_of_3_ending_number :
  ∃ n, ∃ k, k = 93 ∧ (∀ m, 81 + 3 * m = n → 0 ≤ m ∧ m < k) ∧ n = 357 := 
by
  sorry

end multiples_of_3_ending_number_l1466_146670


namespace abc_not_less_than_two_l1466_146686

theorem abc_not_less_than_two (a b c : ℝ) (h : a + b + c = 6) : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

end abc_not_less_than_two_l1466_146686


namespace ratio_of_inscribed_squares_l1466_146604

open Real

-- Condition: A square inscribed in a right triangle with sides 3, 4, and 5
def inscribedSquareInRightTriangle1 (x : ℝ) (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5 ∧ x = 12 / 7

-- Condition: A square inscribed in a different right triangle with sides 5, 12, and 13
def inscribedSquareInRightTriangle2 (y : ℝ) (d e f : ℝ) : Prop :=
  d = 5 ∧ e = 12 ∧ f = 13 ∧ y = 169 / 37

-- The ratio x / y is 444 / 1183
theorem ratio_of_inscribed_squares (x y : ℝ) (a b c d e f : ℝ) :
  inscribedSquareInRightTriangle1 x a b c →
  inscribedSquareInRightTriangle2 y d e f →
  x / y = 444 / 1183 :=
by
  intros h1 h2
  sorry

end ratio_of_inscribed_squares_l1466_146604


namespace perfect_square_expression_l1466_146672

theorem perfect_square_expression (p : ℝ) (h : p = 0.28) : 
  (12.86 * 12.86 + 12.86 * p + 0.14 * 0.14) = (12.86 + 0.14) * (12.86 + 0.14) :=
by 
  -- proof goes here
  sorry

end perfect_square_expression_l1466_146672


namespace solve_consecutive_integers_solve_consecutive_even_integers_l1466_146635

-- Conditions: x, y, z, w are positive integers and x + y + z + w = 46.
def consecutive_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) ∧ (x + y + z + w = 46)

def consecutive_even_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 2 = y) ∧ (y + 2 = z) ∧ (z + 2 = w) ∧ (x + y + z + w = 46)

-- Proof that consecutive integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_integers : ∃ x y z w : ℕ, consecutive_integers_solution x y z w :=
sorry

-- Proof that consecutive even integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_even_integers : ∃ x y z w : ℕ, consecutive_even_integers_solution x y z w :=
sorry

end solve_consecutive_integers_solve_consecutive_even_integers_l1466_146635


namespace faster_ship_speed_l1466_146646

theorem faster_ship_speed :
  ∀ (x y : ℕ),
    (200 + 100 = 300) → -- Total distance covered for both directions
    (x + y) * 10 = 300 → -- Opposite direction equation
    (x - y) * 25 = 300 → -- Same direction equation
    x = 21 := 
by
  intros x y _ eq1 eq2
  sorry

end faster_ship_speed_l1466_146646


namespace factorize_problem1_factorize_problem2_l1466_146603

theorem factorize_problem1 (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

theorem factorize_problem2 (x y : ℝ) : 
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) :=
by sorry

end factorize_problem1_factorize_problem2_l1466_146603


namespace average_height_corrected_l1466_146605

theorem average_height_corrected (students : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ)
  (h1 : students = 20)
  (h2 : incorrect_avg_height = 175)
  (h3 : incorrect_height = 151)
  (h4 : actual_height = 111) :
  (incorrect_avg_height * students - incorrect_height + actual_height) / students = 173 :=
by
  sorry

end average_height_corrected_l1466_146605


namespace black_squares_in_20th_row_l1466_146617

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def squares_in_row (n : ℕ) : ℕ := 1 + sum_natural (n - 2)

noncomputable def black_squares_in_row (n : ℕ) : ℕ := 
  if squares_in_row n % 2 = 1 then (squares_in_row n - 1) / 2 else squares_in_row n / 2

theorem black_squares_in_20th_row : black_squares_in_row 20 = 85 := 
by
  sorry

end black_squares_in_20th_row_l1466_146617


namespace number_of_solutions_l1466_146666

theorem number_of_solutions :
  ∃ (solutions : Finset (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ solutions ↔ (x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1)) ∧ 
  solutions.card = 2 :=
by
  sorry

end number_of_solutions_l1466_146666


namespace expression_increase_fraction_l1466_146612

theorem expression_increase_fraction (x y : ℝ) :
  let x' := 1.4 * x
  let y' := 1.4 * y
  let original := x * y^2
  let increased := x' * y'^2
  increased - original = (1744/1000) * original := by
sorry

end expression_increase_fraction_l1466_146612


namespace problem1_l1466_146657

theorem problem1 (k : ℝ) : (∃ x : ℝ, k*x^2 + (2*k + 1)*x + (k - 1) = 0) → k ≥ -1/8 := 
sorry

end problem1_l1466_146657


namespace rahul_share_l1466_146699

theorem rahul_share :
  let total_payment := 370
  let bonus := 30
  let remaining_payment := total_payment - bonus
  let rahul_work_per_day := 1 / 3
  let rajesh_work_per_day := 1 / 2
  let ramesh_work_per_day := 1 / 4
  
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day + ramesh_work_per_day
  let rahul_share_of_work := rahul_work_per_day / total_work_per_day
  let rahul_payment := rahul_share_of_work * remaining_payment

  rahul_payment = 80 :=
by {
  sorry
}

end rahul_share_l1466_146699


namespace part1_part2_l1466_146615

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

theorem part2 (a x : ℝ) (h : a ≠ -3) :
  (f x a > 4 * a - (a + 3) * x) ↔ 
  ((a > -3 ∧ (x < -3 ∨ x > a)) ∨ (a < -3 ∧ (x < a ∨ x > -3))) :=
by
  sorry

end part1_part2_l1466_146615


namespace track_and_field_unit_incorrect_l1466_146626

theorem track_and_field_unit_incorrect :
  ∀ (L : ℝ), L = 200 → "mm" ≠ "m" → false :=
by
  intros L hL hUnit
  sorry

end track_and_field_unit_incorrect_l1466_146626


namespace solution_set_of_inequality_l1466_146601

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (x^2 - 1 < 0) :=
by
  sorry

end solution_set_of_inequality_l1466_146601


namespace alice_min_speed_l1466_146638

open Real

theorem alice_min_speed (d : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  d = 180 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 4 → d / alice_time > (d / bob_speed) - alice_delay →
  d / alice_time > 45 := by
  sorry


end alice_min_speed_l1466_146638


namespace find_x_l1466_146655

theorem find_x (x : ℕ) (h1 : x ≥ 10) (h2 : x > 8) : x = 9 := by
  sorry

end find_x_l1466_146655


namespace number_of_carbon_atoms_l1466_146620

-- Definitions and Conditions
def hydrogen_atoms : ℕ := 6
def molecular_weight : ℕ := 78
def hydrogen_atomic_weight : ℕ := 1
def carbon_atomic_weight : ℕ := 12

-- Theorem Statement: Number of Carbon Atoms
theorem number_of_carbon_atoms 
  (H_atoms : ℕ := hydrogen_atoms)
  (M_weight : ℕ := molecular_weight)
  (H_weight : ℕ := hydrogen_atomic_weight)
  (C_weight : ℕ := carbon_atomic_weight) : 
  (M_weight - H_atoms * H_weight) / C_weight = 6 :=
sorry

end number_of_carbon_atoms_l1466_146620


namespace probability_of_picking_letter_in_mathematics_l1466_146683

def unique_letters_in_mathematics : List Char := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']

def number_of_unique_letters_in_word : ℕ := unique_letters_in_mathematics.length

def total_letters_in_alphabet : ℕ := 26

theorem probability_of_picking_letter_in_mathematics :
  (number_of_unique_letters_in_word : ℚ) / total_letters_in_alphabet = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l1466_146683


namespace odometer_trip_l1466_146690

variables (d e f : ℕ) (x : ℕ)

-- Define the conditions
def start_odometer (d e f : ℕ) : ℕ := 100 * d + 10 * e + f
def end_odometer (d e f : ℕ) : ℕ := 100 * f + 10 * e + d
def distance_travelled (x : ℕ) : ℕ := 65 * x
def valid_trip (d e f x : ℕ) : Prop := 
  d ≥ 1 ∧ d + e + f ≤ 9 ∧ 
  end_odometer d e f - start_odometer d e f = distance_travelled x

-- The final statement to prove
theorem odometer_trip (h : valid_trip d e f x) : d^2 + e^2 + f^2 = 41 := 
sorry

end odometer_trip_l1466_146690


namespace water_parts_in_solution_l1466_146691

def lemonade_syrup_parts : ℝ := 7
def target_percentage : ℝ := 0.30
def adjusted_parts : ℝ := 2.1428571428571423

-- Original equation: L = 0.30 * (L + W)
-- Substitute L = 7 for the particular instance.
-- Therefore, 7 = 0.30 * (7 + W)

theorem water_parts_in_solution (W : ℝ) : 
  (7 = 0.30 * (7 + W)) → 
  W = 16.333333333333332 := 
by
  sorry

end water_parts_in_solution_l1466_146691


namespace find_area_triangle_boc_l1466_146607

noncomputable def area_ΔBOC := 21

theorem find_area_triangle_boc (A B C K O : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup K] [NormedAddCommGroup O]
  (AC : ℝ) (AB : ℝ) (h1 : AC = 14) (h2 : AB = 6)
  (circle_centered_on_AC : Prop)
  (K_on_BC : Prop)
  (angle_BAK_eq_angle_ACB : Prop)
  (midpoint_O_AC : Prop)
  (angle_AKC_eq_90 : Prop)
  (area_ABC : Prop) : 
  area_ΔBOC = 21 := 
sorry

end find_area_triangle_boc_l1466_146607


namespace find_y_l1466_146643

theorem find_y (y : ℝ) (hy_pos : y > 0) (hy_prop : y^2 / 100 = 9) : y = 30 := by
  sorry

end find_y_l1466_146643


namespace determine_a_l1466_146650

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.exp x - a)^2

theorem determine_a (a x₀ : ℝ)
  (h₀ : f x₀ a ≤ 1/2) : a = 1/2 :=
sorry

end determine_a_l1466_146650


namespace hyunwoo_cookies_l1466_146648

theorem hyunwoo_cookies (packs_initial : Nat) (pieces_per_pack : Nat) (packs_given_away : Nat)
  (h1 : packs_initial = 226) (h2 : pieces_per_pack = 3) (h3 : packs_given_away = 3) :
  (packs_initial - packs_given_away) * pieces_per_pack = 669 := 
by
  sorry

end hyunwoo_cookies_l1466_146648


namespace freeRangingChickens_l1466_146639

-- Define the number of chickens in the coop
def chickensInCoop : Nat := 14

-- Define the number of chickens in the run
def chickensInRun : Nat := 2 * chickensInCoop

-- Define the number of chickens free ranging
def chickensFreeRanging : Nat := 2 * chickensInRun - 4

-- State the theorem
theorem freeRangingChickens : chickensFreeRanging = 52 := by
  -- We cannot provide the proof, so we use sorry
  sorry

end freeRangingChickens_l1466_146639


namespace find_f_of_4_l1466_146600

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_of_4 :
  (∃ α : ℝ, power_function 3 α = Real.sqrt 3) →
  power_function 4 (1/2) = 2 :=
by
  sorry

end find_f_of_4_l1466_146600


namespace probability_A_not_losing_l1466_146679

theorem probability_A_not_losing (P_draw : ℚ) (P_win_A : ℚ) (h1 : P_draw = 1/2) (h2 : P_win_A = 1/3) : 
  P_draw + P_win_A = 5/6 :=
by
  rw [h1, h2]
  norm_num

end probability_A_not_losing_l1466_146679


namespace option_C_is_correct_l1466_146664

-- Define the conditions as propositions
def condition_A := |-2| = 2
def condition_B := (-1)^2 = 1
def condition_C := -7 + 3 = -4
def condition_D := 6 / (-2) = -3

-- The statement that option C is correct
theorem option_C_is_correct : condition_C := by
  sorry

end option_C_is_correct_l1466_146664


namespace train_pass_platform_time_l1466_146682

theorem train_pass_platform_time :
  ∀ (length_train length_platform speed_time_cross_tree speed_train pass_time : ℕ), 
  length_train = 1200 →
  length_platform = 300 →
  speed_time_cross_tree = 120 →
  speed_train = length_train / speed_time_cross_tree →
  pass_time = (length_train + length_platform) / speed_train →
  pass_time = 150 :=
by
  intros
  sorry

end train_pass_platform_time_l1466_146682


namespace min_empty_squares_eq_nine_l1466_146685

-- Definition of the problem conditions
def chessboard_size : ℕ := 9
def total_squares : ℕ := chessboard_size * chessboard_size
def number_of_white_squares : ℕ := 4 * chessboard_size
def number_of_black_squares : ℕ := 5 * chessboard_size
def minimum_number_of_empty_squares : ℕ := number_of_black_squares - number_of_white_squares

-- Theorem to prove minimum number of empty squares
theorem min_empty_squares_eq_nine :
  minimum_number_of_empty_squares = 9 :=
by
  -- Placeholder for the proof
  sorry

end min_empty_squares_eq_nine_l1466_146685


namespace system_solutions_are_equivalent_l1466_146606

theorem system_solutions_are_equivalent :
  ∀ (a b x y : ℝ),
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) ∧
  (a = 8.3 ∧ b = 1.2) ∧
  (x + 2 = a ∧ y - 1 = b) →
  x = 6.3 ∧ y = 2.2 :=
by
  -- Sorry is added intentionally to skip the proof
  sorry

end system_solutions_are_equivalent_l1466_146606


namespace factor_expression_l1466_146622

variable (a b : ℤ)

theorem factor_expression : 2 * a^2 * b - 4 * a * b^2 + 2 * b^3 = 2 * b * (a - b)^2 := 
sorry

end factor_expression_l1466_146622


namespace jane_reading_period_l1466_146651

theorem jane_reading_period (total_pages pages_per_day : ℕ) (H1 : pages_per_day = 5 + 10) (H2 : total_pages = 105) : 
  total_pages / pages_per_day = 7 :=
by
  sorry

end jane_reading_period_l1466_146651
